"""
Extract the CUDA C source TinyGrad *generates* for an op, with no GPU present.

Why this file exists
--------------------
We need TinyGrad's generated CUDA C for a matmul, on a machine with no CUDA
driver, no nvcc and no GPU. TinyGrad's `Tensor.realize()` path needs a real
device, so we stub out just enough of the device to run the *codegen* pipeline
(`Tensor.schedule_linear` -> `lower_and_compile` -> `to_program`) and intercept
the source string at the point where it would have been handed to nvcc.

This is the real TinyGrad codegen path, unmodified: we do not patch any
rewrite pass, scheduler heuristic, or renderer. The only substitutions are:

  1. A `Compiled` device whose only job is to hold `arch` + the renderer list.
     `Compiled.__init__` is pure Python (tinygrad/device.py:352).
  2. `NVCCRenderer`, which is the renderer tinygrad itself selects when
     `DEV='CUDA:NVCC'`. Its `NVCCCompiler.__init__` is also pure Python
     (tinygrad/runtime/support/compiler_cuda.py); only `compile()` shells out.
  3. `compile()` is overridden to capture the source and return a sentinel.
     Everything before it -- instruction selection, LINEARIZE, line rendering,
     the full `Schedule`/`get_rewrite` pipeline -- is stock tinygrad.

Consequence for the experiment: the emitted kernel is a real TinyGrad kernel,
but it is being compiled by *our* nvcc invocation later, not nvrtc. For the
ptxas/SASS analysis in RESULTS.md we compile the captured .cu with
`ptxas -arch=<arch>` ourselves, so the renderer choice only matters in that
CUDARenderer (nvrtc) and NVCCRenderer (nvcc) differ in a few codegen flags
(`--minimal`, and the nvcc path allows `#include`/templating). We use NVCC
because it is the path that can express `__launch_bounds__`-tuned code and is
what the CUTLASS side also compiles with -- i.e. it is the fair comparison.
"""

from __future__ import annotations
import os
from dataclasses import dataclass

# tinygrad reads DEV at import time in several places; set it before any
# tinygrad import so the target/renderer selection is deterministic.
os.environ.setdefault("DEV", "CUDA:NVCC")
os.environ.setdefault("ALLOW_DEVICE_USAGE", "1")

from tinygrad.helpers import Target  # noqa: E402
from tinygrad.device import Compiled, Device  # noqa: E402
from tinygrad.renderer.cstyle import NVCCRenderer  # noqa: E402

CAPTURED: list[dict] = []


class _CapturingNVCC:
  """Stands in for NVCCCompiler. Records the source, returns a sentinel cubin.

  tinygrad's Compiler base class memoises on the source hash, so each distinct
  source is compiled exactly once -- which is what we want, since one matmul
  graph may contain several distinct kernels.
  """

  def __init__(self, arch: str):
    self.arch, self.name = arch, f"capture_nvcc_{arch}"
    self.compile_options: list[str] = [f"--gpu-architecture={arch}"]
    self.ptx = False
    self._seen: set[str] = set()

  def compile(self, src: str, **kw) -> bytes:  # noqa: ARG002
    if src not in self._seen:
      self._seen.add(src)
      CAPTURED.append({"arch": self.arch, "src": src, "options": list(self.compile_options)})
    return b"\x00FAKE_CUBIN\x00"

  def compile_cached(self, src: str, **kw) -> bytes: return self.compile(src, **kw)

  def disassemble(self, lib: bytes) -> None:  # noqa: ARG002
    return None


class CaptureRenderer(NVCCRenderer):
  """NVCCRenderer, but the compiler never runs; it just records the source."""

  def __init__(self, target: Target):
    super().__init__(target)
    self.compiler = _CapturingNVCC(target.arch)


class _NoAlloc:
  """Allocator stub. We never allocate: schedule_linear is pure symbolication."""

  def alloc(self, size: int, *a, **k): raise RuntimeError("capture-only device cannot allocate")
  def free(self, *a, **k): return None
  def alloc_aligned(self, size: int, *a, **k): raise RuntimeError("capture-only device cannot allocate")


class CaptureCUDADevice(Compiled):
  """A CUDA device that can render source for `arch` but cannot run anything."""

  def __init__(self, arch: str = "sm_80", device: str = "CUDA:0"):
    super().__init__(device, _NoAlloc(), [CaptureRenderer], runtime=None, graph=None, arch=arch)
    self.captured: list[dict] = CAPTURED

  def _select_renderer(self) -> CaptureRenderer:
    # Bypass the DEV target-string machinery; there is exactly one renderer.
    if (r := self.cached_renderer.get(self.arch)) is None:
      self.cached_renderer[self.arch] = r = CaptureRenderer(Target(device="CUDA", renderer="NVCC", arch=self.arch))
    return r

  def synchronize(self) -> None: return None
  def _at_profile_finalize(self) -> None: return None
  def count(self) -> int: return 1


@dataclass
class Kernel:
  name: str
  arch: str
  src: str
  options: list[str]


_ACTIVE: list[CaptureCUDADevice] = []


def install(arch: str = "sm_80") -> CaptureCUDADevice:
  """Register a capture-only CUDA device as `Device['CUDA:0']` and return it."""
  dev = CaptureCUDADevice(arch=arch)
  _ACTIVE.clear()
  _ACTIVE.append(dev)
  # device.py instantiates devices through the functools.cache-wrapped
  # `_Device__get_canonicalized_item`. functools' C lru_cache wrapper exposes no
  # way to seed its dict, so shadow it with a plain function on the instance
  # (instance __dict__ wins over the non-data descriptor). This is the only
  # place we touch Device; every codegen path below is stock tinygrad.
  Device._Device__get_canonicalized_item = lambda ix: _ACTIVE[0]  # type: ignore[method-assign]
  Device._opened_devices.discard("CUDA:0")
  return dev


def generate(arch: str, fn, *, tag: str = "", extra_env: dict | None = None) -> list[Kernel]:
  """Run `fn` against a capture-only CUDA device at `arch`; return captured kernels.

  `fn` receives nothing; it should build Tensors on device "CUDA:0" and call
  `Tensor.schedule_linear()` (NOT `realize()` -- see below).

  Gotcha, stated explicitly because it burned us: `Tensor.realize()` filters
  out any uop that is already realised or has buffer identity
  (tinygrad/tensor.py:422). Calling realize() twice, or realizing a graph whose
  buffers are already realised, is a *silent no-op* -- no kernel is ever
  emitted and you get an empty result with no error. `schedule_linear()`
  unconditionally builds the linear program, so it always produces source.
  """
  from tinygrad.codegen import to_program_cache
  from tinygrad.engine.realize import lower_and_compile

  dev = install(arch)
  CAPTURED.clear()
  dev.captured = CAPTURED
  to_program_cache.clear()

  if extra_env:
    from tinygrad.helpers import Context
    with Context(**extra_env):
      linear = fn()
      lower_and_compile(linear)
  else:
    linear = fn()
    lower_and_compile(linear)

  out: list[Kernel] = []
  for i, c in enumerate(CAPTURED):
    name = _first_kernel_name(c["src"]) or f"kernel{i}"
    out.append(Kernel(name=f"{name}{tag}", arch=arch, src=c["src"], options=c["options"]))
  return out


def _first_kernel_name(src: str) -> str | None:
  # `extern "C" __global__ void __launch_bounds__(128) real_name(args) {`
  for line in src.splitlines():
    ls = line.strip()
    if "__global__" in ls and "void" in ls and "(" in ls:
      tail = ls.split("void", 1)[1]
      # skip attribute macros like __launch_bounds__(...) which are followed by
      # the real function name
      for tok in tail.replace("(", " ").replace(")", " ").split():
        if tok and not tok.startswith("__") and not tok.isdigit() and tok not in ("extern", "C"):
          return tok
  return None
