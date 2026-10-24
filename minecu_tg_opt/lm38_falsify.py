"""lm38_falsify.py -- the instrument for `notes/38`.

WHAT QUESTION THIS SERVES
-------------------------
`notes/35` section 5 says, on the strength of a five-point correlation:

    Graph size is upstream of the cost, but the channel is register allocation
    of the fused straight-line region -- not the launch budget, and not the
    instruction count.

and section 6 says the same thing from the other side of the abstraction:
crossing the framework's boundary is free until it spills, then 2.4x.

Both rest on one quantity ordering five points at Pearson +0.9986. Five points
is a correlation, and a quantity that is monotone in graph size orders five
points just as well as a causal one. The case that separates the two readings
is **same graph size, different frame**: the causal reading says the times
differ, the nuisance reading says they are equal.

This module builds that case. It is an instrument and not a port: nothing here
is a candidate answer to "how fast can this be ported", and no number produced
under an intervention is offered as one.

THE LEVER, AND WHY IT IS NOT THE OBVIOUS ONE
---------------------------------------------
The obvious lever is ptxas's own `-maxrregcount`, on the argument that it
changes the *compilation* and not the *program*, so the generated CUDA is a
byte-identical string and the UOp count, the fusion decision, the launch count
and the instruction stream the compiler sees are all unchanged by construction.

**That lever does not work on this workload, and the reason is specific.**
Offline, on the PTX this project's own pipeline captured from the framework's
fused kernel, nineteen `-maxrregcount` settings produced ONE cubin: 255
registers, 1 312 bytes of stack frame, 481 960 bytes, byte-identical every
time. The reason is in the PTX:

    .maxntid 128, 1, 1

`CUDARenderer.kernel_typedef` emits `__launch_bounds__(128)` on every CUDA
kernel (`tinygrad/renderer/cstyle.py:411`), nvrtc turns that into `.maxntid`,
and **ptxas resolves the register ceiling from `.maxntid` and ignores the
command line.** Confirmed on the real artifact: deleting the one `.maxntid`
line restores a fully live lever (ten settings, ten cubins, registers
255 -> 24, frame 1 312 -> 2 992 bytes), and setting `.maxntid 1024` pins the
allocation at 64 registers = 65 536 / 1024, again ignoring the flag.

So the lever this note uses is:

    the framework's own generated CUDA, byte for byte  ->  nvrtc (stock)  ->
    drop the `.maxntid` line from the PTX  ->  ptxas -maxrregcount=N

`.maxntid` is an *upper bound* ptxas places on itself; removing it and putting
an explicit budget in its place does not change the program's arithmetic, does
not change the launch geometry (the kernel is still launched with 128 threads,
which was always inside the 1 024 that ptxas assumes when unconstrained), and
does not change one byte of the source the framework emitted. What moves is
the register allocation, which is the thing under test.

The sweep is ANCHORED: with the flag absent and `.maxntid` dropped, ptxas
reproduces the stock allocation exactly (255 registers, 1 312/1 772/1 296
bytes). So the top of the ladder is the real thing, not a stranger's code.

THE TWO CURRENCIES, AND WHY BOTH ARE RECORDED
---------------------------------------------
ptxas's `-v` prints, per function:

    N bytes stack frame, M bytes spill stores, K bytes spill loads

Those are three numbers and the study has been using one of them under the
wrong name. `1772 + 1296 = 3068`, and 3 068 is the figure `notes/35` section 5
calls "ptxas's local-memory frame" for `flat_gather`; the actual stack frame is
1 312 bytes. They are physically different things:

  * **stack frame** -- a capacity, bytes of local memory reserved per thread,
    whether or not it is touched. A weak proxy for cost.
  * **spill stores + spill loads** -- the traffic ptxas scheduled, i.e. the
    extra memory operations inside the fused region. This is the quantity that
    has a mechanism behind it.

The two rank the five rungs differently, so every cell here records both, and
so does the driver. Whichever one orders the wall clock is the currency the
mechanism is actually about, and the note can say which.

WHAT IS GUARDED, AND HOW
------------------------
* The generated CUDA's sha256 is recorded per cell. "Same graph size" is then a
  fact about a string, not an assumption about a program.
* `to_program_key` is `(ast.key, type(renderer), renderer.target, *config)`
  -- `codegen/__init__.py:496-497` -- and **the compiler is not in it.** Install
  a different compiler, rebuild the identical graph, and you get the previous
  binary back with no error and no warning. The cache is emptied before every
  cell and the compiler's own call count is asserted afterwards, so a cell
  whose compilation did not happen cannot report a frame.
* `NOLOCALS` is the framework's own register lever and *is* in the program key
  (`codegen/__init__.py:493`), so it does not need the cache cleared for the
  key to be right.
* Every timed cell is gated byte-for-byte against the hand-written driver's own
  output on the hand-written driver's own state, on the card, **before** any
  clock is read, and the timing uses `minecu_tg.bench.linear_device_time_ms`,
  which re-executes the compiled CALLs and so cannot be fooled by
  `Tensor.realize()` being a no-op on an already-realized graph.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import re
import subprocess
import tempfile
import time

LAUNCH_BOUNDS_RE = re.compile(
    r'__global__\s+void\s+__launch_bounds__\((\d+)\)\s+(\w+)')
MAXNTID_RE = re.compile(r"^\.maxntid\s+(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*$", re.M)


# ---------------------------------------------------------------------------
# the compiler
# ---------------------------------------------------------------------------
class PtxBudgetCompiler:
    """The stock nvrtc call, then ptxas with an explicit register budget.

    `base` is the device's own compiler object, so the PTX is produced by the
    framework's stock invocation with the framework's stock options. The only
    difference from the stock path is that ptxas runs here instead of inside
    the driver's JIT -- which is what makes the frame reportable at all, and
    what makes the cubin that gets loaded the same cubin `ptxas -v` just
    described.

    `drop_maxntid` is the finding: with the directive present ptxas ignores
    `-maxrregcount`, so the lever is off unless the directive goes.
    """

    def __init__(self, base_compile, arch: str, maxrregcount: int | None = None,
                 drop_maxntid: bool = True, tag: str = ""):
        self.base_compile = base_compile
        self.arch = arch or device_arch()
        self.maxrregcount = maxrregcount
        self.drop_maxntid = drop_maxntid
        self.tag = tag or f"b{maxrregcount if maxrregcount else 'stock'}"
        self.ptx_cache: dict = {}      # src sha -> PTX (nvrtc runs once)
        self.sources: dict = {}        # src sha -> the CUDA itself
        self.facts: dict = {}          # src sha -> what ptxas said
        self.compiled: set = set()     # DISTINCT sources assembled
        self.errors: list = []
        self.invocations = 0
        self.nvrtc_calls = 0

    @property
    def calls(self) -> int:
        """Distinct sources this compiler assembled.

        A property rather than a counter, because the bank hooks TWO methods and
        one compile can legitimately reach both. Counting raw invocations makes
        "did the intervention happen" depend on which hook fired; counting
        distinct sources answers the question that is actually asked.
        """
        return len(self.compiled)

    def compile(self, src: str, ptx_of=None) -> bytes:
        self.invocations += 1
        ptx_of = ptx_of or self.base_compile
        sha = hashlib.sha256(src.encode()).hexdigest()
        if sha in self.compiled:      # already assembled in this cell
            done = self.facts.get(sha, {})
            if "cubin" in done:
                return done["cubin"]
        self.sources[sha] = src
        ptx = self._ptx(src, ptx_of).rstrip(b"\x00")
        text = ptx.decode("utf-8", "replace")
        mn = MAXNTID_RE.search(text)
        if self.drop_maxntid and mn:
            text = MAXNTID_RE.sub("", text, count=1)
        elif mn and int(mn.group(1)) != 0:
            # kept, for the control that shows the lever is off
            pass
        ptx2 = text.encode()

        cmd = ["ptxas", f"-arch={self.arch}", "-v", "-O3"]
        if self.maxrregcount is not None:
            cmd.append(f"-maxrregcount={int(self.maxrregcount)}")
        rec = {"sha256": sha, "ptxas_cmd": " ".join(cmd),
               "maxrregcount": self.maxrregcount,
               "drop_maxntid": bool(self.drop_maxntid),
               "maxntid_in": mn.group(0) if mn else None,
               "maxntid_out": MAXNTID_RE.search(text).group(0)
               if MAXNTID_RE.search(text) else None,
               "ptx_bytes": len(ptx), "ptx_bytes_after": len(ptx2)}
        with tempfile.TemporaryDirectory() as td:
            pf, cf = os.path.join(td, "k.ptx"), os.path.join(td, "k.cubin")
            open(pf, "wb").write(ptx2)
            cmd2 = cmd + ["-o", cf, pf]
            p = subprocess.run(cmd2, capture_output=True, text=True, timeout=2400)
            txt = (p.stderr or "") + (p.stdout or "")
            rec["rc"] = p.returncode
            rec["ptxas_v"] = txt[-4000:]
            rec.update(parse_ptxas_v(txt))
            if p.returncode != 0 or not os.path.exists(cf):
                self.errors.append(f"{sha[:12]} ptxas rc={p.returncode} {txt[-300:]}")
                raise RuntimeError(f"ptxas failed: {txt[-300:]}")
            cub = open(cf, "rb").read()
            rec["cubin_sha256"] = hashlib.sha256(cub).hexdigest()
            rec["cubin_bytes"] = len(cub)
            rec["cubin"] = cub
            rec.update(_sass_census(cub))
        self.facts[sha] = rec
        self.compiled.add(sha)
        return cub

    def _ptx(self, src: str, ptx_of) -> bytes:
        sha = hashlib.sha256(src.encode()).hexdigest()
        if sha not in self.ptx_cache:            # nvrtc is the expensive half
            self.nvrtc_calls += 1
            out = ptx_of(src)
            # ASSERT PTX, do not assume it. `NVRTCCompiler` is built with
            # `ptx = iface.startswith("MOCK") or dev == "CUDA"`
            # (`renderer/cstyle.py:406`), so the stock compiler hands back PTX on
            # a real CUDA device and a CUBIN on `NULL`. Feeding a cubin to ptxas
            # fails with "Unexpected non-ASCII character encountered on line 1",
            # which names neither the cause nor the fix -- and the silent
            # version of this bug, a frame read out of a binary ptxas never
            # assembled, is the entire class of defect this note is about.
            if b".version" not in out[:1024] and not out[:16].lstrip().startswith(b"//"):
                raise TypeError(
                    f"the stock compiler returned a {len(out):,}-byte blob that is "
                    f"not PTX (starts {out[:16]!r}). On dev != 'CUDA' tinygrad "
                    f"builds NVRTCCompiler with ptx=False and it returns a cubin. "
                    f"This instrument needs PTX, because ptxas is what is being "
                    f"varied.")
            self.ptx_cache[sha] = out
        return self.ptx_cache[sha]


FUNC_RE = re.compile(r"^\s*(?:\.text\.|//## File )?\s*Function : (\S+)", re.M)
FUNC_RE2 = re.compile(r"Function : (\S+)")
INSTR_RE = re.compile(r"/\*[0-9a-f]{4}\*/")
#: cuobjdump and nvdisasm live in different CUDA installs, and the offline
#: archlab images carry neither. A missing disassembler must make the SASS
#: count ABSENT and say so -- never 0, because a 0 that means "no
#: disassembler" is a 0 that reads as "no instructions".
_SASS_TOOL: str | None = None


def _sass_tool() -> str | None:
    global _SASS_TOOL
    if _SASS_TOOL is None:
        from shutil import which
        _SASS_TOOL = which("cuobjdump") or which("nvdisasm") or ""
    return _SASS_TOOL or None


def _sass_census(cubin: bytes) -> dict:
    """Instruction count of THIS cubin, by whichever disassembler exists.

    The frame and the instruction count have to come from the same artifact as
    the timing, or the note is comparing a register allocation against an
    instruction count of some other compilation. That is why this runs on the
    cubin that is about to be loaded and not on a re-compile.
    """
    tool = _sass_tool()
    if not tool:
        return {"sass_instructions": None, "sass_tool": None,
                "sass_note": "no cuobjdump and no nvdisasm on this image; the "
                             "SASS count is ABSENT, not zero"}
    argv = ([tool, "-sass", "-"] if tool.endswith("cuobjdump")
            else [tool, "-c", "-"])
    d = subprocess.run(argv, input=cubin, capture_output=True, timeout=900)
    sass = d.stdout.decode("utf-8", "replace")
    if not sass.strip():
        return {"sass_instructions": None, "sass_tool": tool,
                "sass_note": f"{tool} produced no output (rc={d.returncode})"}
    by_fn = {k: len(INSTR_RE.findall(b)) for k, b in _split_functions(sass).items()}
    return {"sass_instructions": sum(by_fn.values()), "sass_by_function": by_fn,
            "sass_tool": tool, "sass_note": None}


def _split_functions(sass: str) -> dict[str, str]:
    out, cur = {}, None
    for line in sass.splitlines():
        m = FUNC_RE.search(line)
        if m:
            cur = m.group(1)
            out.setdefault(cur, [])
            continue
        if cur is not None:
            out[cur].append(line)
    return {k: "\n".join(v) for k, v in out.items()}


PTXV_FRAME = re.compile(
    r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
PTXV_USED = re.compile(r"Used (\d+) registers")


def parse_ptxas_v(text: str) -> dict:
    """ptxas `-v` -> the three byte counts, the register count, and the name.

    `notes/35` section 5 calls the sum of spill stores and spill loads "the
    local-memory frame". The stack frame is a different number and is reported
    as `stack_frame` here so the two can never be confused again.
    """
    out: dict = {}
    fn = None
    for line in text.splitlines():
        m = re.search(r"Compiling entry function '([^']+)'", line)
        if m:
            fn = m.group(1)
            out[fn] = {}
            continue
        if fn is None:
            continue
        m = PTXV_FRAME.search(line)
        if m:
            sf, ss, sl = int(m.group(1)), int(m.group(2)), int(m.group(3))
            out[fn].update(stack_frame=sf, spill_stores=ss, spill_loads=sl,
                           spill_total=ss + sl)
            continue
        m = PTXV_USED.search(line)
        if m:
            out[fn]["registers"] = int(m.group(1))
    # collapse to the single hottest function, which is the fused one
    if out:
        hot = max(out, key=lambda k: out[k].get("stack_frame", -1))
        out["_hot"] = hot
        out.update({k: v for k, v in out[hot].items()})
    return out


# ---------------------------------------------------------------------------
# installing and removing the compiler
# ---------------------------------------------------------------------------
class CompilerBank:
    """Put a register budget in front of EVERY compiler, then take it off.

    Two implementations were tried before this one and both failed SILENTLY,
    which is the reason for the third.

      1. `dev.renderer.compiler = c`.  `Compiled.renderer` is a property
         (`tinygrad/device.py:358-359`) that rebuilds the renderer through
         `select_first_inited(..., self.cached_renderer, t)`, and that cache key
         includes the resolved `Target`. Assign to the instance you happened to
         capture and a differently-keyed renderer carries its own stock
         compiler instead. Every cell then reports an allocation of `None`,
         raises nothing, and reads as a run in which the lever did nothing.
      2. Patching `CUDARenderer.__init__`.  Catches newly built renderers, but
         not a compiler reached any other way.
      3. **This one**: patch `Compiler.compile_cached` on the class itself.
         `do_compile` (`tinygrad/codegen/__init__.py:451`) reaches the binary
         by exactly one call, `ctx.compiler.compile_cached(source.arg)`, and
         there is no second route. Whichever compiler object the framework ends
         up holding, the interception happens.

    `assert_live` is the other half. Every timed cell calls it, so a cell whose
    compiler was never invoked cannot report a frame -- which is what turned
    failure 1 from a silent fabrication into a loud error on the card.
    """

    def __init__(self, dev):
        from tinygrad.device import Compiler
        self.dev = dev
        # Captured as UNBOUND functions on the class, and bound at call time.
        # Binding them here would bind them to whatever object happened to
        # exist when the bank was built, and `_resolve` then hands that
        # one-argument function a source and gets
        # "Compiler.compile() missing 1 required positional argument: 'src'".
        self.orig_compile_cached = Compiler.compile_cached
        self.orig_compile = Compiler.compile
        self.saved: list = []
        self.installed = None
        self._patched = False
        # The DEEP hook, recorded now and patched on first install. On a real
        # T4 the route probe showed `compile_cached` being called for the small
        # kernels and NOT for the ~491 000-byte fused one, so a hook on the
        # single method `do_compile` names is not sufficient on its own.
        # `NVRTCCompiler.compile` is where nvrtc is actually invoked, and a hook
        # there catches every route into it.
        try:
            from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
            self.nv = NVRTCCompiler
            self.orig_nv_compile = NVRTCCompiler.compile
        except Exception:
            self.nv = None
            self.orig_nv_compile = None

    def install(self, maxrregcount=None, drop_maxntid=True, tag=None):
        from tinygrad.device import Compiler
        c = PtxBudgetCompiler(self.orig_compile, self.dev.arch, maxrregcount,
                              drop_maxntid, tag or "")
        self.saved.append(c)
        if not self._patched:
            if getattr(Compiler.compile_cached, "_lm38_bank", False):
                # A bank is already installed (nested). Record the UNPATCHED
                # method so `restore()` puts back the real thing rather than
                # another bank's wrapper, and leave the outer one in place --
                # it routes to whichever bank is active, so the two compose.
                self.orig_compile_cached = getattr(
                    Compiler.compile_cached, "_lm38_prev", Compiler.compile_cached)
            _ic = _make_interceptor(self)
            _ic._lm38_prev = self.orig_compile_cached
            Compiler.compile_cached = _ic
            # The DEEP `NVRTCCompiler.compile` hook was tried here and
            # REMOVED. It looks like the robust choice and is not: the hook
            # receives whichever compiler the framework called, and on a
            # backend that builds its compiler with `ptx=False` that is a
            # cubin producer, so the hook re-enters `act.compile` with a cubin
            # where PTX is required. The card's own route probe settled the
            # question that hook was added to answer -- `compile_cached` IS the
            # path on a real T4, with `NVRTCCompiler` as the class -- so the
            # single hook here is correct and the deep one is not.
            self._patched = True
        self.installed = c
        return c

    def assert_live(self, comp, where: str = "") -> None:
        if getattr(comp, "calls", 0) == 0:
            raise RuntimeError(
                f"{where}: the installed compiler was never called "
                f"(calls={getattr(comp, 'calls', None)}, "
                f"sources={len(getattr(comp, 'sources', {}))}). The intervention "
                f"did not happen, so any allocation this cell reports would be a "
                f"fabrication.")

    def restore(self):
        from tinygrad.device import Compiler
        if self.saved:
            self.saved.pop()
        if not self.saved:
            self.installed = None
            if self._patched:
                Compiler.compile_cached = self.orig_compile_cached
                self._patched = False

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.restore()
        return False


def _make_interceptor(bank: "CompilerBank"):
    """A `Compiler.compile_cached` that routes through `bank.installed`.

    Two things about this function are load-bearing and neither is obvious.

    **The ORIGINAL `compile` is called to get the PTX, never the original
    `compile_cached`.** `compile_cached` is keyed on `compile_cuda_sm_75` plus
    the source string (`device.py:307-310`), and the source string is
    byte-identical across every budget in this experiment, so leaving the disk
    cache in would serve the first budget's binary to all the others. No error,
    no warning, every cell reporting the same frame.

    **The PTX comes from the object's OWN class's `compile`, not from
    `Compiler.compile`.** `Compiler.compile` is the base class's identity
    function -- `def compile(self, src): return src.encode()`
    (`device.py:298`) -- shadowed by `NVRTCCompiler.compile`
    (`runtime/support/compiler_cuda.py:56`). Calling the base one returns the
    *source* rather than a compiled artifact, and the instrument then reports
    "the stock compiler returned a blob that is not PTX" while pointing at a
    perfectly correct compiler. `_resolve` walks the MRO for the first
    `compile` that is not the base identity, so the framework's own override is
    the one that runs.
    """
    from tinygrad.device import Compiler as _C

    def _resolve(obj):
        """The `compile` that turns this object into PTX.

        Chosen by what the object DECLARES it returns, not by which method
        shadows which. `NVRTCCompiler.__init__` sets `self.ptx`
        (`runtime/support/compiler_cuda.py:48`), and that flag is the whole
        difference between handing back text and handing back a cubin. On a
        real CUDA device tinygrad builds it with `ptx=True` and the subclass's
        `compile` calls `nvrtcGetPTX`; the class-level `NVRTCCompiler.compile`
        hook is therefore correct on the card and would be WRONG on `NULL`,
        where the same hook reaches a compiler built with `ptx=False` and
        `nvrtcGetCUBIN` -- which is the ELF blob that arrives here. Selecting on
        `obj.ptx` makes the hook correct on both, and the dry run is what proved
        it had to be done this way.
        """
        if getattr(obj, "ptx", None) is True:
            for klass in type(obj).__mro__:
                fn = klass.__dict__.get("compile")
                if fn is None:
                    continue
                fn = getattr(fn, "__func__", fn)
                if fn is _C.compile:
                    continue
                # Skip THIS MODULE's own hooks. The bank replaces
                # `NVRTCCompiler.compile` with a wrapper, so an MRO walk that
                # does not exclude it finds the wrapper, and calling the
                # wrapper from inside the wrapper is either infinite recursion
                # or, because it re-enters `act.compile` with the cubin path,
                # a cubin arriving where PTX was required. The previous
                # version of this function had that bug and the dry run found
                # it as a 2 600-byte ELF blob where PTX should have been.
                if getattr(fn, "_lm38_bank", False):
                    continue
                return fn.__get__(obj, type(obj))
            return bank.orig_compile.__get__(obj, type(obj))
        # this compiler is not in PTX mode; hand the source straight back and
        # let `_ptx`'s assert explain why, which is more useful than a
        # wrong-format blob surfacing later
        return lambda s: s.encode()

    def _interceptor(self, src: str) -> bytes:
        act = bank.installed
        if act is None:
            return bank.orig_compile_cached(self, src)
        return act.compile(src, ptx_of=_resolve(self))
    _interceptor._lm38_bank = True
    return _interceptor


def clear_program_cache() -> int:
    """Empty TinyGrad's program cache, and report how big it was.

    `to_program_key` is `(ast.key, type(renderer), renderer.target,
    *to_program_config)` -- `codegen/__init__.py:496-497`. The compiler is not
    in it. So once a source has been compiled by one compiler, every later
    request for that source returns the FIRST compiler's binary and no
    compilation happens at all. Install a register budget, rebuild the
    identical graph, and you silently get the un-capped binary. That is the
    same class of defect as `Tensor.realize()` on a realized graph, and it is
    why every cell asserts the compiler's own call count afterwards.
    """
    from tinygrad.codegen import to_program_cache
    n = len(to_program_cache)
    to_program_cache.clear()
    return n


class Nolocals:
    """TinyGrad's own `NOLOCALS`, as a context manager.

    `NOLOCALS` -> `Opt(OptOps.NOLOCALS)` -> `dont_use_locals` ->
    `__launch_bounds__(prod(local_dims.vmax))`, so the knob is the framework's
    own handle on the register budget. It is in the program key
    (`codegen/__init__.py:493`), so unlike a compiler swap it does not need
    the cache cleared. Whether it moves the *frame* is a question, not an
    assumption: `.maxntid 1` gives ptxas a ceiling of 65 536 registers, which
    is far above the hardware maximum of 255, so on this fused kernel it
    cannot lower the allocation. The card decides, and the frame is reported.
    """

    def __init__(self, on: bool = True):
        self.on, self._old = on, None

    def __enter__(self):
        from tinygrad.helpers import NOLOCALS
        self._old = NOLOCALS.value
        NOLOCALS.value = 1 if self.on else 0
        return self

    def __exit__(self, *a):
        from tinygrad.helpers import NOLOCALS
        NOLOCALS.value = self._old
        return False


# ---------------------------------------------------------------------------
# reading the allocation back out of the binary the driver loaded
# ---------------------------------------------------------------------------
def func_facts(program) -> dict:
    """`NUM_REGS` and `LOCAL_SIZE_BYTES` from the loaded `CUfunction`.

    These are the driver talking about the module whose kernel is being timed,
    read a different way from `ptxas -v`. `LOCAL_SIZE_BYTES` is the stack frame
    in bytes, so this is the second, independent reading of that quantity. A
    disagreement between the two routes is a finding, so both are reported and
    neither is quietly reconciled.
    """
    from tinygrad.runtime.autogen import cuda
    out = {}
    for attr, key in ((4, "driver_regs"), (3, "driver_local_bytes")):
        v = ctypes.c_int(0)
        try:
            rc = cuda.cuFuncGetAttribute(ctypes.byref(v), attr, program.prg)
            out[key] = int(v.value) if rc == 0 else None
            out[key + "_rc"] = int(rc)
        except Exception as e:
            out[key] = None
            out[key + "_err"] = f"{type(e).__name__}: {e}"[:120]
    out["kernel_name"] = getattr(program, "name", None)
    return out


class ProgramRecorder:
    """Every `CUDAProgram` the runtime builds, with the allocation it reports.

    `CUDADevice.runtime_t` is how a `TinyELF` becomes a loaded `CUfunction`
    (`tinygrad/device.py:366`), so wrapping `CUDADevice.runtime` sees every
    program the session builds, and nothing is missed by patching the wrong
    layer.
    """

    def __init__(self):
        self.programs: list = []
        self._patched = False

    def __enter__(self):
        if self._patched:
            return self
        from tinygrad.runtime.ops_cuda import CUDADevice
        base, rec = CUDADevice.runtime, self

        def _runtime(self, obj):
            p = base(self, obj)
            rec.programs.append(p)
            return p
        CUDADevice.runtime = _runtime
        self._base, self._patched = base, True
        return self

    def __exit__(self, *a):
        if self._patched:
            from tinygrad.runtime.ops_cuda import CUDADevice
            CUDADevice.runtime = self._base
            self._patched = False
        return False

    def facts_for(self, names) -> list:
        return [func_facts(p) for p in self.programs
                if getattr(p, "name", None) in names]


# ---------------------------------------------------------------------------
# which route does this build of the framework actually compile through?
# ---------------------------------------------------------------------------
#: One entry per compile the framework performed, in order.
COMPILE_LOG: list = []


def install_route_probe() -> dict:
    """Record EVERY compile, and say which method it went through.

    Added after two sessions in which the register-budget intervention silently
    did nothing on a real T4 while working perfectly on `NULL` in the same
    harness. The dry run proved the code path; the card proved the path is not
    the same one. Rather than keep guessing, this logs the route.

    A trivial graph is enough to answer it and the answer costs about a second,
    which is the right price for finding out where compiles happen before
    spending twenty minutes of nvrtc on a run that would measure nothing.
    """
    from tinygrad.device import Compiler
    if getattr(Compiler.compile_cached, "_lm38_probe", False):
        return {}
    orig_cc, orig_c = Compiler.compile_cached, Compiler.compile

    def cc(self, src):
        COMPILE_LOG.append({"route": "compile_cached", "cls": type(self).__name__,
                            "cachekey": getattr(self, "cachekey", None),
                            "src_bytes": len(src)})
        return orig_cc(self, src)

    def c(self, src):
        COMPILE_LOG.append({"route": "compile", "cls": type(self).__name__,
                            "src_bytes": len(src)})
        return orig_c(self, src)

    cc._lm38_probe = True
    c._lm38_probe = True
    Compiler.compile_cached, Compiler.compile = cc, c

    def restore():
        Compiler.compile_cached, Compiler.compile = orig_cc, orig_c
    return {"restore": restore}


def route_census(device) -> list:
    """Realize a trivial graph and report every compile it caused."""
    import tinygrad
    from tinygrad import Tensor
    COMPILE_LOG.clear()
    x = Tensor([1.0, 2.0, 3.0, 4.0]).contiguous()
    (x * 2.0 + 1.0).realize()
    tinygrad.Device[device].synchronize()
    return list(COMPILE_LOG)


def device_arch() -> str:
    """`sm_XY` for the card we are actually on, from the card.

    `Device[...].arch` is `None` on some backends (it is on `NULL`), and a
    `-arch=None` fails with *"Value 'None' is not defined for option
    'gpu-name'"* -- which is how the first dry run found it. Reading the
    compute capability off the driver is also the version of this that cannot
    be wrong about which card it is on, which matters because a number
    attributed to a T4 that was produced on another card is not a T4 number.

    Falls back to `sm_75` only if the driver will not answer, and says so, so
    the fallback is visible rather than silent.
    """
    try:
        ctypes.CDLL("libcuda.so.1").cuInit(0)
        lib = ctypes.CDLL("libcuda.so.1")
        maj, mnr = ctypes.c_int(0), ctypes.c_int(0)
        if (lib.cuDeviceGetAttribute(ctypes.byref(maj), 75, 0) == 0 and
                lib.cuDeviceGetAttribute(ctypes.byref(mnr), 76, 0) == 0):
            return f"sm_{maj.value}{mnr.value}"
    except Exception:
        pass
    return "sm_75"


def source_facts(src: str) -> dict:
    """What can be read off the generated CUDA without a compiler."""
    lb = LAUNCH_BOUNDS_RE.findall(src)
    return {
        "source_bytes": len(src),
        "source_sha256": hashlib.sha256(src.encode()).hexdigest(),
        "source_statements": len(re.findall(r";\s*(?://.*)?$", src, re.M)),
        "launch_bounds": sorted({int(b) for b, _ in lb}),
        "kernel_names": sorted({n for _, n in lb}),
        "n_kernels": len(lb),
    }
