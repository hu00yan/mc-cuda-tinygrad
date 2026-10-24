"""
Generate the CUDA C that TinyGrad emits for matmul, for every arch/dtype/shape
in the study, into ./gen/.

Design notes that matter for interpreting the results:

1. `schedule_linear()`, never `realize()`. realize() filters out already-realised
   or buffer-identity uops (tinygrad/tensor.py:422) and is then a *silent no-op*:
   no kernel, no error. schedule_linear() unconditionally builds the linear
   program, so we always get source.

2. We call `schedule_linear(a, b, out)` with the input tensors as extra args so
   the emitted kernel takes A and B as real parameters (rather than folding them
   into the buffer list of a device-side-allocated temp). Without this, tinygrad
   must first materialise A and B on device.

3. The arch is supplied to a capture-only device, so the *same* Tensor graph is
   rendered once per target arch. This is exactly what the experiment needs: it
   isolates "what does the backend choose to emit for this arch" from "which GPU
   is plugged in". See tg_backend.py for why this is sound.

4. We record both the default schedule and the BEAM=1 schedule, because the
   study's central question is partly "could the compiler have found a better
   schedule?". If beam search finds a materially different kernel, the gap is a
   search/heuristic failure (CAN express, does not FIND). If beam search does
   not, the gap is a codegen-vocabulary failure (CANNOT express).
"""

from __future__ import annotations
import sys, pathlib, json

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from tg_backend import generate  # noqa: E402

GEN = HERE / "gen"
GEN.mkdir(exist_ok=True)

# NOTE on arch naming. TinyGrad builds its target arch as
#   arch=f"sm_{major}{minor}"  from cuDeviceComputeCapability   (ops_cuda.py:122)
# i.e. it can only ever produce "sm_90", "sm_100" -- never "sm_90a"/"sm_100a".
# Its arch parsers do `int(arch[3:])` (codegen/opt/tc.py:98,
# renderer/cstyle.py:471), which raises ValueError on the arch-conditional
# suffixed forms. So we ask TinyGrad to target sm_90 / sm_100 -- which is what
# it would genuinely see on an H100 / B200 -- and separately record that the
# 'a' targets are unrepresentable. We then *cross-compile* the resulting source
# with `ptxas -arch=sm_90a` / `-arch=sm_100a` in the SASS stage, because the
# research question is "given Hopper/Blackwell hardware, what SASS does this
# source produce?", and the 'a' targets are mandatory for wgmma/tcgen05.
ARCHS = ["sm_75", "sm_80", "sm_89", "sm_90", "sm_100"]

# arch-conditional targets, recorded for the "can the compiler express it?" table
ARCHS_UNREPRESENTABLE = ["sm_90a", "sm_100a"]

# (label, M, K, N)  -- C = A(M,K) @ B(K,N)
SHAPES = [
  ("big",    4096, 4096, 4096),
  ("small",  128,  128,  128),
]


def make_fn(m, k, n, in_dtype, acc_dtype):
  def f():
    from tinygrad import Tensor, dtypes
    dt = {"float32": dtypes.float32, "float16": dtypes.float16, "bfloat16": dtypes.bfloat16}[in_dtype]
    # Do NOT call .realize() here. realize() is a silent no-op on a graph whose
    # buffers are already realised (tinygrad/tensor.py:422 filters those out), and
    # on a capture-only device it would also try to allocate. Passing a and b as
    # extra schedule args is what forces them to become real kernel parameters.
    a = Tensor.empty(m, k, dtype=dt)
    b = Tensor.empty(k, n, dtype=dt)
    c = a @ b
    if acc_dtype is not None:
      c = c.cast(acc_dtype)
    return c.schedule_linear(a, b, c)
  return f


def slug(arch, shape, in_dtype, acc, mode):
  s = f"{arch}_{shape[0]}_{shape[1]}x{shape[2]}x{shape[3]}_{in_dtype}_to_{acc}_{mode}"
  return s.replace(".", "")


def main():
  index = []
  # 0. Probe the arch-conditional targets. We EXPECT these to fail; the failure
  #    mode is itself a deliverable (see RESULTS.md "cannot express").
  for arch in ARCHS_UNREPRESENTABLE:
    fn = make_fn(128, 128, 128, "float16", "float32")
    try:
      generate(arch, fn)
      status, err = "UNEXPECTEDLY_OK", None
    except Exception as e:
      status, err = "unrepresentable", f"{type(e).__name__}: {e}"
    print(f"[ARCH] {arch}: {status} {err or ''}")
    index.append({"file": None, "arch": arch, "shape": [128, 128, 128], "in_dtype": "float16",
                  "acc": "float32", "mode": "probe", "status": status, "error": err})

  jobs = []
  for arch in ARCHS:
    for shape in SHAPES:
      for in_dtype, acc in [("float32", "float32"), ("float16", "float32"), ("bfloat16", "float32"), ("float16", "float16")]:
        for mode, env in [("default", None), ("beam1", {"BEAM": 1})]:
          jobs.append((arch, shape, in_dtype, acc, mode, env))

  for arch, shape, in_dtype, acc, mode, env in jobs:
    name = slug(arch, shape, in_dtype, acc, mode)
    fn = make_fn(shape[1], shape[2], shape[3], in_dtype, acc)
    try:
      ks = generate(arch, fn, extra_env=env)
    except Exception as e:
      msg = f"{type(e).__name__}: {e}"
      (GEN / f"{name}.ERR").write_text(msg[:4000] + "\n")
      index.append({"file": f"{name}.ERR", "arch": arch, "shape": list(shape[1:]), "in_dtype": in_dtype,
                    "acc": acc, "mode": mode, "status": "error", "error": msg[:300]})
      print(f"[ERR ] {name}: {msg[:200]}")
      continue

    if not ks:
      index.append({"file": None, "arch": arch, "shape": list(shape[1:]), "in_dtype": in_dtype,
                    "acc": acc, "mode": mode, "status": "NO_SOURCE",
                    "note": "codegen produced no kernel"})
      print(f"[NONE] {name}")
      continue

    for i, k in enumerate(ks):
      fn_out = GEN / f"{name}__{i}.cu"
      fn_out.write_text(k.src)
      rec = {"file": fn_out.name, "arch": arch, "shape": list(shape[1:]), "in_dtype": in_dtype,
             "acc": acc, "mode": mode, "status": "ok", "kernel": k.name,
             "nvcc_options": k.options, "n_kernels": len(ks), "lines": k.src.count(chr(10)) + 1,
             "bytes": len(k.src)}
      index.append(rec)
      print(f"[ OK ] {fn_out.name:66s} {rec['bytes']:7d}B  {k.name}")

  (GEN / "INDEX.json").write_text(json.dumps(index, indent=2))
  print(f"\nwrote {len(index)} records to {GEN/'INDEX.json'}")


if __name__ == "__main__":
  main()
