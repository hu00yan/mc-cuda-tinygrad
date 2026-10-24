#!/usr/bin/env python3
"""probe38.py -- a two-minute script kernel that answers the four questions
that killed the previous two attempts, before the real notebook is built.

    Q1  can this account get a GPU session at all?  (the last two rounds never
        did, and both wrote scripts and no note)
    Q2  is the device a real GPU, and which one?  (`machine_shape: NvidiaL4`
        yields a CPU-only kernel with exit 0 and no warning -- ledger C-28)
    Q3  is `ptxas` on PATH, and does nvrtc -> ptxas -> cuModuleLoadData ->
        cuFuncGetAttribute work end to end?  That chain IS the lever: without
        it there is no way to hold the generated CUDA byte-identical and move
        only the register budget, which is the whole experiment.
    Q4  how long does one nvrtc compile and one ptxas take, so the real
        notebook's cell budgets are not guesses.

Everything is wrapped: a raise in the last step must not lose the first three.
"""
import ctypes, json, os, re, subprocess, sys, time

OUT = "/kaggle/working"
os.makedirs(OUT, exist_ok=True)
res = {"t_start": time.time()}
T0 = time.time()


def say(*a):
    print(*a, flush=True)


def rec(k, v):
    res[k] = v
    say(f"{k:.<40} {v}")
    (open(f"{OUT}/PROBE38.json", "w")).write(json.dumps(res, indent=1, default=str))


# ---------------------------------------------------------------- Q2 device
say("=" * 72)
say("Q2 -- the device, four independent readers")
say("=" * 72)
try:
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                        "--format=csv,noheader"], capture_output=True, text=True, timeout=60)
    rec("nvidia_smi", q.stdout.strip() or f"FAILED {q.stderr.strip()[:120]}")
except Exception as e:
    rec("nvidia_smi", f"FAILED {type(e).__name__}: {e}")

try:
    lib = ctypes.CDLL("libcuda.so.1")
    for attr, nm in ((97, "MaxSharedMemPerBlockOptin"), (75, "CCmajor"), (76, "CCminor"),
                     (16, "MultiprocessorCount"), (8, "MaxSharedMemPerBlock")):
        v = ctypes.c_int(0)
        rc = lib.cuDeviceGetAttribute(ctypes.byref(v), attr, 0)
        rec(f"cuDeviceGetAttribute({attr}) {nm}", v.value if rc == 0 else f"rc={rc}")
except Exception as e:
    rec("libcuda", f"FAILED {type(e).__name__}: {e}")

# ---------------------------------------------------------------- Q3 tools
say("")
say("=" * 72)
say("Q3 -- the toolchain, and whether the ptxas lever works end to end")
say("=" * 72)
for exe in ("nvcc", "ptxas", "nvdisasm", "cuobjdump", "nvcudafe"):
    path = ""
    for d in os.environ.get("PATH", "").split(os.pathsep) + \
             ["/usr/local/cuda/bin", "/usr/local/cuda-12.8/bin", "/opt/cuda/bin"]:
        cand = os.path.join(d, exe)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            path = cand
            break
    rec(f"which {exe}", path or "NOT FOUND")
    if path:
        q = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=60)
        v = (q.stdout or q.stderr).strip().splitlines()
        rec(f"{exe} version", v[-1][:90] if v else "?")

PTXAS = res.get("which ptxas")
NVCC = res.get("which nvcc")
if not PTXAS:
    rec("VERDICT", "NO PTXAS -- the register-budget lever does not exist on this image")
else:
    # The lever, end to end, in miniature. A source with a tunable amount of
    # live-range pressure, compiled to PTX by nvrtc and then to a cubin by
    # ptxas at three register budgets, loaded, and read back with
    # cuFuncGetAttribute -- the same read the real notebook takes.
    src = r'''
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(128) probe(const float* __restrict__ a,
                                                        float* __restrict__ o, int n) {
  float v[24];
  #pragma unroll
  for (int i = 0; i < 24; i++) v[i] = a[threadIdx.x + i];
  float acc = 0.f;
  #pragma unroll
  for (int i = 0; i < 24; i++) acc = fmaf(acc, v[i], v[(i + 7) & 23]);
  o[threadIdx.x] = acc;
}
'''
    ptxf, cubf = f"{OUT}/probe.ptx", f"{OUT}/probe.cubin"
    if NVCC:
        t = time.time()
        # nvcc -ptx is nvrtc-equivalent for this purpose and avoids a ctypes
        # nvrtc binding; the real notebook uses tinygrad's own nvrtc path.
        q = subprocess.run([NVCC, "-ptx", "-arch=sm_75", "-o", ptxf, "-x", "cu", "-"],
                           input=src, capture_output=True, text=True, timeout=300)
        rec("nvcc_ptx", f"rc={q.returncode} in {time.time()-t:.1f}s "
                        f"{(q.stderr or '')[-200:] if q.returncode else ''}")
    else:
        rec("nvcc_ptx", "skipped, no nvcc")

    if os.path.exists(ptxf):
        rec("ptx_bytes", os.path.getsize(ptxf))
        frames = {}
        for mrr in (None, 64, 32, 24):
            cmd = [PTXAS, "-arch=sm_75", "-v", "-O3"] + \
                  ([f"-maxrregcount={mrr}"] if mrr else []) + ["-o", cubf, ptxf]
            t = time.time()
            q = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            txt = (q.stderr or "") + (q.stdout or "")
            fr = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads",
                           txt)
            rg = re.search(r"Used (\d+) registers", txt)
            frames[str(mrr)] = {
                "rc": q.returncode,
                "stack_frame": int(fr.group(1)) if fr else None,
                "spill_stores": int(fr.group(2)) if fr else None,
                "spill_loads": int(fr.group(3)) if fr else None,
                "registers": int(rg.group(1)) if rg else None,
                "secs": round(time.time() - t, 1),
            }
            rec(f"ptxas maxrregcount={mrr}", frames[str(mrr)])
        # and the SASS, so the instruction count of the exact cubin is readable
        if os.path.exists(cubf):
            q = subprocess.run(["nvdisasm", "-c", cubf], capture_output=True, text=True)
            n = len([l for l in q.stdout.splitlines() if "/*" in l and ";" in l])
            rec("sass_instructions_last", n)
        # now load the cubin and read the frame back from the driver, which is
        # the reading the real notebook trusts
        try:
            from tinygrad.runtime.autogen import cuda as _c
        except Exception:
            try:
                _c = ctypes.CDLL("libcuda.so.1")
            except Exception as e:
                rec("driver_readback", f"no cuda binding: {e}")
                _c = None
        if _c is not None:
            for mrr in ("32", "None"):
                cmd = [PTXAS, "-arch=sm_75", "-O3"] + \
                      ([f"-maxrregcount={mrr}"] if mrr != "None" else []) + \
                      ["-o", cubf, ptxf]
                subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                mod = _c.CUmodule()
                rc = _c.cuModuleLoadData(ctypes.byref(mod), open(cubf, "rb").read())
                if rc != 0:
                    rec(f"driver_readback mrr={mrr}", f"cuModuleLoadData rc={rc}")
                    continue
                fn = _c.CUfunction()
                rc = _c.cuModuleGetFunction(ctypes.byref(fn), mod, b"probe")
                if rc != 0:
                    rec(f"driver_readback mrr={mrr}", f"cuModuleGetFunction rc={rc}")
                    continue
                v = ctypes.c_int(0)
                r1 = _c.cuFuncGetAttribute(ctypes.byref(v), 4, fn)   # NUM_REGS
                rec(f"driver NUM_REGS mrr={mrr}", v.value if r1 == 0 else f"rc={r1}")
                v = ctypes.c_int(0)
                r2 = _c.cuFuncGetAttribute(ctypes.byref(v), 3, fn)   # LOCAL_SIZE_BYTES
                rec(f"driver LOCAL_SIZE_BYTES mrr={mrr}", v.value if r2 == 0 else f"rc={r2}")
                rec(f"driver_readback mrr={mrr} VERDICT",
                    f"ptxas said frame {frames.get(mrr, {}).get('stack_frame')}, "
                    f"driver says {v.value}, regs ptxas {frames.get(mrr, {}).get('registers')} "
                    f"driver {rec.get('driver NUM_REGS mrr=' + mrr)}")

# --------------------------------------------------------------- Q4 tinygrad
say("")
say("=" * 72)
say("Q4 -- tinygrad from pip, and its CUDA compiler object")
say("=" * 72)
try:
    t = time.time()
    q = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tinygrad"],
                       capture_output=True, text=True, timeout=420)
    rec("pip_tinygrad", f"rc={q.returncode} in {time.time()-t:.1f}s {(q.stderr or '')[-200:]}")
except Exception as e:
    rec("pip_tinygrad", f"FAILED {type(e).__name__}: {e}")

try:
    import tinygrad
    from tinygrad.helpers import VERSION
    rec("tinygrad_version", VERSION)
    rec("tinygrad_path", tinygrad.__file__)
except Exception as e:
    rec("tinygrad_import", f"FAILED {type(e).__name__}: {e}")

rec("wall_s", round(time.time() - T0, 1))
ok = (isinstance(res.get("nvidia_smi"), str) and "Tesla T4" in res["nvidia_smi"]
      and bool(PTXAS) and isinstance(res.get("tinygrad_version"), str))
rec("VERDICT", "SESSION USABLE -- T4, ptxas present, tinygrad installed" if ok
    else "SOMETHING IS MISSING -- read the rows above before pushing a real notebook")
say("")
say(f"VERDICT: {res['VERDICT']}")
