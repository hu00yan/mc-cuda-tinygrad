#!/usr/bin/env python3
"""
tinygrad_cuda.py -- generate TinyGrad's CUDA C for an operator and compile it
for any architecture, with no GPU present.

This answers a specific question for the study: can the *other* side of the
comparison (TinyGrad's generated CUDA C) also be produced and compiled
offline? It can, and the trick is that TinyGrad's code generator is a pure
function of the UOp graph -- it does not need a device.

The three pieces of machinery:

  1. DEV=NULL:CUDA:<arch>
     TinyGrad's NULL device is a no-op device that initialises with no
     hardware. It accepts an alternative renderer, so "NULL" gives us the
     allocator/scheduler with the CUDA *renderer* attached. No driver, no
     context, no GPU.

  2. CUDARenderer(use_nvcc=True)
     By default the CUDA renderer wants NVRTC, which dlopen()s libnvrtc and
     fails without a CUDA install. Forcing use_nvcc=True makes it shell out
     to the nvcc binary instead, which we do have.

  3. Compiler.compile_cached
     The NULL device's default compiler returns the source string verbatim as
     its "output", so capturing it hands us the generated CUDA C.

Then nvcc -arch=sm_90a -cubin compiles it, and the resulting cubin goes
through the same SASS categoriser as inspect_ptx.py.

    # inside the container
    python3 tinygrad_cuda.py --arch sm_90a --op all --outdir /w/tg
    python3 tinygrad_cuda.py --arch sm_100a --op matmul --outdir /w/tg
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# Reuse inspect_ptx's SASS categoriser so there is one source of truth for what
# a "tensor core op" or a "vectorised load" means across this directory.
from inspect_ptx import CATEGORIES, parse_sass  # noqa: E402


def install_codegen_hook():
    """Force the nvcc driver, and capture generated source instead of compiling."""
    from tinygrad.renderer import cstyle
    from tinygrad.device import Compiler

    _orig = cstyle.CUDARenderer.__init__

    def patched(self, target, use_nvcc=False):
        return _orig(self, target, use_nvcc=True)

    cstyle.CUDARenderer.__init__ = patched

    srcs: list[str] = []

    def capture(self, src):
        srcs.append(src)
        return b""  # pretend it compiled; the NULL device never runs it

    Compiler.compile_cached = capture
    return srcs


# --------------------------------------------------------------------------
# Operators. These are chosen to be the shapes the study actually argues
# about, not generic demo kernels:
#
#   matmul        the control case: should reach tensor cores (HMMA on
#                 Ampere-and-earlier, HMMA/HMMA.16816 style, or wgmma if the
#                 renderer emits it)
#   gather_matmul a strided/gather access feeding a matmul. This is the RC1
#                 "codegen fails to vectorise a strided access" shape from
#                 the prior-work notes: if TinyGrad cannot vectorise the
#                 gather, the global loads collapse to scalar width and that
#                 is visible in the SASS with no GPU at all.
#   fused_reduce  an elementwise chain plus a reduction, i.e. the over-fusion
#                 shape that the KDA result implicates.
# --------------------------------------------------------------------------
def build_ops(name, m=256, k=256, n=128, bs=4, sl=64):
    from tinygrad import Tensor
    from tinygrad.uop.ops import Ops

    if name == "matmul":
        a = Tensor.randn(m, k, device="NULL")
        b = Tensor.randn(k, n, device="NULL")
        return a.matmul(b)

    if name == "gather_matmul":
        a = Tensor.randn(bs, m, k, device="NULL")
        b = Tensor.randn(bs, k, n, device="NULL")
        # force a real gather: index with a non-trivial index tensor
        idx = Tensor.arange(m).flip(axis=0)
        return a[:, idx, :].matmul(b)

    if name == "fused_reduce":
        a = Tensor.randn(bs, sl, k, device="NULL")
        b = Tensor.randn(bs, sl, k, device="NULL")
        return (a * 2.0 + b.sin() * 0.5).sum(axis=1)

    raise SystemExit(f"unknown op {name!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="sm_90a", help="sm_90a, sm_100a, sm_75 ...")
    ap.add_argument("--op", default="all",
                    help="matmul | gather_matmul | fused_reduce | all")
    ap.add_argument("--outdir", type=Path, default=Path("/w/tg"))
    ap.add_argument("--no-compile", action="store_true", help="only emit CUDA C")
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    # DEV must be set before tinygrad builds any device. NULL = no hardware,
    # CUDA = the renderer, <arch> = the tensor-core/dtype table to target.
    #
    # Note the arch split: tinygrad's tensor-core table does `int(arch[3:])`,
    # so it cannot parse an architecture-specific suffix -- "sm_90a" raises
    # ValueError. We hand tinygrad the base arch (sm_90) and let nvcc do the
    # real architecture-specific compile with the full name (sm_90a). The
    # tensor-core selection table is the same for both.
    dev_arch = re.sub(r"(sm_\d+)[a-z]$", r"\1", args.arch)
    os.environ["DEV"] = f"NULL:CUDA:{dev_arch}"
    # TinyGrad's beam search / TC autotuner runs candidate compiles in a
    # multiprocessing pool created with a *spawn* context, so (a) the workers
    # re-import tinygrad and never see the use_nvcc / capture patches below,
    # and (b) each worker dies constructing CUDARenderer because NVRTC cannot
    # be loaded, which hangs the pool forever. PARALLEL=0 makes
    # tinygrad.engine.worker.get_worker_pool() return None, so everything runs
    # in-process and the patches apply. It also skips benchmarking, which is
    # meaningless on a device that cannot execute anything.
    os.environ["PARALLEL"] = "0"
    args.outdir.mkdir(parents=True, exist_ok=True)

    srcs = install_codegen_hook()
    from tinygrad import Tensor  # noqa: F401  (import after DEV is set)
    from tinygrad.device import Device

    ops = ["matmul", "gather_matmul", "fused_reduce"] if args.op == "all" else [args.op]
    report = {"arch": args.arch, "tinygrad_dev_arch": dev_arch, "device": "NULL", "renderer": None, "kernels": {}}

    from tinygrad.renderer import cstyle
    report["renderer"] = type(Device["NULL"].renderer).__name__

    for op in ops:
        srcs.clear()
        try:
            out = build_ops(op, args.m, args.k, args.n)
            out.realize()
        except Exception as e:  # a failed op is a data point, not a crash
            report["kernels"][op] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            continue
        if not srcs:
            report["kernels"][op] = {"ok": False, "error": "no CUDA C emitted (fully optimised away?)"}
            continue
        for i, src in enumerate(srcs):
            cu = args.outdir / f"{op}_{i}.cu"
            cu.write_text(src)
            entry = {"ok": True, "source": str(cu), "bytes": len(src), "kernels_in_program": len(srcs)}
            if not args.no_compile:
                cubin = args.outdir / f"{op}_{i}.cubin"
                p = subprocess.run(
                    ["nvcc", f"-arch={args.arch}", "-cubin", str(cu), "-o", str(cubin)],
                    capture_output=True, text=True)
                if p.returncode == 0:
                    entry["compiled"] = True
                    sass = subprocess.run(["cuobjdump", "-sass", str(cubin)],
                                          capture_output=True, text=True).stdout
                    st = parse_sass(sass)
                    entry["sass_instructions"] = st["total"]
                    entry["categories"] = st["categories"]
                    entry["global_load_widths"] = st["global_ld_widths"]
                    entry["global_store_widths"] = st["global_st_widths"]
                    entry["uses_128bit_vector_ops"] = st["any_128bit"] > 0
                    res = subprocess.run(["cuobjdump", "-res-usage", str(cubin)],
                                         capture_output=True, text=True).stdout
                    reg = re_search_int(res, r"REG:(\d+)")
                    loc = re_search_int(res, r"LOCAL:(\d+)")
                    smem = re_search_int(res, r"SHARED:(\d+)")
                    entry["registers"] = reg
                    entry["local_bytes"] = loc
                    entry["shared_bytes"] = smem
                else:
                    entry["compiled"] = False
                    entry["nvcc_error"] = p.stderr.strip()[-1500:]
            report["kernels"][f"{op}_{i}"] = entry

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print("=" * 72)
    print(f"  TinyGrad CUDA C codegen   arch={args.arch}   device=NULL   renderer={report['renderer']}")
    print("=" * 72)
    for name, e in report["kernels"].items():
        print(f"\n  [{name}]")
        if not e.get("ok"):
            print(f"    FAILED: {e.get('error')}")
            continue
        print(f"    CUDA C          {e['bytes']} bytes -> {e['source']}")
        if "compiled" not in e:
            print("    (not compiled)")
            continue
        if not e["compiled"]:
            print("    nvcc FAILED:")
            for line in e["nvcc_error"].splitlines()[-6:]:
                print(f"      {line}")
            continue
        c = e["categories"]
        print(f"    nvcc -arch={args.arch} -cubin   OK")
        print(f"    registers {e['registers']}, shared {e['shared_bytes']} B, local {e['local_bytes']} B")
        print(f"    SASS {e['sass_instructions']} instructions")
        interesting = [n for n, _, _ in CATEGORIES
                       if c.get(n) and n in ("warpgroup_mma", "tcgen05_mma", "mma_sync",
                                             "cp_async", "tma_bulk", "scalar_ffma",
                                             "global_ld", "global_st")]
        for n in interesting:
            print(f"      {n:<16} {c[n]}")
        print(f"    global load widths    {e['global_load_widths']}")
        print(f"    128-bit vectorised    {'YES' if e['uses_128bit_vector_ops'] else 'no'}")
    print("\n  NOTE: this measures the shape of the generated code only. It says")
    print("  nothing about whether the kernel is numerically correct or fast.")
    return 0


def re_search_int(text, pattern):
    import re
    m = re.search(pattern, text)
    return int(m.group(1)) if m else None


if __name__ == "__main__":
    sys.exit(main())
