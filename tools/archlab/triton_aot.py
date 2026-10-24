#!/usr/bin/env python3
"""
triton_aot.py -- AOT-compile a Triton kernel for an arbitrary NVIDIA
architecture with no GPU present.

Why this matters for the study: the free-tier hardware is T4 (sm_75) and P100
(sm_60), where every modern Triton codegen path is dark. But Triton is a
compiler, and compilers do not need the hardware. Passing an explicit
GPUTarget("cuda", 90, 32) makes it emit code for compute capability 9.0 --
verified here to produce PTX whose header literally reads `.target sm_90a` --
with zero active CUDA drivers.

The critical flag is `target=`. Without it Triton calls the driver to ask what
device it is on, and there is no device. With it, the whole pipeline
(ttir -> ttgir -> llir -> ptx -> cubin) runs in-process.

    # inside the container
    python3 triton_aot.py --kernel fla_kda_chunk_h \\
        --signature k=fla_kda_chunk_h:kernel_sig.json \\
        --cc 90 --outdir /w/tri
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from inspect_ptx import CATEGORIES, parse_sass  # noqa: E402


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ptx_histogram(ptx: str) -> dict:
    ops: dict[str, int] = {}
    for line in ptx.splitlines():
        line = line.strip()
        m = re.match(r"^([a-z][a-z0-9_.]*)", line)
        if m and not line.startswith(("//", ".", "$", "{")):
            ops[m.group(1)] = ops.get(m.group(1), 0) + 1
    return ops


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=Path, default=HERE / "fla_kda_chunk_h.py",
                    help="python file holding a @triton.jit kernel")
    ap.add_argument("--kernel", default="chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
                    help="name of the jit function")
    ap.add_argument("--signature", type=Path, required=True,
                    help="JSON: {arg: type-or-'constexpr'} plus an optional "
                         "'__constexprs__' object of {name: value}")
    ap.add_argument("--cc", type=int, default=90, help="compute capability (90 = Hopper)")
    ap.add_argument("--warp-size", type=int, default=32)
    ap.add_argument("--num-warps", type=int, default=None, help="override if no configs")
    ap.add_argument("--num-stages", type=int, default=None)
    ap.add_argument("--outdir", type=Path, default=Path("/work/tri"))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    import triton
    from triton.backends.compiler import GPUTarget

    spec = json.loads(args.signature.read_text())
    constexprs = spec.pop("__constexprs__", {})
    num_warps = args.num_warps if args.num_warps is not None else constexprs.pop("__num_warps__", 4)
    num_stages = args.num_stages if args.num_stages is not None else constexprs.pop("__num_stages__", 2)

    mod = load_module(args.file)
    fn = getattr(mod, args.kernel, None)
    if fn is None:
        cands = [n for n in dir(mod) if hasattr(getattr(mod, n), "cache_key")]
        raise SystemExit(f"no jit function {args.kernel!r}; candidates: {cands}")
    print(f"triton {triton.__version__}   kernel={args.kernel}   cc={args.cc}")

    # Is there a driver at all? Record it either way: the whole point is that
    # the compile succeeds regardless.
    driver_state = "absent"
    try:
        from triton.runtime import driver
        d = driver.active
        driver_state = f"present ({type(d).__name__})"
    except Exception as e:
        driver_state = f"ABSENT ({type(e).__name__}: {str(e)[:60]})"
    print(f"cuda driver: {driver_state}")

    src = triton.compiler.ASTSource(fn=fn, signature=spec, constexprs=constexprs)
    target = GPUTarget("cuda", args.cc, args.warp_size)
    print(f"target: {target}")

    try:
        k = triton.compile(src, target=target,
                           options={"num_warps": num_warps, "num_stages": num_stages})
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nRESULT: AOT COMPILE FAILED: {type(e).__name__}: {e}")
        return 1

    args.outdir.mkdir(parents=True, exist_ok=True)
    ptx = k.asm.get("ptx", "")
    (args.outdir / f"{args.kernel}.ptx").write_text(ptx)
    (args.outdir / f"{args.kernel}.cubin").write_bytes(k.asm["cubin"])
    for a in ("ttir", "ttgir", "llir"):
        if a in k.asm:
            (args.outdir / f"{args.kernel}.{a}").write_text(k.asm[a])

    hdr = [l for l in ptx.splitlines()[:6] if l.startswith((".version", ".target", ".address_size"))]
    hist = ptx_histogram(ptx)
    interesting = {n: c for n, c in sorted(hist.items(), key=lambda kv: -kv[1])
                   if n in ("ld.global", "ld.global.v4", "ld.global.nc", "st.global",
                            "st.global.v4", "mma.sync", "wgmma.mma_async", "cp.async",
                            "cp.async.bulk.tensor", "cp.async.bulk", "bar.sync",
                            "mbarrier.arrive", "mbarrier.init", "ld.shared", "st.shared",
                            "fma.rn.f32", "add.f32", "cvt.rn.bf16", "cvt.f32.bf16",
                            "ldmatrix", "ex2.approx.f32", "mul.f32", "fma.rn.bf16")}

    # SASS via our own toolchain, if ptxas/cuobjdump are present
    sass = {}
    import shutil
    if shutil.which("ptxas") and shutil.which("cuobjdump") and hdr:
        # Keep the architecture-specific suffix: the PTX says `.target sm_90a`
        # and ptxas refuses to assemble it under plain `-arch=sm_90`.
        sm = re.search(r"\b(sm_\d+[a-z]?)\b", " ".join(hdr))
        smname = sm.group(1) if sm else "sm_90a"
        cub = args.outdir / f"{args.kernel}.cubin"
        # ptxas -v is the AUTHORITATIVE source for spills.
        #
        # Do not trust `cuobjdump -res-usage` here: on this very kernel it
        # reports LOCAL:0 while ptxas reports 380 bytes of spill stores and
        # 372 of spill loads, and the SASS contains 188 LDL/STL instructions.
        # res-usage LOCAL is the local *frame* size, not spill traffic. Its
        # SHARED field is likewise unreliable for the dynamic (extern) shared
        # memory that Triton always emits.
        import subprocess
        v = subprocess.run(
            ["ptxas", f"-arch={smname}", "-v", str(args.outdir / f"{args.kernel}.ptx"),
             "-o", "/tmp/archlab_recheck.cubin"], capture_output=True, text=True)
        # ptxas writes its -v report to STDERR, not stdout.
        report_txt = v.stdout + v.stderr
        for pat, key in ((r"Used (\d+) registers", "registers"),
                         (r"(\d+) bytes stack frame", "stack_frame_bytes"),
                         (r"(\d+) bytes spill stores", "spill_store_bytes"),
                         (r"(\d+) bytes spill loads", "spill_load_bytes"),
                         (r"(\d+) bytes smem", "static_smem_bytes"),
                         (r"used (\d+) barriers", "barriers")):
            m = re.search(pat, report_txt)
            if m:
                sass[key] = int(m.group(1))
        if v.returncode != 0:
            print(f"  warning: ptxas -v returned {v.returncode}: "
                  f"{report_txt.strip()[-200:]}")
        r = subprocess.run(["cuobjdump", "-sass", str(cub)], capture_output=True, text=True)
        if r.returncode == 0:
            sass.update(parse_sass(r.stdout))

    report = {
        "ok": True, "triton": triton.__version__, "kernel": args.kernel,
        "target": str(target), "cuda_driver": driver_state,
        "ptx_header": hdr, "artifacts": sorted(k.asm.keys()),
        "num_warps": num_warps, "num_stages": num_stages,
        "ptx_total_instructions": sum(hist.values()),
        "ptx_selected": interesting,
        "sass": {kk: sass.get(kk) for kk in
                 ("total", "categories", "global_ld_widths", "global_st_widths",
                  "shared_ld_widths", "shared_st_widths", "any_128bit",
                  "registers", "stack_frame_bytes", "spill_store_bytes",
                  "spill_load_bytes", "static_smem_bytes", "barriers")
                 if kk in sass} if sass else None,
        "outdir": str(args.outdir),
    }
    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print("\nRESULT: AOT COMPILE OK")
    print(f"  artifacts: {report['artifacts']}")
    print(f"  ptx header: {hdr}")
    print(f"  num_warps={num_warps} num_stages={num_stages}")
    print(f"  ptx instructions: {report['ptx_total_instructions']}")
    for n, c in list(interesting.items()):
        print(f"    ptx {n:<28} {c}")
    if sass:
        print(f"  SASS: {sass['total']} instructions, "
              f"REG={sass.get('registers')} "
              f"spill={sass.get('spill_store_bytes', 0)}B st/"
              f"{sass.get('spill_load_bytes', 0)}B ld, "
              f"stack={sass.get('stack_frame_bytes')}B")
        for name, _, desc in CATEGORIES:
            if sass["categories"].get(name):
                print(f"    sass {name:<16} {sass['categories'][name]:<5} {desc}")
        print(f"    global load widths  {sass['global_ld_widths']}")
        print(f"    global store widths {sass['global_st_widths']}")
    print(f"  wrote {args.outdir}/{args.kernel}.{{ptx,cubin,ttir,ttgir,llir}}")
    print("\n  NOTE: this proves the kernel COMPILES for this arch. It was never")
    print("  executed, so nothing here says it is correct or fast.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
