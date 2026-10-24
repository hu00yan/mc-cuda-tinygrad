#!/usr/bin/env python3
"""
inspect_ptx.py -- GPU-free static inspection of generated GPU code.

Compiles a .ptx file for an arbitrary NVIDIA architecture and reports the
static resource and instruction-shape facts that matter for a kernel-quality
study, WITHOUT a GPU and WITHOUT a driver. ptxas is a pure compiler: it will
happily target sm_90a or sm_100 on a laptop.

This exists because the free-tier hardware (T4 = sm_75, P100 = sm_60) cannot
run any modern kernel: WGMMA, TMA, tcgen05 and warp specialization are all dark
there, so the newest hand-written kernels cannot be measured. Compiling them
offline lets you at least see the code the compiler produced.

    python3 tools/archlab/inspect_ptx.py out/archlab/probe.ptx --arch sm_90a
    python3 tools/archlab/inspect_ptx.py k.ptx -a sm_90a --block-size 128 --json

WHAT IS MEASURED (exact, straight out of the toolchain)
    registers/thread, static shared memory, stack frame,
    spill stores, spill loads, barrier count
    full SASS instruction histogram, and counts in these categories:
      warpgroup MMA (Hopper wgmma / Blackwell tcgen05), per-warp mma.sync,
      async copy global->shared (cp.async / LDGSTS), TMA bulk copy,
      shared-memory ld/st, global ld/st widths, 128-bit vectorisation,
      FFMA (scalar FMA) count, warp-group sync instructions

WHAT IS NOT MEASURED -- and this tool will say so rather than guess
    * correctness. ptxas validates types, control flow and alignment. It does
      not validate fragment layouts or data mapping. A kernel that assembles
      cleanly can still compute garbage.
    * dynamic (extern) shared memory. The size is a launch-time parameter;
      ptxas does not report it. Only static .shared is measured here.
    * bank conflicts, achieved occupancy, L2/DRAM traffic, latency hiding,
      warp divergence, and anything that needs a clock or a memory system.
    * occupancy needs a block size. If the PTX does not pin one with
      .reqntid/.maxntid, you must pass --block-size; the tool refuses to
      invent one.
    * the per-SM limits in ARCH_LIMITS are a transcribed table, not something
      queried from the toolchain. Occupancy is therefore an ESTIMATE.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
# tools/archlab/inspect_ptx.py -> repo root is two levels up. Tolerate this
# file being imported from a flat directory inside a container image.
REPO = _HERE.parents[1] if len(_HERE.parents) > 1 else _HERE

DEFAULT_IMAGE = "archlab/nvcc:12.9"

# Per-SM hardware limits, transcribed from the CUDA C Programming Guide
# "Technical Specifications" tables. These are NOT queried from the toolchain,
# so every occupancy figure derived from them is an estimate.
#   max_threads   maximum resident threads per SM
#   regs          32-bit registers per SM
#   max_blocks    maximum resident thread blocks per SM
#   smem          usable shared memory per SM, bytes
#   gran          register allocation granularity, registers per thread
ARCH_LIMITS = {
    "sm_60":  dict(max_threads=2048, regs=65536, max_blocks=16, smem=49152,  gran=8),
    "sm_61":  dict(max_threads=2048, regs=65536, max_blocks=16, smem=49152,  gran=8),
    "sm_70":  dict(max_threads=2048, regs=65536, max_blocks=16, smem=98304,  gran=8),
    "sm_75":  dict(max_threads=1024, regs=65536, max_blocks=16, smem=65536,  gran=8),
    "sm_80":  dict(max_threads=2048, regs=65536, max_blocks=32, smem=166912, gran=8),
    "sm_86":  dict(max_threads=1536, regs=65536, max_blocks=16, smem=101376, gran=8),
    "sm_89":  dict(max_threads=1536, regs=65536, max_blocks=16, smem=101376, gran=8),
    "sm_90":  dict(max_threads=2048, regs=65536, max_blocks=32, smem=232448, gran=8),
    "sm_90a": dict(max_threads=2048, regs=65536, max_blocks=32, smem=232448, gran=8),
    "sm_100": dict(max_threads=2048, regs=65536, max_blocks=32, smem=232448, gran=8),
    "sm_100a": dict(max_threads=2048, regs=65536, max_blocks=32, smem=232448, gran=8),
    "sm_120": dict(max_threads=1536, regs=65536, max_blocks=24, smem=101376, gran=8),
}

# SASS mnemonic families. Anything not matched here still shows up in the full
# histogram, so the categorisation cannot hide instructions from you.
CATEGORIES = [
    # (category, regex, human meaning)
    ("warpgroup_mma",   r"^(?:[HQB]GMMA)",        "warpgroup MMA  (wgmma, Hopper/Blackwell)"),
    ("tcgen05_mma",     r"^TCGMMA",               "5th-gen tensor-core MMA (tcgen05, sm_100)"),
    ("mma_sync",        r"^(?:[HQIOB]MMA)(?:\.| )", "per-warp mma.sync (Ampere and earlier)"),
    ("cp_async",        r"^LDGSTS",               "async copy global->shared (cp.async)"),
    ("tma_bulk",        r"^(?:UBLKCP|UTMA)",     "TMA / bulk async copy (cp.async.bulk)"),
    ("shared_ld",       r"^LDS",                  "shared-memory load"),
    ("shared_st",       r"^STS",                  "shared-memory store"),
    ("global_ld",       r"^LDG\.",                "global load (LDGSTS counted as cp_async)"),
    ("global_st",       r"^STG",                  "global store"),
    ("scalar_ffma",     r"^FFMA",                 "scalar float FMA (non-tensor-core math)"),
    ("warpgroup_sync",  r"^WARPGROUP",            "warpgroup synchronisation (wgmma.fence/wait)"),
    ("barrier",         r"^BAR(?:\.| )",          "block barrier (bar.sync)"),
    ("mbarrier",        r"^SYNCS\.",              "mbarrier / cluster sync"),
    ("local_ldst",      r"^(?:LDL|STL)",          "local (spill) memory traffic"),
]


def run_in_container(ptx: Path, arch: str, image: str, timeout: int):
    """Run ptxas -v and cuobjdump on the host's file, inside a linux/amd64
    container. Returns the three raw outputs. Compiling inside the container
    is what makes this work on an arm64 host with no CUDA at all."""
    if not shutil.which("docker"):
        raise SystemExit(
            "docker not found. This tool compiles inside a container because "
            "ptxas is an x86-64 binary and the host may have no CUDA.\n"
            f"Build the image first:  bash {REPO/'tools/archlab/build_image.sh'}"
        )
    marker_ptxas = "@@@PTXAS@@@"
    marker_sass = "@@@SASS@@@"
    marker_res = "@@@RESUSAGE@@@"
    qptx = shlex.quote(ptx.name)
    script = f"""
export PATH=/usr/local/cuda-12.9/bin:$PATH
PTXAS_OUT=$(mktemp)
if ptxas -arch={shlex.quote(arch)} -v {qptx} -o /tmp/archlab.cubin > "$PTXAS_OUT" 2>&1; then
  echo "{marker_ptxas}"
  cat "$PTXAS_OUT"
  echo "{marker_sass}"
  cuobjdump -sass /tmp/archlab.cubin
  echo "{marker_res}"
  cuobjdump -res-usage /tmp/archlab.cubin
else
  echo "{marker_ptxas}"
  cat "$PTXAS_OUT"
fi
"""
    # Mount the file's own directory so relative includes and the .ptx resolve.
    proc = subprocess.run(
        ["docker", "run", "--rm", "--platform", "linux/amd64",
         "-v", f"{ptx.parent}:/in:ro", "-w", "/in", image, "sh", "-c", script],
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0 and marker_ptxas not in proc.stdout:
        raise SystemExit(
            f"container failed (rc={proc.returncode})\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    out = {"ptxas": "", "sass": "", "resusage": "", "stderr": proc.stderr}
    if marker_ptxas in proc.stdout:
        rest = proc.stdout.split(marker_ptxas, 1)[1]
        out["ptxas"] = rest
        if marker_sass in rest:
            out["ptxas"], rest = rest.split(marker_sass, 1)
            out["sass"] = rest
            if marker_res in rest:
                out["sass"], out["resusage"] = rest.split(marker_res, 1)
    return out


def parse_ptx_source(ptx: Path):
    """Facts that are only visible in the PTX text itself."""
    txt = ptx.read_text()
    static_smem = sum(int(m) for m in re.findall(r"^\s*\.shared\b[^;]*\[\s*(\d+)\s*\]", txt, re.M))
    has_dyn = bool(re.search(r"^\s*\.extern\s+\.shared", txt, re.M))
    ntid = re.search(r"^\s*\.(?:reqntid|maxntid)\s+([\d,\s]+)", txt, re.M)
    block = None
    if ntid:
        dims = [int(x) for x in re.findall(r"\d+", ntid.group(1))]
        block = 1
        for d in dims:
            block *= d
    return {
        "static_smem_declared": static_smem,
        "has_dynamic_smem": has_dyn,
        "block_size_from_ptx": block,
    }


def parse_ptxas(text: str):
    res: dict = {
        "registers": None, "barriers": None, "stack_frame": None,
        "spill_stores": None, "spill_loads": None, "smem_bytes": None,
        "gmem_bytes": None, "entries": [], "error": None,
    }
    if re.search(r"\berror\b", text):
        errs = [l.strip() for l in text.splitlines() if "error" in l]
        res["error"] = "\n".join(errs)
        return res
    for m in re.finditer(r"Used (\d+) registers, used (\d+) barriers(?:, (\d+) bytes smem)?", text):
        res["registers"] = int(m.group(1))
        res["barriers"] = int(m.group(2))
        if m.group(3):
            res["smem_bytes"] = int(m.group(3))
    for m in re.finditer(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", text):
        res["stack_frame"] = int(m.group(1))
        res["spill_stores"] = int(m.group(2))
        res["spill_loads"] = int(m.group(3))
    for m in re.finditer(r"(\d+) bytes gmem", text):
        res["gmem_bytes"] = int(m.group(1))
    res["entries"] = re.findall(r"Compiling entry function '(\w+)' for '(\w+)'", text)
    return res


SASS_LINE = re.compile(r"/\*[0-9a-fA-F]{4,6}\*/\s+(@!?\w+\s+)?([A-Z][A-Z0-9._]*)")


def parse_sass(text: str):
    """Instruction histogram plus the category table."""
    hist: dict[str, int] = {}
    for m in SASS_LINE.finditer(text):
        mnem = m.group(2)
        hist[mnem] = hist.get(mnem, 0) + 1
    cats = {name: 0 for name, _, _ in CATEGORIES}
    for name, pattern, _ in CATEGORIES:
        rx = re.compile(pattern)
        cats[name] = sum(c for mn, c in hist.items() if rx.match(mn))

    # vector width analysis for global traffic
    def widths(prefix):
        got = {}
        for mn, c in hist.items():
            if not mn.startswith(prefix):
                continue
            w = re.search(r"\.(\d+)$", mn)
            got[w.group(1) if w else "scalar"] = got.get(w.group(1) if w else "scalar", 0) + c
        return got

    vec128 = sum(c for mn, c in hist.items() if mn.endswith(".128"))
    return {
        "total": sum(hist.values()),
        "histogram": dict(sorted(hist.items(), key=lambda kv: -kv[1])),
        "categories": cats,
        "global_ld_widths": widths("LDG"),
        "global_st_widths": widths("STG"),
        "shared_ld_widths": widths("LDS"),
        "shared_st_widths": widths("STS"),
        "any_128bit": vec128,
    }


def occupancy(reg, block, smem_block, arch):
    """Estimated occupancy. Returns (percent, limiting_constraint, notes)."""
    lim = ARCH_LIMITS.get(arch)
    if lim is None:
        return None, None, [f"arch {arch} not in the local limit table; occupancy not computed"]
    notes = []
    if reg is None:
        return None, None, notes + ["register count unavailable; occupancy not computed"]
    if block is None:
        return None, None, notes + [
            "block size unknown (no .reqntid/.maxntid in the PTX and no --block-size).",
            "Occupancy is undefined without it. Pass --block-size to get an estimate.",
        ]
    if block % 32:
        return None, None, notes + [f"block size {block} is not a multiple of the 32-thread warp size"]
    if smem_block is None:
        notes.append("shared memory is dynamic (extern .shared): the per-block smem cost is a "
                     "launch-time parameter, so smem is NOT included in this limit. Treat the "
                     "result as an upper bound.")
    r_alloc = -(-reg // lim["gran"]) * lim["gran"]          # round up to granularity
    warps_per_block = block // 32
    regs_warps = lim["regs"] // (r_alloc * 32)              # warps/SM by registers
    blocks = min(
        lim["regs"] // (r_alloc * block),                    # by registers
        lim["max_threads"] // block,                        # by resident threads
        lim["max_blocks"],                                   # by block slots
        lim["smem"] // smem_block if smem_block else 1 << 30,
    )
    blocks = max(blocks, 0)
    if blocks == 0:
        return 0.0, "register-file", notes + [
            f"{reg} regs/thread x {block} threads exceeds the {lim['regs']}-register file per SM; "
            "this kernel cannot be launched at this block size"]
    active_warps = blocks * warps_per_block
    max_warps = lim["max_threads"] // 32
    pct = 100.0 * active_warps / max_warps
    # name the binding constraint
    cands = {
        "register-file": lim["regs"] // (r_alloc * block),
        "max-threads/SM": lim["max_threads"] // block,
        "block-slots/SM": lim["max_blocks"],
    }
    if smem_block:
        cands["shared-memory"] = lim["smem"] // smem_block
    binding = min(cands, key=lambda k: cands[k])
    if cands[binding] >= max_warps // warps_per_block and active_warps == max_warps:
        binding = "none (full occupancy)"
    notes.append(f"register allocation: {reg} -> {r_alloc} regs/thread (granularity {lim['gran']})")
    return pct, binding, notes


def fmt_bytes(n):
    if n is None:
        return "unknown"
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n} B ({n/1024:.1f} KiB)"
    return f"{n} B ({n/1024/1024:.1f} MiB)"


def fmt_widths(d):
    return ", ".join(f"{k}: {n}" for k, n in sorted(d.items())) if d else "none"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ptx", type=Path, help=".ptx file to inspect")
    ap.add_argument("-a", "--arch", default="sm_90a",
                    help="target arch, e.g. sm_90a, sm_100a, sm_75 (default: sm_90a)")
    ap.add_argument("-b", "--block-size", type=int, default=None,
                    help="threads per block, for the occupancy estimate")
    ap.add_argument("--image", default=DEFAULT_IMAGE, help=f"container image (default {DEFAULT_IMAGE})")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    ap.add_argument("--sass", action="store_true", help="also dump the full SASS listing")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    if not args.ptx.exists():
        raise SystemExit(f"no such file: {args.ptx}")
    if not shutil.which("docker"):
        raise SystemExit("docker is required; see tools/archlab/build_image.sh")

    raw = run_in_container(args.ptx.resolve(), args.arch, args.image, args.timeout)
    ptxinfo = parse_ptx_source(args.ptx)
    v = parse_ptxas(raw["ptxas"])
    sass = parse_sass(raw["sass"])

    if v["error"]:
        print(f"ptxas FAILED for {args.arch}:\n{v['error']}", file=sys.stderr)
        if args.json:
            print(json.dumps({"arch": args.arch, "ok": False, "error": v["error"]}, indent=2))
        return 1

    # shared memory: ptxas's own figure wins; fall back to the PTX declaration
    smem_block = v["smem_bytes"]
    if smem_block is None and ptxinfo["static_smem_declared"]:
        smem_block = ptxinfo["static_smem_declared"]
        smem_source = "summed from static .shared declarations in the PTX"
    elif smem_block is not None:
        smem_source = "ptxas -v"
    else:
        smem_source = "none reported (shared memory is dynamic/extern)"

    block = args.block_size or ptxinfo["block_size_from_ptx"]
    block_source = ("--block-size" if args.block_size
                    else "PTX .reqntid/.maxntid" if ptxinfo["block_size_from_ptx"]
                    else "UNKNOWN")
    pct, binding, notes = occupancy(v["registers"], block, smem_block, args.arch)

    entry = ", ".join(f"{n}" for n, _ in v["entries"]) or "?"
    report = {
        "file": str(args.ptx),
        "arch": args.arch,
        "entry": entry,
        "ok": True,
        "resources": {
            "registers_per_thread": v["registers"],
            "static_smem_bytes_per_block": smem_block,
            "static_smem_source": smem_source,
            "has_dynamic_smem": ptxinfo["has_dynamic_smem"],
            "stack_frame_bytes": v["stack_frame"],
            "spill_store_bytes": v["spill_stores"],
            "spill_load_bytes": v["spill_loads"],
            "barriers": v["barriers"],
        },
        "occupancy_estimate": {
            "percent": pct,
            "binding_constraint": binding,
            "block_size": block,
            "block_size_source": block_source,
            "notes": notes,
        },
        "sass": {
            "instruction_count": sass["total"],
            "categories": sass["categories"],
            "category_legend": {n: d for n, _, d in CATEGORIES},
            "global_load_widths": sass["global_ld_widths"],
            "global_store_widths": sass["global_st_widths"],
            "shared_load_widths": sass["shared_ld_widths"],
            "shared_store_widths": sass["shared_st_widths"],
            "uses_128bit_vector_ops": sass["any_128bit"] > 0,
            "histogram": sass["histogram"],
        },
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    c = report["sass"]["categories"]
    r = report["resources"]
    print("=" * 72)
    print(f"  {args.ptx}   ->   {args.arch}   ({entry})")
    print("=" * 72)
    print("\n-- resources (exact, from ptxas) --")
    print(f"  registers / thread        {r['registers_per_thread']}")
    print(f"  static smem / block       {fmt_bytes(r['static_smem_bytes_per_block'])}   [{r['static_smem_source']}]")
    if r["has_dynamic_smem"]:
        print("  dynamic smem              PRESENT (extern .shared) -- size is a launch-time")
        print("                            parameter and is NOT measured by ptxas")
    print(f"  stack frame               {fmt_bytes(r['stack_frame_bytes'])}")
    print(f"  spill stores              {fmt_bytes(r['spill_store_bytes'])}")
    print(f"  spill loads               {fmt_bytes(r['spill_load_bytes'])}")
    print(f"  barriers                  {r['barriers']}")

    print("\n-- occupancy (ESTIMATE) --")
    o = report["occupancy_estimate"]
    if o["percent"] is None:
        print("  CANNOT DETERMINE")
        for n in o["notes"]:
            print(f"    - {n}")
    else:
        print(f"  {o['percent']:.1f}%   limited by: {o['binding_constraint']}")
        print(f"  block size {o['block_size']} (from {o['block_size_source']})")
        for n in o["notes"]:
            print(f"    - {n}")

    print("\n-- SASS instruction shape (exact, from cuobjdump) --")
    print(f"  {sass['total']} instructions total")
    for name, _, desc in CATEGORIES:
        mark = " <==" if c[name] else ""
        print(f"    {name:<16} {c[name]:>5}   {desc}{mark}")
    print("\n  global load widths        " + fmt_widths(sass["global_ld_widths"]))
    print("  global store widths       " + fmt_widths(sass["global_st_widths"]))
    print("  shared load widths        " + fmt_widths(sass["shared_ld_widths"]))
    print("  shared store widths       " + fmt_widths(sass["shared_st_widths"]))
    print(f"  128-bit vectorised ops    {'YES' if sass['any_128bit'] else 'no'}  ({sass['any_128bit']})")

    print("\n-- what this tool did NOT measure --")
    for line in (
        "correctness: ptxas checks types/control-flow/alignment, never fragment layout or",
        "             data mapping. Clean assembly does not mean the kernel is right.",
        "bank conflicts, achieved occupancy, DRAM/L2 traffic, latency hiding, divergence.",
        "occupancy is an estimate from a transcribed per-SM limit table, not a measured value.",
    ):
        print(f"  {line}")

    if args.sass:
        print("\n-- full SASS --")
        print(raw["sass"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
