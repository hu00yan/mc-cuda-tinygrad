#!/usr/bin/env python3
"""Instruction-class census of the SASS CUTLASS actually emits, per arch.

This is the part of the lab that produces hard evidence without a GPU. For each
compiled binary it disassembles the embedded SASS with `cuobjdump -sass` and
tallies instruction mnemonics into architectural buckets, so the "what did the
human choose" table in RESULTS.md rests on emitted code rather than on prose.

Two numbers matter most and they are reported separately:
  * the MMA family actually used (HMMA / IMMA / QMMA / WGMMA / TCGEN05 / ...),
    which is the single most load-bearing per-arch choice;
  * how the operands arrive (LDGSTS == cp.async, LDSM == ldmatrix, UTMALDG/UBLKCP
    == TMA, PRECH/TMA-ish == ptxas-intrinsic), which determines whether the
    kernel needs a human-chosen swizzle to avoid shared-memory bank conflicts.

Usage:
    python3 sass_census.py build/sgemm_sm80_simt [build/sgemm_sm80_tensorop ...]
    python3 sass_census.py --out census.json build/*
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter

# Mnemonic -> bucket. Matched as a prefix, most specific first.
BUCKETS: list[tuple[str, tuple[str, ...]]] = [
    # Blackwell tcgen05 tensor core, 5th gen. Must precede the HMMA bucket
    # because ptxas can emit TCGEN05 with an HMMA-looking prefix on some paths.
    ("tcgen05", ("TCGEN05",)),
    # Hopper warpgroup MMA. Also precedes HMMA.
    ("wgmma", ("WGMMA",)),
    # sm_120 / consumer Blackwell tcgen05-mma variants.
    ("qmma", ("QMMA",)),
    # Ampere / Turing tensor core MMA.
    ("hmma", ("HMMA",)),
    # Integer tensor core MMA.
    ("imma", ("IMMA",)),
    # cp.async (Ampere+ global->shared bypasses L1)
    ("ldgsts/cp.async", ("LDGSTS",)),
    # ldmatrix: shared -> register, the operand gather for mma.sync
    ("ldmatrix", ("LDSM",)),
    # TMA (sm90+): tensor-map async copy and the bulk async group
    ("tma", ("UTMALDG", "UBLKCP", "UBLKPF", "TMA", "TMALDG")),
    # Plain vector loads / stores
    ("global_ld", ("LDG",)),
    ("global_st", ("STG",)),
    ("shared_ld", ("LDS",)),
    ("shared_st", ("STS",)),
    # FFMA is the SIMT float pipeline -- present iff the kernel is not on tensor cores
    ("ffma", ("FFMA",)),
    ("fadd", ("FADD",)),
    ("fmul", ("FMUL",)),
    # integer tensor-core / dp4a style IMAD
    ("imad", ("IMAD",)),
    # barriers and arrive/wait
    ("barrier", ("BAR", "ARRIVES", "MEMBAR", "DEPBAR", "BSYNC")),
    # warp-level primitives used by warp specialisation
    ("named_bar", ("ARRIVES",)),
    # async proxy / elect / vote
    ("elect_vote", ("ELECT", "VOTE", "VOTEU", "POPC", "SHFL")),
    # tmem (Blackwell tensor memory)
    ("tmem", ("TCALLOC", "TMEM", "UTCMMA")),
    ("mma_generic", ("MMA",)),
]

# cuobjdump -sass emits one instruction per line as
#     /*0000*/   <opcode>[.qualifiers] <operands...> ;   /* encoding */
# The predicate in SASS is a *trailing* operand (e.g. "..., 0x30, !PT"), not a
# leading "@P0", so the opcode is simply the first token after the address
# comment. Anything else picks up register names instead of mnemonics.
_MNEM_RE = re.compile(
    r"/\*[0-9a-fA-F]{4}\*/\s+([A-Z][A-Z0-9_]*(?:\.[A-Z0-9_]+)*)"
)


def bucketize(mnem: str) -> str:
    base = mnem.split(".")[0]
    for name, prefixes in BUCKETS:
        for p in prefixes:
            if base == p or mnem.startswith(p):
                return name
    return "other"


def disassemble(path: str) -> str:
    for args in (["cuobjdump", "-sass", path], ["nvdisasm", "-c", path]):
        try:
            r = subprocess.run(args, capture_output=True, text=True, timeout=300)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout
    return ""


def kernel_sections(sass: str) -> list[tuple[str, list[str]]]:
    """Split cuobjdump output into (kernel header, mnemonics) sections."""
    out: list[tuple[str, list[str]]] = []
    header: str | None = None
    cur: list[str] = []
    for line in sass.splitlines():
        if line.startswith("\t\tFunction :") or line.startswith("Function :"):
            if header is not None:
                out.append((header, cur))
            header = line.strip()
            cur = []
            continue
        if header is not None:
            m = _MNEM_RE.search(line)
            if m:
                cur.append(m.group(1))
    if header is not None:
        out.append((header, cur))
    return out


def census(path: str) -> dict:
    sass = disassemble(path)
    if not sass:
        return {"binary": path, "error": "could not disassemble (no cuobjdump/nvdisasm, or not an ELF with SASS)"}

    sections = kernel_sections(sass)
    kernels = []
    for header, mnems in sections:
        c: Counter[str] = Counter(bucketize(m) for m in mnems)
        # Pick the dominant MMA-ish family; this is the "human chose this" signal.
        mma_family = None
        for fam in ("tcgen05", "wgmma", "qmma", "imma", "hmma", "mma_generic"):
            if c.get(fam):
                mma_family = fam
                break
        kernels.append(
            {
                "kernel": header,
                "total_instructions": len(mnems),
                "mma_family": mma_family,
                "buckets": dict(sorted(c.items())),
            }
        )
    return {"binary": path, "kernels": kernels}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("binaries", nargs="+")
    ap.add_argument("--out", default=None, help="write the full census as JSON")
    args = ap.parse_args()

    all_data = []
    for b in args.binaries:
        d = census(b)
        all_data.append(d)
        print(f"=== {b} ===")
        if "error" in d:
            print(f"  {d['error']}")
            continue
        for k in d["kernels"]:
            # Long names get abbreviated to keep the table readable.
            fam = k["mma_family"] or "-"
            n = k["total_instructions"]
            interesting = {
                bb: v
                for bb, v in k["buckets"].items()
                if bb in {
                    "tcgen05", "wgmma", "hmma", "imma", "qmma", "mma_generic",
                    "ldgsts/cp.async", "ldmatrix", "tma", "tmem", "ffma",
                    "shared_st", "shared_ld", "barrier",
                }
            }
            print(f"  {fam:>10}  {n:>7} instrs  {interesting}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(all_data, fh, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
