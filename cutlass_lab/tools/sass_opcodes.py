#!/usr/bin/env python3
"""Opcode frequency of the SASS in a compiled binary, anchored on the address.

cuobjdump -sass prints one instruction per line as

    /*0000*/   OPCODE.QUALIFIERS   operands...  ;   /* encoding */

The opcode is the first token after the `/*addr*/` comment. Naive greps over the
whole line pick up register names (RZ, P0, UR5, ...) and predicate operands
(!PT), which is how a first attempt at this reported 2318 "RZ". Anchoring on the
address comment and taking only the first token avoids that.

Usage:
    python3 sass_opcodes.py <binary> [<binary> ...] [--top N] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter

LINE = re.compile(r"/\*[0-9a-fA-F]{4}\*/\s+([A-Z][A-Z0-9_]*(?:\.[A-Z0-9_]+)*)")

# The opcodes the study actually cares about, with the PTX name they implement.
# SASS names do not match PTX names, which is the whole reason this table exists:
#   wgmma.mma_async (PTX)  -> HGMMA  (SASS, Hopper)
#   mma.sync (PTX)         -> HMMA   (SASS, sm75/sm80/sm89 tensor cores)
#   mma.sync.s32           -> IMMA   (SASS, integer tensor cores)
#   cp.async.*             -> LDGSTS (SASS, Ampere+)
#   ldmatrix               -> LDSM   (SASS)
#   cp.async.bulk.tensor   -> UTMALDG / UBLKCP (SASS, TMA)
#   tcgen05.mma            -> TCGEN05 (SASS, Blackwell)
#   FFMA                   -> FFMA   (SASS, SIMT float pipeline, i.e. NO tensor cores)
NOTABLE = [
    ("HGMMA", "wgmma.mma_async (Hopper warpgroup MMA)"),
    ("HMMA", "mma.sync tensor core (Turing/Ampere/Ada)"),
    ("IMMA", "mma.sync.s32 integer tensor core"),
    ("QMMA", "mma.sync s64/misc tensor core"),
    ("LDGSTS", "cp.async (global->shared, Ampere+)"),
    ("LDSM", "ldmatrix (shared->register)"),
    ("UTMALDG", "TMA tensor-map load (cp.async.bulk.tensor)"),
    ("UBLKCP", "TMA bulk copy group (cp.async.bulk)"),
    ("TCGEN05", "tcgen05.mma (Blackwell tensor core)"),
    ("TMEM", "tensor-memory (Blackwell) access"),
    ("FFMA", "SIMT float FMA (no tensor core)"),
    ("FMUL", "SIMT float multiply"),
    ("STS", "shared-memory store"),
    ("LDS", "shared-memory load"),
    ("BAR", "barrier sync"),
    ("DEPBAR", "dependency barrier"),
    ("ARRIVES", "mbarrier arrive"),
]


def disassemble(path: str) -> str:
    r = subprocess.run(["cuobjdump", "-sass", path], capture_output=True, text=True)
    if r.returncode == 0:
        return r.stdout
    r = subprocess.run(["nvdisasm", "-c", path], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("binaries", nargs="+")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    out = []
    for b in args.binaries:
        sass = disassemble(b)
        c: Counter[str] = Counter()
        n = 0
        for line in sass.splitlines():
            m = LINE.search(line)
            if m:
                n += 1
                c[m.group(1).split(".")[0]] += 1
        if n == 0:
            print(f"=== {b}: could not disassemble ===")
            continue
        notable = {k: c.get(k, 0) for k, _ in NOTABLE if c.get(k, 0)}
        print(f"=== {b}  ({n} instructions) ===")
        for k, why in NOTABLE:
            if c.get(k):
                print(f"  {k:<10} {c[k]:>5}   {why}")
        if not notable:
            print("  (no notable tensor-core / transfer opcodes found)")
        print(f"  top: {', '.join(f'{k}={v}' for k, v in c.most_common(args.top))}")
        out.append({"binary": b, "total_instructions": n,
                    "opcodes": dict(c.most_common()),
                    "notable": {k: c.get(k, 0) for k, _ in NOTABLE}})
        print()

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
