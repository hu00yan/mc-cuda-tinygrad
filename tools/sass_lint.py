#!/usr/bin/env python3
"""
SASS static lint over real .sass.txt dumps.

WHY THIS EXISTS AND WHY IT IS BETTER THAN THE SOURCE-LEVEL LINT
--------------------------------------------------------------
`tools/factor_lint.py` reads tinygrad's UOp graph. That graph is not stable: in
this version `a - b` lowers to `ADD(a, MUL(b,-1))`, so an earlier version of that
lint searched for `Ops.SUB` and silently matched nothing -- it reported 0 findings
while appearing to work. **Source-level structure is a moving target.**

SASS is what actually executes. Instruction counts there cannot be fooled by an
IR change. This tool counts on the artifact that runs.

WHAT IT COUNTS  (per kernel file)
---------------------------------
  MUFU.EX2      special-function-unit transcendentals
  STL / LDL     register spills to local memory (store / load)
  HMMA/HGMMA    tensor-core instructions
  total         total instructions

Spill traffic is reported as the headline metric because on the KDA kernel it
dwarfs the exp2 story: one dump shows 2,153 STL + 3,268 LDL against 4,096 total
instructions. A kernel moving more data to local memory than it executes
instructions is not exp2-bound; it is spilling, which is a register-allocation
and tile-size problem, not a transcendental problem.

MANDATORY LABELLING (adopted from the team convention @Orion proposed)
----------------------------------------------------------------------
Every number is printed with `(kernel, cc, config)` because three different
experiments have already been quoted as if they were one gap:
  notes/16 §3.3  : 128 vs 3    (Triton offline compile, in a container)
  T4 AOT batch   : 33/17/2     (Kaggle, cc75, varies with BV/num_warps)
  FACTORISATION  : 64 -> 32    (gram factorisation, code-level)
A figure without those subscripts is not comparable to any other figure.

USAGE
    .venv/bin/python tools/sass_lint.py <path-or-glob>
    .venv/bin/python tools/sass_lint.py out/gpush/runs/kaggle/kda-t4/*/tree/payload/kda_t4/out/sass
"""

from __future__ import annotations

import glob
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass

# SASS lines look like:  /*0010*/  MUFU.EX2 R4, R4 ;
# SASS lines may carry a predicate guard: "@!P0 EXIT ;" / "@P0 BRA 0x340 ;".
# A regex that requires the opcode immediately after the offset misses those.
# Offsets are NOT always 4 hex digits: past 0xffff they widen to 5 (/*10000*/),
# and 6 for larger kernels. A {4} quantifier silently drops every instruction
# beyond the first 64 KB of code -- which on the T4 KDA dump is ~77% of them
# (3,842 counted vs 16,952 actually present). Match 4-8 digits.
_INSTR = re.compile(r"^\s*/\*[0-9a-fA-F]{4,8}\*/\s+(?:@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9._]*)")
# NOTE: spill opcodes carry width suffixes -- `STL.64`, `LDL.LU`, `LDL.LU.64`.
# Matching only bare `STL`/`LDL` silently undercounts spills by ~3x on the T4
# dumps (526 vs 1651 STL on aot16_cc75_BV64_w4_s2). Always match the prefix.
_OPCODE = re.compile(r"^(MUFU\.EX2|STL|LDL|HMMA|HGMMA|IMMA|FFMA|LDG|STG)")


@dataclass
class KernelStats:
    path: str
    name: str
    cc: str
    config: str
    total: int
    counts: Counter
    exits: int = 0
    complete: bool = True

    def get(self, k: str) -> int:
        return self.counts.get(k, 0)

    def label(self) -> str:
        return f"{self.name} (cc{self.cc}, {self.config})"


def parse_name(path: str) -> tuple[str, str, str]:
    """Extract (name, cc, config) from filenames like aot_cc75_BV32_w2_s1.sass.txt."""
    base = os.path.basename(path).replace(".sass.txt", "")
    cc = "?"
    m = re.search(r"cc(\d+)", base)
    if m:
        cc = m.group(1)
    # config is everything after the cc token
    config = base
    if m:
        config = base[m.end():].lstrip("_") or "-"
    return base, cc, config


def analyze(path: str) -> KernelStats:
    counts: Counter = Counter()
    total = 0
    exits = 0
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            m = _INSTR.match(line)
            if not m:
                continue
            total += 1
            op = m.group(1)
            if op == "EXIT":
                exits += 1
            mm = _OPCODE.match(op)
            if mm:
                key = mm.group(1)
                # fold width/variant suffixes into the base counter:
                # STL, STL.64, STL.128 -> "STL";  LDL.LU, LDL.LU.64 -> "LDL"
                counts[key] += 1
            elif op.startswith("MUFU.EX2"):
                counts["MUFU.EX2"] += 1
    name, cc, config = parse_name(path)

    # --- INTEGRITY GATE -----------------------------------------------------
    # A dump can be silently truncated, and a truncated dump still parses, so the
    # counts look authoritative when they are not. @Orion found the first
    # instance: kernels whose dump has zero FFMA *and* zero HMMA/HGMMA have no
    # floating-point arithmetic at all, which is impossible for a matmul kernel.
    #
    # I tried to build a general "complete" test and could not make one that
    # holds. The truncation marker ("..........") appears in healthy files too,
    # EXIT count is inconsistent, and a fixed instruction count is not a
    # property of the kernel. So this gate reports only the *definite* signal:
    # no floating-point instruction of any kind. Everything else is UNVERIFIED,
    # not "complete" -- absence of the signal is not evidence of integrity.
    fp = counts.get("FFMA", 0) + counts.get("HMMA", 0) + counts.get("HGMMA", 0)
    complete = fp > 0
    return KernelStats(path, name, cc, config, total, counts, exits, complete)


def main(argv: list[str]) -> int:
    pat = argv[1] if len(argv) > 1 else "out/**/*.sass.txt"
    if os.path.isdir(pat):
        files = sorted(glob.glob(os.path.join(pat, "**", "*.sass.txt"), recursive=True))
    else:
        files = sorted(f for f in glob.glob(pat, recursive=True) if os.path.isfile(f))
    if not files:
        print(f"no .sass.txt matched {pat!r}", file=sys.stderr)
        return 1

    stats = [analyze(f) for f in files]
    print(f"scanned {len(stats)} SASS dumps from {pat}\n")
    bad = [s for s in stats if not s.complete]
    if bad:
        print(f"!! INTEGRITY GATE: {len(bad)} of {len(stats)} dumps contain NO "
              f"floating-point instruction at all (no FFMA, no HMMA/HGMMA). A matmul "
              f"kernel cannot be that, so these are truncated. Their rows are marked "
              f"INCOMPLETE and must not be quoted.\n"
              f"   NOTE: rows WITHOUT the mark are not proven intact -- the gate only "
              f"catches one failure mode. Treat unmarked as UNVERIFIED.\n")

    header = f"{'kernel (cc, config)':<44}{'total':>7}{'exp2':>6}{'HMMA':>6}{'STL':>7}{'LDL':>7}{'spill%':>8}"
    print(header)
    print("-" * len(header))

    # sorted by spill pressure: the actionable signal first
    for s in sorted(stats, key=lambda s: -(s.get("STL") + s.get("LDL"))):
        spill = s.get("STL") + s.get("LDL")
        pct = (100.0 * spill / s.total) if s.total else 0.0
        mark = "" if s.complete else "  <-- INCOMPLETE, do not quote"
        print(f"{s.label():<44}{s.total:>7}{s.get('MUFU.EX2'):>6}"
              f"{s.get('HMMA') + s.get('HGMMA'):>6}{s.get('STL'):>7}{s.get('LDL'):>7}"
              f"{pct:>7.1f}%{mark}")

    print()
    good = [s for s in stats if s.complete] or stats
    worst = max(good, key=lambda s: s.get("STL") + s.get("LDL"))
    print(f"worst spill pressure: {worst.label()}  "
          f"{worst.get('STL')} STL + {worst.get('LDL')} LDL in {worst.total} instructions "
          f"({100.0*(worst.get('STL')+worst.get('LDL'))/max(worst.total,1):.1f}%)")
    print("\nNOTE: every count above is (kernel, cc, config)-specific. Do not compare")
    print("a number from this table with a number from notes/16 or "
          "tools/FACTORISATION-FINDING.md without matching all three subscripts.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
