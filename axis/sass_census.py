#!/usr/bin/env python3
"""sass_census.py -- per-function SASS instruction census for the reference cubin.

Run on a host with a `cuobjdump -sass` dump. Used on this host (no GPU) to prove
two things about the hand-written arm before any Kaggle quota is spent:

  1. `axis_gemm_f16` really emits HMMA, so the fp16 rung is a tensor-core
     rung and not a scalar-emulation rung wearing a tensor-core label.
     notes/06 C-19 is the correction that makes this necessary: an earlier
     "fp16 reaches a tensor core" claim in this repo was a sweep artefact, and
     the dtype rather than the shape was the real gate.
  2. nothing in the cubin is sm_75-illegal, so a kernel that runs here runs on a
     T4. Same gate minecu_ref/build.sh applies to the vendored reference.
"""
from __future__ import annotations

import re
import sys
from collections import Counter

FUNC_RE = re.compile(r"Function : (\S+)")
OP_RE = re.compile(r"/\*[0-9a-f]{4}\*/\s+(?:@!?P\d\s+)?([A-Z][A-Z0-9_]*)")

# Instructions that do not exist on sm_75. HMMA-16832 and the sm_90+ families are
# the ones a careless kernel picks up. This list is the same one minecu_ref uses.
ILLEGAL_SM75 = ("HMMA.16832", "HGMMA", "LDGSTS", "REDUX", "ELECT", "BAR.RED", "F2FP",
                "LDSM.M88", "UTCCP", "MMA.SP")


def census(sass: str) -> dict:
    out: dict[str, Counter] = {}
    cur = None
    for line in sass.splitlines():
        m = FUNC_RE.search(line)
        if m:
            cur = m.group(1)
            out[cur] = Counter()
            continue
        if cur is None:
            continue
        m = OP_RE.search(line)
        if m:
            out[cur][m.group(1)] += 1
    return out


def main(argv: list[str]) -> int:
    path = argv[1] if len(argv) > 1 else "ref.sass"
    text = open(path).read()
    per = census(text)
    print(f"== SASS census: {path}  ({len(per)} functions) ==")
    for name, c in per.items():
        total = sum(c.values())
        top = ", ".join(f"{k}={v}" for k, v in c.most_common(5))
        print(f"  {name:<44} {total:6d} instrs   {top}")

    print("\n== the two gate questions ==")
    f16 = next((n for n in per if "gemm_f16" in n), None)
    f32 = next((n for n in per if "gemm_f32" in n), None)
    if f16:
        n_hmma = sum(v for k, v in per[f16].items() if k == "HMMA")
        print(f"  fp16 GEMM HMMA count      : {n_hmma}   "
              f"{'PASS -- tensor cores engaged' if n_hmma > 0 else 'FAIL -- no tensor core'}")
    if f32:
        n_ffma = per[f32].get("FFMA", 0)
        n_hmma = sum(v for k, v in per[f32].items() if k == "HMMA")
        print(f"  fp32 GEMM FFMA={n_ffma} HMMA={n_hmma}   "
              f"{'PASS -- FFMA as expected for fp32 on sm_75' if n_hmma == 0 else 'UNEXPECTED -- fp32 used HMMA'}")

    bad = []
    for name, c in per.items():
        for il in ILLEGAL_SM75:
            if il in c:
                bad.append((name, il, c[il]))
    print(f"  sm_75-illegal instructions: {len(bad)}"
          + ("" if not bad else "  " + str(bad)))
    return 0 if (f16 and sum(v for k, v in per[f16].items() if k == "HMMA") > 0) and not bad else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
