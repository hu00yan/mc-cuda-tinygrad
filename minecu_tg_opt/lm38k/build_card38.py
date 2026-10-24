#!/usr/bin/env python3
"""build_card38.py -- assemble the card script with its payload embedded.

Every source the script runs is embedded as base64 and carries its sha256, and
the script refuses to start if any of them disagrees. A notebook that ran
yesterday's `render.py` against today's note is how two earlier rounds of this
project produced measurements of code that had already been fixed, so the check
is a hard assert and not a log line.

    python3 minecu_tg_opt/lm38k/build_card38.py --out minecu_tg_opt/lm38k/card38_run.py
    python3 minecu_tg_opt/lm38k/build_card38.py --check     # verify, do not write
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

#: The minimum needed to build, gate and time the DDA render on a real card.
#: `minecu_ref/src/kernels.cu` is hash-pinned and read-only; it is carried only
#: so the card can hash-check it and `refdrv.cu` can compile against it.
#: Nothing here writes to it.
PAYLOAD_PATHS = [
    "minecu_tg/__init__.py",
    "minecu_tg/blocks.py",
    "minecu_tg/render.py",
    "minecu_tg/bench.py",
    "minecu_tg/world.py",
    "minecu_tg/kaggle/refdrv.cu",
    "minecu_tg_opt/__init__.py",
    "minecu_tg_opt/render.py",
    "minecu_tg_opt/lm38_falsify.py",
    "minecu_ref/src/kernels.cu",
]
#: The hash `notes/35` and `notes/36` were written against. Checked on the card.
KERNELS_CU_SHA = "c6ed8f06ee64d40a8a8ab7ebce3b3ceea1e3ac93f9ee50cfc1e63d3f090d878f"

CARD = HERE / "card38.py"


def payloads(repo: Path) -> list:
    rows = []
    for rel in PAYLOAD_PATHS:
        p = repo / rel
        if not p.is_file():
            raise SystemExit(f"payload missing: {p}")
        data = p.read_bytes()
        h = hashlib.sha256(data).hexdigest()
        if rel.endswith("minecu_ref/src/kernels.cu") and h != KERNELS_CU_SHA:
            raise SystemExit(
                f"minecu_ref/src/kernels.cu sha256 is {h}, not the pinned\n"
                f"  {KERNELS_CU_SHA}\n"
                f"That directory is read-only and hash-pinned. Refusing to run "
                f"the experiment against a reference nobody can reproduce.")
        rows.append([rel, base64.b64encode(data).decode(), h])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--out", type=Path, default=HERE / "card38_run.py")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    rows = payloads(a.repo)
    env = json.dumps({"payloads": rows})
    b64 = base64.b64encode(env.encode()).decode()
    card = CARD.read_text()
    out = card.replace('PAYLOADS = json.loads(base64.b64decode(os.environ["LM38_PAYLOADS"]))',
                      f"PAYLOADS = json.loads(base64.b64decode(\n"
                      f"    '{b64}'))")
    if out == card:
        raise SystemExit("the substitution anchor was not found in card38.py -- "
                         "the assembler and the script have drifted apart")
    compile(out, str(a.out), "exec")
    print(f"card script: {len(out):,} chars, {len(rows)} payloads, "
          f"compiles clean")
    for rel, _, h in rows:
        print(f"  {rel:<38} {h[:16]}")
    if a.check:
        return 0
    a.out.write_text(out)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
