#!/usr/bin/env python3
"""ablate.py -- the whole ladder, priced, in one command.

Runs, for every rung V0..V4:

  1. `verify_opt.py`'s identity check, so a rung that computes something else
     can never be priced (a rung that changed the answer would make every
     number attached to it meaningless, not merely wrong);
  2. the kernel census against `max_steps`, on the NULL device with the CUDA
     renderer attached, which is where the fusion decision is visible;
  3. the capture, the sm_75 compile, and the per-step counts -- graph ops,
     generated statements, and SASS instructions, all by the difference method
     in `opcount.py`.

The rung order is the point. Each rung is the one above plus exactly one
rewrite, so the difference between two consecutive rows is that rewrite's
price, causally, with nothing else in the program changing. The ladder is the
only honest way to say how much of a gap is the port's fault: not "the
optimized version is faster" but "rewriting the index arithmetic removed N
instructions per step, and here is the same program without it".

    python3 minecu/opt/ablate.py --batch 1024 --steps 4,8,16,32,64
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PY = sys.executable


def sh(cmd, **kw):
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return p.returncode, p.stdout, p.stderr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--world", type=int, default=16)
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--steps", default="4,8,16,32,64")
    ap.add_argument("--variants", default="naive,int_cursor,flat_gather,incremental,lean")
    ap.add_argument("--outdir", type=Path, default=REPO / "out" / "minecu.opt")
    ap.add_argument("--arch", default="sm_75")
    ap.add_argument("--skip-verify", action="store_true",
                    help="skip the identity gate. It exists so a rung that "
                         "computes something else can never be priced; skipping "
                         "it makes every downstream number meaningless.")
    ap.add_argument("--skip-sass", action="store_true")
    a = ap.parse_args()

    out = a.outdir / f"ablate-b{a.batch}"
    out.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print(f"  the ladder, priced. batch {a.batch}, world {a.world}, "
          f"res {a.res}, steps {a.steps}")
    print(f"  output: {out}")
    print("=" * 78)

    # ---- 1. the identity gate
    if not a.skip_verify:
        print("\n--- 1. identity: every rung must equal the naive port, bit for bit")
        rc, o, e = sh([PY, str(HERE / "verify_opt.py")], env={**__import__("os").environ})
        tail = [ln for ln in o.splitlines() if ln.strip()][-6:]
        print("\n".join("  " + ln for ln in tail))
        if rc != 0:
            print("  the identity gate FAILED; refusing to price anything.")
            return 1

    # ---- 2+3. capture, compile, count
    print("\n--- 2. the generated CUDA, on the NULL device with the CUDA renderer")
    rc, o, e = sh([PY, str(HERE / "capture_opt.py"), "--outdir", str(out),
                   "--batch", str(a.batch), "--world", str(a.world),
                   "--res", str(a.res), "--steps", a.steps,
                   "--variants", a.variants, "--arch", a.arch])
    print("\n".join("  " + ln for ln in o.splitlines()[-len(a.variants.split(",")) * len(
        a.steps.split(",")) - 4:]))
    if rc != 0:
        print("  capture FAILED")
        print(e[-2000:])
        return 1

    if not a.skip_sass:
        print("\n--- 3. sm_75 SASS, in the archlab container")
        biggest = max(int(x) for x in a.steps.split(","))
        src = out / f"max_steps_{biggest:04d}"
        rc, o, e = sh([str(HERE / "compile_sass.sh"), str(src), a.arch])
        print("\n".join("  " + ln for ln in o.splitlines()[-40:]))
        if rc != 0:
            print("  compile FAILED")
            print(e[-2000:])
            return 1

    # ---- 4. the counts
    print("\n--- 4. the counts")
    rc, o, e = sh([PY, str(HERE / "opcount.py"), "--captured", str(out),
                   "--json", str(out / "opcount.json")])
    print(o)
    if rc != 0:
        print(e[-2000:])
        return 1
    (out / "config.json").write_text(json.dumps(
        {"batch": a.batch, "world": a.world, "res": a.res, "steps": a.steps,
         "variants": a.variants, "arch": a.arch}, indent=2))
    print(f"  wrote {out/'opcount.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
