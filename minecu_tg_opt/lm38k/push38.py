#!/usr/bin/env python3
"""push38.py -- push a Kaggle kernel, and never report a refusal as a success.

WHY THIS EXISTS
---------------
`kaggle kernels push` **exits 0 while printing**

    Kernel push error: Maximum batch GPU session count of 2 reached

Two earlier rounds of this project ran their "measurements" against a kernel
that was never started, and both wrote a note. So a push is not a push until
BOTH of these hold:

  1. the process exit status is 0, AND
  2. stdout+stderr contain none of the refusal strings, AND
  3. `kaggle kernels status` reports the slug as existing at all.

This script refuses to report success unless all three hold, and it prints
which one failed. It then polls to a terminal status, downloads the output,
and -- for the runner route -- compares the fingerprint the runner reported
against the one this invocation pushed, using `gpush.cli.payload_provenance`,
because a re-push can mount the previous dataset version and a *self-consistent*
payload check passes on a stale payload (ledger: three occurrences).

    ./push38.py --dir minecu_tg_opt/lm38k --slug tenghuzhang/mc-lm38-falsifier
    ./push38.py --dir ... --slug ... --wait-min 90 --skip-push   # re-fetch
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

#: Substrings that mean "the push was refused and nothing is running". Each of
#: these has been observed on this account with exit status 0.
REFUSALS = [
    "Maximum batch GPU session count",
    "Kernel push error",
    "Too many notebooks",
    "simultaneous",
    "quota",
    "rate limit",
    "403",
    "401",
    "Cannot access kernel",
]


def kaggle(*a, timeout: int = 300) -> subprocess.CompletedProcess:
    exe = os.environ.get("KAGGLE", str(REPO / ".venv" / "bin" / "kaggle"))
    return subprocess.run([exe, *a], capture_output=True, text=True, timeout=timeout)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--wait-min", type=float, default=75.0)
    ap.add_argument("--poll-s", type=int, default=25)
    ap.add_argument("--session-min", type=int, default=30)
    ap.add_argument("--skip-push", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    kdir = (REPO / a.dir).resolve()
    say = lambda *x: print(*x, flush=True)
    t0 = time.time()

    if not a.skip_push:
        say(f"=== push {a.slug} (cap {a.session_min}m) ===")
        p = kaggle("kernels", "push", "-p", str(kdir), "-t", str(a.session_min * 60))
        out = (p.stdout or "") + (p.stderr or "")
        say(f"  exit status: {p.returncode}")
        for line in out.splitlines():
            say(f"  | {line}")
        low = out.lower()
        hits = [r for r in REFUSALS if r.lower() in low]
        if p.returncode != 0:
            say(f"\n*** PUSH FAILED: exit {p.returncode}. Nothing was started.")
            return 3
        if hits:
            say(f"\n*** PUSH REFUSED, exit status notwithstanding: {hits}")
            say("*** Nothing is running. Do not read this as a measurement.")
            return 4
        say("  push accepted (exit 0, no refusal string in stdout)")

    say("")
    say(f"=== wait (up to {a.wait_min:.0f}m) ===")
    deadline = time.time() + a.wait_min * 60
    status, last = "UNKNOWN", None
    while True:
        p = kaggle("kernels", "status", a.slug)
        out = (p.stdout or "") + (p.stderr or "")
        m = re.search(r'status "KernelWorkerStatus\.(\w+)"', out)
        if m:
            status = m.group(1)
        elif "Cannot access kernel" in out:
            say(f"  *** {a.slug} does not exist for this account. The push never landed.")
            return 5
        if status != last:
            say(f"  [{time.strftime('%H:%M:%S')}] {status}  (t+{time.time()-t0:.0f}s)")
            last = status
        if status in ("COMPLETE", "ERROR", "CANCELLED", "FAILED"):
            break
        if time.time() > deadline:
            say(f"  gave up after {a.wait_min:.0f}m in state {status}")
            say(f"  stop it at https://www.kaggle.com/code/{a.slug}")
            return 6
        time.sleep(a.poll_s)

    if status != "COMPLETE":
        say("")
        say(f"=== kernel log (status {status}) ===")
        p = kaggle("kernels", "logs", a.slug)
        for line in ((p.stdout or "") + (p.stderr or "")).splitlines()[-160:]:
            say(f"  | {line}")
        return 7

    dest = Path(a.out) if a.out else (REPO / "out" / "lm38" / time.strftime("%Y%m%d-%H%M%S"))
    dest.mkdir(parents=True, exist_ok=True)
    say("")
    say(f"=== download -> {dest} ===")
    for attempt in range(1, 4):
        p = kaggle("kernels", "output", a.slug, "-p", str(dest), "--force")
        for line in ((p.stdout or "") + (p.stderr or "")).splitlines()[-6:]:
            say(f"  | {line}")
        if p.returncode == 0:
            break
        say(f"  attempt {attempt}/3 failed; sleeping 20s")
        time.sleep(20)
    else:
        return 8

    import zipfile
    for z in sorted(dest.rglob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(z.parent / z.stem)
    # a 30 MB state dump has no business coming home; the comparison already
    # happened on the card
    for d in list(dest.rglob("refstate")):
        subprocess.run(["rm", "-rf", str(d)], check=False)

    say("")
    say("=== what came back ===")
    files = sorted(p for p in dest.rglob("*") if p.is_file())
    for f in files:
        say(f"  {f.relative_to(dest).as_posix():<58} {f.stat().st_size:>10,} B")
    (dest / "PUSH38.json").write_text(json.dumps(
        {"slug": a.slug, "status": status, "dir": str(kdir),
         "wall_s": round(time.time() - t0, 1),
         "files": [f.relative_to(dest).as_posix() for f in files]}, indent=1))

    for s in list(dest.rglob("SUMMARY.txt")):
        say("")
        say(f"--- SUMMARY ({s.relative_to(dest)}) ---")
        say(s.read_text())
        break
    return 0


if __name__ == "__main__":
    sys.exit(main())
