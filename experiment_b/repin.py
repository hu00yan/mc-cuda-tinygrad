#!/usr/bin/env python3
"""repin.py -- deliberately re-pin an artifact, and show what changed.

    python3 -m experiment_b.repin out/archlab/tri/<file>.ptx
    python3 -m experiment_b.repin --list

A pinned artifact is one this directory was written against. When one changes,
every number derived from it is stale, and `common.require` raises rather than
letting a stale figure be printed. That is the intended behaviour and it is not
something to work around by editing the hash in `common.PINS` -- because the
question the hash is asking is "did the thing I measured change?", and answering
it by hand is the failure mode the pin exists to prevent.

So repinning is a two-step operation on purpose. First look at what the old
pin was and what the new content is, then re-pin, then RE-DERIVE every number
that depended on it and check the new ones against the note. A pin that changes
without the note changing is a silent regression.

`--list` prints the current pins and their live status without changing
anything. That is the same table `run_all.py` prints.
"""

from __future__ import annotations

import sys

from experiment_b.common import PINS, _resolve, sha256


def main() -> int:
    if len(sys.argv) == 1 or sys.argv[1] == "--list":
        for rel, want in sorted(PINS.items()):
            p = _resolve(rel)
            if p is None:
                print(f"  MISSING  {rel}")
                continue
            got = sha256(p)
            mark = "ok " if got == want else "DRIFT"
            print(f"  {mark}  {rel}\n          pinned {want}\n          actual {got}")
        return 0

    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 -m experiment_b.repin <pinned-path> | --list")
    rel = sys.argv[1]
    if rel not in PINS:
        raise SystemExit(
            f"{rel} is not in PINS.\n"
            f"  If a new experiment in this directory reads it, add the entry with\n"
            f"  its sha256 first -- an unpinned read is the failure this module\n"
            f"  exists to prevent.")
    p = _resolve(rel)
    if p is None:
        raise SystemExit(f"cannot resolve {rel} on disk; nothing to repin.")
    old = PINS[rel]
    new = sha256(p)
    print(f"  path    {p}")
    print(f"  old pin {old}")
    print(f"  new     {new}")
    if old == new:
        print("  unchanged.")
        return 0
    print()
    print("  The pin MUST NOT be changed until you have:")
    print("    1. re-run the experiment that reads it")
    print("       (python3 experiment_b/run_all.py --only <exp>)")
    print("    2. diff the new numbers against notes/15-experiment-b.md and")
    print("       update the note for every figure that moved")
    print("    3. only then edit the entry in experiment_b/common.py PINS")
    print()
    print("  Not doing 1 and 2 leaves the note describing an artifact that no")
    print("  longer exists, which is how a re-derivation record stops being one.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
