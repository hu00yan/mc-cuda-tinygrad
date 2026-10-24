#!/usr/bin/env python3
"""
l4_shape_census.py -- does the Kaggle shape table have a live L4 row, and is the
client SDK's list of shapes the whole truth?

This is the instrument behind the reachability finding in `notes/40`. It costs
NO accelerator time: every call is a read, and none of them can start a session.

THE PROBLEM IT SOLVES
    There is no API that enumerates the accelerator options for an account. The
    `machine_shape` field is only written on a kernel that has already run, so
    the only way to read the server's shape table is to ask about kernels that
    have run on the shapes one wants to know about. That sounds circular and is
    not: the value in a kernel's metadata is written by the server, not by the
    uploader, so it is a *behavioural* read of the table rather than a claim.

    `kaggle kernels list` does not carry the field. `kaggle kernels pull -m`
    does.

WHY A CENSUS RATHER THAN A DOCSTRING
    The client SDK's own docstring for `machine_shape` (kagglesdk 2.2.4) says:

        Currently supported options:
           * NvidiaTeslaT4
           * NvidiaTeslaP100
           * Tpu1VmV38

    A first pass at this census over 200 recently-run public kernels found
    `NvidiaRtxPro6000` -- which appears in no docstring. Reading the docstring
    as the table would have concluded the platform has three shapes and that an
    L4 is not one of them. The census is what makes the claim, and the claim is
    the reason the request in the other half of this note was worth a session.

    A live row is not a granted card. This census establishes that L4 is a
    value the server hands out, and how recently; it cannot establish that this
    account may select one, because only a push can establish that and a push
    costs a session.

RATE LIMITS, because a rate-limited "None" is not a kernel that lacks a shape
    An earlier 10-wide parallel pass returned HTTP 429 on every request and was
    recorded as "no non-empty machine_shape" -- a null that looked exactly like
    a result. Serial pulls with a pause, and the failure is reported as a
    failure, not as an absence.

Usage:
    python kda_t4/l4_shape_census.py recent 40      # most recently run kernels
    python kda_t4/l4_shape_census.py hunt 30        # kernels whose text says L4
"""
from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
import tempfile
import time

KAGGLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      ".venv", "bin", "kaggle")
BASE = "https://www.kaggle.com"
HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.path.join(HERE, "out", "census")

# The queries worth asking. An L4 would be attached to somebody's notebook, and
# the words that reach a notebook are the words its author used, not the words
# the scheduler used.
HUNT_QUERIES = ["L4 GPU", "NVIDIA L4", "L4 accelerator", "GPU L4", "sm_89",
                "compute capability 8.9", "Ada Lovelace", "L4S", "NvidiaL4"]


def session():
    import requests
    S = requests.Session()
    S.headers.update({"Content-Type": "application/json",
                      "User-Agent": "kaggle-api/v1.7.0"})
    try:
        with open(os.path.expanduser("~/.kaggle/access_token")) as f:
            S.headers["Authorization"] = "Bearer " + f.read().strip()
    except OSError as e:
        print(f"  no access token at ~/.kaggle/access_token: {e}")
    return S


def kernels_list(page: int = 1, size: int = 100, search: str = ""):
    S = session()
    url = (f"{BASE}/api/v1/kernels/list?pageSize={size}&page={page}"
           f"&sortBy=dateRun&group=everyone")
    if search:
        url += "&search=" + search.replace(" ", "%20")
    r = S.get(url, timeout=40)
    if r.status_code != 200:
        return None, r.status_code
    try:
        data = r.json()
    except ValueError:
        return None, "unparseable"
    ks = data if isinstance(data, list) else (data or {}).get("kernels", [])
    return [k["ref"] for k in ks if k.get("ref")], 200


def last_run(ref: str):
    _, _, slug = ref.partition("/")
    for attempt in range(4):
        refs, code = kernels_list(search=slug)
        if code == 429:
            time.sleep(4 * (attempt + 1))
            continue
        if not refs:
            return f"<list {code}>"
        # The list endpoint does not carry machineShape, so the run time is
        # looked up by name and the shape separately, below.
        S = session()
        r = S.get(f"{BASE}/api/v1/kernels/list?pageSize=100&group=everyone"
                  f"&search={slug}", timeout=30)
        if r.status_code != 200:
            return f"<list {r.status_code}>"
        data = r.json()
        ks = data if isinstance(data, list) else (data or {}).get("kernels", [])
        for k in ks:
            if k.get("ref") == ref:
                return k.get("lastRunTime")
        return "<not in search results>"
    return "<rate limited>"


def pull_shape(ref: str, td: str, i: int):
    d = os.path.join(td, f"k{i:04d}")
    os.makedirs(d, exist_ok=True)
    p = subprocess.run([KAGGLE, "kernels", "pull", ref, "-m", "-p", d],
                       capture_output=True, text=True, timeout=180)
    mp = os.path.join(d, "kernel-metadata.json")
    if p.returncode != 0 or not os.path.exists(mp):
        return ref, None, f"pull rc={p.returncode} {(p.stderr or '')[:60].strip()}"
    try:
        m = json.load(open(mp))
    except ValueError as e:
        return ref, None, f"unparseable metadata: {e}"
    return ref, m.get("machine_shape"), ""


def survey(refs, label, with_time=False):
    os.makedirs(SCRATCH, exist_ok=True)
    hist: collections.Counter = collections.Counter()
    by_shape: dict[str, list[str]] = collections.defaultdict(list)
    errors: collections.Counter = collections.Counter()
    print(f"\n=== {label}: {len(refs)} kernels, serial pulls (rate-limit safe) ===")
    with tempfile.TemporaryDirectory(dir=SCRATCH) as td:
        for i, ref in enumerate(refs):
            r, ms, err = pull_shape(ref, td, i)
            hist[ms] += 1
            if err:
                errors[err.split(" ")[0] + " " + err.split(" ")[1]
                       if len(err.split(" ")) > 1 else err] += 1
            if ms:
                by_shape[ms].append(ref)
            time.sleep(0.4)
    print("  machine_shape -> count")
    for ms, n in hist.most_common():
        print(f"    {str(ms):24s} {n}")
    if errors:
        print(f"  pull failures: {dict(errors)}")
    print("  DISTINCT SERVER-REPORTED SHAPES:")
    for ms, lst in sorted(by_shape.items(), key=lambda kv: -len(kv[1])):
        print(f"    {ms:24s} n={len(lst):4d}  e.g. {lst[:3]}")
    if with_time:
        print("  when the non-T4/non-CPU shapes last ran:")
        for ms, lst in sorted(by_shape.items()):
            if ms in (None, "NvidiaTeslaT4"):
                continue
            for ref in sorted(lst)[:12]:
                print(f"    {str(last_run(ref)):32s} {ms:20s} {ref}")
                time.sleep(0.6)
    return hist, by_shape, errors


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "hunt"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    print("l4_shape_census -- READ ONLY, spends no accelerator quota")
    print(f"  kaggle CLI: {KAGGLE}")

    if mode == "recent":
        refs, code = kernels_list()
        if not refs:
            print(f"  list failed: {code}")
            return 1
        survey(refs[:n], "most recently run public kernels", with_time=True)
        return 0

    if mode == "hunt":
        seen: dict[str, str] = {}
        for q in HUNT_QUERIES:
            refs, code = kernels_list(search=q)
            if not refs:
                print(f"  query {q!r}: list failed ({code})")
                continue
            new = [r for r in refs[:n] if r not in seen]
            for r in new:
                seen[r] = q
            print(f"  query {q!r:28s} -> {len(refs):3d} kernels, {len(new):3d} new")
        hist, by_shape, _ = survey(sorted(seen), "kernels whose text names an L4",
                                   with_time=True)
        os.makedirs(SCRATCH, exist_ok=True)
        with open(os.path.join(SCRATCH, f"hunt-{n}.json"), "w") as f:
            json.dump({k: v for k, v in by_shape.items()}, f, indent=1)
        print(f"\n  wrote {SCRATCH}/hunt-{n}.json")
        return 0

    print(f"  unknown mode {mode!r}: use 'recent' or 'hunt'")
    return 64


if __name__ == "__main__":
    sys.exit(main())
