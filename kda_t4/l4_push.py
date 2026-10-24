#!/usr/bin/env python3
"""
l4_push.py -- ask Kaggle for an L4 and find out whether one is reachable.

WHY THIS EXISTS INSTEAD OF `gpush kaggle run`
    `gpush/backends/kaggle.py:407-419` writes its own kernel-metadata.json and
    has no field for a machine shape, so gpush can only ever ask for "a GPU",
    which on this account is a T4 x2.  `gpush/**` is not mine to change, so the
    machine_shape has to be set by something else.  This is that something else,
    and it borrows every part of gpush that is worth borrowing -- the manifest,
    the dataset upload with its version-READY gate, the generated runner
    notebook, and `payload_provenance` -- rather than reimplementing a second
    transport whose bugs would be its own.

THE THING THIS SCRIPT EXISTS TO GET RIGHT
    `kaggle kernels push` prints

        Kernel push error: Maximum batch GPU session count of 2 reached

    and EXITS 0.  A helper that checks the exit status believes that push
    worked, then downloads whatever the last run on that slug left behind, and
    reports it as this run's numbers.  That is not hypothetical in this project:
    two earlier rounds measured stale output that way.  So:

      * the exit status is checked;
      * stdout AND stderr are grepped for a refusal, and the grep result is
        printed next to the exit status so the two cannot be read as one thing;
      * the fingerprint the runner reports is compared against the fingerprint
        this invocation pushed, via gpush's own `payload_provenance`, and the
        two verdicts (self-consistent / stale) are reported separately;
      * a kernel's `machine_shape` after the run is reported, because the only
        way to know whether an L4 was actually granted is to read back what the
        server recorded.

USAGE
    python kda_t4/l4_push.py --shape NvidiaL4 --cmd bash kda_t4/l4_run.sh
    python kda_t4/l4_push.py --shape NvidiaL4 --dry-run
    python kda_t4/l4_push.py --shape NvidiaL4 --status-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gpush import backend as be                                    # noqa: E402
from gpush import util                                            # noqa: E402
from gpush.backends.runner_nb import (build_runner_notebook,      # noqa: E402
                                      validate)
from gpush.cli import payload_provenance                          # noqa: E402
from gpush.manifest import build                                  # noqa: E402
from gpush.state import StateStore                                # noqa: E402

KAGGLE = os.path.join(REPO, ".venv", "bin", "kaggle")
OUT_ROOT = os.path.join(REPO, "out", "gpush")

# Every phrase this project has seen Kaggle use to refuse a push.  The list is
# deliberately generous: a refusal we do not recognise is still a refusal we
# must not mistake for a success, so the check is "did it say no", and anything
# unrecognised falls through to the fingerprint comparison rather than to trust.
REFUSAL_MARKERS = (
    "maximum batch gpu session count",
    "maximum batch cpu session count",
    "maximum interactive gpu session count",
    "session count of",
    "kernel push error",
    "quota exceeded",
    "not enough accelerator",
    "accelerator is not available",
    "invalid accelerator",
    "invalid machine_shape",
    "unknown machine shape",
    "you have reached",
    "try again later",
    "resource exhausted",
    "too many requests",
)


def kaggle(*args: str, timeout: int = 300) -> util.Proc:
    return util.run([KAGGLE, *args], timeout=timeout)


def refusal_scan(text: str) -> list[str]:
    low = (text or "").lower()
    return [m for m in REFUSAL_MARKERS if m in low]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tenghuzhang/l4-probe-payload")
    ap.add_argument("--kernel", default="tenghuzhang/l4-probe")
    ap.add_argument("--shape", default="NvidiaL4",
                    help="the machine_shape to request")
    ap.add_argument("--include", action="append", default=None)
    ap.add_argument("--cmd", nargs=argparse.REMAINDER, default=[])
    ap.add_argument("--session", type=int, default=25,
                    help="session cap in minutes, enforced server-side by -t")
    ap.add_argument("--queue-wait", type=int, default=60)
    ap.add_argument("--poll", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status-only", action="store_true")
    ap.add_argument("--docker-image", default="",
                    help="pin the container image, e.g. the CUDA byod image a "
                         "T4 request is given. A shape maps to an image on the "
                         "server, and for an L4 request the image it chose had "
                         "no CUDA in it, which is the mechanism behind the "
                         "silent CPU downgrade. Pinning it is the one lever "
                         "left that could turn 'not granted' into 'granted'.")
    ap.add_argument("--no-push", action="store_true",
                    help="upload the payload and build the notebook, do not push")
    args = ap.parse_args()

    cmd = [c for c in args.cmd if c != "--"]
    include = args.include or ["kda_t4"]
    out = os.path.join(OUT_ROOT, "l4")
    os.makedirs(out, exist_ok=True)
    # Every push gets its own run directory.
    #
    # This was a real defect, found by being bitten: the log and the fetched
    # artifacts were written to one fixed path, so a second push overwrote the
    # first push's evidence.  The T4 control's section-1 output was destroyed by
    # the L4 attempt that followed it, which is the worst possible ordering --
    # the control is the thing that makes the negative result attributable, and
    # it is the thing an overwrite deletes.  A harness that cannot hold two
    # runs' evidence at once cannot compare two runs, and this note's whole
    # claim is a comparison of two runs.
    run_stamp = util.stamp()
    safe_shape = "".join(c if c.isalnum() else "_" for c in args.shape)
    out = os.path.join(out, "runs", f"{run_stamp}-{safe_shape}")
    os.makedirs(out, exist_ok=True)
    args.out = out
    print(f"  run dir      {out}")

    print("=" * 78)
    print("l4_push -- asking Kaggle for machine_shape:", args.shape)
    print("=" * 78)

    if args.status_only:
        return show_status(args.kernel, out)

    store = StateStore(OUT_ROOT).bind("kaggle", args.dataset)
    bk = be.get("kaggle", dataset=args.dataset, kernel=args.kernel,
                kaggle=KAGGLE, store=store, out_root=out,
                session_min=args.session, queue_wait_min=args.queue_wait,
                poll_secs=args.poll, include=include, command=cmd,
                gpu=True, tpu=False, allow_cpu=False, skip_verify=False)

    # ---------------------------------------------------------- the payload
    tree = build(REPO, store.manifest() if store.fingerprint else None,
                 include=include)
    # The fingerprint is NOT set by build(); gpush's own CLI calls this
    # explicitly (gpush/cli.py:683).  Leaving it empty is not cosmetic:
    # `payload_provenance` computes `stale = bool(pushed_fp and ran and ...)`,
    # so an empty pushed fingerprint makes `stale` permanently False and the
    # STALE PAYLOAD check silently stops working.  That is the exact defect this
    # push is supposed to be immune to, introduced by forgetting one call.
    tree.compute_fingerprint("kaggle", args.dataset)
    if not tree.fingerprint:
        print("  REFUSED: the manifest has no fingerprint. Refusing to push -- an "
              "empty fingerprint disables the staleness check.")
        return 2
    print(f"  payload      {len(tree)} files, "
          f"{tree.total_bytes/1024:.1f} KB, fingerprint {tree.fingerprint}")
    print(f"  dataset      {args.dataset}")
    print(f"  kernel       {args.kernel}")
    print(f"  shape        {args.shape}")
    print(f"  command      {' '.join(cmd) or '(none)'}")
    plan = bk.plan(tree)
    print(f"  change       {plan.summary()}  ({plan.bytes_in/1024:.1f} KB)")
    if plan.uncovered:
        print(f"  uncovered    {len(plan.uncovered)} working-tree files not in the payload")

    rc, sync = do_push(bk, tree, args, include)
    if rc != 0:
        print("\n  PAYLOAD NOT PUBLISHED. No session was started; stop here.")
        return rc
    version = dataset_version(args.dataset)
    if version is None:
        print("  dataset version: UNKNOWN. The kernel will mount whatever is newest")
        print("  when it starts, which is a race with the dataset, not an address.")
        print("  The runner still re-hashes the tree against the manifest baked into")
        print("  the notebook, so a stale mount is caught -- but the source is not")
        print("  pinned, and that is a weaker guarantee and is reported as one.")
    else:
        print(f"  dataset version published: {version} (pinned in dataset_sources)")

    # ----------------------------------------------------- the kernel itself
    nb = build_runner_notebook(tree, args.dataset, cmd, args.kernel.partition("/")[2])
    problems = validate(nb)
    if problems:
        print("\n  REFUSED: the generated notebook does not parse:")
        for p in problems:
            print("    " + p)
        print("  No session was started.")
        return 2

    # The kernel directory is deliberately NOT under the run directory: it holds
    # the notebook that is about to be pushed, and the runner's own provenance
    # check re-hashes the payload, not this.  Keeping it per-run would leave the
    # pushed artefact and the record of pushing it in different trees.
    kdir = os.path.join(OUT_ROOT, "l4", "kernel", args.kernel.partition("/")[2])
    os.makedirs(kdir, exist_ok=True)
    with open(os.path.join(kdir, "runner.ipynb"), "w") as f:
        json.dump(nb, f, indent=1)
    meta = {
        "id": args.kernel,
        "title": args.kernel.partition("/")[2].replace("-", " "),
        "code_file": "runner.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        # The field under test.  Pinned to the shape, not left to the default,
        # because the default is the T4 this whole study is trying to escape.
        "machine_shape": args.shape,
        "dataset_sources": [f"{args.dataset}/{version}" if version else args.dataset],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }
    if args.docker_image:
        meta["docker_image"] = args.docker_image
    with open(os.path.join(kdir, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"  notebook     {len(nb['cells'])} cells, validates")
    print(f"  metadata     machine_shape={meta['machine_shape']!r} "
          f"docker_image={meta.get('docker_image', '<server default>')}")
    print(f"               dataset_sources={meta['dataset_sources']}")
    print(f"  kernel dir   {kdir}")

    if args.dry_run:
        print("\n  --dry-run: nothing pushed. No session was started.")
        return 0
    if args.no_push:
        print("\n  --no-push: payload published, kernel not pushed. "
              "No session was started.")
        return 0

    # ------------------------------------------------------------- THE PUSH
    print()
    print("  " + util.yellow("THE NEXT LINE SPENDS GPU QUOTA"))
    session_s = args.session * 60
    p = kaggle("kernels", "push", "-p", kdir, "-t", str(session_s),
               "--accelerator", args.shape, timeout=900)

    stdout = p.out or ""
    stderr = p.err or ""
    markers = refusal_scan(stdout) + refusal_scan(stderr)
    print()
    print("  " + "-" * 72)
    print(f"  push exit status        {p.rc}")
    print(f"  refusal markers in out  {refusal_scan(stdout)}")
    print(f"  refusal markers in err  {refusal_scan(stderr)}")
    tail = (stdout.strip() or stderr.strip()).splitlines()[-12:]
    for line in tail:
        print(f"    | {line}")
    print("  " + "-" * 72)

    record = {
        "requested_shape": args.shape,
        "pushed_fingerprint": tree.fingerprint,
        "dataset": args.dataset, "kernel": args.kernel,
        "dataset_version": version, "command": cmd,
        "push_rc": p.rc, "push_stdout": stdout, "push_stderr": stderr,
        "refusal_markers": markers,
        "started": util.stamp(),
    }
    with open(os.path.join(out, "push_record.json"), "w") as f:
        json.dump(record, f, indent=1)

    if p.rc != 0 or markers:
        # A refusal is a result. It is recorded as one, and no session is
        # waited for, because there is no session to wait for.
        why = (f"the push was refused (exit {p.rc}"
               + (f", markers {sorted(set(markers))}" if markers else "") + ")")
        record["verdict"] = "REFUSED"
        record["why"] = why
        with open(os.path.join(out, "push_record.json"), "w") as f:
            json.dump(record, f, indent=1)
        print()
        print("  VERDICT: REFUSED. " + why)
        print("  This is the answer to 'is the L4 reachable' when the platform")
        print("  will not grant one. Quota spent: whatever the server charged,")
        print("  which for a refused push is nothing.")
        return 3

    # --------------------------------------------------------- wait for it
    deadline = time.time() + args.queue_wait * 60
    last = ""
    status = "PUSHED"
    while True:
        s = bk.status()
        if s.phase != last:
            status = s.phase
            print(f"  {util.stamp()}  status: {s.phase}")
            last = s.phase
        if s.phase in ("COMPLETE", "ERROR", "CANCELLED", "FAILED", "NOT FOUND"):
            status = s.phase
            break
        if time.time() >= deadline:
            status = status or "QUEUED"
            print(f"  gave up waiting after {args.queue_wait} min; status was {status}")
            break
        time.sleep(args.poll)

    logs = kaggle("kernels", "logs", args.kernel, timeout=300)
    from gpush.backends.kaggle import kernel_log_text
    text = kernel_log_text(logs.out or logs.err or "")
    lp = os.path.join(out, f"{args.kernel.partition('/')[2]}.log")
    with open(lp, "w") as f:
        f.write(text)
    print(f"  log {len(text)} bytes -> {lp}")
    if status in ("CANCELLED", "CANCELLED", "ERROR", "FAILED"):
        for pat in ("CellTimeoutError", "CancelledError", "Timeout", "quota",
                    "Session stopped", "KeyboardInterrupt"):
            if pat.lower() in text.lower():
                print(f"  the log mentions {pat!r}; that is the cause, read there "
                      f"and not inferred from the status")

    # ------------------------------------------------- did OUR payload run?
    verify = {}
    dest = os.path.join(out, "artifacts")
    os.makedirs(dest, exist_ok=True)
    try:
        bk.fetch({}, dest)
    except BaseException as e:                                    # noqa: BLE001
        print(f"  fetch failed: {type(e).__name__}: {e}")
    for root, _d, files in os.walk(dest):
        for fn in files:
            if fn == "GPUSH-VERIFY.json":
                try:
                    verify = json.load(open(os.path.join(root, fn)))
                except ValueError:
                    pass
    pv = payload_provenance(verify, tree.fingerprint)
    record["verify"] = verify
    record["provenance"] = {"matched": pv.matched, "stale": pv.stale,
                            "pushed": pv.pushed, "ran": pv.ran}
    record["verdict"] = status
    record["finished"] = util.stamp()
    with open(os.path.join(out, "push_record.json"), "w") as f:
        json.dump(record, f, indent=1)

    print()
    print("  PROVENANCE")
    print(f"    runner self-check   {'MATCHED' if pv.matched else 'DID NOT MATCH'}")
    print(f"    identity            {pv.verdict_text()}")
    print(f"      this run pushed   {pv.pushed}")
    print(f"      the runner read   {pv.ran}")
    print(f"    quotable            {pv.quotable}")

    # ------------------------------------ what card did the server actually use?
    after = kernel_shape(args.kernel)
    record["machine_shape_after"] = after
    with open(os.path.join(out, "push_record.json"), "w") as f:
        json.dump(record, f, indent=1)
    print()
    print("  WHAT THE SERVER RECORDED")
    print(f"    machine_shape now    {after!r}")
    print(f"    requested            {args.shape!r}")
    if after == args.shape:
        print("    -> the server granted the shape that was asked for.")
    elif after:
        print(f"    -> the server did NOT grant {args.shape!r}; it ran on {after!r}.")
        print("       A silent fallback is the failure this check exists for.")
    else:
        print("    -> unreadable; treat the card as UNKNOWN, not as a T4.")
    return 0


def do_push(bk, tree, args, include):
    """Upload the payload through gpush, so its gates apply."""
    plan = bk.plan(tree)
    sync = bk.push(tree, plan)
    for name, ok, note in (sync.gates or []):
        print(f"  gate  {name:22s} {'ok ' if ok else 'FAIL'}  {note}")
    if not sync.ok:
        print("\n  a backend gate refused the payload; no session was started.")
        return 3, sync
    return 0, sync


def dataset_version(slug: str):
    """The dataset's current version number, read from the API.

    Used to pin `dataset_sources` to an exact version.  Returns None rather
    than a guess when the API does not say, because "I could not read the
    version" and "the version is 3" are very different statements.
    """
    import requests
    owner, _, name = slug.partition("/")
    S = requests.Session()
    S.headers.update({"Content-Type": "application/json",
                      "User-Agent": "kaggle-api/v1.7.0"})
    try:
        with open(os.path.expanduser("~/.kaggle/access_token")) as f:
            S.headers["Authorization"] = "Bearer " + f.read().strip()
    except OSError:
        return None
    try:
        r = S.get(f"https://www.kaggle.com/api/v1/datasets/view/{owner}/{name}",
                  timeout=30)
        if r.status_code != 200:
            return None
        d = r.json()
        for k in ("currentVersionNumber", "currentVersion", "versionNumber"):
            if isinstance(d, dict) and k in d:
                return int(d[k])
    except Exception:                                            # noqa: BLE001
        return None
    return None


def show_status(kernel: str, out: str) -> int:
    p = kaggle("kernels", "status", kernel, timeout=120)
    print(f"  {kernel}: rc={p.rc}  {(p.out or p.err or '').strip()[:300]}")
    print(f"  machine_shape: {kernel_shape(kernel)!r}")
    d = os.path.join(out, "push_record.json")
    if os.path.exists(d):
        rec = json.load(open(d))
        print(f"  last push record: verdict={rec.get('verdict')!r} "
              f"shape_requested={rec.get('requested_shape')!r} "
              f"provenance={rec.get('provenance')}")
    return 0


def kernel_shape(kernel: str) -> str | None:
    """Read back the machine_shape the server recorded for this kernel.

    This is the only trustworthy answer to "did I get an L4": the request is
    what was asked for, and this is what was granted.  A run whose card cannot
    be identified is a run whose numbers are about an unknown card.
    """
    import tempfile
    with tempfile.TemporaryDirectory(dir=os.path.join(OUT_ROOT, "l4")
                                     if os.path.isdir(os.path.join(OUT_ROOT, "l4"))
                                     else None) as td:
        p = kaggle("kernels", "pull", kernel, "-m", "-p", td, timeout=180)
        mp = os.path.join(td, "kernel-metadata.json")
        if p.rc == 0 and os.path.exists(mp):
            try:
                return json.load(open(mp)).get("machine_shape")
            except ValueError:
                return None
    return None


if __name__ == "__main__":
    sys.exit(main())
