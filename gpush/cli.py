#!/usr/bin/env python3
"""cli.py -- `gpush <backend> <subcommand>`.

WHAT THE SIX SUBCOMMANDS ARE, AND WHAT EACH COSTS

    plan     offline diff of the working tree against the durable baseline.
             ZERO network calls, ZERO credentials. Costs nothing.
    doctor   can this backend be used right now, with what, at what cost.
             Some backends make a network call to answer honestly; it is
             printed when it happens. Costs no accelerator time.
    push     make the remote match the tree. Uploads bytes and spends an API
             call. Costs NO accelerator time on either backend.
    run      push if needed, then execute ONE command on the remote device.
             THIS IS THE ONE THAT SPENDS GPU QUOTA. On Kaggle it is a session
             against a weekly budget, capped server-side by -t; on Colab it is
             the session itself, which may also be reclaimed mid-command.
    status   what the remote is doing right now. Read-only. Costs nothing.
    fetch    bring one run's artefacts home, next to its provenance record.
             Costs nothing.

    The cost distinction is printed before every command that could spend
    anything, because "did that command cost me GPU hours" should never be a
    question about reading the source.

THE ORDER OF OPERATIONS, AND WHY IT IS NOT FLAT

    run = plan -> guards -> [no-op check] -> push -> backend gates -> execute
          -> fetch -> provenance -> promote the baseline

    The two guards and the no-op check happen before push, because they are
    free and offline. The backend gates happen after push and before execute,
    because they are the ones that need the network and because "the payload is
    on Kaggle" is not "the payload is on the runner". The baseline is promoted
    last of all, by the run's completion, and only that promotion makes a
    future invocation a no-op.

    That last line is the whole design. See notes/22-gpush.md.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

from . import __version__, backend as be, gates, provenance, util
from .manifest import Manifest, build
from .probes import capability_note, describe
from .state import StateStore

DEFAULT_OUT = os.path.join("out", "gpush")


# --------------------------------------------------------------------- setup

@dataclass
class PayloadProvenance:
    """Whether the bytes that ran are the bytes THIS invocation pushed.

    The runner's verdict is **self-consistency, not identity**. It says the
    payload on the far side matches the manifest carried *inside that same
    payload*. A cached notebook carries the previous run's `PAYLOAD_FINGERPRINT`
    and the previous run's manifest, so the two agree with each other and the
    verdict is `MATCHED` while the previous run's bytes execute.

    This is the DECISIONS.md #18 shape reached by a different route, and it was
    hit for real: a 9-second `COMPLETE` whose `GPUSH-VERIFY.json` reported the
    prior fingerprint (`out/gpush/runs/kaggle/ut-time-t4/`). The run of record was
    self-consistent at every level and still void -- which is exactly why a
    self-consistent check is the wrong check.

    So identity is decided by comparing the fingerprint the runner reports
    against the one this invocation pushed. A stale payload is not a *mismatch*:
    the payload matched its own manifest perfectly. It is a different run's
    payload, and the two facts must be reported as two different verdicts, or a
    reader sees `MATCHED` and stops looking.
    """
    matched: bool      # the payload matched the manifest inside itself
    stale: bool        # ...but it was not this run's payload
    pushed: str | None
    ran: str | None

    def verdict_text(self) -> str:
        if self.stale:
            return util.red("STALE PAYLOAD")
        if not self.established:
            return util.yellow("IDENTITY UNESTABLISHED")
        return util.green("MATCHED") if self.matched else util.red("DID NOT MATCH")

    @property
    def established(self) -> bool:
        """Do we have both fingerprints, so identity is a *conclusion*?

        The distinction this guards is the one that bit this project three times:
        a run that carries a payload nobody can identify is not a run whose
        payload is known-good. `stale` deliberately stays `False` when a
        fingerprint is missing, because absence is not evidence of staleness —
        but `quotable` used to read that `False` as a pass, so **omitting the
        fingerprint silently disabled the entire staleness check** and the run was
        quotable. Observed in the wild: a lane that forgot to pass
        `compute_fingerprint` got `quotable == True` from a comparison that never
        happened.

        So absence has to be its own verdict, not a quiet pass.
        """
        return bool(self.pushed) and bool(self.ran)

    @property
    def quotable(self) -> bool:
        """May any number from this run be quoted?

        Requires `established`, not merely `not stale`. `not stale` is what an
        absent fingerprint looks like, so the two are not interchangeable and
        conflating them turns a missing input into a positive result.
        """
        return self.matched and not self.stale and self.established


def payload_provenance(verify, pushed_fp) -> PayloadProvenance:
    """Pure decision, no I/O, so it can be tested directly.

    `stale` is deliberately NOT inferred from `matched`. A stale payload passes
    the runner's own check by construction, so folding the two together would
    make the one failure that matters invisible.
    """
    matched = bool(verify.get("ok"))
    ran = verify.get("fingerprint")
    # Identity is only claimed when BOTH sides are present. An absent fingerprint
    # on either side leaves identity unestablished, which is not the same as
    # established-and-equal -- but it is also not evidence of staleness, so it is
    # reported rather than guessed at.
    stale = bool(pushed_fp and ran and ran != pushed_fp)
    return PayloadProvenance(matched=matched, stale=stale,
                             pushed=pushed_fp, ran=ran)


def repo_root() -> str:
    return os.path.abspath(os.getcwd())


def add_backend_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--dataset", default="",
                    help="kaggle: owner/slug of the payload dataset")
    ap.add_argument("--kernel", default="",
                    help="kaggle: owner/slug of the kernel that runs the command")
    ap.add_argument("--host", default=os.environ.get("GPUSH_COLAB_HOST", ""),
                    help="colab: tunnel hostname, e.g. xxxx.trycloudflare.com. "
                         "Changes every session, so it is per-invocation.")
    ap.add_argument("--user", default=os.environ.get("GPUSH_COLAB_USER", "root"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("GPUSH_COLAB_PORT", "22")))
    ap.add_argument("--password", default=os.environ.get("GPUSH_COLAB_PASSWORD", ""),
                    help="colab: tunnel password. remote/colab_ssh.py sets 'probe'.")
    ap.add_argument("--askpass-cmd", default=os.environ.get("GPUSH_COLAB_ASKPASS", ""),
                    help="colab: command that prints the password, if you would "
                         "rather not pass it as a literal")
    ap.add_argument("--identity", default=os.environ.get("GPUSH_COLAB_IDENTITY", ""),
                    help="colab: ssh private key, if you use one")


def make_backend(name: str, args, store: StateStore, out: str):
    # `--cmd` only exists on `run`; everything else gets an empty command,
    # which is correct: a plan or a push does not run anything, and the
    # attestation records the command the run will actually use.
    cmd = getattr(args, "cmd", None) or []
    common = {
        "store": store, "out_root": out,
        "include": args.include or None,
        "command": cmd,
        "gpu": not args.tpu,
        "tpu": args.tpu,
        "allow_cpu": args.allow_cpu,
        "skip_verify": args.skip_verify,
    }
    if name == "kaggle":
        return be.get("kaggle", dataset=args.dataset, kernel=args.kernel,
                      kaggle=args.kaggle_cli, session_min=args.session,
                      queue_wait_min=args.queue_wait, poll_secs=args.poll,
                      **common)
    if name == "colab":
        return be.get("colab", host=args.host, user=args.user, port=args.port,
                      password=args.password or None,
                      askpass_cmd=args.askpass_cmd or None,
                      identity=args.identity or None,
                      remote_dir=args.remote_dir, timeout=args.timeout, **common)
    raise SystemExit(f"no such backend: {name}")


# ------------------------------------------------------------------- reports

def show_plan(bk, tree: Manifest, plan, store: StateStore, *, header: str) -> None:
    print(util.hr(header))
    print(f"  tree         {len(tree)} files, {tree.total_bytes/1024:.1f} KB "
          f"(from {tree.source})")
    print(f"  fingerprint  {tree.fingerprint}")
    print(f"  baseline     {store.describe()}")
    print(f"  change       {plan.summary()}   ({plan.bytes_in/1024:.1f} KB to send)")
    print()
    rows = []
    for kind, paths in (("added", plan.added), ("modified", plan.modified),
                        ("deleted", plan.deleted)):
        for p in paths[:12]:
            rows.append((kind, p))
    if len(plan.added) + len(plan.modified) + len(plan.deleted) > 36:
        n = len(plan.added) + len(plan.modified) + len(plan.deleted) - 36
        rows.append(("...", f"and {n} more"))
    if rows:
        print(util.table(rows, ["change", "path"], widths=[9, 62]))
    else:
        print(util.green("  no changes: the working tree matches the baseline"))
    if plan.uncovered:
        print()
        print(util.yellow(f"  {len(plan.uncovered)} working-tree files are NOT in the "
                           f"payload and cannot be measured remotely:"))
        for p in plan.uncovered[:8]:
            print(f"      {p}")
        if len(plan.uncovered) > 8:
            print(f"      ... and {len(plan.uncovered) - 8} more")


# ------------------------------------------------------------------ commands

def cmd_plan(args, bk, tree, out) -> int:
    """Offline by construction: it never receives a network-capable call.

    Everything below reads the working tree and one local JSON file. That is
    the point of the subcommand: the diff you review before spending anything
    must be the diff that would execute, and it must be reviewable with no
    token on the machine.
    """
    plan = bk.plan(tree)
    show_plan(bk, tree, plan, bk.store, header="plan (offline: no network, no credentials)")
    print()
    print(f"  would send   {plan.change_count} file(s), skip {len(plan.unchanged)}")
    print(f"  would delete {len(plan.deleted)} remote file(s) "
          f"({plan.delete_fraction*100:.1f}% of the remote manifest)")
    print(f"  would skip the transfer entirely if the baseline were proven "
          f"({util.green('it is') if bk.store.is_proven(tree.fingerprint) else util.yellow('it is not')})")
    if args.json:
        print(util.jdump({"fingerprint": tree.fingerprint,
                          "files": len(tree), "bytes": tree.total_bytes,
                          "added": plan.added, "modified": plan.modified,
                          "deleted": plan.deleted, "unchanged": len(plan.unchanged),
                          "base": plan.base_fingerprint,
                          "proven": bk.store.is_proven(tree.fingerprint),
                          "baseline_state": bk.store.state}))
    return 0


def cmd_doctor(args, bk, tree, out) -> int:
    rep = bk.doctor()
    print(util.hr(f"doctor: {bk.name}"))
    print(f"  {bk.transport}")
    print()
    for name, ok, detail in rep.checks:
        print(f"  [{'ok  ' if ok else 'FAIL'}] {name:<22} {detail}")
    print()
    if rep.note:
        print(util.wrap(rep.note))
        print()
    print(util.bold("cost:"))
    print(f"  plan / status / fetch    nothing")
    print(f"  push                     bytes + API calls, no accelerator time")
    print(f"  run                      SPENDS GPU QUOTA")
    if bk.name == "colab" and "UNTESTED" not in (rep.note or ""):
        # The colab backend states its own untested status in its doctor note;
        # this is the fallback for a backend that does not. Printed at most
        # once, because saying "this is untested" twice reads as emphasis
        # rather than as a fact.
        print()
        print(util.yellow("  UNTESTED: gpush's colab path has never been run against a"))
        print(util.yellow("  live Colab runtime. `doctor` above is a checklist of"))
        print(util.yellow("  preconditions, not a verification that it works."))
    return 0 if rep.ok else 1


def do_push(args, bk, tree, out) -> tuple[int, object]:
    """The push phase. Returns (exit code, SyncState or None if nothing ran).

    Shared by `push` and `run` rather than reimplemented, so `run` cannot
    skip a guard or a gate that `push` would have applied -- and so the run's
    provenance can record which dataset version and sha256 the run was given.
    """
    plan = bk.plan(tree)
    store = bk.store

    # -- the guards. Free, offline, and before anything moves.
    gr = gates.GateResult()
    md = gates.mass_deletion(plan, args.delete_limit,
                             args.allow_mass_delete or args.full_resync)
    if args.full_resync:
        md.ok, md.detail, md.remedy = True, \
            "resetting the remote workdir; delete-sync is moot", ""
    gr.add(md.name, md.ok, md.detail, md.remedy)
    cov = gates.coverage(plan)
    gr.add(cov.name, cov.ok, cov.detail, cov.remedy)

    # The skip decision, stated once and used everywhere below. Both halves are
    # required and neither is enough on its own:
    #   plan.empty                   the tree has not moved since the baseline
    #   store.is_proven(fingerprint) a completed run attested to that baseline
    # A matching fingerprint with a `pending` baseline is the shape of the lost
    # run in DECISIONS.md #18, and it must NOT skip.
    proven = store.is_proven(tree.fingerprint)
    skip = plan.empty and proven
    # Control flow is `skip`, decided above. This is only the sentence that
    # says so, printed either way -- silence here reads as "hung".
    nc_detail = gates.no_changes(
        skip, plan.summary(),
        store.why_not_proven(tree.fingerprint) if plan.empty else
        f"{plan.bytes_in/1024:.1f} KB to send")
    gr.add("no changes, skipping", True, nc_detail)

    print(util.hr(f"push: {bk.name}"))
    show_plan(bk, tree, plan, store, header="")
    print()
    if not gr.ok:
        print(gr.report("guards REFUSED the push"))
        return 2, None
    print(gr.report("guards"))
    print()

    if skip:
        print()
        print(util.green(f"no changes, skipping the transfer: the remote fingerprint "
                         f"{tree.fingerprint[:12]} is already proven by a completed "
                         f"run; {len(plan.unchanged)} files, none of them transferred"))
        return 0, None

    st = bk.push(tree, plan, full_resync=args.full_resync)
    print()
    if st.gates:
        print(util.hr("backend gates"))
        for n, ok, d in st.gates:
            print(f"  [{'pass' if ok else 'REFUSE':<6}] {n:<28} {d}")
        print()
    if not st.ok:
        print(util.red("  push REFUSED by a backend gate. No GPU time was spent."))
        if st.note:
            print(util.wrap(st.note))
        return 3, st
    print(f"  {st.note}")
    print("  baseline is now PENDING: uploaded, but not yet attested by a run.")
    if args.json:
        print(util.jdump({"ok": st.ok, "fingerprint": st.fingerprint,
                          "remote": st.remote, "bytes_sent": st.bytes_sent}))
    return 0, st


def cmd_push(args, bk, tree, out) -> int:
    rc, _st = do_push(args, bk, tree, out)
    return rc


def cmd_run(args, bk, tree, out) -> int:
    cmd = args.cmd or []
    if not cmd:
        print("nothing to run: pass the command after --\n"
              "  gpush kaggle run -- python -m kda.perf_kda", file=sys.stderr)
        return 64
    store = bk.store
    plan = bk.plan(tree)
    run_dir = provenance.new_run_dir(out, bk.name, _slug_of(bk), tree.fingerprint)

    print(util.hr(f"run: {bk.name}"))
    print(util.yellow("  THIS SPENDS GPU QUOTA"))
    print(f"  backend   {bk.name}")
    print(f"  command   {' '.join(cmd)}")
    print(f"  runs at   {run_dir}")
    print()

    # The push phase is do_push verbatim, not a reimplementation of it, so
    # `run` can never skip a guard or a gate that `push` would have applied.
    rc, sync = do_push(args, bk, tree, out)
    # The addressing of whatever the run will execute against, recorded even on
    # the failure paths: a run that was refused is still evidence, and the
    # evidence is more useful with the dataset and sha256 it would have used.
    remote = (sync.remote if sync is not None else {}) or {}
    # On a no-op skip there is no SyncState, but the run still executes against
    # whatever the baseline describes -- and that is exactly the addressing the
    # provenance record needs. Three fallbacks, in order of authority: the push
    # that just happened, the durable baseline, then the backend's own
    # configured address. Without the third, a single early run recorded with an
    # empty remote would leave every later record empty too.
    if not remote and store.remote:
        remote = dict(store.remote)
    if not remote:
        remote = bk.describe_target() or {}
    bk.last_gates = (sync.gates if sync is not None else []) or []
    if rc == 2:
        provenance.write(run_dir, backend=bk, tree=tree, plan=plan, command=cmd,
                         device={}, run_id=os.path.basename(run_dir),
                         status="REFUSED BY A GUARD", started="", finished="",
                         exit_code=None, remote=remote, gates=[], artefacts=[],
                         note="a guard refused the push before any upload; "
                              "no GPU time spent")
        print("\n  provenance written to "
              f"{run_dir} (the refusal is the evidence)")
        return rc
    if rc == 3:
        provenance.write(run_dir, backend=bk, tree=tree, plan=plan, command=cmd,
                         device={}, run_id=os.path.basename(run_dir),
                         status="REFUSED BY A BACKEND GATE", started="", finished="",
                         exit_code=None, remote=remote, gates=[], artefacts=[],
                         note="a backend gate refused the payload; no GPU time spent")
        print(f"\n  provenance written to {run_dir}")
        return 3

    if args.dry_run:
        print(util.yellow("  --dry-run: not executing"))
        provenance.write(run_dir, backend=bk, tree=tree, plan=plan, command=cmd,
                         device={}, run_id=os.path.basename(run_dir),
                         status="DRY RUN", started="", finished="", exit_code=None,
                         remote={}, gates=[], artefacts=[],
                         note="dry run: the command was not executed")
        return 0

    print(util.hr("execute on the remote device"))
    r = bk.run(tree, cmd, timeout=args.timeout, allow_cpu=args.allow_cpu)
    print(f"  status     {r.status}")
    if r.device:
        print(f"  device     {describe(r.device)}")
        note = capability_note(r.device)
        if note:
            print(util.dim(f"             {note}"))
    print(f"  started    {r.started}    finished {r.finished or '-'}")
    if r.exit_code is not None:
        print(f"  exit code  {r.exit_code}")
    if r.note:
        print(util.wrap(r.note))

    print()
    print(util.hr("fetch artefacts"))
    artefacts = []
    try:
        artefacts = bk.fetch(r.__dict__ if hasattr(r, "__dict__") else {},
                             run_dir) or []
    except Exception as e:
        print(util.yellow(f"  fetch failed: {type(e).__name__}: {e}"))
    print(f"  {len(artefacts)} file(s) -> {run_dir}")
    for a in sorted(artefacts)[:20]:
        try:
            print(f"    {os.path.relpath(a, run_dir)}  {os.path.getsize(a)} bytes")
        except OSError:
            pass

    # The runner's own verdict on the payload it was given. This is the only
    # evidence produced from the far side about whether the bytes that ran are
    # the bytes that were pushed, so it is read and reported rather than left
    # in a JSON file nobody opens.
    verify = util.read_json(os.path.join(run_dir, "GPUSH-VERIFY.json"), None)
    if verify is not None:
        print()
        pv = payload_provenance(verify, getattr(tree, "fingerprint", None))
        print(f"  runner's payload verdict: {pv.verdict_text()}")

        if not pv.matched:
            print(util.red(f"    reason: {verify.get('reason', 'unknown')}"))
            for k in ("missing", "extra", "differing"):
                if verify.get(k):
                    print(util.red(f"    {k}: {verify[k][:6]}"))

        if pv.stale:
            print(util.red(f"    this run pushed  fingerprint {pv.pushed}"))
            print(util.red(f"    the runner read  fingerprint {pv.ran}"))
            print(util.red("    the notebook that ran was not the notebook that was"))
            print(util.red("    pushed, so it verified the PREVIOUS payload against the"))
            print(util.red("    PREVIOUS manifest. It agrees with itself and is wrong."))

        if not pv.matched or pv.stale:
            r.ok = False
            r.status = f"{r.status} ({'stale payload' if pv.stale else 'payload mismatch'})"
            r.note = (r.note + " " if r.note else "") + \
                ("the notebook that executed was a cached earlier version, so the "
                 "run measured the previous payload; " if pv.stale else "") + \
                "no number from this run may be quoted."

    # Kaggle reports a kernel status, not a process exit code. Derive one so
    # the record is comparable with the SSH backend's, and so "COMPLETE with a
    # mismatched payload" cannot be read as success by a later reader.
    exit_code = r.exit_code
    if exit_code is None:
        exit_code = 0 if (r.ok and r.status == "COMPLETE") else 1

    p = provenance.write(run_dir, backend=bk, tree=tree, plan=plan, command=cmd,
                         device=r.device, run_id=os.path.basename(run_dir),
                         status=r.status, started=r.started, finished=r.finished,
                         exit_code=exit_code, remote=remote,
                         gates=getattr(bk, "last_gates", []),
                         artefacts=artefacts, note=r.note)
    print()
    print(util.hr("provenance"))
    print(f"  file          {p}")
    print(f"  fingerprint   {tree.fingerprint}")
    print(f"  backend       {bk.name}  ({bk.transport[:60]}...)")
    if r.device:
        print(f"  device        {describe(r.device)}")
        print(f"  driver/CUDA   {r.device.get('driver', '?')} / "
              f"{r.device.get('cuda_runtime', '?')}")
    print(f"  command       {' '.join(cmd)}")

    if r.ok:
        store.mark_confirmed(tree, remote, os.path.basename(run_dir))
        print()
        print(util.green(f"  baseline PROMOTED to confirmed. The next push of this "
                         f"tree is a proven no-op."))
    else:
        store.note_run(os.path.basename(run_dir), tree.fingerprint, r.status)
        print()
        print(util.yellow(f"  baseline NOT promoted (run {r.status}). The payload is "
                          f"still unproven, so the next `push` re-uploads rather "
                          f"than skipping. This is the shape of the lost run in "
                          f"DECISIONS.md #18."))
    print()
    print(f"  run directory: {run_dir}")
    return 0 if r.ok else 7


def cmd_status(args, bk, tree, out) -> int:
    s = bk.status()
    print(util.hr(f"status: {bk.name}"))
    print(f"  phase        {s.phase}")
    if s.detail:
        print(util.wrap(s.detail, indent="               "))
    if s.device:
        print(f"  device       {describe(s.device)}")
        n = capability_note(s.device)
        if n:
            print(util.dim(f"               {n}"))
    print()
    print(f"  baseline     {bk.store.describe()}")
    if bk.store.fingerprint and tree.fingerprint != bk.store.fingerprint:
        print(util.yellow(f"  the working tree has moved on: {tree.fingerprint[:12]} "
                          f"vs baseline {bk.store.fingerprint[:12]}"))
    elif not bk.store.is_proven(tree.fingerprint):
        print(util.yellow(f"  {bk.store.why_not_proven(tree.fingerprint)}"))
    else:
        print(util.green("  the baseline is proven by a completed run"))
    return 0


def cmd_fetch(args, bk, tree, out) -> int:
    if not args.run_dir:
        print("pass --run-dir <dir under out/gpush/runs>; `run` prints it.",
              file=sys.stderr)
        return 64
    dest = os.path.abspath(args.run_dir)
    print(util.hr(f"fetch: {bk.name}"))
    files = bk.fetch({}, dest) or []
    print(f"  {len(files)} file(s) -> {dest}")
    for f in sorted(files)[:30]:
        print(f"    {os.path.relpath(f, dest)}  {os.path.getsize(f)} bytes")
    return 0 if files else 7


# ---------------------------------------------------------------------- misc

def _slug_of(bk) -> str:
    return getattr(bk, "kslug", None) or getattr(bk, "host", None) or bk.name


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="gpush",
        description="Sync a working tree to a remote notebook-GPU box and run work there.",
        epilog="plan is offline. push costs bytes. run costs GPU quota.")
    ap.add_argument("--version", action="version", version=f"gpush {__version__}")
    ap.add_argument("--list-backends", action="store_true",
                    help="print the registered backends and their transports")
    # The shape is `gpush <backend> <subcommand>`, two levels, because the
    # backend is chosen before the subcommand and each backend supplies its own
    # addressing options. Choices come from the registry, so adding a backend
    # module makes it appear here with no edit to this file.
    bsub = ap.add_subparsers(dest="backend", metavar="<backend>")
    known = be.known()
    for bname in known:
        bsp = bsub.add_parser(bname, help=be.load_all()[bname].transport[:70])
        sub = bsp.add_subparsers(dest="action", metavar="<subcommand>")

        def common(p, needs_cmd=False):
            p.add_argument("--root", default=".", help="the working tree to sync (default .)")
            p.add_argument("--out", default=DEFAULT_OUT,
                           help=f"where gpush keeps state, scratch and runs (default {DEFAULT_OUT})")
            p.add_argument("--include", action="append",
                           help="path prefix to ship; repeatable. Default: the whole working tree.")
            p.add_argument("--exclude", action="append", help="path prefix to drop; repeatable")
            p.add_argument("--checksum", action="store_true",
                           help="re-read and hash every file instead of trusting "
                                "(size, mtime) against the baseline")
            p.add_argument("--full-resync", action="store_true",
                           help="reset the remote workdir and re-seed from scratch")
            p.add_argument("--allow-mass-delete", action="store_true",
                           help="override the mass-deletion guard")
            p.add_argument("--delete-limit", type=float, default=0.30,
                           help="max fraction of the remote manifest a single sync may "
                                "prune (default 0.30)")
            p.add_argument("--allow-cpu", action="store_true",
                           help="permit a run on a machine with no GPU; the result is "
                                "recorded as a CPU result, not a GPU one")
            p.add_argument("--skip-verify", action="store_true",
                           help="skip the payload integrity gate. This is the gate "
                                "DECISIONS.md #18 depends on; it exists for debugging only.")
            p.add_argument("--tpu", action="store_true", help="kaggle: request a TPU, not a GPU")
            p.add_argument("--json", action="store_true", help="machine-readable summary")
            p.add_argument("--kaggle-cli", default=os.environ.get("GPUSH_KAGGLE_CLI", ""),
                           help="path to the kaggle CLI")
            p.add_argument("--remote-dir", default="/root/gpush",
                           help="colab: the remote working directory")
            p.add_argument("--timeout", type=int, default=1800,
                           help="remote command timeout, seconds")
            # These three are options on every subcommand rather than only on
            # `run`: the backend object is constructed once for all of them and
            # holds them, and a `push` that ignored the session cap would be a
            # push that could start a session it could not wait for.
            p.add_argument("--session", type=int, default=60,
                           help="kaggle: session cap in minutes, enforced server-side")
            p.add_argument("--queue-wait", type=int, default=90,
                           help="kaggle: minutes to wait for a slot")
            p.add_argument("--poll", type=int, default=20, help="kaggle: poll seconds")
            add_backend_args(p)
            if needs_cmd:
                p.add_argument("--cmd", nargs=argparse.REMAINDER,
                               help="the command to run on the device, after --")
            return p

        for name in ("plan", "doctor", "push", "run", "status", "fetch"):
            p = common(sub.add_parser(name), needs_cmd=(name == "run"))
            if name == "run":
                p.add_argument("--dry-run", action="store_true",
                               help="do everything except execute the command")
            if name == "fetch":
                p.add_argument("--run-dir", default="", help="a run directory to re-fetch into")
    return ap


DISPATCH = {"plan": cmd_plan, "doctor": cmd_doctor, "push": cmd_push,
            "run": cmd_run, "status": cmd_status, "fetch": cmd_fetch}


def main(argv=None) -> int:
    # `gpush <backend> run [options] -- <command> [args...]`
    #
    # The split happens here, before argparse sees anything, because argparse
    # consumes the `--` itself and REMAINDER then either swallows the command
    # or hands it back half-parsed. Both happen, and both are silent. Splitting
    # the raw argv on the first standalone `--` is the only version where what
    # follows is guaranteed byte-for-byte what the user typed, which matters
    # because that argv is recorded verbatim in the provenance record.
    raw = list(sys.argv[1:] if argv is None else argv)
    cmd_after: list[str] = []
    if "--" in raw:
        i = raw.index("--")
        cmd_after, raw = raw[i + 1:], raw[:i]

    ap = build_parser()
    args = ap.parse_args(raw)
    if cmd_after:
        args.cmd = cmd_after
    if getattr(args, "list_backends", False):
        print("gpush backends:")
        for n in sorted(be.load_all()):
            c = be.load_all()[n]
            print(f"  {n:<10} {c.transport}")
        print()
        print("interface (six methods, three phases):")
        print("  SYNC  plan(tree) -> Plan | push(tree, plan, full_resync) -> State | status() -> Status")
        print("  RUN   doctor() -> Report | run(tree, argv) -> Run        # only this spends GPU quota")
        print("  OUT   fetch(state, dest) -> [Path]")
        return 0
    if not args.backend:
        ap.print_help()
        return 64

    if not getattr(args, "action", None):
        print(f"\ngpush {args.backend}: pick a subcommand.\n"
              f"  {' | '.join(sorted(DISPATCH))}\n\n"
              f"  plan is offline. push costs bytes. run costs GPU quota.",
              file=sys.stderr)
        return 64

    cmd = args.action
    out = os.path.abspath(args.out)
    root = os.path.abspath(args.root)
    if cmd == "run" and not args.cmd:
        print("nothing to run. Usage:\n"
              "  gpush <backend> run [options] -- <command> [args...]",
              file=sys.stderr)
        return 64

    store = StateStore(out).bind(args.backend, _target_of(args))
    # A full resync deliberately drops the baseline BEFORE anything reads it,
    # so the plan is computed against an empty remote and every file reads as
    # new. Done here rather than inside push(), because the no-op check in
    # cmd_push reads the same store and must see the same state.
    if getattr(args, "full_resync", False):
        store.invalidate(f"--full-resync at {util.stamp()}")
    try:
        bk = make_backend(args.backend, args, store, out)
    except SystemExit:
        raise
    except Exception as e:
        print(f"{util.red('cannot construct the ' + args.backend + ' backend:')} {e}",
              file=sys.stderr)
        return 64

    baseline = store.manifest() if store.fingerprint else None
    tree = build(root, baseline, checksum=args.checksum,
                 include=args.include, exclude=args.exclude)
    tree.compute_fingerprint(bk.name, _target_of(args))

    try:
        return DISPATCH[cmd](args, bk, tree, out)
    except KeyboardInterrupt:
        print("\ninterrupted. the baseline was not promoted, so nothing was "
              "recorded as measured.", file=sys.stderr)
        return 130


def _target_of(args) -> str:
    return getattr(args, "dataset", "") or getattr(args, "kernel", "") or \
        getattr(args, "host", "") or args.backend


if __name__ == "__main__":
    sys.exit(main())
