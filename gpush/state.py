"""state.py -- the durable record of what the remote is believed to hold.

THE ONE RULE THIS FILE EXISTS TO ENFORCE

    A baseline is promoted to `confirmed` only by a run that actually
    completed and whose artefacts came home.

That single rule is what makes the lost-run incident impossible to repeat.

WHAT WENT WRONG BEFORE (notes/DECISIONS.md #18, 2026-09-27)
    The push loop recorded "payload uploaded" as if it meant "the remote has
    this code". A dataset version was published four seconds before the kernel
    was pushed; the version had not finished being created, so it did not
    mount; the kernel ran against an empty `/kaggle/input`; and all 15 battery
    items recorded `skip`. The run *succeeded* from Kaggle's point of view.
    A GPU session was spent and the study had no measured data.

With a state machine the same sequence is:
    push   -> baseline written as `pending`  (uploaded, unattested)
    gate   -> refuses to start the run until the dataset is `ready` and the
              published bytes hash to what we built
    run    -> on COMPLETE + artefacts fetched, baseline is promoted to
              `confirmed`
    next   -> the no-op check is `fingerprint matches AND confirmed`. A
              leftover `pending` is re-pushed, loudly, because the previous run
              never proved anything.

So "skip the transfer" is never decided by a local file. It is decided by a
remote state that only a completed run can set.

WHERE THE RECORD LIVES
    `out/gpush/state/<backend>--<target>.json`, i.e. under the repo's own
    output tree, not a dotfile under $HOME. Two reasons: it is diffable and
    reviewable (one line per file, sorted), and it sits next to the runs it
    describes. A single baseline file per (backend, target) pair is enough: the
    remote holds one working tree, so there is exactly one thing to be wrong.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import tempfile

from . import BASELINE_SCHEMA
from .manifest import Manifest

# A baseline in one of these states means "nothing has been proven yet".
UNPROVEN = ("pending",)

CONFIRMED = "confirmed"
PENDING = "pending"
ABSENT = "absent"


class StateStore:
    def __init__(self, out_root: str):
        self.root = os.path.join(out_root, "state")
        self.backend = ""
        self.target = ""
        self.path = ""
        self.record: dict = {}
        self._load()

    @staticmethod
    def _slug(target: str) -> str:
        return "".join(c if (c.isalnum() or c in "-_.") else "-" for c in target) or "default"

    def bind(self, backend: str, target: str) -> "StateStore":
        self.backend, self.target = backend, target
        self.path = os.path.join(self.root, f"{backend}--{self._slug(target)}.json")
        self._load()
        return self

    def _load(self) -> None:
        self.record = {
            "schema": BASELINE_SCHEMA,
            "backend": self.backend,
            "target": self.target,
            "state": ABSENT,
            "fingerprint": "",
            "remote": {},
            "manifest": {"entries": [], "source": "", "fingerprint": ""},
            "runs": [],
            "updated": "",
        }
        if self.path and os.path.isfile(self.path):
            try:
                with open(self.path) as f:
                    d = json.load(f)
                if d.get("schema") == BASELINE_SCHEMA:
                    self.record.update(d)
            except (OSError, ValueError):
                # A corrupt baseline is treated as no baseline. That is the safe
                # direction: the worst case is a redundant re-push, never a
                # skipped transfer that should have happened.
                pass

    # ------------------------------------------------------------------ read

    @property
    def state(self) -> str:
        return self.record.get("state", ABSENT)

    @property
    def fingerprint(self) -> str:
        return self.record.get("fingerprint", "")

    @property
    def remote(self) -> dict:
        return self.record.get("remote", {})

    @property
    def runs(self) -> list:
        return self.record.get("runs", [])

    def manifest(self) -> Manifest:
        d = self.record.get("manifest", {})
        return Manifest.from_dict(d if d.get("entries") else {"entries": []})

    def is_proven(self, fingerprint: str) -> bool:
        """The one question the no-op check asks.

        Both halves are required. Matching hashes with `pending` state means
        "uploaded once, never run" -- the exact shape of the lost run.
        """
        return (self.state == CONFIRMED and bool(fingerprint)
                and self.fingerprint == fingerprint)

    def why_not_proven(self, fingerprint: str) -> str:
        if not self.fingerprint:
            return "no baseline recorded: this target has never been pushed"
        if self.fingerprint != fingerprint:
            return f"baseline fingerprint {self.fingerprint[:12]} != local {fingerprint[:12]}: the tree changed"
        if self.state != CONFIRMED:
            return (f"baseline fingerprint matches but is {self.state.upper()}: the previous "
                    f"push was never attested by a completed run (this is the shape of the "
                    f"lost run in DECISIONS.md #18), so it is re-pushed")
        return "baseline is proven by a completed run"

    # ----------------------------------------------------------------- write

    def save(self) -> None:
        self.record["schema"] = BASELINE_SCHEMA
        self.record["backend"] = self.backend
        self.record["target"] = self.target
        self.record["updated"] = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        # Atomic: a baseline torn in half by a killed process would parse as
        # corrupt and be silently downgraded to "no baseline", which is safe
        # but loses history. Writing through a temp file in the same directory
        # makes the rename the only failure window.
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.path), prefix=".baseline-")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.record, f, indent=2, sort_keys=True)
                f.write("\n")
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def mark_pending(self, manifest: Manifest, remote: dict, note: str = "") -> None:
        self.record.update({
            "state": PENDING,
            "fingerprint": manifest.fingerprint,
            "manifest": manifest.as_dict(),
            "remote": remote,
            "pending_note": note,
            "pending_since": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        })
        self.save()

    def mark_confirmed(self, manifest: Manifest, remote: dict, run_id: str) -> None:
        # Merge rather than replace: an earlier confirmed run may have recorded
        # the dataset sha256 while a later no-op skip has no sha256 to offer, and
        # replacing it with {} would erase the only record of which bytes the
        # proven tree corresponds to.
        merged = dict(self.remote or {})
        merged.update({k: v for k, v in (remote or {}).items() if v})
        self.record.update({
            "state": CONFIRMED,
            "fingerprint": manifest.fingerprint,
            "manifest": manifest.as_dict(),
            "remote": merged,
            "pending_note": "",
            "pending_since": "",
        })
        runs = self.runs
        runs.append({"run_id": run_id, "fingerprint": manifest.fingerprint,
                     "at": _dt.datetime.now().astimezone().isoformat(timespec="seconds")})
        self.record["runs"] = runs[-50:]   # bounded; a baseline is not a log
        self.save()

    def note_run(self, run_id: str, fingerprint: str, outcome: str) -> None:
        """Record an attempted run without changing the baseline's proven state.

        An ERROR run leaves the baseline exactly as it was: the payload is still
        on the remote, still unproven, and still gets re-pushed next time.
        """
        runs = self.runs
        runs.append({"run_id": run_id, "fingerprint": fingerprint, "outcome": outcome,
                     "at": _dt.datetime.now().astimezone().isoformat(timespec="seconds")})
        self.record["runs"] = runs[-50:]
        self.save()

    def invalidate(self, why: str) -> None:
        self.record.update({"state": ABSENT, "fingerprint": "", "manifest": {"entries": []},
                            "invalidation": why})
        self.save()

    def describe(self) -> str:
        if self.state == ABSENT:
            return "no baseline: never pushed"
        fp = self.fingerprint[:12] or "-"
        return f"{self.state}  {fp}  {len(self.manifest().entries)} files"
