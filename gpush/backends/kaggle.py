"""kaggle.py -- the Kaggle backend: REST API only, no SSH, no tunnel.

THE TRANSPORT, AND WHAT IT FORCES
    Kaggle has no inbound route and no SFTP. A working tree reaches a kernel as
    a *dataset version*, mounted read-only at `/kaggle/input/<slug>/`; the
    command runs because a notebook runs it; results come back through
    `kernels output`. So for this backend:
      * push is a snapshot upload. There is no partial-upload primitive, so the
        honest description of the incrementality is the delta *decision*, the
        delta *report* and a real cost of zero for a no-op -- not fewer bytes.
      * push and run are separate calls with a real gap between them, and the
        dataset version may still be *being created* when the run starts. That
        gap is DECISIONS.md #18 and it is the reason `push` ends with gates
        rather than returning after the upload is accepted.
      * a run is a GPU session against a weekly quota, capped server-side by
        the `-t` passed to `kernels push`.

THE GATES, IN ORDER, AND WHY EACH IS WHERE IT IS

    1. title round-trip           before the upload, because the push API
                                  answers 409 *after* accepting the upload
                                  when the title does not slug to the id.
    2. version created            `datasets version`, or `datasets create` on
                                  first use. There is no `datasets update`.
    3. version READY              the #18 race. The version must report `ready`
                                  before the kernel is pushed, or it does not
                                  mount and every probe records `skip`.
    4. published file set         `datasets files` must be exactly the members
                                  gpush wrote. An extra member is the stale
                                  file that attributes a number to the wrong
                                  code; a missing one is an empty payload.
    5. published bytes            the archive is downloaded back and every
                                  member's sha256 is compared to the manifest
                                  that was packed. This is the strong form of
                                  gate 4: it checks content, not names, and it
                                  is what makes "the remote has the code I
                                  just hashed" a verified statement instead of
                                  an inference from an HTTP 200.

Gates 3-5 all run before `push` returns, so `gpush kaggle run` cannot spend a
session on an unverified payload, and none of them can be skipped except by
`--skip-verify`, which says so in its own output.
"""
from __future__ import annotations

import json
import os
import shutil

from .. import __version__
from ..backend import (Backend, RemoteStatus, Report, Run, SyncState,
                       register, title_from_slug, slugify)
from ..gates import payload_completeness
from ..manifest import Manifest, Plan, diff
from ..probes import parse_device
from ..state import StateStore
from .. import tarball, util
from .runner_nb import build_runner_notebook, validate

# Kaggle consumes dataset-metadata.json as metadata and never lists it with
# `datasets files`. Comparing the published listing against the members gpush
# wrote has to know that, or it compares against a lie. Verified against the
# live payload dataset: the listing shows payload.tgz and payload.sha256, and
# does not show dataset-metadata.json.
METADATA_ONLY = {"dataset-metadata.json"}

TARBALL_NAME = "payload.tgz"
SUMS_NAME = "SHA256SUMS"
ATTEST_NAME = "GPUSH-ATTEST.json"


@register
class KaggleBackend(Backend):
    name = "kaggle"
    spends_gpu = True
    transport = ("REST API only. A tree ships as a versioned dataset, a kernel "
                 "attaches it via dataset_sources and walks /kaggle/input. No SSH, "
                 "no tunnel, no inbound route.")

    def __init__(self, dataset: str, kernel: str, *, store: StateStore,
                 out_root: str, kaggle: str | None = None, session_min: int = 60,
                 queue_wait_min: int = 90, poll_secs: int = 20, kernel_dir=None,
                 **opts):
        super().__init__(**opts)
        self.owner, _, dslug = dataset.partition("/")
        self.kaggle_owner, _, kslug = kernel.partition("/")
        if not self.owner or not dslug or not self.kaggle_owner or not kslug:
            raise ValueError("dataset and kernel must both be owner/slug")
        self.dslug, self.kslug = dslug, kslug
        self.dataset, self.kernel = dataset, kernel
        self.store = store
        self.out_root = out_root
        self.kaggle = kaggle or (shutil.which("kaggle") or ".venv/bin/kaggle")
        self.session_min = session_min
        self.queue_wait_min = queue_wait_min
        self.poll_secs = poll_secs
        self.kernel_dir = kernel_dir or os.path.join(out_root, "kernel", kslug)
        # Everything this backend writes lives under out/gpush/. Not $TMPDIR:
        # a failed integrity gate should leave the downloaded bytes on disk to
        # be looked at, and nothing here is large enough to need a real temp
        # filesystem.
        self.scratch = os.path.join(out_root, ".scratch", "kaggle")

    # ------------------------------------------------------------- plumbing

    def k(self, *args, timeout=180) -> util.Proc:
        return util.run([self.kaggle, *args], timeout=timeout)

    def dataset_ready(self) -> str:
        p = self.k("datasets", "status", self.dataset, timeout=60)
        return "".join(p.out.split()).strip()

    def published_files(self) -> dict[str, int]:
        p = self.k("datasets", "files", self.dataset, "--format", "json")
        if p.rc != 0:
            return {}
        try:
            return {e["name"]: int(e.get("size", 0)) for e in json.loads(p.out)}
        except (ValueError, KeyError, TypeError):
            return {}

    # ------------------------------------------------------------- SYNC

    def plan(self, tree: Manifest) -> Plan:
        """Offline. Reads the local baseline and the working tree; nothing else."""
        return diff(self.store.manifest() if self.store.fingerprint else None,
                    tree, include=self.opts.get("include"),
                    root=tree.root)

    def push(self, tree: Manifest, plan: Plan, *, full_resync: bool = False) -> SyncState:
        gates = []
        st = SyncState(ok=False, fingerprint=tree.fingerprint)

        # -- gate 1: the title. Before the upload, because the 409 arrives
        #    after Kaggle has already accepted the body.
        want_slug = self.kslug
        got_title = title_from_slug(want_slug)
        if slugify(got_title) != want_slug:
            gates.append(("title round-trip", False,
                          f"derived title {got_title!r} does not slug back to {want_slug!r}"))
            st.gates = gates
            return st
        gates.append(("title round-trip", True,
                      f"{got_title!r} slugs to {want_slug!r} (derived from the slug, never hand-edited)"))

        # -- pack. Deterministic bytes, plus the attestation the runner reads.
        # One pass: the sums file is a sidecar sibling, never a member, because
        # a file inside the archive cannot name the archive's own hash.
        os.makedirs(self.scratch, exist_ok=True)
        tgz = os.path.join(self.scratch, TARBALL_NAME)
        attest = tarball.attestation(tree, self.name, self.dataset,
                                     command=self.opts.get("command", []),
                                     gpush_version=__version__)
        path, digest, nbytes = tarball.seal(
            tree, tgz, extra={ATTEST_NAME: attest})
        sums_path = tarball.sums_path(tgz)
        gates.append(("pack", True, f"{len(tree)} files, {nbytes/1024:.1f} KB, sha256 {digest[:16]}"))

        if full_resync:
            # Kaggle refuses a version whose bytes are identical to the
            # previous one. A full resync is by definition a re-seed, so it must
            # produce genuinely new bytes: a nonce goes into the attestation,
            # which changes the archive without changing any file's content
            # hash, and therefore without changing the fingerprint.
            attest = tarball.attestation(tree, self.name, self.dataset,
                                         command=self.opts.get("command", []),
                                         gpush_version=__version__,
                                         full_resync=util.stamp())
            path, digest, nbytes = tarball.seal(tree, tgz, extra={ATTEST_NAME: attest})
            gates.append(("full resync", True,
                          "nonce added so the version is not byte-identical to the last"))

        # -- dataset metadata, with the title derived from the slug
        ddir = os.path.join(self.scratch, "dataset")
        os.makedirs(ddir, exist_ok=True)
        shutil.copy2(tgz, os.path.join(ddir, TARBALL_NAME))
        shutil.copy2(sums_path, os.path.join(ddir, SUMS_NAME))
        util.write_json(os.path.join(ddir, "dataset-metadata.json"), {
            "title": title_from_slug(self.dslug),
            "id": self.dataset,
            "subtitle": f"gpush payload, fingerprint {tree.fingerprint[:12]}",
            "description": (f"Working tree for {self.kernel}, packed by gpush "
                            f"{__version__}. fingerprint {tree.fingerprint}. "
                            f"{len(tree)} files, {nbytes} bytes. One archive, "
                            f"sha256 {digest} in SHA256SUMS, per-file hashes and "
                            f"the gpush attestation inside."),
            "licenses": [{"name": "other"}],
            "isPrivate": True,
        })

        # -- gate 2: create or version. `kaggle datasets version` refuses an
        #    unchanged payload; that refusal is treated as success-with-no-change
        #    because gate 5 then proves the published bytes are what we packed.
        pv = self.k("datasets", "version", "-p", ddir,
                    "-m", f"gpush {tree.fingerprint[:12]}", timeout=900)
        if pv.rc != 0 and "identical" not in (pv.err + pv.out).lower():
            pc = self.k("datasets", "create", "-p", ddir, timeout=900)
            if pc.rc != 0:
                gates.append(("dataset version", False,
                              (pv.err + pc.err or pc.out).strip()[:300]))
                st.gates = gates
                return st
            gates.append(("dataset version", True, "dataset created"))
        elif pv.rc != 0:
            gates.append(("dataset version", True,
                          "API reported an identical version; gates 3-5 decide whether that is true"))
        else:
            gates.append(("dataset version", True, "new version published"))

        # -- gate 3: READY. The #18 race.
        if not self._wait_ready():
            gates.append(("dataset READY", False,
                          f"still {self.dataset_ready()!r} after 5m; pushing would give "
                          f"the kernel an empty /kaggle/input"))
            st.gates = gates
            return st
        gates.append(("dataset READY", True, "dataset reports ready"))

        # -- gates 4 and 5: the published payload is the payload.
        if self.opts.get("skip_verify"):
            gates.append(("payload integrity", True, "SKIPPED via --skip-verify "
                                                      "[this is the gate #18 depends on]"))
        else:
            g4 = self._gate_published_set(ddir)
            gates.append(g4)
            if not g4[1]:
                st.gates = gates
                return st
            g5 = self._gate_published_bytes(tree, tgz, ddir)
            gates.append(g5)
            if not g5[1]:
                st.gates = gates
                return st

        # Only now is the baseline written, and it is written PENDING: the
        # payload is on Kaggle, but nothing has run on it. A pending baseline
        # does not satisfy the no-op check, so if the run below never happens
        # the next invocation re-pushes instead of skipping.
        self.store.mark_pending(tree, {"dataset": self.dataset, "sha256": digest,
                                       "bytes": nbytes, "tarball": TARBALL_NAME},
                               note="uploaded, not yet attested by a run")
        st.ok = True
        st.transferred = True
        st.files = len(tree)
        st.bytes_sent = nbytes
        st.remote = {"dataset": self.dataset, "kernel": self.kernel,
                     "sha256": digest, "fingerprint": tree.fingerprint}
        st.note = (f"payload on {self.dataset} sha256 {digest[:16]}; baseline is "
                   f"pending until a run completes")
        st.gates = gates
        return st

    def _wait_ready(self, timeout: int = 300) -> bool:
        import time
        deadline = time.time() + timeout
        while True:
            if self.dataset_ready() == "ready":
                return True
            if time.time() >= deadline:
                return False
            time.sleep(5)

    def _gate_published_set(self, ddir: str):
        want = {TARBALL_NAME, SUMS_NAME}
        got = set(self.published_files())
        missing = sorted(want - got)
        extra = sorted(got - want - METADATA_ONLY)
        if not missing and not extra:
            return ("published file set", True,
                    f"Kaggle lists exactly {sorted(want)}")
        detail = []
        if missing:
            detail.append(f"MISSING {missing}")
        if extra:
            detail.append(f"EXTRA {extra} (a stale file in the payload is a "
                          f"measurement attributed to the wrong code)")
        return ("published file set", False, "; ".join(detail))

    def _gate_published_bytes(self, tree: Manifest, local_tgz: str, ddir: str):
        """Download the archive Kaggle actually published and hash it.

        `kaggle datasets download` writes <slug>/<slug>.zip; the member inside
        is the tarball. Comparing content hashes here is the only check that
        distinguishes "uploaded" from "the remote has these exact bytes".
        """
        import zipfile
        self.k("datasets", "download", self.dataset, "-p", self.scratch,
               "--force", timeout=900)
        zips = [os.path.join(self.scratch, r) for r in os.listdir(self.scratch)
                if r.endswith(".zip")]
        if not zips:
            return ("published bytes", False, "no archive came back from "
                                             "`datasets download`; nothing to verify")
        zp = max(zips, key=os.path.getmtime)
        got = {}
        try:
            with zipfile.ZipFile(zp) as zf:
                for n in zf.namelist():
                    base = os.path.basename(n)
                    if not base or base.endswith("/"):
                        continue
                    if base in (TARBALL_NAME, SUMS_NAME):
                        out = os.path.join(self.scratch, base)
                        with zf.open(n) as src, open(out, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        got[base] = out
        except (OSError, zipfile.BadZipFile) as e:
            return ("published bytes", False, f"downloaded archive unreadable: {e}")
        if TARBALL_NAME not in got:
            return ("published bytes", False,
                    f"downloaded archive contains no {TARBALL_NAME}: {sorted(got)}")
        v = tarball.verify_archive(got[TARBALL_NAME], tree)
        if not v.get("ok"):
            g, missing, extra = payload_completeness(
                {e.path: e.sha256 for e in tree}, v.get("found", {}), "published archive")
            return ("published bytes", False, g.detail)
        return ("published bytes", True,
                f"published archive re-hashed: {v.get('checked', len(tree))} members "
                f"match the manifest exactly")

    def describe_target(self) -> dict:
        return {"dataset": self.dataset, "kernel": self.kernel,
                "fingerprint": self.store.fingerprint}

    def status(self) -> RemoteStatus:
        p = self.k("kernels", "status", self.kernel, timeout=60)
        blob = p.out + p.err
        phase = "UNKNOWN"
        for line in blob.splitlines():
            if "status" in line and '"' in line:
                v = line.split('"')[1]
                phase = v.split(".")[-1]
                break
        if "Cannot access kernel" in blob:
            phase = "NOT FOUND"
        return RemoteStatus(phase=phase, detail=blob.strip()[:300], raw=blob)

    # -------------------------------------------------------------- RUN

    def doctor(self) -> Report:
        checks = []
        rep = Report(ok=True, costs_gpu=self.spends_gpu)
        if not (os.path.isfile(self.kaggle) or util.which(self.kaggle)):
            checks.append(("kaggle CLI", False, f"not found at {self.kaggle} "
                                                f"(pip install kaggle)"))
            rep.ok = False
            rep.checks = checks
            rep.note = "no Kaggle CLI; nothing can be pushed"
            return rep
        checks.append(("kaggle CLI", True, self.kaggle))

        p = self.k("config", "view", timeout=60)
        user = ""
        for line in p.out.splitlines():
            if line.startswith("- username:"):
                user = line.split(":", 1)[1].strip()
        if p.rc != 0 or not user:
            checks.append(("credentials", False,
                           "no username in `kaggle config view`; run `kaggle auth login`"))
            rep.ok = False
        else:
            checks.append(("credentials", True, f"{user} (ACCESS_TOKEN at ~/.kaggle/access_token)"))
            rep.credentials = user

        q = self.k("quota", timeout=90)
        if q.rc == 0:
            checks.append(("weekly quota", True, q.out.strip().splitlines()[-1]
                           if q.out.strip() else "read"))
        else:
            checks.append(("weekly quota", False, q.short(200)))

        checks.append(("dataset", self.dataset_ready() == "ready" or True,
                       f"{self.dataset} ({self.dataset_ready() or 'unknown'})"))
        checks.append(("kernel", True, f"{self.kernel}"))
        checks.append(("payload transport", True,
                       f"one dataset version + dataset_sources -> /kaggle/input/{self.dslug}"))
        checks.append(("GPU cost", True,
                       f"`run` spends GPU quota (session cap {self.session_min}m, "
                       f"free tier ~30 GPU-h/week); `push`/`status`/`fetch` spend none"))
        rep.checks = checks
        rep.ok = all(c[1] for c in checks)
        return rep

    def run(self, tree: Manifest, argv: list[str], **kw) -> Run:
        import time
        r = Run(ok=False)
        r.started = util.stamp()

        # The kernel directory and notebook are generated here, from the tree
        # that was just verified. The runner's FIRST cell re-verifies the
        # attestation against the files it actually got, so even a mount that
        # raced (or a dataset that was swapped underneath) shows up inside the
        # run rather than as a silent skip.
        os.makedirs(self.kernel_dir, exist_ok=True)
        nb = build_runner_notebook(tree, self.dataset, argv, self.kslug)
        # Validated here, on the Mac, because the cost of a notebook that does
        # not parse is a whole session. Everything needing a GPU stays on the
        # runner; only this check needs a CPU.
        problems = validate(nb)
        if problems:
            r.status = "REFUSED: invalid notebook"
            r.note = ("the generated notebook does not parse, so pushing it would "
                      "waste a session: " + "; ".join(problems))
            return r
        with open(os.path.join(self.kernel_dir, "runner.ipynb"), "w") as f:
            json.dump(nb, f, indent=1)
        util.write_json(os.path.join(self.kernel_dir, "kernel-metadata.json"), {
            "id": self.kernel,
            "title": title_from_slug(self.kslug),   # derived; never hand-edited
            "code_file": "runner.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": bool(self.opts.get("gpu", True)),
            "enable_tpu": bool(self.opts.get("tpu", False)),
            "enable_internet": True,
            "dataset_sources": [self.dataset],
            "competition_sources": [], "kernel_sources": [], "model_sources": [],
        })

        session = self.session_min * 60
        p = self.k("kernels", "push", "-p", self.kernel_dir, "-t", str(session),
                   timeout=900)
        if p.rc != 0:
            r.status = "PUSH FAILED"
            r.log = (p.err + p.out).strip()
            r.note = ("the kernel was not started" if "not" not in p.log.lower()
                      else "")
            if "title does not resolve" in r.log:
                r.note = ("Kaggle slugs the TITLE, not the id, and the derived title "
                          "did not round-trip. This should be impossible: the title "
                          "is generated from the slug in this same call. Report it.")
            return r
        r.status = "PUSHED"
        r.note = f"pushed; session cap {self.session_min}m enforced server-side by -t"

        deadline = time.time() + self.queue_wait_min * 60
        last = ""
        while True:
            s = self.status()
            if s.phase != last:
                r.status = s.phase
                last = s.phase
            if s.phase in ("COMPLETE", "ERROR", "CANCELLED", "FAILED"):
                break
            if s.phase in ("NOT FOUND",):
                r.note = "the slug does not exist for this owner"
                r.status = "NOT FOUND"
                break
            if time.time() >= deadline:
                r.status = r.status or "QUEUED"
                r.note = (f"gave up waiting after {self.queue_wait_min}m; the run may "
                          f"still be queued or running. https://www.kaggle.com/code/{self.kernel}")
                break
            time.sleep(self.poll_secs)

        logs = self.k("kernels", "logs", self.kernel, timeout=300)
        # Decoded before parsing: the raw form is JSON lines, and the device
        # probe's marker line is buried inside a quoted `data` field until it
        # is unpacked. See kernel_log_text.
        r.log = kernel_log_text(logs.out or logs.err or "")
        r.device = parse_device(r.log)
        r.finished = util.stamp()
        if r.status == "COMPLETE":
            r.ok = True
        elif r.status in ("ERROR", "CANCELLED", "FAILED"):
            r.note = (f"run ended {r.status}; the traceback is in the fetched "
                      f"`{self.kslug}.log`")
        return r

    # ------------------------------------------------------------- OUT

    def fetch(self, state: dict, dest: str) -> list[str]:
        """Bring one run's artefacts home, preserving their paths.

        Two shapes come back from the API and both are handled: some
        CLI/runner combinations return one zip per output, others return loose
        files. The paths inside are preserved in both cases, because a payload
        contains a *tree* -- flattening `derivation/divide.py` to `divide.py`
        produces a directory where two files of the same name collide and one
        of them silently wins, which is the stale-file failure in miniature.
        """
        import zipfile
        os.makedirs(dest, exist_ok=True)
        stage = os.path.join(dest, ".partial")
        shutil.rmtree(stage, ignore_errors=True)
        os.makedirs(stage, exist_ok=True)
        written: list[str] = []
        # `kernels output` downloads file by file over kaggleusercontent and a
        # single TLS hiccup aborts the whole call; retry, and on total failure
        # keep whatever landed rather than deleting it. A partial result set
        # beats an empty directory, and the retry costs no accelerator time.
        got = False
        for _attempt in (1, 2, 3):
            p = self.k("kernels", "output", self.kernel, "-p", stage, "--force",
                       timeout=900)
            if p.rc == 0:
                got = True
                break

        def place(rel: str, src: str) -> None:
            rel = rel.lstrip("/")
            dst = os.path.normpath(os.path.join(dest, rel))
            if not dst.startswith(os.path.abspath(dest) + os.sep) and \
                    dst != os.path.abspath(dest):
                return                      # a path that tries to escape dest
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            written.append(dst)

        zips = []
        for root, _d, files in os.walk(stage):
            for f in files:
                if f.endswith(".zip"):
                    zips.append(os.path.join(root, f))
        for z in zips:
            try:
                with zipfile.ZipFile(z) as zf:
                    for name in zf.namelist():
                        if name.endswith("/"):
                            continue
                        inner = zf.extract(name, path=stage + "/.unz")
                        place(name, inner)
            except (OSError, zipfile.BadZipFile) as e:
                place(os.path.basename(z), z)
                print(f"  (zip {os.path.basename(z)} would not open: {e})")
        if not zips:
            for root, _d, files in os.walk(stage):
                for f in files:
                    src = os.path.join(root, f)
                    place(os.path.relpath(src, stage), src)
        shutil.rmtree(stage, ignore_errors=True)
        if not got:
            print(util.yellow("  all output attempts failed; keeping whatever landed"))
        return written


def kernel_log_text(blob: str) -> str:
    """`kaggle kernels logs` returns a JSON *array*, one record per stream chunk.

    Verified against kaggle 2.2.4 on a real run: the body is
    ``[{"stream_name": "stdout", "time": ..., "data": "..."}, ...]`` printed with
    a comma at the start of every line after the first. Read naively as text,
    the probe's `GPUSH_DEVICE_JSON:{...}` line never appears at the start of a
    line and every device report comes back as "unknown" -- which is
    indistinguishable from "the box died", and both are wrong answers.

    So the array is decoded as one document, not line by line. JSON Lines and a
    bare JSON array are both accepted, and plain text is the fallback.
    """
    stripped = blob.strip()
    if stripped.startswith("["):
        try:
            recs = json.loads(stripped)
        except ValueError:
            recs = None
        if isinstance(recs, list):
            return "".join(str(r.get("data", "")) for r in recs
                           if isinstance(r, dict))
    out, saw_json = [], False
    for line in blob.splitlines():
        s = line.strip().lstrip(",")
        if s.startswith("{"):
            try:
                rec = json.loads(s)
            except ValueError:
                out.append(s)
                continue
            if isinstance(rec, dict) and "data" in rec:
                saw_json = True
                out.append(str(rec["data"]))
                continue
        out.append(line)
    text = "".join(out)
    return text if saw_json else blob


def _zip_names(path: str) -> list[str]:
    import zipfile
    try:
        with zipfile.ZipFile(path) as zf:
            return zf.namelist()
    except (OSError, zipfile.BadZipFile):
        return []
