"""colab.py -- the Colab backend: an ordinary Linux VM reached over ssh.

WHY THIS BACKEND IS NOT THE KAGGLE ONE
    A Colab GPU session is a real machine with a real POSIX userspace, a real
    filesystem and a real shell. That is the whole reason to prefer ssh here:
    the right client for it is `ssh` and `tar`, not a notebook API. Wrapping
    execute-and-return in a notebook schema would be inventing a transport
    that does not exist in order to look uniform.

    So this backend is boring on purpose: it pipes a tar in, runs argv, pipes
    the output back, pipes the artefacts home.

THE THREE REAL CONSTRAINTS, AND WHAT EACH COSTS

  1. The machine is disposable and may be reclaimed mid-run.
     Nothing here assumes it survives. The remote workdir is wiped and
     re-seeded on every push (that is what `--full-resync` does natively), and
     `run` polls the session with a bounded wait rather than blocking on a
     socket that may not come back. If the box dies, the run reports what it
     had and the artefacts that arrived are kept.

  2. The tunnel hostname changes every session.
     Never store it as truth. `--host` is re-read per invocation, and the
     resolved host is written into the provenance record alongside the
     fingerprint, because "which machine was this" is part of the measurement
     and a hostname that changes per session is exactly the kind of thing that
     must be recorded rather than remembered.

  3. The runtime may be CPU-only.
     Google does not publish which GPU a free session gets, so this backend
     reads the device off the machine every time and never assumes. With no
     `--allow-cpu`, a CPU-only session is refused before the command runs: it
     is not a slower GPU run, it is a different machine, and a number from it
     is not a GPU number.

PASSWORD AUTH WITHOUT A DEPENDENCY
    `sshpass` would be a new install. It is not needed: OpenSSH >= 8.4 reads a
    password from `SSH_ASKPASS` when `SSH_ASKPASS_REQUIRE=force` and there is
    no tty, which is exactly this case. A two-line helper script on the local
    side supplies the password, so gpush depends on nothing that is not already
    on the machine. The Colab bootstrap in remote/colab_ssh.py sets the
    password; `--password` takes it, and `--askpass-cmd` is there for anyone who
    would rather not have a literal password in a shell history.
"""
from __future__ import annotations

import os
import shlex
import stat
import tempfile

from ..backend import (Backend, RemoteStatus, Report, Run, SyncState, register)
from ..gates import Gate, device_is_gpu, payload_completeness
from ..manifest import Manifest, Plan, diff
from ..probes import DEVICE_PROBE, parse_device
from ..state import StateStore
from .. import tarball, util

DEFAULT_USER = "root"
DEFAULT_PORT = 22


@register
class ColabBackend(Backend):
    name = "colab"
    spends_gpu = True
    transport = ("ssh to a real Linux VM over a cloudflared reverse tunnel. "
                 "Ordinary POSIX userspace: tar in, argv, tar out.")

    def __init__(self, host: str | None = None, *, user: str = DEFAULT_USER,
                 port: int = DEFAULT_PORT, password: str | None = None,
                 askpass_cmd: str | None = None, store: StateStore,
                 out_root: str, remote_dir: str = "/root/gpush", timeout: int = 600,
                 identity: str | None = None, **opts):
        super().__init__(**opts)
        self.host = host            # re-read per invocation; it changes per session
        self.user = user
        self.port = port
        self.password = password
        self.askpass_cmd = askpass_cmd
        self.store = store
        self.out_root = out_root
        self.remote_dir = remote_dir.rstrip("/")
        self.timeout = timeout
        self.identity = identity
        self.scratch = os.path.join(out_root, ".scratch", "colab")
        self._askpass_path = None

    # ------------------------------------------------------------- plumbing

    def _env(self) -> dict:
        env = dict(os.environ)
        if self.askpass_cmd:
            p = self._write_askpass(self.askpass_cmd)
        elif self.password:
            p = self._write_askpass(f"printf '%s' {shlex.quote(self.password)}")
        else:
            p = None
        if p:
            # SSH_ASKPASS_REQUIRE=force is what makes this work with no tty;
            # without it OpenSSH prefers /dev/tty and there is none, so it fails.
            env["SSH_ASKPASS"] = p
            env["SSH_ASKPASS_REQUIRE"] = "force"
            env["DISPLAY"] = ":0"
        # Colab's box is disposable and busy; a control master would outlive the
        # session it was opened against and hang the next one.
        env.pop("SSH_AUTH_SOCK", None)
        return env

    def _write_askpass(self, body: str) -> str:
        if self._askpass_path and os.path.isfile(self._askpass_path):
            return self._askpass_path
        os.makedirs(self.scratch, exist_ok=True)
        fd, p = tempfile.mkstemp(dir=self.scratch, prefix="askpass-")
        with os.fdopen(fd, "w") as f:
            f.write("#!/bin/sh\n" + body + "\n")
        os.chmod(p, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        self._askpass_path = p
        return p

    def target(self) -> str:
        return f"{self.user}@{self.host}"

    def ssh_argv(self, remote_cmd: str, *, timeout=None) -> list[str]:
        argv = ["ssh", "-p", str(self.port), "-o", "StrictHostKeyChecking=accept-new",
                "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=15",
                "-o", "ServerAliveCountMax=4", "-o", "BatchMode="
                + ("no" if (self.password or self.askpass_cmd) else "yes")]
        if self.identity:
            argv += ["-i", self.identity]
        return argv + [self.target(), remote_cmd]

    def scp_argv(self, src: str, dst: str) -> list[str]:
        argv = ["scp", "-P", str(self.port), "-o", "StrictHostKeyChecking=accept-new",
                "-o", "ConnectTimeout=20", "-o", "BatchMode="
                + ("no" if (self.password or self.askpass_cmd) else "yes")]
        if self.identity:
            argv += ["-i", self.identity]
        return argv + [src, f"{self.target()}:{dst}"]

    def ssh(self, remote_cmd: str, *, timeout=None) -> util.Proc:
        return util.run(self.ssh_argv(remote_cmd), timeout=timeout or self.timeout,
                        env=self._env())

    # ------------------------------------------------------------- SYNC

    def plan(self, tree: Manifest) -> Plan:
        """Offline, exactly as in the Kaggle backend.

        Note what this does *not* do: it does not ask the remote what it has.
        The remote is disposable and may be gone by the time you push, so the
        offline answer has to come from the durable baseline, and `push`
        re-verifies against the live machine afterwards.
        """
        return diff(self.store.manifest() if self.store.fingerprint else None,
                    tree, include=self.opts.get("include"), root=tree.root)

    def push(self, tree: Manifest, plan: Plan, *, full_resync: bool = False) -> SyncState:
        st = SyncState(ok=False, fingerprint=tree.fingerprint)
        if not self.host:
            st.note = ("no --host. The tunnel hostname is printed by the "
                       "remote/colab_ssh.py bootstrap and changes every session, so "
                       "it is passed per invocation rather than configured once.")
            return st

        gates = []
        os.makedirs(self.scratch, exist_ok=True)
        tgz = os.path.join(self.scratch, "payload.tgz")
        attest = tarball.attestation(tree, self.name, self.remote_dir,
                                     host=self.host, command=self.opts.get("command", []))
        path, digest, nbytes = tarball.seal(tree, tgz,
                                            extra={"GPUSH-ATTEST.json": attest})
        gates.append(("pack", True, f"{len(tree)} files, {nbytes/1024:.1f} KB, sha256 {digest[:16]}"))

        # The box is disposable: unless asked otherwise, the workdir is reset
        # and re-seeded. That is delete-sync by construction -- there is no
        # stale-file class here at all, because nothing survives -- and it is
        # also why this backend's delete-sync needs no separate guard: the
        # mass-deletion guard in the core still applies to the manifest diff,
        # because a 90%-delete *manifest* is a mistake worth catching even when
        # the machine is going to be wiped anyway.
        wipe = full_resync or not self.store.fingerprint
        pre = (f"rm -rf {shlex.quote(self.remote_dir)} && mkdir -p {shlex.quote(self.remote_dir)}"
               if wipe else f"mkdir -p {shlex.quote(self.remote_dir)}")
        mk = self.ssh(pre, timeout=120)
        if mk.rc != 0:
            st.gates = [("reach the box", False, mk.short(400))]
            st.note = f"could not reach {self.target()}"
            return st
        gates.append(("reach the box", True,
                      f"{self.target()} (tunnel hostname, changes per session)"))

        # tar over one pipe. Not rsync: one code path, no dependency on rsync
        # being installed on a disposable box that gets rebuilt constantly, and
        # a single stream cannot half-apply.
        import subprocess
        with open(tgz, "rb") as f:
            p = subprocess.run(
                self.ssh_argv(f"tar -xzf - -C {shlex.quote(self.remote_dir)}"),
                stdin=f, capture_output=True, text=True, timeout=1800, env=self._env())
        if p.returncode != 0:
            st.gates = gates + [("transfer", False,
                                 (p.stderr or p.stdout).strip()[:400])]
            return st
        gates.append(("transfer", True,
                      f"{nbytes/1024:.1f} KB streamed over the tunnel"))

        # delete-sync for the incremental case: the manifest diff names exactly
        # which remote files no longer exist locally, and only those are pruned.
        # Removing the whole workdir instead would throw away the incremental
        # claim the user asked for.
        if plan.deleted and not wipe:
            rel = [os.path.join(self.remote_dir, "payload", d) for d in plan.deleted]
            rm = self.ssh("rm -f -- " + " ".join(shlex.quote(x) for x in rel),
                          timeout=120)
            gates.append(("delete-sync", rm.rc == 0,
                          f"pruned {len(plan.deleted)} remote file(s)"
                          + ("" if rm.rc == 0 else f": {rm.short(200)}")))

        # The integrity gate, run against the machine rather than the manifest,
        # because on this backend the remote can be a *different* machine from
        # the last one and a local baseline proves nothing about it.
        g, missing, extra = self._verify_remote(tree)
        gates.append(g)
        if not g.ok:
            st.gates = gates
            return st

        self.store.mark_pending(tree, {"host": self.host, "remote_dir": self.remote_dir,
                                       "sha256": digest, "bytes": nbytes},
                               note="streamed and verified on the box")
        st.ok = True
        st.transferred = True
        st.files = len(tree)
        st.bytes_sent = nbytes
        st.remote = {"host": self.host, "remote_dir": self.remote_dir,
                     "sha256": digest, "fingerprint": tree.fingerprint}
        st.note = f"verified on {self.host}; baseline pending until a run completes"
        st.gates = gates
        return st

    def _verify_remote(self, tree: Manifest) -> Gate:
        # sha256sum of every file in the remote payload root, in one round trip.
        names = " ".join(shlex.quote(e.path) for e in tree.sorted_entries())
        if not names:
            return Gate("remote integrity", True, "empty manifest")
        cmd = (f"cd {shlex.quote(self.remote_dir)}/payload && "
               f"sha256sum -- {names} 2>&1")
        p = self.ssh(cmd, timeout=300)
        if p.rc != 0:
            return Gate("remote integrity", False,
                        f"could not hash the remote payload: {p.short(300)}")
        actual = {}
        for line in p.out.splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2 and len(parts[0]) == 64:
                actual[parts[1].strip().lstrip("*")] = parts[0]
        expected = {e.path: e.sha256 for e in tree}
        g, missing, extra = payload_completeness(expected, actual, "remote box")
        return g

    def describe_target(self) -> dict:
        return {"host": self.host, "remote_dir": self.remote_dir}

    def status(self) -> RemoteStatus:
        if not self.host:
            return RemoteStatus(phase="NO HOST",
                                detail="pass --host; the tunnel hostname is printed "
                                       "by remote/colab_ssh.py and changes per session")
        d = self.device()
        if "not found" in str(d.get("reason", "")):
            return RemoteStatus(phase="UNKNOWN", detail=str(d.get("reason")))
        return RemoteStatus(phase="IDLE", device=d,
                            detail=f"{d.get('name') or 'no GPU'} on {self.host}; "
                                   f"remote workdir {self.remote_dir}")

    def device(self) -> dict:
        """Read the device off the machine. Never assume, never cache."""
        if not self.host:
            return {"gpu": False, "reason": "no --host given"}
        p = self.ssh(f"python3 -c {shlex.quote(DEVICE_PROBE)}", timeout=120)
        return parse_device(p.out + p.err)

    # -------------------------------------------------------------- RUN

    def doctor(self) -> Report:
        rep = Report(ok=True, costs_gpu=self.spends_gpu)
        checks = []
        for tool in ("ssh", "tar", "scp"):
            checks.append((tool, bool(util.which(tool)),
                           util.which(tool) or "not found"))
        rep.ok = all(c[1] for c in checks)
        if not rep.ok:
            rep.checks = checks
            rep.note = "macOS ships ssh/tar/scp; nothing to install"
            return rep
        checks.append(("auth", bool(self.password or self.askpass_cmd or self.identity),
                       "SSH_ASKPASS helper (no sshpass needed)"
                       if (self.password or self.askpass_cmd)
                       else (f"key {self.identity}" if self.identity else "none given")))
        if not self.host:
            checks.append(("host", False, "no --host; run the remote/colab_ssh.py "
                                          "bootstrap on the Colab side and paste the host"))
            rep.ok = False
            rep.checks = checks
            rep.note = ("UNTESTED end to end: gpush's Colab path has never been run "
                        "against a live Colab runtime. Treat `doctor` as a checklist, "
                        "not as a verification.")
            return rep
        p = self.ssh("echo alive", timeout=60)
        checks.append(("reach the box", p.rc == 0,
                       p.out.strip()[:80] if p.rc == 0 else p.short(200)))
        rep.ok = p.rc == 0
        d = self.device()
        checks.append(("device", bool(d.get("gpu")),
                       (f"{d.get('name')}  {d.get('memory_mb', '?')} MB  sm_{d.get('compute_cap', '?')}"
                        if d.get("gpu") else f"NO GPU: {d.get('reason')}")))
        if not d.get("gpu"):
            rep.ok = False
        checks.append(("GPU cost", True, "`run` spends the session; `push`/`status` "
                                          "do not"))
        rep.checks = checks
        if not d.get("gpu"):
            rep.note = ("the session reports no GPU. That is a normal state for a "
                        "Colab free session and it is exactly why gpush asks the "
                        "machine instead of assuming T4.")
        return rep

    def run(self, tree: Manifest, argv: list[str], **kw) -> Run:
        r = Run(ok=False)
        r.started = util.stamp()
        allow_cpu = bool(self.opts.get("allow_cpu"))
        d = self.device()
        r.device = d
        # The gate: refuse to spend a session, and refuse to record a number, on
        # a machine that has no GPU -- unless the user asked for a CPU result
        # to be labelled as one.
        g = device_is_gpu(d, require_gpu=not allow_cpu)
        if not g.ok:
            r.status = "REFUSED"
            r.note = g.detail + ". " + g.remedy
            return r
        r.log = g.detail

        quoted = " ".join(shlex.quote(a) for a in argv)
        # Bounded and streamed: a disposable box can vanish mid-run, so the
        # command is wrapped in `timeout` and its output is teed to a file on
        # the box so that whatever completed before a disconnect is still
        # fetchable.
        inner = (f"cd {shlex.quote(self.remote_dir)}/payload && "
                 f"timeout {int(kw.get('timeout', self.timeout))} "
                 f"bash -lc {shlex.quote(quoted)} 2>&1 | tee "
                 f"{shlex.quote(self.remote_dir)}/run.log; "
                 f"echo GPUSH_EXIT=${{PIPESTATUS[0]}}")
        p = self.ssh(inner, timeout=int(kw.get("timeout", self.timeout)) + 120)
        r.log += "\n" + p.out
        r.finished = util.stamp()
        code = None
        for line in p.out.splitlines():
            if line.startswith("GPUSH_EXIT="):
                code = int(line.split("=", 1)[1] or 1)
        r.exit_code = code
        r.run_id = f"colab-{self.started}"
        r.ok = (p.rc == 0 and code == 0)
        r.status = "COMPLETE" if r.ok else ("DISCONNECTED" if code is None else "ERROR")
        if code is None and p.rc != 0:
            r.note = ("the session went away mid-run; the box is disposable and may "
                      "have been reclaimed. Whatever reached run.log was fetched; "
                      "re-run against a fresh session.")
        return r

    # ------------------------------------------------------------- OUT

    def fetch(self, state: dict, dest: str) -> list[str]:
        import subprocess
        os.makedirs(dest, exist_ok=True)
        stage = os.path.join(dest, ".partial")
        os.rmtree(stage, ignore_errors=True)
        os.makedirs(stage, exist_ok=True)
        # tar the remote's run log and artefacts back in one stream.
        src = f"{self.remote_dir}/run.log {self.remote_dir}/payload"
        argv = ["ssh", "-p", str(self.port), "-o", "StrictHostKeyChecking=accept-new",
                "-o", "ConnectTimeout=20", self.target(),
                f"tar -czf - --ignore-failed-read {src}"]
        if self.identity:
            argv[1:1] = ["-i", self.identity]
        with open(os.path.join(stage, "artefacts.tgz"), "wb") as f:
            p = subprocess.run(argv, stdout=f, stderr=subprocess.PIPE, text=True,
                               timeout=self.timeout, env=self._env())
        written = []
        tgz = os.path.join(stage, "artefacts.tgz")
        if os.path.getsize(tgz) > 0:
            import tarfile
            try:
                with tarfile.open(tgz, "r:gz") as tf:
                    tf.extractall(dest, filter="data")
                written.append(tgz)
            except (OSError, tarfile.TarError):
                pass
        os.remove(tgz) if os.path.exists(tgz) else None
        return written
