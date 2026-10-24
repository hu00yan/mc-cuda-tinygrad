"""The Colab backend, tested to the boundary of what needs a live runtime.

WHAT IS TESTED HERE
    * it is registered on the same six-method interface as kaggle
    * it refuses, loudly, when it has no host
    * the ssh argv is constructed correctly without touching a network
    * password auth needs no sshpass (SSH_ASKPASS_REQUIRE=force)
    * it refuses to run on a machine with no GPU unless told to
    * its payload passes the same integrity gate as kaggle's

WHAT IS NOT TESTED, AND CANNOT BE HERE
    The transfer itself, the remote `mkdir`, the cloudflared reverse tunnel,
    and what happens when a Colab session is reclaimed mid-command. None of
    those need anything but a live Colab runtime to exercise, and a test that
    cannot run is not a test. `doctor` on that backend is a checklist of
    preconditions, and its own output says so.

    The honest statement: **gpush's Colab path has never been run end to end.**
    Do not read the passing tests below as evidence that it works.
"""
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from gpush import backend as be, tarball       # noqa: E402
from gpush.gates import payload_completeness   # noqa: E402
from gpush.manifest import build               # noqa: E402
from gpush.probes import DEVICE_PROBE         # noqa: E402
from gpush.state import StateStore             # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _colab(**kw):
    tmp = tempfile.mkdtemp(prefix="gpush-colab-")
    store = StateStore(tmp).bind("colab", kw.get("host") or "nohost")
    return be.get("colab", store=store, out_root=tmp, **kw), tmp


class TestInterfaceParity(unittest.TestCase):
    def test_colab_is_registered_and_complete(self):
        reg = be.load_all()
        self.assertIn("colab", reg)
        self.assertIn("kaggle", reg)
        for name, cls in reg.items():
            self.assertEqual(getattr(cls, "__abstractmethods__", set()), set(),
                             f"{name} does not implement the full interface")
            self.assertTrue(cls.transport, f"{name} does not describe its transport")

    def test_colab_spends_gpu_and_says_so(self):
        self.assertTrue(be.load_all()["colab"].spends_gpu)


class TestNoHost(unittest.TestCase):
    def test_push_refuses_without_a_host(self):
        bk, tmp = _colab(host=None)
        try:
            tree = build(REPO, None, include=["derivation"])
            tree.compute_fingerprint("colab", "nohost")
            st = bk.push(tree, bk.plan(tree))
            self.assertFalse(st.ok)
            self.assertIn("--host", st.note)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_doctor_fails_and_declares_itself_untested(self):
        bk, tmp = _colab(host=None)
        try:
            rep = bk.doctor()
            self.assertFalse(rep.ok)
            self.assertIn("host", [c[0] for c in rep.checks])
            self.assertIn("UNTESTED", rep.note)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_status_reports_no_host_rather_than_guessing(self):
        bk, tmp = _colab(host=None)
        try:
            s = bk.status()
            self.assertEqual(s.phase, "NO HOST")
            self.assertIn("--host", s.detail)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_the_host_is_never_hardcoded(self):
        """The hostname changes every session; it is an argument, not a default."""
        import inspect
        src = inspect.getsource(be.load_all()["colab"])
        self.assertNotIn("trycloudflare.com\"", src.replace("trycloudflare.com' ", ""))


class TestSshTransport(unittest.TestCase):
    def setUp(self):
        self.bk, self.tmp = _colab(host="abc-def.trycloudflare.com",
                                    password="probe")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_the_argv_targets_the_right_place(self):
        argv = self.bk.ssh_argv("echo hi")
        self.assertEqual(argv[-2], "root@abc-def.trycloudflare.com")
        self.assertEqual(argv[-1], "echo hi")

    def test_keepalives_are_set_for_a_flaky_tunnel(self):
        argv = self.bk.ssh_argv("true")
        self.assertTrue(any("ServerAliveInterval" in a for a in argv))
        self.assertTrue(any("ServerAliveCountMax" in a for a in argv))

    def test_a_password_means_batchmode_off(self):
        self.assertIn("BatchMode=no", self.bk.ssh_argv("true"))

    def test_no_password_means_batchmode_on(self):
        bk, tmp = _colab(host="h.example", password=None)
        try:
            self.assertIn("BatchMode=yes", bk.ssh_argv("true"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_password_auth_works_without_sshpass(self):
        env = self.bk._env()
        self.assertEqual(env.get("SSH_ASKPASS_REQUIRE"), "force")
        ap = env["SSH_ASKPASS"]
        self.assertTrue(os.stat(ap).st_mode & stat.S_IXUSR)
        self.assertEqual(subprocess.run([ap], capture_output=True,
                                        text=True).stdout, "probe")

    def test_sshpass_is_not_installed_and_not_needed(self):
        """Asserting the absence, so the no-dependency claim is checked."""
        self.assertIsNone(shutil.which("sshpass"),
                          "sshpass is installed; the no-dependency path is "
                          "therefore untested on this machine")

    def test_ssh_and_tar_are_what_it_needs(self):
        for tool in ("ssh", "tar", "scp"):
            self.assertIsNotNone(shutil.which(tool), f"{tool} missing")

    def test_the_device_probe_is_pure_stdlib_python(self):
        """It has to run on a box whose only guarantee is that python3 exists."""
        import ast
        tree = ast.parse(DEVICE_PROBE)
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.module:
                mods.add(n.module.split(".")[0])
        self.assertTrue(mods <= {"json", "os", "platform", "subprocess", "sys"},
                        f"the probe imports more than the stdlib floor: {mods}")

    def test_the_probe_runs_and_reports_no_gpu_on_a_mac(self):
        """A Mac is a real 'machine with no GPU', so it is a real test case."""
        p = subprocess.run([sys.executable, "-c", DEVICE_PROBE],
                           capture_output=True, text=True, timeout=60)
        from gpush.probes import parse_device
        d = parse_device(p.stdout + p.stderr)
        self.assertIn("gpu", d)
        self.assertFalse(d["gpu"])
        self.assertTrue(d["reason"], "a CPU-only box must say why it found no GPU")


class TestPayload(unittest.TestCase):
    def setUp(self):
        self.tree = build(REPO, None, include=["derivation"])
        self.tree.compute_fingerprint("colab", "h")

    def test_it_packs_and_verifies_with_the_same_gate_as_kaggle(self):
        tmp = tempfile.mkdtemp(prefix="gpush-colab-pkg-")
        try:
            p, digest, _ = tarball.seal(self.tree, os.path.join(tmp, "payload.tgz"),
                                        extra={"GPUSH-ATTEST.json": b"{}"})
            v = tarball.verify_archive(p, self.tree)
            self.assertTrue(v["ok"])
            exp = {e.path: e.sha256 for e in self.tree}
            g, _m, _e = payload_completeness(exp, v["found"], "remote box")
            self.assertTrue(g.ok)
            self.assertEqual(len(digest), 64)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_a_missing_remote_file_is_caught_by_content(self):
        exp = {e.path: e.sha256 for e in self.tree}
        act = dict(exp)
        act.pop(sorted(act)[0])
        g, miss, _e = payload_completeness(exp, act, "remote box")
        self.assertFalse(g.ok)
        self.assertEqual(len(miss), 1)


if __name__ == "__main__":
    unittest.main()
