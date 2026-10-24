"""Every guard and gate must REFUSE, not merely report.

A guard that passes when it should refuse is worse than no guard, because the
user is told the payload was checked. Each test here asserts the refusal AND
the override, so neither the guard nor its escape hatch can rot unnoticed.

Offline. No Kaggle quota, no network.
"""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from gpush import tarball                                  # noqa: E402
from gpush.backend import slugify, title_from_slug         # noqa: E402
from gpush.gates import (device_is_gpu, mass_deletion,     # noqa: E402
                         payload_completeness, no_changes)
from gpush.manifest import build, diff                    # noqa: E402
from gpush.probes import capability_note, parse_device     # noqa: E402
from gpush.state import StateStore                         # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTitleRoundTrip(unittest.TestCase):
    """Kaggle slugs the TITLE, not the id, and 409s when they disagree."""

    def test_derived_title_slugs_back_to_the_slug(self):
        for slug in ("gpush-smoke", "mc-minecu-t4-baseline",
                     "mc-cuda-tinygrad-probe-battery", "a", "x1-y2-z3"):
            self.assertEqual(slugify(title_from_slug(slug)), slug)

    def test_a_hand_written_title_is_what_breaks(self):
        # The shape that produced "Your kernel title does not resolve to the
        # specified id" in minecu_ref/kaggle/make_notebook.py: a title that is
        # not the slug with hyphens turned into spaces.
        self.assertNotEqual(slugify("Probe Battery"),
                            "mc-cuda-tinygrad-probe-battery")
        # underscores are safe: the rule turns any non-alphanumeric run into
        # a hyphen, and a title that differs from the slug in some other way
        # is what the 409 is actually about.
        self.assertEqual(slugify("mc_cuda_tinygrad probe battery"),
                         "mc-cuda-tinygrad-probe-battery")


class TestMassDeletionGuard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="gpush-md-")
        for i in range(20):
            with open(os.path.join(self.tmp, f"f{i}.txt"), "w") as f:
                f.write("x" * 64)
        self.small = build(self.tmp, None)
        self.small.compute_fingerprint("k", "t")
        self.big = build(REPO, None, include=["derivation"])
        self.big.compute_fingerprint("k", "t")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_large_prune_refuses(self):
        p = diff(self.small, self.big)      # 20 local files -> 20 remote gone
        self.assertGreater(p.delete_fraction, 0.30)
        self.assertFalse(mass_deletion(p, 0.30, False).ok)

    def test_the_override_passes_and_says_it_was_overridden(self):
        p = diff(self.small, self.big)
        g = mass_deletion(p, 0.30, True)
        self.assertTrue(g.ok)
        self.assertIn("OVERRIDDEN", g.detail)

    def test_no_deletion_passes_quietly(self):
        g = mass_deletion(diff(self.big, self.big), 0.30, False)
        self.assertTrue(g.ok)
        self.assertNotIn("OVERRIDDEN", g.detail)

    def test_a_refusal_names_the_files_and_a_fix(self):
        p = diff(self.small, self.big)
        g = mass_deletion(p, 0.30, False)
        self.assertIn("allow-mass-delete", g.remedy)


class TestPayloadIntegrityGate(unittest.TestCase):
    """The two incidents: an empty payload, and a stale file in it."""

    def setUp(self):
        self.tree = build(REPO, None, include=["derivation"])
        self.tree.compute_fingerprint("k", "t")
        self.exp = {e.path: e.sha256 for e in self.tree}

    def test_honest_payload_passes(self):
        g, miss, extra = payload_completeness(self.exp, dict(self.exp), "t")
        self.assertTrue(g.ok)
        self.assertEqual((miss, extra), ([], []))

    def test_empty_payload_refuses(self):
        """DECISIONS.md #18: the kernel ran against nothing and said 'skip'."""
        g, miss, extra = payload_completeness(self.exp, {}, "t")
        self.assertFalse(g.ok)
        self.assertEqual(sorted(miss), sorted(self.exp))

    def test_a_stale_extra_file_refuses(self):
        """A payload file nobody asked for is code that is not the code."""
        act = dict(self.exp, **{"derivation/GHOST.py": "0" * 64})
        g, miss, extra = payload_completeness(self.exp, act, "t")
        self.assertFalse(g.ok)
        self.assertEqual(extra, ["derivation/GHOST.py"])

    def test_differing_bytes_refuse(self):
        act = dict(self.exp)
        k = sorted(self.exp)[0]
        act[k] = "f" * 64
        g, _m, _e = payload_completeness(self.exp, act, "t")
        self.assertFalse(g.ok)

    def test_the_gate_is_not_vacuous_on_a_real_archive(self):
        tmp = tempfile.mkdtemp(prefix="gpush-arc-")
        try:
            tgz = os.path.join(tmp, "p.tgz")
            tarball.seal(self.tree, tgz, extra={"GPUSH-ATTEST.json": b"{}"})
            v = tarball.verify_archive(tgz, self.tree)
            self.assertTrue(v["ok"])
            self.assertEqual(v["checked"], len(self.exp))

            t2 = os.path.join(tmp, "q.tgz")
            tarball.seal(self.tree, t2,
                         extra={"GPUSH-ATTEST.json": b"{}", "STOWAWAY.py": b"x"})
            v2 = tarball.verify_archive(t2, self.tree)
            self.assertFalse(v2["ok"])
            self.assertEqual(v2["extra"], ["STOWAWAY.py"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_the_attestation_records_the_fingerprint(self):
        a = tarball.attestation(self.tree, "k", "t", command=["x"])
        import json
        d = json.loads(a)
        self.assertEqual(d["fingerprint"], self.tree.fingerprint)
        self.assertEqual(d["file_count"], len(self.tree))
        self.assertEqual(d["command"], ["x"])


class TestDeviceGate(unittest.TestCase):
    T4 = {"gpu": True, "name": "Tesla T4", "memory_mb": 15360,
          "compute_cap": "7.5", "driver": "580.159.04", "cuda_runtime": "13.0"}
    CPU = {"gpu": False, "reason": "nvidia-smi ran and reported no device"}

    def test_cpu_only_refuses(self):
        self.assertFalse(device_is_gpu(self.CPU, True).ok)

    def test_allow_cpu_permits_and_says_so(self):
        g = device_is_gpu(self.CPU, False)
        self.assertTrue(g.ok)
        self.assertIn("allow-cpu", g.detail)

    def test_a_gpu_passes(self):
        self.assertTrue(device_is_gpu(self.T4, True).ok)

    def test_capability_is_reported_not_interpreted(self):
        note = capability_note(self.T4)
        self.assertIn("sm_75", note)
        # it may note the sm<80 fact, but it must not reach a verdict about
        # whether the work can be done
        self.assertNotIn("cannot", note.lower())
        self.assertNotIn("incompatible", note.lower())

    def test_an_unparseable_probe_is_unknown_not_cpu(self):
        d = parse_device("the box produced nothing recognisable")
        self.assertFalse(d.get("gpu"))
        self.assertIn("no parseable output", d["reason"])


class TestNoChangesGate(unittest.TestCase):
    """The no-op decision reports itself; it is not a verdict."""

    def test_a_skip_explains_itself(self):
        self.assertIn("proven", no_changes(True, "+0 ~0 -0 =5", "proven by a run"))

    def test_a_transfer_says_what_it_sends(self):
        out = no_changes(False, "+3 ~1 -0 =4", "12.0 KB to send")
        self.assertIn("+3 ~1 -0 =4", out)
        self.assertIn("12.0 KB", out)

    def test_a_pending_baseline_is_explained_rather_than_silent(self):
        out = no_changes(False, "+0 ~0 -0 =5",
                         "baseline is PENDING: never attested by a run")
        self.assertIn("PENDING", out)

    def test_it_returns_text_not_a_verdict(self):
        """A Gate here would either refuse the skip or hide a real refusal."""
        self.assertIsInstance(no_changes(True, "s", "r"), str)
        self.assertIsInstance(no_changes(False, "s", "r"), str)


class TestBaselineStateMachine(unittest.TestCase):
    """The rule that makes the lost run impossible to repeat."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="gpush-state-")
        self.store = StateStore(self.tmp).bind("kaggle", "o/s")
        self.tree = build(REPO, None, include=["derivation"])
        self.tree.compute_fingerprint("kaggle", "o/s")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_pushed_but_unrun_baseline_is_not_proven(self):
        self.store.mark_pending(self.tree, {"dataset": "o/s"}, "uploaded")
        self.assertFalse(self.store.is_proven(self.tree.fingerprint))
        self.assertIn("PENDING", self.store.why_not_proven(self.tree.fingerprint))
        self.assertIn("#18", self.store.why_not_proven(self.tree.fingerprint))

    def test_only_a_completed_run_proves_it(self):
        self.store.mark_pending(self.tree, {}, "uploaded")
        self.store.mark_confirmed(self.tree, {"dataset": "o/s"}, "run-1")
        self.assertTrue(self.store.is_proven(self.tree.fingerprint))

    def test_a_different_tree_is_never_proven_by_the_old_baseline(self):
        self.store.mark_confirmed(self.tree, {}, "run-1")
        other = build(REPO, None, include=["kda"])
        other.compute_fingerprint("kaggle", "o/s")
        self.assertFalse(self.store.is_proven(other.fingerprint))

    def test_a_failed_run_leaves_the_baseline_unproven(self):
        self.store.mark_pending(self.tree, {}, "uploaded")
        self.store.note_run("run-2", self.tree.fingerprint, "ERROR")
        self.assertFalse(self.store.is_proven(self.tree.fingerprint))
        self.assertEqual(self.store.state, "pending")

    def test_a_corrupt_baseline_degrades_to_no_baseline(self):
        self.store.mark_confirmed(self.tree, {}, "run-1")
        with open(self.store.path, "w") as f:
            f.write("{ this is not json")
        s2 = StateStore(self.tmp).bind("kaggle", "o/s")
        self.assertEqual(s2.state, "absent")
        self.assertFalse(s2.is_proven(self.tree.fingerprint))

    def test_full_resync_invalidates(self):
        self.store.mark_confirmed(self.tree, {}, "run-1")
        self.store.invalidate("--full-resync")
        self.assertEqual(self.store.state, "absent")
        self.assertFalse(self.store.is_proven(self.tree.fingerprint))


class TestDeterminism(unittest.TestCase):
    def test_an_unchanged_tree_packs_to_identical_bytes(self):
        """gzip stamps the clock unless you pass mtime=0. This catches that."""
        import time
        tmp = tempfile.mkdtemp(prefix="gpush-det-")
        try:
            tree = build(REPO, None, include=["derivation"])
            tree.compute_fingerprint("k", "t")
            a = tarball.attestation(tree, "k", "t")
            d = []
            for i in range(2):
                p = os.path.join(tmp, f"{i}.tgz")
                _, dg, _ = tarball.seal(tree, p, extra={"GPUSH-ATTEST.json": a})
                d.append(dg)
                if i == 0:
                    time.sleep(1.1)   # cross a 1-second gzip timestamp tick
            self.assertEqual(d[0], d[1])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
