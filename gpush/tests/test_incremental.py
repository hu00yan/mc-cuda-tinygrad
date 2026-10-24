"""Incremental sync: the fingerprint must move only when the bytes move.

These tests pin down both directions, because a fingerprint that is too eager
costs an upload every time and a fingerprint that is too lazy silently ships
stale code -- the second is the one that corrupts a measurement.

Also pins the known limitation of the (size, mtime) shortcut, so that the
documented reason `--checksum` exists stays true.

    Offline. No Kaggle quota, no network.
"""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from gpush.manifest import build, diff    # noqa: E402


class TreeFixture:
    """A throwaway working tree the tests mutate freely."""

    def __init__(self, n=8):
        self.dir = tempfile.mkdtemp(prefix="gpush-inc-")
        os.makedirs(os.path.join(self.dir, "derivation"))
        for i in range(n):
            self.write(f"f{i}.py", f"# file {i}\n" * 50)

    def write(self, name, text):
        with open(os.path.join(self.dir, "derivation", name), "w") as f:
            f.write(text)

    def path(self, name):
        return os.path.join(self.dir, "derivation", name)

    def remove(self, name):
        os.remove(self.path(name))

    def manifest(self, baseline=None, checksum=False):
        m = build(self.dir, baseline, include=["derivation"], checksum=checksum)
        m.compute_fingerprint("k", "t")
        return m

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)


class TestIncremental(unittest.TestCase):
    def setUp(self):
        self.t = TreeFixture()
        self.base = self.t.manifest()

    def tearDown(self):
        self.t.close()

    def test_a_cold_tree_is_all_additions(self):
        p = diff(None, self.base)
        self.assertEqual(len(p.added), 8)
        self.assertEqual(p.deleted, [])
        self.assertEqual(p.modified, [])

    def test_nothing_touched_produces_an_empty_plan(self):
        p = diff(self.base, self.t.manifest(self.base))
        self.assertTrue(p.empty)
        self.assertEqual(len(p.unchanged), 8)

    def test_hashes_are_reused_when_nothing_changed(self):
        m = self.t.manifest(self.base)
        self.assertEqual(m.hashed, 0, "an unchanged file was re-read")
        self.assertEqual(m.reused_hashes, 8)

    def test_one_edit_moves_one_file(self):
        self.t.write("f3.py", "# file 3\n" * 50 + "# edited\n")
        p = diff(self.base, self.t.manifest(self.base))
        self.assertEqual(p.modified, ["derivation/f3.py"])
        self.assertEqual(len(p.unchanged), 7)
        self.assertEqual(p.added, [])

    def test_one_edit_rehashes_only_that_file(self):
        self.t.write("f3.py", "# file 3\n" * 50 + "# edited\n")
        m = self.t.manifest(self.base)
        self.assertEqual(m.hashed, 1)
        self.assertEqual(m.reused_hashes, 7)

    def test_an_identical_rewrite_is_not_a_change(self):
        self.t.write("f3.py", "# file 3\n" * 50)     # same bytes, new mtime
        m = self.t.manifest(self.base)
        self.assertEqual(m.fingerprint, self.base.fingerprint)
        self.assertTrue(diff(self.base, m).empty)

    def test_a_deletion_is_a_deletion(self):
        self.t.remove("f7.py")
        p = diff(self.base, self.t.manifest(self.base))
        self.assertEqual(p.deleted, ["derivation/f7.py"])
        self.assertAlmostEqual(p.delete_fraction, 1 / 8)

    def test_total_bytes_is_right_even_when_hashes_are_reused(self):
        m = self.t.manifest(self.base)
        self.assertEqual(m.total_bytes, sum(e.size for e in m))
        self.assertGreater(m.total_bytes, 0,
                           "a reused-hash manifest reported zero bytes")

    def test_a_new_file_changes_the_fingerprint(self):
        self.t.write("f8.py", "# new\n")
        m = self.t.manifest(self.base)
        self.assertNotEqual(m.fingerprint, self.base.fingerprint)


class TestChecksumMode(unittest.TestCase):
    def setUp(self):
        self.t = TreeFixture()
        self.base = self.t.manifest()

    def tearDown(self):
        self.t.close()

    def test_checksum_rereads_everything(self):
        m = self.t.manifest(self.base, checksum=True)
        self.assertEqual(m.hashed, 8)
        self.assertEqual(m.reused_hashes, 0)
        self.assertEqual(m.fingerprint, self.base.fingerprint)

    def test_checksum_catches_a_same_size_same_mtime_edit(self):
        """The reason --checksum exists, asserted rather than asserted-about.

        A `git checkout` of an old revision, `touch -r`, or a copied file can
        preserve both size and mtime while changing the bytes. The shortcut
        cannot see it. --checksum can. If this test ever starts failing the
        shortcut got smarter; if it starts passing, --checksum is pointless.
        """
        p = self.t.path("f5.py")
        st = os.stat(p)
        with open(p) as f:
            data = f.read()
        with open(p, "w") as f:
            f.write(data.replace("# file 5", "# FILE 5"))   # identical length
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))

        fast = self.t.manifest(self.base)
        self.assertEqual(fast.fingerprint, self.base.fingerprint,
                         "the (size,mtime) shortcut is supposed to miss this")

        slow = self.t.manifest(self.base, checksum=True)
        self.assertNotEqual(slow.fingerprint, self.base.fingerprint,
                            "--checksum is supposed to catch this")


class TestFingerprintBinding(unittest.TestCase):
    def setUp(self):
        self.t = TreeFixture(3)
        self.m = self.t.manifest()

    def tearDown(self):
        self.t.close()

    def test_the_fingerprint_is_bound_to_the_backend(self):
        a = self.t.manifest(); a.compute_fingerprint("kaggle", "o/s")
        b = self.t.manifest(); b.compute_fingerprint("colab", "o/s")
        self.assertNotEqual(a.fingerprint, b.fingerprint)

    def test_the_fingerprint_is_bound_to_the_target(self):
        a = self.t.manifest(); a.compute_fingerprint("kaggle", "o/one")
        b = self.t.manifest(); b.compute_fingerprint("kaggle", "o/two")
        self.assertNotEqual(a.fingerprint, b.fingerprint)

    def test_the_fingerprint_is_stable(self):
        a = self.t.manifest(); a.compute_fingerprint("kaggle", "o/s")
        b = self.t.manifest(); b.compute_fingerprint("kaggle", "o/s")
        self.assertEqual(a.fingerprint, b.fingerprint)


class TestGitTrackedAndUntracked(unittest.TestCase):
    """The manifest must be the working tree, not the index."""

    REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def test_it_comes_from_git_when_there_is_a_git_checkout(self):
        m = build(self.REPO, None, include=["derivation"])
        self.assertEqual(m.source, "git")
        # Assert the PROPERTY, not the file set. This test used to pin an exact
        # list, which made it a change-detector: adding any legitimate script to
        # derivation/ turned it red, and the only way to make it green was to
        # paste the new name in without reading the failure. A test that must be
        # updated whenever the tree grows is a test that stops being read. What
        # actually matters is that git sourced the list and that known-present
        # files are in it -- so pin those, and let additions pass.
        paths = set(m.entries)
        for expected in ("derivation/divide.py", "derivation/divide_backends.py",
                         "derivation/_util.py", "derivation/_fp32.py",
                         "derivation/kda_decay.py"):
            self.assertIn(expected, paths,
                          f"git-tracked file missing from the manifest: {expected}")
        self.assertTrue(all(p.startswith("derivation/") for p in paths),
                        "include= must scope the manifest, not merely filter it")
        # and it is the working tree, so an untracked file in scope is present
        self.assertNotIn("derivation/__pycache__", paths)

    def test_gitignored_build_output_is_excluded(self):
        m = build(self.REPO, None)
        paths = set(m.entries)
        self.assertNotIn(".venv/bin/kaggle", paths)
        self.assertFalse(any(p.startswith(".venv/") for p in paths))
        self.assertFalse(any("__pycache__" in p for p in paths))

    def test_an_include_list_is_a_contract_not_a_filter(self):
        from gpush.manifest import uncovered
        m = build(self.REPO, None, include=["derivation"])
        missed = uncovered(self.REPO, m, ["derivation"])
        self.assertGreater(len(missed), 0)
        self.assertTrue(all(not p.startswith("derivation/") for p in missed))


if __name__ == "__main__":
    unittest.main()
