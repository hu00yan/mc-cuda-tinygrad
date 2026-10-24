#!/usr/bin/env python3
"""gpush/tests/test_deletion_is_visible.py -- a deletion must change the fingerprint.

Reported from the field: `gpush plan` printed "no changes" while a *superseded*
module was still what would run. A content-addressed manifest sums per-file
hashes, so a deletion removes a line from that sum and the fingerprint must
change. This test asks the question directly rather than trusting the design.

Why the question is worth asking at all: every measurement in this project runs
on the far side through this tool, and a run that silently executes a deleted
module produces a number that looks fine and describes code that no longer
exists. That is the same shape as the stale-payload defect
(`test_stale_payload.py`) and as `DECISIONS.md` #18 -- a success signal for work
that was not the work.

The mechanism of the reported failure is NOT established. It was reported as
`gpush plan` saying "no changes", which this test does not reproduce; if the
deletion does move the fingerprint, the mechanism lies elsewhere and the honest
next step is to get the failing case rather than to guess at a fix.

Run:  python gpush/tests/test_deletion_is_visible.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpush.manifest import build  # noqa: E402


class TestDeletionIsVisible(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="gpush-del-")
        os.makedirs(os.path.join(self.tmp, "pkg"))
        for name, body in (("pkg/keep.py", "KEEP = 1\n"),
                           ("pkg/gone.py", "GONE = 1\n"),
                           ("pkg/also.py", "ALSO = 1\n")):
            with open(os.path.join(self.tmp, name), "w") as f:
                f.write(body)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _fp(self):
        m = build(self.tmp, None, include=["pkg"])
        m.compute_fingerprint("test-backend", "test-target")
        return m.fingerprint, m

    def test_deleting_a_file_changes_the_fingerprint(self):
        before, mb = self._fp()
        self.assertIn("pkg/gone.py", mb.entries)
        os.remove(os.path.join(self.tmp, "pkg/gone.py"))
        after, ma = self._fp()
        self.assertNotIn("pkg/gone.py", ma.entries)
        self.assertNotEqual(before, after,
                            "a deletion must move the fingerprint, or a removed "
                            "module is indistinguishable from an unchanged tree")

    def test_deleting_a_file_shrinks_the_manifest(self):
        before, _ = self._fp()
        os.remove(os.path.join(self.tmp, "pkg/gone.py"))
        after, ma = self._fp()
        self.assertEqual(len(ma.entries), 2)
        self.assertNotEqual(before, after)

    def test_deleting_everything_still_moves_the_fingerprint(self):
        before, _ = self._fp()
        for name in ("keep.py", "also.py", "gone.py"):
            os.remove(os.path.join(self.tmp, "pkg", name))
        after, ma = self._fp()
        self.assertEqual(len(ma.entries), 0)
        self.assertNotEqual(before, after,
                            "an empty tree must not collide with any non-empty one")

    def test_deleting_two_of_three_leaves_one_and_still_moves(self):
        """Regression guard for the test above's own first draft.

        An earlier version deleted two files and asserted `len(entries) == 0`,
        which failed with `1 != 0` -- correctly. The third file was still there.
        Recorded because a test that asserts the wrong count and then has its
        assertion edited to match is a test that was never checking anything.
        """
        before, mb = self._fp()
        self.assertEqual(len(mb.entries), 3)
        for name in ("keep.py", "also.py"):
            os.remove(os.path.join(self.tmp, "pkg", name))
        after, ma = self._fp()
        self.assertEqual(len(ma.entries), 1, "gone.py is the one that survives")
        self.assertIn("pkg/gone.py", ma.entries)
        self.assertNotEqual(before, after)

    def test_deleting_one_file_is_not_the_same_as_deleting_another(self):
        """A rename must move the fingerprint for a reason, not by coincidence."""
        a, _ = self._fp()
        os.remove(os.path.join(self.tmp, "pkg/gone.py"))
        b, _ = self._fp()
        os.remove(os.path.join(self.tmp, "pkg/also.py"))
        c, _ = self._fp()
        self.assertEqual(len({a, b, c}), 3, "three distinct trees, three fingerprints")


if __name__ == "__main__":
    unittest.main(verbosity=2)
