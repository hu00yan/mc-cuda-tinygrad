#!/usr/bin/env python3
"""gpush/tests/test_stale_payload.py -- a MATCHED verdict is not a provenance claim.

The runner's `GPUSH-VERIFY.json` reports `ok: true` when the payload it received
matches the manifest carried *inside that same payload*. That is
self-consistency, not identity. A cached notebook carries the previous run's
`PAYLOAD_FINGERPRINT` and the previous run's manifest, so the two agree with each
other, the verdict is `MATCHED`, and the previous run's bytes execute.

This is the DECISIONS.md #18 shape reached by a different route, and it was hit
for real: a 9-second `COMPLETE` whose `GPUSH-VERIFY.json` reported the prior
fingerprint (`out/gpush/runs/kaggle/ut-time-t4/`). The run of record was
self-consistent at every level and still void, which is exactly why a
self-consistent check is the wrong check.

These tests call `gpush.cli.payload_provenance` -- the same function `cmd_run`
calls. An earlier version of this file re-implemented the decision in the test
body, which would have kept passing if the decision were deleted from the CLI.
That is the "a check that only prints is a claim" failure in test form, and it is
why the decision is a function and not inline code.

Run:  python gpush/tests/test_stale_payload.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpush.cli import PayloadProvenance, payload_provenance  # noqa: E402

FP_THIS_RUN = "a" * 64
FP_PREVIOUS = "b" * 64


class TestStalePayload(unittest.TestCase):

    def test_matching_fingerprint_is_quotable(self):
        pv = payload_provenance({"ok": True, "fingerprint": FP_THIS_RUN}, FP_THIS_RUN)
        self.assertTrue(pv.matched)
        self.assertFalse(pv.stale)
        self.assertTrue(pv.quotable)
        self.assertIn("MATCHED", pv.verdict_text())

    def test_prior_fingerprint_is_stale_even_though_payload_matched(self):
        """The exact failure: `ok: true`, but a different run's bytes."""
        pv = payload_provenance({"ok": True, "fingerprint": FP_PREVIOUS}, FP_THIS_RUN)
        self.assertTrue(pv.matched, "the payload did match its own manifest")
        self.assertTrue(pv.stale, "but it was not this run's payload")
        self.assertFalse(pv.quotable, "a stale payload must void the run")
        self.assertIn("STALE PAYLOAD", pv.verdict_text())

    def test_stale_is_not_reported_as_matched(self):
        """`MATCHED` must not appear in a stale verdict.

        This is the whole point. If both words could be printed at once, or
        MATCHED won the formatting, a reader scanning the output would keep a
        number from a run that measured someone else's bytes.
        """
        pv = payload_provenance({"ok": True, "fingerprint": FP_PREVIOUS}, FP_THIS_RUN)
        self.assertNotIn("MATCHED", pv.verdict_text())

    def test_mismatch_without_staleness(self):
        pv = payload_provenance({"ok": False, "reason": "differing",
                                 "fingerprint": FP_THIS_RUN}, FP_THIS_RUN)
        self.assertFalse(pv.matched)
        self.assertFalse(pv.stale, "same run, wrong bytes -- a mismatch, not stale")
        self.assertFalse(pv.quotable)
        self.assertIn("DID NOT MATCH", pv.verdict_text())

    def test_both_mismatch_and_stale(self):
        pv = payload_provenance({"ok": False, "fingerprint": FP_PREVIOUS}, FP_THIS_RUN)
        self.assertFalse(pv.matched)
        self.assertTrue(pv.stale)
        self.assertFalse(pv.quotable)
        self.assertIn("STALE PAYLOAD", pv.verdict_text())

    def test_absent_reported_fingerprint_is_not_claimed_as_identity(self):
        """A verdict with no fingerprint establishes neither match nor staleness.

        It must not be read as verified. `stale` is False because there is no
        evidence of a wrong run, but `quotable` is still True only because the
        runner's own check passed -- so the CLI must not *claim* identity, and
        this test pins that the field is None rather than equal.
        """
        pv = payload_provenance({"ok": True}, FP_THIS_RUN)
        self.assertIsNone(pv.ran)
        self.assertEqual(pv.pushed, FP_THIS_RUN)
        self.assertFalse(pv.stale)

    def test_absent_pushed_fingerprint_is_not_stale(self):
        pv = payload_provenance({"ok": True, "fingerprint": FP_THIS_RUN}, None)
        self.assertFalse(pv.stale)
        self.assertIsNone(pv.pushed)

    def test_the_cli_calls_this_function_rather_than_inlining_the_decision(self):
        """Guards against the decision being duplicated or dropped."""
        here = os.path.dirname(os.path.abspath(__file__))
        src = open(os.path.join(here, "..", "cli.py")).read()
        self.assertIn("payload_provenance(verify", src,
                      "cmd_run must route its verdict through the tested function")
        # and the inline version this replaced must be gone
        self.assertNotIn("ran_fp != pushed_fp", src,
                         "the inline duplicate must not survive alongside the function")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestIdentityMustBeEstablished(unittest.TestCase):
    """A missing fingerprint must not read as a pass.

    This is not hypothetical. A lane that forgot to pass `compute_fingerprint`
    got `quotable == True` from a comparison that never happened: `stale` is
    `bool(pushed_fp and ran and ran != pushed_fp)`, so an absent `pushed_fp`
    makes `stale` False, and `quotable` used to be `matched and not stale`.
    Absence of evidence was being consumed as evidence of absence of a problem —
    in the one guard whose entire purpose is catching stale remote results, which
    have bitten this project three separate times.
    """

    def test_absent_pushed_fingerprint_is_not_quotable(self):
        p = payload_provenance({"ok": True, "fingerprint": "abc"}, None)
        self.assertFalse(p.quotable,
                         "a run whose payload cannot be identified is not quotable, "
                         "however well it matched its own manifest")

    def test_absent_ran_fingerprint_is_not_quotable(self):
        p = payload_provenance({"ok": True, "fingerprint": None}, "abc")
        self.assertFalse(p.quotable)

    def test_absent_fingerprint_is_not_reported_as_stale(self):
        """The other half: absence is still not evidence of staleness.

        `stale` must stay False so that STALE PAYLOAD and IDENTITY UNESTABLISHED
        remain two different verdicts. Conflating them would make a missing input
        indistinguishable from a real detection.
        """
        p = payload_provenance({"ok": True, "fingerprint": "abc"}, None)
        self.assertFalse(p.stale)
        self.assertIn("IDENTITY UNESTABLISHED", p.verdict_text())

    def test_established_requires_both_sides(self):
        self.assertFalse(payload_provenance({"ok": True, "fingerprint": "a"}, None).established)
        self.assertFalse(payload_provenance({"ok": True, "fingerprint": None}, "a").established)
        self.assertFalse(payload_provenance({"ok": True, "fingerprint": None}, None).established)
        self.assertTrue(payload_provenance({"ok": True, "fingerprint": "a"}, "a").established)

    def test_all_four_verdicts_are_distinguishable(self):
        """The whole point: four situations, four different answers.

        If any two of these produce the same quotable/verdict pair, the guard has
        collapsed and a caller cannot tell a clean run from an unverified one.
        """
        cases = {
            "clean":            payload_provenance({"ok": True, "fingerprint": "a"}, "a"),
            "stale":            payload_provenance({"ok": True, "fingerprint": "b"}, "a"),
            "unidentified":     payload_provenance({"ok": True, "fingerprint": "a"}, None),
            "did not match":    payload_provenance({"ok": False, "fingerprint": "a"}, "a"),
        }
        quotable = {k: v.quotable for k, v in cases.items()}
        self.assertEqual(quotable["clean"], True)
        for k in ("stale", "unidentified", "did not match"):
            self.assertEqual(quotable[k], False, f"{k} must not be quotable")
        verdicts = {k: v.verdict_text() for k, v in cases.items()}
        self.assertEqual(len(set(verdicts.values())), 4,
                         f"verdicts collapsed: {verdicts}")

    def test_quotable_does_not_depend_on_stale_alone(self):
        """Source-level, so it cannot be reintroduced by a well-meaning edit."""
        import inspect
        src = inspect.getsource(PayloadProvenance.quotable.fget)
        self.assertIn("established", src,
                      "quotable must consult `established`; `not stale` alone "
                      "confuses 'no staleness' with 'no evidence either way'")
