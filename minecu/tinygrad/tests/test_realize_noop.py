#!/usr/bin/env python3
"""minecu/tinygrad/tests/test_realize_noop.py -- prove the no-op is real, then prove the
guard catches it.

`Tensor.realize()` on an already-realized graph returns without running anything.
Time that and you get the cost of an empty function. It has bitten three lanes
(`notes/28` reported a port 13x FASTER than the hand-written CUDA; `notes/32`
reported 24 ns per `cuLaunchKernel`; the T4 KDA lane came close to shipping a
fabricated ~60x).

The guard is only worth having if the hazard is real, so this file does both
halves:

  1. demonstrates the no-op on a real backend, so the guard is not defending
     against a theoretical problem;
  2. demonstrates that `require_fresh_graph` raises on that same tensor;
  3. demonstrates that the instruments actually used for timing do not have the
     defect -- they re-execute compiled calls rather than realizing -- so the
     numbers already in the notes are not affected.

CPU only. No GPU, no network. Runs in under a second.

Run:  python minecu/tinygrad/tests/test_realize_noop.py
"""
import os
import sys
import time
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "minecu", "tinygrad"))

import bench  # noqa: E402


def _fresh():
    from tinygrad import Tensor
    # A graph big enough that a realize is measurably non-trivial, and shaped so
    # realize() has real work to do the first time.
    a = Tensor.rand(256, 256)
    b = (a @ a).relu().sum()
    return a, b


class TestRealizeIsANoOpOnceRealized(unittest.TestCase):

    def test_second_realize_does_no_work(self):
        a, b = _fresh()
        a.realize()
        b.realize()
        # The predicate, not a Tensor attribute -- `is_realized` is not one, which
        # is the bug the guard's first version had.
        self.assertFalse(bench.needs_realize(b), "b should already be realized")

        t0 = time.perf_counter()
        for _ in range(200):
            b.realize()
        t1 = time.perf_counter()
        realized_again_ms = (t1 - t0) * 1e3 / 200

        t0 = time.perf_counter()
        for _ in range(200):
            pass
        t1 = time.perf_counter()
        floor_ms = (t1 - t0) * 1e3 / 200

        self.assertLess(realized_again_ms, max(0.5, floor_ms * 50),
                        "re-realizing should be free; if this is slow the test's "
                        "premise is wrong and the guard is aimed at nothing")

    def test_rebuilding_changes_the_answer(self):
        """The control: a FRESH graph does cost something, so the difference is real."""
        a, b = _fresh()
        b.realize()
        t0 = time.perf_counter()
        b.realize()
        t1 = time.perf_counter()
        again = (t1 - t0) * 1e3

        c = (a @ a).relu().sum()
        t0 = time.perf_counter()
        c.realize()
        t1 = time.perf_counter()
        fresh = (t1 - t0) * 1e3

        self.assertLess(again, fresh,
                        "a realized graph must be cheaper to realize than a fresh one")


class TestGuard(unittest.TestCase):

    def test_needs_realize_is_false_after_realize(self):
        """The predicate must be the framework's, not an attribute that isn't there.

        The first version of the guard tested `getattr(t, "is_realized", None)`,
        which TinyGrad does not have. It therefore returned None on every tensor
        and the guard never fired -- a guard that cannot fail. This test is what
        noticed, and it is the reason the predicate is now copied from
        `tinygrad/tensor.py:422` instead of guessed.
        """
        _, b = _fresh()
        self.assertTrue(bench.needs_realize(b), "a fresh graph needs realizing")
        b.realize()
        self.assertFalse(bench.needs_realize(b),
                         "a realized graph must not need realizing, or realize() "
                         "would have work to do and the no-op would be a myth")

    def test_guard_raises_on_a_realized_tensor(self):
        _, b = _fresh()
        b.realize()
        with self.assertRaises(bench.RealizedGraph):
            bench.require_fresh_graph(b, what="unit test")

    def test_guard_message_names_the_cause_and_the_alternative(self):
        _, b = _fresh()
        b.realize()
        try:
            bench.require_fresh_graph(b, what="the batch-1024 render")
            self.fail("should have raised")
        except bench.RealizedGraph as e:
            msg = str(e)
            self.assertIn("already realized", msg)
            self.assertIn("the batch-1024 render", msg,
                          "the caller's context must survive into the message")
            # name the two functions that ARE safe, by their real names
            self.assertIn("linear_device_time_ms", msg,
                          "the message must name the way out")
            self.assertIn("under 1x", msg,
                          "the message must name the signature of the bug")

    def test_guard_does_not_raise_on_a_fresh_graph(self):
        _, b = _fresh()
        bench.require_fresh_graph(b)          # must not raise

    def test_the_timing_instruments_work_on_a_realized_tensor(self):
        """`linear_device_time_ms` re-executes compiled calls, so the no-op cannot
        affect it -- which is what licenses the DDA numbers already in the notes.

        The property that holds on every host is: **the instrument does not call
        `realize()`, so it does not need the guard and does not raise.** An earlier
        version of this test also asserted `n_kernels >= 1` and a positive device
        time. Both fail on a host with no CUDA backend -- `time_call` returns 0 and
        the LINEAR has no device calls to walk -- and a host that is doing nothing
        wrong should not have to satisfy a GPU assertion.
        """
        _, b = _fresh()
        b.realize()
        self.assertFalse(bench.needs_realize(b), "precondition: b is realized")
        out = bench.linear_device_time_ms(b)          # must not raise RealizedGraph
        self.assertIsNotNone(out)
        self.assertNotIn("error", out, f"linear_device_time_ms failed: {out}")
        for k in ("n_kernels", "median_ms", "min_ms", "max_ms"):
            self.assertIn(k, out, f"the instrument's shape changed: missing {k}")

    def test_the_instruments_do_not_call_realize(self):
        """Source-level, so it holds on a host with no GPU.

        The safety of every number in `notes/23` rests on these two functions not
        going through `realize()`. A test that can only check that on CUDA is a
        test that is not run where it is written, so the property is pinned here
        and the behaviour is checked above.
        """
        import inspect
        for fn in (bench.device_time_ms, bench.linear_device_time_ms):
            src = inspect.getsource(fn)
            self.assertNotIn(".realize(", src,
                             f"{fn.__name__} calls realize(); the numbers it produces "
                             "would be the time of an empty function on a re-run")


if __name__ == "__main__":
    unittest.main(verbosity=2)
