"""Timing and memory, on a real device.

WHAT IS TIMED, AND WHY IT IS NOT THE WHOLE STEP
------------------------------------------------
The hand-written baseline's number is a whole *step*: seven or nine launches,
five of them state kernels, and the last one the render. TinyGrad, given the
render alone, emits a graph of kernels. Those are different objects, and
`notes/17` §8.2 says so before any number is taken. This module therefore times
three separate things and labels all three:

  * `render`      -- the TinyGrad graph for the observation, and nothing else.
                     This is the per-op number, comparable to the reference's
                     `render_kernel_uint8_prebasis` per-kernel median.
  * `render+d2h`  -- the same plus a read-back, so the cost of getting the
                     answer off the device is visible rather than assumed away.
  * `whole graph` -- the whole `obs = render(...)` including the basis kernel,
                     which is the object most comparable to the reference's
                     whole step. It is still not a whole step, because the
                     reference's step also does decode/physics/raycast/reset,
                     and that difference is named rather than hidden.

WHY MEMORY IS MEASURED AND NOT DERIVED
---------------------------------------
At `max_steps=32` the generated schedule writes the DDA state to global memory
once per step and the final kernel takes all of those buffers as inputs. That
makes the peak footprint a function of `max_steps` and of the pixel count, and
at the project's headline shape (batch 32768, 32x32) the arithmetic says the
graph cannot be allocated at all. An arithmetic consequence is not a
measurement, so the footprint is measured with the allocator's own counters and
the run reports what it actually allocated, including the shapes where it
failed.
"""

from __future__ import annotations

import gc
import statistics
import time


def _sync(dev):
    from tinygrad import Device
    d = Device[dev]
    if hasattr(d, "synchronize"):
        d.synchronize()


class MemProbe:
    """Track what the allocator actually hands out, by wrapping `alloc`/`free`.

    This version of tinygrad has no `max_memory_allocated`, so the footprint is
    measured by instrumenting the one place bytes are requested:
    `Allocator.alloc` / `Allocator.free` (device.py). `alloc` records the
    requested size and adds it to a live total; `free` subtracts. `peak` is the
    high-water mark of that total.

    This is a measurement of the real allocator, not an estimate from the
    generated source, and the two are reported separately in the notes because
    they answer different questions: the allocator's peak is what the card had
    to have, and the source's argument list is what the schedule structurally
    requires.

    `oom` is the exception text when an allocation fails, which at the headline
    shape is the finding rather than a failure to hide.
    """

    def __init__(self, dev: str):
        from tinygrad import Device
        from tinygrad.device import Allocator
        self.dev = dev
        self.live = 0
        self.peak = 0
        self.total = 0
        self.n_allocs = 0
        self.oom: str | None = None
        self.d = Device[dev]
        self._a, self._f = Allocator.alloc, Allocator.free
        probe = self                    # the closures take the allocator as `a`

        def alloc(a, size, options=None):
            r = probe._a(a, size, options)
            probe.n_allocs += 1
            probe.total += int(size)
            probe.live += int(size)
            probe.peak = max(probe.peak, probe.live)
            return r

        def free(a, opaque, size, options=None):
            probe.live -= int(size)
            return probe._f(a, opaque, size, options)

        def alloc_oom(a, size, options=None):
            try:
                return alloc(a, size, options)
            except MemoryError as e:
                probe.oom = f"{e}"[:300]
                raise

        Allocator.alloc, Allocator.free = alloc_oom, free

    def reset(self):
        self.live = self.peak = self.total = self.n_allocs = 0
        self.oom = None

    def restore(self):
        from tinygrad.device import Allocator
        Allocator.alloc, Allocator.free = self._a, self._f

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.restore()
        return False

    def as_dict(self) -> dict:
        return {"peak_live_bytes": self.peak, "total_allocated_bytes": self.total,
                "n_allocs": self.n_allocs, "live_at_end_bytes": self.live,
                "oom": self.oom}


def time_call(fn, dev, iters: int, warmup: int = 5) -> dict:
    """Median / min / max / std of `fn()`, in milliseconds, wall clock.

    This is the *end to end* number: whatever `fn` does, host side and device
    side, plus one synchronization. Both halves are real costs and neither is
    divided out, because the point is what a caller pays. `device_time_ms` is
    the device-only counterpart, measured separately.
    """
    for _ in range(warmup):
        fn()
    _sync(dev)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        _sync(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return {
        "iters": iters,
        "median_ms": statistics.median(ts),
        "min_ms": ts[0],
        "max_ms": ts[-1],
        "mean_ms": statistics.fmean(ts),
        "std_ms": statistics.stdev(ts) if len(ts) > 1 else 0.0,
    }


def device_time_ms(tensor, iters: int = 5) -> dict | None:
    """Device-only time for a realized tensor, via tinygrad's own `time_call`.

    `tinygrad/engine/realize.py:time_call` builds an ExecContext with
    `update_stats=False, wait=True` and re-executes a single CALL, returning
    the device-measured duration of the largest kernel in the group. That is
    the same quantity `bench.cu` gets from `cudaEvent` around a launch, and it
    is the right instrument here: wall clock around a Python-level call is
    dominated by tracing, and tracing is not the render.

    Two caveats, both carried into the notes rather than hidden:

      * it re-uses the *compiled* program, so this measures re-execution, not
        the first call's compile. Compilation is reported separately (the
        census's `new_distinct_programs` and the wall-clock build-only figure).
      * it takes the max over the kernels in the group, so for a graph it is a
        per-kernel number, not a whole-graph number. The whole-graph device
        time is therefore measured as `sum of per-kernel device times` by
        walking the LINEAR, which `linear_device_time_ms` does.

    Returns None when the backend does not support it, rather than a zero that
    would read as "infinitely fast".
    """
    from tinygrad.engine.realize import time_call
    try:
        linear, var_vals = _linear_and_vars(tensor)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:200]}
    try:
        best = None
        for call in linear.src:
            t = next(time_call(call, var_vals=var_vals))
            best = t if best is None else max(best, t)
        return {"max_kernel_ms": best * 1e3 if best is not None else None}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:200]}


def needs_realize(tensor) -> bool:
    """True iff `tensor.realize()` would actually run something.

    **This is the framework's own condition, copied from
    `tinygrad/tensor.py:422`:**

        to_realize = [x for x in (self,)+lst
                      if not x.uop.is_virtual and not x.uop.has_buffer_identity()]
        if len(to_realize):
            run_linear(...)

    So a tensor for which this is False is already realized, and calling
    `realize()` on it **returns without running anything**. An earlier version of
    this guard tested `getattr(t, "is_realized", None)`, which is not an attribute
    TinyGrad has -- so it returned None every time and the guard never fired. It
    looked like a guard and was decoration. The test that now pins it exists
    because that happened.
    """
    uop = getattr(tensor, "uop", None)
    if uop is None:
        # Not a Tensor, or a shape-only thing. Be permissive rather than raise:
        # the guard exists to catch a specific mistake, not to police types.
        return True
    try:
        return (not uop.is_virtual) and (not uop.has_buffer_identity())
    except Exception:
        return True


def require_fresh_graph(tensor, what: str = "this measurement") -> None:
    """Refuse to time a tensor whose graph has already been realized.

    ## Why this exists, having been burned three times

    `Tensor.realize()` on an **already-realized** graph is a **silent no-op**: it
    returns without running anything, because there is nothing left to run. Time
    one and you get the time of an empty function -- microseconds -- divide it by
    the reference, and the port comes out **faster than hand-written CUDA**. That
    is not hypothetical: it happened, printed a plausible table across three
    batches with the observation byte-identical everywhere, and was caught only
    because the ratio moved the *wrong way* as batch size grew.

    This file was not bitten, because `device_time_ms` and
    `linear_device_time_ms` re-execute the already-compiled calls with
    `time_call` rather than realizing the graph. That is the right instrument, and
    it is immune -- but immunity by construction is invisible until someone edits
    it, so the property is now stated and checked rather than remembered.

    **Three occurrences, three lanes:** `notes/28` (a port reported 13x *faster*
    than the reference), `notes/32` (a harness reporting 24 ns per
    `cuLaunchKernel`), and the T4 KDA lane (a fabricated ~60x that nearly
    shipped). Same defect, same shape.

    ## What this guard does and does not buy

    It catches the caller who *passes* an already-realized tensor. It cannot
    catch a caller that builds and realizes the graph itself and then times a
    fresh realize -- for that the instrument has to be `time_call` on the LINEAR,
    as it is here. So: call this before timing, use the functions below, and do
    not put `realize()` back into a timing path.

    Raises rather than returning a flag, because a flag a timing path can ignore
    is a flag that will be ignored.
    """
    if not needs_realize(tensor):
        raise RealizedGraph(
            f"{what}: this tensor is already realized, so realizing it again is a "
            "silent no-op and timing that measures an empty function -- a ratio "
            "against the reference will come out under 1x, i.e. the port will "
            "appear faster than hand-written CUDA. Rebuild the graph, or use "
            "device_time_ms / linear_device_time_ms, which re-execute the "
            "compiled calls instead of realizing."
        )


class RealizedGraph(RuntimeError):
    """Raised by `require_fresh_graph`. See that function for why it exists."""


def _linear_and_vars(tensor):
    from tinygrad import Tensor as _T
    lin, vv = _T.linear_with_vars(tensor)
    return lin, vv


def linear_device_time_ms(tensor, iters: int = 3) -> dict | None:
    """Sum of per-kernel device times over the whole graph, in ms.

    Walks the compiled LINEAR and times each CALL with `time_call`, then sums.
    This is the whole-graph device time: it is what the reference's
    whole-step number contains, on the same kind of clock, and it excludes host
    tracing, which the framework adds and the reference does not.
    """
    from tinygrad.engine.realize import time_call
    try:
        linear, var_vals = _linear_and_vars(tensor)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:200]}
    try:
        calls = list(linear.src)
        for _ in range(2):                     # two warm passes
            for c in calls:
                next(time_call(c, var_vals=var_vals))
        tot = []
        for _ in range(max(1, iters)):
            s = 0.0
            for c in calls:
                s += next(time_call(c, var_vals=var_vals))
            tot.append(s * 1e3)
        tot.sort()
        return {"n_kernels": len(calls), "median_ms": statistics.median(tot),
                "min_ms": tot[0], "max_ms": tot[-1],
                "median_per_kernel_us": statistics.median(tot) * 1e3 / max(1, len(calls))}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:200]}


def bench_render(build_inputs, render_fn, dev, iters: int, label: str,
                 warmup: int = 5) -> dict:
    """Build once, time the realize, and report the allocator's peak.

    `build_inputs()` must return whatever `render_fn` needs. It is called inside
    the memory probe, so the reported peak covers the inputs *and* every
    intermediate the render allocates -- which is the right boundary, because
    the reference's per-kernel measurement also runs against a fully allocated
    environment (`bench.cu` allocates the whole state up front and times a
    single launch inside it).
    """
    out: dict = {"label": label, "realized": False, "error": None}
    gc.collect()
    r = None
    with MemProbe(dev) as mp:
        try:
            inputs = build_inputs()
            out["mem_after_inputs"] = mp.as_dict()
            mp.reset()
            r = render_fn(*inputs) if isinstance(inputs, tuple) else render_fn(inputs)
            r = r[0] if isinstance(r, tuple) else r
            # device timing FIRST, while the graph is still unrealized:
            # `linear_with_vars` on a realized tensor returns an empty LINEAR,
            # because there is nothing left to run.
            out["device_time"] = linear_device_time_ms(
                r, iters=min(5, max(2, iters // 2)))
            r.realize()
            out["realized"] = True
        except Exception as e:                  # OOM is a result, not a crash
            out["error"] = f"{type(e).__name__}: {str(e)}"[:400]
        _sync(dev)
        out["mem"] = mp.as_dict()

    if not out["realized"]:
        return out

    out["obs_bytes"] = int(r.nbytes())
    out["obs_shape"] = list(r.shape)
    out["obs_dtype"] = str(r.dtype)
    return out


def d2h_time(r, dev, iters: int, warmup: int = 3) -> dict:
    """The cost of reading the observation back, as its own number."""
    return time_call(lambda: r.numpy(), dev, iters, warmup)
