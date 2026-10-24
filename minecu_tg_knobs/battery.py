#!/usr/bin/env python3
"""battery.py -- the on-card measurement for arm B. Timings, on a real T4.

Run on the device, in one process, so every configuration is timed on the same
card in the same session:

    DEV=CUDA python -m minecu_tg_knobs.battery --out DIR

WHY THE TIMINGS ARE ON THE CARD AND THE COUNTS ARE NOT
------------------------------------------------------
`codegen.py` and `sass_census.py` produce static instruction counts, which need
no hardware. This file produces wall-clock device time, which does. The two are
kept apart on purpose: a static instruction count is not a cost (it knows
nothing about occupancy, latency or divergence), and a device time is not a
count (it moves with clocks the sweep does not control). `notes/19` §4.3 is the
long version of this argument; the short version is that the two tables are read
against each other, never merged into one number.

WHAT IS TIMED, EXACTLY
---------------------
The full generated render graph, the same shape as every other number in this
study: batch 1024, 32x32, `max_steps=32`, world 32^3, on the reference's own
model of the world. **Render against render.** The reference's
`render_kernel_uint8_prebasis` per-kernel median is the only legal denominator
(notes/19 §5.1), and `4.55 M env-steps/s` is a whole-step figure over 7
launches that this file never divides by.

Both sides are measured in the same run, on the same card, at equal batch, and
the reference side comes from `refdrv.cu` linking the UNMODIFIED `kernels.cu` --
which is hash-pinned to upstream and is not touched by this lane.

INSTRUMENT
----------
tinygrad's own `time_call`, per kernel, summed -- the same instrument
`notes/19` §5.2 used, so the two tables are comparable. Not a wall clock around
the Python call: `notes/19` §5.6 measures 211.9 ms of tracing against 8.754 ms
of device time at this shape, so a host-side timer would be measuring tinygrad's
Python, not the GPU.

WHY A WORKLOAD GATE IS PRINTED NEXT TO EVERY TIMING
---------------------------------------------------
A render time without the workload behind it is not a measurement, and this repo
has four weeks of that lesson (`minecu_ref/buried_agent_fix.md`). So the mean DDA
steps per pixel is recomputed from this run's own state and printed beside each
timing, and the reference's own gate is applied: `mean >= max(2.0, 0.10*32)`.
A configuration that made the traversal degenerate would be fast and wrong, and
the step count is what makes that visible.

THE CONTEXT GATE
----------------
Every configuration is timed against the DEFAULT configuration of the SAME port
in the SAME run, and the ratio to it is printed. A knob that "helps" by 3% when
the run-to-run spread is 5% is not a finding, and this study's own reference
runs differ by 2.9% on the same measurement. The control is re-timed
interleaved with the configurations rather than once at the start, so drift
shows up as a changing control rather than as a knob effect.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("PARALLEL", "0")

# `DEV` is set HERE, at module scope, and it is `=` not `setdefault`.
#
# Two reasons, both learned from a gate that did not gate. Setting it inside
# `main()` is too late: tinygrad resolves `Device.DEFAULT` when the device is
# first built, so a `DEV` assigned after the first import is ignored and the
# tensor work runs on the **CPU** -- and the probe then reported "a CUDA kernel
# ran and returned 15.0", which is exactly the false pass this gate exists to
# prevent. A second run refused correctly, one line later, because by then
# `Device.DEFAULT` had been cached from the environment. Which of those two
# behaviours you get depends on import order, so the assignment moves to module
# scope where it cannot be reordered.
os.environ["DEV"] = "CUDA"

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# The reference's own workload gate: notes/19 §1.3, and the same floor the
# reference's own bench applies. mean steps/pixel must clear it.
GATE_FLOOR = 2.0
GATE_FRACTION = 0.10


def _device_report() -> dict:
    import subprocess
    out: dict = {}
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        if smi:
            f = [x.strip() for x in smi.splitlines()[0].split(",")]
            out.update(gpu=f[0], memory_mb=int(float(f[1])),
                       compute_cap=f[2], driver=f[3])
    except Exception as exc:  # noqa: BLE001
        out["nvidia_smi_error"] = f"{type(exc).__name__}: {exc}"
    try:
        nv = subprocess.run(["nvcc", "--version"], capture_output=True,
                            text=True, timeout=30).stdout
        for line in nv.splitlines():
            if "release" in line:
                out["nvcc"] = line.split("release")[1].split(",")[0].strip(" :")
                break
    except Exception:  # noqa: BLE001
        pass
    out["hostname"] = os.uname().nodename
    return out


def _sync(dev: str = "CUDA") -> None:
    """Block until the device has finished, if it has such a method.

    A timing loop without a synchronisation measures the cost of *enqueuing*.
    """
    from tinygrad import Device
    d = Device[dev]
    for attr in ("synchronize", "sync"):
        fn = getattr(d, attr, None)
        if callable(fn):
            fn()
            return
    # CUDAHCQDevice exposes the stream; a full stream sync is the portable form
    rt = getattr(d, "runtime", None)
    for attr in ("synchronize", "stream_synchronize"):
        fn = getattr(rt, attr, None)
        if callable(fn):
            fn()
            return


def time_graph(thunk, iters: int = 8, warmup: int = 3) -> dict:
    """Device time for one configuration, via tinygrad's own per-kernel timer.

    Returns per-iteration medians and the per-kernel breakdown, because the
    kernel COUNT is part of the result (notes/19 §5.2: launch overhead is under
    1.5% on this kernel on this card, and that is itself a result rather than an
    assumption to be made).

    ## THE INSTRUMENT, and why it is this one

    Not tinygrad's `time_call`. That function is `time_call(call: UOp, ...)` in
    this version: it takes a **realized UOp**, not a thunk, and has no `reps`
    argument. An earlier version of this file called it as `time_call(thunk,
    reps=1)` and died on the card with `TypeError: time_call() got an
    unexpected keyword argument 'reps'` -- after the payload was verified, the
    device reported and 211 s of session. The lesson generalises past this call:
    **a function's name is not its signature**, and an instrument assumed rather
    than inspected is an instrument that costs a session to find out.

    So the instrument is stated here, in full, and is the same shape as
    `minecu_tg/bench.py:time_call` -- which is what `notes/19` §5.2 used, so the
    two tables are comparable:

      * warm up, then **synchronise**;
      * for each iteration, `perf_counter` around realize, then synchronise;
      * report the median over `iters`.

    A timing loop without the synchronisation measures the cost of *enqueuing*,
    which for a graph this size is the smaller half of the cost by a wide
    margin, and the resulting number would be a real, confident, wrong
    measurement.

    The kernel COUNT comes from `GlobalCounters.kernel_count` and is reported
    because it is itself a result (`notes/19` §5.2 puts launch overhead under
    1.5% on this kernel on this card).
    """
    from tinygrad import GlobalCounters

    for _ in range(warmup):
        out = thunk()
        (out.realize() if hasattr(out, "realize") else out)
    _sync()

    samples, per_kernel = [], []
    for _ in range(iters):
        GlobalCounters.reset()
        t0 = time.perf_counter()
        out = thunk()
        (out.realize() if hasattr(out, "realize") else out)
        _sync()
        samples.append(time.perf_counter() - t0)
        per_kernel.append(GlobalCounters.kernel_count)
    return {
        "device_s_median": statistics.median(samples),
        "device_s_min": min(samples),
        "device_s_samples": samples,
        "kernels_per_call": max(per_kernel, default=0),
        "instrument": f"perf_counter around realize + _sync() "
                      f"(same shape as minecu_tg/bench.py:time_call, the "
                      f"instrument notes/19 §5.2 used)",
    }


def measure_workload(vox, cams, res: int, max_steps: int, port: str) -> dict:
    """The mean DDA steps per pixel for this configuration's own state.

    Recomputed per configuration rather than assumed constant, because a knob
    that changed the traversal -- and TRANSCENDENTAL=2 rewrites SQRT, which
    feeds `ray_len`, which feeds every `t_max` -- would change the workload and
    a timing taken against a fixed workload would then be comparing two
    different amounts of work.
    """
    from tinygrad import Tensor, dtypes

    from minecu_tg_knobs.lean import camera_basis, render as lean_render
    from minecu_tg.render import render as naive_render

    fn = lean_render if port.startswith("lean") else naive_render
    kw = {} if port == "naive" else dict(cursor_int=True, flat_advance=True,
                                         unsigned_bounds=True)
    vox_t = Tensor(vox, dtype=dtypes.int8)
    basis = camera_basis(Tensor(cams, dtype=dtypes.float32), 70.0, res, res)
    _, steps = fn(vox_t, basis.contiguous(), res, res, max_steps,
                  count_steps=True, **kw)
    s = steps.numpy()
    return {
        "mean_steps": float(s.mean()),
        "at_cap_fraction": float((s >= max_steps).mean()),
        "p_first_voxel_fraction": float((s <= 1).mean()),
        "gate_floor": max(GATE_FLOOR, GATE_FRACTION * max_steps),
    }


def ensure_tinygrad() -> tuple[bool, str]:
    """Is tinygrad importable here, and install it if it is not.

    The payload is a source tree, not an environment. Kaggle's image ships
    Python but not tinygrad, so an earlier run of this file died with
    `ModuleNotFoundError: No module named 'tinygrad'` 0.1 s in -- again after a
    paid session, and again with the payload intact and verified.

    Installing here rather than in the notebook keeps the whole entry sequence
    in one auditable place: `battery.py` states what it needs and refuses with
    a reason if it cannot get it. `minecu_tg/kaggle/make_notebook.py:153` does
    the same thing one layer up, which is why that arm works; this arm's
    notebook is gpush's, and gpush's runner executes the command and does not
    provision a Python environment.

    The version is deliberately **unpinned**, and that is a real limitation
    rather than an oversight: this arm's registry cites `helpers.py:238` and
    `codegen/opt/heuristic.py:28` by line, and those cites are checked against
    whatever tinygrad is actually present. A pin would make the cites
    reproducible and would also mean this arm measured a version the rest of the
    study did not. The check decides which of those two is happening, and it
    runs.
    """
    try:
        import tinygrad  # noqa: F401
        return True, "ok"
    except ImportError:
        pass

    import subprocess
    import sys as _sys
    try:
        proc = subprocess.run([_sys.executable, "-m", "pip", "install", "-q",
                               "tinygrad"],
                              capture_output=True, text=True, timeout=600)
    except Exception as exc:  # noqa: BLE001
        return False, f"tinygrad is absent and pip could not be run: {type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        return False, (f"tinygrad is absent and `pip install tinygrad` failed "
                       f"(rc={proc.returncode}): {proc.stderr[-300:]}")
    try:
        import tinygrad  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, f"pip reported success but tinygrad still will not import: {exc}"
    print(f"  installed tinygrad {getattr(tinygrad, '__version__', '?')} "
          f"({tinygrad.__file__})")
    return True, "ok"


def probe_cuda() -> tuple[bool, str]:
    """Does a CUDA device exist here, and does it actually run a kernel?

    Both halves, because either alone is insufficient. `Device['CUDA']`
    appearing in the device list only means tinygrad believes a backend is
    available; the check that matters is a real allocation and a real kernel
    launch, whose result is compared against the value computed on the CPU. A
    driver that reports a device and cannot execute would otherwise produce a
    battery of zeros or an exception deep inside the first timing.

    Returns `(ok, why)`. `why` is the reason for the failure and is printed, so
    a refusal is diagnosable rather than a bare refusal.
    """
    try:
        from tinygrad import Device, Tensor
    except Exception as exc:  # noqa: BLE001
        return False, f"tinygrad import failed: {type(exc).__name__}: {exc}"

    if "CUDA" not in Device._devices:
        return False, (f"tinygrad does not offer CUDA; it offers "
                       f"{sorted(Device._devices)}")
    # a kernel that must produce a KNOWN number on any working device, so the
    # check is on the result and not merely on the absence of an exception:
    # (1*2+1) + (2*2+1) + (3*2+1) = 3 + 5 + 7 = 15.
    EXPECTED = 15.0
    try:
        b = (Tensor([1.0, 2.0, 3.0]) * 2.0 + 1.0).sum().item()
    except Exception as exc:  # noqa: BLE001
        return False, (f"CUDA device present but a kernel would not run: "
                       f"{type(exc).__name__}: {exc}")

    if abs(float(b) - EXPECTED) > 1e-3:
        return False, (f"a CUDA kernel ran but returned {b}, not {EXPECTED} -- "
                       f"the device is not computing what it claims to")
    return True, "ok"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=32)
    ap.add_argument("--world", type=int, default=32)
    ap.add_argument("--ground", type=int, default=8)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--ports", default="naive,lean")
    ap.add_argument("--out", default="")
    ap.add_argument("--skip-reference", action="store_true",
                    help="timed only the TinyGrad side; the reference column "
                         "then reads 'not measured in this run' and no ratio "
                         "against it may be quoted")
    a = ap.parse_args(argv)

    # `DEV` is SELECTED here, not inherited. A Kaggle GPU notebook does not
    # export `DEV=CUDA`, so tinygrad defaults to CPU and the first run of this
    # file on the card refused with "REFUSING: DEV='CPU'" 0.1 s in, having
    # spent a whole session to learn that the accelerator is not the default.
    # The box is a GPU box -- that is why it was dispatched to one.
    #
    # The gate is therefore NOT "is DEV set to CUDA", which this assignment
    # would make trivially true and therefore vacuous. It is "does a CUDA
    # device actually exist and actually work here", checked by making tinygrad
    # open one and by running a real kernel on it. A CPU number in a table of
    # GPU numbers is the exact failure mode gpush's design forbids, and a gate
    # that cannot fail is not a gate.
    ok, why = ensure_tinygrad()
    if not ok:
        print(f"REFUSING: {why}. This file produces DEVICE TIMES, and a CPU "
              f"number in a table of GPU numbers is the exact failure mode "
              f"gpush's design forbids.", file=sys.stderr)
        return 2

    ok, why = probe_cuda()
    if not ok:
        print(f"REFUSING: no working CUDA device here ({why}). This file "
              f"produces DEVICE TIMES, and a CPU number in a table of GPU "
              f"numbers is the exact failure mode gpush's design forbids. Use "
              f"codegen.py for the offline instruction counts.", file=sys.stderr)
        return 2
    dev = "CUDA"

    from minecu_tg_knobs import registry
    from minecu_tg_knobs.codegen import PORTS, configurations, build_inputs

    registry.check_all()
    dev_info = _device_report()
    print("=" * 78)
    print("  minecu_tg_knobs -- the on-card battery, one run, same card")
    print(f"  device  {json.dumps(dev_info)}")
    print(f"  shape   batch {a.batch}, res {a.res}, max_steps {a.max_steps}, "
          f"world {a.world}^3")
    print("=" * 78)
    if dev_info.get("compute_cap") not in (None, "7.5"):
        print(f"  NOTE: compute capability {dev_info.get('compute_cap')}, not 7.5. "
              f"sm_75 is what every other number in this study was taken on; a "
              f"different card is a different experiment and the note says so.",
              file=sys.stderr)

    cache = Path(a.out) / "inputs.npz" if a.out else \
        REPO / ".agent-scratch" / "knobs" / "inputs.npz"
    vox, cams = build_inputs(a.world, a.ground, a.batch, a.res, cache)

    cfgs = configurations()
    ports = [p for p in a.ports.split(",") if p]
    for p in ports:
        if p not in PORTS:
            print(f"unknown port {p!r}", file=sys.stderr)
            return 2

    from tinygrad import Context

    # The control is timed FIRST and then re-timed after every configuration,
    # interleaved. A knob that "helps" by less than the run-to-run spread is
    # not a finding, and the only way to know the spread is to measure the
    # control more than once in the same run.
    records: list[dict] = []
    control: dict[str, list[float]] = {p: [] for p in ports}

    def build(port: str, cfg: dict):
        mod, fname, kwargs = PORTS[port]
        fn = getattr(__import__(mod, fromlist=[fname]), fname)
        from tinygrad import Tensor, dtypes
        from minecu_tg_knobs.lean import camera_basis

        def thunk():
            with Context(**cfg):
                v = Tensor(vox, dtype=dtypes.int8)
                b = camera_basis(Tensor(cams, dtype=dtypes.float32), 70.0,
                                 a.res, a.res).contiguous()
                return fn(v, b, a.res, a.res, a.max_steps, **kwargs)
        return thunk

    for label, cfg in cfgs:
        for port in ports:
            thunk = build(port, cfg)
            # compile and warm first, outside the timed region, so a first-call
            # nvrtc compile is not charged to the first configuration
            warm = time_graph(thunk, iters=1, warmup=1)
            wl = measure_workload(vox, cams, a.res, a.max_steps, port)
            t = time_graph(thunk, iters=a.iters)

            rec = {
                "label": label, "config": cfg, "port": port,
                "batch": a.batch, "res": a.res, "max_steps": a.max_steps,
                "kernels": t["kernels_per_call"],
                "device_ms_median": t["device_s_median"] * 1e3,
                "device_ms_min": t["device_s_min"] * 1e3,
                "samples_ms": [s * 1e3 for s in t["device_s_samples"]],
                "workload": wl,
                "gate_pass": wl["mean_steps"] >= wl["gate_floor"],
                "compile_warmup_ms": warm["device_s_median"] * 1e3,
                "instrument": t["instrument"],
            }
            if label == "default":
                control[port].append(rec["device_ms_median"])
                rec["ratio_to_control"] = 1.0
            else:
                base = control[port][-1] if control[port] else None
                rec["ratio_to_control"] = (rec["device_ms_median"] / base
                                           if base else None)
            records.append(rec)
            # the ratio is built into the row rather than conditionally
            # selecting between two f-strings: a conditional expression across
            # an f-string boundary is a syntax trap, and the failure mode is a
            # row that silently omits the gate instead of printing it.
            ratio = rec["ratio_to_control"]
            ratio_txt = f"x{ratio:.3f}" if ratio else "  -   "
            gate_txt = "PASS" if rec["gate_pass"] else "**GATE FAIL**"
            print(f"  {port:<17} {label:<32} "
                  f"{rec['device_ms_median']:>9.3f} ms  {rec['kernels']:>3d} kern  "
                  f"mean steps {wl['mean_steps']:>7.3f}  gate {gate_txt:<13}"
                  f"{ratio_txt:>8}", flush=True)

        # re-time the control after every configuration, so drift is visible as
        # a moving control rather than as an effect of whatever ran last
        for port in ports:
            t = time_graph(build(port, {}), iters=3, warmup=1)
            control[port].append(t["device_s_median"] * 1e3)

    spread = {p: (max(v) - min(v)) / min(v) for p, v in control.items() if v}
    print()
    print("  control re-timed through the run (relative spread):")
    for p, s in spread.items():
        print(f"    {p:<17} {s*100:>5.2f}%   "
              f"{'a knob effect smaller than this is not a finding' if s*100 < 1 else ''}")

    out = {"device": dev_info, "shape": {"batch": a.batch, "res": a.res,
                                         "max_steps": a.max_steps,
                                         "world": a.world, "ground": a.ground},
           "control_ms": control, "control_spread": spread,
           "reference": "not measured in this run" if a.skip_reference
                        else "see notes/19 §5.2 for the reference column; this "
                             "file times the TinyGrad side only",
           "records": records}
    if a.out:
        d = Path(a.out)
        d.mkdir(parents=True, exist_ok=True)
        (d / "battery.json").write_text(json.dumps(out, indent=1))
        print(f"\n  wrote {d/'battery.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
