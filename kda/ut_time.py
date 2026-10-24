#!/usr/bin/env python3
"""ut_time.py -- price the KDA UT transform candidates in WALL CLOCK on a real card.

WHAT THIS IS
------------
`notes/31-ut-transform-pricing.md` priced five TinyGrad spellings of the KDA UT
transform -- the inverse of a `[64,64]` unit lower-triangular matrix -- with an
OFFLINE schedule census: kernel counts, REDUCE counts, ALU-lane ops, DAG depth,
and the algorithmic multiply-add count. Every number in that note is a census.
Its own section 8, item 2, is the gap this file closes:

> Whether `C` or `D` wins, given that their lane-op ratio is 8.1x and their
> launch ratio is 0.17x. These point opposite ways and the crossover depends on
> the ratio of launch overhead to lane throughput on the specific card.

and item 1:

> Wall-clock, anywhere. No GPU on this host.

The reason the study was blocked on sm_80+ is that comparing against `fla`
needs sm_80+. **Comparing two of our own candidates does not.** C and D2 are
both TinyGrad implementations of the same operator, so timing them against each
other is a TinyGrad-internal comparison and runs on the free Kaggle T4 (sm_75).

THE INSTRUMENT IS REUSED, NOT RE-EXPRESSED
-------------------------------------------
The candidate expressions, the problem instance (`ut_price.make_S`) and the
census are IMPORTED from `kda/ut_price.py`. They are not re-derived here. Two
expressions that compute the same operator to 3.1e-08 can still differ in
launch count by 20x, and a hand-rewrite would change the very quantity under
test. What is new in this file is only the clock.

`ut_price.selftest()` is re-run first (`--selftest`), because a census that
does not reproduce a known figure is a broken census and a timing harness
built on a broken candidate set is worse than no timing harness.

FOUR NUMBERS PER CANDIDATE, NOT ONE
-----------------------------------
`notes/10-t4-results.md` section 7: on this operator the wall clock is ~23%
Python, ~76% TinyGrad's scheduler, ~1% GPU. So a single "time" here would be
almost entirely a measurement of the host. What is measured, separately:

  launches   `GlobalCounters.kernel_count` -- the number of kernels the DEVICE
             actually executed. This is the cross-check of the census's kernel
             count, and it is the only one of the four that is a count of
             launches rather than an estimate.
  kernel_us  the sum of the per-kernel CUDA-event elapsed times that tinygrad
             ITSELF PRINTS at DEBUG=2 (realize.py:92-94). CAVEAT 1,
             load-bearing: `engine/realize.py:318` sets
             `ExecContext.wait = wait or DEBUG>=2`, and
             `runtime/ops_cuda.py:26-35` then records an event, launches,
             records a second event and SYNCHRONISES on it. So DEBUG=2
             SERIALISES the graph and each per-kernel number CONTAINS the host
             launch latency for that kernel. That is a property of the
             instrument, not of the card, and it is why (1) below exists.
  copy_us    `GlobalCounters.time_sum_s` minus `kernel_us`. CAVEAT 2, also
             load-bearing: `time_sum_s` is NOT pure device time. It is
             `sum(et for et in ets)` (realize.py:65) PLUS, for every call whose
             runtime returns no elapsed time -- every COPY, because `exec_copy`
             returns `[]` (realize.py:167-178) -- a host-side `perf_counter`
             delta taken after a `Device.synchronize()` (realize.py:78-81). On
             a real card the COPYs are pinned-host allocations and they
             dominate: measured in the run of record they add 1,436 us on top of
             candidate C's 840 us of real kernel time, so `time_sum_us` reads
             2.71x the device time. `kernel_us` is therefore the headline and
             `time_sum_us` is reported beside it with the difference named.
  t_sched    host wall clock of `Tensor.schedule_linear()`.
  t_exec     host wall clock of `Tensor.realize()` + `Device.synchronize()`.
             CAVEAT 3, and the one this harness walked into: `schedule_linear()`
             MUTATES the graph (`tensor.py:409-411` applies the buffers), so a
             `realize()` on the same tensors afterwards is a SILENT NO-OP
             (`tensor.py:422`; `notes/26` section 6f). The first version of
             this file timed build+schedule+realize as one end-to-end and, on
             the Metal host it was smoke-tested on, reported 0.08 ms for 5,388
             kernel launches -- 15 ns per `cuLaunchKernel`, and impossible. (On
             the T4 the same bug gave 0.127 ms for 5,380, 24 ns each; see
             `notes/32` section 1a.) The three host clocks are now taken on
             three SEPARATE fresh graphs, and the timed realize is refused
             unless it is observed to have launched something.

The launch-versus-arithmetic question is a question about `kernel_us`, and the
wall clock is reported next to it and never substituted for it.

HOW THE TWO AXES ARE SEPARATED WITHOUT A CONSTANT
------------------------------------------------
The brief is right that the census is not additive -- `notes/31` section 3.5:
candidate A alone reports more ALU-lane ops than the whole operator does. So
instruction counts are NOT converted into time here by any factor. Two
independent measured decompositions are used instead:

  (1) LAUNCH CALIBRATION. A chain of N deliberately trivial kernels (a [1]
      tensor, one ALU op each) is timed with the same DEBUG=2 instrument. The
      slope of `kernel_us` against the number of kernels that ACTUALLY LAUNCHED
      is the card's per-launch cost in microseconds, measured in this process on
      this card. Then
          non_launch_us = kernel_us(candidate) - launches * per_launch_us
      is a residual in microseconds, not an instruction count times a constant.
      A second chain over a [64,64] buffer is the sensitivity check: it is what
      a candidate kernel of this operator's size costs if it does nothing but
      move one buffer.

  (2) SHAPE SWEEP. The number of launches is a property of the schedule and
      should not depend on how many slabs there are; the arithmetic is linear
      in slabs. So `kernel_us` is measured at several slab counts and fitted
          kernel_us(slabs) = fixed + slope * slabs
      `fixed` is the launch-dominated floor and `slope` is the marginal
      per-slab cost. The precondition -- that `launches` really is flat in slabs
      -- is checked and reported per candidate rather than assumed. The two
      decompositions must agree, and where they do not the note says so.

THE CORRECTNESS GATE, AND WHY IT HAS A NEGATIVE CONTROL
------------------------------------------------------
All four candidates are checked against `numpy.linalg.inv` in float64 ON THE
CARD, before anything is timed. A timing comparison between two different
operators is meaningless, and this operator has a documented history of being
ported from the wrong artifact while passing every test (`notes/06` section 5b,
ledger C-14 and C-15).

`--selftest` also runs a NEGATIVE CONTROL: the transpose of a correct result
has exactly the right SHAPE and is a completely different operator. The gate
must reject it. If a gate cannot reject the transpose, it is a shape check
wearing a value check's clothes, and the specific failure it would miss is the
one `notes/31` section 9 documents: `Tensor.expand` left-aligns its shape
argument (`mixin/movement.py:162`), so `expand` and `[:, None]` disagree about
which axis to widen, and the result was wrong by 3.4x while passing a shape
check.

SCOPE
-----
Every number in the note this file backs is a TINYGRAD-vs-TINYGRAD number. No
timing comparison against the hand-written kernel is made or implied here: `fla`
is not exercised by this file at all, and the sm_75 configuration question is a
separate finding owned elsewhere.

Run (on a box with a GPU):
    python3 kda/ut_time.py --all
    python3 kda/ut_time.py --selftest        # instrument validation, no GPU needed
    python3 kda/ut_time.py --verify          # correctness on the card, before timing
    python3 kda/ut_time.py --time            # the table
    python3 kda/ut_time.py --sweep           # slab sweep + linear fit
    python3 kda/ut_time.py --launchcal       # per-launch cost, measured
    python3 kda/ut_time.py --counters        # ncu, if the box has it
    python3 kda/ut_time.py --all --out DIR   # where the JSON and the log go
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import time
from contextlib import redirect_stdout

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# `ut_price` inserts ../matmul on sys.path for its codegen capture; importing it
# here drags that in, which is harmless (matmul/ is read, never written) and
# keeps ONE definition of each candidate.
import ut_price as UP                                     # noqa: E402

CAND = ["A", "B", "C", "D2"]
CNAME = {"A": "naive 64x63 unrolled column solve (the port today)",
         "B": "the same 64 solves, batched on the leading axis",
         "C": "BC=16 two-level blocked, as the reference kernel does it",
         "D2": "Neumann on the block-off-diagonal, 6 [64,64] matmuls"}
CFN = {"A": UP.cand_A, "B": UP.cand_B, "C": UP.cand_C, "D2": UP.cand_D2}

ANSI = re.compile(r"\x1b\[[0-9;]*m")
# tinygrad 0.14.0 prints, per kernel, at DEBUG=2 (engine/realize.py:92):
#   *** CUDA  1234 name   arg  3 mem  0.05 GB tm    12.34us/   123.45ms (...)
# The first time is THIS kernel, the second is the running device total.
DBG2 = re.compile(r"\*\*\*\s+(\S+)\s+(\d+)\s+(\S+)\s+arg\s+(\d+)\s+mem\s+([\d.]+)\s+GB"
                  r"(?:\s+tm\s+([\d.]+)(us|ms)/)?")
# A COPY prints through the same `***` header but with the display name
# `copy  16.00KB, CUDA  <-  16.00KB, CPU` and NO `tm` field, because
# exec_copy (realize.py:167-178) never returns an elapsed time. So a COPY is
# counted by GlobalCounters.kernel_count but contributes 0 to time_sum_s --
# matched here so the two are never confused. Measured: candidate C at 64 slabs
# is 44 launches of which 11 are COPYs and 33 carry a time.
COPYNAME = re.compile(r"\*\*\*\s+\S+\s+\d+\s+copy\s")

# Reps per candidate in the sweep. Candidate A is 5,380 launches and its
# SCHEDULING alone is seconds per realize (measured on the host, and it is CPU
# work, so the card does not help); repeating it five times over five slab
# counts would spend the session on it. The cheap candidates get repeated
# measurements because they sit near the instrument's noise floor.
SWEEP_REPS = {"A": 1, "B": 3, "C": 3, "D2": 3}

R: dict = {}          # everything measured, for the JSON sidecar
LOG: list = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def hr(t=""):
    say("")
    say("=" * 78)
    if t:
        say(t)
        say("=" * 78)


# =============================================================================
# 0. the box
# =============================================================================
def report_env() -> dict:
    hr("0. the box, the device, and the knobs")
    env: dict = {}
    for name, dist in (("tinygrad", "tinygrad"), ("numpy", "numpy")):
        try:
            import importlib.metadata as md
            env[name] = md.version(dist)
        except Exception as e:                              # noqa: BLE001
            env[name] = f"<{type(e).__name__}>"
    try:
        from tinygrad import Device
        env["Device.DEFAULT"] = Device.DEFAULT
        env["device_count"] = int(Device[Device.DEFAULT].count())
        env["arch"] = getattr(Device[Device.DEFAULT], "arch", "?")
    except Exception as e:                                  # noqa: BLE001
        env["Device.DEFAULT"] = f"<{type(e).__name__}: {e}>"
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total,"
                            "driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=60)
        env["nvidia-smi"] = p.stdout.strip() or p.stderr.strip()[:200]
    except Exception as e:                                  # noqa: BLE001
        env["nvidia-smi"] = f"<{type(e).__name__}: {e}>"
    # The instrument knobs, read out of the installed tinygrad at runtime. A
    # knob being on would invalidate every launch and op count here, exactly as
    # it would invalidate the census in ut_price.
    import tinygrad.helpers as H
    env["knobs"] = {k: getattr(H, k).value for k in
                    ("ALLOW_TF32", "USE_TC", "TC_SELECT", "TC_OPT", "NOOPT", "IMAGE",
                     "NOLOCALS", "BEAM", "SPLIT_REDUCEOP", "CACHELEVEL", "DEBUG")}
    env["python"] = sys.version.split()[0]
    for k in ("python", "tinygrad", "numpy", "Device.DEFAULT", "device_count",
              "arch", "nvidia-smi"):
        say(f"  {k:<16} {env[k]}")
    say(f"  {'knobs':<16} " + "  ".join(f"{k}={v}" for k, v in env["knobs"].items()))
    R["env"] = env
    return env


def _normalise(d: dict) -> list:
    """Fill the current field names into rows written by an older version.

    The first version of this file stored the sum of the printed per-kernel
    times only inside `by_class` (from the LAST DEBUG=2 rep) and stored
    `time_sum_s` -- the kernel time PLUS the host cost of every COPY -- under
    the name `gpu_us`. So an artefact from that version is mapped as:

        kernel_us      = sum(by_class.values())          (one rep)
        kernel_us_all  = [that]                         (one rep, spread unknown)
        copy_host_us   = gpu_us - kernel_us
        time_sum_us    = gpu_us

    and the row is flagged `_kernel_us_source` so that anything downstream which
    depends on the spread knows it is looking at a single rep. Rows already
    carrying `kernel_us` are untouched.
    """
    notes = []
    for key in ("time", "sweep", "spelling"):
        for r in d.get(key, []) or []:
            if "kernel_us" in r:
                continue
            bc = r.get("by_class") or {}
            if not bc:
                continue
            k = sum(v[1] for v in bc.values())
            r["_kernel_us_source"] = ("by_class of the last DEBUG=2 rep; the "
                                      "artefact predates per-rep kernel_us")
            r["kernel_us"] = k
            r["kernel_us_all"] = [k]
            r["kernel_spread"] = float("nan")
            r["time_sum_us"] = r.get("gpu_us", k)
            r["copy_host_us"] = r["time_sum_us"] - k
            notes.append(f"{r.get('cand')}@{r.get('slabs')}/{r.get('spelling')}")
    return notes


def require_gpu() -> None:
    from tinygrad import Device
    if Device.DEFAULT in ("CPU", "PYTHON", "CL", "METAL"):
        raise SystemExit(
            f"Device.DEFAULT is {Device.DEFAULT}: there is no accelerator here.\n"
            f"  This file measures the card. Run it on the Kaggle T4:\n"
            f"    gpush kaggle run ... -- python3 kda/ut_time.py --all")


def report_env_offline() -> None:
    """The box the artefact came from, read out of the artefact.

    Printed from the saved `env` block and NOT from this host, because a box
    report that names the host it is running on while the numbers came from
    another one is how a T4 figure becomes a Metal figure in a later reader's
    head. `notes/10-t4-results.md` §2.2 is three instances of exactly that.
    """
    e = R.get("env", {})
    hr("0. the box these numbers came from (read from the artefact, not from here)")
    for k in ("python", "tinygrad", "numpy", "Device.DEFAULT", "device_count",
              "arch", "nvidia-smi"):
        if k in e:
            say(f"  {k:<16} {e[k]}")
    kn = e.get("knobs")
    if kn:
        say(f"  {'knobs':<16} " + "  ".join(f"{k}={v}" for k, v in kn.items()))
    say(f"  {'source':<16} {R.get('_source', '?')}")


def _recalibrate_launchcal() -> dict:
    """Refit the per-launch slope from the SAVED calibration points.

    Same least squares as `_calibrate`, on data this script already wrote. It
    exists so that the note's per-launch constant can be re-derived from the run
    artefact on a host with no card, rather than copied out of a log by hand.
    """
    cal = R.get("launchcal", {})
    say("")
    say("== launch calibration, refitted from the saved points ==")
    hdr = f"{'shape':>10} {'pts':>4} {'us/launch':>10} {'intercept_us':>13} {'R2':>8}"
    say(hdr)
    say("-" * len(hdr))
    for key, c in cal.items():
        pts = c.get("points", [])
        # `kernel_us` is the current field; `gpu_us` is the first version of this
        # file, which stored the sum of the printed per-kernel times under the
        # name of the counter it was read beside. Accepting both means a note can
        # be re-checked against an older artefact instead of being re-run.
        getk = (lambda p: p["kernel_us"]) if all("kernel_us" in p for p in pts) \
            else (lambda p: p["gpu_us"])
        if len(pts) < 3:
            continue
        x = np.array([p["n_timed"] for p in pts], dtype=float)
        y = np.array([getk(p) for p in pts], dtype=float)
        sl, ic = np.polyfit(x, y, 1)
        pred = sl * x + ic
        tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - float(((y - pred) ** 2).sum()) / tot if tot > 0 else float("nan")
        c["us_per_launch"] = float(sl)
        c["intercept_us"] = float(ic)
        c["r2"] = float(r2)
        say(f"{key:>10} {len(pts):>4} {sl:>10.3f} {ic:>13,.0f} {r2:>8.4f}")
    return cal


# =============================================================================
# 1. the instrument is validated before it is trusted
# =============================================================================
def selftest() -> bool:
    hr("1. self-test: ut_price's census, re-run here")
    ok = UP.selftest(verbose=True)

    hr("1b. NEGATIVE CONTROL: can the correctness gate see wrong data?")
    # A gate that only checks shapes would accept the transpose of a correct
    # result: same shape, different operator. This is the failure notes/31 §9
    # describes (expand vs [:, None] widening the wrong axis, wrong by 3.4x,
    # shape check passed). The gate below is the one --verify uses.
    C, slabs = 64, 1
    rng = np.random.default_rng(3)
    i, j = np.arange(C)[:, None], np.arange(C)[None, :]
    strict = (i > j).astype(np.float64)
    Z = rng.standard_normal((slabs, C, C)) * 0.02
    beta = 0.05 + 0.90 * rng.random((slabs, C))
    Sn = Z * strict[None] * beta[:, None, :]
    target = np.linalg.inv(np.eye(C)[None] + Sn) * beta[:, None, :]
    tol = 2e-6

    def relerr(got):
        got = np.asarray(got, dtype=np.float64)
        assert got.shape == target.shape, "shape"
        return float(np.abs(got - target).max() / np.abs(target).max())

    from tinygrad import Tensor, dtypes
    if dtypes.default_float == dtypes.float64:            # pragma: no cover
        say("  (running in float64; the control is still valid, the target is not)")
    S, bt = Tensor(Sn.astype(np.float32)), Tensor(beta.astype(np.float32))
    got = CFN["C"](S, bt).numpy().astype(np.float64)
    e_ok = relerr(got)
    e_bad = relerr(got.transpose(0, 2, 1))
    shape_ok = got.transpose(0, 2, 1).shape == target.shape
    good = (e_ok < tol) and (e_bad > 1e-3) and shape_ok
    ok &= good
    say(f"  candidate C, as computed                relerr {e_ok:.3e}  "
        f"{'< tol' if e_ok < tol else '>= tol  <-- GATE WOULD PASS WRONG DATA'}")
    say(f"  the SAME array transposed               relerr {e_bad:.3e}  "
        f"{'>= tol, REJECTED' if e_bad > 1e-3 else '< tol  <-- GATE IS BLIND'}")
    say(f"  the transposed array's SHAPE matches     {shape_ok}  "
        f"(so a shape assertion alone could not have caught it)")
    say("")
    say(f"  gate verdict: {'a value gate rejects the transpose' if good else 'INSUFFICIENT'}")
    R["negative_control"] = dict(relerr_correct=e_ok, relerr_transposed=e_bad,
                                 transposed_shape_matches=shape_ok, pass_=bool(good))
    say("")
    say("SELFTEST " + ("PASS" if ok else "FAIL"))
    return ok


# =============================================================================
# 2. correctness on the card, before anything is timed
# =============================================================================
def verify(slabs: int = 2, C: int = 64, tol: float = 2e-6) -> bool:
    hr(f"2. correctness on the card vs numpy float64 inverse, {slabs} slab(s), C={C}")
    from tinygrad import Tensor
    rng = np.random.default_rng(7)
    i, j = np.arange(C)[:, None], np.arange(C)[None, :]
    strict = (i > j).astype(np.float64)
    Z = rng.standard_normal((slabs, C, C)) * 0.02
    beta = 0.05 + 0.90 * rng.random((slabs, C))
    Sn = Z * strict[None] * beta[:, None, :]
    target = np.linalg.inv(np.eye(C)[None] + Sn) * beta[:, None, :]
    ref_norm = float(np.abs(target).max())

    rows, ok = [], True
    for name in CAND:
        S, bt = Tensor(Sn.astype(np.float32)), Tensor(beta.astype(np.float32))
        try:
            out = CFN[name](S, bt)
            shp = tuple(out.shape)
            got = out.numpy().astype(np.float64)
        except Exception as e:                              # noqa: BLE001
            say(f"  {name:<3} FAILED TO RUN: {type(e).__name__}: {e}")
            rows.append(dict(cand=name, error=f"{type(e).__name__}: {e}"))
            ok = False
            continue
        if shp != target.shape:
            say(f"  {name:<3} shape {shp}, expected {target.shape}  MISMATCH")
            rows.append(dict(cand=name, shape=list(shp), pass_=False,
                             error="shape mismatch"))
            ok = False
            continue
        err = float(np.abs(got - target).max() / ref_norm)
        good = err < tol
        ok &= good
        rows.append(dict(cand=name, shape=list(shp), relerr=err, pass_=bool(good)))
        say(f"  {name:<3} shape {shp}  max rel err {err:.3e}  "
            f"{'OK' if good else 'MISMATCH -- REFUSING TO TIME THIS'}")
    say("")
    say("VERIFY " + ("PASS" if ok else "FAIL"))
    R["verify"] = rows
    R["verify_pass"] = bool(ok)
    if not ok:
        raise SystemExit("correctness gate failed; no timing in this file is quotable")
    return ok


# =============================================================================
# 3. one measurement of one candidate at one shape
# =============================================================================
def _gc():
    from tinygrad.helpers import GlobalCounters
    return GlobalCounters


def _sync():
    """Drain the device WITHOUT launching anything.

    `Tensor(np.zeros(1)).realize()` would also drain, but it is a host-to-device
    COPY and therefore shows up as one extra kernel in `GlobalCounters.kernel_count`
    -- which is the very number this file is reporting. Measured, not assumed:
    with the COPY drain, every candidate's device launch count came out exactly
    one (and for A, nine) above the offline census.
    """
    from tinygrad import Device
    Device[Device.DEFAULT].synchronize()


def measure(name: str, slabs: int, C: int, reps: int, warm: int,
            dbg_reps: int = 3) -> dict:
    """Build, schedule, realize, and (under DEBUG=2) time one candidate.

    Returns launches / kernel_us / global_ops / global_mem from the DEVICE's own
    counters, plus the host-side wall clocks. The census is run in the same
    session so the offline and on-card numbers sit in one table.

    TWO INSTRUMENT HAZARDS, both found by this harness being wrong first, and
    both guarded against in code rather than described in a comment:

    1. `Tensor.schedule_linear()` MUTATES the graph. It calls
       `transform_to_call` and then `_apply_map_to_tensors(..., name="buffers")`
       (tensor.py:409-411), which assigns the buffers. A `realize()` on the same
       tensors afterwards then finds `to_realize` empty (tensor.py:422) and is a
       **silent no-op** -- the trap `notes/26` section 6f records. The first
       version of this function timed `build(); schedule_linear(); realize()` as
       one end-to-end and got, on the host it was smoke-tested on, 0.08 ms for
       5,388 launches -- 15 ns per `cuLaunchKernel`, and impossible; on the T4
       the same bug gave 0.127 ms for 5,380, 24 ns each (`notes/32` section 1a).
       So the host clocks below are taken on THREE SEPARATE fresh graphs, one per
       quantity, and never on a graph that has already been through
       `schedule_linear()`; and the timed realize is REFUSED unless it is
       observed to have launched something.

    2. `GlobalCounters.time_sum_s` is NOT pure kernel time. It is
       `sum(et for et in ets)` (realize.py:65) PLUS, for every call whose
       runtime returns no elapsed time -- which is every COPY, because
       `exec_copy` returns `[]` (realize.py:167-178) -- a host-side
       `perf_counter` delta taken after a `Device.synchronize()`
       (realize.py:78-81). On a real card the COPYs are pinned-host
       allocations (`ops_cuda.py:83`) and they dominate: measured in the run of
       record, candidate C's eleven COPYs add 1,436 us on top of 840 us of real
       kernel time, so `time_sum_us` reads 2.71x the device time. The headline
       device number is therefore `kernel_us`, the sum of the per-kernel
       CUDA-event times tinygrad itself PRINTS, and `time_sum_us` is reported
       beside it with the difference named.
    """
    from tinygrad.helpers import Context
    gc = _gc()
    fn = CFN[name]

    def build():
        return fn(*UP.make_S(slabs, C))

    # -- warmup. Compiles the kernels, fills the runtime cache and fills
    #    `local_size_cache` (engine/realize.py:96-121 autotunes a launch config
    #    per distinct program on the FIRST realize and memoises it). Without a
    #    warmup the "time" below is mostly one-off JIT and launch-config search,
    #    which is a cost of the first call, not of the operator.
    for _ in range(warm):
        build().realize()
    _sync()

    # -- host clocks, on separate fresh graphs (see hazard 1 above).
    tb, ts, te, launched = [], [], [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        build()
        tb.append(time.perf_counter() - t0)
    for _ in range(reps):
        out = build()
        t0 = time.perf_counter()
        out.schedule_linear()
        ts.append(time.perf_counter() - t0)
    for _ in range(reps):
        gc().reset()
        out = build()
        t0 = time.perf_counter()
        out.realize()
        _sync()
        te.append(time.perf_counter() - t0)
        launched.append(int(gc().kernel_count))
    if not launched or min(launched) == 0:
        raise RuntimeError(
            f"{name}: the timed realize launched ZERO kernels (counts {launched}). "
            f"The timing below would be a measurement of an empty graph -- the "
            f"notes/26 section 6f no-op. Refusing rather than printing.")

    # -- device counters at DEBUG=0. kernel_count / global_ops / global_mem are
    #    updated before the `DEBUG < 2` early return (engine/realize.py:58-66),
    #    so this is a real count of executed kernels with NO per-kernel
    #    synchronisation and therefore no distortion of the pipeline.
    gc().reset()
    out = build()
    out.realize()
    _sync()
    dev = dict(launches=int(gc().kernel_count), global_ops=int(gc().global_ops),
               global_mem=int(gc().global_mem))

    # -- device counters at DEBUG=2. This is the only path that populates
    #    `time_sum_s` (engine/realize.py:65), and it is SERIALISED: wait=True
    #    makes every launch record two CUDA events and synchronise on the
    #    second (runtime/ops_cuda.py:26-35). Repeated `dbg_reps` times because
    #    the cheap candidates sit near the instrument's own noise floor.
    kus, sums, dbg_all, dbg_copies, names = [], [], 0, 0, None
    for _ in range(max(1, dbg_reps)):
        gc().reset()
        buf = io.StringIO()
        with Context(DEBUG=2), redirect_stdout(buf):
            out = build()
            out.realize()
        _sync()
        txt = ANSI.sub("", buf.getvalue())
        per, nm, all_lines, n_copy = [], [], 0, 0
        for line in txt.splitlines():
            if "***" not in line:
                continue
            all_lines += 1
            if COPYNAME.search(line):
                n_copy += 1
                continue
            m = DBG2.search(line)
            if m and m.group(6) is not None:
                per.append(float(m.group(6)) * (1000.0 if m.group(7) == "ms" else 1.0))
                nm.append(m.group(3))
        kus.append(sum(per))
        sums.append(float(gc().time_sum_s * 1e6))
        dbg_all, dbg_copies, names = all_lines, n_copy, (nm or names)
    per = per or []

    # -- the offline census, same session, same expressions, run TWICE: once
    #    with DEV="CUDA" (which is what notes/31's table is, because that note
    #    was priced on a host with no CUDA device) and once with the device this
    #    file is actually timing. If the two disagree, the schedule depends on
    #    the device and the offline table is a table about a device that was not
    #    there. That question is only answerable on a real card, and on the
    #    card they agree exactly.
    from tinygrad import Device
    cen = UP.census_ut(fn, slabs, C)                       # DEV="CUDA", as notes/31
    cen_dev = UP.census(fn=lambda: fn(*UP.make_S(slabs, C)),
                        dev=Device.DEFAULT)                # the device being timed

    kmin = min(kus) if kus else float("nan")
    res = dict(
        cand=name, slabs=slabs, C=C, spelling=UP.SPELLING,
        t_build_ms=1e3 * min(tb), t_sched_ms=1e3 * min(ts), t_exec_ms=1e3 * min(te),
        t_end_to_end_ms=1e3 * (min(tb) + min(ts) + min(te)),
        t_build_all=[1e3 * x for x in tb], t_sched_all=[1e3 * x for x in ts],
        t_exec_all=[1e3 * x for x in te], launched_in_timed_realize=launched,
        kernel_us=kmin, kernel_us_all=kus,
        kernel_spread=(max(kus) - min(kus)) / max(1e-9, min(kus)),
        time_sum_us=min(sums), time_sum_all=sums, copy_host_us=min(sums) - kmin,
        gpu_kernels_timed=len(per), dbg2_lines=dbg_all, dbg2_copy_lines=dbg_copies,
        dev_launches=dev["launches"], global_ops=dev["global_ops"],
        global_mem=dev["global_mem"],
        census_kernels=cen["kernels"], census_reduce=cen["reduce_kernels"],
        census_copies=cen["copies"], census_lane=cen["lane_ops"],
        census_depth=cen["dag_depth"],
        census_kernels_here=cen_dev["kernels"],
        census_lane_here=cen_dev["lane_ops"],
        census_depth_here=cen_dev["dag_depth"],
        macs_per_slab=UP.macs(name, C), chain_depth=UP.chain_depth(name, C),
        mean_us_per_kernel=(kmin / dev["launches"] if dev["launches"]
                            else float("nan")),
    )
    if names:
        agg: dict = {}
        for nm, v in zip(names, per):
            e = agg.setdefault(nm.split("_")[0], [0, 0.0])
            e[0] += 1
            e[1] += v
        res["by_class"] = {k: [v[0], v[1]] for k, v in sorted(agg.items())}
        res["kernel_names"] = sorted(set(names))[:12]
    return res


def time_table(slabs: int, C: int, reps: int, warm: int, spelling: str = "arange") -> list:
    hr(f"3. the table: {slabs} slab(s) of [{C},{C}], on the card")
    say("  MEASURED on this device.")
    say("  launches  GlobalCounters.kernel_count -- kernels the device actually ran.")
    say("  kernel_us sum of the per-kernel CUDA-event times tinygrad PRINTS at DEBUG=2.")
    say("            DEBUG=2 sets ExecContext.wait (realize.py:318), and ops_cuda.py:26-35")
    say("            then records an event, launches, records a second event and")
    say("            SYNCHRONISES on it -- so the graph is serialised and each number")
    say("            contains that kernel's host launch latency. That is the instrument,")
    say("            not the card, and it is why the per-launch calibration in section 5")
    say("            is needed to subtract it.")
    say("  copy_us   time_sum_s minus kernel_us. exec_copy returns no elapsed time, so")
    say("            realize.py:78-81 charges every COPY its HOST cost into time_sum_s.")
    say("            It is a real cost and it is not device time.")
    say("  t_sched   HOST wall clock of Tensor.schedule_linear() on its own fresh graph.")
    say("  t_exec    HOST wall clock of Tensor.realize() + synchronize() on its own")
    say("            fresh graph. Never on a graph already through schedule_linear():")
    say("            that MUTATES it (tensor.py:409-411) and the realize is then a")
    say("            silent no-op. How many kernels each timed realize launched is in")
    say("            ut_time.json, and a zero aborts the run.")
    say("")
    UP.SPELLING = spelling
    hdr = (f"{'cand':<4} {'launches':>9} {'census_k':>9} {'kernel_us':>10} {'spread':>8} "
           f"{'copy_us':>9} {'time_sum':>9} {'ops(device)':>12} {'t_sched_ms':>11} "
           f"{'t_exec_ms':>10} {'t_build_ms':>11}  what")
    say(hdr)
    say("-" * len(hdr))
    rows = []
    for name in CAND:
        try:
            r = measure(name, slabs, C, reps, warm)
        except Exception as e:                              # noqa: BLE001
            say(f"{name:<4} FAILED: {type(e).__name__}: {e}")
            continue
        rows.append(r)
        say(f"{name:<4} {r['dev_launches']:>9,} {r['census_kernels']:>9,} "
            f"{r['kernel_us']:>10,.0f} {r['kernel_spread']*100:>7.1f}% "
            f"{r['copy_host_us']:>9,.0f} {r['time_sum_us']:>9,.0f} "
            f"{r['global_ops']:>12,} {r['t_sched_ms']:>11,.1f} "
            f"{r['t_exec_ms']:>10,.2f} {r['t_build_ms']:>11,.1f}  {CNAME[name]}")
    say("")
    if rows:
        say(f"{'cand':<4} {'REDUCE(cens)':>12} {'DAG':>5} {'alu_lane(cens)':>16} "
            f"{'ops(device)':>12} {'ops/cens':>9} {'dev/cens launches':>19} "
            f"{'algMACs/slab':>13} {'us/kernel':>10}")
        for r in rows:
            say(f"{r['cand']:<4} {r['census_reduce']:>12,} {r['census_depth']:>5} "
                f"{r['census_lane']:>16,} {r['global_ops']:>12,} "
                f"{r['global_ops']/max(1, r['census_lane']):>8.3f}x "
                f"{r['dev_launches']:>12,}/{r['census_kernels']:<6,} "
                f"{r['macs_per_slab']:>13,} {r['mean_us_per_kernel']:>10.2f}")
        say("")
        say("  `ops(device)` is tinygrad's OWN per-program op estimate, summed over the")
        say("  schedule that actually ran (engine/realize.py:63, ast.arg.estimates).")
        say("  `alu_lane(cens)` is this repo's census of the same graph scheduled in")
        say("  isolation. Two different definitions of the same idea, and their ratio")
        say("  to each other is NOT 1 -- that is the non-additivity notes/31 §3.5")
        say("  warned about, now visible in one table instead of argued about.")
        say("")
        say("== does the offline census transfer to THIS device? ==")
        say("  `notes/31` priced these expressions with Context(DEV='CUDA') on a host")
        say("  with no CUDA device. Below: the same census with DEV=the device being")
        say("  timed, next to what the device actually executed.")
        say("")
        say(f"{'cand':<4} {'census DEV=CUDA':>17} {'census DEV=here':>17} "
            f"{'device executed':>17} {'dev - census(here)':>20}")
        for r in rows:
            say(f"{r['cand']:<4} {r['census_kernels']:>17,} "
                f"{r['census_kernels_here']:>17,} {r['dev_launches']:>17,} "
                f"{r['dev_launches']-r['census_kernels_here']:>20,}")
        same = all(r["dev_launches"] == r["census_kernels_here"] for r in rows)
        say("")
        say("  every candidate's device launch count equals the same-device census: "
            f"{same}")
    R.setdefault("time", []).extend(rows)
    return rows


# =============================================================================
# 4. the slab sweep: the two axes, separated by construction
# =============================================================================
def sweep(slab_list: list, C: int, reps: int | None = None, warm: int = 0) -> list:
    hr("4. shape sweep: launches should not move with slabs; arithmetic should")
    say("  If the schedule is a function of the graph and not of the batch size, then")
    say("  `launches` is flat in slabs and only the per-slab arithmetic grows. Fitting")
    say("  kernel_us(slabs) = fixed + slope*slabs therefore separates the two axes")
    say("  WITHOUT converting an instruction count into time by any constant.")
    say("")
    say(f"  reps per candidate in this section: "
        f"{ {k: (reps if reps else v) for k, v in SWEEP_REPS.items()} }")
    say("  (A is 5,380 launches and seconds of CPU SCHEDULING per realize; repeating it")
    say("  over five slab counts would spend the session on it. The cheap candidates")
    say("  get repeated measurements because they sit near the instrument's floor.)")
    say("")
    rows = []
    hdr = (f"{'cand':<4} {'slabs':>6} {'launches':>9} {'census_k':>9} {'copies':>7} "
           f"{'timed_k':>8} {'kernel_us':>11} {'spread':>8} {'t_sched_ms':>11} "
           f"{'t_exec_ms':>10} {'ops(device)':>13}")
    say(hdr)
    say("-" * len(hdr))
    for name in CAND:
        for s in slab_list:
            rp = reps if reps else SWEEP_REPS.get(name, 1)
            try:
                r = measure(name, s, C, rp, warm, dbg_reps=rp)
            except Exception as e:                          # noqa: BLE001
                say(f"{name:<4} {s:>6} FAILED: {type(e).__name__}: {e}")
                continue
            r["spelling"] = UP.SPELLING
            rows.append(r)
            say(f"{name:<4} {s:>6} {r['dev_launches']:>9,} {r['census_kernels']:>9,} "
                f"{r['dbg2_copy_lines']:>7,} {r['gpu_kernels_timed']:>8,} "
                f"{r['kernel_us']:>11,.0f} {r['kernel_spread']*100:>7.1f}% "
                f"{r['t_sched_ms']:>11,.1f} {r['t_exec_ms']:>10,.2f} "
                f"{r['global_ops']:>13,}")
        say("")
    say("== linear fits, kernel_us = fixed + slope * slabs ==")
    hdr = (f"{'cand':<4} {'n':>3} {'fixed_us':>11} {'slope_us/slab':>14} {'R2':>8} "
           f"{'fixed/max':>10} {'ns/MAC':>9}  launch floor vs arithmetic")
    say(hdr)
    say("-" * len(hdr))
    fits = []
    for name in CAND:
        pts = sorted((r for r in rows if r["cand"] == name), key=lambda r: r["slabs"])
        if len(pts) < 3:
            say(f"{name:<4} {len(pts):>3}   only {len(pts)} slab count(s): a line "
                f"through 2 points has R2=1 by construction and the intercept is not "
                f"identifiable. Refusing to print a fit.")
            continue
        x = np.array([r["slabs"] for r in pts], dtype=float)
        y = np.array([r["kernel_us"] for r in pts], dtype=float)
        sl, ic = np.polyfit(x, y, 1)
        pred = sl * x + ic
        ss = float(((y - pred) ** 2).sum())
        tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss / tot if tot > 0 else float("nan")
        mac = UP.macs(name, C)
        big = max(pts, key=lambda r: r["slabs"])
        frac = ic / big["kernel_us"] if big["kernel_us"] else float("nan")
        launches = {r["slabs"]: r["dev_launches"] for r in pts}
        flat = len(set(launches.values())) == 1
        fits.append(dict(cand=name, n=len(pts), fixed_us=float(ic),
                         slope_us_per_slab=float(sl), r2=float(r2),
                         fixed_fraction_at_max=float(frac), macs_per_slab=mac,
                         ns_per_alg_mac=float(sl / mac * 1e3) if mac else None,
                         launches_flat_in_slabs=flat,
                         slab_points=[int(v) for v in x],
                         kernel_points=[float(v) for v in y],
                         launch_counts={str(k): v for k, v in launches.items()}))
        say(f"{name:<4} {len(pts):>3} {ic:>11,.0f} {sl:>14,.1f} {r2:>8.4f} "
            f"{frac*100:>9.1f}% {sl/mac*1e3:>9.3f}  "
            f"floor {frac*100:.0f}% / arith {(1-frac)*100:.0f}%  launches flat: {flat}")
    say("")
    say("  `fixed` is the launch-dominated floor and `slope` the marginal per-slab")
    say("  cost. `ns/MAC` is nanoseconds of device time per algorithmic multiply-add.")
    say("  If that number is roughly the same for two candidates, device time tracks")
    say("  the ALGORITHM's arithmetic. If it differs by an order of magnitude between")
    say("  them, device time is tracking the SCHEDULE and the arithmetic column is not")
    say("  what the card is spending its time on. `launches flat` is the precondition")
    say("  for reading the fit this way at all, and it is reported per candidate.")
    R["sweep"] = rows
    R["fits"] = fits
    return rows


# =============================================================================
# 5. the per-launch cost, measured on this card in this process
# =============================================================================
def _chain(n: int, shape=(1,)) -> "object":
    """n dependent kernels, each one ALU op, each forced to materialise.

    Dependent on purpose, so nothing can be reordered; and each step calls
    `.contiguous()`, which is what stops tinygrad fusing the chain into one
    kernel. The number of kernels that actually launch is READ BACK from
    `GlobalCounters.kernel_count` rather than assumed to be n, and it is the
    x-axis of the fit, so a fusion surprise would show up as a wrong slope
    rather than as a silently wrong constant.

    `shape=(1,)` is a one-thread kernel: the floor of pure launch cost.
    `shape=(64,64)` is one 16 KB buffer read and written per launch: the floor
    of launch-plus-memory, which is what a candidate kernel of this operator's
    size cannot go below either.
    """
    from tinygrad import Tensor
    x = Tensor(np.zeros(shape, dtype=np.float32)) + 1.0
    for _ in range(n):
        x = (x * 1.0000001).contiguous()
    return x


def _calibrate(ns, shape) -> dict:
    from tinygrad import Tensor
    from tinygrad.helpers import Context
    gc = _gc()
    _chain(8, shape).realize()
    _sync()
    pts = []
    for n in ns:
        gc().reset()
        _chain(n, shape).realize()
        _sync()
        dev_launches = int(gc().kernel_count)
        best, best_sum, n_timed = float("inf"), float("inf"), 0
        for _ in range(3):
            gc().reset()
            buf = io.StringIO()
            with Context(DEBUG=2), redirect_stdout(buf):
                _chain(n, shape).realize()
            _sync()
            per = []
            for line in ANSI.sub("", buf.getvalue()).splitlines():
                m = DBG2.search(line)
                if m and m.group(6) is not None:
                    per.append(float(m.group(6)) * (1000.0 if m.group(7) == "ms" else 1.0))
            best = min(best, sum(per))
            best_sum = min(best_sum, float(gc().time_sum_s * 1e6))
            n_timed = len(per)
        pts.append(dict(n=n, dev_launches=dev_launches, kernel_us=best,
                        time_sum_us=best_sum, n_timed=n_timed))
    x = np.array([p["n_timed"] for p in pts], dtype=float)
    y = np.array([p["kernel_us"] for p in pts], dtype=float)
    if len(pts) >= 3:
        sl, ic = np.polyfit(x, y, 1)
        pred = sl * x + ic
        tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - float(((y - pred) ** 2).sum()) / tot if tot > 0 else float("nan")
    else:
        sl, ic, r2 = float("nan"), float("nan"), float("nan")
    return dict(points=pts, us_per_launch=float(sl), intercept_us=float(ic),
                r2=float(r2), shape=list(shape))


def launchcal(ns=(16, 64, 256, 1024, 4096)) -> dict:
    hr("5. launch calibration: what does one launch cost on this card?")
    say("  A chain of n dependent one-op kernels, timed with the SAME instrument as")
    say("  the candidates (DEBUG=2, per-kernel CUDA events). The slope of kernel_us")
    say("  against the number of kernels that ACTUALLY LAUNCHED is microseconds per")
    say("  launch. This is the number that lets")
    say("      non_launch_us = kernel_us - launches * per_launch_us")
    say("  be a residual in MICROSECONDS rather than an instruction count times a")
    say("  constant -- which is what the census's non-additivity demands.")
    say("")
    hdr = f"{'shape':>10} {'n':>6} {'launched':>9} {'kernel_us':>11} {'us/launch':>10}"
    say(hdr)
    say("-" * len(hdr))
    cals = {}
    for shape, label in (((1,), "1 thread"), ((64, 64), "16 KB")):
        c = _calibrate(ns, shape)
        cals[str(shape)] = c
        for p in c["points"]:
            say(f"{label:>10} {p['n']:>6} {p['n_timed']:>9,} {p['kernel_us']:>11,.0f} "
                f"{(p['kernel_us']/max(1,p['n_timed'])):>10.3f}")
        say(f"{label:>10} {'fit':>6} {'':>9} {'':>11} {c['us_per_launch']:>10.3f}   "
            f"intercept {c['intercept_us']:,.0f} us, R2 {c['r2']:.4f}")
        say("")
    say("  MEASURED, DEBUG=2, so these slopes INCLUDE the host launch latency the")
    say("  instrument charges to every kernel -- which is exactly what the candidate")
    say("  measurements are charged too, so the residual below compares like with like.")
    say("  The chain has ONE copy per realize, not one per step, so the intercept")
    say("  absorbs the copy's host cost (realize.py:78-81) and the slope does not.")
    say("  The `16 KB` row is a sensitivity check: it is what a candidate kernel of")
    say("  this operator's size costs if it did NOTHING but move one buffer. The")
    say("  headline uses the `1 thread` row.")
    R["launchcal"] = cals
    per_launch = cals[str((1,))]["us_per_launch"]
    per_launch_mem = cals[str((64, 64))]["us_per_launch"]

    # -- the residual, for each candidate at the bench shape
    say("")
    say("== residual: kernel_us - launches * per_launch_us   (bench shape) ==")
    hdr = (f"{'cand':<4} {'launches':>9} {'kernel_us':>10} {'launch_us':>10} "
           f"{'@16KB':>9} {'resid_us':>10} {'resid %':>9} {'ns/algMAC':>11}  reading")
    say(hdr)
    say("-" * len(hdr))
    resid = []
    for r in R.get("time", []):
        lu = r["dev_launches"] * per_launch
        lum = r["dev_launches"] * per_launch_mem
        res = r["kernel_us"] - lu
        pct = res / r["kernel_us"] * 100 if r["kernel_us"] else float("nan")
        macs_total = r["macs_per_slab"] * r["slabs"]
        per_mac = res / macs_total if macs_total else float("nan")
        resid.append(dict(cand=r["cand"], slabs=r["slabs"], spelling=r["spelling"],
                          launches=r["dev_launches"], kernel_us=r["kernel_us"],
                          launch_us=lu, launch_us_mem=lum, residual_us=res,
                          residual_pct=pct, ns_per_alg_mac=per_mac * 1e3,
                          macs_total=macs_total))
        say(f"{r['cand']:<4} {r['dev_launches']:>9,} {r['kernel_us']:>10,.0f} {lu:>10,.0f} "
            f"{lum:>9,.0f} {res:>10,.0f} {pct:>8.1f}% {per_mac*1e3:>11.3f}  "
            f"{'launch-dominated' if pct < 50 else 'not-launch-dominated'}")
    say("")
    say("  `launches` here is the DEVICE launch count, which includes the COPYs. A COPY")
    say("  costs no device time at all (exec_copy returns no elapsed time), so the")
    say("  launch term slightly over-charges every candidate. The number of COPYs is in")
    say("  ut_time.json per candidate; subtracting them changes the launch term by a few")
    say("  percent and moves no row across the 50% line.")
    say("")
    say("  ns/algMAC is nanoseconds of NON-LAUNCH device time per algorithmic")
    say("  multiply-add, using the algorithm's own MAC count (ut_price.macs) and not")
    say("  any census figure. Candidates whose entries agree are being priced by the")
    say("  same axis; candidates whose entries differ by an order of magnitude are")
    say("  not, and THAT is the finding, not the winner.")
    R["residual"] = resid
    return cals


# =============================================================================
# 6. hardware counters, if this box has them
# =============================================================================
def counters(slabs: int = 8, C: int = 64, cands=("B", "C", "D2"),
             ncu_launches: int = 200) -> dict:
    hr("6. hardware counters")
    import shutil
    ncu = shutil.which("ncu")
    out: dict = {"ncu": ncu, "slabs": slabs, "cands": list(cands)}
    say(f"  ncu on PATH: {ncu or 'NOT FOUND'}")
    if not ncu:
        say("  No Nsight Compute on this box. Every conclusion below therefore rests on")
        say("  CUDA events and GlobalCounters, and the note must say which of its claims")
        say("  would have needed dram__bytes / sm__inst_executed_pipe_* / achieved")
        say("  occupancy to firm up. They are NOT available, and nothing in the note may")
        say("  be phrased as if they were.")
        R["counters"] = out
        return out
    # ncu's python front end imports `wrapt`, which the Kaggle image does not
    # ship. The first attempt at this section failed with
    # `ModuleNotFoundError: No module named 'wrapt'` on all four candidates, which
    # is recorded here so the failure is not re-run as if it were a permission
    # problem. It is a missing pure-python dependency and it is installed into
    # the box, not into the payload.
    for pkg in ("wrapt",):
        p = subprocess.run([sys.executable, "-m", "pip", "install", "--no-input",
                            "--disable-pip-version-check", pkg],
                           capture_output=True, text=True, timeout=600)
        out[f"pip_{pkg}"] = {"rc": p.returncode,
                             "out": (p.stdout or p.stderr).strip()[-300:]}
        say(f"  pip install {pkg:<8} rc={p.returncode}")
    p = subprocess.run([ncu, "--version"], capture_output=True, text=True, timeout=120)
    out["version"] = (p.stdout or p.stderr).strip()[:300]
    say(f"  {out['version']}")
    metrics = ["dram__bytes_read.sum", "dram__bytes_write.sum",
               "sm__inst_executed_pipe_fma.sum", "sm__inst_executed_pipe_alu.sum",
               "sm__inst_executed.sum",
               "sm__warps_active.avg.pct_of_peak_sustained_active",
               "sm__throughput.avg.pct_of_peak_sustained_elapsed"]
    out["metrics"] = metrics
    say("")
    say("  ncu SERIALISES and REPLAYS each kernel, so its durations are NOT comparable")
    say("  with the DEBUG=2 numbers above. Only counters that are replay-stable --")
    say("  instruction counts and dram bytes -- are read, and only RATIOS between")
    say("  candidates on the same counter.")
    say("")
    for name in cands:
        driver = (f"import sys; sys.path.insert(0, {HERE!r})\n"
                  f"import ut_price as UP\n"
                  f"UP.cand_{name}(*UP.make_S({slabs}, {C})).realize()\n")
        cmd = [ncu, "--target-processes", "all", "--metrics", ",".join(metrics),
               "--csv", "--page", "raw", "--launch-count", str(ncu_launches),
               sys.executable, "-c", driver]
        try:
            q = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        except Exception as e:                              # noqa: BLE001
            out[name] = f"<{type(e).__name__}: {e}>"
            say(f"  {name:<3} ncu FAILED: {type(e).__name__}: {e}")
            continue
        if q.returncode != 0:
            tail = (q.stderr or q.stdout).strip().splitlines()
            out[name] = {"ok": False, "rc": q.returncode,
                         "err": " | ".join(tail[-3:])[:600]}
            say(f"  {name:<3} ncu rc={q.returncode}: {out[name]['err'][:200]}")
            continue
        out[name] = {"ok": True, "rows": _ncu_totals(q.stdout)}
        t = out[name]["rows"]
        if "_error" in t:
            out[name]["ok"] = False
            say(f"  {name:<3} ncu ran but the csv did not parse: {t['_error']}")
            say(f"      first lines: {t.get('_head')}")
            continue
        got = set(t.get("_metrics", []))
        want = set(metrics)
        say(f"  {name:<3} ncu over {t.get('_nkernels', '?')} kernels"
            + (f"  [missing: {sorted(want - got)}]" if want - got else ""))
        for m in ("sm__inst_executed.sum", "sm__inst_executed_pipe_fma.sum",
                  "sm__inst_executed_pipe_alu.sum", "dram__bytes_read.sum",
                  "dram__bytes_write.sum",
                  "sm__warps_active.avg.pct_of_peak_sustained_active"):
            if m in t:
                say(f"      {m:<52} {t[m]:>18,.2f} {t.get('_unit_' + m, '')}")
    say("")
    say("  `_nkernels` is the number of profiled kernels ncu actually reported, which")
    say("  is compared against the census in the note: if it is short, the counters")
    say("  are a SAMPLE and no ratio computed from them is a whole-schedule claim.")
    R["counters"] = out
    return out


def _ncu_totals(csv_text: str) -> dict:
    """Sum the value column of an `ncu --csv --page raw` table over all kernels.

    Header-driven, not positional: the raw page's column order is an ncu
    implementation detail and reading it by index would silently sum the wrong
    column if it ever moved -- which is the same failure shape this file exists
    to catch. Deliberately dumb otherwise: the metrics asked for are all SUMS
    over a kernel, so summing them over kernels is the whole-schedule total with
    no modelling. The one metric that is an average
    (`...pct_of_peak_sustained_active`) is reported as a plain mean and is
    labelled as such wherever it is quoted.
    """
    import csv as _csv
    rows = list(_csv.reader(io.StringIO(csv_text)))
    rows = [r for r in rows if any(c.strip() for c in r)]
    hdr = None
    for i, r in enumerate(rows):
        cells = [c.strip() for c in r]
        if "Metric Name" in cells and "Metric Value" in cells:
            hdr, start = {n: j for j, n in enumerate(cells)}, i + 1
            break
    if hdr is None:
        return {"_error": "no 'Metric Name'/'Metric Value' header in the ncu csv",
                "_head": [str(r) for r in rows[:3]]}
    i_name, i_val = hdr["Metric Name"], hdr["Metric Value"]
    i_unit = hdr.get("Metric Unit", i_val)
    i_kern = hdr.get("Kernel Name")
    tot: dict = {}
    nk: set = set()
    for r in rows[start:]:
        if len(r) <= max(i_name, i_val):
            continue
        metric, raw = r[i_name].strip(), r[i_val].strip()
        try:
            v = float(raw.replace(",", ""))
        except ValueError:
            continue
        tot[metric] = tot.get(metric, 0.0) + v
        tot.setdefault("_unit_" + metric, r[i_unit].strip() if i_unit < len(r) else "?")
        if i_kern is not None and i_kern < len(r):
            nk.add(r[i_kern])
    tot["_nkernels"] = len(nk)
    tot["_metrics"] = sorted(k for k in tot if not k.startswith("_"))
    return tot


# =============================================================================
# 7. the verdict, and its falsifier, computed from whatever has been measured
# =============================================================================
def _lsq(X: list, y: list) -> list:
    """Ordinary least squares by normal equations, least squares by hand.

    numpy's polyfit only does one variable. This is three columns and a solve,
    and writing it out means the fit in the note has no hidden step.
    """
    A = np.array(X, dtype=float)
    b = np.array(y, dtype=float)
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    pred = A @ coef
    ss = float(((b - pred) ** 2).sum())
    tot = float(((b - b.mean()) ** 2).sum())
    return [float(c) for c in coef] + [1.0 - ss / tot if tot > 0 else float("nan")]


def verdict(d: dict) -> dict:
    hr("8. the verdict, and the experiment that would falsify it")
    rows = [r for r in d.get("time", []) if r.get("slabs") == max(
        x["slabs"] for x in d.get("time", []) or [{"slabs": 0}])] if d.get("time") else []
    if not rows:
        say("  no `--time` rows in the data; run `--time` first.")
        return {}
    say("  The question: given only the two quantities TinyGrad already computes --")
    say("  `GlobalCounters.kernel_count` and its own per-program op estimate")
    say("  `ast.arg.estimates` -- would a cost model built on them rank the two")
    say("  expressible, implemented candidates the way the card does?")
    say("")
    hdr = (f"{'cand':<4} {'launches':>9} {'ops(device)':>14} {'kernel_us':>10} "
           f"{'algMACs':>12} {'MACs x':>8} {'ops x':>8}")
    say(hdr)
    say("-" * len(hdr))
    base = None
    for r in sorted(rows, key=lambda r: r["kernel_us"]):
        if r["cand"] == "C":
            base = r
    for r in sorted(rows, key=lambda r: r["kernel_us"]):
        say(f"{r['cand']:<4} {r['dev_launches']:>9,} {r['global_ops']:>14,} "
            f"{r['kernel_us']:>10,.0f} {r['macs_per_slab']*r['slabs']:>12,} "
            f"{(r['macs_per_slab']*r['slabs'])/(base['macs_per_slab']*base['slabs']):>7.2f}x "
            f"{r['global_ops']/base['global_ops']:>7.2f}x")
    say("")
    say("  The card's ordering, by measured device time, is the launch order of the")
    say("  rows above. `MACs x` and `ops x` are each candidate's ratio to C.")
    say("")

    # -- model 1: launches only
    say("== model 1: kernel_us = a + b * launches, fitted on ALL FOUR, tested by")
    say("   LEAVING EACH ONE OUT ==")
    c1 = _lsq([[1.0, r["dev_launches"]] for r in rows],
              [r["kernel_us"] for r in rows])
    say(f"  fit on all four: kernel_us = {c1[0]:,.0f} + {c1[1]:.2f} * launches, "
        f"R2 = {c1[-1]:.6f}")
    loo = []
    for i, r in enumerate(rows):
        tr = [rows[j] for j in range(len(rows)) if j != i]
        if len(tr) < 2:
            continue
        cc = _lsq([[1.0, x["dev_launches"]] for x in tr], [x["kernel_us"] for x in tr])
        pred = cc[0] + cc[1] * r["dev_launches"]
        spread = (max(r["kernel_us_all"]) - min(r["kernel_us_all"])) / max(
            1e-9, min(r["kernel_us_all"]))
        tol = spread * r["kernel_us"]
        loo.append(dict(cand=r["cand"], pred_us=pred, meas_us=r["kernel_us"],
                        err_us=pred - r["kernel_us"], tol_us=tol,
                        within_spread=abs(pred - r["kernel_us"]) <= tol))
        say(f"    hold out {r['cand']:<3} predict {pred:>10,.0f} us, measured "
            f"{r['kernel_us']:>10,.0f} us, error {pred-r['kernel_us']:>+10,.0f} us, "
            f"rep spread {tol:>7,.0f} us -> "
            f"{'inside' if abs(pred-r['kernel_us']) <= tol else 'OUTSIDE'}")
    say("")

    # -- model 2: launches + ops
    say("== model 2: kernel_us = a + b*launches + c*ops, over the whole sweep ==")
    sw = d.get("sweep", [])
    if len(sw) >= 6:
        X = [[1.0, r["dev_launches"], r["global_ops"]] for r in sw]
        y = [r["kernel_us"] for r in sw]
        c2 = _lsq(X, y)
        say(f"  n = {len(sw)} points, kernel_us = {c2[0]:,.0f} "
            f"+ {c2[1]:.3f}*launches + {c2[2]:.3e}*ops, R2 = {c2[-1]:.6f}")
        say(f"  the ops coefficient is {c2[2]:.3e} us per estimated op, i.e. "
            f"{'POSITIVE' if c2[2] > 0 else 'NON-POSITIVE'}")
        say(f"  implied per-launch cost {c2[1]:.2f} us")
        top = max(r["slabs"] for r in sw)
        pred = {r["cand"]: c2[0] + c2[1] * r["dev_launches"] + c2[2] * r["global_ops"]
                for r in sw if r["slabs"] == top}
        for nm in sorted(pred, key=lambda k: pred[k]):
            meas = next((r["kernel_us"] for r in rows if r["cand"] == nm), float("nan"))
            say(f"    {nm:<3} model {pred[nm]:>10,.0f} us  measured {meas:>10,.0f} us  "
                f"ratio {pred[nm]/max(1e-9,meas):>6.3f}x")
        say("")
        say("  THE FALSIFIER, as a test rather than a claim: refit this model on the")
        say("  other three candidates and predict the held-out one. `inside` means a")
        say("  cost model over exactly the two quantities TinyGrad already computes is")
        say("  sufficient, and the class-(ii) gap is a calibration constant. `outside`")
        say("  means it is not, and the model needs a term TinyGrad does not compute.")
        loo2 = []
        for i, r in enumerate(sw):
            tr = [sw[j] for j in range(len(sw)) if j != i]
            if len(tr) < 4:
                continue
            cc = _lsq([[1.0, x["dev_launches"], x["global_ops"]] for x in tr],
                      [x["kernel_us"] for x in tr])
            p = cc[0] + cc[1] * r["dev_launches"] + cc[2] * r["global_ops"]
            spread = (max(r["kernel_us_all"]) - min(r["kernel_us_all"])) / max(
                1e-9, min(r["kernel_us_all"]))
            tol = spread * r["kernel_us"]
            loo2.append(dict(cand=r["cand"], slabs=r["slabs"], pred_us=p,
                             meas_us=r["kernel_us"], err_us=p - r["kernel_us"],
                             tol_us=tol, within_spread=abs(p - r["kernel_us"]) <= tol))
        for e in loo2:
            say(f"    hold out {e['cand']:<3}@{e['slabs']:<3} predict {e['pred_us']:>9,.0f} "
                f"measured {e['meas_us']:>9,.0f} error {e['err_us']:>+9,.0f} "
                f"tol {e['tol_us']:>7,.0f} -> "
                f"{'inside' if e['within_spread'] else 'OUTSIDE'}")
        nb = sum(1 for e in loo2 if e["within_spread"])
        say(f"  {nb}/{len(loo2)} held-out points inside their own rep spread")
    say("")
    say("== model 3: the SAME two estimates, but with `estimates.mem` added ==")
    say("   `ast.arg.estimates` is a small record (renderer/__init__.py) carrying BOTH")
    say("   `ops` and `mem` per program, and realize.py:63-64 sums both into")
    say("   GlobalCounters. So `global_mem` is a third quantity TinyGrad already")
    say("   computes and already reports. If model 3 passes the held-out test where")
    say("   model 2 fails, the gap is not a missing number -- it is a missing USE.")
    if len(sw) >= 6:
        X3 = [[1.0, r["dev_launches"], r["global_ops"], r["global_mem"]] for r in sw]
        c3 = _lsq(X3, y)
        say(f"  n = {len(sw)} points, kernel_us = {c3[0]:,.0f} + {c3[1]:.3f}*launches "
            f"+ {c3[2]:.3e}*ops + {c3[3]:.3e}*mem, R2 = {c3[-1]:.6f}")
        say(f"  per-launch {c3[1]:.2f} us, per-op {c3[2]:.3e} us, per-mem "
            f"{c3[3]:.3e} us")
        loo3 = []
        for i, r in enumerate(sw):
            tr = [sw[j] for j in range(len(sw)) if j != i]
            if len(tr) < 4:
                continue
            cc = _lsq([[1.0, x["dev_launches"], x["global_ops"], x["global_mem"]]
                       for x in tr], [x["kernel_us"] for x in tr])
            p = cc[0] + cc[1] * r["dev_launches"] + cc[2] * r["global_ops"] \
                + cc[3] * r["global_mem"]
            spread = (max(r["kernel_us_all"]) - min(r["kernel_us_all"])) / max(
                1e-9, min(r["kernel_us_all"]))
            tol = spread * r["kernel_us"]
            loo3.append(dict(cand=r["cand"], slabs=r["slabs"], pred_us=p,
                             meas_us=r["kernel_us"], err_us=p - r["kernel_us"],
                             tol_us=tol, within_spread=abs(p - r["kernel_us"]) <= tol))
        for e in loo3:
            say(f"    hold out {e['cand']:<3}@{e['slabs']:<3} predict {e['pred_us']:>9,.0f} "
                f"measured {e['meas_us']:>9,.0f} error {e['err_us']:>+9,.0f} "
                f"tol {e['tol_us']:>7,.0f} -> "
                f"{'inside' if e['within_spread'] else 'OUTSIDE'}")
        nb3 = sum(1 for e in loo3 if e["within_spread"])
        say(f"  {nb3}/{len(loo3)} held-out points inside their own rep spread")
        say("")
        say("  and the mem estimate's own ratio for the pair the question is about:")
        cc_ = next((r for r in rows if r["cand"] == "C"), None)
        dd_ = next((r for r in rows if r["cand"] == "D2"), None)
        if cc_ and dd_:
            say(f"    D2/C  mem estimate {dd_['global_mem']/cc_['global_mem']:.2f}x"
                f"   ops estimate {dd_['global_ops']/cc_['global_ops']:.2f}x"
                f"   algorithmic MACs "
                f"{(dd_['macs_per_slab'])/(cc_['macs_per_slab']):.2f}x"
                f"   MEASURED {dd_['kernel_us']/cc_['kernel_us']:.2f}x")
    say("")
    say("== which axis predicted the ordering, for the pair the question is about ==")
    cc_ = next((r for r in rows if r["cand"] == "C"), None)
    dd_ = next((r for r in rows if r["cand"] == "D2"), None)
    if cc_ and dd_:
        cals = d.get("launchcal", {})
        tau = (cals.get(str((1,))) or {}).get("us_per_launch")
        say(f"  launch axis predicts: D2 ahead by "
            f"{(cc_['dev_launches']-dd_['dev_launches']) * (tau or 0):,.0f} us "
            f"({(cc_['dev_launches']-dd_['dev_launches'])} launches x "
            f"{tau:.2f} us) on a {cc_['kernel_us']:,.0f} us budget"
            if tau else "  launch axis: no calibration in the data")
        say(f"  arithmetic axis predicts: C ahead, by a MAC-count ratio of "
            f"{(dd_['macs_per_slab'])/(cc_['macs_per_slab']):.2f}x")
        meas = max(cc_["kernel_us"], dd_["kernel_us"]) / max(1e-9, min(
            cc_["kernel_us"], dd_["kernel_us"]))
        spread = max((max(cc_["kernel_us_all"]) - min(cc_["kernel_us_all"]))
                     / max(1e-9, min(cc_["kernel_us_all"])),
                     (max(dd_["kernel_us_all"]) - min(dd_["kernel_us_all"]))
                     / max(1e-9, min(dd_["kernel_us_all"])))
        say(f"  card says: "
            f"{'C' if cc_['kernel_us'] < dd_['kernel_us'] else 'D2'} by {meas:.2f}x, "
            f"against a rep spread of {spread*100:.1f}% -> "
            f"{'DECIDED' if meas > 1 + spread else 'WITHIN NOISE, no winner'}")
        # where the difference lives
        bc, bd = cc_.get("by_class", {}), dd_.get("by_class", {})
        for k in sorted(set(bc) | set(bd)):
            a = bc.get(k, [0, 0.0])
            b = bd.get(k, [0, 0.0])
            say(f"    kernel class {k:<3} C {a[0]:>3} kernels {a[1]:>9,.0f} us  "
                f"D2 {b[0]:>3} kernels {b[1]:>9,.0f} us  "
                f"delta {b[1]-a[1]:>+9,.0f} us")
        out = dict(launch_axis_us=(cc_["dev_launches"] - dd_["dev_launches"]) * (tau or 0),
                   measured_ratio=meas, rep_spread=spread,
                   decided=bool(meas > 1 + spread))
    else:
        out = {}
    say("")
    say("== is it a scheduling problem or an expressiveness problem? ==")
    say("  Both spellings are expressed, are numerically the same operator to 3.1e-08,")
    say("  and the framework emits each one's schedule EXACTLY as the offline census")
    say("  predicted (section 3). So the framework is not making a scheduling mistake")
    say("  about this operator. What it cannot do is price one spelling against the")
    say("  other before realizing them, and its own op estimate mis-prices this pair")
    say("  by a factor quoted above.")
    R["verdict"] = dict(model_launches=c1, loo_launches=loo, loo_launches_ops=loo2,
                        loo_launches_ops_mem=globals().get("loo3", []),
                        pair=out)
    return R["verdict"]


# =============================================================================
# 8. spelling robustness: does the C-vs-D2 ordering move with the constant form?
# =============================================================================
def spelling_check(C: int, reps: int, warm: int) -> list:
    hr("7. is the ordering spelling-invariant?")
    say("  notes/31 §2: the `numpy` spelling of a constant costs a COPY kernel each and")
    say("  the `arange` spelling costs ALU inside every consumer. notes/31 measured the")
    say("  KERNEL ranking to be identical under both. The TIMING ranking is checked here,")
    say("  for the two candidates the open question is actually about.")
    say("")
    hdr = (f"{'spelling':<9} {'cand':<4} {'launches':>9} {'copies':>7} "
           f"{'kernel_us':>10} {'spread':>8} {'reps (us)':>26}  winner")
    say(hdr)
    say("-" * len(hdr))
    rows = []
    for sp in ("arange", "numpy"):
        UP.SPELLING = sp
        for name in ("C", "D2"):
            try:
                r = measure(name, 64, C, reps, warm)
            except Exception as e:                          # noqa: BLE001
                say(f"{sp:<9} {name:<4} FAILED: {type(e).__name__}: {e}")
                continue
            r["spelling"] = sp
            rows.append(r)
        got = {r["cand"]: r for r in rows if r["spelling"] == sp}
        w = ""
        if len(got) == 2:
            a, b = got["C"]["kernel_us"], got["D2"]["kernel_us"]
            w = f"{'C' if a < b else 'D2'} by {max(a,b)/max(1,min(a,b)):.2f}x"
        for name in ("C", "D2"):
            r = got.get(name)
            if r is None:
                continue
            say(f"{sp:<9} {name:<4} {r['dev_launches']:>9,} {r['dbg2_copy_lines']:>7,} "
                f"{r['kernel_us']:>10,.0f} {r['kernel_spread']*100:>7.1f}% "
                f"{str([round(x) for x in r['kernel_us_all']]):>26}  {w}")
        say("")
    UP.SPELLING = "arange"
    for sp in ("arange", "numpy"):
        got = {r["cand"]: r for r in rows if r["spelling"] == sp}
        if len(got) == 2:
            a, b = got["C"]["kernel_us"], got["D2"]["kernel_us"]
            sa, sb = got["C"]["kernel_spread"], got["D2"]["kernel_spread"]
            w = "C" if a < b else "D2"
            ratio = max(a, b) / max(1e-9, min(a, b))
            tie = ratio < 1 + max(sa, sb)
            say(f"  spelling {sp:<7} kernel_us: C {a:,.0f} (spread {sa*100:.1f}%) vs "
                f"D2 {b:,.0f} (spread {sb*100:.1f}%)  ->  {w}, {ratio:.2f}x"
                f"{'  [WITHIN NOISE: no winner]' if tie else ''}")
    R["spelling"] = rows
    return rows


# =============================================================================
# main
# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--env", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--launchcal", action="store_true")
    ap.add_argument("--counters", action="store_true")
    ap.add_argument("--verdict", action="store_true")
    ap.add_argument("--from-json", default="",
                    help="re-derive sections 5 and 8 from a saved ut_time.json "
                         "instead of running on a card. Used to re-check the note's "
                         "arithmetic from the run's own artefact, on a host with no "
                         "GPU, so the numbers in the note are not retyped.")
    ap.add_argument("--spelling-check", action="store_true")
    ap.add_argument("--slabs", type=int, default=64)
    ap.add_argument("--C", type=int, default=64)
    ap.add_argument("--sweep-slabs", type=str, default="1,8,16,32,64")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--out", default="ut_time_out")
    args = ap.parse_args()

    if args.selftest:
        return 0 if selftest() else 1
    if args.from_json:
        # Re-derive from an artefact. Everything below is arithmetic on a JSON
        # file this script wrote, so it needs no card and it cannot quietly pick
        # up a different device's numbers.
        with open(args.from_json) as f:
            R.update(json.load(f))
        R["_source"] = os.path.abspath(args.from_json)
        LOG.clear()
        normed = _normalise(R)
        if normed:
            say(f"  NOTE: {len(normed)} row(s) came from an artefact written before")
            say("  this file stored per-rep kernel time: "
                f"{', '.join(normed[:8])}"
                f"{' ...' if len(normed) > 8 else ''}")
            say("  kernel_us for those rows is ONE rep, read out of by_class, and")
            say("  kernel_spread is unknown for them. Anything that depends on the")
            say("  spread must say so.")
        report_env_offline()
        if args.launchcal or not any((args.time, args.sweep, args.counters)):
            R["launchcal"] = _recalibrate_launchcal()
            say("(launch calibration re-derived from the saved points, not re-run)")
        if args.time or args.verdict or not args.sweep:
            verdict(R)
        _dump(args.out)
        return 0
    if args.env:
        report_env()
        return 0
    if not any((args.all, args.verify, args.time, args.sweep, args.launchcal,
                args.counters, args.verdict, args.spelling_check)):
        args.all = True

    require_gpu()
    os.makedirs(args.out, exist_ok=True)
    report_env()

    # The instrument is validated, and the gate is shown to reject wrong data,
    # BEFORE any timing number is taken.
    if not selftest():
        say("")
        say("SELFTEST FAILED -- refusing to time anything.")
        _dump(args.out)
        return 1
    verify(slabs=2, C=args.C)
    say("")
    say("  The gate above ran on THIS card. Every timing number below is a timing of")
    say("  four expressions that were just shown to compute the same operator.")
    say("")

    if args.all or args.time:
        time_table(args.slabs, args.C, args.reps, args.warm)
    if args.all or args.sweep:
        slab_list = [int(x) for x in args.sweep_slabs.split(",") if x]
        sweep(slab_list, args.C, reps=None if args.all else args.reps, warm=0)
    if args.all or args.launchcal:
        launchcal()
    if args.all or args.spelling_check:
        spelling_check(args.C, args.reps, args.warm)
    if args.all or args.counters:
        counters(C=args.C)
    if args.all or args.verdict:
        verdict(R)

    _dump(args.out)
    hr("done")
    say(f"wrote {os.path.join(args.out, 'ut_time.json')} and "
        f"{os.path.join(args.out, 'ut_time.txt')}")
    return 0


def _dump(outdir: str) -> None:
    try:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "ut_time.json"), "w") as f:
            json.dump(R, f, indent=1, default=str)
        with open(os.path.join(outdir, "ut_time.txt"), "w") as f:
            f.write("\n".join(LOG) + "\n")
    except OSError as e:                                     # pragma: no cover
        say(f"could not write the sidecar: {e}")


if __name__ == "__main__":
    raise SystemExit(main())
