#!/usr/bin/env python3
"""splitcausal_t4probe.py -- the on-card probe for notes/37.

WHAT THIS MEASURES, AND WHY IT IS ONE PROBE
===========================================
Kaggle kills a cell at 1800 s and the runner captures stdout only at the END, so
a killed probe loses everything. This is therefore ONE probe with a hard internal
budget, ordered so that the most valuable question is answered first, and every
stage prints as it completes. If it dies at 900 s, the stages already printed are
still in the log.

ORDERING, and the reason for it
-------------------------------
`axis/splitcausal.py` settled the structural question on the host with no quota:
the switch is at `2*SPLIT+1` and it MOVES with the constant. What the host cannot
settle is what the card does, and the card's answer is the one that decides
whether `notes/33`'s 25.96x is the two-stage decomposition's cost.

The stages, in the order they run:

  S0  provenance. Card, driver, clock, and the sha256 of the tinygrad actually
      imported, plus the source text of the `SPLIT` literal. If this does not
      print `SPLIT = 256`, nothing below it is about the framework the report
      describes.
  S1  the CHEAP CAUSAL HANDLE: one-stage vs two-stage, at one fixed shape, with
      NO constant moved and NO file edited. `_cumalu` and `Tensor.cumsum` both
      compute cumsum; above the switch they are different decompositions of the
      same answer. If the two-stage form is dramatically SLOWER, the
      decomposition is the cost and the rest is bookkeeping. This is first
      because it needs no patch, so it is valid even if every later stage is
      compromised.
  S2  the baseline density sweep, unmodified, at the ladder's own element count.
      This is the `notes/33` sweep with the hole in it filled.
  S3  the SPLIT sweep: the same density sweep at SPLIT=64 and SPLIT=1024.

WHY THE HOST'S ANSWER IS NOT ENOUGH, stated as the prediction
=============================================================
From the source, at fixed total elements E:

    N <= 2*SPLIT  ->  one stage  ->  N     adds per output  ->  work = E*N
    N >  2*SPLIT  ->  two stages ->  SPLIT adds per output  ->  work = E*SPLIT

so the cost curve RISES to `N = 2*SPLIT` and then DROPS by 2x onto a plateau of
`E*SPLIT`. Moving `SPLIT` therefore moves TWO things, and both are checkable:

  (a) the location of the peak, to `2*SPLIT`;
  (b) the HEIGHT of the plateau, by the factor `SPLIT` itself.

(b) is the stronger and cheaper test, and it is the one this probe is built
around. At the ladder's own shape, `N = 4096`, `E = 2^24`:

    SPLIT=64    plateau work = E*64    -> ~4x FASTER than the installed build
    SPLIT=256   plateau work = E*256   -> the number notes/33 measured
    SPLIT=1024  plateau work = E*1024  -> ~4x SLOWER than the installed build

A 4x swing in the framework's own device time at one shape, from one integer in
its own source, is not something a shape-dependent memory effect or a launch
overhead can imitate. **If the time at N=4096 does not move when SPLIT moves, the
literal is not the mechanism, and that is the result this probe exists to be able
to return.**

THE FALSIFIERS, restated as pass/fail on this probe
===================================================
  F1  N=4096 device time does not track SPLIT              => literal is not it
  F2  the peak is not at 2*SPLIT                            => switch is elsewhere
  F3  the cost is the same at every SPLIT                    => something else dominates
  F4  one-stage is not faster than two-stage at fixed shape  => decomposition is not it
  F5  the time RISES across the switch                        => the model's direction is wrong

INSTRUMENT
==========
Device time is `linear_device_time_ms`'s definition, re-derived here and not
imported, because a notebook that pins its own sources by hash should not be
quietly depending on a file another agent is editing. It walks the compiled
LINEAR and re-executes each CALL with tinygrad's own `time_call`, so it measures
RE-EXECUTION and is immune to the `realize()`-on-an-already-realized-graph trap
that has produced three fabricated numbers in this project. `require_fresh_graph`
is called before every measurement and the graph is rebuilt every time.

Every arm is also checked against `np.cumsum` in float64 before it is timed. A
ratio between two arms that do not both compute cumsum is not a measurement.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# --------------------------------------------------------------------------
# THE PROBE ITSELF.  A string, because it has to travel inside one notebook
# cell and be hash-checked there like every other payload.
# --------------------------------------------------------------------------

PROBE = r'''
# ---------------------------------------------------------------- tinygrad FIRST
#
# THE ORDER OF THESE LINES IS THE WHOLE REASON THIS SECTION EXISTS. The first
# version of this notebook imported `tinygrad` inside S0, and the kernel died in
# 9.5 s with:
#
#     ModuleNotFoundError: No module named 'tinygrad'
#
# The Kaggle image ships `papermill` and `nbconvert` but NOT tinygrad.
# `axis/kaggle/make_notebook.py` installs it in its own cell, and this probe has
# no such cell, so the import has to be done here. It is at the very top, before
# anything else, so a missing tinygrad fails in 30 seconds having printed a
# sentence saying what is missing, rather than at the end of a run.
import subprocess, sys
if "tinygrad" not in sys.modules:
    try:
        import tinygrad  # noqa: F401
    except ImportError:
        print("tinygrad is not in this image; installing it", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tinygrad"])

import hashlib, json, os, re, statistics, time
import numpy as np

DEV = os.environ.get("DEV", "CUDA")
BUDGET_S = float(os.environ.get("PROBE_BUDGET_S", "1000"))
T0 = time.time()

def log(*a):
    print(*a, flush=True)

def left():
    return BUDGET_S - (time.time() - T0)

def relerr(a, want):
    a = np.asarray(a, np.float64)
    return float(np.abs(a - want).max() / max(1e-30, np.abs(want).max()))

# ---------------------------------------------------------------- S0 provenance
log("=" * 78)
log("S0  PROVENANCE")
log("=" * 78)
import tinygrad
from tinygrad import Device, Tensor, GlobalCounters
import tinygrad.mixin.op as M
import tinygrad.tensor as TWMOD

OP_PY = __import__("pathlib").Path(M.__file__)
SRC = OP_PY.read_text()
_h = hashlib.sha256(OP_PY.read_bytes()).hexdigest()
_m = re.search(r"^(\s*)SPLIT = (\d+)\s*$", SRC, re.M)
assert _m, "SPLIT literal not found in the installed file"
INSTALLED = int(_m.group(2))
ORIG_SRC = __import__("inspect").getsource(M.OpMixin._split_cumalu)
SPLIT_RE = re.compile(r"^(\s*)SPLIT = (\d+)\s*$", re.M)

log("  tinygrad      ", getattr(tinygrad, "__version__", "?"), "|", tinygrad.__file__)
log("  op.py         ", OP_PY)
log("  op.py sha256  ", _h)
log("  installed SPLIT", INSTALLED, "-> source predicts the first two-stage N at", 2*INSTALLED+1)
log("  Tensor wraps its mixin methods:",
    Tensor.__dict__.get("_split_cumalu") is not M.OpMixin.__dict__["_split_cumalu"])
for a in ("name", "capability", "driver_version", "arch", "clock_hz"):
    try: log(f"  {a:<14}", getattr(Device.DEFAULT, a))
    except Exception: pass
try:
    from tinygrad.runtime import ops_cuda
    log("  device prop   ", ops_cuda.cuda_props())
except Exception as e:
    log("  device prop    unavailable:", type(e).__name__, e)
try:
    import subprocess as _sp
    log("  nvcc           ", _sp.run(["nvcc", "--version"], capture_output=True,
                                    text=True).stdout.strip().splitlines()[-1])
except Exception as e:
    log("  nvcc            unavailable:", type(e).__name__, e)

# ------------------------------------------------------------- instrument
def require_fresh(t, what):
    u = getattr(t, "uop", None)
    if u is None: return
    try:
        fresh = (not u.is_virtual) and (not u.has_buffer_identity())
    except Exception:
        return
    if not fresh:
        raise RuntimeError(f"{what}: tensor is ALREADY REALIZED; realizing it again "
                           f"is a silent no-op and timing it measures nothing. "
                           f"Rebuild the graph (or use time_call on the LINEAR).")

def linear_device_time_ms(tensor, iters=3):
    """Re-derive of minecu_tg/bench.py:linear_device_time_ms.

    Walks the compiled LINEAR and re-executes each CALL with tinygrad's own
    time_call, then SUMS the per-kernel device times. Immune to the
    already-realized no-op, because it never calls realize().
    """
    from tinygrad.engine.realize import time_call
    lin, vv = Tensor.linear_with_vars(tensor)
    calls = list(lin.src)
    if not calls: return None
    for _ in range(2):
        for c in calls: next(time_call(c, var_vals=vv))
    tot = []
    for _ in range(iters):
        s = 0.0
        for c in calls: s += next(time_call(c, var_vals=vv))
        tot.append(s * 1e3)
    tot.sort()
    return {"n_kernels": len(calls), "median_ms": statistics.median(tot),
            "min_ms": tot[0], "max_ms": tot[-1],
            "per_kernel_us": [round(next(time_call(c, var_vals=vv))*1e6, 2) for c in calls]}

# ------------------------------------------------------------- the patch
import functools, textwrap, difflib
_ORIG_TENSOR = Tensor._split_cumalu

def patch_split(k):
    import importlib
    tinygrad.codegen.to_program_cache.clear()
    new_src, n = SPLIT_RE.subn(lambda m: f"{m.group(1)}SPLIT = {k}", ORIG_SRC, count=1)
    if n != 1 or f"SPLIT = {k}" not in new_src:
        raise RuntimeError(f"the SPLIT substitution did not take exactly once (n={n})")
    body = textwrap.dedent(new_src)
    ns = dict(vars(M))
    exec(compile(body, "<probe:patched _split_cumalu>", "exec"), ns)
    fn = ns["_split_cumalu"]
    Tensor._split_cumalu = functools.wraps(fn)(TWMOD._metadata_wrapper(fn))
    d = list(difflib.unified_diff(ORIG_SRC.splitlines(), new_src.splitlines(),
                                  fromfile=f"installed SPLIT={INSTALLED}",
                                  tofile=f"patched SPLIT={k}", lineterm="", n=1))
    return {"split": k, "substitutions": n, "diff": d}

def restore_split():
    Tensor._split_cumalu = _ORIG_TENSOR
    import tinygrad.codegen
    tinygrad.codegen.to_program_cache.clear()

def clear_caches():
    import tinygrad.codegen as CG
    from tinygrad.engine import realize as RZ
    CG.to_program_cache.clear()
    for nm in ("runtime_cache", "graph_cache", "local_size_cache"):
        if hasattr(RZ, nm): getattr(RZ, nm).clear()

# ------------------------------------------------------------- oracle
def make_input(r, n, seed=12):
    rs = np.random.RandomState(seed)
    return rs.rand(r, n).astype(np.float32)

def arm_cumsum(x_np):
    t = Tensor(x_np).cumsum(-1)
    require_fresh(t, "cumsum arm")
    return t

def arm_one_stage(x_np):
    import tinygrad.mixin.op as _M
    t = Tensor(x_np)._cumalu(-1, _M.Ops.ADD)
    require_fresh(t, "one-stage arm")
    return t

def time_arm(build, x_np, want, label, iters=3):
    """Build fresh, GATE against the float64 oracle, then time by re-execution."""
    g = build(x_np)
    d = linear_device_time_ms(g, iters)
    e = relerr(g.numpy(), want)
    if e >= 1e-3:
        raise RuntimeError(f"{label}: GATE FAILED, relerr={e:.3e} at {x_np.shape}. "
                           f"Refusing to time an arm that does not compute cumsum.")
    return {"label": label, "relerr": e, **d}

# ============================================================ S1  one vs two
log("")
log("=" * 78)
log("S1  THE CHEAP CAUSAL HANDLE: one-stage vs two-stage at ONE fixed shape")
log("    no constant moved, no file edited")
log("=" * 78)
N1 = 4096
E = 1 << 24
R1 = E // N1
assert R1 * N1 == E, f"element count is not 2^24: R={R1} N={N1}"
log(f"  shape R={R1} N={N1}  E={R1*N1}  (the ladder's own shape and element count)")
x1 = make_input(R1, N1)
want1 = np.cumsum(x1.astype(np.float64), -1)
s1 = {}
try:
    t_cum = time_arm(arm_cumsum, x1, want1, f"cumsum (two-stage, N={N1}>{2*INSTALLED})")
    t_one = time_arm(arm_one_stage, x1, want1, f"_cumalu (one-stage, N={N1})")
    s1 = {"R": R1, "N": N1, "elements": R1*N1, "cumsum": t_cum, "one_stage": t_one}
    log(f"  cumsum  (2-stage) {t_cum['median_ms']:8.3f} ms  {t_cum['n_kernels']} kernels"
        f"  per-kernel {t_cum['per_kernel_us']}  relerr {t_cum['relerr']:.2e}")
    log(f"  _cumalu (1-stage) {t_one['median_ms']:8.3f} ms  {t_one['n_kernels']} kernels"
        f"  per-kernel {t_one['per_kernel_us']}  relerr {t_one['relerr']:.2e}")
    s1["ratio_two_over_one"] = t_cum["median_ms"] / t_one["median_ms"]
    log(f"  RATIO two-stage / one-stage = {s1['ratio_two_over_one']:.3f}x")
    log("  F4: one-stage not faster => the DECOMPOSITION is not the cost."
        if s1["ratio_two_over_one"] < 1.0 else
        "  F4 FIRED: two-stage is the slower arm, decomposition could be the cost.")
except Exception as e:
    s1 = {"error": f"{type(e).__name__}: {e}"}
    log("  S1 FAILED:", s1["error"])
log(f"  [{left():.0f}s of budget left]")

# ================================================== S2 / S3  the density sweep
DENSE = [256, 320, 384, 448, 480, 512, 513, 544, 640, 768, 896, 1024,
         1536, 2048, 4096]

# WHY THIS GRID, HONESTLY. It is DENSE where the mechanism is -- every point
# from 256 to 1024, which is the interval `notes/33` had no point in at all, and
# the interval the framework's own switch at N=512 sits in the middle of. It is
# SPARSE below 256 and at the far end, because those points are not where the
# question is and each one costs device time.
#
# It is sparse below 256 for a reason worth stating: the reference
# `axis/refkernels.cu` cannot be launched there at all except through the
# scalar kernel, so a RATIO against the reference is not available below 256 and
# the mechanism claim is a claim about the framework's own cost. The reference
# question is a different experiment and is not this one.

def sweep(split_value, label, iters=3, floor_s=90):
    if split_value is not None:
        pa = patch_split(split_value)
        log(f"  --- patch: {split_value} substitutions, installed on Tensor")
        for l in pa["diff"]: log("      " + l)
    else:
        restore_split()
    out = []
    try:
        log(f"  {'N':>6} {'R':>7} {'path':>10} {'kernels':>8} {'dev ms':>9} {'xE*SPLIT':>9} {'relerr':>9}")
        for n in DENSE:
            if left() < floor_s:
                log(f"  N={n} SKIPPED, out of budget ({left():.0f}s left)")
                continue
            r = max(1, E // n)
            clear_caches()
            x = make_input(r, n)
            want = np.cumsum(x.astype(np.float64), -1)
            t0 = time.time()
            try:
                t = time_arm(arm_cumsum, x, want, f"cumsum N={n}", iters)
            except Exception as e:
                log(f"  {n:>6} {r:>7}  FAILED {type(e).__name__}: {str(e)[:90]}")
                out.append({"N": n, "R": r, "error": f"{type(e).__name__}: {str(e)[:200]}"})
                continue
            sp = split_value if split_value is not None else INSTALLED
            path = "one-stage" if n <= 2 * sp else "two-stage"
            unit = E * sp / 1e6
            out.append({"N": n, "R": r, "path": path, "kernels": t["n_kernels"],
                        "median_ms": t["median_ms"], "min_ms": t["min_ms"],
                        "x_predicted": round(t["median_ms"] / unit, 4),
                        "relerr": t["relerr"], "per_kernel_us": t["per_kernel_us"],
                        "wall_s": round(time.time() - t0, 1)})
            log(f"  {n:>6} {r:>7} {path:>10} {t['n_kernels']:>8} "
                f"{t['median_ms']:>9.3f} {t['median_ms']/unit:>9.3f} {t['relerr']:>9.1e}"
                f"   [{time.time()-t0:.1f}s]")
    finally:
        if split_value is not None:
            restore_split()
    return {"label": label, "split": split_value if split_value is not None else INSTALLED,
            "patch_diff": (pa["diff"] if split_value is not None else None),
            "points": out}

log("")
log("=" * 78)
log("S2  BASELINE density sweep, SPLIT UNMODIFIED (256)")
log("    E = 2^24 held constant; notes/33's grid had NO point between 256 and 1024")
log("=" * 78)
s2 = sweep(None, "baseline")
log(f"  [{left():.0f}s of budget left]")

log("")
log("=" * 78)
log("S3  THE SPLIT SWEEP. The plateau must scale with SPLIT; the peak must sit at 2*SPLIT.")
log("=" * 78)
sweeps = {"baseline": s2}
# THE SPLIT SWEEP IS THE EXPERIMENT, so it gets the budget and the baseline
# sweep is what yields. The baseline is also re-measured as part of the verdict
# if time allows, because an unmodified control in the SAME session is what makes
# a difference a fact rather than a session difference.
SPLIT_VALUES = [64, 1024]
for k in SPLIT_VALUES:
    if left() < 200:
        log(f"  SPLIT={k} SKIPPED, out of budget ({left():.0f}s left)")
        continue
    log("")
    log(f"  ---- SPLIT = {k} ----")
    sweeps[f"split{k}"] = sweep(k, f"split{k}")
    log(f"  [{left():.0f}s of budget left]")

# The control that makes F1 a fact rather than a session difference: the
# UNMODIFIED build re-measured AFTER the patched runs, in the same process, on the
# same card. If the modified N=4096 numbers differ from this one for any reason
# other than SPLIT, F1 is void and the log says so.
if left() > 120:
    log("")
    log("  ---- control: the UNMODIFIED build, re-measured last, same process ----")
    ct = sweep(None, "control-last", iters=3, floor_s=60)
    ct["points"] = [p for p in ct["points"] if p["N"] in (512, 513, 1024, 4096)]
    sweeps["control_last"] = ct
    for p in ct["points"]:
        log(f"    control N={p['N']:<6} {p['median_ms']:8.3f} ms  {p['kernels']} kernels")

# ----------------------------------------------------------------- verdict
log("")
log("=" * 78)
log("VERDICT")
log("=" * 78)
def at(rows, n):
    for r in rows:
        if r.get("N") == n and "median_ms" in r: return r
    return None

base = sweeps["baseline"]["points"]
b4096 = at(base, 4096)
log("  the load-bearing number: framework DEVICE time at N=4096, E=2^24")
if b4096: log(f"    SPLIT=256  {b4096['median_ms']:8.3f} ms  {b4096['kernels']} kernels")
for k in SPLIT_VALUES:
    r = at(sweeps.get(f"split{k}", {}).get("points", []), 4096)
    if r: log(f"    SPLIT={k:<5} {r['median_ms']:8.3f} ms  {r['kernels']} kernels")
    elif f"split{k}" in sweeps: log(f"    SPLIT={k:<5} MISSING")
cl = at(sweeps.get("control_last", {}).get("points", []), 4096)
if cl:
    log(f"    control    {cl['median_ms']:8.3f} ms  (unmodified, re-measured LAST)")
    if b4096:
        d = abs(cl["median_ms"] - b4096["median_ms"]) / b4096["median_ms"]
        log(f"    baseline vs control: {d*100:.2f}% apart"
            f"{'  -- SESSION DRIFT, F1 VOID' if d > 0.05 else '  -- within 5%, control holds'}")

log("")
log("  F1  does the time at N=4096 TRACK SPLIT?")
got = {}
if b4096: got[INSTALLED] = b4096["median_ms"]
for k in SPLIT_VALUES:
    r = at(sweeps.get(f"split{k}", {}).get("points", []), 4096)
    if r: got[k] = r["median_ms"]
drift_ok = (cl is None or b4096 is None
            or abs(cl["median_ms"] - b4096["median_ms"]) / b4096["median_ms"] <= 0.05)
if len(got) == 3 and drift_ok:
    order = [got[64], got[256], got[1024]]
    monotonic = order[0] < order[1] < order[2]
    log(f"      SPLIT=64 {order[0]:.3f} < SPLIT=256 {order[1]:.3f} < SPLIT=1024 {order[2]:.3f}"
        f"  -> {monotonic}")
    log(f"      F1 {'PASSES: the time tracks the literal' if monotonic else 'FIRED: the literal does not move the time'}")
    if monotonic:
        log(f"      ratio 1024/64 = {order[2]/order[0]:.2f}x   (the model predicts 16x on work;"
            f" the realised ratio is what the card does with it)")
elif not drift_ok:
    log("      F1 VOID: the unmodified control moved more than 5% between the start")
    log("         and the end of the session, so a difference attributed to SPLIT")
    log("         could equally be a session difference. Reported as such.")
else:
    log("      F1 INDETERMINATE: not all three SPLIT values completed at N=4096")
log("")
log("  F2  is the peak at 2*SPLIT?")
for key, sw in sweeps.items():
    pts = [p for p in sw["points"] if "median_ms" in p]
    if not pts: continue
    sp = sw["split"]
    below = [p for p in pts if p["N"] <= 2 * sp and "median_ms" in p]
    above = [p for p in pts if p["N"] > 2 * sp and "median_ms" in p]
    pk = max(pts, key=lambda p: p["median_ms"])
    log(f"    SPLIT={sp:<5} slowest N={pk['N']} ({pk['median_ms']:.3f} ms);"
        f" 2*SPLIT={2*sp}; max below={max((p['N'] for p in below), default=None)};"
        f" min above={min((p['N'] for p in above), default=None)}")
log("")
log("  F5  does the cost RISE into the switch and DROP after it? (SPLIT=256)")
if base:
    b = {p["N"]: p["median_ms"] for p in base if "median_ms" in p}
    for pair in ((256, 512), (384, 640), (512, 640), (512, 1024)):
        if pair[0] in b and pair[1] in b:
            log(f"    N={pair[0]:<5} {b[pair[0]]:8.3f} ms   ->  N={pair[1]:<5} {b[pair[1]]:8.3f} ms"
                f"   ratio {b[pair[1]]/b[pair[0]]:.3f}")

log("")
log("=" * 78)
log(f"probe finished, {left():.0f}s of budget unused")
log("=" * 78)

RESULT = {
    "probe": "axis/splitcausal_t4probe.py",
    "dev": DEV,
    "tinygrad_file": tinygrad.__file__,
    "op_py": str(OP_PY), "op_py_sha256": _h,
    "installed_SPLIT": INSTALLED,
    "predicted_first_two_stage_N": 2 * INSTALLED + 1,
    "tensor_wraps_mixin": Tensor.__dict__.get("_split_cumalu") is not M.OpMixin.__dict__["_split_cumalu"],
    "E": E, "dense_N": DENSE,
    "one_vs_two": s1, "sweeps": sweeps,
    "n4096_by_split": got,
    "control_last_n4096_ms": cl["median_ms"] if cl else None,
    "session_drift_pct": (None if (cl is None or not b4096)
                          else round(100*abs(cl["median_ms"]-b4096["median_ms"])/b4096["median_ms"], 3)),
    "budget_s": BUDGET_S, "wall_s": round(time.time() - T0, 1),
    "patch_diff_s64": (sweeps.get("split64") or {}).get("patch_diff"),
}
open(os.environ.get("SPLITCAUSAL_OUT", "splitcausal_t4.json"), "w").write(json.dumps(RESULT, indent=2, default=str))
log("wrote", os.environ.get("SPLITCAUSAL_OUT", "splitcausal_t4.json"))
'''


def build_notebook(out_path: Path) -> dict:
    """Assemble a one-cell-payload notebook with every source sha256-checked.

    The payload travels with its own digest, so the check on the card is against
    a value derived from the same read of the file rather than a literal that
    could drift. This is `axis/kaggle/make_notebook.py`'s discipline, kept
    because a timing taken from a source that is not the one on disk is worse
    than no timing.
    """
    payload = PROBE
    digest = hashlib.sha256(payload.encode()).hexdigest()
    nb = {
        "cells": [
            {"cell_type": "code", "execution_count": None, "id": "splitcausal-probe",
             "metadata": {},
             "outputs": [], "source": [
                 "# generated by axis/splitcausal_t4probe.py -- do not edit by hand\n",
                 "import base64, hashlib, pathlib\n",
                 f"_PAYLOAD = {payload!r}\n",
                 f"_SHA = {digest!r}\n",
                 "assert hashlib.sha256(_PAYLOAD.encode()).hexdigest() == _SHA, 'payload changed in transit'\n",
                 "exec(compile(_PAYLOAD, '<splitcausal-t4-probe>', 'exec'))\n",
             ]},
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(nb, indent=1))
    return {"notebook": str(out_path), "payload_sha256": digest,
            "payload_bytes": len(payload.encode())}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "axis" / "splitcausal_t4" / "probe.ipynb"))
    a = ap.parse_args()
    info = build_notebook(Path(a.out))
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
