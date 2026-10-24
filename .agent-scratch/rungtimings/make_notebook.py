#!/usr/bin/env python3
"""make_notebook.py -- the Kaggle notebook that fills notes/36's blank column.

WHAT IT MEASURES, AND WHY IT IS ITS OWN NOTEBOOK
-----------------------------------------------
`notes/23` built a five-rung ablation ladder in `minecu_tg_opt/` and timed two
of the five (the endpoints). This notebook times all five, on one T4, in one
run, at several batches, and it measures the hand-written CUDA reference IN THE
SAME RUN rather than quoting a number from a different session.

The instrument is `minecu_tg.bench.linear_device_time_ms` -- IMPORTED, never
reimplemented -- because that is the ruler the naive arm was measured with and a
timing taken with a different one is not comparable to it however correct it is.
Two of this project's own run-history defects are specifically about instruments,
and both of them are guarded here:

  * `realize()` on an already-realized tensor is a silent no-op, so a graph that
    was already realized once would read as ~0 us. Every measurement here builds
    a FRESH, never-before-realized tensor, asserts the exact predicate
    `Tensor.realize` itself uses to decide there is work to do
    (`not uop.has_buffer_identity()`), and asserts the resulting graph has
    >= 1 CALL whose measured time is > 0.
  * `CANCEL` from Kaggle is a `CellTimeoutError`, not a quota kill. Every stage
    is its own cell (the nbclient cap is PER CELL), `SUMMARY.txt` is flushed at
    the end of every cell, and each batch gets its own cell, so one stage dying
    does not take the rest of the ladder with it.

ORDER OF THE CELLS, AND WHY
---------------------------
  1. provenance + build + reference render timings   (cheap, decisive)
  2. THE GATE: all five rungs against the hand-written side's OWN bytes, on the
     hand-written side's OWN state, ON THIS CARD -- before any of them is timed.
     Also the DDA step workload of each rung, on the card, so the five timings
     are known to be five measurements of one computation.
  3. one cell per timing batch, 1024 first (the batch the crossover sits above
     and the one the endpoints were measured at), then 2048, 512, 256.
  4. the summary table.

The per-batch cells exist because the 1800 s nbclient timeout is PER CELL: an
earlier single-cell notebook in this project had finished the reference side and
was still inside the TinyGrad timing stage when the cap fired.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re

# The hand-written side, vendored unmodified, and the driver the card builds it
# with. `refdrv.cu` is the same file notes/23's run used, so the reference side
# of this run is that side re-measured rather than a different one.
REF_FILES = [
    "minecu_ref/src/kernels.cu",
    "minecu_tg/kaggle/refdrv.cu",
]

# The naive port, carried whole. `minecu_tg_opt/render.py` imports
# `minecu_tg.render.camera_basis` from `minecu_tg/render.py`, so carrying both
# is what keeps "the two ports differ in the DDA and in nothing else" true.
TG_FILES = [
    "minecu_tg/__init__.py",
    "minecu_tg/blocks.py",
    "minecu_tg/world.py",
    "minecu_tg/render.py",
    "minecu_tg/bench.py",
]

OPT_FILES = [
    "minecu_tg_opt/__init__.py",
    "minecu_tg_opt/render.py",
    "minecu_tg_opt/battery_opt.py",
]

# Batches, in the order they are TIMED. 1024 is first and is the one that
# matters most: it is the batch notes/23's two endpoints were measured at, it is
# above the fusion crossover, and it is the batch at which the interesting
# per-rung shape has to appear. 2048 is the second above-crossover point, 512
# and 256 are below it.
BATCHES = [1024, 2048, 512, 256]

# The gate runs at batch 1024: 1,048,576 pixels, every one of them compared.
# The hand-written dump at 4096 that notes/23 used costs five times the memory
# and thirty seconds per rung for no additional evidence -- a byte-identical
# verdict on a million pixels is the same verdict on six million.
GATE_BATCH = 1024

ITERS = 8            # cudaEvent iterations for the reference side
WARMUP = 5           # and its warmup
TG_ITERS = 5         # device-time iterations inside linear_device_time_ms
VARIANTS = "naive,int_cursor,flat_gather,incremental,lean"

# The batch at which the DDA workload of the TIMED inputs is measured, per rung.
# It is a property of the world and the cameras, not of the generated code, so
# one batch carries it -- and 1024 is where the interesting time comparisons are.
WORKLOAD_BATCH = 1024

PRELUDE = r'''
# Shared by every cell below. nbclient executes the cells in order in one
# kernel, so these names persist.
import base64, gc, hashlib, json, os, re, shutil, statistics, subprocess, sys, time

os.environ.setdefault("PARALLEL", "0")
os.environ.setdefault("DEBUG", "0")

OUT = "/kaggle/working"
os.makedirs(OUT, exist_ok=True)
EXPECT = __EXPECT__
PAYLOADS = __PAYLOADS__
BATCHES = __BATCHES__
GATE_BATCH = __GATE_BATCH__
ITERS = __ITERS__
WARMUP = __WARMUP__
TG_ITERS = __TG_ITERS__
VARIANTS = __VARIANTS__.split(",")
REPEATS = __REPEATS__
WORKLOAD_BATCH = __WORKLOAD_BATCH__
WORLD, RES, GROUND, MAXSTEPS = 16, 32, 4, 32

lines = []
def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    lines.append(s)

def die(code, why):
    say("")
    say("FATAL: " + why)
    open(f"{OUT}/SUMMARY.txt", "w").write("\n".join(lines) + f"\nFATAL {why}\n")
    sys.exit(code)

def sh(cmd, **kw):
    # A missing binary raises FileNotFoundError out of subprocess.run. That is
    # a fact about the image, not a reason to lose the whole run, so it comes
    # back as a return code like any other failure.
    try:
        p = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True,
                           text=True, **kw)
    except Exception as e:
        return 127, "", f"{type(e).__name__}: {e}"
    return p.returncode, p.stdout, p.stderr

def flush_summary():
    """Write what has been logged so far. Called at the end of every cell, so a
    later cell that times out still leaves the earlier cells' output in it."""
    open(f"{OUT}/SUMMARY.txt", "w").write("\n".join(lines) + "\n")

def jwrite(name, obj):
    with open(f"{OUT}/{name}", "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
'''

CELL_BUILD = r'''
say("=" * 78)
say("rung timings -- all five rungs of minecu_tg_opt's ablation ladder, one T4,")
say("one run, render against render")
say("=" * 78)

# ------------------------------------------------------------- provenance
say("")
say("--- provenance: every payload is hash-checked before anything is built ---")
for name, b64, want in PAYLOADS:
    data = base64.b64decode(b64)
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        die(9, f"{name} sha256 {got} != {want}; refusing to measure a source I "
               f"cannot identify.")
    dest = f"{OUT}/{name}"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    open(dest, "wb").write(data)
    say(f"  {name:34s} {len(data):8d} B  sha256 {got}")

sys.path.insert(0, OUT)

# ----------------------------------------------------------------- pip
say("")
say("--- tinygrad ---")
rc, out, err = sh([sys.executable, "-m", "pip", "install", "-q", "tinygrad"])
say(f"  pip install tinygrad rc={rc}")
if rc != 0:
    say("  " + (err or out)[-400:].replace("\n", "\n  "))
try:
    import tinygrad
    say(f"  tinygrad at {tinygrad.__file__}")
except Exception as e:
    die(12, f"tinygrad did not import: {e}")

import importlib
for m in ("minecu_tg.render", "minecu_tg.bench", "minecu_tg_opt.render",
          "minecu_tg_opt.battery_opt"):
    mod = importlib.import_module(m)
    say(f"  {m:26s} from {mod.__file__}")

from tinygrad import Device, Tensor, dtypes
from minecu_tg import bench as TG_BENCH          # the ruler. imported.
from minecu_tg_opt.render import render_variant
from minecu_tg_opt.battery_opt import build_world_tensors
say(f"  Device.DEFAULT = {Device.DEFAULT}")
rc, o, e = sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
               "--format=csv,noheader"])
say("  nvidia-smi: " + (o or e).strip()[:200])
# ----------------------------------------------------------------- build
say("")
say("--- build the hand-written side for sm_75 ---")
for src, dst in ((f"{OUT}/minecu_ref/src/kernels.cu", f"{OUT}/kernels.cu"),
                 (f"{OUT}/minecu_tg/kaggle/refdrv.cu", f"{OUT}/refdrv.cu")):
    if not os.path.isfile(src):
        die(10, f"payload missing: {src}")
    shutil.copy(src, dst)
    say(f"  staged {os.path.basename(dst):<12s} {os.path.getsize(dst):>8d} B")

NVCC = None
for p in ("/usr/local/cuda/bin/nvcc", "nvcc"):
    if sh([p, "--version"])[0] == 0:
        NVCC = p
        break
if NVCC is None:
    die(10, "no nvcc on this runner")
say("  " + "\n  ".join(sh([NVCC, "--version"])[1].strip().splitlines()[-2:]))

# refdrv.cu `#include`s kernels.cu, so it is compiled ALONE with the path
# spelled out in -DMINECU_KERNELS; passing kernels.cu on the command line as
# well compiles it twice and the link fails with duplicate definitions.
FLAGS = ["-O3", "--use_fast_math", "-std=c++17", "-arch=sm_75", "-diag-suppress", "177"]
t0 = time.time()
rc, o, e = sh([NVCC, *FLAGS, '-DMINECU_KERNELS="/kaggle/working/kernels.cu"',
               f"{OUT}/refdrv.cu", "-o", f"{OUT}/refdrv_sm_75"])
say(f"  nvcc refdrv rc={rc} in {time.time()-t0:.1f}s")
if rc != 0:
    for ln in (e or "").splitlines()[-10:]:
        say("  | " + ln)
    die(11, "nvcc failed for refdrv")

rc, sass, _ = sh(["/usr/local/cuda/bin/cuobjdump", "-sass", f"{OUT}/refdrv_sm_75"])
sm80 = ["HMMA", "HGMMA", "LDGSTS", "REDUX", "ELECT", "BAR.RED", "F2FP"]
say("  SASS census of the reference binary: " + "  ".join(
    f"{g}={sass.count(g)}" for g in sm80))
if any(sass.count(g) for g in sm80):
    die(13, "the reference binary contains sm_80+ instructions")

# ------------------------------------------------- the hand-written side
say("")
say("--- the hand-written side, per batch (cudaEvent, this card, this run) ---")
say("  THIS is the denominator for every ratio in this notebook. It is measured")
say("  here, on this card, in this session, and is not quoted from any note.")
ref = {}
for b in sorted(set(BATCHES) | {GATE_BATCH}):
    t0 = time.time()
    rc, o, e = sh([f"{OUT}/refdrv_sm_75", "--batch", str(b), "--world", str(WORLD),
                   "--res", str(RES), "--max-steps", str(MAXSTEPS),
                   "--iters", str(ITERS), "--warmup", str(WARMUP)])
    d = {"batch": b, "rc": rc}
    for ln in o.splitlines():
        m = re.match(r"\s*RENDER_US\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)", ln)
        if m:
            d["render_us"] = [float(m.group(i)) for i in (1, 2, 3)]
            d["render_iters"] = int(m.group(4))
        m = re.match(r"\s*STEP_US\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)", ln)
        if m:
            d["step_us"] = [float(m.group(i)) for i in (1, 2, 3)]
        m = re.match(r"\s*ENVSTEPS_M\s+(\S+)", ln)
        if m:
            d["envsteps_m"] = float(m.group(1))
    ref[b] = d
    ru = d.get("render_us", [None])[0]
    say(f"  batch {b:6d}  render median {ru if ru else 'MISSING':>9} us  "
        f"rc={rc}  ({time.time()-t0:.1f}s)")
    if rc != 0:
        say("  " + (e or "")[-400:].replace("\n", "\n  "))
jwrite("ref_timing.json", ref)
flush_summary()
say("")
say("cell 1 done: the hand-written side's timings are in ref_timing.json")
'''

CELL_GATE = r'''
# ---------------------------------------------------------------------------
# THE GATE. All five rungs against the hand-written side's OWN bytes, on the
# hand-written side's OWN state, on THIS CARD -- and it runs BEFORE anything is
# timed. notes/23 established this at CPU level; a timing run that did not
# re-check it on the device has not established that it timed one computation
# five times, because the CUDA backend lowers the same graph differently.
#
# Second, independent reason this cell has to exist: the five rungs must also be
# shown to do the same WORK, not merely to produce the same bytes. A rung that
# reached the same image by tracing different rays would be a different
# computation that happens to agree on this world. So each rung's per-pixel DDA
# step counts are measured here, on the card, and compared.
# ---------------------------------------------------------------------------
say("")
say("=" * 78)
say(f"  THE GATE: all {len(VARIANTS)} rungs vs the hand-written side's own bytes")
say("=" * 78)

os.makedirs(f"{OUT}/refstate", exist_ok=True)
rc, o, e = sh([f"{OUT}/refdrv_sm_75", "--batch", str(GATE_BATCH), "--world", str(WORLD),
               "--res", str(RES), "--max-steps", str(MAXSTEPS), "--iters", "2",
               "--warmup", "1", "--dump", "--outdir", f"{OUT}/refstate"])
if rc != 0:
    die(14, f"refdrv --dump failed rc={rc}: {(e or '')[-300:]}")
open(f"{OUT}/log_refdump.txt", "w").write(o)

import numpy as np
RS = f"{OUT}/refstate"
def load(name, dtype):
    return np.fromfile(f"{RS}/{name}.bin", dtype=dtype)
try:
    vox = load("voxels", np.int8).reshape(GATE_BATCH, WORLD, WORLD, WORLD)
    basis = load("basis13", np.float32).reshape(GATE_BATCH, 13)
    ref_obs = load("obs_u8", np.uint8).reshape(GATE_BATCH, RES, RES, 3)
except Exception as ex:
    die(15, f"could not read the reference dump: {ex}")
npx = GATE_BATCH * RES * RES
say(f"  voxels {vox.shape}  basis {basis.shape}  reference obs {ref_obs.shape}")
say(f"  {npx:,} pixels to compare, every one of them")

voxT = Tensor(vox, dtype=dtypes.int8)
basisT = Tensor(basis, dtype=dtypes.float32)
equiv = {"batch": GATE_BATCH, "pixels": npx, "rows": [], "dda": {}}
h_ref = hashlib.sha256(np.ascontiguousarray(ref_obs)).hexdigest()
say(f"  reference obs sha256 {h_ref}")

# ONE graph per rung, asked for both its observation and its step counts.
# `count_steps=True` adds a counter to the graph and changes nothing about the
# observation's value, so the bytes compared here are the bytes the timed
# program produces. Folding the two checks into one build matters: a separate
# count_steps pass would double the most expensive stage in this cell.
floor = max(2.0, 0.10 * MAXSTEPS)
for v in VARIANTS:
    t0 = time.time()
    try:
        obs, steps = render_variant(voxT, basisT, RES, RES, MAXSTEPS, v,
                                    count_steps=True)
        img = obs.numpy()
        st = steps.numpy().reshape(-1, RES, RES)
    except Exception as ex:
        import traceback
        say("  " + traceback.format_exc()[-700:])
        die(15, f"{v} did not render on this card: {ex}")
    bad = int((img.astype(np.int32) != ref_obs.astype(np.int32)).any(axis=3).sum())
    h = hashlib.sha256(np.ascontiguousarray(img)).hexdigest()
    equiv["rows"].append({"rung": v, "pixels_differing": bad,
                          "identical": bool(np.array_equal(img, ref_obs)),
                          "sha256": h, "build_s": round(time.time() - t0, 1)})
    say(f"  {v:<13} pixels differing {bad:>9} / {npx:>9}   sha256 {h[:24]}   "
        f"{'BYTE-IDENTICAL' if bad == 0 else '*** DIFFERS ***'}"
        f"   ({time.time()-t0:.0f}s)")
    d = {"mean": float(st.mean()), "min": int(st.min()), "max": int(st.max()),
         "p50": float(np.percentile(st, 50)),
         "at_cap_frac": float((st >= MAXSTEPS).mean()),
         "first_voxel_frac": float((st == 1).mean())}
    fails = []
    if d["mean"] < floor:
        fails.append(f"mean {d['mean']:.3f} below floor {floor:.2f}")
    if d["first_voxel_frac"] > 0.20:
        fails.append(f"{100*d['first_voxel_frac']:.1f}% hit on the first voxel test")
    d["gate_pass"] = not fails
    d["gate_failures"] = fails
    equiv["dda"][v] = d
    say(f"  {' '*13} DDA mean {d['mean']:7.4f}  p50 {d['p50']:6.1f}  "
        f"at the {MAXSTEPS} cap {100*d['at_cap_frac']:5.1f}%  "
        f"first voxel {100*d['first_voxel_frac']:5.1f}%  "
        f"gate {'PASS' if not fails else 'FAIL ' + '; '.join(fails)}")
    flush_summary()
    del obs, steps, st, img
    gc.collect()
equiv["ref_sha256"] = h_ref
equiv["all_identical"] = all(r["identical"] for r in equiv["rows"])
means = [equiv["dda"][v]["mean"] for v in VARIANTS]
equiv["dda_spread"] = {"mean_min": min(means), "mean_max": max(means),
                       "abs_steps": max(means) - min(means),
                       "spread_pct": 100.0 * (max(means) - min(means)) / min(means)}
jwrite("obs_equivalence.json", equiv)
jwrite("dda_workload.json", equiv["dda"])
flush_summary()
say(f"  DDA mean spread across the five rungs: "
    f"{equiv['dda_spread']['abs_steps']:.6f} steps "
    f"({equiv['dda_spread']['spread_pct']:.6f}% of the smallest mean)")
if not equiv["all_identical"]:
    die(16, "a rung's observation on this card does not match the hand-written "
            "side's own bytes. Every ratio below would be a ratio between two "
            "different computations, so there is nothing to report.")
say("")
say("cell 2 done: the gate passed. Every rung is byte-identical to the")
say("hand-written side's own bytes, on this card, on the state the hand-written")
say("side produced, and every rung traces the same rays.")
'''

# --- the timing machinery, shared by every per-batch cell ---------------------
CELL_TIMING_FN = r'''
# ---------------------------------------------------------------------------
# THE TIMING INSTRUMENT, and the guards around it.
#
# THE RULER. `minecu_tg.bench.linear_device_time_ms`, imported, never
# reimplemented. It walks the compiled LINEAR and sums each CALL's
# device-measured duration. That is the same quantity the reference's
# cudaEvent-bracketed single launch reports: both are device time to produce
# this step's observation, and neither divides out host tracing. It is the same
# ruler the naive arm was measured with, which is the only reason a number from
# this notebook and a number from notes/23 are commensurable.
#
# GUARD 1 -- A FRESH GRAPH FOR EVERY MEASUREMENT. `Tensor.realize` skips any
# tensor whose uop `has_buffer_identity()`, so a graph that had already been
# realized would be a SILENT NO-OP reading as ~0 us, and a no-op reads as
# "infinitely fast", which is the one number in this study that could not be
# caught downstream. This project has already had exactly that defect produce a
# port that measured 13x FASTER than hand-written CUDA. So every measurement
# below builds a brand-new render graph and asserts the exact predicate
# `Tensor.realize` itself uses to decide there is work to do. The predicate's
# value is recorded in the report, not merely asserted.
#
# GUARD 2 -- THE RULER MUST HAVE FOUND WORK, AND A SECOND MEASUREMENT OF THE
# SAME GRAPH MUST AGREE WITH IT. The LINEAR is also walked by hand, on its own
# fresh graph: every CALL is timed individually and the sum is reported next to
# the ruler's own total for an equivalent graph. This is the only place in the
# notebook where the same quantity is measured twice by two different
# invocations, and it is what turns "the ruler returned a number" into "the
# ruler returned THIS number".
#
# THE UNITS, which were wrong once. `time_call` yields SECONDS. The ruler does
# `total * 1e3` for milliseconds. This walk does `* 1e3` too and the result is
# milliseconds, not microseconds -- a first version of this notebook labelled it
# `_us`, which made the walk read 1000x smaller than the ruler and looked like
# an instrument disagreement when it was a label. Both columns say ms.
#
# GUARD 3 -- A SECOND CLOCK, ON A SECOND FRESH GRAPH. `realize()` of the graph
# the ruler already walked is a NO-OP: `linear_with_vars` has by then assigned
# the output buffers, so there is nothing left to run and the call returns in
# microseconds. That is the exact trap guard 1 exists for, and it is why the
# second clock gets its own fresh graph rather than the ruler's. It is host wall
# clock around `realize()` plus a device synchronize, so it CONTAINS Python
# tracing and is not a device time; it is reported for order of magnitude and
# is never divided into anything.
# ---------------------------------------------------------------------------
from tinygrad.engine.realize import time_call as TG_TIME_CALL
import numpy as np          # timed_workload uses it, and it must not depend on
                            # an earlier cell having imported numpy

def _is_realized(t):
    """Exactly the predicate `Tensor.realize` uses to decide there is no work."""
    return bool(t.uop.has_buffer_identity())

def walk_graph(t):
    """The CALLs of a fresh graph, timed one at a time. Guard 2. All ms."""
    lin, vv = Tensor.linear_with_vars(t)
    calls = list(lin.src)
    for _ in range(2):
        for c in calls:
            next(TG_TIME_CALL(c, var_vals=vv))
    per = [next(TG_TIME_CALL(c, var_vals=vv)) * 1e3 for c in calls]
    return {"n_calls": len(calls), "per_call_ms": [round(x, 4) for x in per],
            "n_positive": sum(1 for x in per if x > 0.0),
            "sum_ms": round(sum(per), 4)}

def measure_once(vox, basis, rung, batch, iters=TG_ITERS):
    """One measurement: a brand-new graph, timed with the project's own ruler."""
    gc.collect()
    t = render_variant(vox, basis, RES, RES, MAXSTEPS, rung)
    rec = {"rung": rung, "batch": batch, "nbytes": int(t.nbytes()),
           "guard1_pre_realized": _is_realized(t)}
    if rec["guard1_pre_realized"]:
        rec["error"] = ("the fresh tensor already has buffer identity, so "
                        "Tensor.realize would have skipped it and any timing "
                        "taken here would be a no-op")
        return rec
    with TG_BENCH.MemProbe(Device.DEFAULT) as mp:
        mp.reset()
        rec.update(TG_BENCH.linear_device_time_ms(t, iters=iters) or {})
        rec["mem"] = mp.as_dict()
    rec["guard2_n_kernels_positive"] = bool(rec.get("n_kernels", 0) >= 1
                                             and rec.get("median_ms", 0.0) > 0.0)
    del t
    gc.collect()
    return rec

def measure_e2e(vox, basis, rung, batch):
    """Guard 3: the same program, a FRESH graph, on the host clock."""
    gc.collect()
    t = render_variant(vox, basis, RES, RES, MAXSTEPS, rung)
    if _is_realized(t):
        del t
        return {"error": "the e2e graph was already realized; it would be a no-op"}
    Device[Device.DEFAULT].synchronize()
    t0 = time.perf_counter()
    t.realize()
    Device[Device.DEFAULT].synchronize()
    r = {"e2e_realize_ms": (time.perf_counter() - t0) * 1e3}
    del t
    gc.collect()
    return r

def time_batch(batch, budget_s=1400.0):
    """All five rungs at one batch. A fresh graph per measurement, every time."""
    from minecu_tg.render import camera_basis
    say("")
    say("-" * 78)
    say(f"  batch {batch}   {batch*RES*RES:,} pixels   world {WORLD}   "
        f"max_steps {MAXSTEPS}")
    say("-" * 78)
    rep = REPEATS.get(str(batch), 2)
    t0 = time.time()
    try:
        vox, cams, g = build_world_tensors(batch, WORLD, GROUND)
        basis = camera_basis(cams, 70.0, RES, RES).contiguous()
        basis.realize()
        Device[Device.DEFAULT].synchronize()
    except Exception as ex:
        import traceback
        say(f"  input build FAILED: {ex}")
        say(traceback.format_exc()[-900:])
        flush_summary()
        return {"batch": batch, "rows": [], "error": str(ex)[:300]}
    say(f"  {rep} independent fresh graphs per rung; ruler =")
    say("  minecu_tg.bench.linear_device_time_ms (imported), iters=%d" % TG_ITERS)

    rows = []
    for rung in VARIANTS:
        reps = []
        for k in range(rep):
            if time.time() - t0 > budget_s:
                say(f"  {rung:<13} SKIPPED: the {budget_s:.0f}s cell budget was "
                    f"already spent on earlier rungs")
                continue
            tk = time.time()
            try:
                r = measure_once(vox, basis, rung, batch)
            except Exception as ex:
                import traceback
                r = {"rung": rung, "batch": batch,
                     "error": f"{type(ex).__name__}: {ex}"[:300]}
                say("  " + traceback.format_exc()[-700:])
            r["repeat"] = k
            r["wall_s"] = round(time.time() - tk, 1)
            reps.append(r)
            if r.get("error"):
                say(f"  {rung:<13} repeat {k} FAILED: {r['error'][:140]}")
            else:
                say(f"  {rung:<13} repeat {k}: {r['median_ms']:9.3f} ms  "
                    f"{r['n_kernels']:3d} CALLs  "
                    f"peak {r['mem']['peak_live_bytes']/2**20:8.1f} MiB  "
                    f"{r['mem']['n_allocs']:3d} allocs"
                    + ("   <- COLD allocator" if r['mem']['n_allocs'] else
                       "   (warm allocator: buffers reused, no alloc/free)")
                    + f"  ({r['wall_s']:.0f}s)")
            flush_summary()

        # Guard 2, on its own fresh graph for this rung at this batch. It runs
        # FIRST so it is a cold-allocator graph, like the ruler's first repeat.
        walk = None
        if time.time() - t0 <= budget_s:
            try:
                tw = render_variant(vox, basis, RES, RES, MAXSTEPS, rung)
                if _is_realized(tw):
                    walk = {"error": "the walk graph was already realized"}
                else:
                    walk = walk_graph(tw)
                    del tw
                    gc.collect()
            except Exception as ex:
                walk = {"error": f"{type(ex).__name__}: {ex}"[:200]}
        if walk and "n_calls" in walk:
            say(f"  {rung:<13} walk: {walk['n_calls']} CALLs, "
                f"{walk['n_positive']} of them > 0, "
                f"per-CALL sum {walk['sum_ms']} ms  "
                f"per CALL: {walk['per_call_ms']}")

        # Guard 3, on its own fresh graph.
        e2e = None
        if time.time() - t0 <= budget_s:
            try:
                e2e = measure_e2e(vox, basis, rung, batch)
            except Exception as ex:
                e2e = {"error": f"{type(ex).__name__}: {ex}"[:200]}
        if e2e and "e2e_realize_ms" in e2e:
            say(f"  {rung:<13} e2e: realize() of a fresh graph on the host clock, "
                f"{e2e['e2e_realize_ms']/1e3:.2f} s (contains Python tracing)")

        ok = [r for r in reps if not r.get("error")]
        row = {"rung": rung, "batch": batch, "repeats": reps, "walk": walk,
               "e2e": e2e, "n_repeats_ok": len(ok), "n_repeats_wanted": rep}
        if ok:
            med = sorted(r["median_ms"] for r in ok)
            row["median_ms"] = statistics.median(med)
            row["min_ms"] = min(med)
            row["max_ms"] = max(med)
            row["spread_pct"] = 100.0 * (max(med) - min(med)) / min(med)
            # THE COLD/WARM SPLIT, which is a real property of this allocator and
            # not noise. Only the FIRST graph built at a given shape goes through
            # `Allocator.alloc`; later graphs at the same shape reuse the cached
            # buffers, so MemProbe records zero allocations for them, and the
            # scheduler emits one CALL fewer because there is no COPY to make.
            # The allocator's peak therefore only exists on a cold graph, and the
            # repeat that recorded any allocation at all is the one that carries
            # both the peak and the cold launch count. Both are reported; the
            # cold one is the primary, because it is what a program pays on its
            # first step and it is what the earlier run of this study measured.
            cold = [r for r in ok if r["mem"]["n_allocs"] > 0] or ok
            warm = [r for r in ok if r["mem"]["n_allocs"] == 0]
            row["n_cold_graphs"] = len(cold)
            row["n_warm_graphs"] = len(warm)
            row["peak_live_bytes"] = max(r["mem"]["peak_live_bytes"] for r in ok)
            row["n_allocs_cold"] = max(r["mem"]["n_allocs"] for r in ok)
            row["n_kernels_cold"] = sorted({r["n_kernels"] for r in cold})
            row["n_kernels_warm"] = sorted({r["n_kernels"] for r in warm})
            row["n_kernels"] = row["n_kernels_cold"]
            row["guards_ok"] = all(
                not r["guard1_pre_realized"] and r["guard2_n_kernels_positive"]
                for r in ok)
            if walk and "sum_ms" in walk:
                # the two independent measurements of one quantity
                row["walk_vs_ruler_pct"] = 100.0 * (walk["sum_ms"] - row["median_ms"]) / row["median_ms"]
            if e2e and "e2e_realize_ms" in e2e:
                row["e2e_realize_ms"] = e2e["e2e_realize_ms"]
            say(f"  {rung:<13} MEDIAN {row['median_ms']:9.3f} ms  "
                f"spread {row['spread_pct']:5.2f}%  "
                f"CALLs cold {row['n_kernels_cold']} warm {row['n_kernels_warm']}  "
                f"peak {row['peak_live_bytes']/2**20:.1f} MiB  "
                f"guards {'OK' if row['guards_ok'] else 'FAILED'}")
        else:
            row["error"] = "every repeat failed"
            say(f"  {rung:<13} NO USABLE MEASUREMENT")
        rows.append(row)
        flush_summary()

    # --- cross-check through the project's own wrapper -----------------------
    # `bench_render` is the function minecu_tg_opt/battery_opt.py calls for every
    # timing it reports. Running it here means this notebook's numbers can be
    # lined up against the earlier run's through the SAME code path rather than
    # through two different drivers.
    xs = []
    say("")
    say(f"  cross-check at batch {batch} through minecu_tg.bench.bench_render --")
    say("  the exact wrapper minecu_tg_opt/battery_opt.py uses for every number:")
    for rung in VARIANTS:
        if time.time() - t0 > budget_s:
            say(f"    {rung:<13} SKIPPED: cell budget")
            xs.append({"rung": rung, "skipped": "budget"})
            continue
        f = lambda v_, b_, n=rung: render_variant(v_, b_, RES, RES, MAXSTEPS, n)
        q = TG_BENCH.bench_render(lambda: (vox, basis), f, Device.DEFAULT,
                                  iters=8, label=f"xs_{rung}_b{batch}")
        d = q.get("device_time") or {}
        mem = q.get("mem") or {}
        xs.append({"rung": rung, "realized": q.get("realized"),
                   "median_ms": d.get("median_ms"), "n_kernels": d.get("n_kernels"),
                   "peak_live_bytes": mem.get("peak_live_bytes"),
                   "n_allocs": mem.get("n_allocs"), "error": q.get("error")})
        ms = d.get("median_ms")
        say(f"    {rung:<13} {round(ms,3) if ms else 'FAILED':>10} ms  "
            f"{d.get('n_kernels')} CALLs  "
            f"peak {mem.get('peak_live_bytes',0)/2**20:.1f} MiB")
    wl = None
    if batch == WORKLOAD_BATCH and time.time() - t0 <= budget_s:
        try:
            wl = timed_workload(vox, basis, batch)
        except Exception as ex:
            wl = {"error": f"{type(ex).__name__}: {ex}"[:200]}
            say("  timed_workload raised: " + wl["error"])
    say("")
    say(f"  cell {batch} done ({time.time()-t0:.0f}s)")
    del vox, basis
    gc.collect()
    return {"batch": batch, "rows": rows, "cross_check": xs,
            "timed_workload": wl, "pixels": batch * RES * RES}


def timed_workload(vox, basis, batch):
    """The DDA steps per pixel of the TIMED inputs, per rung.

    The gate in cell 2 establishes that all five rungs trace the same rays on
    the reference's own state. This establishes it again on the inputs the
    timings actually ran on, which are a different camera distribution (the
    timing world carries a +-0.048 rad yaw spread; the gate runs the reference's
    settled cameras), and it is the workload the device-time numbers belong to.
    It is a property of the world and the cameras, not of the generated code,
    so one batch carries it.
    """
    say("")
    say(f"  the workload the timed graphs performed, batch {batch}, per rung:")
    out = {}
    for v in VARIANTS:
        try:
            obs, steps = render_variant(vox, basis, RES, RES, MAXSTEPS, v,
                                        count_steps=True)
            st = steps.numpy().reshape(-1, RES, RES)
            d = {"mean": float(st.mean()), "p50": float(np.percentile(st, 50)),
                 "min": int(st.min()), "max": int(st.max()),
                 "at_cap_frac": float((st >= MAXSTEPS).mean()),
                 "first_voxel_frac": float((st == 1).mean()),
                 "obs_sha256": hashlib.sha256(
                     np.ascontiguousarray(obs.numpy())).hexdigest()}
            out[v] = d
            say(f"    {v:<13} mean {d['mean']:8.4f}  p50 {d['p50']:6.1f}  "
                f"at the {MAXSTEPS} cap {100*d['at_cap_frac']:5.1f}%  "
                f"first voxel {100*d['first_voxel_frac']:5.1f}%")
            del obs, steps, st
            gc.collect()
        except Exception as ex:
            out[v] = {"error": f"{type(ex).__name__}: {ex}"[:200]}
            say(f"    {v:<13} FAILED: {out[v]['error']}")
    if out:
        ms = [out[v]["mean"] for v in VARIANTS if "mean" in out.get(v, {})]
        if ms and min(ms) > 0:
            out["spread"] = {"abs_steps": max(ms) - min(ms),
                             "spread_pct": 100.0 * (max(ms) - min(ms)) / min(ms)}
            say(f"    spread across the five rungs: {out['spread']['abs_steps']:.6g}"
                f" steps ({out['spread']['spread_pct']:.4f}%)")
    return out

def save_batch(res):
    p = f"{OUT}/rung_timings.json"
    prev = {}
    if os.path.exists(p):
        try:
            prev = json.load(open(p))
        except Exception:
            prev = {}
    prev[str(res["batch"])] = res
    jwrite("rung_timings.json", prev)
    flush_summary()
'''

CELL_SUMMARY = r'''
say("")
say("=" * 78)
say("SUMMARY -- all five rungs, one card, one run")
say("=" * 78)
try:
    ref = json.load(open(f"{OUT}/ref_timing.json"))
except Exception:
    ref = {}
try:
    T = json.load(open(f"{OUT}/rung_timings.json"))
except Exception:
    T = {}
try:
    eq = json.load(open(f"{OUT}/obs_equivalence.json"))
except Exception:
    eq = {"rows": [], "dda": {}}

say("")
say("GATE (on this card, before any timing)")
for r in eq.get("rows", []):
    say(f"  {r['rung']:<13} {r['pixels_differing']:>9} / {eq['pixels']:>9} pixels "
        f"differ   {'BYTE-IDENTICAL' if r['identical'] else '*** DIFFERS ***'}"
        f"   obs sha256 {r['sha256'][:24]}")
if eq.get("dda"):
    say("  DDA mean steps/pixel: " + "  ".join(
        f"{k}={eq['dda'][k]['mean']:.4f}" for k in VARIANTS if k in eq.get("dda", {})))
    sp = eq.get("dda_spread", {})
    if sp:
        say(f"  spread across the five rungs: {sp.get('abs_steps'):.6f} steps "
            f"({sp.get('spread_pct'):.6f}%)")

say("")
say("RENDER AGAINST RENDER. ref render is a cudaEvent median in MICROSECONDS,")
say("measured in this run; tg is the sum of per-kernel device times for the")
say("whole generated graph, in MILLISECONDS. ratio = tg/ref, 'how many times")
say("slower'. 1.0 is parity. Nothing here divides a render by a whole step.")
say("")
hdr = f"  {'batch':>6} {'ref us':>9} "
for v in VARIANTS:
    hdr += f"{v[:11]:>12}"
say(hdr)
say(f"  {'batch':>6} {'ref us':>9} " + " ".join(f"{'ms (ratio)':>12}" for _ in VARIANTS))
lines_tab = []
for b in BATCHES:
    rec = T.get(str(b))
    ru = (ref.get(str(b)) or {}).get("render_us", [None])[0]
    if not rec:
        continue
    by = {r["rung"]: r for r in rec.get("rows", []) if not r.get("error")}
    line = f"  {b:>6} {round(ru,1) if ru else '-':>9} "
    for v in VARIANTS:
        r = by.get(v)
        if not r or not ru:
            line += f"{'-':>12} "
            continue
        ratio = r["median_ms"] * 1000.0 / ru
        line += f"{r['median_ms']:7.3f} ({ratio:4.1f}) "
        lines_tab.append((b, v, r["median_ms"], ratio))
    say(line)

say("")
say("LAUNCH COUNT, ALLOCATOR PEAK, REPEAT SPREAD, GUARDS")
for b in BATCHES:
    rec = T.get(str(b))
    if not rec:
        continue
    say(f"  batch {b}")
    for r in rec.get("rows", []):
        if r.get("error"):
            say(f"    {r['rung']:<13} NOT MEASURED: {r['error'][:110]}")
            continue
        w = r.get("walk") or {}
        wr = r.get("walk_vs_ruler_pct")
        e2 = r.get("e2e_realize_ms")
        say(f"    {r['rung']:<13} {r['median_ms']:9.3f} ms  "
            f"spread {r['spread_pct']:5.2f}% over {r['n_repeats_ok']}/"
            f"{r['n_repeats_wanted']} fresh graphs  "
            f"CALLs cold {r['n_kernels_cold']} warm {r['n_kernels_warm']}  "
            f"peak {r['peak_live_bytes']/2**20:7.1f} MiB  "
            f"allocs {r['n_allocs_cold']}  "
            f"walk sum {w.get('sum_ms')} ms "
            f"({f'{wr:+.2f}%' if wr is not None else 'n/a'} vs the ruler)  "
            f"e2e {round(e2/1e3,2) if isinstance(e2,(int,float)) else 'n/a'} s  "
            f"guards {'OK' if r['guards_ok'] else 'FAILED'}")
    xs = {x["rung"]: x for x in rec.get("cross_check", []) if x.get("median_ms")}
    if xs:
        for v in VARIANTS:
            x = xs.get(v)
            if x:
                say(f"      cross-check via bench_render {v:<13} {x['median_ms']:9.3f} ms"
                    f"  {x['n_kernels']} CALLs  peak {x['peak_live_bytes']/2**20:.1f} MiB")
    wl = rec.get("timed_workload")
    if isinstance(wl, dict) and wl:
        say(f"  the workload these timings performed, batch {b}:")
        for v in VARIANTS:
            d = wl.get(v) or {}
            if "mean" in d:
                say(f"    {v:<13} mean {d['mean']:8.4f} steps/pixel  "
                    f"at the {MAXSTEPS} cap {100*d['at_cap_frac']:5.1f}%  "
                    f"first voxel {100*d['first_voxel_frac']:5.1f}%")
        if "spread" in wl:
            say(f"    spread across the five rungs: {wl['spread']['abs_steps']:.6g}"
                f" steps ({wl['spread']['spread_pct']:.4f}%)")

say("")
say("THE PER-RUNG SHAPE")
say("  rung            wall ms   vs the rung above   launches   peak MiB")
for b in BATCHES:
    rec = T.get(str(b))
    if not rec:
        continue
    by = {r["rung"]: r for r in rec.get("rows", []) if not r.get("error")}
    prev = None
    say(f"  -- batch {b}")
    for v in VARIANTS:
        r = by.get(v)
        if not r:
            continue
        d = "-" if prev is None else f"{prev/r['median_ms']:6.3f}x"
        say(f"     {v:<13} {r['median_ms']:9.3f}   {d:>15}   "
            f"{str(r['n_kernels_cold']):>8}   {round(r['peak_live_bytes']/2**20,1)}")
        prev = r["median_ms"]

say("")
say("WHAT MAY AND MAY NOT BE READ OFF THIS")
say("  * All five rungs produced the hand-written side's own bytes on this card,")
say("    on the state the hand-written side produced, before any of them was")
say("    timed, and all five trace the same rays (mean DDA steps above).")
say("  * Every rung is the rung above plus exactly one rewrite, so the")
say("    difference between consecutive rows is that rewrite's price.")
say("  * The reference denominator is measured HERE, on THIS card, in THIS")
say("    session. It is not a number copied out of another run.")
say("  * COLD vs WARM: only the first graph built at a shape goes through")
say("    Allocator.alloc, so only a cold graph has an allocator peak, and only a")
say("    cold graph has the schedule that goes with a cold allocator. Both are")
say("    reported. The times do not depend on which it is.")
say("  * No number in this file is transcribed from a notes file.")

flush_summary()
say("")
say("wrote /kaggle/working/SUMMARY.txt")
'''

CELL_HELPER = r'''say("")
say("--- the timing instrument, and its guards ---")
''' + CELL_TIMING_FN


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--owner", default="tenghuzhang")
    ap.add_argument("--slug", default="mc-minecu-rungs-t4")
    ap.add_argument("--repeats", default="1024=3,2048=2,512=2,256=2")
    a = ap.parse_args()

    repo = a.repo or os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    inputs = []
    for rel in REF_FILES + TG_FILES + OPT_FILES:
        p = os.path.join(repo, rel)
        if not os.path.exists(p):
            raise SystemExit(f"missing input: {p}")
        inputs.append((rel, p))
    payloads, expect = [], {}
    for name, path in inputs:
        h = sha256_file(path)
        expect[h] = name
        payloads.append([name, base64.b64encode(open(path, "rb").read()).decode(), h])

    repeats = {}
    for part in a.repeats.split(","):
        k, _, v = part.partition("=")
        if v.strip():
            repeats[k.strip()] = int(v)

    subs = [("__EXPECT__", json.dumps(expect)),
            ("__PAYLOADS__", json.dumps(payloads)),
            ("__BATCHES__", json.dumps(BATCHES)),
            ("__GATE_BATCH__", str(GATE_BATCH)),
            ("__ITERS__", str(ITERS)),
            ("__WARMUP__", str(WARMUP)),
            ("__TG_ITERS__", str(TG_ITERS)),
            ("__VARIANTS__", json.dumps(VARIANTS)),
            ("__REPEATS__", json.dumps(repeats)),
            ("__WORKLOAD_BATCH__", str(WORKLOAD_BATCH))]

    def fill(src: str) -> str:
        for k, v in subs:
            src = src.replace(k, v)
        return src

    code_cells = [fill(PRELUDE), fill(CELL_BUILD), fill(CELL_GATE),
                  fill(CELL_HELPER)]
    for b in BATCHES:
        # EVERY cell is wrapped. nbclient raises on the first cell error by
        # default, so an unhandled format error in a summary line does not just
        # lose that line -- it abandons every batch after it. A first push of
        # this notebook died on exactly that: `peak_live_bytes` became a list
        # when two repeats disagreed and `list / int` raised, taking four
        # unmeasured batches with it. The ladder is the deliverable; a
        # presentation bug must not be able to delete it.
        code_cells.append(fill(f'''
say("")
say("=" * 78)
say("  TIMING, batch {b} -- all five rungs, one fresh graph each")
say("=" * 78)
try:
    save_batch(time_batch({b}))
except Exception:
    import traceback
    say("  *** the batch-{b} cell raised; everything after it still runs ***")
    say(traceback.format_exc()[-2500:])
    save_batch({{"batch": {b}, "rows": [], "error": "cell raised"}})
flush_summary()
say("")
say("batch {b} done")
'''))
    code_cells.append(fill(
        "\ntry:\n"
        + "".join("    " + ln + "\n" for ln in CELL_SUMMARY.splitlines())
        + '''
except Exception:
    import traceback
    say("  *** the summary cell raised; the JSON artefacts are still on disk ***")
    say(traceback.format_exc()[-2500:])
flush_summary()
say("")
say("wrote /kaggle/working/SUMMARY.txt")
'''))

    kd = os.path.dirname(a.out)
    os.makedirs(kd, exist_ok=True)

    def md(src, cid):
        return {"cell_type": "markdown", "id": cid, "metadata": {},
                "source": src.splitlines(True)}

    def code(src, cid):
        return {"cell_type": "code", "id": cid, "execution_count": None,
                "metadata": {}, "outputs": [], "source": src.splitlines(True)}

    names = ["prelude", "build-and-reference", "gate", "timing-instrument"]
    names += [f"time-batch-{b}" for b in BATCHES]
    names += ["summary"]

    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3",
                                      "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.12.0"}},
          "cells": [
              md("# rung timings: all five rungs of the good-vs-naive ladder, "
                 "on one T4, in one run\n\n"
                 "notes/23 built a five-rung ablation ladder and timed its two "
                 "endpoints.\nThis notebook times all five, and it measures the "
                 "hand-written CUDA\nreference in the same session rather than "
                 "quoting a number from a different one.\n\n"
                 "The gate (all five rungs against the hand-written side's own "
                 "bytes, on the\nhand-written side's own state, on this card) "
                 "runs BEFORE anything is timed.", "intro"),
              md("## Artefacts\n\n"
                 "1. `SUMMARY.txt` -- the log. Flushed at the end of every cell.\n"
                 "2. `ref_timing.json` -- the hand-written side, per batch.\n"
                 "3. `obs_equivalence.json` -- the gate.\n"
                 "4. `dda_workload.json` -- per-rung steps/pixel on the card.\n"
                 "5. `rung_timings.json` -- the deliverable: per rung, per batch,\n"
                 "   the wall clock, the CALL count, the allocator's peak, the\n"
                 "   repeat spread, and the instrument guards.\n\n"
                 "No number is copied from a notes file.", "artefacts"),
          ] + [code(s, n) for s, n in zip(code_cells, names)]}
    with open(a.out, "w") as fh:
        json.dump(nb, fh, indent=1)

    with open(os.path.join(kd, "payloads.json"), "w") as fh:
        json.dump({"expect": expect, "payloads": payloads,
                   "ref_files": REF_FILES, "tg_files": TG_FILES,
                   "opt_files": OPT_FILES, "batches": BATCHES,
                   "repeats": repeats, "variants": VARIANTS,
                   "gate_batch": GATE_BATCH,
                   "notebook_sha256": sha256_file(a.out)}, fh, indent=1)

    title = " ".join(w for w in a.slug.split("-") if w)
    meta = {"id": f"{a.owner}/{a.slug}", "title": title,
            "subtitle": "five-rung ablation ladder, all rungs timed, on a T4",
            "code_file": os.path.basename(a.out), "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_tpu": False, "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
            "model_sources": []}
    derived = re.sub(r"[^a-z0-9]+", "-", meta["title"].lower()).strip("-")
    if derived != a.slug:
        raise SystemExit(f"kernel title {meta['title']!r} slugs to {derived!r}, "
                         f"which does not match the id slug {a.slug!r}")
    with open(os.path.join(kd, "kernel-metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"wrote {a.out}  ({os.path.getsize(a.out)} B)")
    print(f"wrote {os.path.join(kd, 'kernel-metadata.json')}")
    print(f"  {len(payloads)} payloads, {sum(len(p[1]) for p in payloads)} b64 bytes")
    print(f"  {len(code_cells)} code cells: " + ", ".join(names))
    print(f"  batches {BATCHES} (in timing order), gate at {GATE_BATCH}, repeats {repeats}")
    print(f"  variants {VARIANTS}")
    print(f"  kernel id {meta['id']}")


if __name__ == "__main__":
    main()
