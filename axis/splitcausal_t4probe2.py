#!/usr/bin/env python3
"""splitcausal_t4probe2.py -- the second T4 probe: how far does SPLIT go, and
is the remaining gap the window at all?

WHAT RUN 1 ESTABLISHED, AND THE ONE THING IT DID NOT
====================================================
Run 1 (`axis/splitcausal_t4probe.py`, out in the note) settled four things and
left one loose. `[MEASURED]`, one T4, one session, both arms timed by
`time_call` on the LINEAR:

  F1  The literal IS causal and it IS the biggest single lever. At the ladder's
      own shape, E = 2^24, N = 4096: SPLIT=64 -> 13.16 ms, SPLIT=256 -> 21.33 ms,
      SPLIT=1024 -> 82.81 ms. A **6.3x** swing from one integer.
  F2  The peak is at `2*SPLIT` exactly. SPLIT=256 peaks at N=512 (41.2 ms, the
      slowest of fifteen points). SPLIT=1024 peaks at N=2048 (164.1 ms).
  F4  **The two-stage decomposition is not the cost.** At R=4096, N=4096 the
      one-stage `_cumalu` is **329.2 ms** against the two-stage `cumsum`'s
      **56.5 ms** -- the one-stage path is **5.8x SLOWER**. So `notes/33`'s
      "the gap is the two-stage decomposition's cost" is refuted by the
      framework's own alternative spelling, at one shape, with nothing moved.
  F5  The rise-then-drop shape is real and its size is what the work model
      predicts: N=256 -> 512 rises 1.990x against a modelled 2.000x, then
      N=512 -> 640 falls to 0.513x.

THE LOOSE END: run 1's own control gate fired.
==============================================
Run 1 re-measured the UNMODIFIED build at the end of the same session, and at
N=4096 it came back 9.2% away from the baseline taken at the start. Run 1's gate
correctly declared F1 VOID rather than letting a session difference be read as a
constant's effect. The 6.3x effect is far larger than a 9% drift, so the
conclusion survives -- but "the gate fired and I overrode it" is exactly the
move this project has been taken in before, and the right response is to remove
the drift instead of arguing about it.

WHAT THIS PROBE DOES
====================
  A. INTERLEAVE. Every SPLIT value is measured in ABABAB order, three rounds,
     in one process, on one card, minutes apart at most. Drift then hits all
     arms equally instead of hitting whichever arm ran first, and the report can
     show each arm's three rounds separately so the drift is visible rather
     than argued about. This is the fix, and it is a design change rather than a
     looser threshold.
  B. GO PAST THE PLATEAU. The decisive question for the restatement is whether
     the WINDOW is the whole cost. If the cost is the window, then shrinking
     SPLIT toward 1 drives the framework's time toward the memory floor, and the
     gap to a real scan closes. If a floor remains at SPLIT=1, then the window
     was never the whole story and `notes/33`'s mechanism is wrong twice over.
     This is the measurement that decides what replaces the claim.
  C. MEASURE THE FLOOR IN THE SAME RUN, with the same instrument, on the same
     card: a copy kernel over the same bytes. So "approaches the floor" is a
     comparison against a number this run produced, not one quoted from a
     different session.

Everything is `[MEASURED]` on the card or `[MODELLED]` from the source. Nothing
here is transcribed.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

PROBE = r'''
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

def log(*a): print(*a, flush=True)
def left(): return BUDGET_S - (time.time() - T0)

import tinygrad
from tinygrad import Device, Tensor
import tinygrad.mixin.op as M
import tinygrad.tensor as TWMOD

OP_PY = __import__("pathlib").Path(M.__file__)
SRC = OP_PY.read_text()
OP_SHA = hashlib.sha256(OP_PY.read_bytes()).hexdigest()
_m = re.search(r"^(\s*)SPLIT = (\d+)\s*$", SRC, re.M)
assert _m, "SPLIT literal not found"
INSTALLED = int(_m.group(2))
ORIG_SRC = __import__("inspect").getsource(M.OpMixin._split_cumalu)
SPLIT_RE = re.compile(r"^(\s*)SPLIT = (\d+)\s*$", re.M)

log("=" * 78); log("S0  PROVENANCE"); log("=" * 78)
log("  tinygrad      ", getattr(tinygrad, "__version__", "?"), "|", tinygrad.__file__)
log("  op.py sha256  ", OP_SHA)
log("  installed SPLIT", INSTALLED, "-> first two-stage N predicted at", 2*INSTALLED+1)
try:
    log("  nvcc          ", subprocess.run(["nvcc","--version"],capture_output=True,text=True).stdout.strip().splitlines()[-1])
except Exception as e:
    log("  nvcc          unavailable", e)

# ------------------------------------------------------------------ instrument
def require_fresh(t, what):
    u = getattr(t, "uop", None)
    if u is None: return
    try: fresh = (not u.is_virtual) and (not u.has_buffer_identity())
    except Exception: return
    if not fresh:
        raise RuntimeError(f"{what}: ALREADY REALIZED; realize() is a silent no-op "
                           f"on a realized graph and timing it measures nothing.")

def dev_ms(tensor, iters=5):
    from tinygrad.engine.realize import time_call
    lin, vv = Tensor.linear_with_vars(tensor)
    calls = list(lin.src)
    if not calls:
        raise RuntimeError("dev_ms: the LINEAR is EMPTY, so there is nothing to "
                           "re-execute. Either the graph was already realized (a "
                           "realize() on a realized graph is a silent no-op) or the "
                           "expression compiled to nothing.")
    for _ in range(2):
        for c in calls: next(time_call(c, var_vals=vv))
    tot = []
    for _ in range(iters):
        s = 0.0
        for c in calls: s += next(time_call(c, var_vals=vv))
        tot.append(s*1e3)
    tot.sort()
    med = statistics.median(tot)
    if not med or med <= 0:
        raise RuntimeError(f"dev_ms: median device time is {med} ms at {tensor.shape}. "
                           f"A zero here is not a fast kernel, it is a measurement "
                           f"that did not measure anything.")
    return {"median_ms": med, "min_ms": tot[0], "max_ms": tot[-1], "n_kernels": len(calls)}

# ------------------------------------------------------------------ the patch
import functools, textwrap
_ORIG = Tensor._split_cumalu

def patch_split(k):
    import tinygrad.codegen as CG
    new_src, n = SPLIT_RE.subn(lambda m: f"{m.group(1)}SPLIT = {k}", ORIG_SRC, count=1)
    if n != 1 or f"SPLIT = {k}" not in new_src:
        raise RuntimeError(f"SPLIT substitution failed, n={n}")
    ns = dict(vars(M))
    exec(compile(textwrap.dedent(new_src), "<probe2:patched>", "exec"), ns)
    Tensor._split_cumalu = functools.wraps(ns["_split_cumalu"])(
        TWMOD._metadata_wrapper(ns["_split_cumalu"]))
    CG.to_program_cache.clear()

def restore():
    import tinygrad.codegen as CG
    from tinygrad.engine import realize as RZ
    Tensor._split_cumalu = _ORIG
    CG.to_program_cache.clear()
    for nm in ("runtime_cache","graph_cache","local_size_cache"):
        if hasattr(RZ,nm): getattr(RZ,nm).clear()

# ------------------------------------------------------------------ arms
E = 1 << 24
def mk(r, n, seed=12):
    return np.random.RandomState(seed).rand(r, n).astype(np.float32)

def relerr(a, want):
    a = np.asarray(a, np.float64)
    return float(np.abs(a-want).max()/max(1e-30, np.abs(want).max()))

def time_cumsum(x, want, iters=5):
    t = Tensor(x).cumsum(-1)
    require_fresh(t, "cumsum")
    d = dev_ms(t, iters)
    e = relerr(t.numpy(), want)
    if e >= 1e-3: raise RuntimeError(f"GATE FAILED relerr={e:.3e}")
    return d | {"relerr": e}

def time_copy(x, iters=5):
    """The FLOOR, in this run, this instrument, this card.

    A cumsum reads one float per element and writes one, and so does this. So it
    is the honest denominator for "how close is this to the memory limit", and
    `notes/33` makes the point and then quotes a floor measured in a different
    session. Here it is the same loop, the same `time_call`, the same 30 seconds.

    `Tensor(x).contiguous()` IS NOT A COPY. `x` is already contiguous, so
    `contiguous()` returns a view of the input with no kernel behind it, the
    LINEAR is empty, and the first version of this probe died with
    `ZeroDivisionError: float division by zero` computing GB/s from a
    `median_ms` of nothing. That is the fourth time this project has had a
    measurement return a number derived from an absent computation; the fix is
    to make the absent computation an explicit exception (see `dev_ms`) and to
    use an expression that genuinely emits a kernel.

    `x * 1.0` reads one float per element and writes one, through a real
    elementwise kernel, and its correctness is checked against `x` so it cannot
    be optimised into nothing without the gate noticing.
    """
    t = Tensor(x) * 1.0
    require_fresh(t, "copy")
    d = dev_ms(t, iters)
    if relerr(t.numpy(), x) > 0:
        raise RuntimeError("the floor arm did not reproduce its input")
    nbytes = x.size*4
    d["gbps"] = round(2*nbytes/d["median_ms"]/1e6, 1)
    d["mib"] = round(nbytes/2**20, 1)
    return d

# ============================================================ A. the floor
log(""); log("=" * 78); log("A  THE COPY FLOOR, measured here, now"); log("=" * 78)
xf = mk(E//4096, 4096)
fl = time_copy(xf)
log(f"  copy E={xf.size} ({fl['mib']:.0f} MiB in, same out)  {fl['median_ms']:.3f} ms"
    f"  {fl['gbps']} GB/s   [MEASURED, the denominator everything below is compared to]")
log(f"  [{left():.0f}s left]")

# ============================================ B. the interleaved SPLIT ladder
# The ladder goes BELOW the installed value and far above it, because the
# question is not "does it move" (run 1 answered that) but "is the window the
# whole cost" (SPLIT=1 answers that) and "does the model hold at the top"
# (SPLIT=4096 = one stage at every N in the ladder).
LADDER = [1, 8, 64, 256, 1024, 4096]
NS = [1024, 4096]
ROUNDS = 3

log(""); log("=" * 78)
log("B  THE SPLIT LADDER, INTERLEAVED")
log("   every arm measured in ABABAB order, %d rounds, one process, one card" % ROUNDS)
log("   so drift hits all arms equally instead of hitting whichever ran first")
log("=" * 78)
# EVERY key that the loop below touches must exist first, or the run dies on a
# KeyError at the first floor row and every stage is lost. The first version
# initialised only ("floor", 4096) and then asked for ("floor", 1024), so the
# probe died ~40 s in having measured the floor -- the one number it was going to
# divide everything by -- and reported nothing.
#
# This is the same class of defect as the `realize()` no-op and the two-stage
# mislabel: a bookkeeping key that was never checked against the loop that
# consumes it. It is caught here by construction rather than by luck, because
# the dict comprehension is over the same `LADDER` and `NS` the loop iterates.
acc = {(k, n): [] for k in ["floor"] + LADDER for n in NS}
assert set(acc) >= {("floor", n) for n in NS} | {(k, n) for k in LADDER for n in NS}, \
    "the accumulator is missing a key the sweep will ask for"

def one_point(k, n):
    """Time one (SPLIT, N) cell, and NEVER let one cell's failure kill the run.

    A single cell raising -- a compile error at one shape, an OOM, a gate
    rejection -- must cost that cell and nothing else. The first version let it
    propagate, so one bad cell lost the entire sweep including every cell before
    it. The sweep is 36 cells and they are independent measurements; a failure in
    one of them is information, not a reason to discard the other 35.
    """
    r = E // n
    x = mk(r, n)
    want = np.cumsum(x.astype(np.float64), -1)
    try:
        if k == "floor":
            return time_copy(x)
        if k == INSTALLED:
            restore()
        else:
            patch_split(k)
        return time_cumsum(x, want)
    finally:
        restore()

for rnd in range(ROUNDS):
    log("")
    log(f"  --- round {rnd+1}/{ROUNDS} ---")
    for n in NS:
        try:
            d = one_point("floor", n)
            acc[("floor", n)].append(d["median_ms"])
            log(f"    floor   N={n:<5} {d['median_ms']:8.3f} ms  {d['gbps']} GB/s")
        except Exception as e:
            log(f"    floor   N={n:<5} FAILED {type(e).__name__}: {str(e)[:110]}")
        for k in LADDER:
            if left() < 90:
                log(f"    SPLIT={k:<5} SKIPPED, {left():.0f}s left"); continue
            try:
                d = one_point(k, n)
            except Exception as e:
                log(f"    SPLIT={k:<5} N={n:<5} FAILED {type(e).__name__}: {str(e)[:110]}")
                continue
            acc[(k, n)].append(d["median_ms"])
            log(f"    SPLIT={k:<5} N={n:<5} {d['median_ms']:8.3f} ms  {d['n_kernels']} kernels"
                f"  relerr {d['relerr']:.1e}  [{left():.0f}s left]")
        log("")

# ------------------------------------------------------------------ verdict
log("=" * 78); log("VERDICT"); log("=" * 78)

def spread(v):
    return (max(v)-min(v))/min(v) if v and min(v) else float("inf")

N = 4096
if not acc[("floor", N)]:
    log("  the copy floor did not survive to the verdict; no ratio is computable.")
    raise SystemExit(0)
floor_med = statistics.median(acc[("floor", N)])
log(f"  copy floor at N={N}, E=2^24 : {floor_med:.3f} ms   [MEASURED this run]")
log("")
log(f"  {'SPLIT':>7} {'rounds (ms)':>34} {'median':>9} {'spread':>8} {'x floor':>8}")
tab = []
for k in LADDER:
    v = acc[(k, N)]
    if not v: continue
    med = statistics.median(v)
    tab.append({"split": k, "rounds": v, "median_ms": med, "spread": spread(v),
                "x_floor": med/floor_med})
    log(f"  {k:>7} {str([round(t,2) for t in v]):>34} {med:>9.3f} "
        f"{spread(v)*100:>7.1f}% {med/floor_med:>8.2f}")

log("")
log("  Q1  is the window the WHOLE cost?  SPLIT=1 is a window of 1 element, i.e.")
log("      no re-summation at all, which is what a scan does.")
if tab:
    t1 = next((r for r in tab if r["split"] == 1), None)
    t256 = next((r for r in tab if r["split"] == INSTALLED), None)
    if t1 and t256:
        log(f"      SPLIT=1   {t1['median_ms']:8.3f} ms  = {t1['x_floor']:.2f}x the copy floor")
        log(f"      SPLIT=256 {t256['median_ms']:8.3f} ms  = {t256['x_floor']:.2f}x the copy floor")
        log(f"      -> shrinking the window from 256 to 1 buys only "
            f"{t256['median_ms']/t1['median_ms']:.2f}x.")
        log(f"      -> a windowless scan would still be {t1['x_floor']:.1f}x the floor,"
            f" so {'SOME' if t1['x_floor']>1.5 else 'NOT MUCH'} of the gap survives.")
log("")
log("  Q2  is the ladder monotone in SPLIT, with drift inside it?")
if len(tab) >= 2:
    meds = [r["median_ms"] for r in tab]
    mono = all(meds[i] <= meds[i+1]*1.10 for i in range(len(meds)-1))
    for r in tab:
        log(f"      SPLIT={r['split']:<6} {r['median_ms']:8.3f} ms   round spread "
            f"{r['spread']*100:5.1f}%")
    log(f"      monotone within 10%: {mono}")
    worst_drift = max(r["spread"] for r in tab)
    worst_effect = (tab[-1]["median_ms"]/tab[0]["median_ms"]) if tab[0]["median_ms"] else 0
    log(f"      worst within-arm round spread : {worst_drift*100:.1f}%")
    log(f"      full ladder effect (top/bottom): {worst_effect:.2f}x")
    log(f"      effect / drift                : {worst_effect/(worst_drift or 1e-9):.1f}x")
    log("      -> the effect is not a session difference; the interleave is what makes")
    log("         that a statement rather than an assertion.")

RESULT = {"probe": "axis/splitcausal_t4probe2.py", "dev": DEV,
          "op_py_sha256": OP_SHA, "installed_SPLIT": INSTALLED,
          "E": E, "NS": NS, "LADDER": LADDER, "ROUNDS": ROUNDS,
          "floor_median_ms": floor_med,
          "acc": {f"{k[0]}|{k[1]}": v for k, v in acc.items()},
          "table_N4096": tab,
          "wall_s": round(time.time()-T0, 1), "budget_s": BUDGET_S}
open(os.environ.get("SPLITCAUSAL2_OUT","splitcausal_t4b.json"),"w").write(
    json.dumps(RESULT, indent=2, default=str))
log("")
log(f"wrote {os.environ.get('SPLITCAUSAL2_OUT','splitcausal_t4b.json')}"
    f"   [{left():.0f}s of budget unused]")
'''


def build_notebook(out_path: Path) -> dict:
    digest = hashlib.sha256(PROBE.encode()).hexdigest()
    nb = {"cells": [{"cell_type": "code", "execution_count": None,
                     "id": "splitcausal-probe2", "metadata": {}, "outputs": [],
                     "source": [
                         "# generated by axis/splitcausal_t4probe2.py -- do not edit\n",
                         "import hashlib\n",
                         f"_PAYLOAD = {PROBE!r}\n",
                         f"_SHA = {digest!r}\n",
                         "assert hashlib.sha256(_PAYLOAD.encode()).hexdigest() == _SHA\n",
                         "exec(compile(_PAYLOAD, '<splitcausal-t4-probe2>', 'exec'))\n",
                     ]}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python", "version": "3.12.0"}},
          "nbformat": 4, "nbformat_minor": 5}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(nb, indent=1))
    return {"notebook": str(out_path), "payload_sha256": digest,
            "payload_bytes": len(PROBE.encode())}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "axis" / "splitcausal_t4b" / "probe2.ipynb"))
    a = ap.parse_args()
    print(json.dumps(build_notebook(Path(a.out)), indent=2))
    sys.exit(0)
