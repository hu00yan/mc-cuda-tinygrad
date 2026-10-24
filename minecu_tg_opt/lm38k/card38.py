#!/usr/bin/env python3
"""card38.py -- the on-card experiment for `notes/38`, as ONE script kernel.

Run on Kaggle's free T4 via `minecu_tg_opt/lm38k/push38.py`.

WHY A SCRIPT AND NOT A NOTEBOOK
-------------------------------
nbclient raises on the first cell error, and in `notes/36` a single `TypeError`
in a summary line destroyed four already-measured batches. A script has no
cell boundary to fail at, and every stage here is wrapped independently and
every configuration is appended to a JSONL file and flushed the moment it is
measured. A stage that raises costs that stage and nothing else, and anything
measured before the session dies still comes home through `kernels output`.

THE ORDER IS THE ARGUMENT'S PRIORITY
------------------------------------
Truncation must cost the least valuable stage, so:

    0  prelude      payload hashes, tinygrad, device gate, ptxas gate
    1  the lever    does `-maxrregcount` move anything on THIS image's PTX?
                    (the offline finding says only with `.maxntid` dropped)
    2  the rungs    all five re-measured, with BOTH spill currencies read off
                    the binaries that ran, so the five-point correlation is
                    re-derived here and not quoted
    3  THE DISCRIMINATOR   one source string, nine register budgets, graph
                    size held equal by the source's sha256
    4  NOLOCALS     the framework's own register lever, as a control
    5  the reference  hand-written CUDA, built, opcode-gated, timed

The reference is LAST on purpose. It is the denominator and the byte-gate, and
it is the one stage whose absence does not invalidate the discriminator: a
register budget cannot change what a kernel computes, and each cell is gated
against the stock rung's own bytes on the card, which is the correctness check
that actually bears on this experiment. If the reference stage never runs, the
note says so and quotes no ratio.
"""

import base64
import gc
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

OUT = "/kaggle/working"
T0 = time.time()
#: Hard wall-clock budget. The session cap is passed to `kernels push -t`; this
#: is the script's own deadline, checked between stages, so a slow stage runs
#: out of room in a controlled place instead of being killed mid-write.
BUDGET_S = float(os.environ.get("LM38_BUDGET_S", "2400"))
REPEATS = int(os.environ.get("LM38_REPEATS", "3"))
ITERS = int(os.environ.get("LM38_ITERS", "5"))
BUDGETS = [None, 200, 168, 128, 96, 64, 48, 32, 24]
WORLD, GROUND, RES, MAXSTEPS, FOV, BATCH = 16, 4, 32, 32, 70.0, 1024

os.makedirs(OUT, exist_ok=True)
ROWS = Path(OUT) / "rows.jsonl"
FACTS: dict = {}


def say(*a):
    print(*a, flush=True)


def elapsed():
    return time.time() - T0


def left():
    return BUDGET_S - elapsed()


def num(v, width=0, dash="-"):
    """Format a number that may be None, WITHOUT raising.

    `f"{None:>6}"` is a TypeError, and one of those in a summary line killed
    four already-measured batches in `notes/36` -- the exact shape of failure
    this script exists to avoid, and it happened here first, on the card, 85
    seconds into a 48-minute session. Nothing in this file formats a
    possibly-absent number directly.
    """
    if v is None:
        return str(dash).rjust(width) if width else str(dash)
    return f"{v:,}".rjust(width) if width else f"{v:,}"


def banner(t):
    say("")
    say("=" * 78)
    say(f"  {t}    [t+{elapsed():.0f}s, {left():.0f}s left]")
    say("=" * 78)


def flush(obj, name):
    """Append a record and rewrite the sidecar. Called after EVERY cell."""
    with open(ROWS, "a") as f:
        f.write(json.dumps(obj, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())
    FACTS[name] = obj
    (Path(OUT) / "facts.json").write_text(json.dumps(FACTS, indent=1, default=str))


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def emit_summary():
    """A human-readable table, rebuilt from the JSONL every time."""
    lines = []
    if not ROWS.exists():
        return "\n".join(lines)
    for ln in ROWS.read_text().splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        ms = r.get("median_ms")
        lines.append(
            f"{str(r.get('stage','')):<11} {str(r.get('rung','')):<14} "
            f"{str(r.get('budget','')):<7} "
            f"ms={('-' if ms is None else f'{ms:.4f}'):<9} "
            f"regs={str(r.get('registers','-')):<5} "
            f"frame={num(r.get('stack_frame'), 7)} "
            f"spill={num(r.get('spill_total'), 7)} "
            f"sass={num(r.get('sass_instructions'), 7)} "
            f"calls={str(r.get('n_calls','-')):<4} "
            f"gate={str(r.get('gate_bad','-')):<5} "
            f"ok={str(r.get('intervention_happened', r.get('pass','-'))):<5} "
            f"src={str(r.get('source_sha256',''))[:10]}")
    (Path(OUT) / "SUMMARY.txt").write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


def guard(name, fn, *a, **kw):
    """Run a stage; a raise costs that stage and nothing else."""
    try:
        return fn(*a, **kw)
    except Exception:
        say(f"\n*** STAGE {name} FAILED ***")
        say(traceback.format_exc()[-1800:])
        flush({"stage": name, "failed": True,
               "traceback": traceback.format_exc()[-2000:]}, f"stage_{name}")
        return None


# ===========================================================================
banner("STAGE 0 -- prelude, payload, device")
# ===========================================================================
PAYLOADS = json.loads(base64.b64decode(os.environ["LM38_PAYLOADS"]))
for name, b64, want in PAYLOADS["payloads"]:
    data = base64.b64decode(b64)
    got = hashlib.sha256(data).hexdigest()
    assert got == want, f"payload hash mismatch {name}: {got} != {want}"
    dst = Path(OUT) / "payload" / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
say(f"  {len(PAYLOADS['payloads'])} payloads, all sha256-verified, written")
# The payload tree is rooted at OUT/payload, so THAT is what goes on sys.path.
# Putting OUT on it instead gives `No module named 'minecu_tg'` 9 seconds into a
# 45-minute session, which is what happened the first time this was pushed.
PAYLOAD_ROOT = f"{OUT}/payload"
sys.path.insert(0, PAYLOAD_ROOT)
os.chdir(OUT)
if OUT in sys.path:
    sys.path.remove(OUT)
say("")
say("  -- the payload tree, as written --")
for p in sorted(Path(PAYLOAD_ROOT).rglob("*")):
    if p.is_file():
        say(f"    {p.relative_to(PAYLOAD_ROOT).as_posix():<40} {p.stat().st_size:>9,} B")
say("")

say("")
say("  -- pip tinygrad --")
r = sh([sys.executable, "-m", "pip", "install", "-q", "tinygrad"], timeout=600)
say(f"  pip rc={r.returncode} {(r.stderr or '')[-160:]}")

say("")
say("  -- the device, from the device --")
n = sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader"])
say(f"  nvidia-smi: {n.stdout.strip() or n.stderr.strip()[:150]}")
if "Tesla T4" not in n.stdout:
    say("  *** NOT A T4. Every number in notes/35/36 is a T4 number. Stopping.")
    raise SystemExit("wrong device")
import ctypes
lib = ctypes.CDLL("libcuda.so.1")
lib.cuInit(0)                      # without this every attribute read returns rc=3
for attr, nm in ((97, "MaxSharedMemPerBlockOptin"), (75, "CCmajor"),
                 (76, "CCminor"), (16, "SMcount"), (8, "MaxSharedMemPerBlock")):
    v = ctypes.c_int(0)
    rc = lib.cuDeviceGetAttribute(ctypes.byref(v), attr, 0)
    say(f"  cuDeviceGetAttribute({attr:>3}) {nm:<28} "
        f"{v.value if rc == 0 else f'rc={rc}'}")
    FACTS.setdefault("device", {})[nm] = v.value if rc == 0 else f"rc={rc}"
p = sh(["ptxas", "--version"])
say(f"  {p.stdout.strip().splitlines()[0] if p.stdout.strip() else 'NO PTXAS'}")
if p.returncode != 0:
    say("  *** ptxas is not on PATH; the lever does not exist on this image.")
    raise SystemExit("no ptxas")
FACTS["device"]["ptxas"] = p.stdout.strip().splitlines()[0]
for exe in ("cuobjdump", "nvdisasm", "nvcc"):
    q = sh(["which", exe])
    say(f"  {exe:<10} {q.stdout.strip() or 'NOT FOUND'}")
    FACTS["device"][exe] = q.stdout.strip() or None

from tinygrad import Device, Tensor, dtypes                     # noqa: E402
from minecu_tg.bench import linear_device_time_ms, require_fresh_graph  # noqa: E402
from minecu_tg.render import camera_basis                       # noqa: E402
from minecu_tg_opt.render import render_variant, VARIANTS      # noqa: E402
from minecu_tg_opt import lm38_falsify as F                     # noqa: E402
say(f"  Device.DEFAULT={Device.DEFAULT} arch={Device[Device.DEFAULT].arch}")
say(f"  lm38_falsify from {F.__file__}")
# An import from the wrong tree is the failure that costs a whole session, so
# every module is checked against the payload directory just written. Looked up
# through `sys.modules`, NOT through the package attribute: `minecu_tg_opt`'s
# __init__ does `from .render import render`, so `minecu_tg_opt.render` is the
# FUNCTION and `.render` the module -- exactly the shadowing `minecu_tg`'s own
# docstring warns about, and an attribute lookup here raises AttributeError
# nine seconds into a 45-minute session.
for _n in ("minecu_tg", "minecu_tg.bench", "minecu_tg.render",
           "minecu_tg_opt", "minecu_tg_opt.render", "minecu_tg_opt.lm38_falsify"):
    _f = getattr(sys.modules.get(_n), "__file__", None)
    if _f and PAYLOAD_ROOT in _f:
        say(f"  {_n:<30} {_f}")
    else:
        say(f"  *** {_n} resolved to {_f!r}, NOT under {PAYLOAD_ROOT}")
        raise SystemExit(f"{_n} did not come from the payload tree")

BANK = F.CompilerBank(Device[Device.DEFAULT])
REC = F.ProgramRecorder()
REC.__enter__()
STOCK_SRC = F.source_facts  # noqa: F841


# ===========================================================================
banner("STAGE 0b -- WHERE does this build compile through?")
# ===========================================================================
# Two sessions established that the register-budget intervention works on
# `NULL` in this exact harness and silently does nothing on a real T4. Before
# spending another twenty minutes of nvrtc guessing which difference matters,
# log every compile a trivial graph causes and read the route off it.
say("")
_probe = F.install_route_probe()
try:
    _census = F.route_census(Device.DEFAULT)
except Exception as e:
    _census = [{"route": f"CENSUS FAILED: {type(e).__name__}: {e}"[:200]}]
# The probe logs every compile. `F.COMPILE_LOG` is now non-empty from the census
# and is never cleared, so the per-cell counts below are cumulative totals --
# which is enough to tell "a compile happened" from "no compile happened",
# and that is the only question the probe is here to answer.
CENSUS_AT = len(F.COMPILE_LOG)
say(f"  the probe has logged {CENSUS_AT} compile(s) so far; it is never cleared")
say(f"  {len(_census)} compile(s) from a 4-element graph:")
for _e in _census[:12]:
    say("    " + json.dumps(_e, default=str))
_routes = sorted({e.get("route") for e in _census})
_classes = sorted({e.get("cls") for e in _census if e.get("cls")})
say(f"  routes: {_routes}")
say(f"  compiler classes: {_classes}")
flush({"routes": _routes, "classes": _classes, "census": _census},
      "route_census")
if "compile_cached" not in _routes:
    say("")
    say("  *** this build does NOT compile through Compiler.compile_cached.")
    say("  *** The bank below intercepts that method and would measure nothing.")
    say(f"  *** observed instead: {_routes} / {_classes}. The note records this")
    say("  *** and no register-allocation number from this session is quoted.")
else:
    say("  compile_cached IS on the path; the bank will intercept it.")

# ===========================================================================
banner("STAGE 1 -- the workloads, and the stock allocation of each")
# ===========================================================================
import numpy as np                                        # noqa: E402
from minecu_tg import generate_world                       # noqa: E402

_one = generate_world(WORLD, GROUND).numpy()
VOX = Tensor(np.ascontiguousarray(np.stack([_one] * BATCH)), dtype=dtypes.int8)
_cams = np.array([[WORLD / 2.0, float(GROUND), WORLD / 2.0, 0.001 * (i % 97 - 48), 0.0]
                  for i in range(BATCH)], dtype=np.float32)
BASIS = camera_basis(Tensor(_cams, dtype=dtypes.float32), FOV, RES, RES).contiguous()
BASIS.realize()
Device[Device.DEFAULT].synchronize()
say(f"  batch {BATCH}: {BATCH * RES * RES:,} pixels, world {WORLD}^3, "
    f"max_steps {MAXSTEPS}")


def build(rung):
    return render_variant(VOX, BASIS, RES, RES, MAXSTEPS, rung)


#: The stock allocation of each rung, captured with the SAME compiler the
#: intervention uses (`.maxntid` KEPT, no budget). This is the control that
#: makes the intervention interpretable: if `drop_maxntid=True, budget=None`
#: reproduces the stock numbers, then the top of the budget ladder is the real
#: thing and the ladder below it is a register-allocation sweep on it.
STOCK_ALLOC: dict = {}
for rung in sorted(VARIANTS):
    def _one_stock(rung=rung):
        F.clear_program_cache()
        comp = BANK.install(maxrregcount=None, drop_maxntid=False,
                            tag=f"stock_{rung}")
        try:
            g = build(rung)
            require_fresh_graph(g, f"{rung} stock graph")
            g.realize()
            return comp, g
        finally:
            BANK.restore()
    got = guard(f"stock_alloc_{rung}", _one_stock)
    if not got:
        continue
    comp, g = got
    # The route probe is independent of the bank, so it answers the question
    # the bank cannot: was a compile attempted at all during this cell, and
    # with which compiler? A cell where the bank saw nothing but the probe saw
    # something is a bank bug; a cell where the probe saw nothing either is a
    # "no compile happened", which is a different problem entirely.
    _log = list(F.COMPILE_LOG)
    say(f"  [{rung}] probe logged {len(_log)} compile(s): "
        + ", ".join(sorted({str(e.get('route')) for e in _log})))
    for _e in _log[-8:]:
        say(f"      {json.dumps(_e, default=str)}")
    if comp.calls == 0:
        say(f"  *** {rung}: the bank saw no compile but the probe saw "
            f"{len(_log)}. Recording and continuing -- this cell's allocation "
            f"is NOT used.")
    rec = {"stage": "stock_alloc", "rung": rung, "tag": f"stock_{rung}",
           "ptxas_calls": comp.calls, "probe_log": _log,
           "n_sources": len(comp.sources),
           "n_compile_cached_seen": sum(1 for e in _log
                                        if e.get("route") == "compile_cached")}
    flush(rec, f"stock_{rung}")
    if comp.calls == 0:
        del g
        gc.collect()
        continue
    hot = max(comp.sources, key=lambda s: len(comp.sources[s])) if comp.sources else ""
    f = comp.facts.get(hot, {})
    sf = F.source_facts(comp.sources[hot]) if hot else {}
    names = set(sf.get("kernel_names", []))
    drv = REC.facts_for(names) if names else []
    hot_drv = next((d for d in drv if d.get("kernel_name") == f.get("_hot")), None)
    STOCK_ALLOC[rung] = {
        "source_sha256": sf.get("source_sha256"),
        "source_statements": sf.get("source_statements"),
        "launch_bounds": sf.get("launch_bounds"),
        "ptxas": {k: v for k, v in f.items()
                  if k in ("stack_frame", "spill_stores", "spill_loads",
                           "spill_total", "registers", "sass_instructions",
                           "cubin_bytes", "_hot", "maxntid_in")},
        "driver": hot_drv,
        "n_driver_programs": len(drv),
    }
    say(f"  {rung:<13} src {str(sf.get('source_sha256'))[:12]} "
        f"stmts {num(sf.get('source_statements'), 6)} "
        f"lb {sf.get('launch_bounds')} "
        f"frame {num(f.get('stack_frame'), 6)} "
        f"spill {num(f.get('spill_total'), 6)} "
        f"regs {num(f.get('registers'), 5)} "
        f"sass {num(f.get('sass_instructions'), 6)} "
        f"drv {hot_drv.get('driver_regs') if hot_drv else None}/"
        f"{hot_drv.get('driver_local_bytes') if hot_drv else None} "
        f"ptxas_calls {comp.calls}")
    del g
    gc.collect()
flush(STOCK_ALLOC, "stock_alloc")
say("")
say("  --- emitted SUMMARY so far ---")
say(emit_summary())


# ===========================================================================
banner("STAGE 2 -- THE DISCRIMINATOR: one source string, many register budgets")
# ===========================================================================
# THE CASE. `flat_gather` at batch 1024. The generated CUDA is the framework's
# own, byte for byte: its sha256 is the same in every cell, so the UOp count,
# the fusion decision, the launch count and the instruction stream the compiler
# sees are the same string's properties and are the same. Only the register
# budget differs, and only after `.maxntid` -- an upper bound ptxas places on
# itself, from `__launch_bounds__(128)` in the source -- is replaced by the
# explicit budget.
#
# `RUNG_A` is the stock allocation of the same source with `.maxntid` KEPT and
# no budget. It is the control: it says what the framework's own compilation of
# this source does, on this card, so the sweep can be read as a departure from
# a real point rather than as a set of stranger's binaries.
RUNG_A = os.environ.get("LM38_RUNG", "flat_gather")

GATE_REF: dict = {}


def gate_and_time(rung, budget, drop_maxntid, tag, nolocals=False, stage="disc"):
    """One configuration, end to end, in the order that makes it honest.

        1. clear the program cache (the compiler is not in the key)
        2. build a FRESH graph and refuse it if realize() would be a no-op
        3. realize and compare the bytes against the stock rung's own bytes
        4. only then time, with linear_device_time_ms
        5. collect the allocation of the binary that just ran, by two routes
    """
    import gc as _gc
    n0 = len(REC.programs)
    F.clear_program_cache()
    rec = {"stage": stage, "rung": rung, "budget": budget,
           "drop_maxntid": drop_maxntid, "nolocals": nolocals, "tag": tag}
    with F.Nolocals(nolocals):
        comp = BANK.install(maxrregcount=budget, drop_maxntid=drop_maxntid,
                            tag=tag)
        try:
            # ---- 3. the gate, before any clock is read ----
            t0 = time.time()
            g = build(rung)
            require_fresh_graph(g, f"{rung}/{tag} gate graph")
            img = g.numpy()
            a = np.ascontiguousarray(img).reshape(len(img), -1, 3)
            rec["gate_pixels"] = int(a.shape[0] * a.shape[1])
            if rung == RUNG_A and budget is None and not drop_maxntid:
                GATE_REF["img"] = a.copy()
                rec["gate_role"] = "the reference bytes for this rung"
                rec["gate_bad"] = 0
            elif RUNG_A in GATE_REF:
                b = GATE_REF["img"]
                bad = int((a.astype(np.int32) != b.astype(np.int32)).any(axis=2).sum())
                rec["gate_bad"] = bad
                rec["gate_pixels_differing"] = bad
            else:
                rec["gate_bad"] = None      # no reference yet; see STAGE 3
                rec["gate_role"] = "no reference yet"
            rec["gate_sha256"] = hashlib.sha256(a.tobytes()).hexdigest()
            rec["gate_s"] = round(time.time() - t0, 1)
            del g, img, a
            _gc.collect()

            # ---- 4. the timing ----
            times = []
            for _ in range(REPEATS):
                t = build(rung)
                require_fresh_graph(t, f"{rung}/{tag} timed graph")
                r = linear_device_time_ms(t, iters=ITERS) or {}
                if r.get("error"):
                    rec.setdefault("errors", []).append(r["error"])
                else:
                    times.append(r["median_ms"])
                del t
                _gc.collect()
            rec["times_ms"] = times
            rec["median_ms"] = statistics.median(times) if times else None
            rec["n_calls"] = r.get("n_kernels")
            rec["spread_pct"] = (100.0 * (max(times) - min(times)) / min(times)
                                  if len(times) > 1 else 0.0)

            # ---- 5. what the binary that just ran was ----
            rec["ptxas_calls"] = comp.calls
            rec["nvrtc_calls"] = comp.nvrtc_calls
            hot = max(comp.sources, key=lambda s: len(comp.sources[s])) \
                if comp.sources else ""
            f = comp.facts.get(hot, {})
            sf = F.source_facts(comp.sources[hot]) if hot else {}
            rec["source_sha256"] = sf.get("source_sha256")
            rec["source_statements"] = sf.get("source_statements")
            rec["launch_bounds"] = sf.get("launch_bounds")
            for k in ("stack_frame", "spill_stores", "spill_loads", "spill_total",
                      "registers", "sass_instructions", "cubin_bytes",
                      "cubin_sha256", "_hot", "maxntid_in", "maxntid_out",
                      "ptxas_cmd", "ptxas_v"):
                if k in f:
                    rec[k] = f[k]
            rec["sass_by_function"] = f.get("sass_by_function")
            names = set(sf.get("kernel_names", []))
            drv = REC.facts_for(names) if names else []
            hd = next((d for d in drv if d.get("kernel_name") == f.get("_hot")), None)
            rec["driver"] = hd
            rec["n_driver_programs"] = len(drv)
        finally:
            BANK.restore()
    # A cell whose compiler was never called measured nothing. RECORD that and
    # carry on rather than aborting: an aborted session returns no data at all,
    # and a row that says "no compile happened" is a real result about the
    # instrument even though it is not a result about the frame. What is
    # forbidden is letting a cell with no compilation report an allocation.
    _log = list(F.COMPILE_LOG)
    rec["probe_compile_cached_calls"] = sum(
        1 for e in _log if e.get("route") == "compile_cached")
    if comp.calls == 0:
        rec["intervention_happened"] = False
        rec["why"] = (f"bank saw no compile; the independent route probe saw "
                      f"{rec['probe_compile_cached_calls']}")
        say(f"  {tag}: NO COMPILE (bank={comp.calls}, "
            f"probe={rec['probe_compile_cached_calls']}) -- not a measurement")
    else:
        rec["intervention_happened"] = True
    rec["pass"] = bool(times) and rec.get("gate_bad") in (0, None) \
        and rec.get("ptxas_calls", 0) > 0
    flush(rec, tag)
    d = rec.get("driver") or {}
    say(f"  {tag:<22} ms {str(rec.get('median_ms') and round(rec['median_ms'], 4)):<9} "
        f"regs {num(rec.get('registers'), 5)} frame {num(rec.get('stack_frame'), 7)} "
        f"spill {num(rec.get('spill_total'), 7)} "
        f"sass {num(rec.get('sass_instructions'), 7)} "
        f"calls {num(rec.get('n_calls'), 4)} gate_bad {rec.get('gate_bad')!s:<5} "
        f"src {str(rec.get('source_sha256'))[:10]} "
        f"drv {d.get('driver_regs')}/{d.get('driver_local_bytes')} "
        f"ptxas_calls {rec.get('ptxas_calls')}")
    return rec


say(f"  the case: rung {RUNG_A}, batch {BATCH}, {BATCH*RES*RES:,} pixels")
say("")
say("  the control: the framework's own compilation of this source")
guard("disc_stock", gate_and_time, RUNG_A, None, False, f"stock_{RUNG_A}")
say("")
say("  the sweep: identical source, explicit register budget")
for b in BUDGETS[1:]:
    if left() < 240:
        say(f"  stopping the sweep: {left():.0f}s left")
        break
    guard(f"disc_{b}", gate_and_time, RUNG_A, b, True, f"budget_{b}")
say("")
say("  --- emitted SUMMARY so far ---")
say(emit_summary())


# ===========================================================================
banner("STAGE 3 -- the five rungs re-measured here, so the correlation is mine")
# ===========================================================================
for rung in sorted(VARIANTS):
    if rung == RUNG_A:
        continue                      # already timed above, at stock
    if left() < 200:
        say(f"  stopping: {left():.0f}s left")
        break
    guard(f"rung_{rung}", gate_and_time, rung, None, True, f"rung_{rung}",
          stage="rung")
say("")
say("  --- emitted SUMMARY so far ---")
say(emit_summary())


# ===========================================================================
banner("STAGE 4 -- NOLOCALS, the framework's own register lever")
# ===========================================================================
for rung in (RUNG_A,):
    if left() < 150:
        break
    guard("nolocals", gate_and_time, rung, None, False, f"nolocals_{rung}",
          nolocals=True, stage="nolocals")
say(emit_summary())


# ===========================================================================
banner("STAGE 5 -- the hand-written reference, built, gated, timed")
# ===========================================================================
def _reference():
    refcu = Path(OUT) / "payload" / "minecu_tg" / "kaggle" / "refdrv.cu"
    kdir = Path(OUT) / "payload" / "minecu_ref" / "src"
    F_ = "-O3 -arch=sm_75 -std=c++17 --use_fast_math"
    t0 = time.time()
    r = sh(["nvcc", *F_.split(), "-o", f"{OUT}/refdrv",
            "-DMINECU_KERNELS=\"" + str(kdir / "kernels.cu") + "\"", str(refcu)],
           timeout=1500)
    if r.returncode != 0:
        return {"built": False, "nvcc_stderr": (r.stderr or "")[-800:]}
    built = {"built": True, "nvcc_s": round(time.time() - t0, 1)}
    d = Path(OUT) / "refstate"
    d.mkdir(parents=True, exist_ok=True)
    r2 = sh([f"{OUT}/refdrv", "--batch", str(BATCH), "--world", str(WORLD),
             "--res", str(RES), "--max-steps", str(MAXSTEPS), "--iters", "5",
             "--warmup", "2", "--dump", "--outdir", str(d)], timeout=1500)
    (Path(OUT) / "log_refdrv.txt").write_text((r2.stdout or "") + (r2.stderr or ""))
    if r2.returncode != 0:
        return {**built, "ran": False, "log": (r2.stdout or "")[-600:]}
    m = re.search(r"RENDER_US\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)", r2.stdout)
    card = re.search(r"card\s+(.*)", r2.stdout)
    built.update({
        "ran": True, "render_us": float(m.group(1)) if m else None,
        "render_us_min": float(m.group(2)) if m else None,
        "render_us_n": int(m.group(4)) if m else None,
        "card": card.group(1).strip() if card else None,
        "dump_dir": str(d)})
    # opcode gate: an sm_80+ instruction means the "T4 number" is about nothing
    s = sh(["cuobjdump", "-sass", f"{OUT}/refdrv"])
    banned = ["HMMA", "HGMMA", "LDGSTS", "REDUX", "ELECT", "BAR.RED", "F2FP"]
    hits = {b: len(re.findall(r"\b" + b + r"\b", s.stdout or "")) for b in banned}
    built["sm80_opcodes"] = {k: v for k, v in hits.items() if v}
    return built


ref = guard("reference", _reference)
if ref:
    flush(ref, "reference")
    say(f"  reference: {json.dumps({k: v for k, v in ref.items() if k != 'dump_dir'}, default=str)[:600]}")
else:
    say("  the reference stage did not complete. No ratio is quoted in notes/38.")

banner("DONE")
say(emit_summary())
say("")
say(f"wall {elapsed():.0f}s of {BUDGET_S:.0f}s budget")
flush({"done": True, "wall_s": round(elapsed(), 1),
       "budget_s": BUDGET_S, "repeats": REPEATS, "iters": ITERS,
       "rung_a": RUNG_A, "budgets": BUDGETS,
       "config": {k: FACTS.get("device", {}).get(k) for k in
                  ("nvidia", "ptxas", "CCmajor", "CCminor", "SMcount")}},
      "run_end")
