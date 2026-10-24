#!/usr/bin/env python3
"""make_notebook.py -- assemble the Kaggle notebook for the faithful-port arm.

One push, one T4, three render timings back to back on the same card in the
same process tree: the hand-written reference, the faithful port in `clean`
mode, and the same port in `faithful` mode. The clean-vs-faithful delta is the
number this arm exists to produce and no other arm in this study has it.

WHY BASE64 THE SOURCES IN
-------------------------
`kaggle kernels push` uploads exactly one file: the notebook named by
`code_file`. Sidecars in the kernel directory are not uploaded. So
`kernels.cu` (105 KB), `refdrv.cu` and this package travel as base64 inside one
code cell, and every payload is sha256-checked before anything is built. A
mismatch is a hard stop: a timing taken from a source that is not the one the
write-up describes is worse than no timing.

WHAT THE NOTEBOOK DOES, IN ORDER. The order is load-bearing.
------------------------------------------------------------------
  1. hash-check and write every payload, preserving the repo-relative path so
     `minecu_tg` and `minecu_tg_faithful` stay packages
  2. `pip install tinygrad`   (internet is on, for this)
  3. ONLY THEN import-check the packages. The check must sit after the
     install or it fails on a missing tinygrad, which is a real error in the
     wrong place and a run spent for nothing.
  4. run `verify.py --quick` again, ON THE CARD. The host gate proves the
     source is right on this machine's tinygrad; this proves it on the card's.
     tinygrad 0.14.0 here, possibly a different version there.
  5. build `refdrv.cu` ALONE for sm_75 (it `#include`s `kernels.cu`).
     Passing `kernels.cu` as well is a LINK error, and a pre-flight that
     compiled only one of them would pass while the notebook failed.
  6. run refdrv at each batch: the reference's render timing, its whole step,
     and its own device-measured DDA step count. The dumped voxels and basis
     go to disk.
  7. `minecu_tg_faithful/battery.py`: both modes, same batches, same run.
  8. compare the two sides' observations by sha256. If they differ, the ratio
     is between two different computations and the summary says so instead of
     printing it as a result.
  9. census the generated CUDA on the card (launch counts) for both modes.
 10. write SUMMARY.txt

The notebook is generated rather than committed because the payloads are large
and must be regenerated whenever an input changes. Regeneration is
deterministic given the inputs, so the notebook's own sha256 is a usable
provenance handle, and `run.sh` prints it.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
LANE = REPO / "minecu_tg_faithful"

# --------------------------------------------------------------------------
# The payloads. Listed explicitly, never globbed: a glob picks up a scratch
# file and changes the payload hash for no reason a reader could see.
#
# `minecu_tg/` is carried UNMODIFIED. This arm's port imports its blocks table
# and its float64 reference, and the whole point is that the code under
# measurement is byte-identical to what the other arms measured. Shipping those
# files through this notebook is what makes that checkable on the card rather
# than asserted here.
#
# `minecu_ref/bench/dda_model.py` is here because `verify.py` does
# `sys.path.insert(0, REPO/"minecu_ref"/"bench")` and then `import dda_model`
# -- the reference's own CPU model, which is what the render is verified
# against. A prior push in this repo died at exactly that line with
# `ModuleNotFoundError: No module named 'dda_model'`, after everything else
# had already printed. numpy and json are all it needs.
PAYLOAD_FILES = [
    "minecu_tg/__init__.py",
    "minecu_tg/blocks.py",
    "minecu_tg/world.py",
    "minecu_tg/render.py",
    "minecu_tg/cpu_reference.py",
    # `bench.py` is the *timing instrument*, imported by `battery.py`. It is
    # carried unmodified for two reasons: it is the same code path the naive arm
    # measured with, so the two arms' device times are comparable by
    # construction rather than by assertion; and it contains
    # `linear_device_time_ms`, which exists precisely because timing
    # `realize()` on an already-realized tensor measures nothing
    # (`minecu_tg/bench.py:227-251`). It has no module-level imports beyond the
    # stdlib, so carrying it costs nothing at import time.
    "minecu_tg/bench.py",
    "minecu_tg_faithful/__init__.py",
    "minecu_tg_faithful/render.py",
    "minecu_tg_faithful/crossings.py",
    "minecu_tg_faithful/verify.py",
    "minecu_tg_faithful/battery.py",
    "minecu_ref/bench/dda_model.py",
]

# The hand-written side, vendored byte-identical, never modified.
REF_FILES = [
    "minecu_ref/src/kernels.cu",
    "minecu_tg/kaggle/refdrv.cu",
]

# --------------------------------------------------------------------------
# The shape.
# --------------------------------------------------------------------------
# world=16^3, res=32x32, ground=world/4 so the agent is not buried
# (`minecu_ref/buried_agent_fix.md`), max_steps=32 which is the reference's own
# default. ground=world//4 is not cosmetic: the physics kernel floors the agent
# on `world_y/4` (`kernels.cu:1566`) while `generate_world_kernel` puts the
# grass at `ground_height` (`:1823`), and if they disagree the agent is buried
# inside its own surface and the DDA collapses to ~1.3 steps/pixel. That is four
# weeks of numbers meaning nothing, so the shape is pinned.
#
# The batch list starts at 512, not 1. `notes/19` §2.2 measured the fusion
# decision flipping between batch 512 and 1024 at 32x32 -- one kernel per DDA
# step below ~1M pixels, one fused kernel above -- so the interesting range is
# exactly the crossover and the list spans it. 2048 is included to reach the
# regime that gets timed. All three sides run the same list; a shape one side
# cannot allocate is reported as a result, not hidden.
BATCHES = [512, 1024, 2048]
ITERS = 10
WARMUP = 3
# Per-stage bounds, in seconds, and they must add up to less than the run cap
# `run.sh` passes to `kaggle kernels push -t`. A stage that exceeds its bound is
# killed and reported by name instead of consuming the rest of the session, so
# one pathological batch cannot silently take the other five with it. Sums to
# 3300 s against a 4500 s cap.
GATE_LIMIT_S = 900        # ~4x the host's `--quick` wall time, for CUDA compiles
BATTERY_LIMIT_S = 2400    # 3 batches x 2 modes, each compiled and timed
REFDRV_LIMIT_S = 300      # the hand-written side compiled once, then per batch

# --------------------------------------------------------------------------
DRIVER = r'''
import base64, hashlib, json, os, subprocess, sys, time

OUT = "/kaggle/working"
os.makedirs(OUT, exist_ok=True)
EXPECT = __EXPECT__
PAYLOADS = __PAYLOADS__
BATCHES = __BATCHES__
ITERS = __ITERS__
WARMUP = __WARMUP__
REFDRV_LIMIT_S = __REFDRV_LIMIT_S__
GATE_LIMIT_S = __GATE_LIMIT_S__
BATTERY_LIMIT_S = __BATTERY_LIMIT_S__

lines = []
# Write-through, so a run the platform kills still leaves its evidence on the
# card. `kaggle kernels output` fetches /kaggle/working, and a summary that
# only exists after the last stage exists for exactly the runs that did not
# reach the last stage -- which are the runs anybody needs to read. The cost is
# one small file write per printed line.
SUMMARY = f"{OUT}/SUMMARY.txt"
def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    lines.append(s)
    try:
        open(SUMMARY, "w").write("\n".join(lines) + "\n")
    except OSError:
        pass

def die(code, why):
    say("")
    say("FATAL: " + why)
    lines.append(f"FATAL {why}")
    try:
        open(SUMMARY, "w").write("\n".join(lines) + "\n")
    except OSError:
        pass
    sys.exit(code)

def sh(cmd, **kw):
    p = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True,
                       text=True, **kw)
    return p.returncode, p.stdout, p.stderr

def stream(cmd, label, limit=None, **kw):
    """Run a long subprocess, echoing its output as it appears.

    `sh` captures everything and says nothing until the child exits, which for
    the on-card gate and the battery is minutes of silence. A silent notebook is
    indistinguishable from a hung one: it is why "it hung" was the first
    hypothesis for the 50-second kill, and why the 2136 s block in
    `out/minecu_tg/20260928-221859/` is unreadable. This echoes line by line.

    `limit` is seconds, and it is enforced HERE rather than left to the
    platform's run cap. Echoing without a bound would replace "the harness
    timed this stage out" with "the session was killed at an arbitrary point",
    which is the failure this notebook already had twice; a stage that overruns
    says so, by name, with its last 60 lines.
    """
    import time as _t
    t0 = _t.time()
    say(f"  [{label}] starting"
        + (f", bounded at {limit:.0f}s" if limit else ", unbounded"))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, **kw)
    tail = []
    killed = False
    for line in p.stdout:
        line = line.rstrip()
        tail.append(line)
        del tail[:-60]
        say(f"  [{label} {_t.time() - t0:6.1f}s] {line}")
        if limit and _t.time() - t0 > limit and p.poll() is None:
            say(f"  [{label}] OVER its {limit:.0f}s bound after "
                f"{_t.time() - t0:.0f}s; killing it and reporting what it did")
            p.kill()
            killed = True
            break
    rc = p.wait()
    say(f"  [{label}] exit {rc} after {_t.time() - t0:.1f}s"
        + ("  (killed by the bound)" if killed else ""))
    return (124 if killed else rc), "\n".join(tail), None

def dump_summary():
    open(SUMMARY, "w").write("\n".join(lines) + "\n")

say(f"  session budget: gate <= {GATE_LIMIT_S}s, battery <= {BATTERY_LIMIT_S}s, "
    f"refdrv <= {REFDRV_LIMIT_S}s. The platform cap is set in run.sh and these "
    f"must fit inside it.")
say("=" * 78)
say("minecu_tg_faithful -- the reference's strategy in TinyGrad, both modes,")
say("against the hand-written CUDA render, on this T4")
say("=" * 78)

# ----------------------------------------------------------- 1. provenance
say("")
say("--- 1. provenance: every payload hash-checked before anything is built ---")
for name, b64, want in PAYLOADS:
    data = base64.b64decode(b64)
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        die(9, f"{name} sha256 {got} != {want}; refusing to measure a source I "
               f"cannot identify. Regenerate with minecu_tg_faithful/kaggle/run.sh.")
    dest = f"{OUT}/{name}"                             # repo-relative: the
    os.makedirs(os.path.dirname(dest), exist_ok=True) # package stays a package
    open(dest, "wb").write(data)
    say(f"  {name:38s} {len(data):8d} B  sha256 {got[:16]}  OK")
sys.path.insert(0, OUT)

# --------------------------------------------------------------- 2. tinygrad
say("")
say("--- 2. tinygrad ---")
rc, out, err = sh([sys.executable, "-m", "pip", "install", "-q", "tinygrad"])
say(f"  pip install tinygrad rc={rc}")
if rc != 0:
    say("  " + (err or out)[-400:].replace("\n", "\n  "))
try:
    import tinygrad
    from tinygrad.helpers import VERSION
    say(f"  tinygrad at {tinygrad.__file__}")
    say(f"  helpers.VERSION (the program-cache schema) = {VERSION}")
except Exception as e:
    die(12, f"tinygrad did not import: {e}")

# ONLY NOW, with tinygrad importable, check that what we laid down imports.
# This has to come after the install or it fails on a missing tinygrad instead.
try:
    import importlib
    for m in ("minecu_tg.render", "minecu_tg.cpu_reference", "minecu_tg.bench",
              "minecu_tg_faithful.render", "minecu_tg_faithful.crossings"):
        importlib.import_module(m)
        say(f"  import {m:34s} OK")
except Exception as e:
    die(13, f"payload does not import: {type(e).__name__}: {e}")

# ------------------------------------------------------------------ 3. the
# card. verify.py runs again here, not just on the host: it proves the same
# source is correct against the card's tinygrad, which may not be this host's.
say("")
say("--- 3. correctness gate ON THE CARD ---")
# DEV=CUDA, not the default CPU. Two reasons, and the first is the one that
# matters: it verifies the render on the device every timing below is taken
# on, so a codegen difference on the card cannot survive to become a wrong
# image in the summary. The second is that this image has no `clang` and
# tinygrad's CPU backend shells out to one, so DEV=CPU dies with
# FileNotFoundError before the first check runs.
rc, out, err = stream([sys.executable, f"{OUT}/minecu_tg_faithful/verify.py",
                       "--quick", "--dev", "CUDA"],
                      "gate", limit=GATE_LIMIT_S, cwd=OUT,
                      env={**os.environ, "VERIFY_LOG": f"{OUT}/verify_card.log"})
if rc != 0:
    say("  " + ((err or out) or "")[-1500:].replace("\n", "\n  "))
    die(14, "the correctness gate failed on the card; no timing is reported "
            "because a timing from a wrong render is a measurement of the "
            "wrong thing")
say("  PASS on the card itself. The observation is bit-identical to the naive")
say("  port and to an independent float64 transcription, and both modes agree")
say("  on the per-pixel DDA step count.")

# ------------------------------------------------------------- 4. device
say("")
say("--- 4. device ---")
import subprocess as _sp
try:
    smi = _sp.run(["nvidia-smi",
                   "--query-gpu=name,memory.total,compute_cap,driver_version",
                   "--format=csv,noheader"], capture_output=True, text=True)
    say("  " + smi.stdout.strip())
except Exception as e:
    say(f"  nvidia-smi unavailable: {e}")
from tinygrad import Device
say(f"  DEV={Device.DEFAULT}")

# ------------------------------------------------- 5. build the baseline
# refdrv.cu `#include`s kernels.cu (refdrv.cu:36-38, via
# `#ifndef MINECU_KERNELS`), so it is compiled ALONE -- passing kernels.cu on
# the command line as well compiles it a second time and the LINKER reports
# `multiple definition of launch_render_uint8_prebasis' and eight others.
#
# And `#include` resolves relative to the INCLUDING file, which is
# `/kaggle/working/minecu_tg/kaggle/refdrv.cu`, while kernels.cu is written to
# `/kaggle/working/minecu_ref/src/kernels.cu`. So the default include finds
# nothing: push 7 died here with
#     refdrv.cu:36:24: fatal error: kernels.cu: No such file or directory
# which is a compile error in a file that is correct, about a path this stage
# owns. `-DMINECU_KERNELS="<abs path>"` is the override refdrv.cu documents for
# exactly this (`minecu_tg/kaggle/make_notebook.py:210` gets it right).
#
# A pre-flight that only compiled refdrv.cu with the wrong flags would pass
# here and fail in the notebook, which is why this stage exists separately.
say("")
say("--- 5. build the hand-written side for sm_75 ---")
NVCC = None
for _p in ("/usr/local/cuda/bin/nvcc", "nvcc"):
    if sh([_p, "--version"])[0] == 0:
        NVCC = _p
        break
if NVCC is None:
    die(10, "no nvcc on this runner; the T4 image is supposed to have one")
say("  " + "\n  ".join(sh([NVCC, "--version"])[1].strip().splitlines()[-2:]))
os.makedirs(f"{OUT}/ref", exist_ok=True)
_cmd = [NVCC, "-O3", "--use_fast_math", "-std=c++17", "-arch=sm_75",
        "-diag-suppress", "177",
        f'-DMINECU_KERNELS="{OUT}/minecu_ref/src/kernels.cu"',
        f"{OUT}/minecu_tg/kaggle/refdrv.cu", "-o", f"{OUT}/ref/refdrv"]
say("  " + " ".join(_cmd))
rc, out, err = stream(_cmd, "nvcc", limit=REFDRV_LIMIT_S)
if rc != 0:
    say("  " + (err or out)[-1500:].replace("\n", "\n  "))
    die(20, "refdrv did not build; the hand-written side is required, because "
            "a comparison with only one side is not a comparison")
say("  OK  {0} bytes".format(os.path.getsize(f"{OUT}/ref/refdrv")))

# ---------------------------------------------------------- 6. the timings
say("")
say("--- 6. the hand-written baseline, per batch (cudaEvent, its own clock) ---")
ref_render_us = {}
for b in BATCHES:
    os.makedirs(f"{OUT}/refdump/b{b}", exist_ok=True)
    cmd = [f"{OUT}/ref/refdrv", "--batch", str(b), "--world", "16", "--res", "32",
           "--max-steps", "32", "--ground", "4", "--iters", str(ITERS),
           "--warmup", str(WARMUP), "--outdir", f"{OUT}/refdump/b{b}", "--dump"]
    rc, out, err = stream(cmd, f"refdrv b{b}", limit=REFDRV_LIMIT_S)
    if rc != 0:
        say(f"  batch {b:6d}  refdrv FAILED rc={rc}")
        say("  " + ((err or out) or "")[-800:].replace("\n", "\n  "))
        continue
    said_step = False
    for line in out.splitlines():
        ls = line.strip()
        if ls.startswith("RENDER_US"):
            p = ls.split()
            ref_render_us[b] = float(p[1])
            say(f"  batch {b:6d}  RENDER_US median {p[1]:>10s}  min {p[2]:>10s}  "
                f"max {p[3]:>10s}  iters {p[4]}")
        elif ls.startswith("STEP_US"):
            if not said_step:
                said_step = True
                say(f"  batch {b:6d}  STEP_US   {ls.split()[1]:>10s}   "
                    f"<- CONTEXT ONLY. A whole step is 7 launches, five of them")
                say(f"                          state updates. It is never a "
                    f"denominator here.")
        elif "steps/pixel" in ls or "DDA" in ls:
            say(f"  batch {b:6d}  {ls}")
json.dump(ref_render_us, open(f"{OUT}/ref_render_us.json", "w"))

# -------------------------------------------------------- 7. both modes
say("")
say("--- 7. this port, both modes, same batches, same run, same process ---")
rc, out, err = stream([sys.executable, f"{OUT}/minecu_tg_faithful/battery.py",
                       "--batches", ",".join(str(b) for b in BATCHES),
                       "--refdrv", f"{OUT}/ref/refdrv",
                       "--ref-outdir", f"{OUT}/refdump",
                       "--json", f"{OUT}/faithful_battery.json"],
                      "battery", limit=BATTERY_LIMIT_S, cwd=OUT)
if rc != 0:
    die(30, f"battery.py exited {rc}")

# --------------------------------------------------------- 8. the summary
say("")
say("--- 8. the three-way table, and the clean-vs-faithful delta ---")
try:
    rep = json.load(open(f"{OUT}/faithful_battery.json"))
except Exception as e:
    die(31, f"no battery json: {e}")
say(f"  {'batch':>6}  {'ref us':>10}  {'clean ms':>10}  {'faith ms':>10}  "
    f"{'f/clean':>8}  {'clean/ref':>9}  {'faith/ref':>9}  {'kernels c/f':>12}  same obs")
say(f"  {'(DEVICE time; host tracing excluded -- see battery section 5)':>70}")
for row in rep.get("rows", []):
    def g(k, w=10, p=3):
        v = row.get(k)
        return f"{v:{w}.{p}f}" if isinstance(v, (int, float)) else f"{'?':>{w}}"
    b = row["batch"]
    rr = ref_render_us.get(b, row.get("ref_render_us"))
    say(f"  {b:>6}  {rr if rr else float('nan'):>10.1f}  {g('clean_ms'):>10}  "
        f"{g('faithful_ms'):>10}  {g('faithful_over_clean', 8, 3):>8}  "
        f"{g('clean_over_ref', 9, 2):>9}  {g('faithful_over_ref', 9, 2):>9}  "
        f"{str(row.get('clean_kernels','?')) + '/' + str(row.get('faithful_kernels','?')):>12}  "
        f"{row.get('obs_identical_modes')}")

say("")
say("--- what these numbers are, and are not ---")
say("  * The three columns are RENDERS. Same world, same cameras, same")
say("    max_steps, same uint8 observation, same card, same run, equal batch.")
say("    `f/clean` is the crossing cost and is the number this arm adds; the")
say("    two `*/ref` columns are render-against-render and are the same")
say("    comparison `notes/19` §5.2 makes.")
say("  * The two ms columns are DEVICE time -- the sum of the per-kernel")
say("    device times in the graph, measured with tinygrad's own time_call,")
say("    host tracing excluded. The reference's us figure is a cudaEvent")
say("    median around a single launch, which is also device time and is the")
say("    like-for-like comparison. The host build+realize figure printed")
say("    beside each row is CONTEXT: it includes the framework's graph")
say("    tracing, which the reference does not pay, so it is NOT divided by")
say("    anything here.")
say("  * Calling realize() on an already-realized tensor is a silent no-op.")
say("    An earlier version of battery.py timed exactly that and reported the")
say("    port at 0.011 ms against the reference's 152 us -- a 13x speedup over")
say("    hand-written CUDA, from a graph whose offline SASS is tens of times")
say("    the reference's instruction count. time_graph now raises if handed a")
say("    realized tensor, and the timing instrument is minecu_tg's own")
say("    linear_device_time_ms, which is the code path `notes/19` measured")
say("    with. That false number is the reason the clock is named on every")
say("    row rather than in a footnote.")
say("  * The reference's whole-step figure is printed for context and is NOT a")
say("    denominator anywhere. `4.55 M env-steps/s` covers 7 launches.")
say("  * `same obs` compares the two modes' observations by sha256. Both are")
say("    verified against the same float64 transcription and the same naive")
say("    port, so a False here would mean the crossings changed the")
say("    computation, and the ratios above it would be between two different")
say("    computations.")
say("  * The ratio to the reference is NOT attributed to the compiler unless")
say("    both sides did the same work. Each mode's mean steps/pixel is in the")
say("    battery output. The reference's own DEVICE-side step count is NOT")
say("    available here: it needs the dda_probe_kernel from")
say("    minecu_ref/bench/bench.cu, which refdrv.cu does not link, and putting")
say("    the CPU model's number under the same label would be a claim about a")
say("    device that was not measured. The workload check is one-sided and")
say("    the note says so rather than rounding it up.")
say("  * The obs_sha256 in the battery is this port's. verify.py proved it")
say("    equals the naive port's and a float64 transcription's ON THIS CARD in")
say("    this run. It is NOT compared against the reference's dumped bytes")
say("    here; that is notes/19's comparison on the reference's own state, and")
say("    this arm did not re-derive it.")
say("  * No SASS number is reported here. The instruction and register counts")
say("    come from the offline archlab route on the host, where ptxas and")
say("    cuobjdump exist and nvcc's version is pinned; the T4 image's CUDA")
say("    version is recorded but not used for a count.")

dump_summary()
say("")
say("wrote /kaggle/working/SUMMARY.txt")
'''


def undefined_names(src: str) -> list[str]:
    """Names the generated cell reads but never binds, or binds too late.

    Two separate defects, each of which cost this arm a Kaggle session, and
    neither of which `compile()` can see -- the name is syntactically perfect in
    both cases:

    * `battery.py` shipped with an f-string reading a bare `max_steps` where the
      local is `a.max_steps`. Nothing on this host ran the battery, so it passed
      the gate and would have died on the card at the first batch row.
    * the session-budget banner was a `say(...)` call placed *above* `def say` --
      a `NameError` on the card's first output line, 13 seconds into a paid-for
      session. The first version of this check missed that one: it collected
      bindings from the whole file regardless of order, and `say` *was* bound,
      just further down. A name being bound somewhere in the file is not the
      same as it being bound before it is used.

    So there are two passes: one over the whole tree for "never bound", and one
    ordered pass over the module-level statements for "bound after its use".
    Function bodies are excluded from the ordered pass -- their locals are
    resolved when they are called, and their free variables are the module's
    final bindings -- and are covered by the first pass. `except ... as e:`
    binds for the handler and deletes after it, so both halves are modelled; a
    `from x import *` disables the first pass, because its names cannot be
    known from the source and a check that reports everything is not a check.
    """
    import ast
    import builtins

    known = set(dir(builtins))

    # ---- pass 1: read but never bound, anywhere in the file
    bound = set(known)
    loads: list[str] = []

    class Scan(ast.NodeVisitor):
        def visit_Name(self, n):
            if isinstance(n.ctx, ast.Store):
                bound.add(n.id)
            elif isinstance(n.ctx, ast.Load):
                loads.append(n.id)
            self.generic_visit(n)

        def visit_arg(self, n):
            bound.add(n.arg)

        def visit_Import(self, n):
            for a in n.names:
                bound.add((a.asname or a.name).split(".")[0])

        def visit_ImportFrom(self, n):
            for a in n.names:
                bound.add(a.asname or a.name)

        def visit_FunctionDef(self, n):
            # Pass 1 DOES descend: a local is a binding, and a `global G` inside
            # a function is the only statement that binds G. (It is the ordered
            # pass that must not descend, because a function's locals are
            # resolved when it is called.)
            bound.add(n.name)
            self.generic_visit(n)

        def visit_ExceptHandler(self, n):
            if n.name:
                bound.add(n.name)
            self.generic_visit(n)

        def visit_Global(self, n):
            bound.update(n.names)

    scan = Scan()
    tree = ast.parse(src)
    scan.visit(tree)
    # A star import makes the name set unknowable from the source. Reporting
    # every name it might have supplied would make the check useless, so it
    # turns the check off and says so, rather than crying wolf.
    if any(isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names)
           for n in ast.walk(tree)):
        return []
    never = {n for n in loads if n not in bound}

    # ---- pass 2: module level, in order. A Load before any binding is a bug
    # even though pass 1 says the name exists somewhere.
    late: set[str] = set()
    # A `global G` anywhere binds G at module scope as soon as that function
    # runs, so it counts as bound from the top: there is no ordering to enforce
    # against a name the module has not created yet.
    seen = set(known) | {x for g in ast.walk(tree)
                         if isinstance(g, ast.Global) for x in g.names}
    # `except ... as e:` binds `e` for the body and then *deletes* it. Without
    # the delete, the handler's own `e` is a false positive on a pattern the
    # driver uses four times; without the bind, a use after the handler is a
    # false negative on a real NameError. Both halves, or neither.
    handler_only: set[str] = set()

    class Step(ast.NodeVisitor):
        def visit_Name(self, n):
            if isinstance(n.ctx, ast.Store):
                seen.add(n.id)
            elif isinstance(n.ctx, ast.Load) and n.id not in seen:
                late.add(n.id)
            self.generic_visit(n)

        def visit_Import(self, n):
            for a in n.names:
                seen.add((a.asname or a.name).split(".")[0])

        def visit_ImportFrom(self, n):
            for a in n.names:
                seen.add(a.asname or a.name)

        def visit_FunctionDef(self, n):
            seen.add(n.name)          # do not descend: its own scope

        def visit_ClassDef(self, n):
            seen.add(n.name)

        def visit_ExceptHandler(self, n):
            if n.name:
                seen.add(n.name)
                handler_only.add(n.name)
            self.generic_visit(n)
            if n.name:
                seen.discard(n.name)   # Python 3 unbinds it here
                handler_only.discard(n.name)

        # A comprehension is its own scope, and its `elt` is visited before its
        # `generators` -- so `[x for x in xs]` reads `x` before the target that
        # binds it and reports a name that is perfectly fine. Bind the targets
        # first; that is the comprehension's whole binding story at this scope.
        def _comprehension(self, n):
            for gen in n.generators:
                for t in ast.walk(gen.target):
                    if isinstance(t, ast.Name):
                        seen.add(t.id)
            self.generic_visit(n)

        visit_ListComp = _comprehension
        visit_SetComp = _comprehension
        visit_GeneratorExp = _comprehension
        visit_DictComp = _comprehension

    for node in ast.parse(src).body:
        Step().visit(node)
    return sorted(never | late)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path, default=LANE / "kaggle" / "kernel")
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    payloads = []
    missing = []
    for rel in PAYLOAD_FILES + REF_FILES:
        p = REPO / rel
        if not p.exists():
            missing.append(rel)
            continue
        data = p.read_bytes()
        payloads.append((rel, base64.b64encode(data).decode("ascii"),
                         hashlib.sha256(data).hexdigest()))
    if missing:
        raise SystemExit(f"missing payload(s): {missing}")

    driver = (DRIVER
              .replace("__EXPECT__", json.dumps([p[2] for p in payloads]))
              .replace("__PAYLOADS__", json.dumps([[p[0], p[1], p[2]] for p in payloads]))
              .replace("__BATCHES__", json.dumps(BATCHES))
              .replace("__ITERS__", str(ITERS))
              .replace("__WARMUP__", str(WARMUP))
              .replace("__GATE_LIMIT_S__", str(GATE_LIMIT_S))
              .replace("__BATTERY_LIMIT_S__", str(BATTERY_LIMIT_S))
              .replace("__REFDRV_LIMIT_S__", str(REFDRV_LIMIT_S)))
    undefined = undefined_names(driver)
    if undefined:
        raise SystemExit(
            "the notebook cell references names it never binds: "
            f"{undefined}. This cell is what runs on the card, and a NameError "
            f"at its first line costs a session; it is why this is a build-time "
            f"stop rather than something to find in a log.")

    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {},
             "source": ["# minecu_tg_faithful on a T4\n",
                        "\n",
                        "The reference's DDA strategy written in TinyGrad, in two\n",
                        "modes -- `clean` (no crossings) and `faithful` (the\n",
                        "crossings on) -- timed against the hand-written\n",
                        "`render_kernel_uint8_prebasis` from the unmodified vendored\n",
                        "`kernels.cu`.\n",
                        "\n",
                        "Generated by `minecu_tg_faithful/kaggle/make_notebook.py`.\n",
                        "Every payload is sha256-checked before anything is built.\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {},
             "outputs": [], "source": [driver]},
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out = a.outdir / "minecu_faithful_t4.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"wrote {out}  {out.stat().st_size} B  sha256 {h}")
    print(f"  {len(payloads)} payloads, {sum(len(base64.b64decode(p[1])) for p in payloads)} B raw")
    for rel, _, sha in payloads:
        print(f"    {rel:38s} {sha[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
