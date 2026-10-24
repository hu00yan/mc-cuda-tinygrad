#!/usr/bin/env python3
"""make_notebook.py -- assemble the Kaggle notebook for the good-vs-naive run.

WHY BASE64 THE SOURCES IN. `kaggle kernels push` uploads exactly one file: the
notebook named by `code_file`. Sidecars in the kernel directory are not
uploaded, so `kernels.cu` (105 KB), `refdrv.cu`, `bench.cu` and both packages
travel as base64 inside the notebook, and every payload is sha256-checked
before anything is built. A hash mismatch is a hard stop, not a warning: a
timing taken from a source that is not the one the write-up describes is worse
than no timing.

WHY FIVE CELLS AND NOT ONE. nbclient enforces a 1800 s timeout PER CELL on
this image. An earlier push was a single cell that had measured the hand-written
side's timings at 39 s and was still inside the two TinyGrad ports' timing stage
when the cap fired, so those measurements went home with the failure. Each
stage is its own cell, ordered so the cheap decisive stages come first, and
`SUMMARY.txt` is flushed at the end of every one.

WHAT THE NOTEBOOK DOES, IN ORDER
  1. cell 1: verify every payload's sha256, install tinygrad, prove the packages
     import from where they were written, compile the vendored `kernels.cu` with
     `refdrv.cu` and with `bench.cu` for sm_75, gate on no sm_80+ opcode, and
     time the hand-written side per batch
  2. cell 2: time both TinyGrad ports per batch, with the allocator's peak and
     the DDA workload of each timed run
  3. cell 3: THE GATE -- the hand-written side dumps its own state, both ports
     render that state on this card, and the device's bytes are compared with
     the device's bytes
  4. cell 4: the render-versus-render table and what may be read off it

The SASS is NOT taken from this notebook. Instruction and register counts are a
property of a compiler version as much as of a source, and this image carries
CUDA 12.8 while the offline route is 12.9 on both sides; mixing them would make
the register numbers meaningless. `minecu_tg_opt/compile_sass.sh` is the route,
and it needs no card.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os

# The hand-written side, vendored unmodified, and the driver the notebook
# builds it with.
REF_FILES = [
    "minecu_ref/src/kernels.cu",
    "minecu_tg/kaggle/refdrv.cu",
    "minecu_ref/bench/bench.cu",
    "minecu_ref/bench/dda_model.py",
]

# The naive port. Carried whole, not just the two files the notebook calls:
# `minecu_tg_opt/render.py` imports `minecu_tg.render.camera_basis` from it, so
# the two ports demonstrably share the camera basis and differ in the DDA.
TG_FILES = [
    "minecu_tg/__init__.py",
    "minecu_tg/blocks.py",
    "minecu_tg/world.py",
    "minecu_tg/render.py",
    "minecu_tg/cpu_reference.py",
    "minecu_tg/verify.py",
    "minecu_tg/codegen.py",
    "minecu_tg/bench.py",
    "minecu_tg/battery.py",
    "minecu_tg/sass_matrix.py",
    # `sass_matrix.py` imports the categoriser from `tools/archlab` by a path
    # relative to itself, so the card needs it there too. Shipping the file
    # rather than copying its category table into sass_matrix.py is what keeps
    # "one categoriser, so the columns mean the same thing on both sides"
    # literally true rather than aspirational.
    "tools/archlab/inspect_ptx.py",
]

OPT_FILES = [
    "minecu_tg_opt/__init__.py",
    "minecu_tg_opt/render.py",
    "minecu_tg_opt/verify_opt.py",
    "minecu_tg_opt/capture_opt.py",
    "minecu_tg_opt/opcount.py",
    "minecu_tg_opt/census_opt.py",
    "minecu_tg_opt/battery_opt.py",
]

# The batches both sides are timed at. The hand-written side runs at all of
# them; both ports run at the first four, so the render-versus-render table has
# one row per timed batch with no interpolation anywhere in it. The largest
# batch is the identity stage's, which is a correctness check rather than a
# timing and costs one render instead of a timing loop.
BATCHES = [256, 512, 1024, 2048, 4096]
TIMING_MAX = 2048
DUMP_BATCH = 4096
ITERS = 8
WARMUP = 5
VARIANTS = "naive,lean"

PRELUDE = r'''
# Shared by every cell below. nbclient executes the cells in order in one
# kernel, so these names persist; the notebook is still four cells rather than
# one because the 1800 s timeout is PER CELL and the first single-cell push
# spent its whole budget inside the TinyGrad timing stage, taking the reference
# side's already-measured timings down with it.
import base64, hashlib, json, os, re, shutil, subprocess, sys, time

OUT = "/kaggle/working"
os.makedirs(OUT, exist_ok=True)
EXPECT = __EXPECT__
PAYLOADS = __PAYLOADS__
BATCHES = __BATCHES__
DUMP_BATCH = __DUMP_BATCH__
ITERS = __ITERS__
WARMUP = __WARMUP__
VARIANTS = __VARIANTS__

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
    p = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, **kw)
    return p.returncode, p.stdout, p.stderr

def flush_summary():
    """Write what has been logged so far. Called at the end of every cell, so a
    later cell that times out still leaves the earlier cells' output in
    SUMMARY.txt."""
    open(f"{OUT}/SUMMARY.txt", "w").write("\n".join(lines) + "\n")
'''

CELL1 = r'''
say("=" * 78)
say("minecu_tg_opt -- a competent TinyGrad port, against the hand-written CUDA")
say("=" * 78)

# ------------------------------------------------------------- provenance
say("")
say("--- provenance: every payload is hash-checked before anything is built ---")
for name, b64, want in PAYLOADS:
    data = base64.b64decode(b64)
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        die(9, f"{name} sha256 {got} != {want}; refusing to measure a source I "
               f"cannot identify. Regenerate with minecu_tg_opt/kaggle/run.sh.")
    dest = f"{OUT}/{name}"            # repo-relative, so a package stays a package
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    open(dest, "wb").write(data)
    say(f"  {name:36s} {len(data):8d} B  sha256 {got[:16]}  OK")

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

# AFTER the install, and printing the paths actually used rather than assuming
# them: the flattened-payload bug killed the sibling notebook at exactly this
# point, after the reference side had already spent its time.
try:
    import importlib
    m1 = importlib.import_module("minecu_tg.render")
    m2 = importlib.import_module("minecu_tg_opt.render")
    say(f"  minecu_tg.render      from {m1.__file__}")
    say(f"  minecu_tg_opt.render  from {m2.__file__}")
except Exception as e:
    die(11, f"a package did not import from {OUT}: {e}")

# --------------------------------------------------------- the CPU gate
# The FULL correctness gate runs on the HOST, in run.sh, before the push. It
# cannot run on this card at all, and that is a fact about the image rather than
# about the code: tinygrad's CPU backend compiles every kernel with
#
#     subprocess.check_output([getenv("CC", 'clang'), '-c', '-x', 'c', ...,
#                              '--target=x86_64-none-unknown-elf', ...])
#
# (`tinygrad/runtime/support/compiler_cpu.py:21`), so it needs a compiler that
# can target bare-metal x86-64. This image has no `clang`, and `CC=gcc` -- the
# documented override -- gets further and then fails on the same
# `--target=x86_64-none-unknown-elf` flag, because the image's gcc is a hosted
# compiler. Two pushes established that, and it is recorded here in one block
# rather than discovered by a third run.
#
# So no CPU gate is attempted here. What replaces it is STRONGER for the
# question this notebook asks: cell 3 feeds the hand-written side's OWN dumped
# state through both ports on THIS card and compares the device's bytes with the
# device's bytes.
say("")
say("--- the CPU correctness gate ---")
say("  NOT RUN ON THIS CARD, and not skipped for time: this image has no C")
say("  compiler that can target bare-metal x86-64, so tinygrad's CPU backend")
say("  cannot execute here (compiler_cpu.py:21 passes")
say("  --target=x86_64-none-unknown-elf; CC=gcc fails the same way). The full")
say("  gate ran on the host in run.sh before this push and is in the push log.")
say("  The gate that matters for this notebook is cell 3: both ports render the")
say("  hand-written side's OWN state on THIS card and are compared to its OWN")
say("  bytes.")

# ----------------------------------------------------------------- build
say("")
say("--- build the hand-written side for sm_75 ---")
# The payloads travel under their repo-relative paths, so the two hand-written
# sources land at /kaggle/working/minecu_ref/src/kernels.cu and
# /kaggle/working/minecu_tg/kaggle/refdrv.cu. The build works on flat copies at
# the top of OUT, and the copies are CHECKED: an unchecked copy is exactly how
# an earlier push of this notebook died with "cc1plus: fatal error:
# /kaggle/working/refdrv.cu: No such file or directory".
for src, dst in ((f"{OUT}/minecu_ref/src/kernels.cu", f"{OUT}/kernels.cu"),
                 (f"{OUT}/minecu_tg/kaggle/refdrv.cu", f"{OUT}/refdrv.cu"),
                 (f"{OUT}/minecu_ref/bench/bench.cu", f"{OUT}/bench.cu")):
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
    die(10, "no nvcc on this runner; the T4 image is supposed to have one")
say("  " + "\n  ".join(sh([NVCC, "--version"])[1].strip().splitlines()[-2:]))

# refdrv.cu `#include`s kernels.cu, and `#include` resolves relative to the
# *including* file, so it is compiled ALONE with the path spelled out in
# -DMINECU_KERNELS. Passing kernels.cu on the command line as well compiles it
# a second time and the link fails with "multiple definition of
# launch_render_uint8_prebasis" -- that cost the sibling notebook a run.
# bench.cu does NOT include it: it declares the extern "C" launchers and links
# against them, so it needs kernels.cu on the command line. Getting these two
# backwards is a LINK error rather than a compile error, which is why the
# distinction is written down here.
FLAGS = ["-O3", "--use_fast_math", "-std=c++17", "-arch=sm_75",
         "-diag-suppress", "177"]
for label, srcs, outbin, extra in (
    ("refdrv", [f"{OUT}/refdrv.cu"], f"{OUT}/refdrv_sm_75",
     ['-DMINECU_KERNELS="/kaggle/working/kernels.cu"']),
    ("bench", [f"{OUT}/kernels.cu", f"{OUT}/bench.cu"], f"{OUT}/bench_sm_75", []),
):
    t0 = time.time()
    rc, o, e = sh([NVCC, *FLAGS, *extra, *srcs, "-o", outbin])
    say(f"  nvcc {label:8s} rc={rc} in {time.time()-t0:.1f}s -> {os.path.basename(outbin)}")
    if rc != 0:
        for ln in (e or "").splitlines()[-8:]:
            say("  | " + ln)
        die(11, f"nvcc failed for {label}")

rc, sass, _ = sh(["/usr/local/cuda/bin/cuobjdump", "-sass", f"{OUT}/refdrv_sm_75"])
sm80 = ["HMMA", "HGMMA", "LDGSTS", "REDUX", "ELECT", "BAR.RED", "F2FP"]
say("  SASS census of the reference binary: " + "  ".join(
    f"{g}={sass.count(g)}" for g in sm80))
if any(sass.count(g) for g in sm80):
    die(13, "the reference binary contains sm_80+ instructions; it would not run "
            "on a T4 and the comparison would be meaningless")

# ------------------------------------------------- the hand-written side
say("")
say("--- the hand-written side, per batch (cudaEvent, this card, this run) ---")
ref = {}
for b in BATCHES:
    t0 = time.time()
    rc, o, e = sh([f"{OUT}/refdrv_sm_75", "--batch", str(b), "--world", "16",
                   "--res", "32", "--max-steps", "32", "--iters", str(ITERS),
                   "--warmup", str(WARMUP)])
    d = {"batch": b, "rc": rc}
    for ln in o.splitlines():
        m = re.match(r"\s*RENDER_US\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)", ln)
        if m:
            d["render_us"] = [float(m.group(i)) for i in (1, 2, 3)]
        m = re.match(r"\s*STEP_US\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)", ln)
        if m:
            d["step_us"] = [float(m.group(i)) for i in (1, 2, 3)]
    ref[b] = d
    say(f"  batch {b:6d}  render {d.get('render_us',[0])[0]:9.1f} us  "
        f"step {d.get('step_us',[0])[0]:9.1f} us  rc={rc}  ({time.time()-t0:.1f}s)")
json.dump(ref, open(f"{OUT}/ref_timing.json", "w"), indent=2)
flush_summary()
say("")
say("cell 1 done: the hand-written side's timings are in ref_timing.json")
'''

CELL2 = r'''
# Cell 2. The two TinyGrad ports, on this card, at the same batches.
#
# The naive port is `minecu_tg/render.py` verbatim. The `lean` port is the same
# computation with the index arithmetic hoisted -- same masked loop, same
# prologue, same colour table, no knob, no custom kernel, no hand-unrolling.
# The difference between the two columns in cell 4 is the cost of the PORT, not
# the cost of the compiler, and separating them is the entire point of this lane.
#
# WHY THE BATCHES STOP AT 2048. Each (batch, variant) is a fresh graph and a
# fresh NVRTC compile of a kernel that grows to a megabyte of CUDA. 256..2048
# is the range the earlier naive-port run covered, so the two runs are directly
# comparable, and the two ports at two batch sizes on either side of the
# fusion crossover is what the question needs.
say("")
say("=" * 78)
say("  the two TinyGrad ports, same card, same run, same world, max_steps 32")
say("=" * 78)
TIMING_BATCHES = [b for b in BATCHES if b <= __TIMING_MAX__]
say(f"  timing batches {TIMING_BATCHES}; the identity stage uses batch {DUMP_BATCH}")
t0 = time.time()
rc, o, e = sh([sys.executable, f"{OUT}/minecu_tg_opt/battery_opt.py",
               "--out", f"{OUT}/tgopt", "--world", "16", "--res", "32",
               "--max-steps", "32", "--variants", VARIANTS,
               "--time-batches", ",".join(str(b) for b in TIMING_BATCHES),
               "--iters", str(ITERS)], timeout=1600)
say(f"  battery_opt.py rc={rc} in {time.time()-t0:.1f}s")
say(o)
if rc not in (0, 2):
    say(f"  battery_opt.py rc={rc}")
    for ln in (e or "").splitlines()[-15:]:
        say("  " + ln)
flush_summary()
say("")
say("cell 2 done: timings in tgopt/report.json")
'''

CELL3 = r'''
# Cell 3. THE GATE. Both ports against the hand-written side's OWN bytes, on
# the hand-written side's OWN state, on this card.
say("")
say(f"--- the hand-written side dumps its settled state (batch {DUMP_BATCH}) ---")
os.makedirs(f"{OUT}/refstate", exist_ok=True)
rc, o, e = sh([f"{OUT}/refdrv_sm_75", "--batch", str(DUMP_BATCH), "--world", "16",
               "--res", "32", "--max-steps", "32", "--iters", str(ITERS),
               "--warmup", str(WARMUP), "--dump", "--outdir", f"{OUT}/refstate"])
if rc != 0:
    die(14, f"refdrv --dump failed rc={rc}: {(e or '')[-300:]}")
open(f"{OUT}/log_refdump.txt", "w").write(o)

import numpy as np
RS = f"{OUT}/refstate"
def load(name, dtype):
    return np.fromfile(f"{RS}/{name}.bin", dtype=dtype)
try:
    vox = load("voxels", np.int8).reshape(DUMP_BATCH, 16, 16, 16)
    basis = load("basis13", np.float32).reshape(DUMP_BATCH, 13)
    ref_obs = load("obs_u8", np.uint8).reshape(DUMP_BATCH, 32, 32, 3)
    say(f"  voxels {vox.shape}  basis {basis.shape}  obs {ref_obs.shape}")
except Exception as e:
    die(15, f"could not read the reference dump: {e}")

from tinygrad import Tensor, dtypes
from minecu_tg.render import render as render_naive
from minecu_tg_opt.render import render_variant

t0 = time.time()
voxT = Tensor(vox, dtype=dtypes.int8)
basisT = Tensor(basis, dtype=dtypes.float32)
npx = DUMP_BATCH * 32 * 32
naive_obs = render_naive(voxT, basisT, 32, 32, 32).numpy()
say(f"  the naive port rendered {DUMP_BATCH} envs in {time.time()-t0:.1f}s")
equiv = {"batch": DUMP_BATCH, "pixels": npx, "rows": []}
h_ref = hashlib.sha256(np.ascontiguousarray(ref_obs)).hexdigest()
h_naive = hashlib.sha256(np.ascontiguousarray(naive_obs)).hexdigest()
say(f"  reference   obs sha256 {h_ref[:32]}")
say(f"  naive port  obs sha256 {h_naive[:32]}")
equiv["ref_sha256"] = h_ref
equiv["naive_sha256"] = h_naive
# The HARD gate. The hand-written side dumped its own observation; both ports
# rendered that same state on this card. If they disagree with the hand-written
# bytes, every ratio in cell 4 is a ratio between two different computations,
# and stopping here is the whole point of having this stage at all.
for name, img in [("naive", naive_obs)] + \
        [(v.strip(), render_variant(voxT, basisT, 32, 32, 32, v.strip()).numpy())
         for v in VARIANTS.split(",")]:
    d = (img.astype(np.int32) != ref_obs.astype(np.int32))
    bad = int(d.any(axis=3).sum())
    h = hashlib.sha256(np.ascontiguousarray(img)).hexdigest()
    equiv["rows"].append({"variant": name, "pixels_differing": bad,
                          "identical": bool(np.array_equal(img, ref_obs)),
                          "sha256": h})
    say(f"  {name:<13} pixels differing {bad:>9} / {npx:>9}  "
        f"({100.0*bad/npx:.4f}%)  sha256 {h[:32]}")
json.dump(equiv, open(f"{OUT}/obs_equivalence.json", "w"), indent=2)
flush_summary()
if not all(r["identical"] for r in equiv["rows"]):
    die(16, "a port's observation on this card does not match the hand-written "
            "side's own bytes. Every ratio would be a ratio between two "
            "different computations, so there is nothing to report.")
say("  BOTH PORTS ARE BYTE-IDENTICAL to the hand-written side, on this card,")
say("  on the state the hand-written side itself produced. The comparison in")
say("  cell 4 is therefore between two implementations of one computation.")
say("")
say("cell 3 done: the gate passed")
'''

CELL4 = r'''
# Cell 4. The table, and what may be read off it.
say("")
say("=" * 78)
say("SUMMARY")
say("=" * 78)

say("")
say("RENDER VERSUS RENDER, equal batch, same card, same run, same world, "
    "max_steps 32, uint8 out")
say("  ref render / ref step are cudaEvent medians in MICROSECONDS.")
say("  tg is the sum of per-kernel device times for the whole generated graph,")
say("  in MILLISECONDS. The two ratio columns convert to a common unit first,")
say("  and read as: how many times SLOWER the generated side is. 1.0 would be")
say("  parity. Every row is render against render: the same computation, the")
say("  same output, the same dtype, the same card. Nothing here divides a")
say("  per-step number by a per-op number.")
say("")
say(f"  {'batch':>7} {'ref render us':>14} {'ref step us':>13} "
    f"{'naive ms':>10} {'lean ms':>10} {'naive/ref':>10} {'lean/ref':>9} "
    f"{'kernels n/l':>14}")
table = []
try:
    rep = json.load(open(f"{OUT}/tgopt/report.json"))
    for b in BATCHES:
        r = ref.get(b, {})
        rr = r.get("render_us", [None])[0]
        rs = r.get("step_us", [None])[0] if r.get("step_us") else None
        row = {v["variant"]: v for v in rep.get("timing", []) if v.get("batch") == b}
        nv, lnv = row.get("naive", {}), row.get("lean", {})
        dm_n = (nv.get("device_time") or {}).get("median_ms")
        dm_l = (lnv.get("device_time") or {}).get("median_ms")
        kn = (nv.get("device_time") or {}).get("n_kernels")
        kl = (lnv.get("device_time") or {}).get("n_kernels")
        # THE UNITS. `ref_us` is MICROSECONDS (cudaEvent, printed by refdrv) and
        # `dm_*` is MILLISECONDS (tinygrad's device_time). Dividing one by the
        # other is a mixed-unit quotient: it reports 6.43 where the real
        # slowdown is 155x, and it lands near 1.0 exactly when the two sides
        # are EQUAL -- the one value a reader cannot mistake for an error. Both
        # are converted to the same unit first, and the ratio is tg/ref, "how
        # many times slower the generated side is", so larger is a bigger gap.
        def ratio(tg_ms, ref_us):
            if not (tg_ms and ref_us):
                return None
            return (tg_ms * 1000.0) / ref_us

        q = ratio(dm_n, rr)
        p = ratio(dm_l, rr)
        qs = f"{q:10.2f}" if q else f"{'-':>10}"
        ps = f"{p:9.2f}" if p else f"{'-':>9}"
        say(f"  {b:>7} {round(rr,1) if rr else '-':>14} "
            f"{round(rs,1) if rs else '-':>13} "
            f"{(round(dm_n,3) if dm_n else '-'):>10} "
            f"{(round(dm_l,3) if dm_l else '-'):>10} {qs} {ps} "
            f"{f'{kn}/{kl}':>14}")
        if q and p:
            table.append((b, rr, dm_n, dm_l, q, p))
except Exception as e:
    say(f"  performance table unavailable: {e}")

say("")
say("MEMORY, the allocator's own peak")
try:
    rep = json.load(open(f"{OUT}/tgopt/report.json"))
    for v in (rep.get("timing") or []):
        if v.get("batch") and v.get("variant") and v.get("mem"):
            say(f"  batch {v['batch']:>6} {v['variant']:<8} peak "
                f"{v['mem']['peak_live_bytes']/2**20:9.1f} MiB  "
                f"allocs {v['mem']['n_allocs']}")
except Exception as e:
    say(f"  unavailable: {e}")

say("")
say("THE DDA WORKLOAD EACH TIMING ACTUALLY DID")
try:
    rep = json.load(open(f"{OUT}/tgopt/report.json"))
    for v in (rep.get("timing") or []):
        if v.get("dda"):
            d = v["dda"]
            say(f"  batch {v['batch']:>6} {v['variant']:<8} mean {d['mean']:6.3f}"
                f"  at the 32 cap {100*d['at_cap_frac']:5.1f}%"
                f"  first voxel {100*d['first_voxel_frac']:4.1f}%"
                f"  gate {'PASS' if v.get('gate', {}).get('pass') else 'FAIL'}")
except Exception as e:
    say(f"  unavailable: {e}")

say("")
say("IDENTITY, on the state the reference itself produced")
for r in equiv["rows"]:
    say(f"  {r['variant']:<13} {r['pixels_differing']:>9} / {npx:>9} pixels differ"
        f"   {'BYTE-IDENTICAL' if r['identical'] else 'DIFFERS'}")

if table:
    say("")
    say("THE TWO RATIOS, AND WHAT EACH IS")
    for b, rr, dm_n, dm_l, q, p in table:
        say(f"  batch {b:>5}: naive {q:6.2f}x, lean {p:6.2f}x, "
            f"and the optimization removes {100*(1 - dm_n/dm_l):5.1f}% of the "
            f"generated side's time")
    if len(table) >= 2:
        say("")
        say("  The naive column is the port that recomputes the flat voxel")
        say("  address from three float coordinates on every one of 32 steps.")
        say("  The lean column is the same computation with that arithmetic")
        say("  hoisted into a maintained address and precomputed strides, and")
        say("  with the per-axis bounds test computed once per ray. Both are")
        say("  plain TinyGrad: no custom_kernel, no tile size, no Context, no")
        say("  hand-unrolling, no codegen knob.")

say("")
say("WHAT TO READ")
say("  * Both ports render the observation the hand-written kernel produced,")
say("    on the state the hand-written kernel produced, byte for byte, on this")
say("    card in this run. That is what makes the ratio a statement about code")
say("    generation rather than about two different computations.")
say("  * `naive` is minecu_tg/render.py verbatim; `lean` is the same")
say("    computation with the index arithmetic hoisted. The difference between")
say("    the two columns is the port's cost, not the compiler's.")
say("  * No number here is transcribed from any notes file. Every figure is")
say("    produced by an instrument in this notebook on this card.")
say("  * Instruction and register counts are NOT taken from this notebook:")
say("    they are a property of a compiler version, this image is CUDA 12.8,")
say("    and the offline sm_75 route is 12.9 on BOTH sides.")
say("    minecu_tg_opt/opcount.py is where the SASS numbers come from, and it")
say("    re-derives the hand-written side's own 776 and its 38-instruction step")
say("    from minecu_ref/build/kernels_sm_75.sass rather than quoting them.")

flush_summary()
say("")
say("wrote /kaggle/working/SUMMARY.txt")
say("cell 4 done.")
'''


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
    ap.add_argument("--slug", default="mc-minecu-tgopt-t4")
    ap.add_argument("--batches", default=None)
    ap.add_argument("--variants", default=VARIANTS)
    a = ap.parse_args()

    # three levels up: minecu_tg_opt/kaggle/make_notebook.py -> minecu_tg_opt/
    # kaggle -> minecu_tg_opt -> the repo root. Two levels lands on
    # minecu_tg_opt/ and every input path becomes minecu_tg_opt/minecu_ref/...,
    # which fails with "missing input" the first time anyone runs this without
    # --repo.
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

    batches = ([int(x) for x in a.batches.split(",")] if a.batches else BATCHES)
    subs = [("__EXPECT__", json.dumps(expect)),
            ("__PAYLOADS__", json.dumps(payloads)),
            ("__BATCHES__", json.dumps(batches)),
            ("__VARIANTS__", json.dumps(a.variants)),
            ("__DUMP_BATCH__", str(DUMP_BATCH)),
            ("__ITERS__", str(ITERS)),
            ("__WARMUP__", str(WARMUP)),
            ("__TIMING_MAX__", str(TIMING_MAX))]

    def fill(src: str) -> str:
        for k, v in subs:
            src = src.replace(k, v)
        return src

    cells_src = [fill(PRELUDE), fill(CELL1), fill(CELL2), fill(CELL3), fill(CELL4)]

    kd = os.path.dirname(a.out)
    os.makedirs(kd, exist_ok=True)

    def md(src, cid):
        # `id` on every cell: nbformat 5.5 requires it and the image warns
        # loudly, and a warning in a run log is a warning somebody stops
        # reading.
        return {"cell_type": "markdown", "id": cid, "metadata": {},
                "source": src.splitlines(True)}

    def code(src, cid):
        return {"cell_type": "code", "id": cid, "execution_count": None,
                "metadata": {}, "outputs": [], "source": src.splitlines(True)}

    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python", "version": "3.12.0"}},
          "cells": [
              md("# minecu_tg_opt: a competent TinyGrad port, against the "
                 "hand-written CUDA, on this T4\n\n"
                 "The same voxel DDA, written by hand in CUDA\n"
                 "(`minecu_ref/src/kernels.cu`, vendored unmodified) and twice\n"
                 "in TinyGrad: once the way a careless port writes it\n"
                 "(`minecu_tg/render.py`) and once the way a competent C\n"
                 "programmer writes it (`minecu_tg_opt/render.py`).\n\n"
                 "Both ports are checked against the hand-written side's own\n"
                 "bytes on this card before anything is timed. Read\n"
                 "`minecu_tg_opt/render.py`'s docstring for the exact list of\n"
                 "things it is not allowed to do, and for the four rewrites\n"
                 "that separate the two.", "intro"),
              md("## What this produces\n\n"
                 "1. `SUMMARY.txt` -- the full log, the primary artefact. It is\n"
                 "   flushed at the end of every cell, so a later cell that\n"
                 "   times out still leaves the earlier cells' output in it.\n"
                 "2. `ref_timing.json` -- the hand-written side, per batch.\n"
                 "3. `tgopt/report.json` -- both ports: identity, timing, memory,\n"
                 "   and the DDA workload of each timed run.\n"
                 "4. `obs_equivalence.json` -- both ports against the reference's\n"
                 "   own observation, on the reference's own state.\n\n"
                 "No number here is copied from a notes file. Every figure is\n"
                 "produced by an instrument in this notebook.", "artefacts"),
              code(cells_src[0], "prelude"),
              code(cells_src[1], "provenance-build-ref"),
              code(cells_src[2], "tgopt-timing"),
              code(cells_src[3], "gate-identity"),
              code(cells_src[4], "summary"),
          ]}
    with open(a.out, "w") as fh:
        json.dump(nb, fh, indent=1)

    # A sidecar listing the payloads, so the pre-flight can check the layout
    # WITHOUT scraping the driver's source text for `PAYLOADS = ...`. The
    # sibling notebook's pre-flight does that scrape, and it breaks the moment
    # the driver's formatting changes -- silently, in the direction of "the
    # pre-flight is not checking anything".
    with open(os.path.join(kd, "payloads.json"), "w") as fh:
        json.dump({"expect": expect, "payloads": payloads,
                   "ref_files": REF_FILES, "tg_files": TG_FILES,
                   "opt_files": OPT_FILES,
                   "notebook_sha256": sha256_file(a.out)}, fh, indent=1)

    # Kaggle derives the kernel's URL slug from `title` and rejects the upload
    # if it does not round-trip to `id` (409), so the title is DERIVED.
    title = " ".join(w for w in a.slug.split("-") if w)
    meta = {"id": f"{a.owner}/{a.slug}", "title": title,
            "subtitle": "a competent TinyGrad DDA against hand-written CUDA, on a T4",
            "code_file": os.path.basename(a.out), "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_tpu": False, "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
            "model_sources": []}
    import re as _re
    derived = _re.sub(r"[^a-z0-9]+", "-", meta["title"].lower()).strip("-")
    if derived != a.slug:
        raise SystemExit(f"kernel title {meta['title']!r} slugs to {derived!r}, "
                         f"which does not match the id slug {a.slug!r}; "
                         f"`kaggle kernels push` will 409.")
    with open(os.path.join(kd, "kernel-metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"wrote {a.out}  ({os.path.getsize(a.out)} B)")
    print(f"wrote {os.path.join(kd, 'kernel-metadata.json')}")
    print(f"  {len(payloads)} payloads, {sum(len(p[1]) for p in payloads)} b64 bytes")
    print(f"  5 code cells: prelude, provenance+build+reference, tgopt timing,")
    print(f"  gate+identity, summary. The 1800 s nbclient cap is PER CELL.")
    print(f"  batches {batches} (timed to {TIMING_MAX}), identity at {DUMP_BATCH}")
    print(f"  variants {a.variants}, iters {ITERS}")
    print(f"  kernel id {meta['id']}  title {meta['title']!r}")


if __name__ == "__main__":
    main()
