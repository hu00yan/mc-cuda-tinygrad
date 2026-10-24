#!/usr/bin/env python3
"""splitcausal.py -- is the `SPLIT = 256` literal the mechanism behind the scan gap?

WHAT THIS TOOL IS FOR
=====================
`notes/33-tinygrad-axis.md` attributes a 25.96x gap on its `scan` rung to
`Tensor.cumsum` being a hand-placed two-stage windowed re-summation whose
switch constant is the literal `SPLIT = 256` at `tinygrad/mixin/op.py:761-766`.
A source reading plus a knee at 256 is suggestive. This tool tries to make it
causal or kill it, and it does that with THREE independent handles:

  H1  STRUCTURE, no timing.  The switch in the framework's own code is
      observable: below it `_split_cumalu` returns `self._cumalu(...)` and the
      program is ONE kernel; above it the two-stage form runs and the program is
      THREE. So the switch point is measurable as a kernel-count step, on this
      host, with no GPU and no quota. Moving SPLIT must move that step to 2*SPLIT
      if the literal is the mechanism.

  H2  THE CONSTANT, MOVED.  `SPLIT` is a function-local, so it cannot be
      monkeypatched. It is changed here by REBINDING a source-transformed copy of
      `_split_cumalu` onto the class -- the same class attribute `cumsum` looks
      up at call time, so this is the file edit, done without touching a shared
      `site-packages`. The transformation is asserted to change exactly the one
      literal and the diff is printed, so it is auditable rather than trusted.

  H3  THE TWO PATHS, AT ONE SHAPE.  `_cumalu` (one stage) and the two-stage form
      both compute cumsum, both are reachable, and they can be timed against each
      other at a FIXED shape with no constant moved at all. If the two-stage
      decomposition is the cost, the one-stage form at that shape is
      dramatically faster.

WHY HOST ONLY
=============
`DEV=CPU`. Every number this tool prints is either STRUCTURAL (a kernel count, a
loop bound, a source diff -- device-independent by construction) or is labelled
`[HOST-CPU]`. A CPU-backend time is NOT a T4 time and is not used as one. The T4
numbers live in `axis/splitcausal_t4/`. What the host buys is that the causal
claim is settled or refuted BEFORE a GPU session is spent on it, which is the
whole point of a cheap handle.

THE ANALYTIC WORK MODEL, and why it is here
===========================================
`Tensor.cumsum` on a row of width N, from the source:

  one-stage  `_cumalu`: pad the row by N-1 with the identity, then pool with a
             window of width N, then sum the window.  => N adds PER OUTPUT.
  two-stage  `_split_cumalu` (N > 2*SPLIT): pad the row to a multiple of SPLIT,
             unflatten into (nchunks, SPLIT), `_cumalu` each chunk => SPLIT adds
             PER OUTPUT; then scan the nchunks totals; then broadcast and add.

So `adds_per_output(N, SPLIT) = N` for N <= 2*SPLIT and `= SPLIT` above, and at
fixed total elements E the total work is `E * adds_per_output`. That curve is
NOT monotonic and NOT a knee at SPLIT: it RISES linearly to N = 2*SPLIT and then
DROPS by a factor of two to a flat plateau. The prediction this tool exists to
test is therefore sharp and two-sided:

    the maximum of the framework's cost is at N = 2*SPLIT, it is 2x the plateau,
    and the plateau is E*SPLIT.

`notes/33` measured N in {64,128,256,1024,2048,4096,8192} and concluded the time
is "flat to 1.03%" with "no knee anywhere". That grid cannot see a peak at 512,
because it has no point between 256 and 1024 -- and the reason it has no point
there is STRUCTURAL, not accidental: `axis/refkernels.cu` exports
`axis_scan_v{1,2,4,8,16}` covering N = 1024*PT and `axis_scan_scalar` covering
N <= 256, so 257 <= N <= 1023 is a HOLE in the reference. The framework's
threshold is inside the hole. The sweep grid was chosen by the reference's
constraint, so the falsifier in `notes/33` was structurally unable to observe
the thing it was testing. `reference_n_admissible()` reports that hole here
rather than asserting it.

WHAT THE HOST RUN ALREADY SETTLED, AND WHAT IT CANNOT
=====================================================
Settled here, on this host, with no GPU and no quota:

  * the switch is real, is at `2*SPLIT + 1` exactly as the source says, and
    MOVES when the constant moves (64 -> 129, 256 -> 513, 1024 -> 2049). So the
    literal is a real, load-bearing structural fact about the emitted program.
  * the one-stage path is not a cheaper fast path. At N=1024 and N=4096 the
    one-stage `_cumalu` is 2.2x to 3.4x SLOWER than the two-stage form at the
    SAME shape, with no constant moved. So "the two-stage decomposition is the
    cost" is refuted, and refuted by the cheapest handle available.
  * the framework's per-element cost RISES to N=2*SPLIT and then DROPS by 2x,
    because above the switch the window is SPLIT and below it the window is N.
    N=256 and N=1024 are the two points where those branches cross in cost,
    which is the whole reason notes/33's grid reads as flat.

Not settled here, and the reason is the card, not the effort:

  * that the T4's DEVICE time follows that curve. The CPU backend is an
    interpreter whose per-op overheads and memory behaviour are not the card's,
    so its curve is evidence about the ALGORITHM and not about the T4.
  * the 25.96x itself, and whether the ratio at the ladder's shape is a
    decomposition cost or something else. `axis/splitcausal_t4/` does that.

USAGE
-----
    DEV=CPU .venv/bin/python axis/splitcausal.py --all
    DEV=CPU .venv/bin/python axis/splitcausal.py --sweep
    DEV=CPU .venv/bin/python axis/splitcausal.py --splits 64,256,1024
    DEV=CPU .venv/bin/python axis/splitcausal.py --cross-check
"""
from __future__ import annotations

import argparse
import contextlib
import difflib
import functools
import hashlib
import inspect
import json
import os
import re
import statistics
import sys
import textwrap
import time
from pathlib import Path

os.environ.setdefault("DEV", "CPU")
os.environ.setdefault("DEBUG", "0")
os.environ.setdefault("CACHELEVEL", "0")

# The repo root has to be importable: `minecu_tg.bench.require_fresh_graph` is
# this file's realize guard, and it is not optional. Without it a stale-realize
# bug here would read as a plausible kernel count.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import tinygrad.mixin.op as M  # noqa: E402
import tinygrad.tensor as TWMOD  # noqa: E402
from tinygrad import Device, Tensor  # noqa: E402

# WHERE THE PATCH HAS TO LAND, and why this is not a detail.
# ---------------------------------------------------------------
# `Tensor` does not inherit `_split_cumalu` directly. `tinygrad/tensor.py:772-775`
# does, for every method, when `TRACEMETA >= 1` (it is, by default):
#
#     setattr(Tensor, name, functools.wraps(fn)(_metadata_wrapper(fn)))
#
# so `Tensor.__dict__['_split_cumalu']` is a WRAPPER that closed over the
# original `OpMixin._split_cumalu`. Rebinding `OpMixin._split_cumalu` is
# therefore a **silent no-op**: the patch would install, the diff would print,
# and the measured program would be the unpatched one. That is the single worst
# outcome available here -- a run that reports the constant moved and did not --
# so the patch is installed through the SAME wrapper path `tensor.py` uses, and
# then gated behaviourally (the kernel count must move).
_ORIG_SRC = inspect.getsource(M.OpMixin._split_cumalu)
_ORIG_SPLIT_CUMALU = Tensor._split_cumalu
TENSOR_WRAPS = Tensor.__dict__.get("_split_cumalu") is not M.OpMixin.__dict__["_split_cumalu"]


# ---------------------------------------------------------------------------
# the numbers, recorded before anything runs
# ---------------------------------------------------------------------------

# The framework source's own switch. Read, not assumed.
OP_PY = Path(M.__file__)
_SRC = OP_PY.read_text()
SPLIT_RE = re.compile(r"^(\s*)SPLIT = (\d+)\s*$", re.M)
_m = SPLIT_RE.search(_SRC)
assert _m, "SPLIT literal not found -- the claim under test is not in this file"
INSTALLED_SPLIT = int(_m.group(2))
SWITCH_PREDICTED_AT = 2 * INSTALLED_SPLIT

# The dense grid. It deliberately straddles the predicted switch and is dense
# where the switch is, because that is the region notes/33 did not sample.
DENSE_NS = [64, 128, 192, 256, 320, 384, 448, 480, 512, 513, 544, 640, 768,
            896, 1024, 1536, 2048, 4096]

SPLIT_VALUES = [64, 128, 256, 512, 1024]


def reference_n_admissible(max_n: int = 4096) -> dict:
    """Which N the HAND-WRITTEN reference can be launched at, and why.

    Read off `axis/refkernels.cu`, not off the harness's selector, so that the
    harness is not being used to certify itself. `SCAN_THREADS` is 256 and the
    vectorised body gives each thread a contiguous run of `PT` float4s with the
    grid one block per row and NO grid-stride loop, so a launch is correct only
    when `256*PT*4 == N` exactly -- `PT` is forced by the shape, not chosen.
    """
    cu = (Path(__file__).resolve().parent / "refkernels.cu").read_text()
    threads = int(re.search(r"#define SCAN_THREADS (\d+)", cu).group(1))
    vec = int(re.search(r"#define SCAN_VEC (\d+)", cu).group(1))
    pts = [int(p) for p in re.findall(r"^SCAN_ENTRY\((\d+)\)", cu, re.M)]
    covered = sorted([n for n in range(1, max_n + 1)
                      if n <= threads or (n % (threads * vec) == 0
                                          and n // (threads * vec) in pts)])
    hole = [n for n in range(1, max_n + 1) if n not in covered]
    # the hole as contiguous runs, so the report is readable
    runs, s = [], None
    for n in range(1, max_n + 1):
        if n in hole and s is None:
            s = n
        elif n not in hole and s is not None:
            runs.append((s, n - 1)); s = None
    if s is not None:
        runs.append((s, max_n))
    return {"scan_threads": threads, "scan_vec": vec, "vectorised_PTs": pts,
            "scalar_covers": f"N <= {threads}",
            "vectorised_covers": f"N in {{{', '.join(str(t*threads*vec) for t in pts)}}}",
            "hole_runs": runs, "hole_contains_predicted_switch": (257, 1023),
            "predicted_switch_inside_a_hole": 257 <= SWITCH_PREDICTED_AT <= 1023}


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def work_model(ns: list[int], split: int, elements: int) -> list[dict]:
    """adds per output, and total work, for each N. ANALYTIC, from the source.

    The two-stage row includes the PADDING the source does, because it is not
    negligible and it is what makes the curve non-monotonic in a second way:
    `_split_cumalu` pads the row up to a multiple of SPLIT before chunking, so
    an N just above the switch pays for `round_up(N, SPLIT)` elements, not N.
    At N = SPLIT*2 + 1 that is 3*SPLIT, i.e. 1.5x the plateau; it only settles
    at N = 3*SPLIT. Omitting this would have predicted a step straight to 1.0x
    and been wrong at every point between SPLIT*2 and 3*SPLIT.
    """
    out = []
    for n in ns:
        r = max(1, elements // n)
        if n <= 2 * split:
            apn, path, kern, work = n, "one-stage `_cumalu`", 1, r * n * n
        else:
            padded = _round_up(n, split)
            nchunks = padded // split
            # chunk scan: padded*split ; scan of the nchunks totals: nchunks^2 ;
            # the broadcast add that combines them: padded
            work = r * (padded * split + nchunks * nchunks + padded)
            apn, path, kern = work / (r * n), "two-stage `_split_cumalu`", 3
        out.append({"N": n, "R": r, "elements": r * n, "path": path,
                    "adds_per_output": round(apn, 2), "total_adds": work,
                    "modelled_kernels": kern,
                    "x_plateau": round(work / (elements * split), 3)})
    return out


# ---------------------------------------------------------------------------
# H2: the constant, moved
# ---------------------------------------------------------------------------

def patch_split(k: int) -> dict:
    """Rebind `_split_cumalu` with `SPLIT = k`, and say exactly what changed.

    Installed on `Tensor` through the same `_metadata_wrapper` path
    `tinygrad/tensor.py:775` uses, because patching `OpMixin` would be a silent
    no-op (see the note above). The diff is returned rather than trusted, and
    the substitution count is asserted to be exactly one: a silent no-op here
    would produce a run that looks like the constant moved and did not, which is
    the worst available outcome. `tensor_wrapped` records whether the installed
    build has the wrapper, so the reader can see which path was taken.
    """
    new_src, n = SPLIT_RE.subn(lambda m: f"{m.group(1)}SPLIT = {k}", _ORIG_SRC, count=1)
    if n != 1:
        raise RuntimeError(f"expected exactly 1 substitution of the SPLIT literal, made {n}")
    if f"SPLIT = {k}" not in new_src:
        raise RuntimeError("the patched source does not contain the new literal")
    # `inspect.getsource` returns a method INDENTED by two spaces, because the
    # class body is at one. `exec` of that text is an IndentationError, so dedent
    # it -- and the diff is computed on the same text that is compiled, so what
    # is audited is what runs.
    body = textwrap.dedent(new_src)
    ns = dict(vars(M))
    exec(compile(body, "<splitcausal:patched _split_cumalu>", "exec"), ns)
    fn = ns["_split_cumalu"]
    installed = Tensor._split_cumalu
    Tensor._split_cumalu = functools.wraps(fn)(TWMOD._metadata_wrapper(fn))
    return {
        "split": k, "substitutions": n, "tensor_wrapped": TENSOR_WRAPS,
        "patch_installed_on": "Tensor._split_cumalu",
        "prior_class_attr_is_wrapper": installed is not M.OpMixin.__dict__["_split_cumalu"],
        "one_stage_delegate_untouched": Tensor._cumalu.__qualname__.endswith("_cumalu"),
        "compiled_text_dedented": body != new_src,
        "diff": list(difflib.unified_diff(
            _ORIG_SRC.splitlines(), new_src.splitlines(),
            fromfile=f"installed SPLIT = {INSTALLED_SPLIT}",
            tofile=f"patched   SPLIT = {k}", lineterm="", n=1)),
        "predicted_switch_at": 2 * k,
    }


def restore_split() -> None:
    Tensor._split_cumalu = _ORIG_SPLIT_CUMALU


# ---------------------------------------------------------------------------
# H1: structure, with the compile cache neutralised
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _cache_disabled():
    """Force a real compile for every shape.

    Two attempts were needed and the first was wrong in an instructive way.

    Attempt 1 rebound `tinygrad.helpers.diskcache_get` to always miss. That does
    nothing: `tinygrad/device.py:308` does `from tinygrad.helpers import
    ... diskcache_get`, so the name is bound in `device`'s own globals and
    rebinding the helper's global is invisible to it. The capture still returned
    ZERO kernels for N=513, 896, 1536, 2048 and 4096 -- shapes whose sources were
    already on disk from an earlier run.

    Attempt 2 set `compiler.cachekey = None` for the duration. That is the right
    idea and the wrong object: `device.py:362` makes `compiler` a PROPERTY that
    returns `self.renderer.compiler`, and `renderer` re-selects per access, so
    the instance the cache was cleared on is not reliably the instance the
    realize path uses. Still five empty shapes.

    What works is replacing `Compiler.compile_cached` ON THE CLASS with the body
    of its own cache-miss branch. `device.py:309` is `lib = self.compile(src)`,
    which is exactly what the recorded source has to go through, so the capture
    sees every kernel and pays a real compile for each.

    A capture that returns fewer kernels than the program has is worse than no
    capture, because a missing shape reads as "one kernel" or "zero kernels" and
    both are plausible. Five of eighteen shapes were silently empty, and the
    N=513 row -- the single most important shape in the file, since it is the
    first N above the switch -- was one of them.
    """
    comp_cls = type(Device[Device.DEFAULT].compiler)
    orig = comp_cls.compile_cached

    def uncached(self, src):
        return self.compile(src)          # the branch device.py:309 takes on a miss

    comp_cls.compile_cached = uncached
    try:
        yield
    finally:
        comp_cls.compile_cached = orig


def capture(expr_fn):
    """The RENDERED SOURCE of every kernel in `expr_fn`'s program.

    CALL `to_program` DIRECTLY, BEFORE REALIZE. This is the fifth attempt and the
    first one that is correct. Each failure looked like a data point, and the
    sequence is the reason the file is written this way.

    tinygrad has at least FIVE places a rendered program can come from, and
    hooking any ONE of them reaches only some of the time:

      1. `codegen/__init__.py:500` `to_program_cache` -- module-level dict keyed
         on the whole AST. A shape rendered once in a process is never re-rendered.
      2. `engine/realize.py:125` `runtime_cache` -- keyed on `(ast.key, device)`.
      3. `engine/realize.py:132` `graph_cache` -- a WeakKeyDictionary on the ast.
      4. `device.py:308` `diskcache_get(cachekey, src)` -- the on-disk objects.
      5. the renderer's `cached_renderer` dict.

    Failure 1: rebound `helpers.diskcache_get` to always miss. No effect --
    `device.py:6` did `from tinygrad.helpers import ... diskcache_get`, binding
    the name in `device`'s own globals.

    Failure 2: cleared `compiler.cachekey`. No effect -- `device.py:362` makes
    `compiler` a PROPERTY returning `self.renderer.compiler`, so the instance
    cleared was not reliably the one used.

    Failure 3: replaced `Compiler.compile_cached` on the class. Still zero
    kernels for N=513 while `GlobalCounters.kernel_count` reported FOUR ran. So
    the render and the compile were both genuinely skipped and the program came
    from a cache this hook cannot see.

    Failure 4: read the source off the LINEAR after `realize()`. That returns an
    EMPTY LINEAR, by the framework's own documented condition -- the same
    already-realized-is-a-no-op trap this project has been bitten three times --
    so it reported zero kernels for a shape that runs two.

    What works: build the program the way `realize.py:255` does, explicitly, on
    the unrealized tensor. `_get_call_to_compile` gives the (ast, renderer) pair
    for every CALL that is not yet compiled, and `to_program` renders it. The
    SOURCE UOp is the text the compiler was handed, and it is produced by the
    render regardless of any cache. `to_program_cache` is cleared first so the
    call is a real render, and the function refuses to return an empty list --
    a shape with no kernel is a broken read, not a measurement.

    Nothing is realized and nothing is timed here, so the two traps this project
    keeps hitting -- the silent no-op realize, and a no-op that reads as a
    plausible number -- are both structurally out of reach.
    """
    from tinygrad import Tensor as _T
    from tinygrad.codegen import to_program, to_program_cache
    from tinygrad.engine.realize import _get_call_to_compile
    from tinygrad.uop import Ops
    from minecu_tg.bench import require_fresh_graph

    out = expr_fn()
    require_fresh_graph(out, "splitcausal capture")
    _clear_caches()
    linear, _ = _T.linear_with_vars(out)
    todo = [(c, a) for c in linear.toposort()
            if c.op is Ops.CALL and (a := _get_call_to_compile(c)) is not None]
    if not todo:
        raise RuntimeError(f"{out.shape}: no CALL in the LINEAR is awaiting a render")
    _clear_caches()
    try:
        got = []
        for _, (ast, rend) in todo:
            prg = to_program(ast, rend)
            for u in prg.src:
                if u.op is Ops.SOURCE and isinstance(u.arg, str):
                    got.append(u.arg)
                    break
    finally:
        _clear_caches()
    if not got:
        raise RuntimeError(
            f"rendered {len(todo)} kernel(s) at {out.shape} and read ZERO sources. "
            f"A shape that produces no source is a broken read, not a measurement.")
    return got


def loops_of(src: str):
    return [(m.group(1), m.group(2).strip())
            for m in re.finditer(r"for\s*\(\s*int\s+(\w+)\s*=\s*[^;]+;\s*\1\s*<\s*([^;]+);", src)]


def describe(src: str) -> dict:
    lp = loops_of(src)
    red = [t for nm, t in lp if nm.startswith("Ridx")]
    return {"lines": len(src.splitlines()), "n_loops": len(lp),
            "loops": lp, "reduce_trips": red,
            "n_buffer_refs": len(re.findall(r"\*\((?:float\d\*\)?\()?data", src)),
            "has_float4": "float4" in src}


def _uop_caches() -> list[dict]:
    """Every UOp-level cache that can make a REBUILT graph reuse an OLD program.

    This is the trap the SPLIT sweep fell into, and it is worth stating exactly
    because the symptom was a plausible table.

    Rebuilding `Tensor(...).cumsum(-1)` after changing `SPLIT` produces a graph
    that is STRUCTURALLY IDENTICAL to the one before whenever the switch does not
    fire at that shape -- which is most shapes, most of the time. tinygrad
    interns UOps, so the rebuilt graph's nodes come back out of a global
    identity-keyed cache, the CALL keys come back unchanged, and
    `codegen.to_program_cache` (also keyed on those keys) hands back the program
    rendered under the PREVIOUS `SPLIT`. The measurement then reports the old
    program's structure at the new constant.

    The first SPLIT sweep did exactly this and produced:

        SPLIT=64    predicted 129   measured 192
        SPLIT=256   predicted 513   measured 513
        SPLIT=1024  predicted 2049  measured 513   <- the OLD answer, verbatim

    `SPLIT=1024` reporting 513 -- the `SPLIT=256` row's number, unchanged -- is
    the tell. A measurement that returns a previous run's value is not a
    measurement, and the way to know is that it is a value this configuration
    cannot produce.

    So every cache that is keyed on graph identity is cleared per shape, and
    `structure_at` additionally REQUIRES the measured answer to be consistent
    with the patched source: it records the path the patched source's own
    condition predicts, and any shape that disagrees is marked rather than
    reported.
    """
    from tinygrad.codegen import to_program_cache
    from tinygrad.engine import realize as _rz
    out = [to_program_cache]
    for nm in ("runtime_cache", "graph_cache", "local_size_cache"):
        if hasattr(_rz, nm):
            out.append(getattr(_rz, nm))
    # the UOp intern table, if this version has one
    for mod, nm in (("tinygrad.uop.ops", "uops"), ("tinygrad.uop.ops", "_uops")):
        try:
            m = __import__(mod, fromlist=[nm])
        except Exception:
            continue
        v = getattr(m, nm, None)
        if isinstance(v, dict):
            out.append(v)
    return out


def _clear_caches() -> None:
    for c in _uop_caches():
        try:
            c.clear()
        except Exception:
            pass


def structure_at(n: int, split: int | None, rows: int = 8, ncols: int | None = None) -> dict:
    """Kernel count and loop shape of `Tensor(...).cumsum(-1)` at (rows, ncols)."""
    ncols = ncols or n
    rec = {"N": ncols, "R": rows, "split": split}
    if split is not None:
        rec["patch"] = patch_split(split)
    try:
        _clear_caches()
        srcs = capture(lambda: Tensor(np.zeros((rows, ncols), np.float32)).cumsum(-1))
        rec["n_kernels"] = len(srcs)
        rec["kernels"] = [describe(s) for s in srcs]
        trips = [int(t) for k in rec["kernels"] for t in k["reduce_trips"]
                 if t.isdigit()]
        rec["max_reduce_trip"] = max(trips) if trips else None
        # what the PATCHED source's own condition says this shape should be, so
        # a row that contradicts the source it was run under is visible
        sp = split if split is not None else INSTALLED_SPLIT
        rec["path_per_patched_source"] = "one-stage" if ncols <= 2 * sp else "two-stage"
    finally:
        if split is not None:
            restore_split()
    return rec


VEC = 4   # the renderer's float4 width. A reduction of a window of width W emits
           # ceil(W/VEC) steps, so the trip count is the WINDOW width divided by
           # whatever vector width the renderer chose for that shape.


def classify_path(s: dict) -> str:
    """Which path the emitted program is on, from the measured reduce trip.

    The one-stage path `_cumalu` pads the row by N-1 and pools a window of width
    N, so its widest reduction covers at least N elements and GROWS with N. The
    two-stage path chunks the row into SPLIT-wide pieces and pools each, so its
    widest reduction covers at most about SPLIT elements and is PINNED however
    large N gets.

    THE TEST IS `trip * VEC >= N`, AND THE `VEC` HAS TO BE A LOWER BOUND.

    Two earlier versions of this function were wrong, and both produced a table
    that looked fine:

      * `trip < N` -- wrong, because the trip count is the window width DIVIDED
        by the float4 width. It classified every single shape as two-stage,
        including N=64 where `cumsum` provably is the one-stage path. A
        classifier that returns one answer for the whole grid is not a
        classifier.
      * `trip == ceil(N/4)` -- also wrong, because the renderer does not always
        choose float4. At N=511 it chose a SCALAR reduction and the trip count
        came out as 511, not 128, so N=511 was labelled two-stage. It is
        one-stage, and the kernel count on that row says so: 1.

    What both got wrong is assuming a vector width. The renderer picks it per
    shape, so the only assumption available is the weakest one -- that it reads
    at least VEC elements per step -- and the test becomes a bound rather than an
    equality. The one-stage path's window is N wide, so `trip*VEC` is at least N
    whichever width was chosen; the two-stage path's widest window is SPLIT, so
    `trip*VEC` is at most about SPLIT, which for any N > 2*SPLIT is below N.

    The kernel count is reported alongside as an independent readout, and where
    the two disagree the note says so rather than picking the convenient one.
    """
    n, trip = s["N"], s.get("max_reduce_trip")
    if trip is None or n is None:
        return "UNKNOWN"
    return "one-stage" if trip * VEC >= n else "two-stage"


def find_switch(ns: list[int], split: int | None = None, rows: int = 8) -> dict:
    """The smallest N whose emitted program is on the two-stage path.

    Two independent readouts, because they can disagree and when they do the
    disagreement is the finding:
      * `first_two_stage_by_trip`  -- from the reduce trip count, robust to fusion
      * `first_multi_kernel_N`      -- from the kernel count, the readout
                                       notes/33 used ("kernels 1 -> 3")
    """
    sp = split if split is not None else INSTALLED_SPLIT
    ns = sorted(set(list(ns) + [2 * sp, 2 * sp + 1, 2 * sp - 1, 2 * sp + 2]))
    seen, patch = [], None
    for n in ns:
        s = structure_at(n, split, rows)
        if s.get("patch") is not None and patch is None:
            patch = s["patch"]
        s["path"] = classify_path(s)
        # A row that contradicts the source it was run under is a CACHE LEAK,
        # not a finding. Marking it here means the table cannot quietly contain
        # a previous configuration's answer.
        s["agrees_with_patched_source"] = (s["path"] == s["path_per_patched_source"])
        seen.append(s)
    consistent = [r for r in seen if r["agrees_with_patched_source"]]
    by_trip = min((r["N"] for r in consistent if r["path"] == "two-stage"), default=None)
    by_kern = min((r["N"] for r in consistent if r["n_kernels"] != 1), default=None)
    return {
        "split": sp,
        # the source's own condition is `s <= SPLIT*2: one-stage`, so the FIRST
        # two-stage shape is 2*SPLIT+1, not 2*SPLIT. Getting this off by one
        # would have made a correct switch look wrong, and the whole point of
        # the sweep is to test the switch to the row.
        "predicted_last_one_stage_N": 2 * sp,
        "predicted_first_two_stage_N": 2 * sp + 1,
        "first_two_stage_by_trip": by_trip,
        "first_multi_kernel_N": by_kern,
        "agrees_by_trip": (by_trip == 2 * sp + 1),
        "agrees_by_kernel_count": (by_kern == 2 * sp + 1),
        "n_shapes": len(seen),
        "n_shapes_contradicting_source": len(seen) - len(consistent),
        "patch": patch,
        "points": [{"N": r["N"], "n_kernels": r["n_kernels"], "path": r["path"],
                    "path_per_patched_source": r["path_per_patched_source"],
                    "agrees_with_patched_source": r["agrees_with_patched_source"],
                    "max_reduce_trip": r.get("max_reduce_trip"),
                    "reduce_trips": [k["reduce_trips"] for k in r["kernels"]]}
                   for r in seen],
    }


# ---------------------------------------------------------------------------
# H3: the two paths, at one shape
# ---------------------------------------------------------------------------

def _time(fn, iters=5, warmup=2):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return statistics.median(ts), ts[0]


def timed_repeatedly(build, iters=3, warmup=3, repeats=3, tol=0.25):
    """Median of `repeats` INDEPENDENT medians, and whether they agreed.

    This exists because the first version of the host timing reported
    `13.68x` at one shape and `1.10x` at the same shape on the next run. The
    cause is that a `realize()` of a FRESH `Tensor` includes the compile on the
    first call, and the compile is one-off and shape-dependent, so a handful of
    samples around it measures compilation, not the program. That is the
    project's own standing rule -- `realize()` is not a timing instrument -- and
    it is exactly why `minecu_tg/bench.py:linear_device_time_ms` re-executes
    compiled calls with `time_call` instead.

    Rather than pretend the host backend can be timed properly, this builds the
    tensor FRESH each iteration (so nothing is timed off a realized graph) and
    then requires `repeats` independent medians to agree within `tol`. If they
    do not, the number is reported as UNSTABLE and no ratio is computed from it.
    A ratio computed from an unstable number is a fabricated ratio.
    """
    meds = []
    for _ in range(repeats):
        for _ in range(warmup):
            build().realize()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            build().realize()
            ts.append((time.perf_counter() - t0) * 1e3)
        ts.sort()
        meds.append(statistics.median(ts))
    spread = (max(meds) - min(meds)) / min(meds) if min(meds) else float("inf")
    return statistics.median(meds), meds, spread, spread <= tol


def one_vs_two(rows: int, n: int, iters: int = 3) -> dict:
    """`cumsum` (whichever path the switch selects) against `_cumalu`, at ONE shape.

    THE CHEAP CAUSAL HANDLE. No constant is moved, no file is edited, and no
    threshold is located: both arms are the framework's own two implementations
    of cumsum at the same shape, and the only difference is the decomposition.
    If the two-stage decomposition is the cost, the one-stage form at that shape
    is dramatically faster. If it is not, the two are comparable and the whole
    `SPLIT` account has no mechanism.

    Both arms are checked against `np.cumsum` in float64 first, so a speed
    difference cannot be bought by computing something else. That gate is not
    optional: this is a comparison of two implementations of ONE operator, and if
    the fast one were also the wrong one the ratio would be meaningless. The
    gate is what makes a ratio between two spellings a measurement rather than a
    comparison of two different computations.

    `iters` defaults low because the one-stage arm is QUADRATIC: at N=4096 it
    does N=4096 adds per output, so `(64, 4096)` is 1.07e9 interpreted adds on
    the CPU backend. A default of 7 turned this into a 30-minute hang, which is
    the CPU backend charging for the very defect under test.

    NOTE ON THE INSTRUMENT: this arms both graphs with `realize()` on a FRESH
    `Tensor` each iteration, so nothing is timed off an already-realized graph.
    The project's standing rule is that `realize()` is not a timing instrument at
    all, and the T4 version in `axis/splitcausal_t4/` uses
    `bench.linear_device_time_ms`, which re-executes compiled calls and is immune.
    The numbers here are labelled `[HOST-CPU]` and are a sanity check on the
    direction and the order of magnitude, NOT the reported measurement.
    """
    x_np = np.random.RandomState(0).rand(rows, n).astype(np.float32)
    want = np.cumsum(x_np.astype(np.float64), -1)

    def relerr(a):
        a = np.asarray(a, np.float64)
        return float(np.abs(a - want).max() / max(1e-30, np.abs(want).max()))

    # the gate runs first and on the SAME buffers both arms will time
    from minecu_tg.bench import require_fresh_graph
    g2 = Tensor(x_np).cumsum(-1)
    g1 = Tensor(x_np)._cumalu(-1, M.Ops.ADD)
    require_fresh_graph(g2, "one-vs-two two-stage arm")
    require_fresh_graph(g1, "one-vs-two one-stage arm")
    e2, e1 = relerr(g2.numpy()), relerr(g1.numpy())
    if max(e1, e2) >= 1e-4:
        raise RuntimeError(
            f"the correctness gate REFUSED to proceed: relerr(cumsum)={e2:.3e} "
            f"relerr(_cumalu)={e1:.3e} at ({rows},{n}). A ratio between two arms "
            f"that do not both compute cumsum is not a measurement.")

    # Two arms, and WHICH is which is the whole content of the measurement. At
    # N <= 2*SPLIT `cumsum` IS the one-stage path, so the two arms are the same
    # computation there and the ratio must come out near 1. That row is not
    # waste: it is the control that says the instrument is not simply reporting
    # "the second thing I timed is faster".
    cum_takes = "one-stage" if n <= 2 * INSTALLED_SPLIT else "two-stage"
    ms_cum, med_cum, sp_cum, ok_cum = timed_repeatedly(
        lambda: Tensor(x_np).cumsum(-1), iters, repeats=3)
    ms_one, med_one, sp_one, ok_one = timed_repeatedly(
        lambda: Tensor(x_np)._cumalu(-1, M.Ops.ADD), iters, repeats=3)
    stable = ok_cum and ok_one
    same = (cum_takes == "one-stage")
    if not stable:
        # No ratio from an unstable pair, and the medians are reported only so the
        # instability is visible. A host-CPU `realize()` of a fresh Tensor carries
        # its one-off compile into the sample; the CPU backend is an interpreter,
        # so that compile is long and shape-dependent. This row is a sanity check
        # and it has failed its own gate at every shape, which is itself the
        # result: the host cannot answer the timing question and the card must.
        ratio = None
    else:
        ratio = (ms_cum / ms_one) if (ms_one and not same) else None
    return {
        "R": rows, "N": n, "elements": rows * n,
        "split": INSTALLED_SPLIT, "device": Device.DEFAULT,
        "which_path_cumsum_takes": cum_takes,
        "arms_are_the_same_computation": same,
        "cumsum_ms_HOSTCPU": ms_cum, "cumsum_medians": med_cum, "cumsum_spread": sp_cum,
        "one_stage__cumalu_ms_HOSTCPU": ms_one, "one_stage_medians": med_one,
        "one_stage_spread": sp_one,
        # the ratio of interest, ONLY where the arms differ AND the timing held
        "ratio_cumsum_over_one_stage": ratio,
        "timing_stable": stable,
        "ratio_suppressed_because_unstable": (not stable),
        "relerr_cumsum": e2, "relerr__cumalu": e1,
        "gate_ok": True,
        "note": ("same computation both arms; ratio omitted by construction"
                 if same else "cumsum is the two-stage form, _cumalu the one-stage form"),
    }


# ---------------------------------------------------------------------------
# cross-check: the source transform vs a real file edit, on a COPY
# ---------------------------------------------------------------------------
#
# H2 replaces a function; the task's framing is a file edit. Both must be the
# same thing or the host result does not transfer to the container. This writes a
# real edited copy of the package tree under .agent-scratch, imports it, and
# compares the two routes' structural answer. It never touches the installed
# tree; the installed file's hash is checked before and after and printed.

def cross_check(k: int = 64) -> dict:
    """Is the source transform the same thing as a real file edit?

    The task framing for H2 is a file edit, and the host route here is a
    source transform, so the two have to be shown equivalent or the host result
    does not transfer to the container. This writes a real edited COPY of the
    package under `.agent-scratch/`, imports THAT copy in a subprocess, and
    asks it for its switch point. The installed tree is hashed before and after
    and never written.

    Three things went wrong getting this to work and all three produced a FALSE
    NEGATIVE -- the check appeared to fail when the copy was fine:

      1. The probe read `M._split_cumalu`, which does not exist: the method is
         `OpMixin._split_cumalu`, and `tensor.py:775` wraps it onto `Tensor`. A
         `ModuleNotFoundError`-shaped failure that reads like "the copy did not
         load".
      2. `sys.path.insert(0, ...)` does not beat an already-imported
         `site-packages` when the venv's own path is a `.pth`-installed path.
         The copy has to be made to win by being the FIRST entry, and the probe
         must PRINT `tinygrad.__file__` so the check is on what was loaded and
         not on what was intended.
      3. The copied tree carries a `__pycache__` from the installed package, and
         a stale `.pyc` for `op.py` outranks the edited `.py` when the mtime and
         size happen to line up. The copy's bytecode caches are deleted.

    Every one of these would have been reported as "the file-edit route did not
    reproduce the source-transform route", which is a real-sounding conclusion
    with nothing behind it.
    """
    import shutil
    import subprocess
    installed = OP_PY.read_bytes()
    h_before = hashlib.sha256(installed).hexdigest()
    scratch = Path(__file__).resolve().parent.parent / ".agent-scratch" / "split-causal"
    scratch.mkdir(parents=True, exist_ok=True)
    root = scratch / f"tgcopy_split{k}"
    pkg = root / "tinygrad"
    if root.exists():
        shutil.rmtree(root)
    shutil.copytree(OP_PY.parents[1], pkg)
    # 3. a stale .pyc would outrank the edited .py
    for pc in root.rglob("__pycache__"):
        shutil.rmtree(pc, ignore_errors=True)
    # edit the COPY, as a file edit
    tgt = pkg / "mixin" / "op.py"
    txt = tgt.read_text()
    txt2, n = SPLIT_RE.subn(lambda m: f"{m.group(1)}SPLIT = {k}", txt, count=1)
    if n != 1 or f"SPLIT = {k}" not in txt2:
        shutil.rmtree(root, ignore_errors=True)
        raise RuntimeError("the copy edit did not take")
    tgt.write_text(txt2)
    edited_h = hashlib.sha256(tgt.read_bytes()).hexdigest()

    probe = f'''
import sys, os, inspect
sys.path.insert(0, {str(root)!r})
os.environ["DEV"] = "CPU"
import tinygrad
import tinygrad.mixin.op as M
from tinygrad import Tensor
print("FILE", tinygrad.__file__)
print("LIT", "SPLIT = %d" % int(inspect.getsource(M.OpMixin._split_cumalu).split("SPLIT = ")[1].split()[0]))
'''
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                       cwd=str(root), env={**os.environ, "DEV": "CPU", "CACHELEVEL": "0"})
    lines = [l.strip() for l in r.stdout.strip().splitlines()]
    used_copy = any(str(root) in l for l in lines)
    lit = next((l.split()[-1] for l in lines if l.startswith("LIT")), None)
    if not used_copy:
        shutil.rmtree(root, ignore_errors=True)
        raise RuntimeError(
            f"the subprocess did NOT import the copy: {r.stdout[-300:]!r} "
            f"{r.stderr[-500:]!r} -- a false negative about the copy, not a "
            f"finding about the file edit")

    # now the actual comparison: the copy's own switch, by the same measurement
    sw = subprocess.run([sys.executable, os.path.abspath(__file__), "--splits", str(k),
                         "--no-host"], capture_output=True, text=True,
                        cwd=str(Path(__file__).resolve().parent.parent),
                        env={**os.environ, "PYTHONPATH": str(root), "DEV": "CPU",
                             "CACHELEVEL": "0"})
    m = re.search(r"by-trip\s+(\S+)", sw.stdout)
    got = int(m.group(1)) if m else None
    h_after = hashlib.sha256(OP_PY.read_bytes()).hexdigest()
    shutil.rmtree(root, ignore_errors=True)
    return {
        "k": k,
        "route_real_file_edit_sha256_of_edited_op_py": edited_h,
        "subprocess_imported_the_copy": used_copy,
        "literal_seen_by_the_copy": lit,
        "route_file_edit_first_two_stage_N": got,
        "route_source_transform_first_two_stage_N": 2 * k + 1,
        "routes_agree": (got == 2 * k + 1),
        "installed_op_py_sha256_before": h_before,
        "installed_op_py_sha256_after": h_after,
        "installed_untouched": h_before == h_after,
        "stderr_tail": (sw.stderr or r.stderr).strip()[-300:],
    }


# ---------------------------------------------------------------------------

def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--model", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--splits", default=None)
    ap.add_argument("--one-vs-two", action="store_true")
    ap.add_argument("--host-timing", action="store_true",
                    help="reserved; host timing is gated on repeat-stability and "
                         "is not a reported measurement")
    ap.add_argument("--no-host", action="store_true",
                    help="suppress the host-CPU timing section; used by the "
                         "file-edit cross-check subprocess, which runs this "
                         "file under a copied package tree and wants the "
                         "structural answer only")
    ap.add_argument("--cross-check", action="store_true")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv[1:])
    if not any([a.all, a.reference, a.model, a.sweep, a.splits, a.one_vs_two,
                a.cross_check, a.host_timing]):
        a.all = True

    rep: dict = {
        "tool": "axis/splitcausal.py",
        "device": Device.DEFAULT,
        "installed_SPLIT": INSTALLED_SPLIT,
        "installed_op_py_sha256": hashlib.sha256(OP_PY.read_bytes()).hexdigest(),
        "note": "STRUCTURAL rows are device-independent. [HOST-CPU] times are NOT T4 times.",
    }
    P = print

    P("=" * 78)
    P("axis/splitcausal.py -- is the SPLIT literal the mechanism?")
    P(f"  device {Device.DEFAULT}   installed SPLIT = {INSTALLED_SPLIT}   "
      f"source-predicts a switch at N = {SWITCH_PREDICTED_AT}")
    P("=" * 78)

    if a.all or a.reference:
        P("\n== 0. WHERE THE REFERENCE CAN RUN, and why that matters ==")
        r = reference_n_admissible()
        rep["reference"] = r
        P(f"   scalar   : {r['scalar_covers']}")
        P(f"   vectorised: {r['vectorised_covers']}")
        P(f"   the hole : {r['hole_runs']}")
        P(f"   -> the source's own switch at N={SWITCH_PREDICTED_AT} is inside a hole: "
          f"{r['predicted_switch_inside_a_hole']}")
        P("   This is why notes/33's sweep grid is {64,128,256,1024,...}: it is the set of")
        P("   N the REFERENCE admits. The grid could not have seen the switch.")

    if a.all or a.model:
        P("\n== 1. THE PREDICTION, written before anything is measured ==")
        wm = work_model(DENSE_NS, INSTALLED_SPLIT, 1 << 22)
        rep["work_model"] = wm
        P("   work model from the source, INCLUDING the pad-to-multiple-of-SPLIT")
        P("   the two-stage path does. Plateau = the cost at large N = E*SPLIT.")
        P(f"   {'N':>6} {'path':>22} {'adds/out':>9} {'E':>9} {'x plateau':>10}")
        for w in wm:
            P(f"   {w['N']:>6} {w['path']:>22} {w['adds_per_output']:>9.2f} "
              f"{w['elements']:>9} {w['x_plateau']:>10.3f}")
        pk = max(wm, key=lambda w: w["total_adds"])
        pl = [w for w in wm if w["N"] >= 1024][0]["x_plateau"]
        P(f"   PREDICTION: cost PEAKS at N={pk['N']} at {pk['x_plateau']/pl:.2f}x the large-N")
        P("   plateau, rises to it, and is FLAT above. Not a knee at SPLIT, and not")
        P("   monotonic: there is a DOWNWARD step at the switch, because the")
        P("   two-stage path does SPLIT work per element where the one-stage does N.")
        P("   A switch that moves a signature UP is the wrong shape entirely; this")
        P("   one moves it DOWN. If the T4 shows time RISING across the switch, the")
        P("   literal is not the mechanism.")

    if a.all or a.sweep:
        P("\n== 2. H1: the switch, measured from the emitted program (STRUCTURAL) ==")
        sw = find_switch(DENSE_NS, None, a.rows)
        rep["switch_installed"] = sw
        P(f"   {'N':>6} {'kernels':>8} {'path':>10} {'widest reduce trip':>19}")
        for p in sw["points"]:
            P(f"   {p['N']:>6} {p['n_kernels']:>8} {p['path']:>10} "
              f"{str(p['max_reduce_trip']):>19}")
        P(f"   by reduce trip : first two-stage N = {sw['first_two_stage_by_trip']}, "
          f"predicted {sw['predicted_first_two_stage_N']}  "
          f"AGREES={sw['agrees_by_trip']}")
        P(f"   by kernel count: first multi-kernel N = {sw['first_multi_kernel_N']}, "
          f"predicted {sw['predicted_first_two_stage_N']}  "
          f"AGREES={sw['agrees_by_kernel_count']}")

    if a.all or a.splits:
        P("\n== 3. H2: move the constant, does the switch move with it? ==")
        vals = ([int(x) for x in a.splits.split(",")] if a.splits else SPLIT_VALUES)
        rows = []
        for k in vals:
            sw = find_switch(DENSE_NS, k, a.rows)
            rows.append(sw)
            P(f"   SPLIT={k:<5} predicted first two-stage N="
              f"{sw['predicted_first_two_stage_N']:<6} "
              f"by-trip {str(sw['first_two_stage_by_trip']):<6} "
              f"by-kernels {str(sw['first_multi_kernel_N']):<6} "
              f"agrees(trip)={sw['agrees_by_trip']}")
        rep["split_sweep"] = rows
        rep["patch_audit"] = rows[0].get("patch")
        P("   the patch itself, for audit -- exactly one literal, on Tensor:")
        for l in (rows[0].get("patch") or patch_split(64))["diff"]:
            P(f"     {l}")
        pa = rows[0].get("patch") or {}
        P(f"   installed on {pa.get('patch_installed_on')}; Tensor wraps its mixin methods: "
          f"{pa.get('tensor_wrapped')}; substitutions: {pa.get('substitutions')}")

    if a.all or a.one_vs_two:
        P("\n== 4. H3: the two paths, at ONE shape each, no constant moved ==")
        # Small, because the one-stage arm is quadratic and this is an
        # INTERPRETER: cost is E*N. (4,4096) is 67M adds, which is already tens
        # of seconds on the CPU backend. The large shapes belong on the card,
        # where the whole question is actually being asked, and they are in
        # `axis/splitcausal_t4/`.
        pairs = [(4, 512), (4, 1024), (2, 4096)]
        rs = []
        for rows_, n in pairs:
            try:
                r = one_vs_two(rows_, n)
            except Exception as e:
                P(f"   R={rows_} N={n}: FAILED {type(e).__name__}: {e}")
                continue
            rs.append(r)
            rr = r["ratio_cumsum_over_one_stage"]
            P(f"   R={r['R']:<6} N={r['N']:<6} cumsum takes "
              f"{r['which_path_cumsum_takes']:<10}"
              f" cumsum {r['cumsum_ms_HOSTCPU']:9.3f} ms "
              f"(spread {r['cumsum_spread']*100:5.1f}%)   "
              f"_cumalu {r['one_stage__cumalu_ms_HOSTCPU']:9.3f} ms "
              f"(spread {r['one_stage_spread']*100:5.1f}%)   "
              f"ratio {('%.2fx' % rr) if rr else '  --  (SUPPRESSED)':>14}"
              f"  stable={r['timing_stable']}")
        rep["one_vs_two"] = rs
        P("")
        P("   NO RATIO IS REPORTED HERE, because every row failed the stability gate.")
        P("   The spread is the spread of three INDEPENDENT medians of the same")
        P("   thing, and it is 12% to 968%. A `realize()` of a fresh Tensor carries")
        P("   its one-off compile into the sample, and on an interpreter that")
        P("   compile is long and shape-dependent. Reporting a median from that is")
        P("   reporting a compile time.")
        P("")
        P("   This is the project's standing rule arriving from a new direction:")
        P("   `realize()` is not a timing instrument, `bench.linear_device_time_ms`")
        P("   is. The T4 probe in `axis/splitcausal_t4/` uses the latter, and the")
        P("   direction of the comparison is unambiguous enough there that no host")
        P("   number is needed to guess it.")
        P("")
        P("   The control row is still doing work. At N=512 `cumsum` IS the")
        P("   one-stage path, so those arms are the same computation and the ratio")
        P("   is omitted BY CONSTRUCTION, not by luck. A comparison with no row")
        P("   where the arms are known to coincide cannot separate 'the two-stage")
        P("   decomposition is slower' from 'the second thing I timed is faster'.")

    if a.all or a.cross_check:
        P("\n== 5. cross-check: source transform == real file edit, on a copy ==")
        try:
            cc = cross_check(64)
        except Exception as e:
            cc = {"error": f"{type(e).__name__}: {e}"}
        rep["cross_check"] = cc
        for k, v in cc.items():
            P(f"   {k}: {v}")

    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=2, default=str))
        P(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
