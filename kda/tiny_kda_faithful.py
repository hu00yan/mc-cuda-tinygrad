"""tiny_kda_faithful.py -- arm 3: the KDA kernel's *actual* strategy in TinyGrad,
and every place the abstraction has to be crossed to get there.

WHY THIS FILE IS NOT `tiny_kda.py`
----------------------------------
A hand-written kernel that beats a DSL is usually explained one of two ways:
the compiler failed to find something a cost model could find (a scheduling
problem), or the optimisation lives *below* the abstraction the DSL offers (an
expressiveness problem).  Comparing against a well-written DSL program cannot
tell them apart, because that program was only allowed the DSL's own ideas.
So this file does the other thing: it reproduces the hand-written kernel's
strategy and **records every place the boundary has to be crossed**.  Where the
boundary is, is the result.

THE PIN.  Every `fla/...:NNN` citation is
`fla-org/flash-linear-attention` @ `79d12e8ec0ae72bc7f2189b4244d6532ce8c268c`,
fetched from `raw.githubusercontent.com` (see `notes/25-fla-ut-reference.md` §0
and `kda/SPEC.md` §6.2).  The three files that carry the algorithm:

    fla/ops/kda/chunk_intra.py                  the UT transform, :44-384
    fla/ops/kda/chunk_intra_token_parallel.py   the DEFAULT diagonal builder
    fla/ops/kda/wy_fast.py                      the only two consumers of Akk

WHAT IS NOT USED.  `Tensor.custom_kernel` is forbidden here and appears nowhere
in this file: it takes CUDA source and emits it, so using it would substitute
the reference's own kernel for the experiment and measure nothing.  No Triton,
no CUDA, no PTX, no side-loaded `.cu`.  Every crossing below is either a real
TinyGrad `ContextVar` -- cited at its definition AND at its read -- or a
structural property of the graph (shapes, dtypes, op order), and each one is
recorded in `CROSSINGS` rather than hidden.

THE ONE MECHANISM.  `Crossing` + `LADDER` + `on()` + `context()`.  `on()` is
the only switch for a structural crossing and `context()` is the only place a
knob is set.  `mode="clean"` is the framework's normal vocabulary; walking the
ladder to `mode="faithful"` turns on one crossing at a time, so the
clean-vs-faithful delta decomposes instead of being a single opaque number.

RUN
---
    DEV=CPU .venv/bin/python kda/tiny_kda_faithful.py --test       # vs ref.py
    DEV=CPU .venv/bin/python kda/tiny_kda_faithful.py --crossings  # the map
    DEV=CPU .venv/bin/python kda/tiny_kda_faithful.py --ladder     # the delta
    DEV=CPU .venv/bin/python kda/tiny_kda_faithful.py --all
"""

from __future__ import annotations

import dataclasses
import os
import sys
from typing import Any

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context, ContextVar

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

F32 = dtypes.float32
# 1/ln2.  `fla/ops/utils/constant.py:10` says 1.4426950216, which is the value
# rounded to 10 significant digits; this is float64's.  Both are used only as a
# scale on a cumsum, so the 1.3e-8 relative difference moves the operator by
# about that much and no more (measured, notes/29 §7).
RCP_LN2 = 1.4426950408889634
BC = 16            # chunk_intra.py:825 -- `BC = 16`, unconditional, distinct
                   # from BT.  chunk.py:442-444 validates BT in {32, 64}; this
                   # port accepts any BT that is a multiple of 16.


# ===========================================================================
# 1.  THE CROSSING REGISTRY -- one row per kernel technique
# ===========================================================================
@dataclasses.dataclass(frozen=True)
class Crossing:
    """One kernel technique, and what it takes to have it in TinyGrad.

    kind:
      "expressible"  the framework's normal vocabulary already covers it
      "structure"    it needs the graph to be *shaped* a particular way -- a
                     real TinyGrad lever (shapes, dtypes, op order), not a knob
      "knob"         it needs a ContextVar; the lever and the read are cited
      "inert"        the technique asks for a distinction the framework cannot
                     even represent, and it happens to come out right anyway
      "impossible"   the framework has no representation for it at all
    """
    name: str
    technique: str
    cite: str
    kind: str
    lever: str = ""
    reads: str = ""
    cost: str = ""
    bought: str = ""
    status: str = "MEASURED"


CROSSINGS: tuple[Crossing, ...] = (
    Crossing(
        "bc_blocking",
        "the UT transform is blocked at BC=16: four 16x16 diagonal inverses by "
        "row-wise substitution, then six off-diagonal blocks by a block-Schur "
        "merge, in a depth-3 DAG of 16 dots",
        "chunk_intra.py:274-313, :320-356, :825",
        "expressible",
        lever="tensor shapes only: view the [BT,BT] system as a [NC,BC,NC,BC] "
              "block grid and contract within a block",
        cost="nothing to pay. NC=4, BC=16 is a reshape and a 5-D matmul.",
        bought="the naive form's 63-deep dependent chain becomes four "
               "independent 14-deep chains plus a depth-3 merge",
    ),
    Crossing(
        "rank1_not_dot",
        "the diagonal stage uses `tl.sum(b_a[:,None]*b_Ai, 0)` -- a rank-1 "
        "accumulate with NO tl.dot, chosen so the work stays on the FMA path "
        "instead of paying the MMA layout round-trip",
        "chunk_intra.py:288 (0 occurrences of tl.dot in lines 274-314)",
        "inert",
        lever="spelling: `(a[:,None]*b).sum(-2)` vs `a.reshape(...,1,BC) @ b`",
        reads="tinygrad/mixin/op.py:392 -- `Tensor.dot` IS `(x*w).sum(-1)`. "
              "The two spellings are the same expression in the framework, not "
              "two lowerings of one.",
        cost="nothing to pay and nothing to buy: both spellings emit "
             "byte-identical CUDA (sha1 40d2e93823d1716e, 2204 B).",
        bought="the kernel's *reason* survives by accident. TinyGrad's "
               "tensor-core filter reads tensor SHAPES, not the op: a "
               "[1,16]x[16,16] contraction has an M of 1 that cannot be upcast "
               "so it stays scalar, while a [16,16]x[16,16] one goes to "
               "mma.sync. Same operator, different answer, for a reason the "
               "program never mentions. Measured, notes/29 §4.",
    ),
    Crossing(
        "unrolled_trip_count",
        "both loops are runtime-bounded: `for i in range(2, min(BC, T - i_tc0))` "
        "with T in `do_not_specialize=['T']`, and Stage A's "
        "`for j in range(i_ts, min(i_t+1, min(T, i_ts+BC)))`",
        "chunk_intra.py:43, :285, :290, :296, :302; "
        "chunk_intra_token_parallel.py:32, :117",
        "impossible",
        lever="none. TinyGrad has no loop, so it has no runtime trip count.",
        cost="the port issues the full-chunk count and masks. Stage A: the "
             "kernel runs sum(r+1 for r in 0..15) = 136 iterations per "
             "16-token sub-chunk, the port runs 16x16 = 256, a measured 1.88x. "
             "Substitution: 56 iterations for a tail chunk where the kernel "
             "issues 16. Both are arithmetic the kernel does not do.",
        bought="nothing",
    ),
    Crossing(
        "token_parallel_grid",
        "the DEFAULT path (`safe_gate=False`) builds the [16,16] diagonal "
        "blocks with grid (B*T, cdiv(HV,BH)) -- one program per token, a "
        "<=16-iteration rank-1 loop, no matmul. The `tl.dot` form is the "
        "non-default `safe_gate` path.",
        "chunk_intra_token_parallel.py:54, :117-127, :168; dispatched at "
        "chunk_intra.py:862-875; safe_gate defaults False at chunk.py:47, :210",
        "impossible",
        lever="none. TinyGrad has no grid, no program id, no thread mapping. A "
              "leading axis is a batch dimension and nothing else.",
        cost="the B*T programs become 4*16 = 64 whole-batch reductions that each "
             "re-read k and g_cum. Kernel count for this stage: 64 against 1. "
             "The SET of (token, key) pairs computed is the same, which is why "
             "the result still matches the oracle.",
        bought="nothing",
    ),
    Crossing(
        "allow_tf32_merge",
        "the merge's 16 dots run in tf32 -- 10 explicit mantissa bits -- while "
        "the diagonal stage does not",
        "chunk_intra.py:21-24, :319, :320-356",
        "knob",
        lever="Context(ALLOW_TF32=1)",
        reads="tinygrad/helpers.py:281 (definition). One read: "
              "codegen/opt/postrange.py:226 -- `if self.ren.target.device in "
              "(\"CUDA\",\"NV\") and tc.dtype_in == dtypes.float and not "
              "ALLOW_TF32: continue`.",
        cost="TWO costs, both real. (i) Precision: 10 mantissa bits, and the "
             "merge is on the critical path of the inverse, so the error "
             "propagates into w and u. Modelled in notes/29 §6. (ii) "
             "Reachability: the gate tests the DEVICE NAME, so on the offline "
             "codegen route (`DEV=NULL:CUDA:sm_80`) the string test is false, "
             "ALLOW_TF32 is inert, and the tf32 tensor core is selected at "
             "ALLOW_TF32=0. An offline capture over-reports tensor-core use "
             "unless the target is relabelled. Measured, notes/29 §4.",
        bought="on a target named CUDA, a batched [16,16]x[16,16] goes from "
               "7266 B of scalar FMA to 2575 B containing "
               "`mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`",
    ),
    Crossing(
        "per_op_precision",
        "tf32 for the merge dots, IEEE for the diagonal stage: a PER-DOT "
        "precision choice inside one kernel",
        "chunk_intra.py:22 versus :288",
        "impossible",
        lever="none. ALLOW_TF32 is one global ContextVar read once in the "
              "tensor-core filter; TinyGrad has no per-op precision selector.",
        cost="the kernel's mixed-precision schedule has no representation. The "
             "reachable schedules are all-tf32 or all-IEA, and the port gets "
             "IEEE in the diagonal stage only because a [1,16] contraction is "
             "ineligible for tensor cores -- a coincidence of shape, not a "
             "control.",
        bought="nothing directly. This is the row that says the port's "
               "precision is one switch for the whole operator.",
    ),
    Crossing(
        "subchunk_reference_exp2",
        "each sub-chunk pair uses exp2(g_i - g_n) and exp2(g_n - g_j) with g_n "
        "the cumsum at the first row of the QUERY sub-chunk. The factors "
        "cancel, so no intermediate exceeds 2^(BC|g|) and Gamma is never formed "
        "or divided.",
        "chunk_intra.py:160,162,164 / 179,181,185,189 / 204,206,210,215,220",
        "structure",
        lever="code shape: two exp2 per block with g_n broadcast from the "
              "query sub-chunk's first row, instead of one exp2 of the "
              "chunk-wide difference",
        cost="2 exp2 per (row-block, key-block, key-channel) instead of 1, so "
             "the gram's transcendental count doubles",
        bought="range safety, for free in exact arithmetic: the product is "
               "exp2(g_i - g_j) identically",
    ),
    Crossing(
        "exp2_rcp_ln2",
        "g_cum is stored pre-multiplied by log2(e) and every decay is an exp2, "
        "so the exponential is one hardware ex2.approx",
        "chunk_fwd.py:61-68; chunk_intra.py:21-24; RCP_LN2 in "
        "fla/ops/utils/constant.py:10",
        "expressible",
        lever="Tensor.exp2, with the RCP_LN2 fold inside the caller's cumsum",
        reads="TRANSCENDENTAL (helpers.py:239, read at codegen/__init__.py:372) "
              "gates the sin/cos/tan patterns and cannot reach exp2, so the "
              "knob on the brief's list is inert for this operator.",
        cost="nothing",
        bought="the MUFU.EX2 path, and a structural guarantee: `/` in the "
               "Tensor API is rewritten to `a * (1/b)` "
               "(mixin/elementwise.py:251), so `K / Gamma` cannot be formed by "
               "accident the way it can in float64 NumPy",
    ),
    Crossing(
        "separate_fp32_diag_buffer",
        "the four diagonal [16,16] blocks live in a SEPARATE fp32 buffer Akkd "
        "and are re-read one 16-float row at a time, because they are what the "
        "14-step chain accumulates into and must not be rounded to k.dtype "
        "between the two kernels",
        "chunk_intra.py:834-835, :261-268, :286",
        "expressible",
        lever="per-tensor dtype: keep the diagonal blocks in float32 and cast "
              "only where the kernel casts, at :366-384",
        cost="one extra buffer and one cast. At float32 throughout -- which "
             "this port is -- the distinction is vacuous, and that is the "
             "point: the port cannot lose precision the kernel keeps.",
        bought="the precision decision, and it is a kernel-only concern here",
    ),
    Crossing(
        "registers_only_offdiag",
        "the twelve [16,16] fp32 accumulators and the four diagonal inverses "
        "stay in REGISTERS for the whole kernel and are never written to memory "
        "-- the kernel's own docstring says so at :73",
        "chunk_intra.py:71-78, :122-135",
        "impossible",
        lever="none. NOLOCALS is not a spill control: it raises the upcast "
              "ceiling from 2048 to 240 elements "
              "(codegen/opt/heuristic.py:81) and forces Opt(OptOps.NOLOCALS) "
              "(:163).",
        cost="whether a [B,NT,HV,BC,BC] buffer reaches the schedule is the "
             "scheduler's decision and cannot be requested either way. "
             "Measured by the M2 census, notes/29 §5.",
        bought="nothing",
    ),
    Crossing(
        "head_major_layout",
        "the kernel indexes [B,T,HV,·] with the token axis outermost, and "
        "every one of its dots has the head as the outermost of the two",
        "chunk_intra.py:101-107, wy_fast.py:73-79",
        "structure",
        lever="one permute to [B,NT,HV,·] so the batched contraction's last "
              "two axes are the two it contracts",
        cost="one layout change; the permuted views are what every later "
             "matmul consumes",
        bought="access to `Tensor.dot`, which is "
               "`(x*w).sum(-1)` (mixin/op.py:392) and therefore only works "
               "when the contracted axes are the last two",
    ),
    Crossing(
        "beta_on_operands",
        "the inverted matrix is I + StrictTril(Diag(beta) Z) and the stored Akk "
        "carries NO trailing Diag(beta); beta row-scales the operands instead",
        "chunk_intra.py:115, :234, :243-244, :255-257; wy_fast.py:102, :113, :117",
        "structure",
        lever="code shape: scale k by beta before the gram, and apply beta to "
              "v and to the recomputed w's right-hand operand",
        cost="a silent order-1 error if the paper's Diag(beta) is ALSO applied "
             "to the inverse. Each is correct alone; both is not a rounding "
             "difference.",
        bought="the kernel's graph. Algebraically identical to the paper's M "
               "(kda/SPEC.md §8 row 11), and this port is verified against the "
               "same float64 oracle either way",
    ),
    Crossing(
        "scale_placement",
        "scale multiplies Aqk as it is stored and the inter-chunk term, not q",
        "chunk_intra.py:123, :230, :238-251; inter-chunk factor at "
        "fla/ops/gla/chunk.py:427",
        "structure",
        lever="code shape: multiply Aqk at build time and (Gamma.Q)S at the "
              "output, instead of pre-scaling q",
        cost="changes where the single factor of scale rounds",
        bought="parity with the kernel. The operator is the same either way "
               "(kda/SPEC.md §7.1 dispute 4); the rounding is not",
    ),
    Crossing(
        "dense_akk",
        "the final BT x BT inverse is materialised DENSELY with 6 of its 16 "
        "blocks identically zero, and the consumer does not exploit that",
        "chunk_intra.py:362-384, :831-833; wy_fast.py:91-94, :103, :131",
        "expressible",
        lever="deliberately NOT taken -- exploiting the block sparsity would be "
              "a divergence from the kernel, so both ends of the ladder waste "
              "the same 37.5% of MACs by block count (49.2% counting the "
              "intra-block triangles)",
        cost="the same waste on both sides",
        bought="nothing. This row is a CONTROL: it shows the clean-vs-faithful "
               "delta is not coming from here.",
    ),
    Crossing(
        "no_triangular_solve",
        "a unit-lower-triangular inverse, however it is scheduled",
        "chunk_intra.py:274-356",
        "impossible",
        lever="none. Tensor has no `inverse` and no `solve` in tinygrad 0.14.0 "
              "(`hasattr(Tensor,'inverse')` and `hasattr(Tensor,'solve')` are "
              "both False; only qr and svd exist).",
        cost="whichever substitution schedule the mode selects -- 62 steps in "
             "clean, 4x14 plus a 16-dot merge in faithful",
        bought="nothing. Recorded because it is the reason the UT transform is "
               "a scheduling question at all in this framework.",
    ),
    Crossing(
        "opt_level_absent",
        "the brief lists OPT_LEVEL as a codegen knob",
        "--",
        "impossible",
        lever="the knob does not exist in this version",
        reads="absent. `tinygrad.helpers.ContextVar._cache` has no 'OPT_LEVEL'; "
              "the registered keys are the first argument to ContextVar(...) "
              "in helpers.py. The nearest neighbours are NOOPT and the per-op "
              "OptOps.",
        cost="cannot be swept. Recorded as an absence, not as a null result.",
        bought="nothing",
    ),
    Crossing(
        "use_tc_spelling",
        "the brief lists USE_TC",
        "--",
        "impossible",
        lever="the registered ContextVar key is 'TC', not 'USE_TC'",
        reads="helpers.py:238 -- `USE_TC, TC_SELECT, TC_OPT = "
              "ContextVar(\"TC\", 1), ContextVar(\"TC_SELECT\", -1), "
              "ContextVar(\"TC_OPT\", 0)`. `Context.__enter__` resolves each "
              "key against ContextVar._cache, so `Context(USE_TC=0)` raises "
              "KeyError and `USE_TC=0` in the environment is read by nothing.",
        cost="a sweep written as Context(USE_TC=0) would report 'no effect' "
             "about a knob that was never set -- a true-sounding sentence "
             "about a null",
        bought="nothing",
    ),
)
CROSSING_BY_NAME = {c.name: c for c in CROSSINGS}


# ===========================================================================
# 2.  THE LADDER -- one crossing at a time, so the delta decomposes
# ===========================================================================
# flag -> which crossing it is.  There is deliberately NO flag for
# `rank1_not_dot`: the framework cannot see that distinction at all, so there is
# nothing to switch.  A flag for it would be a crossing the ladder can turn on
# and off that has no effect, which is the exact shape of a silent null.
FLAG_TO_CROSSING = {
    "beta_left": "beta_on_operands",
    "scale_on_Aqk": "scale_placement",
    "tf32": "allow_tf32_merge",
    "bc_blocking": "bc_blocking",
    "subchunk_ref": "subchunk_reference_exp2",
    "token_parallel": "token_parallel_grid",
}

LADDER: tuple[tuple[str, dict[str, bool], str], ...] = (
    ("clean", {},
     "the framework's normal vocabulary: dense [C,C] gram, one 62-step "
     "row-wise chain, the paper's M, no knobs"),
    ("beta_left", {"beta_left": True, "scale_on_Aqk": True},
     "+ the kernel's beta and scale placement; same algorithm"),
    ("tf32", {"beta_left": True, "scale_on_Aqk": True, "tf32": True},
     "+ the kernel's tf32 precision for every reachable tensor core"),
    ("blocked", {"beta_left": True, "scale_on_Aqk": True, "tf32": True,
                 "bc_blocking": True},
     "+ BC=16 blocking: four 14-deep chains and a 16-dot depth-3 merge"),
    ("faithful", {"beta_left": True, "scale_on_Aqk": True, "tf32": True,
                  "bc_blocking": True, "subchunk_ref": True,
                  "token_parallel": True},
     "+ the sub-chunk decay references and the token-parallel diagonal "
     "builder. Equals chunk_intra.py plus chunk_intra_token_parallel.py."),
)
MODES = tuple(m for m, _, _ in LADDER)
MODE_DOC = {m: d for m, _, d in LADDER}


def _flags(mode: str) -> dict[str, bool]:
    for m, f, _ in LADDER:
        if m == mode:
            return f
    raise KeyError(f"unknown mode {mode!r}; modes are {MODES}")


def on(flag: str, mode: str) -> bool:
    """The ONLY switch for a structural crossing. `flag` is a FLAG_TO_CROSSING
    key, so every call site names a row of the boundary map."""
    assert flag in FLAG_TO_CROSSING, f"{flag!r} is not a declared crossing flag"
    return bool(_flags(mode).get(flag, False))


def crossings_active(mode: str) -> list[Crossing]:
    return [CROSSING_BY_NAME[FLAG_TO_CROSSING[f]] for f in _flags(mode)]


def context(mode: str) -> Context:
    """The ONLY place a TinyGrad knob is set.

    `ALLOW_TF32` is the one knob any mode turns on, and it is the one the
    kernel's merge needs.  It is global: TinyGrad has no per-op precision
    selector, so `faithful` gets tf32 wherever a tensor core is reachable and
    not only on the 16 merge dots.  That is the `per_op_precision` row.
    """
    cfg: dict[str, Any] = {"ALLOW_TF32": 1} if on("tf32", mode) else {}
    for k in cfg:
        assert k in ContextVar._cache, (
            f"{k!r} is not a registered ContextVar key. Registered keys are the "
            f"first argument to ContextVar(...) in tinygrad/helpers.py, and "
            f"they are not always the Python name -- USE_TC is registered as "
            f"'TC'. Refusing to run.")
    return Context(**cfg)


# ===========================================================================
# 3.  small helpers
# ===========================================================================
def _T(x) -> Tensor:
    """numpy constant -> Tensor.  `Tensor.ufix` refuses a bare ndarray
    (mixin/elementwise.py:19 -> uop/ops.py:589), so every mask that meets an
    operator has to be wrapped.  Same trap kda/tiny_kda.py:162-165 records.
    dtypes.bool is preserved because `Tensor.where` requires it
    (uop/ops.py:147)."""
    a = np.asarray(x)
    return Tensor(a if a.dtype == np.bool_ else a.astype(np.float32))


def _outmask(keep: np.ndarray, bcast: tuple[int, ...]) -> Tensor:
    """`0` where `keep`, `-1e4` elsewhere, ready to add to a log difference.

    The sign matters and getting it backwards is silent: the additive-mask
    idiom is `(keep - 1) * 1e4`, NOT `keep * -1e4`.  With the latter the mask
    is large exactly on the entries you meant to KEEP, so every kept decay
    underflows to zero and the gram comes out with its strict-lower /
    strict-upper halves swapped.  The idiom is kda/tiny_kda.py:137 and
    kda/ref.py:273.
    """
    return _T(((keep - 1.0) * 1e4).reshape(bcast))


def _onehot(i: int, n: int, bcast: tuple[int, ...]) -> np.ndarray:
    return (np.arange(n) == i).astype(np.float32).reshape(bcast)


def _strict_lower(c: int) -> np.ndarray:
    """[c,c] 0/1, strictly lower. `chunk_intra.py:275` is `o_i[:,None] > o_i[None,:]`
    and `chunk_intra.py:768` is the same for the safe_gate path: Akk is `>`-masked
    and Aqk is `>=`-masked (:776-781), because Akk's diagonal becomes the unit
    diagonal at :308."""
    return (np.arange(c)[:, None] > np.arange(c)[None, :]).astype(np.float32)


def _causal(c: int) -> np.ndarray:
    """[c,c] 0/1, causal with the diagonal INCLUDED."""
    return (np.arange(c)[:, None] >= np.arange(c)[None, :]).astype(np.float32)


def _grid_put(grid: Tensor, bi: int, bj: int, block: Tensor,
              col: int | None = None) -> Tensor:
    """Write `block` into the block grid `grid` ([B,NT,HV,NC,BC,NC,BC]) at
    block (bi, bj), optionally into just one column of that block.

    Mirrors the kernel's ten separate `tl.store`s at chunk_intra.py:362-384, and
    for Stage A the one-column form of chunk_intra_token_parallel.py:127, whose
    store address is `Akk + i_t*HV*BC + (j - i_ts)`.

    `col` is not optional decoration.  A length-1 axis in a reshape BROADCASTS;
    it does not select.  Passing a [.., BC, 1] block with no `col` writes the
    same column into every column of the block, and the last iteration wins --
    which is a plausible-looking lower-triangular matrix that is not the gram.
    The within-block one-hot has to be in the selection, and this is the only
    place in the file where a block is addressed by a sub-index.
    """
    NC, BCg = grid.shape[3], grid.shape[4]
    sel = _onehot(bi, NC, (1, 1, 1, NC, 1, 1, 1)) * \
        _onehot(bj, NC, (1, 1, 1, 1, 1, NC, 1))
    if col is not None:
        sel = sel * _onehot(col, BCg, (1, 1, 1, 1, 1, 1, BCg))
    # the block's ROW axis lands on the grid's BC axis (4), not on the NC axis
    # (3) -- the grid is (NC, BC, NC, BC), not (BC, NC, BC, NC).
    return grid + _T(sel) * block.reshape(*block.shape[:3], 1, block.shape[3],
                                          1, block.shape[4])


def _prep(q, k, v, g, beta, scale):
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    g = np.asarray(g, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    assert k.shape == q.shape, (k.shape, q.shape)
    assert g.shape == (B, T, HV, K), (g.shape, (B, T, HV, K))
    assert beta.shape == (B, T, HV), beta.shape
    assert HV % H == 0, f"HV={HV} not divisible by H={H}"
    G = HV // H
    if scale is None:
        scale = K ** -0.5
    # GVA: q and k are shared by G = HV//H adjacent value heads; g is not.
    return (np.repeat(q, G, axis=2), np.repeat(k, G, axis=2), v, g, beta,
            B, T, H, HV, K, V, float(scale))


# ===========================================================================
# 4.  STAGE 1 -- the chunk-local cumsum, in log2
# ===========================================================================
def _gcum2(g: np.ndarray, NT: int, BT: int, HV: int, K: int) -> Tensor:
    """g_cum * log2(e), reset every chunk, in [B,NT,HV,BT,K] (head-major).

    `chunk_fwd.py:61-68` calls `chunk_local_cumsum(g, scale=RCP_LN2,
    chunk_size=BT)`.  The scale is folded into the cumsum rather than applied
    after it, which is what the kernel does and what makes every downstream
    `exp2` a single ex2.approx.
    """
    B = g.shape[0]
    return (Tensor(g.reshape(B, NT, BT, HV, K) * np.float32(RCP_LN2))
            .cumsum(axis=2)
            .permute(0, 1, 3, 2, 4))          # -> [B,NT,HV,BT,K]


# ===========================================================================
# 5.  the raw strictly-lower system S = StrictTril(Diag(beta) Z), as a grid
# ===========================================================================
def _system_dense(gc, kc, qc, bc, B, NT, HV, K, BT) -> tuple[Tensor, Tensor]:
    """The framework's normal vocabulary: one dense [BT,BT] gram per
    (batch, chunk, head), built from the [BT,BT,K] decay tensor and contracted
    over the key channel.

    This is what you write if you are writing the operator FOR TinyGrad.  It is
    NOT what chunk_intra.py does, and the difference is the whole subject of
    this file.  It materialises [B,NT,HV,BT,BT,K] floats -- the single largest
    item in the operator -- and the faithful path never builds it at all.

    `beta` row-scales k here in BOTH modes.  That is not a crossing: the matrix
    being inverted is `I + StrictTril(Diag(beta) Z)` however beta is finally
    accounted for (kda/ref.py:352-356, kda/SPEC.md §3.3), and the substitution
    is fed the NEGATED block (kda/ref.py:355, chunk_intra.py:278).  What the
    `beta_on_operands` crossing decides is only where the SECOND factor of beta
    lives: on the right of the inverse, or on the operands of w and u.
    """
    gi = gc.reshape(B, NT, HV, BT, 1, K)
    gj = gc.reshape(B, NT, HV, 1, BT, K)
    kp = kc.reshape(B, NT, HV, BT, 1, K) * bc.reshape(B, NT, HV, BT, 1, 1)
    kj = kc.reshape(B, NT, HV, 1, BT, K)
    m_s = _outmask(_strict_lower(BT), (1, 1, 1, BT, BT, 1))
    m_c = _outmask(_causal(BT), (1, 1, 1, BT, BT, 1))
    # The mask multiplies the LOG DIFFERENCE, never the exponential: masking
    # after the exp still evaluates exp() on entries whose exponent is large
    # and positive (kda/tiny_kda.py:130-134, kda/SPEC.md §4.2).
    Sfull = ((gi - gj + m_s).exp2() * kp * kj).sum(-1)       # [B,NT,HV,BT,BT]
    Afull = ((gi - gj + m_c).exp2() * qc.reshape(B, NT, HV, BT, 1, K)
             * kj).sum(-1)
    return Sfull, Afull


def _system_token_parallel(gc, kc, qc, bc, B, NT, HV, K, BT, BCm, BK, scale):
    """chunk_intra_token_parallel.py:117-127 (the diagonal blocks) plus
    chunk_intra.py:140-257 (the six off-diagonal blocks), the DEFAULT path.

    Stage A, per token t in sub-chunk s, loops
        `for j in range(i_ts, min(i_t+1, min(T, i_ts+BC)))`
    and each iteration is a rank-1 accumulate:
        b_kgj = k[j] * exp2(g[t] - g[j])                          (:122)
        b_Aqk = sum(q[t] * b_kgj, axis=1) * scale                 (:123)
        b_Akk = sum(beta[t]*k[t] * b_kgj, axis=1) * (j < t)      (:115, :124)
    writing column `j - i_ts`.

    The trip count is per-token and runtime-valued, so it is unreproducible
    here.  The port issues all BCm columns for all BCm tokens and masks,
    which is 256 against the kernel's 136 per sub-chunk.  See the
    `unrolled_trip_count` row.

    Stage B builds the six off-diagonal [BC,BC] blocks: twelve dots of
    [BC,BK]x[BK,BC] per `i_k` (:166-167, :186-192, :212-223), twelve [16,16]
    fp32 accumulators (:122-135), a beta row-scale on Akk AFTER the dots
    (:234, :243-244, :255-257), and the sub-chunk decay reference g_n at the
    first row of the QUERY sub-chunk (:160, :179, :204) whose two exp2 factors
    cancel.
    """
    NCm = BT // BCm
    g5 = gc.reshape(B, NT, HV, BT, K)
    k5 = kc.reshape(B, NT, HV, BT, K)
    q5 = qc.reshape(B, NT, HV, BT, K)
    b5 = bc.reshape(B, NT, HV, BT)

    Sg = Tensor.zeros(B, NT, HV, NCm, BCm, NCm, BCm, dtype=F32)
    Ag = Tensor.zeros(B, NT, HV, NCm, BCm, NCm, BCm, dtype=F32)

    # ---- Stage A: the [BC,BC] diagonal blocks, NCm*BCm rank-1 accumulates ----
    for s in range(NCm):
        r0, r1 = s * BCm, (s + 1) * BCm
        gs, ks, qs = g5[:, :, :, r0:r1], k5[:, :, :, r0:r1], q5[:, :, :, r0:r1]
        bs = b5[:, :, :, r0:r1].unsqueeze(-1)                        # beta_t
        # The two masks are [row] vectors, NOT [row, col] matrices.  That is
        # the entire content of the token-parallel path: `j` is a loop index,
        # so `j < i_t` (:124) is a comparison of two scalars, and the [BC,BC]
        # block is never materialised -- one program writes one column.  A
        # framework without a program has to apply the mask on the row axis
        # before the contraction, and the [C,C,K] decay tensor that
        # kda/tiny_kda.py builds never appears in this graph at all.
        for c in range(BCm):
            gj = gs[:, :, :, c:c + 1]                                # g[j]
            kj = ks[:, :, :, c:c + 1]                                # k[j]
            # :122  b_kgj = b_kj * exp2(b_g - b_gj)  --  b_kj is the KEY's k
            # (j is the loop variable) and b_g is the TOKEN's own g.  Swapping
            # them gives a gram of k against itself that is still
            # lower-triangular and still the right shape, so nothing but the
            # oracle catches it.
            kgj = kj * (gs - gj).exp2()
            # The two masks are [row] vectors, NOT [row, col] matrices.  That
            # is the entire content of the token-parallel path: `j` is a loop
            # index, so `j < i_t` (:124) compares two scalars, and the [BC,BC]
            # block is never materialised -- one program writes one column.
            # A framework without a program applies the mask on the row axis
            # before the contraction, and the [C,C,K] decay tensor that
            # kda/tiny_kda.py builds never appears in this graph at all.
            m_le = _T((np.arange(BCm) >= c).reshape(1, 1, 1, BCm, 1))  # j <= t
            m_lt = _T((np.arange(BCm) > c).reshape(1, 1, 1, BCm, 1))   # j <  t
            aq = ((qs * kgj) * m_le).sum(-1) * scale                  # :123
            ak = ((bs * ks * kgj) * m_lt).sum(-1)                     # :115, :124
            # one column of the [BC,BC] block per iteration, which is what
            # the kernel's `Akk + i_t*HV*BC + (j - i_ts)` store address is
            Ag = _grid_put(Ag, s, s, aq.reshape(B, NT, HV, BCm, 1), col=c)
            Sg = _grid_put(Sg, s, s, ak.reshape(B, NT, HV, BCm, 1), col=c)

    # ---- Stage B: the six off-diagonal blocks ----
    for i in range(NCm):
        for j in range(i):
            gi = g5[:, :, :, i * BCm:(i + 1) * BCm]
            gj = g5[:, :, :, j * BCm:(j + 1) * BCm]
            ki = k5[:, :, :, i * BCm:(i + 1) * BCm]
            kj = k5[:, :, :, j * BCm:(j + 1) * BCm]
            qi = q5[:, :, :, i * BCm:(i + 1) * BCm]
            gn = gi[:, :, :, 0:1]                       # :160/:179/:204
            accS = Tensor.zeros(B, NT, HV, BCm, BCm, dtype=F32)
            accA = Tensor.zeros(B, NT, HV, BCm, BCm, dtype=F32)
            for k0 in range(0, K, BK):                  # the `i_k` loop, :140
                k1 = min(k0 + BK, K)
                di = gi[..., k0:k1] - gn[..., k0:k1]    # :162, :181, :206
                dj = gn[..., k0:k1] - gj[..., k0:k1]    # :164, :185, :210
                lhsq = qi[..., k0:k1] * di.exp2()        # :166, :186, :212
                rhsk = kj[..., k0:k1] * dj.exp2()        # :164, :185, :210
                accA = accA + lhsq.matmul(rhsk.transpose(-1, -2))   # Aqk_ij
                accS = accS + (ki[..., k0:k1] * di.exp2()).matmul(
                    rhsk.transpose(-1, -2))                          # Akk_ij
            accS = accS * b5[:, :, :, i * BCm:(i + 1) * BCm].unsqueeze(-1)  # :234
            Sg = _grid_put(Sg, i, j, accS)
            Ag = _grid_put(Ag, i, j, accA * scale)                    # :230
    return Sg, Ag


# ===========================================================================
# 6.  STAGES C+D -- the substitution and the merge
# ===========================================================================
def _diag_inverse(Sg: Tensor, NCm: int, BCm: int, rank1: bool) -> list[Tensor]:
    """chunk_intra.py:274-313, one loop per sub-chunk.

        b_Ai = -tl.where(m_A, b_Ai, 0)                     (:278, NEGATE)
        for i in range(2, min(BC, T - i_tc0)):             (:285, RUNTIME BOUND)
            b_a = -tl.load(Akkd + (i_tc0+i)*HV*BC + o_i)   (:286, RE-LOAD row i)
            b_a = tl.where(o_i < i, b_a, 0.)               (:287)
            b_a += tl.sum(b_a[:, None] * b_Ai, 0)         (:288, NO tl.dot)
            b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)  (:289)
        b_Ai += m_I                                       (:308, + I)

    The loop starts at 2, not 0 or 1: row 0 is the identity and row 1's only
    off-diagonal entry is already right, because the zero column makes the
    update a no-op.  Trip count BC-2.

    The re-loaded row comes from the ORIGINAL raw block, not the progressively
    updated one -- that is what makes the whole thing a correct forward
    substitution rather than an iterative refinement.  `rank1` picks the
    kernel's spelling (a broadcast-multiply-reduce) over the framework's
    (a [1,BC]x[BC,BC] contraction).  Per the `rank1_not_dot` row they emit the
    same CUDA, and the parameter is kept only so the claim is checkable.
    """
    B, NT, HV = Sg.shape[0], Sg.shape[1], Sg.shape[2]
    out: list[Tensor] = []
    eye = _T(np.eye(BCm, dtype=np.float32).reshape(1, 1, 1, BCm, BCm))
    low = _T(_strict_lower(BCm).reshape(1, 1, 1, BCm, BCm))
    for k in range(NCm):
        R = Sg[:, :, :, k, :, k, :]                       # the raw block
        Ai = -R * low                                     # :278, NEGATE
        for i in range(2, BCm):                           # BCm-2 iterations
            a = -R[:, :, :, i, :]                        # :286, the re-load
            a = a * _T((np.arange(BCm) < i).astype(np.float32)
                       .reshape(1, 1, 1, 1, BCm))       # :287
            av = a.reshape(B, NT, HV, 1, BCm)
            # `b_a00 += tl.sum(...)` -- the update is the re-loaded row PLUS the
            # contraction, and the row is the ORIGINAL one, not the accumulated
            # one.  Dropping the `+ a` here is silent: it still produces a
            # lower-triangular matrix, just the wrong one.
            #
            # `.sum(-2)` DROPS the axis, so the two spellings below come back
            # with different ranks (4-D and 5-D) and have to be normalised
            # before they meet.  With B=NT=HV=1 the misalignment is invisible:
            # (1,1,1,16) left-padded to five axes is (1,1,1,1,16) and the add
            # happens to broadcast.  Any real batch finds it.
            red = ((av.reshape(B, NT, HV, BCm, 1) * Ai).sum(-2) if rank1
                   else av.matmul(Ai))
            upd = av + red.reshape(B, NT, HV, 1, BCm)       # :288
            Ai = Tensor.where(
                _T((np.arange(BCm) == i).reshape(1, 1, 1, BCm, 1)), upd, Ai)  # :289
        out.append(Ai + eye)                              # :308
    return out


def _merge(D: list[Tensor], Sg: Tensor, NCm: int) -> list[Tensor]:
    """chunk_intra.py:320-356, block-Schur, 16 dots in a depth-3 DAG.

        b_Ai10 = -dot(dot(Ai11, Akk10), Ai00)                        2 dots
        b_Ai21 = -dot(dot(Ai22, Akk21), Ai11)                        2
        b_Ai32 = -dot(dot(Ai33, Akk32), Ai22)                        2
        b_Ai20 = -dot(Ai22, dot(Akk20,Ai00) + dot(Akk21,Ai10))       3
        b_Ai31 = -dot(Ai33, dot(Akk31,Ai11) + dot(Akk32,Ai21))       3
        b_Ai30 = -dot(Ai33, dot(Akk30,Ai00) + dot(Akk31,Ai10)
                          + dot(Akk32,Ai20))                         4
                                                            -----------
                                                                   16

    `SOLVE_TRIL_DOT_PRECISION` is tf32 when the device supports it
    (chunk_intra.py:21-24, comment at :319: "we used tf32 to maintain matrix
    inverse's precision whenever possible").  That is the one knob crossing
    in this file, and it is global where the kernel's is per-dot.
    """
    if NCm == 1:
        return []                                          # nothing to merge
    R = lambda i, j: Sg[:, :, :, i, :, j, :]               # noqa: E731
    O: dict[tuple[int, int], Tensor] = {}
    # depth 1: the sub-diagonal, 2 dots each (chunk_intra.py:320, :327, :339)
    for i in range(1, NCm):
        O[(i, i - 1)] = -(D[i].matmul(R(i, i - 1))).matmul(D[i - 1])
    # depths 2..NC-1: 3, 4, ... dots each (:332, :344, :350)
    for d in range(2, NCm):
        for i in range(d, NCm):
            acc = None
            for m in range(1, d + 1):
                term = R(i, i - d + m - 1).matmul(
                    D[i - d] if m == 1 else O[(i - d + m - 1, i - d)])
                acc = term if acc is None else acc + term
            O[(i, i - d)] = -D[i].matmul(acc)
    return [O[(i, j)] for i in range(NCm) for j in range(i)]


# ===========================================================================
# 7.  the operator
# ===========================================================================
def kda_chunk(q, k, v, g, beta, scale=None, initial_state=None,
              chunk_size=64, mode="faithful", BK=None, return_parts=False):
    """KDA, chunked, in the style selected by `mode`.  All inputs numpy in,
    (o, S) numpy out.  The oracle is kda/ref.py's float64 kda_recurrent.

    Shapes: q,k [B,T,H,K]; v [B,T,HV,V]; g [B,T,HV,K] log space; beta [B,T,HV].

    The pipeline, in the kernel's order:
      1  g_cum = chunk-local cumsum of g, pre-scaled by log2(e)   (chunk_fwd.py:61)
      2  the [BC,BC] diagonal blocks, token-parallel               (tp:117-127)
      3  the six off-diagonal blocks, 12 dots per key chunk        (ci:140-223)
      4  4x14 row-wise substitution on the diagonals, no tl.dot     (ci:274-313)
      5  16-dot depth-3 block-Schur merge, tf32                    (ci:320-356)
      6  assemble the dense BT x BT Akk                             (ci:362-384)
      7  w = Akk @ (k*beta*exp2(g_cum)), u = Akk @ (v*beta)         (wy:102,113,117)
      8  the NT-step inter-chunk scan                               (chunk_delta_h)
      9  o = (exp2(g_cum)*Q) @ S + Aqk @ v_new                      (gla/chunk.py:427)
    """
    (q, k, v, g, beta, B, T, H, HV, K, V, scale) = _prep(q, k, v, g, beta, scale)
    BT = chunk_size
    assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"
    NT = T // BT
    if on("bc_blocking", mode) or on("token_parallel", mode):
        # BC=16 is a property of the kernel's STRATEGY, not of the token-parallel
        # spelling of Stage A, so it turns on independently.  (Wiring it to
        # `token_parallel` alone makes the `bc_blocking` crossing a no-op and
        # the ladder's third rung identical to its second -- which is exactly
        # the kind of silent null this file is supposed to make impossible.)
        BCm = BC
        assert BT % BCm == 0, (
            f"the kernel's strategy hardcodes BC=16 (chunk_intra.py:825) and "
            f"validates chunk_size in {{32, 64}} (chunk.py:442-444); "
            f"chunk_size={BT} is not a multiple of BC=16")
    else:
        BCm = BT                                   # one block: the clean form
    if BK is None:
        BK = 32 if K >= 32 else K
    NCm = BT // BCm

    with context(mode):
        # ---- 1. the chunk-local cumsum, in log2, head-major -------------
        gc = _gcum2(g, NT, BT, HV, K)                       # [B,NT,HV,BT,K]
        kc = Tensor(k.reshape(B, NT, BT, HV, K).transpose(0, 1, 3, 2, 4))
        qc = Tensor(q.reshape(B, NT, BT, HV, K).transpose(0, 1, 3, 2, 4))
        vc = Tensor(v.reshape(B, NT, BT, HV, V).transpose(0, 1, 3, 2, 4))
        bc = Tensor(beta.reshape(B, NT, BT, HV).transpose(0, 1, 3, 2))
        if not on("scale_on_Aqk", mode):
            qc = qc * scale            # clean: scale q up front, as ref.py does

        # ---- 2/3. the raw strictly-lower system, as a block grid ----------
        if on("token_parallel", mode):
            Sg, Ag = _system_token_parallel(
                gc, kc, qc, bc, B, NT, HV, K, BT, BCm, BK, scale)
        else:
            Sfull, Afull = _system_dense(gc, kc, qc, bc, B, NT, HV, K, BT)
            if on("scale_on_Aqk", mode):
                Afull = Afull * scale          # the kernel's placement, :230
            Sg = Sfull.reshape(B, NT, HV, NCm, BCm, NCm, BCm)
            Ag = Afull.reshape(B, NT, HV, NCm, BCm, NCm, BCm)

        # ---- 4. the substitution -----------------------------------------
        # `rank1` is passed as a CONSTANT, not read from the ladder: the two
        # spellings are the same expression (mixin/op.py:392) and the
        # `rank1_not_dot` row records that as a finding rather than a mode.
        D = _diag_inverse(Sg, NCm, BCm, rank1=True)

        # ---- 5. the merge -----------------------------------------------
        O = _merge(D, Sg, NCm)

        # ---- 6. assemble the dense BT x BT Akk, 10 blocks, rest zero ------
        Akkg = Tensor.zeros(B, NT, HV, NCm, BCm, NCm, BCm, dtype=F32)
        for k in range(NCm):
            Akkg = _grid_put(Akkg, k, k, D[k])
        for n, (i, j) in enumerate([(i, j) for i in range(NCm) for j in range(i)]):
            Akkg = _grid_put(Akkg, i, j, O[n])
        # (NC,BC) on both axes is contiguous, so this reshape is free and it is
        # the kernel's dense tile: chunk_intra.py:831-833 allocates
        # Akk = zeros(B,T,HV,BT) because the kernel writes only the lower blocks.
        Akk = Akkg.reshape(B, NT, HV, BT, BT)
        if not on("beta_left", mode):
            # clean: the paper's M = (I + StrictTril(Diag(beta)Z))^-1 Diag(beta)
            # (kda/ref.py:355-356).  The column scaling goes on the FLATTENED
            # BT axis, which is why it is applied after the reshape and not on
            # the block grid: the grid splits the column axis into (NC, BC) and
            # beta_t varies along the whole of it.
            Akk = Akk * bc.reshape(B, NT, HV, 1, BT)

        # ---- 7. w and u: the only two consumers (wy_fast.py:103, :131) ----
        # beta lives on the OPERANDS here exactly when it is NOT on the inverse.
        # Applying it in both places is the order-1 double-count the
        # `beta_on_operands` row warns about.
        bmul = bc.unsqueeze(-1) if on("beta_left", mode) else 1.0
        kk = kc * bmul * gc.exp2()                          # k*beta*Gamma
        vb = vc * bmul                                      # v*beta
        w = Akk.matmul(kk)                                   # [B,NT,HV,BT,K]
        u = Akk.matmul(vb)                                   # [B,NT,HV,BT,V]
        Aqk = Ag.reshape(B, NT, HV, BT, BT)

        # ---- 8. the NT-step inter-chunk scan ---------------------------
        g_last = gc[:, :, :, BT - 1, :]                      # [B,NT,HV,K]
        kg = kc * (g_last.unsqueeze(3) - gc).exp2()          # Gamma^{i->C} . K
        S = (Tensor(np.zeros((B, HV, K, V), dtype=np.float32)) if initial_state is None
             else Tensor(np.asarray(initial_state, dtype=np.float32)))
        parts = []
        for n in range(NT):
            v_new = u[:, n] - w[:, n].matmul(S)              # 7a
            qg = qc[:, n] * gc[:, n].exp2()                  # Gamma . Q
            if on("scale_on_Aqk", mode):
                o_n = (qg.matmul(S)) * scale + Aqk[:, n].matmul(v_new)
            else:
                o_n = qg.matmul(S) + Aqk[:, n].matmul(v_new)
            parts.append(o_n)
            S = S * g_last[:, n].exp2().unsqueeze(-1) + \
                kg[:, n].transpose(-1, -2).matmul(v_new)    # 7d

        o = Tensor.stack(*parts)                            # [NT,B,HV,BT,V]
        # ref.py:396 takes o from [B,NT,HV,C,V] to [B,NT,C,HV,V] BEFORE the
        # reshape, so the flat token index is n*BT + t. Permuting the other way
        # interleaves the head axis into the token index and raises nothing
        # (the trap kda/tiny_kda.py:244-249 documents).
        o = o.permute(1, 0, 3, 2, 4).reshape(B, T, HV, V)
        parts_out = {"gc": gc, "Akk": Akk, "Aqk": Aqk, "w": w, "u": u}
    if return_parts:
        return o.numpy(), S.numpy(), parts_out
    return o.numpy(), S.numpy()


# Alias with the mode in the name, so a test file can import both ends.
def kda_clean(*a, **k):
    k.pop("mode", None)
    return kda_chunk(*a, mode="clean", **k)


def kda_faithful(*a, **k):
    k.pop("mode", None)
    return kda_chunk(*a, mode="faithful", **k)


# ===========================================================================
# 8.  the boundary map
# ===========================================================================
def print_crossings() -> None:
    w = 78
    print("=" * w)
    print("BOUNDARY MAP -- one row per KDA kernel technique")
    print("=" * w)
    for c in CROSSINGS:
        print(f"\n  {c.name}   [{c.kind.upper()}]   {c.status}")
        print(f"    kernel does : {c.technique}")
        print(f"    cite        : {c.cite}")
        if c.lever:
            print(f"    lever       : {c.lever}")
        if c.reads:
            print(f"    read at     : {c.reads}")
        if c.cost:
            print(f"    costs       : {c.cost}")
        if c.bought:
            print(f"    buys        : {c.bought}")
    print()
    print(f"  {len(CROSSINGS)} rows; "
          f"{sum(1 for c in CROSSINGS if c.kind == 'impossible')} IMPOSSIBLE, "
          f"{sum(1 for c in CROSSINGS if c.kind == 'knob')} needs a knob, "
          f"{sum(1 for c in CROSSINGS if c.kind == 'inert')} INERT (the "
          f"framework cannot see the distinction), "
          f"{sum(1 for c in CROSSINGS if c.kind == 'structure')} structural, "
          f"{sum(1 for c in CROSSINGS if c.kind == 'expressible')} free.")
    print("=" * w)


def print_ladder() -> None:
    print("=" * 78)
    print("THE LADDER -- one crossing at a time")
    print("=" * 78)
    for m, f, d in LADDER:
        cs = ", ".join(CROSSING_BY_NAME[FLAG_TO_CROSSING[x]].name for x in f) or "-"
        print(f"  {m:11s} {d}")
        print(f"  {'':11s} crossings on: {cs}")
    print("=" * 78)


# ===========================================================================
# 9.  the correctness gate
# ===========================================================================
def relerr(a, b) -> float:
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def run_test() -> int:
    """The oracle is kda/ref.py's kda_recurrent -- the per-token definition of
    KDA, not another chunked implementation.  The tolerance convention is
    kda/test_tiny.py's: 1e-5 worst-case relative, float32 against float64."""
    import ref
    import test_ref as T

    TOL = 1e-5
    # kda/test_tiny.py's CASES, verbatim, plus C=64 cases.  A case is runnable
    # in faithful mode only when chunk_size is a multiple of BC=16, because
    # that is a property of the kernel's strategy and not of the framework.
    CASES = list(T.__dict__.get("CASES", [])) or [
        (2, 16, 2, 2, 4, 8, 8), (2, 16, 2, 2, 4, 8, 16), (2, 32, 2, 2, 8, 8, 8),
        (1, 32, 1, 1, 8, 8, 16), (3, 24, 3, 3, 6, 10, 8), (1, 64, 2, 2, 8, 8, 16),
        (2, 64, 2, 2, 8, 4, 32), (2, 16, 1, 3, 4, 4, 8), (2, 16, 1, 4, 8, 8, 16),
    ]
    CASES = list(CASES) + [(2, 64, 2, 2, 8, 8, 64), (1, 64, 1, 1, 8, 8, 64),
                           (2, 32, 2, 2, 8, 8, 32), (1, 32, 1, 1, 8, 8, 16)]

    print("=" * 78)
    print("tiny_kda_faithful vs kda/ref.py  (float32 port, float64 oracle)")
    print(f"  tolerance {TOL:.0e} relative, worst case over |ref| -- the "
          f"kda/test_tiny.py convention")
    print("=" * 78)
    worst = {m: 0.0 for m in MODES}
    skipped = {m: 0 for m in MODES}
    per_mode_detail = {m: [] for m in MODES}
    for i, (B, Tt, H, HV, K, V, C) in enumerate(CASES):
        q, k, v, g, beta = T.rnd(B, Tt, H, HV, K, V, seed=i)
        o_ref, S_ref = ref.kda_recurrent(q, k, v, g, beta)
        h0 = np.random.default_rng(500 + i).normal(size=(B, HV, K, V))
        o_h0, S_h0 = ref.kda_recurrent(q, k, v, g, beta, initial_state=h0)
        for m in MODES:
            # The BLOCKED strategy needs chunk_size to be a multiple of BC=16,
            # which is a property of the kernel and not of the framework:
            # chunk_intra.py:825 hardcodes it and chunk.py:442-444 validates
            # chunk_size in {32, 64}.  `clean` has no such restriction, which
            # is itself a row of the map -- the framework's normal vocabulary
            # reaches a UT transform at any chunk size and the faithful one
            # does not.
            if on("bc_blocking", m) and C % BC != 0:
                skipped[m] += 1
                continue
            o, S = kda_chunk(q, k, v, g, beta, chunk_size=C, mode=m)
            o2, S2 = kda_chunk(q, k, v, g, beta, chunk_size=C, mode=m,
                               initial_state=h0)
            e = max(relerr(o, o_ref), relerr(S, S_ref),
                    relerr(o2, o_h0), relerr(S2, S_h0))
            worst[m] = max(worst[m], e)
            per_mode_detail[m].append((i, C, e))
            flag = "" if e < TOL else "   <-- OVER TOLERANCE"
            if e >= TOL:
                print(f"    case{i:2d} C={C:2d} {m:11s} relerr {e:.3e}{flag}")
    for m in MODES:
        n = len(per_mode_detail[m])
        print(f"  {m:11s} {n:2d}/{len(CASES)} cases   worst relerr "
              f"{worst[m]:.3e}" + (f"   ({skipped[m]} skipped: chunk_size "
                                   f"not a multiple of BC={BC})" if skipped[m] else ""))
    ok = all(worst[m] < TOL for m in MODES)
    print("-" * 78)
    print("  PASS" if ok else "  FAIL",
          "-- every mode against the per-token recurrence" if ok else "")
    print("=" * 78)
    return 0 if ok else 1


def main() -> int:
    args = set(sys.argv[1:])
    if "--crossings" in args:
        print_crossings()
    if "--ladder" in args:
        print_ladder()
    if "--test" in args or "--all" in args:
        return run_test()
    if not args:
        print_crossings()
        print_ladder()
        return run_test()
    return 0


if __name__ == "__main__":
    sys.exit(main())
