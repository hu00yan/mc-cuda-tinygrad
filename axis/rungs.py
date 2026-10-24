#!/usr/bin/env python3
"""rungs.py -- the ladder: one operator class per rung, both arms.

WHAT THIS FILE IS
=================
The axis. Seven rungs, ordered by *what they demand of a compiler*, not by size.
For each rung:

  * `ref`  -- the hand-written kernel in `axis/refkernels.cu`, launched by
              `axis/harness.py` in TinyGrad's own context
  * `tg`   -- the TinyGrad expression here, and ONLY here

THE TINYGRAD SIDE'S CONSTRAINTS, AND WHY THEY ARE ENFORCED IN CODE
==================================================================
The task forbids `Tensor.custom_kernel`, pasted CUDA, inline PTX and a
hand-written .cu compiled alongside *on the TinyGrad side*. This module imports
`tinygrad.Tensor` and `numpy`, and nothing else that can reach a GPU. It never
sees `axis/harness.py` and never touches the driver API. `verify.py` re-checks
that by AST: it parses this file and fails if any call to `custom_kernel`,
`load`, `cuLaunchKernel`, `cuModuleLoadData` or `ctypes` appears in it. A
constraint enforced by a comment is a comment.

THE "ESTABLISH BEST BY FIXING THE HABIT" RULE
=============================================
notes/06 §5b: the two big comparisons in this study were both invalidated by the
TinyGrad side being naive, and notes/26 / notes/31 found the dominant habit --
*reducing to a scalar inside a loop where a wide contraction belongs* -- worth
~60x on its own, hidden by kernel count and exposed by ALU-lane op count.

So "best TinyGrad" here means: every reduction is ONE wide reduction, every
contraction is ONE `matmul`, and no rung is written in a shape whose cost the
framework would have to be stupid to charge. The naive spellings are kept, but
they are kept as *separately labelled measurements* (`tg_naive`), not as the
rung's TinyGrad point. That is the difference between measuring the framework
and measuring the port, and it is the whole point of the exercise.

Every rung therefore reports:
    tg        the idiomatic expression   <- THE rung's TinyGrad point
    tg_naive  the habit spelling, where one exists, as a control
    tg_knob   the same expression under a framework knob, where the knob is
              a scheduling knob rather than a hardware hint

THE FOUR-WAY CLASSIFICATION IS NOT ASSIGNED HERE
================================================
For each rung the classification into (i) cannot express / (ii) can, cost model
could find it / (iii) not implemented / (iv) does not exist for this
architecture is decided from the MEASUREMENTS in the note, not written down in
advance. Writing the verdict before the number is how a classification with no
falsifier gets made. What this module provides instead is, per rung, the
`distinguishing_evidence` field naming the experiment that would have separated
the four classes -- so the note has something to point at.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from tinygrad import Tensor, dtypes

# ===========================================================================
# the rung record
# ===========================================================================


@dataclass
class Rung:
    key: str
    title: str
    demands: str                      # what it demands of a compiler
    shapes: dict                      # the shape dict, passed to both arms
    ref_kernel: str                   # the extern "C" symbol in refkernels.cu
    ref_grid: Callable[[dict], tuple]  # (shapes) -> (gx, gy, gz)
    ref_block: Callable[[dict], tuple]  # (shapes) -> (bx, by, bz)
    ref_args: Callable[[dict], tuple]   # (shapes) -> the scalar/int args
    tg: Callable[[dict], Tensor]        # the idiomatic TinyGrad arm
    inputs: Callable[[dict], dict]      # numpy inputs, shared by BOTH arms
    outputs: list                       # output tensor names (Buffer objects at run time)
    out_dtype: str = "float32"
    tg_naive: Callable[[dict], Tensor] | None = None
    tg_knob: tuple | None = None          # (label, Context kwargs, builder)
    verify: Callable | None = None       # (ref_bytes, tg_bytes, numpy) -> (ok, worst_relerr)
    distinguishing_evidence: str = ""
    bytes_moved: Callable[[dict], int] | None = None   # for the roofline position
    flops: Callable[[dict], int] | None = None
    # The compiled kernel's parameter TYPES, as a string of 'p' (pointer) and
    # 'i' (int). Checked against the PTX on the card before anything is timed.
    #
    # This exists because of a bug that would otherwise have been invisible: the
    # two GEMM kernels took the output pointer THIRD while every other kernel and
    # the harness took it first, so the reference overwrote an input buffer and
    # read the output buffer as A. That is a wrong answer with no crash, and the
    # only thing that caught it was the oracle comparison -- a lucky catch, not a
    # designed one. A signature pattern turns the luck into a check, and it
    # catches ORDER as well as count because the type sequence differs: the old
    # GEMM signature was p p i i i against the expected p p p i i i.
    ref_params: str = ""


# ===========================================================================
# the input generators -- SHARED by both arms
# ===========================================================================
# Both arms read the same device buffers, filled once by these functions. That
# is deliberate: if each arm generated its own inputs the comparison would also
# be comparing generators, and a disagreement about the data is a much harder
# failure to see than a disagreement about the code. The seed is fixed and
# printed by the harness.


def _f32(*shape, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.standard_normal(shape, dtype=np.float32))


def in_ew(s: dict) -> dict:
    n = s["n"]
    return {"a": _f32(n, seed=1), "b": _f32(n, seed=2), "z": _f32(n, seed=3), "n": n}


def tg_ew(s: dict, d: dict) -> Tensor:
    return _T(d["a"]) + _T(d["b"]) * _T(d["z"])


def _T(x):
    """Wrap a numpy array in a Tensor; pass a Tensor through unchanged.

    The rung builders are called with DEVICE tensors by the harness, and with
    numpy arrays by `verify.py` on the CPU gate. Both have to work, and the
    distinction matters: if a builder is handed numpy during a timing loop, the
    `Tensor(...)` it constructs puts a host-to-device copy INSIDE the graph, and
    the measured "device time" becomes mostly PCIe. That happened, and it
    inflated `ew` by 40x. The oracle path is unaffected because it never calls a
    builder.
    """
    from tinygrad import Tensor
    return x if isinstance(x, Tensor) else Tensor(x)


# ===========================================================================
# RUNG 0 -- `ew`  elementwise, pure bandwidth  (the control)
# ===========================================================================
# Demands: nothing. No reduction, no gather, no reuse. Four arrays in, one out.
#
# This rung exists to make the axis falsifiable. If the framework cannot reach
# the memory roofline on THIS, then every gap above it is confounded -- it could
# be arithmetic, or it could be that this framework simply does not saturate
# memory on any graph, and those are very different findings. A ladder with no
# control rung is a list of numbers.
#
# The `red` rung immediately above it is the same traffic with a reduction added,
# so the pair (ew, red) isolates "does a reduction cost more than its bytes".


def in_red(s: dict) -> dict:
    R, K = s["R"], s["K"]
    return {"x": _f32(R, K, seed=4), "w": _f32(R, K, seed=5), "R": R, "K": K}


def tg_red(s: dict, d: dict) -> Tensor:
    # THE IDIOM: one wide reduction over a K-long axis. notes/26 rule 3,
    # "output width, not operator name".
    return (_T(d["x"]) * _T(d["w"])).sum(-1)


def tg_red_naive(s: dict, d: dict) -> Tensor:
    # THE HABIT: K-1 dependent rank-1 reductions, accumulating. This is what a
    # port writes when it treats `k` as the outer loop, and it is the ~60x
    # C-18 is about. Measured as a control, never as the rung's point.
    acc = (_T(d["x"])[..., 0:1] * _T(d["w"])[..., 0:1]).sum(-1)
    for k in range(1, s["K"]):
        acc = acc + (_T(d["x"])[..., k:k + 1] * _T(d["w"])[..., k:k + 1]).sum(-1)
    return acc


# ===========================================================================
# RUNG 2 -- `gemm32`  high arithmetic intensity, fp32  (no tensor core exists)
# ===========================================================================
# Demands: reuse. Every byte of A and B is read many times, so the kernel has to
# keep a tile in registers and the framework has to tile at all.
#
# fp32 on sm_75 has NO tensor core. notes/26 §6b measured that at every shape,
# and notes/06 C-19 corrected the shape-gate version of that claim. So this rung
# is FFMA-or-nothing on both sides, and the comparison is about tiling, register
# blocking and shared-memory staging -- not about instruction-class selection.


def in_gemm32(s: dict) -> dict:
    M, N, K = s["M"], s["N"], s["K"]
    return {"A": _f32(M, K, seed=6), "B": _f32(K, N, seed=7), "M": M, "N": N, "K": K}


def tg_gemm32(s: dict, d: dict) -> Tensor:
    return _T(d["A"]).matmul(_T(d["B"]))


# ===========================================================================
# RUNG 3 -- `gemm16`  high arithmetic intensity, fp16  (tensor cores DO exist)
# ===========================================================================
# The same computation as rung 2 in a dtype that HAS a tensor core on this
# architecture. That makes it the rung where the four-way classification has
# real content: the hardware is present, the framework has a tensor-core table
# to choose from (TC / TC_SELECT), and so "a cost model could find it" is a
# live hypothesis rather than a rhetorical one.
#
# Kept as a separate rung from gemm32 rather than a dtype sweep because notes/06
# C-19's whole content is that conflating "fp32 doesn't reach a tensor core"
# with "small shapes don't reach one" produced a wrong explanation in this repo
# once already.


def in_gemm16(s: dict) -> dict:
    M, N, K = s["M"], s["N"], s["K"]
    rng = np.random.default_rng(8)
    return {"A": np.ascontiguousarray(rng.standard_normal((M, K)).astype(np.float16)),
            "B": np.ascontiguousarray(rng.standard_normal((K, N)).astype(np.float16)),
            "M": M, "N": N, "K": K}


def tg_gemm16(s: dict, d: dict) -> Tensor:
    return _T(d["A"]).matmul(_T(d["B"]))


# ===========================================================================
# RUNG 4 -- `gather`  random access, no reuse  (isolates the gather)
# ===========================================================================
# Demands: an address computed per element, and a table that does not fit in
# L2, so the access actually misses.
#
# The arithmetic is ONE multiply. So whatever the framework does with index
# arithmetic is the entire story, and this rung sits directly against notes/26
# §3c's claim that a gather is a fusion barrier "about the index, not the
# gather".


def in_gather(s: dict) -> dict:
    n, t = s["n"], s["table"]
    rng = np.random.default_rng(9)
    return {"tab": _f32(t, seed=10),
            "idx": np.ascontiguousarray(rng.integers(0, t, size=n, dtype=np.int32)),
            "s": _f32(n, seed=11), "n": n, "table": t}


def tg_gather(s: dict, d: dict) -> Tensor:
    return _T(d["tab"])[_T(d["idx"])] * _T(d["s"])


# ===========================================================================
# RUNG 5 -- `scan`  a carried dependency chain  (subject: the KDA class)
# ===========================================================================
# Demands: a dependent chain. notes/26 §5c says the framework "cannot keep a
# dependent chain in registers" and has no loop, no unrolling, and an `UNROLL`
# opt capped at 32 (notes/06 C-16).
#
# A scan is the purest possible test of that claim, and it is deliberately
# placed as a SEPARATE rung from the KDA triangular solve. The triangular solve
# confounds two things -- a carried chain AND a small dense contraction inside
# the chain -- so a gap there cannot say which one caused it. The scan has the
# chain and nothing else. Whatever the scan measures is attributable to the
# chain alone, and the difference between the scan rung and the trisolve rung is
# then attributable to the contraction.


def in_scan(s: dict) -> dict:
    R, N = s["R"], s["N"]
    return {"x": _f32(R, N, seed=12), "R": R, "N": N}


def tg_scan(s: dict, d: dict) -> Tensor:
    return _T(d["x"]).cumsum(-1)


def tg_scan_naive(s: dict, d: dict) -> Tensor:
    # THE HABIT for a recurrence: unroll it in Python, one add per step. N real
    # ops, no unrolling available (notes/06 C-16), N-1 of them dependent, and
    # N-1 of them writing a full [R,1] intermediate that nothing can fuse away
    # because the next step reads it. Measured as a control, never as the point.
    x = _T(d["x"])
    acc = x[..., 0:1]
    for i in range(1, s["N"]):
        acc = acc + x[..., i:i + 1]
    # `acc` is the running total of the LAST element, i.e. the total sum -- the
    # per-prefix outputs this naive form would need are not what the rung
    # computes, so it is registered as a BOUND on the naive cost, not a
    # same-output control. The note says so at the point of use.
    return acc


# ===========================================================================
# RUNG 6 -- `trisolve`  blocked triangular inverse  (subject: KDA's UT transform)
# ===========================================================================
# Demands: a carried chain AND a dense contraction at every step. This is the
# hardest rung on the ladder and it is one of the two subjects this study
# already has.
#
# WHAT THE REFERENCE IS, PRECISELY
# --------------------------------
# The reference is NOT `fla/ops/kda/naive.py`. notes/06 C-15 is the correction
# that establishes this: the study's KDA port implemented naive.py's unrolled
# forward substitution -- 4 032 sequentially dependent scalar reductions -- and
# passed 6/6 tests. The real kernel is a two-level blocked inverse at BC=16 with
# a 16-dot depth-3 merge. `axis/refkernels.cu`'s `axis_trisolve` implements the
# BLOCKED form, because that is what the GPU is supposed to be running.
#
# The consequence for the axis: this rung is NOT "TinyGrad vs flash-linear-
# attention". It is "TinyGrad vs a hand-written blocked inverse that runs on
# sm_75", and the hand-written side is a *sibling* of the Triton kernel, not the
# Triton kernel. notes/29 already records that the Triton original cannot be
# AOT-compiled in this repo (no triton installed) and that its SASS was therefore
# never measured. The note states this in the rung's row.


def in_trisolve(s: dict) -> dict:
    B, C = s["B"], s["C"]
    rng = np.random.default_rng(13)
    raw = rng.standard_normal((B, C, C)).astype(np.float32)
    tri = np.tril(raw, -1) * 0.1          # strictly lower, small entries
    return {"D": np.ascontiguousarray(tri), "B": B, "C": C}


def tg_trisolve(s: dict, d: dict) -> Tensor:
    """Blocked two-level inverse, in the framework's own units.

    For a block lower-triangular D, the inverse of (I+D) satisfies

        A_kk = (I + D_kk)^-1                                (block diagonal)
        A_ij = -A_ii . S_ij . A_jj       for i > j          (block Schur)
        S_ij = D_ij + sum_{k=j+1}^{i-1} D_ik . A_kj

    and A_kk is a forward substitution over a BC x BC block. That substitution
    IS sequential -- row i of the inverse cannot be formed until rows 0..i-1 are
    -- but each step is ONE wide contraction of shape [1,BC] x [BC,BC] across
    the whole batch, which is the idiom notes/26 rule 3 asks for. The naive port
    (notes/06 C-15) instead did BC-1 dependent SCALAR reductions per step, 4 032
    of them, and passed every test.

    No explicit inverse and no triangular solve appear: notes/26 §5b records
    that the framework has neither, so the substitution is spelled out. That is
    the point of the rung rather than a workaround -- the question is what the
    block algebra costs in the only form the framework accepts.
    """
    B, C, BC = s["B"], s["C"], s["BC"]
    NC = C // BC
    D = _T(d["D"]).reshape(B, NC, BC, NC, BC)   # [B, blockrow, BC, blockcol, BC]

    ar = np.arange(BC)
    arange_bc = np.arange(BC)

    # ---- diagonal blocks: A_kk = (I + D_kk)^-1 by row-wise substitution
    A: dict[tuple[int, int], Tensor] = {}
    for k in range(NC):
        Rk = D[:, k, :, k, :]                                      # [B,BC,BC]
        X = Tensor(np.eye(BC, dtype=np.float32)).reshape(1, 1, BC, BC)
        for i in range(1, BC):
            prev = X[:, :, :i, :]                                  # [B,1,i,BC] rows 0..i-1
            coef = Rk[:, i, :i, None].reshape(B, 1, i, 1)          # [B,1,i,1]
            red = (coef * prev).sum(2, keepdim=True)                # [B,1,1,BC]
            # X[i,:] = e_i - sum_{m<i} R[i,m] X[m,:].  The e_i term is NOT
            # optional: rows 0..i-1 carry their identity entry at column m, so
            # `red` is zero in every column >= i, and writing -red alone would
            # overwrite the identity's diagonal entry with 0. The first version
            # of this omitted e_i and produced a matrix with a zero diagonal that
            # was still lower-triangular and still the right shape -- the exact
            # failure notes/06 §5b is about.
            delta = Tensor((arange_bc == i).astype(np.float32)).reshape(1, 1, 1, BC)
            newrow = (-red + delta).expand(B, 1, BC, BC)
            # Select ROW i, so the mask's True axis must be the second-to-last.
            # Putting it on the last axis selects column i instead and yields a
            # matrix that is upper- rather than lower-triangular -- a wrong
            # answer with the right shape and the right dtype, which is exactly
            # the kind of defect a shape check cannot see (notes/06 C-20 records
            # a transposed-diagonal bug that survived one).
            mask = Tensor((arange_bc == i).reshape(1, 1, BC, 1))
            X = Tensor.where(mask, newrow, X)
        A[(k, k)] = X

    # ---- block-Schur merge, one block column at a time so A_kj exists before use
    #
    #   X_ij = -A_ii . ( sum_{k=j}^{i-1} D_ik . X_kj )
    #
    # The sum runs from k = j, NOT from k = j+1. The k = j term is D_ij . X_jj,
    # which is the whole of the i = j+1 case and the leading term of the others;
    # starting at j+1 silently drops it, and the result is still lower
    # triangular and still the right shape. This was the third defect in this
    # function and the only one the float64 oracle caught rather than a shape
    # check -- the first two (a transposed row mask, a dropped identity term)
    # both produced plausible matrices.
    for j in range(NC):
        for i in range(j + 1, NC):
            S = None
            for k in range(j, i):
                term = D[:, i, :, k, :].reshape(B, 1, BC, BC).matmul(
                    A[(k, j)].reshape(B, 1, BC, BC))
                S = term if S is None else S + term
            A[(i, j)] = A[(i, i)].reshape(B, 1, BC, BC).matmul(S) * -1.0

    # ---- reassemble [B,C,C]
    #
    # The rung's contract is X = (I+D)^-1 - I, whose MAIN diagonal is zero -- but
    # whose block-diagonal BLOCKS are not. A_kk = (I + D_kk)^-1 is unit lower
    # triangular, so A_kk - I is its strictly lower part and is generally
    # nonzero. Dropping the block diagonal (the first version of this) leaves
    # block (0,0) identically zero and is wrong; the result is still
    # lower-triangular, still the right shape, and still passes a shape check,
    # which is the notes/06 §5b failure mode exactly. What IS zero is everything
    # on or above the main diagonal, and that is enforced by the strict-lower
    # mask below rather than by omission.
    eye = np.eye(BC, dtype=np.float32)
    out = np.zeros((B, C, C), dtype=np.float32)
    for (i, j), blk in A.items():
        v = blk.reshape(B, BC, BC).numpy()
        # A block-diagonal block A_kk is unit lower triangular, so subtracting the
        # identity leaves exactly its strictly lower part and no mask is needed.
        #
        # An OFF-DIAGONAL block A_ij (i>j) is NOT triangular as a block: its row 0
        # is strictly below the main diagonal of the FULL matrix and is generally
        # nonzero. Masking it with `tril(ones, -1)` -- which zeroes row 0 of every
        # block -- silently deletes that row. The result stays lower triangular,
        # keeps the right shape, and is wrong; this was the fourth defect in this
        # function and the only one that a per-block comparison against numpy
        # caught only because the block was compared before assembly.
        if i == j:
            v = v - eye
        # explicit slice placement into the [B,C,C] result. An earlier version
        # built a [B,NC,NC,BC,BC] staging array and wrote `out[:, i, j] = v`;
        # with integer indices on a 5-D array that advances the FIRST matching
        # axis, not the two block axes, so every block landed a row off and the
        # reassembled matrix was a shifted, still-lower-triangular, still-
        # right-shaped answer. Slice assignment has no such ambiguity.
        out[:, i * BC:(i + 1) * BC, j * BC:(j + 1) * BC] = v
    return Tensor(out)
