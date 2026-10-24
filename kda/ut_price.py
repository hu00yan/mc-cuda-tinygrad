#!/usr/bin/env python
"""ut_price.py - price five TinyGrad spellings of the KDA UT transform.

WHAT THIS IS
------------
`kda/tiny_kda.py:178-194` computes the inverse of a `[64,64]` unit
lower-triangular matrix by 64 outer x 63 inner unrolled scalar substitutions.
`notes/25-fla-ut-reference.md` establishes that the reference kernel
(`fla/ops/kda/chunk_intra.py` @ 79d12e8) does something else: a two-level
blocked inverse at `BC=16`, four 14-step diagonal row-loops with zero `tl.dot`,
then a block-Schur merge of 16 `tl.dot`s in a depth-3 DAG.

`notes/26-tinygrad-idiom.md` measured what this framework charges for and, in
its section 10, explicitly declined to price the alternatives ("A blocked or
doubling formulation in pure matmuls ... is the remaining candidate, and its
FLOP count, its kernel count, and whether it is numerically the same operator
are all not settled here"). This file settles them. It is a price list, not a
port: every candidate is a short expression, none of them is a deliverable.

THE INSTRUMENT, AND WHY IT IS RE-DERIVED HERE
---------------------------------------------
Two instruments, both re-implemented in this file rather than imported, and
both checked against a known figure before any number here was taken
(`--selftest`):

  1. `Census` -- hooks `tinygrad.schedule.memory_plan_rewrite`, the last point
     at which the LINEAR still holds one UOp per logical buffer with
     `Ops.RANGE` axes. Counts kernels, REDUCE kernels, static ALU uops and
     ALU-lane ops, and the DAG depth of the kernel-launch graph. Never
     realizes, so it needs no device: `Context(DEV="CUDA")` makes
     `Device.DEFAULT` literally read "CUDA" (`device.py:44`).

  2. `codegen` -- the capture in `matmul/tg_backend.py`, used unmodified, for
     the tensor-core question. Counts `__WMMA` call sites in generated CUDA.

The self-test is not decorative. `notes/26` recorded that a *structural copy* of
the mine.cu DDA disagreed with the real port (217 vs 48 kernels), and that an
`nb^2` penalty was described where the measurement is linear. A census that
does not reproduce the known figure is a broken census, so `A` below is the
real port's own loop, and `one_column_chain` is the exact chain `notes/26`
section 3b measured, checked against its published REDUCE count at every C.

LABELS
------
Every number this file prints carries one of:
  MEASURED   run here, on this host, real tinygrad, `DEV=CUDA`, no device used
  RE-DERIVED arithmetic on shapes, or read out of the installed tinygrad
  INFERRED   needs hardware this host does not have
There are no transcribed figures. Nothing in `notes/` is used as a number.

Run:
  .venv/bin/python kda/ut_price.py --selftest     # instrument validation
  .venv/bin/python kda/ut_price.py                # the price list
  .venv/bin/python kda/ut_price.py --verify       # + numerical equivalence
  .venv/bin/python kda/ut_price.py --tc           # tensor-core probe
  .venv/bin/python kda/ut_price.py --slabs 1      # 1 slab instead of 64
  .venv/bin/python kda/ut_price.py --only C,D
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "matmul"))

from tinygrad import Tensor, dtypes                                    # noqa: E402
from tinygrad.helpers import Context                                   # noqa: E402
from tinygrad.uop.ops import Ops, GroupOp                              # noqa: E402

F32 = dtypes.float32

# The KDA shape. `kda/perf_kda.py:90` is B=2, T=512, HV=4, K=V=64, C=64, so
# NT = T/C = 8 and the UT transform runs on B*NT*HV = 64 independent [64,64]
# unit lower-triangular systems. C=64 is `kda/tiny_kda.py:91` default
# chunk_size; BT=64 is what `fla/ops/kda/chunk.py:442-444` accepts.
KDA = dict(B=2, NT=8, HV=4, C=64)


# =============================================================================
# 1. the instrument
# =============================================================================
def _alu_ops() -> frozenset:
    """Ops that cost a lane operation, read out of the installed tinygrad."""
    return frozenset(set(GroupOp.ALU) | {Ops.EXP2})


ALU_OPS = _alu_ops()

_VIEWLIKE = None


def _base(u):
    """Walk a VIEW / BITCAST / CONTIGUOUS / RESHAPE down to its BUFFER."""
    global _VIEWLIKE
    if _VIEWLIKE is None:
        _VIEWLIKE = {getattr(Ops, n) for n in
                     ("VIEW", "BITCAST", "CONTIGUOUS", "RESHAPE", "PERMUTE", "FLAT")
                     if hasattr(Ops, n)}
    while u is not None and u.op in _VIEWLIKE and u.src:
        u = u.src[0]
    return u if (u is not None and u.op is Ops.BUFFER) else None


class Census:
    """Device-free census of one graph's schedule.

    Hooks `memory_plan_rewrite` (`schedule/memory.py:20`, called from
    `schedule/__init__.py:209`), the last point at which the LINEAR is still
    logical. What it receives is exactly the set of buffers a launch can read
    or write; after it, tinygrad suballocates everything into per-lane arenas,
    and the arenas are peaks, not tensors.

    `lane_ops` is this repo's cost metric: static ALU uops times the kernel's
    trip count (the product of its `Ops.RANGE` extents), with uops sitting
    under a REDUCE discounted by that REDUCE's axis length, because a reduce
    body is re-entered once per (trip / reduce_len) rather than once per trip.
    Ops under an INDEX are addressing arithmetic and are excluded, which is what
    tinygrad's own estimator does (`renderer/__init__.py:28-31`).

    `dag_depth` is the longest chain of dependent kernel LAUNCHES in the
    linear program: a step's depth is 1 + the max depth of any step whose
    buffer it reads. This is the quantity the reference's 14-deep critical path
    is competing with, measured rather than asserted.
    """

    def __init__(self) -> None:
        self._sch = None
        self._orig = None
        self.reset()

    def reset(self) -> None:
        self.kernels = 0        # every entry of linear.src, COPY included
        self.copies = 0
        self.reduce_kernels = 0
        self.alu_static = 0     # sum of static ALU uops over all kernels
        self.lane_ops = 0       # sum of ALU uops x trip, reduce axis discounted
        self.trip_total = 0
        self.max_arg_bytes = 0  # widest kernel argument
        self.max_out_bytes = 0  # widest buffer a real kernel writes
        self.arg_bytes = 0      # sum over kernel arguments, with multiplicity
        self.n_buffers = 0      # distinct buffers
        self.reduce_out_elems = 0   # widest REDUCE output, in elements
        self.reduce_hist: dict = {}  # REDUCE count bucketed by output elements
        self.arg_hist: dict = {}    # kernel count bucketed by widest argument
        self.dag_depth = 0      # longest chain of dependent kernel launches
        self.per_kernel: list[dict] = []

    # -- lifecycle -----------------------------------------------------------
    def __enter__(self) -> "Census":
        import tinygrad.schedule as sch
        self._sch, self._orig = sch, sch.memory_plan_rewrite
        c = self

        def spy(linear, held_bufs=None):  # noqa: ARG001
            c._read(linear)
            return c._orig(linear, held_bufs)

        sch.memory_plan_rewrite = spy
        return self

    def __exit__(self, *exc) -> None:
        if self._sch is not None:
            self._sch.memory_plan_rewrite = self._orig

    # -- the read ------------------------------------------------------------
    def _read(self, linear) -> None:
        depth: dict = {}
        seen: set = set()
        maxdepth = 0
        for si in linear.src:
            self.kernels += 1
            is_copy = si.src[0].op is Ops.COPY
            self.copies += int(is_copy)
            # In a LINEAR, si.src[1] is the buffer the step WRITES -- and it is
            # usually a BITCAST over it, not the BUFFER itself -- and si.src[2:]
            # are the buffers it READS.
            outs = [_base(si.src[1])] if len(si.src) > 1 else []
            ins = [s for s in si.src[2:] if s.op is Ops.BUFFER]
            w = 0
            for b in ins:
                nb = b.nbytes()
                self.arg_bytes += nb
                self.max_arg_bytes = max(self.max_arg_bytes, nb)
                w = max(w, nb)
                seen.add(b)
            if not is_copy:
                k = 1 << max(0, int(w).bit_length() - 1) if w else 0
                self.arg_hist[k] = self.arg_hist.get(k, 0) + 1
            d = 1 + max((depth.get(b, 0) for b in ins), default=0)
            maxdepth = max(maxdepth, d)
            for b in outs:
                if b is None:
                    continue
                if not is_copy:
                    self.max_out_bytes = max(self.max_out_bytes, b.nbytes())
                if b not in seen:
                    seen.add(b)
                    depth[b] = d
            if is_copy:
                self.per_kernel.append(dict(copy=True, reduce=False, trip=0, alu=0,
                                            lane=0, nbytes=[b.nbytes() for b in ins]))
                continue
            self._read_body(si.src[0], ins)
        self.dag_depth = maxdepth
        self.n_buffers = len(seen)

    def _read_body(self, ast, bufs) -> None:
        has_red = False
        trip = 1
        for u in ast.src:                     # END nodes
            for r in u.src:
                if r.op is Ops.RANGE:
                    trip *= int(r.vmax)
        excluded: set = set()
        for u in ast.toposort():
            if u.op is Ops.INDEX:
                excluded |= set(u.sink(*u.src[1:]).toposort())

        under: list[tuple[set, int]] = []
        red_out = 0
        for r in ast.toposort():
            if r.op is Ops.REDUCE:
                has_red = True
                rl = max(1, int(r.src[1].vmax)) if len(r.src) > 1 else 1
                under.append((set(r.sink(*r.src[:1]).toposort()), rl))
                try:                                # output numel of the reduce
                    n = int(np.prod(r.src[0].src[0].shape))
                    red_out = max(red_out, n)
                    b = 1 << max(0, n.bit_length() - 1) if n else 0
                    self.reduce_hist[b] = self.reduce_hist.get(b, 0) + 1
                except Exception:                   # noqa: BLE001
                    pass

        alu = 0
        lane_out = lane_in = 0
        for u in ast.toposort():
            if u in excluded or u.op not in ALU_OPS:
                continue
            alu += 1
            w = 2 if u.op is Ops.MULACC else 1
            rl = next((l for s, l in under if u in s), 0)
            if rl:
                lane_in += w * (trip // rl)
            else:
                lane_out += w * trip
        self.reduce_kernels += int(has_red)
        self.reduce_out_elems = max(self.reduce_out_elems, red_out)
        self.trip_total += trip
        self.alu_static += alu
        self.lane_ops += lane_out + lane_in
        self.per_kernel.append(dict(copy=False, reduce=has_red, trip=trip, alu=alu,
                                    lane=lane_out + lane_in,
                                    nbytes=[b.nbytes() for b in bufs]))

    def row(self) -> dict:
        return dict(kernels=self.kernels, copies=self.copies,
                    reduce_kernels=self.reduce_kernels,
                    alu_static=self.alu_static, lane_ops=self.lane_ops,
                    max_arg_bytes=self.max_arg_bytes, max_out_bytes=self.max_out_bytes,
                    arg_bytes=self.arg_bytes,
                    n_buffers=self.n_buffers, dag_depth=self.dag_depth,
                    reduce_out_elems=self.reduce_out_elems, trip=self.trip_total,
                    reduce_hist=dict(self.reduce_hist), arg_hist=dict(self.arg_hist))


def census(fn, dev: str = "CUDA") -> dict:
    """Build one graph with `fn()` and census it. `fn` returns a Tensor.

    `fn` is called INSIDE the Context, so every Tensor and every constant it
    creates lands on the target device. Building the inputs outside and passing
    the tensors in raises `all buffers must be on the same device` from
    `schedule/__init__.py:149`.
    """
    c = Census()
    t0 = time.perf_counter()
    with Context(DEV=dev):
        with c:
            out = fn()
            out.schedule_linear()
    r = c.row()
    r["secs"] = time.perf_counter() - t0
    r["census"] = c
    return r


def census_ut(fn, slabs: int, C: int) -> dict:
    """Census one UT candidate, building its inputs inside the Context."""
    return census(lambda: fn(*make_S(slabs, C)))


# =============================================================================
# 2. the problem instance
# =============================================================================
def make_S(slabs: int, C: int, seed: int = 0) -> Tensor:
    """`S = StrictTril(Diag(beta) Z)`, the matrix the UT transform inverts.

    `fla/ops/kda/chunk_intra.py:234` row-scales the off-diagonal block by beta
    after the dot, so the matrix inverted is `I + StrictTril(Diag(beta) Z)`.
    Shape `[slabs, C, C]`, one slab per (batch, chunk, head).
    """
    rng = np.random.default_rng(seed)
    i = np.arange(C)[:, None]
    j = np.arange(C)[None, :]
    strict = (i > j).astype(np.float32)
    Z = (rng.standard_normal((slabs, C, C)) * 0.02).astype(np.float32)
    beta = (0.05 + 0.90 * rng.random((slabs, C))).astype(np.float32)
    return Tensor(Z * strict[None] * beta[:, :, None]), Tensor(beta)


def with_beta_column(Ai: Tensor, beta: Tensor) -> Tensor:
    """`M = Ai . Diag(beta)`, the paper's M, which is what `kda/tiny_kda.py:194`
    produces. Identical tail on every candidate, so it cannot move the ranking.
    """
    return Ai * beta[:, None, :]


# =============================================================================
# 3. the candidates
# =============================================================================
# SPELLING controls how index masks and the identity are built. It is a real
# fork and both are measured (`--spelling both`), because the two costs pull in
# opposite directions:
#
#   "numpy"  a Tensor built from a numpy array. tinygrad folds a small
#            numpy-backed constant into the consumer kernel as a literal CONST,
#            so the constant itself costs no kernel -- but every DISTINCT
#            constant in the graph is a separate device buffer, and each one
#            gets its own COPY kernel.
#   "arange" the same mask written as `arange(n) == k`, which is symbolic. No
#            constant buffer and no COPY, but the comparison is re-evaluated
#            inside every consumer that uses it, so ALU-lane ops go up.
#
# Neither is "the right one" without knowing what a kernel launch costs on the
# card, which this host cannot measure. The ranking of the five candidates is
# identical under both, which is the only claim the note makes from this.
SPELLING = "arange"


def eye(n: int) -> Tensor:
    """[n, n] identity."""
    if SPELLING == "arange":
        a = Tensor.arange(n)
        return (a[:, None] == a[None, :]).float()
    return Tensor(np.eye(n, dtype=np.float32))


# -- A. the current port: 64 unrolled column solves, row-wise inside a column.
#    Transcribed from kda/tiny_kda.py:161-194. `kda/tiny_kda.py` is not imported
#    (it would drag two numpy round trips into the census); the loop is copied
#    op for op, and the selftest checks the copy against the real file's
#    published per-C REDUCE counts.
def cand_A(S: Tensor, beta: Tensor) -> Tensor:
    C = S.shape[-1]
    eye_all = eye(C)
    eyeT = [eye_all[i] for i in range(C)]
    A = eye_all + S
    M = Tensor(np.zeros(S.shape, dtype=np.float32))
    for i in range(C):
        b = Tensor(np.zeros(S.shape[:-1], dtype=np.float32)) \
            + beta[:, i][:, None] * eyeT[i]
        for r in range(1, C):
            s = (A[:, r, :r] * b[:, :r]).sum(-1)          # [slabs]
            b = b - eye_all[r][None, :] * s[:, None]
        M = M + eye_all[i][None, None, :] * b[:, :, None]
    return M


# -- B. the same 64 solves, batched on the leading axis.
#    The 64 column solves are 64 INDEPENDENT linear systems with the same A, so
#    they are one batched system. What does NOT change is the depth: each
#    column's 63-step row substitution is still 63 dependent steps, now
#    computing 64 of them per step.
def cand_B(S: Tensor, beta: Tensor) -> Tensor:
    C = S.shape[-1]
    eye_all = eye(C)
    A = eye_all + S                                        # [..., C, C], un-batched
    # rhs[ci, slab, r] = delta(ci, r) * beta[slab, ci]
    rhs = eye_all[:, None, None, :] * beta[:, None, :]
    b = rhs.contiguous().reshape((C, S.shape[0], C))        # [col, slab, row]
    for r in range(1, C):
        # A[:, r, :r] is the r-th ROW of the system matrix, indexed by j < r.
        arow = A[:, r, :r].reshape((1, 1, S.shape[0], r))
        s = (arow * b[:, :, :r]).sum(-1).reshape((C, S.shape[0]))
        b = b - eye_all[r] * s[:, :, None]                # [col, slab, C]
    return b.permute(1, 2, 0).contiguous()               # [slab, row, col]


# -- C. BC=16 two-level blocked, the kernel's schedule.
def cand_C(S: Tensor, beta: Tensor) -> Tensor:
    Ai = diag_stage(S)                                    # [4, slab, BC, BC]
    NC, BC = Ai.shape[0], Ai.shape[-1]
    # depth 1: O10 = -D1 R10 D0, O21 = -D2 R21 D1, O32 = -D3 R32 D2
    o1 = -(pick_diag(Ai, (1, 2, 3)) @ bstack(S, ((1, 0), (2, 1), (3, 2)), BC)) \
         @ pick_diag(Ai, (0, 1, 2))
    # depth 2: O20 = -D2 (R21 O10 + R20 D0), O31 = -D3 (R32 O21 + R31 D1)
    o2 = -(pick_diag(Ai, (2, 3)) @ (
        bstack(S, ((2, 1), (3, 2)), BC) @ o1[:2]
        + bstack(S, ((2, 0), (3, 1)), BC) @ pick_diag(Ai, (0, 1))))
    # depth 3: O30 = -D3 (R30 D0 + R31 O10 + R32 O20)
    o3 = -(Ai[3] @ (blk(S, 3, 0, BC) @ Ai[0]
                    + blk(S, 3, 1, BC) @ o1[0]
                    + blk(S, 3, 2, BC) @ o2[0]))
    C = NC * BC
    blocks = {(k, k): Ai[k] for k in range(NC)}
    blocks.update({(1, 0): o1[0], (2, 1): o1[1], (3, 2): o1[2],
                   (2, 0): o2[0], (3, 1): o2[1], (3, 0): o3})
    return with_beta_column(assemble(blocks, C, BC, tuple(S.shape[:-2])), beta)


def diag_stage(S: Tensor, BC: int = 16) -> Tensor:
    """C's stage 1 alone: the NC = C/BC diagonal 16x16 inverses.

    Faithful to `fla/ops/kda/chunk_intra.py:274-313`: negate, zero the upper
    triangle, then `BC-2 = 14` row-updates whose inner op is the kernel's
    `tl.sum(b_a[:, None] * b_Ai, 0)` -- a broadcast multiply and a reduce over
    the ROW axis, with no `tl.dot` -- then `+= I`.

    The masks are built from one `Tensor.arange` rather than from a numpy array
    per iteration. A `Tensor(numpy_array)` is a device buffer and each one
    costs its own COPY kernel, which would be an artefact of this spelling and
    not a property of the algorithm; an arange is a kernel-internal RANGE.
    """
    C, NC = S.shape[-1], S.shape[-1] // BC
    if SPELLING == "arange":
        ar = Tensor.arange(BC)
        rowi, coli = ar[:, None], ar[None, :]
        ar_e = ar
    else:
        rowi = Tensor(np.arange(BC)[:, None].astype(np.float32))
        coli = Tensor(np.arange(BC)[None, :].astype(np.float32))
        ar_e = None
    R4 = bstack(S, tuple((k, k) for k in range(NC)), BC)     # [NC, slab, BC, BC]
    Ai = -R4 * (rowi > coli)
    for i in range(2, BC):
        # `b_a00` is the i-th ROW of the raw block, masked to col < i (line 287).
        # It is held as a row [NC, slab, 1, BC]; the contraction is taken with
        # its own transpose [NC, slab, BC, 1] so that its index lands on the
        # ROW axis of b_Ai00, which is what `tl.sum(b_a[:,None] * b_Ai, 0)`
        # (line 288) does: out[c] = sum_r b_a00[r] * b_Ai00[r, c].
        base = -R4[:, :, i, :] * (coli < i)
        upd = (base[:, :, :, None] * Ai).sum(-2, keepdim=True)
        a = base[:, :, None, :] + upd
        m = (rowi == i)
        Ai = Ai * (1 - m) + a * m
    if ar_e is not None:
        eyeB = (ar_e[:, None] == ar_e[None, :])
    else:
        eyeB = Tensor(np.eye(BC, dtype=np.float32))
    return Ai + eyeB


def blk(S: Tensor, i: int, j: int, BC: int) -> Tensor:
    """block (i,j) of S, as [slab, BC, BC]."""
    return S[:, i * BC:(i + 1) * BC, j * BC:(j + 1) * BC]


def bstack(S: Tensor, pairs, BC: int) -> Tensor:
    """Batch n block slices into [n, slab, BC, BC] by masked adds, on device.

    The one-hot selector is `arange(n) == idx` (symbolic), not a numpy array: a
    `Tensor(numpy_array)` is a device buffer and each one costs its own COPY
    kernel, which would be an artefact of this spelling rather than a property
    of the algorithm. `Tensor.stack` is deliberately not used -- notes/26
    section 1 records that a STACK of per-element matmuls becomes a serial chain
    of `alu` nodes via `expand_horizontal_reduce`, and whether that fires for a
    STACK of plain slices is the sort of thing to measure, not assume.
    """
    n, sl = len(pairs), tuple(S.shape[:-2])
    out = Tensor(np.zeros((n,) + sl + (BC, BC), dtype=np.float32))
    sel = (Tensor.arange(n) if SPELLING == "arange"
           else Tensor(np.arange(n, dtype=np.float32)))[:, None, None, None]
    for idx, (i, j) in enumerate(pairs):
        out = out + blk(S, i, j, BC)[None] * (sel == idx)
    return out


def pick_diag(Ai: Tensor, ks) -> Tensor:
    """Pick diagonal blocks out of a [NC, slab, BC, BC] batch -> [len(ks), ...]."""
    n, sl = len(ks), tuple(Ai.shape[1:-2])
    out = Tensor(np.zeros((n,) + sl + tuple(Ai.shape[-2:]), dtype=np.float32))
    sel = (Tensor.arange(n) if SPELLING == "arange"
           else Tensor(np.arange(n, dtype=np.float32)))[:, None, None, None]
    for idx, k in enumerate(ks):
        out = out + Ai[k][None] * (sel == idx)
    return out


def assemble(blocks: dict, C: int, BC: int, sl: tuple) -> Tensor:
    """Assemble a [.., C, C] matrix out of a dict of (i, j) -> [.., BC, BC].

    Uses `Tensor.cat` only, so it stays on device. The block grid is square and
    `C == BC * ceil(sqrt(len(blocks)))`; the zero blocks above the diagonal are
    what the kernel leaves at the `torch.zeros` of chunk_intra.py:833.
    """
    NC = C // BC
    Z = Tensor(np.zeros(sl + (BC, BC), dtype=np.float32))
    rows = [Tensor.cat(*[blocks.get((i, j), Z) for j in range(NC)], dim=-1)
            for i in range(NC)]
    return Tensor.cat(*rows, dim=-2)


# -- D. pure-matmul block doubling: invert the halves, combine by matmul, recurse.
#    [[A,0],[B,D]]^-1 = [[A^-1, 0], [-D^-1 B A^-1, D^-1]].  Depth 5 at C=64
#    (64 -> 32 -> 16 -> 8 -> 4 -> 2), two [n,n]@[n,n] products per level, base
#    case 1x1 = I. No substitution and no reduce-to-scalar anywhere.
def cand_D(S: Tensor, beta: Tensor) -> Tensor:
    C = S.shape[-1]
    return with_beta_column(inv_doubling(eye(C) + S), beta)


def inv_doubling(A: Tensor) -> Tensor:
    C = A.shape[-1]
    if C == 1:
        return Tensor(np.ones(tuple(A.shape[:-2]) + (1, 1), dtype=np.float32))
    m = C // 2
    sl = tuple(A.shape[:-2])
    Ai = inv_doubling(A[..., :m, :m])
    Di = inv_doubling(A[..., m:, m:])
    C21 = -(Di @ A[..., m:, :m]) @ Ai                    # the two products
    return assemble({(0, 0): Ai, (1, 0): C21, (1, 1): Di}, C, m, sl)


# -- D2. Neumann on the block-off-diagonal. N = E^-1 . S_off is a 4x4
#    strictly-block-lower nilpotent with N^4 = 0, so
#    S^-1 = (I-N)(I+N^2)(I+N^4)(I+N^8) exactly -- six dependent [64,64]
#    products -- on top of the four 16x16 diagonal inverses, which is C's
#    stage 1 verbatim.
def cand_D2(S: Tensor, beta: Tensor, BC: int = 16) -> Tensor:
    C, NC = S.shape[-1], S.shape[-1] // BC
    R4 = bstack(S, tuple((k, k) for k in range(NC)), BC)
    E = block_diag_matrix(diag_stage(S, BC), C, NC, BC)
    S_off = S - block_diag_matrix(R4, C, NC, BC)
    N = E @ S_off
    N2 = N @ N
    N4 = N2 @ N2
    N8 = N4 @ N4
    I = eye(C)
    # (I+P)^-1 E with E = blockdiag(D_k); without the trailing @E this is the
    # inverse of the block-off-diagonal part, not of I + S.
    return with_beta_column((I - N) @ (I + N2) @ (I + N4) @ (I + N8) @ E, beta)


def block_diag_matrix(blocks: Tensor, C: int, NC: int, BC: int) -> Tensor:
    """[NC, slab, BC, BC] -> [slab, C, C], block diagonal."""
    sl = tuple(blocks.shape[1:-2])
    return assemble({(k, k): blocks[k] for k in range(NC)}, C, BC, sl)


# -- E. the token-parallel diagonal path, which is what the kernel's DEFAULT
#    (`safe_gate=False`, `fla/ops/kda/chunk.py:47` and `:210`) actually takes.
#    `chunk_intra_token_parallel.py:117-124` builds each [16,16] diagonal block
#    with a <=16-iteration loop over j of RANK-1 accumulates and no `tl.dot`:
#        b_kgj = where(m_k, b_kj * exp2(b_g - b_gj), 0)
#        b_Akk = sum(b_k * b_kgj, axis=1) * where(j < i_t, 1, 0)
#    grid is (B*T, cdiv(HV,BH)) -- one program per TOKEN. The spelling here
#    lifts the j axis into a shape; the tl.dot form is E_dot.
def cand_E(kg: Tensor, g: Tensor, C: int = 64) -> Tensor:
    """Token-parallel form: every token, every j, one rank-1 contract over K.

    kg, g : [tokens, C, K]  keys and the cumulative log gate
    Returns the four [16,16] diagonal blocks of StrictTril(Z), as [tokens,C,C].

    The j axis of the kernel's <=16-iteration loop is materialised as a SHAPE
    here -- [tokens, 16, K] -- rather than as a loop, so all B*T programs' worth
    of rank-1 accumulates become one contraction over K. That is the whole
    question this candidate is asked to answer.
    """
    NC, BC = C // 16, 16
    T, K = kg.shape[0], kg.shape[-1]
    tri = Tensor(np.tril(np.ones((BC, BC), dtype=np.float32), -1))
    blocks = []
    for c in range(NC):
        b = c * BC
        k_i = kg[:, b:b + BC]                              # [T, BC(row), K]
        g_i = g[:, b:b + BC]
        # `Tensor.expand` left-aligns its shape argument
        # (`mixin/movement.py:162` _align_left), which silently expanded the
        # wrong axis here; explicit `[:, None]` insertion is unambiguous.
        k_j = k_i[:, None, :, :]                           # [T, 1, BC(col), K]
        g_j = g_i[:, None, :, :]
        w = k_j * (g_i[:, :, None, :] - g_j).exp()          # b_kgj
        blocks.append((k_i[:, :, None, :] * w).sum(-1) * tri)
    return block_diag_from(blocks, C, BC, T)


def block_diag_from(blocks: list, C: int, BC: int, T: int) -> Tensor:
    """[T,BC,BC] blocks on the diagonal of a [T,C,C] zero matrix, on device."""
    NC = len(blocks)
    rows = []
    for c, b in enumerate(blocks):
        parts = []
        if BC * c:                       # Tensor.cat rejects a zero-width axis
            parts.append(Tensor(np.zeros((T, BC, BC * c), dtype=np.float32)))
        parts.append(b)
        if BC * (NC - 1 - c):
            parts.append(Tensor(np.zeros((T, BC, BC * (NC - 1 - c)), dtype=np.float32)))
        rows.append(Tensor.cat(*parts, dim=-1))
    return Tensor.cat(*rows, dim=-2)


def cand_E_dot(kg: Tensor, g: Tensor, C: int = 64) -> Tensor:
    """The NON-default form: the same four [16,16] blocks by `tl.dot`.

    `chunk_intra.py:773` `b_Akk = tl.dot(b_k * b_gq, b_kgt)`, one per sub-chunk
    (`safe_gate=True`). The sub-chunk reference `b_gn` cancels inside the
    product, so the two exponents are opposite.
    """
    NC, BC = C // 16, 16
    T = kg.shape[0]
    tri = Tensor(np.tril(np.ones((BC, BC), dtype=np.float32), -1))
    blocks = []
    for c in range(NC):
        b = c * BC
        kc, gc = kg[:, b:b + BC], g[:, b:b + BC]
        gn = gc[:, 0:1, :]                                 # the sub-chunk reference
        lhs = kc * (gc - gn).exp()
        rhs = (kc * (gn - gc).exp()).permute(0, 2, 1)
        blocks.append((lhs @ rhs) * tri)
    return block_diag_from(blocks, C, BC, T)


def make_E_inputs(slabs: int, C: int, K: int = 64, seed: int = 11):
    """One row of KDA, sized so the token-parallel path is the same problem.

    `slabs` is the number of (batch, chunk) pairs, so the token count is
    `slabs * C` -- the same B*T the kernel's grid is built from.
    """
    rng = np.random.default_rng(seed)
    sh = (slabs * C, C, K)
    kg = Tensor((rng.standard_normal(sh) * 0.5).astype(np.float32))
    g = Tensor((-rng.random(sh) * 0.7).astype(np.float32))
    return kg, g


# =============================================================================
# 4. self-test: does this census reproduce a number someone else published?
# =============================================================================
def one_column_chain(C: int) -> Tensor:
    """Exactly the chain `notes/26` section 3b censused, at [1,1,1,C,C]."""
    rng = np.random.default_rng(0)
    A = Tensor((rng.standard_normal((1, 1, 1, C, C)) * 0.02).astype(np.float32))
    b = Tensor(rng.random((1, 1, 1, C)).astype(np.float32))
    eye_all = eye(C)
    for r in range(1, C):
        s = (A[:, :, :, r, :r] * b[:, :, :, :r]).sum(-1)
        b = b - eye_all[r].reshape(1, 1, 1, C) * s.unsqueeze(-1)
    return b


def selftest(verbose: bool = True) -> bool:
    ok = True
    if verbose:
        print("== self-test 1: the single-column chain of notes/26 section 3b ==")
        print(f"{'C':>4} {'steps':>6} {'kernels':>8} {'REDUCE':>7} {'expect':>7} "
              f"{'lane_ops':>11}  verdict")
    # notes/26 section 3b, table "C | steps | kernels | of which REDUCE"
    expect_red = {4: 2, 8: 6, 16: 14, 32: 30, 64: 62}
    for C in (4, 8, 16, 32, 64):
        r = census(lambda C=C: one_column_chain(C))
        good = r["reduce_kernels"] == expect_red[C]
        ok &= good
        if verbose:
            print(f"{C:>4} {C-1:>6} {r['kernels']:>8} {r['reduce_kernels']:>7} "
                  f"{expect_red[C]:>7} {r['lane_ops']:>11,}  {'OK' if good else 'MISMATCH'}")
    if verbose:
        print()
        print("== self-test 2: the port's 64 outer x (C-1) inner loop, 1 slab ==")
        print("   the whole port must be exactly C x the single chain's REDUCE count")
        print(f"{'C':>4} {'kernels':>8} {'REDUCE':>7} {'expect':>7} {'lane_ops':>11}  verdict")
    for C in (8, 16, 32, 64):
        r = census_ut(cand_A, 1, C)
        e = C * expect_red[C]
        good = r["reduce_kernels"] == e
        ok &= good
        if verbose:
            print(f"{C:>4} {r['kernels']:>8} {r['reduce_kernels']:>7} {e:>7} "
                  f"{r['lane_ops']:>11,}  {'OK' if good else 'MISMATCH'}")
    if verbose:
        print()
        print("== self-test 3: no instrument knob is set ==")
        import tinygrad.helpers as H
        print(f"  ALLOW_TF32={H.ALLOW_TF32.value}  TC={H.USE_TC.value} "
              f"  TC_SELECT={H.TC_SELECT.value}  TC_OPT={H.TC_OPT.value} "
              f"  NOOPT={H.NOOPT.value}  IMAGE={H.IMAGE.value}  NOLOCALS={H.NOLOCALS.value} "
              f"  BEAM={H.BEAM.value}  SPLIT_REDUCEOP={H.SPLIT_REDUCEOP.value}")
        print()
        print("SELFTEST " + ("PASS" if ok else "FAIL"))
    return ok


# =============================================================================
# 5. correctness: do the candidates compute the same operator?
# =============================================================================
def verify(slabs: int = 2, C: int = 64) -> bool:
    """Run every candidate on CPU against a float64 numpy reference."""
    rng = np.random.default_rng(7)
    i = np.arange(C)[:, None]
    j = np.arange(C)[None, :]
    strict = (i > j).astype(np.float64)
    Z = rng.standard_normal((slabs, C, C)) * 0.02
    beta = 0.05 + 0.90 * rng.random((slabs, C))
    Sn = Z * strict[None] * beta[:, :, None]
    target = np.linalg.inv(np.eye(C)[None] + Sn) * beta[:, None, :]   # the paper's M

    cands = [("A", cand_A), ("B", cand_B), ("C", cand_C), ("D", cand_D),
             ("D2", cand_D2)]
    print(f"== numerical check vs numpy float64 inverse, {slabs} slab(s), C={C} ==")
    ok = True
    for name, fn in cands:
        S, bt = Tensor(Sn.astype(np.float32)), Tensor(beta.astype(np.float32))
        try:
            got = fn(S, bt).numpy().astype(np.float64)
            err = np.abs(got - target).max() / np.abs(target).max()
        except Exception as e:                            # noqa: BLE001
            print(f"  {name:<3} FAILED TO RUN: {type(e).__name__}: {e}")
            ok = False
            continue
        good = err < 2e-6
        ok &= good
        print(f"  {name:<3} max rel err = {err:.3e}  {'OK' if good else 'MISMATCH'}")
    print("VERIFY " + ("PASS" if ok else "FAIL"))
    return ok


# =============================================================================
# 6. the price list
# =============================================================================
CANDIDATES = [
    ("A", cand_A, "naive 64x63 unrolled column solve (the port today)"),
    ("B", cand_B, "the same 64 solves, batched on the leading axis"),
    ("C", cand_C, "BC=16 two-level blocked, as the kernel does it"),
    ("D", cand_D, "pure-matmul block doubling, 64 -> 32 -> ... -> 2"),
    ("D2", cand_D2, "Neumann on the block-off-diagonal, 6 [64,64] matmuls"),
]

# The algorithmic critical path (data dependence), re-derived from the
# algorithm, not from a note. kda/tiny_kda.py:178-181 is C outer iterations
# each containing an inner chain of C-1 dependent row updates, so the chain is
# C-1 deep. The kernel's diagonal stage is BC-2 = 14 deep (chunk_intra.py:285
# `for i in range(2, BC)`) and its merge DAG is 3 deep. D's doubling is log2(C)
# levels of 2 dependent products. D2 is C's stage 1 plus 6 dependent products.
def chain_depth(name: str, C: int):
    return {"A": C - 1, "B": C - 1, "C": 14 + 3, "D": int(np.log2(C)),
            "D2": 14 + 6}[name]


# -- algorithmic arithmetic, RE-DERIVED from the algorithm in code -----------
# Not a census: this is the multiply-accumulate count of the ALGORITHM, and it
# is what tells you whether a cheaper schedule is doing less work or the same
# work with more overhead around it.
def macs(name: str, C: int) -> int:
    BC = 16
    NC = C // BC
    if name in ("A", "B"):
        # C column solves, each summing r terms for r = 1..C-1, plus the
        # C outer products that scatter each solved vector into M.
        subs = C * sum(range(1, C))
        scatter = C * C * C
        return subs + scatter
    if name == "C":
        # 4 diagonal 16x16 inverses, rows 2..15, each contracting i terms
        diag = NC * sum(range(2, BC))
        # 16 [16,16]@[16,16] merge products
        merge = 16 * BC ** 3
        return diag + merge
    if name == "D":
        # two [n,n]@[n,n] products per level, n = C/2, C/4, ... 2
        return 2 * sum((C >> (k + 1)) ** 3 for k in range(int(np.log2(C)) - 1))
    if name == "D2":
        diag = NC * sum(range(2, BC))
        return diag + 6 * C ** 3
    raise KeyError(name)


def price_list(slabs: int, C: int, only: list[str] | None = None) -> None:
    print(f"== price list: {slabs} slab(s) of [{C},{C}], DEV=CUDA, no device ==")
    hdr = (f"{'cand':<4} {'kernels':>8} {'REDUCE':>7} {'alu_lane':>14} {'DAG':>5} "
           f"{'widest_out_B':>12} {'red_out':>8} {'bufs':>5} {'secs':>7}  description")
    print(hdr)
    print("-" * len(hdr))
    results = {}
    for name, fn, desc in CANDIDATES:
        if only and name not in only:
            continue
        try:
            r = census_ut(fn, slabs, C)
        except Exception as e:                            # noqa: BLE001
            print(f"{name:<4} FAILED: {type(e).__name__}: {e}")
            continue
        results[name] = r
        print(f"{name:<4} {r['kernels']:>8,} {r['reduce_kernels']:>7,} "
              f"{r['lane_ops']:>14,} {r['dag_depth']:>5} {r['max_out_bytes']:>12,} "
              f"{r['reduce_out_elems']:>8,} {r['n_buffers']:>5} {r['secs']:>7.1f}  {desc}")
    if "A" in results and len(results) > 1:
        base = results["A"]
        print()
        print(f"{'cand':<4} {'kernels / A':>12} {'lane_ops / A':>13} {'DAG / A':>8} "
              f"{'arg_bytes / A':>13} {'buffers / A':>11}")
        for name, r in results.items():
            print(f"{name:<4} {r['kernels']/base['kernels']:>11.2f}x "
                  f"{r['lane_ops']/max(1, base['lane_ops']):>12.3f}x "
                  f"{r['dag_depth']/base['dag_depth']:>7.2f}x "
                  f"{r['arg_bytes']/base['arg_bytes']:>12.3f}x "
                  f"{r['n_buffers']/base['n_buffers']:>10.3f}x")
    print()
    print()
    print(f"{'cand':<4} {'critical path':>13} {'algorithmic MACs/slab':>21} {'vs C':>7}")
    base = macs("C", C)
    for name, _fn, _d in CANDIDATES:
        if only and name not in only:
            continue
        m = macs(name, C)
        print(f"{name:<4} {chain_depth(name, C):>13} {m:>21,} {m/base:>6.2f}x")


def price_E(slabs: int, C: int) -> None:
    print(f"== price list E: building the four [16,16] diagonal blocks, "
          f"{slabs * C} tokens x [64,K=64], i.e. {slabs} chunks ==")
    hdr = (f"{'form':<6} {'kernels':>8} {'REDUCE':>7} {'alu_lane':>14} {'DAG':>5} "
           f"{'max_arg_B':>11} {'secs':>7}  description")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for name, fn, desc in (("E", cand_E, "token-parallel, <=16 rank-1 steps, no dot"),
                           ("E_dot", cand_E_dot, "the non-default tl.dot form")):
        try:
            r = census(lambda fn=fn: fn(*make_E_inputs(slabs, C), C))
        except Exception as e:                            # noqa: BLE001
            print(f"{name:<6} FAILED: {type(e).__name__}: {e}")
            continue
        rows.append((name, r))
        print(f"{name:<6} {r['kernels']:>8,} {r['reduce_kernels']:>7,} "
              f"{r['lane_ops']:>14,} {r['dag_depth']:>5} {r['max_arg_bytes']:>11,} "
              f"{r['secs']:>7.1f}  {desc}")
    if len(rows) == 2:
        a, b = rows[0][1], rows[1][1]
        print(f"\nE_dot / E: kernels {b['kernels']/max(1, a['kernels']):.2f}x   "
              f"lane_ops {b['lane_ops']/max(1, a['lane_ops']):.2f}x   "
              f"DAG {b['dag_depth']/max(1, a['dag_depth']):.2f}x")
    # Is the kernel's default and non-default spelling distinguishable at all
    # in the generated code? md5 the captured CUDA.
    try:
        import hashlib
        import tg_backend
        kg, g = make_E_inputs(1, C)
        hs = {}
        for nm, fn in (("E", cand_E), ("E_dot", cand_E_dot)):
            ks = tg_backend.generate("sm_80", lambda fn=fn: fn(kg, g, C).schedule_linear())
            hs[nm] = [hashlib.md5(k.src.encode()).hexdigest()[:10] for k in ks]
        same = hs["E"] == hs["E_dot"]
        print(f"\nfp16 codegen: E      {hs['E']}")
        print(f"fp16 codegen: E_dot  {hs['E_dot']}")
        print("generated CUDA identical: " + str(same))
    except Exception as e:                                            # noqa: BLE001
        print(f"codegen capture unavailable: {type(e).__name__}: {e}")


def price_both(slabs: int, C: int) -> None:
    """Run the price list under both constant spellings and diff them."""
    global SPELLING
    rows: dict = {}
    for sp in ("arange", "numpy"):
        SPELLING = sp
        print(f"\n### spelling = {sp} " + "#" * 40)
        print(f"{'cand':<4} {'kernels':>8} {'COPY':>6} {'REDUCE':>7} {'alu_lane':>12} "
              f"{'DAG':>5} {'max_arg_B':>10}")
        for name, fn, _d in CANDIDATES:
            try:
                r = census_ut(fn, slabs, C)
            except Exception as e:                        # noqa: BLE001
                print(f"{name:<4} FAILED: {type(e).__name__}: {e}")
                continue
            rows[(sp, name)] = r
            print(f"{name:<4} {r['kernels']:>8,} {r['copies']:>6,} "
                  f"{r['reduce_kernels']:>7,} {r['lane_ops']:>12,} {r['dag_depth']:>5} "
                  f"{r['max_arg_bytes']:>10,}")
    SPELLING = "arange"
    print("\n### does the spelling change the ranking? " + "#" * 26)
    for sp in ("arange", "numpy"):
        ks = [(r["lane_ops"], n) for (s, n), r in rows.items() if s == sp]
        kk = [(r["kernels"], n) for (s, n), r in rows.items() if s == sp]
        if not ks:
            continue
        print(f"  {sp:<7} by alu_lane: {' < '.join(n for _, n in sorted(ks))}")
        print(f"  {sp:<7} by kernels : {' < '.join(n for _, n in sorted(kk))}")


# =============================================================================
# 7. the tensor-core question, for THESE shapes
# =============================================================================
def tc_probe() -> None:
    """Does the matmul of the merge / the doubling reach a tensor core?

    Uses `matmul/tg_backend.py` unmodified -- the real tinygrad codegen path
    with only the compiler stubbed. Reports `__WMMA` call sites (the tensor-core
    path) and, as a cross-check, the FP FMA count in the emitted source, so a
    "0 WMMA" row is visibly FFMA rather than visibly nothing.
    """
    import re
    import tg_backend

    def ffma(src: str) -> int:
        """Accumulator-update expressions, `buf0[i] = ((buf0[i]) + a*b + ...)`.

        These are the lines an FFMA GEMM is built from. A tensor-core kernel has
        none of them: it issues `__WMMA_...` / `mma.sync` instead. Counting them
        is what makes a `0 __WMMA` row visibly *FFMA* rather than visibly
        nothing.
        """
        return len(re.findall(r"\(\(\*\(\w+\+\d+\)\)\+\(", src))

    print("== tensor cores: does the merge / doubling matmul shape reach one? ==")
    print(f"{'dtype':>5} {'M':>5} {'N':>5} {'K':>5} {'__WMMA':>7} {'FMA':>7} "
          f"{'src_B':>7}  kernel")
    shapes = [
        (2, 2, 2), (4, 4, 4), (8, 8, 8),          # D's smallest doubling levels
        (16, 16, 16), (16, 16, 64),                # C's merge blocks
        (32, 32, 32), (64, 64, 64), (64, 64, 32), (64, 64, 16),
        (8, 8, 8), (8, 64, 64), (64, 8, 64), (64, 64, 8),
        (256, 256, 256),
    ]
    for dt, dn in ((dtypes.half, "fp16"), (dtypes.bfloat16, "bf16"),
                   (dtypes.float32, "fp32")):
        for (M, N, K) in shapes:
            a = Tensor(np.ones((M, K), dtype=np.float16), dtype=dt)
            b = Tensor(np.ones((K, N), dtype=np.float16), dtype=dt)
            ks = tg_backend.generate("sm_80", lambda: (a @ b).schedule_linear())
            w = sum(k.src.count("__WMMA") for k in ks)
            f = sum(ffma(k.src) for k in ks)
            nb = sum(len(k.src) for k in ks)
            print(f"{dn:>5} {M:>5} {N:>5} {K:>5} {w:>7} {f:>7} {nb:>7}  "
                  f"{ks[0].name[:30] if ks else '-'}")

    print("\n== the same shapes with a BATCH leading axis, as the candidates write them ==")
    print(f"{'dtype':>5} {'nb':>4} {'M':>5} {'N':>5} {'K':>5} {'__WMMA':>7}  kernel")
    for dt, dn in ((dtypes.half, "fp16"), (dtypes.float32, "fp32")):
        for (nb, M, N, K) in ((4, 16, 16, 16), (3, 16, 16, 16), (2, 16, 16, 16),
                              (64, 16, 16, 16), (4, 64, 64, 64)):
            a = Tensor(np.ones((nb, M, K), dtype=np.float16), dtype=dt)
            b = Tensor(np.ones((nb, K, N), dtype=np.float16), dtype=dt)
            ks = tg_backend.generate("sm_80", lambda: (a @ b).schedule_linear())
            w = sum(k.src.count("__WMMA") for k in ks)
            print(f"{dn:>5} {nb:>4} {M:>5} {N:>5} {K:>5} {w:>7}  "
                  f"{ks[0].name[:30] if ks else '-'}")


# =============================================================================
# 8. the ladder: is the win the algorithm or just the fusion?
# =============================================================================
def _merge_only(S: Tensor, BC: int = 16) -> Tensor:
    """C's merge stage without the assembly tail, for stage attribution."""
    Ai = diag_stage(S, BC)
    o1 = -(pick_diag(Ai, (1, 2, 3)) @ bstack(S, ((1, 0), (2, 1), (3, 2)), BC)) \
         @ pick_diag(Ai, (0, 1, 2))
    return o1


def ladder(slabs: int, C: int) -> None:
    """Price the two stages separately, so the win can be attributed."""
    print(f"== stage attribution, {slabs} slab(s), C={C} ==")
    hdr = f"{'what':<34} {'kernels':>8} {'REDUCE':>7} {'alu_lane':>14} {'DAG':>5}"
    print(hdr)
    print("-" * len(hdr))
    items = [
        ("A: full 64x63 substitution", lambda S, b: cand_A(S, b)),
        ("A: one column of it (63 steps)", lambda S, b: cand_B(S, b)[:, :, :1]),
        ("C: diagonal stage alone (4x14)", lambda S, b: with_beta_column(
            diag_stage(S), b[:, :1].repeat(16).reshape(slabs, 16))),
        ("C: full (diag + merge + assemble)", lambda S, b: cand_C(S, b)),
        ("C: merge only, no assembly", lambda S, b: _merge_only(S)),
        ("D: one doubling level (64->32)", lambda S, b: inv_doubling(
            (eye(C) + S)[..., :32, :32])),
        ("D: full 5-level doubling", lambda S, b: cand_D(S, b)),
    ]

    for label, fn in items:
        try:
            r = census(lambda fn=fn: fn(*make_S(slabs, C)))
        except Exception as e:                            # noqa: BLE001
            print(f"{label:<34} FAILED: {type(e).__name__}: {e}")
            continue
        print(f"{label:<34} {r['kernels']:>8,} {r['reduce_kernels']:>7,} "
              f"{r['lane_ops']:>14,} {r['dag_depth']:>5}")


# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slabs", type=int, default=64,
                    help="number of independent [C,C] systems (KDA bench = 64)")
    ap.add_argument("--C", type=int, default=64)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--tc", action="store_true")
    ap.add_argument("--e", action="store_true", help="candidate E only")
    ap.add_argument("--ladder", action="store_true")
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--spelling", type=str, default="arange",
                    choices=("arange", "numpy", "both"))
    args = ap.parse_args()
    only = [s for s in args.only.split(",") if s] or None

    global SPELLING
    if args.spelling in ("arange", "numpy"):
        SPELLING = args.spelling
    if args.selftest:
        return 0 if selftest() else 1
    if args.verify:
        return 0 if verify() else 1
    if args.tc:
        tc_probe()
        return 0
    if args.e:
        price_E(args.slabs, args.C)
        return 0
    if args.ladder:
        ladder(args.slabs, args.C)
        return 0
    if args.spelling == "both":
        price_both(args.slabs, args.C)
        return 0
    print(f"(constant spelling = {SPELLING})")
    price_list(args.slabs, args.C, only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
