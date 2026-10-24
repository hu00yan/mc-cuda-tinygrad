#!/usr/bin/env python3
"""simref.py -- a host-side EMULATOR of `axis/refkernels.cu`.

WHY THIS FILE EXISTS
====================
Two of the four defect classes that have cost this study real money are defects
in the REFERENCE arm -- the hand-written CUDA, which by definition nobody else's
tooling is checking. Both were found on the card, and each cost a full Kaggle
session plus a queue:

  * `axis_scan_v*` carried `run` as a loop-local, so every float4 after the
    first was written as `offset + its own four elements`. A shape check, a clean
    compile, a `cuobjdump` census and a *correct* Python transcription all passed.
    Only the on-card float64 oracle and the error profile (exact at thread-relative
    positions 0-3, wrong at 4-15, 56% of elements off) identified it.
  * `axis_trisolve` tested `c > i` where the row-wise recursion `X[i][j] =
    -sum_{m<i} D[i][m] X[m][j]` needs `c < i`, so it never wrote a strictly-lower
    entry. Reported as `ref_at_worst 0.0, want 0.5505707`.

The generalisation those two facts support: a hand-written kernel needs a check
that runs BEFORE the card, and the check must be a transcription of the KERNEL,
not of the algorithm. The scan bug survived a Python transcription because that
transcription was written from the intended algorithm -- it silently contained
the fix. So this file is deliberately *not* an implementation of the operators.
It is a mechanical restatement of `refkernels.cu`, and the discipline that makes
it worth having is:

  1. Phase-structured, not function-structured. Every phase is a barrier-
     separated span of the .cu, and each is written as `for tid in range(nthreads)`
     around the statements the .cu has between its barriers. A statement the .cu
     has after a barrier and this file has before it is visible as a diff.
  2. Shared memory goes through `Tile2D`, whose writes land in a PENDING buffer
     and are only visible after `sync()`. Delete a `__syncthreads()` from the .cu
     and delete the matching `sync()` here and the emulator's answer changes --
     so barrier placement is load-bearing in the check, not decorative in it.
  3. `__shfl_down_sync` implements the documented out-of-range behaviour
     (returns the caller's own value), so the "lanes 16..31 hold garbage" comment
     in `axis_rowdot` is modelled rather than assumed away.
  4. Every 16-byte access goes through `_ld_f4`/`_st_f4`/`_st_h8`, which check
     the flat index's alignment and name the defect if it fails. That is the
     check that would have caught the `__align__(16)` omission and the
     four-halves-in-a-`float4` size mismatch as host-side errors instead of
     CUDA error 716 on the card.
  5. `wmma` fragments are modelled as their LOGICAL 16x16 tiles. The register
     layout is opaque and is not what a correctness gate should be about; the
     values in and out of `mma_sync` are. Stated here because it is a modelling
     decision, and a reader is entitled to know which properties are unchecked.

WHAT IS AND IS NOT MODELLED
---------------------------
Global memory is visible immediately, with no pending buffer. That is sound for
these seven kernels and only for them, because none of them reads a global
location that another thread in the same launch has written: the GEMMs read
inputs and write a disjoint C, `axis_trisolve` reads D and writes a block-local X,
`axis_scan_*` read x and write a disjoint y. `assert_no_global_RAW` below is that
claim, made checkable -- a kernel that grew a global read-after-write would have
to extend this class, and the file says so rather than silently mis-modelling it.

Arithmetic is float32 with numpy's float32 rounding, so accumulation order
matches the device. Rounding is never what broke these kernels -- all seventeen
defects found so far in this project were structural -- so the emulator's job is
to be exact about structure, not about the last ulp.

This file is part of the hand-written arm, which is SUPPOSED to be CUDA. It is not
a constraint on `axis/rungs.py`; see `axis/verify.py` §1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

F32 = np.float32

SCAN_THREADS = 256
SCAN_VEC = 4
TRI_C = 64
TRI_BC = 16
TRI_NC = 4
BM, BN, BK, TM, TN, GTHREADS = 128, 128, 8, 8, 8, 256
F16_LDA = 16 + 8
F16_LDB = 64 + 8
G16_TILE = 64
G16_BM, G16_BN, G16_BK, G16_THREADS = 64, 64, 16, 128
F16_LDM = 20          # the accumulator's staging leading dimension


# ===========================================================================
# execution model
# ===========================================================================

class Tile2D:
    """Shared memory with a real barrier.

    Reads see COMMITTED state. Writes land in PENDING and become visible only at
    `sync()`. This is what makes `__syncthreads()` placement checkable: a kernel
    that reads a value another thread writes, without a barrier between, returns
    the stale value here exactly as it would on the card.
    """

    __slots__ = ("rows", "cols", "_c", "_p")

    def __init__(self, rows: int, cols: int, fill: float = 0.0):
        self.rows, self.cols = rows, cols
        self._c = np.full((rows, cols), F32(fill), dtype=np.float32)
        self._p: dict[tuple[int, int], np.float32] = {}

    def __getitem__(self, key):
        r, c = key
        return self._c[r, c]

    def __setitem__(self, key, val):
        r, c = key
        self._p[(r, c)] = F32(val)

    def sync(self) -> None:
        for (r, c), v in self._p.items():
            self._c[r, c] = v
        self._p.clear()

    def committed(self) -> np.ndarray:
        """The post-barrier contents, as a dense array. Read-only."""
        if self._p:
            raise AssertionError("pending writes at read-out: a barrier is missing")
        return self._c


class ScalarSmem:
    """1-D shared memory with the same pending semantics as `Tile2D`."""

    __slots__ = ("n", "_c", "_p")

    def __init__(self, n: int, fill: float = 0.0):
        self.n = n
        self._c = np.full(n, F32(fill), dtype=np.float32)
        self._p: dict[int, np.float32] = {}

    def __getitem__(self, i):
        return self._c[i]

    def __setitem__(self, i, v):
        self._p[i] = F32(v)

    def sync(self) -> None:
        for i, v in self._p.items():
            self._c[i] = v
        self._p.clear()

    def committed(self) -> np.ndarray:
        if self._p:
            raise AssertionError("pending writes at read-out: a barrier is missing")
        return self._c


def shfl_down_warp(vals: np.ndarray, delta: int) -> np.ndarray:
    """__shfl_down_sync(0xffffffff, v, delta), whole warp at once.

    Returns the SHUFFLED VALUE ONLY -- the partner's, not `own + partner's`.
    That is deliberate: the .cu spells it `acc += __shfl_down_sync(...)`, and if
    the emulator folded the addition in here, the `acc +=` would silently
    disappear from the transcription and the two would stop being comparable
    line for line. The caller must add its own register.

    Out-of-range source lanes: PTX leaves this undefined, and Volta+ returns the
    caller's own value. Modelling it that way is what makes the .cu's comment
    ("the last two are needed because a partial reduction leaves lanes 16..31
    holding garbage") REPRODUCIBLE rather than assumed. It is also why lane 0 is
    the only lane whose final value matters: every step of its chain reads a
    lane < 16, and lanes < 16 always have an in-range partner at every offset.
    """
    out = np.empty_like(vals)
    for lane in range(32):
        src = lane + delta
        out[lane] = vals[src] if src < 32 else vals[lane]
    return out


def shfl_up_warp(vals: np.ndarray, delta: int) -> np.ndarray:
    """__shfl_up_sync, same convention: partner's value only.

    Out-of-range (lane < delta) returns the caller's own value, which is why
    `SCAN_BODY` can add unconditionally and stay correct on the low lanes.
    """
    out = np.empty_like(vals)
    for lane in range(32):
        src = lane - delta
        out[lane] = vals[src] if src >= 0 else vals[lane]
    return out


# ---- 16-byte access alignment -------------------------------------------------

def _align_check(flat: int, elems: int, what: str, kernel: str) -> None:
    if flat % elems != 0:
        raise AssertionError(
            f"{kernel}: 16-byte {what} at flat index {flat} is not {elems * 4}-byte "
            f"aligned. On the device this is CUDA error 716 (misaligned address). "
            f"The usual cause is a missing __align__(16) on a __shared__ array, "
            f"or writing {elems} elements through a type sized for a different "
            f"count.")


def _ld_f4(a: np.ndarray, flat: int, what: str, kernel: str) -> np.ndarray:
    _align_check(flat, 4, f"load ({what})", kernel)
    return a[flat:flat + 4].astype(np.float32)


def _st_f4(a: np.ndarray, flat: int, v, what: str, kernel: str) -> None:
    _align_check(flat, 4, f"store ({what})", kernel)
    a[flat:flat + 4] = np.asarray(v, dtype=np.float32)


def _st_h8(a: np.ndarray, flat: int, v, kernel: str) -> None:
    _align_check(flat, 8, "fp16 store (8 halves)", kernel)
    a[flat:flat + 8] = np.asarray(v, dtype=np.float16)


def assert_no_global_RAW(kernels: dict) -> None:
    """State the modelling precondition as a checkable list, not a claim.

    Global memory is committed immediately in this emulator. That is only sound
    because none of these kernels reads a global location another thread in the
    same launch writes. Each entry names the reason.
    """
    reasons = {
        "axis_ew": "reads a,b,z; writes y at a disjoint index",
        "axis_rowdot": "reads x,w; writes y[row], one row per warp",
        "axis_gemm_f32": "reads A,B; writes C, disjoint per thread",
        "axis_gemm_f16": "reads A,B; writes C, disjoint per thread",
        "axis_gather": "reads tab,idx,s; writes y at a disjoint index",
        "axis_scan_v*": "reads x; writes y at a disjoint index",
        "axis_scan_scalar": "reads x; writes y at a disjoint index",
        "axis_trisolve": "reads D; writes X, one block-private slice per block",
    }
    for name in kernels:
        base = name.split("_v")[0]
        if not any(base.startswith(k.split("_v")[0][:8]) or k.startswith(base[:8])
                   for k in reasons):
            raise AssertionError(f"{name}: no recorded justification for the "
                                 f"immediate-global-visibility model")


# ===========================================================================
# 0.  floors
# ===========================================================================

def sim_fill(n: int) -> np.ndarray:
    o = np.zeros(n, dtype=np.float32)
    grid, block = 4096, 256
    for blk in range(grid):
        for tid in range(block):
            i = blk * block + tid
            if i < n:
                o[i] = F32(0.0)
    return o


def sim_copy(n: int) -> np.ndarray:
    a = np.arange(n, dtype=np.float32)
    o = np.zeros(n, dtype=np.float32)
    grid, block = 4096, 256
    for blk in range(grid):
        for tid in range(block):
            i = blk * block + tid
            if i < n:
                o[i] = F32(a[i] * F32(1.0))
    return o


# ===========================================================================
# 1.  axis_ew
# ===========================================================================

def sim_ew(a, b, z, n4: int, grid: int = 1280, block: int = 256) -> np.ndarray:
    a4 = np.asarray(a, np.float32).reshape(-1, 4)
    b4 = np.asarray(b, np.float32).reshape(-1, 4)
    z4 = np.asarray(z, np.float32).reshape(-1, 4)
    y = np.zeros((n4, 4), dtype=np.float32)
    stride = grid * block
    for blk in range(grid):
        for tid in range(block):
            i = blk * block + tid
            while i < n4:
                av, bv, zv = a4[i], b4[i], z4[i]
                y[i] = np.array([av[j] + bv[j] * zv[j] for j in range(4)], np.float32)
                i += stride
    return y.reshape(-1)


# ===========================================================================
# 2.  axis_rowdot
# ===========================================================================

def sim_rowdot(x, w, R: int, K: int, block: int = 256) -> np.ndarray:
    x = np.asarray(x, np.float32).reshape(R, K)
    w = np.asarray(w, np.float32).reshape(R, K)
    y = np.zeros(R, dtype=np.float32)
    warps_per_block = block >> 5
    grid = (R + warps_per_block - 1) // warps_per_block
    for blk in range(grid):
        for warp in range(warps_per_block):
            row = blk * warps_per_block + warp
            if row >= R:
                continue
            vals = np.zeros(32, dtype=np.float32)
            for lane in range(32):
                acc = F32(0.0)
                k = lane
                while k < K:
                    acc = F32(acc + F32(x[row, k] * w[row, k]))
                    k += 32
                vals[lane] = acc
            for off in (16, 8, 4, 2, 1):
                # acc += __shfl_down_sync(0xffffffff, acc, off)
                vals = F32(vals + shfl_down_warp(vals, off))
            y[row] = vals[0]
    return y


# ===========================================================================
# 3a. axis_gemm_f32
# ===========================================================================

def sim_gemm_f32(A, Bm, M: int, N: int, K: int) -> np.ndarray:
    A = np.asarray(A, np.float32)
    Bm = np.asarray(Bm, np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    gridx = (N + BN - 1) // BN
    gridy = (M + BM - 1) // BM
    for by in range(gridy):
        for bx in range(gridx):
            As = Tile2D(BK, BM)
            Bs = Tile2D(BK, BN)
            acc = np.zeros((GTHREADS, TM, TN), dtype=np.float32)
            for k0 in range(0, K, BK):
                # ---- stage A tile, transposed into As[kk][row]
                for tid in range(GTHREADS):
                    row, kh = tid >> 1, (tid & 1) * 4
                    v = np.zeros(4, dtype=np.float32)
                    if row0_ok(bx, by, row, M) and k0 + kh + 3 < K:
                        flat = (by * BM + row) * K + k0 + kh
                        v = _ld_f4(A.reshape(-1), flat, "A tile", "axis_gemm_f32")
                    for j in range(4):
                        As[kh + j, row] = v[j]
                # ---- stage B tile, NOT transposed: Bs[kk][col]
                for tid in range(GTHREADS):
                    k, c4 = tid >> 5, (tid & 31) * 4
                    v = np.zeros(4, dtype=np.float32)
                    if bx * BN + c4 + 3 < N:
                        flat = (k0 + k) * N + bx * BN + c4
                        v = _ld_f4(Bm.reshape(-1), flat, "B tile", "axis_gemm_f32")
                    for j in range(4):
                        Bs[k, c4 + j] = v[j]
                As.sync()
                Bs.sync()
                # ---- inner product
                for tid in range(GTHREADS):
                    ty, tx = tid >> 4, tid & 15
                    for kk in range(BK):
                        af = [As[kk, ty * TM + i] for i in range(TM)]
                        bf = [Bs[kk, tx * TN + j] for j in range(TN)]
                        for i in range(TM):
                            for j in range(TN):
                                acc[tid, i, j] = F32(
                                    fmaf(af[i], bf[j], acc[tid, i, j]))
                As.sync()
                Bs.sync()
            # ---- store
            for tid in range(GTHREADS):
                ty, tx = tid >> 4, tid & 15
                for i in range(TM):
                    row = by * BM + ty * TM + i
                    if row >= M:
                        continue
                    for j in range(TN):
                        col = bx * BN + tx * TN + j
                        if col < N:
                            C[row, col] = acc[tid, i, j]
    return C


def row0_ok(bx, by, row, M) -> bool:
    return (by * BM + row) < M


def fmaf(a, b, c):
    return np.float32(np.float32(a) * np.float32(b) + np.float32(c))


# ===========================================================================
# 3b. axis_gemm_f16
# ===========================================================================

def sim_gemm_f16(A, Bm, M: int, N: int, K: int) -> np.ndarray:
    """Fragments are modelled as their logical 16x16 tiles.

    `load_matrix_sync` gives every lane of a warp the same logical matrix in a
    per-lane register order, and `store_matrix_sync` reads it back in row-major
    order, so the logical value is what crosses the modelled boundary. The
    per-lane register layout is NOT checked -- it is opaque by design and is not
    where this study's defects have been.
    """
    A = np.asarray(A, np.float16)
    Bm = np.asarray(Bm, np.float16)
    C = np.zeros((M, N), dtype=np.float16)
    gridx = (N + G16_BN - 1) // G16_BN
    gridy = (M + G16_BM - 1) // G16_BM
    nwarps = G16_THREADS // 32
    for by in range(gridy):
        for bx in range(gridx):
            As = Tile2D(G16_BM + 8, F16_LDA)
            Bs = Tile2D(G16_BK + 8, F16_LDB)
            # one [2][2] of logical 16x16 per warp
            acc = np.zeros((nwarps, 2, 2, 16, 16), dtype=np.float32)
            for k0 in range(0, K, G16_BK):
                # ---- A tile: 64 rows x 16 k, 128 threads x 8 halves
                for tid in range(G16_THREADS):
                    a_r, a_kk = tid >> 1, (tid & 1) * 8
                    v = np.zeros(8, dtype=np.float32)
                    if (by * G16_BM + a_r) < M and k0 + a_kk + 7 < K:
                        flat = (by * G16_BM + a_r) * K + k0 + a_kk
                        _align_check(flat, 8, "A tile (8 halves)", "axis_gemm_f16")
                        v = A.reshape(-1)[flat:flat + 8].astype(np.float32)
                    for j in range(8):
                        As[a_r, a_kk + j] = v[j]
                # ---- B tile: 16 k x 64 n, 128 threads x 8 halves
                for tid in range(G16_THREADS):
                    b_k, b_c = tid >> 3, (tid & 7) * 8
                    v = np.zeros(8, dtype=np.float32)
                    if (bx * G16_BN + b_c + 7) < N and k0 + b_k < K:
                        flat = (k0 + b_k) * N + bx * G16_BN + b_c
                        _align_check(flat, 8, "B tile (8 halves)", "axis_gemm_f16")
                        v = Bm.reshape(-1)[flat:flat + 8].astype(np.float32)
                    for j in range(8):
                        Bs[b_k, b_c + j] = v[j]
                As.sync()
                Bs.sync()
                # ---- wmma: collective loads, then 4 mma_sync per warp
                for w in range(nwarps):
                    wm, wn = w >> 1, w & 1
                    Aa = As.committed()
                    Bb = Bs.committed()
                    af = [Aa[wm * 32 + i * 16: wm * 32 + i * 16 + 16, 0:16]
                          for i in range(2)]
                    bf = [Bb[0:16, wn * 32 + j * 16: wn * 32 + j * 16 + 16]
                          for j in range(2)]
                    for i in range(2):
                        for j in range(2):
                            acc[w, i, j] = F32(af[i] @ bf[j] + acc[w, i, j])
                As.sync()
                Bs.sync()
            # ---- epilogue: stage each fragment through cs[warp], then store
            cs = np.zeros((nwarps, 16 * F16_LDM), dtype=np.float32)
            vec_store = (N % 8 == 0)
            for w in range(nwarps):
                wm, wn = w >> 1, w & 1
                qrow, qcol = by * G16_BM + wm * 32, bx * G16_BN + wn * 32
                for i in range(2):
                    for j in range(2):
                        tile = acc[w, i, j]
                        cs[w] = 0.0
                        for r in range(16):
                            for c in range(16):
                                cs[w, r * F16_LDM + c] = tile[r, c]
                        # ONE staging buffer per warp: see the .cu's comment on
                        # the 0.96-relative-error defect this prevents.
                        rbase, cbase = qrow + i * 16, qcol + j * 16
                        for r in range(16):
                            if rbase + r >= M:
                                continue
                            c = 0
                            if vec_store:
                                while c + 7 < 16 and cbase + c + 7 < N:
                                    src = cs[w, r * F16_LDM + c: r * F16_LDM + c + 8]
                                    packed = np.asarray(src, np.float16)
                                    _st_h8(C.reshape(-1), (rbase + r) * N + cbase + c,
                                           packed, "axis_gemm_f16")
                                    c += 8
                            while c < 16 and cbase + c < N:
                                C[rbase + r, cbase + c] = F32(cs[w, r * F16_LDM + c])
            C = np.ascontiguousarray(C, dtype=np.float16)
    return C


# ===========================================================================
# 4.  axis_gather
# ===========================================================================

def sim_gather(tab, idx, s, n: int, grid: int = 640, block: int = 256) -> np.ndarray:
    tab = np.asarray(tab, np.float32)
    idx = np.asarray(idx, np.int64)
    s = np.asarray(s, np.float32)
    y = np.zeros(n, dtype=np.float32)
    stride = grid * block
    for blk in range(grid):
        for tid in range(block):
            i = blk * block + tid
            while i < n:
                y[i] = F32(tab[idx[i]] * s[i])
                i += stride
    return y


# ===========================================================================
# 5.  axis_scan_v<PT>  and  axis_scan_scalar
# ===========================================================================

def sim_scan(x, R: int, N: int, PT: int) -> np.ndarray:
    """`axis_scan_v{PT}` for one block per row, 256 threads, float4 loads."""
    if N != SCAN_THREADS * PT * SCAN_VEC:
        raise AssertionError(
            f"axis_scan_v{PT} covers N = 256*{PT}*4 = {SCAN_THREADS * PT * SCAN_VEC}; "
            f"got N={N}. `harness._scan_symbol` enforces the same precondition and "
            f"reports UNAVAILABLE rather than launching a kernel that would leave "
            f"the tail of every row stale.")
    x = np.asarray(x, np.float32).reshape(R, N)
    y = np.zeros((R, N), dtype=np.float32)
    for row in range(R):
        xr = x[row]
        warp_tot = ScalarSmem(SCAN_THREADS // 32)
        tmp = np.zeros((SCAN_THREADS, PT, 4), dtype=np.float32)
        total = np.zeros(SCAN_THREADS, dtype=np.float32)
        inc = np.zeros(SCAN_THREADS, dtype=np.float32)
        # --- phase: load PT float4s, sum them
        for tid in range(SCAN_THREADS):
            base = tid * PT * SCAN_VEC
            t = F32(0.0)
            for it in range(PT):
                f4 = xr.reshape(-1, 4)[base // SCAN_VEC + it]
                tmp[tid, it] = f4
                for j in range(4):
                    t = F32(t + f4[j])
            total[tid] = t
        # --- phase: warp shuffle inclusive scan
        #
        # Run per WARP, not per thread. A shuffle reads another lane's register,
        # so a per-thread transcription cannot see it at all -- and a per-thread
        # transcription of a shuffle is exactly the mistake the scan `run` bug
        # came from: the arithmetic is checkable, the EXECUTION is not. The first
        # version of this function added each thread's OWN value instead of its
        # partner's and returned a self-consistent, wrong answer that a
        # per-thread reimplementation of np.cumsum could not have distinguished.
        for w in range(SCAN_THREADS // 32):
            base = w * 32
            v = total[base:base + 32].copy()
            off = 1
            while off < 32:
                o = shfl_up_warp(v, off)
                v = np.where(np.arange(32) >= off, F32(v + o), v)
                off <<= 1
            inc[base:base + 32] = v
        # --- phase: lane 31 publishes the warp total
        for tid in range(SCAN_THREADS):
            if (tid & 31) == 31:
                warp_tot[tid >> 5] = inc[tid]
        warp_tot.sync()
        # --- phase: thread 0 turns warp totals into exclusive offsets
        run = F32(0.0)
        for i in range(SCAN_THREADS // 32):
            t = warp_tot[i]
            warp_tot[i] = run
            run = F32(run + t)
        warp_tot.sync()
        # --- phase: serial scan in registers, offset applied on the way out
        for tid in range(SCAN_THREADS):
            lane, warp = tid & 31, tid >> 5
            base = tid * PT * SCAN_VEC
            # `run` is declared ONCE, outside the float4 loop. The defect this
            # file was written to catch was it being declared INSIDE: each float4
            # would then be written as `offset + its own four elements` and the
            # running total of the previous float4s would be dropped, which is
            # exact at thread-relative positions 0-3 and wrong at 4-15.
            run_offset = F32(warp_tot[warp] + (inc[tid] - total[tid]))
            run = run_offset
            for it in range(PT):
                out = np.zeros(4, dtype=np.float32)
                for j in range(4):
                    out[j] = F32(tmp[tid, it, j] + run)
                    run = out[j]
                y[row].reshape(-1, 4)[base // SCAN_VEC + it] = out
    return y


def sim_scan_scalar(x, R: int, N: int) -> np.ndarray:
    """`axis_scan_scalar`: one element per thread, Hillis-Steele in shared."""
    if N > SCAN_THREADS:
        raise AssertionError(
            f"axis_scan_scalar gives each thread ONE element, so it is only "
            f"correct for N <= {SCAN_THREADS}; got N={N}.")
    x = np.asarray(x, np.float32).reshape(R, N)
    y = np.zeros((R, N), dtype=np.float32)
    for row in range(R):
        srun = ScalarSmem(SCAN_THREADS)
        for tid in range(SCAN_THREADS):
            if tid < N:      # load-bearing, not defensive; see the .cu
                srun[tid] = x[row, tid]
        srun.sync()
        off = 1
        while off < SCAN_THREADS:
            t = [F32(0.0)] * SCAN_THREADS
            for tid in range(SCAN_THREADS):
                if tid >= off and tid < N:
                    t[tid] = srun[tid - off]
            srun.sync()
            for tid in range(SCAN_THREADS):
                if tid < N:
                    srun[tid] = F32(srun[tid] + t[tid])
            srun.sync()
            off <<= 1
        for tid in range(SCAN_THREADS):
            if tid < N:
                y[row, tid] = srun[tid]
    return y


# ===========================================================================
# 6.  axis_trisolve
# ===========================================================================

def sim_trisolve(D, B: int) -> np.ndarray:
    """Blocked triangular inverse, one block per batch item, 256 threads.

    The phase boundaries below are the .cu's `__syncthreads()` calls, in order.
    The diagonal-block loop's barrier is at the TOP of the body, before the
    `if ((tid>>4) != i) continue`, so all 256 threads reach it every iteration
    and lockstep is exact.
    """
    D = np.asarray(D, np.float32).reshape(B, TRI_C, TRI_C)
    Xg = np.zeros((B, TRI_C, TRI_C), dtype=np.float32)
    for b in range(B):
        sD = Tile2D(TRI_C, TRI_C + 1)
        sX = Tile2D(TRI_C, TRI_C + 1)
        sS = Tile2D(TRI_BC, TRI_BC + 1)
        sT = Tile2D(TRI_BC, TRI_BC + 1)
        # ---- load D, zeroing the pad column
        for r in range(TRI_C):
            for tid in range(256):
                sD[r, TRI_C] = F32(0.0)
                c = tid
                while c < TRI_C:
                    sD[r, c] = F32(D[b, r, c])
                    c += 256
        sD.sync()
        sX.sync()
        sS.sync()
        sT.sync()
        # ---- diagonal blocks: X_kk = (I + D_kk)^-1 by ROW-wise substitution
        for k in range(TRI_NC):
            off = k * TRI_BC
            for i in range(TRI_BC):
                sD.sync(); sX.sync()          # __syncthreads() at the top of the body
                for tid in range(256):
                    c = tid & 15
                    if (tid >> 4) != i:
                        continue                # only 16 threads own row i
                    # The recursion is over ROWS, X[i][j] = -sum_{m<i} D[i][m] X[m][j],
                    # so the nonzero region is STRICTLY LOWER TRIANGULAR and the
                    # test is `c < i`. Written as `c > i` the kernel produces an
                    # upper-triangular X and never writes a strictly-lower entry --
                    # the on-card signature was ref_at_worst 0.0 where it wanted
                    # 0.5505707.
                    if c < i:
                        acc = F32(0.0)
                        for m in range(i):
                            acc = F32(acc + F32(sD[off + i, off + m] * sX[off + m, off + c]))
                        sX[off + i, off + c] = F32(-acc)
                    elif c == i:
                        sX[off + i, off + c] = F32(1.0)
                    else:
                        sX[off + i, off + c] = F32(0.0)
            sD.sync(); sX.sync()
        # ---- block-Schur merge, one block column at a time
        for j in range(TRI_NC):
            for i in range(j + 1, TRI_NC):
                sD.sync(); sX.sync(); sS.sync(); sT.sync()
                for tid in range(256):
                    r, c = tid >> 4, tid & 15
                    if r < TRI_BC and c < TRI_BC:
                        sS[r, c] = F32(0.0)
                sS.sync()
                for k in range(j, i):        # k starts at j, NOT j+1
                    sD.sync(); sX.sync(); sT.sync()
                    for tid in range(256):
                        r, c = tid >> 4, tid & 15
                        acc = F32(0.0)
                        if r < TRI_BC and c < TRI_BC:
                            for m in range(TRI_BC):
                                acc = F32(acc + F32(
                                    sD[i * TRI_BC + r, k * TRI_BC + m] *
                                    sX[k * TRI_BC + m, j * TRI_BC + c]))
                        if r < TRI_BC and c < TRI_BC:
                            sT[r, c] = acc
                    sT.sync()
                    for tid in range(256):
                        r, c = tid >> 4, tid & 15
                        if r < TRI_BC and c < TRI_BC:
                            sS[r, c] = F32(sS[r, c] + sT[r, c])
                    sS.sync()
                # X_ij = -X_ii . S
                sX.sync(); sS.sync()
                for tid in range(256):
                    r, c = tid >> 4, tid & 15
                    acc = F32(0.0)
                    if r < TRI_BC and c < TRI_BC:
                        for m in range(TRI_BC):
                            acc = F32(acc + F32(
                                sX[i * TRI_BC + r, i * TRI_BC + m] * sS[m, c]))
                    if r < TRI_BC and c < TRI_BC:
                        sX[i * TRI_BC + r, j * TRI_BC + c] = F32(-acc)
            sD.sync(); sX.sync(); sS.sync(); sT.sync()
        # ---- store the strictly lower part
        Dc, Xc = sD.committed(), sX.committed()
        for r in range(TRI_C):
            for tid in range(256):
                c = tid
                while c < TRI_C:
                    Xg[b, r, c] = Xc[r, c] if r > c else F32(0.0)
                    c += 256
    return Xg


# ===========================================================================
# 7.  the emulated rungs, keyed the way the harness keys them
# ===========================================================================

def emulate(key: str, d: dict, **kw) -> np.ndarray:
    """Run the reference arm for `key` on the same inputs the harness uses."""
    if key == "ew":
        return sim_ew(d["a"], d["b"], d["z"], d["n"] // 4)
    if key == "red":
        return sim_rowdot(d["x"], d["w"], d["R"], d["K"])
    if key == "gemm32":
        return sim_gemm_f32(d["A"], d["B"], d["M"], d["N"], d["K"])
    if key == "gemm16":
        return sim_gemm_f16(d["A"], d["B"], d["M"], d["N"], d["K"])
    if key == "gather":
        return sim_gather(d["tab"], d["idx"], d["s"], d["n"])
    if key == "scan":
        R, N = d["R"], d["N"]
        if N <= SCAN_THREADS:
            return sim_scan_scalar(d["x"], R, N)
        return sim_scan(d["x"], R, N, N // (SCAN_THREADS * SCAN_VEC))
    if key == "trisolve":
        return sim_trisolve(d["D"], d["B"])
    raise KeyError(key)


# ===========================================================================
# 8.  the gate's own falsifier: re-inject the two real defects
# ===========================================================================
# A gate that has never been observed to fail is not known to discriminate. The
# two defects this file was written to catch were both found on the card, so the
# only honest way to claim the host gate would have caught them is to put them
# back and watch it fail -- including reproducing the SIGNATURE, not merely the
# failure, because a wrong number in the wrong place is a different bug from a
# wrong number.
#
# On the card (v17, out/axis/v17/axis_report.json):
#     scan      relerr 1.1e-01, 56.2% of elements off, worst (733,1309)
#               ref 69.041  want 85.875;  exact at thread-relative 0-3,
#               wrong at 4-15
#     trisolve  ref_at_worst 0.0, want_at_worst 0.5505707, at [4135,10,8]
#
# The signatures are what identify the defects, so they are asserted, not just
# the relerr. A mutant that fails for the wrong reason is a mutant that proves
# nothing.

MUTANTS: list[tuple[str, tuple[str, str], str]] = [
    # (label, (find, replace), which function it targets)
    ("scan: `run` reset to loop-local",
     ("            run = run_offset\n"
      "            for it in range(PT):\n"
      "                out = np.zeros(4, dtype=np.float32)",
      "            for it in range(PT):\n"
      "                run = run_offset\n"
      "                out = np.zeros(4, dtype=np.float32)"),
     "scan"),
    ("trisolve: `c > i` (upper-triangular X)",
     ("                    if c < i:\n"
      "                        acc = F32(0.0)\n"
      "                        for m in range(i):",
      "                    if c > i:\n"
      "                        acc = F32(0.0)\n"
      "                        for m in range(i):"),
     "trisolve"),
    ("trisolve: block-Schur sum from k = j+1",
     ("                for k in range(j, i):        # k starts at j, NOT j+1",
      "                for k in range(j + 1, i):"),
     "trisolve"),
    ("scan: drop the `e`-equivalent offset (warp_tot) term",
     ("            run_offset = F32(warp_tot[warp] + (inc[tid] - total[tid]))",
      "            run_offset = F32(inc[tid] - total[tid])"),
     "scan"),
]


def _relerr(a, b) -> float:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(float(np.abs(b).max()), 1e-30))


def mutation_selftest(oracles: dict) -> list[dict]:
    """Re-inject each historical defect and confirm the emulator now disagrees.

    Returns one record per mutant. `caught` is the claim under test: a mutant
    that still agrees with the oracle means this gate does NOT have teeth for
    that defect class, and the caller must fail rather than note it.
    """
    import types

    src = Path(__file__).read_text()
    rows = []
    for label, (find, replace), target in MUTANTS:
        if find not in src:
            rows.append({"label": label, "caught": False, "relerr": None,
                         "detail": "MUTANT DID NOT APPLY -- the anchor text has "
                                   "changed, so this falsifier is testing nothing"})
            continue
        mod = types.ModuleType("simref_mutant")
        mod.__dict__["__file__"] = __file__
        mod.__dict__["__name__"] = "simref_mutant"
        exec(compile(src.replace(find, replace, 1), f"<mutant:{label}>", "exec"),
             mod.__dict__)
        d, err, detail = _run_mutant(mod, target, oracles)
        rows.append({"label": label, "caught": err > 1e-4, "relerr": err,
                     "detail": detail, "inputs": d})
    return rows


def _run_mutant(mod, target: str, oracles: dict):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    rungs = sys.modules.get("rungs")
    if rungs is None:
        import rungs  # noqa: F401
        rungs = sys.modules["rungs"]

    if target == "scan":
        R_, N_, PT = 3, 4096, 4
        d = rungs.in_scan({"R": R_, "N": N_})
        got = mod.sim_scan(d["x"], R_, N_, PT)
        want = oracles["scan"](d)
        e = _relerr(got, want)
        # the positional signature: exact at 0-3, wrong at 4-15
        bad = np.abs(np.asarray(got, np.float64)[0] - want[0]) > 1e-4
        sig = (f"row 0: exact at 0-3 = {not bad[0:4].any()}, "
               f"wrong at 4-15 = {bool(bad[4:16].any())}, "
               f"{bad.mean():.1%} of elements off")
        return d, e, sig

    if target == "trisolve":
        d = rungs.in_trisolve({"B": 8, "C": 64, "BC": 16})
        got = mod.sim_trisolve(d["D"], 8)
        want = oracles["trisolve"](d)
        e = _relerr(got, want)
        i = np.unravel_index(int(np.abs(np.asarray(got, np.float64) - want).argmax()),
                             got.shape)
        sig = (f"worst at {tuple(int(x) for x in i)}: "
               f"got {float(got[i]):.6f} want {float(want[i]):.6f}")
        return d, e, sig

    raise KeyError(target)
