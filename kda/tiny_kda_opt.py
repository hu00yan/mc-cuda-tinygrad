"""tiny_kda_opt.py - KDA in TinyGrad, with the UT transform done the way the
kernel does it.

`tiny_kda.py` is the naive port and stays untouched as the control.  It solves
the UT transform (the inverse of a unit lower-triangular `[C,C]` matrix) with

    for i in range(C):        # 64 columns, independently solvable
        for r in range(1, C): # 63 substitutions, re-run per column
            b = b - e_r * (A[:, :, :, r, :r] * b[:, :, :, :r]).sum(-1)
        M = M + e_i_row * b.reshape(..., C, 1)

Two separate things are wasted there, and it is worth being precise about which
cost is which, because they are not the same cost and only one of them is what
the loop count suggests:

  * the 64 columns are 64 INDEPENDENT triangular solves against the same A, so
    the graph is 64 parallel chains -- but the right-hand side is `Diag(beta)`,
    so all 64 together are just `A^-1`.  The column loop recomputes the same
    thing 64 times over in *sequential structure* rather than in arithmetic.
  * forward substitution on a unit lower-triangular matrix blocks.  Split `C`
    into `NC` blocks of `BC`, invert the `NC` diagonal blocks (which are
    independent of each other, so they batch), then recover the off-diagonal
    blocks with a block-row sweep whose innermost sum is a matmul.

MEASURED at B=2, T=512, H=HV=4, K=V=64, C=64, NT=8 (so B*NT*HV = 64 systems),
on Metal, tinygrad 0.14.0 -- the UT transform alone, same inputs, same device:

                        arithmetic   longest   emitted   device
                        uops         chain     kernels   ms
    tiny_kda (naive)      20,354        381       6,021   270.6
    this file                89         89          24     1.12

`4,032` is the number of inner-loop iterations, i.e. of TinyGrad ops, and it is
not the number the hardware executes: TinyGrad does not keep that chain in
registers, so each op is realised and the solve becomes 6,021 kernels, worse
than the op count.  (Kernels execute in stream order, so 6,021 is also the
physically serial step count.)  The graph-theoretic longest chain is only 381
because the 64 column solves are mutually parallel in the DAG; the structure
the port actually asks for is 64 independent 252-step column chains plus a
64-step accumulation, and the 6,021 is what that costs.

Device times are the steady state: 3 realisations in one process with the
program cache warm.  A single cold run of the naive solve measured anywhere from
147 to 1,323 ms across repeats, which is why the kernel count is the number to
quote rather than the milliseconds.  Method, and two negative results about
which parts of the reference should NOT be copied, in
notes/24-kda-ut-transform.md.

This file does the blocking.  `(BC-2) + (NC-1)` = 17 dependent levels instead of
4,032 iterations, and the column loop is gone rather than batched.

Where this comes from.  The released kernel is
`fla/ops/kda/chunk_intra.py::chunk_kda_fwd_kernel_inter_solve_fused`
(commit 13e67cb, 2026-09-08).  It fixes `BC = 16`, `NC = BT/BC`, and does:

  * four `for i in range(2, BC)` row-substitution loops, one per diagonal
    16x16 block, all four live in registers at the same time;
  * then six `tl.dot` expressions -- `b_Ai10`, `b_Ai21`, `b_Ai20`, `b_Ai32`,
    `b_Ai31`, `b_Ai30` -- which are 16x16x16 matmuls, in a 3-deep DAG
    (`b_Ai30` needs `b_Ai20` which needs `b_Ai10`).

So `kda/SPEC.md` §4.4 is right on both counts -- four diagonal row-loops and
six off-diagonal matmuls -- and §3.3's "C-1 dependent steps" is right for the
UNBLOCKED form only.  (Its own pseudocode in §3.3 is not: it negates the
strictly-lower block and then adds I, which computes `(I - StrictTril(D))^-1`,
contradicting its own Eq. 6 and both implementations.  The same sign error is
live in `ref.py::_forward_substitution`'s docstring, which is why it passes
`-Btri` and why that works.  See notes/24-kda-ut-transform.md §1 for the
`np.linalg.inv` check that pins both.)

The sweep this file implements is that DAG written once, for any `NC`:

    X_0 = Ainv_00
    X_i = Ainv_ii ( E_i - sum_{k<i} A_ik X_k )                (*)

with `E_i` the block-identity row `i` (`I` in column block `i`, zero elsewhere),
so `i = 0` needs no special case and `(*)` covers the diagonal blocks too.
Unrolling `(*)` at `NC = 4` reproduces fla's six expressions exactly.

Everything else in the operator is unchanged except for work that was sitting
*inside* the inter-chunk scan and did not depend on the carried state; see
`kda_chunk` for the list.  The oracle is `kda/ref.py` (float64, 17/17).
"""
import numpy as np
from tinygrad import Tensor, dtypes

F = dtypes.float32

__all__ = ["kda_chunk", "ut_transform", "default_block"]


def default_block(C: int) -> int:
    """`BC`, the secondary chunk width, for a primary chunk of `C`.

    `BC = 16` is the kernel's fixed value (`chunk_intra.py`: `BC = 16`) and it is
    what `flint.fla.ops.kda.chunk_intra` is written around, so it is the
    default here too; `notes/24-kda-ut-transform.md` reports the measured
    alternative.
    """
    return min(16, C)


# ---------------------------------------------------------------------------
# UT transform:  M = (I + StrictTril(Diag(beta) Z))^-1  (paper Eq. 6)
# ---------------------------------------------------------------------------
def ut_transform(a_strict, C, BC=None):
    """`(I + a_strict)^-1` for a STRICTLY lower `[N, C, C]` batch.

    `a_strict` is `StrictTril(Diag(beta) Z)`, so the system matrix
    `A = I + a_strict` is unit lower triangular and the diagonal is 1: there is
    no division anywhere, only additions.

    The block sweep, in the order the graph is built:

    step 1  split `A` into `NC x NC` blocks of `BC x BC`; pull the `NC`
            diagonal blocks out with a block-diagonal mask and one contraction
            over the (now single-valued) column-block axis.
    step 2  invert those `NC` blocks.  They are independent of each other, so
            they invert together -- the `NC` axis is just another batch axis and
            the loop runs `BC-2` times, not `NC*(BC-2)`.
    step 3  `NC-1` block-row levels.  Each is one batched matmul over the
            preceding block rows, one contraction over that axis, and one more
            matmul by the diagonal inverse.

    The sign is the one thing here that can be wrong by a sign and still look
    like a triangular solve, so it is worth stating: the in-place row
    substitution

        X <- A0;  for i:  X[i,:i] += X[i,:i] @ X[:i,:i];  return X + I

    returns `inv(I - A0)`, not `inv(I + A0)`.  That is why step 2 feeds it
    `-A0_ii`.  (It is also why `kda/ref.py::_forward_substitution` is documented
    as returning `(I+StrictTril(D))^-1` for input `D` when it returns `(I-D)^-1`
    -- `ref.py` gets the right answer by passing it `-Btri`, and so does this.)
    """
    BC = default_block(C) if BC is None else BC
    assert C % BC == 0, f"chunk C={C} must be a multiple of BC={BC}"
    N = a_strict.shape[0]
    NC = C // BC
    N2 = N * NC

    # ---- layout -----------------------------------------------------------
    # Ab  [N, NC(row), BC, NC(col), BC]  the plain C x C matrix, blocked
    # Ak  [N, NC, NC, BC, BC]            block (i,k) at Ak[:, i, k]
    Ab = a_strict.reshape(N, NC, BC, NC, BC)
    Ak = Ab.permute(0, 1, 3, 2, 4)

    # ---- step 1: the NC diagonal blocks, as [N, NC, BC, BC] ----------------
    # A block-diagonal mask on the two block axes, then contract the column-block
    # axis away.  The mask is built by reshape on BOTH block axes: a stack of
    # `None`s applied to a 2-D array makes it SIX-dimensional, not five, and the
    # broadcast then lands on the wrong axis without raising.
    dmask = Tensor(np.eye(NC, dtype=np.float32).reshape(1, NC, 1, NC, 1))
    D = -(Ab * dmask).sum(axis=3)                       # [N,NC,BC,BC] = -A0_ii

    # ---- step 2: invert all NC of them at once -----------------------------
    # One batched [1,i]x[i,i] matmul per row i, batched over (n, block).  This is
    # where the 64 independent column solves of the naive port go: the right-hand
    # side is Diagonal(beta), so all 64 columns together are just A^-1, and once
    # the problem is blocked, the 4 diagonal blocks invert *simultaneously* --
    # the NC axis is a batch axis, so this loop runs BC-2 times, not NC*(BC-2).
    #
    # The write-back is an indexed assignment rather than an accumulate.  It does
    # not cost an extra kernel: the whole solve realises to 24 kernels on Metal,
    # i.e. under 2 per level, so the setitem fuses into its consumer.
    #
    # The loop starts at i=2, not 1: row 1's correction is A0[1,0]*X[0,0] and
    # X[0,0] is A0[0,0] = 0, so the step is identically zero.  The kernel starts
    # its loops at 2 for the same reason.
    for i in range(2, BC):
        corr = D[:, :, i, :i].reshape(N2, 1, i) @ D[:, :, :i, :i].reshape(N2, i, i)
        D[:, :, i, :i] += corr.reshape(N, NC, i)
    D = D + Tensor(np.eye(BC, dtype=np.float32))

    # ---- step 3: the block-row sweep --------------------------------------
    # X is carried with its column blocks FLATTENED, [N, NC, BC, NC*BC].  That
    # is what makes the sum in (*) a single matmul with no transpose: the
    # contraction index p of A_ik[b,p] X_k[p,j,c] is the last axis of the left
    # operand and the second-to-last of the right, which is what `@` needs.
    #
    # E_i is the block-identity row i, as a [1,1,BC,NC*BC] constant.  Left-
    # multiplying Ainv_ii by it both PLACES that block in column block i and
    # pads the rest of the row with zeros, so the level-0 seed is one op instead
    # of an indexed write into a zero tensor.
    #
    # The leading 1s are not decoration.  tinygrad's `dot` gives a lower-rank
    # right operand ONE batch axis; it will not insert a second one to match a
    # higher-rank left, so a 4-D @ 3-D here fails on shapes rather than
    # broadcasting.  Every operand of every matmul below is the same rank.
    Erow = Tensor(np.eye(NC, dtype=np.float32)[:, None, :, None]
                  * np.eye(BC, dtype=np.float32)[None, :, None, :])   # [NC,BC,NC,BC]
    Erow = Erow.reshape(NC, 1, BC, NC * BC)
    # X is [N, NC, BC, NC*BC] and stays that way.  Seeding it as [N, 1, BC, NC*BC]
    # and adding the later block rows does NOT work: a size-1 axis in broadcasting
    # REPEATS its data instead of padding with zeros, so the first accumulation
    # would produce row0 + row1 in block row 1.  The one-hot below costs one
    # elementwise op on a [C,C] tensor and makes the shape honest from the start.
    sel0 = Tensor(np.eye(NC, dtype=np.float32)[:, 0])
    # X_0 = Ainv_00.  The general level-i expression with an empty sum is exactly
    # this, so the seed is not a special case -- it is the same left-multiply.
    X = sel0.reshape(1, NC, 1, 1) * (D[:, 0].reshape(N, BC, 1, BC)
                                     @ Erow[0]).reshape(N, 1, BC, NC * BC)
    for i in range(1, NC):
        # sum_{k<i} A_ik X_k, as ONE batched matmul over the k axis.  The
        # leading `N*i` merges (n, i); k is carried inside that reshape and
        # contracted away by the sum on the next line.
        acc = (Ak[:, i, :i].reshape(N * i, BC, BC)
               @ X[:, :i].reshape(N * i, BC, NC * BC)).reshape(N, i, BC, NC * BC)
        acc = acc.sum(axis=1, keepdim=True)               # [N, 1, BC, NC*BC]
        Xi = (D[:, i].reshape(N, BC, 1, BC) @ (Erow[i] - acc)).reshape(N, 1, BC, NC * BC)
        # The one-hot selects X's BLOCK-ROW axis, and `Xi` is one row, so it has
        # to be broadcast in at axis 1 explicitly.  If you forget, the multiply
        # still succeeds whenever NC == BC and quietly selects the wrong axis; at
        # C=4, BC=2 that is a 1.14 relative error with nothing raised anywhere.
        sel = Tensor(np.eye(NC, dtype=np.float32)[:, i])                # [NC]
        X = X + sel.reshape(1, NC, 1, 1) * Xi
    return X.reshape(N, C, C)


# ---------------------------------------------------------------------------
# the operator
# ---------------------------------------------------------------------------
def _causal_mask(C, diagonal):
    """[C, C] 0/1 causal mask, diagonal included or excluded."""
    i = np.arange(C)[:, None]
    j = np.arange(C)[None, :]
    return (i >= j if diagonal else i > j).astype(np.float32)


def kda_chunk(q, k, v, g, beta, scale=None, initial_state=None, chunk_size=64,
              block=None):
    """TinyGrad KDA, blocked UT transform.  All inputs numpy; returns (o, S).

    Shapes: q,k [B,T,H,K]; v [B,T,HV,V]; g [B,T,HV,K] log space; beta [B,T,HV].
    `block` overrides `BC` for the UT transform; it only needs to divide
    `chunk_size`.

    Same operator as `tiny_kda.kda_chunk` and `kda/ref.py::kda_chunk`, verified
    against `kda_recurrent` (the token-at-a-time definition) to the same
    tolerance.  The differences are all in *how* the work is arranged:

    UT transform (`ut_transform` above)
        4,032 inner-loop iterations / 6,021 kernels -> 17 dependent levels and
        24 kernels.  The 64 independent column solves are gone entirely rather
        than batched, because the RHS is diagonal: the whole 64-column loop was
        computing `A^-1` one column at a time.

    `M` is never materialised
        `M = Ainv Diag(beta)` and `M @ X = Ainv @ (beta (.) X)`, so `beta` is
        applied to the right-hand side of the two dense matmuls instead.  The
        naive port accumulates `M` over 64 sequential rank-1 outer products and
        then multiplies it by `Diag(beta)`, both on a `[C,C]` tensor.

    one decay tensor instead of two
        `Dl` (strictly lower) and `Dq` (causal, diagonal in) differ on exactly
        one band -- the diagonal -- where `Dl` is 0 and `Dq` is `exp(0) = 1`.
        Both are built from the same masked log difference, and `Dl`'s diagonal
        is exactly 0 because it comes out of `exp(-1e4)`, which the naive port
        already relies on.  So `Dq = Dl + eye`, i.e. one `exp` over a
        `[B,NT,HV,C,C,K]` tensor instead of two, and one add.

    `Aqk` and the inter-chunk decay leave the scan
        The `NT` inter-chunk steps are a genuine scan -- chunk n's output is
        chunk n+1's input and there is no way to reassociate it.  But three of
        the four things computed inside that loop never touch the carried state:
        `Aqk[n]` (a `[B,HV,C,C,K]` elementwise-and-reduce, the single largest
        piece of per-step work in the whole scan), `exp(g_last - g_cum[n])` (an
        `[B,HV,C,K]` `exp`), and `gam[n] * q[n]`.  All three are hoisted and
        evaluated once for every chunk at once, which takes the per-step work
        down to the two matmuls and the state update that actually depend on
        `S`.

    What is NOT changed
        the `NT`-step scan itself, the sign convention of the UT transform, the
        `exp(gc[c] - gc[j])` difference form (never `K / Gamma`), plain `q` in
        `Aqk`, `scale` on `q` only, and float32 throughout.
    """
    C = chunk_size
    q = np.asarray(q, dtype=np.float32); k = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32); g = np.asarray(g, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    G = HV // H
    NT = T // C
    assert T % C == 0, f"T={T} not divisible by C={C}"
    if scale is None:
        scale = K ** -0.5

    # ---- _prep: GVA head repeat, scale on q only ----
    q = np.repeat(q, G, axis=2) * np.float32(scale)     # [B,T,HV,K]
    k = np.repeat(k, G, axis=2)
    beta = beta.astype(np.float32)

    # ---- to chunks: [B,T,HV,X] -> [B,NT,HV,C,X] ----
    def ch(x):
        return x.reshape(B, NT, C, HV, -1).transpose(0, 1, 3, 2, 4)

    tq, tk, tv, tgc = (Tensor(ch(x)) for x in (q, k, v, g))
    # beta has no trailing feature axis, so it gets its own reshape -- `ch`
    # always returns 5-D and would add a spurious length-1 axis here.
    tb = Tensor(beta.reshape(B, NT, C, HV).transpose(0, 1, 3, 2))    # [B,NT,HV,C]

    # ---- 1. within-chunk cumsum of the log gate ----
    g_cum = tgc.cumsum(axis=-2)                          # [B,NT,HV,C,K]
    gam = g_cum.exp()                                    # Gamma, <= 1

    # ---- 2/3. decayed gram, strictly lower ----
    # Dl[p,j,d] = exp(g_cum[p,d] - g_cum[j,d]) for j < p, else 0.
    #
    # The mask is applied to the EXPONENT DIFFERENCE, not to the exponentials.
    # Masking after the exp still evaluates exp() on the upper triangle, where
    # the difference is positive and unbounded -- inf * 0 = NaN, and on a
    # fast-math backend exp(-1e4) is not guaranteed to be exactly 0 either.  So
    # the masked entries are pushed to -1e4 and never exponentiated as such.
    gcp = g_cum.reshape(B, NT, HV, C, 1, K)
    gcj = g_cum.reshape(B, NT, HV, 1, C, K)
    m_strict = Tensor(_causal_mask(C, False)[None, None, None, :, :, None])
    Dl = ((gcp - gcj) + (m_strict - 1.0) * 1e4).exp()   # [B,NT,HV,C,C,K]

    # Z[p,j] = sum_d Dl[p,j,d] k[p,d] k[j,d].  The [C,C,K] decay is a transient:
    # nothing between here and `Dl`'s last use asks for its shape, and TinyGrad
    # fuses the exp and both multiplies and the reduction into one kernel, so it
    # never becomes a buffer.  k must broadcast over BOTH block axes, which needs
    # explicit 1s -- a negative-axis unsqueeze on a 5-D tensor lands the new axis
    # before HV, not after.
    k_p = tk.reshape(B, NT, HV, C, 1, K)
    k_j = tk.reshape(B, NT, HV, 1, C, K)
    Z = (Dl * k_p * k_j).sum(-1)                         # [B,NT,HV,C,C]
    a_strict = Z * tb.unsqueeze(-1)                      # StrictTril(Diag(beta) Z)
    # Z is already strictly lower -- `Dl` was masked -- so this IS the strictly
    # lower part of A = I + StrictTril(Diag(beta) Z) and the unit diagonal is
    # added inside `ut_transform`.

    # ---- 4. UT transform: blocked, batched over every axis ----
    Ainv = ut_transform(a_strict.reshape(B * NT * HV, C, C), C, block)  # [N,C,C]

    # ---- 5/6. WY representation, with Diag(beta) folded into the RHS ----
    # M = Ainv Diag(beta), so M @ X = Ainv @ (beta (.) X).  Term by term this is
    # the same sum -- column c of the result is scaled by beta_c either way -- so
    # it is a rearrangement, not an approximation, and it removes both the
    # materialised M and a [C,C] multiply.
    bscale = tb.unsqueeze(-1)                            # [B,NT,HV,C,1]
    w = Ainv @ (bscale * gam * tk).reshape(B * NT * HV, C, K)
    u = Ainv @ (bscale * tv).reshape(B * NT * HV, C, V)
    w = w.reshape(B, NT, HV, C, K)
    u = u.reshape(B, NT, HV, C, V)

    # ---- 7. inter-chunk scan, unrolled over NT ----
    # Everything below is hoisted out of the scan because none of it depends on
    # the carried state S.  Building them inside the loop made each of the NT
    # dependent steps carry a [B,HV,C,C,K] elementwise-and-reduce (`Aqk`) or a
    # [B,HV,C,K] `exp` that had nothing to do with the recurrence.
    S = np.zeros((B, HV, K, V), dtype=np.float32)
    if initial_state is not None:
        S = S + np.asarray(initial_state, dtype=np.float32)
    tS = Tensor(S)

    g_last = g_cum[:, :, :, -1]                          # [B,NT,HV,K] = log gamma^C
    # 7d's decay, gamma^C / gamma^n, as a log difference -- never the division.
    kgC = (g_last.reshape(B, NT, HV, 1, K) - g_cum).exp() * tk
    # 7d's Diag(exp(gC)) factor.
    sdecay = g_last.exp()                                # [B,NT,HV,K]
    # 7c's Gamma (.) Q.  And 7b's Aqk, the whole [B,NT,HV,C,C] at once, causal
    # with the diagonal INCLUDED -- which is `Dl` plus the identity band, since
    # the two decays differ only there.
    qg = gam * tq
    Dq = Dl + Tensor(np.eye(C, dtype=np.float32)[None, None, None, :, :, None])
    Aqk = (Dq * tq.reshape(B, NT, HV, C, 1, K)
           * tk.reshape(B, NT, HV, 1, C, K)).sum(-1)      # [B,NT,HV,C,C]

    parts = []
    for n in range(NT):
        # 7a  v_new = u - w @ S
        v_new = u[:, n] - w[:, n] @ tS                     # [B,HV,C,V]
        # 7c  o = (Gamma (.) Q) S + Aqk v_new.  `Aqk` uses PLAIN q (the decay is
        # already inside Dq), matching the asymmetry in ref.py's header note 3.
        parts.append(qg[:, n] @ tS + Aqk[:, n] @ v_new)
        # 7d  S = Diag(exp(gC)) S + ((gamma^C/gamma^n) (.) K)^T v_new
        tS = tS * sdecay[:, n].unsqueeze(-1) + \
            kgC[:, n].permute(0, 1, 3, 2) @ v_new

    # ref.py ends with o.transpose(0, 1, 3, 2, 4).reshape(B, T, HV, V), so the
    # flat token index runs n*C + c.  Permuting to [B,NT,HV,C,V] and reshaping
    # instead would interleave the HEAD axis into the token index, giving every
    # token another head's output with no error raised.
    o = Tensor.stack(*parts)                               # [NT,B,HV,C,V]
    o = o.permute(1, 0, 3, 2, 4).reshape(B, T, HV, V)
    return o.numpy(), tS.numpy()
