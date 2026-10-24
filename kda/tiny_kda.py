"""tiny_kda.py - KDA (Kimi Delta Attention) in TinyGrad.

Structure of the chunked algorithm, and where it fights a graph compiler.

kda/ref.py (validated, 17/17, chunked == per-token recurrence to 5e-15) runs
these steps. The annotations below are the ones that matter for the port:

  1  g_cum = cumsum(g) over the chunk axis          sequential, but a scan with a
                                                   known trip count -> unrollable
  2  Z[c,j] = (k . exp(gc[c]-gc[j])) . k[j], j<c    dense [B,NT,HV,C,C,K]
  3  Btri = StrictTril(Diag(beta) Z)
  4  M = fwdsub(-Btri)^-1 . Diag(beta)              SEQUENTIAL over C, in-row
  5  w = M @ (Gamma . k)                            dense
  6  u = M @ v                                      dense
  7  for n in range(NT):                            SEQUENTIAL over chunks, S
                                                   carried in [B,HV,K,V]
     7a   v_new = u - w @ S
     7b   Aqk = Tril((q . exp(gc[c]-gc[j])) . k[j])
     7c   o[n] = (Gamma . q) @ S + Aqk @ v_new[n]
     7d   S = Diag(gc[C]) S + ((gC-g).k)^T @ v_new[n]

Steps 4 and 7 are the two that a static-shape graph cannot express directly:

  step 4 is a forward substitution -- a triangular solve, C sequential
    substitutions each depending on the previous row. There is no batched BLAS
    for it. Options: unroll C=64 iterations (fine, trip count known), or one
    `inverse()` on the triangular matrix (C^3, and numerically different).

  step 7 is a scan over NT chunks carrying a [B,HV,K,V] state. Again a known
    trip count (T/C), so unrollable, but each iteration is a full K x V state
    update -- which is exactly the shape of operation that needs a real
    sequential loop and does not want to be a chain of materialised tensors.

The interesting question for this project is not "can TinyGrad express these"
(it can, by unrolling) but "what does unrolling cost, and does TinyGrad's
scheduler fuse the result into anything resembling the fused Triton kernel".

This module implements the dense-recompute variant: steps 1-6 vectorised with
a single batched triangular inverse for step 4, and step 7 unrolled.
"""
import numpy as np
from tinygrad import Tensor, dtypes

F = dtypes.float32


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def tril_mask(C, lower, diagonal):
    """[C, C] 0/1 mask. `lower` selects strictly-lower vs causal-with-diagonal."""
    i = np.arange(C)[:, None]
    j = np.arange(C)[None, :]
    m = (i > j) if lower else (i >= j)
    if diagonal and not lower:
        m = (i >= j)
    elif diagonal and lower:
        m = (i >= j)
    else:
        m = m & (i != j) if lower else m
    return m.astype(np.float32)


def _causal_decay(g_cum, lower, diagonal):
    """exp(g_cum[p] - g_cum[j]) masked to the causal band.

    g_cum : [B, NT, HV, C, K]
    out   : [B, NT, HV, C, C, K]   (p is the query slot, j the key slot)

    NOTE this is [C, C, K], not [C, C]. The paper writes the decayed gram as
    (G . K)(K / G)^T which is [C, C]; the per-channel form is what the fused
    kernel actually evaluates, and it is 64x bigger at C=64. That material is
    the single biggest memory item in the whole operator and the first place a
    compiler can help by fusing the exp away.
    """
    B, NT, HV, C, K = g_cum.shape
    gp = g_cum[:, :, :, None, :, :]          # [B,NT,HV,1,C,K]  p
    gj = g_cum[:, :, :, None, :, :]          # placeholder, replaced below
    # want gp[..., p, d] - gj[..., j, d]
    d = gp - g_cum[:, :, :, None, :, :].transpose(0, 1, 2, 4, 3, 5) \
        if False else (g_cum[:, :, :, None, :, :] - g_cum[:, :, :, None, :, :].transpose(0, 1, 2, 4, 3, 5))
    m = tril_mask(C, lower, diagonal)[None, None, None, :, :, None]
    return (np.exp(d) * m).astype(np.float32)


def kda_chunk(q, k, v, g, beta, scale=None, initial_state=None, chunk_size=64):
    """TinyGrad KDA. All inputs numpy; returns (o, S) as numpy.

    Shapes: q,k [B,T,H,K]; v [B,T,HV,V]; g [B,T,HV,K] log space; beta [B,T,HV].
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
    def chs(x):
        return x.reshape(B, NT, C, HV).transpose(0, 1, 3, 2)

    tq, tk, tv, tgc = (ch(x) for x in (q, k, v, g))
    tb = chs(beta)

    tq = Tensor(tq); tk = Tensor(tk); tv = Tensor(tv); tgc = Tensor(tgc); tb = Tensor(tb)

    # ---- 1. within-chunk cumsum of the log gate ----
    g_cum = tgc.cumsum(axis=-2)                          # [B,NT,HV,C,K]

    # ---- 2/3. decayed gram, strictly lower ----
    # Dl[p,j,d] = exp(g_cum[p,d] - g_cum[j,d]) * (p > j)
    # [B,NT,HV,C(p),1,K] - [B,NT,HV,1,C(j),K] -> [B,NT,HV,C,C,K]
    gp = g_cum.reshape(B, NT, HV, C, 1, K)               # p axis
    gj = g_cum.reshape(B, NT, HV, 1, C, K)               # j axis
    Dl = (gp - gj).exp()
    # NOTE: the mask must be applied to the EXPONENT DIFFERENCE, not to the
    # exponentials. Masking after the exp still evaluates exp() on the
    # upper-triangle entries, where the difference is very negative and
    # therefore underflows to 0 -- which is fine numerically, but it is a
    # different (and much larger) graph. Multiply the difference by a large
    # negative constant instead, so the masked entries exp to exactly 0 with
    # no dependence on how negative they would have been.
    m_strict = Tensor(tril_mask(C, True, False)[None, None, None, :, :, None])
    diff = gp - gj
    Dl = (diff + (m_strict - 1.0) * 1e4).exp()
    # Z[p,j] = sum_d Dl[p,j,d] * k[p,d] * k[j,d]
    # Dl is [B,NT,HV,C(p),C(j),K], so k must be broadcast along j and along p.
    # reshape with explicit 1s, not unsqueeze: negative-axis unsqueeze on a
    # 5-D tensor lands the new axis in the wrong place (before HV, not after).
    k_p = tk.reshape(B, NT, HV, C, 1, K)     # broadcast over j
    k_j = tk.reshape(B, NT, HV, 1, C, K)     # broadcast over p
    Z = (Dl * k_p * k_j).sum(-1)             # [B,NT,HV,C,C]
    Btri = Z * tb.unsqueeze(-1)                            # Diag(beta) Z

    # ---- 4. UT transform: forward substitution, UNROLLED -------------------
    #
    # ref.py:  Minv = _forward_substitution(-Btri);  M = Minv * Diag(beta)
    # and `_forward_substitution` solves (I + StrictTril(X)) M = I, so passing
    # -Btri means the system matrix is  A = I + StrictTril(Btri).  Sign matters:
    # using I - StrictTril(Btri) here gives a relative error of 1.0, i.e. a
    # completely different operator, and the tests catch it.
    #
    # TinyGrad has no triangular solve and no general `inverse()` (only `qr`
    # and `svd`), so this is where the port has to choose. Unrolled forward
    # substitution is the faithful one and is what the real kernel does; the
    # trip count is C, a compile-time constant, so unrolling is legitimate.
    #   A @ M = Diag(beta),  A = I + StrictTril(Btri) with unit diagonal
    #   M[:, :, i] = beta_i * e_i  -  sum_{j<i} A[j,i] * M[:, :, j]
    M = Tensor(np.zeros((B, NT, HV, C, C), dtype=np.float32))
    # tinygrad's ufix() raises on a bare numpy array, so every constant that
    # meets an operator has to be a Tensor, including the unit vectors.
    eyeT = [Tensor(np.eye(C, dtype=np.float32)[i]) for i in range(C)]   # e_i, 1-D
    eye_all = Tensor(np.eye(C, dtype=np.float32))                       # erow[i] = e_i
    # A = I + StrictTril(Btri), built on-device. A `np.tril(Btri.numpy())`
    # round trip would realize the graph to host mid-way, forcing a sync and a
    # re-upload; it was the only place this port left the device.
    A = eye_all + Btri * Tensor(                       # [B,NT,HV,C,C]
        np.tril(np.ones((C, C), dtype=np.float32), -1)[None, None, None])
    # Solve column i of  A M = Diag(beta)  by forward substitution WITHIN that
    # column. The subtlety that cost an afternoon: the elimination is
    #   b[r] = rhs[r] - sum_{j<r} A[r,j] * b[j]
    # i.e. row-wise inside the column being built. Subtracting whole previously
    # solved COLUMNS (A[i,j] * M[:,j]) instead is a plausible-looking mistake
    # that gives a relative error of 0.69, not a crash. A has unit diagonal, so
    # there is no division.
    for i in range(C):
        b = Tensor(np.zeros((B, NT, HV, C), dtype=np.float32)) \
            + tb[:, :, :, i].reshape(B, NT, HV, 1) * eyeT[i]
        for r in range(1, C):
            # Forward substitution updates ONE entry per step:
            #   b[r] -= sum_{j<r} A[r,j] * b[j]
            # The subtraction is a scalar (per batch/head), so it must be
            # scattered into row r. Subtracting the whole vector broadcasts
            # (B,NT,HV,r) against (B,NT,HV,C) and fails, or silently
            # misbroadcasts if the shapes happen to line up.
            s = (A[:, :, :, r, :r] * b[:, :, :, :r]).sum(-1)          # [B,NT,HV]
            b = b - eye_all[r].reshape(1, 1, 1, C) * s.unsqueeze(-1)
        # Scatter the solved vector into COLUMN i of M. This needs e_i as a
        # [1, C] ROW (which selects column i under the outer product), not as a
        # [C, 1] column -- the column form selects ROW i and silently produces a
        # matrix whose every column is the same vector, with no error raised.
        M = M + eye_all[i].reshape(1, 1, 1, 1, C) * b.reshape(B, NT, HV, C, 1)

    # ---- 5/6. WY representation ----
    gam = g_cum.exp()
    w = M @ (gam * tk)                                     # [B,NT,HV,C,K]
    u = M @ tv                                             # [B,NT,HV,C,V]

    # ---- 7. inter-chunk scan, unrolled over NT ----
    S = np.zeros((B, HV, K, V), dtype=np.float32)
    if initial_state is not None:
        S = S + np.asarray(initial_state, dtype=np.float32)
    tS = Tensor(S)
    o_out = None
    # g_cum is [B,NT,HV,C,K]; the chunk axis is -2, so indexing it must be
    # `[:, :, :, -1]` and NOT `[:, :, :, -1, :]` -- the latter leaves the K axis
    # and yields [B,NT,HV,1,K], which then silently mis-broadcasts everywhere
    # downstream instead of erroring.
    g_last = g_cum[:, :, :, -1]                           # [B,NT,HV,K]
    # causal band INCLUDING the diagonal: keep the difference, push the
    # out-of-band entries to -1e4 so exp() gives exactly 0. Same trick as Dl.
    m_causal = Tensor(tril_mask(C, False, True)[None, None, None, :, :, None])
    Dq_all = ((g_cum.reshape(B, NT, HV, C, 1, K)
               - g_cum.reshape(B, NT, HV, 1, C, K)) + (m_causal - 1.0) * 1e4).exp()

    parts = []
    for n in range(NT):
        gn = g_cum[:, n]; kn = tk[:, n]; wn = w[:, n]; un = u[:, n]
        # ref.py's qgn is Gamma . q where Gamma = exp(g_cum) INCLUDING this
        # chunk, but its Aqk uses PLAIN q because Dq_all already carries the
        # whole decay. Using qgn in Aqk applies gamma twice. Same asymmetry as
        # in ref.py; see its header note 3.
        qgn = gam[:, n] * tq[:, n]
        v_new = un - wn @ tS                                # [B,HV,C,V]
        # Aqk[c,j] = sum_d Dq_all[c,j,d] * q[c,d] * k[j,d], causal with the
        # diagonal INCLUDED. `wn` and `un` are already [B,HV,C,K] / [B,HV,C,V]
        # because indexing a 5-D [B,NT,HV,C,X] tensor at [:, n] drops NT.
        qn_ = tq[:, n].reshape(B, HV, C, 1, K)
        kn_ = kn.reshape(B, HV, 1, C, K)
        Aqk = (Dq_all[:, n].reshape(B, HV, C, C, K) * qn_ * kn_).sum(-1)
        o_n = qgn @ tS + Aqk @ v_new
        parts.append(o_n)
        # ref.py:  S = Diag(exp(gC)) S + ((exp(gC - gn) * kn)^T) @ v_new
        # gC and gn are [B,HV,K]; gC - gn must broadcast over the CHUNK axis, so
        # it needs an explicit axis between HV and K, not unsqueeze(-2) which
        # lands on the wrong side of a 4-D tensor.
        gC = g_last[:, n]                                   # [B,HV,K]
        tS = tS * gC.exp().unsqueeze(-1) + \
            ((gC.reshape(B, HV, 1, K) - gn.reshape(B, HV, C, K)).exp()
             * kn.reshape(B, HV, C, K)).permute(0, 1, 3, 2) @ v_new

    # Each part is [B,HV,C,V]. ref.py ends with
    #   o.transpose(0, 1, 3, 2, 4).reshape(B, T, HV, V)
    # taking o from [B,NT,HV,C,V] to [B,NT,C,HV,V] first, so that the flat token
    # index runs n*C + c. Permuting to [B,NT,HV,C,V] and reshaping instead
    # interleaves the HEAD axis into the token index -- every token gets another
    # head's output, with no error raised.
    o = Tensor.stack(*parts)                               # [NT,B,HV,C,V]
    o = o.permute(1, 0, 3, 2, 4).reshape(B, T, HV, V)      # [B,NT,C,HV,V] -> [B,T,HV,V]
    return o.numpy(), tS.numpy()
