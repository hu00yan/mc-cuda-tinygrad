"""
KDA (Kimi Delta Attention) reference implementation -- float64 NumPy.

This is the ORACLE for the TinyGrad port.  It is deliberately a flat, unfused,
unoptimised sequence of the chunked steps so that every algebra step can be
checked by eye against SPEC.md.  No bf16, no fused kernels, no tensor cores.

Three independent code paths, all computing the same operator:

  kda_recurrent  Eq. 1 applied one token at a time.  This *is* the definition
                 of KDA: no chunking, no WY, no UT transform.  O(T) python
                 steps, each a [K,V] state update.  The load-bearing oracle.

  kda_dense      The whole sequence as ONE chunk, with the triangular system
                 solved by a dense `numpy.linalg.solve` (LAPACK LU) instead of
                 row-wise forward substitution.  Validates the WY/UT algebra
                 itself, independently of kda_chunk's linear algebra.
                 O(T^3) solve, so test-only.

  kda_chunk      The deliverable: the chunked algorithm of Kimi Linear
                 section 3.1 / Appendix C, chunk size C (default 64).

`kda_chunk == kda_recurrent` is the load-bearing test: it proves the chunking
algebra is right regardless of what any paper says.

--------------------------------------------------------------------------
Sign / notation landmines (see SPEC.md section 7).  Read before porting.
--------------------------------------------------------------------------
1. The forward substitution must be fed the NEGATED strictly-lower block:
       A = -StrictTril(Diag(beta) @ Z)
   The Kimi Linear paper (Appendix C) and the fused Triton kernel
   (fla/ops/kda/chunk_intra.py: `b_Ai00 = -tl.where(m_A, b_Ai00, 0)`) both
   negate.  `fla/ops/kda/naive.py::naive_chunk_kda` does NOT negate and is
   therefore wrong.  test_ref.py asserts that the negated form is correct.

2. The paper defines gamma^{i->j} := prod_{k=i}^{j} alpha_k (INCLUSIVE, 1-based)
   but then uses it in Eq. 3/8 where a HALF-OPEN product is required
   (prod_{k=i+1}^{j} alpha_k = gamma^j / gamma^i).  This implementation uses the
   ratio form gamma^i / gamma^j throughout, which is what the paper's own
   A_{i/j} definition (Eq. 6/9) and every kernel actually compute.  So the
   "K / Gamma" division of Eq. 6/9 is never evaluated as a division; it is
   evaluated as exp(cumsum_i - cumsum_j).

3. `scale` (default 1/sqrt(K)) multiplies q ONLY.  The fused kernel instead
   folds that single factor of `scale` into the precomputed Aqk block and into
   the inter-chunk term (fla/ops/gla/chunk.py:427).  Same result, one factor.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "kda_gate",
    "l2norm",
    "softplus",
    "kda_recurrent",
    "kda_dense",
    "kda_chunk",
    "DEFAULT_CHUNK_SIZE",
]

DEFAULT_CHUNK_SIZE = 64
F = np.float64  # everything in this file is float64.  No exceptions.


# ---------------------------------------------------------------------------
# 0.  Input preparation (Kimi Linear section 4, "Neural Parameterization")
# ---------------------------------------------------------------------------
def l2norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """L2-normalise the last axis.  KDA applies this to q and k
    (`use_qk_l2norm_in_kernel=True`, always on in `KimiDeltaAttention`)."""
    return x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + eps)


def softplus(x: np.ndarray) -> np.ndarray:
    """log(1 + exp(x)), numerically stable."""
    return np.logaddexp(0.0, x)


def kda_gate(g_raw, A_log, dt_bias=None, lower_bound=None):
    """Raw gate logits -> LOG-SPACE per-channel decay `g`, shape [..., H, K].

    This is the function that actually produces KDA's fine-grained gate.
    `A_log` has shape [H] (one learned rate per head); `dt_bias` has shape
    [H*K] (one bias per head per key channel).  Output shape == g_raw shape.

    Two variants, both from fla/ops/kda/gate.py:

    default ("softplus")       g = -exp(A_log) * softplus(g_raw + dt_bias)
                               => g in (-inf, 0), so alpha = exp(g) in (0, 1).
                                  g = 0 (alpha = 1, i.e. NO decay) is only a
                                  limit, not attainable for finite A_log.

    safe_gate ("lower_bound")  g = lower_bound * sigmoid(exp(A_log) * (g_raw + dt_bias))
                               => g in (lower_bound, 0), a HARD lower bound.
                                  lower_bound = -5 recommended, so
                                  alpha >= e^-5 ~= 0.0067.  This is what lets
                                  the kernel use the M=16 TensorCore path
                                  without exp2 underflowing.
    """
    H, K = g_raw.shape[-2:]
    g = np.asarray(g_raw, dtype=F)
    if dt_bias is not None:
        g = g + np.asarray(dt_bias, dtype=F).reshape(H, K)
    a = np.exp(np.asarray(A_log, dtype=F).reshape(H, 1)) if A_log is not None else 1.0
    if lower_bound is None:
        return -a * softplus(g)
    return lower_bound / (1.0 + np.exp(-a * g))


def _prep(q, k, v, g, beta, scale):
    """Shared shape / head bookkeeping.  Everything is float64 after this.

    q, k : [B, T, H,  K]   K = head_dim
    v    : [B, T, HV, V]   HV = num value heads, V = head_v_dim
    g    : [B, T, HV, K]   per-channel log decay, in the VALUE head dim
    beta : [B, T, HV]

    GVA (grouped value attention): when HV > H, q and k are shared by
    G = HV // H adjacent value heads.  The log decay `g` is NOT shared -- it is
    per value head AND per key channel, so its head dim is HV, not H.  This is
    the one place KDA's head layout is not the usual one.

    `scale` multiplies q and nothing else.
    """
    q = np.asarray(q, dtype=F)
    B, T, H, K = q.shape
    HV, V = np.asarray(v).shape[2], np.asarray(v).shape[3]
    assert np.asarray(k).shape == q.shape, (np.asarray(k).shape, q.shape)
    assert np.asarray(g).shape == (B, T, HV, K), (np.asarray(g).shape, (B, T, HV, K))
    assert np.asarray(beta).shape == (B, T, HV), (np.asarray(beta).shape, (B, T, HV))
    assert HV % H == 0, f"HV={HV} must be divisible by H={H}"
    G = HV // H
    if scale is None:
        scale = K ** -0.5

    q = np.repeat(q, G, axis=2) * scale   # [B, T, HV, K]
    k = np.repeat(np.asarray(k, dtype=F), G, axis=2)   # [B, T, HV, K]
    v = np.asarray(v, dtype=F)           # [B, T, HV, V]
    g = np.asarray(g, dtype=F)           # [B, T, HV, K]
    beta = np.asarray(beta, dtype=F)     # [B, T, HV]
    return q, k, v, g, beta


def _chunks(x, B, NT, C, HV):
    """[B, T, HV, X] -> [B, NT, HV, C, X]."""
    return x.reshape(B, NT, C, HV, -1).transpose(0, 1, 3, 2, 4)


def _chunks_scalar(x, B, NT, C, HV):
    """[B, T, HV] -> [B, NT, HV, C]  (no trailing feature axis, e.g. beta)."""
    return x.reshape(B, NT, C, HV).transpose(0, 1, 3, 2)


# ---------------------------------------------------------------------------
# 1.  The definition.  Eq. 1 of Kimi Linear.
# ---------------------------------------------------------------------------
def kda_recurrent(q, k, v, g, beta, scale=None, initial_state=None):
    """Token-at-a-time recurrence, Eq. 1:

        S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
        o_t = S_t^T q_t

    with alpha_t = exp(g_t) in (0, 1) per key channel.  Note the ORDER: the decay
    multiplies S_{t-1} FIRST, and the delta-rule correction is applied to the
    ALREADY-DECAYED state.  This is what makes the rule "gated" in the
    Mamba/GDN sense, and getting it backwards is the single most common
    transcription error.

    S : [B, HV, K, V].  Returns (o [B, T, HV, V], S [B, HV, K, V]).
    """
    q, k, v, g, beta = _prep(q, k, v, g, beta, scale)
    B, T, _, K = q.shape
    HV, V = v.shape[2], v.shape[3]

    S = np.zeros((B, HV, K, V), dtype=F)
    if initial_state is not None:
        S = S + np.asarray(initial_state, dtype=F)
    o = np.zeros((B, T, HV, V), dtype=F)

    for t in range(T):
        S = S * np.exp(g[:, t])[..., None]                 # decay FIRST: [B,HV,K,1]
        kt, vt = k[:, t], v[:, t]                         # [B,HV,K], [B,HV,V]
        # S += beta_t k_t (v_t - k_t^T S)^T, contracting over K
        resid = vt - np.einsum("bhk,bhkv->bhv", kt, S)
        S = S + (beta[:, t][..., None, None] * kt[..., :, None]
                 * resid[..., None, :])
        o[:, t] = np.einsum("bhk,bhkv->bhv", q[:, t], S)

    return o, S


# ---------------------------------------------------------------------------
# 2.  Dense oracle: one chunk of size T, solved by LAPACK.
# ---------------------------------------------------------------------------
def kda_dense(q, k, v, g, beta, scale=None, initial_state=None):
    """Whole sequence as a single chunk; the triangular system is solved with
    `numpy.linalg.solve` rather than forward substitution, so this shares no
    linear-algebra code with kda_chunk.  Cross-validates the WY/UT transform.

    O(T^3) solve.  Test oracle only -- do not use at real T.
    """
    q, k, v, g, beta = _prep(q, k, v, g, beta, scale)
    B, T, _, K = q.shape
    HV, V = v.shape[2], v.shape[3]

    S0 = np.zeros((B, HV, K, V), dtype=F)
    if initial_state is not None:
        S0 = S0 + np.asarray(initial_state, dtype=F)

    gc = np.cumsum(g, axis=1)                    # [B,T,HV,K]  log gamma^r
    gam = np.exp(gc)                             # [B,T,HV,K]  gamma^r
    kg = gam * k                                 # [B,T,HV,K]  Gamma (.) K
    kd = k / gam                                 # [B,T,HV,K]  K / Gamma

    Z = np.einsum("bthd,bshd->bhts", kg, kd)      # [B,HV,T,T]
    L = np.eye(T, dtype=F)[None, None] + np.tril(beta.transpose(0, 2, 1)[..., None] * Z, -1)

    R = np.zeros((B, HV, T, T), dtype=F)          # Diag(beta)
    idx = np.arange(T)
    R[:, :, idx, idx] = beta.transpose(0, 2, 1)

    U = np.linalg.solve(L, R @ v.transpose(0, 2, 1, 3))   # U = M V
    W = np.linalg.solve(L, R @ kg.transpose(0, 2, 1, 3))  # W = M (Gamma (.) K)

    X = U - W @ S0                               # [B,HV,T,V] "pseudo-values"

    qg = gam * q
    A = np.tril(np.einsum("bthd,bshd->bhts", qg, kd))    # causal, diag INCLUDED

    o = (np.einsum("bthk,bhkv->bhtv", qg, S0)
         + np.einsum("bhts,bhsv->bhtv", A, X))
    o = o.transpose(0, 2, 1, 3)                  # [B,T,HV,V]

    gC = gc[:, -1, :, :]                         # [B,HV,K] = log gamma^T
    kgC = np.exp(gC[:, None, :, :] - gc) * k      # [B,T,HV,K] = Gamma^{i->T} (.) K
    # the decay acts on the K axis of S, so it must broadcast as [B,HV,K,1]
    S = S0 * np.exp(gC)[:, :, :, None] + kgC.transpose(0, 2, 3, 1) @ X
    return o, S


# ---------------------------------------------------------------------------
# 3.  The deliverable: the chunked algorithm.
# ---------------------------------------------------------------------------
def _causal_decay(g_cum, lower):
    """Decay tensor for the C x C blocks, built so that no exponent is ever
    positive -- i.e. the causal mask is applied to the LOG DIFFERENCE *before*
    the exp.

        D[c, j, d] = exp( g_cum[c, d] - g_cum[j, d] )   for the kept entries
                   = 0                                  for the rest
    g_cum is non-increasing in c (g <= 0), so for `lower` (j < c) the
    difference is <= 0 and the kept entries land in (0, 1].  For the discarded
    half (j >= c) the difference is >= 0 and can be arbitrarily large, so those
    are forced to 0 by masking, never by exponentiating.

    Returns [B, NT, HV, C, C, K].  This is the O(C^2 K) tensor of NOTES.md
    section 3: it is a transient of the *reference*, and it is why a port must
    not instead write `K / gamma` -- see the `test_division_form_overflows`
    check in test_ref.py, and Kimi Linear section 3.2, which calls out exactly
    this division as the thing DPLR pays for and KDA avoids.
    """
    C = g_cum.shape[-2]
    # Reshape to 3-D so the two new axes land unambiguously.  (Writing this with
    # `...` indexing is a trap: `g[..., None, :, :]` puts the new axis BEFORE the
    # head dim, not before the C dim.)
    shp = g_cum.shape
    g2 = g_cum.reshape(-1, C, shp[-1])
    d = g2[:, :, None, :] - g2[:, None, :, :]               # [N, C, 1, K] - [N, 1, C, K]
    keep = (np.tril(np.ones((C, C), dtype=bool), -1) if lower
            else np.tril(np.ones((C, C), dtype=bool)))[:, :, None]   # [C, C, 1]
    out = np.exp(np.where(keep, d, -np.inf))                 # exp(-inf)=0, no overflow
    return out.reshape(shp[:-2] + (C, C, shp[-1]))


def _forward_substitution(A):
    """Invert (I + StrictTril(D)) in place by row-wise forward substitution.

    `A` arrives NEGATED, i.e. A = -StrictTril(D).  Row i is updated from rows
    0..i-1, which have already been finalised.  This is Gaussian elimination
    specialised to a unit lower-triangular matrix, and it is the O(C) sequential
    spine of the whole operator (Kimi Linear Eqs. 4-6, Appendix B Prop 1+2).

    Returns (I + StrictTril(D))^{-1}; the caller multiplies by Diag(beta).

    ⚠ MUTATES ITS ARGUMENT.  `a` is a reshape *view* of `A`, so the loop
    overwrites the caller's array.  Measured: passing a strictly-lower `L`
    returns `(I - L)^{-1}` to 5.551e-16, so with the caller's `A = -StrictTril(D)`
    this is `(I + StrictTril(D))^{-1}` — which is why `kda_chunk` below passes
    `-Btri`, and it is the sign `kda/SPEC.md` §3.3 and §7.1 pin.

    The in-place behaviour is the load-bearing detail and it is the reason this
    function has been "wrong-signed" twice.  A verification that builds its
    expected value from the same array afterwards compares the inverse against
    the matrix that was already inverted, and gets ~9e+00 of error on the
    correct implementation.  **Copy the input if you need to keep it.**  Ledger
    C-23 records that misreading and the two places it happened.
    """
    C = A.shape[-1]
    a = A.reshape(-1, C, C)          # so the last two axes are always C x C
    for i in range(1, C):
        # row_i (1 x C) @ A[:, :, :i] (C x i) -> (1 x i).  A batched matmul.
        a[:, i, :i] += (a[:, i, :][:, None, :] @ a[:, :, :i])[:, 0, :]
    return (a + np.eye(C, dtype=F)).reshape(A.shape)


def kda_chunk(q, k, v, g, beta, scale=None, initial_state=None,
              chunk_size=DEFAULT_CHUNK_SIZE, return_intermediates=False):
    """Kimi Linear section 3.1, chunk size C (default 64).

    q, k : [B, T, H,  K]    v : [B, T, HV, V]    g : [B, T, HV, K] (log space)
    beta : [B, T, HV]       scale : default 1/sqrt(K), applied to q only.
    initial_state : [B, HV, K, V] or None.
    Returns (o [B, T, HV, V], S [B, HV, K, V]).

    Requires T % chunk_size == 0 (fla pads ragged tails; the reference does not).

    Precondition: g <= 0 elementwise, i.e. alpha = exp(g) in (0, 1].  That is what
    the KDA gate produces and it is what makes the masked decay in
    `_causal_decay` safe: g <= 0 makes g_cum non-increasing, so every exponent
    on a kept (causal) entry is <= 0.  A positive g would be out of spec and
    could overflow those exps.

    Step list, in the order this function runs them.  After `_prep` every
    sequence-major tensor is viewed as [B, NT, HV, C, X] with NT = T // C.
    The paper writes the two C x C blocks as (G (.) K)(K / G)^T; this file
    never forms K / G (see header note 2 and _causal_decay).

      1  g_cum = cumsum(g) over the chunk axis        [B,NT,HV,C,K]  <- the cumsum
      2  Z     = (K (.) D_strict)(K)^T                [B,NT,HV,C,C]  <- dense
                   with D_strict[c,j] = exp(gc[c]-gc[j]) for j < c
      3  Btri  = StrictTril(Diag(beta) Z)             [B,NT,HV,C,C]
      4  M     = fwdsub(-Btri) + I, * Diag(beta)     [B,NT,HV,C,C]  <- SEQUENTIAL
      5  w     = M @ (G (.) K)                        [B,NT,HV,C,K]  <- dense
      6  u     = M @ v                                [B,NT,HV,C,V]  <- dense
      7  ---- inter-chunk scan, SEQUENTIAL over NT, S carried in registers ---
      7a v_new = u - w @ S                            [B,NT,HV,C,V]
      7b Aqk   = Tril((Q (.) D_causal)(K)^T)          [B,NT,HV,C,C]  <- dense
      7c o     = (G (.) Q) @ S + Aqk @ v_new          [B,NT,HV,C,V]
      7d S     = Diag(g_cum[-1]) S + ((G[-1]/G) (.) K)^T @ v_new
    """
    q, k, v, g, beta = _prep(q, k, v, g, beta, scale)
    B, T, _, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    C = chunk_size
    assert T % C == 0, f"T={T} must be divisible by chunk_size={C}"
    NT = T // C

    qc, kc, vc, gc = (_chunks(x, B, NT, C, HV) for x in (q, k, v, g))
    bc = _chunks_scalar(beta, B, NT, C, HV)          # [B,NT,HV,C]

    # --- 1. within-chunk cumulative sums of the log gate --------------------
    g_cum = np.cumsum(gc, axis=-2)                  # [B,NT,HV,C,K]
    gam = np.exp(g_cum)
    kg = gam * kc                                   # Gamma (.) K  (safe: gam <= 1)

    # --- 2/3. decayed k-vs-k gram, strictly lower ---------------------------
    # Z[c,j] = sum_d k[c,d] exp(gc[c,d]-gc[j,d]) k[j,d] for j < c, else 0.
    # Built from masked log-differences, NOT from K/gamma -- see _causal_decay.
    Dl = _causal_decay(g_cum, lower=True)           # [B,NT,HV,C,C,K]
    # kc must broadcast over the SECOND block axis (p), so [.., C, 1, K]
    Z = np.einsum("bnhcpd,bnhpd->bnhcp", Dl * kc[..., None, :], kc)
    del Dl
    Btri = Z * bc[..., :, None]                     # Z is already strictly lower

    # --- 4. UT transform.  Note the NEGATION (header note 1). ---------------
    Minv = _forward_substitution(-Btri)             # (I + StrictTril(.))^{-1}
    Minv = Minv * bc[..., None, :]                   # * Diag(beta)  => paper's M

    # --- 5/6. w and u (the WY representation) -------------------------------
    w = Minv @ kg                                   # [B,NT,HV,C,K]
    u = Minv @ vc                                   # [B,NT,HV,C,V]

    # --- 7. inter-chunk scan.  SEQUENTIAL over NT.  S is the only carried
    #        state: [B,HV,K,V], constant size, does not grow with T. --------
    S = np.zeros((B, HV, K, V), dtype=F)
    if initial_state is not None:
        S = S + np.asarray(initial_state, dtype=F)
    o = np.zeros((B, NT, HV, C, V), dtype=F)
    v_new = np.zeros_like(o)
    Aqk = np.zeros((B, NT, HV, C, C), dtype=F)
    qg = np.zeros((B, NT, HV, C, K), dtype=F)

    g_last = g_cum[..., -1, :]                      # [B,NT,HV,K] = log gamma^C
    # Aqk's decay, causal this time (j <= c, diagonal INCLUDED).  Hoisted out of
    # the scan so it is built once per call, like Dl.
    Dq_all = _causal_decay(g_cum, lower=False)

    for n in range(NT):
        gn, kn, wn, un = g_cum[:, n], kc[:, n], w[:, n], u[:, n]
        qgn = gam[:, n] * qc[:, n]                 # Gamma (.) Q

        v_new[:, n] = un - wn @ S                   # [B,HV,C,V]

        # Aqk[c,j] = sum_d q[c,d] exp(gc[c,d]-gc[j,d]) k[j,d], causal (j <= c).
        # NOTE plain qc, not qgn: Dq_all already carries the whole decay, so
        # using Gamma (.) Q here would apply gamma twice.
        Aqk[:, n] = np.einsum("bhcpd,bhpd->bhcp", Dq_all[:, n] * qc[:, n][..., None, :], kn)
        qg[:, n] = qgn

        o[:, n] = qgn @ S + Aqk[:, n] @ v_new[:, n]

        gC = g_last[:, n]                          # [B,HV,K] = log gamma^C
        S = S * np.exp(gC)[..., None]              # [B,HV,K,1] -> Diag(gamma^C) S
        # Gamma^{i->C} = gamma^C / gamma^i, in the log domain (see header note 2)
        S = S + (np.exp(gC[:, :, None, :] - gn) * kn).transpose(0, 1, 3, 2) @ v_new[:, n]

    o = o.transpose(0, 1, 3, 2, 4).reshape(B, T, HV, V)

    if return_intermediates:
        return o, S, {
            "g_cum": g_cum, "Z": Z, "Btri": Btri, "M": Minv,
            "w": w, "u": u, "Aqk": Aqk, "qg": qg, "v_new": v_new,
        }
    return o, S
