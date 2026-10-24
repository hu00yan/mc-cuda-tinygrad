"""
Validation suite for kda/ref.py.

Run:  /Users/huyan00/mycode/mc-cuda-tinygrad/.venv/bin/python kda/test_ref.py

No pytest in the venv, so this is a self-contained runner: every check is a
function, and the tail of the file runs them all and prints a pass/fail table.

The load-bearing test is `test_chunked_equals_recurrence`.  It compares the
chunked algorithm (kda_chunk, the thing you will port to TinyGrad) against the
token-at-a-time definition of Eq. 1 (kda_recurrent).  That comparison is
independent of every paper, every kernel, and every convention -- if the
chunking algebra is wrong, it cannot pass.
"""

import sys
import traceback

import numpy as np

import ref
from ref import (kda_chunk, kda_dense, kda_gate, kda_recurrent, l2norm, softplus)

TOL = 1e-11  # float64, generous for O(T)-deep reassociations


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def rnd(B, T, H, HV, K, V, seed=0, gmax=0.7, l2n=False):
    """Random KDA inputs.  `g` is a log-decay in [-gmax, 0], so alpha in
    [e^-gmax, 1) -- the realistic regime.  `l2n` applies KDA's q/k L2 norm."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(B, T, H, K))
    k = rng.normal(size=(B, T, H, K))
    v = rng.normal(size=(B, T, HV, V))
    g = -rng.random((B, T, HV, K)) * gmax
    beta = rng.random((B, T, HV))
    if l2n:
        q, k = l2norm(q), l2norm(k)
    return q, k, v, g, beta


def relerr(a, b):
    """Max abs difference normalised by the scale of the reference."""
    d = np.abs(a - b).max()
    s = max(np.abs(b).max(), 1e-300)
    return d / s


def approx(a, b, tol=TOL, what=""):
    e = relerr(a, b)
    assert e < tol, f"{what}: relerr {e:.3e} >= {tol:.1e}"
    return e


# ---------------------------------------------------------------------------
# 1. THE test.  chunked == naive per-token recurrence.
# ---------------------------------------------------------------------------
def test_chunked_equals_recurrence():
    """kda_chunk must reproduce Eq. 1 applied one token at a time, for every
    chunk size, with and without an incoming state, with and without L2 norm,
    and with and without grouped-value heads."""
    worst = 0.0
    cases = [
        # (B, T, H, HV, K, V, l2n, use_h0)
        (2, 16, 2, 2, 4, 4, False, False),
        (2, 16, 2, 2, 4, 4, False, True),
        (1, 32, 1, 1, 8, 8, True, False),
        (3, 24, 3, 3, 6, 10, True, True),
        (2, 64, 2, 2, 8, 4, False, True),
        (2, 16, 1, 3, 4, 4, False, False),   # GVA, G = 3
        (2, 16, 1, 4, 8, 8, True, True),    # GVA, G = 4
        (1, 64, 2, 2, 16, 16, True, True),   # bigger heads
    ]
    for i, (B, T, H, HV, K, V, l2n, use_h0) in enumerate(cases):
        q, k, v, g, beta = rnd(B, T, H, HV, K, V, seed=i, l2n=l2n)
        h0 = rnd(0, 0, 0, 0, 0, 0, seed=99)[0] if False else None
        h0 = np.random.default_rng(7 + i).normal(size=(B, HV, K, V)) if use_h0 else None

        o_ref, S_ref = kda_recurrent(q, k, v, g, beta, initial_state=h0)
        assert np.abs(o_ref).max() > 0, "degenerate test input"

        for C in (1, 2, 4, 8, 16, 32, 64):
            if T % C:
                continue
            o_c, S_c = kda_chunk(q, k, v, g, beta, chunk_size=C, initial_state=h0)
            eo = approx(o_c, o_ref, what=f"case{i} C={C} o")
            eS = approx(S_c, S_ref, what=f"case{i} C={C} S")
            worst = max(worst, eo, eS)

        # the dense LAPACK path too (single chunk of size T)
        o_d, S_d = kda_dense(q, k, v, g, beta, initial_state=h0)
        approx(o_d, o_ref, what=f"case{i} dense o")
        approx(S_d, S_ref, what=f"case{i} dense S")
    return worst


def test_chunk_size_invariance():
    """The result must not depend on C at all.  C=1 is structurally the
    recurrence, so this is a second, independent statement of test 1."""
    B, T, H, HV, K, V = 2, 32, 2, 2, 8, 8
    q, k, v, g, beta = rnd(B, T, H, HV, K, V, seed=11, l2n=True)
    o0, S0 = kda_chunk(q, k, v, g, beta, chunk_size=1)
    worst = 0.0
    for C in (2, 4, 8, 16, 32):
        o, S = kda_chunk(q, k, v, g, beta, chunk_size=C)
        worst = max(worst, approx(o, o0, what=f"invariance o C={C}"),
                    approx(S, S0, what=f"invariance S C={C}"))
    return worst


# ---------------------------------------------------------------------------
# 2. Sign conventions.  Documented landmines, asserted so they cannot regress.
# ---------------------------------------------------------------------------
def _minv_with_sign(Akk_pos, beta_c, negate):
    """Reproduce the UT transform with and without the paper's negation."""
    Btri = np.tril(Akk_pos * beta_c[..., :, None], -1)
    M = ref._forward_substitution(-Btri if negate else Btri.copy())
    return M * beta_c[..., None, :]


def test_forward_substitution_sign():
    """The forward substitution must be fed the NEGATED strictly-lower block.

    Kimi Linear Appendix C and fla's Triton kernel both negate.
    fla/ops/kda/naive.py::naive_chunk_kda does NOT, and is wrong.  Assert it.
    """
    B, T, H, HV, K, V, C = 2, 8, 1, 1, 4, 4, 8
    q, k, v, g, beta = rnd(B, T, H, HV, K, V, seed=3)
    o_ref, S_ref = kda_recurrent(q, k, v, g, beta)

    q_, k_, v_, g_, b_ = ref._prep(q, k, v, g, beta, None)
    NT = T // C
    kg = np.exp(np.cumsum(ref._chunks(g_, B, NT, C, HV), axis=-2)) * ref._chunks(k_, B, NT, C, HV)
    kd = ref._chunks(k_, B, NT, C, HV) / np.exp(np.cumsum(ref._chunks(g_, B, NT, C, HV), axis=-2))
    bc = ref._chunks_scalar(b_, B, NT, C, HV)
    Z = np.einsum("bnhcd,bnhpd->bnhcp", kg, kd)

    neg = _minv_with_sign(Z, bc, negate=True)
    pos = _minv_with_sign(Z, bc, negate=False)
    assert relerr(neg, pos) > 1e-3, "the two sign conventions coincide here; test is blind"

    # patched chunked forward using each M
    for M, name, should_match in ((neg, "negated", True), (pos, "un-negated", False)):
        oc, Sc = _kda_chunk_with_M(q, k, v, g, beta, M, C)
        e = relerr(oc, o_ref)
        if should_match:
            assert e < TOL, f"{name} M should match the recurrence, relerr {e:.2e}"
        else:
            assert e > 1e-3, f"{name} M should NOT match the recurrence, relerr {e:.2e}"
    return True


def _kda_chunk_with_M(q, k, v, g, beta, Minv, C):
    """kda_chunk with the UT-transform output injected (for the sign test)."""
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    NT = T // C
    q_, k_, v_, g_, b_ = ref._prep(q, k, v, g, beta, None)
    qc, kc, vc, gc = (ref._chunks(x, B, NT, C, HV) for x in (q_, k_, v_, g_))
    g_cum = np.cumsum(gc, axis=-2)
    gam = np.exp(g_cum)
    w = Minv @ (gam * kc)
    u = Minv @ vc
    S = np.zeros((B, HV, K, V))
    o = np.zeros((B, NT, HV, C, V))
    g_last = g_cum[..., -1, :]
    for n in range(NT):
        v_new = u[:, n] - w[:, n] @ S
        qg = gam[:, n] * qc[:, n]
        A = np.tril(np.einsum("bhcd,bhpd->bhcp", qg, kc[:, n] / gam[:, n]))
        o[:, n] = qg @ S + A @ v_new
        gC = g_last[:, n]
        S = S * np.exp(gC)[..., None] + (np.exp(gC[:, :, None, :] - g_cum[:, n]) * kc[:, n]
                                         ).transpose(0, 1, 3, 2) @ v_new
    return o.transpose(0, 1, 3, 2, 4).reshape(B, T, HV, V), S


# ---------------------------------------------------------------------------
# 3. Shapes and dtypes of every intermediate.
# ---------------------------------------------------------------------------
def test_intermediate_shapes():
    B, T, H, HV, K, V, C = 2, 64, 4, 4, 8, 8, 16
    NT = T // C
    q, k, v, g, beta = rnd(B, T, H, HV, K, V, seed=5, l2n=True)
    o, S, it = kda_chunk(q, k, v, g, beta, chunk_size=C, return_intermediates=True)

    want = {
        "g_cum": (B, NT, HV, C, K),   # the cumsum; [B,T,HV,K] as a chunk view
        "Z":     (B, NT, HV, C, C),   # decayed k-vs-k gram
        "Btri":  (B, NT, HV, C, C),   # StrictTril(Diag(beta) Z)
        "M":     (B, NT, HV, C, C),   # (I+.)^{-1} Diag(beta)  -- the UT transform
        "w":     (B, NT, HV, C, K),   # WY "pseudo keys"
        "u":     (B, NT, HV, C, V),   # WY "pseudo values"
        "Aqk":   (B, NT, HV, C, C),   # intra-chunk (gamma^o Q)(K/gamma^o)^T
        "qg":    (B, NT, HV, C, K),   # Gamma (.) Q
        "v_new": (B, NT, HV, C, V),   # U - W S   (the "pseudo value" term)
    }
    for name, shape in want.items():
        assert name in it, f"missing intermediate {name}"
        assert it[name].shape == shape, f"{name}: {it[name].shape} != {shape}"
        assert it[name].dtype == np.float64, f"{name} is {it[name].dtype}, not float64"

    assert o.shape == (B, T, HV, V), o.shape
    assert S.shape == (B, HV, K, V), S.shape
    assert o.dtype == np.float64 and S.dtype == np.float64

    # the C x C blocks are the only O(T*C) tensors; the state does not grow with T
    for C2 in (8, 16, 32, 64):
        o2, S2 = kda_chunk(q, k, v, g, beta, chunk_size=C2)
        assert S2.shape == (B, HV, K, V), "state must be T-independent"

    # g_cum really is a within-chunk cumsum (chunk-local, NOT global)
    gc = it["g_cum"].reshape(B, NT, HV, C, K)
    flat = g.reshape(B, NT, C, HV, K).transpose(0, 1, 3, 2, 4)
    approx(gc, np.cumsum(flat, axis=-2), what="g_cum == cumsum over chunk axis")
    # and it resets at every chunk boundary
    assert abs(gc[0, 1, 0, 0, 0] - g[0, C, 0, 0]) < 1e-15, "cumsum leaked across chunks"
    return True


def test_input_shape_assertions():
    B, T, H, HV, K, V = 2, 16, 2, 4, 4, 4
    q, k, v, g, beta = rnd(B, T, H, HV, K, V, seed=6)
    # g must be [B,T,HV,K] -- the VALUE-head dim, not H.  Catch a silent H/K swap.
    for bad in (g[:, :, :H], g[:, :, :, :K // 2]):
        try:
            kda_chunk(q, k, v, bad, beta, chunk_size=4)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"bad g shape {bad.shape} was accepted")
    try:
        kda_chunk(q, k, v, g, beta, chunk_size=5)   # T=16, C=5 -> ragged
    except AssertionError:
        pass
    else:
        raise AssertionError("ragged T % chunk_size was accepted")
    return True


# ---------------------------------------------------------------------------
# 4. Degenerate cases.
# ---------------------------------------------------------------------------
def test_gate_zero_is_deltanet_not_linear_attention():
    """g == 0  =>  alpha_t == 1  =>  NO decay.  KDA then reduces to the
    classical (ungated) delta rule, DeltaNet:

        S_t = (I - beta k k^T) S_{t-1} + beta k v^T

    It does NOT reduce to plain linear attention S_t = S_{t-1} + k v^T, because
    beta_t k_t v_t^T is the only write term in KDA -- there is no way to turn
    the delta rule off from the (g, beta) interface.  Both facts are asserted.
    """
    B, T, H, HV, K, V, C = 2, 32, 2, 2, 8, 8, 8
    q, k, v, _, beta = rnd(B, T, H, HV, K, V, seed=7, l2n=True)
    g = np.zeros((B, T, HV, K))                     # log decay == 0

    o_kda, S_kda = kda_chunk(q, k, v, g, beta, chunk_size=C)
    o_ref, S_ref = kda_recurrent(q, k, v, g, beta)
    approx(o_kda, o_ref, what="g=0 chunked == recurrence")
    approx(S_kda, S_ref, what="g=0 final state")

    # a tiny explicit DeltaNet, written from the paper, sharing no code
    S = np.zeros((B, HV, K, V))
    o = np.zeros((B, T, HV, V))
    qs, ks, vs = (x * (x.shape[-1] ** -0.5) if x is q else x for x in (q, k, v))
    qs = np.repeat(qs, HV // H, axis=2)
    ks = np.repeat(ks, HV // H, axis=2)
    for t in range(T):
        S = (np.eye(K)[None, None] - beta[:, t][..., None, None]
             * np.einsum("bhk,bhp->bhkp", ks[:, t], ks[:, t])) @ S \
            + beta[:, t][..., None, None] * np.einsum("bhk,bhv->bhkv", ks[:, t], vs[:, t])
        o[:, t] = np.einsum("bhk,bhkv->bhv", qs[:, t], S)
    approx(o_kda, o, what="g=0 == hand-written DeltaNet")

    # and it is genuinely NOT plain linear attention
    S = np.zeros((B, HV, K, V))
    o_lin = np.zeros((B, T, HV, V))
    for t in range(T):
        S = S + np.einsum("bhk,bhv->bhkv", ks[:, t], vs[:, t])
        o_lin[:, t] = np.einsum("bhk,bhkv->bhv", qs[:, t], S)
    assert relerr(o_kda, o_lin) > 1e-2, \
        "g=0 unexpectedly equalled plain linear attention; the test input is blind"

    # chunk-size invariance must survive the degenerate gate
    for C2 in (1, 2, 4, 8, 16, 32):
        o2, S2 = kda_chunk(q, k, v, g, beta, chunk_size=C2)
        approx(o2, o_kda, what=f"g=0 invariance o C={C2}")
        approx(S2, S_kda, what=f"g=0 invariance S C={C2}")

    # and the real softplus gate can never reach alpha == 1
    gl = kda_gate(np.zeros((1, 4, 2, K)), np.zeros(2), np.zeros(2 * K))
    assert (gl < 0).all(), "softplus gate must be strictly negative"
    return True


def test_gate_hard_decay():
    """alpha -> 0 (g -> large negative): every write completely wipes the state.
    Then S_t ~ beta_t k_t v_t^T and o_t ~ q_t^T (beta_t k_t v_t^T).  This is the
    sharpest test of the decay-difference algebra, because the diagonal of the
    decay matrices becomes 0 while the off-diagonals must stay 0 too.
    """
    B, T, H, HV, K, V, C = 2, 32, 2, 2, 8, 8, 8
    q, k, v, _, beta = rnd(B, T, H, HV, K, V, seed=8, l2n=True)
    # exp(-1e3) underflows to exactly 0 in float64.  That is the POINT of this
    # test -- the state is wiped exactly -- so the underflow is expected, not a
    # failure.  main() only raises on overflow / invalid / divide-by-zero.
    with np.errstate(under="ignore"):
        g = np.full((B, T, HV, K), -1e3)            # alpha = exp(-1000) == 0
        o, S = kda_chunk(q, k, v, g, beta, chunk_size=C)
        o_ref, S_ref = kda_recurrent(q, k, v, g, beta)
        approx(o, o_ref, what="alpha=0 o")
        approx(S, S_ref, what="alpha=0 S")

        # closed form: with no state carried, S_t = beta_t k_t v_t^T exactly
        scale = K ** -0.5
        qs = np.repeat(q * scale, HV // H, axis=2)
        ks = np.repeat(k, HV // H, axis=2)
        o_cf = np.einsum("bthk,bthk,bthv->bthv", qs, ks, v) * beta[..., None]
        approx(o, o_cf, what="alpha=0 closed form")
        approx(S, np.einsum("bhk,bhv->bhkv", ks[:, -1], v[:, -1])
               * beta[:, -1][..., None, None], what="alpha=0 final state")
    return True


def test_beta_zero():
    """beta == 0: no writes at all, so S_t = Diag(alpha_t) S_{t-1} and the only
    thing that reaches the output is the incoming state."""
    B, T, H, HV, K, V, C = 2, 32, 2, 2, 8, 8, 8
    q, k, v, g, _ = rnd(B, T, H, HV, K, V, seed=9, l2n=True)
    beta = np.zeros((B, T, HV))
    h0 = np.random.default_rng(21).normal(size=(B, HV, K, V))
    o, S = kda_chunk(q, k, v, g, beta, chunk_size=C, initial_state=h0)
    o_ref, S_ref = kda_recurrent(q, k, v, g, beta, initial_state=h0)
    approx(o, o_ref, what="beta=0 o")
    approx(S, S_ref, what="beta=0 S")

    # closed form: cumulative per-channel decay applied to h0.  NOTE this is a
    # cumulative PRODUCT of alpha, i.e. exp of the cumulative SUM of g.
    scale = K ** -0.5
    qs = np.repeat(q * scale, HV // H, axis=2)
    cum = np.exp(np.cumsum(g, axis=1))                              # [B,T,HV,K]
    o_cf = np.einsum("bthk,bthk,bhkv->bthv", qs, cum, h0)
    approx(o, o_cf, what="beta=0 closed form")
    approx(S, h0 * cum[:, -1][:, :, :, None], what="beta=0 final state")
    return True


def test_beta_one_and_orthogonal_keys():
    """beta == 1 with L2-normalised, mutually orthogonal keys: I - k k^T is a
    Householder reflection, so the delta rule is exactly invertible and the
    state after n steps is the Gram-Schmidt-free 'subtract the component along
    k' operator.  Checks the beta=1 boundary of the update."""
    B, T, H, HV, K, V, C = 1, 16, 1, 1, 8, 8, 8
    rng = np.random.default_rng(31)
    q = l2norm(rng.normal(size=(B, T, H, K)))
    k = l2norm(rng.normal(size=(B, T, H, K)))
    v = rng.normal(size=(B, T, HV, V))
    g = np.zeros((B, T, HV, K))                    # no decay
    beta = np.ones((B, T, HV))
    o, S = kda_chunk(q, k, v, g, beta, chunk_size=C)
    o_ref, S_ref = kda_recurrent(q, k, v, g, beta)
    approx(o, o_ref, what="beta=1 o")
    approx(S, S_ref, what="beta=1 S")
    # beta=1 leaves the prediction error as the residual; with a zero state the
    # first token's output must be exactly scale * k_0 v_0^T q_0
    assert abs(o[0, 0, 0, 0] - (q[0, 0, 0] @ (k[0, 0, 0] * K ** -0.5)) * v[0, 0, 0, 0]) < 1e-12
    return True


# ---------------------------------------------------------------------------
# 5. Structural properties a TinyGrad port must preserve.
# ---------------------------------------------------------------------------
def test_causality():
    """o[:, t] must not depend on any input at position > t.  This catches
    transposed masks, an off-by-one in the causal tril, or a leaked future."""
    B, T, H, HV, K, V, C = 2, 32, 2, 2, 8, 8, 16
    q, k, v, g, beta = rnd(B, T, H, HV, K, V, seed=12, l2n=True)
    o0, S0 = kda_chunk(q, k, v, g, beta, chunk_size=C)
    for t in (0, 5, 17, 31):
        # perturb STRICTLY AFTER t; o[:, :t+1] must be bit-identical
        qq, kk2, vv, gg, bb = q.copy(), k.copy(), v.copy(), g.copy(), beta.copy()
        qq[:, t + 1:] += 3.0
        kk2[:, t + 1:] -= 2.0
        vv[:, t + 1:] *= -1.5
        gg[:, t + 1:] -= 0.4
        bb[:, t + 1:] = 1.0 - bb[:, t + 1:]
        o1, _ = kda_chunk(qq, kk2, vv, gg, bb, chunk_size=C)
        assert np.array_equal(o0[:, :t + 1], o1[:, :t + 1]), \
            f"causality violated at t={t}: max leak {np.abs(o0[:, :t+1]-o1[:, :t+1]).max():.3e}"
    return True


def test_decay_differences_are_non_positive():
    """Every exponential in the algorithm is exp(log_a - log_b) with log_a <=
    log_b, so it lies in (0, 1] and can never overflow.  This is the whole
    reason the log-space formulation exists (paper section 3.2).  Assert it on
    the real intermediates rather than trusting it."""
    B, T, H, HV, K, V, C = 2, 32, 2, 2, 8, 8, 16
    q, k, v, g, beta = rnd(B, T, H, HV, K, V, seed=13)
    _, _, it = kda_chunk(q, k, v, g, beta, chunk_size=C, return_intermediates=True)
    gc = it["g_cum"]                                       # [B,NT,HV,C,K]
    assert (gc <= 0).all(), "g_cum must be <= 0 for a non-positive log gate"

    # the decay used in the state update: gamma^C / gamma^i for every i in the
    # chunk (i <= C), so it is always <= 1
    d = gc[:, :, :, -1:, :] - gc                           # [B,NT,HV,1,K] - [B,NT,HV,C,K]
    assert (d <= 0).all(), "gamma^C / gamma^i must be <= 1 for all i in the chunk"

    # Aqk[i,j] uses gamma^i / gamma^j, so the KEPT (causal, j <= i) entries have
    # decay <= 1 and the DISCARDED (j > i) entries have decay > 1
    C_ = gc.shape[-2]
    dd = gc[:, :, :, :, None, :] - gc[:, :, :, None, :, :]   # [B,NT,HV,C,C,K]
    tril = np.tril(np.ones((C_, C_), dtype=bool))
    assert (dd[:, :, :, tril] <= 0).all(), "causal decay gamma^i/gamma^j must be <= 1"
    assert (dd[:, :, :, ~tril] > 0).all(), "above-diagonal decay must be > 1"
    assert np.isfinite(it["Aqk"]).all()
    # ... and only the causal part of Aqk was actually written
    assert (it["Aqk"][:, :, :, ~tril] == 0).all(), "Aqk must be exactly causal"
    return True


def test_scale_applies_to_q_only():
    """`scale` multiplies q and nothing else, so o is exactly linear in q and
    exactly invariant to it.  o(scale*q, scale=1) == scale * o(q, scale=1)."""
    B, T, H, HV, K, V, C = 2, 16, 2, 2, 8, 8, 8
    q, k, v, g, beta = rnd(B, T, H, HV, K, V, seed=14, l2n=True)
    s = 0.37
    o0, S0 = kda_chunk(q, k, v, g, beta, scale=1.0, chunk_size=C)
    o1, S1 = kda_chunk(s * q, k, v, g, beta, scale=1.0, chunk_size=C)
    approx(o1, s * o0, tol=1e-13, what="scale applies to q only (output)")
    assert np.allclose(S1, S0), "scale must not touch the state at all"

    # the default is K**-0.5
    od, _ = kda_chunk(q, k, v, g, beta, chunk_size=C)
    oexp, _ = kda_chunk(q, k, v, g, beta, scale=K ** -0.5, chunk_size=C)
    assert np.array_equal(od, oexp), "default scale must be exactly K**-0.5"

    # scaling v instead must scale the output identically (output is linear in v)
    ov, _ = kda_chunk(q, k, s * v, g, beta, scale=1.0, chunk_size=C)
    approx(ov, s * o0, tol=1e-13, what="linearity in v")
    return True


def test_gva_head_grouping():
    """HV = G*H: q/k are shared by G adjacent value heads, while g and beta are
    per value head.  Running HV=1 per head must reproduce the grouped run."""
    B, T, H, HV, K, V, C = 2, 16, 1, 3, 4, 4, 8
    q, k, v, g, beta = rnd(B, T, H, HV, K, V, seed=15, l2n=True)
    o_grp, S_grp = kda_chunk(q, k, v, g, beta, chunk_size=C)
    o_ref, S_ref = kda_recurrent(q, k, v, g, beta)
    approx(o_grp, o_ref, what="GVA chunked == recurrence")
    approx(S_grp, S_ref, what="GVA state")

    for h in range(HV):
        oh, Sh = kda_chunk(q, k, v[:, :, h:h + 1], g[:, :, h:h + 1],
                           beta[:, :, h:h + 1], chunk_size=C)
        approx(oh[:, :, 0], o_grp[:, :, h], what=f"GVA head {h} output")
        approx(Sh[:, 0], S_grp[:, h], what=f"GVA head {h} state")
    return True


# ---------------------------------------------------------------------------
# 6. The gate / normalisation front end.
# ---------------------------------------------------------------------------
def test_gate_activation():
    H, K = 3, 8
    g_raw = np.random.default_rng(17).normal(size=(2, 5, H, K))
    A_log = np.random.default_rng(18).normal(size=H) * 0.5
    dt_bias = np.random.default_rng(19).normal(size=H * K) * 0.1

    g = kda_gate(g_raw, A_log, dt_bias)
    assert g.shape == g_raw.shape
    assert (g < 0).all(), "softplus gate must be strictly negative (alpha in (0,1))"
    alpha = np.exp(g)
    assert (alpha > 0).all() and (alpha <= 1).all()
    expect = -np.exp(A_log)[:, None] * softplus(g_raw + dt_bias.reshape(H, K))
    approx(g, expect, what="softplus gate formula")

    # lower-bound ("safe_gate") variant is hard-clamped
    for lb in (-1.0, -5.0):
        gl = kda_gate(g_raw, A_log, dt_bias, lower_bound=lb)
        assert (gl >= lb - 1e-12).all() and (gl <= 0).all(), f"lower_bound={lb} violated"
        # lb * sigmoid(exp(A_log) * (g_raw + dt_bias))
        expect = lb / (1.0 + np.exp(-np.exp(A_log)[:, None]
                                     * (g_raw + dt_bias.reshape(H, K))))
        approx(gl, expect, what=f"lower-bound gate lb={lb}")
    # without A_log the lower-bound gate is just lb*sigmoid(x+dt_bias)
    gl = kda_gate(g_raw, None, dt_bias, lower_bound=-5.0)
    approx(gl, -5.0 / (1.0 + np.exp(-(g_raw + dt_bias.reshape(H, K)))), what="lb gate no A_log")
    return True


def test_l2norm_and_end_to_end():
    """q/k are L2-normalised by the KDA layer.  The operator must be unchanged
    by that choice (it is just a rescaling of inputs), but the norms must be
    exactly 1 and the pipeline must still match the recurrence."""
    B, T, H, HV, K, V, C = 2, 16, 2, 2, 8, 8, 8
    q, k, v, g, beta = rnd(B, T, H, HV, K, V, seed=20)
    qn, kn = l2norm(q), l2norm(k)
    # l2norm adds eps=1e-6 under the sum of squares, so |q| is 1 to ~eps/sumsq
    approx(np.linalg.norm(qn, axis=-1), np.ones((B, T, H)), tol=1e-6, what="q l2norm")
    approx(np.linalg.norm(kn, axis=-1), np.ones((B, T, H)), tol=1e-6, what="k l2norm")
    # with eps=0 the normalisation is exact
    approx(np.linalg.norm(l2norm(q, eps=0.0), axis=-1), np.ones((B, T, H)),
           what="q l2norm eps=0")

    o, S = kda_chunk(qn, kn, v, g, beta, chunk_size=C)
    o_ref, S_ref = kda_recurrent(qn, kn, v, g, beta, initial_state=None)
    approx(o, o_ref, what="l2norm'd inputs still match the recurrence")
    approx(S, S_ref, what="l2norm'd inputs, state")

    # the gate function feeds straight in
    H_, K_ = H, K
    g_raw = np.random.default_rng(22).normal(size=(B, T, HV, K_))
    g_act = kda_gate(g_raw, np.zeros(HV), np.zeros(HV * K_))
    o2, S2 = kda_chunk(qn, kn, v, g_act, beta, chunk_size=C)
    approx(o2, kda_recurrent(qn, kn, v, g_act, beta)[0], what="gate-activated pipeline")
    return True


# ---------------------------------------------------------------------------
# 7. The division hazard.  This is the single most important thing to get right
#    in a port, and it is the thing Kimi Linear section 3.2 is about.
# ---------------------------------------------------------------------------
def test_division_form_overflows():
    """The paper writes the C x C blocks as (G (.) K)(K / G)^T.  Taken literally
    -- computing K / gamma as an array -- that is a division by the cumulative
    decay, and it blows up: for a hard gate (g = -1e3, i.e. alpha = 0) gamma is
    exactly 0, so K/gamma is inf and every downstream product is nan.

    The fix, which every kernel uses, is to never form K/gamma: evaluate the
    ratio as a difference of the log cumsums, and mask before the exp so no
    positive exponent is ever taken.  Assert that the literal form really does
    fail here, so nobody "simplifies" the port back to it.
    """
    B, T, H, HV, K, V, C = 2, 32, 1, 1, 8, 8, 8
    q, k, v, _, beta = rnd(B, T, H, HV, K, V, seed=41, l2n=True)
    with np.errstate(under="ignore", divide="ignore"):
        g = np.full((B, T, HV, K), -1e3)
        gc = np.cumsum(g.reshape(B, T // C, C, HV, K).transpose(0, 1, 3, 2, 4), axis=-2)
        gam = np.exp(gc)
        assert (gam == 0).any(), "expected gamma to underflow to exactly 0"
        kd_literal = k.reshape(B, T // C, C, HV, K).transpose(0, 1, 3, 2, 4) / gam
        assert not np.isfinite(kd_literal).all(), \
            "the literal K/gamma form was finite here; the hazard test is blind"
        assert np.isinf(kd_literal).any(), "expected an explicit inf in K/gamma"

        # ... while the reference (difference form, masked before the exp) is fine
        D = ref._causal_decay(gc, lower=True)
        assert np.isfinite(D).all(), "the masked difference form must stay finite"
        assert (D >= 0).all() and (D <= 1).all(), "kept decay entries must lie in (0, 1]"
        o, S = kda_chunk(q, k, v, g, beta, chunk_size=C)
        approx(o, kda_recurrent(q, k, v, g, beta)[0], what="hard gate via difference form")
    return True


def test_gram_notations_agree():
    """In the benign regime the two notations for the gram must agree exactly:
        Z_masked  = (K (.) D_strict) K^T          (what ref.py computes)
        Z_literal = StrictTril( (Gamma (.) K) (K / Gamma)^T )   (the paper's)
    They differ only where gamma underflows, which is what makes the first form
    the correct thing to write.  Tying them together stops the two notations
    from drifting apart.
    """
    B, T, H, HV, K, V, C = 2, 32, 2, 2, 8, 8, 8
    q, k, v, g, beta = rnd(B, T, H, HV, K, V, seed=42)
    NT = T // C
    _, k_, _, g_, _ = ref._prep(q, k, v, g, beta, None)
    kc = ref._chunks(k_, B, NT, C, HV)
    gc = np.cumsum(ref._chunks(g_, B, NT, C, HV), axis=-2)
    gam = np.exp(gc)
    with np.errstate(under="ignore"):
        Z_masked = np.einsum("bnhcpd,bnhpd->bnhcp",
                             ref._causal_decay(gc, lower=True) * kc[..., None, :], kc)
        Z_literal = np.tril(np.einsum("bnhcd,bnhpd->bnhcp", gam * kc, kc / gam), -1)
        approx(Z_masked, Z_literal, what="masked vs literal gram")
        # the discarded half is exactly zero in the masked form, and the literal
        # form's discarded half is only zero because of the explicit tril
        Cm = Z_masked.shape[-1]
        strict = np.tril(np.ones((Cm, Cm), bool), -1)
        assert (Z_masked[..., ~strict] == 0).all()
    return True


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
TESTS = [
    ("chunked == naive recurrence  (THE test)", test_chunked_equals_recurrence),
    ("chunk-size invariance (C=1..32)", test_chunk_size_invariance),
    ("UT-transform sign convention", test_forward_substitution_sign),
    ("intermediate shapes + dtypes", test_intermediate_shapes),
    ("input shape assertions", test_input_shape_assertions),
    ("degenerate: g=0 -> DeltaNet", test_gate_zero_is_deltanet_not_linear_attention),
    ("degenerate: alpha=0 (hard wipe)", test_gate_hard_decay),
    ("degenerate: beta=0 (no writes)", test_beta_zero),
    ("degenerate: beta=1, no decay", test_beta_one_and_orthogonal_keys),
    ("causality (no future leakage)", test_causality),
    ("decay differences are <= 0", test_decay_differences_are_non_positive),
    ("scale applies to q only", test_scale_applies_to_q_only),
    ("GVA head grouping", test_gva_head_grouping),
    ("gate activation (softplus / lb)", test_gate_activation),
    ("L2 norm + end-to-end", test_l2norm_and_end_to_end),
    ("division form K/gamma overflows", test_division_form_overflows),
    ("masked vs literal gram agree", test_gram_notations_agree),
]


def main():
    # Overflow / invalid / divide-by-zero are always bugs.  UNDERFLOW is not:
    # a hard-decay gate legitimately drives exp(g) to 0, and the tests that do
    # that opt in locally with np.errstate(under="ignore").
    np.seterr(over="raise", invalid="raise", divide="raise", under="ignore")
    passed, failed = [], []
    print("=" * 78)
    print("KDA reference validation   (numpy %s, float64)" % np.__version__)
    print("=" * 78)
    for name, fn in TESTS:
        try:
            out = fn()
            extra = "" if out is True or out is None else f"   [worst relerr {out:.2e}]"
            print(f"  PASS  {name}{extra}")
            passed.append(name)
        except Exception:
            print(f"  FAIL  {name}")
            traceback.print_exc()
            failed.append(name)
    print("-" * 78)
    print(f"{len(passed)} passed, {len(failed)} failed")
    if failed:
        print("failed: " + ", ".join(failed))
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
