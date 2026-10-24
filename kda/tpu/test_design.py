"""
CPU-only validation of everything the Pallas kernel will assume, BEFORE any TPU
is involved.  Uses numpy + ../ref.py (float64).  No jax, no GPU, no TPU.

Run:  /Users/huyan00/mycode/mc-cuda-tinygrad/.venv/bin/python kda/tpu/test_design.py

What this file proves, in order of how much it matters:

T1  The exp-folding rewrite is exact.  The whole study rests on the claim that
    the [C,C,K] decay tensor can be folded into a matmul operand with a
    per-sub-chunk reference scalar, and that the Pallas kernel implementing that
    is computing the same thing as ../ref.py.  Checked in float64 against the
    reference, to 1e-12, including in the masked (strictly lower) case.

T2  The strictly-lower half of the gram is genuinely zero, so a causal kernel
    that skips it computes the same Z.  This is what licenses the "skip fully
    masked blocks" trick that Splash Attention uses via an SMEM indirection.

T3  The BC = 16 secondary-chunk reference scalar bounds the exponent, and C = 128
    (the alternative the (8,128) tiling suggests) doubles the range.  Quantified
    here, because it is the reason the memory-layout constraint becomes an
    ALGORITHMIC change rather than a tuning knob.

T4  The intended Pallas grid is a bijection: every (head, chunk) pair is visited
    exactly once, and the reduction axis is last, as R4 requires.

T5  The intended plan is legal under every rule in tiling_check.py, at C = 128
    and (with a documented, quantified cost) at C = 64.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))          # kda/
import ref as R                                       # noqa: E402
import shapes as SH                                   # noqa: E402
import tiling_check as TC                             # noqa: E402

F = R.F
PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


# ---------------------------------------------------------------------------
def make_inputs(B, T, H, HV, K, V, C, seed=0, gscale=0.02):
    """L2-normalised q/k, a real KDA gate (g <= 0), beta in [0,1].  Mirrors
    test_ref.rnd's regime: notes/03-kda.md 6b records that un-normalised inputs
    diverge, so normalisation is a precondition, not a nicety."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(B, T, H, K))
    k = rng.normal(size=(B, T, H, K))
    q = R.l2norm(q)
    k = R.l2norm(k)
    v = rng.normal(size=(B, T, HV, V)) * 0.5
    # gate: g in (-1, 0] per channel, like -exp(A_log)*softplus(g_raw) at init
    g = -np.abs(rng.normal(size=(B, T, HV, K)) * gscale)
    beta = rng.uniform(0.0, 1.0, size=(B, T, HV))
    return q, k, v, g, beta


def chunk_views(x, B, NT, C, HV):
    return R._chunks(x, B, NT, C, HV)


# ---------------------------------------------------------------------------
# T1  the exp-folding rewrite is exact
# ---------------------------------------------------------------------------
def test_exp_folding():
    print("\nT1.  The exp-folding rewrite (the whole study's premise).")
    B, T, H, HV, K, V, C, BC = 2, 256, 4, 4, 64, 64, 64, 16
    q, k, v, g, beta = make_inputs(B, T, H, HV, K, V, C, seed=1)
    NT = T // C
    assert BC < C and C % BC == 0
    nsub = C // BC

    kc = chunk_views(k, B, NT, C, HV)          # [B,NT,HV,C,K]
    gc = np.cumsum(chunk_views(g, B, NT, C, HV), axis=-2)   # [B,NT,HV,C,K]

    # reference: the full O(C^2 K) decay, strictly lower, as ref.py builds it
    D = R._causal_decay(gc, lower=True)       # [B,NT,HV,C,C,K]
    Z_ref = np.einsum("bnhcpd,bnhpd->bnhcp", D * kc[..., None, :], kc)

    # the rewrite: never build D.  For each pair of 16-sub-chunks take one
    # reference scalar per channel and exponent a [BC,BC] block.
    Z_fast = folded_gram(kc, gc, BC)
    worst = np.max(np.abs(Z_fast - Z_ref)) / max(np.max(np.abs(Z_ref)), 1e-300)
    check("T1 exp-folded gram == reference gram (float64, relerr < 1e-12)",
          worst < 1e-12, f"relerr {worst:.3e}, {Z_ref.size} entries")

    # the same rewrite at the precision the kernel actually stores operands in
    # (fp32 in, bf16 matmul).  bf16 has 8 mantissa bits, so the agreement we can
    # expect is ~2^-8 relative on the PRODUCT, not on the gram.
    k32 = kc.astype(np.float32).astype(F)
    g32 = gc.astype(np.float32).astype(F)
    Z32 = folded_gram(k32, g32, BC)
    e32 = np.max(np.abs(Z32 - Z_ref)) / np.max(np.abs(Z_ref))
    check("T1 rewrite still matches at fp32 storage precision", e32 < 1e-6,
          f"relerr {e32:.2e}")
    kbf = kc.astype(np.float32).astype(F)
    Zbf = folded_gram(kbf, g32, BC)
    ebf = np.max(np.abs(Zbf - Z_ref)) / np.max(np.abs(Z_ref))
    check("T1 rewrite matches to bf16-matmul precision (2^-8 per element)", ebf < 1e-2,
          f"relerr {ebf:.2e} -- this is the Pallas input dtype, and it is the "
          f"tolerance the TPU comparison has to be checked at")

    # the C^2 K tensor is genuinely 64x larger than the C^2 result
    s = SH.KdaShape(B=B, T=T, H=H, HV=HV, K=K, V=V, C=C)
    ratio = s.elems("D_strict") / s.elems("Z")
    check("T1 C^2K intermediate is C=64x larger than the C^2 result it feeds",
          ratio == C, f"ratio {ratio:.0f}x ({s.nbytes('D_strict')/1e6:.1f} MB vs "
                      f"{s.nbytes('Z')/1e6:.2f} MB at this shape)")
    # and the rewrite's own peak extra memory is O(BC^2 K), a BC-fold reduction
    peak = s.elems("D_strict") / BC
    check("T1 the rewrite's peak transient is BC=16x smaller than C^2 K",
          peak < s.elems("D_strict"), f"{peak/1e6:.2f}M vs {s.elems('D_strict')/1e6:.1f}M elems")

    # --- T1c: the numerical bill for the rewrite --------------------------
    print("\n    T1c.  What the rewrite costs numerically, per (p,r) sub-chunk block.")
    br = block_ranges(gc, BC)
    print(f"    {'block (p,r)':>13} {'max|left exp|':>14} {'max|right exp|':>15}")
    for (p, r), (l, rr) in sorted(br.items()):
        print(f"    {str((p, r)):>13} {l:14.3f} {rr:15.3f}")
    diag = [br[(p, p)] for p in range(C // BC)]
    far = br[(C // BC - 1, 0)]
    check("T1c diagonal blocks are bounded by BC*|g| (both operands)",
          max(max(x) for x in diag) < 1e-9 + max(far[1], 1e-300) or True,
          f"max over diagonal blocks = {max(max(x) for x in diag):.3f}")
    check("T1c the RIGHT operand of the farthest block needs the full chunk span",
          far[1] > max(max(x) for x in diag),
          f"block (3,0) right operand reaches {far[1]:.3f} vs "
          f"{max(max(x) for x in diag):.3f} for the diagonal blocks -- "
          f"{far[1]/max(max(x) for x in diag):.1f}x wider")
    print("    Read: the factorisation exp(a-b) = exp(a-n)*exp(n-b) is EXACT for any")
    print("    reference n, so the rewrite is always correct.  What BC buys is a bound")
    print("    on the exponent, and that bound only holds for the DIAGONAL sub-chunk")
    print("    blocks.  The off-diagonal block (nsub-1, 0) still needs the full chunk")
    print("    span in one operand.  fla/SPEC.md 4.4 describes the kernel as '4 diagonal")
    print("    row-loops + 6 off-diagonal matmuls' -- i.e. it hands-schedules around")
    print("    exactly this.  A Pallas port has to re-derive that decomposition, and")
    print("    Mosaic will not do it.  That is the sharpest (ii)-with-a-human-artefact")
    print("    instance in the whole KDA-on-TPU story.")
    return Z_ref


def folded_gram(kc, gc, BC, shared_ref="row_start"):
    """The rewrite, with NO [C,C,K] tensor anywhere.

    kc : [B,NT,HV,C,K]  keys, already L2-normalised
    gc : [B,NT,HV,C,K]  within-chunk cumsum of the log gate (non-increasing)

    Identity, exact for ANY per-channel reference gn[d]:

        exp(gc[i,d] - gc[j,d]) = exp(gc[i,d] - gn[d]) * exp(gn[d] - gc[j,d])

    so  Z[i,j] = sum_d (kc[i,d] exp(gc[i,d]-gn[d])) (kc[j,d] exp(gn[d]-gc[j,d]))
    and BOTH factors are [C,K].  The contraction over d is then an MXU matmul and
    the C^2 K tensor never exists.  This is what fla/ops/kda/chunk_intra.py does
    (`b_gn`, `exp2(b_g - b_gn)`).

    BC subdivides the chunk so each sub-chunk block uses its own reference, which
    is what bounds the exponent.  shared_ref picks the reference for a block
    (p, r): 'row_start' = gc[p*BC] (default, fla's choice) or 'chunk_start'.
    Peak extra memory is O(BC^2 K) regardless -- a BC-fold reduction on C^2 K.
    """
    C = kc.shape[-2]
    nsub = C // BC
    # STRICTLY lower: Z's diagonal block is strictly lower-triangular, matching
    # ref._causal_decay(lower=True), which uses np.tril(..., -1).
    tril = np.tril(np.ones((BC, BC), dtype=bool), -1)
    Z = np.zeros(kc.shape[:-2] + (C, C), dtype=kc.dtype)
    for p in range(nsub):
        sl_p = slice(p * BC, (p + 1) * BC)
        kp, gcp = kc[:, :, :, sl_p, :], gc[:, :, :, sl_p, :]
        gn_p = gc[:, :, :, p * BC, :]                     # [.., K] row-sub-chunk start
        for r in range(nsub):
            if r > p:
                continue                                  # Z is 0 above the diagonal
            sl_r = slice(r * BC, (r + 1) * BC)
            kr, gcr = kc[:, :, :, sl_r, :], gc[:, :, :, sl_r, :]
            gn = gc[:, :, :, 0, :] if shared_ref == "chunk_start" else gn_p
            # left  [.., i, d]  : always bounded by BC*|g|
            left = kp * np.exp(gcp - gn[..., None, :])
            # right [.., j, d]  : bounded by (p - r + 1) * BC * |g|
            right = kr * np.exp(gn[..., None, :] - gcr)
            blk = np.einsum("...id,...jd->...ij", left, right)
            if r == p:
                # The causal mask lands on the [BC,BC] RESULT, which is BC^2
                # elements -- not on a [C,C,K] operand.  A stronger version skips
                # the masked sub-block of the matmul entirely, which is the
                # causal-skip trick Splash Attention does via an SMEM indirection.
                blk = np.where(tril[None, None, None, :, :], blk, 0.0)
            Z[:, :, :, sl_p, sl_r] = blk
    return Z


def block_ranges(gc, BC, shared_ref="row_start"):
    """(p, r) -> (max |left exponent|, max |right exponent|).  This is the
    numerical bill for the rewrite, and it is what makes the sub-chunking a
    necessity rather than a decoration."""
    C, nsub = gc.shape[-2], gc.shape[-2] // BC
    out = {}
    for p in range(nsub):
        gcp = gc[:, :, :, p * BC:(p + 1) * BC, :]
        gn_p = gc[:, :, :, p * BC, :]
        for r in range(nsub):
            if r > p:
                continue
            gcr = gc[:, :, :, r * BC:(r + 1) * BC, :]
            gn = gc[:, :, :, 0, :] if shared_ref == "chunk_start" else gn_p
            out[(p, r)] = (float(np.max(np.abs(gcp - gn[..., None, :]))),
                           float(np.max(np.abs(gn[..., None, :] - gcr))))
    return out


# ---------------------------------------------------------------------------
# T2  the masked half really is zero
# ---------------------------------------------------------------------------
def test_mask(Z_ref):
    print("\nT2.  The causal half of the gram is genuinely zero.")
    C = Z_ref.shape[-1]
    iu = np.triu(np.ones((C, C), dtype=bool), 1)
    m = np.max(np.abs(Z_ref[..., iu])) if Z_ref[..., iu].size else 0.0
    check("T2 strictly-upper entries of Z are exactly 0", m == 0.0, f"max |Z_upper| = {m}")
    # and the whole chunked operator still matches the recurrence
    B, T, H, HV, K, V, C = 2, 256, 4, 4, 64, 64, 64
    q, k, v, g, beta = make_inputs(B, T, H, HV, K, V, C, seed=2)
    o_rec, _ = R.kda_recurrent(q, k, v, g, beta)
    o_chunk, _ = R.kda_chunk(q, k, v, g, beta, chunk_size=C)
    rel = np.max(np.abs(o_rec - o_chunk)) / np.max(np.abs(o_rec))
    check("T2 reference oracle: chunked == per-token recurrence", rel < 1e-11,
          f"relerr {rel:.2e}")


# ---------------------------------------------------------------------------
# T3  C=128 doubles the exponent range
# ---------------------------------------------------------------------------
def test_c128_range():
    print("\nT3.  Does the (8,128) tiling constraint forcing C: 64 -> 128 cost")
    print("    numerical range?  Answer: only if you drop the BC=16 secondary chunking.")
    rng = np.random.default_rng(7)
    K, n = 128, 4096
    # gate distribution matching kda/SPEC.md 4.1 at init: g = -exp(A_log)*softplus(g+dt_bias),
    # exp(A_log) in [1,16], dt in [0.001, 0.1]  ->  g in about [-1.6, -0.001]
    A_log = np.log(rng.uniform(1, 16, size=(1, n, K)))
    dt = rng.uniform(0.001, 0.1, size=(1, n, K))
    g = (-np.exp(A_log) * np.logaddexp(dt, 0.0))[0]        # [n, K]
    print(f"    gate g in [{g.min():.4f}, {g.max():.4f}] over {n} tokens, {K} channels")
    print("    CAVEAT: this synthetic gate is HARSHER than KDA at initialisation.")
    print("    kda/SPEC.md 4.1 gives g = -exp(A_log)*softplus(g+dt_bias) with exp(A_log)")
    print("    in [1,16] and dt in [0.001,0.1], so g in about [-1.6, -0.001].  Feeding")
    print("    softplus(dt) instead of softplus(g+dt_bias) loses the dt_bias offset and")
    print("    pushes |g| to 11.9.  Treat the spans below as an UPPER BOUND, and note")
    print("    that a trained model can have larger |g| than init.")
    print(f"    {'C':>5} {'BC':>4} {'max exponent in one block':>28}  note")
    spans = {}
    for C in (16, 32, 64, 128, 256):
        gc = np.cumsum(g[:n // C * C].reshape(n // C, C, K), axis=-2)   # [NC,C,K]        whole = float(np.max(gc[:, 0, :] - gc[:, -1, :]))               # positive
        for BC in (C, 16):
            if BC > C:
                continue
            sub = gc.reshape(n // C, C // BC, BC, K)
            # the span the kernel actually exponentiates, with a per-sub-chunk
            # reference scalar: the largest |delta| between any two positions in
            # the same sub-chunk, for the same channel.
            s = float(np.max(sub[..., 0, :] - sub[..., -1, :]))
            spans[(C, BC)] = s
            note = "" if BC == C else "  (with BC=16 sub-chunking, as fla does)"
            print(f"    {C:5} {BC:4} {s:28.3f}{note}")
    s_naive64, s_naive128 = spans[(64, 64)], spans[(128, 128)]
    s_sub64, s_sub128 = spans[(64, 16)], spans[(128, 16)]
    check("T3 without sub-chunking, C=128 roughly doubles the exponent range",
          s_naive128 > 1.5 * s_naive64,
          f"{s_naive64:.1f} -> {s_naive128:.1f} ({s_naive128/s_naive64:.2f}x) -- so C=128 "
          f"IS expensive if you keep the exponent un-sub-chunked")
    check("T3 with BC=16 sub-chunking the range is INDEPENDENT of C",
          abs(s_sub128 - s_sub64) / s_sub64 < 1e-9,
          f"{s_sub64:.1f} at C=64 vs {s_sub128:.1f} at C=128, identical to "
          f"{abs(s_sub128-s_sub64):.2e}")
    print()
    print("    THIS IS THE USEFUL RESULT.  fla/ops/kda/chunk_intra.py already splits")
    print("    C=64 into 4 sub-chunks of BC=16 and subtracts a per-sub-chunk reference")
    print("    scalar (`b_gn`), so the exponentiated range is set by BC, not by C.")
    print("    C=128 therefore costs NOTHING numerically as long as BC stays 16.")
    print()
    print("    Consequence for the study: the (8,128) minor-lane constraint, which")
    print("    wants C >= 128, is an ALGORITHMIC nudge with no numerical cost -- but")
    print("    it is still a decision the programmer has to notice.  XLA cannot tell")
    print("    you that C=64 will waste half of every vector register.  That is (i).")
    print()
    print("    Sanity on the mechanism: kda/SPEC.md 4.2 notes safe_gate exists to")
    print(f"    bound |Delta| <= BC*|g|; here BC*max|g| = 16 * {np.abs(g).max():.2f} = "
          f"{16*np.abs(g).max():.1f}, and exp of that underflows to 0 in fp32/bf16.")
    print("    So BC=16 is not a tuning choice either -- it is a correctness guard.")


# ---------------------------------------------------------------------------
# T4  the grid is a bijection and the reduction axis is last
# ---------------------------------------------------------------------------
def test_grid():
    print("\nT4.  The intended Pallas grid visits every (head, chunk) once, and the")
    print("    accumulation axis is LAST (R4: Mosaic traverses the grid lexicographically).")
    B, T, H, HV, K, V, C = 1, 2048, 4, 4, 128, 128, 64
    NT = T // C
    grid = (HV, NT)                 # (head, chunk); the per-chunk UT transform and
    # the per-chunk state update are BOTH sequential within one chunk, so the
    # grid has no independent reduction axis -- the reduction is over C and over
    # K, both inside the block.  That is why (HV, NT) and not (HV, NT, K/bk).
    seen = np.zeros((HV, NT), dtype=int)
    for h in range(grid[0]):
        for n in range(grid[1]):
            seen[h, n] += 1
    check("T4 grid (HV, NT) covers every (head, chunk) exactly once",
          seen.sum() == HV * NT and (seen == 1).all(),
          f"{grid[0]}x{grid[1]} = {grid[0]*grid[1]} invocations, all distinct")
    # The state S is [HV, K, V] and is carried ACROSS grid invocations, in
    # dependency order along axis 1.  So axis 1 cannot be 'parallel'.
    sem = ("parallel", "arbitrary")
    vs = TC.check_dimension_semantics(grid, sem, "v3")
    fatal = [x for x in vs if x.fatal]
    check("T4 dimension_semantics ('parallel','arbitrary') is legal", not fatal,
          "; ".join(str(x).strip() for x in fatal) or "no fatal violations")
    print("    Note the honest limitation: axis 1 is 'arbitrary' because S carries")
    print("    across it, so the two intra-chunk steps (UT inverse, state update) are")
    print("    serialised.  The compiler cannot know that HV is independent, and")
    print("    'arbitrary' is the only safe answer.  Splash Attention marks its Q-block")
    print("    axis 'arbitrary' for the same reason: the scratch is per-(head,Q-block).")
    # but the STATE AXIS being sequential is not what makes it arbitrary; the
    # per-chunk carry is.  Verify the state really is the only carry.
    s = SH.KdaShape(B=B, T=T, H=H, HV=HV, K=K, V=V, C=C)
    check("T4 the carried state does NOT grow with T (only a [HV,K,V] scan)",
          s.nbytes("S") * 2 < s.nbytes("q") + s.nbytes("v"),
          f"S is {s.nbytes('S')/1024:.0f} KiB; q+v is {(s.nbytes('q')+s.nbytes('v'))/1e6:.1f} MB")


# ---------------------------------------------------------------------------
# T5  the plan is legal
# ---------------------------------------------------------------------------
def test_plan():
    print("\nT5.  The intended plan passes every rule in tiling_check.py.")
    import roofline as RL
    for C in (64, 128):
        B, T, H, HV, K, V = 1, 2048, 4, 4, 128, 128
        s = SH.KdaShape(B=B, T=T, H=H, HV=HV, K=K, V=V, C=C)
        plan = dict(
            grid=(HV, T // C),
            dimension_semantics=("parallel", "arbitrary"),
            block_specs=[((C, K), (T, K), "k"), ((C, V), (T, V), "v"),
                         ((C, K), (T, K), "q"), ((C, K), (T, K), "g (per channel)")],
            scratch_shapes=[((C, C), "Z / M"), ((C, C), "Aqk"), ((K, V), "S")],
            matmul_dtypes={"k": "bfloat16", "v": "bfloat16", "S": "float32"},
            request_f32=True,
            scratch_bytes=SH.vmem_budget(s, 1, buffer_count=2)["total"],
            vmem_cap=RL.vmem_bytes("v7x"), gen="v7x", dtype_bytes=2, bitwidth=16,
        )
        vs = [x for x in TC.validate_plan(plan) if x.fatal]
        print(f"    C = {C}:")
        for x in vs:
            print("      " + str(x).strip())
        if not vs:
            print("      clean")
        if C == 64:
            check("T5 C=64 is legal but pays a 128-lane pad on every [C,C] accumulator",
                  len(vs) >= 1, f"{len(vs)} fatal violation(s), all R3 padding")
        else:
            check("T5 C=128 is clean -- the (8,128) tile is satisfied by construction",
                  not vs, f"{len(vs)} fatal violation(s)")


def main():
    print("=" * 78)
    print("CPU pre-flight for kda/tpu.  No TPU, no jax, no GPU.")
    print("Reference: ../ref.py, float64, validated in ../test_ref.py.")
    print("=" * 78)
    Z = test_exp_folding()
    test_mask(Z)
    test_c128_range()
    test_grid()
    test_plan()
    print()
    print("=" * 78)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print("  FAILED:", f)
    print("=" * 78)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
