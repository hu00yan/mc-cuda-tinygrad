"""
SIDE 2 of the TPU comparison: KDA hand-tiled in Pallas, lowering to Mosaic.

*** THIS FILE HAS NEVER BEEN EXECUTED. ***
The repo venv has no jax (verified) and there is no TPU reachable from here.
It is written against the Pallas TPU API as documented at
https://docs.jax.dev/en/latest/pallas/tpu/{quickstart,pipelining,details,matmul}.html
and it is meant to be pasted into a Kaggle notebook with `pip install -U "jax[tpu]"`.

What this file is FOR, and what it is not for
-----------------------------------------------
It is a *design sketch with the load-bearing choices made explicit*, not a
finished kernel.  The purpose of the study is to enumerate and price the choices,
and most of the value here is in the annotations: each comment marked DECISION
is one of the taxonomy items, and the argument beside it says why the language
forces the choice rather than the search finding it.

The things I could NOT resolve from the documentation, and have therefore left as
named parameters rather than guessed:
  * whether `pl.when` inside an emit_pipeline body is cheap enough for the
    C-1 dependent steps of the UT transform.  Pallas TPU's own docs warn "loop
    primitives get fully unrolled during the compilation at the moment, so try to
    keep the loop trip count reasonably small" and "Overusing control flow can
    lead to significant regressions in low-level code generation".  A C=128 UT
    transform is 127 dependent steps; that is a real risk and it is the single
    most likely thing to be slow in this kernel.
  * whether the [C,C] fp32 accumulator, physically padded to [C,128], gets
    relayouted more than once.  Only an LLO dump answers it.
  * whether the state S should live in scratch (VMEM, persists across grid
    iterations) or be an explicit output that the caller threads.  Splash
    Attention keeps m/l/o in scratch; RPA writes the KV cache back to HBM.  For
    KDA the state is [HV,K,V] and MUST persist across grid iterations in
    lexicographic order, which is exactly what scratch is for, so scratch it is.
    The hazard: if `dimension_semantics` ever marks the chunk axis "parallel",
    the scratch write is a race, and Pallas will not warn you.
"""

# ---------------------------------------------------------------- imports
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# ---------------------------------------------------------------- config
# DECISION (taxonomy item 1, tile sizes): C = 128, not the paper's 64.
#   Reason: on TPU a 2-D vector tile is (8, 128).  C = 64 means the [C,C] gram
#   and the UT inverse are physically [64,128] fp32 -- 2x the VMEM, plus a
#   relayout on every access.  kda/tpu/tiling_check.py::check_minor_lane_fit
#   flags this, and kda/tpu/test_design.py T3 shows the change is NUMERICALLY
#   FREE as long as the BC = 16 secondary chunking is kept (the exponentiated
#   range is set by BC, not C: measured identical to 2.8e-14 at C=64 and
#   C=128).  This is the clearest (i) in the study: XLA cannot tell you that
#   C = 64 wastes half of every vector register, and the fix is a change to the
#   ALGORITHM, not to a search space.
C = 128
BC = 16          # secondary sub-chunk.  NOT a tuning knob: it bounds the
                 # exponent, and exp of a larger span underflows to 0 in
                 # bf16/fp32.  kda/SPEC.md 4.2.
BUFFER_COUNT = 3   # DECISION (item 9, pipelining depth).  Pallas's default is 2.
                   # Costs VMEM linearly, and VMEM is the same budget the
                   # accumulator lives in, so items 1 and 9 are ONE knob on TPU.
                   # 2 was measured at 5.9% of v5e peak for a 4096^3 matmul vs
                   # 78.5% for larger blocks (docs/pallas/tpu/matmul.html), which
                   # is about MXU starvation, not about the buffer count -- so
                   # sweep this only AFTER the blocks are right.


@functools.partial(jax.jit, static_argnames=("B", "T", "HV", "K", "V", "chunk", "nbuf"))
def kda_pallas(q, k, v, g, beta, *, B, T, HV, K, V, chunk=C, nbuf=BUFFER_COUNT):
    """q,k : [B,T,HV,K]   v : [B,T,HV,V]   g : [B,T,HV,K]   beta : [B,T,HV]

    Returns o [B,T,HV,V] and the final state S [B,HV,K,V].

    The ONE structural fact that shapes this whole file: on TPU the grid is
    traversed LEXICOGRAPHICALLY and the only state that crosses a grid iteration
    is S [HV,K,V].  So the grid is (head, chunk) -- axis 0 parallel, axis 1
    arbitrary -- and Mosaic walks axis 1 in order carrying S in scratch.  There
    is no reduction grid axis, because the UT transform's reduction is over C and
    over K, both of which are INSIDE a block.  kda/tpu/test_design.py T4 checks
    the bijection.

    Pallas's own guidance matches: for a matmul, "the grid axis corresponding to
    the reduction dimension has to be the last one, since the output window does
    not vary along this axis."
    """
    NT = T // chunk
    nsub = chunk // BC

    # DECISION: 2-D shapes with the minor-most extent at 128.  A 3-D [1, chunk, K]
    # block has K minor-most, which is fine, but a 3-D [HV, chunk, K] block would
    # put `chunk` in the second-minor slot -- legal (64 or 128 is a multiple of
    # 8) and then C in the minor slot would be illegal.  So we keep the arrays 3-D
    # as [HV, T, K] and block them as [HV, chunk, K]: the two minor dims are
    # (chunk, K) = (128, 128) and both satisfy the rule.
    k_hbt = jnp.transpose(k, (0, 2, 1, 3)).reshape(B, HV, T, K)     # [B,HV,T,K]
    q_hbt = jnp.transpose(q, (0, 2, 1, 3)).reshape(B, HV, T, K)
    v_hbt = jnp.transpose(v, (0, 2, 1, 3)).reshape(B, HV, T, V)
    g_hbt = jnp.transpose(g, (0, 2, 1, 3)).reshape(B, HV, T, K)
    b_hbt = jnp.transpose(beta, (0, 2, 1)).reshape(B, HV, T)          # [B,HV,T]

    def _spec(feat, last):
        return pl.BlockSpec(
            block_shape=(HV, chunk, last),
            index_map=lambda h, n: (h, n, 0),
            pipeline_mode=pl.Buffered(buffer_count=nbuf),
        )

    # beta's minor dim is 1 -- that is a rank-1-in-minor problem.  Fold it into
    # the gate tensor instead: concat beta as an extra leading 1-slice is not
    # expressible, so we carry beta as [B,HV,T,1] and let the minor-most block
    # dim equal 1, which IS allowed ("must be equal to the respective dimension
    # of the overall array").  This is DECISION-shaped: a human chose the layout
    # that makes the constraint satisfiable.
    b_hbt1 = b_hbt[..., None]

    scratch = dict(
        # the ONLY carried state.  [HV,K,V] fp32, persists across grid iterations.
        S=pltpu.VMEM((HV, K, V), jnp.float32),
        # [HV,chunk,chunk] fp32.  Physically [HV,128,128] because chunk == 128.
        Z=pltpu.VMEM((HV, chunk, chunk), jnp.float32),
        M=pltpu.VMEM((HV, chunk, chunk), jnp.float32),
        Aqk=pltpu.VMEM((HV, chunk, chunk), jnp.float32),
        w=pltpu.VMEM((HV, chunk, K), jnp.float32),
        u=pltpu.VMEM((HV, chunk, V), jnp.float32),
        vn=pltpu.VMEM((HV, chunk, V), jnp.float32),
    )

    def kernel(k_hb, q_hb, v_hb, g_hb, b_hb, o_hb, *, S, Z, M, Aqk, w, u, vn, nsub):
        # --- 1. within-chunk cumsum of the log gate.  DECISION: this is a cumsum
        # over a BLOCK-LOCAL axis of length `chunk`.  Pallas supports sum/max/min
        # reductions but the docs are silent on cumsum, and "loop primitives get
        # fully unrolled" -- so this may need to be an explicit Python loop over
        # `chunk` (128 unrolled steps) or a matrix-multiply-based scan.  THIS IS
        # AN OPEN QUESTION AND THE FIRST THING TO PROFILE.
        g_h = g_hb[0]                                     # [chunk, K]
        b_h = b_hb[0]                                     # [chunk, 1]
        k_h = k_hb[0]                                     # [chunk, K]
        q_h = q_hb[0]
        v_h = v_hb[0]                                     # [chunk, V]

        # --- 2/3. the UT transform's Z, with the exp FOLDED (this is the whole
        # point).  NO [chunk,chunk,K] tensor is ever formed.  For each pair of
        # BC-sized sub-chunks (p, r) with r <= p:
        #     Z_blk = (K_p . exp(g_p - gn)) @ (K_r . exp(gn - g_r)).T
        # The factorisation is exact for any reference gn; BC bounds the
        # exponent.  kda/tpu/test_design.py T1 verifies it against ../ref.py in
        # float64 (relerr 2.5e-16) and T1c measures the numerical bill: the
        # off-diagonal block (nsub-1, 0) needs 2.5x the diagonal bound in its
        # RIGHT operand.  fla hand-schedules around this ("4 diagonal row-loops +
        # 6 off-diagonal matmuls", kda/SPEC.md 4.4).  So does this.
        Zl = jnp.zeros((chunk, chunk), dtype=jnp.float32)
        # the cumsum, expressed as a masked matmul against a lower-triangular
        # all-ones: g_cum = L @ g, with L strictly lower-triangular of ones.
        L = jnp.tril(jnp.ones((chunk, chunk), dtype=jnp.float32), -1) \
            + jnp.eye(chunk, dtype=jnp.float32)
        g_cum = L @ g_h                                  # [chunk, K], exact, ONE matmul
        gam = jnp.exp(g_cum)                             # gamma^r, safe (<= 1)
        kg = gam * k_h

        for p in range(nsub):
            sl_p = slice(p * BC, (p + 1) * BC)
            gn = g_cum[sl_p.start]                        # per-channel reference
            left = k_h[sl_p] * jnp.exp(g_cum[sl_p] - gn)  # <= 0 exponent
            for r in range(nsub):
                if r > p:
                    continue                              # Z is 0 there
                sl_r = slice(r * BC, (r + 1) * BC)
                right = k_h[sl_r] * jnp.exp(gn - g_cum[sl_r])
                blk = left @ right.T                     # [BC,BC] -- THIS is the MXU
                if r == p:
                    # causal mask on the [BC,BC] RESULT.  A stronger version
                    # skips the sub-block entirely via an SMEM indirection, which
                    # is how Splash Attention skips fully-masked KV blocks.
                    m = jnp.tril(jnp.ones((BC, BC), dtype=bool), -1)
                    blk = jnp.where(m, blk, 0.0)
                Zl = Zl.at[sl_p, sl_r].add(blk)
        Z[0] = Zl

        # --- 4. UT transform: (I + StrictTril(Diag(beta) Z))^-1 . Diag(beta)
        # by forward substitution.  C-1 = 127 DEPENDENT steps, and Pallas TPU
        # fully unrolls loops.  This is the most likely slow part of the kernel
        # and it is the one thing here that has no jnp equivalent.
        Btri = (b_h * Zl)                                 # Diag(beta) . Z
        A = -jnp.tril(jnp.ones((chunk, chunk), dtype=bool), -1) * Btri
        A = A + jnp.eye(chunk, dtype=jnp.float32)         # (I + StrictTril)^-1
        for i in range(1, chunk):
            # row_i (1 x i) @ A[:, :i] (i x i) -> update.  ONE MXU per step, 127
            # of them, strictly dependent.  127 MXU ops with a 128-cycle systolic
            # pipeline each is ~16K cycles of pure latency with nothing to hide it.
            row = A[i, :i] @ A[:i, :i]                   # [i]
            A = A.at[i, :i].add(row)
        M[0] = A * b_h[:, 0]                             # * Diag(beta) = the paper's M

        # --- 5/6. w and u.  These are real matmuls, and they are the only place
        # the MXU does useful-width work in this kernel.
        w[0] = M[0] @ kg
        u[0] = M[0] @ v_h

        # --- 7a. v_new = u - w S .  S comes from scratch, i.e. from the PREVIOUS
        # grid iteration.  This is the inter-chunk scan, and it is expressed as a
        # single carveout: the "grid" IS the scan.
        wS = w[0] @ S[0]                                 # [chunk, V]
        vn[0] = u[0] - wS

        # --- 7b. Aqk = Tril((Q . D_causal) K^T).  Uses PLAIN q -- the decay is
        # already in the exp.  (kda/SPEC.md 6.2: using Gamma . q double-counts.)
        qg = q_h * jnp.exp(g_cum)
        A_ = qg @ k_h.T                                  # [chunk,chunk]
        A_ = jnp.where(jnp.tril(jnp.ones((chunk, chunk), dtype=bool)), A_, 0.0)
        Aqk[0] = A_

        # --- 7c. output for this chunk
        o_hb[0] = qg @ S[0] + Aqk[0] @ vn[0]

        # --- 7d. state update, in the SAME buffer S.  Note the direction: this
        # writes S[0], and the next grid iteration reads it.  Mosaic guarantees
        # lexicographic order, so this is safe -- and it is ONLY safe because we
        # marked axis 1 "arbitrary".  If you mark it "parallel" this becomes a
        # silent data race.
        #     S^C = Diag(g_cam[-1]) S^0 + ((g_cam[-1]/g_cam) . K)^T v_new
        # Computed with LOG DIFFERENCES, never the division: kda/SPEC.md 4.2 says
        # K/Gamma must never be materialised because Gamma underflows and the
        # quotient becomes inf.
        g_last = g_cum[-1]
        S[0] = jnp.exp(g_last)[None, :] * S[0] + (kg * jnp.exp(g_last - g_cum)).T @ vn[0]

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((B, HV, T, V), jnp.float32),
        grid=(HV, NT),                       # DECISION: see module docstring
        in_specs=[_spec(K, K), _spec(K, K), _spec(V, V), _spec(K, K),
                  pl.BlockSpec((HV, chunk, 1), lambda h, n: (h, n, 0),
                               memory_space=pltpu.VMEM,
                               pipeline_mode=pl.Buffered(buffer_count=nbuf))],
        out_specs=_spec(V, V),
        scratch_shapes=[(k, v) for k, v in scratch.items()],
        compiler_params=pltpu.CompilerParams(
            # DECISION (item 3's closest TPU analogue).  All "parallel" first,
            # then all "arbitrary" -- Pallas requires that order.  Axis 0 is the
            # heads: independent, because scratch is per (head, chunk).  Axis 1
            # is the chunks: DEPENDENT, because S carries across it.  XLA cannot
            # infer either.
            dimension_semantics=(pltpu.PARALLEL, pltpu.ARBITRARY),
        ),
    )(k_hbt, q_hbt, v_hbt, g_hbt, b_hbt1)


# ---------------------------------------------------------------- what to measure
MEASUREMENTS = """
Every number below is arch-independent EXCEPT the last, and the last is reported
as a fraction of the roofline so it survives a TPU generation change.  Order them
by how much they cost to get wrong.

M1  cost_analysis() HBM bytes and FLOPs on the compiled program.  FREE, no run
    needed beyond .compile().  This is the headline: kda/tpu/shapes.py predicts
    S1 moves ~279 MB where ~10.8 MB is compulsory, because 268 MB of it is the two
    [C,C,K] tensors.  For S2/S3 the same call should report ~10.8 MB.
    PROVES: whether the intermediate is in the program at all.  Nothing else does.

M2  Count and total size of dma.hbm_to_vmem / hbm_to_smem in the LLO dump.
    Dump with LIBTPU_INIT_ARGS=--xla_jf_dump_to=... --xla_jf_dump_llo_text=true
    and xla_mosaic_dump_to for the Mosaic IR.  arXiv:2607.20466 used the same
    approach on Mosaic/TPU; patricktoulme (11 Jan 2026) reports 1,651 VLIW bundles
    and 14 MB for Splash Attention vs 10,245 bundles and 526 MB for naive
    attention on v6e -- a 37x byte ratio that is a LAYOUT fact, not a speed fact.
    PROVES: what the hardware is actually asked to move, independent of clock.

M3  VLIW bundle count from the same LLO dump.  A proxy for issue pressure and,
    more usefully, for how much the compiler had to pad.  Same source as M2.

M4  Number of fusions / custom-calls in the HLO.  The TPU analogue of
    notes/03-kda.md's "12,090 kernels".  For S2 it should be exactly ONE
    custom-call, because a Pallas kernel is opaque to XLA -- so this metric also
    measures the flip side: you have given up the whole-graph view.

M5  MXU utilisation = FLOPs / (median_device_time * peak_bf16).  From
    jax.profiler traces, NOT perf_counter.  Report as a FRACTION, per the
    generation.  Compare against kda/tpu/shapes.py: the OI of the folded kernel is
    ~112 FLOP/byte against a v6e ridge of 560, so KDA-on-TPU is MEMORY BOUND at
    every shape we can afford, and MXU utilisation is expected to be LOW even for
    a perfect kernel.  Do not read that as a failure.
    PROVES: only the schedule, and only after M1-M4 are already known.

M6  Correctness, at three tolerances: fp64 reference (1e-12), fp32 operands
    (1e-6), bf16 matmul (1e-2).  kda/tpu/test_design.py T1 measures what to
    expect at each.  AND: assert that the state S matches.  A wrong S gives a
    finite plausible output on chunk 0 and garbage from chunk 1, which is the
    failure mode kda/test_ref.py exists to catch.
    PROVES: that any speed number is about the same operator.
"""
