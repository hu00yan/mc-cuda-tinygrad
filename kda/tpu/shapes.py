"""
Analytic cost + materialisation model for KDA on TPU, for three candidate
implementations.  CPU-only, numpy-free.  This file pre-registers the study's
predictions BEFORE any TPU is involved.

Run:  /Users/huyan00/mycode/mc-cuda-tinygrad/.venv/bin/python kda/tpu/shapes.py

The three sides
---------------
S1  xla_jax      pure jnp transcription of the algebra (what kda/tiny_kda.py is,
                 in JAX instead of tinygrad).  XLA fuses what it can; the
                 C^2 K decay tensors exist as real values.
S2  pallas_hand  same algebra, hand-tiled in Pallas/Mosaic.  The programmer
                 folds the decay's exp into the contracting operand so C^2 K
                 never exists, and states grid / BlockSpec / dimension_semantics
                 / prefetch depth / block shapes.
S3  jax_rewrite  pure jnp, but the exp-folding rewrite is applied to the graph
                 before XLA sees it.  Same algorithm as S2, different dialect.

S3 is the control that makes the study falsifiable.  S2 - S1 measures "a human
reached for a low-level dialect".  S2 - S3 measures "the low-level dialect
bought anything beyond the rewrite".  If S2 ~ S3, the whole TPU story collapses
to taxonomy item 6 and the dialect was decoration.  We pre-register that we
EXPECT S3 to capture most of S2's win; that is a result, not a failure.

Every number here is ANALYTIC.  Nothing is measured.  Where a count depends on
a compiler decision we do not make, we print the branch instead of guessing.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roofline as rl  # noqa: E402

ES_BF16 = 2   # bytes/element in the kernels' working dtype
ES_F32 = 4


# ---------------------------------------------------------------------------
# The operator's shapes.  Notation follows kda/SPEC.md exactly.
# ---------------------------------------------------------------------------
class KdaShape:
    def __init__(self, B=1, T=2048, H=4, HV=4, K=128, V=128, C=64, es=ES_BF16):
        assert T % C == 0, f"T={T} must be divisible by C={C}"
        assert HV % H == 0
        self.B, self.T, self.H, self.HV = B, T, H, HV
        self.K, self.V, self.C, self.es = K, V, C, es
        self.NT = T // C
        self.G = HV // H

    def __repr__(self):
        return (f"KdaShape(B={self.B}, T={self.T}, H={self.H}, HV={self.HV}, "
                f"K={self.K}, V={self.V}, C={self.C}, NT={self.NT})")

    def elems(self, name):
        B, NT, HV, C, K, V = self.B, self.NT, self.HV, self.C, self.K, self.V
        table = {
            "q":        B * self.T * self.H * K,
            "k":        B * self.T * self.H * K,
            "v":        B * self.T * HV * V,
            "g":        B * self.T * HV * K,       # log gate, per key CHANNEL
            "beta":     B * self.T * HV,
            "o":        B * self.T * HV * V,
            "S":        B * HV * K * V,            # carried state, constant in T
            "g_cum":    B * NT * HV * C * K,
            "kg":       B * NT * HV * C * K,
            "D_strict": B * NT * HV * C * C * K,   # <-- the C^2 K monster
            "D_causal": B * NT * HV * C * C * K,   # <-- ditto
            "Z":        B * NT * HV * C * C,
            "Btri":     B * NT * HV * C * C,
            "M":        B * NT * HV * C * C,
            "w":        B * NT * HV * C * K,
            "u":        B * NT * HV * C * V,
            "v_new":    B * NT * HV * C * V,
            "Aqk":      B * NT * HV * C * C,
            "o_chunk":  B * NT * HV * C * V,
        }
        return table[name]

    def nbytes(self, name, es=None):
        return self.elems(name) * (self.es if es is None else es)

    def flops(self):
        """2 FLOP/MAC, counted consistently.  See kda/cost.py and SPEC.md §8
        row 5: Kimi Linear's Eq. 13 undercounts this by 1.45x because it counts
        three C x C x K intra-chunk matmuls instead of five.  We count five --
        Z, w, u, Aqk, Aqk@v_new -- plus three inter-chunk ones."""
        BNT, C, K, V = self.B * self.NT * self.HV, self.C, self.K, self.V
        per = {
            "Z":         2 * C * C * K,
            "w":         2 * C * C * K,
            "u":         2 * C * C * V,
            "v_new":     2 * C * K * V,
            "Aqk":       2 * C * C * K,
            "Aqk@v_new": 2 * C * C * V,
            "S_update":  2 * C * K * V,
        }
        return sum(per.values()) * BNT, {k: v * BNT for k, v in per.items()}

    def compulsory_bytes(self):
        """Bytes that MUST cross HBM on any implementation: every input once,
        the output once, and the K x V state read+written once per call."""
        return (self.nbytes("q") + self.nbytes("k") + self.nbytes("v")
                + self.nbytes("g") + self.nbytes("beta") + self.nbytes("o")
                + 2 * self.nbytes("S"))


# ---------------------------------------------------------------------------
# Materialisation model.  This is the load-bearing claim of the whole study and
# it is STATIC: it follows from what each side puts in the graph, not from what
# a profiler happened to observe.
# ---------------------------------------------------------------------------
IO_SIDES = {
    "S1 xla_jax": dict(
        note="direct transcription, as kda/tiny_kda.py is.  The graph decides "
             "the intermediate exists; the scheduler cannot undo that.",
        write=["g_cum", "kg", "D_strict", "D_causal", "Z", "Btri", "M", "w", "u"],
        read=["g", "beta", "q", "k", "v"], out=["o"], state=["S"],
        folded=False,
    ),
    "S2 pallas_hand": dict(
        note="exp folded into the contracting operand, so D_* never exists.  "
             "Z/Aqk/M are [C,C] and live in VMEM scratch.",
        write=[], read=["g", "beta", "q", "k", "v"], out=["o"], state=["S"],
        folded=True,
    ),
    "S3 jax_rewrite": dict(
        note="the same rewrite expressed in jnp.  XLA sees a smaller graph.",
        write=[], read=["g", "beta", "q", "k", "v"], out=["o"], state=["S"],
        folded=True,
    ),
}

#: The C^2 K tensors exist ONLY on the unfolded sides.  This is the single
#: structural fact the CUDA study found (notes/03-kda.md: two 33.6 MB
#: intermediates dominating GPU time) and it is architecture-independent.
MONSTERS = ("D_strict", "D_causal")

#: Auxiliary [C,C]-shaped intermediates that a fused kernel re-materialises in
#: registers.  We give S1 the *benefit of the doubt* on these: XLA can fuse
#: elementwise+reduce chains, so we do NOT charge it for them.  Only the
#: C^2 K tensors are charged, because those feed a `dot` and XLA would have to
#: change the graph to avoid materialising them.
AUX_C_C = ("Z", "Btri", "M", "Aqk")


def monster_bytes(s, side):
    return 0.0 if IO_SIDES[side]["folded"] else sum(s.nbytes(m) for m in MONSTERS)


def report(s, gen="v6e", verbose=True):
    f_total, _ = s.flops()
    comp = s.compulsory_bytes()
    rows = []
    for side in IO_SIDES:
        mon = monster_bytes(s, side)
        hbm = comp + mon
        rows.append(dict(side=side, compulsory=comp, monster=mon, hbm=hbm,
                         oi=f_total / max(hbm, 1),
                         c=rl.classify(gen, f_total, hbm)))
    if verbose:
        print(s)
        print(f"  FLOPs (2/MAC, 5 intra-chunk C^2K matmuls): {f_total/1e9:.2f} GFLOP")
        print(f"  {'side':15} {'compulsory':>11} {'C^2K monster':>13} {'HBM total':>11} "
              f"{'OI F/B':>8}  regime @ {gen}")
        for r in rows:
            print(f"  {r['side']:15} {r['compulsory']/1e6:10.1f}M {r['monster']/1e6:12.1f}M "
                  f"{r['hbm']/1e6:10.1f}M {r['oi']:8.1f}  {r['c']['regime']}")
        print("  hard time floors on " + gen + " -- the bytes each side CANNOT avoid:")
        for r in rows:
            print(f"    {r['side']:15} floor {r['c']['t_hard_floor_ms']:8.2f} ms  "
                  f"(t_mem {r['c']['t_mem_lower_bound_s']*1e3:7.2f} / "
                  f"t_flop {r['c']['t_flop_lower_bound_s']*1e3:7.3f} ms)")
        c2k = s.nbytes("D_strict")
        cap = rl.hbm_bytes(gen)
        cap_s = f"{cap/2**30:.0f} GiB" if cap else f"HBM capacity UNSOURCED for {gen}"
        verdict = ("fits HBM" if (cap and c2k < cap) else
                   ("EXCEEDS HBM: S1 cannot run at this shape" if cap
                    else "cannot be checked -- fill GENERATIONS[gen]['hbm_gib']"))
        print(f"  one C^2K tensor: {c2k/1e6:.1f} MB ({s.elems('D_strict')/1e6:.1f}M elems) "
              f"vs {cap_s} -- {verdict}")
        print()
    return rows


def vmem_budget(s, heads_per_core, buffer_count=2, chunk_resident=True):
    """Bytes a hand-written Pallas KDA kernel must keep resident on ONE
    TensorCore, for `heads_per_core` heads.

    Per head, all in VMEM:
      * K block       [C, K]  bf16      (the gram operand)
      * V block       [C, V]  bf16
      * Q block       [C, K]  bf16
      * accumulators  [C, C]  fp32  x4  (Z/Btri, M, the running Aqk, the w,u
                                           staging pair is [C,K]+[C,V], folded in)
      * state S       [K, V]  fp32      (the ONLY carried state; constant in T)
    each DMA block held `buffer_count` times -- the pipeline depth, i.e.
    pl.BlockSpec(..., pipeline_mode=pl.Buffered(buffer_count=n)).  Pallas's
    default is 2 ("The default buffer count is 2 for all inputs and outputs").

    chunk_resident=False models the alternative: keep only the [C,C] and [K,V]
    state on chip and re-stream K/V/Q per use.  That is what the unfolded
    S1 graph effectively forces, and it is why S1 is memory bound.

    KNOWN SIMPLIFICATION: real Pallas TPU kernels expose FOUR block-size knobs
    (bq, bkv, cq, ckv) because the DMA block and the *compute* block differ --
    a compute block must fit in VREGs, a DMA block must fit the 512-byte DMA
    granule.  We collapse them; splitting them is the obvious extension and
    tiling_check.py is where it would go.
    """
    C, K, V = s.C, s.K, s.V
    hb = ES_BF16
    dma = (C * K * hb) + (C * V * hb) + (C * K * hb)     # K, V, Q blocks
    acc = (C * C * ES_F32) * 4 + (K * V * ES_F32)        # [C,C] scratch + state
    live = dma * buffer_count + acc if chunk_resident else acc + dma
    return dict(total=live * heads_per_core, per_head=acc + dma,
                accumulators=acc, dma=dma, buffer_count=buffer_count,
                heads_per_core=heads_per_core, chunk_resident=chunk_resident)


def main():
    print("=" * 78)
    print("A.  KDA shapes.  We need a C^2K tensor big enough to see, small")
    print("    enough that all three sides can run on a free TPU.")
    print("=" * 78)
    shapes = [
        KdaShape(B=2, T=1024, H=4, HV=4, K=64, V=64, C=64),   # CUDA-study shape
        KdaShape(B=1, T=2048, H=4, HV=4, K=128, V=128, C=64),
        KdaShape(B=1, T=4096, H=8, HV=8, K=128, V=128, C=64),
        KdaShape(B=4, T=4096, H=32, HV=32, K=128, V=128, C=64),  # notes/03 target
    ]
    for s in shapes:
        report(s, gen="v6e")
        print("  " + "-" * 72)

    print()
    print("=" * 78)
    print("B.  Does the KDA intra-chunk [C,C] block fit a TPU MXU at all?")
    print("=" * 78)
    # Taxonomy item 4's analogue, and the reason KDA on TPU is not a mere port.
    # The MXU is 128x128 pre-v6e, 256x256 on v6e+.  A [C,C]=[64,64] block
    # contracting K=128 leaves the array mostly idle AND runs the systolic
    # pipeline for at most min(K, MXU) steps.  arXiv:2507.11331 measured exactly
    # this failure for FlashAttention on TPUv5e: 4.83x lower attention FLOP/s
    # utilisation than a hardware that keeps everything on the array.  KDA has
    # MORE small matmuls than attention does, so we predict worse.
    for gen in ("v3", "v5e", "v6e"):
        mxu = rl.GENERATIONS[gen]["mxu"]
        C, K = 64, 128
        fill = (min(C, mxu) / mxu) ** 2 * (min(K, mxu) / mxu)
        print(f"  {gen}: MXU {mxu}x{mxu}; [C,C]={C} contracting K={K} -> "
              f"array fraction {fill:.3f}, pipeline {min(K, mxu)}/{mxu} steps")
    print("  A [64,64] output tile on a 256-wide MXU wastes 75% of the array on")
    print("  the N axis alone.  Taxonomy item 1 (tile sizes) is not a tuning knob")
    print("  on TPU for KDA; it is a structural mismatch with the hardware shape.")

    print()
    print("=" * 78)
    print("C.  VMEM residency vs pipeline depth, v7x (64 MB/TC -- our only")
    print("    sourced VMEM figure, arXiv:2604.15464 Fig. 1).")
    print("=" * 78)
    cap = rl.vmem_bytes("v7x", scope="per_core")
    s = KdaShape(B=1, T=2048, H=4, HV=4, K=128, V=128, C=64)
    print(f"  VMEM per TensorCore: {cap/2**20:.0f} MiB;  kernel shape {s}")
    print(f"  {'heads/core':>10} {'bufs':>5} {'VMEM MiB':>9}  chunk-resident?")
    for hpc in (1, 2, 4, 8, 16):
        for nb in (2, 3, 4):
            b = vmem_budget(s, hpc, buffer_count=nb)
            ok = "yes" if b["total"] < cap else "NO -- OOM or VMEM spill"
            print(f"  {hpc:10} {nb:5} {b['total']/2**20:9.1f}  {ok}")
    print()
    b1 = vmem_budget(s, 1, buffer_count=2)
    print(f"  Per head: {b1['accumulators']/1024:.0f} KiB of fp32 accumulator+state, "
          f"{b1['dma']/1024:.0f} KiB of DMA blocks per buffer.")
    print(f"  The [K,V] state alone is {s.K*s.V*ES_F32/1024:.0f} KiB/head and does NOT")
    print("  shrink with T, so heads-per-core is bounded by VMEM / (state + accum).")
    print("  This is taxonomy item 7 on TPU: the budget is VMEM, and the")
    print("  compensating knob is how many heads you keep resident, not warps.")
    print()
    print("  Raising pipeline depth (bufs) is free in the taxonomy's language --")
    print("  it is one integer -- but it costs VMEM linearly, so items 1 and 9 are")
    print("  the SAME knob on TPU.  On CUDA num_stages and block sizes are coupled")
    print("  only loosely (shared memory); here they are the same resource.")
    print()
    print("  UNVERIFIED: v3/v5e VMEM.  Pallas docs only say '16MB+'.  Run tpu-info")
    print("  and fill roofline.GENERATIONS['v5e']['vmem_mb'] before quoting v5e rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
