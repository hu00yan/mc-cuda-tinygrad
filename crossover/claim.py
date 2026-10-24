"""
claim.py -- the crossover thesis stated as predictions with quantities.

Run:  .venv/bin/python crossover/claim.py
      .venv/bin/python crossover/claim.py --model 2      # one model only

THE CLAIM BEING TESTED (from the brief, quoted so it cannot drift)
---------------------------------------------------------------
  "The crossover between 'the compiler wins' and 'the human wins' is not a
   quality gradient at all. It is set by whether the materialised intermediate
   fits in on-chip memory, and it is a step function rather than a slope."

Stated as a prediction, for operator X on device D with intermediate size s:

  P1   winner(D, X, s) = compiler  if  s <= C(D)
                          human     if  s  > C(D)
  P2   C is a CAPACITY, not a quality score: moving to a device with a bigger C
       moves the crossover to a bigger s, and nothing else does.
  P3   the transition is a STEP in the win-ratio, not a slope.
  P4   the arch matters less than s/C(D).

THREE MODELS, because the brief's C is underspecified and the three readings
give three different answers.  Reporting all three is the point.

  MODEL 1  device capacity.  C = L2.  The intermediate either hides in the
           device-wide cache or it does not.  This is the reading the brief's
           "fits in on-chip memory" most naturally suggests.

  MODEL 2  per-SM tile.  A *fused* kernel never needs the whole intermediate on
           chip -- it needs one row-slab of it, [R, n], resident on the SM that
           produces and consumes it.  The slab fits if
               R * n * es <= C_onchip_per_sm
           and the fused form is only competitive if that slab is TALL enough to
           fill the machine's arithmetic pipeline, R >= M_native.  Eliminating R:
               n <= C_onchip_per_sm / (M_native * es)
           This is the reading that is actually true of a fused CUDA kernel, and
           it is a much stronger constraint than Model 1.  See notes/07 §4.

  MODEL 3  bandwidth, no cache at all.  The crossover is where the materialising
           form's DRAM time equals the avoiding form's:
               P * s * es / BW_compiler  ==  s_comp * es / BW_human
               s  =  s_comp * BW_compiler / (P * BW_human)
           This is what the brief's own "8.00 GiB against 0 bytes of DRAM" is
           really asserting.  It is arithmetic, not science, and it is included
           so that Model 1 can be shown to disagree with it.

  P2 IS WHERE MODELS 1 AND 2 DISAGREE MOST, and it is the interesting part: the
  model that uses the device-wide L2 says a Blackwell B200 should absorb a 126 MB
  intermediate comfortably; the model that uses per-SM shared memory says the B200
  and the H100 are IDENTICAL, because both have 228 KB of scratch per SM.  That
  is a 2.5x disagreement about the largest card in the study, and it is settled
  by a measurement, not by an argument.

M_native: rows of the machine's native MMA tile.
  CUDA: the m16n8k16 / m16n8k8 tensor-core shape has M=16 per instruction; a
  warp-tiled fp32 FFMA tile and a Pallas BlockSpec's minor dimension both want
  64-128 rows to fill a pipeline.  64 is the conservative choice and is the one
  used below; the sensitivity to 16 and 128 is printed.
  [STRUCTURAL / READ-FROM-SOURCE, and genuinely arguable -- see notes/07 §4.3]
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import devices as D  # noqa: E402

MiB, GiB = 1024 ** 2, 1024 ** 3

# bytes per element.  fp32 is the study's default for KDA (notes/06 §2 is
# emphatic that the measurement must be fp32); bf16 is the kernels' working dtype.
ES_FP32, ES_BF16 = 4, 2

# the two shapes the study actually uses.  THEY ARE NOT THE SAME SHAPE, and the
# difference is 128x in the intermediate.  See notes/07 §6.
KDA_BENCH_SHAPE = dict(B=2, T=512, H=4, K=64, C=64, es=ES_FP32,
                       where="kda/perf_kda.py:90  (the shape that is benchmarked)")
KDA_COST_SHAPE = dict(B=4, T=4096, H=32, K=128, C=64, es=ES_BF16,
                      where="kda/cost.py:17  (the shape that is cost-modelled, and "
                            "the source of the '8.00 GiB' figure)")

M_NATIVE_CUDA = 64


# -----------------------------------------------------------------------------
# 1. attention: the intermediate is [B,H,n,n]
# -----------------------------------------------------------------------------
def attn_score_bytes(B: int, H: int, n: int, es: int) -> int:
    return B * H * n * n * es


def attn_flops(B: int, H: int, n: int, d: int, causal: bool) -> int:
    """Two matmuls, 2 FLOP/MAC.  Causal halves the score elements and both
    matmuls, so a causal run is exactly half a dense one at this shape."""
    f = 4.0 * B * H * n * n * d
    return int(f * (0.5 if causal else 1.0))


def attn_compulsory_bytes(B: int, H: int, n: int, d: int, es: int,
                          causal: bool) -> int:
    """q, k, v in, o out, one touch each.  This is the FLOOR for any
    implementation, fused or not, and it is the s_comp in Model 3."""
    return 4 * B * H * n * d * es


def model1_crossover_n(dev: str, B: int, H: int, es: int) -> float:
    """n at which the score matrix equals L2.  Model 1: device capacity."""
    return math.sqrt(D.DEVICE_SPECS[dev]["l2_bytes"] / (B * H * es))


def model2_crossover_n(dev: str, B: int, H: int, es: int,
                       m_native: int = M_NATIVE_CUDA) -> float:
    """n at which the per-SM slab can no longer be a full-height tile.
    Model 2: per-SM on-chip capacity divided by the machine's native tile shape."""
    return D.onchip_per_sm(dev) / (m_native * es)


def model3_crossover_n(dev: str, B: int, H: int, d: int, es: int, causal: bool,
                       passes: int, eff_ratio: float = 1.0) -> float:
    """n at which the materialised form's DRAM time equals the avoiding form's.
    Model 3: pure bandwidth, no cache.

    `eff_ratio` is BW_compiler / BW_human, where BW_* are *achieved* bandwidths.
    With eff_ratio = 1 this is a property of the shape alone and is IDENTICAL on
    every device -- which is Model 3's real content: the bandwidth crossover does
    not know what card it is on.  Any device dependence has to enter through
    eff_ratio, i.e. through how well each side uses the memory system, and that
    is exactly the quantity nobody has measured.  Sweep it and watch the
    predicted crossover move.
    """
    per_token = attn_compulsory_bytes(B, H, 1, d, es, causal)      # bytes per token
    # passes * n^2 * B*H * es * BW_human == n * per_token * BW_compiler
    return per_token * eff_ratio / (passes * B * H * es)


# -----------------------------------------------------------------------------
# MODEL 4 -- the one that comes out of the byte accounting, and it is the
# opposite sign to the brief.
# -----------------------------------------------------------------------------
def max_key_block(dev: str, d: int, es: int) -> int:
    """The largest key-block width Bc whose [Bc, d] slab fits in one SM's scratch.

    This is the quantity the brief's "fits in on-chip memory" is really about,
    and note that L2 does not appear in it at all.
    """
    return D.onchip_per_sm(dev) // (d * es)


def max_byte_ratio(dev: str, d: int, es: int, passes: int) -> float:
    """The LARGEST traffic ratio the avoiding form can ever achieve over the
    materialising one, and it does NOT depend on n.

    Materialised: the score tensor is written and read `passes` times.
                    traffic ~ passes * n^2 * es
    Online:       the score tile is on chip, but q and k are re-read once per key
                  block, so
                    traffic ~ 2 * (n^2 / Bc) * d * es
    ratio = passes * Bc / (2 * d)
    and Bc is capped by on-chip capacity, so the ratio is capped too.

    THIS IS THE STRUCTURAL RESULT.  Both forms' traffic is O(n^2).  They differ
    in the CONSTANT, not the growth rate.  So making the intermediate bigger
    makes both forms equally worse, and cannot by itself make the human win by
    more.  What on-chip capacity buys is a bigger Bc, which lowers the human's
    constant -- i.e. capacity helps the HUMAN, the opposite sign to the brief.
    """
    bc = max_key_block(dev, d, es)
    return passes * bc / (2.0 * d)


# -----------------------------------------------------------------------------
# 2. KDA: the intermediate is [B, NT, HV, C, C, K], NT = T/C
# -----------------------------------------------------------------------------
def kda_port_bytes(B: int, H: int, T: int, K: int, C: int, es: int) -> int:
    """[B, NT, HV, C, C, K] with NT = T/C.  Elements = B*H*T*C*K.

    Note the shape: LINEAR in C and in T, not quadratic.  The C^2 that makes this
    tensor frightening is cancelled by NT = T/C.  A sweep over T and a sweep over
    C trace the same line, so one crossover number answers both.
    """
    assert T % C == 0
    return B * H * T * C * K * es


def kda_kernel_bytes(B: int, H: int, T: int, K: int, es: int) -> int:
    """The hand-written Triton kernel's decay rows.

    Read from fla/ops/kda/chunk_intra.py: it splits the C=64 chunk into
    NC = C/BC = 4 sub-chunks of BC=16 and builds ONE reference row per
    sub-chunk, so per (chunk, head) it materialises NC * BC * K = C * K elements
    -- NOT BC * K.  Elements = B*H*T*K, independent of C entirely.
    [READ-FROM-SOURCE, not executed -- no GPU, no triton in this environment]
    """
    return B * H * T * K * es


def kda_subchunk_bytes(B: int, H: int, T: int, K: int, C: int, es: int,
                       BC: int = 16) -> int:
    """The kernel's decay rows counted the way `kda/codegen_diff.py` counts them:
    ONE sub-chunk's worth, per chunk, i.e. B*NT*H*BC*K -- the NC-1 sibling
    sub-chunks are not counted.  This is the pairing that produces the study's
    headline 256x, and it compares a whole chunk against a quarter of a chunk.
    Kept so the test suite can assert the discrepancy rather than delete it."""
    NT = T // C
    assert NT * C == T and C % BC == 0
    return B * NT * H * BC * K * es


# -----------------------------------------------------------------------------
def which_side(nbytes: int, cap: int) -> str:
    if nbytes <= cap:
        return f"fits   ({nbytes/cap:.3f} of cap)"
    return f"OVER   ({cap/nbytes:.3f} of cap)"


def section_model4(d: int, es: int, passes: int) -> None:
    print()
    print("=" * 100)
    print("1b. MODEL 4 -- WHAT THE BYTE ACCOUNTING ACTUALLY SAYS.  OPPOSITE SIGN.")
    print("=" * 100)
    print("""
  Models 1-3 all ask "at what intermediate size does the compiler stop winning".
  That question is malformed, and the reason is visible in the byte counts:

    materialised form:   the [B,H,n,n] score tensor is written and read
                         `passes` times                     -> O(n^2)
    online form:         the score tile is on chip, BUT q and k are re-read once
                         per key block, so                  -> O(n^2 * d / Bc)

  BOTH ARE O(n^2).  They differ in the CONSTANT, not the growth rate.  The
  brief's mechanism -- "the intermediate stops fitting, so the human wins more and
  more" -- requires the two growth rates to differ, and they do not.  Enlarging
  the intermediate enlarges both forms' traffic proportionally and moves the
  RATIO not at all.

  So the ratio is a property of (passes, Bc, d) and of nothing else:

        byte_ratio  =  passes * Bc / (2 * d)                 [causal factors aside]

  and Bc is capped by on-chip capacity, Bc <= C_onchip_per_SM / (d * es).  So
  capacity enters -- but it caps the HUMAN's advantage, and a bigger cache makes
  the HUMAN better, not the compiler.  The sign is the reverse of the brief's.

  Note also what is absent from that expression: L2.  The device-wide cache does
  not enter Model 4 at all.  On a fused CUDA kernel the working set is per-SM, and
  a row-slab is produced and consumed on the SM that owns it, so the device-wide
  total is the wrong denominator.  The brief's "relative to the cache hierarchy"
  is a category error: there are two levels and the thesis names the wrong one.
""")
    hdr = (f"  {'device':<7}{'smem/SM':>10}{'Bc_max':>9}{'byte ratio max':>16}"
           f"{'at Bc=128':>11}{'at Bc=256':>11}{'L2 used?':>11}")
    print(hdr)
    print("  " + "-" * 75)
    for dev in D.DEVICE_SPECS:
        bc = max_key_block(dev, d, es)
        rmax = max_byte_ratio(dev, d, es, passes)
        r128 = passes * min(128, bc) / (2 * d)
        r256 = passes * min(256, bc) / (2 * d)
        print(f"  {dev:<7}{D._fmt(D.onchip_per_sm(dev)):>10}{bc:>9}{rmax:>15.1f}x"
              f"{r128:>10.1f}x{r256:>10.1f}x{'no':>11}")
    print()
    h = max_byte_ratio("H100", d, es, passes)
    t = max_byte_ratio("T4", d, es, passes)
    b = max_byte_ratio("B200", d, es, passes)
    print(f"  PREDICTIONS, and they are about magnitudes, not thresholds:")
    print(f"   1. The human's advantage CEILING is {t:.0f}x on a T4 and {h:.0f}x on an H100,")
    print(f"      at d={d}, bf16/fp32 as configured -- and there is no n at which it is")
    print(f"      larger.  Any measured ratio above ~{h:.0f}x refutes Model 4 and means")
    print(f"      something else is going on.")
    print(f"   2. H100 and B200 predict the SAME ceiling ({h:.1f}x and {b:.1f}x), because")
    print(f"      smem/SM is 228 KiB on both, while their L2 differs by 2.5x.  Model 1")
    print(f"      predicts the opposite.  Two cards, one sweep, and the two models are")
    print(f"      separated by a factor of 2.5x in opposite directions.")
    print(f"   3. Going from fp32 to bf16 ({es} -> {es//2} B/elem) doubles Bc_max and")
    print(f"      therefore doubles the human's ceiling, with the hardware unchanged.")
    print(f"      A dtype flag is not a tuning knob, it is a move on the axis the")
    print(f"      thesis is about.  [PREDICTED]")
    print()
    print("  4. THE `passes` TERM IS THE ONE NOBODY HAS MEASURED.  passes is how many")
    print("     times the materialised form moves the score tensor: 2 if the softmax")
    print("     is single-pass, 3 for a numerically-safe max-then-exp, 4 if the")
    print("     P@V read is not fused with the exp.  Every ratio above is LINEAR in")
    print("     it.  A crossover prediction that does not state its pass count is not")
    print("     a prediction, which is why this is refutation criterion R-E.")
    print()
    print(f"  sensitivity of the ceiling to the two unmeasured quantities:")
    print(f"  {'passes':>8}" + "".join(f"{'d='+str(dd):>9}" for dd in (32, 64, 128, 256)))
    for p in (2, 3, 4):
        print(f"  {p:>8}" + "".join(f"{max_byte_ratio('H100', dd, es, p):>8.1f}x"
                                   for dd in (32, 64, 128, 256)))
    print("  Both axes are unmeasured.  The ceiling moves by 2x along each.")


def section_attention(B: int, H: int, d: int, es: int, causal: bool,
                      passes: int, eff_ratio: float,
                      m_natives=(16, 64, 128)) -> None:
    print()
    print("=" * 100)
    print(f"1. ATTENTION -- intermediate [B,H,n,n], n swept; B={B} H={H} "
          f"head_dim={d} {es} B/elem")
    print(f"   (causal={causal}, {passes} DRAM passes over the score tensor modelled)")
    print("=" * 100)
    print("   Three models of where the compiler stops winning.  They disagree by")
    print("   more than an order of magnitude on the largest card; that disagreement")
    print("   IS the experiment.")
    print()
    hdr = f"{'device':<7}{'L2':>10}{'smem/SM':>10}{'M1 n*':>9}{'M2 n*':>9}{'M3 n*':>9}   {'M2/M1':>8}"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for dev in D.DEVICE_SPECS:
        m1 = model1_crossover_n(dev, B, H, es)
        m2 = model2_crossover_n(dev, B, H, es)
        m3 = model3_crossover_n(dev, B, H, d, es, causal, passes, eff_ratio)
        rows.append((dev, m1, m2, m3))
        print(f"{dev:<7}{D._fmt(D.DEVICE_SPECS[dev]['l2_bytes']):>10}"
              f"{D._fmt(D.onchip_per_sm(dev)):>10}{m1:>9.0f}{m2:>9.0f}{m3:>9.0f}   "
              f"{m2/m1:>7.2f}x")
    print()
    print("   M1 n* : the score matrix equals L2.  Device capacity.")
    print("   M2 n* : the per-SM row-slab can no longer be a full-height tile.")
    print("            C_onchip_per_sm / (M_native * es).")
    print("   M3 n* : the materialised form's DRAM time equals the tiled form's.")
    print("            No cache, no capacity.  DEVICE-INDEPENDENT unless eff_ratio")
    print("            (achieved BW_compiler / achieved BW_human) != 1; here "
          f"{eff_ratio:.2f}.")
    print()
    print("   M3's sensitivity to eff_ratio, which is the real content of Model 3:")
    for er in (0.25, 0.5, 1.0, 2.0, 4.0):
        n3 = model3_crossover_n("H100", B, H, d, es, causal, passes, er)
        print(f"     eff_ratio {er:>5.2f}  ->  M3 n* = {n3:>8.0f}   "
              f"(vs M1 {rows[3][1]:.0f}, M2 {rows[3][2]:.0f})")
    n3_lo = model3_crossover_n("H100", B, H, d, es, causal, passes, 0.25)
    n3_hi = model3_crossover_n("H100", B, H, d, es, causal, passes, 4.0)
    m1_spread = max(r[1] for r in rows) / min(r[1] for r in rows)
    m2_spread = max(r[2] for r in rows) / min(r[2] for r in rows)
    print(f"   A 16x change in eff_ratio moves the predicted crossover by "
          f"{n3_hi/n3_lo:.0f}x in n")
    print(f"   ({n3_lo:.0f} -> {n3_hi:.0f}), against {m1_spread:.1f}x for moving from the")
    print(f"   smallest L2 to the largest under Model 1 and {m2_spread:.1f}x under Model 2.")
    print("   So an UNMEASURED efficiency ratio moves the crossover further than the")
    print("   entire device list does, and the ratio s/capacity does not determine the")
    print("   crossover on its own.  Any version of the thesis that omits the")
    print("   efficiency term is incomplete, not merely imprecise.")
    print()
    print("   M2 with M_native = 16 / 64 / 128 (the sensitivity that matters):")
    hdr2 = f"{'device':<7}" + "".join(f"{'M='+str(m):>10}" for m in m_natives)
    print(hdr2)
    print("-" * len(hdr2))
    for dev in D.DEVICE_SPECS:
        cells = "".join(f"{model2_crossover_n(dev, B, H, es, m):>10.0f}"
                        for m in m_natives)
        print(f"{dev:<7}{cells}")
    print()
    same = [r for r in rows if math.isclose(r[2], 0)]
    h100 = model2_crossover_n("H100", B, H, es)
    b200 = model2_crossover_n("B200", B, H, es)
    l2_ratio = (D.DEVICE_SPECS["B200"]["l2_bytes"] /
                D.DEVICE_SPECS["H100"]["l2_bytes"])
    print(f"   *** PREDICTION (sharp, and it splits the corpus): under Model 2, H100")
    print(f"       and B200 predict the SAME crossover n* -- {h100:.0f} and {b200:.0f} --")
    print(f"       because both have 228 KB of shared memory per SM, even though B200")
    print(f"       has {l2_ratio:.1f}x the L2.  Under Model 1 they differ by the same")
    print(f"       {l2_ratio:.1f}x.  So: run the same attention sweep on an H100 and a")
    print(f"       B200.  Model 1 says the crossover moves {math.sqrt(l2_ratio):.2f}x in n;")
    print(f"       Model 2 says it does not move at all.  One sweep settles it.")
    print(f"       [PREDICTED -- not a result]")


def section_kda() -> None:
    print()
    print("=" * 100)
    print("2. KDA -- intermediate [B,NT,HV,C,C,K], NT = T/C, so elements = B*H*T*C*K")
    print("=" * 100)
    for label, sh in (("BENCHMARKED", KDA_BENCH_SHAPE), ("COST-MODELLED", KDA_COST_SHAPE)):
        B, T, H, K, C, es = sh["B"], sh["T"], sh["H"], sh["K"], sh["C"], sh["es"]
        port = kda_port_bytes(B, H, T, K, C, es)
        kern = kda_kernel_bytes(B, H, T, K, es)
        sub = kda_subchunk_bytes(B, H, T, K, C, es)
        print()
        print(f"  {label} shape  B={B} T={T} H={H} K={K} C={C} es={es}  ({sh['where']})")
        print(f"    port   [B,NT,HV,C,C,K]  {port/GiB:>12.3f} GiB  ({port/1e9:.3f} GB)")
        print(f"    kernel [NC,BC,K]        {kern/MiB:>12.3f} MiB  ({kern/1e6:.3f} MB)")
        print(f"    ratio port/kernel      {port/kern:>12.1f}x   <- and this is exactly C = {C}")
        print(f"    codegen_diff pairing   {sub/MiB:>12.3f} MiB  ratio {port/sub:.1f}x "
              f"<- the study's 256x, cross-scope")
        print()
        print("    per-capacity verdict, port side (ratio = intermediate / capacity):")
        for capname, capfn in (("L2       ", lambda d: D.DEVICE_SPECS[d]["l2_bytes"]),
                               ("on-chip  ", lambda d: D.onchip_aggregate(d)),
                               ("smem/SM  ", lambda d: D.onchip_per_sm(d))):
            cells = []
            for dev in D.DEVICE_SPECS:
                r = port / capfn(dev)
                cells.append(f"{dev}:{r:9.2f}x {'HUMAN' if r > 1 else 'compiler?'}")
            print(f"      {capname} " + "  ".join(cells))
        print()
        print("    the same, kernel side -- this is the side that is already fine:")
        for capname, capfn in (("L2       ", lambda d: D.DEVICE_SPECS[d]["l2_bytes"]),
                               ("on-chip  ", lambda d: D.onchip_aggregate(d))):
            cells = [f"{dev}:{kern/capfn(dev):6.3f}x" for dev in D.DEVICE_SPECS]
            print(f"      {capname} " + "  ".join(cells))
        print()
        kern_worst = max(kern / D.DEVICE_SPECS[d]["l2_bytes"] for d in D.DEVICE_SPECS)
        if kern_worst > 1:
            print(f"    *** AT THIS SHAPE THE KERNEL SIDE IS ALSO OVER L2 "
                  f"({kern_worst:.1f}x on the worst card).")
            print("        Neither side is comfortable here, so this is a bad place to")
            print("        look for a crossover and a bad place to quote a 256x or a")
            print("        13x from.  The benchmarked shape, not this one, is where the")
            print("        two sides actually straddle the threshold.  [MODELLED]")
        else:
            print("    => the crossover is not a claim about a ratio, it is a per-device")
            print("       statement, and it says the SAME operator at the SAME shape is")
            print("       'human wins, obviously' on T4 and 'compiler might win' on B200.")
        print()

    print()
    print("  2a. THE CROSSOVER IN T (chunk size C fixed at 64, dtype fixed)")
    print()
    B, H, K, C, es = (KDA_BENCH_SHAPE["B"], KDA_BENCH_SHAPE["H"], KDA_BENCH_SHAPE["K"],
                      KDA_BENCH_SHAPE["C"], KDA_BENCH_SHAPE["es"])
    per_t = B * H * C * K * es                             # bytes per token, C fixed
    print(f"     port bytes = B*H*T*C*K*es = {per_t/1e6:.3f} MB per token at "
          f"B={B},H={H},C={C},K={K},fp32")
    hdr = f"     {'device':<7}{'capacity':>11}{'T* = cap/per_t':>18}"
    print(hdr)
    print("     " + "-" * (len(hdr) - 5))
    for dev in D.DEVICE_SPECS:
        for capname, cap in (("L2", D.DEVICE_SPECS[dev]["l2_bytes"]),
                             ("on-chip", D.onchip_aggregate(dev))):
            print(f"     {dev:<7}{capname:>11}{cap/per_t:>18.0f}")
    print()
    print(f"     The benchmarked shape is T=512.  Read that column:")
    for dev in D.DEVICE_SPECS:
        cap = D.DEVICE_SPECS[dev]["l2_bytes"]
        tstar = cap / per_t
        verdict = ("HUMAN wins" if 512 > tstar else "compiler may win")
        print(f"       {dev:<6} T*={tstar:7.0f}   T=512 is {512/tstar:5.2f}x that -> {verdict}")
    print()
    print("  2b. THE CROSSOVER IN DTYPE -- the same shape, one flag apart")
    print()
    for label, sh in (("benchmarked", KDA_BENCH_SHAPE), ("cost-modelled", KDA_COST_SHAPE)):
        B, T, H, K, C = sh["B"], sh["T"], sh["H"], sh["K"], sh["C"]
        f32 = kda_port_bytes(B, H, T, K, C, ES_FP32)
        b16 = kda_port_bytes(B, H, T, K, C, ES_BF16)
        print(f"     {label} shape, C={C}: fp32 {f32/MiB:.1f} MiB   bf16 {b16/MiB:.1f} MiB")
        for dev in D.DEVICE_SPECS:
            cap = D.DEVICE_SPECS[dev]["l2_bytes"]
            flip = (f32 > cap) != (b16 > cap)
            mark = "  <== FLIPS THE PREDICTION" if flip else ""
            print(f"       {dev:<6} L2={cap/MiB:6.1f} MiB  fp32 {'OVER' if f32>cap else 'fits'}"
                  f"  bf16 {'OVER' if b16>cap else 'fits'}{mark}")
        print()

    print("  2c. THE CROSSOVER IN C (T fixed at the cost-modelled shape)")
    print()
    B, T, H, K, es = (KDA_COST_SHAPE["B"], KDA_COST_SHAPE["T"], KDA_COST_SHAPE["H"],
                      KDA_COST_SHAPE["K"], KDA_COST_SHAPE["es"])
    per_c = kda_port_bytes(B, H, T, 1, 1, es) * 1   # placeholder, computed below
    per_c = B * H * T * K * es                      # bytes per unit C
    print(f"     port bytes = B*H*T*K*es*C = {per_c/MiB:.0f} MiB per unit of C at "
          f"B={B},T={T},H={H},K={K},bf16")
    hdr = f"     {'device':<7}{'C* = cap/per_c':>16}{'C=32':>10}{'C=64':>10}{'C=128':>10}"
    print(hdr)
    print("     " + "-" * (len(hdr) - 5))
    for dev in D.DEVICE_SPECS:
        cap = D.DEVICE_SPECS[dev]["l2_bytes"]
        cstar = cap / per_c
        cells = "".join(f"{per_c*c/MiB:9.0f}M" for c in (32, 64, 128))
        print(f"     {dev:<7}{cstar:>16.2f}{cells}")
    print()
    print("     C* < 32 everywhere, and fla's chunk_kda raises unless C in {32,64},")
    print("     so at this shape the port is over every device's L2 at every LEGAL")
    print("     chunk size: there is no crossover in C, the human wins by construction")
    print("     at 4 GiB of avoidable traffic.  [PREDICTED]")
    print()
    print("     BUT the cost-modelled shape is NOT the benchmarked shape.  See §3.")


def section_scope() -> None:
    print()
    print("=" * 100)
    print("3. THE 256x IS CROSS-SCOPE.  THE HONEST RATIO IS 64x.")
    print("=" * 100)
    B, T, H, K, C, es = (KDA_BENCH_SHAPE["B"], KDA_BENCH_SHAPE["T"],
                         KDA_BENCH_SHAPE["H"], KDA_BENCH_SHAPE["K"],
                         KDA_BENCH_SHAPE["C"], KDA_BENCH_SHAPE["es"])
    BC = 16
    NC = C // BC
    NT = T // C
    port_elems = B * NT * H * C * C * K
    kern_whole_chunk_elems = B * NT * H * NC * BC * K          # all NC sub-chunks
    kern_one_subchunk_elems = B * NT * H * BC * K              # one sub-chunk only
    print(f"  C={C}, BC={BC}, NC={NC}, NT={NT}, B={B}, H={H}, K={K}")
    print(f"  port,   per (chunk,head): C^2*K          = {C*C*K:>8,}   "
          f"whole op {port_elems:>12,}")
    print(f"  kernel, per (chunk,head): NC*BC*K = C*K   = {NC*BC*K:>8,}   "
          f"whole op {kern_whole_chunk_elems:>12,}")
    print(f"  kernel, ONE sub-chunk:   BC*K            = {BC*K:>8,}   "
          f"whole op {kern_one_subchunk_elems:>12,}")
    print()
    print(f"  whole-operator ratio, both sides counted over a whole chunk: "
          f"{port_elems/kern_whole_chunk_elems:.1f}x   (= C)")
    print(f"  ratio if the kernel side is counted per SUB-chunk:            "
          f"{port_elems/kern_one_subchunk_elems:.1f}x   (= C*C/BC)")
    print()
    print("  The second number is the study's 256x.  It is obtained by comparing the")
    print("  port's whole chunk against the kernel's ONE OF FOUR sub-chunks -- a")
    print("  cross-scope comparison.  notes/06 C-03 correctly records that 256x is a")
    print("  slab ratio and not a byte or time ratio, and it correctly records that")
    print("  '256x' has two incompatible derivations in the record.  What it does not")
    print("  record is that the ratio itself is comparing two different scopes, so")
    print("  256 is not 'the slab ratio' -- it is the slab ratio with a factor of NC")
    print("  thrown away.  At the same scope it is 64x, and 64x is what")
    print("  notes/03-kda.md already said ('16,777,216 / 262,144 = 64').")
    print()
    print("  [MODELLED] -- exact arithmetic on shapes, zero measurement.  The BC=16")
    print("  and NC=4 on the kernel side are [READ-FROM-SOURCE] from")
    print("  fla/ops/kda/chunk_intra.py and have not been executed.")
    print("  Asserted in crossover/test_control.py so it cannot be lost again.")


def section_refutation() -> None:
    print()
    print("=" * 100)
    print("4. PRE-REGISTERED REFUTATION CRITERIA -- WRITTEN BEFORE ANY DATA")
    print("=" * 100)
    print("""
  R-A  THE STEP CRITERION (this is the one that tests P3 directly).
       Measure the human/compiler time ratio R(n) for n on a geometric grid
       spanning a factor of 64 on either side of the predicted crossover, at a
       fixed B, H, d and dtype, on ONE device.
         STEP  supported if R(n) is flat (within a factor of 1.3) over a decade
               of n below the predicted crossover, and rises monotonically
               through it, reaching >= 2x within one doubling of n past it.
         STEP  REFUTED if R(n) is already varying by more than a factor of 2
               across a decade of n that lies ENTIRELY on one side of the
               predicted crossover -- i.e. if the transition is a slope, or if
               there is no transition.
       A gradual transition is a refutation.  It is the expected outcome, see
       notes/07 §3, and it is written down here so that the finding cannot be
       re-described as a step after the fact.

  R-B  THE MODEL-DISCRIMINATION CRITERION (tests P2).
       Same sweep on H100 and on B200.
         MODEL 1 supported if the crossover n* differs by ~sqrt(126/50) = 1.59x.
         MODEL 2 supported if the crossover n* is the same on both cards.
       This is one sweep run twice and it settles which capacity matters.

  R-C  THE NATURAL-LOWERING CRITERION (tests the load-bearing premise, and is
       the one most likely to fail).
       For the crossover to be about anything, the materialising
       implementation must be the one a compiler would actually produce from a
       reasonable source program.  Test: write the operator in the obvious
       tensor form, compile it, and count the bytes that reach DRAM.
         holds if dram__bytes.sum is within ~1.3x of the modelled materialised
              traffic, i.e. the compiler really did let the intermediate out;
         REFUTED if the compiler fused it away at every n, in which case there is
              no crossover to find on that device at that shape, because the
              "compiler side" does not exist.
       This is the confound the brief asks to be stated, promoted to a test.

  R-D  THE EQUALITY CRITERION (tests "the only thing changing is whether the
       intermediate is materialised").
       attn_control.py's two implementations must agree numerically to within
       fp32 accumulation error, and the avoiding version must be invariant to
       the block size.  If either fails, the control is not a control and every
       timing from it is unattributable.
       Runs today, on CPU, and is the only criterion in this list that is
       already satisfiable.

  R-E  THE THROUGHPUT-PASSES CRITERION (the weakest link, stated so it is not
       hidden).
       The number of DRAM passes the materialised form makes over the score
       tensor (2? 3? 4?) is a MODELLING CHOICE, not a measurement, and R(n) is
       linear in it.  It must be measured -- dram__bytes.sum -- before any
       predicted ratio is quoted.  A crossover prediction that does not state
       its pass count is not a prediction.
""")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--H", type=int, default=1)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--es", type=int, default=ES_FP32)
    ap.add_argument("--passes", type=int, default=3,
                    help="DRAM passes over the materialised score tensor "
                         "(MODELLED, not measured -- see criterion R-E)")
    ap.add_argument("--eff-ratio", type=float, default=1.0,
                    help="Model 3: achieved BW_compiler / achieved BW_human")
    ap.add_argument("--no-causal", action="store_true")
    args = ap.parse_args()

    print("=" * 100)
    print("CROSSOVER CLAIM -- PREDICTIONS.  Nothing in this file is a measurement.")
    print("=" * 100)
    print(f"  attention shape: B={args.B} H={args.H} d={args.d} es={args.es} "
          f"causal={not args.no_causal} passes={args.passes}")
    print(f"  M_native (CUDA) = {M_NATIVE_CUDA}")
    print()
    print("  [MEASURED]   nothing in this file")
    print("  [MODELLED]   all capacities, all byte counts, all crossover points")
    print("  [PREDICTED]  every 'which side wins' and every n* / T* / C*")
    print("  [READ-FROM-SOURCE]  the BC=16, NC=4 kernel tiling; the device table")
    print("  [UNVERIFIED]  TPU VMEM capacity; see devices.UNVERIFIED")
    section_attention(args.B, args.H, args.d, args.es, not args.no_causal,
                      args.passes, args.eff_ratio)
    section_model4(args.d, args.es, args.passes)
    section_kda()
    section_scope()
    section_refutation()
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
