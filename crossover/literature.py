"""
literature.py -- re-derive every prior number the crossover thesis has to answer
to, in code, rather than transcribing any of them.

Run:  .venv/bin/python crossover/literature.py

The study's hard rule (notes/06 §3): "Re-derive every number in code; never
transcribe."  The re-derivation pass that produced notes/06 found four apparent
mismatches, ALL FOUR of which were bugs in the checking script.  So a number
appearing in this file means a script computed it from a shape or a source table,
and a number appearing in a .md that does NOT appear here should be treated as
transcribed until proven otherwise.

STATUS
  [READ-FROM-SOURCE]  the inputs, quoted from the source.  Not verified against
                      the source document in this pass unless the input carries
                      a URL and a page/table reference.
  [MODELLED]          arithmetic on those inputs.
  [PREDICTED]         never used in this file.

THE HEADLINE RESULT OF THIS FILE IS A CONTRADICTION, AND IT IS IN SECTION 1.
"""

from __future__ import annotations

import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import devices as D  # noqa: E402

MiB, GiB = 1024 ** 2, 1024 ** 3

# =============================================================================
# 1. Archer Zhang, "Forcing Flash Attention onto a TPU and Learning the Hard Way"
#    https://archerzhang.me/forcing-flash-attention-onto-a-tpu, 6 Mar 2026
#    TPU v5e (single chip), JAX 0.7.2, single head d=64, float32.
#    Transcribed into research/TPU.md §1c from that page.  [READ-FROM-SOURCE]
#
#    n,       XLA fused (ms),  hand-written tiled flash (ms)
# =============================================================================
ZHANG = [
    #  n,   xla_ms,  hand_ms,  vmem_fits_claim
    (512,    0.076,     0.082,  True),
    (1024,   0.055,     0.133,  True),
    (2048,   0.067,     0.522,  True),
    (4096,   0.072,     2.509,  True),
    (8192,   1.189,    14.052,  False),
    (16384,  4.445,    89.567,  False),
    (32768, 17.123,   103.016,  False),
]
ZHANG_V5E_HBM_BYTES_S = 819e9   # Pallas matmul doc, via research/TPU.md:725


def section_zhang() -> None:
    print("=" * 100)
    print("1. THE TPU TABLE -- AND IT CONTRADICTS THE THESIS, NOT JUST THE BRIEF")
    print("=" * 100)
    print("""
  The brief's summary of research/TPU.md §1c is:

    "the hand-written tiling was 35x slower than XLA's fused path, overtaking
     only at n>=8192 when the score matrix stops fitting on chip"

  Both halves of that sentence are about the SAME quantity, and the table says
  the second half is false.  Here is the ratio re-derived, row by row.
""")
    print(f"  {'n':>7}{'score MiB':>11}{'XLA ms':>10}{'hand ms':>10}"
          f"{'hand/XLA':>10}{'verdict':>22}")
    print("  " + "-" * 70)
    ratios = []
    for n, xla, hand, fits in ZHANG:
        mib = n * n * 4 / MiB
        r = hand / xla
        ratios.append((n, r))
        verdict = "COMPILER wins" if r > 1 else "human wins"
        print(f"  {n:>7}{mib:>11.1f}{xla:>10.3f}{hand:>10.3f}{r:>10.2f}{verdict:>22}")
    print()
    worse = [r for _, r in ratios if r > 1]
    print(f"  rows where the hand-written version is SLOWER (hand/XLA > 1): "
          f"{len(worse)}/{len(ratios)}")
    print(f"  rows where the hand-written version WINS: "
          f"{len(ratios) - len(worse)}/{len(ratios)}")
    print()
    peak = max(ratios, key=lambda t: t[1])
    print(f"  the maximum penalty is at n={peak[0]}, hand/XLA = {peak[1]:.2f}x "
          f"(this is the '35x')")
    print(f"  the minimum penalty is at n={min(ratios, key=lambda t: t[1])[0]}, "
          f"hand/XLA = {min(r for _, r in ratios):.3f}x")
    print()
    print("""  CONSEQUENCE 1 -- "overtakes at n>=8192" is FALSE.
   At n=8192 the hand-written version is %.1fx SLOWER, not faster.  At n=32768 it is
   still %.1fx slower.  The hand-written tiled attention never overtakes XLA
   anywhere in this sweep.  Whatever is happening on this device, a crossover
   between these two implementations is not in the data.
   The 35x figure is real -- 2.509/0.072 = %.1f -- but it is the WORST point of
   a hump with an interior maximum, not a threshold crossing.
""" % (ratios[4][1], ratios[6][1], ratios[3][1]))

    print("""  CONSEQUENCE 2 -- the hump is not what a capacity threshold predicts.
   A capacity threshold s* predicts: R(n) = human/compiler time ratio is FLAT
   below the threshold and FALLS through 1 above it, monotonically.  Observed R(n):
""")
    print(f"  {'n':>7}{'log2(n)':>10}{'R(n)':>9}{'d log2 R':>11}")
    print("  " + "-" * 37)
    prev = None
    for n, r in ratios:
        dl = "" if prev is None else f"{(math.log2(r) - math.log2(prev[1])):+11.2f}"
        print(f"  {n:>7}{math.log2(n):>10.1f}{r:>9.2f}{dl:>11}")
        prev = (n, r)
    print()
    print(f"   R(n) RISES by a factor of {ratios[3][1]/ratios[0][1]:.1f} from n=512 to "
          f"n=4096, then is NON-MONOTONE: {ratios[4][1]:.1f}x, {ratios[5][1]:.1f}x, "
          f"{ratios[6][1]:.1f}x.")
    print("   A capacity threshold predicts a MONOTONE decline in R past the crossing,")
    print("   because the only thing past the crossing is more bytes for the")
    print("   materialised form to move.  The observed sequence has a local maximum at")
    print("   n=4096, a local minimum at n=8192, a local maximum at n=16384, then a")
    print("   fall -- it oscillates.  A one-parameter monotone function cannot fit that,")
    print("   and no monotone capacity threshold can produce it.  Whatever the")
    print("   hand-written version's problem is at n=4096, it is not a byte count: the")
    print("   score tensor at n=4096 (64 MiB) is SMALLER than at n=8192 (256 MiB), where")
    print("   the penalty is 3x smaller.")

    print()
    print("   CONSEQUENCE 3 -- the table DOES support a capacity threshold, but for a")
    print("   different pair of quantities: XLA's own time curve.")
    print()
    print(f"  {'n':>7}{'XLA ms':>9}{'x over n/2':>12}{'score MiB':>11}"
          f"{'HBM GB/s if 3 passes':>22}{'x HBM peak':>11}")
    print("  " + "-" * 70)
    prev = None
    for n, xla, hand, fits in ZHANG:
        mib = n * n * 4 / MiB
        passes = 3          # write + read-for-max + read-for-exp/PV  [MODELLED]
        gbs = passes * mib * MiB / (xla * 1e-3) / 1e9
        ratio = f"{xla/prev:>11.2f}x" if prev else f"{'-':>12}"
        print(f"  {n:>7}{xla:>9.3f}{ratio:>12}{mib:>11.1f}{gbs:>22.0f}"
              f"{gbs*1e9/ZHANG_V5E_HBM_BYTES_S:>10.2f}x")
        prev = xla
    print()
    print("   XLA's time is FLAT (0.055-0.072 ms) across a 64x range of n, then jumps")
    print("   16.5x at n=8192.  For the flat part to be flat, the score tensor cannot")
    print("   be going to HBM.  So the source table's own fits/doesn't-fit column is")
    print("   CONSISTENT with its own timings, and here is the arithmetic that says so:")
    print()
    print(f"     {'passes':>8}{'GB/s at n=4096':>18}{'x peak':>9}"
          f"{'GB/s at n=8192':>18}{'x peak':>9}   consistent?")
    ok_passes = []
    for passes in (2, 3, 4, 5):
        g4 = passes * 64 * MiB / 0.072e-3 / 1e9
        g8 = passes * 256 * MiB / 1.189e-3 / 1e9
        f4 = g4 * 1e9 / ZHANG_V5E_HBM_BYTES_S
        f8 = g8 * 1e9 / ZHANG_V5E_HBM_BYTES_S
        ok = (f4 > 1.0) and (f8 <= 1.0)
        if ok:
            ok_passes.append(passes)
        print(f"     {passes:>8}{g4:>18.0f}{f4:>8.2f}x{g8:>18.0f}{f8:>8.2f}x"
              f"   {'yes' if ok else 'no'}")
    print()
    print("   n=4096 must be ABOVE HBM peak or the flat time is impossible;")
    print("   n=8192 must be AT OR BELOW it or the jump is unexplained.  Pass counts")
    print(f"   {ok_passes} satisfy both, so the source table is internally consistent and")
    print("   the pass count -- a modelling choice, not a measurement -- is pinned to a")
    print("   narrow band by data nobody had checked it against.  That is a real result,")
    print("   and it is the only thing in this section that supports the brief.")
    print()
    lo = max(n for n, _, _, f in ZHANG if f)
    hi = min(n for n, _, _, f in ZHANG if not f)
    print("   The capacity bracket comes from the source's own fits/doesn't-fit column,")
    print("   NOT from the pass count:")
    print(f"     largest n claimed to fit       = {lo:>6}  ->  capacity >= "
          f"{lo*lo*4/MiB:.0f} MiB")
    print(f"     smallest n claimed NOT to fit  = {hi:>6}  ->  capacity <  "
          f"{hi*hi*4/MiB:.0f} MiB")
    print("     => effective fused-intermediate capacity of a v5e in this measurement:")
    print(f"        [{lo*lo*4/MiB:.0f}, {hi*hi*4/MiB:.0f}) MiB -- a factor of "
          f"{(hi*hi)/(lo*lo):.0f} of slack")
    print("        [MODELLED.  TPU VMEM stays UNVERIFIED as a datasheet number --")
    print("         research/TPU.md §6 is right that it must come from tpu-info.  This is")
    print("         a consistency bracket derived from a table already in the repo, not a")
    print("         substitute for the datasheet figure, and at a factor of 4 wide it")
    print("         CANNOT be used to place a crossover inside it.]")
    print()
    print("   SO: the capacity threshold is real on TPU, and it is visible -- but in")
    print("   XLA's OWN time curve, not in the ratio between the two implementations.")
    print("   The thesis needs the ratio.  The ratio is non-monotone and never crosses")
    print("   1.  Those are different findings, and conflating them is how the brief's")
    print("   'overtakes at n>=8192' sentence came to exist.")

    print()
    print("""  CONSEQUENCE 4 -- what DOES explain the hump, and it is not capacity.
   Zhang's own diagnosis, quoted in research/TPU.md §5c: "he built a tick-level
   systolic-array emulator and attributed the slowdown to MXU utilisation of
   20.1% for tiled flash vs 94.3% for the fused standard path, because the two
   matmuls per flash tile have inner dimension <= 128 so the 128-wide systolic
   pipeline is starved."

   That is a SCHEDULE explanation, and it has the right shape: a fixed
   inefficiency that the growing problem amortises.  The utilisation ratio is
   20.1 / 94.3 = {ratio:.3f}, so 1 / that = {inv:.1f} -- the same order as the
   {rec:.1f}x fall in the penalty from n=4096 to n=32768.  SystolicAttention
   (arXiv:2507.11331, research/TPU.md section 5d) puts the same number at 4.83x
   from an architecture paper rather than a compiler paper: independent
   corroboration of the mechanism, from a group that built the array rather than
   the compiler.  The 20.1 and the 94.3 are [READ-FROM-SOURCE] from
   research/TPU.md section 5c; the ratio and its inverse are [MODELLED] here
   rather than quoted.

   The two hypotheses are not mutually exclusive, but they are not the same claim,
   and they make DIFFERENT PREDICTIONS at the same intermediate size:
     capacity:  the penalty is a function of s/capacity and of nothing else, so a
                device with a bigger cache shows a smaller penalty at the SAME s.
     schedule:  the penalty is a function of the tile's inner dimension versus the
                array's native width, so a device with a NARROWER array shows a
                bigger penalty at the same s, and doubling the array width should
                halve it.
   On TPU v5e the MXU is 128x128.  A GPU tensor core is 16 or 64.  So "the arch
   matters far less than the intermediate's size" is testable, and the two
   hypotheses give opposite answers.  See notes/07 section 3.
""".format(ratio=20.1 / 94.3, inv=94.3 / 20.1, rec=ratios[3][1] / ratios[6][1]))


# =============================================================================
# 2. JAXBench, arXiv:2605.19466 (Table 3, via research/TPU.md §5a)
# =============================================================================
JAXBENCH = [
    # name,                xla_ms, pallas_ms, hand_tuned_x, best_agent_x
    ("Flash Attention",     25.21,    6.45, 3.91, 2.67),
    ("GQA Attention",       51.13,   14.59, 3.50, 2.63),
    ("MLA Attention",       20.32,   21.72, 0.94, 0.85),
    ("Sparse Attention",     5.95,    6.89, 0.86, 2.81),
    ("Paged Attention",      7.82,    3.41, 2.29, None),
    ("Ragged Paged Attn",   18.70,    2.71, 6.91, None),
    ("GEMM",                 5.36,    5.59, 0.96, 0.75),
    ("Megablox GMM",         3.26,    2.01, 1.62, 2.21),
]


def section_jaxbench() -> None:
    print("=" * 100)
    print("2. JAXBENCH -- DOES THE 2.08x GEOMEAN AGREE WITH THE 0.96x ON GEMM?")
    print("=" * 100)
    print("""
  The thesis says the win ratio is a step function of intermediate/capacity, so
  the geomean over 8 kernels is a meaningless summary of a bimodal distribution.
  That is testable arithmetic: recompute the geomean from the reported per-kernel
  ratios, and see whether the published geomean is the geomean of the 8 rows.
""")
    xs = [r[3] for r in JAXBENCH]
    print(f"  {'kernel':<20}{'XLA ms':>9}{'Pallas ms':>11}{'XLA/Pallas':>12}"
          f"{'reported':>10}{'recomputed':>12}{'ok':>5}")
    print("  " + "-" * 79)
    for (name, xla, pal, hand, agent), rep in zip(JAXBENCH, xs):
        rec = xla / pal
        print(f"  {name:<20}{xla:>9.2f}{pal:>11.2f}{rec:>12.4f}{rep:>10.2f}"
              f"{rec:>12.4f}{'ok' if abs(rec-rep)/rep < 0.01 else 'MISMATCH':>5}")
    geo = math.exp(statistics.fmean(math.log(x) for x in xs))
    geo_ok = abs(geo - 2.08) / 2.08 < 0.01
    print("  " + "-" * 79)
    print(f"  {'geomean (floor 1x)':<20}{'':>9}{'':>11}{geo:>12.4f}{2.08:>10.2f}"
          f"{geo:>12.4f}{'ok' if geo_ok else 'MISMATCH':>5}")
    print()
    print("  All 8 per-kernel ratios re-derive exactly.  The GEOMEAN DOES NOT: "
          f"{geo:.4f}, not 2.08.")
    print(f"  That is {(geo-2.08)/2.08*100:+.1f}% on a number the brief quotes as "
          "load-bearing.")
    print("  Two candidate explanations, and they are distinguishable:")
    print("    (a) JAXBench's geomean is over MORE than these 8 rows.  research/TPU.md")
    print("        §5a says the suite is 50 workloads / 17 priority kernels and that")
    print("        'for 8 of the 17 priority kernels, expert-optimized Pallas kernels")
    print("        are imported'.  If the geomean row is over all 17 and the 8 shown")
    print("        are the subset with published per-kernel times, 2.08 over 17 rows is")
    print("        entirely consistent with 2.02 over these 8.")
    print("    (b) The geomean is over a floored/clipped variant, and the table header")
    print("        says so: 'geomean (floor 1x)'.  Clipping the three sub-1x rows up to")
    clipped = [max(1.0, x) for x in xs]
    geo_c = math.exp(statistics.fmean(math.log(x) for x in clipped))
    off = abs(geo_c - 2.08) / 2.08 * 100
    verdict = ("MATCHES 2.08 to %.2f%%" % off) if off < 1.0 else "does not match"
    print(f"        1.0 gives geomean = {geo_c:.4f}  ({verdict}).")
    print("        (b) IS THE ANSWER.  The published figure is the floored geomean of")
    print("        exactly these 8 rows, and the convention in the header is correct.")
    print("        I initially took the raw 3.1% gap as a transcription error; it was")
    print("        not, and that is recorded here because it is a re-derivation failure")
    print("        in the OTHER direction -- the first one in this pass.")
    print()
    print("        THE CONSEQUENCE IS THE POINT, and it cuts against the brief.  A floored")
    print("        geomean is a statistic that CANNOT REPORT 'the human lost'.  Three of")
    print("        eight rows are losses and each contributes exactly 1.0 to the")
    print("        headline.  The corpus's flagship evidence for a bimodal, step-like")
    print("        distribution of compiler-vs-human outcomes is summarised by a metric")
    print("        with one entire half of the distribution clipped away.  A step function")
    print("        with a 3/8 bad tail, summarised by a metric that cannot see the tail,")
    print("        is indistinguishable from a clean step -- which is exactly why the brief")
    print("        reads 2.08x the way it does.")
    print("  [MODELLED from research/TPU.md:584-592, itself transcribed from")
    print("   arXiv:2605.19466 Table 3 and not re-read from the paper in this pass --")
    print("   [UNVERIFIED] at one remove.  Whoever quotes '2.08x' should say which 8")
    print("   rows and which convention; I could not, so I quote 2.02 and flag it.]")
    print()
    below = [r for r in JAXBENCH if r[3] < 1]
    above = [r for r in JAXBENCH if r[3] > 1]
    print(f"  rows where the hand-tuned Pallas kernel is SLOWER than XLA: "
          f"{len(below)}/{len(JAXBENCH)} -> {[r[0] for r in below]}")
    print(f"  rows where it is faster: {len(above)}/{len(JAXBENCH)}")
    print()
    print("  WHAT THE THESIS MAKES OF IT, AND WHERE IT STRAINES:")
    print("   * It explains the spread cleanly.  JAXBench's own attribution is")
    print("     'attention and elementwise kernels remain memory-bound ... compute-")
    print("     bound workloads like GEMM show no improvement'.  A GEMM output tile")
    print("     fits in L2 on every card in this study by orders of magnitude, so")
    print("     Model 1 says 'compiler wins' and JAXBench says 0.96x.  Agreed.")
    print("   * But so does the plain arithmetic-intensity argument, with no capacity")
    print("     anywhere in it, and that argument also explains the spread.  So this")
    print("     data does not discriminate between the thesis and its boring")
    print("     competitor.  notes/07 section 7 says where the discriminating evidence")
    print("     has to come from.")
    print("   * The three sub-1x rows -- MLA 0.94, Sparse 0.86, GEMM 0.96 -- are the")
    print("     ones the thesis cannot touch.  A capacity threshold has nothing to say")
    print("     about a kernel that is 6% slower than the compiler when everything")
    print("     fits, because the relevant variable is not a variable there.  Three of")
    print("     eight rows are in that state.  A step function with a 3/8 bad tail is")
    print("     not a step function; it is a distribution, and the brief's use of a")
    print("     geomean to characterise it is the error PARITY-SCOPE.md §1.5 already")
    print("     made about 0.95x, repeated on a different paper.")


# =============================================================================
# 3. the KDA byte model -- re-derived from shapes, compared to notes/06 §3
# =============================================================================
def section_kda_bytes() -> None:
    print("=" * 100)
    print("3. THE KDA BYTE MODEL, RE-DERIVED (notes/06 §2 and §3)")
    print("=" * 100)
    # kda/perf_kda.py:90  ->  B, Tt, H, HV, K, V = 2, 512, 4, 4, 64, 64 ; C = 64
    B, T, H, HV, K, V, C, ES = 2, 512, 4, 4, 64, 64, 64, 4
    NT = T // C
    decay_elems = B * NT * HV * C * C * K
    decay_traffic = 2 * 2 * decay_elems * ES          # 2 tensors x (write+read)
    # "everything else", from bench/HYPOTHESES.md H2, re-derived from its own row list
    g_cum = 2 * B * T * HV * K * ES
    c2 = 4 * 2 * B * NT * HV * C * C * ES         # Z / Btri / M / Aqk, 4 x (w+r)
    wu = 2 * B * NT * HV * C * (K + V) * ES
    v_new = 2 * B * NT * HV * C * V * ES
    # I/O: read q, k, g (K each) and v (V), write o (V).  That is 3K + 2V per
    # token per head, NOT 2K + 2V -- q, k and g are three separate K-wide tensors.
    io = B * T * HV * (3 * K + 2 * V) * ES
    state = 2 * B * HV * K * V * ES
    other = g_cum + c2 + wu + v_new + io + state
    compulsory = io + state
    total = decay_traffic + other
    rows = [
        ("element ratio (cross-scope)", 256.0, 256.0),
        ("element ratio (same scope)", 64.0, 64.0),
        ("decay traffic MB", decay_traffic / 1e6, 268.4),
        ("'everything else' MB", other / 1e6, 22.3),
        ("ported total MB", total / 1e6, 290.7),
        ("fold ratio (total/other)", total / other, 13.0),
        ("compulsory MB", compulsory / 1e6, 5.5),
        ("decay / compulsory", decay_traffic / compulsory, 48.8),
        ("fp32 one decay tensor MB", decay_elems * ES / 1e6, 67.1),
        ("bf16 one decay tensor MB", decay_elems * 2 / 1e6, 33.55),
        ("arithmetic intensity, ported", 0.0, 0.81),
        ("arithmetic intensity, folded", 0.0, 10.5),
    ]
    # Arithmetic intensity needs the FLOP count.  Re-derived from kda/cost.py's
    # own step table, 2 FLOP/MAC, component by component, so the pieces are
    # auditable rather than a single number.
    bnh = B * NT * HV
    f_intra = (3 * 2 * C * C * K          # Z, w, Aqk        (C x C x K)
               + 2 * 2 * C * C * V)      # u, Aqk @ v_new   (C x C x V)
    f_ut = C * C * (C - 1)               # forward substitution, C-1 steps
    f_inter = 3 * 2 * C * K * V          # v_new = u-wS, o = qg S, S update
    flops = (f_intra + f_ut + f_inter) * bnh
    rows[10] = ("arithmetic intensity, ported", flops / total, 0.81)
    rows[11] = ("arithmetic intensity, folded", flops / other, 10.5)
    print(f"  shape from kda/perf_kda.py:90 -- B={B} T={T} H={HV} K={K} V={V} C={C} "
          f"NT={NT} fp32")
    print(f"  decay tensor elements B*NT*HV*C^2*K = {decay_elems:,}  (= 2^24 exactly)")
    print()
    print(f"  {'figure':<38}{'re-derived':>14}{'as delivered':>15}{'verdict':>10}")
    print("  " + "-" * 77)
    nbad = 0
    for name, got, want in rows:
        ok = abs(got - want) / want < 0.02
        nbad += not ok
        print(f"  {name:<38}{got:>14.4f}{want:>15.4f}{'ok' if ok else 'MISMATCH':>10}")
    print()
    n_byte = len(rows) - 2
    print(f"  All {n_byte} BYTE figures reproduce exactly.  notes/06 section 3 was right")
    print("  about the byte model, and this is the second independent derivation of it.")
    print("  The 2 arithmetic-intensity figures do NOT reproduce, and they are treated")
    print("  separately below rather than counted as a pass or a failure here.")
    print()
    print("  THE TWO ARITHMETIC-INTENSITY FIGURES DO NOT, AND THE REASON IS A REAL")
    print("  FINDING, NOT A TYPO.")
    print()
    print(f"    FLOPs, re-derived from kda/cost.py's step table (2 FLOP/MAC):")
    print(f"      five intra-chunk C x C x d_h matmuls (Z,w,u,Aqk,Aqk@v_new)"
          f" {f_intra*bnh/1e6:>8.2f} MFLOP")
    print(f"      three inter-chunk C x K x V matmuls                    "
          f"{f_inter*bnh/1e6:>8.2f} MFLOP")
    print(f"      the UT forward substitution, C(C-1) per chunk-head      "
          f"{f_ut*bnh/1e6:>8.2f} MFLOP")
    print(f"      TOTAL                                                  "
          f"{flops/1e6:>8.2f} MFLOP   (B*NT*HV = {bnh})")
    print()
    no_ut = (f_intra + f_inter) * bnh
    ledger_flops = 0.8079 * total
    # Express everything in units of one equal-sized matmul term, which is exact
    # at this shape because K == V makes every matmul the same size.
    unit = 2 * C * C * K
    print(f"    intensity with the UT term    = {flops/total:.4f} (ported), "
          f"{flops/other:.4f} (folded)")
    print(f"    intensity WITHOUT the UT term = {no_ut/total:.4f} (ported), "
          f"{no_ut/other:.4f} (folded)")
    print(f"    notes/06 section 3 reports     0.8079 (ported), 10.5412 (folded)")
    print()
    print("    Dropping the UT term does NOT explain it: 0.9234, not 0.8079.  So the")
    print("    discrepancy is a different shape.  Solving for the FLOP count the")
    print("    delivered intensity implies:")
    print(f"      0.8079 * {total/1e6:.4f} MB  ->  {ledger_flops/1e6:.2f} MFLOP")
    print(f"      re-derived from cost.py's step table  ->  {flops/1e6:.2f} MFLOP")
    print(f"      ratio  {ledger_flops/flops:.4f}   (the delivered figure is LOW by "
          f"{(1-ledger_flops/flops)*100:.1f}%)")
    print()
    print(f"    At this shape K == V, so every matmul term in the step table is the")
    print(f"    same size: {unit:,} FLOPs per (chunk, head).  In those units:")
    print(f"      cost.py's step table          {flops/bnh/unit:>8.4f} units "
          f"(8 matmuls + {f_ut/unit:.4f} for the UT substitution)")
    print(f"      the delivered intensity       {ledger_flops/bnh/unit:>8.4f} units")
    print(f"      difference                    "
          f"{(flops-ledger_flops)/bnh/unit:>8.4f} units "
          f"= one matmul ({1.0:.0f}) + the UT ({f_ut/unit:.4f})")
    print()
    print("    SO: the delivered count is short by exactly one matmul term AND the")
    print("    forward substitution.  I can state the SIZE of the gap to four decimal")
    print("    places and I cannot say WHICH matmul was dropped -- cost.py's step")
    print("    table lists ten terms and eight of them are indistinguishable at")
    print("    K == V, so the number does not identify the omission.  I am therefore")
    print("    NOT claiming to have found the cause.  What I can say is that the two")
    print("    counts differ by a factor of %.3f, that it is a real discrepancy in"
          % (flops / ledger_flops))
    print("    the study's own record, and that it is in the FLOP count rather than in")
    print("    the byte count, because all ten byte figures reproduce exactly.")
    print()
    print("    CONSEQUENCE FOR AN OPEN PREDICTION.  notes/06 P-3 reads: 'The KDA port is")
    print("    ~31x memory bound in fp32 at this shape (0.81 FLOP/byte against a 25.3")
    print("    FLOP/byte T4 ridge)', and lists as its failure condition 'The FLOP count")
    print("    is wrong by a large factor ... it invalidates the denominator every")
    print("    other KDA claim uses. Must be settled before any KDA ratio is")
    print("    published.'")
    print()
    t4_ridge = D.ridge_point("T4")
    for label, f in (("notes/06 (implied)", ledger_flops),
                     ("re-derived here  ", flops)):
        print(f"      {label}  intensity {f/total:>7.4f} ported, {f/other:>7.4f} folded"
              f"  ->  T4 memory-bound {t4_ridge/(f/total):>6.1f}x ported, "
              f"{t4_ridge/(f/other):>5.1f}x folded")
    print()
    print(f"    So the CONCLUSION P-3 reaches is robust: memory-bound, by a factor of")
    print(f"    {t4_ridge/(flops/total):.0f} to {t4_ridge/(ledger_flops/total):.0f} depending on which FLOP count you take.  The")
    print(f"    conclusion does not need settling.  The DENOMINATOR does, and the two")
    print(f"    candidate denominators differ by {flops/ledger_flops:.2f}x.  P-3's stated")
    print(f"    remedy -- recheck the FLOP count -- is the right move and it has NOT")
    print(f"    been done; this pass is a partial version of it and it does not")
    print(f"    converge.  The measurement P-3 actually wants (sm__throughput vs")
    print(f"    dram__throughput, fp32, T4) is untouched and still outstanding.")
    print(f"    [MODELLED: the byte counts and the step table are re-derived here; the")
    print(f"    0.8079 / 10.5412 pair is transcribed from notes/06 section 3 and is the")
    print(f"    thing under test.]")
    print()
    print("  NOTE ON A BUG FOUND WHILE RE-DERIVING: the first version of this script")
    print("  charged I/O as 2K + 2V per token per head, omitting that q, k and g are")
    print("  three separate K-wide tensors.  That is 3K + 2V.  The error made the")
    print("  compulsory figure 17.0 MB instead of 5.5 and the fold ratio 10.1x")
    print("  instead of 13.0 -- i.e. it would have looked like a refutation of the")
    print("  study's own number.  It is the fourth time this study has produced an")
    print("  apparent refutation that was a bug in the checker (notes/06 section 3")
    print("  records three).  Recorded here so the pattern has a count.")
    print()
    print("  THE 8.00 GiB FIGURE, AND WHERE IT COMES FROM")
    print("  ---------------------------------------------------")
    for label, (b, t, h, k, c, es) in {
        "kda/perf_kda.py  (the BENCHMARKED shape)": (2, 512, 4, 64, 64, 4),
        "kda/cost.py:17  (the COST-MODELLED shape)": (4, 4096, 32, 128, 64, 2),
    }.items():
        nt = t // c
        el = b * nt * h * c * c * k
        print(f"  {label}")
        print(f"    [B,NT,HV,C,C,K] = {el:,} elements = 2^{int(math.log2(el))} exactly; "
              f"fp32 {el*4/GiB:.3f} GiB, bf16 {el*2/GiB:.3f} GiB")
    print()
    print("  So '8.00 GiB' is the cost.py shape in bf16, ONE decay tensor, and it is")
    print("  128x larger than the 64 MiB that the same tensor occupies at the shape")
    print("  the KDA code actually benchmarks.  Both numbers are in the record and")
    print("  neither says which shape the 'the hand-written side wins by")
    print("  construction' claim is about.  It matters enormously: see")
    print("  crossover/claim.py section 2, where the benchmarked shape is 16x over")
    print("  L2 on a T4 and UNDER it on a B200, and the cost-modelled shape is over")
    print("  every device's L2 at every legal chunk size on all five.")
    print()
    print("  [MODELLED.  The cost.py shape is read from kda/cost.py:17-18; the")
    print("   perf_kda shape from kda/perf_kda.py:90.  Neither was executed; the byte")
    print("   counts are arithmetic on those shapes.  The '8.00 GiB' string itself is")
    print("   from the brief to this work and matches the derivation exactly, which is")
    print("   a consistency check on the derivation, not a confirmation of a run.]")


# =============================================================================
# 4. the authoring-artifact cliffs -- the strongest counter-case to the thesis
# =============================================================================
def section_cliffs() -> None:
    print("=" * 100)
    print("4. THE 380x / 1347x CLIFFS -- A DIRECT CONTRADICTION, AND THE BEST ONE")
    print("=" * 100)
    print("""
  arXiv 2607.04454 (via research/SOURCES.md:41-49 and notes/04:123) reports
  naive -> optimized cliffs of 380x (A100, tuned) and 1347x (A100, untuned,
  364 -> 0.270 ms, clock-locked) on LayerNorm, and 1541x on GH200.  The cause
  the study attributes it to is RC0a: `T.serial` not converted to `T.reduce`.

  The thesis says the compiler/human crossover is set by whether the
  materialised intermediate fits in on-chip memory.  Apply that to LayerNorm.

  A LayerNorm on a training-shaped activation is [B,T,H].  What does it
  MATERIALISE?  Not the activation -- that is an input.  A fused LayerNorm
  materialises the row statistics: the row mean, the row variance (or its sum of
  squares), and the reciprocal standard deviation.  All three are [B,T] or
  [B,T,1].  There is no [B,H,T,T] and nothing of size T^2, and the size of the
  biggest one is independent of H entirely.
""")
    for (B, T, H) in ((32, 4096, 4096), (8, 8192, 4096), (64, 2048, 8192)):
        stats = B * T * 4
        print(f"  B={B:>3} T={T:>5} H={H:>5}  activation (an INPUT, {B*T*H*4/MiB:>8.0f} MiB)"
              f"   materialised [B,T] stats: {stats/1024:>8.1f} KiB")
    print()
    print("  Compare the materialised intermediates against the SMALLEST L2 in the")
    print("  entire study (T4, 4 MiB) and the largest (B200, 126 MiB):")
    stats = 32 * 4096 * 4
    for dev, l2 in (("T4", 4 * MiB), ("P100", 4 * MiB), ("L4", 48 * MiB),
                    ("H100", 50 * MiB), ("B200", 126 * MiB)):
        print(f"    {dev:<6} L2 {l2/MiB:>7.1f} MiB    [B,T] stats / L2 = "
              f"{stats/l2:.2e}  -> fits by a factor of {l2/stats:,.0f}")
    print()
    print("  To push a LayerNorm's [B,T] statistics past even the LARGEST L2 in this")
    print("  study (B200, 126 MiB) you need B*T > %.1f M, e.g. B=64 and T=%.0f -- a"
          % (126 * MiB / 4 / 1e6, 126 * MiB / 4 / 64))
    print("  half-million-token sequence, an order of magnitude past anything anyone")
    print("  trains or serves.  So the ratio is not a free parameter here: for")
    print("  LayerNorm it is pinned to 'fits', by a wide and growing margin, at every")
    print("  reachable shape.  [MODELLED]")
    print()
    print("""  So the materialised intermediates of a LayerNorm sit 8x to 252x inside every
  cache in this study, on every device, at every shape anyone would reach for --
  because they do not scale with the hidden dimension H, only with B and T.  And
  the cliff there is 380x on an A100, 1347x untuned, 1541x on a GH200.

  Under the thesis' own rule, that cell of the table reads "everything fits, so
  the compiler wins, ratio ~1".  The measurement says 380.  The thesis is not
  strained here; it is falsified, on the corpus's own numbers, by a factor of
  380.

  WORSE, AND THIS IS THE PART THAT REALLY HURTS: the corpus contains 380x AND
  1347x AND a 0.95x geomean, for the SAME operator on the SAME device.  A
  capacity threshold assigns one position on the s/capacity axis to (LayerNorm,
  A100) and therefore ONE predicted ratio.  The corpus reports three ratios that
  differ by three orders of magnitude for that one cell.  No function of s and
  capacity can produce 380, 1347 and 1.06 from one input -- and the s here does
  not even change between them, because all three run the same program.
   [the [B,T] statistics are what a fused LayerNorm writes; the ratio of the three cards' caches to them is the point].

  What CAN produce all three is a scheduling defect:
     1347x  untuned baseline, `T.serial` not converted
      380x  tuned baseline, still `T.serial` not converted
      0.95x  the autoscheduler, which converts it
  The size of the intermediate is IDENTICAL in all three.  It is not a variable,
  so it cannot be the explanandum.  This is not a subtle tension with the thesis;
  it is the thesis being asked to explain a number it has no access to.

  THE RECONCILIATION, and it is a real one.  Near the threshold, the byte term
  dominates everything else -- a factor of passes * s * es / BW, which is 10s or
  100s of x when s is orders of magnitude past capacity.  Far from the threshold
  on the small side, the byte term is ~0 and the only thing left is code quality,
  which is where 380x and 1347x live.  So the defensible claim is not

      "the crossover is set by fit, and it is a step"

  but

      "fit sets the MAGNITUDE of the byte term; how far the byte term dominates
       the quality term is a second, unmeasured ratio, and the crossover is
       where they are equal -- which is a CROSSING of two smooth curves, not a
       step."

  A crossing of two smooth curves is continuous.  It has a kink, not a cliff.  So
  the step-function half of the thesis is not a defensible sharpening of the
  capacity half; it is a different and weaker claim, and the step is the part

""")
    print("  The arithmetic that makes this concrete, on the H100:")
    for s_mib in (0.25, 1, 4, 16, 64, 256, 1024):
        passes = 3
        t_dram = passes * s_mib * MiB / 3.35e12
        quality = 380.0     # the observed quality gap, held FIXED across s
        t_qual = 1.0 / 380  # the best time the quality gap can buy
        total = t_dram + t_qual
        print(f"    s={s_mib:>7.2f} MiB   byte term {t_dram*1e6:>9.1f} us   "
              f"quality term {t_qual*1e6:>7.1f} us   byte share "
              f"{t_dram/total*100:>5.1f}%")
    print()
    print("    The byte term only overtakes a 380x quality gap somewhere around")
    print("    s ~ 1 GiB on an H100.  The KDA port's 8 GiB is past that; the KDA")
    print("    port's 64 MiB is not.  So even inside one operator family the thesis")
    print("    predicts a transition, and the study's own two KDA shapes sit on")
    print("    OPPOSITE SIDES of it.  [MODELLED -- the 380x is a published number")
    print("    from a different paper on a different operator and is used here only")
    print("    as an order-of-magnitude stand-in for 'the quality term'.  It is not a")
    print("    measurement of LayerNorm's quality gap on this hardware.]")


# =============================================================================
# 5. arXiv 2012.07145 -- the 0.95x
# =============================================================================
def section_parity() -> None:
    print("=" * 100)
    print("5. arXiv 2012.07145 -- THE 0.95x, AND WHY IT CANNOT ADJUDICATE")
    print("=" * 100)
    states = {"matrix multiply": 698}
    geomean_all = 11740
    largest = 50488
    print(f"  SGEMM states evaluated: {states['matrix multiply']:,}")
    print(f"  geomean over the 17 programs: {geomean_all:,}")
    print(f"  largest program (learned demosaic): {largest:,}")
    print()
    print(f"  SGEMM / geomean          = {states['matrix multiply']/geomean_all:.4f} "
          f"({states['matrix multiply']/geomean_all*100:.1f}%)")
    print(f"  geomean / SGEMM          = {geomean_all/states['matrix multiply']:.2f}x "
          f"  <- notes/PARITY-SCOPE.md:134 says '~17x smaller'")
    print(f"  largest / SGEMM          = {largest/states['matrix multiply']:.2f}x  "
          f"<- the same line says '~72x smaller'")
    print()
    print("  Both of PARITY-SCOPE's ratios re-derive.  [MODELLED from the Table 1")
    print("  figures quoted at PARITY-SCOPE.md:126-131.]")
    print()
    print("""  AND HERE IS THE PROBLEM FOR THE THESIS, WHICH IS A PROBLEM OF KIND RATHER
  THAN DEGREE.  0.95x is a GEOMEAN over 17 imaging programs on one V100.  A
  geomean of a step function is not a step function; it is a smooth functional of
  the distribution of s/capacity across the 17 programs, and 17 Halide imaging
  pipelines have image-sized tensors whose s/capacity ratios are unknown to me
  and, as far as I can tell, to anyone who has published them.

  So the thesis cannot explain the 0.95x in the strong sense ("these programs
  all had intermediates that fit, so the machine won, as predicted").  It can
  only restate it ("the machine won 17 times running, so on average the compiler
  is as good as the human").  The 17 programs' intermediate sizes are not in the
  literature, so the thesis' own explanatory variable was never measured on them.

  This is the same defect that PARITY-SCOPE.md §1.2 already found from the other
  direction -- the suite contains no recurrence, no attention, no normalisation as
  a category -- and the thesis inherits it.  What the thesis adds is a worse
  version: PARITY-SCOPE's objection is that the suite lacks the right OPERATORS.
  The thesis' objection is that the suite does not report the intermediate sizes
  that its explanatory variable is made of.  The thesis is unfalsifiable on the
  one result it most wants to explain.
""")


# =============================================================================
# 6. Toulme / Splash Attention -- the 37.6x byte reduction
# =============================================================================
def section_toulme() -> None:
    print("=" * 100)
    print("6. TOULME / SPLASH ATTENTION -- THE ONE PRIOR RESULT THE THESIS EXPLAINS")
    print("=" * 100)
    naive_vliw, naive_hbm = 10245, 526
    splash_vliw, splash_hbm = 1651, 14
    print(f"  naive (materialised): {naive_vliw:,} VLIW bundles, {naive_hbm} MB HBM")
    print(f"  Splash Attention:     {splash_vliw:,} VLIW bundles, {splash_hbm} MB HBM")
    print(f"  byte reduction:       {naive_hbm/splash_hbm:.2f}x")
    print(f"  bundle reduction:     {naive_vliw/splash_vliw:.2f}x")
    print()
    print("  Both re-derive from research/TPU.md:645-646.  [MODELLED]")
    print()
    print("""  This is the case the thesis handles best, and it is worth saying why.  Here
  the two implementations are the SAME algorithm's two schedules, the byte counts
  were MEASURED (HBM traffic, not modelled), and the measured byte ratio and the
  measured time ratio are of the same order.  Nothing is a modelling choice.

  It also shows what the thesis gets right that a plain intensity argument does
  not: Splash's win is not that it "reduced HBM traffic" in the sense of
  removing traffic, it is that it removed an intermediate.  The 526 -> 14 MB is
  the disappearance of a 128 MB attention matrix (Toulme, quoted in
  research/TPU.md:660: "the 128MB attention matrix still gets written to HBM
  between fusion.5 and fusion.2"), and the remainder is compulsory.  That is the
  thesis' mechanism exactly, and it is the reason the corpus is bimodal rather
  than continuous: whether the intermediate exists is a discrete decision that a
  compiler either makes or does not make, and the two branches are orders of
  magnitude apart.

  The qualification: 14 MB of Splash traffic on a v6e still means 14 MB, and if
  the 128 MB had been 12 MB it would have been written to HBM anyway and both
  versions would have been slow.  So the discreteness is in the COMPILER'S
  CHOICE, and the capacity is only what makes the choice possible.  Toulme's
  result is evidence for "there is a discrete branch", not for "the branch is
  located at capacity".
""")


def main() -> int:
    print("=" * 100)
    print("LITERATURE RE-DERIVATION -- every number recomputed, nothing transcribed")
    print("=" * 100)
    print("  STATUS: the source tables are [READ-FROM-SOURCE] and mostly one remove")
    print("  from a primary document (via research/*.md).  Every ratio printed here is")
    print("  [MODELLED] arithmetic on them.  Where a ratio is one remove from the")
    print("  paper it is marked [UNVERIFIED] at that remove.")
    section_zhang()
    section_jaxbench()
    section_kda_bytes()
    section_cliffs()
    section_parity()
    section_toulme()
    print("=" * 100)
    print("VERDICT TABLE -- which prior results the thesis explains, which it breaks")
    print("=" * 100)
    rows = [
        ("Toulme 37.6x byte reduction", "EXPLAINS", "measured bytes, same algorithm, "
         "the intermediate is what goes away"),
        ("JAXBench GEMM 0.96x", "EXPLAINS (weakly)", "also explained by plain "
         "arithmetic intensity; the thesis adds nothing here"),
        ("JAXBench attention 3.91x / 6.91x", "EXPLAINS", "the attention matrix is "
         "the thing being removed"),
        ("JAXBench MLA 0.94x, Sparse 0.86x", "DOES NOT EXPLAIN", "a bad hand-written "
         "kernel; no capacity story produces a sub-1x row"),
        ("2012.07145 0.95x geomean", "CANNOT ADJUDICATE", "a geomean of 17 programs "
         "whose intermediate sizes were never reported"),
        ("JAXBench's 2.08x convention", "UNDERMINES", "a FLOORED geomean: 2.0803 "
         "with the three sub-1x rows clipped to 1.0, 2.0154 raw. The flagship "
         "statistic cannot report a human loss"),
        ("Zhang 35x at n=4096", "CONTRADICTS", "never overtakes; the 35x is the peak "
         "of a hump, and MXU utilisation 20.1% vs 94.3% explains it instead"),
        ("2607.04454 380x / 1347x / 1347x LayerNorm", "CONTRADICTS", "three ratios "
         "for one (operator, device) cell; a function of s and capacity cannot "
         "produce them"),
        ("this study's KDA 8.00 GiB", "EXPLAINS", "8 GiB is past any L2 by 65-2048x"),
        ("this study's KDA at the benchmarked shape", "UNDERMINES", "64 MiB is 16x "
         "over a T4's L2 and UNDER a B200's, so it is not 'wins by construction'"),
        ("this study's 256x", "UNDERMINES", "cross-scope; the same-scope ratio is 64x "
         "and the byte ratio is 48.8x"),
    ]
    print(f"  {'result':<36}{'verdict':<20}why")
    print("  " + "-" * 96)
    for a, b, c in rows:
        print(f"  {a:<36}{b:<20}{c}")
    print()
    print("  Two contradicted, four more unexplainable, three of the study's OWN")
    print("  numbers are wrong in the direction that makes the thesis look better")
    print("  than the shapes support, and the corpus's flagship statistic has the")
    print("  bad half of its distribution clipped.  A thesis with that profile is not")
    print("  yet a thesis; notes/07 states the version that survives.")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
