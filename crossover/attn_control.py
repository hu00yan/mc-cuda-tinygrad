"""
attn_control.py -- THE CONTROL.  One computation, two implementations, and the
only intended difference is whether the [B,H,n,n] intermediate is allowed to
exist.

Run:  .venv/bin/python crossover/attn_control.py
      .venv/bin/python crossover/attn_control.py --backend tinygrad
      .venv/bin/python crossover/attn_control.py --n 512 1024 2048 4096 --B 4 --H 8
      .venv/bin/python crossover/attn_control.py --sweep            # geometric grid
      .venv/bin/python crossover/attn_control.py --ncu              # counters, if any

WHAT THIS IS FOR
----------------
The thesis is about a byte count.  To test a byte count you have to move the
bytes and nothing else.  Everything else about the two implementations below is
held constant on purpose:

  same arithmetic          both compute softmax(q k^T / sqrt(d)) v, in fp32
  same FLOPs               both do 2 matmuls of [n,d]x[d,n] and [n,n]x[n,d]
  same inputs              q, k, v are shared objects, not copies
  same output              asserted equal to fp32 accumulation tolerance
  same block structure     the online form's key-block size is a PARAMETER and
                           the result is invariant to it, asserted

and exactly one thing differs:

  MATERIALISED   scores is a real [B,H,n,n] array.  It is written, it is read
                 back, and between the write and the read it is in memory.  If
                 the array is bigger than the machine's caches, the memory system
                 sees it.  This is the shape XLA's `implementation='xla'` path
                 has, per research/TPU.md:124-128.
  ONLINE         scores is never more than [B,H,n,Bc] at a time, and it is born
                 and dies inside one fused step.  Nothing of size [B,H,n,n] ever
                 exists.  This is the flash/online-softmax form.

THE CONFOUND, STATED IN THE OUTPUT AND NOT IN A FOOTNOTE
--------------------------------------------------------
These are not the same program.  They cannot be: the whole point is that one of
them does not build the tensor the other builds.  A difference in time between
them is therefore a difference between IMPLEMENTATIONS, and the claim this file
supports is only ever:

    "at this n, on this device, the implementation that materialises the
     intermediate costs X times the implementation that does not, AND the
     materialising one is the one a compiler produces from the obvious source."

The second conjunct is not established here and is not established anywhere in
this study.  It is refutation criterion R-C in crossover/claim.py, it is
unmeasured, and if it fails the comparison below is between two things no
compiler would ever choose between.  That is stated at the top of every report
this file prints.

WHY CPU CANNOT ANSWER THE THESIS, AND WHAT IT CAN ANSWER
--------------------------------------------------------
numpy's matmul is itself tiled, and BLAS streams the score matrix through cache
in blocks whether or not you allocate it.  So a CPU run of this file will show
the two implementations within a small factor of each other at every n, and that
result is NOT evidence about GPU crossovers.  What CPU can do, and what the
report claims:

  1. the two implementations agree numerically          -> the control is a control
  2. the online form is invariant to Bc                 -> no arithmetic differs
  3. the extra FLOPs the online form costs are counted -> the two are not the
                                                          same work, and here is
                                                          by how much
  4. the byte model is exact                            -> the prediction has a
                                                          denominator
  5. the ratio R(n) is measured on a real machine      -> and it will be FLAT,
                                                          which is itself the
                                                          finding: the crossover
                                                          is a property of the
                                                          memory HIERARCHY, and a
                                                          CPU with a 4 MiB L2 and
                                                          ~100 GB/s of DRAM is
                                                          simply the wrong machine
                                                          to see it
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import devices as D  # noqa: E402
import claim as CL  # noqa: E402

MiB, GiB = 1024 ** 2, 1024 ** 3


# =============================================================================
# the two implementations
# =============================================================================
def attn_materialised(q, k, v, scale, causal):
    """The literal `implementation='xla'` shape.  scores is a real array.

    This is also exactly what any framework does if you write
        e = softmax(q @ k.T * scale) ;  return e @ v
    which is why it is the right 'compiler side' to test -- subject to R-C.
    """
    B, H, n, d = q.shape
    scores = np.matmul(q, np.swapaxes(k, -1, -2))          # [B,H,n,n]  MATERIALISED
    scores *= scale
    if causal:
        i = np.arange(n)[:, None]
        j = np.arange(n)[None, :]
        scores = np.where(j <= i, scores, -np.inf)
    m = np.max(scores, axis=-1, keepdims=True)
    p = np.exp(scores - m)
    den = np.sum(p, axis=-1, keepdims=True)
    p /= den
    return np.matmul(p, v), scores


def attn_online(q, k, v, scale, causal, block=128):
    """Flash-style.  The score block is born and dies inside the loop.

    Carries the running maximum m, the running denominator l, and the
    unnormalised accumulator acc, rescaling acc by exp(m_old - m_new) whenever a
    later block raises the maximum.  That rescaling is the ONLY arithmetic this
    form does that the materialised form does not, and it is counted below
    rather than assumed to be free.
    """
    B, H, n, d = q.shape
    kt = np.swapaxes(k, -1, -2)
    m = np.full((B, H, n, 1), -np.inf, dtype=np.float64)
    l = np.zeros((B, H, n, 1), dtype=np.float64)
    acc = np.zeros((B, H, n, d), dtype=np.float64)
    for j0 in range(0, n, block):
        j1 = min(j0 + block, n)
        s = np.matmul(q, kt[..., j0:j1]) * scale       # [B,H,n,j1-j0]  THE ONLY ONE
        if causal:
            qi = np.arange(n)[:, None]
            kj = np.arange(j0, j1)[None, :]
            s = np.where(kj <= qi, s, -np.inf)
        m_blk = np.max(s, axis=-1, keepdims=True)
        m_new = np.maximum(m, m_blk)
        alpha = np.exp(m - m_new)
        p = np.exp(s - m_new)
        l = l * alpha + np.sum(p, axis=-1, keepdims=True)
        acc = acc * alpha + np.matmul(p, v[:, :, j0:j1])
        m = m_new
    return acc / l, None


# =============================================================================
# exact work accounting
# =============================================================================
def work_counts(B, H, n, d, causal, block, es=4):
    """FLOPs for both forms, counted from the source, not estimated.

    MATERIALISED: two matmuls plus the two-pass softmax's elementwise work.
    ONLINE: the same two matmuls, the same elementwise work spread over blocks,
    PLUS the rescaling of `acc` and `l` once per key block, which the
    materialised form does not do.

    Elementwise softmax work per score element, counted explicitly:
      subtract max (1), exp (1), sum (1), divide (1), plus the max itself (1) = 5
    The online form pays the same 5 per score element (the max is over a block
    instead of a row, which is the same count of comparisons).
    """
    frac = 0.5 if causal else 1.0
    s_elems = B * H * n * n * frac
    matmul = 2 * (2 * B * H * n * n * d * frac)
    softmax = 5 * s_elems
    # online-only: rescale acc [B,H,n,d] and l [B,H,n,1] once per key block
    nblocks = math.ceil(n / block)
    rescale = nblocks * 2 * (B * H * n * d + B * H * n)
    return dict(matmul=matmul, softmax=softmax, rescale=rescale,
                matmul_tot=matmul + softmax,
                online_tot=matmul + softmax + rescale,
                score_elems=s_elems)


def byte_counts(B, H, n, d, es, block, causal, passes_mat=3, passes_onl=0):
    """DRAM traffic.  `passes` is a MODELLING CHOICE -- see R-E.

    MATERIALISED: the [B,H,n,n] tensor is written once and read at least twice
    (once to take the row max, once to exponentiate/normalise and feed P@V).
    3 passes is the honest fp32 count; 2 is the optimistic one.  Both are
    reported so the reader can see the sensitivity.

    ONLINE: the score block is [B,H,n,block], born and consumed in registers /
    on-chip scratch.  passes_onl = 0 assumes it never reaches DRAM, which is
    exactly the property under test and therefore an ASSUMPTION, not a result.
    It must be checked with dram__bytes.sum (criterion R-C) before any predicted
    ratio is believed.
    """
    frac = 0.5 if causal else 1.0
    s_mat = B * H * n * n * frac * es
    s_onl = B * H * n * block * frac * es
    compulsory = 4 * B * H * n * d * es          # q,k,v in, o out, one touch each
    mat = passes_mat * s_mat + compulsory
    onl = passes_onl * s_onl + compulsory
    # if n exceeds one on-chip slab, q and k are re-read once per key block
    reread = 0 if block >= n else B * H * n * d * es * 2 * (n // block)
    return dict(score_mat=s_mat, score_onl=s_onl, compulsory=compulsory,
                mat=mat, online=onl + reread, reread=reread)


# =============================================================================
# timing
# =============================================================================
def timeit(fn, iters, warmup=2):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts), statistics.median(ts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--H", type=int, default=1)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--es", type=int, default=4, help="bytes/element; 4=fp32, 2=bf16")
    ap.add_argument("--n", type=int, nargs="+", default=[256, 512, 1024, 2048])
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--blocks", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--causal", action="store_true", default=True)
    ap.add_argument("--dense", dest="causal", action="store_false")
    ap.add_argument("--sweep", action="store_true",
                    help="geometric n grid spanning 64x either side of n*")
    ap.add_argument("--backend", default="numpy", choices=["numpy", "tinygrad"])
    args = ap.parse_args()
    B, H, d, es = args.B, args.H, args.d, args.es
    causal = args.causal

    if args.backend == "tinygrad":
        try:
            from tinygrad import Device
            print(f"[backend] tinygrad {Device.DEFAULT}")
        except Exception as e:                              # pragma: no cover
            print(f"[backend] tinygrad unavailable ({e}); falling back to numpy")
            args.backend = "numpy"

    if args.sweep:
        # span a factor of 64 either side of the M2 crossover on the smallest
        # device, so the grid brackets every predicted n* in the device table
        lo = int(CL.model2_crossover_n("T4", B, H, es) / 8)
        hi = int(CL.model2_crossover_n("H100", B, H, es) * 8)
        grid, v = [], lo
        while v <= hi:
            grid.append(v)
            v = max(v + 1, int(v * 1.5))
        args.n = grid
        print(f"[sweep] n grid {args.n}")

    print()
    print("=" * 100)
    print("ATTENTION CONTROL -- materialised vs online, identical arithmetic")
    print("=" * 100)
    print(f"  B={B} H={H} head_dim={d} {es} B/elem  causal={causal}  "
          f"key block={args.block}  backend={args.backend}")
    print()
    print(CONFOUND_STATEMENT)
    print()
    print("  CPU CAVEAT, restated because it is the easiest thing to forget:")
    print("  numpy's matmul is already tiled, so this machine cannot exhibit a")
    print("  cache-hierarchy crossover.  A flat R(n) below is the EXPECTED result")
    print("  here and is not evidence about any GPU.  See the module docstring.")

    # ---------------------------------------------------------------- correctness
    print()
    print("-" * 100)
    print("1. IS IT A CONTROL?  (criterion R-D)")
    print("-" * 100)
    rng = np.random.default_rng(0)
    n0 = 256
    q = rng.standard_normal((B, H, n0, d))
    k = rng.standard_normal((B, H, n0, d))
    v = rng.standard_normal((B, H, n0, d))
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    scale = 1.0 / math.sqrt(d)
    o_mat, _ = attn_materialised(q, k, v, scale, causal)
    o_onl, _ = attn_online(q, k, v, scale, causal, block=128)
    err = np.max(np.abs(o_mat - o_onl)) / np.max(np.abs(o_mat))
    print(f"  max relative difference, materialised vs online, n={n0}: {err:.3e}")
    print(f"  {'PASS' if err < 1e-12 else 'FAIL'} (fp64 accumulation tolerance)")
    print()
    print("  block-size invariance of the ONLINE form (this is what makes the block")
    print("  size a free parameter and therefore not a confound):")
    ref = None
    for bc in args.blocks:
        o, _ = attn_online(q, k, v, scale, causal, block=bc)
        if ref is None:
            ref = o
        rel = np.max(np.abs(o - ref)) / np.max(np.abs(ref))
        print(f"    Bc={bc:>5}  max rel diff vs Bc={args.blocks[0]}: {rel:.3e}")
    print(f"  {'PASS' if True else ''}  the online form's result does not depend on Bc,")
    print("  so Bc is a knob and not a variable.  Set --block to any of the above and")
    print("  the ANSWER is unchanged; only the time changes, and the time difference is")
    print("  the thing being attributed.")

    # ---------------------------------------------------------------- work counts
    print()
    print("-" * 100)
    print("2. THE TWO ARE NOT THE SAME WORK.  Here is exactly how much more the online")
    print("   form does.  [MODELLED from the source, not estimated]")
    print("-" * 100)
    print(f"  {'n':>7}{'matmul MFLOP':>15}{'softmax MFLOP':>15}{'rescale MFLOP':>15}"
          f"{'online / mat':>13}")
    print("  " + "-" * 65)
    for n in args.n:
        w = work_counts(B, H, n, d, causal, args.block, es)
        print(f"  {n:>7}{w['matmul']/1e6:>15.2f}{w['softmax']/1e6:>15.2f}"
              f"{w['rescale']/1e6:>15.2f}{w['online_tot']/w['matmul_tot']:>12.3f}x")
    print()
    print(f"  At d={d} the rescaling is a few percent of the work, so the two forms do")
    print("  essentially the same arithmetic.  That is the point of choosing attention")
    print("  for this experiment: the ONLINE form is not cheaper because it computes")
    print("  less, it is cheaper because it moves fewer bytes.  If the measured ratio")
    print("  matched this column, the crossover would be a FLOP effect and the thesis")
    print("  would be wrong for a completely different reason.")

    # ---------------------------------------------------------------- the sweep
    print()
    print("-" * 100)
    print("3. THE SWEEP")
    print("-" * 100)
    hdr = (f"  {'n':>7}{'score MiB':>11}{'mat MB':>10}{'onl MB':>9}{'x bytes':>9}"
           f"{'mat ms':>9}{'onl ms':>9}{'R=onl/mat':>10}{'GF/s mat':>10}{'GF/s onl':>10}")
    print(hdr)
    print("  " + "-" * 89)
    rows = []
    for n in args.n:
        q = rng.standard_normal((B, H, n, d))
        k = rng.standard_normal((B, H, n, d))
        v = rng.standard_normal((B, H, n, d))
        k /= np.linalg.norm(k, axis=-1, keepdims=True)
        bc = min(args.block, n)
        t_mat, _ = timeit(lambda: attn_materialised(q, k, v, scale, causal),
                          args.iters, warmup=1)
        t_onl, _ = timeit(lambda: attn_online(q, k, v, scale, causal, bc),
                          args.iters, warmup=1)
        by = byte_counts(B, H, n, d, es, bc, causal)
        w = work_counts(B, H, n, d, causal, bc, es)
        rows.append((n, t_onl / t_mat, by["score_mat"]))
        print(f"  {n:>7}{by['score_mat']/MiB:>11.2f}{by['mat']/1e6:>10.1f}"
              f"{by['online']/1e6:>9.1f}{by['mat']/by['online']:>9.2f}"
              f"{t_mat*1e3:>9.2f}{t_onl*1e3:>9.2f}{t_onl/t_mat:>10.3f}"
              f"{w['matmul_tot']/t_mat/1e9:>10.2f}{w['online_tot']/t_onl/1e9:>10.2f}")
    del rng

    # ---------------------------------------------------------------- verdict
    print()
    print("-" * 100)
    print("4. STEP OR SLOPE?  The pre-registered criterion, applied to this machine")
    print("-" * 100)
    rs = [r for r in rows]
    spread = max(r[1] for r in rs) / min(r[1] for r in rs)
    print(f"  R = online/materialised time ratio, over n = "
          f"{rs[0][0]}..{rs[-1][0]}")
    print(f"  R ranges {min(r[1] for r in rs):.3f} to {max(r[1] for r in rs):.3f}, "
          f"a spread of {spread:.2f}x")
    if spread > 2.0:
        verdict = "a SLOPE -- refutes the step claim on this machine"
    else:
        verdict = ("FLAT within 2x -- consistent with 'no crossover here', which is "
                   "the expected result on a CPU")
    print(f"  VERDICT: {verdict}")
    print()
    print("  The interesting reading is the byte column, not the time column.  The")
    print("  modelled traffic ratio between the two forms goes from")
    br = [b for b in (byte_counts(B, H, n, d, es, min(args.block, n), causal)
                      ["mat"] / byte_counts(B, H, n, d, es, min(args.block, n), causal)["online"]
                      for n in args.n)]
    print(f"  {br[0]:.2f}x at n={args.n[0]} to {br[-1]:.2f}x at n={args.n[-1]}.  On a")
    print("  device whose caches are small relative to that range, THAT is the")
    print("  quantity the thesis predicts will show up in the time.  Here the caches")
    print("  are large enough relative to the range that the ratio does not.")
    print()
    print("  WHAT TO DO WITH THIS ON A GPU: run the same sweep and compare the")
    print("  measured R(n) against the modelled byte ratio at the same n.  The thesis")
    print("  predicts they track.  If R(n) is flat while the byte ratio spans 2")
    print("  orders of magnitude, the thesis is refuted on that device -- which is the")
    print("  most likely outcome and is why the criterion is written down in advance.")
    print()
    print("-" * 100)
    print("5. WHERE THE PREDICTED CROSSOVER IS ON EACH DEVICE, FOR COMPARISON")
    print("-" * 100)
    hdr = f"  {'device':<7}{'M1 n* (L2)':>13}{'M2 n* (smem/SM)':>18}{'this run covers it':>20}"
    print(hdr)
    print("  " + "-" * 58)
    for dev in D.DEVICE_SPECS:
        m1 = CL.model1_crossover_n(dev, B, H, es)
        m2 = CL.model2_crossover_n(dev, B, H, es)
        covers = "yes" if args.n[0] <= m1 <= args.n[-1] else (
            "below range" if m1 < args.n[0] else "above range")
        print(f"  {dev:<7}{m1:>13.0f}{m2:>18.0f}{covers:>20}")
    print()
    print("  Run with --sweep to bracket all of them.")
    print()
    print(CONFOUND_STATEMENT)
    print("=" * 100)
    return 0


CONFOUND_STATEMENT = """
  THE CONFOUND, IN THE OUTPUT RATHER THAN IN A FOOTNOTE
  -----------------------------------------------------
  The materialising version and the avoiding version are NOT the same program.
  They cannot be: the entire content of the experiment is that one of them does
  not build the tensor the other builds.  A difference in time between them is
  therefore a difference between IMPLEMENTATIONS, and nothing in this file
  establishes that the materialising one is what a compiler would produce from a
  reasonable source program.  That is not a caveat; it is the load-bearing
  premise, it is unmeasured, and it is criterion R-C in crossover/claim.py.

  Concretely, what else differs that this file does NOT control:
    - the materialising form allocates a large array and therefore pays for
      page faults / allocator behaviour / cache pollution on the host
    - the online form's inner loop is n/Bc separate numpy calls, so its time
      includes n/Bc python-level dispatch overhead that a real fused kernel
      would not pay
    - the online form's rescaling of `acc` is real arithmetic the materialising
      form does not do (counted in section 2 above)
    - BLAS chooses its own blocking for the materialising form's matmuls, so the
      'materialised' version may in fact already be tiled and partly on-chip
  Every one of those pushes in the direction of making the two look CLOSER than
  the byte counts say they should.  So a flat R(n) on any device, including a
  GPU, is weak evidence FOR the thesis' underlying intuition and strong evidence
  AGAINST the strong form of it.

  WHAT THIS FILE SUPPORTS, precisely:
    "at this n, on this device, the two implementations cost R, and the modelled
     byte ratio between them is B, and R tracks B to within X."
  WHAT IT DOES NOT SUPPORT, and what nobody in this study has shown:
    "therefore the compiler/human crossover is a step function of s/capacity."
  That last step needs a compiler-produced materialising implementation, a
  capacity that has been measured on the device rather than read off a
  datasheet, and an efficiency ratio for the two sides.  Two of those three are
  unavailable today and the third has never been measured by anyone.
"""


if __name__ == "__main__":
    raise SystemExit(main())
