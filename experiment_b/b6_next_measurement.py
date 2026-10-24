#!/usr/bin/env python3
"""b6_next_measurement.py -- the ONE experiment that turns (b) into a cost.

THE PROBLEM WITH THE STUDY AS IT STANDS
---------------------------------------
Comparison (b) is settled: the decay is not materialised, and the compiler
construction is arch-invariant. What (b) has NO number for is COST. The study
has a kernel count, a byte accounting, an instruction-class census, and exactly
one timing, and that timing is event-bracketed and host-synchronised per launch,
so it measures its own harness at least as much as it measures the kernels.

So the question is not "what is true" -- b1..b5 answered that -- it is "what
would it cost". And there is one measurement that answers most of it.

THE MEASUREMENT: dram__bytes.sum FOR ONE kda_chunk CALL
--------------------------------------------------------
  It is the right one for three reasons.

  1. It is the only quantity that can OVERRULE the construction-level findings,
     and a study that cannot be overruled is not a study. Every byte claim here
     is an argument from signatures: a kernel can only touch buffers in its
     argument list, so the absence of a 67,108,864 B argument is a strong bound
     on traffic. It is a bound, not a measurement. dram__bytes.sum IS the
     measurement, and it is the one number in the whole study that would
     confirm or destroy the headline.

  2. The study has already pre-registered its own falsification condition for
     this number, with a band and a threshold. Re-registering it here is not new
     work; it is executing a condition that was written down in advance and then
     not run. That asymmetry -- a pre-registered refutation that was never
     executed -- is a much worse state to be in than a missing measurement.

  3. It is one `ncu` invocation on one shape that has already been benchmarked,
     on hardware that the study can rent for an hour. It is not a research
     project. It is a command.

WHAT IT COSTS
-------------
  Re-derived from the repo's own logs, and labelled as an estimate because no
  run of it exists. The `dram__bytes_read/write` pair is a single-pass counter,
  so ncu does not need to replay the kernel for it. See section 3 for the
  command, section 4 for the runtime, section 5 for the decision each outcome
  forces, and section 6 for the second measurement if the first is ambiguous.

  python3 experiment_b/b6_next_measurement.py            # the whole argument
  python3 experiment_b/b6_next_measurement.py --emit-cmd # just the ncu line
"""

from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment_b.common import (  # noqa: E402
    BENCH, INFERRED, MODELLED, NOT_HERE, RE_DERIVED, head, kv, read, require,
    rule, verdict)

T4_LOG = "out/kaggle/20260928-014248-gpu/logs/perf_kda.log"
T4_REPLAY = "out/kaggle/20260928-014248-gpu/logs/replay_kda.log"

# The driver. One call, no benchmark harness, no DEBUG, so the counter is not
# fighting an event wrapper for the same device.
DRIVER = """python -c "
import sys, numpy as np
sys.path.insert(0, 'kda')
from tinygrad import Device
import test_ref as T, tiny_kda
print('device', Device.DEFAULT, file=sys.stderr)
B,Tt,H,HV,K,V,C = 2,512,4,4,64,64,64
q,k,v,g,beta = T.rnd(B,Tt,H,HV,K,V,seed=0,l2n=True)
o,S = tiny_kda.kda_chunk(q,k,v,g,beta,chunk_size=C)
print('one kda_chunk call, B=2 T=512 H=HV=4 K=V=64 C=64', file=sys.stderr)
\""""


def parse_us(path: str, key: str) -> float | None:
    """Pull an integer ms figure off a line whose label is `key`, tolerating a
    parenthetical the log puts between the label and the colon."""
    try:
        for ln in open(path):
            m = re.match(rf"\s*(?:\d+\.\s*)?{re.escape(key)}[^:]*:\s*([\d,]+) ms", ln)
            if m:
                return int(m.group(1).replace(",", ""))
    except OSError:
        pass
    return None


def parse_us_us(path: str, key: str) -> float | None:
    try:
        for ln in open(path):
            m = re.match(rf"\s*{re.escape(key)}\s*:\s*([\d,]+) us", ln)
            if m:
                return int(m.group(1).replace(",", ""))
    except OSError:
        pass
    return None


# =============================================================================
# the byte model, re-derived, so the band is this file's arithmetic
# =============================================================================

def byte_model() -> dict:
    """The study's traffic model, rebuilt from shapes.

    The model ASSUMES the decay is materialised. b1 and b2 say it is not. That
    is the whole tension, and the band below is what the model predicts if the
    tension resolves in the model's favour.
    """
    s = BENCH
    NT = s["T"] // s["C"]
    BNT = s["B"] * NT * s["HV"]
    e = 4  # fp32 everywhere
    dec = BNT * s["C"] * s["C"] * s["K"] * e          # one decay tensor
    # the port builds TWO of them (Dq_all for the intra-chunk scores, Dl for the
    # decay-weighted gram); each is written once and read once.
    dec_traffic = 2 * 2 * dec
    # the eight non-decay intermediates the port materialises, each [B,NT,HV,C,.]
    # with 262,144 elements at this shape, each written once and read once
    inter = 8 * (BNT * s["C"] * s["K"])
    other = 2 * inter * e
    io = 4 * (s["B"] * s["T"] * s["HV"] * s["K"]) * e   # q, k, v, g in
    out = 1 * (s["B"] * s["T"] * s["HV"] * s["V"]) * e  # o out
    state = 2 * (s["B"] * s["HV"] * s["K"] * s["V"]) * e  # S read+written
    other += io + out + state
    compulsory = io + out + state
    total = dec_traffic + other
    return dict(NT=NT, dec=dec, dec_traffic=dec_traffic, other=other,
                inter=inter, io=io, out=out, state=state,
                compulsory=compulsory, total=total, ratio=total / other,
                dec_over_comp=dec_traffic / compulsory,
                band_lo=0.8 * total, band_hi=1.2 * total,
                kill=0.2 * total, kill_hard=58e6)


# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emit-cmd", action="store_true", help="print the ncu line and exit")
    a = ap.parse_args()

    m = byte_model()
    cmd = ("ncu --csv --target-processes all --replay-mode application "
           "--metrics dram__bytes_read.sum,dram__bytes_write.sum "
           "--log-file ncu_kda_dram.csv -- " + DRIVER)

    if a.emit_cmd:
        print(cmd)
        return 0

    head("B6. THE ONE MEASUREMENT THAT WOULD TURN (b) INTO A COST")
    kv("what (b) has settled", "whether the decay is materialised, and whether "
                                "the compiler can reach the human's construction")
    kv("what (b) has NO number for", "cost. Not bytes moved, not time, not "
                                     "achieved bandwidth. Nothing.")
    kv("the gap", "one `ncu` invocation on one shape, on hardware that can be "
                  "rented for an hour")
    print()

    # ------------------------------------------------------------------ 1
    head("1. WHY dram__bytes AND NOT SOMETHING ELSE")
    print("  Every byte claim in this study is an argument from SIGNATURES: a")
    print("  kernel can only touch buffers in its argument list, so the absence of")
    print("  a 67,108,864 B argument is a strong bound on traffic. It is a bound.")
    print("  dram__bytes.sum is the MEASUREMENT of the thing being bounded, and it")
    print("  is the only number in the study that can overrule the headline.")
    print()
    print("  A study whose central claim cannot be falsified by any measurement it")
    print("  could actually run is not a study; it is a position. This is the")
    print("  measurement that makes it a study, and it is cheap.")
    print()
    print("  It is ALSO the measurement the study already pre-registered. The")
    print("  band and the refutation threshold below are re-derived from shapes")
    print("  here, not copied. The point is not that the number is new -- it is")
    print("  that a falsification condition was written down in advance, with a")
    print("  threshold, and then never executed. A pre-registered refutation that")
    print("  was not run is a worse state than a missing measurement, because the")
    print("  missing measurement does not pretend to have been checked.")
    print()

    # ------------------------------------------------------------------ 2
    head("2. THE MODEL, AND THE BAND IT PREDICTS  [MODELLED]")
    kv("shape", f"B={BENCH['B']} T={BENCH['T']} H={BENCH['HV']} K={BENCH['K']} "
               f"V={BENCH['V']} C={BENCH['C']}  NT={m['NT']}")
    kv("one [B,NT,HV,C,C,K] fp32", f"{m['dec']:,} B")
    kv("decay traffic, 2 tensors x (1 write + 1 read)", f"{m['dec_traffic']:,} B")
    kv("8 non-decay intermediates, 2 accesses each", f"{m['inter']:,} elements")
    kv("everything else", f"{m['other']:,} B")
    kv("  of which compulsory I/O + state", f"{m['compulsory']:,} B")
    kv("model total", f"{m['total']:,} B = {m['total']/1e6:.2f} MB")
    kv("whole-operator fold ratio", f"{m['ratio']:.2f}x")
    kv("decay / compulsory", f"{m['dec_over_comp']:.2f}x")
    print()
    print(f"  PREDICTED BAND  [{m['band_lo']/1e6:.0f}, {m['band_hi']/1e6:.0f}] MB"
          f"   (total +/- 20%)")
    print(f"  REFUTATION THRESHOLD   < {m['kill']/1e6:.0f} MB   (20% of the "
          f"predicted total)")
    print(f"  the study's own harder threshold  < {m['kill_hard']/1e6:.0f} MB")
    print()
    print("  The model ASSUMES the decay reaches DRAM. b1 and b2 say it does")
    print("  not. So the measurement has two possible winners and that is the")
    print("  point: it is not a formality, it is the one thing that could make")
    print("  this note wrong.")
    print()

    # ------------------------------------------------------------------ 3
    head("3. THE EXACT COMMAND")
    print("  fp32. Not bf16. That is not a preference, it is the whole test: at")
    print(f"  bf16 one decay tensor is {m['dec']//2:,} B, which fits inside L2 on")
    print("  these cards, so a bf16 run would report a number that LOOKS like the")
    print(f"  refutation while being an artefact of the dtype. At fp32 it is")
    print(f"  {m['dec']:,} B, past L2, so the measurement is valid. A bf16 run")
    print("  cannot distinguish the two stories and must not be argued from.")
    print()
    print("  --replay-mode application, because dram__bytes_read/write are")
    print("  single-pass counters: ncu does not need to save and restore the")
    print("  kernel state 12,082 times, and kernel replay would cost hours for")
    print("  no information this metric contains.")
    print()
    for ln in cmd.splitlines():
        print("    " + ln)
    print()
    print("  Aggregate with:")
    print("    python3 -c \"import csv,collections;d=collections.Counter();"
          "[d.update({r['Metric Name']:float(r['Metric Value'].replace(',',''))})"
          " for r in csv.DictReader(open('ncu_kda_dram.csv'))];print(d)\"")
    print()

    # ------------------------------------------------------------------ 4
    head("4. WHAT IT COSTS  [INFERRED -- no run of it exists]")
    gpu = parse_us_us(require(T4_LOG), "GPU time")
    warm = parse_us(require(T4_REPLAY), "WARM")
    build = parse_us(require(T4_REPLAY), "3. BUILD only")
    sched = parse_us(require(T4_REPLAY), "=> schedule + launch + execute")
    kv("T4 event-bracketed GPU time, DEBUG=2", f"{gpu:,} us" if gpu else "n/a")
    kv("WARM wall, rebuild graph each call", f"{warm:,} ms" if warm else "n/a")
    kv("  of which BUILD only (no realize)", f"{build:,} ms" if build else "n/a")
    kv("  of which schedule+launch+execute", f"{sched:,} ms" if sched else "n/a")
    print()
    print("  Cost of the run, in order of size:")
    print(f"    rental          the only real cost. 1 A100/H100 for a couple of")
    print(f"                    hours; the whole battery this study would need to")
    print(f"                    produce one number fits in one session.")
    print(f"    CPU             the driver must still build and schedule the graph")
    print(f"                    ONCE: about {(build or 0)/1000:.0f} s of Python plus")
    print(f"                    about {(sched or 0)/1000:.0f} s of scheduler, from the")
    print(f"                    replay log, so well under a minute. Unchanged by ncu.")
    print(f"    device          {gpu/1000:.0f} ms of event-bracketed GPU time without")
    print(f"                    ncu, and dram__bytes is single-pass, so the same")
    print(f"                    order of magnitude WITH ncu. [INFERRED]")
    print(f"    ncu overhead    per-kernel instrumentation. At 12,082 launches this")
    print(f"                    is the term that actually decides the wall clock, and")
    print(f"                    it is UNKNOWN. Expect 1-10 us per launch, so")
    print(f"                    12-120 s of overhead, and treat the whole run as")
    print(f"                    minutes rather than seconds. [INFERRED]")
    print()
    print("  TOTAL: minutes of device time plus a rental. Not a week, not a")
    print("  rewrite. The reason the study is still without a cost number is not")
    print("  that this is expensive.")
    print()
    print("  IF the estimate is wrong and it takes hours: it is still the cheapest")
    print("  informative measurement available, and the fallback in section 6 is")
    print("  cheaper still.")
    print()

    # ------------------------------------------------------------------ 5
    head("5. THE DECISION EACH OUTCOME FORCES")
    dec = [
        ("in band",
         f"{m['band_lo']/1e6:.0f}-{m['band_hi']/1e6:.0f} MB",
         "the decay DOES reach DRAM",
         "then b1 and b2 are right about the signatures and wrong about the "
         "conclusion: the tensor is absent from the argument list and present in "
         "traffic. That is possible -- a kernel argument need not be a whole "
         "tensor, and a 12,082-kernel plan can stream a window of the decay "
         "repeatedly -- but it means the study's traffic story survives and the "
         "over-fusion reading is live again. The work then is an ncu "
         "per-kernel-name attribution to find WHICH kernels carry the traffic, "
         "and the next lever is still bandwidth, not fusion."),
        ("below the threshold",
         f"< {m['kill']/1e6:.0f} MB",
         "the decay does NOT reach DRAM",
         "the model is refuted, the byte model in notes/06 is refuted, and b1/b2 "
         "are confirmed by measurement rather than by argument. The study's "
         "headline stops being a memory claim and becomes a schedule claim: 12,082 "
         "launches, 2,172 exp2 in registers, and the fix is the scheduler or the "
         "API, not the traffic. Different owner, different pass, different "
         "estimate. This is the outcome the construction-level evidence favours "
         "and the one most worth wanting to know."),
        ("in between",
         f"{m['kill']/1e6:.0f}-{m['band_lo']/1e6:.0f} MB",
         "PARTIAL residency, and the pre-registered threshold was mis-set",
         "neither story is confirmed and the study has learned that its own "
         "falsification condition was not sharp. The fix is a sharper "
         "discriminator, not a rerun: attribute dram__bytes per kernel name "
         "(section 6) so the traffic is identified rather than totalled, and "
         "re-register the condition against THAT."),
    ]
    for name, band, meaning, then in dec:
        print(f"  {name.upper()}  {band}")
        print(f"    means: {meaning}")
        import textwrap
        for i, ln in enumerate(textwrap.wrap("then:  " + then, width=92,
                                            subsequent_indent="       ")):
            print("    " + ln)
        print()
    print("  In none of the three branches is the right response to stop. That is")
    print("  the property of a pre-registered condition worth having.")
    print()

    # ------------------------------------------------------------------ 6
    head("6. IF THE FIRST MEASUREMENT IS AMBIGUOUS, THE CHEAPER ONE")
    print("  A single total over 12,082 kernels cannot say WHICH kernel carries")
    print("  the traffic, and that attribution is what the 'in between' branch")
    print("  needs. It is also much cheaper, because --launch-count bounds it:")
    print()
    print("    ncu --csv --target-processes all --replay-mode application \\")
    print("      --metrics dram__bytes_read.sum,dram__bytes_write.sum \\")
    print("      --launch-count 512 --launch-skip 0 \\")
    print("      --log-file ncu_kda_first512.csv -- " + DRIVER.split("\n")[0])
    print()
    print("  512 of 12,082 launches, per kernel name, with the byte totals. That")
    print("  is 4.2% of the plan and it produces the attribution table, and it")
    print("  answers the 'which kernel' question on its own if the per-kernel")
    print("  distribution is strongly peaked -- which it should be, because the")
    print("  argument-size histogram has only four distinct values, and 8,181 of")
    print("  the plan's 52,724 argument slots share the largest one (1,048,576 B).")
    print()
    print("  THE SECOND MEASUREMENT, and it is the one that settles the other")
    print("  open question (is 12,082 launches expensive?):")
    print("    nsys profile --stats=true -t cuda " + DRIVER.split("\n")[0])
    print("  or, without a profiler, three lines: re-run kda/perf_kda.py with")
    print("  DEBUG=0 and time realize() with a single synchronize() at the end.")
    print("  If the wall time of the realize is near the event-bracketed figure")
    print("  the kernels are launch-bound; if it is an order of magnitude larger,")
    print("  the 215.9 ms was measuring the event wrapper. No rental needed beyond")
    print("  the one already required, and it is the difference between 'the")
    print("  operator is slow' and 'the operator is 12,082 launches'.")
    print()

    # ------------------------------------------------------------------ 7
    head("7. WHAT IS STILL MISSING AFTER BOTH")
    print("  Honest list, so the gap is not discovered later:")
    print("    * achieved wall clock for either construction. Needs a card that")
    print("      clears fla's gates, i.e. sm_80+ with >= 166,912 B. That is an")
    print("      A100 or better. Nothing about (b) changes; this is (a).")
    print("    * achieved tensor-core utilisation for either construction. The")
    print("      class matrix says what each can REACH; only a profiler says")
    print("      what each USES. `sm__pipe_tensor_op_hmma_cycles_active` and")
    print("      `sm__inst_executed_pipe_tensor` answer it on Hopper.")
    print("    * whether the hand-written reference's wgmma path is worth more")
    print("      than the 380 B spill costs it. That is a performance question")
    print("      and it is the one place where the (b) finding could cut BOTH")
    print("      ways: a baseline that spills at 12.5% occupancy is not a clean")
    print("      fast reference. Until it is timed, the (b) verdict is about")
    print("      REACHABILITY, not about merit.")
    print(f"    * occupancy and achieved bandwidth for tinygrad's own kernels. All")
    print(f"      {NOT_HERE} on this host; no toolkit, no card.")
    print()
    print("  None of the four is on the critical path for the (b) verdicts. They")
    print("  are the boundary between what this study can now say and what it")
    print("  cannot, and naming them is part of the result.")

    verdict("B6 VERDICT  one ncu invocation for dram__bytes.sum on one kda_chunk\n"
            "         call, in fp32, on a card this study can rent for an hour,\n"
            "         costs minutes of device time. It is the only measurement\n"
            "         that can overrule the note's own headline, and its result\n"
            "         has a pre-registered band and a pre-registered\n"
            "         refutation threshold. Its absence -- not its cost -- is why\n"
            "         the study has no cost number.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
