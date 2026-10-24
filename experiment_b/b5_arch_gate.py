#!/usr/bin/env python3
"""b5_arch_gate.py -- why comparison (a) is invalid, and why (b) is not.

TWO COMPARISONS, AND THE STUDY HAS BEEN RUNNING THE IMPOSSIBLE ONE
----------------------------------------------------------------
  (a)  same ALGORITHM on both sides.  The gap is codegen and scheduling
       quality.  Requires running both, on the same card, and timing them.
  (b)  the human chose a BETTER ALGORITHM and the compiler cannot reach it.
       Not a speed race.  Asks whether the compiler can arrive at the human's
       CONSTRUCTION at all.  The quantities are bytes materialised and
       instruction classes emitted, and both are device-free.

(a) is invalid on every machine this study can reach, and no amount of local
work changes that. This file re-derives the three gates that make it invalid,
by reading the vendored baseline's own source rather than a notes file:

  GATE 1  the baseline needs sm_80+. `check_shared_mem('ampere')` is
          `max_shared_memory >= 166912` (Backend.AMPERE). The only card this
          study ran on is a T4 at 65,536 B, so the check is False and the
          config space collapses: BV_LIST, BK_LIST and num_stages all narrow.

  GATE 2  on that card the dots stop being tensor-core dots.
          `IS_TF32_SUPPORTED` is `capability[0] >= 8`, so on sm_75 it is False,
          and `fla/utils/_device.py` then sets a PROCESS-GLOBAL
          `TRITON_F32_DEFAULT = 'ieee'`. On an sm_75 the "hand-written GPU
          kernel" being compared is running IEEE fp32 dot products.

  GATE 3  on consumer Ada the config space is also collapsed, for a different
          reason: Ada's 101,376 B is above the DEFAULT threshold and below
          AMPERE's, so `num_stages` is [2,1] -- and num_stages=1 disables the
          software pipeline on a kernel that streams a [B,HV,K,V] state.

The thresholds are re-derived from the source and evaluated against each card's
opt-in shared-memory figure. The card figures are datasheet values and are
labelled as such; the ARITHMETIC is done here.

AND THE OTHER HALF, WHICH IS THE POINT
--------------------------------------
The reason (b) survives is that the compiler's ceiling is a TABLE, and the table
is readable. `tinygrad/codegen/opt/tc.py:get_cuda` returns `cuda_sm89` for any
arch >= 89 and cuda_sm89 is `cuda_sm80 + fp8` -- an Ada catalog. There is no
sm_90 row and no sm_100 row. And `tinygrad/runtime/ops_cuda.py:122` builds
`sm_{major}{minor}` with no arch suffix, because the table does
`int(arch[3:])` and would raise on `sm_90a`.

So on a real H100 the backend reports arch `sm_90`, the table returns the Ada
catalog, and the largest tensor-core shape available is (N=8, M=16, K=32) --
`mma.sync.m16n8k32`, an Ada/Hopper-consumer instruction. There is no wgmma
shape and no tcgen05 shape in the catalog to select. This file EXECUTES
get_cuda at each arch and prints the identity of the returned list, so the
claim is a measurement of the installed compiler rather than a reading of it.

    python3 experiment_b/b5_arch_gate.py
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment_b.common import (  # noqa: E402
    ARTIFACT, INFERRED, MEASURED, MODELLED, NOT_HERE, RE_DERIVED, head, kv,
    read, require, rule, verdict)

FLA_DEV = "portability/subj/fla/flash-linear-attention-main/fla/utils/_device.py"
FLA_H = ("portability/subj/fla/flash-linear-attention-main/fla/ops/common/"
         "chunk_delta_h.py")
TC = "tinygrad/codegen/opt/tc.py"
OPS_CUDA = "tinygrad/runtime/ops_cuda.py"

# opt-in shared memory per block, B. NVIDIA's published figures; [ARTIFACT:
# datasheet], not measured here and not checkable on this host. The COMPARISONS
# against them are the re-derivation.
CARDS = [
    ("Tesla T4",        7.5,  65536, "the only card this study has actually run on"),
    ("RTX 2080 Ti",     7.5,  65536, "also Turing, same gate"),
    ("RTX 3090",        8.6,  101376, "Ampere consumer: passes the DEFAULT gate"),
    ("RTX 4090",        8.9,  101376, "Ada consumer: the counterexample that matters"),
    ("A100 80GB",       8.0, 167936, "the first card that clears AMPERE"),
    ("H100 SXM",        9.0, 232448, "Hopper"),
    ("B200",           10.0, 232448, "Blackwell datacenter"),
    ("RTX 5090",       12.0, 101376, "Blackwell consumer"),
]


# =============================================================================
# 1. read the baseline's own gates
# =============================================================================

def parse_fla_gates() -> dict:
    dev = read(FLA_DEV)
    h = read(FLA_H)
    out = {}
    out["ampere"] = int(re.search(r"AMPERE\s*=\s*(\d+)", dev).group(1))
    out["ada"] = int(re.search(r"ADA\s*=\s*(\d+)", dev).group(1))
    out["hopper"] = int(re.search(r"HOPPER\s*=\s*(\d+)", dev).group(1))
    out["default"] = int(re.search(r"DEFAULT\s*=\s*(\d+)", dev).group(1))
    out["tf32_rule"] = re.search(
        r"IS_TF32_SUPPORTED\s*=\s*(.+)", dev).group(1).strip()
    out["ieee_setter"] = bool(re.search(
        r"os\.environ\['TRITON_F32_DEFAULT'\]\s*=\s*'ieee'", dev))
    out["ieee_guard"] = re.search(
        r"if IS_NVIDIA and not IS_TF32_SUPPORTED:.*?"
        r"os\.environ\['TRITON_F32_DEFAULT'\]\s*=\s*'ieee'", dev, re.S) is not None
    out["smem_check_body"] = re.search(
        r"def check_shared_mem.*?return max_shared_memory >= "
        r"Backend\.get_shared_memory\(arch\)", dev, re.S) is not None
    # the consumer-side narrowing, read out of the kernel's own autotune list
    ns = re.findall(r"for num_stages in \(\[([\d, ]+)\] if "
                    r"check_shared_mem\('?(\w+)'?\) else \[([\d, ]+)\]\)", h)
    out["num_stages_gates"] = ns
    bv = re.findall(r"for BV in \(\[([\d, ]+)\] if check_shared_mem\('?(\w+)'?\) "
                    r"else \[([\d, ]+)\]\)", h)
    out["bv_gates"] = bv
    return out


# =============================================================================

def main() -> int:
    head("B5. THE ARCH GATES  --  why (a) is invalid here and (b) is not")
    kv("what this file is", "an argument about WHICH COMPARISON is available, "
                            "not a performance result")
    print()

    # ------------------------------------------------------------------ gates
    head("1. THE BASELINE'S OWN GATES, READ OUT OF ITS SOURCE")
    g = parse_fla_gates()
    kv(f"{FLA_DEV}", "pinned")
    kv("Backend.AMPERE (the sm_80 gate)", f"{g['ampere']:,} B")
    kv("Backend.HOPPER", f"{g['hopper']:,} B")
    kv("Backend.ADA", f"{g['ada']:,} B")
    kv("Backend.DEFAULT", f"{g['default']:,} B")
    kv("check_shared_mem is a >= on opt-in smem", g["smem_check_body"])
    kv("IS_TF32_SUPPORTED", g["tf32_rule"])
    kv("process-global TRITON_F32_DEFAULT='ieee' set under that guard", g["ieee_guard"])
    print()
    print("  the config-space gates, in the kernel's own autotune list:")
    for full, arch, narrow in g["num_stages_gates"]:
        print(f"    num_stages  [{full.strip()}]  if smem >= {arch.upper()}={g.get(arch.lower(), 102400):,}  else [{narrow.strip()}]")
    for full, arch, narrow in g["bv_gates"]:
        print(f"    BV          [{full.strip()}]  if smem >= {arch.upper()}={g.get(arch.lower(), 102400):,}  else [{narrow.strip()}]")
    print()
    print("  Re-deriving them by hand, so the source read is a check and not the")
    print("  argument:")
    print(f"    num_stages  = [2,3,4] if smem >= {g['ampere']:,}  else [2,1]")
    print(f"    BV          = [32,64] if smem >= {g['ada']:,}  else [32]")
    print(f"    tf32 dots   = capability.major >= 8, else the fla module sets")
    print(f"                  TRITON_F32_DEFAULT='ieee' process-globally, which")
    print(f"                  makes every tl.dot an IEEE fp32 dot.")
    print()

    # ------------------------------------------------------------------ cards
    head("2. WHICH CARDS SURVIVE THE GATES")
    print(f"  {'card':<14}{'cc':>5}{'opt-in smem B':>15}{'clears AMPERE':>16}"
          f"{'clears ADA':>13}{'tf32 dots':>11}   verdict for comparison (a)")
    rule("-", 104)
    rows = []
    for name, cc, smem, note in CARDS:
        amp = smem >= g["ampere"]
        ada = smem >= g["ada"]
        tf32 = cc >= 8.0
        if not amp:
            say = "INVALID: baseline is not fla"
        elif not tf32:
            say = "INVALID: IEEE fp32 dots"
        else:
            ns = "[2,3,4]" if amp else "[2,1]"
            say = f"usable (num_stages {ns})"
        rows.append((name, cc, smem, amp, ada, tf32, say, note))
        print(f"  {name:<14}{cc:>5}{smem:>15,}{str(amp):>16}{str(ada):>13}"
              f"{str(tf32):>11}   {say}")
    print()
    t4 = rows[0]
    print("  The study's only card:")
    print(f"    {t4[0]} cc {t4[1]}, {t4[2]:,} B opt-in shared memory")
    print(f"    clears AMPERE ({g['ampere']:,} B): {t4[3]}")
    print(f"    -> num_stages = [2,1], BV = [32], BK/BV_LIST = [16,32], and")
    print(f"       TRITON_F32_DEFAULT='ieee'")
    print("    -> the 'hand-written GPU kernel' in the T4 cells of this study is")
    print("       fla running on Turing, in IEEE fp32, with a collapsed config")
    print("       space. It is fla-on-Turing, not fla.")
    print()
    ada = next(r for r in rows if r[0] == "RTX 4090")
    print("  The counterexample that makes this a schedulability problem rather")
    print("  than a hardware problem. A current-generation consumer card, sold as")
    print("  a fast card, with:")
    print(f"    opt-in shared memory  {ada[2]:,} B")
    print(f"    clears ADA     ({g['ada']:>9,}) -> {ada[4]}, so BV = [32,64]")
    print(f"    clears DEFAULT ({g['default']:>9,}) -> {ada[2] >= g['default']}")
    print(f"    clears AMPERE  ({g['ampere']:>9,}) -> {ada[3]}, so num_stages = [2,1]")
    print("  and num_stages=1 disables the software pipeline on a kernel that")
    print("  streams a [B,HV,K,V] state across the chunk axis. So the card that")
    print("  looks like the obvious upgrade from a T4 is the one card in this")
    print("  table where the baseline's config space is silently halved, and no")
    print("  datasheet says so. Reading one property off the device does.")
    print()

    # ------------------------------------------------- the compiler's ceiling
    head("3. THE COMPILER'S CEILING IS A TABLE, AND THE TABLE IS READABLE")
    from tinygrad.codegen.opt import tc
    kv(f"{TC}", "pinned")
    src = read(TC)
    line = re.search(r"def get_cuda\(arch\).*", src).group(0)
    kv("get_cuda, verbatim", line)
    print()
    print(f"  {'arch string':<14}{'catalog returned':<18}{'n shapes':>10}"
          f"{'max (N,M,K)':>15}   emitted instruction")
    rule("-", 92)
    for a in ("sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_100", "sm_120"):
        cat = tc.get_cuda(a)
        ident = ("cuda_sm89" if cat is tc.cuda_sm89 else
                 "cuda_sm80" if cat is tc.cuda_sm80 else
                 "cuda_sm75" if cat is tc.cuda_sm75 else "?")
        mx = max((t.dims for t in cat), default=(0, 0, 0))
        insn = f"mma.sync.m{mx[1]}n{mx[0]}k{mx[2]}"
        print(f"  {a:<14}{ident:<18}{len(cat):>10}{str(mx):>15}   {insn}")
    try:
        tc.get_cuda("sm_90a")
        raised = False
    except ValueError as e:
        raised, why = True, str(e)
    print()
    print(f"  get_cuda('sm_90a') raises ValueError: {raised}  ({why if raised else ''})")
    print("  -> so ops_cuda.py cannot pass an arch-specific string at all, which is")
    print("     why it strips one. Read it:")
    oc = read(OPS_CUDA)
    m = re.search(r"^.*arch=f\"sm_\{major.*$", oc, re.M)
    kv("the line that builds the arch string", m.group(0).strip() if m else "?")
    print()
    print("  THE CONSEQUENCE, stated as a prediction the table makes:")
    print("    on a real H100, CUDADevice computes cc (9,0), so arch == 'sm_90'.")
    print("    get_cuda('sm_90') returns cuda_sm89 -- the Ada catalog -- whose")
    print("    largest shape is (N=8, M=16, K=32), i.e. mma.sync.m16n8k32.")
    print("    There is no wgmma (m64nNk16) shape and no tcgen05 shape in the")
    print("    catalog to select, so no amount of scheduling on that card can")
    print("    produce one. [MEASURED on the installed compiler, by calling")
    print("    get_cuda; the 'on a real H100' half is a reading of one line of")
    print("    source and is labelled as such.]")
    print()
    print("  b4 confirms the consequence at the output level: the whole rendered")
    print("  program is byte-identical across sm_80..sm_100 and contains zero")
    print("  wgmma, tcgen05, cp.async, stmatrix, ldmatrix, mbarrier, tensormap.")
    print()

    # ------------------------------------------------------------------ verdict
    head("4. WHICH COMPARISON IS AVAILABLE, THEN")
    rowsx = [
        ("(a) same algorithm, timed", "NEEDS A CARD",
         "the T4 fails fla three separate gates; an H100 would clear them but the "
         "study has none, and the comparison would then be valid"),
        ("(b) does the compiler reach the human's construction",
         "AVAILABLE NOW",
         "bytes materialised = the schedule census (b2); instruction classes = "
         "the renderer plus a pinned PTX (b4). Neither needs a card."),
    ]
    print(f"  {'comparison':<48}{'status':<14}why")
    rule("-", 104)
    for a, s, w in rowsx:
        print(f"  {a:<48}{s:<14}{w}")
    print()
    print("  (a) is not 'hard'. It is INVALID on the hardware this study has, for")
    print("  reasons in the baseline's own source. That is a categorically")
    print("  different thing from hard, and treating them as the same is how the")
    print("  study ended up calling itself blocked when it had already measured")
    print("  the only part of the question that is measurable.")
    print()
    print("  (b) is available NOW, on this host, and is what notes/15 is about.")
    print(f"  It is also {INFERRED}-free: everything in it is either MEASURED on a")
    print("  pinned artifact, or RE-DERIVED from shapes and from source text.")
    print()

    head("5. THE EDGES")
    print("  * The card figures in section 2 are datasheet values, labelled")
    print("    [ARTIFACT]. The COMPARISONS against the fla thresholds are")
    print("    re-derived here; the thresholds themselves are read from the pinned")
    print("    source. No card was measured.")
    print("  * The register-file size used in b4's occupancy arithmetic is also a")
    print("    datasheet value, and is labelled there.")
    print("  * The spill figures for the hand-written reference (380 B stores,")
    print("    372 B loads, 376 B frame, 255 registers at sm_90a) are quoted from")
    print("    a recorded ptxas run. ptxas is not installed on this host, so")
    print("    `b4_codegen.py --verify-spill` exits nonzero rather than letting")
    print(f"    them pass as verified. Marked {NOT_HERE} to re-derive.")
    print("  * `TRITON_F32_DEFAULT` is read from fla 0.5.2's source as vendored")
    print("    in this repo. The wheel and the git tree in portability/subj/fla")
    print("    are two copies; only the tree is pinned. If they diverge, the")
    print("    threshold arithmetic is still right and the line numbers are not.")
    print("  * None of this says which construction is FASTER. See notes/15 §3.")

    verdict("B5 VERDICT  comparison (a) is invalid on every card this study can\n"
            "         reach, for three reasons that are in the baseline's own\n"
            "         source: sm_80's 166,912 B shared-memory gate, the\n"
            "         process-global TRITON_F32_DEFAULT='ieee' on pre-Ampere, and\n"
            "         the num_stages=[2,1] collapse that Ada hits too. Comparison\n"
            "         (b) needs none of it, and is settled by b2 and b4.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
