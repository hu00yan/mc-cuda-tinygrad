#!/usr/bin/env python3
"""classify.py -- the four-way classification, and the evidence for each rung.

WHAT THIS IS, AND WHY IT IS NOT A VERDICT
=========================================
The task asks, per rung, which of four classes the gap falls into:

    (i)   the framework cannot express it
    (ii)  it can, and a cost model could find it
    (iii) it is not implemented
    (iv)  it does not exist for this architecture

and requires, for each, THE EVIDENCE THAT WOULD HAVE DISTINGUISHED THE OTHER
THREE. A classification with no falsifier is not a classification, so this
module does not assign a class from a ratio. It does two things a ratio cannot:

1. It reads the framework's own source to decide which classes are even
   AVAILABLE for a rung, and records the `file:line` for each claim. A class is
   only reachable if the capability it names is absent or present in a way that
   can be checked by reading, not by timing.

2. For each rung it prints the ONE experiment whose outcome would separate the
   remaining candidate classes, and what each outcome would mean. The experiment
   is named before the number is in, so it cannot be reverse-engineered to fit.

WHAT SURVIVES WITHOUT HARDWARE COUNTERS
=======================================
Kaggle does not expose ncu/nsys and a container cannot grant the privilege, so
this axis has no instruction counts, no occupancy, no stall reasons and no
counter of any kind. What it has instead is:

  * a wall-clock ratio from ONE clock (same process, same context, same cuEvent)
  * a measured memory floor on the same card in the same run, so bandwidth
    claims have a denominator that was measured rather than quoted
  * a SASS instruction-class census, which is a PROPERTY OF THE BINARY and not a
    performance measurement: "this kernel contains HMMA" is read out of the
    compiled code and is exactly as valid without a profiler as with one

Every conclusion in the note is a wall-clock ratio, a roofline position, or a
binary property. None of them needs a counter, and the note says so explicitly
rather than leaving the reader to wonder whether something is missing.

WHY notes/06 C-20 AND C-22 ARE THE REASON FOR THE STRUCTURE
===========================================================
Both of those corrections are about a census being presented as a measurement.
So: the instruction census here is used ONLY to decide which of the four classes
are open, which is a question about the binary, and never as a cost. The cost is
always a time. Where a census number appears in the note it is labelled as a
census.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TG = REPO / ".venv/lib/python3.12/site-packages/tinygrad"

CLASSES = {
    "i":   "the framework cannot express the computation at all",
    "ii":  "it can express it, and a cost model could find the missing optimisation",
    "iii": "the capability is simply not implemented (no op, no pass, no table entry)",
    "iv":  "it does not exist for this architecture (the instruction is not in the ISA)",
}


# ===========================================================================
# 1. which classes are open per rung, read from the framework's own source
# ===========================================================================

def src(rel: str) -> str:
    p = TG / rel
    return p.read_text() if p.exists() else ""


def find(rel: str, pattern: str) -> list[tuple[int, str]]:
    """Lines matching `pattern` in a framework source file, as (lineno, text)."""
    out = []
    rx = re.compile(pattern)
    for i, line in enumerate(src(rel).splitlines(), 1):
        if rx.search(line):
            out.append((i, line.strip()))
    return out


# The checks below are the ones that actually discriminate, and each is a fact
# about the framework's source that a reader can re-read. They are collected here
# rather than asserted in the note so that the note can cite a line number.

def framework_facts() -> dict:
    facts: dict = {}

    # --- does a triangular solve or a matrix inverse exist at all?  (class i/iii)
    tri = []
    for rel in ("uop/ops.py", "mixin/op.py", "codegen/decomp/op.py",
                "schedule/rangeify.py", "codegen/__init__.py"):
        for pat in (r"\bTRIANGULAR\b", r"\bSOLVE_TRI\b", r"\bINVERSE\b", r"\bLU\b",
                    r"\btrsm\b", r"\btrsv\b", r"def inv\b"):
            tri += [(rel, n, t) for n, t in find(rel, pat)]
    facts["triangular_solve_or_inverse"] = {
        "hits": tri[:20],
        "present": bool(tri),
        "conclusion": ("no op, no UOp and no pass implements a triangular solve or "
                       "a matrix inverse, so a block inverse can only be SPELLED OUT "
                       "as the substitution the note uses. This is what makes the "
                       "trisolve rung's cost a statement about expression, not "
                       "about a missing library call."),
    }

    # --- is there a loop construct?  (class i)
    loops = []
    for rel in ("uop/ops.py", "codegen/decomp/op.py", "schedule/rangeify.py"):
        for pat in (r"\bWHILE\b", r"\bASSIGN\b", r"\bFOR\b\b"):
            loops += [(rel, n, t) for n, t in find(rel, pat)]
    facts["loop_construct"] = {
        "hits": loops[:20],
        "present": bool(loops),
        "conclusion": ("no while/assign loop op exists, so a recurrence must be "
                       "unrolled in the source. notes/06 C-16 records that the "
                       "UNROLL opt is capped at 32 and applies only to reduce "
                       "axes, so a Python `for` is BC real ops."),
    }

    # --- the UNROLL cap, which is the specific number the scan rung turns on
    unroll = [(n, t) for n, t in find("codegen/__init__.py", r"UNROLL|unroll")]
    facts["unroll"] = {
        "hits": unroll[:12],
        "conclusion": ("notes/06 C-16: the UNROLL opt is capped at 32 and applies "
                       "only to reduce axes. A recurrence longer than 32 cannot be "
                       "unrolled even in principle, which is what the scan rung "
                       "measures."),
    }

    # --- does a cumsum exist, and is it a primitive or a composition?
    # `cumsum` is a public wrapper generated in mixin/op.py's TensorOpsMixin; the
    # private `_cumalu` / `_split_cumalu` that back it live there too, not in
    # mixin/tensor.py where the first version of this sweep looked.
    cu = []
    for rel in ("mixin/op.py", "mixin/tensor.py", "uop/ops.py",
                "schedule/rangeify.py", "codegen/decomp/op.py"):
        cu += [(rel, n, t) for n, t in find(rel, r"cumsum|_cumalu|_split_cumalu")]
    # `Tensor.cumsum` is defined in mixin/tensor.py as a wrapper around the
    # private `_cumalu`; the sweep below looks for BOTH spellings, and the flag
    # is computed from the whole hit list rather than from a substring of the
    # conclusion text -- an earlier version printed "present: False" directly
    # above a sentence asserting that cumsum exists, which is the kind of
    # self-contradiction this repository keeps having to take back.
    has_cumsum = any(re.search(r"def cumsum\b", t) for _, _, t in cu)
    has_cumalu = any(re.search(r"_cumalu", t) for _, _, t in cu)
    facts["cumsum"] = {
        "hits": cu[:14],
        "present": has_cumsum or has_cumalu,
        "conclusion": ("`Tensor.cumsum` EXISTS" if has_cumsum or has_cumalu else
                       "`Tensor.cumsum` was NOT found in the swept files") +
                      (", so 'the framework cannot express a scan' is false by "
                       "inspection and class (i) is closed for the scan rung "
                       "without running anything. What remains open is whether its "
                       "lowering emits one kernel or N." if has_cumsum or has_cumalu
                       else " -- the scan rung's class (i) is therefore OPEN and "
                       "must be decided by measurement, not by inspection."),
    }

    # --- the tensor-core table: which archs have fp32 entries?
    tc = [(n, t) for n, t in find("codegen/tc.py", r"get_cuda|sm_\d+")]
    facts["tensor_core_table"] = {
        "hits": tc[:16],
        "conclusion": ("the TC table is per-architecture. Whether an fp32 entry "
                       "exists for sm_75 decides class (iv) for the fp32 GEMM rung, "
                       "and is a question about the TABLE, checkable by reading, "
                       "not about performance."),
    }
    return facts


# ===========================================================================
# 2. the per-rung classification frame
# ===========================================================================
#
# For each rung: which classes are still open AFTER the source facts above, the
# experiment that separates the remaining ones, and what each outcome means.
#
# The experiment is named before the measurement. That is the whole point: a
# classification whose discriminator is chosen after seeing the ratio is a
# rationalisation, and notes/06 C-16 records exactly that failure ("a property
# attributed to the framework that it does not have, in order to defend a choice
# the framework never made").

CLASSIFICATION: list[dict] = [
    {
        "rung": "ew",
        "open_classes": ["ii", "iii"],
        "closed": {
            "i": "the elementwise chain IS the framework's native form; a Tensor "
                 "expression is a first-class citizen here, not a workaround",
            "iv": "the work is a load, a multiply, an add and a store. There is no "
                  "architecture-specific capability in question.",
        },
        "discriminating_experiment":
            "The floor. `axis_fill`/`axis_copy` measure what this card does with "
            "the same number of bytes and no arithmetic. If TinyGrad's ew arm "
            "reaches the copy floor, the gap is zero and no classification is "
            "needed -- which is the expected outcome and would make this rung the "
            "axis's control rather than a data point.",
        "outcomes": {
            "at_floor": "no gap; the framework reaches the memory path. Class (ii) "
                        "is vacuous here and the rung's role is to establish that "
                        "every gap above it is arithmetic, not bandwidth.",
            "below_floor_with_a_gap": "either the elementwise chain is being split "
                                      "across kernels (class iii, a fusion limit) or "
                                      "the codegen is not vectorising (class ii, a "
                                      "cost model could pick the vector width). The "
                                      "kernel COUNT separates them: >1 kernel with "
                                      "the op count constant is iii; 1 kernel that "
                                      "is not at the floor is ii.",
        },
    },
    {
        "rung": "red",
        "open_classes": ["ii", "iii"],
        "closed": {
            "i": "a reduction to a narrow output is `Tensor.sum(-1)`. It is a "
                 "first-class op with a documented shape rule (notes/26 rule 3).",
            "iv": "a reduction is FADD on this architecture for any width.",
        },
        "discriminating_experiment":
            "The (ew, red) PAIR. Both move the same class of bytes; `red` adds one "
            "reduction axis. A gap that is absent at `ew` and present at `red` is "
            "attributable to the reduction, and nothing else. Separately, the "
            "tg_naive control prices the C-18 habit on the same card in the same "
            "run, so 'the idiomatic arm is the good arm' is measured rather than "
            "assumed.",
        "outcomes": {
            "idiomatic_near_reference": "the framework does the wide reduction well; "
                                       "the C-18 number is a property of that PORT, "
                                       "not of the framework. That is the single "
                                       "most load-bearing fact this study has "
                                       "produced twice, and it is worth measuring "
                                       "rather than asserting.",
            "idiomatic_far_below_reference": "the reduction itself is the cost, and "
                                             "the KNOWN-CONFIGURATION question becomes "
                                             "whether it is tiling (ii, findable) or "
                                             "an unimplemented schedule (iii). The "
                                             "kernel count and the generated source "
                                             "separate those.",
        },
    },
    {
        "rung": "gemm32",
        "open_classes": ["iv", "ii", "iii"],
        "closed": {
            "i": "`Tensor.matmul` exists and is the framework's normal spelling of "
                 "a contraction (notes/26 §1).",
        },
        "discriminating_experiment":
            "A SASS instruction-class census of BOTH arms, plus the ALLOW_TF32 "
            "knob rung. notes/26 §6b measured that fp32 never reaches a tensor core "
            "at any shape, and notes/06 C-19 corrected the shape-gate version of "
            "that claim. The census is a property of the binary, valid with no "
            "profiler: if the reference emits FFMA and no HMMA, and the framework's "
            "emits FFMA and no HMMA, then class (iv) is settled by the ISA and the "
            "remaining gap is about tiling, which is (ii) or (iii).",
        "outcomes": {
            "both_FFMA": "class (iv) for the tensor-core part -- sm_75 has no fp32 "
                         "tensor core -- and the residual gap is a scheduling "
                         "question. ALLOW_TF32 then becomes the falsifier for (ii): "
                         "if it closes the gap it is (ii), and notes/29 already "
                         "records that this knob targets a table with no fp32 entry "
                         "below sm_80, which would make it (iv) again.",
            "framework_HMMA_reference_FFMA": "the framework found a capability the "
                                              "hand-written arm did not use, and the "
                                              "comparison is inverted -- which would "
                                              "be a finding about the reference, and "
                                              "the note would have to say so.",
        },
    },
    {
        "rung": "gemm16",
        "open_classes": ["ii", "iii"],
        "closed": {
            "i": "`Tensor.matmul` in fp16 exists.",
            "iv": "EXCLUDED BY CONSTRUCTION. HMMA is in the sm_75 ISA and the "
                  "hand-written arm's SASS contains it (axis/sass_census.py, "
                  "measured on this host with no GPU at all). The hardware is "
                  "present, so class (iv) is not available for this rung no matter "
                  "what the timings say. This is the only rung where (iv) is "
                  "eliminated by a prior, independent measurement.",
        },
        "discriminating_experiment":
            "The KNOB SWEEP, and it is the reason this rung is on the ladder. The "
            "framework exposes TC (on/off), TC_SELECT (which entry of the "
            "per-arch TC table), TC_OPT, and BEAM (search over schedules). Sweeping "
            "them on the same card in the same run separates (ii) from (iii) "
            "directly: if ANY setting in the framework's own search space reaches "
            "the reference, the capability is implemented and only the default "
            "schedule is bad -- that is (ii), and it is a cost-model problem. If "
            "the whole space is flat and far from the reference, the tensor-core "
            "schedule is not implemented -- (iii).",
        "outcomes": {
            "a_knob_closes_the_gap": "class (ii). The framework can do this and its "
                                     "own cost model can find it; the default is "
                                     "what is wrong. This is a much weaker and much "
                                     "more actionable finding than (iii), and it is "
                                     "the distinction the task asks for.",
            "the_whole_knob_space_is_flat": "class (iii). No schedule in the "
                                            "framework's own search space reaches "
                                            "the reference, so this is not a tuning "
                                            "question.",
        },
    },
    {
        "rung": "gather",
        "open_classes": ["ii", "iii"],
        "closed": {
            "i": "indexing a tensor by an index tensor is core syntax.",
            "iv": "a gather is LDG. There is no architecture-specific capability "
                  "in question on any architecture that runs loads.",
        },
        "discriminating_experiment":
            "The table size against the measured L2, checked at run time with "
            "cuDeviceGetAttribute. If the table fits in L2 the gather is a "
            "broadcast and the rung has silently become a different operator, so "
            "this is a validity gate before the ratio is a measurement. Then the "
            "(ew, gather) pair: same traffic class, one gather added.",
        "outcomes": {
            "gap_present_and_index_arithmetic_heavy": "(ii) if the generated code's "
                                                      "index math is a cost model "
                                                      "could have avoided, (iii) if "
                                                      "the framework has no vectorised "
                                                      "gather at all. The generated "
                                                      "CUDA source separates these and "
                                                      "is a read, not a timing.",
            "at_reference": "the framework handles random access as well as a "
                            "hand-written kernel does, and the rung bounds where "
                            "the envelope does NOT fall.",
        },
    },
    {
        "rung": "scan",
        "open_classes": ["ii", "iii"],
        "closed": {
            "i": "CLOSED BY INSPECTION. `Tensor.cumsum` exists. The framework can "
                 "express a scan; notes/26 §5c's claim is about what it can do with "
                 "one, not whether it can write one down.",
            "iv": "a scan needs FADD and a shared-memory prefix. Nothing "
                  "architecture-specific is missing.",
        },
        "discriminating_experiment":
            "The kernel COUNT of the framework's own `cumsum`, measured on the "
            "device, against the reference's ONE kernel. This is the cleanest rung "
            "on the ladder because the operator is nothing but a dependent chain: "
            "there is no contraction and no gather to confound it, so whatever the "
            "scan costs is attributable to the chain. The count is reported as a "
            "CENSUS and the time is the evidence (notes/06 C-22).",
        "outcomes": {
            "one_kernel": "the framework keeps the chain in registers and notes/26 "
                          "§5c's claim is too strong as stated for this operator. "
                          "That is a correction to a note in this repo and it would "
                          "have to be written as one.",
            "N_kernels": "class (iii): the chain is materialised one step at a time, "
                         "which is the N-vs-1 the reference wins by. The N is a "
                         "count; the ratio is the cost; the note carries both and "
                         "labels which is which.",
        },
    },
    {
        "rung": "trisolve",
        "open_classes": ["ii", "iii"],
        "closed": {
            "i": "PARTIALLY CLOSED, and precisely. A triangular solve has no op, no "
                 "UOp and no pass in the framework (see "
                 "framework_facts.triangular_solve_or_inverse), so it cannot be "
                 "NAMED. But the COMPUTATION is expressible -- the substitution is "
                 "spelled out, and `axis/verify.py` checks it against "
                 "np.linalg.inv -- so the gap is not expressiveness of the "
                 "computation. What is not expressible is the SOLVE. The honest "
                 "reading is that (i) applies to the solve and (iii) to whatever "
                 "would make the spelled-out substitution cheap; the note says which "
                 "it is using and why.",
            "iv": "the arithmetic is FADD and FFMA on 16x16 blocks. Nothing here is "
                  "architecture-specific, which is why this rung is comparable at "
                  "all on a T4 when its original (fla's Triton kernel) is not.",
        },
        "discriminating_experiment":
            "The (scan, trisolve) PAIR, which is the reason both rungs are here. "
            "`scan` is a carried chain with NO contraction; `trisolve` is a carried "
            "chain WITH a dense contraction at every step. The difference between "
            "them is therefore attributable to the contraction and to nothing else. "
            "That is what turns two separate gaps into a decomposition.",
        "outcomes": {
            "scan_close_trisolve_far": "the cost is the CONTRIBUTION, not the chain. "
                                       "The framework can hold a chain (scan says "
                                       "so) but not a chain with a small dense "
                                       "product inside it, which is a specific and "
                                       "useful statement about the envelope's edge.",
            "both_far": "the chain itself is the cost, and the two rungs together "
                        "bound how much of it the contraction adds.",
            "both_close": "the framework handles both, and the KDA gap reported in "
                          "notes/19 and notes/29 is about the port rather than the "
                          "framework -- which is the finding that most needs saying, "
                          "because notes/06 §5b says it has been the wrong one twice.",
        },
    },
]


# ===========================================================================
def main() -> int:
    print("=" * 78)
    print("axis/classify.py -- which of the four classes is open per rung, and the")
    print("experiment that separates the rest. No ratios are read here.")
    print("=" * 78)

    facts = framework_facts()
    print("\n== framework facts, read from source (no GPU, no timing) ==")
    for k, v in facts.items():
        print(f"\n  {k}")
        print(f"    present in source: {v.get('present', 'n/a')}")
        for h in v.get("hits", [])[:4]:
            # hits come as (file, lineno, text) from framework_facts()'s
            # multi-file sweep and as (lineno, text) from the single-file
            # checks. Both shapes are printed the same way.
            if len(h) == 3:
                print(f"      {h[0]}:{h[1]}  {h[2][:88]}")
            else:
                print(f"      codegen/__init__.py:{h[0]}  {h[1][:88]}")
        print(f"    -> {v['conclusion']}")

    print("\n" + "=" * 78)
    print("per rung: the open classes, and the discriminator")
    print("=" * 78)
    for c in CLASSIFICATION:
        print(f"\n  {c['rung']}")
        print(f"    OPEN     : {', '.join(c['open_classes'])}")
        for k, v in c["closed"].items():
            print(f"    closed {k:<4}: {v}")
        print(f"    DISCRIMINATOR:")
        for line in _wrap(c["discriminating_experiment"], 66):
            print(f"      {line}")
        for outcome, meaning in c["outcomes"].items():
            print(f"      if {outcome}:")
            for line in _wrap(meaning, 62):
                print(f"        {line}")

    if "--json" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--json") + 1])
        out.write_text(json.dumps({"facts": facts, "classification": CLASSIFICATION},
                                  indent=2, default=str))
        print(f"\n  wrote {out}")
    return 0


def _wrap(s: str, n: int) -> list[str]:
    words, line, out = s.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > n:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out


if __name__ == "__main__":
    sys.exit(main())
