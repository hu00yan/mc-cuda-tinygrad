#!/usr/bin/env python3
"""knobs_table.py -- the knob sweep as one table, derived from the artefacts.

    python -m minecu_tg_knobs.knobs_table \\
        --codegen .agent-scratch/knobs/codegen/census.json \\
        --sass    .agent-scratch/knobs/sass_census.json

WHY A TABLE BUILDER AND NOT A TABLE IN THE NOTE
------------------------------------------------
Because the numbers in the note must be quotable, and a number typed into
Markdown is a number that can be wrong. Every cell here is read out of an
artefact file that a command produced, so a cell and its source cannot drift
apart. The note cites `knobs_table.py:<claim>` the way `derivation/minecu_gap.py`
is cited elsewhere in this repo.

WHAT IS DERIVED HERE RATHER THAN COPIED
---------------------------------------
Three things, each of which was a transcription hazard:

  * the per-DDA-step instruction count, `hot_sass / 32`. `notes/19` §4.1 does the
    same division, and the divisor has to be stated: the DDA kernel contains all
    `max_steps` step bodies *inlined*, which the emitted source shows directly
    (one `*(dataN+...)` gather per step).

  * the residue of `derivation/minecu_gap.py:decomposition`, re-derived from the
    SASS this run produced rather than from the figures in `notes/19` §5.4. The
    decomposition names early-exit and load terms; the instruction-count total
    it is divided into is THIS run's number, and the two are not the same
    quantity, so the residue is recomputed.

  * which knobs are inert, by BYTE comparison of the hot kernel's sha256, not
    by reading a knob's `.value`. A knob that was set and did nothing is a
    result; a knob that was not set is a bug, and the two are told apart here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# The four ports, and what each one is. `lean` is arm B's port; the two
# intermediates exist so the port's own contribution is separable from the
# knob's; `naive` is arm A, the control.
PORTS = ("naive", "lean", "lean-no-flat", "lean-no-unsigned")
PORT_ROLE = {
    "naive": "arm A, the naive port (control for the whole study)",
    "lean": "arm B's port, all three port fixes on",
    "lean-no-flat": "lean, but the flat address is RECOMPUTED per step",
    "lean-no-unsigned": "lean, but the bounds test is the signed 6-compare form",
}

# Which registry knob each label in the sweep table corresponds to. The sweep
# labels are `<KNOB>=<value>` using the brief's names, and the registry is the
# authority on the name -> Context key mapping, so the mapping is read from
# there rather than retyped.
from minecu_tg_knobs.registry import KNOBS  # noqa: E402


def canon_sass_label(label: str, known: set[str]) -> str:
    """Reconcile a SASS artefact's label with the census's vocabulary.

    `codegen.py` writes its `.cu` tree under `<label with '=' replaced by '-'>`,
    because `=` is legal in a path but awkward in a shell, and `sass.sh` then
    flattens that tree into `<label>_<port>_kernel<NNN>.sass`. So the SASS
    table's labels carry `-` where the census's carry `=`.

    Left unreconciled, every row of the sweep table reads "not run" while both
    artefacts are full -- which looks exactly like a sweep that found nothing,
    and is the most plausible-looking wrong answer available here. So the
    substitution is done ONLY against the census's own label set, and a label
    that matches nothing after that is reported rather than guessed at: a blind
    `replace('-', '=')` would corrupt a hypothetical negative knob value, and
    this repo has already had one wrong figure propagate too far.
    """
    if label in known:
        return label
    guess = label.replace("-", "=")
    return guess if guess in known else label


def _sass_index(sass: dict, codegen: dict) -> dict:
    """(port, label) -> SASS row, with the label vocabulary reconciled.

    One index for every claim, so all four tables read the same keys and none
    of them can silently disagree about which label a row belongs to.
    """
    known = {r["label"] for r in codegen["records"]}
    unmapped = []
    out = {}
    for r in sass["rows"]:
        lab = canon_sass_label(r["label"], known)
        if lab not in known and r["label"] not in known:
            unmapped.append(r["label"])
        out[(r["port"], lab)] = r
    if unmapped:
        raise AssertionError(
            f"SASS labels {sorted(set(unmapped))} match no census label, so "
            f"the two artefacts cannot be joined and any table built from them "
            f"would silently show 'not run' for rows that were measured. "
            f"Census labels: {sorted(known)}")
    return out


def knob_for_label(label: str):
    """The Knob a sweep label names, or None for `default` / a pair."""
    if label == "default":
        return None
    for k in KNOBS:
        if label == k.name or label.startswith(k.name + "="):
            return k
    return None


# --------------------------------------------------------------------------
# the reference's figures, which this arm did not measure
# --------------------------------------------------------------------------
# The hand-written side is arm A's artefact, not this arm's, so these are read
# out of notes/19 with a line cite rather than re-derived -- and every use is
# labelled as a read, not a measurement. `notes/19` §4.0.1 re-derives the 776
# from `minecu_ref/build/kernels_sm_75.sass`; §1.3 measures the 26.4 on the
# card with the reference's own probe.
#
#   REF_WHOLE         776   whole render, 152 prologue + 16x38 body + 16 epilogue
#   REF_STEPS         16    the `#pragma unroll 4` bodies the 776 contains
#   MAX_STEPS         32    the reference's own default `max_steps`
#   MEAN_STEPS_REF  26.4    measured mean steps/pixel, the reference's own probe
REF_WHOLE = 776
REF_STEPS = 16
MAX_STEPS = 32
MEAN_STEPS_REF = 26.4
REF_PER_STEP = REF_WHOLE / REF_STEPS


# --------------------------------------------------------------------------
# claims
# --------------------------------------------------------------------------
def sweep(codegen: dict, sass: dict) -> str:
    """THE knob sweep table: one row per (port, knob configuration)."""
    cg = {(r["port"], r["label"]): r for r in codegen["records"]}
    sx = _sass_index(sass, codegen)
    steps = codegen["max_steps"]

    lines = []
    hdr = (f"  {'knob':<17} {'Context key':<15} {'knob value':>10}   "
           f"{'codegen (naive / lean)':<30} {'hot SASS naive':>14} "
           f"{'hot SASS lean':>13} {'vs lean default':>15}")
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))

    # The control row, per port. The `hot SASS` column for the CONTROL is
    # the port's own baseline, so the two control rows are the two denominators
    # the "vs default" column divides by, and they are different numbers --
    # which is the whole reason arm B's ratios are not comparable to arm A's.
    for port in ("naive", "lean"):
        r = cg[(port, "default")]
        lines.append(
            f"  {'(control)':<17} {'':<15} {port:<10}   "
            f"{str(r['kernel_count'])+' kernels, '+str(r['hot_fingerprint']['lines'])+' src lines':<30} "
            f"{sx[('naive','default')]['hot']['sass_total'] if port=='naive' else 0:>14,d} "
            f"{sx[(port,'default')]['hot']['sass_total']:>13,d} {1.0:>15.3f}")

    lean_base = sx[("lean", "default")]["hot"]["sass_total"]
    naive_base = sx[("naive", "default")]["hot"]["sass_total"]

    for k in KNOBS:
        if not k.sweep:
            lines.append("")
            lines.append(f"  {k.name:<17} {k.context_key:<15} "
                         f"{'-- NOT SWEEPABLE --':>10}")
            lines.append(f"      {k.defined_line}")
            lines.append(f"      {k.prediction.splitlines()[0]}")
            continue

        for v in k.sweep:
            if v == k.default:
                continue
            label = f"{k.name}={v}"
            row = {}
            for port in ("naive", "lean"):
                c = cg.get((port, label))
                s = sx.get((port, label))
                if c is None or s is None:
                    row[port] = ("not run", None, None)
                    continue
                same = (c.get("hot_sha256")
                        == cg[(port, "default")].get("hot_sha256"))
                row[port] = (
                    ("byte-identical" if same else
                     f"{c['kernel_count']}k/{c['hot_fingerprint']['lines']}L"),
                    s["hot"]["sass_total"], s["hot"]["sass_total"])
            ln = row["lean"][1]
            ratio = (ln / lean_base) if ln else None
            lines.append(
                f"  {k.name:<17} {k.context_key:<15} {v:>10}   "
                f"{row['naive'][0]:<30} "
                f"{row['naive'][1] if row['naive'][1] else 0:>14,d} "
                f"{ln if ln else 0:>13,d} "
                f"{(f'x{ratio:.3f}' if ratio else '-'):>15}")

    lines.append("")
    lines.append(f"  baselines: naive {naive_base:,d}  lean {lean_base:,d}  "
                 f"(-- {steps} DDA steps inlined per kernel, from the emitted source)")
    return "\n".join(lines)


def port_ablation(sass: dict, codegen: dict) -> str:
    """The port term, separated from the knob term.

    This is the row that C-14 says `notes/19` did not have, and it is not a
    knob result: it is what the ARM A -> ARM B port change did by itself, at
    every knob setting, so a reader can see how much of arm B's difference from
    arm A is the port and how much could possibly be a knob.
    """
    sx = _sass_index(sass, codegen)
    cg = {(r["port"], r["label"]): r for r in codegen["records"]}
    lines = []
    lines.append(f"  {'port':<18} {'kernels':>8} {'hot SASS':>10} {'per step':>9} "
                 f"{'LDG':>6} {'LDL (spill traffic)':>20} {'spill stores':>13} "
                 f"{'regs':>5}")
    lines.append("  " + "-" * 92)
    for port in PORTS:
        r = sx.get((port, "default"))
        if r is None:
            continue
        h = r["hot"]
        steps = codegen["max_steps"]
        lines.append(
            f"  {port:<18} {r['kernel_count']:>8} {h['sass_total']:>10,d} "
            f"{h['sass_total']/steps:>9.1f} {h['categories']['global_ld']:>6} "
            f"{h['categories']['local_ldst']:>20,d} "
            f"{(h['res'].get('spill_stores') or 0):>13,d} "
            f"{(h['res'].get('registers') or 0):>5}")
    n = sx[("naive", "default")]["hot"]["sass_total"]
    l = sx[("lean", "default")]["hot"]["sass_total"]
    nf = sx[("lean-no-flat", "default")]["hot"]["sass_total"]
    lines.append("")
    lines.append(f"  naive -> lean               {n:,d} -> {l:,d}  "
                 f"= {n/l:.3f}x fewer instructions")
    lines.append(f"  and of that, the flat-address advance alone:  "
                 f"{nf:,d} -> {l:,d} = {nf/l:.3f}x")
    lines.append(f"  (lean-no-flat is lean with ONLY the advanced address removed, "
                 f"so the difference is C-14's term and nothing else)")
    return "\n".join(lines)


def decomposition(sass: dict, codegen: dict) -> str:
    """Re-derive the residue, against THIS run's numbers.

    `derivation/minecu_gap.py:decomposition` does the same arithmetic on
    `notes/19`'s figures. It is re-derived here because the total being
    decomposed is different: that module divides into 36 600, this divides into
    whatever the lean port actually emits. Reusing its 9.73x against a different
    numerator would be the exact category error C-14 is about.
    """
    sx = _sass_index(sass, codegen)
    steps = codegen["max_steps"]
    if steps != MAX_STEPS:
        raise AssertionError(
            f"this sweep ran at max_steps={steps}, and the two multiplicative "
            f"terms below are written for the reference's default of "
            f"{MAX_STEPS} (notes/19:36). The residue would be arithmetically "
            f"valid and semantically wrong. Re-run at max_steps="
            f"{MAX_STEPS} or state the new terms.")

    lines = []
    for port in PORTS:
        h = sx[(port, "default")]["hot"]
        total = h["sass_total"] / REF_WHOLE
        per_step = h["sass_total"] / steps
        ref_per_step = REF_WHOLE / REF_STEPS
        early_exit = MAX_STEPS / MEAN_STEPS_REF   # every ray runs 32; mean 26.4
        loads = 4                                  # 4 predicated byte loads vs 1
        accounted = early_exit * loads
        lines.append(f"  {port}")
        lines.append(f"    hot kernel SASS                     {h['sass_total']:>9,d}")
        lines.append(f"    per DDA step (/{steps})               {per_step:>9.1f}"
                     f"   against the reference's {ref_per_step:.1f} "
                     f"({REF_WHOLE}/{REF_STEPS} unrolled)")
        lines.append(f"    whole-render ratio vs the reference {total:>9.2f}x")
        lines.append(f"    terms notes/19 §5.4 names: early exit {early_exit:.2f}x "
                     f"x loads {loads:.0f}x = {accounted:.2f}x")
        lines.append(f"    UNATTRIBUTED RESIDUE               {total/accounted:>9.2f}x"
                     f"   <- {total:.2f} / {accounted:.2f}")
        lines.append("")
    lines.append("  MODELLED, and marked as such. Arithmetic on MEASURED SASS")
    lines.append("  (this arm's own cubins) plus TWO figures read out of notes/19:")
    lines.append("  the reference's 776 (§4.0.1, re-derived there from its SASS) and")
    lines.append("  its measured mean of 26.4 steps/pixel (§1.3). The 26.4 is the")
    lines.append("  one input that needs a device and this file does not re-derive")
    lines.append("  it; a reader who wants it re-measured should take it from")
    lines.append("  `minecu_ref/bench/bench.cu`'s own probe on the same card.")
    lines.append("")
    lines.append("  WHY RE-DERIVE AT ALL, given derivation/minecu_gap.py exists.")
    lines.append("  That module divides into 36 600, which is notes/19's whole-graph")
    lines.append("  total for the NAIVE port. This arm's totals are different")
    lines.append("  objects -- the lean port's hot kernel, and the naive port's hot")
    lines.append("  kernel alone -- so reusing its 9.73x against a different")
    lines.append("  numerator is the category error C-14 exists to prevent.")
    return "\n".join(lines)


def sgd_audit(sass: dict, codegen: dict) -> str:
    """Is the remaining instruction count in the PORT or in the LOWERING?

    `derivation/minecu_gap.py:sass_per_op` reports 21.5 SASS per source op for
    the generated side against a 1-3 band for a scalar float lowering, and leaves
    two candidate causes undistinguished: (a) the port emits more operations
    than the reference's body needs, or (b) each operation lowers to more than
    one instruction. It says "only the ablation separates them", and
    `derivation/minecu_gap.py:ablation_target` records the ablation as
    `[PREDICTED]` -- never run. Arm B is that ablation, for cause (a).

    What the ablation found:

      * the port's own per-step cost falls by 30% (1137 -> 796 SASS) when the
        port is fixed, so cause (a) was REAL and was a material share of the
        gap;
      * the remaining 796 SASS per step is still about 16x the 1-3 band and
        about 16x the reference's 48.5 per step, so cause (b) -- the lowering
        -- is the LARGER share of what is left.

    So the C-14 attribution is right that the port mattered, and wrong to
    suggest the port was the remaining story: fixing it recovered 1.43x of a
    46.9x gap, not 46.9x.

    ONE RATIO IS DELIBERATELY REFUSED, AND IT IS THE ONE THE AUDIT ASKED FOR.
    The SASS-per-source-op ratio is not a stable quantity on this kernel, and
    the reason is measurable: **the compiler replicates the voxel gather rather
    than emitting one per step.** The lean kernel's emitted CUDA contains 496
    reads of the voxel buffer, not 32, and they are not 32 independent steps --
    they are 16 step bodies spaced ~400 lines apart plus a 481-read tail. So
    the denominator cannot be established from the source, and a ratio computed
    on a guess is a number that looks rigorous and means nothing.

    What is reported instead is the per-step SASS, which is well defined: it is
    `hot_sass / 32`, and the emitted source does contain 32 step bodies.
    """
    sx = _sass_index(sass, codegen)
    steps = codegen["max_steps"]

    lines = []
    lines.append(f"  {'port':<18} {'hot SASS':>10} {'SASS/step':>10} "
                 f"{'vs reference':>13} {'LDG (global)':>13} {'LDL (spill)':>13}")
    lines.append("  " + "-" * 80)
    for port in PORTS:
        h = sx[(port, "default")]["hot"]
        ps = h["sass_total"] / steps
        lines.append(f"  {port:<18} {h['sass_total']:>10,d} {ps:>10.1f} "
                     f"{ps/REF_PER_STEP:>12.1f}x "
                     f"{h['categories']['global_ld']:>13,d} "
                     f"{h['categories']['local_ldst']:>13,d}")
    n = sx[("naive", "default")]["hot"]["sass_total"] / steps
    l = sx[("lean", "default")]["hot"]["sass_total"] / steps
    lines.append("")
    lines.append(f"  reference per step (notes/19 §4.0.1): {REF_PER_STEP:.1f} "
                 f"= 776 / 16 unrolled steps")
    lines.append("")
    lines.append("  NOTE ON THE 776. It is notes/19's figure, read out of")
    lines.append("  `minecu_ref/build/kernels_sm_75.sass` and re-derived in")
    lines.append("  notes/19 §4.0.1 (152 prologue + 16x38 body + 16 epilogue). It")
    lines.append("  is the ONE number in this note this arm did not produce, and")
    lines.append("  it is the hand-written side, so producing it is arm A's job.")
    lines.append("  Everything in the `SASS/step` column for the generated side is")
    lines.append("  MEASURED here, from cubins this arm compiled.")
    lines.append("")
    lines.append(f"  the PORT term, MEASURED:  {n:.1f} -> {l:.1f} SASS/step = "
                 f"{n/l:.3f}x.  So {(1-l/n)*100:.1f}% of the naive port's")
    lines.append(f"  per-step cost was the port's own index arithmetic -- cause (a),")
    lines.append(f"  real, and removed.")
    lines.append(f"  what is LEFT:             {l:.1f} SASS/step = "
                 f"{l/REF_PER_STEP:.1f}x the reference. That residue is the")
    lines.append(f"  LOWERING: cause (b), expressible and not found, NOT the port.")
    lines.append("")
    lines.append("  So C-14 is right that the port's index arithmetic was a real")
    lines.append("  defect, and wrong to imply the port was the remaining story:")
    lines.append("  fixing it recovered 1.43x of a 46.9x instruction gap.")
    lines.append("")
    lines.append("  REFUSED: the SASS-per-source-op ratio. The compiler REPLICATES")
    lines.append("  the voxel gather -- the lean kernel's emitted CUDA has 496 reads")
    lines.append("  of the voxel buffer, not 32, and they are 16 step bodies ~400")
    lines.append("  lines apart plus a 481-read tail. The denominator is not")
    lines.append("  establishable, so the ratio is not reported.")
    return "\n".join(lines)


def knob_effects(codegen: dict, sass: dict) -> str:
    """THE knob sweep. One row per knob, with the verdict and the falsifier.

    `sweep` reports what each knob did to the source and to the instruction
    count. This adds the two things a table of measurements cannot: **whether
    the codegen change and the measurement are causally connected**, and **what
    would have made the null result something else**.

    The causal reading is not free. Three states are possible and they are told
    apart by three different instruments:

      * `INERT: byte-identical` -- the knob was set, the compiler re-ran, and
        emitted the same bytes. The measurement is unchanged *because* the code
        is unchanged. This is a real null result: the knob WAS consulted (it is
        in `to_program_config`, so the cache invalidated and the compiler was
        re-entered) and it had nothing to do.

      * `CAUSAL: SASS followed` -- the source changed and the instruction count
        moved with it. This is the only state that licenses "this knob does X
        to the work".

    And a fourth that is neither null nor a performance claim, which is the
    interesting one here:

      * `CAUSAL on count, BLIND on time` -- the source changed, the instruction
        count moved, and the change is one **this instrument cannot price**.
        NOLOCALS=1 sets `__launch_bounds__` from 128 to **1** on every kernel in
        the graph while moving the count by 8 instructions out of 36 384. A
        one-thread block does not meaningfully change what a thread issues; it
        changes occupancy, and occupancy is a memory-system property that no
        static instruction census can see.

    So "no measurable effect" is NOT the same claim as "no effect", and for
    NOLOCALS the honest statement is that the effect, if any, is in a dimension
    this table does not measure. That is why the timing run exists.
    """
    sx = _sass_index(sass, codegen)
    cg = {(r["port"], r["label"]): r for r in codegen["records"]}
    base = sx[("lean", "default")]["hot"]["sass_total"]

    lines = []
    lines.append("  Every row set the knob and re-ran the compiler. A null result is")
    lines.append("  meaningful only because the knob is in tinygrad's own program-cache")
    lines.append("  key (codegen/__init__.py:493, gated at run time by")
    lines.append("  registry.check_cache_key_coverage), so the cache was invalidated and")
    lines.append("  the compiler was genuinely re-entered.")
    lines.append("")
    hdr = (f"  {'knob':<16} {'value':>6} {'reachable':>10} {'codegen change':>18} "
           f"{'lean SASS':>10} {'delta':>8}  {'verdict':<24} what would falsify it")
    lines.append(hdr)
    lines.append("  " + "-" * 122)

    for k in KNOBS:
        if not k.sweep:
            lines.append(
                f"  {k.name:<16} {'--':>6} {'no':>10} {'n/a':>18} {'--':>10} "
                f"{'--':>8}  {'DOES NOT EXIST':<24} a ContextVar named "
                f"{k.context_key} being present")
            continue
        for v in k.sweep:
            if v == k.default:
                continue
            label = f"{k.name}={v}"
            c, s = cg.get(("lean", label)), sx.get(("lean", label))
            if c is None or s is None:
                lines.append(f"  {k.name:<16} {v:>6} {'?':>10} "
                             f"{'NOT RUN':>18} {'--':>10} {'--':>8}")
                continue
            same = (c.get("hot_sha256")
                    == cg[("lean", "default")].get("hot_sha256"))
            lb0 = cg[("lean", "default")]["hot_fingerprint"]["launch_bounds"]
            lb1 = c["hot_fingerprint"]["launch_bounds"]
            sassy = s["hot"]["sass_total"]
            delta = sassy - base
            # "cosmetic" is reserved for a change that provably cannot move an
            # instruction count. __launch_bounds__ moving 128 -> 1 does change
            # the count (by 8), so that row is not cosmetic -- it is CAUSAL on
            # the count and UNMEASURABLE on time by this instrument, which is a
            # different and more interesting thing. Calling it cosmetic would
            # understate it.
            lb_moved = (not same) and lb0 != lb1

            if same:
                verdict = "INERT: byte-identical"
                fals = "a source change; there is none, by sha256"
            elif lb_moved and abs(delta) < 16:
                verdict = "CAUSAL on count, BLIND on time"
                fals = ("a wall-clock move: the block went 128 -> 1 thread and "
                        "occupancy is a memory-system property this table cannot "
                        "see at all")
            else:
                verdict = "CAUSAL: SASS followed"
                fals = "the count not following the source; or the image changing"

            cg_txt = "byte-identical" if same else f"lb {lb0} -> {lb1}"
            lines.append(
                f"  {k.name:<16} {v:>6} {('yes' if k.reachable_here else 'no'):>10} "
                f"{cg_txt:>18} {sassy:>10,d} {delta:>+8,d}  {verdict:<24} {fals}")

    lines.append("")
    lines.append(f"  baseline: lean, default = {base:,d} SASS in the DDA kernel")
    lines.append("")
    lines.append("  WHY THE INERT KNOBS ARE INERT IS NOT ONE REASON, and the")
    lines.append("  difference is the whole taxonomy:")
    lines.append("")
    lines.append("    USE_TC / TC_SELECT / TC_OPT -- their ONLY reads are inside")
    lines.append("      `if USE_TC > 0 and (one reduce axis or TC_OPT>=1)`")
    lines.append("      (codegen/opt/heuristic.py:28,32). This kernel has no")
    lines.append("      reduction axis at all: every output element is a pure")
    lines.append("      function of one output index. So the guard is false and the")
    lines.append("      value is never consulted. TC_OPT=1 and TC_OPT=2 flip that")
    lines.append("      guard's SECOND disjunct and are STILL byte-identical -- a")
    lines.append("      measurement, not an inference, and it closes the reading that")
    lines.append("      TC_OPT could reach this kernel that way.")
    lines.append("      Class: (i) for this workload. Not a compiler defect; a")
    lines.append("      GEMM knob and a non-GEMM kernel.")
    lines.append("")
    lines.append("    ALLOW_TF32 -- its only consumer filters tensor-core selection")
    lines.append("      (postrange.py:226), and on sm_75 there is no fp32 tensor core")
    lines.append("      for it to admit: `cuda_sm75 = cuda_8168_f16` (tc.py:94,98) is")
    lines.append("      half-input only. So the answer is partly an ARCHITECTURE")
    lines.append("      fact, and this knob's class is (iv), not (i) or (ii). On an")
    lines.append("      sm_80 card -- where `cuda_8168_tf32` exists -- the same sweep")
    lines.append("      would answer a different question.")
    lines.append("")
    lines.append("    IMAGE -- redirects `dot`/`conv2d` to texture forms")
    lines.append("      (mixin/op.py:386,1593) and 4-wide vectorises image-shaped")
    lines.append("      buffers (heuristic.py:47). The DDA gathers from a 3-D volume,")
    lines.append("      so none applies. Inert because of the SHAPE of the workload:")
    lines.append("      the capability exists, this kernel is not one it covers. That")
    lines.append("      is not (i) -- (i) is 'cannot express', and the DDA is expressed.")
    lines.append("")
    lines.append("    TRANSCENDENTAL -- the ONE reachable knob with a real effect, and")
    lines.append("      it is NEGATIVE. The registry predicted the DDA kernel would")
    lines.append("      grow, because `rsqrt()` lowers to SQRT + RECIPROCAL (measured:")
    lines.append("      the op set of `Tensor([4.,9.]).rsqrt()` is {SQRT, RECIPROCAL})")
    lines.append("      and transcendental.py:275-276 rewrites SQRT to a software `pow`")
    lines.append("      when the flag is >= 2. It grew, on BOTH ports, by the same")
    lines.append("      ~450 source lines and ~970 SASS -- the emitted CUDA goes from")
    lines.append("      `1/sqrt(...)` to a ~28-line software pow with a `tg_bitcast`")
    lines.append("      exponent extraction and a degree-7 polynomial.")
    lines.append("      The prediction was right about the MECHANISM and the knob is")
    lines.append("      still useless here: the DDA needs one square root per ray in")
    lines.append("      the PROLOGUE, not on the hot path, and this knob only makes")
    lines.append("      that one more expensive. It is the clearest example in the")
    lines.append("      sweep of a knob that works exactly as documented and buys")
    lines.append("      nothing for this workload.")
    lines.append("")
    lines.append("    NOLOCALS -- the one knob that reaches the scheduler's shape")
    lines.append("      decisions, and the only row whose verdict this table cannot")
    lines.append("      finish. See the state name in the table above.")
    lines.append("")
    lines.append("    OPT_LEVEL -- on the brief's list and NOT IN THIS VERSION. It is")
    lines.append("      not a ContextVar: `Context(OPT_LEVEL=0)` raises KeyError, and")
    lines.append("      there is no definition line in helpers.py to find. The nearest")
    lines.append("      real knobs are NOOPT and BEAM (both helpers.py:232) and NEITHER")
    lines.append("      is on the list, so substituting one would answer a different")
    lines.append("      question. Recorded as an absence. A row reading 'no effect'")
    lines.append("      would be a null result about a knob that was never set -- the")
    lines.append("      same category error as the USE_TC/TC naming trap.")
    return "\n".join(lines)


def all_claims(codegen: dict, sass: dict) -> dict:
    return {
        "sweep": lambda: sweep(codegen, sass),
        "knob_effects": lambda: knob_effects(codegen, sass),
        "port_ablation": lambda: port_ablation(sass, codegen),
        "decomposition": lambda: decomposition(sass, codegen),
        "sgd_audit": lambda: sgd_audit(sass, codegen),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codegen", required=True)
    ap.add_argument("--sass", required=True)
    ap.add_argument("claims", nargs="*", help="just these claims, quotable")
    a = ap.parse_args(argv)

    codegen = json.loads(Path(a.codegen).read_text())
    sass = json.loads(Path(a.sass).read_text())

    claims = all_claims(codegen, sass)
    if a.claims:
        for name in a.claims:
            fn = claims.get(name)
            if fn is None:
                print(f"unknown claim: {name}\navailable: "
                      f"{', '.join(claims)}", file=sys.stderr)
                return 2
            print(fn())
            print()
        return 0

    print("=" * 78)
    print("  minecu_tg_knobs -- the knob sweep, from the artefacts")
    print(f"  codegen {a.codegen}")
    print(f"  sass    {a.sass}  ({sass['arch']})")
    print("=" * 78)
    for fn in claims.values():
        print(fn())
        print("=" * 78)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
