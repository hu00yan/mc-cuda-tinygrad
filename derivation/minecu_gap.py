#!/usr/bin/env python3
"""derivation/minecu_gap.py -- **VOID. Do not quote any claim in this file.**

    Ledger C-24. Every quantity below is derived from one division, and that
    division is a category error.

    The generated TinyGrad kernel is vectorised at roughly 16 pixels per
    thread. The hand-written reference is 1 pixel per thread. The two SASS
    instruction counts therefore cover different amounts of work, and dividing
    them does not produce a ratio of anything.

    What the number actually is, normalised per pixel per DDA step against the
    reference's measured 38-instruction period:

        naive port     74.4 SASS   ->  1.96x
        optimised port 53.5 SASS   ->  1.41x
        raw quotient   47.2x       ->  meaningless

    So the "UNATTRIBUTED RESIDUE 9.73x" this file published is 9.73x of a units
    error, not a missing term in a table. The `:ablation_target` claim is
    withdrawn for the same reason. The `:op_census` and `:sass_per_op` claims
    are unaffected -- they count one port's own operations and do not divide two
    differently-normalised counts.

    The file is kept rather than deleted because "prefer deletion to a hedged
    number" is the rule for *numbers*, and this file is the record of how a
    plausible-looking decomposition of 47.2x was wrong from its first line. A
    reader who finds the claim in a note can follow it here and find this.

    For what the comparison is now, read `notes/23-good-vs-naive.md`:
    per-pixel-per-step instructions are 1.41x while wall clock is 9.0x, so
    arithmetic is not the mechanism. The chain is graph size -> the scheduler's
    fusion budget -> 10 kernels / 11 allocs / 39 MiB becoming 6 / 1 / 4.0 MiB.

Run it:
    python derivation/minecu_gap.py                  # prints the void notice
    python derivation/minecu_gap.py <claim> [...]    # still runs; still void

A document may quote the value of a claim and nothing else. The citation that
licenses it is `derivation/minecu_gap.py:<claim>`.

WHY THIS MODULE EXISTS
    `notes/19-tinygrad-vs-handwritten.md` reports that generated TinyGrad code
    is 26.2x slower than the hand-written CUDA render on a T4, and traces 47.2x
    of it to instruction count. §5.4 then decomposes that 47.2x into named
    terms. This module asks one question of that table: **do the terms account
    for the total?** They do not, and the residue is large enough to change the
    conclusion. Ledger **C-14**.

    The residue is not a rounding artefact and it is not a mystery term -- it is
    a per-step term the table never lists, namely the cost of recomputing the
    voxel address from three float coordinates on every one of 32 steps instead
    of advancing a precomputed address, which is what Amanatides-Woo actually
    specifies. The table has rows for early exit and for byte loads and for
    spill, and no row for index arithmetic.

    So this module publishes two things and they must not be confused:

      decomposition   what the note's own terms account for, and the residue
      ablation_target what the residue *predicts* the hoisting experiment will
                      find, and what falsifies the prediction

    The second is `[PREDICTED]`. It has not been run. This module does not
    measure anything on a GPU; it does arithmetic on figures that
    `notes/19` obtained, and it is honest about which of them are its inputs.

WHAT IS IN HERE, AND WHY EACH IS SEPARATE
      totals            the two bases notes/19 §4.1 divides on, and why they
                       differ by 2x
      decomposition     the accounted fraction and the unattributed residue
      independence      why the accounted fraction is itself only an upper bound
      op_census         the per-step source-operation census, with line cites,
                       so the hand count is auditable rather than asserted
      sass_per_op       SASS instructions per source operation, both versions
      ablation_target   the residue as a falsifiable prediction
      mechanism_sentence  THE sentence, quotable, with its scope stated

SOURCES OF THE INPUT FIGURES
    Every input is `[ARTIFACT]`, read out of `notes/19`, with a line cite:

      36 600 / 776 = 47.2x      notes/19:597  whole render, one thread
      36 384 / 776 = 46.9x      notes/19:713  fused kernel alone
      36 384 / 32  = 1137.0     notes/19:598  generated, per DDA step
      776 / 16     = 48.5       notes/19:598  reference, per DDA step
      32 steps vs 26.4 mean     notes/19:714  early-exit term, 1.21x
      4x read instructions      notes/19:715  predicated byte loads per step
      16 unrolled steps         notes/17 §7, notes/19:611
      max_steps = 32            notes/19:36

    The 26.4 mean and the 20 932 B spill are not inputs to any claim here.
"""

import sys

# ---------------------------------------------------------------- inputs
# [ARTIFACT] notes/19:597 -- whole render, one thread, both prologues included.
REF_INSTR_WHOLE = 776
TG_INSTR_WHOLE = 36_600

# [ARTIFACT] notes/19:598 -- the per-step basis.
TG_INSTR_STEP = 36_384 / 32   # 1137.0
REF_INSTR_STEP = 776 / 16     # 48.5

# [ARTIFACT] notes/19:714 -- every ray runs all 32 steps; measured mean 26.4.
MAX_STEPS = 32
MEAN_STEPS_REF = 26.4

# [ARTIFACT] notes/19:715 -- 4 predicated byte loads per step vs 1.
READS_PER_STEP_TG = 4
READS_PER_STEP_REF = 1

# ---------------------------------------------------------------- claims


def totals():
    """The two bases notes/19 §4.1 divides on, and the 2x between them."""
    whole = TG_INSTR_WHOLE / REF_INSTR_WHOLE
    single = TG_INSTR_STEP / REF_INSTR_STEP
    lines = [
        "notes/19 §4.1 -- instruction counts per thread, whole render",
        "",
        f"  reference            {REF_INSTR_WHOLE:>8,d} SASS",
        f"  generated            {TG_INSTR_WHOLE:>8,d} SASS",
        f"  factor               {whole:>8.2f}x     <- the one notes/19:601 says to quote",
        "",
        "per DDA step, the other basis",
        "",
        f"  reference            {REF_INSTR_STEP:>8.1f} = {REF_INSTR_WHOLE} / 16 unrolled steps",
        f"  generated            {TG_INSTR_STEP:>8.1f} = {TG_INSTR_STEP:.0f} / {MAX_STEPS} steps",
        f"  factor               {single:>8.2f}x",
        "",
        f"  the two bases differ by {whole / single:.2f}x, because the reference's",
        "  prologue is 152 of its 776 instructions (19.6%) -- paid once and",
        "  amortised over 16 steps -- while the fused kernel's prologue is",
        "  negligible against 32. Same object, different denominator.",
    ]
    return "\n".join(lines)


def decomposition():
    """What notes/19 §5.4's own terms account for, and what is left over.

    [MODELLED]. This is arithmetic on notes/19's figures. It measures nothing.
    """
    total = TG_INSTR_WHOLE / REF_INSTR_WHOLE

    # the two multiplicative terms notes/19:714-715 lists
    early_exit = MAX_STEPS / MEAN_STEPS_REF
    loads = READS_PER_STEP_TG / READS_PER_STEP_REF

    accounted = early_exit * loads
    residue = total / accounted

    lines = [
        "notes/19 §5.4 -- the decomposition, audited against its own total",
        "",
        f"  total, from the table's own first row        {total:>8.2f}x",
        "",
        "  terms the table names and quantifies:",
        f"    no per-ray early exit  {MAX_STEPS}/{MEAN_STEPS_REF}      {early_exit:>8.2f}x",
        f"    4 predicated byte loads {READS_PER_STEP_TG}/{READS_PER_STEP_REF}                  {loads:>8.2f}x",
        f"    product (see :independence)                  {accounted:>8.2f}x",
        "",
        f"  UNATTRIBUTED RESIDUE    {total:>8.2f} / {accounted:.2f}   = {residue:>6.2f}x",
        "",
        "  The table reports 47.2x as explained by instruction count and names",
        f"  {accounted:.2f}x of it. The remaining {residue:.2f}x has no row.",
        "",
        "  Its unlisted terms -- local-memory spill and the 256 B/pixel state",
        "  round trip -- are marked non-separable or regime-specific in the table",
        "  itself, so they do not close it either.",
    ]
    return "\n".join(lines)


def independence():
    """Why the accounted fraction is an upper bound, not a measurement.

    notes/19 multiplies the early-exit term by the load term. Multiplying terms
    is only valid if they are independent. They are not, and the direction of
    the violation matters: both terms are consequences of the *same* fact, that
    the tensor form has no per-ray early exit. A ray that stops early also stops
    issuing loads. So the product double-counts, which means the true accounted
    fraction is at or below the product, which means the residue is at or above
    the figure :decomposition prints.

    [MODELLED]. The claim is about the logic of the table, not about any GPU.
    """
    early_exit = MAX_STEPS / MEAN_STEPS_REF
    loads = READS_PER_STEP_TG / READS_PER_STEP_REF
    product = early_exit * loads
    # the loosest defensible bound: if the load term is entirely *inside* the
    # early-exit term rather than additional to it, nothing is double counted
    # and the accounted fraction collapses to the early-exit term alone.
    lines = [
        "notes/19 §5.4 -- independence of the two multiplicative terms",
        "",
        f"  product, as the table computes it               {product:>8.2f}x",
        f"  if the load term is already inside the exit term {early_exit:>8.2f}x",
        "",
        "  Both terms come from one fact: the tensor form cannot exit a ray",
        "  early. That same fact makes the ray issue fewer steps AND fewer",
        "  loads. Multiplying them charges the overlap twice.",
        "",
        f"  So the residue is bounded below by {TG_INSTR_WHOLE / REF_INSTR_WHOLE / product:.2f}x",
        f"  and above by {TG_INSTR_WHOLE / REF_INSTR_WHOLE / early_exit:.2f}x,",
        "  and the table's own presentation picks the optimistic end.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- the census
# A HAND COUNT, and it is labelled as one. It is reproduced here so that a
# reader can check it against the source rather than take it on trust, and so
# that if the port changes the census has to be revisited in the same file.
#
# minecu/tinygrad/render.py:235-261, the body of `for step in range(max_steps)`.
# Operation counts are source-level TinyGrad operations, one per operator
# application, with a comparison and a boolean AND counted separately.
CENSUS = [
    ("render.py:241-243", "inside = 6 float compares + 5 bool ANDs", 11),
    ("render.py:244-246", "ix/iy/iz = 3 cast(i32) + 3 clip", 6),
    ("render.py:247", "4-index gather voxels[b, iy, ix, iz] + cast", 9),
    ("render.py:249-251", "solid, hit-select, found-or", 6),
    ("render.py:254-255", "step_taken_x, step_taken_y", 6),
    ("render.py:256-258", "t_max x/y/z updates", 9),
    ("render.py:259-261", "voxel x/y/z updates", 6),
]

# The reference's per-step body, kernels.cu:666-678, at the granularity
# notes/19:611 measured it: period 38 SASS instructions over 16 unrolled steps.
CENSUS_REF_PERIOD = 38
CENSUS_REF_STEPS = 16


def op_census():
    """The per-step source-operation census for the port, with line cites.

    [HAND-COUNT], audited by reading the cited lines. Not machine-derived: a
    mechanical op counter would have to decide what counts as one operation
    across a 4-index gather, which is exactly the judgement in dispute.
    """
    total = sum(n for _, _, n in CENSUS)
    lines = [
        "minecu/tinygrad/render.py:235-261 -- per-DDA-step source operations",
        "",
        f"  {'site':<20} {'what':<44} ops",
        f"  {'-'*20} {'-'*44} ---",
    ]
    for site, what, n in CENSUS:
        lines.append(f"  {site:<20} {what:<44} {n:>3d}")
    lines += [
        f"  {'-'*20} {'-'*44} ---",
        f"  {'total':<20} {'':<44} {total:>3d}",
        "",
        f"  {MAX_STEPS} steps -> {total * MAX_STEPS:,d} source operations per render",
        "",
        "  Two of those rows are the subject of C-14 and neither is hardware",
        "  knowledge to remove:",
        "",
        "    6 ops  the three cast+clip pairs exist only to index the gather.",
        "          Inside can be folded into one unsigned compare per axis, and",
        "          the clip collapses to a single clamp on a flat index.",
        "    9 ops  the gather recomputes the flat address from three float",
        "          coordinates. Amanatides-Woo -- the algorithm the reference",
        "          implements -- precomputes tMax and tDelta and then advances",
        "          incrementally. Recomputing is not that algorithm.",
    ]
    return "\n".join(lines)


def sass_per_op():
    """SASS instructions per source operation, generated side against reference.

    This is the ratio that says whether the generated code is inefficient or the
    port is. Scalar float code issues on the order of 1-3 SASS per source
    operation; a large figure means the lowering is multiplying each operation
    by a lane factor or re-materialising a spilled value.
    """
    census_total = sum(n for _, _, n in CENSUS)
    tg = TG_INSTR_STEP / census_total
    lines = [
        "SASS instructions per source operation, per DDA step",
        "",
        f"  generated   {TG_INSTR_STEP:>8.1f} SASS / {census_total:>3d} ops = {tg:>6.1f}",
        f"  reference   {CENSUS_REF_PERIOD:>8d} SASS / {CENSUS_REF_STEPS:>2d} steps,"
        f" whole kernel amortised {REF_INSTR_STEP:.1f} per step",
        "",
        f"  {tg:.0f} SASS per operation is {tg / 1.5:.0f}x to {tg / 3.0:.0f}x above the",
        "  1-3 band a scalar float lowering produces. Two candidate causes, not",
        "  distinguished by this figure:",
        "",
        "    (a) the port emits more operations than the reference's body needs,",
        "        in which case the census above is the defect;",
        "    (b) each operation lowers to more than one instruction, in which",
        "        case the lowering is the defect.",
        "",
        "  Only the ablation separates them. That is the whole point of it.",
    ]
    return "\n".join(lines)


def ablation_target():
    """The residue as a falsifiable prediction. [PREDICTED] -- not run.

    Hoisting the voxel address out of the step loop is the cheapest way to
    remove the unlisted 9 ops/step, and it is what the reference does. If the
    census is the defect, the ablation should recover most of the residue.
    """
    total = TG_INSTR_WHOLE / REF_INSTR_WHOLE
    early_exit = MAX_STEPS / MEAN_STEPS_REF
    loads = READS_PER_STEP_TG / READS_PER_STEP_REF
    product = early_exit * loads
    residue = total / product
    census_total = sum(n for _, _, n in CENSUS)
    removable = 6 + 9  # the cast+clip row and the gather index arithmetic

    lines = [
        "PREDICTION, not measured -- the target the hoisting ablation must hit",
        "",
        f"  residue to explain                    {residue:>8.2f}x",
        f"  ops per step the hoisting removes      {removable:>8d} of {census_total}",
        f"  after removal, ops per step            {census_total - removable:>8d}",
        "",
        f"  If the {residue:.2f}x residue is mostly the census, hoisting should leave",
        f"  the generated side near {census_total - removable}/{census_total} of its",
        "  per-step cost, and the 47.2x should fall by roughly that factor.",
        "",
        "  FALSIFIED if the optimised version still issues more than about",
        f"  {TG_INSTR_STEP * 0.8:.0f} SASS per step, i.e. if removing {removable} of",
        f"  {census_total} operations per step barely moves the instruction",
        "  count. In that case the cost is in the lowering -- lane duplication,",
        "  or re-materialisation of spilled state -- and C-14's attribution is",
        "  wrong in the other direction.",
        "",
        "  Note what this prediction does NOT test. It says nothing about the",
        "  fusion crossover near 1M pixels, and nothing about the 20 932 B",
        "  spill, both of which are decisions about fusion and register",
        "  allocation rather than about index arithmetic. Those are recorded in",
        "  notes/19 and are not withdrawn by C-14.",
    ]
    return "\n".join(lines)


def mechanism_sentence():
    """VOID — see C-24. Retained only so the correction is traceable."""
    return (
        "WITHDRAWN (ledger C-24). The decomposition this file published attributed a "
        "47.2x gap to the generated code and left 9.73x unlisted. The 47.2x divided "
        "an instruction count taken at ~16 pixels per thread by one taken at 1 pixel "
        "per thread, so it was never a ratio; per pixel per DDA step the gap is "
        "1.96x naive and 1.41x optimised against the reference's 38-instruction "
        "period. [VOID]"
    )


CLAIMS = {
    "totals": totals,
    "decomposition": decomposition,
    "independence": independence,
    "op_census": op_census,
    "sass_per_op": sass_per_op,
    "ablation_target": ablation_target,
    "mechanism_sentence": mechanism_sentence,
}


def main(argv):
    if len(argv) > 1:
        for name in argv[1:]:
            fn = CLAIMS.get(name)
            if fn is None:
                print(f"unknown claim: {name}")
                print("available: " + ", ".join(CLAIMS))
                return 2
            print(fn())
            print()
        return 0
    print(__doc__.strip())
    print()
    print("=" * 72)
    print()
    for name, fn in CLAIMS.items():
        print(fn())
        print("=" * 72)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
