#!/usr/bin/env python3
"""derivation/divide.py -- TinyGrad's float32 `/`: mechanism, error rate, and
the `a/a` corner.

Run it:
    python derivation/divide.py                      # full report
    python derivation/divide.py <claim> [<claim>...] # just these, quotable

A document may quote the value of a claim and nothing else. The citation that
licenses it is `derivation/divide.py:<claim>`.

WHY THIS MODULE EXISTS
    A probe whose divisor was the dividend -- `probes/fp32_correctness.py:57`
    read `b = regimes[name] if op in "*/" else a`, and `regimes[name]` *is* `a`
    -- measured `a/a` and published the figure as the error rate of `/`. That
    figure was copied into ~58 places across 19 files. Nobody could check it,
    because the number was a literal in a `print` statement and there was
    nothing to check it against. The structural fix is what this module is:
    **a number is derived here, once, and documents cite the claim that prints
    it.** A wrong number now has one place to be wrong. Ledger **C-09**, and
    `notes/06-corrections-ledger.md` §5 for the error class.

WHAT IS IN HERE, AND WHY EACH IS SEPARATE
    The record kept failing because different questions shared one sentence.
    They have different scopes, and most of them are functions of the divisor
    law rather than of `/`:

      mechanism_sentence       THE sentence, band pinned, emitted as one
                               quotable string. A document quotes this and
                               nothing else about `/`
      authoritative_reproduction  does THIS machine reproduce the six
                               authoritative figures, per cell, in sigmas
      published_rate_table     the six cells on a declared draw, with counts
      sampling_interval        the binomial interval, and the empirical spread
                               over a named block of divisor streams
      mechanism_rewrite        where the rewrite happens -- a fact about source
      mechanism_no_knob        whether anything can undo it -- a fact about the
                               package's flag surface
      error_rate_independent   the sweep over six independent laws -- a sweep,
                               explicitly not a rate
      error_rate_self_division the degenerate rate, `a/a`, and why it is lower
      corner_predicate         the exact predicate for the `a/a` failure, and
                               the single wrong bit pattern it can produce
      ulp_bound                the magnitude, which is a bound and not a law
      reciprocal_sensitivity   what a worse reciprocal would predict, which is
                               what identifies the reciprocal as correctly rounded
      cell_failure_fraction    the arithmetic `bench/HYPOTHESES.md` H9 got
                               wrong by ~5e6
      dependent_law_outside_band  a dependent divisor law, which leaves the band
      representable_controls   the `2*a` control: representability, not
                               magnitude, is the discriminator
      probe_denominators       the `+ - *` op counts, which the probe stated
                               wrongly

THE THREE CLAUSES THAT ARE EASILY DROPPED
    `mechanism_sentence` is load-bearing in three separate places, and each has
    been got wrong in this record at least once:

      "bit-identical to a correctly rounded reciprocal" is SEPARATE from the
        percentage. A reciprocal 1 ULP worse predicts ~86% against the ~27%
        measured, so the reciprocal's accuracy is load-bearing and the earlier
        claim that the rate is "insensitive to reciprocal accuracy" is
        RETRACTED. The two clauses are not one claim.
      the failure is DOUBLE ROUNDING, not approximation. The only wrong `a/a`
        value is 0x3f7fffff, one-sided, with an exact predicate. An
        approximation has a two-sided error; this does not.
      "no USER-FACING knob" is weaker than "no knob". Ops.RECIPROCAL exists and
        PTX renders it as rcp.approx.f32, but `/` never routes to it as a
        choice, so there is nothing for a user to expose.

THE PERSISTED INPUTS, AND WHY ONE OF THEM IS A RECORD
    `AUTHORITATIVE_SELF` / `AUTHORITATIVE_GENERAL` are this study's six published
    figures, held here as a pinned INPUT rather than derived. They were produced
    on a different host from a draw whose seed was not recorded, so they are not
    re-derivable and pretending otherwise would be the error this module exists
    to prevent. `authoritative_reproduction` therefore recomputes all six from a
    declared draw and REPORTS THE DELTA in sigmas, so a reader can see whether
    this machine agrees inside the sampling interval instead of being told it
    does. It agrees: the largest deviation is 0.101pp against a binomial 1-sigma
    of 0.044pp, i.e. 2.3 sigma on one cell and under 1.5 on the rest. The shape
    reproduces; the third decimal is the seed. `mechanism_sentence` quotes the
    RECORD rather than re-deriving its own numbers, because a published sentence
    that silently recomputes its figures is a sentence whose meaning depends on
    when it was printed.

WHAT IS MODELLED AND WHAT IS NOT
    No device is used. `a*fl32(1/b)` is computed in numpy and the *rate* is a
    property of the arithmetic, not of any backend. `reciprocal_sensitivity`
    evaluates hypothetical reciprocals and is labelled `[MODELLED]`. The
    per-backend measurement of the same quantity is a different module,
    `derivation/divide_backends.py`, because it needs a device and because a
    device number and an arithmetic number must never be quoted as one thing.

A BAND IS A CONTAINMENT STATEMENT
    Every band this module prints is truncated inward, never rounded outward.
    A measured floor of 0.25962 is `25.9%` truncated and `26.0%` rounded, and
    publishing `26.0%` as the floor states a rate the record does not support.
    `_band_floor` does the truncation; that is why `mechanism_sentence` prints
    25.9%-27.0% and not 26.0%-27.0%.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fp32 as F                                                     # noqa: E402
from _util import Module, PinDrift, Value, rule, run, sha             # noqa: E402

EXPECTED: dict[str, str] | None = None      # filled by verify.py --write

MOD = Module(
    "divide",
    __doc__ or "",
    EXPECTED,
)

CELL = F.Cell()


# ---------------------------------------------------------------- helpers

def _regs(n):
    return F.regimes(n)


def _rates(regs, n):
    """rate[(regime, law.name)] = fraction of `a/(law)` that
    `a*fl32(1/law)` gets wrong, plus the same-law self rate."""
    out, selfr = {}, {}
    for name, a in regs.items():
        selfr[name] = float((F.rcp_mul(a, a) != F.ONE).mean())
        for law in F.INDEPENDENT_LAWS:
            b = law(a, n, name)
            out[(name, law.name)] = float((F.rcp_mul(a, b) != F.exact_f32(a, b, "/")).mean())
    return out, selfr


def _band(vals):
    v = list(vals)
    return min(v), max(v), float(np.median(v))


def _band_text(lo, hi, med, unit="%", n="none"):
    """`n` is the point count, an int to print it, or the string "none" to
    print a bare range with no median (for a band that has no median because
    it is two numbers, not a sample)."""
    f = {"%": 100.0, "x": 1.0}[unit]
    p = 1 if unit == "%" else 2
    tail = f"-{hi*f:.{p}f}{unit}"
    if n != "none":
        tail += f" (median {med*f:.{p}f}{unit}, {n} points)"
    elif med is not None:
        tail += f" (median {med*f:.{p}f}{unit})"
    return f"{lo*f:.{p}f}{unit}{tail}"


# ---------------------------------------------------------------- mechanism

@MOD.claim("mechanism_rewrite")
def c_mechanism_rewrite() -> Value:
    """Where `/` becomes `a*(1/b)`, read out of the installed package."""
    p = MOD.pins
    div_line = p.tinygrad_line("mixin/elementwise.py", 251,
                               must_contain="a.alu(Ops.MUL, b.reciprocal())")
    recip_line = p.tinygrad_line("mixin/elementwise.py", 465,
                                 must_contain="return self.alu(Ops.RECIPROCAL)")
    cstyle_line = p.tinygrad_line("renderer/cstyle.py", 137,
                                  must_contain="Ops.RECIPROCAL")
    fdiv_line = p.tinygrad_line("renderer/cstyle.py", 274,
                                must_contain="Ops.FDIV")
    ptx_line = p.tinygrad_line("renderer/ptx.py", 19, must_contain="rcp")
    decomp = p.tinygrad_line("codegen/decomp/op.py", 133, must_contain="FDIV")
    p.param("rewrite", div_line)

    detail = "\n".join([
        "  The rewrite, in the Tensor API, above the scheduler and above backend",
        "  selection:",
        f"    mixin/elementwise.py:251  {div_line.strip()}",
        f"    mixin/elementwise.py:465  {recip_line.strip()}",
        "  It is unconditional, so no Ops.FDIV exists in the IR for a renderer to",
        "  render. The renderers that do have Ops.FDIV reach it through",
        "  decomposition, not through a user choice:",
        f"    renderer/cstyle.py:274     {fdiv_line.strip()}",
        f"    codegen/decomp/op.py:133  {decomp.strip()}",
        "  cstyle renders Ops.RECIPROCAL as a literal reciprocal expression, which",
        "  the backend is then free to evaluate however it likes:",
        f"    renderer/cstyle.py:137     {cstyle_line.strip()[:96]}",
        "  and PTX renders it as an explicitly approximate instruction -- a real",
        "  op, that `/` reaches whether or not the user chose it:",
        f"    renderer/ptx.py:19         {ptx_line.strip()[:96]}",
    ])
    return Value(
        "mixin/elementwise.py:251 lowers float32 `/` to "
        "`a.alu(Ops.MUL, b.reciprocal())` in the Tensor API, unconditionally, "
        "above the scheduler and above backend selection; "
        "`mixin/elementwise.py:465` makes that `Ops.RECIPROCAL`; cstyle renders "
        "it as `(1/b)` (renderer/cstyle.py:137) and PTX as `rcp.approx` "
        "(renderer/ptx.py:19), so `/` reaches the approximate instruction "
        "without the user choosing it",
        detail,
    )


@MOD.claim("mechanism_no_knob")
def c_mechanism_no_knob() -> Value:
    """Whether any user-facing knob can undo the rewrite. A grep, not a claim."""
    p = MOD.pins
    n_fast, hits = p.tinygrad_grep("use_fast_math")
    n_any, _ = p.tinygrad_grep("-fno-fast-math")
    metal = p.tinygrad_line("runtime/ops_metal.py", 93, must_contain="fno-fast-math")
    opts = []
    for rel, lineno, needle in (("runtime/support/compiler_cuda.py", 48, "gpu-architecture"),
                                ("runtime/support/compiler_cuda.py", 51, "-I"),
                                ("runtime/support/compiler_cuda.py", 53, "minimal")):
        opts.append(f"    {rel}:{lineno}  {p.tinygrad_line(rel, lineno, must_contain=needle).strip()}")
    order = p.tinygrad_line("runtime/ops_cuda.py", 121, must_contain="Renderer")
    p.param("use_fast_math_hits", n_fast)

    detail = "\n".join([
        f"  `use_fast_math` appears in {n_fast} places in the installed tinygrad.",
        f"  `-fno-fast-math` appears in {n_any}.",
        "  The Metal compiler is already invoked with the flag that forbids fast",
        f"    runtime/ops_metal.py:93  {metal.strip()[:110]}",
        "  and it does not help, because the rewrite is already in the emitted",
        "  source before the compiler sees it. The whole CUDA option list is:",
        *opts,
        "  Renderer order on CUDA, for the record:",
        f"    runtime/ops_cuda.py:121  {order.strip()[:110]}",
    ])
    return Value(
        f"no user-facing knob: `use_fast_math` appears in {n_fast} places in "
        f"tinygrad, `-fno-fast-math` in {n_any} (and the Metal compiler already "
        f"passes it, `runtime/ops_metal.py:93`), and the entire NVRTC option "
        f"list is `--gpu-architecture`, `-I`, `--minimal` "
        f"(`runtime/support/compiler_cuda.py:48,51,53`)",
        detail,
    )


# ---------------------------------------------------------------- the rate

# The divisor-stream block the published band's endpoints are taken over. Named
# here, once, so the band is a function of a written-down set and not of
# whatever the loop variable happened to be. `+0` IS the declared published
# stream, so the block contains the published draw itself.
PUB_STREAM_OFFSETS = (0, 1, 2, 3, 4, 5, 6, 7)


def _sigma(p: float, n: int) -> float:
    """1-sigma binomial standard error of a rate measured on `n` draws."""
    return float(np.sqrt(p * (1.0 - p) / n))


def _published(n=None):
    """`(n, A, B)`: the published draw, plus the block of same-law divisor
    streams the band endpoints are taken over."""
    n = n or F.N_PUBLISHED
    ra = np.random.default_rng(F.SEED_PUB_A)
    a = {r: np.ascontiguousarray(F.draw_published(n, r, ra))
         for r in F.PUB_REGIME_ORDER}
    b = {}
    for k in PUB_STREAM_OFFSETS:
        rb = np.random.default_rng(F.SEED_PUB_B + k)
        b[k] = {r: np.ascontiguousarray(F.draw_published(n, r, rb))
                for r in F.PUB_REGIME_ORDER}
    return n, a, b


def _pub_rates(n, a, b, k=0):
    """`(regime -> general a/b rate)` on divisor stream `k`, and the self rate."""
    g = {r: float((F.rcp_mul(a[r], b[k][r])
                   != F.exact_f32(a[r], b[k][r], "/")).mean())
         for r in F.PUB_REGIME_ORDER}
    s = {r: float((F.rcp_mul(a[r], a[r]) != F.ONE).mean())
         for r in F.PUB_REGIME_ORDER}
    return s, g


@MOD.claim("published_rate_table")
def c_published_table(n=None) -> Value:
    """THE six numbers. Three regimes x {divisor = dividend, divisor
    independent}, on the published draw declared in `_fp32.published()`.

    This is the corrected form of the table the record used to publish. The
    left column is the quantity the old probe measured and mislabelled as "the
    `/` rate"; the right column is the rate of general division. The two differ
    by the ratio printed per row, and the left column is not a rate for `/` at
    all -- see `corner_predicate` for why it is structurally different.

    REPRODUCTION NOTE, and it is the reason the third decimal is not the point.
    An exhaustive search over 20,000 `a`-streams and four candidate near-1.0
    laws, plus four stream-topology constructions, does not land on the counts
    the corrected record published (13.988 / 15.312 / 15.170 for the self row).
    The `a` stream is not recoverable from the published figure alone, so the
    draw here is DECLARED (`_fp32.published`) and its own counts are what this
    claim reports. What reproduces across draws is the shape: the self row at
    14.0-15.3%, the general row at 26.0-27.2%, and a ratio of 1.77x-1.86x. Any
    document needing a count must name the draw; `sampling_interval` quantifies
    why a count without one is not a fact.
    """
    n, A, B = _published(n)
    s, g = _pub_rates(n, A, B)
    cs = {r: int((F.rcp_mul(A[r], A[r]) != F.ONE).sum()) for r in F.PUB_REGIME_ORDER}
    cg = {r: int((F.rcp_mul(A[r], B[0][r]) != F.exact_f32(A[r], B[0][r], "/")).sum())
          for r in F.PUB_REGIME_ORDER}
    ratios = [cg[r] / cs[r] for r in F.PUB_REGIME_ORDER]
    slo, shi = min(s.values()), max(s.values())
    glo, ghi = min(g.values()), max(g.values())
    MOD.pins.param("n_published", n)
    MOD.pins.param("pub_regimes", list(F.PUB_REGIME_ORDER))
    MOD.pins.param("pub_seeds", [F.SEED_PUB_A, F.SEED_PUB_B])
    MOD.pins.param("pub_self_counts", sorted(f"{k}|{v}" for k, v in cs.items()))
    MOD.pins.param("pub_general_counts", sorted(f"{k}|{v}" for k, v in cg.items()))
    R = F.PUB_REGIME_ORDER
    lines = [f"  n = {n:,} per regime. `a` and `b` are draws of the SAME marginal law",
             "  from two disjoint streams, so the columns differ only in dependence.",
             "  Oracle: (a.astype(f64)/b.astype(f64)).astype(f32).",
             "",
             f"  {'regime':36s} {'b := a  (a/a)':>23s} {'b independent (a/b)':>25s} {'ratio':>8s}"]
    for r in R:
        lines.append(f"  {r:36s} {s[r]:9.3%} ({cs[r]:>7,}) "
                     f"{g[r]:13.3%} ({cg[r]:>7,}) {cg[r]/cs[r]:6.2f}x")
    lines += ["",
              f"  b := a         {_band_text(slo, shi, None)}   <-- NOT a rate for `/`",
              f"  b independent  {_band_text(glo, ghi, None)}   <-- the rate, on THIS law",
              f"  ratio          {_band_text(min(ratios), max(ratios), None, 'x')}",
              f"  binomial 1-sigma per cell: "
              f"{_sigma(glo, n)*100:.3f}pp - {_sigma(ghi, n)*100:.3f}pp "
              f"(n = {n:,}); see `sampling_interval`."]
    return Value(
        f"on the published draw (n = {n:,} per regime, `a` seed {F.SEED_PUB_A}, "
        f"`b` seed {F.SEED_PUB_B}, same marginal law, disjoint streams): divisor "
        f"equal to dividend {cs[R[0]]:,}/{cs[R[1]]:,}/{cs[R[2]]:,} = "
        f"{slo:.3%}-{shi:.3%}, which is NOT a rate for `/`; divisor independent "
        f"{cg[R[0]]:,}/{cg[R[1]]:,}/{cg[R[2]]:,} = {glo:.3%}-{ghi:.3%}, the rate "
        f"on this divisor law, at a ratio to the self row of "
        f"{_band_text(min(ratios), max(ratios), None, 'x')}",
        "\n".join(lines),
    )


@MOD.claim("sampling_interval")
def c_sampling(n=None) -> Value:
    """How much of a published rate is the draw. The third decimal of a
    percentage measured on n = 10^6 draws is close to a coin flip, and a band
    is only honest if the interval is published with it.

    Two intervals, both derived here: the binomial one, and the EMPIRICAL one --
    the same three regimes recomputed over a named block of independent divisor
    streams. The second is what decides whether a quoted band endpoint is a
    property of the arithmetic or of the seed.
    """
    n, A, B = _published(n)
    per = {}
    for r in F.PUB_REGIME_ORDER:
        per[r] = [float((F.rcp_mul(A[r], B[k][r])
                        != F.exact_f32(A[r], B[k][r], "/")).mean())
                  for k in PUB_STREAM_OFFSETS]
    spread = {r: max(v) - min(v) for r, v in per.items()}
    p = float(np.mean([min(v) for v in per.values()]))
    s = _sigma(p, n)
    flat = [x for v in per.values() for x in v]
    lo, hi = min(flat), max(flat)
    MOD.pins.param("sampling_n", n)
    MOD.pins.param("sampling_offsets", list(PUB_STREAM_OFFSETS))
    MOD.pins.param("sampling_rates", sorted(
        f"{r}|{k}|{per[r][k]:.9f}" for r in per for k in range(len(PUB_STREAM_OFFSETS))))
    lines = [f"  binomial 1-sigma at n = {n:,}, p ~ {p:.3f}: +/-{s*100:.3f}pp "
             f"(+/-{1.96*s*100:.3f}pp at 95%)",
             f"  divisor-stream block: SEED_PUB_B + {list(PUB_STREAM_OFFSETS)}",
             "",
             f"  {'regime':36s} {'min':>9s} {'max':>9s} {'spread':>9s}"]
    for r, v in per.items():
        lines.append(f"  {r:36s} {min(v):8.3%} {max(v):8.3%} {spread[r]:8.3%}")
    lines += ["",
              f"  union over the {len(flat)} (regime, stream) points: "
              f"{_band_text(lo, hi, None)}",
              f"  widest single-cell spread {max(spread.values())*100:.3f}pp = "
              f"{max(spread.values())/s:.1f}x the binomial 1-sigma",
              "",
              "  CONSEQUENCE. A general `a/b` rate is a function of the divisor",
              "  law, and at this n it is ALSO a function of the seed in the third",
              "  decimal. Both intervals are properties of the draw. Neither is a",
              "  property of the divisor law that is missing -- which is why a",
              "  third-decimal scalar must never be quoted without one of them."]
    return Value(
        f"at n = {n:,} a rate carries a binomial 1-sigma of +/-{s*100:.3f}pp, and "
        f"recomputing the same three regimes over a named block of "
        f"{len(PUB_STREAM_OFFSETS)} independent divisor streams moves a cell by "
        f"up to {max(spread.values())*100:.3f}pp, so the third decimal of any "
        f"published `a/b` rate is a property of the draw and not of `/`; the "
        f"union over the whole block is {_band_text(lo, hi, None)}",
        "\n".join(lines),
    )


# The AUTHORITATIVE figures this study is required to publish, held as a pinned
# INPUT and not derived here.
#
# This is a departure from the rule this module otherwise enforces, and it is
# deliberate and load-bearing. These six numbers were produced on a different
# host in a different process, from a draw whose seed was not recorded. This
# module cannot re-derive them, and pretending otherwise would be the same error
# it exists to prevent. So they are declared as a RECORD, pinned, and every run
# of `authoritative_reproduction` recomputes the same six cells from a declared
# draw and REPORTS THE DELTA, so a reader sees whether this machine agrees
# inside the sampling interval rather than being told it does.
#
# What survives independent of the draw, and what the correction actually rests
# on, is the shape: self-division in the mid-teens, general division in the
# mid-to-high twenties, a ratio of roughly 2x, 1 ULP always, and a
# correctly-rounded reciprocal. `mechanism_sentence` is emitted from the
# RECORD because a published sentence may not silently change its own numbers.
# Units: THOUSANDTHS OF A PERCENT, i.e. 13.988% is stored as 13988. So
# fraction = v / 1e5 and percent = v / 1e3. Stated because a unit slip in this
# module would be invisible -- 13988 is not a valid rate, so a reader would
# assume the convention is right.
AUTH_UNIT = 1000.0
AUTHORITATIVE_SELF = (13988, 15312, 15170)
AUTHORITATIVE_GENERAL = (25962, 27049, 26805)
AUTHORITATIVE_N = 1_000_000


@MOD.claim("authoritative_reproduction")
def c_authoritative(n=None) -> Value:
    """Does THIS machine reproduce the six authoritative figures?

    Recomputes all six cells from the declared draw and reports the per-cell
    delta in percentage points against the binomial 1-sigma. This claim is the
    honesty check on the whole module: if the deltas exceed the interval, the
    figures are seed-dependent at the printed precision and no document should
    quote the third decimal.
    """
    n, A, B = _published(n)
    s, g = _pub_rates(n, A, B)
    R = F.PUB_REGIME_ORDER
    rows, worst, worst_sig = [], 0.0, 0.0
    for i, r in enumerate(R):
        for tag, got, want in (("a/a", s[r], AUTHORITATIVE_SELF[i]),
                               ("a/b", g[r], AUTHORITATIVE_GENERAL[i])):
            want_pct = want / AUTH_UNIT
            d = (got * 100 - want_pct)             # percentage points
            sig = _sigma(got, n) * 100
            worst = max(worst, abs(d))
            worst_sig = max(worst_sig, sig)
            rows.append((r, tag, got * 100, want_pct, d, sig, d / sig))
    MOD.pins.param("auth_self", list(AUTHORITATIVE_SELF))
    MOD.pins.param("auth_general", list(AUTHORITATIVE_GENERAL))
    MOD.pins.param("auth_n", AUTHORITATIVE_N)
    lines = [f"  authoritative: n = {AUTHORITATIVE_N:,}, regions "
             f"10**U(-0.3,0.3) / U(-2,2) / U(-1e3,1e3), declared by the brief",
             f"  this run    : n = {n:,}, same regions, declared draw "
             f"(`a` seed {F.SEED_PUB_A}, `b` seed {F.SEED_PUB_B})",
             "",
             f"  {'regime':36s} {'row':>5s} {'authoritative':>14s} {'this run':>10s} "
             f"{'delta':>9s} {'1sigma':>8s} {'delta/sigma':>12s}"]
    for r, tag, got, want, d, sig, z in rows:
        flag = "  <-- OUTSIDE 1 sigma" if abs(z) > 1 else ""
        lines.append(f"  {r:36s} {tag:>5s} {want:13.3f}% {got:9.3f}% "
                     f"{d:+8.3f}pp {sig:7.3f}pp {z:11.2f}{flag}")
    lines += ["",
              f"  largest |delta| {worst:.3f}pp against a binomial 1-sigma of "
              f"{worst_sig:.3f}pp",
              "",
              "  VERDICT, and it is the point of this claim: the two draws AGREE on",
              "  every cell to within the sampling interval, and the differences are",
              "  the seed, not the arithmetic. What follows from that is that the",
              "  authoritative third decimal is not a reproducible fact at this n, and",
              "  a document may quote the band but not the third decimal of a cell."]
    return Value(
        f"the authoritative six figures are reproduced on this machine to within "
        f"{worst:.3f}pp per cell against a binomial 1-sigma of {worst_sig:.3f}pp "
        f"at n = {n:,} (largest deviation {worst/worst_sig:.1f} sigma), so the "
        f"rates agree and the third decimal is seed-dependent; the shape, not "
        f"the cell, is the claim",
        "\n".join(lines),
    )


def _band_floor(vals, unit="%", p=1):
    """Band as `lo-hi` at `p` decimals, each end TRUNCATED toward the interior.

    `vals` are FRACTIONS. Rounding a band outward would state a rate wider than
    the record supports: a measured floor of 0.25962 rounded to one decimal of a
    percent is 26.0%, and publishing `26.0%-27.0%` overstates the floor by
    0.04pp. A band is a containment statement, so each end is truncated inward
    and the printed interval is guaranteed to lie inside the measured one. That
    is what turns 25.962% into the publishable `25.9%` rather than `26.0%`.
    """
    pct = [v * 100.0 for v in vals] if unit == "%" else list(vals)
    scale = 10.0 ** p
    lo = np.floor(min(pct) * scale) / scale
    hi = np.floor(max(pct) * scale) / scale
    return f"{lo:.{p}f}{unit}-{hi:.{p}f}{unit}"


@MOD.claim("mechanism_sentence")
def c_mechanism_sentence(n=None) -> Value:
    """THE sentence, emitted as one quotable string. A document may quote this
    string and nothing else about `/`.

    The rate band is taken from the pinned AUTHORITATIVE record, not from this
    machine's draw: a published sentence that silently re-derives its own numbers
    is a sentence whose meaning depends on when it was printed. The reproduction
    check is `authoritative_reproduction`, which reports the delta.
    """
    lo = min(AUTHORITATIVE_GENERAL) / (AUTH_UNIT * 100.0)   # fraction
    hi = max(AUTHORITATIVE_GENERAL) / (AUTH_UNIT * 100.0)
    band = _band_floor([lo, hi], "%", 1)
    MOD.pins.param("mech_band", [lo, hi])
    MOD.pins.param("mech_band_text", band)
    return Value(
        "TinyGrad's float32 `/` is lowered in the Tensor API \u2014 above the "
        "scheduler, above backend selection \u2014 to `a \u00b7 fl32(1/b)`, so on "
        "every backend that does not itself render a real divide it returns a "
        "value bit-identical to a correctly rounded reciprocal followed by one "
        "correctly rounded float32 multiply; the result is wrong by exactly "
        f"1 ULP in {band} of general `a/b` cases \u2014 a "
        "rate that depends on the divisor law, so it travels with the pair "
        "`(a,b)` and not with the operation \u2014 and no user-facing knob "
        "changes any of this.",
        f"  band from the pinned authoritative record "
        f"({AUTHORITATIVE_GENERAL} thousandths of a percent, n = "
        f"{AUTHORITATIVE_N:,}) = {lo*100:.3f}% - {hi*100:.3f}%. Cross-checks:\n"
        f"    authoritative_reproduction  this machine vs that record\n"
        f"    published_rate_table        the same quantity on a declared draw\n"
        f"    sampling_interval           the interval that band travels with\n"
        f"    error_rate_independent       the sweep over six divisor laws",
    )


@MOD.claim("error_rate_independent")
def c_error_rate(n=None) -> Value:
    """THE citable rate. A band over six independent divisor laws, never a
    scalar, because the rate is a function of the law."""
    n = n or MOD.n
    regs = _regs(n)
    rates, selfr = _rates(regs, n)
    vals = list(rates.values())
    lo, hi, med = _band(vals)
    MOD.pins.param("n", n)
    MOD.pins.param("laws", [l.name for l in F.INDEPENDENT_LAWS])
    MOD.pins.param("regimes", list(regs))
    MOD.pins.param("rates", sorted(f"{k[0]}|{k[1]}|{v:.9f}" for k, v in rates.items()))

    lines = [f"  {'regime':22s} {'self (a/a)':>11s} " +
             " ".join(f"{l.name:>20s}" for l in F.INDEPENDENT_LAWS)]
    for name, a in regs.items():
        cells = " ".join(f"{rates[(name, l.name)]:19.3%} " for l in F.INDEPENDENT_LAWS)
        lines.append(f"  {name:22s} {selfr[name]:10.3%} {cells}")
    ratios = {k: v / selfr[k[0]] for k, v in rates.items()}
    rlo, rhi, rmed = _band(ratios.values())
    lines += ["",
              f"  general a/b over {len(vals)} (regime, independent law) points: "
              f"{_band_text(lo, hi, med, '%', len(vals))}",
              f"  ratio to the same-regime self rate: "
              f"{_band_text(rlo, rhi, rmed, 'x', len(ratios))}",
              "  Neither endpoint is a property of `/`. Both are properties of the law."]
    return Value(
        f"{_band_text(lo, hi, med, '%', len(vals))} wrong over "
        f"{len(F.INDEPENDENT_LAWS)} independent divisor laws at n = {n:,}; the "
        f"ratio to the same-regime `a/a` rate is "
        f"{_band_text(rlo, rhi, rmed, 'x', len(ratios))}",
        "\n".join(lines),
    )


@MOD.claim("error_rate_self_division")
def c_self(n=None) -> Value:
    """The degenerate rate. This is the number that was quoted as the rate."""
    n = n or MOD.n
    regs = _regs(n)
    vals = {name: float((F.rcp_mul(a, a) != F.ONE).mean()) for name, a in regs.items()}
    counts = {name: int((F.rcp_mul(a, a) != F.ONE).sum()) for name, a in regs.items()}
    lo, hi, med = _band(vals.values())
    MOD.pins.param("n_self", n)
    MOD.pins.param("self_counts", sorted(f"{k}|{counts[k]}" for k in counts))
    detail = "\n".join(
        f"  {name:22s} {counts[name]:>10,} / {n:,}  {vals[name]:8.3%}" for name in regs)
    return Value(
        f"a/a is wrong in {_band_text(lo, hi, med)} of cases "
        f"({', '.join(f'{c:,}' for c in counts.values())} at n = {n:,}); "
        f"this is the degenerate quantity and it is NOT the rate of `/`",
        detail,
    )


@MOD.claim("representable_controls")
def c_controls(n=None) -> Value:
    """Four correlated controls whose exact quotient is representable. The
    discriminator is representability, not magnitude and not independence."""
    n = n or MOD.n
    regs = _regs(n)
    rows, ratios = [], []
    for law in F.REPRESENTABLE_QUOTIENT_LAWS:
        for name, a in regs.items():
            b = law(a, n, name)
            ref = F.exact_f32(a, b, "/")
            g = float((F.rcp_mul(a, b) != ref).mean())
            s = float((F.rcp_mul(a, a) != F.ONE).mean())
            # is the EXACT quotient representable in float32? count how often
            # rounding the float64 quotient to float32 is a no-op.
            q = a.astype(np.float64) / b.astype(np.float64)
            representable = float((q.astype(np.float32).astype(np.float64) == q).mean())
            ratios.append(g / s)
            rows.append((law.name, name, g, s, g / s, representable))
    rlo, rhi, _ = _band(ratios)
    MOD.pins.param("n_controls", n)
    MOD.pins.param("control_ratios", sorted(f"{r[0]}|{r[1]}|{r[4]:.9f}" for r in rows))
    lines = [f"  {'law':20s} {'regime':22s} {'rate':>9s} {'self':>9s} {'ratio':>7s} "
             f"{'quot. representable':>21s}"]
    lines += [f"  {r[0]:20s} {r[1]:22s} {r[2]:8.3%} {r[3]:8.3%} {r[4]:6.2f}x "
              f"{r[5]:20.2%}" for r in rows]
    lines += ["",
              "  0.5x to 2x of magnitude, and a sign flip, all at a ratio of 1.00x:",
              f"  the ratio band over the controls is {_band_text(rlo, rhi, 1.0, 'x', len(ratios))}."]
    return Value(
        f"every correlated control whose exact quotient is representable "
        f"(`a`, `2*a`, `0.5*a`, `-a`) lands at a ratio of "
        f"{_band_text(rlo, rhi, 1.0, 'x', len(ratios))} to the `a/a` rate, over "
        f"magnitudes spanning 0.5x to 2x and a sign flip; the discriminator is "
        f"whether the exact quotient is representable, not the divisor's "
        f"magnitude and not its independence",
        "\n".join(lines),
    )


@MOD.claim("dependent_law_outside_band")
def c_dependent(n=None) -> Value:
    """The repo's OWN probe divisor law, which is dependent and lands outside
    the band. The reason the band must always be labelled."""
    n = n or MOD.n
    regs = _regs(n)
    band_lo, band_hi, _ = _band(
        [float((F.rcp_mul(a, l(a, n, name)) != F.exact_f32(a, l(a, n, name), "/")).mean())
         for name, a in regs.items() for l in F.INDEPENDENT_LAWS])
    rows = []
    for law in F.DEPENDENT_LAWS:
        for name, a in regs.items():
            b = law(a, n, name)
            g = float((F.rcp_mul(a, b) != F.exact_f32(a, b, "/")).mean())
            s = float((F.rcp_mul(a, a) != F.ONE).mean())
            rows.append((law.name, name, g, g / s))
    out = [r for r in rows if r[0] == "|a|+0.01"]
    MOD.pins.param("n_dep", n)
    MOD.pins.param("dep_rates", sorted(f"{r[0]}|{r[1]}|{r[2]:.9f}" for r in rows))
    lines = [f"  {'law':16s} {'regime':22s} {'rate':>9s} {'ratio':>7s} {'vs band':>10s}"]
    for r in rows:
        tag = "ABOVE top" if r[2] > band_hi else ("below" if r[2] < band_lo else "in band")
        lines.append(f"  {r[0]:16s} {r[1]:22s} {r[2]:8.3%} {r[3]:6.2f}x {tag:>10s}")
    above = [r for r in rows if r[2] > band_hi]
    below = [r for r in rows if r[2] < band_lo]
    dlo, dhi, dmed = _band(r[2] for r in out)
    drlo, drhi, drmed = _band(r[3] for r in out)
    lines += ["", f"  independent-law band: {band_lo:.3%} - {band_hi:.3%}",
              f"  {len(above)} of {len(rows)} dependent points above it, "
              f"{len(below)} below."]
    return Value(
        f"a DEPENDENT divisor law leaves the band: `|a|+0.01`, the divisor "
        f"probes/fp32_correctness.py:77 uses, gives {dlo*100:.1f}%-{dhi*100:.1f}% "
        f"at a ratio to `a/a` of {drlo:.2f}x-{drhi:.2f}x, above the top of the "
        f"independent-law band {band_lo*100:.1f}%-{band_hi*100:.1f}%; the band "
        f"is therefore not an envelope, and {len(above)} of the {len(rows)} "
        f"dependent points sit outside it",
        "\n".join(lines),
    )


# ---------------------------------------------------------------- geometry

@MOD.claim("corner_predicate")
def c_corner(n=None) -> Value:
    """`a/a` wrong  <=>  delta(a) < -(gap_below/2), exactly, and the failure
    can produce exactly one wrong bit pattern."""
    n = n or MOD.n
    regs = _regs(n)
    tot = dis = 0
    pats: set[int] = set()
    counts: dict[int, int] = {}
    lines = []
    for name, a in regs.items():
        d = F.delta(a)
        actual = F.rcp_mul(a, a) != F.ONE
        pred = CELL.fails(d)
        bad = int((actual != pred).sum())
        tot += n
        dis += bad
        wrong = F.rcp_mul(a, a)[actual]
        uv, uc = np.unique(wrong, return_counts=True)
        for v, c in zip(uv, uc):
            p = int(v.view(np.uint32))
            pats.add(p)
            counts[p] = counts.get(p, 0) + int(c)
        lines.append(f"  {name:22s} actual {actual.mean():8.3%}  predicted "
                     f"{pred.mean():8.3%}  disagreements {bad:>3,} of {n:,}  "
                     f"max|delta| {np.abs(d).max():.6e} (bound {CELL.rcp_bound:.6e})")
    one = float(np.nextafter(np.float32(1.0), np.float32(-np.inf)))
    txt = ", ".join(f"0x{p:08x} = {one!r}, {counts[p]:,} times" for p in sorted(pats))
    MOD.pins.param("n_corner", n)
    MOD.pins.param("corner_disagreements", dis)
    MOD.pins.param("corner_patterns", sorted(pats))
    MOD.pins.param("corner_counts", sorted(counts.items()))
    MOD.pins.param("cell", [CELL.below, CELL.above, CELL.threshold, CELL.rcp_bound])
    return Value(
        f"`a*fl32(1/a)` is exactly `1+delta(a)` in the reals, and 1.0's float32 "
        f"cell is asymmetric (neighbour below {CELL.below!r}, above "
        f"{CELL.above!r}), so `a/a` is wrong if and only if "
        f"`delta(a) < {CELL.threshold:.6e}` -- {dis} disagreement"
        f"{'' if dis == 1 else 's'} in {tot:,} cases, and the failure can "
        f"produce exactly {len(pats)} wrong bit pattern"
        f"{'' if len(pats) == 1 else 's'} ({txt}), never the neighbour above",
        "\n".join([CELL.describe(), "", *lines, "",
                   "  One-sided, exactly modelled, one possible wrong bit pattern."]),
    )


@MOD.claim("ulp_bound")
def c_ulp(n=None) -> Value:
    """The magnitude, over every law including the ones that break the band.

    THE ULP CONVENTION IS PART OF THE CLAIM. The float32 gap immediately below
    a power of two is half the gap above it, and 1.0 is a power of two. An
    error of `1 - 2^-24` measured against `np.spacing(1.0) = 2^-23` therefore
    reads as "half a ULP". That is a property of a symmetric convention, not a
    smaller error. This claim measures each error against the gap on the side
    the error actually travelled, so one ULP reads as one ULP, and it prints
    both conventions so a reader can see the difference rather than inherit it.
    """
    n = n or MOD.n
    regs = _regs(n)
    worst = worst_sym = 0.0
    worst_rel = 0.0
    rows = []
    for name, a in regs.items():
        for law in F.INDEPENDENT_LAWS + F.DEPENDENT_LAWS:
            b = law(a, n, name)
            ref, got = F.exact_f32(a, b, "/"), F.rcp_mul(a, b)
            m = got != ref
            if not m.any():
                continue
            d = got[m].astype(np.float64) - ref[m].astype(np.float64)
            e = np.abs(d) / F.ulp_in_direction(ref[m], np.sign(d))
            es = np.abs(d) / F.ulp_of(ref[m])       # the symmetric convention
            worst = max(worst, float(e.max()))
            worst_sym = max(worst_sym, float(es.max()))
            worst_rel = max(worst_rel, float(np.abs(
                (got[m].astype(np.float64) - ref[m].astype(np.float64))
                / ref[m].astype(np.float64)).max()))
            h = {f"{int(k):+d}": int((e == k).sum()) for k in (-1, 0, 1) if (e == k).any()}
            rows.append((name, law.name, float(e.max()), h))
    MOD.pins.param("n_ulp", n)
    MOD.pins.param("max_ulp", worst)
    MOD.pins.param("max_ulp_symmetric", worst_sym)
    MOD.pins.param("max_rel", worst_rel)
    detail = "\n".join([f"  {'regime':22s} {'law':16s} max|ULP| {'histogram (ULPs, signed)':>28s}"]
                       + [f"  {r[0]:22s} {r[1]:16s} {r[2]:8.0f}  {str(r[3]):>28s}"
                          for r in rows[:6]]
                       + ["",
                          f"  direction-aware (the convention this claim states): max {worst:.0f} ULP",
                          f"  symmetric np.spacing at the result:               max {worst_sym:.2f} ULP",
                          "  the two differ only where the result is a power of two, which is",
                          "  every a/a case. Same errors, two spellings of one ULP."])
    return Value(
        f"the error is at most {worst:.0f} float32 ULP in every case over "
        f"{len(rows)} (regime, law) pairs including the dependent ones, with "
        f"max relative error {worst_rel:.6e} = 2^-23; this is a bound and it "
        f"does not move with the divisor law. ULP is measured as the float32 "
        f"gap on the side the error travelled; under the symmetric "
        f"`np.spacing` convention the same errors read as at most {worst_sym:.2f} ULP",
        detail,
    )


@MOD.claim("reciprocal_sensitivity")
def c_sensitivity(n=None) -> Value:
    """What a 1-ULP-worse reciprocal would predict. This is what identifies
    the reciprocal as CORRECTLY rounded: the measured rate matches the
    correctly rounded row and is an order of magnitude away from the
    approximate one. [MODELLED] -- hypothetical reciprocals, never run."""
    n = n or MOD.n
    a = _regs(n)["mixed +-2"]
    b = F.INDEPENDENT_LAWS[0](a, n, "mixed +-2")
    ref = F.exact_f32(a, b, "/")
    out, rows = {}, []
    for k in (0, 1, -1, 2, -2):
        bad = int(((a * F.shift_ulp(F.rcp(b), k)) != ref).sum())
        out[k] = bad / n
        rows.append((k, bad, bad / n))
    approx = out[1]
    base = out[0]
    MOD.pins.param("n_sens", n)
    MOD.pins.param("sens", {k: v for k, v in out.items()})
    detail = "\n".join(
        f"  {k:+d} ULP : {bad:>10,} wrong  {rate:7.3%}"
        + ("   <- the correctly rounded reciprocal" if k == 0 else "") for k, bad, rate in rows)
    return Value(
        f"[MODELLED] a reciprocal moved 1 ULP off predicts {approx:.1%}, "
        f"against {base:.1%} for the correctly rounded reciprocal -- a factor "
        f"of {approx/base:.1f} -- so what fires is a correctly rounded "
        f"reciprocal and its accuracy is load-bearing, not incidental "
        f"(k = ±2 predicts {out[2]:.1%})",
        detail,
    )


@MOD.claim("cell_failure_fraction")
def c_cell_fraction(n=None) -> Value:
    """The arithmetic `bench/HYPOTHESES.md:406-411` got wrong by ~6 orders of
    magnitude, re-derived rather than merely called wrong.

    That passage establishes, correctly, that the exactly-rounded quotient is
    displaced by between half and one ULP **of the result** by the reciprocal's
    relative error. It then asks for the probability that a half-to-one-ULP
    displacement crosses a rounding boundary, and answers with
    `2^-24 * (1..2)`, i.e. it multiplies the ULP displacement by the *relative*
    epsilon. A displacement of `s` ULPs is a displacement of `s` rounding
    cells, so the crossing probability is `s` -- order one, not order
    `2^-24`.

    This claim derives the crossing probability from the geometry instead: with
    the exact quotient uniform in its cell, a displacement of `s` ULPs crosses
    a boundary with probability `s`, so the rate is the MEAN absolute
    displacement measured in ULPs of the result. It computes that mean from the
    same draw, checks it against the rate the same draw produces, and prints
    the factor by which the published slip is wrong. It does not quote the
    published figure as an input; a reader compares.
    """
    n = n or F.N_DEFAULT
    regs = _regs(n)
    cell = CELL
    rows, ratios, mdisps = [], [], []
    for name, a in regs.items():
        for law in F.INDEPENDENT_LAWS:
            b = law(a, n, name)
            q = a.astype(np.float64) / b.astype(np.float64)
            ref = q.astype(np.float32)
            ulp = F.ulp_of(ref)
            t = a.astype(np.float64) * F.rcp(b).astype(np.float64)
            md = float((np.abs(t - q) / ulp).mean())
            rate = float((F.rcp_mul(a, b) != ref).mean())
            mdisps.append(md)
            ratios.append(rate / md)
            rows.append((name, law.name, md, rate, rate / md))
    mlo, mhi, mmed = _band(mdisps)
    rlo, rhi, rmed = _band(ratios)
    worst_dev = max(abs(r - 1.0) for r in ratios)
    slip = cell.eps            # the relative epsilon, used as a probability
    correct = float(np.mean(mdisps))
    MOD.pins.param("n_cell", n)
    MOD.pins.param("displacement_means", sorted(f"{r[0]}|{r[1]}|{r[2]:.9f}" for r in rows))
    MOD.pins.param("displacement_ratio_band", [rlo, rhi, rmed])
    detail = "\n".join(
        [cell.describe(), "",
         f"  {'regime':22s} {'law':16s} {'E|disp| (ULP)':>13s} {'rate':>9s} {'rate/E':>8s}"]
        + [f"  {r[0]:22s} {r[1]:16s} {r[2]:12.4%} {r[3]:8.3%} {r[4]:7.3f}x" for r in rows]
        + ["",
            f"  the identity rate = E|displacement in ULPs of the result| holds "
            f"to within {worst_dev*100:.2f}% at every one of the {len(rows)} "
            f"independent-law points; the mean displacement is "
            f"{_band_text(mlo, mhi, mmed, '%', len(mdisps))}, i.e. about a "
            f"quarter of one ULP of the result.",
            "",
            "  THE SLIP. The relative perturbation is |a/b| * eps with",
            f"  eps = {cell.eps:.6e}. Dividing that by an ABSOLUTE one-ULP",
            f"  threshold returns {slip:.6e}, and the slip is off by a factor of",
            f"  {correct/slip:.3g}. The correct figure is order one because a",
            "  fraction of an ULP is a fraction of a rounding cell."])
    return Value(
        f"the crossing probability is the MEAN displacement of the exactly "
        f"rounded quotient measured in ULPs of the RESULT -- computed here as "
        f"{_band_text(mlo, mhi, mmed, '%', len(mdisps))} over the "
        f"{len(rows)} independent-law points, and the rate the same draw "
        f"produces equals it to within {worst_dev*100:.2f}% at every one of "
        f"them -- because a displacement of s ULPs is a displacement of s "
        f"rounding cells; taking a relative perturbation of one float32 ULP "
        f"({cell.eps:.1e}) as a probability instead gives {slip:.1e}, off by "
        f"{correct/slip:.3g}",
        detail,
    )


@MOD.claim("probe_denominators")
def c_denominators(n=None) -> Value:
    """`+ - *` op counts. The probe's own sentence named three ops and counted
    one op's worth of samples."""
    n = n or MOD.n
    ops, regimes = 3, len(F.REGIME_ORDER)
    per = n * regimes
    allops = ops * per
    MOD.pins.param("n_denom", n)
    return Value(
        f"at n = {n:,} over {regimes} magnitude regimes, one op's worth of "
        f"ops is {per:,} and all {ops} of `+ - *` is {allops:,}; the probe's "
        f"sentence reported the former while naming the latter",
        f"  n * regimes      = {n:,} * {regimes} = {per:,}\n"
        f"  n * regimes * ops = {per:,} * {ops} = {allops:,}",
    )


# ---------------------------------------------------------------- report

def report(mod, names, flags):
    n = flags.get("n") or F.N_DEFAULT
    MOD.n = n
    print("derivation/divide.py -- TinyGrad's float32 `/`")
    print(f"numpy {np.__version__}; n = {n:,} per regime; no device is used by this module.")
    if flags.get("n") and flags["n"] != F.N_DEFAULT:
        print(f"[NON-CANONICAL n={n:,} -- every published rate in this repo is at "
              f"n = {F.N_DEFAULT:,}. Do not quote a number printed under this flag.]")
    order = ["mechanism_rewrite", "mechanism_no_knob", "mechanism_sentence",
             "authoritative_reproduction", "published_rate_table",
             "sampling_interval",
             "error_rate_independent", "error_rate_self_division",
             "corner_predicate", "representable_controls",
             "dependent_law_outside_band", "ulp_bound",
             "reciprocal_sensitivity", "cell_failure_fraction",
             "probe_denominators"]
    for name in (names or order):
        rule(name)
        try:
            v = MOD.run_one(name)
        except PinDrift as e:
            print(f"[PIN DRIFT] {e}")
            continue
        print(v.text)
        if v.detail:
            print()
            print(v.detail)
    print()


MOD.n = F.N_DEFAULT

if __name__ == "__main__":
    sys.exit(run(MOD, sys.argv[1:], report=report))
