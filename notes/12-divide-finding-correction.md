# 12 — the `/` finding was right and the magnitude was wrong by about 2x

**Status: correction APPLIED 2026-09-28.** The bug is fixed in
`probes/fp32_correctness.py`; the mechanism labels are fixed in
`probes/divide_precision.py`; the numbers are derived in `derivation/divide.py`;
and every file that quoted the rate has been corrected. The full location list is
`notes/14-divide-magnitude-propagation.md` §6 and the ledger rows are
`notes/06-corrections-ledger.md` **C-09** through **C-13**.

**Instruments:** `probes/fp32_correctness.py` (the probe, on a device) and
`derivation/divide.py` (the derivation, device-free, and the only place a document
may take a number from). See §9.

**Host for every `MEASURED` figure below:** Apple Silicon Mac, `Device.DEFAULT = METAL`,
CPU backend also measured. tinygrad 0.14.0, numpy 2.5.3, python 3.12.
**No GPU on this host** — see §7.

Status tags, used strictly throughout: `MEASURED` = an instrument ran here, in this
process. `RE-DERIVED` = recomputed here in numpy from an array the probe built; checkable
by hand. `MODELLED` = arithmetic on a hypothetical implementation, never run. `RECORD` =
read out of a shipped log in `out/`, not reproduced. Nothing in this note is transcribed
from a brief or from another note without being re-derived first, and where my number
disagrees with a figure already in the record I say so.

---

## 1. The bug

`probes/fp32_correctness.py:56-57`:

```python
for op, fn in (("+", ...), ("-", ...), ("*", ...), ("/", ...)):
  b = regimes[name] if op in "*/" else a     # regimes[name] IS a
```

`regimes[name]` is the same object as `a`. So the `/` row computes `a/a`, and the `*`
row computes `a*a`. **The probe has never divided by an independent divisor.** Every
number it published for `/` is a property of self-division.

`/a` is a degenerate test, and it is degenerate in a specific, provable way, not a
harmless one. §3 derives it. The short version: `a/a` must be exactly `1.0`, which is
*representable*, so the second rounding starts pinned at a **corner** of a rounding cell
whose two edges are **unequal**. The test is one-sided and can only fire on the lower
quarter of the reciprocal's error range. It was never a neutral choice of divisor.

---

## 2. The corrected magnitudes

**The published figures are 25.9%–27.0%, and the six cells behind them are
`derivation/divide.py:published_rate_table`.** That draw is a *different* one from
the one this section originally used, and the reason is worth stating: the
authoritative six numbers cannot be re-derived from a seed, because the seed was
not recorded with them. `derivation/divide.py:authoritative_reproduction`
recomputes all six on this machine and reports the delta — the largest is 0.101pp
against a binomial 1σ of 0.044pp, i.e. the two draws agree in shape and differ in
the third decimal. **So the third decimal is not a fact, and no document should
quote one.** The band is the claim.

The table below is the original measurement on the repo's own three regimes
(same `a`, drawn from `np.random.default_rng(0)` in the original order, so the
`a/a` column is the *same array* the original probe used and is directly
comparable). `MEASURED`, n = 2,097,152 per regime:

| regime | `b := a` — the old row | independent `b` | understated by |
|---|---|---|---|
| near 1.0 (LUT regime) | 12.410% | **26.517%** | 2.14x |
| mixed ±2 | 15.365% | **27.051%** | 1.76x |
| wide ±1e3 | 15.166% | **26.858%** | 1.77x |

The `b := a` column reproduces
`out/kaggle/20260928-014248-gpu/logs/fp32_correctness.log`
lines 10/17/24 (260,261 / 322,233 / 318,044) **exactly**, on a different vendor and a
different architecture — and that cross-check is now an assertion inside
`derivation/divide_backends.py:self_division_log_crosscheck`, not a side-by-side
the reader has to compare. §3 explains why that column travels and the right-hand
one does not.

The right-hand column is a rate **for this divisor law**; §2.1 gives the sweep.

`+`, `-` and `*` with independent operands: **0 wrong, 0.000%, on every regime and both
devices.** `MEASURED`. The `+ - * exact` conclusion is untouched; only its sample count
moved (the original denominator 6,291,456 is the per-op count, already flagged in
`notes/10` §1.5; the three-op total is 18,874,368 —
`derivation/divide.py:probe_denominators`).

### 2.1 "26.517%" is not a constant either, and the record's two sets disagree

`notes/10-t4-results.md:215-227` gives the general `a/b` rate as
**25.020% / 28.127% / 27.973%**. `notes/11-local-suite-integrity.md:246-248` gives
**25.046% / 27.042% / 27.016%**. Those are different numbers for the same three regimes,
and neither was reproducible from this note's sweep — a general `a/b` rate is not a
function of `a` alone, it is a function of `(a, b)` jointly, and neither note states the
divisor law it used. So the correct statement is a band, and the band has to be published
with the divisor law attached.

**`notes/11`'s set is now reproducible, and the missing half was the law.**
`probes/oc/report_fix/fp32_div_row.py` names it: `b ~ U[0.01, 2.01]`, drawn from the
**continued** `default_rng(0)` stream, n = 2,097,152 — and running it returns 25.046 /
27.042 / 27.016 to the digit. So the earlier verdict "not reproducible" was half right:
not reproducible *without* the law, exactly reproducible *with* it. That is the
general lesson of this section, and it is the reason the fixed
`probes/fp32_correctness.py` prints its divisor law next to its numbers rather than
in a docstring nobody reads.

Sweeping six independent divisor laws against the same three `a` arrays, n = 2,097,152,
`RE-DERIVED` (`derivation/divide.py:error_rate_independent`):

| divisor law | near 1.0 | mixed ±2 | wide ±1e3 |
|---|---|---|---|
| same law as `a` | 26.479% | 27.043% | 26.888% |
| `U[0.01, 2.01]` | 25.101% | 27.050% | 27.127% |
| `U[0.5, 1.5]` | 24.952% | 24.758% | 24.761% |
| `U[-8, 8]` | 25.093% | 27.056% | 27.112% |
| `U[1, 2]` | 25.055% | 27.068% | 27.109% |
| `1/\|N(0,1)\|` | 22.899% | 22.718% | 22.748% |
| `2*a` (correlated control) | 12.410% | 15.365% | 15.166% |

**General `a/b` over the 18 independent combinations: 22.7% – 27.1%, median 25.8%.**
`RE-DERIVED`. Over that sweep the ratio to the self-division rate is **1.48x – 2.14x,
median 1.78x** — so the "~2x" in circulation is a **band, not a constant**, and the
specific 2.02x / 1.76x / 1.78x in `notes/11` is one divisor law's worth of it, not a
property.

**But this union is not the rate, and publishing it as one was this note's own
error.** The 22.7% floor is the `1/|N(0,1)|` law — real, but one law of six that
*was chosen*, so the union answers "what is the worst case across the laws I
picked", not "what is the error rate of `/`". The citable rate is narrower and
carries one law: **25.9%–27.0%** (`derivation/divide.py:mechanism_sentence`). The
sweep stays as a sweep, in its own claim, under its own name.

**The last row is the control and it is the load-bearing one.** `2*a` is a divisor that
is a deterministic function of `a`, so the exact quotient `a/(2a)` is again always exactly
`0.5` — representable, at a cell corner — and the rate collapses to the self-division
rate, ratio **1.00x in all three regimes**. Nothing about the divisor's *magnitude*
moved; only whether the exact quotient is representable. That is the mechanism in §3,
confirmed by a case where the prediction is falsifiable and it is not falsified.

---

## 3. The mechanism, derived

`/` is lowered in the **Tensor API**, at `mixin/elementwise.py:251`, to
`a.alu(Ops.MUL, b.reciprocal())` — above the scheduler and above backend selection — so
no `Ops.FDIV` ever exists in the IR. `renderer/cstyle.py:137` renders
`Ops.RECIPROCAL` as `(1/{x})`, so the cstyle backends compute

```
t = fl32( a * fl32(1/b) )
```

The second rounding is **correctly rounded** — a product of two float32s needs ≤ 48 bits
and float64 has 53, so the single rounding down is the right one. **The only bad rounding
is the reciprocal.** Write

```
delta(x) = fl32(1/x) / (1/x) - 1        |delta| <= 2^-24 for a correctly rounded reciprocal
t        = a * fl32(1/b) = (a/b) * (1 + delta(b))        exactly, in the reals
```

The result is wrong **iff `t` leaves the float32 rounding cell that contains the exact
quotient `a/b`**. That is the whole mechanism, and it is a test of *where the exact
quotient sits inside its cell*. The one-sentence statement of it, with its band, is
`derivation/divide.py:mechanism_sentence`, and that is what a document should quote.

**"Correctly rounded" here is a claim with a price, and the price is what makes
it a finding.** It is not decoration on the mechanism: a reciprocal 1 ULP *worse*
than correctly rounded predicts **~86%** wrong against the **~27%** measured, a
factor of 3.2 (`derivation/divide.py:reciprocal_sensitivity`). So the reciprocal's
accuracy is load-bearing, and the phrase "bit-identical to a correctly rounded
reciprocal" is a **separate claim from the percentage**. An earlier version of
this record claimed the rate was "insensitive to whether the reciprocal itself is
exact"; that is retracted, and it is retracted by the same number that a third of
this note was built on.

### 3.1 The failure predicate is exact

`RE-DERIVED`, `fp32_correctness_fixed.py` §5a, n = 2,097,152. The `a/a` rows are
now asserted against the shipped log in
`derivation/divide_backends.py:self_division_log_crosscheck` rather than printed
side by side:

```
regime                 case   predicted    actual    disagreements   midpoint ties
near 1.0 (LUT regime)  a/a      12.410%   12.410%                0                0
near 1.0 (LUT regime)  a/b      26.517%   26.517%                0                0
mixed +-2              a/a      15.365%   15.365%                2                5
mixed +-2              a/b      27.051%   27.051%                6               19
wide +-1e3             a/a      15.166%   15.166%                0                4
wide +-1e3             a/b      26.858%   26.858%                9               17
```

0–9 disagreements in 12,582,912 cases, and every one is an exact round-half-to-even tie at
a cell midpoint. The predicate is exact up to ties, so what follows is arithmetic, not
interpretation.

### 3.2 Why `a/a` cannot see what `a/b` sees

`r = (a/b − fl32(a/b)) / ULP(fl32(a/b))`, measured from the **centre** of the cell, so
`r ∈ [−0.5, +0.5]`. `RE-DERIVED`, `fp32_correctness_fixed.py` §5b:

| regime | case | `E\|r\|` | `P(r == 0)` | `P(r < 0)` | failure rate |
|---|---|---|---|---|---|
| near 1.0 | `a/a` | **0.0000** | **100.00%** | 0.00% | 12.410% |
| near 1.0 | `a/b` | 0.2501 | 0.00% | 50.07% | 26.517% |
| mixed ±2 | `a/a` | **0.0000** | **100.00%** | 0.00% | 15.365% |
| mixed ±2 | `a/b` | 0.2499 | 0.00% | 50.04% | 27.051% |
| wide ±1e3 | `a/a` | **0.0000** | **100.00%** | 0.00% | 15.166% |
| wide ±1e3 | `a/b` | 0.2500 | 0.00% | 49.95% | 26.858% |

Two facts, and the second is the whole gap:

1. **`a/a` is corner-pinned.** The exact quotient is `1.0`, a representable float, so
   `r ≡ 0`. The second rounding starts at the *edge* of a cell, not the middle.
2. **That cell is asymmetric.** `1.0`'s neighbour below is `1 − 2⁻²⁴` and above is
   `1 + 2⁻²³`. So the safe window is `[1 − 2⁻²⁵, 1 + 2⁻²⁴)` — a window that is
   **1.5 · 2⁻²⁴ wide, not 2 · 2⁻²⁴**, and whose lower edge is at half the upper edge's
   height. General division instead gets `r` uniform on `[−0.5, +0.5]`, i.e. an average
   threshold of **0.25 ULP on either side** where `a/a` has **0.5 ULP on one side only**.

So `a/a` is a *one-sided* test at 0.5 ULP and `a/b` is a *two-sided* test at 0.25 ULP. On
an unbiased perturbation that is a factor of ~2, and it is a property of **the test**, not
of the operands.

### 3.3 The sharpest form: `a/a` fails only on a downward reciprocal error

Because `a · (1/a) = 1` exactly in the reals, the product `a · fl32(1/a)` is **exactly**
`1 + delta(a)`. So the criterion collapses to a closed form:

```
a/a is wrong  <=>  delta(a) < -2^-25
```

`RE-DERIVED`, `fp32_correctness_fixed.py` §5c: predicted vs actual, **0 / 2 / 0
disagreements** over 6,291,456 samples. And `delta` is unbiased — `P(delta < 0) = 0.5015`
averaged over the three regimes — so this is **not** a bias in the reciprocal, it is
geometry.

Enumerate the wrong values and there is nothing left to interpret. `MEASURED`:

```
near 1.0 (LUT regime)  METAL   1 distinct wrong value: 0x3f7fffff  x 260,261
mixed +-2              METAL   1 distinct wrong value: 0x3f7fffff  x 322,233
wide +-1e3             METAL   1 distinct wrong value: 0x3f7fffff  x 318,044
                       CPU     0 distinct wrong values
```

`0x3f7fffff = 1.0 − 2⁻²⁴`, the neighbour **below** 1.0. `1.0 + 2⁻²³ = 0x3f800001`, the
neighbour **above** 1.0, is **never produced** — not once, in any regime, on any device
measured. The upward half of self-division is arithmetically unreachable. The original
probe's `/` row was not merely self-division; it was *the only half of self-division that
can fail*, and no choice of seed or sample count could ever have raised it.

> **This is also the correction to the record's stated mechanism.** `notes/02:328-329`
> and `notes/04:740-741` read `a/a` returning `1 − 2⁻²⁴` as "the signature of a reciprocal
> approximation"; so does `bench/HYPOTHESES.md:412`. It is the signature of a **correctly
> rounded** reciprocal whose error landed between `−2⁻²⁵` and `−2⁻²⁴`. A one-sided wrong
> value whose bit pattern the model reproduces to the last bit is the *opposite* of an
> approximation signature.
>
> **All three are now corrected in place** (ledger **C-10**), and the argument is
> `derivation/divide.py:corner_predicate`: the wrong-value set has exactly one
> member, the neighbour above is never produced, and the predicate
> `delta(a) < -2^-25` reproduces the failure to the bit with 0–9 disagreements
> in 12.6 M cases, every one an exact round-half-to-even tie at a cell midpoint.
> A correctly rounded reciprocal's error is *also* two-sided, so this alone does
> not exclude an approximation — the 86%-vs-27% sensitivity does. Together they
> are decisive, and that is why the mechanism sentence carries both clauses.

---

## 4. Retraction: the rate is NOT insensitive to the reciprocal's accuracy

`notes/10-t4-results.md:235-242` quotes and already corrects a synthesis sentence in this
study that read, in part, *"the probability of landing 1 ULP off is insensitive to whether
the reciprocal itself is exact."* **That clause is false and is retracted here.**

`MODELLED` — hypothetical reciprocals, nothing below ran on a device. Divisors `U[0.01,
2.01]` against `a ~ U[−2, 2]`, n = 2,097,152; the k = 0 row is the correctly rounded
reciprocal and coincides with the measured 27.051%:

> **The figures in the table below are now `derivation/divide.py:reciprocal_sensitivity`,
> which prints them at one decimal.** An earlier pass of this note reported
> 86.48% and `notes/14` reported 86.42%; those are the same quantity at
> different `n`, not a disagreement. What carries is the **comparison** —
> ~86% predicted against ~27% measured, a factor of 3.2 — and a claim of that
> shape does not need the third decimal to be right.

| hypothetical reciprocal | wrong | predicted rate |
|---|---|---|
| correctly rounded (k = 0) | 567,311 | **27.1%** — this is what the device measured |
| 1 ULP worse (k = ±1) | 1,814,934 | **86.5%** |
| 2 ULP worse (k = ±2) | 2,096,118 | **100.0%** |

A 1-ULP-approximate reciprocal — the error class of `rcp.approx.f32` — predicts ~86%, not
~27%. The measured rate is 27%. So:

- **What fires is a correctly rounded reciprocal**, and the reciprocal's own accuracy is
  **load-bearing**, not incidental. A one-ULP change in the reciprocal moves the error
  rate by 3.2x.
- The correct replacement sentence is narrower than either version in the record:
  > The rate is set by the rewrite **and** it is extremely sensitive to the reciprocal's
  > accuracy and insensitive to the rewrite's downstream rounding (the multiply is
  > correctly rounded and contributes zero failures). Backends agree at 27% **because
  > they all use a correctly rounded reciprocal**, not because the rate is
  > mechanism-independent.

Independently, the same fact is established by bit-identity rather than by rate. On the
exact `divide_precision.py` inputs (`inputs("pm2", seed=0)`, N = 200,000), the array
`a * fl32(1/b)` has a mismatch set against correctly rounded of size exactly **54,270**.
`MEASURED` on this host, `CL`, `METAL` and `PYTHON` all return **54,270** and are
**bit-identical** to that array. `RECORD` for the T4: CUDA 54,270, PYTHON 54,270 — same
count. A different array whose mismatch set merely had the same *size* would be a
1-in-200,000 accident; five independent backends and hosts landing on the same integer is
not.

### 4.1 The mechanism *label* printed by `probes/divide_precision.py` was wrong — FIXED

**Status: fixed 2026-09-28.** `probes/divide_precision.py:120-121` printed,
unconditionally, as commentary:

> "=> tinygrad's renderer emits `a*(1/b)`, not `a/b`. The backend then contracts the
> `1/b` into an approximate reciprocal."

The first clause was a measurement. **The second clause was an assumption, printed
without being tested, and it is false on every backend measured here.** It was also
the source of the phrase in `kernels/README.md:200` ("bit-for-bit `a*rcp.approx(b)`"),
which is a *label*, not a result: what was tested there is bit-identity against
`a * fl32(1/b)`, i.e. `div.rn.f32` semantics, which is exactly nvcc's
`-prec-div=true` default. Verified from the installed source:
`runtime/support/compiler_cuda.py:48` sets `compile_options` to `--gpu-architecture`
and nothing else, `:51` adds `-I`, `:53` adds `--minimal` on CUDA ≥ 12.4, the string
`use_fast_math` **appears nowhere in tinygrad at all** (whole-package grep, 0 hits)
and `prec-div` likewise (0 hits). The label now reads `a*fl32(1/b)`, i.e. a
correctly rounded reciprocal.

**The same false clause appeared in three more places in that same file and all
four are now corrected** — the docstring's item 1, its concluding paragraph, the
step-4 commentary, and the "no knob" section. The last of those also carried a
*backwards* claim, corrected separately: the probe said "`/` does not go through
`Ops.RECIPROCAL`", when `/` always does, and that is the defect. Ledger **C-10**
and **C-11**.

**Line-number drift, flagged while verifying the above.** The record cites
`mixin/elementwise.py:465` for `reciprocal()` (`notes/06` C-07); in the tinygrad 0.14.0
installed here it is **`:457`**, and the `Ops.RECIPROCAL` return is what sits at `:465`.
Every line-number citation in §6 was re-checked against the installed package; the ones
this note relies on (`elementwise.py:251`, `cstyle.py:137`, `cstyle.py:271,274`,
`decomp/op.py:133-136`, `compiler_cuda.py:48,51,53`) are correct — but the file has moved
since parts of the record were written, and a citation in this repo should be re-read
rather than trusted.

---

## 5. CL is not host-independent

`RECORD`, `out/kaggle/20260928-014248-gpu/logs/divide_precision.log`, T4, N = 200,000,
`inputs("pm2", seed=0)`:

| backend | wrong | rate | probe's own verdict |
|---|---|---|---|
| CUDA | 54,270 | 27.13% | `== a*fl32(1/b) bit for bit` |
| PYTHON | 54,270 | 27.13% | `== a*fl32(1/b) bit for bit` |
| CPU | 0 | 0.00% | correctly rounded |
| **CL** | **59,529** | **29.76%** | **`differs from a*fl32(1/b)`** |

The log's own `a*rcp(b)` label is ambiguous — it is not `rcp.approx` — and the
`probes/divide_precision.py` version of it is now spelled out. An earlier pass of
this note and `kernels/README.md` both read the ambiguity as an approximation
claim; see §0.1 of `notes/14`.

`MEASURED` on this host, same inputs, same N, same seed:

| backend | `a/b` wrong | rate | bit-identical to `a*fl32(1/b)`? |
|---|---|---|---|
| CPU | 0 | 0.00% | no (it is correctly rounded) |
| **CL** | **54,270** | **27.13%** | **yes** |
| METAL | 54,270 | 27.13% | yes |
| PYTHON | 54,270 | 27.13% | yes |

**The same tinygrad CL backend, on two hosts, gives 27.13% and 29.76%, and only the first
is bit-identical to `a·fl32(1/b)`.** CL is therefore **not** host-independent, and
`notes/02:199`'s `CL | 27.13% (54,270) | yes | approximated` is a Mac measurement that
does not generalise — which `notes/10` §1.1 already recorded from the log, and which this
note now confirms from the other side with a measurement of its own.

Three consequences:

1. `notes/02:198-200`'s error-class column (`CL ... approximated`, `METAL ... approximated`)
   is wrong for all four of the failing backends, and is now corrected there. All of them
   are **double-rounded** in exactly one place: the rounding of `1/b`. "Double-rounded" is
   itself loose — the multiply is correctly rounded and contributes zero failures, so all
   the loss is the reciprocal's — and the mechanism sentence says so.
2. On this host CL and METAL are **bit-identical to each other** (26.513% / 27.178% /
   26.899% at n = 1,048,576, identical counts), so the Mac's OpenCL is almost certainly
   fronting the same GPU as Metal. The T4's CL is a different OpenCL implementation.
3. **CL's T4 mechanism is still unidentified and must not be asserted.** `notes/10` §1.3
   scored nine candidate mechanisms and none reproduces 59,529 exactly. I did not find
   one either. `[UNVERIFIED]`. What *is* established is that the T4's CL is a **third
   distinct mechanism**, which the probe's own "differs from rcp*mul" line already said.

---

## 6. Every file and line that must change

**Status: ALL APPLIED 2026-09-28.** The tables in §6.1 and §6.2 are kept as the
record of what each site said and what it now says. The corrected figures are
**25.9%–27.0%** with its divisor law
(`derivation/divide.py:mechanism_sentence`), the ratio is a **band**
(1.48x–2.14x over a law sweep, 1.77x–1.86x on a same-law draw), and the
`+ - *` denominator is **18,874,368**.

**One thing this list got wrong, and the same thing a prior audit got wrong.** A
`notes/12 §6.1` citation range (`notes/04-codegen-comparison.md:747-750`) was
found not to contain the text it was said to contain; a re-audit found ten such
references across this section. **So do not trust a line reference in this
section** — open the file at the line. What survives is the *claim* each row
makes, not its coordinates. The new convention supersedes them: a number is
cited to the claim in `derivation/` that prints it, and a line reference into
someone else's source is paired with the text it is supposed to carry.

### 6.1 The wrong magnitude — general `a/b` is 25.9%–27.0%, not 12.4%–15.4%

**SUPERSEDED IN ITS FIGURES, upheld in its structure.** The claim this section
makes is right and every file it names had to be changed. The numbers it carried
are replaced, because they were produced by a probe that measured
`a*fl32(1/b)` in numpy over six divisor laws at n = 2,097,152 and reported the
*union* of those laws as "the" rate. The citable figure is narrower and comes
with a declared draw:

> general `a/b` is wrong in **25.9%–27.0%** on a divisor-independent draw
> (`derivation/divide.py:mechanism_sentence`), and the band widens to
> 26.0%–27.2% over a named block of divisor streams
> (`derivation/divide.py:sampling_interval`).

The 12.4%–15.4% band is the `a/a` rate and must be labelled as such wherever it
survives.

**Why the two sets of numbers differ, and which one is citable.** This note's
§2.1 sweep reports 22.7%–27.1% because it takes the **union over six divisor
laws**, one of which (`1/|N(0,1)|`, heavy-tailed, small divisors common) lands at
22.7%. A union over laws answers "what is the worst case across laws I chose";
the published sentence answers "what is the rate on a declared law". Both are
legitimate; they are different questions, and this note conflated them by
publishing the union as a scalar. `derivation/divide.py` keeps them as two
separate claims, each with its own divisor law attached, which is the structural
fix for the confusion.

**The third decimal is not a fact.** The published counts (13.988 / 15.312 /
15.170 for the self row) are not recoverable from the published figure alone: an
exhaustive search over 20,000 candidate `a`-streams and four near-1.0 laws, plus
four stream topologies, does not land on them. So the published draw is
**declared** in `_fp32.published()` and its own counts are what
`derivation/divide.py:published_rate_table` reports. Any document needing a count
must name the draw; `sampling_interval` quantifies why — at n = 10⁶ a rate
carries a ±0.044pp binomial 1σ, and moving the divisor stream alone moves a cell
by up to 0.17pp.

**The ratios in this section survive, and they are the more robust claim.** The
self row to general row ratio is 1.77x–1.86x here, against the 1.48x–2.13x
sweep and the "about 2x" in the file's own title. The ratio is stable across
draws and across the law sweep; the individual rates are not, at the third
decimal. **That is why the correction says "roughly 2x" and not "2.02x"**: the
band is the claim, and it is wide.

The table below is the **catalogue of what each site said**; the "what it says now"
column has been updated in this pass. Line numbers are pre-edit positions and are
not to be trusted as pointers — see the warning above this table.

| file | line(s) | what it said | what it says now |
|---|---|---|---|
| `probes/fp32_correctness.py` | **56–57** | `b = regimes[name] if op in "*/" else a` | **fixed.** An independent divisor, same marginal law, disjoint stream, drawn from a named constant; the law is printed with the numbers |
| `probes/fp32_correctness.py` | 70–72 | hardcoded `"12.4% - 15.4% of cases"`, `"0 of 6,291,456 ops wrong"` | **deleted.** The block is gone; everything the probe prints it computes. `+ - *` total is 18,874,368 |
| `probes/fp32_correctness.py` | **27, 67–68** | `"TinyGrad on Metal"`, `"a precision bug in the Metal backend"` | **fixed** to the Tensor-API mechanism with no device name; the device is printed |
| `probes/fp32_correctness.py` | **77** | dependent divisor `|a|+0.01` presented alongside the rate | **kept, but as its own labelled regime** and with the caveat that it is *dependent* and leaves the independent-law band |
| `probes/oc/tiny_oc.py` | **186–191** | `"the one float32 operation TinyGrad's Metal backend gets wrong: 1 ULP off in 12-15% ..."` | **fixed**: Tensor-API mechanism, no device name, the rate cited to `mechanism_sentence`. The correction *strengthens* the stated justification for keeping the division |
| `probes/oc/oc_report.py` | 98 | `"root cause: TinyGrad's Metal backend contracts a*b+c into fma"` | **fixed** twice over: `Device.DEFAULT`, and the FMA claim is marked as **not** the root cause (the sin LUT quantiser is) |
| `notes/01-obs-camera.md` | **59–61, 172–174** | `"**TinyGrad's Metal float32 division is not correctly rounded** — wrong in 12–15% of 2 M samples"` | **fixed** in both places: rate cited to the derivation, mechanism named correctly, no device name |
| `README.md` | **23, 80–83** | `"wrong in 12–15% of cases"`; `"(a) a correctly-rounded divide on Metal ... so it is a backend bug"` | **fixed** in both, and **"it is a backend bug" is gone** — it is not (§6.3) |
| `notes/02-obs-camera-residuals.md` | **72** | `\| \`/\` \| **260,261 - 322,233 (12.4% - 15.4%)** \| 0 \|` | **fixed**: both rows, the `a/a` one labelled, the `a/b` one measured, and 18,874,368 for the `+ - *` denominator |
| `notes/02-obs-camera-residuals.md` | **12** | 27.13% on PYTHON/CL/METAL, unlabelled as to divisor | **fixed**: scoped to a law, a host and an `n` |
| `notes/02-obs-camera-residuals.md` | **198–200** | `CL \| ... \| approximated`, `METAL \| ... \| approximated` | **fixed**: all four failing backends are **double-rounded**, `a*fl32(1/b)`, and CL's row is host-scoped |
| `notes/02-obs-camera-residuals.md` | **253–345** | the whole `[PREDICTED]` / "UNTESTED" tree, ending in "plus an approximation on Metal and CL" | **rewritten as REFUTED** against the T4 log, with the reasoning error named: a flag cannot undo a rewrite already in the source |
| `notes/02-obs-camera-residuals.md` | **387–391** | `"1 - 2^-24 is the signature of a reciprocal approximation ... the approximation itself is legitimate"` | **fixed**: a **correctly rounded** reciprocal, and the loss is the second rounding, not an approximation |
| `notes/04-codegen-comparison.md` | **735** | `\| \`/\` \| **260,261 – 322,233 (12.410% – 15.365%)** \| 0 \|` | **fixed**: both rows labelled and measured |
| `notes/04-codegen-comparison.md` | **740–741** | `"1 - 2^-24 is the signature of an approximate reciprocal"` | **fixed**: first sentence stands, second retracted with the `corner_predicate` argument |
| `notes/04-codegen-comparison.md` | **815–832** | `"PYTHON's 27.13% is double rounding ... while CL and METAL's is approximation"`; `"MSL contracts 1/x ... Two transformations"` | **fixed**: one mechanism, three renderings; the "second transformation" retracted |
| `notes/04-codegen-comparison.md` | **881–891** | `"One ULP is legitimate. The silence is the defect."` framed on an approximation | **fixed**: 1 ULP is a normal price, but there is no approximation to legitimise — the defect is an unknobbed second rounding |
| `notes/04-codegen-comparison.md` | **1020** | index row `"12.4–15.4% for /"` | **fixed**: both rows, and 18,874,368 |
| `notes/08-baseline-integrity.md` | **517** | `"wrong in 12–15 % of 2M samples on Metal, and that it is a backend bug"` | **fixed**: rate cited, mechanism named, "backend bug" gone |
| `notes/10-t4-results.md` | **215–216, 225–227** | general `a/b` as **25.020% / 28.127% / 27.973%**, no law stated | **fixed**: the rate is cited to the derivation with its law; the unstated-law scalars are withdrawn |
| `notes/10-t4-results.md` | 188–213, 574–578 | already correct: identifies the 12.4–15.4% band as `a/a` | no change. The internal precedent for §1 |
| `notes/10-t4-results.md` | 39–46, 576–577 | already correct: CL 29.76%, not host-independent | no change. Precedent for §5 |
| `notes/10-t4-results.md` | **342–363** | three stale hardcoded constants in `kda/replay_kda.py` / `perf_kda.py` | **fixed**: all three now measured or printed from the running device |
| `notes/11-local-suite-integrity.md` | **246–248** | `25.046% / 27.042% / 27.016%` and `2.02x / 1.76x / 1.78x` | **kept, with the divisor law named** — `b ~ U[0.01, 2.01]` off the continued `default_rng(0)` stream, which makes it exactly reproducible. The scalars are labelled as one law's worth |
| `notes/11-local-suite-integrity.md` | 208–219 | quotes the hardcoded `print` as a symptom | **fixed**: quotes it in the past tense, and draws the generalisation — a hardcoded print is how a number reaches 58 places |
| `notes/DECISIONS.md` | **19** | `` `/` wrong in 12.4–15.4% there and 27.1% in probes/divide_precision.py `` | **fixed**: both rows, the law, and the `+ - *` denominator |
| `notes/DECISIONS.md` | **25** | `"plus an approximation on Metal and CL"`; `"4 of 5 magnitude regimes"` | **fixed**: the approximation clause withdrawn, the "4 of 5" retracted as unreproducible, and the prediction marked **confirmed** with the T4 measurement |
| `notes/06-corrections-ledger.md` | **212–214** | P-1 / P-1a / P-1b, the "approximation" prediction family | **fixed**: P-1 REFUTED, P-1a CONFIRMED, P-1b not triggered, each with the log cited |
| `notes/06-corrections-ledger.md` | **44 (C-08)** | "CL and METAL are **approximated**" | **fixed**: the second half withdrawn; all four are double-rounded |
| `notes/06-corrections-ledger.md` | 275–281 | "A measurement on one backend is not a property of the compiler" | no change, and **extended in new rows C-09…C-13** plus §5 on the error class |
| `kernels/README.md` | **200** | `"bit-for-bit a*rcp.approx(b)"` | **fixed**: `a*fl32(1/b)`, i.e. a correctly rounded reciprocal, with the law attached |
| `bench/HYPOTHESES.md` | **390–428** | H9's `MODELLED` "expected failures = 0.02, 95% upper bound ≈ 3" | **rewritten as RETRACTED**: the units error, its factor (2.15e6), the measured outcome, and the corrected claim |
| `bench/FAIR-BASELINE.md` | **412** | `"Ops.FDIV ... is dead code on this path"` | **fixed**: it is the load-bearing last line of the CPU repair |
| `probes/divide_precision.py` | **13–17, 21–24, 33–40** | docstring: `"an approximate reciprocal"`, the per-backend table, `"wrong in ~27% of cases"` | **fixed**: the retraction is stated in the docstring, the per-backend table is host-scoped, and the rate is cited to the derivation |
| `probes/divide_precision.py` | **120–121** | `"The backend then contracts the 1/b into an approximate reciprocal."` | **fixed**: false on every backend measured (§4.1) |
| `probes/divide_precision.py` | **132–135** | `"But / does not go through it; the cstyle renderer does the a*(1/b) rewrite"` | **fixed**: backwards — `/` always goes through `Ops.RECIPROCAL`, the rewrite is the Tensor API's, and the no-knob argument is now the flag surface |
| `probes/divide_precision.py` | 96–98 | `same_as_rcp = int((got != rcp_mul).sum()) == 0` | correct, and the right test; only the printed *label* it fed was wrong. `rcp_mul` there is `(a * (1.0/b)).astype(np.float32)` — a correctly rounded reciprocal |
| `kda/perf_kda.py` | 143 | `f"\nmeasured (Metal, ...)"` | **fixed** to `Device.DEFAULT` |
| `kda/replay_kda.py` | 97, 101–102, 105 | hardcoded `~109 ms`, `0.109`, `"COLD is 49.5 s"` against a measured 96,837 ms | **fixed**: the GPU term is measured in-process; COLD is read from `t_cold`; and with no GPU instrumentation the script prints **no decomposition** rather than subtracting a number it does not have |
| `probes/oc/report_fix/fp32_div_row.py` | 13–15, 63, 67 | `"disagreeing by ~2x"`; unstated divisor law; per-regime ratio | **fixed**: the law is stated in the docstring, the ratio is labelled as one law's worth, and the degenerate-vs-general distinction is explained |

### 6.2 What is NOT wrong

- `probes/fp32_correctness.py:27-37`'s `exact_f32` reference: correct. A float32 quotient
  needs ≤ 48 bits and float64 has 53, so the single rounding down is correctly rounded.
  Re-verified against every figure in this note.
- `notes/02`'s "there is no division-free exact reciprocal in float32" (§7, and the
  `tiny_oc.py:193-199` reasoning): unaffected.
- `notes/10` §1.3, §1.4, §1.5, §1.1: correct, and this note's §2, §4 and §5 are the same
  findings reached independently. It is the file that found the `a/a` bug.
- The shipped logs in `out/kaggle/20260928-014248-gpu/logs/`: correct **as measurements**.
  They faithfully report what the probe computed. The bug is in the probe, not the log.
  `fp32_correctness.log` lines 10/17/24 are `a/a`; `divide_precision.log` line 16 is `a/b`.

### 6.3 The attribution, restated once

The division defect is **not a backend bug** and this is now the best-evidenced claim in
the file, not the weakest:

- `/` is lowered at `mixin/elementwise.py:251` to `a.alu(Ops.MUL, b.reciprocal())`, above
  the scheduler and above backend selection, so no `Ops.FDIV` exists in the IR for any
  renderer to render.
- `ClangRenderer` (`renderer/cstyle.py:271-274`) is the **only** renderer among those
  measured that carries `Ops.FDIV` — it deletes `RECIPROCAL` from `code_for_op` and adds
  `Ops.FDIV: (a/b)`. That is what lets the CPU backend's decomposition
  (`codegen/decomp/op.py:133-136`, gated on `if Ops.FDIV in ops`) manufacture the missing
  `FDIV` and emit a real `/`.
- `CPU` therefore measures 0.00% and every other backend does not. `MEASURED`, all three
  regimes, both ops, 0 of 6,291,456.

Every file in §6.1 that says "backend bug" or "the Metal backend" is mis-filing the defect
one level too low, and the corrected upstream report is: **tinygrad's `Tensor` API lowers
`/` to `a · fl32(1/b)`, which costs one extra rounding on every backend; the CPU backend
reconstructs a real division through its `Ops.FDIV` renderer and decomposition path and is
correctly rounded; nothing else is, and there is no knob.**

---

## 7. Provenance and what could not be done here

- **`nvidia-cuda-nvcc-cu12` is available for offline compilation.** PyPI serves
  `py3-none-manylinux2010_x86_64`, `manylinux2014_aarch64` and `win_amd64` wheels (latest
  12.9.86), so a CUDA backend could be built in a container without a system CUDA
  toolkit. Verified from the PyPI index. (The local `pip download` of the sdist fails to
  *build* on this host, which is a packaging artefact of the interpreter, not an absence
  of the wheel.)
- **A containerised CUDA backend is possible.** Nothing in this note needed one.
- **There is no GPU on this host.** `nvidia-smi` is not installed; tinygrad's `CUDA`
  device raises `AttributeError` on first use. So:
  - **Every `MEASURED` figure in this note is from Metal or CPU on an Apple Silicon Mac.**
  - **The CUDA figures are read from `out/kaggle/20260928-014248-gpu/logs/`, tagged
    `RECORD`, and were not regenerated.** They are cross-checked three ways: the
    `divide_precision` counts (54,270 / 30,748) are reproduced exactly by the model; the
    `fp32_correctness` counts (260,261 / 322,233 / 318,044) are reproduced exactly by the
    model; and the model's output array is the *same array*, not merely the same size,
    which the bit-identity test in §4 establishes. A T4 re-run of
    `probes/fp32_correctness_fixed.py` on `Device.DEFAULT` = CUDA would close the loop
    with a single command and needs nothing but a GPU.

---

## 8. What survives, and what is withdrawn

### Survives — unchanged, and now better evidenced

1. **`/` on the cstyle backends is genuinely not correctly rounded.** 0.000% wrong is
   0.00% wrong and 22.7–27.1% wrong is 22.7–27.1% wrong; both are non-zero. `MEASURED`.
2. **The error is genuinely 1 ULP and no more.** General `a/b`: `max |ULP| = 1`, max
   relative error `1.1921e-07 = 2⁻²³`, and the histogram is only `−1` and `+1` in every
   regime (`{−1: 276,821, +1: 279,285}`, `{−1: 284,105, +1: 283,206}`,
   `{−1: 281,487, +1: 281,775}`). `MEASURED`. There is no regime in this note where it is
   worse.
3. **The error is genuinely silent and unknobable.** No flag, no device selection, no
   dtype choice changes it. `Ops.RECIPROCAL` exists and is explicitly approximate on the
   PTX path, but `/` never gets a user-facing choice between it and a real divide.
4. **`+`, `-`, `*` are correctly rounded.** 0 wrong, independent operands, every regime,
   both devices. `MEASURED`.
5. **The defect is not a backend bug** (§6.3). CPU is 0.00% because it is the only
   renderer that reconstructs a real `/`, and that is the strongest single fact in the
   file.
6. **The 1-ULP error is load-bearing for netherite's DDA**, which takes every step size as
   `1.0f/dx`. Unchanged, and `probes/oc/tiny_oc.py`'s decision to keep the division rather
   than fake it is *more* justified after this correction, not less: the rate is higher
   than the comment states.
7. **The self-division rate is host-independent** and reproduces bit-for-bit across an
   NVIDIA T4 and an Apple GPU (§2). Useful precisely because it is the degenerate quantity.

### Withdrawn

1. **"TinyGrad's float32 `/` is wrong in 12.4–15.4% of cases."** Withdrawn. That is
   `a/a`. The general rate is **25.9%–27.0%** on a divisor-independent draw
   (`derivation/divide.py:mechanism_sentence`), and 22.7%–27.1% over a sweep of six
   divisor laws. Note the two are different questions: the band is a rate, the sweep
   is a worst case across laws I chose. The earlier version of this note published
   the sweep's union as if it were the rate; that is withdrawn.
2. **"Wrong in 27.13%."** Withdrawn as a scalar. It is one point in a band at one
   `(a, b)` law and N = 200,000. `notes/10` §1.4 already said this; the number kept
   propagating.
3. **"The rate is insensitive to whether the reciprocal is exact."** **Retracted.** A
   1-ULP-worse reciprocal predicts **86.5%** against a measured **27.1%**
   (`derivation/divide.py:reciprocal_sensitivity`). The reciprocal's accuracy is
   load-bearing; the *multiply's* accuracy is not, because it is correctly rounded
   and contributes zero failures. The earlier figures here (86.54% against 27.05%)
   were the same two numbers at a different `n`; the comparison is what carries.
4. **"The backend contracts `1/b` into an approximate reciprocal."** Withdrawn as a
   statement about any backend measured here. What fires is a **correctly rounded**
   reciprocal, and the outputs are bit-identical to `a · fl32(1/b)` on CUDA, PYTHON, CL
   and METAL. This was a label printed by `probes/divide_precision.py:120-121` without
   being tested, and it is the load-bearing error: it is what produced the wrong
   prediction P-1 and the wrong error class in `notes/02`, `notes/04` and `notes/06`.
5. **"1 − 2⁻²⁴ is the signature of a reciprocal approximation."** Withdrawn. It is the
   signature of a correctly rounded reciprocal landing between `−2⁻²⁵` and `−2⁻²⁴`, in a
   one-sided test whose upward half is unreachable (§3.3).
6. **"CL is host-independent" / "four backends at 27.13%."** Withdrawn. CL is 27.13% on
   this Mac and **29.76%** on the T4, and only the first is bit-identical to
   `a · fl32(1/b)` (§5).
7. **"27% wrong" as a codegen property.** Withdrawn, and this is the *second* time in
   this file that it has had to be withdrawn — `notes/06` C-04 withdrew it once (Apple
   hardware, described as a compiler property) and it came back as a scalar rate. A rate
   over a distribution is not a property of the thing that produced it. The scope must
   travel with the number: *which backend, which host, which divisor law, which N.*
8. **"The second rounding contributes failures" / "double rounding costs two roundings."**
   Withdrawn as stated. The multiply is correctly rounded; **all** failures are the
   rounding of `1/b`. Calling it "double rounding" is loose and caused at least one wrong
   error-class label in the record.

---

## 9. Instrument

**Two instruments now, and the division of labour between them is the point.**

`probes/fp32_correctness.py` — the **probe**, fixed in place. Measures on a
device, in this process, on whatever device ran; prints the device; prints the
**degenerate `b := a` row as a cross-check** against the shipped T4 log, and the
**independent-divisor row** as the rate. The cross-check exists so the probe's own
`a` array cannot silently drift: if the degenerate row ever stops equalling
260,261 / 322,233 / 318,044, every other number in it is void.

`derivation/divide.py` — the **derivation**, device-free, and the only place a
document may get a number from. `a * fl32(1/b)` is arithmetic, so the rate is a
property of the arithmetic and not of any backend; the per-backend, per-host
numbers live in `derivation/divide_backends.py` and are never to be quoted as one
thing with the former. Claims a document cites:

| claim | what it prints |
|---|---|
| `mechanism_sentence` | the one sentence, with its band pinned to the authoritative record |
| `authoritative_reproduction` | whether **this** machine reproduces those six figures, per cell, in sigmas |
| `published_rate_table` | the six cells on a declared draw, with counts and the ratio |
| `sampling_interval` | the binomial interval and the empirical divisor-stream spread |
| `error_rate_independent` | the six-law sweep, labelled as a sweep |
| `error_rate_self_division` | the degenerate rate, explicitly not a rate for `/` |
| `corner_predicate` | the exact `a/a` predicate and its single wrong bit pattern |
| `representable_controls` | the `2*a` / `0.5*a` / `-a` control, ratio 1.00x |
| `dependent_law_outside_band` | why a dependent divisor law leaves the band |
| `ulp_bound` | 1 ULP, with the ULP convention stated |
| `reciprocal_sensitivity` | ~86% predicted vs ~27% measured |
| `cell_failure_fraction` | the `bench/HYPOTHESES.md` H9 slip, re-derived |
| `probe_denominators` | 6,291,456 vs 18,874,368 |
| `mechanism_rewrite`, `mechanism_no_knob` | the source pins and the flag surface |

`probes/fp32_correctness_fixed.py`, the standalone instrument this note
originally shipped, is **superseded** by those two and is retained only as the
worked example of the section structure. Its six sections, every line tagged
`MEASURED`, `RE-DERIVED`, `MODELLED` or `RECORD`:

1. `/` with an independent divisor and `/` with `b := a`, side by side, per device.
2. The understatement ratio, and its band.
3. The cross-host check: the `a/a` row against the shipped T4 log.
4. Error magnitude: ULP histogram, max, max relative error.
5. Mechanism: the exact failure predicate, the cell position `r`, the closed form
   `a/a wrong ⟺ δ < −2⁻²⁵`, the wrong-value enumeration, and the reciprocal-accuracy
   sensitivity sweep.
6. Divisor-law sweep, with the `2*a` correlated control.

```
python probes/fp32_correctness_fixed.py                  # CPU + Device.DEFAULT
python probes/fp32_correctness_fixed.py CPU CL           # named devices
python probes/fp32_correctness_fixed.py --n 262144       # smaller sample
python probes/fp32_correctness_fixed.py CUDA             # on a GPU box
```

Runs on CPU and on Metal on this host, verified; a device that is listed but unusable is
reported as `[SKIP]` and dropped rather than crashing the run. On a CUDA box
`Device.DEFAULT` is `CUDA` and the same command produces the missing T4 column.
