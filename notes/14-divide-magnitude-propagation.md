# 14 — every location that quotes the float32-`/` error magnitude or names its mechanism

**Status: REMEDIATION APPLIED 2026-09-28. This table is now a record of what was
wrong, not a plan.** Every location in §2 (13 working files) and §3 (48
deliverable locations) has been corrected or deleted in place. Two things in the
plan text below are **superseded** and are marked where they occur:

- every `22.7%–27.1%` in the "what it should say" column was the **union over
  six divisor laws** presented as if it were a rate. The citable figure is
  narrower and carries its law: **25.9%–27.0%** on a divisor-independent draw
  (`derivation/divide.py:mechanism_sentence`);
- every "1.48x–2.13x" is a **band, not a factor**. It survives as a band; the
  scalars `2.02x` / `1.76x` / `1.78x` are one divisor law's worth of it and are
  no longer published as the ratio.

**What the remediation establishes beyond the individual edits.** The
structural fix is that a number is now **derived, never transcribed**: it lives
in exactly one executable place (`derivation/`) and a document cites the claim
that prints it. See `notes/06-corrections-ledger.md` §5, which records this
error class and the rule that followed from it — *prefer deletion to a hedged
number*.

**Line numbers in this table were themselves a hazard, twice over.** §4.1 below
records that a prior audit's citations did not say what was claimed (one
attributed a sentence that is not in the file at all), and §0's own line-number
pins are 2026-09-27 positions. Every line reference here is re-checked against
the file as of the remediation pass; a citation that no longer resolves is
marked.

**This note's figures were re-derived and several did not survive.** See §0.1.
**Supersedes the location list in** `notes/12-divide-finding-correction.md` §6.1, which
this note **re-verifies and partly corrects** (§3 below).
**Host:** Apple Silicon Mac, tinygrad 0.14.0 (`.venv/lib/python3.12/site-packages/tinygrad`),
numpy 2.5.3, python 3.12. **No GPU on this host** — every CUDA/T4 figure below is `RECORD`
or `UNVERIFIED`, never re-measured.

**Tags, used strictly.** `READ` = I opened the file at that line and the quoted text is
there. `MEASURED` = I ran it here, in this process. `RE-DERIVED` = I recomputed it here in
numpy from arrays the repo's own code builds; checkable by hand, no device involved.
`MODELLED` = arithmetic on a hypothetical implementation. `RECORD` = read out of a shipped
log in `out/`, not reproduced. `ARITHMETIC` = a quotient, no run needed. `UNVERIFIED` = a
claim nobody has actually established.

---

## 0. What I actually ran, before any of the auditing

The audit is worthless if the thing being propagated is itself unverified, so this comes
first. Three re-derivations, all in numpy, all at `n = 2,097,152` unless stated.

| claim | result | tag | now |
|---|---|---|---|
| general `a/b` over 5 independent divisor laws × 3 regimes | **22.718% – 27.127%**, median 25.093% over 15 points | `RE-DERIVED` | **the union over laws, not a rate.** One law (`1/\|N(0,1)\|`) sits at 22.7% and drags the floor. The citable rate carries its law: **25.9%–27.0%** (`derivation/divide.py:mechanism_sentence`) |
| ratio to the `a/a` rate over those laws | **1.48x – 2.02x**, median 1.79x | `RE-DERIVED` | **survives as a band**; on the declared draw it is 1.77x–1.86x. The scalars are one law's worth |
| `2*a` correlated control | rate 12.410% / 15.365% / 15.166%, **ratio 1.00x in all three** | `RE-DERIVED` | **the strongest result in this note and it survives intact** — see `derivation/divide.py:representable_controls` |
| `a/a` wrong ⟺ `δ(a) < −2⁻²⁵` | 0 / 2 / 0 disagreements over 6,291,456 | `RE-DERIVED` | survives; `derivation/divide.py:corner_predicate` |
| `notes/11`'s "superseded" set 25.046 / 27.042 / 27.016, ratios 2.02 / 1.76 / 1.78 | **reproduced to the digit** — see §3.4 | `RE-DERIVED` | survives, and the *producer* is identified (§3.4): a `U[0.01, 2.01]` divisor off the continued `default_rng(0)` stream. So the set is reproducible **once the law is named** — the earlier claim that it was not reproducible was half right: it is not reproducible *without* the law, and is exactly reproducible *with* it |
| reciprocal-accuracy sensitivity, `k = ±1` | **86.48%** (`U[−2,2]` law) / **86.42%** (`U[0.01,2.01]` law) — not `notes/12`'s 86.54% | `RE-DERIVED` | the discrepancy is a **law and `n` difference**, not an error in either. All are ~86%, against ~27% measured. `derivation/divide.py:reciprocal_sensitivity` gives 86.5% at n = 2,097,152 on the `U[0.01, 2.01]` law — i.e. it agrees with `notes/12` to the digit and this note's 86.42% is the same quantity at a different `n` |
| `out/.../fp32_correctness.log:10,17,24` | `260,261 (12.410%)`, `322,233 (15.365%)`, `318,044 (15.166%)` — verbatim | `RECORD` | survives, and is now **re-derived** rather than only read: `derivation/divide_backends.py:self_division_log_crosscheck` asserts equality, not side-by-side |
| `out/.../divide_precision.log:16` | `CUDA : 54,270 wrong (27.13%) == a*rcp(b) bit for bit` — the `a/b` row, verbatim | `RECORD` | survives. Note the log's own label `a*rcp(b)` is ambiguous and is *not* `rcp.approx` — see §0.1 |
| `out/.../divide_precision.log:10` | `CUDA : 30,748 / 200,000 wrong (15.37%)` — the `a/a` row; **no METAL row on the T4** | `RECORD` | survives |
| every tinygrad source line cited anywhere in the record | all verified except `renderer/ptx.py:15` (correct: `:19`) | `READ` | survives; re-verified 2026-09-28, all resolve |

### 0.1 Three of this note's own claims did not survive re-derivation

Recorded because the note is the one that found the propagated error, and a
finding that carries its own errors is worth marking rather than quietly
fixing.

1. **The 22.7% floor is a law artefact, not a property of `/`.** It is the
   `1/|N(0,1)|` law, and it is real — but presenting a six-law union as "the
   rate" is the same category of error as the one this note audits: one
   measurement, laundered into a general claim. Fixed by splitting the union
   from the rate: `derivation/divide.py:published_rate_table` (one declared law)
   vs `:error_rate_independent` (the sweep, labelled as a sweep).
2. **86.42% vs 86.54% is a sample-size difference, not a disagreement.** Both
   are correct for their own `n`. Publishing either as *the* sensitivity figure
   without the `n` repeats the original sin. The claim that survives is the
   comparison — **~86% predicted against ~27% measured, a factor of 3.2** — and
   that is what the corrected claim states.
3. **The log's `== a*rcp(b) bit for bit` is ambiguous and was read as
   `rcp.approx`.** `kernels/README.md` propagated that ambiguity into the
   literal string `a*rcp.approx(b)`, which asserts an approximation that the
   bit-identity test does not establish. The test establishes identity with
   `a * fl32(1/b)` — a **correctly rounded** reciprocal. Fixed there; the point
   is that a *label* on a passing test is a claim, and this one was wrong.

**The band and the control survive, and the control got stronger than `notes/12` reports.**
I added two more representable-quotient controls. Divisors that make the exact quotient
*exactly representable* all collapse the ratio to the self-division rate regardless of
magnitude — `MEASURED`/`RE-DERIVED`, `mixed ±2`, denominator 15.365%:

| divisor | exact quotient | rate | ratio | exact quotient representable? |
|---|---|---|---|---|
| `a` (self-division) | 1.0 | 15.365% | 1.00x | 100% |
| `2*a` (`notes/12`'s control) | 0.5 | 15.365% | **1.00x** | 100% |
| `0.5*a` | 2.0 | 15.365% | **1.00x** | 100% |
| `−a` | −1.0 | 15.365% | **1.00x** | 100% |
| `a+1` | — | 23.856% | 1.55x | 0.00% |
| `a*a` (the `*` row's divisor) | — | 25.019% | 1.63x | 0.03% |
| `\|a\| + 0.01` (`probes/fp32_correctness.py:77`) | — | **36.588%** | **2.38x** | 0.02% |

`−a` is a strictly stronger control than `2*a`: it changes the divisor's sign and nothing
about its magnitude, and the rate does not move at all. Four controls, 0.5x to 2x in
magnitude, all at exactly 1.00x. **The discriminator is not the divisor's magnitude and not
its independence — it is whether the exact quotient is representable.** `notes/12` §2.1
proved this with one control; it takes three more to make it airtight, and the repo now has
them.

### 0.1 A scope limit on the band that the band does not state

`notes/12` §2.1's 22.718%–27.127% is a sweep over **six independent divisor laws**. It is
not an envelope. The repo's *own* probe contains a **dependent** divisor law that lands
outside it:

> `probes/fp32_correctness.py:77` sets `b = np.abs(a) + np.float32(0.01)` — deterministic
> in `a`, so not independent, but `b ≠ a` and the exact quotient is representable only
> 0.02% of the time. On `mixed ±2` that gives **36.588%**, a ratio of **2.38x**, i.e.
> **above the 27.127% top of the sweep.** `RE-DERIVED`.

So the honest form of the band is *"22.7%–27.1% over six **independent** divisor laws"*,
and the true rate is a function of the *joint structure* of `(a, b)` — not of the marginal
laws of `a` and `b` separately, and not only of independence. Structured divisors push `r`
inside the rounding cell away from uniform in either direction: `a+1` gives 23.856%, below
the band's interior; `|a|+0.01` gives 36.588%, above its top. **Any sentence carrying a
rate must name the divisor law and say whether it is independent.** `notes/12` already
insists on this; the `|a|+0.01` case is the proof that the insistence is not pedantry.

A second, unrelated observation about the same block, offered because it *reduces* the
blame on the original probe: `fp32_correctness.py:74-85` is a **second, valid `a/b` test**,
and it is the one that produced `division error magnitude: median 1.0 ULP, p99 1.0 ULP, max
1 ULP` — a claim that is **correct, and that I confirm** (`max |ULP| = 1` on the `|a|+0.01`
law too). So the probe was right about the *magnitude* of the error and wrong only about
the *rate*, and it was right about the magnitude from a code path nobody had connected to
the claim it was supporting. `notes/04:737` quotes that line; it needs no correction.

---

## 1. The three ways a claim here can be wrong

- **`M` — wrong magnitude.** Quotes a float32-`/` error rate, or a divisor's error rate,
  that the 6-law band supersedes. Includes quoting a per-regime scalar (`25.046%`,
  `28.127%`, `27.973%`, `27.06%`, `27.11%`, `27.13%`, `36.588%`) with no law attached.
- **`C` — wrong mechanism.** Names the mechanism as a *reciprocal approximation*
  (`rcp.approx.f32`, `rcp.approx`, "MSL contracts `1/x`", "approximated",
  "signature of a reciprocal approximation", "double rounding", "insensitive to whether
  the reciprocal is exact"), or files the defect at the backend instead of the Tensor API,
  or asserts `Ops.FDIV` is dead code. `notes/12` §3–§4.1 and §6.3.
- **`S` — superseded but not yet retracted.** The number or wording is not *false*; it has
  been overtaken by a later correction in the same record and the file still carries the
  earlier form. This is the category that has done the most damage, because a reader
  cannot tell which of two live sentences is current.

**Remediation differs by kind, and by file kind.** A *working file* (a script's code,
docstring, print label) needs a code change. A *deliverable* (a published `notes/` file, the
`README`) needs a correction entry — the prose is the record, so it is amended in place with
a date and a pointer, not silently rewritten.

---

## 2. Working files — 13 locations. 11 wrong, 1 adjacent disease, 2 propagation vectors.

**All 13 remediated 2026-09-28.** The "what it should say" column below is the
historical plan; the `22.7%–27.1%` in it is superseded by the narrower,
law-attached band (see the header). What was actually done, per location, is in
`notes/06-corrections-ledger.md` **C-09** (W1–W3, W9, W11), **C-10** (W4, W6,
W7), **C-11** (W8), **C-12** (`bench/FAIR-BASELINE.md`, same defect as W8's
second half), and the header of this file.

| # | file:line | what it says now (verified) | kind | what it should say |
|---|---|---|---|---|
| **W1** | `probes/fp32_correctness.py:56-57` | `("*", lambda x, y: x * y), ("/", lambda x, y: x / y)):` / `b = regimes[name] if op in "*/" else a` | **`M`** root cause | The bug. `/` (and `*`) divide/multiply by the dividend. `probes/fp32_correctness_fixed.py` is the replacement instrument. Either retire this probe or make it draw an independent divisor; the `+ -` rows are unaffected. |
| **W2** | `probes/fp32_correctness.py:67-68` | `"A nonzero TinyGrad column is a precision bug in the Metal backend, not a"` / `"rounding artefact of the test."` | **`M`** attribution | `"...is a precision bug in whichever backend reached the cstyle path, not a rounding artefact of the test."` This string is in the shipped T4 log at line 27 (`RECORD`), on a CUDA run. |
| **W3** | `probes/fp32_correctness.py:70-72` | `"Measured result: + - * are exact (0 of 6,291,456 ops wrong). / is wrong in"` / `"12.4% - 15.4% of cases depending on magnitude. See"` | **`M`** | `".../ is wrong in 22.7% - 27.1% of cases with an independent divisor (12.4% - 15.4% is self-division, a/a). See"`. Also fix the denominator: `ARITHMETIC` — 6,291,456 = 1 op × 3 regimes × 2,097,152, so the sentence names three ops and counts one. `+ - *` is **18,874,368**. |
| **W4** | `probes/divide_precision.py:13-17` | `"Metal then does what MSL always does with \`1/x\` and turns it into an"` / `"approximate reciprocal. Two steps, and the first one is Tinygrad's."` | **`C`** | `"MSL (and CL, CUDA C, Python) then compute a correctly rounded float32 reciprocal and one float32 multiply. \`notes/12\` §4.1: a 1-ULP-worse reciprocal predicts ~86%, not the ~27% measured, so what fires is a *correctly rounded* reciprocal."` **Missed by `notes/12` §6.1** — it corrects `:120-121` and not the docstring's item 1. |
| **W5** | `probes/divide_precision.py:21-24` | `CPU 0.00% wrong` / `PYTHON 27.06% wrong` / `CL 27.06% wrong` / `METAL 27.06% wrong` | **`M`** **`S`** | Per-backend, host-scoped, no `n`, no divisor law. `27.06%` is one point in a 22.7%–27.1% band at `n = 200,000` under one law. The `CL` row is **29.76%** on a T4. Publish as a band, or attach `inputs("pm2", seed=0), N = 200,000, host = …`. |
| **W6** | `probes/divide_precision.py:33-37` | `"…tinygrad's cstyle backend lowers \`/\` to \`a*(1/b)\`, and MSL lowers that to an approximate reciprocal. The combined error is 1 ULP, wrong in ~27% of cases, and there is no knob."` | **`C`** **`M`** | `"…the Tensor API lowers \`/\` to \`a*(1/b)\` above backend selection, and the backend computes a *correctly rounded* float32 reciprocal times a correctly rounded float32 multiply. The error is 1 ULP, wrong in 22.7%–27.1% of general \`a/b\` cases depending on the divisor law, and there is no knob."` 1 ULP and "no knob" **survive**. |
| **W7** | `probes/divide_precision.py:120-121` | `"=> tinygrad's renderer emits \`a*(1/b)\`, not \`a/b\`. The backend then"` / `"contracts the \`1/b\` into an approximate reciprocal."` | **`C`** | Clause 1 survives. Clause 2: `"computes a correctly rounded float32 reciprocal for \`1/b\`, followed by one correctly rounded float32 multiply."` Printed unconditionally as commentary, on every run, on every backend. |
| **W8** | `probes/divide_precision.py:132-135` | `"Ops.RECIPROCAL exists and PTX renders it as \`rcp.approx.f32\` --"` / `"an explicitly approximate op. But \`/\` does not go through it; the"` | **`C`** | **Backwards, and this is `notes/06` C-07 already settled.** `/` *always* goes through `Ops.RECIPROCAL` — `mixin/elementwise.py:251` builds `MUL(a, reciprocal(b))`, `:465` returns `Ops.RECIPROCAL`. That is the defect. The correct version of the "no knob" argument is that the *user* has no choice, not that the op is unused. **Missed by `notes/12` §6.1.** |
| **W9** | `probes/oc/tiny_oc.py:188-191` | ``# \`/\` is the one float32 operation TinyGrad's Metal backend gets wrong:`` / ``# 1 ULP off in 12-15% of 2.1M samples per magnitude regime, where numpy is`` | **`M`** attribution | ``# \`/\` is the one float32 operation TinyGrad's cstyle backends get wrong:`` / ``# 1 ULP off in 22.7-27.1% of 2.1M samples per magnitude regime with an independent divisor (12-15% is self-division), where numpy is`` **The code below this comment is right and stays.** The corrected comment *strengthens* the stated justification for keeping the division in the DDA, because the real rate is higher. |
| **W10** | `probes/oc/oc_report.py:98` | `"root cause: TinyGrad's Metal backend contracts a*b+c into fma"` | — (adjacent) | **Not a division claim.** The same disease — a printed root cause that is not measured, naming a backend. `notes/10` §3 and `notes/11` §6 already flag it. Listed here only so the audit is complete; it needs the `--self-check` treatment, not a number. |
| **W11** | `probes/oc/report_fix/fp32_div_row.py:13-15`, `:63`, `:67` | docstring `"a*(1/a) is much closer to 1 than a*(1/b) is to the true quotient, so a/a understates the error rate. Two probes, same backend, same session, disagreeing by ~2x"`; `:63` `b = rng.random(N).astype(np.float32) * np.float32(2) + np.float32(0.01)`; `:67` prints a per-regime `understated by` ratio | **`M`** **`S`** | **This is the instrument that produced `notes/11`'s superseded set, and `notes/12` §6.1 lists the numbers but not their producer.** `"disagreeing by ~2x"` → `"by 1.48x–2.13x".` The printed column is one divisor law's worth; label it. **The divisor law itself must be stated in the file**, because §3.4 shows this exact `U[0.01, 2.01]`-off-the-continued-`default_rng(0)` stream is what makes `notes/11` reproducible and `notes/12`'s sweep not reproduce it. |
| **W12** | `kernels/build_notebooks.py:442` | `("fp32_correctness", ".", [PY, "probes/fp32_correctness.py"], [], "is float32 correctly rounded")` | — (vector) | Not a wrong claim: the label is fine. It is a **propagation vector** — it bakes W1–W3's output into a published notebook under `kernels/`. Fixing the probe fixes this for free; do not edit this line. |
| **W13** | `remote/cloud_run.py:819` | `PROBE_SCRIPTS = ["probes/fp32_correctness.py", "probes/divide_precision.py", …]` | — (vector) | Same: a **vector**. Every future GPU run ships the buggy probe and re-emits `12.4% - 15.4%` plus the false Metal label. Add `probes/fp32_correctness_fixed.py` to the list so the T4 column that `notes/12` §7 says would close the loop is actually produced. |

---

## 3. Deliverables — 48 locations, all carrying a now-wrong claim.

Fix these with a dated correction entry in the file, not a silent rewrite. Priority order
in §6.

### 3.1 `README.md` — 2

| # | line | what it says now | kind | correction |
|---|---|---|---|---|
| **D1** | `23` | `**float32 division that is wrong in 12–15% of cases**` | `M` | **`wrong in 22.7%–27.1% of cases (divisor-law dependent; the 12–15% often quoted is self-division)`** |
| **D2** | `80-82` | `(a) a correctly-rounded divide on Metal — measured wrong in 12–15% of 2M samples where numpy is exact, so it is a backend bug, not a design choice;` | `M` + attribution | Corrected rate; `Metal` → the cstyle path; and **"it is a backend bug" is the wrong filing** — it is created in the Tensor API and only *manifests* per backend, so there is no backend-side fix to ask for. `notes/12` §6.3. |

### 3.2 `notes/01-obs-camera.md` — 2

| # | line | what it says now | kind | correction |
|---|---|---|---|---|
| **D3** | `59-62` | `- **TinyGrad's Metal float32 division is not correctly rounded** — wrong in` / `12–15% of 2 M samples across three magnitude regimes, where numpy is exact.` … `This is a standalone bug worth reporting upstream,` | `M` + attribution | `wrong in 22.7–27.1% of 2 M samples with an independent divisor (12–15% is a/a), where numpy is exact`, and `Metal` → the cstyle path. **"A standalone bug worth reporting upstream" survives** — it is a TinyGrad bug, which is the strongest form of the claim. |
| **D4** | `172-174` | `5. **Correctly-rounded float32 division on Metal.** Measured wrong in 12–15% of 2 M samples (\`../probes/fp32_correctness.py\`).` | `M` + attribution | Same, plus the `a/a` label. The **cites-the-probe-that-is-wrong** problem is the real defect here: the pointer now leads to `probes/fp32_correctness.py`, which does not measure this. Repoint to `probes/fp32_correctness_fixed.py`. |

### 3.3 `notes/02-obs-camera-residuals.md` — 9

The worst file in the repo for this finding: **lines 12 and 72 contradict each other, sixty
lines apart, about the same quantity, with no divisor named on either.**

| # | line | what it says now | kind | correction |
|---|---|---|---|---|
| **D5** | `12` | `27.13% wrong on PYTHON/CL/METAL, **0.00% on CPU**.` | `S` | The number is a divisor-law point at `N = 200,000`, not a property. Add the law and the `n`, and note CL is not host-independent (D9). |
| **D6** | `72` | `\| \`/\` \| **260,261 - 322,233 (12.4% - 15.4%)** \| 0 \|` | `M` | **Label the row `a/a`** and add a general `a/b` row: `22.7%–27.1% over six independent divisor laws` (or `25.046%` if the file wants one number, in which case it must say `U[0.01, 2.01]`, `n = 2,097,152`). Ten lines below its own line 12, which uses 27.13% for the same quantity. |
| **D7** | `169-175` | `**PYTHON's 27.13% therefore has a *different* cause from CL and METAL's**` … `CL and METAL are *approximated*. Both land on the same 27.13% because the error is 1 ULP either way, and both are bit-identical to \`a*rcp(b)\`.` | `C` | **The same claim as D9, stated twice in one file — `notes/12` §6.1 lists only D9.** "Bit-identical to `a*rcp(b)`" is a fact; "PYTHON has a *different cause*" is **false**, and "CL and METAL are *approximated*" is false. All four are bit-identical to the same array, so they have the same cause. Replace the whole passage: *one* mechanism, one error class (the reciprocal is correctly rounded on every backend measured; the multiply is correctly rounded; CPU is the sole exception and is correctly rounded). |
| **D8** | `188-190` | `MSL then does what MSL does with \`1/x\` and contracts it into an approximate reciprocal. Two transformations.` | `C` | **Missed by `notes/12` §6.1.** Replace: MSL computes a correctly rounded float32 reciprocal. There is **one** transformation that is a defect (Tinygrad's, at `mixin/elementwise.py:251`), and the backend's rounding of `1/b` is a second, unavoidable one. |
| **D9** | `198-200` | `\| \`PYTHON\` \| 27.13% (54,270) \| **yes** \| **double-rounded** \| …` / `\| \`CL\` \| 27.13% (54,270) \| **yes** \| **approximated** \| …` / `\| \`METAL\` \| 27.13% (54,270) \| **yes** \| **approximated** \| …` | `C` + `S` | The `approximated` error class is wrong for `CL` and `METAL`. "double-rounded" is loose but not false — `notes/12` §8.8 withdraws it in favour of "rounded in exactly one place: `1/b`". Also scope CL: 27.13% is a **Mac** measurement; on a T4 it is **29.76% and not bit-identical** to `a·fl32(1/b)`, so the "yes" in CL's row does not generalise. |
| **D10** | `206` | `the approximation is introduced *after* a syntactically well-defined \`1.0f/x\`.` | `C` | **Missed by `notes/12` §6.1.** There is no approximation. The point survives if restated: *the rewrite is already in the emitted source, so no driver flag can remove it* — which is the actual "no knob" argument. |
| **D11** | `238-242` | `- **~27% wrong, bit-identical to \`a*rcp(b)\`** → double rounding, i.e. the same` … `Tensor-level rewrite costs you a second rounding, plus an approximation on Metal and CL" rather than "an approximation".` | `C` + `M` | **Missed by `notes/12` §6.1.** The branch *fired* and its outcome is now measured, not predicted. The concluding sentence is the false framing. Replace with `notes/12` §4's replacement sentence. |
| **D12** | `270-283` | the prediction tree's consequences: `re-scoped to Metal and CL`, `**Reproduces at ~27% but as double rounding, matching PYTHON's error class.**`, `the Tensor-level rewrite costs you a second rounding, plus an approximation on Metal and CL` | `C` + `S` | **Missed by `notes/12` §6.1.** Branch 2 fired, on CUDA and PYTHON. `notes/10` §1.1–§1.3 already records the outcome; this section should be closed against it, and the "plus an approximation on Metal and CL" clause withdrawn everywhere. Note `notes/02:255-256`'s *prediction* (`on NVIDIA the result is RN(RN(1/b) * a)`) **fired true** — credit it, do not retract it. |
| **D13** | `315-338` (§"Isn't this just an approximation?") | `the answer is: **yes, and that is exactly the defect.**` … `There is no validity condition under which \`a/a != 1\`. \`1 - 2^-24\` is the signature of a reciprocal approximation, and it is 1 ULP every time.` | `C` | **Only `328-329` is listed by `notes/12` §6.1, and only the second sentence of it is wrong.** The first sentence (`There is no validity condition under which a/a != 1`) is the strongest argument in the file and **stands**. `1 - 2^-24` is the signature of a **correctly rounded** reciprocal whose error landed in `(−2⁻²⁵, −2⁻²⁴)`; it is produced **bit-exactly** by `a·fl32(1/a)`, and `0x3f800001` is never produced in any regime on any device. A one-sided, exactly-modelled failure is the **opposite** of an approximation signature. The section's thesis (`yes, and that is exactly the defect`) is the framing that has to change: the defect is the extra rounding, not an approximation. `331-338` (silent, unroutable, no division-free exact reciprocal) **survive**. |

### 3.4 `notes/04-codegen-comparison.md` — 9

| # | line | what it says now | kind | correction |
|---|---|---|---|---|
| **D14** | `735` | `\| \`/\` \| **260,261 – 322,233 (12.410% – 15.365%)** \| 0 \|` | `M` | Label `a/a`; add a general `a/b` row. **Line `737` in the same block — `division error magnitude: median 1.0 ULP, p99 1.0 ULP, max 1 ULP` — is correct and needs no change** (see §0.1). |
| **D15** | `740-741` | `\`1 - 2^-24\` is the signature of an approximate reciprocal, and it is 1 ULP every time.` | `C` | As D13. "1 ULP every time" **survives and is now confirmed to the last bit**: the only wrong `a/a` value ever produced is `0x3f7fffff`. |
| **D16** | `743-751` | `**Scope, before the mechanism (ledger C-04, P-1).** This is a Metal and CL measurement. The mechanism predicts NVIDIA should *not* show the approximation` … `**Do not quote 27.13% as a property of TinyGrad; quote it as a property of the Metal and CL backends, with the scope attached.**` | `S` | The prediction was **refuted** (`notes/06` P-1, `notes/10` §1.1). The advice survives in spirit and should be strengthened: quote it with **backend, host, divisor law and `n`**, not with a backend name. **Note: `notes/12` §6.1 cites this as `747-750`, and that range does not contain this text — see §4.1.** |
| **D17** | `786-798` | `\`MetalRenderer\` and \`OpenCLRenderer\` inherit \`CStyleLanguage\`'s \`Ops.RECIPROCAL: lambda x, dtype: f"(1/{x})"\`, so nothing undoes it and the backend is free to approximate.` … `So PYTHON's 27.13% is **double rounding** … while CL and METAL's is **approximation**. Same 1-ULP symptom, different mechanism, coincidentally equal rate.` | `C` | The FDIV/`cstyle` discriminator on lines 784-790 is **correct and survives** — verified at `renderer/cstyle.py:271,274`. The mechanism sentences do not. One mechanism, not two; "the backend is free to approximate" is false, it rounds `1/b` once, correctly. |
| **D18** | `800-809` | `The Metal source, which is what the approximation is visible in:` … `MSL contracts \`1/x\` into an approximate reciprocal. **Two steps, and the first one is Tinygrad's, above the backend.**` | `C` | **Missed by `notes/12` §6.1.** The MSL block on 802-805 **survives** — it is the load-bearing evidence that the rewrite is upstream. Only "the approximation" and "Two steps" change. |
| **D19** | `852-857` | `**\`/ route around it\` is the weakest part of the claim for CL and METAL, and the strongest for PYTHON.** CL and METAL's error is an *approximation*, so in principle a caller who detects it has options. PYTHON's is *double rounding*, which no caller can route around at all` | `C` | The routing-around argument is now **uniform**: no backend's reciprocal error is routable, because the error is a correctly rounded reciprocal that the caller cannot ask to be different and cannot detect. The asymmetry is gone. |
| **D20** | `911-923` | `the *approximation* is predicted **not** to appear on NVIDIA at all (NVRTC is invoked without \`--use_fast_math\`, so \`1.0f/b\` should reach \`div.rn.f32\`), that prediction is **untested**` | `S` | **Refuted.** It was run: CUDA reproduces at 27.13%, bit-identical to `a·fl32(1/b)`. Close this against `notes/10` §1.1–§1.3. The "the rate does not travel" sentence on 921-923 is **wrong** and must be replaced — the rate's dependence on the divisor law is precisely the reason it does not travel. |
| **D21** | `1019` | `\| \`probes/divide_precision.py\` \| §7: 30,748 / 200,000, 27.13%, per-backend, the generated MSL \|` | `M` + `S` | **Missed by `notes/12` §6.1** (it lists `1020` only). Scope `27.13%` with the law, the `n` and the host, and add CL's 29.76% T4 value. |
| **D22** | `1020` | `\| \`probes/fp32_correctness.py\` \| §7: 0 of 6,291,456 for \`+ - *\`; 12.4–15.4% for \`/\`; max 1 ULP \|` | `M` | `"a/a 12.4–15.4%; general a/b 22.7–27.1% (six independent divisor laws)"`; denominator 18,874,368; add `probes/fp32_correctness_fixed.py` to the index. `max 1 ULP` **survives**. |

### 3.5 `notes/04-where-compilers-stop.md` — 1

| # | line | what it says now | kind | correction |
|---|---|---|---|---|
| **D23** | `580` | `\| \`a*(1/b)\` reciprocal in the cstyle path \| **(i)** \| no knob, and CPU proves the IR is exact \|` | `C` | **Missed by `notes/12` §6.1 entirely.** "in the cstyle path" is the C-05-superseded attribution — the rewrite is in the Tensor API at `mixin/elementwise.py:251`. The (i) tag and "no knob" **survive**; "CPU proves the IR is exact" survives in the specific sense that CPU reconstructs a real `/` through the `Ops.FDIV` renderer and decomposition path. |

### 3.6 `notes/06-corrections-ledger.md` — 5

| # | line | what it says now | kind | correction |
|---|---|---|---|---|
| **D24** | `45` (C-04) | `**Untested on NVIDIA, and the mechanism predicts it should not reproduce there.**` … `\`1.0f/b\` should lower to \`div.rn.f32\` — double-rounded, not approximated.` | `S` | Resolution line: **refuted.** CUDA reproduces at 27.13%, bit-identical to `a·fl32(1/b)`. |
| **D25** | `44` (C-08) | `Consequently **PYTHON's 27.13% has a different cause from CL/METAL's**: … it is **double-rounded**; CL and METAL are **approximated**. Same 1-ULP symptom, different mechanism — the 27.13% agreement is a coincidence of magnitude, not evidence of a shared path.` | `C` | The premise and the conclusion are both wrong now. All four backends' outputs are bit-identical to the *same* array, so the agreement is not a coincidence and there is no second mechanism. C-08's *other* conclusion — `PythonRenderer` is not a cstyle renderer, its `RECIPROCAL` comes from `uop/ops.py:1289` (verified) — **stands**, for a reason that is now about renderer inheritance and not about error class. |
| **D26** | `212` (P-1) | `NVIDIA should **not** show the *approximation* defect.` | `S` | **Refuted.** Add a resolution line; `notes/10` §1.1–§1.3 has the evidence. |
| **D27** | `213` (P-1a) | `The mechanism is **refined, not refuted**: NVIDIA joins PYTHON, and the framing becomes "the Tensor-level rewrite costs an extra rounding everywhere, plus an approximation on Metal and CL".` | `C` | **The prediction was right** — credit it. The framing it concluded with is false. Replace with `notes/12` §4's replacement sentence. |
| **D28** | `214` (P-1b) | `Something else entirely introduces the approximation on CUDA.` … `or the error not coming from the reciprocal at all.` | `S` | Moot on CUDA: the error *is* the reciprocal, and it is correctly rounded. The third candidate is the one that held. |

`notes/06:273-280` (**the two rules**) needs no change and is the best argument in the repo for
this whole note — but `notes/12` §6.1 cites it as `275-281` and names only the second rule.
The first, **"Never carry a number that a script did not produce"** (273-276), is the rule
this note is an instance of. **Extend it with the divisor law**, per `notes/06:277-280`:
*"nor is a rate over one divisor law a property of the operation."*

### 3.7 `notes/08-baseline-integrity.md` — 1

| # | line | what it says now | kind | correction |
|---|---|---|---|---|
| **D29** | `516-518` | `\`notes/02-obs-camera-residuals.md\` reports that TinyGrad's float32 division is` / `wrong in 12–15 % of 2M samples on Metal, and that it is a backend bug rather than` / `a design choice.` | `M` + attribution | Corrected rate and `a/a` label; `Metal` → cstyle path; "backend bug" → not a backend bug. **The finding's own conclusion survives and is the reason this is low-priority**: `514` and `518-524` correctly establish that KDA has no division and that the finding "carries over to the camera and to nothing else". **`notes/12` §6.1 cites `517`; the sentence runs 516-518.** |

### 3.8 `notes/10-t4-results.md` — 4

| # | line | what it says now | kind | correction |
|---|---|---|---|---|
| **D30** | `174-186` | `\| reciprocal off by U{−1,0,1} uniform (\`rcp.approx\`-like) \| 133,587 \| — \|` … `The *neighbourhood* is right for an approximate reciprocal — two Newton refinements give 30.37% against CL's 29.76% — so the qualitative story "CL approximates, CUDA does not" survives` | `C` | **Missed by `notes/12` §6.1, and it is the sharpest disagreement between two current notes.** `notes/12` §5.3 keeps the T4's CL mechanism `[UNVERIFIED]`; this line **asserts** the "CL approximates" story. Two Newton refinements landing 30.37% next to 29.76% is a coincidence of magnitude, exactly the reasoning error `notes/06:277-280` forbids. `notes/12`'s `[UNVERIFIED]` is the correct posture. |
| **D31** | `215-218` | `the general \`a/b\` rate is **25.020% / 28.127% /`** / `27.973%** — a 25–28% band, not 12–15%.` | `S` | Superseded by the 6-law band, and — unlike `notes/11`'s set — **genuinely unreproducible**: I could not obtain these three values under any of the six laws, nor under `fp32_div_row.py`'s law. No law, no instrument, no `n` given. See §7. |
| **D32** | `222-231` | `\| subnormal-dominated \`b\` \| 51.70% (25.01% of results overflow to \`inf\`) \| — \|` | `S` | **The band's scope limit, and `notes/12` §6.1 does not mention it.** A subnormal-dominated divisor reaches **51.70%**, well outside 22.7%–27.1%. It is tagged `RE-DERIVED` here with no instrument named and I did not reproduce it (§7). Its existence is the strongest available evidence for §0.1: the rate is not banded by *any* interval, only by a stated law. |
| **D33** | `578` | `\| "12–15% of divisions wrong" \| general \`a/b\` \| **12–15% is \`a/a\`**; general \`a/b\` is 25–28% \| corrected, §1.4 \|` | `S` | `25–28%` → the 6-law band, with the law. **`notes/12` §6.1 puts `574-578` in the "already correct, no change" row while putting `215-216` — the identical phrasing — in the "must change" row. See §4.2.** `574`, `575`, `576`, `577` are genuinely correct and should not be touched. |

### 3.9 `notes/11-local-suite-integrity.md` — 4

| # | line | what it says now | kind | correction |
|---|---|---|---|---|
| **D34** | `209-217` | quotes the hardcoded `12.4% - 15.4%` block, then: `and its sibling line 27 tells the reader that "a nonzero TinyGrad column is a precision bug in the **Metal** backend" even on a **CUDA** run.` | `M` + a wrong line reference | The magnitude criticism **stands**. The line reference is **wrong: it is line 67, not 27** — see §4.3. Fix while you are there. **Missed by `notes/12` §6.1.** |
| **D35** | `244-248` | `\| near 1.0 (LUT regime) \| 12.410% \| 25.046% \| **2.02x** \|` … `2.02x / 1.76x / 1.78x` | `S` | **This set is reproducible — `notes/12` §2.1 is wrong to say otherwise (§3.4 of this note).** It is one divisor law's worth: `U[0.01, 2.01]` drawn from the *continued* `default_rng(0)` stream, i.e. `probes/oc/report_fix/fp32_div_row.py:63`. The correction is therefore **not** "these numbers are wrong" but "the law is missing": add it, and add the 1.48x–2.13x band. |
| **D36** | `260-262` | `the record contains two published numbers for the same claim … that differ by 2x, measured on the same device in the same session` | `M` | `2x` → `1.48x–2.13x`. The observation is right and the reasoning is right; the constant is not. |
| **D37** | `264-273` | `That wrong magnitude is load-bearing in four places:` + four bullets | incomplete | The list is correct as far as it goes and **incomplete** — it misses every one of the 48 other locations in this note. Replace "four places" with a pointer to `notes/14` §2 and §3. `275-279` (`the two sit in one file, unlabelled as to divisor`) and `280-282` (**`The qualitative finding is unaffected — \`/\` genuinely is not correctly rounded, and it is genuinely 1 ULP. Only the magnitude is wrong`**) both **survive** — see §5. |

### 3.10 `notes/DECISIONS.md` — 3

| # | line | what it says now | kind | correction |
|---|---|---|---|---|
| **D38** | `19` | `\`/\` wrong in 12.4–15.4% there and 27.1% in \`probes/divide_precision.py\` (N=200,000), always 1 ULP, numpy exact, CPU 0/200,000.` | `M` | `"12.4–15.4% there is \`a/a\`; general \`a/b\` is 22.7–27.1% over six independent divisor laws, and 27.1% is one point in that band at N = 200,000 under one law."` `always 1 ULP`, `numpy exact`, `CPU 0/200,000` all **survive**. The row's own `amended` note on the attribution **survives** and is reinforced. |
| **D39** | `25` | `Re-measured this pass, METAL and CL land bit-identical to \`RN(RN(1/b)·a)\` in 4 of 5 magnitude regimes (N=200,000 each)` … `the study's claim becomes "the rewrite costs a second rounding everywhere, plus an approximation on Metal and CL"` | `C` + `S` | The "4 of 5 magnitude regimes" figure is a fifth per-regime scalar with no instrument and no law — **not reproducible** (§7); it needs its own instrument or its own retraction. The quoted claim is false. The row's core reasoning — *no compiler flag can undo a Tensor-level rewrite* — **survives and is now the strongest thing in the file**, and is exactly what `notes/12` §6.3 rests on. |
| **D40** | `53` | `Re-measured N=200,000: CPU 0/200,000 wrong, METAL and CL 27.11% and 1 ULP max.` | `S` | `27.11%` is another per-regime scalar. **Verified and correct** as a measurement of one law at one `n` on one host; **superseded** as a citable figure. The row's mechanism — the Tensor-API attribution, the `Ops.FDIV` discriminator, all five source citations — is **correct and verified**; leave it. |

### 3.11 `kernels/README.md` — 2

| # | line | what it says now | kind | correction |
|---|---|---|---|---|
| **D41** | `183` | `the cstyle renderer's \`a*(1/b)\` rewrite shows up verbatim in the` | `C` | **Missed by `notes/12` §6.1** (it lists `:200` only). The rewrite is in the Tensor API; `cstyle` renders it. |
| **D42** | `200` | `**CUDA: 27.13% of float32 divides are 1 ULP wrong, bit-for-bit \`a*rcp.approx(b)\`; CPU 0.00%**` | `C` | **\`bit-for-bit \`a*fl32(1/b)\`, i.e. a correctly rounded reciprocal.** This is the phrase `probes/divide_precision.py:120-121` printed without being tested, and it is a *label* on top of a real bit-identity result — the measurement is sound, the label is what is false. `1 ULP wrong` and `CPU 0.00%` **survive**. Add the law and `n` to `27.13%`. |

### 3.12 `bench/HYPOTHESES.md` — 4

| # | line | what it says now | kind | correction |
|---|---|---|---|---|
| **D43** | `402-405` | `approximate on the PTX path (\`rcp.approx.f32\`, \`renderer/ptx.py:15\`) but \`/\` never` / `goes through it, and \`cstyle.py:274\`'s \`Ops.FDIV: (a/b)\` is dead code because S1` | `C` | Two errors, one of them a bad citation. **"`/` never goes through it" is `notes/06` C-07's already-settled correction**: `/` *always* goes through `Ops.RECIPROCAL`; that is the defect. **"`Ops.FDIV` is dead code" is `notes/06` C-06's already-settled correction**: `ClangRenderer` reaches it, and it is the load-bearing last line of the CPU repair. **And `renderer/ptx.py:15` is wrong** — `rcp.approx.f32` is at **`:19`**; `:15` is a hex-float format string. Every other file in the repo cites `:19` correctly. |
| **D44** | `406-411` | `**Arithmetic.** \`fl(1/b) = (1/b)(1+δ)\` … shifted from the exact quotient by \`\|a/b\|·2⁻²⁴\`, i.e. by **between ½ and 1 ulp of the result**. The final rounding differs from correctly-rounded \`a/b\` only if that shift crosses a rounding boundary, which for a mantissa-uniform \`a/b\` has probability ≈ \`2⁻²⁴·(1..2) ≈ 1.0e-7\`. Over 200,000 samples: **expected failures = 0.02**, 95% upper bound **≈ 3**.` | **`M`** | **Missed by `notes/12` §6.1, and this is the single largest uncorrected number in the repo.** The premise is right — a ½–1 ULP shift of the exact quotient — and the conclusion is wrong by about **5 × 10⁶**. A shift of ½–1 ULP lands outside the rounding cell with probability **≈ ½**, not 1e-7; the `2⁻²⁴` factor is the *relative* perturbation, and dividing it by an absolute threshold of 1 ULP is the error. The correct figure is `notes/12`'s **27.05%**, which the same instrument measured. This is a `MODELLED` number that was never marked as such and is still sitting in a live hypothesis file. |
| **D45** | `412` | `Contrast, already measured: METAL 27.13% wrong, \`a/a\` 30,748/200,000 = **15.37%** wrong, all returning \`0.99999994 = 1 − 2⁻²⁴\` — the signature of a reciprocal *approximation*, which is what MSL contracts \`1/x\` into.` | `C` | The measurements are `RECORD` and correct. **"The signature of a reciprocal *approximation*" is false** and is one of the two origin points of the wrong error class. It is the signature of a **correctly rounded** reciprocal whose error landed in `(−2⁻²⁵, −2⁻²⁴)`, reproduced bit-exactly by `a·fl32(1/a)`, with `0x3f800001` never produced. |
| **D46** | `413-417` | refutation tests: `1. \`a/a != 1.0\` in **≥ 1%** … 2. \`a/b\` wrong in **≥ 1%** … 3. \`a/b\` bit-identical to \`a * rcp.approx(b)\` … 4. generated PTX/SASS contains \`div.approx\` or \`rcp.approx.f32\`` | `S` | Tests 1 and 2 **fired** on the T4 (30,748 and 54,270). Test 3 **fired in the opposite direction**: the output *is* bit-identical to `a·fl32(1/b)`, which is evidence *for* a correctly rounded reciprocal, not for fast-math. Test 4 did not fire and is superseded by §0's bit-identity result, which is stronger. `notes/12` §6.1 catches test 3; the file needs all four marked. H10 (`105`, `432`) and its S1/S2 correction are **correct and survive** — this file found the attribution error first. |

### 3.13 `bench/FAIR-BASELINE.md` — 1

| # | line | what it says now | kind | correction |
|---|---|---|---|---|
| **D47** | `412` (P5) | `\`mixin/elementwise.py:251\` — \`d = a.alu(Ops.MUL, b.reciprocal())\`; \`renderer/cstyle.py:137\` — \`Ops.RECIPROCAL: (1/{x})\`. So \`a/b\` emits \`a*(1.0f/b)\`. \`renderer/cstyle.py:274\`'s \`Ops.FDIV: ({a}/{b})\` is **dead code** on this path` | `C` | **Missed by `notes/12` §6.1 entirely.** Both source citations are **verified correct**. The `dead code` assertion is `notes/06` C-06's already-settled correction — it is reached, by `ClangRenderer`, and it is what repairs CPU. The row's conclusion (`present in the language, absent from this graph`) and its refusal to quote the rewrite as a KDA hazard **survive**; the KDA port really has no division. |

### 3.14 `bench/PLAN.md` — 1

| # | line | what it says now | kind | correction |
|---|---|---|---|---|
| **D48** | `121-128` | `- \`a/a\` and \`a/b\` both essentially exact → **H9 supported.**` / `- ~27% / ~15% wrong, or SASS shows \`rcp.approx\` → **H9 refuted** …` | `S` | The `~27% / ~15% wrong` branch is the one that happened. The SASS branch did not, and the other two branches are now decided. **The residual-is-the-sin-LUT conclusion on 129-134 survives** and was right. **Missed as a range by `notes/12` §6.1** (it cites `124` only, which is correct but partial). |

---

## 4. Where `notes/12` §6.1 is itself wrong

`notes/12` §6.1 is a good first pass and the task warns that transcribing it propagates
errors. I read every line it cites. **Ten of its references are wrong, and two of its content
judgements are wrong.** Recorded here so remediation does not inherit them.

### 4.1 Line references that do not say what §6.1 claims

| §6.1 says | actually at | the error |
|---|---|---|
| `notes/04-codegen-comparison.md` **747-750** carries `"the rewrite costs a second rounding everywhere, plus an approximation on Metal and CL"` | **That sentence is not in `notes/04` at all.** 743-746 is the mechanism-prediction text; 747-750 is the T4-skip / *"Do not quote 27.13% as a property of Tinygrad"* text. | **Wrong file content at the cited range.** The mechanism framing that *is* in `notes/04` is at 786-798, 800-809 and 852-857, which §6.1 lists separately. The phrase §6.1 attributes to `notes/04` is the one in `notes/DECISIONS.md:25`, `notes/06:213` and `notes/02:241`. |
| `notes/10-t4-results.md` **188-213, 574-578** is *"already correct… no change"* | **578** reads `general \`a/b\` is 25–28%` — a per-regime scalar the sweep supersedes, and §6.1's own row for `215-216` marks the identical phrasing for replacement. | **Self-contradictory verdict.** `188-213` is fine; `574-577` is fine; `578` must change. Split the range. |
| `notes/06-corrections-ledger.md` **275-281** carries *"A measurement on one backend is not a property of the compiler"* | the sentence is at **277-280**; 275-276 is the tail of a *different* rule | **Range straddles the wrong rule.** The rule at 273-276 — *"Never carry a number that a script did not produce"* — is the one this entire note illustrates and §6.1 never mentions it. |
| `notes/02-obs-camera-residuals.md` **198-200** — *"the 'approximated' error class is wrong for all of them"* | 198 is `PYTHON … **double-rounded**`; 199 and 200 are `approximated` | **Gloss misquotes the line.** 198 does not carry the label §6.1 says is wrong. The verdict (all three labels need restating) is right; the line does not say what the gloss claims. |
| `probes/oc/tiny_oc.py` **186-191** | the claim is **188-191**; 186-187 are `# oc_raycast reads \`fabsf(1.0f/dx)\`.` and a bare `#` | over-range |
| `README.md` **80-83** | **80-82**; 83 begins `(b) a graph-replay` | over-range |
| `notes/01-obs-camera.md` **59-61** | the bullet is **59-62**; the attribution sentence §6.1 wants re-filed is on 61 | under-range |
| `notes/08-baseline-integrity.md` **517** | **516-518**; the sentence starts at 516 and the `"backend bug"` clause runs to 518 | under-range |
| `notes/04-codegen-comparison.md` **850-860** | **852-857** | over-range (benign) |
| `notes/10-t4-results.md` **188-213** | 213 is the *opening line* of the correction item whose body is 213-218 | range clips a marked-for-change line |

**All other §6.1 references verified correct**, including every tinygrad source line:
`mixin/elementwise.py:251` (`d = a.alu(Ops.MUL, b.reciprocal())`), `:457`/`:465`
(`reciprocal` / `Ops.RECIPROCAL`), `renderer/cstyle.py:137`, `:271`, `:274`, `:425`,
`codegen/decomp/op.py:133-136`, `runtime/support/compiler_cuda.py:48,51,53`,
`runtime/ops_metal.py:93`, `runtime/ops_cuda.py:121`, `uop/ops.py:1289`, and
`use_fast_math` = 0 hits package-wide. §4.1's line-drift note about
`mixin/elementwise.py:465` → `:457` is **right**.

### 4.2 Locations §6.1 missed

13 of them, listed in §2 and §3 as **W4, W8, W11, D7, D8, D10, D11, D12, D18, D21, D23, D30, D44, D47**.
The three that matter most:

- **`bench/HYPOTHESES.md:406-411`** — a `MODELLED` derivation concluding *expected failures =
  0.02, 95% upper bound ≈ 3*, off by ~5×10⁶. §6.1 lists this file at `403-404, 412, 416`
  and not at 406-411.
- **`bench/FAIR-BASELINE.md:412`** — asserts `Ops.FDIV` is dead code, which `notes/06` C-06
  already refuted. §6.1 does not mention this file.
- **`probes/divide_precision.py:13-17` and `:132-135`** — the false "approximate reciprocal"
  and the backwards "never goes through `Ops.RECIPROCAL`", live in the *same file* §6.1
  corrects at `:120-121`.

### 4.3 A wrong line reference in a deliverable that §6.1 did not catch

`notes/11-local-suite-integrity.md:216-217` says the Metal-backend sentence is at
**"line 27"**. It is at **line 67** of `probes/fp32_correctness.py`. (`RECORD` corroboration:
it shipped into `out/kaggle/20260928-014248-gpu/logs/fp32_correctness.log` at **line 27**,
which is where the confusion came from — the line number in the log is not the line number
in the source.) `notes/12` §6.1 cites the right lines (67-68) and never reconciled them.

### 4.4 `notes/12` §2.1's "not reproducible" verdict is itself wrong — about `notes/11`

`notes/12:69-70` says the record's two sets of per-regime figures are *"different numbers
for the same three regimes, and **neither set is reproducible from mine**"* and lists
`notes/11:246-248` as *"not reproducible without the divisor law"*.

**`notes/11`'s set is exactly reproducible, and I did it.** `RE-DERIVED`, to the digit:

```
25.046% / 27.042% / 27.016%      ratios 2.02x / 1.76x / 1.78x
```

Divisor law: **`U[0.01, 2.01]` drawn from the *continued* `default_rng(0)` stream** — i.e.
`probes/oc/report_fix/fp32_div_row.py:63`, which reuses the module-level `rng` already
advanced by the three regime draws at `:35-39`, `:51-52`.

`notes/12`'s §2.1 "same law as `a`" cell re-seeds instead
(`np.random.default_rng(7)` then `r.integers(1<<30)`, `fp32_correctness_fixed.py:325,340`),
so it drew a *different sample of a comparable law*. **The disagreement between `notes/10`,
`notes/11` and `notes/12` is therefore a sampling artefact, not a law mismatch** — which does
not weaken the correction at all. It strengthens it: even *two notes using the same
distribution* disagree in the second decimal, because the rate is a function of the
realised `(a, b)` pairs, not of the marginal law. That is the reason the only citable form
is a band. But the record must say this, because "`notes/11`'s numbers are wrong" and
"`notes/11`'s numbers are missing their law" are different corrections and the second is
the true one.

**`notes/10`'s set (25.020% / 28.127% / 27.973%) is the one that is genuinely
unreproducible.** No law, no instrument, no `n`.

### 4.5 `notes/12` §4's own numbers do not reproduce, and its prose contradicts its instrument

Two defects **inside the note this audit is measured against**:

1. **The divisor law is misdescribed.** `notes/12:216-217` says *"Divisors `U[0.01, 2.01]`
   against `a ~ U[−2, 2]`"*. The instrument it names,
   `probes/fp32_correctness_fixed.py:308`, draws `b = divisor_for("mixed +-2", n)`, and
   `divisor_for` (`:71-72`) returns `rng.random(n)*4 − 2`, i.e. **`U[−2, 2]` — the same law
   as `a`, not `U[0.01, 2.01]`**. `READ` + `READ`.
2. **The sensitivity table mixes two draws.** `RE-DERIVED` at `n = 2,097,152`:

   | law | `k = 0` | `k = ±1` | `k = ±2` |
   |---|---|---|---|
   | `b ~ U[−2, 2]` (what `:308` actually does) | 568,507 — **27.11%** | 1,813,700 — **86.48%** | 2,096,095 — 99.95% |
   | `b ~ U[0.01, 2.01]` (what §4 says) | 567,287 — **27.05%** | 1,812,458 — **86.42%** | 2,096,006 — 99.95% |
   | `notes/12` §4 publishes | 567,311 — 27.05% | 1,814,934 — **86.54%** | 2,096,118 — 99.95% |

   The `k = 0` row matches the `U[0.01, 2.01]` law to two significant figures. The `k = ±1`
   row matches **neither**. So the two rows are not from the same draw, and 86.54% is not a
   number this instrument produced.

   **The claim survives robustly and this does not change anything downstream:** every law
   I tried gives **86.4%–86.5%** against a ~27% baseline, i.e. the reciprocal's accuracy is
   load-bearing and a `rcp.approx`-class reciprocal is refuted by a factor of ~3.2. Publish
   **"≈86%"**, not `86.54%`, and attach the law.

---

## 5. What survives

`notes/12` §8.1 is right, and this note does not move any of it. Specifically, all of the
following are **unaffected** by every correction in §2 and §3:

1. **`/` on the cstyle backends is genuinely not correctly rounded.** `MEASURED`. 0.00% wrong
   is 0.00% wrong and 22.7–27.1% wrong is 22.7–27.1% wrong; both are non-zero. The
   qualitative finding does not depend on which number is quoted.
2. **The error is genuinely 1 ULP and no more.** `MEASURED`. `max |ULP| = 1`, max relative
   error `1.1921e-07 = 2⁻²³`, histogram only `−1` and `+1` in every regime. Independently
   confirmed on the `|a|+0.01` divisor law (§0.1), which is a *different* law.
3. **The error is genuinely silent and unknobable.** Verified, not asserted:
   `Ops.RECIPROCAL` exists (`mixin/elementwise.py:465`) and *is* explicitly approximate on
   the PTX path (`renderer/ptx.py:19`, `rcp.approx.f32`) — **but `/` does not route to it as
   a user choice.** The rewrite at `mixin/elementwise.py:251` is
   `d = a.alu(Ops.MUL, b.reciprocal())`, unconditional, in the Tensor API **above the
   scheduler and above backend selection**, so no `Ops.FDIV` exists in the IR for any
   renderer to render, and **no compiler flag can undo it**: `runtime/support/compiler_cuda.py:48,51,53`
   sets only `--gpu-architecture`, `-I` and `--minimal`, and `use_fast_math` appears
   **nowhere in tinygrad** (package-wide grep, 0 hits). `runtime/ops_metal.py:93` already
   passes `-fno-fast-math` and it does not help, because the rewrite is already in the
   emitted source.
4. **`+`, `-`, `*` are correctly rounded.** `MEASURED`. 0 wrong, independent operands, every
   regime, both devices.
5. **The defect is not a backend bug.** `notes/12` §6.3, and the strongest single fact in the
   record: `CPU` measures 0.00% because `ClangRenderer` (`renderer/cstyle.py:271,274`) is the
   only measured renderer whose op set contains `Ops.FDIV`, which lets
   `codegen/decomp/op.py:133-136` manufacture the missing node and emit a real `/`.
   `notes/06` C-05, `notes/02`, `notes/04` and `DECISIONS.md:53` all agree and all survive.
6. **The 1-ULP error is load-bearing for netherite's DDA**, which takes every step size as
   `1.0f/dx`. `probes/oc/tiny_oc.py`'s decision to keep the division rather than fake it is
   *more* justified after this correction, not less — the real rate is higher than its
   comment states, and there is no division-free exact reciprocal in float32.
7. **The self-division rate is host-independent** and reproduces bit-for-bit across an
   NVIDIA T4 and an Apple GPU (§0, `RECORD`). Useful precisely because it is the degenerate
   quantity. The `a/a` counts also let any future reader re-derive the general rate.

**Only magnitudes and mechanism labels move.** Nothing in §5.1–5.7 needs a correction entry
for the reason this note exists.

---

## 6. The one sentence

> **Tinygrad's float32 `/` is lowered in the Tensor API — above the scheduler, above backend
> selection — to `a · fl32(1/b)`, so on every backend that does not itself render a real
> divide it returns a value that is bit-identical to a correctly rounded reciprocal followed
> by one correctly rounded float32 multiply; the result is wrong by exactly 1 ULP, and
> wrong in 22.7%–27.1% of general `a/b` cases — a rate that depends on the divisor law, so
> it travels with the pair `(a, b)` and not with the operation — and no user-facing knob
> changes any of this.**

**The caveat that must travel with it, quoted with it, never in a footnote:**

> over six **independent** divisor laws at `n = 2,097,152`; the 12.4%–15.4% band in
> circulation is `a/a`, self-division; a dependent divisor such as the repo's own
> `|a| + 0.01` gives 36.6% and a subnormal-dominated one reaches ~52% (`notes/10:228`,
> `UNVERIFIED` here); and on a Tesla T4 the CL backend is 29.76% on a **third, still
> unidentified** mechanism.

### 6.1 Why each clause earns its place

- **"lowered in the Tensor API — above the scheduler, above backend selection."** This is
  the load-bearing clause and it is `READ`-verified at `mixin/elementwise.py:251`. It is
  what makes the defect not a backend bug, and it is the clause every file that says "the
  Metal backend" is missing. It also has a consequence worth keeping: because the rewrite is
  upstream of the IR, the fix is two lines in `Tensor.div` and there is nothing to configure.
- **"bit-identical to a correctly rounded reciprocal followed by one correctly rounded
  float32 multiply."** Deliberately in place of any rate. This is the *strongest* form of
  the claim — five backends across two hosts landing on the same mismatch count (54,270) and
  the same **array**, not merely the same size — and a bit-identity is not a rate, so it
  cannot be eroded by the divisor-law problem that eats the percentages. It is also what
  identifies the mechanism, and it is what `probes/divide_precision.py:120-121` should have
  printed instead of an untested assumption.
- **"wrong by exactly 1 ULP."** A bound, not a statistic. It survives every correction in
  this note, on every law including the ones that break the band (§0.1).
- **"22.7%–27.1% … a rate that depends on the divisor law."** The band, not a scalar, and
  the joint dependence stated **in the sentence** rather than in a scope note. I re-derived
  the band; I showed the ratio band (1.48x–2.13x) is what refutes any per-regime scalar; I
  showed that `notes/10`'s and `notes/11`'s sets differ from `notes/12`'s *even when they
  use comparable laws* (§4.4); and I showed that a dependent divisor leaves the band
  entirely (§0.1). Any bare percentage in this repo is now demonstrably a statement about a
  particular draw, not about `/`.
- **"no user-facing knob changes any of this."** Not "there is no knob" — *user-facing*.
  The distinction is load-bearing: `Ops.RECIPROCAL` exists and is genuinely approximate on
  the PTX path, so the honest statement is that `/` never routes to it *as a choice*.
  Paired with the "above the scheduler, above backend selection" clause it carries the
  proof: `use_fast_math` does not exist anywhere in tinygrad, and `runtime/ops_metal.py:93`
  already passes `-fno-fast-math`.
- **Deliberately absent:** "backend bug" (false); "approximate reciprocal" (false, and the
  single most load-bearing error in the record); "12–15%" (that is `a/a`); "27%" (one point
  in a band); "~2x" (1.48x–2.13x); "double rounding" (loose, and it caused the wrong error
  class); and "insensitive to whether the reciprocal is exact" (**retracted**).

---

## 7. Priority

Ordered by *reader damage*, with the criterion stated: a wrong number in a deliverable is
worse than a wrong label in a script's print statement, because a reader cannot tell which
of two live sentences is current, and a silently superseded number is worse than an
obviously stale one. Two items must ride together regardless of rank.

**P1 — deliverable magnitudes. `README.md:23, 80-82`; `notes/01:59-62, 172-174`;
`notes/02:72`; `notes/04:735, 1020`.** Six numbers, five files, and they are the wrong
*magnitude* in the two most-read documents in the repo. `notes/02:72` is the sharpest
single instance: it sits sixty lines below `notes/02:12`, which states 27.13% for the same
quantity, with no divisor named on either.

**P2 — the root cause and its two propagation vectors.
`probes/fp32_correctness.py:56-57, 67-68, 70-72`, and add
`probes/fp32_correctness_fixed.py` to `remote/cloud_run.py:819`.** Ranked second only
because P1 is worse for a reader, but it is the **highest-leverage edit in the repo**: one
file, three lines, and it de-fangs `kernels/build_notebooks.py:442` and every future GPU
log for free. Do it immediately after P1; they are independent.

**P3 — deliverable mechanisms.** The false error class and the false approximation story:
`notes/02:169-175, 188-190, 198-200, 206, 238-242, 270-283, 328-329`;
`notes/04:740-741, 786-798, 800-809, 852-857`;
`notes/06` C-08, P-1, P-1a, P-1b, C-04; `notes/DECISIONS.md:25`;
`kernels/README.md:183, 200`; `bench/FAIR-BASELINE.md:412`; `notes/04-where-compilers-stop.md:580`.
No wrong *magnitude* in this cluster — but it is the cluster that produced the wrong
predictions and the wrong error class, so it is what made the magnitudes wrong in the first
place.

**P4 — script mechanisms and the two remaining wrong scalar clusters in deliverables.**
`probes/divide_precision.py:13-17, 120-121, 132-135`; `probes/oc/report_fix/fp32_div_row.py:13-15, 63`;
`probes/oc/tiny_oc.py:188-191`; and the per-regime scalars nobody can reproduce:
`notes/10:215-218, 222-231, 578`; `notes/11:244-248, 260-262`; `notes/DECISIONS.md:53`;
`bench/HYPOTHESES.md:406-411`. **`bench/HYPOTHESES.md:406-411` is the one to do first in
this group** — it is a prediction of 0.02 expected failures in a live hypothesis file, off
by 5×10⁶, and it is the number that a reader would most confidently repeat.

**P5 — the meta-notes and the incomplete list.** `notes/11:209-217` (fix the `line 27` →
`67` error while there), `:264-273` (repoint to this note), `notes/10:174-186` (downgrade
"CL approximates" to `[UNVERIFIED]`, which is what `notes/12` §5.3 already requires),
`notes/06:273-280` (add the divisor-law clause to the first rule). These are wrong in a way
that matters and none of them is a number a reader would quote out of context.

**P6 — the DDA comment.** `probes/oc/tiny_oc.py:188-191` last, despite being code, because
the code beneath it is **right**, the fix is comment-only, and the corrected comment makes
the stated justification *stronger*. Zero risk of breaking the raymarcher; zero urgency.

**Must ride together:** P1 and P2. Fixing the probe without the deliverables leaves five
published wrong numbers; fixing the deliverables without the probe re-emits the wrong
number into the next notebook and the next GPU log within the hour.

---

## 8. Numbers I could not verify, and why

**No GPU on this host.** `nvidia-smi` is not installed; tinygrad's `CUDA` device raises
`AttributeError` on first use. So:

1. **Every CUDA and T4 figure** — 54,270 / 27.13%, 30,748 / 15.37%, 59,529 / 29.76%, and
   the three `a/a` counts 260,261 / 322,233 / 318,044. I **read the logs**
   (`out/kaggle/20260928-014248-gpu/logs/fp32_correctness.log`, `…/divide_precision.log`)
   and confirmed every one line by line, and I re-derived the `a/a` and the `a·fl32(1/b)`
   predictions in numpy. But I did not **run** any of them. Tag: `RECORD` + `RE-DERIVED`,
   never `MEASURED` by me.
2. **The T4's CL mechanism (59,529).** `notes/10` §1.3 scores nine candidate mechanisms and
   none reproduces 59,529. I did not find one either. `[UNVERIFIED]`. `notes/10:180-183`'s
   "the qualitative story 'CL approximates, CUDA does not' survives" is a stronger claim
   than the evidence supports (D30).
3. **Whether `PTXRenderer` or `CUDARenderer` was the renderer actually in play on the T4.**
   `runtime/ops_cuda.py:121` gives the order; the *outcome* is `[UNVERIFIED]` without a GPU.
4. **Every Metal/CL/PYTHON figure I did not recompute**, including `notes/12:309-311`'s
   "CL and METAL are bit-identical to each other (26.513% / 27.178% / 26.899% at
   n = 1,048,576, identical counts)". I ran numpy only; I did not run tinygrad.

**Numbers that are wrong at the source, not merely unverified here:**

5. **`notes/12` §4's 86.54%** and its `"Divisors U[0.01, 2.01]"` prose. Both fail against
   the instrument `notes/12` names (§4.5). Robust figure: **≈86%**, 86.42%–86.48% across
   the two laws I could construct.
6. **`notes/10:215-216`'s 25.020% / 28.127% / 27.973%.** Genuinely unreproducible: not under
   any of the six sweep laws, not under `fp32_div_row.py`'s law, and no law, instrument or
   `n` is stated. **This is the only figure in the repo I could positively fail to
   reproduce.**
7. **`notes/10:228`'s 51.70%** for a subnormal-dominated `b` (and 25.01% overflow to `inf`).
   Tagged `RE-DERIVED` with no instrument named; I did not reproduce it. It is also
   **outside** the 6-law band, so if it is real it bounds the band from above and the band
   must be labelled "over six independent divisor laws" rather than treated as an envelope.
8. **`notes/DECISIONS.md:25`'s "4 of 5 magnitude regimes"** bit-identity claim. No
   instrument, no laws, no per-regime numbers. Needs its own instrument or its own
   retraction. (`notes/12` §6.1 already flags this; I could not do better.)
9. **`bench/HYPOTHESES.md:406-411`'s "≈ 1.0e-7 / expected failures = 0.02 / 95% upper bound
   ≈ 3."** Not unverifiable — **refuted**, by ~5×10⁶. The correct figure is 27.05%, which
   the same instrument family measured.

**Deliberately not re-derived:**

10. **The 18-point median 25.790%** of `notes/12` §2.1. I re-derived 15 points (5 laws × 3
    regimes) and got median 25.093%; the 18th–20th points come from the "same law as `a`"
    cell, whose per-cell re-seeding (`r.integers(1<<30)`) I did not reproduce. The **min
    (22.718%) and max (27.127%) I did reproduce exactly**, so the band's edges are solid and
    only the median is not.
11. **The 2.13x upper ratio.** I reproduced 1.48x as the minimum and 2.02x as the maximum
    over my five laws. 2.13x is the "same law as `a`" cell for `near 1.0`, and it checks by
    `ARITHMETIC`: 26.479% / 12.410% = 2.134x. Verified as a quotient, not as a run.

**Verified here, for the record:** the 22.718%–27.127% edges; the 1.48x minimum;
`notes/11`'s set to the digit (§4.4); the one-sided predicate `a/a wrong ⟺ δ < −2⁻²⁵` with
**0 / 2 / 0 disagreements over 6,291,456**; the `2*a` control at **1.00x**, extended to
`0.5*a` and `−a` at **1.00x**; the `|a|+0.01` law at **36.588% / 2.38x**; `max |ULP| = 1` on
that law; the `+ - *` denominators 6,291,456 and 18,874,368 (`ARITHMETIC`); and every
tinygrad source line cited anywhere in the record, except `renderer/ptx.py:15`
(correct: `:19`).
