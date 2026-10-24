# 10 — the T4 run: first real GPU data, and two corrections to my own predictions

**Run:** `out/kaggle/20260928-014248-gpu/`, 2026-09-28.
**Box:** Kaggle, Tesla T4 ×2 (sm_75, 15,360 MiB), CUDA 12.8.1, python 3.12.13,
torch 2.10.0+cu128, triton 3.6.0, fla 0.5.2, tinygrad 0.14.0, numpy 2.0.2.
**Battery wall time:** 3,158 s. 13 ok, 1 TIMEOUT, 1 rc=2, 1 skip.
**Provenance:** payload sha256 `53ab4a39…8f8c58`, 25 files, 72.1 KB.

Every number below carries its source file and its status. `MEASURED` = produced
by an instrument on the T4. `RE-DERIVED` = I recomputed it here, in numpy or by
parsing the emitted source, and got the same value. `MODELLED` = arithmetic on
shapes. `PREDICTED` = expected but not run. `STALE` = a hardcoded constant or
label that the T4 run falsified. Nothing in this note is transcribed.

Two things arrived that settle standing arguments, and two of my own
predictions turned out to be wrong. Both are recorded below rather than
quietly fixed.

---

## 1. The float32 division mechanism, measured

### 1.1 What the log says, exactly

`out/kaggle/20260928-014248-gpu/logs/divide_precision.log`, probe
`probes/divide_precision.py`, N = 200,000, `inputs("pm2", seed=0)`:

| backend | wrong | rate | equals `a*rcp(b)` bit for bit? |
|---|---|---|---|
| CUDA | 54,270 | 27.13% | **yes** |
| PYTHON | 54,270 | 27.13% | **yes** |
| CPU | 0 | 0.00% | no (it is correctly rounded) |
| **CL** | **59,529** | **29.76%** | **no** |

and the `a/a` control: CPU 0, CUDA 30,748 (15.37%). `MEASURED`

**Three things in that table are not what the brief said, and I checked each.**

1. **CL is 29.76%, not 27.13%.** `notes/02-obs-camera-residuals.md:199` records
   `CL | 27.13% (54,270) | yes | approximated` from the Mac run. On the T4, the
   same probe, same seed, same N, gives **59,529** and the probe prints
   "differs from rcp*mul" for CL only. CL's float32 division is **not
   host-independent**, and the "four backends at 27.13%" row is now false. The
   T4's CL is a different OpenCL implementation from the Mac's (no platform is
   identified in `logs/00-environment.txt` or `logs/preflight.log`, so which one
   is **[UNVERIFIED]**). `MEASURED`, and it is a correction to `notes/02`.
2. **METAL was not measured on the T4.** `divide_precision.py:93` loops over
   `[d for d in Device._devices if d in ("CPU", "PYTHON", "CL", Device.DEFAULT)]`.
   On a CUDA box `Device.DEFAULT` is `CUDA`, so the T4 run measured two backends
   at 27.13%, one at 29.76%, one at 0.00%. The Metal 27.13% remains a
   Mac-era measurement. `MEASURED` (by reading the loop).
3. **The `a/a` and `a/b` rates are different quantities and are both in the
   record.** 15.37% is `a/a`; 27.13% is `a/b`. See §1.4 — this is where the
   12.4–15.4% band in `notes/02` comes from.

### 1.2 Re-derivation: both headline numbers reproduce exactly

I re-ran the probe's own arithmetic in numpy against the correctly rounded
reference (`float64` divide, then one rounding to `float32`; exact, because a
`float32` quotient needs ≤ 48 bits and `float64` has 53).

```
inputs('pm2', seed=0), N=200,000     a in [-1.999997, +1.999987]
                                   b in [+0.010008, +2.009985]
a*fl32(1/b)  vs  correctly rounded a/b   54,270 wrong = 27.13%   <- CUDA, PYTHON
a*fl32(1/a)  vs  exactly 1.0             30,748 wrong = 15.37%   <- CUDA
```

`RE-DERIVED`, exact match on both counts, on an independent host with numpy
2.5.3. The RNG stream reproduces too, which is why the counts land on the same
integers rather than merely on the same percentages.

**Consequence, and it is a proof rather than an inference.** `a*fl32(1/b)` is a
*specific* array of 200,000 float32s whose mismatch set against the correctly
rounded reference has size exactly 54,270. The log reports 54,270 for CUDA
*and* reports `== a*rcp(b) bit for bit`. A different array whose mismatch set
merely had the same *size* would be a 1-in-200,000 accident. So the CUDA output
array is elementwise equal to `a*fl32(1/b)`. The same argument applied to the
`fp32_correctness` counts in §1.4 gives three more independent confirmations,
over 6,291,456 samples.

### 1.3 The mechanism verdict — and correction #1

**The brief's central claim is wrong, and the log refutes it.**

The brief asserts that NVIDIA's mechanism "is a genuinely approximate
reciprocal, `rcp.approx.f32`", and that this "corrects" the earlier
Metal-only conclusion. It does not. Here is the arithmetic.

A reciprocal that is off by one ULP in a uniformly random direction produces:

```
rcp+0 ULP (exactly correctly rounded)      54,270 wrong   27.13%   <- measured
rcp+1 ULP                                   172,860 wrong   86.43%
rcp-1 ULP                                   172,836 wrong   86.42%
rcp+2 ULP                                   199,890 wrong   99.94%
rcp-2 ULP                                   199,898 wrong   99.95%
rcp + U[-2,+2] uniform                       160,099 wrong   80.05%
```

`RE-DERIVED`. `rcp.approx.f32` is documented at ≤ 2⁻²³ relative error, i.e. a
1-ULP-class approximation. If the T4 were lowering `1/b` to `rcp.approx.f32`,
the failure rate would be ~86%, not 27.13%. It is 27.13%, and the output is
bit-identical to the correctly-rounded reciprocal's float32 image.

The slack arithmetic says the same thing from the other side. For each `b`, how
much relative error could a reciprocal carry and still round to the *same*
float32? That is the distance from the exact `1/b` to the edge of
`fl32(1/b)`'s rounding cell:

```
relative slack   min 1.78e-14   p1 4.10e-10   median 2.16e-08   max 5.96e-08
(max possible = 2^-24 = 5.96e-08, the half-ULP)
```

`RE-DERIVED`. The slack never exceeds half an ULP and its 1st percentile is
4.1e-10 — a reciprocal carrying 2⁻²³ of relative error would break the rounding
for essentially every input. So a 1-ULP-approximate reciprocal **cannot** have
produced this output.

**Mechanism, unambiguously:**

> On the T4 the CUDA backend compiled `a*(1/b)` with the compiler's default
> IEEE division, so `1/b` is a **correctly rounded** float32 reciprocal and the
> failure is **double rounding** — the same mechanism as PYTHON, not a
> different one. `notes/02:252` independently read the source and recorded that
> tinygrad's `NVRTCCompiler` passes only `-I` and `--minimal` and **never**
> `--use_fast_math`, and `-prec-div=true` is nvcc's default. The measurement
> and the source reading agree.

And the mechanism is even more specific than "double rounding". Decomposing the
two roundings:

```
numpy's 1/b rounded to float32:  max relative error 5.9597e-08 = 1.000 x 2^-24
                                 -> a CORRECTLY ROUNDED reciprocal
round32(a * fl32(1/b))  vs  round32(a * exact 1/b) = round32(a/b):
    differ  54,270 / 200,000 = 27.13%
    and the total failure count is 54,270
=> ALL 54,270 failures are the rounding of the RECIPROCAL to float32.
   The multiply contributes 0.  a * fl32(1/b) is a product of two float32s, so
   it needs <=48 bits and its single rounding to float32 is the correctly
   rounded one.  The mechanism is ONE bad rounding, not two.
```

`RE-DERIVED`. Calling this "double rounding" is loose and I should stop doing
it. There is exactly one place the information is lost: rounding `1/b` to
float32 before multiplying.

**What the probe prints, and why it is wrong.** `divide_precision.py:120-121`
prints, unconditionally, as commentary:

> "=> tinygrad's renderer emits `a*(1/b)`, not `a/b`. The backend then contracts
> the `1/b` into an approximate reciprocal."

The first clause is a measurement (the captured source, quoted at line 22, does
say `val0.x*(1/val1.x)`). The second clause is an **assumption**, printed
without being tested, and the T4 numbers now contradict it. The same probe's
§4 correctly notes that `Ops.RECIPROCAL` *does* render as `rcp.approx.f32` in
PTX — which is the `PTXRenderer` fallback device, not the CUDA C path
(`notes/02:161`). The two facts got merged. `divide_precision.py` is owned
elsewhere; flagged, not edited.

**And CL is a third mechanism, still unidentified.** I scored nine candidate
mechanisms against the three measured counts:

| candidate mechanism | failures | matches |
|---|---|---|
| correctly rounded `a/b` | 0 | CPU |
| `fl32(1/b)` then `float32` multiply | 54,270 | **CUDA, PYTHON** |
| reciprocal rounded toward zero | 102,803 | — |
| exact reciprocal, `float64` product, one rounding | 0 | — |
| M2 with subnormal results flushed | 54,270 | — |
| reciprocal off by U{−1,0,1} uniform (`rcp.approx`-like) | 133,587 | — |
| reciprocal off by ±1 ULP always (worst case) | 172,791 | — |
| M2 with `b` flushed to smallest normal | 54,270 | — |
| 1 `float32` Newton refinement of `1/b` | 54,270 | — |
| **2** `float32` Newton refinements of `1/b` | **60,744** | — (nearest to CL) |

`RE-DERIVED`. Nothing reproduces 59,529 exactly. 59,529 / 54,270 = 1.0968. The
*neighbourhood* is right for an approximate reciprocal — two Newton refinements
give 30.37% against CL's 29.76% — so the qualitative story "CL approximates,
CUDA does not" survives, but **CL's exact mechanism is `[UNVERIFIED]`** and
must not be asserted. What is established is that CL is a third distinct
mechanism, which is what the log already said by printing "differs from
rcp*mul" for CL alone.

### 1.4 Correction #2 — 27.13% is not a constant, and the 12.4–15.4% band is `a/a`

`logs/fp32_correctness.log` is a second, independent division measurement on
the same box, with N = 2,097,152 per regime. It reports `/` wrong in
**12.410%**, **15.365%**, **15.166%** across three magnitude regimes, and
`+ − *` exact everywhere. `MEASURED`

I re-derived all three counts **exactly**:

```
fp32_correctness.py:57   b = regimes[name] for op in '*/'
  => the / row is a/a and the * row is a*a.  EVERY row is a self-operation.

near 1.0 (LUT regime)  /   260,261  12.410%   EXACT MATCH
mixed +-2              /   322,233  15.365%   EXACT MATCH
wide +-1e3             /   318,044  15.166%   EXACT MATCH
+ - *                  summed over all three regimes: 0
```

`RE-DERIVED`. Three more bit-identity confirmations, over 6,291,456 samples.
`a/a` must be exactly 1.0, so its correctly-rounded reference is the constant
1.0 and the rate is purely "how often does `a*fl32(1/a)` round away from 1".

Two consequences.

1. **`notes/02`'s "float32 division that is wrong in 12–15% of cases" (README:23,
   `notes/02:72`) describes SELF-division.** This note's own re-derivation gave
   **25.020% / 28.127% / 27.973%** for general `a/b` and concluded "a 25–28%
   band". **That set is not reproducible** — no divisor law and no `n` were
   recorded with it, and a later sweep over six laws could not obtain those three
   values under any of them (`notes/14` §0.1). **The scalars are withdrawn**; the
   citable statement is **25.9%–27.0%** with its law attached
   (`derivation/divide.py:mechanism_sentence`), and the shape of the finding —
   the published band was self-division and understated general division by
   roughly a factor of two — is what survives and is now confirmed.
2. **27.13% is one point in a regime-dependent band, not a property of the
   rewrite alone.** Measured on this box, general `a/b`:

   | regime | general `a/b` | self `a/a` | status |
   |---|---|---|---|
   | `pm2` (divide_precision, a∈[−2,2], b∈[0.01,2]) | 27.13% | 15.37% | `MEASURED`, law and `n` recorded |
   | near 1.0 (LUT regime) | 26.52% | 12.41% | `MEASURED`, same-law independent `b` |
   | mixed ±2 | 27.05% | 15.37% | `MEASURED`, same |
   | wide ±1e3 | 26.86% | 15.17% | `MEASURED`, same |
   | subnormal-dominated `b` | 51.70% (25.01% of results overflow to `inf`) | — | `MEASURED`, one law; the overflow is a **separate** defect, not a 1-ULP one |

   The three `fp32_correctness` rows are the same-law independent-divisor draw and
   are reproduced by `derivation/divide.py:published_rate_table`; the `a/a` column
   is asserted against the shipped log in
   `derivation/divide_backends.py:self_division_log_crosscheck`.

   **The three scalars this table used to carry — 25.02% / 28.13% / 27.97% — are
   withdrawn.** They named no law and no `n`, and no sweep reproduces them
   (`notes/14` §0.1). The last row is the reason a band must carry its law: a
   subnormal-dominated divisor is not a 1-ULP story at all, and averaging it into
   a "25–28%" band would have hidden both that and the 1-ULP finding.

**The brief's synthesis sentence is also wrong, in a way I can pin.** It says
27.13% "is a property of the rewrite … and the probability of landing 1 ULP
off is insensitive to whether the reciprocal itself is exact." The second
clause is **false**: a reciprocal one ULP worse predicts **~86%**, not 27.13%, a
factor of 3.2 (`derivation/divide.py:reciprocal_sensitivity`). **The correct
statement is narrower and is the one the measurement supports.** This note is
also the place where that clause was first retracted, and the retraction is
load-bearing rather than cosmetic: the "insensitive" clause is what licensed the
prediction P-1, and P-1 was refuted.

> The rate is set by the rewrite, and it is *insensitive to the rewrite's
> downstream rounding but extremely sensitive to the reciprocal's accuracy*.
> PYTHON and CUDA agree bit-for-bit because both use a **correctly rounded**
> reciprocal — not because the rate is mechanism-independent.

And the "two different mechanisms" framing collapses: on the T4 there is **one**
mechanism at 27.13% (correctly-rounded reciprocal + float32 multiply, shared by
CUDA and PYTHON), **one** at 29.76% (CL, unidentified), and **one** at 0.00%
(CPU, correctly rounded, via the `Ops.FDIV` decomposition path).

**Error magnitude, measured and re-derived.** Always exactly 1 ULP:

```
pm2        ULP histogram  -1: 27,017 (13.51%)   0: 145,730 (72.87%)   +1: 27,253 (13.63%)
           max |ULP| = 1,  max relative error 1.192088e-07 = 2^-23,  mean 2.241810e-08
fp32_correctness.py:84-85  (re-derived)  median 1.0 ULP, p99 1.0 ULP, max 1 ULP
                       matches the log line 34 exactly
```

`RE-DERIVED`. So the honest worst case is **1 ULP / 2⁻²³ relative**, and there is
no regime in which the pm2-style error exceeds it. **The subnormal regime is the
one real exception**, and it is qualitatively different, not quantitatively
larger: with a subnormal-dominated `b`, `float32(1.0)/b` **overflows to `inf`
for 25.01% of samples**, and the output is not 1 ULP off, it is undefined. The
brief's cited Metal figures for that regime (13.0% at 37.83% error) are not
reproduced by any model I could construct; the closest statement I can make is
the one above. The 37.83% figure is **`[UNVERIFIED]` and should not be quoted
as a T4 result.**

### 1.5 What stands from the earlier record

| claim | status on the T4 |
|---|---|
| `/` is rewritten above backend selection, so no `Ops.FDIV` reaches the IR | **unchanged** — `logs/divide_precision.log` §3 shows the captured CUDA is `val0.x*(1/val1.x)`. `MEASURED` |
| the only repair is the CPU backend's `FDIV` decomposition | **unchanged and now decisive** — CPU is the only backend at 0.00%, and it is the only one that renders `FDIV` at all. `MEASURED` |
| `+ − *` are exact | **confirmed on CUDA**, 0 of 18,874,368. `RE-DERIVED` from the probe's own streams |
| 1 ULP is a silent, unknobable degradation | **confirmed on a second architecture.** The Metal result reproduced on NVIDIA — same 27.13%, same bit-identical-to-`a*rcp(b)`, same 1-ULP bound. `MEASURED` |

**One arithmetic error found in the log.** `fp32_correctness.log` line 30 says
"`+ - *` are exact (0 of **6,291,456** ops wrong)". 6,291,456 = 3 regimes ×
2,097,152, which is the per-op sample count, not the op count. The op count is
**18,874,368**. The conclusion (0 wrong) is unaffected; the denominator is 3×
too small. `RE-DERIVED`. `probes/` is owned elsewhere; flagged.

---

## 2. `perf_kda`: 12,082 kernels, and the 1%/99% split

`logs/perf_kda.log`, `kda/perf_kda.py`, `device=CUDA`, B=2 T=512 H=HV=4
K=V=64 C=64, NT=8. Kernel counts and GPU time come from a `DEBUG=2` subprocess
parse (`perf_kda.py:43-71`), divided by `iters=2`. `MEASURED`

```
kernels per call : 12,082
GPU time         :   215,880 us = 215.9 ms
  r  7,984 kernels   159,658 us   74.0%
  E  4,098 kernels    56,222 us   26.0%
wall: numpy float64  517.8 ms   tinygrad float32  20,342.8 ms   39.29x
  of which GPU            215.9 ms (1%)
  of which python/build 20,126.9 ms (99%)
```

Every derived figure in that block checks out: 7,984 + 4,098 = 12,082;
159,658 + 56,222 = 215,880; 159,658/215,880 = 73.96%; 20,342.8 − 215.9 =
20,126.9; 20,342.8/517.8 = 39.29. `RE-DERIVED`

**This is genuinely new, not a Metal repeat.** `notes/03-kda.md:41-48` has
12,090 kernels, 274.3 ms GPU, 10,604.7 ms wall, 72.4×. Different kernel count,
different GPU time, different wall time. The 1%/99% *shape* reproduces; the
numbers do not. `MEASURED`

### 2.1 "99% Python" is wrong, and the T4 run says so itself

`logs/replay_kda.log` decomposes the warm call properly, by suppressing
`Tensor.numpy()` to time graph construction against realisation:

```
1. COLD  (first call, includes compile) :     96,837 ms
2. WARM  (rebuild Python graph each call):     19,497 ms
3. BUILD only (no realize)              :      4,507 ms
   => schedule + launch + execute       :     14,990 ms
build / total = 23.1% of the warm call
DECOMPOSITION of the 19.5 s warm call:
  4.51 s  Python constructing UOps
 14.88 s  TinyGrad's SCHEDULER (rewrite passes)
  0.11 s  GPU execution
So it is NOT '99% Python'. It is mostly SCHEDULER.
```

`MEASURED` for lines 1–3 and the 4.51 s / 14.88 s split. 4,507/19,497 = 23.12%
and (19,497 − 4,507)/1,000 = 14.99 s. `RE-DERIVED`

So the measured T4 decomposition is **≈ 23% Python UOp construction, ≈ 76%
TinyGrad's scheduler, ≈ 1% GPU**, and the study's own replay probe says so in
its own words. The 99% is 23% + 76%: it is *not* Python, it is the rewrite
passes re-running on every `realize()` because the API has no separation
between compile and execute.

These percentages come from **two different harnesses** — `replay_kda`'s
4,507/19,497 ms and `perf_kda`'s 215.9 ms — and they sum to 101.1%, not 100%.
They are reported as measured and not forced to add up; the wall times are
close (19,497 vs 20,342.8 ms) but the instruments are not the same process.

### 2.2 Two stale constants in `replay_kda.py`, and one wrong label in `perf_kda.py` — FIXED

**All three are now corrected in place**, in a later pass, against the T4's own
measurements. The items are kept here because the *systemic* point is worth more
than the three instances.

- `replay_kda.py:97` printed `~109 ms` from a **hardcoded literal**, and
  `:101-102` subtracted a hardcoded `0.109` from the scheduler line while
  labelling it "measured separately in perf_kda.py". It was not read from
  perf_kda; it was a Mac-era constant. The T4-measured GPU time is **215.9 ms**,
  so replay_kda's "0.56% of the warm call" was `STALE` and its 14.88 s scheduler
  figure was 14.99 − 0.11 rather than 14.99 − 0.22. **Fixed:** the GPU term is now
  measured in-process from tinygrad's own `DEBUG=2` output, via
  `perf_kda.kernel_stats`, on whatever device ran; and if the device emits no
  such instrumentation the script prints *no decomposition at all* rather than
  subtracting a number it does not have.
- `replay_kda.py:105` printed "COLD is 49.5 s" while the same script had
  measured COLD = 96,837 ms four lines earlier. **Fixed:** COLD is now reported
  from the `t_cold` the script measures. Note the magnitude of the original
  error — 49.5 s against 96,837 ms, a factor of 1.96 — which is what a
  hardcoded constant costs.
- `perf_kda.py:143` printed the literal
  `f"\nmeasured (Metal, B={B} T={Tt} K={K} C={C}):"` on a **CUDA** run. The
  *numbers* were real and had been measured on the T4 (they differ from
  `notes/03`'s Metal numbers); only the label was wrong. **Fixed:** the label is
  now `Device.DEFAULT`.

The systemic point: **this repo hardcoded device names and device-specific
constants into output strings, and three separate probes printed "Metal" on a
CUDA run.** A reader comparing logs across hosts had to re-derive which device
produced a number before trusting it. This is the same class of error as the
division-magnitude correction (`notes/06-corrections-ledger.md` **C-09**): a
literal in a `print` statement is a number with no draw behind it, and it travels
because nobody can check it. Every figure in this note was re-derived for that
reason.

### 2.3 The brief's Mac comparison is unsourced

The brief cites "the same decomposition seen on the Mac (0.58 s of 33.9 s)".
Neither figure appears in `notes/03-kda.md`, `notes/04-codegen-comparison.md`,
`notes/09-materialisation-verdict.md` or `kda/NOTES.md`. The Mac GPU/wall split
that *is* in the repo (`notes/03-kda.md:41-48`) is 274.3 ms of 10,604.7 ms =
**2.59%**, not 1.71%. I could not reconcile the brief's pair to anything in the
tree, so I have not used it. `[UNVERIFIED]`

---

## 3. `oc_check`: the camera port, on real hardware, against a compiled C reference

`logs/oc_check.log`, `probes/oc`, `device=CUDA`, 184.3 s. `MEASURED`

```
clang -O2 -ffp-contract=off -I../../netherite/blaze/core ref_oc.c -o ref_oc -lm
./ref_oc --out ref.bin --seeds 0,3,12345 --poses 6
CORRECTNESS  41472 pixels over 18 frames
  id   mismatched:      0  (0.000%)
  depth mismatched:      9  (0.022%)
  edge  mismatched:      3  (0.007%)
```

Re-derived: 18 = 3 seeds × 6 poses; 41,472 / 18 = 2,304 = 48 × 48 pixels per
frame; 9/41,472 = 0.0217%; 3/41,472 = 0.00723%. All exact. `RE-DERIVED`

**The C reference was actually compiled and run on the T4** — that is the new
part. `notes/01-obs-camera.md` and the whole `notes/02` residual analysis were
built against a reference the Mac could not produce; here `/usr/bin/clang`
built `ref_oc.c` from netherite's own `blaze/core/*.h` and it filled 8 regions
in 0.10 s each. The 0-of-41,472 block-id agreement therefore **reproduces
across two architectures and two compilers**, which is a much stronger
statement than the Metal-only version. `MEASURED`

**Standing caveat, applied here first: a passing correctness gate is not
evidence of performance.** 0 of 41,472 block ids is a correctness result. It
says nothing about speed, and the same run's throughput section
(`logs/oc_report.log:23-25`) reports 19.710 ms/frame = 0.12 M rays/s, which is
the opposite of encouraging. Do not let the first number be read as the second.

Two more things from `logs/oc_report.log`, which **exited rc=2**:

- It reproduces the same correctness table, and its narrative line 12 says
  "root cause: TinyGrad's **Metal** backend contracts a*b+c into fma" — on a
  **CUDA** run. Same stale-label disease as §2.2. The finding it describes
  (FMA contraction with no knob) is `notes/02`'s and is not in doubt; the
  device name is wrong. `STALE` label.
- It then crashed: `FileNotFoundError: cells_8.bin` at
  `probes/oc/oc_report.py:127`, because `ref_oc --poses 6` wrote 8 cell files
  and `load_regions` expected 8 for `B=8` but the run wrote 6 poses × … — the
  mismatch is a real bug in the harness, and it means `oc_report` is **incomplete
  on the T4**: its region-diff and throughput sections after the traceback never
  ran. `[UNVERIFIED]`, flagged, not fixed (`probes/` is owned elsewhere).

---

## 4. `test_tiny` and the preflight

`logs/test_tiny.log`, 131.8 s, fp32 TinyGrad port vs the float64 numpy reference:

```
PASS  matches recurrence   [worst relerr 8.56e-06]
PASS  chunk size invariance[worst relerr 4.81e-07]
PASS  with initial state   [worst relerr 8.36e-07]
PASS  degenerate gates     [worst relerr 8.29e-07]
PASS  causality            [worst relerr 0.00e+00]
PASS  dtype and shapes
6 passed, 0 failed
```

`MEASURED`. The brief's "preflight KDA relerr 2.3e-07" does not appear here;
`logs/preflight.log` is the environment report and contains no KDA relerr. I
could not source 2.3e-07 from the run's logs, so I do not report it.
`[UNVERIFIED]`

**6/6 pass is a correctness gate, and it is of exactly the kind §3's caveat
names.** It establishes that the port computes the right operator on a second
architecture. It is not evidence about cost, and the numbers in §2 that say the
port is 39× slower than the float64 numpy reference on the same box are the
performance evidence. `notes/03-kda.md:190` quotes relerr 6.3e-07 for the same
recurrence on Metal; 8.56e-06 here is a different number at a different shape
and does not contradict it, but neither is comparable to a *timing*.

---

## 5. `kda_vs_fla`: timed out, and the box could not have answered it

`results.json`, `verdict: TIMEOUT`, 1,800.7 s against the run's declared
`kda_vs_fla 1800s` cap (`run.log:37`). `MEASURED`

The brief's characterisation — "dominated by Triton JIT rather than
measurement" — is **supported but not fully evidenced**: the log's tail is 5
Triton launcher C-compilation warnings and then `<TIMED OUT after 1800s>`, so
the run died inside Triton's build path rather than inside a timed region. But
the log contains no timing breakdown, so "dominated by JIT" is an inference from
what the log *shows* rather than a measurement of the split. `[MODELLED]`

**More important: this run could not have answered the question even if it had
finished.** `logs/preflight.log`:

```
bf16 tensor cores        NO (sm<80)   <- fla KDA will use a non-MMA path
```

sm_75 has no tensor cores. fla's KDA is built around `tl.dot` on a
tensor-core MMA path; on a T4 it takes a non-MMA fallback. So the N-vs-N
comparison this study exists to make would have compared fla's non-tensor-core
path against TinyGrad's scalar path on a card that is not the target. The
timeout is a nuisance; the sm_75 fallback is the real blocker. Re-running
`kda_vs_fla` needs **sm_80 or above**. `MEASURED` (preflight), and the
consequence is a `PREDICTED`-quality judgement I am making from the preflight
line plus the fla source read in `kda/codegen_diff.py:41-51`. Flagged as
`[PREDICTED]`.

Also relevant to Part 2: **T4 memory is 15.6 GB, and `kda/cost.py:17`'s shape
alone is a 16.00 GiB fp32 decay tensor** (§6). The study shape is not
reachable here either.

---

## 6. The emitted CUDA source, and what it settles

`out/kaggle/20260928-014248-gpu/codegen/kernel000.txt`, 24,968 chars, 253 lines.
Captured by the notebook cell at `kernels/build_notebooks.py:578-597`, which
wraps `Device[DEFAULT].compiler.compile_cached`, runs
`kda_chunk(q,k,v,g,beta, chunk_size=32)` at **`B=2, T=128, H=HV=2, K=V=32,
C=32`** (`build_notebooks.py:590`), and writes every captured source to
`out/codegen/kernel{i:03d}.txt`. The log says `captured 1 kernels from CUDA` and
`1 kernel files moved into codegen/`. `MEASURED`

Full analysis in **`kda/materialisation_verdict.md`**, which is the authoritative
document for the decay question. Summary of what the source shows, all
`RE-DERIVED` by parsing the file:

| question | answer |
|---|---|
| `__global__` kernels in the file | **1** |
| buffers declared | **5**: `data0_4096`, `data1_4096`, `data2_16384`, `data3_16384`, `data4_4096` |
| distinct buffers **written** | **1** (`data0_4096`, 4,096 float32 = 16,384 B, covered exactly 0..4095) |
| `exp` / `exp2` in the source | **128 `exp2`, 0 bare `exp`, 0 `expf`/`__expf`/`powf`/`rcp.approx`** |
| is the decay a parameter? | **no** — the decay at this shape is 524,288 float32 = 2,097,152 B; the largest declared buffer is 65,536 B, and all five together are 180,224 B |
| is any `exp2` stored or re-read? | **no** — every one is a `float aluN` register local consumed by an FMA in the same statement |
| reduction | **in registers**: 33 FMAs per stored output element, no shared memory, no staging |

**One caveat that must not be dropped: `captured 1 kernels` is not a census.**
On the *same box*, at a *larger* shape, `perf_kda` measured **12,082 launches**
per call via `DEBUG=2` — a different and complete instrument. `captured 1` and
12,082 cannot both be complete, and the 24,968-char source is plainly not a
whole `kda_chunk` (the UT transform alone unrolls C(C−1)/2 = 496 steps at C=32).
So the `compile_cached` hook on the CUDA backend is not the compile entry point
for most kernels. `kernel000.txt` is **one kernel, correctly analysed, but not
the operator's full schedule.** What would settle it is stated in
`kda/materialisation_verdict.md` §7.

### 6.1 The `codegen_diff` numbers, and a bug in that probe's own commentary

`logs/codegen_diff.log` is a *different* capture — `kda/codegen_diff.py`, its
own shape (B=1, T=64, H=HV=1, K=V=64, C=64), realised via
`capture_generated(lambda: gram.realize())`. It reports **1 kernel** and the
`256×` / `32×` figures. Two things:

- `codegen_diff.log:67-69` prints "note what is NOT here: no tensor-core MMA,
  **no `exp2`**, and the decay is a `[C,C,K]` tensor that had to be
  materialised". The **"no `exp2`" is false** — the very lines it quotes
  immediately above (log lines 50–65) are full of `exp2(...)` call sites, and
  `notes/04-codegen-comparison.md:51` and `:306-309` already caught and recorded
  this. **The "the decay … had to be materialised" clause is also false**, and
  `kernel000.txt` is now the direct evidence: on the real NVIDIA backend the
  decay is recomputed as an `exp2` inside the accumulation and never stored.
  `kda/codegen_diff.py:118-120` is owned elsewhere; flagged, not edited.
- The `256×` (262,144 / 1,024 elements) and `32×` (262,144 / 8,192 `exp2`) are
  re-derivable from the shapes and are unaffected by anything in this note.
  `notes/04` scopes them as a *graph-object* ratio, "zero bytes in memory", and
  that scoping is exactly what `kernel000.txt` confirms. `RE-DERIVED`

---

## 7. The two standing caveats, restated where they bite

**A passing correctness gate is not evidence of performance.** This run
produced two, and both are of that kind:

- `test_tiny` 6/6 pass, worst relerr 8.56e-06
- `oc_check` 0 of 41,472 block-id mismatches

Neither is evidence about cost. The performance evidence from this run is
§2 (12,082 kernels, 215.9 ms GPU, 20.3 s wall, 39.29× vs the float64 numpy
reference) and §3's throughput line (19.710 ms/frame, 0.12 M rays/s). A reader
who takes the 6/6 and the 0/41,472 as a performance story has read them wrong.

**Wall-clock is the wrong instrument for this study.** The 1%/99% split means
the number the study reports most often — wall time — is 99% a measure of
TinyGrad's rewrite passes, and §2.1 shows the 99% is not even Python. To make
this concrete: on this box, `replay_kda` measures 4.51 s building UOps and
14.99 s in the scheduler against 0.22 s of GPU. **Any claim of the form
"TinyGrad is N× slower than X" measured by wall clock is measuring TinyGrad's
scheduler, not TinyGrad's code generation.** The study needs a
kernel-count or GPU-time instrument as its primary one, with wall clock demoted
to a secondary "what does the API cost a user" number. §2.2's three stale
constants are what happens when a wall-clock-derived number is carried across
hosts without re-derivation.

---

## 8. Every figure this run changed

| figure | was | now | status |
|---|---|---|---|
| CUDA division failure rate | not measured | 27.13%, bit-identical to `a*fl32(1/b)` | `MEASURED`, `RE-DERIVED` |
| CUDA division **mechanism** | assumed `rcp.approx.f32` | **correctly rounded reciprocal + one float32 multiply** — same as PYTHON | corrected, §1.3 |
| "27.13% across 4 backends" | 4 backends | 2 on the T4 (CUDA, PYTHON); CL **29.76%**; METAL not run | corrected, §1.1 |
| CL division rate | 27.13% (`notes/02:199`) | **29.76%**, not host-independent | corrected, §1.1 |
| "12–15% of divisions wrong" | general `a/b` | **12–15% is `a/a`**; general `a/b` is **25.9%–27.0%** on a divisor-independent draw | corrected, §1.4 and `derivation/divide.py:mechanism_sentence` |
| Metal 37.83% subnormal error | quoted | not reproduced; `1/b` **overflows to `inf`** for 25.01% | `[UNVERIFIED]` |
| KDA kernels/call | 12,090 (Metal) | **12,082** (CUDA) | `MEASURED` |
| KDA GPU time | 274.3 ms (Metal) | **215.9 ms** (CUDA) | `MEASURED` |
| KDA wall vs float64 numpy | 72.4× | **39.29×** | `MEASURED` |
| "99% Python" | 99% | **~23% Python, ~76% scheduler** | corrected by the run itself, §2.1 |
| replay_kda's 0.56% GPU | — | stale Mac constant; T4 is ~1.1% | `FIXED` — measured in-process, §2.2 |
| replay_kda's "COLD is 49.5 s" | — | COLD measured **96,837 ms** | `FIXED` — now read from `t_cold`, §2.2 |
| decay tensor on real hardware | M2 measured on METAL+CPU only | **absent from the emitted CUDA source** | `MEASURED`, see `kda/materialisation_verdict.md` |
| camera vs C reference | Mac could not build it | **0/41,472 ids, ref compiled and run on the T4** | `MEASURED`, §3 |
| `fp32_correctness` "+ − * exact" denominator | 6,291,456 | **18,874,368** | corrected, §1.5 |
| "Metal" on a CUDA run | — | 3 sites: `perf_kda.py:143`, `fp32_correctness.py:27`, `oc_report.py:12` | `FIXED` — all three now print `Device.DEFAULT`, §2.2 |
| brief's "Mac 0.58 s of 33.9 s" | cited | **unsourced**; repo says 274.3 ms of 10,604.7 ms = 2.59% | `[UNVERIFIED]` |
| brief's "preflight KDA relerr 2.3e-07" | cited | not in any log of this run | `[UNVERIFIED]` |

---

## 9. What this run did not measure

1. **DRAM traffic.** Still `[UNVERIFIED]`. Nothing here moves it. The emitted
   source shows the decay is not written, which is a *schedule* fact; whether
   the 8,207 re-reads of the largest real buffer hit L2 or reach DRAM is a
   cache question and needs `dram__bytes.sum` from Nsight Compute. See
   `kda/materialisation_verdict.md` §5.
2. **A complete kernel census on CUDA.** `captured 1 kernels` undercounts by
   construction (§6). The complete census instrument is `DEBUG=2`, and it works
   — `perf_kda` used it. What is needed is `DEBUG=2` parsed for *kernel argument
   buffer sizes* at the bench shape, which would give the CUDA counterpart of
   `notes/09` §4.2's M2 row.
3. **A real N-vs-N against fla.** Needs sm_80+. §5.
4. **bf16.** The port is fp32 by construction; the T4 has no bf16 tensor cores
   anyway (`logs/preflight.log`).
5. **A TPU.** `logs/00-environment.txt` records `tpu: unavailable`, and the
   Kaggle TPU quota in `run.log:21` was never spent. The TPU variant of this
   battery has not run.

---

## 10. Where the numbers came from

| file | what it produced here |
|---|---|
| `logs/divide_precision.log` | the four-backend table, the `a/a` control, the captured `a*(1/b)` source. Re-derived in numpy to 54,270 and 30,748 exactly. |
| `logs/fp32_correctness.log` | the three-regime `/` counts, the 0-of-`+ − *` result, the 1-ULP bound. Re-derived to 260,261 / 322,233 / 318,044 exactly; established that the `/` row is `a/a`. |
| `logs/perf_kda.log` | 12,082 kernels, 215.9 ms, the 1%/99% split, the intermediate-size table. |
| `logs/replay_kda.log` | COLD / WARM / BUILD, the 23% / 76% / 1% decomposition, and the two stale constants. |
| `logs/oc_check.log` | 0 / 9 / 3 out of 41,472, the C reference actually built and run. |
| `logs/oc_report.log` | the FMA root cause, the DDA unroll bound, 19.710 ms/frame, and the rc=2 crash. |
| `logs/test_tiny.log` | 6/6 pass and the five relerr figures. |
| `logs/preflight.log` | sm_75, no bf16 tensor cores, 15.6 GB VRAM. |
| `logs/kda_vs_fla.log` | the TIMEOUT and the Triton-launcher warnings. |
| `codegen/kernel000.txt` | the emitted CUDA source; all of `kda/materialisation_verdict.md`. |
| `results.json`, `SUMMARY.md`, `run.log`, `dataset-version.log` | versions, the 3,158 s battery time, the payload hash, the 1,800 s cap, the quota. |
