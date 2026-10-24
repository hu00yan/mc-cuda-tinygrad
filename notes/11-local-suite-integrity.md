# 11 — Local-suite integrity: what a report reads that nothing produces

**Scope.** The `oc_report` battery item returned `rc=2` on the real GPU run
`out/kaggle/20260928-014248-gpu/`. This note establishes the root cause, the fix,
and the audit of the same failure class across `probes/`, `kda/` and the
Makefiles. I did not edit a single file outside `probes/oc/report_fix/` and this
note; where a finding needs a change in a file another agent owns, the patch
says exactly what to change and the evidence is here.

Every count below was re-derived by running code in this repo. Nothing is taken
from a summary, a note, or a log line. Where a number *is* quoted from a note or
a log, it is because I am checking the note against a re-derivation, and both are
shown.

---

## 1. Root cause, in one sentence

`probes/oc/oc_report.py` hardcoded `B = 32` and loaded `cells_0.bin` …
`cells_31.bin`, while the Makefile's `cells` target emits only 9 fixtures
(seeds 0–7 plus 12345) and `probes/oc/.gitignore` documents a *third*, shorter
set (`0 1 2 3 12345`) — so `make report` raised `FileNotFoundError:
'cells_8.bin'` on every machine, from a clean checkout, ~2.5 minutes in, after
printing the entire correctness section.

```
out/kaggle/20260928-014248-gpu/logs/oc_report.log:44
  FileNotFoundError: [Errno 2] No such file or directory: 'cells_8.bin'
```

It is not a Kaggle artefact. `probes/oc/cells_*.bin` is gitignored and
`kernels/pack.py` deliberately excludes it, so the runner regenerates fixtures
from the recipe — and the recipe is short. On this Mac the same target fails
identically, which I confirmed before changing anything.

### What is produced vs what is consumed

| | seeds | count | re-derived? |
|---|---|---|---|
| `Makefile` `cells` recipe, today | `0 3 12345 1 2 4 5 6 7` | **9** | yes — regenerated all 9, sha256 identical to the 9 on disk |
| `.gitignore` comment | `0 1 2 3 12345` | **5** | read |
| `oc_report.py` needs (`range(B)`, B=32) | `0`…`31` | **32** | read |
| `oc_report.py` needs overall (`SEEDS` ∪ `range(B)`) | `0`…`31`, `12345` | **33** | yes — 33 files, 87.6 MB, all regenerable in 2.0 s |

Three sources of truth, three answers, and no assertion anywhere that they agree.
The on-disk 9 files are byte-identical to what the recipe produces, so nothing
was "lost" — the recipe is simply short by 24 files.

### Which side was wrong: the Makefile

**The Makefile was under-producing; the report's expectation of 32 is correct.**
B=32 is the published configuration, and it re-derives exactly:

| published in `notes/01-obs-camera.md` | re-derived |
|---|---|
| "At B=32 the region buffer is 170 MB" (line 210) | `32 × 113·104·113 × 4 B = 169.98 MB` |
| "`B=32, GPU only : 39.79 ms/frame 1.85 M rays/s`" (line 97) | `32 × 2304 / 39.79 ms = 1.852 M rays/s` |
| "`B=32, wall (rebuild) : 952.45 ms/frame 0.08 M rays/s`" (line 98) | `73,728 / 952.45 ms = 0.0774 M rays/s` |
| "Wall time is flat in B: B=1 costs 923 ms, B=32 costs 998 ms" (line 103) | B-specific by construction |

Lowering the report to `B=8` would have made it pass and silently invalidated
every one of those rows. So the fix belongs in the recipe.

**One correction to the obvious evidence, because I checked it.** The natural
argument is "`notes/01` quotes 1,365 kernels and 101 reduces, so the config must
be B=32". That argument is *wrong*: I measured **1365 kernels / 101 reduces at
both B=8 and B=32, identically**. Those two numbers are set by the 96-step unroll
(≈14 kernels/step) and are batch-invariant. They carry no information about B.
The B-discriminating evidence is the `rays/s` rows and the 170 MB buffer, above.

**A second thing the kernel count does tell us, which is worth recording:**
`notes/01:107` presents "1,365 kernels" inside a B=32 discussion, and Finding 3's
"96 gathers per frame … matches the measured 101 reduce kernels" is a statement
about unroll depth that reads as if it were about batch size. Both are true; only
the second's framing invites the wrong inference.

---

## 2. The fix

`probes/oc/report_fix/` holds the corrected report, an executable form of the
corrected recipe, the two verification scripts, and a patch. Nothing outside that
directory was touched.

### 2a. The Makefile (`probes/oc/Makefile`)

- `NENVS ?= 32` is the single batch count, and the recipe builds
  `$(SEEDS) $(BENCH_SEEDS)` where `BENCH_SEEDS = $(shell seq 0 $(NENVS-1))`.
- `cells` gets a **second loop that asserts it produced what the report needs**,
  and echoes the count. This is the part that actually prevents recurrence: the
  recipe now fails if it is short, instead of the report failing 2.5 minutes
  later inside `np.fromfile`.
- `SEEDS_CSV` is derived from `SEEDS` (`$(subst $(space),$(comma),$(SEEDS))`) so
  `ref.bin`, `check` and `cells` cannot disagree about the seed list.
- `report` and `check` are invoked as `OC_B=$(NENVS) $(PY) …`, so the batch
  reaches the report from the one place it is declared.

### 2b. The report (`probes/oc/oc_report.py`)

Six changes, none of which touches a number or a definition:

1. `B` comes from `$OC_B` (default 32), not a literal.
2. Fixtures resolve against **one** directory. The original resolved the parent
   process's files against `$PWD` and `kernel_us`'s subprocess against
   `__file__`'s directory — two different answers to "where is `cells_0.bin`".
3. A **preflight** pass runs before anything is measured, and reports the exact
   missing seeds and the command that produces them. Cost of the bug: 0.12 s and
   10 lines instead of 150 s and a traceback.
4. The report **checks `ref.bin`'s frame count** against `SEEDS × CANON` before
   indexing it. The original did `ref[si*6+p]`; if `ref.bin` is ever rebuilt with
   `--poses` ≠ 6 that indexes the wrong frames and reports them as correct.
5. The narrative claims are **derived from the measured counts** instead of
   printed unconditionally — see §4, this was a live bug too.
6. `--b-sweep` measures the "wall is flat in B" claim, which `notes/01` publishes
   and no code backed.

### 2c. Verified, not asserted

| check | result |
|---|---|
| `make report` on a **clean tree** (no fixtures, no `ref_oc`, no `ref.bin`) with the patched Makefile + report | **rc=0**, full report printed |
| `make check` on the same clean tree | **rc=0**, `0 / 9 / 3` of 41,472 — identical to `oc_check.log` from the T4 |
| 33 fixtures regenerated from scratch by the patched recipe | 2.0 s, 33 files |
| the 9 pre-existing fixtures vs freshly generated | sha256 **identical**, all 9 |
| all 33 fixtures distinct | 33 distinct sha256 |
| `make report` on the **unpatched** tree | rc=1, `FileNotFoundError: 'cells_8.bin'` (reproduced before any change) |
| `git apply --check -p1` against a pristine copy of all four patched files | applies cleanly, and the result is byte-identical to the two files `make report` passed with |

---

## 3. The corrected report's numbers, and the category definitions

`verify_defs.py` re-derives the quoted numbers from the fixture files rather than
trusting them, and checks the corrected report's categories against `run_oc.py`'s
by two independently written loops that index the same arrays differently.

```
ref.bin header      : 3 seeds x 6 poses
body bytes          : 165888 = 18 frames (x 9216 B/frame) -> True
re-derived pixels   : 41,472   (report prints 41,472) -> True

per-frame mismatch vectors, report loop vs run_oc loop:
  identical entry-wise   : True
  id     totals         : report   0  run_oc   0   nonzero frames []
  depth  totals         : report   9  run_oc   9   nonzero frames [0, 3, 4] counts [3, 2, 4]
  edge   totals         : report   3  run_oc   3   nonzero frames [0, 3, 4] counts [1, 1, 1]
percentages            : depth 0.022%  edge 0.007%
```

- `41,472` is re-derived from the `ref.bin` **header and body length**
  (`165888 / 9216 = 18` frames × 2304), not from the report's own arithmetic.
- `0 / 9 / 3` is re-derived per frame. The corrected report's `id`/`depth`/`edge`
  definitions are byte-for-byte the original's: a per-pixel `!=` against the
  same `ref.bin`, summed over the same 3 seeds × 6 poses × 2304 pixels. The
  per-frame vectors are *element-wise identical* between the two loops, so no
  category moved.
- `0.022%` and `0.007%` are `9/41472` and `3/41472`, matching the Kaggle
  `oc_check` log exactly.
- All 12 residuals are in **frames 0, 3 and 4**, which is seed 0 only
  (frames 0–5 = seed 0, 6–11 = seed 3, 12–17 = seed 12345).

The B=32 run, re-measured on this Mac (Metal):

| | published `notes/01` | re-measured |
|---|---|---|
| `np_oc` id/depth/edge | 0 / 0 / 0 | 0 / 0 / 0 |
| `tiny_oc` id/depth/edge | 0 / 41,472 · 9 · 3 | 0 · 9 · 3 |
| unroll | 96, mean 19.8, p99 83, max 84 | identical |
| waste | 4.8x | 4.8x |
| B=32 GPU-only | 39.79 ms, 1.85 M rays/s | 40.89 ms, 1.80 M rays/s |
| B=32 wall | 952.45 ms, 0.08 M rays/s | 968.45 ms, 0.08 M rays/s |
| kernels / reduces | 1,365 / 101 | 1,365 / 101 (and **also at B=8** — see §1) |
| `ref_oc` | 5.67 M rays/s | 5.63 M rays/s |
| B=1 → B=32 wall | 923 → 998 ms (claimed, unmeasured) | 978.34 → 1023.81 ms (**now measured**, 1.05x) |

---

## 4. A second, quieter bug in the same file: claims printed as if measured

`oc_report.py` **used to** print, unconditionally, on every run:

```
  all id mismatches are depth +-1 or a single edge bit; the DDA reaches
  the same block on every one of 41,472 pixels.
  root cause: TinyGrad's Metal backend contracts a*b+c into fma
  (SPEC rule 4 requires -ffp-contract=off / --fmad=false).
```

**Both are now fixed** in `oc_report.py`, and the fix is the shape this whole
section argues for: the sentences are **derived from the counts and from the
device that actually ran**, and the retracted root cause is named as retracted
rather than reprinted as fact. `probes/oc/report_fix/oc_report_fixed.py` was the
worked version of that change.

Two problems, both now gone. The first sentence was **vacuous at the measured
values** (id mismatches are 0), and the third was a root cause that
`notes/01:43-49` and `notes/02` had already **retracted** in favour of the `sin`
LUT bucket boundary. The report was still printing the superseded explanation as
fact, 8 months of note-editing later.

The corrected report derives both from the counts, and *verifies* the ±1 claim
instead of asserting it — which it can, because it is one `np.abs` over data it
already holds:

```
  block ids agree on all 41,472 pixels, so every pixel reaches the same block;
  the 9 depth and 3 edge residuals are one quantised value, not a different block.
  largest depth error 1 quantised unit(s); differing edge values [0.0, 1.0].
```

`max |Δdepth| = 1` and the only differing edge values are `{0, 1}` — i.e. the
original claim, now as a measurement.

**This is not a one-off; it is a pattern, and it is the strongest argument for
the guard in §6.** `probes/fp32_correctness.py:70-72` used to end every run with

```python
print("\nMeasured result: + - * are exact (0 of 6,291,456 ops wrong). / is wrong in")
print("12.4% - 15.4% of cases depending on magnitude. See")
```

hardcoded, on any device — and its sibling line 27 told the reader that "a
nonzero TinyGrad column is a precision bug in the **Metal** backend" even on a
**CUDA** run. Both are now gone: the probe computes everything it prints, prints
the device it ran on, and prints both divisor rows.

**The generalisation this note did not draw, and should have.** A hardcoded
`print` of a measurement is not a stylistic wart; it is the mechanism by which
`12.4% - 15.4%` reached 58 places. Nothing downstream could check it, because
there was nothing to check it against. The fix is structural and it is now in
place: **a number is derived, never transcribed.** The rates live in
`derivation/`, as executable claims; documents cite the claim that prints them.
A document that cannot derive a number deletes it. See
`notes/06-corrections-ledger.md` §5.

`probes/real_gather.py:9-10` cites "101 extra kernels per camera frame
(**37.6% of GPU time**)" — a figure no probe in the tree produces. **Still
unresolved**: on the prefer-deletion rule, that figure should be deleted rather
than hedged, and the file is not in this pass's scope.

---

## 5. Audit: the same class of failure elsewhere

Listed, not fixed. Ordered by consequence.

### The one that matters most

**A. `probes/fp32_correctness.py`'s `/` row divides each element by itself, so
the published division error rate is ~2x too low.**

```python
for op, fn in (("+",...), ("-",...), ("*",...), ("/",...)):
    b = regimes[name] if op in "*/" else a
```

`regimes[name]` *is* `a` — the same dict entry. So the `/` row computes `a/a`, not
`a/b`. The `*` row is `a*a` (harmless: multiplication is exact either way). The
probe never divides by an independent divisor at all.

Re-derived on Metal, same code path, same process
(`probes/oc/report_fix/fp32_div_row.py`). **The divisor law is part of the
number and is now stated: `b ~ U[0.01, 2.01]`, drawn from the continued
`default_rng(0)` stream, n = 2,097,152.** That is what makes this table exactly
reproducible; without the law it is not, and an earlier pass of this note called
it irreproducible, which was half right:

| regime | the probe's row (`a/a`) | general divide (`a/b`), `U[0.01, 2.01]` | understated by |
|---|---|---|---|
| near 1.0 (LUT regime) | 12.410% | 25.046% | **2.02x** |
| mixed ±2 | 15.365% | 27.042% | **1.76x** |
| wide ±1e3 | 15.166% | 27.016% | **1.78x** |

**These three ratios are one divisor law's worth, and the ratio is a band.**
Over a sweep of independent laws it runs 1.48x–2.13x (median 1.78x); on a
same-law independent draw it is 1.77x–1.86x
(`derivation/divide.py:published_rate_table`). So "understated by 2x" is a
usable summary and "understated by 2.02x" is not a fact. The citable rate is
**25.9%–27.0%** with its law attached
(`derivation/divide.py:mechanism_sentence`).

Three independent confirmations that the left column is the degenerate case:

1. The `a/a` figures are **bit-for-bit the CUDA numbers in the shipped log**
   (`fp32_correctness.log` lines 10/17/24: 12.410% / 15.365% / 15.166%). An
   `a/b` rate does not travel across two vendors unchanged; an `a/a` rate does.
2. `divide_precision.log` **from the same run, same T4** reports `CUDA: 54,270
   wrong (27.13%)` for `a/b`, and its step 1 reports `a/a` at **15.37%** —
   which is the `a/a` number, to two decimals.
3. `probes/divide_precision.py`'s own docstring records METAL at **27.06%**.

So the record contains two published numbers for the same claim — "TinyGrad's
float32 `/` is wrong in *N*% of cases" — that differ by 2x, measured on the same
device in the same session, and the reason is that one of them divides by itself.

That wrong magnitude turned out to be load-bearing in ~58 locations across 19
files; the full catalogue is `notes/14-divide-magnitude-propagation.md` §6 and
every one has been corrected. The four this note named:

- `probes/fp32_correctness.py:71` (the hardcoded "Measured result" block) — **fixed**;
  the hardcoded block is gone entirely and the probe computes what it prints
- `probes/oc/tiny_oc.py:189-191` ("1 ULP off in 12-15% of 2.1M samples per
  magnitude regime") — **fixed**; inside the DDA, so it is a stated
  justification for keeping the division, and the correction *strengthens* it
- `notes/01-obs-camera.md:60` — "**12–15% of 2 M samples** across three magnitude
  regimes" — **fixed**
- `notes/01-obs-camera.md:173` — Finding 5, "Measured wrong in 12–15% of 2 M
  samples" — **fixed**

`notes/02-obs-camera-residuals.md:12` used the **correct** 27.13% for the same
quantity while its own table carried the 12.4–15.4% row — the two sat in one
file, unlabelled as to divisor. **Both are now labelled**, and the correction is
in the ledger (`notes/06-corrections-ledger.md` **C-09**, **C-10**, **C-11**)
rather than living only in prose. **The qualitative finding is unaffected** —
`/` genuinely is not correctly rounded, and it is genuinely 1 ULP. Only the
magnitude is wrong, and the
magnitude is what four files quote.

### Same class, smaller

**B. `probes/oc/run_oc.py --bench N` has the identical bug, one flag away.**
Line 93 is `load_regions([f"cells_{i}.bin" for i in range(B)])` — the same blind
`range(B)` the report had. `--bench` defaults to 0, so `make check` never hits
it; `--bench 32` dies in `np.fromfile` on `cells_8.bin`. My patch does **not**
fix this, because the corrected recipe now guarantees 33 fixtures and any
`B ≤ 32` works — but the failure mode is identical and the guard is not shared.

**C. `probes/oc/oc_report.py:60,126-127,164` resolved fixtures two different ways.**
The parent process used `$PWD`; `kernel_us`'s `DEBUG=2` subprocess used
`cwd=HERE`. `python probes/oc/oc_report.py` from the repo root would have found
`ref.bin` missing while the subprocess found the cells. Fixed in §2b item 2.

**D. `probes/oc/oc_report.py` hardcoded the pose count instead of reading it.**
`for p in range(6)` three times, and `ref[si*6+p]`. `read_ref` already returns
`nseed × npose` frames from the header. Rebuild `ref.bin` with `--poses 64` and
the report silently reports on 3 of 64 poses per seed, and `NPIX_TOTAL` is
computed from `len(ref)` so the pixel count would *not* flag it. Fixed in §2b
item 4.

**E. `probes/gather_scale.py:22` hardcodes `OUT = 32 * 2304`** — the same B=32
restated a third time, this one harmless because the tensor is random rather than
loaded from a fixture. `SRC_SIZES` also embeds `1328176`, the OC region size.
Nothing asserts either against `tiny_oc.RG`.

**F. Four copies of the `DEBUG=2` parser, with four different regexes, none
tested.** `oc_report.kernel_us`, `gather_floor.run`, `gather_scale.run`,
`perf_kda.kernel_stats`, plus a fifth shape in `kda_vs_fla.py`. I checked the
current tinygrad 0.14 format against all of them: they still parse
(`gather_floor` at src=65536/1328176/16777216 → 28.6/40.4/667.7 µs, 1/1/2
reduces, against `notes/01`'s 29.0/44.8/669.9 µs and 1/1/2). So this is latent,
not active. The failure mode is a `ZeroDivisionError` on `src*4/t/1e3`, not a
wrong number, which is the better of the two outcomes. `gather_scale.py:41-42`
even documents that the stats "have landed in both across versions", so the
fragility is known and untested.

**G. `probes/oc/np_oc.py:90` loops `range(200)`, `tiny_oc.OC_STEPS = 96`, and the
analytic bound is 86.1.** Three different trip counts for the same DDA, none
cross-checked. 200 > 86.1 so it is safe, but the numpy mirror being bit-exact
with C is presented as evidence that the transcription is right, and that
evidence rests on a bound nothing verifies.

**H. The KDA "config" is restated in at least five places with no shared source.**
`kda/cost.py:17` `B,T,H,K,V,C = 4,4096,32,128,128,64`; `kda/perf_kda.py:90`
`2,512,4,4,64,64`; `kda/replay_kda.py:33` the same; `kda/materialisation.py:81,85,88`
three separate dicts; `kda/codegen_diff.py:77` `1,64,1,1,64,64,64`. `cost.py`'s
docstring says "the shapes in ref.py", but `ref.py` takes them from the caller
and has no default. This is documentation-drift risk rather than a hard failure,
and it has already bitten once: `notes/04:154` had to correct a shape quoted
from the brief. `kda/codegen_diff.py:83` also computes `theirs = ... * 16 * K`
with `16` and `4` hardcoded next to a `C = 64` it does not use.

**I. `probes/real_gather.py:107-111` asserts two undefined behaviours are equal.**
Three of the six test cases use out-of-range indices (`idx[...] = src_n + 7`) and
require `custom_kernel` to match stock `Tensor.__getitem__` on them. Neither is
specified to do anything. It passed on the T4, but it is a flake by construction
and the script's success criterion is the literal string `ALLPASS` in stdout.

### Paths that only ever run on GPU (task 4)

The brief's premise — "`../../.venv/bin/python` will not exist inside a Kaggle
container" — is **true of the Makefile and false of the lane**. It is a real host
coupling and it is satisfied on purpose, at four independent levels, which is why
it has never cost a run:

1. `kernels/build_notebooks.py:199` creates the venv at `ROOT/.venv`, i.e. the
   payload root, because `probes/oc/../../.venv` resolves there — with a comment
   at `:191-194` saying exactly that ("That is not a style choice, it is what the
   Makefile reads").
2. `:230-258` handles Kaggle's missing `python3-venv` by falling back to the
   system interpreter *and* symlinking `.venv/bin/python` at it, with a comment
   naming the failure mode: a venv that died in `ensurepip` "is worse than
   nothing: make then runs that python and every probe in the camera lane dies
   with ModuleNotFoundError: numpy".
3. `kernels/local_dryrun.py:131-140` symlinks both `.venv` and `netherite/` into
   the extracted payload so the dry run exercises the same resolution.
4. `kernels/build_notebooks.py:317` clones netherite to `ROOT/netherite` and
   comments that the Makefile's `../../netherite/blaze/core` is relative to
   `probes/oc`, "i.e. at the payload root, not at the cwd".

So `.venv` is the best-engineered part of the remote lane. What *is* still
off-host-fragile:

- **`kernels/pack.py:43-46` understates what the runner now regenerates.** Its
  docstring says the excluded fixtures are "23 MB of regenerable reference data"
  and that "`make check` rebuilds all of it on the runner in well under a
  minute". 23 MB is `9 × 2.66 MB` — the *short* recipe's output. With the fix it
  is **33 files, 87.6 MB**, and the claim now covers `make report`'s 33 dumps:
  measured **2.0 s** for all 33 locally, so "well under a minute" still holds.
  The size figure is stale and is updated in `report_fix/oc-report-fix.patch`.
  (The rest of the exclusion list is right: `pack.py:79` does ship
  `probes/bench.py`, which the report imports.)
- **`remote/preflight.py:1` has `#!/usr/bin/env bash` on a Python file.** It is
  packed verbatim (`pack.py:69` copies raw bytes), and it is the first thing that
  runs on any new box. Harmless while invoked as `PY remote/preflight.py` — which
  is the only way it is invoked — but `./preflight.py` or a `bash` dispatch dies
  with a syntax error on line 1.
- **`remote/preflight.py:39` reads `/proc/meminfo`.** Linux-only, inside a
  `try`, so on macOS the RAM line is silently absent rather than reported as
  unavailable. The `find / -name 'libcuda.so*'` hint at `:103` is likewise
  Linux-shaped and is printed on a Mac.
- **`kernels/build_notebooks.py:540`** checks that each battery item's `.py` argv
  path exists, but the two camera items are shell strings (`CAMERA_C`) and get no
  such check — a payload missing `probes/oc/oc_report.py` would surface as a
  shell `make` error, not a recorded skip. `:448-452` documents the *other*
  `shell=True` trap (argv lists reduce to `$0`) and `:545-549` passes the
  strings correctly, so the surviving gap is only the existence precheck.
- **`kda/cost.py:4` and `kda/test_ref.py:4` docstrings tell the reader to run
  `/Users/huyan00/mycode/mc-cuda-tinygrad/.venv/bin/python …`.** Both files are
  in the payload. These are the only host-absolute paths in the tree
  (`grep -rn /Users/` over `*.py *.sh Makefile *.ipynb`, excluding `.venv/`,
  `netherite/` and the vendored `portability/subj/`), and they are docstrings, so
  nothing executes them — but they are the exact smell the brief describes,
  shipped to a container.

---

## 6. Which of these deserve an automated guard

Fixing these one at a time is the wrong shape. Four of the six defects share one
root cause: **a report or a recipe asserts a relationship that nothing checks.**
Three are worth automating, in this order.

**1. A fixture preflight, run before any measurement. (Highest value, trivial.)**
The report bug cost 168 s of a GPU session to produce a stack trace. A generic
"every path this run is about to open exists, with the right size" pass, taken
from a manifest the producer also reads, turns that into a 0.1 s failure with a
sentence naming the missing files. This subsumes A-of-§5, B, C, and the
`pack.py` docstring drift. It is the one guard I would insist on.

**2. A `--self-check` mode that recomputes what the report asserts. (High value.)
— PARTIALLY BUILT, in the form that turned out to matter.** The suite's
characteristic failure is prose that is not derived: the retracted FMA root cause
in `oc_report.py`, the `"Measured result: … 12.4% - 15.4%"` block in
`fp32_correctness.py`, the "37.6% of GPU time" in `real_gather.py` with no
producer, the "Metal backend" in a sentence printed on CUDA. A `--self-check` that
re-derives the headline numbers from the run and **fails when a printed summary
disagrees with the measured value** is roughly 30 lines and would have caught §4
and finding A both.

**What was built, and it is the stronger form of the same idea.** A per-probe
`--self-check` mode is a guard bolted onto each report; the guard that actually
scales is that **a number is derived in exactly one executable place and
documents cite the claim that prints it**. That is now the structure for the
division finding: `derivation/divide.py` holds every rate, the probes measure on a
device and print what they measured, and `probes/oc/report_fix/fp32_div_row.py`
is retained as the worked example of a self-check that *asserts* (its degenerate
row must equal the shipped log) rather than printing two numbers and asking the
reader to compare — the same move as
`derivation/divide_backends.py:self_division_log_crosscheck`.

**Still open on this item:** `probes/real_gather.py`'s "37.6% of GPU time" has
no producer in the tree. On the rule that follows from C-09 — *prefer deletion to
a hedged number* — it should be deleted, and it has not been, because the file is
outside this pass's scope.

**3. One shared `fixtures.py` with the seed list, the batch count and the shape
config, imported by the Makefile-adjacent code, the reports and the tests.**
`NENVS`/`SEEDS`/`OC_B` (§2a) is this for the camera lane, and it is worth doing
for the KDA shape too (§5-H). Three seed lists in one directory — Makefile,
`.gitignore`, report — is the entire root cause of the original bug. The
Makefile cannot import Python, so the honest shape is: Python owns the list, the
Makefile gets it from `$(shell …)`.

**4. A tinygrad-DEBUG=2 parser fixture. (Low value, do it when convenient.)**
One recorded `DEBUG=2` sample and a test that the five parsers extract the
expected kernel count and time from it. It cannot catch a format change before
the upgrade lands — that is what §5-F's `ZeroDivisionError` is for — but it
stops a refactor from silently halving a count.

**Not worth a guard:** the host-absolute docstrings (§5, last bullet) and the
`/proc/meminfo` no-op. Those are edits, not invariants.

**The one thing I would not automate:** the fp32 `/`-row finding A. It needs a
human to notice that two numbers in the record describe the same quantity. No
lint finds that. It needs to go into `notes/06-corrections-ledger.md` as a row,
with the re-derivation in `report_fix/fp32_div_row.py` as its evidence column —
which is the format that file already requires ("A correction with 'the audit
found it' and nothing else is not admissible here").
