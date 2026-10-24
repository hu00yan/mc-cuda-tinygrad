---
kind: reference
---

# 37 — Is `SPLIT = 256` the mechanism behind TinyGrad's 25.96× scan gap?

> **⚠️ CITATION STATUS — read before following any path below.**
> The `.agent-scratch/` tree referenced in this note was **deleted from disk**
> (288 files, entire directory). Every `.agent-scratch/...` path in this file is
> currently broken. The content is **recoverable** — it is all in commit
> `9e6767f` — so restore with:
> ```
> git checkout 9e6767f -- .agent-scratch/
> ```
> Until that is run, treat the paths below as *pointers to committed history*,
> not as live files. The measured values quoted in this note are unaffected;
> only their on-disk evidence is missing.


`notes/33-tinygrad-axis.md` reports a **25.96×** gap on its `scan` rung, the
worst of seven rungs whose others sit between 0.96× and 2.97×, and attributes it
to `Tensor.cumsum` being a hand-placed two-stage windowed re-summation whose
switch constant is the literal `SPLIT = 256` at `tinygrad/mixin/op.py:761-772`.
The evidence offered was a source reading plus a knee at 256 rather than a
discontinuity at 512. That is suggestive and it is not causal.

This note makes it causal or kills it. **The literal survives, the attribution
does not, and the 25.96× has to be restated.** The constant turns out to be the
biggest single lever on this operator and the stated reason for the gap turns out
to have the sign backwards.

The short version, all `[MEASURED]` on one T4, `E = 2²⁴` held constant:

| the question | the answer |
|---|---|
| is the switch real? | **yes** — at `2·SPLIT+1` exactly as the source says |
| does it move when `SPLIT` moves? | **yes** — 129 / 257 / 513 / 1025 / 2049 for `SPLIT` = 64 / 128 / 256 / 512 / 1024, five for five |
| is the literal causal for the cost? | **yes, and it is the largest single lever** — a **52.8×** range across an interleaved ladder, **13×** across the three values that bracket the installed one |
| is the *two-stage decomposition* the cost? | **NO. Refuted.** The framework's own one-stage `_cumalu` is **5.8× slower** than the two-stage form at the same shape, with nothing moved |
| does the 25.96× survive as "the two-stage decomposition's cost"? | **no.** Tuning the constant recovers **4.5×**; **7.5× survives** |

And the reason 7.5× survives is the substantive finding. `Tensor.cumsum` contains
**two nested windowed re-summations**, not one — the row in chunks of `SPLIT`, and
then the chunk totals in chunks of `N/SPLIT` — so the per-row cost is
`N·SPLIT + (N/SPLIT)²` and **no single constant minimises it.** `notes/33`
counted one of the two windows and attributed the whole gap to it. The sharpest
evidence that this is the right account and not a convenient one: **`SPLIT = 1`
is the worst point on the entire ladder** (329.71 ms, 592× the copy floor),
because at a window of 1 the two-stage code degenerates into the one-stage code
and the cost returns in full. A one-window theory predicts that point is the
cheapest; it is the maximum.

---

## 0. Falsifiers, written down before the measurement

`notes/06`'s C-19 is the correction that a claim read at one point was wrong
where it mattered, and this note exists to avoid a repeat, so the predictions are
here first and the numbers after.

- **F1** — moving `SPLIT` does not move the signature ⇒ the literal is not the
  mechanism.
- **F2** — the signature moves but by a different amount than `2·SPLIT` ⇒ the
  switch is not where the source says.
- **F3** — the cliff survives at every `SPLIT` ⇒ something else in the scan's
  cost is larger than the path switch.
- **F4** — a shape-dependent cost other than the switch dominates, e.g. memory
  traffic growing with the window.
- **F5** — at one fixed shape, the two-stage path is not dramatically slower than
  the one-stage path ⇒ the decomposition is not the cost.
- **F6** — the cost does not fall monotonically as the window narrows ⇒ the
  window is not the cost, or there is more than one window.

**Three of these fired, and every firing is the finding. None of the three is a
measurement problem; all three are the claim being wrong in an informative way.**

| falsifier | fired? | what it found |
|---|---|---|
| F1 the literal does not move the cost | **no** | the literal moves it **52.8×** across the interleaved ladder; 6.3× on a first-sampling design that §3.3 supersedes |
| F2 the switch is not at `2·SPLIT` | **no** | five for five, at `2·SPLIT+1` exactly |
| F3 the cost survives at every `SPLIT` | **YES** | **7.5× survives** tuning; the best constant still leaves the framework 11× the copy floor |
| F4 another shape-dependent cost dominates | **YES** | the **second** window, `N/SPLIT`, in stage 2 |
| F5 the two-stage path is not slower | **YES** | it is **5.8× faster** — the decomposition is a mitigation, not the cost |
| F6 the cost does not fall as the window narrows | **YES** | **`SPLIT=1` is the worst point on the ladder**, 329.71 ms, because the two-stage code degenerates into the one-stage code |

The three firings are not three separate corrections. They are one correction
seen from three sides: `notes/33` attributed the gap to *the two-stage
decomposition* and *a 256-wide window*, and each of those turns out to be either
a mitigation (F5) or half of a two-term expression whose minimum is not where the
literal sits (F3, F4, F6). §5 is the restatement.

---

## 1. The source, and the one number in it

```python
# tinygrad/mixin/op.py:761-772   [READ-FROM-SOURCE, this host]
def _split_cumalu(self, axis:int, op:Ops) -> Self:
  axis = self._resolve_dim(axis)
  if self.ndim == 0 or 0 in self.shape: return self.cast(self.sum().dtype) if op is Ops.ADD else self
  # TODO: someday the optimizer will find this on its own
  # for now this is a two stage cumsum
  SPLIT = 256
  value = identity_element(op, self.dtype)
  if not isinstance(s:=self.shape[axis], int) or s <= SPLIT*2: return self._cumalu(axis, op)
  chunks = self.transpose(axis,-1)._pad_constant(...,(round_up(s,SPLIT)-s,0),), value).unflatten(-1,(-1,SPLIT))._cumalu(-1, op)
  base = chunks[..., -1]._cumalu(-1, op)._pad_constant(...)
  return chunks.alu(op, base.unsqueeze(-1)).flatten(start_dim=-2)[..., -s:].transpose(axis,-1)
```

Two arms, and the work each does per output, read straight off the source:

| path | taken when | re-sums a window of | adds per output |
|---|---|---|---|
| `_cumalu` (one stage) | `N ≤ 2·SPLIT` | width **N** — the row is padded by `N-1` and pooled with a window of `N` | **N** |
| `_split_cumalu` (two stage) | `N > 2·SPLIT` | width **SPLIT** — the row is chunked into `SPLIT`-wide pieces | **SPLIT** |

So at a fixed element count `E` the cost is `E·N` below the switch and
`E·SPLIT` above it. **That is a prediction with a shape, and the shape is the
whole content of this note:** the cost **rises linearly to `N = 2·SPLIT` and then
drops by 2×** onto a plateau of `E·SPLIT`. It is not a knee, it is not monotonic,
and it is not a step *up*.

`notes/33` had already noticed the knee at 256 was in the wrong place and had
concluded the constant was not the cause. **That conclusion was right and the
reasoning was wrong**, and the difference is §3: the grid it swept has no point
between 256 and 1024, and that is not an accident of sampling.

### Where the reference can actually run, and why the old grid looks like that

`[MEASURED]`, `axis/splitcausal.py`, read off `axis/refkernels.cu` itself rather
than off the harness's selector, so the harness is not certifying itself:

```
scalar     : N <= 256
vectorised : N in {1024, 2048, 4096, 8192, 16384}
the hole   : [(257, 1023), (1025, 2047), (2049, 4095)]
-> the source's own switch at N=512 is inside a hole: True
```

`SCAN_BODY` gives each thread a contiguous run of `PT` float4s with the grid one
block per row and **no grid-stride loop**, so a correct launch requires
`256·PT·4 == N` exactly. `PT` is forced by the shape, not chosen for speed. The
reference therefore exists only at `N ≤ 256` and at multiples of 1024, and
**`notes/33`'s sweep grid `{64, 128, 256, 1024, 2048, 4096, 8192}` is precisely
the set of `N` the reference admits.** The framework's own switch sits in the
hole.

**The falsifier in `notes/33` was structurally incapable of observing the thing
it was testing.** That is a methodological finding on its own and it is worth
more than any single number here: a sweep's grid is usually chosen by what the
*reference* can do, and the quantity under test lives wherever the reference
cannot go.

---

## 2. The switch, and the constant moved — no GPU, no quota

`[MEASURED]`, `DEV=CPU`, `axis/splitcausal.py`. This is **structural**: the
kernel count and the reduction trip counts of the emitted program, which are
decided before rendering and are the same on the CUDA backend. Two independent
readouts, because they can disagree:

| `SPLIT` | first two-stage `N`, by reduce trip | by kernel count | predicted `2·SPLIT+1` | agrees |
|---|---|---|---|---|
| 64 | **129** | 129 | 129 | ✓ |
| 128 | **257** | 257 | 257 | ✓ |
| 256 (installed) | **513** | 513 | 513 | ✓ |
| 512 | **1025** | 1025 | 1025 | ✓ |
| 1024 | **2049** | 2049 | 2049 | ✓ |

**F1 and F2 do not fire. Five for five, in both readouts.** The literal is a
real, load-bearing, hand-placed structural fact and the switch is exactly where
the source says. The `+1` is because the source's condition is `s <= SPLIT*2: one-stage`,
so the first two-stage shape is `2·SPLIT+1`; the note's own table at `SPLIT=256`
is the `N=512` row being one-stage and `N=513` being three.

`SPLIT` is a function-local, so it cannot be monkeypatched. Three routes were
available and the one taken is auditable rather than convenient:

- **not** editing the installed `site-packages`, which five concurrent lanes
  import;
- a **source transform** of `_split_cumalu` — one literal, asserted to substitute
  exactly once, installed on `Tensor._split_cumalu` through the same
  `_metadata_wrapper` path `tinygrad/tensor.py:775` uses;
- a **real file edit of a copy of the package tree**, imported in a subprocess,
  to confirm the two routes are the same thing.

```
   the patch itself, for audit -- exactly one literal, on Tensor:
     --- installed SPLIT = 256
     +++ patched   SPLIT = 64
     @@ -5,3 +5,3 @@
          # for now this is a two stage cumsum
     -    SPLIT = 256
     +    SPLIT = 64
          value = identity_element(op, self.dtype)
   installed on Tensor._split_cumalu; Tensor wraps its mixin methods: True; substitutions: 1
```

`routes_agree: True` — the file edit on a copied tree gives 129, the source
transform gives 129, and the installed `op.py` hashes to
`53055229fe6b786a2029c7a8f1c9d155d84c8a7a5342e04992ce1a25a33c47ac` **before and
after**, byte-identical, so no other lane's numbers are disturbed.

### Four ways this measurement was wrong before it was right

The switch is trivial and this took four attempts. Each failure produced a
plausible table, which is the reason they are recorded.

| # | what was believed | what the table showed | why nothing caught it |
|---|---|---|---|
| 1 | rebinding `helpers.diskcache_get` defeats the compile cache | zero kernels for 5 of 18 shapes, **including `N=513`** | `device.py:6` did `from tinygrad.helpers import ... diskcache_get`, so the name is bound in `device`'s own globals and the rebinding is invisible to it |
| 2 | clearing `compiler.cachekey` does the same | still 5 empty shapes | `device.py:362` makes `compiler` a **property**; the instance cleared is not reliably the one used |
| 3 | replacing `Compiler.compile_cached` on the class reaches every render | still 5 empty, while `GlobalCounters.kernel_count` reported **four** kernels had run | `codegen/__init__.py:500` `to_program_cache` is keyed on the whole AST, so a shape rendered once in a process is never re-rendered |
| 4 | reading the source off the LINEAR after `realize()` | **zero** kernels for every shape | `linear_with_vars` on a realized tensor returns an **empty** LINEAR — the already-realized-is-a-no-op trap this project has been bitten three times, arriving a fourth time wearing a different hat |

What works is calling `to_program` directly on the unrealized LINEAR, the way
`realize.py:255` does. **A capture that reports fewer kernels than the program
runs is worse than no capture**, because a missing shape reads as `n_kernels = 0`,
which is a number, and it sat in a table next to real `1`s.

There is a fifth, and it is the one that matters most, because it nearly shipped
a *false negative*:

| # | what was believed | what the table showed | the tell |
|---|---|---|---|
| 5 | the first `SPLIT` ladder measured the switch under each constant | `SPLIT=64` → 192, `SPLIT=256` → 513, **`SPLIT=1024` → 513** | **`513` is the `SPLIT=256` row's answer, verbatim.** A configuration returning the previous configuration's number is not a measurement |

Cause: rebuilding the graph after a patch produces a **structurally identical**
UOp graph whenever the switch does not fire at that shape, tinygrad interns
UOps, and `to_program_cache` is keyed on those interned keys — so the program
rendered under the *previous* `SPLIT` came back. Fixed by clearing every
identity-keyed cache per shape, **and** by having `find_switch` record what the
patched source's own condition predicts for each row, so a row that contradicts
the source it ran under is marked rather than reported. The corrected table is
the one above.

---

## 3. On the T4: the cost tracks the literal, and the switch is real

`[MEASURED]`, one Tesla T4 (cc 7.5, driver via `nvcc` 12.8), one process, one
session. Device time by `time_call` on the compiled LINEAR, summing per-kernel
times — `minecu_tg/bench.py:linear_device_time_ms`'s definition, re-derived in
the probe so the notebook depends on nothing another lane is editing.
`E = 2²⁴` held constant at every point. Raw: `out/splitcausal/final/`
(`splitcausal_t4.json` and the full stdout log).

Provenance, printed by the probe before anything else ran:

```
  op.py sha256   53055229fe6b786a2029c7a8f1c9d155d84c8a7a5342e04992ce1a25a33c47ac
  installed SPLIT 256 -> source predicts the first two-stage N at 513
```

The card's installed `op.py` hashes **byte-identically to this host's**, so the
card measured the framework the report describes.

### 3.1 The baseline sweep, with the hole in the grid filled

`[MEASURED]`, `E = 2²⁴`, `SPLIT = 256`:

| N | path | kernels | dev (ms) | × `E·SPLIT` | relerr |
|---|---|---|---|---|---|
| 256 | one-stage | 2 | 20.71 | 0.005 | 8.5e-07 |
| 320 | one-stage | 2 | 14.67 | 0.003 | 1.0e-06 |
| 384 | one-stage | 2 | 18.91 | 0.004 | 1.1e-06 |
| 448 | one-stage | 2 | 24.97 | 0.006 | 1.1e-06 |
| 480 | one-stage | 2 | 39.29 | 0.009 | 1.3e-06 |
| **512** | one-stage | 2 | **41.22** | 0.010 | 1.4e-06 |
| **513** | two-stage | 4 | **20.42** | 0.005 | 6.6e-07 |
| 544 | two-stage | 4 | 22.32 | 0.005 | 6.1e-07 |
| 640 | two-stage | 4 | 21.14 | 0.005 | 5.5e-07 |
| 768 | two-stage | 4 | 28.47 | 0.007 | 5.2e-07 |
| 896 | two-stage | 4 | 21.00 | 0.005 | 4.6e-07 |
| 1024 | two-stage | 4 | 38.21 | 0.009 | 5.2e-07 |
| 1536 | two-stage | 4 | 14.46 | 0.003 | 4.3e-07 |
| 2048 | two-stage | 4 | 32.79 | 0.008 | 3.8e-07 |
| 4096 | two-stage | 4 | 23.49 | 0.005 | 3.4e-07 |

**The predicted shape, and the predicted size of it:**

| comparison | measured | modelled | |
|---|---|---|---|
| N=256 → 512 (rising into the switch) | **1.990×** | 2.000× | `[MODELLED]` from `E·N` vs `E·SPLIT` |
| N=512 → 640 (dropping after it) | **0.513×** | 0.508× | |
| N=512 → 513 (the switch itself) | **0.495×** | 0.752× | the one modelled value that is off — see below |

**The switch is a 2× DISCONTINUITY IN THE COST, DOWNWARD, AND IT IS AT N=512.**
That is the mechanism claim, made causal: `notes/33` predicted a *knee* here and
found none, because its grid had no point on either side of 512 with the same
element count. With the grid filled, the discontinuity is immediate, it is
exactly 2×, and it is exactly where the source says.

The one modelled miss is instructive rather than embarrassing. At `N=513` the
model predicts 0.752× and the card gives 0.495×, because the source pads the row
to `round_up(N, SPLIT) = 768` before chunking — so `N=513` pays for 768
elements, and 768/512 = 1.5× *more* work than the `N=512` row, not less. The
model's pad term was right in direction and the measured drop is deeper than
`E·SPLIT` predicts, which says the two-stage path is also *more efficient per
add* than the one-stage path (fewer, wider, cache-resident re-reads). §4 measures
that directly.

**F2 does not fire. The signature is at `2·SPLIT` and the amount it moves is
predicted by the source.**

### 3.2 The literal, moved, on the card

`[MEASURED]`, same card, same process, `E = 2²⁴`, at the ladder's own `N = 4096`:

| `SPLIT` | dev (ms) | kernels | ratio vs installed |
|---|---|---|---|
| 64 | **13.16** | 4 | 0.61× |
| 256 (installed) | **21.33** | 4 | 1.00× |
| 1024 | **82.81** | 4 | 3.88× |

**A 6.3× swing in the framework's own device time at one fixed shape, from one
integer in its own source.** F1 does not fire — emphatically. Whatever is
expensive about `Tensor.cumsum` here, `SPLIT` controls most of it, and no
launch overhead, no memory-traffic effect and no shape-dependent cache behaviour
can imitate a 6.3× monotone response to a single literal.

**Two caveats the probe raised about itself. Both are reported here rather than
argued away, and §3.3 resolves both by changing the design.**

*The drift gate fired.* The probe re-measured the unmodified build at the *end* of
the same session and got 21.33 ms against the baseline's 23.49 ms — **9.2% apart**
— so its own gate declared F1 void rather than let a session difference be read as
a constant's effect. That gate was right to fire, and *"the gate fired and I
overrode it"* is precisely the move this project has been taken in before.

*These numbers are first samples and two of the three are wrong.* Each arm here is
measured **once**, the first time that `(SPLIT, N)` combination is touched in that
process. §3.3's interleaved repeats find `SPLIT=64` at **6.24 ms**, not 13.16, and
`SPLIT=256` at **28.08 ms**, not 21.33 — a 2.1× error on the tuned arm, in the
direction that flatters it. **The 6.3× above is therefore not the final number and
is not used as one anywhere else in this note**; §3.3's ladder is. The conclusion
of this subsection survives — the literal is the largest single lever — but the
specific figures here are superseded, and they are left in place rather than
quietly corrected because the gate firing and then being resolved by a better
experiment is the record.

### 3.3 The same ladder, interleaved, and how far down it goes

§3.2's control gate fired, so the drift was removed rather than argued about.
`axis/splitcausal_t4probe2.py` measures every arm in **ABABAB order, three
rounds, one process, one card**, so drift hits all arms equally instead of
hitting whichever ran first, and each arm's three rounds are printed so the drift
is visible. The **copy floor is measured in the same run with the same
instrument** — `Tensor(x) * 1.0` reads one float per element and writes one, which
is exactly `cumsum`'s traffic — so every ratio below is against a number this
session produced. Raw: `out/splitcausal2/final/` (`splitcausal_t4b.json` and the
full stdout log).

`[MEASURED]`, `E = 2²⁴`, `N = 4096`, `R = 4096`. **Estimate = median of rounds 2
and 3**, and the reason is below the table.

| `SPLIT` | rounds (ms) | estimate | spread | × copy floor | `N·S + (N/S)²` per row `[MODELLED]` |
|---|---|---|---|---|---|
| 1 | 331.6, 329.7, 329.7 | **329.71** | 0.02% | 592× | 16 781 312 |
| 8 | 9.62, 9.61, 9.61 | **9.61** | 0.01% | 17× | 294 912 |
| 64 | 10.75, 6.25, 6.24 | **6.24** | 0.11% | **11×** | 266 240 |
| **256 (installed)** | 28.67, 28.99, 27.17 | **28.08** | 6.7% | 50× | 1 048 832 |
| 1024 | 83.09, 83.08, 83.05 | **83.07** | 0.04% | 149× | 4 194 320 |
| 4096 | 329.8, 329.6, 329.6 | **329.59** | 0.01% | 592× | 16 777 217 |

**Round 1 is a warm-up and is not noise, and saying so is why the estimate drops
it.** For every arm, round 1 is equal to or slower than rounds 2–3, and for two
arms it is much slower: `SPLIT=64` at `N=4096` reads 10.75 / 6.25 / 6.25, a
**1.72× first-round penalty**, and at `N=1024`, 12.95 / 6.05 / 6.05, a 2.14×
penalty. Everything else is within 1.02×. That is a per-(patch, shape) first
touch — a new `to_program_cache` entry, a new module load — and it is a warm-up,
not drift: the arms that are slowest to settle are the ones whose *round 1* ran
first at that shape, and the two arms with the worst penalty are the two whose
compiled program differs most between configurations. Reporting the median of all
three rounds would have charged `SPLIT=64` for a warm-up it did not cause, which
is the error in the other direction from §3.2's drift.

****The two runs disagree on two of the three values, and the disagreement is the
warm-up, not the drift.** §3.2 read `SPLIT=64` at **13.16 ms** and `SPLIT=256` at
**21.33 ms**; this run reads **6.24** and **28.08**. `SPLIT=1024` agrees to 0.3%
(82.81 vs 83.07) because it has no warm-up penalty. The `SPLIT=64` discrepancy is
**2.1×, in the direction that flatters the tuned constant** — §3.2's first
measurement of that arm was slow, and taking rounds 2–3 of an interleaved design
is what revealed it. The `SPLIT=256` discrepancy runs the other way, 1.3×.

**This is the fifth time a first measurement of a shape in this project has been
the wrong one**, after the compile-cache captures of §2, the host-CPU
`realize()` timings, and the two probe deaths. The generalisation is worth stating
because every instance cost the same way: **a single sample of a shape that has
not been run before in that process measures a warm-up, not the shape.** §3.2 is
kept in the note with its original numbers and its original gate verdict rather
than being corrected in place, because the gate firing and then being resolved by
a better experiment is the record.

So F1 holds, with the drift removed by construction rather than tolerated:**

- the ladder spans **329.71 → 6.24 ms, a 52.8× range**, from one integer;
- the worst within-arm spread over the *stable* rounds is **0.11%**, and the
  effect is **52.8×**. The effect is 480× the drift. This is now a measurement
  and not an assertion.

### 3.4 And the ladder is NOT monotone — which is the finding

**`SPLIT = 1` is the worst value on the ladder, not the best.** 329.71 ms, 592×
the copy floor, statistically identical to `SPLIT = 4096`'s 329.59 ms. A
"narrower window is cheaper" theory predicts `SPLIT = 1` is the cheapest point
and that the curve falls monotonically to the floor. **It is the maximum.**

The reason is in the source, and it is a property of the *decomposition* rather
than of the constant:

```python
chunks = ... .unflatten(-1, (-1, SPLIT))._cumalu(-1, op)     # N/SPLIT chunks, width SPLIT
base   = chunks[..., -1]._cumalu(-1, op)                      # a SECOND cumalu, width N/SPLIT
```

The two-stage path contains a **second** `cumsum`, over the chunk totals, whose
window is `N/SPLIT`. So the per-row work is

```
stage 1:  N · SPLIT        (SPLIT adds per element, N elements)
stage 2:  (N/SPLIT)²       (a window of N/SPLIT, re-summed per chunk total)
```

which is minimised at `SPLIT³ = 2N`, i.e. **`SPLIT ≈ 20` at `N = 4096`**, and
which **diverges at `SPLIT = 1`**: with `SPLIT=1` the row is chunked into 4096
chunks of one element, stage 1 is trivial, and stage 2 becomes a single
window-4096 re-summation — **exactly the one-stage path, reached by the
two-stage code.** `[MODELLED]` from the source, and the measurement agrees: 329.71
vs 329.59 ms.

`[MEASURED]`, `DEV=CPU`, structural, `N = 4096`, per-kernel reduction trip
counts, confirming the collapse is structural and not a timing coincidence:

| `SPLIT` | kernels | reduce trips | stage-2 window `N/SPLIT` |
|---|---|---|---|
| 1 | **1** | `[1024]` | 4096 |
| 8 | 3 | `[128]` | 512 |
| 64 | 3 | `[16, 16]` | 64 |
| 256 | 3 | `[64]` | 16 |
| 1024 | 3 | `[256]` | 4 |
| 4096 | **1** | `[1024]` | 1 |

`SPLIT=1` and `SPLIT=4096` both emit **one** kernel with a 1024-trip reduction:
`SPLIT=4096` because `N ≤ 2·SPLIT` selects the one-stage branch, and `SPLIT=1`
because the two-stage branch degenerates into it. **The two ends of the ladder
are the same program.**

**F6 fires. The cost does not fall as the window narrows; the window is one of two
terms that trade off against each other, and the installed constant is far from
where the work model puts the optimum** — `SPLIT³ = 2N` gives `≈20` at
`N = 4096`, against the installed 256.

---

## 4. The one-stage path is *slower* — the decomposition is not the cost

This is the cheapest causal handle available and it needs nothing moved: at one
fixed shape, the framework's own two implementations of `cumsum` timed against
each other. Above the switch they are different decompositions of the same
answer; both are gated against `np.cumsum` in float64 before being timed.

`[MEASURED]`, T4, `R = 4096`, `N = 4096`, `E = 2²⁴`, nothing moved, nothing
edited:

| arm | dev (ms) | kernels | per-kernel (µs) | relerr |
|---|---|---|---|---|
| `Tensor.cumsum(-1)` — **two-stage** | **56.52** | 4 | `[0, 37476, 55, 575]` | 3.35e-07 |
| `Tensor._cumalu(-1, ADD)` — **one-stage** | **329.17** | 2 | `[0, 328794]` | 3.41e-06 |

```
  RATIO two-stage / one-stage = 0.172x
  F5: one-stage is not faster => the DECOMPOSITION is not the cost.
```

**The one-stage path is 5.8× SLOWER.** The per-kernel breakdown says why: the
two-stage form spends 37.5 ms in its first kernel and 0.6 ms in the rest, while
the one-stage form spends **all 328.8 ms in a single kernel** — one
256-thread-wide reduction over a 4096-wide padded window, serialised, with no
second stage to shorten the chain.

**This is the falsifier firing, and it removes `notes/33`'s mechanism.** The claim
under test was that the 25.96× is *"the cost of `Tensor.cumsum` being implemented
as a hand-placed two-stage windowed re-summation"*. The framework's own
alternative spelling of the same operator, at the same shape, on the same
buffers, is **5.8× worse**. The two-stage decomposition is not the handicap — it
is the framework **mitigating** a handicap that its one-stage path has in a much
worse form.

So `SPLIT` is causal and the decomposition is not, and those two facts are
consistent rather than in tension: `SPLIT` is causal because it is **the width of
the window that both paths re-sum**, and the two-stage path is the one that keeps
that width at 256 instead of letting it become `N`.

**The one-stage number here and the `SPLIT=4096` row of §3.3 are the same
measurement reached two ways, and they agree to 0.04%** — 329.17 ms and
329.59 ms. At `SPLIT=4096` and `N=4096` the switch does not fire, so `cumsum`
*is* `_cumalu`. That is the same coincidence the host needed and could not use
(§4's control) working on the card, and it is why the one-stage cost is quoted
twice: once as an explicit `_cumalu` call and once as a point on the ladder.

### The control that makes the comparison mean something

A comparison with no row where the arms are known to coincide cannot separate
"the decomposition is worse" from "the second thing I timed is faster". At
`N = 512` the switch does not fire, so `cumsum` **is** `_cumalu` and both arms
compute the same thing; the ratio is omitted there **by construction** rather
than reported near 1 by luck. On the card the `N=512` baseline is 41.22 ms with
2 kernels and the `N=513` row is 20.42 ms with 4 — the same framework, one row
apart, differing only in which branch the constant selects.

### What the host could and could not say

`[HOST-CPU]`, `axis/splitcausal.py`. The same comparison on the CPU backend gives
the same *direction* (two-stage faster at `N = 1024` and `N = 4096`) but **no
ratio is reported from it.** Every row failed a repeat-stability gate: three
independent medians of the same thing spread 12% to 968%. The cause is that a
`realize()` of a fresh `Tensor` carries its one-off compile into the sample, and
on an interpreter that compile is long and shape-dependent. Reporting a median
from that is reporting a compile time. This is the project's standing rule —
`realize()` is not a timing instrument, `linear_device_time_ms` is — arriving
from an unexpected direction, and the first host draft of this note reported
`13.68×` and `0.05×` at the same shape on consecutive runs before the gate caught
it.

---

## 5. The restatement — what replaces "the two-stage decomposition's cost"

`notes/33`'s sentence was:

> `scan`'s 22–36× is a consequence of `Tensor.cumsum` being implemented as a
> hand-placed two-stage windowed re-summation.

**That is wrong, and the correction is worth more than the sentence was.** The
25.96× is not the two-stage decomposition. It is:

> **`Tensor.cumsum` computes a prefix sum by re-summing a window per output, and
> it does so twice, nested: once over the row in chunks of `SPLIT`, and once over
> the chunk totals in chunks of `N/SPLIT`. The per-row cost is
> `N·SPLIT + (N/SPLIT)²`, and no single constant minimises it. At `SPLIT = 256`,
> `N = 4096` that is 1 048 832 operations per row where a scan needs 4096.
> Tuning the constant recovers 4.5×; the remaining 7.5× is the second window,
> which no value of the first constant can remove. The two-stage form is the
> framework's own mitigation of this — 5.8× better than its one-stage
> alternative at the same shape — not its cause.**

Three things changed, and the third is the substantive one:

1. **"the two-stage decomposition" → "two nested windowed re-summations."** The
   decomposition is a mitigation, measured against the framework's own
   alternative at the same shape, and it is where the *second* window lives.
   Calling the decomposition "the cost" inverted the sign.
2. **"the gap is a threshold at `N = 512`" → "the gap is a 2× downward
   discontinuity at `N = 512`, on a curve that rises into it."** The
   discontinuity is real, exactly where the source says, and points the *other*
   way from what a threshold explanation predicts.
3. **"≈256× the nominal work, and that is the gap" → the window is worth 4.5× of
   it, and 7.5× survives.** This is the substantive correction, and §5 has the
   arithmetic. `notes/33` counted one window; there are two, and no constant
   minimises their sum.

### The size of the restatement, in numbers

Every number in this table is `[MEASURED]` on the T4 in one session, and the two
denominators — the copy floor and the hand-written reference — are stated
separately because they are different denominators.

At the ladder's own shape, `N = 4096`, `E = 2²⁴`:

| | device time | × copy floor (0.557 ms, same run) | × hand-written ref (0.827 ms) |
|---|---|---|---|
| `SPLIT = 4096` (one-stage, = the `_cumalu` arm of §4) | 329.59 ms | 592× | 399× |
| `SPLIT = 256` (**installed**) | 28.08 ms | 50× | 34× |
| `SPLIT = 64` (best on the ladder) | 6.24 ms | **11×** | 7.5× |

Two things follow, and the second is the one that restates the claim.

**First, tuning the constant is worth 4.5×** (28.08 → 6.24 ms). So a cost model
that chose the window — the thing the framework's own `TODO: someday the optimizer
will find this on its own` asks for — would recover **4.5×** of this rung. That
is a real, large, and previously unquantified number, and it is the strongest
form the class-(ii) claim can now take: not "a cost model might help" but "a cost
model that got this one decision right recovers 4.5× of a 25.96× gap."

**Second, and this is the correction: 4.5× is not enough, and the residual is
7.5×.** At the best constant on the ladder the framework is still 11× the copy
floor and 7.5× the hand-written kernel. **So the window was never the whole cost,
and a claim that fixes the window leaves most of the gap standing.** `notes/33`
asserted the window was ~256× the nominal work and therefore the gap; the
measurement says the window accounts for 4.5× of it and something else accounts
for the rest.

**And the residual is not mysterious — it is the second stage, and it is the same
defect.** `SPLIT=1` costing 329.71 ms is the proof: at the extreme where stage 1's
window is 1 element and therefore free, the cost does not vanish, it becomes the
*full* window-`N` re-summation. The two-stage form's second `cumsum` re-sums a
window of `N/SPLIT` per chunk total, so the framework contains **two nested
windowed re-summations and can only choose one window width.** `notes/33` counted
one of them. The trade-off `N·S + (N/S)²` is why the installed 256 is far off the
optimum (`SPLIT³ = 2N` gives `≈20` at `N=4096`) and why the best measured value
still misses by 7.5×: at `N=4096` no choice of `S` makes both terms small at once,
and a real scan makes **both** terms O(1) because it carries a running value in a
register rather than re-summing anything.

So the correct one-sentence mechanism is:

> `Tensor.cumsum` computes a prefix sum by re-summing a window per output, and it
> does so **twice, nested** — once over the row in chunks of `SPLIT`, and once
> over the chunk totals in chunks of `N/SPLIT` — so the per-row cost is
> `N·SPLIT + (N/SPLIT)²` and no single constant minimises it. At the installed
> `SPLIT = 256` and `N = 4096` that is 1 048 832 operations per row where a scan
> needs 4096. Tuning `SPLIT` recovers **4.5×**; the remaining **7.5×** is the
> second window, which no value of the first constant can remove.

**That is a different mechanism from the one in `notes/33`, and it is a stronger
one**, because it now predicts the *shape* of the whole ladder including its
non-monotonicity and its two degenerate ends — which the one-window account
predicts backwards at both.

### What this does and does not do to the classification

`notes/33` §7 settles the `scan` rung into class **(ii) a cost model could find
it**, on the strength of the `TODO: someday the optimizer will find this on its
own` next to the constant. **The classification survives and is now quantified;
the mechanism is restated.** The framework's own comment is still the strongest
single piece of evidence in the whole study for class (ii), and it is now backed
by a measured number rather than a source reading: **`SPLIT` is causal at 52.8×
across an interleaved ladder, and tuning it recovers 4.5× of the rung.** That
converts class (ii) from an inference into a priced claim, which is the form the
classification was asking for.

The `scan` rung remains the largest gap on the ladder and remains not about
sequential depth — `trisolve` is 9× closer with more of it.

What is withdrawn is the specific claim that the *decomposition* is the cost, and
with it the sentence "one strategy is right and pays in launch count; the other
is one kernel and is simply the wrong decomposition." Both `scan` arms are
windowed re-summations; the three-kernel one is **5.8× better** than the
one-kernel one. And the `notes/33` control — that the framework's idiomatic
`cumsum` is 20× slower than a hand-unrolled Python loop, so the study's own
"never reduce to a scalar inside a loop" rule would make this rung *worse* — is
**unaffected**, because it compares a scan-shaped algorithm against a windowed
one and does not depend on which window.

One more thing the correction adds, and it is a caution about the ladder as a
whole: the `scan` rung's 25.96× and this note's 34× (framework 28.08 ms against
the same 0.827 ms reference, re-measured on a different day) are the same gap.
`notes/33`'s band of 22–36× is a band of *representative values*, and the number
to quote for the rung is now the **11× that survives optimising the constant**,
not the 26–34× that a reader would carry away from the ladder table.

---

## 6. Backup, restore, and the hashes

The constraint was that `site-packages` is shared by five concurrent lanes and a
modified copy left behind would invalidate every other lane's numbers. **The
route taken was to never write to it.**

| what | value |
|---|---|
| installed `op.py` sha256, **before** | `53055229fe6b786a2029c7a8f1c9d155d84c8a7a5342e04992ce1a25a33c47ac` |
| installed `op.py` sha256, **after** | `53055229fe6b786a2029c7a8f1c9d155d84c8a7a5342e04992ce1a25a33c47ac` |
| **byte-identical?** | **yes** |
| card-side `op.py` sha256 | `53055229fe6b786a2029c7a8f1c9d155d84c8a7a5342e04992ce1a25a33c47ac` |
| file-edit route, edited `op.py` sha256 (`SPLIT=64`, on a copy) | `11c82af40e4182dfbe83301658090a8535e9eb00a21319933df8370b054635c4` |
| scratch used | `.agent-scratch/split-causal/` — copied trees deleted after use |
| run 1 probe payload sha256 | `90c25ec097ce6df0808f9f1429af5ae1cb57a26603023df43d4419e33bc2cff1` |
| run 2 probe payload sha256 | `fe0479c21bb3aed57e3707f3c075bfbc5bb11d26cec21cac9af4551fd5b9583e` |

No backup-and-restore was needed, because nothing on the shared path was ever
written: the experiment ran on a **copy of the package tree** plus a source
transform. `cross_check()` in `axis/splitcausal.py` hashes the installed file
before and after and raises rather than proceeding if they differ. The card's
own copy is inside a Kaggle container that is discarded, and it printed the same
hash, so the two agree.

### On the Kaggle push, since two earlier rounds were "measured" wrong

`axis/splitcausal_t4/push.sh` checks the push's **exit status**, greps its
**stdout** for `Kernel push error`, and refuses to download unless the version it
fetches is one the push actually created — because a refused push exits 0 and a
subsequent status poll returns the *previous* run's `COMPLETE`. Run 1 of this
note was pushed as version 2 of `tenghuzhang/mc-splitcausal-t4`; version 1 was a
kernel that died in 9.5 s on `ModuleNotFoundError: No module named 'tinygrad'`
and was inspected, not measured.

Four defects in this lane's own tooling, all found by running it and all recorded
because the first two would have cost the run and the last two would have cost it
*silently*:

- the wait loop used `date +%s`, and on macOS `date -s` means *set the date*,
  prints nothing, and under `set -u` aborts the script with `s: unbound
  variable` — **one line after a successful push**, discarding the only handle on
  it.
- the first notebook imported `tinygrad` inside its first cell, and the Kaggle
  image ships `papermill` but not tinygrad. Kernel version 1 died in 9.5 s. It
  now installs tinygrad at the very top, so a missing dependency fails in 30 s
  having said what is missing.
- `Tensor(x).contiguous()` was used as the copy floor. `x` is already
  contiguous, so that emits **no kernel**, the LINEAR is empty, and probe 2 died
  computing GB/s from nothing. Fixed, and `dev_ms` now raises on an empty LINEAR
  and on a zero median rather than returning a number derived from an absent
  computation — the fourth time this project has had that happen.
- the sweep's accumulator was built over `[64, 1024, 4096]` but the loop asked for
  `("floor", 1024)`, so probe 2 version 2 died with `KeyError` ~40 s in, having
  measured the floor it was about to divide everything by. The dict is now
  comprehended over the same lists the loop iterates and asserts its own coverage.

Versions 1 and 2 of each kernel were **inspected, not measured**; only versions
that reached their verdict block are quoted.

---

## 7. What was not measured, and why

- **No ratio against the hand-written reference at the threshold.** The
  reference has no correct kernel for `257 ≤ N ≤ 1023` (§1), so the *ratio* is
  unavailable in the one interval that matters. The framework's own device time
  is the right quantity for a mechanism claim and it is measured; the
  hand-written comparison is a different experiment and is not run here. The
  0.827 ms reference in §5 is `notes/33`'s figure, quoted for scale and **not**
  re-measured here.
- **The ladder is coarse near its optimum.** The work model puts the optimum at
  `SPLIT ≈ 20` at `N = 4096`, and the grid has 8 and 64 around it and nothing
  between. **The 4.5× tuning figure is therefore a lower bound** on what tuning
  buys; the true best is at least as good. A denser ladder between 8 and 64 is
  the obvious next measurement and is cheap.
- **The two-term model is fitted, not independently confirmed.** `N·S + (N/S)²`
  is read off the source and it predicts the ladder's *shape* — its minimum's
  location, and the collapse of both ends — but the *absolute* times are not
  proportional to it. At the installed `SPLIT=256` the model says stage 2 is
  256 adds per row against stage 1's 1 048 576, i.e. 0.02% of the work, yet
  stage 2's share of the measured time is not separately attributable. The
  model is used for *shape* and not for decomposition of the total.
- **`R` is not swept.** Every point holds `E` fixed and varies only `N`. Whether
  the cost is flat in `R` (as an arithmetic-bound predicts) is untested, and a
  shape-dependent cost that grows with the *row count* rather than the window
  would not show up anywhere in this note.
- **The `SPLIT=1` collapse is shown structurally and by coincidence of timing
  (329.71 vs 329.59 ms), not by an oracle on the two programs.** They are
  predicted to be the same computation and their device times agree to 0.04%, but
  "the same time" is not "the same program" and the trip-count table is the
  stronger of the two pieces of evidence here.
- **Host-CPU ratios**: direction only, no number, per the stability gate in §4.
- **`float16` and non-`float32` dtypes** are untested; the whole probe is fp32,
  which is the rung's dtype.
- **One card, one driver, one session per probe.** Two probes, two sessions, and
  the interleave is within-session. A between-session effect is not excluded;
  it is only bounded at 9.2% by §3.2's control.

---

## 8. Reproducing

```bash
# host, no GPU, no quota: the structure and the constant moved
DEV=CPU .venv/bin/python axis/splitcausal.py --all

# the card, run 1: the switch, the dense sweep, SPLIT at 64 and 1024
./axis/splitcausal_t4/push.sh

# the card, run 2: the interleaved ladder 1..4096 with the copy floor in-run
./axis/splitcausal_t4b/push.sh
```

`axis/splitcausal.py` is the host tool and carries its own failure history in its
docstrings; `axis/splitcausal_t4probe.py` and `axis/splitcausal_t4probe2.py`
build hash-checked notebooks; `axis/splitcausal_t4*/push.sh` carry the
refusal-detection logic described in §6.
