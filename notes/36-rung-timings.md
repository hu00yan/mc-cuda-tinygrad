---
kind: reference
---

# 36 — the five rungs, all timed: where the wall clock leaves the instruction count

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


**The gap this fills.** `notes/23` built a five-rung ablation ladder in
`minecu_tg_opt/` — `naive`, `int_cursor`, `flat_gather`, `incremental`, `lean`,
each rung the one above plus exactly one algorithmic rewrite — and priced the
**graph operations and SASS instructions** of all five. It timed the wall clock
of **two** of them, the endpoints. The middle three were never timed, so the
performance table in `notes/23` §3 has a blank column between `naive` and `lean`
and no way to say which rewrite bought what. **This note times all five, on one
T4, in one run, and reports each rung's number.**

**The answer, first.**

1. **Above the fusion crossover the time drops in two steps, and the big one is
   the second, not the last.** At batch 1024: `naive` 8.880 ms, `int_cursor`
   6.034 ms, `flat_gather` 3.020 ms, `incremental` 2.741 ms, `lean` 2.950 ms.
   The two rewrites that remove almost no machine instructions take 1.47× and
   2.00×; the rewrite that removes the *most* SASS (`incremental`, 7.4 of the
   ladder's 20.9) takes 1.10×, and the one after it takes **0.93× — `lean` is
   7.6% slower than the rung above it.** `MEASURED`, §3–§4.
2. **The breakaway is therefore at rewrite 2, `int_cursor → flat_gather`, and
   it is worth 1.70× over what the instruction count predicts. All of the
   ladder's total excess over the instruction count sits in rewrites 1 and 2;
   rewrites 3 and 4 contribute 0.96× and 0.96× — i.e. nothing.** The SASS count
   predicts the whole ladder at 1.391× and the ladder is worth 3.011×.
   `MEASURED`, §5.
3. **This supports the direction of `notes/23`'s mechanism and refutes the chain
   it spells out.** `notes/23` §3.2 says the causal chain is *graph small enough
   to fuse → loop-carried state never leaves the register file → 3.0×*, and
   offers the launch count and the allocator's peak as the evidence. **The
   launch count does not move where the time moves**: at batch 1024 the ladder
   goes 10 → 9 → **9** → 7 → 6 CALLs, and the single largest time step
   (2.00×) is the one that removes **zero** launches. What does move there, and
   move by 4.31×, is ptxas's local-memory spill: 20,932 → 13,216 → **3,068** →
   2,984 → 3,572 bytes. `MEASURED`, §5.2.
4. **So the mechanism is register allocation, not fusion.** Above the crossover
   the local-spill figure orders the five rungs' wall clock with Pearson
   r = **+0.9986** (batch 1024) and **+0.9985** (batch 2048), and the SASS count
   orders them perfectly by rank (Spearman +1.0000) but predicts the magnitude
   2.16× short. Graph size is upstream of both; it is not itself the cost.
   `MEASURED`, §5.1.
5. **Below the crossover the ladder inverts and nothing predicts it.** At
   batches 512 and 256 `naive` 13.68/13.72 ms, `int_cursor` 13.29/13.29,
   `flat_gather` 13.37/13.40, `incremental` 15.90/16.26, `lean` 16.04/16.51 —
   the two smallest graphs are 16–20% *slower* than the naive port, and **no
   candidate currency comes out with a Spearman better than −0.67** against time
   in that regime; three of the six come out with the wrong sign outright.
   `notes/23` §3.1 reports this inversion for its two endpoints; the ladder
   shows it is a property of the two *smallest* rungs specifically, and it is
   where the ladder stops being informative. `MEASURED`, §3.
6. **`lean` is not the fastest rung.** The submitted version is 7.6% slower than
   `incremental` at batch 1024 and 8.0% slower at 2048, consistently, in both
   regimes above the crossover. `notes/23` already reported that rewrite 4 makes
   the instruction count slightly *worse* (52.0 → 53.5) and kept it "because it
   is smaller"; the wall clock now says the same thing in the currency that
   matters. `MEASURED`, §4. `PREDICTED`, §6: this was not the expected outcome
   when the ladder was built.

**Labels.** `MEASURED` — an instrument ran and the artefact is named.
`MODELLED` — arithmetic on measured shapes. `PREDICTED` — stated in advance of
the measurement and not borne out. **No number in this note is transcribed from
another note, a README, or another agent's report.** Every time figure comes
from `rung_timings.json` in this run's own output directory; every instruction,
register and spill figure comes from an instrument invoked for this note against
the sources hashed in §2.

**Runs and artefacts.**

| what | where |
|---|---|
| the T4 run (all four batches, five rungs, gate, reference) | `.agent-scratch/rungtimings/out/20260929-205341/` |
| the run's own log, cell by cell | same directory, `SUMMARY.txt`, `mc-minecu-rungs-t4.log` |
| the notebook as it ran | <https://www.kaggle.com/code/tenghuzhang/mc-minecu-rungs-t4>, kernel version 2 |
| the per-step operation and SASS counts, re-derived offline | `.agent-scratch/rungtimings/sass/opcount-b1024.json` |
| the generated CUDA and the sm_75 SASS, 5 rungs × 5 `max_steps` | `.agent-scratch/rungtimings/sass/ablate-b1024/` (the CUDA) and `…/sass-b1024/` (`.sass` + `.ptxas`) |
| the tables and the statistics in §3–§5, recomputed from the JSON | `.agent-scratch/rungtimings/analyse.py`, `…/predict.py`, output in `…/out/analysis.txt` and `…/out/predict.txt` |
| the notebook generator and the push recipe | `.agent-scratch/rungtimings/make_notebook.py`, `…/run.sh` |
| the host-side smoke test that exercises every instrument call | `.agent-scratch/rungtimings/smoke.py`, log in `…/out/smoke.log` |
| the host-side CPU correctness gate | `.agent-scratch/rungtimings/out/verify_opt.log` — 91 checks, 91 PASS, 0 FAIL |

---

## 1. What is timed, and with what

**The ruler is imported, not reimplemented.** `minecu_tg.bench.linear_device_time_ms`
— the same function `minecu_tg_opt/battery_opt.py` calls for every timing it
reports, and therefore the same function the naive port was timed with in
`notes/23`. It walks the compiled LINEAR and sums each CALL's device-measured
duration. That is the same quantity the hand-written side's
`cudaEvent`-bracketed single launch reports: device time to produce this step's
observation, with host tracing excluded from both sides.

**Every measurement is a brand-new graph.** `Tensor.realize` skips any tensor
whose uop `has_buffer_identity()`, so a graph already realized is a **silent
no-op that reads as infinitely fast**. This project's own run history records
that defect doing exactly that — a `time_graph` helper reported a port **13×
faster than hand-written CUDA** because it timed `realize()` on an
already-realized tensor — so each measurement here builds a fresh render graph
and asserts the exact predicate `realize` itself uses. Three further guards run
alongside, and all four are recorded as booleans in the report rather than
merely asserted:

| guard | what it establishes | result |
|---|---|---|
| 1 — fresh tensor, `has_buffer_identity()` false before the ruler sees it | the graph has work to do | PASS, all 20 (batch, rung) cells |
| 2 — CALL count ≥ 1 and total > 0 after the ruler returns | the ruler found a graph | PASS, all 20 |
| 2b — the same graph's CALLs timed individually, on a **separate** fresh graph | two independent invocations of one quantity agree | **\|Δ\| ≤ 0.35%** at all 20 except `lean` at 256/512 (+2.58%, −2.32%), where `lean`'s own repeat spread is 5.07% and 0.11% respectively |
| 3 — `realize()` of a **third** fresh graph, bracketed by `perf_counter` and a device synchronize | a second clock sees real work on the same program | 0.22–0.56 s per graph, all 20 |

**Guard 2b and guard 3 are the ones that would have caught the historical
defect.** `realize()` of the graph the *ruler already walked* is a no-op and
returns in microseconds — a first version of this notebook measured it that way
and got a second clock reading 0.0 s on a program the ruler had just timed at
8.9 ms. The second clock therefore gets its own fresh graph. `MEASURED`,
`SUMMARY.txt`.

**Two more instruments agree with the ruler, by different routes.**
`minecu_tg.bench.bench_render` — the wrapper `battery_opt.py` itself calls, run
again over all five rungs at all four batches as a cross-check — agrees with
the fresh-graph loop to **within 0.8%** at 19 of the 20 cells; the one
exception is `int_cursor` at batch 256 (+6.4%), in the unfused regime where that
rung's own repeat spread is 0.18% and `bench_render`'s is one build. And the
TinyGrad-side numbers reproduce across two independent T4 sessions to within
1%: this run measures `naive` 8.880 ms at batch 1024 where `notes/23` measured
8.811 ms, and `lean` 2.950 ms where it measured 2.923 ms. `MEASURED`, §3, §7.

**Render against render, at equal batch, both sides in one run.** The
hand-written side is `refdrv.cu` linking the unmodified vendored `kernels.cu`
(sha256 `c6ed8f06…`, hash-checked inside the notebook before anything is
compiled, and `cuobjdump`-gated on the absence of every sm_80+ opcode). Every
ratio in this note divides a TinyGrad render time by a `cudaEvent` render median
for the same batch, taken in the same session. **Nothing here divides a render by
a whole step**, and the whole-step figure is in `ref_timing.json` only as
context. §7 reports how badly a ratio taken across two sessions would have been
wrong here.

---

## 2. Provenance: what code produced these numbers

The notebook carries its sources as base64 and sha256-checks each one before it
compiles anything. The hashes this run actually used:

| payload | sha256 (first 16) |
|---|---|
| `minecu_ref/src/kernels.cu` | `c6ed8f06ee64d40a` |
| `minecu_tg/kaggle/refdrv.cu` | `d3a0931d642d67bd` |
| `minecu_tg/render.py` | `fe85581f80ed7fdb` |
| `minecu_tg/bench.py` | `7ec6927c711e0eec` |
| `minecu_tg/world.py` | `6c75c0e1e9cfc105` |
| `minecu_tg/blocks.py` | `752ffaea2028475d` |
| `minecu_tg_opt/render.py` | `8f6af7329eabc37b` |
| `minecu_tg_opt/battery_opt.py` | `5359bd9d97ea3ecf` |

`minecu_tg_opt/render.py` differs from the copy the earlier run of this study
pushed. The difference is four removals of unused local variables (`B`, `wxyz`)
inside three step bodies; the graphs, the statements and the generated CUDA are
unchanged, and §4's re-derived instruction counts match that run's to the last
digit. The ladder measured here is therefore the ladder `notes/23` described.
`MEASURED`, by diffing the two files.

---

## 3. The five rungs, timed

`MEASURED` `.agent-scratch/rungtimings/out/20260929-205341/rung_timings.json`,
one Tesla T4, one session, world 16³, ground 4, 32×32, `max_steps` 32, uint8
output. `ms` is the median over independent fresh graphs (2–3 per cell, spread
in §3.2). `ref µs` is `refdrv`'s `cudaEvent` median render for the same batch in
this same session. Ratio = `tg ms × 1000 / ref µs`, i.e. how many times slower.

| batch | pixels | ref render µs | `naive` | `int_cursor` | `flat_gather` | `incremental` | `lean` |
|---|---|---|---|---|---|---|---|
| 256 | 262,144 | 77.5 | 13.720 (177.0×) | 13.293 (171.5×) | 13.402 (172.9×) | 16.258 (209.8×) | 16.508 (213.0×) |
| 512 | 524,288 | 145.8 | 13.682 (93.8×) | 13.285 (91.1×) | 13.369 (91.7×) | 15.897 (109.0×) | 16.039 (110.0×) |
| **1024** | **1,048,576** | **241.1** | **8.880 (36.8×)** | **6.034 (25.0×)** | **3.020 (12.5×)** | **2.741 (11.4×)** | **2.950 (12.2×)** |
| 2048 | 2,097,152 | 466.9 | 17.460 (37.4×) | 11.949 (25.6×) | 5.817 (12.5×) | 5.105 (10.9×) | 5.512 (11.8×) |

**Two regimes, and the crossover is between batches 512 and 1024, which is where
`notes/23` §4.2 puts it** (it locates it between 524,288 and 540,672 *pixels*,
i.e. between batch 512 and batch 528). Above the crossover the ladder is
monotone downward in cost, except `lean`; below it the two smallest graphs are
the two slowest. §4 walks both.

### 3.1 The gate, on the card, before anything was timed

`MEASURED` `obs_equivalence.json`, same run, batch 1024. `refdrv` dumps its own
settled world, its own 13-slot camera basis and its own uint8 observation; each
rung then renders that exact state on the same card.

| rung | pixels differing | | mean DDA steps/pixel on the reference's state |
|---|---|---|---|
| `naive` | 0 / 1,048,576 | BYTE-IDENTICAL | 27.334025 |
| `int_cursor` | 0 / 1,048,576 | BYTE-IDENTICAL | 27.334025 |
| `flat_gather` | 0 / 1,048,576 | BYTE-IDENTICAL | 27.334025 |
| `incremental` | 0 / 1,048,576 | BYTE-IDENTICAL | 27.334025 |
| `lean` | 0 / 1,048,576 | BYTE-IDENTICAL | 27.334025 |

Spread across the five rungs: **0 steps, 0.0000%.** So the five timings are five
measurements of one computation, established on the device where the CUDA
backend lowers the graph — `notes/23` §1.1 established it on the host, where only
the CPU backend ever runs.

**And the workload the timed graphs actually did**, measured separately on the
timing inputs (the timing world carries a ±0.048 rad yaw spread, which is not the
reference's settled cameras, so it is a different distribution and is checked
too): all five rungs **16.5384** mean DDA steps per pixel at batch 1024, spread
0. `MEASURED`, `rung_timings.json["1024"].timed_workload`. This independently
reproduces the 16.540 that `notes/23` §1.3 measured, to 0.01%, from the same
`build_world_tensors` call — a cross-run check on the workload that costs nothing.

### 3.2 Spread, launch counts, allocator peak

`MEASURED` same file. **Cold** is the first graph built at that shape; **warm**
is a later one.

| batch | rung | ms | spread over fresh graphs | CALLs cold | CALLs warm | allocator peak | guards |
|---|---|---|---|---|---|---|---|
| 1024 | `naive` | 8.880 | 0.17% | 10 | 9 | 24.0 MiB / 24 allocs | OK |
| 1024 | `int_cursor` | 6.034 | 0.46% | 9 | 9 | — see below | OK |
| 1024 | `flat_gather` | 3.020 | 0.22% | 9 | 9 | — | OK |
| 1024 | `incremental` | 2.741 | 0.13% | 7 | 7 | — | OK |
| 1024 | `lean` | 2.950 | 0.15% | 6 | 6 | — | OK |
| 2048 | `naive` | 17.460 | 0.23% | 11 | 10 | 86.0 MiB / 12 allocs | OK |
| 2048 | `int_cursor` | 11.949 | 0.18% | 10 | 10 | — | OK |
| 2048 | `flat_gather` | 5.817 | 0.12% | 10 | 10 | — | OK |
| 2048 | `incremental` | 5.105 | 0.09% | 7 | 7 | 8.0 MiB / 1 alloc | OK |
| 2048 | `lean` | 5.512 | 0.05% | 6 | 6 | — | OK |
| 512 | `naive` | 13.682 | 0.18% | 41 | 40 | 145.6 MiB / 12 allocs | OK |
| 512 | `int_cursor` | 13.285 | 0.11% | 40 | 40 | — | OK |
| 512 | `flat_gather` | 13.369 | 0.03% | 40 | 40 | — | OK |
| 512 | `incremental` | 15.897 | 0.05% | 39 | 39 | 130.0 MiB / 1 alloc | OK |
| 512 | `lean` | 16.039 | 0.11% | 38 | 38 | — | OK |
| 256 | `naive` | 13.720 | 0.38% | 41 | 40 | 134.8 MiB / 12 allocs | OK |
| 256 | `int_cursor` | 13.293 | 0.18% | 40 | 40 | — | OK |
| 256 | `flat_gather` | 13.402 | 0.42% | 40 | 40 | — | OK |
| 256 | `incremental` | 16.258 | 0.07% | 39 | 39 | 129.0 MiB / 1 alloc | OK |
| 256 | `lean` | 16.508 | 5.07% | 38 | 38 | — | OK |

**One honest reading of the blank cells, and it is not "this rung allocates
nothing".** `MemProbe` sees exactly one thing: calls to `Allocator.alloc`.
tinygrad's CUDA backend **caches device buffers**, so only the first graph built
at a given shape reaches `cudaMalloc`; every later graph at that shape is served
from buffers the process already holds, so `n_allocs` reads 0 and the probe's
live-bytes high-water mark reads 0. Five rungs in one process therefore yield
**one** clean peak per shape, on whichever rung was measured first. The
`incremental` column is non-zero at three batches because its schedule needed a
buffer size no earlier rung had put in the cache. **A per-rung allocator peak is
not separable with this instrument in a single process**, and I did not spend a
second GPU session to get it one process per rung; §6 says so again.

**And a second thing the launch column says that is worth stating.** The cold
graph carries **one more CALL than the warm one** at exactly those cells where
the first-measured rung is measured: 10 vs 9 at batch 1024, 11 vs 10 at 2048,
41 vs 40 at 512 and 256. The extra CALL is the copy out of a freshly allocated
output buffer. The *time* does not care — `naive`'s three repeats read 8.870 /
8.880 / 8.885 ms, a 0.17% spread across the cold and the warm graphs — but the
*launch count* does, so a launch count quoted without saying whether the
allocator was cold is not a well-defined number. `notes/23`'s launch counts are
cold numbers, and so are the ones in the table above. `MEASURED`.

---

## 4. The ladder, rung by rung, next to the counts

`MEASURED` the time column is §3. The count columns are **re-derived offline**
by `minecu_tg_opt/opcount.py` against
`.agent-scratch/rungtimings/sass/sass-b1024/`, batch 1024, `max_steps` 32,
sm_75, CUDA 12.9, and by the **difference method** — the same program compiled
at `max_steps` ∈ {4, 8, 16, 32, 64}, slope against trip count, r² ≥ 0.99993 on
every rung. Neither is a static count read off one kernel.

| rung | ms @1024 | × vs rung above | ms @2048 | × vs rung above | CALLs @1024 | graph ops /px/step | Δops | SASS /px/step | ΔSASS | SASS predicts | local spill B @ms32 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `naive` | 8.880 | — | 17.460 | — | 10 | 30.00 | — | 74.4 | — | — | 20,932 |
| `int_cursor` | 6.034 | **1.472×** | 11.949 | **1.461×** | 9 | 27.00 | −3.0 | 69.9 | −4.5 | 1.064× | 13,216 |
| `flat_gather` | 3.020 | **1.998×** | 5.817 | **2.054×** | 9 | 9.00 | −18.0 | 59.4 | −10.5 | 1.177× | **3,068** |
| `incremental` | 2.741 | 1.102× | 5.105 | 1.139× | 7 | 9.00 | 0.0 | 52.0 | **−7.4** | 1.143× | 2,984 |
| `lean` | 2.950 | **0.929×** | 5.512 | **0.926×** | 6 | 8.00 | −1.0 | 53.5 | **+1.5** | 0.972× | 3,572 |
| hand-written reference | 0.241 | | 0.467 | | 1 | — | | **38** | | | 0 |

End to end at batch 1024: **measured 3.011×, SASS explains 1.391×.** The gap is
2.16×, and §5 says where in the ladder it is.

**The shape, stated as one sentence: the two rewrites that remove almost no
machine instructions are worth 1.47× and 2.00×, and the rewrite that removes the
single largest block of them is worth 1.10×.** `incremental` takes 7.4 SASS out
of the ladder's 20.9 — more than any other rung — and buys the least time of
the four. That is the discriminator this note was for, and it is unambiguous.

**`lean` is a regression, twice, at both above-crossover batches.** 0.929× at
1024, 0.926× at 2048, with a repeat spread of 0.15% and 0.05% respectively. Its
own instruction count got *worse* (52.0 → 53.5) and its own local spill got
*worse* (2,984 → 3,572 B) and its wall clock got worse by the same 7%. **Three
currencies agree on rewrite 4 and the ladder kept it anyway**, which is the
defensible choice if smaller is the goal and the wrong choice if fast is the
goal.

---

## 5. Where the wall clock breaks away from the instruction count

### 5.1 Six candidates, scored the same way

`MEASURED` `predict.py`, Pearson and Spearman of each candidate against the
measured median device time at each batch. **The sign of the correlation is
itself the result**: above the crossover every cost correlates *positively* with
time, which is the direction "more cost, more time" predicts. Below it the
sign flips, which means the relationship has inverted — the cheapest rung is the
slowest — and no candidate survives that.

**Above the crossover.**

| candidate | values across the five rungs | Spearman vs time @1024 | Pearson @1024 | Pearson @2048 |
|---|---|---|---|---|
| **SASS / pixel / step** | 74.4, 69.9, 59.4, 52.0, 53.5 | **+1.0000** | +0.9458 | +0.9517 |
| **local spill bytes** | 20,932, 13,216, 3,068, 2,984, 3,572 | +0.9000 | **+0.9986** | **+0.9985** |
| generated statements / pixel / step | 30.5, 24.4, 16.3, 19.0, 17.0 | +0.6000 | +0.9786 | +0.9770 |
| graph ops / pixel / step | 30, 27, 9, 9, 8 | +0.8208 | +0.9576 | +0.9617 |
| launch count | 10, 9, 9, 7, 6 | +0.7826 | +0.6105 | +0.7467 |
| registers | 168, 168, 255, 255, 255 | −0.8660 | −0.9267 | −0.9320 |

**Below the crossover (batches 256 and 512), every one of them collapses:**

| candidate | Spearman vs time @512 | Spearman vs time @256 |
|---|---|---|
| launch count | −0.6669 | −0.6669 |
| graph ops | −0.6156 | −0.6156 |
| SASS / pixel / step | −0.6000 | −0.6000 |
| registers | +0.5774 | +0.5774 |
| local spill bytes | **−0.3000** | −0.3000 |
| generated statements | −0.2000 | −0.2000 |

Three things follow, and only the third is a surprise.

* **The SASS count orders the five rungs perfectly by rank** (Spearman +1.0000)
  above the crossover. On the *ordering* question the instruction count is not
  wrong. What it cannot do is the *magnitude*: 1.391× against a measured 3.011×.
* **The local-spill figure fits the time almost linearly** — Pearson +0.9986 at
  batch 1024 and +0.9985 at batch 2048, the highest of any candidate at both —
  on five points. The `registers` row is worth reading as its complement:
  168, 168, 255, 255, 255. The two rungs that spill most are the two that *did
  not* reach 255 registers, because ptxas hit the ceiling and gave up on the
  rest. Registers is not a cost here; its ceiling is.
* **Below the crossover nothing predicts, and spill predicts worst of all**
  (−0.30). That is not a defect in the spill figure; it is the regime where the
  spill does not exist yet as a cost, because the loop-carried state is going to
  global memory once per step anyway and the fused kernel's local frame is a
  rounding error beside 38–41 kernel launches. The spill story is a **fused-regime**
  story and this note scopes it there.

### 5.2 The step-by-step accounting, which is the actual answer

At batch 1024, per rewrite: what it was measured to be worth, what the SASS
count predicted, and the ratio of the two.

| rewrite | measured | predicted from SASS | **excess** | predicted from spill | excess |
|---|---|---|---|---|---|
| 1 — cursor cast once | 1.472× | 1.064× | **1.383×** | 1.584× | 0.929× |
| 2 — one flat index | **1.998×** | 1.177× | **1.697×** | 4.308× | 0.464× |
| 3 — maintained address | 1.102× | 1.143× | **0.964×** | 1.028× | 1.072× |
| 4 — no `found` flag | 0.929× | 0.972× | **0.956×** | 0.835× | 1.112× |
| **cumulative** | **3.011×** | 1.391× | **2.164×** | 5.860× | 0.514× |

**So: the wall clock breaks away from the instruction count at rewrite 2,
`int_cursor → flat_gather`, and it stops breaking away at rewrite 3.** Rewrites
1 and 2 contribute 1.383× and 1.697× of excess; rewrites 3 and 4 contribute
0.964× and 0.956×, i.e. they are *slightly better* than their instruction counts
predict and contribute nothing. 100% of the ladder's 2.16× of unexplained
improvement is in the first half. The same arithmetic at batch 2048 gives
1.461×, 2.054×, 1.139×, 0.926× measured and 1.373×, 1.745×, 0.997×, 0.953×
excess — the same story, with the ladder's total excess at 2.277×.

**This is one step earlier than the hypothesis this note set out to test.** The
stated discriminator was: *if the big jump is between `flat_gather` and
`incremental` the ladder says the cost is graph size; if it is earlier, `notes/23`'s
mechanism is wrong.* It is earlier — at the rewrite that takes the graph from 27
to 9 operations per pixel per step — so the cost is graph size, **but the
channel is not the one `notes/23` named.**

### 5.3 What the channel actually is

`notes/23` §3.2 states the causal chain as *graph small enough to fuse → the
loop-carried state never leaves the register file → 3.0×*, and supports it with
the launch count falling and the allocator's peak falling. **This run's launch
counts do not support it at the step where the time moves**, and the peak half
of that evidence I could not re-test at all (§7.1), so what follows is a test of
the launch-count half only. Above the crossover the ladder is:

| rung | graph ops /px/step | CALLs cold | CALLs warm | distinct programs | ms @1024 | local spill B |
|---|---|---|---|---|---|---|
| `naive` | 30 | 10 | 9 | 6 | 8.880 | 20,932 |
| `int_cursor` | 27 | 9 | 9 | 6 | 6.034 | 13,216 |
| **`flat_gather`** | **9** | **9** | **9** | **6** | **3.020** | **3,068** |
| `incremental` | 9 | 7 | 7 | 4 | 2.741 | 2,984 |
| `lean` | 8 | 6 | 6 | 3 | 2.950 | 3,572 |

(The distinct-program column is the offline NULL-device census at the same
shape, which sees the *warm* launch count 9 / 9 / 9 / 7 / 6 — consistent with
§3.2, since a NULL process has no fresh output buffer to copy out of.)

**`flat_gather` removes 18 of the ladder's 22 graph operations — two thirds of
the whole address-arithmetic rewrite — and removes zero launches and zero
distinct programs.** It is worth 2.00×. Whatever is being bought there is not
being bought with a smaller launch count, and it is not being bought with
fewer machine instructions, because it removes 10.5 SASS against the 4.5 the
*previous* rung removed for 1.47×.

What it does change, 4.31× in one step, is **ptxas's local-memory frame for the
fused kernel: 13,216 → 3,068 bytes per thread.** At `max_steps` 32 that frame is
21 KB, 13 KB, 3 KB, 3 KB, 3.6 KB across the ladder, and a CUDA local-memory
frame lives in device memory reached through the L1/L2 hierarchy, not in the
register file. That is a register-allocation cost, and `notes/23`'s own §4.3 already established the precondition for it:
**every fused generated kernel on either side of the ladder has no repeating
body** (best exact period 8, one repetition, on all five), so ptxas cannot
allocate registers one step at a time and must allocate the whole straight-line
region at once. `notes/23` attributed the residual 1.41× of *instructions against
the hand-written kernel* to that, and said the time did not account for it. The
time now accounts for it: **the whole 2.16× by which the ladder beats the
instruction count is spent between the two rungs whose frames are 13 KB and
21 KB and the three whose frames are about 3 KB.** The two quantities are not
the same and I am not equating them — this is a correlation on five points, not
a decomposition (§7) — but the ordering is identical and the magnitude is not
available from the instruction count at all.

**The corrected statement of the mechanism, in one sentence: the ladder's speed
comes from graph size because graph size sets the size of the fused
straight-line region, and a big enough one cannot be allocated out of the
register file — not because graph size changes the launch count, which it does
not.**

### 5.4 And what this says about the port

`int_cursor` alone — a three-operation change, one cast hoisted out of a 32-step
loop, no hardware knowledge involved — is worth **1.47×** on its own. The two
rewrites that "only remove index arithmetic" are together worth 2.94× of the
3.01×, and the two that look like the real algorithmic wins (maintained address,
smaller live state) together add 1.02× and then give 7% back. **The cheapest
correction to the naive port is the one that makes its own address arithmetic
stop re-deriving itself every step, and most of the remaining cost is ptxas's
register allocator rather than anything the programmer wrote.**

---

## 6. What was predicted, and what was not

* **`PREDICTED` and not borne out: that `lean` would be the fastest rung.** It
  is the slowest of the three small graphs above the crossover, at both batches,
  in both the device-time sum and the independent per-CALL walk. The ladder was
  built with `lean` as the submitted version on the argument that it is
  *smaller*; the wall clock says `incremental` is 7.1% faster at batch 1024 and
  7.4% at 2048. `notes/23` §2.3 already flagged that rewrite 4 makes the
  instruction count worse and kept it anyway, so this is a confirmation of a
  known anomaly rather than a new surprise — but the anomaly is now 7% of wall
  clock and not 1.5 instructions.
* **`PREDICTED` and not borne out: that the breakaway would be at
  `flat_gather → incremental`.** It is at `int_cursor → flat_gather`. The rung
  that removes no graph operations removes no launches, and it is worth 1.10×.
* **`PREDICTED` and borne out: that the effect would be confined above the
  crossover.** At 512 and 256 the ladder inverts and the two smallest graphs are
  16–20% *slower* than the naive port. `notes/23` §3.1 reports the inversion for
  its two endpoints; the ladder locates it exactly — the three rungs that keep
  the per-axis integer cursor span only 3.4% across both batches (13.28–13.72 ms)
  and `naive` is at one end of that span, and the inversion begins at
  `incremental`, the first rung whose per-step live state is smaller than three
  integer cursors, three clips and a `found` flag. **This run does not decompose
  why**: all five rungs are still in the 38–41 CALL regime below the crossover,
  so nothing here distinguishes "smaller state in an unfused schedule" from "any
  change to the unfused schedule", and §5.1's lower table is the honest size of
  the answer.

---

## 7. What I could not measure, and why

1. **Per-rung allocator peaks.** Only the first graph built at each shape
   reaches `cudaMalloc`, so `MemProbe` returns a real peak for one rung per batch
   and 0 for the rest (§3.2). Measuring the other four needs one process per
   (rung, batch) — twenty processes, each paying a cold NVRTC compile — and I
   judged that not worth a second T4 session for a column that is not the
   question. **The blanks are blanks, not zeros.** The schedule's own footprint
   estimate does not substitute: across the five rungs it ranges only over
   23.1–23.9 MiB (3.4% apart) because it is dominated by the world array every
   rung reads.
2. **A ratio against the earlier session's reference denominator would have been
   badly wrong, and this is worth recording as a trap.** `notes/23` §3 reports
   the hand-written render at 323.6 µs at batch 1024; this session, same binary,
   same command, same `refdrv.cu`, measures **241.1 µs** — 25% lower, and 34%
   lower when set against `notes/23`'s 27.2× ratio. The work is identical: the
   render/whole-step ratio is 0.8720 here against 0.8766 there, a 0.5%
   difference, so the two sessions traced the same rays. The card's absolute
   time for a 0.24 ms kernel moved 25–34% between sessions while its 8.9 ms
   TinyGrad kernel moved 0.8%. **The short kernel is sensitive to whatever else
   the shared free-tier card is doing; the long one is not.** Every ratio in §3
   therefore uses this run's own denominator, and a ratio of two sessions on a
   shared free T4 is not a ratio. This is the concrete reason the reference had
   to be re-measured rather than reused.
3. **No batch between 512 and 1024.** The crossover's exact location is
   `notes/23` §4.2's subject on a NULL device, not mine; on the card I ran the
   two endpoints 256/512 and 1024/2048 and nothing between. Every
   above-crossover claim in §5 rests on batches 1024 and 2048.
4. **Batches above 2048, and below 256.** One world, one resolution, one
   `max_steps`.
5. **Nothing about why the fused kernel spills, beyond the size of the frame.**
   The 21 KB → 3 KB figure is ptxas's own `local memory frame` from the sm_75
   compile, re-derived here. That it costs what it costs in *this* schedule is an
   inference from its 0.9986 correlation with wall clock across five points, not
   a profiler measurement — nothing in this run attributes time to memory traffic
   directly, and five points is a small sample for a correlation, however high.
6. **Nothing about the unfused regime.** §5.1's lower table is five points and
   a table of near-zero correlations, and the honest reading of it is "this
   measurement does not explain that regime", not "the inverse of the fused
   regime's explanation holds there".
7. **The CPU gate ran on this host, the card gate ran on the card.**
   `minecu_tg_opt/verify_opt.py` on this machine — five shapes (worlds 16/16/32/48/
   16, pitches 0/−0.2/0/0/0, `max_steps` 32/32/32/32/16) × five rungs, against
   `minecu_tg/render.py`'s own output and against an independent float64
   transcription: **91 checks, 91 PASS, 0 FAIL.** Every rung's observation is
   bit-identical to the naive port's on every shape, and every rung's per-pixel
   DDA step counts agree with float64 at **1024/1024 (100.00%)** on the three
   world-16 shapes and at **1022/1024 (99.80%)** on the world-32 and world-48
   shapes — the two pixels that differ are float32-versus-float64 at an exact
   tie, and the colours are unaffected because both sides stop at the same voxel.
   Full log in `…/out/verify_opt.log`. The card cannot run this gate at all —
   tinygrad's CPU backend needs a compiler that can target bare-metal x86-64
   (`tinygrad/runtime/support/compiler_cpu.py:21`) and the Kaggle image has none
   — so §3.1's on-card gate is what stands in for it on the device, and it is
   the stronger of the two.

---

## 8. Re-deriving every number here

No GPU and no card — the §4 and §5.1 count columns and every statistic:

```
.venv/bin/python minecu_tg_opt/ablate.py --batch 1024 --steps 4,8,16,32,64 \
    --skip-verify --outdir .agent-scratch/rungtimings/sass
# ablate.py never passes --sass, so opcount.py finds no SASS and silently emits
# no `sass` section -- and, worse, emits the register and spill fields as `null`
# rather than failing. Compile each shape and stage both halves:
for d in 0004 0008 0016 0032 0064; do
  ./minecu_tg_opt/compile_sass.sh \
      .agent-scratch/rungtimings/sass/ablate-b1024/max_steps_$d sm_75
  cp .agent-scratch/rungtimings/sass/ablate-b1024/max_steps_$d/_sass/* \
     .agent-scratch/rungtimings/sass/sass-b1024/
done
.venv/bin/python minecu_tg_opt/opcount.py \
    --captured .agent-scratch/rungtimings/sass/ablate-b1024 \
    --sass     .agent-scratch/rungtimings/sass/sass-b1024 \
    --json     .agent-scratch/rungtimings/sass/opcount-b1024.json
```

**Keep the `.ptxas` files.** The SASS slopes come from the `.sass` files and the
register and spill counts come from the `ptxas -v` report beside them, matched by
function name. Delete the `.ptxas` half and `opcount.py` still runs, still
prints the slopes, and reports `local_bytes: null` for every rung — a silently
empty column, which is the same failure shape as the timed-realize no-op this
note spent its first guard on. `…/sass-b1024/` therefore holds all three
extensions.

No GPU — the host-side gate and the instrument smoke test:

```
DEV=CPU .venv/bin/python minecu_tg_opt/verify_opt.py
.venv/bin/python .agent-scratch/rungtimings/smoke.py     # DEV=NULL; structure only
```

With a card — the whole measurement, one command, push and retry included:

```
.agent-scratch/rungtimings/run.sh --build-only    # assemble + pre-flight, no push
.agent-scratch/rungtimings/run.sh                 # push, poll, download, print
```

From a run directory, the tables and the statistics:

```
.venv/bin/python .agent-scratch/rungtimings/analyse.py \
    .agent-scratch/rungtimings/out/20260929-205341 \
    --opcount .agent-scratch/rungtimings/sass/opcount-b1024.json \
    --census  .agent-scratch/rungtimings/sass/ablate-b1024/census.json
.venv/bin/python .agent-scratch/rungtimings/predict.py \
    .agent-scratch/rungtimings/out/20260929-205341 \
    .agent-scratch/rungtimings/sass/opcount-b1024.json
```

`smoke.py` runs the notebook's own instrumentation under `DEV=NULL` on a host
with no CUDA, and costs nothing but host time. **It found two defects that would
otherwise have cost a queued T4 session each**: a `NameError` in the workload
probe, and a per-repeat disagreement that turned a summary value into a list and
raised `TypeError: unsupported operand type(s) for /: 'list' and 'int'` in the
middle of the first timing cell — which, because nbclient raises on the first
cell error, took all four unmeasured batches with it. Every cell is now wrapped,
so a presentation bug cannot delete the ladder.

**Two things about `smoke.py` that are worth knowing before trusting it.** Its
output is structural, not numerical — under `DEV=NULL` nothing executes and the
ruler returns 1 µs per CALL, so every "ms" it prints is meaningless by
construction and only the *shape* of the result is being checked. And its
per-batch budget (`time_batch(8, budget_s=2400)`) is a real budget: on a slow
host it can skip the tail of the ladder and report `None` for the last rungs,
which is the mechanism working and not a failure — but a `None` there means
"skipped", and the run it came from said so.
