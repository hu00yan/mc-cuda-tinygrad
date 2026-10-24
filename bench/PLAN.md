# PLAN — run order, decision procedure, stopping rule

Companion to [`HYPOTHESES.md`](HYPOTHESES.md). That file says what is claimed; this one
says what is run, in what order, and what each result *forces*. Written before any GPU is
touched. If the two disagree, the disagreement is a bug in one of them.

**Design principle: cheapest measurement that could kill a hypothesis runs first.** Not the
most important experiment — the cheapest *killer*. A hypothesis that survives a 10-minute
laptop run does not need a GPU hour, and a GPU hour spent on it is a GPU hour stolen from
a hypothesis that has no cheap test.

---

## 1. The order, and why

| run | what | device | GPU-min | kills | cost class |
|---|---|---|---|---|---|
| **R0** | static source audit of tinygrad 0.14.0 | **laptop, no GPU** | 0 | H10, H11 (source half), gates H1/H9/H14 | free |
| **R1** | baseline-construction gate | T4, P100 | 20 | **H14** (deleted — decides if the study is comparative at all) | cheap, blocking |
| **R2** | `divide_precision.py` on CUDA | T4, P100 | 6 | **H9**, H10 (device half) | cheapest real GPU run |
| **R3** | device calibration: L2 size, achievable BW, achievable fp32, achievable fp16-TC, **and the A/A noise floor** | T4, P100 | 40 | gates **H4**, H13; sets `r_noise` for *everything* | cheap, blocking |
| **R4** | `gather_floor.py` source-size sweep | T4, P100 | 20 | H8 | cheap |
| **R5** | `real_gather.py::test_gather_matches_stock` on CUDA | T4, P100 | 10 | H7 | cheap |
| **R6** | KDA kernel count vs `C` (no timing) | laptop METAL or CPU | 0 | **H3** | free, device-independent |
| **R7** | KDA gram codegen capture, `TC_OPT` sweep | T4 | 15 | H1 | cheap |
| **R8** | KDA fp32 GPU time + Nsight counters | T4, (P100) | 60 | H2, H4, **H13** | expensive |
| **R9** | KDA plan/schedule cost | T4 | 30 | H5 | medium |
| **R10** | camera port: kernel count, step histogram, counters | T4, P100 | 40 | H6, H12 | medium |
| **R11** | full comparison | T4 (+A100 if rented) | hours | nothing new | **only if R1 passed** |

**Total to a complete answer on the free cards: ≈ 3.5 GPU-hours.** R11 is unbounded and
conditional.

The three properties of this order worth stating explicitly:

- **R0 and R6 cost zero GPU time and gate the most.** R0 determines whether the KDA arm
  is even meaningful (H11: no tensor cores for fp32 or bf16 on either free card). R6
  determines the shape of the kernel-count story. Running either of these after R8 wastes
  R8.
- **R1 is blocking and comes second.** The study's headline ratio is *deleted* (HYPOTHESES
  §6). R1 decides whether a comparative result is obtainable at all. Finding that out
  after eight GPU-hours of timings is the expensive version of finding out first.
- **R3 produces the noise floor, which gates every claim in the study.** `r_noise` from
  A/A pairs is measured *before* any A/B pair is run, and written down. See §5.

---

## 2. Run-by-run: what, how long, what it forces

### R0 — Static source audit (laptop, 0 GPU-minutes)

**What.** Read, and record verbatim with file:line:
1. `get_cuda(arch)` and the `cuda_sm75` / `cuda_sm80` / `cuda_81616` / `cuda_8168_tf32`
   tables in `codegen/opt/tc.py`.
2. `USE_TC`, `TC_OPT`, `TC_SELECT` defaults in `helpers.py`; the `TC_OPT` docstring and the
   `len(...) == 1` gate in `codegen/opt/heuristic.py:27-28`.
3. The `__truediv__` rewrite in `mixin/elementwise.py:251` and `Ops.RECIPROCAL` in
   `renderer/cstyle.py:137` and `renderer/ptx.py:15`.
4. The NVRTC compile options in `runtime/support/compiler_cuda.py` — confirm no
   `--use_fast_math` / `--prec-div=false`.
5. `Tensor.gather` in `mixin/op.py:1026-1044`; the `Ops.INDEX` render at
   `renderer/cstyle.py:50`; the coalescer assertion at `codegen/late/coalesce.py:156`.
6. `mixin/op.py` line 1044 (the one-hot) versus the 1-D fancy-index path in `__getitem__`.
7. Whether `runtime/` contains an `ops_tpu.py` (it does not).

Then, **still with no GPU**: instantiate `CUDARenderer` directly with `arch="sm_75"` and
`arch="sm_60"` and render `Tensor(a).realize() / Tensor(b).realize()` for a small tensor.
`CUDARenderer` is a pure renderer; NVRTC compilation happens server-side on Kaggle. This
yields the actual CUDA C++ for `a/b` on both cards, at zero GPU cost.

**Time:** 30–60 minutes, one sitting.

**Decisions forced.**
- Does the generated CUDA source contain `*(1/val)` and not a real `/`? → **supports** H10
  and the source half of H9. If it contains a real `/`, **H10 and H9 are both refuted
  before a GPU is used**, and `notes/02`'s "no knob" claim is wrong for 0.14.0.
- Does `get_cuda("sm_60")` return `[]` and does `cuda_sm75` lack bf16 and tf32? → confirms
  **H11**, and forces one of two study scopes, chosen now:
  - **(A) Free-hardware scope.** KDA is reported as fp32 CUDA-core only. Headline claims
    restricted to H2, H3, H4, H5, H13. No tensor-core claim. No bf16 claim. **This is the
    recommended scope** — it is honest, it is fully answerable, and it is where the study's
    real contribution is (the intermediate is (i); the planning cost is (iii)).
  - **(B) Rent an A100-class card** for the tensor-core arm, scope (A) plus H1 with a real
    MMA. Cost: money and scheduling. Decide before R7, not after.
- Is there an `ops_tpu.py`? → confirms **U5**. The TPU arm is dropped from this plan and
  left to `kda/tpu/shapes.py`, which already pre-registers S1/S2/S3 with an S3 control.
  Do not spend Kaggle TPU quota on TinyGrad hypotheses.

### R1 — Baseline-construction gate (T4 + P100, ~20 min)

**What.** Before any timing, answer one question: **can a hand-written baseline for this
exact operator be built, run, and validated on this exact card?**
1. Import Triton. Run a `tl.dot` smoke test. Record: works / fails / which error.
2. If Triton works: attempt `fla/ops/kda/chunk_intra.py`'s structure (BC=16 sub-chunks,
   `tl.dot` on `[16,K] × [K,16]`, `exp2` on an operand) on sm_75 and sm_60.
3. If it does not: **stop and re-baseline** (see below).

**Decisions forced.** This is the gate on the whole study's framing.
- Triton fails on sm_60 (expected — Triton requires CC ≥ 7.0): **the P100 is dropped from
  the KDA arm entirely.** It stays available for H2/H4/H13 (bandwidth and regime are
  dtype-independent) but carries no comparative claim.
- Triton fails on sm_75: **no comparative claim is possible on the free cards.** The study
  switches to scope (A) + roofline fractions + self-deltas, and says so in the abstract.
  This is the default expectation and it should be assumed, not hoped against.
- Triton works on sm_75: a hand-written sm_75 KDA kernel must be written **from
  `kda/SPEC.md`, by the same author, in the same session, with the same effort budget as
  the TinyGrad port**, and validated to ≤ 1e-5 against `kda/ref.py` before a single timing
  is taken. Only then does HYPOTHESES §6's deleted ratio come back, as an explicitly
  re-registered hypothesis with a *stated* baseline ("one author's sm_75 schedule", which
  is a **floor on the gap, not an estimate of it**). **An un-tuned baseline is not
  admissible** — see §6.

### R2 — `divide_precision.py` on CUDA (T4 + P100, ~6 min)

**What.** The existing probe, unchanged, on `Device.DEFAULT` = CUDA. Plus: dump the
generated PTX (`nvdisasm`) for the divide kernel and grep for `div.approx`, `rcp.approx`,
`div.rn`, `rcp.rn`. Plus: re-run it against a hand-written `1.0f/b` in CUDA C via
`custom_kernel` if that compiles, to separate TinyGrad's rewrite from NVRTC's.

**Decisions forced. RESOLVED — the second branch is the one that fired, on the
T4 (`out/kaggle/20260928-014248-gpu`). H9's own arithmetic was retracted separately
(ledger **C-13**); the branches below are kept because they are the pre-registered
decision tree and the tree is right even where H9's derivation was not.**
- `a/a` and `a/b` both essentially exact → **H9 supported.** The Metal finding is
  architecture-specific; the study's bit-exactness narrative is Metal-only and must be
  re-scoped. **DID NOT HAPPEN.** Neither was essentially exact on any backend measured.
- ~27% / ~15% wrong, or SASS shows `rcp.approx` → **H9 refuted**, and it is a headline:
  the defect is general, and the fix is a `cstyle`/API change upstream, worth reporting.
  **THIS IS WHAT HAPPENED:** CUDA 54,270 / 200,000 wrong (27.13%), `a/a` 30,748
  (15.37%), and the result is **bit-identical to `a*fl32(1/b)`** — so the first
  clause fired and the `rcp.approx` clause did not. The fix is a **Tensor-API**
  change, not a `cstyle` one: `mixin/elementwise.py:251`. And the mechanism is
  **double rounding**, not approximation — a reciprocal 1 ULP worse predicts ~86%,
  not ~27% (`derivation/divide.py:reciprocal_sensitivity`).
- SASS shows `div.rn.f32` but the result is still wrong → **H9 refuted in its causal half
  and supported in its observational half**: something after `div.rn` is wrong. Follow-up
  required; do not report either half alone. **DID NOT HAPPEN in this form** — the
  result *is* `a*fl32(1/b)`, so the wrongness is before `div.rn` ever applies, and
  there is no "something after `div.rn`" to follow up.
- **In every branch**, the camera's residual is still the sin-LUT quantiser, not the
  divide. Report them separately, always. A correct divide does not make
  `obs_camera` bit-exact, because a 9.587e-05 bucket amplifies a 0-ULP difference to
  nothing and a 1-ULP difference to 1e-4 — but a *zero*-ULP implementation still has to
  match clang's constant folding, which is the untested third requirement in `notes/01`
  Finding 1.

### R3 — Device calibration, and the noise floor (T4 + P100, ~40 min)

**What.**
1. `cudaDeviceProp`: name, CC, SM count, L2 (`l2CacheSize`), clock rates, total memory.
   **L2 size is a required output** — HYPOTHESES H8 states its prediction in terms of it,
   and that prediction must be written down against the measured value before the sweep.
2. **Achievable** HBM bandwidth: a large triad / copy, ≥ 1 GB, reported as GB/s and as a
   fraction of datasheet. Three sizes to expose the knee.
3. **Achievable** fp32 FMA throughput: a register-resident FMA loop. Fraction of 8.1 /
   18.7 TFLOP/s.
4. **Achievable** fp16 tensor-core throughput: a square GEMM at 1024, 2048, 4096, in fp16
   in and fp32 accumulate. Fraction of 65 / 125 TFLOP/s. This is the number that decides
   H1's *magnitude* — and the gap between it and datasheet is the answer to "can this card
   do tensor cores at all in practice", which matters more than the datasheet number.
5. **The noise floor.** 30 A/A pairs: the same graph, the same device, the same process,
   two labels, timed independently and interleaved. Report the distribution of
   `r = t_A / t_A'`, and set **`r_noise` = the 95th percentile of |log r|**, converted to a
   ratio. Write the number down. Everything downstream is gated on it.

**Decisions forced.**
- If `achievable_BW / datasheet_BW < 0.5` on either card: the free cards are not
  bandwidth machines and every byte-denominated prediction in HYPOTHESES must be
  re-derived against the measured value. Do this before R8.
- If `r_noise ≥ 1.10`: **no difference smaller than 10% is reportable on this hardware**,
  and that is a headline finding of its own, not a footnote. Under this condition R8's H2
  prediction (13×) survives comfortably, H1's predicted effect (**1.00–1.05×**) is
  definitionally unresolvable — so H1 must be decided on the *generated source*, never on
  time, which is why R7 is a source check — and H4's residual 2.4× folded-vs-ported margin
  is marginal. The plan says all of that now rather than discovering it.
- If `r_noise` is not stable across the two cards, the device with the smaller `r_noise`
  becomes the primary and the other is reported as a portability check only.

### R4 — `gather_floor.py` sweep (T4 + P100, ~20 min)

**What.** The existing probe unchanged: fixed 64 outputs, source grown 65,536 → 16,777,216;
then the many-from-small contrast. Add two source sizes bracketing each card's measured L2.

**Decisions forced.** H8's crossover prediction (0.5–1.0× L2) is confirmed or refuted.
A refutation is informative — it means the crossover is a compile-time constant, and that
is a *cleaner* mechanism to write up. Either way, H6's per-step cost model gets a
card-specific number, which is required before H6 can be tested at all in R10.

### R5 — `real_gather.py::test_gather_matches_stock` on CUDA (T4 + P100, ~10 min)

**What.** The existing 6-case test, plus the coalescer error message captured verbatim,
plus one extra case at the camera's real shape (1,328,176 source, 73,728 indices) to see
whether the second reduce-shaped kernel appears on CUDA at the same threshold.

**Decisions forced.** H7 supported (6/6 fail, same message) → the gather's cost on
NVIDIA is the same structural cost as on Metal, and `notes/01` Finding 3 transfers.
H7 refuted (6/6 pass) → **H6's whole mechanism is in doubt** and R10 must be re-planned
before running, because the camera's per-step kernel count would then be a backend
difference rather than a language limitation. **R5 before R10, always.**

### R6 — KDA kernel count vs `C` (laptop, 0 GPU-minutes)

**What.** `kda/perf_kda.py`'s kernel counter, at `C ∈ {16, 32, 64}`, `T=512` held fixed so
`NT = 32, 16, 8`. Kernel count only. No timing — kernel count is
architecture-independent, and the laptop's METAL backend is fine for it. Use
`rnd(..., l2n=True)`; un-normalised `q`/`k` make the graph produce NaN, which has already
happened once (`notes/DECISIONS.md` row 16) and which does not change the kernel count but
does change the honesty of the number.

**Decisions forced.** H3 supported → the O(C²) story holds and `C` is a lever.
H3 refuted → the kernel count is driven by something else and H5's fix (plan caching) is
aimed at the wrong cost.

### R7 — KDA gram codegen, `TC_OPT` sweep (T4, ~15 min)

**What.** `kda/codegen_diff.py`'s capture, extended:
- `TC_OPT ∈ {0, 1, 2}`, `TC ∈ {1, 2}`, `TC_SELECT ∈ {-1, 0, 1}`, and `BEAM=1`.
- For each: does the generated CUDA source contain `wmma` or `mma.sync`? Does the SASS?
- Report the **kernel count and generated-source size** for each setting, not just time.
  A setting that adds a WMMA and 3× the kernels is not a win.

**Decisions forced.** H1 supported → the search space contains the answer and the cost
model declines it. That is a **(ii)** and the fix is a better eligibility test: a small,
well-scoped, upstreamable change, and the most valuable single finding this study could
produce. H1 refuted → the matcher does not recognise the operand shape at all: **(iii)**,
a different patch, different owner.
Either way: if **no** setting produces an MMA, and H11's source audit also says no shape
exists, then H1 is confirmed as **(iv)** on this card and the arm is dead here — go to
scope (A) or rent an A100 (R0's decision B).

### R8 — KDA fp32 GPU time + counters (T4, then P100 if useful, ~60 min)

**What.** The one expensive run.
1. **Correctness first.** The port must pass `kda/test_tiny.py` (6/6, ≤ 1e-5) at the exact
   shape being timed, with `l2n=True`. No timing before the correctness gate. 2607.04454's
   methodological point: every kernel in their suite passed strict numerical validation
   while running 323× slow.
2. `l2n=True` asserted at run time; assert the output is finite and not NaN. Report `|S|max`.
3. GPU time, from Nsight Compute (not `DEBUG=2`, which is a device-side timer and is
   itself a variable).
4. Counters: `dram__bytes.sum`, `dram__throughput.avg.pct_of_peak_sustained_elapsed`,
   `sm__throughput.avg.pct_of_peak_sustained_elapsed`,
   `sm__warps_active.avg.pct_of_peak_sustained_active`,
   `l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum` and `..._st.sum` (the spill signature),
   `launch__grid_size`, `launch__block_size`, and the achieved register count.
5. Repeat under a `custom_kernel` fold of the decay's `exp` into the contracting operand,
   if and only if that compiles. This is the self-delta that makes H2 attributable: same
   author, same session, one change.

**Decisions forced.**
- Measured `dram__bytes.sum` in **[233, 349] MB** (i.e. within 20% of the predicted 290.7)
  → H2 supported, the 13× is real, and the exp-folding pass is the study's contribution.
- **< 58 MB** (< 20% of predicted) → H2 refuted, and the cause is L2 residency: the fix
  becomes tiling, not exp-folding. **This is the most consequential branch in the plan.**
  See `HYPOTHESES` H2.
- `dram__throughput < 40%` and `sm__throughput < 40%` → H13 refuted, the gap is
  latency/occupancy/launch, and the RC label changes.
- Local-memory sectors > 1% of total → H13 refuted *toward* RC3 (spill), and
  `NOTES.md`'s "no spills" assumption needs checking on this arch.
- Folded run **< 1.2×** the unfolded run → the exp-folding pass is not worth writing, and
  that is a clean negative result worth publishing.

### R9 — KDA plan/schedule cost (T4, ~30 min)

**What.** Time four cells, each in its own labelled column, **never compared across
columns** (see §4):

| cell | what is timed | what it is for |
|---|---|---|
| `COLD` | first call, fresh process, empty JIT cache | the cost a user pays once |
| `WARM-REBUILD` | rebuild the graph inside the timed region, warm process | the current API's steady state |
| `WARM-ASSIGN` | same realized graph, inputs changed with `Tensor.assign`, re-`realize()` | the ceiling a plan cache could reach |
| `GPU` | device time only, Nsight | the floor |

`WARM-ASSIGN` is the number that decides H5. If `WARM-ASSIGN ≈ GPU`, the whole 85% is
the rebuild+reschedule path and a plan cache removes it. If `WARM-ASSIGN` is still >> GPU,
the cost is downstream of scheduling and the fix moves.

**Decisions forced.** H5 supported → "separate planning from execution" is the right
recommendation, with a measured multiplier (pre-declared: 5.6× at the floor, 97× at the
ceiling). H5 refuted → the recommendation in `notes/03-kda.md` §4 is aimed at the wrong
pass; find the real one with a `cProfile` of `realize()` and say so.

### R10 — Camera on NVIDIA (T4 + P100, ~40 min)

**What.**
1. Per-step kernel count, decomposed by UOp type, exactly as `notes/01` §3 does on Metal.
   The per-step count is the measurement; the throughput is downstream of it.
2. Per-ray step histogram on the **T4 workload** (H12), not the M1 Pro one. Mean, max,
   and the fraction of the 96-step unroll that is dead. Also assert the analytic bound
   `L·√3 + 3 = 86.1` still holds for the T4's region.
3. Correctness against the C reference: block-id mismatches, depth, edge. Expect 0 / ~9
   / ~3 as on Metal, but **on CUDA the residual may be different** because the divide
   behaves differently (R2). Report what happens; do not carry the Metal numbers across.
4. Nsight counters: `dram__bytes.sum` per frame, `smsp__inst_executed.sum` (instruction
   count, for the dead-work fraction), launch counts.

**Decisions forced.** H6 supported (14 kernels/step reproduces) → the camera's NVIDIA gap
is a language-level fact and transfers. H6 refuted (≤ 2/step) → it is a backend difference
and `notes/01`'s framing must be narrowed. H12 confirmed/refuted by the histogram mean
against 48.

### R11 — Full comparison (conditional, hours)

Only entered if R1 passed. See §6 for the conditions on what counts as an admissible
baseline, and note the standing restriction: **one card at a time, and the result is a
statement about one cell of a grid the field has declared infeasible**
(2607.04454 §VI-D: a correct, hand-optimized kernel collapsed 580× on a second GPU with
no correctness signal). Say so every time.

---

## 3. The stopping rule

**Further measurement is pointless when any one of these becomes true. Each is a
sufficient condition; check all of them at every run boundary, and record which fired.**

1. **The baseline gate failed (R1).** Triton does not run on the free cards, so no
   comparative claim is obtainable. **Stop the comparative arm.** Everything in scope (A)
   — H2, H3, H4, H5, H12, H13 — is still answerable, and answer it. Do not substitute a
   weaker baseline and continue.
2. **The dtype/arch gate failed (R0).** No tensor-core path exists for KDA's dtype on
   either card (H11 confirmed). **Stop the tensor-core arm**, permanently, on this
   hardware. Do not spend more GPU time looking for MMA variants; the table is the table.
   The remaining question is money: rent an A100, or accept scope (A).
3. **The noise floor is above the largest predicted effect (R3).** If
   `r_noise ≥ 1.10` and the largest surviving predicted magnitude is below `r_noise`,
   **stop timing and switch to counters only.** Counters (`dram__bytes.sum`,
   `local_op_ld/st`, `l2 hit rate`) are far less noisy than wall clock and answer H2, H8
   and H13 directly. Wall-clock A/B on a shared free GPU is the least reliable measurement
   in the study and it should be dropped first, not last.
4. **H2 is refuted (R8).** The decay tensor is not reaching DRAM. **Stop the KDA
   bandwidth line of enquiry.** The remaining KDA questions (graph size, planning cost,
   regime) are unaffected and cheaper; the traffic question is answered.
5. **The correctness gate fails at the target shape (R8).** `Dl`/`Dq_all` at
   `B=4,T=4096,H=32,K=128` is `4 × 64 × 32 × 64² × 128 = 4,294,967,296` elements per
   tensor = 8.59 GB in bf16, and the two of them write-then-read is **34.4 GB** — against
   16 GB of T4 memory, a **2.1×** overshoot before anything else is allocated. The port
   **cannot run at the target shape on this hardware** (`notes/03-kda.md` §7). If the
   rewrite of `notes/03-kda.md` §4 is not implemented, **stop trying to run the target
   shape** and report the small-shape result with the extrapolation clearly labelled as
   extrapolation. Do not shrink the shape until it fits and then call it the target shape.
6. **The arm is answered.** Every hypothesis in §4 of `HYPOTHESES.md` that is testable on
   this hardware has a verdict, and the remaining ones are all in §5 (unrefutable here).
   **Stop.** Do not keep collecting counters "in case something turns up" — that is how a
   pile of numbers becomes a story.

**Explicitly not a stopping rule:** "the result is surprising", "one more device would
help", "we could also measure X". Surprise is not a reason to measure more; if the result
is surprising, the response is to re-read the harness, not to add runs.

---

## 4. Confounders, and how each is controlled

### 4.1 The cold/warm trap — the one the brief names

`Tensor.realize()` on an **already-realized** graph is a **silent no-op**
(`README.md`, Conventions). Getting this wrong does not produce a wrong answer, it
produces a **missing** one: a time of a few nanoseconds, or a kernel count of 0, which
reads as an infinite speedup. It is dangerous precisely because it does not error.

**Rules, in force from the first line of the harness:**

- **R1.1 — Build inside the timed region by default.** Every timed cell constructs its
  graph inside the timed region. Exception: only if the thing under test *is* the plan
  (R9's `WARM-ASSIGN`), and then only with the `PLAN_CACHE` flag set and the output
  labelled `REPLAY` in the log line, never `TIMED`.
- **R1.2 — Never compare across columns.** The four R9 cells are four different
  quantities. They appear in one table with a `cell` column and are never differenced
  against each other. `COLD` vs `WARM-REBUILD` is the one comparison that *is* legal, and
  only when **both** arms are cold-process runs, each in its own fresh subprocess, N = 5
  subprocesses each, medians compared.
- **R1.3 — A no-op is visible, not silent.** The harness prints, per cell: kernels
  launched, UOps in the realized graph, and a boolean `realized`. `kernels == 0` on a
  non-trivial graph is a **hard error**, not a fast measurement.
- **R1.4 — Change the inputs.** `WARM-ASSIGN` must write new values into the input
  buffers (via `Tensor.assign`) and then re-realize. If the output does not change, the
  replay is not happening and the cell is void. `kda/replay_kda.py` already does this
  correctly (it uses a different seed and asserts the output differs) — copy that
  discipline.
- **R1.5 — One process per configuration, documented env.** Record tinygrad version,
  `arch`, the renderer chosen out of `[CUDARenderer, PTXRenderer, NVCCRenderer]`, the
  JIT/compile cache directory state, and every `getenv`-visible flag that was set. A
  process that inherits `TC_OPT=1` from the R7 sweep is not a default TinyGrad.
- **R1.6 — `MOCKGPU` must be unset, and `Device.DEFAULT` must be asserted.** tinygrad has
  a mock-GPU mode that runs the whole pipeline and produces timings that mean nothing.
  Assert `Device.DEFAULT` in the expected set and assert the generated source is
  non-empty at run time.

### 4.2 Everything else

| # | confounder | control |
|---|---|---|
| C1 | **Free-GPU clock/power state.** Kaggle does not guarantee locked clocks, persistence mode, or an idle GPU. Two arms run minutes apart are not two samples. | All comparisons **interleaved within one process**, A/B/A/B, arm order **randomised by a seed that is written to the log**. Report the **paired ratio** `r = t_A/t_B`, never two independent means. No cross-process timing comparison anywhere except R1.2's explicit cold/cold case. |
| C2 | **L2 residency depends on what ran before.** The first arm in a pair gets a cold cache; the second gets the first arm's leftovers. | Interleaving (C1) handles the order. Additionally: a fixed ≥ 3-iteration warmup that is discarded, and a fixed cooldown (a large dummy allocation touched between cells) whose parameters are declared before the run. |
| C3 | **dtype mismatch between arms.** The port is fp32; any hand-written baseline would be bf16/fp16. | Either both arms in the same dtype, or the dtype is stated in the table header and the comparison is explicitly labelled as confounded. **Never** a fp32-vs-bf16 ratio. |
| C4 | **The host is a different, slower machine.** 9.01 s of scheduler time was measured on the laptop. Kaggle's CPU is different and shared. | Host-side numbers (H5, R9) are reported **with the host recorded** and are **never compared across machines**. H5's target multiplier (5.6×/97×) is a *ratio*, which is more portable than a time — state the ratio as the finding and the absolute time as environment. |
| C5 | **Correctness admits 300× slowdowns.** 2607.04454's whole point. Every kernel in their suite passed. | The correctness gate precedes every timing (§R8.1). A cell that is fast but wrong is not a cell. And a cell that is *correct* is not thereby good — which is why every headline here is a roofline fraction or a self-delta, never "it produced the right answer". |
| C6 | **The host-side graph is arch-dependent, the timings are not.** Kernel *count* is claimed device-independent in R6. | Verified by running R6 on two backends (METAL and CPU) and requiring the counts to agree. If they do not, R6's prediction is void and the count is re-measured per backend. |
| C7 | **Tinygrad's own version moves.** All source facts (S1–S7) are from 0.14.0. | The version is recorded per run and the §1 table is re-verified if it changes. A finding that depended on 0.14.0's specific rewrite is reported as version-indexed. |
| C8 | **The camera residual depends on the divide, and the divide is card-dependent (H9).** | R2 runs before R10. R10's correctness numbers are reported per card and **not** carried over from Metal. Expect them to differ; report the difference rather than explaining it away. |
| C9 | **Register/shared-memory budgets differ per arch.** 2607.04454 §VI-D: A100-tuned, correct, 582% of PyTorch; on GH200 it spilled and ran at 1.0%, with no correctness signal. | Any register-pressure or occupancy claim is reported **with the arch's `regs/thread` and achieved occupancy**, and never generalised across cards. H13 is pre-registered per card for exactly this reason. |
| C10 | **L2 size is used in H8's prediction but is not yet known.** | R3 step 1 measures it and the prediction is re-stated against the measured value **before** R4 runs. Writing the prediction after the sweep is forbidden. |

---

## 5. What counts as a difference — declared now, before any data

**The problem this rules out:** you run the sweep, find a p-value, then choose the test or
the threshold that makes the story work. So: one statistic, one threshold, one test, all
fixed here.

### 5.1 The primary statistic

**The paired ratio `r = t_A / t_B`**, computed within one process, interleaved, over
**N ≥ 30 pairs**, with pair order randomised by a seed recorded in the log. Median `r` is
reported, never the mean (timing distributions are right-skewed). This is a *within-process*
quantity, so C1 (clock state) and C4 (host identity) are differenced out by construction.

### 5.2 The noise floor — the gate

`r_noise` is measured in **R3 step 5** as the 95th percentile of `|log r|` over 30 A/A
pairs, and **written into the study's methods section before any A/B pair is run.**

> **A difference is claimed if and only if `median(r) > r_noise`.**

- `r ≤ r_noise` → the result is reported as **"not resolved"**. Not as "no difference".
  A null does not confirm a hypothesis and does not appear in a confirmation table.
- If `r_noise ≥ 1.10`, that is itself reported as a finding about the hardware, in the
  abstract, because it bounds what the study can say.
- **No hypothesis whose predicted magnitude is below `r_noise` is tested at all.** It is
  reported as untestable on this hardware. This is decided from `r_noise` and the
  prediction table alone, before the measurement.

### 5.3 The one statistical test, named now

Where a p-value is legitimate at all — N ≥ 30 pairs, paired data, and only for comparisons
that already cleared the `r_noise` gate:

- **Test:** two-sided **Wilcoxon signed-rank** on `log r`, **α = 0.05**, null `r = 1`.
- **Applied to:** every primary comparison, or to none. There is no second test, and no
  test is selected after seeing the data.
- **Reported with:** `median r`, a 95% CI by the distribution-free order-statistic method,
  and the p-value — **the p-value never appears without `r` and its CI next to it, and
  never appears as the sole support for a claim.**
- **Multiplicity: no correction is applied, and no test is used to select among the 13
  hypotheses.** A p-value here has exactly one job: to attach a number to a comparison that
  already passed a magnitude gate. Choosing which hypothesis to report based on which one
  got `p < 0.05` is **forbidden**, and the study will report all 13 verdicts including the
  nulls and the untestable ones.
- If the harness's own noise estimate disagrees with `r_noise` by more than 2×, the run is
  discarded and repeated once; a second disagreement is reported as an instrumentation
  problem, not smoothed over.

### 5.4 Counter-based attributions

2607.04454 attributes causes by **mitigation delta**, with a counter signature each. So:

> A cause is attributed to a kernel **only if the corresponding counter moves by ≥ 2×**
> under the pre-specified mitigation for that cause, and the mitigation's code is written
> and committed **before** the counter is read.

Mapping, fixed now:

| counter | ≥ 2× move under | attributed to |
|---|---|---|
| `dram__bytes.sum` | exp-folding the decay into the operand | **RC1 / (i)** — bytes, i.e. the materialised intermediate |
| `dram__throughput` + `lts__t_sector_hit_rate` | same | **RC2b / (ii)** — L2 residency |
| `local_op_ld.sum` / `local_op_st.sum` | none of the above | **RC3** — register spill |
| `sm__warps_active` with no move in the other three | — | occupancy/latency, reported as "unattributed" and not given an RC label |
| `wmma` present, bytes unchanged | `TC_OPT` sweep | **RC2a / (ii)** — search |

**Anything that does not clear a 2× move is reported as "not attributed".** A counter that
moved by 15% is noise and is written down as noise.

### 5.5 Arithmetic claims

Any predicted magnitude in `HYPOTHESES.md` is published **with its arithmetic**, as in
§4 above. A prediction that turns out wrong is reported with the same visibility as one
that turns out right, and §7 of `HYPOTHESES.md` requires distinguishing "wrong about the
world" from "wrong about the arithmetic" — a prediction falsified by a bug in the harness
is a harness bug, and saying so is mandatory, because the alternative is quietly banking a
refutation.

---

## 6. The known limitation, stated as a limitation

**A fair NVIDIA-vs-NVIDIA comparison on an old card is very likely a comparison against a
different, older, hand-written kernel, because the modern ones do not compile there at
all.** This is not a caveat to be minimised; it is the reason HYPOTHESES §6 deletes the
study's headline hypothesis.

**The mechanism, from the sources already in this repo:**
- `get_cuda("sm_60")` returns `[]` (S4). The P100 has **no** tensor-core path in tinygrad
  for **any** dtype. fla's KDA kernels are built on `m16n8k16` bf16 MMA. There is no
  version of that comparison that runs on a P100.
- sm_75 has exactly one shape, `m8n16k8` f16. fla's kernels use BC=16 sub-chunking,
  `exp2` in the operand load, and `SOLVE_TRIL_DOT_PRECISION`/tf32 — and **tf32 does not
  exist on sm_75** (`cuda_8168_tf32` enters at sm_80). So even on a T4, the modern kernel
  cannot be the modern kernel.
- Triton requires CC ≥ 7.0, so sm_60 is out for the entire Triton corpus.

**So the honest statement of what a T4 comparison is:** *TinyGrad 0.14.0 versus an sm_75
kernel written from the same SPEC.md by the same author in the same session.* That is
(a) a **floor** on the gap, not an estimate of it, and (b) a statement about one author's
schedule, not about the state of the art. 2607.04454's framing is the one to borrow: "the
genuine residuals survive best-effort optimization because cuBLAS, cuDNN and `torch.argmax`
are **themselves well-tuned**" — a residual measured against a human is a floor, not a
defect. KernelBench's 1.43× → 0.88× under a realistic baseline is the same lesson in one
number pair.

**Three pre-declared consequences, to prevent the comparison from being quietly laundered:**

1. **An un-tuned baseline is inadmissible.** The hand-written side must be tuned to the
   same standard as the compiler side: at minimum a sweep over the knobs the architecture
   actually has (block sizes, pipeline depth, L2 swizzle), validated to ≤ 1e-5 against
   `kda/ref.py`. A single un-tuned schedule makes the ratio a measurement of the author's
   afternoon. The sweep is part of the cost, or the comparison does not happen.
2. **If no baseline is built, no comparative claim is made.** The study reports roofline
   fractions (H4, H13) and self-deltas (H2's folding, H5's plan cache, H12's compaction,
   H7's plugin) and says explicitly: "this measures the compiler against the hardware, and
   deliberately makes no claim about the compiler against a human, because on this
   hardware no human's kernel exists to compare against." That is a publishable result and
   it is the one the free hardware can actually support.
3. **Every number is labelled by cell.** `(tinygrad, fp32 CUDA cores, sm_75, T4, tinygrad
   0.14.0)`, not "TinyGrad". 2607.04454 §VI-D is the cautionary tale: a correct,
   hand-optimized kernel ran 582% of PyTorch on A100 and **1.0%** on GH200, with no
   correctness signal to warn you. A single-GPU, single-shape, single-dtype result is one
   cell of a grid the field has declared infeasible, and saying so is not hedging, it is
   the finding.

**What this means for where to spend money**, which is the actionable output of the whole
pre-registration:

| if you want to claim | you must rent | cost class |
|---|---|---|
| anything about tensor cores for KDA | A100-class (sm_80+), for bf16 MMA and tf32 | money |
| anything comparative (compiler vs human) | sm_80+ **and** a hand-written sm_80 kernel tuned to the same standard | money + a week |
| the intermediate is the whole story (H2) | **nothing** — T4 is enough | free |
| the planning cost is real and fixable (H5) | **nothing** — host-bound, card barely matters | free |
| graph size vs graph depth (H3) | **nothing** — device-independent | free |
| which RC label owns the gap (H13) | T4 is enough; a second card is a portability check | free |
| bit-exactness against netherite (H9) | **nothing on NVIDIA** — and note the answer is likely "NVIDIA is fine, Metal is not" | free |

That table is the deliverable. The free hardware is genuinely good for five of the
thirteen hypotheses, and the two that need money are the two the study should be explicit
about not having answered.
