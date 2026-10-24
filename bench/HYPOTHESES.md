# HYPOTHESES — pre-registered, before any GPU run

**Status: frozen. Written on a laptop with no GPU in the loop. Nothing below is a
measurement.** Every number that is not already in the repo is either (a) arithmetic
from shapes that are already in the repo, or (b) an explicitly-labelled source-reading
of tinygrad 0.14.0, listed in §1. The GPU runs that would test these are in
[`PLAN.md`](PLAN.md). If a hypothesis here disagrees with what the hardware says, the
hypothesis is wrong and gets amended in the decision log — not softened.

**Why this file exists.** 2607.04454's central methodological result is that *correctness
admits ~300× slowdowns*: an idiomatic kernel passes the correctness gate and runs 323×
slower than the baseline. Every port in this repo passes its correctness gate
(0/41,472 block-id mismatches; KDA at 6.6e-06). Passing a correctness gate is therefore
not evidence of anything, and a pile of counters with no prior is how a 300× becomes an
unexamined result. So: the prior comes first.

**What is *not* in this file.** The 12,090-kernel count, the 256× decay-intermediate
ratio, the 85%-scheduler split, the 0/41,472 camera match, the 27.13% METAL division
error rate, the 30,748/200,000 `a/a` failures, the gather crossover table — all of that
is already measured and recorded in `notes/`. It is not restated here as new. What *is*
here is: what those results do **not** license, and what has never been measured on any
NVIDIA device.

---

## 0. Classification tags

Aligned to `research/TAXONOMY.md` §"Tag definitions". Tags (i)–(iii) are that file's,
verbatim in meaning; **(iv) is an extension**, added because two of the hypotheses below
are about hardware for which the compiler has no table entry at all, which is a
different failure from any of (i)–(iii) and must not be filed under them.

| tag | meaning | fix |
|---|---|---|
| **(i)** | CANNOT EXPRESS — no construct in the language/IR denotes the decision | a port, or accept it |
| **(ii)** | CAN EXPRESS, DOES NOT FIND — reachable in principle; the search, cost model, analysis or budget declines | better search / analysis / budget. **No port.** |
| **(iii)** | NOT YET IMPLEMENTED — specified or already present one layer down, absent from the shipping path | write the pass |
| **(iv)** | DOES NOT EXIST FOR THIS ARCHITECTURE — the compiler's tables have no entry for this arch/dtype | **not a compiler defect. Buy different hardware or change the dtype.** |

(iv) is the tag that matters most for the hardware this study is about, and it is the one
the study is most likely to mis-report. Almost every interesting KDA result is bf16-flavoured,
and neither free card has a bf16 tensor core in TinyGrad's table. See **H11**.

---

## 1. Source facts this file is built on

Read from the installed tinygrad 0.14.0 (`pip`-installed in `.venv`), not measured. Each is
a *precondition* whose GPU confirmation is a planned run.

| # | fact | where | tag |
|---|---|---|---|
| S1 | `a / b` is rewritten to `a * reciprocal(b)` **at the Tensor API**, before any backend is chosen | `mixin/elementwise.py:251` — `d = a.alu(Ops.MUL, b.reciprocal())` | — |
| S2 | `Ops.RECIPROCAL` renders as `(1/x)` in the C-style family, so CUDA C++ gets `a*(1.0f/b)` | `renderer/cstyle.py:137` | — |
| S3 | tinygrad's CUDA compile is **NVRTC with `--gpu-architecture=sm_XX` only** — no `--use_fast_math`, no `--prec-div=false` | `runtime/support/compiler_cuda.py` | — |
| S4 | `get_cuda(arch)` returns `[]` for sm_60 (P100); `cuda_sm75` (T4) is exactly one shape, `m8n16k8` f16→f32 and f16→f16. **No bf16, no tf32 on either.** | `codegen/opt/tc.py` | **(iv)** |
| S5 | tensor cores are *attempted by default* (`USE_TC=1`) but the heuristic fires only for a single reduce axis with a direct `Ops.LOAD → Ops.MUL`; `TC_OPT` defaults to 0 (2 under BEAM) | `codegen/opt/heuristic.py:27-28`, `helpers.py:238` | (ii) |
| S6 | `Tensor.gather` is still one-hot → `where` → `sum(-1)`; `Ops.INDEX` renders a real address but the 1-D fancy-index path does not reach it | `mixin/op.py:1026-1044`, `renderer/cstyle.py:50` | (iii) |
| S7 | there is **no TPU backend** in tinygrad 0.14.0 (`runtime/` has no `ops_tpu.py`) | `ls runtime/` | (iv) |

S1+S2+S3 together say something the repo currently gets subtly wrong, and it is worth
stating because it is a correction, not a new result: `notes/02-obs-camera-residuals.md`
concludes "tinygrad's cstyle renderer emits `a*(1/b)`, not Metal's divide… CPU is exact,
so this is a property of the cstyle path." In 0.14.0 the rewrite is **one layer up, in
the Tensor API** (S1), so the CPU backend's exactness is a *coincidence of how clang
lowers `1.0f/x` in C*, not evidence that the IR is exact. The distinction matters: a
renderer-level bug is per-backend, an API-level rewrite is universal and the backends
differ only in whether they approximate the reciprocal. **H10** is the falsifiable version.

---

## 2. Hardware facts used for arithmetic

**Datasheet peaks, not measured.** Run 0 and Run 3 in `PLAN.md` replace every row of this
table with the device's own `cudaDeviceProp` values and a measured achievable ridge. If a
measured ridge is below the datasheet ridge by more than 2×, the regime calls in §3 flip
and the whole plan is re-derived before any A/B is run.

| device | cc | fp32 (CUDA cores) | fp16 tensor | mem BW | ridge fp32 | ridge fp16-TC |
|---|---|---|---|---|---|---|
| Kaggle T4 (TU104) | sm_75 | 8.1 TFLOP/s | 65 TFLOP/s | 320 GB/s | **25.3 FLOP/B** | **203 FLOP/B** |
| Kaggle P100 (GP100) | sm_60 | 18.7 TFLOP/s | 125 TFLOP/s | 732 GB/s | **25.5 FLOP/B** | **171 FLOP/B** |
| (for contrast) H100 SXM | sm_90 | — | 989.5 TFLOP/s | 3.35 TB/s | — | 295 FLOP/B |

The two fp32 ridges being within 1% of each other is a coincidence, and it is a useful
one: **it means the T4 and the P100 are interchangeable for the fp32 KDA regime question
and a T4-only run answers it.** The fp16-TC ridges differ by 19%, so a bf16 result is
*not* portable between the two cards and must be reported per card.

---

## 3. Index

| id | subject | operator | device | tag | killed by |
|---|---|---|---|---|---|
| **H1** | KDA gram gets no tensor-core MMA | KDA | T4 / P100 | (ii) T4, (iv) P100 | R7 |
| **H2** | the `[C,C,K]` decay really costs DRAM bytes | KDA | T4 | (i) | R8 |
| **H3** | kernel count is O(C²) from the unrolled substitution | KDA | any (count only) | (ii) | R6 |
| **H4** | fp32 KDA is memory bound, not 8× compute bound | KDA | T4 / P100 | (i)+dtype | R3, R8 |
| **H5** | 85% of KDA wall is the scheduler, and it is cacheable | KDA | T4 | (iii) | R9 |
| **H6** | the camera's 14 kernels/step is a memory-model fact, not a backend fact | camera | T4 / P100 | (i) | R10 |
| **H7** | the gather plugin fails identically on CUDA and Metal | gather | T4 / P100 | (iii) | R5 |
| **H8** | the gather's source-bound crossover is L2 capacity | gather | T4 / P100 | (i) | R4 |
| **H9** | CUDA `a/b` is double-rounded, ≈0 wrong, not 27% wrong | divide | T4 / P100 | (i) | R2 |
| **H10** | the `a*(1/b)` rewrite is in the Tensor API, not the renderer | divide | any | — (correction) | R0/R2 |
| **H11** | fp32 and bf16 KDA have no tensor-core path on the free cards | KDA | T4 / P100 | **(iv)** | R0/R7 |
| **H12** | ≥80% of the camera's 96-step unroll is dead work | camera | T4 | (ii) | R10 |
| **H13** | KDA's cost on a T4 is bandwidth, not occupancy or latency | KDA | T4 | (i) | R3/R8 |
| **H14** | ~~"TinyGrad's KDA is N× slower than fla's kernel"~~ | KDA | T4/P100 | — | **DELETED, §6** |

---

## 4. The hypotheses

Each entry: the claim stated so it could be false · where · what would refute it · the
predicted magnitude with the arithmetic · what a refutation would mean.

---

### H1 — The KDA gram gets no tensor-core MMA for a *searchable* reason, and one environment variable is the whole search space

**Claim.** In the generated code for `Z[p,j] = Σ_d decay[p,j,d]·k[p,d]·k[j,d]`, no
`wmma`/`mma.sync` appears, **and the reason is S5, not the absence of a capability**:
`OptOps.TC` is in the search space and is attempted by default, but the default
`TC_OPT=0` admits only a single reduce axis with a direct `LOAD → MUL`, and this
operand is a computed product. Setting `TC_OPT=1` (admits multiple reduce axes and
cast-buffer multiplies) on **sm_75** makes a WMMA appear in the generated source.

- **Operator / device:** KDA decayed key-key gram, isolated as in `kda/codegen_diff.py`. sm_75 (T4).
- **Tag:** **(ii)** on sm_75 — the construct is in the search space (`OptOps.TC`), the cost model / eligibility test declines it. (iv) on sm_60: `get_cuda("sm_60")` returns `[]`, so on a P100 there is nothing to find. **H11** is the (iv) half; do not double-count.
- **Predicted magnitude if true — and note this is a prediction of *nothing*.** T4 fp32
  CUDA-core peak 8.1 TFLOP/s vs fp16-TC 65 TFLOP/s = **8.0×** on the FLOP-bound portion.
  But the gram as the port writes it is not remotely FLOP-bound: it reads the
  16,777,216-element decay tensor, so its intensity is
  `524,288 FLOP / (67.1 MB decay + 2.1 MB of k) = 0.0076 FLOP/byte` against a
  **203 FLOP/byte** fp16-TC ridge — **27,000× memory bound**. So the predicted
  end-to-end effect of finding a tensor core here is **1.00–1.05×, i.e. indistinguishable
  from no change.** The 8.0× headroom is entirely unrealisable at this intensity.
- **Why that is the most useful thing in this entry.** H1 therefore makes a *sharp
  two-outcome prediction that separates searchability from usefulness*: **the WMMA will
  appear in the generated source and the measured time will not change.** Those are
  separable and they are reported separately. A study that reports "TinyGrad finds no
  MMA here, which is why it is slow" would be wrong twice over: it never looked, and
  looking would not have helped. The interesting result is that the *same* fold that
  would remove the 67 MB (H2) is a precondition for the MMA being worth anything at all —
  the two hypotheses are not independent and H2 dominates.
- **Measurement that would refute it:** capture the generated CUDA source for the isolated gram at `TC_OPT=1, TC=1` on a T4 and find **no** `wmma` / `mma.sync` token. That refutes the searchability claim and promotes H1 to (iii) — the pattern exists but the matcher never fires for this operand shape.
- **What a refutation means:** the study's "TinyGrad emits a generic `sum(-1)` where a
  human writes an MMA" framing is wrong in kind. It is not a declined search; it is a
  matcher that does not recognise a scaled-broadcast operand at all, and that is a
  different patch with a different owner.
- **Cost:** minutes, one process, no timing. R7.

---

### H2 — The 256× larger intermediate is worth ~13×, not 256×

**Claim.** At the port's shape (`B=2, T=512, H=HV=4, K=V=64, C=64`, `NT=8`, fp32 — the
shape in `kda/perf_kda.py`), the two `[B,NT,HV,C,C,K]` decay tensors are `16,777,216`
elements each and dominate DRAM traffic; and **the exp-folding rewrite is worth about
13×, not 256×**, because the other intermediates do not go away with them.

- **Operator / device:** KDA, full port, sm_75.
- **Tag:** **(i)** for the fusion itself (TAXONOMY item 6, "online / no-intermediate",
  (i) for every DSL — the firmest (i) in the corpus). The *quantification* below is not (i).
- **Arithmetic.** All figures fp32 (4 B/element), one write + one read of each
  intermediate — the same convention as `kda/cost.py`.
  - Decay tensors: `B·NT·HV·C²·K = 2·8·4·4096·64 = 16,777,216` el each = 67.1 MB each.
    Two of them, write + read: `2 × 2 × 67.1` = **268.4 MB**.
  - Everything else: `g_cum` 2.1 + `Z`/`Btri`/`M`/`Aqk` 8.4 + `w`,`u` 4.2 + `v_new` 2.1
    + I/O (read `q,k,v,g`, write `o`) 5.2 + `S` read/write 0.3 = **22.3 MB**.
  - Ported total = 268.4 + 22.3 = **290.7 MB**. Folded total = **22.3 MB**.
  - Ratio = 290.7 / 22.3 = **13.0×**.
  - Decay ÷ compulsory (`5.5 MB` = I/O + state) = **48.8×**.
  - The "256×" in the repo is the *slab* ratio `C²K / (BC·K) = 262,144 / 1,024 = 256`. It
    is correct as a statement about one `(b, chunk, head)` slab and it is **not** a
    statement about the operator: the slab is not the whole traffic budget, and 19/20 of
    the win is claimed by four tensors out of nine. This is pre-registered as a
    **correction to the study's own framing**, not a new measurement.
- **Measurement that would refute it:** `dram__bytes.sum` for one KDA call, from Nsight
  Compute, divided by the analytic compulsory 5.5 MB. If the measured total is
  **< 58 MB** (i.e. < 20% of the predicted 290.7 MB), the decay tensor is not reaching
  DRAM and the whole 13× prediction is wrong — most likely because a 67 MB tensor is
  being absorbed by L2 on that card.
- **What a refutation means:** the KDA bottleneck story is an *L2-residency* story
  (RC2b) rather than a bandwidth story, and the fix changes: it would be a tiling change
  (keep the decay in SMEM / L2 by chunking the contraction) rather than an exp-folding
  rewrite. **This is the most consequential refutation in the list**, because it changes
  which pass to write. It is also why KDA must be run in **fp32 first**: 67.1 MB is
  comfortably past any L2 on these cards, whereas the bf16 33.55 MB is not, and a bf16
  run would produce a number that looks like a refutation when it is an artefact.
- **Cost:** one Nsight profile, ~30 min. R8.

---

### H3 — The 12,090 kernels are O(C²), and that is the unrolled forward substitution

**Claim.** The kernel count is dominated by the unrolled UT transform's
`C(C−1)/2` dependent scalar-elimination steps, not by the `NT`-step scan and not by the
`[C,C,K]` elementwise+reduce pair. Therefore **kernel count is superlinear in `C`**, and
`C=16` gives at most 1/3 the kernel count of `C=64`.

- **Operator / device:** KDA, kernel count only — no timing, so it can run on the laptop's
  METAL backend and the numbers are architecture-independent.
- **Tag:** **(ii)** — a cost model that knew the substitution is a serial island with 16
  FLOP/byte intensity (`kda/NOTES.md` §2) would fuse it differently.
- **Arithmetic.** `C(C−1)/2`: at `C=64` that is **2,016** steps; at `C=16` it is **120**
  — a **16.8×** reduction. The scan goes the other way, `NT = T/C`: 8 at `C=64`, 32 at
  `C=16`, a 4× increase. Prediction: total kernel count at `C=16` lies between
  1/16 and 1/3 of the `C=64` count, i.e. **3,800–12,090** against the measured 12,090.
- **Measurement that would refute it:** kernel count at `C=16` **≥ 0.5×** the count at
  `C=64` (i.e. ≥ 6,045). That means the count is linear-or-worse in `C` from something
  else, and the substitution is not the driver.
- **What a refutation means:** the "sequential control flow is absorbed at the cost of
  planning time, which is recoverable by unrolling" story in `notes/03-kda.md` §3 needs
  its mechanism re-identified. The 85%-scheduler cost would then be attributable to
  graph *size* rather than graph *depth*, which points at a different fix (graph
  compaction, not plan caching).
- **Cost:** 10 minutes, laptop. R6. **Do this one before spending any GPU time on KDA** —
  it is the cheapest KDA experiment in the study and it is device-independent.

---

### H4 — In float32, KDA is memory bound by ~10×, not 8× — and that inverts on bf16

**Claim.** The repo's central KDA conclusion ("8× memory bound, fusion is everything,
`NOTES.md` §6") is a property of **bf16 on an H100**, not of KDA. In the dtype the port
actually uses (fp32) at the shape the port actually runs, the operator's intensity is
**0.81 FLOP/byte against a 25.3 FLOP/byte ridge on a T4**, i.e. **31× memory bound** —
and fused it is 10.5 FLOP/byte, still 2.4× memory bound. So: memory bound on both dtypes
on both cards, but the *margin* changes by 13×, and no prediction of the form "N× memory
bound" transfers between dtypes, between the fused and unfolded graph, or between
shapes.

- **Operator / device:** KDA, fp32, `B=2,T=512,HV=4,K=V=64,C=64`; T4 and P100.
- **Tag:** **(i)** for the regime (you cannot raise the intensity of an operator by
  scheduling it); the *dtype-dependence* is a measurement-design point, not a tag.
- **Arithmetic.**
  - FLOPs at this shape: per `(b, chunk, head)` = 7 matmuls × `2·64·64·64` = **3,670,016**; × `B·NT·HV = 2·8·4 = 64` → **234,881,024 FLOP**.
  - Bytes, kernel rows only, one write + one read, fp32: **290.7 MB** (H2's arithmetic).
  - Intensity = 234.88e6 / 290.7e6 = **0.81 FLOP/byte**. Folded: 234.88e6/22.3e6 = **10.5 FLOP/byte**.
  - vs T4 fp32 ridge 25.3 → **31×** and **2.4×** memory bound. vs P100 fp32 ridge 25.5 → the same, by construction of the coincidence in §2.
  - Contrast: `NOTES.md` reports 37.5 FLOP/byte at `B=4,T=4096,H=32,K=128` **with the decay rows excluded** (`ref_only=True` in `cost.py`), which is the *fused* kernel, and reports it against a 295 FLOP/byte bf16 ridge → 7.9×. The ported graph is a different object.
- **Measurement that would refute it:** `sm__throughput.avg.pct_of_peak_sustained_elapsed`
  **> 60%** while `dram__throughput` is **< 40%**, on the fp32 T4 run. That would make the
  port compute bound, which the arithmetic above forbids, and would mean the FLOP count
  in `cost.py` is wrong.
- **What a refutation means:** the cost model's FLOP side is wrong by a large factor.
  That is worse than a scheduling gap because it invalidates the roofline denominator
  used by every other KDA claim in the study, and it must be found before any ratio is
  published.
- **Second, sharper consequence.** The 256×-intermediate gap and the ridge-point margin
  are **the same number viewed twice**. If the free cards cannot resolve a 2.5× margin,
  they cannot resolve the H2 comparison either, and `PLAN.md`'s noise floor will say so
  before the run rather than after.
- **Cost:** R3 (calibration) then R8. R3 first — it decides whether R8 is worth running.

---

### H5 — 85% of KDA wall time is the scheduler, and a program cache removes it

**Claim.** The 9.01 s of scheduler time per warm call is host-side, deterministic, and
re-paid on every `realize()` because tinygrad has no separation between plan and execute.
A path that caches the *scheduled program list* (not just the compiled kernels) brings the
warm call to within 1.2× of GPU time. tinygrad already has the pieces: `CUDAGraph` in
`runtime/graph/cuda.py` is wired into the CUDA device, and compiled kernels are already
cached (COLD 49.5 s, WARM does not recompile).

- **Operator / device:** KDA, T4 (or laptop METAL — this is host-bound, so the card barely
  matters; what matters is that the host is recorded).
- **Tag:** **(iii)** — `docs/developer/speed.md` says it itself: the scheduler "is
  currently a big bottleneck of training speed" and recompute-vs-store has "no systematic
  way".
- **Arithmetic.** On the laptop: warm 10,604.7 ms, of which build 1.78 s (16%), scheduler
  9.01 s (85%), GPU 0.109 s (1%). If a plan cache removes the scheduler term, warm →
  1.78 + 0.109 = **1.89 s**, i.e. a **5.6×** reduction. If it *also* removes the rebuild, →
  0.109 s, a **97×** reduction. **Pre-declare that the 97× figure is the target and the
  5.6× is the floor**, and that only the 5.6× is testable without writing a new API.
- **Measurement that would refute it:** with a plan cache in place, warm wall is still
  **> 2×** GPU time. That means the cost is not the rewrite passes but something
  re-executed per launch (driver, allocator, `memory_coalescing`, or UOp→range lowering
  after the cache point), and the fix moves down a layer.
- **What a refutation means:** the study's most actionable recommendation
  ("separate planning from execution", `notes/03-kda.md` §4) is aimed at the wrong pass.
  Worth knowing early because it is the recommendation most likely to be acted on.
- **Cost:** ~30 min, and the measurement protocol around it is the fiddly part — see
  `PLAN.md` §4, which is written specifically to make this one not fall into the
  `realize()`-is-a-no-op trap.

---

### H6 — The camera's 14 kernels per DDA step is a memory-model fact, and it reproduces on CUDA

**Claim.** The semantic camera's cost on NVIDIA is not the kernel and not the arithmetic;
it is that tinygrad has no thread-local state, so a 96-step data-dependent DDA becomes
tensor state plus one memory round-trip per indexed load. `notes/01` measured 1,365
kernels for 96 steps (14/step) and 101 reduce kernels (≈1 per gather) on METAL. **This
reproduces on sm_75 and sm_60 with the same per-step kernel count**, because the cause is
`codegen/late/coalesce.py` asserting that every store to a buffer has a distinct affine
address — a property of tinygrad's IR, not of any backend.

- **Operator / device:** netherite `obs_camera`, `B=32, 2304 rays/batch`, T4 and P100.
- **Tag:** **(i)** — no construct denotes "state in registers". `Tensor.custom_kernel` is
  the escape hatch and `probes/real_gather.py` documents that it does not compile.
- **Arithmetic.**
  - Steps: 96. Kernels at 14/step: `96 × 14 = 1,344`, + 21 for setup/edge = **1,365** —
    which reproduces the count in `notes/01` exactly. That is a free self-check on the
    model, and R10 verifies it on the T4 before trusting any of the rest.
  - Elementwise work: `96 × 32 × 2,304 = 7,077,888` element-steps at ~20 flop =
    **142 MFLOP**, which at the T4's 8.1 TFLOP/s fp32 is **17 µs**. Not the bottleneck.
  - Gathers: `96 × 32 = 3,072`, each returning 73,728 fp32 = 0.29 MB. At 320 GB/s that is
    **0.9 µs each = 2.8 ms** if they are genuine indexed reads.
  - The falsifying alternative, and the thing to watch for: if each gather instead scans
    the whole batched region (`32 × 1,328,176 × 4 B = 170 MB`), that is **531 µs each,
    51 ms per frame** — 18× worse, and it would dominate everything else. The METAL
    measurement rules this out (`probes/gather_floor.py`: cost is flat in source size for
    small output counts), so the indexed-read reading is what carries into R10 — but
    R4/R5 exist to re-derive it per card rather than assume it.
  - Predicted T4 GPU time therefore **≈ 3–10 ms/frame at `B=32`**, i.e. 7–25 M rays/s.
  - netherite's published 396 M rays/s is on **their box, not a T4**, and a T4 is a
    fraction of an A100 in both bandwidth and fp32 throughput. The ratio against
    396 M rays/s is **not a valid comparison** and must not be reported
    (see `PLAN.md` §6).
- **Measurement that would refute it:** per-step kernel count on sm_75 is **≤ 2**, i.e.
  tinygrad's CUDA codegen fuses the gather into the elementwise chain where METAL did
  not. Then the camera gap on NVIDIA is a codegen difference between two backends of the
  same compiler, not a language limitation, and the "TinyGrad has no thread-local state"
  claim is backend-specific.
- **What a refutation means:** the camera finding stops being about the language and
  becomes an (ii) about tinygrad's CUDA fusion depth. That is a *smaller* and more
  fixable claim, and it is the kind of thing that would be embarrassing to have asserted
  on Metal evidence alone.
- **Cost:** ~30 min, one process. R10.

---

### H7 — The gather plugin's failure is the memory model, not the backend

**Claim.** `probes/real_gather.py`'s `Tensor.custom_kernel` route to a real indexed load
fails at `codegen/late/coalesce.py` with `attempting multiple stores: 6` on **CUDA as
well as METAL**, for the same reason: the index tensor must be materialised before it can
serve as an address, and `memory_coalescing` groups by `(op, buf, root_src, valid)`, so
six different index values become six stores to one key. `DMC=1` then emits illegal
pointer arithmetic — on CUDA that would be a different error message but the same cause.

- **Operator / device:** 1-D int32 fancy-index gather, T4 and P100.
- **Tag:** **(iii)** for the Tensor API (which never asks for `Ops.INDEX`; S6), **(i)**
  for the address model (a runtime permutation is by definition not affine in the thread
  index). The claim under test is which of the two dominates on CUDA.
- **Arithmetic.** `probes/real_gather.py::test_gather_matches_stock` has 6 cases
  (3 in-range, 1 large-source, 2 out-of-range). Prediction: **6/6 fail to compile** on
  sm_75, and the first failure is `AssertionError: attempting multiple stores: 6`.
- **Measurement that would refute it:** 6/6 pass on CUDA. Then the coalescer behaves
  differently under the CUDA renderer's address-space model and the Metal result is a
  backend artefact.
- **What a refutation means:** the study's portability claim ("the indexed-load cliff is a
  TinyGrad-level property so it should reproduce", `notes/01` §5) is wrong, and the
  camera's per-step kernel count on NVIDIA must be re-derived from scratch.
- **Cost:** ~10 min. R5. Cheap, and it decides whether H6's mechanism is real.

---

### H8 — The gather's source-bound crossover is set by L2 capacity, and it moves between the two cards

**Claim.** The measured M1 Pro crossover (cost flat in source size to 4.2 MB, then
670 µs at a 67.1 MB source ≈ 100 GB/s) is **L2 capacity, not a fixed algorithmic
threshold**. Therefore the crossover source size scales with the device's L2: predicted
**0.5–1.0× L2 size** on each card. P100 (smaller L2) crosses first, T4 second, and the
M1 Pro point is the outlier that made it look like a constant.

- **Operator / device:** `probes/gather_floor.py`, 6 source sizes, T4 and P100.
- **Tag:** **(i)** — `Tensor.gather` is a one-hot + `where` + `sum` (S6); a compiler
  optimising over a fixed arithmetic-intensity budget cannot remove the reduction.
- **Arithmetic.** L2 sizes are **not asserted here** — Run 0 reads them from
  `cudaDeviceProp.l2CacheSize` and the prediction is then stated against the *measured*
  numbers before the sweep runs. The falsifiable content is the *relation*, not the
  absolute: `log2(crossover_bytes)` is a linear function of `log2(L2_bytes)` across the
  three devices with slope between 0.5 and 1.0.
- **Measurement that would refute it:** the T4 and P100 crossover source sizes are within
  **20% of each other** despite a ≥ 2× difference in L2. Then the crossover is not
  cache-driven and there is an unmodelled mechanism.
- **What a refutation means:** the mechanism is something else — most likely the
  one-hot's own materialisation threshold, which *is* a compile-time constant and would
  then reproduce identically on every device. That is a cleaner story and it is worth
  knowing before the camera numbers are interpreted.
- **Cost:** ~10 min per card. R4.

---

### H9 — RETRACTED. The `≈0 wrong in 200,000` prediction was a units error, off by ~5e6

**Status: retracted. The prediction was wrong by a factor of ~2×10⁶, and the
arithmetic that produced it is the interesting part. The hardware was never at
fault; the derivation was.**

**The prediction, as it stood.** `a / b` is rewritten to `a * reciprocal(b)` in
the Tensor API (S1) and `RECIPROCAL` renders as `(1/x)` (S2), which NVRTC compiles
**without** `--use_fast_math` (S3) into `div.rn.f32` — a correctly-rounded IEEE
reciprocal. So the CUDA path is **double-rounded**, not approximate. Predicted:
`a/a == 1.0` for **all but ≤ 3** of 200,000 inputs, and `a/b` differs from
correctly-rounded in **≤ 3** of 200,000.

**What actually happened, measured.** On the T4
(`out/kaggle/20260928-014248-gpu/logs/divide_precision.log`, N = 200,000):
CUDA **54,270 wrong (27.13%)**, **bit-identical** to `a * fl32(1/b)`; `a/a`
**30,748 wrong (15.37%)**, every one of them `0x99999994 = 1 − 2⁻²⁴`. Refutation
criterion (1) fired on the first count and (2) on the second, by four orders of
magnitude. See ledger **P-1** and **C-04**.

**Where the derivation went wrong — a units error, and it is worth keeping.**
The original arithmetic was:

> the exact quotient is displaced by between ½ and 1 ulp of the result; the
> final rounding differs only if that shift crosses a boundary, which for a
> mantissa-uniform `a/b` has probability ≈ `2⁻²⁴·(1..2) ≈ 1.0e-7`

The displacement is computed in **ULPs of the result** and the threshold is
**one ULP of the result**; multiplying by `2⁻²⁴` there divides a *relative*
epsilon by an *absolute* threshold. A displacement of `s` ULPs **is** a
displacement of `s` rounding cells, so the crossing probability is `s` — order
one, not order `2⁻²⁴`. The correct figure is the mean displacement of the
exactly-rounded quotient measured in ULPs of the result, and it is:

| quantity | value | where |
|---|---|---|
| mean displacement, 18 independent-law points | **22.7% – 27.1%** (median 25.8%) | `derivation/divide.py:cell_failure_fraction` |
| rate the same draw actually produces | equal to it **to within 0.3%** at all 18 points | same |
| what the retracted derivation gave | 1.2e-7 | — |
| **error factor** | **2.15e6** | same |

So the derivation was not merely imprecise — it was wrong in a way that a
sanity check would have caught, and the check was one line: a probability of
`2⁻²⁴` cannot be the crossing rate of a half-ULP displacement.

**Three further claims in the original entry were wrong, independently of the
arithmetic.** All three were also in the wider record, and all three are
corrected here rather than left standing:

- *"the signature of a reciprocal **approximation**, which is what MSL contracts
  `1/x` into."* **False.** `1 − 2⁻²⁴` is the signature of a **correctly rounded**
  reciprocal whose error landed between `−2⁻²⁵` and `−2⁻²⁴`. Two independent
  confirmations: a hypothetical reciprocal 1 ULP *worse* than correctly rounded
  predicts **86.5%** wrong against the **27.1%** measured
  (`derivation/divide.py:reciprocal_sensitivity`); and the only wrong `a/a` value
  is `0x3f7fffff`, with `0x3f800001` above **never** produced in any regime on
  any device — a one-sided, exactly-predicted failure
  (`derivation/divide.py:corner_predicate`). An approximation has a two-sided
  error. This is the **double rounding** the entry itself named, and it is not
  an approximation.
- *"`/ ` never goes through `Ops.RECIPROCAL`".* **Backwards.** `div()` builds
  `MUL(a, reciprocal(b))` and `reciprocal()` is `alu(Ops.RECIPROCAL)`
  (`mixin/elementwise.py:251`, `:465`); `/` always goes through it. Ledger
  **C-11**.
- *"`cstyle.py:274`'s `Ops.FDIV: (a/b)` is dead code because S1 already rewrote
  the division into a multiply."* **False.** `Ops.FDIV` is the load-bearing last
  line of the CPU repair: `ClangRenderer` deletes `RECIPROCAL` and adds `FDIV`
  (`renderer/cstyle.py:271,274`), and the decomposition at
  `codegen/decomp/op.py:133-136` — gated on `if Ops.FDIV in ops` —
  **manufactures** the node S1 never emitted, which is why CPU measures 0/200,000.
  Ledger **C-12**.

**What the corrected claim is.** Not a prediction — a derivation, and the
citation is `derivation/divide.py:mechanism_sentence`:

> TinyGrad's float32 `/` is lowered in the Tensor API — above the scheduler,
> above backend selection — to `a · fl32(1/b)`, so on every backend that does not
> itself render a real divide it returns a value bit-identical to a correctly
> rounded reciprocal followed by one correctly rounded float32 multiply; the
> result is wrong by exactly 1 ULP in 25.9%-27.0% of general `a/b` cases — a rate
> that depends on the divisor law, so it travels with the pair `(a,b)` and not
> with the operation — and no user-facing knob changes any of this.

Two clauses there are load-bearing and are commonly dropped:
**"bit-identical to a correctly rounded reciprocal"** is separate from the
percentage (a reciprocal 1 ULP worse predicts ~86%), and **"no *user-facing*
knob"** is weaker than "no knob" — `Ops.RECIPROCAL` exists and PTX renders it as
`rcp.approx.f32`, but `/` never routes to it as a choice, so there is nothing to
expose.

**The camera conclusion is unaffected and should be reported separately:** the
residual is a 65,536-entry sin LUT with 9.587e-05 buckets, so a 1-ULP divide is
one of two contributions and not the whole story. Do not let the divide result be
reported as *the* camera explanation.

---

### H10 — The `a*(1/b)` rewrite is in the Tensor API, not the renderer (a correction, made falsifiable)

**Claim.** In tinygrad 0.14.0 the rewrite is at `mixin/elementwise.py:251`
(`d = a.alu(Ops.MUL, b.reciprocal())`) — **above** backend selection — so every
cstyle-path backend (PYTHON, CL, METAL, CUDA, HIP, WEBGPU) gets it, and the CPU
backend's exactness is a coincidence of clang lowering `1.0f/x` in C, not evidence that
the IR is exact.

- **Operator / device:** any; the source check needs no device.
- **Tag:** not a performance claim. This is a **provenance correction** to
  `notes/02-obs-camera-residuals.md` §"Correction: it is TinyGrad's codegen, not
  Metal's division", which attributes the rewrite to the renderer. Getting this wrong
  would make the study file an upstream bug against the wrong layer.
- **Measurement that would refute it:** the generated **CPU** C source for `a/b` contains
  a real `/` (or the CUDA C++ source contains `div.rn` *written by tinygrad*, i.e.
  `Ops.FDIV` is reachable), meaning a scheduler pass between the Tensor API and the
  renderer re-associates the multiply back into a divide. Then the rewrite is not
  universal and the CPU's exactness is structural after all.
- **Predicted magnitude if true:** the generated source contains the substring `(1/` in
  **100%** of the backends that use the cstyle family (CPU, CUDA, HIP, METAL, CL,
  WEBGPU) and `0` real `/` operators at the divide site. Binary, and it is checkable by
  reading one file per backend. There is no magnitude to compute — which is exactly why
  this is R0 and not R2: it is a string comparison, and it settles both H9's causal claim
  and this correction at zero GPU cost.
- **What a refutation means:** `notes/02-obs-camera-residuals.md`'s "Correction: it is
  TinyGrad's codegen, not Metal's division" would be **doubly wrong** — wrong about the
  layer (renderer vs API) *and* wrong about the scope (per-backend vs universal). The
  upstream bug report would have to be rewritten against the Tensor API, which is a
  different fix with a different blast radius: it would change `/` for every float user,
  not just Metal ones. That is a materially different patch, which is why this correction
  is pre-registered rather than quietly applied.
- **How to check for free:** `CUDARenderer` is a pure renderer class. Instantiating it
  locally with `arch="sm_75"` and rendering the same expression yields the CUDA C++
  source **with no GPU present**, because NVRTC compilation happens server-side on Kaggle.
  So the source half of both H9 and H10 costs **zero GPU minutes**. Only the SASS
  disassembly (`nvdisasm`) needs the card.
- **Cost:** seconds. R0.

---

### H11 — Neither free card has a tensor core for the dtype KDA is measured in **(iv)**

**Claim.** `get_cuda(arch)` returns `[]` for sm_60 and a single `m8n16k8` f16 shape for
sm_75 (S4). Therefore:
- **On the P100, TinyGrad has no tensor-core MMA for any dtype.**
- **On the T4, fp32 has none** (the sm_75 list is f16-only; tf32 and bf16 enter at sm_80).
- **On the T4, bf16 has none either** (Turing has no bf16 tensor core, and tinygrad's
  `cuda_81616` bf16 entry is sm_80+).
- The KDA port is **fp32 only** (`tiny_kda.py:44`, `F = dtypes.float32`). So **on both
  free cards, KDA as written has no tensor-core path at all**, and the 8.0× fp32→fp16-TC
  headroom that H1 is about is unreachable on this hardware.
- Separately: `NOTES.md` §6's entire arithmetic-intensity argument is stated in **bf16
  bytes** (`ES = 2`). A bf16 KDA result on a T4 is a result about a path that **cannot use
  the tensor cores**, so it will not be comparable to any published bf16 KDA number.

- **Operator / device:** KDA, both cards.
- **Tag:** **(iv)**. Not (i): the compiler is fine, the hardware table is right. Not (ii):
  no search budget fixes a missing arch entry. This is the tag the brief asks for
  explicitly and it is the one with the clearest money consequence.
- **Arithmetic.** 8.1/65 = 0.125 (T4 fp32 vs fp16-TC, **8.0×**), 18.7/125 = 0.150
  (P100, **6.7×**). Those are the numbers a Blackwell/A100-class card would have
  available and these cards do not.
- **Measurement that would refute it:** `wmma` or `mma.sync` appears in the generated
  source for a **bf16** or **fp32** KDA matmul on a T4. That would mean `get_cuda` is not
  the whole story (e.g. the `PTXRenderer` fallback in the `[CUDARenderer, PTXRenderer,
  NVCCRenderer]` chain gets picked and has its own path).
- **What a refutation means:** the free cards are sufficient for the KDA arm after all
  and the study does not need to rent anything.
- **What confirmation means — and this is the actionable finding:** the KDA arm's
  headline claims must be restated as **fp32 CUDA-core** results, and the study needs
  **either** (a) an A100-class card for the tensor-core arm, **or** (b) an explicit
  narrowing to the questions that *are* answerable on old hardware — which are, per this
  list: H2 (bandwidth/intermediate), H3 (graph size), H4 (regime), H5 (host-side
  planning), H13 (which RC label owns the gap). Those are the questions the free hardware
  is actually good for, and they are not the interesting ones about tensor cores.
  **Decide this in Run 0, before any GPU time is spent.**

---

### H12 — ≥80% of the camera's 96-step unroll is dead, and that fraction is architecture-independent

**Claim.** Rays march a mean of **19.8** steps with a measured max of **84** against an
analytic bound of 86.1 and an unroll of 96 (`notes/01` §2). So the mean ray pays
`96 − 19.8 = 76.2` dead iterations, i.e. **79% of the unrolled work is provably dead**.
This is a property of the *algorithm and the region*, not of the device, so the fraction on
a T4 is 79% ± the sampling error of a different region/pose seed.

- **Operator / device:** camera, T4.
- **Tag:** **(ii)** — a cost model that knew the trip count, or a stream-compaction
  primitive, fixes it. No algorithm change. (`notes/01` §4 item 1.)
- **Arithmetic.** If dead iterations cost the same as live ones (they do: the tensor is
  `[B, 2304]` at full width for all 96 steps), then compaction into 4 buckets of 24
  removes `96 − 4·24 = 0` … more carefully: with mean 19.8 and max 84, a 4-bucket split
  at 24/48/72/96 gives per-bucket means bounded above by 24, and the total is
  `24+24+24+24 = 96` iterations *on the compacted tensor*, whose width per bucket is the
  number of live rays in it. The predicted saving is on the **elementwise** work only:
  **~60%** of elementwise FLOPs (`notes/01` §4 item 1 says "removes ~60% of the dead
  work"). The gathers are unaffected — a dead ray still gathers, unless compaction also
  removes it from the index set, which is the hard part and is why `notes/01` calls it
  item 1 with a caveat.
- **Measurement that would refute it:** the measured per-ray step histogram **on the T4
  workload** has a mean above **48** — i.e. more than half the unroll is live. That would
  mean the M1 Pro region's air fraction is not representative and the 4.8× waste figure
  is dataset-specific.
- **What a refutation means:** the compaction fix is worth ~0, and `notes/01`'s item 1
  (the highest-leverage recommendation for the camera) should be struck rather than
  ranked first.
- **Cost:** zero GPU (it is a step-count statistic). Bundle it into R10.

---

### H13 — KDA's cost on a T4 is bandwidth, and the RC label is (i)/RC2b — not RC3, not RC1

**Claim.** Given H2's 290.7 MB against a 25.3 FLOP/byte ridge, the ported KDA call on a
T4 is bandwidth bound: `dram__throughput.avg.pct_of_peak_sustained_elapsed` **> 60%** and
`sm__throughput` **< 20%**, with `sm__warps_active.avg.pct_of_peak_sustained_active`
**< 30%** and `local` memory traffic (`l1tex__t_sectors_pipe_lsu_mem_local_op_ld/st`)
**< 1%** of total.

- **Operator / device:** KDA fp32, T4.
- **Tag:** this is a **classification** hypothesis, in the vocabulary of 2607.04454. It
  asks which of RC0/RC1/RC2/RC3 owns the KDA gap. Claimed: **RC1/RC2b (bytes), not RC3
  (spill)**.
- **Arithmetic.** Predicted bytes 290.7 MB; at 320 GB/s datasheet that is **908 µs**;
  predicted FLOP time 234.88e6 / 8.1e12 = **29 µs**. Ratio **31:1** (H4). If the measured
  `dram__bytes.sum` lands within 20% of 290.7 MB, i.e. in **[233, 349] MB**, the bandwidth
  model is right. A spill would show up as *extra* traffic in the `local` counters and as
  `achieved_occupancy` collapse; a latency bound would show both pipes low.
- **Measurement that would refute it:** `dram__throughput < 40%` **and**
  `sm__throughput < 40%`. Then neither pipe is the bottleneck and the gap is latency /
  occupancy / launch, which is RC3 or the 12,090-launch problem — and the entire
  "bandwidth is the whole game" framing of `NOTES.md` §6 is wrong for this device.
- **What a refutation means, concretely:** the study's recommended fix changes target. The
  exp-folding pass (H2) buys **0 bytes**, because nothing is bandwidth-bound; the fix
  becomes launch-count or occupancy, i.e. H5 (plan cache, 5.6×) and H3 (graph size). And
  the RC labels in the write-up move from RC1/RC2b to RC3 or to "unattributed". Because
  H2 and H13 share the same Nsight profile, **they are decided in the same 30 minutes** and
  only one of them can be right — that is deliberate, and it is why R8 is a single run and
  not two.
- **Why this matters beyond KDA:** 2607.04454's own cautionary result is that the
  register-pressure hypothesis was *refuted where it was most expected* (`n_spills = 0`
  across every tested conv filter) and that spills appeared on a different GPU with no
  correctness signal. On a T4 the register file, the shared-memory budget and the L2 are
  all different from an A100, so the RC label is **architecture-indexed** and the study's
  one-GPU-per-dtype result is one cell of a grid the field has declared infeasible. This
  hypothesis is the study's own version of that caution, pre-registered so it cannot be
  quietly re-labelled after the counters come back.
- **Cost:** same profile as H2. R8.

---

## 5. Hypotheses that no available measurement can refute

Listed because a finding. Each says where the money should go.

| # | claim | why it is unrefutable here | what would refute it |
|---|---|---|---|
| U1 | **"TinyGrad cannot express a sequential scan / persistent kernel"** | It is (i) by construction, decided by reading the Tensor API: there is no `scan`/`while` construct, and a Python `for` is unrolling. There is no candidate mechanism to test on a T4 or P100, so there is nothing to measure. The conclusion is already fixed and cannot be moved by any run. | An upstream scan primitive. Until then this is a *definition*, not a hypothesis, and it should be reported as a definition. |
| U2 | **"TinyGrad's generated KDA is 256× less efficient than the hand-written kernel"** | The hand-written side is a **Triton** kernel (`fla/ops/kda`) that will not run on sm_60 at all and was tuned for Ampere+ tensor cores. On a T4 the comparison is against a *different, older* kernel that someone has to write. Until that kernel exists, the ratio is unmeasurable — and the tempting alternative, quoting fla's published speed or quoting the 396 M rays/s camera figure from a different card, is exactly the p-hacking this file exists to prevent. | See §6 and `PLAN.md` §6. |
| U3 | **"The KDA cost model is right."** | `kda/cost.py` is analytic and its own docstring says nothing in it is measured. The FLOP side is independently checkable (17/17 `test_ref.py`, and Eq. 13's own 1.45× undercount is documented in `SPEC.md` §8), but the **byte** side assumes "one write + one read of each intermediate", which the kernels explicitly do not honour. On hardware this *is* refutable (H2), so U3 is not a standalone unrefutable claim — it is subsumed by H2. | H2. |
| U4 | **"The camera port is 4 orders of magnitude off."** | 396 M rays/s is netherite's box; the port was measured on an M1 Pro. The two numbers have no common device. On a T4 the honest statement is a roofline fraction, and a roofline fraction is not comparable to a published throughput. | Nothing on free hardware. Needs the same box, or a T4-native hand-written baseline. |
| U5 | **Anything about TPU v3-8 and TinyGrad.** | tinygrad 0.14.0 has **no TPU backend** (S7). Every (i)/(ii)/(iii) tag in this file is a claim about tinygrad, so **no TPU measurement can refute any of them.** The TPU arm can only test the hardware-independent structural claim (that the `C²K` intermediate dominates) using XLA as a second compiler — which `kda/tpu/shapes.py` has *already* pre-registered as S1/S2/S3, with the S3 control that makes it falsifiable. **Do not re-derive it here; read that file.** Also note v3-8 is deprecated and sessions auto-switch to v5e-8 (`kda/tpu/roofline.py`), so any TPU number is a number about hardware that may not be the hardware it claims. | Nothing. This is a scope finding, not a hypothesis. |

---

## 6. The deleted hypothesis

**Deleted: "TinyGrad's KDA port is *N*× slower than flash-linear-attention's kernel on
this GPU."**

I had this in the list and it is the study's headline. It is deleted because **nothing
available can refute it**, and the reason is a hardware fact that had to be discovered
before writing it down:

1. **The P100 (sm_60) cannot run the comparison at all.** `get_cuda` returning `[]` (S4)
   tells us tinygrad has no tensor cores there; the same table's shape vocabulary tells us
   why: the comparison needs `m16n8k16` bf16 MMA, and `cuda_81616` only enters at sm_80.
   fla's KDA kernels are written against exactly those shapes. A hand-written baseline
   on sm_60 is a **different kernel written from the same SPEC.md by a different author**,
   and a ratio between two such kernels measures the authors, not the compilers.
2. **The T4 (sm_75) has a *smaller* gap but the same problem in milder form.** Only
   `m8n16k8` f16 exists, with **no bf16**. The KDA port is fp32. So on the T4 the honest
   comparison is *fp32 CUDA cores* against *f16 tensor cores* — a dtype comparison wearing
   a compiler comparison's clothes.
3. **The tempting fix is the exact error this pre-registration exists to prevent.** One
   can write an sm_75 kernel from `kda/SPEC.md` in an afternoon, time it, and report
   "TinyGrad is *N*× slower." But 2607.04454's residual analysis says the surviving gaps
   are "where cuBLAS, cuDNN and `torch.argmax` are **themselves well-tuned**" — a residual
   against a human is a **floor, not a defect**. An *untuned* sm_75 kernel is worse than
   tuned: it makes the gap look larger, and the study would be reporting the author's
   effort. KernelBench's 1.43× → 0.88× is the same lesson.
4. **What replaces it** is not a weaker claim but a *different* one that is measurable
   here: compare TinyGrad against **the device roofline** (H4, H13) and against **itself
   with one pass changed** (H2's 13× folding, H5's 5.6× plan cache, H12's compaction).
   Those are all-else-equal ratios where the author controls both arms, so they are
   attributable. The compiler-vs-human ratio is the one ratio where neither arm is
   controlled.

**Consequence for the plan:** `PLAN.md` places a baseline-construction gate (R1) *before*
any timing. If a hand-written sm_75 baseline is not built and validated to ≤ 1e-5 against
`kda/ref.py`, the study reports roofline fractions and self-deltas and **says nothing
comparative about TinyGrad versus a human**. That is a legitimate, publishable result. The
deleted ratio is not.

---

## 7. Amendment rule

1. A hypothesis is **refuted** or **supported**. Never "partially supported" — split it
   into two hypotheses with separate refuting measurements, or drop it.
2. A hypothesis whose predicted magnitude is below the measured noise floor (`PLAN.md` §5)
   is reported **untestable on this hardware**. It is not refuted and not supported.
3. A hypothesis whose refuting measurement the hardware cannot produce (U1, U2, U4, U5)
   is reported **unrefutable here**, with the missing hardware named. This is a finding and
   goes in the paper.
4. Any amendment records: what was predicted, what happened, and whether the prediction was
   wrong about the *world* or about the *arithmetic*. A prediction falsified by a bug in
   the harness is a harness bug, not a refutation, and must say which.
5. Nothing is added to this list after the first GPU run. Anything discovered later goes
   into a clearly-marked "post hoc" section, with its prior stated as "not pre-registered",
   and is not used to support the study's conclusions.
