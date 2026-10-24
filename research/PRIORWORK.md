# PRIORWORK

**The recommendation, up front.** Do not build your own harness. Reimplement
**Li, Rathnasuriya & Yang, arXiv 2607.04454v3 — "Correct but Slow"** (UT Dallas; artifact at
`github.com/tingxi-li/DSLPerfGap`). It is the only source found that does the three things your
study needs: it compares a DSL against a *deliberately maximal* library baseline, it **attributes**
each gap to a cause with a hardware-counter signature, and it **repairs** each attributed cause to
show how much was recoverable. Their artifact is public and their kernel suite, baselines, tolerances
and counter list are all specified in enough detail to reimplement. Everything else below is
supporting protocol you graft onto it.

Then add one thing they did not do and you must: **the 2607.16241 TF32-baseline correction**, because
it is the difference between a result and a non-result.

---

## Option A (recommended): 2607.04454's protocol, adapted to TinyGrad

### A.1 What to copy verbatim

**Unit of comparison.** One (operator, shape, dtype) cell. Not a model. Not an op chain. 2607.04454's
22-kernel suite is explicitly "a diagnostic probe, not a proposed benchmark" — keep that framing; it
buys you the right to a small, targeted suite instead of an impossible grid.

**Baseline definition (this is the load-bearing decision).** For each workload, "the strongest
available vendor-library path through PyTorch," per category:
- GEMM → cuBLAS, with a **large cuBLAS workspace** and the algorithm recorded.
- Convolution → cuDNN, with `cudnnFind`-style selection.
- Attention → PyTorch SDPA with the **FlashAttention** backend, and **reported separately**, because
  "the Triton and TileLang implementations are not algorithmically identical to this baseline."

Their justification for maximal baselines is a sentence worth copying: "a stronger unexercised baseline
would only widen the gaps." The same logic should make you *beat* the baseline before you report a gap.

**Two metrics, and both are required.**

```
E_lib = (t_lib / t_DSL) x 100%        # library-relative, baseline-dependent
rho   = W / (t x P)                    # roofline fraction, baseline-independent
```
- `W` = bytes moved if memory-bound, FLOPs if compute-bound.
- `P` = the relevant hardware peak — **measured on your machine, not copied from a data sheet.**
- Report `E_lib` against the *library* and `E_lib` against *PyTorch eager* separately. 2607.04454 does
  exactly this, and the divergence is the whole point: TileLang's fused linear+activation is
  **468.5%** of PyTorch (single launch, no intermediate allocation) while remaining below cuBLAS.

**Why both.** The library-relative screen is *circular* and they say so: "PyTorch is both the baseline
and the de-facto definition of 'good,' so the screen cannot certify a kernel that merely matches an
already-slow baseline, and it can credit a kernel that beats an *unfused* eager path for the wrong
reason." Their 873% RMSNorm "win" is exactly that, and they exclude it as "an unfair-baseline
artifact." The roofline anchor de-inflates it: that same kernel sits at `rho = 0.69` — efficient, but
nowhere near the hardware.

**The same artifact marks two more cells, and you must not quote them either.** Their Table IV carries
TileLang RMSNorm at **990% (A100)** and **1303% (GH200)**, both flagged with the identical footnote:
"RMSNorm Elib>100%: the fixed kernel fuses the scaling pass PyTorch's eager path leaves unfused."
§VI-B repeats it in prose. So all three of 873% / 990% / 1303% are one baseline defect, not three
capability results — and none of them may be cited as evidence of a DSL capability gap. If you want a
citable number from that family, the paper's own choice is `rho = 0.69` or the below-parity LayerNorm
residuals, not the `E_lib > 100%` cells.

**The decision rule, which is the transferable contribution.** High recovered `rho` + large
naive→optimized "cliff" = **authoring artifact**, one dominant pattern repairs it. Low `rho`
surviving best-effort optimization = **structural residual**. Their two checks "disagree exactly in a
judgment band near rho ≈ 0.5": Softmax (`rho = 0.47`, `E_lib = 0.87`) and Matmul (`rho = 0.47`,
`E_lib = 0.68`) — same roofline, opposite calls. **Report both. Never report one.**

Their headline use of this: recovered families land at `rho = 0.41–0.89`; structural residuals at
`rho ≤ 0.34` (A100) and `rho ≤ 0.28` (GH200) **after best-effort optimization**; the two architectures
agree on every classification. That agreement is the validation of the instrument.

### A.2 Correctness protocol — adopt the strict version, it is free

2607.04454's own floor: per-dtype tolerances **fp32 1e-5, fp16 1e-3, bf16 1e-2, relaxed 2× for
reductions**, plus an edge-case suite (NaN / Inf / denormal / all-equal), across all kernels, 0 crashes.

That is *not sufficient*, and three papers agree. You should do all of the following, and cite them:

| Addition | Why | Source |
|---|---|---|
| **TF32-enabled baseline** | "providing a more realistic estimation of execution on modern GPUs"; flips the best frontier model from **1.43× to 0.88×** geomean | 2607.16241 |
| **Multiple hidden input distributions**, not one | models "exploit the narrow test distribution by hardcoding bypasses for specific tensor values" | 2607.16241 |
| **Scale-invariant correctness check** | `torch.allclose` with *absolute* tolerance "is satisfied by a tensor of zeros on **4 of 60** level-1 problems" | 2609.21058 |
| **fp64 CPU reference**, per-(op,dtype) absolute tolerances, seeded fuzzing | flags **9/9** LLM-style transcription bugs, passes **15/15** correct controls; extended run gives **10/10** caught, **16/16** clean, **identical verdicts across RTX 3060 / A10 / L40S / A100 SXM4 / H100 NVL** | 2606.20128 |
| **Adversarial-seed suite** | the official check "misses **one in six** witnessed faults (16.9%) … 8.7% of arithmetic faults escape, but **78.6% of precision faults** do" | 2609.22220 |
| **Incomplete-write detection** | one kernel scored **283×** while writing **0.3%** of its output buffer | 2609.21058 |

The last one is the one people skip and the one that will embarrass you: assert that the output buffer
was fully written, not just that the written part is right. The fact that GPU verdicts are
**GPU-independent** (2606.20128) is what lets you run a 2-GPU sweep under one correctness protocol.

### A.3 Measurement protocol

Copy these; they are cheap and they are what make the numbers survive review.

- **Lock clocks.** Pin graphics and memory clocks. Their locked-clock runs hold run-to-run relative
  **stddev ≤ 0.9%**; NCU counters stable within **2% across five runs**. Every table must state
  whether it is clock-locked. Their absolute numbers are quoted both ways where it matters
  (LayerNorm 1293× unlocked vs 323× locked) — be explicit.
- **Host-synchronize** around every timed region.
- **100 runs**, report median; separate compile/autotune time from kernel time from end-to-end.
- **Re-tune at the deployment shape.** Their caution: "auto-tuning at small shapes can pick
  configurations arbitrary at deployment shapes" — they re-tuned the three affected kernels at the
  real shape. Re-tuning one kernel at one deployment shape costs **211 s** vs under a second at a
  small shape, and that asymmetry is a *finding*, not an inconvenience: report it as your search-cost
  axis.
- **Pre-realize inputs and warm up**, so you are not timing allocation.
- **Averaging over a shape sweep**, not a single shape. Both of their cross-architecture traps
  (logsumexp on sm_90; the shared-memory-caching Softmax, **20× slower at N=8192 and 111× at
  N=32768**, which hits parity on GH200 purely because Hopper's larger shared-memory budget hides the
  defect) were single-cell failures.

### A.4 The counter set — the attribution instrument

This is what turns "slow" into a cause. 2607.04454 uses **seven Nsight Compute counters**, each with
a distinct signature. Reimplement this table; it is the reusable part of their contribution.

| Root cause | Diagnostic counter | Observed signature |
|---|---|---|
| Serial reduction loop (authoring) | `warp_stall_long_scoreboard` (and check `warp_stall_barrier` to rule barriers out) | long_scoreboard **59.56 → ≈0**; barrier already ≈0 in both variants |
| Absent vectorized loads (codegen) | `l1tex__t_bytes.sum.per_second` / **bytes-per-sector** | **50% bytes/sector**, 1 sector/request, DRAM throughput 59.2% |
| Strided-access non-vectorization | load **efficiency (bytes/sector)** + sectors per request | Triton conv2d **12.55%** at **16.4 sectors/request** |
| Autotune coverage (tuning) | TFLOP/s vs the swept config set; sweep the axes the default grid omits | 12-config block-tile grid gives **Δ = 0pp**; that grid varies **only** block-M/N/K, while the expanded search adds `GROUP_SIZE_M` (L2 swizzle) + `num_warps` + `num_stages` and gives **1.37×** (59.7%→81.9% at 16384²) |
| L2 residency (library maturity) | **L2 hit rate** + DRAM throughput | Triton **49.4%** hit / 85.9% DRAM; cuBLAS **80.5%** hit / 28.8% DRAM |
| Register spill (occupancy) | `local_op` load/store bytes, registers/thread, achieved occupancy | TileLang LayerNorm: **254 regs, 12.4% occupancy, 51.5 GB local-load + 34.4 GB local-store** |
| Algorithm selection (library) | SM-utilization delta from a deterministic A/B | cuDNN deterministic vs non-deterministic 3×3 s1: **0.03%** (ratio 0.9997) |

Methodology lessons from this section. **1–2 are their techniques; 3–4 are transcription traps in
their own tables, and they are where a faithful reader most easily inverts the paper:**

1. **Refute the obvious hypothesis with a counter.** Their register-pressure section begins: "A natural
   hypothesis is that large-filter convolution hits this wall; **NCU refutes it: Triton conv2d reports
   n_spills = 0 at every tested filter** (1×1 through 7×7), even though register usage rises with K²
   (up to 224 regs/thread)." Reporting a refuted hypothesis is worth more than a confirmed one.
2. **Isolate each cause with a targeted mitigation and report the recovered delta.** Their Table II is a
   contribution column per cause: RC0a → **1347×** (LayerNorm, A100), RC1+RC2 → **2.2×**,
   RC2a → **1.37×** (matmul), RC4 → **≈2–3%**. An unattributed gap is not a result.
3. **Never let two of your own rows sit unreconciled.** This is the one place their protocol is easy to
   mis-transcribe, and getting it backwards inverts the conclusion. Table II's RC0a cell reads
   **1347×** (LayerNorm, A100) — that is the *untuned* variant, **364 → 0.270 ms**, clock-locked, from
   **Table IV and footnote 1**. **Table III's** A100 LayerNorm cliff is **380×**, the *tuned* variant;
   GH200 is 1541× (worst-case, Table III). A third, separate cell is A100 LayerNorm **vs PyTorch =
   299×** (97.0 ms), and the GH200-vs-PyTorch set is 323× locked / 329× suite profile / 1293×
   worst-case. Their own footnote 1 exists precisely to keep these apart.
4. **The Δ = 0pp row and the 1.37× row are the same finding, not a contradiction.** If you transcribe
   "tile autotuning gave zero improvement across all 22 kernels" without the resolution, a reader
   concludes the technique is worthless — the inverse of what the paper shows. Their resolution,
   verbatim: "The tension with the matmul speedup of Section VI-D is a **search-space distinction, not
   a shape distinction**: the 12-config grid varies **only** the block-M/N/K tile dimensions, while the
   expanded search additionally sweeps `GROUP_SIZE_M` (the 'L2-swizzle'), `num_warps`, and
   `num_stages`." Recovered: **59.7% → 81.9%** at 16384², **71.5% → 77.3%** at 4096². Their verdict:
   "**the missing configurations, not an exhausted budget, bound default performance**" (restated in
   §VI-D: further iterations "converged without gain, so the expanded configuration set and L2 swizzle
   are the effective ceiling"). Copy the resolution with the number, or do not copy the number.

### A.5 The repair protocol

2607.04454 uses **AKO** (an existing LLM-based agentic kernel-optimization harness) "as an
off-the-shelf tool under a fixed protocol" — correctness-gated iterations, large-input shapes, **no
language switching** — to instantiate the root-cause-derived patterns, then re-validates
"provenance-agnostically" against the same tolerances and counters, and reports recovered performance
as **a lower bound** on user-space-recoverable gain.

That is the right shape: an *off-the-shelf* optimizer under a *fixed* protocol, so the credit goes to
the attributed cause and not to the optimizer's cleverness. Keep "no language switching" — it is what
makes the comparison mean anything.

The portable outcome of that phase is the **dominant-pattern dichotomy**: most dramatic gaps are
authoring artifacts one pattern fully repairs; a minority are genuine residuals that survive
best-effort optimization. For TinyGrad, the two patterns you should pre-registered as predictions are
(a) *the search budget / BEAM* and (b) *SRAM staging + L2 awareness*, because those are the two the
docs themselves name.

### A.6 The suite to build for TinyGrad

Keep 2607.04454's five categories — matrix multiplication, attention, convolution, normalization,
element-wise/reduction — because they are chosen to "capture the dominant computation patterns in
modern deep learning," and keep the sizes: GEMM at 4096² / 16384² / batched 128×2048², normalization
at **8192×8192**, convolution at 1×1–7×7 **at the large shape**, and a depthwise variant.

Three deviations, each for a stated reason:

- **Exclude fp32 GEMM after a documented check.** TileLang "silently lowers float32 to TF32 on SM80,
  failing an exact 1e-5 check on **31.8% of outputs** while a manual FMA kernel passes." Do the same
  probe on TinyGrad and report the result; it is a finding either way. If it fails, measure TileLang/Triton
  GEMM in fp16 and say why.
- **Do not put attention in the same table.** Report it separately, or you will be comparing
  algorithmic tiers and calling the result a schedule gap.
- **Add a recommender-shaped workload.** 2609.21058 shows transformers are a bad target for a
  kernel-win study: **80–86% of runtime is cuBLAS GEMM + FlashAttention**, bounding end-to-end at
  **≈1%**. Recommenders are **58.2%** addressable, concentrated in a single embedding kernel, and
  41.7% winnable at **1.552×** median → **8.63%** end-to-end. This is the workload where a TinyGrad
  result would actually be interesting, and TinyGrad's own docs point at exactly this layer.

### A.7 What you must build yourself, and what is genuinely missing

Stated plainly, because this is a real result about the field:

- **There is no TinyGrad benchmark. At all.** arXiv API query for `all:"tinygrad"` returns
  `totalResults = 0`. No paper, no conference result, no citable number. Every TinyGrad figure you
  publish will be the first of its kind, and every number in this repo is for *other* systems.
- **No published Triton-vs-TinyGrad comparison exists.** The only community performance claims I
  found are a 2024 Hacker News comment reporting one GPT-2 training step on an AMD 7900 XTX as
  "tinygrad ~440ms, PyTorch ~97ms, llm.c ROCm ~79ms, llm.c custom kernels ~58ms" — **a forum comment,
  self-reported, single configuration, and four years old. Do not cite it.** If you want it, measure it.
- **No one has run the correct-but-slow methodology against a *third* DSL.** 2607.04454 covers Triton
  and TileLang only. TinyGrad is a structurally different design (graph-level laziness + UOp rewrites +
  BEAM, not a tile DSL), which makes the transfer of their methodology the open question rather than a
  formality — see the design note below.
- **Ansor's per-operator-vs-cuDNN table could not be extracted** from the HTML (present in the paper,
  lost to text extraction). If you need "auto-scheduling gets within X% of cuDNN," get the PDF. Do
  not quote a figure I could not read.

### A.8 One design decision that makes TinyGrad awkward, and how to handle it

TinyGrad is not a tile DSL. Its own docs decompose speed into **compile (Python) / execution
(driver) / model (scheduler) / kernel (codegen)**, and the load-bearing claim — "tinygrad has the best
execution speed of any framework because it usually **bypasses the GPU driver and prebuilds the command
queue**" — is about **launch overhead, not kernel throughput**.

So a naive port of 2607.04454 will produce a misleading number in both directions: TinyGrad will look
excellent end-to-end because of launch overhead while losing badly per-kernel, and a per-kernel-only
comparison will hide the graph-scheduling layer where TinyGrad says its own worst bottleneck lives.

Report **two** figures and label them: **per-kernel** `t` at large shapes with overhead amortized (this
is the layer comparable to Triton and TileLang), and **layered** `t` following TinyGrad's own four-way
decomposition. Then state the prediction in advance — graph scheduling > codegen (SRAM/L2/TC/TMA) >
nothing at the launch layer — and report which layer owns the gap. That converts your study from a
measurement into a test of a stated model of the compiler stack, which is the thing that is actually
missing from the literature.

---

## Option B (cheaper, narrower): use TritonBench as the human-kernel corpus

If the full study is too large, the highest-value 20% is:

- **Corpus:** TritonBench's **184 real-world operators curated from a GitHub channel spanning 95
  repositories** (`github.com/thunlp/TritonBench`). These are curated *human* Triton kernels — a
  better "hand-written" baseline than LLM output, and already human-optimized, so a gap you find is
  not an authoring gap.
- **Second arm:** 2607.04454's suite with TritonBench's kernels substituted in (they did exactly
  this for LayerNorm, taking the TorchInductor reference instead of the TritonBench version).
- **Measurement:** 2607.04454's locked-clock, 100-run, median protocol, plus the `E_lib` / `rho` pair.
- **Attribution:** the seven-counter table above, which is what makes the result publishable.
- **Correctness:** the strict protocol in A.2.

**Cost of going narrow:** TritonBench covers Triton, one dtype, one shape, and one GPU architecture per
operator. 2607.04454's own critique applies to you, so state it up front and treat your suite as a
diagnostic probe over a sampled grid, not a benchmark. Also note TritonBench already reports a
roofline-anchored efficiency metric, so your contribution over it is the **attribution**, not the
measurement.

---

## What NOT to reuse

- **KernelBench's `fast_p` as a headline.** It gates on correctness and aggregates speed, so a slow
  kernel enters the accepted pool; the correct-but-slow paper's whole point is that "a benchmark that
  reports speedup without gating on performance certifies the wrong property." Use `fast_p` only as
  the *vulnerability* you demonstrate.
- **Ansor's search-space numbers as a "hand-written gap" proxy.** Ansor's headline is
  3.8×/2.6×/1.7× against *AutoTVM*, a search-based baseline, not against a hand-written library. Its
  relevant claim is qualitative and structural: "Compared with vendor-specific static libraries, Ansor
  has more advantages on **uncommon shapes and small batch sizes**, because it is not easy to manually
  optimize for these cases." That sentence is the actual hand-written-vs-auto-scheduled finding, and it
  suggests your study should include unusual shapes and small batches — the regime where search wins.
- **Any "LLM beats PyTorch" framing.** CUDA Agent (arXiv 2602.24286) already makes the sharper claim
  you should adopt: "**LLMs remain uncompetitive with compiler-based systems such as torch.compile** for
  CUDA kernel generation." Your baseline should be the compiler, not eager.

---

## Concrete build list

1. Two GPUs, at least two architectures (their cross-architecture replay used A100-SXM4 + GH200, then
   A100-PCIE + H100 — one architecture family apart is the minimum that catches the logsumexp class).
2. `nsys`/`ncu` (Nsight Compute 2026.2.0 or later) for the seven counters.
3. `torch` (2.8.0+cu128 or later) as the baseline path; pin cuBLAS/cuDNN versions and record them.
4. tinygrad at a pinned commit; pin the BEAM settings and **report the compile/tuning time separately
   from kernel time** — TinyGrad's own docs note BEAM "gets even slower … since that's compiling many
   variants of each kernel."
5. Clock-locked timing harness, ≥100 runs, median, per (kernel, shape, dtype, arch).
6. A hidden-input correctness harness: ≥4 distributions, fp64 reference, per-(op,dtype) tolerances,
   seeded replay, and a **full-buffer-write assertion**.
7. The repair phase: off-the-shelf optimizer, fixed protocol, no language switching, provenance-agnostic
   revalidation, and report recovered gain as a **lower bound**.
8. One pre-registered prediction about which of TinyGrad's four layers owns the gap.
