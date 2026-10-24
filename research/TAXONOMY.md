# TAXONOMY

**Read this first.** A tag is meaningless without a compiler attached. Every item below is tagged
relative to a **named system**, because the same decision is (i) in one compiler and (ii) in another.
Where the paper itself made the distinction, I quote it.

Tag definitions, kept strictly separate:
- **(i) CANNOT EXPRESS** — no construct in the language/IR can denote the decision. Only a *port*
  (different algorithm, different lowering target, hand-written escape hatch) reaches it.
- **(ii) CAN EXPRESS, DOES NOT FIND** — the construct exists and the decision is reachable in
  principle; the search, the cost model, the analysis, or the search budget fails to select it.
  Fixable by a better search, a bigger budget, a better cost model, or a better static analysis.
- **(iii) NOT YET IMPLEMENTED** — the capability is understood and specified, absent from the
  shipping compiler. Fixable by writing the pass.

The same symptom is routinely tagged differently by different authors. Where that happens, I say so
and give my reading, rather than picking one silently.

---

# C1 — What a human must supply that a compiler will not infer

## 1. Tile sizes (BLOCK_M/N/K, num_warps, num_stages)

**Triton: (ii). TileLang: (ii). tinygrad: (ii).**

The construct is fully expressible — `tl.dot` takes any tile shape, and `@triton.autotune` exists
precisely to sweep them. What fails is the *default configuration list* the programmer hands to the
autotuner. Quantified in 2607.04454 (RC2a): at 16384² Triton reaches **59.7%** of cuBLAS because
the shipped matmul-tutorial grid omits `GROUP_SIZE_M`, `num_warps` and `num_stages`; adding an
expanded search recovers it to **81.9%** (1.37×).

**And the Δ = 0pp result next to it is NOT a negative result about autotuning.** 2607.04454 also
reports: "Hardware-agnostic heuristic tuning of the block-tile grid yields **Δ ≈ 0 pp for both DSLs
on every kernel**" — 22 kernels, convolution included, no movement. Quoted alone that says the
technique is worthless. It is not, and the paper reconciles the two findings itself, in the next
paragraph:

> "The tension with the matmul speedup of Section VI-D is a **search-space distinction, not a shape
> distinction**: the 12-config grid varies **only** the block-M/N/K tile dimensions, while the
> expanded search additionally sweeps `GROUP_SIZE_M` (the "L2-swizzle" of RC2a), `num_warps`, and
> `num_stages`."

That expanded search recovers Triton **59.7% → 81.9%** at 16384² and **71.5% → 77.3%** at 4096².
The paper's own verdict, quoted verbatim (RC2a): "**the missing configurations, not an exhausted
budget, bound default performance**" — reinforced in §VI-D, where further iterations (persistent
kernel, `max_num_imprecise_acc`) "converged without gain, so the expanded configuration set and L2
swizzle are the effective ceiling — **confirming RC2: the gain is a property of the expanded search
space, not of a different problem shape.**"

**Read correctly: tile *shape* is insensitive; the axes the default grid omits are the entire
1.37×.** The two results are consistent, and together they are a one-line (ii) diagnosis: *widen the
grid.* Read the other way round, a reader would abandon a technique the paper shows is promising.

Budget is part of the tag. 2607.04454 measures re-tuning one kernel at one deployment shape at
**211 s** vs under a second at a small shape, and concludes exhaustive (DSL × dtype × shape × arch)
coverage is infeasible. Droplet (arXiv 2406.20037) shows the same lever measured differently: better
kernels from **300** Ansor samples than from **10,000**.

*Not a port problem, and explicitly **not** a budget problem — a search-space-coverage problem.*

## 2. Memory-hierarchy placement, and cache blocking across the whole tile grid

**Triton: (ii). tinygrad: (iii) for L2, (ii) for the rest.**

At 16384², A+B+C is ≈1.6 GB against the A100's 40 MB L2. 2607.04454 (RC2b) measures the divide with
counters: **Triton 49.4% L2 hit rate at 85.9% DRAM throughput; cuBLAS 80.5% at 28.8%**, a ≈1.7×
latency penalty (33.5 ms vs 56.1 ms). The named cause is not the tile grid: "cuBLAS employs
hardware-specific cache-blocking (via **cuBLASLt's plan selector**) while Triton's generic `tl.dot`
compilation lacks equivalent heuristics." That is the vendor's hand-written, per-shape
plan-selection heuristics — the part of cuBLAS that is genuinely human and has no compiler analogue.

`GROUP_SIZE_M` grouping recovers part of this, so the *reachable* part is (ii). The *hardware-model-
specific plan selector* part has no Triton construct at all and I would call it (i).

tinygrad's own docs are the strongest statement available on this item: "**we aren't doing a good job
with L2 cache awareness** (the locals handle L1 quite well)." No L2-awareness mechanism is described,
so for tinygrad this is **(iii)**. Compare the neighbouring sentence, which is (ii) — you *can* stage
to SRAM, tinygrad just never does: "we never copy the inputs to on chip SRAM, but this is often quite
helpful for kernel speed."

*Mixed. For tinygrad this is a codegen gap; for cuBLAS-class libraries it is the irreducible
human artefact.*

## 3. Warp cooperation shape — who co-issues a Tensor Core MMA, and how

**Triton: (i). Twill: solved. Tawa: solved. tinygrad: (i)/(iii), unverified.**

Twill §3.2 states the expressiveness failure directly, and it is the sharpest (i) claim in the whole
set:

> "the scale of TCs in recent GPUs requires multiple warps to cooperatively issue large GEMM
> computations. Most GPU compilers are built on **sequential intermediate representations (e.g., LLVM
> and PTX) and are therefore ill-equipped to generate code for cooperative warps**."

Twill's diagnosis is that this is not one problem but four, and the others are listed in §3.2 too:
modulo schedules inflate the working set past the 255-register limit; TMA latency has >1 order of
magnitude of dynamic range so static sync placement is unsound in both directions; and Hopper/
Blackwell fixed-function units need *blocking* sync, which — because threads issue in order —
interrupts concurrent issue (their example: `GEMM.WAIT` killing a concurrent `EXP`).

Twill folds the fix into the same SMT problem; Tawa fixes it with `aref`. **This is the item where the
boundary demonstrably moved within the last ~18 months.**

**Triton has moved too, and it is closing from inside the language — not on a roadmap.** Verified
against the repository, not a search result (this supersedes the earlier "roadmap item" claim and its
"read only as a search result" caveat): Tawa's own footnote 1 states the implementation "has been
upstreamed to OpenAI Triton: `github.com/triton-lang/triton/tree/aref_auto_ws`, with the **first PR
available at #6288**." In the repo, `aref_auto_ws` is a real branch (updated 29 Oct 2025) and **PR
#6288 — "Add an Nvidia Warp Specialization Dialect with an Async Reference Type and Operations"
(mbrookhart, opened 24 Mar 2025) — is MERGED into `triton-lang/main`**, not merely open. It adds the
NVWS dialect and the `aref` IR definitions, crediting the abstraction to Vinod Grover, Alexander
Collins and Bin Fan, who are Tawa authors.

**Scope, so the tag stays honest:** what landed in `main` is the **IR infrastructure**. The PR's own
closing line scopes the rest as future work: "Future work will focus on lowering to standard ttg/ttng
types along with higher level passes for warp specialization analysis." The user-facing automated flow
(Tawa §III-A: `enable_warp_specialization=True`, Hopper and newer) is **not** in a release. So the tag
stays **(iii)** — but "(iii) per its roadmap" is materially wrong, and the reader must not conclude
the capability is speculative. It has an in-tree implementation under active development in the
language itself, and two independent research systems converging on it.

*Was (i) at the IR level; is now automated by two research systems, and is being absorbed in-tree by
the language whose IR was the original example of the failure.*

## 4. MMA fragment layout and register-file mapping

**Triton: automated. TinyGrad: (iii), unverified. Twill: expressed as a constraint.**

Triton lists "tensor core-aware instruction selection" as compiler-automatic, and a user does not
hand-write fragment layouts in Triton. Twill makes the register mapping an explicit constraint
family (Register Limit, Cross-Warp Spills, per-warp liveness), which is the right way to expose it.

tinygrad's docs are notably weak here and I will not overstate them: "**We have a simple framework
in tinygrad for adding these ALU blocks and achieving good performance from them.**" That is a
statement about the existence of a mechanism, not about matching CUTLASS. **I could not verify a
single measured claim about tinygrad's Tensor Core performance from any source.** Tag it (iii) on the
strength of the self-description alone, and treat that as untested.

*The failure mode here is RC3 (spill), which is item 7.*

## 5. Accumulate dtype vs I/O dtype

**Triton: (ii). TileLang: (iii) — it silently changed the user's semantics. tinygrad: (iii).**

Two distinct problems, and the split matters.

2607.04454 §VI-B: TileLang LayerNorm ran at 364 ms; **two** fixes took it to 0.270 ms — replacing
`T.serial` with `T.reduce` (item 13), and using **native bf16/fp16 I/O with no intermediate
`.float()` casts**. The paper is explicit that the first is user-space and the second is not: "the
RC0b codegen deficiency is **sidestepped, not fixed**."

RC0b is the pure (ii) case: TileLang's compiled norm kernel "fetch[es] 16-bit bfloat16 elements
individually instead of issuing 128-bit LDG.128 loads," with NCU confirming 50% bytes/sector at one
sector per request. The data was already 16-byte aligned. Nothing was missing from the language; the
codegen declined to vectorize.

The (iii) case is a correctness failure, not a speed one: "**T.gemm silently lowers float32 to TF32 on
SM80**, failing an exact 1e-5 check on **31.8% of outputs** while a manual FMA kernel passes." A
compiler that cannot be asked to keep a precision is (i) for precision control and (iii) for the
silence.

For tinygrad, dtype policy lives in the graph (`Tensor.default_float`, per-op dtypes), so "don't
materialize the fp32 intermediate" is a *graph-level* decision, and the docs admit the family is
unhandled: "once you add movement ops and casts this is even harder to figure out. **tinygrad doesn't
yet have a systematic way to do it.**"

## 6. Online / reciprocal algorithms that avoid materialising an intermediate

**Every DSL in this study: (i). This is the firmest (i) in the taxonomy.**

Triton, TileLang and tinygrad all take a *function* and compile it. If the function materialises the
N×N score matrix, no schedule removes it — the compiler optimises over a fixed arithmetic-intensity
budget. 2607.04454 refuses to even tabulate across this boundary, which is the authors implicitly
declaring it out of scope for schedule search:

> "the Triton (tiled FlashAttention-2) and TileLang (untiled O(n²)) kernels are **algorithmically
> different**"; "because the Triton and TileLang implementations are not algorithmically identical to
> this baseline, we report attention results separately."

And within a *single* operator the fix is still an algorithm change, not a schedule: repairing the
TileLang softmax family required "streaming the reduction dimension in tiles (**an online,
max-rescaled
pass**) rather than materializing the full row in a fragment." The compiler had no online-softmax
rewrite; the optimizer wrote one.

The two-sided proof is the in-tree TileLang Softmax that "caches the entire row in shared memory" —
**20× slower than PyTorch at N=8192 and 111× at N=32768, while numerically correct.** The language
happily let a human make a bad *algorithmic* choice, and the compiler's opinion of the result was
irrelevant. Authoring choice, not compiler quality, decided 111×.

Algorithm diversity as a *search-system* concern is (iii): Ansor needed a **new derivation rule** for
Winograd and for TensorCore intrinsics ("some special algorithms … require special tile structures to
be effective"). 2607.04454 then *measured* the prize: cuDNN's Winograd F(2,3) cuts MACs ≈2.25×, but a
deterministic A/B on 3×3 stride-1 differs by **0.03%**, so absent Winograd selection accounts for at
most **≈2–3%** of the conv gap — and the gap "does not track Winograd eligibility" (3×3 stride-1 at
28.5% is no better than 3×3 stride-2 at 32.1%; the worst are 5×5 at 17.8% and 7×7 at 13.1%).

**Important correction to the framing of this item.** The literature often implies algorithm
diversity is where the big wins are. For convolution on Ampere it is not — it is worth 2–3%, and the
residual is "general cuDNN implicit-GEMM tuning — kernel selection, split-K, shared-memory tile
sizing." The big algorithmic win in the whole corpus is **avoiding the intermediate** (materialised
reductions/softmax rows), and that is a fusion/streaming decision at the graph level. Note also
Saha & Ye (PMLR v235): FlashAttention is IO-optimal only for fast-memory capacity ≥ d², with a
different regime below — so even here "closed" is conditional.

*This is the one item where the answer is genuinely "a human must port the algorithm."*

## 7. Occupancy and register pressure

**Triton: (ii). TileLang: (ii) for the large-norm case, (iii)-flavoured for the cross-arch case. tinygrad: (ii).**

Register allocation is automatic; the *knob* that causes spills is a user-chosen fragment width, so
this is (ii) — reachable by tuning. RC3 quantifies it: the TileLang LayerNorm kernel used **254
registers/thread, 12.4% achieved occupancy, and spilled 51.5 GB of local-load + 34.4 GB of
local-store** on A100 (GH200 mirrors at 45.1/34.4 GB).

Two findings here are more important than the number:

**(a) The obvious hypothesis is refuted.** "A natural hypothesis is that large-filter convolution hits
this wall; NCU refutes it: Triton conv2d reports **n_spills = 0 at every tested filter** (1×1 through
7×7), even though register usage rises with K² (up to 224 regs/thread at large shapes)." The conv gap
is occupancy- and coalescing-bound, not spill-bound. Do not assume spills explain a slow conv.

**(b) Occupancy is a per-architecture quantity, and re-tuning does not always fix it.** The
`logsumexp` kernel is the study's best cautionary result: A100-tuned, 0 spill, 93 regs, 582% of
PyTorch — and on GH200 it register-spills (255 regs, ≈280 GB of spill traffic, 12% occupancy) and
collapses to **1.0%**. Retuning the fragment width recovers it only to **63%**. An "optimized" kernel
was 100× off on another GPU **with no correctness signal**. tinygrad's docs anticipate the mechanism:
"if there's too much register pressure on the GPU the warp scheduler may not be able to fit many
warps, or even worse, it could be spilling to local memory."

## 8. Vectorised access for strided patterns

**Triton: (ii) for the analysis failure, (i) for NCHW convolution. TileLang: (ii). tinygrad: (iii), unverified.**

This is the item where the (i)/(ii) split is sharpest *within one paper*, and 2607.04454 gets it
right in a way that is worth copying.

The capability exists: Ampere reaches peak bandwidth only via 128-bit LDG.128, and Triton emits them
for GEMM because the layout is row-major and tiles are 16-byte aligned. Convolution breaks this in
**two different ways**, and they have different tags.

*Break 1 — the analysis fails (ii).* "for KH×KW>1 Triton's **alias analysis cannot prove** successive
spatial-loop iterations are aligned, so it **conservatively emits scalar 32-bit loads**." Same
language, same PTX target, reachable by a better analysis. The consequence cascades: "un-vectorized
loads prevent `cp.async` from firing, so the software pipeline cannot overlap data movement with
computation and the kernel stalls on global-memory latency at every tile boundary." NCU: **12.55%
load efficiency at 16.4 sectors per request** for Triton conv2d. The authors classify the fix as "a
**code-generation fix, not an API change**" — the textbook (ii).

*Break 2 — the layout makes it impossible (i).* "each output gathers a (KH×KW) window strided across
the innermost NCHW dimension (PyTorch's default), **breaking LDG.128 alignment**." No analysis fix
helps; a human has to change the data layout. And indeed the repair was a port: "restructuring Conv2d
as an FP16 Tensor-Core **implicit GEMM with aligned padding** (RC1) and an expanded autotune space
(RC2) reaches **36–77%** of cuDNN across 1×1–7×7 on the A100 — narrowing but not closing the gap."

Depthwise convolution is the clearest still-open case in the whole corpus: "its grouped lowering
degenerates into **512 per-group GEMM launches** whose cost is launch- and code-generation-bound …
none of our patterns repair it. We flag it as **the clearest open convolution gap**." Both DSLs fall
to **≤5.5%** of cuDNN.

Contrast RC0b (item 5): TileLang failed to vectorise *already-aligned* data. Same symptom, different
tag, because there was nothing for a human to fix.

## 9. Software pipelining depth

**Triton: (ii) via `num_stages`. TileLang: (i) by construction. Twill: solved optimally. tinygrad: (ii).**

Triton performs software pipelining automatically; the depth is a user parameter swept by
`@triton.autotune`. 2607.04454 RC2a puts a number on what the default gets wrong: sweeping `num_stages`
along with the L2 swizzle recovers Triton 59.7%→81.9%.

TileLang made this a deliberate architectural choice with a different tag: "TileLang's
**ahead-of-time** `num_stages` **cannot adapt** to the register pressure large filters create."
Compile-time-constant by design — that is (i) for the *adaptivity*, even though the knob is
expressible.

Twill is the item's resolution, and the finding is sharper than "it automates the depth." Twill
**proves** the optimum: SWP reduces to modulo scheduling, which is solvable to minimal initiation
interval; the SMT formulation returns M\*, I\* and a warp assignment A\*. Applying it to a tile-level
description of attention with Hopper machine costs "**yields the exact pipeline developed by experts
in Flash Attention 3** from only a high-level description of the algorithm and machine." And on the
human side it found something the experts did not: "Twill discovers the software pipeline used in FA3,
but applies it only to **one warp group**, determining this to be sufficient to saturate the
functional units." Implemented schedules come "**within 1%** of hand-tuned cuDNN and FlashAttention"
on Hopper and Blackwell, having also matched FA4's.

Why a static depth is not a safe abstraction — Twill §3.2, factor 3: TMA transfer latency "may have
more than an order of magnitude difference between the fastest and slowest possible execution times
for the same transfer … overestimation results in under-utilization, while underestimation results in
pipeline stalls."

## 10. Warp specialization roles (producer / consumer / MMA-issuing / epilogue)

**Triton: (iii), but closing from inside the language (aref/NVWS merged into `main`, PR #6288; the
user-facing automated flow is not yet in a release). Twill: solved. Tawa: solved. TinyGrad:
(i)/(iii), unverified.**

Twill names both the status and the specific defect of prior work:

> "Currently, all automated approaches that we are aware of depend upon **ad-hoc heuristics** to
> decide these trade-offs. These heuristics often involve canonical warp 'roles' (e.g., loader and
> compute warps) … Without a technical framework for understanding the optimality of these heuristics,
> it is difficult to know if programs generated using them are achieving peak performance."

So pre-Twill, the honest tag is **not (i)** — the roles are expressible and generated — it is (ii) with
no optimality guarantee. Twill converts it to a *provably* optimal assignment, and states the ordering
constraint that makes the joint problem necessary: "even with WS, it might be impossible to synthesize
a program that actually achieves a modulo schedule … treating the computation of the modulo schedule
and code generation using WS as two independent steps in compiling a program can lead to sub-optimal
code."

Tawa's numbers are the outcome: **1.2× over Triton** on attention, **matching hand-optimized CUTLASS
C++ FlashAttention-3**, **up to 1.1× over cuBLAS GEMM** on H100. Its attribution ablation is what
makes the claim useful: without warp specialization the same tile-based program reaches only **104
TFLOP/s**.

Triton is the interesting cell: warp specialization used to be a published design-and-roadmap
document, i.e. a bare **(iii)**. It is no longer that. The `aref`/NVWS dialect is **merged into Triton
`main` via PR #6288** on the `aref_auto_ws` branch line, so the (iii) is **closing from inside the
language** — with the higher-level passes still to come, per the PR's own scope statement.

## 11. Cross-kernel scheduling: recompute vs store, and fusion

**Nobody has automated this. tinygrad: (iii), and says so. Triton/Inductor: (ii).**

Not in your list, but it is the item that most shapes a TinyGrad-vs-hand-written study, and tinygrad
names it as its own worst bottleneck. From `docs/developer/speed.md`:

> "The scheduler determines how operations are grouped into kernels and which Tensors are written to
> memory. **This is currently a big bottleneck of training speed.** The decisions are often not
> obvious. For example, when is it worth recomputing an arithmetic operation instead of storing and
> loading from memory? … In this case: with recompute (6 reads + 2 writes), no recompute (6 reads +
> 3 writes), so we should probably recompute. **However, once you add movement ops and casts this is
> even harder to figure out. tinygrad doesn't yet have a systematic way to do it.**"

Note the shape of the admission: the *arithmetic* is trivial and a human gets it right; the
*difficulty is the surrounding context*. That is the signature of a **graph-level** decision, and it
explains why 2607.04454's surviving residuals are all at that level (cuDNN implicit-GEMM's split-K
and smem tile sizing; cuBLASLt's plan selector) while its per-kernel gaps closed.

2607.04454 also shows fusion is where a DSL legitimately *wins*, and it is careful about the
mechanism: TileLang hits **468.5%** of PyTorch on fused linear+activation, and the paper attributes
this to "kernel design, not a compiler artifact — the single launch avoids PyTorch's intermediate
allocation — and says nothing about shapes with no second operation to fuse." It simultaneously flags
the *reverse* trap, and the flag covers **three** cells, not one:

- **873%** (Triton rms_norm, Table I) is excluded outright as an "unfused-baseline artifact" / "an
  unfair-baseline artifact."
- **990% (A100)** and **1303% (GH200)** — TileLang rms_norm, **Table IV** — carry the *same* mark.
  Table IV's footnote † covers both identically: "RMSNorm Elib>100%: the fixed kernel fuses the
  scaling pass PyTorch's eager path leaves unfused." §VI-B says the same in prose.
- Note these are the *post-fix* values, so they are doubly not a capability claim: the >100% is
  created by a fusion the eager baseline does not perform, and it appears only after a fix that is
  about reduction authoring.

**Do not reach for 990% or 1303% as evidence of a capability gap.** They are evidence of a baseline
defect. The roofline anchor exists for exactly this: the paper's own de-inflation of this class puts
the kernel at `ρ = 0.69` — efficient, but hardly beyond what the hardware allows.

## 12. Summary table

| # | Decision | Triton | TileLang | tinygrad | Twill | Tawa |
|---|---|---|---|---|---|---|
| 1 | Tile sizes | (ii) | (ii) | (ii) | — | — |
| 2 | Memory-hierarchy / L2 blocking | (ii) | (ii) | (iii) L2 | — | — |
| 3 | Warp cooperation shape (co-issuing MMA) | (i) | (i) | (i)/(iii) | **solved** | **solved** |
| 4 | MMA fragment layout | auto | auto | (iii) unverified | constraint | (iii) |
| 5 | Accumulate vs I/O dtype | (ii) | (ii)+(iii)¹ | (iii) | — | — |
| 6 | Online / no-intermediate algorithm | **(i)** | **(i)** | **(i)** | (i) | (i) |
| 7 | Occupancy / register pressure | (ii) | (ii) | (ii) | constraint | (ii) |
| 8 | Vectorised strided access | (ii) *and* (i)² | (ii) | (iii) unverified | — | — |
| 9 | Software pipelining depth | (ii) | (i) AOT | (ii) | **solved, optimal** | **solved** |
| 10 | Warp specialization roles | (iii), **closing in-tree**³ | (i) | (i)/(iii) | **solved, optimal** | **solved** |
| 11 | Recompute vs store / cross-kernel fusion | (ii) | (ii) | **(iii)**, self-admitted | — | — |

¹ TileLang: (ii) for failing to vectorise aligned data (RC0b); (iii) for silently lowering fp32→TF32.
² Two different causes of one symptom: alias-analysis failure on a *layout-fixable* pattern is (ii);
gathering a (KH×KW) window across the innermost NCHW dimension is (i) until someone ports it to
implicit GEMM with padding.
³ Verified in the repository, not from a roadmap document: Tawa's aref/NVWS dialect is merged into
Triton `main` (PR #6288, first PR on the `aref_auto_ws` line). The tag is still (iii) because the
higher-level warp-specialization passes are scoped as future work in that PR — but "(iii) per its
roadmap" would be wrong.

---

# C2 — What IS automated now, and by how much

Ordered by how well-quantified the automation is.

**Fully automated, no human in the loop (Triton's own list, Programming Guide ch. 1):**
automatic coalescing, thread swizzling, pre-fetching, **automatic vectorization**, tensor
core-aware instruction selection, shared memory allocation/synchronization, **asynchronous copy
scheduling**. Halide adds the polyhedral machinery: loop interchange, distribution, fusion,
vectorization, storage padding, parallelism. Inductor adds pointwise-chain fusion via its
loop-level IR.

**Automated, and measured — at parity or better. Read the result column, not the section title: one
entry below (arXiv 2012.07145) is a GPU *tie*, which is the honest ceiling on GPU auto-scheduling, not
nearness.**

| Capability | Result | Source |
|---|---|---|
| Parallel reduction (once `T.reduce` is used) | TileLang LayerNorm **124%** / RMSNorm **990%** of PyTorch (A100); LayerNorm **120%**, family **84–1303%** (GH200) — **but the RMSNorm cells (990%, 1303%, and 873% in Table I) are fusion artifacts of an unfused eager baseline and are excluded as capability evidence; see item 11** | 2607.04454 §VI-B/D, Tables III–IV, footnotes † and ‡ |
| Mean/max reduction | **99.5%** / **131.6%** of PyTorch | 2607.04454 Table IV |
| Tile autotuning (general) | Triton overall median **≈98%** of PyTorch/cuBLAS across 22 non-attention kernels | 2607.04454 Table I |
| Tile autotuning (general) | TileLang overall median **≈68%** (A100), **≈67%** (GH200) | 2607.04454 Table I |
| Triton vs cuBLAS on GEMM | **59.7–97.9%** A100; worst at 16384², best (97.9%) on the batched 128×2048² shape | 2607.04454 Table I |
| Schedule search efficiency (vs AutoTVM) | matches it with **8.6×–88.6×** less wall-clock time; **2.7×–16.5×** fewer measurements | Ansor Table 3 |
| Search quality (vs Ansor) | within **2.2%** geomean of Ansor-10K using **10× fewer** measurements; **1.37×** faster than Ansor at a matched 64-trial budget | arXiv 2606.09312 (abstract) |
| Learned schedule vs *previous* autoscheduler | "almost twice as fast" without autotuning, "more than twice" with; "first automatic scheduling algorithm to **significantly outperform human experts on average**" — **CPU image/deep-learning pipelines, SIGGRAPH 2019, pre-Tensor-Core. Do not read this row as a GPU result; the GPU ceiling is the tie in the row above** | Adams et al. 2019, authors' abstract |
| Learned schedule vs *expert* hand-scheduling on GPU | **A TIE, not nearness: 0.95×.** "Matches experts' best effort to beat it with unconstrained manual schedules," using the entire scheduling language, against experts given the autoscheduler's own best runtime as a target to beat. §6: "on par with our best known manual schedules (0.95×)" — nominally 5% *slower*. The paper's 1.7× is a win over the previous *autoscheduler* (Sioutas 2020), not over humans | arXiv 2012.07145 §1/§6/§6.5/§9 |
| Learned schedule vs hand-tuned vendor libraries | "performance competitive with state-of-the-art hand-tuned libraries" on low-power CPU, mobile GPU, server GPU | arXiv 1805.08166 (abstract) |
| SWP + WS joint optimization | provably optimal schedules; **rediscovers FA3 and FA4**; implemented schedules "**within 1%** of hand-tuned cuDNN and FlashAttention" | Twill §6, abstract |
| Warp specialization from a tile program | **1.2×** Triton; **matches CUTLASS FA3**; **up to 1.1×** cuBLAS GEMM (H100) | Tawa, abstract |
| Autotuning budget | better kernels from **300** samples than Ansor's **10,000** | arXiv 2406.20037 (abstract) |
| Auto-schedule reuse | **88.41%** (49.13% avg) of Ansor's best speedup, with Ansor needing **6.5×** more search time | arXiv 2201.05587 (abstract) |
| Roofline-anchored efficiency *measurement* | already a first-class metric in TritonBench | arXiv 2502.14752 |

**Not automated, and the honest reasons:**

| Capability | Status | Evidence |
|---|---|---|
| Algorithm selection (Winograd, implicit GEMM, split-K) | (i) in the DSL; needs a **new search rule**; and on Ampere conv it is worth only **≈2–3%** | Ansor §4.1; 2607.04454 RC4 |
| Recompute-vs-store, cross-kernel scheduling | (iii), self-admitted as "a big bottleneck of training speed" | tinygrad `docs/developer/speed.md` |
| Hardware-model-specific plan selection (cuBLASLt-style) | no analogue in any DSL studied; the residual after everything else | 2607.04454 RC2b, §VI-C |
| Cross-architecture schedule reuse | broken in practice: an A100-optimized kernel ran **100×** off on GH200 with no correctness signal; retuning recovered only to 63% | 2607.04454 §VI-D |
| TMA / Tensor Memory Accelerator | (i) in tinygrad: "not supported in tinygrad yet" | tinygrad docs |
| **Correctness-oracle robustness** | 16.9% of witnessed faults escape KernelBench's official check; **78.6% of precision faults** escape; 4 of 60 L1 problems accept an all-zeros tensor; one kernel scored **283×** while writing 0.3% of its output | arXiv 2609.22220, 2609.21058, 2607.16241 |
| **Larger-scale work on the same operator class** | **zero papers on arXiv mention tinygrad** | arXiv API query, `totalResults = 0` |

---

# Where the boundary sits now, and whether it is closing

## The claim

**The boundary is not "compiler versus human." It is "static machine model versus adaptive search."**

Every thing the studied compilers do well is a consequence of a **static, closed model of the
target**: functional-unit capacities and costs, a register file size, a shared-memory budget, and a
roofline peak. Coalescing, vectorisation, swizzling, tiling and pipelining are all derivable from that
model plus the program.

Every place a human wins is outside that model, and the sources sort cleanly into four buckets:

1. **An algorithm that changes arithmetic intensity** — materialising vs streaming vs online.
   Unreachable by any schedule. (i) for all DSLs.
2. **A layout change that makes an access pattern provably aligned** — NCHW conv to implicit GEMM
   with padding. Needs a port. (i) until a human ports it, then (ii).
3. **Hardware-model-specific plan selection** — cuBLASLt's per-shape algorithm choice, cuDNN's
   split-K and smem tile sizing. This is the residue that survives every search-based fix, because
   it is *the human library's own schedule selector*. No DSL has an analogue.
4. **Search budget allocated over a real shape distribution** — 211 s to re-tune one kernel at one
   deployment shape; the (DSL × dtype × shape × arch) space is explicitly declared infeasible to
   cover.

## Is it closing? Yes, on a specific front, and the front is identifiable.

**It is closing for decisions that are simultaneously (a) local to one kernel, (b) enumerable in a
small action space, and (c) scoreable by a cost model or a cheap measurement.** That is exactly the
class Twill and Tawa attack, and they succeed: Twill proves optimality and recovers FA3/FA4 from a
tile-level description; Tawa matches hand-written CUTLASS from a tile program. The convergence is
real and recent — Twill (Dec 2025) and Tawa (Oct 2025) postdate every "compilers can't do this"
system in this set.

**It is not closing, and there is no mechanism that would close it, for the four buckets above.** The
strongest evidence is that the residuals 2607.04454 could not remove are precisely where a hand-written
library still wins: "The *genuine residuals* — Conv2d (42%), large square GEMM (77%), and
index-returning Argmax (27%) — survive best-effort optimization because cuBLAS, cuDNN, and
`torch.argmax` are themselves well-tuned on the data-center A100; of the Conv2d gap only ≈2–3% is
absent Winograd selection." Note the framing: the baseline is a human, so a residual against it is a
floor, not a defect.

## Three results that should discipline your study's claims

1. **The measured boundary moves when you fix the baseline.** KernelBench's best model reports
   **1.43×** geometric mean under its own protocol and **0.88×** under a TF32-enabled baseline with
   four hidden input distributions. "No model consistently outperforms PyTorch when evaluated against
   realistic baselines." A "compiled kernel is 1.4× faster" claim is a claim about the *baseline*.

2. **A kernel-level win may be irrelevant end to end.** On transformers, **80–86% of runtime is
   cuBLAS GEMM + FlashAttention**, bounding realistic end-to-end improvement at **≈1%** and shrinking
   with scale. On a recommender, 58.2% is addressable and 41.7% of problems can be won at 1.552×
   median → **8.63%** end-to-end. If your study reports only kernel speedups it is answering a
   question nobody asked.

3. **"Optimized" is architecture-indexed.** logsumexp, tuned and correct on A100, spills on sm_90 and
   runs 100× slower, with no correctness signal. Any single-GPU result — including anything you
   measure — is a statement about one cell of the grid the field has declared infeasible.

## The specific bet, and where a TinyGrad study has an opening

For TinyGrad specifically, the boundary is **lower on the stack** than for Triton or TileLang, and its
own documentation says so. It sorts "speed" into four layers and locates the bottlenecks:

- *Execution/driver* — "almost never the bottleneck." The claim "tinygrad has the best execution
  speed of any framework because it usually **bypasses the GPU driver and prebuilds the command
  queue**" is about **launch overhead, not kernel throughput**, and is the single most likely thing
  to be mis-cited. Do not let it stand in for a kernel-speed claim.
- *Codegen* — self-admitted: no SRAM staging, poor L2 awareness, a "simple framework" for tensor
  cores, no TMA, and BEAM's compile cost makes compile speed merely "on par with torch.compile."
- *Graph scheduling* — "**a big bottleneck of training speed**," and recompute-vs-store has "no
  systematic way" to be decided.

So the ordering of expected gaps is: **graph scheduling > codegen (SRAM/L2/tensor-core/TMA) > nothing
at the launch layer.** That is a prediction you can state in advance and then falsify, which is worth
more than another confirmation of "TinyGrad is slower."

The author's own thesis is that the remaining gap is a **search** gap, not an expressiveness gap —
"expose the underlying search problem," "fully capture all dimensions of the search space," and then
"Burn compute. Use LLMs to guide. **Use SAT solvers.** Reinforcement learning." That is the same
technique Twill uses and it reaches the same place. It is also, notably, an untested bet stated as a
refutation of a hypothesis rather than a measurement: "One hypothesis is that tinygrad is only smaller
because it's not speed or feature competitive … But I just don't think that's true."

**And the opening:** arXiv contains **zero** papers mentioning tinygrad (`totalResults = 0`). There is
no published TinyGrad performance number, no TinyGrad-vs-Triton comparison, and no measured
evaluation of the 14,556-line / "a model for the hardware is simple compared to a model for CUDA"
claim. That is the gap your study fills, and the reason none of the numbers above can be borrowed.
