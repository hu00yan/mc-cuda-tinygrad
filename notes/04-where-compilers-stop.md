# Where GPU compilers stop — and what that means for this study

Answers the question: *what does a human still have to supply that a compiler
will not infer, and is there existing work I should be using instead of building
my own?*

Short version: **the question is well-posed and unusually well-studied, and the
boundary is not where I had it.** I had been writing "a static graph compiler
cannot express warp cooperation" as though it were a law. It is not a law, it is
a statement about 2020-era IRs, and two 2025 systems have already converted most
of it into a solved constraint problem. The correct boundary is **static machine
model versus adaptive search** — a sharper claim, and one that also explains
which parts of my own findings are TinyGrad's fault and which are nobody's.

Revised after `research/SOURCES.md`, `research/TAXONOMY.md` and
`research/PRIORWORK.md`. Every number below is marked **[abstract]** if it comes
from a landing page rather than a table I read, and **[unverified]** if the
research agent could not open the source.

---

## 1. The framing I was using is wrong, and the replacement is sharper

The law I had been leaning on:

> a static graph compiler cannot express warp cooperation

It is not a law. It is a claim about sequential IRs, and it was already false
when Twill and Tawa were written. The defensible version:

> **The boundary is not "compiler versus human". It is "static machine model
> versus adaptive search".**

Two consequences follow, and they are the whole content of this note.

**Everything the studied compilers do well follows from a closed model of the
target.** Functional-unit capacities and costs, a register-file size, a
shared-memory budget, a roofline peak. Coalescing, vectorisation, swizzling,
tiling and pipelining are all derivable from that model plus the program. This
is why Triton's own list of what it automates is entirely local-intent work
(Triton Programming Guide, ch. 1): "automatic coalescing, thread swizzling,
pre-fetching, **automatic vectorization**, tensor core-aware instruction
selection, shared memory allocation/synchronization, **asynchronous copy
scheduling**".

**Everything a human wins on lies outside that model.** The sources sort into
four buckets, and only the first three have any prospect of closing:

1. **An algorithm that changes arithmetic intensity** — materialise vs stream vs
   online. Unreachable by any schedule. (i) for every DSL.
2. **A layout change that makes an access pattern provably aligned** — NCHW conv
   to implicit GEMM with padding. Needs a port. (i) until someone ports it.
3. **Hardware-model-specific plan selection** — cuBLASLt's per-shape algorithm
   choice, cuDNN's split-K and shared-memory tile sizing. This is *the human
   library's own schedule selector*, and no DSL has an analogue.
4. **Search budget allocated over a real shape distribution** — 211 s to re-tune
   one kernel at one deployment shape; the (DSL × dtype × shape × arch) grid is
   explicitly declared infeasible to cover.

### 1a. The honest counterweight, which the note was missing

**arXiv 2012.07145** — Anderson, Adams, Ma, Li, Jin, Ragan-Kelley, "Efficient
Automatic Scheduling of Imaging and Vision Pipelines for the GPU". 17
imaging/vision programs on GPU. This is the single strongest piece of evidence
that "compilers can't" is overstated, and it lands in the *opposite* direction
from what I expected:

> It generates code that **matches experts' best effort to beat it with
> unconstrained manual schedules**, and significantly outperforms the current
> state-of-the-art Halide GPU autoscheduler.

The bar is deliberately maximal, and the detail that makes this the honest
version is the experimental design: the experts used "the entire scheduling
language" and "iteratively improved these schedules during the course of this
work using **the best runtimes found by the autoscheduler as a target to beat**
(but without looking at the generated schedules), so they represent a very high
bar". The abstract's phrasing: "competitive with what the best human experts
were able to achieve in an active effort to beat our automatic results".
**[abstract]**

**The stated reason is search-space expressiveness, not search quality.** The
predecessor (Sioutas et al. 2020) "only supports a single level of tiling, and
as we discuss in Section 6.2, this excludes a number of high performance
schedules". So a bigger action space converted a loss into a tie, with the
algorithm held fixed.

**So parity, not victory, is the ceiling on GPU auto-scheduling** — and it was
reached in 2012. Two caveats I must not forget: it is Halide (a pure algorithm/
schedule language), not a tile DSL, and it is pre-Tensor-Core in the
pre-Halide-autoscheduler lineage. The comparable "beat the humans on average"
result (Adams et al., SIGGRAPH 2019) is **CPU** image/deep-learning pipelines,
and must not be generalised to modern GPUs. Hidet (ASPLOS 2023) makes the same
point from the other side: TVM's primitive set is "insufficient to cover
state-of-the-art tensor program optimizations", and widening it gives 1.22×
average **[abstract]** — an expressiveness fix, not a search fix.

**What this means for my framing.** I was treating a human's wins as evidence
of a compiler's limits. The counterweight says: on GPU, against experts given
the autoscheduler's own runtime as a target, a scheduler *ties*. What is left is
the four buckets above, and the largest of them is a library's own hand-written
plan selector. That is a much more defensible thing to write a study about than
"compilers can't".

---

## 2. The prior art, with the numbers fixed

### 2a. "Correct but Slow" — arXiv 2607.04454

Li, Rathnasuriya, Yang (UT Dallas), v3 13 Jul 2026. 22 Triton + TileLang
kernels, 5 operator categories, A100-SXM4-40GB and GH200, against the strongest
library path through PyTorch (cuBLAS / cuDNN / SDPA+FlashAttention). The single
best-designed empirical answer to "where do compilers stop", and the one whose
methodology I should copy.

**Correcting the 1520× figure, which I had badly mis-described.** 1520× *is* in
the paper, but it is one cell of Table III, not a headline:

| what | value | what it actually is |
|---|---|---|
| **the headline** | **>300×** | the abstract's own words: an idiomatic TileLang LayerNorm "passes KernelBench's correctness check while running more than 300× slower than the PyTorch baseline". 323× locked on GH200 at the large shape, **51.5 ms vs 0.16 ms** |
| **1520×** | Table III, GH200 column | the **naive → optimized cliff for TileLang RMSNorm**. It sits in Table III's *recovered family* block — the paper's own verdict is that one authoring pattern repairs it |
| LayerNorm cliff | **1541×** GH200, **380×** A100 tuned, **1347×** A100 untuned | naive → optimized, 364 → 0.270 ms locked |
| A100 LayerNorm vs PyTorch | **299×** | 97.0 ms vs 0.324 ms, tuned suite profile |
| TileLang RMSNorm, GH200 vs PyTorch | 117× | the paired cell to the headline |
| worst-case constructed variant | 1293× (176.4 ms, unlocked) | |
| naive argmax | passes, 16.7× slower | |

I had been quoting the RMSNorm cliff cell as if it were the paper's worst case.
It is neither the headline nor the worst case, and quoting it that way
overstated the paper in a way that would not survive review.

The methodological point survives intact and is the reason this paper matters:
**correctness admits ~300× slowdowns.** Every kernel in their suite passes
strict numerical validation. That is exactly the trap my `test_tiny.py` (6/6
passing) and `run_oc.py` (0/41,472 mismatches) fell into — I was treating
"correct" as the finish line, and this paper says the finish line is a roofline
measurement. Their two screening checks (library-relative efficiency and
roofline utilisation) "together flag every functionally valid but inefficient
kernel in our suite" and separate repairable authoring defects from structural
residuals.

Its root-cause taxonomy, empirically attributed with a counter signature each:

| code | cause | contribution | diagnostic counter |
|---|---|---|---|
| RC0a | `T.serial` not converted to `T.reduce` | **1347×** (LayerNorm, A100) | `warp_stall_long_scoreboard` (barrier ≈ 0, so not a barrier problem) |
| RC0b | absent vectorised loads / dtype casts | — | `l1tex` bytes-per-sector: **50%**, 1 sector/request |
| RC1 | strided access → scalar LD, so `cp.async` never fires | **2.2×** | load efficiency **12.55%** at 16.4 sectors/request |
| RC2a | autotuning coverage — configs missing from the default grid | **1.37×** (Matmul) | TFLOP/s vs the swept config set |
| RC2b | L2 residency | — | Triton **49.4%** hit / 85.9% DRAM vs cuBLAS **80.5%** / 28.8% |
| RC3 | register spill collapsing occupancy | — | TileLang LayerNorm: **254 regs, 12.4% occupancy, 51.5 GB local-load + 34.4 GB local-store** |
| RC4 | absent Winograd | **≈2–3%** | deterministic-vs-not A/B on 3×3 s1 |

### 2b. Twill — arXiv 2512.18134 (not the Stanford PDF I linked before)

**Rupanshu Soi, Rohan Yadav, Fredrik Kjolstad, Alex Aiken, Maryam Mehri Dehnavi,
Michael Garland, Michael Bauer** — Stanford and NVIDIA. "Optimal Software
Pipelining and Warp Specialization for Tensor Core GPUs", cs.PL, 19 Dec 2025.
**[abstract]**

I had cited this as `compilers.stanford.edu/publications/twill.pdf`. That PDF
was never verified (the fetch was blocked) and the arXiv ID was missing, so the
citation was unusable. Correct one: **arXiv 2512.18134**. Note the author list —
Soi, Yadav, Kjolstad, Aiken are the Stanford side; Mehri Dehnavi, Garland and
Bauer are NVIDIA.

What it establishes, and why it is the sharpest (i)-claim in the literature
*and* its own refutation:

- Reduces software pipelining to **modulo scheduling** (solvable to minimal
  initiation interval) and folds warp specialisation into the *same* constraint
  problem, solved by off-the-shelf SMT solvers. Heuristic-free,
  architecture-portable by swapping the machine costs. Solved the Flash
  Attention instance in **28 seconds**.
- Its own diagnosis of the status quo, from the abstract: determining how best to
  use SWP and WS together "is a challenging problem that is currently handled
  through a mix of **brittle compilation heuristics and fallible human
  intuition, with little insight into the space of solutions**."
- The sharpest (i) statement available, and the one I had misattributed to
  Tawa: "the scale of TCs in recent GPUs requires multiple warps to cooperatively
  issue large GEMM computations. Most GPU compilers are built on **sequential
  intermediate representations (e.g., LLVM and PTX) and are therefore
  ill-equipped to generate code for cooperative warps**." Plus three more, in
  §3.2: modulo schedules inflate the working set past the **255-register**
  limit; TMA transfer latency has **>1 order of magnitude** of dynamic range so
  static sync placement is unsound in both directions; Hopper/Blackwell
  fixed-function units need *blocking* sync, which — because threads issue in
  order — interrupts concurrent issue.
- The result: it **rediscovers, and thereby proves optimal**, the SWP and WS
  schedules that experts developed by hand for Flash Attention on both Hopper and
  Blackwell, and implemented schedules "come **within 1%** of the performance of
  hand-tuned implementations from libraries like cuDNN and Flash Attention".
- Its FA3 anecdote is the cost of the current state: "a year transpired between
  the release of Hopper and the development of Flash Attention 3, which proposed
  a custom SWP schedule and WS strategy for Hopper, **wasting countless compute
  cycles on critical inference workloads in the interim**."
- **The result I had missed, and it is the one that matters:** "Twill discovers
  the software pipeline used in FA3, but applies it **only to one warp group**,
  determining this to be sufficient to saturate the functional units." That is
  an *automated* determination of pipeline depth, and the experts did not make
  it. So the system beat the hand-schedule in a way beyond recovering it.

### 2c. Tawa — arXiv 2510.14719

Chen, Fan, Collins, Hagedorn, Gaburov, Masuda, Brookhart, Sullivan, Knight,
Zhang, Grover (Cornell + NVIDIA), CGO'26. "Automatic Warp Specialization for
Modern GPUs with Asynchronous References". **[abstract]** The mechanism is an
IR abstraction — asynchronous references, `aref` — that expresses warp-level
communication without exposing hardware details.

Its three named challenges are the crispest published statement of the manual
work, and challenge 2 is *literally* the inline-assembly intuition I had:

1. **Coordinating concurrent warp roles.** "Even modern, high-level DSLs like
   Triton retain SIMT-centric abstractions, which **preclude the explicit
   assignment of warp roles** and thus cannot naturally express
   warp-specialized programs."
2. **Low-level communication.** "CUDA does not currently expose C++-level
   interfaces for TMA and transaction barrier (mbarriers) operations.
   Consequently, programmers must rely on **inline PTX** to configure TMA
   descriptors, initialize and manipulate mbarriers, and orchestrate
   synchronization. This requires reasoning explicitly about **arrival counts,
   barrier phases, and shared-memory aliasing**, where subtle errors can easily
   lead to deadlock."
3. **Resource allocation and pipeline orchestration.** Multi-buffering, manual
   prologue/main/epilogue partitioning, and a tightly coupled chain: "increasing
   the pipeline depth affects shared memory capacity, which in turn constrains
   thread block size and impacts TMA descriptor layouts."

Results on H100: up to **1.1× over cuBLAS GEMM**, **1.2× over Triton** on
attention, and it **matches the hand-optimized CUTLASS C++ FlashAttention-3
kernel**. The attribution ablation is what makes the claim useful: Triton
without warp specialization delivers **104 TFLOP/s**; enabling WS with a single
warp group improves it **3.78× to 393 TFLOP/s**. It is WS doing the work, not
tiling.

One more fact I should carry, because it dates the taxonomy's "Triton ships
warp specialization as a roadmap item": Tawa's implementation **has been
upstreamed into Triton** (`aref_auto_ws`, first PR #6288). So Triton's (iii) is
closing from the inside too.

### 2d. Triton — MAPL 2019, and what it says in its own words

**Tillet, Kung, Cox, "Triton: an intermediate language and compiler for tiled
neural network computations", MAPL 2019**, DOI `10.1145/3315508.3329973`. Not
OSDI, not 2022. (OSDI 2018 is **TVM**; OSDI 2020 is **Ansor** — I suspect that
adjacency is where the confusion came from.)

The paper PDF was unreachable, but the official Programming Guide restates the
paper's position verbatim, and it is the canonical (ii) claim:

> Regrettably, these systems – whether they be based on polyhedral machinery
> (e.g., Tiramisu, Tensor Comprehensions) or scheduling languages (e.g., Halide,
> TVM) – remain less flexible and (for the same algorithm) **markedly slower than
> the best handwritten compute kernels** available in libraries like cuBLAS,
> cuDNN or TensorRT.

And the residual, named: "The main challenge posed by our proposed paradigm is
that of **work scheduling**, i.e., how the work done by each program instance
should be partitioned for efficient execution." So Triton's own position is
exactly the (ii)/(i) split: it owns intra-block locality, the human owns the
block decomposition.

### 2e. arXiv 2601.15727 is a citation map, not evidence

"Towards Automated Kernel Generation in the Era of LLMs" is a **9-page IJCAI
2026 survey** (FlagOS AI), not an empirical study. **[abstract]** Its abstract
states the thesis I already held — near-optimal kernels "requires expert-level
understanding of hardware architectures and programming models, making kernel
engineering a critical but notoriously **time-consuming and non-scalable**
process" — and that LLMs "are well-suited to **compress expert-level kernel
knowledge that is difficult to formalize**". Use it to find references. Do not
cite it for a number.

---

## 3. The taxonomy, tagged

A tag is meaningless without a compiler attached: the same decision is (i) in
one system and (ii) in another.

- **(i) CANNOT EXPRESS** — no construct denotes the decision. Only a port
  reaches it.
- **(ii) CAN EXPRESS, DOES NOT FIND** — reachable in principle; the search, cost
  model, analysis or budget fails to select it. Fixable by better search/budget/
  analysis. **No port required.**
- **(iii) NOT YET IMPLEMENTED** — understood and specified, absent from the
  shipping compiler. Fixable by writing the pass.

| # | decision | Triton | TileLang | tinygrad | Twill | Tawa |
|---|---|---|---|---|---|---|
| 1 | tile sizes | (ii) | (ii) | (ii) | — | — |
| 2 | memory-hierarchy / L2 blocking | (ii) | (ii) | **(iii)** L2 | — | — |
| 3 | warp cooperation shape (co-issuing an MMA) | (i) | (i) | (i)/(iii) | **solved** | **solved** |
| 4 | MMA fragment layout | automated | automated | (iii) *unverified* | constraint | (iii) |
| 5 | accumulate dtype vs I/O dtype | (ii) | (ii)+(iii) | (iii) | — | — |
| 6 | **online / no-intermediate algorithm** | **(i)** | **(i)** | **(i)** | (i) | (i) |
| 7 | occupancy / register pressure | (ii) | (ii) | (ii) | constraint | (ii) |
| 8 | vectorised strided access | (ii) *and* (i) | (ii) | (iii) *unverified* | — | — |
| 9 | software pipelining depth | (ii) | (i) AOT | (ii) | **solved, optimal** | **solved** |
| 10 | warp-specialisation roles | (iii) | (i) | (i)/(iii) | **solved, optimal** | **solved** |
| 11 | **recompute vs store / cross-kernel fusion** | (ii) | (ii) | **(iii)**, self-admitted | — | — |

Item 5 splits: TileLang is (ii) for declining to vectorise *already-aligned*
data (RC0b) and (iii) for silently lowering fp32→TF32 on SM80, "failing an exact
1e-5 check on **31.8% of outputs** while a manual FMA kernel passes". Item 8
splits too, and this is the sharpest (i)/(ii) distinction in the corpus: Triton
"**cannot prove**" successive spatial-loop iterations are aligned for KH×KW>1 and
so "conservatively emits scalar 32-bit loads" — a better analysis fixes it, so
(ii). But NCHW "**breaking LDG.128 alignment**" is not fixable by analysis; a
human has to restructure as implicit GEMM with padding, so (i) until ported.

### 3a. The two entries I had framed most optimistically

**(6) Online / no-intermediate algorithms are (i) for every DSL.** Triton,
TileLang and tinygrad all take a *function* and compile it. If the function
materialises the N×N score matrix, no schedule removes it — the compiler
optimises over a fixed arithmetic-intensity budget. 2607.04454 refuses to even
tabulate across this boundary ("the Triton and TileLang implementations are **not
algorithmically identical** to this baseline"), which is the authors implicitly
declaring it out of scope for schedule search.

The two-sided proof is the best single piece of evidence in the whole corpus. An
**in-tree** TileLang Softmax "caches the entire row in shared memory", so its
per-block footprint erodes occupancy: **20× slower than PyTorch at N=8192 and
111× at N=32768, while numerically correct** — and the identical kernel hits
parity on the GH200, whose larger shared-memory budget hides the defect. The
language let a human make a bad *algorithmic* choice and the compiler's opinion
of the result was irrelevant. **111× was decided by an authoring choice, not by
compiler quality.** The repair was also an algorithm change, not a schedule:
"streaming the reduction dimension in tiles (an online, max-rescaled pass) rather
than materializing the full row in a fragment."

**(11) Cross-kernel scheduling is (iii) and tinygrad says so itself.** From
`docs/developer/speed.md`: "The scheduler determines how operations are grouped
into kernels and which Tensors are written to memory. **This is currently a big
bottleneck of training speed.**" And on recompute-vs-store, after working
through a trivial example correctly (with recompute, 6 reads + 2 writes; without,
6 reads + 3 writes): "**However, once you add movement ops and casts this is even
harder to figure out. tinygrad doesn't yet have a systematic way to do it.**"

Note the shape of that admission. The *arithmetic* is trivial and a human gets
it right; **the difficulty is the surrounding context**. That is the signature of
a graph-level decision, and it is why 2607.04454's surviving residuals are all
at that level while its per-kernel gaps closed.

---

## 4. Is the boundary closing? Yes on one identifiable front, and no on the rest

**Closing:** decisions that are simultaneously (a) local to one kernel, (b)
enumerable in a small action space, and (c) scoreable by a cost model or a cheap
measurement. That is exactly the class Twill and Tawa attack, and they succeed —
Twill *proves* optimality and rediscovers FA3/FA4 from a tile-level description;
Tawa matches hand-written CUTLASS from a tile program.

**The convergence is recent and it postdates everything.** Twill (Dec 2025) and
Tawa (Oct 2025) are both newer than **every** "compilers cannot do this" system in
this note. So "static graph compilers cannot express warp cooperation" describes
a generation of systems, not a law, and citing it as a law in 2026 is
anachronistic. TinyGrad's own author proposes the same technique independently:
expose the search, "fully capture all dimensions of the search space", then "Burn
compute. Use LLMs to guide. **Use SAT solvers.** Reinforcement learning."
(That is a stated bet, not a measurement — and his line counts are explicitly
ChatGPT estimates, not measurements.)

**Not closing, and with no mechanism that would close it:** the three genuine
residuals. After best-effort optimisation, 2607.04454 is left with Conv2d
**42%**, large square GEMM **77%**, and index-returning argmax **27%** of the
library, and says why in one sentence:

> The genuine residuals — Conv2d (42%), large square GEMM (77%), and
> index-returning Argmax (27%) — survive best-effort optimization because
> cuBLAS, cuDNN, and `torch.argmax` are **themselves well-tuned** on the
> data-center A100.

**This is the epistemic point I must not skip.** A residual measured against a
human-written baseline is a **floor, not evidence of a fixable gap**. If cuDNN
beats your conv, you have learned that cuDNN is good. The same logic retires a
whole family of LLM-kernel claims: KernelBench's best frontier model reports
**1.43×** geomean under its own protocol and **0.88×** under a TF32-enabled
baseline with four hidden input distributions — "no model consistently
outperforms PyTorch when evaluated against realistic baselines" **[abstract]**.
"Beat the baseline" is a property of the baseline.

### 4a. Two corrections to received wisdom

**Algorithm diversity is not where the convolution win is.** The intuitive
answer to "why is hand-written better?" is "they have better algorithms".
Measured on Ampere, Winograd is worth **≈2–3%**: a deterministic-vs-
non-deterministic A/B of cuDNN on 3×3 stride-1 differs by **0.03%** (ratio
0.9997). And the gap **does not track Winograd eligibility** — the eligible 3×3
stride-1 filter (28.5%) is no better than the ineligible stride-2 case (32.1%),
and the deepest gaps fall on 5×5 (17.8%) and 7×7 (13.1%). The residual is
"general cuDNN implicit-GEMM tuning — **kernel selection, split-K, shared-memory
tile sizing**". So the honest statement is: the algorithmic lever exists and is
nearly irrelevant at this scale; the residual is a library's hand-written plan
selector. (This is also why Ansor's own qualitative claim is the useful one: it
has "more advantages on **uncommon shapes and small batch sizes**, because it is
not easy to manually optimize for these cases" **[abstract]** — search wins in
the regime nobody hand-optimises.)

**FlashAttention's IO optimality is conditional.** Saha & Ye (PMLR v235, 2024)
prove a pointwise IO lower bound matching FlashAttention **for fast-memory
capacity M ≥ d²**, and identify a different regime below it. So it is incorrect
to call the algorithm optimal for every possible SRAM size. Any statement that
the online-algorithm question is *closed* is wrong below d².

---

## 5. Negative results, which are load-bearing for this study

Three, all from 2607.04454, and all of them are the kind of finding that
survives review *because* it is negative.

**(a) The register-pressure hypothesis was refuted where it was most expected.**
Their register section opens: "A natural hypothesis is that large-filter
convolution hits this wall; **NCU refutes it: Triton conv2d reports
`n_spills = 0` at every tested filter** (1×1 through 7×7), even though register
usage rises with K² (up to 224 regs/thread at large shapes)." The conv gap is
occupancy- and coalescing-bound, not spill-bound. **Do not assume spills explain
a slow conv.** The real spill is TileLang-LayerNorm-specific: 254 regs/thread,
12.4% achieved occupancy, 51.5 GB local-load + 34.4 GB local-store on A100.

**(b) Tile-size autotuning gave literally zero improvement — and widening the
grid gave 1.37×.** "Hardware-agnostic heuristic tuning of the block-tile grid
yields **Δ ≈ 0 pp for both DSLs on every kernel** — default and tuned latencies
are near-identical, and the convolution gap is unchanged (28.9%). … (Δ = 0 pp on
every row, convolution included): the gaps are insensitive to block-tile shape."
All 22 kernels, no movement.

The precision matters, or this reads as "tile autotuning is worthless": the
12-config grid varies *only* block-M/N/K, and the expanded search additionally
sweeps `GROUP_SIZE_M` (the L2 swizzle), `num_warps` and `num_stages`, which
recovers Triton **59.7% → 81.9%** at 16384² and **71.5% → 77.3%** at 4096². So
the correct reading is: **tile *shape* is not the lever; the axes the default grid
omits are.** Their verdict — "the missing configurations, not an exhausted
budget, bound default performance" — is a (ii) diagnosis with a one-line fix.

**(c) "Optimized" is architecture-indexed, with no correctness signal.** The
LogSumExp kernel: 581.7% of PyTorch on A100 (0 spills, 93 regs), and on GH200 it
register-spills (255 regs, ~280 GB spill traffic, 12% occupancy) and collapses to
**1.0%**. Retuning recovers only to **63%**. A correct, hand-optimized kernel was
~580× off on another GPU with no error signal at all. Their conclusion: "a
correct, hand-optimized DSL kernel can pass on one GPU yet collapse on another, so
**single-GPU or single-shape benchmarking cannot certify it**." Any single-GPU
result — including anything I measure — is a statement about one cell of a grid
the field has declared infeasible.

---

## 6. The consequence: there are no numbers to borrow

**An arXiv API query for `all:"tinygrad"` returns `totalResults 0`.** I
re-ran it today to be sure; the response is an empty feed.

> `https://export.arxiv.org/api/query?search_query=all:%22tinygrad%22`
> → `<opensearch:totalResults>0</opensearch:totalResults>`

So: **no published work benchmarks tinygrad against hand-written kernels or
against other compilers. No TinyGrad-vs-Triton comparison exists. There is no
citable TinyGrad performance number in the literature.** Every figure in §2 is
for some *other* system, and none of them transfer.

The only circulating number is a **2024 Hacker News comment** (tinygrad 0.9.0,
May 2024, thread 40504212): for one GPT-2 training step on a 7900 XTX, "tinygrad
is ~440ms, PyTorch 2.4.0.dev… ~97ms, Karpathy's llm.c with ROCm is ~79ms, and
llm.c with custom kernels is ~58ms". A forum comment, self-reported, one
configuration, four years old, with no harness. **Unusable.** If I want it, I
measure it.

**So the study is first-or-nothing.** That is the strongest justification for it
and it is also the risk: there is no prior number to agree or disagree with, so
every claim rests on my own harness. `research/PRIORWORK.md` says reimplement
2607.04454's protocol rather than build my own, and add the TF32-baseline
correction, because that is the difference between a result and a non-result.

### 6a. The structural warning that makes a naive port misleading

**TinyGrad is not a tile DSL.** Its own docs decompose "speed" into four layers,
and that decomposition *is* the honest frame for this study:

| layer | what tinygrad says (verbatim) |
|---|---|
| **compile (Python)** | "Currently it's a bit slow, but on par with torch.compile. It gets even slower if you are using BEAM, since that's compiling many variants of each kernel." |
| **execution (driver)** | "tinygrad has the best execution speed of any framework because it usually **bypasses the GPU driver and prebuilds the command queue**. It's tons faster than normal CUDA, and often even faster than CUDA Graph. There's very little to improve here, as this is almost never the bottleneck." |
| **model (scheduler)** | "The scheduler determines how operations are grouped into kernels and which Tensors are written to memory. **This is currently a big bottleneck of training speed.**" |
| **kernel (codegen)** | what BEAM searches, over a set of `OptOps` |

**The execution-speed claim is about launch overhead, not kernel throughput, and
it is the most mis-citable sentence in the project.** Do not let it stand in for
a kernel-speed claim. It also has a specific consequence for my own KDA
measurement: 85% of wall time is the **scheduler** re-running its rewrite passes
on every `realize()`, which is a graph-layer cost, not a launch cost.

The codegen layer's self-admitted gaps, verbatim: "we never copy the inputs to on
chip SRAM, but this is often quite helpful for kernel speed"; "we aren't doing a
good job with **L2 cache awareness** (the locals handle L1 quite well)"; "We
have a simple framework in tinygrad for adding these ALU blocks and achieving
good performance from them" (i.e. no claim of SOTA tensor-core codegen — (iii),
untested); and on TMA, "**this is not supported in tinygrad yet**" ((i)).

**Therefore: report per-kernel `t` *and* its four-layer decomposition, and
pre-register which layer owns the gap** before measuring. The prediction, stated
in advance so it can be falsified: **graph scheduling > codegen (SRAM / L2 /
tensor cores / TMA) > nothing at the launch layer.** A per-kernel-only comparison
hides the layer where TinyGrad says its own worst bottleneck lives, and an
end-to-end-only comparison lets launch overhead flatter it. Both are wrong in
opposite directions.

---

## 7. My own findings map onto documented phenomena

I had been treating these as discoveries about TinyGrad. They are not — but the
mapping is sharper than "known phenomenon", and one of them upgrades.

**Over-fusion → the materialised intermediate.** From the compilers-and-kernels
chapter (Changkun Ou, *AI as an Infrastructure*; a living single-author draft, so
usable for framing and not as a source of facts):

> Fusion lengthens live ranges and can increase register pressure or
> shared-memory demand. If values spill to local or global memory, traffic
> returns in a more expensive form. A larger kernel can lower occupancy, reduce
> scheduling freedom, duplicate work, or prevent a call to a highly tuned
> library. **This is why over-fusion is possible.**

That is exactly the KDA result: the port fuses correctly, and fusing is why the
`[C,C,K]` tensor exists. TinyGrad is not malfunctioning; it is making a
locally-reasonable decision whose cost only shows up as bytes.

**And it upgrades, because of §3a.** "Don't materialise the intermediate" is not
just "a known effect" — it is **(i) for every DSL in the study**, the firmest (i)
in the taxonomy, with a 111× measurement attached. The hand-optimised KDA kernel
does not fuse better; it *never builds the tensor*, by folding the decay's `exp`
into a matmul operand load. That is a data-layout choice made before the
compiler runs. So the KDA finding is not a TinyGrad deficiency and not a general
over-fusion hazard — it is the one gap in this corpus that **no compiler of any
kind can close**, and every human who beats it beats it by writing a different
algorithm.

**FMA → mathematical equivalence is not bitwise identity.** Same source:

> Even pointwise fusion can introduce fused multiply-add instructions or
> reassociation, so **mathematical equivalence does not imply bitwise
> identity**.

That is the camera result, stated as a general compiler fact before I measured
it. The full residual is in `02-obs-camera-residuals.md`: FMA is real and
controllable, but the 12 pixels come from MC's sin LUT quantiser, and
`product.realize()` is a complete FMA barrier where `operands.contiguous()` is
not.

**What Triton still leaves to the programmer.** Same source:

> Triton raises the programmer's unit of work from a scalar thread to a blocked
> program, while its compiler handles coalescing, vectorization, shared-memory
> allocation, and instruction selection. **The author still chooses the grid,
> masks, tile sizes, number of warps or stages**, and sometimes target-specific
> mechanisms.

And on autotuning, which reframes it from a solution to a limitation:

> Autotuning results are conditional artifacts. A winning tile for one shape,
> dtype, stride pattern, device, driver, or library version may lose for
> another. **The cache key must contain every fact on which the choice depends.**

So "just autotune it" does not converge. There is no universally optimal
schedule; there is a schedule per (shape, dtype, stride, device, driver,
library version) tuple, and someone has to pay for the search.

### 7a. Re-tagging my own findings

| finding | tag | basis |
|---|---|---|
| 96-step unroll, 4.8× dead iterations | **(ii)** | a cost model that knew rays terminate fixes it; no algorithm change |
| 101 gather kernels per frame | **(iii)** | `Ops.INDEX` exists; the Tensor API does not emit it. A patch |
| FMA contraction with no knob | **(i)** for the caller | no way to say "I need bit-exactness" |
| `a*(1/b)` reciprocal in the cstyle path | **(i)** | no knob, and CPU proves the IR is exact |
| sin LUT amplifying 1 ULP | **(i)** | unfixable from the port side at all |
| KDA `[C,C,K]` decay intermediate | **(i)**, universal | §3a item 6 — the firmest (i) in the literature |

---

## 8. What this changes about the project

**The KDA finding is not a TinyGrad deficiency and must not be framed as one.**
The correct statement is: *materialising the pairwise decay is the natural
lowering of any whole-graph compiler, and it is 256× larger than what a human
who knows the chunk structure can get away with* — and no compiler closes it.
The `C²K` fold is a rewrite pass over the matmul, and it is worth having as a
TinyGrad contribution precisely because it is a case the literature says no
compiler handles.

**The N-card run is still worth doing, but for a different reason than I said.**
Not to get "representative numbers" — because §5(c) says a single-GPU result
cannot certify anything. RC1, RC2a and RC3 are all hardware-specific and none of
them reproduce on Metal, so the run is a *portability* measurement: does the
"optimized" kernel stay optimized when the machine changes, with no correctness
signal to warn me. `kda_vs_fla.py` is the right instrument, and the prediction to
pre-register is §6a's layer ordering.

**The roofline check is the missing instrument in my harness.** Both KDA reports
give wall time and kernel counts; neither gives arithmetic intensity against the
device ridge point. That is why I could not tell that "12,090 kernels" was
mostly irrelevant while "256× intermediate" was everything. `kda/NOTES.md` has
the FLOP/byte analysis; it needs to be wired into the measurement rather than
living in a separate document.

**I should stop looking for TinyGrad bugs and start tagging gaps as
(i)/(ii)/(iii).** The bug hunt was worth it — it produced the `a*(1/b)` finding
and the correct diagnosis of the camera's residual — but it is not the study.
And the tag is not a rhetorical device: it is the difference between "write a
pass", "widen the grid", and "reconsider the operator", and only the third is a
port.

**Pre-register, before measuring:** (1) which of the four layers owns the gap
(§6a), (2) that the winner is (i)/(ii)/(iii) rather than a defect, and (3) the
baseline, because §4 shows the measured boundary moves when you fix the baseline.
On (3): a study whose baseline is PyTorch *eager* is not measuring TinyGrad, it
is measuring the launch overhead.

---

## Sources

Citations verified this pass: every arXiv ID resolved through the arXiv API, the
two DSL papers' claims read from arXiv HTML, the tinygrad doc quotes read from
`docs.tinygrad.org/developer/speed/`, the Triton quotes read from the
Programming Guide, and the `all:"tinygrad"` query re-run.

| source | establishes | status |
|---|---|---|
| **arXiv 2607.04454** "Correct but Slow" (Li, Rathnasuriya, Yang) | the experiment to copy: 22 kernels, 5 categories, A100 + GH200, counter-attributed causes, two-check screen. Table III/IV read in full | read |
| **arXiv 2012.07145** (Anderson, Adams et al.) | auto-scheduling **matches** expert hand-scheduling on GPU when experts target the autoscheduler's own runtime; reason is search-space expressiveness | read |
| **arXiv 2512.18134** Twill (Soi, Yadav, Kjolstad, Aiken + NVIDIA) | SWP+WS as a joint, provably optimal constraint problem; rediscovers and proves optimal FA3/FA4; within 1% of cuDNN/FlashAttention; "brittle heuristics and fallible human intuition" | read |
| **arXiv 2510.14719** Tawa (Cornell + NVIDIA) | WS automated via `aref`; 1.1× cuBLAS GEMM, 1.2× Triton, matches CUTLASS FA3; 104 → 393 TFLOP/s ablation; three manual challenges; upstreamed to Triton #6288 | read |
| **MAPL 2019**, Tillet, Kung, Cox (DOI `10.1145/3315508.3329973`) | the canonical (ii) claim and Triton's own automated-optimisation list, via the Programming Guide | guide read; **paper PDF blocked** |
| **arXiv 2601.15727** (IJCAI 2026) | a 9-page survey — a citation map only, no numbers | abstract read |
| **arXiv 2502.14752** TritonBench | 184 curated *human* Triton operators; the right "hand-written" corpus if I go narrow | abstract |
| **arXiv 2210.09603** Hidet (ASPLOS 2023) | TVM's primitive set is "insufficient"; widening it is an expressiveness fix (1.22× avg) | abstract |
| **arXiv 2406.20037** Droplet | better kernels from 300 Ansor samples than from 10,000 — a (ii) result in measurement-budget currency | abstract |
| **arXiv 2607.16241** KernelBench-Verified | 1.43× → 0.88× geomean under a realistic baseline; the baseline determines the result | abstract |
| **Saha & Ye, PMLR v235 (2024)** | FlashAttention is IO-optimal only for fast-memory capacity ≥ d² | secondary |
| **Changkun Ou, "Compilers and Kernels"** (living draft, not peer reviewed) | over-fusion; FMA ≠ bitwise identity; what Triton leaves the author; autotuning as a conditional artifact | read |
| `docs.tinygrad.org/developer/speed/` | the four layers, verbatim; "execution speed" is launch overhead; the codegen self-admissions | read |
| arXiv API `all:"tinygrad"` | **`totalResults 0`** — no published TinyGrad performance number exists | re-run this pass |
