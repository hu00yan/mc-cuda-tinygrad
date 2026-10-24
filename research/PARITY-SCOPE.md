# PARITY-SCOPE

**One question:** does the 0.95× GPU auto-scheduling parity result (arXiv 2012.07145) extend to
chunked linear attention, convolution, and reductions — or is it confined to single-level-tiling,
GEMM-like problems?

**Answer: it does not extend.** The parity result is a property of a search space with a named,
enumerated boundary, measured on imaging pipelines on a V100, with no recurrence, no attention, and
no normalisation operator in the suite. The reasoning that would license extension does not survive
contact with the paper, and two of the three load-bearing premises in the framing are misattributed.
Details below, then the exact replacement wording.

---

## 0. Two corrections that must be made before anything else

### 0.1 "One level of tiling" describes the *predecessor*, not 2012.07145

The framing attributes the parity shortfall to the autoscheduler's space containing only one level
of tiling. It does not. That sentence is about **Sioutas et al. 2020**, the prior state of the art
that 2012.07145 *replaced*, and 2012.07145 cites multi-level tiling as the source of its own win
over it. Verbatim, §7 Related work:

> "Sioutas et al. (2020) uses the same search algorithm but extends it to support the GPU with an
> expanded search space and a hand designed GPU cost model. It runs quickly without any benchmarking,
> but its search space is smaller compared to ours among other reasons because **it only supports a
> single level of tiling**, and as we discuss in Section 6.2, this excludes a number of high
> performance schedules."

2012.07145's own space has **three** levels of tiling. §4.1:

> "If we decide to compute the Func in its own kernel, we defer its tiling to the second phase of the
> search […] At that point, **it will be tiled into 3 levels.**"

Outer loops are blocks, inside them thread loops, inside those serial loops (§4). So multi-level
tiling is *inside* the envelope, not outside it. For completeness, the sibling lineage has even
more: Ansor's GPU tile structure is **"SSSRRSRS"** — 8 levels, mixing space (S) and reduction (R)
tile levels (§4.3, arXiv 2006.06762v5). A claim that multi-level tiling is what caps GPU
auto-scheduling is not supportable from these papers; it is the opposite of what they argue.

**Consequence:** the real expressiveness boundary is *not* "levels of tiling". It is narrower and
sharper, and is stated in §0.2.

### 0.2 The actual expressiveness limit is reductions, not tiling depth

2012.07145 §8 Limitations, verbatim:

> "We currently make tiling decisions **on a per-Func basis**, but could make those decisions for all
> the update stages in a Halide function. This would allow us to **tile and unroll reduction
> variables** (sequential loops that are not directly parallelizable) in update stages, which has
> proven an important optimization on applications like *conv layer* and *IIR blur*. Making tiling
> decisions on a per-stage basis is likely also necessary to **support parallelizing reductions**
> (including the Halide scheduling options **rfactor and atomic**), which is an important optimization
> on *histogram equalize*."

Ansor independently confirms this from the other side (§7.1): "the significant speedup on **NRM** is
because Ansor can **parallelize reduction loops**, while other frameworks do not," and "the
unfolding rules in **Halide do not split the reduction loop** in GMM and do not split reduction
loops in C2D when padding is computed outside of reduction loops."

This is the load-bearing limit for our operators, and it is a *reduction* limit. For a reduction,
normalisation, or gated recurrence, the thing that matters is whether the search space can split and
parallelise the reduction axis — not how many levels it tiles the space axes.

### 0.3 The third premise: the parity claim is algorithm-conditional, and that is the whole point

2012.07145 searches **schedules of a fixed Halide algorithm**. The paper's own framing (§1) is that
"Halide… decouples the algorithm — *what* to compute — from the schedule — *how* to compute it," and
all contributions are to schedule search. The autoscheduler cannot change the program it is given.
Everything in §2 below is therefore parity *within the space the autoscheduler was given*, and the
paper's own §6.5 says the space is missing transformations that win. For our study — where the
decisive choice is to change which tensor exists at all — this is the whole distinction.

---

## 1. The exact experimental envelope of 0.95×

All quotes below are from arXiv **2012.07145v2** (= OOPSLA 2021, DOI 10.1145/3485486), read in full.

### 1.1 Hardware and protocol

§6, verbatim:

> "All our results were generated on an IBM AC922 with 2×20 core Power9 CPUs and 4×**NVIDIA V100
> SXM2** cards with 32GB of memory. […] All benchmarks were performed on a single V100 in isolation."

- **One GPU model: V100 SXM2 (Volta, sm_70).** No A100, no H100, no GH200. Volta has no bf16
  tensor cores and no async-copy pipeline that Ampere/Hopper scheduling spaces target.
- The 0.95× figure is the **Autotuning mode**: 20 iterations × 80 samples = **1600 compiled and
  benchmarked schedules per program**, taking 9–71 minutes (§6, *Autotuning*). It is a
  best-of-1600 number, not a compile-time number. The paper's own cheaper modes are far worse:
  **1.07×** geomean in One Shot and **1.30×** in Top 5 versus the prior autoscheduler (§6.2).
- One Shot selects from 80 cost-model-ranked candidates with **no benchmarking at all**. §6.4: the
  One-Shot-selected schedule is "on average (geomean) **within 48%** of the actual fastest schedule"
  (within 33% excluding bilateral grid), against 106% for an average sample. **The cost model is
  weakly predictive**: "For most of the applications, without retraining […] the cost model is only
  weakly predictive of actual run times."

### 1.2 Operators — the suite contains no attention, no recurrence, no normalisation

§6, verbatim, all 17:

> "a diverse set of 17 imaging and vision programs […] including 15 applications from the Halide
> repository: *bilateral grid*, *local laplacian*, *non-local means*, *lens blur*, *camera pipe*, a
> 32-stage *stencil chain*, *Harris corner detection*, *histogram equalize*, *max filter*, *unsharp
> mask*, *interpolate*, a neural network *conv layer* with ReLU activation, *SGEMM* (Single float
> precision General Matrix Multiply), an *IIR blur*, *BGU* (bilateral guided upsampling). To this we
> added a *depthwise-separable convolution*, and a learned demosaicing algorithm."

Mapping to our study's three categories:

| Our category | In the 2012.07145 suite? |
| --- | --- |
| **Chunked linear attention / gated recurrence** | **Absent.** No recurrence, no state-carrying operator, no sequence-dimension parallelism anywhere in the suite. |
| **Convolution** | Present, and it is where the autoscheduler **lost**: *conv layer*, *depthwise-separable conv*. |
| **Reductions** | **Absent as a category.** *Histogram equalize* and *max filter* are reductions, and the paper names histogram equalize as one of the losses. There is no LayerNorm/RMSNorm/softmax row. |

Program sizes: "span a diverse set of imaging and learning programs from a few to around one hundred
stages" (Figure 6 caption). Halide programs, image-sized tensors.

### 1.3 The single most damaging fact about the 0.95× figure

There is **exactly one GEMM in the suite** — SGEMM — and it is the program with by far the smallest
search space. Table 1, states evaluated per beam search, sampling+freezing on:

| Program | States evaluated | vs. geomean (11,740) |
| --- | --- | --- |
| **matrix multiply** | **698** | **6% of geomean** |
| learned demosaic | 50,488 | 4.3× geomean |
| stencil chain | 22,384 | 1.9× |
| *geomean, all 17* | *11,740* | — |

**Arithmetic I did, not a published number:** SGEMM's 698 states are ~17× smaller than the suite
geomean, and ~72× smaller than the largest. The one program closest to "GEMM-like" was the one the
scheduler had the least room to work on. So 0.95× over 17 programs carries essentially **no
information about GEMM**, let alone about anything with a two-level decay tensor. If 0.95× cannot be
extended to GEMM from this paper, it certainly cannot be extended to chunked linear attention.

### 1.4 What "expert hand-scheduled" meant operationally

§6, verbatim:

> "We compare our autoscheduler against the best Halide schedules experts are capable of writing by
> hand, using the entire scheduling language. Our experts iteratively improved these schedules
> during the course of this work **using the best runtimes found by the autoscheduler as a target to
> beat (but without looking at the generated schedules)**, so they represent a very high bar."

Three constraints on the strength of this baseline:

1. **Same DSL.** The human wrote Halide schedules, not CUDA. A comparison inside Halide cannot
   measure the value of leaving Halide. Our study's "hand-written kernel" is a different artefact
   class.
2. **Same algorithm.** The human re-scheduled the same Halide `Func` graph. The expert could not
   change which intermediate exists.
3. **Not independent.** The human used the autoscheduler's runtime as the target, and — per §6.5 —
   spent effort reactively. This makes the comparison *favourable to the autoscheduler* in one
   respect (the human is anchored to a target the machine chose) and *unfavourable* in another
   (the human is not exploring, they are hill-climbing toward a number). It is not a clean
   "best human effort" ceiling, and the paper does not claim it is.

### 1.5 Verdict on the framing

> **The parity claim is "parity within the search space the autoscheduler was given", on
> low-arithmetic-intensity image pipelines, on one Volta GPU, with no recurrence, attention, or
> normalisation operator present.**

§6.5 says this in the paper's own words, and adds the detail that a geomean hides:

> "The manual schedules outperform our autoscheduler in several instances. […] Despite operating
> without all of these features, **the autoscheduler was able to beat the manual schedules the
> majority of the time**, and has geomean performance on par with a human expert making their best
> effort to beat our results. […] **The large gains come in the most complex, heterogeneous
> applications, which are intractably difficult for humans to schedule.**"

So 0.95× is a geomean over a bimodal distribution: the machine wins most programs (and wins big on
complex heterogeneous ones, which is where its 1.7× lives) and loses on a specific, *enumerated*
set. It is not "the compiler is 5% off the human everywhere." Never cite it as nearness — that
error is already flagged in `SOURCES.md` §4 and `TAXONOMY.md`.

**[Could not verify] per-program speed ratios.** Figure 6 is the only place per-program ratios
appear, and it is an image; I could not read values out of it. §6.5 gives the qualitative list of
losses only. Any per-op number quoted for 2012.07145 must come from someone who has read Figure 6.

---

## 2. The expressiveness limit, stated precisely

### 2.1 What 2012.07145's search space can and cannot express

From §4.1–§4.3, the space is fully enumerable:

| Level | Options enumerated | Verbatim |
| --- | --- | --- |
| **Serial** (inner, unrolled) | 1, 2, 4, 8 per dim; plus odd 3, 5, 7 conditionally | "We enumerate serial tile sizes that are small powers of 2: 1, 2, 4, and 8 in each dimension. We also consider small odd tilings (3, 5, 7) if they will enable the resulting thread loop's extent to be a multiple of the warp size" |
| **Thread** | 16, 32, 64 innermost; powers of 2 up to 16 elsewhere | "We enumerate thread loop sizes that encourage favourable warp sizes: 16, 32, 64 in the innermost loop dimension, and powers of 2 up to 16 in the other dimensions." |
| **Block** | remainder after serial+thread | "The remaining loop extents after choosing thread sizes will become the outer block loops." |
| **Compute location** | own kernel / fused at consumer's block (shared mem) / inside consumer's threads (registers) / inline | §4 |

And the hard edges of that space:

- **§4:** "**Inline options and Funcs fused inside their producer's thread loop are not considered
  for tiling.**"
- **§4.3:** compute-at-block funcs get "all serial loop options only" — no block tiling. Inside-thread
  funcs get "tilings [that] are not enumerated."
- **§8:** reduction variables cannot be tiled or unrolled at all; `rfactor` and `atomic` are
  unsupported.

**So the space has 3 tiling levels, but the innermost level caps at 8 (serial) and 64 (thread), and
the reduction axis is not tileable at all.** For an operator whose decisive structure is the size of
a tensor along a reduction-like axis, that is the binding constraint — not level count.

### 2.2 What the human schedules did that the space could not (§6.5, verbatim)

> "For all of *conv layer*, *IIR blur*, and *depthwise separable conv*, the expert schedules **split
> and unroll the reduction loops**. In *BGU* and *histogram equalize* the manual schedules use
> **atomic floating point adds to memory to expose more data parallelism**. In *conv layer*, *IIR
> blur*, *learned demosaic*, and *matrix multiply*, the manual schedule uses **warp-shuffle
> instructions** to share data between the threads in a warp without requiring a full barrier across
> the entire thread block. These transformations can have significant performance advantages but are
> **not currently in the search space of our autoscheduler**. Other common patterns in the manual
> schedules not exploited by our autoscheduler are **SIMD vectorization of load instructions** to
> reduce the total number of memory transactions, and **pre-staging of stencil inputs into shared
> memory and/or registers** to reduce the total number of loads to device memory."

Note what this is: **four of the six missing transformations are reduction/parallelism
restructurings**, not tile-depth changes. That is consistent with §0.2 and with Ansor's independent
finding. It is also why the two convolution programs in the suite are losses.

### 2.3 Fusion boundaries: is there published evidence an intermediate is cheaper to avoid than to consume?

**Yes, and it is the strongest and most on-point citation available for our framing** — but it comes
from the operator literature, not the compiler literature. Yang, Wang, Zhang, Shen, Kim, arXiv
**2406.06484v6** ("Parallelizing Linear Transformers with the Delta Rule over Sequence Length",
NeurIPS 2024), §3.1:

> "However, computing **u**_t *naively* requires explicitly materializing **S**_{t-1} to compute
> **v**_t^old, which would require O(d²) memory. We now show that we can obtain the **u**_t's
> *without* explicitly materializing **S**_{t-1} **in O(d) memory**."

Same paper, footnote 1 in §2.1 — this is the general principle, stated as a reason to reject an
algorithm:

> "It is possible in theory to use parallel scan to parallelize the recurrent form, which would enable
> the computations to be performed in O(log L) steps and O(Ld²) FLOPs. However, this approach
> **requires materializing the 2D hidden state for each time step, which would incur significant memory
> I/O cost unless the state size is small enough such that materialization can happen in faster
> memory** (i.e., as in Mamba)."

And §3.2, the reason the pairwise form is rejected outright:

> "However, computing **T** requires a matrix inverse (Eq. 10), **which scales cubically with sequence
> length without further algorithmic changes. Due to the above we avoid using the fully parallel form
> for training DeltaNet**."

**The decisive structural sentence**, §3.2 "Practical considerations":

> "In the above, Eq. 7 is **fully recurrent and thus cannot use tensor cores written as is**. To solve
> this, we further leverage the *UT transform* […] to rewrite most operations in matmuls."

And the conclusion, which is the whole thesis of our study:

> "Note that **u**_t does not require materializing any of the hidden states and requires O(d) memory
> to compute […] However, computing **u**_t's for all L (that is, **U**) takes **O(L²d)** and
> moreover **cannot be fully parallelized**, unlike in linear attention where we can calculate all the
> value vectors **V** in parallel in O(1) steps."

This is the compiler-side framing of our design choice, in the peer-reviewed source, with the cost
model attached: the naive form is O(L²d) and unparallelisable; the per-sub-chunk form is
O(LCd + Ld²) (§2.1, for linear attention) and parallelisable. **Our ~256× is the same phenomenon at
our parameters** — and note it is a factor of *work and traffic*, obtained by an algorithm change
(UT transform, block-diagonal triangular solve per chunk) that a schedule search over a given
algorithm cannot make. *(The 256× figure is ours by arithmetic on our shapes: the fully-parallel
DeltaNet form's pairwise object is [L,L] while the chunked one is [C,C] per chunk, so the factor is
L/C; with L=16384 and C=64 — the sequence length 16384 is stated in 2406.06484 §3.2 "Speed
comparison", and C "usually 64 or 128" in §2.1 — L/C = 256. Not a published number.)*

Two further supporting citations, both verified:

- **Triton Programming Guide** (via `SOURCES.md` §5): "these systems – whether they be based on
  polyhedral machinery […] or scheduling languages (e.g., Halide, TVM) – remain less flexible and
  (for the same algorithm) **markedly slower than the best handwritten compute kernels** available in
  libraries like cuBLAS, cuDNN or TensorRT." The parenthetical **(for the same algorithm)** is the
  load-bearing phrase: the DSLs concede on tuning, and the escape hatch is the algorithm.
- **2607.04454** RC4 "Absence of Algorithm-Level Diversity" (§V), verbatim:
  > "cuDNN selects Winograd for 3×3 filters when arithmetic reduction dominates (Winograd F(2,3)
  > cuts multiply-accumulates ≈2.25×), but **neither Triton nor TileLang exposes the transformation as
  > a schedulable primitive.**"
  This is the compiler literature's own label for our exact situation: the win lives in the
  algorithm, and no tile shape reaches it. Their measurement of its size is in §VI-E: "**of the
  Conv2d gap only ≈2–3% is absent Winograd selection (RC4)**" — i.e. for cuDNN-vs-DSL on conv, the
  algorithm-selection term is *small*. Worth stating explicitly, because it is a genuine
  counter-consideration: it means the algorithm term is not always dominant. It is small *there*
  because cuDNN's residual is cache blocking and tuning, not algorithm choice.

---

## 3. The counterweight: has auto-scheduling ever been shown losing to a human?

**Yes — and the best instance is the same paper that produces the 0.95× figure.** This is the
cleanest available answer, and it needs no external hunt.

### 3.1 The loss is published, with named operators and named causes

2012.07145 §6.5 enumerates it: the expert hand schedules **beat** the autoscheduler on *conv layer*,
*IIR blur*, *depthwise separable conv* (reduction-loop split/unroll); *BGU* and *histogram
equalize* (atomic adds for data parallelism); *learned demosaic* and *matrix multiply* (warp
shuffles). Figure 6's caption repeats it: "**Some of the human expert schedules (IIR blur, conv
layer, depthwise separable conv) use scheduling options that are outside the autoscheduler's search
space.**"

And §6.2, on the predecessor, is a loss with a magnitude:

> "Sioutas et al. generate code that **fails to run** on the *conv layer*, and performs poorly on
> several apps […] Its single largest weakness is a search space issue with workloads that include
> *matrix multiply* or *conv layer*. A fast matrix multiplication contains an **inner unrolled block
> over a tile of O(MN) accumulators**, which share the O(M+N) loads required to compute each term.
> Sioutas et al. **do not consider this form of unrolling.**"

Read that last sentence carefully for our purposes: the missing thing is an *unrolled register-block
of accumulators*, i.e. **multi-level tiling of a GEMM's reduction axis**. That is precisely the
"second level" the framing is worried about, and it is a documented, quantified failure of a
single-level space — which is the correct argument to make, just not about 2012.07145.

### 3.2 DSL-vs-hand-tuned-library losses, measured, with causes

2607.04454 (`SOURCES.md` §1) is the best-designed instance on modern hardware. A100, clock-locked
protocol, 7 Nsight Compute counters, one root cause per counter signature:

| Category | Triton E_lib (A100) | TileLang E_lib (A100) | Classification |
| --- | --- | --- | --- |
| GEMM | 59.7–97.9% | 78.1–94.2% | **genuine residual** after expansion (77% square) |
| Convolution | 28.9% (47.4% @1×1 → 13.3% @7×7) | 43.6–59.3% | **genuine residual** (42% after optimisation) |
| Normalisation | 90.2% (layer_norm) | 0.3–3.2% | authoring artifact (RC0a) |
| Overall median | ≈98% | ≈68% | — |

The two residuals that survive best-effort optimisation on **both** A100 and GH200 are convolution
and large square GEMM. Their §VI-E verdict: "The *genuine residuals* — Conv2d (42%), large square
GEMM (77%), and index-returning Argmax (27%) — survive best-effort optimization because cuBLAS,
cuDNN, and torch.argmax are **themselves well-tuned** on the data-center A100." Roofline anchors
§VI-A: residual kernels sit at ρ ≤ 0.34 (A100) and ≤ 0.28 (GH200), so the shortfall is real headroom,
not a fast baseline. And the tile-shape knob does nothing: "Hardware-agnostic heuristic tuning of the
block-tile grid yields **Δ ≈ 0 pp for both DSLs on every kernel**."

That last line is the important one for our framing. In a *tiling DSL*, the tile-shape dimension is
nearly worthless; the wins came from widening the search (RC2a: 59.7% → 81.9% at 16384² by adding
`GROUP_SIZE_M`/`num_warps`/`num_stages`) and from restructuring (RC2b L2 residency, RC4 algorithm
selection). "the missing configurations, not an exhausted budget, bound default performance" (RC2a).

### 3.3 The honest meta-observation: the comparisons were not set up to find a loss

Where a paper's own comparison design selects *for* the automated method, the field produces no
evidence either way. Two self-declared instances:

- **Ansor** §7.3, verbatim: "Compared with vendor-specific static libraries, Ansor has more
  advantages **on uncommon shapes and small batch sizes, because it is not easy to manually optimize
  for these cases**." That is an explicit statement that the library comparison is run where humans
  have not tuned. And Ansor's reporting structure cannot express a loss: single-operator results are
  **normalised to the best framework** and reported as geomean-with-error-bars across four shapes
  (§7.1), with the claim "Ansor performs the best or equally the best in all operator and batch size
  settings." A framework normalised to 1.0 has no room to report below 1.0. (Also note: Ansor's
  single-operator benchmark §7.1 is **CPU/Intel** — the GPU comparison is the subgraph and
  end-to-end benchmarks, against AutoTVM and manual libraries on *ConvLayer* and *TBS*.)
- **KernelBench / TritonBench** (`SOURCES.md` §6, §7): correctness is the *gate*; a slow kernel is
  never rejected. 2607.04454's contribution is precisely to show this admits a 300× slowdown.

So the truthful statement of part 3 is a split one, and both halves are citable:

> There is a published, enumerated set of operators where GPU auto-scheduling loses to a human for
> **structural** reasons — 2012.07145 §6.5 (reduction-loop splitting, atomics, warp shuffles) and
> §6.2 (multi-level accumulator unrolling, with a 2.5× graph-partitioning penalty) — and a
> measured set where tile DSLs lose to tuned vendor libraries for structural reasons (2607.04454:
> conv 28.9%, large square GEMM 59.7% on A100). But there is **no** published study designed to
> find a human win over auto-scheduling on a non-GEMM operator, because the two dominant comparison
> designs (normalise-to-best, correctness-gated) are structurally incapable of reporting one. The
> absence of such a study is a property of the field's evaluation methodology, not evidence that
> auto-scheduling wins.

**[Could not verify]** that no such study exists anywhere. I searched for published autoscheduler
losses to human schedules and found none beyond the above. This is a negative claim and should be
stated as one.

---

## 4. Multi-level and non-GEMM specifically

### 4.1 Attention: the best-designed study deliberately refuses to compare it

This is the single most useful item in this document for our framing. 2607.04454 includes one
attention workload and then **excludes it from its headline comparison**, because the algorithms
differ. §III-B, verbatim:

> "For attention, we use PyTorch scaled dot-product attention with the FlashAttention backend;
> **because the Triton and TileLang implementations are not algorithmically identical to this
> baseline, we report attention results separately.**"

Table I, Attention row, verbatim: **"baselines non-equivalent"**. Figure 1's caption: "**Attention
excluded**: the Triton (tiled FlashAttention-2) and TileLang (**untiled O(n²)**) kernels are
algorithmically different." And from the Appendix C table, the two attention cells: Triton attention
**7691%** (a FlashAttention-style tiled kernel beating PyTorch's *unfused eager* path — a
baseline-path artifact, excluded) and TileLang attention **54.3%** (untiled O(n²)).

**This is our argument, made by a careful group, about our operator class, and they declined to put a
number on it.** The reason they declined is the reason our study exists: when the generated path
and the reference implement *different algorithms*, a ratio is not a measure of compiler quality.
They also note there is no anchor available: "for the fused, model-specific operators that motivate
DSLs in the first place, **no strong vendor baseline exists to anchor the comparison at all**"
(§I). Chunked linear attention is exactly such an operator — cuDNN/cuBLAS do not implement it, so
the "expert" is the algorithm's authors, not a vendor library.

### 4.2 Normalisation and reductions: measured, and the *reasons differ by cause*

- 2607.04454's reduction/normalisation collapses (up to 1347×) are **repairable authoring defects**,
  not structural: RC0a (a two-line `T.serial` → `T.reduce` rewrite plus native-dtype I/O) closes
  the entire TileLang family to PyTorch-comparable or better (LayerNorm 124% A100; max_reduction
  132%, logsumexp 582%, softmax 86%). The surviving exception is index-returning argmax at 27%, where
  the value-only reduction over the same shape hits 132% — "isolating the residual to the index
  pass."
- Read across `TAXONOMY.md`: the three RMSNorm cells over 100% (990% A100, 1303% GH200, 873% Triton)
  are **fusion artifacts of an unfused eager baseline** and are excluded as capability evidence by
  the paper itself (footnotes †, ‡). Do not use them as "DSL beats library".
- For a *reduction whose axis is the state axis* — our case — none of this applies. The relevant
  published facts are 2012.07145 §8 (reduction variables untileable, no rfactor/atomic) and Ansor's
  NRM result (parallelising the reduction loop was the entire win).

### 4.3 Is there any published comparison on a *recurrence* where a generated path lost for a structural reason?

**No. I could not find one, and this is the finding, not a gap in the search.** Specifically:

- **No autoscheduling study on a linear-attention / gated-recurrence operator exists that I could
  find.** 2012.07145's 17 programs, Ansor's operator suite, 2607.04454's 22 kernels, and
  TritonBench's 184 operators contain no chunked linear attention, no delta rule, and no
  state-carrying recurrence. 2607.04454's five categories are GEMM, attention, convolution,
  normalisation, element-wise/reduction.
- The nearest thing is the *architecture* literature, and its direction is the opposite of a
  compiler loss: the chunkwise/UT-transform algorithm is a **hand-written Triton kernel** that
  beats the sequential form (2406.06484 §3.2 "Speed comparison", and its Figure 1 — a chart I could
  not read numerically, so **no number is quotable**; a third-party lecture slide claims "up to 30×
  faster than sequential DeltaNet", which I mark as **unverified, secondary, from a course slide
  deck, not the paper**). KDA itself (arXiv **2510.26692**, Kimi Linear) is likewise a hand-written
  kernel: "We open-source the KDA kernel and vLLM implementations."

**Interpretation for the study owner:** the empirical question — *does a tile DSL, given the chunkwise
KDA algorithm, recover the hand-written kernel's performance?* — is unanswered. That is the novel
contribution, and 2012.07145 is not a baseline for it, because it was never asked the question. Note
also that both 2406.06484 and 2510.26692 chose **Triton** — a single-level tile DSL — as the target
language for these kernels, and needed no autotuner. That is weak evidence that hand-autoried Triton
is the state of the art here, and it means a study comparing "compiler-generated vs hand-written"
must be careful that its "compiler-generated" arm is not simply a worse version of what the
algorithm's own authors wrote in that same DSL.

---

## 5. Conclusion and the recommended wording

### 5.1 Plain answer

**No — 0.95× cannot be cited as an expectation-setting baseline for chunked linear attention.** It is
not merely out of domain; the reasoning that would extend it is a misreading:

1. The "one level of tiling" limit belongs to the **predecessor** (§0.1). 2012.07145's own space has
   three tiling levels; Ansor's has eight. Level count is not the boundary.
2. The suite is **17 imaging pipelines on one V100**, with **no recurrence, no attention, and no
   normalisation category** (§1.2). Its single GEMM had a **17×-smaller-than-average search space**
   (§1.3), so the number is nearly silent on GEMM-shaped problems.
3. The "expert" was a **Halide** expert tuning **schedules of the same algorithm** (§1.4) — so the
   comparison cannot see an algorithm change, which is precisely our decisive axis.
4. The real limit is **reduction-axis expressiveness** (§0.2, §2.1): the space cannot tile or unroll
   reduction variables and has no `rfactor`/`atomic`. For a gated recurrence that is the binding
   constraint.
5. The parity is **algorithm-conditional by construction**, and the paper says so in its own words
   (§0.3, §6.5): the human schedules won by using transformations "not currently in the search
   space of our autoscheduler."

### 5.2 The sentence for a notes file

> The widely-cited 0.95× result (arXiv 2012.07145, V100) shows GPU auto-scheduling matching expert
> *Halide schedules of the same algorithm* on 17 imaging pipelines, not on recurrence, attention, or
> normalisation — its single GEMM had a search space ~17× smaller than the suite average, and the
> experts' wins are attributed to transformations the space cannot express (splitting reduction
> loops, `rfactor`/`atomic`, warp shuffles), so it sets no expectation for chunked linear attention,
> where the decisive choice is to change which tensor is materialised at all.

Shorter, if one sentence is the budget:

> 2012.07145's 0.95× parity with expert hand-scheduling is measured within a fixed algorithm on
> imaging pipelines, and its stated residual — transformations outside the search space, chiefly
> reduction-loop splitting — is exactly the class of problem chunked linear attention poses, so it is
> not a usable expectation baseline for it.

**Strongest single citation for that sentence: arXiv 2012.07145 §6.5** ("Manual Schedules Outside the
Search Space"), read together with §8 ("Limitations & Future Work"). §6.5 names the four missing
transformations and says they are "not currently in the search space"; §8 states the mechanism
(per-Func rather than per-update-stage tiling, so reduction variables are untileable and
`rfactor`/`atomic` are unsupported). Together they are a self-declared, mechanism-level statement
that the space cannot reach our class of operator — which is a stronger argument than the geomean.

### 5.3 What *can* be cited instead

| Claim you want to make | Cite | Verified how |
| --- | --- | --- |
| The 256× is an **algorithm/work** change, not a tuning change; materialising the pairwise tensor is the thing to avoid | **2406.06484v6** §3.1 + §3.2 + fn. 1 | full text; quotes above |
| A tile DSL concedes on *tuning*; the escape is the algorithm — note the parenthetical "(for the same algorithm)" | **Triton Programming Guide**, ch. 1 intro | via `SOURCES.md` §5 |
| The compiler literature's own label for our situation is RC4, "Absence of Algorithm-Level Diversity" | **2607.04454v3** §V (RC4) | full text; quote above |
| Attention is excluded from that paper's headline comparison *because the algorithms differ*; no vendor baseline exists for fused model-specific operators | **2607.04454v3** §III-B, §I, Table I, Fig. 1 caption | full text; quotes above |
| Tile-shape tuning alone is worth ≈0 pp; widening the search is what pays | **2607.04454v3** §IV-C, RC2a | full text ("Δ ≈ 0 pp for both DSLs on every kernel") |
| A measured structural loss on conv and large GEMM against tuned vendor libraries | **2607.04454v3** Tables I/III/IV, §VI-E | full text |
| Published, enumerated auto-scheduling losses to humans, with mechanisms | **2012.07145** §6.5, §6.2, §8 | full text; quotes above |
| Single-level tiling is a real, documented failure mode — just not this paper's | **2012.07145** §6.2 (Sioutas et al.) | full text; quote above |

### 5.4 The gap map — where the novel contribution is

Three things the field does **not** have, each directly load-bearing for this study:

1. **No autoscheduling evaluation on a gated/linear-attention recurrence.** No published number
   exists for "tile DSL vs hand-written kernel" on chunked linear attention. The best available
   artefact is 2406.06484's *algorithmic* comparison (chunkwise vs recurrent, both hand-written
   Triton) — which does not involve a compiler-vs-human contrast at all.
2. **No published measurement of the materialisation trade-off at the compiler level.** The
   algorithm papers state the *asymptotics* (O(L²d) unparallelisable vs O(LCd + Ld²)); nobody has
   measured what a compiler does with the naive form, or how much of the naive form's cost is
   materialisation vs. redundancy. A [C,C,K] vs [BC,K] measurement with Nsight counters
   (bytes-to-HBM, achieved bandwidth) would be new.
3. **No study separating (i) from (ii) for this operator class** on a *recurrence*, with the causes
   decomposed the way 2607.04454 decomposes conv/GEMM. The taxonomy in `TAXONOMY.md` is complete for
   GEMM/conv/normalisation; it has no row for state-space or gated-recurrence operators.

**[Could not verify]** the per-program ratios behind 2012.07145's Figure 6 (image only), and
**[could not verify]** that no published autoscheduling-loss study exists anywhere (negative claim;
I searched and found none, which is weaker than a systematic survey). The chunkwise-vs-recurrent
speedup magnitude in 2406.06484 Figure 1 is also unreadable as a number — do not quote "30×" without
going back to the figure.

---

## Verification log

| Source | Version | Access |
| --- | --- | --- |
| 2012.07145 (Anderson, Adams, Ma, Li, Jin, Ragan-Kelley) | v2, 27 Aug 2023 = OOPSLA 2021, DOI 10.1145/3485486 | arXiv HTML full text read |
| 2607.04454 (Li, Rathnasuriya, Yang) | v3, 13 Jul 2026 | arXiv HTML full text read |
| 2406.06484 (Yang, Wang, Zhang, Shen, Kim) | v6, 15 Jan 2025, NeurIPS 2024 | arXiv HTML full text read |
| 2510.26692 (Kimi Linear / KDA) | v2, 1 Nov 2025 | abs page only — no numeric claim drawn from it |
| 2006.06762v5 (Ansor) | v5, 15 Oct 2023, OSDI 2020 | arXiv HTML full text read |
| Triton Programming Guide ch. 1 | — | via `SOURCES.md` §5, not re-fetched here |
| Songlin Yang, "DeltaNet Explained (Part II)" | blog, 2024 | **blog** — used only for its speed *comparison chart*, which is unquotable as a number |
| Ansor per-operator vs cuDNN table | — | **[could not verify]** — that table did not survive extraction; I use only the §7.1/§7.3 prose |

Cross-check note: 2406.06484 is the correct ID for the delta-rule paper (v6, 15 Jan 2025). The ID
`2407.06464` circulates in some citations and is **wrong**. KDA's origin is 2510.26692; a later,
distinct paper 2605.22791 ("Gated DeltaNet-2") also discusses KDA and is not the source of the
operator.
