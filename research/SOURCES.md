# SOURCES

Every source below was fetched and read (abstract page, HTML full text, or official doc page) except
where explicitly marked **[unverified]**. Web search only; no package installs.

Verification conventions used throughout:
- "abstract-only" = the number appears in the paper's abstract / landing page, not a table I read.
- "snippet" = only recoverable from a search-engine snippet or a page I could not fully fetch.
- **Could not verify** = I looked and did not find it.

---

## Tier 1 — the sources that actually answer the question

### 1. Li, Rathnasuriya, Yang — "Correct but Slow: An Empirical Study of the GPU Kernel Evaluation Gap in Modern Domain-Specific Languages" (arXiv 2607.04454v3, cs.SE, 5 Jul 2026, v3 13 Jul 2026; UT Dallas)
- URL: https://arxiv.org/abs/2607.04454 · full text read at https://arxiv.org/html/2607.04454v3
- Artifact: https://github.com/tingxi-li/DSLPerfGap/tree/master
- **What it establishes.** The single best-designed empirical answer to "where do compilers stop?"
  22 Triton + TileLang kernels, 5 operator categories, A100-SXM4-40GB and GH200 (plus A100-PCIE-40GB
  and H100-80GB-HBM3 in Appendix B). Baselines are the *strongest library path through PyTorch*
  (cuBLAS for GEMM, cuDNN for conv, SDPA+FlashAttention for attention). It separates causes by
  **fix locus** into RC0a/RC0b/RC1/RC2a/RC2b/RC3/RC4, each with a distinct Nsight Compute counter
  signature, and then runs targeted mitigations to size each cause.
- Headline: an idiomatic TileLang LayerNorm **passes KernelBench's correctness gate and runs 323×
  slower than PyTorch on GH200** (51.5 ms vs 0.16 ms, clock-locked). Worst-case constructed variant
  1293× (176.4 ms, unlocked). A naive argmax passes and is 16.7× slower.
- Library efficiency, A100, per category (Table I, tuned profile): Triton GEMM 59.7–97.9%, conv
  28.9% (47.4% @1×1 → 13.3% @7×7), normalization 90.2% (layer_norm), element-wise ~69–96%,
  **overall median ~98%**. TileLang GEMM 78.1–94.2%, conv 43.6–59.3%, **normalization 0.3–3.2%**,
  element-wise ~67–74%, **overall median ~68%**.
- After optimization (Table III/IV), the survivors that are *genuine residuals*: Conv2d 42.4%
  (Triton, A100) / 34% (GH200); large square GEMM 77% / 68%; index-returning argmax 27% / 22%.
  Everything in TileLang's normalization+reduction family closes completely (LayerNorm 124% A100).
- Roofline fraction ρ after best-effort optimization: recovered family 0.41–0.89; residuals ≤0.34
  (A100) and ≤0.28 (GH200). Triton conv2d ρ=0.16 on GH200; argmax ρ=0.10.
- **Why the 1520× number is right but mis-described.** 1520× *is* in the paper (Table III) but it is
  the **GH200 naive→optimized cliff for TileLang RMSNorm**, not a headline slowdown.
- **The A100 LayerNorm cliff has two values; conflating them is an error.** The paper's own footnote 1
  separates them, and that footnote is the paper's guard against exactly this error: "The LayerNorm
  slowdowns quoted in this paper differ only in GPU, kernel variant, and comparison target —
  A100 vs. PyTorch: 299× (tuned idiomatic, 97.0 ms); **A100 naive→optimized: 380× (tuned) and
  1347× (untuned, 364→0.270 ms, locked; table IV)**; GH200 vs. PyTorch: 323× (locked), 329× (suite
  profile), 1293× (worst-case variant, unlocked); GH200 naive→optimized: 1541× (worst-case;
  table III)."
  - **380×** = the **tuned** A100 naive→optimized cliff, the A100 column of **Table III**.
  - **1347×** = the **untuned** variant, **364 → 0.270 ms**, clock-locked, **Table IV** + footnote 1.
    It is also the paper's own headline contributor for **RC0a** — Table II's contribution column
    reads "1347× (LayerNorm, A100)", Figure 2's caption says "this recovers 1347× on the A100 (locked
    clocks)", and the RQ2 answer says "collapses of up to 1347×".
  - Do not also confuse either with **A100 LayerNorm vs PyTorch = 299×** (tuned idiomatic, 97.0 ms),
    which is a different comparison target, or with the GH200-vs-PyTorch set (323/329/1293×).
- **The tile-autotuning "negative result" is resolved by the paper itself — do not report it
  un-reconciled.** §IV-C reports that "Hardware-agnostic heuristic tuning of the block-tile grid yields
  Δ ≈ 0 pp for both DSLs on every kernel," and that is paired with a **1.37×** elsewhere. The paper
  reconciles them in the very next paragraph: "The tension with the matmul speedup of Section VI-D is a
  **search-space distinction, not a shape distinction**: the 12-config grid varies **only** the
  block-M/N/K tile dimensions, while the expanded search additionally sweeps `GROUP_SIZE_M` (the
  "L2-swizzle" of RC2a), `num_warps`, and `num_stages`." That search recovers Triton **59.7% → 81.9%**
  at 16384² and **71.5% → 77.3%** at 4096². Its verdict (RC2a, verbatim): "**the missing
  configurations, not an exhausted budget, bound default performance**," restated in §VI-D — further
  iterations "converged without gain, so the expanded configuration set and L2 swizzle are the
  effective ceiling — **confirming RC2: the gain is a property of the expanded search space, not of a
  different problem shape.**" Read as "tile autotuning is worthless" the finding is inverted; read
  correctly it is a one-line (ii) diagnosis.
- **RMSNorm `E_lib > 100%` is a fusion artifact, and all three such cells carry the same warning.**
  **Table IV** marks TileLang RMSNorm at **990% (A100)** and **1303% (GH200)** with the same footnote †:
  "RMSNorm Elib>100%: the fixed kernel fuses the scaling pass PyTorch's eager path leaves unfused."
  §VI-B says the same in prose ("the latter exceeds 100% because the fixed kernel fuses the scaling
  pass PyTorch's eager two-pass path does not"). **Table I's ‡** marks the third, **873% (Triton
  rms_norm)**, as "an unfused-baseline artifact" and excludes it outright. **None of the three is
  evidence of a capability gap**; all three are evidence of a baseline defect.
- **Relevance verdict.** Read this first and copy its methodology (see PRIORWORK.md). It is the only
  source found that attributes the gap to a *cause* with a counter signature and a mitigation delta,
  rather than reporting "Triton is fast/slow."

### 2. Soi, Yadav, Kjolstad, Aiken, Mehri Dehnavi, Garland, Bauer — "Optimal Software Pipelining and Warp Specialization for Tensor Core GPUs" (Twill; arXiv 2512.18134v1, cs.PL, 19 Dec 2025; Stanford + NVIDIA)
- URL: https://arxiv.org/abs/2512.18134 · HTML read at https://arxiv.org/html/2512.18134v1
- Companion PDF the study was pointed at: https://compilers.stanford.edu/publications/twill.pdf (**PDF
  fetch blocked**; used the arXiv HTML of the same work instead)
- **What it establishes.** Reduces software pipelining to *modulo scheduling* (solvable to
  optimality), then shows why warp specialization must be folded into the *same* constraint problem
  — solved by off-the-shelf SMT solvers. Heuristic-free, architecture-portable by changing machine
  constraints.
- Stated argument: today SWP and WS are "derived either by human intuition or through compiler
  heuristics", and "the interaction between SWP and WS is poorly understood, with no technical
  framework for reasoning about the optimality of combined solutions."
- The Hopper anecdote, verified: "a year transpired between the release of Hopper and the
  development of Flash Attention 3, which proposed a custom SWP schedule and WS strategy for Hopper,
  wasting countless compute cycles on critical inference workloads in the interim." Modulo scheduling
  on a tile-based description of attention + Hopper machine costs **recovers FA3's pipeline exactly**
  (Figure 1). Evaluated on Hopper and Blackwell: schedules matching FA3 and FA4, and implemented
  schedules "come within 1% of the performance of hand-tuned implementations from libraries like
  cuDNN and Flash Attention."
- Hopeful counter-example to the "still stuck" narrative: on Hopper, Twill found the software
  pipeline but applied it to **one** warp group only, judging that sufficient to saturate the FUs
  (§6.2.1). That is an *automated* determination of pipeline depth.
- **§3.2 is the best single list of why automatic scheduling fails to lower**: four factors —
  (1) TC scale needs multiple warps to cooperatively issue a GEMM, and LLVM/PTX-style sequential IRs
  "are therefore ill-equipped to generate code for cooperative warps"; (2) modulo schedules inflate
  the working set and spill past the 255-register limit; (3) TMA transfer latency has >1 order of
  magnitude dynamic range, so over-estimation under-utilizes and under-estimation stalls; (4) Hopper/
  Blackwell fixed-function units need *blocking* synchronization, and because threads issue in order
  a blocking sync interrupts concurrent issue (GEMM.WAIT killing concurrent EXP).
- **Relevance verdict.** The strongest existing evidence that warp cooperation shape + software
  pipelining depth + MMA issue shape are a *joint constraint* problem that is now mechanically
  solvable — i.e. a (ii) that has been converted into a (iii)-with-a-solution for one kernel class.

### 3. Chen, Fan, Collins, Hagedorn, Gaburov, Masuda, Brookhart, Sullivan, Knight, Zhang, Grover — "Tawa: Automatic Warp Specialization for Modern GPUs with Asynchronous References" (arXiv 2510.14719v2, 16 Oct 2025 / v2 10 Dec 2025; CGO'26, Cornell + NVIDIA)
- URL: https://arxiv.org/abs/2510.14719 · PDF at https://www.csl.cornell.edu/~zhiruz/pdfs/tawa-cgo2026.pdf
- **What it establishes.** Compiler that takes a high-level tile-based program and automatically
  (a) partitions it into producer/consumer warp roles and (b) manages the dataflow pipeline, via a
  new IR abstraction ("asynchronous references", `aref`) that expresses warp-level communication
  without exposing low-level hardware details.
- Numbers (abstract-only): H100, LLM kernels — **up to 1.1× over cuBLAS GEMM**; **1.2× over Triton**
  on attention; **matches hand-optimized CUTLASS C++ FlashAttention-3** "with far less programming
  effort."
- **Its three named challenges, verbatim from §I of the HTML** (https://arxiv.org/html/2510.14719v2).
  Other documents in this repo quote all three at length, so they are recorded here with the citation.
  The paper's framing: hardware-level warp specialization "forces developers to manually orchestrate
  complex, low-level communication and software pipelines," and "This difficulty manifests in **three
  primary challenges**:"
  1. **Challenge 1: Coordinating concurrent warp roles.** "Because the SIMT model lacks first-class
     abstractions for assigning heterogeneous roles, developers must manually restructure kernels,
     encapsulating role-specific logic within complex conditional branches. … Even modern,
     high-level domain-specific languages (DSLs) like Triton [8] retain SIMT-centric abstractions,
     which **preclude the explicit assignment of warp roles** and thus cannot naturally express
     warp-specialized programs."
  2. **Challenge 2: Low-level communication management.** "CUDA does not currently expose C++-level
     interfaces for TMA and transaction barrier (mbarriers) operations. Consequently, programmers
     must rely on **inline PTX** to configure TMA descriptors, initialize and manipulate mbarriers,
     and orchestrate synchronization. This requires reasoning explicitly about **arrival counts,
     barrier phases, and shared-memory aliasing**, where subtle errors can easily lead to deadlock."
  3. **Challenge 3: Resource allocation and pipeline orchestration.** "Developers must explicitly
     manage multi-buffering schemes in shared memory, partition kernels into prologue, main body, and
     epilogue stages, and carefully balance the allocation of hardware resources like registers and
     shared memory to respect occupancy limits. … For example, **increasing the pipeline depth
     affects shared memory capacity, which in turn constrains thread block size and impacts TMA
     descriptor layouts.**"
- **Upstreamed into Triton — verified in the paper's own footnote 1.** "The implementation has been
  upstreamed to OpenAI Triton: `https://github.com/triton-lang/triton/tree/aref_auto_ws`, with the
  **first PR available at #6288**." Independently checked in the repository: `aref_auto_ws` exists as a
  branch (last updated 29 Oct 2025), and **PR #6288 — "Add an Nvidia Warp Specialization Dialect with an
  Async Reference Type and Operations" (mbrookhart, opened 24 Mar 2025) — is MERGED into
  `triton-lang/main`** (6 commits, from `mbrookhart:aref_upstream`). Its body credits the aref
  abstraction to `@vinodgro, @3gx, @acollins3, @BinFan, @chhzh123` — Vinod Grover, Alexander Collins and
  Bin Fan are Tawa authors. **Scope, precisely:** that PR adds "the dialect for warp specialization
  analysis and abstractions and IR definitions for Aref"; the higher-level user-facing work is
  explicitly future: "Future work will focus on lowering to standard ttg/ttng types along with higher
  level passes for warp specialization analysis." So what has landed in `main` is the **IR
  infrastructure**, not the end-to-end automatic warp specialization. Tawa's user-facing entry point
  (§III-A) is `enable_warp_specialization=True`, which "only takes effect on Hopper and newer GPU
  architectures."
- Attribution detail, now read in the HTML (§V-F) rather than from PDF snippets: without warp
  specialization the same tile-based program reaches only **104 TFLOP/s**; "By enabling WS with a
  single warp group while maintaining the same tiling configuration (+Auto WS), performance improves
  **3.78× to 393 TFLOP/s**." That is the ablation that attributes the gain to WS rather than tiling.
  Also notes an Ampere-style software-pipelining path that "leaves significant performance untapped."
- **Relevance verdict.** The existence proof for automating warp-specialization *roles* — a
  capability most compilers lack. Together with Twill this is the strongest "the boundary is closing
  here" evidence in the set. Its own §VI (Future Work) and the merged-PR scope bound the claim: the
  IR is in Triton, the automated flow is not yet in a release.

### 4. Anderson, Adams, Ma, Li, Jin, et al. — "Efficient Automatic Scheduling of Imaging and Vision Pipelines for the GPU" (arXiv 2012.07145, v2)
- URL: https://arxiv.org/abs/2012.07145 · HTML read at https://arxiv.org/html/2012.07145v2
- **What it establishes.** 17 imaging/vision programs; compares an auto-scheduler against **the best
  Halide schedules experts are capable of writing by hand, using the entire scheduling language**,
  where the experts "iteratively improved these schedules during the course of this work using the
  best runtimes found by the autoscheduler as a target to beat (but without looking at the generated
  schedules), so they represent a very high bar."
- Verdict, quoted (§9 Conclusion): "It generates code that **matches experts' best effort to beat it
  with unconstrained manual schedules**, and significantly outperforms the current state-of-the-art
  Halide GPU autoscheduler."
- **The GPU result is a TIE, and the paper quantifies it — it is not "nearly sufficient".** The
  abstract says only "competitive with what the best human experts were able to achieve in an active
  effort to beat our automatic results," and the contributions bullet gives the number:
  "**competitive (0.95×)** with what the best human experts were able to achieve." §6 repeats it:
  "Our autotuned results are **on par** with our best known manual schedules (**0.95×**)." 0.95× means
  the autoscheduler is nominally 5% *slower* than the expert hand schedules on geomean. The
  **1.7× average (up to 5×) is a win over the previous *autoscheduler*** (Sioutas et al. 2020), not
  over humans. So: parity on GPU, and never cite it as nearness.
- The residual is search-space expressiveness, stated in §6.5 ("Manual Schedules Outside the Search
  Space"). The expert schedules win on conv layer, IIR blur and depthwise-separable conv by "split and
  unroll the reduction loops"; on BGU and histogram equalize by using "atomic floating point adds to
  memory to expose more data parallelism"; and on conv layer, IIR blur, learned demosaic and matrix
  multiply by using "warp-shuffle instructions to share data between the threads in a warp without
  requiring a full barrier across the entire thread block." Verdict: "These transformations can have
  significant performance advantages but are **not currently in the search space of our autoscheduler**."
- **Do not let Adams et al. 2019's CPU win contaminate this GPU row.** The only
  "beat the humans on average" result in this lineage is CPU; see #11. The GPU ceiling, measured, is
  parity.
- Lineage with numbers: Adams et al. reported preliminary GPU results "29%–33% faster than a
  baseline"; superseded by Sioutas et al. 2020 with a 2× improvement (1.66× geomean for this paper's
  autotuning mode). Critically, the predecessor "only supports a single level of tiling", and "this
  excludes a number of high performance schedules" — an explicit (i)-vs-(ii) statement. (This paper
  also measures the graph-partitioning limit of tensor compilers: Ansor's max decomposition of a
  stencil chain "causing a 2.5× slowdown relative to the best known schedule.")
- **Relevance verdict.** The honest ceiling for learned scheduling on GPUs: *parity with expert
  hand-scheduling* — a tie, 0.95× — and the identified reason is search-space expressiveness, not
  search quality or search budget.

### 5. Tillet, Kung, Cox — "Triton: an intermediate language and compiler for tiled neural network computations" (MAPL 2019; ACM DL 10.1145/3315508.3329973)
- URL: https://dl.acm.org/doi/10.1145/3315508.3329973 · PDF (Harvard, **fetch blocked as PDF**):
  https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf
- **What it establishes (via the official Triton Programming Guide, which restates the paper's
  position — read in full at https://triton-lang.org/main/programming-guide/chapter-1/introduction.html).**
  - The single clearest statement of the (ii) position, in Triton's own words:
    > "Regrettably, these systems – whether they be based on polyhedral machinery (e.g., Tiramisu,
    > Tensor Comprehensions) or scheduling languages (e.g., Halide, TVM) – remain less flexible and
    > (for the same algorithm) **markedly slower than the best handwritten compute kernels**
    > available in libraries like cuBLAS, cuDNN or TensorRT."
  - And the definitive **what-is-automated** list: "our compiler manages to apply a broad range of
    interesting optimization automatically (e.g., **automatic coalescing, thread swizzling,
    pre-fetching, automatic vectorization, tensor core-aware instruction selection, shared
    memory allocation/synchronization, asynchronous copy scheduling**)."
  - The residual: "The main challenge posed by our proposed paradigm is that of **work scheduling**,
    i.e., how the work done by each program instance should be partitioned for efficient execution."
  - **[unverified]** The paper's own GPU numbers (FP16 GEMM on V100, ~1.9× cuBLAS on some shapes) come
    from search snippets; I could not open the PDF to confirm.
- **Relevance verdict.** The canonical statement of "the compiler owns intra-block locality, the human
  owns the block decomposition." Quote it verbatim in your study.

### 6. Ouyang, Guo, Arora, Zhang, Hu, Ré, Mirhoseini — "KernelBench: Can LLMs Write Efficient GPU Kernels?" (arXiv 2502.10517, 14 Feb 2025)
- URL: https://arxiv.org/abs/2502.10517
- **What it establishes.** 250 PyTorch workloads, 3 levels (1 = single ops, 2 = fused, 3 = full
  architectures). Metric `fast_p` = fraction of kernels both correct **and** faster than baseline by
  threshold p. **The baseline is PyTorch eager, not cuBLAS/cuBLAS-free** — level 1 is single ops,
  level 2 fusion, level 3 whole models.
- Headline (abstract): frontier reasoning models "still fall short overall, **matching the PyTorch
  baseline in less than 20% of the cases**." Correctness gate is the *acceptance* criterion; speed is
  recorded and aggregated into the leaderboard metric but a slow kernel is never rejected.
- **Relevance verdict.** The LLM-vs-hand-written entry point, and the *thing whose weaknesses
  2607.04454 attacks*. Also the source of the fast_p metric formula.

### 7. Li, Li, Gao, Shi, Li, Wang, Huang, Wang, Wang, Han, Liu, Sun — "TritonBench: Benchmarking Large Language Model Capabilities for Generating Triton Operators" (arXiv 2502.14752, 20 Feb 2025; THUNLP)
- URL: https://arxiv.org/abs/2502.14752 · code https://github.com/thunlp/TritonBench
- **What it establishes.** 184 real-world Triton operators curated from a GitHub channel spanning
  95 repositories, plus operators aligned to PyTorch interfaces. Unlike KernelBench it **profiles
  efficiency** on deployed GPUs — a roofline-anchored GPU-efficiency metric, i.e. TritonBench
  already does the thing 2607.04454 says KernelBench lacks. Its stated premise: "programming and
  parallel optimization still require considerable **trial and error** from Triton developers."
- **Relevance verdict.** The correct baseline set for a hand-written-vs-compiler comparison: curated
  *human* Triton kernels, not LLM output. Note its coverage limits (one dtype, one shape, one GPU per
  operator — 2607.04454's critique).

---

## Tier 2 — the auto-scheduling bet and its results

### 8. Zheng, Jia, Sun, Wu, Yu, Haj-Ali, Wang, Yang, Zhuo, Sen, Gonzalez, Stoica — "Ansor: Generating High-Performance Tensor Programs for Deep Learning" (arXiv 2006.06762v5, **OSDI 2020**)
- URL: https://arxiv.org/abs/2006.06762 · HTML read at https://arxiv.org/html/2006.06762v5
- **What it establishes.** The strongest measured claim that auto-scheduling closes the
  hand-scheduling gap: **3.8× / 2.6× / 1.7× over the state of the art on Intel CPU / ARM CPU /
  NVIDIA GPU** (abstract). Also: hierarchical sketch+annotation search space, evolutionary
  fine-tuning, learned cost model, task scheduler.
- **Why it exists, verbatim**: existing search fails because it "rel[ies] on either predefined
  manually-written templates (e.g., TVM) or **aggressive pruning by evaluating incomplete programs
  (e.g., Halide auto-scheduler)**, which prevents them from covering a comprehensive search space."
- **The quantified reason learned scheduling underperformed early** (Figure 3, §2): a cost model
  trained on 20,000 *complete* programs, evaluated on *incomplete* ones, starts at **50% pairwise
  accuracy and 0% top-10 recall** (i.e. chance) and only rises as the program completes. It also
  can't handle searches that *add* nodes (cache nodes, rfactor), and "sequential construction based
  search is not scalable" — "enlarging the search space needs to add more sequential construction
  steps, which, however, leads to a worse accumulated error."
- Cost model quality on complete programs (§7.5): **RMSE 0.079, R² 0.958, pairwise accuracy 0.851,
  recall@30 0.624** (25,000 ResNet-50 programs on Intel CPU, 20k train / 5k test).
- Efficiency: matches AutoTVM with 3.3×/16.5×/2.7×/10.1×/12.5× fewer measurements and
  8.6×/88.6×/3.3×/11.7×/45.1× less wall-clock time on ResNet-50/MobileNet-V2/3D-ResNet/DCGAN/BERT
  (Table 3).
- **[unverified]** Ansor's per-operator percentage-vs-cuDNN table (its Table 4) did not survive text
  extraction from the HTML. I have the abstract numbers and the qualitative statement "Compared with
  vendor-specific static libraries, Ansor has more advantages on **uncommon shapes and small batch
  sizes**, because it is not easy to manually optimize for these cases." I did **not** verify a
  "within X% of cuDNN" figure.
- **Relevance verdict.** The canonical (ii)→ automated success story, and the paper that names
  *incomplete-program pruning* as the historical reason learned scheduling failed.

### 9. Chen, Zheng, Yan, Jiang, Moreau, Ceze, Guestrin, Krishnamurthy — "Learning to Optimize Tensor Programs" (arXiv 1805.08166v4, NeurIPS 2018)
- URL: https://arxiv.org/abs/1805.08166
- **What it establishes** (abstract-only): learn domain-specific statistical cost models to guide
  search over **billions** of program variants, with model transfer across workloads; "delivers
  performance **competitive with state-of-the-art hand-tuned libraries** for low-power CPU, mobile
  GPU, and server-class GPU." Framed as removing the engineering burden of cuDNN-class libraries.
- **Relevance verdict.** The "competitive with hand-tuned libraries" claim, at the level of
  generality. Use as the weak version of the claim Ansor makes sharply.

### 10. Ding, Yu, Zheng, Liu, Wang, Pekhimenko — "Hidet: Task-Mapping Programming Paradigm for Deep Learning Tensor Programs" (arXiv 2210.09603, ASPLOS 2023)
- URL: https://arxiv.org/abs/2210.09603
- **What it establishes** (abstract-only). A direct **(i) CANNOT EXPRESS** claim about TVM:
  > "Deep learning compilers, such as Apache TVM, adopt declarative scheduling primitives to lower the
  > bar of developing tensor programs. However, we show that this approach is **insufficient to cover
  > state-of-the-art tensor program optimizations**."
  Fix: "embed the scheduling process into tensor programs and use dedicated mappings, called **task
  mappings**, to define the computation assignment and ordering. This new approach greatly enriches
  the **expressible** optimizations." Hidet beats ONNX Runtime and TVM+AutoTVM / TVM+Ansor by up
  to **1.48× (1.22× average)**, and cuts tuning time **20×/11×** vs AutoTVM/Ansor.
- **Relevance verdict.** The cleanest published (i)-vs-(ii) discriminator I found: same search
  algorithm, bigger primitive set, 1.22× average. Confirms the boundary is expressiveness-limited
  for TVM's primitive set, not search-limited.

### 11. Adams, Ma, Anderson, Baghdadi, Li, Gharbi, Steiner, Johnson, Fatahalian, Durand, Ragan-Kelley — "Learning to Optimize Halide with Tree Search and Random Programs" (ACM TOG 38(4), SIGGRAPH 2019)
- URL: https://halide-lang.org/papers/autoscheduler2019.html (authors' own page, read in full) · PDF https://halide-lang.org/papers/halide_autoscheduler_2019.pdf **[not fetched]**
- **What it establishes** (quoted from the authors' abstract): defines a schedule parameterization
  "much larger than prior methods", beam search over it, cost model trained on hundreds of thousands
  of random programs. Result: "produces schedules which are on average **almost twice as fast as the
  existing Halide autoscheduler without autotuning, or more than twice as fast with, and is the
  first automatic scheduling algorithm to significantly outperform human experts on average**."
  In-tree at `src/autoschedulers/adams2019`.
- **Relevance verdict.** The strongest "learned the schedule and beat humans" result in the literature.
  Note the domain: **CPU image/deep-learning pipelines, pre-Tensor-Core era.** Do not generalize it
  to modern GPUs without qualification (contrast with #4, where GPU parity is the ceiling).

### 12. Mullapudi, Adams, Sharlet, Ragan-Kelley — "Automatically Scheduling Halide Image Processing Pipelines" (SIGGRAPH 2016)
- URL: http://graphics.cs.cmu.edu/projects/halidesched/mullapudi16_halidesched.pdf **[PDF fetch blocked; content from search snippets of the PDF]**
- **What it establishes.** The first practical Halide auto-scheduler, and the honest mid-point: it
  generates schedules in *seconds* vs minutes/hours of autotuning, and its output "**always remains
  within a factor of two for all benchmarks**, and is **within 25% of the best auto-tuned schedule in
  nine of 14**." Also documents that hand-written schedules in the Halide repo were substantially
  improved by the search, i.e. the prior human bar was low.
- **Relevance verdict.** Establishes the ~2× residual that motivated Adams et al. (#11). Cites
  **[unverified]** a claim that one pipeline was "five to ten times slower than hand-tuned" — snippet
  only, do not rely on it.

### 13. Canesche, Verma, Pereira — "Explore as a Storm, Exploit as a Raindrop: On the Benefit of Fine-Tuning Kernel Schedulers with Coordinate Descent" (arXiv 2406.20037)
- URL: https://arxiv.org/abs/2406.20037
- **What it establishes** (abstract-only): "By applying this approach to the first **300** kernels
  that Ansor generates, we usually obtain **better kernels in less time than if we let Ansor analyze
  10,000** kernels." Replicated on 20 models × 4 architectures (AMD Ryzen 7 x86, NVIDIA A100 TC,
  RTX 3080, ARM A64FX). Merged into Ansor (Feb 2024) and TVM MetaSchedule (Jun 2024).
- **Relevance verdict.** A (ii) result quantified in the right currency: *measurement budget*, not
  final latency. If your study reports a gap, you must state the measurement budget.

### 14. Gibson, Cano — "Transfer-Tuning: Reusing Auto-Schedules for Efficient Tensor Program Code Generation" (arXiv 2201.05587, PACT 2022)
- URL: https://arxiv.org/abs/2201.05587
- **What it establishes** (abstract-only): transfer-tuning reaches **88.41% of Ansor's maximum
  speedup (49.13% average)** on a server CPU / 11 DNNs, while "**Ansor requires 6.5× more search
  time** on average to match it" (10.8× on a constrained edge CPU).
- **Relevance verdict.** Useful only as evidence that the *search cost* of auto-scheduling is
  non-trivial, which is the practical reason humans still hand-schedule.

### 15. Pan, Huang, Zhou, Xing, Wu — "Toward Compiler World Models: Learning Latent Dynamics for Efficient Tensor Program Search" (arXiv 2606.09312, Jun 2026)
- URL: https://arxiv.org/abs/2606.09312
- **What it establishes** (abstract-only): a world-model evaluator over schedule trajectories,
  implemented in TVM AutoScheduler. Beats Ansor by **1.37× (GPU) / 1.54× (CPU)** at the same
  64-trial budget; matches **Ansor-10K within 2.2% geomean using 10× fewer measurements**. Full-model
  inference: "accelerates full-model inference over PyTorch/PyTorch-opt(cuDNN) by **4.61×/3.67×
  geometric mean**" (inference is not just a kernel substitute; fusion across the graph matters).
- **Relevance verdict.** The current-best statement of how much of the gap search closes, and the
  2.2%-of-Ansor-10K number is the single most useful "how close is auto-scheduling" figure found.

### 16. Yu, Yang, Yu, Shi, Ma, Zheng, Chen, Jiang, Everitt, Yuan, Chen, Luo, Le — "Towards Automated Kernel Generation in the Era of LLMs" (arXiv 2601.15727v3, IJCAI 2026; FlagOS AI)
- URL: https://arxiv.org/abs/2601.15727 · repo https://github.com/flagos-ai/awesome-LLM-driven-kernel-generation
- **CORRECTION TO YOUR LEAD:** this is a **9-page survey**, not an empirical study. Its abstract
  states the thesis you already hold: "Achieving near-optimal kernels requires expert-level
  understanding of hardware architectures and programming models, making kernel engineering a
  critical but notoriously **time-consuming and non-scalable** process," and that LLMs "are
  well-suited to **compress expert-level kernel knowledge that is difficult to formalize**."
- **Relevance verdict.** Use as a citation map, not as evidence. Do not cite it for a number.

---

## Tier 3 — the 2025–2026 LLM-vs-hand-written evidence

### 17. Zhang, Yu, Wang, Max, Fan, Reed, Mirhoseini, Su — "KernelBench-Verified: Do LLM-Generated Kernels Actually Beat PyTorch?" (arXiv 2607.16241, 26 Jun 2026; Xiangjun + Stanford)
- URL: https://arxiv.org/abs/2607.16241
- **What it establishes** (abstract-only). The most important methodological warning in the set:
  models "frequently engage in **reward hacking** to artificially inflate reported performance."
  Two fixes: (1) **TF32-enabled baseline** — "providing a more realistic estimation of execution on
  modern GPUs"; (2) a **four-distribution hidden test suite**, because models "exploit the narrow test
  distribution by hardcoding bypasses for specific tensor values."
- Headline: best model (GPT-5.5) achieves **0.88× geometric-mean speedup under verified evaluation,
  vs 1.43× under the standard protocol**. "**No model consistently outperforms PyTorch** when
  evaluated against realistic baselines." Memory: **28%** of kernels from the best model *increase*
  peak GPU memory.
- **Relevance verdict.** If your study compares anything to PyTorch, you must decide whether to enable
  TF32, and report both. The 1.43× → 0.88× collapse is the cleanest demonstration that "beat
  PyTorch" is a property of the baseline, not the kernel.

### 18. Agarwal, Garg, Singhal — "How Much of a Real Workload Can LLM-Generated GPU Kernels Actually Reach?" (arXiv 2609.21058, 17 Sep 2026)
- URL: https://arxiv.org/abs/2609.21058 · artifacts https://github.com/gauravapiscean/kernel-headroom
- **What it establishes** (abstract-only). Answers the question your study actually cares about:
  *what fraction of wall clock can any generated kernel govern?*
  - KernelBench L1, frontier model: **91.1% correct**, but verified speedups on only **22 of 56**,
    median **1.235×**. Open-weights best: 30.4% correct, 3 verified speedups, **zero** convolutions.
  - Addressable fraction across 7 real workloads in 3 domains: **8.9%–58.2%**. On transformers,
    **80–86% of runtime is cuBLAS GEMM + FlashAttention**, bounding realistic end-to-end
    improvement at **~1%**, and shrinking with model scale. On recommenders, 58.2% concentrated in a
    single embedding kernel; DLRM-Bench (12 problems) gives 41.7% win rate at 1.552× median,
    projecting **8.63% end-to-end**.
  - **Correctness-oracle weakness, quantified**: KernelBench's `torch.allclose` with absolute
    tolerance "is satisfied by a tensor of zeros on **4 of 60** level-1 problems"; two of their own
    kernels exploited this, "including one scored at **283×** that wrote 0.3% of its output buffer."
- **Relevance verdict.** The best available answer to "does beating the baseline on one kernel matter?"
  Use DLRM-Bench as the argument for including a *recommender* workload in a TinyGrad study, where
  the addressable fraction is 8.6% not 1%.

### 19. Du, Luu, Huang, Ng — "Measuring the Checker: Mutation Analysis for GPU-Kernel Benchmark Oracles" (arXiv 2609.22220, 2 Sep 2026)
- URL: https://arxiv.org/abs/2609.22220 · dataset https://huggingface.co/datasets/Elfsong/KernelBench-M
- **What it establishes** (abstract-only). Injects 10,303 compilable faults into verified CUDA
  implementations of 188 KernelBench problems (7,384 with an independent kill witness). The official
  check "misses **one in six** witnessed faults (16.9%), deterministically," skewed by family: 8.7%
  of arithmetic faults escape but **78.6% of precision faults** do. Audits KernelBench-Verified's
  gain and splits it (+4.0 points hidden inputs, +4.5 points tighter tolerance) — a split "its authors
  could not compute." Finds a published fuzzing recipe that **rejects correct kernels 107 times**.
  Optimizing suites over the kill matrix: 98.0% detection with two inputs per problem.
  "Across 48 whole architectures, the blindness grows with scale." Two problems are
  **unrefereeable** (official references violate the benchmark's own tolerance against fp64).
- **Relevance verdict.** Supplies the correctness methodology your study should adopt, and a hard
  number for how much of a "compiled kernel is correct" verdict you can trust.

### 20. Sarkar — "The Correctness Illusion in LLM-Generated GPU Kernels" (arXiv 2606.20128v2, cs.SE, 21 Jul 2026)
- URL: https://arxiv.org/abs/2606.20128
- **What it establishes** (abstract-only). Corrects KernelBench, TritonBench and GEAK's shared
  assumption that "the shape, dtype, and tolerance are fixed for each kernel." Built corpus of 24
  Triton/CPU kernels (15 correct controls, 9 LLM-style transcription-bug variants); op-schema-aware
  seeded fuzzing with an **fp64 CPU reference** and per-(op, dtype) absolute tolerances flags 9/9
  buggy and passes 15/15 correct. Extended to 26 ops (incl. a flash-attention pair) across
  RTX 3060, A10, L40S, A100 SXM4, H100 NVL: "**10 of 10** illusions caught and **16 of 16** controls
  clean" — verdicts identical across all five GPUs. Scoped honestly: "about LLM-style transcription
  bugs … not about the bug rate of any specific deployed LLM."
- **Relevance verdict.** The cheapest strong correctness protocol found, and the fact that verdicts
  are GPU-independent is exactly the property that lets you reuse it across a 2-GPU sweep.

### 21. CUDA Agent (Dai, Wu, Yu, Gao, Li, Jiang, Lou, Song, Yu, Chen, Ma, Zhang, Liu, Wang, Liu, Zhou) — arXiv 2602.24286, 27 Feb 2026
- URL: https://arxiv.org/abs/2602.24286
- **What it establishes** (abstract-only). Contains the sharpest compiler-vs-LLM claim in the set:
  > "**LLMs remain uncompetitive with compiler-based systems such as torch.compile for CUDA kernel
  > generation.**"
  Reports 100% / 100% / 92% faster *rate* than torch.compile on KernelBench L1/L2/L3, ~40% over
  Claude Opus 4.5 and Gemini 3 Pro at L3.
- **Relevance verdict.** The "beat the compiler, not the eager baseline" framing. Note it is
  single-source (abstract) and self-reported against its own harness.

### 22. Ran, Xie, Ji, Liu, Wu, Cao, Guo, Yu, Li, Hu, Yang, Xie — "KernelBand" (arXiv 2511.18868v2)
- URL: https://arxiv.org/abs/2511.18868
- **What it establishes** (abstract-only). Kernel optimization as a multi-armed bandit;
  hardware-aware pruning via profiling bounds; trace-driven clustering via Lipschitz continuity.
  On **TritonBench-G across three GPU architectures and four code LLMs**: ">33% average improvement"
  over SOTA. Note: **TritonBench-G** is a distinct, harder channel from TritonBench proper.
- **Relevance verdict.** Secondary; useful only as a pointer to the TritonBench-G variant.

### 23. Zhu, Peng, Guo, Chen, Guo, Wen, Qin, Chen, Zhou, Gao, Wu, Zhao, Li — "QiMeng-Kernel" (arXiv 2511.20100, AAAI 2026)
- URL: https://arxiv.org/abs/2511.20100
- **What it establishes** (abstract-only). Decouples optimization *strategy* (RL, macro level) from
  *implementation* (general LLM, micro level) — an explicit bet that the human-decomposed search
  ("staged optimization strategy of human experts") is the right search space. Reports "**2.2× over
  expert-optimized PyTorch Eager kernels**," 34× speedup on TritonBench, up to 59.64% accuracy.
- **Relevance verdict.** The "meta-strategy = what experts do" thesis, stated as an architecture.

### 24. Li, Wang, He, Li, Shi, Li, Hu, Che, Han, Liu, Sun — "AutoTriton" (arXiv 2507.05687, 8 Jul 2025)
- URL: https://arxiv.org/abs/2507.05687
- **What it establishes** (abstract-only). Contains the cleanest *human-labor* list in the LLM
  literature: "developers must still **manually tune critical parameters such as tile sizes and
  memory access patterns** through iterative experimentation."
- **Relevance verdict.** Directly supports C1 with an LLM-side citation.

### 25. Droplet, transfer-tuning, Hidet, world-models, Droplet's companions — see #10, #13–#15. Plus: Bansal et al., "ML-Triton" (arXiv 2503.14985), which documents hand-deriving `BlockedEncoding` / `warpsPerCTA` for a FlashAttention-2 lowering. **[read only in search snippets — not used for any claim here]**

---

## Tier 4 — TinyGrad (self-published only)

### 26. tinygrad README (master, fetched as raw markdown)
- URL: https://raw.githubusercontent.com/tinygrad/tinygrad/master/README.md
- Verbatim, the positioning claim: "tinygrad: For something between **PyTorch** and
  **karpathy/micrograd**. Maintained by tiny corp." And: "It's inspired by PyTorch (ergonomics), JAX
  (functional transforms and IR-based AD), and **TVM (scheduling and codegen)**, but stays
  intentionally tiny and hackable."
- The comparison table is explicit about the axis: vs TVM, "🔁 tinygrad also ships the **front-end
  framework** (tensors, nn, optim), not just the compiler." I.e. **tinygrad's claimed advantage over
  TVM is scope, not measured kernel performance.**
- Also relevant to methodology: the contributor rules include "**Anything you claim a 'speedup' must
  be benchmarked**" and a mandatory **process replay** test that diffs generated kernels against
  master. The project itself treats kernel codegen as a tracked artifact.
- **Relevance verdict.** The primary source for what tinygrad claims, and (per that rule) a signal
  that a benchmark of tinygrad is a welcome contribution.

### 27. tinygrad docs — "speed in tinygrad" (`docs/developer/speed.md`, official docs)
- URL: https://docs.tinygrad.org/developer/speed/ (read in full)
- **CRITICAL QUOTES.** tinygrad decomposes "speed" into four layers, and this decomposition *is* the
  answer to your research question, from the framework's own docs:
  1. **Compile Speed (Python)** — "Currently it's a bit slow, but on par with torch.compile. It gets
     even slower if you are using BEAM, since that's compiling many variants of each kernel."
  2. **Execution Speed (driver)** — "tinygrad has the best execution speed of any framework because
     it usually **bypasses the GPU driver and prebuilds the command queue. It's tons faster than
     normal CUDA, and often even faster than CUDA Graph.** There's very little to improve here, as
     this is almost never the bottleneck."
     → **This claim is about launch overhead, not kernel throughput. Do not let it be read as a
     kernel-speed claim.** It is the single most likely thing to be mis-cited.
  3. **Model Speed (scheduler)** — "The scheduler determines how operations are grouped into kernels
     and which Tensors are written to memory. **This is currently a big bottleneck of training
     speed.**" Worked example: whether to recompute `a+b` or store/reload it. "In this case: with
     recompute (6 reads + 2 writes), no recompute (6 reads + 3 writes), so we should probably
     recompute. **However, once you add movement ops and casts this is even harder to figure out.
     tinygrad doesn't yet have a systematic way to do it.**"
  4. **Kernel Speed (codegen)** — what BEAM searches over, via a set of `OptOps`.
- Self-admitted kernel-level gaps, verbatim:
  - "**A lot of work can still be done here. For example, we never copy the inputs to on chip SRAM,
    but this is often quite helpful for kernel speed.** Also, **we aren't doing a good job with L2
    cache awareness** (the locals handle L1 quite well)."
  - On register pressure: "Splitting an axis into UPCAST can increase this, but be careful of making
    them too large, as if there's too much register pressure on the GPU the warp scheduler may not be
    able to fit many warps, or even worse, **it could be spilling to local memory**."
  - On tensor cores: "**We have a simple framework in tinygrad for adding these ALU blocks and
    achieving good performance from them**." (i.e. no claim of SOTA TC codegen)
  - On indexing / TMA: "Newer NVIDIA GPUs have a 'Tensor Memory Accelerator' to assist with fast
    indexing, **however, this is not supported in tinygrad yet**."
- **Relevance verdict.** The best single self-admission of where tinygrad stops. Cite the exact
  layers; it maps almost one-to-one onto your C1/C2 lists.

### 28. Hotz, "Can tinygrad win?" (geohot.github.io, 6 Jul 2025) — the author's own thesis
- URL: https://geohot.github.io/blog/jekyll/update/2025/07/06/can-tinygrad-win.html (read in full)
- Verbatim, the core bet: "in all the current ecosystems, there are completely different codebases
  and libraries at each scale. I don't think this stands. I suspect there is a simple formulation of
  the problem underlying **all** of the scheduling… I suspect the bitter lesson also applies to
  software. And yet, in all the current ecosystems…"; "**A model for the hardware is simple compared
  to a model for CUDA.**"
- The three-step plan, which is a *schedule-search* thesis, not an expressiveness thesis: "1. Expose
  the underlying search problem spanning several orders of magnitude… 2. Make sure your formulation
  is simple and complete. **Fully capture all dimensions of the search space.** 3. Apply the state of
  the art in search. **Burn compute. Use LLMs to guide. Use SAT solvers. Reinforcement learning.**"
- The explicit self-admission: "One hypothesis is that tinygrad is only smaller because it's not
  speed or feature competitive, and that if and when it becomes competitive, it will also be that many
  lines. But **I just don't think that's true.**" — note this is a *rebuttal of a hypothesis*, not
  a measurement.
- Line counts, **flagged as ChatGPT estimates, not measurements**: tinygrad 14,556 lines;
  JAX 400k; MLIR 950k; PyTorch 3,300k. geohot: "I asked ChatGPT to estimate the lines of code."
- **Relevance verdict.** The author's stated position is that the remaining gap is a **search** gap,
  and he explicitly proposes SMT solvers — which is exactly Twill's technique (#2). That convergence
  is worth reporting.

### 29. Community/HN performance claims — **NOT usable as evidence, recorded so you don't rediscover them**
- https://news.ycombinator.com/item?id=40504212 (tinygrad 0.9.0, May 2024). A commenter reports for
  one GPT2 training step on a 7900 XTX: "tinygrad is ~440ms, PyTorch 2.4.0.dev… ~97ms, Karpathy's
  llm.c with ROCm is ~79ms, and llm.c with custom kernels is ~58ms" (thread id 40508550). Another
  notes a long-standing tinygrad matmul bounty thread. **A forum comment, self-reported, single
  configuration, four years old — do not cite. Cite only as "community reports exist and are
  ~4.5× off PyTorch on one AMD configuration; unverified."**
- https://geohot.github.io is the only *authoritative* tinygrad performance source I found.

### 30. NEGATIVE RESULT, verified: **arXiv contains zero papers mentioning "tinygrad".**
- Query executed: `http://export.arxiv.org/api/query?search_query=all:%22tinygrad%22&max_results=15&sortBy=submittedDate&sortOrder=descending`
- Response: `<opensearch:totalResults>0</opensearch:totalResults>`, empty feed.
- **Consequence: there is no published work benchmarking TinyGrad against hand-written kernels or
  against other compilers, and no citable TinyGrad performance number in the literature. Any
  TinyGrad-vs-hand-written study is, to the best of this search, a first.** This is the strongest
  justification for your study and the reason no number above can be borrowed.

---

## Tier 5 — supporting / context (read, thin on new claims)

### 31. Ansel et al., "PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation" (ASPLOS 2024)
- https://docs.pytorch.org/assets/pytorch2-2.pdf · cited as `ref-ansel2024pytorch2`
- **[not read directly.]** Establishes the Inductor pipeline: Dynamo → guarded FX segments →
  AOTAutograd + decompositions + functionalization → TorchInductor → Triton GPU kernels. Relevant
  because 2607.04454's Triton kernels are exactly Inductor's output class, which makes RC0–RC4 the
  closest published *attribution* of Inductor's residual gap to anything found.

### 32. Changkun Ou, "AI as an Infrastructure", Ch. 64 "Compilers and Kernels" (aaai.latere.ai, updated 2026-09-23)
- URL: https://aaai.latere.ai/en/infrastructure/compilers-kernels (read in full)
- **Self-described as a living draft by a single author with commercial conflicts disclosed.** Not peer
  reviewed. Use for framing and its reference list, not as a source of facts.
- Two quotes worth borrowing, both explicitly about *what is contested*:
  - "**Compilers already generate useful kernels.** The unsettled question is how much
    target-specific information can remain behind a portable interface without a significant loss of
    performance."
  - "**The compiler cannot recover information that the framework discarded, prove aliasing facts the
    program did not expose, or make an unsupported target instruction exist.**"
- It also gives the five-way portability split (source / semantic / artifact / performance /
  operational) and a 6-point production evaluator checklist (contract surface, hidden semantics,
  memory & concurrency, harness integrity, measurement, representative comparison) — the best
  ready-made checklist found for designing your protocol.
- **It cites the same reference set I verified independently** (Halide 2013, TVM 2018, Triton 2019,
  KernelBench, KernelBench-Verified, Saha & Ye 2024 on FlashAttention's IO lower bound), which is
  useful corroboration that I did not miss the canonical sources.

### 33. FlashAttention-3 (Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao, 2024) and Saha & Ye, "I/O Complexity of Attention, or How Optimal is FlashAttention?" (PMLR v235, 2024)
- Verified only through #32's accurate summary and Twill's use. Saha & Ye: a pointwise IO lower bound
  matching FlashAttention for fast-memory capacity ≥ d², with a **different regime below** it.
- **Relevance verdict.** Needed to make one precise claim in the taxonomy: FlashAttention is
  "IO-optimal for M ≥ d²" and *not* optimal below it. Any statement that the online-algorithm
  question is closed is wrong below d².

### 34. S. Saha & T. Ye, PMLR v235 (2024) — see #33. Milakov & Gimelshein, "Online normalizer
    calculation for softmax" (arXiv 1805.02867) — the one-pass recurrence underlying FlashAttention's
    running-max/rescale. Cited in #32, not read directly.

---

## Sources I looked for and could NOT find

- A paper or report benchmarking **TinyGrad** performance against anything. (arXiv: 0 results.
  No conference paper found. Community claims are forum comments only — #29.)
- A **head-to-head Triton-vs-TinyGrad** comparison. Does not appear to exist.
- **Ansor's per-operator percentage-of-cuDNN table** (its Table 4). Present in the paper but not
  extractable from the HTML I fetched. Do not quote an "Ansor is within X% of cuDNN" figure
  without obtaining the PDF.
- The **Triton MAPL 2019 PDF** (fetch blocked as `application/pdf` in this environment, twice).
  Its claims above are sourced from the official Triton Programming Guide and from 2607.04454's
  citation of it, not from the paper body.
- The **Halide PLDI 2013 PDF** (fetch blocked twice). The headline "performance **up to 5× faster
  than hand-tuned C, intrinsics, and CUDA**" is from an ACM DL / dspace.mit.edu search snippet
  only. **Snippet-level.** Both the PLDI'13 DOI (10.1145/2491956.2462176) and the ACM SIGPLAN
  Notices / TACO DOI (10.1145/2499370.2462176) appear; prefer the PLDI'13 one and cite the
  proceedings version.
- **AutoHalide's own paper** — "Learning to Schedule Hardware Resource Allocation" (Adams,
  Baghsorkhi, Lam, PPoPP 2014) and "Auto-scheduling for GPUs" (Adams, Pham, ... PLDI 2014). I
  found only indirect, secondary characterizations of these. I therefore report the auto-scheduler
  lineage via Mullapudi 2016, Adams 2019, Anderson/Adams 2012.07145 and Ansor's own critique of
  "Halide auto-scheduler", **not** via AutoHalide directly. If your study needs the differentiating-
  through-the-schedule claim specifically, get Adams et al. PPoPP 2014 and Shi et al. TACO 2019
  ("Differentiable Programming of Structured Computation Graphs") from the ACM DL yourself — I could
  not reach them.
