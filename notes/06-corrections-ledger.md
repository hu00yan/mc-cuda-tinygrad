---
kind: changelog
---

# Corrections ledger

The register of record for claims this study has had to take back.

**Why this file exists.** `notes/DECISIONS.md` keeps a "Claims I got wrong"
table, but it is a section inside a decision log, it is not append-only, and it
does not record *who* found the correction or *what evidence* settled it. This
study's owner has now had to correct several claims, and a correction that is
not written down is a claim that will be repeated — twice already, in this
repo's own files, in two different forms. This file is the append-only,
cross-referenced version.

**Relationship to `DECISIONS.md`.** `DECISIONS.md` remains the log of *what was
decided and why*. It is not duplicated here. Where a correction is already
recorded there, this file carries a pointer row marked `already in DECISIONS.md`
rather than a second copy of the text, **plus any amendment** — because two rows
in this register amend rows in `DECISIONS.md`, and a reader who consults only
`DECISIONS.md` will get the superseded version.

**Conventions for status tags.** Used throughout, and deliberately loud:

| tag | meaning |
|---|---|
| `[MEASURED]` | executed on hardware/software, number reproducible from a script in this repo |
| `[MODELLED]` | arithmetic on known shapes; exact given the inputs, but no measurement of the quantity itself |
| `[PREDICTED]` | a claim about what *would* be observed. **Not a result.** Never to be cited as one |
| `[READ-FROM-SOURCE]` | read out of a file (ours or a third party's), not executed |
| `[UNVERIFIED]` | could not be checked in this pass. Carried forward as unknown, not as true |

A row's evidence column must contain a path, a line number, or a command. A
correction with "the audit found it" and nothing else is not admissible here.

---

## 1. The register

Newest first. IDs are stable and never reused.

| id | date | file | original claim | correction | evidence | status |
|---|---|---|---|---|---|---|
| **C-26** | 2026-09-29 | `notes/33-tinygrad-axis.md`'s opening block (pre-edit) — "**What the ladder already shows, `[MEASURED]`, and it decides the framing:** `ew` 0.97×, `gather` 0.97×, `red` 1.9×, `gemm32` 4.2×, `gemm16` 1.3×" | Those five ratios labelled **`[MEASURED]`**, under a definition in the same paragraph reading "`[MEASURED]` means **executed on a T4** by a script in `axis/`". | **The note's own §7 and §9 say the T4 timing table does not exist** — §7 opens "the timing table is the queued part", §9 "The T4 timings are queued", the stated reason being Kaggle's **2-concurrent-batch-session cap** with this study's other agents holding both. And `out/axis/` contains two run logs and **no `axis_report.json`**. So the five numbers were labelled with a provenance the same document denies them, and I was one step from reporting them to the user as measured. Now `[NOT-T4-MEASURED]`, with the number kept because the *shape* is the point and the T4 run is being asked to confirm it | `notes/33` §7 and §9; `out/axis/` contains `20260929-1543{57,09}-run.log` and no report; the five figures appear nowhere else in the note, so **their actual origin is not established and is left open** rather than guessed | `[NOT-T4-MEASURED]` — **tag withdrawn; the five ratios are unconfirmed and their provenance is open** |
| **C-27** | 2026-09-29 | the remote-run path, in three separate places, all found on 2026-09-29 | That a run's *result* is attributable to the run that fetched it. | **It is not, anywhere, and this produced three independent wrong-or-void results today.** (i) `gpush`: a cached Kaggle notebook carries the previous `PAYLOAD_FINGERPRINT` and the previous manifest, so it self-verifies and reports `MATCHED` while executing the **previous** payload — fixed by `gpush.cli.payload_provenance`, ledger note in `notes/22`; (ii) `axis/kaggle/run.sh`: **`kaggle kernels push` prints `Kernel push error: Maximum batch GPU session count of 2 reached` and exits 0**, the script ignored the message, polled, saw the previous run's `COMPLETE` and downloaded **its** output — **two rounds of the harness were "measured" against stale code**; (iii) a T4 timing session that executed the previous run's payload and reported `COMPLETE` in 9 s. **The shared defect is that every one of these returns a plausible previous result rather than an obvious error, and a plausible previous result is the one thing a check cannot detect by looking at itself.** The rule that follows is narrow and mechanical: **a remote-run helper must test the push's exit status, grep stdout for a refusal, and compare the returned fingerprint to the one it sent — or it must not be allowed to print a ratio** | `gpush/cli.py:payload_provenance`; `axis/kaggle/run.sh` (now refuses to poll on a push error); Kaggle's exit-0-on-refusal behaviour, observed not documented | `[MEASURED]` — three instances, one mechanism |
| **C-24** | 2026-09-29 | `derivation/minecu_gap.py` in full (written 2026-09-29); `notes/06` C-14, whose basis it computes; `notes/19` §4.1 and the 47.2× it licenses | The note's own accounting, re-derived by a script: total 47.16×, terms 1.21× (early exit) × 4× (predicated byte loads) = 4.85×, **"UNATTRIBUTED RESIDUE 9.73×"** — presented as a missing row in the table, with a falsifiable ablation target derived from it. | **The division is a category error, so the residue is arithmetic on two different quantities and means nothing.** The generated kernel is vectorised at roughly **16 pixels per thread**; the hand-written reference is **1 pixel per thread**. The two instruction counts therefore cover work that differs by more than an order of magnitude, and dividing them does not produce a ratio of anything. Properly normalised **per pixel per DDA step**: the naive port is **74.4 SASS against the reference's 38 = 1.96×**, and the optimised port is **53.5 = 1.41×**. Not 47.2×. **The 9.73× residue was 9.73× of a units error** — I built a script, derived a number from it, wrote it into the ledger as a finding, and quoted it to the user as the missing term. The missing term was the normalizer | `notes/23-good-vs-naive.md` (53.5 and 74.4 per pixel per step, against the reference's 38 measured period); the vectorisation width is the generated kernel's own, from the SASS capture | `[MEASURED]` — **47.2×, 4.85×, 9.73× and the ablation target derived from it are all withdrawn as quantities; `derivation/minecu_gap.py` is marked void** |
| **C-25** | 2026-09-29 | `notes/19` §5.4, "no per-ray early exit … 32 steps vs a measured 26.4 mean — a **1.21×** term"; and `derivation/minecu_gap.py`'s early-exit lambda, which encoded the same 26.4 | The tensor form runs all 32 steps for every ray while the reference averages 26.4, so the term is 32/26.4 = 1.21×. | **Measured at 1.94× — the reference's mean is 16.54 steps, not 26.4.** `32 / 16.54 = 1.94`. **RESOLVED against 26.416:** a later run measured **16.5384** steps per pixel with all five rungs byte-identical to the hand-written driver's own output over 1 048 576 pixels, so two independent measurements agree on 16.54 and the single 26.416 figure has not been reproduced by anything since it was written. The 1.21× was the smaller error, which is the part worth noticing: it made the port look **better** than it is | `notes/36-rung-timings.md` (16.5384, byte-identical gate); `notes/23-good-vs-naive.md`; against `out/minecu-t4/` and `minecu_ref/buried_agent_fix.md` for the unreproduced 26.416 | `[MEASURED]` — **1.21× withdrawn; 1.94× substituted; the 26.4-versus-16.54 conflict is closed against 26.416** |
| **C-21** | 2026-09-29 | the KDA port's headline accuracy figure, "worst relative error **8.6e-06**", carried in the project summary and in the rewrite brief | "The existing port meets a worst relative error of **8.6e-06**" — stated as a property of `kda/tiny_kda.py` and repeated in at least one brief. | **The repo's own test reports 6.64e-06, and the figure could not be reproduced at all.** `kda/test_tiny.py` prints 6.64e-06; a fresh measurement of the same port against the same oracle gives **1.106e-06** on CPU and 1.381e-06 for the rewritten form on Metal. So three different numbers have been in circulation for one artefact, none of them traceable to a command in this repo. Nobody caught it because 8.6e-06 and 6.64e-06 are both "about 1e-05" and both pass the gate — **a number that only has to be approximately right will not be caught, so it should not be quoted at all** | `kda/test_tiny.py` (prints 6.64e-06); `notes/24` §correctness (1.381e-06 CPU / 6.096e-06 Metal, 9/9) and the naive control at 1.106e-06 | `[MEASURED]` — **8.6e-06 withdrawn as unreproducible** |
| **C-22** | 2026-09-29 | `notes/31-ut-transform-pricing.md`'s launch and REDUCE counts as absolute quantities | The table's "**launches**" column — A = 5 380, B = 86, C = 44 — presented as the quantity to compare implementations on, computed by an **offline census with no device present** (`DEV=CUDA`, no card). | **An offline census does not report the kernel count, and on this operator it is wrong by enough to matter: a real device emits 6 021 kernels for A and 24 for C, not 5 380 and 44.** The direction is consistent — TinyGrad does not keep a carried chain in registers, so kernels can *exceed* source op count, and the physically serial depth is the DAG longest chain, measured at **381** for A. **The ranking survives; the absolute counts do not**, and any argument that cites "5 380 launches" as a measured cost is citing a census. This compounds C-20: that one said a subgraph census does not decompose a graph, this one says a whole-graph census is still not the kernel count | `notes/24` (device-measured: 20 354 / 89 uops, 6 021 / 24 kernels, chain 381 / 89, 270.6 ms / 1.12 ms on Metal); `notes/31` (offline: 5 380 / 44) | `[MEASURED]` — **ordering retained, absolute counts withdrawn** |
| **C-23** | 2026-09-29 | `kda/ref.py:285` — the docstring of the float64 oracle. **I edited this file on the strength of the claim below and have reverted the edit.** | A rewrite lane reported: "`kda/ref.py:277-287` says `_forward_substitution` returns `(I+StrictTril(D))⁻¹`. It returns `(I−D)⁻¹` — measured against `np.linalg.inv` at both signs (**4.65e+00 vs 3.55e-15**)." I relayed that as a docstring bug and amended the oracle. | **Wrong, and I propagated it before checking. The original docstring was correct.** The measurement is what failed, not the docstring. **`_forward_substitution` mutates its argument** — `a` is a reshape *view* of `A` (`ref.py:296`) and the loop overwrites the caller's array — so a verification that builds its expected value from the same array afterwards compares the inverse against the matrix that was already inverted. Both of us made it independently: the lane got ~9e+00, I reproduced the identical error with my own script before reading the function. Empirical contract, measured: **passing a strictly-lower `L` returns `(I − L)⁻¹` to 5.551e-16**, so with the caller's `A = −StrictTril(D)` the return is `(I + StrictTril(D))⁻¹` — exactly what the docstring said and what `SPEC.md` §3.3 and §7.1 pin. The code and both documents were right all along | `kda/ref.py:296` (the view), the `−Btri` call site, and a direct probe: `f(L)` versus `inv(I−L)` = 5.551e-16, versus `inv(I+L)` = 3.490e+00. The docstring now states the in-place contract instead, which **is** a real gap — it was documented only as the phrase "in place by row-wise forward substitution" | `[MEASURED]` — **the reported sign error is retracted; the in-place hazard is real and is now stated** |
| **C-19** | 2026-09-29 | `notes/26-tinygrad-idiom.md` §6c | "**fp16 needs N≥8, M≥64, K≥8** — which is exactly KDA's `C=64` matmul shape, so a port that reaches for a matmul there gets FFMA *because the shape is small*." | **The `M≥64` gate is an artefact of the sweep, and the conclusion built on it is wrong in the direction that matters.** Swept only at `K=64`, the gate reads `M≥64`; **at the merge's own shape the gate is `M≥16, K≥8`, and `[16,16]` is inside it** — measured **17 `__WMMA`**, i.e. a real tensor-core kernel. So a `[16,16]` fp16 matmul *does* reach a tensor core, and the reason a KDA port sees FFMA is the **dtype** (the study runs fp32, and `notes/26`'s other finding — fp32 never reaches a tensor core at any shape — is confirmed) **and not the shape.** The reference's merge runs in **tf32**; this study's fp32 gets FFMA. **The comparison will therefore report a wider tensor-core gap than the shape analysis predicted, and the shape analysis is what would have been used to explain it away** | `notes/31-ut-transform-pricing.md` (re-measured at the merge shape), cross-check against `notes/26`'s own table which the claim contradicts | `[MEASURED]` — **the fp32 finding stands; the fp16 shape gate does not** |
| **C-20** | 2026-09-29 | any claim of the form "candidate X is *n*% of this operator", in `notes/26`, `notes/29`, `notes/31` | Subtracting or comparing a **subgraph** census against a whole-graph total to attribute cost — e.g. "the substitution is 44.0% of the operator's ALU-lane ops". | **A subgraph census does not decompose a graph, so those percentages are not measurements of anything and must not be read as shares.** The tell is arithmetic: candidate A alone reports **192.6M** lane ops while the *entire* operator reports **57.8M** — the part is 3.3× the whole. Fusion removes the intermediate traffic a per-op census counts, so a subgraph's op count and the whole graph's op count are not on the same scale. **Only the relative ordering between candidates measured the same way is load-bearing** — A>B>C and D<D2 in that table stand; every "% of the operator" does not. Two of the subagent's own bugs make the point sharper: a `Tensor.expand` shape-argument bug transposed the diagonal blocks by **3.4× and still passed a shape check** | `notes/31-ut-transform-pricing.md` (the A-versus-whole contradiction, stated by the author); `tinygrad/mixin/movement.py:162` | `[MEASURED]` — **scope withdrawn, ordering retained** |
| **C-18** | 2026-09-29 | `notes/06-corrections-ledger.md` C-15, the figure "**4 032 sequentially dependent operations**"; and the same figure as quoted in `kda/SPEC.md` §3.3's correction note | C-15 recorded the naive UT transform as "**4 032 sequentially dependent operations, each reducing to one scalar per (batch, head)**", and this was reported to the user as the port's cost. | **4 032 is a count of what the port *wrote*, not of what TinyGrad *charged*, and I presented it as the second.** Three different numbers, about three different things, and I used the wrong one: the port's source-level loop iterations are **4 032**; what the framework turned those into is **96 kernels / 12 097 ALU-lane ops** (`notes/26`, measured on the real port); and **what the framework's normal expression costs is 62 dependent steps**, with `BC=16` blocking taking 62 → 56 (`notes/29`, `--crossings` row 1). So the arithmetic bubble sort cost the port **~65x in dependent steps** (4032 → 62), not 4032, and the blocking the whole rewrite is built around buys **6 steps of 62, about 10%**. **The fix was never the blocking.** It was that `Tensor.dot` on a rank-1 contraction is literally `(x*w).sum(-1)` (`mixin/op.py:392`), so writing 4 032 rank-1 reductions where one wide contraction belongs is a **TinyGrad-idiom failure, not an algorithmic one** — `notes/26` rule 3, "output width, not operator name". I chose the number that made the finding sound bigger and it was the one number in the set that is not a cost | `notes/29-kda-faithful-port.md` `--crossings` row 1 and the clean/blocked SASS rows; `notes/26-tinygrad-idiom.md` rules 2 and 3; `kda/tiny_kda_faithful.py` | `[MEASURED]` — **C-15's conclusion stands, its cost figure is withdrawn** |
| **C-16** | 2026-09-29 | `kda/tiny_kda.py:24-32` (docstring, pre-edit) | Step 4 and step 7 justify their loop shape by unrolling: "unroll `C=64` iterations (**fine, trip count known**)"; "**a known trip count (T/C), so unrollable**"; and `:155-158` adds "the trip count is C, a compile-time constant, **so unrolling is legitimate**". | **TinyGrad does not unroll, and there is nothing to be legitimate about.** No `WHILE`/`ASSIGN` loop op exists; the `UNROLL` opt is **capped at 32** and applies only to reduce axes. So a `range(C)` loop is 64 real ops, and a trip count being constant buys nothing. The claim was used to justify the very structure that C-15 shows is wrong, which makes it the same failure in a new costume: a property attributed to the framework that it does not have, invoked to defend a choice the framework never made. The correction is not "pick a different unroll" — there is no unroll to pick | `notes/26-tinygrad-idiom.md` rule 5 (`[READ-FROM-SOURCE]`, op inventory + the `UNROLL` cap); `codegen/__init__.py:222` `expand_horizontal_reduce` for why a `range()` loop costs a reduce axis per iteration | `[READ-FROM-SOURCE]` |
| **C-17** | 2026-09-29 | `notes/19-tinygrad-vs-handwritten.md` (the proposed size-bisect for locating the fusion crossover) | A plan to locate the fusion threshold by bisecting on tensor size until the launch count stops changing — i.e. that there is a **constant** launch count at the small end to find. | **There is no constant to find.** Measured on the real port, the kernel count is **flat at 48 from 8 192 to 524 288 pixels** — a four-order-of-magnitude range with no step in it — while being **linear in `max_steps` (slope 1)**. So the method's premise is wrong in both directions at once: the thing it would bisect is not a step function, and the thing that *is* a step function (`max_steps + 8` versus a flat 9) is not a function of size. The crossover in `notes/19` is real — it was measured by a different method — but the proposed way to re-derive it would have returned "no threshold", and a method that cannot fail is not a check | `notes/26-tinygrad-idiom.md` rule 4 (census on the real `render.py`, not a structural copy — the agent's own first copy read 217 kernels against the real 48 and was discarded) | `[MEASURED]` |
| **C-15** | 2026-09-29 | `kda/tiny_kda.py:157-158`; `kda/SPEC.md:134-136` (pre-edit, and the second half of `:282-283`) | "Unrolled forward substitution is the faithful one **and is what the real kernel does**" — and `SPEC.md:136`, "its inverse is a row-wise forward substitution … `C−1` *dependent* steps, **not** a matmul". The port implemented `for i in range(C): for r in range(1, C): …` — **4 032 sequentially dependent operations, each reducing to one scalar per (batch, head)**. | **The port is faithful to `fla/ops/kda/naive.py`, not to the kernel.** The real kernel is a **two-level blocked inverse at `BC=16`**: 4 diagonal row-loops (`chunk_intra.py:285,290,296,302`, inner op `tl.sum(b_a[:,None]*b_Ai, 0)`, **zero `tl.dot`**) plus a block-Schur merge of **16 `tl.dot`s in a depth-3 DAG** (`:320-356`) — **56 issued iterations on a critical path of 14**, against the port's 4 032 sequential steps. The substitution is **row-wise**; the kernel never solves columns, which is why 4 032 was never required. Pinned to `fla-org/flash-linear-attention` @ `79d12e8ec0ae72bc7f2189b4244d6532ce8c268c`, inlined in `chunk_intra.py:44-384` and **not** in `utils/solve_tril.py`, which KDA never calls. `SPEC.md` **contradicted itself** between `:136` and `:283`; §4 was right. Also corrected there: 16 matmuls not 6, 63 displaced steps not 64. | `notes/25-fla-ut-reference.md` (full pin and line-level reading); `kda/SPEC.md` §3.3 and §4 as amended; `fla/ops/kda/chunk_intra.py:44-384` at the pinned commit | `[READ-FROM-SOURCE]` — **and see the pattern note below: this is the second wrong baseline, and the two are the same error** |
| **C-14** | 2026-09-29 | `notes/19-tinygrad-vs-handwritten.md:10-15` (the rule, pre-edit); and the numbers it licenses, §0.4, §4.1, §5.2, §5.4 — **26.2×, 47.2×, 36 600, 36 384, 20 932** | "**The TinyGrad side is not tuned and must not be.** It does not call `custom_kernel`, set a tile size or `TC_SELECT` or `ALLOW_TF32`, hand-unroll the DDA loop, hand-fuse stages, hand-stage data through global memory, or pass a `Context(...)`." — a rule that made the port's quality a *constant* of the experiment. | **The rule conflated "no hardware hints" with "naive", and the port implemented the second while believing it was obeying the first.** The 47.2× has two causes of opposite ownership and the note never separated them. (i) **The port's:** `minecu_tg/render.py:235-261` recomputes the flat voxel address from three float coordinates on all 32 steps — 9 ops/step for a 4-index gather — plus 3 `cast(i32)` + 3 `clip` to index it, 6 more. Amanatides–Woo, the algorithm the reference implements, **precomputes `tMax`/`tDelta` and then advances incrementally**; recomputing is not that algorithm, so this is a worse *implementation*, not merely an untuned one. (ii) **The compiler's:** the fusion crossover and the 20 932 B spill, which are decisions about fusion and register allocation and are independent of index arithmetic. **A second, separate defect in the same table: it does not sum.** §5.4 names 1.21× (early exit) and 4× (predicated byte loads) and reports 47.2× as explained by instruction count — **4.85× of 47.16×, leaving 9.73× with no row.** And it multiplies those two terms although both follow from the single fact that the tensor form cannot exit a ray early, so the product double-counts and the residue is bounded below by 9.73× and above by 38.91×. | `derivation/minecu_gap.py:decomposition` (47.16/4.85 = **9.73×**), `:independence` (9.73× ≤ residue ≤ 38.91×), `:op_census` (53 source ops/step, hand-counted, `render.py:235-261`), `:sass_per_op` (**21.5 SASS per source op**, against a 1–3 band), `:ablation_target` (the falsifiable target for the hoisting experiment) | `[MODELLED]` on the attribution, `[HAND-COUNT]` on the census, `[PREDICTED]` on what hoisting will recover — **the ablation has not been run, and the split between (i) and (ii) is not yet known** |
| **C-09** | 2026-09-28 | `probes/fp32_correctness.py:56-57`; and the magnitude propagated from it into ~58 locations across 19 files, listed in `notes/14-divide-magnitude-propagation.md` §6 | The `/` row of the float32-correctness probe was a measurement of division, and its figure — **12.4% – 15.4%** — was the error rate of TinyGrad's float32 `/`. | **The divisor was the dividend.** The line read `b = regimes[name] if op in "*/" else a`, and `regimes[name]` *is* `a`, so the row computed `a/a`. It never divided by an independent divisor. `a/a` is degenerate in a provable way — its exact quotient is `1.0`, representable, so the second rounding starts pinned at a **corner** of an asymmetric rounding cell and can only fail on the lower quarter of the reciprocal's error range. Measured against a divisor-independent draw, general division is wrong **1.77x–1.86x** more often on the same `a`, and the failure is **double rounding**, not approximation. | `probes/fp32_correctness.py` (fixed: independent divisor, both rows printed, divisor law declared); `derivation/divide.py:published_rate_table`, `:corner_predicate`, `:sampling_interval`, `:mechanism_sentence`; `derivation/divide_backends.py:self_division_log_crosscheck` re-derives the shipped T4 log's 260,261 / 322,233 / 318,044 **exactly**, which is what pins the old `a` array | `[MEASURED]` + `[MODELLED]` |
| **C-10** | 2026-09-28 | `notes/02-obs-camera-residuals.md` §4–5; `probes/divide_precision.py` docstring; `kernels/README.md:200`; `notes/02:329`; `bench/HYPOTHESES.md:412` | "The backend contracts `1/x` into an **approximate reciprocal**"; error class `approximated` for CL and METAL; "the signature of a reciprocal approximation" for `a/a` returning `1 - 2^-24`; "it is a **legitimate approximation**". | **All false. The reciprocal is correctly rounded; the loss is a second rounding.** Every failing backend is bit-identical to `a * fl32(1/b)` — a correctly rounded reciprocal and one correctly rounded float32 multiply. Two independent confirmations: (i) a hypothetical reciprocal 1 ULP *worse* than correctly rounded predicts **~86%** wrong against the **~27%** measured, so the reciprocal's accuracy is load-bearing; (ii) the only wrong `a/a` value is `0x3f7fffff`, the neighbour **below** 1.0, and `0x3f800001` above is **never** produced — a one-sided, exactly-predicted failure, which is the opposite of an approximation signature. The consequence for the record: the earlier claim that the rate is "insensitive to reciprocal accuracy" is **retracted** | `derivation/divide.py:reciprocal_sensitivity` (86.5% vs 27.1%, factor 3.2), `:corner_predicate` (exact predicate, 1 wrong bit pattern), `:ulp_bound` (1 ULP, with the direction-aware convention stated); `probes/divide_precision.py` step 2 re-measures bit-identity on this host | `[MEASURED]` + `[MODELLED]` |
| **C-11** | 2026-09-28 | `probes/divide_precision.py:132-135` (pre-edit) | "`Ops.RECIPROCAL` exists and PTX renders it as `rcp.approx.f32` — an explicitly approximate op. But `/` does not go through it; the cstyle renderer does the `a*(1/b)` rewrite unconditionally, and there is no flag for it." | **Backwards in the middle clause, and it moved the blame.** `/` **does** go through `Ops.RECIPROCAL` — `div()` builds `MUL(a, reciprocal(b))` and `reciprocal()` is `alu(Ops.RECIPROCAL)` — and the rewrite is not the cstyle renderer's, it is the Tensor API's. The "no flag" conclusion is right and the reasoning now says why: `use_fast_math` appears **zero** times in the package and the Metal driver already passes `-fno-fast-math` (`runtime/ops_metal.py:93`), neither of which could help because the rewrite is in the emitted source before the compiler runs. Note C-07 said the same thing correctly and was not applied here | `mixin/elementwise.py:251`, `:465`; `renderer/ptx.py:19`; `runtime/ops_metal.py:93`; package-wide grep, 0 hits for `use_fast_math`; `derivation/divide.py:mechanism_no_knob` | `[READ-FROM-SOURCE]` + `[MEASURED]` |
| **C-12** | 2026-09-28 | `bench/FAIR-BASELINE.md:412` (pre-edit) | "`Ops.FDIV` is dead code — the rewrite has already happened before the renderer sees it." | **False, and it is the same false claim as C-06**, which had already corrected it in this ledger and did not reach the file. `Ops.FDIV` is the load-bearing last line of the CPU repair: `ClangRenderer` deletes `RECIPROCAL` and adds `FDIV` (`renderer/cstyle.py:271,274`), and the decomposition at `codegen/decomp/op.py:133-136` — gated on `if Ops.FDIV in ops` — **manufactures** the node the Tensor API never emitted, which is why CPU measures 0/200,000 | `renderer/cstyle.py:271,274`; `codegen/decomp/op.py:133-136`; generated CPU source `(val0[0]/val1[0])` | `[READ-FROM-SOURCE]` + `[MEASURED]` |
| **C-13** | 2026-09-28 | `bench/HYPOTHESES.md:406-411` (pre-edit) | A `MODELLED` derivation concluding "expected failures = 0.02, 95% upper bound ≈ 3", on the reasoning that a `2^-24` relative perturbation divided by an absolute one-ULP threshold is a probability. | **The slip is a units error, and it is off by ~5e6.** A displacement of `s` ULPs of the result is a displacement of `s` **rounding cells**, so the crossing probability is `s` — order one, not order `2^-24`. Dividing a *relative* perturbation by an *absolute* threshold is what produces `2^-24`. The correct figure is the mean displacement of the exactly-rounded quotient measured in ULPs of the result, and it equals the rate the same draw produces to within 0.3% at all 18 independent-law points | `derivation/divide.py:cell_failure_fraction` (mean displacement 22.7%–27.1%, median 25.8%; slip factor 2.15e6) | `[MODELLED]` — the slip re-derived, not merely called wrong |
| **C-07** | 2026-09-27 | `notes/02-obs-camera-residuals.md` §4 (pre-edit) | "`Ops.RECIPROCAL` exists and `ptx.py` renders it as `rcp.approx.f32`, so tinygrad does have a first-class *explicitly approximate* reciprocal. `/` simply never goes through it." | **Backwards.** `/` *always* goes through `Ops.RECIPROCAL` — that is the defect. `rcp.approx.f32` (`renderer/ptx.py:19`) is the honest spelling of an approximation, and on CUDA it is not what fires, because CUDA prefers `CUDARenderer` over `PTXRenderer`. **Superseded in its conclusion by C-10**: `rcp.approx` is not what fires anywhere measured, on any backend. | `mixin/elementwise.py:251` builds `MUL(a, reciprocal(b))`; `mixin/elementwise.py:465` → `Ops.RECIPROCAL`; `runtime/ops_cuda.py:121` renderer order `[CUDARenderer, PTXRenderer, NVCCRenderer]` | `[READ-FROM-SOURCE]` + `[MEASURED]` (op-set enumeration) |
| **C-06** | 2026-09-27 | `notes/02-obs-camera-residuals.md` §4 (pre-edit) | "`renderer/cstyle.py:274`'s `Ops.FDIV` handling is dead code — the rewrite has already happened before the renderer sees it." | **The audit's own claim is wrong.** Line 274 is in `ClangRenderer` (class opens at `renderer/cstyle.py:256`), not the generic path, and it *is* reached — the `FDIV` decomposition at `codegen/decomp/op.py:133-136` manufactures the `FDIV` node the Tensor API failed to emit. It is the load-bearing last line of the CPU repair. | `renderer/cstyle.py:271` (deletes `RECIPROCAL`) / `:274` (adds `FDIV`); `codegen/decomp/op.py:136` rewrites `a*(1/b)` → `a/b`; generated CPU source is `(val0[0]/val1[0])` and CPU measures 0.00% wrong | `[READ-FROM-SOURCE]` + `[MEASURED]` |
| **C-05** | 2026-09-27 | `notes/02-obs-camera-residuals.md`; `notes/DECISIONS.md` row 13 | "`a/b` is wrong on **the cstyle path**"; root cause "tinygrad's `cstyle` renderer emits `a*(1/b)`". | The rewrite is at **`tinygrad/mixin/elementwise.py:251`** in `ElementwiseMixin.div` — the Tensor API layer, above the scheduler and above backend selection. `cstyle` is the wrong discriminator: `ClangRenderer` is also a cstyle renderer and it is the one backend that is *correct*, because it is the only one whose op set contains `Ops.FDIV`, which lets the `codegen/decomp/op.py:133-136` decomposition undo the rewrite. | `mixin/elementwise.py:251`, `:465`; `renderer/cstyle.py:271-274`; `renderer/ptx.py:19`; `codegen/decomp/op.py:133-136`; per-renderer op sets enumerated; generated source captured per backend; CPU 0/200,000, PYTHON/CL/METAL 54,270/200,000 | `[MEASURED]` + `[READ-FROM-SOURCE]` |
| **C-08** | 2026-09-27 | `notes/02-obs-camera-residuals.md` §4–5 (pre-edit); `probes/divide_precision.py` docstring | "PYTHON, CL and METAL" treated as one set — "a property of the codegen-path of the `cstyle` renderer (used by PYTHON/CL/METAL)". | **`PythonRenderer` is not a cstyle renderer** — MRO `PythonRenderer → Renderer → object`, no `CStyleLanguage`, and its `RECIPROCAL` comes from `uop/ops.py:1289` rather than `renderer/cstyle.py:137`. The *second* half of the original correction is **withdrawn**: it concluded that PYTHON's 27.13% "has a different cause" from CL and METAL's, labelling the latter `approximated`. All three are bit-identical to `a * fl32(1/b)` and all three are **double-rounded** — see C-10. The rendering differs; the mechanism does not | `PythonRenderer.__mro__`; `uop/ops.py:1289`; `renderer/cstyle.py:137`; bit-identity per backend in the corrected table | `[READ-FROM-SOURCE]` + `[MEASURED]` |
| **C-04** | 2026-09-27 | `notes/02-obs-camera-residuals.md` §"Correction: …" (pre-edit) | The division defect is a general property of TinyGrad's code generation, demonstrated on the Apple backends. **Must be scoped to a Metal/CL property until the prediction P-1 is run.** | **P-1 was run and it failed. This scoping is retracted, and the finding is broader than claimed.** `out/kaggle/20260928-014248-gpu/logs/divide_precision.log` on a T4: CUDA **27.13% (54,270) wrong, bit-identical to `a*fl32(1/b)`**; `a/a` 30,748 / 200,000. The prediction's step 4 (`--use_fast_math` absent ⇒ `div.rn.f32` ⇒ no approximation) reasons about a compiler flag reaching a rewrite that is **already in the emitted source**; it cannot. There is no Metal/CL property to scope to | `runtime/support/compiler_cuda.py:48`, `:53`; `renderer/cstyle.py:425`; the T4 log; `derivation/divide.py:mechanism_rewrite`, `:mechanism_no_knob` | `[MEASURED]` — **C-04's scoping retracted** |
| **C-03** | 2026-09-27 | 7 files, listed in §1a below | "256×" is *the* headline KDA finding, and in at least one place is reported as though it were a performance ratio. | It is an **element-count ratio** for one `(b, chunk, head)` slab. Three distinct quantities are being conflated; see §2. Only two of the three exist, and **neither is a measured performance ratio**. | re-derived from shapes in `kda/perf_kda.py` — see §3. `C²K/(BC·K) = 262,144/1,024 = 256`; byte model 290.7 MB vs 22.3 MB; `bench/HYPOTHESES.md:156-192` and `notes/04-codegen-comparison.md:171` already state the narrowing | `[MODELLED]` + `[READ-FROM-SOURCE]` |
| **C-02** | 2026-09-27 | `research/PARITY-SCOPE.md:269-272` vs `bench/HYPOTHESES.md:175`, `kda/kda_vs_fla.py:42` | "256×", as a single study-wide number. | **Two incompatible derivations of the same headline number now coexist in the record** and happen to agree numerically: `C²K/(BC·K) = 262,144/1,024 = 256` (KDA decay slab, `bench/HYPOTHESES.md:175`) and `L/C = 16384/64 = 256` (DeltaNet pairwise-vs-chunked, `research/PARITY-SCOPE.md:272`). They are different quantities. A reader cannot tell which one any given "256×" refers to. | both re-derived: `64²·64/(16·64) = 256`; `16384/64 = 256`. Coincidence at these parameters, not identity | `[MODELLED]` — the `L=16384` input is `[UNVERIFIED]` (cited to 2406.06484 §3.2, not checked in this pass) |
| **C-01** | 2026-09-27 | `research/PARITY-SCOPE.md:269-272` | "Our ~256× is … a factor of *work and traffic*." | The 256 is an element count on the port side. The *traffic* factor for this operator, at this shape, is **48.8×** (decay traffic ÷ compulsory) and the whole-operator traffic ratio is **13.0×**. "A factor of work and traffic" invites exactly the conflation C-03 corrects. | §2 and §3 below; re-derived | `[MODELLED]` |

### 1a. Where "256×" actually appears, and how each use is scoped

Seven non-vendored files make the KDA 256× claim. **Six are already correctly
scoped** as a graph-object or operand-size statement. One over-claims. This is
recorded because the useful output is not "256× is wrong" — it is *which* use
needs the fix.

| file:line | wording | verdict |
|---|---|---|
| `bench/HYPOTHESES.md:156` | "The 256× larger intermediate is worth **~13×, not 256×**" | **correct** — already does C-03's work |
| `bench/HYPOTHESES.md:175` | "the *slab* ratio `C²K/(BC·K) = 262,144/1,024 = 256`" | **correct** — the derivation, stated |
| `notes/04-codegen-comparison.md:131` | "256× as a graph object, **zero bytes in memory**" | **correct** — explicitly not a byte count |
| `notes/04-codegen-comparison.md:171` | "the 256× is **not** a byte count" | **correct** |
| `notes/04-where-compilers-stop.md:590` | "it is 256× larger than what a human [writes]" | **correct** — a size claim, and it says *larger*, not *slower* |
| `kda/kda_vs_fla.py:42` | "a 256x difference … a property of the two programs, not of the GPU" | **acceptable** — "difference" is vague but the caveat is right |
| `bench/selftest.py:324` | "the brief quotes ~256x; the operands behind that figure are not stated" | **correct** — flags the ambiguity itself |
| `research/PARITY-SCOPE.md:269-270` | "**a factor of *work and traffic***" | **OVER-CLAIM** → this is C-01. The traffic factor for this operator is 48.8× (decay ÷ compulsory), not 256× |

`notes/03-kda.md` is **not** on this list: it never says 256×. It says the decay
tensors are "**64×** larger than anything else in the operator", which is
`16,777,216 / 262,144 = 64` and is correct.

### Already in `DECISIONS.md` — pointers, not copies

These five are recorded in `DECISIONS.md` §"Claims I got wrong" and are **not**
reproduced here. Listed only so the register is complete and so the amendments
below are findable.

| # | subject | where | amendment since |
|---|---|---|---|
| D-1 | "the gather lowers to a one-hot reduction" | `DECISIONS.md` "Claims I got wrong" row 1 | none |
| D-2 | "the 12 residual pixels are FMA contraction" | row 2 | none |
| D-3 | "TinyGrad's **Metal** division is not correctly rounded" | row 3 | **AMENDED by C-05.** The row's *correction* column still names "tinygrad's `cstyle` renderer" as the cause. Superseded: the cause is `mixin/elementwise.py:251`, and `cstyle` is the wrong discriminator. **A reader consulting `DECISIONS.md` alone will get the superseded attribution.** |
| D-4 | "99% of KDA's wall time is Python building the graph" | row 4 | none |
| D-5 | KDA perf numbers measured on a NaN-producing graph | row 5 | none |

`DECISIONS.md` decision row 13 ("Treat the two precision findings as upstream
bug reports … both are backend defects") is **partly amended by C-05**: the
division defect is not a backend defect. It is created in the Tensor API and
merely *manifests* differently per backend. The decision to file it upstream
still stands; the stated reason does not.

---

## 2. The 256×: three quantities, sorted by what kind of thing they are

`C-07`…`C-01` above. The point of this table is that **the study's headline number
is not a performance number**, and had been read as one in at least four files.

| quantity | value | what it is | status |
|---|---|---|---|
| **element ratio** | **256×** | `C²K / (BC·K) = 262,144 / 1,024` for **one** `(b, chunk, head)` slab: the port's `[C,C,K]` decay tensor against the hand-written kernel's `[BC,K]` per sub-chunk. Equivalently `C²/BC = 4096/16`; `K` cancels | `[MODELLED]` — port side is a real, executed graph shape; kernel side (`BC=16`) is `[READ-FROM-SOURCE]` from `fla/ops/kda`, not executed. **Exact arithmetic, zero measurement.** It is a count, not a time and not a byte count |
| **byte / traffic ratio** | **48.8×** | decay-tensor traffic ÷ compulsory traffic: `268.4 MB / 5.51 MB`, where compulsory = I/O (`q,k,v,g` in, `o` out) + state `S` | `[MODELLED]` — fp32, one write + one read per intermediate, shapes from `kda/perf_kda.py` |
| **predicted time payoff** | **13.0×** | whole-operator traffic ratio: `290.7 MB / 22.3 MB`. The exp-folding rewrite is worth this much **if** the operator is bandwidth-bound at this shape | `[PREDICTED]` — **not a measurement.** Depends on the bandwidth-bound premise, which is itself unmeasured on this hardware |

**Only the element ratio and the byte model exist at present.** No DRAM traffic
has been measured, and no fold has been written or timed. The 13.0× is a
prediction whose premise (memory-bound at fp32) is asserted in
`bench/HYPOTHESES.md` H4, not established.

**The two byte-model figures the study quotes, and the confusion to avoid:**
`290.7 MB` is the *predicted total*, and `48.8×` is decay ÷ compulsory — **not**
`290.7 / 5.5`, which would be `52.8×`. The threshold below is a fraction of the
predicted **total**.

### Falsification condition for the byte model

**Recorded so that a future run can kill this, and so that a passing run means
something.**

- **Test:** `dram__bytes.sum` for one KDA call, from Nsight Compute, on an
  NVIDIA card, at the shape in `kda/perf_kda.py` (`B=2, T=512, H=HV=4, K=V=64,
  C=64`).
- **Predicted band `[MODELLED]`:** **[233, 349] MB** (290.7 MB ± 20%).
- **Refutation threshold:** measured total **< 58 MB** (i.e. 20% of the
  prediction). Below that, the decay tensor is not reaching DRAM.
- **What a refutation would mean:** the KDA bottleneck is **L2 residency**, not
  bandwidth, and the fix changes identity — from an exp-folding rewrite to a
  **tiling** change (keep the decay in SMEM/L2 by chunking the contraction).
  Different pass, different owner. This is the most consequential refutation
  open in the study, because it changes *which* work to do.
- **The measurement MUST be fp32, not bf16.** At bf16 one decay tensor is
  **33.55 MB**, comfortably inside L2 on these cards, so the tensor would be
  absorbed and the run would report a number that *looks* like the refutation
  while being an artefact of the dtype. At fp32 the tensor is **67.11 MB**,
  past L2, so the test is valid. A bf16 run cannot distinguish the two stories
  and must not be used to argue either.

---

## 3. Re-derivation record

Hard rule for this study: **re-derive every number in code; never transcribe.**
The audit that prompted this ledger found three arithmetic errors in the
author's own first draft (the fold ratio, the gram's arithmetic intensity, and a
camera time scale) that were caught *only* by re-deriving. Assume the same class
of error is present in anything not checked below.

All figures in §2 were rebuilt from the shapes in `kda/perf_kda.py` rather than
copied from `bench/HYPOTHESES.md`. **All thirteen reproduce:**

| figure | re-derived | as delivered | |
|---|---|---|---|
| element ratio | 256.0000 | 256 | ok |
| decay traffic | 268.4355 MB | 268.4 MB | ok |
| "everything else" | 22.2822 MB | 22.3 MB | ok |
| ported total | 290.7177 MB | 290.7 MB | ok |
| fold ratio | 13.0471× | 13.0× | ok |
| compulsory | 5.5050 MB | 5.5 MB | ok |
| decay ÷ compulsory | 48.7619× | 48.8× | ok |
| band low | 232.5742 MB | 233 MB | ok |
| band high | 348.8612 MB | 349 MB | ok |
| fp32, one decay tensor | 67.1089 MB | 67.1 MB | ok |
| bf16, one decay tensor | 33.5544 MB | 33.55 MB | ok |
| intensity, ported | 0.8079 FLOP/byte | 0.81 | ok |
| intensity, folded | 10.5412 FLOP/byte | 10.5 | ok |

**The delivered byte model is sound.** The three errors the audit found were in a
draft, not in what shipped.

The first re-derivation pass *did* produce four apparent mismatches
(`22.3 → 27.8 MB`, `13.0× → 10.66×`, `0.81 → 0.79`, band `[237, 355]`). All four
were a bug in the checking script — a subtotal double-counted the I/O and state
rows into the `other` group and then added them again. Recorded because it is the
exact failure mode the re-derivation rule exists to catch, and it would have been
reported as "the study's numbers are wrong" had it not been traced.

### `[UNVERIFIED]` — carried as unknown, not as true

These are inputs the byte model needs and that shapes alone do not settle. None
of them changes the 13× materially; all of them are marked so nobody downstream
mistakes a modelling choice for a fact.

1. **State `S` is charged one read + one write.** If it is actually carried per
   scan step it is `NT`× that term: `0.26 → 2.10 MB`, moving the fold ratio from
   **13.05× to 12.05×**. The `0.3 MB` in the delivered model is the
   one-touch reading.
2. **`folded total` reuses the ported `[C,C]` shapes.** At `BC=16` the
   sub-chunk gram is `16×16` × 4 sub-chunks = 1,024 elements against 4,096 for
   the full slab, so true folded traffic is *below* 22.3 MB. **The 13.0× is
   therefore a lower bound on the payoff** and the delivered figure errs
   conservatively. Worth stating out loud, since the temptation in a correction
   is to over-correct in the other direction.
3. **`beta` `[B,T,HV]` is not in `perf_kda.py`'s `items` list** and so is not
   charged. At 4,096 elements × 4 B it is 0.02 MB — immaterial, but it is an
   omission, not a decision.
4. **Whether `Z`/`Btri`/`M`/`Aqk` are each really written-then-read** (versus
   fused into one pass by the scheduler) is a *scheduling* property, not a shape
   property. `DEBUG=2` yields kernel count, not bytes. This is the
   least well-founded term in the model: it contributes 8.4 of the 22.3 MB.
5. **The `L = 16384` input to C-02** is cited to 2406.06484 §3.2 and was not
   checked in this pass.

---

## 4. Open predictions

Kept in one place so a prediction cannot be quietly promoted to a result by being
repeated in prose elsewhere. **Nothing in this section has been measured.**

| id | prediction | test | if it holds | if it fails |
|---|---|---|---|---|
| **P-1** (was C-04) — **REFUTED** | NVIDIA should **not** show the defect: `1.0f/b` lowers to `div.rn.f32` under NVRTC's defaults, so `a/b` should be far more accurate than the 27.13% measured on CL/METAL, and `a/a` should return exactly `1.0` far more often than 169,252 / 200,000. | `probes/divide_precision.py` on any NVIDIA backend. **No code change needed** — it already enumerates `Device._devices` | The "no fast-math" mechanism holds. C-04 stands: narrow the finding to a **Metal/CL property** | **This is what happened.** `out/kaggle/20260928-014248-gpu/logs/divide_precision.log`: CUDA **27.13% (54,270) wrong, bit-identical to `a*fl32(1/b)`**; `a/a` 30,748 / 200,000. A compiler flag cannot undo a rewrite already present in the emitted source. C-04 is **retracted**, not narrowed. |
| **P-1a** (refinement of P-1, **cut against P-1**) — **CONFIRMED** | On NVIDIA, the error may be **double rounding** rather than approximation, and therefore still ~27% — the same error class as the **PYTHON** backend, which computes a correctly rounded reciprocal and then multiplies in float32. | same probe; distinguish the two by whether the result is bit-identical to `a*fl32(1/b)` **and** consistent with `RN(RN(1/b)·a)` computed in float64 | The mechanism is **refined, not refuted**: NVIDIA joins PYTHON, METAL and CL, and the framing becomes "the Tensor-level rewrite costs a second rounding on every backend that does not repair it". The upstream report gets **stronger**, because the rewrite is lossy on *every* backend including the ones that looked fine | — |
| **P-1b** (if P-1 and P-1a both fail) — **not triggered** | Something else entirely introduces the approximation on CUDA. | same probe, plus reading the emitted PTX for `div.rn` vs `div.approx` vs `rcp.approx` | — | Not reached: CUDA is bit-identical to `a*fl32(1/b)`. Recorded for completeness, and the `rcp.approx` reading it contemplated is separately wrong — see **C-09** |

**Why P-1 was a genuine prediction, and why it failed.** The audit's reasoning
("no `--use_fast_math`, therefore `div.rn.f32`, therefore no approximation")
contained the error, and it is worth writing down because this repo has now made
it twice. The flag argument reasons about the **compiler's** treatment of a
`1.0f/b` that is already in the emitted source — but the source does not contain
`1.0f/b` as a division, it contains `a * (1.0f/b)`, and the `1.0f` is a
reciprocal the C standard does not *require* to be correctly rounded, and no
compiler flag in tinygrad's surface says otherwise (`use_fast_math` appears zero
times in the package). The error is not in the division; it is in the rewrite,
and it was already in the graph before NVRTC was invoked.

What the prediction *did* get right is the part that mattered. It was never "0%
wrong": `RN(RN(1/b)·a)` differs from `RN(a/b)` whenever the two roundings
disagree, and the **PYTHON** backend was already a measured instance of exactly
that. So the honest expectation for NVIDIA was *not* "0% wrong" — it was
"data-dependent, possibly the same rate as PYTHON" — and P-1a named that
alternative. **P-1a is the branch that happened.** C-08 is what made this
visible: grouping PYTHON with CL and METAL as "the approximating backends" hid
the control that calibrated the prediction, and the wrong label is what produced
the wrong prediction.
| **P-2** | The byte model in §2 predicts `[233, 349] MB` of DRAM traffic per KDA call, and specifically **not** below 58 MB. | `dram__bytes.sum`, Nsight Compute, **fp32** (see §2 for why bf16 manufactures a false refutation) | The operator is bandwidth-bound at this shape; the exp-folding rewrite is the right fix | The decay tensor is being absorbed by L2. The fix becomes **tiling**, not exp-folding — a different piece of work |
| **P-3** | The KDA port is **~31× memory bound** in fp32 at this shape (0.81 FLOP/byte against a 25.3 FLOP/byte T4 ridge), falling to ~2.4× if folded (10.5 FLOP/byte). | `sm__throughput.avg.pct_of_peak_sustained_elapsed` > 60% while `dram__throughput` < 40%, fp32 T4 | The roofline denominator is sound | The FLOP count is wrong by a large factor. **Worse than a scheduling gap:** it invalidates the denominator every other KDA claim uses. Must be settled before any KDA ratio is published |

### P-1 is runnable now, and was already missed once

A **Tesla T4 (sm_75)** run completed on 2026-09-27 23:19 UTC. Per
`out/kaggle/gpu-v1/SUMMARY.md`, every one of the 15 battery items reported
`skip`, and `divide_precision` — the exact probe P-1 needs — skipped with
`payload has no probes/divide_precision.py`. The summary records
`payload sha256: n/a`. The cause is in
`out/kaggle/20260927-231908-gpu/dataset-version.log`: the payload dataset version
was still being created (`Dataset version is being created`) when the kernel was
pushed, so the kernel ran against no payload. Same log: **30.00 h of GPU quota
remained**, refreshing 2026-10-03.

The hardware, the probe, and the quota were all present. The measurement was
lost to a packaging race, not to a missing capability. P-1 costs one re-run.

---

## 5. The class of error that produced C-09, and the structural fix

C-09 is not a typo. It is worth naming the *shape* of it, because the shape is
what made it survive an audit and then a second audit.

**A probe whose divisor was the dividend, propagated 58 times.** The bug lived
in one line of one probe. What it produced was a plausible, two-significant-
figure, correctly-formatted number — `12.4% - 15.4%` — that read as a
measurement. It was then copied into ~58 locations across 19 files, roughly 45
of them in deliverables a reader sees: a README table, a kernels table, a
decision row, a corrections table, every note. Nothing in that chain ever asked
what the probe *computed*; each copy was checked only for agreement with the
previous copy. Agreement is not correctness, and a chain of 58 agreeing copies
is exactly as wrong as its first link, with 58 witnesses instead of one.

Three things had to be true for this to be invisible, and each is a general
lesson:

1. **The wrong quantity looked like the right one.** `a/a` produces a finite,
   non-zero error rate at a plausible magnitude. There is no shape to the output
   that says "this is not a division".
2. **The bug was in the test's input, not its arithmetic.** The oracle was
   correct, the comparison was correct, the backend was measured correctly. A
   reviewer checking the *method* would have passed it; the defect was in what
   the method was pointed at.
3. **Nobody could check the number, because there was nothing to check it
   against.** No script printed it. It was a literal in a `print` statement.

**The structural fix: derive, do not transcribe.** A number in an output path
is now a literal in exactly one place — an executable claim in `derivation/` —
and every document cites the claim that prints it
(`derivation/divide.py:published_rate_table`, `:mechanism_sentence`,
`:sampling_interval`). Three consequences follow, and they are the point:

- A wrong number has **one** place to be wrong. Fixing the probe fixes every
  document that cites it, because no document contains it.
- A number nobody can reproduce **cannot be published**. The test is
  mechanical: if no script prints it, it is deleted. The rule this pass applied
  is *prefer deletion to a hedged number* — a figure that survives only as
  "roughly 2x" has lost information and kept the authority, which is the worst
  of both.
- **The correction is auditable by construction.** `derivation/divide_backends.py:self_division_log_crosscheck`
  re-derives the old probe's shipped counts (260,261 / 322,233 / 318,044)
  exactly, so C-09's *diagnosis* is itself derived rather than asserted.

**And a second rule, which the record needed twice:** a per-backend number is a
property of a **backend and a host**, never of the library. The same tinygrad
`CL` backend measures 27.13% on one Mac and 29.76% on a T4. Quoting either as
"what TinyGrad does" is the same class of error as C-09 with a different
mechanism — a single measurement, laundered into a general claim.

## 5b. The class of error that produced C-14 and C-15: **the baseline was the reference implementation, not the reference kernel**

This study's deliverable is a comparison — hand-written GPU code against what
TinyGrad generates from the same computation. That comparison has a
precondition, and it is easy to satisfy by accident and never on purpose: **the
TinyGrad side has to be as good as the human side.** Not better, not equal —
*as good*. Because the gap is the measurement, and the measurement is only
attributable to the compiler if the port is not the reason for it.

Both large comparisons got this wrong, in the same way, on different subjects:

| | the port actually implemented | the kernel actually does | cost |
|---|---|---|---|
| **C-14** mine.cu DDA | recomputes the flat voxel address from three float coordinates on all 32 steps, plus 3 `cast` + 3 `clip` per step | advances a precomputed address — which is what Amanatides–Woo *is* | 15 of 53 source ops per step wasted; 9.73x of the published 47.2x had no row at all |
| **C-15** KDA UT transform | `fla/ops/kda/naive.py`'s unrolled forward substitution | a two-level blocked inverse at `BC=16`, 56 iterations on a critical path of 14 | 4 032 sequentially dependent scalar reductions |

**The common shape: in both cases the thing ported was a file that exists to be
read, not a file that exists to run.** `naive.py` is fla shipping a reference
implementation *of its own kernel*; the plain high-level DDA is what a person
writes *before* they optimise. Both are legitimate artifacts. Neither is the
thing the GPU is supposed to be doing. And in both cases the port was
**correct** — `tiny_kda.py` passes 6/6 and was bit-faithful — which is exactly
why it survived: correctness is not the axis that fails, so nothing flags it.

**Three rules that would have caught both, and are cheap.**

1. **A passing correctness gate is not evidence that the baseline is right.** It
   is evidence about the math. `6/6 pass` was true of a port that was 72x off
   the kernel's structure. Never let a green test stand in for a review of what
   the *fast* implementation does.
2. **Before writing the port, read the fast implementation.** Not the spec's
   restatement of it either — `kda/SPEC.md` described `naive.py` at `:136` and
   the kernel at `:283` **in the same file, contradicting itself**, and the port
   followed `:136`. A spec written from a paper will describe the paper's
   algorithm, which is not always the kernel's. **Pin a commit and cite
   `file:line`, or you are paraphrasing.**
3. **Count the dependent steps before writing the loop, not after.** Both
   failures were visible as a number before any GPU ran: 4 032 dependent scalar
   reductions for a `[64,64]` triangular inverse, against 63 for the naive form
   and 14 for the real one. A dependent-step count is computable on paper from
   the loop structure. **It is the cheapest bug in this project to catch and we
   caught it zero times before measuring.**

**The general statement, which is the one worth keeping:** when a study about
compilers reports a large gap, the first hypothesis is not that the compiler is
bad. It is that the comparison is measuring the author. That hypothesis is free
to check and it has now been correct twice, on two unrelated subjects, by two
different routes.

**And a third instance, C-16, of a shape that deserves its own line: the
rationalisation.** `kda/tiny_kda.py` justified its loop shape with "the trip
count is C, a compile-time constant, **so unrolling is legitimate**". TinyGrad
has no unrolling, and the `UNROLL` opt is capped at 32 — so the justification
described a framework capability that does not exist, in order to defend a
choice the framework never made. Note the sequence: **first** a property is
attributed to the framework that it does not have, **then** the choice that
property supposedly endorses turns out to be the wrong one anyway (C-15). The
attribution was not sloppy reasoning about a real capability; it was a reason
manufactured after the fact.

**The check that catches it is cheap and does not require knowing the answer:
ask what the framework would have to do for the justification to be load-bearing,
then go read whether it does.** Nobody read. `notes/26` found it by inventorying
the op set, which is a half-hour exercise and would have caught it before the
port was written.

## 6. How to add a row

The point of this file is that it is cheap to write down, so that the correction
is cheaper to make than to repeat.

1. **Allocate the next id.** `C-nn`. Never reuse, never renumber.
2. **Quote the original claim verbatim**, from the file, with `file:line`. Not a
   paraphrase — a paraphrase is how "256×" became three numbers.
3. **Separate the correction from the mechanism.** "It is wrong, and here is
   where the code actually does it" is two claims. The second is the one that
   generalises, and the one that goes stale on a version bump.
4. **Evidence must be checkable**: a path with a line number, or a command. If
   the evidence is a number, the number must be re-derivable by a script. If the
   number was transcribed, say so and mark it `[UNVERIFIED]`.
5. **Tag status.** If you cannot decide between `[MODELLED]` and `[PREDICTED]`,
   it is `[PREDICTED]`. A prediction repeated often enough starts to sound like a
   measurement; that is the specific failure this file exists to prevent.
6. **Check whether the correction amends a `DECISIONS.md` row** and say so
   explicitly. `DECISIONS.md` is not append-only, so an amendment that lives only
   here is invisible to anyone reading the decision log.
7. **If the correction invalidates a number already quoted elsewhere**, grep for
   it. `256×` appears in **seven** non-vendored files (§1a) with **two**
   incompatible derivations; a fix applied to one of them is not a fix.

### Two rules that would have prevented most of this register

- **Never carry a number that a script did not produce.** Every figure in §2 was
  transcribed at least once before it was re-derived, and the re-derivation
  found a bug in the checker before it found a bug in the study. Assume the next
  transcription is wrong too.
- **A measurement on one backend is not a property of the compiler.** C-04
  exists because "27% wrong" was measured on Apple hardware and then described as
  a codegen property. The description outlived the measurement's scope because
  nothing recorded the scope. Record the scope with the number.

| **C-28** | 2026-09-30 | My own claim to the user, twice in one turn: first that only an A100 could make class (iv) observable and that the card could not be afforded; then that "Kaggle offers an L4" and so the recommendation was free | I had inferred an L4 route from Kaggle's P100-sunset announcement and my own `CARD_GATE` table, without testing whether the account can honour it | **Tested and wrong.** The accelerator is *specified*, not drawn — `kaggle kernels push --accelerator` exists and Kaggle CLI's own docs give `NvidiaL4` as the value, so no amount of restarting notebooks was ever going to help. Pushed with it: **accepted, exit 0, kernel COMPLETE, full log, and no GPU at all** — `nvidia-smi` missing, `libcuda.so.1` missing, `torch.cuda.get_arch_list() == []`, and Kaggle logged no error or warning. The identical script with `machine_shape: NvidiaTeslaT4` got a real T4 (cc 7.5, 65 536 B optin, triton arch 75), so the failure is specific to the L4 shape, not to the kernel. **Both of my recommendations were wrong in the direction that costs the most** — one sent the study to a paid A100 it did not need, the other promised a free L4 that does not exist here | `kernels/l4-gate/gate.py` and `kernels/l4-ctl/`; run output under `.agent-scratch/l4gate/`, `.agent-scratch/l4ctl2/`; Kaggle kernels `tenghuzhang/mc-l4-gate`, `tenghuzhang/mc-l4-ctl` | `[MEASURED]` — **the free L4 route is closed on this account; the KDA scope label is unchanged and rides on every KDA number** |

| **C-29** | 2026-09-30 | `kernels/l4-gate/gate.py`'s first version, which printed two `cuDeviceGetAttribute` readings under labels that were wrong | I labelled attribute 75 as `MaxSharedMemoryPerBlock` and 16 as `ComputeCapabilityMinor`; 16 is not a compute-capability attribute at all and returned **40**, a plausible-looking number meaning nothing | A mislabelled attribute is worse than a missing one because it survives review. Corrected to the enum's real mapping — 8 non-optin, 75 cc major, 76 cc minor, 97 optin — and **verified on the card**, where all three readers now agree exactly: non-optin 49 152 both ways, optin 65 536 both ways, cc 7.5 both ways. The corrected run also re-derived the T4 figures the study rests on, independently | `kernels/l4-ctl/` version 2; `.agent-scratch/l4ctl2/out/l4_ctl.json` | `[MEASURED]` — **corrected and verified; the earlier labels are withdrawn** |

| **C-30** | 2026-09-30 | My own consolidation message to the user: "帧说 `flat_gather` 该更快,实测它更慢。关系在这对上反了,而 r 仍然是 0.9986" — offered as a hit against the register-allocation mechanism | I had read a rank inversion between the two smallest frames as evidence against the claim, without asking whether the difference is resolvable | **Overstated, and I corrected it in the same turn.** The disagreeing pair is `flat_gather` (3 068 B, 3.020 ms) against `lean` (3 572 B, 2.950 ms) — **588 B on a ~3 KB frame and 0.070 ms on a ~3 ms time, both below the resolution of the measures**, so it is a noise floor and not a counterexample. A pairwise sign test restricted to resolvable pairs finds **7 of 7 agreeing**, which supports the claim *more* than "a correlation over five points" did. The Pearson **+0.9986 was independently re-derived from the five frame values in `notes/36:321` and reproduced exactly**; that check is now in `notes/35` §5, which had only quoted it | `notes/36-rung-timings.md` lines 280–281, 321, 397–398, 413–416; `notes/35` §5 | `[MEASURED]` — **Pearson reproduced exactly; the mechanism claim is better supported than stated, and the causal ambiguity is unchanged** |

| **C-31** | 2026-09-30 | `gpush/cli.py:PayloadProvenance.quotable`, which I wrote earlier today specifically to stop this project quoting stale remote results | `stale = bool(pushed_fp and ran and ran != pushed_fp)`, so an **absent** `pushed_fp` yields `stale == False`; `quotable` was `matched and not stale`, therefore **a run with no fingerprint to compare was quotable** | **A missing input was consumed as a passing check, in the one guard whose whole purpose is catching stale results.** Observed for real, not theorised: a lane that omitted `compute_fingerprint` got `quotable == True` from a comparison that never happened, and `notes/40` reports it. The irony is structural — the docstring three lines above the bug states the distinction correctly ("identity unestablished … is not the same as established-and-equal"), and `quotable` then discards it. Fixed by adding `established` (both fingerprints present) as its own condition and its own third verdict, `IDENTITY UNESTABLISHED`, so absence is reported rather than absorbed. Six tests, including a four-verdict distinctness test and a source-level assertion that `quotable` consults `established` | `gpush/cli.py`; `gpush/tests/test_stale_payload.py::TestIdentityMustBeEstablished`; defect reported in `notes/40` §6 | `[MEASURED]` — **reproduced live, fixed, 6 tests pin it; `stale` deliberately still False on absence so STALE PAYLOAD and IDENTITY UNESTABLISHED stay distinct verdicts** |

| **C-32** | 2026-09-30 | My own statement to the user that "Kaggle 的 L4 要付费、付费就能稳定开" was left as a hypothesis — correct — but my accompanying framing implied the platform lacked the card | `notes/35` §7 as I first wrote it said only that `NvidiaL4` "does not work on this account" | **Sharper, and it changes the diagnosis.** `notes/40` ran seven sessions varying only the shape string and then did a read-only behavioural census: a **public kernel reporting `NvidiaL4` ran 2026-09-29 08:06 UTC**, ~19 h before, and `NvidiaRtxPro6000` likewise. **Other accounts are granted these cards; this one is not.** So the blocker is an **account-level grant, not a platform capability and not a stockout** — and therefore whether payment is the mechanism is *still* unmeasured, because a shape in a metadata field is not a policy. Two further facts: pinning the CUDA image does not help (the attach is the blocker, not the container), and GPU quota rose 17.02 h → 17.20 h across seven sessions **including four that ran on a CPU box**, so a downgraded request appears to spend quota and return nothing | `notes/40-l4-probe.md` §1; `kda_t4/l4_shape_census.py` | `[MEASURED]` — **account-scoped denial, cause of the denial unmeasured** |

| **C-33** | 2026-09-30 | `notes/35` §8 and `notes/33`'s headline problem statement: that the hardcoded `SPLIT = 256` "is the entire mechanism behind the one cliff on the measured ladder", and `notes/35` §8's `scan` row reading **25.96×** | The two-stage cumsum re-sums a window of width `SPLIT` per output, the window is a hand-placed literal, and the framework's own comment says *"TODO: someday the optimizer will find this on its own"* | **Refuted by the lane sent to confirm it, in three ways, one of them a sign error.** (1) **The decomposition is a mitigation, not the injury**: at a fixed shape the one-stage path measures **329.17 ms** against two-stage **56.52 ms**, so it is **5.8× faster** than what it replaces; cross-checked at `SPLIT=4096` where the switch cannot fire, 329.59 vs 329.17 ms, agreement to 0.04%, so the arms are demonstrably the same program. (2) **The literal is causal and correctly located** — `2·SPLIT+1` predicts 129/257/513/1025/2049 and the measured switch points are exactly those, two independent readouts, five for five. (3) **The cost is two nested windows**, per row `N·SPLIT + (N/SPLIT)²` minimised at `SPLIT³=2N` ≈ 20 and diverging at `SPLIT=1`; I counted one. **Tuning the literal recovers 4.5× and 7.5× survives** (lower bound: the grid brackets 8 and 64 with nothing between). Cliff restated 25.96× → **34×** against the reference, 50× the copy floor. **Class (ii) survives and is now priced rather than inferred**, which strengthens the study's thesis while invalidating my attribution of *this* rung | `notes/37-split-causal.md`; the `mixin/op.py:761-766` reading it rests on | `[MEASURED]` — **"the entire mechanism" withdrawn; the cliff stands, larger, and 4.5× of it is attributable to the literal** |

| **C-34** | 2026-09-30 | `notes/33`'s `scan` rung, and therefore `notes/35` §8's row for it | The hand-written scan reference was taken to be correct across the ladder's N grid | **It has no correct kernel for 257 ≤ N ≤ 1023** — precisely the band where `_split_cumalu` switches decomposition. So the rung's reference is missing exactly where its mechanism changes, which is **why the grid could not see its own falsifier**: every N it did measure was on one side of the switch, so the two-stage form looked like a fixed cost rather than a window-width trade-off | `notes/37-split-causal.md` §"Not measured" | `[MEASURED]` — **a rung whose reference is absent in the mechanism's transition band cannot price the mechanism. Recorded so the next person does not treat `notes/33`'s N grid as uniform** |

| **C-35** | 2026-09-30 | Nothing quoted — recorded because it is the most dangerous member of the stale-result family and it is new | — | **`notes/37` had five wrong first measurements of a shape, and the two worst were not instrument drift but TinyGrad's program cache returning the *previous* configuration's program.** Twice from that cache, twice from probe deaths on missing keys, and once from `realize()` timing a warm-up — which is the **sixth** `realize()` no-op occurrence in this project after three that produced fabricated ratios. The cache case is the one worth carrying: **nothing about the returned object looks wrong**, it is a valid program that answers a question nobody asked, so no shape check or self-consistency test catches it. Only re-reading the emitted program does | `notes/37-split-causal.md` closing note | `[MEASURED]` — **the rule that catches it is: assert on the emitted program's structure, never on the fact that a program came back** |

| **C-36** | 2026-09-30 | `notes/35` §5's mechanism ("the channel is register allocation of the fused straight-line region"), §6's law ("the cost of the abstraction boundary is the register pressure the crossing induces"), and both places that name `13 216 → 3 068` as **ptxas's local-memory frame** | The decisive rewrite removed 18 of 22 graph operations and zero launches, and the spill figure orders all five wall clocks at Pearson +0.9986, so spill was read as the proximate cost | **Falsified by the lane built to break it — and this time not merely under-determined.** `NOLOCALS=1` on `flat_gather` at batch 1024: wall clock **3.0151 → 57.2876 ms (19.0×)** with launches 9 → 9, output **0 of 1 048 576 bytes differing from the reference**, op count 8 891 → 8 890, and **frame, spill, registers, SASS, LDL and STL byte-identical**. The quantity did not move and the clock moved an order of magnitude. **Reading B is not rescued either**: `dont_use_locals` deletes the local dimension (`gpudims.py:61-62`), so the real variable is **work per thread**, and the clock captures only **0.148** of the 128× a full per-thread scaling predicts. **Also mislabelled**: 13 216 → 3 068 is **spill stores + spill loads**, not the stack frame, which is **5 472 → 1 312** and ranks the rungs differently. The KDA corroboration of §6 is suspect for the same reason — co-varying spill and time is not evidence that spill causes time. **What survives: the launch budget is genuinely not the channel, and the mechanism connecting the two regimes is now unidentified** | `notes/38-local-memory-falsifier.md`; run evidence `out/lm38/20260930-233053/`; `notes/35` §5c | `[MEASURED]` — **§5's mechanism and §6's law both withdrawn; the honest residue is "not the launch budget, not spill, and the real variable is work per thread"** |

| **C-37** | 2026-09-30 | Nothing claimed — recorded because it is a concrete, actionable gap found as a by-product of a falsification | — | **`-maxrregcount` is inert on any kernel TinyGrad emits.** `__launch_bounds__(128)` becomes `.maxntid` in PTX and ptxas then ignores the flag: **nineteen settings produce one cubin**, while deleting the `__launch_bounds__` line gives **ten settings, ten cubins**. So the register-budget knob is reachable only by removing a line the framework emits unconditionally. That is a small, demonstrable gap rather than an inference, and it is the one lever that would have enabled the ideal version of C-36's experiment | `notes/38-local-memory-falsifier.md` | `[MEASURED]` — **a real, reproducible finding; the ideal experiment's blocker is now named** |
