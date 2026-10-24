# What makes KDA hard for a compiler

Companion to `SPEC.md` (the math) and `ref.py` (the validated oracle).
Reproduce every number: `.venv/bin/python kda/cost.py`.

**Method and honesty.** No KDA GPU kernel exists in this repo yet, and there is
no GPU, no torch and no triton in this environment. So **nothing here is
benchmarked.** The FLOP counts are analytic formulas evaluated from the shapes in
`ref.py`; the byte counts are analytic (one write + one read of each
intermediate, bf16) and are therefore *optimistic* — a real fused kernel moves
less, an unfused one moves more. Arithmetic intensities are the ratio of those
two, i.e. estimates. The claims about what the Triton kernels do are from
reading their source, not from running them. Where I say "a compiler will do X"
that is a prediction, flagged as such.

Default config throughout: `B=4, T=4096, H=HV=32, K=V=128, C=64, NT=64`.

---

## 0. The one-paragraph version

KDA is ~8 matmuls and 1 cumsum wrapped around **two things a static-shape graph
compiler structurally cannot express**: a sequential scan over `NT` chunks
carrying a `[K,V]` state, and a `C×C` unit-lower-triangular inverse computed by
forward substitution. Everything else is bookkeeping — but the bookkeeping
contains a numerical trap (§4) that a compiler will walk straight into, because
the *wrong* form is the one that looks like clean tensor-core matmuls. On top of
that the whole operator runs at **37.5 FLOP/byte against a ~295 FLOP/byte ridge
point**, i.e. ~8× memory bound, so the only thing that matters is how many
intermediates you avoid touching. A correct TinyGrad port that materialises its
tensors will be bandwidth-bound at roughly 1/8 of the machine's bf16 capability,
and no amount of matmul-replacement optimisation will fix that — it needs
different fusions, not better matmuls.

---

## 1. Ranked: hardest first

### 1. The inter-chunk scan — a loop, not a graph

```
for n in range(NT):                       # NT = 64 at T=4096, C=64
    v_new = u[n] - w[n] @ S               # S: [B,HV,K,V], loop-carried
    o[n]   = qg[n] @ S + Aqk[n] @ v_new
    S      = Diag(gC[n]) * S + KG[n].T @ v_new
```

`S` is the *only* thing carried, it is `K×V`, and it does **not** grow with `T`
— which is the entire appeal of the operator. But chunk `n`'s input is chunk
`n−1`'s output, so the loop cannot be reassociated, and there is no closed form.

This is the thing that decides the port:

* **Unrolling is the only option in a traced graph.** A `for` loop in the
  reference becomes `NT` copies of the subgraph. Correct, but: (a) graph size
  ×64, (b) every unrolled step makes the loop boundary a *materialisation
  point*, so `v_new` — `[B,HV,C,V]`, 6 MB per step, **384 MB** total — is
  round-tripped to DRAM 64 times even though it is consumed by the very next
  step, and (c) you throw away the one fusion that matters (see 6a/6e below,
  which together are 34.4 GFLOP at 0 extra traffic when fused and ~2.6 GB when
  not).
* **A real loop primitive does not exist in the tensor language.** TinyGrad has
  no `scan`/`while`; it has a Python-level `for`, which is unrolling.
* **The scan is also the parallelism story.** The fla kernel runs it as a
  *persistent* kernel with the `NT` steps sequential inside one program and
  `B·HV` independent across the grid
  (`chunk_gated_delta_rule_fwd_h`, `fla/ops/common/chunk_delta_h.py`). That is
  a scheduling decision, not an algebraic one.

There is a second, sneakier version of the same problem in the **backward**
pass: it is a reverse scan over the same `NT` steps, and fla *recomputes*
`w, u, qg, kg, v_new, h` in backward rather than saving them (they are freed
after forward, `chunk_fwd.py`, `disable_recompute=False`). So the backward
graph is not the transpose of the forward graph, and its memory profile is a
compiler decision, not a derivation.

### 2. The UT transform — a triangular inverse that is not a matmul

```
A ← −StrictTril(Diag(β) Z)                # 64x64 per (b, chunk, head)
for i in 1..C−1:                          # SEQUENTIAL
    A[i,:i] += A[i,:] @ A[:,:i]
M ← (A + I) · Diag(β)
```

* Row `i` depends on rows `< i`. It is Gaussian elimination specialised to a
  unit lower-triangular matrix. It is **not** expressible as a matmul, and there
  is no batched-`triangular-inverse` op that both uses tensor cores and is worth
  its own kernel.
* A naive TinyGrad port reaches for `Tensor.solve` / `inv` / `lu`. That is
  `O(C³/3) ≈ 87k` FLOP per `(b,chunk,head)` — 3.4× the 25.8k of the row loop —
  with no tensor cores, a serial dependency chain inside BLAS, and a pivoting
  decision (a `C×C` unit lower-triangular matrix has no pivots, but the op
  doesn't know that).
* **The fix is a hand-scheduled blocking that a compiler will not find.** Split
  `C=64` into `NC=4` sub-blocks of `BC=16`. Forward-substitute the 4 diagonal
  16×16 blocks row-wise (15 dependent steps each, but the 4 blocks are
  *independent* → real parallelism), then get the 6 off-diagonal inverse blocks
  from **3 batched 16×16 matmuls**. That is what
  `chunk_kda_fwd_kernel_inter_solve_fused` does. Cost: 162.8k vs 258.0k
  FLOP/chunk/head (**1.58× cheaper**) *and* parallel. Recovering that from a
  generic `triangular_inverse(M) -> [C,C]` primitive means inventing a blocked
  algorithm, not tuning a kernel.
* Per-step intensity is only **16 FLOP/byte** (2.11 GFLOP against 128 MB) — the
  worst of all the real steps. It is a small, latency-bound, serial island
  sitting in the middle of an otherwise tensor-core-friendly operator.

### 3. The `K/Γ` division trap — the most likely *silent* wrongness

The paper writes the `C×C` blocks as `(Γ ⊙ K)(K/Γ)ᵀ` (Eq. 6, Eq. 9). A compiler
that recognises this as "two clean matmuls with a scale folded into the operand"
— which is exactly what a matmul-replacement pass is built to do — will
materialise `K/Γ = kc / gam` and **divide by zero** the moment `Γ` underflows.
For a realistic gate `Γ` underflows readily: the KDA gate is
`−exp(A_log)·softplus(·)` with `exp(A_log)` initialised in `[1,16]`, so `|g|` runs
to several units per token and a single 64-wide chunk's cumulative decay
`exp(Σg)` reaches `exp(−100)` on long strong-decay stretches. With the
`safe_gate` variant it is bounded (`g ≥ −5` → `Γ ≥ e^{−5·64} = e^{−320}`), which
is exactly why that variant exists.

The safe form is `exp(gc[i] − gc[j])` with the causal mask applied to the log
difference **before** the `exp` — a three-operand contraction that produces an
`[C,C,K]` intermediate. So the correct expression is precisely the one a compiler
is built to *eliminate*. This is a numerics-vs-graph-shape tension, not a tuning
question.

I hit this in my own reference and it is worth recording exactly how it
behaved: the first version of `ref.py` computed `kd = kc / gam` and
`Z = einsum(gam*kc, kd)`. It passed every test with gates in `[−0.7, 0]` —
including the main chunked-vs-recurrence check (worst relerr 4.9e-15), chunk-size
invariance, causality, and every degenerate case. **14 of the 15 checks were
green, and the single failure was the extreme-gate test** (`alpha=0`,
`FloatingPointError: divide by zero`). It looked completely healthy. **The wrong
form is not wrong-looking.** Any port validated only on random small gates will
ship it.

`test_ref.py::test_division_form_overflows` asserts the literal form really does
produce `inf` there, and `test_gram_notations_agree` asserts the two forms are
bit-comparable where both are representable — so the safe form is provably the
same function, just not the same program.

A closely related trap in the same family: **`Aqk` must use un-scaled `q`.**
`Aqk[i,j] = Σ_d q[i,d]·exp(gc[i,d]−gc[j,d])·k[j,d]`. Written as `(Γ⊙Q)(K/Γ)ᵀ` the
`Γ` appears on `q`; written as a difference it must not. I made this exact
mistake too — `Γ` applied twice — and it produced a `3.9e-1` relative error that
looked like nothing more than a precision problem. Any fusion pass that decides
*where* to put the `Γ` scale needs both forms pinned down by a test.

### 4. The decay is a `cumsum` feeding a pairwise-broadcast structure

```
gc   = cumsum(g)                     # [B,T,HV,K], C-long axis   <- scan
D    = exp(gc[:,None] - gc[None,:])  # [B,NT,HV,C,C,K]  <- 8 GB at this shape
Z    = (K ⊙ D) @ Kᵀ                  # [C,K] x [K,C]
```

* The `cumsum` itself is a scan, but a *chunk-local* one (`C=64` long, resets
  every chunk) — so it parallelises over `B·HV·NT` trivially. This one is easy.
* The hard part is the next line. Every entry of the `C×C` block is a
  **pairwise difference of the cumsum**, so the dataflow is
  `scan → broadcast-pairwise-subtract → exp → contract`, and the natural
  intermediate is `[B,NT,HV,C,C,K]` = **8.2 GB** at the target shape (and
  16.4 GB counting a write + a read). No kernel builds this. The Triton kernel
  folds the `exp` into the matmul's *operand load* with a per-16-sub-chunk
  reference value `b_gn` (`b_kgt = tl.trans(b_k * exp2(b_gn − b_g0))`), so the
  `C²K` tensor never exists.
* So the fusion a compiler needs is: **fold an `exp` of a shifted difference into
  a matmul operand, where the shift is a per-sub-chunk scalar loaded from
  elsewhere in the same tile.** That is a load-time-computed operand transform
  with a data-dependent offset. This is not an elementwise fusion (which
  TinyGrad does have) and not a matmul fusion (which it also has) — it is
  neither, and it is what `BC=16` secondary chunking buys: the `exp` range drops
  from `C·|g|` to `BC·|g|`, which is what makes the fp16/bf16 tensor-core
  `exp2` path legal at all.

### 5. Grouped-value attention: two different head counts in one operator

```
q, k : [B, T, H,  K]        H heads
v    : [B, T, HV, V]        HV = G·H heads
g    : [B, T, HV, K]        HV heads  <-- the gate is per VALUE head
beta : [B, T, HV]           HV heads
```

The forget gate is indexed by the **value** head while the keys are indexed by
the **qk** head. The gate is *not* shared across a GVA group — only `q` and `k`
are. `fla/ops/kda/chunk.py` asserts `HV % H == 0` and `g.shape == (B,T,HV,K)`.

Consequences for a port:

* Either repeat `q,k` to `HV` (a `[B,T,HV,K]` materialisation, **128 MB** at
  this shape, doubling every `q`/`k` tensor) or thread two head indices through
  every single op. fla does the latter — `i_hv = i_bh % HV`,
  `i_h = i_hv // (HV // H)` appears in *every* KDA kernel.
* A compiler will insert the broadcast for you, and then the `[B,T,H,K]` inputs
  stop being `[B,T,H,K]`, so shared-memory tiles double in the q/k path while the
  `g` path is already at `HV`. Register pressure and occupancy suffer in a way
  that only shows up as "this kernel is slower than the FLOPs suggest".
* With `HV == H` (the default config, and Kimi Linear's actual setting) this
  vanishes. So the GVA path is easy to leave untested and easy to get wrong.

### 6. Memory traffic: the whole game

Per-step, from `cost.py` (bf16, one write + one read, `S` counted as living in
registers):

| step | GFLOP | MB | FLOP/byte | note |
|---|---|---|---|---|
| 0 gate activation | 0 | 768 | 0 | pure bandwidth: `exp`, `softplus`, 2 fma |
| 1 `cumsum(g)` | 0 | 512 | 0 | pure bandwidth |
| 2b `Z` | 8.59 | 128 | 64 | matmul |
| 3 UT transform | 2.11 | 128 | **16** | serial island |
| 4 `w` | 8.59 | 256 | 32 | matmul |
| 5 `u` | 8.59 | 256 | 32 | matmul |
| 6a `v_new = u − wS` | 17.18 | 6 | 2731 | only because `S` is in registers |
| 6b `Aqk` | 8.59 | 128 | 64 | matmul |
| 6c `o = (Γ⊙Q)S` | 17.18 | 256 | 64 | matmul |
| 6d `o += Aqk v_new` | 8.59 | 0 | ∞ | only if fused into 6c |
| 6e `S` update | 17.18 | 16 | 1024 | only because `S` is in registers |
| **total (kernel rows)** | **96.60** | **2454** | **37.5** | |

Ridge points: **295 FLOP/byte** on H100 SXM (989.5 TFLOP/s bf16 ÷ 3.35 TB/s),
**201** on A100 80GB. So at `C=64, K=128` this operator is **~8× memory bound on
H100**. Read that as: the theoretical floor is set by 2.45 GB of traffic, and
even perfect matmul codegen leaves you at ~38 FLOP/byte.

Two steps carry this, and both are free to fix *if and only if* they are fused:

* **6a + 6e are 34.4 GFLOP (36% of all the work) at ~0 traffic when fused.**
  They sit either side of the scan; `v_new` is produced in 6a and consumed
  immediately in 6c/6d/6e. Materialised, `v_new` is `[B,NT,HV,C,V]` = 128 MB,
  so 256 MB of write+read for work that is otherwise register-resident. That
  alone moves the operator from **37.5 to 34.0 FLOP/byte** (`cost.py`). The FLOPs
  are untouched; only the traffic is wasted, which is exactly the point.
* **Steps 0 and 1 are 1280 MB of the 2454 MB — 52% — and do zero FLOPs.** They
  are irreducible: the gate has to be computed and cumsummed. And the absolute
  floor — read `q,k,v,g`, write `o`, nothing else — is 640 MB, i.e.
  **143.9 FLOP/byte, still 2.1× below the H100 ridge point.** So KDA at
  `K=128, C=64` is memory bound *no matter how well you fuse it*. That is the
  most important number in this document: the ceiling is set by the bandwidth of
  the operator's inputs, not by any property of the graph.

The tensor that dominates is not one of the exotic ones: it is
`[B,T,HV,K]`/`[B,T,HV,V]` at **128 MB each**. There are four of them
(`g_cum`, `qg`/`kg`, `w`, `u`/`v_new`, plus `o`). Every step that could avoid
touching one of those is worth more than every step that could be made a faster
matmul.

### 7. Arithmetic intensity is bought with FLOPs, not saved

Sweep over `C` at fixed `T` (`cost.py`):

| C | NT | GFLOP | traffic MB | FLOP/byte | share of work in `C²` terms |
|---|---|---|---|---|---|
| 16 | 256 | 62.4 | 2161 | 27.5 | 17% |
| 32 | 128 | 73.5 | 2259 | 31.0 | 30% |
| **64** | 64 | **96.6** | **2454** | **37.5** | **47%** |
| 128 | 32 | 146.0 | 2844 | 49.0 | 65% |
| 256 | 16 | 257.6 | 3624 | 67.8 | 80% |

The `C²` terms (five intra-chunk matmuls + the forward substitution) cost
`O(T·C·K)` in total, so they scale **linearly** in `C`, while the `C·K·V`
inter-chunk terms are `C`-independent. So larger `C` raises intensity only by
adding work: `C=256` is 1.8× the intensity of `C=64` for 2.7× the FLOPs *and*
a 4× longer serial forward substitution. `C=64` sits at 47% `C²`-work, which is
a reasonable balance and is why the paper fixes it.

The practical consequence for a compiler: **there is no intensity win available
from restructuring the graph.** The wins are (a) fusing away the 128 MB-class
tensors and (b) making the 16-FLOP/byte serial island cheap. Both are fusion
and scheduling problems, neither is a matmul-selection problem.

### 8. Two regimes, two kernels, one static-shape graph cannot serve both

`fla/layers/kda.py`:

```python
if torch.is_grad_enabled():              mode = "chunk"            # training/prefill
elif q_len <= 64 and not self.training:  mode = "fused_recurrent"  # decode
else:                                    mode = self.mode          # "chunk"
```

* **Training/prefill**: the chunked form of §SPEC 3.3–3.5.
* **Decode**: the token recurrence (Eq. 1), one token at a time, state carried in
  a KV cache. Its arithmetic intensity is `2KV` FLOP per `K·V` state = **2
  FLOP/byte** — it is a pure memory-*latency* kernel, and no amount of
  arithmetic optimisation touches it.

These are different shapes (`T=4096` vs `T=1`), different graphs, different
bottlenecks. A single static-shape graph must be built twice. Worse, the layer
switches on `q_len <= 64 and not self.training`, i.e. on a **runtime control
flow value**, so the boundary is not a static property of the model.

Note also `state_v_first=True` in the layer: the cached state is stored
`[B,HV,V,K]` rather than `[B,HV,K,V]`. That flips which axis is contiguous in the
state-update matmul (`[C,K]ᵀ@[C,V]` vs `[C,V]ᵀ@[C,K]`), which changes the memory
access pattern of the most latency-sensitive kernel in the operator. A compiler
cannot infer this; it is a layout choice made for the decode path and inherited
by the prefill path.

### 9. Smaller things that will still bite

* **`scale` placement.** `scale = K^{-0.5}` multiplies `q` and nothing else, but
  the fused kernel distributes that one factor across two places (pre-scaled
  `Aqk`, plus `b_o *= scale` on the inter-chunk term only). It is a *runtime
  scalar*, so folding it into a matmul operand forces a materialised multiply.
  Pinned by `test_scale_applies_to_q_only`.
* **Two dtypes in one operator, at one specific tensor.** The `C×C` blocks are
  computed at higher precision than they are stored: `chunk_intra.py` allocates
  `Akkd` (the 16×16 diagonal blocks) as a dedicated **fp32** buffer and forces
  `tf32` on the triangular-inverse dot products
  (`SOLVE_TRIL_DOT_PRECISION`, with a comment "we used tf32 to maintain matrix
  inverse's precision whenever possible"), then merges into `Akk` (= `M`) in the
  *input* dtype, which then feeds bf16 matmuls. The decay `g_cum` and the
  `[C,C]` inverse genuinely need more than bf16 — an ill-conditioned triangular
  inverse loses the low bits that the whole delta rule depends on. So a compiler
  needs per-*region* dtype control, not per-tensor, and needs to know that the
  precision has to be raised for a sub-block and lowered again on the way out.
* **Recompute-vs-store is a policy knob, not a derivation.** fla frees
  `w, u, qg, kg, v_new, h` after forward and recomputes in backward
  (`disable_recompute`). A compiler that keeps everything is ~5× the activation
  memory for a ~30% backward speedup; one that keeps nothing recomputes too
  much. The right answer is shape-dependent, and TinyGrad's default
  (keep the graph, recompute nothing) picks the wrong end.
* **`allow_neg_eigval` widens β to `[0,2)`.** With L2-normalised `k`,
  `β ∈ [0,1]` makes `I − βkkᵀ` a genuine Householder reflection; `β > 1` gives it
  a negative eigenvalue. That is the entire point of the flag
  (Yang et al., *Unlocking State-Tracking in Linear RNNs Through Negative
  Eigenvalues*), and it means the `β` scaling cannot be folded into a
  fixed `Diag(β)` constant — it changes the conditioning of `M` and of the
  forward substitution.

---

## 10. What would actually have to change

Ranked by payoff, not by difficulty:

1. **A first-class sequential-scan / persistent-kernel primitive.** Even a
   modest "run this subgraph `NT` times with a loop-carried tensor" that keeps
   the carried state in registers and fuses across iterations would recover the
   36% of FLOPs in steps 6a/6e and the 384 MB of `v_new` traffic. This is the
   single biggest win and it is a language feature, not an optimisation.
2. **Fusion of an `exp`/shifted-difference into a matmul operand**, with a
   per-sub-chunk reference scalar loaded from a different tile. This is what
   removes the `C²K` intermediate and what makes the decay numerically legal in
   bf16. Related: a way to say "this matmul's `C×C` block is causal, don't
   compute the other half" so the compiler can skip it rather than compute
   garbage and mask.
3. **A blocked triangular-inverse rewrite.** 16×16 diagonal row-loops (4
   independent chains) + 3 batched matmuls, instead of `solve`. 1.58× fewer
   FLOPs *and* parallel; a `triangular_inverse` pattern that a backend can lower
   to that schedule.
4. **Per-tensor dtype policy.** fp32 for `g_cum`, `M` and the solve; bf16 for the
   bulk. Needs fp32↔bf16 casts placed at the right boundaries automatically.
5. **Activation recompute as a default for this class of operator.** The
   intermediate set is `[B,T,HV,K]`-scale and 2.4 GB of traffic; recompute in
   backward is the right trade at these shapes.
6. **A numerical guard on gate-derived divisions.** Not a compiler feature so
   much as a lint: "this expression divides by a cumsum of a non-positive
   tensor" is a pattern a compiler could flag, and it is the failure mode that
   passes every random test (§3).

Items 1–3 are what a hand-written kernel gets and a compiler does not, and they
are worth roughly 8× on this operator. Items 4–6 are hygiene.

---

## 11. What I could not resolve

* **No measurement.** No GPU, no torch, no triton (`import torch` fails in both
  `/opt/homebrew/bin/python3` and the repo venv). Every number here is
  analytic. The first thing to do with a working TinyGrad port is replace §6
  with real timings; the *ranking* should survive, the constants will not.
* **I never ran the Triton kernels**, so my reading of them (the negation, the
  `BC=16` blocking, the `RCP_LN2` trick, the `b_gn` shift) is source-reading
  only. The negation claim in particular I verified *indirectly but rigorously*:
  I reimplemented both variants and showed the un-negated one disagrees with Eq. 1
  by `~1e-1` relative while the negated one agrees to `<1e-11`
  (`test_forward_substitution_sign`, which also asserts the two differ so it
  cannot pass vacuously). But I have not run `fla`'s own test suite, so I cannot
  say whether `naive_chunk_kda` is exercised anywhere or simply stale.
* **The backward pass is not implemented in `ref.py` at all.** Everything above
  is forward-only. The backward is a reverse scan with the same structure and
  roughly double the matmuls, so the difficulty ranking should carry over, but
  that is a prediction.
* **The paper's `γ^{i→j}` notation (SPEC §3.4) is internally inconsistent and I
  chose a reading.** I am confident in the half-open/ratio reading — it is the
  paper's own `A_{i/j}`, it is what the derivation gives, and it is what the
  kernels compute — but it is a judgement call on a genuinely broken piece of
  notation, and if Moonshot ever corrects the paper this is the line to recheck.
* **Context parallelism.** `fla` has a CP path for KDA (`FLACPContext`) that
  all-gathers the per-chunk states across ranks. That adds a distributed scan on
  top of everything in §1. I have not looked at it and it is out of scope here.
* **Whether TinyGrad's existing fusion passes would in fact pick the bad
  `K/Γ` form** is a prediction from reading the paper's notation, not something I
  tested — I have not written or run any TinyGrad code. It is the hypothesis §3
  is built around, and it is cheap to falsify: write both forms and diff.
