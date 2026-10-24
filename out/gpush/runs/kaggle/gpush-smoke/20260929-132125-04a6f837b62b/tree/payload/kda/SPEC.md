# KDA (Kimi Delta Attention) — the exact math

Reference side of the TinyGrad port. Everything below is derived from the
primary sources listed in §1 and is implemented in `ref.py`, which is validated
against a token-at-a-time transcription of Eq. 1 by `test_ref.py`.

Notation for the whole document: `B` batch, `T` tokens, `H` qk-heads,
`HV` value-heads, `K` = `head_k_dim` (key/query head dim), `V` = `head_v_dim`,
`C` chunk size (64), `NT = T / C` chunks. `G = HV / H` is the GVA group size and
is 1 unless grouped-value attention is on. Index `t` is a token, `i` is a
position inside a chunk (1-based in the paper, 0-based in code — I flag which).

---

## 1. Sources

| what | where |
|---|---|
| Kimi Linear paper | **arXiv:2510.26692** — v1 30 Oct 2025, v2 1 Nov 2025. <https://arxiv.org/abs/2510.26692> · HTML: <https://arxiv.org/html/2510.26692v2> |
| The released KDA kernel (cited by the paper itself, footnote 1) | <https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda> |
| official repo | <https://github.com/MoonshotAI/Kimi-Linear> — **README only, zero `.py` files** (checked via the GitHub trees API). It defers the kernel to fla and ships HF checkpoints. |
| naive reference (recurrent + chunked) | `fla/ops/kda/naive.py` |
| chunked op entry point + arg contract | `fla/ops/kda/chunk.py` |
| fused forward pipeline | `fla/ops/kda/chunk_fwd.py` |
| intra-chunk + UT transform (Triton) | `fla/ops/kda/chunk_intra.py` |
| w/u recompute (Triton) | `fla/ops/kda/wy_fast.py` |
| gate activation (Triton + naive) | `fla/ops/kda/gate.py` |
| inter-chunk state scan (Triton) | `fla/ops/common/chunk_delta_h.py` |
| output kernel (shared with GLA) | `fla/ops/gla/chunk.py` |
| layer (short conv, norms, output gate, mode switching) | `fla/layers/kda.py` |
| `RCP_LN2 = 1/ln 2` | `fla/ops/utils/constant.py` |

Caveat on provenance: I read the above but **did not execute any of it** — there
is no torch, no triton and no GPU in this environment (checked: `import torch`
fails in both `/opt/homebrew/bin/python3` and the repo venv). Every statement
about kernel behaviour below is from reading source, not from running it. The
only things I ran are `ref.py` and `test_ref.py`.

There is no separate "official reference implementation" in the mathematical
sense. The paper's Appendix C PyTorch listing *is* the reference, and it is
byte-for-byte the same algorithm as `fla/ops/kda/naive.py` modulo GVA
expansion and one sign (see §7.1).

---

## 2. The operator in five lines

State `S ∈ R^{K×V}` per head. Per token:

```
S_t = (I - β_t k_t k_t^T) · Diag(α_t) · S_{t-1} + β_t k_t v_t^T
o_t = S_t^T q_t
```

1. The delta rule: `(I − β k kᵀ)` is a (generalised) Householder projection —
   read the state through `k_t`, compute the prediction error
   `v_t − k_tᵀ S_t`, and write `β_t k_t` times that error back.
2. The gate is **per key channel**, not per head: `α_t ∈ (0,1)^K`, applied as
   `Diag(α_t)`, and it multiplies `S_{t-1}` **before** the delta correction.
3. `β_t ∈ [0,1]` is the per-token learning rate of the delta rule (GDN uses
   `sigmoid`; `allow_neg_eigval` widens it to `2·sigmoid ∈ [0,2)`).
4. Output is a plain read-out `S_tᵀ q_t` — no softmax, no normalisation
   inside the operator.
5. `α_t = exp(g_t)` with `g_t ≤ 0` in log space, and the *only* nonlinearity on
   the gate is a per-channel `softplus` (§4).

The "K" in KDA = **K**imi-**D**elta-**A**ttention: Gated DeltaNet with the
scalar forget gate replaced by a per-channel one, plus a bespoke chunkwise
algorithm. Nothing else changes.

Plain linear attention (`S_t = S_{t-1} + k_t v_tᵀ`, no delta, no gate) is **not**
a special case of KDA: `β_t k_t v_tᵀ` is the only write term, so there is no
way to switch the delta rule off from the `(g, β)` interface. What *is*
reachable is `g ≡ 0` ⇒ `α ≡ 1` ⇒ the ungated delta rule (DeltaNet). Both facts
are asserted in `test_ref.py::test_gate_zero_is_deltanet_not_linear_attention`.

---

## 3. The delta rule and the WY / UT transform

### 3.1 Recurrence (paper Eq. 1, Appendix B)

Expanding one step is unambiguous and is what `ref.kda_recurrent` does
literally:

```
S ← Diag(α_t) S                       # decay FIRST
resid ← v_t - k_tᵀ S                  # then read out the prediction error
S ← S + β_t k_t residᵀ
o_t ← q_tᵀ S                          # read out AFTER the update
```

The order matters. `(I − βkkᵀ)Diag(α)S` is **not** `Diag(α)(I − βkkᵀ)S`, and
transcribing it the other way round is the single most common error.

### 3.2 Why a WY representation is needed

Expanding the recurrence over a whole chunk gives `S^r = P^r S^0 + H^r` where

```
P^r = ∏_{i=1..r} (I − β^i k^i k^{i⊤}) Diag(α^i)          (transition)
H^r = Σ_{i=1..r} (∏_{j=i+1..r} (I − β^j k^j k^{j⊤}) Diag(α^j)) β^i k^i v^{i⊤}
```

`P` is a product of `C` dense `K×K` Householders — `O(C K²)` to form naively,
`O(C² K²)` to expand, and completely sequential. The WY representation
(paper Appendix B, Props 1 and 2) rewrites both as *diagonal minus low-rank*:

```
P^r = Diag(γ^r) − Σ_{i≤r} Diag(decay) k^i (w^i)ᵀ
H^r =        Σ_{i≤r} Diag(decay) k^i (u^i)ᵀ
```

with scalar auxiliary recursions `w^r`, `u^r` (§3.4). That collapses `C`
sequential `K×K` products into a `C×C` triangular solve. `KDA` is, in the
paper's own framing (§6.2), the DPLR form with `a = b = k`.

### 3.3 The UT transform (paper Eq. 6–7)

Let `Γ^r = γ^r` be the per-channel cumulative decay (log form below), `⊙` be
broadcast row-wise scaling, and define the **decayed gram**

```
Z = (Γ ⊙ K)(K / Γ)ᵀ        i.e.  Z[i,j] = (γ^i/γ^j) ⟨k^i, k^j⟩
```

Then the paper's Eq. 6–7 are

```
M = ( I + StrictTril( Diag(β) · Z ) )^{-1} · Diag(β)        (Eq. 6)
W = M (Γ ⊙ K)          U = M V                            (Eq. 7)
```

`M` is the UT transform. `(I + StrictTril(·))` is unit lower triangular, so its
inverse is a forward substitution (Gaussian elimination specialised to a unit
lower-triangular matrix). **The block below is the NAIVE form — the whole point
of this subsection used to be that it is *the* form, and that was wrong.** Read
§4 before implementing:

```
A ← −StrictTril( Diag(β) Z )              # NEGATE first — see §7.1
for i in 1..C−1:                            # sequential, C−1 dependent steps
    A[i, :i] += A[i, :] @ A[:, :i]
M ← (A + I) · Diag(β)
```

> **Correction (ledger C-15).** The sentence this replaced read "its inverse is
> a row-wise forward substitution … `C−1` *dependent* steps, **not** a matmul",
> and `kda/tiny_kda.py:157-158` was written from it, claiming the unrolled form
> "is what the real kernel does". **It is not.** The real kernel does a
> **two-level blocked inverse at `BC=16`** — §4 describes it and pins the
> source. The naive loop above is what `fla/ops/kda/naive.py` does, and this
> repo ported *that file*. `C−1` is a correct count for the naive form and a
> wrong description of the kernel. **This subsection and §4 contradicted each
> other until now; §4 is the one that describes the kernel.**

`W` is the "pseudo-key" block, `U` the "pseudo-value" block. `U − W S` is the
paper's "pseudo-value" term of Eq. 9: the values the delta rule *would* have
written, had the incoming state been visible.

### 3.4 The scalar recursions, and where the paper's notation breaks

Props 1 and 2 state

```
w^r = β^r ( Diag(γ^r) k^r − Σ_{i<r} w^i (k^{i⊤} Diag(γ^{i→r}) k^r) )
u^r = β^r ( v^r − Σ_{i<r} u^i (k^{i⊤} Diag(γ^{i→r}) k^r) )
```

**The printed definition of `γ^{i→r}` is inconsistent with these two equations.**
The paper defines `γ^{i→j} := ∏_{k=i}^{j} α^k` (§2.1) — an *inclusive* product.
But the derivation requires the *half-open* product `∏_{k=i+1}^{j} α^k`, i.e.
`γ^j / γ^i`. The two differ by a factor `α^i`.

I checked this three ways:

* **The paper's own `A_{i/j}`.** §2.1 defines `A_{i/j} := γ^i/γ^j`, and Eq. 6/9
  use exactly `A_{i/j}`. That is the half-open form. So the paper defines
  `A` correctly and then uses a different, wrong symbol for the same quantity.
* **Direct algebra.** Expanding `P^2` by hand gives the `i=1` term
  `Diag(α²) k¹ (w¹)ᵀ = Diag(γ²/γ¹) k¹ (w¹)ᵀ` — the half-open form. The
  printed proof writes `Diag(γ^{i→r})` and separately `Diag(γ^{i→r−1})` in the
  same chain of equalities, which cannot both be right.
* **The kernels.** Every kernel computes the ratio `g_cum[i] − g_cum[j]`, i.e.
  the half-open form. `fla/ops/kda/chunk_intra.py`:
  `b_kgt = tl.trans(b_k * exp2(b_gn - b_g0))` and
  `b_Akk = tl.dot(b_k * exp2(b_g - b_gn), b_kgt)`, whose product is
  `exp2(g_i − g_j)`.

**Reading I implemented: `γ^i / γ^j` (= `∏_{k=i+1}^{j} α^k`) everywhere.** It is
the paper's own `A_{i/j}`, it is what the derivation gives, and it is what the
kernels do. The paper's `γ^{i→r}` symbol should be read as half-open.

### 3.5 State update and output (paper Eq. 8–9)

With `S^0` the state at the start of the chunk:

```
S^C      = Diag(γ^C) S^0 + (Γ^{i→C} ⊙ K)ᵀ (U − W S^0)              (Eq. 8)
O_chunk  = (Γ^{1→C} ⊙ Q) S^0 + Tril( (Γ^{1→C} ⊙ Q)(K/Γ^{1→C})ᵀ ) (U − W S^0)   (Eq. 9)
```

In ratio/decay form, with the half-open reading:

```
S^C      = Diag(γ^C) S^0 + Σ_i Diag(γ^C/γ^i) k^i (u^i − (w^i)ᵀ S^0)   # note: ONE decay for both terms
O[c]     = Σ_{i≤c} (γ^c/γ^i) ⟨q^c, k^i⟩ (u^i − (w^i)ᵀ S^0) + (γ^c ⊙ q^c)ᵀ S^0
```

The fact that the *same* decay `γ^C/γ^i` multiplies both the `u` term and the
`w` term is what makes Eq. 8's factorisation valid. That only holds because
Props 1 and 2 use the same half-open decay — which is another reason the
half-open reading is the right one. `Tril` here includes the diagonal (the
token attends to itself); `StrictTril` excludes it.

The two terms of Eq. 9 are the **inter-chunk** (reads `S^0`, the state carried
across the chunk boundary) and **intra-chunk** (the `C×C` causal score) parts.

---

## 4. The gating — the interesting part

This is the only real difference from Gated DeltaNet. GDN has one scalar
`α_t ∈ [0,1]` per head; KDA has `K` independent ones.

### 4.1 Gate activation

Two variants, both from `fla/ops/kda/gate.py`. Shapes: raw gate `g_raw`
`[..., HV, K]`, `A_log` `[HV]`, `dt_bias` `[HV·K]`.

```
default:     g = − exp(A_log) · softplus(g_raw + dt_bias)          g ∈ (−∞, 0)
safe_gate:   g = lower_bound · sigmoid( exp(A_log)·(g_raw + dt_bias) )
                                                        g ∈ [lower_bound, 0)
```

`lower_bound ∈ [−5, 0)` is enforced by `chunk_kda` (`−5 ≤ lower_bound < 0`),
recommended `−5`, i.e. minimum per-step decay `exp(−5) ≈ 6.7e−3`.

Note these are **not** a smooth reparametrisation of each other: in the default
branch `exp(A_log)` scales `softplus`, in the `safe_gate` branch it scales the
*sigmoid argument*.

Initialisation (`fla/layers/kda.py`): `A_log = log U(1,16)` so
`exp(A_log) ∈ [1,16]`; `dt_bias = inv_dt` with `dt ~ U(0.001, 0.1)` and
`inv_dt = dt + log(1 − e^{−dt})`, chosen so that `softplus(dt_bias) = dt`
exactly. So at init `g = −exp(A_log)·dt ∈ [−1.6, −0.001]`.

### 4.2 Difference from zero, and the log domain

`α_t = exp(g_t) ∈ (0,1)`. The *only* way a cumulative decay is ever evaluated is
as a **difference of two log cumsums**:

```
gc[r] = Σ_{i≤r} g_i                       (log γ^r)      # WITHIN the chunk only
decay(i → j) = exp( gc[i] − gc[j] )        for i ≥ j,  ∈ (0, 1]
```

Two consequences that a port must respect:

* **No exponent is ever positive on a kept entry**, because `g ≤ 0` makes `gc`
  non-increasing and the causal half always has `gc[i] ≤ gc[j]`. That is why
  the log formulation is used at all. (Paper §3.2: "such fine-grained decay
  introduces numerical precision issues during division operations".)
* **`K / Γ` must never be materialised.** The paper's `(Γ ⊙ K)(K/Γ)ᵀ` is
  *notation*; evaluating the division gives `inf` as soon as `Γ` underflows.
  `ref.py` asserts this: with `g = −1e3` the literal `kc/gam` is `inf`, while
  the difference form stays in `(0,1]` and still matches the recurrence. See
  `test_ref.py::test_division_form_overflows`.
* The discarded (upper-triangular) half *does* have positive exponents, so the
  causal mask must be applied to the log difference **before** the `exp`.
  `ref.py::_causal_decay` does this via `np.where(keep, d, -inf)`.

`safe_gate` exists because with a bounded `g` the decay over a secondary chunk
of width `BC=16` satisfies `|Δ| ≤ 16·5 = 80`, and `exp2(±80)` is representable
in fp16-range-limited tensor-core math — which unlocks the `M=16` fast path.

### 4.3 Chunk-level cumsum, and the log2 trick

`gc` is a **chunk-local** cumsum over `C` elements, not over `T`. It resets at
every chunk boundary; `ref.py` asserts that (`g_cum[0,1,0,0,0] == g[0,C,0,0]`).

The kernels store `gc · log2(e)` and use `exp2` instead of `exp`, so that the
exponential is a hardware `ex2.approx` on the SFU path. `fla/ops/kda/gate.py`
and `chunk_fwd.py` pass `scale=RCP_LN2`; `RCP_LN2 = 1.4426950216` in
`fla/ops/utils/constant.py`. Purely an implementation detail — `exp2(x·log2 e) =
exp(x)`. The reference uses `exp` throughout and takes `g` in natural-log units.

### 4.4 Secondary chunking

The Triton kernel splits each `C=64` chunk into `NC = 64/16 = 4` sub-chunks of
`BC=16` and subtracts a per-sub-chunk reference `g_n` before exponentiating
(`b_gn`). Within a sub-chunk the exponent is bounded by `BC·|g|` instead of
`C·|g|`. The same `BC=16` blocking is reused to do the UT transform.

**The UT transform is a two-level blocked inverse, not a substitution loop.**
Source pinned in `notes/25-fla-ut-reference.md`: `fla-org/flash-linear-attention`
@ `79d12e8ec0ae72bc7f2189b4244d6532ce8c268c` (2026-09-28), inlined in
`fla/ops/kda/chunk_intra.py:44-384` — **not** in `utils/solve_tril.py`, which
KDA never calls.

- **Diagonal stage** (`:285, 290, 296, 302`): four `for i in range(2, min(BC, T−i_tc0))`
  loops, one per sub-chunk, over a `[16,16]` block. The inner op is
  `tl.sum(b_a[:, None] * b_Ai, 0)` — a broadcast-reduce with **zero `tl.dot`**
  (verified: none in lines 274-314). On the **default** path (`safe_gate=False`)
  the diagonal blocks are built token-parallel over `B·T` programs, not through
  a matmul at all; the `tl.dot` form is the non-default path.
- **Merge stage** (`:320-356`): the off-diagonal blocks combined by block-Schur,
  all `[16,16]×[16,16]`, **16 `tl.dot`s in a depth-3 DAG**.

**Cost: 56 issued substitution iterations (4 × 14) + 16 `tl.dot`s, on a
data-dependency critical path of 14, not 56** — the four loops touch disjoint
registers, so worst-case serial latency is ~17 round-trips. Trip counts are
runtime-bounded (`T` is `do_not_specialize`), not unrolled; a 20-row tail chunk
issues 14, 2, 0, 0, and `BT=32` gives 28 iterations and 2 dots.

The substitution is **row-wise**, one row per iteration, reducing over the row
axis. The kernel does not solve columns at all.

Three things this repo had wrong, now corrected: the count is 16 matmuls where
§4 previously said 6, and the displaced count is 63 where it said 64; §3.3's
"C−1 dependent steps, not a matmul" described `naive.py` (ledger C-15); and
`kda/tiny_kda.py:157-158`'s claim that the unrolled form "is what the real
kernel does" is false.

---

## 5. Chunk size, and why the split is where it is

**C = 64.** `fla/ops/kda/chunk.py` raises unless `chunk_size ∈ {32, 64}`;
the paper fixes `C = 64` in §6.3. (GDN's op also allows 16; KDA does not.)

The split point is forced by the algebra, not chosen for hardware:

* **Inside a chunk** the `C` Householder products collapse into a `C×C`
  triangular solve plus `C×C` matmuls. All `C²` pairs are independent, so the
  whole chunk is one dense parallel block — this is the part that goes on
  tensor cores.
* **Across chunks** the only thing carried is the `K×V` state. Chunk `n`'s
  input is chunk `n−1`'s output. That is a **scan**: `NT` strictly sequential
  steps, no way to reassociate. The state is `K×V` and does *not* grow with `T`.

So: intra-chunk = dense/parallel, inter-chunk = sequential scan. The split is
where the algebra stops being a product and becomes a recursion. `C` trades
these off: larger `C` means more `C²` work per token (better arithmetic
intensity) but a longer forward substitution and a larger `exp` range.

---

## 6. Shapes

`[B, T, H, K]` style throughout. `HV` and `H` differ only under GVA.

### 6.1 Operator inputs / outputs

| tensor | shape | notes |
|---|---|---|
| `q` | `[B, T, H, K]` | L2-normalised, then `× scale` |
| `k` | `[B, T, H, K]` | L2-normalised, **not** scaled |
| `v` | `[B, T, HV, V]` | |
| `g` | `[B, T, HV, K]` | **log space**, per value head *and* per key channel |
| `beta` | `[B, T, HV]` | scalar per token per value head |
| `initial_state` / `S` | `[B, HV, K, V]` | (or `[B, HV, V, K]` with `state_v_first=True`, which is what `KimiDeltaAttention` uses) |
| `o` | `[B, T, HV, V]` | |
| `scale` | scalar | default `K ** -0.5`, **multiplies `q` only** |

### 6.2 Per-chunk intermediates (everything below is `[B, NT, HV, …]`, `NT = T/C`)

| step | tensor | shape | class |
|---|---|---|---|
| 1 | `g_cum` | `[B, NT, HV, C, K]` | scan (elementwise + cumsum) |
| 1 | `kg = Γ ⊙ K` | `[B, NT, HV, C, K]` | elementwise |
| 2 | `Z` | `[B, NT, HV, C, C]` | matmul `[C,K]×[K,C]` |
| 3 | `Btri = StrictTril(Diag(β)Z)` | `[B, NT, HV, C, C]` | elementwise + negate |
| 4 | `M = (I+Btri)⁻¹ Diag(β)` | `[B, NT, HV, C, C]` | **sequential**, `C−1` steps |
| 5 | `w = M (Γ ⊙ K)` | `[B, NT, HV, C, K]` | matmul `[C,C]×[C,K]` |
| 6 | `u = M V` | `[B, NT, HV, C, V]` | matmul `[C,C]×[C,V]` |
| 7a | `v_new = u − w S` | `[B, HV, C, V]` | matmul `[C,K]×[K,V]` + `S` |
| 7b | `Aqk` | `[B, NT, HV, C, C]` | matmul `[C,K]×[K,C]`, causal incl. diagonal |
| 7c | `qg = Γ ⊙ Q` | `[B, NT, HV, C, K]` | elementwise |
| 7d | `o` | `[B, NT, HV, C, V]` | matmul `[C,K]×[K,V]` + `[C,C]×[C,V]` |
| 7e | `S` | `[B, HV, K, V]` | matmul `[K,C]×[C,V]` + per-channel decay |

`Aqk[c,j] = Σ_d q[c,d]·exp(gc[c,d] − gc[j,d])·k[j,d]` for `j ≤ c`, else 0.
Note it uses **plain `q`** — the decay is already in the `exp`. Using `Γ ⊙ q`
double-counts `Γ`; I made exactly that mistake and `test_ref.py` caught it.

### 6.3 Layer wrapper (`fla/layers/kda.py`), outside the operator

```
q, k = L2Norm(Swish(ShortConv_{k=4}(W_{q,k} x)))       # short conv, no bias
v    =          Swish(ShortConv_{k=4}(W_v     x))
g_raw = W^up_a (W^down_a x)                            # low-rank, rank = head dim
beta   = Sigmoid(W_beta x)                             # or 2*sigmoid if allow_neg_eigval
o    = W_o ( Sigmoid(W^up_g W^down_g x) ⊙ RMSNorm(KDA(q,k,v,g,beta)) )
```

`ShortConvolution` is depthwise, kernel 4, `silu` activation. Ablations (§5.2 of
the paper) show all three of short-conv, output gate and the gate
parametrisation matter; `Sigmoid` beats `swish` for the output gate.

---

## 7. Normalisation, scaling, and the two formulations

### 7.1 Normalisation and scaling — where each factor goes

* **L2Norm on `q` and `k`** — always on (`use_qk_l2norm_in_kernel=True` hard-coded
  in `KimiDeltaAttention.forward`). Motivated by eigenvalue stability, per
  Yang et al. `‖k_t‖ = 1` makes `I − βkkᵀ` a genuine Householder reflection for
  `β ∈ [0,1]`, and is what keeps the UT transform well conditioned. My
  `l2norm` uses `eps=1e-6` under the sum of squares, so `|q| = 1 − O(eps/‖q‖²)`.
* **`scale = K^{-0.5}`** — applied to `q` **only**, per `naive_recurrent_kda`
  (`q.repeat_interleave(G, dim=2) * scale`) and the paper's Appendix C
  (`q = q * K**-0.5`). The fused kernel distributes that one factor differently:
  it pre-scales the `Aqk` block (`b_Aqk = tl.dot(...) * scale`) and scales the
  inter-chunk term in the output kernel (`fla/ops/gla/chunk.py:427`,
  `b_o *= scale` — applied to the `(Γ⊙Q)S` term only, *before* `A@v` is added).
  Same single factor, two places. `test_ref.py::test_scale_applies_to_q_only`
  pins that `scale` touches `q` and nothing else, and that `S` is invariant to it.
* **`v`** is not normalised.
* **RMSNorm + output gate** are applied *after* the operator, per head, on the
  `V` axis. Not part of KDA.
* No softmax anywhere. No `1/sqrt(d)` other than `scale`.

### 7.2 The sign in the forward substitution — a real bug in fla's naive.py

The forward substitution must be fed the **negated** strictly-lower block:

```
A ← −StrictTril( Diag(β) Z )       # negate
```

* Kimi Linear Appendix C: `A = -A.masked_fill(mask, 0)` ✓ has the minus.
* `fla/ops/kda/chunk_intra.py`: `b_Ai00 = -tl.where(m_A, b_Ai00, 0)` and
  `b_a00 = -tl.load(Akkd + ...)` ✓ has the minus, in both the fused and the
  `safe_gate` paths.
* `fla/ops/kda/naive.py::naive_chunk_kda`: `A = A.masked_fill(mask, 0)` — **no
  minus**. That is wrong.

I verified this rather than assuming it:
`test_ref.py::test_forward_substitution_sign` builds the chunked forward twice
from the same `Z`, once with the negation and once without, and asserts the
negated one matches `kda_recurrent` to `<1e-11` while the un-negated one
disagrees by `>1e-3` (it lands around `1e-1` relative on my inputs). The two
matrices are also asserted to actually differ, so the test cannot pass
vacuously. So: **implement the paper's version, not the `naive.py` version.**

### 7.3 Short vs long: two kernels, and which one is "the" operator

Both formulations exist and both are implemented:

| regime | what runs | source |
|---|---|---|
| training, prefill | the chunked algorithm, §3.3–3.5 | `fla/ops/kda/chunk.py` → `chunk_kda_fwd` |
| decode (autoregressive) | the token recurrence, Eq. 1 | `fla/ops/kda/fused_recurrent.py` → `fused_recurrent_kda` |

`fla/layers/kda.py` picks between them:

```python
if torch.is_grad_enabled():                        mode = "chunk"
elif q_len <= 64 and not self.training:            mode = "fused_recurrent"
else:                                              mode = self.mode    # "chunk"
```

The paper agrees: §6.3 "During the prefill phase, the model utilizes a
FLOP-intensive chunk kernel, while switching to the more efficient recurrent
kernel (Eq. 2) for autoregressive generation."

**The optimized kernels implement the chunkwise form**, with the `C×C`
intra-chunk algebra and the `NT`-step inter-chunk scan fused into ~5 kernels
(`chunk_kda_fwd` orchestrates: gate+cumsum → intra/UT → inter-chunk state scan
→ output). The `C×C` block work is further decomposed into 16×16 sub-blocks
for both the decay range and the triangular inverse. There is no
"short/long convolution" in KDA — the short conv of §6.3 is a depthwise
`k=4` conv on the *inputs*, shared with GDN, not a KDA-specific two-regime
formulation.

`kda_dense` in `ref.py` is a third path: one chunk of size `T` with the
triangular system solved by LAPACK `solve` instead of forward substitution. It
is a test oracle for the UT algebra, not a third production kernel.

---

## 8. Where the sources disagree

| # | disagreement | resolution |
|---|---|---|
| 1 | `γ^{i→j}` is defined as the inclusive `∏_{k=i}^{j} α^k` but is used in Props 1/2 and Eq. 3/8 where the half-open product is required. Off by a factor `α^i`. | **I used the half-open form** (`γ^i/γ^j`), because that is the paper's own `A_{i/j}` definition in §2.1 which Eq. 6/9 use, it is what the derivation gives, and it is what the kernels compute. §3.4. |
| 2 | Negation before the forward substitution: paper + Triton kernel yes, `fla/ops/kda/naive.py` no. | **I used the paper's (negated) version**; it is what the Triton kernel does and the only one that matches Eq. 1. Asserted in `test_ref.py`. §7.2. |
| 3 | `K/Γ` in Eq. 6/9: the paper writes a division, the kernels compute log differences. | **I never form the division** — masked log differences. Mathematically identical where both are representable (`test_gram_notations_agree`), numerically different where `Γ` underflows (`test_division_form_overflows`). |
| 4 | `scale` placement: paper and `naive.py` scale `q`; the fused kernel scales `Aqk` and the inter-chunk term. | **Equivalent.** I scale `q`. Asserted that `scale` is exactly linear in `q` and does not touch `S`. |
| 5 | Eq. 13's FLOP count `6Td_h² + 3TCd_h + TC²` undercounts. There are **five** `C×C×d_h` intra-chunk matmuls (`Z`, `w`, `u`, `Aqk`, `Aqk@v_new`), not three, and Eq. 13 counts the `3TCd_h` term at 1 FLOP/MAC while counting `6Td_h²` at 2. Counted consistently at 2 FLOP/MAC the real figure is **1.45× Eq. 13** (96.6 vs 66.6 GFLOP at `B=4,T=4096,H=32,K=128,C=64`). | Reported as a finding, not silently "fixed". See `cost.py`. |
| 6 | `Aqk`'s γ: the `(Γ⊙Q)(K/Γ)ᵀ` factorisation vs the difference form. | Identical; the difference form uses plain `q`. Both asserted. |

I did not find any disagreement about the recurrence itself (Eq. 1) — the
paper, `naive_recurrent_kda` and `fused_recurrent_kda` all agree, which is why
`kda_recurrent` is a trustworthy oracle.

---

## 9. Reproducing

```bash
cd /Users/huyan00/mycode/mc-cuda-tinygrad
.venv/bin/python kda/test_ref.py     # 17 checks; the load-bearing one is
                                     # "chunked == naive recurrence"
.venv/bin/python kda/cost.py         # every FLOP/byte number in NOTES.md
```

`ref.py` is float64 throughout and has no dependencies beyond NumPy. PyTorch is
not installed anywhere in this environment, and nothing here needs it.
