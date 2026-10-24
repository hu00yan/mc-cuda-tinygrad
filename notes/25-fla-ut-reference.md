---
kind: reference
---

# 25 — What the real KDA kernel actually does to the UT transform

**The problem this note closes.** A previous audit found that this repo's
spec and its TinyGrad port disagree about how the UT transform is performed,
and did not say which is right. This note settles it from the source.

---

## 0. The pin

Every line-level claim below is at:

| field | value |
|---|---|
| repository | `fla-org/flash-linear-attention` |
| commit | `79d12e8ec0ae72bc7f2189b4244d6532ce8c268c` |
| commit date | 2026-09-28T06:21:02Z |
| branch | `main` (HEAD at time of reading) |
| read on | 2026-09-29 |

Files read, all at that commit:

| file | lines | role |
|---|---|---|
| `fla/ops/kda/chunk_intra.py` | 988 | **the UT transform lives here** |
| `fla/ops/kda/chunk_intra_token_parallel.py` | 188 | default path's diagonal-block builder |
| `fla/ops/kda/wy_fast.py` | 381 | consumes the dense 64×64 inverse |
| `fla/ops/kda/chunk_fwd.py` | 142 | driver: gate+cumsum → intra → scan → output |
| `fla/ops/kda/chunk.py` | 494 | public API, `chunk_size` / `safe_gate` validation |
| `fla/ops/kda/naive.py` | 166 | PyTorch oracle — **not** the kernel |
| `fla/ops/utils/solve_tril.py` | 409 | generic helper — **not on the KDA path** |
| `fla/ops/utils/constant.py` | 10 | `RCP_LN2` |

Version robustness: `fla/ops/kda/chunk_intra.py` at tags `v0.4.2`, `v0.5.0`
and `v0.5.2` has the same `BC = 16` and the same four 14-iteration
substitution loops. KDA does not exist at `v0.3.2`. The finding below has
held for at least five minor releases.

Reached over HTTP from `raw.githubusercontent.com`; no local checkout of `fla`
exists in this repo (`_upstream/` contains only `mine.cu`).

---

## 1. Verdict, first

> **`kda/tiny_kda.py:157-158` is wrong. `kda/SPEC.md:282-283` is right.**

`SPEC.md:282-283` says:

> The same `BC=16` blocking is reused to do the UT transform in 4
> diagonal row-loops + 6 off-diagonal matmuls instead of 64 dependent steps.

That is an accurate description of `fla/ops/kda/chunk_intra.py` at the pinned
commit: 4 diagonal row-loops (lines 285, 290, 296, 302), 6 off-diagonal blocks
(320, 327, 332, 339, 344, 350), and not 64 dependent steps. It is the only
place in the repo's KDA material that describes the kernel's actual schedule.

Two loose numbers in it, neither of which changes the conclusion: the 6
off-diagonal blocks are produced by **16** `tl.dot`s, not 6 (§4), and the
serial alternative it displaces is `BT−1 = 63` steps, not 64.

`tiny_kda.py:157-158` says:

> Unrolled forward substitution is the faithful one **and is what the real
> kernel does**; the trip count is C, a compile-time constant, so unrolling is
> legitimate.

The **unrolled forward substitution is faithful** — it computes the right
inverse, and `kda/ref.py:277-292` computes the same thing. The clause
"**and is what the real kernel does**" is **false**. The real kernel does
neither `C` steps nor `C-1` steps. It does a two-level `BC=16`-blocked
inverse: 4 sub-chunks of 14 sequential row-updates each, plus 6 off-diagonal
16×16 blocks obtained from 16 `tl.dot` products.

`SPEC.md:136` — "its inverse is a row-wise forward substitution … `C−1`
*dependent* steps, **not** a matmul" — is also **wrong as a description of the
kernel**. It is a correct description of the *mathematical object* and of
`naive.py`, but the kernel does not execute `C−1 = 63` dependent steps. §4
gives the count.

So: **one claim is right (`SPEC.md:282-283`), one is half-right
(`tiny_kda.py:157-158`: right algorithm, false provenance), and one is right
about the math and wrong about the kernel (`SPEC.md:136`)** — and `SPEC.md`
contradicts *itself* between line 136 and line 283.

---

## 2. Q1 — How does it invert the unit lower-triangular matrix?

**Answer: an explicit two-level blocked inverse, `BC=16`, via row-wise forward
substitution inside each diagonal block and matrix products for the
off-diagonal blocks. Not sequential unrolled substitution, not Neumann
series, not a generic triangular solve.**

The whole thing is `chunk_kda_fwd_kernel_inter_solve_fused` in
`fla/ops/kda/chunk_intra.py:44-384`. Its own docstring, lines 67-78:

```
    Fused kernel: compute inter-subchunk Akk + solve_tril in one pass.
    Prerequisite: token_parallel has already computed diagonal Akk blocks in Akkd.

    This kernel:
    1. Computes off-diagonal Aqk blocks -> writes to global
    2. Computes off-diagonal Akk blocks -> keeps in registers
    3. Loads diagonal Akk blocks from Akkd (fp32)
    4. Does forward substitution on diagonals
    5. Computes merged Akk_inv
    6. Writes Akk_inv to Akk
```

### 2a. The algebra

Let `S` be the 64×64 unit lower-triangular matrix `I + StrictTril(D)`, cut into
a 4×4 grid of 16×16 blocks. Write

```
S = [ S00  0    0    0  ]        S_ij = 0 for i < j,  S_ii unit lower triangular
    [ S10  S11  0    0  ]
    [ S20  S21  S22  0  ]
    [ S30  S31  S32  S33]
```

Then the block inverse is obtained by the block-Schur recursion — first order,
not a truncated series:

```
S^-1_00 = S^-1_00,  S^-1_ii = S_ii^-1                              (diagonal)
S^-1_10 = -S11^-1 · S10 · S00^-1
S^-1_21 = -S22^-1 · S21 · S11^-1
S^-1_32 = -S33^-1 · S32 · S22^-1
S^-1_20 = -S22^-1 · ( S20 · S00^-1 + S21 · S^-1_10 )
S^-1_31 = -S33^-1 · ( S31 · S11^-1 + S32 · S^-1_21 )
S^-1_30 = -S33^-1 · ( S30 · S00^-1 + S31 · S^-1_10 + S32 · S^-1_20 )
```

Each of the 6 off-diagonal blocks is exact, not truncated, because `S_ii^-1`
is exact. This is what "6 off-diagonal matmuls" means.

### 2b. The loop structure, verbatim

**Diagonal stage** — `fla/ops/kda/chunk_intra.py:274-313`, four separate
top-level `for` statements. Quoting the first and third:

```python
274:    if not USE_SAFE_GATE:
275:        m_A = o_i[:, None] > o_i[None, :]
276:        m_I = o_i[:, None] == o_i[None, :]
277:
278:        b_Ai00 = -tl.where(m_A, b_Ai00, 0)
279:        b_Ai11 = -tl.where(m_A, b_Ai11, 0)
280:        if NC >= 3:
281:            b_Ai22 = -tl.where(m_A, b_Ai22, 0)
282:        if NC >= 4:
283:            b_Ai33 = -tl.where(m_A, b_Ai33, 0)
284:
285:        for i in range(2, min(BC, T - i_tc0)):
286:            b_a00 = -tl.load(Akkd + (i_tc0 + i) * HV*BC + o_i)
287:            b_a00 = tl.where(o_i < i, b_a00, 0.)
288:            b_a00 += tl.sum(b_a00[:, None] * b_Ai00, 0)
289:            b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
290:        for i in range(BC + 2, min(2*BC, T - i_tc0)):
291:            b_a11 = -tl.load(Akkd + (i_tc0 + i) * HV*BC + o_i)
292:            b_a11 = tl.where(o_i < i - BC, b_a11, 0.)
293:            b_a11 += tl.sum(b_a11[:, None] * b_Ai11, 0)
294:            b_Ai11 = tl.where((o_i == i - BC)[:, None], b_a11, b_Ai11)
295:        if NC >= 3:
296:            for i in range(2*BC + 2, min(3*BC, T - i_tc0)):
...
301:        if NC >= 4:
302:            for i in range(3*BC + 2, min(4*BC, T - i_tc0)):
...
308:        b_Ai00 += m_I
309:        b_Ai11 += m_I
310:        if NC >= 3:
311:            b_Ai22 += m_I
312:        if NC >= 4:
313:            b_Ai33 += m_I
```

- Range starts at **2**, not 0 or 1. Row 0 is identity; row 1's only
  off-diagonal entry `A[1,0]` is already correct because the tile was negated
  and the zero column makes the update a no-op. So the trip count is
  `BC - 2 = 14`.
- **The inner reduction is `tl.sum(b_a00[:, None] * b_Ai00, 0)` — a
  broadcast-multiply-and-reduce over a 16-element axis. There is not one
  `tl.dot` in this stage.** Verified: lines 274-314 contain 0 occurrences of
  `tl.dot`. This is a deliberate choice — the operation is a vector–matrix
  product, so it stays on the FMA path instead of paying the MMA layout
  round-trip. A porter should not "optimise" this into a matmul inside the
  kernel's model; in TinyGrad the natural lowering *is* a matmul, which is
  fine and equivalent (§9).
- Note the re-load at line 286: the raw diagonal block is read back out of
  global memory **one 16-float row at a time**, `BC` rows × 14 iterations.
  It was already loaded as a `[16,16]` tile at line 261.

**Merge stage** — `fla/ops/kda/chunk_intra.py:320-356`, all sixteen `tl.dot`s.
Two exact quotes, the shallowest and the deepest node of the DAG:

```python
320:    b_Ai10 = -tl.dot(
321:        tl.dot(b_Ai11, b_Akk10, input_precision=SOLVE_TRIL_DOT_PRECISION),
322:        b_Ai00,
323:        input_precision=SOLVE_TRIL_DOT_PRECISION
324:    )
...
350:        b_Ai30 = -tl.dot(
351:            b_Ai33,
352:            tl.dot(b_Akk30, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION) +
353:            tl.dot(b_Akk31, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION) +
354:            tl.dot(b_Akk32, b_Ai20, input_precision=SOLVE_TRIL_DOT_PRECISION),
355:            input_precision=SOLVE_TRIL_DOT_PRECISION
356:        )
```

Every operand is `[16,16]`, every result `[16,16]`, every product 16×16×16.
The merge is three matmul stages deep (§4).

**Store stage** — `chunk_intra.py:362-384`, ten separate `[16,16]` stores
assembling the dense 64×64 result. Only blocks `00, 10, 11, 20, 21, 22, 30, 31,
32, 33` are ever written; the six blocks above the diagonal are left at the
`torch.zeros` of line 833.

### 2c. The same algorithm exists twice, and KDA uses the inlined copy

`fla/ops/utils/solve_tril.py` contains a generic, nearly identical
implementation — `solve_tril_16x16_kernel` (lines 44-97),
`merge_16x16_to_32x32_inverse_kernel` (114-187), and
`merge_16x16_to_64x64_inverse_kernel` (204-356), dispatched by `BT` at lines
392-397. **KDA never calls it.** The KDA path is a hand-inlined fork inside
`chunk_intra.py`.

Two things follow, and both are traps:

1. The generic helper's default precision is `ieee`, from
   `FLA_TRIL_PRECISION = os.environ.get('FLA_TRIL_PRECISION', 'ieee')`
   (`solve_tril.py:19`). KDA does **not** read that variable. It hardcodes
   tf32-when-available (`chunk_intra.py:21-24`):
   ```python
   21: if IS_TF32_SUPPORTED:
   22:     SOLVE_TRIL_DOT_PRECISION = tl.constexpr('tf32')
   23: else:
   24:     SOLVE_TRIL_DOT_PRECISION = tl.constexpr('ieee')
   ```
   with the comment at line 319: `# we used tf32 to maintain matrix inverse's
   precision whenever possible.` So KDA's merge matmuls run in **tf32**
   (10 explicit mantissa bits, fp32 exponent range) on Ampere and later.
   `kda/ref.py` is float64 throughout. Any accuracy comparison against the
   kernel is measuring this, not the algorithm.
2. `utils/solve_tril.py` has no `β` and no decay anywhere in it. It inverts
   whatever strictly-lower-triangular matrix it is handed. Reading it to learn
   what the KDA kernel computes will send you down the wrong path.

---

## 3. Q2 — What is `BC`?

**`BC` is 16, hardcoded, and it is a distinct constant from `BT` (the chunk
size). It is the *same* variable the decay blocking uses.**

`fla/ops/kda/chunk_intra.py:821-829`:

```python
821:    B, T, H, K, HV = *k.shape, gk.shape[2]
822:    BT = chunk_size
823:    if BT not in (32, 64):
824:        raise ValueError(f"KDA intra chunk kernel only supports chunk_size 32 or 64, got {BT}.")
825:    BC = 16
826:    if chunk_indices is None and cu_seqlens is not None:
827:        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
828:    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
829:    NC = triton.cdiv(BT, BC)
```

So `BC = 16` unconditionally, for `BT = 32` (`NC = 2`) or `BT = 64`
(`NC = 4`). `BT` is validated separately in `fla/ops/kda/chunk.py:442-444`:
`chunk_size = kwargs.pop("chunk_size", 64)` then `if chunk_size not in (32,
64): raise`. `SPEC.md:289` is correct about that.

The backward driver uses `BC = min(16, BT)` (`chunk_intra.py:934`) — same value
given `BT ≥ 32`.

**It is genuinely the same blocking for decay and for the inverse.** The
`BC`-row groups are used as:

- the *decay reference* group: `b_gn1/b_gn2/b_gn3` are loaded at the **first
  row of sub-chunk 1/2/3** — `chunk_intra.py:160, 179, 204`:
  `b_gn1 = tl.load(g + i_tc1 * HV*K + o_k, mask=m_k, other=0)`;
- the *inverse* block size: `b_Ai00/11/22/33` are `[BC, BC]`, the six
  off-diagonal blocks are `[BC, BC]`, and every merge `tl.dot` is
  `[BC,BC]×[BC,BC]→[BC,BC]`.

`SPEC.md:282`'s "The same `BC=16` blocking is reused to do the UT transform"
is therefore literally true, not a loose paraphrase.

One asymmetry worth knowing: the *reference point* differs between the two
gate paths. The default path uses the sub-chunk's **first** row
(`i_tc1`), the `safe_gate` path uses the sub-chunk's **midpoint**
(`chunk_intra.py:756, 759`: `min(BC // 2, T - i_ti - 1)`, i.e. `+8`), with
the comment at 763-764 `# current block, keep numerical stability by
subtracting the left boundary less than 85 to avoid overflow in exp2`. The
reference cancels in the product either way, so both are the same operator —
but the default path's exponent range is up to `BC·|g| = 16·|g|` across a
sub-chunk, not `8·|g|`.

---

## 4. Q3 — How many genuinely dependent steps?

First, a correction to the framing of the question. A 64×64 unit
lower-triangular matrix has `64·63/2 = 2016` strictly-lower entries. The
4,032 figure that appears in this repo is `C(C-1) = 64·63`, and
`notes/13-schedule-vs-decay.md:489-493` attributes it correctly — to
`kda/tiny_kda.py:178-194`, this repo's own nested `for i in range(C)` ×
`for r in range(1, C)`, and it uses it as a measured constant of a measured
**TinyGrad** launch time. That attribution is right. What 4,032 is *not* is a
property of the real kernel, and it should not be used as the bar the kernel
has to clear. The kernel's actual counts are below.

### The kernel's count, at `BT=64`, `BC=16`, `NC=4`

Three different counts are defensible, and which one you want depends on
whether you are counting instruction issue, data-dependency latency, or
tensor-core pipeline depth. All three are given.

| what | count |
|---|---|
| loop iterations actually issued (4 loops × 14) | **56** |
| data-dependency critical path of the substitution | **14** |
| `tl.dot` calls in the merge | **16** |
| tensor-core dependency depth of the merge DAG | **3** |
| worst-case serial latency, substitution + merge | **17** round-trips |

Re-deriving the merge DAG (`chunk_intra.py:320-356`):

- **depth 1**, needs only the four diagonal inverses, mutually independent:
  `b_Ai10` (2 dots, line 320-324), `b_Ai21` (2, 327-331), `b_Ai32` (2,
  339-343) → 6 dots
- **depth 2**, needs `b_Ai10` / `b_Ai21`: `b_Ai20` (3, 332-337), `b_Ai31`
  (3, 344-349) → 6 dots
- **depth 3**, needs `b_Ai20`: `b_Ai30` (4, 350-356) → 4 dots

6 + 6 + 4 = **16**. Verified by counting `tl.dot` in lines 315-357: 16.

The 4 substitution loops write disjoint register tiles (`b_Ai00/11/22/33`),
so although they are written as four separate top-level statements and issue
serially (56 iterations of real work), the *data* dependency within each is
only 14 deep and the four are independent of one another. So:

> **The UT transform costs 56 issued substitution iterations plus 16 `tl.dot`
> products in a depth-3 DAG. Not 63. Not 2016. Not 4032.**

### It depends on tensor shapes

The trip counts are **runtime-bounded, not unrolled**. `BC` is `constexpr`,
but the upper bound `min(BC, T - i_tc0)` is a runtime `int32` (`T` is
declared `do_not_specialize=['T']` at `chunk_intra.py:43`), so Triton emits a
real loop.

For sub-chunk `k` (`k = 0..NC-1`) with base `16k`, the trip count is

```
max(0,  min(16(k+1), T - 64·i_t)  -  (16k + 2))
```

- **Full chunk** (`T - 64·i_t ≥ 64`): `14` per sub-chunk, `4 × 14 = 56` total.
- **Tail chunk** with `r = T - 64·i_t` rows remaining, `1 ≤ r ≤ 63`: the four
  counts become `min(16,r)-2`, `min(32,r)-18`, `min(48,r)-34`,
  `min(64,r)-50`, each clamped at 0. For `r = 20`: `14, 2, 0, 0` = 16 total.
- **`BT = 32`** (`NC = 2`): `2 × 14 = 28` iterations, and the merge is a
  single block pair — `b_Ai10` only, 2 `tl.dot`s. The `if NC >= 3` /
  `if NC >= 4` guards at lines 326, 338 and the store guards at 369, 376
  make this work.

The loop bound is also load-bearing for *correctness*, not just speed: the
re-load at line 286 is unmasked, and the `tl.where(o_i < i, b_a00, 0.)` at
line 287 is what makes reading a possibly-out-of-range row safe. Stop the loop
early and that invariant changes.

---

## 5. Q4 — The matmul structure

`chunk_intra.py:140-223`, the `i_k` loop that builds the six off-diagonal
blocks. `BK` is autotuned over `{32, 64}` (line 37).

**Building the six off-diagonal `[16,16]` blocks** (`Akk`) and the six
matching `[16,16]` `Aqk` blocks. All 12 dots per `i_k` iteration, all
`[BC,BK] × [BK,BC] → [BC,BC]`:

| line | block | operands |
|---|---|---|
| 166-167 | `Aqk10`, `Akk10` | `dot(q1·exp2(g1−gn1), (k0·exp2(gn1−g0))ᵀ)` |
| 186-187 | `Aqk20`, `Akk20` | `dot(q2·exp2(g2−gn2), (k0·exp2(gn2−g0))ᵀ)` |
| 191-192 | `Aqk21`, `Akk21` | `dot(q2·exp2(g2−gn2), (k1·exp2(gn2−g1))ᵀ)` |
| 212-213 | `Aqk30`, `Akk30` | `dot(q3·…, (k0·exp2(gn3−g0))ᵀ)` |
| 217-218 | `Aqk31`, `Akk31` | `dot(q3·…, (k1·exp2(gn3−g1))ᵀ)` |
| 222-223 | `Aqk32`, `Akk32` | `dot(q3·…, (k2·exp2(gn3−g2))ᵀ)` |

Each sub-chunk pair is registered once as `b_kgt` and used twice, once for
`Aqk` and once for `Akk`, so the `k`-side load and the `exp2` are shared. For
`K=128, BK=64`: 2 `i_k` iterations × 12 dots = 24 dots of `[16,64]×[64,16]`,
accumulating into 12 `[16,16]` fp32 accumulators declared at lines 122-135.

The sub-chunk reference `b_gn` **cancels inside each product**:
`exp2(g_i − gn) · exp2(gn − g_j) = exp2(g_i − g_j)`. It does not change the
mathematical range — the product still spans the full 63-step decay of the
chunk — but it keeps each *individual* `exp2` argument inside one sub-chunk of
decay, so no intermediate is ever larger than `e^{BC·|g|}`, and no `Γ` is ever
formed or divided. This confirms `SPEC.md:172-176` in substance. That
passage quotes a snippet with no line numbers, and the snippet is still
recognisably the same code, but the names have since grown indices: it writes
`b_kgt = tl.trans(b_k * exp2(b_gn - b_g0))` where line 164 is now
`b_kgt = tl.trans(b_k0 * exp2(b_gn1[None, :] - b_g0))`, and
`b_Akk = tl.dot(b_k * exp2(b_g - b_gn), b_kgt)` where the `exp2` is hoisted to
`b_gqn` at line 162 and the dot is `b_Akk10 = tl.dot(b_k1 * b_gqn, b_kgt,
b_Akk10)` at line 167.

`β` is applied as a **row scaling of the off-diagonal block only**, after the
dot: `chunk_intra.py:234, 243-244, 255-257`,
`b_Akk10 = b_Akk10 * b_b1[:, None]`. So the matrix inverted is
`I + StrictTril(Diag(β)·Z)`.

**The merge.** 16 `tl.dot`s, all `[16,16]×[16,16]→[16,16]`, in tf32. §2b.

**Materialised or implicit?**

- The six off-diagonal `[16,16]` raw blocks: **registers only**, never
  written to memory (docstring line 73). Twelve `[16,16]` fp32 accumulators
  live across the whole kernel.
- The four diagonal `[16,16]` raw blocks: **a separate global fp32 buffer**,
  `Akkd`, shape `[B, T, HV, BC]` (line 835, comment: `# Separate fp32 buffer
  for diagonal 16x16 blocks (for precision in solve_tril)`), produced by a
  *different* kernel and re-read one row at a time (line 286).
- The four diagonal `[16,16]` inverses: registers (`b_Ai00/11/22/33`).
- The final 64×64 inverse: **materialised densely** to `Akk`, shape
  `[B, T, HV, BT]` (`chunk_intra.py:831-833`, stores at 362-384).
  `Akk = torch.zeros(...)` at line 833, with the comment `# Akk must be
  zero-initialized - kernel only writes lower triangular`.
- The consumer does **not** exploit the block structure.
  `fla/ops/kda/wy_fast.py:91-94` loads `A` as one dense `[BT, BT]` tile with
  only a row mask (`m_A = m_t[:, None] & (o_A[None, :] < BT)`), then
  line 131 `b_w = tl.dot(b_A, b_kb.to(b_k.dtype))` and line 103
  `b_u = tl.dot(b_A, b_vb)` are both dense `[64,64]×[64,·]` products
  (`BK = BV = 64`, `wy_fast.py:286-287`).

That last point is a real, quantified inefficiency the kernel does *not* take.
`Akk` is strictly lower triangular, so:

- 6 of its 16 blocks are identically zero, so **37.5%** of those products' MACs
  multiply exact zeros and could be dropped by block sparsity alone;
- and each of the 4 diagonal blocks is itself triangular, so counting entries
  the true figure is **2016 / 4096 = 49.2%** of the MACs wasted.

Same in the backward — `wy_fast.py:260-261` does `tl.dot(b_dA.to(b_A.dtype),
b_A)` then `tl.dot(b_A, b_dA.to(b_A.dtype))`, two more dense 64×64 products,
both of which could drop the same share. Nobody in `fla` does. See §10.

---

## 6. Q5 — Column solves independently, or in a loop?

**Neither, and this is the question that decides whether 4,032 is required at
all: the kernel does not solve columns.**

The substitution is **row-wise**. Line 288:

```python
b_a00 += tl.sum(b_a00[:, None] * b_Ai00, 0)
```

- `b_a00` is a 16-vector indexed by **column** `j` (`tl.where(o_i < i, …)`
  keeps `j < i`).
- `b_Ai00` is `[16, 16]` indexed `[row, column]`.
- The reduction is over **axis 0, the row axis**.

So each iteration computes, for the current row `i` and every column `j < i`:
`A[i,j] += Σ_r A[i,r] · Ai[r,j]`. It is building one **row** of the inverse,
one row per loop iteration, and the loop is a genuine sequential dependency
chain within its own 16×16 block.

Consequences:

- The kernel never iterates over the 64 columns, and never over the 2016
  entries. It iterates `4 × 14` row-updates and then forms the 6 remaining
  blocks with products.
- 4,032 is not required, and neither is 63. For calibration, the inter-chunk
  state scan in the same operator *is* genuinely `NT` sequential steps:
  `fla/ops/common/chunk_delta_h.py:174-175`, inside
  `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` (def at line 59), reads
  ```python
  174:    # main recurrence
  175:    for i_t in range(NT):
  ```
  and the backward is `for i_t in range(NT - 1, -1, -1)` at line 489. So the
  deepest serial chain anywhere in KDA is `NT`, and the UT transform's 14-deep
  chains are a constant against it.
- `kda/ref.py:289-291` also does row-wise substitution, and its inner step
  `a[:, i, :][:, None, :] @ a[:, :, :i]` is a `[1,64]×[64,i]` matmul — i.e.
  **the same operation as the kernel's line 288, just at `C=64` instead of
  `BC=16`, and unrolled 63 times instead of 14×4.**

So `ref.py` is structurally the closest thing in this repo to the kernel, and
`tiny_kda.py` — which substitutes **column-wise**, solving column `i` by
`b[r] -= Σ_{j<r} A[r,j] b[j]` (`tiny_kda.py:171-189`, and the comment at
172-177 explains at length why the column-wise reading is the non-obvious one)
— is a *different* traversal of the same recurrence. Both give the same
inverse; only `ref.py`'s matches the kernel's traversal.

**Verified numerically.** `.agent-scratch/fla-ref/verify_ut.py` re-derives the
inverse four ways in float64 from a random `β ∈ [0.05, 0.95]` and
`Z ~ N(0, 0.02²)`:

```
   ref.py: max rel err vs exact = 2.082e-17   MATCH
naive.py: max rel err vs exact = 2.082e-17   MATCH
triton BC=16: max rel err vs exact = 2.082e-17   MATCH
```

with `triton BC=16` transcribed line-for-line from `chunk_intra.py:285-289`
(including the `range(2, BC)` start) and the merge transcribed from lines
320-356. All three are the same operator to round-off.

---

## 7. `naive.py` — a distinction worth preserving

`SPEC.md:449` records a divergence here. It is still real, and the cleanest
way to state it is:

> **`naive.py` is the kernel's algorithm with the blocking removed.**

`fla/ops/kda/naive.py:133-136`:

```python
133:    A = -A.masked_fill(mask, 0)
134:    for i in range(1, BT):
135:        A[..., i, :i] = A[..., i, :i].clone() + (A[..., i, :, None].clone() * A[..., :, :i].clone()).sum(-2)
136:    A = (A + torch.eye(BT, dtype=torch.float, device=q.device)) * beta[..., None, :]
```

Line-by-line against `chunk_intra.py:278-289`:

| | `naive.py` | `chunk_intra.py` |
|---|---|---|
| negate | `A = -A.masked_fill(mask, 0)` (133) | `b_Ai = -tl.where(m_A, b_Ai, 0)` (278) |
| start index | `range(1, BT)` (134) | `range(2, min(BC, …))` (285) |
| inner op | `(A[i,:,None] * A[:, :i]).sum(-2)` (135) | `tl.sum(b_a[:,None] * b_Ai, 0)` (288) |
| trip count | 63, **unrolled** (Python `range`) | 14, **runtime-bounded** (Triton `for`) |
| + I | `(A + eye(BT))` (136) | `b_Ai += m_I` (308-313) |
| merge | none | 6 blocks, 16 `tl.dot` (320-356) |

The `i=0` vs `i=1` start and the slice-`[:i]` vs `tl.where(o_i < i)` masking
are equivalent: at `i=1` the update is `A[1,0] += A[1,0]·A[0,0] = A[1,0] + 0`.
So `naive.py` is exactly `chunk_intra.py` with `BC` forced to `BT=64` and the
loops unrolled. Verified above — 2.08e-17 either way.

So `SPEC.md:449`'s row is **stale in its premise**. It records that "Negation
before the forward substitution: paper + Triton kernel yes, `naive.py` no" —
but `naive.py:133` is `A = -A.masked_fill(mask, 0)`, so `naive.py` *does*
negate and agrees with the kernel. The divergence `naive.py` still has is the
**blocking**, and it is not a numerical divergence at all: it is cost and
schedule only. `naive.py` agrees with the blocked result to 2.08e-17 in
float64, at 63 unrolled steps; `chunk_intra.py` gets there in 14-deep chains
and 16 matmuls. `SPEC.md:449` should be re-pointed at that, not at negation.

**The structural difference between the Triton kernel and `naive.py` is
*where `β` goes*:**

- `naive.py:131, 136` — `A = A * beta[..., None]` (rows) then
  `A = (...) * beta[..., None, :]` (columns). The trailing column scaling puts
  `Diag(β)` **inside** the operator: `naive`'s `A` is
  `(I + StrictTril(Diag(β)Z))^{-1} · Diag(β)`, the paper's `M`, and
  `naive.py:138-139` then does plain `A @ (g.exp() * k)` and `A @ v`.
- `chunk_intra.py:234` + `wy_fast.py:102, 113, 117` — `β` row-scales the
  *operands* (`b_vb = b_v * b_b[:,None]`, `b_kb = b_k * b_b * exp2(b_gk)`),
  and the stored `Akk` carries **no** trailing `Diag(β)`.

These are algebraically identical — verified, `M_kernel @ (β⊙V)` vs
`M_paper @ V` agree to **4.44e-16** — but they are different graphs, and a
porter who copies the paper's `M` *and* applies `β` to `v` and `k` double-applies
it and gets a relative error of order 1, not a rounding difference.

---

## 8. Q6 — Other things in the intra-chunk path a porter will get wrong

**8a. The default path does not build the diagonal 16×16 blocks with `tl.dot`
at all.** This is the biggest one. `safe_gate` defaults to `False`
(`chunk.py:47` and `chunk.py:210`), so the default is the `else` branch at
`chunk_intra.py:862-875`, which calls
`chunk_kda_fwd_intra_token_parallel`. That kernel
(`chunk_intra_token_parallel.py:117-127`) is **token-parallel**: the grid is
`(B*T, cdiv(HV, BH))` (line 168), one program per token, and:

```python
117:    for j in range(i_ts, min(i_t + 1, min(T, i_ts + BC))):
...
122:        b_kgj = tl.where(m_k[None, :], b_kj * exp2(b_g - b_gj), 0.0)
123:        b_Aqk = tl.sum(b_q * b_kgj, axis=1) * scale
124:        b_Akk = tl.sum(b_k * b_kgj, axis=1) * tl.where(j < i_t, 1.0, 0.0)
```

A `≤16`-iteration loop over `j`, each iteration a rank-1 accumulate, **no
`tl.dot`**. The `[16,16]` diagonal blocks are assembled by `B·T` concurrent
programs, each writing one column of one block. The `tl.dot(b_k * b_gq,
b_kgt)` form at `chunk_intra.py:773` is the **`safe_gate=True`** path only
(`chunk_kda_fwd_kernel_intra_sub_chunk`, grid `(NT, NC, B*HV)`, line 840),
and it is also the only place the diagonal substitution runs *inside* the
sub-chunk kernel (`chunk_intra.py:796-803`) — which is why the fusion kernel
guards its own substitution with `if not USE_SAFE_GATE:` (line 274) and loads
already-inverted diagonals at lines 261-268.

A port that assumes "16×16 blocks via matmul" has ported the non-default path.

**8b. `Diag(β)` goes on the operands, not on the inverse.** §7. Silent
order-1 error if you get it wrong.

**8c. The merge runs in tf32.** `chunk_intra.py:21-24`, with the comment at
line 319. 10 explicit mantissa bits. Not in the `FLA_TRIL_PRECISION` env var
that the generic helper uses. Anything comparing a float32/float64 port against
the kernel's own output is measuring tf32.

**8d. The `Γ` division is never formed, and the asymmetry is real.** The
decayed gram is **not symmetric** — `max|Z − Zᵀ| / max|Z| = 1.000` in float64
on three independent draws: `Z` and `Zᵀ` have nothing in common, so any port
that builds `Z` as a symmetric Gram is simply computing a different operator.
And never materialise `Γ`. Measured over three seeds of a plausible decay
(per-token log-decay `~U(0, 0.7)`, 64 tokens, `d_h = 11`), where the exponent
`g_i − g_j` spans `e^{−26}..e^{+26}` and the entries of `Z` span the same
range:

| how `Z` is formed | max rel err vs the log-difference form |
|---|---|
| `(Γ ⊙ K)(K/Γ)ᵀ` in float64 | 6.7e-16, 7.6e-16, 6.7e-16 (seeds 7/11/23) |
| `(Γ ⊙ K)(K/Γ)ᵀ` with `Γ` in float32 | **5.4e-08, 1.8e-08, 3.5e-08** |

`SPEC.md:253-257` reaches the right conclusion ("`K / Γ` must never be
materialised… evaluating the division gives `inf` as soon as `Γ` underflows");
that table is the number for it. The kernel's position — compute
`exp2(g_i − g_j)` as a difference and let the sub-chunk reference `b_gn` cancel
— is the only one of the three that is both safe and accurate.

**8e. The final inverse is dense and half of it is zero, and nobody exploits
it.** §5 — 6 of 16 blocks identically zero (37.5% of the MACs by block
sparsity), 2016 of 4096 entries (49.2%) counting the intra-block triangles.
A genuine, quantified free win for a port, and *not* what the kernel does — so
do not claim kernel parity for it.

**8f. `recompute_w_u_fwd` is a recompute, and the name hides it.**
`chunk_intra.py:900-910` calls it with `A=Akk`; `wy_fast.py:34-132` recomputes
`w` and `u` from `A`, and *also* optionally stores `qg`/`kg` (lines 118-129),
`kg` being `k * exp2(gn − g)` — the re-decayed key the inter-chunk scan
consumes (`chunk_fwd.py:100-113`). This is the boundary between the
intra-chunk and inter-chunk phases. `SPEC.md:430-433` describes the fusion
correctly.

**8g. The `Akkd` fp32 buffer exists and is a precision decision, not
leftovers.** `chunk_intra.py:834-835`. The diagonal blocks are the ones the
14-step chain accumulates into, and they must not be rounded to `k.dtype`
between the two kernels.

**8h. `AQK` is `≥` masked, `AKK` is `>` masked.** `chunk_intra.py:776-781`
(`m_Aqk = o_i[:,None] >= o_i[None,:]`, `m_Akk = o_i[:,None] > o_i[None,:]`)
and the matching `tl.where` pair. The diagonal of `Akk` must be zero, because
it becomes the unit diagonal by `b_Ai += m_I` (line 308). Getting this
backwards gives `I` doubled on the diagonal, which the tests would catch but
a "close enough" eyeball check would not.

**8i. An alternative the kernel does *not* use, for reference.** For a 16×16
strictly-lower `S`, `S^16 = 0`, so
`(I + S)^{-1} = (I − S)(I + S²)(I + S⁴)(I + S⁸)` exactly — three squarings
plus three products, 6 dependent 16×16 matmuls, **replacing the kernel's 14
sequential steps with a 6-deep matmul chain**. Verified to 1.4e-17 against
`np.linalg.inv` on all four diagonal blocks. The kernel does not do this; it
does the substitution. Recorded only so the two are not confused, and because
it is the shortest exact 16×16 unit-lower inverse available to a port.

---

## 9. How to implement it, without re-reading the kernel

Shapes: `S = StrictTril(Diag(β)·Z)` at `[B, NT, HV, 64, 64]`. Block view
`[B, NT, HV, 4, 16, 4, 16]`.

```python
# Names follow the kernel: D[k] = diagonal inverse of block k (the b_Ai{k}{k}
# register tiles), R[i,j] = the raw strictly-lower block S[i,j], O[i,j] = the
# inverse block. Loops over k and i are unrolled at trace time.
NC, BC = 4, 16                      # BC == 16 always; NC = BT // BC
R = block_view(StrictTril(Diag(beta) @ Z))   # [.., 4, 16, 4, 16]

# --- stage 1: four 16x16 diagonal inverses, 14 sequential row updates each ---
# NOT a loop over 64 columns, NOT a loop over 2016 entries, NOT a matmul.
D = {}
for k in range(NC):                                  # 4
    Ai = -tril(R[k, :, k, :], -1)                     # negate, zero upper incl. diag
    for i in range(2, BC):                            # 14
        a = where(col < i, -R[k, i, :], 0)            # the update vector, [16]
        a = a + (a[:, None] * Ai).sum(0)              # contract over the ROW axis
        Ai = where(row == i, a[None, :], Ai)          # write row i
    D[k] = Ai + eye(BC)                               # b_Ai{k}{k} += m_I

# --- stage 2: six off-diagonal blocks by 16 matmuls, 3 levels deep ---
# depth 1
O[1,0] = -D[1] @ R[1,:,0,:] @ D[0]                                        # 2 dots
O[2,1] = -D[2] @ R[2,:,1,:] @ D[1]                                        # 2 dots
O[3,2] = -D[3] @ R[3,:,2,:] @ D[2]                                        # 2 dots
# depth 2
O[2,0] = -D[2] @ (R[2,:,0,:] @ D[0] + R[2,:,1,:] @ O[1,0])                # 3 dots
O[3,1] = -D[3] @ (R[3,:,1,:] @ D[1] + R[3,:,2,:] @ O[2,1])                # 3 dots
# depth 3
O[3,0] = -D[3] @ (R[3,:,0,:] @ D[0] + R[3,:,1,:] @ O[1,0]
                  + R[3,:,2,:] @ O[2,0])                                  # 4 dots

Akk = block_cat({**D_diag, **O})   # [.., 64, 64], strictly lower, 10 nonzero blocks
```

Notes for whoever implements:

- The inner step can be `where(row == i, (a[None, :] @ Ai), Ai)` instead of
  `where(row == i, (a[:,None] * Ai).sum(0)[None, :], Ai)`. That is the same
  contraction written as a `[1,16]×[16,16]` matmul, and `kda/ref.py:291` already
  does it. Measured in float64 the two agree to 2.1e-17. In a tensor framework
  use the matmul form — it is fewer nodes and it is what a compiler will pick.
- Do **not** then apply `Diag(β)` to `Akk`. Apply it to the right operand, or
  fold it into the row scaling of `k` and `v` — `β` is a row scaling of
  `StrictTril(Z)` when inverting, and a column scaling of the result if you
  want the paper's `M`, but you only need one of the two places.
- `w = Akk @ (k·exp2(g_cum))`, `u = Akk @ (β⊙v)`. Those are the only two
  consumers (`wy_fast.py:103, 131`).
- If you want the paper's `M` rather than the kernel's `Akk`, multiply by
  `Diag(β)` on the right. Both are correct; they are different graphs and the
  kernel uses the second.

Throwaway verification scripts, kept for now and safe to delete:
`.agent-scratch/fla-ref/verify_ut.py` (four-way inverse equivalence and the
`β`-placement identity), `.agent-scratch/fla-ref/verify_extra.py` (the 16×16
doubling identity), `.agent-scratch/fla-ref/verify_gamma.py` (the `Γ` table in
§8d). The `fla` sources they were written against are the flattened copies
`fla_ops_*.py` in the same directory.

---

## 10. Corrections this note forces, itemised

| # | file:line | claim | verdict |
|---|---|---|---|
| 1 | `kda/tiny_kda.py:157-158` | "Unrolled forward substitution … **is what the real kernel does**" | **FALSE.** The kernel blocks at `BC=16`: 4 × 14 row-updates + 16 `tl.dot`. The port is faithful to `naive.py`, not to `chunk_intra.py`. |
| 2 | `kda/SPEC.md:282-283` | "4 diagonal row-loops + 6 off-diagonal matmuls instead of 64 dependent steps" | **ESSENTIALLY CORRECT.** 4 loops: `chunk_intra.py:285, 290, 296, 302`. 6 off-diagonal blocks: `320, 327, 332, 339, 344, 350`. Two loose counts: 16 `tl.dot`s, not 6; and 63 serial steps displaced, not 64. |
| 3 | `kda/SPEC.md:136` | "`C−1` *dependent* steps, **not** a matmul" | **WRONG about the kernel.** Right about the math and right about `naive.py`. The kernel's substitution stage contains 0 `tl.dot`; the merge stage contains 16. Actual: 56 issued iterations, 14-deep data chain, 16 matmuls in a depth-3 DAG. |
| 4 | `kda/SPEC.md:136` vs `:282` | — | The spec **contradicts itself**. §3.3 (paper framing) and §4.4 (kernel description) describe different algorithms under the same operator name. |
| 5 | `kda/SPEC.md:449` | "paper + Triton kernel yes, `fla/ops/kda/naive.py` no" (re negation) | **STALE.** `naive.py:133` is `A = -A.masked_fill(mask, 0)` — it does negate, agreeing with the kernel. The `naive.py` divergence that remains is the *blocking*, and it is cost-only, not numerical (2.08e-17 either way). |
| 6 | `kda/SPEC.md:173-176` | quotes a `b_kgt` / `b_Akk` snippet from `chunk_intra.py` | **Substance correct; variable names have since grown indices.** SPEC quotes no line numbers, and the snippet is still recognisably the same code. At this commit line 164 is `b_kgt = tl.trans(b_k0 * exp2(b_gn1[None, :] - b_g0))` and the dot is `b_Akk10 = tl.dot(b_k1 * b_gqn, b_kgt, b_Akk10)` at line 167. The claim that the sub-chunk reference `b_gn` cancels in the product is correct. |
| 7 | `notes/13-schedule-vs-decay.md:489-493` | `C(C-1) = 4,032` iterations | **CORRECTLY ATTRIBUTED** — to `kda/tiny_kda.py:178-194`, and used as a TinyGrad launch-time constant, which is what it is. No edit needed. Recorded here only so 4,032 is never again quoted as a kernel number: the matrix has 2,016 strictly-lower entries, and the kernel pays 56 loop iterations. |
| 8 | `kda/SPEC.md:432-433` | "The `C×C` block work is further decomposed into 16×16 sub-blocks for both the decay range and the triangular inverse" | **CORRECT.** Confirmed independently at `chunk_intra.py:160, 179, 204` (decay) and `108, 259-268, 320-356` (inverse). |
| 9 | `kda/SPEC.md:289` | "`chunk.py` raises unless `chunk_size ∈ {32, 64}`" | **CORRECT.** `chunk.py:442-444`, `chunk_intra.py:822-824`. |
| 10 | `kda/SPEC.md:273` | "`RCP_LN2 = 1.4426950216` in `fla/ops/utils/constant.py`" | **CORRECT.** Line 10 of that file. |
| 11 | `kda/ref.py:355-356` | `Minv = _forward_substitution(-Btri)` then `Minv = Minv * bc[..., None, :]` | **Faithful to the paper's `M`, not to the kernel's `Akk`.** Both are correct and agree to 4.4e-16, but the kernel keeps `Diag(β)` off the inverse and row-scales the operands instead (`chunk_intra.py:234`, `wy_fast.py:102, 113`). Not an error in `ref.py` — but `ref.py` and `tiny_kda.py` are on the `naive.py` side of this split, and a comparison against the kernel's own `Akk` has to account for it. §7. |
