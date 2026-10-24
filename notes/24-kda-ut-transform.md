---
kind: reference
---

# 24 — The UT transform: what the kernel does, what the naive port did, and what blocking it is worth

**The claim under test.** `kda/tiny_kda.py:178-194` solves the UT transform with
`C x (C-1) = 4032` sequentially dependent TinyGrad ops, one scalar per
`(batch, chunk, head)`, and its comment says that unrolled forward substitution
"is what the real kernel does". The brief for this note was that the released
kernel is `BC=16` blocked instead, that `kda/SPEC.md` says so, and that the port
therefore does 64x more sequential steps than the reference's own count.

**The verdict.** The kernel is blocked at `BC=16`, the spec is right, and the port
was leaving a large factor on the table — but *not* for the reason the 4,032
figure suggests, and the win is worth more than the 4,032 figure suggests,
because the op count is a floor and not a cost.

| UT transform alone, `C=64`, `B*NT*HV=64`, Metal, tinygrad 0.14.0 | naive (`tiny_kda.py`) | blocked (`tiny_kda_opt.py`) |
|---|---|---|
| inner-loop iterations | 4,032 | 14 |
| **arithmetic uops in the graph** | **20,354** | **89** |
| **emitted kernels** | **6,021** | **24** |
| longest dependent chain in the graph | 381 | 89 |
| reduce ops (a matmul is one of these) | 3,968 | 24 |
| device time, steady state | 270.6 ms (range 147.5-298.4) | 1.12 ms (range 1.10-1.18) |

The whole operator, same shape: **12,090 -> 89 kernels (136x)**, 41,052 -> 403
uops (102x), 236.5 -> 11.8 ms device time (20x).

The number to take away is the kernel count. 4,032 is how many *ops the Python
loop asks for*; it is **not** how many steps the hardware executes. TinyGrad
does not keep that chain in registers — it realises every step — so the solve
becomes 6,021 kernels, which is **worse than the op count**. Kernels run in
stream order, so 6,021 is also the physically serial step count.

The graph-theoretic longest chain is only 381, because the 64 column solves are
mutually parallel in the DAG. A depth count is the right way to compare
algorithms and the wrong way to predict time; the kernel count is what the
launcher's queue sees, and the gap between the two is the interesting part.

---

## 1. What the reference actually computes

`fla/ops/kda/chunk_intra.py::chunk_kda_fwd_kernel_inter_solve_fused`, from
`fla-org/flash-linear-attention`:

- URL: `https://raw.githubusercontent.com/fla-org/flash-linear-attention/main/fla/ops/kda/chunk_intra.py`
- commit read: **`13e67cb17a3e3a934c5dda71fb2d8f32ceea608d`**, dated 2026-09-08,
  which is the current `HEAD` for that path (I asked the commits API rather than
  trusting that `main` is what I think it is). SHA-256 of the file:
  `5e9d3c0d8110bc8c865788cd4037bb5a8a80e54a04082563090590c0d473ece8`.
- `kda/SPEC.md` §4.4 pins `79d12e8ec0ae72bc7f2189b4244d6532ce8c268c`. I fetched
  that commit too: it returns a **byte-identical** file, same SHA-256. The two
  pins name the same code, so there is no disagreement to resolve.

| what | lines in `chunk_intra.py` | what it is |
|---|---|---|
| `BC = 16`, `NC = triton.cdiv(BT, BC)` | 628-631 | fixed 16x16 sub-chunking of the `C=64` chunk |
| the 6 off-diagonal `Aqk`/`Akk` blocks, each a `tl.dot` | 137-219 | `[BC,BC]` from `[BC,K] x [K,BC]`, batched over the `K` loop |
| `b_Akk10 = b_Akk10 * b_b1[:, None]` etc. | 229-253 | each off-diagonal gram block is row-scaled by `beta` |
| **diagonal stage**: four `for i in range(2, min(BC, T - i_tc0))` loops | 288-317 | **one per sub-chunk**, all four live in registers simultaneously |
| the in-block inner op `b_a += tl.sum(b_a[:, None] * b_Ai, 0)` | 291-294 | a broadcast-reduce. **No `tl.dot` in the diagonal stage at all.** |
| `b_Ai00 += m_I` | 306-314 | the unit diagonal, added per block |
| **merge stage**: `b_Ai10`, `b_Ai21`, `b_Ai20`, `b_Ai32`, `b_Ai31`, `b_Ai30` | 316-357 | 6 `tl.dot`s of `[16,16]x[16,16]`, in a 3-deep DAG |
| `b_Ai00 = -tl.where(m_A, b_Ai00, 0)` | 276-286 | the sign: the strictly-lower block is **negated** before the substitution |

So the brief's question — *is the reference `BC=16` blocked?* — is **yes, on
both counts the spec claims** (`kda/SPEC.md` §4.4, "The same `BC=16` blocking is
reused to do the UT transform"): 4 diagonal row-loops + 6 off-diagonal
matmuls. Four loops of `BC-2 = 14` iterations, one per diagonal block, then six
16x16x16 matmuls for the off-diagonal blocks.

> Line numbers into `kda/SPEC.md` are given **as of the copy I read** (mtime
> 12:16, while I was working) and are cited by section as well, because another
> agent was editing that file concurrently and the numbers have already moved
> once: §3.3's "C−1 dependent steps" was at line 136 and is now at 142, §4.4's
> blocking sentence was at 282 and is now at 294. Section numbers are stable;
> line numbers are not.

Two things in that source that are easy to get wrong when porting:

1. **The diagonal stage is not a matmul.** It is a row-wise broadcast-reduce
   over a `[16,16]` register tile. Porting it as a matmul multiplies the
   arithmetic by 16 for nothing. The matmuls are only in the merge stage.
2. **The merge stage is a DAG, not a loop.** `b_Ai30` depends on `b_Ai20`, which
   depends on `b_Ai10`; `b_Ai31` depends on `b_Ai21`. Unrolled at `NC=4` that is
   four levels: `{Ai10, Ai21, Ai32}`, `{Ai20, Ai31}`, `{Ai30}`.

### The sweep, written once

Those six hand-written expressions are the `NC=4` unrolling of one recurrence.
For any `NC`, with `X = A^{-1}` in `NC x NC` blocks of `BC x BC` and `E_i` the
block-identity *row* `i` (`I` in column block `i`, zero elsewhere):

```
X_i = Ainv_ii · ( E_i − Σ_{k<i} A_ik X_k )                                    (*)
```

`i = 0` is `X_0 = Ainv_00` with an empty sum, so there is no special case, and
`(*)` also covers the diagonal blocks — so the whole inverse needs only the
diagonal stage and `(*)`. Expanding `(*)` at `NC = 4` reproduces fla's six
expressions term for term. `kda/tiny_kda_opt.py::ut_transform` implements `(*)`
with the inner sum as **one batched matmul over the block axis `k`**, which is
possible because a batched `@` *is* a loop over that axis, and the contraction
index `p` of `A_ik[b,p] X_k[p,j,c]` then sits as the last axis of the left
operand and the second-to-last of the right, which is what `@` needs.

### A sign error in this repo's documentation (not in the code)

`kda/ref.py:277-287` — still there as I write this — says of
`_forward_substitution(A)` that it "Returns `(I + StrictTril(D))^{-1}`" for input
`D`. **It returns `(I - D)^{-1}`.** The in-place `+=` row substitution is the
inverse of `I - A`, which is exactly why `ref.py:355` has to pass it `-Btri`.
Measured, not inferred:

```python
A0 = np.tril(rng.normal(size=(6, 6)), -1)
ref._forward_substitution(A0)  - np.linalg.inv(np.eye(6) + A0)   # 4.65e+00
ref._forward_substitution(-A0) - np.linalg.inv(np.eye(6) - A0)   # 3.55e-15
```

`kda/SPEC.md` §3.3 (lines 141-144 as of my read) carried the same sign in its
pseudocode: `A ← −StrictTril(Diag(β)Z)`, then `A[i,:i] += A[i,:] @ A[:,:i]`, then
`M ← (A + I) Diag(β)`. That is `(I − StrictTril(Diag(β)Z))^{-1} Diag(β)`, which
contradicts its own Eq. 6 on line 130 and both implementations. Another agent
amended §3.3 while I was working — it now marks the block as the naive form and
points at §4, and a correction note at line 147 attributes the "C−1 dependent
steps" wording to a ledger entry. I am not claiming that edit; my contribution
here is the `ref.py` docstring, which is still wrong, and the `np.linalg.inv`
check above, which is the evidence for what that routine actually returns.

**Correction to the brief.** The brief asked me to check the sign convention in
`tiny_kda.py:150-153`, which explains that `ref._forward_substitution` "solves
`(I + StrictTril(X)) M = I`, so passing `-Btri` means the system matrix is
`A = I + StrictTril(Btri)`". The conclusion is right and the port is right; the
premise is not — the routine *inverts* rather than solves, and passing `-Btri`
gives `I − StrictTril(Btri)`. The port is correct anyway because it never calls
that routine: it builds `A = I + StrictTril(Diag(β)Z)` itself and substitutes
against `A` with a subtraction.

---

## 2. The dependent-step count, counted

Four different quantities get called "the step count". All four, measured at
`C=64`, `N = B*NT*HV = 64`:

| quantity | naive | blocked `BC=16` | how |
|---|---|---|---|
| inner-loop iterations | 4,032 | 14 | the Python `for r in range(1, C)` nest |
| **arithmetic uops in the graph** | **20,354** | **89** | walked the realised `UOp` DAG |
| **emitted kernels** | **6,021** | **24** | `DEBUG=2` capture on the real device |
| reduce ops | 3,968 | 24 | `REDUCE` uops; a matmul is one |
| longest dependent chain in the DAG | 381 | 89 | longest path over that DAG |
| device time, steady state | 270.6 ms | 1.12 ms | per-realisation, program cache warm |

**Ops, not steps.** 4,032 is `64 x 63` iterations of a body that is four
arithmetic uops (a multiply, a reduce, a multiply, a subtract) — about 16,128 —
plus the outer loop's 64 RHS constructions and 64 accumulations: 20,354
measured. Those 20,354 uops are **not** what runs. TinyGrad materialises each
one, so the hardware executes 6,021 kernels. 6,021 > 4,032: the op count is a
*floor* on kernel count here, and the gap is the cost of the naive structure.

**Why the DAG chain is 381 and the kernel count is 6,021.** The 64 column solves
are independent: column `i` of `A^{-1}` does not depend on column `i'`. So the
graph is 64 parallel chains of 252 uops feeding one 64-step accumulation, and
the longest path through it is `252 + 129 = 381`. That is a *lower* bound on
what the hardware must do. What it does is 6,021 kernels in stream order. Both
numbers are in the table because the gap between them is the finding.

### Where the 89 goes

Exact per-phase arithmetic-uop counts, obtained by building the graph to a cut
point and differencing (`C=64`, `BC=16`, `NC=4`):

| phase | uops | of which reduces |
|---|---|---|
| extract the 4 diagonal blocks (1 mask-multiply + 1 reduce) | 3 | 1 |
| **in-block substitution, 14 steps, all 4 blocks at once** | **56** | 15 |
| add the unit diagonal | 1 | — |
| seed `X_0` (1 matmul + 1 one-hot) | 3 | 1 |
| **sweep level 1** | **8** | 2 |
| **sweep level 2** | **9** | 3 |
| **sweep level 3** | **9** | 3 |
| **total** | **89** | **24** |

Every one of the 89 is on the critical path — longest chain == total == 89 — so
there is no off-path arithmetic to overlap. The graph is a pure chain, and the
only lever on time is the number of links in it.

### How the 4,032 decomposed, and where each piece went

- **Batched: 4,032 -> 59** (the 56 in-block steps + 3 to extract the blocks). The
  64 independent column solves were not a loop that needed batching; they were
  unnecessary. The right-hand side is `Diag(beta)`, so all 64 columns together
  are just `A^{-1}` — the column loop was computing `A^{-1}` one column at a
  time, 64 times over, in sequential structure. Once blocked, the `NC = 4`
  diagonal blocks invert **simultaneously** (the block axis is a batch axis), so
  the substitution runs `BC-2 = 14` times instead of 4,032, handling every block
  and every `(batch, chunk, head)` at once. **68x fewer iterations.**
- **Blocked: 63 -> 17.** The 63 sequential substitutions *within* one system
  become 14 in-block steps plus `NC-1 = 3` block levels, where each level's
  innermost sum `Σ_{k<i} A_ik X_k` is one matmul. **3.7x fewer levels.**
- **Also removed, outside the 4,032:** the `M` materialisation. `M = Ainv Diag(beta)`
  and `M @ X = Ainv @ (beta (.) X)` term for term, so `beta` goes onto the
  right-hand side of the two dense matmuls instead, and the naive port's 64
  sequential rank-1 accumulations into a `[C,C]` tensor plus a `[C,C]` multiply
  both disappear.
- **Kernels per arithmetic op: 0.30 -> 0.84.** The blocked form fuses almost
  perfectly; the naive one realised more kernels than it had ops. The indexed
  write-back in the in-block loop is not an extra kernel — the whole solve is
  24 kernels, under 2 per level.

### The whole operator

The solve is not the whole operator, so the whole operator was measured too,
`B=2 T=512 H=HV=4 K=V=64 C=64 NT=8`, same protocol, steady state:

| | DEP | uops | reduces | kernels | device ms (steady range) |
|---|---|---|---|---|---|
| naive (control) | 449 | 41,052 | 7,999 | 12,090 | 236.11 – 236.81 |
| **blocked `BC=16`** | 157 | 403 | 104 | **89** | **10.88 – 12.68** |
| blocked `BC=8` | 161 | 411 | 112 | 93 | 9.76 – 13.53 |
| blocked `BC=32` | 203 | 495 | 124 | 111 | 15.43 – 15.44 |
| unblocked `BC=64` | 322 | 733 | 182 | 167 | 45.16 – 46.47 |

**136x fewer kernels, 102x fewer uops, 20x less device time.** The naive port's
UT transform was 381 of the 449-step critical path; after blocking it is 89 of
157. Note the operator-level win (20x) is much smaller than the solve-level win
(241x), because the blocked port's remaining cost is the inter-chunk scan, which
is not the solve.

`unblocked BC=64` is the honest control on the *batching* claim alone: it
batches the column solves into one 64x64 block with `NC=1` and no blocking at
all. It still gets 72x fewer kernels than naive, but only 17.8x less time and
4x worse than `BC=16`. **So the two effects are real and separable: batching
alone recovers most of the kernel-count win (12,090 -> 167), blocking on top
recovers most of the time win (167 -> 89 kernels, 46.5 -> 11.8 ms), because
blocking is what turns 63 scalar-producing reduce steps into 3 matmuls.**

### Measurement protocol, because the first attempt was wrong

The first protocol ran the graph once per fresh subprocess and reported the
`DEBUG=2` total. It gave naive 592.8, then 1,150.1, then 1,322.8 ms for three
identical runs — a 2.2x spread — and the `BC=8` vs `BC=16` ordering flipped
between runs. Quoting any single one of those would have been quoting a noise
sample. The protocol above instead realises the graph 3x inside one child and
reports each realisation separately, which is possible because the kernel count
is exact and identical every time. Under it:

* every blocked variant is stable to **±2%**;
* the naive variant is the only unstable one, 147.5 – 298.4 ms, a 2x spread.

That instability is itself a result: 6,021 back-to-back kernel launches is enough
that the time depends on machine state. It is also why I lead with the kernel
count, which is exact and identical on every run.

Wall-clock is deliberately **not** in these tables. The naive port compiles
12,090 kernel signatures on its first call, so a wall-clock comparison of the
two ports is a benchmark of TinyGrad's compiler, not of this operator. It was
50,000 – 93,000 ms for naive against 600 – 1,600 ms for blocked across runs, and
the spread is the compiler's.

---

## 3. Correctness

Oracle: `kda/ref.py::kda_recurrent`, the token-at-a-time transcription of Eq. 1,
float64. The port is float32 throughout, so the gap is purely float32 rounding.
`kda/test_tiny.py`'s tolerance is `1e-5` relative.

| check | worst relerr (CPU) | worst relerr (Metal) |
|---|---|---|
| matches the recurrence, all 9 shapes of `test_tiny.py` | 1.381e-06 | 6.096e-06 |
| chunk-size invariance, `C = 1,2,4,8,16,32` | pass | pass |
| nonzero initial state, 5 shapes | 5.33e-07 | 9.24e-07 |
| degenerate gates (`g=0`, `beta=0`, `beta=1`, `g=-30`) | 7.88e-07 | 2.01e-06 |
| causality (no future leakage) | exact, 0.0 | 0.0 |
| block-width sweep, every `BC` dividing `C` | 5.73e-07 | 2.22e-06 |
| **`Dq = Dl + eye` is bit-exact** | pass | pass |
| agrees with the naive port | < 1e-4 | < 1e-4 |

**9/9 pass on both devices. Worst relative error: 1.381e-06 on CPU, 6.096e-06 on
Metal.**

The naive control, measured over the *same* ground, is 1.106e-06 (CPU) /
6.643e-06 (Metal). So the blocked port is neither more nor less accurate than
the control; the two are within a factor of 1.25 on identical inputs against an
identical oracle, which is what two float32 evaluations of one operator should
look like. (The brief quotes 8.6e-06 for the existing port. I do not reproduce
that over `test_tiny.py`'s ground — `test_tiny.py` itself reports 6.64e-06, and
6.643e-06 is what I measure. Whatever produced 8.6e-06, it was not this.)

Two checks the existing suite does not have, both of which the implementation
depends on:

* **Every `BC` dividing `C` gives the same answer** (`BC ∈ {1,2,4,8,16,32,64}` at
  `C ∈ {16,32,64}`). The block width is a performance knob, not a semantic one,
  and that is only true if the placement in `(*)` is right.
* **`Dq = Dl + eye` is bit-exact, and `Dl`'s diagonal is bit-exactly 0.** `Dq`
  is the causal-including-diagonal decay and `Dl` the strictly-lower one; they
  differ only on the diagonal, so `Dq = Dl + eye` halves the `exp` count on the
  single largest tensor in the operator. It is valid only because `Dl`'s
  diagonal underflows to **exactly** `0.0`, which the naive port already relies
  on for its own masking. Verified: `max|diag(Dl)| == 0.0` exactly, and
  `max|Dq - (Dl + eye)| == 0.0` exactly. If a backend's `exp(-1e4)` ever
  returned a denormal instead of `0`, this would silently double the self-decay
  term, and this is the check that catches it.

---

## 4. Two negative results

### 4.1 The reference's BC=16 blocking of the *gram* should not be copied

`kda/SPEC.md` §4.4 says the same `BC=16` blocking is reused for the decayed k-vs-k
gram. It is — and in TinyGrad that is a pessimisation. The port's form is one
`[C,C,K]` decay tensor, masked, two multiplies and one reduction over `K`. The
reference's form is `NC(NC+1)/2 = 10` blocks of `[BC,BC]`, each against a
per-sub-chunk reference `b_gn`. Measured, **gram only**,
`B=2 NT=8 HV=4 C=64 K=64`, Metal, steady state:

| form | DEP | uops | kernels | device ms | Python build ms |
|---|---|---|---|---|---|
| port: `[C,C,K]` + one reduce | 9 | 11 | 2 | 3.2 | 1 |
| reference: 10 blocked matmuls | 19 | 99 | 2 | 13.1 | 110 |

The two agree to 5.7e-06 (float32). The blocked form is **9x the uops and 4.1x
the device time** for the same matrix.

The reason is `matmul/RESULTS.md`: TinyGrad 0.14.0 has no `MATMUL` uop, no
matmul lowering pass, and calls no vendor GEMM — `Tensor.dot` is
`(x*w).sum(-1)`, a broadcast-multiply plus a reduction
(`.venv/.../tinygrad/mixin/op.py:392`, which I read rather than took from the
note). A "matmul decomposition" is a **tensor-core** decomposition, and this is
an fp32 port on a backend with no tensor cores for fp32. There is nothing for
the decomposition to buy, and 10 block matmuls plus 10 block writes cost more
than one fused reduce. Both forms land in 2 kernels — the `exp` was already
being fused away, see `kda/materialisation_verdict.md` §2.3 — so the split buys
no fusion either.

**Copy the reference's blocking of the solve, not its blocking of the gram.**
Worth stating because "follow the kernel" is the right instinct for the solve
and the wrong one here.

### 4.2 The reference's gram form narrows the operator's valid input domain

The per-sub-chunk reference `b_gn` makes the second factor `exp(gn_j − gc_j[p])`
span `[0, BC·max|g|]` rather than `[0, C·max|g|]`. That is the point of `BC=16`
secondary chunking, and it is a real constraint: the factor is `>= 1`, it grows
with `|g|`, and it has to stay finite in fp32.

`BC=16`, fp32 max `3.4e38 = exp(88.7)`:

| `max|g|` | `exp(BC·|g|)` | |
|---|---|---|
| 0.7 | `exp(11.2)` | ok |
| 5.0 | `exp(80.0)` | ok |
| 12.0 | `exp(192.0)` | **overflow** |
| 30.0 | `exp(480.0)` | **overflow** |

Threshold: `|g| <~ 5.5`. Measured end to end, at `g = -30` — a case
`kda/test_tiny.py::test_tiny_degenerate_gates` *requires the port to get right*
— the reference's gram form produces non-finite output and the port's difference
form does not.

`kda/SPEC.md` §4 (line 237 as of my read) says fla clamps `lower_bound ∈ [−5, 0)`.
That is not a modelling preference: it is this `BC·|g| < 88.7` fp32 range
limit. `BC = 16` buys 4x the headroom of `C = 64`, and 5.5 is where it runs
out.

### 4.3 My own model of the best `BC` was wrong, and the measurement beat it

I predicted `BC ≈ sqrt(C)` — i.e. `BC = 8` at `C = 64` — from the depth of a
single-level block sweep, `(BC-2) + (NC-1)`, minimised at `BC=8`. Wrong.
Solve alone, same controlled protocol:

| `BC` | `NC` | DEP | uops | reduces | kernels | device ms (steady range) |
|---|---|---|---|---|---|---|
| 4 | 16 | 149 | 149 | 48 | 42 | 4.50 – 4.55 |
| 8 | 8 | 93 | 93 | 28 | 26 | 1.54 – 1.56 |
| **16** | **4** | **89** | **89** | **24** | **24** | **1.10 – 1.18** |
| 32 | 2 | 135 | 135 | 34 | 35 | 2.49 – 2.53 |
| 64 | 1 | 254 | 254 | 63 | 63 | 17.96 – 18.12 |
| 2 | 32 | 285 | 285 | 94 | 80 | 14.23 – 14.88 |

`BC=16` wins on every exact metric and the ranges do not overlap with `BC=8`'s.
The model was wrong because it priced a block level as one matmul; a level is
actually ~3 dependent uops (batched matmul, contraction over `k`, subtract,
matmul, accumulate) against the in-block step's ~4 uops for a `BC` times shorter
recurrence. `BC=8` is 4 uops worse on *work*, not just on depth.

The one place the time ordering disagrees with the uop ordering is `BC=2` versus
`BC=64` — 285 uops but 80 kernels runs in 14.4 ms, while 254 uops and 63
kernels takes 18.1 ms. Launch count and arithmetic are not the same axis, which
is the same lesson as the naive port's 6,021-versus-20,354 from the other side.

For the *full operator* the `BC=8` and `BC=16` device times do overlap
(9.76–13.53 versus 10.88–12.68), so at operator level the choice is inside the
noise; the exact structural metrics still put `BC=16` ahead, and the isolated
solve measurement separates them cleanly. `default_block()` returns 16, and its
comment says the measurement rather than a model.

---

## 5. What I did not fix

- **The inter-chunk scan is untouched, and it is now the dominant cost.** It is
  genuinely sequential: chunk `n`'s output is chunk `n+1`'s input. After
  blocking it is 68 of the 157-step critical path and most of the remaining 89
  kernels. I removed everything *inside* it that did not depend on the carried
  state — `Aqk` for all chunks at once (a `[B,HV,C,C,K]` elementwise-and-reduce,
  the largest single piece of per-step work in the loop, recomputed on every one
  of the `NT` dependent steps), `exp(g_last − g_cum[n])` for all chunks at once,
  and `gam[n]·q[n]` for all chunks at once — but the `NT` matmuls and the state
  update are irreducible without a different algorithm. A sequential scan is
  also the one thing here a compiler cannot fuse, so it is the honest place for
  a compiler-versus-human gap to live.
- **No tile sizes, no `Context`, no scheduling flags, no `custom_kernel`, no
  PTX**, per the brief. Everything above is hoisting, batching, blocking and
  algorithm choice, all in ordinary TinyGrad.
- **`BC = 1` is supported but is not a good idea.** `NC = 64` gives 285 uops and
  80 kernels, worse than `NC = 1`. It is in the sweep only because "every `BC`
  dividing `C` gives the same answer" is a claim worth being able to test.
- **I did not build a recursive (multi-level) block decomposition.** Applying
  `(*)` to the diagonal blocks again would give depth `log2(NC)` instead of
  `BC-2 + NC-1` — about 6 levels instead of 17 at `C=64` — at the cost of more
  total arithmetic. I did not measure it, so I have no number and am not going
  to quote one. It is the obvious next thing to try and the obvious place a
  compiler might eventually win.
- **`kda/tiny_kda.py` is untouched**, as the control. Its comment at 155-158
  still claims unrolled forward substitution is what the kernel does, which §1
  contradicts; I could not fix it because I do not own that file.
- **`kda/ref.py:277-287`'s docstring is still wrong.** Same reason. It is a
  one-line fix (`(I - D)^{-1}`), it changes no behaviour, and it is exactly the
  kind of thing that produces a second wrong port.

---

## 6. TinyGrad-specific traps, for whoever does this next

Every one of these cost time and none of them raised a useful error. The value
of this note is mostly this list.

1. **A size-1 axis in broadcasting REPEATS; it does not pad.** Seeding an
   accumulator with a `[N, 1, ...]` tensor and then adding a later block row
   gives `row0 + row1` in that block row, not `row1`. The seed needs the full
   axis from the start. This is the same "a length-1 axis broadcasts against
   anything" trap `tiny_kda.py:139-142` warns about, in the opposite direction.
2. **The one-hot selector picks the block-row axis, and a per-row result has no
   block-row axis.** `sel.reshape(1, NC, 1, 1) * Xi` with `Xi` shaped
   `[N, 1, BC, NC*BC]` is right; drop the reshape on `Xi` and the multiply
   still succeeds **whenever `NC == BC`**, quietly selecting the wrong axis. At
   `C=4, BC=2` that is a relative error of 1.14 with nothing raised anywhere.
3. **A stack of `None`s on a 2-D array can make it six-dimensional.**
   `np.eye(NC)[None, None, None, :, None]` is shape `(1,1,1,NC,1,NC)`, not
   `(1,1,1,NC,1)`. The broadcast then lands on the wrong axis and nothing
   raises. Use `reshape` for every mask.
4. **A singleton at different positions in two operands silently transposes the
   result.** `(Gi[:, :, :, None, :, :] * Fj[:, :, :, :, None, :])` produces
   `(..., p, b)`; the operand order is what decides which index lands where.
   numpy and TinyGrad agree on the wrong answer, so no framework catches it for
   you. This cost the most time of anything in this note, and it produces a
   *plausible* number rather than an error.
5. **`Tensor.dot` will not invent a batch axis.** A 4-D left operand against a
   3-D right one fails on shapes rather than broadcasting. Keep the ranks equal
   and it is a non-issue.
6. **A chain of small ops does not stay in the graph.** This is the headline,
   and it is the opposite of what `kda/materialisation_verdict.md` found at a
   small shape: the whole operator there fused into **one** kernel. At the bench
   shape the UT transform emits 6,021. Fusion depends on the size of the thing
   being fused, and a 20,354-op chain of 64-float tensors is past the point
   where it pays. **Do not assume a small capture tells you what a large one
   does** — that is the transferability warning in this note.
7. **An indexed write-back (`t[..., i, :i] += x`) is not an extra kernel.** It is
   what the in-block substitution uses, and the whole solve still realises to 24
   kernels, under 2 per level. Worth knowing before replacing it with an
   outer-product trick.
8. **The block-diagonal extraction is a mask-and-contract, not a gather.**
   `Ab[:, idx, :, idx, :]` with two separated index arrays is exactly the kind
   of expression that returns its axes in an order you did not intend. Masking
   the two block axes and contracting the column-block axis is one line, has no
   such failure mode, and is what this port does.

---

## 7. How to reproduce each number

The scripts that produced this note were scratch (outside the deliverable, in
`.agent-scratch/kda-ut-opt/`, and deleted when the task closed). What each
number needs, so nothing here has to be taken on trust:

| number | how to get it |
|---|---|
| kernel count | run the graph under `DEBUG=2` on the real device and count the `***` lines that are not `copy`. Both outputs must be realised; `kda_chunk` returns numpy, so one call realises both |
| arithmetic uops, reduce count, longest chain | walk the realised `UOp` DAG from the output tensor (`tensor.uop.toposort()`). Count uops that are in `GroupOp.Elementwise ∪ Unary ∪ Binary ∪ Ternary ∪ Comparison` or are `Ops.REDUCE`, and are not movement (`GroupOp.Movement`) and not `is_virtual`. Movement and index arithmetic pass a depth through without costing a step; counting them makes a slice look as expensive as a matmul |
| device time, steady state | realise the graph 3+ times **inside one process**, and sum only the last realisation's `DEBUG=2` lines. The kernel count is exact and identical every time, so `len(lines)//iters` gives the per-realisation count and the last chunk is the one to time. This matters: a fresh subprocess per run re-pays compilation of every kernel signature and produced a 2.2x spread across identical repeats (592.8, 1,150.1, 1,322.8 ms) with the `BC` ordering flipping between runs |
| graph of a non-realising build | wrap `Tensor.numpy` to capture the tensors and return a zero array of the right shape, so `kda_chunk`'s graph can be inspected after it has "returned". Re-implementing `kda_chunk` instead would risk measuring a different graph from the one that runs |
| the two gram forms | the reference's is 10 `[BC,BC]` blocks against a per-sub-chunk reference, placed into a `[C,C]` matrix; compare `uops` and device time against the port's single masked reduce. The two must agree to float32 before either timing means anything |
| `exp` overflow threshold | `BC·max|g| < ln(3.4e38) = 88.7`; table in §4.2 |
| the sign of `_forward_substitution` | `np.linalg.inv(np.eye(n) ± A0)` against the routine, both signs; code in §1 |

The operator is `kda/tiny_kda_opt.py`; the control it is measured against is
`kda/tiny_kda.py`, unmodified.

