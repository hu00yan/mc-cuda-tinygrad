---
kind: reference
---

# 31 — what the KDA UT transform costs in TinyGrad, candidate by candidate

**Why this file exists.** `notes/25` established what the reference kernel
actually does to the UT transform: a two-level blocked inverse at `BC=16`, four
14-step diagonal row-loops with zero `tl.dot`, then a block-Schur merge of 16
`tl.dot`s in a depth-3 DAG. `kda/tiny_kda.py:178-194` does the naive thing
instead. `notes/26` measured what TinyGrad charges for and, in its section 10,
explicitly declined to price the alternatives:

> Whether a wide-output reformulation of the KDA UT transform is possible at
> all. … A blocked or doubling formulation in pure matmuls — forward-substitution
> doubling, `C/2` independent half-solves composed with a `[C/2, C/2]`
> correction — is the remaining candidate, and its FLOP count, its kernel count,
> and whether it is numerically the same operator are all **not settled here.**

This file settles them. It is a **price list, not a port**: five candidate
expressions, a number for each, and no maintained implementation. Two agents own
`kda/tiny_kda_opt.py` and `kda/tiny_kda_faithful.py`; nothing here is intended
to land in either.

**Instrument:** `kda/ut_price.py`, re-derived here and validated before use
(§1). Every number below is `MEASURED` on this host with no GPU, or
`RE-DERIVED` from the algorithm in code, or `INFERRED` and labelled. **No figure
is transcribed from another note** — the figures quoted in the brief were used
only to know what to check against, and two of them turned out to be wrong or
self-contradictory (§6, §8).

```
.venv/bin/python kda/ut_price.py --selftest      # instrument validation
.venv/bin/python kda/ut_price.py --verify        # all five compute the same thing
.venv/bin/python kda/ut_price.py --slabs 64      # the price list, at the KDA shape
.venv/bin/python kda/ut_price.py --spelling both # both constant spellings
.venv/bin/python kda/ut_price.py --e             # candidate E
.venv/bin/python kda/ut_price.py --tc            # tensor cores, these shapes
.venv/bin/python kda/ut_price.py --ladder        # stage attribution
```

---

## 1. The instrument, and the check that it is not broken

`notes/26` records that a *structural copy* of the mine.cu DDA disagreed with
the real port (217 vs 48 kernels), and that an `nb²` penalty was described where
the measurement is linear. A census that does not reproduce a known figure is a
broken census, so this one is checked against two, before any number in §3 was
taken.

The instrument hooks `tinygrad.schedule.memory_plan_rewrite`
(`schedule/memory.py:20`, called from `schedule/__init__.py:209`) — the last
point at which the LINEAR still holds one UOp per logical buffer. It never
realizes, and `Context(DEV="CUDA")` makes `Device.DEFAULT` literally read
`"CUDA"` (`device.py:44`), so the real CUDA scheduler runs on a machine with no
CUDA device. That is the same argument `kda/schedule_census_cuda.py:20-27`
makes, re-derived rather than assumed.

Two new quantities beyond the usual census:

- **`dag_depth`** — the longest chain of *dependent kernel launches*. A step's
  depth is 1 + the max depth of any step whose buffer it reads. In a LINEAR,
  `si.src[1]` is the buffer a step **writes** and is usually a `BITCAST` over
  it, not a `BUFFER`; `si.src[2:]` are what it reads. Getting that backwards
  makes every graph report a depth of 1, which is what a first cut of this
  instrument did.
- **`max_out_bytes`** — the widest buffer a real kernel *writes*, as distinct
  from the widest kernel *argument*, which is dominated by the input.

### Self-test 1 — the single-column chain of `notes/26` §3b

`one_column_chain` is that chain, at `[1,1,1,C,C]`, op for op.

| C | steps | kernels | REDUCE | REDUCE expected | alu_lane |
|---|---|---|---|---|---|
| 4 | 3 | 5 | 2 | 2 | 55 |
| 8 | 7 | 11 | 6 | 6 | 259 |
| 16 | 15 | 21 | 14 | 14 | 1,147 |
| 32 | 31 | 43 | 30 | 30 | 4,843 |
| 64 | 63 | 85 | **62** | **62** | 19,915 |

The REDUCE column is `C − 2`, matching `notes/26`'s table at every C, and the
kernel count is its figure plus the COPYs the symbolic masks introduce (§2).

### Self-test 2 — the port's own loop, at one slab

Candidate `A` is `kda/tiny_kda.py:161-194` copied op for op. It must be exactly
`C ×` the single chain, because it is `C` independent copies of it.

| C | kernels | REDUCE | REDUCE expected = C × (C−2) |
|---|---|---|---|
| 8 | 76 | 48 | 48 |
| 16 | 324 | 224 | 224 |
| 32 | 1,316 | 960 | 960 |
| 64 | 5,380 | **3,968** | **3,968** |

`3,968 = 64 × 62`. The number `notes/13` attributes to this loop is `C(C−1) =
4,032`; the census of the same loop gives **3,968 dependent scalar reductions**
and **5,380 kernel launches**. 4,032 is the count of multiply-adds in the inner
loop, not the count of scheduled reductions, and the two are not the same thing.

### Self-test 3 — no instrument knob is set

```
ALLOW_TF32=0  TC=1  TC_SELECT=-1  TC_OPT=0  NOOPT=0  IMAGE=0  NOLOCALS=0  BEAM=0  SPLIT_REDUCEOP=1
```

Printed by `--selftest`, read out of `tinygrad.helpers` at runtime. This study
uses no hardware hints, so a knob being on would invalidate every row below.

**`SELFTEST PASS`.**

---

## 2. One fork that had to be settled before anything else: the constant spelling

Every candidate needs index masks (`col < i`, `row == i`, one-hot block
selectors) and an identity. There are two ways to write them and they cost
opposite things:

| spelling | what it is | cost |
|---|---|---|
| `numpy` | `Tensor(np.eye(C))`, `Tensor((arange<i))` | folded into the consumer as a literal CONST, so the constant is free — but every *distinct* constant is its own device buffer and gets its own COPY kernel |
| `arange` | `Tensor.arange(n) == k` | no buffer, no COPY — but the comparison is re-evaluated inside every consumer, so ALU-lane ops go up |

At the KDA shape, `arange` buys 5,380 → 5,380 kernels on `A` (−12%) and 22.1M
vs 12.1M lane ops on `C` (+83%). **Neither is right without knowing what a kernel
launch costs on the card, which this host cannot measure** — so both are
reported and neither is the headline. What matters is that **the ranking of the
five candidates is identical under both spellings**, by kernels exactly and by
lane ops up to a C/D2 swap:

```
  arange  by alu_lane: D < D2 < C < B < A
  arange  by kernels : D2 < C < B < D < A
  numpy   by alu_lane: D < C < D2 < B < A
  numpy   by kernels : D2 < C < B < D < A
```

The tables in §3 use `arange` (fewer launches) and the §3.1 table gives both.

---

## 3. The price list

**Shape.** `B*NT*HV = 2 × 8 × 4 = 64` independent `[64,64]` unit
lower-triangular systems — `kda/perf_kda.py:90`'s bench shape with `NT = T/C = 8`
and `C = 64` (`kda/tiny_kda.py:91`). Every candidate computes the same object:
`(I + S)^{-1} · Diag(β)` with `S = StrictTril(Diag(β) Z)`, which is the paper's
`M` and is what `kda/tiny_kda.py:194` produces. The trailing `· Diag(β)` is
identical on all five and cannot move the ranking.

**Correctness first.** `--verify` runs all five on CPU against
`numpy.linalg.inv` in float64, at 2 slabs, `C=64`:

| cand | max rel err vs float64 inverse |
|---|---|
| A | 3.108e-08 |
| B | 3.108e-08 |
| C | 3.108e-08 |
| D | 3.108e-08 |
| D2 | 3.108e-08 |

All five are the same operator to fp32 round-off, and all five have the *same*
error as the port. A candidate that computed a different thing is not a
candidate, so this runs before the prices do. (`notes/25` §6 reaches 2.08e-17
in float64 for the same three ways; this is fp32, hence 3e-8.)

### 3.1 The table

`MEASURED`, 64 slabs, `DEV=CUDA`, no device, `arange` spelling unless stated.

| cand | what it is | kernels | of which COPY | REDUCE | alu_lane | dag_depth | widest written | arg bytes vs A |
|---|---|---|---|---|---|---|---|---|
| **A** | naive 64 × 63 unrolled column solve — the port today | **5,380** | 67 | **3,968** | 192,625,524 | **85** | 16,384 B | 1.000× |
| **B** | the same 64 solves, batched on the leading axis | **86** | 2 | 62 | 79,888,114 | **85** | 16,384 B | 0.036× |
| **C** | `BC=16` two-level blocked, as the kernel does it | **44** | 11 | 20 | 22,081,474 | **34** | 16,384 B | 0.012× |
| **D** | pure-matmul block doubling, 64→32→…→2 | 253 | 129 | 62 | **2,737,604** | **16** | 16,384 B | 0.016× |
| **D2** | Neumann on the block-off-diagonal, 6 `[64,64]` matmuls | **41** | 6 | 22 | 17,731,413 | 34 | 16,384 B | 0.013× |

`numpy` spelling, same shape:

| cand | kernels | REDUCE | alu_lane | dag_depth |
|---|---|---|---|---|
| A | 6,085 | 3,968 | 81,595,648 | 96 |
| B | 98 | 62 | 48,086,415 | 96 |
| C | 71 | 20 | 12,080,539 | 36 |
| D | 286 | 62 | 2,747,747 | 17 |
| D2 | 64 | 22 | 29,096,550 | 36 |

### 3.2 Algorithmic axes, `RE-DERIVED` in code

These are properties of the *algorithm*, computed in `ut_price.py:macs` and
`ut_price.py:chain_depth`, not of the schedule. They are what tells you whether
a cheaper schedule is doing less work or the same work with more overhead.

| cand | dependent steps in the data | multiply-adds per `[64,64]` system | vs C |
|---|---|---|---|
| A | 63 | 391,168 | 5.93× |
| B | 63 | 391,168 | 5.93× |
| **C** | **17** (14 substitution + 3 merge) | **66,012** | 1.00× |
| D | 6 | 74,896 | 1.13× |
| D2 | 20 | 1,573,340 | **23.83×** |

`A` and `B`: `C` column solves, each summing `r` terms for `r = 1..C−1`, i.e.
`C·Σr = 129,024`, plus `C` outer products of `C×C` to scatter each solved vector
into `M` — 262,144, and the scatter is 2/3 of the arithmetic.
`C`: four 16×16 inverses, rows 2..15 → `4·Σ_{i=2..15} i = 476`, plus 16
`[16,16]³` merge products = 65,536.
`D`: two `[n,n]³` products per level at `n = 32,16,8,4,2`.
`D2`: C's diagonal stage plus 6 full `[64,64]³` products.

**`D2` does 23.8× the multiply-adds of the kernel-faithful form and still has
fewer launches than it.** That is the whole story of `D2` in one line, and it is
why §5 does not pick it.

### 3.3 Where the widest intermediate goes: nowhere

`max_out_bytes` is **16,384 B for all five** — one `[64,64]` fp32 matrix, which
is both the input and the output. The UT transform at this shape is
*bandwidth-trivial*: it is a launch-count and dependency-depth problem, not a
memory problem, and no candidate can win on bytes. `notes/26`'s output-width
rule is what makes `A` and `B` fuse at all, but it has nothing to discriminate
between `C`, `D` and `D2`.

### 3.4 Stage attribution

`MEASURED`, 1 slab, so the pieces are legible.

| what | kernels | REDUCE | alu_lane | dag_depth |
|---|---|---|---|---|
| A: full 64 × 63 substitution | 5,380 | 3,968 | 3,056,396 | 85 |
| A: one column of it (63 steps) | 86 | 62 | 1,267,902 | 85 |
| C: diagonal stage alone (4 × 14) | 30 | 14 | 260,832 | 28 |
| C: merge only, no assembly | 34 | 16 | 261,567 | 30 |
| C: full (diag + merge + assemble) | 44 | 20 | 350,499 | 34 |
| D: one doubling level (64→32) | 124 | 30 | 5,307 | 13 |
| D: full 5-level doubling | 253 | 62 | 43,454 | 16 |

Two things to read off this:

- **C's 14 substitution steps cost exactly 14 REDUCE kernels.** One per step, as
  the kernel has one per iteration. The chain is not being split more than it
  is written.
- **C's 16 merge products cost exactly 16 REDUCE kernels, and they do not
  lengthen the chain by 16.** `dag_depth` goes 28 → 34, i.e. +6, because
  `Tensor.matmul` is `(x*w).sum(-1)` (`mixin/op.py:392`) and a `[16,16]@[16,16]`
  has a **width-16** reduce output. That is `notes/26` §4's rule landing
  correctly for once: the merge's contractions have width, so they fuse, and
  the depth-3 DAG stays depth-3. A's inner contraction has a width-**1** output
  per slab and does not.

### 3.5 How much of the operator this is

`MEASURED`, whole `kda/tiny_kda.py:kda_chunk` at the same bench shape, same
instrument:

| | kernels | REDUCE | alu_lane |
|---|---|---|---|
| whole `kda_chunk` | **12,221** | **7,984** | 57,764,876 |
| UT transform alone, candidate A | 5,380 | 3,968 | 192,625,524 |

**The naive UT transform, censused on its own, is 44.0% of the operator's kernel
launches and 49.7% of its REDUCE kernels.** Swapping `A` for `C` would take the
operator to `12,221 − 5,380 + 44 = 6,885` launches and `7,984 − 3,968 + 20 =
4,036` REDUCE kernels — a 1.77× reduction in launches, from one operator.

**That arithmetic is an identity on the measured counts, not a measurement, and
it is known to be wrong in one of its terms.** The `alu_lane` column disproves
additivity outright: `A` alone (192.6M) exceeds the *whole* operator (57.8M).
A subgraph scheduled in isolation is not scheduled the same way it is scheduled
in context. So the launch identity is an estimate; the 44.0% / 49.7% ratios are
an indication of scale; and only the relative ordering of the five candidates —
all measured the same way, in isolation — is load-bearing. This is
`notes/26` §3c's warning in its most expensive form: **a structural copy's
census is not the port's census.** The right check is to re-run
`kda/schedule_census_cuda.py` on the rewritten `tiny_kda.py` after the rewrite
lands, which is why this note prices candidates and does not ship one.

---

## 4. What candidate E is, and the answer to its open question

Candidate E is not a reformulation of the *inverse*; it is a reformulation of one
of its **inputs**, and the brief asked whether the kernel's default path has a
natural TinyGrad expression. That is the sharpest question in the set, so it
gets a sharper instrument than a kernel count.

`notes/25` §8a: `safe_gate` defaults to `False`, so the default is
`chunk_kda_fwd_intra_token_parallel`, which builds each `[16,16]` diagonal block
with a `≤16`-iteration loop over `j` of rank-1 accumulates and **no `tl.dot`**
(`chunk_intra_token_parallel.py:117-124`), on a grid of `(B*T, cdiv(HV,BH))` —
one program per token. The `tl.dot` form at `chunk_intra.py:773` is the
`safe_gate=True` path. A port that assumes "16×16 blocks via matmul" has ported
the non-default path.

`MEASURED`, 512 tokens (`B=2, T=...`, 8 chunks), `K=64`, both spellings:

| form | kernels | REDUCE | alu_lane | dag_depth | generated kernel name | `__WMMA` |
|---|---|---|---|---|---|---|
| **E** — token-parallel, `≤16` rank-1 steps, no dot | **10** | **1** | 32,422,950 | **2** | `r_64_2_4_8_16_4_64_64_64_16_4` | **0** |
| **E_dot** — the non-default `tl.dot` form | **10** | **1** | 32,882,850 | **2** | `r_64_2_4_8_16_4_64_64_64_16_4` | **0** |

**The answer, and it is worth more than the counts: TinyGrad cannot tell the
kernel's default path from its non-default path.** Both spellings produce the
same number of kernels, the same single REDUCE, the same launch depth of 2, and
the *same generated kernel name*. The CUDA is not byte-identical (md5
`e2834ceb4e` vs `5f890c0d61`) but the diff is a reassociation — `k_i·k_j·exp2(g_i−g_j)`
versus `k_i·exp2(g_i−g_n)·k_j·exp2(g_n−g_j)`, the sub-chunk reference cancelling
as `notes/25` §5 describes — with the same accumulator structure, the same
tiling, and no `mma.sync` in either.

That is the expected consequence of `notes/26` §4 arriving one level up: the
kernel's choice between a rank-1 accumulate loop and a `tl.dot` is a choice
*inside* one contraction, and inside one contraction `Tensor.matmul` **is** the
rank-1 loop. There is no spelling in this framework that selects one over the
other, so a port does not have to make the choice and cannot be credited for it.

**Expressiveness result, stated in those words: the token-parallel shape has a
natural TinyGrad expression, it is one kernel of depth 2, and it is free of the
distinction the kernel draws.** The `≤16` j-iterations are lifted from a loop
into a shape — `[tokens, 16, K]` — and all `B·T` programs' worth of rank-1
accumulates become one contraction over `K`. Nothing about the token parallelism
is lost: it was never a control-flow property, it was a grid dimension, and
TinyGrad has grid dimensions.

`MEASURED` verification: both forms match a float64 numpy transcription of
`chunk_intra_token_parallel.py:123-124` to **2.86e-06** and **3.34e-06**
respectively, with the off-diagonal blocks exactly zero.

**The cost `E` does hide.** 10 kernels and depth 2, but 32.4M ALU-lane ops for
512 tokens — the `[tokens,16,K]` intermediate is 16× the key tensor, and unlike
the kernel's registers it is a graph node that a schedule census cannot see the
size of until the fusion pass decides. At `K=64` and 16 j's, `E` is 16× the
arithmetic of the `tl.dot` form and gets the same schedule, which is only
possible because the extra arithmetic was never in the kernel to begin with: the
kernel's loop over `j` is 16 iterations of a `[K]` accumulate, and its grid is
`B·T` programs, so its total work is `B·T × 16 × K` — exactly what `E` computes.
It is not more work; it is the same work with 16× the register pressure per
program and no dependence on the compiler noticing.

---

## 5. The ranking, and the single dominant change

**By measured cost** (`arange` spelling, KDA shape):

| rank | by launches | by ALU-lane | by launch depth | by arithmetic |
|---|---|---|---|---|
| 1 | **D2** (41) | **D** (2.74M) | **D** (16) | **C** (66k) |
| 2 | C (44) | D2 (17.7M) | C, D2 (34) | D (74.9k) |
| 3 | B (86) | C (22.1M) | — | A, B (391k) |
| 4 | D (253) | B (79.9M) | A, B (85) | D2 (1.57M) |
| 5 | A (5,380) | A (192.6M) | | |

**The single dominant change is: take the scalar out of the contraction's
output.** Concretely, the number of dependent steps whose contraction reduces to
one scalar per `(batch, chunk, head)` goes **63 → 14**, and every other
difference in the table is a second-order consequence of that.

The decomposition, from the ladder:

1. **`A` → `B` is 5,380 → 86 launches, a 62.6× reduction, for *zero* change in
   the algorithm.** The 64 column solves are 64 independent linear systems with
   the same matrix; batching them onto a leading axis changes nothing about the
   work, the depth (85 → 85) or the arithmetic (391,168 MACs either way), and
   removes 99.2% of the launches. This is `notes/26` Rule 1 — never loop over a
   batch, put the batch in the leading axis — applied to an inner loop the
   survey did not know existed. It is the largest single effect in the whole
   price list and **it is not a change of algorithm**.
2. **`B` → `C` is 86 → 44 launches, 79.9M → 22.1M lane ops, and depth 85 → 34.**
   This is the algorithmic change the reference makes: block at `BC=16` so the
   sequential chain is 14 deep instead of 63, and pay for the remaining six
   off-diagonal blocks with products.
3. **`C` → `D` removes the substitution entirely (14 → 0 rank-1 reduces) and
   buys 8.1× fewer lane ops — but costs 5.7× the launches and 1.13× the
   arithmetic.** It is a win on one axis and a loss on two.
4. **`D` → `D2` buys 3 launches and pays 24× the arithmetic.** Not a trade worth
   making.

So the change that accounts for most of the difference is **not** "use a matmul
where the naive form used a loop". It is **"stop reducing to a scalar"** — first
by batching (§1 above, which is free), then by shortening the remaining chain
from 63 to 14. The matmul reformulations, which are what the brief anticipated
would matter, are the *third* and *fourth* items on this list and the second one
is a loss.

---

## 6. Faithful versus cheap: they do not disagree, and that is the finding

The brief asked me to name the one candidate that is both cheap and faithful,
and to say so plainly if the two criteria disagree. **They do not disagree, and
the reason is specific.**

- **The reference does only `C`.** `chunk_intra.py:274-313` is four 14-step
  row-loops with no `tl.dot`, and `:320-356` is 16 `[16,16]³` products in a
  depth-3 DAG. `notes/25` §2b says explicitly that a porter "should not
  'optimise' this into a matmul inside the kernel's model". `D` is 100% pure
  matmul with no substitution anywhere — **not what the kernel does**.
  `D2` is the Neumann series that `notes/25` §8i records as "an alternative the
  kernel does *not* use". `A` and `B` are `naive.py`'s algorithm: the kernel's
  substitution is row-wise and it never solves columns, so batching the column
  solves is on the *other side* of the line from the kernel too, even though it
  changes nothing about the graph.
- **`C` is 2nd of 5 on launches (44 vs D2's 41), 3rd on lane ops (22.1M vs D's
  2.74M), 2nd on depth (34 vs D's 16), and 1st on arithmetic (66,012).** It is
  the cheapest candidate on the one axis where the naive port is catastrophic,
  it is the cheapest of all on arithmetic, and it is within 7% of the best on
  launches.

**The faithful candidate is also essentially the cheapest one, and that is not
the outcome the framing expected.** The reason is in §4 and §3.4: the merge's
contractions have *width*, so they lower to 16 reduce kernels that fuse into a
depth-3 DAG, and the substitution's 14 steps lower to exactly 14 reduce kernels.
`C` is 44 launches because there is nothing left in the graph to launch. `D` is
253 because five levels of `Tensor.cat` assembly are *not* free in a framework
with no register-resident tile.

Where the two criteria *do* pull apart, it is in the opposite direction from the
one the brief anticipated: the candidate that is most expensive on paper (`D2`,
23.8× the arithmetic) is the one with the fewest launches. **So the danger is not
that a cheap reformulation is unfaithful — it is that an unfaithful reformulation
can look cheap on the metric this framework is measured by.** `D2` is 41 launches
and 17.7M lane ops and 1.57M multiply-adds; `C` is 44 launches and 22.1M lane
ops and 66,012 multiply-adds. If launches are what the T4 spends its time on,
`D2` wins by 7%. If multiply-adds are, `D2` loses by 24×. Pricing that needs a
card, and it is in §8.

**The recommendation, with the reasoning attached:** ship `C`. It is the
reference's algorithm, it is the cheapest on arithmetic, and it is close enough
to the cheapest on launches that a launch-dominated regime does not overturn the
choice. A study that compared `D` against the kernel would be measuring
TinyGrad's codegen on an algorithm the kernel never runs, which is ledger
**C-14** and **C-15** again with a different subject.

---

## 7. Tensor cores, at the shapes these candidates actually use

`notes/26` §6b and §6c are re-measured here, at the merge's `[16,16]` and at
`[64,64]`, through `matmul/tg_backend.py` unmodified (`sm_80`, no GPU).

**fp32 gets no tensor core at any shape** — `__WMMA` count 0 at every one of
`2³, 4³, 8³, 16³, 16×16×64, 32³, 64×64×16, 64×64×32, 64³, 256³`, and at every
batched shape `[nb,16,16]@[nb,16,16]` and `[nb,64,64]@[nb,64,64]`. This confirms
§6b and is not in doubt.

**fp16 does reach one at the merge's own shape, and `notes/26` §6c's prose is
wrong about this.** §6c's own table says `N≥32 → 17` at `K=64` and then
concludes "a `[64,64] @ [64,64]` fp16 matmul is **not** a tensor-core kernel" —
its table contradicts its sentence. Measured: `[64,64]@[64,64]` fp16 is **17
`__WMMA`**, a tensor-core kernel. And the shape that actually matters here,
`[16,16]@[16,16]`, is **3 `__WMMA`** — also a tensor core.

§6c's `M ≥ 64` rule is an artefact of sweeping only at `K=64`. Re-swept at the
merge's own `N=K=16` (`__WMMA` count, fp16):

| sweep | result |
|---|---|
| N at M=16, K=16 | N=4 → 0, N=8 → **2**, N=16 → **3**, N=32 → 5, N=64 → 5 |
| M at N=16, K=16 | M=4 → 0, M=8 → 0, M=16 → **3**, M=32 → 5, M=64 → 9 |
| K at M=N=16 | K=4 → 0, K=8 → **3**, K=16 → 3, K=32 → 3, K=64 → 3 |

So at `N=K=16` the gate is **`M ≥ 16` and `K ≥ 8`**, not `M ≥ 64`. `[16,16]` is
inside the gate; `[8,8]`, `[4,4]` and `[2,2]` are not — which matters for `D`,
whose smallest doubling levels are `[2,2]`, `[4,4]` and `[8,8]`. One asymmetry to
carry: `bf16` at `[64,64]@[64,8]` is **0** `__WMMA` where fp16 is 17, so fp16 and
bf16 do not share a gate at small `K`.

**Why this matters for the comparison, and it is not a footnote.** The reference
kernel's merge runs in **tf32** — `chunk_intra.py:21-24` hardcodes
`SOLVE_TRIL_DOT_PRECISION = 'tf32'` on Ampere and later, with the comment "we
used tf32 to maintain matrix inverse's precision whenever possible", and
`notes/25` §8c flags it. TinyGrad's fp32 gets no tensor core at any shape unless
`ALLOW_TF32` is set (`postrange.py:226`), and this study uses no knobs. So:

> **A matmul-based candidate `C` is being priced in the FFMA regime and
> compared against a reference whose merge issues `mma.sync`. The measured
> launch and lane-op numbers in §3 are therefore a lower bound on what the
> reference spends and an upper bound on what the port spends, on the merge
> specifically.** The port is fp32 because `kda/tiny_kda.py:44` makes the whole
> operator fp32; moving the port to fp16 to get a tensor core is a study-design
> decision (`notes/26` §10 leaves it open) and is not made here.

---

## 8. What could not be priced, and what would price it

Each of these is a real gap, not a hedge. **None is filled with a guess.**

1. **Wall-clock, anywhere.** No GPU on this host (`nvidia-smi` absent,
   `Device.DEFAULT` is `METAL`). Everything above is a schedule census plus
   codegen capture. *What would price it:* run the rewritten port on the T4 and
   read the launch count out of the profile, then compare against the
   census's prediction. The census has been right about kernel counts on this
   repo's own terms; whether kernel count is what the T4 spends its time on at
   this shape is exactly the unverified link.

2. **Whether `C` or `D` wins, given that their lane-op ratio is 8.1× and their
   launch ratio is 0.17×.** These point opposite ways and the crossover depends
   on the ratio of launch overhead to lane throughput on the specific card.
   *What would price it:* one T4 run of each, same inputs, same seed, `ncu`
   launch counts and durations. That is one run each, not a study.

3. **Whether `D2`'s 3-launch advantage survives being 24× the arithmetic.** Same
   instrument-free dilemma as (2). *What would price it:* the same two runs. The
   arithmetic is already known exactly, so only the per-launch cost is missing.

4. **Register residency.** The kernel holds `b_Ai00/11/22/33` and twelve
   `[16,16]` fp32 accumulators in registers, and the six off-diagonal blocks in
   registers only, never written to memory (`notes/25` §5). A schedule census
   sees buffers and launch counts and is blind to per-thread register pressure,
   so the register-file cost of `C`'s 14-deep chain and of `E`'s `[tokens,16,K]`
   intermediate is **not** in any number here. *What would price it:* the
   generated SASS, which needs `ptxas` on a machine with the toolkit — the
   `matmul/` harness can already do this for a captured `.cu`, and the ported
   kernels are in `out/archlab/triton/`.

5. **The fusion boundary of the whole graph.** §3.5 shows the census is not
   additive: candidate `A` alone reports more ALU-lane ops than the entire
   operator. Every candidate was priced the same way, in isolation, which makes
   the *ranking* a measurement and the *absolute* numbers an upper bound.
   *What would price it:* rewrite `kda/tiny_kda.py` and run
   `kda/schedule_census_cuda.py` on the real file, which is the only method that
   `notes/26` §3c found trustworthy for this operator.

6. **Whether the fp16-vs-fp32 question is a port question or a study question.**
   §7 shows an fp32 port and a tf32 reference are comparing different instruction
   classes. *What would price it:* not a measurement — a decision, and it belongs
   to whoever owns the study design. Flagged, not taken.

---

## 9. One expressiveness sharp edge, found the hard way

`Tensor.expand(shape)` left-aligns its shape argument —
`mixin/movement.py:162`, `_align_left(self.shape, argfix(shape, *args))` — so
`x.reshape(T, BC, 1, K).expand(T, BC, BC, K)` expands the axis at position −2
while `x[:, None]` on the *same* tensor expands position −3. Both are "expand a
size-1 axis to `BC`", they disagree, and neither raises. In candidate `E` this
silently transposed the diagonal blocks and produced a result that was wrong by
a factor of 3.4 in absolute terms and still passed a shape check.

The fix is to write `x[:, None, :, :]` and `x[:, :, None, :]` rather than to
`expand` at all. This is a `READ-FROM-SOURCE` + `MEASURED` observation about the
installed 0.14.0, not a claim about TinyGrad in general, and it is not a
correction to `notes/26` — but it is the kind of thing that silently survives a
correctness gate that checks shapes and not values, which is the failure mode
ledger **C-15** describes.

---

## 10. What this note hands on

- **`kda/tiny_kda.py:157-158` should be corrected twice over.** It says
  unrolled forward substitution "is what the real kernel does", which
  `notes/25` §10 already refuted; this note adds that its *cost* is 5,380
  launches and 3,968 dependent scalar reductions, 44.0% of the operator's
  launches, and that `notes/13`'s 4,032 is the multiply-add count rather than
  the reduction count.
- **The faithful rewrite is `C`, and it is also nearly the cheapest.** Whoever
  owns `kda/tiny_kda_faithful.py` should implement `chunk_intra.py`'s schedule
  and can expect 44 launches and 20 REDUCE kernels at the bench shape, and
  should re-run the census on the real file rather than trusting §3.5's identity.
- **Whoever owns `kda/tiny_kda_opt.py` has a free 62.6× on the table before
  touching the algorithm.** Batching the 64 column solves onto a leading axis is
  `notes/26` Rule 1, it costs nothing, and it is the largest single effect in
  the price list. It is also *not* the kernel's algorithm, so it is a fallback,
  not a destination.
- **The gate from `notes/26` §10 is now closed.** A pure-matmul blocked
  reformulation is expressible (`D`, `D2`), it is numerically the same operator
  to 3.108e-08, and it is **not** cheaper on launches than the faithful form
  (253 and 41 vs 44). The FLOP count and the kernel count `notes/26` left open
  are in §3.1 and §3.2; the "is it numerically the same operator" half is
  answered yes, in fp32, verified in code.
- **The UT transform's widest intermediate is its own input and output, for every
  candidate.** Nobody should spend a rewrite on bytes here.
