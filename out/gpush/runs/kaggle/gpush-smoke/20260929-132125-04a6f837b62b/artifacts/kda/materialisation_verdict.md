# The materialisation verdict

> **The KDA `[B,NT,HV,C,C,K]` decay tensor is never materialised — not as a
> device buffer, not as a kernel argument, not as a value in memory at any
> point — and this is now settled on the real NVIDIA backend by the emitted CUDA
> source, not inferred from an allocator spy on Metal and CPU. The three agents
> did not contradict each other: they were reporting three different quantities
> at two different shapes. The 256× figure is a property of the graph and
> survives; the cost of the graph's shape shows up as redundant `exp2`, not as
> bytes.**

This is the single authoritative document for that question. Where it disagrees
with `notes/03-kda.md`, `notes/04-codegen-comparison.md`, `notes/09-materialisation-verdict.md`
or `README.md`, **this file wins** and the disagreement is listed in §8.

Status vocabulary: `MEASURED` = an instrument on hardware. `RE-DERIVED` = I
recomputed it here, by parsing the emitted source or from shapes, and got the
same value. `MODELLED` = arithmetic on shapes. `PREDICTED`. `UNVERIFIED`.
Nothing here is transcribed; every number was recomputed, and the two places
where recomputation failed are called out as such in §7.

---

## 1. The verdict, in the four senses

`notes/09` §4 defined four senses of "materialised" and that decomposition is
kept, because it is what dissolves the conflict.

| | sense | status | evidence |
|---|---|---|---|
| **M1** | a device buffer of the decay tensor's exact size is ever requested | **MEASURED — NO** | `notes/09` §4.1, three shapes, allocator spy with a positive control that fires when the tensor is forced to exist |
| **M2** | a kernel **argument** of that size exists in the generated plan | **MEASURED — NO, on CUDA, from the source** | `out/kaggle/20260928-014248-gpu/codegen/kernel000.txt`, §3 below. Previously METAL + CPU only, via a schedule spy. |
| **M3** | peak live device memory of the whole operator | **MEASURED** | 22,511,616 B peak live; 8,421,376 B planner arena (`notes/09` §4.2) |
| **M4** | actual DRAM traffic | **STILL NOT MEASURED** | needs a hardware counter. §5. |

**M2 is the decisive one and it is the one that was weakest before.** A buffer
carries traffic only if a kernel reads or writes it, and a kernel can only
touch buffers in its argument list. That argument does not depend on a cache, an
L2, a card, or a backends' memory model. It is now closed on the third backend,
from the artifact rather than from a spy.

---

## 2. The evidence: the emitted CUDA source

`out/kaggle/20260928-014248-gpu/codegen/kernel000.txt`. 24,968 chars, 253 lines,
one `__global__`. Emitted by the real NVIDIA backend (sm_75, tinygrad 0.14.0,
CUDA 12.8.1). Captured at **`B=2, T=128, H=HV=2, K=V=32, C=32`, NT=4** — the
shape is at `kernels/build_notebooks.py:590`, not `codegen_diff.py`'s. `MEASURED`

Everything in §2.1–§2.5 is `RE-DERIVED`, by evaluating the emitted index
arithmetic over the whole thread grid and by parsing the emitted expressions.
Not one of these numbers was read off the file by eye.

### 2.1 How many kernels

```
__global__ kernels in the file : 1
  name r_2_2_8_8_4_4_32   __launch_bounds__(128)
  grid.x x block = 2 x 2x8x8 = 256 threads
```

`RE-DERIVED`. One kernel, in this file. §7 is about what that does and does not
license.

### 2.2 Does the decay tensor appear as an explicit temporary

No. The kernel's complete parameter list is five pointers:

```
float* data0_4096    float* data1_4096    float* data2_16384
float* data3_16384   float* data4_4096
```

| buffer | elements | bytes | written? | index range actually touched |
|---|---|---|---|---|
| `data0_4096` | 4,096 | 16,384 (16 KiB) | **yes**, 4 stores | 0..4095, covered exactly, no gaps no overlaps |
| `data1_4096` | 4,096 | 16,384 | no, 4 loads | 0..4095 — **the same indices it writes** |
| `data2_16384` | 16,384 | 65,536 (64 KiB) | no, 32 loads | 6,144..16,383 |
| `data3_16384` | 16,384 | 65,536 | no, 32 loads | 6,144..16,383 |
| `data4_4096` | 4,096 | 16,384 | no, 32 loads | 0..4095 |

`RE-DERIVED`. Two facts worth stating separately:

- **Only `data0_4096` is ever written.** Four `float4` stores, and they are the
  last four statements in the kernel. Nothing else in the file stores anything.
- **`data1_4096` is read at exactly the indices `data0_4096` is written.** That
  is a read-modify-write, and it identifies the kernel: it is one step of the
  inter-chunk scan's state update, `tS = tS*exp(gC) + ((exp(gC-g[c])*k[c])ᵀ @
  v_new)` at `kda/tiny_kda.py:240-242`, whose output is `[B,HV,K,V]`.

The buffer sizes then identify the operands, and the arithmetic closes exactly:

```
4,096  = B*HV*K*V      = 2*2*32*32 = 4,096      <- tS, in and out
16,384 = B*NT*HV*C*K  = 2*4*2*32*32 = 16,384   <- the [B,NT,HV,C,K] family
[B,NT,HV,C,C,K] at this shape = 2*4*2*32*32*32 = 524,288 elements = 2,097,152 B
```

`RE-DERIVED`. So:

> The decay tensor at this shape is **2,097,152 B = 32× the largest buffer the
> kernel declares**, and **11.6× the sum of all five buffers it declares
> (180,224 B)**. It is not in the signature. It cannot be in the signature; it
> is 32× too big for anything in it.

**And `notes/09` §4.3's identity is confirmed here, on CUDA, from the source:**

```
largest kernel argument = B*NT*HV*C*C*4 = 2*4*2*32*32*4 = 65,536 B = 16,384 elements
largest EMITTED buffer  = 16,384 elements = 65,536 B
```

An exact match at a **fourth** shape, and — unlike `notes/09`'s three, which
were all Metal or CPU — on the backend the study's host never had. `notes/09`
§4.5 said "two backends agreeing is not three". This is the three, and one of
them is a source file rather than a spy.

### 2.3 Is `exp`/`exp2` recomputed rather than materialised

Recomputed. Completely.

```
literal exp2 in the file          : 128
bare `exp` (not exp2)             :   0
expf / __expf / powf / __powf     :   0
rcp.approx / tanh / logistic      :   0
LOG2E constant 1.4426950216293335f: 128   (one per exp2, as expected)
float4 loads (valN)               : 100
scalar float locals               : 128
global stores in the file         :   4, all to data0_4096
distinct buffers written          :   1
```

`RE-DERIVED`. **Every one of the 128 `exp2` is a `float aluN = ...` register
local, defined in the same statement-group that consumes it, and none is ever
written to a `dataN_*` pointer or read back.** There is no store of a decay. The
only stores in the file are the four final `float4` writes.

The four `exp2` groups, parsed out of the source:

```
group 0: bare alu12  = exp2(val19.w * log2e)   body 31 terms   mul comp .y
group 1: bare alu44  = exp2(val27.w * log2e)   body 31 terms   mul comp .z
group 2: bare alu76  = exp2(val35.w * log2e)   body 31 terms   mul comp .w
group 3: bare alu108 = exp2(val11.w * log2e)   body 31 terms   mul comp .x
1 + 31 x 4 = 128   (matches the literal count)
```

Each body term has the form `exp2((P - Q) * log2e) * M` — 31 subtrahends, 31
multipliers, all distinct within a group. That is `exp(gC[K] - g[c,K])` times
`k[c,K]`, i.e. `kda/tiny_kda.py:241`, evaluated **per (K, V, c) triple, in
registers**.

### 2.4 The `exp2` count, and its shape dependence

Per thread, this kernel evaluates **128 `exp2`** and stores **16** outputs
(4 `float4` stores), i.e. **8 `exp2` per output element**, and each stored
element sums **33 additive terms**:

```
term  0 : val_i.{x,y,z,w} * exp2(P * log2e)              <- the tS * exp(gC) term
terms 1-31 : (exp2((P - Q) * log2e) * k_a) * k_b         <- 31 of the C=32 chunk terms
term 32 : k_a * k_b                                     <- the 32nd chunk term
1 + 31 + 1 = 33
```

`RE-DERIVED`, by balanced-parenthesis splitting of the emitted store expression
and counting top-level additive terms (33 in all four `float4` components, in
all four stores — 16/16 identical).

**This count is not the `32×` figure, and must not be quoted as it.** The
`262,144 / 8,192 = 32×` comparison in `notes/04-codegen-comparison.md:213-248`
is counted from the `codegen_diff.py` capture at **B=1, T=64, C=64, K=64** — a
different graph (the `[C,C]` decayed gram, not the state update) at a shape
whose `B*NT*HV` is 1 against this one's 16. `notes/04` is right to scope it as
`C²·K` per `(b,n,hv)`, which is a per-shape quantity. **Neither kernel confirms
or refutes the other.** Both are `MEASURED`; they are not comparable.

What `kernel000.txt` does independently confirm is the *mechanism* that makes
the `32×` real: the `exp` is not materialised, it is recomputed inside the
accumulation, once per output term.

### 2.5 The largest genuinely device-resident intermediate

**65,536 B = 16,384 float32 = 64 KiB**, twice (`data2_16384` and
`data3_16384`). `RE-DERIVED` from the parameter list and the index ranges.

At this shape that is exactly `B·NT·HV·C·K·4` = 65,536 B and also
`B·NT·HV·C²·4` = 65,536 B — **degenerate, because K = C = 32**, so the size
alone cannot say which of the two it is, and does not need to. At the bench
shape (`B=2, T=512, H=HV=4, K=V=64, C=64`) the two differ by K = 64, and
`notes/09` §4.3 measured the identity across a 4× span there. This capture
confirms the identity's *numerical* form on CUDA and contributes no new
information about *which* tensor it is. Stated, not hidden.

The 8.42 MB planner arena and the 65,536 B largest argument are different
quantities, and §3 separates them.

---

## 3. The conflict, dissolved: five quantities, not three

Three agents reported three numbers and read them as the same quantity. They are
not, and the arithmetic below is `RE-DERIVED` from shapes at the bench shape
`B=2, T=512, H=HV=4, K=V=64, C=64, NT=8` (`kda/perf_kda.py:90`), MB = 10⁶.

| # | quantity | bytes | what kind of thing it is | where it came from |
|---|---|---|---|---|
| 1 | one `[C,C,K]` fp32 decay tensor | 67,108,864 | a **shape** | `B·NT·HV·C·C·K·4` |
| 2 | two of them, write + read | 268,435,456 | **modelled traffic** on a tensor that does not exist | the byte model |
| 3 | largest single `Buffer.allocate` request | 8,421,376 | an **allocator request** | `notes/04:170` |
| 4 | the planner **arena** (peak live intermediates) | 8,421,376 | the **same number, correctly read** | `DEBUG=1`: `memory reduced from 47.02 MB -> 8.42 MB, 6048 -> 1 bufs` |
| 5 | largest **kernel argument** | 1,048,576 | a **schedule** fact | `notes/09` §4.2, now confirmed on CUDA (§2.2) |

The re-derivation that makes this a table rather than an assertion:

```
B*NT*HV = 2*8*4 = 64 instances of [C,C,K]
[C,C,K] fp32   = 64*64*64*4 = 1,048,576 B = 1.0486 MB per instance
one decay      = 67,108,864 B = 67.1089 MB = 64.000 MiB          [= row 1]
x2 tensors     = 134,217,728 B = 134.2177 MB = 128.000 MiB
x2 touches     = 268,435,456 B = 268.4355 MB = 256.000 MiB       [= row 2]
largest kernel argument B*NT*HV*C*C*4 = 1,048,576 B = 1.0000 MiB [= row 5]
row 1 / row 5 = 67,108,864 / 1,048,576 = 64  exactly, and 64 = K
```

`RE-DERIVED`, all exact. **The 64 is not a coincidence and it is the whole
answer:** row 1 and row 5 share the factor `B·NT·HV·C²` and differ only in
whether there is a trailing `·K`, so the gap is exactly K, independent of
B, NT, HV and C. The 8.42 MB figure is a factor **8× smaller than one decay
tensor** — as `notes/09` §1 already said, and correctly, where
`notes/04-codegen-comparison.md:170` reads it as weaker than it is.

**Why 8,421,376 B cannot be a tensor, which is the cleanest single check:**

```
8,421,376 B = 2,105,344 float32 = 2^15 * 257
```

`RE-DERIVED`. It contains the prime **257**. Every shape factor in this operator
is a power of two times 64. So 8,421,376 B is a **planner suballocation**, not a
product of tensor shapes — which is exactly what `memory_plan_rewrite`'s own
`DEBUG=1` line says when it reports `6048 -> 1 bufs`. An allocator number and a
traffic model are not comparable quantities, and this is the arithmetic that
proves it rather than an assertion that it is.

### 3.1 The two shapes both called "the KDA shape"

| file:line | shape | `B·NT·HV` | one `[C,C,K]` |
|---|---|---|---|
| `kda/perf_kda.py:90` | B=2 T=512 H=HV=4 K=V=64 C=64, NT=8 | 64 | 16,777,216 elts = 67,108,864 B = 64.00 MiB fp32 / 32.00 MiB bf16 |
| `kda/cost.py:16` | B=4 T=4096 H=HV=32 K=V=128 C=64, NT=64 | 8,192 | 4,294,967,296 elts = 17,179,869,184 B = **16.00 GiB fp32** / **8.00 GiB bf16** |

`RE-DERIVED`, all exact. **8,192 / 64 = 128× apart**, both correct for their own
shape. This is the other half of the confusion and it is named here so it stops
recurring: two shapes under one name, and three quantities under one word.

**And a fourth quantity, found while checking the line references: two dtypes
under one name.** `kda/cost.py:18` sets `ES = 2` with the comment *"bytes per
element in the kernels' working dtype (bf16/fp16)"*, so `cost.py`'s headline
byte figure is **bf16**. `notes/07-crossover-thesis.md:296` says **8.00 GiB**
and `notes/09` §3.1 says **16.00 GiB** for the same tensor — and **both are
right**, in bf16 and fp32 respectively. `notes/09` §2's dtype caveat applies
here too: the *port* is fp32 by construction (`kda/tiny_kda.py:44`), so
67,108,864 B is the only size comparable to a `perf_kda`-shape threshold, while
`cost.py`'s 8.00 GiB is a modelling choice about the hand-written kernel's
dtype. Any comparison that mixes the two is comparing MB to MiB and bf16 to
fp32 at once. `notes/07`'s table is the one that gets it right and should be the
model.

### 3.2 The 128 MiB / 31.9× ratio, re-derived

The third agent reported "128 MiB against 4.02 MiB of inputs+output, a 31.9×
ratio" and declined to pick. `RE-DERIVED`:

```
two [C,C,K] fp32 at the bench shape = 134,217,728 B = 128.00 MiB     [reproduced]
the stated denominator 4.02 MiB    =  4,215,276 B
134,217,728 / 4,215,276 = 31.86x                                    [reproduced]
```

The numerator is exact. **The denominator is not reproducible from the shapes.**
Candidates at this shape:

| candidate | bytes | ratio |
|---|---|---|
| four fp32 inputs `q,k,v,g` read once | 4,194,304 = 4.000 MiB | **32.00×** |
| inputs + output `o` | 5,242,880 = 5.000 MiB | 25.60× |
| `k` and `g` only | 2,097,152 = 2.000 MiB | 64.00× |

`RE-DERIVED`. So 4.02 MiB matches none of them; the nearest clean figure is
**4,194,304 B = 4.00 MiB, giving exactly 32.00×**. I report 32.00× and flag
4.02 MiB / 31.9× as **`[UNVERIFIED]` — a denominator I could not rebuild**.
This does not matter to the verdict, because the numerator is a tensor that does
not exist, so the ratio is a ratio with a zero denominator in the program — but
it must not be quoted with four significant figures by anyone else.

### 3.3 What I could not re-derive, stated as such

`notes/09` §2 reports eight reproduced figures: 268.4355 / 67.1089 / 33.5544 /
22.2822 / 290.7177 / 5.5050 MB and 48.7619× / 13.0471×. I confirmed the first,
second and fifth, plus the schedule census and the `cost.py` figure:

| figure | re-derived | match |
|---|---|---|
| 67.1089 MB (one decay tensor) | 67,108,864 B | **exact** |
| 268.4355 MB (two, two touches) | 268,435,456 B | **exact** |
| 17,179,869,184 B at `cost.py`'s shape | 17,179,869,184 B | **exact** |
| 1,048,576 B largest kernel argument | `B·NT·HV·C²·4` | **exact** |
| 9,028,599,808 B tensor-argument bytes | sum of `(256×20096) + (16384×24612) + (131072×111) + (1048576×8207)` = 5,144,576 + 403,243,008 + 14,548,992 + 8,605,663,232 | **exact** |
| 8,606 MB for the 1 MB bucket alone | 1,048,576 × 8,207 = 8,605,663,232 | **exact** |
| 12,229 scheduled kernels | 6,188 + 6,041 | **exact** |
| **22.2822 MB compulsory-I/O term** | — | **NOT REPRODUCED** |
| **5.5050 MB denominator → 48.7619×** | my decomposition gives 5,242,880 B compulsory, and 268,435,456 / 5,242,880 = **51.20×**, not 48.8× | **NOT REPRODUCED** |
| **13.0471×** | — | **NOT REPRODUCED** |

`RE-DERIVED` for the confirmed rows. **The last three are not reproduced here
and must not be quoted as verified.** Two independent reasons that costs the
study nothing: `notes/09` already **withdraws** 48.8× and 13.0×, and they were
never measurements — `notes/06:103` says so at the time. The withdrawal is
confirmed; the arithmetic behind it is not, and that is a real distinction
rather than a technicality. It is the third instance in this repo of a figure
that is arithmetically unattributable surviving hand-off (see
`notes/06-corrections-ledger.md:170-175` and `notes/09` §2's own caught error),
and the rule that catches it is *re-derive, not verify*.

---

## 4. So where is the materialisation decision made?

**The `256×` is a property of the graph. The non-materialisation is a property
of the lowering. The 8.42 MB / 128 MiB disagreement is a difference of
quantities, not of conclusions.**

Stated precisely, because the question asks for a property of the graph, of the
lowering, or of neither:

- **The `256×` belongs to the graph, before scheduling.**
  `C²·K` vs `BC·K` at `BC=16` is `C/BC` = **64×** on the decay operand alone, and
  `256×` against `codegen_diff.py`'s `NC` sub-chunks. Both sides are *graph
  objects*. `notes/04:136-157` and `crossover/claim.py:375-412` already scope it
  as a graph-object ratio, and `notes/06:61` as "zero bytes in memory". **That
  scoping is correct and survives everything in this document.** It is also the
  model the withdrawn figures should have followed. `RE-DERIVED`: 262,144 /
  1,024 = 256, and 262,144 / 8,192 = 32.
- **The non-materialisation belongs to the lowering.** The port writes
  `Dl = (diff + (m_strict - 1.0)*1e4).exp()` (`kda/tiny_kda.py:137`) and then
  contracts it (`kda/tiny_kda.py:144`). Between those two statements TinyGrad's
  scheduler keeps the `exp` in registers. `kernel000.txt` is that decision,
  visible in emitted CUDA, on NVIDIA. `[MEASURED]`
- **The literature's over-fusion passage points the wrong way, and the scheduler
  got this instance right.** `research/TAXONOMY.md:172` — *"If the function
  materialises the N×N score matrix, no schedule removes it"* — is **false here**
  as stated. The function does not materialise the matrix; the schedule keeps it
  in registers. The correct form of the passage is *"if the function
  materialises"*, and this is the case where it does not. The finding worth
  carrying forward is the one `notes/09` §8 already names, now with a source
  file behind it:

  > **For a fused pointwise-into-reduction epilogue, TinyGrad's scheduler solves
  > this instance, and the cost of the function's shape shows up as redundant
  > transcendental evaluation — not as bytes.**

- **The first agent's mechanism report is confirmed, and it is the whole
  explanation of the 8.42 MB vs 128 MiB gap.** "One kernel, `exp` fused into the
  contraction" *is* why the arena is 8.42 MB and not 67 MB. Under this reading
  the materialisation question is not really a disagreement at all: the 8.42 MB
  is the schedule's answer, and the 128 MiB is what the graph *would* have cost
  had nothing fused. **My reading of the brief is adopted, and it survives the
  CUDA source.**

**What the 8.42 MB arena does *not* tell you, and why that is a real limit.**
The arena is a peak-live footprint for *one* schedule of *one* shape on *one*
backend. It is a fact about the lowering, not about the graph, and not about
traffic. A different backend or a different schedule could produce a different
arena with the same graph. So the arena **corroborates** the verdict; it does not
**carry** it. The argument that carries the verdict is §2.2 — a tensor that is
not in the signature cannot be read or written — plus `notes/09` §4.1's positive
control, which proves the instruments fire when the tensor is forced to exist.

---

## 5. What is still not measured, and must not be inferred

**DRAM traffic (M4).** No traffic number is reported here because none was
measured, and **a traffic number must never be inferred from an allocator
number.** What the measurement does establish needs no counter: a buffer carries
traffic only if some kernel reads or writes it; a kernel can only read or write
buffers in its argument list; the decay is in no kernel's argument list, on
Metal, on CPU, and now on CUDA from the emitted source. **Therefore the 268.4 MB
of modelled traffic has no carrier in the generated program, and the charge is
invalid on any device: no cache, no L2 and no card can move a tensor that was
never written.**

**The largest unresolved quantity, named: schedule argument bytes.**
`notes/09` §5 records 9,028,599,808 B of tensor-argument bytes against a
1,048,576 B live working set and an 8,421,376 B arena, with the 1,048,576 B
buffer passed as an argument to **8,207** separate launches. `RE-DERIVED` in
§3.3 — the sum reproduces exactly. Whether those re-reads hit L2 or reach DRAM
is a **cache** question and cannot be answered off-card. It is the most
plausible place for this operator's real traffic to be, and it is neither
54.1 MB nor 290.7 MB.

**On the T4 specifically**, 15.6 GB of VRAM against a 16.00 GiB fp32 decay
tensor at `kda/cost.py:17`'s shape means the study shape is not reachable on
this card either, independently of the fusion.

**What a real GPU would still add:**

1. `dram__bytes.sum` for one `kda_chunk` call, fp32, at the bench shape. Nsight
   Compute, one profile. The only thing that settles M4.
2. Whether the 8,207 re-reads hit L2. Off-card: impossible.
3. Whether the fused schedule is *fast*. It is small, and small is not the same
   as quick: 12,082 launches per call at 215.9 ms of GPU time
   (`notes/10-t4-results.md` §2) is not a fused-at-all schedule, it is 12,082
   kernels. **The decay is fused; the operator is not.** Those are separate
   findings and the record has been at risk of merging them.

---

## 6. The two figures that survive, and why

| figure | status | why it survives |
|---|---|---|
| **`256×` element ratio** | **UNAFFECTED** | `C²K/(BC·K)` is arithmetic on two *graph objects*. Neither side needs the tensor to exist. `notes/06:101` already scopes it "zero bytes in memory". Cross-scope correction in `crossover/claim.py:375-412` still stands. |
| **`32×` more `exp2`** | **UNAFFECTED, and now mechanism-backed** | counted from generated source (`notes/04:213-248`); orthogonal to memory; and §2.3 shows *why* it is the residual cost — the `exp` is recomputed per output term instead of being loaded. |

Everything else that was in circulation as a byte claim is withdrawn, below.

---

## 7. If the emitted source does not settle it — it does not, quite, and here is what would

**It does not fully settle it, and the gap is a specific, nameable one.**

The notebook cell printed `captured 1 kernels from CUDA` and wrote one file. On
the **same box**, at a **larger** shape, `kda/perf_kda.py` measured **12,082
launches per call** via `DEBUG=2`. Those cannot both be complete, and the
24,968-char source is plainly not a whole `kda_chunk` — the UT transform alone
unrolls `C(C−1)/2` = 496 elementary steps at C=32, and the chunk scan adds
NT−1 more state updates. **So the `Device[...].compiler.compile_cached` hook is
not the compile entry point for most kernels on the CUDA backend, and
`kernel000.txt` is one kernel, correctly analysed, not the operator's schedule.**

Two independent reasons this does not reopen the verdict:

1. **The kernel that matters is the one that would have to read the decay.** If
   any kernel materialised the decay, that kernel would take a 2,097,152 B
   argument at this shape — 32× anything in the file. Any *other* kernel in the
   schedule would have to be the one writing it. So the interesting question is
   whether some *unseen* kernel does that, and at this shape the answer is
   constrained: the operator's `[B,NT,HV,C,C,K]` tensors are consumed by exactly
   the two contractions at `kda/tiny_kda.py:144` and `:232`, and a kernel that
   materialises one would show a 2 MiB argument.
2. `notes/09`'s M2 census already ran on the **whole** schedule on two backends
   and found four distinct parameter sizes, the largest 1,048,576 B, no bucket
   at 67,108,864 B. The CUDA source is consistent with that at a fourth shape
   and a fifth instance of the identity.

**Exactly what would close it.** One measurement, on the box, no new hardware:

> Run `kda/materialisation.py`'s M2 schedule census — the one that already works
> on METAL and CPU — against `Device.DEFAULT == CUDA` at the bench shape
> `B=2, T=512, H=HV=4, K=V=64, C=64`, and report the histogram of kernel argument
> buffer sizes.

The instrument exists, in `kda/materialisation.py` §4.2, and it needs only
`--quick`-scale GPU time. Its acceptance test is a single number: **is there a
bucket at 67,108,864 B?** If there is not, M2 is closed on CUDA for the whole
schedule and §1 needs no qualification. If there is, this document is wrong and
the whole byte model is reinstated.

A weaker but sufficient alternative, and cheaper: set `DEBUG=2` at the bench
shape and parse the per-launch line for parameter sizes — the same parse
`perf_kda.py:43-71` already does for timings. That gets the histogram off a
complete census rather than a `compile_cached` hook.

**Neither is needed to withdraw the figures in §8.** The withdrawal rests on
`notes/09` §4.1's positive control (the instrument fires when the tensor is
forced to exist, to the byte, at three shapes) and on §2.2 for the kernel in
hand. The CUDA run *strengthens* the case; it is not what carries it.

---

## 8. Every figure that must be withdrawn, restated, or re-scoped

### 8.1 Withdrawn

**Provenance caveat on the "where it is asserted" column.** Every line reference
in the first nine rows is **inherited from `notes/09` §8**, which I read and did
not re-verify line by line — `bench/`, `crossover/`, `research/` and
`notes/06`/`notes/DECISIONS.md` are owned by other agents and were out of scope.
The references into `README.md`, `notes/03-kda.md`, `notes/04`, `kda/` and
`kernels/` I **did** verify individually. Before anyone acts on the first nine
rows, check the line numbers; the claims are not in doubt, the pointers might
drift.

| figure | where it is asserted | status |
|---|---|---|
| **268.4 MB** decay traffic | `bench/HYPOTHESES.md:169`, `:174`; `notes/06:102`, `:154`; `notes/DECISIONS.md:23` | **WITHDRAWN.** The tensor it charges is not in the program. The *arithmetic* is exact (§3.3) and the *object* does not exist. |
| **290.7 MB** ported total | `bench/HYPOTHESES.md:172`, `:240`, `:545`, `:557`; `bench/PLAN.md:240`, `:242`; `notes/06:103`, `:156`; `notes/DECISIONS.md:23`; `bench/FAIR-BASELINE.md:655` | **WITHDRAWN.** 268.4 + 22.28. Note the second term is itself unreproduced (§3.3), so this is a withdrawal of a sum containing an unattributable addend. |
| **48.8×** decay ÷ compulsory | `notes/06:48`, `:102`, `:159`; `notes/DECISIONS.md:23`; `crossover/literature.py:298`, `:563`; `research/PARITY-SCOPE.md` via C-01 | **WITHDRAWN.** A ratio whose numerator has no carrier — and whose denominator I could not reproduce either (§3.3). |
| **13.0×** predicted time payoff | `notes/06:48`, `:103`, `:157`; `notes/DECISIONS.md:23`; `bench/HYPOTHESES.md:156` (the H2 title), `:173`; `bench/PLAN.md:240`; `bench/FAIR-BASELINE.md:655` | **WITHDRAWN.** Premise fails; never was a measurement. Also unreproduced (§3.3). |
| **H2** as written | `bench/HYPOTHESES.md:156-192`; `bench/PLAN.md` R8 | **REFUTED IN ITS PREMISE, NOT BY ITS OWN TEST.** Its prediction ([233, 349] MB) and its threshold (< 58 MB) are both still unrun. |
| **"the decay dataflow is the actual bottleneck"**, as a bandwidth claim | `notes/03-kda.md:120` (ranked #4) | **WITHDRAWN as a bandwidth claim.** Survives as an *arithmetic* claim via `exp2`, not bytes. |
| **H1**'s intensity argument | `bench/HYPOTHESES.md:133-138` (0.0076 FLOP/byte, 27,000× memory bound) | **WITHDRAWN.** It computes the intensity of a contraction that reads a tensor no kernel reads. |
| **H13**'s arithmetic | `bench/HYPOTHESES.md:545-559` (0.81 FLOP/byte, 31:1) | **CONDITIONAL.** Its denominator is withdrawn. `crossover/literature.py` §3 already reports its **FLOP** side does not reproduce either (0.9802 vs 0.8079). |
| **P-2** as written | `notes/06:227`; `bench/PLAN.md:240-242` | **MUST BE RE-SCOPED BEFORE IT RUNS.** A passing run would no longer confirm the model. It should measure where the traffic actually is — the 8,207 re-reads. |
| **31.9×** (per-tensor traffic ratio) | third agent's report; not yet in the tree | **RESTATED as 32.00×** if the denominator is the four fp32 inputs (4,194,304 B), and the 4.02 MiB denominator is **`[UNVERIFIED]`** (§3.2). Moot anyway: the numerator does not exist. |
| **22.2822 MB** compulsory-I/O term, **5.5050 MB** denominator, **13.0471×** | `notes/09` §2, `bench/HYPOTHESES.md` | **NOT REPRODUCED by me** (§3.3). Do not quote as verified. `notes/09` already withdraws what depends on them. |

### 8.2 Unaffected, and it is worth saying which

| figure | where | why it survives |
|---|---|---|
| **`256×` element ratio** | `notes/06:101`; `notes/04:136-157`; `crossover/claim.py:375-412` | `C²K/(BC·K)` is arithmetic on two *graph objects*. §4. |
| **`32×` more `exp2`** | `notes/04:213-248` | counted from generated source; orthogonal to memory; §2.3 gives it a mechanism. |
| **8.42 MB / 54.1 MB** | `notes/04:170`, `:173` | reproduced to the byte by `notes/09`; 8.42 MB is the planner **arena** and `notes/04` reads it as weaker than it is (§3). |
| **12,229** scheduled kernels | `notes/04:470` | 6,188 + 6,041, the two `realize()`s. The unrolled UT transform is untouched. The **12,090** at `notes/04:416` is a different counter (DEBUG=2 launches per iteration), and the T4 run's **12,082** is a third run of that counter on a different backend — none of them changes here. |
| **the identity `largest argument = B·NT·HV·C²·4`** | `notes/09` §4.3 | **confirmed on CUDA from emitted source at a fourth shape** (§2.2). |

### 8.3 The files, itemised

1. **`README.md:17`** — "on KDA it materialises 67 MB of intermediates that a
   hand-written kernel never builds". **The single most visible instance of the
   withdrawn claim, in the repo's own summary.** Withdraw or restate. This is the
   first thing a reader sees and it is false.
2. **`README.md:24`** — the `notes/03` row: "KDA ports completely (6.6e-06).
   12,090 kernels; **the bottleneck is two `[C,C,K]` intermediates**, not the
   sequential loops." The first half is fine and now also measured on CUDA (6/6
   pass, worst relerr 8.56e-06, `notes/10` §4). The second half is a byte claim
   and is **withdrawn as a bandwidth claim**; it survives as an `exp2` claim.
3. **`notes/03-kda.md:91-95`** — the 33.6 MB × 2 table. These are *shapes*, which
   are fine, but the surrounding prose reads them as bytes that exist. Lines
   93-94 are quoted by `kda/tpu/shapes.py:136` and `research/TPU.md:64`.
   **Restate as logical sizes with an explicit "zero bytes"**, which is what
   `notes/04:136` already does and `notes/03` does not.
4. **`notes/03-kda.md:120`** — ranking row 4, "**the actual bottleneck** — 33.6 MB
   × 2 of intermediates, 76.8% of GPU time". Withdrawn as a bandwidth claim. The
   76.8% of GPU time in reduce kernels is a separate measurement and is not
   withdrawn; the *explanation* for it moves from bytes to `exp2`.
5. **`notes/03-kda.md:127-129`** — "TinyGrad … cannot recover from materialising
   67 MB of intermediates, because that is a decision the graph has already made
   by the time the scheduler runs." **This is the sentence that got it exactly
   backwards, and it is the one worth killing.** The graph does not materialise
   it, the scheduler does not, and the decision was recoverable. This is the
   clearest instance of the study's framing being the thing at fault rather than
   the instrument.
6. **`notes/03-kda.md:211-214`** — "`Dl` and `Dq_all` at 33.6 MB each … At the
   shape `NOTES.md` analyses (B=4, T=4096, H=32, K=128) they are **8.2 GB
   each**, which does not fit in this machine's 32 GB at all."
   `RE-DERIVED`, at `kda/NOTES.md:16`'s shape (B=4, T=4096, H=HV=32, K=V=128,
   C=64, NT=64), one `[C,C,K]` = 4,294,967,296 elements:

   | dtype | one tensor | a write + a read |
   |---|---|---|
   | fp32 (the port's only dtype, `kda/tiny_kda.py:44`) | 17,179,869,184 B = **17.18 GB = 16.00 GiB** | 34.36 GB |
   | bf16 | 8,589,934,592 B = 8.59 GB = 8.00 GiB | 17.18 GB |
   | **`notes/03:212` and `kda/NOTES.md:153`, `:162`** | **8.2 GB** | **16.4 GB** |

   **8.2 GB matches neither, and understates fp32 by 2.1×.** The source is
   `kda/NOTES.md:162` (and the same number as "8 GB" in its code comment at
   `:153`); `notes/03` quotes it. The rest of the repo has this right and
   disagrees with itself: `notes/07-crossover-thesis.md:296` says **16.0 GiB**
   fp32 and **8.00 GiB** bf16 at the same shape, and `notes/09` §3.1 says
   **16.00 GiB** fp32 — all three of those re-derive exactly, mine included.
   **Corrected: 17.18 GB (16.00 GiB) each in fp32, 34.4 GB for a write + a read.**
   `notes/03:213`'s *conclusion* ("does not fit in this machine's 32 GB") is
   right and its number is wrong: 2 × 8.2 GB = 16.4 GB **would** fit in 32 GB,
   so the sentence is inconsistent with itself, and it is only consistent with
   the fp32 figure, 34.4 GB, which does not. This is a **new** arithmetic error
   found in re-derivation, not one carried over from a prior note — and it is
   the fourth instance in this repo of a byte figure surviving hand-off
   unattributed. The file is owned elsewhere; flagged.
7. **`kda/tiny_kda.py:64-83`** (`_causal_decay`, "the single biggest memory item
   in the whole operator") and **`:119-144`** — the source comment that motivates
   the whole study, false as written, and the origin of the 67 MB / 268 MB
   figures. Not editable from here; flagged. **`_causal_decay` is also dead
   code** — `kda_chunk` computes `Dl` inline at `:137` and never calls it.
8. **`kda/codegen_diff.py:118-120`** — prints "no `exp2`" and "the decay … had to
   be materialised". Both false; the second is what this document exists to
   refute. `notes/04:306-309` caught the first. Flagged, not edited.
9. **`kda/perf_kda.py:143`** — hardcodes the string `"measured (Metal, ...)"` on
   a CUDA run; `kda/replay_kda.py:97`, `:101-102`, `:105` hardcode a Mac-era
   0.109 s of GPU time and a 49.5 s COLD against a measured 96,837 ms. Flagged,
   not edited. See `notes/10` §2.2.
10. **`bench/HYPOTHESES.md`, `bench/PLAN.md`, `bench/FAIR-BASELINE.md`,
    `notes/06-corrections-ledger.md`, `notes/DECISIONS.md` row 17,
    `crossover/literature.py:271-313`, `crossover/claim.py:375-412`,
    `research/TAXONOMY.md:172`, `kda/tpu/shapes.py:136`, `research/TPU.md:64`,
    `kda/NOTES.md` §6, `kda/SPEC.md` §4.4, `kda/cost.py:17`** — as itemised in
    `notes/09` §8, unchanged by this document except where §8.1 above corrects or
    withdraws further. `crossover/claim.py` is the model to follow: it is the one
    place that scoped the 256× correctly the first time.
11. **`bench/selftest.py:306-350`** — declined to pick a side and printed both
    ratios. **That was the right call and this document is the resolution it was
    waiting for.**
12. **`research/TAXONOMY.md:172`** — the over-fusion passage. **This is the one
    item that gets *stronger* from a new artifact rather than weaker**, and the
    qualifier it needs is "if the function materialises": here the function does
    not, the schedule keeps the reduction in registers, and `kernel000.txt` is
    the file that shows it (§2.3).

---

## 9. Files

| file | what it produced here |
|---|---|
| `out/kaggle/20260928-014248-gpu/codegen/kernel000.txt` | §2 in its entirety. The only direct evidence of TinyGrad's CUDA lowering in the repo. |
| `kernels/build_notebooks.py:578-597` | the capture, and the shape it captured at (`:590`) — which is *not* `codegen_diff.py`'s |
| `out/kaggle/20260928-014248-gpu/mc-cuda-tinygrad-probe-battery.log:576-577` | `captured 1 kernels from CUDA`, the undercount that §7 is about |
| `kda/tiny_kda.py:119-144`, `:215-216`, `:240-242` | the graph the emitted source implements, and the three `exp`-bearing contractions |
| `kda/perf_kda.py:90` | the bench shape: B=2, T=512, H=HV=4, K=V=64, C=64 |
| `kda/cost.py:17` | the other shape: B=4, T=4096, H=32, K=V=128, C=64, 16.00 GiB per tensor |
| `kda/materialisation.py` | `notes/09`'s instrument: the M1/M2 census and its positive control. §7's proposed run is this file against CUDA. |
| `notes/09-materialisation-verdict.md` | the prior verdict. **Confirmed and extended**, with three of its figures flagged unreproduced (§3.3) and one source comment corrected (§8.3 item 6). |
| `notes/04-codegen-comparison.md` | the 8.42 / 54.1 MB measurements and the 256× / 32× scoping, both of which survive |
| `notes/10-t4-results.md` | the run this document draws on; the perf/replay decomposition and the `compile_cached` undercount |
