# 09 — the materialisation verdict

**The tensor is not materialised. The study's headline byte figures are withdrawn.**

The KDA port's `[B,NT,HV,C,C,K]` decay tensor is never written to device memory,
at any shape or backend measured. The 268.4 MB / 48.8× / 13.0× framing rests on a
premise that the generated program contradicts, and the compiler beat the
analytic model. This is the more interesting outcome of the two the brief allowed,
and it is the outcome the brief said it would accept.

Everything below is re-derived in `kda/materialisation.py`, which is the only
file in this note that produced a number. Nothing here is transcribed.

```
.venv/bin/python kda/materialisation.py            # ~4 min, 3 shapes + 2 backends
.venv/bin/python kda/materialisation.py --quick    # bench shape only, ~1 min
```

---

## 1. The conflict, stated exactly

Three numbers are in circulation that read as the same quantity. Two are
allocator numbers and one is a traffic model, and the record compares them
directly.

| number | what it is | where it came from |
|---|---|---|
| **8.42 MB** | 8,421,376 B, "largest single allocation" | `notes/04-codegen-comparison.md:170` `[MEASURED]` |
| **54.1 MB** | 54,149,120 B, "total bytes allocated (all calls)" | `notes/04-codegen-comparison.md:173` `[MEASURED]` |
| **268.4 MB** | modelled DRAM traffic, 2 decay tensors × write+read | `bench/HYPOTHESES.md:168-169` `[MODELLED]` |

`notes/06-corrections-ledger.md:124-125` pre-registered a refutation threshold
of **measured total < 58 MB** for the byte model. The 54.1 MB figure is under it.
The ledger therefore looks refuted by its own §2a row.

**The brief was right that the conflict is only apparently sharp, and right that
the premise is the load-bearing thing.** §3 below shows the shapes are *not*
different (both measured numbers and the model are at the same shape, so the
conflict is real), and §4 shows the premise fails on a third axis neither number
touches: the tensor is absent from the **schedule**, which is a stronger and
cleaner fact than being small in the allocator.

**What 54.1 MB is, precisely.** It is the sum of every
`Buffer.allocate` request made by one `kda_chunk` call, including graph-build
constants, counted with multiplicity — 4,591 requests, 54,149,120 B. It is not a
peak, not a footprint, and not traffic. The 6.52 MB of it that is *not*
realize-time comes from `tiny_kda.py:117` (5 input uploads, 4,210,688 B),
`:161` (`M`, 1,048,576 B), `:164`+`:179` (the 64 `eyeT` rows and 64 `b` zeros,
1,064,960 B), `:205` (`S`, 131,072 B) and five small masks. The remaining
47.63 MB is the intermediates of the two `realize()`s. **[MEASURED]** — and both
recorded figures reproduce to the byte, which is what makes the comparison worth
making at all.

**What 8.42 MB is, precisely — and the note that recorded it understates it.**
8,421,376 B is the memory planner's **arena**: tinygrad's
`memory_plan_rewrite` suballocates every logical intermediate into one buffer per
device lane, and its own `DEBUG=1` line says so —
`memory reduced from 47.02 MB -> 8.42 MB, 6048 -> 1 bufs`. Captured first-hand
in `kda/materialisation.py` §5. So 8.42 MB is not "the biggest buffer the port
happened to ask for"; it is the **peak live intermediate footprint of the entire
operator**, which is the answer to "how much device memory does this need", and
it is **8× smaller than one decay tensor** rather than merely smaller.
`notes/04-codegen-comparison.md:170` records the number correctly and reads it as
weaker than it is. **[MEASURED]**

---

## 2. Provenance, re-derived

`kda/materialisation.py` §1 rebuilds the whole byte model from the shapes on
`kda/perf_kda.py:90` and reproduces all eight delivered figures exactly:
268.4355 / 67.1089 / 33.5544 / 22.2822 / 290.7177 / 5.5050 MB and 48.7619× /
13.0471×. **The delivered byte model is arithmetically sound. It is charged to
the wrong thing.**

**The dtype caveat, which matters and was in doubt.** All of it is fp32. The port
is fp32 by construction — `tiny_kda.py:44` sets `F = dtypes.float32` and five
`np.float32` casts follow — so one decay tensor is **67,108,864 B**, and the
ledger's "< 58 MB, and it must be fp32 because bf16's 33.55 MB is inside L2"
argument is aimed at the right number. The brief's "the fp32 size is 67.1 MB
where bf16 is 33.6 MB, and 58 MB is sensitive to which is used" is correct and
resolves cleanly: the port cannot run in bf16 at all, so 67.1 MB is the only
size that can be compared to 58 MB. **[MEASURED dtype, `kda/tiny_kda.py:44`]**

**My own re-derivation found an error, and it is worth recording because it is
the second instance.** My first draft of the I/O term was
`4 × (2K + V + V) × B × T × HV × 4 B` = 16.78 MB, on the reasoning that there are
four tensors and two touches. It is 3× wrong, and the error was invisible in the
decay term, so it moved exactly the four figures that matter — 22.3 → 29.6 MB,
5.5 → 17.0 MB, 13.0× → 10.06×, 48.8× → 15.75× — while 268.4 MB sat there
untouched looking right. It was caught only by writing each coefficient out
instead of importing one. That is `notes/06-corrections-ledger.md:170-175`'s
failure mode (a subtotal that double-counted I/O) arriving from a different
direction. The rule is *re-derive*, not *verify*.

---

## 3. The units problem: what 268.4 MB counts

**It is not narrow, and there is no units error in it.** 268.4 MB is the
whole-graph figure for both decay tensors at the `perf_kda.py` shape, term by
term, and every factor is an exact product:

```
B·NT·HV                  = 2 × 8 × 4  =        64   instances of [C,C,K]
[C,C,K] fp32             = 64·64·64·4 = 1,048,576 B = 1.049 MB per instance
one decay tensor         =        67,108,864 B =  67.11 MB
× 2 tensors (Dl, Dq_all) =       134,217,728 B = 134.22 MB
× 2 touches (write+read) =       268,435,456 B = 268.44 MB   ← 268.4 MB
```

Not per-step, not per-program, not per-instance, not a single slab.

**So where does the brief's 17 GB come from? From a different shape, multiplied
by hand.** The repo names two different things "the KDA shape":

| file:line | shape | `[B,NT,HV,C,C,K]` fp32 |
|---|---|---|
| `kda/perf_kda.py:90` | B=2 T=512 H=HV=4 K=V=64 C=64, NT=8 | 67,108,864 B = 67.11 MB |
| `kda/cost.py:17` | B=4 T=4096 H=32 K=V=128 C=64, NT=64 | 17,179,869,184 B = **16.00 GiB** |

They are 256× apart, both correct for their own shape, and
`crossover/literature.py` §3 already prints both side by side without
connecting them. The brief's parenthetical *"B4, T4096, HV32, K=V=128, C=64, as
used in `kda/perf_kda.py`"* is itself a conflation — that is `cost.py`'s shape;
`perf_kda.py:90` uses B=2, T=512, HV=4, K=V=64. **Both contested numbers and
the model are at the `perf_kda.py` shape, so the conflict is real and sharp, not
a shape artefact.** The 17 GB is real too, and it is `[MODELLED]` for
`cost.py`, which no measurement in this note reaches (§6).

**Named as the source of the confusion, then:** two shapes both called "the KDA
shape", and three quantities under one word — modelled traffic called "bytes"
next to allocator bytes, which is exactly the conflation C-01 and C-03 were filed
against, surviving a third round.

---

## 4. The measurement

Four senses of "materialised", kept apart (`kda/materialisation.py` §0):

| | sense | status |
|---|---|---|
| **M1** | a device buffer of the decay tensor's exact size is ever requested | **MEASURED** |
| **M2** | a kernel **argument** of that size exists in the generated plan | **MEASURED** — decisive |
| **M3** | peak live device memory of the whole operator | **MEASURED** |
| **M4** | actual DRAM traffic | **NOT MEASURED** — needs a counter |

### 4.1 The positive control, which is the reason the null result is evidence

An allocator spy's null result is worthless without proving the spy fires. So:

| case, isolated decay at B=2 NT=8 HV=4 C=64 K=64 fp32, tensor = 67,108,864 B | largest allocation | saw it? |
|---|---|---|
| **A** fused, exactly as `tiny_kda.py:137-144` writes it | 1,048,576 B | **no** |
| **B** forced: `Dl.realize()` first, then contract | **67,108,864 B** | **YES** |
| **C** forced: `Dl.contiguous().realize()` first | **67,108,864 B** | **YES** |

And the same pair at two more shapes the record cares about:

| shape | decay tensor | fused → largest alloc | forced → largest alloc |
|---|---|---|---|
| `codegen_diff.py` B=1 NT=1 HV=1 C=64 K=64 | 1,048,576 B | 16,384 B — **no** | 1,048,576 B — **YES** |
| **`cost.py` family** B=4 NT=1 HV=32 K=V=128 C=64 | **268,435,456 B** | 4,194,304 B — **no** | 268,435,456 B — **YES** |

The second is `kda/cost.py:17`'s shape with **B, HV, K, V and C all as
`NOTES.md` and `cost.py` have them** and only NT cut 64× — as close to the study
shape as this host can afford, since at full NT the fp32 decay tensor alone is
16.00 GiB of arithmetic. Its fused largest allocation is 4,194,304 B =
`[B,NT,HV,C,K]` — the input — and 268,435,456 / 4,194,304 = **64 = K**, so the
identity of §4.3 holds here too. **[MEASURED]**

The `codegen_diff` row's forced allocation is exactly `C·C·K·4` = 1,048,576 B —
the `[C,C,K]` slab of one `(b, chunk, head)`, which is the 262,144-element
object `notes/04-codegen-comparison.md` §2 measures. A slab exists only when the
schedule demands one.

**Three shapes, one answer: fused never allocates the decay tensor; forced
always does, to the byte.** **[MEASURED]**

### 4.2 The whole operator, three shapes, fresh graph per run

`kda/tiny_kda.py:kda_chunk` at `B=2 T=512 H=HV=4 K=V=64 C=64`, fp32. Allocator
spy on `Buffer.allocate`/`deallocate`; schedule spy on the memory planner's
input, so it sees buffers *before* arena coalescing.

```
M1  ALLOCATOR
    allocation calls            =     4,591     (reproduces notes/04 exactly)
    distinct sizes requested    =         6     (reproduces notes/04 exactly)
    largest single request      =  8,421,376 B = 8.42 MB  (reproduces notes/04)
    sum of all requests         = 54,149,120 B = 54.15 MB  (reproduces notes/04)
    peak LIVE device memory     = 22,511,616 B = 22.51 MB  ← NEW, not in the record
    any request >= 67,108,864 B =        NO

M2  SCHEDULE
    kernel launches (2 realizes) =    12,229
    distinct param buffer sizes  =         4
    largest param buffer         =  1,048,576 B = 1.05 MB   ← NEW
    any param == 67,108,864 B    =        NO
    largest param / decay tensor = 1/64 = 1/K
    identity  B*NT*HV*C*C*4      =  1,048,576 B  -> MATCH
    tensor-argument bytes        = 9,028,599,808 B = 9,028.6 MB  ← NEW
```

Every parameter buffer size is identified by matching nbytes against the port's
own tensor shapes: 256 B = `[C]` (an `eyeT` row), 16,384 B =
`[B,NT,HV,C]`/`[C,C]`, 131,072 B = `[B,HV,K,V]` (the state), 1,048,576 B = the
`[B,NT,HV,C·]` family. **The decay tensor is in no bucket.** At this shape
K=V=C=64 and T=NT·C, so `[B,NT,HV,C,C]`, `[B,NT,HV,C,K]`, `[B,NT,HV,C,V]` and
`[B,T,HV,V]` are *all* 1,048,576 B — the size alone cannot say which tensor the
largest argument is, and does not need to. The decay tensor is K× that size and
is absent.

### 4.3 The identity, confirmed at three shapes

| shape | decay tensor fp32 | largest kernel argument | ratio | identity `B·NT·HV·C²·4` |
|---|---|---|---|---|
| B=2 NT=8 HV=4 K=64 | 67,108,864 B | 1,048,576 B | 1/64 | 1,048,576 **MATCH** |
| B=1 NT=8 HV=4 K=64 | 33,554,432 B | 524,288 B | 1/64 | 524,288 **MATCH** |
| B=2 NT=16 HV=4 K=64 | 134,217,728 B | 2,097,152 B | 1/64 | 2,097,152 **MATCH** |

```
largest kernel argument  =  B·NT·HV·C·C·4
one decay tensor         =  B·NT·HV·C·C·K·4   =  K × that argument
```

Both sides contain `B·NT·HV·C²`, so the gap is exactly K and is **independent of
B, NT, HV and C**. This is `[MEASURED]` over a 4× span and is the only thing that
carries to shapes this host cannot schedule (§6).

### 4.4 Largest intermediate that genuinely reaches device memory, and its byte model

**1,048,576 B = 1.0486 MB.** One write + one read = **2,097,152 B = 2.0972 MB**.
The same rule applied to the decay tensor would give 134,217,728 B = 134.2 MB, so
the modelled figure is **64×** the largest real one at this shape. That 64 is K.

### 4.5 Why a laptop settles this, and what it does not

"Does this tensor reach device memory" is a property of the **generated
schedule**, not of the card: a buffer is either in the plan or it is not, and
which buffers a kernel may touch is decided by the scheduler. Each backend is run
in its own process (tinygrad picks its backend at import time):

```
CENSUS dev=METAL ... max_param=1048576 arg_bytes=9028599808 params=[(256,20096),(16384,24612),(131072,111),(1048576,8207)]
CENSUS dev=CPU   ... max_param=1048576 arg_bytes=9028599808 params=[(256,20096),(16384,24612),(131072,111),(1048576,8207)]
the two schedules agree on max_param / kernels / arg_bytes / params: YES
```

Two backends, two memory models, **byte-identical buffer sets**. The allocator
census differs by 16 B out of 54,149,120 (4,591 vs 4,593 requests) — backend
bookkeeping in constant uploads, changing nothing.

Sufficient for M1, M2, M3. **Not sufficient for M4**, and **not a claim about
CUDA**, which is untested for the same reason P-1 is untested. The correct form
of the claim is *"tinygrad's scheduler fuses this, on every backend measured"*.

### 4.6 The gotcha, handled

`Tensor.realize()` on an already-realized graph is a silent no-op, so a repeat
call on the same tensors measures nothing. `kda_chunk` takes numpy inputs and
builds a fresh graph per call, and the script **runs every call twice and requires
the two censuses to be identical** — they are, at all three shapes, because
tinygrad's `schedule_cache` stores the lowered *plan* and buffers are re-resolved
and re-allocated on every `realize()`. Both runs are reported.

---

## 5. DRAM traffic is not measured, and this does not change that

No traffic number is reported here, because none was measured, and **a traffic
number must never be inferred from an allocator number.** What the measurement
does establish needs no counter:

> A buffer carries DRAM traffic only if some kernel reads or writes it. A kernel
> can only read or write buffers that appear as its arguments. The decay tensor
> appears as no kernel's argument, at three shapes, on two backends, with an
> instrument proven able to see it when it is there. **Therefore the 268.4 MB of
> modelled traffic has no carrier in the generated program, and the charge is
> invalid on any device: no cache, no L2 and no card can move a tensor that was
> never written.**

**The largest unresolved quantity, now named: SCHEDULE ARGUMENT BYTES.** The plan
passes a 1,048,576 B buffer as an argument to **8,207** separate launches —
8,606 MB of argument bytes against a 1.05 MB live working set and an 8.42 MB
arena. Whether those re-reads hit L2 or reach DRAM is a **cache** question and
cannot be answered off-card. It is a real number in the record now, it is neither
54.1 MB nor 290.7 MB, and it is the most plausible place for the operator's
traffic to actually be. **`notes/06` §3's `[UNVERIFIED]` item 4** — "whether
Z/Btri/M/Aqk are each really written-then-read is a *scheduling* property" — now
has a measurement for the part that is not the decay tensor: 8,207 of them are
not, and the largest thing that is, is 64× smaller than the model charges.

**What a real GPU would add, none of which is substituted for here:**

1. `dram__bytes.sum` for one `kda_chunk` call, fp32, at the bench shape. The only
   number that settles M4. Nsight Compute, one profile, R8 in `bench/PLAN.md`.
2. Whether the 8,207 re-reads hit L2. Off-card: impossible.
3. Whether the CUDA backend's schedule differs from Metal's and CPU's. Two
   backends agreeing is not three.
4. bf16. The port is fp32 by construction; a bf16 run could not distinguish the
   two stories, and this port cannot produce one.

---

## 6. What was not measured

The full operator at `kda/cost.py:17`'s shape (B=4, T=4096, HV=32, K=V=128,
C=64) is **not** run. **[UNVERIFIED]**, deliberately. Its `B·NT·HV` is 8,192
against the bench shape's 64 — **128×** — and the graph is `O(B·NT·HV·C²)` in
UOps because the UT transform unrolls C forward-substitution steps each carrying
a `[B,NT,HV,C,C]` matrix. First-hand at the bench shape (`kda/materialisation.py`
§5): one `kda_chunk` schedules 6,188 kernels in ~15.0 s with **288,406** live
UOps, then 6,041 kernels with 462,345. ×128 is ~37M UOps before a single kernel
launches. This host cannot schedule that in reasonable time.

The **isolated decay is** run at a shape in the same family (B, HV, K, C all equal
to `cost.py`'s, only NT reduced) so the decay-fusion question at the study shape
is not left untested. The identity predicts a largest kernel argument of
**134,217,728 B** against a 16.00 GiB decay tensor there — a gap of 128 = K.
`[MODELLED]` for that shape, `[MEASURED]` for the identity.

---

## 7. The verdict, and the five quantities

> **The KDA `[B,NT,HV,C,C,K]` decay tensor is NEVER MATERIALISED. At
> B=2 T=512 H=HV=4 K=V=64 C=64 fp32 it is never allocated (largest device
> allocation is the 8.42 MB planner arena) and is never a kernel argument; the
> largest kernel argument is 1,048,576 B = `B·NT·HV·C²·4`, exactly K=64 smaller
> than the 67,108,864 B tensor. The 268.4 MB / 48.8× / 13.0× figures are
> WITHDRAWN; the byte model charged a tensor that has no carrier in the generated
> program. DRAM traffic itself remains unmeasured on this host.**

| quantity | bytes | MB | in the record? |
|---|---|---|---|
| allocator — sum of all requests | 54,149,120 | 54.15 | yes: `notes/04` 54.1 MB |
| allocator — largest single request | 8,421,376 | 8.42 | yes: `notes/04` 8.42 MB |
| allocator — **peak live memory** | 22,511,616 | 22.51 | **no — new** |
| schedule — **largest kernel argument** | 1,048,576 | 1.05 | **no — new** |
| schedule — **tensor-argument bytes** | 9,028,599,808 | 9,028.6 | **no — new** |
| **DRAM traffic** | — | — | **must not be inferred from these** |

Five rows, five different quantities. The record had two of them and a modelled
third and compared all three. That is the confusion, and it is now named.

---

## 8. What is withdrawn, what is conditional, and where

### Withdrawn

| figure | where it is asserted | status |
|---|---|---|
| **268.4 MB** decay traffic | `bench/HYPOTHESES.md:169`, `:174`; `notes/06-corrections-ledger.md:102`, `:154`; `notes/DECISIONS.md:23` | **WITHDRAWN.** The tensor it charges is not in the program. The *arithmetic* is correct; the *object* does not exist. |
| **290.7 MB** ported total | `bench/HYPOTHESES.md:172`, `:240`, `:545`, `:555`, `:557`; `bench/PLAN.md:240`, `:242`; `notes/06:103`, `:156`; `notes/DECISIONS.md:23`; `bench/FAIR-BASELINE.md:655` | **WITHDRAWN**, same reason. Built by adding 268.4 to 22.3. |
| **48.8×** decay ÷ compulsory | `notes/06:48`, `:102`, `:159`; `notes/DECISIONS.md:23`; `crossover/literature.py:298`, `:563`; `research/PARITY-SCOPE.md` via C-01 | **WITHDRAWN.** A ratio with a zero numerator term. |
| **13.0×** predicted time payoff | `notes/06:48`, `:103`, `:157`; `notes/DECISIONS.md:23`; `bench/HYPOTHESES.md:156` (the H2 title), `:173`; `bench/PLAN.md:240`; `bench/FAIR-BASELINE.md:655` | **WITHDRAWN.** Its premise (bandwidth-bound on 268.4 MB) fails; it was never a measurement, and `notes/06:103` already said so. |
| **H2** as written | `bench/HYPOTHESES.md:156-192`; `bench/PLAN.md` R8 | **REFUTED IN ITS PREMISE, NOT BY ITS OWN TEST.** Its prediction ([233, 349] MB) and its refutation threshold (< 58 MB) are both still unrun. See the P-2 row below. |
| **"the decay dataflow is the actual bottleneck"** | `notes/03-kda.md:120` (ranked #4) | **WITHDRAWN as a bandwidth claim.** It remains true as an *arithmetic* claim: the decay still dominates the 76.8% of GPU time, via 32× too many `exp2` (`notes/04-codegen-comparison.md:213-248`), not via bytes. |
| **P-2** as written | `notes/06:227`; `bench/PLAN.md:240-242` | **MUST BE RE-SCOPED BEFORE IT RUNS.** A passing run would no longer confirm the model; the threshold (< 58 MB) was derived against a term that does not exist. The test should now measure where the traffic *actually* is — the 8,207 re-reads — not whether a tensor reaches DRAM. |
| **H13**'s arithmetic | `bench/HYPOTHESES.md:545-559` | **CONDITIONAL.** Its intensity denominator (290.7 MB) is withdrawn, so 0.81 FLOP/byte and 31× memory-bound are unsupported. `crossover/literature.py` §3 already reports that its **FLOP** side does not reproduce either (0.9802 vs 0.8079), which P-3 flags as the more serious problem. |
| **H1**'s intensity argument | `bench/HYPOTHESES.md:133-138` (0.0076 FLOP/byte, 27,000× memory bound) | **WITHDRAWN.** It computes the intensity of a contraction that reads a tensor no kernel reads. The *predicted end-to-end effect of 1.00-1.05×* may well be right for a different reason — the contraction is fused and does not move the tensor — but the stated mechanism is void. |

### Unchanged — and it is worth saying which

| figure | where | why it survives |
|---|---|---|
| **256×** element ratio | `notes/06:101`; `notes/04-codegen-comparison.md:136-157`; `crossover/claim.py:375-412` | `C²K/(BC·K)` is arithmetic on two *graph objects*, and `crossover/claim.py` correctly adds that the kernel side is `NC` sub-chunks, so the same-scope ratio is 64×. Neither side needs the tensor to exist. `notes/06:61` already scopes it as "zero bytes in memory". |
| **32× more `exp2`** | `notes/04-codegen-comparison.md:213-248` | Counted from generated source. Orthogonal to memory, and now the *only* thing the 256× costs. |
| **8.42 MB / 54.1 MB** | `notes/04-codegen-comparison.md:170`, `:173` | Reproduced to the byte. 8.42 MB is the planner arena (peak live intermediates), which the note under-reads. |
| **12,229** scheduled kernels | `notes/04-codegen-comparison.md:470` | Reproduced exactly (6,188 + 6,041, the two `realize()`s). The unrolled UT transform is untouched by any of this. Note that the **12,090** at `notes/04:416` is a *different counter* — DEBUG=2 launches per iteration, not scheduled kernels — and neither number changes here. |

### Every file carrying the now-conditional claim

1. **`bench/HYPOTHESES.md`** — H1 (intensity), H2 (entirely), H4 (0.81 FLOP/byte,
   31×), H13 (290.7 MB, 31:1). U3 inherits: "the byte side assumes one write + one
   read of each intermediate, which the kernels explicitly do not honour" — that
   is now *measured*, and the answer is that the largest intermediate is 64×
   smaller than assumed.
2. **`notes/06-corrections-ledger.md`** — C-01's 48.8×, §2's 268.4 / 290.7 /
   48.8 / 13.0 / band [233, 349] / threshold 58 MB, P-2, P-3's denominator, and
   §3's `[UNVERIFIED]` item 4 (now partly measured). **C-03's 256× row stands.**
   New rows needed: one for the byte model's premise, one for the two shapes.
3. **`notes/DECISIONS.md` row 17** — the row already says the tensor is "never
   materialised" *and* still lists 48.8× and 13.0× as live. It is internally
   inconsistent today; this note resolves it in favour of the first half.
4. **`notes/04-codegen-comparison.md`** — §2 is **confirmed and strengthened**
   (the 8.42 MB is a planner arena; the tensor is also absent from the schedule,
   which the note did not test). §8 item 7's "33.6 MB is a logical size" stands.
5. **`notes/03-kda.md`** §3 and §2b — the 33.6 MB table and "the actual
   bottleneck" ranking.
6. **`bench/PLAN.md`** — R8, the `< 58 MB` decision rule, and the "H2 refuted →
   stop the KDA work" branch at line 321, whose *conclusion* (re-scope) is right
   but whose *reason* was L2 absorption, not non-existence.
7. **`bench/FAIR-BASELINE.md:655`** — the B1 row citing "H2's 13.0× arithmetic".
8. **`crossover/literature.py:271-313`** — reproduces the byte model and reports
   "all ten reproduce". Accurate about the *arithmetic*, and it should now carry
   the one line that the object being charged does not exist.
9. **`crossover/claim.py:375-412`** — the 256× is cross-scope; **unaffected**,
   and worth citing as the model this note wishes the other figures followed.
10. **`kda/tiny_kda.py:64-83`** (`_causal_decay`, "the single biggest memory item
    in the whole operator") and **`:119-144`** — the source comment that motivates
    the whole study is false as written and is the origin of the 67 MB / 268 MB
    figures. It cannot be edited from here (`kda/` is owned elsewhere), so it is
    flagged.
11. **`kda/NOTES.md`** §6, **`kda/SPEC.md` §4.4, `kda/cost.py:17`** — the
    `B=4 T=4096 H=32 K=128` shape and its `[C,C,K]` model. Correct in their own
    shape; the two shapes are the other half of the confusion.
12. **`kda/tpu/shapes.py:136`** and **`research/TPU.md:64`** — both quote "two
    33.6 MB intermediates" from `notes/03-kda.md:93-94`. `[MODELLED]`, at the
    bench shape, and the object is not real. `research/TPU.md:64` already softens
    the claim in prose ("the *claimed* 67 MB"); the number it softens is still
    quoted at `kda/tpu/shapes.py:136`.
13. **`README.md:17`** — "on KDA it materialises 67 MB of" — **the most visible
    instance of the withdrawn claim**, in the repo's own summary. Withdraw or
    restate.
14. **`bench/selftest.py:306-350`** — declined to pick a side, correctly, and
    printed both ratios rather than choosing. That was the right call and this
    note is the resolution it was waiting for.

### One thing that gets *stronger*

`research/TAXONOMY.md:172` — *"Triton, TileLang and tinygrad all take a function
and compile it. If the function materialises the N×N score matrix, no schedule
removes it."* — is **not** supported for tinygrad here. The function does not
materialise the matrix; the schedule keeps it in registers. That is a
`[MEASURED]` exception to a claim the study treats as its firmest (i), and it is
the finding worth carrying forward: **for a fused pointwise-into-reduction
epilogue, tinygrad's scheduler solves this instance, and the cost of the
function's shape shows up as redundant transcendental evaluation (32× `exp2`),
not as bytes.** The taxonomy's item needs the qualifier "if the function
materialises", and this measurement is the case where it does not.

---

## 9. Files

| file | what it produced here |
|---|---|
| `kda/materialisation.py` | everything in this note. The verdict, the positive control, the three shapes, the CPU/METAL schedule identity, the byte model re-derivation and the error it caught |
| `notes/04-codegen-comparison.md` | the 8.42 MB / 54.1 MB measurements, re-derived and confirmed; §2's "never materialised" |
| `bench/HYPOTHESES.md` | H2's 268.4 / 290.7 / 48.8 / 13.0 arithmetic; H1's intensity argument; H13's 31:1 |
| `notes/06-corrections-ledger.md` | the < 58 MB threshold, P-2, P-3, `[UNVERIFIED]` item 4 |
| `notes/DECISIONS.md` | row 17, which asserts both halves at once |
| `kda/perf_kda.py:90` | the benchmarked shape: B=2, T=512, H=HV=4, K=V=64, C=64 |
| `kda/cost.py:17` | the cost-modelled shape: B=4, T=4096, H=32, K=V=128, C=64 |
| `kda/tiny_kda.py` | the port, the fp32 dtype (`:44`), and the decay expression (`:119-144`) |
