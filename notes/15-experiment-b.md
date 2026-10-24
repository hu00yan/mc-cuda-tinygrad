# 15 — Experiment B: the comparison that does not need a GPU

> **The verdict.** For Kimi Delta Attention the human's construction is one the
> compiler provably cannot reach on any sm_80-or-newer device, and this is
> measurable with no card. The `[B,NT,HV,C,C,K]` decay tensor — 67,108,864 B at
> the bench shape, exactly 64× the largest buffer any kernel in the plan is
> allowed to touch — is absent from the schedule entirely, so it cannot be
> generating traffic; the cost the graph's shape imposes is paid instead in
> arithmetic (2,172 `exp2` per distinct program, four times that on the kernel
> the record has been counting) and in schedule depth
> (`N(C,NT) = 3C² − 4C + 5·NT + 10`, exact, 12,082 kernels at C=64). And the
> compiler's rendering of that plan is **byte-identical across sm_80, sm_86,
> sm_89, sm_90 and sm_100** — one sha256 over the whole program — while the
> hand-written Triton kernel reaches 16 `wgmma` on sm_90a and 32 `tcgen05` on
> sm_100a from the same source file. The gap is a gap in what the construction
> space contains, not in what a scheduler does with it.
>
> **This is a code-and-construction comparison. It is not a speed comparison,
> it is not a performance result, and it must never be read as one.** §3.

Every number below is re-derived by a script in `experiment_b/`, on this host,
with no GPU. None is transcribed from another notes file, from a log, or from a
brief — including the ones that disagree with the record. Where a figure is read
out of a file rather than recomputed, the file is named, sha256-pinned, and
labelled `[ARTIFACT]`. Where something needs a card, it is labelled
`[INFERRED]` or `[NOT-HERE]` and the measurement that would settle it is named.

**Status vocabulary** (identical in code and prose, see `experiment_b/common.py`):

`MEASURED` · an instrument ran here, on this host, on a pinned artifact
`RE-DERIVED` · arithmetic recomputed here from shapes or from source text
`MODELLED` · arithmetic on shapes with the premise stated
`INFERRED` · needs hardware this host does not have; labelled
`ARTIFACT` · read out of a pinned file, quoted with its path
`NOT-HERE` · not measured here, and named what would measure it

---

## 1. The distinction the study kept collapsing, made structural

The word "compare" was carrying two different questions and they have different
availability, different instruments, and different verdicts.

**(a) Same algorithm on both sides.** Hold the algorithm fixed, run both, and
attribute the gap to codegen and scheduling quality. This is a speed
comparison. It needs a card that can run the hand-written baseline at full
strength.

**(b) The human chose a better algorithm and the compiler cannot reach it.**
Hold nothing fixed. The question is not *which is faster* but *can the compiler
arrive at the human's construction at all*. The reachable quantities are:

* **bytes materialised** — a property of the schedule, decided before anything
  is rendered;
* **instruction classes emitted** — a property of the renderer;
* **whether the program expresses the human's form** at all.

None of those is a clock. All three are device-free. `experiment_b/` measures
all three.

### 1.1 Comparison (a) is invalid here, and that is not the same as hard

`experiment_b/b5_arch_gate.py` re-derives, from the vendored baseline's own
source, the three gates that make (a) invalid on every machine this study can
reach. `MEASURED` on a pinned source file; the arithmetic is done in the script.

| gate | source | value | consequence |
|---|---|---|---|
| sm_80 shared-memory gate | `fla/utils/_device.py`, `Backend.AMPERE` | **166,912 B** | `num_stages = [2,3,4]` requires it; below it, `[2,1]` |
| tf32 availability | `IS_TF32_SUPPORTED = capability.major >= 8` | — | below sm_80, `fla` sets the **process-global** `TRITON_F32_DEFAULT = 'ieee'` and every `tl.dot` becomes an IEEE fp32 dot |
| Ada is not enough either | `check_shared_mem('ada')` = 101,376 B | — | Ada clears it, so `BV = [32,64]`, but **not** AMPERE, so `num_stages` is still `[2,1]` — and `num_stages=1` disables the software pipeline on a kernel that streams a `[B,HV,K,V]` state |

The study's only card is a Tesla T4 at 65,536 B opt-in. It fails all three. The
T4 cells in this study's tables are **fla-on-Turing in IEEE fp32 with a collapsed
config space**, not fla. The most instructive line in b5's table is the RTX 4090:
101,376 B, which clears the ADA gate, does **not** clear DEFAULT (102,400) or
AMPERE (166,912), and therefore gets `num_stages = [2,1]`. The card that looks
like the obvious upgrade is the one where the baseline's config space is
silently halved, and no datasheet says so.

An A100, an H100 or a B200 would clear all three. The study has none. That is
why (a) is blocked — and it is blocked *only* because of that, which is the
part the record stopped reading.

### 1.2 …and the ceiling the compiler cannot cross is a table you can read

`experiment_b/b5_arch_gate.py` **executes** `tinygrad/codegen/opt/tc.py:get_cuda`
at each arch rather than quoting it. `MEASURED` on the installed compiler:

```
def get_cuda(arch): return cuda_sm89 if (ver:=int(arch[3:])) >= 89 else cuda_sm80 if ver >= 80 else cuda_sm75 if ver >= 75 else []
```

| arch string | catalog | shapes | largest (N,M,K) | instruction it can reach |
|---|---|---|---|---|
| `sm_75` | `cuda_sm75` | 2 | (8, 16, 8) | `mma.sync.m16n8k8` |
| `sm_80`, `sm_86` | `cuda_sm80` | 6 | (8, 16, 16) | `mma.sync.m16n8k16` |
| `sm_89`, **`sm_90`**, `sm_100`, `sm_120` | **`cuda_sm89`** | 8 | (8, 16, 32) | `mma.sync.m16n8k32` |

There is no `sm_90` row and no `sm_100` row. `cuda_sm89` is `cuda_sm80 + fp8` —
an **Ada** catalog. `get_cuda("sm_90a")` raises `ValueError`, which is why
`tinygrad/runtime/ops_cuda.py:122` builds the string without an arch suffix:

```python
arch=f"sm_{major.value}{minor.value}")
```

So on a real H100 the backend reports `sm_90`, the table returns the Ada
catalog, and the largest tensor-core shape available is `mma.sync.m16n8k32`.
**There is no `wgmma` shape and no `tcgen05` shape in the catalog to select**, so
no amount of scheduling on that card can produce one. `[MEASURED]` for the table;
`[INFERRED]` for the "on a real H100" step, which is a reading of one line.
`b4` confirms the consequence at the output level, and does not need the card.

---

## 2. (b) in its full form

**Claim under test.** *There exists a construction of the KDA operator that a
hand-written kernel reaches and a compiler does not, and this is decidable from
the code rather than from a clock.*

Not the same as: the hand-written kernel is faster. Not the same as: the compiler
is worse. Those are claims about time. This is a claim about the contents of a
construction space, and it has three sub-claims, each with its own instrument:

* **B1 — the decay is not materialised.** The `[B,NT,HV,C,C,K]` tensor never
  reaches device memory in any form: not as an allocator request, not as a
  kernel argument, not as a value written out.
* **B2 — and the cost of the graph that replaces it is arithmetic, not bytes.**
  The redundancy the decay tensor would have absorbed is recomputed. The
  *non-materialisation* is a property of the lowering; the *cost* of the graph's
  shape shows up as `exp2` in registers and as schedule depth, not as traffic.
* **B3 — and the compiler's construction space does not contain the
  architecture-specific instructions the human's toolchain reaches.**

### 2.1 B1. The decay is not materialised

`experiment_b/b1_bytes.py`, on the one emitted CUDA kernel in the repo —
`out/kaggle/20260928-014248-gpu/codegen/kernel000.txt`, 24,968 B, `ARTIFACT`,
sha256 `4de9665c…`, emitted by the real NVIDIA backend on a Tesla T4 (sm_75) at
`B=2, T=128, H=HV=2, K=V=32, C=32`, NT=4.

Everything below is recomputed by the script, including the index ranges, which
are produced by **evaluating the emitted index arithmetic over all 256 threads
of the grid** — not by pattern-matching strings. The script has a small
interpreter for the emitted temporaries and raises on any expression it cannot
evaluate, so a shape it cannot handle produces a refusal rather than a guess.

```
1 __global__  r_2_2_8_8_4_4_32   __launch_bounds__(128)   grid 2 x block 2x8x8

buffer          elements      bytes  stores  loads  identity
data0_4096          4,096     16,384       4      0  [B,HV,K,V]    the live state S
data1_4096          4,096     16,384       0      4  [B,HV,K,V]    the live state S
data2_16384        16,384     65,536       0     32  [B,NT,HV,C,K]
data3_16384        16,384     65,536       0     32  [B,NT,HV,C,K]
data4_4096          4,096     16,384       0     32  [B,HV,K,V]    the live state S
```

* total declared **180,224 B**; largest declared **65,536 B**
* 4 store statements in the entire file, all `float4`, all to `data0_4096` —
  nothing else in the file stores anything
* 100 load statements, all `float4`. Fully 128-bit vectorised, no scalar access
* 3 of 5 buffers are covered exactly, no gaps and no overlaps; the two
  16,384-element buffers are **25% touched** — the kernel reads a 4,096-element
  window of each at indices 6,144..16,383. The working set is *smaller* than the
  signature implies.
* `data0` and `data1` are the same tensor read and written: this is one
  read-modify-write step of the inter-chunk scan's state update.

And the decay tensor, at that shape:

```
[B,NT,HV,C,C,K] fp32 = 2 x 4 x 2 x 32 x 32 x 32 x 4 B = 2,097,152 B
  / largest buffer in this signature  =  32.00x
  / all five buffers in this signature =  11.64x
  in bf16 it is 1,048,576 B, still 5.82x the whole signature
```

`RE-DERIVED`. The conclusion is dtype-robust, and it is about a *signature*: it
is not a measurement of traffic. `NOT-HERE`, and §5 names the measurement.

**Scope, and this is a correction to the record.** The record quotes "32×"
without saying 32× what. There are two exact ratios and they differ by 2×:

| scope | ratio | where measured |
|---|---|---|
| decay ÷ largest buffer in **one kernel's** signature | 2,097,152 / 65,536 = **32×** | b1 |
| decay ÷ largest buffer in **the whole plan** may touch | 67,108,864 / 1,048,576 = **64×** | b2 |

Both are correct. Neither is a restatement of the other, and a document that
quotes one while calling it *the* ratio is a scope error rather than an
arithmetic one.

### 2.2 B1 at the scale that matters: the whole plan

`experiment_b/b2_schedule.py`. This is the instrument the record named as the
one that would close the CUDA gap — "is there a bucket at 67,108,864 B?" — and it
runs with **no GPU at all**, because `memory_plan_rewrite` is a pure UOp rewrite
over device *strings* and `Context(DEV="CUDA")` short-circuits device discovery
so no `CUDADevice` is ever constructed. `b2` proves both properties in code at
the top of section 1 and fails if either stops being true.

At the bench shape `B=2, T=512, H=HV=4, K=V=64, C=64` (NT=8):

```
          nbytes     count   what it is in the port
             256    19,968   [C]  an eyeT row
          16,384    24,466   [B,NT,HV,C,K]  g_cum / tq / tk / gam
         131,072       109   [B,T,HV,K]  q / k / g
       1,048,576     8,181   [B,NT,HV,C,C]  Z / Btri / A / M
                                        <- the largest legal argument
  distinct argument sizes: 4   argument slots: 52,724   (a kernel may hold
  several, so this is not the kernel count)      total argument bytes: 8,998,649,856
  the decay bucket 67,108,864 B  PRESENT: False   count: 0
```

`MEASURED`, device-free. **Four distinct argument sizes, and the decay is not
one of them.** A buffer carries traffic only if a kernel reads or writes it, and
a kernel can only touch buffers in its argument list. That argument does not
depend on a cache, an L2, a card, or a backend's memory model. So at this shape
**the decay tensor cannot be generating DRAM traffic in this plan**, and no
bandwidth-bound story for KDA on tinygrad can be told from these shapes.

**And the census reproduces the GPU log to the exact kernel.** `out/kaggle/20260928-014248-gpu/logs/perf_kda.log`, `ARTIFACT`, sha256 `600c6dc5…`:

| | census, no device | T4 log, `DEBUG=2` | match |
|---|---|---|---|
| kernels per call | **12,082** | **12,082** | yes |
| `r_` (carries a REDUCE) | **7,984** | **7,984** | yes |
| `E_` (elementwise) | **4,098** | **4,098** | yes |

Three numbers, two instruments, one with a card and one without, on a quantity
neither can perturb — the count is decided by the scheduler before anything is
rendered. That is what makes it safe to reason about offline.

**Two corrections fall out of the same table.**

1. **12,082 is a CUDA number; 12,090 is a METAL number.** b2 runs three
   backends: CUDA 12,082 (7,984 / 4,098) and METAL and CPU both 12,090
   (7,984 / 4,106). The +8 is the buffer-count-limiting rewrite, keyed on device
   string with no CUDA entry, so the rewrite that merges the elementwise tail
   never runs on CUDA. A document that quotes "12,082" without naming the
   backend has not made a claim.
2. **The T4 log's own device label is wrong.** It prints
   `measured (Metal, B=2 T=512 K=64 C=64)`, but that string is a hard-coded
   f-string at `kda/perf_kda.py:143`, and the run's own `SUMMARY.md` records
   `Tesla T4, 15360 MiB, 7.5`. The 12,082 and the 215.9 ms were measured on the
   T4, on CUDA. This is a small error with a large consequence, because it makes
   the only hardware count in the repo look like it came from a backend the study
   does not use.

**And the schedule is arch-independent.** Re-running the census at
`DEV=NULL:CUDA:<arch>` for sm_75 / sm_80 / sm_89 / sm_90 / sm_100 gives an
identical kernel count and an identical buffer histogram on all five — **one
fingerprint**. So the plan does not depend on the architecture, which is why the
census needs no card and no card architecture. It is also the first half of the
answer to §5.

### 2.3 B2. The cost is arithmetic, not bytes — and this inverts the framing

This is the finding that inverts the study's own reading of its evidence, so it
is worth being exact about what it does and does not say.

`b1`, same artifact:

```
exp2 sites in the file                        128
... assigned to a `float aluN` register       128   all float: True   all aluN: True
... distinct destinations                     128
... inside a store statement                    0
... used as an ADDRESS operand (a re-read)      0
... local register arrays `float bufN[k]`        0
__shared__ / __syncthreads                     0 / 0
```

The kernel writes 16 floats and evaluates 128 `exp2` — 8 per written element —
and **every one of them is dead at the end of the statement that computed it.**
The decay *value* is not stored.

The record has been reading this as "the decay is fused, therefore the cost is
bytes and the fix is more fusion." It is the other way round. The graph-object
ratio is real and survives — `C²/BC = 64²/16 = 256` at
`kda/codegen_diff.py`'s gram shape, with `K` cancelling, comparing the port's
`[C,C,K]` decay against the hand-written kernel's `[BC,K]` per sub-chunk
`[RE-DERIVED, MODELLED; the kernel side is read from fla's source, not
executed]`. But **that ratio is a property of the graph, and the non-
materialisation is a property of the lowering.** The two live at different
levels. A 256× larger graph object that never becomes a byte does not cost 256×
anything in traffic; it costs whatever the recomputation costs, and here the
recomputation is `exp2` in registers.

So the cost the graph's shape imposes is **redundant `exp2`**, and the
over-fusion literature points the *wrong way* for this operator: the thing that
is missing is not fusion, it is *sharing of a computation that a fused kernel
never needs to do twice*. Fusing more of this graph would fuse more `exp2` into
more kernels, not fewer.

And here is the correction that makes b1's number honest rather than merely
right:

> **The 128 is an sm_75 number and it is an understatement on every card the
> study would want.** Re-rendering the *same graph* offline at
> `DEV=NULL:CUDA:<arch>` and counting: the state-update kernel emits **128 `exp2`
> at sm_75 and 512 at sm_80+** — 4×. Whole-program: 2,756 `exp2` across 95
> distinct sources at sm_75, 2,172 at sm_80+, with `mma.sync` going 0 → 10.
> `b4` renders this in a fresh process per arch and **fails loudly if any arch
> produces an empty program**, because an empty program has zero of every
> instruction class and would make the next result appear true for the wrong
> reason.

The split is exactly at the sm_75 / sm_80 boundary, and it changes the
**operator's own arithmetic**, not just its instruction selection: sm_75 has no
tf32 tensor core, so the matmuls take a different route and the fusing pattern
around the decay's `exp2` changes with it. The one emitted artifact in the repo
happens to be from the one card in the study's history that the baseline cannot
use. `b1` is right about its artifact and must not be quoted as arch-neutral.

### 2.4 B3. The operator is not fused, and "12,082" is a fact about a shape

The record has been reading one sentence two ways. "The decay is fused within
each kernel; the operator is not" contains two claims about two levels, and
they have different evidence and opposite implications:

* **the decay, per kernel** — `b1`. True, and the cost is `exp2` (§2.3).
* **the operator's schedule** — `b2`. Not fused: 12,082 sink kernels.

`experiment_b/b3_kernel_count.py` measures the second across a grid of
`(C, NT)` with no device, and the count separates **exactly** — not "within
0.041%":

```
N(C, NT) = 3*C^2 - 4*C + 5*NT + 10        for NT >= 2
N(C, 1)   = 3*C^2 - 4*C + 13
```

Zero residual at every shape in the grid. The −2 step at NT=1 is not a fudge:
with one chunk the carried state is identically zero, so `v_new = u − w@S` has no
`w @ S` product to emit and the two kernels that carry it do not exist. A step
function at NT=1, not a term in C — which is the difference between a law and a
patch.

**Read the two terms as two different failures**, because they are:

| term | what it is | scaling | depends on sequence length? |
|---|---|---|---|
| `3C(C−1) + (12−C)` | the UT transform: step 4 is a forward substitution of C sequential row solves, unrolled because the trip count is known | **quadratic in C** | no |
| `5·NT + 5` | the inter-chunk scan: step 7 carries a `[B,HV,K,V]` state across NT chunks, also unrolled | **linear in NT** | yes |

So "the operator is not fused" is two non-fusions with different scaling that the
study has been quoting as one number. Raising `T` probes the linear term.
Raising `C` probes the quadratic one. **They need different fixes.** And the
study's own choice of `C` is already visible in the formula. T is `C·NT`, so
these rows trade the C term against the NT term — which is precisely what
choosing a chunk size *is*:

| C | NT | T | kernels | vs C=64 |
|---|---|---|---|---|
| 16 | 32 | 512 | 874 | 0.07× |
| 32 | 16 | 512 | 3,034 | 0.25× |
| 48 | 10 | 480 | 6,780 | 0.56× |
| **64** | **8** | **512** | **12,082** | 1.00× |
| 96 | 5 | 480 | 27,299 | 2.26× |
| 128 | 4 | 512 | 48,670 | 4.03× |

Halving `C` to 32 at fixed `T` removes **9,048 kernels**, a **3.98× reduction**,
entirely through the `C` term. That is the largest single lever available
without touching the algorithm, and it is invisible to any benchmark that holds
`C` fixed and varies `T` — which is what the study has been doing.

**And the fit the record quotes is a curve fit, not a law.** `2.99·C(C−1) +
5.06·NT − 6.8` has a residual; `b3` re-fits that basis over its own grid and
reports the residual next to the closed form. A residual is the fit reporting
that the basis is wrong. The closed form has none, because it *is* the
structural decomposition: one coefficient per unrolled sequential construct, and
the constructs are countable in the source. `MODELLED` and `RE-DERIVED` are
different epistemic grades and the record has been quoting the second as the
first.

### 2.5 B3. The instruction-class matrix

`experiment_b/b4_codegen.py`. Both sides on the same axes, both device-free: the
human side is a pinned PTX file produced by `triton.compile` AOT with an explicit
`GPUTarget` and the driver verifiably absent; the compiler side is re-rendered
here at each arch with `DEV=NULL:CUDA:<arch>`.

```
class                            sm_90a           sm_100a    tinygrad sm_75    tinygrad sm_80
-------------------------------------------------------------------------------------------------
wgmma.mma_async                      16                 0                 0                 0
wgmma.wait_group                      4                 0                 0                 0
wgmma.commit_group                    4                 0                 0                 0
wgmma.fence                           4                 0                 0                 0
tcgen05.mma                           0                32                 0                 0
tcgen05.ld                            0                 8                 0                 0
tcgen05.wait                          0                16                 0                 0
mma.sync                              0                 0                 0                10
ldmatrix                              0                32                 0                 0
stmatrix                             12                 0                 0                 0
cp.async                              6                 9                 0                 0
cp.async.commit_group                 4                 6                 0                 0
mbarrier                              0                24                 0                 0
ex2.approx                            3                66             2,756             2,172
ld.global.b16                       160               320                 0                 0
ld.global.v4                          0                 0             1,860             1,028
st.global.v4                          0                 0               126               106
__shared__                            0                 0                 3                 3
__syncthreads                         0                 0                 3                 3
```

(The `tinygrad` columns are representative; the full run prints sm_80, sm_86,
sm_89, sm_90 and sm_100, which are identical to sm_80 in every row.)

**The human construction is arch-adapted.** The same Triton source file emits 16
`wgmma` and 12 `stmatrix` on sm_90a, and 32 `tcgen05.mma` + 8 `tcgen05.ld` + 32
`ldmatrix` + 24 `mbarrier` on sm_100a. Two tensor-core generations, chosen by the
toolchain. The frontend IR (`*.ttir`, 21,231 B, pinned once) is *identical*
between the two runs, so the architecture reaches the choice as a target.

**The compiler construction is arch-invariant.** The entire rendered program —
all 95 distinct sources concatenated — has **one sha256 across sm_80, sm_86,
sm_89, sm_90 and sm_100**:

```
901110459bae2ddf3bd4e44b3d06830e77e94a300d8cb9591005395db06cc79f
```

and it contains **zero of the eight arch-specific instruction classes** at every
architecture. Its entire tensor-core vocabulary is
`mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` — the *smallest* tf32 shape
in the catalog, on the five architectures that emit it, and nothing at all on
sm_75.

### 2.6 The findings, tagged

The study's own vocabulary, one row each, with the evidence:

| # | finding | tag | evidence |
|---|---|---|---|
| 1 | `wgmma`, `tcgen05`, `stmatrix`, `ldmatrix`, `mbarrier`, `cp.async`, `tensormap` | **(i) cannot express** | 0 occurrences in the whole rendered program at every arch. No renderer path and no catalog entry would produce them (`b4` §3, `b5` §3) |
| 2 | the `mma.sync` that *is* emitted sits at the smallest tf32 shape in the catalog, `m16n8k8`, while `m16n8k16` and `m16n8k32` are available | **(ii) can express but does not find** | the mechanism exists in `tc.py`; the search does not reach it for this operator (`b4` §3) |
| 3 | the Hopper and Blackwell renderer paths | **(iii) not yet implemented** | one `asm("mma.sync...")` template in `renderer/cstyle.py`; `get_cuda` has no sm_90/sm_100 row; `ops_cuda.py:122` cannot pass `sm_90a` because the table does `int(arch[3:])` (`b5` §3) |
| 4 | — | **(iv) does not exist for this architecture** | **nothing carries this tag.** Every class above exists on both sm_90 and sm_100, and the human side emits them. Tag (iv) would mean the compiler was being blamed for something the hardware cannot do; that is not what happened |
| 5 | the decay is absent from the plan by 64× | **neither — it is a `MEASURED` non-existence** | b1, b2. The four tags are for *gaps*; this is a present absence |

**Tag (iv) being empty is the load-bearing part.** The distinction the study has
been blurring is between "the compiler cannot express this" and "the compiler
chooses not to". Findings 1 and 3 are the first; finding 2 is the second. Only
the second is a claim about quality, and it is a much weaker one. The evidence
here is almost entirely (i) and (iii): **gaps in the construction space, not
judgements about it.**

### 2.7 The baseline is not free, and this is the honest counterweight

If the (b) verdict is "the human reaches instructions the compiler cannot", the
obvious objection is that the human kernel is bad. On the evidence available, it
is — and this belongs in the primary result, not in a footnote.

The same AOT compile that shows the human reaching `tcgen05` shows a kernel
with **zero `.local` declarations, zero `ld.local` and zero `st.local` in its
PTX, on both architectures** (`b4` §4) — and yet a recorded `ptxas -v` run for
the sm_90a cubin reports **380 B of spill stores, 372 B of spill loads, a 376 B
stack frame at the 255-register ceiling**. `ARTIFACT`, quoted: `ptxas` is not
installed on this host, so `b4_codegen.py --verify-spill` **exits nonzero**
rather than letting it pass as verified. `NOT-HERE` to re-derive.

The two facts compose, and the composition is the lesson. **`ptxas` introduces
the local traffic during register allocation, after the PTX exists** — so it is
absent from the PTX by construction. And `cuobjdump -res-usage` reports the local
*frame*, a declared array, and correctly reports 0 for this kernel, because
there is no declared array. A study that had trusted `-res-usage` would have
certified the hand-written baseline as spill-free. `ptxas -v` is the only
authoritative source here, and it is the one that was not run.

Occupancy, re-derived arithmetic on a datasheet figure that cannot be checked
here (`INFERRED`, 65,536 32-bit registers per SM): 255 × 128 = 32,640 per block,
⌊65,536 / 32,640⌋ = 2 blocks/SM = 8 warps of 64 = **12.5%**.

So the hand-written reference, on the one architecture where its own toolchain
emits the newest tensor-core instructions, runs at the occupancy that
`notes/04-where-compilers-stop.md` names as the mechanism by which a *correct*
kernel collapses to a slow one. The study's fast reference carries the defect
the study attributes to codegen limits.

**This does not make the (b) verdict wrong, and it does not make it right about
merit.** It makes it a statement about **reachability**, and it is the reason
§3 has to be as narrow as it is.

---

## 3. What this licenses, and what it must never be used for

### 3.1 The line

**This note is a code-and-construction comparison. There is no timing in it, no
profiler ran, and no kernel executed while producing any of it.** The only
hardware number quoted anywhere in `experiment_b/` is the T4's `DEBUG=2`
`kernels per call` and `GPU time`, and both are used as *counts the census
reproduces*, never as costs.

### 3.2 What it licenses

* **That the KDA decay tensor is not materialised**, at the shapes measured, in
  the plans measured, on the compiler measured. Byte accounting, schedule
  census, and instruction-level census, each with its own instrument.
* **That the operator's schedule is not fused**, with the count given as a
  function of `(C, NT)` and therefore usable at other shapes — 12,082 is a
  statement about C=64 and T=512, and nothing else.
* **That the redundancy the decay tensor would have absorbed is paid in
  arithmetic**, with the mechanism named (`exp2` recomputed into registers) and
  its arch-dependence bounded (4× understated on the captured sm_75 kernel for
  every sm_80+ card).
* **That the compiler's rendering is arch-invariant** across sm_80..sm_100, and
  **that the reason is a table with no sm_90 and no sm_100 row** — which is a
  concrete, dated, checkable claim about a specific commit of a specific
  compiler, and one that a future release could change.
* **That the hand-written baseline reaches the newest tensor-core instructions on
  both Hopper and Blackwell, and carries a 380 B register spill at sm_90a.**

### 3.3 What it must never be used for

* **Any speed claim, ratio, or ordering.** "The compiler is N× slower" is not
  licensed by anything here. Not "N× slower than the hand-written kernel" — the
  two were never compared. Not "the hand-written kernel is faster" — it was
  never timed here either. The only number in the repo that looks like a
  comparison, 215.9 ms, is 12,082 **event-bracketed, host-synchronised** launches
  from a `DEBUG=2` run in which `engine/realize.py:318` makes `wait` true, so
  every launch does a create/record/synchronise/destroy event pair and the host
  blocks. It is an **upper bound that measures its own harness at least as much
  as it measures the kernels.** It is not a cost.
* **Any statement about DRAM traffic.** A kernel *signature* bounds what a kernel
  *may* touch. `dram__bytes.sum` measures what it *did*. This note has the
  first and not the second, and §4 is about getting the second.
* **Anything about what the hardware achieves.** Occupancy, achieved bandwidth,
  cache behaviour, stall reasons, tensor-core utilisation, and SASS — the last
  because the compiler side of the matrix is *pre-ptxas*. "Which families of
  instruction each construction can reach" is well-posed and answered. "Who
  issues more instructions" is not well-posed across a post-ptxas/pre-ptxas gap,
  and is not answered.
* **Generalising the captured kernel.** One kernel out of 2,974, from one
  capture, on sm_75. Its emitted name `r_2_2_8_8_4_4_32` is a *shape
  signature, not an identity*: re-rendering the same graph offline produces a
  different kernel with the same name at every arch including sm_75. Do not key
  anything on it. The 128 `exp2` is a fact about that kernel; the 2,172 is a
  fact about the distinct-source program; both are lower bounds on the launched
  total, and the plan size is `b3`'s job.
* **Merit.** (b) is about reachability. Tag (iv) is empty precisely because
  nothing here is "does not exist for this architecture" — the architecture can
  do all of it. Whether *doing* it is worth more than the 380 B spill the human
  kernel pays is a performance question this study cannot currently ask.

### 3.4 The failure mode this is written to prevent

`notes/06-corrections-ledger.md` §2 records a table headed:

> *"the study's headline number is not a performance number"*, and had been read
> as one in at least four files.

The same ledger's §3 records that a re-derivation pass produced four apparent
mismatches that turned out to be bugs **in the checking script**, and that the
delivered byte model survived all thirteen re-derivations. So the study has both
a documented habit of over-reading its own numbers **and** a documented habit of
finding errors by re-deriving them.

Those two habits are compatible, and they are the reason `experiment_b/` exists
in this shape: every figure re-derived, every regime labelled, every artifact
pinned so a changed file raises instead of printing a stale number. The pattern
to avoid is specific and it is not hypothetical: **a passing correctness gate
read as performance evidence.** `test_tiny` passing 17/17 against a float64
reference says the port computes the right function. It says nothing about how,
how much memory, how many launches, or how fast. The same is true, less
obviously, of "the decay is not materialised": that is a *correctness-shaped*
claim about code structure, and it is one short step from being read as "so
this is fast now" or "so this is not the problem". It is neither. It means the
cost of this operator is **not** a memory-materialisation cost, and the study
therefore has no memory cost to point at any more — it has a schedule and a
transcendental count, and no measurement of either.

---

## 4. The one experiment that would move (b) from construction to measured cost

**`dram__bytes.sum` for one `kda_chunk` call, fp32, at the bench shape.**
`experiment_b/b6_next_measurement.py` re-derives the band and the threshold and
prints the command.

### 4.1 Why this one

Three reasons, in order of how much they matter.

**It is the only measurement that can overrule this note.** Every byte claim
above is an argument from signatures. It is a strong argument — a kernel cannot
touch a buffer it was not passed — and it is still an argument. `dram__bytes.sum`
is the measurement of the quantity being bounded, and it is the only number the
study could produce that would make the headline wrong. A study whose central
claim cannot be falsified by any measurement it could actually run is not a
study; it is a position.

**The study already pre-registered this falsification condition, with a band
and a threshold, and then never executed it.** `notes/06-corrections-ledger.md`
§2 records it: test = `dram__bytes.sum` for one KDA call, fp32, at the
`perf_kda.py` shape; predicted band ±20%; refutation threshold 20% of the
prediction; and an explicit warning that the measurement **must** be fp32
because at bf16 one decay tensor is 33.55 MB, inside L2, and a bf16 run would
report a number that *looks* like the refutation while being an artefact of the
dtype. Re-deriving that band here is not new work; it is `b6` recomputing it
from shapes:

```
one [B,NT,HV,C,C,K] fp32                        67,108,864 B
decay traffic, 2 tensors x (1 write + 1 read)  268,435,456 B
8 non-decay intermediates, 2 accesses each      2,097,152 elements
everything else                                 22,282,240 B
  of which compulsory I/O + state                5,505,024 B
model total                                    290,717,696 B = 290.72 MB
whole-operator fold ratio                               13.05x
decay / compulsory                                    48.76x
PREDICTED BAND            [233, 349] MB  (total +/- 20%)
REFUTATION THRESHOLD        <  58 MB      (20% of the predicted total)
```

`MODELLED`. A pre-registered refutation that was not run is a worse state than a
missing measurement, because the missing measurement does not pretend to have
been checked.

**It is one command on a shape that has already been benchmarked.** Not a
research project.

### 4.2 The command

fp32, not bf16 — that is not a preference, it is the whole test, as above.
`--replay-mode application` because `dram__bytes_read/write` are single-pass
counters: ncu does not need to save and restore kernel state 12,082 times, and
kernel replay would cost hours for information this metric does not contain.

```sh
ncu --csv --target-processes all --replay-mode application \
  --metrics dram__bytes_read.sum,dram__bytes_write.sum \
  --log-file ncu_kda_dram.csv -- python -c "
import sys, numpy as np
sys.path.insert(0, 'kda')
from tinygrad import Device
import test_ref as T, tiny_kda
print('device', Device.DEFAULT, file=sys.stderr)
B,Tt,H,HV,K,V,C = 2,512,4,4,64,64,64
q,k,v,g,beta = T.rnd(B,Tt,H,HV,K,V,seed=0,l2n=True)
o,S = tiny_kda.kda_chunk(q,k,v,g,beta,chunk_size=C)
print('one kda_chunk call, B=2 T=512 H=HV=4 K=V=64 C=64', file=sys.stderr)
"
```

`b6 --emit-cmd` prints exactly this.

### 4.3 What it costs

`INFERRED` — no run of it exists, and it is labelled as an estimate throughout.
Built from the repo's own logs.

| term | cost | basis |
|---|---|---|
| rental | one A100/H100 for a couple of hours | the only real cost |
| CPU | ~5 s Python + ~15 s scheduler, from the replay log | unchanged by ncu |
| device | ~216 ms of event-bracketed GPU time without ncu; same order with | `perf_kda.log` |
| ncu per-kernel overhead | **unknown, and it decides the wall clock.** expect 1–10 µs × 12,082 = 12–120 s | `INFERRED` |

**Minutes, not days.** The reason the study still has no cost number is not that
this is expensive.

If it turns out to be hours: it is still the cheapest informative measurement
available, and §4.5's fallback is cheaper.

### 4.4 The decision each outcome forces

| outcome | means | then |
|---|---|---|
| **in band**, 233–349 MB | the decay **does** reach DRAM | b1 and b2 are right about the signatures and wrong about the conclusion. That is *possible* — a kernel argument need not be a whole tensor, and a 12,082-kernel plan can stream a window of the decay repeatedly — but the traffic story survives and the over-fusion reading is live again. Next: per-kernel-name `dram__bytes` attribution to find **which** kernels carry it. The next lever is bandwidth, not fusion. |
| **below threshold**, < 58 MB | the decay does **not** reach DRAM | the model is refuted, `b1`/`b2` are confirmed by measurement rather than argument, and the headline stops being a memory claim and becomes a schedule claim: 12,082 launches, 2,172 `exp2` in registers, and the fix is the scheduler or the API. Different owner, different pass, different estimate. **This is the outcome the construction-level evidence favours, and the one most worth wanting to know.** |
| **in between**, 58–233 MB | partial residency; the pre-registered threshold was not sharp | neither story confirmed, and the study has learned its own falsification condition was mis-set. The fix is a sharper discriminator, not a rerun: attribute `dram__bytes` per kernel name so the traffic is *identified* rather than totalled, and re-register against that. |

In no branch is the right response to stop. That is the property of a
pre-registered condition worth having.

### 4.5 The two follow-ups, in cost order

1. **Which kernels carry the traffic**, if the total is ambiguous. 512 of 12,082
   launches, per kernel name:
   `--launch-count 512 --launch-skip 0`. 4.2% of the plan. It answers the
   attribution question on its own if the per-kernel distribution is peaked —
   which it should be, because the argument-size histogram has only four
   distinct values and **8,181 of the plan's 52,724 argument slots share the
   largest one**, 1,048,576 B.
2. **Whether 12,082 launches is expensive.** Either `nsys profile --stats=true
   -t cuda` on the same driver, or, with no profiler at all, three lines:
   re-run `kda/perf_kda.py` with `DEBUG=0` and time `realize()` with a single
   `Device[...].synchronize()` at the end. If the wall time of the realize is
   near the event-bracketed figure, the kernels are launch-bound; if it is an
   order of magnitude larger, the 215.9 ms was measuring the event wrapper. This
   is the difference between "the operator is slow" and "the operator is 12,082
   launches", and it needs no rental beyond the one already required.

---

## 5. What the missing hardware would and would not add

**Plainly: an H100 or a B200 would add wall clock and cost data, and would
change no (b) verdict in this note.** That is a claim with a specific structure,
so it is worth breaking down which finding rests on what.

### 5.1 Arch-independent — hardware cannot move these

| finding | why it cannot move |
|---|---|
| the decay is absent from the plan (64×) | `b2` re-runs the census at sm_75/80/89/90/100 and gets **one fingerprint**: identical kernel count, identical buffer histogram. The plan does not depend on the arch. |
| the byte accounting | a property of the emitted source and the shape. The source is emitted by the renderer; the shape is the shape. |
| `N(C,NT) = 3C²−4C+5NT+10` | decided by the scheduler before anything is rendered. `b2`'s arch sweep is the direct evidence. |
| the rendered program is byte-identical sm_80..sm_100 | **this is the direct answer to "what would an H100 add"**. It is not that we could not test it — we *did* test it, at sm_90 and sm_100, offline, and the answer is one sha256. |
| `get_cuda` has no sm_90 / sm_100 row | executed on the installed compiler. A new release could add rows; an H100 cannot. |
| the decay is fused within each kernel, the operator is not | per-kernel vs graph-level; the graph level is the arch sweep. |
| the 380 B spill and 12.5% occupancy in the baseline | a `ptxas` output at sm_90a. It is the *baseline*'s arch, and it is quoted from a recorded run. An H100 would show the same: this is what `ptxas` did. |

### 5.2 Arch-dependent — and it does not touch a verdict

Two, both on the compiler side, and both **strengthen** rather than weaken the
finding:

* **the sm_75 / sm_80 split.** The only real arch-dependence in tinygrad's
  output, and it moves the operator's *arithmetic* (2,756 vs 2,172 `exp2`;
  `mma.sync` 0 vs 10) rather than its reachability. It is a boundary in the
  tensor-core catalog, and it sits below every card the study would use. Its
  effect on this note is to make the captured sm_75 exp2 count an
  **understatement** for sm_80+.
* **which catalog is selected.** `sm_90` and `sm_100` both select the Ada
  catalog, so an H100 and a B200 render identically here. That is the finding,
  and it is the same on both cards.

### 5.3 What the hardware *would* add

Exactly two things, and they are the two things §3.3 refuses to license:

1. **Wall clock**, for both constructions, on a card that clears fla's gates
   (sm_80+ with ≥ 166,912 B). That makes comparison **(a)** valid, and (a) is a
   different question from (b).
2. **Cost data for the (b) findings**: `dram__bytes.sum` (§4.1); achieved
   tensor-core utilisation (`sm__pipe_tensor_op_hmma_cycles_active`,
   `sm__inst_executed_pipe_tensor` on Hopper); occupancy and achieved bandwidth
   for tinygrad's own kernels; and SASS, which closes the pre-ptxas side of the
   instruction matrix.

### 5.4 The claim this changes

**"The study is blocked, pending hardware."**

That is a claim about the study's *scope*, and it was drawn around the wrong
comparison. The correct statement is:

> The study's **(a)** comparison is blocked pending hardware, and its **(b)**
> comparison is complete, with every figure re-derived on a CPU and every
> artifact pinned. The (b) results are the study's primary finding, because
> they are the ones it can currently support, and they say something (a) could
> not: the gap is in what the construction space **contains**, not in what a
> scheduler does with it — so the fix is a compiler capability, not a tuning
> pass, and a benchmark sweep would not have found it.

That last clause is the operational payoff and it is worth stating plainly,
because it inverts the study's own next step. The study's implied plan was to
tune. `N(C,NT)` says the largest available lever is `C` — 3.98× from halving
it — and `b5` says the next lever after that is a catalog entry, which is a
compiler patch, not a benchmark.

---

## 6. Corrections this note makes to the record

Every one of these was found by re-deriving, and each is reproducible from
`experiment_b/`.

| # | the record says | re-derived | where |
|---|---|---|---|
| 1 | "256×", unqualified | two exact ratios: **32×** against one kernel's largest buffer, **64×** against the largest argument the whole plan may touch | `b1` §2.1, `b2` §2 |
| 2 | "12,082 sink kernels" | CUDA = 12,082; METAL = CPU = **12,090** (7,984 / 4,106). The number needs a backend | `b2` §3 |
| 3 | the T4 log reads `measured (Metal, ...)` | hard-coded f-string at `kda/perf_kda.py:143`; that run was a **T4 on CUDA** | `b2` §4 |
| 4 | "the decay is fused, so over-fusing is the problem" | the decay *is* fused per kernel, and the cost is **redundant `exp2` in registers**, not bytes. More fusion would fuse *more* `exp2`, not fewer | `b1` §4 |
| 5 | the captured kernel emits 128 `exp2` | true **at sm_75**; the same kernel shape emits **512** at sm_80+, and the whole program goes 2,756 → 2,172 | `b4` §2 |
| 6 | `N ≈ 2.99·C(C−1) + 5.06·NT − 6.8`, "max residual 0.041%" | an exact law exists: `N = 3C²−4C+5·NT+10` (NT≥2), `3C²−4C+13` (NT=1), **zero residual**. The quoted figure is a curve fit, and a residual is the fit reporting the basis is wrong | `b3` §1, §5 |
| 7 | `12,082` as a property of KDA | a property of `C=64, T=512`. Quadratic in `C`, linear in `NT`: two failures, and only one depends on sequence length | `b3` §2 |
| 8 | "the operator is not fused" | two claims merged: the decay **is** fused within each kernel; the operator's **schedule** is not. Different evidence, opposite implications | `b2` §6, `b1` §4 |
| 9 | the hand-written baseline, checked with `cuobjdump -res-usage` | the reference PTX has **zero** `.local`, `ld.local`, `st.local` on both archs; `ptxas` introduces the spill *after* the PTX, which is **why** `-res-usage` reports `LOCAL:0` and would have hidden it | `b4` §4 |
| 10 | a 4090 "clears the default gate" | 101,376 < 102,400, so it **does not** clear DEFAULT; it clears ADA, and still gets `num_stages=[2,1]` | `b5` §2 |
| 11 | `9,028,599,808 B` argument bytes | that is the **METAL** figure. CUDA is **8,998,649,856 B** | `b2` §2 |
| 12 | the operator's byte model, 290.7 MB | re-derives to **290,717,696 B**, band [233, 349] MB, threshold 58 MB, fold ratio 13.05×, decay/compulsory 48.76×. The model is sound. **It was never tested**, and the pre-registered test for it was never run | `b6` §2 |

Item 12 is the one that matters most. The model is arithmetically correct and
was correctly re-derived thirteen times over. It was still **never measured**,
and the condition that would have refuted it was written down and not executed.
Being right about a model and having tested it are different properties, and only
one of them is why this study can be trusted.

---

## 7. Reproducing this

```sh
cd <repo>
.venv/bin/python experiment_b/run_all.py --fast    # ~8 min
.venv/bin/python experiment_b/run_all.py           # full grid
```

No GPU, no flags, no arguments. The runner **refuses to start if a GPU is
visible**, because a green run on a machine with a card would no longer be
evidence that the questions are device-free: `Device.DEFAULT` could resolve to a
real CUDA device somewhere and change what `DEV="CUDA"` means. It also checks
every artifact pin before running anything, and stops if one has drifted, because
a stale number printed confidently is worse than a crash. Where a result depends
on a file the study cannot regenerate, the script that needs it **exits
nonzero** rather than letting the figure pass as verified — `b4 --verify-spill`
is the working example.

| script | settles |
|---|---|
| `b1_bytes.py` | the byte accounting, index ranges evaluated over the whole grid |
| `b2_schedule.py` | the device-free CUDA schedule census; is there a 67,108,864 B argument |
| `b3_kernel_count.py` | `N(C,NT)`, and the two terms inside it |
| `b4_codegen.py` | the instruction-class matrix, both sides; `--verify-spill` |
| `b5_arch_gate.py` | why (a) is invalid here; why the compiler's ceiling is a table |
| `b6_next_measurement.py` | `dram__bytes`: the command, the cost, the decision each outcome forces |
| `common.py` | shapes, pins, the status vocabulary, the device-free harness |
| `repin.py` | deliberately re-pin an artifact, after re-deriving what depended on it |

The one thing this directory does not do is produce a number where none can be
produced. Everything above was re-derivable on a machine with no GPU, which is
the finding. The one thing it cannot do is give (b) a cost, and §4 says what
that would take, what it would cost, and what each answer would mean.
