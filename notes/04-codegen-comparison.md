# TinyGrad's generated code, next to the hand-written kernel

The deliverable the study was set up to produce: a side-by-side of what TinyGrad
actually emits for the KDA decayed key-key gram against the Triton kernel a human
wrote for the same operation, and what the difference costs.

Six findings. Each carries the number behind it and the file that produced it.
Each is tagged **(i)** the compiler cannot express it, **(ii)** it can express it
but does not find it, **(iii)** not yet implemented — the three-way split from
`research/TAXONOMY.md` §C1, applied per named system.

| # | finding | number | tag |
|---|---|---|---|
| 1 | the decay intermediate | 256× as a graph object, **never materialised**; 32× more `exp2` | **(i)**, universal |
| 2 | no tensor-core MMA in that contraction | 0 MMA tokens in 96 generated lines; MMA present elsewhere in the same graph | **(ii)**, TinyGrad 0.14.0 |
| 3 | kernel count | 12,090 launches; 99.5% from one `C=64` loop | **unrolling**, not a defect |
| 4 | the schedule is cached, the lowering is not | 18% build / 82% lowering, per warm call | **(i)**, TinyGrad-specific |
| 5 | FMA | 0 of 8 M triples differ from exact FMA; barriers buy 0 of 41,472 pixels | universal, documented |
| 6 | float32 `/` | `a/a != 1` in 30,748 of 200,000 on METAL; no knob; **scope untested on NVIDIA** | **(i)**, TinyGrad defect |

**Two of the six contradict the brief they were given, and those two are the
result.** Finding 1: the port does **not** materialise `[B,NT,HV,C,C,K]`, so the
256× is not a byte count — the cost is 32× more `exp2` instead (§2). Finding 2:
TinyGrad 0.14.0 **does** emit tensor cores, so the gap moves from (i) to (ii)
and is a one-pattern-matcher fix rather than a missing capability (§3). Finding 4
adds a third: the recorded "85% scheduler" is not `create_schedule` either, which
tinygrad already caches (§5).

---

## 0. What was run, and what was not

**Run, on this machine, 2026-09-27.** Apple M1 Pro, `Device.DEFAULT = METAL`,
tinygrad **0.14.0** (`.venv`), float32 throughout, no torch, no CUDA, no Triton.

```
.venv/bin/python kda/codegen_diff.py          # dumps the generated source
CI=1 DEBUG=2 .venv/bin/python <one kda_chunk call>   # per-kernel census
.venv/bin/python probes/divide_precision.py
.venv/bin/python probes/fp32_correctness.py
```

Extra instrumentation I wrote for this note and did **not** commit: an allocator
spy on `tinygrad.device.Buffer.allocate`, a spy on `MetalCompiler.compile`, a
`cProfile` of one warm call, a four-regime FMA sweep, and the build-vs-realize
split from `kda/replay_kda.py`'s method. All are described inline where used.
Every number below is first-hand except where the text says otherwise.

**One environment caveat worth stating up front.** `codegen_diff.py`'s own
console output makes two claims its own captured source contradicts — it prints
"no `exp2`" (there are 64 of them, as `exp2(x * 1.4426950216293335f)`, the same
`log₂e` folding fla does) and it implies the `[C,C,K]` tensor had to be
materialised (one kernel was compiled, and the allocator never sees a buffer
larger than 8.42 MB). The script is a useful scaffold and its arithmetic is
right; its prose is not. Everything quoted here is read out of the captured
source rather than out of the script's commentary. `codegen_diff.py:334` in
`kda/kda_vs_fla.py` has the same class of bug: its MMA check uses the substring
`"mma."`, which also matches `gamma.`, and reported a false positive when I first
ran it.

**Not run.** fla's Triton kernels. There is no `torch` and no `triton` in this
environment, so every Triton line below is a **transcription** of
`fla/ops/kda/chunk_intra.py`, taken from `kda/codegen_diff.py:39-51`,
`kda/SPEC.md` §4.4 and `kda/NOTES.md` item 4. Treat the Triton column as
source-reading, not measurement. It is flagged again at every use.

**Not run: CUDA.** The brief said "TinyGrad's actual generated CUDA source". The
code below is Metal, because that is this host's default device. The relevant
point survives the substitution — `cstyle` is one renderer shared by CUDA, HIP,
OpenCL and Metal, and the rewrite in §7 lives in the *frontend*
(`tinygrad/mixin/elementwise.py:251`), above every backend.

**Also not run: the backward pass, bf16, and the two-regime split.** All
forward-only, all float32.

---

## 1. The operation

Both sides compute KDA's decayed key-key gram, the step that decides whether the
operator is memory-bound or compute-bound:

```
Z[p,j] = sum_d  exp(gc[p,d] - gc[j,d]) * k[p,d] * k[j,d]        (j < p)
```

`gc` is a chunk-local `cumsum` of the log gate over `C = 64` tokens; `k` is
L2-normalised; `p, j` index the 64 slots of one chunk. Default shapes throughout:
`C = 64`, `K = V = 64`.

### The hand-written side (transcribed from `fla/ops/kda/chunk_intra.py`)

```python
BT = 64;  BC = 16;  NC = cdiv(BT, BC) = 4
grid = (NT, NC, B * HV)                       # one program per (chunk, sub-chunk, head)

b_k0  = tl.load(p_k0, mask=m_ck0).to(tl.float32)          # [BC, K]
b_g0  = tl.load(p_g0, mask=m_ck0).to(tl.float32)          # [BC, K]
b_gn1 = tl.load(g + ... + o_k, mask=m_k).to(tl.float32)   # [K]      <- ONE row

b_gqn  = tl.where(m_tc1[:, None], exp2(b_g1 - b_gn1[None, :]), 0)   # [BC, K]
b_kgt  = tl.trans(b_k0 * exp2(b_gn1[None, :] - b_g0))               # [K, BC]

b_Akk10 = tl.dot(b_k1 * b_gqn, b_kgt, b_Akk10)                      # [BC, BC]  acc in regs
tl.store(p_Akk10, b_Akk10.to(dtype), mask=m_c1)
```

Three properties do all the work:

1. `b_gn1` is **one reference row per 16-wide sub-chunk**, not per 64-wide chunk.
   The decay operand is therefore `[BC, K] = [16, 64] = 1,024` elements.
2. `tl.dot` is a **tensor-core MMA** with the accumulator in registers. The
   `[16,16]` score block never leaves them; the only extra buffer is
   `Akkd = empty(B, T, HV, BC)` — `[T, 16]`, not `[T, T]`.
3. The `exp2` is computed **once per operand element** and then reused across
   all 16×16 = 256 products of that sub-chunk.

### The port's side (`kda/tiny_kda.py:125-144`)

```python
gp = g_cum.reshape(B, NT, HV, C, 1, K)     # p axis
gj = g_cum.reshape(B, NT, HV, 1, C, K)     # j axis
m_strict = Tensor(tril_mask(C, True, False)[None, None, None, :, :, None])
diff = gp - gj
Dl = (diff + (m_strict - 1.0) * 1e4).exp()          # [B,NT,HV,C,C,K]  <-- 64x64x64
k_p = tk.reshape(B, NT, HV, C, 1, K)      # broadcast over j
k_j = tk.reshape(B, NT, HV, 1, C, K)      # broadcast over p
Z = (Dl * k_p * k_j).sum(-1)              # [B,NT,HV,C,C]
```

Same function. The intermediate is `[C,C,K]` instead of `[BC,K]`, and the
contraction is a generic `sum(-1)` instead of a dot.

---

## 2. Finding 1 — the decay intermediate: 256× as a graph object, zero bytes in memory

**Tag: (i) for every DSL in the study. Universal, not a TinyGrad defect — and,
as §"the correction" shows, not a memory cost either.**

### The arithmetic

At the shape `kda/codegen_diff.py` actually uses (`B=1, T=64, H=HV=1, K=V=64`,
so `NT=1`), per `(batch, chunk, value-head)`:

| | shape | elements |
|---|---|---|
| fla decay operand | `[BC, K] = [16, 64]`, × NC=4 sub-chunks | **1,024** |
| port decay intermediate | `[C, C, K] = [64, 64, 64]` | **262,144** |
| | | **256×** |

In bf16 (2 bytes) that is 2 KB against 512 KB per `(b,n,hv)`. **Neither of those
byte figures is ever spent** — see immediately below. *(The brief quoted
"B=2, NT=128, HV=2, C=64, K=64" for the 262,144 figure; that shape gives
134,217,728. 262,144 is the per-`(b,n,hv)` value at the `codegen_diff.py` shape
`B=1, T=64, H=HV=1, K=V=64`, and that is what the generated code corresponds
to.)*

### The correction: it is never materialised

This is the part the earlier write-up got wrong, and it is the part that matters
most, because it changes what the 256× *means*.

**Measured.** I wrapped `tinygrad.device.Buffer.allocate` and recorded every
allocation made by one full `kda_chunk` call at
`B=2, T=512, H=HV=4, K=V=64, C=64` (`NT=8`):

```
distinct allocation sizes ever requested : 6
largest single allocation               : 8,421,376 B  =  8.42 MB
[ B,NT,HV,C,C,K ] in float32            : 67,108,864 B  ->  NEVER ALLOCATED
total allocation calls                   : 4,591
total bytes allocated (all calls)        : 54.1 MB
```

Six distinct sizes in the whole operator. A single 67 MB buffer never appears.
Cross-checked against TinyGrad's own memory planner (`DEBUG=1`):
`memory reduced from 47.02 MB -> 8.42 MB, 6048 -> 1 bufs`.

**So the `[C,C,K]` tensor exists in the port's source and in the UOp graph, and
then does not exist in memory.** The `exp` is fused into the contraction. The
isolated experiment in `kda/codegen_diff.py` shows exactly this: **1 kernel
compiled**, 96 lines, 64 `exp2` call sites, zero intermediate buffers.

This is the *opposite* of over-fusion, and that is worth saying plainly because
the framing note expected the opposite. Over-fusion is the failure mode the
"Compilers and Kernels" chapter warns about — *"Fusion lengthens live ranges and
can increase register pressure or shared-memory demand… **This is why
over-fusion is possible**"* (quoted at `notes/04-where-compilers-stop.md` §5,
source `research/SOURCES.md` #32) — and `research/TAXONOMY.md` §C1.11 records
TinyGrad's self-admission that recompute-vs-store is "a big bottleneck of
training speed" with no systematic implementation. For this operator the
scheduler made the locally-reasonable call and it was the *right* one: the
`[C,C,K]` tensor was never born.

So the over-fusion citation does not explain this finding — it explains the
*hypothesis* that this finding was originally filed under, and the measurement
refutes it. What the finding actually is: the 256× is a statement about (a) the
graph the port writes and (b) the operand shape fla chose, it costs **zero**
extra DRAM traffic, and anyone quoting "256× more memory" from this study is
quoting a number that was never allocated. The cost shows up somewhere else
entirely, and it is arithmetic.

`notes/06-corrections-ledger.md` §1a independently audits all seven places the
256× appears in the tree and rules six of them correctly scoped — including this
one, on the grounds that it says "zero bytes in memory" — with
`research/PARITY-SCOPE.md:269` the single over-claim, for calling it "a factor of
work and traffic". It also records a second, incompatible derivation of the same
number (`L/C = 16384/64`, C-02) which coincidentally equals this one at these
parameters. Two numbers called 256× is precisely the ambiguity the ledger exists
to prevent, and this file is one of the two places it is defined.

### What the port actually pays: 32× more `exp2`

The 256× does not disappear; it moves from bytes into transcendental
evaluations, and there it is real. Counted from the generated source
(`kda/codegen_diff.py`, gram kernel, `B=1, T=64, H=HV=1, K=64`, so `NT=1` and one
`(b,n,hv)` triple; 256 threads × 16 outputs per thread):

```
loop body        : 64 exp2(   (16 (p,j) pairs x 4 vector lanes of one float4)
Ridx0 iterations : 16         (K = 64 lanes / 4 per float4)
threads          : 256        (gid.x 2 x lid.x 8 x lid.y 16)
=> per (b, n, hv): 64 x 16 x 256 = 262,144 exp2  =  C^2 * K
```

Cross-check: the kernel's four buffers are all `*_4096` — three inputs and
`data0` the output — and the output is `C*C = 4,096` elements over 256 threads ×
16 accumulator slots. The count is consistent with the kernel's own shape
arithmetic, not a guess.

fla, from the same transcription: `b_gqn` `[16,64]` + `b_kgt` `[64,16]` per
sub-chunk = 2,048 `exp2`, × NC=4 = **8,192** per `(b, n, hv)`.

> **262,144 / 8,192 = 32× more `exp2`, measured from the generated source on one
> side and counted from the transcribed kernel on the other.** The fla count is
> arithmetic on transcribed shapes, not a measured instruction count; treat the
> ratio as accurate to the shape algebra, which is exact, and not as a
> measurement.

The mechanism is exactly the one `kda/NOTES.md` item 4 predicted and no schedule
removes. fla makes the decay operand `[BC,K]` and then *reuses it* across the
16×16 score block. The port evaluates `exp(gc[p,d] - gc[j,d])` afresh for every
one of the 64×64 pairs. Second-guessing the port is not possible here: item 3 of
the same file documents that the cheaper-looking `(Γ⊙K)(K/Γ)ᵀ` form divides by
zero when Γ underflows, so the difference form is the only numerically safe one,
and the difference form is what costs 32×.

`research/TAXONOMY.md` §C1.6 is the right frame and calls this **the firmest (i)
in the taxonomy**: *"Triton, TileLang and tinygrad all take a function and
compile it. If the function materialises the N×N score matrix, no schedule
removes it."* arXiv 2607.04454 refuses to tabulate across that boundary, which is
the authors implicitly declaring it out of scope for schedule search: *"the
Triton (tiled FlashAttention-2) and TileLang (untiled O(n²)) kernels are
**algorithmically different**"*. Not a schedule gap. A port gap.

---

## 3. Finding 2 — the contraction gets no tensor cores

**Tag: (ii) for TinyGrad, and it is TinyGrad-specific — which the brief got
wrong in the other direction too.**

### The generated code, verbatim

One kernel, 96 lines, from `kda/codegen_diff.py` on `Device.DEFAULT = METAL`.
Buffer names carry element counts; there is no `data?_16777216` anywhere, which
is §2's correction visible in the signature itself. The four inputs are
`g_cum`-derived (`data2`), the two `k` broadcasts (`data1`, `data3`) and the
output (`data0`).

```metal
kernel void r_2_8_16_4_4(device float* data0_4096, device float* data1_4096,
                          device float* data2_4096, device float* data3_4096,
                          uint3 gid [[threadgroup_position_in_grid]],
                          uint3 lid [[thread_position_in_threadgroup]]) {
  float buf0[16];
  int gidx0 = gid.x;                    /* 2  */
  int lidx0 = lid.x;                    /* 8  */
  int lidx1 = lid.y;                    /* 16 */
  int alu0 = ((gidx0<<11)+(lidx0<<8));
  int alu1 = (alu0+(lidx1<<2));
  ...
  // mask bias, hoisted out of the reduction:  (0 - 1) * 10000
  float alu17 = ((val3.x+-1.0f)*10000.0f);
  ...
  for (int Ridx0 = 0; Ridx0 < 16; Ridx0++) {
    ...
    *(buf0+0) = ((*(buf0+0))+(exp2((((val7.x-val11.x)+alu17)*1.4426950216293335f))*val15.x*val19.x)
                           +(exp2((((val7.y-val11.y)+alu17)*1.4426950216293335f))*val15.y*val19.y)
                           +(exp2((((val7.z-val11.z)+alu17)*1.4426950216293335f))*val15.z*val19.z)
                           +(exp2((((val7.w-val11.w)+alu17)*1.4426950216293335f))*val15.w*val19.w));
    ... 15 more of these, one per (p,j) pair ...
  }
  *((device float4*)((data0_4096+alu1))) = float4((*(buf0+0)),(*(buf0+1)),(*(buf0+2)),(*(buf0+3)));
  ...
}
```

**Measured: 0 occurrences of `simdgroup_matrix`, `simdgroup_*`, `mma`,
`matrix_multiply`, `tensor_core` or `gemm` in 8,473 characters / 96 lines.**
(That check is what `kda/kda_vs_fla.py:334-336` already does; the regex matters —
a naive `"mma." in body` also matches `gamma.` and reports a false positive. My
first pass did exactly that and "found" an MMA in a kernel that has none.)

Note also that `exp2` **is** present, folded to `exp2(x * 1.4426950216293335f)` —
TinyGrad does the `log₂e` scaling the same way `fla` does
(`kda/SPEC.md` §4.3, `RCP_LN2` in `fla/ops/utils/constant.py`). The transcript in
`kda/codegen_diff.py:118-120` claims "no `exp2`" and is wrong about that; it is
right that there is no MMA.

### The side-by-side

| | fla (transcribed) | TinyGrad 0.14.0 (generated) |
|---|---|---|
| grid | `(NT, NC=4, B*HV)` — 4× more programs, one per 16-wide sub-chunk | 2 × 8 × 16 = 256 threads, whole 64×64 score block per program |
| decay operand | `[BC,K] = [16,64]`, computed once and reused 16× | none; `exp2` recomputed for every one of the 64×64 `(p,j)` pairs |
| contraction | `tl.dot(b_k1 * b_gqn, b_kgt, b_Akk10)` | `*(buf0+N) = ((*(buf0+N)) + exp2(...)*a*b + ...)` × 4 lanes × 16 iters |
| accumulator | MMA fragment, in registers | `float buf0[16]`, scalar, read-modify-write per iteration |
| `exp2` per `(b,n,hv)` | 8,192 | **262,144** |
| lines of code for the contraction | 1 | 16 (one per `(p,j)` pair) + 1 loop + 1 load block |
| MMA | tensor cores | none |

### But TinyGrad *does* have tensor cores, and uses them elsewhere

This is the correction, and it moves the finding from (i) to (ii). The brief
assumed TinyGrad could not emit MMA at all; it can, and `research/TAXONOMY.md`
§C1.4's "I could not verify a single measured claim about tinygrad's Tensor Core
performance from any source" is now answerable for this version. It is also
tagged "(iii), unverified" there on the strength of tinygrad's own docs — "We
have a simple framework in tinygrad for adding these ALU blocks" — which is
accurate about the docs and understates the framework: 5 `TensorCore`
descriptors, an `Opt` that applies them, and a working `simdgroup_multiply_accumulate`
in the output.

Measured on the same machine, same session, same renderer:

```
plain Tensor(64,64) @ Tensor(64,64):
  renderer              : MetalRenderer
  renderer.tensor_cores : 5 descriptors, e.g. TensorCore(dims=(8,8,8), threads=32,
                          elements_per_thread=(2,2,2), dtype_in=float, dtype_out=float, ...)
  generated source      : simdgroup_float8x8 mat_a, mat_b; simdgroup_float8x8 mat_c;
                          simdgroup_multiply_accumulate(mat_c, mat_a, mat_b, mat_c);
  MMA tokens found      : simdgroup_multiply_accumulate, simdgroup_float8x8
```

And inside the KDA port itself, one kernel fuses a matmul with an `exp2` and
*does* get tensor cores (`r_2_32_2_2_4_4_8`, 187 generated lines, captured from
the port's own compile):

```metal
float2 __WMMA_8_8_8_float_float(float2 a, float2 b, float2 c){
  simdgroup_float8x8 mat_a, mat_b; simdgroup_float8x8 mat_c;
  mat_a.thread_elements()[0] = a[0]; mat_b.thread_elements()[0] = b[0]; mat_c.thread_elements()[0] = c[0];
  mat_a.thread_elements()[1] = a[1]; mat_b.thread_elements()[1] = b[1]; mat_c.thread_elements()[1] = c[1];
  simdgroup_multiply_accumulate(mat_c, mat_a, mat_b, mat_c);
  return float2(mat_c.thread_elements()[0], mat_c.thread_elements()[1]);
}
```

So the capability is present, selected by an optimisation pass, and used. What
it will not do is tensor-core **this** contraction, and the reason is in the
source. `tinygrad/codegen/opt/postrange.py:214-227`:

```python
def _apply_tc_opt(self, use_tensor_cores, axis, tc_select, opt_level):
  reduceop = self.reduceops[0]
  if use_tensor_cores and reduceop.arg[0] is Ops.ADD:
    mul = reduceop.src[0] if reduceop.src[0].op is not Ops.CAST else reduceop.src[0].src[0]
    if mul.op is not Ops.MUL: return None
    in0, in1 = mul.src
    ...
    in0_ranges = sorted([u for u in in0.ranges if u not in in1.ranges], ...)
    in1_ranges = sorted([u for u in in1.ranges if u not in in0.ranges], ...)
```

The gate is `REDUCE(ADD(MUL(A, B)))` with **two operands carrying disjoint range
sets**. The KDA gram is `REDUCE(ADD(MUL(MUL(MUL(EXP(...), k_p), k_j))))` — the
innermost `MUL` has one operand that is itself a `MUL` of an `EXP` and a tile,
not a tile, so `in0_ranges` and `in1_ranges` cannot be separated and the pass
bails at `if mul.op is not Ops.MUL` or shortly after. That is a **(ii)**: the
schedule exists in the search space, the pattern matcher does not reach it.
Fixable by extending `_apply_tc_opt` to see through a chain of elementwise ops
and pick the two range-carrying tiles underneath. Not a port.

**Corrected statement of finding 2:** TinyGrad 0.14.0 emits tensor-core MMA on
Metal — measured, twice, on a bare matmul and inside the KDA port itself — and
applies it to the port's `w = M(Γ⊙K)`, `u = MV`, `v_new = u − wS` and
state-update matmuls. It does not apply it to the decayed gram, because that
reduction is a three-operand product rather than a `REDUCE(ADD(MUL(tile, tile)))`
and `codegen/opt/postrange.py:_apply_tc_opt` requires the two-operand form.

Note the corollary, which is the interesting part for the study: even if the
tensor cores were found, the *arithmetic* would still be 32× too much `exp2`
(§2), because a tensor core consumes a pre-transformed tile and cannot compute
`exp2` itself. Fixing finding 2 without fixing finding 1 buys a faster wrong
computation — and it is the same shape of mistake as the `K/Γ` trap in
`kda/NOTES.md` item 3, where optimising the *form* would have introduced a
divide-by-zero. The order of these two fixes is not interchangeable.

---

## 4. Finding 3 — 12,090 kernels, and 99.5% of them are one `C=64` loop

**Tag: not a compiler defect at all — it is what unrolling costs, in every DSL
in the study. The brief is right that most of them do not matter, and §"which of
them matter" below shows by how much.**

### The count, reproduced exactly

`CI=1 DEBUG=2` on one `kda_chunk` call at
`B=2, T=512, H=HV=4, K=V=64, C=64`:

```
kernel launches        : 12,090
total device time      : 506.7 ms
  r_ (reduce/matmul)   :  7,984 launches   372.5 ms   73.5%
  E_ (elementwise)     :  4,106 launches   134.1 ms   26.5%
```

That is the number in `notes/03-kda.md` §2 (12,090 / 7,984 / 4,106) reproduced
from scratch on a different run of a cold process: **the counts match exactly**,
and the counts are what this finding is about. Device time differs — 506.7 ms
here against 274.3 ms recorded there — because that figure came from
`kda/perf_kda.py`'s 2-iteration average (and the first iteration of a cold
process pays pipeline warm-up), and because this host was also running other
work. Treat the *shares* (73.5% / 26.5% against 72.9% / 27.1%) as the
reproducible quantity and the absolute milliseconds as one machine's number.

### Where they come from

`DEBUG=2` prints one line per launch with the kernel's name. Grouping by name:

```
  launches  shape (numel)   what it is
  --------  --------------  --------------------------------------------------
     4,096  E_2_8_16_4_4     the `b`-vector update in the forward substitution
            (4,096)         numel = B*NT*HV*C = 2*8*4*64.  C^2 = 64*64 = 4,096.
     7,936  62 distinct     the `s = (A[:,:,:,r,:r] * b[:,:,:,:r]).sum(-1)`
            shapes x 128     reduction. 128 = 2 realize() calls x 64 columns.
     58      various         everything else
  --------
  12,090
```

The arithmetic closes: 4,096 = C², and 62 × 128 = 7,936, summing to 12,090
exactly. Together **12,032 of 12,090 launches (99.5%)** come from the UT
transform's inner double loop — `C = 64` columns × `C−1 = 63` dependent row
updates = 2,016 elementary steps per `realize()`, each becoming a separate
`[B,NT,HV,C]` elementwise kernel and a separate contraction kernel, and
`realize()` runs twice because `tiny_kda.py:252` returns `o.numpy(), tS.numpy()`.

The 62 distinct reduce shapes are the giveaway: 63 possible values of `r` would
give 63 shapes, and 62 appear because `r = 0` degenerates (the slice is empty, so
TinyGrad folds the reduction away). Every other shape appears exactly 128 times
= 2 `realize()`s × 64 columns, which is the signature of an unrolled `for i in
range(C)` with nothing else varying inside it.

`notes/03-kda.md` §2b reaches the same diagnosis — *"the UT transform is O(C²) in
graph ops, not O(C)"* — but its own pre-registered estimate was ~72 kernels
against an actual 12,090, **168×**. The estimate counted *sequential* steps
(`C + NT`); the cost is *quadratic* in `C`.

### The count is a function of C, not of the workload

| shape | NT | HV | scheduled kernels (DEBUG=1) | launches |
|---|---|---|---|---|
| `B=1, T=128, H=HV=1` | 2 | 1 | 6,170 + 6,029 = 12,199 | — |
| `B=2, T=512, H=HV=4` | 8 | 4 | 6,188 + 6,041 = 12,229 | 12,090 |

4× the chunks and 4× the heads, same kernel count. **Inferred, not measured:**
since both terms are `O(C²)`, doubling `C` should roughly quadruple the count.
Running that would need a `C=128` port, and `kda/SPEC.md` §5 explains why the
paper fixes `C=64`, so it is not a shape anyone should port.

### Which of them matter: almost none

| | kernels | device time | share |
|---|---|---|---|
| forward substitution (`E` + `r` families) | 12,032 | 480.1 ms | 94.8% |
| **all real KDA arithmetic** (both grams, `w`, `u`, 8-step scan, output) | **58** | **26.6 ms** | **5.2%** |
| wall-clock time of the call | | 10,199 ms | device time is **5.0%** of wall |

The 58 kernels that do the actual operator are 0.5% of the count and 5.2% of the
GPU time. The kernel count is a *fingerprint of the unrolling*, not a cost
driver: 94.8% of the device time is spent in a serial triangular solve that
`kda/NOTES.md` item 2 already priced at 16 FLOP/byte and called "a small,
latency-bound, serial island".

**This is the honest version of `notes/04-where-compilers-stop.md` §6's claim
that "12,090 kernels was mostly irrelevant while 256× intermediate was
everything".** The second half no longer holds — the intermediate is not
materialised (§2). What *is* true is that the kernel count is a symptom of
control-flow unrolling, which is recoverable, and that a count with no
attribution attached to it is not evidence of anything. This is exactly the
methodological point arXiv 2607.04454 makes: an idiomatic kernel that
**passes its correctness gate while running more than 300× slower** than the
PyTorch baseline (323× on GH200, clock-locked; the RMSNorm family reaches
**1520×** in the GH200 naive→optimized cliff, `research/SOURCES.md` #1) is the
reason a count, or a pass/fail, is not a result.

Universal mechanism: a static-shape graph compiler has no loop primitive, so a
Python `for` becomes `NT`/`C` copies of the subgraph, and each copy's
loop boundary is a materialisation point. `research/TAXONOMY.md` §C1.11,
TinyGrad's own `docs/developer/speed.md`: *"The scheduler determines how
operations are grouped into kernels… **This is currently a big bottleneck of
training speed.**"*

---

## 5. Finding 4 — the schedule is cached; the lowering is not

**Tag: (i) for the caller. TinyGrad-specific. This supersedes the 16% / 85%
split in `notes/03-kda.md` §6a, which was itself the correction of "99% Python".**

### The correction that was already made

`notes/03-kda.md` §2 reported "99% of wall time is Python building the graph",
then corrected it in §6a to **16% graph construction / 85% scheduler rewrite
passes re-running on every `realize()`**, with the right conclusion: *"the
compiled code is fine; the planning is redone per call."* `kda/replay_kda.py`
produced `1.78 s` build / `9.01 s` scheduler / `0.11–0.27 s` GPU.

I re-measured the split with the same method (`Tensor.numpy` monkeypatched to a
no-op) at the same shape, and it reproduces:

```
warm call (build + lower + execute)  : 10,199 ms
build only (Tensor.numpy no-op)      :  1,839 ms   = 18.0%
remainder (lower + launch + execute) :  8,360 ms   = 82.0%
```

18% / 82%, against the recorded 16% / 85%. Same conclusion, small drift.

### But the mechanism is not what either version says

`DEBUG=1` prints the scheduler step's own time and whether it hit its cache.
Cold, at the perf shape `B=2, T=512, H=HV=4`; warm, at the cheaper
`B=1, T=128, H=HV=1` used to keep the loop short (the two `scheduled` lines per
call are the two `realize()`s from `tiny_kda.py:252`):

```
cold:  scheduled  6188 kernels in 15501.75 ms | CACHE MISS  288406 uops in cache
warm:  scheduled  6170 kernels in    68.34 ms |  cache hit  253927 uops in cache
warm:  scheduled  6029 kernels in    65.28 ms |  cache hit  219658 uops in cache
```

So `create_schedule` — the toposort that emits the `LINEAR` program list, and the
step whose own timer `DEBUG=1` prints — **is** cached, and costs **68 ms, not
9 s**. tinygrad 0.14.0 has `SCACHE = 1` by default
(`tinygrad/helpers.py:283`) and an in-process `schedule_cache: dict[bytes, UOp]`
(`tinygrad/schedule/__init__.py:117`, consulted in `lower_sink_to_linear`). The
8.36 s of "remainder" is everywhere else: `spec`, `rangeify`, `linearize`, and
`do_to_program` (linearize → render → compile) all sit outside that cache.

`cProfile` of one warm call, 25.3 s total under the profiler:

```
ncalls   tottime   cumtime   function
42,435     5.914 s  15.955 s  tinygrad/uop/ops.py:1675(unified_rewrite)
 2,149,480 1.430 s   7.593 s  tinygrad/uop/ops.py:1473(rewrite)
13,639,667 1.576 s   1.603 s  {method 'get' of 'dict' objects}
 1,642,875 0.777 s   3.688 s  tinygrad/uop/ops.py:1644(cached_bpm_rewrite)
   221,573 1.050 s   1.075 s  runtime/support/objc.py:33(f)      <- device dispatch
```

The top three entries are all the UOp graph-rewrite machinery, and they nest:
`unified_rewrite` alone accounts for **15.96 s of 25.32 s cumulative (63%)** over
42,435 invocations. So the accurate version of finding 4 is narrower and sharper
than either recorded one:

> **tinygrad 0.14.0 caches the schedule (68 ms) and the compiled kernels, and
> still spends 8.36 s of a 10.2 s warm call re-running the UOp-level rewrite
> pipeline — spec, rangeify, linearize, codegen-to-source — over a graph with
> ~220,000–290,000 live UOps. `create_schedule` is not the cost. The
> per-`realize()` lowering is.**

Note what this does to `notes/03-kda.md` §6a's remedy, which proposed caching the
schedule. **`SCACHE` already does that, by default, and it recovers 68 ms of
8,360 ms.** The proposal is not wrong so much as aimed one layer too high, which
is the same error as the `/` finding one section later: identify the nearest
mechanism, fix that, and the cost turns out to be in the layer underneath.

**Tag (i), for the caller.** There is no API to say "plan this once and re-launch".
`notes/03-kda.md` §4 proposes a `Tensor.plan()` returning a re-dispatchable
program list, which would take the port from 10.2 s to the 0.11–0.27 s of GPU
time. Two things make that more tractable than it looks, and one makes it harder.
Easier: the schedule is already computed, already deterministic, and already
cached — `SCACHE` proves the key exists and that the result is reusable.
Harder: the schedule is only ~68 ms of the 8.36 s, so caching *it* is nearly
worthless; the win requires caching the whole lowering through
`Ops.BINARY`, which is what `notes/03-kda.md` §4's "cache the scheduled program
list, not just the compiled kernels" is really asking for. `research/TAXONOMY.md`
§C1.11 classifies the underlying problem as (iii) — TinyGrad says so itself.

Universal caveat, from `research/TAXONOMY.md` §C1.11 and PRIORWORK §A.8: this is
the layer TinyGrad itself names as its worst, and the layer a naive port of
arXiv 2607.04454's protocol would hide, because TinyGrad *"bypasses the GPU
driver and prebuilds the command queue"* — a launch-overhead win that has nothing
to do with kernel throughput.

---

## 6. Finding 5 — FMA is real, and the barriers buy nothing

**Tag: universal compiler phenomenon, and the literature said so before anyone here
measured it. The novel part is the second half: the barrier machinery is
measurably worthless for this operator.**

### The measurement

`a*b+c` in float32 on `Device.DEFAULT`, 2,000,000 triples, reference = the same
expression in float64 rounded once to float32 (exact: a float32 product needs 48
bits). Four input regimes, because the *rate* is regime-dependent and only one
column is not:

| regime | differs from unfused mul-then-add | differs from exact FMA |
|---|---|---|
| `U[0,1)³` | 277,953 (13.90%) | **0 (0.00%)** |
| `U[0,1)² × U[-2,2]` | 279,946 (14.00%) | **0 (0.00%)** |
| `U[-2,2]³` | 484,234 (24.21%) | **0 (0.00%)** |
| wide `±10³` | 501,435 (25.07%) | **0 (0.00%)** |

TinyGrad contracts `a*b+c` into a single fused multiply-add, and the fused result
is *bit-identical* to exact FMA in all **8,000,000** cases across the four
regimes. It is not a different approximation; it is FMA, exactly.

`notes/02-obs-camera-residuals.md` records **27.14%** against the unfused result.
That is **above all four** of my rates, and nearest the two widest (24.2%,
25.1%) — so it is consistent with a wider or differently-shaped regime than any
I ran, and clearly inconsistent with the two narrowest. The lesson is that a
contraction *rate* is a property of the input distribution and is not comparable
across probes that do not record it; the `0.00%` column is the invariant, and it
is the column that identifies the mechanism.

### The barriers

`probes/oc/tiny_oc.py:110` carries `BARRIER = [False]` and line 114 returns
`t.realize()` when it is on. `notes/02-obs-camera-residuals.md` measured, over
500,000 triples:

| | vs unfused | vs exact FMA |
|---|---|---|
| plain | 135,993 | 0 |
| `product.realize()` | 0 | 135,993 |
| `operands.contiguous()` | 135,993 | 0 |

`product.realize()` is a complete and exact FMA barrier, because it forces the
product through memory. Copying the *operands* is not a barrier at all.

**And then:** with the barrier set on at every contraction site in the camera
port — the ray-setup products, the `sqrt` argument, the edge term, the per-step
`tmx + tdx` increments — the result was **bit-identical with the barrier on and
off**, over all 41,472 pixels. FMA is real, it is controllable, and it was not
the mechanism behind the 9 depth quanta and 3 edge bits that were off. Those are
`1 ULP` from `/` (§7) amplified by Minecraft's 65,536-entry `sin` LUT, whose
buckets are `2π/65536 = 9.587e-05` wide. `notes/02-obs-camera-residuals.md`
traces the arithmetic exactly: for yaw = 180° the float32 product
`3.1415927 × 10430.378` is `32768.0` under numpy and `32767.998` (1 ULP low)
under TinyGrad, so `floor()` gives 32768 vs 32767, so the LUT index differs by
one, so `sin` differs by 9.587e-05, and `(int)(t*4)` flips on 9 pixels.

**This is the load-bearing lesson of the whole camera probe and it is not about
FMA at all:** a quantiser in the middle of the dataflow turns a last-bit
difference into a visible one, so a port has to be bit-stable end to end or not
at all. `notes/DECISIONS.md` #3 records the decision that follows — semantic
equivalence, not bit-exactness, as the target — and #13 records that two of the
three blockers (`no FMA`, `correctly rounded /`) are TinyGrad-side and *all
three* are absent knobs. `notes/06-corrections-ledger.md` C-05 **amends** #13:
the division blocker is not a *backend* defect, it is created in the Tensor API
(§7), so the decision to file upstream stands while its stated reason does not.

So `BARRIER = [True]` costs extra kernels and returns nothing, and the general
statement holds:

> **Mathematical equivalence does not imply bitwise identity.**
> *"Even pointwise fusion can introduce fused multiply-add instructions or
> reassociation, so mathematical equivalence does not imply bitwise identity."*
> — "Compilers and Kernels", quoted at `notes/04-where-compilers-stop.md` §5,
> source `research/SOURCES.md` #32.

That sentence is a general compiler fact. It was written down as one before
anyone here measured it, and this probe is the confirmation. The KDA port is a
second, independent instance: `kda/tiny_kda.py` agrees with `kda/ref.py` to
**6.6e-06** in float32 while the float64 reference agrees with the per-token
recurrence to **5.15e-15** — a 10⁹ gap, from FMA contraction and reassociation
plus the `/` of §7, and `kda/test_tiny.py` passes 6/6 anyway. That is
arXiv 2607.04454's thesis in miniature, and it is the reason this study needs a
roofline measurement rather than a test suite as its finish line
(`research/PRIORWORK.md` §A.1: correctness is the *acceptance* criterion, and a
slow kernel is never rejected).

There is no knob. `research/TAXONOMY.md` §C1.4 is right to call this (i) *for
the caller*: there is no way to say "I need bit-exactness".

---

## 7. Finding 6 — float32 `/` is one ULP wrong, silently, with no knob

**Tag: (i). A real TinyGrad defect, and the only finding in this document that is
a bug report rather than a boundary.**

**This section is subordinate to `notes/06-corrections-ledger.md` rows C-05
through C-08 and P-1, which are sharper than what I first wrote here and which I
have adopted rather than restated.** Two of them change what may be claimed:
the scope is **not** established on NVIDIA (C-04, untested), and the three
27.13% backends do **not** share a mechanism (C-08).

### The measurement

`probes/divide_precision.py`, 200,000 samples, reference = float64 divide rounded
once to float32 (exact: a float32 quotient needs at most 48 bits):

```
1. a / a, which must be exactly 1.0 for every finite nonzero a
   CPU   :         0 / 200,000 wrong  ( 0.00%)
   METAL :  30,748 / 200,000 wrong  (15.37%)      all returning 0.99999994 = 1 - 2^-24

2. a / b against correctly rounded, per backend
   PYTHON :  54,270 wrong (27.13%)   == a*rcp(b) bit for bit
   CPU    :       0 wrong ( 0.00%)   differs from rcp*mul
   CL     :  54,270 wrong (27.13%)   == a*rcp(b) bit for bit
   METAL  :  54,270 wrong (27.13%)   == a*rcp(b) bit for bit
```

`probes/fp32_correctness.py`, 2,097,152 samples per cell, three magnitude
regimes:

| op | TinyGrad wrong | numpy wrong |
|---|---|---|
| `+` | 0 (0.000%) | 0 |
| `-` | 0 (0.000%) | 0 |
| `*` | 0 (0.000%) | 0 |
| `/`, divisor = dividend (**`a/a`**) | **260,261 – 322,233 (12.410% – 15.365%)** | 0 |
| `/`, divisor independent (**`a/b`**) | **556,107 – 567,877 (26.517% – 27.078%)** | 0 |

and: `division error magnitude: median 1.0 ULP, p99 1.0 ULP, max 1 ULP`.

Two corrections to the version of this table that stood here until 2026-09-28:

1. **The `/` row is `a/a`, not `a/b`.** The probe read
   `b = regimes[name] if op in "*/" else a`, and `regimes[name]` *is* `a`. The
   12.410%–15.365% band is therefore self-division and was never a rate for
   division; on the same `a`, general division is wrong **1.77x–1.86x** more
   often. The citable rate is a band with its divisor law attached —
   `derivation/divide.py:mechanism_sentence`, and
   `probes/fp32_correctness.py` now prints both rows.
2. **The `+ - *` denominator was 6,291,456 and should be 18,874,368** (3 ops × 3
   regimes × 2,097,152). 6,291,456 is one op's worth of samples; the sentence
   naming all three ops quoted it. `derivation/divide.py:probe_denominators`.

`a/a != 1` needs no statistics. There is no validity condition under which a
finite nonzero `a` divided by itself is not 1. **But `1 - 2^-24` is not the
signature of an approximate reciprocal** — the earlier claim in this file is
retracted. It is the signature of a **correctly rounded** reciprocal whose error
landed between `−2⁻²⁵` and `−2⁻²⁴`: the only wrong `a/a` value is `0x3f7fffff`,
and `0x3f800001` above is **never** produced in any regime on any device
(`derivation/divide.py:corner_predicate`). A one-sided, exactly-predicted failure
is the opposite of an approximation signature. The failure is **double
rounding**, and it is 1 ULP every time — that part survives.

**Scope, before the mechanism (ledger C-04, P-1 — now RESOLVED, and against the
old reading).** This file previously scoped the finding to Metal and CL on the
ground that NVIDIA should not reproduce it. **That prediction was run and
refuted.** `out/kaggle/20260928-014248-gpu/logs/divide_precision.log` on a T4:
CUDA **27.13% (54,270) wrong, bit-identical to `a*fl32(1/b)`**; `a/a` 30,748 /
200,000. The prediction's mechanism step — "no `--use_fast_math`, therefore
`div.rn.f32`, therefore no defect" — reasons about a compiler flag reaching a
rewrite that is **already in the emitted source**; it cannot. There is no
Metal/CL-only property to scope to. **And 27.13% is still not a property of
TinyGrad**: it is one divisor law at one `n`, and the same tinygrad `CL` backend
measures 27.13% on one host and 29.76% on another. Quote the band and its law
(`derivation/divide.py:published_rate_table`), not the scalar.

### Where it happens: the Tensor API, and every backend inherits it

`notes/02-obs-camera-residuals.md` §"Correction" and the probe's own docstring
say *"tinygrad's `cstyle` renderer rewrites `a / b` into `a * (1/b)`"*, and then
*"`Ops.RECIPROCAL` exists … `/` simply never goes through it."* Ledger rows
**C-05** and **C-07** correct both halves:

```python
# tinygrad/mixin/elementwise.py:249-251, inside Tensor.div
#   alu, not *: _broadcasted already promoted these, and a second promote would
#   cast 1/b (only a bare weak CONST is kept weak)
d = a.alu(Ops.MUL, b.reciprocal())
```

`Ops.DIV` does not exist in tinygrad's op set for floats at all, so `/` is
`MUL(a, RECIPROCAL(b))` **by construction**, above the scheduler and above
backend selection. The correct statement is the reverse of the one on record:
**`/` always goes through `Ops.RECIPROCAL`, and that is the defect** (C-07). Two
backends then *undo* it, and only two:

```python
# tinygrad/codegen/decomp/op.py:134-136
if Ops.FDIV in ops:
  pat += [(UPat.var("x").reciprocal(), lambda x: UOp.const(1.0).alu(Ops.FDIV, x))]
  pat += [(UPat.var("a") * UPat(Ops.FDIV, dtypes.floats, src=(UPat.const(1), UPat.var("b"))), lambda a,b: a.alu(Ops.FDIV, b))]

# tinygrad/renderer/cstyle.py:271-274, ClangRenderer.code_for_op
{**({k:v for k,v in CStyleLanguage.code_for_op.items() if k not in [..., Ops.RECIPROCAL]}),
 Ops.FDIV: lambda a,b,dtype: f"({a}/{b})"}
```

`ClangRenderer` deletes `Ops.RECIPROCAL` from its op set and adds a real `Ops.FDIV`,
so the decomposition fires and `/` is repaired — which is why CPU measures
0.00%. `MetalRenderer` and `OpenCLRenderer` inherit `CStyleLanguage`'s
`Ops.RECIPROCAL: lambda x, dtype: f"(1/{x})"`, so nothing undoes it and the
backend is free to approximate. So the discriminator is **not** "is it a cstyle
renderer" — `ClangRenderer` is one too, and it is the correct one (C-05, C-06).
It is "does the renderer's op set contain `Ops.FDIV`".

**And PYTHON is not a cstyle renderer at all** (C-08). Its MRO is
`PythonRenderer → Renderer → object`, and its `RECIPROCAL` comes from
`uop/ops.py:1289`. So the *rendering* differs — Python's `1/x` is a
double-precision divide — but the *mechanism* does not.

**This file's next sentence was wrong and is retracted.** It read: "PYTHON's
27.13% is **double rounding** — a correctly rounded double-precision reciprocal
multiplied in float32 — while CL and METAL's is **approximation**. Same 1-ULP
symptom, different mechanism, coincidentally equal rate." **All of CL, METAL,
PYTHON and CUDA are double rounding, and all four are bit-identical to
`a * fl32(1/b)`.** The `approximated` error class was an assumption printed as a
result. Two derivations settle it:

- a hypothetical reciprocal **1 ULP worse** than correctly rounded predicts
  **86.5%** wrong against the **27.1%** measured, at a factor of 3.2
  (`derivation/divide.py:reciprocal_sensitivity`). An approximate reciprocal
  cannot produce 27%;
- the only wrong `a/a` value is `0x3f7fffff` and `0x3f800001` above is never
  produced, in any regime, on any device
  (`derivation/divide.py:corner_predicate`). A correctly rounded reciprocal's
  error is two-sided too, so this alone does not decide it — but a *one-sided*
  failure is not what an approximation produces, and the exact predicate
  `delta(a) < -2^-25` reproduces it to the bit.

The consequence for the record: an earlier claim that the rate is "insensitive
to whether the reciprocal itself is exact" is **retracted**. The reciprocal's
accuracy is load-bearing, and identifying it as *correctly rounded* is a
separate claim from the rate.

The Metal source, which is where the whole mechanism is visible:

```metal
*((device float4*)((data0_64+alu0))) = float4((val0.x*(1/val1.x)),(val0.y*(1/val1.y)),
                                               (val0.z*(1/val1.z)),(val0.w*(1/val1.w)));
```

**One step, and it is TinyGrad's, above the backend.** The earlier reading of this
line — "MSL contracts `1/x` into an approximate reciprocal. Two steps" — is
retracted: there is no second step, and the second step was the part that was
wrong. What the backend then evaluates `(1/b)` to is its own business, and on
every backend measured here it is the correctly rounded reciprocal. The
`a * (1/b)` rewrite is the mechanism; the 1-ULP loss is the second rounding it
introduces.

### There is no knob, and `-fno-fast-math` does not help

```python
# tinygrad/runtime/ops_metal.py:93
params = f'-fno-fast-math -std={metal_version} --driver-mode=metal -x metal ...'
```

That flag is already passed, to the clang-like frontend; MSL's own math mode is
separate. The precise statement of "no knob" is the ledger's, not the earlier
one: there is no way to ask for a correctly rounded divide, and the one place a
backend *could* have honoured the request — emitting `Ops.FDIV` so the
decomposition at `codegen/decomp/op.py:134-136` fires — is opted into per
renderer by whether `Ops.FDIV` happens to be in its op set. `MetalRenderer` and
`OpenCLRenderer` do not.

The fix is upstream and small, and there are two of them: stop manufacturing
`RECIPROCAL` in `Tensor.div` and emit `FDIV` (which every cstyle-family renderer
already knows how to spell), or keep the rewrite and expose the choice (a
`Tensor.div_exact`, or a context flag) so a caller who needs exactness can get
it. The first is better and is the one that makes CPU's behaviour the default
everywhere.

### One ULP is legitimate. The silence is the defect — but it is double rounding, not an approximation

A GPU user expects fast reciprocals and 1 ULP is a normal price. The defect is
that **the caller cannot request a correctly rounded divide, cannot detect the
loss, and cannot route around it.** `notes/DECISIONS.md` #12 records the
attempt: computing `1/|dx|` as `|dx| * (1/|dx|)` was measured to differ from the
exact reciprocal in **100%** of cases — it still contains one division plus a
second rounding. There is no division-free exact reciprocal in float32.

The **loss is a second rounding**, and that distinction changes the fix. The
earlier version of this section read "cannot detect the **approximation**" and
justified itself with "the approximation itself is legitimate". Both halves are
withdrawn. The reciprocal is **correctly rounded** — so there is no approximation
to legitimise and nothing for a caller to detect as an approximation — and the
fix is the one above: make the `Ops.FDIV` repair reachable, which is a rendering
choice, not a licence to be sloppy.

And it is not a corner case in this repo: netherite's DDA takes every step size
as `1.0f/dx` and feeds it into a 65,536-entry LUT with 9.587e-05 buckets, which is
the worst possible combination, and it is the root cause of the camera port's
9 depth quanta and 3 edge bits (§6). `notes/02-obs-camera-residuals.md` closes
with the honest status: bit-exactness **not achieved**, and it would need (a) no
FMA, (b) correctly rounded division, (c) the same constant folding as clang —
two TinyGrad-side, all three absent knobs. `notes/DECISIONS.md` #13 already
classifies it as an upstream bug report; ledger C-05 amends the *reason* (not a
backend defect) while leaving the decision standing.

One caveat that belongs here rather than in a footnote: **`/ route around it`
is the weakest part of the claim for CL and METAL, and the strongest for
PYTHON.** CL and METAL's error is an *approximation*, so in principle a caller
who detects it has options. PYTHON's is *double rounding*, which no caller can
route around at all, and on NVIDIA — if P-1a is right — that is what everyone
gets. The finding gets more serious as it becomes less avoidable.

---

## 8. What this comparison establishes, and what it does not

### It establishes

1. **For one operator, at one shape, on one device, TinyGrad 0.14.0 emits code
   that computes the right function and pays for it in the wrong currency.** The
   decayed gram evaluates 32× more `exp2` than the hand-written kernel and uses no
   tensor cores, from a graph whose decay intermediate is 256× larger than the
   kernel's `[BC,K]` operand and 0 bytes in memory. The cost is arithmetic, not
   bandwidth — the opposite of what the earlier version of this comparison
   concluded.
2. **The interesting differences are not the ones the kernel count suggests.**
   99.5% of 12,090 launches come from one unrolled `C=64` loop and cost 94.8% of
   device time to compute a triangular inverse; the 58 launches doing the actual
   KDA arithmetic are 5.2% of device time, and the device is busy 5.0% of a
   10.2 s call.
3. **All three tags are inhabited, and only one of the six findings is a
   TinyGrad defect.** (i) — findings 4 and 6, and the caller cannot demand
   bit-exactness, a correctly rounded divide, or a cached plan; plus finding 1 for
   every DSL in the study. (ii) — finding 2, the tensor-core schedule exists and
   the pattern matcher does not reach a three-operand contraction. (iii) — no
   finding, but `research/TAXONOMY.md` §C1.11 records TinyGrad's own
   self-admission that recompute-vs-store has no systematic implementation.
   Conflating them, which is what "TinyGrad is slow" does, is what made the
   earlier version of this comparison wrong twice.
4. **Where the fault lies is not TinyGrad.** Findings 1, 3 and 5 are properties
   of compiling a fixed function, documented in the literature before they were
   measured here: §C1.6 of `research/TAXONOMY.md` (no schedule removes a
   materialised intermediate — *though see §2, where TinyGrad's did not
   materialise and so was never charged*), arXiv 2607.04454's refusal to compare
   "algorithmically different" kernels, and the FMA passage of "Compilers and
   Kernels" (`research/SOURCES.md` #32). Finding 6 is the only defect report;
   findings 2 and 4 are TinyGrad-specific but are a pattern-matcher gap and an
   API gap, not bugs. **The one thing this study has produced that is a bug
   report is the float32 division, and the fix is two lines in
   `tinygrad/mixin/elementwise.py` — emit `FDIV` instead of
   `MUL(a, reciprocal(b))`, which is what `ClangRenderer` already does for CPU.**
   `notes/06-corrections-ledger.md` C-05 records that this is not a *backend*
   defect, which amends `notes/DECISIONS.md` #13: the decision to file it
   upstream stands, the stated reason does not.
5. **The measurement discipline this comparison needed is the one the research
   agent already specified.** Nothing here is a speedup claim, and
   `research/PRIORWORK.md` §A.1's two-metric protocol — library-relative
   efficiency *and* roofline fraction, never one alone — is what would convert
   §2's 32× instruction count and §4's 12,090 launches into a result. Until
   `kda/kda_vs_fla.py` runs on a real GPU, the honest summary is a structural
   comparison plus one defect, not a benchmark.

### It does not establish

1. **Nothing about CUDA or H100.** Every number here is Metal on an M1 Pro, and
   no NVIDIA device and no Triton kernel was available. The Triton column is a
   transcription. `research/TAXONOMY.md` is explicit that "optimized" is
   architecture-indexed: a kernel tuned and correct on A100 spills on sm_90 and
   runs 100× slower **with no correctness signal**. For finding 6 specifically,
   `notes/06-corrections-ledger.md` C-04/P-1 records the sharper version: the
   *approximation* is predicted **not** to appear on NVIDIA at all (NVRTC is
   invoked without `--use_fast_math`, so `1.0f/b` should reach `div.rn.f32`),
   that prediction is **untested**, P-1a names the double-rounding alternative,
   and the one T4 run attempted returned 15/15 `skip`
   (`out/kaggle/gpu-v1/SUMMARY.md`). The backend-independent part survives either
   way — the Tensor API manufactures `RECIPROCAL`, so the answer depends on which
   renderer you got — but **the rate does not travel**.
2. **Nothing about performance relative to fla.** There is no head-to-head
   measurement, and `kda/kda_vs_fla.py` is the instrument that would produce one.
   Every "32× more `exp2`" here is a count of generated instructions, not a
   throughput; the same count on a device where the SFU is not the bottleneck is
   worth much less, and on one where it is (`kda/NOTES.md` §6: the operator runs
   at 37.5 FLOP/byte against a ~295 FLOP/byte H100 ridge point, so ~8× memory
   bound) a 32× cut in transcendental work may not move wall time at all.
3. **Nothing about bf16 or the backward pass.** The port is float32, forward only.
   `kda/NOTES.md` §9 records that the real kernels compute the `C×C` blocks in
   higher precision than they store and force tf32 on the triangular-inverse
   dots; none of that is in the generated code above.
4. **Nothing about the other two regimes.** `kda/NOTES.md` §8: the KDA layer
   switches on `q_len <= 64 and not self.training`, so the decode path is a
   different graph with a 2 FLOP/byte profile. Not touched.
5. **Nothing about a *speedup*.** Nothing here says TinyGrad's KDA is N× slower
   than fla's. It says what code is generated, how much `exp2` it evaluates, and
   how long one call takes to plan. Converting any of that into a speedup needs
   `kda/kda_vs_fla.py` on a real GPU with `torch` and `triton` installed, and
   `research/PRIORWORK.md` §A.3's protocol on top of it (locked clocks, 100 runs,
   median, compile time reported separately from kernel time).
6. **The 16%/85% split in `notes/03-kda.md` §6a should now be read as 18%/82%,
   and neither number describes `create_schedule`.** See §5.
7. **`notes/03-kda.md` §3's "Dl and Dq_all at 33.6 MB each" is a logical size, not
   an allocated one.** The allocator never sees a buffer larger than 8.42 MB. The
   33.6 MB figure is `B*NT*HV*C*C*K` × 2 bytes for *this* shape and is a correct
   statement about the tensor; it is not a statement about memory the port used.
   `notes/DECISIONS.md` #17 and `notes/03-kda.md` §3 both rank this as the top
   finding, and §2b calls the intermediates "64× larger than anything else in
   the operator, and they are pure intermediates". The *shape* claim is right;
   the implication that they cost bytes is not. This is the single largest
   correction in the document, and it is why the file exists: the earlier
   version of this comparison was built on it.
8. **`notes/03-kda.md` §3's ranking (#4, the decay dataflow, "the actual
   bottleneck") does not survive as stated, and neither does its proposed fix.**
   Two things change. First, the cost is 32× too many `exp2`, not materialised
   bytes. Second — and this is the part I had wrong before measuring — **the
   scheduler already performs the `exp`-into-contraction fold that §4 asks for**,
   so §4's proposal is a no-op. What is actually missing is §2 and §3's other
   half: a `[BC,K]` pre-transformed decay operand, which is what would cut the
   `exp2` count and simultaneously make the reduction a two-operand
   `REDUCE(ADD(MUL(tile, tile)))` that `_apply_tc_opt` can match. One change,
   both findings.
   **And the honest caveat, which cuts the other way:** `kda/NOTES.md` §6 puts
   this operator at 37.5 FLOP/byte against a ~295 FLOP/byte H100 ridge point,
   i.e. ~8× memory bound. On a memory-bound operator, cutting arithmetic by 32×
   may buy nothing. Whether the combined §2+§3 fix is worth doing at all is a
   roofline question, it is open, and it is precisely what
   `notes/06-corrections-ledger.md` P-2 and P-3 were opened to settle. Do not
   treat the fix as obviously valuable because the instruction count went down.

### Two errors in the framing document, not propagated here

Both are in `notes/04-where-compilers-stop.md`, which I was told to read and not
to copy. Recording them here so the correction is not lost.

**Error 1 — the 1520×.** §1 quotes the "Correct but Slow" paper (RC0–RC4
attribution) and states:

> an idiomatic TileLang LayerNorm kernel **passes KernelBench's correctness
> check while running more than 300x slower** than the PyTorch baseline
> (RMSNorm up to **1520x**)

The 300× is right and is the headline. The parenthetical is not: **1520× is not
a slowdown against PyTorch.** Per `research/SOURCES.md` #1, 1520× is the **GH200
naive→optimized cliff for TileLang RMSNorm** — an entry in Table III measuring
how much of the gap the paper's own repair closed, not how slow the kernel is
versus the library. The correct statements are: **323× slower than PyTorch on
GH200, clock-locked** (51.5 ms vs 0.16 ms), worst-case constructed variant 1293×
unlocked, and LayerNorm cliffs of 1541× (GH200) / 1347× (A100) in the same table.
Both notes should be fixed; neither number is used above.

**Error 2 — Twill's identifier.** The framing note §2 and its source table cite
Twill as `compilers.stanford.edu/publications/twill.pdf`. It is **arXiv
2512.18134** (Soi, Yadav, Kjolstad, Aiken, Mehri Dehnavi, Garland, Bauer; 19 Dec
2025; Stanford + NVIDIA). `research/SOURCES.md` #2 records that the Stanford PDF
fetch was blocked and the arXiv HTML of the same work was read instead — so the
framing note is citing a document the study never successfully opened. Its
substantive claims check out against the arXiv version: SWP reduced to modulo
scheduling, folded into the same SMT problem as warp specialization, recovering
FA3's pipeline from a tile-level description, implemented schedules "within 1%" of
hand-tuned cuDNN and FlashAttention.

---

## Files

| file | what it produced here |
|---|---|
| `kda/codegen_diff.py` | the 96-line generated gram kernel; the 262,144 vs 1,024 / 256× arithmetic; the side-by-side frame |
| `kda/kda_vs_fla.py:334` | the MMA-token check, and the `"mma."`-matches-`gamma.` false positive (§0) |
| `kda/tiny_kda.py:125-144`, `:252` | the port's `[C,C,K]` source; the two `realize()`s behind the ×2 |
| `kda/SPEC.md` §4.3-4.4, `kda/NOTES.md` item 4 | `RCP_LN2`, `BC=16`, `NC=4`, the `b_gn` reference row |
| `kda/NOTES.md` items 2, 3; §6-§7 | the triangular-inverse pricing; the `K/Γ` trap; intensity and ridge points |
| `kda/replay_kda.py` | the method for the build-vs-realize split in §5 |
| `kda/test_ref.py` / `kda/test_tiny.py` | 17/17 and 6/6; 5.15e-15 and 6.6e-06 (§6) |
| `probes/divide_precision.py` | §7: 30,748 / 200,000, 27.13%, per-backend, the generated MSL |
| `probes/fp32_correctness.py` | §7: 0 of **18,874,368** for `+ - *`; `a/a` 12.4–15.4% and general `a/b` 26.5–27.1% (now 25.9–27.0% on the published draw, `derivation/divide.py:mechanism_sentence`); max 1 ULP |
| `probes/oc/tiny_oc.py:110,114` | §6: the `BARRIER` switch |
| `notes/02-obs-camera-residuals.md` | §6, §7: the 500,000-triple barrier table; the sin-LUT mechanism; the renderer-vs-frontend claim, corrected in §7 |
| `notes/06-corrections-ledger.md` C-05…C-08, C-04/P-1 | §7: the division mechanism, the PYTHON regrouping, and the NVIDIA scope caveat. §7 adopts these rather than restating them |
| `notes/03-kda.md` §2, §2b, §3, §6a | §4, §5, §8: 12,090; the 16%/85% correction; the 33.6 MB claim |
| `notes/DECISIONS.md` #3, #12, #13, #17 | §6, §7, §8: the semantic-equivalence decision; the 100% reciprocal result; the blocker list; the ranking that §2 revises |
| `research/TAXONOMY.md` §C1.4, §C1.6, §C1.11 | the (i)/(ii)/(iii) tags; the firmest (i); the recompute admission |
| `research/SOURCES.md` #1, #2, #32 | the 323×/1520× correction; Twill's arXiv ID; the FMA and over-fusion quotes |
| `research/PRIORWORK.md` §A.8 | TinyGrad's four-layer speed decomposition, and why it matters here |

Uncommitted instrumentation written for this note: allocator spy on
`Buffer.allocate`, spy on `MetalCompiler.compile`, `cProfile` of a warm call, a
four-regime FMA sweep, and the build/realize split. Each is described at its
point of use. None is required to reproduce any number in this document except
the allocator and build/realize measurements in §2 and §5.
