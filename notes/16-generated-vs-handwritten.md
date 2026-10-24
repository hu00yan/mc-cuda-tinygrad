# 16 — generated vs hand-written: the same reduction, quoted side by side

> **The verdict.** One real CUDA kernel, 253 lines, emitted by TinyGrad 0.14.0's
> CUDA backend on a Tesla T4, and one real Triton kernel, the vendored
> `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` from flash-linear-attention
> 0.5.2. Both were compiled here, for five architectures, with no GPU.
>
> **The generated kernel issues zero tensor-core instructions on every
> architecture. The hand-written one issues 128 `HMMA` on sm_80 and 16 `HGMMA`
> on sm_90a.** That part is unsurprising and mostly dtype: TinyGrad's CUDA
> tensor-core catalogue for the T4 is half-only, and the KDA decay is fp32.
>
> The sharp difference is not the tensor cores. **It is the transcendental.**
> TinyGrad's kernel contains **128 `MUFU.EX2`** — special-function-unit
> instructions, 12.40% of its floating-point instructions, and by throughput
> weighting the majority of its arithmetic time. The hand-written kernel
> contains **3**. A 528-product matmul needs none. The generated code recomputes
> the decay once per contracted term, *inside* the reduction; the hand-written
> author hoisted the decay out into a single elementwise pre-pass on the tile and
> then contracted with `tl.dot`. Same maths, 43× fewer `exp2`.
>
> **This is a code-and-construction comparison. It is not a speed comparison,
> there is no GPU on this host, and nothing here is a timing.** §7.

Every number below was re-derived on this host. None is transcribed from
another notes file. Labels:

- `MEASURED` — an instrument ran here, on this host, on the artifact named.
- `[ARTIFACT]` — read out of a named, sha256-pinned file rather than recomputed.
- `MODELLED` — derived arithmetically from measured inputs plus a hardware
  constant that is **not** measured here.
- `PREDICTED` — a claim about hardware this host does not have.
- `[NOT-HERE]` — the measurement that would settle it cannot be run here.

Difference tags, used throughout:

| tag | meaning |
|---|---|
| **(i)** | cannot express |
| **(ii)** | can express it, but does not find it |
| **(iii)** | not yet implemented |
| **(iv)** | does not exist for this architecture |

---

## 0. The artifact, and that it is reproducible

`out/kaggle/20260928-014248-gpu/codegen/kernel000.txt` — 253 lines, 24,968 B.
`[ARTIFACT]`

It was captured on a Tesla T4 by wrapping the device compiler
(`kernels/build_notebooks.py:578-596`). I reproduced it offline on the NULL
device with the capture hook from `tools/archlab/tinygrad_cuda.py`, forcing
`PARALLEL=0` so compilation stays in-process:

```
DEV=NULL:CUDA:sm_75 PARALLEL=0
q,k,v,g,beta = test_ref.rnd(2,128,2,2,32,32,seed=0,l2n=True)
tiny_kda.kda_chunk(q,k,v,g,beta,chunk_size=32)
```

Result: 253 lines, 24,968 B, same kernel name `r_2_2_8_8_4_4_32`, same 100
`float4` loads, same 128 `exp2`, same four 33-product stores. **64 of the 100
load lines differ only in a compile-time base constant** (`+2048` where the
artifact has `+6144`, `+2080` vs `+6176`, …), i.e. a different base view into
the same pair of buffers. `MEASURED`

So the artifact is not a one-off; it is the deterministic rendering of that
graph shape. Everything below is safe to re-derive.

**One capture, not the schedule.** `out/kaggle/.../codegen/` contains exactly
one file. `MEASURED` `kda/perf_kda.py` reports `kernels per call : 12082` for
`B=2 T=512 H=4 HV=4 K=64 V=64 chunk C=64 NT=8`. `MEASURED`
(`out/kaggle/20260928-014248-gpu/logs/perf_kda.log:6,31`) Those are **different
configurations** — the capture is at C=32/K=32/V=32/T=128, the census at
C=64/K=64/V=64/T=512 — so they are not two counts of one thing. See §7.

With the pool disabled I got the whole in-process schedule at the *capture*
shape: **94 kernels, 37 distinct names, 10 of which contain `exp2`, carrying
2,628 `exp2` between them.** `MEASURED` `r_2_2_8_8_4_4_32` appears 11 times,
and the two `exp2`-bearing variants of it (15,207 B and 24,968 B) alternate.
The reason the original capture saw one kernel and not 94 is
`realize.py:273`:

```python
pool = None if len(todo) == 1 or any(getattr(c.src[0].arg, "beam", 0) for c in ar) else get_worker_pool()
```

so an in-process `compile_cached` hook only observes rounds where
`len(todo) == 1`. `MEASURED` (line number and text read from the installed
package.) With `PARALLEL=0` the pool is never created and all 94 are visible.

---

## 1. What the generated kernel is

### 1.1 Shape and thread mapping

```c
extern "C" __global__ void __launch_bounds__(128) r_2_2_8_8_4_4_32(
    float* data0_4096, float* data1_4096, float* data2_16384,
    float* data3_16384, float* data4_4096) {
  int gidx0 = blockIdx.x; /* 2 */
  int lidx0 = threadIdx.x; /* 2 */
  int lidx1 = threadIdx.y; /* 8 */
  int lidx2 = threadIdx.z; /* 8 */
```

- Five **`float*`** arguments, not `float4*`. The vectorisation is in the
  *accesses*, not the signature. `MEASURED`
- `__launch_bounds__(128)`; block is 2×8×8 = 128 threads. `MEASURED`
- Grid is `blockIdx.x` **only**. `gridDim` appears 0 times. `MEASURED`
- `if` × 0, `for` × 0, `while` × 0, `&&` × 0, `||` × 0. **No grid-stride loop
  and no bounds check of any kind.** `MEASURED`

The kernel is fully specialised to one problem size. Change C and it is a
different program.

### 1.2 Exact instruction census of the source

| quantity | count | how counted |
|---|---|---|
| lines | 253 | `wc -l` |
| `float4` loads | **100** | `float4 val<N> = ` declarations, all distinct (`val0`..`val99`) |
| — from `data1_4096` | 4 | offsets `alu4`,`alu5`,`alu6`,`alu7` |
| — from `data2_16384` | 32 | stride 4, base +2048 |
| — from `data3_16384` | 32 | stride 32, base +2048 |
| — from `data4_4096` | 32 | stride 32 |
| `exp2(` | **128** | textual count |
| `1.4426950216293335f` (= log2 e) | 128 | one per `exp2` |
| — of which "bare" `exp2((t.w*log2e))` | 4 | lines 122, 154, 186, 218 |
| — of which `exp2(((a-b)*log2e)*c)` | 124 | all match one regex exactly, 0 exceptions |
| `make_float4` stores | **4** | lines 250–253 |
| output components | 16 | 4 stores × 4 components |
| products per output component | **33** | top-level `+` count + 1, identical for all 16 |
| total products | **528** | 16 × 33 |
| `mma.sync` / `wmma` / `mma` / `tensor_core` | **0** | grep |
| `.shared` / `shared` / `smem` / `__syncthreads` / `bar.sync` | **0** | grep |

All `MEASURED`. Every load is a 128-bit vectorised access: the cast
`(*((float4*)((ptr+i))))` appears 104 times = 100 loads + 4 stores. The codegen
is **not** scalar. `MEASURED`

**Correction to a common reading of this file: the store lines have 33
products each, not ~64.** 33 = 32 contracted terms + 1. The 32 are
`val36..val67` (one buffer, stride 32) paired with `val68..val99` (another
buffer, stride 32) — 32 float4 each. The +1 is the diagonal, which appears
separately as the bare `exp2` on lines 122/154/186/218 and is multiplied
directly (`val0.x*alu12`) rather than going through a subtraction.

### 1.3 What it computes

Tracing which buffer and which lane feeds which term:

```
out[i][c] = a[i][c] · Σ_{n=0..31} exp2( (t[i] - t[n]) · log2 e ) · u[i][n] · v[n][c]
```

- `a` = `data1_4096`, indexed `[4][4]` — 16 floats
- `t` = `data2_16384`, indexed `[i][n]`, stride 4
- `u` = `data3_16384`, indexed `[i][n]`, stride 32, **lane permuted by `i`**:
  store 250 reads `.y`, 251 reads `.z`, 252 reads `.w`, 253 reads `.x`
- `v` = `data4_4096`, indexed `[n][c]`, stride 32, lane fixed at `c`
- outputs written to `data0_4096` at the same `[4][4]` offsets as `a`

`MEASURED` (reconstructed programmatically from the source text; verified
component-by-component for all 16 outputs.)

**This is a three-operand weighted elementwise reduction over a 32-extent axis
— not a `[C,C]` matmul.** The contracted operand is a per-`n` product of two
scalars, `u[i][n]·v[n][c]`, not a dot product over `n`. That distinction
matters for §4: the thing the tensor-core recogniser is being asked to match is
literally a three-operand product, but as §4 shows, that is *not* what stops it.

The only `.exp()` of a difference of two reshapes of `g_cum` in the port is
`kda/tiny_kda.py:127,137`:

```python
diff = gp - gj
Dl = (diff + (m_strict - 1.0) * 1e4).exp()
```

`gp`, `gj` are the reshaped `g_cum` (`kda/tiny_kda.py:120,125-126`), so this is
`exp(g_cum[p,d] − g_cum[j,d])`, contracted over `d` = K = 32 at this shape —
which matches the kernel's 32-extent reduction and the `EXP2` node sitting
directly under the reduce (§1.4). TinyGrad lowers `exp` to
`exp2(x · log2 e)`, which is where the 128 `1.4426950216293335f` literals come
from — one per transcendental, unshared, re-emitted every time.

`[NOT-HERE]` I did **not** prove the buffer aliasing: I did not trace which
`data*_16384` buffer corresponds to which named tensor in `kda/tiny_kda.py`'s
graph, so the identification of *which KDA step* this kernel implements rests
on the AST shape plus the reduction extent, not on a verified data map. The
§1.3 data-flow reconstruction itself *is* verified — it is derived entirely from
the emitted source.

### 1.4 The transpiled AST, and why the recogniser never gets to look

I instrumented `do_to_program` and dumped the reduce's operand tree for the
kernels that produce this source. `MEASURED`

```
REDUCE(ADD,
  MUL(                      <- reduceop.src[0]
    MUL(EXP2(...), INDEX),  <- the decay weight:  in0
    INDEX),                 <-                   in1
  RANGE)
```

Across the whole 94-kernel schedule the reduce-op signatures are:

| count | `REDUCE.arg[0]` | `src[0].op` | `mul` candidate at `postrange.py:219` | gate passes |
|---|---|---|---|---|
| 52 | `ADD` | `MUL` | `MUL` | **True** |
| 1 | `ADD` | `WHERE` | `WHERE` | False |

`MEASURED` And **zero of the 94 emitted programs contain the string `mma` or
`wmma`.** `MEASURED`

---

## 2. What the hand-written side writes

The comparison target is a vendored extract of
`flash-linear-attention` 0.5.2, `fla/ops/common/chunk_delta_h.py`,
kernel `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`, at
`out/archlab/triton/fla_kda_chunk_h.py`. It is there because the fla 0.5.2
wheel *and* sdist on PyPI ship only `fla/layers` and `fla/models` — `fla/ops`,
where every kernel lives, is missing from both. The file header says so; I did
not re-verify the wheel contents here. `[NOT-HERE]`

### 2.1 The decay — same expression, applied once

```python
# fla_kda_chunk_h.py:223
b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
```

Compare with the generated kernel, which emits the same `exp2` **128 times,
one per contracted term** (§1.2). The hand-written author applies it as a
single elementwise multiply of a `[BT, BV]` tile by a `[BT]` vector, *before*
the contraction. `b_g` is already in log2 units, so there is no `· log2 e`
multiply at all — the generated code pays 128 of those (§1.2).

### 2.2 The contraction — expressed as a matmul

```python
# fla_kda_chunk_h.py:186-189   (state correction)
p_w = w + o_t[:, None] * (HV*K) + o_k1[None, :]
b_w = tl.load(p_w, mask=m_t[:, None] & m_k1[None, :], other=0.0)
b_v = tl.dot(b_w, b_h1.to(b_w.dtype))

# fla_kda_chunk_h.py:263-266   (state update)
p_k = k + o_k1[:, None] + o_t[None, :] * (H*K)
b_k = tl.load(p_k, mask=m_k1[:, None] & m_t[None, :], other=0.0)
b_h1 += tl.trans(tl.dot(b_k, b_v))
```

`tl.dot` is the whole point. **16 `tl.dot` sites** in the file. `MEASURED` The
operands are explicitly down-cast (`.to(b_w.dtype)`, `b_v.to(k.dtype.element_ty)`)
— the author *decides* the precision, and the signature is `*bf16`
(`tools/archlab/fla_kda_chunk_h_sig.json`). `MEASURED`

This is the (i)/(ii) boundary the study cares about, and it is the **Triton
side that has it right**: the contraction is written as a contraction, so
nothing has to be discovered.

---

## 3. SASS, measured

Compiled with `nvcc -arch=… -Xptxas -v` and `cuobjdump -sass` inside
`archlab/nvcc:12.9` (CUDA 12.9, x86-64 under emulation). No device present;
`ptxas` is a pure compiler. `MEASURED`

### 3.1 The generated kernel, per architecture

| arch | SASS | FFMA | FMUL | FADD | **MUFU.EX2** | FSETP.GEU | LDG.E.128 | STG.E.128 | **REG** | **spill st / ld** | STACK | SHARED | barriers | tensor core |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sm_75 | 1552 | 512 | 268 | 124 | **128** | 128 | 100 `.SYS` | 4 `.SYS` | 200 | **0 / 0** | 0 | 0 | 0 | **0** |
| sm_80 | 1560 | 512 | 268 | 124 | **128** | 128 | 100 | 4 | 200 | **0 / 0** | 0 | 0 | 0 | **0** |
| sm_86 | 1560 | 512 | 268 | 124 | **128** | 128 | 100 | 4 | 203 | **0 / 0** | 0 | 0 | 0 | **0** |
| sm_89 | 1560 | 512 | 268 | 124 | **128** | 128 | 100 | 4 | 203 | **0 / 0** | 0 | 0 | 0 | **0** |
| sm_90 | 1592 | 512 | 268 | 124 | **128** | 128 | 100 | 4 | 255 | **48 / 48** | 48 | 0 | 0 | **0** |
| sm_90a | 1592 | 512 | 268 | 124 | **128** | 128 | 100 | 4 | 255 | **48 / 48** | 48 | 0 | 0 | **0** |

sm_90 and sm_90a produce **byte-identical instruction histograms**; only the
arch string differs, and the CUTLASS harness in §5 is the thing that actually
needs the `a` suffix. `MEASURED`

The source↔SASS correspondence is **exact**, not approximate:

| source | SASS | arithmetic |
|---|---|---|
| 100 `float4` loads | 100 `LDG.E.128` | ✅ 100 = 100 |
| 4 `make_float4` stores | 4 `STG.E.128` | ✅ 4 = 4 |
| 128 `exp2(` | 128 `MUFU.EX2` | ✅ 128 = 128 |
| 128 `log2 e` literals | 128 `FSETP.GEU.AND` | ✅ 128 = 128 |
| 124 subtractions `(a−b)` | 124 `FADD` | ✅ 124 = 124 |
| 528 products in the stores | **512 `FFMA` + 16 `FMUL`** | ✅ 528 = 528 |

`MEASURED` for every count on both sides. Two attributions deserve their
reasoning stated rather than assumed:

- The `FSETP.GEU.AND` sits immediately before each `MUFU.EX2` in the
  disassembly (`FSETP.GEU.AND P1, PT, R8, -126, PT` → `MUFU.EX2 R2, R2`), so
  it is the `exp2f` overflow guard, one per transcendental. The count 128
  matches `exp2(` 128; it is *not* the count of the `log2 e` literals, which
  merely happen to also be 128.
- The `FADD` count is 124, not 128. Since a range-guard-ish float operation
  would give 128, and the source has exactly 124 subtractions
  `(valA.w - valB.x)` — the 4 diagonal `exp2`s subtract nothing — 124 identifies
  the `FADD`s as the source-level subtractions. This is inference by
  elimination, not a one-to-one disassembly match; the *count* is measured.

The 16 `FMUL` are the diagonal terms (`val0.x*alu12` etc.), which have no
accumulation to fold into, leaving 512 `FFMA` for the other 512 products.

**Zero tensor-core instructions, on every architecture.** Verified two ways:
zero textual hits in the source, and a `cuobjdump -sass` grep for
`HMMA|QMMA|IMMA|OMMA|BMMA|HGMMA|TCGMMA|UMMA` returning 0. `MEASURED`

One trap worth naming: sm_80 and sm_86 emit **2 `HFMA2.MMA`**. That is *not* a
tensor-core instruction — it is ptxas's idiom for materialising a constant:

```
HFMA2.MMA R161, -RZ, RZ, 0, 2.384185791015625e-07 ;
HFMA2.MMA R0,   -RZ, RZ, 0, 2.384185791015625e-07 ;
```

`RZ` operands, a literal in the multiplier slot. Counting `*MMA` mnemonics
would over-report tensor-core use here. `MEASURED`

### 3.2 The register-pressure hypothesis, tested

The brief predicted ~128 live scalars plus ~100 live `float4`s would spill.
**That is wrong for sm_75 through sm_89, and right for sm_90 and sm_90a.**
`MEASURED`

`ptxas -v` (stderr, authoritative) for sm_75/80/86/89:

```
0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
Used 200 registers, used 0 barriers, 392 bytes cmem[0]
```

for sm_90 and sm_90a:

```
48 bytes stack frame, 48 bytes spill stores, 48 bytes spill loads
Used 255 registers, used 0 barriers, 48 bytes cumulative stack size
```

and the sm_90 SASS shows it directly: **12 `STL` + 12 `LDL.LU`** = 48 B each
way, and nothing like that on sm_80/86/89. `MEASURED`

ptxas absorbs 200–203 registers on Volta-through-Turing-plus and simply runs
out of headroom on Hopper, where it pins at the 255 ceiling and spills the
remainder. Same source, same optimisation, different allocator.

**The brief's warning about `cuobjdump` is right and worth restating with the
exact shape.** On sm_90a, `cuobjdump -res-usage` reports:

```
REG:255 STACK:48 SHARED:0 LOCAL:0 CONSTANT[0]:568
```

`LOCAL:0` is the misleading field. The visible tell is `STACK:48`, which is
absent on every non-spilling arch. `ptxas -v` is the authority.

`MODELLED` occupancy, from measured `REG` plus the 65,536-register / 8-register-
granularity SM budget: sm_75 and sm_80 → 10 resident warps per SM; sm_86 and
sm_89 → 9; sm_90 → 8. The 8-warp figure is the launch configuration
(`__launch_bounds__(128)` = 4 warps/block), so the Hopper spill costs a
resident-warp slot rather than a correctness or capacity failure. **No occupancy
claim here is measured** — no card, and the hardware limits are a transcribed
table.

### 3.3 The hand-written kernel, same toolchain

`tools/archlab/triton_aot.py` AOT-compiles the vendored kernel at
`GPUTarget("cuda", cc, 32)`. `MEASURED`

| cc | PTX target | SASS | **HMMA family** | **HGMMA (wgmma)** | TCGMMA | **MUFU.EX2** | LDS/STS/LDSM | BAR.SYNC | cp.async | REG | spill |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 80 | sm_80 | 2672 | **128** | 0 | 0 | **3** | 342 | 15 | 4 | 255 | 380 B st / 372 B ld |
| 90 | sm_90a | 2632 | 0 | **16** | 0 | **3** | 264 | 15 | 4 | 255 | 380 B st / 372 B ld |
| 100 | sm_100a | 4096 | **32** | 0 | 0 | 66 | 579 | 42 | 6 | — | — |

`MEASURED` (mnemonic counts re-grepped by me from the cubins, not taken from
the tool's summary line.)

Two things to flag honestly:

- **The two kernels are not computing the same function.** The Triton file is
  the fused chunk state-passing kernel (`chunk_delta_h`); the TinyGrad file is
  one reduction from the dense-recompute port. This is a comparison of *how each
  side writes the same two ideas* — a decay and a contraction — not a like-for-
  like algorithmic match, and no timing follows from it either way.
- **The cc=100 SASS is not re-verifiable here.** Triton emits `.version 9.3`,
  and the container's ptxas 12.9 tops out at 8.8:
  `fatal : Unsupported .version 9.3; current version is '8.8'`. The cc=100 cubin
  was produced by Triton's own bundled ptxas. `MEASURED` (the failure) — and
  that is a genuine **(iv)** for the *hand-written* side: the newest instruction
  Triton wants for Blackwell cannot be assembled by the pinned toolchain, while
  the same toolchain assembles the generated kernel for sm_90a without complaint.

---

## 4. Why there is no tensor core here — the recogniser, quoted, and four controls

The recogniser is `tinygrad/codegen/opt/postrange.py:214-220`:

```python
def _apply_tc_opt(self, use_tensor_cores:int, axis:int, tc_select:int, opt_level: int) -> None|list[UOp]:
  if not (reduceops := self.reduceops): raise KernelOptError("no reduce ops for TensorCore")
  reduceop = reduceops[0]
  if use_tensor_cores and reduceop.arg[0] is Ops.ADD:
    mul = reduceop.src[0] if reduceop.src[0].op is not Ops.CAST else reduceop.src[0].src[0]
    if mul.op is not Ops.MUL: return None
    in0, in1 = mul.src
```

**The brief's stated mechanism is wrong.** The KDA decay *does* satisfy this
pattern. §1.4 shows `reduceop.src[0]` is a `MUL`, so line 219 is `PASS`, not
`return None`. The three-operand product is not what blocks it.

The gate that actually fires is the dtype check at line 227:

```python
if tc.dtype_in == in0.dtype and tc.dtype_in == in1.dtype and tc.dtype_out == reduceop.dtype:
```

preceded by line 226:

```python
if self.ren.target.device in ("CUDA", "NV") and tc.dtype_in == dtypes.float and not ALLOW_TF32: continue
```

and the catalogue for the CUDA target is `codegen/opt/tc.py:94-98`:

```python
cuda_sm75: list[TensorCore] = cuda_8168_f16
cuda_sm80: list[TensorCore] = cuda_81616 + cuda_8168_f16 + cuda_8168_tf32
cuda_sm89: list[TensorCore] = cuda_sm80 + cuda_81632_f8
def get_cuda(arch): return cuda_sm89 if (ver:=int(arch[3:])) >= 89 else cuda_sm80 if ver >= 80 else cuda_sm75 if ver >= 75 else []
```

**On sm_75 the entire catalogue is half.** `cuda_8168_f16` is the only entry,
and it is `dtype_in = dtypes.half`. The KDA decay is fp32, so no candidate can
match and the loop falls through to `return None` at line 311. `MEASURED`

Four controlled experiments, each logging every `_apply_tc_opt` call on the
NULL device:

| # | graph | in/out dtypes | line 219 | line 227 dtype match | outcome | `mma` tokens in CUDA C |
|---|---|---|---|---|---|---|
| A | KDA decay shape, **fp32** | f32/f32/f32 | **PASS** | **False** (sm_75 catalogue is half-only) | `None` | 0 |
| B | KDA decay shape, fp16 | f16/f16/f32 | **PASS** | True | `None` (fails later, at range selection / `PADTO`) | 0 |
| C | plain matmul, **fp32** | f32/f32/f32 | **PASS** | **False** | `None` | 0 |
| D | plain matmul, fp16 | f16/f16/f32 | **PASS** | True | **TC APPLIED** | **81** |

`MEASURED` Every call is logged as
`OptOps.TC proposed axis=… arg=(-1, 0, 1)` →
`in0.dtype=… in1.dtype=… reduceop.dtype=… catalog: …` →
`_apply_tc_opt returned None` → `OptOps.TC REJECTED: no tensor core available`.

Reading of the table:

- **A vs C.** A plain fp32 matmul — the easiest possible case, textbook
  `REDUCE(ADD(MUL(a,b)))` — is blocked by exactly the same line 227 as the KDA
  decay. The shape is not the variable. **fp32 is.**
- **D.** With fp16 the identical pipeline emits 81 `wmma`. Nothing is broken.
- **B.** The three-operand product *does* cost something, but only once the
  dtypes already line up; it is blocked downstream at the range-picking and
  `PADTO` stage (lines 235-255), not at line 219. This is where the brief's
  (ii) instinct has real purchase — but it is not what happens at fp32, which
  is the dtype KDA actually runs in.

### 4.1 `ALLOW_TF32`, and an artefact of measuring on the NULL device

`cuda_8168_tf32` *does* exist, with `dtype_in = dtypes.float`, and it is in the
sm_80+ catalogue. `MEASURED` (via `tc.get_cuda`, which I called directly for
sm_75/80/86/89/90/100/120.)

But `helpers.py:281` is `ALLOW_TF32 = ContextVar("ALLOW_TF32", 0)` — off by
default — and line 226 skips every float tensor core when it is off *on a
device named CUDA or NV*. My NULL-device runs report `target.device == "NULL"`,
so **line 226 does not fire and I measure TF32 being applied when a real
device would not.** I tested this rather than assuming it: forcing
`target.device` to `"CUDA"` and re-running case C at `ALLOW_TF32=0` gives
**0 of 5 kernels containing `mma`**. `MEASURED`

So the correct statement is:

| condition | fp32 matmul reaches a tensor core? |
|---|---|
| sm_75, `ALLOW_TF32` 0 or 1 | **No** — no float entry in the catalogue at all |
| sm_80+, real CUDA device, `ALLOW_TF32=0` (default) | **No** — line 226 skips `cuda_8168_tf32` |
| sm_80+, `ALLOW_TF32=1` | **Yes**, in principle — `cuda_8168_tf32` matches at line 227 |

Whether an fp32 KDA decay would then reach a TC is **not established**: case A
at fp16 (B) shows a three-operand product still fails downstream, and I have no
measurement of case A with TF32 forced on. `[NOT-HERE]`

**And this is not, on its own, a defect.** NVIDIA has no native fp32 tensor
core; fp32 SIMT is the correct lowering for an fp32 matmul. The interesting
part is that the *hand-written* side reaches tensor cores by **authoring** in
bf16 (§2.2) — a decision made in the source, visible in the signature, not
something a pass has to infer. That asymmetry, not a missed optimisation, is
the finding.

### 4.2 The (iv) that is not there

`matmul/` reported that TinyGrad builds `sm_{major}{minor}` with no
arch-conditional suffix. Confirmed: `runtime/ops_cuda.py:122` is

```python
arch=f"sm_{major.value}{minor.value}"
```

and `tc.get_cuda` does `int(arch[3:])`, so an `"a"` suffix would raise. Called
directly: `tc.get_cuda('sm_90a')` → `ValueError: invalid literal for int()
with base 10: '90a'`. `MEASURED` Because the arch string never carries the
suffix, that `ValueError` is unreachable in a real run: **on an H100 TinyGrad
selects the sm_89 (Ada) catalogue, which additionally advertises
`fp8e4m3`/`fp8e5m2` shapes that were never selected for by arch.**

`grep -rl` over the whole installed package for `wgmma`, `tcgen05`, `cp.async`,
`tensormap`, `stmatrix`, `ldmatrix`: **0 files each.** `MEASURED`

So there is no Hopper- or Blackwell-specific machinery to reach for.

**Does this kernel compile for sm_90a? Yes.** §3.1: 255 registers, 48 B of
spill, clean build. The brief expected a failure and there is none. This
artifact therefore supplies **no (iv) instance**. The (iv) that does exist in
this comparison is on the other side of the table — the hand-written kernel's
sm_100a PTX, which the pinned ptxas cannot assemble (§3.3).

---

## 5. CUTLASS on the same reduction

Rebuilt from source here, not read out of `cutlass_lab/RESULTS.md`.

`harness/print_cfg.cu` instantiates the same `device::Gemm` the harness uses
and prints compile-time constants. Re-run for all five rungs, fp32 SIMT
(`LAB_MODE=0`). `MEASURED`

| | sm_750 | sm_800 | sm_890 | sm_900 |
|---|---|---|---|---|
| threadblock tile | `<128,128,8>` | `<128,128,8>` | `<128,128,8>` | `<128,128,8>` |
| warp tile | `<32,64,8>` | `<32,64,8>` | `<32,64,8>` | `<32,64,8>` |
| mma instruction | `<1,1,1>` | `<1,1,1>` | `<1,1,1>` | `<1,1,1>` |
| pipeline stages | 2 | 2 | 2 | 2 |
| alignment A/B | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| warps per block | 8 | 8 | 8 | 8 |
| threads per block | 256 | 256 | 256 | 256 |
| warps M×N | 4 × 2 | 4 × 2 | 4 × 2 | 4 × 2 |
| mma per warp k-step | 8 | 8 | 8 | 8 |
| kernel smem bytes | **16,896** | **16,896** | **16,896** | **16,896** |
| — epilogue | 9,280 | 9,280 | 9,280 | 9,280 |
| — mainloop | 7,616 | 7,616 | 7,616 | 7,616 |
| smem / capacity | not recorded | not recorded | not recorded | 16,896 / 232,448 = **7.3%** |

The brief's figures — 128×128×8, 8 warps, 16,896 B — are **confirmed by
independent re-derivation**. Note `mma instruction <1,1,1>`: CUTLASS's own
fp32 SIMT "MMA" is a scalar multiply-add, i.e. CUTLASS agrees that fp32 does not
get a tensor core.

Compiled and disassembled (`harness/sgemm.cu`, `-arch=… -Xptxas -v`): `MEASURED`

| arch | REG | STACK | SHARED (static) | barriers | SASS | FFMA | STS | LDS | tensor core |
|---|---|---|---|---|---|---|---|---|---|
| sm_75 | 133 | 0 | 0 | 1 | 2048 | 576 | 144 | 128 | **0** |
| sm_80 | 136 | 0 | 0 | 1 | 2344 | 576 | 144 | 128 | **0** |
| sm_89 | 136 | 0 | 0 | 1 | 2328 | 576 | 144 | 128 | **0** |
| sm_90a | 136 | 0 | 1024 | 1 | 2368 | 576 | 144 | 128 | **0** |

**Caveat on the 16,896 B.** `ptxas` reports `SHARED:0` (sm_75/80/89) or
`SHARED:1024` (sm_90a) because CUTLASS allocates its tile staging as *dynamic*
(`extern __shared__`). ptxas cannot see a launch-time size, which is why
`tools/archlab/inspect_ptx.py` lists dynamic smem under "what is not measured".
The 16,896 B figure is the compile-time `SharedStorage` size printed by
`print_cfg`, and I could not corroborate it from the cubin. `[NOT-HERE]`

**The comparison that matters is the staging, not the tile size.** CUTLASS's
fp32 SIMT path spends 144 `STS` + 128 `LDS` + 1 `BAR.SYNC` to get a 128×128
output tile out of a 2-stage shared-memory pipeline. The TinyGrad kernel spends
**0 `STS`, 0 `LDS`, 0 `BAR.SYNC`, 0 bytes of shared memory**, because
`__launch_bounds__(128)` with one block in x leaves 4 warps that never
communicate and every operand in registers. Neither is wrong: TinyGrad's tile is
4×4 with a 32-long K, and a 4×4 tile has nothing to stage. But it does mean the
generated kernel has **no mechanism by which a shared-memory pipeline could be
expressed**, which is **(iii)**, and `cp.async` returning 0 files across the
package (§4.2) is the direct evidence.

---

## 6. Generated vs hand-written, by dimension

Generated = TinyGrad `r_2_2_8_8_4_4_32`, fp32, sm_80 numbers unless noted.
Hand-written = fla `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`, bf16
signature. CUTLASS = fp32 SIMT SGEMM, sm_80.

| dimension | generated | hand-written (Triton) | CUTLASS fp32 SIMT | tag |
|---|---|---|---|---|
| **instruction class** | 512 FFMA, 268 FMUL, 124 FADD, 128 MUFU.EX2. **0 tensor core.** | 128 HMMA (sm_80) / 16 HGMMA (sm_90a). 3 MUFU.EX2. | 576 FFMA. **0 tensor core** (`mma <1,1,1>`). | (ii) for the generated fp32 path; the hand-written side gets it by authoring in bf16 |
| **special-function units** | **128 `MUFU.EX2` — 12.40% of FP instructions, 8.21% of all SASS** | **3 `MUFU.EX2` — 0.11% of SASS** | 0 `MUFU.EX2` | (ii) — decay is inside the reduction instead of a pre-pass |
| **decay placement** | recomputed per contracted term: `exp2((t[i]−t[n])·log2e)` × 128, each with its own `log2 e` multiply | one elementwise pre-pass: `b_v * where(m_t, exp2(b_g_last − b_g), 0)[:, None]`, no `log2 e` at all | n/a | (ii) |
| **memory staging** | **0 `STS`, 0 `LDS`, 0 `BAR.SYNC`, 0 B shared memory** | 342 LDS/STS/LDSM, 15 `BAR.SYNC`, 4 `cp.async` | 272 STS+LDS, 1 `BAR.SYNC`, 2-stage pipeline, 16,896 B dynamic smem | **(iii)** — no shared-memory or async-copy machinery exists in the backend |
| **vectorisation** | 100/100 `LDG.E.128`, 4/4 `STG.E.128` — every access 128-bit | 160 global loads, **all scalar width** | 36 `LDS.128` of 128 | — (the generated side is *better* here) |
| **reduction mechanism** | fully unrolled: 33 products × 16 outputs, **528 FMA/FMUL, no loop** | `tl.dot` (21 sites) → `mma`/`wgmma` | warp-tile `MmaSimt<32,64,8>`, 8 mma per warp k-step | (ii) |
| **register footprint** | 200 (sm_75/80), 203 (sm_86/89), **255 + 48 B spill (sm_90/90a)** | 255, 380 B st / 372 B ld spill | 133–136, 0 spill | (iii) — no smem alternative to fall back on, so it spills instead |
| **shared footprint** | **0 B** | not measured here `[NOT-HERE]` | 16,896 B dynamic (compile-time constant) | (iii) |
| **shape specialisation** | total: 0 `if`, 0 `for`, 0 `while`, 0 bounds check. Grid is `blockIdx.x` only, 2 blocks. | `tl.constexpr` on `H HV K V BT BV USE_G USE_GK …`; masks `m_t`, `m_k1` present throughout | runtime M/N/K, one binary for all | — (specialisation is *correct*, not a defect) |
| **bounds handling** | none needed — sizes are compile-time | explicit `tl.load(..., mask=…, other=0.0)` on every load | predication + `cp.async` predication | — |
| **arch coverage** | one source compiles for sm_75…sm_90a unchanged; the *same* 512/268/124/128 counts on every arch | cc 80/90/100 all compile, but reach *different* tensor-core families (128 HMMA → 16 HGMMA → 32 HMMA) | sm_75…sm_100a all compile | **(iv)** on the hand-written side at cc=100: `.version 9.3` PTX vs ptxas 12.9's 8.8 ceiling |
| **tries tensor cores at all** | yes — `OptOps.TC` is proposed for every axis, then rejected at `postrange.py:227` | n/a — `tl.dot` is the source-level primitive | n/a — the operator class is chosen by the caller | (i) never; (ii) tries and misses |

### 6.1 The SFU arithmetic

`MUFU.EX2` share of the generated kernel's floating-point work, sm_80: `MEASURED`

```
512 FFMA + 268 FMUL + 124 FADD + 128 MUFU.EX2 = 1032 float instructions
MUFU share by instruction count  = 128 / 1032 = 12.40%
MUFU share of all SASS           = 128 / 1560 =  8.21%
```

`MODELLED`, weighting for throughput: on sm_8x and sm_9x each sub-partition has
4 SFU lanes against 32 FP32 lanes, so a warp `MUFU.EX2` occupies 8 issue cycles
for every 1 that an `FFMA` does. That hardware constant is from the CUDA C
Programming Guide arithmetic-instruction table and is **not measured on this
host**.

```
time-equivalent = 8*128 / (8*128 + 512 + 268 + 124) = 1024 / 1928 = 53.1%
```

So the special-function unit is a small minority of the instructions and, on
that arithmetic, a majority of the arithmetic time. **This is a modelled
figure.** It says nothing about achieved clock, memory stalls, or the actual
critical path, and there is no timing anywhere in this note.

### 6.2 Where the 128 `exp2` go

`MEASURED`

```
528 products, 128 exp2  ->  0.242 exp2 per product
16 output components    ->  8 exp2 each
4 store lines           ->  32 exp2 each (1 bare diagonal + 31 off-diagonal)
33 products per component -> 32 contracted terms + 1 diagonal
```

A 528-product matmul needs 0 `exp2`. The generated kernel needs 128 because the
graph is `Σ_n exp2(Δ_n)·u_n·v_n` and `exp2(Δ_n)` must be a *value* before it can
multiply anything. The hand-written author writes the same maths as
`b_v * decay[:, None]` — one vector, one multiply, then a pure `tl.dot` — and
pays 3 `exp2` on an 8× larger kernel. That is a construction difference, not a
scheduling one, and it is the difference this note exists to name.

---

## 7. What this artifact cannot show

Stated plainly, because the temptation to over-read it is strong.

1. **It is one kernel, not the operator's schedule.** The 128 `exp2` are a
   property of *this* program. Whether the operator as a whole recomputes the
   decay more than necessary is a **separate and unsettled claim.** What I can
   say: at the capture shape the whole in-process schedule is 94 kernels /
   2,628 `exp2` (§0), and `perf_kda` counts 12,082 kernel launches per call at a
   *different, larger* shape (§0). Those are three numbers about three
   configurations. This note does not adjudicate between them.

2. **The capture saw one kernel; the schedule has 94.** The original capture's
   `compile_cached` hook only observes rounds where `len(todo) == 1`
   (`realize.py:273`). Every claim here about "the generated kernel" is a claim
   about `r_2_2_8_8_4_4_32` specifically, chosen because it is the one that was
   captured — not a sample, not a median, and not evidence about the other 93.

3. **The two sides do not compute the same function.** The Triton file is the
   fused chunk state-passing kernel; the TinyGrad file is one reduction from a
   dense-recompute port. §3.3 states this and the comparison is structural
   (how each side writes a decay and a contraction), not algorithmic.

4. **No timings. No GPU on this host.** Every number is a static property:
   instruction counts, register and shared-memory allocation, spill stores and
   loads, and which instruction families the compiler chose. Nothing here
   measures latency, throughput, occupancy in practice, bandwidth, or
   achieved clock. The `MODELLED` occupancy in §3.2 and the `MODELLED` SFU
   weighting in §6.1 are the only non-measured figures, and both are labelled
   as such with their unmeasured input named.

5. **Compilation is not correctness.** `ptxas` validates types, control flow
   and alignment. It does not validate fragment layouts or data mapping. A kernel
   that assembles cleanly can compute garbage. Nothing in this note says the
   generated kernel is *right*, only what shape it is.

6. **I did not re-verify the fla packaging claim.** The statement in the §2
   preamble that the 0.5.2 wheel and sdist omit `fla/ops` is taken from the
   vendored file's own header, not re-tested. Nor is the kernel numerically
   validated — it is inspected and compiled, never executed.

7. **The CUTLASS 16,896 B is not corroborated by the cubin** (§5), because
   CUTLASS stages through dynamic shared memory. It is a compile-time constant
   from a host binary, which is the right instrument for it, but it is a single
   source.

8. **The TF32 question is open.** §4.1 establishes that fp32 does not reach a
   tensor core on a real CUDA device at default settings, and that a float
   tensor core exists in the sm_80+ catalogue. Whether an fp32 KDA decay would
   reach it with `ALLOW_TF32=1` is untested. `[NOT-HERE]`

---

## 8. Corrections to the brief this note was written from

Recorded because the brief this note was written from carried several
load-bearing claims, and the measurements either contradict them or replace
them with something sharper. The last two rows are confirmations; the rest are
corrections.

| brief said | measured |
|---|---|
| "5 float4-typed arguments" | 5 **`float*`** arguments; the `float4` is in the 104 accesses, not the signature |
| "four `make_float4` stores, each a fully unrolled sum of **~64 products**" | **33** products each, 528 total (= 32 contracted + 1 diagonal) |
| "the KDA decay is a `[C,C]` product" | this kernel is a three-operand weighted elementwise reduction over a **32-extent** axis; `u[i][n]·v[n][c]`, not a dot product |
| "128 live scalars + 100 live `float4`s" — "ask what this compiles to … and whether it spills" | asked, not asserted. Answer: **no spill on sm_75/80/86/89** (200–203 regs); **spills only on sm_90/sm_90a** (255 regs, 48 B st + 48 B ld, 12 `STL` + 12 `LDL.LU`) |
| "`REDUCE(ADD(MUL(tile,tile)))` … this three-operand product does not match, so it falls back to scalar FMA" | it **does** match — `postrange.py:219` is `PASS`. The blocker is **`postrange.py:227`**, dtype: the sm_75 CUDA catalogue is half-only, and `ALLOW_TF32=0` (default) skips float TCs on a real CUDA device |
| "Check whether this kernel compiles at all for sm_90a, and if not, say so" | **it compiles cleanly** — 255 regs, 48 B spill. No (iv) instance from this artifact. The (iv) is on the Triton side at cc=100 |
| "`fla` … read the kernel from a git checkout or from whatever `portability/` and `cutlass_lab/` already vendored" | the vendored extract is at `out/archlab/triton/fla_kda_chunk_h.py`; it AOT-compiles at cc 80/90/100 with the repo's own tooling |
| CUTLASS fp32 SIMT "128×128×8, 8 warps, 16,896 B" | **confirmed** by re-running `print_cfg.cu` for all five rungs; plus 2 stages, warp tile `<32,64,8>`, mma `<1,1,1>`, epilogue 9,280 / mainloop 7,616 |

---

## 9. Reproducing every number in this note

No GPU. Host CUDA is not installed; the toolchain runs in
`archlab/nvcc:12.9` (`tools/archlab/build_image.sh`).

```bash
# 0. the artifact and its reproduction
cp out/kaggle/20260928-014248-gpu/codegen/kernel000.txt /tmp/k.cu
DEV=NULL:CUDA:sm_75 PARALLEL=0 .venv/bin/python <capture-hook script>
#   -> 253 lines, 24,968 B, 100 float4 loads, 128 exp2, 4x33-product stores

# 1-3. SASS, registers, spills, per arch
docker run --rm --platform linux/amd64 -v /tmp:/in:ro \
  -e PATH=/usr/local/cuda-12.9/bin:/usr/bin:/bin archlab/nvcc:12.9 \
  nvcc -arch=sm_80 -Xptxas -v -cubin /in/k.cu -o /tmp/k.cubin   # stderr has the spills
docker run --rm ... cuobjdump -sass /tmp/k.cubin

# 4. the recogniser, quoted and controlled
.venv/bin/python -c "import tinygrad.codegen.opt.postrange as p, inspect; \
  print(inspect.getsource(p.Scheduler._apply_tc_opt))"
.venv/bin/python -c "from tinygrad.codegen.opt import tc; print(tc.get_cuda('sm_75'))"
.venv/bin/python -c "from tinygrad.codegen.opt import tc; tc.get_cuda('sm_90a')"   # ValueError

# 3.3. the hand-written side
docker run --rm --platform linux/amd64 -v tools/archlab:/lab:ro -v <dir>:/aot \
  -e PATH=/usr/local/cuda-12.9/bin:/usr/bin:/bin archlab/compile:12.9 \
  python3 /lab/triton_aot.py --file /aot/fla_kda_chunk_h.py \
    --signature /aot/fla_kda_chunk_h_sig.json --cc 90 --outdir /aot/out90

# 5. CUTLASS
docker run --rm --platform linux/amd64 -v cutlass_lab/cutlass:/cutlass:ro \
  -v cutlass_lab/harness:/harness:ro -e PATH=/usr/local/cuda-12.9/bin:/usr/bin:/bin \
  archlab/nvcc:12.9 sh -c 'nvcc -DLAB_SM=900 -DLAB_MODE=0 -I/cutlass/include \
    -I/cutlass/tools/util/include -I/harness -o /tmp/pc /harness/print_cfg.cu && /tmp/pc'
```

**No timings are reported in this note, and none can be: there is no GPU on
this host.** Static compilation and disassembly only.
