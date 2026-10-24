# Matmul: TinyGrad-generated vs CUTLASS-hand-written, across the CUDA arch ladder

Study question: *at what architectural rung does a compiler stop emitting the
hardware's intended instruction classes, and does the hand-written kernel use
them?*

Everything below was measured by compiling both sides with the **same** ptxas
(12.9.86) and disassembling with `cuobjdump`/`nvdisasm`. There was **no GPU**,
so there are **no timings** — see [Limitations](#limitations).

---

## 1. The fork: does TinyGrad generate matmul, or call a library?

**TinyGrad 0.14.0 generates a matmul kernel. It does not call cuBLAS, cuBLASLt
or any vendor GEMM.** The comparison is meaningful.

Evidence, from the installed source in `.venv`:

| Claim | Evidence |
|---|---|
| No cuBLAS anywhere in the package | `grep -r cublas .venv/.../tinygrad/` → **zero hits** (case-insensitive, all files) |
| No GEMM UOp | `grep -r "MATMUL\|Matmul"` over `uop/`, `schedule/` → **zero hits**. There is no `Ops.MATMUL`. |
| No matmul lowering pass | `tinygrad/schedule/` contains only `allreduce, indexing, memory, multi, rangeify` — nothing matmul-specific. |
| `dot` is a broadcast-multiply plus a reduction | `tinygrad/mixin/op.py:392`: `return (x*w).sum(-1, dtype=dtype).cast(...)` |

So `Tensor.matmul` → `Tensor.dot` → a fused `MUL` + `REDUCE_AXIS`, and the whole
GEMM is produced by TinyGrad's general-purpose kernel generator. The kernel is
real: for 4096³ fp16 it comes out as `r_64_32_32_4_2_2_4_4_256_2`, 128 threads,
a 64-float per-thread accumulator, and a 4-deep unrolled reduction loop.

**Consequence:** there is no 1.0x artefact to discount. The TinyGrad-vs-human
gap is a real codegen gap, and the instruction-level differences below are
attributable to the compiler, not to a library boundary.

One caveat on how the source was obtained: `Tensor.realize()` filters out
already-realised uops (`tinygrad/tensor.py:422`) and is a **silent no-op** on an
already-realised graph — no kernel, no error. All captures therefore use
`Tensor.schedule_linear()`, which unconditionally builds the linear program.
See `tg_backend.py` for the capture-only device and the full argument.

---

## 2. Toolchain and what it substitutes for

| Role | Used | Note |
|---|---|---|
| PTX assembler | **ptxas 12.9.86** | from the `nvidia-cuda-nvcc-cu12` pip wheel. This is NVIDIA's own assembler, so the *instruction-class lowering* in the table below is authoritative. |
| SASS disassembler | `cuobjdump` 12.9.82, `nvdisasm` 12.9.88 | from NVIDIA's conda channel (`fetch_conda_pkg.sh`) |
| CUDA frontend | **clang-20**, not nvcc | **Substitution.** nvcc is genuinely unobtainable in this environment: the pip nvcc wheel ships *only* ptxas, and NVIDIA's conda channel publishes `cuda-nvcc` / `cuda-nvcc_linux-64` as metapackages containing nothing but activate scripts. |
| Arch support | sm_75, sm_80, sm_89, sm_90a, sm_100a all supported by ptxas 12.9.86 | **No target was skipped.** |

The clang substitution matters in one place and only there: CUTLASS 4.8 gates all
Blackwell tensor-core support on `#if !CUTLASS_CLANG_CUDA`
(`cutlass/arch/config.h:87`) and therefore refuses to emit `tcgen05` under clang.
We pass `-DCUTLASS_ARCH_MMA_SM100A_ENABLED=1`, which is *correct* here (clang-20
targets sm_100a and does define `__CUDA_ARCH_FEAT_SM100_ALL`), but it means the
sm_100a CUTLASS cell was produced with an override NVIDIA does not sanction and
we could not cross-check against an nvcc build. Flagged again in Limitations.

Two documented CUTLASS patches were needed, both outside any GEMM inner loop:
* `cutlass2x/include/cutlass/matrix.h`: `m.set_slice3x3(...)` → `set_slice_3x3`
  in `Matrix::rotation`/`reflection` (nvcc tolerates the nonexistent name).
* `cutlass/include/cute/arch/copy_sm100.hpp`: 12 malformed inline-asm operand
  lists on `tcgen05.cp` helpers (`:` immediately followed by `:`), which clang
  rejects. Fix is `: : "r"(a), "l"(b)`.
  Both are in `patch_cutlass2x.sh` / `patch_cutlass.sh` and are diffed in-place.

---

## 3. THE DELIVERABLE: feature-usage matrix

Cell = number of matching instructions in the **SASS** (via `nvdisasm -c`).
`~n` = present in the PTX but folded to nothing by ptxas. `·` = absent.
This is a *code-shape* measurement, not a dynamic-instruction count.

| arch | side | dtype | FFMA (FP32 FMA pipe) | MMA.sync (HMMA/IMMA) | cp.async / LDGSTS | TMA | wgmma | tcgen05 | ldmatrix | stmatrix | elect | cluster | FP8 | blk-scaled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **sm_75** | TinyGrad | fp32 | **64** | · | · | · | · | · | · | · | · | · | · | · |
| **sm_75** | TinyGrad | fp16 | · | **16** | · | · | · | · | · | · | · | · | · | · |
| **sm_75** | CUTLASS | fp32 | **576** | · | · | · | · | · | · | · | · | · | · | · |
| **sm_75** | CUTLASS | fp16 | 128 | **128** | · | · | · | · | **20** | · | · | · | · | · |
| **sm_80** | TinyGrad | fp32 | **64** | · | · | · | · | · | · | · | · | · | · | · |
| **sm_80** | TinyGrad | fp16 | · | **16** | · | · | · | · | · | · | · | · | · | · |
| **sm_80** | CUTLASS | fp32 | **192** | · | **23** | · | · | · | · | · | · | · | · | · |
| **sm_80** | CUTLASS | fp16 | 128 | **64** | **36** | · | · | · | **24** | · | · | · | · | · |
| **sm_89** | TinyGrad | fp32 | **64** | · | · | · | · | · | · | · | · | · | · | · |
| **sm_89** | TinyGrad | fp16 | · | **16** | · | · | · | · | · | · | · | · | · | · |
| **sm_89** | CUTLASS | fp32 | **192** | · | **23** | · | · | · | · | · | · | · | · | · |
| **sm_89** | CUTLASS | fp16 | 128 | **64** | **36** | · | · | · | **24** | · | · | · | · | · |
| **sm_90a** | TinyGrad | fp32 | **64** | · | · | · | · | · | · | · | · | · | · | · |
| **sm_90a** | TinyGrad | fp16 | · | **16** | · | · | · | · | · | · | · | · | · | · |
| **sm_90a** | CUTLASS | fp16 | 64 | · | ~2 | **2** | **8** | **2** | · | · | **7** | **8** | · | · |
| **sm_100a** | TinyGrad | fp32 | **64** | · | · | · | · | · | · | · | · | · | · | · |
| **sm_100a** | TinyGrad | fp16 | · | **16** | · | · | · | · | · | · | · | · | · | · |
| **sm_100a** | CUTLASS | fp16 | 64 | · | ~29 | **12** | · | **20** | **8** | **8** | **30** | **32** | · | · |

Notes on the cells:
* Blackwell `tcgen05` MMA disassembles as **`UTCHMMA.2CTA`** (8 of them), not
  `UMMA` and not `tcgen05` — `tcgen05` is the PTX spelling only. The
  tensor-memory traffic is `UTMALDG` / `SM100_TMEM_LOAD_16dp256b2x`.
* `~29` cp.async at sm_100a is *multicast* (`cp.async.bulk.tensor` 2-CTA) which
  ptxas folds into `UTMALDG.3D.MULTICAST.2CTA`.
* CUTLASS's own SM100 kernel was resolved by `KernelScheduleAuto` to
  `MainloopSm100TmaUmmaWarpSpecialized<13, 2, 4, C<2>, 1>` — 13 stages, 2-CTA
  MMA, `SM100_MMA_F16BF16_2x1SM_SS`, `SM100_TMA_2SM_LOAD_MULTICAST`.
* CUTLASS's SM90 kernel resolved to `MainloopSm90TmaGmmaWarpSpecialized<7>` +
  `KernelTmaWarpSpecializedCooperative` + `SM90_4xGMMA MMA_64x128x16` +
  `SM90_TMA_LOAD_MULTICAST` — 7 stages.
* TinyGrad's row is **invariant across all five targets**. Verified by md5: the
  fp16 4096³ source for sm_80, sm_89, sm_90 and sm_100 is **byte-identical**
  (`9202b0d2…`) and all four issue
  `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`. Only sm_75 differs, and
  only in the MMA shape — `m16n8k8`, because Turing's half-precision
  `mma.sync` tops out at K=8, and TinyGrad's arch-gated tensor-core catalog
  (`cuda_8168_f16`) correctly encodes that. So TinyGrad *does* react to arch, but
  only to a fact expressible as "which `mma.sync` shape fits", and never to a
  fact that requires an arch-conditional target or a new instruction class. Its
  sm_80…sm_100a SASS contains the same instruction-class set as its sm_75 SASS.
* FP8 and block-scaled are empty everywhere because *neither* side was asked for
  them (both baselines are fp16). TinyGrad's `codegen/opt/tc.py` does contain an
  fp8 tensor-core shape (`cuda_81632_f8`, unscaled, sm_89+), so the FP8 column is
  "not exercised", not "cannot express".

### Software-pipeline depth actually emitted

Provenance differs between the two halves of this table, and it matters: for
sm_75/80/89 the stage count is a value **I chose** and CUTLASS accepted (a
CUTLASS 2.x stage count is a template specialisation, so a successful compile
proves it was honoured — but it is not something I measured from the code). For
sm_90a/sm_100a I left `StageCountAuto` to CUTLASS and **read the result out of
the kernel's own mangled name**, so those two are measurements.

| arch | side | stages | provenance / evidence |
|---|---|---|---|
| sm_75 | TinyGrad | **0** | no `cp.async` at all; the K loop is a plain `for` with synchronous `LDG` |
| sm_75 | CUTLASS | 2 (fp16), 2 (fp32) | configured by me; `threadblock::MmaMultistage` in the mangled name |
| sm_80 / sm_89 | TinyGrad | **0** | no `cp.async` |
| sm_80 / sm_89 | CUTLASS | 4 (fp16), 3 (fp32) | configured by me; 36 `LDGSTS` present, so the multistage pipeline is real |
| sm_90a | TinyGrad | **0** | — |
| sm_90a | CUTLASS | **7** | **measured**: `MainloopSm90TmaGmmaWarpSpecialized<7>` |
| sm_100a | TinyGrad | **0** | — |
| sm_100a | CUTLASS | **13** | **measured**: `MainloopSm100TmaUmmaWarpSpecialized<13, 2, 4, C<2>, 1>` |

All six CUTLASS 2.x kernels resolve to `threadblock::MmaMultistage`, including
the sm_75 ones. (`MmaPipelined` — the older smem→register double-buffered
mainloop — would have been the natural Turing choice, but `MmaMultistage` with 2
stages is what the Sm75 specialisation in 2.11 accepts, and it still cannot use
`cp.async` because the instruction does not exist on Turing: the sm_75 CUTLASS
rows correctly show zero `LDGSTS`.)

The single cleanest number in the study: the human pipeline depth goes
2 → 4 → 7 → 13 as features accumulate. The compiler's is 0 at every rung.

---

## 4. Registers / shared memory / spills

From `ptxas -arch=<arch> -v`, and `cuobjdump -res-usage`. `stack` is the stack
frame ptxas reserved; CUTLASS 2.x/3.x request *dynamic* shared memory, which
`cuobjdump -res-usage` reports as `SHARED:0` — the static figure below is only
the statically-allocated part, so **the real smem for the CUTLASS rows is
understated** and is not captured by these tools.

| arch | side | kernel | regs | spill stores | spill loads | stack frame | static smem | stages |
|---|---|---|---|---|---|---|---|---|
| sm_75 | TinyGrad | fp32 4096³ `r_128_64_8_16_4_4_1024_4` | 62 | 0 | 0 | 0 | 0 | 0 |
| sm_75 | TinyGrad | fp16 4096³ `r_64_32_32_4_2_2_4_4_512` | 94 | 0 | 0 | 0 | 0 | 0 |
| sm_75 | TinyGrad | bf16 4096³ | 59 | 0 | 0 | 0 | 0 | 0 |
| sm_75 | TinyGrad | fp32 128³ `r_4_2_8_16_4_4_32_4` | 62 | 0 | 0 | 0 | 0 | 0 |
| sm_75 | CUTLASS | SIMT fp32 (2.11) | 126 | 0 | 0 | 512 | 0 (dynamic) | 2 (cfg) |
| sm_75 | CUTLASS | tensor-op fp16 (2.11) | 254 | **472** | **472** | 240 | 0 (dynamic) | 2 (cfg) |
| sm_80 | TinyGrad | fp32 4096³ | 62 | 0 | 0 | 0 | 0 | 0 |
| sm_80 | TinyGrad | fp16 4096³ | 94 | 0 | 0 | 0 | 0 | 0 |
| sm_80 | CUTLASS | SIMT fp32 (2.11) | 126 | 0 | 0 | 512 | 0 (dynamic) | 3 (cfg) |
| sm_80 | CUTLASS | tensor-op fp16 (2.11) | 254 | **472** | **472** | 240 | 0 (dynamic) | 4 (cfg) |
| sm_89 | TinyGrad | fp32 4096³ | 62 | 0 | 0 | 0 | 0 | 0 |
| sm_89 | TinyGrad | fp16 4096³ | 94 | 0 | 0 | 0 | 0 | 0 |
| sm_89 | CUTLASS | SIMT fp32 (2.11) | 126 | 0 | 0 | 512 | 0 (dynamic) | 3 (cfg) |
| sm_89 | CUTLASS | tensor-op fp16 (2.11) | 254 | **472** | **472** | 240 | 0 (dynamic) | 4 (cfg) |
| sm_90a | TinyGrad | fp32 4096³ | 56 | 0 | 0 | 0 | 0 | 0 |
| sm_90a | TinyGrad | fp16 4096³ | 92 | 0 | 0 | 0 | 0 | 0 |
| sm_90a | CUTLASS | 4.8 `KernelTmaWarpSpecializedCooperative` | 168 | 88 | 248 | 944 | 1024 (+dynamic) | 7 (auto) |
| sm_100a | TinyGrad | fp32 4096³ | 56 | 0 | 0 | 0 | 0 | 0 |
| sm_100a | TinyGrad | fp16 4096³ | 95 | 0 | 0 | 0 | 0 | 0 |
| sm_100a | CUTLASS | 4.8 `MainloopSm100TmaUmmaWarpSpecialized<13>` | 80 | 0 | 0 | 1584 | 1024 (+dynamic) | 13 (auto) |

`(cfg)` = a stage count I configured; `(auto)` = chosen by CUTLASS and read out of
the mangled kernel name.

Two honest observations, neither flattering to the human side:
* The CUTLASS **2.x fp16** baseline spills (472 B stores, 472 B loads) on every
  target. That is an artifact of the tile I copied from an example, not of
  CUTLASS in general — an autotuner would not have kept it. It is exactly why
  the study must not read register counts as a quality ranking.
* TinyGrad never spills and never uses shared memory, so it looks "clean" in
  this table while being far slower. Register/spill counts are not a quality
  metric across such different strategies.

### Load vectorisation (max bits moved by one global load)

| arch | TinyGrad fp32 | TinyGrad fp16 | CUTLASS fp16 |
|---|---|---|---|
| sm_75 | 128 (`float4`) | **32 (`LDG.E.U16`)** | 128 |
| sm_80 / sm_89 | 128 | **32** | 128 |
| sm_90a / sm_100a | 128 | **32** | 64 / 32 (TMA moves the data instead) |

TinyGrad vectorises fp32 perfectly (4×`float4` per K step) and then fails to
vectorise fp16 at all, falling back to **scalar 16-bit global loads** — 16 of
them per K iteration. This is a pure scheduling/vectorisation miss on code whose
vocabulary (`half4`, `half8` structs) is already present in the generated file.

---

## 5. The two sides side by side (sm_80, fp16, 4096³)

**TinyGrad's actual generated CUDA C** (`gen/sm_80_big_4096x4096x4096_float16_to_float32_default__0.cu`,
lines 1–16 and the K loop):

```cuda
struct __align__(8) half4 { half x, y, z, w; }; __device__ half4 make_half4(...) {...}
__device__ float4 __WMMA_8_16_16_half_float(half8 a, half4 b, float4 c){
  int *a_pk=(int*)(&a), *b_pk=(int*)(&b), *c_pk=(int*)(&c);
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    : "+r"(c_pk[0]),"+r"(c_pk[1]),"+r"(c_pk[2]),"+r"(c_pk[3])
    : "r"(a_pk[0]),"r"(a_pk[1]),"r"(a_pk[2]),"r"(a_pk[3]),"r"(b_pk[0]),"r"(b_pk[1]));
  return c;
}
extern "C" __global__ void __launch_bounds__(128) r_64_32_32_4_2_2_4_4_256_2(
    float* data0, half* data1, half* data2) {
  float buf0[64];                       // 64-float accumulator, 128 threads
  ...
  for (int Ridx0 = 0; Ridx0 < 256; Ridx0++) {          // <-- NO shared memory anywhere
    half val0  = (*(data2+(alu70+8)));                 // <-- scalar 16-bit load
    half val1  = (*(data2+(alu70+16)));
    ... 16 scalar half loads ...
    __WMMA_8_16_16_half_float(make_half8(val0..val7), make_half4(val8..val11), buf_val);
  }
  *(data0+alu0+0) = buf0[0];             // 128-bit stores
}
```

**CUTLASS's corresponding structure** (`cutlass_baseline_tgarch.cu`, CUTLASS
2.11, `OpClassTensorOp` + `Sm80`, 128×128×32 / 64×64×32 / 16×8×16, 4 stages).
Resolved from the kernel's own mangled name:

```
cutlass::gemm::kernel::Gemm<
  threadblock::MmaMultistage<GemmShape<128,128,32>,          // 4 global->smem stages
    PredicatedTileAccessIterator<...>                       // A operand
    RegularTileAccessIterator<RowMajorTensorOpMultiplicandCrosswise<16,32,16>, ...>
    RegularTileAccessIterator<ColumnMajorTensorOpMultiplicandCrosswise<16,32,16>, ...>
    warp::MmaTensorOp<GemmShape<64,64,32>,                  // 8 warps
      cutlass::arch::Mma<GemmShape<16,8,16>, float, half_t, half_t, float, OpMultiplyAdd>,
      cutlass::arch::OpMultiplyAdd>,
    epilogue::threadblock::Epilogue<OutputTileShape<128,8,2,1,1>, ...>
  >::GemmKernel
```

The difference in one sentence: **CUTLASS stages A and B through a 4-deep
shared-memory pipeline filled by `cp.async`/`LDGSTS` and read back with
`ldmatrix`; TinyGrad reads the tensor-core operands straight out of global
memory with scalar loads and never allocates shared memory at all.** Both
issue the same `mma.sync.m16n8k16`. That is the sm_80 rung in a nutshell: same
math instruction class, completely different data-movement discipline.

For the fp32 side, TinyGrad at sm_80:

```cuda
  for (int Ridx0 = 0; Ridx0 < 1024; Ridx0++) {
    float4 val0 = (*((float4*)((data1+(alu18+4096)))));
    float4 val1 = (*((float4*)((data1+(alu18+8192)))));
    float4 val2 = (*((float4*)((data1+(alu18+12288)))));
    float4 val3 = (*((float4*)((data1+alu18))));
    int alu19 = (alu0+(Ridx0<<14));
    float4 val4 = (*((float4*)((data2+(alu19+4096)))));
    ...
    *(buf0+0) = ((*(buf0+0))+(val3.x*val7.x)+(val3.y*val4.x)+(val3.z*val5.x)+(val3.w*val6.x));
```

A correct, well-vectorised, register-tiled SGEMM — with 1024 K iterations and
every one of the 8 loads a separate global access. This is the "old GPU" case
where the gap really is a constant-factor efficiency gap.

---

## 6. Verdict on the hypothesis

The hypothesis as stated:

> On older GPUs with few architectural features, compilers already generate
> near-optimal code […] On newer GPUs […] the gap should be much larger — and
> should change in KIND, from a constant-factor efficiency gap to a categorical
> gap where the compiler simply does not use the new instruction classes at all.
> Testable prediction: the gap grows SUPERLINEARLY with feature count, and there
> is a regime change when `cp.async` appears at sm_80.

### Supported

**The change in KIND happens, and it is real.** At sm_75/sm80/sm_89 the two
sides' SASS for fp16 both contain `HMMA` and nothing exotic: the vocabulary
matches and the disagreement is about discipline (shared-memory staging,
`cp.async`, `ldmatrix`, 128-bit vs scalar loads, 4 stages vs 0). At sm_90a and
sm_100a, CUTLASS's SASS contains `HGMMA`, `UTCHMMA`, `UTMALDG`, `ELECT`,
`CCTL`/cluster traffic and 2-CTA multicast TMA — and **TinyGrad's SASS contains
none of them, and in fact is byte-for-byte the same instruction-class set as its
own sm_75 output.** The categorical gap is exactly where predicted.

**The "old end is a constant-factor problem" claim is directly supported, and
more strongly than expected**: TinyGrad already emits `mma.sync` tensor-core
instructions on sm_75, a GPU whose feature count is low. The thing a human does
that the compiler does not do at sm_75 is *not* a new instruction class at all —
it is putting the operands in shared memory first.

**The compiler's pipeline depth is 0 at every rung** while the human's is
2 → 4 → 7 → 13. The growth is superlinear in *that* variable, at least.

### Refuted or needing correction

**"There is a regime change when `cp.async` appears at sm_80" — one regime change
is the wrong model. There are two, of different kinds.**

1. At **sm_80** the *data-movement* vocabulary diverges: CUTLASS gains
   `cp.async`/`LDGSTS` (36) and `ldmatrix` (24); TinyGrad gains nothing. The
   math instruction class is *unchanged* — both are `mma.sync.m16n8k16`, and
   TinyGrad's source is byte-identical across sm_80/89/90/100. This is
   the "different efficiency, same instructions" regime the hypothesis describes
   as the *old* regime, arriving at a GPU with an extra feature.
2. At **sm_90a** the *math* vocabulary diverges: `wgmma` (8) plus TMA appears,
   and `mma.sync` disappears from CUTLASS entirely. Only here does the gap become
   genuinely categorical for the arithmetic.

So the categorical change is at **sm_90a, not sm_80**. sm_80 is an intermediate
regime that the hypothesis's single-threshold model does not have a name for.
Note also that `cp.async` at sm_80 is the *cheapest* of the three classes to
reach for: it is a pure scheduling choice over an existing shared-memory
pipeline, and CUTLASS even uses it in its SIMT fp32 kernel. If the study is
looking for where "a human can do something a compiler will not", sm_80's
`cp.async` is the strongest example of a *small*, well-trodden, non-novel
optimisation being missed — which is closer to the hypothesis's "old GPU"
description than to its "new GPU" one.

**"The gap grows SUPERLINEARLY with feature count" — this experiment cannot
decide it, and the instruction matrix is the wrong instrument for it.** The
claim is about performance; we produced no performance data. The feature matrix
is a step function, not a superlinear one, and reading a superlinear trend off a
count of instruction classes would be exactly the kind of over-reading this study
warns against. What the matrix does support is the *qualitative* claim (a change
in kind), and the pipeline-depth row, which is a single variable measured at five
points and does grow faster than linearly (2, 4, 7, 13).

### Which parts are CANNOT EXPRESS / DOES NOT FIND / NOT YET IMPLEMENTED

This is the distinction the whole study turns on, so it is worth being exact.
The decisive evidence is that these strings appear **zero times anywhere in the
installed tinygrad package**:

```
grep -rE "wgmma|tcgen05|cp\.async|cp_async|clusterlaunch|tensormap|stmatrix" \
     .venv/lib/python3.12/site-packages/tinygrad/   →  0 hits
```

| Feature | Verdict | Why |
|---|---|---|
| `mma.sync` (HMMA/IMMA) | **CAN express, and DOES** | Hardcoded inline-PTX `asm("mma.sync.aligned.m{M}n{N}k{K}…")` template in `renderer/cstyle.py:461`, gated by the arch table in `codegen/opt/tc.py:98`. Emitted at every rung. |
| TF32 tensor core | **CAN express** | `cuda_8168_tf32` in the catalog; requires an explicit cast TinyGrad does not insert, so it is *present but not chosen*. |
| FP8 (unscaled) | **CAN express, not exercised** | `cuda_81632_f8`, sm_89+. No fp8 matmul was requested in this study. |
| Shared-memory staging | **CAN express, DOES NOT FIND** | The renderer emits `__shared__` and TinyGrad uses smem for reductions; the matmul schedule simply never introduces it. This is a search/heuristic failure, not a missing feature. |
| 128-bit vectorised fp16 loads | **CAN express, DOES NOT FIND** | `half4`/`half8` structs are in the generated file; the schedule still emits scalar `LDG.E.U16`. |
| `cp.async` / LDGSTS | **CANNOT EXPRESS** | Zero occurrences. There is no general inline-PTX escape hatch reachable from the scheduler — the *only* `asm()` in the renderer is the hardcoded `mma.sync` template. No schedule can produce it. |
| `ldmatrix`, `stmatrix` | **CANNOT EXPRESS** | Zero occurrences. |
| `wgmma` | **CANNOT EXPRESS** | Zero occurrences. See also the next row — this is deeper than a missing special case. |
| `tcgen05` (UMMA) | **CANNOT EXPRESS** | Zero occurrences. |
| TMA (`cp.async.bulk.tensor`, tensormap) | **CANNOT EXPRESS** | Zero occurrences. |
| `elect.sync`, thread-block clusters | **CANNOT EXPRESS** | Zero occurrences. |
| Block-scaled FP8/FP6/FP4 | **NOT YET IMPLEMENTED** | Absent from the CUDA tensor-core catalog entirely (present for AMD: `amd_cdna_1616128`). |
| Arch-conditional targets `sm_90a`, `sm_100a` | **CANNOT EXPRESS** | See below — TinyGrad cannot even *name* these targets. |

### The sharpest single finding

**TinyGrad cannot target an arch-conditional GPU at all.** It builds its target
arch as `arch = f"sm_{major}{minor}"` from `cuDeviceComputeCapability`
(`runtime/ops_cuda.py:122`), and its arch parsers do `int(arch[3:])`
(`codegen/opt/tc.py:98`, `renderer/cstyle.py:471`). Asking for `sm_90a` or
`sm_100a` raises:

```
ValueError: invalid literal for int() with base 10: '90a'
ValueError: invalid literal for int() with base 10: '100a'
```

So on a real H100 TinyGrad renders for `sm_90` and on a real B200 for `sm_100`.
`get_cuda(90)` and `get_cuda(100)` both fall through to `cuda_sm89` — the Ada
fragment catalog. TinyGrad is structurally blind to the distinction that Hopper
and Blackwell tensor cores are built on, and never sees the `a` that unlocks
wgmma and tcgen05. This is not a missed optimisation; it is a missing concept
in the target model, and it sits one level *below* the missing instruction
classes. (To answer the hardware question anyway, this study cross-compiles the
`sm_90`/`sm_100` sources with `ptxas -arch=sm_90a`/`sm_100a` — a valid
`ptxas` target, and the right question to ask of the hardware.)

And the reason those classes are unreachable is architectural, not a missing
special case: TinyGrad's entire tensor-core layer is a single `TensorCore`
dataclass (`codegen/opt/tc.py`) that emits `nvcuda::wmma` fragment
`load_matrix_sync`/`mma_sync`. **`wgmma` and `tcgen05` are not expressible in the
`wmma` API at all** — they are warpgroup- and CTA-async instructions requiring
tmem allocation, mbarrier-based pipelines and a CuTe-style `TiledMMA`
abstraction. So for these rows the honest label is *both* "not yet implemented"
*and* "the current abstraction cannot express it" — and the second is the more
important one, because it says the work is not a missing special case in
`tc.py` but a new instruction-generation layer. Note the contrast: `cp.async`
and `ldmatrix` are ordinary PTX that any inline-asm template could reach, so for
those the failure is genuinely "TinyGrad's codegen vocabulary is too narrow",
without the extra "and the abstraction is wrong" caveat.

---

## 7. Limitations

**No timings. None.** No GPU was available, so task 5 (the sm_75 / sm_89
CUTLASS-generated-vs-handwritten wall-clock ratio) was **not performed**. No
performance number appears anywhere above, and none should be inferred: the
instruction counts measure *code shape*, not executed frequency or achieved
occupancy. A single statically-unrolled `HMMA` and a trip-count-dependent one
both count as 1. Anything in this document that looks like a speed claim is not
one.

**nvcc was unavailable; clang-20 is the frontend.** The instruction-class results
are on NVIDIA's own ptxas and are solid. The residual risk is narrow: clang and
nvcc could in principle differ in *whether* they emit `cp.async` for a given
source. That cannot affect this comparison, because neither side's source asks
for it — the absence is a source-level fact, not a codegen-level one.

**The sm_100a CUTLASS cell required overriding a CUTLASS safety gate**
(`-DCUTLASS_ARCH_MMA_SM100A_ENABLED=1` against `#if !CUTLASS_CLANG_CUDA`). The
override is technically justified (clang-20 does target sm_100a and does define
`__CUDA_ARCH_FEAT_SM100_ALL`) but NVIDIA does not sanction it, and I could not
cross-check the result against an nvcc build. This is the **least trustworthy
cell in the matrix** and the one to re-run first if nvcc becomes available.

**CUTLASS configurations are published defaults, not autotuned.** No GPU, no
profiler. For sm_75/80/89 I copied tile shapes verbatim from CUTLASS's own
example and test files so the instantiations are ones CUTLASS supports, but the
**stage counts are not the output of CUTLASS's autotuner on this problem size**.
The 2.x fp16 baseline spilling 472 B is direct evidence these are not tuned
configs. This is fine for an instruction-class comparison and **not** fine for a
peak-TFLOP/s claim.

**CUTLASS version split.** sm_75/80/89 baselines come from **CUTLASS 2.11.0**
(`cutlass2x/`), sm_90a/sm_100a from **CUTLASS 4.8.0** (`cutlass/`). Not by
choice: 4.8 has no 2.x `OpClassSimt` specialisation for Sm75 at all, its Sm80
Ampere path does not compile under clang-20, and the 2.x code has no Hopper or
Blackwell path. Both trees are the shipping implementation for their rungs, but
the sm_80 vs sm_90a rows are not from one codebase.

**Beam search was not exercised.** `BEAM=1` produced byte-identical source to
the default schedule. TinyGrad's beam needs to *time* candidates on a device
(`engine/realize.py`: "beam needs device access to time candidates"), which a
capture-only device cannot provide. So I cannot report the size of TinyGrad's
search space or whether it contains a shared-memory-tiled variant. The
"CANNOT EXPRESS" verdicts above are unaffected — no schedule can emit a string
that appears nowhere in the package — but the "DOES NOT FIND" verdicts for shared
memory and fp16 vectorisation rest on reading the generated code, not on
exhausting the search space.

**Narrow shape coverage.** 4096³ and 128³ only, four dtype combinations. TinyGrad
picks different tiles for the two sizes (`r_128_64_8_16_4_4_1024_4` vs
`r_4_2_8_16_4_4_32_4`) but the *feature usage is identical*, so the shape
sweep adds no information to the matrix. No batched, strided, split-K or
non-square shapes.

**Dynamic shared memory is not captured.** `ptxas -v` and
`cuobjdump -res-usage` report `SHARED:0` for the CUTLASS 2.x/3.x rows because
those kernels take `extern __shared__` and request it at launch. The CUTLASS
smem figures in the table are therefore **understated**, most severely for the
rows where staging matters most (4-stage Ampere, 7-stage Hopper, 13-stage
Blackwell). This does not affect any feature-usage conclusion, since `LDGSTS` /
TMA / `ldmatrix` presence is the substantive claim and those are counted from
SASS.

**Two CUTLASS source patches** were required (see §2). Both are in non-GEMM code
and both are diffed, but they are modifications to the baseline library and are
recorded here rather than buried.

**Not attempted:** FP8, FP6/FP4, block-scaled, int8, TF32, grouped/batched GEMM,
attention fusion. TinyGrad's `codegen/opt/tc.py` shows fp8 shapes and TF32 shapes
exist in its catalog; a follow-up that generates an fp8 matmul would be the
cheapest way to test the "CAN express but does not FIND" claim in a regime where
both sides have the instruction.

---

## 8. Reproducing

> **Note for the coordinator:** the sibling `cutlass_lab/` lane did not exist
> when this work started (checked at kickoff), so I cloned CUTLASS myself into
> `matmul/cutlass/` (v4.8.0, sparse, 79 MB) and `matmul/cutlass2x/` (v2.11.0,
> 11 MB). `cutlass_lab/` appeared mid-run, so the two lanes overlap and have not
> been reconciled. One concrete difference worth knowing: `cutlass_lab/logs/
> build_sm800_tensorop.log` ends in **56 compile errors** — its
> `harness/sgemm.cu` sm_80 tensor-op instantiation (128×256×64) fails with
> `<error-type>` in the epilogue `Array<>`. The sm_80 fp16 baseline here
> (128×128×32, warp 64×64×32, instr 16×8×16, 4 stages, CUTLASS 2.11) **does**
> compile and is the usable baseline for that cell. Both lanes agree that
> nothing has been executed, so neither has a numerical verification result.
> Treat `cutlass_lab/` as authoritative for its own lane and this directory as
> authoritative for the comparison; the duplication is a reconciliation TODO,
> not something I resolved.

```bash
cd matmul

# 1. capture TinyGrad's generated CUDA C  (host venv, no GPU needed)
../.venv/bin/python gen_src.py                 # -> gen/*.cu, gen/INDEX.json
python3 -c "import json;print(len(json.load(open('gen/INDEX.json'))))"

# 2. apply the two documented CUTLASS patches
./patch_cutlass2x.sh
./patch_cutlass.sh

# 3. build the toolchain image and run the whole matrix
docker build --platform linux/amd64 -f Dockerfile.toolchain -t mc-cuda-nvcc:3 .
docker run --rm --platform linux/amd64 -v "$PWD:/work" mc-cuda-nvcc:3 \
    bash -c 'cd /work && ./build_matrix.sh'   # -> analysis/matrix.jsonl

# 4. render the tables
../.venv/bin/python render.py
```

| file | role |
|---|---|
| `tg_backend.py` | capture-only CUDA device; extracts TinyGrad's source with no GPU |
| `gen_src.py` | generates `gen/*.cu` for 5 targets × 2 shapes × 4 dtypes |
| `cutlass_baseline_tgarch.cu` | sm_75/80/89 baselines (CUTLASS 2.11), `-DSELECT=0` fp32 SIMT / `-DSELECT=1` fp16 tensor-op |
| `cutlass_baseline_sm90a.cu` | Hopper baseline, mirrors `examples/49_hopper_gemm_with_collective_builder` |
| `cutlass_baseline_sm100a.cu` | Blackwell baseline, mirrors `examples/71_blackwell_gemm_with_collective_builder` |
| `compile_inspect.sh`, `compile_cutlass.sh` | `.cu` → PTX → cubin → SASS, with `ptxas -v` |
| `build_matrix.sh` | runs all (kernel, arch) pairs, emits `analysis/matrix.jsonl` |
| `render.py` | renders the two tables in §3 and §4 |
| `patch_cutlass2x.sh`, `patch_cutlass.sh` | the two documented CUTLASS fixes |
| `Dockerfile.toolchain`, `fetch_conda_pkg.sh`, `cuda_stubs/` | compile-only CUDA toolchain |

### CUTLASS baseline provenance, in one place

| target | source of truth | file | what I chose |
|---|---|---|---|
| sm_75 fp32 | `test/unit/gemm/device/simt_sgemm_tn_sm80.cu` (2.11) | `cutlass_baseline_tgarch.cu` `-DSELECT=0` | tile 128×128×8, warp 32×64×8, 2 stages (generic-ArchTag SIMT specialisation fixes stages) |
| sm_75 fp16 | `examples/12_gemm_bias_relu` (Sm75, half) | `…cu -DSELECT=1` | tile 128×128×32, warp 64×64×32, instr 16×8×8, 2 stages |
| sm_80/89 fp32 | `test/unit/gemm/device/simt_sgemm_tn_sm80.cu` (2.11) | `…cu -DSELECT=0` | same tile, 3 stages (Sm80-specific SIMT specialisation) |
| sm_80/89 fp16 | `examples/47_ampere_gemm_universal_streamk` (ArchTag Sm80, 4 stages) | `…cu -DSELECT=1` | tile 128×128×32, warp 64×64×32, instr 16×8×16, 4 stages |
| sm_90a fp16 | `examples/49_hopper_gemm_with_collective_builder` (4.8) | `cutlass_baseline_sm90a.cu` | `KernelScheduleAuto` + `EpilogueTileAuto` + `StageCountAutoCarveout`, tile 128×128×64, cluster 2×1×1, fp16→fp16 |
| sm_100a fp16 | `examples/71_blackwell_gemm_with_collective_builder` (4.8) | `cutlass_baseline_sm100a.cu` | `KernelScheduleAuto`, tile 128×128×64 (1-SM), cluster 2×2×1, fp16 |

For the three modern targets the schedule and stage count were left to CUTLASS's
own auto-selection rather than guessed, which is the right call for a "human
state of the art" baseline: CUTLASS's auto-scheduling is part of the artefact.
None of these was numerically verified against numpy — the kernels are never
executed, so their correctness is CUTLASS's and was not re-checked. Their
*instruction mix* is what this study measures, and that is compiler-determined
and independent of runtime data.
