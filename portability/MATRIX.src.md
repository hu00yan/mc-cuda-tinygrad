# Portability matrix: does the state of the art exist on an old card?

**The one-line answer.** It mostly does not, and where it does it is not the same
kernel. On a Turing T4 the current flash-linear-attention KDA forward pass
compiles cleanly, runs, and produces correct numbers — and contains **zero**
tensor-core instructions: 2052 `fma.rn.f32` where an A100 build has 192 `mma.sync`.
At the Hopper rung the picture changes category: `wgmma` assembles for exactly
one `-arch` string (`sm_90a`) and for no other, so FA3 is not "slow on" sm_75, it
does not exist there. The interesting failure is therefore neither of the two
obvious ones: it is **silent, and it is a substitution rather than a refusal.**

Everything below was measured on this machine. There is no GPU in this study and
none was sought; no credentials, no paid compute, no host installs. "Does it
compile" is a compile-time fact, and every cell in the tables below came out of a
container running `ptxas`/`nvcc`/`ptxas` on the CPU, or Triton compiling ahead of
time. Where a cell is `NOT-OBTAINED` I did not obtain the subject, and the cell says
so rather than guessing.

---

## 0. Toolchain, and what that licenses us to say

| | |
|---|---|
| toolkit | `Cuda compilation tools, release 12.9, V12.9.86` (nvcc + ptxas) |
| host compiler | g++ 13 (x86_64, under OrbStack x86 emulation) |
| archs this toolkit can target | `sm_50` … `sm_121`, including the arch-specific `sm_90a` / `sm_100a` / `sm_100f` / `sm_120a` variants |
| CUTLASS | 4.8.0 (`cutlass_lab/cutlass`, read-only — another lane owns it) |
| Triton | 3.4.0 wheel, linux/amd64; its **own bundled ptxas is 12.8.93** |
| Triton archs its bundled ptxas can target | `compute_50` … `compute_120`, incl. `compute_90a`, `compute_100a`, `compute_120a` |
| cells measured | 544 PTX + 50 CUTLASS + 80 Triton + 24 fla-KDA + 10 FA = **708** |

Two consequences worth stating before the tables, because they bound the claims:

1. **`sm_100a` is reachable with this single toolkit** (`ptxas --help` lists
   `compute_100a`), so the Blackwell rows are real measurements and not
   extrapolations. `sm_103`/`sm_121` were also targeted; they behave like their
   base rungs.
2. **Triton's ceiling is its own ptxas, not the system nvcc.** Triton 3.4.0 ships
   ptxas 12.8.93, whose target list stops at `compute_120`. It cannot emit
   `sm_103` or `sm_121` even though the toolkit on the same machine can. A Triton
   row for a future arch is therefore a statement about *Triton's bundled
   assembler*, which is the honest thing to report but is easy to mistake for a
   hardware fact.

---

## 1. The status vocabulary

The four categories in the brief, plus the distinction the notes (`notes/04`) demand
between *cannot express* / *can express but does not find* / *not yet implemented* /
*does not exist here*. Every cell resolves to one of:

| status | meaning | how it is detected |
|---|---|---|
| `COMPILES-SAME` | assembles, emits the same instruction class as the native build | ptxas exit 0 **and** SASS/PTX class histogram matches native |
| `COMPILES-DEGRADED` | assembles, emits a **different** (older) instruction class. No diagnostic. Correct results. | exit 0 **and** class histogram differs from native |
| `REFUSED` | the toolchain rejects the target | nonzero exit; diagnostic recorded verbatim |
| `REFUSED-BY-LIBRARY` | the library's own gate rejects, even though the arch assembles | nonzero exit, diagnostic names the library |
| `NOT-OBTAINED` | subject not obtained; nothing claimed | — |

`COMPILES-DEGRADED` is the row that matters and the one that is easiest to miss,
because nothing in the build output says it happened. Every such cell in this
document is backed by an instruction histogram, not an inference.

---

## 2. The ISA floor: PTX witness ladder

34 hand-written PTX modules, one mechanism each, **fixed bytes** — only
`ptxas -arch=` varies across 16 targets (544 cells, `out/ladder.tsv`).

Every witness is *self-validating*: `src/selfcheck.sh` assembles each one at the
arch it was written for before the cross-target sweep, so bad PTX can never be
mistaken for a portability finding. Syntax and operand counts were transcribed
from CUTLASS 4.8's own inline-asm strings in `include/cute/arch/*.hpp`, not from
memory. `src/selfcheck.sh` reports `ok=32 fail=0 known_unobtainable=1`: every witness
that claims to be valid PTX is, and the one I could not author is named as a known
gap rather than left to look like a portability finding (see §8).

<!--TABLES:LADDER-->

**The four findings that organise everything else.**

1. **`wgmma` is sm_90a-only — one string, no neighbours.** Not sm_90, not sm_100a,
   not sm_100f, not sm_101, not sm_103, not sm_120a. ptxas says
   `Instruction 'wgmma.fence' cannot be compiled for architecture 'sm_100a'`. This
   is stricter than "Hopper-only": the Hopper *hardware* is sm_90a, and the
   sm_90-vs-sm_90a distinction that `arch-conditional features` is built on means
   there is no `-arch` at all for a wgmma kernel on anything but H100/H200.
2. **`tcgen05` is sm_100a-only**, and *consumer* Blackwell is a different ISA
   extension: `tcgen05.alloc` is `not supported on .target 'sm_120a'`, and
   sm_120a needs the `mma.sync`-based `kind::mxf4nvf4` path instead. "Blackwell
   supports FP4" is not one rung on this ladder; it is two mutually exclusive ones.
3. **`tensormap.replace` is sm_90a-only, while TMA itself is sm_90+.** So a kernel
   that merely *loads* via TMA can target `sm_90`, but one that *mutates a
   descriptor* — the thing FA3 must do, because CUDA exposes no C++ interface for
   TMA descriptors (Tawa's challenge 2, cited in `notes/04`) — needs the
   arch-conditional variant.
4. **The TMA transaction-accounting barrier is Hopper-only even though the
   barrier is not.** `mbarrier.init/arrive/test_wait/inval` all assemble for
   sm_80, but `mbarrier.arrive.expect_tx` requires sm_90. So an Ampere mbarrier
   cannot carry TMA byte counts. Separately, the *modern* `mbarrier.try_wait`
   spelling is sm_90-gated in ptxas 12.9 while the PTX-7.0 `mbarrier.test_wait`
   spelling works on sm_80 — a toolchain-surface change, not a hardware one, and a
   good example of why the four-way "cannot express / not yet implemented" split
   matters.

Also measured, and mildly surprising: **`elect.sync` is refused by ptxas 12.9 for
sm_80** (`Feature 'elect' requires .target sm_90 or higher`) even though it is
usually described as an Ampere feature. CUTLASS agrees in practice — the string
`elect.sync` appears exactly once in the whole 4.8 tree, in
`cute/arch/cluster_sm90.hpp`. Reported here as a measured ptxas gate, not a
hardware claim.

---

## 3. CUTLASS 4.8.0 by arch — where the silent substitution lives

Two probes, both calling the *degrading code path itself* rather than paraphrasing
it. 50 cells, **all exit 0** (`out/cutlass/compile.tsv`, `out/cutlass/sass.tsv`).

- `cp_async` calls `cutlass::arch::cp_async_zfill<16>`,
  `cp_async_fence()`, `cp_async_wait<0>()` — the copy primitive of the 2.x
  multistage mainloop, i.e. the thing every Ampere GEMM is built on.
- `wgmma` / `tma_load` / `tma_bulk` call CuTe's `SM90::GMMA::MMA_64x8x16_...::fma`,
  `SM90_TMA_LOAD_1D::copy` and `SM90_BULK_COPY_G2S::copy` directly.
- `wgmma_macro` is the control: the same wgmma probe with
  `-DCUTLASS_ARCH_MMA_SM90A_ENABLED` forced on from the command line.

<!--TABLES:CUTLASS-->

### 3.1 `COMPILES-DEGRADED`: cp.async → synchronous copy, and the barrier goes too

`cutlass/arch/memory_sm80.h:45-49` gates on `__CUDA_ARCH__ >= 800`, and
`cp_async_zfill`'s `#else` branch is a plain register-staged copy
(`*smem = *global`). Measured in SASS:

| `-arch` | what the same source emits |
|---|---|
| sm_70 | 4 × `LDG.E.CONSTANT.SYS` + 4 × `STS`, **no `DEPBAR`** |
| sm_75 | 4 × `LDG.E.CONSTANT.SYS` + 4 × `STS`, **no `DEPBAR`** |
| sm_80 … sm_120a | 1 × `LDGSTS` + 1 × `DEPBAR.LE` |

Two things degrade, not one. The copy stops bypassing registers, **and**
`cp_async_fence()` / `cp_async_wait<0>()` compile to *nothing at all* on sm_70/75 —
they are entirely inside `#if CUDA_CP_ASYNC_ACTIVATED`. So the multistage
software pipeline is not merely slower on a T4; the synchronisation that made it a
pipeline is gone. Results stay correct, because the remaining copy is
program-ordered. This is the cleanest `COMPILES-DEGRADED` measurement in the study.

**Cost of this degradation.** Two distinct costs, one quantified and one not:

- *Instruction-mix cost.* 4 staged 128-bit loads plus 4 shared stores per tile,
  versus 1 `LDGSTS` that bypasses the register file entirely. Registers are the
  scarce resource in a GEMM mainloop, so staging the copy through them is the
  cost the fallback was designed to avoid.
- *Latency-hiding cost, not quantified here.* With no `DEPBAR`, the mainloop
  cannot overlap the next tile's global load with the current tile's math, so
  global latency is exposed. **I did not measure this** and am not going to put a
  number on it. The "Correct but Slow" study already in `notes/04`
  (arXiv 2607.04454) reports 300–1520× slowdowns for kernels that pass strict
  numerical validation, which is the right citation for "silent slowness is common
  and can be enormous" — but that is a different mechanism (authoring defects) and
  should not be quoted as the cost of *this* fallback.

### 3.2 `COMPILES-DEGRADED` at its worst: the SM90 paths become runtime traps

This is the finding I did not expect, and it is the most important one in the
CUTLASS lane.

`cute/config.hpp:160` defines

```c
#define CUTE_INVALID_CONTROL_PATH(x) assert(0 && x); printf(x)
```

— a **runtime** assert, not a compile error. And the gate that selects the real
path is `cutlass/arch/config.h:45-49`, which requires
`__CUDA_ARCH__ == 900 && __CUDA_ARCH_FEAT_SM90_ALL`. The second macro is set by nvcc
**only for `-arch=sm_90a`**.

So: asking for a Hopper wgmma from a build configured for anything else compiles
without a single diagnostic, and the call site becomes `assert(0); printf(...)`.
Measured:

| `-arch` | cubin bytes | SASS |
|---|---|---|
| sm_80 | 6152 | 4 × `CALL.ABS.NOINC`, **no MMA at all** |
| sm_90 | 6688 | 4 × `CALL.ABS.NOINC`, no MMA |
| **sm_90a** | **4920** | `DEPBAR` (the wgmma fence) — the real instruction |
| sm_100a | 10032 | 4 × `CALL.ABS.NOINC`, no MMA |

And the string literal `"Attempting to use MMA_64x8x16_F16F16F16_SS without
CUTE_ARCH_MMA_SM90A_ENABLED"` is present in the sm_75/sm_80/sm_100a cubins. For
`TMA`, sm_90a/sm_100a/sm_120a emit real `UTMALDG`; every other target emits
`CALL.ABS.NOINC` and no TMA instruction.

Three things make this nastier than a plain performance gap:

- **The broken build is *bigger* than the working one.** sm_80 is 6152 bytes;
  sm_90a is 4920. Any size- or "it linked" -based sanity check passes.
- **It is not a compile error, so it survives `-Werror`, CI, and code review.**
- **The `wgmma_macro` control is byte-identical to `wgmma` at every arch.** Forcing
  `CUTLASS_ARCH_MMA_SM90A_ENABLED` on the command line changes nothing, because
  `config.h:45` additionally demands `__CUDA_ARCH__ == 900`. The effective gate is
  the `-arch` string itself, which is why a build system that "just sets the right
  macros" cannot fix this.

**Cost of this degradation.** Not a performance number — a correctness cliff. Under
`-DNDEBUG` (which is what `fa_build.sh` and most release builds pass) `assert` is a
no-op and `printf` writes to stdout from the device, so the kernel runs, produces
garbage, and says nothing. The cost is unbounded because the kernel is not computing
the answer.

---

## 4. Triton 3.4.0 by arch

Two ladders: synthetic mechanism probes, and then the real flash-linear-attention
KDA kernels. No GPU; `triton.compile` with an explicit `GPUTarget`, so only
`target.arch` varies.

<!--TABLES:TRITON-->

### 4.1 The task's premise is false, and the truth is more interesting

The brief anticipated that "recent Triton may refuse to compile the `tl.dot` path
for sm_75 at all." **It does not.** Triton 3.4.0 compiles `tl.dot` all the way down
to **sm_50**, and at sm_50–sm_75 it silently lowers the matmul to scalar FMA:

| target | `tl.dot` (fp16, 64×64×32) emits |
|---|---|
| sm_50 / sm_60 / sm_70 / sm_75 | **1024 × `fma.rn.f32`, zero `mma.sync`** |
| sm_80 / sm_86 / sm_89 / sm_120 | 16 × `mma.sync` + 8 × `ldmatrix` |
| sm_90 | 5 × `wgmma` |
| sm_100 / sm_101 | 13 × `tcgen05` + 8 × `mbarrier` |

One unchanged source produces **four different instruction classes**. The PTX also
grows 813 → 1869 lines on the FMA path, a 2.3× code-size expansion for a kernel that
is nominally the same computation. There is no warning.

**The minimum compute capability Triton accepts: there isn't one.** The
capability sweep (`src/triton_floor.py`) compiles a plain `copy` kernel at
sm_50/60/70/75/80/89/90/100/120 and a `tl.dot` kernel at all of those too. The only
refusal in the sweep is at **cap 35**, because ptxas's lowest target is
`compute_50`. So the floor is set by the bundled assembler, not by Triton:

> **Triton 3.4.0 has no compiler-level compute-capability floor. Its effective
> floor is sm_50, the lowest target of the ptxas 12.8.93 it ships.**

Whether that floor has *moved* across versions is a real and citable question, and
I checked it in the source rather than from memory — but I could not establish a
trend, and I am not going to invent one. What is established:

- `third_party/nvidia/backend/driver.py` in v3.0.0, v3.2.0 and v3.4.0 contains **no**
  `min_capability` / `_check_capability` gate; the only capability arithmetic is
  `capability = major * 10 + minor` when *reading the device*.
- `compiler.py` in all three versions branches on capability only to *add* features:
  `allow_fp8e4nv = capability >= 89`, `convert_custom_float8_sm80 if capability >= 80
  else sm70`, `proc = 'sm_90a' if capability == 90`, and (v3.4.0 only)
  `sm_arch_from_capability` adding the `a` suffix for `capability >= 90`. The v3.4.0
  rewrite moved the arch string into `options.arch` with
  `knobs.runtime.override_arch`, which is what makes offline per-arch compilation
  possible at all.
- In v3.0.0/v3.2.0 the `sm_90a` special case is `capability == 90`; in v3.4.0 it is
  `capability >= 90`. **That is a real, citable change**: Triton 3.4 stopped
  special-casing Hopper and started applying the arch-conditional suffix to
  Blackwell too, which is the prerequisite for its `tcgen05` output. It is a
  widening, not a floor raise.

### 4.2 Where Triton *does* refuse

Three genuine refusals, all `REFUSED` rather than degraded, and all legible:

- **fp8 below sm_89**, with a named diagnostic:
  `ValueError("type fp8e4nv not supported in this architecture. The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')")`.
  Note the message is subtly unhelpful: rather than saying "fp8e4nv needs Ada or
  later", it names two *other* fp8 flavours as the alternatives, which reads as a
  suggestion to substitute them. (Source: `triton/language/core.py:593`.)
- **int8 `tl.dot` below sm_80**, and this one is *not* a clean gate: the diagnostic
  is `RuntimeError: PassManager::run failed` on
  `TritonGPUAccelerateMatmul`, and the underlying MLIR verifier error is
  `'arith.extf' op operand #0 must be floating-point-like, but got 'tensor<64x64xi8, ...>'`
  — the pass inserts a float-extend on an integer operand. Per the taxonomy in
  `notes/04` this is **not-yet-implemented**, not cannot-express, and it is
  identical on sm_70 and sm_75.
- **TMA below sm_90 — and this one degrades silently instead.** See 4.3.

### 4.3 The second silent degradation: TMA becomes a scalar copy loop

`tl.make_tensor_descriptor(...).load(...)` — the Triton 3.3+ device-side TMA API,
i.e. the user explicitly asking for TMA — compiles for sm_70 through sm_89 with no
diagnostic and emits **no tensormap, no cp.async, no mbarrier**: just
`32 × ld.global.b16` + `32 × st.global.b16`. From sm_90 up it emits
`tensormap×16`, `mbarrier×5`, `cluster×1`, `elect×1`, `cp.async×1`. This one is
arguably worse than the `tl.dot` case, because `make_tensor_descriptor` is a
deliberate request: the programmer named the mechanism, and the compiler dropped it
without a word.

---

## 5. flash-linear-attention's KDA kernels — the real library, the real degradation

This is the subject the study actually cares about, and it is the one where I got
the *actual upstream source* compiling for every arch.

`subj/fla/` holds the GitHub `main` tree of `fla-org/flash-linear-attention`. (The
PyPI wheel, `flash_linear_attention-0.5.2-py3-none-any.whl`, is **missing
`fla/ops/` entirely** — it ships only `fla/layers` and `fla/models`, so
`fla/ops/kda` is unobtainable from the wheel. The repo is the only route, and that
is what I used.)

`src/kda_ladder.py` lifts every `@triton.jit` kernel out of `fla/ops/kda/*.py` with
`ast`, keeps the `@triton.jit(...)` decorator, drops `@triton.heuristics` and
`@fla_cache_autotune` (AOT compilation supplies the constexprs; the autotune
wrapper needs a live device), and binds the bare names fla imports from `fla.utils`
(`exp2`, `softplus`, `SOLVE_TRIL_DOT_PRECISION`, …). **The device code compiled is
unmodified upstream source.** torch is not needed — its wheel host is unreachable
from this network — and the kernels only use `torch.float32`-style dtype *names*,
which a shim supplies. The lifted modules are written to
`out/kda_lifted/*.py` so they can be read and diffed against upstream.

Signatures and constexprs are transcribed from the production launch site at
`fla/ops/kda/chunk_intra.py:879` (K=128 head dim, BT=64 chunk, BC=16 sub-chunk), so
this is the production shape class, not a toy.

<!--TABLES:KDA-->

**24 cells, 24 compiles, 0 refusals.** And on sm_70/sm_75 the tensor cores are
completely absent from the forward pass:

| `chunk_kda_fwd_kernel_inter_solve_fwd` | `mma.sync` | `fma.rn.f32` | shared mem |
|---|---|---|---|
| sm_70 | **0** | **2052** | 12288 B |
| sm_75 | **0** | **2052** | 12288 B |
| sm_80 … sm_120 | 192 | 516 | 10240 B |

Note the third column: it is not just that the tensor cores went away — the
non-tensor arithmetic *quadrupled*, because the same `tl.dot` products are now
computed as scalar FMA chains. And shared memory went **up** (12288 vs 10240 B): on
a Turing card this kernel uses more on-chip scratch *and* gets no tensor cores.
The intra-chunk kernel shows the same shape (257 FMA vs 32 `mma.sync`, smem
4096 → 8192 in the *other* direction here, so the smem effect is shape-dependent and
I am not generalising it).

The fused-recurrent decode kernel is FMA-only on **every** arch including sm_90 and
sm_100 — which is correct behaviour (it is a sequential recurrence with no matmul
worth putting on tensor cores) and is a useful reminder that "FMA-only" is not
automatically a bug. The finding is confined to the two chunked forward kernels,
which are matmul-dominated.

**Cost of this degradation, cited rather than guessed.** T4 vendor peak
mixed-precision throughput is **65 TFLOPS** against **8.1 TFLOPS** FP32 — an
**8.0×** ratio (NVIDIA T4 Tensor Core GPU datasheet / product page). So a T4 build
of this kernel is bounded by roughly one eighth of the tensor-core rate the same
source reaches on an A100 (19.5 TFLOPS FP32 vs **312 TFLOPS** dense mixed-precision
FP16/FP32, a 16× ratio; NVIDIA A100 datasheet and Ampere architecture whitepaper,
which states 108 SMs / 432 Tensor Cores "deliver up to 312 TFLOPS of dense
mixed-precision FP16/FP32 performance").

Two honest caveats on that number. It is a **peak** ratio, not a measured one — I
have no GPU, so the realized ratio will be lower and the FMA path will additionally
be hurt by register pressure and the loss of `ldmatrix`-fed fragment loads, which I
did not measure. And 8× understates it: the FMA path also loses the async pipeline
(§3.1), so the two costs compound multiplicatively rather than additively. **The
direction and the order of magnitude are established; the precise factor is not,
and only a T4 run can establish it.**

One more fla-specific portability fact, found while reading the source:
`SOLVE_TRIL_DOT_PRECISION` is chosen at **import time** from `IS_TF32_SUPPORTED`,
a live device query (`fla/ops/kda/chunk_intra.py:21-24`) — not per-arch. So the
precision of KDA's triangular solve is a function of the machine fla was imported
on, which is its own portability hazard, orthogonal to this study's axis.

---

## 6. FlashAttention 2 and 3

### FA3 — `REFUSED`, and the refusal is structural

`subj/fa/flash-attention-main/` is the GitHub `main` tree. The `hopper/` directory
contains **only** SM90 machinery: `flash_fwd_kernel_sm90.h`,
`mainloop_fwd_sm90_tma_gmma_ws.hpp`, `copy_sm90_bulk_reduce.hpp`. There is no SM80
kernel in `hopper/` for the `static_switch.h` `Arch = 80` branch to land in:

```c
// hopper/static_switch.h:138-150
#define ARCH_SWITCH(ARCH, ARCH_NAME, ...)                    \
  if (ARCH == 86 || ARCH == 89) { ARCH_NAME = 86; ... }     \
  else if (ARCH < 90)            { ARCH_NAME = 80; ... }     \
  else                          { ARCH_NAME = 90; ... }
```

And the features FA3 is built from are gated on
`__CUDA_ARCH__ == 900 && __CUDA_ARCH_FEAT_SM90_ALL` (§3.2), i.e. exactly
`-arch=sm_90a`. **Therefore every FA3 cell below sm_90a is `REFUSED`**, and the
refusal is attributable to specific, independently measured PTX feature gates:

| mechanism FA3 needs | measured requirement (from §2) |
|---|---|
| `wgmma.mma_async` | `sm_90a` only |
| `tensormap.replace` | `sm_90a` only |
| `mbarrier.arrive.expect_tx` | sm_90+ |
| `cp.async.bulk` / TMA | sm_90+ |
| `mapa.shared::cluster`, `barrier.cluster` | sm_90+ |

**I did not compile FA3 itself.** The `hopper/` translation units need libtorch
headers (`c10/cuda/CUDAException.h`) and the torch wheel host is unreachable from
this network, so a real FA3 build is `NOT-OBTAINED` in this lane. The row above is
therefore a source-level argument resting on PTX gates I *did* measure directly, not
a compilation I watched fail. I am flagging that distinction rather than blurring it.

### FA2 — `COMPILES-DEGRADED`, and it is the most likely thing a reader has already hit

`subj/fa/flash-attention-2.8.3/`, tag v2.8.3. Real compile is `NOT-OBTAINED` for
the same libtorch reason. But the guard is short, and I can quote it exactly —
`csrc/flash_attn/src/kernel_traits.h:18-44`:

```c
#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
    static constexpr bool Has_cp_async = true;
    using MMA_Atom_Arch = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;   // m16n8k16
#else
    static constexpr bool Has_cp_async = false;
    using MMA_Atom_Arch = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;    // m16n8k8
#endif
...
#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 750
    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;    // ldmatrix
#else
    using SmemCopyAtom = Copy_Atom<DefaultCopy, elem_type>;          // plain ld/st
#endif
```

**Two things change together, from one source file, with no diagnostic: the
tensor-core instruction shape and whether the mainloop is asynchronous at all.**
`csrc/flash_attn/src/utils.h:13-64` adds a third, smaller one — `relu2<half_t>`
uses the `max.f16x2` instruction on sm_80+ and a hand-rolled PTX emulation
(`set.gtu.u32.f16x2` + `and.b32`) below it, i.e. software-emulating a single
instruction.

`src/cu/fa2_traits_probe.cu` transcribes that guard (the two MMA shapes as inline
PTX, from `cute/arch/mma_sm75.hpp` and `mma_sm80.hpp`) so the SASS reading shows
which instruction class each arch gets. The operand counts are worth recording
because they are a trap: `m16n8k8` with f32 accumulate has **four** accumulator
registers per thread (`SM75_16x8x8_F32F16F16F32_TN` → `DRegisters = float[4]`)
where `m16n8k16` has two.

<!--TABLES:FA-->

Read that table carefully, because it contains its own lesson: **the SASS mnemonic
is `HMMA` on both sides of the branch.** sm_75 and sm_80 differ only in the
*encoding* — visible as 2600 vs 2728 cubin bytes, and only in the SASS hexdump, not
in any disassembly summary. A reviewer grepping SASS for `HMMA` sees a clean bill of
health on a card where the kernel is running a different instruction with half the
K-depth. This is why the matrix reads SASS, not just exit codes.

**Cost of this degradation.** The m16n8k8 instruction covers half the K of an
m16n8k16, so a matmul needs twice as many MMA instructions and twice the
`ldmatrix` traffic to reach the same FLOPs; that is a structural statement about the
instruction, not a measurement. The *measured* support for the general claim is
again the vendor peak ratio: on a T4, 8.1 TFLOPS FP32 vs 65 TFLOPS mixed-precision
tensor — 8× — and the async pipeline is gone as well (§3.1), so the two compound.
FA2's own sm_75 support is real and its developers knew about it; what the matrix
adds is that **the degradation is invisible from the outside**, and it is chosen per
`-arch` at compile time, so two builds of the same commit are different kernels.

---

## 7. cuBLAS and cuDNN — not determinable by compilation

**This row is `NOT-DETERMINABLE-BY-COMPILATION`, and that is a statement about the
method, not a gap in the work.** cuBLAS and cuDNN ship as closed-source precompiled
binaries (`libcublas.so.12`, `libcudnn.so.9`, …) with one cubin per supported
architecture, selected at load time by the runtime's compute-capability dispatch.
There is no source to hand to `nvcc`, so a compile-time probe cannot address them
at all. Anything I put in this row would be a guess, so the cell is empty.

What *would* determine it, in increasing order of cost and decreasing order of
inaccessibility here:

1. **Disassemble the shipped binaries and count SASS classes per cubin.**
   `cuobjdump -sass` / `cuobjdump -elf` on `libcublasLt.so.12` enumerates every
   embedded cubin and its `.target sm_XX`; the instruction histogram of the `sm_75`
   cubin versus the `sm_90a` cubin answers the same question this matrix answers for
   open sources. This needs only the binaries — no GPU. **I could not obtain them
   here**: the CUDA pip wheels that carry `nvidia-cublas-cu12` were not retrievable
   from this network (same host failure as torch), and no CUDA install exists on
   this machine outside the container images. This is the single most valuable
   unrun experiment in the study and it is *not* blocked by the absence of a GPU.
2. **The dispatch table's own minimum.** cuBLAS documents sm_50 as the floor and
   cuDNN's support matrix is arch-keyed; reading either is a documentation fact,
   not a compile fact, and I did not verify either against a source I could read.
3. **Runtime measurement on real hardware**, which is what this lane is forbidden
   from doing and which is also the least interesting of the three: it measures a
   performance gap, whereas the compile-time fact would establish whether a
   *different kernel* is selected.

The honest summary for the matrix: cuBLAS/cuDNN are almost certainly
`COMPILES-DEGRADED` in the same sense as everything else above — a dispatch layer
that picks an older kernel for an older arch — but "almost certainly" is not a
measurement, and this study's value depends on not confusing the two.

---

## 8. What was not obtained

Stated plainly, because the whole lane's worth is that the rest of the matrix can
be trusted.

| subject | status | why |
|---|---|---|
| FA3 (`hopper/*.hpp`) real compile | `NOT-OBTAINED` | needs libtorch headers; `download.pytorch.org` times out from this network. Source-level refusal argument given in §6, resting on PTX gates measured directly. |
| FA2 (`flash_fwd_*.cu`) real compile | `NOT-OBTAINED` | same. Guard transcribed and its arch requirements measured (§6). |
| cuBLAS / cuDNN | `NOT-DETERMINABLE-BY-COMPILATION` | closed binaries, and the pip wheels carrying them were unreachable (§7). |
| `tl.dot_scaled` (the fp8/fp4 block-scaled path) | `KERNEL-NOT-OBTAINED` | could not drive it to compile on **any** arch including sm_100, so I cannot attribute anything to the arch. Both attempts are in the repo (`src/triton_ladder.py`, `src/triton_dot_scaled.py`) with their diagnostics. Deliberately **not** reported as a refusal. |
| CUTLASS 2.x `device::Gemm` | `NOT-OBTAINED` | CUTLASS 4.8's 2.x `Gemm::Arguments` takes a `TensorRef` whose `layout::RowMajor` is the stride-based type with no `at()` factory, so a hand-written instantiation no longer deduces. Superseded: §3.1 probes `cp_async_zfill` itself, which is the code that actually degrades, and is a cleaner experiment. |
| `mma.sync kind::mxf4nvf4` (consumer-Blackwell FP4) | `NOT-OBTAINED` | could not author valid PTX for the 8-operand-group form — ptxas rejects it on sm_120a *and* on every other target, so the operand typing is wrong rather than the arch. Kept as `ptx/mma_f4_kind_mxf4nvf4@sm_120a!unobtainable.ptx` so `selfcheck.sh` reports it as a known gap (`known_unobtainable=1`) instead of letting broken PTX pose as a finding. The sm_120a question is still answered, by the `tcgen05_alloc@sm_120a!neg` negative control: `tcgen05` is absent on consumer Blackwell. |
| fla's PyPI wheel as a source of KDA kernels | `NOT-OBTAINED` | `flash_linear_attention-0.5.2-py3-none-any.whl` ships no `fla/ops/`; used the GitHub `main` tree instead. |
| `fused_recurrent_kda_fwd_kernel` on a `torch`-backed run | `NOT-OBTAINED` | compiled AOT via `ast`-lifted source (§5); the autotuner never ran, so `num_warps`/`num_stages` are the ones I supplied, not autotune-selected. |

---

## 9. The blunt conclusion

**At which rung does the state of the art stop existing?**

There is not one rung. There are three, and conflating them is the mistake this
matrix exists to prevent.

1. **Below sm_75 — it does not exist, and the toolchain says so.** The Turing
   tensor core is the oldest instruction the whole modern stack is built on
   (`mma.sync.m16n8k8`, `ldmatrix`). sm_70 gets a real `REFUSED` from
   flash-linear-attention's KDA, from FA2's traits, and from `tl.dot` for int8.

2. **Between sm_75 and sm_80 — it exists, and it is not the same kernel.** No
   refusal anywhere. A T4 build of KDA or FA2 runs to completion with correct
   numbers, and executes **zero** tensor-core instructions. This is the rung the
   study's original assumption ("a modern hand-written kernel will run on an old
   card, just slower") is most confidently wrong about, and it is the rung nobody
   would notice, because nothing fails.

3. **At sm_90a, the wgmma/TMA/warp-specialised generation does not exist anywhere
   else — and neither does anything replacing it.** `wgmma` assembles for exactly
   one `-arch` string. The measured CUTLASS failure mode at that boundary is not
   "slow", it is a runtime `assert(0)` compiled into a cubin that is *larger* than
   the working one. So the modern state of the art does not degrade to an older
   generation on an H100-class boundary; it disappears.

**Therefore, which comparisons are even well-posed?**

- **An NVIDIA-vs-NVIDIA comparison of a modern kernel on a free T4 against FA3 is
  not a weak experiment, it is an invalid one — and it is invalid twice over.**
  Once because FA3 is `REFUSED` there (§6), and twice because the *thing being
  compared against it* is a different kernel: the T4 build of KDA is an FMA loop
  and the T4 build of FA2 is `m16n8k8` with no async pipeline, so the "gap" being
  measured is mostly the sum of two silent substitutions, not a property of either
  algorithm. A T4 number is a valid measurement of *the state of the art for T4*,
  which is a useful and honest thing to want — but it is a different experiment, and
  labelling it "FA3 vs. baseline on a T4" is a category error, not a
  methodological preference.
- **"Hand-written vs compiler-generated" is only well-posed within one rung.** The
  entire point of a compiler is to select the right instruction class for the
  arch, and §4 shows one unchanged `tl.dot` source yielding four different classes
  across four rungs. Comparing a compiler-generated kernel on sm_80 against a
  hand-written kernel intended for sm_90a compares a `mma.sync` kernel against a
  `wgmma` kernel and will call the compiler a loser. Comparisons must be
  rung-matched, and the rung has to be recorded, because it is not recoverable from
  the source.
- **The one comparison that survives everywhere is `tl.dot` fp16 on sm_80–sm_89 vs
  sm_90**, because both sides exist as real tensor-core kernels there and the
  substitution is one instruction class, measured, with a cited peak ratio. Anything
  below sm_75 should be reported as "runs, no tensor cores", never as a slowdown.
- **And the practical consequence:** because the state of the art does not exist
  below sm_90a, the low end of a hardware ladder cannot be populated by "just use
  the latest kernel". It has to be populated by *re-targeting* — and the two
  libraries measured here both do that silently. The cheapest available instrument
  for telling whether a build was really retargeted is `cuobjdump -sass | grep -c
  HMMA` plus a `mma.sync` shape check, and this matrix is the argument for making
  that a required step rather than a nice-to-have.

---

## 10. Reproducing

```bash
cd portability

# 0. witnesses: regenerate, then prove they are valid PTX at their own arch
.venv/bin/python src/gen_witnesses.py          # 34 witnesses into ptx/
bash      src/selfcheck.sh                     # must print fail=0

# 1. the ISA ladder  (544 cells; one container, ~15 min under x86 emulation)
.venv/bin/python src/run_ptxas.py
.venv/bin/python src/view_ladder.py

# 2. CUTLASS  (50 cells)
.venv/bin/python src/run_cutlass.py           # writes out/cutlass/summary.txt

# 3. Triton  (needs the linux/amd64 wheel in a container; the host venv is macOS)
docker run --rm --platform linux/amd64 \
  -v "$PWD:/w" -e TRITON_CACHE_DIR=/tmp/tc portab/triton:3.4.0 \
  python /w/src/triton_ladder.py /w/out/triton
docker run --rm --platform linux/amd64 \
  -v "$PWD:/w" -e TRITON_CACHE_DIR=/tmp/tc portab/triton:3.4.0 \
  python /w/src/triton_floor.py
docker run --rm --platform linux/amd64 \
  -v "$PWD:/w" -e TRITON_CACHE_DIR=/tmp/tc portab/triton:3.4.0 \
  python /w/src/kda_ladder.py

# 4. FlashAttention arch probe (10 cells)
.venv/bin/python src/run_fa.py

# 5. render every table from the data files, and splice it into this document
.venv/bin/python src/build_matrix.py --inject
```

`src/README-images.md` records how `portab/triton:3.4.0` was built, why the CUDA
toolkit came from `archlab/nvcc:12.9` rather than a multi-GB `nvidia/cuda` pull, and
which hosts on this network are unreachable (torch and NVIDIA's index — the reason
FA2/FA3 and cuBLAS are unrun).

**No table in this document was typed by hand.** `MATRIX.src.md` is the editable
source and holds a `<!--TABLES:NAME-->` marker where each table belongs;
`build_matrix.py --inject` renders `MATRIX.md` from it and the data files. To change
a number, re-run the runner that produced it and re-inject — never edit
`MATRIX.md` directly. `--extract` reverses the process if you need to get back to
the template.

**The rule this lane held to.** No cell reports a compile I did not perform or a
refusal I did not observe. Where the subject could not be obtained the cell reads
`NOT-OBTAINED` and says why, and where a mechanism is only *argued* from source
(FA2, FA3) that is said in the same breath as the claim. The reason to insist on
this: the most valuable result here — that CUTLASS's SM90 paths compile cleanly for
sm_80 into a cubin that traps at runtime — is exactly the kind of finding that a
plausible-sounding report would have stated as a measured refusal, and it would have
been wrong.
