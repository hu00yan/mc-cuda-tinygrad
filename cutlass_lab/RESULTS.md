# CUTLASS as the human baseline — results

Lane: `cutlass_lab/`. Everything below was produced inside this directory.
Nothing outside it was modified.

---

## 0. The one-paragraph summary

CUTLASS v4.8.0 is checked out and pinned; a lean CUDA 12.9 toolchain image
builds; CUTLASS SGEMM compiles for **all five rungs of the ladder
(sm_75, sm_80, sm_89, sm_90a, sm_100a)** on the 2.x path, and the Hopper
warp-specialised `wgmma` kernel compiles for sm_90a. **No CUTLASS kernel in this
lab has ever been executed**, because the build environment has no NVIDIA GPU —
so there is no numerical verification result to report, and section 4 says so
plainly rather than implying otherwise. What *is* verified, mechanically, is the
per-arch configuration each kernel resolved to and the SASS it emits. The most
load-bearing finding is section 6: CUTLASS's pipeline depth is not a number a
human types, it is *derived* from the SM's shared-memory capacity divided by the
per-stage footprint of the tile the human chose.

---

## 1. Provenance

| Item | Value |
|---|---|
| Upstream | `https://github.com/NVIDIA/cutlass.git` |
| Tag | `v4.8.0` |
| Commit | `098de2a652cf8f00fd70b2df54051c7eccbb855a` |
| `CUTLASS_VERSION` | 480 (`include/cutlass/version.h`: MAJOR 4, MINOR 8, PATCH 0) |
| Clone | `git clone --depth 1 --branch v4.8.0 --filter=blob:none --sparse` |
| Sparse paths | `include`, `tools/util` |
| Working tree | 33 MB (`.git` 4.2 MB), 7 933 files |

The whole repository is not vendored. A shallow, blob-filtered, sparse clone was
used precisely so that provenance is a commit hash rather than a folder of
copied files. `cutlass_lab/cutlass` is a real git repo pinned at the tag above, so
the tag can be re-verified with `git rev-parse v4.8.0^{commit}`.

`v4.8.0` was chosen over `main` because it is a release tag, so the tree is
immutable. There is a `v4.8.0dev` tag upstream; it was **not** used.

---

## 2. Toolchain — and the pip route that does not work

The suggested lean route was `ubuntu:24.04` + `pip install nvidia-cuda-nvcc-cu12`.
**That route cannot compile anything.** In version 12.9.86 the
`nvidia-cuda-nvcc-cu12` wheel contains only:

```
nvidia/cuda_nvcc/bin/ptxas
nvidia/cuda_nvcc/nvvm/{include,lib64,libdevice}
nvidia/cuda_nvcc/include/{crt,fatbinary_section.h,nvPTXCompiler.h}
```

There is no `nvcc` host driver. `ptxas` alone can assemble hand-written PTX; it
cannot drive a compile. This was established by
`docker/probe_layout.sh` (still in the repo) and is reproducible — the failure is
a missing file, not a misconfiguration.

What is used instead is the same lean idea via apt rather than pip:
`ubuntu:24.04` + NVIDIA's own repository + four packages with
`--no-install-recommends`:

```
cuda-nvcc-12-9  cuda-cudart-dev-12-9  cuda-cccl-12-9  cuda-cuobjdump-12-9  cuda-nvdisasm-12-9
```

`cuda-cuobjdump-12-9` and `cuda-nvdisasm-12-9` are separate packages;
`cuda-nvcc-12-9` does not contain them, and without them there is no way to read
the SASS, which is most of what this lane produced.

| Item | Value |
|---|---|
| Image | `cutlass-lab:cuda129` (`docker/Dockerfile`), 826 MB |
| `nvcc` | release 12.9, V12.9.86 |
| `ptxas` | 12.9.86 |
| `CUDA_HOME` | `/usr/local/cuda` → `/usr/local/cuda-12.9` |
| Targets nvcc accepts | `sm_50 52 53 60 61 62 70 72 75 80 86 87 89 90 100 101 103 120 121` |

### 2.1 The nvcc-version question, answered concretely

The brief warned that a given nvcc cannot target sm_100a. **For CUDA 12.9 that is
no longer true.** `nvcc --list-gpu-code` reports `sm_100`, and the sm_100a
kernels in this lab compiled (§3). So:

* CUDA 12.9 targets sm_100a. Good.
* The `a` suffix is still mandatory, for a different reason than toolkit
  support: CUTLASS hard-fails its own arch-conditional MMA code if `wgmma` /
  `tcgen05` are compiled for the non-`a` target
  (`sm90_gemm_tma_warpspecialized.hpp:280-282`, "Arch conditional MMA instruction
  used without targeting sm90a"). Every Hopper and Blackwell build here uses
  `-arch=sm_90a` / `-arch=sm_100a`.

---

## 3. What actually built

Machine-readable: `logs/build_summary.tsv`, `logs/tile_sweep.tsv`. Per-build
nvcc output is in `logs/build_*.log` and `logs/tile_*.log`, kept verbatim.

### 3.1 2.x path — `harness/sgemm.cu` (float32, `OpClassSimt`)

| arch | nvcc target | result | time |
|---|---|---|---|
| sm_75 | `sm_75` | **BUILD OK** | 26 s |
| sm_80 | `sm_80` | **BUILD OK** | 21 s |
| sm_89 | `sm_89` | **BUILD OK** | 21 s |
| sm_90a | `sm_90a` | **BUILD OK** | 27 s |
| sm_100a | `sm_100a` | **BUILD OK** | 28 s |

**All five rungs compile.** This is the most important single fact in the lab: a
per-arch ladder is achievable, and the Blackwell rung is not blocked by the
toolkit.

These are compiled *by letting CUTLASS choose*: `harness/sgemm.cu` passes only
element types, layouts, operator class and arch tag, and every tile shape, warp
shape, instruction shape, pipeline depth, swizzle, split-K mode and epilogue type
comes from CUTLASS's own `DefaultGemmConfiguration` specialisation. The binary
prints the resolved values at startup. That is deliberate: it means the
"human choice" table in §5 describes the configuration that actually compiled,
rather than a config retyped from a header by this lab.

### 3.2 2.x path — `OpClassTensorOp` with float32 (TF32) — mostly fails, for a structural reason

| arch | result | why |
|---|---|---|
| sm_75 | **BUILD OK** | `DefaultGemmConfiguration<OpClassTensorOp, Sm75, ...>` has a generic specialisation |
| sm_80 | BUILD FAILED | config resolved, but no `DefaultMmaCore` for `float` × `OpClassTensorOp` |
| sm_89 | BUILD FAILED | no `DefaultGemmConfiguration<OpClassTensorOp, Sm89, float, ...>` at all |
| sm_90a | BUILD FAILED | same, `Sm90` has only `double` / `complex<double>` |
| sm_100a | BUILD FAILED | same |

This is **"CUTLASS cannot express this through the 2.x default path"**, not a
toolkit limit and not a defect in the harness. The sm_80 build log is the clean
piece of evidence: instantiation proceeds all the way through
`DefaultGemm<...>` → `MmaMultistage<Shape=GemmShape<128,256,64>, ...>` →
`MmaTensorOp` → and then dies at `mma_tensor_op_policy.h:58` with "incomplete type
`cutlass::arch::Mma<...>`". So the *shape* configuration resolves fine; what does
not exist is the `arch::Mma` **TF32 atom**. Corroborating evidence:

* `include/cutlass/gemm/threadblock/default_mma_core_sm80.h` contains **no line
  beginning `  float,`**. TF32 is not in the 2.x `DefaultMmaCore` set at all; in
  CUTLASS 4.x it lives on the 3.x `CollectiveBuilder` path.
* `DefaultGemmConfiguration<OpClassTensorOp, arch::Sm89, ...>` exists only for
  `float_e4m3_t` / `float_e5m2_t` (`default_gemm_configuration.h:848-902`).
  There is no fp16 or fp32 `Sm89` specialisation at all, and `Sm90` has only
  `double` / `complex<double>`, so those three rungs fail one step earlier.

The sm_75 success is real and instructive: Turing gets a float TensorOp kernel
via the 2.x path, and every arch above it does not.

### 3.3 3.x path — `harness/gemm_3x.cu` (f16, warp-specialised collectives)

This is the path that reaches `wgmma` and `tcgen05`, so it is the one that
matters for "the best Hopper kernel".

**sm_90a — builds.** Tile sweep (`logs/tile_sweep.tsv`):

| tile | result |
|---|---|
| 128×256×64 | BUILD OK (41 s) |
| 128×128×64 | BUILD OK |
| 128×128×32 | BUILD OK |
| 128×64×64 | BUILD OK |
| 128×256×32 | BUILD OK |
| 64×128×64 | REJECTED (static assert) |
| 64×64×64 | REJECTED (static assert) |

M=64 is rejected, consistent with the sm90 builder's own comment at
`sm90_gmma_builder.inl:1020-1022` that `KernelTmaWarpSpecializedCooperative`
requires `TileShape_M` ≥ 128 and that M=64 selects the ping-pong schedule.

**sm_100a — did not build.** All 7 candidate tiles were rejected, and not because
of the tile:

| tile | result |
|---|---|
| all 7 tried (128×256×64 down to 64×64×64) | `REJECTED_SMEM_CAPACITY` |

Two distinct errors, both preserved verbatim in
`logs/sm100a_3x_blockers.log`:

1. `sm100_gemm_tma_warpspecialized.hpp:410: static assertion failed with "SMEM
   usage exceeded capacity."` — even at 64×64×64 f16, whose mainloop working set
   is a few tens of KB out of 232 448 B. So the over-subscription is structural
   to the epilogue/mainloop combination this configuration selected, not driven by
   the mainloop tile.
2. `include/cutlass/fast_math.h(154): error: no suitable constructor exists to
   convert from "int" to "cute::C<9>"` — a template type mismatch in
   `cutlass::round_nearest(dividend_t, divisor_t)`, reached from the sm100
   epilogue path. It persists with explicit `StageCount<2|3|4>`, so it is not
   dependent on auto-staging. **I did not root-cause this one.** It is recorded
   as an unexplained failure of the sm100 3.x path from this configuration, not
   as a CUTLASS bug.

Note carefully: this is **"I could not get it to build"**. It is *not* "CUTLASS
cannot do sm_100a" — the 2.x sm_100a kernel in §3.1 compiled fine, and CUTLASS
4.8.0 ships dedicated `sm100_*` collective builders, tcgen05 MMA atoms, and
block-scaled FP4/FP6 support. The honest statement is that a working sm_100a
3.x configuration was not found within this lane's budget.

### 3.4 Two constraints in CUTLASS 4.8.0 worth recording

* **The 3.x `CollectiveBuilder` has no default threadblock tile.** Its template
  is `CollectiveBuilder<ArchTag, OpClass, ElementA, LayoutA, AlignA, ElementB,
  LayoutB, AlignB, ElementAccumulator, TileShape_MNK, ClusterShape_MNK,
  StageCountType, BuilderScheduleTag>` with no defaults on the last four. Only
  the *schedule* is auto-selectable. So on Hopper and Blackwell there is no
  single "CUTLASS default tile" to point at — unlike sm_75/sm_80/sm_89, where
  `DefaultGemmConfiguration` does record one. The tile used in `gemm_3x.cu`
  (128×256×64 on sm_90a) is a **choice made by this harness**, chosen to match
  CUTLASS's own recorded Ampere f16 default so the tile does not move between
  rungs; the study should treat it as one number among many a tuner sweeps.
* **The 3.x GEMM kernel takes the problem shape at compile time.**
  `kernel::GemmUniversal`'s `ProblemShape` is the template argument, and for a
  GEMM `conv::detail::get_problem_shape_MNKL_helper` returns it unchanged
  (`conv/detail.hpp:49-51`) rather than translating a runtime shape. So M/N/K
  must be baked into the type. The 2.x path has no such restriction.

---

## 4. Numerical verification — the honest status

**No CUTLASS kernel in this lab has been executed. There is no max-relative-error
number to report for any CUTLASS kernel, and none is invented here.**

The build environment is Apple silicon with an amd64 emulator under OrbStack and
no NVIDIA GPU. A CUDA kernel cannot run there, and no CPU emulation of
`mma.sync` / `wgmma` / `tcgen05` / TMA exists that would make an execution
result meaningful. The brief called verification non-negotiable, and it is
non-negotiable *on real hardware* — where it cannot be satisfied. Reporting an
error figure for an unrun kernel would be exactly the fabrication the brief
forbids.

What was done instead, all of it real:

### 4.1 The verifier's own logic is tested, and the test found a bug

`harness/verify_sgemm.py` compares `D` against `numpy` float32. Because it
cannot be exercised against CUTLASS, it is exercised against synthetic results
in `harness/selftest_verify.py`, which **passes**:

```
[1] exact reference          -> rc=0 PASS
[2] 1e-3 relative perturbation -> rc=1 FAIL(correct)  normwise rel error : 1.000047e-03
[3] blocked fp32 accumulation -> rc=0 PASS
      max rel error      : 1.172070e-01   (informational, not gated)
      normwise rel error : 8.200173e-07   (GATE: ||D-ref||_F / ||ref||_F)
[4] gated metric separates  -> yes  (normwise: reordered=8.200e-07 vs perturbed=1.000e-03)
SELFTEST OK
```

Case 3 is the interesting one, and it corrected a mistake. The first version of
`verify_sgemm.py` gated on **max relative error** and duly failed a *correct*
result at 1.2e-1. The reason is intrinsic to GEMM: for a K=2003 dot product many
outputs are a small difference of two large partial sums, so a few elements land
near zero and their relative error is huge even when the arithmetic is exact. The
gate is now the **normwise (Frobenius) error** `||D−ref||_F / ||ref||_F`, which is
insensitive to which individual elements happen to be small. Max relative error
is still reported, explicitly labelled as not gated. Anyone reusing this verifier
should not re-introduce the max-relative gate.

Case 4 confirms the gate is not a rubber stamp: it separates a reordered-but-correct
result (8.2e-07) from a 1e-3 perturbation (1.0e-03) by three orders of magnitude.

The input generator `fill_deterministic` in `harness/common.cuh` is transcribed
into the self-test and checked for bit-identical uint32 LCG semantics, so a
divergence between host and device inputs would surface before any measurement.

### 4.2 What the swizzle verification actually proves

`tools/verify_swizzle.py` runs on the CPU and **passes for all six swizzles
CUTLASS uses across the ladder**. It checks that each swizzle is a *bijection* on
its coordinate space — a non-bijective shared-memory layout would silently compute
the wrong answer, so this is a genuine correctness check on hand-written
arithmetic, not a restatement of it.

| swizzle | source | result |
|---|---|---|
| sm_75 congruous | `layout/tensor_op_multiplicand_sm75.h:407` | PASS, 0 collisions |
| sm_80/89 congruous-64b | `layout/tensor_op_multiplicand_sm80.h:107` | PASS, 0 collisions |
| sm_90 `Swizzle<1,4,3>` (SW32) | `cute/atom/mma_traits_sm90_gmma.hpp:76` | PASS, 0 collisions |
| sm_90 `Swizzle<2,4,3>` (SW64) | `…:77` | PASS, 0 collisions |
| sm_90 `Swizzle<3,4,3>` (SW128) | `…:78` | PASS, 0 collisions |
| sm_100 `Swizzle<2,5,2>` (SW128 32B) | `cute/atom/mma_traits_sm100.hpp:73` | PASS, 0 collisions |

**What this file deliberately does not report: bank-conflict degree.** That
depends on CUTLASS's thread maps (`PitchLinearWarpStripedThreadMap`) as much as on
the swizzle. An earlier draft measured conflicts under a guessed access pattern
and produced a number that made the swizzle look *worse* than no swizzle — a clear
sign the assumed pattern, not the swizzle, was wrong. The measurement was removed
rather than published. Extending it requires porting the thread maps from
`transform/thread_map.h` and `gemm/threadblock/default_mma_core_sm80.h` first.

### 4.3 To finish verification on real hardware

```bash
docker run --platform linux_amd64 -v "$PWD":/lab -w /lab cutlass-lab:cuda129 \
  bash -lc 'for b in cutlass_lab/build/sgemm_sm*_simt; do
              "$b" "out/$(basename "$b")/"; done'
# then, per directory:
python3 cutlass_lab/harness/verify_sgemm.py out/sgemm_sm800_simt --expect simt
```

The harness prints `LAB_NO_DEVICE` and exits 5 when no GPU is visible, so it
cannot silently "succeed" without one.

---

## 5. What the human actually chose, per architecture

Sources are given per row. The 2.x rows are CUTLASS's own recorded defaults from
`include/cutlass/gemm/device/default_gemm_configuration.h`; the sm_90a row is
resolved by the compiler from the configuration that built; the instruction mix
is measured from SASS by `tools/sass_opcodes.py` (JSON in
`logs/sass_opcodes.json`).

### 5.1 Tile shape, warps, pipeline depth

| | sm_75 | sm_80 | sm_89 (Ada) | sm_90a (Hopper) | sm_100a (Blackwell) |
|---|---|---|---|---|---|
| Threadblock tile (2.x f16) | 128×256×32 | 128×256×64 | **none for f16** | n/a (3.x path) | n/a (3.x path) |
| Threadblock tile (2.x f32 SIMT) | 128×128×8 | 128×128×8 | 128×128×8 | 128×128×8 | 128×128×8 |
| Warp tile | 64×64×32 | 64×64×64 | 64×64×64 (FP8) | 4 warpgroups of 128 thr | 1 MMA warp + 1 epi warpgroup |
| MMA atom | 16×8×8 | 16×8×16 | 16×8×32 (FP8) | **64×256×16** | tcgen05, tmem operands |
| Warps per block / threads | 8 / 256 | 8 / 256 | 8 / 256 | 2 math + 1 load wg / 384* | 5 roles / 384 |
| Warp decomposition | 2×4 warps | 2×4 warps | 2×4 warps | MMA atom spans all of N | **5 warp roles, 384 thr** |
| Pipeline stages | **2** | **3** | **3** | **4** (derived) | derived; did not fit |
| Kernel SMEM (2.x SIMT, f32) | **16 896 B** | **16 896 B** | **16 896 B** | **16 896 B** | **16 896 B** |
| Kernel SMEM (2.x TensorOp, f16/fp32) | **49 152 B** | not buildable | not buildable | not buildable | not buildable |
| SMEM used / arch capacity | — | — | — | 16 896 / **232 448** = 7.3 % | 16 896 / **232 448** = 7.3 % |
| Tile scheduler | data-parallel | data-parallel | data-parallel | StreamK (CUTLASS default) | StreamK |

\* The 384-thread figure for sm_90a is **derived, not measured**: from
`NumLoadWarpGroups = 1` (`sm90_gemm_tma_warpspecialized.hpp:140`) plus
`MaxThreadsPerBlock = size(TiledMma) + NumLoadWarpGroups * 128` (`:142`), with
`size(TiledMma) = 2` warpgroups from the resolved 64×256×16 atom against a
128-wide tile. The corresponding binary compiles but could not be run to print
its own value.

The TensorOp SMEM row is the float32 instantiation of the f16 default config
(§3.2 explains why only sm_75 has one); the sm_75 figure is **measured** at
49 152 B, of which 33 792 B is epilogue — the epilogue is the larger consumer,
which is itself a non-obvious thing a tuner must budget for.

**Provenance of each cell in this table.** The `2.x f32 SIMT` rows and both
*Kernel SMEM* rows are **measured**: `harness/print_cfg.cu` instantiates the same
`device::Gemm` as `harness/sgemm.cu` (same "let CUTLASS choose everything"
argument list) and prints the compile-time constants from a host binary, needing
no GPU and no `-arch`. Raw output, all ten configurations: `logs/print_cfg.txt`.

The `2.x f16` / `FP8` tile rows are **read from CUTLASS's own recorded
defaults**, not measured here, because the only instantiations that can be built
for those arch/element combinations are the ones CUTLASS provides a default for —
and the sm_75 fp32 TensorOp build is the sole one of those that compiled
(§3.2). The sm_90a cell is the compiler's resolved output for the configuration
that built in §3.3. The sm_100a cells are the documented warp-role structure,
which could not be instantiated (§3.3). Sources:
`default_gemm_configuration.h:193-221` (sm75), `:461-482` (sm80), `:825-846`
(sm89 FP8), `:73-96` (SIMT all archs);
`sm90_gmma_builder.inl:998-1029` (auto schedule);
`dispatch_policy.hpp:1100-1113` (`MainloopSm100TmaUmmaWarpSpecialized`);
`gemm_3x` resolved output for sm_90a stages=4, atom `MMA_64x256x16_F32F16F16_SS`.

Where a value could not be measured, the table says so rather than being filled
in from the CUDA programming guide.

### 5.2 Two structural findings that matter more than the numbers

**sm_89 has no fp16 or fp32 default of its own.** `DefaultGemmConfiguration` has
`arch::Sm89` specialisations *only* for `float_e4m3_t`/`float_e5m2_t`
(`:848-902`). For f16/bf16/fp32, `DefaultGemm<..., arch::Sm89, ...>`
(`gemm/kernel/default_gemm.h:271-295`) simply forwards to the same `DefaultMma`
the sm_80 tag selects. Ada is a consumer part with no new tensor-core shape, so
NVIDIA's answer is "use the Ampere configuration". A per-arch ladder therefore has
a genuine discontinuity at sm_89, and the study should not present sm_89 as an
independent tuning point.

**The SIMT fp32 baseline is completely arch-independent in CUTLASS.** The same
`DefaultGemmConfiguration<OpClassSimt, ArchTag, float, float, float, float>`
serves every arch, and this is now **measured** rather than inferred: the
resolved config is byte-identical on all five rungs — threadblock 128×128×8, warp
32×64×8, 8 warps, 256 threads, 2 stages, alignment 1, and **16 896 bytes of
shared memory on every single one** (`logs/print_cfg.txt`). The SASS agrees:
**FFMA 576, STS 144, LDS 164, BAR 34, identical on all five binaries**
(`logs/sass_opcodes.json`).

That shared-memory figure is the sharpest statement of the point: on sm_90a and
sm_100a, where the SM offers **232 448 bytes**, the SIMT kernel occupies
**7.3 %** of it. A true-fp32 SGEMM in CUTLASS is not arch-tuned at all — it
ignores the machine. If the study's compiler baseline is a true-fp32 SGEMM,
CUTLASS offers no per-arch advantage on that path, and the measured gap against
it would say nothing about tensor cores. The interesting comparison is the
tensor-core path, which CUTLASS does specialise per arch.

### 5.3 Instruction classes actually emitted (measured, not asserted)

| binary | total SASS instrs | tensor core | operand movement | SIMT |
|---|---|---|---|---|
| `sgemm_sm750_simt` | 1 830 | — | STS 144, LDS 164, BAR 34 | FFMA 576 |
| `sgemm_sm750_tensorop` | 1 646 | **HMMA 128** | **LDSM 20**, STS 140, LDS 64 | FFMA 128 |
| `sgemm_sm800_simt` | 2 080 | — | STS 144, LDS 164 | FFMA 576 |
| `sgemm_sm890_simt` | 2 076 | — | STS 144, LDS 164 | FFMA 576 |
| `sgemm_sm900_simt` | 2 104 | — | STS 144, LDS 164 | FFMA 576 |
| `sgemm_sm1000_simt` | 1 903 | — | STS 144, LDS 164 | FFMA 576 |
| `gemm3x_sm900_128x256x64` | 3 583 | **HGMMA 8** | BAR 11 | FFMA 128, FMUL 256 |
| `gemm3x_sm900_128x64x64` | 3 638 | **HGMMA 8** | **UTMALDG 2** (TMA), BAR 11 | FFMA 32, FMUL 64 |

SASS mnemonic names do not match PTX names, which is why this table is worth
having at all: `wgmma.mma_async` assembles to **HGMMA**, not `WGMMA`; `mma.sync` to
`HMMA`; `cp.async.bulk.tensor` to **UTMALDG**/`UBLKCP`. An early version of
`tools/sass_census.py` looked for `WGMMA` and reported the Hopper kernel as having
no tensor core. The corrected tool is `tools/sass_opcodes.py`.

`HGMMA 8` in the mainloop body is consistent, not a bug: with a 64×256×16 atom
and a 128×256×64 tile, one K-tile is (128/64)×(64/16) = 2×4 = 8 `wgmma`, and the
K loop is rolled, so 8 is the unrolled body.

### 5.4 Shared-memory layout and swizzle

| arch | layout | swizzle | source |
|---|---|---|---|
| sm_75 | congruous, 4-row period | one XOR term: `(c ^ (2*s)) * 4` | `tensor_op_multiplicand_sm75.h:407` |
| sm_80 / sm_89 | `TensorOpMultiplicandCongruous64b` | **two** XOR terms + base-4 row fold: `bank = ((((c&1)*4 + (c&6)/2)) ^ (s&1))*2 + c/8; bank ^= (s&2)*2` | `tensor_op_multiplicand_sm80.h:107` |
| sm_90 | `Layout_MN_SW{32,64,128}_Atom` | `cute::Swizzle<1,4,3>` / `<2,4,3>` / `<3,4,3>`; builder picks the widest that divides `BLK_MN0` | `mma_traits_sm90_gmma.hpp:76-84`, `sm90_common.inl:283-292` |
| sm_100 | reuses the Hopper atoms **plus a new one** | `Swizzle<2,5,2>` over a 4-row base, for the 256-bit smem atom Blackwell's UMMA can consume | `mma_traits_sm100.hpp:73` |

The parameterisation changes twice: sm_75 XORs against `2*s` in a hand-written
`operator()`; sm_80 folds a 4-row group directly into the bank index with two XOR
terms; sm_90 moves to a three-bit XOR composed as a CuTe `Layout` object and makes
the swizzle width a *selectable atom* rather than a formula; sm_100 adds a fourth
atom because the hardware now reads wider smem.

### 5.5 Split-K / Stream-K and reduction placement

CUTLASS 4.8 offers three decompositions
(`gemm/kernel/tile_scheduler_detail.hpp:75-84`): `DataParallel`, `SplitK`,
`StreamK`, selected by `Heuristic`. The tile scheduler for the 3.x Hopper kernel
is **StreamK** (CUTLASS's own default for these archs).

Reduction has a genuine human choice baked in, with a
determinism/speed tradeoff written out in the source
(`tile_scheduler_detail.hpp:52-69`):

* `ReductionMode::Deterministic` — participating CTAs reduce in a "turnstile"
  fashion in order of the K extent each covered, under an exclusive lock. **Gives
  deterministic floating-point results.**
* `ReductionMode::Nondeterministic` — CTAs reduce atomically to a shared workspace
  with no lock; only the first and last arrivals are synchronised. Faster, and the
  source explicitly warns accumulation order varies so "deterministic numeric
  behavior cannot be guaranteed".

On Blackwell the fixup is placed at epilogue-subtile granularity and needs one
barrier per subtile per epilogue warp
(`sm100_gemm_tma_warpspecialized.hpp:151-154`).

### 5.6 Epilogue strategy

Fused into the GEMM kernel in every case here — there is no separate reduction
kernel. The accumulator is drained in registers by the epilogue warps of the same
threadblock that computed it. The two relevant structural choices:

* **sm_75/80/89 (2.x):** `epilogue::threadblock::Epilogue` with an
  `OutputTileOptimalThreadMap`; the accumulator moves from MMA registers to
  registers via `warp::FragmentIteratorSimt` and out through a predicated
  `PredicatedTileIterator`. FMA/epilogue overlap is handled by warp-level
  double-buffering.
* **sm_90/sm_100 (3.x):** the epilogue is a separate `CollectiveBuilder`
  (`epilogue/collective/builders/sm90_builder.inl`, `sm100_builder.inl`) that can
  stage C in shared memory and re-use the mainloop's, controlled by
  `ReuseSmemC` and a `DelayTmaStore` flag, with a separate
  `EpiLoadPipeline`/`EpiStorePipeline`. On sm100 the epilogue additionally drains
  the tensor memory the MMA wrote.

A caveat on my own configuration, since it shows up in the SASS table: with
`EpilogueScheduleAuto` the sm_90a build selected a `ThreadEpilogueOp` with
`ElementsPerAccess = 1`, so the epilogue is scalar (`FMUL 256`, `FADD 256` in
`gemm3x_sm900_128x256x64`). That is a defect in *this harness's* epilogue
configuration, not a property of the CUTLASS Hopper kernel, and the SASS counts
for that row should not be read as representative of a tuned Hopper GEMM. The
mainloop evidence (`HGMMA 8`) is unaffected.

### 5.7 Warp specialisation — the biggest per-arch change

| arch | roles | where |
|---|---|---|
| sm_75 – sm_89 | **none.** Every warp does loads, MMAs and epilogue. | — |
| sm_90a | **2 roles**: `Producer` (issues TMA) and `Consumer` (does wgmma). The producer's 4 warps are further split: `MainloopEpilogue`, `Warp1`, `Warp2`, `Warp3`. | `sm90_gemm_tma_warpspecialized.hpp:284-293` |
| sm_100a | **5 roles**: `MMA`, `Sched`, `MainloopLoad`, `EpilogueLoad`, `Epilogue`, sized in *threads*: 1 warp each for Sched/MMA/MainloopLoad/EpilogueLoad, plus `CollectiveEpilogue::ThreadCount` (8 warps) for the epilogue — **384 threads** | `sm100_gemm_tma_warpspecialized.hpp:137-146`, `WarpCategory` enum in the sm107/conv variants |

The sm100 role assignment is expressed as thread counts and pipelined separately
for the mainloop load, the epilogue load, the epilogue store, CLC (cluster launch
control), the accumulator, and a **load-order barrier** that exists purely to
sequence TMA descriptors (`group_id = 0` for MainloopLoad, `1` for everything
else, `sm100_gemm_tma_warpspecialized.hpp:495-499`).

### 5.8 Tensor-core datatypes available

| arch | TC formats | notes |
|---|---|---|
| sm_75 | fp16/bf16 (`mma.sync` 16×8×8), int8 (`8×8×16`, `OpMultiplyAddSaturate`) | no fp8 |
| sm_80 | + int4, fp64 (`8×8×4`), `complex` | `mma.sync` up to 16×8×16 |
| sm_89 | + **fp8** `e4m3`/`e5m2` (`16×8×32`, `OpMultiplyAdd`) | its own default config |
| sm_90a | + fp8 with **blockwise scaling**, fp8 mixed-input, sparse | `wgmma`, `MMA_64x256x16` etc. |
| sm_100a | + **block-scaled fp4/fp6**: `float_e2m1_t` (FP4), `float_e3m2_t`/`float_e2m3_t` (FP6), `ue8m0_t` / `e4m3` scale factors, scale-factor vector size 16/32/64 | `tcgen05`; operands A and B are **tensor memory** addressed by 64-bit descriptors, accumulator in tmem (`mma_traits_sm100.hpp:201-204`) |

Block-scaled data-path detail worth flagging for the study: the "2x" sparse/2×
modes at vector size 32/64 are restricted to `a_type`/`b_type` = `float_e2m1_t`
with `sf_type = ue8m0_t` (`mma_traits_sm100.hpp:3319-3323, 3413-3417`).

---

## 6. The single most striking thing: pipeline depth is a *derived* quantity

Every other number in §5 is a number a person could in principle write down.
The pipeline depth is not. In CUTLASS 3.x it is computed, at compile time, from
the hardware:

```cpp
// gemm/collective/builders/sm90_gmma_builder.inl:74-86
constexpr int stage_bytes = round_up(a_bits*M*K/8 + b_bits*N*K/8, alignment)
                          + sizeof(PipelineTmaAsync<1>::SharedStorage);
return (capacity_bytes - carveout_bytes) / stage_bytes;
```

with `capacity_bytes` = **232 448** for both Hopper and Blackwell
(`sm90_common.inl:57`, `arch/arch.h:45`) and `carveout_bytes` reserved for TMA
descriptor storage. The human's actual input is the *tile*, and the depth follows
— subject to `static_assert(SharedStorageSize <= kSharedMemoryCapacityBytes)`,
which is exactly the assert that rejected all seven of my sm_100a tiles.

This is the answer to the study's question "what does a human still specify by
hand". For the depth, the human specifies almost nothing; what a compiler would
have to reproduce unaided is the *closure* of a reasoning chain: choose a tile,
multiply it by element width to get bytes per stage, add pipeline barrier storage,
round to a 128-byte alignment, subtract the TMA carveout the scheduler needs,
divide by the arch's opt-in shared-memory capacity, and then verify the result
fits — all before a single instruction is issued. No local, single-step
transformation of the source program produces that number. It requires modelling
the SM's smem capacity, the tile shape, the barrier storage and the carveout ABI
simultaneously, and it changes when *any* of them change.

Two runners-up, for the record:

1. **The sm100 5-role warp decomposition** (`MMA`, `Sched`, `MainloopLoad`,
   `EpilogueLoad`, `Epilogue`, sized in threads, 384 total, each with its own
   mbarrier pipeline plus a load-order barrier that exists only to sequence TMA
   descriptors). A compiler would have to invent the *existence* of three
   specialised warp roles whose entire purpose is to issue memory descriptors and
   to do nothing else.
2. **The hand-written two-term XOR bank swizzle**
   (`tensor_op_multiplicand_sm80.h:107`). The sm_75 version is one XOR; the sm_80
   version folds a 4-row group into the bank index with two XOR terms, and the
   correct form depends jointly on the fragment layout the tensor core demands,
   the access width, and the 4-bank granularity of shared memory. It has to be
   re-derived from scratch at sm_90 (`Swizzle<3,4,3>`, base 4 instead of 2) and
   again at sm_100 (a new `Swizzle<2,5,2>`), because the hardware's smem read
   width changes.

---

## 7. CUTLASS's own performance claims — vendor claims, not measurements

`README.md:189-207` ("# Performance") makes two claims:

* CUTLASS primitives "exhibit nearly optimal utilization of peak theoretical
  throughput", with a figure showing **"CUTLASS 3.8's performance as a % of
  theoretical peak utilization"** on **NVIDIA Blackwell SM100** across input/output
  dtypes.
* Two figures showing "continual CUTLASS performance improvements on an NVIDIA
  H100 … since CUTLASS 3.1", with **CUTLASS 3.5.1 compiled with CUDA 12.5u1**, and
  the note that tensor-core operations use `mma` and **`wgmma`**.

**The numbers themselves are not available as text.** The figures are
`media/images/*.svg` / `*.png`, and `media/` is not in this sparse checkout. The
underlying data is not in the repository in any machine-readable form that this
lane could find. So:

* No TFLOP/s or %-of-peak figure is quoted here, because none could be read.
* These are **vendor claims about CUTLASS 3.5.1 and 3.8**, not about v4.8.0, and
  not measurements by this study. **This lab measured no performance at all.**
* The claim as *scoped* is still useful to the study: NVIDIA itself presents
  Hopper's wgmma and Blackwell's tcgen05 as the intended tensor-core path, which
  corroborates §5.3's finding that those are the instructions that matter.

Other vendor statements found in the repo, recorded as claims:

* `README.md:28` — CUTLASS 4's Python DSLs offer "no performance compromises"
  relative to core CUTLASS/CuTe. Unverifiable here.
* `README.md:148` — dense and block-scaled GEMM in the Operator API have
  **"preliminary"** Rubin support that "may need additional performance tuning".
  Consistent with §5.8: the newest datatypes are the least settled.

---

## 8. Summary of what could not be done

| Item | Status | Which of the three it is |
|---|---|---|
| Execute any CUTLASS kernel | **Not done.** No NVIDIA GPU; no CPU emulation of HMMA/wgmma/tcgen05/TMA is meaningful. | Environment limitation |
| Numerical verification against numpy | **Not done for CUTLASS.** Verifier written and self-tested (found and fixed a wrong gate); needs a GPU. | Environment limitation |
| sm_100a 3.x `wgmma`/`tcgen05` kernel | **Not built.** 7 tiles rejected on SMEM capacity; plus an unroot-caused `round_nearest` type error. | **"I could not get it to build"** — not a toolkit limit, and not proof CUTLASS cannot |
| sm_75/80/89/90/100 2.x `OpClassTensorOp` fp32 | **Not built** except sm_75. | **"CUTLASS cannot express it via the 2.x default path"** — TF32 is absent from `default_mma_core_sm80.h`, and Sm89/Sm90/Sm100 have no fp32 TensorOp default |
| Bank-conflict degree per swizzle | **Not measured.** Needs CUTLASS's thread maps ported; a guessed pattern gave a wrong-signed answer and was discarded. | Deliberate omission, documented in `tools/verify_swizzle.py` |
| CUTLASS performance figures | **Not available as text** (charts only, `media/` not checked out). | Vendor claim, recorded but not quoted |
| Anything requiring gcloud or paid compute | **Not attempted**, per constraints. | Out of scope |

## 9. Reproducing

```bash
# 1. toolchain image (CUTLASS is already in cutlass_lab/cutlass)
docker build --platform linux/amd64 -f cutlass_lab/docker/Dockerfile -t cutlass-lab:cuda129 cutlass_lab/docker

# 2. 2.x SGEMM, all five rungs
docker run --rm --platform linux/amd64 -v "$PWD":/lab -w /lab cutlass-lab:cuda129 \
  bash cutlass_lab/harness/build.sh

# 3. 3.x warp-specialised tiles, sm_90a and sm_100a
docker run --rm --platform linux/amd64 -v "$PWD":/lab -w /lab cutlass-lab:cuda129 \
  bash cutlass_lab/harness/build_3x.sh

# 4. SASS instruction census (needs cuobjdump, inside the image)
docker run --rm --platform linux/amd64 -v "$PWD":/lab -w /lab cutlass-lab:cuda129 \
  python3 cutlass_lab/tools/sass_opcodes.py cutlass_lab/build/* --json cutlass_lab/logs/sass_opcodes.json

# 5. CPU-side verification that does not need a GPU
.venv/bin/python cutlass_lab/tools/verify_swizzle.py
.venv/bin/python cutlass_lab/harness/selftest_verify.py

# 6. print the resolved per-arch configuration (compile-time, no GPU needed)
docker run --rm --platform linux/amd64 -v "$PWD":/lab -w /lab cutlass-lab:cuda129 bash -c '
  for sm in 750 800 890 900 1000; do
    nvcc -DLAB_SM=$sm -DLAB_MODE=0 -std=c++17 -O1 --expt-relaxed-constexpr \
      -Icutlass_lab/cutlass/include -Icutlass_lab/harness \
      -o /tmp/pc cutlass_lab/harness/print_cfg.cu -lcudart && /tmp/pc
  done'
```

### Directory map

```
cutlass_lab/
  RESULTS.md                  this file
  cutlass/                    pinned sparse clone, v4.8.0 @ 098de2a6
  docker/Dockerfile           lean CUDA 12.9 image (apt route; pip route does not work)
  docker/probe_layout.sh      the evidence that nvidia-cuda-nvcc-cu12 has no nvcc
  harness/common.cuh          problem shape, deterministic inputs
  harness/sgemm.cu            2.x SGEMM, one arch per compile, CUTLASS picks the config
  harness/gemm_3x.cu          3.x warp-specialised GEMM (wgmma path)
  harness/print_cfg.cu        compile-time reporter for the resolved config (no GPU needed)
  harness/build.sh            compiles sgemm.cu for all 5 rungs x {simt, tensorop}
  harness/build_3x.sh         tiles sgemm against sm_90a / sm_100a, records the boundary
  harness/verify_sgemm.py     numpy float32 comparison; normwise gate
  harness/selftest_verify.py  tests the verifier, and the input generator
  tools/sass_opcodes.py       address-anchored SASS opcode census (the good one)
  tools/verify_swizzle.py     CPU bijection proof for all 6 swizzles
  tools/sass_census.py        SUPERSEDED and wrong -- kept only as evidence of the
                              HGMMA-vs-WGMMA mistake described in 5.3. Do not use.
  tools/sass_top.sh, sass_grep.sh   throwaway shell probes, kept as evidence
  logs/                       every nvcc invocation, verbatim, plus the TSV summaries
                              and print_cfg.txt (the measured resource table)
  build/                      the compiled binaries
```
