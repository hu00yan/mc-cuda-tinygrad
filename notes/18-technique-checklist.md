# 18 — Hand-written CUDA inventory for `mine.cu`: what the human actually did

> **The verdict.** `netherite/mine.cu` (a shallow clone of `Infatoshi/mine.cu`) is
> 3,245 lines of CUDA in `src/kernels.cu` plus 1,269 lines of host glue in
> `src/bindings.cpp`. I read all of it. **34 distinct entries are inventoried in
> §3: 7 negative findings and 27 techniques. Three of the 27 are written,
> non-obvious, and unreachable from Python, so the shipped set is 24.** The seven
> negative findings are not padding — the technique the brief asked me to look for
> is absent, and its absence is the result.
>
> **The warp-level family is not here.** `grep -n "__shfl\|__ballot\|__popc\|__any\|__all\|__activemask\|__syncwarp\|__match"` over `src/kernels.cu` returns **zero
> hits in 3,245 lines**. There is not one `atomicAdd`, no `__launch_bounds__`, no
> `-maxrregcount`, no cooperative-groups grid sync, no sort, no compaction, no
> prefix scan, no `cp.async` (the header at `kernels.cu:831` claims cp.async and
> `#include <cuda_pipeline.h>` is at `kernels.cu:6`, but the copy at
> `kernels.cu:868-879` is a plain `int4` load plus `__syncthreads()`). There is no
> RNG: `generate_world_kernel` takes `unsigned int seed` at `kernels.cu:1800` and
> never reads it. A study that assumes this file is a warp-cooperation or
> occupancy-regime exhibit is pointed at the wrong artifact.
>
> **What is actually here, in one line each:** a pixel-linear thread→pixel mapping
> with no swizzle; a batch-major voxel layout that makes the one irregular write
> conflict-free; a whole-step CUDA-graph capture; a basis-hoist that splits a
> kernel in two; and a build flag (`--use_fast_math`, `setup.py:19`) that quietly
> owns the entire precision budget.
>
> **The number that bounds everything else** is in §2: at the quoted peak config
> the float32 observation write alone is **610 GB/s, 65.2% of the RTX 3090's 936
> GB/s spec bandwidth, and its DRAM floor is 430 µs of a 660 µs step**. The render
> kernel at peak is write-bandwidth-bound, and the only lever large enough to move
> it is observation dtype — which is a fidelity change, not a schedule. There is
> very little room in the memory schedule for a compiler to find.
>
> **No GPU, no builds, no timings.** Every line number is `[ARTIFACT]`, read off
> the file. Every performance attribution is `GUESS` or `MODELLED` and is labelled
> as such. Nothing here is a measurement of speed. §8 lists what I did not do.
>
> **The sibling `craftax.cu` obs-encode figure is not here.** The brief cites ~71%
> of peak write bandwidth from a sibling repo. I found **no transposed write
> pattern, no swizzle, and no lane→view-cell mapping** in this file. The one
> transposed *read* layout that exists (`[Y,X,Z,B]`, `kernels.cu:794`) is dead
> code — bound as `render_uint8_coalesced` (`bindings.cpp:1266`) and called from
> nowhere. Do not carry the 71% claim across; the two repos are not the same code.
> See §3.1 row 3, §3.2 entry 11, and §5.3.

Labels, used throughout, following `notes/16`:

| tag | meaning |
|-----|---------|
| `[ARTIFACT]` | read out of a named file at a named line; no inference |
| `MODELLED` | arithmetic on `[ARTIFACT]` inputs plus a published hardware constant |
| `GUESS` | my estimate; stated basis given, not measured |
| `[NOT-HERE]` | the measurement that would settle it cannot be run on this host |

Difference tags are the four from the brief and from `notes/16`:

| tag | meaning |
|-----|---------|
| **(i)** | cannot express — no representation in the compiler's IR; needs a new language construct |
| **(ii)** | can express, does not find — representable schedule, search/cost-model problem |
| **(iii)** | not yet implemented — known, bounded work |
| **(iv)** | does not exist for this architecture — targets hardware the card lacks |

**A note on (i).** Several entries below are not schedule choices at all: they
change what the program computes (uint8 observations, flat colours, fp16
traversal, `--use_fast_math`). A compiler is *correct* not to find these, because
the transformation is not semantics-preserving. I file them as **(i)** on the
narrow reading "no node in the IR of the given program corresponds to this, and
no rule may introduce one", and I say so explicitly on each row rather than
letting the tag carry it silently. This is the largest single category in the
inventory and it is the most important thing a codegen study should not
mis-attribute.

---

## 0. The artifact

```
netherite/mine.cu/          shallow clone, single commit ec2085e
  src/kernels.cu           3,245 lines, 116,131 B   22 __global__ kernels
  src/bindings.cpp         1,269 lines,  50,359 B
  minecu/__init__.py         479 lines              MineEnv.step dispatch
  minecu/config.py           101 lines
  scripts/plot_throughput.py 216 lines              the benchmark table, hardcoded
  scripts/benchmark_new_kernels.py 292 lines        references a module not in the repo
  setup.py                   32 lines               -O3 --use_fast_math
  docs/{architecture,benchmarks,environment,LLM}.md
```

`[ARTIFACT]` All line numbers below are into this tree as of commit `ec2085e`.

### 0.1 The docs are partly unreliable. Cite the source, not the prose.

`docs/architecture.md` is a different quality class from the code. It gives kernel
line numbers as `line ~85`, `line ~295`, `line ~370`, `line ~440` — the actual
`__global__` lines are 87, 1511, 1687, 1760. It shows a `PYBIND11_MODULE` body
calling `&launch_render` (`docs/architecture.md:77-85`); the real module is
`bindings.cpp:985` and the real wrapper is `py_render` (`bindings.cpp:215`). It
lists "Voxel-based collision detection" and "Block placement action" as TODOs in
the README (`README.md:72-73`) when `raycast_interact_kernel` already places blocks
(`kernels.cu:1666-1670`).

So the optimisation list in `docs/architecture.md:34-38` and repeated at
`docs/LLM.md:35` — *"AABB early termination, loop unrolling, `__ldg()`, block_size=128"* —
must be verified against the source rather than quoted. §3 entries 13, 15, 16, 17
and 36 do exactly that, and three of the five do not survive intact.

---

## 1. Which code path produced the 49.6M figure

This matters more than any single technique, because the headline number belongs
to one specific path and the interesting kernels are not on it.

`docs/benchmarks.md:51` gives the memory table for the peak row:

> | 32K batch, 16 world, 32px | 128 MB | 384 MB | ~800 MB |

- Voxels: `32768 × 16³ = 134,217,728 B` = **128 MiB** ✓ (they are writing MiB and
  calling it MB)
- Obs: `32768 × 32 × 32 × 3 × ?` = 100,663,296 B = **96 MiB** if uint8,
  **384 MiB** if float32.

`[MODELLED]` The table says 384. **The benchmark was run with float32
observations**, which means `obs_dtype="float32"` — the default at
`minecu/__init__.py:164` — and therefore `_C.step` (`__init__.py:417`) →
`launch_step` (`bindings.cpp:462`) → `launch_step_kernels`
(`kernels.cu:2436-2495`) with `render_kernel` (`kernels.cu:87`), the fp32 AABB +
`#pragma unroll 4` + `__ldg` + 128-thread kernel.

This is corroborated by `scripts/plot_throughput.py:13`, whose data block is
headed: `# Optimized kernel: AABB early termination + loop unrolling + __ldg + block_size=128`.
That comment names `render_kernel` and nothing else.

`[ARTIFACT]` **Consequence: `render_kernel_uint8_prebasis`,
`render_kernel_uint8_smem`, `render_kernel_uint8_coalesced`,
`render_kernel_uint8_fp16`, `render_kernel_uint8_minimal`,
`render_kernel_uint8_fast` and `precompute_camera_basis_kernel` are not on the
path the headline number came from.** The basis-hoist (§3.2 entry 26, §5) is real and
citable as a technique the human wrote, but its throughput contribution is
*unknown and unreported* — no table in this repo measures it. Any study that
attributes 49.6M to the prebasis or smem kernels is wrong.

---

## 2. Launch count per step — the hard fact

`grep -c "<<<" src/kernels.cu` = **71**. Breakdown `[ARTIFACT]`:

| group | launches | lines |
|-------|-----------|-------|
| one-shot per-kernel launchers | 15 | 2087-2310 |
| `launch_decode_actions` / `_v2` / `_raycast_interact` | 3 | 2394, 2413, 2429 |
| `launch_step_kernels` (fp32, **graph-captured**) | **6** | 2458-2494 |
| `launch_step_uint8` | 6 | 2615-2651 |
| `launch_step_uint8_fast` | 6 | 2677-2710 |
| `launch_step_uint8_minimal` | 6 | 2737-2773 |
| `launch_step_uint8_fp16` | 6 | 2797-2833 |
| `launch_step_uint8_prebasis_kernels` (**graph**) | **7** | 2864-2904 |
| `launch_step_uint8_prebasis_fast_kernels` (**graph**, the `_full` path) | **9** | 2938-2990 |
| `launch_step_uint8_smem` | 7 | 3201-3242 |

**Per step, in CUDA: 6 kernels (fp32 path), 7 (prebasis / smem), or 9
(`uint8_prebasis_full`).** Three of the seven step functions are wrapped in a
CUDA graph (§3.2 entry 25), so the *host-side* launch count is 1 for those and 6-9
for the rest.

Plus, on the Python side, per `MineEnv.step` call `[ARTIFACT]`:

- non-`_full` paths: `self.step_count >= self.episode_length` (`__init__.py:364`),
  `step_count += 1` (`:435`), `masked_fill_` (`:436`) = **3 extra torch kernels**
- `self.do_reset.any()` (`:356`, `:439`) = **1 torch kernel plus a
  device-to-host synchronisation**, unconditionally, every step
- `self.do_reset.clone()` (`:360`, `:444`) = 1 torch kernel **plus a fresh
  allocation every step**
- on reset, `generate_world` + `place_tree` + `place_house` (`:440-442`) = 3
  more CUDA launches, unguarded full-`B × W³` rewrites (§3.2 entry 31)

**So the real per-step cost is 6 CUDA launches + 5 torch ops + 1 forced host
sync, or 9 + 4 + 1.** There is **no megakernel fusion anywhere in the file** — the
five narrow kernels are never merged.

Why that is not the lever, quantitatively. At the peak config the render launch is
`(32768×32×32 + 127)/128 = 262,144` blocks of 128 threads = **33,554,432
threads**; the four narrow kernels are `ceil(32768/256) = 128` blocks = **32,768
threads**. `[MODELLED]` **A 1024:1 thread-count imbalance**, and the five narrow
kernels are serialised against a render kernel that is 65% DRAM-bound. Fusing them
would save at most 4-8 launches × ~2-4 µs = **8-32 µs of a 660 µs step, i.e.
1-5%**. The megakernel-fusion lever is real and it is *small*. `GUESS` (launch
latency figure is from experience, not measured here).

The bigger cost at that same line is not the launches. `kernels.cu:2547` puts a
`cudaStreamSynchronize(stream)` *inside* `launch_step` on the captured path
(`kernels.cu:2544-2547`). In an RL loop the next step's actions depend on the
policy forward pass over the current observation, so a per-step device sync is a
hard pipeline barrier that serialises CPU policy work against GPU step work. The
`_prebasis` and `_prebasis_full` paths do **not** do this — `kernels.cu:3034` and
`:3124` call bare `cudaGraphLaunch` with no sync. This is one of the few places
where the two paths differ in kind rather than degree. `GUESS` on the magnitude;
`[NOT-HERE]` to settle it: delete `kernels.cu:2547`, re-run the
`docs/benchmarks.md:60-81` snippet, compare. That is a one-line diff.

---

## 3. The inventory

**34 rows: 7 negative findings (§3.1) + 27 techniques (§3.2), of which 3 are
written-but-never-shipped.** §3.1 comes first because "not present" is the answer
to several of the brief's specific questions and burying it would be dishonest.

### 3.1 What is not in the file

| # | technique searched for | verdict | evidence |
|---|----------------------|---------|----------|
| 1 | `__shfl_*`, `__ballot`, `__popc`, `__any`/`__all`, `__activemask`, `__syncwarp`, `__match_any_sync` | **absent — 0 hits in 3,245 lines** | `grep` over `src/kernels.cu` |
| 2 | warp-aggregated atomics | **absent — no atomics of any kind** | `grep "atomicAdd\|atomicCAS\|atomicExch\|atomicMax\|atomicMin"` → 0 |
| 3 | one-warp-per-env encoder with lane→view-cell mapping | **absent.** The env-to-thread mapping takes exactly three forms. (a) **Flat linear**, `idx = blockIdx.x*blockDim.x+threadIdx.x` then `b = idx/(H*W)` — five kernels, `kernels.cu:99, :312, :997, :1139, :1281`. (b) **Env-per-block 2-D grid**, `b = blockIdx.y` — two kernels, `kernels.cu:584` and `:851`. (c) **Warp-spans-envs**, `b = blockIdx.y*blockDim.x+threadIdx.x` — one kernel, `kernels.cu:722`, **dead**. **In every case the mapping is expressed in the launch configuration and the index decomposition, never in the algorithm**: no kernel's body contains a notion of a warp, a lane, or a view cell. | cited per form |
| 4 | `__launch_bounds__` / `-maxrregcount` / register pinning to an occupancy target | **absent**; `setup.py:17-20` passes only `-O3 --use_fast_math`. Occupancy is whatever ptxas picks | `grep "__launch_bounds__"` → 0 |
| 5 | per-thread stack / local-memory scratch moved to a global arena (the sibling project's 132 KB → 16 KB move) | **absent, and the question does not apply.** There are exactly **two** `__shared__` declarations in 3,245 lines: `kernels.cu:581` (56 B of camera basis) and `kernels.cu:846` (the 4 KiB voxel grid, sized `64 + ws3` at `:2205`). No local arrays, no `__local__`, no spill buffers, no arena. All state is preallocated global tensors (`minecu/__init__.py:214-256`). | `grep "__shared__"` → 2 hits |
| 6 | `cp.async` / `__pipeline_memcpy_async` | **claimed in a comment, not implemented.** `kernels.cu:831` says "Uses cp.async for async global->shared copy"; the code at `kernels.cu:868-880` is `int4` load + `__syncthreads()` | `grep "cp.async\|__pipeline\|memcpy_async"` → only `kernels.cu:6` and `kernels.cu:831` |
| 7 | sort / compaction / prefix scan for divergence | **absent.** Divergence is handled by doing nothing: each thread runs its own variable-length DDA and leaves via `goto` | `grep "sort\|compact\|scan\|prefix\|thrust\|cub::"` → 0; see `kernels.cu:225-255` |

Rows 1-4 and 6-7 are classification answers as well as absences. If a study
observes a gap in warp-level cooperation or occupancy-regime tuning **on this
workload**, the class is **(iv) — does not exist for this architecture/workload**:
there is no warp-level code to beat, no register pressure to relieve, and no
scatter/gather to compact. `raycast_interact_kernel` is the only irregular-access
kernel in the file and it is 32,768 threads of work.

### 3.2 The techniques that are present

Format: technique | file:line | mechanism | class | why that class.

#### Memory layout and index arithmetic

| # | technique | file:line | mechanism | class | why |
|---|-----------|-----------|-----------|-------|-----|
| 8 | Batch-major voxel layout makes the divergent scatter conflict-free | `kernels.cu:1655` `long long idx = i*wxyz + vy*wxz + vx*wz + vz;` (and the place-write at `:1668`) | The batch index `i` is multiplied by the whole world volume, so it is the **slowest-varying** dimension. Two lanes in a warp hold different `i`, therefore their write addresses differ by `wxyz ≥ 4096 B`. No two lanes can ever alias. | **(i)** | The property is statically checkable from the IR, but *acting* on it means mutating the program's memory layout. A schedule search operates on a fixed layout; nvcc compiles one function and cannot move a tensor's batch dimension. Needs a layout/tiling IR. **See §6 — this one is arguably not a compiler gap at all.** |
| 9 | Pixel-linear obs write, no swizzle | `kernels.cu:167` `((long long)b*H*W + py*W + px)*3`; identical at `:373, :634, :759, :916, :1049, :1191, :1342` | `idx` runs linearly over pixels, so `px` is consecutive in a warp. A warp writes 32×3 = **96 contiguous bytes** for uint8 or 384 B for fp32, and 96 is a multiple of the 32-B sector so warps stay sector-aligned. | **(ii)** | Representable, and this is the mapping a compiler would pick anyway. Honest answer: **the human did not beat the compiler here.** This is the default thread→element map. |
| 10 | 3-byte RGB interleave, three 1-byte stores | `kernels.cu:692-694` `output[out_idx+0..2] = BLOCK_COLORS_U8[hit_block][0..2]` | No packed 24-bit store. PTX has `st.u8/u16/u32/u64/v2/v4` and no 24-bit form, so the epilogue is 3 separate byte stores per thread. | **(i)** | No node in the IR expresses a 3-byte store. Small (bandwidth is unaffected; only issue slots), but a true representational floor. |
| 11 | `[Y,X,Z,B]` transposed read layout, warp spans envs | `kernels.cu:722` `int b = blockIdx.y*blockDim.x + threadIdx.x;`; `:794` `int idx = voxel_y*ws*ws*batch_size + voxel_x*ws*batch_size + voxel_z*batch_size + b;` | Puts the env index in the **fastest** dimension so a warp reading the same voxel of 32 envs gets 32 contiguous bytes. Classic AoS→SoA transpose for the gather. | **(i)**, and **DEAD** | Layout is a program-level decision, not a schedule. But: the layout is **never produced** — `render_uint8_coalesced` is bound at `bindings.cpp:1266` and called from nowhere (`grep` over `minecu/`, `examples/`, `scripts/`). And it is a **predicted net loss**: it makes the *write* maximally scattered, `out_idx = (b*H*W + py*W + px)*3` at `:759` now has a 3072-B stride across the 32 lanes of a warp, and the write is the dominant term (§4). `GUESS`; test = A/B at the peak config. |
| 12 | Env-per-block 2D grid, `blockIdx.y = env` | `kernels.cu:584` `int b = blockIdx.y;`; launched at `kernels.cu:2182` `dim3 grid(blocks_per_batch, batch_size)` and `:2860`, `:2934`, `:3192` | Reshapes the grid so every thread in a block shares one env, which is what makes the shared-memory basis (§3.2 entry 26) possible. Also bounds `grid.y ≤ 65535`, so batch must stay under that. | **(i)** | Launch geometry is a 3D-grid property the source author chose; nvcc sees a 1-D `blockIdx.x` and cannot re-derive that a specific *semantic* index (the env) is block-uniform. Needs a grid-shape-aware scheduler. |
| 13 | 64-bit index widening throughout | `kernels.cu:167, 230, 420, 634, 794, 916, 1087, 1191, 1229, 1342, 1393, 1655, 1668, 1802, 1854, 1897, 1998, 2058, 2066` (`grep -c "long long"` = 35) | Every voxel and obs index computed in `long long`. | **(i)** | A compiler can only ever *narrow* an index, never widen one. At the peak config this is unnecessary: `b*4096` maxes at 134 M and `b*1024*3` at 100 M, both inside int32. It becomes necessary at `world=48` (`32768 × 110592 = 3.6e9 > 2³¹`), so the widening is defensive. **This is a case where the hand-written code is likely *slower* than compiler output** — 2× the address registers, 2× the IMAD work, in the hottest kernel. Honest, and it cuts against the "hand-written wins" framing. |
| 14 | `basis` row stride is 56 B, not 16-B aligned | `kernels.cu:513` (`float* basis, // [B, 14]`), `:537` `int idx = b * 14;`, writes `:540-560` (13 of 14 slots; slot 13 is documented "padding" at `:509` and never written) | The row is 14 floats = **56 bytes**. The author padded to 14 — clearly intending 16-byte alignment — and stopped one short. Rows are therefore not 16-B aligned, and each `__ldg(&basis[bidx+k])` at `kernels.cu:731-743` straddles a sector boundary for every odd `b`. | **(ii)** | Trivially expressible: the fix is a stride of 16. But the stride is a *tensor shape* chosen in Python (`minecu/__init__.py:239` allocates `(B, 14)`), and a compiler cannot change a tensor's shape. Only material in the dead coalesced kernel (`:731-743`, 13 scalar loads per thread) and mildly in the 14-thread shared load (`:594-596`). Small; flagged because it is a real defect in a load-bearing struct. |
| 15 | Struct-of-arrays agent state | `kernels.cu:1512-1516` (`positions`, `velocities`, `yaws`, `pitches`, `on_ground` as five separate `[B]`/`[B,3]` pointers); allocated `minecu/__init__.py:224-228` | Each field is its own tensor, so a warp reads `positions[i*3+0]` at a 12-B stride instead of a 40-B struct stride. | **(ii)** | Standard, and **forced**: these are five separate PyTorch tensors by construction, so a `struct Agent` AoS is not expressible across the binding boundary. The human got the SoA benefit for free from the framework. Worth recording as a *free* win, not a skill. |
| 16 | uint8 observations + precomputed uint8 colour LUT | `kernels.cu:479-493` `__constant__ uint8_t BLOCK_COLORS_U8[13][3]`, `:495` `SKY_COLOR_U8`; epilogue `:691-699` is a direct LUT store with **no float math at all** | Removes the fog/shade/fma epilogue (`:285-287`, `:458-460`, `:1434-1436`) and cuts obs traffic 4× (fp32 → uint8). | **(i)** | Changes the program's output. Not a schedule; a fidelity reduction. **This is the single largest contributor to the headline number and it is not a compiler question.** §5.2. |
| 17 | Camera state packed into a 5-float `[B,5]` camera buffer | `kernels.cu:2320-2335` `update_cameras_kernel` writes `cameras[i*5+0..4]` from `positions`/`yaws`/`pitches` | One 20-B contiguous record per env, written by one thread, replacing three separate scatters. | **(ii)** | Expressible and cheap. Note this kernel exists only because the render kernel wants a single packed read; it is a 1-launch tax for a 5-load saving per pixel. |

#### Coalescing and bounds

| # | technique | file:line | mechanism | class | why |
|---|-----------|-----------|-----------|-------|-----|
| 18 | AABB entry-side early-out | `kernels.cu:51-79` `ray_intersects_aabb`; test at `:170` `if (!hits_world \|\| t_entry > view_distance)` | Slab test; miss or beyond view distance → write sky and return, skipping the whole DDA. | **(ii), but inert for every configuration this repo ships** | Representable and findable. **However the camera sits inside the box.** Spawn is `(ws/2, ground_height+2, ws/2)` and the eye adds 1.6 (`kernels.cu:118`, `:541`); physics clamps x,z to `[1, ws-1]` (`:1577-1578`) and gravity pins y to a floor at `world_y/4` (`:1566-1573`), so the eye cannot leave in x or z and is pulled back in y. At the default `ground_height=8` and every world size benchmarked (16/32/48), eye_y ∈ (4, 16) ⊂ the box. Inside the box, `tmin < 0 < tmax`, so `hits_world` is always true and `t_entry > view_distance` is always false. `[MODELLED]` from the spawn and physics code. *Counterexample I am obliged to name:* a user who sets `ground_height` near `world_size` would spawn the eye above the box, and then this early-out **would** fire — so the technique is not dead in general, it is dead in the shipped regime. This is the headline "optimisation" in `docs/architecture.md:35` and `docs/LLM.md:35`. |
| 19 | AABB exit-side test inside the DDA | `kernels.cu:236` `} else if (t > t_exit) { goto render_done; }`; same at `:426`, `:1399` | Once the ray has left the box, stop stepping instead of grinding to `max_steps`. | **(ii)** | Expressible, and genuinely live: this is the part of the AABB that pays. Test: the DDA loop already bounds-checks every axis, so the exit test is a *distance* test the bounds check cannot make. |
| 20 | `__ldg` read-only path, 48 sites | `kernels.cu:110-114`, `:321-325`, `:1006-1010`, `:1149-1153`, `:1290-1294`, `:731-743`; 48 total | Forces the non-coherent `LDG.E.CONSTANT` read path. | **(ii) — and almost certainly free** | nvcc emits `LDG.CONSTANT` automatically for a `const T* __restrict__` parameter that is never written in the kernel, which every one of these is (`kernels.cu:88-90` etc.). **The explicit `__ldg` is documentation, not advantage.** Recorded so the study does not credit it. |
| 21 | `__restrict__` on 81 parameters | `kernels.cu:88-90` and 81 sites total | Without it nvcc cannot prove non-aliasing, so it cannot use the read-only path, cannot keep values in registers across a store, and cannot reorder loads. | **(ii), and load-bearing** | This one is real and is the precondition for entry 20. It is also the single cheapest thing a code generator must emit. |

#### Shared memory, register budget, occupancy

| # | technique | file:line | mechanism | class | why |
|---|-----------|-----------|-----------|-------|-----|
| 22 | Whole-voxel-grid residency in dynamic shared memory, hand-partitioned with an alignment pad, `int4`-vectorised load | `kernels.cu:846` `extern __shared__ int8_t shared_mem[];`, `:847-848` `float* s_basis = (float*)shared_mem; int8_t* s_voxels = shared_mem + 64;` (comment: "Align to 64 bytes (14 floats = 56 bytes, pad to 64)"), `:868-879` `const int4* src = ...; dst[idx] = src[idx];` + `__syncthreads()`; size computed at launch, `kernels.cu:2205` `int shared_mem_size = 64 + ws3;` and `:3198` `int shared_mem_size = 64 + wxyz;` | Loads the entire 4 KiB grid for one env into smem so every DDA lookup is an LDS instead of an L1/DRAM gather. Two manual sub-allocations in one dynamic arena, with a 64-B pad to keep `s_voxels` 16-B aligned for the `int4` cast. | **(i)** for the carveout, **(ii)** for the `int4` cast | The carveout: the size is a *runtime* launch parameter computed in host code from `world_size`; the kernel body only sees `extern __shared__`. There is no IR node for "sub-allocate a dynamic smem arena with an alignment invariant" and nvcc cannot infer a size from a `__syncthreads()`-protected region. The `int4` cast is the opposite case — contiguity and alignment are in the IR, so a good vectoriser could find it; the author had to cast through `long long` to keep `b*ws3` in range. |
| 23 | **This smem kernel is predicted to be a loss at the peak config** | `kernels.cu:2196-2200`, `:3190-3192` `blocks_per_batch = (H*W + 127)/128` | At 32×32 there are **8 blocks per env**. Each block loads the full 4 KiB grid (`:868-879`), so the grid is loaded **8× per env** = 32 KiB of L2 traffic where the prebasis path would load it 0× (L1 already holds a 4 KiB working set trivially — 4 KiB out of 128 KiB L1). | **(ii)** | Representable and findable; a cost model with a line count would have predicted the loss. **No table in this repo measures it.** `[NOT-HERE]` to settle: A/B `obs_dtype="uint8_prebasis"` against `obs_dtype="uint8_smem"` at `batch=32768, world=16, 32×32`. The harness already exists (`docs/benchmarks.md:60-81`). `GUESS`: I predict smem loses by 1.5-3×. |
| 24 | Register budget / occupancy pinning | — | **Absent.** See §3.1 row 4. No `__launch_bounds__`, no `-maxrregcount`, no `cudaFuncSetAttribute` for a preferred shared-memory carveout, no `cudaOccupancyMaxPotentialBlockSize` call. | **(iv)** | There is no register pressure to manage: the hot kernel's live set is ~20 scalars (`t_max_x/y/z`, `t_delta_x/y/z`, `step_x/y/z`, `voxel_x/y/z`, `hit_block`, `step`, plus a few), well under 255, and there is no local memory at all. The register/occupancy dimension of a codegen study **does not exist on this workload**. Class (iv), and the reason is the workload, not the card. |

#### Launch overhead and schedule

| # | technique | file:line | mechanism | class | why |
|---|-----------|-----------|-----------|-------|-----|
| 25 | Whole-step CUDA-graph capture with pointer+config invalidation | `kernels.cu:2338-2376` `struct StepGraphState` (`cudaGraph_t`, `cudaGraphExec_t`, `WARMUP_THRESHOLD = 10`, cached `last_batch_size`/`last_width`/`last_height`/`last_voxels`/`last_positions`/`last_obs_buffer`); capture at `:2550-2572`, `:3037-3058`, `:3127-3149`; three independent graph states at `:2376`, `:2379`, `:2382` | 6 or 9 launches collapse to one `cudaGraphLaunch` after 10 warmup iterations, and the graph is destroyed and re-captured if any tensor address or the batch/width/height changes (`:2526-2535`). | **(ii)** | Representable in the strongest possible sense: the CUDA runtime ships the API. A framework-level pass would do this. **Value is bounded at 1-5%** (§2) because the render kernel dominates. **Fragility worth flagging:** on the non-`_full` paths `__init__.py:364` allocates a *new* `do_reset` tensor every step and `:444` clones a new one; if the caching allocator ever hands back a different address, the graph is torn down and re-captured on that step. `[NOT-HERE]` to settle: run 1e5 steps with `episode_length=1` (forces a reset every step) and watch whether the graph is thrashing. |
| 26 | `precompute_camera_basis_kernel` — split the trig out of the pixel loop | `kernels.cu:511-561` (one thread per env; `__sincosf` at `:530-531`; 13 outputs at `:540-560`) consumed by `kernels.cu:568-700` (`:600-616` read basis from shared, **no trig at all**); wired at `:2896-2904` and `:2982-2990` | One extra 32,768-thread kernel replaces per-pixel `cosf/sinf/cosf/sinf/tanf/2 divides` (`kernels.cu:331-349`) with zero arithmetic. | **(i) / (iii)** — **genuinely ambiguous, see §7.1** | It is a restructuring across a launch boundary: the source program went from 1 render kernel to 2. nvcc's unit of compilation is the function; the "program" is a launch sequence in host C, and no shipped CUDA compiler builds a cross-kernel dependency graph with costs. **This is my pick for §5.** |

#### Branch divergence and game logic

| # | technique | file:line | mechanism | class | why |
|---|-----------|-----------|-----------|-------|-----|
| 27 | Divergence handled by `goto` out of a serial per-thread DDA; no compaction, no sorting, no predication | `kernels.cu:225-255` `DDA_STEP()` macro with `goto render_done`; same pattern at `:415-441`, `:664-679`, `:789-805`, `:946-961`, `:1082-1097`, `:1224-1239`, `:1384-1409` | Each thread walks its own variable-length ray. Lanes in a warp exit at different iterations; the warp pays `max(lane steps)`. The `#define`+`goto` structure exists to let `#pragma unroll 4` fire (a `return` inside the loop would block it). | **(iii)** for compaction/sorting — and leaning **(iv)** for this workload | Compaction and sorting are representable and are known work; nobody has built them for ray traversal. But the natural coherence here (adjacent lanes = adjacent pixels of the *same* camera, so they walk the same 4 KiB) means there is little to compact. `[NOT-HERE]` to settle: measure the ratio `max(lane DDA steps) / mean(lane DDA steps)` per warp; if it is near 1, sorting is worthless and the honest class is (iv). |
| 28 | Warp voting — tried, in a module that is not in the repo | `scripts/benchmark_new_kernels.py:8` (`"warp_vote: Warp voting for uniform execution paths"`), `:192` `exp.render_warp_vote`, `:282-286` (`warp_benefit = "beneficial" if warp_speedup > 1.0 else "not beneficial"` and `"Warp voting overhead may exceed benefit when rays are highly divergent"`) | The harness benchmarks `baseline` / `warp_vote` / `early_term` / `combined` from `minecu._C_experimental` (`:25`) — **an extension whose source is not in this tree** (`git ls-files` confirms; only `src/kernels.cu` and `src/bindings.cpp`). The harness's own analysis text is written to cope with warp voting *losing*. | **(iii)** for the technique, **(iv)** for this workload | A warp-uniform DDA is representable (`__ballot`/`__popc`/`__shfl` all exist in the ISA) and is bounded, known work. But the harness's own conclusion — that divergent rays make the vote overhead exceed the benefit — is the domain fact that makes it a dead end here. **No result table for it is committed, so I cannot tell which way it went.** `[NOT-HERE]`. |
| 29 | Ground-plane analytic early-terminate | `kernels.cu:1386-1389` `if (voxel_y < gh) { hit_block = (voxel_y == gh-1) ? GRASS : DIRT; goto render_done_fast; }` | A downward ray terminates at the analytic terrain plane with **no memory access at all**, instead of a DDA gather. Would be a big win on flat ground. | **(i)**, and **DEAD** | The compiler would have to know that all voxels below `gh` are solid — a value-range analysis of the whole voxel array, which no cost model does and which the program never states. **Unreachable:** `"uint8_fast"` is not in the accepted `obs_dtype` list at `minecu/__init__.py:167-168`, so `_C.step_uint8_fast` is never called (`grep uint8_fast` over `minecu/`, `examples/`, `scripts/` → 0 hits). The kernel is bound (`bindings.cpp:759`) and dead. This is the technique I most wish had a number for. |
| 30 | `#pragma unroll 4` on a serial dependency chain | `kernels.cu:258` (and `:413, :681, :807, :963, :1099, :1241, :1411`); rationale in the comment at `:257` "Unroll 4x per iteration for better instruction-level parallelism" | The four copies are **strictly sequential**: `t_max_x` computed in copy *n* is the input to copy *n+1*, and each copy contains a data-dependent `goto`. | **(ii)** — and the stated rationale is **wrong** | There is no ILP in a serial chain to expose. The unroll's real (small) effect is amortising the loop counter and letting nvcc drop some redundant bounds compares. nvcc would unroll this loop on its own under `-O3`. **Recorded so the study does not credit `#pragma unroll` here**, since it is named in `docs/architecture.md:36`. |
| 31 | Masked world regeneration, written and then not used | `kernels.cu:1989-2028` `generate_world_kernel_masked` (`:2010` `if (!do_reset[b]) return;`) and `:2031-2069` `place_tree_kernel_masked` (`:2046`) vs the live `generate_world_kernel` `:1811-1827` and the Python dispatch at `minecu/__init__.py:439-442` | The masked versions skip all `B × W³` voxel writes for the 99.9% of envs that did not reset. The live path instead branches on the **host** and runs the *unmasked* full-grid rewrite. | **(ii)** | The masked form is a plain predicated write — representable and findable. Note both masked kernels have **no `<<<` launch anywhere in the file**; they are dead. The live design trades a 1-warp-wide predicate for a host sync (`__init__.py:439` `self.do_reset.any()`). **The human wrote the better kernel and then did not use it**, which is worth knowing: the dead code is not a failed experiment, it is an unimplemented win. |

#### Precision

| # | technique | file:line | mechanism | class | why |
|---|-----------|-----------|-----------|-------|-----|
| 32 | `--use_fast_math` | `setup.py:19` `'nvcc': ["-O3", "--use_fast_math"]` | One flag sets `--ftz=true --prec-div=false --prec-sqrt=false --fmad=true`, which turns `sinf/cosf` into `__sinf/__cosf`, `tanf` into `__tanf`, every `/` into a reciprocal-multiply, and `sqrtf` into `sqrt.approx.f32`, and flushes denormals. | **(i)** | A build flag, not IR. No compiler picks it by default and no cost model will ever propose reducing numerical accuracy. **This single line owns the whole precision budget**, and it means the explicit intrinsics below are largely redundant with it. |
| 33 | `__sincosf` | `kernels.cu:530-531` | One instruction sequence for both sine and cosine of a camera angle. | **(i)**, and **subsumed by 32** | Under `--use_fast_math` `sincosf` already lowers to `__sincosf`. The explicit spelling is redundant. |
| 34 | `rsqrtf` in five kernels vs `sqrtf` + three divides in three | `rsqrtf`: `:626, :753, :911, :1043, :1185` (`prebasis`, `coalesced`, `smem`, `minimal`, `fp16`). `sqrtf` + `/=`: `:151-154` (`render_kernel`, **the peak path**), `:358-361` (`render_kernel_uint8`), `:1327-1330` (`render_kernel_uint8_fast`) | `rsqrtf` is one MUFU instruction (plus range handling); `sqrtf` + 3 divides is 4 MUFU + 4 mul/fma. | **(i)**, and **subsumed by 32** | Again, `--use_fast_math` converts the slow forms. And note the *inconsistency*: three of the eight render kernels keep the `sqrtf`+divide form, they are the three oldest (`:87`, `:300`, `:1268`), and **one of them is the one the headline number came from**. There is no bit-exactness requirement anywhere in this file — no `__fmaf_rn`, no `__frcp_rn`, no `-fmad=false`. **This is an RL environment where fidelity is traded for throughput by design**, which is the correct thing to record and the wrong thing to attribute to a compiler gap. |

---

## 4. Ranked shortlist: the three or four techniques that would matter most to a compiler on this workload

Ordering is by *value to a code generator trying to match this file*, not by how
clever the technique is. All values are `GUESS` unless marked. The basis is given
in every case and, where possible, a specific experiment that would settle it.

### 4.1 Rank 1 — Observation dtype and the elimination of the float epilogue. Class **(i)**.

`kernels.cu:479-495` + `:691-699`. float32 obs → uint8 obs; the fog, face-shading
and fog-blend fma chain (`:285-287`) replaced by a `__constant__` byte LUT store.

**Worth:** 4× on the dominant memory term. `MODELLED` from §5.1's arithmetic: obs
write goes 402.65 MB → 100.66 MB per vector step, and the DRAM floor for the write
goes **430 µs → 108 µs of a 660 µs step**. Corroborated by the repo's own table:
`docs/benchmarks.md:34-35` shows 32px→64px costs **3.6×** (49.63M → 13.74M) for
exactly 4× the pixels, i.e. obs traffic *is* the cost.

**Basis for the estimate:** the memory table at `docs/benchmarks.md:51` and the
bandwidth arithmetic in §5.1. Not measured. `[NOT-HERE]` to settle: run
`obs_dtype="uint8_minimal"` at the peak config and compare to the documented 49.63M.

**Why rank 1 for a *compiler* study:** it is the largest number in the repository
and **it is not a schedule.** A generated kernel that matched `mine.cu`'s
schedule exactly, in fp32, would still be 4× off the shipped uint8 path. Any
comparison that holds dtype fixed measures almost nothing, and any comparison that
lets dtype float is measuring fidelity, not codegen.

### 4.2 Rank 2 — Camera-basis hoist via kernel split. Class **(i)/(iii)**, ambiguous.

`kernels.cu:511-561` + `:568-700`, wired at `:2896-2904`.

**Worth: ~10% of the render kernel, hence ~8% end-to-end. `GUESS`, and this is my
least confident number.** Basis: counting the eliminated per-pixel work against
the retained work, all in `render_kernel_uint8` (`:300`), the flat-mapped original
of the prebasis kernel:

| eliminated by the hoist | line | count |
|---|---|---|
| `cosf/sinf/cosf/sinf` | `:331-334` | 4 MUFU + ~8 mul/add |
| `forward`/`right`/`up` construction | `:336-345` | 4 mul + 2 negate |
| `tanf` + fov-radian multiply + aspect divide | `:347-349` | 1 MUFU + 1 mul + 1 RCP + 1 mul |

versus retained: per-pixel ray direction (`:354-356`, 6 fma/mul) and
`rsqrtf` normalisation (`:358-361`, 1 MUFU + 3 mul), plus the DDA (`:415-436`,
~14-18 instructions per step, 3-8 steps typical). **That puts the removed setup at
roughly 20 of ~105 instructions per pixel, i.e. ~19% by instruction count — but
the render kernel is 65% DRAM-bound (§5.1), so ALU savings convert to wall-clock
at a discount. My ~10% of *kernel time* is the discounted figure, and it is a
guess.** Note the hoist does **not** remove the per-pixel ray direction or its
normalisation, and note `--use_fast_math` (`setup.py:19`) already made the
eliminated trig 4 MUFU instructions rather than library calls.

**Better basis than my arithmetic, and it already exists in the repo:**
`scripts/benchmark_new_kernels.py:73-108` is a working per-kernel A/B harness with
`torch.cuda.synchronize()` on both sides. Adding a `prebasis` variant to its
`kernels` dict (`scripts/benchmark_new_kernels.py:190-195`) is a ~20-line
experiment that measures this directly. **Recommend this over my number.**

**Caveat that matters for the study:** the hoist is *not on the path the 49.6M came
from* (§1). Its contribution is unreported in this repo. Do not let the study
inherit the 49.6M number and attribute it to this technique.

### 4.3 Rank 3 — Batch-major layout making the irregular scatter conflict-free. Class **(i)**.

`kernels.cu:1655` and `:1668`.

**Worth: small in absolute terms.** `MODELLED`: `raycast_interact_kernel` is 32,768
threads against the render kernel's 33,554,432 — **0.1% of the step's threads**.
Even making it free buys little wall-clock.

**Why it still ranks 3:** it is the *only* place in the entire program where a
divergent store could alias, and the layout is the reason it does not. It is the
clean worked example of the "irregular scatter degrades to scalar" phenomenon the
brief asked about — **and the answer here is that the phenomenon never arises,
because the scatter key was made the slowest-varying dimension.** That is a
technique a compiler structurally cannot perform (§6) and it is worth having in
the checklist even at 0.1%, because a study that cannot find it in `mine.cu` has
not found the right artifact.

**Test:** permute the voxel layout to put the batch index *fastest* — i.e. the
`[Y,X,Z,B]` order that row 11 already implements — and A/B
`raycast_interact_kernel` in isolation. Expect a large regression, and expect it to
be the *largest* relative regression of anything in this file. Note this is the
one A/B in the note that **cannot** be run by only editing a kernel: it changes
the tensor shape, so `minecu/__init__.py:221` must change too.

### 4.4 Rank 4 — Whole-step CUDA-graph capture. Class **(ii)**.

`kernels.cu:2338-2376`, `:2544-2572`.

**Worth: 1-5%. `GUESS`**, basis in §2: 4-8 launches saved × ~2-4 µs against a
660 µs step. Ranked 4 not because it is small but because it is the one technique
here that a compiler-side effort could *actually* close today — the CUDA runtime
ships the capture API, and any framework targeting this workload can call it
without a new abstraction. That makes it the most actionable item in the list and
the least intellectually interesting one.

**Its real cost is the adjacent line, not the graph.** `kernels.cu:2547` puts a
`cudaStreamSynchronize` inside the captured path; `:3034` and `:3124` do not. See
§2. **I rank this adjacent defect above the graph capture itself.**

### 4.5 Considered and not ranked

- **Shared-memory voxel residency** (§3.2 entries 22-23). Predicted to *lose* at
  32px because of 8× redundant grid loads. Not a candidate for a study's
  "optimisation" list without a measurement.
- **`[Y,X,Z,B]` transposed layout** (entry 11). Dead code, and predicted to lose:
  it wrecks the write stride, which is the dominant term.
- **Warp voting / compaction / sorting** (§3.1 rows 1, 7; entry 28). No
  measurement exists, the harness's own text suggests the vote loses, and the
  natural ray coherence makes compaction near-useless. **If a study wants to
  demonstrate a warp-level opportunity, `mine.cu` will not provide one.**
- **`__ldg` and the unsigned-compare idiom** (entries 20, and the `(unsigned)x <
  (unsigned)ws` pattern at `:666, :791, :948, :1084, :1226`). Both are what nvcc
  already emits. Listing them as wins would be a category error.

---

## 5. §3 — The single most important observation

### 5.1 First, the constraint that reframes everything: at peak the render kernel is write-bound

`MODELLED`, all inputs `[ARTIFACT]`:

```
B = 32768 envs, H = W = 32, world = 16^3, obs float32      [docs/benchmarks.md:7-10, :51]
throughput                       49.63e6 env-steps/s        [docs/benchmarks.md:7]
  => 1514.6 vector-steps/s  =>  660.2 us per vector-step
obs write per vector-step  = 32768 * 32 * 32 * 3 * 4 = 402,653,184 B = 384 MiB
obs write bandwidth         = 1514.6 * 402.65e6 = 610 GB/s
RTX 3090 spec DRAM bandwidth                            = 936 GB/s
  => the write alone is 65.2% of spec
  => the write's DRAM floor is 402.65e6/936e9 = 430 us of the 660 us step
voxel working set           = 32768 * 4096 B = 128 MiB = 22.4x the 3090's 6 MB L2
voxel gathers, worst case   = 33,554,432 px * 32 steps = 1.07 G per vector-step
  => 1.62 T gathers/s at 660 us
L1 lane-return capability   ~= 170 SM * 32 lanes * 1.7 GHz = 9.2 T/s
  => the gather is at ~18% of L1 capability even at the absolute worst-case step count
```

Consistency check, all `MODELLED`: 430 µs of DRAM floor + ~100 µs of DDA/setup
arithmetic (my instruction count, `GUESS`) leaves ~130 µs for the five narrow
kernels, the launches, and host overhead. The model closes. **So: the render kernel
is bound by the float32 observation write, at roughly 65% of spec bandwidth, and
it is not bound by the gather.**

Two consequences, and they point in opposite directions from the usual expectation:

1. **There is essentially no coalescing headroom left in the write.** 610 of 936
   GB/s with a 3-byte interleave (`kernels.cu:692-694`, three 1-byte stores) and no
   swizzle. A compiler cannot find a better memory schedule for the *same* dtype,
   because there is no better one — you would have to write fewer bytes.
2. **A generated kernel could plausibly match this one.** If the hand-written
   render kernel is 65% DRAM-bound and its ALU work is ~100 µs against a 430 µs
   memory floor, then a compiler-generated kernel with an equal-or-worse schedule
   loses at most ~15%. **This is a case where the hand-written advantage may be
   near zero, and it should be tested rather than assumed.**

`[NOT-HERE]`: the decisive experiment is a null render kernel that writes 384 MiB
of constants and reads nothing, timed at the peak config. If it runs in ~430-460 µs,
the render kernel is memory-bound and §5.1's conclusion is measured rather than
modelled. That is a 20-line kernel and it is the highest-value experiment in this
note.

### 5.2 The rival candidate for "most important", and why it cannot be the separator

Observation dtype (§4.1) is by far the largest lever in the file. But it **cannot
serve as the "cannot express vs does not find" separator**, because it is not a
candidate gap at all: the compiler is *correct* not to convert a float32
observation buffer to uint8. A study that files this under "the compiler missed
it" has mis-framed the question — the human changed the program's output, and no
source-to-source compiler does semantic-changing transformations. Same for the
flat-colour epilogue (`:691-699`), the fp16 traversal (`:1209-1238`), and
`--use_fast_math` (`setup.py:19`). **These belong in the study's "not a compiler
problem" list, not its gap list** — which is why §6 exists.

### 5.3 The separator: the camera-basis hoist

> **Pick: hoisting per-environment camera trig out of the per-pixel inner loop by
> splitting one kernel into two (`precompute_camera_basis_kernel` at
> `kernels.cu:511-561` + `render_kernel_uint8_prebasis` at `:568-700`), together
> with the block-per-environment grid reshape at `kernels.cu:584`/`:2182` that
> makes the hoisted value block-uniform. Class: (i) or (iii) — see §7.1. I lean
> (i) on the *split*, and I concede the *effect* is (ii).**

**The technique, precisely.** In `render_kernel_uint8` (`:300`) and every other
flat-mapped render kernel, each of the 33,554,432 threads computes the full camera
basis: `cosf/sinf/cosf/sinf` (`:331-334`), 4 mul + 2 negate to build
`forward`/`right`/`up` (`:336-345`), `tanf` + fov radians + aspect divide
(`:347-349`). Every one of those 33.5 M computations depends only on
`cameras[b*5 + 3]` and `cameras[b*5 + 4]` — **two numbers that are uniform across
the 1024 pixels of an env and the 128 threads of a block.** The human's fix was not
a pragma: it was to add a kernel that computes the basis once per env
(`:511-561`, one thread per env, `:519`) into a `[B,14]` global buffer, and a
second render kernel that reads it — via 14 threads plus `__syncthreads()` into
`__shared__ float s_basis[14]` (`:581`, `:594-597`) — and does zero trig
(`:600-616`). The grid had to be reshaped to 2-D, `(blocks_per_batch, batch_size)`,
so that a block *is* an env (`:584`, `:2182`).

**Side A — this is (i), the compiler cannot express it.**

The unit of compilation for nvcc is the function. The "program" is a launch
sequence written in host C++ (`launch_step_uint8_prebasis_kernels`,
`kernels.cu:2836-2905`), and the compiler never builds a graph over it: it does
not know that `precompute_camera_basis_kernel` and `render_kernel_uint8_prebasis`
are related, that the first's output is the second's only input, that the first
costs 32,768 threads and the second 33,554,432, or that merging them would let the
per-env values be computed once per block rather than once per thread. Every piece
of that reasoning lives in host code, is invisible to `ptxas`, and the boundary is
semantic, not incidental: the two kernels are separately callable, separately
scheduled on separately sized grids. **To do this, a compiler needs a
device-call-graph IR with node costs and a legal node-merge/split rule** — and,
before that, a way to see that the producer is an *invariant hoist* of a consumer
inner loop rather than an unrelated kernel. There is no such thing in the CUDA
compiler. **The new construct needed is a cross-kernel schedule representation.**

*A caveat I am obliged to record, because it weakens my own Side A.* nvcc *is*
handed both kernels in one translation unit (`src/kernels.cu` compiles as a
whole), and the launcher that calls them in order
(`launch_step_uint8_prebasis_kernels`, `kernels.cu:2836-2905`) is in that same
file. So a sufficiently whole-translation-unit analysis *can* see both kernels and
the call order. What it still cannot see is the **cost** of each node, because
that is a launch-configuration property of the caller, and it has no
representation in which to write the *merged* schedule's grid shape. The
obstruction is therefore narrower than "the compiler never sees both kernels". It
is: **the compiler sees two kernels and no reason to believe merging them helps,
and no node in its IR for the schedule that results.** That still lands on (i) for
the construct, but it is a smaller claim than I first wanted to make.

**Side B — this is (ii)/(iii), and the compiler could find it.**

Three strong counterarguments, and the third is the one that hurts.

- *B1: it is a fusion, and fusions exist.* MLIR's `gpu-to-gpu` passes, and
  whole-graph GPU schedulers, operate over exactly this. The IR to express it
  (`gpu.launch_func` in a module-level `gpu.module`) demonstrably exists. So the
  abstraction is not missing — it is just not in nvcc. That is **bounded, known
  work: (iii)**.
- *B2: the schedule is trivially representable inside one kernel.* Nothing about
  the hoisting requires a second launch. Put the trig behind a
  `if (threadIdx.x == 0)` with a `__syncthreads()`, or simply let LICM hoist it —
  the values are loop-invariant with respect to the DDA loop and nvcc's loop
  invariant code motion handles exactly that shape. If a one-kernel version gets
  the same speed, then the win is **an ordinary LICM that a compiler performs
  every day**, and the honest class collapses to **(ii)**. *This is the crux and I
  do not think it is closable by reading.*
- *B3: the fast-math flag already destroys most of the win.* Under
  `--use_fast_math` (`setup.py:19`) `cosf`/`sinf` are `__cosf`/`__sinf` — single
  MUFU instructions. So the "eliminated" work is ~6 MUFU + ~10 mul/fma per pixel,
  not 6 library calls. My 10% estimate (§4.2) is consistent with that, and a
  10% win is small enough that nvcc's own scheduling noise could be comparable.

**My call.** I pick this technique, and I classify it **as (i) on the kernel
split, while recording that the *effect* is (ii) if B2 holds.** The honest
statement is: *the mechanism by which the human captured this win — restructuring
the program across a launch boundary — has no representation in the compiler's
IR, and a whole-graph optimiser would have to both build that representation and
discover a specific invariant-sinking rule for a nested, data-dependent loop. The
win itself, ~10%, is small enough that the within-kernel alternative may capture
most of it, in which case the gap was never (i) at all.* That is a defensible
answer and it is not a clean one; §7.1 gives the test.

**Why not the batch-major layout (§4.3) instead — and the reason it is actually the
more interesting finding.** Entry 8 looks like a textbook (i): no compiler ever
changes a program's memory layout, and putting the batch index in the slowest
dimension so that a divergent store cannot alias is exactly the kind of decision a
schedule search is structurally blind to. But it loses as the *separator*, for a
subtle and important reason: **there is no gap to explain.** A naive compiler
emitted a `[B,Y,X,Z]`-layout scatter would not have added any serialisation or
atomics either — the hazard-free-by-construction property is a property of the
*reference program*, not a transformation the compiler declined to make. So entry 8
is a case of *avoiding* a problem rather than *solving* one, and a study has no
way to observe a gap that the baseline never had. **The study's four-class taxonomy
has no bucket for "the human did not create a problem", and that absence is
itself the finding.** §6.

---

## 6. What is genuinely not a compiler problem here

Being precise about this boundary is worth more than another row in the table.

**6.1 The game logic, and there is not much of it.** `generate_world_kernel`
(`kernels.cu:1793-1828`) is 15 lines: bedrock at y=0, dirt below
`ground_height-1`, grass at `ground_height-1`, air above. `place_tree_kernel`
(`:1831-1865`) writes 4 voxels. `place_house_kernel` (`:1868-1946`) is a
5×4×4 box with a door. `physics_kernel` (`:1511-1584`, 74 lines) is: rotate,
`cosf/sinf` of yaw, Euler-integrate x/z, apply sprint, jump-if-on-ground, gravity,
clamp to a plane at `world_y/4`, clamp x/z to `[1, ws-1]`. Collision is a flat
plane (`:1566-1573`) — the README lists voxel collision as a TODO
(`README.md:72`) and the code agrees. **None of this is a scheduling problem. The
difficulty is the algorithm, and the algorithm is trivial.**

**6.2 There is no RNG.** `generate_world_kernel` accepts `unsigned int seed`
(`kernels.cu:1800`) and **never reads it** — `grep "seed"` in the file returns only
the parameter at `:1800` and its two forwardings at `:2282`/`:2289`. World
generation is fully deterministic and analytic; every env gets an identical world
(`_generate_world` calls it once over the whole batch,
`minecu/__init__.py:260`). **So the entire class of "how do you get cheap
divergent random access" has no instance here.** If the study's checklist has a
row for RNG-driven irregularity, `mine.cu` cannot fill it.

**6.3 Five kernels are ~1% of the threads.** `decode_actions_kernel`
(`:1459-1504`), `physics_kernel`, `raycast_interact_kernel`, `reset_kernel`,
`update_cameras_kernel` and `episode_*` are all one-thread-per-env: 32,768 threads
each, against the render kernel's 33,554,432. **Their schedules, their
vectorisation, and their fusion are numerically irrelevant** to the 660 µs step
(§2). A study that profiles the whole step and reports "kernel X is 2% of time"
has profiled the wrong 2%.

**6.4 Fidelity reductions are not compiler gaps.** uint8 obs (`:479-495`), flat
colours (`:691-699`), no fog (contrast `:285-287` with `:691`), fp16 traversal
(`:1209-1238`), `--use_fast_math` (`setup.py:19`). Every one changes what the
program computes. **The compiler is right not to find them, and a study that
credits itself for matching them has matched the wrong program.** If the study
wants a fidelity-matched comparison, dtype, epilogue and the fast-math flag must
be pinned identical on both sides — and then the remaining gap is, by §5.1's
arithmetic, small.

**6.5 A large fraction of the wall clock is host-side, and that is not a GPU
code-generation question at all.** Per `step()` call, unconditionally: a
device-to-host sync at `minecu/__init__.py:356`/`:439` (`self.do_reset.any()`), a
fresh tensor allocation at `:360`/`:444` (`self.do_reset.clone()`), 3-5 torch
kernel launches, and **20 `torch::Tensor` arguments plus 15 scalars** marshalled by
value into `py_step` (`bindings.cpp:408-452`) on the fp32 path — each `torch::Tensor`
by value is a refcount bump plus a `data_ptr` virtual call, and `TORCH_CHECK` runs
3-5 times per call (`bindings.cpp:453-457`). **The benchmark harness
(`docs/benchmarks.md:73-78`) synchronises only at the end of the loop, so all of
this host time is inside the measured 660 µs.** `GUESS`: 50-150 µs, i.e. 8-23% of
the step. `[NOT-HERE]` to settle: time `env.step()` with the render launch
commented out. If the number barely moves, the step is host-bound and no GPU
schedule change matters.

**6.6 And the techniques the human wrote that are simply not used.** Five items,
of which three carry their own table row: the `[Y,X,Z,B]` coalesced layout
(row 11, `:709-826`, dead), the ground-plane accelerated render (row 29,
`:1268-1442`, unreachable — `"uint8_fast"` is not an accepted `obs_dtype`,
`minecu/__init__.py:167`), the masked world-regen kernels (row 31, `:1989-2069`,
never launched); plus the standalone `prebasis`/`smem` render entry points
(`bindings.cpp:179-187`, bound but never called) and the warp-vote experimental
module (`scripts/benchmark_new_kernels.py:25`, source absent). **A technique
inventory built from a repo like this over-counts.** Count the shipped set:
**24, not 27.**

---

## 7. Where the class is genuinely ambiguous, and the test that settles it

### 7.1 The basis hoist: (i) on the split, (ii) on the effect. See §5.3.

**The test, and it is cheap.** Take `render_kernel_uint8_minimal`
(`kernels.cu:985-1119`) and add **one** variant that is identical except the
per-pixel trig at `:1017-1033` is replaced by a read of a `[B,14]` basis buffer
written by `precompute_camera_basis_kernel` (`:511-561`) — i.e. keep the 1-D
`idx = blockIdx.x*blockDim.x+threadIdx.x` mapping at `:997`, keep **one** render
kernel, and drop the shared-memory staging and the 2-D grid entirely. Add it to
`scripts/benchmark_new_kernels.py:190-195`, run at `batch=32768, world=16, 32×32`.

- If the one-kernel variant matches the two-kernel one → the win is **pure LICM,
  class (ii)**, and the kernel split is an implementation convenience, not the
  technique. The study's headline should be (ii), not (i).
- If the two-kernel version is materially faster → the split itself is load-bearing
  and the class is **(i) or (iii)**, specifically "cross-kernel schedule
  representation".

Either outcome is a real result and neither is available from reading the source.
The 128-thread block already contains 128 pixels of the same env (at 32px, 8
blocks per env), so shared-memory staging buys little over L1 — which is *itself*
an argument for (ii), and I want that on the record as the reason I am not
confident.

### 7.2 Is divergence exploitable at all? (iii) vs (iv).

`kernels.cu:225-255`. Compaction and sorting are representable and known work, so
the technique is (iii). Whether it is *worth* anything is (iv)-leaning, and it
turns on one number nobody has published: the ratio of `max(lane DDA steps)` to
`mean(lane DDA steps)` within a warp. Adjacent lanes are adjacent pixels of the
same camera, so their rays are similar and they should exit within a few steps of
each other — which would make the ratio near 1 and sorting worthless.
`[NOT-HERE]`: instrument `DDA_STEP` to `atomicMax` a per-warp step count into a
`[gridDim.x][32]` buffer and histogram it. The 1-byte histogram output at that size
would itself validate the technique it is measuring, which is a nice property.

### 7.3 Is the shared-memory voxel kernel a win or a loss? (ii) either way, but the
sign is unknown.

§3.2 entry 23. I predict a loss at 32px (8 blocks/env → 8× redundant 4 KiB grid
loads, against an L1 that holds the whole grid 32× over). `[NOT-HERE]`: A/B
`obs_dtype="uint8_prebasis"` against `obs_dtype="uint8_smem"`. Both entry points
already exist (`minecu/__init__.py:375-391` and `:392-408`).

### 7.4 How much of the step is host-side? §6.5. Unclassifiable from source.

The 20-tensor pybind11 boundary, the per-step `do_reset.any()` sync, and the
per-step `.clone()` are all `[ARTIFACT]`. Their *cost* is `[NOT-HERE]`. Note the
graph-capture fragility interacts here: `__init__.py:364` allocates a new
`do_reset` tensor every step on the non-`_full` paths, and `StepGraphState`
invalidates on pointer change (`kernels.cu:2529-2535`). **If the caching allocator
ever returns a different address, the graph is destroyed and re-captured on that
step.** Test: 1e5 steps with `episode_length=1`.

### 7.5 Did warp voting actually lose?

§3.2 entry 28. `scripts/benchmark_new_kernels.py:282-286` is written to handle both
outcomes and no result table is committed, and the module it imports
(`minecu._C_experimental`, `:25`) is not in the tree. **`[NOT-HERE]`, and it may
be unrecoverable.** If the study needs a datapoint on whether warp uniformity pays
for divergent ray traversal, it will have to write the kernel.

---

## 8. What I did not do, and what would change these conclusions

- **No GPU, no builds, no compilation, no disassembly.** `ptxas -v` register
  counts, occupancy, and the actual SASS for `LDG.CONSTANT` and `MUFU` counts are
  all unmeasured here. My claim in entry 20 that `__ldg` is free rests on nvcc's
  documented behaviour for `const T* __restrict__`, not on the SASS.
- **No timing of anything.** Every percentage in §4 is `GUESS` with a stated
  basis; §5.1's numbers are `MODELLED` arithmetic on `docs/benchmarks.md`'s own
  table. None is a measurement.
- **The 49.6M figure is taken on faith from `docs/benchmarks.md:7` and
  `scripts/plot_throughput.py:15-52`, which is a hardcoded dict.** There is no
  benchmark script in the repo that regenerates it; `docs/benchmarks.md:60-81`
  contains an inline snippet that would. So the single number the whole §5.1
  analysis rests on is unverified and, per §6.5, includes host time.
- **I did not read `examples/visualize.py`** — it is not on any performance path.
  `minecu/__init__.py` is 479 lines and I read all of it.
- **I did not audit the two obvious latent bugs**, because they are not techniques:
  `render_kernel_uint8_prebasis` returns at `kernels.cu:588` *before* the
  `__syncthreads()` at `:597` on a non-uniform condition
  (`pixel_offset >= pixels_per_batch`), which is a deadlock hazard that happens not
  to fire only because H×W is a multiple of 128 at all three shipped resolutions;
  and `examples/train_wood.py:106` calls `env.step(actions)` with one argument
  against the two-argument signature at `minecu/__init__.py:304-308`, and `:84`
  references an `env.action_dim` that does not exist anywhere in the package
  (`grep action_dim minecu/` → 0 hits). So **the shipped training example cannot
  run**, and therefore no number in `docs/benchmarks.md` came from it.
- **What would most change the conclusions:** the null-render experiment in §5.1
  (30 minutes, settles the memory-bound claim), the one-kernel basis variant in
  §7.1 (~20 lines, settles (i) vs (ii), the document's central classification),
  and a `ptxas -v` build to check the register and `LDG` assumptions in entries 20
  and 24. All three need a GPU; none needs paid compute.

---

## 9. One-paragraph summary for the study

`mine.cu` is not a warp-cooperation or occupancy-regime artifact: 3,245 lines with
zero warp intrinsics, zero atomics, zero `__launch_bounds__`, and no local memory
anywhere. Its shipped optimisation set is 24 techniques, of which the largest by
far — uint8 observations and a byte-LUT epilogue, worth 4× on a term that is
already 65% of DRAM spec bandwidth — is a *fidelity change no compiler may make*.
Of the genuine schedule techniques, the interesting one is the cross-kernel
hoist of per-env camera trig (`kernels.cu:511-561` + `:568-700`): the mechanism
requires a cross-kernel schedule representation nvcc does not have, but the effect
is a ~10% LICM that a within-kernel variant might capture, so its class is
genuinely (i)-or-(ii) and I have given the test. The second real technique, a
batch-major voxel layout that makes the program's only divergent store
conflict-free by construction (`kernels.cu:1655`), is a case of *not creating* a
problem rather than solving one, and the study's four-class taxonomy has no bucket
for that. Launch overhead is real but small (6-9 kernels per step, 1-5%), and a
larger fraction of the quoted 660 µs step is host-side pybind11 marshalling and a
per-step device-to-host sync at `minecu/__init__.py:356`. The most useful single
number in this file is therefore negative: **at the peak configuration there is
almost no memory-schedule headroom left for a compiler to find**, which is the
opposite of what a codegen study usually goes looking for and is worth saying
plainly.
