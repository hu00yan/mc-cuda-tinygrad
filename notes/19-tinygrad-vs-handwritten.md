---
kind: reference
---

# 19 — the same DDA in TinyGrad: where the generated code lands, and what kind of gap that is

**The question.** `notes/17-minecu-spec.md` specifies a hand-written CUDA voxel
raymarcher and stops short of one thing: not knowing what a code generator
makes of the same computation. That is this note. `minecu_tg/` is the same DDA
written the way a person writes it with array operations and a loop; this note
measures where TinyGrad's CUDA for it lands relative to
`minecu_ref/src/kernels.cu`, and **classifies the gap**.

**The rule that makes it a measurement.** The TinyGrad side is not tuned and
must not be. It does not call `custom_kernel`, set a tile size or `TC_SELECT`
or `ALLOW_TF32`, hand-unroll the DDA loop, hand-fuse stages, hand-stage data
through global memory, or pass a `Context(...)`. The full list, and the reason
each is excluded, is in `minecu_tg/render.py`'s docstring. **The gap is the
result.** Closing it would destroy the experiment.

> ### ⚠ WITHDRAWN IN PART — ledger **C-14**. Read this before quoting any number below.
>
> **The rule above was mis-stated, and it is the rule that made this a
> measurement.** "Not tuned" was executed as "naive", and the two are not the
> same thing. What it actually forbids is **hardware knowledge**: `custom_kernel`,
> tile sizes, `TC_SELECT`, `ALLOW_TF32`, hand-unrolling, hand-staging, launch
> bounds, per-thread occupancy, and consulting the reference's SASS while
> writing. What it accidentally permitted is **a worse algorithm** — which is
> not a neutral baseline, it is a second variable.
>
> `minecu_tg/render.py:235-261` recomputes the flat voxel address from three
> float coordinates on every one of 32 steps, and pays 3 `cast(i32)` + 3 `clip`
> to index the resulting gather. **Amanatides–Woo — the algorithm the reference
> implements — precomputes `tMax`/`tDelta` and then advances incrementally.**
> Recomputing the address each step is not that algorithm, and removing it
> requires no knowledge of any GPU. At 53 source ops/step against 1 137 SASS
> that is **21.5 SASS per operation**, where a scalar float lowering produces
> 1–3.
>
> **Two consequences, and they must not be merged.**
>
> 1. **Withdrawn as a measurement of the compiler: 26.2×, 59.5×, 29.9×, 47.2×,
>    36 600, 36 384.** These are an upper bound on *"the port plus the
>    compiler"*, and no part of the record may attribute them to TinyGrad until
>    the hoisting ablation separates the two. Note also that **§5.4 does not
>    sum**: it names 1.21× and 4× and reports 47.2× as explained, which is
>    4.85× of 47.16×, **leaving 9.73× with no row** — and it multiplies two
>    terms that both follow from the one fact that a tensor form cannot exit a
>    ray early, so the residue is bounded at **9.73×–38.91×**.
>    Derivation and the falsifiable target for the ablation:
>    `derivation/minecu_gap.py:decomposition`, `:independence`, `:op_census`,
>    `:sass_per_op`, `:ablation_target`.
> 2. **Not withdrawn, because it is not about index arithmetic: the fusion
>    crossover near 1 M pixels, `max_steps + 8` launches below it and 9 above,
>    and the 20 932 B spill.** These are decisions about fusion and register
>    allocation, measured from launch counts and stack frames. They survive
>    C-14 as observations about what the scheduler did. **Whether the decision
>    it made was the right one is still open**, because a kernel with 15 fewer
>    operations per step might not have spilled at all.
>
> The corrected comparison lives in `notes/23-good-vs-naive.md`. Where the two
> notes disagree, **23 is right and this one is wrong.**

**Labels.** `MEASURED` — an instrument ran here, on the artefact named, and the
artefact is named. `MODELLED` — arithmetic on shapes and on hardware constants.
`CLAIM` — the author's number, not reproduced. `[ARTIFACT]` — read out of a file.

**Runs.** The T4 numbers are from `out/minecu_tg/20260928-221859/`, one run, one
card, one session, both sides. The generated CUDA is from
`minecu_tg/capture_cuda.py` and the SASS from `minecu_tg/sass_tg.sh`; preserved
under `out/minecu_tg/codegen-sm75-b8/`, `codegen-sm75-b1024/`,
`sass-sm75-b8/`, `sass-sm75-b1024/`.

---

## 0. The answer, first

1. **TinyGrad unrolls the DDA loop, and whether it then *fuses* across the
   unrolled body depends on the tensor size — which is the finding, and it is
   sharper than the prediction was.** Below ~1 M pixels the loop is not fused at
   all: the launch count is **`max_steps + 8`, exactly**, one new kernel per DDA
   step, state materialised in global memory between them. Above ~1 M pixels the
   scheduler *does* fuse, into **one kernel of 36 384 SASS instructions with
   20 932 bytes of spill**. The prediction "the loop is not unrolled or fused"
   is **confirmed below the crossover and falsified above it**, and the
   crossover sits between batch 512 and 1024 at 32×32.
2. **The dominant gap is class (ii): expressible, and a cost model could find
   it — but in *both* regimes for different reasons, and the regime that gets
   timed is the worse-behaved one.** Unrolling and loop-carried state are in
   the graph either way. What the compiler will not do is make the two
   decisions that matter: fuse the small case *and* keep the fused kernel inside
   the register file.
3. **The output is not the problem. It is bit-identical.** On the state the
   reference itself produced, on the same card in the same run, the TinyGrad
   render's uint8 observation is **byte-for-byte the reference's**: 0 of
   4 194 304 pixels differ, identical sha256.
4. **Performance. ⚠ WITHDRAWN as a compiler measurement — ledger C-14. Read the
   banner above; the figures below are an upper bound on "the port plus the
   compiler" and the split is not yet known.** 26× to 60× slower than the
   hand-written render on the same card at the same batch, same output, same
   dtype. The ratio is a decomposition, not a verdict: the generated code issues
   **47.2× the SASS instructions per thread** (36 600 against 776, whole render,
   both sides' prologues included) and, in the regime that gets timed, spills
   **20 932 bytes to local memory per thread**. Of that 47.2×, **§5.4 accounts
   for 4.85× and leaves 9.73× unlisted** — `derivation/minecu_gap.py`.

The category question — what may be divided by what — is §5.1, answered before
any number is divided. **The 4.55 M whole-step figure is not the denominator.**

---

## 1. What was built, and that it is the same computation

### 1.1 The world, against the reference's own model

`MEASURED` `minecu_tg/verify.py`, this host, CPU, no GPU.
`minecu_tg/world.py` against `minecu_ref/bench/dda_model.py:build_world` — a
transcription that is not ours, which is the only kind that catches a shared
misreading of the spec:

| world³ | ground | solid voxels | identical to `dda_model`? |
|---|---|---|---|
| 16 | 4 | 1062 / 4096 | **yes, exact** |
| 32 | 8 | 8274 / 32768 | **yes, exact** |
| 48 | 12 | 27730 / 110592 | **yes, exact** |

### 1.2 The observation, on the T4, against the reference's own bytes

This is the strongest check available and it is the one that decides whether the
performance comparison means anything. `refdrv.cu` — a driver in
`minecu_tg/kaggle/` that links the **unmodified** vendored `kernels.cu` — builds
the world, runs the same settle with the same action stream, and dumps the
voxels, the 13-slot camera basis and the uint8 observation. The notebook then
feeds those exact bytes to the TinyGrad render.

`MEASURED` `out/minecu_tg/20260928-221859/obs_equivalence.json`, on the card:

```
pixels differing from the reference : 0 / 4194304  (0.0000%)
envs with at least one differing pixel: 0 / 4096
max channel difference              : 0
distinct colours, reference         : 6
distinct colours, tinygrad          : 6
obs sha256 reference 21e13f67fa003c488a7d5c9868944850d93e63f4e19f80e93b53a42432cdbfa7
obs sha256 tinygrad  21e13f67fa003c488a7d5c9868944850d93e63f4e19f80e93b53a42432cdbfa7
```

**Byte-identical, 4.2 million pixels, on the same card in the same run.** That is
stronger than the float64 CPU contract below, and it is the right one to lead
with, because it needs no argument about fast-math at all.

It is worth saying *why* it is not a foregone conclusion. The two renderers are
not bit-comparable in principle: the reference is compiled `--use_fast_math`, so
its `sinf`/`cosf`/`tanf`, its `1.0f/ray` and its `rsqrtf` are approximate, while
TinyGrad's CUDA backend emits an ordinary `1/sqrt(...)`. And a DDA step is chosen
by `t_max_x < t_max_y`, so a one-ulp difference in a `t_max` moves a ray to a
different voxel and the image differs by a whole block colour. That it came out
identical means the approximation gap never landed on a tie in this
configuration. That is a fact about this configuration, not a guarantee, and the
CPU-side check below is the one that does not depend on it.

### 1.3 The workload both sides executed, on the device

`MEASURED`, same run. A throughput number without the workload behind it is not
a measurement — `minecu_ref/buried_agent_fix.md` is four weeks of that lesson —
so both sides' DDA step counts are reported, and they were measured by
different instruments on the same state.

| | instrument | mean steps/pixel | at the 32 cap | first-voxel |
|---|---|---|---|---|
| reference | its own `dda_probe_kernel`, batch 4096, `minecu_ref/bench/bench.cu` | **26.447** | 80.91% | 5.005% |
| TinyGrad | `render(count_steps=True)` on the dumped state, batch 4096 | **27.123** | 82.1% | 6.0% |

Same world, same settled poses, same step cap, a 2.6% difference in the mean
and the same distribution shape. **The two sides did the same work**, which is
the precondition for every ratio in §5. Both are non-degenerate: the reference's
own gate is `mean ≥ max(2.0, 0.10 × 32) = 3.2` and it passed at 8.3× the floor,
and `minecu_tg/battery.py` carries the same gate on the TinyGrad side and prints
`gate: PASS` next to every timing.

For completeness, the same comparison on the CPU against an independent float64
transcription (`minecu_tg/cpu_reference.py`), which is the contract that does
not depend on any of this:

| case | pixels differing | DDA step counts agreeing | mean, tinygrad / float64 |
|---|---|---|---|
| world 16, pitch 0, `max_steps` 32 | **0 / 1024** | **1024 / 1024 (100%)** | 16.861 / 16.861 |
| world 16, pitch −0.2, 32 | **0 / 1024** | **1024 / 1024 (100%)** | 13.766 / 13.766 |
| world 32, pitch 0, 32 | **0 / 1024** | 1022 / 1024 (99.80%) | 16.029 / 16.027 |
| world 48, pitch 0, 32 | **0 / 1024** | 1022 / 1024 (99.80%) | 16.029 / 16.027 |
| world 16, pitch 0, 16 | **0 / 1024** | **1024 / 1024 (100%)** | 9.971 / 9.971 |

`MEASURED` `minecu_tg/verify.py`. The uint8 observation is bit-identical to the
float64 transcription in all five cases; the residual 2/1024 step-count
disagreements are float32-versus-float64 rounding of `t_max` at exact ties and do
not change the colour.

Two supporting checks, both `MEASURED` there: the step counter is inert
(identical observation with and without it, so the verified program and the timed
program are the same program), and per-env cameras are handled — at a ±1.2 rad
yaw spread the batch gives 3–5 distinct images and **every env matches a float64
render of its own camera**. That check exists because a `B=1` smoke test did not
catch a bug: `basis[:, i, None, None]` drops to rank 1, and `[None, None]` on a
rank-1 tensor is `(1, 1, B)`, not `(B, 1, 1, 1)`. The slice must be `i:i+1`.

### 1.4 A dead end worth recording

`MODELLED`, then `MEASURED` against `dda_model`: worlds 32 and 48 have the
**same** mean DDA step count, 16.029. Not a bug, and not mine. The camera sits at
`(world/2, ground+1.6, world/2)` and the tree and house are placed at fixed
offsets from *that point* (`kernels.cu:1831`, `:1868`), so the local geometry
around the camera is identical at every world size and only the far wall
recedes — and the far wall contributes no hits, because a ray that reaches it is
out of bounds and marches to the cap either way. `dda_model.py` reports 16.029
for both. Three independent implementations agreeing is the point of the
cross-check, not a defect in it.

---

## 2. The schedule: the prediction, and the test that settles it

> **The prediction under test.** A plain loop over DDA steps will make TinyGrad
> emit one kernel per step, materialising the ray position every step, whereas
> the human's 16× static unroll keeps 16 independent load chains in registers.
> The decisive test is that the kernel count scales with `max_steps`.

### 2.1 Against `max_steps`, at the capture shape — the law holds exactly

`MEASURED`, `PARALLEL=0`, batch 8, world 16, 32×32. Two columns because the
question deserves two answers: the CUDA **renderer** (the one that produces the
code being compared, via the NULL device so it runs with no hardware) and the
CPU backend, which exercises the same scheduler.

| `max_steps` | launches, CUDA renderer | launches, CPU backend | distinct programs launched (CUDA) | `launches − 8` |
|---|---|---|---|---|
| 1 | **8** | 8 | 6 | 0 |
| 2 | **10** | 10 | 7 | 2 |
| 4 | **12** | 12 | 9 | 4 |
| 8 | **16** | 16 | 13 | 8 |
| 16 | **24** | 24 | 21 | 16 |
| 32 | **40** | 47 | 37 | 32 |
| 64 | **72** | 95 | 69 | 64 |

`MEASURED` `minecu_tg/census_main.py` (CPU) and `minecu_tg/capture_cuda.py`
(NULL:CUDA:sm_75). `launches = max_steps + 8`, exactly, at every `max_steps` from
2 to 64, on both backends. The two columns differ only above `max_steps` 16,
where the CPU backend adds layout kernels the CUDA renderer does not need.

The CUDA-renderer column is preserved whole at
`out/minecu_tg/codegen-sm75-b8/census.json` — all seven `max_steps` values with
their `launch_keys_ordered`, so the `distinct` column can be re-derived rather
than trusted:

```
  max_steps=  1  launches=  8  distinct=  6  names=6
  max_steps=  2  launches= 10  distinct=  7  names=7
  max_steps=  4  launches= 12  distinct=  9  names=7
  max_steps=  8  launches= 16  distinct= 13  names=7
  max_steps= 16  launches= 24  distinct= 21  names=7
  max_steps= 32  launches= 40  distinct= 37  names=7
  max_steps= 64  launches= 72  distinct= 69  names=7
```

**The prediction is confirmed here.** And two further facts, both `MEASURED`:

- **No program is ever reused.** `distinct_programs_launched` is
  `launches − 3` at every `max_steps ≥ 4`. Every launch is a *different*
  compiled program of a different size.
- **Names are not identities.** All 16 DDA programs at `max_steps=32`, batch 8,
  are named `E_8_32_4_4_8_16_4` — the name is derived from the kernel's
  *bounds*, not its body. A census that counted names would report "31 launches
  of one kernel", which is false. `minecu_tg/codegen.py` records the program
  **object id** alongside the name for exactly this reason, and this is the
  second time in this project a name has been mistaken for a thing
  (`notes/13` §4 was the first).

### 2.2 Against batch — and here the prediction breaks

`MEASURED`, CPU backend, `max_steps=32`, `PARALLEL=0`. The row that matters is
the last column, and the answer is not constant.

Re-run and preserved at `out/minecu_tg/census-cpu-b{1,8,64,256,512,1024,2048,4096}/census.json`,
one directory per batch, so this table is re-derivable rather than remembered:

```
DEV=CPU python3 minecu_tg/census_main.py --batch <B> --steps 1,4,16,32 \
    --outdir out/minecu_tg/census-cpu-b<B>
```

| batch | pixels | `ms=1` | `ms=4` | `ms=16` | `ms=32` | `launches − 8` at `ms=32` |
|---|---|---|---|---|---|---|
| 1 | 1 024 | 7 | 8 | 8 | 8 | 0 |
| 8 | 8 192 | 8 | 12 | 24 | 47 | 39 |
| 64 | 65 536 | 8 | 12 | 24 | 47 | 39 |
| 256 | 262 144 | 8 | 12 | 24 | 47 | 39 |
| 512 | 524 288 | 8 | 12 | 24 | 47 | 39 |
| **1024** | **1 048 576** | 8 | **9** | **9** | **9** | **1** |
| 2048 | 2 097 152 | 9 | 10 | 10 | 10 | 2 |
| 4096 | 4 194 304 | 8 | 9 | 9 | 9 | 1 |

`MEASURED` on the **real CUDA backend**, the same crossover: batch 1024 gives
**9 launches at `max_steps=32`**, and 10 at `max_steps=4`
(`out/minecu_tg/20260928-221859/tg/report.json`, `census`, and
`out/minecu_tg/codegen-sm75-b1024/census.json`). And the T4's own timed
batteries saw it: 40 kernels at batch 256 and 512, **9 at batch 1024 and 10 at
2048** (`device … over 40 kernels`, `over 9 kernels`, `over 10 kernels`).

**So the honest statement of the prediction's verdict is two-sided, and the
second half matters more than the first:**

- **≤ 524 288 pixels: confirmed.** The loop is unrolled and *not* fused. One
  kernel per DDA step, `max_steps + 8` launches, the loop-carried state in
  global memory between them.
- **≥ 1 048 576 pixels: falsified.** The launch count stops depending on
  `max_steps` entirely — 9 launches at `max_steps = 4` and 9 at `max_steps = 32`
  — because the scheduler *does* fuse the whole unrolled body, into one kernel.

The fusion decision is therefore **not absent**; it is **size-dependent**, and
it flips between batch 512 and batch 1024 at 32×32. A study that had measured
only at the capture shape would have reported "TinyGrad never fuses a loop" and
been wrong about the shapes it went on to time. That is the single most useful
thing in this note, and it is why the sweep in §2.2 runs over batch as well as
`max_steps`.

### 2.3 What each of those kernels is, in both regimes

`MEASURED` `minecu_tg/analyse_codegen.py` over the captured sources; SASS from
`minecu_tg/sass_tg.sh`. `max_steps = 32` throughout.

**Regime A — batch 8, 8192 pixels: 17 distinct programs, 40 launches.**

| programs | args | state buffers | source lines | DDA compares | what it is |
|---|---|---|---|---|---|
| 1 | 2 | 0 | 12 | 0 | scalar constant (the colour LUT) |
| 1 | 1 | 1 | 9 | 0 | the per-env camera basis |
| 1 | 2 | 1 | 147 | 4 | the **first** DDA program: prologue + 1 step |
| 1 | 2 | 1 | +15 each | +1 | prologue + 2 steps, … |
| 15 | 2 | 1 | 552 | 31 | prologue + **16** steps |
| 1 | **33** | **33** | 14 546 | 496 | the remaining 16 steps **and** the colour store |

A chain of nested prefix fusions, each one instruction longer than the last
(`+15` lines, `+1` compare, `+3` FFMA per step), terminated by one enormous
kernel that takes all 31 materialised state buffers as inputs.

**Regime B — batch 1024, 1 048 576 pixels: 6 distinct programs, 10 launches.**

| program | args | state buffers | source lines | what it is |
|---|---|---|---|---|
| `E_8_32_4` | 1 | 0 | 13 | scalar constant |
| `E_8192_32_4` | 1 | 1 | 9 | the basis, one env per thread |
| `E_128_4_8_16_4` | 3 | 3 | 40 | the pixel grid |
| `r_1024_16_16` | 3 | 2 | 43 | layout |
| **`E_512_2_8_8_4_4`** | 8 | 4 | **15 860** | **the entire 32-step DDA and the colour store, in one kernel** |
| `E_32768_32_3` | 3 | 2 | 18 | the final store |

No chain, no materialised-per-step state, no prefix ladder. One kernel that does
the whole render. **And that is the kernel the performance number is about**,
because batch 1024 and above is where it is timed.

---

## 3. The generated CUDA, beside the reference's

Quoted verbatim. Reference from `minecu_ref/src/kernels.cu`; generated from
`out/minecu_tg/codegen-sm75-b8/max_steps_0032/kernel000.cu`, batch 8, compiled
for sm_75.

### 3.1 Ray construction and the traversal setup

`kernels.cu:619-659`:
```c
    u = (2.0f * (px + 0.5f) / width - 1.0f) * scale_u;
    v = (1.0f - 2.0f * (py + 0.5f) / height) * scale_v;
    ray_x = forward_x + u * right_x + v * up_x;
    ray_y = forward_y + v * up_y;
    ray_z = forward_z + u * right_z + v * up_z;
    float ray_len = rsqrtf(ray_x*ray_x + ray_y*ray_y + ray_z*ray_z);
    ...
    float t_max_x = (ray_x >= 0) ? ((float)(voxel_x + 1) - eye_x) * inv_ray_x
                                 : ((float)voxel_x - eye_x) * inv_ray_x;
```

generated, `kernel000.cu:28-45`, SSA-renamed, comment appended:
```c
  float alu8  = ((((val14+0.5f)*0.0625f)+-1.0f)*val10);        // u   (0.0625 = 2/32)
  float alu9  = ((1.0f+((val13+0.5f)*-0.0625f))*val11);          // v
  float alu10 = (val3+(alu9*alu8));                              // ray_z
  float alu11 = (val2+(alu8*alu5)+(alu9*alu7));                  // ray_x
  float alu12 = (val4+(alu8*alu6)+(alu9*alu9));                  // ray_y
  float alu13 = (1/sqrt(((alu11*alu11)+(alu10*alu10)+(alu12*alu12))));
  float alu14 = (alu10*alu13);                                   // ray_z * rsqrt
  bool  alu15 = (val0<alu2);                                      // ray_x >= 0
  bool  alu16 = (alu14<0.0f);                                     // ray_z >= 0
  float alu17 = (alu15?alu5:alu2);                               // floor(eye_x)+1 : floor(eye_x)
  float alu18 = (alu16?-1.0f:1.0f);
  float alu19 = (alu17+alu18);                                    // t_max init term
```

Faithful, including the association of the sum of squares, which is `(x²+y²)+z²`
— the C source's left-to-right reading, and *not* what
`minecu_ref/bench/dda_model.py` writes. And `rsqrt` is emitted as
`1/sqrt(...)`, a correctly-rounded divide by a correctly-rounded square root,
where the reference's `rsqrtf` is `MUFU.RSQ`. That is a real numerical
difference between the two builds; §1.2 is where it gets checked.

### 3.2 The DDA step: the reference's macro, and what was generated

`kernels.cu:664-679`, the body `ptxas` unrolls 16 times:
```c
    #define DDA_STEP_PRE() \
        if (step >= max_steps) goto done_pre; \
        if ((unsigned)voxel_x < (unsigned)wx && \
            (unsigned)voxel_y < (unsigned)wy && \
            (unsigned)voxel_z < (unsigned)wz) { \
            int8_t block = __ldg(&voxels[b * wxyz + voxel_y * wxz + voxel_x * wz + voxel_z]); \
            if (block >= 0) { hit_block = block; goto done_pre; } \
        } \
        if (t_max_x < t_max_y && t_max_x < t_max_z) { \
            t_max_x += t_delta_x; voxel_x += step_x; \
        } else if (t_max_y < t_max_z) { \
            t_max_y += t_delta_y; voxel_y += step_y; \
        } else { \
            t_max_z += t_delta_z; voxel_z += step_z; \
        } \
        ++step;
```

generated, `kernel000.cu:283-326`. The whole step, and it is where the comparison
has to be made carefully:
```c
  int alu264 = (alu227?((int)((alu223+alu30))):alu263);   // voxel_x += step_x
  int alu265 = ((alu264<0)?0:alu264);                    // clip(0, ...)
  int alu266 = ((15<alu265)?15:alu265);                  // clip(..., 15)
  int alu267 = (alu21?((int)(alu6)):((int)(alu3)));      // the other two axes
  int alu268 = (alu43?((int)(alu25)):alu267);
  ...
  int alu286 = (((alu247<16)&(alu266<16)&(alu285<16))    // inside: THREE compares
                ?((gidx2<<12)+(alu247<<8)+(alu266<<4)+alu285)   // flat [B,Y,X,Z] index
                :0);
  int alu287 = (gidx0>>2);
  int alu288 = ((lidx1<<9)+(alu287<<13));
  bool alu289 = (alu286<(alu288+128));
  bool alu290 = (alu286<(alu288+256));
  signed char val15 = (((alu289!=1)&alu290)?*(data4_32768+alu286):((signed char)(0)));
  bool alu291 = (alu286<(alu288+384));
  signed char val16 = (((alu290!=1)&alu291)?*(data4_32768+alu286):((signed char)(0)));
  signed char val17 = (((alu291!=1)&(alu286<(alu288+512)))?*(data4_32768+alu286):((signed char)(0)));
  signed char val18 = ((((alu286<alu288)!=1)&alu289)?*(data4_32768+alu286):((signed char)(0)));
  int alu292 = ((lidx0<<8)+(alu1<<11)+(lidx1<<2)+(alu287<<6)+(gidx1<<13)+(gidx2<<18));
  *(data0_2097152+(alu292+1)) = val15;                  // state to GLOBAL memory
  *(data0_2097152+(alu292+2)) = val16;                  // ... one round trip per step
  *(data0_2097152+(alu292+3)) = val17;
  *(data0_2097152+alu292)   = val18;
```

Four things to read off, all measurements of what the generator did:

1. **The unsigned bounds trick is gone.** The reference's
   `(unsigned)voxel_x < (unsigned)wx` folds three axes into three
   `ISETP.GE.U32.OR`; the generated form is `(a<16)&(b<16)&(c<16)`, three signed
   compares. **Both are 3 instructions on sm_75**, so on this architecture the
   fold is free and the cost model lost nothing measurable. §6, gap 6.
2. **The voxel read costs 4 global loads, not 1.** `data4_32768 + alu286` is
   loaded four times, each under a different predicate, because the four `bool`
   consumers of the step are masked separately and the compiler cannot prove the
   predicates are related. The reference's `__ldg` is one
   `LDG.E.S8.CONSTANT.SYS`. A direct cost of the masked formulation.
3. **In regime A the state goes to global memory at the end of every program.**
   The four stores at `alu292` are the DDA state leaving registers. The grid is
   `(16, 32, 8)` with block `(8, 16)` — 524 288 threads for 8192 pixels, **64
   threads per pixel**, each storing 4 bytes, every address distinct.
   `data0_2097152` is 262 144 bytes per env = **256 B of int8 state per pixel**,
   all of it written on each of the 32 steps, and the buffer is **8× replicated**
   relative to one entry per pixel (65 536 four-byte slots per env for 1024
   pixels).
4. **That 256 B/pixel does not transfer to other batches, and this is worth
   stating because an earlier draft of this note generalised from it and was
   wrong.** It is a consequence of the *64-threads-per-pixel* mapping that regime
   A's grid produces. `MODELLED` from it: if it held at batch 32768, one state
   buffer would be 8 GiB and the final kernel's 31 of them 248 GiB, so the
   generated side could not run at the headline shape at all. `MEASURED`
   allocator peak on the T4: **996.1 MiB at batch 32768**, i.e. ~30 B/pixel.
   The prediction was off by ~8× and the memory is not a blocker at all. §5.3
   has the measured curve.

### 3.3 The store

`kernels.cu:690-699` is a branch:
```c
done_pre:
    if (hit_block >= 0) {
        output[out_idx + 0] = BLOCK_COLORS_U8[hit_block][0];   // LDC.U8 from constant memory
        ...
    } else { output[out_idx + 0] = SKY_COLOR_U8[0]; ... }
```
The tensor version maps AIR to row 13 of a 14-row table and turns the branch
into the same gather the hit path already does. A change of *expression*, not of
schedule, and the one place this implementation is deliberately shaped rather
than naive. It is disclosed in `render.py` and priced in §6, gap 12.

---

## 4. The SASS matrix, both sides, one categoriser

`MEASURED`. Reference from `minecu_ref/build/kernels_sm_75.sass` +
`resources_sm_75.txt` (`nvcc 12.9 -O3 --use_fast_math -arch=sm_75`, via
`minecu_ref/build.sh`). Generated side: the captured `.cu` compiled in the
`archlab/nvcc:12.9` container by `minecu_tg/sass_tg.sh`, same flags. Both through
`minecu_tg/sass_matrix.py`, which **imports**
`tools/archlab/inspect_ptx.py:CATEGORIES` rather than defining a second
vocabulary — and that file is shipped to the card as a payload so the claim is
literally true on both sides.

**One instrument bug, found and fixed, and it would have produced a
plausible-looking wrong number.** The SASS offset field is 4 hex digits for the
first 4096 instructions and 5 after, so a `{4}` pattern reports *exactly 4096
instructions* for any larger kernel. Three kernels here with 277, 1045 and 2069
global loads all came out as "4096 instructions". `tools/archlab`'s own
`SASS_LINE` already used `{4,6}`; `sass_matrix.py` now uses `{4,8}`, and the
reference's 776 is unchanged under both, which is the check that the fix is
right.

### 4.0 Which CUDA built which SASS

| SASS | built by | nvcc | used for |
|---|---|---|---|
| the reference | `minecu_ref/build.sh`, this host | **12.9** | the reference column |
| the generated code, batch 8 and 1024 | `archlab/nvcc:12.9` container, this host | **12.9** | the TinyGrad columns |
| the generated code on the card | the T4 notebook | **12.8** | not used: that stage hit a missing `clang` and is reported as unavailable |

**The §4.1 table is 12.9 against 12.9**, deliberately: a register count and an
instruction count are properties of a *compiler version* as much as of a source,
and comparing across toolchains would make the register numbers meaningless. The
T4 image carries CUDA 12.8, which is recorded and not silently used.

### 4.0.1 The reference's SASS, re-derived here rather than quoted

`notes/17` §2.7 states the hot kernel is 776 instructions with a 38-instruction
load-to-load spacing, 15 times. Re-derived for this note from
`minecu_ref/build/kernels_sm_75.sass` by walking the instruction stream of
`_Z28render_kernel_uint8_prebasis`:

```
instructions in the hot kernel:  776
LDG.E.S8* count: 16   opcodes: {LDG.E.S8.CONSTANT.SYS}
their instruction indices: 152 190 228 266 304 342 380 418 456 494 532 570 608 646 684 722
load-to-load spacings: 38 38 38 38 38 38 38 38 38 38 38 38 38 38 38
all equal? True    value: 38
before the first load: 152   after the last: 53   body span: 570
BSSY: 33   BAR.SYNC: 1   STG.E.U8.SYS: 6
```

The 16 loads are the unrolled `__ldg` voxel reads and the spacing is identical
fifteen times — the strongest available evidence that ptxas emitted all 16
unrolled steps and did not fold, reorder or spill them. The 776 decomposes
exactly: **152 of prologue + 16 × 38 of body + 16 of epilogue**; the 53 trailing
instructions are the last step's body tail plus the epilogue. §4.1's
instruction comparison rests on this, so it is derived rather than inherited.

### 4.1 The two sides, `max_steps=32`, in both regimes

| | reference `render_kernel_uint8_prebasis` | TinyGrad regime A (batch 8, 16 DDA programs) | TinyGrad regime A (final kernel) | **TinyGrad regime B (batch 1024, the timed one)** |
|---|---|---|---|---|
| kernels (launches) | **1** | 16 | 1 | **1 of 9** |
| SASS instructions | **776** | 440 → 712 | **23 936** | **36 384** |
| registers | **28** | 31 → 32 | **255** (the ceiling) | **168** |
| spill stores / loads | **0 / 0** | **0 / 0** | **1292 / 2328** | **7460 / 13 472** |
| stack frame | 0 | 0 | **1512 B** | **5472 B** |
| barriers | 1 | 0 | 0 | 0 |
| `BSSY`/`BSYNC` pairs | **33** | **0** | **0** | **0** |
| global loads | 17 (16 are `__ldg`) | 19 | 1045 | **517** |
| global stores | 6 | 4 | 4 | 4 |
| FFMA | 9 | 54 → 99 | 1524 | 52 |
| launch bounds | 128 | 128 | 128 | 128 |

**Static instructions in the whole render: 776 against 33 184 in regime A and
36 600 in regime B — a factor of 42.8 and 47.2.** And the comparison is
like-for-like: the reference's 776 *contains* all 16 of its unrolled DDA steps
— 152 of prologue + 16 × 38 of body + 16 of epilogue = 776 exactly — so both
numbers are "the whole render", and both TinyGrad figures contain all 32 of
theirs.

Both totals are the sum over every program in the regime, not just the hot one,
and both are re-derivable from the preserved SASS by counting the instruction
stream of all 17 and all 6 files respectively. The regime-B sum is
`16 + 16 + 80 + 72 + 36 384 + 32`; the fused kernel alone is 36 384 of it, so
`36 384 / 776 = 46.9` is the single-kernel ratio and `47.2` is the whole-graph
one. **An earlier draft of this note printed 36 536 here, having undercounted
the 80-instruction pixel-grid kernel `E_128_4_8_16_4` as 16.** The factor it
supported, 47.1, is unchanged to one decimal place and the conclusion is
unaffected, but the total was wrong and is corrected.

### 4.2 The four findings in that table

**(a) The reference keeps the state in registers; the generated code does not,
and the boundary is exactly the kernel boundary.** 28 registers, zero spills,
for a kernel whose DDA body is 16 copies of 38 instructions. The 16 regime-A
programs are also spill-free at 31–32 registers — *within* a program the state
is in registers too. The difference is entirely the store at the end of each
program and the load at the start of the next, which is 32 round trips per
pixel that the reference does not make at all.

**(b) In regime B the fusion is achieved and the kernel is still 47× bigger and
it spills 20 932 bytes.** 168 registers, 5472 B of stack frame, 7460 B of spill
stores and 13 472 B of spill loads. The register file is not the ceiling here
(255 is) — the live set simply is larger than 168 registers, and ptxas did the
only thing it could: move the excess to local memory. **This is the clearest
single piece of evidence that the generated schedule is not a different *order*
of the same work. It is a decomposition that defeats the register allocator, and
it happens in the regime that gets timed.** In regime A the same 32 steps are
spread over 17 spill-free programs; in regime B they are in one and the
allocator loses. §4.3 gives the measured reason: the reference's 16 steps are
byte-identical repeats that ptxas can allocate one at a time, and regime B's 32
are not identical at all.

**(c) `BSSY` 33 versus 0 is the early exit, and it is worth 33 instructions per
copy of the DDA body.** The reference's `goto done` compiles to real branch
synchronisation — 33 pairs, one per unrolled step, because a warp diverges: some
threads have hit and some have not. The generated code has **zero**, because
there is no early exit to synchronise on. That is not efficiency; it is the
masked formulation, in which every ray runs every step and "done" is a
predicate. The generated code is *cheaper* per step and does *more total work*;
the reference is *more expensive* per step and does less. The clearest instance
in the comparison of a number that looks like a regression and is not.

**(d) The 517 global loads in regime B, against 16.** Even fully fused, the
generated kernel issues 32 predicated quads of voxel reads against the
reference's 16 single `__ldg`s, plus the reads of the 8 argument buffers. Per
DDA step that is 4 predicated loads of one address (§3.2, finding 2) against 1.

### 4.3 Load-bound or issue-bound?

**The reference is issue- and latency-bound, and its own SASS says so.** 608 of
its 776 instructions are the DDA body, at 38 per step, for a measured mean of
26.4 steps per pixel. The body is a compare-and-accumulate with **one** load and
no arithmetic dependency a stall could hide, so there is no memory-level
parallelism to exploit *within* a thread — which is exactly why the 16× unroll
matters: it hands the scheduler 16 independent load chains. `notes/17` §7
computes the issue-bound ceiling and finds the workload 6.1× past it on a 3090;
the T4 numbers in §5 are the same arithmetic with the T4's issue rate.

**The generated code is issue-bound too, and by tens of times.** Two bases,
because they differ and only one of them is measurable:

| basis | reference | generated, regime B | factor |
|---|---|---|---|
| whole render, one thread | 776 | 36 600 | **47.2×** |
| whole kernel amortised per DDA step | `776 / 16` = 48.5 | `36 384 / 32` = 1137.0 | **23.4×** |
| DDA body only, per step | 38 | **not measurable** — see below | — |

**The 47.2× is the one to quote** — ⚠ **as a count, not as a compiler result
(C-14)** — and it is the only basis on which the two kernels are the same kind
of object: each is one thread's entire render, with its own prologue included.
The per-step row is lower because the reference's prologue is 152 of its 776
instructions — **19.6% of the kernel** — paid once for ray construction and
shared by 16 unrolled steps, whereas the fused kernel's prologue is negligible
against 32. **What the count is evidence *of* is not yet settled: the port's
per-step address arithmetic and the lowering are not separated, and
`derivation/minecu_gap.py:ablation_target` is the experiment that separates
them.**

**The third row is not measurable, and that is itself the finding.** A body-only
per-step count requires the step bodies to be identical in both kernels, and
they are not. `MEASURED`, by scanning each opcode stream for a period: the
reference's 16 unrolled steps are **byte-identical repeats at period 38**, with
an exact match run of 569 instructions; the fused kernel's 32 steps **do not
repeat at any period** — the longest exact match found for any period from 20 to
200 is 2943 instructions, and no candidate is a 32-fold repeat. ptxas has
CSE'd, reordered and re-materialised *across* the whole 32-step body, treating
it as one optimisation problem rather than 32 copies of a 38-instruction step.

**That is the mechanism behind §4.2b, measured rather than inferred**: a schedule
with 32 independent identical steps can be register-allocated step by step, and
this one is. A schedule that is one 36 384-instruction straight-line region is
allocated as a whole, and 168 registers is what it got. The spill is not a
secondary symptom of the fusion; it is what "one problem instead of 32" costs.

Where the extra ~1100 per step goes is legible in §3.2: six signed bounds
compares instead of three folded unsigned ones, four predicated byte loads
instead of one `__ldg`, the full state update as selects rather than in-place
accumulation, and no `goto`-based exit to skip the rest of the body for a ray
that has already hit — so a ray that hits on step 1 still executes steps 2
through 32.

The state round trip in regime A is a *second*, additive cost and it does not
exist in regime B: in regime B the state never leaves the register file except
for what spills. So the two regimes trade one dominant cost for the other, and
the measured ratios (59.5× at batch 512, 26.2× at 1024, 29.9× at 2048) are
consistent with that trade.

---

## 5. Performance

`MEASURED`, `out/minecu_tg/20260928-221859/`, one Tesla T4, one session. Both
sides, same card, same world (the reference's own dumped voxels), same
`max_steps=32`, same uint8 output, verified byte-identical (§1.2), same measured
DDA workload (§1.3).

### 5.1 The category question, answered before any number is divided

**A framework emits a graph of kernels. The 4.55 M baseline is a whole-step
number over 7 launches. Dividing one by the other is a category error**, and
`notes/17` §8.2 says so before any measurement exists to be misled by.

| candidate comparison | legal? | why |
|---|---|---|
| TinyGrad render graph ÷ 4.55 M whole-step env-steps/s | **no** | the numerator is one kernel's worth of 9; the denominator is 7 kernels including five state updates. It divides a per-op by a per-step. |
| TinyGrad render graph ÷ the reference's `render_kernel_uint8_prebasis` per-kernel median, **same batch** | **yes** | both sides are one kernel's worth of work, same output, same dtype, same world, same card, same run. §5.2. |
| TinyGrad whole graph ÷ the reference's whole step | **partly** | both are "produce this step's observation", but the reference's step also does decode, physics, raycast-interact and reset: 464 SASS instructions and ~4 MiB against a 776-instruction render. A known constant, not a rounding error, so the ratio reads as "the render part of a step against a step". |

**The 4.55 M figure appears in §5.5 as context, and is not the denominator
anywhere.** A ratio against it would be arithmetically computable and
semantically empty.

### 5.2 The defensible comparison, per batch

The reference side, `cudaEvent` medians over 8 iterations after 5 warmup, from
`refdrv.cu` linking the unmodified `kernels.cu`. The TinyGrad side is the sum of
per-kernel device times over the whole generated graph, via tinygrad's own
`time_call`.

| batch | ref render, µs | ref whole step, µs | TinyGrad device, ms | kernels | **ratio, render vs render** | TinyGrad peak, MiB |
|---|---|---|---|---|---|---|
| 512 | 228.0 | 278.5 | 13.573 | 40 | **59.5×** | 139.6 |
| 1024 | 334.8 | 381.3 | 8.754 | 9 | **26.2×** | 31.3 |
| 2048 | 579.4 | 626.7 | 17.323 | 10 | **29.9×** | 62.0 |
| 4096 | 1146.0 | 1198.8 | — | — | not timed | 125.0 |
| 8192 | 1611.8 | 1668.2 | — | — | not timed | 250.0 |
| 16384 | 3246.1 | 3497.2 | — | — | not timed | 500.0 |
| 32768 | 6668.8 | 6985.7 | — | — | not timed | 996.1 |

`MEASURED` same run. Two things about the shape of this table.

**The ratio is 26× to 60×, and it is not one number.** The 59.5× is regime A
(40 kernels, state materialised per step); the 26–30× is regime B (9–10 kernels,
one fused kernel with 20 932 B of spill). Fusing the loop bought about 2×. The
remaining factor of ~26 is arithmetic and spilling, and §4.1 says exactly which.

**Timing stops at batch 2048 and memory does not.** That is deliberate and it
was a prediction that failed: I expected the generated graph to be
un-allocatable above batch ~2048, from the 256 B/pixel state figure in §3.2.
The measured allocator peak is 996.1 MiB at batch 32768 — about 30 B/pixel — so
**the generated side runs, and fits, at the project's own headline batch.** The
`not timed` cells are a sweep bound, not a failure: `--time-batches` was capped
at 2048 to bound the run, and §2.2 shows the kernel count there is the same 9–10
the memory sweep implies. The `time_call` instrument is what costs the time, not
the kernel.

### 5.3 Where the generated side stops fitting

It does not, and that is worth saying plainly because an earlier draft of this
note predicted a 248 GiB footprint from the batch-8 capture. The measured curve
is `MEMPHYS`: 139.6, 31.3, 62.0, 125.0, 250.0, 500.0, 996.1 MiB at batch 512
through 32768 — linear in batch, ~30 B per pixel, 9 allocations at every batch.
The batch-512 point is the outlier at 139.6 MiB, consistent with regime A's
8×-replicated state layout; from batch 1024 the layout is one thread per pixel
and the cost is 30 B/pixel. **`MODELLED` from the batch-8 capture and
`MEASURED` on the card disagree by 8×, and the measurement wins.**

### 5.4 Decomposing the difference

Take batch 1024, where both a ratio and a mechanism exist.

| term | how it is obtained | value |
|---|---|---|
| **instructions issued per thread** | 36 600 vs 776 SASS for the whole graph, §4.1 (36 384 vs 776 for the fused kernel alone, 46.9×) | **47.2×** |
| of which: no per-ray early exit | every ray runs all 32 steps; `BSSY` 33 → 0, §4.2c | ⚠ **SUPERSEDED (C-25)** — this row read "32 steps vs a measured 26.4 mean — a 1.21× term". The mean is **16.54**, measured twice, so the term is **1.94×**. The 1.21× was the *smaller* error, which made the port look better than it is |
| of which: 4 predicated byte loads per step | §3.2 finding 2, §4.2d | **4×** the read instructions |
| of which: local-memory spill | 7460 B stores + 13 472 B loads per thread, §4.1 | not separable from the instruction count without a profiler; the 5472 B stack frame at 168 registers is the mechanism |
| **kernel count and launch overhead** | 9 vs 1 launches | **small**: 9 launches at ~4 µs is ~36 µs against 8754 µs, so **< 0.5%**. The 40-launch regime-A case is 40 × 4 µs = 160 µs against 13 573 µs, **1.2%** |
| **memory traffic** | regime A: 256 B/pixel written + read per step, 64 threads/pixel; regime B: no state round trip, but 20 932 B of local traffic per thread | regime A is the dominant term at batch 512; regime B is not |
| **⚠ MISSING ROW — per-step voxel-address arithmetic** | `render.py:244-247`: 3 `cast(i32)` + 3 `clip` + a 4-index gather that recomputes the flat address from three float coordinates, on all 32 steps. 15 of the 53 source ops/step. The reference advances a precomputed address instead. | **9.73×**, by subtraction — the table above names 4.85× of 47.16×. `derivation/minecu_gap.py:decomposition`. **This row was absent, and its absence is what let 26.2× be reported as a compiler result (C-14).** |
| **measured, whole** | §5.2 | **26.2×** at batch 1024, 59.5× at 512 — **⚠ C-14, void as a compiler measurement** |

**⚠ This sentence was wrong, and it was wrong by omission, which is the hardest
kind to catch. It claimed the instruction count was "the un-unrolled early exit
plus the un-folded load" — two terms, 1.21× and 4×, together 4.85× of 47.16×.
The remaining 9.73× was never listed.** The kernel count and launch overhead
part of it stands and is a genuine negative result: **for this kernel, on this
card, launch count is not the cost** — under 1.5% on both sides of the
crossover, against what the "12,082 kernels" framing in `notes/13` would lead
one to expect. But the decomposition does not close, and the unlisted term is
the port's own per-step address arithmetic (C-14, ledger).

### 5.5 The 4.55 M number, and why it is context rather than denominator

`CLAIM`, not reproduced: the project's own figure is 4.55 M env-steps/s for a
whole step at batch 32768. `MEASURED` on this card in this run, the same
reference at that batch: whole step **6985.7 µs, 4.684 M env-steps/s** — 2.9%
above the baseline, which is the run-to-run variance of a shared T4
(`minecu_ref/buried_agent_fix.md` records ~1%, and the two runs in
`out/minecu_tg/` differ by 2.9% on the same measurement).

The TinyGrad side at batch 2048 — the largest batch timed — is **0.118 M
env-steps/s** (device time only; end-to-end including Python tracing is 610.7 ms
for the same graph, 0.0034 M). Against the 4.55 M headline that is a factor of
**38.6**, and against the reference's *own whole step at the same batch*
(626.7 µs, 3.27 M) it is **29.9**.

**Neither is the number this note reports, and the reasons are the ones in
§5.1.** The 38.6× divides a per-op at batch 2048 by a per-step at batch 32768. The
29.9× is defensible and is §5.2's row. Quoting 38.6× would be a better-sounding
and worse number, and the whole point of §5.1 is that the better-sounding one is
the wrong one.

### 5.6 What the two host-side numbers add

`MEASURED` same run, batch 1024: Python tracing of the unrolled 32-step graph is
**211.9 ms** and trace-plus-realize is **590.1 ms**, against 8.754 ms of device
time. **The framework's host cost is 24× its device cost at this shape.** A
caller that does not hoist the graph out of its inner loop is not measuring the
GPU at all. That is not a defect — it is what a tracing framework costs, and it
is invisible in the per-op comparison because both sides are compared at their
device time. It is also the one term where the reference has no analogue, and
`notes/17` §4.2 attributes exactly this to CUDA graphs on the hand-written side.

---

## 6. Classification of every gap

The four classes: **(i)** cannot express, **(ii)** can express but does not find,
**(iii)** not yet implemented, **(iv)** does not exist for this arch. Each gap
below carries the single cheapest experiment that would settle it if the
placement is ambiguous.

| # | gap | class | evidence | cheapest experiment that would settle it |
|---|---|---|---|---|
| 1 | **Fusing the unrolled DDA body at small tensor sizes** | **(ii)** | `launches = max_steps + 8` exactly, at and below 524 288 pixels, on both backends (§2.1). The unrolled op chain is in the graph — 1.05 MB of generated CUDA at `max_steps=32` — and one kernel per step of it is emitted. | Raise the fusion-size threshold for this one shape and re-run `census_main.py` at batch 256. If the count collapses toward 9, it is (ii) and a single heuristic parameter is the whole gap. |
| 2 | **The fusion decision is size-dependent, and flips between batch 512 and 1024** | **(ii)** | §2.2, and the T4's own timed runs: 40 kernels at 512, 9 at 1024. | Bisect the batch axis at 512, 640, 768, 896, 1024 and read off the exact pixel count where the decision flips, then find the constant in `schedule/` that it corresponds to. One sweep, one grep. |
| 3 | **The fused kernel defeats the register allocator** | **(ii)** | Regime B: 36 384 instructions, 168 registers, **20 932 B of spill**, against the reference's 776 / 28 / 0 (§4.1, §4.2b). The fusion is *achieved* and the result is worse than the unfused version. | `nvcc -maxrregcount=128` on the captured source. If the instruction count is unchanged and only the spill profile moves, the register allocator is the whole story; if both move, the fusion boundary is. |
| 4 | **No per-ray early exit in the masked formulation** | **(ii)**, partly **(i)** | `BSSY` 33 → 0; every ray runs all 32 steps against a measured mean of 26.4 (§4.2c). A per-ray trip count is *not* a fixed-size tensor, which is the structural half. | Express the DDA as a TinyGrad `While` over a `done` mask and re-run the census. If `max_steps` stops scaling the launch count, the mask was the binding constraint; if nothing changes, it was never the cost. |
| 5 | **Keeping 16 load chains in flight** | **(ii)** | The reference's 16 unrolled steps have an identical 38-instruction load-to-load spacing, re-derived in §4.0.1, and `notes/17` §7 shows that is what its throughput rests on. The generated code's per-step structure is legible in §3.2. | `DEBUG=2` with `codegen.py`'s launch recorder, and count the LDGs between consecutive `ISETP` in the fused kernel's SASS. `sass_matrix.py` already parses it; the question is whether the 32 iterations' loads are independent. |
| 6 | **The unsigned bounds fold** | **not a gap on this arch** | Three `ISETP.GE.U32.OR` against three signed compares — **both 3 instructions** on sm_75 (§3.2 finding 1). | Nothing worth running. Recorded because "the compiler lost an optimisation" is a claim to check before making it, and here the honest answer is that there was nothing to lose on this instruction set. The same rewrite with `.cast(dtypes.uint32) < wx` would confirm it in one line. |
| 7 | **4× redundant voxel load** | **(ii)** | Four predicated 1-byte loads of one address, per step (§3.2 finding 2); 517 global loads in the fused kernel against the reference's 16 (§4.2d). | Recompile the capture with `--fmad=false` and re-census. If the four loads merge, ptxas needed a cheaper proof; if not, the predicates are structurally distinct and the cost is the masked formulation. |
| 8 | **The camera basis, re-read at the top of every program** | **(ii)** | `data1_104` is 416 bytes read by all 524 288 threads in regime A. | It is a broadcast L1 read and is **not** in the traffic estimate in §4.3, so it is not a gap. Recorded to say so explicitly: the largest-looking term in the argument list is the cheapest one. |
| 9 | **Launch overhead and kernel count** | **not a gap here** | 9 launches vs 1: ~36 µs against 8754 µs. 40 vs 1: ~160 µs against 13 573 µs. Under 1.5% either way (§5.4). | Already settled, with the arithmetic shown. This is the negative result that matters: for this kernel on this card, launch count is not the cost, which is the opposite of the `notes/13` framing. |
| 10 | **Per-ray early exit / warp divergence** | **(i)** on this API | The generated code has no `goto`; there is no first-class way to say "this warp diverges here". `BSSY` 33 → 0. | None exists, and that is the finding. The cheap substitute is to *count* it: `sass_matrix.py`'s `bss_bsync_pairs` column does that on any capture, and the reference's 33 is the price of a per-ray exit. |
| 11 | **The host cost of tracing an unrolled loop** | **(ii)** | 211.9 ms of Python tracing against 8.754 ms of device time at batch 1024, a 24× ratio (§5.6). | `TinyJit` on the render and re-time the end-to-end number. If the trace cost disappears, it is (ii) and it is a one-word change. |
| 12 | **The colour lookup as a 14-row table rather than a branch** | **(i)**, deliberately accepted | §3.3. A tensor program has no `if` on a tensor, so "hit" and "sky" become one gather. | None, and none needed: it removes a branch and changes no traffic. Disclosed because it is the one place this implementation is shaped rather than naive, and a reader diffing it against the reference should know. |
| 13 | **Nothing sm_75-illegal in either side** | **not a gap** | `HMMA HGMMA LDGSTS REDUX ELECT BAR.RED F2FP` = 0 in the reference binary, gated on the card before anything is measured and again in the offline pre-flight. The generated code compiles for sm_75. | Already settled. Recorded because the KDA comparison could not run on sm_75 at all, so "both sides reach the card" is a precondition here and not a result. |

**The dominant gap, in one sentence: the unrolled DDA body is in the graph in
both regimes and the compiler makes the fusion decision badly in both — too
little fusion below ~1 M pixels (one kernel per step, state through global
memory) and too much above it (one kernel that hits 168 registers and spills
20 932 bytes) — so the gap is class (ii): representable, and findable by a cost
model that unrolls *and* blocks registers across the loop, but not by a heuristic
that only asks "is this cheap to fuse".**

---

## 7. What would change the answer, in the order I would do it

1. **One fusion heuristic, two bugs.** Gaps 1 and 2 are the same parameter: fuse
   more when small, and when large, fuse *into a register budget* rather than
   into a single kernel. A per-shape threshold is the cheapest experiment in
   this note and it is the difference between (ii) and (i) for the dominant gap.
2. **A register-pressure check on the fusion boundary.** Gap 3. The fused kernel
   is 47× the reference's instruction count and spills; a rule that splits a
   fusion when the live set exceeds the register file would have produced
   regime A's 17 clean programs at batch 1024 instead of one broken one — and
   regime A was 2× *worse* in wall clock, so the right answer is neither, it is
   "fuse up to the register budget and no further".
3. **A gather with "load once, many consumers" semantics.** Gap 7. Four
   predicated byte loads for one voxel is 4× the read instructions on the hot
   path, and it is a consequence of the masked formulation rather than of the
   optimiser.

None of the three is a tiling problem and none needs a new backend. That is what
makes the answer (ii) rather than (i) the useful one: the capability is there
and the search is not.

---

## 8. Files

| what | where |
|---|---|
| the tensor implementation | `minecu_tg/render.py` — read its docstring first |
| the correctness gate, CPU, no GPU | `minecu_tg/verify.py` |
| the census and the generated-source capture | `minecu_tg/codegen.py`, `census_main.py`, `capture_cuda.py` |
| the per-kernel table | `minecu_tg/analyse_codegen.py` |
| the SASS matrix, both sides, one categoriser | `minecu_tg/sass_matrix.py`, `sass_tg.sh` |
| the T4 run and its push recipe | `minecu_tg/kaggle/`; results in `out/minecu_tg/20260928-221859/` |
| the captured generated CUDA, sm_75 | `out/minecu_tg/codegen-sm75-b8/`, `out/minecu_tg/codegen-sm75-b1024/` |
| the SASS and `ptxas -v` reports, all 17 + all 6 kernels | `out/minecu_tg/sass-sm75-b8/out/`, `out/minecu_tg/sass-sm75-b1024/out/` |
| the §2.2 batch sweep, one dir per batch | `out/minecu_tg/census-cpu-b*/census.json` |
| the hand-written side, untouched | `minecu_ref/src/kernels.cu`, sha256 `c6ed8f06…` |
| the driver the T4 run uses for the reference | `minecu_tg/kaggle/refdrv.cu` |
| the five push failures and the gates they caused | `minecu_tg/README.md`, "the pre-flight gates" |

### 8.1 Re-deriving every SASS number in §4

The whole SASS matrix rebuilds from preserved sources on this host, with no
NVIDIA card and nothing installed outside the container:

```
# both sides' generated code -> sm_75 SASS + ptxas -v
PY="$PWD/.venv/bin/python" \
  ./minecu_tg/sass_tg.sh out/minecu_tg/codegen-sm75-b8/max_steps_0032    sm_75 out/minecu_tg/sass-sm75-b8
PY="$PWD/.venv/bin/python" \
  ./minecu_tg/sass_tg.sh out/minecu_tg/codegen-sm75-b1024/max_steps_0032 sm_75 out/minecu_tg/sass-sm75-b1024

# then, through the same categoriser on both sides
python3 minecu_tg/sass_matrix.py --sass minecu_ref/build/kernels_sm_75.sass \
    --res minecu_ref/build/resources_sm_75.txt --select render_kernel_uint8_prebasis
python3 minecu_tg/sass_matrix.py \
    --sass out/minecu_tg/sass-sm75-b1024/out/__kernel004_cu.sass \
    --res  out/minecu_tg/sass-sm75-b1024/out/__kernel004_cu.ptxas
```

which prints, for the fused regime-B kernel and the reference side by side:

```
E_512_2_8_8_4_4   instr=36384 reg=168 spill=20932 LDG=517 STG=4 FFMA= 52 BAR=0 BSSY=0
_Z28render_...    instr=  776 reg= 28 spill=    0 LDG= 17 STG=6 FFMA=  9 BAR=1 BSSY=33
```

The counts in §4.1 are `INSTR_RE` matches over each kernel's instruction
stream, and the register and spill figures are read from the `ptxas -v`
blocks in the same directories. The reference's 776 and its 16 loads at a
spacing of exactly 38 are re-derived here rather than inherited from
`notes/17`, because the §4.1 comparison rests on them:

```
instructions in the hot kernel:  776
LDG.E.S8* count: 16   opcodes: {LDG.E.S8.CONSTANT.SYS}
their instruction indices: 152 190 228 266 304 342 380 418 456 494 532 570 608 646 684 722
load-to-load spacings: 38 × 15, all equal
before the first load: 152   after the last: 53
```

The §4.3 "the two sides do not have the same shape" claim is a period scan over
the same two opcode streams — for each candidate period `P`, the length of the
longest run of `ops[i] == ops[i + P]`:

```
reference   n=  776   P=38  -> exact repeat run 569      (16 identical steps)
regime B    n=36384   no P in [20,200] gives a 32-fold repeat; best run 2943
```
