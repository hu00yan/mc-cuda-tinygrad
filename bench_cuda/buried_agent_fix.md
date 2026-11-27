# The buried agent: root cause, fix, and the numbers on both sides

## One-sentence root cause

`physics_kernel` collides the agent against a flat **analytic** plane at
`y = world_y / 4` (`kernels.cu:1566`) that never reads a voxel, while
`generate_world_kernel` puts the visible grass surface at `y = ground_height`
(`kernels.cu:1823`); at the shape the study first measured — `world=16`,
`ground=8` — those disagree by 4 blocks, so the agent comes to rest **inside
solid dirt** and `render_kernel_uint8_prebasis`, which tests the current voxel
*before* it steps (`kernels.cu:666`), hits on iteration 0 for every pixel.

## What the previous run actually measured

Results in `out/minecu-t4/20260928-145231/`, shape
`batch=32768 world=16^3 res=32x32 max_steps=32 ground=8`:

| launcher | kernels/step | us/step | M env-steps/s |
|---|---|---|---|
| uint8 prebasis full | 9 | 1824.8 | 17.96 |
| uint8 prebasis | 7 | 1824.1 | 17.96 |
| fp32 obs | 6 | 2807.0 | 11.67 |

Per-kernel medians (us): `precompute_camera_basis` 26.6, `render uint8
prebasis` 1751.0, `render uint8 minimal` 1990.7, `render uint8 fp16` 2109.1,
`render fp32` 2814.0.

Measured floors on that card: write 234.3 GB/s (73% of the 320.1 datasheet),
read 278.2 (87%). The uint8 step reached 128.8 GB/s = 40% of datasheet and the
obs write alone 55.2 GB/s = 17% of the write floor, so the kernel was
**DDA-latency/issue bound, not bandwidth bound** — which refutes the earlier
inference, made from a 3090 datasheet calculation, that the kernel was near the
write-bandwidth floor.

**But the measurement point was degenerate.** The step count was never
reported. Re-measured on the same card in the same shape, it is **1.291 mean
DDA steps per pixel, median 1**, against a 32-step cap, with 96.9% of pixels
hitting on their very first voxel test. The render kernel was doing one `__ldg`
and one store for all but 3% of pixels. 17.96 M env-steps/s is a valid
measurement of *this code in this state*; it is not a measurement of one
env-step of a voxel RL env.

## The fix: `ground_height = world_y / 4`

No kernel change. `kernels.cu` is byte-identical to upstream
(`sha256 c6ed8f06…`, re-checked by `build.sh` on every build and by the Kaggle
notebook before it compiles anything).

`bench/bench.cu` now derives the default ground from the world size:

```c
static int ground_of(const Cfg& c) { return c.ground_h >= 0 ? c.ground_h : c.W / 4; }
```

with `int ground_h = -1;` meaning "derive it". This is the only value at which
the two notions of ground coincide:

| | value |
|---|---|
| grass top face (`kernels.cu:1823`) | `y = ground_height` |
| physics floor (`kernels.cu:1566`) | `y = world_y / 4` |
| coincide iff | `ground_height == world_y / 4` |

It is also self-consistent at every world size (12% of the world edge at
W=16/32/48), so the harness cannot re-introduce the disagreement by changing
`--world` and leaving `--ground` at a constant.

### Why the kernel could not express this

It could not, and that is a finding rather than something to patch around.
`physics_kernel` has **no voxel collision anywhere**. It cannot stand the agent
on a hill, on a tree, or on a placed block — only on the plane at
`world_y / 4`. So "the reference cannot stand on its own world" is a real
limitation of the reference, and the honest response is to report it, not to
edit the vendored file to hide it. The harness aligns the config instead, and
`kernels.cu` is untouched.

There is a visible consequence in the fixed configuration: because nothing
stops an agent walking into a tree or the house, **1–3% of envs end a run with
their eye inside a log**. That is not a harness artefact; it is what this
physics model does. It is small enough not to affect the aggregate, and the
gate accounts for it (§ the new assertion).

## The new assertion: the DDA gate

The prior lesson from this project is that a cross-check which only *prints* is
a claim, not a check. So the step count is measured **on the device** and
**asserted**, and a failure means the run reports no timings at all.

`bench/bench.cu` adds `dda_probe_kernel`, a transcription of
`render_kernel_uint8_prebasis` (`kernels.cu:568-700`) with exactly one
difference: it counts. The DDA body, the `__ldg`, the bounds test and the step
order are the vendored macro verbatim; the tally replaces the early exit. It
reads the same device voxels and the same device basis the timed kernel reads.
The probe's arithmetic was cross-checked offline against the independent
`bench/dda_model.py` reference and agrees exactly: delta 0.000 at
world=16/ground=4, and the identical 1.000 for a fully-buried eye.

Two assertions, both real, both gating:

```c
check(s.mean >= floor_steps, "DDA step count is above the degeneracy floor", …);
check(eye_solid_frac < kEyeSolidMaxPct, "the eye is not inside solid rock (fraction)", …);
```

with

* `floor_steps = max(2.0, 0.10 * max_steps)` — an absolute clause a buried eye
  cannot pass at *any* cap (measured 1.291, and 1.000 in the idealised case
  where every eye is in rock), plus a clause that scales with the cap so raising
  `--max-steps` cannot buy a pass. Measured margins: 26.416 / 3.20 = 8.3× for
  the fixed config, 1.291 / 3.20 = 0.40× for the degenerate one.
* `kEyeSolidMaxPct = 20.0` — the fraction of pixels that hit on their first
  voxel test. Measured: 96.9% degenerate, 3.0% fixed. A fraction rather than
  `== 0` precisely because a few envs walk into the tree; demanding zero would
  fail the *fixed* config, and "fixing" that by weakening the mean floor would
  re-admit the original bug.

Failure ⇒ `g_fail++` ⇒ `selftest()` returns non-zero ⇒ `main()` prints
`selftest FAILED: no timings reported` and **exits 2 before the timing battery
runs**. The gate is re-checked after the settle that precedes the timing
section, so a configuration that drifts into degeneracy between the gate and
the tables also aborts.

`--measure-degenerate` exists solely to reproduce the pre-fix numbers for the
table below. It does not lower the floor and does not skip the check: it prints
the measured value, stamps `DEGENERATE` in the run header and in every timing
table, and is used exactly once in the whole battery.

## The new assertion caught two bugs in itself

Both were found by reading the T4 output rather than the source, and both are
worth recording because they are the failure modes this project has been bitten
by before.

**1. Two checks, one buffer.** The first version of the gate formatted both
detail strings into the same `char buf`, so the second `snprintf` silently
overwrote the first and both checks printed identical text — a cross-check that
looked like it was checking two things and was checking one. Fixed with two
buffers (`buf_steps`, `buf_eye`).

**2. A histogram fed one entry per block.** `g_dda_hist` was incremented from
`if (tid == 0)` using thread 0's own step count, so it recorded 262 144
entries instead of 33 554 432. The bucket totals summed to ~3% of the pixel
count, `at_cap` was wrong, and `p50`/`p90` came back unset — the only visible
symptom was two columns of `-1` next to a perfectly plausible mean. The mean
itself was correct, because `s_steps` is a block *sum* rather than a sample, so
the gate verdict was never wrong; but every distributional statistic printed
beside it was. Fixed with a per-block shared-memory histogram, plus an
invariant in `measure_dda` that the histogram total must equal the pixel count
and `exit(3)`s loudly if it does not. A statistic that cannot be silently
wrong is worth more than the statistic.

## Before and after, measured on the same T4, same run

Both rows below come from the **same Kaggle run** (same binary, same card, same
measured floors), not from an old log. Source:
`out/minecu-t4/20260928-154527/`, kernel version 7 of
`tenghuzhang/mc-minecu-t4-baseline`. All 13 battery entries exited `rc=0` with
`selftest: PASS (0 failures)` and no `[FAIL]` line anywhere in the run.

### The workload behind each number

| | buried (before) | on-surface (after) |
|---|---|---|
| shape | batch 32768, world 16³, res 32×32, cap 32 | identical |
| `ground_height` | 8 | 4 (= `world_y / 4`) |
| surface top face vs physics floor | 8 vs 4 — **disagree** | 4 vs 4 — **agree** |
| settled eye y | 5.60 | 5.60 |
| voxel row at the eye | 5 = **DIRT** | 5 = air |
| **mean DDA steps / pixel** | **1.291** (of a 32 cap) | **26.416** |
| **median (p50) DDA steps** | **1** | **32** |
| p90 DDA steps | 1 | 32 |
| pixels hitting their first voxel | 32 505 856 = **96.9%** | 1 004 544 = **3.0%** |
| pixels reaching the 32-step cap | 263 033 = 0.8% | 26 516 200 = **79.0%** |
| gate floor | 3.20 | 3.20 |
| verdict | **DEGENERATE** | PASS, 8.3× the floor |

The distribution, not just the mean, is the difference. Buried, the median ray
tests **one** voxel; on-surface, the median ray tests **32** — it exhausts the
budget and gives up. The render kernel was doing **20× less DDA work** than the
shape implies, and the shape of that work is entirely different, not merely
smaller.

### Throughput

| launcher | kern | buried us/step | buried M/s | on-surface us/step | on-surface M/s |
|---|---|---|---|---|---|
| `launch_step` (fp32 obs) | 6 | 3206.8 | 10.22 | 5439.4 | 6.02 |
| `launch_step_uint8_prebasis` | 7 | 2003.6 | 16.35 | 7198.9 | 4.55 |
| `launch_step_..._prebasis_full` | 9 | 2005.0 | 16.34 | 7197.8 | 4.55 |

### Per-kernel medians (us)

| kernel | buried | on-surface |
|---|---|---|
| `decode_actions_kernel` | 6.0 | 6.1 |
| `physics_kernel` | 6.1 | 6.2 |
| `raycast_interact_kernel` | 12.3 | 12.7 |
| `reset_kernel` | 4.1 | 4.1 |
| `precompute_camera_basis_kernel` | 26.1 | 27.0 |
| `render_kernel` (fp32 obs) | 3211.0 | 5306.4 |
| `render_kernel_uint8_minimal` | 2308.1 | 7227.1 |
| `render_kernel_uint8_fp16` | 2461.7 | 7456.2 |
| **`render_kernel_uint8_prebasis`** | **1968.1** | **7034.4** |

The five state kernels are unchanged, as they must be: the fix changes where
the agent stands, not what those kernels do. The four render kernels roughly
triple-to-quadruple, which is the DDA work appearing.

Run-to-run variance on this card is about 1% (Kaggle T4s are shared): a prior
run of the identical binary gave 7165.1 / 2052.1 us for the two uint8 whole
steps against 7198.9 / 2003.6 here, and reproduced the DDA statistics to the
digit (26.416 / 1.291, p50 32 / 1). The DDA numbers are deterministic given
the state; the timings are not, and the ~4× gap between the two configurations
is three orders of magnitude larger than that noise.

### Achieved bandwidth, against floors measured in the same run

Floors: write 234.5 GB/s, read 278.4 GB/s (on-surface); 234.1 / 278.1
(buried). Identical to within 0.2%, so the comparison is fair.

| quantity | buried | on-surface |
|---|---|---|
| u8 step, obs write alone | 50.2 GB/s = 21.5% of write floor | 14.0 GB/s = **6.0%** |
| u8 step, obs+voxel counted | 114.5 GB/s = 1.7% of 1:1 floor | 32.8 GB/s = **0.5%** |
| fp32 step, obs write alone | 125.6 GB/s = 53.6% of write floor | 74.0 GB/s = **31.6%** |
| `render_kernel_uint8_prebasis` | 119.3 GB/s = 51% of write floor | 33.4 GB/s = **14%** |

The obs write reaches **6% of the measured write floor** in the fixed
configuration. Whatever the earlier 3090-datasheet inference was, this is not a
bandwidth-bound kernel, and the on-surface numbers make that unambiguous rather
than arguable.

## Which configuration is the defensible baseline

**`headline_w16` — the on-surface configuration — is the baseline.** 4.55 M
env-steps/s is the honest number.

The buried 17.96 M is not a baseline for a compiler comparison, for a reason
that is not "it is a smaller number": the two configurations are *different
kernels*. A DDA that exits on iteration 0 and a DDA that marches 26 steps have
nothing in common in their inner loop, their memory access pattern, or their
latency chain. A TinyGrad-generated kernel scored against 17.96 M would be
scored against an obs store. A ratio there is a ratio of two stubs.

The on-surface number is slower and it is the one that costs 4× more DDA work,
so it is the harder and more informative target.

### Per configuration

Measured DDA means from the same run, for every entry in the battery:

| configuration | agent position | mean DDA | usable? |
|---|---|---|---|
| **world 16, ground 4 (new default)** | on the surface | **26.83** | **yes — baseline** |
| world 16, ground 4, `--max-steps 8` | on the surface | 7.32 | yes |
| world 16, ground 4, `--max-steps 64` | on the surface | 52.69 | yes |
| world 16, ground 4, 64×64 res | on the surface | 26.86 | yes |
| world 32, ground 8 (previous default) | on the surface | 26.35 | yes — coincidentally already correct, since 32/4 = 8 |
| world 48, ground 12 (new default) | on the surface | 21.21 | yes |
| **world 16, ground 8 (previous default)** | **3 blocks inside dirt** | **1.31** | **no — degenerate, gated** |

Note the DDA mean is stable at 26.3–26.9 across batch, resolution, and step
cap. That stability is the point: it says the gate is measuring a property of
the *configuration*, not an artefact of a particular shape, and that raising the
cap actually raises the work rather than changing the verdict.

The world-48 inversion is worth stating explicitly: the disagreement does not
always bury the agent, it sometimes **floats** it. At `world=48, ground=8` the
spawn (10) is below the physics floor (12), so the clamp pushes the agent *up*
and gravity becomes unobservable — a different failure from the same root
cause, and one the new default is immune to because it recomputes the ground
from the world size rather than hardcoding it.

## AABB early termination: confirmed inert

The claim was that the `AABB early termination` credited at `kernels.cu:84` is
inert. **Both halves are confirmed, and the finding is stronger than claimed.**

**Half 1 — the timed kernel contains zero calls.** The timed kernel is
`render_kernel_uint8_prebasis`, which never calls `ray_intersects_aabb` at all.
The SASS settles it: `FMNMX` (the min/max op the AABB is built from) is **0** in
that kernel and in the four other non-calling render kernels, and **11** in
exactly the three that call it.

| kernel (sm_75) | instrs | FMNMX | calls AABB |
|---|---|---|---|
| `render_kernel_uint8_fast` | 1296 | 11 | yes |
| `render_kernel_uint8` | 1240 | 11 | yes |
| `render_kernel` (fp32) | 1224 | 11 | yes |
| `render_kernel_uint8_fp16` | 824 | 0 | no |
| `render_kernel_uint8_minimal` | 808 | 0 | no |
| **`render_kernel_uint8_prebasis`** | **776** | **0** | **no** |
| `render_kernel_uint8_smem` | 760 | 0 | no |
| `render_kernel_uint8_coalesced` | 696 | 0 | no |

**Half 2 — the three call sites can never fire, and the reason is stronger
than "the eye is inside the box".** The eye is strictly interior to
`[0, ws]³` for 100.0000% of (step, env) pairs at W=16/32/48 (2048 envs × 240
steps). A ray starting strictly inside a box always intersects it, so
`ray_intersects_aabb` returns `true` unconditionally. Worse, its other output is
`t_entry = tmin ≤ 0`, so the guard

```c
if (!hits_world || t_entry > view_distance)   // kernels.cu:170, :1344
```

reduces to `t_entry > 32`, which is `≤ 0 > 32` — **never true**. The
early-out is not merely rarely taken; it is unreachable, and the ~90
instructions it costs per pixel in those three kernels are pure overhead.

**What this changes about the 17.96 M number.** It does not change it — the
timed kernel never paid for the AABB in the first place. The credit at
`kernels.cu:84` ("Optimizations: AABB early termination, `#pragma unroll 4`,
`__ldg()`") should be read as: `#pragma unroll 4` and `__ldg()` are real and
present in the timed kernel; **AABB early termination is not**. The comment is
a header describing the file, not a per-kernel changelog.

## What is in the battery now

13 entries, all gated:

* `selftest_w16 / w32 / w48` — correctness **and** the DDA gate, at three
  world sizes. The new default makes all three self-consistent without a
  special case. Measured means 26.86 / 26.35 / 21.21.
* `headline_w16` — **the baseline.** No `--ground`, so it takes the derived
  `world_y/4`. Mean 26.83.
* `sweep_w16`, `batch4k/8k/16k_w16`, `w32_32px`, `w16_64px`,
  `w16_ms8`, `w16_ms64` — the throughput sweep, all on the fixed config, all
  between 7.32 and 52.69 mean steps depending only on the step cap.
* `degenerate_w16_ground8` — the before case, re-measured on this card in this
  run, with `--ground 8` and `--measure-degenerate`. Mean 1.31. Its timings are
  stamped `DEGENERATE` by the binary itself.

All 13 entries exited `rc=0`; every selftest reported `PASS (0 failures)`; no
`[FAIL]` line appears anywhere in the run.

The sm_75 SASS census gate is unchanged and still runs, from the same run's own
binary: `arch = sm_75`, `HMMA=0 HGMMA=0 LDGSTS=0 REDUX=0 ELECT=0 BAR.RED=0
F2FP=0`, verdict `no sm_75-illegal instructions`. The probe kernels live in
`bench.cu` and are not part of that census, which covers the vendored
`kernels.cu` object — the right scope, since the gate exists to certify the
*reference's* SASS for sm_75.

## Summary for a compiler comparison

Compare against **`headline_w16`: 4.55 M env-steps/s at a mean of 26.416 DDA
steps per pixel (median 32, 79% of pixels at the cap)**, per-kernel
`render_kernel_uint8_prebasis` **7034.4 us**.

Do not compare against 17.96 M. That figure is real, measured, and reproducible
— and it is the throughput of a kernel that never marched.
