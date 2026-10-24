# 17 — mine.cu: the hand-written side, specified, on sm_75

**What this is.** A complete specification of the hand-written CUDA in
`netherite/mine.cu` (a clone of `Infatoshi/mine.cu`, commit `ec2085e`), written
so that the computation can be reimplemented from this document alone in a
different framework. It is the spec the TinyGrad lane is written against.

Labels used throughout:

- `MEASURED` — an instrument ran here, on this host, on the named artifact.
- `[ARTIFACT]` — read out of a named file rather than recomputed.
- `MODELLED` — arithmetic on shapes and on hardware constants this host does not
  have. Not a timing.
- `CLAIM` — the author's number, not reproduced here.

**Three facts up front, because they change what the comparison can be.**

1. **`kernels.cu` compiles for sm_75, cleanly, with zero register spills and no
   instruction that a T4 cannot execute.** `MEASURED` `minecu_ref/build.sh sm_75`.
   The full census is in `minecu_ref/build/sass_census_sm_75.txt`. The
   difference from flash-linear-attention's KDA — which needs `HMMA`, so sm_80+
   and cannot run on a T4 at all — is that a voxel DDA has no tensor-core
   requirement. **The free Kaggle T4 is a valid comparison card.** §6.
2. **The hot kernel is the ray-marching render, and it is not
   bandwidth-bound.** It is bound by a serial dependent chain of one-byte global
   loads, and on the project's own headline shape the mean DDA length is 23.3
   steps out of a 32 cap. §4, §5.
3. **The 49.6M env-steps/sec headline is not reachable from this source at the
   shape it is quoted for.** An instruction-issue bound — generous to the claim,
   since it assumes the SM issues DDA SASS every cycle with no stalls — allows
   a mean of 3.8 DDA steps per pixel; the kernel on the world the project's own
   worldgen produces takes 23.3. `MODELLED` + `MEASURED` §7. This is a bound
   and not a measurement, and only a measured RTX 3090 number settles it. What
   it does establish is that a TinyGrad number must **not** be divided by 49.6M,
   because the two would not be measuring the same thing.

---

## 0. The artifact

`netherite/mine.cu/src/kernels.cu`, 3,245 lines, sha256
`c6ed8f06ee64d40a8a8ab7ebce3b3ceea1e3ac93f9ee50cfc1e63d3f090d878f`, remote
`https://github.com/Infatoshi/mine.cu`. `[ARTIFACT]`

Vendored byte-identical at `minecu_ref/src/kernels.cu`, same hash. `MEASURED`
(`minecu_ref/build.sh` re-checks this on every run and prints both hashes in
`build/MANIFEST.txt`, so a silent edit to the reference is caught rather than
measured.)

`src/bindings.cpp`, 1,269 lines, sha256 `53b0d8c7…`, is the pybind11/PyTorch
side. **It is not needed for anything below and is not vendored.** Every
launcher in `kernels.cu` is `extern "C"` and takes raw pointers;
`minecu_ref/bench/bench.cu` declares them itself and links only `libcudart`. No
libtorch, no Python, no JIT extension build. That is what makes the baseline
reproducible in a lean CUDA container.

### 0.1 Structure

| lines | what |
|---|---|
| 27–44 | two colour LUTs: `BLOCK_COLORS[13][3]` float32, `SKY_COLOR[3]` float32 |
| 51–79 | `ray_intersects_aabb`, the slab test |
| 87–293 | `render_kernel` — fp32 obs, AABB, fog, face shading |
| 300–470 | `render_kernel_uint8` — same, uint8 out |
| 479–495 | `BLOCK_COLORS_U8[13][3]`, `SKY_COLOR_U8[3]` — the pre-quantised LUT |
| 511–561 | `precompute_camera_basis_kernel` — trig once per camera |
| 568–700 | `render_kernel_uint8_prebasis` — **the uint8 hot path** |
| 709–826 | `render_kernel_uint8_coalesced` — `[Y,X,Z,B]` voxel layout |
| 835–982 | `render_kernel_uint8_smem` — whole world in shared memory |
| 985–1119 | `render_kernel_uint8_minimal` — no AABB, no fog |
| 1127–1260 | `render_kernel_uint8_fp16` — fp16 DDA accumulator |
| 1268–1442 | `render_kernel_uint8_fast` — ground-plane early-out |
| 1459–1504 | `decode_actions_kernel` |
| 1511–1584 | `physics_kernel` |
| 1591–1680 | `raycast_interact_kernel` — break and place |
| 1687–1753 | `raycast_break_kernel` — legacy, superseded |
| 1760–1786 | `reset_kernel` |
| 1793–1946 | `generate_world_kernel`, `place_tree_kernel`, `place_house_kernel` |
| 1954–2069 | `episode_check_kernel`, `episode_update_kernel`, two masked worldgen |
| 2076–3245 | 25 `extern "C"` launchers, 3 CUDA-graph states, 8 step variants |

There are **8 render kernels**. They are the same algorithm at 8 different
points on a trade curve, and which one the headline used is **not recorded
anywhere in the repo** — `docs/benchmarks.md` and
`scripts/plot_throughput.py` give numbers with no `obs_dtype` and no
`obs_dtype`-equivalent recorded, and `minecu/__init__.py:164` defaults to
`obs_dtype="float32"`, which selects `render_kernel`. §7 carries both readings.

---

## 1. State, shapes, dtypes, memory layout

Everything lives in device memory. There is no host round trip in the step.
`[ARTIFACT]` `minecu/__init__.py:214-256`.

| tensor | shape | dtype | bytes/env | written by |
|---|---|---|---|---|
| `voxels` | `[B, WY, WX, WZ]` | `int8` | `WX·WY·WZ` | worldgen, raycast_interact |
| `positions` | `[B, 3]` | `float32` | 12 | physics, reset |
| `velocities` | `[B, 3]` | `float32` | 12 | physics, reset |
| `yaws` | `[B]` | `float32` | 4 | physics, reset |
| `pitches` | `[B]` | `float32` | 4 | physics, reset |
| `on_ground` | `[B]` | `bool` | 1 | physics, reset |
| `cameras` | `[B, 5]` | `float32` | 20 | `update_cameras_kernel` |
| `basis` | `[B, 14]` | `float32` | 56 | `precompute_camera_basis_kernel` |
| `obs_buffer` | `[B, H, W, 3]` | `float32` **or** `uint8` | `12·H·W` or `3·H·W` | render |
| `buttons` (in) | `[B, 8]` | `int8` | 8 | caller |
| `look` (in) | `[B, 2]` | `float32` | 8 | caller |
| `do_reset` | `[B]` | `bool` | 1 | episode_check, caller |
| `step_count` | `[B]` | `int32` | 4 | episode_update |
| `rewards` | `[B]` | `float32` | 4 | raycast_interact |
| decode scratch | `[B]` × 4 fp + `[B]` × 3 bool + `[B]` fp | | 23 | `decode_actions_kernel` |

**The obs dtype is the single biggest performance lever in the project and it
is a factor of 4 in bytes.** At `B=32768, 32×32`: 384 MiB per step for fp32,
96 MiB for uint8. Any comparison must state which one it is on.

### 1.1 Voxel layout, precisely

`[B, Y, X, Z]`, contiguous, so the flat index is

```
idx = b·(WX·WY·WZ) + y·(WX·WZ) + x·WZ + z
```

`MEASURED` by reading every index expression in the file; they agree.

- `kernels.cu:230` — `render_kernel`, uses `ws3/ws2` (cubic assumption)
- `kernels.cu:669` — `render_kernel_uint8_prebasis`, uses `wxz = wx*wz` and
  `wxyz = wx*wy*wz`, **and the comment at line 635 says "Y stride for [Y,X,Z]
  layout" while indexing `voxel_y * wxz + voxel_x * wz + voxel_z`.** With
  `wxz = WX·WZ` that is `y·(WX·WZ) + x·WZ + z`, which *is* `[Y,X,Z]`. The
  comment and the code agree; the comment is just confusingly worded. The
  non-cubic support added in commit `ec2085e` is real.
- `kernels.cu:794` — `render_kernel_uint8_coalesced` is the exception:
  `[Y, X, Z, B]`, index `y·ws²·B + x·ws·B + z·B + b`. It is the only kernel
  designed for cross-batch read coalescing. It is **not wired into any step
  path** — `launch_render_uint8_coalesced` (kernels.cu:2213) is exposed to
  Python as a standalone benchmark entry only. The step functions never call
  it, so no `49.6M` number can have come from it.

Block IDs: `-1` = air, `0` grass, `1` dirt, `2` stone, `3` oak log, `4` leaves,
`5` sand, `6` water, `7` glass, `8` brick, `9` cobblestone, `10` planks, `11`
snow, `12` bedrock. `[ARTIFACT]` `kernels.cu:11-24`.

---

## 2. The hot path, step by step

This is the whole-step sequence for the fastest step variant,
`launch_step_uint8_prebasis_full` (`kernels.cu:3081`), 9 kernels.
`[ARTIFACT]` `kernels.cu:2909-2991`.

```
 0  episode_check_kernel        step_count >= episode_length -> do_reset
 1  decode_actions_kernel       buttons[8] + look[2] -> control signals
 2  physics_kernel              control signals -> pos, vel, yaw, pitch
 3  raycast_interact_kernel     break/place, writes rewards, mutates voxels
 4  reset_kernel                do_reset -> spawn, zero vel, zero yaw/pitch
 5  episode_update_kernel       step_count++ or =1
 6  update_cameras_kernel       pos,yaw,pitch -> cameras[5]
 7  precompute_camera_basis_kernel cameras[5] -> basis[14]
 8  render_kernel_uint8_prebasis  basis + voxels -> obs[H,W,3] uint8
```

`launch_step_uint8_prebasis` (`:2993`) is the same minus 0 and 5.
`launch_step` (`:2497`) is 6 kernels, no basis kernel, fp32 obs.

**Data dependencies are a chain, not a DAG.** Each kernel depends on the
previous one's output:

```
buttons,look ──► decode ──► physics ──► raycast_interact ──┐
                             │                            │ both read pos/yaw/pitch
                             └──────────► reset ─────────┘
                                          │
                                          ▼
                                    update_cameras ──► precompute_basis ──► render
                                                                          ▲
                                            voxels ◄───────────────────────┘
```

`raycast_interact` reads `positions`/`yaws`/`pitches` *after* physics wrote
them, so it aims from the post-move eye. `reset` then may overwrite them, so
the interaction uses the pre-reset pose while the render uses the post-reset
one. `render` reads the voxel grid *after* `raycast_interact` may have broken a
block, so a broken block is not visible in the observation of the step that
broke it. `[ARTIFACT]` these are the argument orders in the launch sequence.

### 2.1 `decode_actions_kernel` — `kernels.cu:1459`

One thread per env, 256 threads/block, `gridDim = ceil(B/256)`.

Button indices: 0 forward, 1 backward, 2 strafe_left, 3 strafe_right, 4 jump,
5 break, 6 place, 7 sprint. `[ARTIFACT]` `kernels.cu:1450-1457`.

```
forward_in     = (float)buttons[0] - (float)buttons[1]     // +1, -1, or 0
strafe_in      = (float)buttons[3] - (float)buttons[2]     // +1 right, -1 left
delta_yaw_in   = look[0]
delta_pitch_in = look[1]
jump_in        = buttons[4] != 0
do_break       = buttons[5] != 0
do_place       = buttons[6] != 0
speed_mult     = (buttons[0] && buttons[7]) ? sprint_multiplier : 1.0f
```

`buttons` is multi-hot `int8`, so `fwd` and `back` can both be 1 and give 0.
`speed_mult` requires **forward**; sprinting sideways does nothing. 64 SASS
instructions, 30 registers, no spills. `MEASURED`

### 2.2 `physics_kernel` — `kernels.cu:1511`

One thread per env, 256/block. 88 SASS instructions, 26 registers.
`MEASURED`

```
yaws[i]   += delta_yaw_in[i]
pitches[i] = clamp(pitches[i] + delta_pitch_in[i], -1.56f, +1.56f)

cos_yaw = cosf(yaws[i]);  sin_yaw = sinf(yaws[i])
speed   = walk_speed * speed_mult[i]
move_x  = sin_yaw*fwd*speed*dt + cos_yaw*str*walk_speed*dt
move_z  = cos_yaw*fwd*speed*dt - sin_yaw*str*walk_speed*dt

vel_y = velocities[i][1]
if (jump_in[i] && on_ground[i]) vel_y = jump_vel
vel_y += gravity * dt

new = positions[i] + (move_x, vel_y*dt, move_z)

ground_y = world_y / 4.0f                      // an INTEGER division
if (new.y <= ground_y) { new.y = ground_y; vel_y = 0; on_ground[i] = true; }
else                  { on_ground[i] = false; }

margin = 1.0f
new.x = clamp(new.x, margin, world_x - margin)
new.z = clamp(new.z, margin, world_z - margin)
```

Four things a reimplementation must copy exactly, because each is observable:

- **`ground_y = world_y / 4` is integer division** (`kernels.cu:1566`), so for
  `world=16` the floor is at `y = 4`, not `y = 4.0` by any float route. For
  `world=48` it is `y = 12`. `ground_height` (default 8) is what the *worldgen*
  uses to place grass, and **the two disagree for every world size**: at
  `world=16` the renderer draws grass at `y=7` but physics floors the agent at
  `y=4`, so the agent is buried 3 voxels. `[ARTIFACT]`, both expressions read
  directly. `MODELLED` consequence: 4 of the 8 documented throughput rows are
  measuring agents inside solid rock, which changes the DDA and therefore the
  step count.
- **Strafing is not scaled by `speed_mult`.** `move_x`'s second term uses
  `walk_speed` directly (`:1548`), so sprint speeds up forward motion only.
- **`velocities[i][0]` and `[2]` are never written.** `vel` is read at index 1
  and written back at index 1; x and z velocity are dead. `reset_kernel` zeroes
  all three.
- **There is no collision against the voxel grid.** The ground is a plane at
  `world_y/4` and the walls are a clamp. `physics_kernel` does not read
  `voxels` at all — its signature has no voxel pointer. So "physics" here is
  not Minecraft physics, it is a 3-line integrator and two clamps.

### 2.3 `raycast_interact_kernel` — `kernels.cu:1591`

One thread per env, 256/block. 240 SASS instructions, 29 registers. `MEASURED`
**This is not a DDA.** It is a fixed-step ray march:

```
rewards[i] = 0.0f
if (!(do_break[i] || do_place[i])) return

dir = (sin_yaw*cos_pitch, sin_pitch, cos_yaw*cos_pitch)     // == forward, no FOV
eye = (pos.x, pos.y + 1.6, pos.z)
reach = 4.0f

prev = (-1, -1, -1)
for (step = 0; step < 32; ++step) {
    t = 0.1f + step * (reach / 32.0f);       // 0.1 .. 3.975, uniform, 0.125 spacing
    p = eye + dir * t
    v = floor(p)
    if (v in bounds) {
        if (voxels[v] >= 0) {                            // hit
            if (do_break) {
                if (target_block < 0 || block == target_block) rewards[i] = reward_value;
                voxels[v] = AIR;
            } else if (do_place && prev.x >= 0) {
                voxels[prev] = DIRT;                     // always DIRT
            }
            return;
        }
        prev = v;
    }
}
```

Points a reimplementation must note: the ray starts at `t = 0.1`, not 0, so the
agent's own voxel is never hit. The 32 samples are **uniformly spaced in `t`**,
not voxel-walking, so it can step over a 1-voxel-thick leaf block entirely.
`reach` is hardcoded 4.0, not a parameter. `target_block < 0` means "reward for
any block". `do_place` writes `DIRT` always, ignoring `target_block`. Break
takes precedence over place when both are held.

### 2.4 `reset_kernel` — `kernels.cu:1760`

One thread per env. 40 SASS instructions. `MEASURED` Masked: writes only where
`do_reset[i]`. Sets position to spawn, all three velocities to 0, yaw and pitch
to 0, `on_ground` to **false** (so the first step after a reset always applies
gravity, and a jump on step 1 is ignored because `on_ground` is false).

### 2.5 `update_cameras_kernel` — `kernels.cu:2320`

32 SASS instructions — the smallest kernel in the step. `MEASURED` A pure
gather: `cameras[i*5+{0,1,2,3,4}] = (pos.x, pos.y, pos.z, yaw, pitch)`. It is
a separate launch purely to convert a 3-stride layout into a 5-stride one so the
render kernel's camera loads are five consecutive `LDG`s. It has **no `extern
"C"` launcher of its own** and is only reachable inside a step path, so a
standalone per-kernel benchmark cannot time it. `minecu_ref/bench/bench.cu`
reproduces its write on the host for the render probes and says so.

### 2.6 `precompute_camera_basis_kernel` — `kernels.cu:511`

One thread per **camera**, 256/block. 56 SASS instructions. `MEASURED` The
single most effective optimisation in the file for its cost, and the one a
compiler would never invent: it hoists four `sinf`/`cosf` calls and one `tanf`
out of the per-pixel path.

```
basis[b*14 + 0..2]  = (cam.x, cam.y + 1.6, cam.z)        // the eye
basis[b*14 + 3..5]  = (sin_yaw*cos_pitch, sin_pitch, cos_yaw*cos_pitch)   // forward
basis[b*14 + 6..7]  = (cos_yaw, -sin_yaw)                // right.x, right.z; right.y == 0
basis[b*14 + 8..10] = (-sin_yaw*sin_pitch, cos_pitch, -cos_yaw*sin_pitch) // up
basis[b*14 + 11]    = half_fov * (width/height)           // scale_u
basis[b*14 + 12]    = half_fov                            // scale_v
basis[b*14 + 13]    = unwritten padding
```

`__sincosf` (`:530-531`) is the fast intrinsic, so these are approximate, not
IEEE. `right.y` is dropped because it is identically 0 — a 3-vector basis
stored in 2 slots, which a compiler would have to be told. The camera-basis
convention is **yaw 0 looks along +z** (`forward = (sin0·cos p, sin p, cos0·cos
p) = (0, sin p, cos p)`), not along −z as in OpenGL.

### 2.7 `render_kernel_uint8_prebasis` — `kernels.cu:568` — **THE HOT KERNEL**

776 SASS instructions, 28 registers, 64 B static smem, 1 barrier, **0 spills**.
`MEASURED` `minecu_ref/build/sass_census_sm_75.txt`, `resources_sm_75.txt`.

#### Grid and thread mapping

```c
dim3 grid(blocks_per_batch, batch_size);   // x = pixel block, y = env
dim3 block(128);
```

So **`blockIdx.y` is the environment** and `blockIdx.x` walks that env's
pixels. This is the "one block per env's slice of pixels" mapping, and it is
what makes the shared-memory basis load possible: every thread in a block
belongs to the same env, so the basis is a per-block invariant.
`pixels_per_batch = H*W`; `blocks_per_batch = ceil(H*W/128)`. At `H=W=32`,
`H*W = 1024` and `blocks_per_batch = 8`, so the grid is **`(8, 32768)`** =
262,144 blocks of 128 threads, all 128 busy. `[ARTIFACT]`
`kernels.cu:2858-2861`, arithmetic above.

#### The basis load into shared memory

```c
__shared__ float s_basis[14];
if (threadIdx.x < 14) s_basis[threadIdx.x] = basis[b*14 + threadIdx.x];
__syncthreads();
```

One `BAR.SYNC` per block. `MEASURED` (the SASS has exactly two `BAR.SYNC` in
the whole file, one in this kernel and one in the smem variant). Note this is
**not** a vectorised load: 14 scalar `LDG`s guarded by a thread-id compare, then
a barrier. Every other thread in the block waits on those 14.

#### Ray construction, per thread

```
b           = blockIdx.y
pixel_offset = blockIdx.x*128 + threadIdx.x
if (b >= batch_size || pixel_offset >= H*W) return
py = pixel_offset / width;  px = pixel_offset % width     // integer div by a
                                                            // runtime value: ~20 SASS

u = (2*(px+0.5)/width  - 1) * scale_u
v = (1 - 2*(py+0.5)/height) * scale_v

ray = forward + u*right + v*up        // right.y is 0, so ray.y = forward.y + v*up.y
ray *= rsqrt(dot(ray,ray))            // rsqrtf, one MUFU.RSQ
```

All `float32`. `rsqrtf` is a hardware approximation, so the ray is unit length
to about 2⁻²², not exactly. The `/width` and `/height` are **integer divisions
by runtime values**; the SASS expands each to a `MUFU.RCP` plus a
Newton-refinement sequence, visible at `build/kernels_sm_75.sass` instructions
20–42. That is ~23 instructions of address arithmetic per pixel before any
ray marching, and it is 15% of the kernel's static size.

#### The DDA

`DDA_STEP_PRE`, `kernels.cu:664-679`, transcribed:

```
DDA_STEP_PRE:
  if (step >= max_steps) goto done;
  if ((unsigned)vx < (unsigned)wx && (unsigned)vy < (unsigned)wy && (unsigned)vz < (unsigned)wz) {
      int8_t block = __ldg(&voxels[b*wxyz + vy*wxz + vx*wz + vz]);
      if (block >= 0) { hit_block = block; goto done; }
  }
  if      (t_max_x < t_max_y && t_max_x < t_max_z) { t_max_x += t_delta_x; vx += step_x; }
  else if (t_max_y < t_max_z)                     { t_max_y += t_delta_y; vy += step_y; }
  else                                             { t_max_z += t_delta_z; vz += step_z; }
  ++step;

#pragma unroll 4
for (int i = 0; i < max_steps; i += 4) { DDA_STEP_PRE(); DDA_STEP_PRE();
                                          DDA_STEP_PRE(); DDA_STEP_PRE(); }
```

Standard Amanatides–Woo grid traversal. Initialisation, `kernels.cu:638-659`:

```
v0     = (floor(eye.x), floor(eye.y), floor(eye.z))
step   = (ray.x >= 0, ray.y >= 0, ray.z >= 0) ? (+1,+1,+1) : (-1,-1,-1)
inv    = (ray == 0) ? 1e30f : 1.0f/ray
t_delta = |inv|
t_max   = (ray >= 0) ? (floor(eye)+1 - eye) * inv
                     : ( floor(eye)     - eye) * inv
```

The bounds test is the unsigned trick: `(unsigned)vx < (unsigned)wx` is one
`ISETP.GE.U32.OR` doing two comparisons by unsigned wraparound, instead of two
signed compares. `MEASURED` the SASS has exactly three such instructions per
unrolled step.

**The `#pragma unroll 4` on a loop that already advances by 4 gives 16 steps
per outer iteration, and ptxas unrolls all 16 statically.** `MEASURED` The
SASS body between the first and last `LDG.E.S8` is 608 instructions containing
exactly 16 `LDG.E.S8.CONSTANT.SYS`, and the **load-to-load spacing is exactly
38, fifteen times, with no variation.** That 38 is the per-step cost, measured
rather than estimated, and it is the number §4 and §7 turn on.

So the structure is: 16 independent DDA steps, each with its own global load,
laid out so the loads can all be in flight at once. That is the hand-written
part. A compiler unrolling a loop it is handed would produce 4 or 8; the author
asked for 16 and ptxas obeyed.

#### The store — and why it is the thing to look at

```
done:
  if (hit_block >= 0) output[out_idx+0..2] = BLOCK_COLORS_U8[hit_block][0..2];
  else                output[out_idx+0..2] = SKY_COLOR_U8[0..2];
```

`out_idx = ((long long)b*H*W + py*W + px) * 3`.

**Three separate 1-byte stores per pixel.** `MEASURED` The SASS is three
`STG.E.U8.SYS` at `[R12]`, `[R12+0x1]`, `[R12+0x2]`, fed by three scalar
`LDC.U8` from `c[0x3][...]` — a `__constant__` LUT read, not an `LDG`. There is
no `STG.E.U8.v2` or `.v4` anywhere in the file.

Consequence, and it is the single most important thing in this kernel for a
compiler comparison: **a warp of 32 threads writing one byte each to
consecutive addresses 3 apart touches 96 bytes, so each of the three store
instructions covers 3× the 32-byte sectors it writes.** Consecutive lanes hit
the same sector with different bytes, and the sectors get merged in L2 rather
than at the store. Whether the obs write costs 1× or 3× the compulsory traffic
depends on L2 write-combining, and that is a **hardware** question, not a
source-level one. A compiler emitting `output[idx] = colour` from a
`[B,H,W,3]` tensor produces exactly the same three byte stores, because the
`[B,H,W,3]` layout invites nothing else. **This is a case where the compiler is
right and the layout is the constraint** — and the author's own
`render_kernel_uint8_coalesced` is the attempt to escape it, by transposing the
*voxel* layout to `[Y,X,Z,B]` so reads coalesce across batch instead. That
kernel is never called from any step path.

### 2.8 World generation — `kernels.cu:1793`

One thread per **voxel**, 256/block, `gridDim = ceil(B·WX·WY·WZ/256)`.

```
if      (y == 0)             block = BEDROCK
else if (y < ground_height-1) block = DIRT
else if (y == ground_height-1) block = GRASS
else                          block = AIR
voxels[idx] = block
```

Flat, deterministic, and **identical for every env** — the `seed` parameter is
accepted and never used. At `world=16, ground=8` that is 2,086 of 4,096 voxels
solid (50.9%). `MEASURED` `minecu_ref/bench/stepcount_prebasis.json`.

Then `place_tree_kernel` (`:1831`) puts 3 `OAKLOG` at `y = ground+dy` and one
`LEAVES` at `ground+3`, at `(spawn_x, spawn_z+2)`, one thread per env, and
`place_house_kernel` (`:1868`) a 5×4×4 house at `(spawn_x+6, spawn_z-2)`, one
thread per env, ~80 stores. `MEASURED` 680 SASS instructions, the largest
non-render kernel, because it is a fully unrolled nest of 5×4×4 stores.

**At `world=16` the house does not fit.** `house_x = 8+6 = 14`, and the house is
5 wide, so `dx ∈ {0,1}` land and `dx ∈ {2,3,4}` are dropped by the bounds check;
`house_z = 6`, 4 deep, all fit. Two of the documented throughput rows
(`16³, 32px` and `16³, 64px`, the peak row) therefore measure a world with a
half-house. `MEASURED` by evaluating the bounds check in
`minecu_ref/bench/dda_model.py:build_world`.

---

## 3. Dataflow, restated as something a framework could express

For a reimplementation, the whole step is:

```
buttons[B,8] int8, look[B,2] f32, do_reset[B] bool
  │
  ├─ decode ──► forward_in, strafe_in, d_yaw, d_pitch, jump, do_break, do_place, speed_mult   (all [B])
  │
  ├─ physics(forward_in, …, pos, vel, yaw, pitch, on_ground) ──► pos, vel, yaw, pitch, on_ground
  │
  ├─ raycast_interact(pos', yaw', pitch', do_break, do_place, voxels) ──► rewards[B], voxels
  │
  ├─ reset(do_reset) ──► pos, vel, yaw, pitch, on_ground
  │
  ├─ cameras = stack(pos, yaw, pitch, axis=-1)                                  # [B,5]
  ├─ basis   = camera_basis(cameras, fov, W, H)                                  # [B,14]
  └─ obs     = render(voxels, basis, W, H, max_steps)                            # [B,H,W,3] uint8
```

Everything in that chain is elementwise or a scan over the env index, plus one
gather. There is no matmul, no reduction, no tensor-core-shaped operation
anywhere in the whole step. **The 3,245 lines of hand-written CUDA contain zero
tensor-core instructions on any architecture** — `HMMA=0, HGMMA=0, IMMA=0,
MMA=0` across the entire object. `MEASURED`
`minecu_ref/build/sass_census_sm_75.txt`. §6.

---

## 4. Which kernel dominates

**The render kernel. Not close.**

Per-step traffic and instruction counts at the headline shape
`B=32768, world=16, 32×32, max_steps=32`. `MODELLED` on `MEASURED` shapes and
`MEASURED` SASS.

| kernel | SASS instr | regs | bytes touched/step | why it is or is not the bottleneck |
|---|---|---|---|---|
| `render_kernel_uint8_prebasis` | **776** (152 prologue + 38/step + 54 epilogue) | 28 | 96 MiB obs + voxel reads | the whole cost |
| `render_kernel` (fp32) | 1224 (243 + 59.6/step + 87) | 33 | 384 MiB obs + reads | 1.6× the instructions and 4× the bytes |
| `render_kernel_uint8_fast` | 1296 | 32 | 96 MiB | a ground test that saves ~1 DDA step |
| `render_kernel_uint8_minimal` | 808 | 25 | 96 MiB | no AABB, no fog, same DDA |
| `render_kernel_uint8_fp16` | 824 | 26 | 96 MiB | see below: it is a pessimisation |
| `render_kernel_uint8_smem` | 760 | 38 | 96 MiB + 4 KB/env smem | only viable at small world |
| `precompute_camera_basis_kernel` | 56 | 32 | 1.75 MiB | nothing |
| `update_cameras_kernel` | 32 | 18 | 0.6 MiB | nothing |
| `physics_kernel` | 88 | 26 | 0.4 MiB | nothing |
| `decode_actions_kernel` | 64 | 30 | 0.5 MiB | nothing |
| `raycast_interact_kernel` | 240 | 29 | scattered 1-byte | nothing, early-exits 7/8 of envs |
| `reset_kernel` | 40 | 20 | 0.4 MiB | nothing |
| `place_house_kernel` | 680 | 48 | 80 B/env | only at worldgen |
| `generate_world_kernel` | 160 | 16 | 128 MiB | only at worldgen |

The four state kernels plus `update_cameras` sum to 464 SASS instructions
(432 without it) and about 4 MiB of traffic per step. The render kernel is 776 **static** instructions and 96–384 MiB of
traffic, and its static count understates the dynamic cost by 6× because the
DDA body repeats. **The five state kernels together are cheaper than a single
DDA iteration of the render kernel.** Any per-step number is a render number.

### 4.1 The fp16 DDA is a pessimisation, and that is visible in the SASS

`render_kernel_uint8_fp16` (`:1127`) keeps the camera setup in fp32 and moves
the `t_max`/`t_delta` accumulators to `__half`, with `__hadd`/`__hlt`. On sm_75
this maps to native `HADD2` and `HSETP2.GEU.AND`: **48 `HADD2` and 32
`HSETP2.GEU` in the kernel.** `MEASURED` It replaces 8 `FADD.FTZ` with 48
`HADD2` and 48 `FSETP.GEU` with 32 `HSETP2.GEU`, and the total goes **up**, from
776 to 824 SASS instructions.

The reason is that sm_75's fp16 pipe is half-rate relative to fp32 for scalar
`HADD2` (which does two halves at once but is issued on the FP16 pipe), while
`FSETP` is fp32-native. So the arithmetic gets cheaper and the **comparison**,
which is what the DDA is actually made of, does not. And fp16 has 11 bits of
mantissa: `t_max` values around 1.0 have a quantum of 2⁻¹⁰ ≈ 0.001, and after
32 accumulations of `t_delta` the comparison `t_max_x < t_max_y` starts
producing ties, which resolve to the z branch. `MODELLED` — the tie rate is not
measured here, and the T4 run is what would measure it.

**This kernel is the clearest example in the file of a hand-written
optimisation that is worse than the thing it replaced**, and it is the one a
compiler would be least likely to make.

### 4.2 The whole-step number is not the render number

Two facts that must not be conflated, and that the project itself conflates:

- The step is 6–9 kernels. `launch_step` ends with `render_kernel<<<…>>>` on
  the same stream, so the render is the last thing and the other five must
  finish first. The render's time is *contained in* the step time, but the
  step time is *not* the render time.
- The **CUDA graph** removes launch overhead for the other five. At 6–9
  launches per step, eager launch overhead on a modern host is 3–5 µs each, so
  20–45 µs of the step is launch overhead that a graph removes and that a
  framework's own launch path may or may not remove. `kernels.cu:2547` even
  adds a `cudaStreamSynchronize` inside `launch_step`, so that path is
  synchronous by construction.

`minecu_ref/bench/bench.cu` therefore reports the two tables separately and
never forms a ratio between them.

---

## 5. The DDA step count, measured on the CPU

`minecu_ref/bench/dda_model.py` is a CPU transcription of
`render_kernel_uint8_prebasis` (and of `render_kernel` and
`render_kernel_uint8_fast`, behind `--variant`). It runs in numpy on this host
and **times nothing**. Its outputs are a step count, a colour census, and an
image sha256.

At the headline shape, `B` does not matter for the step count, so the model
traces one env. `MEASURED` `minecu_ref/bench/stepcount_prebasis.json`:

```
world 16^3, 4096 voxels, 2086 solid (50.9%), 2010 air
eye (8.00, 11.60, 8.00), yaw 0, pitch 0, fov 70, max_steps 32
image 32x32, 65.1% sky pixels, 4 distinct colours
DDA steps: mean 23.326  min 3  max 32  p50 32  p90 32
pixels that ran the full 32-step budget: 667/1024 (65.1%)
image sha256 88fb50b8f2189ec6...
```

**65% of pixels exhaust the step budget.** The median ray takes all 32 steps.
The reason is geometric, not incidental: the camera sits at `y = 10 + 1.6 =
11.6` in a 16-voxel world whose ground is at `y = 7`, and it looks level. A
level ray in a 16-wide box with a 70° FOV mostly travels through air and exits
the far side without ever hitting anything.

Sensitivity, all `MEASURED`:

| variant | mean steps | at step cap |
|---|---|---|
| `prebasis`, pitch 0 | 23.33 | 65.1% |
| `prebasis`, pitch −0.2 | 19.49 | 48.8% |
| `prebasis`, pitch −0.5 | 14.28 | 27.0% |
| `prebasis`, pitch +0.3 | 27.17 | 83.3% |
| `fast` (ground early-out), pitch 0 | 22.10 | 53.6% |
| `fp32`, pitch 0 | 23.33 | 65.1% |

The ground early-out in `render_kernel_uint8_fast` — one extra compare per step
— buys 1.2 steps out of 23.3. `MODELLED` it is a 5% win on a kernel that costs
1296 SASS instructions to 776. **That is a bad trade and it is in the file.**

The `max_steps` sweep is the one that matters, because the documented grid has
`max_steps` varying:

| max_steps | mean DDA steps | reach the 49.6M claim? |
|---|---|---|
| 3 | 3.00 | yes |
| 4 | 3.87 | yes |
| 5 | 4.65 | yes |
| **6** | **5.42** | **no** |
| 8 | 6.97 | no |
| 16 | 12.90 | no |
| **32 (the documented peak config)** | **23.33** | **no** |
| 64 | 44.17 | no |

---

## 6. Does it compile for sm_75? Yes. Here is the evidence.

`minecu_ref/build.sh sm_75`, CUDA 12.9.86, `-O3 --use_fast_math -std=c++17`,
flags copied from upstream's `setup.py`. `MEASURED`

**It compiles. Zero errors. Zero warnings (after `-diag-suppress 177` for
upstream's own unused-variable warnings). Zero register spills in every one of
the 22 kernels.** The only diagnostics are upstream's own: unused `b`, `x`, `z`
in the two worldgen kernels (`kernels.cu:1811-1815`, `2014-2015`) and unused
block-type constants.

Resources, from `ptxas -v`, `minecu_ref/build/resources_sm_75.txt`:

| kernel | registers | spills | barriers | smem |
|---|---|---|---|---|
| `render_kernel_uint8_prebasis` | 28 | 0 | 1 | 64 B |
| `render_kernel` | 33 | 0 | 0 | 0 |
| `render_kernel_uint8_fp16` | 32 | 0 | 0 | 0 |
| `render_kernel_uint8_minimal` | 25 | 0 | 0 | 0 |
| `render_kernel_uint8_smem` | 38 | 0 | 1 | dynamic |
| `render_kernel_uint8_coalesced` | 28 | 0 | 0 | 0 |
| `precompute_camera_basis_kernel` | 32 | 0 | 0 | 0 |
| `raycast_interact_kernel` | 29 | 0 | 0 | 0 |
| `physics_kernel` | 26 | 0 | 0 | 0 |
| `place_house_kernel` | 48 | 0 | 0 | 0 |

25–38 registers across the render kernels means full 4-warp occupancy on sm_75
(64K registers/SM ÷ (128 threads × 38) = 13 blocks/SM, and sm_75 caps at 16
blocks and 64 KB smem, so occupancy is limited by blocks not registers).

### 6.1 The portability gate

`minecu_ref/build/sass_census_sm_75.txt`, 22 functions, 9,648 SASS
instructions, and every one of these counts is **zero**:

| opcode | why it would matter | count |
|---|---|---|
| `HMMA` | fp16 tensor core, **sm_80+** | 0 |
| `HGMMA` | fp8 tensor core, **sm_90a only** | 0 |
| `LDGSTS` | `cp.async`, **sm_80+** | 0 |
| `REDUX` | `redux.sync`, **sm_80+** | 0 |
| `ELECT` | ballot-elect, **sm_80+** | 0 |
| `BAR.RED` | warp reduce barrier, **sm_80+** | 0 |
| `F2FP` | fp8 conversion, **sm_89+** | 0 |
| `IMMA` / `MMA` / `LDSM` | legal on sm_75 anyway | 0 |

And these are **present**, which is what the kernel is made of:

| opcode | count | what it is |
|---|---|---|
| `LDG.E.S8.CONSTANT.SYS` | 112 | the `__ldg` voxel read, `ld.global.nc.sys` — the DDA's 16 loads × 7 render kernels |
| `STG.E.U8.SYS` | 140 | the uint8 obs store, three per pixel |
| `BSSY` / `BSYNC` | 277 each | branch-synchronisation pairs, i.e. the DDA's early-exit `goto`s |
| `BRA` | 848 | unconditional branches, mostly the same early exits |
| `MUFU.RCP` | 84 | the runtime `/width` and `/height` divisions |
| `MUFU.RSQ` | present | `rsqrtf` for the ray normalise |
| `HADD2` / `HSETP2.GEU` | 48 / 32 | the fp16 DDA variant, native on sm_75 |

`build.sh` **fails the build** if any gated opcode appears, so this is a gate and
not a report.

### 6.2 The one portability trap, and it is not an instruction

`kernels.cu:6` includes `<cuda_pipeline.h>`, whose comment says "For cp.async
(Ampere+)". **`cp.async` is never called.** `LDGSTS=0` confirms it. The
`render_kernel_uint8_smem` comment at `:831` says "Uses cp.async for async
global→shared copy" and the code at `:868-879` does a plain `int4` copy loop
instead. So the sm_80 dependency is a comment, and the kernel that would have
needed it is sm_75-clean. `[ARTIFACT]` both comments read directly.

The real sm_75 limit is **shared memory, not instructions**, and it is
`render_kernel_uint8_smem` that hits it:

| world | dynamic smem requested (`64 + W³`) | sm_75 limit | result |
|---|---|---|---|
| 16 | 4,160 B | 48 KB default | fits |
| 32 | 32,832 B | 48 KB default | fits, 1 block/SM |
| **48** | **110,656 B** | 64 KB even with opt-in | **launch fails** |

`launch_step_uint8_smem` (`kernels.cu:3170`) computes
`shared_mem_size = 64 + world_x*world_y*world_z` with no cap and no
`cudaFuncSetAttribute` opt-in, so **the 48³ rows in `docs/benchmarks.md` cannot
have come from the smem path.** Another reason the throughput table's provenance
is unrecorded.

### 6.3 Why this is a better comparison card than KDA

`notes/16-generated-vs-handwritten.md` established that flash-linear-attention's
KDA is not runnable on sm_75 at all: the hand-written kernel issues `HMMA`, which
is sm_80+, so on a T4 the comparison could only be made on the generated side
and the hand-written side would be absent. **`mine.cu` has no such problem.**
`minecu_ref/build.sh` compiles it for sm_75 and `minecu_ref/kaggle/run.sh` will
measure it on a T4. The hand-written baseline is reachable, which is the whole
precondition for a gap measurement.

---

## 7. The 49.6M claim

`CLAIM` `docs/benchmarks.md:7` and `scripts/plot_throughput.py:20`, both
`49.63` M env-steps/s at `batch=32768, world=16, 32×32`, on an RTX 3090. Not
reproduced here; there is no GPU on this host.

Two independent things can be said about it, and only two.

### 7.1 Which render path it used is unrecorded

`docs/benchmarks.md` gives a `(world, resolution, batch)` table with no dtype
and no path. `plot_throughput.py:15-52` hardcodes the same numbers. The eight
render kernels differ by 1.6× in instruction count and 4× in output bytes, and
`MineEnv.__init__.py:164` defaults to `obs_dtype="float32"`, which selects
`render_kernel` — the **fp32** one, the largest. Nothing in the repo says which
one produced the table. `scripts/benchmark_new_kernels.py` imports
`minecu._C_experimental`, which `setup.py` does not build, so that script cannot
have produced it either.

### 7.2 The instruction-issue bound says the claim is not reachable

Take the claim at face value: 49.63 M env-steps/s at `B=32768` is
**660.2 µs per step**, for 33,554,432 pixels, i.e. 50.8 Gpixel/s.

An RTX 3090 has 82 SMs × 4 schedulers × 1 warp-instruction/cycle at 1.70 GHz
= **557.6 G warp-instructions/s** (datasheet, not measured here). At 660.2 µs
the whole step can issue 3.68 × 10⁸ warp-instructions. Divided over 1,048,576
warps, that is **351 SASS instructions per pixel**. `MODELLED`

`render_kernel_uint8_prebasis` needs 152 before the first DDA step and 54 after
the last, leaving 351 − 206 = 145 for the DDA body, at 38 instructions per step
= **3.82 steps per pixel**. The kernel as written on the world the project's own
worldgen produces takes **23.33**. `MEASURED` (`stepcount_prebasis.json`)

**Verdict: not reachable, by 6.1×, on the most generous reading.** The bound
assumes zero stalls, full 4-warp occupancy, and that all 82 SMs are available —
a real kernel is slower, so the bound flatters the claim.

For the fp32 path it is far worse: 1224 SASS instructions with 243 and 87 of
them fixed leaves 21 for the DDA, i.e. **0.35 steps against a measured 23.33**.

Sweeping the whole documented hyperparameter grid, `MODELLED` + `MEASURED`, the
verdict holds everywhere: `world ∈ {16, 32, 48}` × `max_steps ∈ {8, 16, 32, 64}`
× `pitch ∈ {0, −0.2, −0.5, +0.3}` all miss, with the best case (world 16,
`max_steps` 8, pitch 0) at 6.97 steps against a 3.82 budget. The claim becomes
arithmetically reachable only at `max_steps ≤ 5`, which is not in the table.

**What this does and does not establish.** It establishes that a TinyGrad number
must **not** be reported as a ratio against 49.6M, because at the shape the
claim is quoted for, the hand-written source and the claim do not describe the
same computation. It does **not** establish that the author was wrong: the
claim could have been measured on a source revision, a shape, or a
`max_steps`/`obs_dtype` combination that is not in this commit, and the
throughput table records none of them. §8 says what would settle it.

### 7.3 What the band does allow

At 49.63 M env-steps/s the required bandwidth, `MODELLED` on 936.2 GB/s
datasheet:

| path | obs write only | obs + compulsory voxel read |
|---|---|---|
| uint8 obs | 152.5 GB/s (16.3% of datasheet) | 355.7 GB/s (38.0%) |
| fp32 obs | 609.9 GB/s (65.1%) | 813.1 GB/s (86.9%) |

The fp32 numbers are *at or past* a 3090's realistic streaming ceiling, which is
a second independent reason to doubt the fp32 reading of the table. The uint8
numbers are comfortable, which is consistent with the table being generated but
with a different source revision than `ec2085e`.

---

## 8. Validity conditions for the comparison

Stated explicitly, because the failure mode here is not "the compiler is slow",
it is "the two numbers were never comparable".

### 8.1 What must match

| dimension | requirement | why |
|---|---|---|
| **architecture** | both on sm_75 (T4), or both on the same other arch | `mine.cu` compiles for sm_75 with zero spills; KDA does not compile for sm_75 at all. A T4 comparison is possible for this pair and impossible for the KDA pair. |
| **shape** | same `B`, `W`, `H`, `max_steps`, `ground_height` | the DDA step count is a function of all of them, and §5 shows `max_steps` 5 → 32 moves the mean from 4.65 to 23.33 steps, a 5× swing |
| **obs dtype** | same. fp32 is 384 MiB/step, uint8 is 96 MiB | a 4× byte difference makes any bandwidth comparison meaningless |
| **render variant** | same one of the 8 | 776 to 1296 SASS instructions, 1.6× |
| **output semantics** | bit-identical obs where the layout allows; if not, a declared tolerance and a stated reason | fp16 DDA and `rsqrtf` and `--use_fast_math` all move bits |
| **timing boundary** | the same boundary on both sides | see below |
| **world content** | the same worldgen | §2.6, `ground_height=8` vs `world_y/4=4` disagree; the two 16³ throughput rows measure half-houses |

### 8.2 The timing boundary — the category error to avoid

**The project's headline is a whole-loop number. A framework emits kernels.
Dividing one by the other is a category error**, and it is the error this
section exists to prevent.

- `49.6M env-steps/s` is `B / (time for all 6–9 kernels + graph launch)`. It
  includes five state kernels that together are 464 SASS instructions and 4 MiB
  of traffic, plus CUDA-graph launch overhead, plus — in `launch_step` — a
  `cudaStreamSynchronize` inside the timed region (`kernels.cu:2547`).
- A TinyGrad graph for the same computation emits **its own** per-op kernels and
  its own schedule. Its whole-graph time is comparable to the hand-written
  whole-step time, and only that.

**The defensible comparison, in order of preference:**

1. **Whole-step vs whole-graph**, both on the same device, same shapes, same
   dtype, same variant, same `max_steps`. This is apples to apples and it is
   what `minecu_ref/bench/bench.cu` reports as its whole-step table.
2. **Per-kernel vs per-op**, for `render` alone. Both sides timed with
   `cudaEvent` around a single launch, same input state, warm. This is what
   isolates the codegen question from the scheduling question, and it is the
   comparison the study's thesis actually needs.
3. **Not defensible:** any per-kernel TinyGrad number against the 49.6M
   headline; any T4 number against a 3090 number; any uint8 TinyGrad number
   against an fp32 hand-written number.

### 8.3 The three numbers to report, separately

- hand-written, whole step, per launch, T4, `MEASURED`
- hand-written, `render_kernel_uint8_prebasis` alone, T4, `MEASURED`
- measured memory floors on the same T4 in the same run, so every "% of peak"
  has a denominator from the card making the claim, not from a datasheet

`minecu_ref/bench/bench.cu` produces exactly these three and labels them so they
cannot be merged.

### 8.4 The three T4 predictions, to be measured rather than argued

`MODELLED`, all from `minecu_ref/bench/dda_model.py` and
`sass_census_sm_75.txt`. **None of these is a timing.**

| prediction | why | what would falsify it |
|---|---|---|
| the T4 renders ~2.5× slower than a 3090 on this kernel, i.e. ~6–10 M env-steps/s at the headline shape | T4 issue rate is 225.6 G warp-instr/s vs 557.6 G, a 2.47× gap, and the kernel is issue-bound | T4 reaching above ~20 M would mean something other than issue rate is the limit |
| the `uint8_prebasis` step is **not** bandwidth bound on a T4, and lands well under its 96 MiB/step of compulsory traffic | 23.3 dependent loads/pixel is a latency chain, and a T4's 320 GB/s over 96 MiB is 315 µs against an issue bound of 5,077 µs | a measured step near 315 µs would mean the DDA cost is much lower than the instruction count implies |
| `render_kernel_uint8_fp16` is **slower** than `render_kernel_uint8_minimal` | 824 vs 808 SASS instructions, and fp16 `t_max` quantises to 11 mantissa bits over 32 accumulations | fp16 winning would mean the sm_75 fp16 pipe beats fp32 for this compare-and-accumulate pattern, which the SASS argues against |

The third is the most interesting: it is a hand-written optimisation that should
lose, and losing is a real result about what a compiler will and will not do on
its own.

---

## 9. The technique inventory, verified

The sibling repo's description of its own hand-written work names six techniques.
Each was checked against `kernels.cu` rather than carried over. **Three are
present, three are not, and the three that are present are the least
sophisticated ones.**

| # | technique (from the sibling's description) | present here? | evidence |
|---|---|---|---|
| 1 | warp-transposed coalesced obs encode | **no** | §2.7. The obs store is three 1-byte `STG.E.U8.SYS` per pixel, `MEASURED` in the SASS. There is no warp transposition, no shared-memory staging of the output, and no vectorised store anywhere in the file. The one kernel that *tries* to fix coalescing, `render_kernel_uint8_coalesced` (`:709`), transposes the **voxel** layout to `[Y,X,Z,B]` so *reads* coalesce across batch — and **is never called from any step path** (`:2213` is a standalone benchmark entry). |
| 2 | warp-cooperative: one warp per env, one lane per view cell | **no, and the opposite** | §2.7. The mapping is **one 128-thread block per env, one thread per pixel**, with `blockIdx.y` = env. Adjacent threads take adjacent pixels of the *same* env, so a warp's 32 lanes read 32 *different* voxels. The one-warp-per-env mapping the technique describes would make a warp read one voxel, broadcast to 32 lanes — the exact opposite of what this does. |
| 3 | list-free two-pass scans | **not applicable** | There is no scan. The step is a chain of elementwise ops over the env index plus one gather; nothing reduces. `grep` for a scan-shaped construct returns nothing. |
| 4 | megakernel: fuse small kernels to cut launch overhead | **partial, and the wrong way** | §4.2. The step is **6–9 separate launches**, not a megakernel. The overhead is instead removed by **CUDA graph capture** (#6), which is the tool for exactly this problem and is used correctly. Fusing would have been the wrong answer: the state kernels are 464 SASS instructions in total, so a
    fused kernel would save launch overhead and lose the ability to overlap the
    render with the tail of the state updates. |
| 5 | move scratch to a global arena | **no** | All scratch is preallocated by the caller as plain device tensors (`minecu/__init__.py:244-251`) and passed in by pointer. There is no arena, no bump allocator, no offset bookkeeping. The step does zero allocation after `_allocate_state`, which achieves the same *effect* (no `cudaMalloc` in the step) by a different and simpler means. |
| 6 | CUDA-graph capture to remove launch overhead | **yes, and it is the best-executed technique in the file** | `StepGraphState` (`:2338-2374`), three independent instances (`:2376-2382`). Capture after a 10-call warmup (`WARMUP_THRESHOLD = 10`), keyed on `batch_size`/`width`/`height` **and on the `voxels`, `positions` and `obs_buffer` pointers** (`:2526-2535`) so a new env or a reallocated tensor forces a recapture. `cudaStreamBeginCapture(..., cudaStreamCaptureModeGlobal)`, then `EndCapture` + `Instantiate`, then `cudaGraphLaunch` on the fast path. `MEASURED` exercised: `minecu_ref/bench/bench.cu` drives 12 calls to cross the threshold and then times the replay. |

### 9.1 What is actually there, cited

Techniques present that the sibling's list does not name:

| technique | file:line | what it is |
|---|---|---|
| precomputed camera basis | `kernels.cu:511-561` | hoists 4 `sinf`/`cosf` + 1 `tanf` out of the per-pixel path, into a `[B,14]` tensor consumed by the render kernel. 56 SASS instructions replacing ~1,100 per pixel |
| 2-D grid keyed on the env | `kernels.cu:2182-2188`, `2858-2861` | `grid(blocks_per_batch, batch_size)`, so `blockIdx.y` is the env and the basis is a per-block invariant |
| basis in shared memory | `kernels.cu:581-597` | 14 floats, loaded by 14 threads, one `BAR.SYNC`, read by all 128. **Not** a vectorised load, and it is the only barrier in the kernel |
| 16× static DDA unroll | `kernels.cu:681-687` | `#pragma unroll 4` on a loop stepping by 4 containing 4 macro bodies = 16 steps; ptxas emits all 16, **load-to-load spacing exactly 38** |
| unsigned bounds test | `kernels.cu:666-668` | `(unsigned)v < (unsigned)w` folds 6 signed compares into 3 `ISETP.GE.U32.OR` per unrolled step |
| pre-quantised colour LUT | `kernels.cu:479-495` | `BLOCK_COLORS_U8[13][3]` removes a per-pixel float→uint8 conversion; read with `LDC.U8` from `c[0x3][...]`, i.e. it stays in constant cache and never touches L1 |
| `__ldg` on the voxel read | `kernels.cu:669` | `ld.global.nc.sys`, confirmed as `LDG.E.S8.CONSTANT.SYS` |
| `rsqrtf` not `sqrtf`+divide | `kernels.cu:626` | one `MUFU.RSQ` |
| fp16 DDA accumulator | `kernels.cu:1209-1247` | `__hadd`/`__hlt` on `t_max` — **and a pessimisation, §4.1** |
| `goto`-based DDA early exit | `kernels.cu:664-679` | compiles to `BSSY`/`BREAK`/`BSYNC`, i.e. real branch-synchronisation, not predication |
| whole world in shared memory | `kernels.cu:835-982` | `[B,14]` basis + `W³` bytes, `int4` copies, then the DDA reads shared. **Only viable at small `W`**, §6.2 |
| transposed voxel layout | `kernels.cu:709-826` | `[Y,X,Z,B]` so a warp reads 32 contiguous bytes. Correct idea, never wired into a step |
| masked worldgen | `kernels.cu:1989-2069` | regenerate only reset envs, with a comment at `:2908` recording the 0.5 ms it costs and the decision to move it to Python instead |
| graph keyed on tensor pointers | `kernels.cu:2529-2535` | a graph captures addresses, so a changed `voxels`/`positions`/`obs_buffer` must invalidate it. This is the subtle part of CUDA graphs and it is handled |

Techniques named in the sibling's description and **absent** from the whole
file, confirmed by reading all 3,245 lines: warp-level primitives of any kind
(`__shfl`, `__ballot`, `__activemask`, `__reduce_*`, `__any`/`__all`), shared
memory as a *staging buffer for global writes* (it is used for the camera basis
and for voxels, never for output), vectorised global stores
(`float4`/`int4` appear once, in the smem voxel load at `:868`), `cp.async`
(the include is at `:6`, the comment at `:831`, the call is nowhere), atomics,
`cudaMemcpyAsync` inside the step, and persistent kernels / grid-stride loops.

---

## 10. Deliverables, and what is deferred

### 10.1 Done here, on this host, with no GPU

| artefact | what |
|---|---|
| `minecu_ref/src/kernels.cu` | the reference, byte-identical, hash-checked on every build |
| `minecu_ref/bench/bench.cu` | standalone CUDA benchmark: selftest gate, memory floors, per-kernel, whole-step, roofline. No Python, no PyTorch |
| `minecu_ref/bench/dda_model.py` | CPU reference of the render kernel. DDA step counts, colour census, image sha256, and the §7 bound. **Times nothing** |
| `minecu_ref/build.sh` | containerised build for any `-arch`; produces SASS, `ptxas -v` resources, an instruction census with a hard portability gate, and a hash manifest |
| `minecu_ref/kaggle/run.sh`, `make_notebook.py` | one command, bounded, self-verifying |
| `notes/17-minecu-spec.md` | this document |

**sm_75 compile: it works.** §6. Zero spills, zero illegal opcodes, and the
gate fails the build if that stops being true.

### 10.2 Deferred to a Kaggle T4 run, and exactly what it will print

`minecu_ref/kaggle/run.sh` pushes `tenghuzhang/mc-minecu-t4-baseline`,
`-arch=sm_75` explicit, 25-minute server-side session cap, 12 battery entries
under a 1,500 s in-notebook budget. It prints, in this order:

1. `kernels.cu` and `bench.cu` sha256, or a hard stop
2. the nvcc version, the `-arch=sm_75` compile, the binary size
3. the SASS census: `arch = sm_75`, `HMMA=0 HGMMA=0 LDGSTS=0 REDUX=0 ELECT=0
   BAR.RED=0 F2FP=0`, the `STG.E.U8.SYS` and `LDG.E.S8.CONSTANT.SYS` counts
4. three `selftest_*` entries — a **gate**: any FAIL makes the run exit non-zero
5. `headline_w16`: the measured write/read/copy/memcpy floors, then the
   per-kernel table (`med/min/max/std us`, `GB/s`, `%` of the write floor), then
   the whole-step table for all three graph paths with M env-steps/sec, then the
   roofline position
6. a batch sweep 4K → 32K, `world=32`, `64×64`, and `max_steps ∈ {8, 64}`
7. a summary table of M env-steps/s per entry, the budget used, and a
   five-line reading guide that names the 49.6M/3090 figure as not comparable

`README.md` in `minecu_ref/` has the full output shape.

### 10.3 What no run here can settle

- **Whether the 49.6M claim reproduces.** It needs an RTX 3090. §7 gives a
  bound that says it does not, from this source, at that shape; a 3090 run would
  turn that into a measurement or refute it.
- **What the T4 numbers are.** §8.4 gives three falsifiable predictions, all
  `MODELLED`.
- **Anything about TinyGrad.** Deliberately not touched. That lane's own
  numbers, at the shapes in §8.1, are what the gap is.
