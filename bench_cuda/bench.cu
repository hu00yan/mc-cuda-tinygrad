// bench.cu -- standalone, no-Python benchmark for the mine.cu hand-written kernels.
//
// WHY THIS FILE EXISTS
//   The upstream project is a PyTorch extension. The kernels are reachable from
//   C because every launcher in src/kernels.cu is `extern "C"` and takes raw
//   pointers, so nothing in the launcher surface needs torch/ATen/pybind. This
//   driver therefore links ONLY against libcudart and src/kernels.cu: no
//   libtorch, no Python, no JIT extension build. That is what makes an sm_75
//   baseline reproducible in a plain CUDA container.
//
// WHAT IT MEASURES, AND WHAT IT REFUSES TO MEASURE
//   Two different quantities, reported separately and never mixed:
//
//   (1) WHOLE-STEP.  The 6-to-8-kernel `launch_step*` sequence, timed with
//       cudaEvent, after the upstream CUDA-graph capture threshold. This is the
//       quantity the project's "49.6M env-steps/sec" headline is a claim about.
//
//   (2) PER-KERNEL.  Each kernel launched on its own through its own `extern
//       "C"` launcher and timed with cudaEvent. This is the quantity a
//       framework comparison must use, because a framework emits one kernel per
//       op and has no "whole loop" to hand back.
//
//   It also measures a MEMORY FLOOR on the same card in the same run, because
//   "close to the write-bandwidth floor" is only a statement once the floor has
//   been measured on the device making the claim. Four probes:
//     write  : a store-only kernel, no loads          -> floor for a kernel
//                                                       whose only DRAM traffic
//                                                       is the obs tensor
//     read   : a load-only kernel, no stores
//     copy   : a read-modify-write kernel, 1:1
//     memcpy : cudaMemcpyAsync D2D, 1:1
//   One direction is counted for write/read, two for copy/memcpy. Quoting the
//   write row against a 2x-counted row would be a category error, so the table
//   prints which it is.
//
// SELF-VERIFICATION.  With --no-selftest absent (the default) the driver
//   checks, on the actual device, that the world generator, the action decode,
//   the physics clamp, the camera-basis precompute, all four render variants,
//   the block-break reward and the full CUDA-graph step path actually do what
//   their names say, then prints PASS/FAIL per check. A FAIL skips the timing
//   section and sets exit code 2: a timing from a wrong kernel is not a timing.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHK(x)                                                                  \
  do {                                                                          \
    cudaError_t e_ = (x);                                                       \
    if (e_ != cudaSuccess) {                                                    \
      fprintf(stderr, "FATAL %s:%d %s -> %s\n", __FILE__, __LINE__, #x,         \
              cudaGetErrorString(e_));                                          \
      exit(3);                                                                  \
    }                                                                           \
  } while (0)

// ---------------------------------------------------------------------------
// Launcher declarations, copied from the extern "C" block in src/kernels.cu.
// Declared here rather than in a header because upstream ships no header, and
// upstream's files are not modified by this study.
// ---------------------------------------------------------------------------
extern "C" {

void launch_decode_actions(const int8_t* buttons, const float* look, float* forward_in,
                           float* strafe_in, float* delta_yaw_in, float* delta_pitch_in,
                           bool* jump_in, bool* do_break, bool* do_place, float* speed_mult,
                           int batch_size, float sprint_multiplier);

void launch_physics_v2(float* positions, float* velocities, float* yaws, float* pitches,
                       bool* on_ground, const float* forward_in, const float* strafe_in,
                       const float* delta_yaw_in, const float* delta_pitch_in,
                       const bool* jump_in, const float* speed_mult, int batch_size,
                       int world_size, float dt, float gravity, float walk_speed,
                       float jump_vel);

void launch_raycast_interact(int8_t* voxels, const float* positions, const float* yaws,
                             const float* pitches, const bool* do_break, const bool* do_place,
                             float* rewards, int batch_size, int world_size,
                             int8_t target_block, float reward_value);

void launch_reset(float* positions, float* velocities, float* yaws, float* pitches,
                  bool* on_ground, const bool* do_reset, int batch_size, float spawn_x,
                  float spawn_y, float spawn_z);

void launch_generate_world(int8_t* voxels, int batch_size, int world_x, int world_y,
                           int world_z, int ground_height, unsigned int seed);

void launch_place_tree(int8_t* voxels, int batch_size, int world_x, int world_y, int world_z,
                       int tree_x, int tree_z, int ground_height);

void launch_place_house(int8_t* voxels, int batch_size, int world_x, int world_y, int world_z,
                        int house_x, int house_z, int ground_height);

void launch_precompute_camera_basis(const float* cameras, float* basis, int batch_size,
                                    int width, int height, float fov_degrees);

void launch_render_uint8_prebasis(const int8_t* voxels, const float* basis, uint8_t* output,
                                  int batch_size, int world_size, int width, int height,
                                  int max_steps);

void launch_render(const int8_t* voxels, const float* cameras, float* output, int batch_size,
                   int world_size, int width, int height, int max_steps, float view_distance,
                   float fov_degrees);

void launch_render_uint8_minimal(const int8_t* voxels, const float* cameras, uint8_t* output,
                                 int batch_size, int world_size, int width, int height,
                                 int max_steps, float view_distance, float fov_degrees);

void launch_render_uint8_fp16(const int8_t* voxels, const float* cameras, uint8_t* output,
                              int batch_size, int world_size, int width, int height,
                              int max_steps, float view_distance, float fov_degrees);

void launch_step(int8_t* voxels, float* positions, float* velocities, float* yaws,
                 float* pitches, bool* on_ground, const int8_t* buttons, const float* look,
                 const bool* do_reset, float* forward_in, float* strafe_in, float* delta_yaw_in,
                 float* delta_pitch_in, bool* jump_in, bool* do_break, bool* do_place,
                 float* speed_mult, float* cameras, float* obs_buffer, float* rewards,
                 int batch_size, int world_size, int width, int height, int max_steps,
                 float view_distance, float fov_degrees, float dt, float gravity,
                 float walk_speed, float sprint_mult, float jump_vel, int8_t target_block,
                 float reward_value, float spawn_x, float spawn_y, float spawn_z);

void launch_step_uint8_prebasis(
    int8_t* voxels, float* positions, float* velocities, float* yaws, float* pitches,
    bool* on_ground, const int8_t* buttons, const float* look, const bool* do_reset,
    float* forward_in, float* strafe_in, float* delta_yaw_in, float* delta_pitch_in,
    bool* jump_in, bool* do_break, bool* do_place, float* speed_mult, float* cameras,
    float* basis, uint8_t* obs_buffer, float* rewards, int batch_size, int world_x, int world_y,
    int world_z, int width, int height, int max_steps, float view_distance, float fov_degrees,
    float dt, float gravity, float walk_speed, float sprint_mult, float jump_vel,
    int8_t target_block, float reward_value, float spawn_x, float spawn_y, float spawn_z);

void launch_step_uint8_prebasis_full(
    int8_t* voxels, float* positions, float* velocities, float* yaws, float* pitches,
    bool* on_ground, const int8_t* buttons, const float* look, bool* do_reset, int* step_count,
    float* forward_in, float* strafe_in, float* delta_yaw_in, float* delta_pitch_in,
    bool* jump_in, bool* do_break, bool* do_place, float* speed_mult, float* cameras,
    float* basis, uint8_t* obs_buffer, float* rewards, int batch_size, int world_x, int world_y,
    int world_z, int width, int height, int max_steps, float view_distance, float fov_degrees,
    float dt, float gravity, float walk_speed, float sprint_mult, float jump_vel,
    int8_t target_block, float reward_value, float spawn_x, float spawn_y, float spawn_z,
    int episode_length);

}  // extern "C"

// ---------------------------------------------------------------------------
// Config. Defaults are the project's own headline shape: batch 32768, world 16,
// resolution 32x32, max_steps 32 (minecu/config.py FAST_ENV for batch and res,
// world 16 from the docs/benchmarks.md peak row).
// ---------------------------------------------------------------------------
struct Cfg {
  int B = 32768;
  int W = 16;
  int H = 32;
  int Wd = 32;
  int max_steps = 32;
  float view_dist = 32.0f;
  float fov = 70.0f;
  float dt = 0.05f;
  float gravity = -20.0f;
  float walk_speed = 4.0f;
  float sprint_mult = 1.5f;
  float jump_vel = 8.0f;
  // -1 means "derive it", see ground_of() below. The previous hard default of 8
  // is what buried the agent at every world size except 32.
  int ground_h = -1;
  int8_t target_block = 3;  // OAKLOG
  float reward_value = 1.0f;
  int episode_len = 0;  // 0 = no limit
  int iters = 200;
  int graph_iters = 200;
  int warmup = 25;
  int copy_bytes = 512 << 20;
  int settle_iters = 64;      // cap on the settle probe
  double min_dda_steps = -1;  // -1 means "derive from max_steps", see dda_floor()
  bool selftest = true;
  bool selftest_only = false;
  bool measure_degenerate = false;  // see the DDA gate comment in selftest()
  bool sweep = false;
};

struct Dev {
  int8_t* voxels = nullptr;
  float *pos = nullptr, *vel = nullptr, *yaw = nullptr, *pitch = nullptr;
  bool* on_ground = nullptr;
  int8_t* buttons = nullptr;
  float* look = nullptr;
  bool* do_reset = nullptr;
  float *fwd = nullptr, *strafe = nullptr, *dyaw = nullptr, *dpitch = nullptr;
  bool *jump = nullptr, *do_break = nullptr, *do_place = nullptr;
  float* speed_mult = nullptr;
  float *cameras = nullptr, *basis = nullptr;
  float* obs_f32 = nullptr;
  uint8_t *obs_u8 = nullptr, *obs_u8b = nullptr;
  float* rewards = nullptr;
  int* step_count = nullptr;
  // host mirrors, only the few a selftest check needs
  std::vector<int8_t> h_vox, h_vox2, h_coal;
  std::vector<float> h_f3;
  std::vector<uint8_t> h_u3;
  std::vector<float> h_rew;
};

template <typename T>
static T* devalloc(size_t n) {
  T* p = nullptr;
  CHK(cudaMalloc(&p, n * sizeof(T)));
  return p;
}

static size_t wvox_of(const Cfg& c) { return (size_t)c.W * c.W * c.W; }
static size_t hw_of(const Cfg& c) { return (size_t)c.H * c.Wd; }

// WHY THE DEFAULT GROUND IS world_y/4 AND NOT A CONSTANT.
//
// kernels.cu has TWO independent notions of "the ground" and they do not agree:
//
//   generate_world_kernel (kernels.cu:1819-1825) writes GRASS into voxel row
//   y == ground_height-1, so the top face of the visible surface is at
//   y == ground_height.
//
//   physics_kernel (kernels.cu:1566) collides the agent against a flat
//   ANALYTIC plane at y == world_y/4. It never reads a voxel. There is no
//   voxel collision anywhere in the physics path.
//
// The agent therefore rests at y = world_y/4 with its eye at world_y/4 + 1.6,
// and render_kernel_uint8_prebasis tests the CURRENT voxel before it steps
// (kernels.cu:666). If floor(eye_y) is solid the ray hits on iteration 0: one
// __ldg, one store, done. At the shape this study measured (world 16, ground 8)
// the floor is y=4, the eye is at 5.6, voxel row 5 is DIRT, and EVERY pixel hit
// at step 0 -- a mean of 1.291 DDA steps against a 32-step cap, with 96.9% of
// pixels hitting on their first voxel test (measured on a T4).
//
// Setting ground_height = world_y/4 makes the analytic plane the top face of
// the visible surface, so the eye sits in air one voxel above its own ground
// and the DDA marches. This is a config change in the harness only; the
// vendored kernels.cu is untouched. See bench_cuda/buried_agent_fix.md.
static int ground_of(const Cfg& c) { return c.ground_h >= 0 ? c.ground_h : c.W / 4; }

// The DDA-step floor the selftest enforces. Two clauses:
//   * an absolute floor of 2.0 mean steps, which a buried eye (1.000 when fully
//     in rock, 1.291 measured in the batch) cannot reach at ANY step cap; and
//   * 10% of max_steps, so raising the cap cannot be used to pass the gate by
//     making the cap bigger.
// Measured on a T4 at the headline shape: 26.416 non-degenerate (8.3x the
// 3.20 floor) against 1.291 degenerate (0.40x). The floor sits far enough below
// the real workload that ordinary x/z wander cannot trip it, and far enough
// above the degenerate value that nothing short of re-burying the agent can.
static double dda_floor(const Cfg& c) {
  return c.min_dda_steps >= 0 ? c.min_dda_steps : fmax(2.0, 0.10 * (double)c.max_steps);
}

// The complementary gate: at most this fraction of pixels may find their FIRST
// tested voxel solid. Measured on the settled state with the bench action
// stream: 100.0% at the degenerate ground=8/world=16, 1-4% at the fixed
// ground=world/4, because agents with no voxel collision sometimes walk into
// the tree or the house. 20% sits ~5x from either side.
static const double kEyeSolidMaxPct = 20.0;

static void alloc(Dev& d, const Cfg& c) {
  size_t B = (size_t)c.B, wv = wvox_of(c), hw = hw_of(c);
  d.voxels = devalloc<int8_t>(B * wv);
  d.pos = devalloc<float>(B * 3);
  d.vel = devalloc<float>(B * 3);
  d.yaw = devalloc<float>(B);
  d.pitch = devalloc<float>(B);
  d.on_ground = devalloc<bool>(B);
  d.buttons = devalloc<int8_t>(B * 8);
  d.look = devalloc<float>(B * 2);
  d.do_reset = devalloc<bool>(B);
  d.fwd = devalloc<float>(B);
  d.strafe = devalloc<float>(B);
  d.dyaw = devalloc<float>(B);
  d.dpitch = devalloc<float>(B);
  d.jump = devalloc<bool>(B);
  d.do_break = devalloc<bool>(B);
  d.do_place = devalloc<bool>(B);
  d.speed_mult = devalloc<float>(B);
  d.cameras = devalloc<float>(B * 5);
  d.basis = devalloc<float>(B * 14);
  d.obs_f32 = devalloc<float>(B * hw * 3);
  d.obs_u8 = devalloc<uint8_t>(B * hw * 3);
  d.obs_u8b = devalloc<uint8_t>(B * hw * 3);
  d.rewards = devalloc<float>(B);
  d.step_count = devalloc<int>(B);
  d.h_vox.assign(B * wv, 0);
  d.h_vox2.assign(B * wv, 0);
  d.h_f3.assign(B * 3, 0.f);
  d.h_rew.assign(B, 0.f);
}

static float spawn_x_of(const Cfg& c) { return c.W / 2.0f; }
static float spawn_z_of(const Cfg& c) { return c.W / 2.0f; }
static float spawn_y_of(const Cfg& c) { return (float)(ground_of(c) + 2); }

// Mirrors minecu/__init__.py _allocate_state + _generate_world.
static void init_state(Dev& d, const Cfg& c) {
  size_t B = (size_t)c.B, wv = wvox_of(c), hw = hw_of(c);
  CHK(cudaMemset(d.pos, 0, B * 3 * sizeof(float)));
  CHK(cudaMemset(d.vel, 0, B * 3 * sizeof(float)));
  CHK(cudaMemset(d.yaw, 0, B * sizeof(float)));
  CHK(cudaMemset(d.pitch, 0, B * sizeof(float)));
  CHK(cudaMemset(d.on_ground, 1, B * sizeof(bool)));
  CHK(cudaMemset(d.do_reset, 0, B * sizeof(bool)));
  CHK(cudaMemset(d.step_count, 0, B * sizeof(int)));
  CHK(cudaMemset(d.jump, 0, B * sizeof(bool)));
  CHK(cudaMemset(d.do_break, 0, B * sizeof(bool)));
  CHK(cudaMemset(d.do_place, 0, B * sizeof(bool)));
  CHK(cudaMemset(d.speed_mult, 0x3f, B * sizeof(float)));  // 1.0f
  CHK(cudaMemset(d.obs_f32, 0, B * hw * 3 * sizeof(float)));
  CHK(cudaMemset(d.obs_u8, 0x7f, B * hw * 3));

  // A fixed, non-degenerate action stream. Deterministic from the env index so
  // a run is reproducible, and the raycast kernel has something to hit.
  std::vector<int8_t> hb(B * 8, 0);
  std::vector<float> hl(B * 2, 0.f);
  for (size_t i = 0; i < B; i++) {
    hb[i * 8 + 0] = 1;                                // BTN_FORWARD
    hl[i * 2 + 0] = 0.002f * (float)((i % 97) - 48);  // delta yaw
    hb[i * 8 + 5] = (int8_t)((i % 32) == 0 ? 1 : 0);  // BTN_BREAK, 1/32 of envs
  }
  CHK(cudaMemcpy(d.buttons, hb.data(), B * 8, cudaMemcpyHostToDevice));
  CHK(cudaMemcpy(d.look, hl.data(), B * 2 * sizeof(float), cudaMemcpyHostToDevice));

  launch_generate_world(d.voxels, c.B, c.W, c.W, c.W, ground_of(c), 42u);
  CHK(cudaDeviceSynchronize());
  CHK(cudaGetLastError());
  launch_place_tree(d.voxels, c.B, c.W, c.W, c.W, (int)spawn_x_of(c), (int)spawn_z_of(c) + 2,
                    ground_of(c));
  launch_place_house(d.voxels, c.B, c.W, c.W, c.W, (int)spawn_x_of(c) + 6,
                     (int)spawn_z_of(c) - 2, ground_of(c));
  CHK(cudaDeviceSynchronize());
  CHK(cudaGetLastError());

  std::vector<float> hp(B * 3);
  for (size_t i = 0; i < B; i++) {
    hp[i * 3 + 0] = spawn_x_of(c);
    hp[i * 3 + 1] = spawn_y_of(c);
    hp[i * 3 + 2] = spawn_z_of(c);
  }
  CHK(cudaMemcpy(d.pos, hp.data(), B * 3 * sizeof(float), cudaMemcpyHostToDevice));
  CHK(cudaMemset(d.on_ground, 1, B * sizeof(bool)));

  // cameras, so a bare render probe is well defined (pitch down, as
  // scripts/benchmark_new_kernels.py does)
  std::vector<float> hc(B * 5);
  for (size_t i = 0; i < B; i++) {
    hc[i * 5 + 0] = spawn_x_of(c);
    hc[i * 5 + 1] = spawn_y_of(c);
    hc[i * 5 + 2] = spawn_z_of(c);
    hc[i * 5 + 3] = 0.0f;
    hc[i * 5 + 4] = -0.2f;
  }
  CHK(cudaMemcpy(d.cameras, hc.data(), B * 5 * sizeof(float), cudaMemcpyHostToDevice));
}

// `update_cameras_kernel` has no extern "C" launcher of its own; the step paths
// call it internally. Reproduce its exact write from the current agent state so
// the bare render probes see a coherent camera.
static void refresh_cameras(Dev& d, const Cfg& c) {
  size_t B = (size_t)c.B;
  std::vector<float> hp(B * 3), hy(B), hi(B);
  CHK(cudaMemcpy(hp.data(), d.pos, B * 3 * sizeof(float), cudaMemcpyDeviceToHost));
  CHK(cudaMemcpy(hy.data(), d.yaw, B * sizeof(float), cudaMemcpyDeviceToHost));
  CHK(cudaMemcpy(hi.data(), d.pitch, B * sizeof(float), cudaMemcpyDeviceToHost));
  std::vector<float> hc(B * 5);
  for (size_t i = 0; i < B; i++) {
    hc[i * 5 + 0] = hp[i * 3 + 0];
    hc[i * 5 + 1] = hp[i * 3 + 1];
    hc[i * 5 + 2] = hp[i * 3 + 2];
    hc[i * 5 + 3] = hy[i];
    hc[i * 5 + 4] = hi[i];
  }
  CHK(cudaMemcpy(d.cameras, hc.data(), B * 5 * sizeof(float), cudaMemcpyHostToDevice));
}

// Drive the full graph step until every agent has landed on the physics floor
// and stayed there, then refresh the camera so the probes look from the eye the
// TIMED kernels will look from.
//
// WHY THIS IS NECESSARY, NOT COSMETIC. init_state puts the agent at the spawn
// (ground+2). Gravity then pulls it down to world_y/4 and it rests there for
// the rest of the run, because episode_len=0 disables resets. The eye is 1.6
// above that resting height (kernels.cu:541). So the camera the render probes
// and the whole-step graph see is the RESTING one, not the spawn one -- and
// those two can be on opposite sides of the ground. Measuring before the
// settle measures a state the benchmark never spends time in.
//
// Without this the run would report a DDA step count for the spawn eye while
// timing the resting eye, which is exactly the class of error this file exists
// to prevent. The settle therefore runs before anything is measured, so the
// number the gate sees is the number the timed kernel produces.
static int settle_state(Dev& d, const Cfg& c) {
  size_t B = (size_t)c.B;
  std::vector<float> prev(B), cur(B);
  CHK(cudaMemcpy(prev.data(), d.pos + 1, B * sizeof(float), cudaMemcpyDeviceToHost));
  int steps = 0;
  for (int i = 0; i < c.settle_iters; i++) {
    launch_step_uint8_prebasis_full(
        d.voxels, d.pos, d.vel, d.yaw, d.pitch, d.on_ground, d.buttons, d.look, d.do_reset,
        d.step_count, d.fwd, d.strafe, d.dyaw, d.dpitch, d.jump, d.do_break, d.do_place,
        d.speed_mult, d.cameras, d.basis, d.obs_u8, d.rewards, c.B, c.W, c.W, c.W, c.Wd, c.H,
        c.max_steps, c.view_dist, c.fov, c.dt, c.gravity, c.walk_speed, c.sprint_mult,
        c.jump_vel, c.target_block, c.reward_value, spawn_x_of(c), spawn_y_of(c),
        spawn_z_of(c), c.episode_len);
    steps++;
    CHK(cudaDeviceSynchronize());
    CHK(cudaGetLastError());
    CHK(cudaMemcpy(cur.data(), d.pos + 1, B * sizeof(float), cudaMemcpyDeviceToHost));
    // y unchanged for every env => nobody is still falling.
    bool settled = true;
    for (size_t e = 0; e < B; e++)
      if (cur[e] != prev[e]) { settled = false; break; }
    if (settled && i > 0) break;
    prev = cur;
  }
  refresh_cameras(d, c);
  launch_precompute_camera_basis(d.cameras, d.basis, c.B, c.Wd, c.H, c.fov);
  CHK(cudaDeviceSynchronize());
  CHK(cudaGetLastError());
  return steps;
}

// ---------------------------------------------------------------------------
// Memory-floor probes
// ---------------------------------------------------------------------------
__global__ void floor_write(uint4* p, size_t n4, uint4 v) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) p[i] = v;
}
__global__ void floor_read(const uint4* __restrict__ p, size_t n4, uint4* sink) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  uint4 acc = make_uint4(0, 0, 0, 0);
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
    uint4 v = p[i];
    acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
  }
  // A sink value the compiler cannot fold away.
  if (acc.x == 0xdeadbeefu && acc.y == 0x12345678u) sink[0] = acc;
}
__global__ void floor_copy(uint4* p, size_t n4) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) p[i] = p[i];
}

// ---------------------------------------------------------------------------
// DDA STEP PROBE
// ---------------------------------------------------------------------------
// WHY THIS EXISTS. The render kernel's cost is the number of DDA iterations it
// executes, and nothing in the timed output reports that number. A kernel that
// hits at step 0 and a kernel that marches 23 steps produce the same image
// size, the same bytes written, and the same "PASS"; only the step count
// separates them. The shape this study first measured was degenerate: the
// agent's eye was inside solid dirt, so render_kernel_uint8_prebasis hit on
// iteration 0 for EVERY pixel -- one __ldg and one store -- and the reported
// 17.96 M env-steps/s was the throughput of a stub, not of a voxel renderer.
//
// WHY IT IS A TRANSCRIPTION AND NOT A PARAPHERNALIA CHANGE. The step count has
// to come from the DDA the timed kernel actually runs, so this kernel is
// render_kernel_uint8_prebasis (kernels.cu:568-700) with exactly one addition:
// a counter, incremented in place of the early-exit, accumulated per block and
// then per launch. The DDA body, the load, the bounds test and the step order
// are byte-for-byte the vendored macro. The vendored kernels.cu is untouched;
// this lives in the harness, next to the other probes.
//
// It writes NO image and does not perturb the timing battery: it has its own
// buffer, its own launcher and its own counters.

// Per-launch accumulators, so no cudaMalloc per probe.
__device__ unsigned long long g_dda_steps = 0;  // sum of DDA iterations
__device__ unsigned long long g_dda_pixels = 0; // pixels that marched at all
__device__ unsigned long long g_dda_hist[65] = {0};
__device__ unsigned long long g_dda_eye_solid = 0;  // pixels whose FIRST voxel is solid

__global__ void dda_probe_reset() {
  g_dda_steps = 0;
  g_dda_pixels = 0;
  g_dda_eye_solid = 0;
  for (int i = 0; i <= 64; i++) g_dda_hist[i] = 0;
}

__global__ void dda_probe_kernel(const int8_t* __restrict__ voxels,
                                 const float* __restrict__ basis,  // [B, 14]
                                 int batch_size, int world_x, int world_y, int world_z,
                                 int width, int height, int max_steps) {
  __shared__ float s_basis[14];
  // Same 2D grid as the vendored kernel: blockIdx.y = batch, blockIdx.x = pixel
  // block. Identical thread->pixel mapping, so the probe walks the same rays in
  // the same order on the same data.
  int b = blockIdx.y;
  int pixel_offset = blockIdx.x * blockDim.x + threadIdx.x;
  int pixels_per_batch = height * width;
  // NOTE: no early `return` here. The block reduction below uses __syncthreads(),
  // and a thread that returned before it would leave that barrier with a
  // mismatched arrival count. Out-of-range threads clamp to a valid pixel, walk
  // it for real, and then contribute 0 steps and 0 pixels to the reduction, so
  // the counters describe exactly the live pixels.
  bool live = (b < batch_size) && (pixel_offset < pixels_per_batch);
  if (!live) { b = 0; pixel_offset = 0; }

  int py = pixel_offset / width;
  int px = pixel_offset % width;

  if (threadIdx.x < 14) s_basis[threadIdx.x] = basis[b * 14 + threadIdx.x];
  __syncthreads();

  float eye_x = s_basis[0], eye_y = s_basis[1], eye_z = s_basis[2];
  float forward_x = s_basis[3], forward_y = s_basis[4], forward_z = s_basis[5];
  float right_x = s_basis[6], right_z = s_basis[7];
  float up_x = s_basis[8], up_y = s_basis[9], up_z = s_basis[10];
  float scale_u = s_basis[11], scale_v = s_basis[12];

  float u = (2.0f * (px + 0.5f) / width - 1.0f) * scale_u;
  float v = (1.0f - 2.0f * (py + 0.5f) / height) * scale_v;

  float ray_x = forward_x + u * right_x + v * up_x;
  float ray_y = forward_y + v * up_y;
  float ray_z = forward_z + u * right_z + v * up_z;

  float ray_len = rsqrtf(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z);
  ray_x *= ray_len;
  ray_y *= ray_len;
  ray_z *= ray_len;

  int wx = world_x, wy = world_y, wz = world_z;
  long long wxyz = (long long)wx * wy * wz;
  int wxz = wx * wz;

  int voxel_x = (int)floorf(eye_x);
  int voxel_y = (int)floorf(eye_y);
  int voxel_z = (int)floorf(eye_z);

  int step_x = (ray_x >= 0) ? 1 : -1;
  int step_y = (ray_y >= 0) ? 1 : -1;
  int step_z = (ray_z >= 0) ? 1 : -1;

  float inv_ray_x = (ray_x != 0.0f) ? 1.0f / ray_x : 1e30f;
  float inv_ray_y = (ray_y != 0.0f) ? 1.0f / ray_y : 1e30f;
  float inv_ray_z = (ray_z != 0.0f) ? 1.0f / ray_z : 1e30f;

  float t_delta_x = fabsf(inv_ray_x);
  float t_delta_y = fabsf(inv_ray_y);
  float t_delta_z = fabsf(inv_ray_z);

  float t_max_x = (ray_x >= 0) ? ((float)(voxel_x + 1) - eye_x) * inv_ray_x
                               : ((float)voxel_x - eye_x) * inv_ray_x;
  float t_max_y = (ray_y >= 0) ? ((float)(voxel_y + 1) - eye_y) * inv_ray_y
                               : ((float)voxel_y - eye_y) * inv_ray_y;
  float t_max_z = (ray_z >= 0) ? ((float)(voxel_z + 1) - eye_z) * inv_ray_z
                               : ((float)voxel_z - eye_z) * inv_ray_z;

  int step = 0;
  int solid_first = 0;

  // The vendored DDA body (kernels.cu:664-679) with `goto done_pre` replaced by
  // a break and a step tally. The tally is the ONLY difference.
  #pragma unroll 1
  for (int i = 0; i < max_steps; i++) {
    if (step >= max_steps) break;
    if ((unsigned)voxel_x < (unsigned)wx && (unsigned)voxel_y < (unsigned)wy &&
        (unsigned)voxel_z < (unsigned)wz) {
      int8_t block = __ldg(&voxels[b * wxyz + voxel_y * wxz + voxel_x * wz + voxel_z]);
      if (block >= 0) {
        if (step == 0) solid_first = 1;
        break;
      }
    }
    if (t_max_x < t_max_y && t_max_x < t_max_z) {
      t_max_x += t_delta_x; voxel_x += step_x;
    } else if (t_max_y < t_max_z) {
      t_max_y += t_delta_y; voxel_y += step_y;
    } else {
      t_max_z += t_delta_z; voxel_z += step_z;
    }
    ++step;
  }

  // `step` is the number of ADVANCES taken, so the number of voxels TESTED is
  // step+1 when a hit ended the walk, and max_steps when the budget ran out.
  // The vendored kernel's cost is the number of tests, and dda_model.py counts
  // tests, so count tests here too: a buried eye lands on exactly 1.
  int tests = (step < max_steps) ? (step + 1) : max_steps;
  if (!live) { tests = 0; solid_first = 0; }

  // Block reduction, then one atomic per block. At batch 32768 x 1024 px this
  // is 33.5M pixels; a per-pixel atomicAdd onto a single counter would
  // serialise on one L2 line. 256 slots covers blockDim up to 256, which is
  // every block size the render launchers in kernels.cu use.
  __shared__ unsigned long long s_steps[256];
  __shared__ unsigned long long s_pix[256];
  __shared__ unsigned long long s_solid[256];
  // A SHARED histogram, because the global one has to be fed per pixel. An
  // earlier version fed it from thread 0 only, which recorded one entry per
  // BLOCK carrying thread 0's step count: the bucket totals summed to ~3% of
  // the pixel count and p50/p90 came back unset. The mean was unaffected
  // (s_steps is summed rather than sampled), so the gate verdict stayed
  // correct -- but every distributional statistic printed beside it was wrong,
  // which is worse than not printing it.
  __shared__ unsigned long long s_hist[65];
  int tid = threadIdx.x;
  if (tid < 65) s_hist[tid] = 0ULL;
  __syncthreads();
  if (live) atomicAdd(&s_hist[tests < 65 ? tests : 64], 1ULL);
  s_steps[tid] = (unsigned long long)tests;
  s_pix[tid] = live ? 1ULL : 0ULL;
  s_solid[tid] = (unsigned long long)solid_first;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_steps[tid] += s_steps[tid + s];
      s_pix[tid] += s_pix[tid + s];
      s_solid[tid] += s_solid[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    atomicAdd(&g_dda_steps, s_steps[0]);
    atomicAdd(&g_dda_pixels, s_pix[0]);
    atomicAdd(&g_dda_eye_solid, s_solid[0]);
  }
  // One global atomic per NON-EMPTY bucket, so a block that used a dozen
  // distinct step counts costs a dozen atomics rather than 65. The probe is a
  // one-shot measurement outside every timing loop, so its own cost cannot
  // contaminate the numbers it reports.
  if (tid < 65 && s_hist[tid] != 0ULL) atomicAdd(&g_dda_hist[tid], s_hist[tid]);
}

struct DdaStat {
  double mean = 0;
  int p50 = 0, p90 = 0, max = 0;  // 0 means "no pixel had that few/many"
  unsigned long long pixels = 0, at_cap = 0, eye_solid = 0;
  bool cap_clamped = false;  // max_steps >= 64, so the histogram saturates
  unsigned long long hist[65] = {0};
};

// Run the probe over the current device state: the voxels the render kernel
// reads and the basis the render kernel reads, both straight off the device,
// so this counts the rays the timed kernel is about to trace.
static DdaStat measure_dda(const Dev& d, const Cfg& c) {
  int pixels_per_batch = c.Wd * c.H;
  int block = 128;
  int gx = (pixels_per_batch + block - 1) / block;
  dda_probe_reset<<<1, 1>>>();
  dda_probe_kernel<<<dim3(gx, c.B), block>>>(d.voxels, d.basis, c.B, c.W, c.W, c.W, c.Wd, c.H,
                                            c.max_steps);
  CHK(cudaDeviceSynchronize());
  CHK(cudaGetLastError());

  unsigned long long h_steps = 0, h_pix = 0, h_solid = 0, h_hist[65] = {0};
  CHK(cudaMemcpyFromSymbol(&h_steps, g_dda_steps, sizeof h_steps));
  CHK(cudaMemcpyFromSymbol(&h_pix, g_dda_pixels, sizeof h_pix));
  CHK(cudaMemcpyFromSymbol(&h_solid, g_dda_eye_solid, sizeof h_solid));
  CHK(cudaMemcpyFromSymbol(h_hist, g_dda_hist, sizeof h_hist));

  DdaStat s;
  s.pixels = h_pix;
  s.eye_solid = h_solid;
  s.mean = h_pix ? (double)h_steps / (double)h_pix : 0.0;
  // Bucket 64 is the overflow bucket: the kernel clamps any test count above 64
  // into it. A max_steps of 128 therefore lands entirely in bucket 64, and
  // reading the cap from bucket `max_steps` would read an empty bucket and
  // report 0%. Read the cap from the clamped index instead.
  s.at_cap = h_hist[c.max_steps < 64 ? c.max_steps : 64];
  s.cap_clamped = (c.max_steps >= 64);
  // Percentiles and max in one pass, with no early exit: bailing out at p90
  // would leave s.max describing the p90 bucket rather than the largest seen.
  // A percentile is -1 only if the population is empty, which cannot happen
  // here: every rendered pixel is tested at least once, so 0 is never a real
  // value in this histogram and an unset percentile is always a bug.
  double want50 = 0.50 * (double)h_pix, want90 = 0.90 * (double)h_pix, cum = 0.0;
  unsigned long long hist_total = 0;
  for (int k = 0; k <= 64; k++) hist_total += h_hist[k];
  // The histogram must account for every counted pixel. It did not, once: an
  // earlier kernel fed it one entry per block, the total came to 3% of the
  // pixel count, and the only visible symptom was unset percentiles. Assert
  // the invariant so that a recurrence is loud and named rather than a column
  // of zeros next to a correct mean.
  if (h_pix != 0 && hist_total != h_pix) {
    fprintf(stderr,
            "FATAL %s:%d dda_probe histogram covers %llu of %llu pixels; bucket "
            "accounting is broken, so p50/p90/at-cap would be meaningless\n",
            __FILE__, __LINE__, hist_total, h_pix);
    exit(3);
  }
  s.p50 = s.p90 = -1;
  for (int k = 0; k <= 64; k++) {
    s.hist[k] = h_hist[k];
    if (h_hist[k] == 0) continue;
    s.max = k;
    cum += (double)h_hist[k];
    if (s.p50 < 0 && cum >= want50) s.p50 = k;
    if (s.p90 < 0 && cum >= want90) s.p90 = k;
  }
  return s;
}

struct Stat {
  double min = 0, med = 0, mean = 0, max = 0, std = 0;
};

static Stat stats_of(std::vector<double> v) {
  Stat s;
  if (v.empty()) return s;
  std::sort(v.begin(), v.end());
  s.min = v.front();
  s.max = v.back();
  s.med = (v.size() % 2) ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
  double sum = 0;
  for (double x : v) sum += x;
  s.mean = sum / (double)v.size();
  double ss = 0;
  for (double x : v) ss += (x - s.mean) * (x - s.mean);
  s.std = v.size() > 1 ? sqrt(ss / (double)(v.size() - 1)) : 0.0;
  return s;
}

// Time `iters` back-to-back launches, events on the default stream, median of
// per-launch event pairs. Per-launch events (not one event around the loop) so
// the distribution and the outliers are visible.
template <typename F>
static std::vector<double> time_each(int warmup, int iters, F f) {
  for (int i = 0; i < warmup; i++) f();
  CHK(cudaDeviceSynchronize());
  std::vector<double> v;
  v.reserve((size_t)iters);
  cudaEvent_t a, b;
  CHK(cudaEventCreate(&a));
  CHK(cudaEventCreate(&b));
  for (int i = 0; i < iters; i++) {
    CHK(cudaEventRecord(a, 0));
    f();
    CHK(cudaEventRecord(b, 0));
    CHK(cudaEventSynchronize(b));
    float ms = 0.f;
    CHK(cudaEventElapsedTime(&ms, a, b));
    v.push_back((double)ms * 1000.0);
  }
  CHK(cudaEventDestroy(a));
  CHK(cudaEventDestroy(b));
  CHK(cudaGetLastError());
  return v;
}

static int g_fail = 0;
// Set only by the DDA gate in selftest() when --measure-degenerate waived the
// abort on a below-floor configuration. Every table header in the timing
// section consults this, so a degenerate run cannot be read as a clean one.
static bool g_dda_degenerate = false;
static void check(bool ok, const char* name, const std::string& detail) {
  printf("  [%s] %-44s %s\n", ok ? "PASS" : "FAIL", name, detail.c_str());
  if (!ok) g_fail++;
}

static void hdr(const char* s) {
  printf("\n%s\n", s);
  for (size_t i = 0; i < strlen(s); i++) putchar('=');
  putchar('\n');
}

// ---------------------------------------------------------------------------
// selftest
// ---------------------------------------------------------------------------
static int selftest(Dev& d, const Cfg& c) {
  size_t B = (size_t)c.B, wv = wvox_of(c), hw = hw_of(c);
  char buf[320];
  printf("== selftest (on this device, this build) ==\n");

  {  // 1. world generation
    CHK(cudaMemcpy(d.h_vox.data(), d.voxels, B * wv, cudaMemcpyDeviceToHost));
    size_t bad = 0, air = 0, dirt = 0, logn = 0, leaf = 0, brick = 0, cobble = 0, planks = 0;
    for (size_t i = 0; i < B * wv; i++) {
      int8_t v = d.h_vox[i];
      if (v < -1 || v > 12) bad++;
      if (v == -1) air++;
      if (v == 1) dirt++;
      if (v == 3) logn++;
      if (v == 4) leaf++;
      if (v == 8) brick++;
      if (v == 9) cobble++;
      if (v == 10) planks++;
    }
    snprintf(buf, sizeof buf,
             "ids all in [-1,12]; air=%zu dirt=%zu log=%zu leaf=%zu brick=%zu cobble=%zu planks=%zu",
             air, dirt, logn, leaf, brick, cobble, planks);
    check(bad == 0, "generate_world_kernel: every id in range", buf);
    snprintf(buf, sizeof buf, "log=%d leaf=%d", logn > 0, leaf > 0);
    check(logn > 0 && leaf > 0, "place_tree_kernel: a log and a leaf exist", buf);
    snprintf(buf, sizeof buf, "cobble=%d planks=%d brick=%d", cobble > 0, planks > 0, brick > 0);
    check(cobble > 0 && planks > 0 && brick > 0,
          "place_house_kernel: floor, walls and roof exist", buf);
  }

  {  // 2. decode + physics
    float y0 = 0.f, x0 = 0.f, y1 = 0.f, x1 = 0.f;
    CHK(cudaMemcpy(&y0, d.pos + 1, sizeof(float), cudaMemcpyDeviceToHost));
    CHK(cudaMemcpy(&x0, d.pos + 0, sizeof(float), cudaMemcpyDeviceToHost));
    launch_decode_actions(d.buttons, d.look, d.fwd, d.strafe, d.dyaw, d.dpitch, d.jump,
                          d.do_break, d.do_place, d.speed_mult, c.B, c.sprint_mult);
    launch_physics_v2(d.pos, d.vel, d.yaw, d.pitch, d.on_ground, d.fwd, d.strafe, d.dyaw,
                      d.dpitch, d.jump, d.speed_mult, c.B, c.W, c.dt, c.gravity, c.walk_speed,
                      c.jump_vel);
    CHK(cudaDeviceSynchronize());
    CHK(cudaGetLastError());
    CHK(cudaMemcpy(&y1, d.pos + 1, sizeof(float), cudaMemcpyDeviceToHost));
    CHK(cudaMemcpy(&x1, d.pos + 0, sizeof(float), cudaMemcpyDeviceToHost));
    snprintf(buf, sizeof buf, "x %.4f -> %.4f, y %.4f -> %.4f", x0, x1, y0, y1);
    check(x1 != x0 || y1 != y0, "physics_kernel moved the agent", buf);
    // kernels.cu:1566 collides the agent against a flat analytic plane at
    // world_y/4. It does NOT collide against the terrain: worldgen puts the
    // visible surface's top face at ground_height (kernels.cu:1823). The two are
    // independent, and when they disagree the outcome INVERTS with world size:
    // at W=16 with ground=8 the floor (4) is below the surface (8) and the agent
    // ends up buried; at W=48 with ground=8 the spawn (10) is below the floor
    // (12), so the clamp pushes the agent UP and gravity is unobservable.
    //
    // With the new default (ground = world_y/4) the spawn is always ground+2,
    // which is 2 above the floor at every world size, so the gravity branch is
    // the one that runs. The clamp branch is still reachable -- and still a real
    // assertion -- for any config that passes --ground explicitly and lands the
    // spawn at or under the floor. Neither branch is skipped.
    float floor_y = (float)(c.W / 4);
    if (y0 > floor_y) {
      snprintf(buf, sizeof buf, "y1=%.5f < y0=%.5f (gravity=-%.1f, dt=%.3f)", y1, y0, -c.gravity,
               c.dt);
      check(y1 < y0, "physics_kernel applied gravity downward", buf);
    } else {
      snprintf(buf, sizeof buf,
               "spawn y0=%.2f is at/below the physics floor world_y/4=%.2f so the clamp "
               "dominates; y1=%.5f must be the floor. reference defect: floor=world_y/4 but "
               "the visible surface is ground_h-1=%.0f",
               y0, floor_y, y1, (double)(ground_of(c) - 1));
      check(fabsf(y1 - floor_y) < 1e-5f, "physics_kernel clamped to its floor y=world_y/4", buf);
    }
    // Unconditional, and true at every shape: the agent is never left under the
    // collision plane, and the plane really is world_y/4.
    snprintf(buf, sizeof buf, "y1=%.5f must be >= floor world_y/4=%.2f", y1, floor_y);
    check(y1 >= floor_y - 1e-5f, "physics_kernel never leaves the agent below the floor", buf);
    snprintf(buf, sizeof buf, "x=%.4f must be in [1,%d]", x1, c.W - 1);
    check(x1 >= 1.0f && x1 <= (float)c.W - 1.0f, "physics_kernel clamped x to the world box", buf);
  }

  {  // 3. camera basis precompute
    refresh_cameras(d, c);
    launch_precompute_camera_basis(d.cameras, d.basis, c.B, c.Wd, c.H, c.fov);
    CHK(cudaDeviceSynchronize());
    CHK(cudaGetLastError());
    std::vector<float> hb(B * 14);
    CHK(cudaMemcpy(hb.data(), d.basis, B * 14 * sizeof(float), cudaMemcpyDeviceToHost));
    // The contract is basis[1] == cameras[1] + 1.6 (kernels.cu:541), so the
    // expectation is derived from the cam_y that refresh_cameras just wrote --
    // NOT from spawn_y. Check 2 already ran one physics step, so by now the
    // agent is at 9.95, not the 10.0 spawn, and a hardcoded spawn_y + 1.6 is
    // stale by exactly the distance gravity has moved the agent. That mismatch
    // (want 11.6000, got 11.5500) was a stale expectation in this harness, not
    // a kernel bug; the kernel did exactly what its header comment says.
    std::vector<float> hcam(5);
    CHK(cudaMemcpy(hcam.data(), d.cameras, 5 * sizeof(float), cudaMemcpyDeviceToHost));
    float cam_y = hcam[1];
    float want_eye_y = cam_y + 1.6f;
    snprintf(buf, sizeof buf, "eye_y want %.4f got %.4f (cam_y %.4f + 1.6; spawn_y is %.4f, stale)",
             want_eye_y, hb[1], cam_y, (double)spawn_y_of(c));
    check(fabsf(want_eye_y - hb[1]) < 1e-3f, "precompute_camera_basis wrote the eye", buf);
    float half_fov = tanf(c.fov * 0.01745329f * 0.5f);
    float want_u = half_fov * ((float)c.Wd / (float)c.H);
    snprintf(buf, sizeof buf, "scale_u want %.6f got %.6f", want_u, hb[11]);
    check(fabsf(want_u - hb[11]) < 1e-4f, "precompute_camera_basis wrote scale_u", buf);
    snprintf(buf, sizeof buf, "scale_v want %.6f got %.6f", half_fov, hb[12]);
    check(fabsf(half_fov - hb[12]) < 1e-4f, "precompute_camera_basis wrote scale_v", buf);
  }

  {  // 4. uint8 prebasis render
    launch_render_uint8_prebasis(d.voxels, d.basis, d.obs_u8, c.B, c.W, c.Wd, c.H, c.max_steps);
    CHK(cudaDeviceSynchronize());
    CHK(cudaGetLastError());
    d.h_u3.assign(hw * 3, 0);
    CHK(cudaMemcpy(d.h_u3.data(), d.obs_u8, hw * 3, cudaMemcpyDeviceToHost));
    bool seen[256] = {false};
    for (size_t i = 0; i < hw * 3; i++) seen[d.h_u3[i]] = true;
    int nd = 0;
    for (int i = 0; i < 256; i++) nd += seen[i];
    size_t sky = 0;
    for (size_t i = 0; i + 2 < hw * 3; i += 3)
      if (d.h_u3[i] == 135 && d.h_u3[i + 1] == 206 && d.h_u3[i + 2] == 235) sky++;
    snprintf(buf, sizeof buf, "env 0: %d distinct byte values, %zu/%zu sky pixels", nd, sky, hw);
    check(nd >= 2, "prebasis render wrote more than one colour", buf);
    snprintf(buf, sizeof buf, "%zu sky pixels, want > 0", sky);
    check(sky > 0, "prebasis render contains SKY_COLOR_U8 {135,206,235}", buf);
  }

  {  // 5. the uint8 variants agree with each other on the sky pixels
    launch_render_uint8_minimal(d.voxels, d.cameras, d.obs_u8b, c.B, c.W, c.Wd, c.H, c.max_steps,
                                c.view_dist, c.fov);
    CHK(cudaDeviceSynchronize());
    CHK(cudaGetLastError());
    std::vector<uint8_t> b(hw * 3, 0);
    CHK(cudaMemcpy(b.data(), d.obs_u8b, hw * 3, cudaMemcpyDeviceToHost));
    size_t sa = 0, sb = 0, both = 0;
    for (size_t i = 0; i + 2 < hw * 3; i += 3) {
      bool A = (d.h_u3[i] == 135 && d.h_u3[i + 1] == 206 && d.h_u3[i + 2] == 235);
      bool Bb = (b[i] == 135 && b[i + 1] == 206 && b[i + 2] == 235);
      sa += A; sb += Bb; both += (A && Bb);
    }
    snprintf(buf, sizeof buf, "prebasis sky %zu, minimal sky %zu, agree %zu", sa, sb, both);
    check(both > 0, "prebasis and minimal agree on sky pixels", buf);
  }

  {  // 6. float32 render
    launch_render(d.voxels, d.cameras, d.obs_f32, c.B, c.W, c.Wd, c.H, c.max_steps, c.view_dist,
                  c.fov);
    CHK(cudaDeviceSynchronize());
    CHK(cudaGetLastError());
    d.h_f3.assign(hw * 3, 0.f);
    CHK(cudaMemcpy(d.h_f3.data(), d.obs_f32, hw * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    int nonfinite = 0, oob = 0;
    for (size_t i = 0; i < hw * 3; i++) {
      if (!isfinite(d.h_f3[i])) nonfinite++;
      if (d.h_f3[i] < -1e-3f || d.h_f3[i] > 1.001f) oob++;
    }
    snprintf(buf, sizeof buf, "non-finite=%d out-of-[0,1]=%d of %zu values", nonfinite, oob, hw * 3);
    check(nonfinite == 0 && oob == 0, "float32 obs finite and inside [0,1]", buf);
  }

  {  // 7. block break and reward
    // The reward is gated on the block type. kernels.cu:1662:
    //     if (target_block < 0 || block == target_block) rewards[i] = reward_value;
    // and the break itself is unconditional (voxels[idx] = AIR, kernels.cu:1665).
    // So "a voxel was mutated but no reward was paid" is the CORRECT and expected
    // observation whenever the ray hits a block that is not the target -- the
    // interaction fired, the payout did not. cfg.target_block is OAKLOG (3).
    //
    // This check previously aimed a level ray down +z ("the tree is at
    // z_spawn + 2") and got 4096 mutations with 0 rewards. It was not digging the
    // wrong block by accident of geometry alone: the eye sits at pos + 1.6
    // (kernels.cu:1630) = spawn_y + 1.6 = ground_h + 3.6, and init_state's tree
    // is, relative to ground_height (place_tree_kernel, kernels.cu:1851-1863),
    //   logs   at y = ground_h + 0 .. ground_h + 2
    //   leaves at y = ground_h + 3
    // A level ray from the eye therefore travels at voxel row ground_h+3 for its
    // whole 4-block reach, so it strikes the LEAF canopy (id 4), not the trunk
    // (id 3): mutate, no payout. Exactly what was seen. That is why the negative
    // case below pins pitch=0 rather than aiming.
    //
    // So: aim at the middle log, and derive yaw/pitch from where the agent
    // actually is after check 2's physics step, because the eye offset and the
    // fall both move the target. Then check both directions of the contract --
    // target block pays, non-target block does not.
    // All of this is expressed RELATIVE to ground_h, so it holds at every world
    // size and at both the derived and an explicit --ground.
    std::vector<int8_t> hb(B * 8, 0);
    std::vector<float> hl(B * 2, 0.f), hy(B, 0.f), hi(B, 0.f);
    // hpos/hyaw/hpitch are the LIVE state, kept aside so this check can put it
    // back exactly as it found it; hdig is where the dig starts from.
    std::vector<float> hpos(B * 3), hdig(B * 3), hy_saved(B), hi_saved(B);
    CHK(cudaMemcpy(hpos.data(), d.pos, B * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    CHK(cudaMemcpy(hy_saved.data(), d.yaw, B * sizeof(float), cudaMemcpyDeviceToHost));
    CHK(cudaMemcpy(hi_saved.data(), d.pitch, B * sizeof(float), cudaMemcpyDeviceToHost));

    // Put the agent on the spawn before digging, so this check does not depend on
    // whatever check 2's physics step left behind. Same class of bug as the stale
    // eye_y expectation: an expectation read off a state you are not controlling.
    // The spawn is one block above the tree's base, and the tree is 2 blocks in
    // +z, so the trunk is comfortably inside the 4.0 reach at every shape.
    const float kEye = 1.6f;                              // kernels.cu:1630
    float ex = spawn_x_of(c), ey = spawn_y_of(c) + kEye, ez = spawn_z_of(c);
    float tx = spawn_x_of(c) + 0.5f;                      // tree trunk column
    float ty = (float)(ground_of(c) + 1) + 0.5f;            // middle of 3 logs
    float tz = spawn_z_of(c) + 2.0f + 0.5f;                // init_state: z_spawn+2
    float dx = tx - ex, dy = ty - ey, dz = tz - ez;
    // raycast_interact builds dir = (sin yaw cos p, sin p, cos yaw cos p), so this
    // choice of angles makes dir parallel to (dx, dy, dz) exactly.
    float dig_yaw = atan2f(dx, dz);
    float dig_pitch = atan2f(dy, sqrtf(dx * dx + dz * dz));
    float dig_dist = sqrtf(dx * dx + dy * dy + dz * dz);
    // The kernel marches t in [0.1, 0.1 + 31*(reach/32)] with reach = 4.0
    // (kernels.cu:1633,1644), so a target farther than ~3.98 is unreachable.
    printf("    (dig aim: log (%.1f,%.1f,%.1f) from eye (%.2f,%.2f,%.2f), dist %.2f of 4.00 "
           "reach, yaw %.4f pitch %.4f)\n",
           tx, ty, tz, ex, ey, ez, dig_dist, dig_yaw, dig_pitch);
    snprintf(buf, sizeof buf,
             "aim at log (%.1f,%.1f,%.1f) from eye (%.2f,%.2f,%.2f), dist %.2f of 4.00 reach",
             tx, ty, tz, ex, ey, ez, dig_dist);

    for (size_t i = 0; i < B; i++) {
      hb[i * 8 + 5] = 1;  // BTN_BREAK
      hy[i] = dig_yaw;
      hi[i] = dig_pitch;
      hdig[i * 3 + 0] = spawn_x_of(c);
      hdig[i * 3 + 1] = spawn_y_of(c);
      hdig[i * 3 + 2] = spawn_z_of(c);
    }
    CHK(cudaMemcpy(d.buttons, hb.data(), B * 8, cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d.look, hl.data(), B * 2 * sizeof(float), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d.yaw, hy.data(), B * sizeof(float), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d.pitch, hi.data(), B * sizeof(float), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d.pos, hdig.data(), B * 3 * sizeof(float), cudaMemcpyHostToDevice));
    launch_decode_actions(d.buttons, d.look, d.fwd, d.strafe, d.dyaw, d.dpitch, d.jump,
                          d.do_break, d.do_place, d.speed_mult, c.B, c.sprint_mult);
    launch_raycast_interact(d.voxels, d.pos, d.yaw, d.pitch, d.do_break, d.do_place, d.rewards,
                            c.B, c.W, c.target_block, c.reward_value);
    CHK(cudaDeviceSynchronize());
    CHK(cudaGetLastError());
    CHK(cudaMemcpy(d.h_rew.data(), d.rewards, B * sizeof(float), cudaMemcpyDeviceToHost));
    size_t nrew = 0;
    double rewsum = 0.0;
    for (size_t i = 0; i < B; i++)
      if (d.h_rew[i] > 0.f) { nrew++; rewsum += d.h_rew[i]; }
    CHK(cudaMemcpy(d.h_vox2.data(), d.voxels, B * wv, cudaMemcpyDeviceToHost));
    size_t nmut = 0;
    for (size_t i = 0; i < B * wv; i++)
      if (d.h_vox2[i] != d.h_vox[i]) nmut++;
    // Which trunk voxel went to air in env 0? The ray descends through the
    // column, so it may land on any of the 3 logs depending on float rounding;
    // the contract is "one log from the trunk column became air", so accept any
    // of them rather than pinning one index.
    int broke_log = 0, nlog_broken = 0;
    for (int dy = 0; dy < 3; ++dy) {
      long long li = (long long)(ground_of(c) + dy) * c.W * c.W + (int)spawn_x_of(c) * c.W +
                     (int)spawn_z_of(c) + 2;
      if (d.h_vox[li] == 3 && d.h_vox2[li] == -1) { nlog_broken++; broke_log = ground_of(c) + dy; }
    }
    snprintf(buf, sizeof buf,
             "%zu of %zu envs rewarded (mean %.2f, target_block=%d), %zu voxels mutated, "
             "%d trunk log(s) of the column now air (row %d)",
             nrew, B, nrew ? rewsum / (double)nrew : 0.0, (int)c.target_block, nmut,
             nlog_broken, broke_log);
    // Every env must be paid exactly cfg.reward_value, exactly one voxel must go
    // to air per env, and that voxel must have been OAKLOG (3) in the trunk
    // column and must now be AIR (-1).
    check(nrew == B && nmut == B && nlog_broken == 1 &&
              fabs(rewsum - (double)B * (double)c.reward_value) < 1e-3,
          "raycast_interact_kernel broke the target log and paid reward", buf);

    // Negative case: the same dig into the leaf canopy (id 4) with the same
    // target_block (OAKLOG=3) must still mutate and must NOT pay. Pins the
    // gating down in both directions instead of only the paying one.
    // d.h_vox is reused here as the before-image scratch: check 1's census has
    // already consumed it, and a third full-world host buffer would be another
    // 1 GiB at batch 32768 / world 32.
    CHK(cudaMemcpy(d.h_vox.data(), d.voxels, B * wv, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < B; i++) hi[i] = 0.f;  // level ray -> voxel row 11 = leaves
    CHK(cudaMemcpy(d.pitch, hi.data(), B * sizeof(float), cudaMemcpyHostToDevice));
    launch_raycast_interact(d.voxels, d.pos, d.yaw, d.pitch, d.do_break, d.do_place, d.rewards,
                            c.B, c.W, c.target_block, c.reward_value);
    CHK(cudaDeviceSynchronize());
    CHK(cudaGetLastError());
    CHK(cudaMemcpy(d.h_rew.data(), d.rewards, B * sizeof(float), cudaMemcpyDeviceToHost));
    size_t nrew2 = 0;
    for (size_t i = 0; i < B; i++)
      if (d.h_rew[i] > 0.f) nrew2++;
    CHK(cudaMemcpy(d.h_vox2.data(), d.voxels, B * wv, cudaMemcpyDeviceToHost));
    size_t nmut2 = 0, nleaf_gone = 0;
    for (size_t i = 0; i < B * wv; i++)
      if (d.h_vox2[i] != d.h_vox[i]) nmut2++;
    for (size_t e = 0; e < B; e++) {
      long long li = (long long)(ground_of(c) + 3) * c.W * c.W + (int)spawn_x_of(c) * c.W +
                     (int)spawn_z_of(c) + 2;
      if (d.h_vox[li] == 4 && d.h_vox2[li] == -1) nleaf_gone++;
    }
    snprintf(buf, sizeof buf,
             "%zu of %zu envs rewarded (want 0), %zu voxels mutated (want %zu), "
             "%zu leaf canopies broken",
             nrew2, B, nmut2, B, nleaf_gone);
    check(nrew2 == 0 && nmut2 == B && nleaf_gone == B,
          "raycast_interact_kernel broke a non-target block and paid nothing", buf);

    // Undo the digging, and put the tree and house back: the previous version
    // re-ran worldgen only, which silently removed the tree and house that
    // init_state had placed, so every later check and the whole timing battery
    // ran against a world that was not the documented one.
    launch_generate_world(d.voxels, c.B, c.W, c.W, c.W, ground_of(c), 42u);
    launch_place_tree(d.voxels, c.B, c.W, c.W, c.W, (int)spawn_x_of(c), (int)spawn_z_of(c) + 2,
                      ground_of(c));
    launch_place_house(d.voxels, c.B, c.W, c.W, c.W, (int)spawn_x_of(c) + 6,
                       (int)spawn_z_of(c) - 2, ground_of(c));
    CHK(cudaDeviceSynchronize());
    CHK(cudaGetLastError());
    // restore the live state this check overwrote, so the later checks and the
    // timing battery see exactly what they saw before check 7 ran
    CHK(cudaMemcpy(d.yaw, hy_saved.data(), B * sizeof(float), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d.pitch, hi_saved.data(), B * sizeof(float), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d.pos, hpos.data(), B * 3 * sizeof(float), cudaMemcpyHostToDevice));
    // put the real action stream back
    std::vector<int8_t> rb(B * 8, 0);
    std::vector<float> rl(B * 2, 0.f);
    for (size_t i = 0; i < B; i++) {
      rb[i * 8 + 0] = 1;
      rl[i * 2 + 0] = 0.002f * (float)((i % 97) - 48);
      rb[i * 8 + 5] = (int8_t)((i % 32) == 0 ? 1 : 0);
    }
    CHK(cudaMemcpy(d.buttons, rb.data(), B * 8, cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d.look, rl.data(), B * 2 * sizeof(float), cudaMemcpyHostToDevice));
  }

  {  // 8. reset
    std::vector<uint8_t> m(B, 1);
    CHK(cudaMemcpy(d.do_reset, m.data(), B, cudaMemcpyHostToDevice));
    launch_reset(d.pos, d.vel, d.yaw, d.pitch, d.on_ground, d.do_reset, c.B, spawn_x_of(c),
                 spawn_y_of(c), spawn_z_of(c));
    CHK(cudaDeviceSynchronize());
    CHK(cudaGetLastError());
    std::vector<float> h(B * 3);
    CHK(cudaMemcpy(h.data(), d.pos, B * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (size_t i = 0; i < B; i++)
      if (fabsf(h[i * 3 + 0] - spawn_x_of(c)) > 1e-5f ||
          fabsf(h[i * 3 + 1] - spawn_y_of(c)) > 1e-5f ||
          fabsf(h[i * 3 + 2] - spawn_z_of(c)) > 1e-5f)
        ok = false;
    snprintf(buf, sizeof buf, "spawn (%.1f,%.1f,%.1f) for all %zu envs", spawn_x_of(c),
             spawn_y_of(c), spawn_z_of(c), B);
    check(ok, "reset_kernel restored the spawn point", buf);
    std::vector<uint8_t> z(B, 0);
    CHK(cudaMemcpy(d.do_reset, z.data(), B, cudaMemcpyHostToDevice));
  }

  {  // 9. the CUDA-graph step path
    for (int i = 0; i < 12; i++) {  // past WARMUP_THRESHOLD=10 in kernels.cu
      launch_step_uint8_prebasis_full(
          d.voxels, d.pos, d.vel, d.yaw, d.pitch, d.on_ground, d.buttons, d.look, d.do_reset,
          d.step_count, d.fwd, d.strafe, d.dyaw, d.dpitch, d.jump, d.do_break, d.do_place,
          d.speed_mult, d.cameras, d.basis, d.obs_u8, d.rewards, c.B, c.W, c.W, c.W, c.Wd, c.H,
          c.max_steps, c.view_dist, c.fov, c.dt, c.gravity, c.walk_speed, c.sprint_mult,
          c.jump_vel, c.target_block, c.reward_value, spawn_x_of(c), spawn_y_of(c),
          spawn_z_of(c), c.episode_len);
    }
    CHK(cudaDeviceSynchronize());
    cudaError_t e = cudaGetLastError();
    snprintf(buf, sizeof buf, "cudaGetLastError -> %s", cudaGetErrorString(e));
    check(e == cudaSuccess, "12x launch_step_uint8_prebasis_full, graph path", buf);
    std::vector<uint8_t> h(hw * 3, 0);
    CHK(cudaMemcpy(h.data(), d.obs_u8, hw * 3, cudaMemcpyDeviceToHost));
    bool seen[256] = {false};
    for (size_t i = 0; i < hw * 3; i++) seen[h[i]] = true;
    int nd = 0;
    for (int i = 0; i < 256; i++) nd += seen[i];
    snprintf(buf, sizeof buf, "%d distinct byte values in env 0 after 12 steps", nd);
    check(nd >= 2, "graph step produced a non-constant observation", buf);
  }

  {  // 10. THE DDA GATE: the timed workload is not degenerate
    //
    // WHY THIS CHECK EXISTS. Every other check here asks "is the kernel correct".
    // This one asks "is the kernel doing the work the study is about to compare".
    // A previous run of this harness passed all 19 other checks at world=16,
    // ground=8 and reported 17.96 M env-steps/s -- while the render kernel hit
    // its very first voxel test on every single pixel, because the agent's eye
    // was inside solid dirt. Nothing in the image, the byte counts, the PASS
    // lines or the GB/s figures distinguished that from a real render. The only
    // thing that distinguishes them is the DDA step count, so it is measured and
    // asserted here.
    //
    // It is a real assertion, not a print: a FAIL increments g_fail, which makes
    // selftest() return non-zero, which makes main() print "selftest FAILED: no
    // timings reported" and exit 2 BEFORE the timing battery runs. A degenerate
    // configuration therefore cannot report a throughput at all.
    //
    // --measure-degenerate exists ONLY to reproduce the old broken numbers for
    // the before/after table. It does not lower the floor and does not skip the
    // check: it prints the measured value, marks the run DEGENERATE in the
    // header, and stamps every timing table. The check still fails; only the
    // abort is waived, and only for a run that has announced itself as
    // degenerate in the first two lines of its output.
    int settled = settle_state(d, c);
    DdaStat s = measure_dda(d, c);
    double floor_steps = dda_floor(c);
    double eye_y = (double)(c.W / 4) + 1.6;
    double eye_solid_frac = s.pixels ? 100.0 * (double)s.eye_solid / (double)s.pixels : 0.0;
    // Report the geometry, but do NOT assert ground == world_y/4 on its own: the
    // eye-in-solid check below is the direct assertion, and it fails for the
    // right reason (a voxel is actually solid) rather than because a config
    // number happens to differ.
    snprintf(buf, sizeof buf,
             "settled in %d steps; eye y=%.2f in voxel row %d, surface top face y=%d "
             "(ground=%d, physics floor world_y/4=%d)",
             settled, eye_y, (int)floorf((float)eye_y), ground_of(c), ground_of(c), c.W / 4);
    printf("  ---- geometry: %s ----\n", buf);
    // NOTE: two detail strings, two buffers. They used to share `buf`, so the
    // second snprintf silently replaced the first and both checks printed the
    // same text -- a cross-check that looked like it was checking two things
    // and was checking one. Exactly the failure mode this check exists to
    // catch, caught in the check itself.
    char buf_steps[320], buf_eye[320];
    snprintf(buf_steps, sizeof buf_steps,
             "mean %.3f DDA steps/pixel over %llu pixels, p50=%d p90=%d max=%d, "
             "%llu (%.2f%%) at the %d-step cap, floor %.2f",
             s.mean, s.pixels, s.p50, s.p90, s.max, s.at_cap,
             s.pixels ? 100.0 * (double)s.at_cap / (double)s.pixels : 0.0, c.max_steps,
             floor_steps);
    snprintf(buf_eye, sizeof buf_eye,
             "%.3f%% of pixels hit on their FIRST voxel test (eye in solid rock); the "
             "degenerate config is 100%%, the fixed config is a few %%",
             eye_solid_frac);
    // The two assertions are the same fact seen twice, and they are the whole
    // point of this check. Both must hold for a timing to be reported.
    //
    // WHY THE SECOND IS A FRACTION AND NOT "zero": physics_kernel has no voxel
    // collision, so nothing stops an agent from walking into the tree or the
    // house. A few percent of envs legitimately end a run with their eye inside
    // a log. Requiring exactly zero would fail the FIXED configuration, and
    // "fixing" that by weakening the mean floor would re-admit the real bug.
    // The measured values are ~2-4% fixed and 100% degenerate, so 20% separates
    // them with a 5x margin on the near side and 5x on the far.
    bool degenerate = (s.mean < floor_steps) || (eye_solid_frac >= kEyeSolidMaxPct);
    if (degenerate && c.measure_degenerate) {
      printf("  [WARN] %-44s %s\n", "DDA workload is DEGENERATE",
             "timings are printed anyway because --measure-degenerate was set");
      printf("  ---- DEGENERATE: %s\n", buf_steps);
      printf("  ---- DEGENERATE: %s\n", buf_eye);
      printf("  ---- DEGENERATE: these timings measure the obs store, not a voxel\n"
             "  ---- renderer. They exist to document the pre-fix baseline.\n");
      g_dda_degenerate = true;
    } else {
      printf("  ---- DDA gate: mean >= %.2f steps/pixel (max(2.0, 0.10*max_steps); a\n"
             "  ---- a fully buried eye scores 1.000) AND first-voxel hits < %.0f%%.\n",
             floor_steps, kEyeSolidMaxPct);
      check(s.mean >= floor_steps, "DDA step count is above the degeneracy floor", buf_steps);
      check(eye_solid_frac < kEyeSolidMaxPct, "the eye is not inside solid rock (fraction)",
            buf_eye);
    }
  }

  printf("== selftest: %s (%d failure%s) ==\n", g_fail ? "FAIL" : "PASS", g_fail,
         g_fail == 1 ? "" : "s");
  return g_fail;
}

// ---------------------------------------------------------------------------
// One measurement pass at a given config.
// ---------------------------------------------------------------------------
struct Floors {
  double write_gbs = 0, read_gbs = 0, copy_gbs = 0, memcpy_gbs = 0;
  double write_us = 0, read_us = 0, copy_us = 0, memcpy_us = 0;
};

static Floors measure_floors(const Cfg& c) {
  hdr("memory floors, measured on this device, in this run");
  size_t n = (size_t)c.copy_bytes;
  void *a = nullptr, *b = nullptr;
  CHK(cudaMalloc(&a, n));
  CHK(cudaMalloc(&b, n));
  CHK(cudaMemset(a, 1, n));
  uint4* sink = devalloc<uint4>(1);
  uint4 v = make_uint4(0x01010101u, 0x01010101u, 0x01010101u, 0x01010101u);
  size_t n4 = n / 16;
  int it = 20;
  Floors f;
  {
    Stat s = stats_of(time_each(3, it, [&] {
      floor_write<<<(int)((n4 + 255) / 256), 256>>>((uint4*)b, n4, v);
    }));
    f.write_us = s.med;
    f.write_gbs = (double)n / (s.med * 1e-6) / 1e9;
  }
  {
    Stat s = stats_of(time_each(3, it, [&] {
      floor_read<<<2048, 256>>>((const uint4*)a, n4, (uint4*)sink);
    }));
    f.read_us = s.med;
    f.read_gbs = (double)n / (s.med * 1e-6) / 1e9;
  }
  {
    Stat s = stats_of(time_each(3, it, [&] {
      floor_copy<<<(int)((n4 + 255) / 256), 256>>>((uint4*)b, n4);
    }));
    f.copy_us = s.med;
    f.copy_gbs = (2.0 * (double)n) / (s.med * 1e-6) / 1e9;
  }
  {
    Stat s = stats_of(time_each(3, it, [&] {
      CHK(cudaMemcpyAsync(b, a, n, cudaMemcpyDeviceToDevice, 0));
    }));
    f.memcpy_us = s.med;
    f.memcpy_gbs = (2.0 * (double)n) / (s.med * 1e-6) / 1e9;
  }
  CHK(cudaFree(a));
  CHK(cudaFree(b));
  CHK(cudaFree(sink));
  printf("  probe    traffic counted     bytes        us      GB/s\n");
  printf("  write    1x (store only)  %9.0f  %8.1f  %8.1f\n", (double)n, f.write_us, f.write_gbs);
  printf("  read     1x (load only)   %9.0f  %8.1f  %8.1f\n", (double)n, f.read_us, f.read_gbs);
  printf("  copy     2x (1:1 RMW)     %9.0f  %8.1f  %8.1f\n", 2.0 * (double)n, f.copy_us,
         f.copy_gbs);
  printf("  memcpy   2x (1:1 D2D)     %9.0f  %8.1f  %8.1f\n", 2.0 * (double)n, f.memcpy_us,
         f.memcpy_gbs);
  printf("  read this carefully: write/read count ONE direction, copy/memcpy count\n"
         "  TWO. Dividing a one-direction figure by a two-direction figure is the\n"
         "  single easiest way to be wrong by exactly 2x here.\n");
  return f;
}

struct StepResult {
  double med_us = 0, min_us = 0, max_us = 0, std_us = 0;
  int kernels = 0;
};

int main(int argc, char** argv) {
  Cfg c;
  auto val = [&](int i) {
    if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", argv[i]); exit(64); }
    return argv[i + 1];
  };

  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "-B" || a == "--batch") { c.B = atoi(val(i)); i++; }
    else if (a == "-w" || a == "--world") { c.W = atoi(val(i)); i++; }
    else if (a == "--res") { c.Wd = c.H = atoi(val(i)); i++; }
    else if (a == "--width") { c.Wd = atoi(val(i)); i++; }
    else if (a == "--height") { c.H = atoi(val(i)); i++; }
    else if (a == "--max-steps") { c.max_steps = atoi(val(i)); i++; }
    else if (a == "--ground") { c.ground_h = atoi(val(i)); i++; }
    else if (a == "--view-dist") { c.view_dist = (float)atof(val(i)); i++; }
    else if (a == "--iters") { c.iters = atoi(val(i)); i++; }
    else if (a == "--graph-iters") { c.graph_iters = atoi(val(i)); i++; }
    else if (a == "--warmup") { c.warmup = atoi(val(i)); i++; }
    else if (a == "--copy-bytes") { c.copy_bytes = atoi(val(i)); i++; }
    else if (a == "--settle-iters") { c.settle_iters = atoi(val(i)); i++; }
    else if (a == "--min-dda-steps") { c.min_dda_steps = atof(val(i)); i++; }
    else if (a == "--no-selftest") { c.selftest = false; }
    else if (a == "--selftest-only") { c.selftest_only = true; }
    else if (a == "--measure-degenerate") { c.measure_degenerate = true; }
    else if (a == "-h" || a == "--help") {
      printf(
          "usage: %s [options]\n"
          "  -B,--batch N      envs per step (default 32768, the project's headline shape)\n"
          "  -w,--world N      cubic world edge in voxels (default 16)\n"
          "     --res N        render resolution NxN (default 32)\n"
          "     --width N      render width (default 32)\n"
          "     --height N     render height (default 32)\n"
          "     --max-steps N  DDA step cap (default 32)\n"
          "     --ground N     ground height in voxels. DEFAULT = world_y/4, which is\n"
          "                    the only value at which the visible surface (top face at\n"
          "                    y=ground_height) coincides with the flat analytic plane\n"
          "                    physics_kernel collides against (y=world_y/4). Anything\n"
          "                    else buries or floats the agent and the DDA gate fails.\n"
          "     --view-dist N  ray cutoff in blocks (default 32)\n"
          "     --iters N      per-kernel timed iterations (default 200)\n"
          "     --graph-iters N iterations for the CUDA-graph step paths (default 200)\n"
          "     --warmup N     warmup iterations (default 25)\n"
          "     --copy-bytes N bytes for the floor probes (default 536870912)\n"
          "     --settle-iters N  cap on the pre-measurement settle (default 64)\n"
          "     --min-dda-steps N  DDA gate floor, mean steps/pixel\n"
          "                    (default max(2.0, 0.10*max_steps)). Lowering this is how\n"
          "                    you would re-admit the degenerate workload; do not.\n"
          "     --no-selftest  skip the correctness gate\n"
          "     --selftest-only  run the correctness gate and exit\n"
          "     --measure-degenerate  print timings for a below-floor configuration.\n"
          "                    Does NOT lower the floor and does NOT skip the check: the\n"
          "                    run is stamped DEGENERATE in its header and in every\n"
          "                    timing table. For reproducing the pre-fix numbers only.\n",
          argv[0]);
      return 0;
    } else {
      fprintf(stderr, "unknown option %s (try --help)\n", a.c_str());
      return 64;
    }
  }

  int devcount = 0;
  // Not CHK: "no device" and "driver too old" are ordinary conditions for a
  // containerised run, not memory corruption, and they need to be reported in
  // words rather than as a raw error string with a source line.
  cudaError_t de = cudaGetDeviceCount(&devcount);
  if (de != cudaSuccess || devcount == 0) {
    fprintf(stderr,
            "NO GPU: cudaGetDeviceCount -> %s (devices=%d)\n"
            "  This binary measures GPU time and has nothing to measure without a\n"
            "  device. It compiled fine; that is a separate fact, checked by\n"
            "  bench_cuda/build.sh, which needs no GPU at all.\n",
            cudaGetErrorString(de), devcount);
    return 4;
  }
  cudaDeviceProp p{};
  CHK(cudaGetDeviceProperties(&p, 0));
  char cc[16];
  snprintf(cc, sizeof cc, "sm_%d%d", p.major, p.minor);

  hdr("bench_cuda -- hand-written CUDA baseline for the mine.cu reference");
  printf("device            %s\n", p.name);
  printf("cc                %s\n", cc);
  printf("SMs               %d\n", p.multiProcessorCount);
  printf("l2 bytes          %zu\n", (size_t)p.l2CacheSize);
  printf("smem per block     %zu\n", (size_t)p.sharedMemPerBlock);
  printf("mem clock/bus      %d MHz / %d-bit  -> datasheet %0.1f GB/s\n", p.memoryClockRate,
         p.memoryBusWidth,
         2.0 * p.memoryClockRate * 1e3 * (p.memoryBusWidth / 8) / 1e9);
  printf("runtime/driver     ");
  {
    int rt = 0, dr = 0;
    CHK(cudaRuntimeGetVersion(&rt));
    // cudaDriverGetVersion is the only one that can fail without a device, and
    // it is allowed to here, so it is not wrapped in CHK.
    cudaDriverGetVersion(&dr);
    printf("CUDA runtime %d, driver %d\n", rt, dr);
  }
  printf("shape             batch=%d world=%d^3 res=%dx%d max_steps=%d ground=%d view=%.1f\n",
         c.B, c.W, c.Wd, c.H, c.max_steps, ground_of(c), (double)c.view_dist);
  printf("iters             per-kernel=%d graph=%d warmup=%d\n", c.iters, c.graph_iters, c.warmup);
  // The ground/physics-floor agreement is a property of the CONFIG and is
  // visible before anything runs, so print it up front rather than leaving a
  // reader to infer it from the DDA gate 40 lines later.
  printf("ground vs physics floor   visible surface top face y=%d, physics_kernel plane "
         "y=world_y/4=%d -> %s\n",
         ground_of(c), c.W / 4,
         ground_of(c) == c.W / 4 ? "AGREE, the eye is above its own ground"
                                 : "DISAGREE, the agent is inside or floating above it");
  if (c.measure_degenerate) {
    printf("\n"
           "  ########################################################################\n"
           "  # --measure-degenerate IS SET.                                            #\n"
           "  # This run will print timings even if the DDA step gate fails. Those       #\n"
           "  # timings are NOT a measurement of a voxel renderer. They exist only to  #\n"
           "  # document the pre-fix baseline. Do not quote them.                       #\n"
           "  ########################################################################\n"
           "\n");
  }

  size_t need = (size_t)c.B * (2 * wvox_of(c) + 2 * hw_of(c) * 3 + 2 * hw_of(c) * 3 * 4 + 512);
  size_t free_b = 0, tot_b = 0;
  CHK(cudaMemGetInfo(&free_b, &tot_b));
  printf("memory            need ~%.0f MiB, free %.0f MiB of %.0f MiB\n", need / 1048576.0,
         free_b / 1048576.0, tot_b / 1048576.0);
  if (free_b < need) {
    fprintf(stderr, "not enough device memory: need ~%.0f MiB, free %.0f MiB\n", need / 1048576.0,
            free_b / 1048576.0);
    return 5;
  }

  Dev d;
  alloc(d, c);
  init_state(d, c);

  if (c.selftest) {
    if (selftest(d, c) != 0) {
      printf("\nselftest FAILED: no timings reported. A timing from a wrong kernel is not a\n"
             "timing. Exiting 2.\n");
      return 2;
    }
  }
  if (c.selftest_only) { printf("\nDONE (selftest only)\n"); return 0; }

  // Settle again, here, unconditionally.
  //
  // The selftest settles, but the selftest also MUTATES state: check 7 digs a
  // log and a leaf out of the tree (it regenerates the world afterwards, so the
  // voxels are fine) and check 8 resets every agent back to the spawn, which
  // UN-SETTLES them. So the state entering the timing section is not the state
  // the DDA gate approved. Settling here makes the timed state and the reported
  // workload the same state, and it also covers --no-selftest, where nothing
  // else settles at all. Settling an already-settled agent is a no-op that
  // costs two graph launches, so there is nothing to optimise away.
  {
    int s2 = settle_state(d, c);
    printf("settled for timing: %d step%s (agent at rest on the physics floor)\n", s2,
           s2 == 1 ? "" : "s");
  }

  // ---------------------------------------------------------------- floors
  Floors f = measure_floors(c);

  size_t B = (size_t)c.B, hw = hw_of(c), wv = wvox_of(c);
  size_t b_obs_f32 = B * hw * 3 * 4;
  size_t b_obs_u8 = B * hw * 3;
  size_t b_vox = B * wv;

  // ------------------------------------------------------ DDA workload
  // Re-measured HERE, immediately before the timing battery, on the same voxels
  // and the same basis the timed render kernels are about to read. The gate in
  // selftest() ran on an earlier state; this repeat is what appears next to the
  // timings, so a reader never has to correlate two sections to know what
  // workload produced them. If the two disagree, the one printed HERE is the
  // one that describes the timings, because it is the later of the two.
  //
  // They can differ slightly, and that is expected rather than alarming. The
  // selftest's check 7 digs a log and a leaf out of the tree and then
  // regenerates the world; the settle moves every agent; and the agent wanders
  // in x/z under the forward button, so the exact set of rays is not frozen.
  // What the gate asserts is a property with a wide margin (26.4 vs a floor of
  // 3.2 at the fixed config, 1.3 vs 3.2 at the degenerate one), not a value
  // reproduced to the third decimal.
  DdaStat pre = measure_dda(d, c);
  double pre_eye_frac = pre.pixels ? 100.0 * (double)pre.eye_solid / (double)pre.pixels : 0.0;
  if (!c.measure_degenerate && (pre.mean < dda_floor(c) || pre_eye_frac >= kEyeSolidMaxPct)) {
    // The state drifted into degeneracy between the gate and the timings, or
    // --no-selftest skipped the gate entirely. Either way this is exactly the
    // silent regression the gate exists to catch, so it aborts here rather than
    // printing a table nobody would distrust.
    printf("\nDDA workload is degenerate at the point of timing: mean %.3f (floor %.2f), "
           "first-voxel hits %.2f%% (limit %.0f%%).\nNo timings reported. Exiting 2.\n",
           pre.mean, dda_floor(c), pre_eye_frac, kEyeSolidMaxPct);
    return 2;
  }
  hdr("DDA workload of the render kernel being timed");
  printf("  %llu pixels = batch %zu x %d x %d, cap %d steps. Counted by replaying the\n",
         pre.pixels, B, c.Wd, c.H, c.max_steps);
  printf("  vendored DDA (kernels.cu:664-679) over the device's own voxels and basis.\n\n");
  printf("  %-28s %12s\n", "DDA tests per pixel", "value");
  printf("  %-28s %12.3f\n", "  mean", pre.mean);
  printf("  %-28s %12.3f\n", "  floor (DDA gate)", dda_floor(c));
  printf("  %-28s %12d\n", "  p50", pre.p50);
  printf("  %-28s %12d\n", "  p90", pre.p90);
  printf("  %-28s %12d\n", "  max", pre.max);
  printf("  %-28s %12llu  (%.1f%%)%s\n", "  pixels at the step cap", pre.at_cap,
         pre.pixels ? 100.0 * (double)pre.at_cap / (double)pre.pixels : 0.0,
         pre.cap_clamped ? "  (>=64; histogram saturates)" : "");
  printf("  %-28s %12llu  (%.2f%%)  <- the buried-eye tell; gate is < %.0f%%\n",
         "  first-voxel hits", pre.eye_solid,
         pre.pixels ? 100.0 * (double)pre.eye_solid / (double)pre.pixels : 0.0,
         kEyeSolidMaxPct);
  printf("\n");
  if (g_dda_degenerate) {
    printf("  *** DEGENERATE WORKLOAD: mean %.3f is below the floor %.2f. A ray here\n"
           "  *** tests one voxel and stops, so these timings measure the obs store and\n"
           "  *** nothing else. They are printed because --measure-degenerate was set.\n\n",
           pre.mean, dda_floor(c));
  } else {
    printf("  Non-degenerate: the mean is %.1fx the floor, so the timed kernels are\n"
           "  marching real rays rather than exiting on their first voxel test.\n\n",
           pre.mean / dda_floor(c));
  }

  // ----------------------------------------------------------- per kernel
  hdr("per-kernel, each launched on its own through its own extern \"C\" launcher");
  if (g_dda_degenerate) printf("  *** DEGENERATE: see the DDA workload section above ***\n");
  printf("  %d timed iterations after %d warmup, cudaEvent pair per launch, median reported\n",
         c.iters, c.warmup);
  printf("  %-36s %10s %10s %10s %10s %9s %9s\n", "kernel", "med us", "min us", "max us",
         "std us", "GB/s", "%wr");
  struct PK { std::string name; double us; double gbs; double pct; };
  std::vector<PK> pks;
  char pctbuf[16];

  auto bench = [&](const char* name, double bytes, auto&& fn) {
    Stat s = stats_of(time_each(c.warmup, c.iters, fn));
    double gbs = bytes > 0 ? bytes / (s.med * 1e-6) / 1e9 : 0.0;
    double pct = (bytes > 0 && f.write_gbs > 0) ? 100.0 * gbs / f.write_gbs : 0.0;
    if (bytes > 0) snprintf(pctbuf, sizeof pctbuf, "%.0f%%", pct);
    else snprintf(pctbuf, sizeof pctbuf, "-");
    printf("  %-36s %10.1f %10.1f %10.1f %10.1f %9.1f %9s\n", name, s.med, s.min, s.max, s.std,
           gbs, pctbuf);
    pks.push_back({name, s.med, gbs, pct});
  };

  bench("decode_actions_kernel", 0, [&] {
    launch_decode_actions(d.buttons, d.look, d.fwd, d.strafe, d.dyaw, d.dpitch, d.jump,
                          d.do_break, d.do_place, d.speed_mult, c.B, c.sprint_mult);
  });
  bench("physics_kernel", 0, [&] {
    launch_physics_v2(d.pos, d.vel, d.yaw, d.pitch, d.on_ground, d.fwd, d.strafe, d.dyaw,
                      d.dpitch, d.jump, d.speed_mult, c.B, c.W, c.dt, c.gravity, c.walk_speed,
                      c.jump_vel);
  });
  bench("raycast_interact_kernel", 0, [&] {
    launch_raycast_interact(d.voxels, d.pos, d.yaw, d.pitch, d.do_break, d.do_place, d.rewards,
                            c.B, c.W, c.target_block, c.reward_value);
  });
  bench("reset_kernel", 0, [&] {
    launch_reset(d.pos, d.vel, d.yaw, d.pitch, d.on_ground, d.do_reset, c.B, spawn_x_of(c),
                 spawn_y_of(c), spawn_z_of(c));
  });
  bench("precompute_camera_basis_kernel", 0, [&] {
    launch_precompute_camera_basis(d.cameras, d.basis, c.B, c.Wd, c.H, c.fov);
  });
  bench("render_kernel (fp32 obs)", b_vox + b_obs_f32, [&] {
    launch_render(d.voxels, d.cameras, d.obs_f32, c.B, c.W, c.Wd, c.H, c.max_steps, c.view_dist,
                  c.fov);
  });
  bench("render_kernel_uint8_minimal", b_vox + b_obs_u8, [&] {
    launch_render_uint8_minimal(d.voxels, d.cameras, d.obs_u8, c.B, c.W, c.Wd, c.H, c.max_steps,
                                c.view_dist, c.fov);
  });
  bench("render_kernel_uint8_fp16", b_vox + b_obs_u8, [&] {
    launch_render_uint8_fp16(d.voxels, d.cameras, d.obs_u8b, c.B, c.W, c.Wd, c.H, c.max_steps,
                             c.view_dist, c.fov);
  });
  bench("render_kernel_uint8_prebasis", b_vox + b_obs_u8, [&] {
    launch_render_uint8_prebasis(d.voxels, d.basis, d.obs_u8, c.B, c.W, c.Wd, c.H, c.max_steps);
  });

  double sum_pk = 0;
  for (const auto& p : pks) sum_pk += p.us;
  printf("  %-36s %10.1f  <- sum of the five state kernels, not the step\n", "SUM (informative)",
         sum_pk);

  // ------------------------------------------------------------- whole step
  hdr("whole step, CUDA graph, as the project's headline counts it");
  if (g_dda_degenerate) printf("  *** DEGENERATE: see the DDA workload section above ***\n");
  printf("  each row replays one captured graph. `kern` is the number of kernels the\n"
         "  graph holds. env-steps/sec = batch / median_us.\n");
  printf("  %-36s %4s %10s %10s %10s %9s %9s\n", "step launcher", "kern", "med us", "min us",
         "max us", "std us", "Mstep/s");

  auto run = [&](const char* name, int kern, auto&& fn) {
    Stat s = stats_of(time_each(12, c.graph_iters, fn));
    StepResult r{s.med, s.min, s.max, s.std, kern};
    printf("  %-36s %4d %10.1f %10.1f %10.1f %9.1f %9.2f\n", name, kern, s.med, s.min, s.max,
           s.std, (double)c.B / (s.med * 1e-6) / 1e6);
    return r;
  };

  StepResult rs_f32 = run("launch_step (fp32 obs, 6 kern)", 6, [&] {
    launch_step(d.voxels, d.pos, d.vel, d.yaw, d.pitch, d.on_ground, d.buttons, d.look,
                d.do_reset, d.fwd, d.strafe, d.dyaw, d.dpitch, d.jump, d.do_break, d.do_place,
                d.speed_mult, d.cameras, d.obs_f32, d.rewards, c.B, c.W, c.Wd, c.H, c.max_steps,
                c.view_dist, c.fov, c.dt, c.gravity, c.walk_speed, c.sprint_mult, c.jump_vel,
                c.target_block, c.reward_value, spawn_x_of(c), spawn_y_of(c), spawn_z_of(c));
  });
  StepResult rs_pre = run("launch_step_uint8_prebasis (7)", 7, [&] {
    launch_step_uint8_prebasis(
        d.voxels, d.pos, d.vel, d.yaw, d.pitch, d.on_ground, d.buttons, d.look, d.do_reset, d.fwd,
        d.strafe, d.dyaw, d.dpitch, d.jump, d.do_break, d.do_place, d.speed_mult, d.cameras,
        d.basis, d.obs_u8, d.rewards, c.B, c.W, c.W, c.W, c.Wd, c.H, c.max_steps, c.view_dist,
        c.fov, c.dt, c.gravity, c.walk_speed, c.sprint_mult, c.jump_vel, c.target_block,
        c.reward_value, spawn_x_of(c), spawn_y_of(c), spawn_z_of(c));
  });
  StepResult rs_full = run("launch_step_..._prebasis_full (9)", 9, [&] {
    launch_step_uint8_prebasis_full(
        d.voxels, d.pos, d.vel, d.yaw, d.pitch, d.on_ground, d.buttons, d.look, d.do_reset,
        d.step_count, d.fwd, d.strafe, d.dyaw, d.dpitch, d.jump, d.do_break, d.do_place,
        d.speed_mult, d.cameras, d.basis, d.obs_u8, d.rewards, c.B, c.W, c.W, c.W, c.Wd, c.H,
        c.max_steps, c.view_dist, c.fov, c.dt, c.gravity, c.walk_speed, c.sprint_mult,
        c.jump_vel, c.target_block, c.reward_value, spawn_x_of(c), spawn_y_of(c), spawn_z_of(c),
        c.episode_len);
  });

  // ------------------------------------------------------------- roofline
  hdr("roofline position against the floors measured above");
  if (g_dda_degenerate) printf("  *** DEGENERATE: see the DDA workload section above ***\n");
  printf("  write-only floor            %8.1f GB/s   (measured)\n", f.write_gbs);
  printf("  read-modify-write floor     %8.1f GB/s   (measured, 1:1)\n", f.copy_gbs);
  printf("  D2D memcpy floor            %8.1f GB/s   (measured, 1:1)\n", f.memcpy_gbs);
  printf("\n");
  printf("  per step, float32 obs path: write %.1f MiB + at least %.1f MiB of voxel read\n",
         b_obs_f32 / 1048576.0, b_vox / 1048576.0);
  printf("  per step, uint8    obs path: write %.1f MiB + at least %.1f MiB of voxel read\n",
         b_obs_u8 / 1048576.0, b_vox / 1048576.0);
  printf("\n");
  double t_write_only = b_obs_f32 / (f.write_gbs * 1e9) * 1e6;
  double t_write_only_u8 = b_obs_u8 / (f.write_gbs * 1e9) * 1e6;
  printf("  obs-write-only time, fp32   %8.1f us  -> %7.2f M env-steps/s if the step did "
         "nothing but store obs\n", t_write_only, (double)c.B / (t_write_only * 1e-6) / 1e6);
  printf("  obs-write-only time, uint8   %8.1f us  -> %7.2f M env-steps/s if the step did "
         "nothing but store obs\n", t_write_only_u8,
         (double)c.B / (t_write_only_u8 * 1e-6) / 1e6);
  printf("\n");
  printf("  measured whole-step, fp32    %8.1f us   -> %8.2f M env-steps/s\n", rs_f32.med_us,
         (double)c.B / (rs_f32.med_us * 1e-6) / 1e6);
  printf("  measured whole-step, u8 pre  %8.1f us   -> %8.2f M env-steps/s\n", rs_pre.med_us,
         (double)c.B / (rs_pre.med_us * 1e-6) / 1e6);
  printf("  measured whole-step, u8 full %8.1f us   -> %8.2f M env-steps/s\n", rs_full.med_us,
         (double)c.B / (rs_full.med_us * 1e-6) / 1e6);
  printf("\n");
  {
    double t = rs_f32.med_us;
    double tu = rs_pre.med_us;
    double wr_f32 = (double)b_obs_f32 / (t * 1e-6) / 1e9;
    double wr_u8 = (double)b_obs_u8 / (tu * 1e-6) / 1e9;
    printf("  fp32 step, obs write alone   : %8.1f GB/s = %5.1f%% of the write floor\n", wr_f32,
           100.0 * wr_f32 / f.write_gbs);
    printf("  fp32 step, obs+voxel counted : %8.1f GB/s = %5.1f%% of the 1:1 floor\n",
           (double)(b_obs_f32 + b_vox) / (t * 1e-6) / 1e9,
           100.0 * (double)(b_obs_f32 + b_vox) / (t * 1e-6) / 1e9 / f.copy_gbs);
    printf("  u8   step, obs write alone   : %8.1f GB/s = %5.1f%% of the write floor\n", wr_u8,
           100.0 * wr_u8 / f.write_gbs);
    printf("  u8   step, obs+voxel counted : %8.1f GB/s = %5.1f%% of the 1:1 floor\n",
           (double)(b_obs_u8 + b_vox) / (tu * 1e-6) / 1e9,
           100.0 * (double)(b_obs_u8 + b_vox) / (tu * 1e-6) / 1e9 / f.copy_gbs);
    printf("\n");
    printf("  The obs-write fraction is the number to compare against a claim like\n"
           "  \"close to the write-bandwidth floor\". It is near 100%% only if the\n"
           "  render kernel is bandwidth bound; if it is far below 100%% while the\n"
           "  obs+voxel fraction is also far below, the kernel is bound by neither\n"
           "  and is DDA-latency or issue bound, which is a different claim entirely.\n");
  }

  printf("\nDONE\n");
  return 0;
}
