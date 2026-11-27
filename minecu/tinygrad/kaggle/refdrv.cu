// refdrv.cu -- the hand-written side of the comparison, driven from one place.
//
// WHAT THIS IS FOR. notes/17-minecu-spec.md says the defensible comparison is
// per-kernel against per-kernel. This binary exists so that comparison can
// actually be made, and it does two jobs that `bench.cu` cannot do together:
//
//   1. It DUMPS THE STATE. After the settle, it writes the voxel volume, the
//      camera basis and the uint8 observation to plain binary files, so the
//      TinyGrad side can be given byte-identical inputs and its output
//      compared to the reference's byte for byte on the same card in the same
//      run. Without this the two sides would be rendering different worlds
//      and the comparison would be unattributable -- the failure mode
//      notes/17 section 8.1 lists under "world content".
//
//   2. It TIMES the reference render kernel and the whole step at a list of
//      batch sizes, with cudaEvent, and prints them. `bench.cu` reports the
//      batch-32768 whole step; this reports a per-batch table so the TinyGrad
//      side can be compared at every batch where it can still be allocated,
//      rather than only at the one batch where it cannot.
//
// WHY IT INCLUDES kernels.cu RATHER THAN LINKING IT. One translation unit, so
// there is no duplicate-symbol question and no ABI question. kernels.cu itself
// is byte-identical to upstream (sha256 c6ed8f06..., re-checked by the
// notebook before anything is compiled); including it does not modify it.
//
// THE DUMP FORMAT. Raw little-endian, no header magic beyond a fixed layout the
// notebook knows: <name>.bin files, each exactly the bytes of one tensor, plus
// a <name>.meta text file with the shape. 134 MB of voxels at batch 32768 is
// written to the runner's own disk and never downloaded; the notebook reads it
// in place and only the small artefacts come home.

// Path to the vendored reference, overridable at compile time with
// -DMINECU_KERNELS=... because `#include` resolves relative to THIS file, and
// the notebook writes kernels.cu to /kaggle/working rather than next to this.
#ifndef MINECU_KERNELS
#define MINECU_KERNELS "kernels.cu"
#endif
#include MINECU_KERNELS

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHK(x)                                                                \
  do {                                                                        \
    cudaError_t e = (x);                                                      \
    if (e != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e),      \
              __FILE__, __LINE__);                                            \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

static int ground_of(int world, int ground) { return ground >= 0 ? ground : world / 4; }

template <typename T> static T* devalloc(size_t n) {
  T* p = nullptr;
  CHK(cudaMalloc(&p, n * sizeof(T)));
  return p;
}

struct Cfg {
  int B = 32768;
  int W = 16;
  int H = 32, Wd = 32;
  int max_steps = 32;
  float view_dist = 32.0f;
  float fov = 70.0f;
  float dt = 0.05f, gravity = -20.0f, walk_speed = 4.0f, sprint_mult = 1.5f;
  float jump_vel = 8.0f;
  int ground = -1;
  int8_t target_block = 3;
  float reward_value = 1.0f;
  int episode_len = 0;
  int iters = 50;
  int warmup = 10;
  int settle_iters = 64;
  std::string outdir = "/kaggle/working/refstate";
};

struct Dev {
  int8_t* voxels; float *pos, *vel, *yaw, *pitch; bool* on_ground;
  int8_t* buttons; float* look; bool* do_reset; int* step_count;
  float *fwd, *strafe, *dyaw, *dpitch; bool *jump, *do_break, *do_place;
  float* speed_mult; float *cameras, *basis; uint8_t* obs_u8; float* rewards;
};

static void init_state(Dev& d, const Cfg& c) {
  size_t B = (size_t)c.B, wv = (size_t)c.W * c.W * c.W, hw = (size_t)c.H * c.Wd;
  d.voxels    = devalloc<int8_t>(B * wv);
  d.pos       = devalloc<float>(B * 3);
  d.vel       = devalloc<float>(B * 3);
  d.yaw       = devalloc<float>(B);
  d.pitch     = devalloc<float>(B);
  d.on_ground = devalloc<bool>(B);
  d.buttons   = devalloc<int8_t>(B * 8);
  d.look      = devalloc<float>(B * 2);
  d.do_reset  = devalloc<bool>(B);
  d.step_count= devalloc<int>(B);
  d.fwd = devalloc<float>(B); d.strafe = devalloc<float>(B);
  d.dyaw = devalloc<float>(B); d.dpitch = devalloc<float>(B);
  d.jump = devalloc<bool>(B); d.do_break = devalloc<bool>(B); d.do_place = devalloc<bool>(B);
  d.speed_mult = devalloc<float>(B);
  d.cameras = devalloc<float>(B * 5);
  d.basis   = devalloc<float>(B * 14);
  d.obs_u8  = devalloc<uint8_t>(B * hw * 3);
  d.rewards = devalloc<float>(B);

  CHK(cudaMemset(d.pos, 0, B * 3 * sizeof(float)));
  CHK(cudaMemset(d.vel, 0, B * 3 * sizeof(float)));
  CHK(cudaMemset(d.yaw, 0, B * sizeof(float)));
  CHK(cudaMemset(d.pitch, 0, B * sizeof(float)));
  CHK(cudaMemset(d.on_ground, 1, B));
  CHK(cudaMemset(d.do_reset, 0, B));
  CHK(cudaMemset(d.step_count, 0, B * sizeof(int)));
  CHK(cudaMemset(d.jump, 0, B)); CHK(cudaMemset(d.do_break, 0, B));
  CHK(cudaMemset(d.do_place, 0, B));
  CHK(cudaMemset(d.speed_mult, 0x3f, B * sizeof(float)));
  CHK(cudaMemset(d.obs_u8, 0x7f, B * hw * 3));

  // The same fixed, non-degenerate action stream as bench_cuda/bench.cu's
  // init_state, byte for byte, so the settled state this dumps is the state the
  // reference's own benchmark measures. Forward held, a per-env yaw drift, and
  // break on 1 env in 32 so the raycast has something to hit.
  std::vector<int8_t> hb(B * 8, 0);
  std::vector<float> hl(B * 2, 0.f);
  for (size_t i = 0; i < B; i++) {
    hb[i * 8 + 0] = 1;
    hl[i * 2 + 0] = 0.002f * (float)((i % 97) - 48);
    hb[i * 8 + 5] = (int8_t)((i % 32) == 0 ? 1 : 0);
  }
  CHK(cudaMemcpy(d.buttons, hb.data(), B * 8, cudaMemcpyHostToDevice));
  CHK(cudaMemcpy(d.look, hl.data(), B * 2 * sizeof(float), cudaMemcpyHostToDevice));

  launch_generate_world(d.voxels, c.B, c.W, c.W, c.W, ground_of(c.W, c.ground), 42u);
  CHK(cudaDeviceSynchronize()); CHK(cudaGetLastError());
  launch_place_tree(d.voxels, c.B, c.W, c.W, c.W, (int)(c.W / 2.0f), (int)(c.W / 2.0f) + 2,
                    ground_of(c.W, c.ground));
  launch_place_house(d.voxels, c.B, c.W, c.W, c.W, (int)(c.W / 2.0f) + 6,
                     (int)(c.W / 2.0f) - 2, ground_of(c.W, c.ground));
  CHK(cudaDeviceSynchronize()); CHK(cudaGetLastError());

  std::vector<float> hp(B * 3);
  for (size_t i = 0; i < B; i++) {
    hp[i * 3 + 0] = c.W / 2.0f;
    hp[i * 3 + 1] = (float)(ground_of(c.W, c.ground) + 2);
    hp[i * 3 + 2] = c.W / 2.0f;
  }
  CHK(cudaMemcpy(d.pos, hp.data(), B * 3 * sizeof(float), cudaMemcpyHostToDevice));
  CHK(cudaMemset(d.on_ground, 1, B));
  std::vector<float> hc(B * 5);
  for (size_t i = 0; i < B; i++) {
    hc[i * 5 + 0] = c.W / 2.0f;
    hc[i * 5 + 1] = (float)(ground_of(c.W, c.ground) + 2);
    hc[i * 5 + 2] = c.W / 2.0f;
    hc[i * 5 + 3] = 0.0f;
    hc[i * 5 + 4] = -0.2f;
  }
  CHK(cudaMemcpy(d.cameras, hc.data(), B * 5 * sizeof(float), cudaMemcpyHostToDevice));
}

static float spawn_x_of(const Cfg& c) { return c.W / 2.0f; }
static float spawn_z_of(const Cfg& c) { return c.W / 2.0f; }
static float spawn_y_of(const Cfg& c) { return (float)(ground_of(c.W, c.ground) + 2); }

// Drive the full graph until every agent has landed on the physics floor and
// stayed there, then rebuild the camera from the RESTING state. This is
// verbatim in intent from bench_cuda/bench.cu's settle_state, and the
// reason for it is recorded there: without it the DDA step count is reported
// for the spawn eye while the timing is taken from the resting eye, and at
// world 16 those two are on opposite sides of the ground.
static void settle(Dev& d, const Cfg& c) {
  size_t B = (size_t)c.B;
  std::vector<float> prev(B), cur(B);
  CHK(cudaMemcpy(prev.data(), d.pos + 1, B * sizeof(float), cudaMemcpyDeviceToHost));
  for (int i = 0; i < c.settle_iters; i++) {
    launch_step_uint8_prebasis_full(
        d.voxels, d.pos, d.vel, d.yaw, d.pitch, d.on_ground, d.buttons, d.look, d.do_reset,
        d.step_count, d.fwd, d.strafe, d.dyaw, d.dpitch, d.jump, d.do_break, d.do_place,
        d.speed_mult, d.cameras, d.basis, d.obs_u8, d.rewards, c.B, c.W, c.W, c.W, c.Wd, c.H,
        c.max_steps, c.view_dist, c.fov, c.dt, c.gravity, c.walk_speed, c.sprint_mult,
        c.jump_vel, c.target_block, c.reward_value, spawn_x_of(c), spawn_y_of(c),
        spawn_z_of(c), c.episode_len);
    CHK(cudaDeviceSynchronize()); CHK(cudaGetLastError());
    CHK(cudaMemcpy(cur.data(), d.pos + 1, B * sizeof(float), cudaMemcpyDeviceToHost));
    bool settled = true;
    for (size_t e = 0; e < B; e++) if (cur[e] != prev[e]) { settled = false; break; }
    if (settled && i > 0) break;
    prev = cur;
  }
  // update_cameras_kernel has no extern "C" launcher, so reproduce its write.
  std::vector<float> hp(B * 3), hy(B), hi(B), hc(B * 5);
  CHK(cudaMemcpy(hp.data(), d.pos, B * 3 * sizeof(float), cudaMemcpyDeviceToHost));
  CHK(cudaMemcpy(hy.data(), d.yaw, B * sizeof(float), cudaMemcpyDeviceToHost));
  CHK(cudaMemcpy(hi.data(), d.pitch, B * sizeof(float), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < B; i++) {
    hc[i * 5 + 0] = hp[i * 3 + 0]; hc[i * 5 + 1] = hp[i * 3 + 1]; hc[i * 5 + 2] = hp[i * 3 + 2];
    hc[i * 5 + 3] = hy[i]; hc[i * 5 + 4] = hi[i];
  }
  CHK(cudaMemcpy(d.cameras, hc.data(), B * 5 * sizeof(float), cudaMemcpyHostToDevice));
  launch_precompute_camera_basis(d.cameras, d.basis, c.B, c.Wd, c.H, c.fov);
  CHK(cudaDeviceSynchronize()); CHK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// dump
// ---------------------------------------------------------------------------
template <typename T>
static void dump(const Dev& d, const Cfg& c, const char* name, const T* dev, size_t n) {
  std::string base = c.outdir + "/" + name;
  std::vector<T> host(n);
  CHK(cudaMemcpy(host.data(), dev, n * sizeof(T), cudaMemcpyDeviceToHost));
  FILE* f = fopen((base + ".bin").c_str(), "wb");
  if (!f) { fprintf(stderr, "cannot write %s.bin\n", base.c_str()); exit(1); }
  fwrite(host.data(), sizeof(T), n, f);
  fclose(f);
  char m[256];
  snprintf(m, sizeof m, "%s count %zu item %zu\n", name, n, sizeof(T));
  FILE* mf = fopen((base + ".meta").c_str(), "w");
  fputs(m, mf); fclose(mf);
  printf("  dumped %-10s %12zu x %2zu B = %9.1f MiB\n", name, n, sizeof(T),
         (double)n * sizeof(T) / 1048576.0);
}

// The basis is [B, 14] with slot 13 never written by the kernel
// (kernels.cu:559 leaves it unwritten), so only 13 slots per env are dumped.
// Dumping the 14th would dump uninitialised memory and invite a comparison
// against garbage.
static void dump_basis13(const Dev& d, const Cfg& c) {
  size_t B = (size_t)c.B;
  std::vector<float> all(B * 14);
  CHK(cudaMemcpy(all.data(), d.basis, B * 14 * sizeof(float), cudaMemcpyDeviceToHost));
  std::vector<float> tight(B * 13);
  for (size_t b = 0; b < B; b++)
    for (int k = 0; k < 13; k++) tight[b * 13 + k] = all[b * 14 + k];
  std::string base = c.outdir + "/basis13";
  FILE* f = fopen((base + ".bin").c_str(), "wb");
  if (!f) { fprintf(stderr, "cannot write basis13.bin\n"); exit(1); }
  fwrite(tight.data(), sizeof(float), B * 13, f);
  fclose(f);
  FILE* mf = fopen((base + ".meta").c_str(), "w");
  fprintf(mf, "basis13 count %zu item 4\n", B * 13); fclose(mf);
  printf("  dumped %-10s %12zu x %2zu B = %9.1f MiB\n", "basis13", B * 13, (size_t)4,
         (double)B * 13 * 4 / 1048576.0);
}

// ---------------------------------------------------------------------------
// timing
// ---------------------------------------------------------------------------
struct Timing { double med, min, max; int iters; };

static Timing time_render(const Dev& d, const Cfg& c) {
  cudaEvent_t a, b;
  CHK(cudaEventCreate(&a)); CHK(cudaEventCreate(&b));
  for (int i = 0; i < c.warmup; i++)
    launch_render_uint8_prebasis(d.voxels, d.basis, d.obs_u8, c.B, c.W, c.Wd, c.H, c.max_steps);
  CHK(cudaDeviceSynchronize());
  std::vector<double> ts;
  for (int i = 0; i < c.iters; i++) {
    CHK(cudaEventRecord(a));
    launch_render_uint8_prebasis(d.voxels, d.basis, d.obs_u8, c.B, c.W, c.Wd, c.H, c.max_steps);
    CHK(cudaEventRecord(b));
    CHK(cudaEventSynchronize(b));
    float ms = 0; CHK(cudaEventElapsedTime(&ms, a, b));
    ts.push_back(ms);
  }
  std::sort(ts.begin(), ts.end());
  Timing t{ts[ts.size() / 2], ts.front(), ts.back(), (int)ts.size()};
  CHK(cudaEventDestroy(a)); CHK(cudaEventDestroy(b));
  return t;
}

static Timing time_step(const Dev& d, const Cfg& c) {
  cudaEvent_t a, b;
  CHK(cudaEventCreate(&a)); CHK(cudaEventCreate(&b));
  auto one = [&]() {
    launch_step_uint8_prebasis(
        d.voxels, d.pos, d.vel, d.yaw, d.pitch, d.on_ground, d.buttons, d.look, d.do_reset,
        d.fwd, d.strafe, d.dyaw, d.dpitch, d.jump, d.do_break, d.do_place, d.speed_mult,
        d.cameras, d.basis, d.obs_u8, d.rewards, c.B, c.W, c.W, c.W, c.Wd, c.H, c.max_steps,
        c.view_dist, c.fov, c.dt, c.gravity, c.walk_speed, c.sprint_mult, c.jump_vel,
        c.target_block, c.reward_value, spawn_x_of(c), spawn_y_of(c), spawn_z_of(c));
  };
  for (int i = 0; i < c.warmup; i++) one();
  CHK(cudaDeviceSynchronize());
  std::vector<double> ts;
  for (int i = 0; i < c.iters; i++) {
    CHK(cudaEventRecord(a)); one(); CHK(cudaEventRecord(b));
    CHK(cudaEventSynchronize(b));
    float ms = 0; CHK(cudaEventElapsedTime(&ms, a, b));
    ts.push_back(ms);
  }
  std::sort(ts.begin(), ts.end());
  Timing t{ts[ts.size() / 2], ts.front(), ts.back(), (int)ts.size()};
  CHK(cudaEventDestroy(a)); CHK(cudaEventDestroy(b));
  return t;
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  Cfg c;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto val = [&]() { return argv[++i]; };
    if (a == "--batch") c.B = atoi(val());
    else if (a == "--world") c.W = atoi(val());
    else if (a == "--res") { c.H = c.Wd = atoi(val()); }
    else if (a == "--max-steps") c.max_steps = atoi(val());
    else if (a == "--ground") c.ground = atoi(val());
    else if (a == "--iters") c.iters = atoi(val());
    else if (a == "--warmup") c.warmup = atoi(val());
    else if (a == "--outdir") c.outdir = val();
    else if (a == "--dump") { /* handled below */ }
    else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 64; }
  }
  bool dump_it = false;
  for (int i = 1; i < argc; i++) if (std::string(argv[i]) == "--dump") dump_it = true;

  cudaDeviceProp prop;
  CHK(cudaGetDeviceProperties(&prop, 0));
  printf("== refdrv: hand-written side, driven ==\n");
  printf("  card                 %s\n", prop.name);
  printf("  batch %d  world %d  res %dx%d  max_steps %d  ground %d\n",
         c.B, c.W, c.Wd, c.H, c.max_steps, ground_of(c.W, c.ground));

  Dev d;
  init_state(d, c);
  settle(d, c);

  // one render, so d.obs_u8 holds the reference's own output
  launch_render_uint8_prebasis(d.voxels, d.basis, d.obs_u8, c.B, c.W, c.Wd, c.H, c.max_steps);
  CHK(cudaDeviceSynchronize()); CHK(cudaGetLastError());

  if (dump_it) {
    printf("  --- dump ---\n");
    dump<int8_t>(d, c, "voxels", d.voxels, (size_t)c.B * c.W * c.W * c.W);
    dump_basis13(d, c);
    dump<uint8_t>(d, c, "obs_u8", d.obs_u8, (size_t)c.B * c.H * c.Wd * 3);
  }

  Timing tr = time_render(d, c);
  printf("  --- timing (cudaEvent, %d iters, %d warmup) ---\n", c.iters, c.warmup);
  // cudaEventElapsedTime returns MILLISECONDS. The first version of this line
  // printed that number under a `_US` label, which is a 1000x mislabel: it
  // reported the reference's batch-32768 render as 6.8 us when the baseline in
  // bench_cuda/buried_agent_fix.md is 7034.4 us. Converted here, once, so the
  // label and the unit agree, and the throughput lines below are unaffected
  // (they already divided milliseconds by 1000).
  printf("  RENDER_US %.1f %.1f %.1f %d\n", tr.med * 1000.0, tr.min * 1000.0,
         tr.max * 1000.0, tr.iters);
  Timing ts = time_step(d, c);
  printf("  STEP_US   %.1f %.1f %.1f %d\n", ts.med * 1000.0, ts.min * 1000.0,
         ts.max * 1000.0, ts.iters);
  printf("  ENVSTEPS_M %.4f\n", (double)c.B / (tr.med / 1000.0) / 1e6);
  printf("  STEP_ENVSTEPS_M %.4f\n", (double)c.B / (ts.med / 1000.0) / 1e6);
  printf("  (RENDER_US and STEP_US are MICROSECONDS; cudaEvent reports ms and this\n"
         "   line converts once. The baseline for comparison is 7034.4 us render /\n"
         "   7198.9 us whole step at batch 32768.)\n");
  return 0;
}
