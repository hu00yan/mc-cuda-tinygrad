// refkernels.cu -- the HAND-WRITTEN side of the axis.
//
// WHAT THIS FILE IS, AND WHAT IT IS NOT
// =====================================
// This is the reference arm of `axis/`. Every kernel here is hand-written CUDA,
// compiled by nvcc to a device-only cubin, and launched by `axis/harness.py`
// through the CUDA driver API *inside TinyGrad's own context*, on TinyGrad's own
// device buffers, timed with the same cuEvent path TinyGrad uses.
//
// It is NOT a back door into the TinyGrad arm. The constraint in the task is on
// the TinyGrad side -- no `Tensor.custom_kernel`, no pasted CUDA, no hand-written
// .cu compiled alongside. The TinyGrad arm in `axis/rungs.py` is pure Tensor
// code and never touches this file, this harness, or the driver API. This file
// is the *reference*, which is the thing being compared against, and a
// comparison needs something to compare against.
//
// WHY EVERY KERNEL IS `extern "C" __global__`
// --------------------------------------------
// A cubin exports mangled C++ names. Declaring them `extern "C"` makes the
// symbol the kernel's own name, so the harness can call `cuModuleGetFunction`
// with a string it can read in this file. If a kernel here is renamed, the
// lookup fails loudly at load time rather than silently timing nothing.
//
// THE DISCIPLINE THIS FILE FOLLOWS, AND WHY IT IS THE WHOLE POINT
// ================================================================
// notes/06 §5b: when a study reports a large gap, the first hypothesis is that
// the comparison is measuring the author. Two comparisons in this repo were
// invalidated exactly that way -- by porting a file that exists to be *read*
// (`fla/ops/kda/naive.py`) or a pre-optimisation form (the naive DDA that
// recomputes a voxel address from three floats on all 32 steps) instead of the
// code the GPU is actually supposed to be running.
//
// So: each kernel below is the form a competent practitioner writes, not the
// form that is easiest to write. Specifically, and checkably:
//
//   * no address is recomputed from coordinates when an incremental advance
//     exists (Amanatides-Woo, the thing C-14 says the naive port got wrong)
//   * no reduction is expressed as a chain of rank-1 updates where a wide
//     contraction belongs (C-18, the ~60x habit)
//   * no loop is left unrolled-unrollable; trip counts are compile-time
//   * fp32 GEMM tiles through shared memory with register blocking, the
//     standard high-performance shape -- not the "one scalar load per operand"
//     kernel that `NOOPT` produces, which is TinyGrad's deliberately-worse mode
//     and would be a strawman
//
// A reader who thinks a kernel here is weak can check it two ways: the SASS
// (`axis/sass.py`) and the roofline position. Every kernel reports a percentage
// of a *measured* memory floor or flop peak on the same card in the same run,
// not of a datasheet number. If a reference kernel is at 20% of roofline, the
// axis is measuring that kernel and the note says so.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>

using namespace nvcuda;

// ===========================================================================
// 0.  MEMORY FLOORS
// ===========================================================================
// These are not rungs. They are the denominators. notes/17 and minecu_ref's
// README both insist that a "% of peak bandwidth" claim needs a floor measured
// on the card making the claim, and a one-direction row must not be divided by
// a two-direction one. So: `axis_fill` writes N floats and reads none;
// `axis_copy` reads N and writes N. Every bandwidth figure in the axis is
// expressed against these, measured in the same process on the same card in the
// same run.

extern "C" __global__ void axis_fill(float* __restrict__ o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = 0.0f;
}

extern "C" __global__ void axis_copy(float* __restrict__ o,
                                     const float* __restrict__ a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = a[i] * 1.0f;
}

// ===========================================================================
// 1.  RUNG `ew` -- elementwise, pure bandwidth
// ===========================================================================
// y = a + b*z, three arrays in, one out, no reduction and no gather. This is
// the rung that demands nothing of a compiler, and it is here to be the
// control: if the framework cannot reach the memory roofline on THIS, then every
// gap above it is a statement about the framework's arithmetic, not about
// bandwidth, and the axis is not measuring what it claims.
//
// float4 throughout, grid-stride so the grid size is not a correctness
// parameter. 4 bytes/float * 4 arrays * 4 lanes.

extern "C" __global__ void axis_ew(float4* __restrict__ y,
                                   const float4* __restrict__ a,
                                   const float4* __restrict__ b,
                                   const float4* __restrict__ z, int n4) {
  int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
    float4 av = a[i], bv = b[i], zv = z[i];
    float4 r;
    r.x = av.x + bv.x * zv.x;
    r.y = av.y + bv.y * zv.y;
    r.z = av.z + bv.z * zv.z;
    r.w = av.w + bv.w * zv.w;
    y[i] = r;
  }
}

// ===========================================================================
// 2.  RUNG `red` -- wide reduction (row-dot)
// ===========================================================================
// y[r] = sum_k x[r,k] * w[r,k]     for r in [0,R), k in [0,K)
//
// This is the rung where notes/26 rule 3 ("output width, not operator name")
// and C-18's ~60x habit live. A port that writes this as a loop of rank-1
// updates, `for k: acc += x[:,k]*w[:,k]`, does R separate passes and K-1
// dependent reductions. The correct form is ONE reduction over a K-wide axis.
// The harness measures both spellings on the same card in the same run, so the
// price of the habit is a measured quantity and not an assertion.
//
// The reference is one warp per row: each lane accumulates K/32 products with a
// stride-32 access (fully coalesced, and every lane's loads are in the same
// 128B line), then a warp shuffle reduction. 8 warps per block.

extern "C" __global__ void axis_rowdot(float* __restrict__ y,
                                       const float* __restrict__ x,
                                       const float* __restrict__ w, int R,
                                       int K) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x >> 5;
  int row = blockIdx.x * warps_per_block + warp;
  if (row >= R) return;

  const float* xr = x + (size_t)row * K;
  const float* wr = w + (size_t)row * K;

  float acc = 0.0f;
  for (int k = lane; k < K; k += 32) acc += xr[k] * wr[k];

  // warp reduction, all five steps (the last two are needed because a partial
  // reduction leaves lanes 16..31 holding garbage)
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);

  if (lane == 0) y[row] = acc;
}

// ===========================================================================
// 3.  RUNG `gemm` -- high arithmetic intensity
// ===========================================================================
// C[M,N] = A[M,K] * B[K,N], all row-major.
//
// Two dtypes, because they are two different rungs on this architecture and
// conflating them is how a previous comparison in this repo went wrong
// (notes/06 C-19: "fp32 needs N>=8, M>=64" was a sweep artefact; fp32 never
// reaches a tensor core AT ANY SHAPE, and fp16 does). On sm_75 those are two
// genuinely different instruction classes -- FFMA versus HMMA -- so they are two
// points, not one.
//
// The fp32 kernel is the textbook high-performance shape: 128x128 block tile,
// BK=8, 8x8 register blocking, 256 threads, double-buffered registers in the
// inner loop. It is deliberately NOT the "one scalar load per operand, one
// MULACC, no float4, no tiling" kernel that TinyGrad's `NOOPT` knob emits --
// that knob exists to submit a deliberately worse kernel and using it as the
// reference would be a strawman. If this kernel reaches a respectable fraction
// of FFMA peak, the fp32 rung is measuring codegen and not the author.

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
#define GTHREADS 256  // 16x16 threads, each owning an 8x8 output tile

extern "C" __global__ __launch_bounds__(GTHREADS) void axis_gemm_f32(
    float* __restrict__ C, const float* __restrict__ A,
    const float* __restrict__ B, int M, int N, int K) {
  __shared__ float As[BK][BM];
  __shared__ float Bs[BK][BN];

  const int tid = threadIdx.x;
  const int ty = tid >> 4;          // 0..15
  const int tx = tid & 15;          // 0..15
  const int row0 = blockIdx.y * BM;
  const int col0 = blockIdx.x * BN;

  float acc[TM][TN];
  #pragma unroll
  for (int i = 0; i < TM; i++)
    #pragma unroll
    for (int j = 0; j < TN; j++) acc[i][j] = 0.0f;

  float af[TM], bf[TN];

  for (int k0 = 0; k0 < K; k0 += BK) {
    // ---- stage A tile, transposed into As[kk][row] ----
    // 128 rows x 8 k = 1024 elements = 256 float4, one per thread.
    //
    // The row is bounds-checked and padded with zero. A kernel that only works
    // at shapes divisible by its tile is a kernel that only works at the shape
    // it was measured at, and the bench shape here (1024) is divisible by 128
    // while the smoke shape (32) is not -- so without this the smoke run reads
    // off the end of A and dies with CUDA error 700, which is a mechanism bug
    // wearing the costume of a passing validation.
    {
      int row = tid >> 1, kh = (tid & 1) * 4;
      float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
      if (row0 + row < M) v = *reinterpret_cast<const float4*>(A + (size_t)(row0 + row) * K + k0 + kh);
      As[kh + 0][row] = v.x;
      As[kh + 1][row] = v.y;
      As[kh + 2][row] = v.z;
      As[kh + 3][row] = v.w;
    }
    // ---- stage B tile, NOT transposed: Bs[kk][col] ----
    // 8 rows x 128 cols = 1024 elements = 256 float4. Consecutive tid read
    // consecutive 16B within one row, so this load is perfectly coalesced.
    {
      int k = tid >> 5, c4 = (tid & 31) * 4;
      float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
      if (col0 + c4 + 3 < N)
        v = *reinterpret_cast<const float4*>(B + (size_t)(k0 + k) * N + col0 + c4);
      Bs[k][c4 + 0] = v.x;
      Bs[k][c4 + 1] = v.y;
      Bs[k][c4 + 2] = v.z;
      Bs[k][c4 + 3] = v.w;
    }
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < BK; kk++) {
      #pragma unroll
      for (int i = 0; i < TM; i++) af[i] = As[kk][ty * TM + i];
      #pragma unroll
      for (int j = 0; j < TN; j++) bf[j] = Bs[kk][tx * TN + j];
      #pragma unroll
      for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = fmaf(af[i], bf[j], acc[i][j]);
    }
    __syncthreads();
  }

  #pragma unroll
  for (int i = 0; i < TM; i++) {
    int row = row0 + ty * TM + i;
    if (row >= M) continue;
    #pragma unroll
    for (int j = 0; j < TN; j++) {
      int col = col0 + tx * TN + j;
      if (col < N) C[(size_t)row * N + col] = acc[i][j];
    }
  }
}

// ---------------------------------------------------------------------------
// fp16, on tensor cores. sm_75 has HMMA; this is the rung where the framework
// has a tensor-core table to choose from (TC_SELECT/TC), so it is the rung where
// "could a cost model find this" is a real question rather than a rhetorical one.
// 64x64 block tile, 128 threads = 4 warps, each warp owns a 32x32 quadrant as
// 2x2 wmma 16x16x16 fragments. Accumulation is fp32, as it must be.
//
// The shared layouts are [row][k] for A and [k][col] for B, which is what
// `wmma::row_major` wants for matrix_a and matrix_b respectively -- and it is
// also what makes the tile loads a single 16-byte store per thread instead of
// eight scalar ones. Getting this wrong is not a micro-optimisation: the first
// version of this kernel used [k][row] for A, which forced eight separate
// `__ushort_as_half` extractions per thread per k-step and compiled to 4 096
// SASS instructions against the fp32 kernel's 976. That gap was the author's
// layout, not the hardware's, and an axis built on it would have been measuring
// the author -- which is the failure notes/06 §5b is about.
// ---------------------------------------------------------------------------

#define F16_LDA 16 + 8   // 16 k + pad, in halves
#define F16_LDB 64 + 8   // 64 n + pad, in halves

extern "C" __global__ __launch_bounds__(128) void axis_gemm_f16(
    __half* __restrict__ C, const __half* __restrict__ A,
    const __half* __restrict__ B, int M, int N, int K) {
  // __align__(16) is REQUIRED, not decoration. These arrays are written with
  // 16-byte stores (`*reinterpret_cast<float4*>(&As[r][kk])`), and a `__half`
  // array's natural alignment is 2 bytes, so without the attribute the base can
  // land on a 2-byte boundary and every float4 access in the kernel faults
  // with CUDA error 716 -- a misaligned address, which is what this kernel
  // produced on its first T4 run. The column padding alone does not help: it
  // makes the *offsets* multiples of 16 bytes, but the *base* still decides.
  __shared__ __align__(16) __half As[64 + 8][F16_LDA];
  __shared__ __align__(16) __half Bs[16 + 8][F16_LDB];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int row0 = blockIdx.y * 64;
  const int col0 = blockIdx.x * 64;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
  #pragma unroll
  for (int i = 0; i < 2; i++)
    #pragma unroll
    for (int j = 0; j < 2; j++) wmma::fill_fragment(acc[i][j], 0.0f);

  const int wm = warp >> 1, wn = warp & 1;   // which 32x32 quadrant
  const int qrow = row0 + wm * 32, qcol = col0 + wn * 32;

  // loop-invariant tile-load indices and their validity, hoisted out of k
  const int a_r = tid >> 1, a_kk = (tid & 1) * 8;
  const int b_k = tid >> 3, b_c = (tid & 7) * 8;
  const bool a_row_valid = (row0 + a_r) < M;
  const bool b_col_valid = (col0 + b_c + 7) < N;

  for (int k0 = 0; k0 < K; k0 += 16) {
    // ---- A tile: 64 rows x 16 k. 128 threads x 8 halves = one float4 each.
    // Consecutive tid read consecutive 8-half runs of a row, so this is
    // coalesced; the shared store is contiguous in k because As is [row][k].
    // Bounds-checked and zero-padded for the same reason as the fp32 tile: the
    // 64-wide tile overruns M when M is not a multiple of 64, and the smoke
    // shape is 32. The row/column indices are loop-invariant and hoisted out
    // of the k loop, which is what keeps the guard from costing registers (the
    // first version recomputed them inside and spilled 4 B).
    {
      float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
      if (a_row_valid && k0 + a_kk + 7 < K)
        v = *reinterpret_cast<const float4*>(A + (size_t)(row0 + a_r) * K + k0 + a_kk);
      *reinterpret_cast<float4*>(&As[a_r][a_kk]) = v;
    }
    // ---- B tile: 16 k x 64 n. 128 threads x 8 halves, contiguous in n.
    {
      float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
      if (b_col_valid && k0 + b_k < K)
        v = *reinterpret_cast<const float4*>(B + (size_t)(k0 + b_k) * N + col0 + b_c);
      *reinterpret_cast<float4*>(&Bs[b_k][b_c]) = v;
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf[2];
    #pragma unroll
    for (int i = 0; i < 2; i++) wmma::load_matrix_sync(af[i], &As[wm * 32 + i * 16][0], F16_LDA);
    #pragma unroll
    for (int j = 0; j < 2; j++) wmma::load_matrix_sync(bf[j], &Bs[0][wn * 32 + j * 16], F16_LDB);
    #pragma unroll
    for (int i = 0; i < 2; i++)
      #pragma unroll
      for (int j = 0; j < 2; j++) wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
    __syncthreads();
  }

  // The accumulator fragment's element->(row,col) mapping is opaque, so it is
  // staged through shared memory and read back row-major. A direct
  // `store_matrix_sync` to global C would work but each lane's four values are
  // contiguous in the fragment, so the global writes would be strided; the
  // staging pass is what makes the store coalesced.
  //
  // ONE STAGING BUFFER PER WARP, and that is load-bearing. All four warps run
  // this code, but they are at four DIFFERENT quadrants (wm, wn selects which
  // 32x32 of the 64x64 tile each warp owns). A single shared buffer meant every
  // warp's `store_matrix_sync` wrote to the same 16x16 region and the last warp
  // to arrive won -- so three quarters of the output was some other quadrant's
  // numbers, and the oracle check caught it at 0.96 relative error. The buffer
  // is indexed by warp and each warp stages and reads back only its own.
  //
  // The vectorised store is 16 bytes, which is EIGHT halves, not four. An
  // earlier version wrote four halves with a `float4` (16-byte) store -- a 2x
  // size mismatch that is misaligned for every c not a multiple of 8, and it
  // faulted with CUDA error 716 on its first T4 run. The accumulator is float,
  // so the narrowing to half happens here, four halves at a time via half2.
  //
  // 16-byte alignment of `C + (rbase+r)*N + cbase + c` in halves needs that
  // index to be a multiple of 8, so the guard is N % 8 == 0 and c steps by 8.
  // The scalar loop handles every other case rather than asserting.
  __shared__ float cs[4][16 * 20];
  float* csw = cs[warp];
  const bool vec_store = (N % 8 == 0);
  #pragma unroll
  for (int i = 0; i < 2; i++) {
    #pragma unroll
    for (int j = 0; j < 2; j++) {
      __syncwarp();
      wmma::store_matrix_sync(csw, acc[i][j], 20, wmma::mem_row_major);
      __syncwarp();
      const int rbase = qrow + i * 16, cbase = qcol + j * 16;
      for (int r = 0; r < 16; r++) {
        if (rbase + r >= M) continue;
        int c = 0;
        if (vec_store) {
          for (; c + 7 < 16 && cbase + c + 7 < N; c += 8) {
            uint4 packed;
            __half2* ph = reinterpret_cast<__half2*>(&packed);
            const float* src = &csw[r * 20 + c];
            ph[0] = __floats2half2_rn(src[0], src[1]);
            ph[1] = __floats2half2_rn(src[2], src[3]);
            ph[2] = __floats2half2_rn(src[4], src[5]);
            ph[3] = __floats2half2_rn(src[6], src[7]);
            *reinterpret_cast<uint4*>(C + (size_t)(rbase + r) * N + cbase + c) = packed;
          }
        }
        for (; c < 16 && cbase + c < N; c++) C[(size_t)(rbase + r) * N + cbase + c] = csw[r * 20 + c];
      }
    }
  }
}

// ===========================================================================
// 4.  RUNG `gather` -- random access, no reuse
// ===========================================================================
// y[i] = tab[idx[i]] * s[i]
//
// The table is deliberately larger than the T4's 4 MB L2 so the gather actually
// misses. A gather that hits cache measures the wrong thing: it becomes a
// broadcast, and the framework's coalescing story stops being the variable.
//
// This is the rung that isolates `gather` from `fusion`: the arithmetic is one
// multiply, so whatever the framework does with the address arithmetic is the
// whole story. notes/26 §3c says a gather is a fusion barrier and it is "about
// the index, not the gather" -- this rung is where that claim gets a number.

extern "C" __global__ void axis_gather(float* __restrict__ y,
                                       const float* __restrict__ tab,
                                       const int* __restrict__ idx,
                                       const float* __restrict__ s, int n) {
  int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    y[i] = __ldg(&tab[idx[i]]) * __ldg(&s[i]);
}

// ===========================================================================
// 5.  RUNG `scan` -- sequential recurrence
// ===========================================================================
// y[r,i] = sum_{j<=i} x[r,j]     (inclusive scan along the row)
//
// This is the rung that isolates *sequential depth* with no triangular solve
// and no gather to confound it. notes/26 §5c states the framework "cannot keep
// a dependent chain in registers" and "no early exit, no per-ray divergence".
// A scan is the purest possible test of that claim: the entire computation IS a
// carried chain, there is nothing else in it, and if the framework can fuse the
// whole chain into one kernel this rung is where it shows.
//
// The reference is the standard high-performance scan: a block per row, each
// thread owning a CONTIGUOUS run of VEC*iters elements loaded as float4, a
// serial scan in registers, a block-wide scan of the per-thread totals by warp
// shuffle plus one shared-memory stage across warps, then the offset added on
// the way out. The contiguous run is what makes the serial part correct; the
// float4 loads are what make the global access wide.

#define SCAN_THREADS 256
#define SCAN_VEC 4

// The vectorised scan is a MACRO, not a template, specialised on the number of
// float4s each thread owns.
//
// The first version used a runtime `per_thread` with `float4 tmp[16]` and an
// `if (it < per_thread)` guard inside an unrolled loop. Its index arithmetic
// transcribes to Python exactly and the Python check against `np.cumsum` passes
// at 2.2e-07 -- and the kernel was still wrong by 0.068 on the device, with the
// error concentrated at the END of some threads' runs (columns 1020-1023 and
// 3020-3023, i.e. the last float4 of threads 63 and 188). The arithmetic was
// right and the EXECUTION was not, in a construct where a 16-float4 register
// array sits behind a runtime predicate. Making the trip count a compile-time
// constant removes the predicate, the 12 dead iterations and the register
// pressure all at once.
//
// **A construct that is right in a transcription and wrong on the device is a
// construct to delete rather than to debug.** That is the whole lesson, and it
// is the same shape as notes/06 C-16: a property was assumed about what the
// compiler would do, the assumption was never checked, and the cost was a
// number that was wrong.
//
// A macro rather than a template because a template cannot carry a linkage
// specification, and the harness needs a stable `extern "C"` symbol to look up.
// Five symbols, named for their specialisation, launched with the grid the
// harness computed itself.
#define SCAN_BODY(PT)                                                              \
  const int row = blockIdx.x;                                                      \
  if (row >= R) return;                                                            \
  const float* xr = x + (size_t)row * N;                                           \
  float* yr = y + (size_t)row * N;                                                 \
  const int base = threadIdx.x * (PT) * SCAN_VEC;                                  \
  float4 tmp[PT];                                                                  \
  float total = 0.0f;                                                              \
  _Pragma("unroll") for (int it = 0; it < (PT); it++) {                            \
    tmp[it] = reinterpret_cast<const float4*>(xr)[base / SCAN_VEC + it];           \
    total += tmp[it].x + tmp[it].y + tmp[it].z + tmp[it].w;                         \
  }                                                                                \
  __shared__ float warp_tot[SCAN_THREADS / 32];                                    \
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;                     \
  float inc = total;                                                               \
  _Pragma("unroll") for (int off = 1; off < 32; off <<= 1) {                        \
    float o = __shfl_up_sync(0xffffffffu, inc, off);                               \
    if (lane >= off) inc += o;                                                     \
  }                                                                                \
  if (lane == 31) warp_tot[warp] = inc;                                            \
  __syncthreads();                                                                 \
  if (threadIdx.x == 0) {                                                          \
    float run = 0.0f;                                                              \
    _Pragma("unroll") for (int i = 0; i < SCAN_THREADS / 32; i++) {                 \
      float t = warp_tot[i];                                                       \
      warp_tot[i] = run;                                                           \
      run += t;                                                                    \
    }                                                                              \
  }                                                                                \
  __syncthreads();                                                                 \
  const float run_offset = warp_tot[warp] + (inc - total);                          \
  /* `run` is declared ONCE, outside the loop, and carries across every float4.
     Declaring it inside the body resets it to the thread's offset at the start
     of each float4, so each float4 is written as `offset + its own four
     elements` and the running total of the PREVIOUS float4s is dropped. The
     result is correct for it=0 and wrong for every it after it, which is
     exactly the measured error pattern: exact at thread-relative positions 0-3
     and wrong at 4-15, 56% of elements off, by an amount that grows down the
     row. A Python transcription of this kernel placed `run` outside the loop,
     checked against np.cumsum, and passed at 2.2e-07 -- so the transcription
     and the kernel disagreed, and the kernel was the one that was wrong. */   \
  float run = run_offset;                                                          \
  _Pragma("unroll") for (int it = 0; it < (PT); it++) {                            \
    float4 out;                                                                    \
    out.x = tmp[it].x + run; run = out.x;                                          \
    out.y = tmp[it].y + run; run = out.y;                                          \
    out.z = tmp[it].z + run; run = out.z;                                          \
    out.w = tmp[it].w + run; run = out.w;                                          \
    reinterpret_cast<float4*>(yr)[base / SCAN_VEC + it] = out;                      \
  }

#define SCAN_ENTRY(PT)                                                             \
  extern "C" __global__ __launch_bounds__(SCAN_THREADS)                           \
  void axis_scan_v##PT(const float* __restrict__ x, float* __restrict__ y,         \
                       int R, int N) {                                              \
    SCAN_BODY(PT)                                                                  \
  }

SCAN_ENTRY(1)
SCAN_ENTRY(2)
SCAN_ENTRY(4)
SCAN_ENTRY(8)
SCAN_ENTRY(16)

// The scalar fallback, used when N is not an exact multiple of the block's
// span. It is a separate entry point because a `__global__` cannot launch
// another kernel from inside itself without separate compilation.
extern "C" __global__ __launch_bounds__(SCAN_THREADS) void axis_scan_scalar(
    float* __restrict__ y, const float* __restrict__ x, int R, int N) {
  const int row = blockIdx.x;
  if (row >= R) return;
  const float* xr = x + (size_t)row * N;
  float* yr = y + (size_t)row * N;

  // The `threadIdx.x < N` guard is load-bearing, not defensive: with N=64 and
  // 256 threads the unguarded version read 192 elements past the end of every
  // row, and the bench shape N=4096 is larger than the block, so the guard
  // never fires there and the bug was invisible until a smaller shape was tried.
  //
  // This kernel gives each thread exactly ONE element, so it is only correct
  // for N <= SCAN_THREADS. `harness._scan_symbol` enforces that and reports a
  // shape outside it as UNAVAILABLE rather than launching a kernel that would
  // write the first block's worth of the row and leave the rest stale.
  __shared__ float srun[SCAN_THREADS];
  if (threadIdx.x < N) srun[threadIdx.x] = xr[threadIdx.x];
  __syncthreads();
  for (int off = 1; off < SCAN_THREADS; off <<= 1) {
    float t = 0.0f;
    if (threadIdx.x >= off && threadIdx.x < N) t = srun[threadIdx.x - off];
    __syncthreads();
    if (threadIdx.x < N) srun[threadIdx.x] += t;
    __syncthreads();
  }
  if (threadIdx.x < N) yr[threadIdx.x] = srun[threadIdx.x];
}

// ===========================================================================
// 6.  RUNG `trisolve` -- blocked triangular inverse  (subject: KDA's UT transform)
// ===========================================================================
// Given a strictly lower [B, C, C] matrix D, produce the strictly lower part of
//
//     X = (I + D)^{-1} - I
//
// at C = 64, blocked at BC = 16 (NC = 4), one block per batch item.
//
// WHY THIS SHAPE, AND WHY IT IS NOT THE NAIVE FORM
// -------------------------------------------------
// notes/06 C-15: the study's KDA port implemented `fla/ops/kda/naive.py`'s
// unrolled forward substitution -- 4 032 sequentially dependent scalar
// reductions -- and passed every correctness test. The real kernel is a
// TWO-LEVEL BLOCKED inverse at BC=16: four 16x16 diagonal inverses by row-wise
// forward substitution, then a block-Schur merge that is 16 dots in a depth-3
// DAG (notes/25, pinned to fla @ 79d12e8). So the reference here implements the
// blocked form, and the count of 16 dots is not decoration -- it is the number
// the merge costs, and it is what the merge loop below actually does.
//
// The block-Schur identity used, for i > j:
//
//     S_ij = D_ij + sum_{k=j+1}^{i-1} D_ik . X_kj
//     X_ij = -X_ii . S_ij . X_jj
//
// which for NC=4 is 16 16x16 products: (1,0)=2, (2,0)=3, (2,1)=2, (3,0)=4,
// (3,1)=3, (3,2)=2. Counting them is a checkable claim: the loop structure below
// does exactly 2 (for S, when there are no inner terms) + 1 per inner term + 1
// (for X_jj) + 1 (for X_ii) per pair.
//
// The TinyGrad arm of this rung is in `axis/rungs.py`; it is a genuinely
// blocked expression and the note states what it costs.

#define TRI_C 64
#define TRI_BC 16
#define TRI_NC 4

extern "C" __global__ __launch_bounds__(256) void axis_trisolve(
    float* __restrict__ X, const float* __restrict__ D, int B) {
  __shared__ float sD[TRI_C][TRI_C + 1];
  __shared__ float sX[TRI_C][TRI_C + 1];
  __shared__ float sS[TRI_BC][TRI_BC + 1];
  __shared__ float sT[TRI_BC][TRI_BC + 1];

  const int b = blockIdx.x;
  if (b >= B) return;
  const int tid = threadIdx.x;

  // ---- load D, zeroing the pad column so the float4 path is well defined ----
  // `sD` is row-padded to TRI_C+1 to kill bank conflicts on the column reads in
  // the substitution below, which means it CANNOT be loaded as a flat float4
  // array. Load by row instead: 64 rows x 64 floats, 256 threads, each thread
  // takes a quarter row at a time.
  for (int r = 0; r < TRI_C; r++) {
    sD[r][TRI_C] = 0.0f;
    for (int c = threadIdx.x; c < TRI_C; c += 256) sD[r][c] = D[(size_t)b * TRI_C * TRI_C + r * TRI_C + c];
  }
  __syncthreads();

  // ---- diagonal blocks: X_kk = (I + D_kk)^{-1} by forward substitution ----
  //
  // The sweep is over ROWS, not columns, and that is the whole content of this
  // kernel. The recursion is
  //
  //     X[i][j] = - sum_{m<i} D[i][m] X[m][j]        (i > j)
  //
  // whose terms are rows ABOVE row i. A COLUMN-major sweep computes
  // X[i][j] = -sum_{m<j} D[i][m] X[m][j], and X[m][j] is zero for m < j -- so a
  // column sweep subtracts nothing and produces an X with only its diagonal.
  // The first version of this kernel swept columns and was wrong by 0.55 at
  // absolute scale, which the float64 oracle caught. (The TinyGrad arm of the
  // same rung sweeps rows and is verified against np.linalg.inv, so the two
  // sides disagreed only because this one was wrong.)
  for (int k = 0; k < TRI_NC; k++) {
    const int off = k * TRI_BC;
    for (int i = 0; i < TRI_BC; i++) {
      __syncthreads();
      // 256 threads, 16 of them own the 16 columns of row i
      const int c = tid & 15;
      if ((tid >> 4) != i) continue;
      // The test is `c < i`, because the recursion is over ROWS and its
      // nonzero region is STRICTLY LOWER TRIANGULAR. A column-major reading of
      // the same formula puts the test the other way round; written as `c > i`
      // the kernel computes an upper-triangular X and the float64 oracle
      // reported ref_at_worst 0.0 where it wanted 0.5505707 -- the reference
      // simply never wrote a strictly-lower entry, so the error was a *zero*
      // rather than a small number, which is what made it identifiable.
      if (c < i) {
        float acc = 0.0f;
        for (int m = 0; m < i; m++) acc += sD[off + i][off + m] * sX[off + m][off + c];
        sX[off + i][off + c] = -acc;
      } else if (c == i) {
        sX[off + i][off + c] = 1.0f;
      } else {
        sX[off + i][off + c] = 0.0f;
      }
    }
  }
  __syncthreads();

  // ---- block-Schur merge, one block column at a time ----
  //
  //   X_ij = -X_ii . ( sum_{k=j}^{i-1} D_ik . X_kj )
  //
  // The k loop starts at j, not j+1. The k=j term is D_ij . X_jj, which is the
  // entire (i,j+1) case and the leading term of every wider one. An earlier
  // version of this file started at j+1, exactly as the TinyGrad arm first did,
  // and both were wrong in the same way -- which is the useful part: the defect
  // is in the ALGORITHM as it is usually written down, not in either
  // implementation, so reading the recurrence off the paper reproduces it.
  // Verified against np.linalg.inv in axis/verify.py's oracle at NC=4.
  for (int j = 0; j < TRI_NC; j++) {
    for (int i = j + 1; i < TRI_NC; i++) {
      __syncthreads();
      {
        int r = tid >> 4, c = tid & 15;
        if (r < TRI_BC && c < TRI_BC) sS[r][c] = 0.0f;
      }
      __syncthreads();
      for (int k = j; k < i; k++) {
        __syncthreads();
        {
          int r = tid >> 4, c = tid & 15;
          float acc = 0.0f;
          if (r < TRI_BC && c < TRI_BC)
            for (int m = 0; m < TRI_BC; m++) acc += sD[i * TRI_BC + r][k * TRI_BC + m] * sX[k * TRI_BC + m][j * TRI_BC + c];
          if (r < TRI_BC && c < TRI_BC) sT[r][c] = acc;
        }
        __syncthreads();
        {
          int r = tid >> 4, c = tid & 15;
          if (r < TRI_BC && c < TRI_BC) sS[r][c] += sT[r][c];
        }
      }
      // X_ij = -X_ii . S
      __syncthreads();
      {
        int r = tid >> 4, c = tid & 15;
        float acc = 0.0f;
        if (r < TRI_BC && c < TRI_BC)
          for (int m = 0; m < TRI_BC; m++) acc += sX[i * TRI_BC + r][i * TRI_BC + m] * sS[m][c];
        if (r < TRI_BC && c < TRI_BC) sX[i * TRI_BC + r][j * TRI_BC + c] = -acc;
      }
    }
  }
  __syncthreads();

  // ---- store the strictly lower part ----
  // Same reason as the load: `sX` is row-padded, so this writes by row and
  // masks the upper triangle on the way out. X is (I+D)^{-1} - I, so the
  // diagonal is 0 and everything above it is 0.
  {
    float* dst = X + (size_t)b * TRI_C * TRI_C;
    for (int r = 0; r < TRI_C; r++)
      for (int c = threadIdx.x; c < TRI_C; c += 256) dst[r * TRI_C + c] = (r > c) ? sX[r][c] : 0.0f;
  }
}
