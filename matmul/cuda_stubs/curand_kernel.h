/* Stub. cutlass/util/reference/device/tensor_fill.h includes curand_kernel.h and
 * uses curandState_t to fill reference tensors with random numbers. We never run
 * any of that (compile only: no GPU, no driver, no CUDA runtime), we only need
 * the declarations to exist so the preprocessor and the type system are
 * satisfied. Nothing here is ever called.
 *
 * curandState_t is 48 bytes of curand's private state, which is all that is
 * needed for the declarations in tensor_fill.h to typecheck.
 */
#pragma once

/* curandState_t must be a complete type: CUTLASS embeds one by value in
 * RandStructuredState. curand's real state is 48 bytes of private data; the
 * size is irrelevant to us because we never run any of this, but the type must
 * be complete to compile.
 */
struct curandState { unsigned long long _v[6]; };
typedef struct curandState curandState_t;

/* Real curand's host API is extern "C"; CUTLASS's tensor_fill.h includes this
 * header and then calls curand_init/curand_normal from inline __device__ code,
 * which clang lowers to PTX `call.extern`. ptxas then needs an *extern
 * function* declaration with the C-mangled name, so these must be extern "C" --
 * with C++ linkage clang emits call instead and ptxas aborts with
 * "Unresolved extern function '_Z11curand_inityyyP11curandState'".
 */
extern "C" {
__device__ void curand_init(unsigned long long seed, unsigned long long subsequence,
                            unsigned long long offset, curandState_t *state);
__device__ float curand_uniform(curandState_t *state);
__device__ double curand_uniform_double(curandState_t *state);
__device__ float curand_normal(curandState_t *state);
__device__ double curand_normal_double(curandState_t *state);
}
