/* Stub. clang's __clang_cuda_runtime_wrapper.h unconditionally #includes
 * curand_mtgp32_kernel.h; the pip CUDA headers we assemble do not ship the
 * curand headers (they live in a separate package we do not need -- we never
 * run anything, we only compile). Nothing here is ever called.
 */
#pragma once
