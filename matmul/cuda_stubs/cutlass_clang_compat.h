/* clang-vs-nvcc header compatibility shim. Force-included for CUTLASS builds.
 *
 * nvcc's <cuda_runtime.h> pulls in crt/device_functions.h, which declares the
 * memory-order enum used by CUTLASS's subbyte_reference.h
 * (`atomicAdd(subbyte, ..., __NV_ATOMIC_RELAXED)`).
 *
 * clang defines __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__ before including
 * nvcc's headers, and that macro blanks the body of crt/device_functions.h --
 * so the enum never becomes visible and CUTLASS fails to compile with
 * "use of undeclared identifier '__NV_ATOMIC_RELAXED'". We have no nvcc here
 * (see Dockerfile.toolchain), so we re-declare the enum with NVIDIA's own
 * values. It is a declaration only; the definitions still come from clang's
 * __clang_cuda_runtime_wrapper.h.
 */
#pragma once

#ifndef __CUTLASS_CLANG_COMPAT_H
#define __CUTLASS_CLANG_COMPAT_H

enum {
  __NV_ATOMIC_RELAXED = 0,
  __NV_ATOMIC_CONSUME = 1,
  __NV_ATOMIC_ACQUIRE = 2,
  __NV_ATOMIC_RELEASE = 3,
  __NV_ATOMIC_ACQ_REL = 4,
  __NV_ATOMIC_SEQ_CST = 5
};

/* Two more host-side CUDA runtime constants that CUTLASS references from
 * gemm/device/gemm_universal_adapter.h (it opts into oversized shared memory on
 * sm_90+). The pip CUDA header set we assemble predates them, and clang's
 * runtime wrapper does not declare them, so CUTLASS fails to compile. Declared
 * as plain constants rather than enums, because the surrounding enum
 * cudaFuncAttribute is already defined by the real headers and redefining it
 * would clash. Only ever used as a function argument, never called.
 *
 * cudaFuncAttributeSharedMemoryMode = 11 (CUDA 12.x)
 * cudaSharedMemoryModeAllowOversizedSharedMemory = 1
 */
#ifndef CUTLASS_CUDA_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE
#define CUTLASS_CUDA_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE 11
#endif
#ifndef CUTLASS_CUDA_SHARED_MEMORY_MODE_ALLOW_OVERSIZED
#define CUTLASS_CUDA_SHARED_MEMORY_MODE_ALLOW_OVERSIZED 1
#endif
#if !defined(__CUDACC__) || !defined(CUTLASS_HAVE_SHARED_MEMORY_MODE_ENUM)
constexpr int cudaFuncAttributeSharedMemoryMode = CUTLASS_CUDA_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE;
constexpr int cudaSharedMemoryModeAllowOversizedSharedMemory = CUTLASS_CUDA_SHARED_MEMORY_MODE_ALLOW_OVERSIZED;
#endif

#endif
