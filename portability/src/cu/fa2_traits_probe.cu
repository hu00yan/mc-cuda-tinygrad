// FA2's arch-conditional selection, taken from the real header.
//
// flash-attention 2.8.3, csrc/flash_attn/src/kernel_traits.h:18-42:
//     #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
//         static constexpr bool Has_cp_async = true;
//         using MMA_Atom_Arch = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;   // m16n8k16
//     #else
//         static constexpr bool Has_cp_async = false;
//         using MMA_Atom_Arch = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;    // m16n8k8
//     #endif
// and, at :39-44,
//     #if __CUDA_ARCH__ >= 750
//         using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;    // ldmatrix
//     #else
//         using SmemCopyAtom = Copy_Atom<DefaultCopy, elem_type>;          // ld/st
//
// Two things change together, from one source file, with no diagnostic:
// the tensor-core instruction shape AND whether the mainloop is async at all.
//
// This TU reproduces the guard with the two MMA instructions spelled as inline
// PTX (the two shapes CuTe's atoms above expand to), so the SASS reading shows
// which instruction class each arch actually gets from the *same* source.  It is
// a transcription of FA2's guard, not FA2 itself -- building FA2 for real needs
// libtorch headers, and the torch wheel host is unreachable from this network.
// See portability/MATRIX.md for that limitation, stated in the table.
#include <cstdint>

__device__ __forceinline__ uint32_t sm80_mma_16x8x16(float (&d)[4], const uint32_t (&a)[4],
                                                      const uint32_t (&b)[2]) {
  float c[4] = {d[0], d[1], d[2], d[3]};
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#pragma unroll
  for (int i = 0; i < 4; ++i) d[i] = c[i];
  return 0;
}

// m16n8k8 with f32 accumulate has FOUR accumulator registers per thread
// (cute/arch/mma_sm75.hpp: SM75_16x8x8_F32F16F16F32_TN -> DRegisters = float[4],
// ARegisters = uint32_t[2], BRegisters = uint32_t[1]), whereas m16n8k16 has two.
__device__ __forceinline__ uint32_t sm75_mma_16x8x8(float (&d)[4], const uint32_t (&a)[2],
                                                     const uint32_t& b0) {
  float c[4] = {d[0], d[1], d[2], d[3]};
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(b0));
  d[0] = c[0]; d[1] = c[1]; d[2] = c[2]; d[3] = c[3];
  return 0;
}

extern "C" __global__ void k_fa2_traits(float* out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  constexpr bool Has_cp_async = true;
  float d[4] = {0, 0, 0, 0};
  uint32_t a[4] = {1, 2, 3, 4}, b[2] = {5, 6};
  sm80_mma_16x8x16(d, a, b);
#else
  constexpr bool Has_cp_async = false;
  float d[4] = {0, 0, 0, 0};
  uint32_t a[2] = {1, 2};
  sm75_mma_16x8x8(d, a, uint32_t(5));
#endif
  out[threadIdx.x] = d[0] + (Has_cp_async ? 0.f : 0.f);
}

int main() { return 0; }
