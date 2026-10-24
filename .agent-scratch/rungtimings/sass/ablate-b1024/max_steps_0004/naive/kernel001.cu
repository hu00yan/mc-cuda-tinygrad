typedef unsigned int uint;
#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
extern "C" __global__ void __launch_bounds__(32) E_8192_32_4(int* data0_1048576) {
  int gidx0 = blockIdx.x; /* 8192 */
  int lidx0 = threadIdx.x; /* 32 */
  *((int4*)((data0_1048576+((gidx0<<7)+(lidx0<<2))))) = make_int4(-1,-1,-1,-1);
}