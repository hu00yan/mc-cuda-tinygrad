typedef unsigned int uint;
#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
extern "C" __global__ void __launch_bounds__(32) E_8_32_4(bool* data0_1024) {
  int gidx0 = blockIdx.x; /* 8 */
  int lidx0 = threadIdx.x; /* 32 */
  int alu0 = ((gidx0<<7)+(lidx0<<2));
  *(data0_1024+(alu0+1)) = 0;
  *(data0_1024+(alu0+2)) = 0;
  *(data0_1024+(alu0+3)) = 0;
  *(data0_1024+alu0) = 0;
}