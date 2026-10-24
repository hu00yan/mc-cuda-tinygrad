typedef unsigned int uint;
#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
extern "C" __global__ void __launch_bounds__(32) E_32768_32_3(unsigned char* data0_3145728, int* data1_1048576, unsigned char* data2_42) {
  int gidx0 = blockIdx.x; /* 32768 */
  int lidx0 = threadIdx.x; /* 32 */
  int val0 = (*(data1_1048576+(lidx0+(gidx0<<5))));
  int alu0 = (val0*3);
  bool alu1 = ((-1<val0)&(val0<14));
  unsigned char val1 = (alu1?*(data2_42+(alu0+1)):((unsigned char)(0u)));
  unsigned char val2 = (alu1?*(data2_42+(alu0+2)):((unsigned char)(0u)));
  unsigned char val3 = (alu1?*(data2_42+alu0):((unsigned char)(0u)));
  int alu2 = ((gidx0*96)+(lidx0*3));
  *(data0_3145728+(alu2+1)) = val1;
  *(data0_3145728+(alu2+2)) = val2;
  *(data0_3145728+alu2) = val3;
}