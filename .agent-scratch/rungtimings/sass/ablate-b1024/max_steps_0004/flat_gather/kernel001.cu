typedef unsigned int uint;
#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
extern "C" __global__ void __launch_bounds__(16) r_1024_16_16(signed char* data0_1024, signed char* data1_262144) {
  signed char buf0[1];
  signed char buf1[1];
  __shared__ __align__(16) signed char buf2[16];
  int gidx0 = blockIdx.x; /* 1024 */
  int lidx0 = threadIdx.x; /* 16 */
  *(buf0+0) = ((signed char)(0));
  for (int Ridx0 = 0; Ridx0 < 16; Ridx0++) {
    signed char val0 = (*(data1_262144+((lidx0<<4)+Ridx0+(gidx0<<8))));
    *(buf0+0) = ((*(buf0+0))+val0);
  }
  *(buf2+lidx0) = (*(buf0+0));
  __syncthreads();
  *(buf1+0) = ((signed char)(0));
  for (int Ridx102 = 0; Ridx102 < 16; Ridx102++) {
    signed char val1 = (*(buf2+Ridx102));
    *(buf1+0) = ((*(buf1+0))+val1);
  }
  if ((lidx0==0)) {
    *(data0_1024+gidx0) = (*(buf1+0));
  }
}