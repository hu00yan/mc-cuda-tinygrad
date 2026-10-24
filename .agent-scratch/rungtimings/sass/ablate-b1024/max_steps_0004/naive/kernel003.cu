typedef unsigned int uint;
#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
extern "C" __global__ void __launch_bounds__(16) r_1024_16_16(signed char* data0_1024, float* data1_13312, signed char* data2_262144) {
  signed char buf0[1];
  signed char buf1[1];
  __shared__ __align__(16) signed char buf2[16];
  int gidx0 = blockIdx.x; /* 1024 */
  int lidx0 = threadIdx.x; /* 16 */
  *(buf0+0) = ((signed char)(0));
  for (int Ridx0 = 0; Ridx0 < 16; Ridx0++) {
    signed char val0 = (*(data2_262144+((lidx0<<4)+Ridx0+(gidx0<<8))));
    *(buf0+0) = ((*(buf0+0))+val0);
  }
  *(buf2+lidx0) = (*(buf0+0));
  __syncthreads();
  *(buf1+0) = ((signed char)(0));
  for (int Ridx102 = 0; Ridx102 < 16; Ridx102++) {
    signed char val1 = (*(buf2+Ridx102));
    *(buf1+0) = ((*(buf1+0))+val1);
  }
  int alu8 = (gidx0*13);
  float val2 = (*(data1_13312+(alu8+1)));
  float val3 = (*(data1_13312+(alu8+2)));
  float val4 = (*(data1_13312+alu8));
  float alu9 = trunc(val2);
  float alu10 = trunc(val3);
  float alu11 = trunc(val4);
  int alu12 = ((val2<alu9)?((int)((alu9+-1.0f))):((int)(alu9)));
  int alu13 = ((alu12<0)?0:alu12);
  int alu14 = ((val3<alu10)?((int)((alu10+-1.0f))):((int)(alu10)));
  int alu15 = ((alu14<0)?0:alu14);
  int alu16 = ((val4<alu11)?((int)((alu11+-1.0f))):((int)(alu11)));
  int alu17 = ((alu16<0)?0:alu16);
  int alu18 = ((15<alu13)?15:alu13);
  int alu19 = ((15<alu15)?15:alu15);
  int alu20 = ((15<alu17)?15:alu17);
  signed char alu21 = (((alu18<16)&(alu20<16)&(alu19<16))?(*(buf1+0)):((signed char)(0)));
  if ((lidx0==0)) {
    *(data0_1024+gidx0) = alu21;
  }
}