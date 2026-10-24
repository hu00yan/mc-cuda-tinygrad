typedef unsigned int uint;
#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
extern "C" __global__ void __launch_bounds__(128) E_128_4_8_16_4(signed char* data0_262144, float* data1_13312, signed char* data2_4194304) {
  int gidx1 = blockIdx.y; /* 128 */
  int lidx0 = threadIdx.x; /* 8 */
  int alu0 = ((gidx1*104)+(lidx0*13));
  float val0 = (*(data1_13312+(alu0+1)));
  float val1 = (*(data1_13312+(alu0+2)));
  float val2 = (*(data1_13312+alu0));
  int gidx0 = blockIdx.x; /* 4 */
  int lidx1 = threadIdx.y; /* 16 */
  float alu1 = trunc(val0);
  float alu2 = trunc(val1);
  float alu3 = trunc(val2);
  int alu4 = ((val0<alu1)?((int)((alu1+-1.0f))):((int)(alu1)));
  int alu5 = ((alu4<0)?0:alu4);
  int alu6 = ((15<alu5)?15:alu5);
  int alu7 = ((val2<alu3)?((int)((alu3+-1.0f))):((int)(alu3)));
  int alu8 = ((alu7<0)?0:alu7);
  int alu9 = ((15<alu8)?15:alu8);
  int alu10 = ((val1<alu2)?((int)((alu2+-1.0f))):((int)(alu2)));
  int alu11 = ((alu10<0)?0:alu10);
  int alu12 = ((15<alu11)?15:alu11);
  int alu13 = (((alu6<16)&(alu9<16)&(alu12<16))?(((lidx0+(gidx1<<3))<<12)+(alu6<<8)+(alu9<<4)+alu12):0);
  int alu14 = ((gidx0<<20)+(lidx1<<16));
  bool alu15 = (alu13<(alu14+16384));
  bool alu16 = (alu13<(alu14+32768));
  signed char val3 = (((alu15!=1)&alu16)?*(data2_4194304+alu13):((signed char)(0)));
  bool alu17 = (alu13<(alu14+49152));
  signed char val4 = (((alu16!=1)&alu17)?*(data2_4194304+alu13):((signed char)(0)));
  signed char val5 = (((alu17!=1)&(alu13<(alu14+65536)))?*(data2_4194304+alu13):((signed char)(0)));
  signed char val6 = ((((alu13<alu14)!=1)&alu15)?*(data2_4194304+alu13):((signed char)(0)));
  int alu18 = ((gidx0<<6)+(lidx1<<2)+(gidx1<<11)+(lidx0<<8));
  *(data0_262144+(alu18+1)) = val3;
  *(data0_262144+(alu18+2)) = val4;
  *(data0_262144+(alu18+3)) = val5;
  *(data0_262144+alu18) = val6;
}