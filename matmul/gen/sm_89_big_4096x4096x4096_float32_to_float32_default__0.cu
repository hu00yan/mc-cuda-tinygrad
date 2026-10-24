typedef unsigned int uint;
#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
extern "C" __global__ void __launch_bounds__(128) r_128_64_8_16_4_4_1024_4(float* data0_16777216, float* data1_16777216, float* data2_16777216) {
  float buf0[16];
  int gidx0 = blockIdx.x; /* 64 */
  int gidx1 = blockIdx.y; /* 128 */
  int lidx0 = threadIdx.x; /* 8 */
  int lidx1 = threadIdx.y; /* 16 */
  int alu0 = ((gidx0<<6)+(lidx1<<2));
  int alu1 = ((gidx1<<17)+(lidx0<<14));
  *(buf0+0) = 0.0f;
  *(buf0+1) = 0.0f;
  *(buf0+2) = 0.0f;
  *(buf0+3) = 0.0f;
  *(buf0+4) = 0.0f;
  *(buf0+5) = 0.0f;
  *(buf0+6) = 0.0f;
  *(buf0+7) = 0.0f;
  *(buf0+8) = 0.0f;
  *(buf0+9) = 0.0f;
  *(buf0+10) = 0.0f;
  *(buf0+11) = 0.0f;
  *(buf0+12) = 0.0f;
  *(buf0+13) = 0.0f;
  *(buf0+14) = 0.0f;
  *(buf0+15) = 0.0f;
  for (int Ridx0 = 0; Ridx0 < 1024; Ridx0++) {
    int alu18 = (alu1+(Ridx0<<2));
    float4 val0 = (*((float4*)((data1_16777216+(alu18+4096)))));
    float4 val1 = (*((float4*)((data1_16777216+(alu18+8192)))));
    float4 val2 = (*((float4*)((data1_16777216+(alu18+12288)))));
    float4 val3 = (*((float4*)((data1_16777216+alu18))));
    int alu19 = (alu0+(Ridx0<<14));
    float4 val4 = (*((float4*)((data2_16777216+(alu19+4096)))));
    float4 val5 = (*((float4*)((data2_16777216+(alu19+8192)))));
    float4 val6 = (*((float4*)((data2_16777216+(alu19+12288)))));
    float4 val7 = (*((float4*)((data2_16777216+alu19))));
    *(buf0+0) = ((*(buf0+0))+(val3.x*val7.x)+(val3.y*val4.x)+(val3.z*val5.x)+(val3.w*val6.x));
    *(buf0+1) = ((*(buf0+1))+(val3.x*val7.y)+(val3.y*val4.y)+(val3.z*val5.y)+(val3.w*val6.y));
    *(buf0+2) = ((*(buf0+2))+(val3.x*val7.z)+(val3.y*val4.z)+(val3.z*val5.z)+(val3.w*val6.z));
    *(buf0+3) = ((*(buf0+3))+(val3.x*val7.w)+(val3.y*val4.w)+(val3.z*val5.w)+(val3.w*val6.w));
    *(buf0+4) = ((*(buf0+4))+(val0.x*val7.x)+(val0.y*val4.x)+(val0.z*val5.x)+(val0.w*val6.x));
    *(buf0+5) = ((*(buf0+5))+(val0.x*val7.y)+(val0.y*val4.y)+(val0.z*val5.y)+(val0.w*val6.y));
    *(buf0+6) = ((*(buf0+6))+(val0.x*val7.z)+(val0.y*val4.z)+(val0.z*val5.z)+(val0.w*val6.z));
    *(buf0+7) = ((*(buf0+7))+(val0.x*val7.w)+(val0.y*val4.w)+(val0.z*val5.w)+(val0.w*val6.w));
    *(buf0+8) = ((*(buf0+8))+(val1.x*val7.x)+(val1.y*val4.x)+(val1.z*val5.x)+(val1.w*val6.x));
    *(buf0+9) = ((*(buf0+9))+(val1.x*val7.y)+(val1.y*val4.y)+(val1.z*val5.y)+(val1.w*val6.y));
    *(buf0+10) = ((*(buf0+10))+(val1.x*val7.z)+(val1.y*val4.z)+(val1.z*val5.z)+(val1.w*val6.z));
    *(buf0+11) = ((*(buf0+11))+(val1.x*val7.w)+(val1.y*val4.w)+(val1.z*val5.w)+(val1.w*val6.w));
    *(buf0+12) = ((*(buf0+12))+(val2.x*val7.x)+(val2.y*val4.x)+(val2.z*val5.x)+(val2.w*val6.x));
    *(buf0+13) = ((*(buf0+13))+(val2.x*val7.y)+(val2.y*val4.y)+(val2.z*val5.y)+(val2.w*val6.y));
    *(buf0+14) = ((*(buf0+14))+(val2.x*val7.z)+(val2.y*val4.z)+(val2.z*val5.z)+(val2.w*val6.z));
    *(buf0+15) = ((*(buf0+15))+(val2.x*val7.w)+(val2.y*val4.w)+(val2.z*val5.w)+(val2.w*val6.w));
  }
  int alu37 = (alu0+alu1);
  *((float4*)((data0_16777216+(alu37+4096)))) = make_float4((*(buf0+4)),(*(buf0+5)),(*(buf0+6)),(*(buf0+7)));
  *((float4*)((data0_16777216+(alu37+8192)))) = make_float4((*(buf0+8)),(*(buf0+9)),(*(buf0+10)),(*(buf0+11)));
  *((float4*)((data0_16777216+(alu37+12288)))) = make_float4((*(buf0+12)),(*(buf0+13)),(*(buf0+14)),(*(buf0+15)));
  *((float4*)((data0_16777216+alu37))) = make_float4((*(buf0+0)),(*(buf0+1)),(*(buf0+2)),(*(buf0+3)));
}