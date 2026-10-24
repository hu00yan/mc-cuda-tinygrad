#!/usr/bin/env python3
"""
Generate the PTX witness corpus.

Each witness is a minimal PTX module exercising exactly ONE mechanism from the
Hopper/Blackwell state of the art.  Naming is `mechanism@native_arch`, so the file
name records the arch the author intended it for.

Hand-written rather than compiler-generated on purpose: the experiment must hold
the input PTX byte-identical and vary only `ptxas -arch`.  Every witness is
*self-validating* -- src/selfcheck.sh assembles it at its declared native arch
first, so malformed PTX is caught immediately rather than being reported as a
spurious "refusal".  Syntax and operand counts were transcribed from CUTLASS 4.8
`include/cute/arch/*.hpp` and `include/cutlass/arch/*.h` inline-asm strings (the
canonical upstream form), not from memory.
"""
import pathlib

OUT = pathlib.Path(__file__).resolve().parent.parent / "ptx"
OUT.mkdir(parents=True, exist_ok=True)
for stale in OUT.glob("*.ptx"):
    stale.unlink()

HDR = """.version {ver}
.target {target}
.address_size 64
.extern .shared .align 128 .b8 sm[];
.extern .shared .align 8 .b8 mbar[];
"""

W = {}

# ================================================================= Hopper wgmma
# The instruction FlashAttention-3 and CUTLASS 3.x SM90 GEMMs are built on.
# SS variant (both operands from smem via 64-bit descriptors):
# cute/arch/mma_sm90_gmma.hpp MMA_64x8x16_F16F16F16_SS  (AReg=BReg=uint64_t[1],
# CRegisters=float[4] -- note the *4* f32 accumulators, not 2).
# (f16 accumulate -> CRegisters = uint32_t[2], i.e. TWO b32 registers, not four.
#  The f32-accumulate variants below use float[N].  Getting this wrong is the
#  single easiest way to author invalid wgmma.)
W["wgmma_f16_m64n8k16_ss@sm_90a"] = dict(
    ver="8.3", target="sm_90a", body="""
  .reg .b32 %r<16>;
  .reg .b64 %rd<16>;
  .reg .pred %p<4>;
  mov.b32 %r1, 0;
  mov.b64 %rd1, 0;
  mov.b64 %rd2, 0;
  setp.ne.b32 %p1, %r1, 0;
  wgmma.fence.sync.aligned;
  wgmma.mma_async.sync.aligned.m64n8k16.f16.f16.f16
    {%r1, %r2}, %rd1, %rd2, %p1, 1, 1, 0, 0;
  wgmma.wait_group.sync.aligned 0;
""")

# RS variant: A fragment in registers (ARegisters=uint32_t[4]).  The closest
# analogue to what Triton's tl.dot emits on a Hopper target.
W["wgmma_f16_m64n8k16_rs@sm_90a"] = dict(
    ver="8.3", target="sm_90a", body="""
  .reg .b32 %r<16>;
  .reg .b64 %rd<16>;
  .reg .pred %p<4>;
  mov.b32 %r1, 0;
  mov.b64 %rd1, 0;
  setp.ne.b32 %p1, %r1, 0;
  wgmma.fence.sync.aligned;
  wgmma.mma_async.sync.aligned.m64n8k16.f16.f16.f16
    {%r1, %r2}, {%r2, %r3, %r4, %r5}, %rd1, %p1, 1, 1, 0;
  wgmma.wait_group.sync.aligned 0;
""")

# The bf16/f32-accumulate shape CUTLASS 3.x SM90 actually emits at N=128
# (MMA_64x128x16_F32BF16BF16_SS: CRegisters = float[64]).
W["wgmma_bf16_m64n128k16_ss@sm_90a"] = dict(
    ver="8.3", target="sm_90a", body="""
  .reg .b32 %r<16>;
  .reg .b64 %rd<16>;
  .reg .f32 %f<80>;
  .reg .pred %p<4>;
  mov.b32 %r1, 0;
  mov.b64 %rd1, 0;
  mov.b64 %rd2, 0;
  setp.ne.b32 %p1, %r1, 0;
  wgmma.fence.sync.aligned;
  wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16
    {%f1, %f2, %f3, %f4, %f5, %f6, %f7, %f8, %f9, %f10, %f11, %f12, %f13,
     %f14, %f15, %f16, %f17, %f18, %f19, %f20, %f21, %f22, %f23, %f24, %f25,
     %f26, %f27, %f28, %f29, %f30, %f31, %f32, %f33, %f34, %f35, %f36, %f37,
     %f38, %f39, %f40, %f41, %f42, %f43, %f44, %f45, %f46, %f47, %f48, %f49,
     %f50, %f51, %f52, %f53, %f54, %f55, %f56, %f57, %f58, %f59, %f60, %f61,
     %f62, %f63, %f64},
    %rd1, %rd2, %p1, 1, 1, 0, 0;
  wgmma.wait_group.sync.aligned 0;
""")

# wgmma in FP8 -- the "just quantize more" path.
# cute/arch/mma_sm90_gmma.hpp MMA_64x8x32_F16E4M3E4M3_RS_TN
# (ARegisters=uint32_t[4], CRegisters=uint32_t[2] -- f16 accumulate, and only
#  two trailing scale selectors because the RS/TN form takes no B transpose bit).
W["wgmma_f8_m64n8k32_rs@sm_90a"] = dict(
    ver="8.3", target="sm_90a", body="""
  .reg .b32 %r<16>;
  .reg .b64 %rd<16>;
  .reg .pred %p<4>;
  mov.b32 %r1, 0;
  mov.b64 %rd1, 0;
  setp.ne.b32 %p1, %r1, 0;
  wgmma.fence.sync.aligned;
  wgmma.mma_async.sync.aligned.m64n8k32.f16.e4m3.e4m3
    {%r1, %r2}, {%r3, %r4, %r5, %r6}, %rd1, %p1, 1, 1;
  wgmma.wait_group.sync.aligned 0;
""")

# wgmma sparse: 2:4 structured sparsity.  No sm_80/86/89 counterpart exists.
# cute/arch/mma_sm90_gmma_sparse.hpp GMMA_64x8x32_F16F16F16_SS.
W["wgmma_sparse_f16_m64n8k32@sm_90a"] = dict(
    ver="8.3", target="sm_90a", body="""
  .reg .b32 %r<16>;
  .reg .b64 %rd<16>;
  .reg .pred %p<4>;
  mov.b32 %r1, 0;
  mov.b64 %rd1, 0;
  mov.b64 %rd2, 0;
  mov.b32 %r3, 0;
  setp.ne.b32 %p1, %r3, 0;
  wgmma.fence.sync.aligned;
  wgmma.mma_async.sp.sync.aligned.m64n8k32.f16.f16.f16
    {%r1, %r2}, %rd1, %rd2, %r4, 0, %p1, 1, 1, 0, 0;
  wgmma.wait_group.sync.aligned 0;
""")

# ================================================================= tensor core MMA
# Volta, bottom of the ladder: m8n8k4, f16 accumulate
# (cute/arch/mma_sm70.hpp SM70_8x8x4_F16F16F16F16_TN: D=4, A=2, B=2, C=4).
W["mma_f16_m8n8k4_f16acc@sm_70"] = dict(
    ver="7.0", target="sm_70", body="""
  .reg .b32 %r<16>;
  mov.b32 %r1, 0;
  mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16
    {%r1, %r2, %r3, %r4}, {%r5, %r6}, {%r7, %r8}, {%r1, %r2, %r3, %r4};
  ret;
""")

# Turing: the first m16n8k8.  This is the T4's tensor core, in both accumulate
# flavours (cute/arch/mma_sm75.hpp SM75_16x8x8_*).
W["mma_f16_m16n8k8_f32acc@sm_75"] = dict(
    ver="7.0", target="sm_75", body="""
  .reg .b32 %r<16>;
  .reg .f32 %f<16>;
  mov.b32 %r1, 0;
  mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
    {%f1, %f2, %f3, %f4}, {%r2, %r3}, {%r4}, {%f1, %f2, %f3, %f4};
  ret;
""")

W["mma_f16_m16n8k8_f16acc@sm_75"] = dict(
    ver="7.0", target="sm_75", body="""
  .reg .b32 %r<16>;
  mov.b32 %r1, 0;
  mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16
    {%r1, %r2}, {%r3, %r4}, {%r5}, {%r1, %r2};
  ret;
""")

# Ampere: m16n8k16 with f32 accumulate (cute/arch/mma_sm80.hpp
# SM80_16x8x16_F16F16F16F16_TN: D=2, A=4, B=2, C=2).  The instruction the whole
# tensor-core ecosystem targets, and the concrete thing a wgmma kernel must become.
W["mma_f16_m16n8k16_f16acc@sm_80"] = dict(
    ver="7.8", target="sm_80", body="""
  .reg .b32 %r<16>;
  mov.b32 %r1, 0;
  mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
    {%r1, %r2}, {%r3, %r4, %r5, %r6}, {%r7, %r8}, {%r1, %r2};
  ret;
""")

W["mma_f16_m16n8k16_f32acc@sm_80"] = dict(
    ver="7.8", target="sm_80", body="""
  .reg .b32 %r<16>;
  .reg .f32 %f<16>;
  mov.b32 %r1, 0;
  mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    {%f1, %f2, %f3, %f4}, {%r2, %r3, %r4, %r5}, {%r6, %r7}, {%f1, %f2, %f3, %f4};
  ret;
""")

W["mma_bf16_m16n8k16_f32acc@sm_80"] = dict(
    ver="7.8", target="sm_80", body="""
  .reg .b32 %r<16>;
  .reg .f32 %f<16>;
  mov.b32 %r1, 0;
  mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
    {%f1, %f2, %f3, %f4}, {%r2, %r3, %r4, %r5}, {%r6, %r7}, {%f1, %f2, %f3, %f4};
  ret;
""")

# int8 tensor core: the widest sm_80 path (IMMA).
W["mma_s8_m16n8k32_s32@sm_80"] = dict(
    ver="7.8", target="sm_80", body="""
  .reg .b32 %r<16>;
  .reg .s32 %s<16>;
  mov.b32 %r1, 0;
  mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
    {%s1, %s2, %s3, %s4}, {%r2, %r3, %r4, %r5}, {%r6, %r7}, {%s1, %s2, %s3, %s4};
  ret;
""")

# Ada: the first sm_8x with FP8 tensor cores.  PTX ISA 8.4 gates the *type*, and
# sm_89 gates the *instruction*.
W["mma_f8e4m3_m16n8k32@sm_89"] = dict(
    ver="8.4", target="sm_89", body="""
  .reg .b32 %r<16>;
  .reg .f32 %f<16>;
  mov.b32 %r1, 0;
  mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
    {%f1, %f2, %f3, %f4}, {%r2, %r3, %r4, %r5}, {%r6, %r7}, {%f1, %f2, %f3, %f4};
  ret;
""")

W["mma_f8e5m2_m16n8k32@sm_89"] = dict(
    ver="8.4", target="sm_89", body="""
  .reg .b32 %r<16>;
  .reg .f32 %f<16>;
  mov.b32 %r1, 0;
  mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32
    {%f1, %f2, %f3, %f4}, {%r2, %r3, %r4, %r5}, {%r6, %r7}, {%f1, %f2, %f3, %f4};
  ret;
""")

# ================================================================= async copy
# sm_80: `cp.async` / LDGSTS global->shared bypass.  The mechanism every Ampere
# pipelined GEMM is built on, and precisely the thing TMA replaces.
W["cp_async_ca_shared_global@sm_80"] = dict(
    ver="7.8", target="sm_80", body="""
  .reg .b32 %r<16>;
  .reg .b64 %rd<16>;
  mov.b64 %rd1, 0;
  mov.b32 %r1, 0;
  cp.async.ca.shared.global [%r1], [%rd1], 16, %r2;
  cp.async.commit_group;
  cp.async.wait_group 0;
  ret;
""")

W["cp_async_cg_16B@sm_80"] = dict(
    ver="7.8", target="sm_80", body="""
  .reg .b32 %r<16>;
  .reg .b64 %rd<16>;
  mov.b64 %rd1, 0;
  mov.b32 %r1, 0;
  cp.async.cg.shared.global [%r1], [%rd1], 16;
  cp.async.commit_group;
  cp.async.wait_group 0;
  ret;
""")

# ================================================================= mbarrier
# sm_80 introduced mbarrier, and with the PTX-7.0 spelling the whole thing --
# init, arrive, wait, inval -- is expressible.  (The *modern* `mbarrier.try_wait`
# spelling is sm_90-gated; see the two witnesses below.  The distinction matters:
# it is a toolchain surface change, not a hardware change.)
W["mbarrier_full@sm_80"] = dict(
    ver="7.8", target="sm_80", body="""
  .reg .b32 %r<16>;
  .reg .pred %p<8>;
  mov.b32 %r1, 0;
  mov.b32 %r2, 1;
  mbarrier.init.shared.b64 [%r1], %r2;
  mbarrier.arrive.shared.b64 _, [%r1];
  mbarrier.test_wait.parity.shared.b64 %p1, [%r1], 0;
  mbarrier.inval.shared.b64 [%r1];
  ret;
""")

# The modern wait spelling.  ptxas 12.9 gates it at sm_90.
W["mbarrier_try_wait@sm_90"] = dict(
    ver="8.3", target="sm_90", body="""
  .reg .b32 %r<16>;
  .reg .pred %p<8>;
  mov.b32 %r1, 0;
  mov.b32 %r2, 1;
  mbarrier.init.shared.b64 [%r1], %r2;
  mbarrier.arrive.shared.b64 _, [%r1];
  mbarrier.try_wait.parity.shared.b64 %p1, [%r1], 0;
  mbarrier.inval.shared.b64 [%r1];
  ret;
""")

# The transaction-count arrival every TMA pipeline depends on.  Measured: this
# needs sm_90, not sm_80.  So even the *barrier* half of a TMA pipeline is
# Hopper-only even though the barrier itself is not.
W["mbarrier_arrive_expect_tx@sm_90"] = dict(
    ver="8.3", target="sm_90", body="""
  .reg .b32 %r<16>;
  .reg .pred %p<8>;
  mov.b32 %r1, 0;
  mov.b32 %r2, 1;
  mbarrier.init.shared.b64 [%r1], %r2;
  mbarrier.arrive.expect_tx.shared.b64 _, [%r1], 128;
  mbarrier.try_wait.parity.shared.b64 %p1, [%r1], 0;
  ret;
""")

# TMA's byte-accounting barrier flavour.
W["mbarrier_complete_tx@sm_90"] = dict(
    ver="8.3", target="sm_90", body="""
  .reg .b32 %r<16>;
  .reg .b64 %rd<16>;
  mov.b32 %r1, 0;
  mov.b32 %r2, 1;
  mov.b64 %rd1, 0;
  mbarrier.init.shared.b64 [%r1], %r2;
  cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes
    [%r1], [%rd1], 256, [%r1];
  cp.async.bulk.commit_group;
  cp.async.bulk.wait_group 0;
  ret;
""")

# ================================================================= TMA
# sm_90: tensor-memory-access.  Replaces issuing per-thread global loads at all.
W["tma_bulk_tensor_2d@sm_90"] = dict(
    ver="8.6", target="sm_90", body="""
  .reg .b32 %r<16>;
  .reg .b64 %rd<16>;
  mov.b64 %rd1, 0;
  mov.b64 %rd2, 0;
  mov.b32 %r1, 0;
  mov.b32 %r3, 0;
  mov.b32 %r4, 0;
  cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint
    [%r1], [%rd1, {%r3, %r4}], [%r2], %rd2;
  cp.async.bulk.commit_group;
  cp.async.bulk.wait_group 0;
  ret;
""")

# 1D bulk copy: the *non-tensor* half of TMA.  Still absent before sm_90.
W["tma_bulk_global_to_shared@sm_90"] = dict(
    ver="8.3", target="sm_90", body="""
  .reg .b32 %r<16>;
  .reg .b64 %rd<16>;
  mov.b64 %rd1, 0;
  mov.b32 %r1, 0;
  mov.b32 %r2, 256;
  cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes
    [%r1], [%rd1], %r2, [%r1];
  cp.async.bulk.commit_group;
  cp.async.bulk.wait_group 0;
  ret;
""")

# TMA store (1D) -- the epilogue path.
W["tma_bulk_shared_to_global@sm_90"] = dict(
    ver="8.3", target="sm_90", body="""
  .reg .b32 %r<16>;
  .reg .b64 %rd<16>;
  mov.b64 %rd1, 0;
  mov.b32 %r1, 0;
  mov.b32 %r2, 256;
  cp.async.bulk.global.shared::cta.bulk_group [%rd1], [%r1], %r2;
  cp.async.bulk.commit_group;
  cp.async.bulk.wait_group 0;
  ret;
""")

# TMA descriptor *mutation* (tensormap.replace) -- the piece FA3 must reach for
# inline PTX because CUDA exposes no C++ interface for it.
# Note the native arch: this form is sm_90a-gated, not sm_90.
W["tensormap_replace_addr@sm_90a"] = dict(
    ver="8.3", target="sm_90a", body="""
  .reg .b32 %r<16>;
  .reg .b64 %rd<16>;
  mov.b32 %r1, 0;
  mov.b32 %r2, 0;
  mov.b64 %rd1, 0;
  tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%r1], %rd1;
  tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned
    [%r1], [%r2], 128;
  ret;
""")

# ================================================================= Blackwell
# tcgen05: the tensor-memory MMA.  sm_100a only.  The instruction class CUTLASS
# 4.x SM100 kernels are built from.  Transcribed from
# cute/arch/mma_sm100_umma.hpp + cutlass/arch/barrier.h.
W["tcgen05_mma_f16kind@sm_100a"] = dict(
    ver="8.7", target="sm_100a", body="""
  .reg .b32 %r<32>;
  .reg .b64 %rd<16>;
  .reg .f32 %f<16>;
  .reg .pred %p<4>;
  mov.b32 %r1, 0;
  mov.b32 %r2, 1;
  mov.b32 %r3, 0;
  tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%r1], %r2;
  tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
  setp.ne.b32 %p1, %r3, 0;
  tcgen05.mma.cta_group::1.kind::f16 [%r1], %rd1, %rd2, %r3, {%r4, %r5, %r6, %r7}, %p1;
  tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%r8];
  tcgen05.wait::ld.sync.aligned;
  ret;
""")

# tcgen05.ld: reading the TMEM accumulator -- the Blackwell equivalent of
# "copy the accumulator out of the MMA registers" (cute/arch/copy_sm100.hpp).
W["tcgen05_ld_16x256b@sm_100a"] = dict(
    ver="8.7", target="sm_100a", body="""
  .reg .b32 %r<32>;
  .reg .b64 %rd<16>;
  mov.b32 %r1, 0;
  tcgen05.ld.sync.aligned.16x256b.x4.b32
    {%r2, %r3, %r4, %r5, %r6, %r7, %r8, %r9,
     %r10, %r11, %r12, %r13, %r14, %r15, %r16, %r17}, [%r1];
  ret;
""")

# MXFP4 / NVFP4 UMMA: the reason people buy data-centre Blackwell.
# cute/arch/mma_sm100_umma.hpp SM100_MMA_MXF4NVF4_SS.
W["tcgen05_mma_mxf4nvf4@sm_100a"] = dict(
    ver="8.7", target="sm_100a", body="""
  .reg .b32 %r<32>;
  .reg .b64 %rd<16>;
  .reg .pred %p<4>;
  mov.b32 %r1, 0;
  mov.b32 %r2, 1;
  mov.b32 %r3, 0;
  tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%r1], %r2;
  tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
  setp.ne.b32 %p1, %r3, 0;
  tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X
    [%r1], %rd1, %rd2, %r3, [%r4], [%r5], %p1;
  tcgen05.wait::ld.sync.aligned;
  ret;
""")

# Consumer Blackwell uses a *different* FP4 path (mma.sync, not tcgen05), so the
# same file must be probed on both Blackwell targets.  cute/arch/mma_sm120.hpp
# SM120_16x8x64_F8F6F4 (VS=16) shows 8 operand groups.
#
# !! NOT-OBTAINED.  I could not author valid PTX for this form: ptxas rejects it
# with "Arguments mismatch for instruction 'mma'" on sm_120a *and* on every other
# target, so the operand typing is wrong, not the arch.  It is kept, named
# `!unobtainable`, so that selfcheck.sh reports it as a known gap instead of
# letting a broken witness masquerade as a portability finding.  The sm_120a
# question is still answered -- by tcgen05_alloc@sm_120a!neg below, which shows
# tcgen05 is absent on consumer Blackwell.
W["mma_f4_kind_mxf4nvf4@sm_120a!unobtainable"] = dict(
    ver="8.7", target="sm_120a", body="""
  .reg .b32 %r<16>;
  .reg .b16 %h<8>;
  .reg .f32 %f<16>;
  mov.b32 %r1, 0;
  mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X
    .f32.e2m1.e2m1.f32.ue8m0
    {%f1, %f2, %f3, %f4}, {%r1, %r2, %r3, %r4}, {%r5, %r6}, {%f1, %f2, %f3, %f4},
    {%r7}, {%h1, %h2}, {%r8}, {%r9, %r10};
  ret;
""")

# Negative control: the tcgen05 *instruction class* is not available on consumer
# Blackwell either.  Data-centre and consumer Blackwell are separate ISA
# extensions, so "Blackwell supports FP4" is not a single rung on the ladder.
# Named with a `!neg` suffix so selfcheck.sh does not flag it as bad PTX.
W["tcgen05_alloc@sm_120a!neg"] = dict(
    ver="8.7", target="sm_120a", body="""
  .reg .b32 %r<32>;
  .reg .b64 %rd<16>;
  .reg .f32 %f<16>;
  .reg .pred %p<4>;
  mov.b32 %r1, 0;
  mov.b32 %r2, 1;
  mov.b32 %r3, 0;
  tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%r1], %r2;
  tcgen05.mma.cta_group::1.kind::f16 [%r1], %rd1, %rd2, %r3, {%r4, %r5, %r6, %r7}, %p1;
  tcgen05.wait::ld.sync.aligned;
  ret;
""")

# ================================================================= warp/cluster
# sm_75 baseline: ldmatrix (Turing introduced it).
W["ldmatrix_x4_b16@sm_75"] = dict(
    ver="7.0", target="sm_75", body="""
  .reg .b32 %r<16>;
  .reg .b64 %rd<16>;
  .reg .pred %p<4>;
  mov.b32 %r1, 0;
  setp.lt.u32 %p1, %r1, 128;
  @%p1 ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1, %r2, %r3, %r4}, [%r5];
  ret;
""")

# sm_90: warp-level elect.  ptxas 12.9 gates it at sm_90 -- and CUTLASS only ever
# emits it from cute/arch/cluster_sm90.hpp, consistent with that gate.
W["elect_sync@sm_90"] = dict(
    ver="8.3", target="sm_90", body="""
  .reg .b32 %r<16>;
  mov.s32 %r1, 0;
  {
    .reg .b32 %rx;
    .reg .pred %px;
    elect.sync %rx|%px, %r1;
    @%px mov.s32 %r1, 1;
  }
  ret;
""")

# sm_90: stmatrix -- writing fragments back as a matrix, used by wgmma epilogues.
W["stmatrix_x4_b16@sm_90"] = dict(
    ver="8.3", target="sm_90", body="""
  .reg .b32 %r<16>;
  .reg .b64 %rd<16>;
  .reg .pred %p<4>;
  mov.b32 %r1, 0;
  mov.b32 %r2, 0;
  setp.lt.u32 %p1, %r1, 128;
  @%p1 stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r2], {%r1, %r3, %r4, %r5};
  ret;
""")

# sm_90: distributed shared memory / CTA-rank mapping -- the substrate
# warp-specialised producer/consumer kernels need.
W["mapa_shared_cluster@sm_90"] = dict(
    ver="8.3", target="sm_90", body="""
  .reg .b32 %r<16>;
  .reg .b64 %rd<16>;
  mov.b32 %r1, 0;
  mov.b32 %r2, 0;
  mov.b32 %r3, 0;
  mapa.shared::cluster.u32 %r1, %r2, %r3;
  ret;
""")

# sm_90: named cluster barrier -- the other half of warp specialisation.
W["barrier_cluster@sm_90"] = dict(
    ver="8.3", target="sm_90", body="""
  .reg .b32 %r<16>;
  mov.b32 %r1, 0;
  barrier.cluster.arrive.aligned;
  barrier.cluster.wait.aligned;
  ret;
""")


def main():
    for name, spec in W.items():
        txt = (HDR.format(ver=spec["ver"], target=spec["target"])
               + ".visible .entry k() {\n"
               + spec["body"].rstrip() + "\n}\n")
        (OUT / f"{name}.ptx").write_text(txt)
    print(f"wrote {len(W)} witnesses to {OUT}")
    for n in sorted(W):
        print("  ", n)


if __name__ == "__main__":
    main()
