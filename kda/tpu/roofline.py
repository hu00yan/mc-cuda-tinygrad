"""
Per-generation TPU roofline constants, with sources.

Run:  /Users/huyan00/mycode/mc-cuda-tinygrad/.venv/bin/python kda/tpu/roofline.py

CPU-only, numpy not required.  Every constant here is either quoted from a
primary Google document or explicitly marked UNVERIFIED.  Nothing in this file
is measured by us -- there is no TPU in this environment.

WHY THIS FILE EXISTS
--------------------
Two reasons, both forced by facts found during research (see ../README.md §1):

1. Kaggle's TPU v3-8 is DEPRECATED.  Google's own announcement says v3-8 is being
   phased out and sessions "automatically switch to v5e-8".  So a study that
   hard-codes v3-8 numbers is writing a result against hardware it can no longer
   obtain.  The design therefore reports *fractions of the roofline*, and this
   file is the only place a peak number appears.

2. The peak numbers are per CHIP, and the "cores" that JAX exposes are
   TensorCores, of which there are 2 per chip.  TPU v3-8 = 4 chips = 8
   TensorCores.  Getting this wrong by 2x silently doubles or halves every
   roofline fraction.  `chips` is carried explicitly for that reason.

SOURCES (retrieved 2026-09-27)
-----------------------------
[v3]  https://cloud.google.com/tpu/docs/v3
      "Each v3 TPU chip contains two TensorCores. Each TensorCore has two
       matrix-multiply units (MXUs), a vector unit, and a scalar unit."
      Table: peak compute per chip 123 TFLOP/s bf16; HBM2 32 GiB @ 900 GB/s.
      v3-8 = 4 chips, so 128 GiB HBM and 3.6 TB/s aggregate.  (That 128 GiB is
      where the "128 GB HBM" figure in common circulation comes from.)
[arch] https://cloud.google.com/tpu/docs/system-architecture-tpu-vm
      "An MXU is composed of either 256 x 256 (TPU v6e and TPU7x) or 128 x 128
       (TPU versions prior to v6e) multiply-accumulators in a systolic array.
       ... Each MXU is capable of performing 16K multiply-accumulate operations
       per cycle. All multiplies take bfloat16 inputs, but all accumulations are
       performed in FP32."
[pallas-matmul] https://docs.jax.dev/en/latest/pallas/tpu/matmul.html
      "TPU v5es have 197 TFLOP/s of bf16/f32 compute and 819 GB/s of memory
       bandwidth" (link target: .../system-architecture-tpu-vm#tpu_v5e)
[v6e] arXiv:2607.20466 (JAXBench) Fig. 2 caption
      "a single TPU v6e chip (918 TFLOPS bf16, 1640 GB/s HBM)"
[v7x] arXiv:2604.15464 (Ragged Paged Attention) Table 3
      "HBM Capacity per chip 192 GB / Peak HBM Bandwidth per chip 7380 GB/s /
       Peak BF16 Compute per chip 2307 TFLOPs/s" (TPU7x Ironwood, 2 TCs/chip)
[rpa] arXiv:2604.15464 Fig. 1 caption
      "two TensorCores (TCs). Each TC features 64 MB VMEM and 96 GB HBM (chip
       total: 128 MB VMEM, 192 GB HBM, 7380 GB/s HBM bandwidth)"
[vmem] https://docs.jax.dev/en/latest/pallas/tpu/details.html
      "VMEM is fairly large for such a low-level memory hierarchy (16MB+)"
      -- a floor, not a per-generation table.  Only the v7x 64 MB/TC figure is
      sourced.  Every other VMEM entry below is UNVERIFIED.

  def gb(n): return n / 1024**3
"""

GENERATIONS = {
    # name:      (chips, tcs_per_chip, mxu,      bf16_tflops/chip, hbm_gib/chip, hbm_gbps/chip, vmem_mb/tc, source)
    "v3": dict(chips=4, tcs_per_chip=2, mxu=128, bf16_tflops=123.0, hbm_gib=32.0, hbm_gbps=900.0,
               vmem_mb=None, note="v3-8: 4 chips x 2 TC.  [v3][arch]"),
    "v4": dict(chips=2, tcs_per_chip=2, mxu=128, bf16_tflops=None, hbm_gib=None, hbm_gbps=None,
               vmem_mb=None, note="UNVERIFIED.  [arch] only fixes the 128x128 MXU for pre-v6e."),
    "v5e": dict(chips=1, tcs_per_chip=1, mxu=128, bf16_tflops=197.0, hbm_gib=16.0, hbm_gbps=819.0,
                vmem_mb=None, note="[pallas-matmul].  1 TC per chip. hbm_gib UNVERIFIED."),
    "v5p": dict(chips=2, tcs_per_chip=2, mxu=128, bf16_tflops=None, hbm_gib=None, hbm_gbps=None,
                vmem_mb=None, note="UNVERIFIED.  2 TC per chip per [rpa]'s description of modern parts."),
    "v6e": dict(chips=1, tcs_per_chip=1, mxu=256, bf16_tflops=918.0, hbm_gib=32.0, hbm_gbps=1640.0,
                vmem_mb=None, note="[v6e] peak; [arch] 256x256 MXU. hbm_gib UNVERIFIED (32 GiB is the commonly cited figure)."),
    "v7x": dict(chips=1, tcs_per_chip=2, mxu=256, bf16_tflops=2307.0, hbm_gib=192.0, hbm_gbps=7380.0,
                vmem_mb=64.0, note="[v7x] peak per chip (2 TCs), [rpa] for VMEM/HBM. Only gen with a sourced VMEM."),
}

#: Sub-lane x lane of a TPU vector register, 32-bit.  [pallas-matmul] says
#: "8x128 for 32-bit values (as of TPU v6e)"; [rpa] Table 1 says
#: "Vector register of shape (8x128) with 32-bit entries" for v7x; [v3]-era parts
#: are 8x128 for 32-bit too.  JAXBench cites the same for v6e.  Treating this as
#: generation-invariant is a MINOR UNVERIFIED assumption -- if a v3 vector
#: register is 8x128 only after packing, the (8, 128) BlockSpec divisibility
#: rule in tiling_check.py would be wrong for v3.  Verify with tpu-info before
#: trusting v3 numbers.
VREG_SUBLANES, VREG_LANES = 8, 128

#: Bytes moved per DMA granule.  [rpa] §2.1.2: tiling "ensures alignment with
#: the hardware's DMA granularity, which typically operates on 512-byte words,
#: equivalent to 128 contiguous 32-bit entries."
DMA_GRANULE_BYTES = 512

#: bf16 MXU peak in FLOP/s per *TensorCore*, derived.  A TensorCore has
#: MXU_count MXUs; each MXU is mxu x mxu MACs at 1 MAC/cycle.  At 1 GHz that is
#: 2 * mxu**2 GFLOP/s per MXU, which is the right order for every generation
#: here (v6e: 2*256^2 = 131 GFLOP/s/MXU; 918/131 = 7 MXUs -- consistent with
#: the published 6e figure within clock uncertainty).  We do not rely on this
#: derivation anywhere: use the measured per-chip peaks above.


def cores(gen):
    """TensorCores in the whole slice.  JAX's `jax.device_count()`."""
    g = GENERATIONS[gen]
    return g["chips"] * g["tcs_per_chip"]


def peak_flops(gen):
    """bf16 peak FLOP/s for the whole slice (or None if unsourced)."""
    g = GENERATIONS[gen]
    return None if g["bf16_tflops"] is None else g["bf16_tflops"] * 1e12 * g["chips"]


def peak_bw(gen):
    """HBM bytes/s for the whole slice (or None if unsourced)."""
    g = GENERATIONS[gen]
    return None if g["hbm_gbps"] is None else g["hbm_gbps"] * 1e9 * g["chips"]


def hbm_bytes(gen):
    """HBM capacity in BYTES for the whole slice, or None if unsourced."""
    g = GENERATIONS[gen]
    if g["hbm_gib"] is None:
        return None
    return g["hbm_gib"] * g["chips"] * (1 << 30)


def vmem_bytes(gen, scope="per_core"):
    """VMEM bytes for one TensorCore ('per_core') or the whole slice ('slice').

    GENERATIONS[gen]['vmem_mb'] is always PER TensorCore.  Only v7x has a
    sourced figure (64 MB/TC, arXiv:2604.15464 Fig. 1: "Each TC features 64 MB
    VMEM ... chip total: 128 MB VMEM").
    """
    g = GENERATIONS[gen]
    if g["vmem_mb"] is None:
        return None
    mb = 1 << 20
    return g["vmem_mb"] * mb * (cores(gen) if scope == "slice" else 1)


def ridge_point(gen):
    """FLOP/byte at which HBM and MXU are equally busy.

    Below this an operator is memory bound no matter how well it is compiled;
    above it a compiler that stalls the array is the only thing that matters.
    This is the threshold that decides which taxonomy items are even *visible*
    on a given generation -- see ../README.md section 6.
    """
    pf, pb = peak_flops(gen), peak_bw(gen)
    if pf is None or pb is None:
        return None
    return pf / pb


def oi(flops, hbm_bytes_moved):
    """Operational intensity, FLOP per byte of HBM traffic."""
    if hbm_bytes_moved == 0:
        return float("inf")
    return flops / hbm_bytes_moved


def regime(gen, flops, hbm_bytes_moved):
    """'memory' | 'compute' | 'unknown'."""
    r = ridge_point(gen)
    if r is None:
        return "unknown"
    return "compute" if oi(flops, hbm_bytes_moved) >= r else "memory"


def classify(gen, flops, hbm_bytes_moved):
    """One-line summary: intensity, ridge, regime, and the achievable-time bound.

    The two bounds are the point of the whole file.  `t_mem` is the fastest any
    compiler could possibly be if the bytes must move; `t_flop` is the fastest
    any compiler could be if the FLOPs must issue.  max(t_mem, t_flop) is a
    hard floor, and it is a *floor*, so a measured time above it is a statement
    about the schedule, not about the hardware.
    """
    pf, pb = peak_flops(gen), peak_bw(gen)
    if pf is None or pb is None:
        return dict(gen=gen, verdict="UNSOURCED PEAKS for this generation", flops=flops, hbm=hbm_bytes_moved)
    t_mem = hbm_bytes_moved / pb
    t_flop = flops / pf
    return dict(gen=gen, intensity_flop_per_byte=oi(flops, hbm_bytes_moved), ridge=ridge_point(gen),
                regime=regime(gen, flops, hbm_bytes_moved), t_mem_lower_bound_s=t_mem,
                t_flop_lower_bound_s=t_flop, t_hard_floor_s=max(t_mem, t_flop),
                t_hard_floor_ms=1e3 * max(t_mem, t_flop),
                note="bounds assume perfect DMA/MXU overlap and 100% of the cited peak")


if __name__ == "__main__":
    import sys
    print(f"{'gen':5} {'chips':>5} {'TCs':>4} {'MXU':>5} {'bf16 TF/chip':>13} {'HBM GiB':>8} "
          f"{'HBM GB/s':>9} {'ridge F/B':>10}  sources")
    for name, g in GENERATIONS.items():
        tf = "-" if g["bf16_tflops"] is None else f"{g['bf16_tflops']:.0f}"
        hb = "-" if g["hbm_gib"] is None else f"{g['hbm_gib'] * g['chips']:.0f}"
        bw = "-" if g["hbm_gbps"] is None else f"{g['hbm_gbps'] * g['chips']:.0f}"
        rp = "-" if ridge_point(name) is None else f"{ridge_point(name):.0f}"
        print(f"{name:5} {g['chips']:5} {g['tcs_per_chip']:4} {g['mxu']:5} {tf:>13} {hb:>8} "
              f"{bw:>9} {rp:>10}  {g['note']}")

    print()
    print(f"vector register: {VREG_SUBLANES} x {VREG_LANES} x 32-bit  "
          f"({VREG_SUBLANES * VREG_LANES * 4 / 1024:.0f} KiB)")
    print(f"DMA granule:     {DMA_GRANULE_BYTES} B = {VREG_LANES} x 32-bit lanes")
    print()
    print("Sanity check on the two numbers we will actually quote:")
    for gen in ("v3", "v6e"):
        # a 4096^3 bf16 matmul, the shape JAXBench uses for its GEMM baseline
        n, es = 4096, 2
        f = 2 * n ** 3
        b = (n * n * 2 + n * n) * es
        c = classify(gen, f, b)
        print(f"  {gen}: 4096^3 bf16 matmul -> {c['regime']:>7} bound, "
              f"OI={c['intensity_flop_per_byte']:.0f} vs ridge {c['ridge']:.0f}, "
              f"floor {c['t_hard_floor_ms']:.2f} ms for the slice")
    print()
    print("Kaggle note: v3-8 is DEPRECATED (phased out -> v5e-8).  The 128 GiB")
    print("figure for v3-8 is 4 chips x 32 GiB, not 8 cores x 32 GiB.")
    sys.exit(0)
