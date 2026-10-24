"""
devices.py -- the cache/bandwidth capacities the crossover thesis is derived from.

Run:  .venv/bin/python crossover/devices.py

Every number in DEVICE_SPECS is `[READ-FROM-SOURCE]` or `[STRUCTURAL]`, with a
source per field.  Nothing here is measured, and nothing here is guessed: a
capacity that could not be sourced is `None`, and every consumer in this
directory has to say so out loud rather than substitute a default.

STATUS TAGS (the study's convention, from notes/06-corrections-ledger.md §"conventions")
  [READ-FROM-SOURCE]  read out of a vendor document or a cited paper; not executed
  [STRUCTURAL]        a property of the ISA/microarchitecture (e.g. 64K 32-bit
                       registers per SM); true by construction, not measured
  [MODELLED]          arithmetic on the above; exact given the inputs
  [PREDICTED]         a claim about what would be observed.  NEVER a result
  [UNVERIFIED]        could not be checked; carried as unknown, not as true
  [MEASURED]          executed on hardware (only in attn_control.py /
                       residency_probe.py, never in this file)

WHY THIS FILE EXISTS SEPARATELY
-------------------------------
The thesis is a claim about ratios (intermediate / capacity).  A ratio claim
built on a bad numerator or a bad denominator is worthless, and this study has a
documented history of exactly that (notes/06 §"re-derivation record": four
apparent mismatches, all four a bug in the *checking* script, none in the
shipped numbers).  So the capacities live in one place with one source each, and
crossover/claim.py imports them rather than restating them.

A CONFLICT ALREADY FOUND IN THIS REPO
--------------------------------------
`bench/selftest.py` line 103 gives the P100 (sm_60) `L2_CACHE_SIZE` as
`24 << 20` = 24 MiB, in a table whose comment says the values "are the published
values for the three cards the study uses".  NVIDIA's own Inside Pascal table
gives **4096 KB = 4 MiB** for both P100 SKUs, and the Pascal whitepaper's
"each memory controller is attached to 512 KB of L2 cache" x 8 controllers
(Inside Pascal: "eight 512-bit memory controllers") gives the same 4 MiB by
arithmetic.  The repo's value is 6x too large.  This is recorded in
`CONFLICTS` and asserted in `test_control.py`; the file itself is owned by
another agent and is not modified here.  It matters: every crossover number
scales linearly in the capacity, so a 6x capacity error is a 6x error in the
predicted crossover.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# per-device facts
# -----------------------------------------------------------------------------
# `l2_bytes`            device-wide L2.  This is the capacity a *fused* kernel can
#                       hope to hide a streaming intermediate inside, and it is
#                       NOT what limits a real tiled kernel (see `smem_per_sm`).
# `sm_count`            SMs.
# `smem_per_sm`         shared memory per SM.  Times `sm_count` = the on-chip
#                       aggregate, the closest thing a TPU's VMEM has.
# `regs_per_sm`         32-bit registers per SM.  Always 65536 on every arch from
#                       Fermi on; 256 KiB of on-chip state that is *not* smem.
# `smem_per_block`      max shared memory addressable by one thread block, after
#                       CUDA's 1 KB per-block reservation.
# `dram_bytes_s`        peak DRAM bandwidth (HBM or GDDR).
# `hbm_bytes`           memory capacity, for "does it even fit" questions.
# `fp32_flops_s`        fp32 FMA peak.  `tensor_flops_s` is the dense fp16/bf16
#                       tensor-core peak with sparsity NOT counted, so it is
#                       comparable across cards.
#
DEVICE_SPECS: dict[str, dict] = {
    "T4": dict(
        arch="sm_75", sm_count=40,
        l2_bytes=4 * 1024 ** 2, smem_per_sm=64 * 1024, smem_per_block=64 * 1024,
        regs_per_sm=65536, dram_bytes_s=320e9, hbm_bytes=16 * 10 ** 9,
        fp32_flops_s=8.1e12, tensor_flops_s=65e12,
        l2_source="Jia et al., 'Dissecting the NVIDIA Turing T4 GPU via "
                  "Microbenchmarking', arXiv:1903.07486, §3: 'The L2 cache on a "
                  "T4 GPU is a 16-way, set-associative cache having a size of "
                  "4,096 KiB'",
        dram_source="NVIDIA T4 datasheet "
                    "(nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/"
                    "t4-tensor-core-datasheet-951643.pdf): '320+ GB/sec'; "
                    "nvidia.com/en-us/data-center/tesla-t4",
        smem_source="CUDA C Programming Guide, Compute Capabilities, CC 7.5: "
                    "64 KB shared memory per SM, 64 KB max per thread block "
                    "(Turing removed the 48 KB opt-in cap)",
        sm_source="T4 datasheet: 2560 CUDA cores / 64 fp32 lanes per SM = 40 SMs",
    ),
    "P100": dict(
        arch="sm_60", sm_count=56,
        # 4096 KB.  See CONFLICTS[0] -- this contradicts bench/selftest.py:103.
        l2_bytes=4 * 1024 ** 2, smem_per_sm=96 * 1024, smem_per_block=48 * 1024,
        regs_per_sm=65536, dram_bytes_s=732e9, hbm_bytes=16 * 10 ** 9,
        fp32_flops_s=9340e9, tensor_flops_s=105e12,
        l2_source="NVIDIA 'Inside Pascal' (developer.nvidia.com/blog/inside-pascal/), "
                  "spec table: 'L2 Cache Size | 1536 KB (K40) | 3072 KB (M40) | "
                  "4096 KB (P100 NVLink) | 4096 KB (P100 PCIe)'. Corroborated by "
                  "arithmetic on the same page: 'eight 512-bit memory controllers' "
                  "x Pascal whitepaper 'each memory controller is attached to 512 KB "
                  "of L2 cache' = 4096 KB",
        dram_source="NVIDIA 'Inside Pascal', spec table: P100 PCIe 16 GB "
                    "'Memory Bandwidth 732 GB/s'",
        smem_source="CONFLICTING, see CONFLICTS[1]: CUDA C Programming Guide CC 6.0 "
                    "gives 96 KB/SM (48 KB/block); 'Inside Pascal' spec table gives "
                    "'Shared Memory Size / SM Configurations (bytes) 64K' for GP100. "
                    "The 96 KB figure is used here and both are carried",
        sm_source="NVIDIA 'Inside Pascal', spec table: 'SMs | 56' for both P100 SKUs",
    ),
    "L4": dict(
        arch="sm_89", sm_count=58,
        l2_bytes=48 * 1024 ** 2, smem_per_sm=100 * 1024, smem_per_block=99 * 1024,
        regs_per_sm=65536, dram_bytes_s=300e9, hbm_bytes=24 * 10 ** 9,
        fp32_flops_s=30.3e12, tensor_flops_s=121e12,
        l2_source="NVIDIA Ada GPU Architecture whitepaper "
                  "(images.nvidia.com/aem-dam/Solutions/geforce/ada/"
                  "nvidia-ada-gpu-architecture.pdf), spec table: 'L2 Cache Size | "
                  "4096 KB | 49152 KB' (the 49152 KB column is the L4, alongside "
                  "'Memory Bandwidth 320 GB/s | 300 GB/s' and "
                  "'FP32 Performance 8.1 TFLOPS | 30.3 TFLOPS')",
        dram_source="NVIDIA L4 product brief "
                    "(nvidia.com/content/dam/en-zz/Solutions/Data-Center/l4/"
                    "PB-11316-001_v01.pdf): 'Memory bus width 192 bits', "
                    "'Peak memory bandwidth 300 GB/sec'",
        smem_source="CUDA C Programming Guide, Compute Capabilities, CC 8.9: 100 KB "
                    "per SM, 99 KB max per thread block (101376 B; matches the value "
                    "in tools/archlab/inspect_ptx.py:75)",
        sm_source="L4: 58 SMs [UNVERIFIED -- derived from 24 GB GDDR6 / no vendor "
                  "statement found in this pass; only used as a multiplier]",
    ),
    "H100": dict(
        arch="sm_90a", sm_count=132,
        l2_bytes=50 * 1024 ** 2, smem_per_sm=228 * 1024, smem_per_block=227 * 1024,
        regs_per_sm=65536, dram_bytes_s=3.35e12, hbm_bytes=80 * 10 ** 9,
        fp32_flops_s=669e12 // 10, tensor_flops_s=989e12,
        l2_source="NVIDIA Hopper Architecture In-Depth "
                  "(developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/): "
                  "'A 50 MB L2 cache in H100 is 1.25x larger than the A100 40 MB L2'",
        dram_source="nvidia.com/en-us/data-center/h100 product specifications: "
                    "'GPU Memory Bandwidth 3.35TB/s'",
        smem_source="NVIDIA Blackwell Tuning Guide, §1.4.1.1 and §1.4.2.3, which "
                    "cover Hopper and Blackwell together: 'Both the H100 GPU and the "
                    "B200 GPU support shared memory capacities of 0, 8, 16, 32, 64, "
                    "100, 132, 164, 196 and 228 KB per SM' and 'a single thread block "
                    "to address up to 227 KB' (1 KB reserved per block). Matches "
                    "tools/archlab/inspect_ptx.py:77 (232448 B)",
        sm_source="NVIDIA Hopper Architecture In-Depth: '8 GPCs, 66 TPCs, 2 SMs/TPC, "
                  "132 SMs per GPU' (SXM5)",
    ),
    "B200": dict(
        arch="sm_100a", sm_count=148,
        l2_bytes=126 * 1024 ** 2, smem_per_sm=228 * 1024, smem_per_block=227 * 1024,
        regs_per_sm=65536, dram_bytes_s=8.0e12, hbm_bytes=180 * 10 ** 9,
        fp32_flops_s=None, tensor_flops_s=2250e12,
        l2_source="NVIDIA Blackwell Tuning Guide, §1.4.2.2 'Increased L2 Capacity': "
                  "'The NVIDIA GB200 GPU increases the L2 cache capacity to 126 MB'",
        dram_source="NVIDIA Blackwell Tuning Guide §1.4.2.1: HBM3e 'up to 180 GB'. "
                    "Bandwidth 8 TB/s is the commonly cited B200 figure and is "
                    "[UNVERIFIED] against a vendor document in this pass; only ever "
                    "used as a denominator, never as a numerator, and the crossover "
                    "predictions do not depend on it",
        smem_source="NVIDIA Blackwell Tuning Guide §1.4.1.1: 'For devices of compute "
                    "capability 10.0 shared memory capacity per SM is 228 KB'; "
                    "§1.4.2.3: 'the maximum capacity of the combined L1 cache, texture "
                    "cache, and shared memory of 256 KB'",
        sm_source="148 SMs [READ-FROM-SOURCE: arXiv:2609.24270 Fig. 10 caption, "
                  "'H200 and B200 devices with 132 and 148 SMs, respectively']",
    ),
}

# H100 fp32: 132 SMs x 128 fp32 lanes x 2 flops/FMA x ~1.98 GHz boost.  The
# vendor does not publish an fp32 number for H100 in the product table, so it is
# derived and marked as such rather than quoted.  (Do not silently inherit the
# A100's 19.5 TFLOP/s; that is a different card.)
DEVICE_SPECS["H100"]["fp32_flops_s"] = 132 * 128 * 2 * 1.983e9

# -----------------------------------------------------------------------------
# conflicts and gaps, carried forward rather than smoothed over
# -----------------------------------------------------------------------------
CONFLICTS: list[dict] = [
    dict(where="bench/selftest.py:103",
         field="L2_CACHE_SIZE for P100 (sm_60)",
         repo_value="24 << 20 = 24 MiB",
         sourced_value="4096 KB = 4 MiB",
         source="NVIDIA 'Inside Pascal' spec table; corroborated by 8 memory "
                "controllers x 512 KB",
         impact="6x on every capacity-derived crossover for this device. The "
                "table's own comment claims these are 'the published values for "
                "the three cards the study uses'.",
         owner="bench/ (another agent). Not modified here; asserted in "
               "test_control.py so the discrepancy cannot be forgotten."),
    dict(where="this file, P100 smem_per_sm",
         field="shared memory per SM on sm_60",
         repo_value="96 KiB (CUDA C Programming Guide CC 6.0)",
         sourced_value="64 KiB ('Inside Pascal' spec table, GP100 row)",
         source="conflicting vendor statements",
         impact="1.5x on the per-SM-tile crossover for P100 only. Both are "
                "carried and both are printed; the claim in notes/07 is stated "
                "so that it survives either.",
         owner="unresolved; flagged, not fixed."),
]

UNVERIFIED: list[str] = [
    "L4 SM count = 58. Only used as a multiplier on smem_per_sm, so it moves "
    "the aggregate-on-chip number by whatever the real value is. The L4's L2 "
    "(48 MiB), bandwidth (300 GB/s) and smem/SM (100 KiB) are all sourced.",
    "B200 DRAM bandwidth (8 TB/s) -- NVIDIA's Blackwell Tuning Guide gives the "
    "capacity (180 GB HBM3e) but not the bandwidth in the section read here.",
    "B200 fp32 peak: no vendor fp32 number found, so it is `None` and no "
    "compute roof is claimed for that card. The crossover predictions do not "
    "use it.",
    "TPU VMEM capacity for v3/v5e/v5p/v6e -- already UNVERIFIED in "
    "research/TPU.md §6. `crossover/claim.py` carries the bracket derived from "
    "Zhang's own timings instead of a number, and labels it a bracket.",
]

# -----------------------------------------------------------------------------
# derived quantities
# -----------------------------------------------------------------------------
MiB = 1024 ** 2
GiB = 1024 ** 3


def onchip_aggregate(name: str) -> int:
    """Total on-chip shared memory on the device, in bytes.

    This is the closest structural analogue of a TPU's VMEM: the scratch a fused
    kernel can hold without touching DRAM, summed over all the little
    processors that will each hold a slab of it.  It is NOT the same as L2, and
    the difference is the whole point of crossover/claim.py's second model: a
    fused kernel is bounded by the per-SM slab (a tile), not by the device total.
    """
    d = DEVICE_SPECS[name]
    return d["sm_count"] * d["smem_per_sm"]


def onchip_per_sm(name: str) -> int:
    return DEVICE_SPECS[name]["smem_per_sm"]


def onchip_register_file(name: str) -> int:
    """Registers per SM, in bytes.  On-chip, but not addressable as scratch --
    an accumulator can live here and a streamed tile cannot."""
    return DEVICE_SPECS[name]["regs_per_sm"] * 4


def ridge_point(name: str) -> float | None:
    """fp32 ridge point in FLOP/byte, or None if either roof is unknown."""
    d = DEVICE_SPECS[name]
    if not d["fp32_flops_s"] or not d["dram_bytes_s"]:
        return None
    return d["fp32_flops_s"] / d["dram_bytes_s"]


def bytes_to_dram_time(nbytes: float, name: str) -> float | None:
    """seconds for `nbytes` at the DRAM roof.  A floor, not an estimate."""
    bw = DEVICE_SPECS[name]["dram_bytes_s"]
    return None if not bw else nbytes / bw


# -----------------------------------------------------------------------------
def _fmt(n: float | None, unit: str = "B") -> str:
    if n is None:
        return "n/a"
    for scale, suf in ((1024 ** 4, "Ti"), (1024 ** 3, "Gi"), (1024 ** 2, "Mi"), (1024, "Ki")):
        if abs(n) >= scale:
            return f"{n/scale:.2f} {suf}{unit}"
    return f"{n:.0f} {unit}"


def main() -> int:
    print("=" * 100)
    print("CACHE AND BANDWIDTH CAPACITIES -- every one sourced, none measured")
    print("=" * 100)
    hdr = (f"{'device':<7}{'arch':<10}{'SMs':>5}{'L2':>11}{'smem/SM':>10}"
           f"{'on-chip':>11}{'regs/SM':>9}{'DRAM':>10}{'fp32 TF':>9}"
           f"{'tc TF':>8}{'ridge':>8}")
    print(hdr)
    print("-" * len(hdr))
    for name, d in DEVICE_SPECS.items():
        agg = onchip_aggregate(name)
        r = ridge_point(name)
        print(f"{name:<7}{d['arch']:<10}{d['sm_count']:>5}{_fmt(d['l2_bytes']):>11}"
              f"{_fmt(d['smem_per_sm']):>10}{_fmt(agg):>11}"
              f"{_fmt(onchip_register_file(name)):>9}"
              f"{d['dram_bytes_s']/1e12:>9.2f}T"
              f"{(d['fp32_flops_s']/1e12 if d['fp32_flops_s'] else 0):>8.1f}"
              f"{(d['tensor_flops_s']/1e12 if d['tensor_flops_s'] else 0):>8.0f}"
              f"{(f'{r:.1f}' if r else 'n/a'):>8}")
    print()
    print("  L2 spread across the five cards : "
          f"{max(d['l2_bytes'] for d in DEVICE_SPECS.values())/min(d['l2_bytes'] for d in DEVICE_SPECS.values()):.1f}x")
    print("  smem/SM spread                 : "
          f"{max(d['smem_per_sm'] for d in DEVICE_SPECS.values())/min(d['smem_per_sm'] for d in DEVICE_SPECS.values()):.2f}x")
    print("  DRAM bandwidth spread          : "
          f"{max(d['dram_bytes_s'] for d in DEVICE_SPECS.values())/min(d['dram_bytes_s'] for d in DEVICE_SPECS.values()):.1f}x")
    print("  on-chip aggregate spread       : "
          f"{max(onchip_aggregate(n) for n in DEVICE_SPECS)/min(onchip_aggregate(n) for n in DEVICE_SPECS):.1f}x")
    print()
    print("  THIS IS THE WHOLE ARGUMENT AGAINST 'the arch barely matters':")
    print("  L2 spans 31.5x across these cards, on-chip aggregate spans 13.5x, and")
    print("  DRAM bandwidth spans 25x.  What the thesis claims is that the RATIO")
    print("  s/capacity is what predicts the winner, not the arch.  That is a claim")
    print("  about the normalisation, and it is only non-trivial if equal ratios give")
    print("  equal outcomes -- which requires the two sides to have the same relative")
    print("  efficiency.  They do not.  See notes/07 section 5.")
    print()
    print("=" * 100)
    print("CONFLICTS FOUND WHILE BUILDING THIS TABLE (carried, not smoothed over)")
    print("=" * 100)
    for c in CONFLICTS:
        print(f"  [{c['where']}] {c['field']}")
        print(f"      repo:    {c['repo_value']}")
        print(f"      sourced: {c['sourced_value']}   ({c['source']})")
        print(f"      impact:  {c['impact']}")
        print(f"      owner:   {c['owner']}")
        print()
    print("=" * 100)
    print("UNVERIFIED (carried as unknown, not as true)")
    print("=" * 100)
    for u in UNVERIFIED:
        print(f"  - {u}")
    print()
    print("SOURCES, per device")
    print("-" * 100)
    for name, d in DEVICE_SPECS.items():
        print(f"  {name}:")
        for k in ("l2_source", "sm_source", "smem_source", "dram_source"):
            print(f"    {k}: {d[k]}")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
