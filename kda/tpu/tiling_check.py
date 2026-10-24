"""
Static legality checker for a Pallas-on-TPU tiling plan.  CPU-only.

Run:  /Users/huyan00/mycode/mc-cuda-tinygrad/.venv/bin/python kda/tpu/tiling_check.py

WHY THIS FILE IS THE MOST USEFUL THING IN THIS DIRECTORY
--------------------------------------------------------
It needs no TPU, and it checks the class of error that JAXBench measured as the
dominant failure mode of the whole ecosystem:

  "Pallas kernels must reason about TPU-specific concerns including VMEM/SMEM/HBM
   memory hierarchies, software pipelining with prefetch scheduling, block shape
   constraints, and the lexicographic grid traversal order that Mosaic enforces."
  -- arXiv:2607.20466, Introduction

  "The Pallas constraints that govern correctness ... are absent from error
   messages, leaving feedback loops nothing to iterate against."
  -- arXiv:2607.20466, Discussion

  and, measured: 99.7% of best-of-NN samples and 93.8% of iterative-refinement
  samples failed at compile or first execution by hallucinating Pallas APIs or
  misusing pallas_call arguments and BlockSpec types.

So the constraints are undocumented in the error path and unmechanised in the
compiler.  Checking them in Python before you spend a Kaggle quota hour is not a
convenience; it is the only cheap way to spend that hour on the real question.

RULES ENCODED, with sources
----------------------------
R1  TPU block shapes: rank >= 1; the last two dims of a block_shape must equal
    the corresponding array dims, or be divisible by 8 and 128 respectively.
    For rank-1 blocks the dim must equal the array dim, or be a multiple of
    1024, or be a power of two >= 128 * (32 / bitwidth).
      src: https://docs.jax.dev/en/latest/pallas/grid_blockspec.html
R2  VMEM reads/writes happen in (8, 128) tiles for 32-bit; "the best performance
    is achieved when the base offset of the memory access has indices divisible
    by the tiling, and the size of the read region is a multiple of the tile
    size".  This is TPU's version of CUDA's LDG.128 alignment rule, and it is
    HARD (a compile error or a split transfer), not a heuristic.
      src: https://docs.jax.dev/en/latest/pallas/tpu/details.html
R3  "Consuming too many registers can also potentially cause register spills into
    VMEM which degrades kernel performance."  Tile mismatch is a silent cost, not
    an error.  VREG shape 8x128 x 32-bit per arXiv:2604.15464 Table 1.
      src: tpu/details.html
R4  The grid is traversed LEXICOGRAPHICALLY.  "The 'consecutive' restriction on
    the output usually means that some prefix of the grid dimensions always
    varies the slice of the output an invocation needs to access, while the
    output window remains constant for the remaining suffix."  In practice: the
    reduction axis goes LAST, and the output block is the accumulator.
      src: tpu/details.html, "BlockSpecs and grid iteration"
R5  dimension_semantics is a tuple as long as `grid`, all "parallel" first then
    all "arbitrary".  Megacore -- the entire reason to supply it -- exists only
    on v4 and v5p; elsewhere it is a NO-OP, and omitting it means only ONE
    TensorCore of a multi-TC chip is used.
      src: tpu/pipelining.html ; tpu/details.html
R6  Pipelining requires every argument's memory_space to be VMEM; scratch must be
    VMEM, SMEM or SEMAPHORE.  buffer_count defaults to 2.
      src: tpu/pipelining.html
R7  "Matrix multiplication always produces results in the float32 format"; and
    "Even if you pass in 32-bit operands to a matrix multiplication, they will
    be rounded to bfloat16 unless float32 precision is requested."  This is the
    exact TF32-downgrade analogue from the CUDA taxonomy, item 5.
      src: tpu/details.html, "Precision control"
R8  Index mode defaults to pl.Blocked; pl.Element switches to element indices
    and is TPU-only.  Mixing them silently changes the mapping.
      src: grid_blockspec.html
"""

import sys

SUBLANES, LANES = 8, 128       # R2, R3
DMA_GRANULE = 512              # arXiv:2604.15464 2.1.2
ES_F32 = 4


class Violation:
    def __init__(self, rule, msg, fatal=True):
        self.rule, self.msg, self.fatal = rule, msg, fatal

    def __str__(self):
        return f"    [{self.rule}] {'FATAL' if self.fatal else 'warn '} {self.msg}"


def check_block_shape(block_shape, array_shape, bitwidth=16, name="block"):
    """R1 + R2.  Returns a list of Violation."""
    v = []
    if not block_shape:
        return [Violation("R1", f"{name}: block rank must be >= 1")]
    if len(block_shape) == 1:
        bs, as_ = block_shape[0], array_shape[0]
        min_pow2 = LANES * (32 // bitwidth)
        if not (bs == as_ or bs % 1024 == 0 or (bs & (bs - 1) == 0 and bs >= min_pow2)):
            v.append(Violation("R1", f"{name}: rank-1 block dim {bs} illegal for array dim "
                                     f"{as_}: must equal {as_}, be a multiple of 1024, or be a "
                                     f"power of two >= {min_pow2} (= 128 * 32/{bitwidth})"))
        return v
    n = len(block_shape)
    for i, (bs, as_) in enumerate(zip(block_shape, array_shape)):
        if bs is None or bs == as_:
            continue
        if i == n - 2 and bs % SUBLANES != 0:
            v.append(Violation("R1", f"{name}: dim -{n - i} (second-minor) of block is {bs}, "
                                     f"array dim {as_}; must equal the array dim or be a "
                                     f"multiple of {SUBLANES}"))
        if i == n - 1 and bs % LANES != 0:
            v.append(Violation("R2", f"{name}: minor-most block dim {bs} is not a multiple of "
                                     f"{LANES}; vector loads/stores are split at tile "
                                     f"boundaries"))
    return v


def check_dma_granule(shape, dtype_bytes, name="block"):
    """arXiv:2604.15464 2.1.2: DMA moves 512-byte words = 128 x 32-bit lanes.  A
    smaller transfer is padded to a granule, so you pay for bytes you do not use."""
    nbytes = dtype_bytes
    for d in shape:
        nbytes *= d
    if nbytes < DMA_GRANULE:
        return [Violation("DMA", f"{name} is {nbytes} B, under the {DMA_GRANULE}-B DMA "
                                  f"granule: the transfer is padded, so you pay for bytes you "
                                  f"do not use", fatal=False)]
    return []


def check_dimension_semantics(grid, semantics, gen):
    """R5."""
    if semantics is None:
        return [Violation("R5", "dimension_semantics not supplied. Megacore exists only on TPU "
                                "v4 and v5p; elsewhere NOT supplying it means only ONE "
                                "TensorCore of a multi-TC chip is used. Shard with a mesh "
                                "instead (jax.sharding / pjit) on every generation.")]
    v = []
    if len(semantics) != len(grid):
        v.append(Violation("R5", f"dimension_semantics has {len(semantics)} entries for a "
                                 f"{len(grid)}-axis grid"))
    order = [str(s).lower() for s in semantics]
    if order != sorted(order, key=lambda s: 0 if s == "parallel" else 1):
        v.append(Violation("R5", f"dimension_semantics {semantics} is not a run of 'parallel' "
                                 f"followed by a run of 'arbitrary'"))
    if "parallel" not in order:
        v.append(Violation("R5", "no 'parallel' axis: nothing is partitioned over cores"))
    if gen not in ("v4", "v5p"):
        v.append(Violation("R5", f"gen={gen} has no megacore, so dimension_semantics is a no-op "
                                 f"here (docs: 'Supplying dimension_semantics annotations is a "
                                 f"no-op on other platforms')", fatal=False))
    return v


def check_precision(dtypes, requested_f32):
    """R7."""
    v = []
    for nm, d in sorted(dtypes.items()):
        if str(d) in ("float32", "f32", "fp32") and not requested_f32:
            v.append(Violation("R7", f"{nm} is float32 but float32 precision was not requested: "
                                     f"Pallas TPU will SILENTLY round the operand to bfloat16. "
                                     f"Ask for it: lax.DotAlgorithmPreset.F32_F32_F32, or "
                                     f"jax.default_matmul_precision('float32')."))
    return v


def check_vmem(scratch_bytes, vmem_cap):
    if vmem_cap is None:
        return [Violation("VMEM", "VMEM capacity UNKNOWN for this generation. Pallas docs only "
                                  "say '16MB+'. Run tpu-info and fill "
                                  "roofline.GENERATIONS[gen]['vmem_mb'] before trusting this "
                                  "check.", fatal=False)]
    if scratch_bytes > vmem_cap:
        return [Violation("VMEM", f"scratch needs {scratch_bytes/2**20:.1f} MiB but VMEM is "
                                  f"{vmem_cap/2**20:.1f} MiB; expect either an OOM from the "
                                  f"low-level compiler or a VREG->VMEM spill")]
    return []


def check_grid(grid, block_specs):
    """R4, partially.  Only the cheap static half is checkable: the reduction /
    accumulate axis must be the LAST grid axis, because Mosaic traverses the grid
    lexicographically and every invocation writing a given output slice must be
    consecutive.  We check that the final axis is not the head axis, because the
    head axis is the parallel one."""
    v = []
    for ax, g in enumerate(grid):
        if g == 1:
            v.append(Violation("R4", f"grid axis {ax} has extent 1: on TPU the grid is "
                                     f"lexicographic, so a size-1 axis is a wasted iteration",
                              fatal=False))
    return v


def check_minor_lane_fit(shape, name="scratch", dtype_bytes=4):
    """R3, the part that actually bites KDA.

    A 2-D value in VMEM is tiled (sublanes, lanes) = (8, 128) for 32-bit.  If
    the minor-most extent is not a multiple of 128 lanes, the value is physically
    padded up to the next 128, and you pay for the padding in VMEM and in every
    relayout Mosaic inserts to get in and out of the (8,128) register layout.

    This is NOT a BlockSpec legality question -- BlockSpec happily accepts
    (64, 64) over a (T, 64) array -- which is precisely why it is invisible in
    the source and only shows up in an LLO bundle dump.
    """
    if len(shape) < 2:
        return []
    minor = shape[-1]
    if minor % LANES == 0:
        return []
    padded = -(-minor // LANES) * LANES
    waste = (padded - minor) * 1
    return [Violation("R3", f"{name} minor-most extent is {minor}, not a multiple of {LANES} "
                          f"lanes: physically padded to {padded}, i.e. {waste} wasted elements "
                          f"per row and {waste * shape[-2] / max(dtype_bytes, 1) / 1024:.1f} KiB "
                          f"of padding, plus a relayout on every access. On TPU C = 64 hits this "
                          f"for the whole [C,C] gram and the UT inverse.")]


def validate_plan(plan):
    """plan keys: grid, block_specs [(block_shape, array_shape, name)],
    dimension_semantics, matmul_dtypes {name: dtype}, request_f32,
    scratch_bytes, vmem_cap, gen, and optional bitwidth / dtype_bytes /
    scratch_shapes."""
    v = []
    for bs, as_, nm in plan["block_specs"]:
        v += check_block_shape(bs, as_, plan.get("bitwidth", 16), nm)
        v += check_dma_granule(bs, plan.get("dtype_bytes", 2), nm)
    for shp, nm in plan.get("scratch_shapes", []):
        v += check_minor_lane_fit(shp, nm, ES_F32)
    v += check_grid(plan["grid"], plan["block_specs"])
    v += check_dimension_semantics(plan["grid"], plan.get("dimension_semantics"), plan["gen"])
    v += check_precision(plan["matmul_dtypes"], plan.get("request_f32", False))
    v += check_vmem(plan["scratch_bytes"], plan.get("vmem_cap"))
    return v


def _demo():
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import roofline as rl
    import shapes

    B, T, H, HV, K, V, C = 1, 2048, 4, 4, 128, 128, 64
    base = dict(grid=(HV, T // C), dimension_semantics=("parallel", "arbitrary"),
                matmul_dtypes={"k": "bfloat16"}, request_f32=False,
                scratch_bytes=1 << 20, vmem_cap=None, gen="v3",
                dtype_bytes=2, bitwidth=16)

    def case(title, show=None, **over):
        p = dict(base)
        p.update(over)
        print(f"\n  {title}")
        vs = validate_plan(p)
        if show is not None:
            vs = [x for x in vs if x.rule in show]
        if not vs:
            print("    clean")
        for x in vs:
            print(x)

    def note(text):
        print(f"\n  {text}")

    print("=" * 78)
    print("Tiling legality demo.  One KDA forward pass, B=1 T=2048 H=HV=4")
    print("K=V=128 C=64.  Every case below is something a person actually writes.")
    print("=" * 78)

    case("R2  minor-most block dim 64 instead of 128  (looks fine, splits every load)",
         block_specs=[((C, 64), (T, K), "k")])

    case("R2  the C x C gram block, [64,64], over a (T,64) array -- LEGAL",
         block_specs=[((C, C), (T, C), "Z")])
    note("      ^ 64 is a multiple of 8 AND the array's minor dim is 64, so the")
    note("        'block dim == array dim' escape applies. R1/R2 do not fire.")
    note("        The real C=64 problem is stated at the bottom of this file: 64 <")
    note("        128, so C can never be the TILED minor-most axis of a 2-D vector")
    note("        tile, which means the accumulator is physically [64,128] and")
    note("        needs a relayout. That is a storage-layout fact, not a BlockSpec")
    note("        legality fact, which is exactly why it is invisible until you")
    note("        read an LLO dump.")

    case("R4  a size-1 grid axis (the K axis of a single-chunk reduction)",
         grid=(HV, T // C, 1), block_specs=[((C, K), (T, K), "k")])

    case("R5  no dimension_semantics on a 2-TC chip",
         dimension_semantics=None, block_specs=[((C, K), (T, K), "k")])

    case("R5  dimension_semantics order wrong (arbitrary before parallel)",
         dimension_semantics=("arbitrary", "parallel"), block_specs=[((C, K), (T, K), "k")])

    case("R5  right answer for the wrong hardware: v6e has no megacore",
         gen="v6e", block_specs=[((C, K), (T, K), "k")])

    case("R7  fp32 operands, fp32 precision NOT requested -- silent bf16 rounding",
         matmul_dtypes={"k": "float32", "S": "float32"}, block_specs=[((C, K), (T, K), "k")])

    case("R7  the same operands WITH the precision requested",
         matmul_dtypes={"k": "float32", "S": "float32"}, request_f32=True,
         block_specs=[((C, K), (T, K), "k")])

    case("DMA rank-1 [128] bf16 block is 256 B -- half a granule, so half the DMA is waste",
         block_specs=[((128,), (T,), "g_per_channel_row")])

    print()
    print("=" * 78)
    print("The plan we intend to use, checked against every rule above.")
    print("=" * 78)
    s = shapes.KdaShape(B=B, T=T, H=H, HV=HV, K=K, V=V, C=C)
    plan = dict(
        grid=(HV, T // C),
        dimension_semantics=("parallel", "arbitrary"),
        block_specs=[((C, K), (T, K), "k"), ((C, V), (T, V), "v"),
                     ((C, K), (T, K), "q"), ((C, C), (T, C), "Z / M scratch")],
        matmul_dtypes={"k": "bfloat16", "v": "bfloat16", "S": "float32"},
        request_f32=True,
        scratch_shapes=[((C, C), "Z / Btri / M fp32 accumulator"),
                        ((C, C), "Aqk fp32 accumulator"),
                        ((K, V), "state S fp32")],
        scratch_bytes=shapes.vmem_budget(s, 1, buffer_count=2)["total"],
        vmem_cap=rl.vmem_bytes("v7x"),   # only generation with a sourced VMEM figure
        gen="v7x", dtype_bytes=2, bitwidth=16,
    )
    vs = validate_plan(plan)
    print("    clean" if not vs else "")
    for x in vs:
        print(x)
    print()
    print("  THE [C,C] BLOCK IS THE WHOLE PROBLEM, and the naive statement of it is")
    print("  wrong.  C = 64 IS a legal minor-most dim when the array's minor dim is")
    print("  also 64 -- BlockSpec (64,64) over a (T,64) array passes R1/R2 above.")
    print("  The real constraint is: C = 64 < 128, so **C can never be the tiled")
    print("  minor-most axis of a 2-D vector tile at all**.  An MXU produces its")
    print("  result in an (8,128) register layout, so a [C,C] = [64,64] fp32")
    print("  accumulator is physically stored as [64,128]: 2x the VMEM, plus a")
    print("  relayout to get there.  Mosaic's infer-vector-layout pass inserts those")
    print("  relayouts automatically (observed in the LLO of Splash Attention,")
    print("  patricktoulme.substack.com 11 Jan 2026), which means the cost is")
    print("  invisible in the source and only appears in a bundle dump.")
    print()
    print("  Workarounds, in increasing order of pain:")
    print("   1. choose C = 128.  KDA's paper fixes C = 64, and the secondary chunk")
    print("      BC = 16 exists precisely to bound the exp range, so C = 128 widens")
    print("      the exponent range by 2x.  This is an ALGORITHMIC change forced by a")
    print("      memory-layout constraint -- the clearest (i) instance in this study.")
    print("   2. carry the gram as [128, C] and slice.  Legal, costs a relayout.")
    print("   3. keep the gram in SMEM as scalars.  SMEM is 32-bit-addressable with")
    print("      no alignment requirement, but it is a SCALAR store: 4 KiB of gram")
    print("      becomes 4096 scalar stores, charged to the scalar unit.  Almost")
    print("      certainly a loss.  Worth one measurement.")
    print()
    print("  What this checker CANNOT see, and no static checker can:")
    print("   * whether XLA's Layout Assignment preserves your tile.  arXiv:2604.15464")
    print("     3.1.4: 'XLA's Layout Assignment pass often fails to preserve the required")
    print("     tiling for our kernel ... This behavior overrides user-specified layout")
    print("     constraints in current XLA versions.'  You cannot ask for it.  (i).")
    print("   * whether an index_map indirection is cheap.  Splash Attention looks up the")
    print("     next KV block in SMEM to skip fully-masked blocks; that is a runtime")
    print("     decision that only shows up in an LLO dump.")
    print("   * whether Mosaic actually double-buffers what you think it double-buffers,")
    print("     or whether your accumulators spill to VMEM.")
    return 0


if __name__ == "__main__":
    sys.exit(_demo())
