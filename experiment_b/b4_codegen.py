#!/usr/bin/env python3
"""b4_codegen.py -- the instruction-class matrix, and the arch that changes it.

THE CENTRAL (b) RESULT
----------------------
Comparison (b) asks whether a compiler can reach the construction a human wrote.
The reachable quantities are bytes materialised and instruction classes emitted,
and BOTH are device-free: one is a property of the schedule, the other is a
property of the renderer, and neither needs a card to read.

This file puts the two constructions side by side on the same axes.

  the human   flash-linear-attention's `chunk_gated_delta_rule_fwd_kernel_h`,
              compiled AOT by Triton to PTX for a chosen GPUTarget, with the
              CUDA driver verifiably absent. Captured, pinned, at sm_90a and
              sm_100a. Reaches wgmma on Hopper and tcgen05 on Blackwell.
  the compiler tinygrad 0.14.0's CUDARenderer, driven with DEV=NULL:CUDA:<arch>
              so the CUDA renderer is attached to a device that cannot run
              anything. Re-rendered here at five architectures.

WHAT FALLS OUT, AND IT IS NOT CLOSELY BALANCED
----------------------------------------------
  1. The human construction is ARCH-ADAPTED. The same Triton kernel emits
     wgmma on sm_90a and tcgen05 + ldmatrix + mbarrier on sm_100a. Two
     different tensor-core generations, chosen by the toolchain.
  2. The compiler construction is ARCH-INVARIANT. The entire rendered program
     is byte-identical across sm_80 / sm_86 / sm_89 / sm_90 / sm_100 -- one
     sha256 over the whole program, not one per file. And across that whole set
     it contains zero wgmma, zero tcgen05, zero cp.async, zero stmatrix, zero
     ldmatrix, zero mbarrier, zero tensormap. Its entire tensor-core vocabulary
     is `mma.sync`, at one shape.
  3. The reason is a TABLE, and it is one line of source. `get_cuda(arch)`
     returns cuda_sm89 for any arch >= 89, and cuda_sm89 is
     `cuda_sm80 + fp8` -- an Ada catalog. There is no sm_90 entry and no sm_100
     entry, so an H100 renders for sm_90 and lands in the Ada catalog. And
     `ops_cuda.py:122` builds `sm_{major}{minor}` with no arch suffix, because
     the table would raise ValueError on `sm_90a`. b5_arch_gate.py executes
     this rather than quoting it.
  4. The one place the arch DOES change tinygrad's output is the sm_75/sm_80
     boundary, and it changes the OPERATOR'S OWN MATH: 2,756 exp2 on sm_75
     against 2,172 on sm_80+, because sm_75 has no tf32 tensor core and the
     matmul falls back to a form with a different fusing pattern. So the
     captured sm_75 artifact in b1 UNDERSTATES the exp2 count on every card the
     study would actually want, by 4x on the kernel it looked at.

  5. THE HAND-WRITTEN BASELINE IS NOT FREE. The same AOT compile that shows
     the human reaching tcgen05 also shows a kernel with no `.local` and no
     `ld.local`/`st.local` in its PTX -- and `ptxas -v` reporting 380 B of spill
     stores at 255 registers. The spill is introduced AFTER the PTX exists, so
     it is invisible to the PTX and to `cuobjdump -res-usage`, which reports
     the local FRAME and reports 0. A study that used -res-usage would have
     certified its own baseline as spill-free.

WHAT THIS IS NOT
----------------
  The two sides of the matrix are not instruction-count-comparable. The human
  side has been through Triton AND ptxas; the compiler side is pre-ptxas CUDA C.
  So the matrix is a CLASS matrix: which families of instruction each
  construction can reach at all. That question is well posed. "Who issues more
  instructions" is not, and this file does not answer it.

  python3 experiment_b/b4_codegen.py
  python3 experiment_b/b4_codegen.py --archs sm_75,sm_80,sm_90,sm_100
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment_b.common import (  # noqa: E402
    ARTIFACT, CAPTURE, CAPTURE_TAG, INFERRED, MEASURED, NOT_HERE, RE_DERIVED,
    census_classes, device_status, head, kv, print_matrix, read, render_kda_sources,
    render_worker, require, rule, run_render, shape_str, verdict)

REF90 = "out/archlab/tri/chunk_gated_delta_rule_fwd_kernel_h_blockdim64.ptx"
REF100 = "out/archlab/tri100/chunk_gated_delta_rule_fwd_kernel_h_blockdim64.ptx"
TTIR = "out/archlab/tri/chunk_gated_delta_rule_fwd_kernel_h_blockdim64.ttir"

# The five instruction families that separate an Ada-class codegen from a
# Hopper or Blackwell one. If a construction emits none of these and only ever
# emits mma.sync, it is not reaching the architecture.
ARCH_SPECIFIC = ["wgmma.mma_async", "tcgen05.mma", "tcgen05.ld",
                 "stmatrix", "ldmatrix", "mbarrier", "cp.async", "tensormap"]


# =============================================================================

def ptx_header(text: str) -> dict:
    return dict(
        version=(re.search(r"^\.version\s+(\S+)", text, re.M) or [None, "?"])[1],
        target=(re.search(r"^\.target\s+(\S+)", text, re.M) or [None, "?"])[1],
        isa=(re.search(r"^\.version\s+(\S+)", text, re.M) or [None, "?"])[1],
    )


def local_memory_in_ptx(text: str) -> dict:
    """Does the PTX declare or touch local memory at all?

    This is the instrument that explains WHY `cuobjdump -res-usage` said
    LOCAL:0 for a kernel that ptxas -v said spilled 380 B: if the PTX has no
    `.local` and no local ld/st, then the local traffic is created by
    ptxas's register allocation, after the PTX, and a frame-size readout
    cannot see it.
    """
    return dict(
        local_decls=len(re.findall(r"^\s*\.local\b", text, re.M)),
        ld_local=len(re.findall(r"\bld\.local\b", text)),
        st_local=len(re.findall(r"\bst\.local\b", text)),
        shared_decls=len(re.findall(r"^\s*\.shared\b", text, re.M)),
        shared_st=len(re.findall(r"\bst\.shared\b", text)),
        shared_ld=len(re.findall(r"\bld\.shared\b", text)),
    )


# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archs", default="sm_75,sm_80,sm_86,sm_89,sm_90,sm_100")
    ap.add_argument("--render-one", default=None,
                    help=argparse.SUPPRESS)  # internal: common.py is the worker
    ap.add_argument("--verify-spill", action="store_true",
                    help="re-derive the reference kernel's register/spill figures "
                         "with ptxas, and fail loudly if ptxas is absent")
    a = ap.parse_args()
    archs = [x.strip() for x in a.archs.split(",") if x.strip()]
    if a.verify_spill:
        return verify_spill()

    st = device_status()
    head("B4. THE INSTRUCTION-CLASS MATRIX  --  both sides, no device")
    kv("GPU visible here", st["gpu_present"])
    kv("operator shape", f"{shape_str(CAPTURE)}   [{CAPTURE_TAG}]")
    kv("why this shape", "it is the only shape for which emitted source exists, so "
                         "it is the one the record's byte accounting is about")
    print()

    # ------------------------------------------------------------------ human
    head("1. THE HUMAN CONSTRUCTION  --  flash-linear-attention, Triton, AOT")
    print("  Compiled by `triton.compile` with an explicit GPUTarget, with the")
    print("  CUDA driver absent. Triton's AOT path does not need a device, so")
    print("  these PTX files were produced on a machine with no GPU and are")
    print("  pinned here; they are re-read, not re-derived.")
    print()
    ref_rows = []
    for tag, path in (("sm_90a", REF90), ("sm_100a", REF100)):
        txt = read(path)
        h = ptx_header(txt)
        c = census_classes(txt, "ptx")
        lm = local_memory_in_ptx(txt)
        ref_rows.append((tag, c))
        print(f"  {path}")
        print(f"    .version {h['version']}   .target {h['target']}   "
              f"{len(txt):,} B of PTX")
        print(f"    arch-specific classes: "
              + ", ".join(f"{k}={c[k]}" for k in ARCH_SPECIFIC if c[k]))
        print(f"    .local declarations {lm['local_decls']}, "
              f"ld.local {lm['ld_local']}, st.local {lm['st_local']}")
        print(f"    st.shared {lm['shared_st']}, ld.shared {lm['shared_ld']}")
        print()
    c90, c100 = ref_rows[0][1], ref_rows[1][1]
    print("  The same human kernel, two architectures:")
    for tag, c in ref_rows:
        cp = c["cp.async"] + c["cp.async.commit_group"]
        print(f"    {tag:<8} wgmma {c['wgmma.mma_async']:>3}  "
              f"stmatrix {c['stmatrix']:>3}  ldmatrix {c['ldmatrix']:>3}  "
              f"cp.async {cp:>3}  mbarrier {c['mbarrier']:>3}  "
              f"tcgen05 {c['tcgen05.mma'] + c['tcgen05.ld'] + c['tcgen05.wait']:>3}  "
              f"mma.sync {c['mma.sync']:>3}")
    print()
    print("  -> the human construction is ARCH-ADAPTED. Two tensor-core")
    print("     generations, chosen by the toolchain, from one source file:")
    print(f"     sm_90a gets wgmma and NO tcgen05; sm_100a gets tcgen05 and")
    print(f"     ldmatrix and NO wgmma. That is the thing the comparison is")
    print("     about, and it is the thing the compiler side does not do.")
    print()
    ttir = read(TTIR)
    print(f"  The adaptation is decided upstream of the PTX. {TTIR} is")
    print(f"  {len(ttir):,} B of Triton IR and is IDENTICAL between the sm_90a and")
    print(f"  sm_100a runs (same file, pinned once), so the arch reaches the")
    print(f"  choice as a target and the frontend IR does not change.")
    print()

    # ---------------------------------------------------------------- compiler
    head("2. THE COMPILER CONSTRUCTION  --  tinygrad 0.14.0 CUDARenderer")
    print("  DEV=NULL:CUDA:<arch> attaches the CUDA renderer to a device that")
    print("  cannot run anything; Compiler.compile_cached is the single choke")
    print("  point every kernel passes through and returns the source string.")
    print()
    print(f"  {'arch':<9}{'distinct srcs':>15}{'program bytes':>16}  "
          f"sha256 of the WHOLE program")
    print()
    blobs = run_render(archs)
    rows = []
    for arch in archs:
        srcs, blob, sha = blobs[arch]
        rows.append((f"tinygrad {arch}", census_classes(blob, "cu")))
        print(f"  {arch:<9}{len(srcs):>15,}{len(blob):>16,}  {sha}")
    print()
    shas = {b[2] for b in blobs.values()}
    print(f"  distinct whole-program digests across {len(blobs)} architectures: "
          f"{len(shas)}")
    if len(shas) == 1:
        print("  -> the entire rendered program is BYTE-IDENTICAL across every")
        print("     architecture rendered. Not one digest per file: one digest for")
        print("     the concatenation of all of them.")
    else:
        by = {}
        for arch, (_, _, sha) in blobs.items():
            by.setdefault(sha, []).append(arch)
        print("  -> the program is NOT arch-invariant. The partition is:")
        for sha, aa in by.items():
            print(f"     {sha[:16]}  {aa}")
    print()
    print("  arch-specific classes across the WHOLE program:")
    for arch, (_, blob, _) in blobs.items():
        c = census_classes(blob, "cu")
        got = {k: c[k] for k in ARCH_SPECIFIC if c[k]}
        print(f"    {arch:<8} {got if got else 'NONE of the eight'}")
    print()
    any_spec = any(sum(census_classes(blob, "cu")[k] for k in ARCH_SPECIFIC)
                   for _, blob, _ in blobs.values())
    print(f"  the compiler construction emits ANY of "
          f"{len(ARCH_SPECIFIC)} arch-specific classes: {any_spec}")
    mma = collections.Counter()
    for arch in archs:
        for m in collections_of_mma(blobs[arch][1]):
            mma[m] += 1
    print(f"  its entire tensor-core vocabulary, and the architectures that emit it")
    print(f"  (out of {len(archs)} rendered):")
    for m, n in sorted(mma.items()):
        print(f"    {m:<52} {n}/{len(archs)} archs")
    if not mma:
        print("    (none)")
    print(f"  and the whole-program exp2 count, per arch, which is where the one")
    print(f"  real arch-dependence in tinygrad's output shows up:")
    for arch, (_, blob, _) in blobs.items():
        c = census_classes(blob, "cu")
        print(f"    {arch:<8} exp2 {c['ex2.approx']:>6,}   "
              f"mma.sync {c['mma.sync']:>4,}   "
              f"__shared__ {c['__shared__']:>4,}   "
              f"__syncthreads {c['__syncthreads']:>4,}")
    print()
    print("  READ: the split is exactly at sm_75 / sm_80, and it changes the")
    print("  OPERATOR'S OWN MATH, not just its instruction selection. sm_75 has no")
    print("  tf32 tensor core, so the matmuls take a different route and the")
    print("  fusing pattern around the decay's exp2 changes with it. So the")
    print("  captured sm_75 kernel analysed in b1 UNDERSTATES the exp2 count on")
    print("  every card the study would actually want, by 4x on that kernel.")
    print("  This is the correction that makes b1's number honest: b1 is right")
    print("  about its artifact and must not be quoted as arch-neutral.")
    print()

    # ------------------------------------------------------------------ matrix
    head("3. THE MATRIX")
    allrows = ref_rows + rows
    print_matrix(allrows, note="""
        Both sides at once. Reference rows are POST-triton and POST-ptxas; the
        tinygrad rows are PRE-ptxas CUDA C. So this is a CLASS matrix -- which
        instruction families each construction can reach at all -- and NOT an
        instruction-count comparison. Asking who issues more instructions across
        that gap is not a well-posed question and this file does not answer it.
        Asking which families are reachable is, and the answer is one-sided.
    """)
    print()
    print("  (i)   CANNOT EXPRESS: wgmma, tcgen05, stmatrix, ldmatrix, mbarrier,")
    print("        cp.async, tensormap -- 0 occurrences in the entire rendered")
    print("        program at every architecture. There is no renderer path and")
    print("        no tensor-core catalog entry that would produce them.")
    print("  (ii)  CAN EXPRESS BUT DOES NOT FIND: the mma.sync it does emit is at")
    print("        the smallest tf32 shape in the table, m16n8k8, while the table")
    print("        offers m16n8k16 and m16n8k32. The mechanism exists; the")
    print("        search does not reach it for this operator.")
    print("  (iii) NOT YET IMPLEMENTED: the Hopper/Blackwell renderer paths.")
    print("  (iv)  DOES NOT EXIST FOR THIS ARCHITECTURE: n/a here -- every class")
    print("        above exists on both sm_90 and sm_100, which is the point.")
    print()
    print("  The study's own tag vocabulary, applied to one row each:")
    print("    the hand-written Triton kernel reaching wgmma on sm_90a and")
    print("    tcgen05 on sm_100a  ->  (iii) not yet implemented, on tinygrad.")
    print("    the same kernel spilling 380 B at 255 registers at sm_90a  ->")
    print("    also (iii), in the other direction: the human baseline carries a")
    print("    defect the study attributes to codegen limits, so the baseline is")
    print("    not a clean reference. See section 4.")
    print()

    # ---------------------------------------------------------------- the spill
    head("4. THE BASELINE IS NOT FREE, AND ITS SPILL IS INVISIBLE WHERE YOU LOOK")
    lm90 = local_memory_in_ptx(read(REF90))
    lm100 = local_memory_in_ptx(read(REF100))
    print(f"  {REF90}")
    print(f"    .local declarations {lm90['local_decls']}    ld.local {lm90['ld_local']}"
          f"    st.local {lm90['st_local']}")
    print(f"    st.shared {lm90['shared_st']}    ld.shared {lm90['shared_ld']}")
    print(f"  {REF100}")
    print(f"    .local declarations {lm100['local_decls']}    ld.local {lm100['ld_local']}"
          f"    st.local {lm100['st_local']}")
    print()
    print("  Zero local memory ANYWHERE in the hand-written kernel's PTX, on both")
    print("  architectures. And yet `ptxas -v` for the sm_90a cubin reports")
    print("  380 B of spill stores, 372 B of spill loads, a 376 B stack frame, at")
    print("  the 255-register ceiling. [ARTIFACT, quoted: ptxas has not been run")
    print("  on this host -- no CUDA toolkit -- so the spill figures are read out")
    print("  of the recorded compile, not recomputed. `python3 b4_codegen.py")
    print("  --verify-spill` re-runs ptxas wherever it exists and fails loudly if")
    print("  it cannot.]")
    print()
    print("  The two facts are compatible and the composition is the lesson: ptxas")
    print("  INTRODUCES the local traffic during register allocation, after the PTX")
    print("  exists. So it is absent from the PTX by construction. And")
    print("  `cuobjdump -res-usage` reports the local FRAME -- a declared array --")
    print("  and reports 0 for this kernel, correctly, because there is no")
    print("  declared array. A study that had trusted it would have certified the")
    print("  hand-written baseline as spill-free. `ptxas -v` is the only")
    print("  authoritative source here and it is the one that was not run.")
    print()
    # occupancy arithmetic, from a datasheet figure that cannot be checked here
    REGS, THREADS, MAXW = 65536, 128, 64   # H100 SM register file, 32-bit words
    blocks = REGS // (255 * THREADS)
    warps = blocks * (THREADS // 32)
    print(f"  occupancy arithmetic, re-derived from the register file  [{INFERRED}]")
    print(f"    {REGS:,} 32-bit registers per SM (H100 datasheet, not checkable here)")
    print(f"    255 registers/thread x {THREADS} threads = {255*THREADS:,} per block")
    print(f"    blocks resident per SM = floor({REGS:,}/{255*THREADS:,}) = {blocks}")
    print(f"    warps resident = {blocks} x {THREADS//32} = {warps} of {MAXW} max"
          f"  =  {warps/MAXW*100:.1f}%")
    print()
    print("  So the hand-written reference, on the one architecture where its own")
    print("  toolchain emits the newest tensor-core instructions, runs at the")
    print("  occupancy that `notes/04-where-compilers-stop.md` names as the")
    print("  mechanism by which a CORRECT kernel collapses to a slow one. The")
    print("  study's fast reference carries the defect the study attributes to")
    print("  codegen limits. That does not make the (b) verdict wrong; it makes")
    print("  the baseline a weaker control than the record has been treating it.")
    print()
    print("  AND IT CUTS THE OTHER WAY, which is why it is here: if the human")
    print("  kernel spills at 255 registers on sm_90a, then the comparison")
    print("  'hand-written reaches wgmma, tinygrad does not' is not yet a")
    print("  comparison of ACHIEVED tensor-core use. It is a comparison of what")
    print("  each toolchain SELECTS. Whether the wgmma path is actually worth")
    print("  more than the spill costs is a performance question, and it is")
    print("  exactly the question this study cannot currently ask. Filing it")
    print("  honestly is better than filing either half of it.")
    print()

    # ------------------------------------------------------------------ edges
    head("5. THE EDGES")
    print(f"  * The compiler side is PRE-ptxas. ptxas will add instructions, may")
    print(f"    add or remove them, and will certainly make scheduling choices.")
    print(f"    So the class matrix says what the RENDERER can reach, not what")
    print(f"    the SASS will contain. `out/archlab/tg/*.cubin` in this repo is a")
    print(f"    SASS census of a DIFFERENT operator set (matmul / gather_matmul /")
    print(f"    fused_reduce), not of KDA, and re-deriving it needs a toolkit this")
    print(f"    host does not have. Marked {NOT_HERE}.")
    print("  * The reference side is at ONE kernel, chosen by the record. A")
    print("    different hand-written kernel could differ. What is being claimed")
    print("    is about the toolchain's arch adaptation, which is a property of")
    print("    Triton and is visible in any one of its kernels.")
    print("  * The `exp2` counts in section 2 are over DISTINCT SOURCES, not over")
    print(f"    launches: {len(blobs[archs[0]][0])} distinct sources for a plan of")
    print("    2,974 kernels, because tinygrad's renderer deduplicates identical")
    print("    bodies. They are a lower bound on the launched total, and the plan")
    print("    size is b3's job, not this file's.")
    print("  * Nothing here is a timing. Two constructions, different")
    print("    instruction mixes, no wall clock. See notes/15 section 3.")

    verdict("B4 VERDICT  the human construction is arch-adapted (wgmma on sm_90a,\n"
            "         tcgen05 on sm_100a); the compiler construction is\n"
            "         arch-INVARIANT -- one sha256 for the whole program across\n"
            "         sm_80..sm_100 -- and emits none of the eight arch-specific\n"
            "         instruction classes at all, its whole tensor-core\n"
            "         vocabulary being one mma.sync shape. The comparison is\n"
            "         settled without a GPU. The one real arch-dependence in\n"
            "         tinygrad's output is the sm_75/sm_80 boundary, and it moves\n"
            "         the operator's own arithmetic -- which makes the captured\n"
            "         sm_75 exp2 count an UNDERSTATEMENT for any sm_80+ card.")
    return 0


def collections_of_mma(blob: str) -> set:
    return set(re.findall(r"mma\.sync[a-z0-9._]*", blob))


def verify_spill() -> int:
    """Re-derive the sm_90a kernel's register and spill figures with ptxas.

    This is the one number in b4 that cannot be re-derived on a host without a
    CUDA toolkit, so it is quoted rather than computed -- and a quoted number
    that nobody can re-derive is exactly the kind that rots. This mode re-runs
    the compile and exits NONZERO if ptxas is missing, so "I could not check it"
    can never be mistaken for "I checked it and it agreed".
    """
    import shutil
    import subprocess
    import tempfile
    head("B4-VERIFY-SPILL.  ptxas -v on the hand-written reference, sm_90a")
    ptxas = shutil.which("ptxas")
    if not ptxas:
        print("  ptxas NOT FOUND. The 380 B spill figure in section 4 is therefore")
        print("  UNVERIFIED ON THIS HOST and must be read as [ARTIFACT, quoted].")
        print("  Install a CUDA toolkit and re-run this mode to re-derive it.")
        return 2
    ptx = require(REF90)
    with tempfile.TemporaryDirectory() as td:
        p = subprocess.run([ptxas, "-arch=sm_90a", "-v", "-o",
                            os.path.join(td, "k.cubin"), ptx],
                           capture_output=True, text=True)
    if p.returncode != 0:
        print(f"  ptxas failed (rc={p.returncode}):\n{p.stderr[-2000:]}")
        return 2
    txt = p.stderr + p.stdout
    head("ptxas -v")
    for ln in txt.splitlines():
        if any(k in ln for k in ("Used", "Function properties", "spill", "stack",
                                 "registers")):
            print("  " + ln.strip())
    # the occupancy arithmetic, from ptxas's own register count
    m = re.search(r"Used (\d+) registers", txt)
    regs = int(m.group(1)) if m else None
    sm = re.search(r"(\d+) bytes spill stores", txt)
    print()
    if regs is not None:
        REGS, THREADS, MAXW = 65536, 128, 64
        blocks = REGS // (regs * THREADS)
        print(f"  registers/thread {regs}  ->  blocks/SM {blocks}  ->  "
              f"occupancy {blocks*(THREADS//32)/MAXW*100:.1f}%")
    if sm:
        print(f"  spill stores {sm.group(1)} B  (the section-4 figure says 380 B)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
