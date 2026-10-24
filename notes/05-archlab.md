# archlab — inspecting generated GPU code with no GPU

The free-tier hardware for this study is a T4 (sm_75) and a P100 (sm_60). On
both of those, essentially every modern feature is dark: no WGMMA, no TMA, no
tcgen05, no warp specialization, no `cp.async` pipeline. So the newest
hand-written kernels — the ones the literature says close the gap against a
DSL — **cannot be measured on the hardware I have**. That was the thing blocking
the study.

This is the move that unblocks it: **ptxas, nvcc and Triton are compilers, and
compilers do not need the hardware.** They will target any architecture
offline. So I can compile the Hopper and Blackwell code I cannot run, and
measure the *shape* of what the compiler produced.

Everything below was produced on a Mac arm64 host with **no CUDA, no GPU and no
NVIDIA driver**, inside `linux/amd64` containers under OrbStack.

---

## 1. Headline: it works

| question | answer |
|---|---|
| Can ptxas assemble a Hopper-only WGMMA kernel with no GPU? | **yes** |
| Does the WGMMA survive into SASS? | **yes**, as `HGMMA.64` |
| Can TinyGrad's generated CUDA C be produced and compiled offline? | **yes** |
| Can a real FLA Triton kernel be AOT-compiled offline? | **yes** |
| Does Triton publish linux/amd64 wheels? | **yes** (3.8.0, 248 MB) |
| Can the Triton→SASS loop be closed for sm_90a? | **yes** |
| …for sm_100a? | **no**, and the reason is specific (see §5) |

The go/no-go passed on the first real attempt, and the WGMMA came through
intact:

```
ptxas -arch=sm_90a -v probe.ptx -o probe.cubin
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
    Used 30 registers, used 0 barriers
```

```
HGMMA.64x8x16.F32.BF16 R24, gdesc[UR4].negA, R24, !UP0, gsb0 ;
WARPGROUP.ARRIVE ;                    <- wgmma.wait_group
WARPGROUP.DEPBAR.LE gsb0, 0x0 ;       <- wgmma.fence
LDGSTS.E.BYPASS.128 [R9], desc[UR14][R4.64] ;  <- cp.async, 128-bit (x2)
SYNCS.PHASECHK.TRANS64.TRYWAIT        <- mbarrier try_wait loop
STG.E.128                             <- vectorised global store
```

The `HGMMA` line is worth reading rather than skimming: `64x8x16` is the matrix
shape, `F32.BF16` the accumulate and input types, `gdesc[UR4].negA` the A
descriptor with `imm-scale-a = -1`, `gsb0` the scale-B operand. ptxas decoded
the shape, the dtypes *and* the scale operands correctly. That is a stronger
statement than "it compiled" — the instruction was assembled as written, not
merely accepted.

Every Hopper-specific construct in the source is present in the SASS. That is
the thing I needed to know.

---

## 2. Getting a toolchain without the 4 GB image

`nvidia/cuda:12.6.0-devel-ubuntu24.04` timed out at 10 minutes. It is also the
wrong tool: it carries cudnn, cublas, NCCL and the samples, none of which a
code-*shape* inspection needs. Two much smaller routes work:

| image | size | how | what you get |
|---|---|---|---|
| `archlab/ptxas:12.9` | **232 MB** | `pip install nvidia-cuda-nvcc-cu12` on `python:3.12-slim-bookworm` | `ptxas` only |
| `archlab/nvcc:12.9` | **811 MB** | three apt components off NVIDIA's repo on `ubuntu:24.04` | `nvcc`, `ptxas`, `cuobjdump`, `nvdisasm` |
| `archlab/compile:12.9` | 1.85 GB | the above + `triton`, `tinygrad`, `fla` | everything, incl. task 4 |

Two things cost me time and are worth recording:

**The `nvidia-cuda-nvcc-cu12` wheel contains only `ptxas`.** Despite the name,
it ships `ptxas` and the nvvm/libdevice, and *not* `nvcc`, `cuobjdump` or
`cudafe++`. For SASS I needed NVIDIA's apt repo anyway, split across three
packages: `cuda-nvcc`, `cuda-cuobjdump` and `cuda-nvdisasm`. The last is
non-obvious — `cuobjdump -sass` shells out to `nvdisasm` and fails with
`Could not find executable file 'nvdisasm'` if it is missing.

**`ptxas -v` writes its report to stderr, not stdout.** Any wrapper that reads
only stdout silently reports zero registers and zero spills. `inspect_ptx.py`
merges the streams; this one cost an hour of "why is REG=None".

**x86 emulation needs a raised pip timeout.** The default read timeout is not
enough for a 40 MB wheel, let alone Triton's 248 MB. `--timeout 120 --retries 10`
is the floor.

---

## 3. What the instrument measures

`tools/archlab/inspect_ptx.py` — given a `.ptx` and an arch, runs `ptxas -v`
and `cuobjdump -sass` in the container and reports.

**Exact, straight from the toolchain:** registers/thread, static shared memory
per block, stack frame, spill stores, spill loads, barrier count; the full SASS
histogram; and counts in fixed categories — warpgroup MMA, tcgen05 MMA,
per-warp `mma.sync`, `cp.async`, TMA/bulk copy, shared ld/st, global ld/st with
their **widths**, scalar FFMA, warpgroup sync, block barriers, mbarrier ops,
and local/spill traffic.

Several of those category patterns were wrong until I checked them against
real SASS rather than assuming. `cp.async.bulk` is `UBLKCP`, not `UTMALDG`.
And `^LDG` silently swallows `LDGSTS`, which double-counted the async copies as
global loads; it is now `^LDG\.`.

**The demo: a contrast pair.** Same tile, same shared-memory budget, same
64×8×16 bf16 shape, 128 threads, both for sm_90a — one using the Hopper WGMMA
path, one doing the classic scalar staging + FFMA k-loop:

| | `probe.ptx` (WGMMA) | `naive.ptx` (naive loop) |
|---|---|---|
| registers / thread | 30 | 31 |
| spill stores / loads | 0 / 0 | 0 / 0 |
| **warpgroup MMA** | **1** | **0** |
| **scalar FFMA** | **0** | **64** |
| **shared loads** | 3 | **48** (all `LDS.U16`, 16-bit) |
| shared stores | 0 | 9 (`STS.U16`) |
| **global load widths** | **128-bit ×2** (`LDGSTS`) | **scalar ×9** (`LDG.E.U16`, 2 bytes) |
| `cp.async` | 2 | 0 |
| block barriers | 0 | 1 |
| SASS instructions | 80 | 304 |

The instrument separates the two kernels on every axis the study cares about,
including the one I would have guessed wrong: the naive kernel's *shared memory
traffic* is 48 scalar 16-bit loads against the WGMMA kernel's 3, because each
thread re-reads six shared words per k-step to feed eight FMAs. On a T4 the
naive kernel is not "the same computation, slower" — it is a different memory
access pattern, and that is visible statically, on a machine with no T4.

---

## 4. The other two producers, offline

### TinyGrad's generated CUDA C → `nvcc -arch=sm_90a -cubin` : **works**

TinyGrad's code generator is a pure function of the UOp graph, so it needs no
device. Three pieces make it run on a GPU-less box:

1. `DEV=NULL:CUDA:sm_90` — the NULL device initialises with no hardware but
   accepts an alternative renderer, so you get the allocator and scheduler with
   the CUDA *renderer* attached.
2. `CUDARenderer(use_nvcc=True)` — by default it wants NVRTC, which `dlopen`s
   `libnvrtc` and fails. Forcing the nvcc driver makes it shell out instead.
3. `Compiler.compile_cached` — NULL's default compiler returns the source string
   verbatim as its "output", so capturing it yields the generated CUDA C.

One trap worth flagging: **`PARALLEL=0` is mandatory.** TinyGrad's beam search
and TC autotuner run candidate compiles in a `multiprocessing` pool built with a
*spawn* context. The workers re-import tinygrad, never see your monkeypatches,
die constructing `CUDARenderer` (no NVRTC), and the pool hangs forever.
`PARALLEL=0` makes `get_worker_pool()` return `None` and everything runs
in-process. The autotune wrapper is also exactly the part that *cannot* work
offline, since it has to benchmark configs on a real device — so a kernel that
relied on autotuning is only measurable at the config level, not as chosen.

Results for sm_90a, all compiled offline:

| op | kernel | regs | SASS | notes |
|---|---|---|---|---|
| `matmul` | 6 kernels | 10–128 | 24–3512 | largest: 16 `mma.sync`, 588 FFMA |
| `gather_matmul` | 7 kernels | 10–119 | 24–1656 | largest: 16 `mma.sync`, 40 **scalar** global loads, no 128-bit |
| `fused_reduce` | 5 kernels | 10–28 | 24–1552 | largest: 1088 B smem, 128-bit vectorised |

The gathered matmul failing to vectorise is the RC1 shape from the prior-work
notes, visible with no GPU. I am recording it as *what the instrument shows*,
not as a verdict on TinyGrad: the shapes here are small and one gather is not a
study.

### FLA's Triton kernel → `triton.compile` AOT : **works**, with caveats

Triton does publish linux/amd64 wheels (3.8.0, 248 MB, cp312), and it installed
and imported cleanly. AOT compile needs `target=` — without it Triton asks the
driver what device it is on, and there is no device. With an explicit
`GPUTarget`, the whole pipeline runs in-process.

Compiling FLA's real `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` at
`GPUTarget("cuda", 90, 32)`, with the driver verifiably absent:

```
cuda driver: ABSENT (RuntimeError: 0 active drivers ([]))
RESULT: AOT COMPILE OK
  artifacts: ['cubin', 'llir', 'ptx', 'source', 'ttgir', 'ttir']
  ptx header: ['.version 8.8', '.target sm_90a']
  SASS: 2632 instructions, REG=255 spill=380B st/372B ld, stack=376B
    sass warpgroup_mma    16
    sass cp_async          4
    sass shared_ld        76
    sass shared_st       188
    sass local_ldst      188     <- real spilling
    global load widths  {'scalar': 168}
```

**So AOT compile does not need a GPU present.** Triton emits genuine `.target
sm_90a` PTX and a cubin with no driver in sight, and it does reach WGMMA (16
`HGMMA`) plus `cp.async` — the Hopper path, offline.

**And the first result the study actually wants fell out of it.** This is FLA's
own kernel, the hand-written baseline the repo treats as the fast reference, and
at sm_90a it **spills**: 380 bytes of spill stores, 372 of spill loads, 376 of
stack frame, at the 255-register ceiling. 62 `LDL` + 31 `LDL.LU` + 95 `STL` in
the SASS. At 255 registers with 128-thread blocks, occupancy is 2 blocks/SM =
8 warps of 64 = **12.5%**.

That is the RC3 "register spill collapsing occupancy" mechanism from the
prior-work paper, reproduced in a leading hand-written Triton kernel, on
hardware nobody in this study owns. It also says something uncomfortable about
the study's own baseline: the comparison target is not free of the defect the
paper attributes to codegen limits.

A methodological warning that came out of the same run: **`cuobjdump
-res-usage` reported `LOCAL:0` for this kernel.** It reports the local *frame*
size, not spill traffic. Had I trusted it I would have reported a
hand-written kernel as spill-free. `ptxas -v` is the only authoritative source
here, and its `SHARED` field is equally unreliable for the dynamic shared memory
Triton always emits. The instrument therefore parses `ptxas -v`, and treats
`-res-usage` as a cross-check only.

---

## 5. Where it stops — the honest edges

**sm_100 needs a newer toolkit than I could justify here.** Triton at
`cc=100` compiles fine offline and emits `.target sm_100a`. But it emits
**PTX ISA 9.3**, and ptxas 12.9 says:

```
fatal : Unsupported .version 9.3; current version is '8.8'
```

So for sm_100 the loop closes Triton→PTX and stops. Getting SASS needs CUDA
13.x. sm_90a is fully covered by 12.9. Hand-written `tcgen05.alloc` and
`tcgen05.wait` do compile for sm_100a under 12.9, but I could not get
`tcgen05.mma` to assemble — the `{ disable-output-lane }` vector width is
unresolved and the ISA spec is not clear about it. I left the
`tcgen05_mma` category in the instrument as `^TCGMMA` with a comment saying the
mnemonic is **unverified** in this environment. I would rather ship an
unverified pattern that shows 0 plus a comment than a confident one.

**FLA's published package is broken.** `flash-linear-attention` 0.5.2 — *both
the wheel and the sdist* — ships only `fla/layers` and `fla/models`. `fla/ops`,
where every Triton kernel lives, is absent from both. I vendored the kernel
from GitHub (`fla/ops/common/chunk_delta_h.py`, MIT) with provenance in
`out/archlab/triton/fla_kda_chunk_h.py`. Anything that assumes
`pip install flash-linear-attention` gives you the kernels is wrong today.

**Occupancy is an estimate, and the tool says so.** It needs a block size, which
the PTX does not carry unless it pins one with `.reqntid`/`.maxntid`; the tool
refuses to invent one and prints `CANNOT DETERMINE` instead. When it does
compute a figure, the per-SM limits come from a transcribed table in
`ARCH_LIMITS`, not from the toolchain, so the number inherits my transcription.

**Dynamic shared memory is not measured at all.** `probe.ptx` and `naive.ptx`
use `extern .shared`, and ptxas does not report dynamic shared memory at all —
it is a launch-time parameter. The instrument reports the *declared* size and
flags that occupancy is then an upper bound.

---

## 6. What this can and cannot answer

**Can:**
- confirm that a Hopper/Blackwell instruction is actually reachable, survived
  optimisation, and reached SASS — the thing that is otherwise unverifiable
  without owning a Hopper
- count tensor-core vs scalar-FMA work, and see the *balance* between them
- measure register pressure and spilling exactly (`ptxas -v`), and therefore
  compute an occupancy estimate
- see vectorisation width on global and shared traffic, which is the RC1
  question, statically
- compare two implementations of the same operator on code shape alone, with a
  fixed and auditable definition of "shape"
- read TinyGrad's and Triton's real codegen for an arch I do not own, and check
  both against ptxas

**Cannot:**
- say anything about **correctness**. ptxas validates types, control flow and
  alignment — never fragment layout or data mapping. `probe.ptx` assembles
  cleanly and its WGMMA fragment mapping is nominal. A clean compile is not a
  correct kernel, and per the prior-work notes correctness is not even evidence
  of replacement quality.
- measure **anything temporal**: no wall time, no achieved occupancy, no DRAM or
  L2 traffic, no bank conflicts, no latency hiding, no divergence. Every one of
  those needs a clock and a memory system.
- substitute for the library-relative-efficiency or roofline check. Code shape
  predicts the *ceiling*, not the achieved rate. A kernel with perfect
  instruction mix and 12.5% occupancy can still lose to one that looks worse.
- validate the autotuning question. Triton's autotune and TinyGrad's beam search
  both need to *run* candidates. Offline I can only inspect a specific config,
  never the one the tuner would have chosen — which is RC2, and RC2 is exactly
  the class this instrument is blind to.

The blunt summary: **this is a static-shape instrument, and its most valuable
output so far has been negative.** It found real register spilling in the
hand-written kernel the study uses as its reference, and it could not tell me
whether that spilling costs anything, because that still needs the hardware.
What it buys is that the *next* free-tier run is no longer the first time anyone
looks at this code, and that the sm_90a question stops being unanswerable.

---

## 7. Files

| path | what |
|---|---|
| `tools/archlab/inspect_ptx.py` | the instrument: `.ptx` + arch → resources, occupancy estimate, SASS categories |
| `tools/archlab/tinygrad_cuda.py` | TinyGrad CUDA C codegen on a GPU-less host, compiled with nvcc |
| `tools/archlab/triton_aot.py` | Triton AOT compile at an explicit `GPUTarget` |
| `tools/archlab/build_image.sh` | builds the three images; documents the 4 GB → 232 MB argument |
| `tools/archlab/demo.sh` | the whole thing end to end |
| `tools/archlab/test_inspect.py` | 30 self-checks on the parsers and the occupancy model; no GPU or network needed |
| `tools/archlab/fla_kda_chunk_h_sig.json` | the FLA kernel's signature + constexprs for AOT |
| `out/archlab/probe.ptx` | the Hopper WGMMA probe |
| `out/archlab/naive.ptx` | the contrast kernel (same tile, no WGMMA) |
| `out/archlab/triton/fla_kda_chunk_h.py` | vendored FLA kernel extract, with provenance |
| `out/archlab/demo_output.txt` | captured output of the demo above, so the numbers can be audited |

```
bash tools/archlab/build_image.sh     # once: 232 MB / 811 MB / 1.85 GB
python3 tools/archlab/test_inspect.py # 30 self-checks, no GPU or network
bash tools/archlab/demo.sh            # everything above, no GPU needed
```

Two things about the numbers above that are worth stating plainly, because both
were caught by checking rather than by assuming.

**Occupancy is arithmetic, and I got it wrong once.** My first test asserted
that 128 registers at 256-thread blocks would reach 100% occupancy on sm_90a.
It reaches 25%: 128 × 256 = 32768 registers per block, and the H100 register
file is 65536, so exactly two blocks fit. The instrument was right and the
expectation was wrong. The lesson generalises — *few registers is not the same
as high occupancy* — and it is why the tool names the binding constraint rather
than printing a bare percentage. `test_inspect.py` now pins all of these cases,
including the ones where the tool must refuse to answer.

**`ptxas -v` goes to stderr.** A wrapper that reads only stdout reports zero
registers and zero spills and looks like it worked. Both tools merge the
streams, and there is a regression test for it.
