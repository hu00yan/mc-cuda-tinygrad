# TPU / Pallas — research and design

Owner: this file and `kda/tpu/`. Everything here was checked on 2026-09-27.
Sources are inline with URLs; anything I could not verify from a primary source
is marked **UNVERIFIED**.

**Nothing in `kda/tpu/` has been run on a TPU.** There is no TPU reachable from
this environment and the repo venv has no `jax` (verified: `import jax` →
`ModuleNotFoundError`). The two `.py` files that need JAX are written but not
executed. Everything that *can* run on CPU does, and does:

```bash
cd /Users/huyan00/mycode/mc-cuda-tinygrad
.venv/bin/python kda/tpu/roofline.py      # per-generation peaks + ridge point
.venv/bin/python kda/tpu/shapes.py        # the 3 sides, materialisation, floors
.venv/bin/python kda/tpu/tiling_check.py  # Pallas legality rules, with sources
.venv/bin/python kda/tpu/test_design.py   # 16/16: rewrite exactness, C=128, grid
```

---

## 1. Three corrections to the brief, before anything else

### 1a. tinygrad has no TPU backend. There is no `ops_tpu.py`.

The brief said "tinygrad HAS a TPU backend that lowers through XLA (check
`tinygrad/runtime/ops_tpu.py` in the installed venv at `.venv` — the tinygrad
0.14.0 source is there, read it)".

I read it. **There is no such file, and there never was.**

- `.venv/lib/python3.12/site-packages/tinygrad/runtime/` contains: `ops_amd.py`,
  `ops_cl.py`, `ops_cpu.py`, `ops_cuda.py`, `ops_disk.py`, `ops_dsp.py`,
  `ops_hip.py`, `ops_metal.py`, `ops_npy.py`, `ops_null.py`, `ops_nv.py`,
  `ops_python.py`, `ops_qcom.py`, `ops_rdma.py`, `ops_tinyfs.py`, `ops_webgpu.py`.
  No `ops_tpu.py`.
- `grep -rn -i "tpu" tinygrad/**/*.py` returns **only false positives** — every
  hit is the substring in "out**put**", "o**utput**s", "o**utpu**t",
  "o**utp**ut", "T**PU**t". Zero real matches.
- The documented runtimes (`https://docs.tinygrad.org/runtime/`) are NV, AMD,
  QCOM, METAL, CUDA, CL, CPU, WEBGPU. TPU is not among them.
- `GET api.github.com/repos/tinygrad/tinygrad/commits?path=tinygrad/runtime/ops_tpu.py`
  returns `[]` — an empty array, i.e. the file has never existed in the repo's
  history.

**Consequence for the study.** The "same operator in TinyGrad vs hand-written"
comparison *cannot* be run on TPU, because TinyGrad cannot target TPU. This is
not a gap to work around; it is a finding in its own right, and it is arguably
the most interesting architecture-level fact in this whole exercise:

> **On CUDA, a 14k-line compiler that emits its own kernels can at least be
> pointed at the hardware. On TPU, the only route to the hardware runs through
> a compiler (XLA) whose tile-level control the user cannot reach, so a
> framework that wants tile control has to adopt a *second* language (Pallas)
> whose semantics are not the framework's.**

That is the CUDA/TPU asymmetry the study is actually about, stated in one
sentence, and it is the reason TPU is a good second architecture: on CUDA the
taxonomy's "can a compiler infer this?" question is asked of Triton vs tinygrad,
two compilers in the same room. On TPU it is asked of XLA vs Mosaic-through-Pallas,
where the boundary is a *language boundary*, not a search budget.

So the TPU comparison is **XLA-compiled-from-JAX vs a hand-written Pallas kernel**,
and TinyGrad drops out. The "TinyGrad materialises 67 MB of intermediates"
finding from `notes/03-kda.md` does not transfer as a TinyGrad claim, but the
*structural* fact behind it — a `[C,C,K]` tensor that a graph-level semantic
choice has already committed to — is architecture-independent and does transfer.

### 1b. Kaggle's TPU v3-8 is deprecated. The current accelerator is v5e-8.

- Kaggle's own announcement, "New TPU v5e-8s! Phasing out v3-8s":
  "TPU v3-8s will be phased out slowly, and sessions that currently use v3-8s
  will automatically switch to v5e-8 when they are no longer available."
  Comment from a Kaggle staffer: "The v3 hardware is no longer available to us so
  we can only support the newer hardware."
- Kaggle's Q&A index lists "TPU v3-8 (Deprecated): This accelerator is
  deprecated and is no longer actively supported on Kaggle."
- Quota, from `https://www.kaggle.com/docs/tpu` (the official page): **"You can
  use up to 20 hours per week of TPUs and up to 9h at a time in a single
  session."** So the brief's "roughly 20–30 h/week" is 20 h/week, and the
  per-session cap is 9 h. There is also a third-party page claiming
  "30 hours per week"; I did not find that on Kaggle and am not using it.
- **Corroborated in this repo, first-party.** Another agent's Kaggle run
  (`out/kaggle/tpu-facts-v2/`, not mine) came back with
  `jax_sees_tpu: false`, `jax_devices: ["cpu x1"]`, and
  `ls: cannot access '/dev/accel*': No such file or directory`. So the practical
  situation is worse than "v3-8 is being phased out": a TPU session is
  **not reliably obtainable at all**. The v5e-8 comments on the announcement
  thread are consistent — "TPU v5e-8 is live, but currently oversubscribed,
  quota-limited, or temporarily disabled", and "Utilization is not currently
  available for TPU VMs" from multiple users. **Any plan that assumes a TPU
  appears on demand is a plan that will not run.** This raises the value of the
  CPU-first design in `kda/tpu/` considerably: `cost_analysis()` (M1) and the
  tiling checker need no TPU at all, and they are the two measurements that
  decide whether a session is worth spending.

**The v3-8 hardware facts are still correct**, and the "128 GB HBM" number is
right, but the arithmetic matters: Google's TPU v3 page gives **per chip** 123
TFLOP/s bf16 and 32 GiB HBM at 900 GB/s, and states "Each v3 TPU chip contains two
TensorCores. Each TensorCore has two matrix-multiply units (MXUs), a vector unit,
and a scalar unit." v3-8 is 4 chips, so 128 GiB and 3.6 TB/s. The 8 that JAX
exposes are 8 TensorCores = 4 chips × 2. Confusing "cores" with "chips" here
silently doubles or halves every roofline fraction, which is why
`kda/tpu/roofline.py` carries `chips` and `tcs_per_chip` separately and
`jax.device_count()` is documented as returning TensorCores.

**Design consequence:** every reported number is a fraction of the roofline for
the generation it was measured on, and `roofline.py` is the only place a peak
appears. A v3-8 result from a notebook run last month and a v5e-8 result from
next month are directly comparable; two raw millisecond figures are not.

### 1c. XLA's whole-graph fusion is much stronger than the brief assumes, and
### that cuts the other way.

The brief's premise is that TPU isolates the "what must a human supply"
question because the confounds disappear. The confounds *do* disappear. But
XLA's fusion is genuinely excellent, and I verified from source what it does
with attention:

`jax/_src/nn/functions.py::_dot_product_attention_xla` (fetched from
`raw.githubusercontent.com/jax-ml/jax/main`) builds

```
logits = einsum("BTNH,BSNH->BNTS", q, k)      # materialises [B,N,T,S]
logits *= scale ; logits = where(mask, logits, large_neg)
probs   = softmax(logits.astype(f32)).astype(key.dtype)
encoded = einsum("BNTS,BSNH->BTNH", probs, v)
```

There is **no online softmax, no tiling, no loop, and no custom call** on the
XLA path. The `cudnn` implementation is a different branch and is GPU-only.
What makes it fast is that XLA fuses the entire O(n²) expression into one
kernel that keeps the intermediate in VMEM — and it can, because TPU VMEM is
enormous. Measured, by Archer Zhang on a Colab TPU v5e (single chip), JAX 0.7.2,
single head `d=64`, float32:

| n | scores (fp32) | fits VMEM? | XLA fused (ms) | hand-written flash in JAX (ms) |
|---|---|---|---|---|
| 512 | 1.0 MB | yes | 0.076 | 0.082 |
| 1024 | 4.0 MB | yes | 0.055 | 0.133 |
| 2048 | 16.0 MB | yes | 0.067 | 0.522 |
| 4096 | 64.0 MB | yes | 0.072 | 2.509 |
| 8192 | 256.0 MB | **no** | 1.189 | 14.052 |
| 16384 | 1024.0 MB | **no** | 4.445 | 89.567 |
| 32768 | 4096.0 MB | **no** | 17.123 | 103.016 |

*(`https://archerzhang.me/forcing-flash-attention-onto-a-tpu`, 6 Mar 2026.
I reproduce the table; I did not run it.)*

So the hand-written tiling is **35× slower at n=4096** and only overtakes at
n ≥ 8192, i.e. exactly when the materialised score matrix no longer fits on
chip. Below that threshold XLA's fusion is not a weak baseline; it is nearly
optimal, and the human's tiling is a downgrade.

**This is the single most important thing to internalise before designing the
TPU experiment**, because it means the TPU headline is *not* "hand-written beats
compiled". It is:

> XLA is very good at not moving bytes it does not have to move, and cannot do
> anything about bytes it must. The human's job on TPU is therefore narrower
> and lower than on CUDA: it is *not* fusion, and it is *not* tiling for its own
> sake. It is the algorithm, the DMA pipeline, and the memory layout.

Which is a *sharper* question, and one where "XLA cannot express tile-level
control" has a much sharper answer than on CUDA, because on CUDA that claim is
contested by Triton (which does expose tile-level control) whereas on TPU it is
a plain fact about where Pallas sits in the pipeline.

---

## 2. Is Pallas really the "you must hand-specify everything" counterpoint?

**Verdict: yes, and stronger than the brief claims — but not for the reason the
brief gives, and with one important qualification.**

### 2a. What is verified, from the actual API

From `https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html` and
`.../tpu/details.html` and `.../pallas/grid_blockspec.html` (fetched 2026-09-27):

A Pallas TPU kernel is called with mutable `Ref`s, and the programmer supplies:

| what the programmer must state | the API | quote / source |
|---|---|---|
| the grid | `grid=` / `grid_spec=` | `grid=(num,)`; "The kernel is executed as many times as `prod(grid)`" |
| the block shape per operand | `pl.BlockSpec(block_shape, index_map)` | "the last two dimensions of your block shape must be equal to the respective dimension of the overall array, or be divisible by 8 and 128 respectively" |
| the grid→slice mapping | `index_map` | takes the invocation indices, returns **block** indices (or element indices under `pl.Element`) |
| the memory space of every operand | `memory_space=pltpu.VMEM / SMEM / SEMAPHORE` | "Pipelining on TPUs is typically done between HBM (DRAM) to VMEM (Vector SRAM)"; "pipelining is not allowed unless the memory_space is marked as VMEM" |
| the pipeline depth | `pl.Buffered(buffer_count=n)` | "The default buffer count is 2 for all inputs and outputs" |
| the lookahead behaviour | `pl.Buffered(..., use_lookahead=True)` | "disabled by default" |
| which grid axes are independent | `dimension_semantics=(...)` | "`pltpu.PARALLEL` means the first dimension of the grid is distributed across the TensorCores"; `ARBITRARY` "cannot be parallelized" |
| the scratch buffers and their space | `scratch_shapes=[pltpu.VMEM(...)]` | "A scratch buffer must reside in VMEM, SMEM, or SEMAPHORE" |
| the SMEM-prefetched index arrays | `pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=n)` | Splash Attention uses 3 |
| the dtype of the accumulator | `preferred_element_type=jnp.float32` | "Matrix multiplication always produces results in the float32 format" |
| the matrix precision | `jax.default_matmul_precision` | **"Even if you pass in 32-bit operands to a matrix multiplication, they will be rounded to bfloat16 unless `float32` precision is requested."** |
| the traversal-order contract | *not stated; inferred* | "the grid is generally not processed in parallel, but sequentially, in lexicographic order" |

So: **13 decisions the programmer writes down, none of them searched.** The
brief's list (grid, block shape, index mapping, pipeline depth, dimension
semantics/layout) is correct and slightly under-counts.

### 2b. The strongest citation for "XLA cannot express tile-level control"

Ranked by how hard the source is to argue with.

**#1 — the Pallas design document, from the Pallas team.**
`https://docs.jax.dev/en/latest/pallas/design/design.html`:

> "For a wide variety of ML applications, XLA does a good job of compiling user
> programs but inevitably some users hit XLA's limitations. In these cases, we
> need to provide an 'escape hatch' to allow experts to write hand-tuned kernels
> that outperform XLA at that point in time. Furthermore, advances in ML systems
> research take some time to be incorporated into XLA and users often want to run
> ahead with them."

This is the primary source, written by the people who built Pallas, and it
makes three claims at once: (i) XLA's limits are real and known to its authors;
(ii) the intended remedy is *expert hand-tuning*, i.e. a human, not a better
search; and (iii) the compiler is expected to absorb the trick later, so the
human is a *provisional* subsidy. That third clause is the most useful thing in
the quote for this study, because it is the field's own statement of the
mechanism `research/TAXONOMY.md` calls "hardware-model-specific plan selection —
the human library's own schedule selector".

**#2 — the Pallas docs front page**, which is the user-facing statement of the
same thing, and which lists exactly the three capabilities:

> "Use it to achieve peak performance when XLA's automated optimizations fall
> short — specifically when you need to: **Write custom fusions that XLA cannot
> generate automatically. Take explicit control over memory movement and
> pipelining. Directly access low-level hardware features.**"
> — `https://docs.jax.dev/en/latest/pallas/index.html`

**#3 — a Google-authored paper that states it as a hard limitation with a
number attached.** arXiv:2604.15464, *Ragged Paged Attention* (Jiang, Chen,
Hechtman, Zhang, Mu — all Google; the vLLM/SGLang TPU kernel), §3.1.1:

> "One limitation of current XLA and Pallas/Mosaic is the **inability for users
> to explicitly control memory tiling**. For example, given a tensor F32(32, 128),
> XLA cannot be forced to use a tile of T(1, 128) and instead selects its own
> tile, e.g., T(8, 128). The memory layout of inputs and outputs is fully
> determined by XLA's Layout Assignment pass in HLO, **even for custom kernels**."

and §3.1.4, which is the sharpest sentence in the corpus for taxonomy item 4:

> "our empirical evaluations revealed that **XLA's Layout Assignment pass often
> fails to preserve the required T(packing, 128) tiling for our kernel**. While
> XLA typically respects a descending minor-to-major layout for the boundary
> inputs and outputs of a Mosaic custom kernel, it frequently reorders the layout
> of intermediate tensors — effectively performing implicit transpositions — to
> optimize for general compute efficiency elsewhere in the HLO graph. **This
> behavior overrides user-specified layout constraints in current XLA versions.**"

The workaround they were forced into is itself evidence: they added a *dummy
packing dimension* to the tensor shape purely to coerce XLA's tile choice, and
then had to move the shape transformation from offline weight preparation into a
runtime preprocessing stage because XLA kept reordering it. A compiler whose
layout decision can be dictated by adding a size-1 axis is not offering layout
control. **Tag: (i), stated by the team that ships the kernel.**

**#4 — where Pallas sits in the pipeline**, which explains *why* (i) rather
than (ii). arXiv:2604.15464 §2.2:

> "Mosaic is the MLIR-based compiler backend for Pallas, providing an 'escape
> hatch' from the standard XLA compilation pipeline. Instead of being fully
> lowered through XLA, a Pallas kernel is encapsulated as a custom HLO operation
> with encoded metadata."

and independently observed by Patrick Toulme, who dumped the HLO of Splash
Attention (`https://patricktoulme.substack.com/p/when-xla-isnt-enough-from-pallas`,
11 Jan 2026): the entire kernel is one HLO op,

```
%custom-call = ... custom-call(..., q, k, v),
  custom_call_target="tpu_custom_call",
  backend_config={"custom_call_config": {"body": "TUzvUgFNTElSMjEuMC4wZ2l0...", ...}}
```

with the whole kernel body as a base64 MLIR blob. XLA's fusion and layout passes
cannot look inside it. So the tile-level control is not something XLA *refuses*
to infer after a search — it is *not representable in the IR XLA reasons over*.
That is the definition of (i) in `research/TAXONOMY.md`, and it is now visible in
a dump rather than argued.

### 2c. The qualification: Pallas's own abstractions are not enough, and that
### is the strongest evidence of all

If Pallas were a well-designed escape hatch, a competent user would stay inside
it. The best evidence that tile-level control is genuinely hard is that the
production TPU kernels **abandon Pallas's own pipelining abstraction and drop
to primitives.** arXiv:2604.15464 §2.3:

> "While Pallas provides high-level abstractions — such as block specifications
> — to automatically generate multi-buffered pipelines, these often abstract away
> low-level DMA primitives and semaphore management (e.g., `async_copy` and
> `semaphore_signal`). In RPA, **we depart from these high-level abstractions and
> explicitly implement a custom pipelining strategy using Pallas's primitive
> APIs.** By retaining granular control over data movement and compute
> scheduling, we can more tightly integrate KV cache updates directly into the
> attention kernel."

So the ranking inside Pallas is: `pltpu.emit_pipeline` + `pl.Buffered` is a
*convenience*; to win you hand-issue `async_copy` + `semaphore_signal` + explicit
`wait`s. That is a two-level escape hatch, and the outer level is not where the
performance is. Compare Triton, where `num_stages` is genuinely the knob and the
compiler does the double-buffering.

This is the closest TPU analogue of anything in the CUDA taxonomy, and it is
**not** item 9 (pipelining depth) as previously framed. It is closer to
`research/TAXONOMY.md` item 2's "the vendor's hand-written, per-shape plan
selector": the human is writing the schedule selector themselves, in
synchronisation primitives, because no layer above it will.

### 2d. One hard (i) that the taxonomy does not have on CUDA: the block-shape
### divisibility rule

This is the cleanest (i) in the whole study and it is *architectural*, not a
search failure.

Pallas TPU's rule (`grid_blockspec.html`): for rank ≥ 2 blocks, "the last two
dimensions of your block shape must be equal to the respective dimension of the
overall array, or be divisible by 8 and 128 respectively. For blocks of rank 1,
the block dimension must be equal to the array dimension, or be a multiple of
1024, or be a power of 2 and at least `128 * (32 / bitwidth(dtype))`."

Contrast CUDA, where a misaligned block is an `(ii)`: Triton "cannot *prove*
successive iterations are aligned, so it conservatively emits scalar 32-bit
loads" — a better analysis fixes it. On TPU there is no better analysis. The
language says 8 and 128, full stop. If your block does not divide, it does not
compile (or it splits, silently).

`kda/tpu/tiling_check.py` encodes all of this with the source quote on each rule
and demonstrates the failure on a plan a person would plausibly write. The
deeper version, which the checker *cannot* catch, is in
`check_minor_lane_fit`: a 2-D TPU value is tiled `(8, 128)`, so **C = 64 can
never be the tiled minor-most axis of a 2-D vector tile**, and the `[C,C]` gram
and the UT inverse are physically `[64, 128]` fp32 — 2× the VMEM plus a relayout
on every access, inserted automatically by Mosaic's `infer-vector-layout` pass.
The cost is invisible in the source and appears only in a bundle dump.

**This is why KDA on TPU is not a port of KDA on CUDA**, and it has a
concrete resolution that I measured. See §3.

---

## 3. Taxonomy mapping: what the CUDA items become on TPU

`research/TAXONOMY.md` C1 items 1–11, with the study's tags (i) CANNOT EXPRESS /
(ii) CAN EXPRESS, DOES NOT FIND / (iii) NOT YET IMPLEMENTED, applied to
**Pallas/Mosaic vs XLA** (not to tinygrad, which cannot target TPU).

| # | CUDA item | TPU analogue | tag | what changes and why |
|---|---|---|---|---|
| 1 | Tile sizes (`BLOCK_M/N/K`, `num_warps`, `num_stages`) | `BlockSpec.block_shape` + the MXU's systolic shape | **(ii)**, with an (i) floor | Fully expressible and the effect is enormous. **Measured**: `docs/pallas/tpu/matmul.html`, TPU v5e, bf16: `bm=bk=bn=128` gives **5.93%** of peak on a 4096³ matmul; `bm=512, bk=1024, bn=1024` gives **78.47%**. That is **13.2× from tile shapes alone.** XLA gets 84.33% on the same shape. Independently: arXiv:2607.20466 grid-searched block sizes on 8 hand-tuned Pallas kernels, 203 configurations, for **up to 2.79×** over the Pallas defaults. But the (i) floor is the (8,128) rule: shapes that do not divide are not available to search. |
| 2 | Memory hierarchy / L2 blocking | VMEM ↔ HBM placement, scratch vs stream | **(ii)** for placement, **(i)** for layout | Placement is a first-class user decision (`memory_space=`, `scratch_shapes=`), so the "have we got a placement knob" gap that tinygrad admits ("we never copy the inputs to on chip SRAM") does not exist here — XLA and Pallas both stage to VMEM. What the user *cannot* choose is the **tile**, per arXiv:2604.15464 §3.1.1 and §3.1.4. There is no L2-analogue problem because there is no user-managed cache hierarchy between VMEM and HBM. |
| 3 | Warp cooperation (co-issuing an MMA) | **NO ANALOGUE.** The MXU is a fixed 128×128 (pre-v6e) or 256×256 systolic array; the systolic pipeline is a hardware property of the array, not a schedule the programmer writes. There are no warps, no `__shfl`, no warp-level roles. | **n/a** | **This is the item the brief predicted would vanish, and it does — completely.** Its disappearance is the strongest argument for running the study on TPU: 4 of the CUDA taxonomy's 11 items are pure SIMT artefacts. |
| 4 | MMA fragment layout | the systolic array's operand mapping and the (sublane, lane) register layout | **(i)**, stated by Google | On CUDA this is "Triton: automated". On TPU the array is weight-stationary and the register layout is `(8,128)` 32-bit, and — the load-bearing part — **you cannot dictate it**: arXiv:2604.15464 §3.1.1/§3.1.4, layout assignment "overrides user-specified layout constraints in current XLA versions". The counter-evidence is also instructive: arXiv:2507.11331 (SystolicAttention) measures **4.83× lower attention FLOP/s utilisation** for FlashAttention-style tiling than for a modified array that keeps everything on the array, because the tile shapes do not match the systolic width. The human *can* match them; the human cannot *name* the layout. |
| 5 | Accumulate vs I/O dtype | bf16 multiply / fp32 accumulate, and the precision default | **(ii) + (iii), identical to CUDA** | MXU takes bf16 in, accumulates fp32, always: "All multiplies take bfloat16 inputs, but all accumulations are performed in FP32" ([system-architecture-tpu-vm]). And the silent downgrade is the *same bug*: **"Even if you pass in 32-bit operands to a matrix multiplication, they will be rounded to bfloat16 unless `float32` precision is requested."** This is the TF32-on-SM80 case from the CUDA taxonomy, verbatim, one generation of hardware later. A compiler that cannot be asked to keep a precision is (i) for precision control. |
| 6 | Online algorithm / no materialised intermediate | **The same (i), and it is where the whole gap lives.** | **(i)** | Unchanged in kind, different in magnitude. The evidence: (a) JAX's XLA attention path, read from source, has no online softmax and no tiling — it materialises `[B,N,T,S]`; (b) XLA still beats hand-written tiling until that tensor exceeds VMEM (§1c); (c) Toulme's dump: naive attention 10,245 VLIW bundles / 526 MB HBM vs Splash Attention 1,651 bundles / 14 MB. **(i) confirmed, with a 37× byte ratio.** |
| 7 | Occupancy / register pressure | **NO ANALOGUE as stated, but a real analogue exists.** There is no occupancy: one instruction stream, no warps to pack. The analogue is the **VMEM budget** and VREG spilling into it. | **(ii)**, different mechanism | Pallas: "Consuming too many registers can also potentially cause register spills into VMEM which degrades kernel performance", and "it is possible to specify a window size that (together with space necessary to hold spilled vector registers) exceeds the size of VMEM. In this case, you will likely see a low-level compiler error message." So: register pressure → VMEM, expressed as a compile-time OOM or a silent spill. arXiv:2604.15464 §3.5 exposes four block-size knobs specifically because `cq`/`ckv` "are tuned to reduce VREGs pressure and mitigate VREG spilling by processing smaller compute blocks at a time." **Items 1, 7 and 9 are one coupled resource on TPU (VMEM) where on CUDA they are three loosely coupled ones.** `kda/tpu/shapes.py` section C prices this. |
| 8 | Vectorised / strided access | the (8,128) tile divisibility rule + the 512-byte DMA granule | **(i), decisively** | The sharpest tag flip in the table. On CUDA this is split (ii)/(i) — a better alias analysis fixes some patterns, a layout change fixes others. On TPU there is no analysis to improve: the block shape *is* the constraint, it is stated as a hard rule, and the DMA granule is a fixed 512 B = 128 × 32-bit lanes (arXiv:2604.15464 §2.1.2), so a sub-granule transfer is padded and you pay for bytes you do not use. `tiling_check.py` catches both. |
| 9 | Software pipelining depth | `pl.Buffered(buffer_count=n)`; `num_scalar_prefetch`; `use_lookahead` | **(ii) for the knob, (i) for the mechanism** | The depth is one integer, default 2 — the same shape as Triton's `num_stages`. But per §2c, the integer is only the outer abstraction: to actually overlap DMA with compute at the granularity you need you hand-issue `async_copy` + `semaphore_signal` + `wait`. Note also that raising `buffer_count` costs VMEM linearly, so this is *not* independent of items 1 and 7 the way `num_stages` is independent of `BLOCK_M` on CUDA. |
| 10 | Warp specialisation (producer/consumer roles) | **NO ANALOGUE.** Async DMA engines and the MXU/VPU are separate hardware units scheduled by Mosaic; there are no roles for a programmer to assign. | **n/a** | Deleted. The closest surviving thing is Mosaic's automatic dual-MXU distribution, visible in the LLO as `vmatmul...mxu0` and `vmatmul...mxu1` in adjacent bundles (Toulme's dump). That is (iii) → automated. |
| 11 | Recompute vs store / cross-kernel fusion | which buffers are scratch vs output | **(ii)**, and XLA is *good* at it | XLA's fusion is the reason it beats hand-written tiling below the VMEM threshold (§1c), so this is the one item where the compiler wins on TPU. Pallas's answer is `scratch_shapes` (persist across grid iterations, never touch HBM) plus `pltpu.emit_pipeline` (nested pipelines). The cost of taking the Pallas path is that you hand the whole kernel to XLA as one opaque custom-call, so you lose the whole-graph view for that region. That trade is the real subject of the study. |

**Summary: four items (3, 4, 10, and the register-file half of 7) are pure CUDA
confounds and have no TPU analogue. Two items (6, 8) are harder on TPU than on
CUDA — they are (i) with no "a better analysis would fix it" escape. One item
(2) moves from (iii) to (i) for a different reason. One item (11) is the one the
compiler wins.**

The interesting consequence: **the taxonomy's load-bearing item on CUDA, #6, is
also load-bearing on TPU, and the items that disappear (3, 4, 10) are the ones
the CUDA literature spent its most recent effort on.** Twill and Tawa — the two
systems `research/TAXONOMY.md` says closed the gap within the last 18 months —
are both about warp cooperation and warp specialisation. On TPU that entire line
of work has no subject. So TPU is not a second data point for the same claim; it
tests a *different* and older claim: that what a compiler cannot infer is the
algorithm and the layout.

---

## 4. KDA on TPU: the design

### 4a. What a free Kaggle TPU can and cannot run

`kda/tpu/shapes.py` prints the whole picture. The load-bearing number:

| shape | one `[C,C,K]` bf16 tensor | S1 total HBM (compulsory + 2 monsters) | S2/S3 | OI of S2/S3 | v6e ridge |
|---|---|---|---|---|---|
| B=2, T=1024, H=4, K=64, C=64 | 67.1 MB | 139.6 MB | 5.4 MB | 87.1 | 560 |
| B=1, T=2048, H=4, K=128, C=64 | 134.2 MB | 279.2 MB | 10.8 MB | 112.2 | 560 |
| B=1, T=4096, H=8, K=128, C=64 | 536.9 MB | 1116.3 MB | 42.5 MB | 113.6 | 560 |
| B=4, T=4096, H=32, K=128, C=64 (the CUDA study's target) | 8590 MB | 17.9 GB | 680 MB | 113.6 | 560 |

**Two pre-registered findings, before any TPU is touched:**

1. **KDA is memory bound on every TPU generation, at every shape we can afford.**
   Folded, its arithmetic intensity is ~113 FLOP/byte against a v6e ridge of 560
   and a v3 ridge of 137. So the folded kernel is memory bound too. MXU
   utilisation will be low *even for a perfect kernel*, and reporting MXU
   utilisation as if it were the score would be a mistake. The score is bytes.
2. **The achievable speedup is bounded, and the bound is analytic.** S1 must move
   268 MB of monsters where S2/S3 must move 0, against ~10.8 MB of compulsory
   traffic. So S1/S2 ≤ 279.2/10.8 = **25.9×** at that shape, whatever the
   schedules do. This is a *floor* on the gap and it is computable on a laptop.
   If the measured gap is much smaller than 25.9×, the honest reading is that the
   binding constraint is not the intermediate — and that is a finding.

The design tension, stated plainly: JAXBench insists on shapes where the XLA
baseline reaches ≥60% MXU utilisation, because "a benchmark that evaluates
kernels at undersized, memory-bound, or launch-overhead-bound shapes would not
reflect the regime in which TPU users actually operate". KDA cannot satisfy that
and also have a visible `[C,C,K]` intermediate, because the intermediate is
64× larger than everything else. **Resolution: run two shape regimes and report
the crossover, not a single number.** Regime A (`B=2, T=1024, H=4, K=64, C=64`,
67 MB monsters) is where all three sides run. Regime B (`B=1, T=4096, H=8, K=128`)
is where S1's traffic is 26× compulsory and S1 may OOM on a 16 GiB v5e-8.

### 4b. The two results I got by running the CPU checks

These came out of `kda/tpu/test_design.py` (16/16 pass) and they changed the
design, so they belong here.

**Result 1: C = 64 → C = 128 is numerically free, and that is the point.**
The (8,128) minor-lane constraint says the `[C,C]` accumulator is physically
`[64,128]` — 2× VMEM plus a relayout. The obvious response is to raise the chunk
size to 128, and the obvious objection is that KDA's paper fixes C=64 and the
secondary chunking `BC=16` exists to bound the exponent, so a wider chunk
widens the exponent range. **Measured, on a synthetic gate drawn from
`kda/SPEC.md` §4.1 (g ∈ [−11.9, −0.70], 4096 tokens, 128 channels):**

| C | BC | max exponent in one block |
|---|---|---|
| 64 | 64 | 485.8 |
| 64 | 16 | **135.8** |
| 128 | 128 | 901.6 |
| 128 | 16 | **135.8** |
| 256 | 16 | **135.8** |

With BC=16 the exponentiated range is **independent of C, identical to 2.8e-14**.
`fla/ops/kda/chunk_intra.py` already does the BC=16 sub-chunking, so the port
inherits it for free. So: raise C to 128, keep BC at 16, change nothing else.
And the thing XLA cannot tell you is that C=64 was wasting half of every vector
register — a fact that is in the *tile shape*, invisible in the *program*.

(Caveat, stated because it matters: my synthetic gate is harsher than KDA at
init, where `g ∈ [−1.6, −0.001]`. Treat 135.8 as an upper bound. A trained model
can have larger |g| than init, and `kda/SPEC.md` §4.2 notes `safe_gate` exists
precisely to bound this — so BC=16 is a correctness guard, not a tuning knob.)

**Result 2: the rewrite is exact, but its numerical bill is not uniform — and
that is the hard part of the port.** The factorisation
`exp(gc_i − gc_j) = exp(gc_i − gn)·exp(gn − gc_j)` is exact for *any* per-channel
reference `gn`, so the C²K tensor can always be eliminated. What BC buys is a
*bound*, and the bound only holds for the diagonal sub-chunk blocks. Measured
per (p, r) block at C=64, BC=16, on the same gate:

| block | max &#124;left exp&#124; | max &#124;right exp&#124; |
|---|---|---|
| (0,0) … (3,3) diagonal | 0.407 – 0.434 | 0.407 – 0.434 |
| (3,2) | 0.407 | 0.418 |
| (3,1) | 0.407 | 0.747 |
| **(3,0) farthest** | 0.407 | **1.074** |

The farthest off-diagonal block's right operand needs **2.5×** the diagonal
bound. `kda/SPEC.md` §4.4 describes what fla actually does: "4 diagonal
row-loops + 6 off-diagonal matmuls" — the kernel hand-schedules around exactly
this. **A Pallas port has to re-derive that decomposition and Mosaic will not do
it.** That is the sharpest instance in the whole KDA-on-TPU story of a decision
that is (ii) in principle and human in practice, and it is where I expect the
hand-written side to actually earn its keep — not in the fusion, which XLA wins.

### 4c. The three sides

| side | what it is | the variable it isolates |
|---|---|---|
| **S1 `xla_jax`** | KDA transcribed into `jnp` exactly as `kda/tiny_kda.py` is into tinygrad, `jax.jit`'d. The `[C,C,K]` decay tensors are in the graph. File: `kda/tpu/xla_kda.py`. | the baseline. Asks: *does a much better whole-graph compiler fix the CUDA study's finding?* |
| **S2 `pallas_hand`** | The same algebra, hand-tiled: exp folded into the contracting operand, C=128, BC=16, grid `(HV, NT)`, `dimension_semantics=(PARALLEL, ARBITRARY)`, `buffer_count=3`, explicit `scratch_shapes`, fp32 accumulators, `async_copy`/`semaphore` for the pipeline. File: `kda/tpu/pallas_kda.py`. | "a human reached for a low-level dialect". |
| **S3 `jax_rewrite`** | Pure `jnp` with the **same rewrite** applied to the graph before XLA sees it. C=128, BC=16, `jnp.where`-masked `[BC,BC]` blocks, no C²K. | "the low-level dialect bought anything *beyond* the rewrite". |

**S3 is the control that makes this falsifiable, and it is the part I would
argue hardest for.** `S2 − S1` is the boring comparison — of course a hand-written
kernel beats a graph that materialises a 64×-oversized tensor; that is
`research/TAXONOMY.md` item 6 restated. `S2 − S3` is the real question: *how
much of the hand-written win is the algorithm, and how much is the dialect?*
If S3 ≈ S2, the TPU story is entirely item 6 and Pallas was decoration — which
is a publishable result and it is consistent with what the sources say about
XLA's fusion strength. If S3 ≪ S2, then tile-level control has independent value
on TPU, which is the CUDA taxonomy's item 4/9 claim surviving into an
architecture with no warps.

**Pre-registration: I expect S3 to capture most of S2's win, and I expect the
residual S2/S3 to be under 1.5×.** Reasons: (a) XLA fuses the intra-chunk algebra
into the same kernel as the decay, per §1c; (b) at OI ≈ 113 against a ridge of
560 the kernel is memory bound, and memory traffic is decided by the graph, not
by the schedule; (c) the KDA intra-chunk `[C,C]` blocks are 64×64 and do not fill
a 128×128 or 256×256 MXU anyway, so neither side gets the systolic array working
hard. If the residual comes out above 1.5× that is a surprise and gets reported
as one.

### 4d. Metrics, ordered by how little they depend on the wall clock

| # | metric | how | arch-dependent? | what it proves |
|---|---|---|---|---|
| **M1** | HBM bytes + FLOPs in the compiled program | `jax.jit(f).lower(*a).compile().cost_analysis()`, summed | **no** | Whether the intermediate is in the program at all. The headline. Nothing else does this job, and it costs no TPU time. |
| **M2** | `dma.hbm_to_vmem` / `hbm_to_smem` count and total bytes in the LLO | `LIBTPU_INIT_ARGS=--xla_jf_dump_to=… --xla_jf_dump_llo_text=true`, plus `xla_mosaic_dump_to` for the Mosaic IR | no | What the hardware is asked to move. The same method produced the 14 MB vs 526 MB ratio in the literature. |
| **M3** | VLIW bundle count | same dump | no | Issue pressure, and how much the compiler had to pad. |
| **M4** | fusion / custom-call count in the HLO | `lower().compile().as_text()` | no | The TPU analogue of `notes/03-kda.md`'s "12,090 kernels", *and* the flip side: S2 is exactly one opaque custom-call, so this also measures what you gave up. |
| **M5** | static constraint count | `tiling_check.py` over each side's plan | no | The study's actual claim, as a count: how many of the 13 decisions above the programmer had to write down. |
| **M6** | correctness at three tolerances | fp64 1e-12, fp32 1e-6, bf16 1e-2, and the final state S | no | That any speed number is about the same operator. A wrong `S` gives a plausible chunk-0 output and garbage from chunk 1. |
| **M7** | MXU utilisation = FLOPs / (t·peak) | Perfetto trace device events, median of 50, divided by the *generation's* peak | yes, but as a fraction | Only the schedule, and only after M1–M4. Report against the analytic floor from `roofline.classify`; expect it to be low and say so in advance. |
| **M8** | `t / t_hard_floor` | M7 against `max(t_mem, t_flop)` from `roofline.py` | fraction | The cleanest single number: how far above the physically-possible floor each side lands. 1.0 is perfect. |

**Deliberately excluded:** raw milliseconds, and any cross-generation absolute
comparison. Per `research/TAXONOMY.md`'s discipline ("a residual against a human
is a floor, not a defect") and the v3-8 deprecation, a raw ms on an unknown
generation is not a result.

### 4e. Runbook (9 h cap, 20 h/week, and a TPU is not guaranteed)

Step 0, and it is not a formality: **a Kaggle TPU session is not reliably
obtainable.** `out/kaggle/tpu-facts-v2/` in this repo is a run that asked for
one and got `jax_sees_tpu: false`. So the plan has to be built so that a session
is an *opportunity*, not a resource.

1. `pip install -U "jax[tpu]"`; `tpu_info()` — record the generation and fill
   `roofline.GENERATIONS[gen]['vmem_mb']` and `['hbm_gib']` from `tpu-info`.
   Without this, M7 is meaningless and the VMEM check in `tiling_check.py` is a
   warning rather than a result.
2. Run the CPU checks in the notebook (`test_design.py` is numpy-only) to
   confirm the shape and the rewrite before touching the TPU. **This is free and
   it is what you do while waiting for a session.**
3. **M1 first, for all three sides, before any timing.** `cost_analysis()` on a
   compiled program costs seconds and needs no TPU time at all. If S1's byte
   count is not ~25× S2's, the rest of the experiment is not worth the session —
   stop and find out why. This is the pre-registered go/no-go gate, and it is
   the reason to write S1, S2 and S3 in Python first.
4. Dump LLO for all three (M2, M3). This is the number I most want.
5. Only then time (M7) and only as a fraction of the roofline.
6. Sweep exactly one knob: `buffer_count ∈ {2,3,4}` at fixed blocks. Then block
   shapes. Then `C ∈ {64, 128}`. In that order, because the first two are (ii)
   and the third is a different algorithm. One knob at a time or the attribution
   is worthless — the CUDA taxonomy's own §VI-D cautionary result (an A100-tuned
   kernel running 100× off on GH200) is the reason.
7. Save the dumps. The whole study's most reusable artefact is an LLO dump pair
   (XLA vs Mosaic) for the same operator, and it does not go stale the way a
   millisecond figure does.

---

## 5. Prior work: does the Pallas-vs-XLA comparison exist, and with what numbers?

**Yes. Two published sources with numbers, plus a third blog with a full IR
walkthrough. None of them is "hand-written vs compiler" in the CUDA taxonomy's
sense; all of them measure Pallas (hand-written) against XLA (compiled), which is
exactly our S2-vs-S1.**

### 5a. arXiv:2607.20466, *JAXBench: Benchmarking Autonomous TPU Kernel Optimization* (19 May 2026)

Google authors (Tschand, Hong, Walker, Cai, Wang, Subramanian, Dev, Reddi,
Yazdanbakhsh, Sankaran). TPU v6e (Trillium), single chip. This is the closest
existing work to our study and it is the one to cite.

**Methodology.** 50 JAX workloads: 17 production operators extracted from MaxText
(Llama-3.1, DeepSeek-V3, Mixtral, Mamba-2, AlphaFold2) at production-scale dims,
plus 33 KernelBench Level-2 fused operators translated to JAX. Shapes are tuned
per-operator until "the XLA baseline reaches at least 60% MXU utilization". Each
workload is implemented in idiomatic JAX and `jax.jit`'d — "the strongest
reproducible baseline for TPU because XLA already performs aggressive whole-program
optimization including operator fusion, buffer assignment, and layout
optimization." For 8 of the 17 priority kernels, expert-optimized Pallas kernels
are imported from the upstream JAX Pallas ops library (Tokamax) and their block
sizes **grid-searched on v6e: 203 configurations, up to 2.79× over the Pallas
defaults** (Megablox GMM). Timing is device-side via `jax.profiler` traces,
median of 50, explicitly *not* `perf_counter` + `block_until_ready` because that
includes 10–20% host overhead for sub-millisecond kernels. FLOPs come from XLA's
`cost_analysis()`.

**The numbers that matter to us (Table 3):**

| benchmark | XLA (ms) | hand-tuned Pallas (ms) | hand-tuned | best agent |
|---|---|---|---|---|
| Flash Attention | 25.21 | 6.45 | **3.91×** | 2.67× |
| GQA Attention | 51.13 | 14.59 | **3.50×** | 2.63× |
| MLA Attention | 20.32 | 21.72 | 0.94× | 0.85× |
| Sparse Attention | 5.95 | 6.89 | 0.86× | 2.81× |
| Paged Attention | 7.82 | 3.41 | **2.29×** | — |
| Ragged Paged Attention | 18.70 | 2.71 | **6.91×** | — |
| GEMM | 5.36 | 5.59 | 0.96× | 0.75× |
| Megablox GMM | 3.26 | 2.01 | 1.62× | 2.21× |
| **geomean (floor 1×)** | | | **2.08×** | 1.60× |

**How they attribute the gap** — this is the part that answers question 4
directly:

> "Attention and elementwise kernels remain memory-bound. Hand-optimized Pallas
> kernels close this gap for the 8 priority workloads **by tiling computation to
> reduce redundant HBM traffic**, achieving speedups of up to 16.3× on Ragged
> Paged Attention and 3.5× on Flash Attention. **Compute-bound workloads like GEMM
> show no improvement, confirming that XLA already saturates the MXU for high-OI
> operations and that optimization effort is best directed at memory-bound
> kernels.**"

(Fig. 2 caption. The 16.3× is the "after" in vLLM; the 6.91× is the isolated
kernel.)

**This attribution is exactly the prediction `research/TAXONOMY.md` makes and
`kda/tpu/shapes.py` section A computes independently:** the gap is HBM traffic,
and it only exists when the operator is memory bound. Their OI analysis puts
ragged paged attention at `OI → G = H_q/H_kv` in the long-context limit, which
for Llama-3.1-405B (G=16) is 16 FLOP/byte "far below the v6e ridge point of
OI ≈ 560, confirming that ragged paged attention is memory-bandwidth-bound at
any context length." **Our KDA at OI ≈ 113 against the same ridge of 560 is the
same regime, arrived at from a different operator.** That is independent
corroboration, from a different group, on different hardware, of a roofline
calculation we did from shapes.

Their second attribution result, which is the strongest evidence in the corpus
for the design in §4b: **"the [Pallas] constraints that govern correctness,
including lexicographic grid traversal, VMEM/SMEM/HBM placement, (8,128) block
divisibility, and prefetch scheduling, are absent from error messages, leaving
feedback loops nothing to iterate against."** And measured: 99.7% of best-of-NN
samples and 93.8% of iterative-refinement samples fail at compile or first
execution on API misuse. That is why `tiling_check.py` exists and why it is the
most useful artefact in this directory.

One more number worth having: conditioning on curated TPU documentation raised
per-sample correctness from **5.8% to 37.3%** and geomean speedup from 1.18× to
1.28×, a larger effect than going from one model to a larger one. For a
sparsely-documented DSL, the bottleneck is information, not reasoning. That is a
statement about the same thing this study is about, from the opposite direction.

### 5b. Patrick Toulme, *When XLA Isn't Enough: From Pallas to VLIW with Splash Attention on TPU* (11 Jan 2026)

A blog, not peer-reviewed, and the author says "**No production compiler does
this today (to my knowledge)**" — so treat the generalisation as an assertion,
not a proof. But the measurements are concrete and the IR dumps are published
(`github.com/patrick-toulme/justabyte/tree/main/tpu_pallas_post`).

TPU v6e Trillium. Same hardware, same compiler, two implementations of attention:

| | VLIW bundles | HBM traffic |
|---|---|---|
| naive (materialised) attention | **10,245** | **526 MB** |
| Splash Attention, `block_size=512` | **1,651** | **14 MB** |

**A 37.6× byte reduction and a 6.2× bundle reduction.** The attribution:

> "XLA optimized both aggressively. The 6× gap isn't a compiler failure — **it's a
> compiler limitation.** XLA can fuse ops and schedule instructions, but it
> can't rewrite your algorithm. **It can't infer the streaming/online-softmax
> reformulation**, that you never need to materialize the full attention matrix."

and, on what XLA does fuse:

> "fusion.5: Q @ K^T + mask + reduce_max → (max, scores) … fusion.2: exp(scores −
> max) + reduce_sum → sum … fusion: normalize + S @ V → output. XLA fuses the
> matmul with the mask and max reduction. It fuses the exp with the sum
> reduction. It's doing real work here. **But the 128MB attention matrix still
> gets written to HBM between `fusion.5` and `fusion.2`.**"

The IR walkthrough is also the best available documentation of what Pallas
actually produces, and it is where the 256×256 MXU, the `(8,128)` register
tiling, the `dma.hbm_to_vmem` double-buffering, and the dual-MXU bundle
assignment are all visible. That dump is the model for our M2/M3.

### 5c. Archer Zhang, *Forcing Flash Attention onto a TPU and Learning the Hard Way* (6 Mar 2026)

A blog, single-head, TPU v5e, and it is the *counter*-example: §1c above. 35×
slower at n=4096, and the fix was not a Pallas kernel but a one-line change of
`lax.fori_loop` to `jax.vmap` on the outer (independent) Q dimension — 14.1× at
n=4096 and 44.8× at n=16384 from that alone, with identical arithmetic.

Two things worth stealing. First, the *diagnostic*: he built a tick-level
systolic-array emulator and attributed the slowdown to **MXU utilisation of
20.1% for tiled flash vs 94.3% for the fused standard path**, because the two
matmuls per flash tile have inner dimension ≤ 128 so the 128-wide systolic
pipeline is starved. That is the TPU analogue of taxonomy item 4 and it is a
*cycle model*, not a wall clock — we should build one for KDA too. Second, the
honest failure: he tried to write the Pallas kernel and could not — "Mosaic's
constraints are restrictive: no dynamic indexing (`k_all[indices, :]` lowers to
an unsupported gather), 1D blocks must be multiples of 128, kernels that compile
on one JAX version fail on another. The code didn't survive into this post."

### 5d. arXiv:2507.11331, *SystolicAttention: Fusing FlashAttention within a Single Systolic Array* (v4, 8 Dec 2025)

Not a Pallas/XLA comparison, but the best quantification of taxonomy item 4 on
TPU. They build a modified array (FSA) that runs FlashAttention entirely on the
systolic array "without external vector units" and measure:

> "FSA achieves **1.77× and 4.83×** higher attention FLOP/s utilization compared
> to AWS Neuron-v2 and **Google TPUv5e**, respectively."

So a flash-style tile schedule on a stock 128×128 TPU array leaves **4.83×** on
the table purely from the array/schedule mismatch. Independent third-party
evidence for the same item, from an architecture paper rather than a compiler
paper.

### 5e. What does NOT exist

- **No Pallas-vs-XLA comparison for a linear-attention / state-space operator.**
  JAXBench's 17 priority kernels include RetNet retention and Mamba-2 SSD (both
  are linear attention / SSMs, and the paper notes Mamba-2 SSD is one of the
  benchmarks where the agents "still fail" and where no expert Pallas reference
  exists), but neither has a published hand-tuned Pallas baseline. **Our KDA
  comparison would be the first with one.** That is the opening.
- **No measurement of the S2-vs-S3 decomposition** — i.e. nobody has separated
  "the human changed the algorithm" from "the human changed the dialect" on TPU.
  JAXBench compares Pallas to XLA and to *other Pallas*; Toulme compares Splash
  to naive; Zhang compares tiled to fused. Nobody runs the rewrite in pure JAX as
  a control. **This is the actual gap this study fills**, and §4c is designed
  around it.
- **No TPU measurement of anything tinygrad**, because tinygrad cannot target
  TPU (§1a). The CUDA study's "zero arXiv papers mention tinygrad" becomes, on
  TPU, "zero can".

---

## 6. What I could not resolve

| question | status |
|---|---|
| VMEM capacity for v3, v5e, v5p, v6e | **UNVERIFIED.** Pallas docs say only "16MB+". Only v7x has a sourced figure (64 MB/TC, arXiv:2604.15464 Fig. 1). `roofline.py` carries `None` and `tiling_check.py` downgrades the VMEM check to a warning where it is `None`. **Get this from `tpu-info` before quoting any VMEM number.** |
| HBM capacity for v6e (32 GiB is the commonly cited figure) and v5e (16 GiB) | **UNVERIFIED.** The peaks *are* sourced (918 TFLOPS / 1640 GB/s for v6e from JAXBench Fig. 2; 197 TFLOP/s / 819 GB/s for v5e from the Pallas matmul doc), the capacities are not. |
| TPU v4, v5p peaks | **UNVERIFIED.** `roofline.py` prints `None` and `classify()` returns "unknown" rather than guessing. |
| Whether the vector register is `(8,128)` on *v3* | **UNVERIFIED.** Sources give 8×128 for v6e and v7x and describe v3 as a 128×128 systolic array, but I did not find a v3-specific VREG shape. If v3's VREG is not `(8,128)`, the divisibility rule in `tiling_check.py` is wrong for v3. |
| Whether XLA honours `async-start`/`async-update`/`async-done` chains as a user-controlled buffer count on TPU | **UNVERIFIED.** The HLO opcodes exist (`openxla.org/xla/async_ops`) and are a general mechanism, but I did not find a source stating that the number of `async-update` steps is the pipeline depth. This matters for taxonomy item 9's tag: if it works, pipelining depth is (ii) in HLO, not just in Pallas. |
| Whether `pl.when` inside an `emit_pipeline` body survives unrolling for a C=128 UT transform (127 dependent steps) | **UNVERIFIED, and the most likely thing to be slow.** Pallas TPU docs warn that loop primitives "get fully unrolled during the compilation at the moment, so try to keep the loop trip count reasonably small". |
| Whether Mosaic relayouts the `[C,C]` accumulator more than once | **UNVERIFIED.** Only an LLO dump answers it. `check_minor_lane_fit` predicts a pad and a relayout; the prediction is untested. |
| Whether `jax.device_count()` on a v3-8 or v5e-8 slice reports TensorCores or chips | **UNVERIFIED.** v3-8 is 4 chips / 8 TensorCores; v5e-8 is 8 chips / 8 TensorCores. `roofline.cores()` assumes TensorCores, which is what the docs' `--accelerator-type=v3-8` naming implies ("the number of TPU cores"), but the split between chips and cores differs between the two and I could not confirm the JAX-visible count for v5e-8. Check `tpu_info()` first. |
| The KDA Pallas kernel itself | **NOT RUN.** It is a design sketch with the load-bearing choices annotated. The gate activation (`kda_gate`), the short conv, the output gate and RMSNorm of the layer wrapper are all out of scope, as in the CUDA study. |

## 7. Files

| file | role | runs on CPU today? |
|---|---|---|
| `kda/tpu/README.md` | this document | — |
| `kda/tpu/roofline.py` | per-generation peaks, ridge point, `classify()`, hard time floors. One peak per line, each with a URL. | **yes** |
| `kda/tpu/shapes.py` | the three sides; FLOP and byte counts; the `[C,C,K]` materialisation model; the 25.9× analytic speedup bound; VMEM residency vs pipeline depth; the MXU-fill calculation for `[C,C]` blocks. | **yes** |
| `kda/tpu/tiling_check.py` | Pallas TPU legality rules R1–R8, each with its source quote, plus the (8,128) minor-lane check that BlockSpec does *not* catch. 10 worked examples. | **yes** |
| `kda/tpu/test_design.py` | 16 checks against `kda/ref.py` in float64: the exp-folding rewrite is exact (2.5e-16) at fp64/fp32/bf16 tolerance; the causal half is zero; **C=128 is numerically free given BC=16**; the rewrite's per-block numerical bill; the grid is a bijection; the plan is legal at C=128. | **yes** |
| `kda/tpu/xla_kda.py` | S1, the `jnp` transcription. | **no** — no jax in the venv |
| `kda/tpu/pallas_kda.py` | S2, the hand-written Pallas/Mosaic design, with every forced decision annotated. | **no** — no jax, no TPU |
