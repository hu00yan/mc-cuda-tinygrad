# FAIR-BASELINE — when a measured ratio licenses a claim, and when it does not

**Status: pre-registration, written before any comparative GPU run.** No experiment was
run to produce this document. Every number in it is arithmetic, a datasheet value marked as
such, or a source reading with `file:line`. `bench/HYPOTHESES.md` §6 deletes the
comparative hypothesis; this file is the protocol that governs whether it can be re-entered,
and on what terms. It does not re-enter it.

**What it is for.** The study's baseline for KDA is the expert-written Triton kernel from
`flash-linear-attention` — written by the algorithm's own authors. That is the strongest
baseline obtainable, and it is also the easiest to use dishonestly, in *both* directions. A
ratio against it can be inflated by degrading the baseline, and it can be mis-labelled by
attributing an algorithm choice to codegen. Both failure modes are silent, both produce a
number, and neither produces an error. This file makes them require an explicit act.

**Scope note.** This document governs the comparative arm only. It does not touch
`HYPOTHESES.md` §2, H2, H3, H4, H5, H8, H12, H13, or the roofline/self-delta claims, all of
which are unaffected and remain answerable on the free cards. It adds requirements to R1 and
R11 and adds one required read to R3; it removes nothing and contradicts nothing in
`PLAN.md`.

---

## 1. The one-line rule

> **A ratio between a compiler and a human is a claim about the algorithm, the hardware, the
> toolchain, the dtype and the author's afternoon, until every one of those five has been
> held fixed and shown fixed. The number licenses a codegen claim only when the two programs
> are the same program — which for KDA means the same intermediate tensor, at the same
> granularity, with the same instruction class available to both.**

The operational form, and the one that goes in the methods section:

> **Report the ratio only together with the cell label
> `(framework, algorithm, intermediate granularity, dtype, accumulate precision, smem-config
> gate, instruction class observed in generated code)`. A cell missing any field of that
> label is not reportable.**

This is the study's existing `PLAN.md` §6.3 consequence ("every number is labelled by cell")
pushed to its conclusion. The cell label already existed; what is new here is the claim that
a label missing the *intermediate* and *instruction class* fields is not a cell at all — it
is a number that will be read as a codegen measurement.

---

## 2. The two comparisons, and what each one is allowed to say

### 2.1 Comparison (a) — same-algorithm

**Definition.** Both sides implement the *same* algorithm, expressed in the two frameworks.
Concretely, for every tensor the algorithm materialises, the two programs materialise the
*same* tensor at the *same* size, from the same inputs, in the same dtype with the same
accumulation behaviour, and both are given the opportunity to emit the same instruction
class.

**Granularity is the hard part, and it is the part people get wrong.** "Same algorithm" is not
a property of the *operator*; it is a property of a *pair of tensors*. Two programs that
compute KDA can disagree about which intermediate exists, and if they do, the ratio is not a
codegen ratio. Concretely, this port and `fla` disagree at two identified points:

| # | disagreement | TinyGrad port | `fla` kernel | class |
|---|---|---|---|---|
| D1 | size of the decayed key-key gram | `[B,NT,HV,C,C,K]` at `C=64` — 16,777,216 el at the port shape (`HYPOTHESES.md` H2 arithmetic) | `[BC,K]` per sub-chunk, `BC=16` hard-coded (`fla/ops/kda/chunk_intra.py:825`), held in registers; 64× smaller | **(b)**, see §2.2 |
| D2 | the triangular solve of step 4 | unrolled forward substitution, one entry per step, `C` outer × `C` inner = 2,016 dependent steps (`kda/tiny_kda.py:178-194`; `HYPOTHESES.md` H3) | block-triangular/Neumann series of 12 `tl.dot`s on `[BC,BC]` blocks (`fla/ops/kda/chunk_intra.py:320-355`) | **(b)**, see §2.2 |

**Therefore comparison (a) is not runnable at whole-operator granularity. It is runnable only
at the granularity of a stage where the two programs' dataflow is isomorphic.** The
candidate stages, from the port's own step list (`kda/tiny_kda.py:8-20`):

| stage | both sides materialise the same tensor? | verdict |
|---|---|---|
| 1 `g_cum = cumsum(g)` | yes, `[B,NT,HV,C,K]` | **(a) admissible** |
| 2 decayed gram | no — D1 | **(b)** |
| 3 `Btri = StrictTril(Diag(beta)·Z)` | yes given stage 2, `[B,NT,HV,C,C]` | **(a) admissible**, but only if stage 2 is fed identically |
| 4 solve for `M` | no — D2 | **(b)** |
| 5/6 `w = M(Γ·k)`, `u = Mv` | yes, `[B,NT,HV,C,K]` / `[…,C,V]` | **(a) admissible** |
| 7a `v_new = u − wS` | yes, `[B,T,HV,V]` | **(a) admissible** |
| 7b `Aqk = Tril((q·exp(Δgc))·k)` | yes in shape, no in granularity — `fla` does it at `BC=16` and fuses | **(b)** unless forced to `C=64` |
| 7c `o = (Γ·q)S + Aqk·v_new` | yes, `[B,T,HV,V]` | **(a) admissible** |
| 7d `S = Diag(γ^C)S + (…)ᵀv_new` | yes, `[B,HV,K,V]` | **(a) admissible** |

**A second, independent construction of (a) exists and is cheaper.** Write a Triton kernel
that transcribes `kda/tiny_kda.py` step for step — same seven steps, same `[C,C,K]`
materialisation, same `C=64`, same unrolled substitution — and compare that against the
TinyGrad port. This is the only construction of (a) that is well-posed at whole-operator
granularity, because it makes the intermediate identical by construction rather than by
argument. Its risk is the one `PLAN.md` §6.1 already names: it is one author's second
implementation, so it is a **floor on the gap, not an estimate of it**, and it is admissible
only if it clears §5. It is also the same work `PLAN.md` R1 already budgets for the sm_75
case ("a hand-written sm_75 KDA kernel must be written from `kda/SPEC.md`, by the same
author, in the same session, with the same effort budget as the TinyGrad port"). The
`PLAN.md` R1 fallback and the whole-operator form of comparison (a) are therefore the *same
piece of work*; this file just names what it is for.

**Licenses a reader to claim:** that TinyGrad's expression of this algorithm, on this
architecture, in this dtype, produces device time `r×` the best hand-scheduled expression of
the *same* algorithm that we were able to write. The gap is attributable to **codegen and
scheduling quality** — the `OptOps` search, the fusion decisions, the kernel count — and to
nothing else. It remains a statement about *one author's best* hand schedule (HYPOTHESES
§6.1, 2607.04454 §VI-E).

**Must NOT be used for:**
- any statement about the operator KDA rather than the program we wrote;
- any statement about the state of the art in hand-writing this operator;
- any statement about tensor cores, unless §3.4's instruction-class check has passed **for
  the stage being timed**;
- any gap smaller than `r_noise` from `PLAN.md` §5.2;
- any extrapolation across dtypes, across chunk sizes, or across architectures.

### 2.2 Comparison (b) — algorithm-choice

**Definition.** The human side is free to use a better algorithm than the one the compiler is
given. `fla` was written by people who know the algorithm, including its key trick:
computing the per-timestep updates naively would materialise a `d`-dimensional state and cost
`O(d²)` memory, which the experts avoid entirely. arXiv 2406.06484 §3.1 makes the argument in
peer-reviewed form: *"computing **u**_t naively requires explicitly materializing **S**_{t−1}
… which would require O(d²) memory. We now show that we can obtain the **u**_t's without
explicitly materializing **S**_{t−1} in O(d) memory."* (`PARITY-SCOPE.md` §2.3)

**This is a real and interesting difference. It is not codegen quality.**

**Licenses a reader to claim** — and the wording is fixed, because the loose version of this
sentence is where the study's thesis gets stolen:

> Given the chunked delta-rule algorithm as specified, hand-written Triton computes it in
> `r×` less device time than TinyGrad's generated code. Decomposed, this is the sum of:
> **(i) cannot express** — the intermediate that is avoided by construction has no
> schedulable representation, so the generated code must write and re-read it; **(ii) can
> express, does not find** — the transform exists in the search space and the cost model or
> eligibility test declines it; **(iii) not yet implemented** — the pass is specified or
> present one layer down and absent from the shipping path; **(iv) does not exist for this
> architecture** — the compiler's table has no entry. Each term is reported separately, in
> that vocabulary, each with its own evidence per `PLAN.md` §5.4.

**Must NOT be used for:**
- "TinyGrad's codegen is `r×` worse than a human's" — it licenses nothing about codegen;
- "a tile DSL cannot reach this operator class" without naming the algorithm change that
  would be required;
- any comparison against a published baseline number from different hardware. 2406.06484
  §3.2's speed comparison is a **chart**, not a number (`PARITY-SCOPE.md` §4.3: "a chart I
  could not read numerically, so **no number is quotable**"), and the widely repeated "30×"
  is from a course slide and is marked unverified there. Neither may appear.

**The precedent to borrow, and it is a precedent for *declining to publish a number*:
2607.04454 §III-B, verbatim** — *"because the Triton and TileLang implementations are not
algorithmically identical to this baseline, we report attention results separately"*; Table I
carries the literal cell **"baselines non-equivalent"**; Figure 1's caption: *"Attention
excluded: the Triton (tiled FlashAttention-2) and TileLang (**untiled O(n²)**) kernels are
algorithmically different."* A careful group declined to put a number on exactly our operator
class. Two of the two excluded attention cells are baseline-path artifacts (Triton
FlashAttention-2 at **7691%** of PyTorch's *unfused eager* path; TileLang at 54.3% from
untiled O(n²)) — i.e. **both excluded numbers were flattering to the DSL, and excluding them
was the conservative act.** This study should expect its own excluded numbers to be flattering
to `fla`, and should exclude them for the same reason.

**One more reason to keep (b) separate, which cuts the other way.** 2607.04454 §VI-E
measures the algorithm-selection term for cuDNN-vs-Triton on Conv2d as *"only ≈2–3%"* — so
the algorithm term is **not always dominant**, because cuDNN's residual there is cache
blocking and tuning, not algorithm choice. A large (b) number for KDA is therefore not a
foregone conclusion and must be measured, not assumed in either direction. The one thing that
is not in doubt: the term is **unbounded above by the (a) number**, since the two overlap in
the accounting unless (a) is run at stage granularity.

---

## 3. Conditions under which the baseline is not degraded

This is the section that stops the study from reporting an artifact it built itself.

### 3.1 The rule

> **A comparison on architecture `A` is INVALID unless all four gates pass on `A`. If any gate
> fails, the cell is reported as `INVALID` and the string `INVALID` appears in the results
> table. A number is never reported alongside `INVALID`.**

The reason for the absolute form: a degraded baseline inflates the gap **in TinyGrad's
disfavour**, and it does so invisibly, because the degradation makes the human side slower and
a slower human side is exactly what a study wanting a large number looks like. "Report it with
a caveat" fails because the caveat is read by nobody and the ratio is read by everybody.

### 3.2 Gate 1 — instruction class is the same on both sides

Both sides must be *capable* of the same instruction class, and the class actually used must
be **observed in generated code, not assumed from a datasheet**.

Verified facts about the two sides, at the version this study pins (tinygrad 0.14.0 in
`.venv`; `fla` and `triton` read from `main`/`3.4.0` on 2026-09-27):

- **TinyGrad's table.** `codegen/opt/tc.py`: `def get_cuda(arch): return cuda_sm89 if
  (ver:=int(arch[3:])) >= 89 else cuda_sm80 if ver >= 80 else cuda_sm75 if ver >= 75 else
  []`, with `cuda_sm75 = cuda_8168_f16` (one shape, f16-only) and
  `cuda_sm80 = cuda_81616 + cuda_8168_f16 + cuda_8168_tf32`. So: `sm_60 → []`, sm_75 → one
  f16 shape, `sm_80+` → bf16 `m16n8k16` and tf32 both present. (This is `HYPOTHESES.md` S4,
  re-verified; see §8.)
- **The `fla` side on sm_75 is not just "no MMA" — it is degraded three ways at once.**
  1. *Precision.* `fla/utils/_device.py:162` — `IS_TF32_SUPPORTED = (IS_NVIDIA and
     torch.cuda.get_device_capability(0)[0] >= 8)`. And
     `fla/utils/_device.py:180-183`, verbatim:
     ```python
     if IS_NVIDIA and not IS_TF32_SUPPORTED:
         # Make old card happy, since triton will use tf32 by default.
         # This is a workaround for old nvidia card.
         os.environ['TRITON_F32_DEFAULT'] = 'ieee'
     ```
     This is a **process-global** flag applied to the whole `fla` library on sm_75, not a
     local kernel option, and it changes dot precision everywhere in the run.
  2. *The solve's precision specifically.* `fla/ops/kda/chunk_intra.py:21-24`:
     ```python
     if IS_TF32_SUPPORTED:
         SOLVE_TRIL_DOT_PRECISION = tl.constexpr('tf32')
     else:
         SOLVE_TRIL_DOT_PRECISION = tl.constexpr('ieee')
     ```
     used at 12 `tl.dot` sites (lines 320-355) — i.e. the block-triangular inverse that is
     stage 4, computed in IEEE fp32 on sm_75 and tf32 on sm_80+.
  3. *Triton's own lowering.* `third_party/nvidia/backend/compiler.py:305` —
     `emuTF32 = (capability // 10 >= 8)` — then line 311
     `passes.ttgpuir.add_f32_dot_tc(pm, emuTF32)`. On sm_75 fp32 `tl.dot` does **not** use
     tensor cores.
  Plus the arch has **no bf16 MMA instruction at all** (Turing), so a bf16 `tl.dot` has no MMA
  to go to. **Confirmed: `fla`'s KDA takes a non-MMA path on sm_75 and emits MMA on sm_80+**,
  as hypothesised — but the mechanism is broader than "MMA", and the smem gate below bites
  harder than the instruction-class one.

### 3.3 Gate 2 — the baseline's autotune space is not narrowed by the architecture

**This gate is the one that is easy to miss, it is not about compute capability at all, and it
invalidates cards that pass Gate 1.** `fla` narrows its own config lists on shared-memory
thresholds. `fla/utils/_device.py:212-233` defines the thresholds, and they are **not
monotone in compute capability**:

| `Backend` member | bytes | card |
|---|---|---|
| `AMPERE` | 166,912 | A100 |
| `HOPPER` | 232,448 | H100 |
| `ADA` | 101,376 | RTX 4090 |
| `DEFAULT` (used by `check_shared_mem()` with no arch) | **102,400** | — |

The dependent config lists, verbatim from `main` on 2026-09-27:

- `fla/ops/common/chunk_delta_h.py:52` — `for num_stages in ([2, 3, 4] if
  check_shared_mem('ampere') else [2, 1])`, and `:53` — `for BV in ([32, 64] if
  check_shared_mem('ada') else [32])`. These are two of the four KDA-fwd kernel families
  (`chunk_fwd.py` calls into `chunk_delta_h.chunk_gated_delta_rule_fwd_h`).
- `fla/ops/gla/chunk.py:21` — `BK_LIST = [32, 64] if check_shared_mem() else [16, 32]`; `:22`
  — `BV_LIST = [64, 128] if check_shared_mem('ampere') else [16, 32]`. This is the output
  kernel, `chunk_gla_fwd_o_gk`, which the KDA forward calls.

**`num_stages ∈ {2,1}` is not a mild narrowing. `num_stages=1` disables the software pipeline
entirely.** A T4 (64 KB opt-in shared memory) fails *both* `check_shared_mem()` and
`check_shared_mem('ampere')`, so the KDA baseline on a T4 is autotuned over a different,
smaller, and structurally worse config space than the same code on an A100.

Two consequences that must be in the paper:

1. **The sm_75 baseline is not the modern kernel.** It is the same source with tf32 dots
   off, the solve computed in IEEE, the shared-memory autotune space collapsed to
   `num_stages ∈ {2,1}`, `BV = 32`, `BK ∈ {16,32}`, `BV_o ∈ {16,32}`, and no bf16 MMA
   available to it. Any ratio measured on a T4 is a ratio against **that configuration**.
2. **Compute capability is the wrong validity test, and the Ada cards prove it.** An RTX 4090
   (sm_89) has `mma.sync.aligned.m16n8k16` bf16 — it passes Gate 1 outright — and 101,376 B
   of opt-in shared memory, which is **below `DEFAULT = 102,400`**, so it fails
   `check_shared_mem()` as well. A 4090 is *more* MMA-capable than a T4 by a wide margin and
   gets a *narrower* autotune space than an A100, and the gate is decided by a shared-memory
   constant. The validity test this study applies is therefore **per-card and empirical**
   (§3.5), not a lookup of `sm_XX`.

### 3.4 Gate 3 — the executed baseline is the one that was measured

**`fla`'s KDA is not one artefact. It is a runtime dispatch over three backends, and the
highest-priority one is a CUTLASS kernel from a different organisation.** From
`fla/ops/backends/__init__.py` (the generic `dispatch`/`BaseBackend` registry) and
`fla/ops/kda/backends/__init__.py`:

| priority | backend | available when | implements |
|---|---|---|---|
| 0 | `TritonAscendKDABackend` | `IS_NPU` only — never on NVIDIA | the whole KDA fwd/bwd for Ascend |
| 3 | `FlashKDABackend` | `flash_kda` importable **and** `not torch.is_grad_enabled()` **and** `q.dtype == torch.bfloat16` **and** `K == 128` **and** `V == 128` **and** no GVA **and** `use_gate_in_kernel` and `use_qk_l2norm_in_kernel` and `use_beta_sigmoid_in_kernel` and `state_v_first` and `safe_gate` | `flash_kda.fwd` — a fused CUTLASS forward, © 2026 Moonshot AI, <https://github.com/MoonshotAI/FlashKDA>, "replaces the multi-kernel Triton path" |
| 5 | `KDATileLangBackend` | `tilelang` importable and usable nvcc | `chunk_kda_bwd_wy_dqkg_fused` only (backward) |
| — | default | fallback | the mainline Triton path |

Required protocol, all of it:

- **P3.1 — Pin the dispatch.** Set `FLA_DISABLE_BACKEND_DISPATCH=1` to force the mainline
  Triton path (`fla/ops/backends/__init__.py:28-30`, and `dispatch` returns the undecorated
  function at line 167-168). Or, if the CUTLASS path is deliberately wanted, install
  `flash_kda` deliberately, run under `torch.inference_mode()`, and set
  `FLA_FLASH_KDA=0` in every cell that must *not* use it.
- **P3.2 — Record which one ran.** The registry logs at `logger.info` only, and only once per
  `(op, func, backend)` (`fla/ops/backends/__init__.py:206-210`). Default Python logging
  swallows it. The cell label therefore carries an explicit `fla_backend=` field, and the
  run enables `logging.INFO` on `fla` and archives the lines.
- **P3.3 — `fla`'s "KDA baseline" in this study means the mainline Triton path.** Anything
  else is a different, undocumented arm, and a CUTLASS arm would make the comparison
  "TileLang-style compiler vs vendor library", which is a different study (and one 2607.04454
  already did, on A100).

**There are three more silent switches on the baseline, all orthogonal to architecture:**

- **P3.4 — `FLA_USE_FAST_OPS`.** `fla/ops/utils/op.py:27-48`. Default `'0'` uses
  `tl.exp` / `tl.math.exp2` / `tl.log` / `tl.log2` / libdevice `tanh`; `'1'` uses
  `fast_expf` / `exp2` / `fast_logf` / `fast_log2f` / `fast_tanhf`. This switches the
  precision of KDA's dominant nonlinearity. It must be **absent or `'0'`**, recorded, and a
  run with `'1'` is a different operator and is not comparable to anything.
- **P3.5 — `FLA_USE_TMA` and `IS_TMA_SUPPORTED`.** `fla/utils/_device.py:168-178`: TMA
  requires CC ≥ 9 **and** `os.environ['FLA_USE_TMA'] == '1'`. On sm_90+ this silently changes
  the load path. Record it; the study's natural cell is without it.
- **P3.6 — the Triton version gates the algorithm, not just the codegen.**
  `IS_GATHER_SUPPORTED = hasattr(triton.language, 'gather')` (`fla/utils/_device.py:163`),
  consumed at `fla/ops/kda/chunk_intra.py:859` as `USE_GATHER=IS_GATHER_SUPPORTED`. On a
  Triton without `tl.gather`, the diagonal-block kernel takes a different path
  (`fla/ops/utils/op.py:79-89` makes `gather` return `None` to keep the compiler happy).
  A Triton-version difference can therefore change the *algorithm* on the baseline side.
  Pin `triton.__version__` in the cell label. `fla/utils/_device.py:46-54` warns below
  3.3.0; this study should use the version `fla` itself recommends, ≥ 3.3.0.

### 3.5 Gate 4 — the whole validity question is settled on the device, at run time

Gates 1 and 2 are static claims about a card. Neither can be trusted from a datasheet — §3.3
just produced a counterexample to exactly that kind of reasoning. So the decision is made by
**three measured reads on the actual device, before any timing**, appended to `PLAN.md` R3
step 1 (which already reads `cudaDeviceProp`; add one field to the same call):

- **R3.6a (new) — `cudaDeviceProp.sharedMemoryPerBlockOptin`.** Feed it to `fla`'s own
  predicates: `check_shared_mem()` and `check_shared_mem('ampere')` are already callable in
  the harness (`fla/utils._device.py:227-233`). Print the *resolved* `num_stages` set, the
  resolved `BK_LIST`/`BV_LIST` from `fla/ops/gla/chunk.py:21-22`, and the resolved
  `num_stages`/`BV` from `fla/ops/common/chunk_delta_h.py:52-53`. **Compare those lists
  against `main`'s values above.** Equality is Gate 2 passed. A difference is a `fla` change
  and is caught here rather than at 3am on a rented A100.
- **R3.6b (new) — `IS_TF32_SUPPORTED`, `IS_NVIDIA_HOPPER`, `IS_NVIDIA_BLACKWELL`,
  `IS_GATHER_SUPPORTED`, `IS_TMA_SUPPORTED`, `SOLVE_TRIL_DOT_PRECISION`,
  `TRITON_F32_DEFAULT`, `FLA_USE_FAST_OPS`, `FLA_FLASH_KDA`, `FLA_TILELANG`,
  `FLA_DISABLE_BACKEND_DISPATCH`.** All ten are importable or readable; print all ten into the
  run log. This is 15 lines of code and it converts five invisible configuration variables
  into five recorded ones.
- **R3.6c (new) — instruction class actually emitted, per side, per stage.** Not the
  datasheet: the generated code. `nvdisasm` both sides' cubins and count, per kernel,
  `mma.sync` / `HMMA` / `wmma` / `IMMA` / `fma.rn.f32` / `ex2.approx.f32`. Gate 1 passes for
  a stage only if the two sides' counts of *tensor-core* instructions are both > 0 (if the
  class is available) or both = 0 **and the stage has been declared FMA-matched** (if it is
  not). **Both zero is not automatically a pass** — on sm_75 both sides are FMA-matched by
  accident, and that is why sm_75 is INVALID anyway. This read is also what makes H1 falsifiable
  on the same run: `R7` already captures whether `wmma`/`mma.sync` appears in TinyGrad's
  source under the `TC_OPT` sweep, and `kda/kda_vs_fla.py:1092` already greps for
  `("mma.sync", "wmma", "mma.", "HMMA")`. Extend that grep to the baseline side and to SASS.

### 3.6 The architecture list, and the verdict per architecture

| arch | card | Gate 1 (instr class) | Gate 2 (autotune space) | **(a) valid?** |
|---|---|---|---|---|
| sm_60 | P100 (Kaggle) | **FAIL** — no MMA instruction in the ISA at all; no MMA witness exists for sm_60 in `portability/ptx/` | n/a | **INVALID.** No bf16 MMA, no tf32, `get_cuda("sm_60") == []`. Both sides are FMA-only *by hardware*, so a ratio here measures two FMA implementations, neither of which is the kernel the authors wrote. |
| sm_70 | V100 | **FAIL** for bf16 — Volta's only MMA is `mma.m8n8k4` f16 (`portability/ptx/mma_f16_m8n8k4_f16acc@sm_70.ptx`); `fla` is written bf16, so bf16 `tl.dot` has no MMA to reach | `check_shared_mem()` on 98,304 B opt-in < 102,400 → **FAIL** | **INVALID.** Separately: this is the card 2012.07145's 0.95× was measured on (`PARITY-SCOPE.md` §1.1), and quoting parity from it onto KDA is the misreading `PARITY-SCOPE.md` §5.1 exists to prevent. |
| sm_75 | T4 (Kaggle) | **FAIL** — no bf16 MMA; `emuTF32=False` forces FMA dots; `fla` sets `TRITON_F32_DEFAULT=ieee` process-globally; `SOLVE_TRIL_DOT_PRECISION='ieee'` at 12 sites | **FAIL** — 65,536 B opt-in < 102,400, so `BK_LIST=[16,32]`, `BV_LIST=[16,32]`, `num_stages ∈ {2,1}`, `BV=32` | **INVALID as the headline.** Usable only as a clearly-labelled secondary cell with all four degradations named in the caption, and even then the row is about `fla`-on-Turing, not `fla`. |
| sm_80 | A100 / A30 / GA10x | **PASS** — bf16 `mma.sync.m16n8k16` and native tf32 both in `cuda_sm80` and in `portability/ptx/mma_{bf16,f16}_m16n8k16_*@sm_80.ptx` | **PASS** — 166,912 B opt-in ≥ `AMPERE` and ≥ `DEFAULT`, so all four config lists are full | **VALID.** The reference cell. |
| sm_86 | RTX 3090 | **PASS** (Ada/GA102 class, bf16 MMA present) | **FAIL** — consumer smem ≈ 100 KB < 102,400 | **INVALID** by Gate 2 despite passing Gate 1. |
| sm_89 | RTX 4090 / L40S | **PASS** (`cuda_sm89`, bf16 MMA present) | **FAIL** — 101,376 B is below `DEFAULT = 102,400`, so *both* predicates fail | **INVALID** by Gate 2 despite passing Gate 1. This is the counterexample that makes Gate 2 necessary. |
| sm_90 | H100 | **PASS** — plus `wgmma_*@sm_90a`, `mbarrier_*@sm_90`, `tma_*@sm_90` in `portability/ptx/` | **PASS** — 232,448 B ≥ every threshold. (Note `chunk_delta_h.py:25` *narrows* `NUM_WARPS` to `[2,4]` on Hopper — a Hopper-specific choice, recorded, not a degradation.) | **VALID.** The strongest cell, and the one where the `FLA_USE_TMA` / `IS_TMA_SUPPORTED` switch (P3.5) must be pinned. |
| sm_89 L40S / A40 | — | PASS | A40 is Ada-class, 101,376 B → **FAIL** | **INVALID** by Gate 2. Worth naming because a "datacenter-looking" rental SKU can fail. |
| sm_100 | B100/B200 | **PASS** — `tcgen05_mma_f16kind@sm_100a` in `portability/ptx/` | **PASS** (227 KB) | **VALID with a caveat.** `fla/ops/utils/op.py:51-72`: on Blackwell `safe_dot` wraps `tl.dot` in inline PTX (`mov.f32 $0, $1`) to work around a `TritonGPUHoistTMEMAlloc` defect — fla issue #638, triton-lang/triton#8695, "TODO: Remove this workaround once the Triton compiler bug is fixed." The sm_100 baseline is a **patched** kernel whose generated code is not what the source reads like. Also `fla/utils/_device.py:156-159` carves out SM100 and SM120 separately. Report, do not lead with it. |

**The list: comparison (a) is valid on sm_80 (GA10x-class) and sm_90 (H100).** Nothing below
sm_80. Nothing in the sm_80–sm_89 consumer band, despite those cards having better tensor
cores than the sm_80 the gate is keyed on.

**Purchasing consequence, and it is not obvious:** a valid cell requires
`sharedMemoryPerBlockOptin ≥ 166,912`, which means **a GA10x datacentre part (A100-class) or
a Hopper/Blackwell part**. Consumer Ampere, consumer Ada, and the A40/L40S line all fail
Gate 2 despite being advertised as datacenter cards. If the study rents, it must rent one of
those and must read `sharedMemoryPerBlockOptin` **before** paying, not after.

### 3.7 Correction to a claim already in the repo

`PLAN.md` §6 and `HYPOTHESES.md` §6 both assert *"Triton requires CC ≥ 7.0, so sm_60 is out
for the entire Triton corpus."* **I could not verify this in current Triton.** I read
`third_party/nvidia/backend/compiler.py` and `.../driver.py` on `main` (2026-09-27) and
found **no minimum-capability assertion**; `GPUTarget("cuda", capability, warp_size)` is
constructed from whatever `torch.cuda.get_device_capability` returns, and
`supports_target(target)` is `target.backend == 'cuda'`. The claim may have been true of older
Triton.

**The P100 exclusion does not depend on it and should be re-stated on the architecture-level
ground**, which is airtight: Pascal GP100 has **no MMA instruction in the ISA**, so there is no
possible version of the comparison in which `fla`'s KDA runs as written. Mark the Triton
version claim **[could not verify]** in `PLAN.md` §6 rather than leaving it as a load-bearing
premise. (`HYPOTHESES.md` R1's "Triton fails on sm_60 (expected — Triton requires CC ≥ 7.0)"
is a *test*, so it is unaffected; only the parenthetical reason is.)

---

## 4. dtype, precision, and accumulation behaviour

**Rule: "the same maths" does not mean the same bits, and a bitwise-divergent baseline is a
measurement hazard to be checked and reported, not a footnote.**

The KDA graph contains the operator's dominant nonlinearity (`exp2` on the gate, thousands of
times per call) and a *triangular solve*. Both are precision-critical: a gated recurrence
compounds per-token error through the state, so a 1-ULP difference in the decay does not stay
1 ULP — it is amplified by the recurrence's own conditioning. `HYPOTHESES.md` H2 already runs
KDA in **fp32 first** for a related reason (the 33.55 MB bf16 decay tensor can be absorbed by
L2 and produce a number that looks like a refutation when it is an artifact). The same logic
applies to precision.

### 4.1 Matched precision is mandatory, and here is the list of the mismatches that exist

| # | item | TinyGrad 0.14.0 | `fla` / Triton | status |
|---|---|---|---|---|
| P1 | working dtype | **fp32** (`kda/tiny_kda.py:44`, `F = dtypes.float32`) | **bf16 intended** — `FlashKDABackend.chunk_kda_verifier` returns `False, f"FlashKDA requires bfloat16, got {q.dtype}"` | **mismatch by default.** `PLAN.md` C3 already forbids "a fp32-vs-bf16 ratio" and requires either both arms in one dtype or an explicit `confounded` label. |
| P2 | accumulate dtype | fp32 (there is no narrow-accumulate path in the port) | fp32 accumulate for the bf16/f16 MMA | match **if** both run bf16 on sm_80+ |
| P3 | the UT solve's precision | fp32 throughout; the port's forward substitution is exact-arithmetic on a unit-diagonal system ("A has unit diagonal, so there is no division", `kda/tiny_kda.py:177`) | `tf32` on sm_80+, **`ieee` on sm_75** (`chunk_intra.py:21-24`), at 12 sites | **mismatch on sm_75 and a deliberate precision *drop* on sm_80+.** In (a) the solve must be run at matched precision or the stage is void. |
| P4 | `exp2` | `renderer/cstyle.py:423` — `Ops.EXP2: lambda x,dtype: f"hexp2({x})" if dtype in (half, bfloat16) else f"exp2({x})"`. fp32 → `exp2(x)` → NVRTC | `fla/ops/utils/op.py:42` — `tl.math.exp2(x.to(tl.float32))`, or `exp2`/`fast_expf` family under `FLA_USE_FAST_OPS=1` (P3.4) | **unknown divergence.** "The same function name" does not mean the same implementation: `tl.math.exp2` routes to libdevice `__nv_exp2f` (accurate) while an `exp2f` route lowers to `ex2.approx.f32` (≤2 ULP). **Must be checked on the device, not assumed** — see P4.1. |
| P5 | division | `mixin/elementwise.py:251` — `d = a.alu(Ops.MUL, b.reciprocal())`; `renderer/cstyle.py:137` — `Ops.RECIPROCAL: (1/{x})`. So `a/b` emits `a*(1.0f/b)`, and the result is 1 ULP off the correctly rounded quotient in 25.9%–27.0% of general cases on a divisor-independent draw (`derivation/divide.py:mechanism_sentence`). `renderer/cstyle.py:274`'s `Ops.FDIV: ({a}/{b})` is **not** dead code: it is the load-bearing last line of the CPU repair, and `codegen/decomp/op.py:133-136` manufactures the node `elementwise.py:251` never emitted (`renderer/cstyle.py:271` deletes `RECIPROCAL` from the inherited map). CPU is exactly this repair, which is why it measures 0 wrong | Triton emits a real `div` / `fdiv` | **present in the language, and the repair is reachable — but only on the backend whose op set contains `Ops.FDIV`.** The KDA port's only comment on the matter is `kda/tiny_kda.py:177`: "A has unit diagonal, so there is no division." **So the `a*(1/b)` rewrite does not currently touch the KDA dataflow.** Stated because the study has already been bitten twice by misdescribing this path: by attributing the rewrite to the renderer rather than the Tensor API (`HYPOTHESES.md` H10), and by calling `Ops.FDIV` dead code (ledger **C-12**). If any stage later introduces a `/` (a stable triangular solve would), P5 becomes live and the hazard is real. |
| P6 | FMA contraction | **on, with no user-facing knob.** `runtime/support/compiler_cuda.py:48` — `self.compile_options = [f'--gpu-architecture={arch}']`, with the only other additions being `-I<cuda>/include` and `--minimal` on CUDA ≥ 12.4. No `--fmad=false`, no `--use_fast_math`, no `--prec-div=false`. NVRTC's default is `fmad=true`. | Triton contracts; `TRITON_F32_DEFAULT` is forced to `'ieee'` on sm_75 (`_device.py:180-183`) which changes dot precision but **not** FMA contraction | **both contract; neither exposes a knob.** Therefore the *contraction pattern* is a free variable of the measurement and must be controlled structurally: see P6.1. |

### 4.2 The four required checks (all device-side, all pre-timing)

- **P4.1 — ULP probe of the two `exp2` implementations.** Build a 1-D input of
  ~2^20 values spanning the range the gate actually occupies (`g ≤ 0` per `kda/SPEC.md` §2,
  so `x = RCP_LN2·g` lands in `(-∞, 0]`), evaluate `exp2(x)` on both sides, and report
   - `max |ULP_tinygrad − ULP_triton|` and the **fraction of elements that differ at all**,
   - and separately, the fraction differing by **more than 2 ULP**.
  Threshold, pre-declared: **if more than 1% of elements differ in any way, or any element
  differs by more than 2 ULP, the comparison is labelled `PRECISION-CONFOUNDED`** and the
  cell's caption must say so. A recurrence compounds this; a 1-ULP gate difference is not a
  1-ULP output difference.
- **P4.2 — Contraction control by structure, not by flag.** Since no flag exists (P6), the
  only way to make contraction comparable is to make the two programs' contraction
  *patterns* the same: both must read the same inputs into the same accumulator in the same
  order at the contraction sites. Operational rule for (a) at stage granularity: **at every
  (a)-admissible stage, the two implementations must be given operands in the same order and
  must not be reassociated by either compiler.** If a stage fails this, it drops to (b) and
  the ratio moves to the (b) table. Record the SASS `ffma`/`fadd`/`fsub` counts per kernel on
  both sides (same disassembly pass as R3.6c) — a large asymmetry is a *finding about
  contraction*, reported as such, not absorbed into the ratio.
- **P4.3 — Bitwise-divergence report, mandatory and always published.** For every arm, report
  the max/mean ULP distance and the fraction of differing elements of the **final output**
  against `kda/ref.py`'s float64 reference, at the exact shape timed. This is the
  same measurement `HYPOTHESES.md` R8.1 already gates on for correctness (≤ 1e-5) but at
  **ULP granularity**, because ≤ 1e-5 on a K=64, T=4096, 4-head output is a very loose gate
  that a 300×-slow kernel passes — which is 2607.04454's entire methodological point
  (`HYPOTHESES.md` header, R8.1).
- **P4.4 — No ULP gate on the timing path.** If the two arms' ULP distances differ by more
  than a factor of 2 from each other, the cell is **not** used for a performance claim; it is
  used only for the precision claim. This is the direction that *helps* the study, and it is
  the direction people skip.

### 4.3 What the precision protocol forbids

- **A ratio between different working dtypes.** `PLAN.md` C3: "**Never** a fp32-vs-bf16
  ratio." Unchanged, and reinforced: on sm_75, bf16 is the only dtype with any MMA, and there
  is none, so bf16-on-T4 buys *nothing* and costs precision.
- **A ratio reported without the accumulate dtype.** "bf16" is not a precision; "bf16 in,
  fp32 accumulate" and "bf16 in, bf16 accumulate" differ by 8× of mantissa.
- **A ratio between the `ieee` and `tf32` solve.** `SOLVE_TRIL_DOT_PRECISION` is a
  `tl.constexpr` fixed at import time by the device's capability, so on an sm_75 host the
  Triton arm is *permanently* `ieee` while an A100 host is `tf32`. If the same stage is timed
  on two cards, the two ratios are not comparable and must not be differenced.
- **A `FLA_USE_FAST_OPS=1` cell appearing in any table alongside a `=0` cell.**

---

## 5. The controls that make the result credible

`HYPOTHESES.md` §6.1 already states that an un-tuned baseline is inadmissible. This section
is the checklist that makes that operable, plus the two controls the repo does not yet have.

### 5.1 The baseline is the best available human version

- **C1 — Config sweep, and it is not optional.** The hand-written side must be tuned over the
  knobs **the architecture actually has**. This has a specific consequence from §3.3: the
  sweep is over the **resolved** config lists, and the resolved lists must be printed first
  (R3.6a). On an A100 the KDA-forward families autotune over `num_stages ∈ {2,3,4}`,
  `BK ∈ {32,64}`, `BV ∈ {32,64}` (`chunk_delta_h.py`) and `BV_o ∈ {64,128}` (`gla/chunk.py:22`)
  — plus `chunk_intra.py:36-38`'s `BK ∈ {32,64} × num_warps ∈ {1,2,4}`. The reported number is
  the **minimum over the full resolved grid**, and the grid size and the argmin are published
  next to it. A ratio against a single hand-picked config is a ratio against a guess.
- **C2 — Both arms get a sweep, and the compiler's is its own default.** TinyGrad's search is
  `BEAM=0` by default with `TC_OPT=0`/`TC_SELECT=-1`. The compiler arm is run at **its shipped
  defaults**, because a compiler measured with its autotuner disabled measures the wrong
  compiler. The `TC_OPT` sweep is a **separate experiment** (it is `PLAN.md` R7 / H1), and its
  result is reported in the (b) table as the evidence for the (ii)/(iii) term — it is never
  used to shrink the (a) gap. If TinyGrad's `BEAM=1` sweep closes part of the gap, that
  closes it in the (a) table and that is a legitimate result; but the headline (a) number uses
  defaults, with the swept number published beside it.
- **C3 — Same author, same session, same effort budget, when the human side is a re-
  implementation.** `PLAN.md` R1 already requires this for the sm_75 case. It extends to the
  Triton re-expression of the port's own algorithm (§2.1): same session, same effort, and the
  effort is written down (hours, and what was tried).
- **C4 — The oracle is shared.** Both arms validate against the **same** `kda/ref.py`
  float64 reference with the **same** tolerance, at the **same** shape, before any timing
  (`PLAN.md` R8.1). `kda/ref.py` has three independent code paths (recurrent / chunked /
  closed-form) and `kda/test_ref.py` is 17/17 — it is a better oracle than either side
  deserves, and using a *different* oracle per arm would be the cheapest possible way to
  launder a gap.

### 5.2 Reproducing the expert's own numbers

- **C5 — Two reproductions are required, and they are different things.**
  1. **The expert's operator-level numbers.** 2406.06484 §3.2 reports a speed comparison for
     the chunkwise-vs-recurrent contrast, and **its magnitude is not quotable** — the figure
     is a chart that could not be read numerically (`PARITY-SCOPE.md` §4.3). KDA's own
     published speed is on A100/H800-class hardware (2510.26692), and
     `HYPOTHESES.md` H6 already names the analogous trap: "netherite's published 396 M
     rays/s is on **their box, not a T4** … The ratio against 396 M rays/s is **not a valid
     comparison** and must not be reported."
     **Therefore: the study may not compare its `fla` number to any published `fla` number,
     and `PARITY-SCOPE.md` §4.3's "up to 30×" must not appear in the paper.** What may be
     done is: re-measure `fla` on the study's card and **report that number as the baseline,
     with no cross-card ratio anywhere**.
  2. **`fla`'s own regression suite on the study's card.** This is the reproduction that is
     actually available and it is the one that matters: run `fla`'s KDA tests on the exact
     card, exact dtype, and exact Triton version the study will use. If they pass there, `fla`
     is behaving as its authors intend **on that card** — which is the real question behind
     "is the baseline degraded". If they do not pass, or are skipped, that is reported in the
     cell caption.
- **C6 — If the expert's published numbers cannot be reproduced, the pre-declared response.**
  Since the magnitudes are not quotable (§C5.1), the protocol is:
  - **Never substitute a published number for a measured one, and never form a ratio across
    cards.** `PLAN.md` §6.2 already forbids this for the camera figure; it generalises.
  - **Report the reproduction attempt, including its failure mode, as a finding.** A
    published Triton kernel that needs a workaround to run on the study's own hardware is
    evidence *for* the study's thesis, not a reason to hide the attempt — and the mechanism
    (arch, dtype, Triton version, dispatch backend) is exactly the content of §3.
  - **The study's claim degrades in scope, not in confidence.** "We could not reproduce the
    expert's reported configuration on available hardware" is a publishable sentence. "The
    expert's kernel is `r×` faster" is not available, and never was.

### 5.3 The one control that is easiest to forget

- **C7 — The comparison is one cell of a grid the field has declared infeasible.** Say it every
  time, in the caption, with the version-indexed tuple. 2607.04454 §VI-D is the reason: a
  correct, hand-optimised kernel ran **582% of PyTorch on A100 and 1.0% on GH200**, with **no
  correctness signal to warn you** (HYPOTHESES §4 C9; `PLAN.md` §6.3). With §3's gates in
  place, `fla` on sm_80 vs sm_90 is precisely such a pair. **Any A100 result is a portability
  check on the H100 result, not an independent measurement, and is labelled that way.**

---

## 6. Host-side overhead: the pre-declared treatment

### 6.1 The problem, in the numbers the study already has

Two separate measurements exist and they are on different hardware:

- **Laptop, warm, one shape** (`notes/03-kda.md` §2; `HYPOTHESES.md` H5): total warm
  **10,604.7 ms**, of which Python graph construction **1.78 s (16%)**, **TinyGrad's scheduler
  9.01 s (85%)**, GPU execution **0.109 s (1%)**. Cold, first call: 49.5 s including
  compilation.
- **The brief's shape** (85% of end-to-end in the Python scheduler, **0.58 s of 33.9 s** in
  device execution, i.e. 1.7%). *Cited as given in the task brief; I did not locate this pair
  in the repo and it is not reproduced by `HYPOTHESES.md` H5's 10.6 s/0.109 s breakdown. It is
  a different shape on different hardware and is treated as a second independent instance of
  the same phenomenon, not as a contradiction.*

Both agree on the ratio and neither is in dispute. The protocol is about what may be done with
it.

### 6.2 The rule

> **The comparison reports device time. It does not report end-to-end wall time. Where
> end-to-end time is reported, it is reported as a separate, separately-labelled row with the
> host recorded, and it is never differenced against a device-time row or against the
> baseline's wall time.**

**A number that mixes Python and GPU measures Python.** TinyGrad's host-side cost is a real
and separately-interesting finding (H5, tag (iii), 5.6× floor / 97× ceiling) but it is a
finding about *tinygrad's API*, and folding it into a codegen ratio would (i) inflate the gap
by a factor of ~59 on the laptop shape, (ii) make the ratio a measurement of the author's
process, and (iii) be **structurally unfair to the baseline**, whose host cost is the Triton
autotuner's, not a Python rewrite pipeline. That last point is the one that matters: a
wall-clock comparison pits a Python scheduler that re-plans a 12,090-UOp graph on every call
against a C++ autotuner that does not re-plan at all. It would produce a large, true,
publishable, and **entirely irrelevant** number. `PLAN.md` C1's "**No cross-process timing
comparison anywhere**" is the same rule applied to processes; this applies it to the host/device
split.

### 6.3 How the split is measured — `PLAN.md` R9's four cells, unchanged and re-purposed

`PLAN.md` R9 already separates exactly these quantities and its `PLAN.md` §4.1 already forbids
comparing across columns. This file adds one rule and one cell:

| cell | what | role here |
|---|---|---|
| `GPU` | device time from Nsight Compute (`launch__*`, `gpu__time_duration.sum`) | **the only cell the comparison may use** |
| `COLD` | first call, fresh process, empty JIT cache | reports the one-time compile cost, per side, separately |
| `WARM-REBUILD` | graph built inside the timed region, warm process | **the honest end-to-end for a first-time integrator**; report it as its own row |
| `WARM-ASSIGN` | realized graph replayed with new inputs (`Tensor.assign`) | the ceiling a plan cache could reach (H5) |
| **`WARM-DIRECT`** (new) | **TinyGrad's per-kernel device time summed over the same launch set as `GPU`, with the host time excluded by construction** | the cell that makes the comparison safe to state as a ratio — it is `GPU` re-expressed in the same units as the baseline's timing so no reader has to do the subtraction |

**Host separation, in order of preference:**
1. **Nsight Compute**, per-kernel durations, summed. Primary. `HYPOTHESES.md` R8.3 already
   mandates it and already rejects `DEBUG=2` ("a device-side timer and is itself a variable").
2. **CUDA events** around the launch sequence, with the graph already realized and the launch
   list already materialised — a *replay* of the same kernels. This is the cross-check, and it
   is the only way to get a device-side number without profiling overhead on 12,090 launches.
3. **A subtraction** (`wall − scheduler`) — **forbidden as a primary method.** Profiling
   overhead and CPU-side launch cost are not separable by subtraction, and
   `HYPOTHESES.md` R8.3's objection to `DEBUG=2` is the same objection at a different layer.

**The asymmetry trap, named.** TinyGrad launches **12,090 kernels**; `fla`'s KDA forward
launches on the order of ten. Even with device time measured on both sides, the two arms have
wildly different *launch* counts, so a wall-clock number would also be measuring launch
overhead — by a factor of ~1000 in the TinyGrad arm's disfavour. This is the same mechanism
`notes/04-where-compilers-stop.md` §"layer decomposition" warns about ("tinygrad has the best
execution speed of any framework" is a **launch** claim and "is almost never the bottleneck"
is a claim about a regime TinyGrad's 12,090-kernel graph is not in). **Kernel count is
therefore reported in every cell label**, and a per-kernel-mean row (`WARM-DIRECT / kernels`)
is published beside the total, so that a reader can see the launch-count asymmetry instead of
inheriting it invisibly.

### 6.4 The one sentence the paper may use

> Device time is reported. End-to-end time for the generated path additionally carries
> 85% Python-scheduler overhead on a 12,090-UOp graph (`PLAN.md` R9); that overhead is a
> property of the API and is excluded, because the baseline's host cost is an autotuner cache
> hit, and a wall-clock ratio would measure the difference between the two host-side
> architectures rather than the two GPU codegens.

---

## 7. The table the paper should present

### Table 1 — Comparison (a), same-algorithm, stage-granular. **The codegen table.**

Each row is one stage at which the two programs materialise the same tensor. `valid?` is
`PASS` / `FAIL` per gate, and **a `FAIL` in any gate cell means the row is not printed with a
ratio** — it is printed with the word `INVALID` in the ratio cell.

| stage (both same tensor) | dtype | accumulate | TinyGrad: kernels · MMA count · device ms | `fla` Triton: kernels · MMA count · device ms | `r` = tg/fla | `r_noise` | valid? (G1 instr / G2 cfg / G3 backend / G4 observed) | 95% CI | Wilcoxon p | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 `g_cum` cumsum | fp32 | fp32 | | | | | | | | |
| 3 `Btri` | fp32 | fp32 | | | | | | | | |
| 5 `w = M(Γ·k)` | bf16 | fp32 | | | | | | | | |
| 6 `u = Mv` | bf16 | fp32 | | | | | | | | |
| 7a `v_new` | bf16 | fp32 | | | | | | | | |
| 7c `o` | bf16 | fp32 | | | | | | | | | |
| 7d `S` update | bf16 | fp32 | | | | | | | | | |
| *(row repeated per (arch, dtype, Triton version, fla commit, tinygrad version) cell)* | | | | | | | | | | |

**Column semantics, fixed:**
- `MMA count` = tensor-core instructions per kernel in the SASS (`nvdisasm`), not a flag and
  not a datasheet. `0 / 0` is `FMA-MATCHED`, which is a pass only if the row is explicitly
  declared FMA-matched *and* arch is sm_80+; it is not a pass on sm_75.
- `r_noise` = the study-wide floor from `PLAN.md` §5.2, quoted in every row.
- `valid?` has four sub-cells so a reader can see **which** gate failed. A blank ratio with a
  non-blank gate column is the intended and only permitted way to report an invalid cell.
- The last row of every block is a **µ`-only`** variant of the dtype rows, because
  `HYPOTHESES.md` §4.2 notes the 2012.07145 experts "used the best runtimes found by the
  autoscheduler as a target to beat" and the same anchoring question applies here.

### Table 2 — Comparison (b), algorithm-choice. **The expressiveness table.**

| term | what the compiler is given | what the expert did | the change | magnitude | how measured | tag |
|---|---|---|---|---|---|---|
| B1 | chunked UT-transform with a `[B,NT,HV,C,C,K]` per-channel decay, `C=64` | same, at `BC=16`, in registers (`chunk_intra.py:825`) | intermediate granularity: 64× on that tensor | `H2`'s 13.0× arithmetic, to be measured via `dram__bytes.sum` | Nsight `dram__bytes.sum` under the pre-registered exp-folding mitigation (`PLAN.md` §5.4) | **(i)** CANNOT EXPRESS |
| B2 | unrolled forward substitution, `C(C−1)/2 = 2,016` dependent steps (`tiny_kda.py:178-194`) | 12 `tl.dot`s forming the block-triangular/Neumann series on `[BC,BC]` (`chunk_intra.py:320-355`) | algorithm *for the subproblem*: 2,016 scalar-substitution steps → 12 blocked matmuls | `H3`'s kernel-count prediction, plus its own time term | kernel count vs `C` (`PLAN.md` R6) and device ms | **(i)** CANNOT EXPRESS |
| B3 | the same graph, with `OptOps.TC` in the search space | MMA on every `tl.dot` | tensor-core selection | `H1`'s predicted 1.00–1.05× at this intensity, i.e. **predicted to be nothing** | `TC_OPT ∈ {0,1,2}` source/SASS capture (`PLAN.md` R7) | **(ii)** CAN EXPRESS, DOES NOT FIND — *if* a WMMA appears; **(iv)** if not |
| B4 | | | not-yet-written fusion/tiling pass | | | **(iii)** |
| B5 | | | no arch/dtype entry in `get_cuda` | | `cuda_sm60 = []` | **(iv)** |
| | | | **total (b)** | **not a codegen number** | | |

**Table 2's total row is mandatory and must carry the sentence "not a codegen number" in the
cell.** A paper that prints `total (b) = 13×` and a paper that prints `total (a) = 1.4×` in
the same table has made the mistake this file exists to prevent. Two tables, never one.

### Table 3 — Validity matrix, one row per architecture, printed regardless of outcome

| arch | card(s) | G1 instr class (bf16 MMA / tf32) | opt-in smem | G2 resolved cfg lists (vs `main`) | G3 `fla` backend that ran | G4 MMA observed, both sides | (a) verdict | (b) usable? |
|---|---|---|---|---|---|---|---|---|
| sm_60 | P100 | none in ISA | — | — | — | — | **INVALID** | yes, as a raw-work statement only |
| sm_70 | V100 | f16 `m8n8k4` only, no bf16 | 98,304 | **narrowed** | — | — | **INVALID** | yes |
| sm_75 | T4 | none (bf16); tf32 emulated off | 65,536 | **narrowed** — `num_stages ∈ {2,1}`, `BV=32`, `BK ∈ {16,32}`, `BV_o ∈ {16,32}` | — | — | **INVALID** | yes, **with all four degradations in the caption** |
| sm_80 | A100 / A30 / GA10x | bf16 `m16n8k16` ✓, tf32 ✓ | 166,912 | **full** | pin per P3.1 | required | **VALID** | yes |
| sm_86 | RTX 3090 | bf16 ✓ | ~100 KB | **narrowed** | — | — | **INVALID** (G2) | yes |
| sm_89 | RTX 4090 / L40S / A40 | bf16 ✓ | 101,376 | **narrowed** — below `DEFAULT=102,400`, so both predicates fail | — | — | **INVALID** (G2) | yes |
| sm_90 | H100 | bf16 ✓ + `wgmma`/`mbarrier`/`tma` | 232,448 | **full** | pin `FLA_USE_TMA` | required | **VALID** | yes |
| sm_100 | B100/B200 | bf16 ✓ + `tcgen05` | 227 KB | **full** | note `safe_dot` PTX patch (issue #638) | required | **VALID, caveated** | yes |

This table is the one that makes the study's negative results legible. A reader who sees only
Table 1 and two `VALID` rows learns nothing; a reader who sees Table 3 learns that **five of
seven plausible architectures cannot host this comparison**, and that the reason for three of
them is a shared-memory constant, not compute power.

### Tables 4 and 5 — the two small tables that must accompany them

**Table 4 — host/device split** (`PLAN.md` R9, one row per arm, no cross-arm subtraction):

| arm | cell | wall ms | scheduler ms | device ms (Nsight) | kernels | device µs/kernel | host |
|---|---|---|---|---|---|---|---|
| TinyGrad 0.14.0 | COLD | | | | | | |
| | WARM-REBUILD | | | | | | |
| | WARM-ASSIGN | | | | | | |
| | WARM-DIRECT | | | | | | |
| | `GPU` | — | — | | | — |
| `fla` @ `<commit>` / Triton `<ver>` | autotune warmup (excluded) | | | | | |
| | WARM-DIRECT | | — | | | |
| | `GPU` | — | — | | | — |

**Table 5 — precision divergence** (§4.2, published even when it is zero, especially when it
is zero):

| quantity | TinyGrad | `fla`/Triton | ULP distance (max / mean) | fraction differing | verdict |
|---|---|---|---|---|---|
| `exp2` on the gate's actual range | | | | | |
| accumulate dtype, gram | | | | | |
| accumulate dtype, UT solve | fp32 | `tf32` on sm_80+ / `ieee` on sm_75 | | | |
| final output vs `kda/ref.py` float64 | | | | | |
| `SASS` `ffma`+`fadd`+`fsub` per kernel | | | | | |
| FMA contraction knob available? | **no** (`compiler_cuda.py:48`) | `TRITON_F32_DEFAULT` only (sm_75) | — | — | — |

---

## 8. Verification log

All source reads made 2026-09-27. Line numbers are from the revision named.

### 8.1 `flash-linear-attention`, `main` branch, via `raw.githubusercontent.com/fla-org/flash-linear-attention/main/`

| what | where | how verified |
|---|---|---|
| `SOLVE_TRIL_DOT_PRECISION = 'tf32' if IS_TF32_SUPPORTED else 'ieee'` | `fla/ops/kda/chunk_intra.py:21-24` | full file read |
| 12 `tl.dot` sites using that constexpr, in the fused inter+solve kernel | `fla/ops/kda/chunk_intra.py:320-355` | line-by-line |
| `IS_TF32_SUPPORTED = IS_NVIDIA and capability[0] >= 8` | `fla/utils/_device.py:162` | full file read |
| `os.environ['TRITON_F32_DEFAULT'] = 'ieee'` on old NVIDIA, with fla's own comment "This is a workaround for old nvidia card" | `fla/utils/_device.py:180-183` | full file read |
| `IS_GATHER_SUPPORTED = hasattr(triton.language, 'gather')` | `fla/utils/_device.py:163` | full file read |
| `IS_TMA_SUPPORTED` requires CC ≥ 9 **and** `FLA_USE_TMA=1` | `fla/utils/_device.py:168-178` | full file read |
| `IS_NVIDIA_HOPPER` / `_BLACKWELL` / `_SM100` / `_SM120` predicates | `fla/utils/_device.py:150-159` | full file read |
| `Backend` shared-memory thresholds: `ADA=101376`, `AMPERE=166912`, `HOPPER=232448`, `DEFAULT=102400`; `check_shared_mem(arch="none")` compares `max_shared_mem` against `DEFAULT` | `fla/utils/_device.py:201-233` | full file read |
| `num_stages in ([2,3,4] if check_shared_mem('ampere') else [2,1])`; `BV in ([32,64] if check_shared_mem('ada') else [32])` | `fla/ops/common/chunk_delta_h.py:52-53` | full file read |
| `NUM_WARPS = [2,4] if IS_NVIDIA_HOPPER else [2,4,8,16]` | `fla/ops/common/chunk_delta_h.py:25` | full file read |
| `BK_LIST = [32,64] if check_shared_mem() else [16,32]`; `BV_LIST = [64,128] if check_shared_mem('ampere') else [16,32]` | `fla/ops/gla/chunk.py:21-22` | full file read |
| KDA fwd is a composition of four families: `chunk_kda_fwd_intra` → `chunk_gated_delta_rule_fwd_h` → `chunk_gla_fwd_o_gk`, plus a cumsum | `fla/ops/kda/chunk_fwd.py:20-142` | full file read |
| `BC = 16` hard-coded; `NC = cdiv(BT, BC)`; `BT` restricted to 32 or 64 | `fla/ops/kda/chunk_intra.py:823-829` | full file read |
| `@dispatch('kda')` on `chunk_kda_fwd_intra` / `chunk_kda_bwd_intra` | `fla/ops/kda/chunk_intra.py:806, 914` | full file read |
| `FLA_DISABLE_BACKEND_DISPATCH=1` bypasses all dispatch; registry iterates backends by ascending `priority` and calls `verify()` | `fla/ops/backends/__init__.py:28-30, 160-220` | full file read |
| The decision is logged at `logger.info`, once per `(op, func, backend)` | `fla/ops/backends/__init__.py:206-210` | full file read |
| Registration order: `TritonAscend` (priority 0, NPU-only), `FlashKDA` (priority 3, `default_enable=True`), `KDATileLang` (priority 5) | `fla/ops/kda/backends/__init__.py:15-19` | full file read |
| `FlashKDABackend` requires: no grad, bf16, `K==128`, `V==128`, no GVA, `use_gate_in_kernel`, `use_qk_l2norm_in_kernel`, `use_beta_sigmoid_in_kernel`, `state_v_first`, no CP, `safe_gate` | `fla/ops/kda/backends/flash_kda.py:41-88` | full file read |
| It calls `flash_kda.fwd` — "Fused CUTLASS forward (replaces the multi-kernel Triton path)", © 2026 Moonshot AI, <https://github.com/MoonshotAI/FlashKDA>; disabled with `FLA_FLASH_KDA=0` | `fla/ops/kda/backends/flash_kda.py:10-12, 28-31, 115` | full file read |
| `TritonAscendKDABackend.is_available()` returns `IS_NPU` — never active on NVIDIA | `fla/ops/kda/backends/triton_ascend/__init__.py:59-62` | full file read |
| `KDATileLangBackend` implements only `chunk_kda_bwd_wy_dqkg_fused` (backward) | `fla/ops/kda/backends/tilelang/__init__.py:30-54` | full file read |
| `FLA_USE_FAST_OPS` switches `exp/exp2/log/log2/tanh` to the `fast_*` family | `fla/ops/utils/op.py:27-48` | full file read |
| `safe_dot` on Blackwell wraps `tl.dot` in inline PTX to work around `TritonGPUHoistTMEMAlloc` — fla issue #638, triton-lang/triton#8695, "TODO: Remove this workaround" | `fla/ops/utils/op.py:51-77` | full file read |
| `gather` silently returns `None` when `tl.gather` is absent; `USE_GATHER=IS_GATHER_SUPPORTED` passed to the diagonal-block kernel | `fla/ops/utils/op.py:79-89`; `fla/ops/kda/chunk_intra.py:859` | full file read |
| Triton < 3.3.0 warned against by fla itself | `fla/utils/_device.py:46-54` | full file read |

### 8.2 `triton-lang/triton`, `main` branch

| what | where | how verified |
|---|---|---|
| `emuTF32 = (capability // 10 >= 8)`; `add_f32_dot_tc(pm, emuTF32)` — fp32 `tl.dot` is FMA on sm_70/75 | `third_party/nvidia/backend/compiler.py:305, 311` | full-file grep + targeted read |
| `add_optimize_dot_operands(pm, capability >= 80)` | same file, line 320 | targeted read |
| `get_ptxas(arch)`; `sm_arch_from_capability` | same file, lines 41-42, 110-113 | targeted read |
| **No minimum compute-capability assertion.** `supports_target` is `target.backend == 'cuda'`; `GPUTarget("cuda", capability, warp_size)` built from the device query; no `>= 7` / `>= 70` check in `compiler.py` or `driver.py` | `third_party/nvidia/backend/compiler.py:178-179`; `.../driver.py:353-378` | both files read/grepped in full |
| ⇒ `PLAN.md` §6's "Triton requires CC ≥ 7.0" is **[could not verify]** in current Triton | — | — |

### 8.3 `tinygrad` 0.14.0, installed in this repo's `.venv`

| what | where | how verified |
|---|---|---|
| `def get_cuda(arch): return cuda_sm89 if (ver:=int(arch[3:])) >= 89 else cuda_sm80 if ver >= 80 else cuda_sm75 if ver >= 75 else []` | `codegen/opt/tc.py` | read in this session |
| `cuda_sm75 = cuda_8168_f16`; `cuda_sm80 = cuda_81616 + cuda_8168_f16 + cuda_8168_tf32` | `codegen/opt/tc.py` | read in this session |
| `d = a.alu(Ops.MUL, b.reciprocal())` in `__truediv__` | `mixin/elementwise.py:251` | read in this session |
| `Ops.RECIPROCAL: lambda x,dtype: f"(1/{x})"`; `Ops.FDIV: lambda a,b,dtype: f"({a}/{b})"` at line 274 | `renderer/cstyle.py:137, 274` | grep + read |
| `Ops.EXP2: hexp2(x)` for half/bf16 else `exp2(x)` | `renderer/cstyle.py:423` | grep |
| NVRTC options are **only** `--gpu-architecture`, plus `-I<cuda>/include` and `--minimal` on CUDA ≥ 12.4. No `--use_fast_math`, no `--prec-div=false`, no `--fmad=false` ⇒ FMA contraction on, unknobbed | `runtime/support/compiler_cuda.py:48-58` | read |

### 8.4 This repo

| what | where | how verified |
|---|---|---|
| MMA witnesses per arch — `mma_f16_m8n8k4@sm_70`; `mma_f16_m16n8k8_{f16,f32}acc@sm_75`; `mma_{bf16,f16}_m16n8k16*@sm_80`; `mma_s8_m16n8k16@sm_80`; `wgmma_*@sm_90a`; `tcgen05_mma_f16kind@sm_100a`. **No sm_60 witness exists** | `portability/ptx/*.ptx`, `portability/out/selfval/*.cubin` | directory listing (another agent's output, in progress) |
| Port is fp32-only; seven-step structure; `[C,C,K]` per-channel decay is 64× the paper's `[C,C]`; step 4 is an unrolled forward substitution with "A has unit diagonal, so there is no division" | `kda/tiny_kda.py:8-20, 44, 71-74, 155-194` | read |
| `kda/ref.py` has three independent code paths; `kda/test_ref.py` 17/17 | `kda/ref.py:6`; `HYPOTHESES.md` H5/U3 | read + repo statement |
| TinyGrad's own MMA-token grep already exists on the baseline side of the harness | `kda/kda_vs_fla.py:1092` | read |
| 2406.06484 §3.1 / §3.2 quotes on `O(d²)` materialisation, the parallel-scan footnote, and "fully recurrent and thus cannot use tensor cores written as is … we further leverage the *UT transform*" | via `research/PARITY-SCOPE.md` §2.3 | re-read; quotes there were taken from arXiv HTML full text |
| 2607.04454 §III-B / Table I / Fig. 1 on excluding attention because the algorithms differ; the two excluded cells 7691% and 54.3%; §VI-E's "≈2–3%" for the algorithm term on Conv2d; §VI-D's 582%-then-1.0% portability collapse | via `research/PARITY-SCOPE.md` §2.3, §3.2, §4.1, §3.3 | re-read; full-text quotes there |
| 0.95× is 17 imaging programs on one V100, its only GEMM had a 17×-smaller-than-average search space, no recurrence/attention/normalisation | via `research/PARITY-SCOPE.md` §1 | re-read |
| 85% scheduler / 16% graph build / ~1% GPU, warm 10,604.7 ms, scheduler 9.01 s, GPU 0.109 s, cold 49.5 s | `notes/03-kda.md` §2; `HYPOTHESES.md` H5 | read |
| 0.58 s of 33.9 s in device execution (85% host) | **task brief only** — not located in the repo | **not independently verified;** treated as a second instance of the same phenomenon, not a contradiction of §6.1's 0.109 s/10.6 s |
| `HYPOTHESES.md` §6 already deletes the comparative hypothesis; `PLAN.md` §6.1 already declares an un-tuned baseline inadmissible; `PLAN.md` C3 already forbids a fp32-vs-bf16 ratio; `PLAN.md` §5.4 already fixes the 2× counter-move rule | those files | read in full |
| `research/TAXONOMY.md` has **no row for state-space / gated-recurrence operators** — the gap `PARITY-SCOPE.md` §5.4.3 names | `research/TAXONOMY.md`; `research/PARITY-SCOPE.md` §5.4 | read |

---

## 9. Verdict for the study owner

1. **The one-line rule** is §1. Its shortest form for a caption: *a ratio against an expert
   kernel is a claim about the algorithm, the hardware, the toolchain, the dtype and the
   author's afternoon until all five are shown fixed.*

2. **Comparison (a) is valid on sm_80 (GA10x/A100-class) and sm_90 (H100).** It is invalid
   on sm_60, sm_70, sm_75, sm_86, sm_89, and on consumer Ada of any kind. Three of those
   failures are *not* about tensor cores — they are about `fla`'s own shared-memory-threshold
   autotune narrowing, and an RTX 4090 fails a gate that a T4 passes on the instruction-class
   axis.

3. **The study cannot run (a) on hardware it is realistically going to get for free.** Kaggle
   gives sm_75 and sm_60, and both are INVALID. This is the finding, and it agrees with
   `HYPOTHESES.md` §6.2 and `PLAN.md` §6 — which reached the same conclusion by a different
   route. What is new here is (i) that the exclusion is *stronger* than "the modern kernel
   does not compile there": it fails on an autotune-space collapse that no amount of writing a
   better sm_75 kernel fixes, and (ii) that the conclusion **transfers to money**: the A100
   rental that `PLAN.md` R0's "decision (B)" already asks for is exactly the card that makes
   (a) runnable, and it is a *sufficient* purchase rather than a step on a ladder. The
   actionable form: **read `cudaDeviceProp.sharedMemoryPerBlockOptin` before paying, and reject
   anything under 166,912 B** — that single read distinguishes the valid rentals (A100-class
   GA10x, H100) from the plausible-looking invalid ones (A40, L40S, any consumer card) and
   costs nothing.

4. **What the study should do instead, and it is already pre-registered.** `HYPOTHESES.md` §6.4
   and `PLAN.md` §6.2: compare against the **device roofline** (H4, H13) and against **itself
   with one pass changed** (H2's 13× exp-folding, H5's 5.6×/97× plan cache, H12's compaction,
   H7's plugin). Those are all-else-equal ratios where one author controls both arms, so they
   are attributable. This file does not weaken them; it adds the cell-label discipline to them
   and it says what to do about the 12,090-kernel launch asymmetry when they are reported.

5. **The fallback is well-defined and it is not a weaker study.** On the free cards the study
   reports (b) — the algorithm/intermediate term, decomposed B1–B5, each with its own counter
   signature per `PLAN.md` §5.4 — plus roofline fractions and self-deltas, and it says
   explicitly: *"this measures the compiler against the hardware and deliberately makes no
   claim about the compiler against a human, because on this hardware no human's kernel
   exists to run as written."* The three arch tables in §3.6 and §7 are publishable in their
   own right: **five of seven plausible architectures cannot host this comparison, and the
   reason for three of them is a shared-memory constant.** That is a finding about the
   evaluation methodology of the field, in the same class as 2607.04454's "correctness admits
   300× slowdowns" — and like that one, it does not require a single GPU hour.

6. **One amendment to file, and one addition, neither of which changes a pre-registered
   decision:**
   - `PLAN.md` §6 and `HYPOTHESES.md` §6: change "Triton requires CC ≥ 7.0, so sm_60 is out"
     to the architecture-level statement (Pascal has no MMA in the ISA) and mark the Triton
     minimum-capability claim **[could not verify]** in current Triton. `HYPOTHESES.md` R1's
     *test* is unaffected; only the parenthetical reason is wrong.
   - `PLAN.md` R3 step 1: add three reads, all ~15 lines, all pre-timing — R3.6a
     `sharedMemoryPerBlockOptin` resolved through `fla`'s own `check_shared_mem`, R3.6b the
     ten `fla`/Triton environment and capability predicates, R3.6c per-kernel MMA/FMA counts
     from SASS on both sides. **These are the difference between a VALID cell and an INVALID
     one, and they are cheaper than any experiment in the plan.**
