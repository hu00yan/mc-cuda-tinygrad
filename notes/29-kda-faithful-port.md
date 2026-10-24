---
kind: reference
---

# 29 — Where TinyGrad's abstraction ends on the KDA kernel, measured

> **⚠️ CITATION STATUS — read before following any path below.**
> The `.agent-scratch/` tree referenced in this note was **deleted from disk**
> (288 files, entire directory). Every `.agent-scratch/...` path in this file is
> currently broken. The content is **recoverable** — it is all in commit
> `9e6767f` — so restore with:
> ```
> git checkout 9e6767f -- .agent-scratch/
> ```
> Until that is run, treat the paths below as *pointers to committed history*,
> not as live files. The measured values quoted in this note are unaffected;
> only their on-disk evidence is missing.


**The question this note answers.** A hand-written kernel that beats a DSL is
usually explained one of two ways: the compiler failed to find something a cost
model could find (a *scheduling* problem), or the optimisation lives below the
abstraction the DSL offers (an *expressiveness* problem). Comparing against a
well-written DSL program cannot tell them apart, because that program was only
allowed the DSL's own ideas. So this arm does the other thing: it reproduces
the hand-written kernel's actual strategy in TinyGrad, **crosses the boundary
everywhere it has to be crossed, and records each crossing as a row.**

The deliverable is the boundary map in §6. The numbers under it are in §4–§5.

**Artifacts.**

| what | where |
|---|---|
| the port, both modes, the crossing registry | `kda/tiny_kda_faithful.py` |
| correctness gate (all 5 modes vs `kda/ref.py`) | `kda/tiny_kda_faithful.py --test` |
| the map, printed | `kda/tiny_kda_faithful.py --crossings` |
| the ladder, printed | `kda/tiny_kda_faithful.py --ladder` |
| offline codegen capture + sm_75/sm_80 SASS + the tf32 model | `.agent-scratch/faithful/` (four scripts, §8) |

**One-sided, stated up front.** The SASS table in §4 has **no hand-written
row**, because `triton` is not installed here and `chunk_intra.py` therefore
cannot be AOT-compiled the way `notes/16` §3.3 compiled its Triton kernel. Every
codegen number below is TinyGrad's. The comparison this arm supports is
*internal* — clean against faithful, one framework, one instrument — and the
kernel-side figures quoted are source counts and source-level derivations, each
labelled at the point of use.

**The pin.** Every `fla/…:NNN` citation is `fla-org/flash-linear-attention` @
`79d12e8ec0ae72bc7f2189b4244d6532ce8c268c`, fetched by me from
`raw.githubusercontent.com` for this note rather than taken from
`notes/25`'s summary, and checked line-for-line against
`kda/SPEC.md` §6.2. The three files that carry the algorithm are
`fla/ops/kda/chunk_intra.py` (the UT transform, `:44-384`),
`fla/ops/kda/chunk_intra_token_parallel.py` (the **default** diagonal builder)
and `fla/ops/kda/wy_fast.py` (the only two consumers of `Akk`).

**Rules this arm held itself to.** `Tensor.custom_kernel` is forbidden and
appears nowhere: it emits CUDA source, so using it would substitute the
reference's own kernel for the experiment. No Triton, no CUDA, no PTX, no
side-loaded `.cu`. Every crossing is a real `ContextVar` (cited at its
definition *and* at its read) or a structural property of the graph. All five
modes are checked against `kda/ref.py`'s float64 `kda_recurrent` — the
per-token definition, not another chunked implementation — at
`kda/test_tiny.py`'s 1e-5 convention.

---

## 1. The verdict, first

> **Seven of the seventeen kernel techniques have no TinyGrad representation at
> all. One more has a knob, and the knob costs 10 mantissa bits and targets a
> table that has no fp32 entry below sm_80 — the only card this project can
> reach. And the two techniques that `notes/25` identified as the kernel's
> signature — the "no `tl.dot` in the diagonal stage" split and the `BC=16`
> blocking — turn out to be, in TinyGrad, respectively invisible and free.**

That last sentence is the result, and it is the opposite of what the framing
predicted. §4 has the measurements. §5 prices the crossing that does hurt.

---

## 2. What the kernel does, and how the port is shaped

The default path (`safe_gate=False`, `chunk.py:47`) is **three** kernels, and
the port reproduces all three:

| kernel | what it does | where the port puts it |
|---|---|---|
| `chunk_kda_fwd_kernel_intra_token_parallel` | grid `(B*T, cdiv(HV,BH))`; a ≤16-iteration rank-1 loop per token; **no `tl.dot` anywhere**; writes one column of each `[16,16]` diagonal block | `_system_token_parallel` Stage A |
| `chunk_kda_fwd_kernel_inter_solve_fused` | 12 dots per key chunk into 12 `[16,16]` fp32 accumulators; 4×14 row-wise substitution; 16-dot depth-3 merge in tf32; ten `[16,16]` stores | Stage B, `_diag_inverse`, `_merge` |
| `recompute_w_u_fwd` | `u = Akk @ (v·β)`, `w = Akk @ (k·β·exp2(g_cum))` | `kda_chunk` step 7 |

`β` row-scales `k` before the gram in both the diagonal
(`chunk_intra_token_parallel.py:115`) and the off-diagonal
(`chunk_intra.py:234`) stages, and the stored `Akk` carries **no** trailing
`Diag(β)`; β is re-applied to the `w`/`u` operands at `wy_fast.py:102,113,117`.
That is algebraically the paper's `M` (`kda/SPEC.md` §8 row 11), and the port
carries it as a ladder switch so the two placements can be priced separately.

The whole UT transform is expressed in **one** code path whose only parameter
is `NC = BT // BC`. `NC = 1` is the clean form (one 62-step chain over the whole
`[BT,BT]`); `NC = 4` is the kernel's (four 14-step chains plus the merge). That
is what makes the clean-vs-faithful delta decompose instead of being one opaque
number.

---

## 3. The one mechanism

```python
on(flag, mode)      # the ONLY switch for a structural crossing
context(mode)       # the ONLY place a TinyGrad knob is set
```

`on()` takes a flag that is a key of `FLAG_TO_CROSSING`, so **every call site
names a row of the boundary map** and a crossing cannot be introduced without
appearing in it. `context()` is three lines and sets `ALLOW_TF32` and nothing
else; it asserts the key is a registered `ContextVar`.

There is deliberately **no flag for `rank1_not_dot`**. The framework cannot see
that distinction, so a flag for it would be a crossing the ladder can toggle
that has no effect — the exact shape of the silent null this study has already
been bitten by once (`notes/06-corrections-ledger.md` §5b).

The ladder, monotone, one crossing at a time:

| mode | crossings on |
|---|---|
| `clean` | — |
| `beta_left` | `beta_on_operands`, `scale_placement` |
| `tf32` | + `allow_tf32_merge` |
| `blocked` | + `bc_blocking` |
| `faithful` | + `subchunk_reference_exp2`, `token_parallel_grid` |

---

## 4. Clean versus faithful — the numbers

All at `kda/perf_kda.py`'s bench shape, `B=2 T=512 H=HV=4 K=V=64 C=64`,
`tinygrad 0.14.0`, generated CUDA captured through `DEV=NULL:CUDA:sm_80` with
the CUDA renderer attached, compiled for sm_80 with
`nvcc -arch=sm_80 --use_fast_math -std=c++17 -Xptxas -v` and censused by the
repo's one categoriser (`minecu_tg/sass_matrix.py` →
`tools/archlab/inspect_ptx.py:CATEGORIES`), so the columns mean the same thing
here as in `notes/16`. `MEASURED`.

| mode | kernel launches | SASS instructions | `FFMA` | `MMA` (tensor core) | spill | max registers |
|---|---|---|---|---|---|---|
| `clean` | 147 | 27,544 | 10,008 | **0** | 0 B | 168 |
| `beta_left` | 147 | 27,608 | 10,008 | **0** | 0 B | 168 |
| `tf32` | 147 | 34,064 | 8,856 | **2,304** | 0 B | 175 |
| `blocked` | **143** | **22,960** | 2,472 | 2,320 | 0 B | 175 |
| `faithful` | 283 | 74,688 | 14,248 | 2,320 | **52,332 B** | **255** |

Launch-weighted: each distinct kernel shape is counted once, then multiplied by
how many times the mode launches it. The unweighted per-shape table, the
register and spill figures, and the raw cubins are in
`.agent-scratch/faithful/sass80/`.

Read the deltas one crossing at a time.

**`beta_on_operands` + `scale_placement`: 147 → 147 launches, 27,544 → 27,608
instructions (+0.2%).** Where β and the single `scale` factor live is an
algebraic rearrangement, not a schedule. `MEASURED`.

**`allow_tf32_merge`: +2,304 `MMA`, −1,152 `FFMA`, launches unchanged, spill
unchanged.** One `ContextVar` moves every fp32 contraction in the operator onto
the tensor core. The kernel does this for 16 dots; TinyGrad does it for
everything reachable, because the knob is global. `MEASURED`; the numerical
cost is in §5.

**`bc_blocking`: −4 launches and −32.6% instructions against the rung it is
isolated from, and it is the *only* mode where the substitution shrinks.** The
clean form is 62 dependent reduction kernels (`BT − 2` at `BT = 64`); the
blocked form is 4×14 = 56. The merge adds **exactly 16 `MMA`** (2,304 → 2,320) —
the same 16 `tl.dot`s the kernel issues at `chunk_intra.py:320-356`, found by
the framework's own scheduler from the shapes and with no further help. Net
against `tf32`: 34,064 → 22,960 instructions, 147 → 143 launches. Against
`clean` it is still negative, 27,544 → 22,960. `MEASURED`.

> **This is the load-bearing correction to C-15, and it is a number this study
> did not have.** The ledger records that `kda/tiny_kda.py` paid 4,032
> sequentially dependent scalar reductions against the kernel's 56. That 4,032
> was a *porting* artefact, not a framework cost: `Tensor.dot` is
> `(x*w).sum(-1)` (`mixin/op.py:392`), so a substitution step written in
> TinyGrad's normal vocabulary is one batched reduce over the whole
> `(B,NT,HV)`, not `C(C−1)` scalar updates. The framework's normal expression
> pays **62**. The kernel's blocking takes 62 to **56** — a 1.11× reduction on
> the critical path, not the 72× the naive port's shape implied. The blocking
> is real and worth having; it is not where a factor of seventy lives.

**`token_parallel_grid`: +140 launches, +51,728 instructions (3.25× over
`blocked`), 0 B → 52,332 B of spill, 255 registers.** Attributable per shape,
not in aggregate:

| kernel shape | launches | what it is |
|---|---|---|
| `r_16_4_16_16_4` | 128 | Stage A: `NCm · BCm · 2` = 4·16·2 whole-batch reductions |
| `r_16_4_4_4_4_4_32_32` | 12 | one per (off-diagonal block, tensor): the 6 `Akk` and 6 `Aqk` `[16,16]` accumulators of `chunk_intra.py:122-135`, each covering both `i_k` iterations at `BK=32` |
| `E_64_4_4_4_4_4_4` | 1 | the block-grid assembly, 16,520 SASS instructions, 49,068 B spill |
| `E_128_32_4` | 1 | a 24-instruction elementwise kernel |

**The kernel issues ONE kernel for Stage A** — `B·T` programs, each doing ≤16
rank-1 accumulates and writing one column (`chunk_intra_token_parallel.py:168,
:117-127`). **The port issues 128.** Same arithmetic, same set of (token, key)
pairs; a completely different schedule. And the twelve `[16,16]` accumulators
the kernel keeps in registers for the whole kernel's life (`chunk_intra.py:73`,
`:122-135`) each get their own kernel and their own memory traffic. This is
§5's crossing.

**Timing: none, and none is available.** The reference needs sm_80+ and the only
card reachable from here is an sm_75 T4, so no wall-clock comparison against
`fla` was made and none is implied. Everything above is codegen shape.

**And there is no reference row in the SASS table, which is a limitation of this
arm and not a formatting choice.** `triton` is not installed in this
environment (`import triton` → `ModuleNotFoundError`), so
`fla/ops/kda/chunk_intra.py` cannot be AOT-compiled the way `notes/16` §3.3
compiled its Triton kernel, and this arm has **no measured instruction count,
register count or spill figure for the hand-written KDA kernel at all.** Every
number in §4 and §5 is TinyGrad's. The kernel-side figures that do appear are
either counts read off its source (the 16 dots, the 56 iterations, the twelve
`[16,16]` accumulators) or source-level derivations (the 136-vs-256 trip
count), and each is labelled where it is used. The comparison this arm can make
is therefore *internal*: clean against faithful, on the same framework, through
the same instrument.

---

## 5. The crossings that cost something

### 5.1 `token_parallel_grid` — IMPOSSIBLE, +140 launches, +225% instructions

TinyGrad has no grid, no program id, no thread mapping. A leading axis is a
batch dimension and nothing else, so "one program per token, ≤16 rank-1
accumulates, no matmul" becomes "64 whole-batch reductions per chunk, each
re-reading `k` and `g_cum`", and the twelve register-resident `[16,16]`
accumulators each get their own kernel and their own memory traffic.

The trip-count cost is exact and derivable on paper. The kernel's loop bound is
`min(i_t + 1, min(T, i_ts + BC))` — per-token, runtime-valued — so for a
16-token sub-chunk it issues `sum(r+1 for r in 0..15) = 136` iterations. The
port, having no runtime trip count, issues all 16 columns for all 16 tokens:
`16 × 16 = 256`. **1.88×.** `MEASURED` against the kernel's own source;
`MODELLED` as a launch count because the kernel is not built here.

There is one real recovery, and it is a *framework* lever rather than a kernel
technique. `NOLOCALS` (`helpers.py:239`; read at `codegen/opt/heuristic.py:81`
and `:163`, where it raises the upcast ceiling from 2048 to 240 elements and
forces `Opt(OptOps.NOLOCALS)`) applied to the two spilling kernels:

| kernel | default | `NOLOCALS=1` |
|---|---|---|
| `E_64_4_4_4_4_4_4` | 16,520 instr, 32 reg, **49,068 B spill** | 10,368 instr, 64 reg, **24,184 B spill** |
| `r_16_4_4_4_4_4_32_32` | 1,912 instr, 255 reg, **272 B spill** | 1,968 instr, 255 reg, **16 B spill** |

`TC_OPT=2` is the counter-example worth keeping: it leaves the 49,068 B spill
untouched and *creates* a new 20,012 B spilling kernel elsewhere in the graph
while adding 16 `MMA`. Forcing tensor cores onto ineligible shapes makes
register pressure worse. `MEASURED`.

### 5.2 `allow_tf32_merge` — a knob, and it changes the numbers

`ALLOW_TF32` is `helpers.py:281`; its only read is `codegen/opt/postrange.py:226`:

```python
if self.ren.target.device in ("CUDA", "NV") and tc.dtype_in == dtypes.float and not ALLOW_TF32: continue
```

Two consequences, both measured, both inconvenient.

**(a) The gate tests the device NAME, so it is inert on the offline codegen
route.** `DEV=NULL:CUDA:sm_80` builds `Target(device="NULL", renderer="CUDA",
arch="sm_80")` (`helpers.py:207`), the string test is false, and the tf32
tensor core is selected *at `ALLOW_TF32 = 0`*. A batched `[16,16]x[16,16]`
captured four ways:

| `target.device` | `ALLOW_TF32` | generated CUDA | `MMA` |
|---|---|---|---|
| `"NULL"` | 0 | 2,575 B, sha1 `510e6232…` | 11 (`.f32.tf32.tf32.f32`) |
| `"NULL"` | 1 | 2,575 B, sha1 `510e6232…` | 11 |
| `"CUDA"` | 0 | **7,266 B**, sha1 `66667cc6…` | **0** |
| `"CUDA"` | 1 | 2,575 B, sha1 `510e6232…` | 11 |

On the whole graph this is not a detail. Captured at sm_80 **without** the
relabel, the `clean` mode emits **1,458 `MMA`** — byte-for-byte the `MMA` count
of the `tf32` mode, at `ALLOW_TF32 = 0`:

| sm_80 capture | launches | `MMA` | `tf32` sites |
|---|---|---|---|
| `clean`, no relabel | 147 | **1,458** | 18 |
| `clean`, relabelled | 147 | **0** | 0 |
| `tf32`, relabelled | 147 | 2,304 | 18 |
| `blocked`, no relabel | 143 | 1,502 | 22 |
| `blocked`, relabelled | 143 | 1,502 | 22 |

So an un-relabelled capture makes `clean` and `tf32` look like the same schedule
on tensor cores, makes the whole `ALLOW_TF32` crossing disappear, and turns the
`blocked` row into a constant. **Everything in §4 is captured with the relabel**,
which is what a real CUDA backend looks like to `postrange.py:226`. This is a
measurement hazard other arms should inherit. `MEASURED`.

**(b) On the reachable card there is nothing to gate.** `codegen/opt/tc.py:94-96`,
read from the installed 0.14.0:

```python
cuda_sm75: list[TensorCore] = cuda_8168_f16
cuda_sm80: list[TensorCore] = cuda_81616 + cuda_8168_f16 + cuda_8168_tf32
cuda_sm89: list[TensorCore] = cuda_sm80 + cuda_81632_f8
```

**`cuda_sm75` has no fp32/tf32 tensor core at all.** The entry is added at sm_80.
So on the only GPU this study can reach, `ALLOW_TF32` is not a knob that fails
to fire — the target it selects from contains nothing to select, the
`ALLOW_TF32 = 0` and `ALLOW_TF32 = 1` captures at sm_75 are byte-identical
(384,651 B, 0 `MMA` either way), and the tensor-core counts in §4 are all
sm_80. Two independent reasons the crossing is unreachable on the T4 — the gate
is a device-name test, and the target table has no fp32 entry below sm_80 — and
they compose. `READ-FROM-SOURCE` for the table, `MEASURED` for the
byte-identity.

The build also refuses, which is the same fact arriving a third time. With the
relabel, 11 of 64 representative kernels fail at sm_75 with

```
ptxas ... error : Feature '.tf32' requires .target sm_80 or higher
```

`MEASURED`.

**(c) The precision cost, modelled.** The crossing is invisible on a CPU
backend, so a CPU gate cannot see it. Rounding every merge operand
(`chunk_intra.py:320-356` — the exact set passed as
`input_precision='tf32'`) onto the 10-explicit-mantissa-bit grid, three trials
at `N=256 HV=8 C=64 K=64`:

| | max rel err of `Akk` vs the exact inverse |
|---|---|
| merge in float32 | 1.664e-08 |
| merge operands in tf32 | **1.124e-03** |

`MODELLED` — the operands are rounded, the accumulator is not, so this is a
lower bound; a real sm_80 run would be worse. **113× the 1e-5 correctness
gate**, from one knob, on the critical path of the inverse. A CPU-only gate
passes all five modes at ≤5.03e-06 and cannot see any of it.

### 5.3 `per_op_precision` — IMPOSSIBLE

The kernel chooses tf32 for the 16 merge dots and IEEE for the diagonal stage,
*per `tl.dot`*, inside one kernel (`chunk_intra.py:22` versus `:288`). TinyGrad
has no per-op precision selector: `ALLOW_TF32` is read once, globally, in a
tensor-core filter. The port's `faithful` mode gets IEEE in the diagonal stage
only because a `[1,16]x[16,16]` contraction is *ineligible* for a tensor core —
a coincidence of shape, not a control. There is no schedule that expresses
"tf32 here, IEEE there". `MEASURED` that the knob is global; the absence of a
per-op selector is `READ-FROM-SOURCE` (`helpers.py:281`, one read site).

### 5.4 `unrolled_trip_count` — IMPOSSIBLE

Both kernel loops are runtime-bounded (`do_not_specialize=['T']` at
`chunk_intra.py:43`; `range(2, min(BC, T - i_tc0))` at `:285`). TinyGrad has no
loop, so it has no runtime trip count, and the port's unroll is the full-chunk
count with the tail masked. Cost: the 1.88× in §5.1, and — for a ragged tail —
a static 56 issued where the kernel would issue 16. **Untestable here**: this
port refuses ragged `T` (`T % BT == 0`), exactly as `kda/ref.py` does, so the
tail case is a source-level derivation, not a measurement.

---

## 6. THE BOUNDARY MAP

One row per kernel technique. `MEASURED` = measured on this host against real
tinygrad 0.14.0 and real nvcc. `MODELLED` = arithmetic or source derivation.
`PREDICTED` = not established. `UNTESTABLE` = no instrument here reaches it.

| # | technique (kernel side) | cite | in TinyGrad's vocabulary? | crossing | what it cost | what it bought | status |
|---|---|---|---|---|---|---|---|
| 1 | UT transform blocked at `BC=16`, and `BC` hardcoded / `chunk_size ∈ {32,64}` | `chunk_intra.py:274-313`, `:320-356`, `:825`; `chunk.py:442-444` | **yes, and free** — the framework is *more* general, reaching a UT transform at any chunk size | none | **−4 launches, −32.6% instructions** vs the rung it is isolated from; 4 of 13 test cases unrunnable in the blocked modes | 62→56 dependent steps; **exactly 16** `MMA` in the merge | MEASURED |
| 2 | diagonal stage uses `tl.sum`, **no** `tl.dot` | `:288` (0 `tl.dot` in 274-314) | **cannot tell the difference** | *none possible* | 0 | 0 — and see the note below | MEASURED |
| 3 | runtime-bounded loop trip counts | `:43`, `:285-290-296-302` | **no** | — | 1.88× the Stage-A iterations; 56 vs 16 on a tail | 0 | MEASURED / UNTESTABLE (tail) |
| 4 | default path is token-parallel, grid `(B·T, …)` | `chunk_intra_token_parallel.py:54,117-127,168` | **no** | — | **+140 launches, +225% instructions, 0→52,332 B spill, 255 reg** | 0 | MEASURED |
| 5 | 12 `[16,16]` accumulators stay in registers | `:71-78`, `:122-135` | **no** | — | 12 kernels + their memory traffic; the kernel has 0 kernels, by design | 0 | MEASURED |
| 6 | merge in tf32, diagonal stage not | `:21-24`, `:319-320` | one global `ContextVar` over a target table with no fp32 entry below sm_80 (`tc.py:94-96`) | `ALLOW_TF32=1` | 10 mantissa bits → **1.124e-03** modelled error; **unreachable on sm_75** | 2,304 `MMA` | MEASURED + MODELLED |
| 7 | tf32 *per dot*, not per kernel | `:22` vs `:288` | **no** | — | mixed-precision schedule unrepresentable | 0 | READ-FROM-SOURCE |
| 8 | sub-chunk `exp2` reference `g_n` | `:160,162,164` etc. | **yes** | code shape | 2 `exp2` per block instead of 1 | every intermediate ≤ `2^(BC·\|g\|)`; Γ never formed | MEASURED (correct) |
| 9 | `exp2` with `RCP_LN2` folded into the cumsum | `chunk_fwd.py:61-68` | **yes** | `Tensor.exp2` | 0 | `MUFU.EX2`; and `/` → `a*(1/b)` (`mixin/elementwise.py:251`) makes `K/Γ` unformable by accident | READ-FROM-SOURCE |
| 10 | `Akkd` in a **separate fp32 buffer** | `:834-835`, `:261-268`, `:286` | **yes** (per-tensor dtype) | — | 0 at float32 | the kernel's precision decision; vacuous in a float32 port | MEASURED |
| 11 | head-major `[B,NT,HV,·]` layout, token axis outermost | `:101-107`; `wy_fast.py:73-79` | **yes**, one permute | code shape | one layout change, consumed by every later contraction | access to `Tensor.dot`, which is `(x*w).sum(-1)` and so only works when the contracted axes are last | MEASURED |
| 12 | `β` on the operands, not on the inverse | `:115`, `:234`; `wy_fast.py:102,113,117` | **yes** | code shape | +0.2% instructions | kernel graph parity; a double-`β` bug here is order-1, not rounding | MEASURED |
| 13 | `scale` on `Aqk` and the inter-chunk term | `:123`, `:230`; `gla/chunk.py:427` | **yes** | code shape | 0 | kernel parity; the operator is the same either way | MEASURED |
| 14 | dense `BT×BT` `Akk`, 6 of 16 blocks zero, unexploited | `:362-384`, `:831-833`; `wy_fast.py:91-94` | **yes — and deliberately not taken** | none | 37.5% of MACs wasted **on both sides** | a control row: the delta is not from here | MEASURED |
| 15 | a unit-lower-triangular inverse at all | `:274-356` | `Tensor.inverse`/`solve` do not exist; only `qr`, `svd` | — | whichever substitution is selected | — | READ-FROM-SOURCE |
| 16 | `OPT_LEVEL` as a codegen knob | — | **does not exist** in 0.14.0 | — | cannot be swept | — | MEASURED (absence) |
| 17 | `USE_TC` as spelled on the brief's list | — | registered as **`"TC"`**, not `"USE_TC"` | — | `Context(USE_TC=0)` raises `KeyError`; a sweep would report "no effect" about a knob never set | — | MEASURED |

**Tally: 7 IMPOSSIBLE, 1 needs a knob, 1 INERT, 4 structural, 4 free.** These
are the seventeen rows of `CROSSINGS` in `kda/tiny_kda_faithful.py`, in the
same order; `kda/tiny_kda_faithful.py --crossings` prints them with the full
`file:line` for each lever, and a `flag` in `FLAG_TO_CROSSING` may only name a
row here, so the two cannot drift apart without an assertion firing.

### The one row worth reading twice — #2

`Tensor.dot` **is** `(x*w).sum(-1)` (`mixin/op.py:392`). So the kernel's
distinction between `tl.sum(b_a[:,None]*b_Ai, 0)` and `tl.dot` — the single
most load-bearing choice in the diagonal stage, made *so the work stays off the
MMA path* — is **not a distinction the framework has**. Both spellings emit
byte-identical CUDA: 2,204 B, sha1 `40d2e93823d1716e`, zero `MMA`, invariant
across all four `(spelling × ALLOW_TF32)` cells.

The kernel's *reason* still holds, and it holds for a reason the program never
mentions: TinyGrad's tensor-core filter reads tensor **shapes**, and a
`[1,16]x[16,16]` contraction has an `M` of 1 that cannot be upcast, while a
`[16,16]x[16,16]` one goes to `mma.sync`. Same operator, different answer,
decided below the line the port writes at.

---

## 7. The verdict on the framing

**This is an expressiveness problem, and the evidence is not close.**

A scheduling problem would mean the kernel's win is reachable by re-associating
or re-tiling work the framework already has. Row 1 rules that out: `bc_blocking`
— the technique `notes/25` and ledger C-15 identify as *the* thing the naive
port got wrong — costs TinyGrad **negative** work. Four fewer launches and 32.6%
fewer instructions than the rung it is isolated from, with the framework's own
scheduler reaching the 16-dot merge from the shapes alone. The blocked inverse
is what a good cost model would find.

What is left is rows 3–7: a program-level decomposition with no representation
in a framework whose unit of parallelism is a tensor axis (4, 5); a loop bound
with no representation because there are no loops (3); and a **per-operation
precision choice with no representation because precision is a global
`ContextVar` over a target table that has no fp32 entry below sm_80** (6, 7).
Those are not schedule choices. The framework cannot express them, so no cost
model can find them, and the best a port can do is pay for them in the currency
the framework does have — 140 extra kernel launches and 52 KB of spill.

**The single technique that decides it** is the one in row 7. Everything else
here is a cost the framework can charge; `per_op_precision` is a *decision the
framework cannot represent at all*, and it is the reason a faithful port cannot
be faithful: TinyGrad's precision is one switch for the whole operator, so
reproducing "tf32 for the merge, IEEE for the substitution" requires either
changing the algorithm (and it is a different operator) or accepting a schedule
the kernel does not run. The tf32 error model in §5.2(b) is that crossing's
price, and it is 113× the correctness gate — larger than any scheduling
difference measured anywhere in this arm.

## 8. Reproducing, and what is untestable

```bash
DEV=CPU .venv/bin/python kda/tiny_kda_faithful.py --all          # map + gate
# offline codegen, one mode per process (no GPU):
.venv/bin/python .agent-scratch/faithful/capture.py <mode> sm_80 .agent-scratch/faithful/cu --relabel
bash .agent-scratch/faithful/sass.sh .agent-scratch/faithful/sel .agent-scratch/faithful/sass80
.venv/bin/python .agent-scratch/faithful/table.py .agent-scratch/faithful/sass80 sm_80
.venv/bin/python .agent-scratch/faithful/tf32_cost.py
```

`ARCH=sm_75` for the T4 comparison. The three gotchas that cost time and are
recorded so they cost none to the next arm: **`JIT=0` is mandatory** (the JIT
returns the existing `Program` when a kernel signature repeats, which is exactly
what happens when two modes lower to the same source, so a mode goes silently
unmeasured); **`--relabel` is mandatory** for any `ALLOW_TF32` claim (§5.2a);
and the container script goes in a file, not a `docker run` heredoc, because a
quoting error there produces an empty script that looks exactly like "no
kernels compiled".

`.agent-scratch/faithful/` holds only the four re-derivation scripts
(`capture.py`, `sass.sh`, `table.py`, `tf32_cost.py`) plus `probe_all.py`, the
one-cell-per-process probe behind §5.2a's table. The captured `.cu`, the cubins
and the SASS dumps have been deleted: they are re-derivable from
`kda/tiny_kda_faithful.py` plus the commands above, and they are large. The
pinned `fla` sources were fetched for this note with

```bash
for f in chunk_intra chunk_intra_token_parallel wy_fast chunk_fwd chunk naive; do
  curl -sSf -o "$f.py" \
    "https://raw.githubusercontent.com/fla-org/flash-linear-attention/79d12e8ec0ae72bc7f2189b4244d6532ce8c268c/fla/ops/kda/$f.py"
done
```

and deleted after reading; the commit is the citation, not the local copy.

### Untestable here, named

- **Any timing, anywhere.** The reference needs sm_80+; the reachable card is an
  sm_75 T4; no wall-clock comparison against `fla` was made and none is implied.
- **The reference's own SASS.** `triton` is not installed, so
  `chunk_intra.py` cannot be AOT-compiled as `notes/16` §3.3 did for its Triton
  kernel. Every SASS number in this note is TinyGrad's; there is no
  hand-written row to put beside it. Named here rather than left as a silently
  one-sided table.
- **The ragged-tail trip count** (row 3): this port refuses `T % BT != 0`.
- **Real tf32 rounding of the accumulator** (row 6): the §5.2c model rounds
  operands only, so it is a lower bound.
- **Occupancy.** Register counts are measured; the warps-per-SM they imply are
  `MODELLED` from a transcribed hardware table, as in `notes/16` §3.2.
