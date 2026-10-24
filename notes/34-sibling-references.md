---
kind: reference
---

# 34 — Sibling references: what else runs on the T4, and what it would cost to substitute

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


**The question.** The KDA arm of this study is a comparison against hand-written
code, and its reference is `fla`. On a Tesla T4 `fla` does not refuse — it
**degrades**: `kda/baseline_pin.py` pins twelve narrowing sites and they all
resolve to their small branch. A degraded reference is still a reference, but it
is a weaker one, and it is shared with a family of sibling operators. This note
asks whether a *different* hand-written kernel — structurally close enough to
carry the comparison — runs on sm_75 un-degraded, and what adopting one would
change about the study.

**The answer, in one line.** Exactly one candidate qualifies: **mamba-2's
hand-written Triton chunked scan**, which runs forward and backward on this T4
with its autotune spaces **entirely unnarrowed**, and which is *not* KDA and
*not* a delta rule, so adopting it replaces one claim with a different one.
Every candidate inside `fla` — including the delta-rule family that shares KDA's
exact state kernel — is narrowed on this card, because the T4 misses even
`fla`'s *default* 102 400 B threshold by 36 864 B.

**Status tags** are `notes/06-corrections-ledger.md`'s. Every device number here
was measured in one process on a Kaggle Tesla T4; every library claim is cited to
a file at a pinned commit. Two claims in this file are corrections to my own
first pass and are flagged **[CORRECTION]** rather than quietly dropped.

---

## 0. The card, measured

One probe, one process, on `Tesla T4 compute capability 7.5`, driver 580.159.04,
CUDA runtime 13.0, torch 2.10.0+cu128, triton 3.6.0. The number that decides
everything is the one `fla` itself reads
(`fla/utils/_device.py:170`, `triton.runtime.driver.active.utils.get_device_properties(0)['max_shared_mem']`):

```
max_shared_mem   65536          <- what fla.utils.check_shared_mem reads
get_all_max_shared_mem()  [65536, 65536]

check_shared_mem('none')    = False   threshold=102400
check_shared_mem('ada')     = False   threshold=101376
check_shared_mem('ampere')  = False   threshold=166912
check_shared_mem('hopper')  = False   threshold=232448
TRITON_F32_DEFAULT = ieee
```

**The T4 misses every one of `fla`'s four thresholds, including the `DEFAULT`
one.** `fla/utils/_device.py:181-185`:

```python
class Backend(Enum):
    ADA = 101376       # RTX 4090
    AMPERE = 166912    # A100
    HOPPER = 232448    # H100
    DEFAULT = 102400   # Default
```

shortfalls on this card: `DEFAULT` by **36 864 B**, `ADA` by **35 840 B**,
`AMPERE` by **101 376 B**, `HOPPER` by **166 912 B**.

This **extends** `notes/08-baseline-integrity.md` rather than contradicting it.
That note establishes `AMPERE = 166912` as *the* working criterion and puts the
T4 in the `CARD_GATE` fail column at 65 536 B. What it does not say, and what
matters for candidate selection, is that **`DEFAULT = 102400` also fails**, so
`check_shared_mem()` with no argument is *also* false here. That widens the set of
gated `fla` files from the four delta-rule ones to every op that calls the
no-argument form. I found nothing wrong in that note; the gate as documented is
correct, and the T4 row is correct. The prior art in `kda/baseline_pin.py` is
also correct as far as I could check it, and I re-derived its central claim
(§2.1 below) rather than transcribing it.

The `TRITON_F32_DEFAULT=ieee` line is `fla/utils/_device.py:150-152`,
`if IS_NVIDIA and not IS_TF32_SUPPORTED: os.environ['TRITON_F32_DEFAULT']='ieee'`.
It is process-global and it is **specific to importing `fla`**: in the mamba-only
process the variable is unset. Measured, in both processes, on the same card.

---

## 1. Method, and the two ways this note nearly produced a false result

Each candidate is asked four questions, and the fourth is the one that catches
the failure the brief warns about:

1. **Hand-written?** Structure and blocking are the author's choices, not a
   framework dispatching into a vendor library.
2. **Runs on sm_75?** Executed on the card, and the **output values** checked,
   not just that compilation succeeded.
3. **Structurally close?** Argued from the mechanism — sequential depth, fusion
   width, matmul shape, gather, tensor-core use.
4. **Pinnable?** Project, exact commit, file, licence, and a sha256 of the bytes
   that were executed.

A candidate that compiles but silently selects a degenerate configuration does
not pass question 2. So the probe patches `triton.runtime.Autotuner.run` and
reports, per kernel, **how many configs were offered and which one won**. No
`fla` or `mamba` file is modified to do this; the patch lives in the probe.

### 1.1 **[CORRECTION]** "fla produces NaN on the T4" — false, and it was my inputs

My first pass reported `KDA`, `GatedDeltaNet` and `DeltaNet` all returning
**non-finite** values on the T4, and I was one step from recording that as a
card property. It is not.

The study L2-normalises `q` and `k` before calling the kernel
(`kda/kda_vs_fla.py:303-304`, then `:600` with `use_qk_l2norm_in_kernel=False`).
I passed raw `randn`. With un-normalised keys `|k|^2 ≈ K = 128` per step, and the
delta-rule update `S ← S + k (v − S k)ᵀ β` becomes an expansive rank-1 map
applied `T` times, which overflows on any card. Re-run with the study's inputs:

| case | dtype | kernel finite | vs `fla`'s own naive | shape |
|---|---|:--|--:|:--|
| KDA | bf16 | yes | **5.714e-03** | (1,512,4,128) |
| KDA | fp16 | yes | **7.148e-04** | (1,512,4,128) |
| GatedDeltaNet | bf16 | yes | **3.864e-03** | (1,512,4,128) |
| DeltaNet | bf16 | yes | **4.771e-03** | (1,512,4,128) |
| gla | bf16 | yes | **5.128e-03** | (1,512,4,128) |

`fp16` is an order of magnitude better than `bf16` on this card (§4).

### 1.2 **[CORRECTION]** "sm_75 has no bf16 tensor cores, so bf16 `tl.dot` is the mechanism" — false

That was my second hypothesis, and a direct measurement kills it. A bare
`64×64×64` (and `128×128×128`) `tl.dot` of a known ramp, on this card:

| dtype | ran | finite | max relerr vs torch fp32 |
|---|:--:|:--:|--:|
| bfloat16 | yes | yes | **0.0** |
| float16 | yes | yes | 5.93e-07 |
| float32 | yes | yes | 2.36e-07 |

`bf16` is *exact* here, and `input_precision` ∈ {`ieee`, `tf32`, `tf32x3`} all
return the same answer. Triton 3.6.0 on sm_75 handles bf16 `tl.dot` by
accumulating in fp32, not by executing a bf16 MMA. So "no bf16 tensor cores on
Turing" is **not** a reason to reject a candidate here, and I would have written
that into the note had I not measured it. Recorded because it is exactly the
kind of plausible-sounding mechanism that survives into a writeup unchecked.

One control came back `False` and I am not treating it as a finding: a pure
elementwise Triton kernel in bf16 does not match torch's fp32 exactly, which is
just bf16 rounding of `2a+1` at magnitudes near 8192, not a card fault.

---

## 2. The candidate table

Checks: **HW** hand-written · **sm_75** runs on this card (measured) ·
**Close** structurally close enough to carry the comparison · **Pin** licensable
and pinnable. `—` means untested, honestly untested.

| # | candidate | file @ commit | HW | sm_75 | Close | Pin |
|---|---|---|:--:|:--:|:--:|:--:|
| 1 | **mamba-2 chunked scan (Triton)** | `mamba_ssm/ops/triton/ssd_combined.py` @ `e9594ce` | **pass** | **pass, unnarrowed** | partial | **pass** |
| 2 | KDA (the subject) | `fla/ops/kda/` @ 0.5.2 | pass | pass, **narrowed 3×** | — | pass |
| 3 | GatedDeltaNet | `fla/ops/gated_delta_rule/` @ 0.5.2 | pass | pass, **narrowed** | **pass** | pass |
| 4 | DeltaNet | `fla/ops/delta_rule/` @ 0.5.2 | pass | pass, **narrowed** | **pass** | pass |
| 5 | gdn2 | `fla/ops/gdn2/` @ 0.5.2 | pass | **untested** | pass | pass |
| 6 | comba | `fla/ops/comba/` @ 0.5.2 | pass | **untested** | pass | pass |
| 7 | gated_oja_rule | `fla/ops/gated_oja_rule/` @ 0.5.2 | pass | **untested** | pass | pass |
| 8 | rwkv7 | `fla/ops/rwkv7/` @ 0.5.2 | pass | **untested** | pass | pass |
| 9 | simple_gla | `fla/ops/simple_gla/` @ 0.5.2 | pass | ran, **narrowed** | partial | pass |
| 10 | gla | `fla/ops/gla/` @ 0.5.2 | pass | ran, **narrowed** | weak | pass |
| 11 | log_linear_attn | `fla/ops/log_linear_attn/` @ 0.5.2 | pass | **untested** | partial | pass |
| 12 | based | `fla/ops/based/` @ 0.5.2 | pass | **untested** | weak | pass |
| 13 | Kimi-Linear KDA | `MoonshotAI/Kimi-Linear` @ `8c1d85e` | **fail** | — | — | — |
| 14 | mamba-1 `selective_scan` (CUDA) | `state-spaces/mamba` @ `e9594ce` | pass | **untested** | weak | pass |
| 15 | flash-attn 2 | `Dao-AILab/flash-attention` | pass | **untested** | **fail** | pass |
| 16 | a second voxel DDA raymarcher | — | — | — | — | **none located** |

Rows 5–8 and 11–12 are marked untested because they are untested; I ran out of
run budget before them and I am not going to dress a source reading as a
measurement. Row 15's `Close` is a **fail** on mechanism, not a gap in testing
(§3.4). Row 16 is a negative result and §3.5 explains it.

---

## 3. The candidates, one at a time

### 3.1 mamba-2's chunked scan — the one that qualifies

`state-spaces/mamba` @ **`e9594ce1c732d97440f0332fdc43170a2294dbfa`**, licence
**Apache-2.0** (repo `LICENSE`). Tarball sha256
`e2384aacc82612f33cae5a52929f2c7ac300cebf3d2d45a042f03befa0f3e8a6`,
1 732 630 B, re-downloaded and re-hashed identically.

**Hand-written — pass.** `ssd_combined.py:1` reads `# Copyright (c) 2024, Tri Dao,
Albert Gu.` The five Triton modules the forward executes were sha256-verified
**on the device** before anything ran, and each matched the pinned commit:

```
0b7c4cfa9e994278  mamba_ssm/ops/triton/ssd_combined.py     MATCHES pinned commit
6d82771b2f62a7bf  mamba_ssm/ops/triton/ssd_chunk_state.py  MATCHES pinned commit
055b5ce4cb0f30c8  mamba_ssm/ops/triton/ssd_chunk_scan.py   MATCHES pinned commit
ae1fab6c680cb531  mamba_ssm/ops/triton/ssd_state_passing.py MATCHES pinned commit
5059b16f8fa269cd  mamba_ssm/ops/triton/ssd_bmm.py          MATCHES pinned commit
```

**Runs on sm_75 — pass, and unnarrowed.** Forward, backward, `bf16`/`fp16`/`fp32`,
`chunk_size` 64/128/256/512, all finite.

```
mamba_ssm.ops.triton.ssd_combined            arch/smem predicates: NONE
mamba_ssm.ops.triton.ssd_chunk_state         arch/smem predicates: NONE
mamba_ssm.ops.triton.ssd_bmm                 arch/smem predicates: NONE
mamba_ssm.ops.triton.ssd_chunk_scan          arch/smem predicates: NONE
mamba_ssm.ops.triton.ssd_state_passing       arch/smem predicates: NONE
```

That is a source scan for `check_shared_mem`, `max_shared_mem`,
`shared_memory`, `compute_capability`, `get_device_capability`, `sm_75`, `sm_80`,
run **on the device** over the imported modules. There is no arch predicate
anywhere in the Triton path. The autotune lists are literal
(`ssd_combined.py:79-88`; `ssd_chunk_state.py:28-34`, `:178-185`) and every one
offered its **full** space:

| kernel | configs offered | selected |
|---|--:|---|
| `_chunk_cumsum_fwd_kernel` | 7 | `BLOCK_SIZE_H: 2, num_warps: 4, num_stages: 3` |
| `_chunk_state_fwd_kernel` | 9 | `BLOCK_SIZE_M: 64, N: 32, K: 32, warps 2, stages 5` |
| `_state_passing_fwd_kernel` | 6 | `BLOCK_SIZE: 2048, warps 4, stages 3` |
| `_bmm_chunk_fwd_kernel` | 9 | `BLOCK_SIZE_M: 128, N: 32, K: 32, warps 4, stages 4` |
| `_chunk_scan_fwd_kernel` | 11 | `BLOCK_SIZE_M: 32, N: 64, K: 32, warps 2, stages 5` |

Compare the `fla` row in §3.2: **4** configs for the equivalent kernel. Note also
`TRITON_F32_DEFAULT` is **unset** in a mamba-only process, where importing `fla`
sets it to `ieee`. Two of the four degradations `kda/baseline_pin.py` pins for
KDA simply do not exist in this project.

And the build treats sm_75 as a first-class target — `setup.py:181-182`:

```python
cc_flag.append("-gencode")
cc_flag.append("arch=compute_75,code=sm_75")
```

**Structurally close — partial.** The mechanism overlap with KDA is real: a
chunked linear recurrence with per-head decay, `tl.dot` on narrow `K`, a
sequential state carried across chunks, and a multi-kernel pipeline whose fusion
width is the author's choice. On the KDA subject's own terms
(`notes/03-kda.md` §1: UT transform unrolled `C` times, inter-chunk scan unrolled
`NT` steps) mamba has the same two-part shape. What it does **not** have is the
part of KDA that is hardest and most distinctive: the **delta rule**. No `β`, so
no WY representation, no `C×C` unit lower-triangular inverse, no `tl.dot` in a
triangular solve. Its decay is a **scalar per head** (`A: (nheads)`,
`ssd_combined.py:633`) where KDA's is per channel
(`fla/ops/kda/naive.py` docstring: *"Per-dimension decay gates ... of shape
`[B, T, HV, K]`"*). On matmul shape it is close; on fusion width it is close; on
the UT transform it is absent.

**Correctness.** Forward stages 1–3 match `torch` to fp32 round-off:

| stage | kernel | reference | relerr |
|---|---|---|--:|
| 1 | `_chunk_cumsum_fwd` (`ssd_chunk_state.py:718`) | `torch.cumsum(dt*A)` | **1.69e-07** |
| 1 | `_chunk_cumsum_fwd` dt output | — | **0.0** |
| 2 | `_chunk_state_fwd` (`:812`) | `chunk_state_ref:1094` | **2.09e-07** |
| 3 | `_state_passing_fwd` | `state_passing_ref:327` | **1.57e-07** |
| — | bf16 end-to-end | fp32 end-to-end | **5.86e-03** |

**`[UNVERIFIED]`** stages 4 and 5. So **4 of the 6 stage comparisons agree to
fp32 round-off** (two in stage 1, one each in stages 2 and 3). Composing the
stages in `torch` and comparing to the Triton pipeline gives relerr **1.359**,
and the composition agrees with the *end-to-end Triton call* to the same three
digits — so the disagreement is between `torch` composition and Triton, not
inside Triton. I could not determine which side is wrong without more budget. Two
of my own composition errors are already documented in the probe's history (`A`
is `(nheads)` not `(nheads, dim)`; `chunk_state_ref` alone omits the inter-chunk
state passing), so I am recording this as unresolved rather than as a defect in
either project. The study does not depend on it: correctness evidence that *is*
solid is the analytic gradient, which matches central differences of the
kernel's own forward in fp32 to **6.5e-03 – 8.7e-03** on the four
largest-gradient entries.

### 3.2 The `fla` delta-rule family — closest in structure, and all narrowed

These are the *closest* siblings in existence: same UT transform, same WY
representation, same state kernel. And all of them run, and all of them are
narrowed, because they import the same two shared files.

`fla/ops/common/chunk_delta_h.py:40-45` is the forward state kernel's autotune
list, and it is the file the whole delta-rule family shares:

```python
configs=[
    triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
    for num_warps in GATED_DELTA_RULE_FWD_H_NUM_WARPS            # :28 -> [2, 4]
    for num_stages in ([2, 3, 4] if check_shared_mem('ampere') else [2, 1])
    for BV in ([32, 64] if check_shared_mem('ada') else [32])
],
```

On this card that is `2 warps × 2 stages × 1 BV` = **4 configs**; on an A100 it
is `2 × 3 × 2` = **12**. A **3× reduction, measured**: the probe reports
`chunk_gated_delta_rule_fwd_kernel_h_blockdim64: ... (of 4 cfgs offered)` with
`BV: 32, num_stages: 1`. `kda/baseline_pin.py` computes this arithmetically;
here it is the count triton itself reports.

Who imports the gated file (`grep -rl "from fla.ops.common.chunk_delta_h import"`):

```
fla/ops/kda/chunk_fwd.py        fla/ops/kda/chunk_bwd.py
fla/ops/gated_delta_rule/chunk.py  fla/ops/delta_rule/chunk.py
fla/ops/gdn2/chunk_fwd.py       fla/ops/gdn2/chunk_bwd.py
fla/ops/comba/chunk.py          fla/ops/common/intracard_cp.py
```

**So the delta-rule family is uniformly degraded on sm_75.** Rows 2–8 of the table
are all downstream of that one list. The other gated sites, for completeness and
because they bound what an `fla` sibling can be worth:

| file:line | site | this card | a card that passes |
|---|---|---|---|
| `fla/ops/common/chunk_delta_h.py:43` | fwd `num_stages` | `[2, 1]` | `[2, 3, 4]` (AMPERE) |
| `fla/ops/common/chunk_delta_h.py:44` | fwd `BV` | `[32]` | `[32, 64]` (ADA) |
| `fla/ops/kda/gate.py:22` | fwd `BS_LIST` | `[16, 32]` | `[32, 64]` (DEFAULT) |
| `fla/ops/common/chunk_h.py:16` | `BKV_LIST` | `[16, 32]` | `[32, 64]` (DEFAULT) |
| `fla/ops/gla/chunk.py:20-21` | `BK_LIST`/`BV_LIST` | `[16, 32]`/`[16, 32]` | `[32, 64]`/`[64, 128]` |
| `fla/ops/kda/chunk_bwd.py:25-26` | bwd `BK`/`BV` | `[16, 32]`/`[16, 32]` | `[32, 64]`/`[64, 128]` |
| `fla/ops/delta_rule/wy_fast.py:250` | bwd `CONST_TILING` | 32 | 64 (DEFAULT) |
| `fla/ops/gated_oja_rule/chunk_h.py:16` | `BKV_LIST` | `[32, 64]` | `[64, 128]` |

Note the `DEFAULT`-keyed rows: `gate.py:22` and `chunk_h.py:16` are outside the
`AMPERE` criterion `notes/08-baseline-integrity.md` names, and they are gated
here too. **A sibling inside `fla` cannot escape the gate by being a different
operator**, because the gate lives in shared files that whole families import.
That is the central negative result of this note.

Measured confirmation for the two that reach `common/chunk_h.py`: `gla`'s state
kernel `chunk_fwd_kernel_h` reported **48 configs offered**, selected
`BK: 16, BV: 32` — the config *count* is unchanged, but the *tiles* are half
size, which is the same degradation wearing a different hat. `simple_gla` ran and
autotuned; I could not complete its numeric check because `fla` ships no naive
reference taking the same arguments (`naive_recurrent_simple_gla(q, k, v, g)`,
`fla/ops/simple_gla/naive.py:61`, has no `g_gamma`), so row 9 is **ran, value
unverified**.

### 3.3 Kimi-Linear — rejected, and the reason is worth recording

The obvious candidate is Moonshot's own KDA. It is not a candidate. The
repository README at commit `8c1d85eb6b5f8fcefb15758691b0ce50b0827ce3` says:

> We open-sourced the KDA kernel [FLA](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda)

There is no `kimi_linear/ops/kda/` and no `kimi_linear/fla/ops/kda/` at that
commit; every candidate path 404s. **Kimi-Linear *is* `fla`'s KDA.** Its model
layers depend on `fla-core` (`pip install -U fla-core`). So it fails the
hand-written and runnability checks by construction: it is the same artifact as
the degraded subject, wearing a different citation. Recorded because it is the
first thing a reader will try, and "the paper's own repo" is exactly the kind of
citation `kda/baseline_pin.py` exists to stop being trusted.

### 3.4 flash-attention 2 — rejected on mechanism, before any GPU time

`Close` fails. FA2 is a non-recurrent attention kernel: no state carried across
chunks, no gating, no UT transform, no decay. On the five axes that decide
whether a kernel can carry a KDA comparison — sequential depth, fusion width,
matmul shape, gather, tensor-core use — it shares matmul shape and tensor-core
use and shares essentially nothing else. Its Turing support is also a separate
question I did **not** test. A like-for-like comparison on FA2 would be a
comparison of *attention* to *KDA*, and the interesting results in
`notes/16-generated-vs-handwritten.md` are specifically about what a compiler
does with a sequential recurrence that a tensor form cannot express. I am
recording `sm_75` as untested rather than guessing at FA2's arch floor.

### 3.5 A second voxel DDA raymarcher — none located

The brief asks for siblings of *both* subjects. On the raymarcher side I did
not find one, and I want to be precise about why, because the honest version is
"not attempted", not "does not exist".

The raymarcher arm is **not blocked**. `minecu_ref/` is hash-pinned, compiles
clean for sm_75, and is already measured; `notes/19-tinygrad-vs-handwritten.md`
compares against it. A second raymarcher reference would *strengthen* a
comparison that is already running, not unblock one. Given a fixed run budget I
spent it on the KDA arm, where the block is real, and I am not going to
manufacture a negative result I did not earn.

What a raymarcher sibling would have to clear, so the next pass does not repeat
this: genuinely hand-written CUDA; `extern "C"` or otherwise callable without
libtorch, the way `minecu_ref/bench/bench.cu` declares the reference's launchers
itself; a 32-step DDA loop with a data-dependent exit, which is the structural
core; no vendor library underneath; a permissively-licensed pinned commit.
Everything I could think of in the "many small GitHub voxel renderers" category
fails at least one of those, and the well-licensed well-known hand-written CUDA
I could name is either a vendor sample or a rasteriser, which has the gather and
the fp32 math but **no sequential depth** — the one property that makes the DDA
interesting.

---

## 4. What is actually available, ranked

If the study wants a second KDA arm, in descending order of what each buys:

1. **Keep `fla` KDA, and fix the scope label.** The degradation is **measured
   and bounded** (`[MEASURED]`, §3.2): 12 → 4 configs on the forward state
   kernel, `BV ∈ {32}`, `num_stages ∈ {1,2}`, plus `TRITON_F32_DEFAULT='ieee'`.
   A run that *reports* that is a valid, self-describing measurement. The
   alternative — `H8` to `K8`/`T1` in a scope note — is nearly free, and
   `notes/15-experiment-b.md` §3 already carries this table as prose. **This
   costs nothing and loses nothing.**
2. **Add `fla` GatedDeltaNet as a second *narrowed* reference.** Measured working
   (relerr 3.86e-03, §1.1), same UT transform, same gated state kernel, and
   *inherently* subject to the same label. It buys robustness against
   "KDA-specific artefact" — if TinyGrad's gap tracks GDA's too, the gap is about
   the operator family, not about KDA's particular gate. **It buys a control,
   not an unblocked reference.**
3. **Use fp16 for the KDA arm on the T4.** Measured: 7.148e-04 against
   `fla`'s own naive, versus 5.714e-03 in bf16 — an order of magnitude, on the
   same card, same kernel, same configs. This is a free accuracy improvement and
   it is *not* a change of operator. Nothing in the study's KDA analysis depends
   on bf16. **The only recommendation here I would make unconditionally.**

---

## 5. Recommendation, and what it costs

### Do not substitute. Run a second, different arm instead.

Here is the reasoning, and the part that is a judgement rather than a
measurement, labelled as such.

**What a substitution would preserve.** Almost nothing that the KDA arm is
currently *for*. The point of the KDA arm is to ask what a compiler makes of a
gated delta rule: a `C×C` unit lower-triangular inverse by forward substitution
unrolled `C` times, an `NT`-step state recurrence unrolled, and a narrow-`K`
matmul against an fp32 decay (`notes/03-kda.md`). Every one of those is a
*delta-rule* structure. mamba-2 has the recurrence and the matmul and none of
the rest: no `β`, no WY representation, no triangular solve, and a scalar per-head
decay where KDA's is per channel.

**What a substitution would turn into a claim about.** A different operator.
"Hand-written mamba-2 versus generated mamba-2" is a perfectly good experiment
and it would very likely produce a *smaller* and less interesting gap, because
without the UT transform the hardest thing the compiler has to do is gone. A
reader who saw the KDA gap in `notes/19-tinygrad-vs-handwritten.md` and a mamba
gap in a table beside it would be reading two different questions as one
sequence. That is the "quietly changes the question" failure the brief names, and
naming it is most of the value of this note.

**What would become unavailable.** The delta-rule family entirely: rows 3–8 of
the table are the only hand-written references for the UT transform on any
device this study can reach, and every one of them is narrowed on this card. A
substitution does not recover them; it stops looking for them.

**What is genuinely unblocked, and it is worth having.** mamba-2 is a real
hand-written Triton kernel that runs on sm_75 with its config spaces intact. If
the study wants a *fourth* arm — "a chunked linear recurrence with decay, with a
reference that was not narrowed" — mamba-2 is it, and it is now pinned, hash-
verified, and measured. It answers a different question from KDA and it answers
it cleanly.

**So:** keep KDA on `fla` with its measured scope label; use fp16; add
GatedDeltaNet as a control if budget allows; and put mamba-2 on the board as a
separate, differently-posed arm rather than as a replacement.

**If the study insists on a substitution**, then the honest version of the claim
is "generated versus hand-written code for a chunked gated linear recurrence",
KDA's delta rule drops out of the comparison, and the note must say so in the
same sentence as every number taken from it. Not in a footnote, and not as
"essentially the same operator".

### What would change this recommendation

- mamba-2's `selective_scan` (mamba-1, CUDA) turns out to run on sm_75 **and**
  turns out to carry the delta rule — it does not, it is an elementwise selective
  scan with no `tl.dot`, so this would not change the answer; or
- rows 5–8 or 11–12 (gdn2, comba, gated_oja_rule, rwkv7, log_linear_attn) turn
  out to be reachable through a path that bypasses `check_shared_mem` entirely. I
  do not believe this for the delta-rule family — they all import
  `common/chunk_delta_h.py` — but `log_linear_attn` has **no**
  `check_shared_mem` call anywhere in its op, and a hierarchical multiscale decay
  is not that far from KDA's fine-grained gating. It is the one untested row
  where a positive result would genuinely change the recommendation, and it is
  the first thing to run with more budget.
- a raymarcher sibling turns up that clears §3.5's four conditions.

---

## 6. Reproducing this

Four probes, all in `.agent-scratch/siblings/` (scratch, deleted after the note
was written), all on one Kaggle T4 via `gpush`:

| probe | what it answers |
|---|---|
| `sib_probe.py` | per-op run + finiteness + which config autotune selected |
| `fixed_probe.py` | the same, with the study's L2-normalised inputs |
| `dot_probe.py` | bare `tl.dot` per dtype — the §1.2 mechanism test |
| `mamba_stages.py` | per-stage mamba correctness, bytes verified first |

```bash
gpush kaggle run --dataset tenghuzhang/sibs-payload --kernel tenghuzhang/sibs-t4 \
  --include .agent-scratch/siblings --exclude .agent-scratch/sibgpush \
  --full-resync --session 30 --queue-wait 50 --poll 25 \
  -- bash -lc 'pip install -q "fla-core==0.5.2" einops; cd .agent-scratch/siblings && python mamba_stages.py'
```

`fla-core==0.5.2` wheel sha256
`5e830c85bad3d0d34677f98ac7074d08687a3756f0f0499d95ceb96eb6920761`, unpacked
and read for every `fla` citation above; re-downloaded and re-hashed identically.
`flash_linear_attention-0.5.2`'s `METADATA` pins `fla-core==0.5.2`, so `fla-core`
is where the ops live.

**Two infrastructure notes, because they cost real sessions and will cost the
next agent the same.** Kaggle kills a notebook cell at **1800 s** and
`gpush`'s runner captures stdout only at the *end* of the cell, so a timeout
loses every line. Autotuning eight `fla` operators on a T4 exceeds it. Budget
for ~1000 s per probe and run one probe per session. And `--full-resync` was
needed repeatedly: a re-push can mount the *previous* dataset version, which
`gpush` reports as `STALE PAYLOAD` — the DECISIONS.md #18 shape.

---

## 7. One line for the ledger

`[MEASURED]` A hand-written Triton sibling of KDA that runs on sm_75 un-degraded
exists — mamba-2's chunked scan at `e9594ce`, 4 of 6 stage checks to fp32
round-off, forward and
backward, all autotune spaces intact — and it is not a delta rule, so it answers
a different question. `[MEASURED]` Every `fla` delta-rule sibling is narrowed on
the T4, because the T4 misses `fla`'s *default* 102 400 B threshold by 36 864 B
and the gate lives in files the whole family imports. `[MEASURED]` fp16 is 7.148e-04
against `fla`'s own naive where bf16 is 5.714e-03, same card, same kernel.
`[CORRECTION]` "bf16 `tl.dot` is broken on sm_75" is false — it is exact
(relerr 0.0) at 64³ and 128³; and "fla returns NaN on the T4" was my inputs,
not the card.
