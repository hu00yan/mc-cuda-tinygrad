---
kind: reference
---

# 30 — flash-linear-attention on a T4: the gate narrows, it does not refuse

**Run:** `out/gpush/runs/kaggle/kda-t4/20260929-192919-ea5f3f490d70`, 2026-09-29,
all six stages in one session. Payload fingerprint `ea5f3f490d70`, 42 files,
`GPUSH-VERIFY.json` `ok: true` with zero missing/extra/differing. Driver `gpush`,
backend `kaggle`, dataset `tenghuzhang/kda-t4-payload`, kernel
`tenghuzhang/kda-t4`. Stage exit codes `install probe pin gate sass time` = all
zero (`kda_t4/out/run.sh.log`).

> **Where the logs are.** Every `kda_t4/out/X` below is a path *inside the run's
> artefact tree*, i.e.
> `out/gpush/runs/kaggle/kda-t4/20260929-192919-ea5f3f490d70/artifacts/kda_t4/out/X`.
> The card writes into the payload tree, the runner copies the tree into
> `/kaggle/working/artifacts`, and `gpush fetch` brings it home next to the run.

**Box:** Kaggle, Tesla T4 ×2 (sm_75, 14.56 GiB, 40 SM each), driver 580.159.04,
CUDA runtime 13.0, nvcc/ptxas 12.8, python 3.12.13, torch 2.10.0+cu128,
triton 3.6.0, numpy 2.0.2, einops 0.8.2, tinygrad 0.14.0, **fla-core 0.5.2**,
flash-linear-attention 0.5.2. `MEASURED` (`kda_t4/out/probe_env.txt`).
Reproduce with `bash kda_t4/run.sh`.

> **What this note corrects.** `notes/10-t4-results.md:499-507` and
> `notes/17-minecu-spec.md:846` both say a wall-clock comparison against `fla`
> is impossible on the free-tier card. The first attributes it to a preflight
> line that reads `bf16 tensor cores NO (sm<80)`; the second states flatly
> *"KDA does not compile for sm_75 at all"*. **Both are wrong.** fla's
> shared-memory gate is a list-comprehension condition, not a refusal. It
> returns `False`, the autotune config list gets shorter, and `fla` runs. This
> note records what actually ran, what configuration it ran in, what it costs,
> and what its instruction mix is — all on one T4, all measured.

---

## 0. The answer, first

| question | answer | where |
|---|---|---|
| which code path executed? | **fla's mainline Triton path** — the undecorated `chunk_kda`, reached because no registered backend survives. Not CUTLASS, not TileLang, not NPU. | §1 |
| did it refuse on sm_75? | **No.** `check_shared_mem('ampere')` returns `False`; `fla` autotunes over 4 configs and runs, finite. | §2 |
| what configuration? | autotune space **`BV=[32]`, `num_stages=[1,2]`, `num_warps=[2,4]`**; autotune **selected** `BV=32, num_warps=4, num_stages=1` | §2 |
| is it fp32-IEEE? | **Yes, process-globally.** `TRITON_F32_DEFAULT` goes `None → 'ieee'` across `import fla`, in a scrubbed process | §2 |
| does the T4 have a tensor core path? | not for this kernel. **0 tensor-core instructions across all 45 of fla's real cubins**; 8,192 `FFMA` at `BV=32/w=2` where sm_80 emits 128 `HMMA` and 0 `FFMA` | §3 |
| does it cost anything? | 2.14 → 18.5 ms across a 4-point shape ladder, **in that configuration only** | §4 |
| clean-vs-degraded gap? | **open unknown, not estimated** | §5 |

---

## 1. Which object actually ran

`kda/baseline_pin.py` exists because "we compared against fla" can be true of
three different objects (`baseline_pin.py:6-15`). It was run on the card, as a
subprocess, in all three documented modes, and its exit codes and outputs are
kept verbatim as artefacts. `MEASURED` (`kda_t4/out/pin_t4.txt`,
`pin_t4.baseline_pin.strict.txt`, `pin_t4.build_pin.json`).

**Exit codes.** `--json` → 0, `--strict` → **1**, plain report → 0.
`baseline_pin.py:1441` maps `{"VALID":0, "INVALID":1, "UNKNOWN":2}`, so
`--strict` returning 1 says the verdict is **INVALID**, not UNKNOWN.

**Verdict and its complete list of causes — two items, both the same card:**

```
VERDICT              : INVALID
  FAIL   shared-memory gate INVALID: 65,536 < 166,912 (AMPERE),
         num_stages={2,1}, BV={32}, BK/BV_LIST={16,32}
  FAIL   device 1 (Tesla T4) is INVALID: 65,536 < 166,912 (AMPERE), ...
```

`unknowns (0)`. That matters: **nothing about the baseline is undecided.** The
study has zero UNKNOWNs, so the INVALID verdict is a statement about the card's
shared memory and about nothing else.

**The registry, live, in dispatch order** (`fla/ops/kda/backends/__init__.py:15-18`):

| prio | backend | `is_available` | `is_enabled` | implements `chunk_kda` |
|---|---|---|---|---|
| 0 | `TritonAscendKDABackend` | **False** (NPU only) | True | no |
| 3 | `FlashKDABackend` (CUTLASS) | **False** (`flash_kda` not importable) | True | yes |
| 5 | `KDATileLangBackend` | **False** (`tilelang` not importable) | True | no |

`registry_drift` vs the file's own mirror: **0 differences.** The mirror in
`baseline_pin.py:136-163` matches the installed registry exactly.

**Selection, for both call shapes:**

```
study harness  (kda/kda_vs_fla.py:600,673)  ->  triton_ascend(mainline-fallback)
fla layer shape (fla/layers/kda.py:260-278)  ->  triton_ascend(mainline-fallback)
   triton_ascend  skipped   is_available() and is_enabled()   available=False
   flash_kda      skipped   is_available() and is_enabled()   available=False
   tilelang       skipped   is_available() and is_enabled()   available=False
   (mainline)     SELECTED  no backend survived -> undecorated
                  function (fla/ops/backends/__init__.py:214)
```

**So: the object measured is fla's own hand-written Triton `chunk_kda`, the
mainline path, at `fla-core 0.5.2` on triton 3.6.0.** It is not a vendor CUTLASS
kernel, not a naive reference implementation, and not a fallback. This is the
one thing a comparison against fla has to establish and it is now established
from the registry rather than assumed.

**The dtype, and why there was no choice.** `bfloat16`. Three independent
constraints, all recorded rather than asserted:
`FlashKDABackend.chunk_kda_verifier` requires `q.dtype == torch.bfloat16`
(`fla/ops/kda/backends/flash_kda.py:64-88`); the study's own call shape is
`dtype="bfloat16"` (`kda/baseline_pin.py:392`); and fla builds its fp32 gates
and masks around a bf16 payload, so fp16 is not a substitute that leaves the
computation alone — it is a different payload type with a different numerics
story and a different set of compiled kernels. It is called at
`bf16/ieee-fp32-dots` in every scope label below.

### 1a. Two defects in `kda/baseline_pin.py` (read-only here, so reported)

Both are re-derivable on any machine with no GPU, and both are recorded rather
than patched because `kda/` is not this note's to edit.

1. **`--json` does not emit JSON.** `main()` at `:1436-1440` calls
   `print_report(p)` and *then* `json.dumps(p, indent=2, default=str)`. So stdout
   is a human report followed by a JSON fragment, and `json.loads` on the whole
   of it raises at line 2. Independently of that, `Pin` is a `@dataclass`
   (`:827-828`) and dataclasses are not JSON-serialisable, so the `default=str`
   hook fires on the *entire object* and the "JSON" is a single string containing
   `repr(p)`. `kda/baseline_pin.py:37` advertises `--json` as
   *"machine-readable, for a run log"*; it is neither machine-readable nor
   alone. `kda_t4/pin_t4.py` detects both and builds the structured object
   itself from the same `build_pin()` call, labelled as derived.
2. **`flash_linear_attention` is in `DEPS` (`:68`) and can never import.**
   `flash-linear-attention` 0.5.2's wheel has exactly one top-level package and
   its `top_level.txt` reads `fla` — the same directory `fla-core` provides.
   There is no `flash_linear_attention` module. So the fallback at `:89-90`
   (`if not dep("fla") and dep("flash_linear_attention")`) is dead code against
   the version this file claims to target, and the report's dependency table
   shows `flash_linear_attention NO` on a correctly-installed system.
   `MEASURED` on the card (`kda_t4/out/probe_env.txt`) and on the wheel's
   `top_level.txt`.

Neither defect changes the verdict. Both change how loudly a reader should
trust the tool's own output format.

### 1b. The citations hold

`kda_t4/probe_env.py` re-reads 21 of `baseline_pin.py`'s `file:line`
citations in the *installed* wheel and checks the cited token is on the cited
line. **21/21 hold**, four of them off by exactly one line (`_device.py:181`
→ `ADA` is on 182, and the enum block is quoted `:181-185` for a
`:181`-anchored check — harmless). The ones this note leans on hardest —
`_device.py:137` `IS_TF32_SUPPORTED`, `:152` `TRITON_F32_DEFAULT`,
`chunk_delta_h.py:43` `[2, 3, 4]`, `:44` `[32, 64]`,
`chunk.py:176` `@dispatch('kda')`, `backends/__init__.py:16-18` — are all
exact. `MEASURED`.

### 1c. The vendored-extract rationale in this repo is stale

`out/archlab/triton/fla_kda_chunk_h.py:8-11` says fla's PyPI wheel and sdist
"ship only fla/layers and fla/models — fla/ops, where every Triton kernel
lives, is missing from both." **That is no longer true of `fla-core` 0.5.2.**
Its wheel has 319 entries, 264 of them under `fla/ops`, including
`fla/ops/common/chunk_delta_h.py` (794 lines),
`fla/ops/kda/chunk_bwd.py`, `fla/ops/kda/chunk_intra.py` and
`fla/ops/kda/backends/flash_kda.py` — every file `baseline_pin.py` cites.
`MEASURED` from the wheel and from the installed tree on the card. So
`kda_t4/sass_t4.py` compiles the **installed** kernel rather than a vendored
extract, and the extraction it does is a mechanical decorator strip with the
body preserved byte-for-byte (§3, AOT section).

---

## 2. The config collapse: `[MODELLED]` → `[MEASURED]`

Before this note the collapse was a paragraph of inference from source. It is
now a log line. `MEASURED` on the card, `kda_t4/out/gate_observed.txt`.

### 2a. The gate, called

```
IS_TF32_SUPPORTED   False        IS_GATHER_SUPPORTED  True
IS_TMA_SUPPORTED    False        IS_NVIDIA_HOPPER     False
IS_NVIDIA           True         IS_NVIDIA_BLACKWELL  False

Backend:  ADA 101,376   AMPERE 166,912   HOPPER 232,448   DEFAULT 102,400

check_shared_mem('none',   0) = False
check_shared_mem('ada',    0) = False
check_shared_mem('ampere', 0) = False
check_shared_mem('hopper', 0) = False
```

**All four tags fail, including `DEFAULT`.** A T4 is below every threshold fla
defines, not just the two the brief names.

`max_shared_mem`, read four independent ways, **all agreeing on 65,536**:

| reader | device 0 | device 1 |
|---|---|---|
| `fla.utils.get_all_max_shared_mem()` | `[65536, 65536]` | — |
| `triton … get_device_properties(i)['max_shared_mem']` | 65,536 | 65,536 |
| `torch .shared_memory_per_block_optin` | 65,536 | 65,536 |
| `ctypes cudaDeviceGetAttribute(97)` | 65,536 | 65,536 |
| `torch .shared_memory_per_block` — *not* the field fla reads, listed to show they are different numbers | 49,152 | 49,152 |

The box is homogeneous (`heterogeneous: False`), so fla's device-0-only sizing
rule (`baseline_pin.py:554-559`) is not a hazard here. `torch` 2.10.0 does have
`shared_memory_per_block_optin`, which is the field fla-core 0.5.2's `torch>=2.7`
requirement was there to guarantee.

### 2b. The config space, read off the live object

The decorator chain is `Heuristics -> CachedAutotuner`, so
`chunk_gated_delta_rule_fwd_kernel_h_blockdim64.configs` does not exist on the
module-bound object; the autotuner is two `.fn` hops down. Reading it:

```
GATED_DELTA_RULE_FWD_H_NUM_WARPS = [2, 4]

THE CONFIG SPACE, live: 4 configs
    BV=32  num_warps=2  num_stages=2
    BV=32  num_warps=2  num_stages=1
    BV=32  num_warps=4  num_stages=2
    BV=32  num_warps=4  num_stages=1

distinct BV         [32]
distinct num_stages [1, 2]
distinct num_warps  [2, 4]

the model says, from the gate return values:
    num_stages [2, 1]   (check_shared_mem('ampere') = False)
    BV         [32]      (check_shared_mem('ada')    = False)

MODEL vs OBSERVED: AGREE
```

`[MEASURED]`. The collapse is **3×**: an sm_80 card gets
`2 warps × 3 stages × 2 BV = 12` configs, this card gets `2 × 2 × 1 = 4` — one
third of the space. `BV=64`
is gone (it needs ≥ 101,376 B) and the 3- and 4-stage pipeline is gone (it needs
≥ 166,912 B). `num_warps=8` and `16`, which `NUM_WARPS = [2, 4, 8, 16]` allows
on line 23 of the same file, were never in this kernel's list to begin with.

### 2c. The config autotune actually picked

The config space is what was *offered*; the cache is what *won*. After one
forward at `B=1, T=128, H=HV=2, K=V=128, bf16`:

```
Autotuner.cache, after ONE call:
  key=(2, 2, 128, 128, 64, False, bf16, bf16, bf16, bf16, fp32, bf16)
    -> BV=32  num_warps=4  num_stages=1     (from a space of 4)

returned o: shape=(1,128,2,128) dtype=bfloat16 finite=True absmax=6.1523e-02
```

`[MEASURED]`. **`num_stages=1` is the winner on a T4.** That is the pipeline
depth cut all the way off, and it is the autotuner's own verdict on this
silicon rather than an inference from a threshold.

### 2d. The IEEE switch, and a process-global contamination worth naming

`fla/utils/_device.py:149-152` writes `os.environ['TRITON_F32_DEFAULT']='ieee'`
at **import**. Asked in a process whose environment had the variable
explicitly scrubbed:

```
BEFORE fla import: None
AFTER  fla import: 'ieee'
triton knob        : 'ieee'
fla IS_TF32_SUPPORTED  False
SOLVE_TRIL_DOT_PRECISION  constexpr[ieee]
```

`[MEASURED]`. The scrub matters: an earlier version of this probe asked in a
process that had *already* imported fla (section 5 of the same script does) and
reported `BEFORE 'ieee'`, which reads as though someone else had set it.

And then the functional version, which does not read a variable at all — a
32×32×32 `tl.dot` with no explicit `input_precision`, compiled on this card,
PTX searched:

```
DOT_PROBE {"ptx_mma_sync": 0, "ptx_mma": 0, "ptx_fma_rn_f32": 256,
           "ptx_cvt_tf32": 0, "ptx_target": [".version 8.7", ".target sm_75"],
           "n_regs": 59, "n_spills": 0,
           "env_TRITON_F32_DEFAULT": "ieee",
           "triton_knob_fp32_default": "'ieee'"}
```

`[MEASURED]`. Zero MMA, 256 `fma.rn.f32`, in the artifact.

### 2e. What the T4's silicon can do, asked of the compiler, with a control

The same `tl.dot`, AOT-compiled per arch by passing an explicit
`GPUTarget("cuda", cc, 32)`, crossed with the two values of the knob:

```
TRITON_F32_DEFAULT = 'ieee'   <- what fla forces on this card
cc    mma.sync   mma(any)   wgmma  tcgen05   fma.rn.f32   cvt->tf32
75           0          0       0        0          256            0
80           0          0       0        0          256            0
86           0          0       0        0          256            0
89           0          0       0        0          256            0
90           0          0       0        0          256            0
100          0          0       0        0          256            0

TRITON_F32_DEFAULT = 'tf32'   <- what an sm_80+ card resolves to
cc    mma.sync   mma(any)   wgmma  tcgen05   fma.rn.f32   cvt->tf32
75           0          0       0        0          256            0
80           8          8       0        0            0            0
86           8          8       0        0            0            0
89           8          8       0        0            0            0
90           8          8       0        0            0            0
100          8          8       0        0            0            0
```

`MEASURED` (`kda_t4/out/gate_observed.txt` §3, `dot_probe.json`).
The absolute FMA count depends on how Triton unrolls the reduction, so it is
the **0 : 256 ratio and the sign of the difference** that carry the claim, not
the 256.

**This is the sharpest thing in the note, and it is a two-factor result, not a
one-factor one.**

| | sm_75 | sm_80 |
|---|---|---|
| `TRITON_F32_DEFAULT='ieee'` (fla on this card) | 0 MMA, 256 FMA | **0 MMA, 256 FMA** |
| `TRITON_F32_DEFAULT='tf32'` (an A100 would resolve to this) | **0 MMA, 256 FMA** | 8 MMA, 0 FMA |

The two gates fail **independently** and **in the same direction**:

- **the precision switch is enough on its own.** `fla` writes
  `TRITON_F32_DEFAULT='ieee'` into `os.environ` at import, `tl.dot` resolves an
  unspecified `input_precision` through that variable, and *every* fp32 dot in
  the process becomes FMA. On the `ieee` row, **sm_80 loses its MMA too** — so
  a fla's KDA run inside a T4 process would be an FMA kernel even if the tensors
  lived on an A100.
- **the architecture is enough on its own as well.** At cc=75 with `tf32`
  forced, there is still no MMA: there is no tf32 tensor core below sm_80.

**The control is not optional, and getting it wrong is a plausible-looking
wrong answer.** A first version of this probe ran once, inside a process where
`fla` had already set `TRITON_F32_DEFAULT='ieee'`, and returned **0 `mma.sync`
at all six architectures** — which reads as "no architecture has a tensor-core
dot path" and is a statement about the *environment*, not the silicon. Sweeping
the knob is what separates two causes that otherwise look like one.

### 2f. Where observation and model disagreed

They did not, on anything this note rests on:

| modelled from source | observed on the card | agree? |
|---|---|---|
| `check_shared_mem('ampere') = 65536 < 166912 → False` | `False` | ✅ |
| `check_shared_mem('ada') = 65536 < 101376 → False` | `False` | ✅ |
| `num_stages [2,3,4] → [2,1]` | `[1, 2]` as a set | ✅ |
| `BV [32,64] → [32]` | `[32]` | ✅ |
| `num_warps [2,4]` (Blackwell pin is `[2]`, T4 is not Blackwell) | `[2, 4]` | ✅ |
| 4 configs, not 12 | 4 configs | ✅ |
| `IS_TF32_SUPPORTED = cc_major >= 8 → False` | `False` | ✅ |
| `TRITON_F32_DEFAULT='ieee'` written at import | `None → 'ieee'` | ✅ |

**The one place the model was incomplete** is not a disagreement but an
omission worth recording: `baseline_pin.py:221-250` frames the T4 as failing
`AMPERE` and `ADA` and says nothing about `DEFAULT`. `check_shared_mem('none')`
— the 102,400 B threshold — also returns `False` on a T4, and it gates
`BK_LIST`/`BV_LIST` at `fla/ops/kda/chunk_bwd.py:25-26`. Those are backward-only
config lists, so they do not touch any number in §4, but the T4 fails *all four*
of the thresholds in that enum and a table that lists only two of them
understates the collapse.

### 2g. The scope label, assembled from the above

> **fla on T4, degraded configuration — `num_stages=[1, 2]`, `BV=[32]`,
> `num_warps=[2, 4]`, process-global IEEE fp32 (`TRITON_F32_DEFAULT='ieee'`,
> written by fla at import) — this is what the reference does in its worst
> configuration, not what it does at its best.**

short form, for a table cell: `fla/T4 degraded [stages=1/2, BV=32, warps=2/4, ieee-fp32 dots]`

`kda_t4/time_t4.py` **refuses to print an `fla` number if this label is not
present in the session's `gate_observed.json`.** That refusal is the C-04 rule
made mechanical: a number without its configuration is not a result.

---

## 3. The reference's real SASS, per architecture

Two families, from two different artifacts, and they answer two different
questions. The repo's offline method is `tools/archlab/triton_aot.py`
(`GPUTarget("cuda", cc, 32)` + `ptxas` + `cuobjdump -sass`); §3B is the same
method run **on the card**, from the **installed** `fla` source rather than a
vendored extract, and with the arch coverage extended to cc=75.

### 3A. The real artifact — cubins `fla` compiled on this card

One `chunk_kda(B=1, T=256, H=HV=2, K=V=128, bf16)` call, then Triton's on-card
kernel cache is walked for the cubins it just wrote and each is disassembled
with `cuobjdump -sass`. **57 cubins: 45 are fla's own, 12 are this script's
`dot_probe`/`_dot_probe` controls** — the controls exist precisely so that the
"fla's cubins" claim can be read against something that *should* contain a
tensor core.

Per kernel name, summed over every cubin of that kernel:

| kernel | cubins | total SASS | total `FFMA` | TC | max `MUFU.EX2` | max spill LDL/STL |
|---|---|---|---|---|---|---|
| `chunk_local_cumsum_vector_kernel` | 6 | 1,912 | 0 | **0** | 0 | 0 / 0 |
| `chunk_kda_fwd_kernel_intra_token_parallel` | 16 | 4,032 | 141 | **0** | 8 | 0 / 0 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 2 | 25,184 | 12,288 | **0** | 8 | 3,223 / 1,967 |
| `recompute_w_u_fwd_kda_kernel` | 3 | 31,968 | 14,336 | **0** | 128 | 6,337 / 2,295 |
| `chunk_kda_fwd_kernel_inter_solve_fused` | 6 | 50,160 | 23,408 | **0** | 288 | 2,691 / 1,444 |
| `chunk_gla_fwd_kernel_o` | 12 | 217,680 | 75,264 | **0** | 64 | 22,287 / 12,036 |
| `dot_probe` + `_dot_probe` *(control, not fla)* | 12 | 3,680 | 1,536 | **40** | 0 | 0 / 0 |

`fla`'s whole KDA forward on this card is **125,437 `FFMA` and 0 tensor-core
instructions**, and the arithmetic lives almost entirely in two kernels:
`chunk_gla_fwd_kernel_o` (75,264) and `chunk_kda_fwd_kernel_inter_solve_fused`
(23,408) together are 79% of it.

**The control row is the point of the table.** 12 cubins from one `tl.dot`, one
compiler, one process; 5 of them contain a tensor core and 7 do not, and the
5 are exactly the cc ≥ 80 / `tf32` cells of §2e.

**Tensor-core instruction count, summed over all 45 of fla's real cubins on
this card: 0.** `HMMA` 0, `HGMMA` 0, `UTCHMMA` 0, `TCGMMA` 0. Not "few" — none.
The 12 control cubins carry 40, which is 5 rows × 8 `HMMA`, and those 5 are
exactly the cc≥80 / `tf32` cells of §2e. **The same `tl.dot`, the same
compiler, the same process: a tensor core on five of twelve, none on the
seven that fla compiled.** `MEASURED` (`kda_t4/out/sass_t4.txt` §A, cubins and
`cuobjdump -sass` text under `kda_t4/out/sass/`).

Two things worth stopping on:

- **these figures are replicated across two independent runs.** The first run
  (fingerprint `e01fcd9740f9`) and the second (`395d0176a03d`, which re-ran
  `install gate sass` with the aggregator fixed) produce byte-identical
  per-kernel totals, spill counts and the same `0` / `40` split between fla's
  cubins and the controls. Every number in §3A and §3B is from the second.
- **`MUFU.EX2` is not the story on the reference side, and a first pass got
  that wrong.** The naïve reading of the §3B table — "`notes/16` said 3, this
  card says 2–17, so the SFU share collapsed" — is a config artefact, not a
  property of the arch: the state kernel is a *state* kernel and carries almost
  no `exp2` at any arch. The kernels that do carry it are elsewhere:
  `chunk_kda_fwd_kernel_inter_solve_fused` reaches **288** `MUFU.EX2` on this
  card, and `recompute_w_u_fwd_kda_kernel` **128**, against `notes/16`'s 3 for
  a kernel (`chunk_delta_h`) that neither of those is. **The counts are not
  comparable across different kernels** and §3B's `MUFU.EX2` column is only
  meaningful within one row. Recording this because the wrong reading is the
  attractive one and the note would otherwise have shipped it.
- **The state kernel and the output kernel spill heavily on this card.** 3,223
  local loads / 1,967 local stores in the forward state pass, 22,287/12,036 in
  `chunk_gla_fwd_kernel_o`. On a 65,536 B shared-memory budget with a
  `num_stages=1` pipeline there is nowhere else to put it — and §3B shows the
  same source at the same config on sm_80 spilling 0/0 B at `BV=32, w=4`
  against 6,468/7,960 B here.

### 3B. The AOT sweep — same source, same triton, six architectures

A Triton compiler does not need the silicon it targets, so the other cards in
this study's card table can be *compiled* on a T4 by passing an explicit
`GPUTarget`. What comes out is a statement about a compilation at that arch. It
is never a timing.

The source is the **installed** `fla/ops/common/chunk_delta_h.py` (794 lines,
sha256 `ff461bcc40e9cd0b…`), and the kernel is extracted by
`ast.get_source_segment` — 289 lines, byte-identical to the file's own text,
with only `@triton.heuristics({...})` and `@fla_cache_autotune(configs=[...])`
removed. (An earlier extraction sliced from `def` to end-of-file and swept in
every later kernel in the module together with *its* `fla_cache_autotune`
decorator; the stripped copy then failed to import with
`NameError: name 'fla_cache_autotune' is not defined`, which reads like a
source problem and is an extraction problem. `ast` is the fix and the fix is
checked.)

`fla` computes this kernel's autotune list **at import from device 0**, so
importing it in order to ask it about another card is circular. The list is
therefore reconstructed from the gate values in §2a, per arch, and passed as
explicit constexprs:

| cc | opt-in smem | gated config space |
|---|---|---|
| **75** | 65,536 | **4 configs** — `BV=[32]`, stages `[1,2]`, warps `[2,4]` |
| 80 | 166,912 | 12 configs — `BV=[32,64]`, stages `[2,3,4]`, warps `[2,4]` |
| 86 | 166,912 | 12 configs |
| 89 | 101,376 | 8 configs — `BV=[32,64]`, stages `[1,2]`, warps `[2,4]` |
| 90 | 232,448 | 12 configs |
| 100 | 232,448 | 12 configs |

Every configuration in each space was compiled, disassembled, and counted.
`MUFU.EX2` and `FFMA` in this table are exact opcode tallies from
`cuobjdump -sass`; `REG` and `spill st/ld` are from `ptxas -v`, which
`tools/archlab/triton_aot.py:139-145` names as the authoritative source
(`cuobjdump -res-usage` reports the local *frame*, not spill traffic).

**The row that decides everything, and the two it is measured against:**

| | SASS | TC | HMMA | HGMMA | FFMA | MUFU.EX2 | REG | spill st/ld |
|---|---|---|---|---|---|---|---|---|
| **cc=75, `BV=32, w=2`** | **19,328** | **0** | 0 | 0 | **8,192** | 2 | 64 | 12,806 / 14,588 B |
| **cc=75, `BV=32, w=4`** | **10,112** | **0** | 0 | 0 | **4,096** | 17 | 64 | 6,468 / 7,960 B |
| cc=80, `BV=32, w=4` | 1,936 | 64 | 64 | 0 | 0 | 3 | 255 | 0 / 0 B |
| cc=80, `BV=32, w=2` | 4,104 | 128 | 128 | 0 | 0 | 5 | 255 | 1,162 / 1,204 B |

Three numbers, side by side, at matched configuration:

- **`cc=75` / `cc=80` at `BV=32, w=2`: 19,328 SASS vs 4,104 — 4.7×; 8,192
  `FFMA` vs 0; 128 `HMMA` vs 0.** The reference's own kernel, same source, same
  compiler, same config, different arch.
- **register pressure inverts.** 64 registers on sm_75 against 255 on sm_80,
  with 12.8 KB of spill stores where sm_80 spills 1.2 KB. The FMA expansion
  needs registers the tensor-core path does not, and the allocator gives up.
- **and `num_warps=4` is the autotuner's own answer, twice.** §2c selected
  `num_warps=4`, and the w=4 row is **1.91×** smaller in SASS
  (19,328 → 10,112) with **1.98×** less spill traffic (12,806 → 6,468 B stores)
  relative to w=2. The
  autotuner picked the configuration this table independently identifies as the
  better one, on a criterion the table can see and the autotuner cannot.

Full table, 48 configurations, in `kda_t4/out/sass_t4.txt` §C:

```
cc | BV/warps/stages | SASS  | TC  | HMMA | HGMMA | FFMA | EX2 | REG | spill st/ld B | other-TC
75 | 32/2/1          | 19328  |   0 |    0 |     0 | 8192 |   2 |  64 | 12806/14588  | -
75 | 32/2/2          | 19328  |   0 |    0 |     0 | 8192 |   2 |  64 | 12806/14588  | -
75 | 32/4/1          | 10112  |   0 |    0 |     0 | 4096 |  17 |  64 |  6468/7960   | -
75 | 32/4/2          | 10112  |   0 |    0 |     0 | 4096 |  17 |  64 |  6468/7960   | -
75 | 64/4/2  (a)    | 16952  |   0 |    0 |     0 | 8192 |  33 |  64 | 10404/11568  | -
80 | 32/2/2          |  4104  | 128 |  128 |     0 |    0 |   5 | 255 |  1162/1204   | -
80 | 32/4/2          |  1936  |  64 |   64 |     0 |    0 |   3 | 255 |     0/0     | -
80 | 64/2/2          | 11104  | 256 |  256 |     0 |    0 |   5 |  32 | 15044/17708  | -
80 | 64/4/2          |  2768  | 128 |  128 |     0 |    0 |   3 | 255 |   256/252    | -
86 | 32/4/2          |  1944  |  64 |   64 |     0 |    0 |   3 | 255 |    24/24     | -
89 | 32/4/2          |  1944  |  64 |   64 |     0 |    0 |   3 | 255 |    24/24     | -
89 | 32/2/1          |  4016  | 128 |  128 |     0 |    0 |   2 | 255 |  1134/1192   | -
90 | 32/2/2          |  4000  | 128 |  128 |     0 |    0 |   5 | 255 |   852/884    | -
90 | 32/4/2          |  1848  |  16 |    0 |    16 |    0 |   3 | 255 |     0/0     | HGMMA.64
100| 32/2/2          |  3960  | 128 |  128 |     0 |    0 |   5 |  -  |   192/191    | -
100| 32/4/2          |  4552  |  32 |    0 |     0 |    0 |  34 |  -  |     0/0     | UTCHMMA
```

**A correction the brief's framing invites, made by measurement.** The brief
says "`sm_75` has no `HMMA` at all". That is **not** true of the *instruction
set*: sm_75 has `HMMA.1688.F16` and `HMMA.16816.F16` (fp16 in, fp32 accumulate),
which is Turing's second-generation tensor core. It is true of *this kernel*,
and the reason is precision, not silicon: fla's payload is bf16 (there is no
bf16 MMA below sm_80) and the dot precision has been switched to IEEE, so
nothing in the kernel is left in a shape any tensor core can accept. The
`tl.dot` control in §2e is the same fact at the smallest possible size.

### 3C. `notes/16` — checked, not copied

`notes/16` §3.3 reports `128 HMMA` at sm_80 and `16 HGMMA` at sm_90a for the
hand-written side, and §6 quotes `512 FFMA / 268 FMUL / 124 FADD / 128 MUFU.EX2,
0 tensor core` for the generated side. **Those were not read off source.**
`notes/16` §9's reproduction block runs `tools/archlab/triton_aot.py`, which
AOT-compiles and disassembles a cubin, and §3.3 says so explicitly ("`MEASURED`
(mnemonic counts re-grepped by me from the cubins…)"). The method is sound.

The gap was the **arch coverage and the config**, and both are now closed.
Compiling the same kernel at the configuration `notes/16` used
(`BV=64, num_warps=4, num_stages=2` — `tools/archlab/fla_kda_chunk_h_sig.json`):

| arch | this note | `notes/16` §3.3 | match |
|---|---|---|---|
| sm_80 | 2,768 SASS, **128 HMMA**, **3 MUFU.EX2** | "2672, **128**, **3**" | instruction counts ✅ (SASS total +72) |
| sm_90a | 2,536 SASS, **16 HGMMA.64**, **3 MUFU.EX2** | "2632, **16 HGMMA**, **3**" | instruction counts ✅ (SASS total −96) |
| sm_100a | 6,864 SASS, **32 TC, mnemonic `UTCHMMA`**, 66 MUFU.EX2 | "4096, **32**, 66" | number ✅, **label ✗** — see below |
| **sm_75** | **16,952 SASS, 0 TC, 33 MUFU.EX2, REG 64, 10,404/11,568 B spill** | **absent** | — |

So the `notes/16` numbers reproduce on an independent toolchain, from the
installed wheel rather than a vendored extract, with a different triton patch
level in the pipeline. The SASS totals differ by ~3.6% either way
(2,768 vs 2,672; 2,536 vs 2,632), which is the `.llir`/pipelining difference
between the two toolchains; the instruction-family counts are identical at
cc=80 and cc=90.

**The cc=100 row is a genuine disagreement and is not smoothed over.** Both
counts say 32 tensor-core instructions, but this note's mnemonic is `UTCHMMA`
and `notes/16`'s column is headed *HMMA family*. `UTCHMMA` is in neither
`notes/16`'s regex (`:321`: `HMMA|QMMA|IMMA|OMMA|BMMA|HGMMA|TCGMMA|UMMA`) nor
the one used here before it was extended, so a count built on that regex should
have read 0 at cc=100. Either `notes/16` counted more broadly than its own
stated regex, or the two toolchains emit different families for the same
kernel. **That is unresolved, and it is recorded as unresolved rather than
decided in this note's favour.** It does not touch cc=75, where both are zero.

**The row `notes/16` does not have is sm_75, and it is the row this study runs
on.**

One methodological defect in this repo's own instrument, found while using it:
`sass_t4.py` originally counted `HGMMA` with `hist.get("HGMMA", 0)` and
reported **0** for a cubin that plainly contains sixteen `HGMMA.64`, because
`cuobjdump` writes the operand shape into the mnemonic. `tools/archlab/`
shares the assumption. Both are now prefix matches, and the sm_90a row above
is the one that caught it.


---

## 4. The wall clock, scoped

`kda/t4`-side conventions, decided before any number was produced and applied
uniformly:

- **`fla` is called as `chunk_kda(..., output_final_state=False)`, bf16,**
  which returns `(o, None)`. So fla's work here is exactly "produce `o`".
  The TinyGrad ports return `(o, S)`, so the **device column is `o` only** and
  the cost of `S` is measured and printed but kept out of the ratio.
- **inputs are the study's own generator**, `kda/kda_vs_fla.py:296 rnd(seed=0)`,
  L2-normalised `q,k` — the identical function object on both sides.
- **device time is CUDA-event bracketed**, median of 5 after 2 warmups; the
  TinyGrad graph is built fresh *outside* the event window on every iteration
  (§6, last paragraph — the reason is a bug this note nearly shipped).
- **fla's call is timed by the identical instrument in the identical process**,
  immediately before the ports, on the same tensors.

### 4.1 The table

`MEASURED`, `kda_t4/out/time_t4.txt` §3. `x fla` is side/fla.

| shape | side | device ms | min | kernels | **× fla** | scope |
|---|---|---|---|---|---|---|
| B1 T512 H4 | **fla** | **2.135** | 2.110 | — | — | `stages=1/2, BV=32, warps=2/4, ieee-fp32 dots` |
| B1 T512 H4 | `tiny_kda` (naive) | 7,961.7 | 7,901.2 | 6,184 | **3,729** | — |
| B1 T512 H4 | `tiny_kda_opt` | 217.1 | 212.7 | 60 | **102** | — |
| B1 T512 H4 | `tiny_kda_faithful` | 1,591.7 | 1,565.0 | 699 | **746** | — |
| B1 T1024 H8 | **fla** | **4.747** | 4.695 | — | — | `stages=1/2, BV=32, warps=2/4, ieee-fp32 dots` |
| B1 T1024 H8 | `tiny_kda` (naive) | 9,197.8 | 9,082.2 | 6,208 | **1,938** | — |
| B1 T1024 H8 | `tiny_kda_opt` | 304.3 | 300.0 | 76 | **64.1** | — |
| B1 T1024 H8 | `tiny_kda_faithful` | 1,600.0 | 1,589.7 | 715 | **337** | — |
| B2 T1024 H8 | **fla** | **9.155** | 9.125 | — | — | `stages=1/2, BV=32, warps=2/4, ieee-fp32 dots` |
| B2 T1024 H8 | `tiny_kda` (naive) | 9,308.5 | 9,199.3 | 6,208 | **1,017** | — |
| B2 T1024 H8 | `tiny_kda_opt` | 332.3 | 330.4 | 76 | **36.3** | — |
| B2 T1024 H8 | `tiny_kda_faithful` | 2,186.8 | 2,169.4 | 715 | **239** | — |
| B2 T2048 H8 | **fla** | **18.490** | 18.264 | — | — | `stages=1/2, BV=32, warps=2/4, ieee-fp32 dots` |
| B2 T2048 H8 | `tiny_kda` (naive) | 9,541.0 | 9,455.8 | 6,256 | **516** | — |
| B2 T2048 H8 | `tiny_kda_opt` | 520.1 | 516.6 | 108 | **28.1** | — |
| B2 T2048 H8 | `tiny_kda_faithful` | 2,512.9 | 2,484.5 | 747 | **136** | — |

**Run-to-run variance, measured rather than assumed.** An earlier session on
the same payload and the same box
(`out/gpush/runs/kaggle/kda-t4/20260929-161441-e01fcd9740f9`) gives, at
B1 T1024 H8: `fla` 4.992 ms, naive 7,893.1, opt 268.9, faithful 1,415.1 — so
`fla` moved 4.8%, `tiny_kda_opt` 13.2%, `tiny_kda_faithful` 13.0% and
`tiny_kda` 16.5% between two sessions. **The T4 in this box is a shared
instance**, and the framework-side numbers are the noisier ones. Every figure in
the table above is from the single session `20260929-192919-ea5f3f490d70`; the
ratio column is not quoted to better than ~2 significant figures for that
reason, and nothing in §4.2 is claimed to a precision the variance does not
support.

`B*` is batch, `T` sequence length, `H*` heads, `K=V=128`, `C=64`. All three
TinyGrad ports were present in the payload (`kda_tiny.py` 13,213 B,
`tiny_kda_opt.py` 20,327 B, `tiny_kda_faithful.py` 52,975 B); **none was
missing, none was recreated, none was waited for.** `MEASURED`.

**Read every `fla` cell with this and nothing else:**

> *fla on T4, degraded configuration — `num_stages=[1,2]`, `BV=[32]`,
> `num_warps=[2,4]`, process-global IEEE fp32 — this is what the reference does
> in its worst configuration, not what it does at its best.*

The ratio column is consequently **"how far a TinyGrad port is from the
reference in the reference's worst case on this card"**, and it shrinks with
size (3,729 → 1,938 → 1,017 → 516 for the naive port) because `fla` scales with
work while the ports are dominated by a fixed cost that does not. That trend is
a property of *this* pairing and must not be extrapolated.

### 4.2 What the ladder says, which is the more useful half

The three ports are the instrument the brief asks for, and the ordering is not
the one a "the better port is faster" prior predicts:

| | kernels at B1 T1024 H8 | device ms | vs previous |
|---|---|---|---|
| `tiny_kda` (naive, dense recompute, 4,032 dependent scalar reductions in the UT transform) | 6,208 | 9,197.8 | — |
| `tiny_kda_opt` (blocked UT transform, `M` never materialised) | **76** | **304.3** | **30.2× faster, 81.7× fewer kernels** |
| `tiny_kda_faithful` (the kernel's actual strategy: token-parallel diagonals, 12 dots per key chunk, 3-deep block-Schur DAG) | 715 | 1,600.0 | **5.3× slower than `_opt`, 9.4× more kernels** |

Three things follow, and the third is the one this study exists to find.

1. **`tiny_kda_opt` reproduces ledger C-15 on the card.** The naive → opt step
   is 30.2× at 1.05 M elements, and the kernel count falls 6,208 → 76. The
   structural cause was identified on paper before any GPU ran
   (`notes/06` §5b: "count the dependent steps before writing the loop, not
   after"); here it is a measured 30.2× and 81.7× fewer launches. In the
   earlier session the same step read 29.4×, so **the effect is ~30× and not an
   artifact of one run.**
2. **`tiny_kda_faithful` is 5.3× *slower* than `tiny_kda_opt`, and that is not
   a bug in the port.** Its docstring exists to record exactly this: the
   faithful structure adds 639 kernels (the 16×16 diagonal inversions and the
   twelve 16×16×16 dots per key chunk) that `tiny_kda_opt` replaces with dense
   matmuls TinyGrad can schedule as 76 kernels. **On a card with no tensor
   cores, the extra kernel launches cost more than the extra structure buys.**
   So "more faithful to the kernel" and "faster" are independent axes here, and
   the sign of the gap is a property of the arch.
3. **The naive port is launch-bound, not arithmetic-bound.** Its device time is
   7,962 → 9,541 ms across a **16× increase in work** (0.26 M → 4.19 M q
   elements): a **1.20× increase for 16× the work**. At 6,208 launches, the cost is
   the launches. `notes/19` §5.4 measured launch overhead at **under 1.5%** on
   its pair, on 9–40 kernels. Here the same mechanism is the **dominant term**
   on 6,208 launches: the fraction went from negligible to most of the runtime.
   Same mechanism, opposite sign, and the mechanism is identical — a thousand
   tiny kernels — which is the useful part. It also means the naive port's
   3,729× is not a statement about arithmetic, and should not be quoted as one.

### 4.3 Host cost, kept separate

`notes/19` §5.6 measured TinyGrad's host cost at 24× its device cost. Here it
is smaller but still not negligible, and it is never divided into anything:

| shape | `tiny_kda_opt` device | `tiny_kda_opt` host, end-to-end | host/device |
|---|---|---|---|
| B1 T1024 H8 | 304.3 | 728.0 | 2.39× |
| B2 T2048 H8 | 520.1 | 1,173.7 | 2.26× |

`fla` has no analogue worth quoting separately — its whole operator is one
python call into a compiled library — and the asymmetry is reported, not
normalised away.

### 4.4 Agreement, printed separately and not as evidence about the baseline

`notes/06` §5b: two comparisons in this project were invalidated by porting the
wrong object while passing every test. So this check is evidence about the
**math** and about nothing else. Which artifact ran is §1, from the registry.

| shape | pair | cosine | max&#124;d&#124; | median rel | finite |
|---|---|---|---|---|---|
| B1 T512 H4 | `tiny_kda_opt` vs fla | 0.99998914 | 6.271e-04 | 4.367e-03 | true |
| B1 T1024 H8 | `tiny_kda_opt` vs fla | 0.99998898 | 5.874e-04 | 4.426e-03 | true |
| B2 T2048 H8 | `tiny_kda_faithful` vs fla | 0.99998899 | 6.155e-04 | 4.436e-03 | true |

All three ports agree with `fla` to a cosine of 0.99999 and a median relative
error of 4.4e-03, in bf16, on a card with no bf16 MMA. **This is the setup in
which the ledger's warning applies most sharply: three different algorithms, a
green check, and 28× to 3,729× of wall clock between them.** The gap is not a
correctness gap and was never going to be detected here.

The agreement figures are **byte-identical across both sessions** (cosine
0.99998898 and median rel 4.426e-03 at B1 T1024 H8, to every printed digit),
while the timings moved 5–17%. That is the right way round: the numerics are
deterministic, the wall clock on a shared T4 is not, and the two are being
reported by two different instruments.


---

## 5. What this measurement can and cannot reach

**This section is the point of the note, and the first half of it is a refusal.**

### 5.1 The clean-vs-degraded gap: an open unknown, not a number

`MEASURED`: `fla` on one T4, in the configuration §2g labels.

**NOT MEASURED, and not estimated anywhere in this note or in `kda_t4/`:** what
the same code does on a card with ≥ 166,912 B opt-in shared memory **and** cc
major ≥ 8. Scaling any number above by a stage ratio, a tile ratio or a
precision ratio would be inventing a measurement. §3B gets as close as a
compiler can get — it shows that the *same source at the same config* emits
8,192 `FFMA` and 0 `HMMA` on sm_75 against 0 `FFMA` and 128 `HMMA` on sm_80 —
but that is a codegen fact, not a duration, and the mapping from one to the
other is exactly the unknown. The unknown stays open.

### 5.2 The four gap classes, and which this run can reach

The four classes are `notes/19` §6's: **(i)** cannot express, **(ii)** can
express but does not find, **(iii)** not yet implemented, **(iv)** does not
exist for this arch.

| | reachable here? | why |
|---|---|---|
| **(ii) can express, does not find** | **YES** | All three ports are within a finite measured factor of `fla` *on this card in this configuration*, they all compute the same operator, and the residual decomposes: `tiny_kda_opt` is 28–102× off with 60–108 kernels where fla's 6 distinct kernels each carry thousands of instructions. That is a scheduling and codegen question and this run can genuinely address it. |
| **(i) cannot express** | **only negatively** | A port that lands 100×+ off cannot be separated into (i) vs (ii) by timing alone. The `tiny_kda` → `tiny_kda_opt` step is the one place where it can: `tiny_kda` *could* express the blocked transform and the naive one was a choice, and 30.2× says so. |
| **(iii) not yet implemented** | **YES, and §4.2 is the measurement of it** | The three-port ladder exists to be this instrument. `tiny_kda_opt` is 30.2× better than `tiny_kda` and 5.3× better than `tiny_kda_faithful`; which of those is the right target is a (iii) question with a measured cost attached. |
| **(iv) does not exist for this arch** | **NO — structurally unreachable, at any run quality** | See §5.3. |

### 5.3 The one class this measurement cannot reach, and why no amount of running helps

**The one-sentence answer, stated carefully, because the obvious phrasing of it
is wrong.**

The obvious phrasing is *"`fla` uses a tensor core and TinyGrad emits `FFMA`,
so the gap is class (iv)."* **That is not what is happening here.** On this card
`fla` does not use a tensor core either: §3A measured 0 tensor-core
instructions across all 45 of its real cubins, and §2d measured 0 `mma.sync`
against 256 `fma.rn.f32` in the PTX of a `tl.dot` compiled on the device.

So the residual measured in §4.1 is a gap **between two FMA kernels**, and it
is *not* the (iv) gap. The (iv) component is **unobservable, not absent**: it
is switched off on this silicon, and the experiment cannot see how big it is.

- There is no bf16 MMA below sm_80, and
- `fla` writes `TRITON_F32_DEFAULT='ieee'` process-globally at import,

so `fla`'s `tl.dot` cannot reach a tensor core on this card **however long the
run takes**. `MEASURED`, three independent ways: the 45 cubins; the `tl.dot`
PTX; and the §2e control showing cc=75 emits no MMA even with `tf32` forced.

Every one of the ratios in §4.1 is therefore a **lower bound on** the gap
against `fla` in its tensor-core configuration, by an unknown amount, and the
unknown is exactly §5.1. The comparison this study actually wanted — a
hand-written tensor-core kernel against generated FMA — is the one comparison a
T4 cannot supply, for a **more specific reason than the two notes gave**: not
that KDA fails to compile, but that the tensor core is dark on both sides, so
the quantity of interest is multiplied by zero.

**This is the honest residue of the correction at the top of this note.** `fla`
does not refuse on sm_75, so a scoped comparison **is** possible and §4 is it.
What is still unobtainable on the free tier is the (iv) component, and §5.4
item 1 says which cheap card turns it back on.

### 5.4 What would settle the rest, cheapest first

1. **An RTX 4090, for one hour — and the pin's own table says it fails, and the
   table is wrong to be read that way.** `baseline_pin.py:246-247` lists the 4090
   as INVALID, and it is: 101,376 B < 166,912 B, so `check_shared_mem('ampere')`
   is `False` and `num_stages` collapses to `[2,1]`. **But the 4090 is still the
   cheapest card that turns the (iv) component back on**, and the AOT sweep
   proves it rather than assuming it:
   - `101,376 >= 101,376` (ADA), so `check_shared_mem('ada')` is `True` and
     **`BV=[32,64]` survives** — 8 configs instead of 4, and `BV=64` is exactly
     the tile width that doubles the MMA count (64 → 256 `HMMA` at cc=80);
   - cc 8.9 has major 9, so `IS_TF32_SUPPORTED` is `True`, fla's import writes
     **nothing** to `TRITON_F32_DEFAULT`, and the dots stay tf32;
   - and §3B's **cc=89 row emits 128 `HMMA` with 0 `FFMA`** — the tensor core is
     back.

   So the 4090 buys back the tensor cores while keeping a narrowed pipeline,
   which is a legitimate second measurement, and `kda/baseline_pin.py:19-26`'s
   own point — that the constraint is non-monotone in compute capability — cuts
   the other way here. **Not done**: this study has no paid compute, by
   constraint. Recorded so the recommendation is not the wrong one.
2. **Any sm_80+ datacenter card** (A100/A30, H100/H200) — clears all four
   thresholds and is the only configuration that measures `fla` at its best.
   This is what would close §5.1 outright, and it is the expensive version of
   item 1.
3. **`FLA_FLASH_KDA=0` on a card that has `flash_kda` installed** — would arm
   the CUTLASS arm, which is a different comparison entirely. Not this one.
4. **Re-run this payload with `TRITON_F32_DEFAULT` pre-set to `tf32` and
   `TRITON_OVERRIDE_ARCH=sm_80`.** Free, offline, one line, and §2e already
   predicts the answer for the dot path. **Not done, and it should not be**: a
   kernel whose PTX says `sm_80` executing on an sm_75 device is incoherent, and
   the number it produces would be a third thing rather than a better one.
   The §2e AOT matrix is the honest form of that experiment, and it is in this
   note.


---

## 6. Ledger

`kda/baseline_pin.py` is read-only for this work. Six items, five of which are
in `kda/`, are reported rather than fixed.

| # | file:line | what is wrong | consequence | fixed? |
|---|---|---|---|---|
| R-1 | `kda/baseline_pin.py:37`, `:1436-1440` | `--json` prints the human report first, and `json.dumps(p, default=str)` on a `@dataclass` degenerates the JSON to one string | `--json` is neither machine-readable nor alone, against its own docstring | no — `kda/` read-only. `kda_t4/pin_t4.py` builds the object itself and labels it derived |
| R-2 | `kda/baseline_pin.py:68`, `:89-90` | `flash_linear_attention` is in `DEPS`; the 0.5.2 wheel's only top-level package is `fla` | the report shows a dependency as `NO` on a correct install, and the `:89-90` fallback is dead code | no |
| R-3 | `kda/baseline_pin.py:221-250` | the card table lists `AMPERE` and `ADA` but not `DEFAULT` | a T4 fails all four thresholds in that enum; the table names two | no |
| R-4 | `kda/kda_vs_fla.py:640-661` | `time_tinygrad` calls `tiny_kda_chunk(..., realize=False)`; no port accepts a `realize` kwarg (all three end in `return o.numpy(), tS.numpy()`) | the study's own TinyGrad device-time instrument raises `TypeError` against the ports as they stand | no |
| R-5 | `out/archlab/triton/fla_kda_chunk_h.py:8-11` | claims `fla/ops` is absent from the PyPI distributions; `fla-core` 0.5.2 ships 264 files under `fla/ops` | the vendored extract is no longer necessary, and citing it as the source is a stale claim | no — outside this note's scope |
| R-6 | `notes/10-t4-results.md:499-507`, `notes/17-minecu-spec.md:846` | both say a T4 comparison against fla is impossible | corrected by this note | no — `notes/19`–`notes/29` are read-only; **§0 of this note is the correction** |

Plus two things that are this note's own, recorded because the first nearly
shipped a fabricated ~60×: an earlier `kda_t4/time_t4.py` timed `o.realize()`
on an already-realised tensor. `Tensor.realize` is
`to_realize = [x for x in (self,)+lst if not x.uop.is_virtual and not
x.uop.has_buffer_identity()]; if len(to_realize):`
(tinygrad 0.14.0 `tensor.py:420-425`), so every call after the first ran
**nothing** — a flat 0.28 ms across all three ports and all four shapes, which
looks plausible and is a no-op. Fixed by building a fresh unrealised graph
outside the CUDA-event window on every iteration
(`kda_t4/time_t4.py:cuda_ms_prepare`), and the docstring says why. **A
measurement that is flat where the work grows is the signature of this, and it
should be checked before the number is used.**

And a second, in the same file: `sass_t4.py` keyed its per-cubin results by
*kernel name*, so 45 cubins collapsed to 8 entries in the JSON and the aggregate
tensor-core count in the machine-readable artifact became meaningless — while
the human-readable log kept all 45 rows and was right. The disagreement between
the two is how it was caught. **When a script writes both a log and a JSON, check
that they agree; one of them is always a summary and the other is the data.**

---

## 7. Files

```
kda_t4/INSTALL.sh        pinned installs, idempotent, logs what it did
kda_t4/world.py          one input generator, the study's own (kda_vs_fla.py:296)
kda_t4/probe_env.py      the box, and 21 of baseline_pin's citations re-read in the wheel
kda_t4/pin_t4.py         baseline_pin as a subprocess, --strict and --json, exit codes kept
kda_t4/gate_observed.py  the config collapse observed; the dot-precision probe
kda_t4/sass_t4.py        real cubins from this card, plus the AOT per-arch sweep
kda_t4/time_t4.py        the scoped wall clock; refuses to print an fla number unscoped
kda_t4/run.sh            all six stages, one session, stage exit codes
kda_t4/out/              everything the card produced (comes home with the run)
kda_t4/out/sass/         cubins, ptx and cuobjdump -sass text
```

Reproduce:

```bash
./gpush/gpush kaggle run \
  --dataset tenghuzhang/kda-t4-payload --kernel tenghuzhang/kda-t4 \
  --include kda_t4 --include kda --include tools \
  --session 75 --timeout 4000 --cmd bash kda_t4/run.sh
```

### 7a. Why this run has its own dataset and kernel slugs, and one more guard

It does not share `tenghuzhang/gpush-smoke` with the other agents working in
this repository, and that is a measured fact rather than caution. A run at
16:14 local pushed payload fingerprint `75102d1f1604`; its own manifest
recorded the correct file hashes; and the kernel that executed mounted
`ff7abae673bf`, an *earlier* version of the same dataset, belonging to someone
else's run — which that run's `GPUSH-VERIFY.json` records verbatim.

**The integrity gate did not fail. It did the opposite, and that is the
dangerous part.** It verified the tree against the fingerprint **baked into the
notebook that ran**, and that notebook was not mine. On a contended slug the
verify cell confirms a payload that is not the one the caller pushed, because
the dataset version and the notebook are published by two separate API calls
and another agent can republish the first between them. A gate that verifies the
wrong thing is worse than no gate, because it is green. **Separate slugs per
concurrent agent are the fix.**

A second guard, this one behaved: when a scratch directory was left inside
`kda_t4/` and shipped in the payload, the next push was **REFUSED** —

```
[REFUSE] mass-deletion guard  would prune 559/601 = 93.0% of the remote
         manifest (limit 30%)
```

— because removing the scratch would have deleted 93% of the remote tree. The
refusal is in `out/gpush/runs/kaggle/kda-t4/20260929-181629-395d0176a03d` as
the evidence, and `--full-resync` is the documented way through. Scratch for
this work now lives in `out/kda_t4_scratch/`, outside the include list.

### 7b. Stage exit codes, final run

```
install 0   probe 0   pin 0   gate 0   sass 0   time 0
```

`kda_t4/out/run.sh.log` in the run directory. `run.sh` deliberately does **not**
use `set -e`: a stage that fails must not stop the others, because the
interesting outcome is frequently "this half did not work" and the other half
still has to be recorded. The per-stage exit code is the status, and
`time_t4.py` returns 2 if the scope label is missing, which is a *refusal* and
not a measurement.
