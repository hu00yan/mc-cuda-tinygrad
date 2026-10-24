---
kind: reference
---

# 08 — Baseline integrity: which KDA baseline is actually running

The comparison in `kda/kda_vs_fla.py` is only worth anything if the thing it
calls "fla" is the thing this study says it is. This note establishes that from
source, and `kda/baseline_pin.py` enforces it on every machine.

**The short version.** `fla`'s KDA is not one kernel; it is a priority-ordered
backend registry. The highest-priority backend that can run on an NVIDIA GPU is
`FlashKDABackend` — a CUTLASS forward, © 2026 Moonshot AI — and it is **enabled by
default**. Separately, the real validity gate is **not tensor cores**: `fla`
narrows its own autotune config lists on a *shared-memory* threshold, so the
verdict is not a function of compute capability at all — sm_89 (4090, L40S, A40,
101 376 B) fails where sm_80 (A100, 166 912 B) passes, despite having the better
tensor cores. Both failure modes are silent. The first is a wrong-object problem
and the second a wrong-config problem; they are independent and both are live.

---

## 0. What this note is based on, and what it is not

Ground truth was read from the released source, not from the audit that prompted
this work:

```
pip download flash-linear-attention --no-deps   ->  flash_linear_attention-0.5.2-py3-none-any.whl
pip download fla-core==0.5.2 --no-deps         ->  fla_core-0.5.2-py3-none-any.whl
```

Both wheels unpacked and read. `flash_linear_attention-0.5.2.dist-info/METADATA`
line `Requires-Dist: fla-core==0.5.2` — so **`fla-core` is where the ops live**
(the `flash-linear-attention` wheel ships only `fla/layers/`; the registry,
`fla/ops/**` and `fla/utils/**` are in `fla-core`). 0.5.2 is the newest release
on PyPI. `kda/kda_vs_fla.py:23` installs it unpinned
(`!pip install -q tinygrad triton flash-linear-attention`), which means **the
baseline is whatever Py served on the day the cell ran**; pin it.

`fla_core-0.5.2.dist-info/METADATA`: `torch>=2.7.0` and `triton>=3.3` for the
cuda extra.

What could **not** be established here, and is marked as such below:

- **No GPU was available on the machine that wrote this note** (darwin/arm64, no
  `nvidia-smi`, no CUDA). Every device number in here is a *threshold read from
  source*, not a measurement. The one measurement-shaped claim — what
  `sharedMemoryPerBlockOptin` a given card reports — is what
  `kda/baseline_pin.py` exists to obtain on the actual card.
- **`flash_kda` was not installable here, and is not installable from PyPI at
  all**: `pip download flash-kda` → *"No matching distribution found"*. It ships
  from `github.com/MoonshotAI/FlashKDA` (default branch `master`, not `main`),
  installed by `git clone` + `pip install -v --no-build-isolation .`. Its
  `setup.py` builds a `CUDAExtension` named `flash_kda_C` with
  `SUPPORTED_CUDA_ARCHS = ["90a", "100a", "103a", "120a"]` and `--use_fast_math`
  in the nvcc flags. **A CUTLASS arm is therefore not reachable from a plain
  `pip install`** — but also not unreachable, because anyone who clones it once
  into an image turns the backend on for every later run in that image.

---

## 1. The registry, in the order the dispatcher walks it

`fla/ops/kda/chunk.py:176` wraps the public op:

```python
@dispatch('kda')
@torch.compiler.disable
def chunk_kda(q, k, v, g, beta, scale=None, initial_state=None, ...):
```

`fla/ops/kda/backends/__init__.py:15-18`:

```python
kda_registry = BackendRegistry("kda")
kda_registry.register(TritonAscendKDABackend())
kda_registry.register(FlashKDABackend())
kda_registry.register(KDATileLangBackend())
```

The walk is `fla/ops/backends/__init__.py:182-211`, in full:

1. backends sorted by `(priority, registration index)` (`:118-126`);
2. skip unless `is_available() and is_enabled()` (`:186-187`);
3. call `<func>_verifier`; skip if it returns False (`:189-198`);
4. skip if the backend does not implement `<func>` (`:200-202`);
5. **first survivor runs**; if nobody survives, the undecorated mainline runs
   (`:214`).

| prio | backend | `is_available()` | `is_enabled()` | implements `chunk_kda`? | what it is |
|---:|---|---|---|:--:|---|
| 0 | `TritonAscendKDABackend` | `IS_NPU` only — **never on NVIDIA** (`triton_ascend/__init__.py:46-49`) | always (`env_var = None`) | no (NPU kernels for gate/intra/WY/bwd) | the whole KDA fwd/bwd for Ascend NPUs |
| 3 | `FlashKDABackend` | `find_spec("flash_kda") is not None` | `FLA_FLASH_KDA != "0"` (`default_enable=True`) | **yes** (`:90-151`) | `flash_kda.fwd` — fused CUTLASS, © 2026 Moonshot AI, `github.com/MoonshotAI/FlashKDA`; docstring: *"Fused CUTLASS forward (replaces the multi-kernel Triton path)"* |
| 5 | `KDATileLangBackend` | `tilelang` importable **and** `has_usable_nvcc()` | `FLA_TILELANG != "0"` (inherited `default_enable=True`) | no — only `chunk_kda_bwd_wy_dqkg_fused` | backward-only, TileLang JIT |
| — | *mainline* | — | — | fallback at `:214` | the Triton path this study means by "fla" |

Two details that decide whether the middle row can fire:

- **Priority 5 and `default_enable=True` for TileLang are inherited**, not
  declared: `BaseBackend.priority = 5` and `default_enable = True` at
  `fla/ops/backends/__init__.py:56-58`. `KDATileLangBackend` overrides neither.
- **TileLang cannot take the forward anyway**, and not because of a predicate:
  it defines no `chunk_kda_verifier`, so `verify()` returns `(True, None)`
  (`:78-88`) and the loop then hits `impl is None` at `:200-202` and continues.
  A backend that passes every gate and still does not run.

### What the registry logs, and why you will not see it

Both the hit and the rejection reasons go to `logger.info`
(`fla/ops/backends/__init__.py:194-197` and `:206-210`), **once per
`(operation, func, backend, outcome)`** key (`:191-193`, `:206-208`). Default
Python logging has no handler above WARNING, so a run that silently dispatched to
CUTLASS prints *nothing* unless the harness turns on INFO for `fla`. The
FlashKDA README itself says the debug line is `[FLA Backend] kda.chunk_kda ->
flashkda`; the installed fla logs `be.backend_type`, i.e. `flash_kda` — so do
not grep the README's spelling, grep for `-> flash`.

---

## 2. What actually arms `FlashKDABackend`

`fla/ops/kda/backends/flash_kda.py:41-88`, in test order — this is the complete
predicate, verbatim, and `baseline_pin.py` prints it on every run:

| # | condition | rejection message |
|--:|---|---|
| 1 | `not torch.is_grad_enabled()` | `FlashKDA only supports inference mode` |
| 2 | `q.dtype == torch.bfloat16` | `FlashKDA requires bfloat16, got {q.dtype}` |
| 3 | `q.shape[-1] == 128` | `FlashKDA requires K=128, got {…}` |
| 4 | `v.shape[-1] == 128` | `FlashKDA requires V=128, got {…}` |
| 5 | `v.shape[2] == q.shape[2]` | `FlashKDA does not support GVA (HV != H)` |
| 6 | `use_gate_in_kernel` | `FlashKDA requires use_gate_in_kernel=True` |
| 7 | `use_qk_l2norm_in_kernel` | `FlashKDA requires use_qk_l2norm_in_kernel=True` |
| 8 | `use_beta_sigmoid_in_kernel` | `FlashKDA requires use_beta_sigmoid_in_kernel=True` |
| 9 | `state_v_first` | `FlashKDA requires state_v_first=True` |
| 10 | `cp_context is None` | `FlashKDA does not support context parallel` |
| 11 | `not return_intermediate_states` | `… does not support return_intermediate_states` |
| 12 | `safe_gate` | `FlashKDA requires safe_gate=True` |

**This is a correction to the brief I was given.** "Activates silently when
`flash_kda` is installed and the run is in `inference_mode` with bf16 and
K=128, V=128" is necessary but *not sufficient*: gates 6–9 and 12 are five
boolean call flags, and the mainline `chunk_kda` defaults **all five to False**
(`fla/ops/kda/chunk.py:187-195` — `use_qk_l2norm_in_kernel`,
`use_gate_in_kernel`, `use_beta_sigmoid_in_kernel`, `state_v_first`, `safe_gate`).

So for the study as written the honest statement is **armed, not fired**:

- the study's own call — `kda/kda_vs_fla.py:600` and `:673`:
  `chunk_kda(t(q), t(k), t(v), t(g), t(beta), use_qk_l2norm_in_kernel=False)`
  under `torch.no_grad()` — is rejected at **gate 6**
  (`FlashKDA requires use_gate_in_kernel=True`), because gate 6 is tested before
  gate 7;
- the fla *layer* `KimiDeltaAttention` passes gates 6–9 as True
  (`fla/layers/kda.py:260-278`: `use_qk_l2norm_in_kernel=True`,
  `use_gate_in_kernel=True`, `use_beta_sigmoid_in_kernel=True`,
  `state_v_first=True`) but its `safe_gate` defaults to **False**
  (`fla/layers/kda.py:93,105`), so it is rejected at **gate 12**.

That is one flag away, in either direction. Turning on `use_gate_in_kernel` (or
`safe_gate`) in the harness, on an image where someone has cloned FlashKDA,
converts a Triton-vs-TinyGrad comparison into a vendor-CUTLASS-vs-TinyGrad
comparison, and nothing in the output says so. Hence the pin.

**Version skew worth knowing:** the FlashKDA README's own example passes
`transpose_state_layout=True`, which is **not a parameter** of
`chunk_kda` in fla 0.5.2 (it has `state_v_first`), and the README is written
against fla PR #852. Run that snippet against 0.5.2 and gate 9 rejects it. If a
future fla release renames the flag, the verifier moves with it and the pin's
rejection reason changes — which is why the pin prints the reason string instead
of a boolean.

### Pinning protocol

- **Required, and cheap:** `export FLA_FLASH_KDA=0` (kills the CUTLASS backend
  specifically) *or* `FLA_DISABLE_BACKEND_DISPATCH=1` (kills the whole
  registry — `fla/ops/backends/__init__.py:28-30` reads it at import, and
  `:167-168` returns the undecorated function).
- **Required, and free:** record the selected backend **in the cell label**,
  produced by `baseline_pin.py`, not reconstructed afterwards. The registry's own
  log line is a fallback, not the primary record: it is INFO-level and
  once-per-process.
- Note that `FLA_DISABLE_BACKEND_DISPATCH=1` is *itself* a configuration the
  pin flags as UNKNOWN, because with dispatch bypassed the registry report is not
  a description of what ran. It is a valid way to run, but then the run's claim
  is "I pinned the Triton path and proved the device can hold it", not "the
  dispatcher chose Triton".

---

## 3. Every environment switch, and which of them change the algorithm

Enumerated by reading every `os.environ` / `os.getenv` in the 0.5.2 wheel
(28 call sites). "algorithmic" means: the switch can change *what is computed*,
not merely *how*, and therefore invalidates a same-algorithm comparison against
TinyGrad. `kda/baseline_pin.py` prints this table with the live values.

### dispatch and backend selection

| var | default | algorithmic | effect | source |
|---|---|:--:|---|---|
| `FLA_DISABLE_BACKEND_DISPATCH` | unset → dispatch **on** | no | `=1` returns the undecorated function; no backend can be selected | `fla/ops/backends/__init__.py:28-30, 167-168` |
| `FLA_FLASH_KDA` | unset → **enabled** | no | `=0` rejects the CUTLASS backend (`default_enable=True`, `:38`) | `fla/ops/kda/backends/flash_kda.py:37-38`; enable test `fla/ops/backends/__init__.py:66-71` |
| `FLA_TILELANG` | unset → **enabled** | no | `=0` rejects TileLang (backward kernel only) | `fla/ops/kda/backends/tilelang/__init__.py:24` |
| `CUDA_HOME` / `CUDA_PATH` | unset | no | only feeds `has_usable_nvcc()`, i.e. whether TileLang counts as available at all | `fla/utils/_compat.py:47-49` |

### numerics

| var | default | algorithmic | effect | source |
|---|---|:--:|---|---|
| `FLA_USE_FAST_OPS` | `'0'` | precision only | `=1` swaps `exp`/`log`/`log2`/`tanh` to the `fast_*` libdevice family. **`exp2` is unchanged**: the fast branch is `tldevice.exp2`, the default branch is `tl.math.exp2` (`fla/ops/utils/op.py:16-37`) — so this switch does **not** touch KDA's dominant nonlinearity, contrary to how it is usually described | `fla/ops/utils/op.py:16-37` |
| `FLA_USE_TMA` | `'0'` | **yes** | `=1`, on CC ≥ 9, with a tensor-descriptor API → `IS_TMA_SUPPORTED`, which changes the load path **and** widens the tril solve's dot-precision autotune list to `{"ieee", FLA_TRIL_PRECISION}` | `fla/utils/_device.py:139-147`; `fla/ops/utils/solve_tril.py:22` |
| `FLA_TRIL_PRECISION` | `'ieee'` | precision | dot precision offered to the tril-solve autotune; only reachable when `IS_TMA_SUPPORTED` | `fla/ops/utils/solve_tril.py:19-22` |
| `TRITON_F32_DEFAULT` | **written by fla**, see §5 | precision, **process-global** | every `tl.dot` in fla that passes no explicit `input_precision` resolves to `'ieee'` instead of `'tf32'` | written `fla/utils/_device.py:149-152`; read `triton/language/core.py` (main `:2267-2270`, v3.5.1 `:2012-2015`) via `knobs.language.fp32_default` (`knobs.py` main `:497`) |
| `TRITON_DEFAULT_FP_FUSION` | `'false'` (triton main) | precision | FMA contraction in fp32 code | triton main `knobs.py:498, 501-502` |
| `TRITON_FORCE_DISABLE_FP_FUSION` | unset | precision | same knob, hard override | triton main `knobs.py:499` |

### config, cache, and the two switches that can retarget codegen

| var | default | algorithmic | effect | source |
|---|---|:--:|---|---|
| `FLA_CACHE_MODE` | **`'disabled'`** | **yes** | anything else lets a JSON file **override Triton's autotune result** for a kernel — `STRICT`/`FUZZY`/`FULL`/`DEFAULT`/`ALWAYS`, including a *fuzzy* match where numeric leaves are ignored ("a config tuned for `seq_len=1024` applies to 2048") | `fla/ops/utils/cache.py:52, 293-334, 367-378` |
| `FLA_CONFIG_DIR` | unset | **yes** | directory those JSONs are read from | `fla/ops/utils/cache.py:107-112` |
| `FLA_GPU_NAME` | unset → `torch.cuda.get_device_name(0)` | **yes** | overrides the GPU name used to pick that directory | `fla/ops/utils/cache.py:83-84` |
| `FLA_CACHE_RESULTS` | **`'1'`** | not by itself | passed as `cache_results=` to `triton.autotune`, so **autotune results persist in the Triton cache dir and are reused by a later process** | `fla/utils/_config.py:11`; `fla/utils/_compat.py:32-33` |
| `TRITON_CACHE_DIR` | unset → `~/.triton/cache` | **yes** | where compiled kernels *and* autotune results live. A pre-warmed cache (baked into a rental image, or a shared home directory) means a "timed" run inherits a config chosen by an earlier process | triton; interacts with the row above |
| `TRITON_OVERRIDE_ARCH` | unset | **yes** | `arch = knobs.runtime.override_arch or f"sm{self.target.arch}"` — compiles for a different arch than the device you are running on | triton main `third_party/nvidia/backend/compiler.py:219` |
| `TRITON_INTERPRET` | unset | **yes** | runs the Triton interpreter, not the compiler | triton main `knobs.py:472` |
| `FLASH_KDA_CUDA_ARCHS` | `'auto'` | **yes, if `flash_kda` is installed** | which archs the CUTLASS extension is built for; `auto` needs a visible CUDA device at build time | `MoonshotAI/FlashKDA` `setup.py`: `SUPPORTED_CUDA_ARCHS = ["90a","100a","103a","120a"]` |

Read but irrelevant to KDA's forward (listed so the list is complete, and so
nobody re-derives it): `FLA_CI_ENV` (`fla/utils/_config.py:10`),
`FLA_DISABLE_TENSOR_CACHE` (`:13`), `FLA_TENSOR_CACHE_SIZE` (`:15`),
`FLA_INTRACARD_MAX_SPLITS` (`fla/ops/common/backends/intracard.py:26`),
`FLA_USE_COMPILE` (`fla/ops/rwkv7/fused_addcmul.py:31`),
`FLA_CONV_BACKEND` (`fla/modules/conv/short_conv.py:87`),
`WALL_ATTN_DEBUG` / `WALL_ATTN_DKV_DIAG_BF16`
(`fla/ops/wall_attn/parallel.py:26,30`).

Two defaults are worth stating out loud because they are *not* the safe ones a
reader would guess: **`FLA_FLASH_KDA` defaults to enabled**, and
**`FLA_CACHE_RESULTS` defaults to on**. Only `FLA_CACHE_MODE` defaults to off.

---

## 4. The validity gate is shared memory, and it is non-monotone

`fla/utils/_device.py:181-192`:

```python
class Backend(Enum):
    ADA = 101376       # RTX 4090
    AMPERE = 166912    # A100
    HOPPER = 232448    # H100
    DEFAULT = 102400   # Default

    @classmethod
    def get_shared_memory(cls, arch):
        try:
            return cls[arch.upper()].value
        except KeyError:
            return cls.DEFAULT.value          # unknown arch -> 102400

@cache
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    device_shared_mem_list = get_all_max_shared_mem()          # :170-178
    max_shared_memory = device_shared_mem_list[tensor_idx]
    return max_shared_memory >= Backend.get_shared_memory(arch)
```

`check_shared_mem()` is the 102 400 `DEFAULT` threshold. `check_shared_mem('ampere')`
is **166 912**, and that is the number that decides the shape of the autotune
space in the kernel the KDA forward actually runs
(`fla/ops/common/chunk_delta_h.py:43-44`, reached as
`fla/ops/kda/chunk_fwd.py:10,94 → chunk_gated_delta_rule_fwd_h`):

```python
for num_warps in GATED_DELTA_RULE_FWD_H_NUM_WARPS
for num_stages in ([2, 3, 4] if check_shared_mem('ampere') else [2, 1])
for BV in ([32, 64] if check_shared_mem('ada') else [32])
```

Three thresholds, and the list each one selects:

| threshold | constant | gates |
|---|---:|---|
| `check_shared_mem('ampere')` | 166 912 | `num_stages = [2,3,4]` vs `[2,1]` in the fwd state kernel |
| `check_shared_mem('ada')` | 101 376 | `BV = [32,64]` vs `[32]` |
| `check_shared_mem()` (no arch → `DEFAULT`) | 102 400 | `BK_LIST = [32,64]` vs `[16,32]`, `BV_LIST = [64,128]` vs `[16,32]` |

101 376 and 102 400 are 1 024 bytes apart, and every consumer card sits at or
below the lower one — so a consumer card is never "partially" in: it is out of
all three at once, by 65 KB.

The same pattern at every site, all keyed on the same read:

| site | what | ≥ 166 912 (A100, H100) | 101 376–166 911 (3090, 4090, L40S, A40) | < 101 376 (T4, V100) |
|---|---|---|---|---|
| `fla/ops/common/chunk_delta_h.py:43` | `num_stages` (fwd state kernel) | `[2, 3, 4]` | `[2, 1]` | `[2, 1]` |
| `fla/ops/common/chunk_delta_h.py:44` | `BV` | `[32, 64]` | `[32, 64]` | `[32]` |
| `fla/ops/common/chunk_delta_h.py:352` | `num_stages` (bwd dhu) | `[2, 3, 4]` | `[1]` | `[1]` |
| `fla/ops/kda/chunk_bwd.py:25` | `BK_LIST` | `[32, 64]` | `[16, 32]` | `[16, 32]` |
| `fla/ops/kda/chunk_bwd.py:26` | `BV_LIST` | `[64, 128]` | `[16, 32]` | `[16, 32]` |
| `fla/ops/gla/chunk.py:21-22` | `BK_LIST`/`BV_LIST` (GLA) | full | narrowed | narrowest |

**The non-monotonicity, stated precisely.** Three quantities decide whether a KDA
comparison measures what it claims, and only one of them is architectural:

- *instruction class* — does the card have bf16 MMA? A100 (sm_80) does; 4090
  (sm_89) does, better; T4 (sm_75) does not.
- *pipeline depth* — `sharedMemoryPerBlockOptin >= 166 912`? A100 yes, H100 yes,
  4090 **no** (101 376), T4 no (65 536).
- *tile width* — `>= 101 376`? A100 yes, 4090 **yes** (exactly on the
  boundary), T4 no.

Resolved per card:

| card | arch | opt-in B | `num_stages` (fwd) | `BV` (fwd) | `BK/BV_LIST` (bwd) | bf16 MMA |
|---|---|---:|---|---|---|:--:|
| A100, A30 | sm_80 | 166 912 | `[2,3,4]` | `[32,64]` | full | yes |
| H100, H200 | sm_90 | 232 448 | `[2,3,4]` | `[32,64]` | full | yes (+ `wgmma`, TMA) |
| B100/B200, GB200 | **sm_100** (datacenter) | 227 KB (audit) | `[2,3,4]` | `[32,64]` | full | yes |
| **RTX PRO 6000 Blackwell, RTX 5090, DGX Spark** | **sm_120** (consumer) | **101 376** | `[2,1]` | `[32,64]` | `[16,32]` | yes (MMAv2, **no** `wgmma`/TMEM) |
| RTX 3090 | sm_86 | 101 376 | `[2,1]` | `[32,64]` | `[16,32]` | yes |
| RTX 4090 | sm_89 | 101 376 | `[2,1]` | `[32,64]` | `[16,32]` | yes |
| L4, L40S, A40 | sm_89 | 101 376 | `[2,1]` | `[32,64]` | `[16,32]` | yes |
| T4 | sm_75 | 65 536 | `[2,1]` | `[32]` | `[16,32]` | **no** |
| V100 | sm_70 | 98 304 (audit) | `[2,1]` | `[32]` | `[16,32]` | f16 only |

The A100/H100/consumer numbers are corroborated twice — fla's own enum comments
(`ADA = 101376  # RTX 4090`, `AMPERE = 166912  # A100`,
`HOPPER = 232448  # H100`, `fla/utils/_device.py:182-184`) and the CUDA
`cudaDevAttrMaxSharedMemoryPerBlockOptin` documentation (101 376 for CC 8.6).
**98 304 (V100) and 227 KB (sm_100) are the audit's figures and were not
verified here** — read them on the card, not from this table.

> ### ⚠ "Blackwell" is two different cards, and this table used to conflate them
>
> **sm_120 consumer Blackwell has 101 376 B, not 227 KB — it fails the gate by
> 65 536 B, exactly like Ada.** The 227 KB figure is **sm_100 datacenter**
> Blackwell (B100/B200/GB200) and it passes. A card's marketing name does not
> tell you which one you have; `sm_120` is what does, and the SKU names (RTX
> PRO 6000 Blackwell, RTX 5090, DGX Spark GB10) are consumer.
>
> This is not a pedantic split. It is the difference between a card that clears
> `num_stages = [2,3,4]` and one that does not, and the only places consumer
> Blackwell silicon is currently offered **for free** are Hugging Face ZeroGPU
> and similar — so a reader following that route would have concluded, from the
> old table, that the free card works. It does not. **Read the number off the
> card with `kda/baseline_pin.py`, which is exactly what it is for.**
>
> Consumer Blackwell is still worth a row for a different reason: it has **MMAv2,
> not `wgmma` or TMEM** (`mma.sync.aligned.m16n8k16`, per the sm_120 adaptation
> notes), so it is a third *data* point on this study's "does the reference
> implementation's advantage survive without warp-specialisation" question, and
> the 708-cell matrix has no measured Blackwell row at all. fla does support it
> explicitly — `fla-org/flash-linear-attention` `c525f49` (2026-06-06) widened
> `IS_NVIDIA_BLACKWELL` from `capability[0] == 10` to `in (10, 12)`.

So the ordering by compute capability is **reversed** by the gate: sm_89 loses
the 3- and 4-stage pipeline that sm_80 has, and nothing in the output says so.
`num_stages ∈ {2,1}` on a kernel that streams `[B,HV,K,V]`-scale state is the
software pipeline effectively off. Meanwhile the sm_89 card keeps `BV = 64`, so
it looks partially equipped while the depth that matters is gone — and the T4,
which is worse at everything, additionally loses `BV = 64`, i.e. the "narrower
config space than a T4" claim is false in the `BV` column and true only in the
`num_stages` and `BK/BV_LIST` columns. Any validity test keyed on `sm_XX` or on
"has tensor cores" gets this backwards. The test this study applies is
per-card and empirical:

> **`sharedMemoryPerBlockOptin >= 166,912` ⇒ the full config space.**

A100 and H100 pass. Consumer Ampere, consumer Ada, A40 and L40S all fail — the
"datacenter-looking" SKUs fail, which is the commercially expensive surprise.
`sm_75` is doubly out: 65 536 B, *and* no bf16 MMA.

Two read paths, and they are not the same number by construction:

- `torch.cuda.get_device_properties(i).shared_memory_per_block_optin` — the CUDA
  field itself (`&cudaDeviceProp::sharedMemPerBlockOptin`,
  `torch/csrc/cuda/Module.cpp`; present in v2.7.0, **absent in v2.5.0**, which
  is why fla-core 0.5.2's `torch>=2.7.0` floor is convenient here);
- `triton.runtime.driver.active.utils.get_device_properties(i)['max_shared_mem']`
  — **literally what fla compares against** (`fla/utils/_device.py:170-178`).
  Its provenance inside triton's compiled extension could not be verified from
  source here; the reported numbers are consistent with
  `cudaDevAttrMaxSharedMemoryPerBlockOptin` (101 376 for 8.6 per the CUDA
  Runtime API docs and an NVIDIA forum answer; 166 912 / 232 448 for A100 /
  H100), so `baseline_pin.py` prints **both** and fails the pin if they disagree
  rather than picking one.
- Fallback when neither is available:
  `cudaDeviceGetAttribute(&v, 97, i)` through `ctypes`, where 97 is
  `cudaDevAttrMaxSharedMemoryPerBlockOptin` (CUDA Runtime API docs, 13.2.0
  archive: *"cudaDevAttrMaxSharedMemoryPerBlockOptin = 97"*).

### A second shared-memory bug: a check that never fires

`fla/ops/kda/chunk_bwd.py:330-333`:

```python
if check_shared_mem('hopper', k.device.index):
    CONST_TILING = 128
elif check_shared_mem:                 # <-- the decorated function object
    CONST_TILING = 64
else:
    CONST_TILING = 32                  # <-- unreachable
```

`check_shared_mem` is a function object, so the `elif` is **always true**: on
every card that is not Hopper, `CONST_TILING` is 64 and the `else` is dead code.
The pattern is correct one function away — the `else` is the narrowing branch
that the audit reads everywhere else. This is backward-only, so it does not
touch the forward the study times; it is recorded because it is the same class
of defect (a shared-memory threshold silently not applied) and because it is
evidence that these lists are not covered by a test.

### The config space is device 0's, always

`fla/utils/_device.py:130-147` computes `IS_NVIDIA_HOPPER`, `IS_NVIDIA_SM100`,
`IS_NVIDIA_BLACKWELL`, `IS_TF32_SUPPORTED` and `IS_TMA_SUPPORTED` from
`torch.cuda.get_device_name(0)` / `torch.cuda.get_device_capability(0)`, and
`check_shared_mem` defaults to `tensor_idx=0`. The config lists
(`BK_LIST`, `BV_LIST`, the `num_stages` comprehensions) are **module-level
constants**, evaluated once at import from device 0. So on a multi-GPU box the
autotune space belongs to device 0 whatever device the kernel runs on, and a
`torch.cuda.set_device(1)` run is measured under another card's thresholds.
Cheap to check, silent otherwise.

### One more narrowing, and it is deliberate

`fla/ops/common/chunk_delta_h.py:23,28`:

```python
NUM_WARPS = [2, 4] if IS_NVIDIA_HOPPER else [2, 4, 8, 16]
# TODO: Triton mainline fixes a Blackwell tl.dot recurrence race.
# Keep this kernel on num_warps=2 for Blackwell until Triton 3.8 is released
GATED_DELTA_RULE_FWD_H_NUM_WARPS = [2] if IS_NVIDIA_BLACKWELL else [2, 4]
```

Two corrections to the audit here. The fwd state kernel's `num_warps` is
`[2, 4]` on **everything** that is not Blackwell, not a Hopper special case;
and on Blackwell it is pinned to `[2]` to dodge a **documented correctness race**
in Triton. The Blackwell arm is therefore a patched baseline whose warp count is
chosen to avoid a known bug — a caveat to report, not a headline.

---

## 5. sm_75: the baseline is IEEE fp32, and that claim survives verification

`fla/utils/_device.py:137, 149-152`:

```python
IS_TF32_SUPPORTED = (IS_NVIDIA and torch.cuda.get_device_capability(0)[0] >= 8)
...
if IS_NVIDIA and not IS_TF32_SUPPORTED:
    # Make old card happy, since triton will use tf32 by default.
    # This is a workaround for old nvidia card.
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
```

Note the condition is `CC major < 8`, so it covers sm_6x, sm_7x **and** sm_75,
not sm_75 alone. And the effect is real, not decorative: triton's `tl.dot`
resolves an unspecified `input_precision` through that variable —
`python/triton/language/core.py` (main `:2267-2270`, v3.5.1 `:2012-2015`):

```python
if input_precision is None:
    supports_tf32 = "tf32" in _semantic.builder.options.allowed_dot_input_precisions
    input_precision = knobs.language.fp32_default or ("tf32" if (supports_tf32 and
                                                      (allow_tf32 or allow_tf32 is None)) else "ieee")
```

with `knobs.language.fp32_default = env_opt_str("TRITON_F32_DEFAULT")`
(`knobs.py` main `:497`). The read happens at `.get()` →
`os.environ` **at call time** (`env_opt_str.get`, `knobs.py:224-225`), so fla's
late write at import is seen by every `tl.dot` compiled afterwards. End to end:
on any NVIDIA card with CC major < 8, **every fp32 `tl.dot` in fla that does not
pass an explicit `input_precision` is IEEE**, and fla's own KDA intra kernel
fixes the one dot it cares about explicitly —
`fla/ops/kda/chunk_intra.py:20-23`:

```python
if IS_TF32_SUPPORTED:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr('tf32')
else:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr('ieee')
```

with that constexpr feeding 12 `tl.dot(..., input_precision=SOLVE_TRIL_DOT_PRECISION)`
sites in that file. Combine it with the hardware: below sm_80 there is no bf16
MMA at all, so a bf16 KDA cell on a T4 is a *pure FMA* kernel computing in *IEEE
fp32* — the two gates fail independently and in the same direction. Any ratio
measured there is a ratio against that configuration, and the row is about
`fla`-on-Turing, not `fla`.

**What I could not verify.** The brief attributes the FMA-forcing to Triton's
`emuTF32 = (capability // 10 >= 8)`. That expression **is** in triton `main`,
`third_party/nvidia/backend/compiler.py:305`, feeding
`passes.ttgpuir.add_f32_dot_tc(pm, emuTF32)` at `:311`. It is **not** in the
released tags: in v3.3.0 and v3.5.1 the same call site is
`passes.ttgpuir.add_f32_dot_tc(pm)` (`compiler.py:256` / `:262`) with no
argument, so the decision is inside the C++ pass and I could not read it from
Python source. Read it in the installed triton before quoting the mechanism; the
*conclusion* above does not depend on it, because it rests on
`SOLVE_TRIL_DOT_PRECISION` and `TRITON_F32_DEFAULT`, both of which are
fla-side and verified.

---

## 6. `IS_GATHER_SUPPORTED`: a switch, but not the algorithm

`fla/utils/_device.py:138` — `IS_GATHER_SUPPORTED = hasattr(triton.language, 'gather')`
— is consumed at `fla/ops/kda/chunk_intra.py:845` and `:955` as
`USE_GATHER=IS_GATHER_SUPPORTED` (two sites, not one), and the kernel branches on
it at `:508-514`, `:606-612` and `:741-749`:

```python
if USE_GATHER:
    b_gn = gather(b_g, tl.full([1, BK], min(BC//2, T - i_ti - 1), dtype=tl.int16), axis=0)
else:
    p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * HV*K + o_k
    b_gn = tl.load(p_gn, mask=m_k, other=0)[None, :]
```

and `fla/ops/utils/op.py:68-78` defines `gather` as a stub returning `None` when
`tl.gather` is absent, "just to make triton compiler happy".

**Correction to the brief.** Both branches read *the same element* — the gate row
at absolute index `i_ti + min(BC//2, T-i_ti-1)` — one from registers, one from
global memory, with slightly different masks (`m_k` vs `tl.arange(0, BK) < K`).
So this is a **codegen / data-movement switch, not a change of the mathematical
algorithm**, and the "an algorithmic change in the baseline invalidates a
same-algorithm comparison" framing is not supported by the source read here. It
still has to be pinned, for three real reasons: it changes the generated code, the
masks differ at the tail, and a Triton upgrade can flip it with no fla change.
`triton.__version__` belongs in the cell label; fla warns below 3.3.0
(`fla/utils/_device.py:46-54`) and fla-core 0.5.2 requires `triton>=3.3`.

---

## 7. This study's float32-division finding does **not** apply to KDA

`notes/02-obs-camera-residuals.md` reports that TinyGrad's float32 division
returns `a · fl32(1/b)` rather than `fl32(a/b)` — 1 ULP wrong in **25.9%–27.0%**
of general `a/b` cases on a divisor-independent draw
(`derivation/divide.py:mechanism_sentence`) — and that the rewrite is in the
Tensor API rather than being a backend's choice. KDA has no division in its
dataflow: the forward substitution
solves `A M = Diag(beta)` where `A = I + StrictTril(Btri)`, and `A` **has a unit
diagonal**, so the elimination is subtraction only — stated at
`kda/tiny_kda.py:176-177` (*"A has unit diagonal, so there is no division"*) and
visible in the loop at `:178-189`, which computes `b[r] -= sum_{j<r} A[r,j] b[j]`
and never divides. That finding therefore carries over to the camera and to
nothing else; a KDA cell must not be labelled with it.

**The live numerical question on KDA is the decay, and it is unverified in both
directions.** The decay is `exp2` on both sides and the two sides do not
necessarily call the same function:

- TinyGrad, CUDA path: `renderer/cstyle.py:423` emits `hexp2` for
  half/bf16 and `exp2(x)` for fp32, and the study's port is fp32
  (`kda/tiny_kda.py:44`, `F = dtypes.float32`) — so it is libdevice `exp2f`
  through NVRTC. (The PTX renderer would instead emit `ex2.approx.{name}`,
  `renderer/ptx.py:20` — a different answer on the same hardware; the backend
  under test decides which.)
- fla: `fla/ops/utils/op.py:31`, `tl.math.exp2(x.to(tl.float32))`, and the
  `FLA_USE_FAST_OPS=1` branch at `:20` is `tldevice.exp2` — the same call, not
  the `fast_expf` family. So the fast-ops switch does not touch `exp2` (§3).

**Nobody has measured the divergence, in either direction.** Treat it as a
required probe, not a claim: a ULP histogram of `exp2` over the actual gate
range on the actual device, then the same over the full KDA output, decomposing
into (a) the decay, (b) the dot precision (`SOLVE_TRIL_DOT_PRECISION`, and
`TRITON_F32_DEFAULT` on pre-Ampere), and (c) the bf16 storage of the fla arm
versus fp32 in the TinyGrad arm. Component (c) is a dtype asymmetry, not a kernel
property, and it will dominate (a) unless both arms are run in the same dtype.
A "6.6e-06 vs fla" number that has not been decomposed is consistent with a decay
mismatch and equally consistent with bf16 rounding, and the two are not
distinguishable by looking harder at the aggregate.

---

## 8. Every way a run can be silently invalid

Ordered by cost of discovery. **Group A is checkable before spending money.**

### A. Before the rental exists — zero marginal cost

| # | route | how it is caught |
|--:|---|---|
| A1 | card has < 166 912 B opt-in shared memory → config space collapsed to `num_stages ∈ {2,1}`, `BV = 32`, `BK ∈ {16,32}`, `BV_o ∈ {16,32}`; the T4 adds IEEE fp32 and no MMA | `kda/baseline_pin.py` §4. **This is the commercially decisive read: it separates a genuinely usable card from a plausible-looking SKU at zero cost, and a datasheet will not tell you — the 4090's 101 376 B is the counterexample that motivates reading it** |
| A2 | the image has `flash_kda` installed (someone cloned FlashKDA once) | `baseline_pin.py` §1: `find_spec('flash_kda')`; the registry is reported live and the armed call is simulated |
| A3 | the image ships a pre-warmed `TRITON_CACHE_DIR` / `~/.triton/cache`, so "autotuned" configs are inherited | `FLA_CACHE_RESULTS=1` is the fla default; check the cache dir is empty or owned by this run |
| A4 | `TRITON_OVERRIDE_ARCH` set in the image | `baseline_pin.py` §5, hard FAIL |
| A5 | fla version unpinned (`kda/kda_vs_fla.py:23`), so the registry may not be the one audited here | `baseline_pin.py` §2 compares the live registry to the 0.5.2 mirror and reports drift — including **new** backends as drift |
| A6 | heterogeneous or multi-GPU box: fla sizes every config list from device 0 | `baseline_pin.py` §4/§9; `--device N` reports the mismatch |

### B. At run start on the rented card — seconds, no extra cost

| # | route | catch |
|--:|---|---|
| B1 | `FLA_FLASH_KDA`, `FLA_DISABLE_BACKEND_DISPATCH`, `FLA_TILELANG` differ between the cells in one table | all printed by §5; a `FLA_USE_FAST_OPS=1` cell next to a `=0` cell is a different operator |
| B2 | `FLA_CACHE_MODE` / `FLA_CONFIG_DIR` / `FLA_GPU_NAME` set → a JSON overrides the autotuned config, possibly by *fuzzy* key | §5, hard FAIL |
| B3 | `FLA_USE_TMA=1` on sm_90+ → different load path **and** a wider dot-precision autotune list | §5, hard FAIL |
| B4 | `TRITON_INTERPRET=1` | §5, hard FAIL |
| B5 | `IS_GATHER_SUPPORTED` flipped by the Triton version | §7; `triton.__version__` in the label |
| B6 | the registry chose a backend other than the Triton mainline, and logged it at INFO where nobody was listening | §3 simulates the real verifiers; the label carries the decision |
| B7 | fla mutates `TRITON_F32_DEFAULT` **process-globally** on import, so it is invisible in a per-command env dump taken before the run | §8 prints the resolved value *after* import, and flags when fla set it |
| B8 | fla registers a TMA or Blackwell scratch allocator on import (`fla/utils/_device.py:159-167`) | visible in the fla INFO log if enabled |

### C. Only visible in the numbers — needs a planned probe

| # | route | catch |
|--:|---|---|
| C1 | the decay (`exp2`) diverges by more ULP than the dtype difference accounts for | §7's ULP probe, decomposed |
| C2 | the two arms ran in different dtypes (fp32 TinyGrad vs bf16 fla) and the ratio absorbs it | run both arms in one dtype, or label the row `confounded` |
| C3 | autotune picked a config from a cache written by another process | A3 |
| C4 | the Blackwell `num_warps=2` pin is load-bearing for correctness, not speed | §4's last block; report, do not lead with it |
| C5 | the fla arm's `safe_dot` wraps `tl.dot` in inline PTX on Blackwell (`fla/ops/utils/op.py:40-61`, `mov.f32 $0, $1;`, working around fla #638 / triton #8695) | the sm_100 baseline is a patched kernel whose generated code is not what the source reads like |

---

## 9. Verification ledger

Every claim above, and how it was checked. "Read" means the file was opened and
the cited lines were seen in `flash-linear-attention` / `fla-core` 0.5.2 as
unpacked from PyPI on 2026-09-27.

| claim | source | how verified |
|---|---|---|
| three backends, registration order, priorities 0/3/5 | `fla/ops/kda/backends/__init__.py:15-18`; each backend class body | read |
| dispatcher walk, fall-through on `impl is None`, mainline fallback, INFO-once logging | `fla/ops/backends/__init__.py:118-126, 182-214, 194-197, 206-210` | read |
| `FLA_DISABLE_BACKEND_DISPATCH` read at import and honoured in the decorator | `fla/ops/backends/__init__.py:28-30, 167-168` | read |
| `is_available` / `is_enabled` semantics and `find_spec_cached` | `fla/ops/backends/__init__.py:60-71`; `fla/utils/_compat.py:33-35` | read |
| FlashKDA is CUTLASS, © 2026 Moonshot AI, priority 3, `default_enable=True`, env `FLA_FLASH_KDA`, calls `flash_kda.fwd` | `fla/ops/kda/backends/flash_kda.py:8, 25-28, 35-39, 115, 139-149` | read |
| the 12 verifier gates, in order, with exact messages | `fla/ops/kda/backends/flash_kda.py:64-88` | read |
| `chunk_kda` is the dispatched entry and all five flags default False | `fla/ops/kda/chunk.py:176, 186-196` | read |
| the layer sets gates 6–9 True but `safe_gate` defaults False | `fla/layers/kda.py:260-278, 93, 105` | read |
| TileLang: inherited priority 5 / `default_enable=True`, no `chunk_kda` impl, needs nvcc | `fla/ops/kda/backends/tilelang/__init__.py:20-28`; `fla/utils/_compat.py:37-72` | read |
| `Backend` enum values and `check_shared_mem` incl. `tensor_idx=0` and `@cache` | `fla/utils/_device.py:181-202` | read |
| `num_stages`/`BV` narrowing in the fwd state kernel | `fla/ops/common/chunk_delta_h.py:43-44` (and `:352-353`) | read |
| KDA reaches that kernel | `fla/ops/kda/chunk_fwd.py:10, 94` | read |
| `BK_LIST`/`BV_LIST` narrowing, KDA and GLA | `fla/ops/kda/chunk_bwd.py:25-26`; `fla/ops/gla/chunk.py:21-22` | read |
| `elif check_shared_mem:` is always true (dead `else`) | `fla/ops/kda/chunk_bwd.py:330-333` | read |
| Blackwell `num_warps=[2]` with a documented race; fwd is `[2,4]` elsewhere | `fla/ops/common/chunk_delta_h.py:23, 25-28` | read |
| `IS_GATHER_SUPPORTED` definition and both consumption sites | `fla/utils/_device.py:138`; `fla/ops/kda/chunk_intra.py:845, 955` | read |
| `USE_GATHER` branches read the same `g` row | `fla/ops/kda/chunk_intra.py:508-514, 606-612, 741-749`; `fla/ops/utils/op.py:68-78` | read |
| `TRITON_F32_DEFAULT` written by fla for CC major < 8 | `fla/utils/_device.py:137, 149-152` | read |
| …and read by `tl.dot` when `input_precision` is unspecified, at call time | triton `language/core.py` main `:2267-2270`, v3.5.1 `:2012-2015`; `knobs.py` main `:497, 224-225` | read from GitHub raw, **triton `main` and v3.5.1** — not the installed version |
| `emuTF32 = (capability // 10 >= 8)` | triton **main** `third_party/nvidia/backend/compiler.py:305, 311` | read; **absent from v3.3.0 / v3.5.1**, where the call is `add_f32_dot_tc(pm)` at `:256` / `:262` — see §5 |
| `SOLVE_TRIL_DOT_PRECISION` tf32/ieee by capability | `fla/ops/kda/chunk_intra.py:20-23` | read |
| `FLA_USE_FAST_OPS` switches exp/log/log2/tanh, **not** exp2 | `fla/ops/utils/op.py:16-37` | read |
| `IS_TMA_SUPPORTED` and the tril dot-precision autotune list | `fla/utils/_device.py:139-147`; `fla/ops/utils/solve_tril.py:19-22` | read |
| config-cache override, fuzzy matching, `FLA_CACHE_MODE` default | `fla/ops/utils/cache.py:27-62, 99-112, 240-251, 271-334, 367-378` | read |
| fla reads device 0 for every architecture predicate | `fla/utils/_device.py:130-147, 196-200` | read |
| `FLA_CACHE_RESULTS` default `'1'` → `cache_results=` to `triton.autotune` | `fla/utils/_config.py:11`; `fla/utils/_compat.py:32-33` | read |
| complete env-var inventory (28 call sites) | every `os.environ`/`os.getenv` in the 0.5.2 wheel | grepped and each call site read |
| `cudaDevAttrMaxSharedMemoryPerBlockOptin = 97` | CUDA Runtime API docs 13.2.0 archive; `cuda.bindings` docs | read |
| torch exposes `shared_memory_per_block_optin` | `torch/csrc/cuda/Module.cpp` (main, v2.7.0 present; v2.5.0 absent) | read |
| `flash_kda` not on PyPI; `git clone` install; sm_90+; arch list; `--use_fast_math` | `pip download flash-kda` error; `MoonshotAI/FlashKDA` README + `setup.py` on `master` | read |
| TinyGrad emits `exp2` (fp32) / `hexp2` (half, bf16) | installed tinygrad 0.14.0 `renderer/cstyle.py:423`; `renderer/ptx.py:20` | read |
| KDA's dataflow has no division (unit diagonal) | `kda/tiny_kda.py:44, 176-189` | read |

### Explicitly unverified

- **[could not verify]** what `triton`'s `add_f32_dot_tc` does inside C++ for
  sm_75 in any *released* version. The conclusion in §5 does not depend on it.
- **[could not verify]** that trit's `max_shared_mem` is
  `cudaDeviceAttrMaxSharedMemoryPerBlockOptin`; it lives in a compiled extension.
  The pin reads both numbers and fails on disagreement instead of assuming.
- **[could not verify]** any *measured* device value. No GPU was available. Every
  card verdict in §4 is a threshold argument, and the first measurement is
  `kda/baseline_pin.py` on the actual card.
- **[could not verify]** the behaviour of any fla version other than 0.5.2. Line
  numbers in `bench/FAIR-BASELINE.md` were written against an older release and
  several do not match 0.5.2 — substance verified, line numbers corrected here
  (`_device.py:138/149-152` not `163/180-183`; `op.py:16-37` not `27-48`;
  `chunk_delta_h.py:43-44` not `52-53`; `chunk_intra.py:845` not `859`).

---

## 10. The cell label

Every KDA cell in every table carries, verbatim, the output of:

```bash
kda/baseline_pin.py --json
```

which records: the live registry and its drift against the 0.5.2 mirror, the
selected backend and **the predicate that decided it** for both the harness's
call and the layer-shaped call, every fla/triton env var with its value and its
default, `sharedMemoryPerBlockOptin` from up to three independent reads with the
VALID/INVALID verdict, compute capability and generation, `IS_GATHER_SUPPORTED`
and the triton version, the resolved `TRITON_F32_DEFAULT`, and the config lists
fla itself built at import. A run whose log does not contain that block is not a
cell in this study.

`--strict` exits nonzero unless the verdict is VALID, and the module raises
`BaselineNotPinned` on import when `BASELINE_PIN_ON_IMPORT=1` — so a harness
cannot start on an unpinned baseline. UNKNOWN is treated as a failure, not a
pass: on a machine with no torch, no triton, no fla and no GPU, the pin reports
UNKNOWN and exits 2, because "I could not check" is not "it is fine".

`kda/baseline_pin.py --selftest` runs 17 scenarios against a **stub** registry
(17/17 pass as written). It checks this file's logic — priority order, the
`impl is None` fall-through, `FLA_FLASH_KDA=0` disarming, the shared-memory
verdict at 65 536 / 101 376 / 102 400 / 166 912 / 232 448, an A100 vs a 4090
with the same tensor cores, drift detection for an unknown backend, and the three
env switches that must fail the pin. It does **not** test real fla, and the
output says so. It earned its keep immediately: it caught `simulate_dispatch`
reading the head count from the wrong key, which had made the simulated GVA check
reject every call.

The one thing `baseline_pin.py` cannot check, and which stays a probe: the
numerical divergence itself (§7).
