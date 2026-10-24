"""
baseline_pin.py - prove which KDA baseline will actually run, before paying for it.

Two independent failure modes are closed here, and they are not the same failure:

  1. WRONG OBJECT.  `flash-linear-attention` does not dispatch Kimi Delta
     Attention to one kernel.  `fla/ops/kda/chunk.py:176` wraps the op in
     `@dispatch('kda')`, and `fla/ops/kda/backends/__init__.py:15-18` registers
     three backends in priority order.  The highest-priority one that can run on
     an NVIDIA GPU is `FlashKDABackend` -- a CUTLASS forward, "Copyright (c)
     2026 Moonshot AI", `fla/ops/kda/backends/flash_kda.py:25-28` -- which is
     enabled by default and whose only gates are "flash_kda is importable" plus a
     per-call verifier.  A run that believes it is comparing against a
     hand-written Triton kernel and is actually running a vendor CUTLASS kernel
     is not a precision problem, it is measuring the wrong object.

  2. WRONG CONFIG.  Even with the Triton path selected, fla's autotune config
     space is narrowed by a *shared-memory* threshold, not by tensor cores
     (`fla/ops/common/chunk_delta_h.py:43-44`, `fla/ops/kda/chunk_bwd.py:25-26`).
     The threshold is `Backend.AMPERE = 166912` against
     `cudaDevAttrMaxSharedMemoryPerBlockOptin`, so the constraint is
     non-monotone in compute capability: an RTX 4090 has far better tensor cores
     than a T4 and gets a *narrower* config space.  The working criterion is
     `sharedMemoryPerBlockOptin >= 166912`, and it is readable before a rental
     is provisioned -- which is the whole point of this file existing as code
     rather than as a claim in a note.

Every fact below carries the file:line it was read from, against
flash-linear-attention 0.5.2 / fla-core 0.5.2 (PyPI wheels unpacked and read;
`flash_linear_attention-0.5.2` METADATA pins `fla-core==0.5.2`).  Where a claim
could not be read from source it says so in the text.  Nothing here asserts a
predicate that was not read.

Run:
    kda/baseline_pin.py                 # print the report
    kda/baseline_pin.py --strict        # nonzero exit unless every gate is VALID
    kda/baseline_pin.py --json          # machine-readable, for a run log

Import:
    from baseline_pin import assert_pinned, print_report
    assert_pinned()                     # raises BaselineNotPinned

No GPU, no torch and no fla are all supported: the script reports UNKNOWN
instead of crashing, and never claims a run that did not happen.  With
BASELINE_PIN_ON_IMPORT=1 in the environment, importing this module runs
`assert_pinned()` -- opt-in, because an import that raises is a hostile default.

Exit codes: 0 all gates VALID, 1 at least one gate INVALID, 2 at least one gate
UNKNOWN (i.e. not enough installed to decide -- not the same as valid).
"""

from __future__ import annotations

import argparse
import importlib.util
import contextlib
import dataclasses
import json
import os
import shutil
import sys
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 0. environment probes.  Every heavy dependency is probed, never imported
#    blindly, and no probe is allowed to raise.  (Same discipline as
#    kda/kda_vs_fla.py:86-95.)
# ---------------------------------------------------------------------------
OPT = {}          # name -> (module | None, error string | None)
DEPS = ("torch", "triton", "fla", "flash_linear_attention", "flash_kda", "tilelang")


def dep(name):
    """The module if it imported, else None. Never raises."""
    return OPT.get(name, (None, "not probed"))[0]


def dep_err(name):
    return OPT.get(name, (None, "not probed"))[1]


def probe_deps():
    for name in DEPS:
        try:
            OPT[name] = (__import__(name), None)
        except BaseException as e:                     # noqa: BLE001
            OPT[name] = (None, f"{type(e).__name__}: {str(e)[:72]}")
    # fla is importable as `fla` (the ops live in the fla-core wheel); the
    # `flash_linear_attention` distribution ships the layers. Treat them as one
    # feature, as kda_vs_fla.py does.
    if not dep("fla") and dep("flash_linear_attention"):
        OPT["fla"] = (dep("flash_linear_attention"), None)
    return OPT


def spec_present(name):
    """`importlib.util.find_spec`, which is what fla itself uses
    (`fla/ops/backends/__init__.py:60-64` -> `fla/utils/_compat.py:33-35`).
    Returns (bool, reason). Never raises: find_spec imports parent packages and
    can raise ModuleNotFoundError."""
    try:
        return importlib.util.find_spec(name) is not None, ""
    except BaseException as e:                          # noqa: BLE001
        return False, f"{type(e).__name__}: {str(e)[:60]}"


def hr(title=""):
    print()
    print("=" * 78)
    if title:
        print(title)
        print("=" * 78)
    else:
        print("=" * 78)


# ---------------------------------------------------------------------------
# 1. the registry, mirrored from source.
#
#     fla/ops/kda/backends/__init__.py:15-18
#       15  kda_registry = BackendRegistry("kda")
#       16  kda_registry.register(TritonAscendKDABackend())
#       17  kda_registry.register(FlashKDABackend())
#       18  kda_registry.register(KDATileLangBackend())
#
#     `BackendRegistry._get_sorted_backends` sorts by (priority, registration
#     index) -- fla/ops/backends/__init__.py:118-126 -- and the dispatch wrapper
#     walks that list, skipping a backend when `is_available() and
#     is_enabled()` is false, when `verify()` returns False, or when the backend
#     does not implement this function name -- :182-211.  The first survivor
#     runs; if none survive, the undecorated mainline function runs (:214).
#
#     This table is a *mirror* of that source, so the script can still say
#     something when fla is not installed.  When fla IS importable the live
#     registry is read and compared against this mirror, and any difference is
#     reported as a hard failure (a new backend can be armed by an upgrade).
# ---------------------------------------------------------------------------
MIRRORED_REGISTRY = [
    dict(order=0, backend_type="triton_ascend", cls="TritonAscendKDABackend",
         priority=0, package_name=None, env_var=None, default_enable=True,
         implements={"chunk_kda_fwd_intra", "chunk_kda_fwd_intra_token_parallel",
                     "recompute_w_u_fwd", "kda_gate_fwd", "kda_gate_bwd",
                     "fused_kda_gate", "kda_gate_chunk_cumsum", "chunk_kda_bwd_*"},
         src="fla/ops/kda/backends/triton_ascend/__init__.py:41-49",
         available_when="IS_NPU only (fla.utils.IS_NPU) -- never on NVIDIA",
         is_kda_forward=False),
    dict(order=1, backend_type="flash_kda", cls="FlashKDABackend",
         priority=3, package_name="flash_kda", env_var="FLA_FLASH_KDA",
         default_enable=True, implements={"chunk_kda"},
         src="fla/ops/kda/backends/flash_kda.py:35-39",
         available_when="`importlib.util.find_spec('flash_kda') is not None`",
         is_kda_forward=True,
         note="CUTLASS forward, (c) 2026 Moonshot AI. Calls flash_kda.fwd (:139)"),
    dict(order=2, backend_type="tilelang", cls="KDATileLangBackend",
         priority=5, package_name="tilelang", env_var="FLA_TILELANG",
         default_enable=True,
         implements={"chunk_kda_bwd_wy_dqkg_fused"},
         src="fla/ops/kda/backends/tilelang/__init__.py:22-28",
         available_when="tilelang importable AND fla.utils.has_usable_nvcc()",
         is_kda_forward=False,
         note="priority 5 and default_enable True are INHERITED from "
              "BaseBackend (fla/ops/backends/__init__.py:56-58); it implements "
              "no chunk_kda_verifier and no chunk_kda, so for the forward the "
              "loop hits `impl is None` and continues (:200-202)"),
]
MIRROR_NOTE = "mirrored from fla-core 0.5.2 source, not read from a live import"

# The Triton implementation this study intends to compare against.  Anything
# else is a different arm and must be labelled as one.
INTENDED_BACKEND = "triton_ascend"     # only registered backend that is Triton
INTENDED_LABEL = "fla mainline Triton path (no backend takes chunk_kda)"

# The 5 call-shape flags FlashKDABackend.chunk_kda_verifier requires, in the
# order the verifier tests them (flash_kda.py:64-88).  Printed so a reader can
# see which one rejected, without reading the source.
FLASHKDA_GATES = [
    ("not torch.is_grad_enabled()", "FlashKDA only supports inference mode"),
    ("q.dtype == torch.bfloat16", "FlashKDA requires bfloat16"),
    ("q.shape[-1] == 128", "FlashKDA requires K=128"),
    ("v.shape[-1] == 128", "FlashKDA requires V=128"),
    ("v.shape[2] == q.shape[2]", "FlashKDA does not support GVA"),
    ("use_gate_in_kernel", "FlashKDA requires use_gate_in_kernel=True"),
    ("use_qk_l2norm_in_kernel", "FlashKDA requires use_qk_l2norm_in_kernel=True"),
    ("use_beta_sigmoid_in_kernel", "FlashKDA requires use_beta_sigmoid_in_kernel=True"),
    ("state_v_first", "FlashKDA requires state_v_first=True"),
    ("cp_context is None", "FlashKDA does not support context parallel"),
    ("not return_intermediate_states", "FlashKDA does not support return_intermediate_states"),
    ("safe_gate", "FlashKDA requires safe_gate=True"),
]

# ---------------------------------------------------------------------------
# 2. the shared-memory gate.
#
#     fla/utils/_device.py:181-192
#       181  class Backend(Enum):
#       182      ADA = 101376       # RTX 4090
#       183      AMPERE = 166912    # A100
#       184      HOPPER = 232448    # H100
#       185      DEFAULT = 102400   # Default
#       188      def get_shared_memory(cls, arch): ...  # unknown arch -> DEFAULT
#       196  def check_shared_mem(arch="none", tensor_idx=0):   # @cache
#       198      device_shared_mem_list = get_all_max_shared_mem()
#       200      return max_shared_memory >= Backend.get_shared_memory(arch)
#
#     and the read it compares against:
#       170  def get_all_max_shared_mem(): triton ... get_device_properties(i)['max_shared_mem']
#
#     `check_shared_mem()` (no arch) is the 102400 DEFAULT threshold;
#     `check_shared_mem('ampere')` is 166912.  166912 is the one that matters:
#     it is what selects `num_stages=[2,3,4]` and `BV=[32,64]` in the kernel
#     that the KDA forward actually runs
#     (fla/ops/common/chunk_delta_h.py:43-44, reached via
#     fla/ops/kda/chunk_fwd.py:10,94 -> chunk_gated_delta_rule_fwd_h).
# ---------------------------------------------------------------------------
SMEM_AMPERE = 166912
SMEM_DEFAULT = 102400
SMEM_ADA = 101376
SMEM_HOPPER = 232448
SMEM_BLACKWELL_DATACENTER = 232448   # sm_100  B100/B200/GB200  -- audit figure, unverified
SMEM_BLACKWELL_CONSUMER = 101376     # sm_120  RTX PRO 6000 Blackwell, RTX 5090, DGX Spark
VALID_SMEM = SMEM_AMPERE          # the study's working validity criterion

# Which card to rent, decided BEFORE renting rather than after.  The trap this
# table exists to close: **"Blackwell" is two different cards**, and the one
# anybody can get for free is the one that fails.
#
#   sm_120 consumer Blackwell has 101_376 B, NOT 227 KB.  It misses the gate by
#   65_536 B -- exactly like Ada.  227 KB is sm_100 *datacenter* Blackwell, which
#   passes.  The SKU names (RTX PRO 6000 Blackwell, RTX 5090, DGX Spark GB10) are
#   all consumer, so a name-based choice spends money on a card that lands in the
#   same bucket as a T4.
#
# `None` in the opt-in column means the figure is the audit's and was not read
# off a card; treat it as unverified.  Read the real number with
# `python kda/baseline_pin.py`, which prints it from the device.
#
# Sources: fla `fla/utils/_device.py:181-185` for the four thresholds the kernel
# actually keys on; `fla-org/flash-linear-attention` `c525f49` (2026-06-06) for
# fla running on sm_120 at all; the sm_120 shared-memory and MMAv2 figures are
# from the sm_120 adaptation notes and are UNVERIFIED on this host.
CARD_GATE = [
    # name,                                    sm,     opt-in B,             clears gate?
    ("A100 / A30",                              "80", SMEM_AMPERE,            True),
    ("H100 / H200",                             "90", SMEM_HOPPER,            True),
    ("B100 / B200 / GB200 (DATACENTER)",        "100", SMEM_BLACKWELL_DATACENTER, True),
    ("RTX PRO 6000 Blackwell (CONSUMER)",        "120", SMEM_BLACKWELL_CONSUMER, False),
    ("RTX 5090 / DGX Spark GB10 (CONSUMER)",    "120", SMEM_BLACKWELL_CONSUMER, False),
    ("RTX 3090",                                "86", SMEM_ADA,               False),
    ("L4 / L40S / A40 / RTX 4090",              "89", SMEM_ADA,               False),
    ("T4",                                      "75", 65536,                  False),
    ("V100",                                    "70", 98304,                  False),
]

# Autotune-space narrowing sites, all `check_shared_mem(...)`-keyed.  Printed as
# "what the config space would be" so a change in fla is visible in the log.
NARROWING_SITES = [
    ("fla/ops/common/chunk_delta_h.py:43", "num_stages",
     lambda sm: [2, 3, 4] if sm >= SMEM_AMPERE else [2, 1],
     "fwd state kernel; this is the KDA forward's gated config"),
    ("fla/ops/common/chunk_delta_h.py:44", "BV",
     lambda sm: [32, 64] if sm >= SMEM_ADA else [32],
     "fwd state kernel"),
    ("fla/ops/common/chunk_delta_h.py:352", "num_stages (bwd dhu)",
     lambda sm: [2, 3, 4] if sm >= SMEM_AMPERE else [1],
     "backward only"),
    ("fla/ops/common/chunk_delta_h.py:353", "BV (bwd dhu)",
     lambda sm: [32, 64] if sm >= SMEM_ADA else [32],
     "backward only"),
    ("fla/ops/kda/chunk_bwd.py:25", "BK_LIST",
     lambda sm: [32, 64] if sm >= SMEM_DEFAULT else [16, 32],
     "backward only"),
    ("fla/ops/kda/chunk_bwd.py:26", "BV_LIST",
     lambda sm: [64, 128] if sm >= SMEM_AMPERE else [16, 32],
     "backward only"),
    ("fla/ops/gla/chunk.py:21", "BK_LIST (gla)",
     lambda sm: [32, 64] if sm >= SMEM_DEFAULT else [16, 32],
     "same pattern, GLA -- cited by the audit as the instance"),
    ("fla/ops/gla/chunk.py:22", "BV_LIST (gla)",
     lambda sm: [64, 128] if sm >= SMEM_AMPERE else [16, 32],
     "same pattern, GLA"),
    # --- re-derived from the SHIPPED fla-core 0.5.2 wheel, not from GitHub main.
    # Two KDA sites were missing from the list above. The first is FORWARD, so
    # the scope label attached to a KDA forward timing was understating the
    # degradation it was supposed to be scoping.
    ("fla/ops/kda/gate.py:22", "BS_LIST (fwd gate)",
     lambda sm: [32, 64] if sm >= SMEM_DEFAULT else [16, 32],
     "FWD. kda_gate_fwd / kda_gate_fwd_kernel; the gate/decay kernel"),
    ("fla/ops/kda/wy_fast.py:332", "CONST_TILING (bwd wy)",
     lambda sm: 64 if sm >= SMEM_DEFAULT else 32,
     "bwd only. prepare_wy_repr_bwd; caps BK and BV at :333-334"),
    ("fla/ops/kda/chunk_bwd.py:330", "backward path selection",
     lambda sm: "hopper" if sm >= SMEM_HOPPER else ("ampere" if sm >= SMEM_AMPERE else "fallback"),
     "bwd only. :330 dispatch, :332 the else branch"),
    ("fla/ops/kda/backends/tilelang/chunk_bwd_dqkg.py:433", "CONST_TILING (tilelang bwd)",
     lambda sm: 64 if sm >= SMEM_DEFAULT else 32,
     "bwd only, and only on the tilelang backend"),
]

# The T4 (sm_75, 65 536 B opt-in) resolves every one of these to its degraded
# branch. Computed, not asserted, so a card change cannot silently invalidate
# the scope label. `notes/15-experiment-b.md` §3 carries the same table as prose.
def resolve_narrowing(opt_in_bytes: int):
    """What fla actually selects on a card with `opt_in_bytes` of shared memory."""
    return [(site, what, fn(opt_in_bytes), note)
            for site, what, fn, note in NARROWING_SITES]

# fwd state kernel num_warps, fla/ops/common/chunk_delta_h.py:23,28
def num_warps_fwd_h(arch_tag):
    # line 28: [2] if IS_NVIDIA_BLACKWELL else [2, 4]  -- unconditional, not a
    # Hopper special case, and narrowed further on Blackwell to dodge a
    # documented tl.dot recurrence race (comment at :25-27).
    return [2] if arch_tag in ("sm_100", "sm_103", "sm_120", "sm_121") else [2, 4]

# ---------------------------------------------------------------------------
# 3. every environment variable fla-core 0.5.2 reads, with the default that
#    applies when it is unset, what it changes, and whether that is an
#    ALGORITHM change (which invalidates a same-algorithm comparison) or only an
#    implementation/precision change.  Enumerated by reading every os.environ /
#    os.getenv call in the wheel; the `src` field is the call site.
# ---------------------------------------------------------------------------
# (name, default, algorithmic, what it changes, src)
FLA_ENV = [
    ("FLA_DISABLE_BACKEND_DISPATCH", "(unset -> dispatch ON)", "no",
     "=1 returns the undecorated function, so no backend can be selected",
     "fla/ops/backends/__init__.py:28-30,167-168"),
    ("FLA_FLASH_KDA", "(unset -> enabled)", "no",
     "=0 rejects the CUTLASS backend; default_enable=True (:38)",
     "fla/ops/kda/backends/flash_kda.py:37-38, 66-71 (BaseBackend.is_enabled)"),
    ("FLA_TILELANG", "(unset -> enabled)", "no",
     "=0 rejects the TileLang backend (backward kernel only)",
     "fla/ops/kda/backends/tilelang/__init__.py:24; enabled check "
     "fla/ops/backends/__init__.py:66-71"),
    ("FLA_USE_FAST_OPS", "'0'", "no (precision)",
     "=1 switches exp/log/log2/tanh to the fast_* libdevice family. NOTE: "
     "exp2 is `tldevice.exp2` in BOTH branches, so exp2 is unaffected",
     "fla/ops/utils/op.py:16-37"),
    ("FLA_USE_TMA", "'0'", "yes (autotune space + load path)",
     "=1 on CC>=9 with a tensor-descriptor API enables IS_TMA_SUPPORTED, which "
     "also widens the tril solve's dot-precision autotune list",
     "fla/utils/_device.py:139-147; fla/ops/utils/solve_tril.py:22"),
    ("FLA_TRIL_PRECISION", "'ieee'", "no (precision)",
     "dot precision offered to the tril-solve autotune (only reachable when "
     "IS_TMA_SUPPORTED, per solve_tril.py:22)",
     "fla/ops/utils/solve_tril.py:19-22"),
    ("FLA_CACHE_MODE", "'disabled'", "yes (picks the config)",
     "anything but 'disabled' lets a JSON file OVERRIDE Triton's autotune "
     "result for a kernel, by exact or FUZZY key match",
     "fla/ops/utils/cache.py:52, 293-334, 367-378"),
    ("FLA_CONFIG_DIR", "(unset)", "yes (picks the config)",
     "directory the override JSONs are read from",
     "fla/ops/utils/cache.py:107-112"),
    ("FLA_GPU_NAME", "(unset -> torch.cuda.get_device_name(0))", "yes (picks the config)",
     "overrides the GPU name used to choose that directory",
     "fla/ops/utils/cache.py:83-84"),
    ("FLA_CACHE_RESULTS", "'1'", "no (but changes what is re-tuned)",
     "passes cache_results=... to triton.autotune, so autotune results persist "
     "in the Triton cache dir and can be reused by a later process",
     "fla/utils/_config.py:11; fla/utils/_compat.py:32-33"),
    ("FLA_CI_ENV", "(unset)", "no",
     "=1 marks CI; read by fla/utils/_config.py:10",
     "fla/utils/_config.py:10"),
    ("FLA_DISABLE_TENSOR_CACHE", "'0'", "no",
     "=1 disables fla's tensor cache", "fla/utils/_config.py:13"),
    ("FLA_TENSOR_CACHE_SIZE", "'4'", "no",
     "tensor-cache size", "fla/utils/_config.py:15"),
    ("FLA_INTRACARD_MAX_SPLITS", "'32'", "no",
     "intracard backend split cap (not the KDA forward)",
     "fla/ops/common/backends/intracard.py:26"),
    ("FLA_USE_COMPILE", "'1'", "no",
     "rwkv7 fused_addcmul only; KDA does not read it",
     "fla/ops/rwkv7/fused_addcmul.py:31"),
    ("FLA_CONV_BACKEND", "(unset)", "no",
     "short conv only; KDA does not read it", "fla/modules/conv/short_conv.py:87"),
    ("CUDA_HOME / CUDA_PATH", "(unset)", "no",
     "only affects has_usable_nvcc(), i.e. whether the TileLang backend counts "
     "as available", "fla/utils/_compat.py:47-49"),
    # written by fla, not read from the environment: the dangerous one
    ("TRITON_F32_DEFAULT", "SET BY FLA to 'ieee' when IS_NVIDIA and not "
     "IS_TF32_SUPPORTED, i.e. every NVIDIA card with CC major < 8",
     "no (precision, process-global)",
     "every `tl.dot` in fla that passes no explicit input_precision resolves to "
     "'ieee' instead of 'tf32'",
     "written fla/utils/_device.py:149-152; read triton "
     "python/triton/language/core.py (main:2267-2270, v3.5.1:2012-2015) via "
     "knobs.language.fp32_default (triton knobs.py, main:497)"),
    # triton knobs that can silently retarget or re-precise the baseline
    ("TRITON_OVERRIDE_ARCH", "(unset)", "yes (retargets codegen)",
     "compiles for a different arch than the device you are running on",
     "triton main third_party/nvidia/backend/compiler.py:219 "
     "(`knobs.runtime.override_arch or f'sm{self.target.arch}'`)"),
    ("TRITON_DEFAULT_FP_FUSION", "'false'", "no (precision)",
     "FMA contraction in fp32 elementwise/matmul code",
     "triton main python/triton/knobs.py:498,501-502"),
    ("TRITON_FORCE_DISABLE_FP_FUSION", "(unset)", "no (precision)",
     "same knob, hard override", "triton main python/triton/knobs.py:499"),
    ("TRITON_INTERPRET", "(unset)", "yes (different engine)",
     "runs the Triton interpreter instead of compiling",
     "triton main python/triton/knobs.py:472"),
    ("TRITON_CACHE_DIR", "(unset -> ~/.triton/cache)", "yes (reuses results)",
     "where compiled kernels AND autotune results live; a pre-warmed cache "
     "means a 'timed' run inherits another process's config choice",
     "triton; interacts with FLA_CACHE_RESULTS above"),
    ("FLASH_KDA_CUDA_ARCHS", "'auto'", "yes (if flash_kda is installed)",
     "which archs the CUTLASS extension is compiled for; 'auto' needs a "
     "visible CUDA device at build time",
     "MoonshotAI/FlashKDA setup.py (master): SUPPORTED_CUDA_ARCHS = "
     "['90a','100a','103a','120a']"),
]

# The flags the study's own harness passes, read from
# kda/kda_vs_fla.py:600 and :673:
#   chunk_kda(t(q), t(k), t(v), t(g), t(beta), use_qk_l2norm_in_kernel=False)
# under torch.no_grad().  The FlashKDA verifier tests use_gate_in_kernel
# *before* use_qk_l2norm_in_kernel, so the deciding rejection for this call is
# the former.  Recorded here so the report and the code cannot drift apart.
STUDY_CALL = dict(
    src="kda/kda_vs_fla.py:600,673",
    dtype="bfloat16",
    K=128, V=128, H=16, HV=16,
    use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False, state_v_first=False, safe_gate=False,
    grad_enabled=False,
)
# What the fla *layer* passes (fla/layers/kda.py:260-278), i.e. how close the
# trap is to being armed by an innocuous edit.  safe_gate defaults False in the
# layer (fla/layers/kda.py:93,105), so the layer is not armed either.
LAYER_CALL = dict(
    src="fla/layers/kda.py:260-278 (KimiDeltaAttention, chunked branch)",
    dtype="bfloat16",
    K=128, V=128, H=16, HV=16,
    use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True, state_v_first=True, safe_gate=True,
    grad_enabled=False,
)

# ---------------------------------------------------------------------------
# 4. device reads.  Three independent ways to get the number that gates
#    everything, so a disagreement is visible instead of invisible:
#      a) torch.cuda.get_device_properties(i).shared_memory_per_block_optin
#         -- the CUDA field itself, `&cudaDeviceProp::sharedMemPerBlockOptin`,
#         exposed by torch/csrc/cuda/Module.cpp (present in v2.7.0, absent in
#         v2.5.0).  fla-core 0.5.2 requires torch>=2.7.0, so this is the
#         expected path.
#      b) triton ... get_device_properties(i)['max_shared_mem'] -- literally what
#         fla compares against (fla/utils/_device.py:170-178).  Its provenance
#         inside triton's compiled extension could NOT be verified from source
#         here, so (a) and (b) are reported side by side and a mismatch is a
#         failure.
#      c) cudaDeviceGetAttribute(&v, 97 /* MaxSharedMemoryPerBlockOptin */, i)
#         through ctypes.  Enum value 97 per the CUDA Runtime API docs
#         (docs.nvidia.com/cuda/archive/13.2.0/.../group__CUDART__TYPES.html).
# ---------------------------------------------------------------------------
CUDA_ATTR_MAX_SHMEM_OPTIN = 97


def arch_name(major, minor=None):
    if major is None:
        return "?"
    if minor is None:
        return f"sm_{major}"
    # triton spells consumer Blackwell as sm_120a etc; for reporting, major
    # digits are what matters to every threshold in fla.
    return f"sm_{major}{minor}"


def smem_via_ctypes(idx):
    """(value, source) or (None, reason). Used only when torch/triton cannot say."""
    lib = None
    for cand in ("libcudart.so", "libcudart.so.12", "libcudart.so.13",
                 "libcudart.dylib"):
        try:
            lib = __import__("ctypes").CDLL(cand)
            break
        except BaseException:                           # noqa: BLE001
            continue
    if lib is None:
        return None, "libcudart not loadable"
    import ctypes
    v = ctypes.c_int(0)
    try:
        rc = lib.cudaDeviceGetAttribute(ctypes.byref(v), CUDA_ATTR_MAX_SHMEM_OPTIN, ctypes.c_int(idx))
    except BaseException as e:                          # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:48]}"
    if rc != 0:
        return None, f"cudaDeviceGetAttribute rc={rc}"
    return int(v.value), f"libcudart cudaDeviceGetAttribute({CUDA_ATTR_MAX_SHMEM_OPTIN})"


def read_device(idx=0):
    """Everything this file needs to know about one device. Never raises."""
    out = dict(index=idx, name=None, cc=None, smem_optin=None, smem_src=None,
               smem_triton=None, smem_fla=None, note=[])
    torch = dep("torch")
    if torch is None:
        out["note"].append(f"torch unavailable ({dep_err('torch')})")
        return out
    if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
        out["note"].append("torch present but no CUDA device visible")
        return out
    try:
        cap = torch.cuda.get_device_capability(idx)
        out["cc"] = tuple(cap)
        out["name"] = torch.cuda.get_device_name(idx)
    except BaseException as e:                          # noqa: BLE001
        out["note"].append(f"device props failed: {type(e).__name__}")
        return out
    # (a) the CUDA field itself
    try:
        p = torch.cuda.get_device_properties(idx)
        if hasattr(p, "shared_memory_per_block_optin"):
            out["smem_optin"] = int(p.shared_memory_per_block_optin)
            out["smem_src"] = "torch .shared_memory_per_block_optin"
        else:
            out["note"].append("this torch has no shared_memory_per_block_optin "
                               "(absent in torch<=2.5; fla-core 0.5.2 wants >=2.7)")
    except BaseException as e:                          # noqa: BLE001
        out["note"].append(f"prop read failed: {type(e).__name__}: {str(e)[:40]}")
    # (c) ctypes fallback
    if out["smem_optin"] is None:
        v, src = smem_via_ctypes(idx)
        if v is not None:
            out["smem_optin"], out["smem_src"] = v, src
    # (b) what fla itself compares against
    triton = dep("triton")
    if triton is not None:
        try:
            props = triton.runtime.driver.active.utils.get_device_properties(idx)
            out["smem_triton"] = int(props["max_shared_mem"])
        except BaseException as e:                      # noqa: BLE001
            out["note"].append(f"triton max_shared_mem failed: {type(e).__name__}")
    try:
        from fla.utils import get_all_max_shared_mem
        out["smem_fla"] = int(get_all_max_shared_mem()[idx])
    except BaseException as e:                          # noqa: BLE001
        out["note"].append(f"fla get_all_max_shared_mem failed: {type(e).__name__}")
    return out


def smem_verdict(smem):
    """VALID / INVALID / UNKNOWN, with the number that decided it.

    The three thresholds are fla/utils/_device.py:181-192: AMPERE 166912 ->
    num_stages [2,3,4] vs [2,1]; ADA 101376 -> BV [32,64] vs [32]; DEFAULT
    102400 -> BK_LIST/BV_LIST.  Note ADA is 1,024 B below DEFAULT, so a consumer
    card at 101,376 keeps BV=64 and loses the 3- and 4-stage pipeline."""
    if smem is None:
        return "UNKNOWN", "sharedMemoryPerBlockOptin not readable here"
    if smem >= SMEM_AMPERE:
        return "VALID", f"{smem:,} >= {SMEM_AMPERE:,} (AMPERE): full config space"
    parts = [f"{smem:,} < {SMEM_AMPERE:,} (AMPERE)"]
    parts.append("num_stages={2,1}" if smem < SMEM_AMPERE else "num_stages full")
    parts.append(f"BV={{32,64}}" if smem >= SMEM_ADA else "BV={32}")
    parts.append("BK/BV_LIST full" if smem >= SMEM_DEFAULT else "BK/BV_LIST={16,32}")
    return "INVALID", ", ".join(parts)


def device_report(idx=0):
    d = read_device(idx)
    verdict, why = smem_verdict(d["smem_optin"])
    d["smem_verdict"] = verdict
    d["smem_why"] = why
    if d["cc"]:
        d["arch"] = arch_name(*d["cc"])
        d["generation"] = {
            (6,): "Pascal", (7,): "Volta/Turing", (8, 0): "Ampere (A100 class)",
            (8, 6): "Ampere consumer (GA10x)", (8, 7): "Ampere (Orin)",
            (8, 9): "Ada", (9, 0): "Hopper", (10, 0): "Blackwell datacenter",
            (12, 0): "Blackwell consumer",
        }.get(d["cc"], f"compute capability {d['cc'][0]}.{d['cc'][1]}")
    else:
        d["arch"] = d["generation"] = "?"
    # the three numbers must agree, or the verdict rests on a contested read
    nums = {k: v for k, v in (("torch", d["smem_optin"]), ("triton", d["smem_triton"]),
                              ("fla", d["smem_fla"])) if v is not None}
    d["smem_reads"] = nums
    d["smem_agree"] = len(set(nums.values())) <= 1 if nums else None
    return d


# fla computes every architecture predicate from DEVICE 0, and the autotune
# lists are module-level constants evaluated at import with tensor_idx=0
# (fla/utils/_device.py:130-147 uses get_device_capability(0)/get_device_name(0);
#  check_shared_mem defaults to tensor_idx=0 at :196-200).  On a box with more
# than one device, the config space is device 0's, whatever device the kernel
# runs on.  Cheap to check, silent if not.
def fleet_report():
    torch = dep("torch")
    out = dict(count=0, devices=[], heterogeneous=False, notes=[])
    if torch is None or not getattr(torch, "cuda", None) or not torch.cuda.is_available():
        out["notes"].append("no CUDA device visible")
        return out
    try:
        n = torch.cuda.device_count()
    except BaseException as e:                          # noqa: BLE001
        out["notes"].append(f"device_count failed: {type(e).__name__}")
        return out
    out["count"] = n
    if n == 0:
        out["notes"].append("torch.cuda.device_count() == 0")
        return out
    for i in range(n):
        out["devices"].append(device_report(i))
    smems = {d["smem_optin"] for d in out["devices"] if d["smem_optin"] is not None}
    ccs = {d["cc"] for d in out["devices"] if d["cc"] is not None}
    out["heterogeneous"] = len(smems) > 1 or len(ccs) > 1
    if out["heterogeneous"]:
        out["notes"].append("fla gates the config space on DEVICE 0 only; this "
                            "box is not uniform, so a kernel on device != 0 is "
                            "autotuned against device 0's thresholds")
    return out


# ---------------------------------------------------------------------------
# 5. the live registry, read from the installed package when there is one.
# ---------------------------------------------------------------------------
def live_registry():
    """(entries, error). entries is the registry exactly as fla would walk it."""
    if dep("fla") is None:
        return None, f"fla not importable ({dep_err('fla')})"
    try:
        from fla.ops.backends import BackendRegistry
        BackendRegistry.ensure_initialized("kda")
        reg = BackendRegistry._registries.get("kda")
        if reg is None:
            return None, "BackendRegistry has no 'kda' registry after init"
        out = []
        for be in reg._get_sorted_backends():
            try:
                avail = be.is_available()
            except BaseException as e:                  # noqa: BLE001
                avail = f"raised {type(e).__name__}: {str(e)[:40]}"
            try:
                enab = be.is_enabled()
            except BaseException as e:                  # noqa: BLE001
                enab = f"raised {type(e).__name__}: {str(e)[:40]}"
            out.append(dict(
                backend_type=getattr(be, "backend_type", "?"),
                cls=type(be).__name__,
                priority=getattr(be, "priority", None),
                package_name=getattr(be, "package_name", None),
                env_var=getattr(be, "env_var", None),
                default_enable=getattr(be, "default_enable", None),
                implements_chunk_kda=hasattr(be, "chunk_kda"),
                has_chunk_kda_verifier=hasattr(be, "chunk_kda_verifier"),
                is_available=avail, is_enabled=enab,
            ))
        return out, ""
    except BaseException as e:                          # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:80]}"


def registry_drift(live):
    """Differences between the installed registry and MIRRORED_REGISTRY."""
    if live is None:
        return ["registry not readable: cannot confirm the mirror"]
    drift = []
    live_by_type = {e["backend_type"]: e for e in live}
    for m in MIRRORED_REGISTRY:
        e = live_by_type.get(m["backend_type"])
        if e is None:
            drift.append(f"mirrored backend {m['backend_type']} is ABSENT from the "
                         f"installed registry")
            continue
        for field_name in ("priority", "package_name", "env_var", "default_enable"):
            if e.get(field_name) != m[field_name]:
                drift.append(f"{m['backend_type']}.{field_name}: installed "
                             f"{e.get(field_name)!r} != mirrored {m[field_name]!r}")
    for e in live:
        if e["backend_type"] not in {m["backend_type"] for m in MIRRORED_REGISTRY}:
            drift.append(f"NEW backend {e['backend_type']} (priority {e['priority']}) "
                         f"is registered and not in the mirror -- re-read its "
                         f"verifier before trusting any run")
    return drift


# ---------------------------------------------------------------------------
# 6. which backend would this call actually select?  This re-executes the
#    dispatch loop of fla/ops/backends/__init__.py:182-211 against the REAL
#    verifier methods, with a simulated call.  Tensors are 1x1 in B and T: the
#    verifiers only read dtype and shape[-1]/shape[2], which are preserved.
# ---------------------------------------------------------------------------
def simulate_dispatch(call, func="chunk_kda", idx=0):
    """(selected, decisions, error). selected is None when nothing is installed."""
    torch = dep("torch")
    if torch is None:
        return None, [], f"torch not importable ({dep_err('torch')})"
    if not torch.cuda.is_available():
        return None, [], "no CUDA device visible"
    live, err = live_registry()
    if live is None:
        return None, [], err
    H, HV, K, V = call["H"], call["HV"], call["K"], call["V"]
    dt = getattr(torch, call["dtype"].split(".")[-1])
    try:
        q = torch.zeros((1, 1, H, K), dtype=dt, device=f"cuda:{idx}")
        k = torch.zeros((1, 1, H, K), dtype=dt, device=f"cuda:{idx}")
        v = torch.zeros((1, 1, HV, V), dtype=dt, device=f"cuda:{idx}")
        g = torch.zeros((1, 1, HV, K), dtype=dt, device=f"cuda:{idx}")
        beta = torch.zeros((1, 1, HV), dtype=dt, device=f"cuda:{idx}")
    except BaseException as e:                          # noqa: BLE001
        return None, [], f"could not build probe tensors: {type(e).__name__}: {str(e)[:60]}"
    kwargs = dict(
        q=q, k=k, v=v, g=g, beta=beta, scale=None, initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=call["use_qk_l2norm_in_kernel"],
        use_gate_in_kernel=call["use_gate_in_kernel"],
        use_beta_sigmoid_in_kernel=call["use_beta_sigmoid_in_kernel"],
        state_v_first=call["state_v_first"], cu_seqlens=None, cu_seqlens_cpu=None,
        safe_gate=call["safe_gate"], lower_bound=None, disable_recompute=False,
        return_intermediate_states=False, cp_context=None,
    )
    from fla.ops.backends import BackendRegistry
    BackendRegistry.ensure_initialized("kda")
    reg = BackendRegistry._registries.get("kda")
    decisions = []
    selected = None
    deciding_predicate = ""
    grad_ctx = torch.enable_grad() if call["grad_enabled"] else torch.inference_mode()
    with grad_ctx:
        for be in reg._get_sorted_backends():
            bt = getattr(be, "backend_type", "?")
            if not (be.is_available() and be.is_enabled()):
                decisions.append(dict(backend=bt, outcome="skipped",
                                      predicate="is_available() and is_enabled()",
                                      reason=f"available={be.is_available()} "
                                             f"enabled={be.is_enabled()} "
                                             f"env={be.env_var}"))
                continue
            ok, reason = be.verify(func, **kwargs)
            if not ok:
                decisions.append(dict(backend=bt, outcome="rejected",
                                      predicate=f"{func}_verifier", reason=reason))
                continue
            if getattr(be, func, None) is None:
                decisions.append(dict(backend=bt, outcome="no-impl",
                                      predicate=f"hasattr(backend, '{func}')",
                                      reason=f"verifier passed but {func} is not "
                                             f"implemented by this backend"))
                continue
            selected = bt
            deciding_predicate = f"first available+enabled+verified backend with {func}"
            decisions.append(dict(backend=bt, outcome="SELECTED",
                                  predicate=deciding_predicate, reason=reason))
            break
    if selected is None:
        decisions.append(dict(backend="(mainline)", outcome="SELECTED",
                              predicate="no backend survived -> undecorated function "
                                        "(fla/ops/backends/__init__.py:214)",
                              reason="this is the mainline Triton path"))
        selected = "triton_ascend(mainline-fallback)"
    return selected, decisions, ""


# ---------------------------------------------------------------------------
# 7. fla module-level switches, read live when fla is importable.
# ---------------------------------------------------------------------------
FLA_FLAGS = ("IS_NVIDIA", "IS_AMD", "IS_NPU", "IS_INTEL",
             "IS_TF32_SUPPORTED", "IS_TMA_SUPPORTED", "IS_GATHER_SUPPORTED",
             "IS_NVIDIA_HOPPER", "IS_NVIDIA_SM100", "IS_NVIDIA_BLACKWELL",
             "device_platform", "device")


def fla_flags():
    out, err = {}, ""
    if dep("fla") is None:
        return out, f"fla not importable ({dep_err('fla')})"
    try:
        import fla.utils as fu
        for name in FLA_FLAGS:
            out[name] = getattr(fu, name, "<not exported>")
        try:
            from fla.ops.kda.chunk_intra import SOLVE_TRIL_DOT_PRECISION
            out["SOLVE_TRIL_DOT_PRECISION"] = str(SOLVE_TRIL_DOT_PRECISION)
        except BaseException as e:                      # noqa: BLE001
            out["SOLVE_TRIL_DOT_PRECISION"] = f"unavailable ({type(e).__name__})"
        try:
            import fla.ops.kda.chunk_bwd as cb
            out["BK_LIST"] = list(cb.BK_LIST)
            out["BV_LIST"] = list(cb.BV_LIST)
        except BaseException as e:                      # noqa: BLE001
            out["BK_LIST"] = out["BV_LIST"] = f"unavailable ({type(e).__name__})"
    except BaseException as e:                          # noqa: BLE001
        err = f"{type(e).__name__}: {str(e)[:70]}"
    return out, err


def gather_report():
    """IS_GATHER_SUPPORTED decides a constexpr that branches the diagonal-block
    kernel.  fla/ops/kda/chunk_intra.py:68-78 defines `gather` as tl.gather when
    supported and as a stub returning None when not; the kernel then takes a
    different load (op.py is fla/ops/utils/op.py:68-78) -- tl.gather from
    registers, else a global tl.load of the same g row
    (chunk_intra.py:508-514, 606-612, 741-749).  The two branches read the same
    element, so this is an implementation/codegen switch, NOT a change of the
    mathematical algorithm -- the brief's stronger claim is not supported by the
    source read here.  It still has to be pinned: it changes the generated code
    and the masks, and a Triton upgrade can flip it."""
    triton = dep("triton")
    if triton is None:
        return dict(active=None, source=None,
                    detail=f"triton not importable ({dep_err('triton')})")
    try:
        active = hasattr(triton.language, "gather")
        ver = getattr(triton, "__version__", "?")
    except BaseException as e:                          # noqa: BLE001
        return dict(active=None, source=None, detail=f"{type(e).__name__}")
    live, _ = fla_flags()
    fla_val = live.get("IS_GATHER_SUPPORTED", "<fla not importable>")
    detail = f"tl.gather present={active} (triton {ver}); fla says {fla_val}"
    if active and fla_val is False:
        detail += "  <-- fla disagrees with the installed triton"
    return dict(active=active, triton_version=ver, fla_value=fla_val,
                source="hasattr(triton.language, 'gather') -- fla/utils/_device.py:138",
                detail=detail)


def tf32_report(dev):
    """The sm_75 story, stated only as far as the sources support it."""
    out = dict(TRITON_F32_DEFAULT=os.environ.get("TRITON_F32_DEFAULT", "<unset>"),
               set_by_fla=None, resolved_knob="<unavailable>", is_tf32_supported=None,
               solve_tril="<unavailable>", cc=dev.get("cc"), arch=dev.get("arch"))
    # was TRITON_F32_DEFAULT set by fla, or by the operator?
    out["set_by_fla"] = (out["TRITON_F32_DEFAULT"] == "ieee"
                         and dev.get("cc") is not None
                         and dev["cc"][0] < 8
                         and dev.get("smem_src") is not None)
    triton = dep("triton")
    if triton is not None:
        try:
            import triton.knobs as tk
            out["resolved_knob"] = repr(tk.language.fp32_default)
        except BaseException as e:                      # noqa: BLE001
            out["resolved_knob"] = f"<{type(e).__name__}>"
    live, _ = fla_flags()
    out["is_tf32_supported"] = live.get("IS_TF32_SUPPORTED", "<fla not importable>")
    out["solve_tril"] = live.get("SOLVE_TRIL_DOT_PRECISION", "<unavailable>")
    if dev.get("cc") and dev["cc"][0] < 8:
        out["verdict"] = ("IEEE-fp32 dots: fla sets TRITON_F32_DEFAULT='ieee' "
                          "process-globally (fla/utils/_device.py:149-152) and "
                          "tl.dot without an explicit input_precision resolves to "
                          "it; fla also fixes SOLVE_TRIL_DOT_PRECISION='ieee' on "
                          "this arch. No bf16 MMA exists below sm_80 either.")
    elif dev.get("cc"):
        out["verdict"] = "tf32-capable arch; see notes/08 on what tl.dot resolves to"
    else:
        out["verdict"] = "no device read: cannot state anything about dot precision"
    return out


# ---------------------------------------------------------------------------
# 8. assemble / print / assert
# ---------------------------------------------------------------------------
@dataclass
class Pin:
    env: dict = field(default_factory=dict)
    deps: dict = field(default_factory=dict)
    device: dict = field(default_factory=dict)
    fleet: dict = field(default_factory=dict)
    registry_live: list = field(default_factory=list)
    registry_error: str = ""
    drift: list = field(default_factory=list)
    study_call: dict = field(default_factory=dict)
    layer_call: dict = field(default_factory=dict)
    fla_flags: dict = field(default_factory=dict)
    fla_flags_error: str = ""
    gather: dict = field(default_factory=dict)
    tf32: dict = field(default_factory=dict)
    problems: list = field(default_factory=list)
    unknowns: list = field(default_factory=list)

    @property
    def ok(self):
        return not self.problems and not self.unknowns

    def verdict(self):
        if self.problems:
            return "INVALID"
        if self.unknowns:
            return "UNKNOWN"
        return "VALID"


def build_pin(idx=0, simulate=True):
    probe_deps()
    p = Pin()
    p.env = {name: os.environ.get(name) for name, _, _, _, _ in FLA_ENV}
    p.deps = {n: (dep(n) is not None, dep_err(n)) for n in DEPS}
    p.device = device_report(idx)
    p.fleet = fleet_report()
    p.registry_live, p.registry_error = live_registry()
    p.drift = registry_drift(p.registry_live)
    p.study_call = dict(STUDY_CALL, **{f"sim_{k}": v for k, v in (
        ("selected", None), ("error", ""), ("decisions", []))})
    p.layer_call = dict(LAYER_CALL)
    if simulate:
        sel, dec, err = simulate_dispatch(STUDY_CALL, idx=idx)
        p.study_call.update(sim_selected=sel, sim_error=err, sim_decisions=dec)
        sel2, dec2, err2 = simulate_dispatch(LAYER_CALL, idx=idx)
        p.layer_call.update(sim_selected=sel2, sim_error=err2, sim_decisions=dec2)
    p.fla_flags, p.fla_flags_error = fla_flags()
    p.gather = gather_report()
    p.tf32 = tf32_report(p.device)

    # ---- gates ------------------------------------------------------------
    v = p.device.get("smem_verdict", "UNKNOWN")
    if v == "INVALID":
        p.problems.append(f"shared-memory gate INVALID: {p.device['smem_why']}")
    elif v == "UNKNOWN":
        p.unknowns.append(f"shared-memory gate UNKNOWN: {p.device['smem_why']}")
    if p.device.get("smem_agree") is False:
        p.problems.append(f"the three shared-memory reads disagree: {p.device['smem_reads']}")
    for d in p.fleet.get("devices", [])[1:]:
        if d.get("smem_verdict") == "INVALID":
            p.problems.append(f"device {d['index']} ({d.get('name')}) is INVALID: {d['smem_why']}")
    if p.fleet.get("heterogeneous"):
        p.problems.append("fla sizes its config space from device 0 on a "
                          "non-uniform box; pin the device explicitly")
    if idx != 0:
        p.unknowns.append(f"reporting device {idx}, but fla sizes every config "
                          f"list from DEVICE 0 (check_shared_mem defaults to "
                          f"tensor_idx=0); compare against device 0's read too")
    if p.registry_live is None:
        p.unknowns.append(f"live registry read failed ({p.registry_error}); the "
                          f"registry table is the mirror, not this installation")
    elif p.drift:
        p.problems.append("registry drift: " + "; ".join(p.drift))
    sel = p.study_call.get("sim_selected")
    if p.study_call.get("sim_error"):
        p.unknowns.append(f"backend selection UNKNOWN: {p.study_call['sim_error']}")
    elif sel not in (None, "triton_ascend(mainline-fallback)"):
        p.problems.append(f"the study's own call would be served by {sel!r}, not the "
                          f"Triton path. Set FLA_DISABLE_BACKEND_DISPATCH=1 (or "
                          f"FLA_FLASH_KDA=0) and re-run.")
    sel2 = p.layer_call.get("sim_selected")
    if p.layer_call.get("sim_error"):
        pass          # already reported through the study call
    elif sel2 not in (None, "triton_ascend(mainline-fallback)"):
        p.problems.append(f"the fla LAYER-shaped call would be served by {sel2!r} -- "
                          f"the CUTLASS backend is armed for this configuration")
    if p.gather.get("active") is None:
        p.unknowns.append(f"IS_GATHER_SUPPORTED UNKNOWN: {p.gather.get('detail')}")
    if p.env.get("FLA_CACHE_MODE") not in (None, "disabled"):
        p.problems.append(f"FLA_CACHE_MODE={p.env['FLA_CACHE_MODE']!r}: a JSON file may "
                          f"override Triton's autotune result for a kernel")
    if p.env.get("FLA_USE_FAST_OPS") == "1":
        p.unknowns.append("FLA_USE_FAST_OPS=1: exp/log/log2/tanh use the fast_* "
                          "family (exp2 is unchanged) -- a different operator")
    if p.env.get("FLA_USE_TMA") == "1":
        p.problems.append("FLA_USE_TMA=1: the load path AND the tril dot-precision "
                          "autotune list change")
    if p.env.get("FLA_DISABLE_BACKEND_DISPATCH") == "1":
        p.unknowns.append("FLA_DISABLE_BACKEND_DISPATCH=1: dispatch is bypassed, so "
                          "the registry report below is not what will run")
    if p.env.get("TRITON_OVERRIDE_ARCH"):
        p.problems.append(f"TRITON_OVERRIDE_ARCH={p.env['TRITON_OVERRIDE_ARCH']!r}: "
                          f"triton will not compile for this device's arch")
    if p.env.get("TRITON_INTERPRET") == "1":
        p.problems.append("TRITON_INTERPRET=1: the interpreter runs, not the compiler")
    if dep("torch") is None:
        p.unknowns.append(f"torch not importable ({dep_err('torch')})")
    if dep("fla") is None:
        p.unknowns.append(f"fla not importable ({dep_err('fla')}); the registry "
                          f"table below is the mirror, not a live read")
    return p


def print_report(p, idx=0):
    hr("BASELINE PIN -- what will actually run")
    print(f"  fla-core mirror      : 0.5.2 (flash-linear-attention 0.5.2 -> "
          f"fla-core==0.5.2)")
    print(f"  device               : index {idx}")
    print(f"  VERDICT              : {p.verdict()}")
    if p.problems:
        for m in p.problems:
            print(f"    FAIL   {m}")
    if p.unknowns:
        for m in p.unknowns:
            print(f"    UNKNOWN {m}")

    hr("1. installed dependencies (absence is reported, never assumed)")
    for n in DEPS:
        ok, err = p.deps[n]
        print(f"  {n:<22}{'yes' if ok else 'NO '}  {'' if ok else err}")
    for name in ("flash_kda", "tilelang"):
        present, why = spec_present(name)
        print(f"  find_spec({name!r})      {present}  {why}")
    # nvcc decides whether the TileLang backend counts as available at all
    # (fla/utils/_compat.py:39-72), so it belongs in the pre-flight read.
    print(f"  nvcc on PATH            {shutil.which('nvcc') or 'NO'}")

    hr("2. backend registry, in the order the dispatcher walks it")
    print("  fla/ops/backends/__init__.py:182-211 -- first survivor wins;")
    print("  if none survive, the undecorated mainline runs (:214).")
    if p.registry_live is None:
        print(f"  LIVE READ UNAVAILABLE: {p.registry_error}")
        print("  --- mirror ({}) ---".format(MIRROR_NOTE))
        for m in MIRRORED_REGISTRY:
            print(f"  prio {m['priority']}  {m['backend_type']:<14} "
                  f"env={m['env_var'] or '-':<22} impl chunk_kda="
                  f"{'yes' if 'chunk_kda' in m['implements'] else 'no '}")
            print(f"             available when: {m['available_when']}")
            print(f"             source          : {m['src']}")
            if m.get("note"):
                print(f"             note            : {m['note']}")
    else:
        for e in p.registry_live:
            print(f"  prio {e['priority']}  {e['backend_type']:<14} "
                  f"class={e['cls']}")
            print(f"             package_name={e['package_name']}  env_var={e['env_var']}  "
                  f"default_enable={e['default_enable']}")
            print(f"             is_available={e['is_available']}  is_enabled={e['is_enabled']}  "
                  f"implements chunk_kda={e['implements_chunk_kda']}  "
                  f"has chunk_kda_verifier={e['has_chunk_kda_verifier']}")
    if p.registry_live is None:
        print("  DRIFT vs mirror: NOT CHECKED (no live registry to compare against)")
    elif p.drift:
        print("  DRIFT vs mirror:")
        for d in p.drift:
            print(f"    - {d}")
    else:
        print("  mirror matches the installed registry (no drift)")

    hr("3. which backend would each call actually select?")
    for label, call in (("study harness", p.study_call), ("fla layer shape", p.layer_call)):
        print(f"  {label}  ({call['src']})")
        print(f"    dtype={call['dtype']} K={call['K']} V={call['V']} H={call['H']} "
              f"HV={call['HV']} grad_enabled={call['grad_enabled']}")
        print(f"    flags: gate={call['use_gate_in_kernel']} l2norm={call['use_qk_l2norm_in_kernel']} "
              f"beta_sigmoid={call['use_beta_sigmoid_in_kernel']} "
              f"state_v_first={call['state_v_first']} safe_gate={call['safe_gate']}")
        if call.get("sim_error"):
            print(f"    UNKNOWN: {call['sim_error']}")
        else:
            print(f"    -> {call.get('sim_selected')}")
            for d in call.get("sim_decisions", []):
                print(f"       {d['backend']:<14} {d['outcome']:<9} by {d['predicate']}")
                if d.get("reason"):
                    print(f"                     {d['reason']}")
    print("  FlashKDABackend.chunk_kda_verifier gates, in test order")
    print("  (fla/ops/kda/backends/flash_kda.py:64-88):")
    for i, (pred, msg) in enumerate(FLASHKDA_GATES, 1):
        print(f"    {i:>2}. {pred:<34} -> {msg}")

    hr("4. shared memory -- the gate that decides whether this card is usable")
    d = p.device
    print(f"  device            : {d.get('name')}")
    print(f"  compute capability: {d.get('cc')}  ({d.get('arch')}, {d.get('generation')})")
    print(f"  sharedMemoryPerBlockOptin : {d.get('smem_optin')}")
    print(f"    read via        : {d.get('smem_src')}")
    print(f"  triton max_shared_mem     : {d.get('smem_triton')}   "
          f"(what fla compares against)")
    print(f"  fla get_all_max_shared_mem: {d.get('smem_fla')}")
    print(f"  all three agree    : {d.get('smem_agree')}")
    print(f"  thresholds         : DEFAULT={SMEM_DEFAULT:,}  ADA={SMEM_ADA:,}  "
          f"AMPERE={SMEM_AMPERE:,}  HOPPER={SMEM_HOPPER:,}")
    print(f"  VERDICT            : {d.get('smem_verdict')} -- {d.get('smem_why')}")
    sm = d.get("smem_optin")
    if sm is not None:
        print("  config space this device would get:")
        for src, what, fn, note in NARROWING_SITES:
            try:
                val = fn(sm)
            except BaseException:                       # noqa: BLE001
                val = "?"
            print(f"    {src:<42}{what:<22}{val}   ({note})")
        print(f"    fwd state kernel num_warps        "
              f"{num_warps_fwd_h(d.get('arch'))}")
    for n in d.get("note", []):
        print(f"  note: {n}")

    hr("5. every fla / triton environment switch, as this process sees it")
    for name, default, algo, what, src in FLA_ENV:
        val = p.env.get(name)
        shown = "<unset>" if val is None else repr(val)
        flag = "SET " if val is not None else "    "
        print(f"  {flag}{name:<32}{shown:<22} algorithmic={algo}")
        print(f"        default when unset: {default}")
        print(f"        changes: {what}")
        print(f"        source : {src}")

    hr("6. fla module-level switches (live read)")
    if p.fla_flags_error:
        print(f"  UNAVAILABLE: {p.fla_flags_error}")
    for k, v in p.fla_flags.items():
        print(f"  {k:<26}{v}")
    if p.device.get("cc"):
        print("  config lists as imported (fla computes these at import time from "
              "device 0):")
        print(f"    BK_LIST={p.fla_flags.get('BK_LIST')}  "
              f"BV_LIST={p.fla_flags.get('BV_LIST')}")

    hr("7. IS_GATHER_SUPPORTED -- the triton-version switch")
    print(f"  active              : {p.gather.get('active')}")
    print(f"  triton version      : {p.gather.get('triton_version')}")
    print(f"  fla's own value     : {p.gather.get('fla_value')}")
    print(f"  source              : {p.gather.get('source')}")
    print(f"  detail              : {p.gather.get('detail')}")
    print("  what it switches    : fla/ops/kda/chunk_intra.py:508-514, 606-612, "
          "741-749 read the")
    print("                        same g row either with tl.gather from registers or "
          "with a global")
    print("                        tl.load; fla/ops/utils/op.py:68-78 makes `gather` "
          "return None when")
    print("                        absent. Implementation/codegen switch -- the "
          "value is the same.")

    hr("8. float32 dot precision (the sm_75 story, only as far as read)")
    for k in ("cc", "arch", "TRITON_F32_DEFAULT", "set_by_fla", "resolved_knob",
              "is_tf32_supported", "solve_tril", "verdict"):
        print(f"  {k:<22}{p.tf32.get(k)}")

    hr("9. fleet")
    print(f"  CUDA devices visible: {p.fleet.get('count')}")
    for dev in p.fleet.get("devices", []):
        print(f"    {dev['index']}: {dev.get('name')}  {dev.get('arch')}  "
              f"smem={dev.get('smem_optin')}  {dev.get('smem_verdict')}")
    for n in p.fleet.get("notes", []):
        print(f"  note: {n}")


class BaselineNotPinned(RuntimeError):
    """Raised by assert_pinned() when the baseline is not the Triton path, or
    the device cannot support it, or too much is unknown to decide."""


# ---------------------------------------------------------------------------
# 10. self-test of THIS file's logic, on a stub registry.
#
#     What can be tested without a GPU: that the selection loop picks the first
#     available+enabled+verified backend in priority order, that a backend
#     without an implementation for the function is skipped, that FLA_FLASH_KDA=0
#     removes the CUTLASS backend, that the shared-memory verdict is right at
#     101376 / 102400 / 166912, and that a registry change is reported as drift.
#     What CANNOT be tested here: that real fla behaves as the mirror says.
#     So the stub is labelled as a stub everywhere it appears.
# ---------------------------------------------------------------------------
def _stub_modules(smem=166912, cc=(8, 0), with_flash_kda=True, extra_backend=False):
    import contextlib
    import types

    class _T:
        def __init__(self, shape, dtype):
            self.shape, self.dtype = tuple(shape), dtype

    class _Props:
        def __init__(self, s):
            self.shared_memory_per_block_optin = s

    torch = types.ModuleType("torch")
    torch.bfloat16, torch.float16, torch.float32 = "bfloat16", "float16", "float32"
    torch.is_grad_enabled = lambda: False
    torch.inference_mode = contextlib.nullcontext
    torch.enable_grad = contextlib.nullcontext
    torch.zeros = lambda shape, dtype=None, device=None: _T(shape, dtype)
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 1,
        get_device_capability=lambda i=0: cc,
        get_device_name=lambda i=0: f"SELFTEST sm_{cc[0]}{cc[1]}",
        get_device_properties=lambda i=0: _Props(smem))

    class _Backend:
        backend_type, package_name, env_var, default_enable = "x", None, None, True

        def is_available(self):
            return True

        def is_enabled(self):
            if self.env_var is None:
                return True
            return os.environ.get(self.env_var, "1") != "0"

        def verify(self, func_name, *a, **kw):
            v = getattr(self, f"{func_name}_verifier", None)
            return (True, None) if v is None else v(*a, **kw)

    class Ascend(_Backend):
        backend_type, priority, package_name = "triton_ascend", 0, None

        def is_available(self):
            return False                       # never on NVIDIA (fla:46-49)

    class FlashKDA(_Backend):
        backend_type, priority = "flash_kda", 3
        package_name, env_var, default_enable = "flash_kda", "FLA_FLASH_KDA", True

        def is_available(self):
            return with_flash_kda

        def chunk_kda_verifier(self, q=None, k=None, v=None, g=None, beta=None,
                               use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
                               use_beta_sigmoid_in_kernel=False, state_v_first=False,
                               cp_context=None, return_intermediate_states=False,
                               safe_gate=False, **kw):
            # the real gate order, fla/ops/kda/backends/flash_kda.py:64-88
            if q.dtype != "bfloat16":
                return False, "FlashKDA requires bfloat16"
            if q.shape[-1] != 128:
                return False, "FlashKDA requires K=128"
            if v.shape[-1] != 128:
                return False, "FlashKDA requires V=128"
            if v.shape[2] != q.shape[2]:
                return False, "FlashKDA does not support GVA"
            for flag, msg in ((use_gate_in_kernel, "use_gate_in_kernel=True"),
                              (use_qk_l2norm_in_kernel, "use_qk_l2norm_in_kernel=True"),
                              (use_beta_sigmoid_in_kernel, "use_beta_sigmoid_in_kernel=True"),
                              (state_v_first, "state_v_first=True"),
                              (safe_gate, "safe_gate=True")):
                if not flag:
                    return False, f"FlashKDA requires {msg}"
            return True, None

        def chunk_kda(self, *a, **kw):
            return "CUTLASS-OUTPUT"

    class TileLang(_Backend):
        """priority 5 and default_enable True are inherited; no chunk_kda impl."""
        backend_type, priority = "tilelang", 5
        package_name, env_var = "tilelang", "FLA_TILELANG"

    class NewBackend(_Backend):
        """A backend the mirror has never heard of -- must show up as drift."""
        backend_type, priority = "vendor_kda", 1

        def chunk_kda(self, *a, **kw):
            return "NEW-OUTPUT"

    backends = [Ascend(), FlashKDA(), TileLang()]
    if extra_backend:
        backends.insert(1, NewBackend())

    class _Reg:
        _registries = {}

        def __init__(self, name, blist):
            self.name, self.blist = name, blist
            _Reg._registries[name] = self

        @classmethod
        def ensure_initialized(cls, op):
            return None

        def _get_sorted_backends(self):
            return sorted(self.blist, key=lambda b: (b.priority, self.blist.index(b)))

    reg = _Reg("kda", backends)
    fla = types.ModuleType("fla")
    ops = types.ModuleType("fla.ops")
    ops_backends = types.ModuleType("fla.ops.backends")
    ops_backends.BackendRegistry = _Reg
    utils = types.ModuleType("fla.utils")
    for k, val in (("IS_NVIDIA", True), ("IS_AMD", False), ("IS_NPU", False),
                   ("IS_INTEL", False), ("IS_TF32_SUPPORTED", cc[0] >= 8),
                   ("IS_TMA_SUPPORTED", False), ("IS_GATHER_SUPPORTED", True),
                   ("IS_NVIDIA_HOPPER", cc[0] == 9), ("IS_NVIDIA_SM100", cc[0] == 10),
                   ("IS_NVIDIA_BLACKWELL", cc[0] in (10, 12)),
                   ("device_platform", "cuda"), ("device", "cuda")):
        setattr(utils, k, val)
    utils.get_all_max_shared_mem = lambda: [smem]
    ops_kda = types.ModuleType("fla.ops.kda")
    ops_kda.__path__ = []
    ci = types.ModuleType("fla.ops.kda.chunk_intra")
    ci.SOLVE_TRIL_DOT_PRECISION = "tf32" if cc[0] >= 8 else "ieee"
    cb = types.ModuleType("fla.ops.kda.chunk_bwd")
    cb.BK_LIST = [32, 64] if smem >= SMEM_DEFAULT else [16, 32]
    cb.BV_LIST = [64, 128] if smem >= SMEM_AMPERE else [16, 32]
    fla.__path__ = []                    # make the stubs packages, so that
    ops.__path__ = []                    # `from fla.ops.backends import ...`
    ops_backends.__path__ = []           # resolves against sys.modules
    fla.utils, ops.backends = utils, ops_backends
    ops_backends.BackendRegistry = _Reg
    ops_backends.kda_registry = reg
    ops_kda.chunk_intra, ops_kda.chunk_bwd = ci, cb
    ops.kda, ops_kda.backends = ops_kda, ops_kda
    triton = types.ModuleType("triton")
    triton.__version__ = "0.0.0+selftest"
    triton.language = types.SimpleNamespace(gather=object())
    triton.runtime = types.SimpleNamespace(
        driver=types.SimpleNamespace(active=types.SimpleNamespace(
            utils=types.SimpleNamespace(
                get_device_properties=lambda i=0: {"max_shared_mem": smem}))))
    triton.knobs = types.SimpleNamespace(
        language=types.SimpleNamespace(fp32_default=os.environ.get("TRITON_F32_DEFAULT")))
    mods = {
        "torch": torch, "triton": triton, "fla": fla, "fla.ops": ops,
        "fla.utils": utils, "fla.ops.backends": ops_backends, "fla.ops.kda": ops_kda,
        "fla.ops.kda.chunk_intra": ci, "fla.ops.kda.chunk_bwd": cb,
        "flash_linear_attention": fla, "tilelang": types.ModuleType("tilelang"),
    }
    if with_flash_kda:
        mods["flash_kda"] = types.ModuleType("flash_kda")
    return mods


STUB_MODULES = ("torch", "triton", "fla", "flash_kda", "tilelang",
                "flash_linear_attention", "fla.ops", "fla.utils",
                "fla.ops.backends", "fla.ops.kda", "fla.ops.kda.chunk_intra",
                "fla.ops.kda.chunk_bwd")


def _clear_stubs():
    for name in STUB_MODULES:
        sys.modules.pop(name, None)


def selftest():
    """Exercise the logic in this file against a stub registry. Returns failures."""
    saved_sys = dict(sys.modules)
    saved_opt = dict(OPT)
    fails = []

    def scenario(label, expect, **kw):
        env = kw.pop("env", {})
        saved_env = {k: os.environ.get(k) for k in env}
        os.environ.update({k: v for k, v in env.items() if v is not None})
        try:
            mods = _stub_modules(**kw)
            _clear_stubs()
            sys.modules.update(mods)
            OPT.clear()
            p = build_pin(simulate=True)
            got = expect(p)
            ok = got is True
            print(f"  {'pass' if ok else 'FAIL'}  {label}"
                  + ("" if ok else f"   (expected True, got {got!r})"))
            if not ok:
                fails.append(label)
                for m in p.problems + p.unknowns:
                    print(f"          {m}")
        except BaseException as e:                      # noqa: BLE001
            import traceback
            print(f"  FAIL  {label}   raised {type(e).__name__}: {e}")
            traceback.print_exc()
            fails.append(label)
        finally:
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            _clear_stubs()
            sys.modules.update(saved_sys)
            OPT.clear()
            OPT.update(saved_opt)

    def armed_is_cutlass(p):
        return p.study_call.get("sim_selected") == "triton_ascend(mainline-fallback)" \
            and p.layer_call.get("sim_selected") == "flash_kda"

    print("SELFTEST: stub torch/fla/triton in sys.modules. This tests THIS file's")
    print("selection and gate logic only -- it is not a run against real fla, and a")
    print("pass here says nothing about the installed package.")

    # 1. flash_kda importable, study call (flags off) vs layer-shaped call
    scenario("study call -> mainline, layer-shaped call -> flash_kda, A100 smem",
             armed_is_cutlass, smem=166912, cc=(8, 0), with_flash_kda=True)
    # 2. the deciding reason for the study call is the FIRST unmet flag
    scenario("study call rejected at use_gate_in_kernel (first unmet flag)",
             lambda p: any(d["backend"] == "flash_kda" and d["outcome"] == "rejected"
                           and "use_gate_in_kernel" in str(d.get("reason"))
                           for d in p.study_call.get("sim_decisions", [])),
             smem=166912, cc=(8, 0), with_flash_kda=True)
    # 3. FLA_FLASH_KDA=0 disarms it
    scenario("FLA_FLASH_KDA=0 -> mainline even for the armed call",
             lambda p: p.layer_call.get("sim_selected") == "triton_ascend(mainline-fallback)"
             and p.verdict() == "VALID",
             smem=166912, cc=(8, 0), with_flash_kda=True, env={"FLA_FLASH_KDA": "0"})
    # 4. FLA_DISABLE_BACKEND_DISPATCH=1 is reported as "not what will run"
    scenario("FLA_DISABLE_BACKEND_DISPATCH=1 is flagged as unverified-by-dispatch",
             lambda p: any("FLA_DISABLE_BACKEND_DISPATCH" in u for u in p.unknowns),
             smem=166912, cc=(8, 0), with_flash_kda=True,
             env={"FLA_DISABLE_BACKEND_DISPATCH": "1"})
    # 5. flash_kda not installed at all
    scenario("flash_kda absent -> mainline for both calls",
             lambda p: p.layer_call.get("sim_selected") == "triton_ascend(mainline-fallback)"
             and p.verdict() == "VALID",
             smem=166912, cc=(8, 0), with_flash_kda=False)
    # 6. shared-memory verdicts at the three interesting numbers
    for smem, want in ((166912, "VALID"), (102400, "INVALID"), (101376, "INVALID"),
                       (65536, "INVALID"), (232448, "VALID")):
        scenario(f"smem {smem:,} -> {want}",
                 lambda p, w=want: p.device["smem_verdict"] == w,
                 smem=smem, cc=(8, 0), with_flash_kda=False)
    # 7. an Ada 4090 is INVALID even though it has bf16 MMA
    scenario("sm_89 / 101,376 B is INVALID (non-monotone in capability)",
             lambda p: p.device["smem_verdict"] == "INVALID" and p.verdict() == "INVALID",
             smem=101376, cc=(8, 9), with_flash_kda=False)
    # 8. an unknown backend in the registry is drift, not a silent pass
    scenario("an unregistered-to-the-mirror backend is reported as drift",
             lambda p: any("NEW backend" in d for d in p.drift) and p.verdict() == "INVALID",
             smem=166912, cc=(8, 0), with_flash_kda=False, extra_backend=True)
    # 9. env switches that must fail the pin
    scenario("FLA_USE_TMA=1 fails the pin", lambda p: p.verdict() == "INVALID",
             smem=166912, cc=(9, 0), with_flash_kda=False, env={"FLA_USE_TMA": "1"})
    scenario("FLA_CACHE_MODE=full fails the pin", lambda p: p.verdict() == "INVALID",
             smem=166912, cc=(8, 0), with_flash_kda=False, env={"FLA_CACHE_MODE": "full"})
    scenario("TRITON_OVERRIDE_ARCH fails the pin", lambda p: p.verdict() == "INVALID",
             smem=166912, cc=(8, 0), with_flash_kda=False,
             env={"TRITON_OVERRIDE_ARCH": "sm_90"})
    # 10. the raised exception, on a genuinely bad configuration
    def raises(p):
        try:
            assert_pinned()
        except BaselineNotPinned as e:
            return "shared-memory" in str(e)
        return False
    scenario("assert_pinned() raises on an INVALID device", raises,
             smem=101376, cc=(8, 9), with_flash_kda=False)
    print(f"  {'all scenarios passed' if not fails else f'{len(fails)} FAILED'}")
    return fails


def assert_pinned(idx=0, simulate=True):
    p = build_pin(idx=idx, simulate=simulate)
    if p.problems:
        raise BaselineNotPinned(
            "baseline is not pinned:\n" + "\n".join("  - " + m for m in p.problems))
    if p.unknowns:
        raise BaselineNotPinned(
            "baseline could not be decided (unknown is not valid):\n"
            + "\n".join("  - " + m for m in p.unknowns))
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    ap.add_argument("--device", type=int, default=0, help="CUDA device index")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 unless the verdict is VALID")
    ap.add_argument("--json", action="store_true", help="also print the pin as JSON")
    ap.add_argument("--no-simulate", action="store_true",
                    help="skip the verifier simulation (no live backend verdict)")
    ap.add_argument("--selftest", action="store_true",
                    help="run this file's logic against a stub registry, then exit")
    args = ap.parse_args(argv)
    if args.selftest:
        hr("SELFTEST")
        return 1 if selftest() else 0
    p = build_pin(idx=args.device, simulate=not args.no_simulate)
    # `--json` means the STREAM IS JSON, not "JSON is somewhere in the output".
    # It used to print the human report to stdout first and then the JSON, so
    # `baseline_pin.py --json | jq` failed on the report's first line while the
    # flag read as though it worked. A flag whose output cannot be parsed is a
    # promise the file did not keep, and a lane on the T4 had to work around it.
    # The human report goes to stderr so nothing is lost, and stdout carries the
    # JSON alone. `--json --strict` is now a usable one-liner in a run log.
    if args.json:
        # `Pin` is a dataclass of dicts, lists and strings, so it is serialisable
        # directly. It used to go through `json.dumps(p, default=str)`, and
        # `default` is only reached for objects json cannot encode -- so `p` was
        # stringified to its repr and the run log received one long
        # "<...Pin object at 0x...>" string instead of the pin. `asdict` is the
        # whole fix; the flag was a promise the file had never kept.
        json.dump(dataclasses.asdict(p) if dataclasses.is_dataclass(p)
                  else p, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        sys.stdout.flush()
        # `print_report` is all bare `print`, so redirect rather than thread a
        # stream parameter through ~200 lines of output code.
        with contextlib.redirect_stdout(sys.stderr):
            print_report(p, idx=args.device)
    else:
        print_report(p, idx=args.device)
    code = {"VALID": 0, "INVALID": 1, "UNKNOWN": 2}[p.verdict()]
    if args.strict and code != 0:
        return code
    return 0


# Import-time gate. Off by default because an import that raises is a hostile
# default; the study's harness sets BASELINE_PIN_ON_IMPORT=1 so a run cannot
# start on an unpinned baseline.
if os.environ.get("BASELINE_PIN_ON_IMPORT") == "1":
    assert_pinned()


if __name__ == "__main__":
    sys.exit(main())


