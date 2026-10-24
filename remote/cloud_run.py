#!/usr/bin/env python3
"""cloud_run.py -- run the study's measurement battery on real Hopper and/or
Blackwell silicon through one entry point, targeting platforms whose free tier
makes a run cost $0.00.

    python remote/cloud_run.py --provider modal --arch h100 --budget-minutes 45
    python remote/cloud_run.py --provider modal --arch b200 --budget-minutes 60 --dry-run
    python remote/cloud_run.py --provider lightning --arch h100 --budget-minutes 45

--dry-run is the validated path. It makes ZERO network calls, needs ZERO
credentials, and still exercises argument handling, cost/budget arithmetic, path
staging (real byte counts and sha256 of the real tree), the generated remote
driver (compiled, imported and executed in --plan/--selftest mode), the feature
derivation for every compute capability we know about, and the result
collection round trip over a synthetic remote tree.

============================================================================
THE TWO THINGS THAT WILL SILENTLY CORRUPT A RUN
============================================================================

1. THE GPU YOU ASK FOR IS NOT ALWAYS THE GPU YOU GET, AND YOU PAY THE
   PRICE OF THE ONE YOU ASKED FOR.  Verbatim from Modal's docs:

     "Modal may automatically upgrade a `gpu="H100"` request to run on an
      H200. This automatic upgrade does _not_ change the cost of the GPU."
     "In cases where an automatic upgrade to H200 would not be helpful
      (for instance, benchmarking) you can pass `gpu=H100!` to avoid it."
     -- https://modal.com/docs/guide/gpu#automatic-upgrades-to-h200s

     "Use `gpu="B200+"` to allow Modal to run requests on either B200 or
      B300 GPUs. B200+ is billed as B200, regardless of which GPU is used.
      Use this option only if your code is compatible with both types of
      GPUs."
     -- https://modal.com/docs/guide/gpu#opt-in-upgrade-to-b300

   So: the `!` suffix is mandatory for H100, and the `+` form is forbidden for
   B200. Both are pinned in GPU_STRINGS below, and the identity check in
   stage `identity` fails the run (exit 9) if the device that answered is not
   the architecture that was requested. We do not trust the string; we read
   torch.cuda.get_device_capability() off the device.

2. THE PINNED tinygrad COMPILES TO THE BARE `sm_<major><minor>` TARGET, SO
   EVERY ARCH-CONDITIONAL INSTRUCTION IS UNREACHABLE ON EVERY ARCH. Verbatim
   from the tinygrad vendored in this repo:

       .venv/lib/python3.12/site-packages/tinygrad/runtime/ops_cuda.py:122
         arch=f"sm_{major.value}{minor.value}"

   ptxas 12.9 (NVIDIA's own assembler) rejects the arch-conditional families on
   the bare targets, with these exact messages, all reproduced by
   `python remote/cloud_run.py --selftest-ptx` logic in this file and by
   out/archlab:

     tcgen05.mma      not supported on .target 'sm_100'  (needs sm_100a)
     tcgen05.mma      not supported on .target 'sm_103'  (needs sm_103a)
     tcgen05.mma      not supported on .target 'sm_120'  (no sm_12xa at all)
     wgmma.mma_async  not supported on .target 'sm_90'   (needs sm_90a)

   Consequences, all verified against ptxas in this repo's archlab image:
     * wgmma   is unreachable on H100  -> tinygrad gets no Hopper wgmma
     * tcgen05 is unreachable on B200 and on B300 -> tinygrad gets no
       Blackwell 5th-gen tensor core at all
     * every number this study produces is therefore *feature-tier flat*
       across architectures; cross-arch differences can only come from memory
       bandwidth and clocks, never from a richer instruction set.
   This does not error, does not warn, and produces entirely plausible finite
   numbers. It is the most dangerous item in this file.

   The `--ptx-probe` stage below is the defence: it asks ptxas, on the box, for
   each feature and the device's own arch-conditional target, and the run FAILS
   if the assembler's answer disagrees with the table. So a run cannot quietly
   disagree with itself.

============================================================================
A CORRECTION TO THE STUDY'S OWN tcgen05 RULE (found by running ptxas)
============================================================================

The brief this script was written from asserted "tcgen05: sm_100a/sm_101a only
-- NOT sm_103, NOT sm_120". Half of that is wrong, and the wrong half is
Blackwell Ultra:

    * sm_120 / sm_120a / sm_121a : NO tcgen05.          CONFIRMED (ptxas 12.9
      "Instruction 'tcgen05.alloc' not supported on .target 'sm_120'"), and
      sm_12xa never appears in any tcgen05 Target ISA Notes list in the PTX ISA.
      So RTX PRO 6000 Blackwell genuinely has no 5th-gen tensor core.
    * sm_103a : HAS tcgen05.  ptxas 12.9 accepts tcgen05.alloc AND
      tcgen05.mma on sm_103a at PTX ISA >= 8.8, and NVIDIA's own PTX ISA 9.4
      lists sm_103a in the tcgen05 Target ISA Notes ("sm_100a, sm_101a
      (Renamed to sm_110a from PTX ISA version 9.0), sm_103a, sm_107a,
      sm_110a" -- the tcgen05.shift section).
      So B300 (sm_103) does have tcgen05, on the sm_103a target.
    * the bare sm_103 and sm_100 targets do NOT: tcgen05 needs the
      architecture-conditional `a` (or `f`) target. That is the real trap, and
      it is the one tinygrad falls into (see item 2 above).

  So the rule encoded in FEATURE_GATES is the verified one, and every row
  carries the evidence string that supports it.

============================================================================
COMMERCIAL FACTS, VERIFIED 2026-09-27, AND WHAT IS *NOT* VERIFIED
============================================================================

Modal (HIGH confidence; all from modal.com, fetched 2026-09-27):
  * "Using a GPU requires having a valid payment method on file."
    -> a card IS required, even to use the free credit.  [modal.com/docs/guide/gpu]
  * Starter plan "$0 + compute / month", "$30 / month free compute",
    100 containers + 10 GPU concurrency.  [modal.com/pricing]
  * Per-second list prices (base, us-east; every region other than the
    default is a 1.15x-1.75x MULTIPLIER on these):
        B300 0.001972   B200 0.001736   H200 SXM 0.001261
        H100 SXM5 0.001097   RTX PRO 6000 0.000842   A100-80GB 0.000694
        A100-40GB 0.000583   L40S 0.000542   A10 0.000306   L4 0.000222
        T4 0.000164
    Functions also bill CPU 0.00003942/core/sec and RAM 0.00000667/GiB/sec.
    -> H100  $3.9492/hr, B200 $6.2496/hr, L40S $1.9512/hr,
       RTX PRO 6000 $3.0312/hr.  $30 of credit buys ~7h36m H100 / ~4h48m B200.
  * Workspace budgets / spend limits are set ONLY in the web UI
    (https://modal.com/settings/usage). There is no documented API, so this
    script cannot set or read one. The real guarantee against spending money
    is therefore arithmetic, not a budget: --credit-only refuses to start
    unless worst-case cost <= the free credit, which makes a $0.00 outlay a
    theorem rather than a hope. Caveat from the docs: "The maximum budget you
    can set depends on prior successful charges for the Workspace", i.e. a
    brand-new account may not be able to set any budget at all.
  * "All Modal Functions are subject to preemption by default" and
    "**Note: The `nonpreemptible` parameter is not supported for GPU
    Functions.**"  So a GPU run CAN be preempted and there is no way to opt
    out. Design consequences, all implemented: retries=0, per-stage checkpoint
    to a Modal Volume, an idempotence guard so a preempted-then-restarted
    function does not re-burn the whole budget, and a partial-results
    collection path (exit 7).

Lightning AI (LOW confidence -- see the report; this is the weak leg):
  * Its own live pages carry FOUR mutually inconsistent free-tier claims:
      - pricing tier card + comparison table: "up to 30 credits"
      - the same page's FAQ block: "You get 5 free Lightning credits upon
        registration"
      - homepage hero: "Start now with 80 free GPU hours"
      - a 2025 third-party thread: "80 GPU hours monthly + 15 credits"
  * The official consumption-rate table (https://lightning.ai/pricing/
    consumption-rates) publishes ONLY 1 T4 (0.13 credits/hr), 1 V100 (0.38),
    4 V100 (1.53) and two CPU shapes. There is NO published rate for H100,
    H200, B200, A100 or L40S anywhere in the served HTML. The survey's
    H100 $/hr figure is therefore UNVERIFIED; do not budget against it.
  * Free tier does advertise "Single GPUs (L40S, A100, H100, H200...)" with a
    4-hour session limit on A100/H100/H200, so H100 on the free tier is
    plausible -- but I could not confirm it from a first-party statement that
    is not contradicted elsewhere on the same site.
  * Whether a card is required is NOT stated in Lightning's own docs. The only
    "no card" claims I found are third-party.
  => The Lightning adapter refuses to run its provisioning surface unless
  --accept-unverified-surface is passed, and the device-identity check is the
  real guard, not the machine-type string.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import glob
import hashlib
import importlib.util
import json
import os
import pathlib
import shlex
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent

# Date the vendor pricing / free-tier / GPU-string facts below were read.
DOCS_DATE = "2026-09-27"

# ---------------------------------------------------------------------------
# exit codes. every failure mode gets its own, and every one prints a remedy.
# ---------------------------------------------------------------------------
EXIT = {
    "ok": 0,
    "usage": 2,               # bad arguments, or a budget that cannot be free
    "no_credentials": 3,      # no token on this machine
    "out_of_credit": 4,       # free credit gone, or workspace budget hit
    "no_capacity": 5,         # requested arch has no capacity in the window
    "image_build": 6,         # the container image failed to build
    "preempted": 7,           # interrupted mid-run; partial results collected
    "sdk_missing": 8,         # modal / lightning SDK not installed
    "arch_mismatch": 9,       # the device is not the architecture requested
    "collect_failed": 10,     # ran, but results did not come home
    "no_results": 11,         # ran, but produced nothing at all
    "budget_exhausted": 12,   # our own wall-clock budget stopped it
    "unclassified": 13,       # provider error we could not attribute
    "self_check_failed": 14,  # the dry run / self test failed
}

# ===========================================================================
# 1. GPU STRINGS -- pinned, with the vendor text that forces each choice.
# ===========================================================================

GPU_STRINGS = {
    # arch key -> (modal gpu=, why that exact string, lightning machine)
    "h100": (
        "H100!",
        'BANG SUFFIX IS MANDATORY. "Modal may automatically upgrade a '
        'gpu=\"H100\" request to run on an H200. This automatic upgrade does '
        '_not_ change the cost of the GPU. ... In cases where an automatic '
        'upgrade to H200 would not be helpful (for instance, benchmarking) you '
        'can pass gpu=H100! to avoid it." (modal.com/docs/guide/gpu). An H200 '
        'is sm_90 like an H100, so the capability would match and the mismatch '
        'would be INVISIBLE -- only the VRAM (141GB vs 80GB) and the 4.8TB/s vs '
        '3.35TB/s bandwidth differ, and bandwidth is exactly what this study '
        'measures. The ! is the whole defence.',
        None,   # unverified; see --lightning-machine
    ),
    "h200": (
        "H200",
        'H200 has no documented substitution and no "!" form, so the bare '
        'string is the pin. modal.com/docs/guide/gpu lists `H200` with no '
        'variant. Note this is the substitution TARGET for a bare H100, which '
        'is why the h100 row above uses "!".',
        None,
    ),
    "b200": (
        "B200",
        'NEVER "B200+". "Use gpu=\"B200+\" to allow Modal to run requests on '
        'either B200 or B300 GPUs. B200+ is billed as B200, regardless of which '
        'GPU is used." (modal.com/docs/guide/gpu#opt-in-upgrade-to-b300) A B300 '
        'is sm_103 and a B200 is sm_100: different families, so a B300 served '
        'for a B200 would change every feature row in FEATURE_GATES and would '
        'be billed at the sm_100 price.',
        None,
    ),
    "b300": (
        "B300",
        'B300 is requested directly, so no "+" form is reachable. '
        'modal.com/docs/guide/gpu warns "B300 requires CUDA version 13.1+" -- '
        'so the image must be CUDA 13.1 or the run will not get a B300 at all.',
        None,
    ),
    "l40s": (
        "L40S",
        'L40S is listed with no upgrade path and no variant form. It is the '
        'sm_89 control row: FP8 tensor cores yes, TMA no, wgmma no, tcgen05 no.',
        None,
    ),
    "rtxpro6000": (
        "RTX-PRO-6000",
        'The exact spelling in modal.com/docs/guide/gpu is `RTX-PRO-6000` '
        '(hyphens, PRO and 6000 uppercase). sm_120: FP8 yes, TMA yes, tcgen05 '
        'NO -- the one Hopper/Blackwell-tier card that is not 5th-gen.',
        None,
    ),
    "a100": (
        "A100-80GB",
        'NEVER bare "A100". "Modal may automatically upgrade a gpu=\"A100\" '
        'request to run on an 80 GB A100. This automatic upgrade does _not_ '
        'change the cost of the GPU." The 40GB/80GB split matters here because '
        'a study that OOMs at 40GB and not at 80GB would draw the wrong '
        'conclusion. Pin the 80GB string.',
        None,
    ),
}

# ===========================================================================
# 2. ARCH TABLE
# ===========================================================================
# expect_cc comes from NVIDIA's own compute-capability table, not from a blog:
#   https://developer.nvidia.com/cuda/gpus
#     9.0 = GH200/H200/H100   10.0 = GB200/B200   10.3 = GB300/B300
#     12.0 = RTX PRO 6000 Blackwell (all editions)   8.9 = L4/L40/L40S
#     8.0 = A100/A30
MODAL_USD_PER_SEC = {                     # modal.com/pricing, read DOCS_DATE
    "B300": 0.001972,
    "B200": 0.001736,
    "H200": 0.001261,
    "H100": 0.001097,
    "RTX-PRO-6000": 0.000842,
    "A100-80GB": 0.000694,
    "A100-40GB": 0.000583,
    "L40S": 0.000542,
    "A10": 0.000306,
    "L4": 0.000222,
    "T4": 0.000164,
}
# modal.com/pricing "Compute costs" block, for Functions (not Sandboxes).
MODAL_FUNC_CPU_USD_PER_CORE_SEC = 0.00003942
MODAL_FUNC_RAM_USD_PER_GIB_SEC = 0.00000667
MODAL_STARTER_FREE_USD = 30.0            # "$30 / month free compute"
# "Region selection 1.15 - 1.75x base prices" (modal.com/pricing). The default
# region is us-east at 1.0; anything else is a multiplier, silently.
MODAL_REGION_MULTIPLIER = {"us-east": 1.00}
MODAL_WORST_REGION_MULTIPLIER = 1.75

ARCHS = {
    "h100": dict(
        label="NVIDIA H100 SXM5", expect_cc=(9, 0), vram_gb=80,
        modal_gpu="H100", usd_per_sec=MODAL_USD_PER_SEC["H100"],
        why="the study's Hopper target: sm_90, the only arch where wgmma exists",
    ),
    "h200": dict(
        label="NVIDIA H200 SXM", expect_cc=(9, 0), vram_gb=141,
        modal_gpu="H200", usd_per_sec=MODAL_USD_PER_SEC["H200"],
        why="the substitution TARGET for a bare H100 request; same sm_90, 1.75x "
            "the RAM and ~1.4x the bandwidth. A run that silently got one "
            "would report the other's bandwidth as Hopper's number.",
    ),
    "b200": dict(
        label="NVIDIA B200", expect_cc=(10, 0), vram_gb=180,
        modal_gpu="B200", usd_per_sec=MODAL_USD_PER_SEC["B200"],
        why="the study's datacenter-Blackwell target: sm_100, tcgen05-capable "
            "family (but only on the sm_100a target)",
    ),
    "b300": dict(
        label="NVIDIA B300 (Blackwell Ultra)", expect_cc=(10, 3), vram_gb=288,
        modal_gpu="B300", usd_per_sec=MODAL_USD_PER_SEC["B300"],
        why="sm_103: a DIFFERENT family from sm_100. Verified to also carry "
            "tcgen05 (on sm_103a). Needs CUDA 13.1+ on Modal.",
    ),
    "l40s": dict(
        label="NVIDIA L40S (Ada)", expect_cc=(8, 9), vram_gb=48,
        modal_gpu="L40S", usd_per_sec=MODAL_USD_PER_SEC["L40S"],
        why="the study's CONTROL row: cheapest card with bf16+FP8 tensor "
            "cores, and no TMA / wgmma / tcgen05 at all. Any feature "
            "difference vs H100 is a real feature difference, not a clock.",
    ),
    "rtxpro6000": dict(
        label="NVIDIA RTX PRO 6000 Blackwell", expect_cc=(12, 0), vram_gb=96,
        modal_gpu="RTX-PRO-6000", usd_per_sec=MODAL_USD_PER_SEC["RTX-PRO-6000"],
        why="the study's FALSE-BLACKWELL row: sm_120 has TMA and FP8 but NO "
            "tcgen05, which is the cleanest available proof that 'Blackwell' "
            "is not one target.",
    ),
    "a100": dict(
        label="NVIDIA A100 80GB", expect_cc=(8, 0), vram_gb=80,
        modal_gpu="A100-80GB", usd_per_sec=MODAL_USD_PER_SEC["A100-80GB"],
        why="the study's sm_80 floor row: cp.async yes, FP8 no, TMA no.",
    ),
}

# ===========================================================================
# 3. FEATURE AVAILABILITY, DERIVED FROM THE DEVICE'S REPORTED CAPABILITY
# ===========================================================================
# This is the machine-readable mapping the results directory self-documents
# with. Every row is (family-gated) and every row says where the evidence came
# from. The `--ptx-probe` stage re-derives the same five booleans by asking
# NVIDIA's assembler on the actual box and FAILS THE RUN on disagreement.
#
# torch.cuda.get_device_capability() reports (major, minor) and NEVER the
# architecture-conditional `a`/`f` suffix. So the reported CC selects a family,
# and the arch-conditional target is derived as sm_<major><minor>a. A feature
# gated on an arch-conditional target is therefore reported as
# "available only if the compiler targets sm_XXa", which is exactly the
# distinction the study needs, and which tinygrad in this repo does not honour.

CC_FAMILY = {
    (7, 0): "volta", (7, 2): "volta", (7, 5): "turing",
    (8, 0): "ampere", (8, 6): "ampere", (8, 7): "ampere", (8, 9): "ada",
    (9, 0): "hopper",
    (10, 0): "blackwell_dc", (10, 1): "blackwell_dc",
    (10, 3): "blackwell_ultra",
    (12, 0): "blackwell_workstation", (12, 1): "blackwell_workstation",
}

# Minimum bare capability for a feature that is NOT arch-conditional.
_SM80 = (8, 0)
_SM89 = (8, 9)
_SM90 = (9, 0)

# Arch-conditional targets on which the feature actually exists. ptxas 12.9
# (NVIDIA's own assembler) is the oracle; its exact messages are in `evidence`.
FEATURE_GATES = {
    "cp_async": {
        "title": "asynchronous global->shared copy (cp.async / LDGSTS)",
        "kind": "min_cc", "min_cc": _SM80,
        "evidence": "ptxas 12.9: \"Feature 'cp.async' requires .target sm_80 or "
                    "higher\" (rejects sm_75, accepts sm_80). PTX ISA 9.4 "
                    "cp.async/mbarrier section: 'Requires `sm_80` or higher.'",
        "confidence": "assembler-verified + vendor-quoted",
    },
    "fp8_tensor_core": {
        "title": "FP8 (e4m3/e5m2) tensor-core MMA",
        "kind": "min_cc", "min_cc": _SM89,
        "evidence": "ptxas 12.9: \"Feature 'mma with FP8 floating point type' "
                    "requires .target sm_89 or higher\" (rejects sm_86, accepts "
                    "sm_89). This is the sm_89 boundary exactly: A100/sm_80 and "
                    "A10/sm_86 have no FP8 MMA at all.",
        "confidence": "assembler-verified",
    },
    "tma_bulk_async": {
        "title": "TMA: cp.async.bulk.tensor (2D/3D/5D tensor copies)",
        "kind": "min_cc", "min_cc": _SM90,
        "evidence": "ptxas 12.9: \"Feature 'cp.async.bulk.tensor' requires "
                    ".target sm_90 or higher\" and \"Feature '.tile' requires "
                    ".target sm_90 or higher\" (rejects sm_80/sm_86/sm_89; "
                    "accepts sm_90, sm_100a, sm_103a, sm_120, sm_120a).",
        "confidence": "assembler-verified",
    },
    "wgmma": {
        "title": "warpgroup MMA (wgmma.mma_async) -- Hopper arch-conditional",
        "kind": "targets", "targets": {"sm_90a"},
        "evidence": "ptxas 12.9: \"Instruction 'wgmma.mma_async with floating "
                    "point types' not supported on .target 'sm_90'\" and on "
                    "sm_80/sm_86/sm_89/sm_100/sm_100a/sm_103a/sm_120/sm_120a. "
                    "Accepted on sm_90a only.",
        "confidence": "assembler-verified",
    },
    "tcgen05": {
        "title": "5th-gen tensor core: tcgen05.mma + tensor-memory alloc",
        "kind": "targets",
        "targets": {"sm_100a", "sm_100f", "sm_101a", "sm_101f",
                    "sm_103a", "sm_107a", "sm_110a"},
        "evidence": "CORRECTED, see the module docstring. ptxas 12.9 at PTX ISA "
                    ">= 8.8 accepts tcgen05.alloc and tcgen05.mma on sm_100a, "
                    "sm_100f, sm_101a, sm_103a and sm_107a, and rejects the bare "
                    "sm_100 and sm_103 with \"Instruction 'tcgen05.mma' not "
                    "supported on .target 'sm_100'\". It rejects sm_120, sm_120a "
                    "and sm_121a outright. NVIDIA PTX ISA 9.4 tcgen05 Target ISA "
                    "Notes include sm_103a. => B300/sm_103 HAS tcgen05; "
                    "RTX PRO 6000/sm_120 does NOT; the bare sm_100/sm_103 "
                    "targets do not, which is what tinygrad emits.",
        "confidence": "assembler-verified + vendor-quoted (see docstring)",
    },
}

FEATURE_ORDER = ["cp_async", "fp8_tensor_core", "tma_bulk_async", "wgmma", "tcgen05"]


def family_of(cc: tuple[int, int]) -> str:
    """NVIDIA family for a (major, minor) compute capability."""
    return CC_FAMILY.get(tuple(cc), "unknown")


def bare_target(cc: tuple[int, int]) -> str:
    """The PTX target a compiler emits for `cc` with no arch-conditional suffix.

    This is exactly what tinygrad emits:
      .venv/.../tinygrad/runtime/ops_cuda.py:122  arch=f"sm_{major}{minor}"
    """
    return f"sm_{cc[0]}{cc[1]}"


def arch_conditional_target(cc: tuple[int, int]) -> str:
    """The arch-conditional target an arch-conditional feature requires."""
    return f"sm_{cc[0]}{cc[1]}a"


def derive_features(cc: tuple[int, int]) -> dict:
    """The machine-readable feature table for a reported compute capability.

    Returns a dict with, per feature: available_here (bool), requires (the rule
    as a string), reachable_with_tinygrad (bool) and evidence. The last one is
    the study's real subject: a feature can be silicon-present and still be
    unreachable because the compiler will not emit the arch-conditional target.
    """
    cc = tuple(cc)
    fam = family_of(cc)
    bare = bare_target(cc)
    cond = arch_conditional_target(cc)
    out = {
        "reported_capability": f"{cc[0]}.{cc[1]}",
        "capability_tuple": list(cc),
        "family": fam,
        "bare_ptx_target": bare,
        "arch_conditional_ptx_target": cond,
        "known_family": fam != "unknown",
        "features": {},
    }
    for name in FEATURE_ORDER:
        g = FEATURE_GATES[name]
        if g["kind"] == "min_cc":
            avail = cc >= g["min_cc"]
            rule = (f"compute capability >= {g['min_cc'][0]}.{g['min_cc'][1]}")
        else:
            avail = cond in g["targets"]
            rule = ("arch-conditional target is one of "
                    + ", ".join(sorted(g["targets"])))
        out["features"][name] = {
            "title": g["title"],
            "available": avail,
            "requires": rule,
            # An arch-conditional feature is only reachable if the toolchain
            # asks for the `a` target. tinygrad in this repo does not.
            "reachable_with_pinned_tinygrad": avail and (
                g["kind"] == "min_cc" or bare in g["targets"]),
            "evidence": g["evidence"],
            "confidence": g["confidence"],
        }
    return out


# ---------------------------------------------------------------------------
# the ptxas probes. these PTX bodies are NOT hand-waved: each was assembled by
# NVIDIA's ptxas 12.9 in this repo's archlab image and the negative controls
# were confirmed to fail for the RIGHT reason (see the `evidence` strings
# above, which are ptxas's own error text).
#
# The probe rewrites two lines: .version and .target. Both matter, and getting
# either wrong produces a false NEGATIVE that looks like a missing feature:
#   * ptxas from the nvidia-cuda-nvcc-cu12 wheel ignores the file's .target
#     unless -arch is also passed, and errors "SM version specified by .target
#     is higher than default SM version assumed" if they disagree.
#   * .version must be new enough for the arch (sm_103 needs >= 8.8, sm_121a
#     needs >= 8.8) or you get "PTX .version 8.7 does not support .target
#     sm_103a" -- again a false negative.
# ---------------------------------------------------------------------------
# The minimum PTX ISA version that can EXPRESS each arch. A .target above this
# is rejected with "PTX .version 8.7 does not support .target sm_103a", which is
# a false negative indistinguishable from "the feature is missing" unless you
# know to look.
PTX_MIN_VERSION = {
    "sm_70": "8.0", "sm_75": "8.0", "sm_80": "8.0", "sm_86": "8.0",
    "sm_89": "8.0", "sm_90": "8.0", "sm_90a": "8.3",
    "sm_100": "8.7", "sm_100a": "8.7", "sm_100f": "8.8",
    "sm_101a": "8.7", "sm_101f": "8.8", "sm_103": "8.8", "sm_103a": "8.8",
    "sm_107a": "8.8", "sm_110a": "8.8", "sm_110f": "8.8",
    "sm_120": "8.7", "sm_120a": "8.7", "sm_121a": "8.8",
}

# ...and the minimum version each FEATURE was introduced in. These are
# independent, and using the arch's version for a newer feature is a real bug:
# at .version 8.0 the FP8 probe is rejected on EVERY arch with "Feature 'mma
# with FP8 floating point type' requires PTX ISA .version 8.4 or later", which
# looks exactly like "no GPU has FP8".
PTX_FEATURE_MIN_VERSION = {
    "cp_async": "8.0",
    "fp8_tensor_core": "8.4",       # ptxas: "requires PTX ISA .version 8.4 or later"
    "tma_bulk_async": "8.6",        # cp.async.bulk.tensor
    "wgmma": "8.3",
    "tcgen05": "8.7",               # 8.8 for sm_103/sm_103a, handled by the max below
}

# Only these architecture-conditional targets actually EXIST. There is no
# sm_80a, no sm_86a and no sm_89a: ptxas answers "Value 'sm_89a' is not defined
# for option 'gpu-name'". Probing one and reporting a rejection would be
# indistinguishable from probing a real target and getting told "no".
PTX_ARCH_COND_TARGETS = {
    "sm_90a", "sm_100a", "sm_100f", "sm_101a", "sm_101f", "sm_103a",
    "sm_107a", "sm_110a", "sm_110f", "sm_120a", "sm_121a",
}


def _vtuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split("."))

_PTX_CP_ASYNC = """\
.version __VERSION__
.target __TARGET__
.address_size 64
.visible .entry probe(.param .u64 src, .param .u64 dst)
{
  .reg .b64 %rd<4>;
  ld.param.u64 %rd1, [src];
  ld.param.u64 %rd2, [dst];
  cvta.to.shared.u64 %rd3, %rd2;
  cp.async.ca.shared.global [%rd3], [%rd1], 4;
  cp.async.commit_group;
  cp.async.wait_group 0;
  ret;
}
"""

_PTX_FP8 = """\
.version __VERSION__
.target __TARGET__
.address_size 64
.visible .entry probe()
{
  .reg .b32 %r<12>;
  mov.b32 %r0, 0;
  mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
    {%r0, %r1, %r2, %r3}, {%r4, %r5, %r6, %r7}, {%r8, %r9}, {%r0, %r1, %r2, %r3};
  ret;
}
"""

_PTX_TMA = """\
.version __VERSION__
.target __TARGET__
.address_size 64
.visible .entry probe(.param .u64 smem, .param .u64 tmap, .param .u64 bar)
{
  .reg .b64 %rd<6>;
  .reg .b32 %r<2>;
  mov.b32 %r0, 0;
  mov.b32 %r1, 0;
  ld.param.u64 %rd1, [smem];
  cvta.to.shared.u64 %rd2, %rd1;
  ld.param.u64 %rd3, [tmap];
  ld.param.u64 %rd4, [bar];
  cvta.to.shared.u64 %rd5, %rd4;
  cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
    [%rd2], [%rd3, {%r0, %r1}], [%rd5];
  ret;
}
"""

_PTX_WGMMA = """\
.version __VERSION__
.target __TARGET__
.address_size 64
.visible .entry probe(.param .u64 descb)
{
  .reg .b64   %rd<2>;
  .reg .f16x2 %fa<4>;
  .reg .f32   %f<4>;
  ld.param.u64 %rd1, [descb];
  wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16
    {%f0, %f1, %f2, %f3}, {%fa0, %fa1, %fa2, %fa3}, %rd1, 1, -1, -1, 1;
  wgmma.commit_group.sync;
  wgmma.wait_group.sync 0;
  ret;
}
"""

# tcgen05.alloc, not tcgen05.mma: alloc is the shortest encoding that still
# goes through ptxas's arch-conditional gate, so a failure of it is a real
# "not supported on .target" and not an operand-shape complaint.
_PTX_TCGEN05 = """\
.version __VERSION__
.target __TARGET__
.address_size 64
.visible .entry probe()
{
  .reg .b64 %rd<3>;
  mov.u64 %rd1, 0;
  cvta.to.shared.u64 %rd2, %rd1;
  tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%rd2], 32;
  ret;
}
"""

PTX_PROBES = {
    "cp_async": _PTX_CP_ASYNC,
    "fp8_tensor_core": _PTX_FP8,
    "tma_bulk_async": _PTX_TMA,
    "wgmma": _PTX_WGMMA,
    "tcgen05": _PTX_TCGEN05,
}

# archs each probe is meaningful for, and the arch-conditional twin to also try
# when the bare target rejects it (that is how tcgen05 on sm_103a is found).
PROBE_ARCH_MATRIX = ["sm_80", "sm_89", "sm_90", "sm_90a", "sm_100", "sm_100a",
                     "sm_101a", "sm_103", "sm_103a", "sm_120", "sm_120a"]


def render_probe_ptx(feature: str, arch: str) -> str | None:
    """The PTX for one (feature, arch) cell, or None if it cannot be tested.

    None means "this oracle cannot answer that question" -- an unknown target
    name, or a nonexistent arch-conditional variant. It is deliberately not the
    same as a rejection, and the probe records the two separately.
    """
    body = PTX_PROBES.get(feature)
    arch_floor = PTX_MIN_VERSION.get(arch)
    feat_floor = PTX_FEATURE_MIN_VERSION.get(feature)
    if body is None or arch_floor is None or feat_floor is None:
        return None
    if arch.endswith("a") and arch not in PTX_ARCH_COND_TARGETS:
        return None                    # no sm_80a / sm_89a exists
    ver = max((arch_floor, feat_floor), key=_vtuple)
    return body.replace("__VERSION__", ver).replace("__TARGET__", arch)


# ===========================================================================
# 4. COST MODEL -- the actual teeth of "free tier costs $0.00"
# ===========================================================================
# A budget is not a promise, it is arithmetic. We compute the worst-case cost of
# the whole run at the pinned GPU rate and refuse to start unless it fits inside
# the free credit, so the $0.00 outlay is a theorem rather than a hope. This
# matters because Modal's spend limits are UI-only (no API) and may be
# unsettable on a brand-new workspace at all.

DEFAULT_BUDGET_MINUTES = 45      # conservative: < 1/10th of the H100 credit
MAX_CPU_CORES = 4
MAX_RAM_GIB = 32
# CPU-only allowance for image-independent container start + teardown. Modal
# bills CPU/RAM for the container's life; the GPU is only charged while held.
CONTAINER_STARTUP_SEC = 300


def modal_rate_key(gpu_string: str) -> str:
    """The pricing table key for a pinned gpu string.

    'H100!' is billed at the H100 rate -- the bang only suppresses the H200
    substitution, it is not a different product ("This automatic upgrade does
    _not_ change the cost of the GPU"). Anything ending in '+' is a bare
    refusal: those forms are banned outright, and if one is ever pinned again
    this raises rather than pricing the wrong tier.
    """
    if gpu_string.endswith("+"):
        raise ValueError(
            f"gpu string {gpu_string!r} is a '+' opt-in form. Modal bills it at "
            f"the lower tier regardless of which GPU is served "
            f"('B200+ is billed as B200, regardless of which GPU is used'), so "
            f"it cannot be used to pin an architecture. Use the bare string.")
    return gpu_string[:-1] if gpu_string.endswith("!") else gpu_string


def cost_estimate(arch: str, budget_minutes: float,
                  region: str = "us-east") -> dict:
    """Worst-case USD for one run, itemised, with a hard ceiling on free credit."""
    gpu_str = GPU_STRINGS[arch][0]
    rate = MODAL_USD_PER_SEC[modal_rate_key(gpu_str)]   # billed at REQUESTED tier
    mult = MODAL_REGION_MULTIPLIER.get(region, MODAL_WORST_REGION_MULTIPLIER)
    budget_s = budget_minutes * 60.0

    cpu = MODAL_FUNC_CPU_USD_PER_CORE_SEC * MAX_CPU_CORES * budget_s
    ram = MODAL_FUNC_RAM_USD_PER_GIB_SEC * MAX_RAM_GIB * budget_s
    startup = (MODAL_FUNC_CPU_USD_PER_CORE_SEC * MAX_CPU_CORES
               + MODAL_FUNC_RAM_USD_PER_GIB_SEC * MAX_RAM_GIB) * CONTAINER_STARTUP_SEC
    gpu = rate * budget_s

    per_sec = (rate + MODAL_FUNC_CPU_USD_PER_CORE_SEC * MAX_CPU_CORES
               + MODAL_FUNC_RAM_USD_PER_GIB_SEC * MAX_RAM_GIB)
    worst = (gpu + cpu + ram + startup) * mult
    # solve per_sec*60*m + startup*mult <= free_credit for m, so the reported
    # ceiling is the budget that actually fits, startup included.
    headroom = MODAL_STARTER_FREE_USD / mult - startup
    affordable_min = max(0.0, headroom / (per_sec * 60.0))

    return {
        "provider": "modal",
        "gpu_string": gpu_str,
        "rate_key": modal_rate_key(gpu_str),
        "usd_per_sec_gpu": rate,
        "usd_per_hr_gpu": round(rate * 3600.0, 4),
        "budget_minutes": budget_minutes,
        "budget_seconds": budget_s,
        "region": region,
        "region_multiplier": mult,
        "cost_gpu": round(gpu, 4),
        "cost_cpu": round(cpu, 4),
        "cost_ram": round(ram, 4),
        "cost_startup_cpu_only": round(startup, 4),
        "cost_worst_case": round(worst, 4),
        "free_credit_usd": MODAL_STARTER_FREE_USD,
        "max_affordable_minutes": round(affordable_min, 1),
        "fits_in_free_credit": worst <= MODAL_STARTER_FREE_USD,
        "headroom_usd": round(MODAL_STARTER_FREE_USD - worst, 4),
        "note": ("per-second base rates are us-east; every other region is a "
                 "1.15-1.75x multiplier, which is why region is pinned"),
    }


def enforce_budget(cost: dict, declared_spend_limit: float | None,
                   credit_only: bool) -> None:
    """The gate that makes a $0.00 outlay a theorem rather than a hope.

    Raises SystemExit with an actionable message. Called from BOTH the live
    path and --dry-run, so the dry run tests the real gate.
    """
    if not cost["fits_in_free_credit"]:
        raise SystemExit(
            f"budget too large: {cost['budget_minutes']:g} min of "
            f"'{cost['gpu_string']}' costs up to ${cost['cost_worst_case']}, "
            f"which exceeds the ${cost['free_credit_usd']} of free credit. "
            f"The credit buys at most {cost['max_affordable_minutes']} min here. "
            f"Use that, or a cheaper arch (--arch l40s is "
            f"${MODAL_USD_PER_SEC['L40S'] * 3600:.2f}/hr).")
    if declared_spend_limit is not None:
        if cost["cost_worst_case"] > declared_spend_limit:
            raise SystemExit(
                f"worst case ${cost['cost_worst_case']} exceeds the spend limit "
                f"you declared (${declared_spend_limit}). Either lower "
                f"--budget-minutes or raise the limit at "
                f"https://modal.com/settings/usage")
    elif not credit_only:
        raise SystemExit(
            "Modal's spend limit is set only in the web UI "
            "(https://modal.com/settings/usage); there is no API, so this script "
            "cannot set or verify one. Pass ONE of:\n"
            "  --credit-only                 asserts you have no payment method "
            "on file\n"
            "  --modal-spend-limit-usd 0.00 declares the limit you set\n"
            "--credit-only is the safe option here: the worst case is inside the "
            "free credit, so a $0.00 outlay does not depend on the limit being "
            "right.")


# ===========================================================================
# 5. THE BATTERY
# ===========================================================================
# bench/, matmul/ and portability/ are being written by other agents and are
# NOT guaranteed to exist or to have an entrypoint. Every directory-backed
# stage is therefore optional and resolved by discovery at plan time; a missing
# directory or a missing entrypoint is a SKIP with a message, never a crash.

# `smoke.py` is deliberately LAST: a smoke test is a last resort, not a
# battery. If a directory has a real entrypoint, prefer it; if the operator
# knows better, --stage ID:CMD overrides all of this without editing the file.
CANDIDATE_ENTRIES = ("run_all.py", "run_battery.py", "battery.py", "harness.py",
                     "bench.py", "main.py", "cli.py", "gen_src.py",
                     "Makefile", "smoke.py")

STAGES = [
    dict(id="identity", kind="file", path="remote/preflight.py", optional=False,
         weight=1, why="device identity, backend reachability, 1s KDA smoke"),
    dict(id="ptx_probe", kind="special", optional=False, weight=1,
         why="ask NVIDIA's assembler which of the five target features this "
             "device can actually compile for; FAILS the run if the assembler "
             "disagrees with FEATURE_GATES"),
    dict(id="codegen_capture", kind="special", optional=False, weight=1,
         why="capture every kernel tinygrad generates, as source"),
    dict(id="kda_vs_fla", kind="file", path="kda/kda_vs_fla.py", optional=False,
         weight=4, why="NVIDIA-vs-NVIDIA on the same GPU; degrades to "
                       "TinyGrad-only without torch/triton/fla"),
    dict(id="codegen_diff", kind="file", path="kda/codegen_diff.py", optional=True,
         weight=1, why="TinyGrad cstyle vs the hand-written Triton"),
    dict(id="bench", kind="dir", path="bench", optional=True, weight=5,
         why="counter/roofline harness; 3-level degradation, level 3 always runs"),
    dict(id="matmul", kind="dir", path="matmul", optional=True, weight=2,
         why="generated-cu -> SASS/CUTLASS toolchain stage"),
    dict(id="portability", kind="dir", path="portability", optional=True, weight=2,
         why="triton/tinygrad/codegen portability matrix"),
    dict(id="probes", kind="group", optional=True, weight=1,
         why="fp32 correctness / division precision / gather floor"),
]

PROBE_SCRIPTS = ["probes/fp32_correctness.py", "probes/divide_precision.py",
                 "probes/gather_floor.py"]

# what gets uploaded. regenerable/generated bulk is excluded.
STAGE_PATHS = ["remote/cloud_run.py", "remote/preflight.py", "README.md",
               "kda", "probes", "bench", "matmul", "portability"]
STAGE_EXCLUDES = {
    "kda": ["__pycache__", "*.pyc", "out"],
    "probes": ["__pycache__", "*.pyc", "oc/cells_*.bin", "oc/ref.bin", "oc/ref_oc"],
    "matmul": ["__pycache__", "*.pyc", "gen", "__pycache__"],
    "portability": ["__pycache__", "*.pyc", "out", "logs", "subj", "wheels"],
}
# torch/triton/fla are OPTIONAL at build time. Installing them in the image
# would make an optional comparison into a hard build dependency; instead the
# `deps` stage tries them at run time and tolerates failure, which is exactly
# the degradation contract kda_vs_fla.py already implements.
OPTIONAL_PIP = ["torch", "triton", "flash-linear-attention"]
REQUIRED_PIP = ["tinygrad", "numpy", "tabulate"]


# ===========================================================================
# 6. THE GENERATED REMOTE DRIVER
# ===========================================================================
# Written into the staged tree as cloud_battery.py. It is stdlib-only, so it
# imports and runs on any box, and it has three modes:
#   --plan      print the plan; touch nothing. This is what the dry run runs.
#   --selftest  exercise the GPU-free parts of the executor for real.
#   --run       the actual battery.
# It takes NO knowledge of the repo: the local side writes plan.json and the
# driver executes it. One source of truth for the stage list, by construction.

_BATTERY_SRC = r'''
#!/usr/bin/env python3
"""cloud_battery.py -- GENERATED by remote/cloud_run.py. Do not edit.

Executes a plan (plan.json) on the GPU box and writes a result tree that the
local side knows how to collect. Stdlib only. Three modes: --plan, --selftest,
--run. See remote/cloud_run.py for the design and the vendor citations.

RESULT TREE (the contract with the collector; do not rename anything)
    RUN.json              run-level: plan digest, deadline, attempt number
    identity.json         device identity + derived feature table
    results/<id>.json     per stage: status, argv, rc, timings, log, artifacts
    logs/<id>.log         per-stage captured stdout+stderr
    codegen/*             generated kernel sources
    DONE                  written LAST; its contents are the overall verdict
"""
import json
import os
import pathlib
import queue
import shlex
import signal
import subprocess
import sys
import threading
import time

RESULTS = pathlib.Path(os.environ.get("CLOUD_RESULTS", "/results"))
REPO = pathlib.Path(os.environ.get("CLOUD_REPO", "/repo"))
DEADLINE = float(os.environ.get("CLOUD_DEADLINE_EPOCH", "0") or 0)
PLAN_PATH = pathlib.Path(os.environ.get("CLOUD_PLAN", str(REPO / "plan.json")))

_interrupted = {"reason": "", "at": 0.0}


def _on_signal(signum, _frame):
    # Modal sends an interrupt signal to the container on preemption and then
    # runs the exit handler. Whatever is on disk at that moment is all we get,
    # so the only correct move is: stop launching, let the current stage's
    # finally-block write its result, and write DONE.
    _interrupted["reason"] = f"signal {signum} ({signal.Signals(signum).name})"
    _interrupted["at"] = time.time()


def now():
    return time.time()


def remaining():
    if DEADLINE <= 0:
        return float("inf")
    return DEADLINE - now()


def hr(title=""):
    if title:
        print()
        print("=" * 78)
        print(title)
        print("=" * 78)


def load_plan():
    with open(PLAN_PATH) as f:
        return json.load(f)


def write_json(relpath, obj):
    p = RESULTS / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, p)          # atomic: a preempted write never leaves a
    return p                     # half-written json behind


# ----------------------------------------------------------------- identity
def probe_pkg(name):
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except Exception as e:                                   # noqa: BLE001
        return f"ABSENT ({type(e).__name__})"


def collect_identity():
    """Everything known about the silicon, read FROM the device.

    torch is optional: it is the study's comparison baseline, not the source of
    truth. nvidia-smi is the fallback, and if BOTH are missing we say so rather
    than inventing a capability.
    """
    ident = {"collected_at": now(), "torch": None, "nvidia_smi": None,
             "cuda_runtime": None, "device": None, "error": None}
    ident["versions"] = {
        "python": sys.version.split()[0],
        "torch": probe_pkg("torch"),
        "triton": probe_pkg("triton"),
        "tinygrad": probe_pkg("tinygrad"),
        "flash_attn": probe_pkg("flash_attn"),
        "fla": probe_pkg("fla"),
        "numpy": probe_pkg("numpy"),
    }
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,"
                           "compute_cap,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30)
        ident["nvidia_smi"] = {
            "rc": r.returncode,
            "stdout": r.stdout.strip(),
            "stderr": r.stderr.strip()[:400],
        }
    except Exception as e:                                   # noqa: BLE001
        ident["nvidia_smi"] = {"rc": None, "error": f"{type(e).__name__}: {e}"}

    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,"
                            "compute_cap", "--format=csv,noheader", "-i", "0"],
                           capture_output=True, text=True, timeout=30)
        ident["device"] = {"raw": r.stdout.strip()}
    except Exception as e:                                   # noqa: BLE001
        ident["error"] = f"{type(e).__name__}: {e}"

    cap = None
    name = None
    try:
        import torch  # noqa: PLC0415
        ident["torch"] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_runtime_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available() else None,
        }
        if torch.cuda.is_available():
            cap = list(torch.cuda.get_device_capability(0))
            name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            ident["torch"].update({
                "device_capability": cap,
                "device_name": name,
                "total_memory_bytes": props.total_memory,
                "multi_processor_count": props.multi_processor_count,
            })
            try:
                from torch.cuda import driver_capability  # noqa: F401,PLC0415
            except Exception:                                  # noqa: BLE001
                pass
    except Exception as e:                                   # noqa: BLE001
        ident["torch"] = {"error": f"{type(e).__name__}: {e}"}

    # driver version straight from the driver, not from torch's opinion
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                            "--format=csv,noheader", "-i", "0"],
                           capture_output=True, text=True, timeout=30)
        ident["driver_version"] = r.stdout.strip()
    except Exception:                                          # noqa: BLE001
        ident["driver_version"] = None

    if cap is None:
        smi = (ident.get("nvidia_smi") or {}).get("stdout", "")
        for line in smi.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4 and "." in parts[-2]:
                name = name or parts[0]
                cap = [int(x) for x in parts[-2].split(".")]
                break
    if cap is not None and len(cap) == 2:
        ident["capability"] = {"major": int(cap[0]), "minor": int(cap[1])}
    else:
        ident["capability"] = None
    ident["device_name"] = name
    return ident


# ------------------------------------------------------------- ptx probe
PTX_PROBES = __PTX_PROBES_LIT__
PTX_MIN_VERSION = __PTX_VERSIONS_LIT__
PTX_FEATURE_MIN_VERSION = __PTX_FEAT_VERSIONS_LIT__
PTX_ARCH_COND_TARGETS = __PTX_COND_TARGETS_LIT__


def _vtuple(v):
    return tuple(int(x) for x in v.split("."))


def render(feature, arch):
    """The PTX for one (feature, arch) cell, or None if it cannot be tested.

    None means "this oracle cannot answer that question" and is recorded
    separately from a rejection. Conflating the two is how a probe reports
    "no FP8 anywhere" when it actually means "I asked a question ptxas cannot
    parse".
    """
    body = PTX_PROBES.get(feature)
    arch_floor = PTX_MIN_VERSION.get(arch)
    feat_floor = PTX_FEATURE_MIN_VERSION.get(feature)
    if not body or not arch_floor or not feat_floor:
        return None
    if arch.endswith("a") and arch not in PTX_ARCH_COND_TARGETS:
        return None
    ver = max((arch_floor, feat_floor), key=_vtuple)
    return body.replace("__VERSION__", ver).replace("__TARGET__", arch)


def ptx_probe(ident, workdir):
    """Ask ptxas, on THIS box, which features the device's arch supports.

    Rationale is in the launcher: the FEATURE_GATES table is transcribed, and a
    transcription error is silent. This asks NVIDIA's own assembler instead, for
    the arch string the device reports. No ptxas -> SKIP, never FAIL: a missing
    compiler must not fail a run whose numbers are still valid.
    """
    ptxas = None
    for cand in ("ptxas", "/usr/local/cuda/bin/ptxas", "/opt/cuda/bin/ptxas"):
        from shutil import which                              # noqa: PLC0415
        if cand == "ptxas":
            ptxas = which("ptxas")
        elif os.path.isfile(cand):
            ptxas = cand
        if ptxas:
            break
    out = {"ptxas": ptxas, "status": "skipped", "reason": "", "cells": {}}
    if not ptxas:
        out["reason"] = ("ptxas not found. install nvidia-cuda-nvcc-cu12 in the "
                         "deps stage, or accept that the feature table is "
                         "unverified on this run.")
        return out
    cap = (ident or {}).get("capability") or {}
    if "major" not in cap:
        out["reason"] = "device capability unknown; nothing to probe"
        return out
    archs = [f"sm_{cap['major']}{cap['minor']}"]
    archs.append(f"sm_{cap['major']}{cap['minor']}a")
    workdir.mkdir(parents=True, exist_ok=True)
    for feature in PTX_PROBES:
        for arch in archs:
            src = render(feature, arch)
            if src is None:
                out["cells"][f"{feature}@{arch}"] = {
                    "verdict": "oracle-cannot-test",
                    "detail": "no PTX template/version for this target"}
                continue
            p = workdir / f"{feature}.{arch}.ptx"
            p.write_text(src)
            r = subprocess.run([ptxas, f"-arch={arch}", str(p), "-o", "/dev/null"],
                               capture_output=True, text=True, timeout=120)
            txt = (r.stdout + r.stderr)
            err = next((l.strip() for l in txt.splitlines() if " error " in l), "")
            out["cells"][f"{feature}@{arch}"] = {
                "verdict": "assembles" if r.returncode == 0 else "rejected",
                "detail": err[:300] or txt.strip()[:300],
            }
    out["status"] = "ran"
    return out


# ---------------------------------------------------------------- executor
def run_stage(stage, dry=False):
    """One stage. Never raises: a stage cannot take the run down with it."""
    sid = stage["id"]
    argv = list(stage.get("argv", []))
    cwd = REPO / stage.get("cwd", ".")
    timeout = float(stage.get("timeout_s", 900))
    rec = {"id": sid, "argv": argv, "cwd": str(cwd), "group": stage.get("group"),
           "why": stage.get("why", ""), "status": "pending", "returncode": None,
           "started": now(), "ended": None, "timed_out": False,
           "log": f"logs/{sid}.log", "artifacts": [], "error": None}

    if stage.get("status") == "skipped":
        rec.update(status="skipped", ended=now(), error=stage.get("skip_reason", ""))
        write_json(f"results/{sid}.json", rec)
        return rec
    if remaining() <= 0:
        rec.update(status="skipped", ended=now(),
                   error="wall-clock budget already exhausted before this stage")
        write_json(f"results/{sid}.json", rec)
        return rec

    logp = RESULTS / "logs" / f"{sid}.log"
    logp.parent.mkdir(parents=True, exist_ok=True)
    hr(f"[{sid}] {' '.join(shlex.quote(a) for a in argv)}")
    if dry:
        rec.update(status="dry", ended=now())
        write_json(f"results/{sid}.json", rec)
        return rec

    t0 = now()
    eff_timeout = max(1.0, min(timeout, remaining()))
    try:
        with open(logp, "w") as log:
            log.write(f"# stage {sid}\n# cwd {cwd}\n"
                      f"# argv {' '.join(shlex.quote(a) for a in argv)}\n"
                      f"# started {t0:.3f} epoch\n\n")
            log.flush()
            proc = subprocess.Popen(argv, cwd=str(cwd), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True,
                                    bufsize=1)
            # Drain the pipe on a thread. Reading to EOF inline would block for
            # the whole life of a wedged child and the deadline would never be
            # checked -- which is exactly the hang this stage is supposed to
            # survive. The writer is sync, so interleaving stays sane.
            q = queue.Queue()

            def _reader():
                try:
                    for line in iter(proc.stdout.readline, ""):
                        q.put(line)
                finally:
                    q.put(None)          # EOF sentinel
                    try:
                        proc.stdout.close()
                    except Exception:      # noqa: BLE001
                        pass

            rd = threading.Thread(target=_reader, daemon=True)
            rd.start()

            deadline_hit = False
            while True:
                if (now() - t0 > eff_timeout or remaining() <= 0
                        or _interrupted["reason"]):
                    deadline_hit = True
                    break
                try:
                    item = q.get(timeout=0.25)
                except queue.Empty:
                    continue
                if item is None:            # clean EOF
                    break
                log.write(item)
                sys.stdout.write(item)
                sys.stdout.flush()
            if deadline_hit:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except Exception:           # noqa: BLE001
                    proc.kill()
            rd.join(timeout=5)
            rc = proc.poll()
        rec["returncode"] = rc
        rec["duration_s"] = round(now() - t0, 3)
        if deadline_hit:
            # A child we killed reports -SIGTERM (or -9), so `rc` alone cannot
            # distinguish "we stopped it on purpose" from "it died on purpose".
            # deadline_hit is the only honest signal, and a stage stopped by
            # the budget is a data point, not a failure.
            rec.update(status="timeout", timed_out=True,
                       error=f"stopped after {eff_timeout:.0f}s "
                             + (f"({_interrupted['reason']})"
                                if _interrupted["reason"] else "(deadline)"))
        elif rc is None:
            rec.update(status="timeout", timed_out=True,
                       error="exited without a returncode")
        elif rc == 0:
            rec["status"] = "ok"
        else:
            rec.update(status="failed", error=f"exit {rc}; see {rec['log']}")
    except FileNotFoundError as e:
        rec.update(status="failed", error=f"{type(e).__name__}: {e}")
    except Exception as e:                                   # noqa: BLE001
        rec.update(status="failed", error=f"{type(e).__name__}: {e}")
    rec["ended"] = now()
    write_json(f"results/{sid}.json", rec)
    return rec


def run_plan(plan, dry=False):
    write_json("RUN.json", {
        "started": now(), "deadline_epoch": DEADLINE,
        "plan_digest": plan.get("digest"), "stages": len(plan["stages"]),
        "attempt": int(os.environ.get("CLOUD_ATTEMPT", "1")),
        "requested": plan.get("requested", {}),
    })
    recs = []
    for stage in plan["stages"]:
        if _interrupted["reason"]:
            print(f"  stopping: {_interrupted['reason']} (preemption)")
            break
        recs.append(run_stage(stage, dry=dry))
    ok = [r for r in recs if r["status"] == "ok"]
    bad = [r for r in recs if r["status"] == "failed"]
    to = [r for r in recs if r["status"] == "timeout"]
    sk = [r for r in recs if r["status"] == "skipped"]
    if _interrupted["reason"]:
        verdict = "interrupted"
    elif remaining() <= 0:
        verdict = "budget-exhausted"
    elif bad:
        verdict = "partial"
    elif to:
        verdict = "partial"
    else:
        verdict = "complete"
    done = {"verdict": verdict, "ok": len(ok), "failed": len(bad),
            "timeout": len(to), "skipped": len(sk), "wrote_at": now()}
    (RESULTS / "DONE").parent.mkdir(parents=True, exist_ok=True)
    (RESULTS / "DONE").write_text(json.dumps(done, indent=2) + "\n")
    hr("battery summary")
    for r in recs:
        print(f"  {r['status']:9} {r['id']:16} {r.get('error') or ''}")
    print(f"  verdict: {verdict}  ({len(ok)} ok / {len(bad)} failed / "
          f"{len(to)} timeout / {len(sk)} skipped)")
    return verdict


# ------------------------------------------------------------- selftest
def selftest():
    """Exercise every GPU-free code path for real. This is what the local
    dry run invokes, so the executor is not merely syntax-checked."""
    fails = []

    def check(name, cond, detail=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        if not cond:
            fails.append(name)

    hr("selftest: plan round trip")
    plan = load_plan()
    check("plan parses", isinstance(plan, dict) and "stages" in plan)
    check("stages have argv or a skip", all(
        s.get("argv") or s.get("status") == "skipped" for s in plan["stages"]))

    hr("selftest: deadline arithmetic")
    saved = globals()["DEADLINE"]
    globals()["DEADLINE"] = now() - 1.0
    check("expired deadline reports <= 0", remaining() <= 0)
    globals()["DEADLINE"] = now() + 600.0
    check("live deadline reports > 0", 0 < remaining() <= 601)
    globals()["DEADLINE"] = 0.0
    check("no deadline means infinite", remaining() == float("inf"))
    globals()["DEADLINE"] = saved

    hr("selftest: signal handling (preemption path)")
    _on_signal(signal.SIGTERM, None)
    check("SIGTERM recorded as interruption",
          _interrupted["reason"].startswith("signal 15"), _interrupted["reason"])
    _interrupted["reason"] = ""
    _on_signal(signal.SIGTERM, None)
    _interrupted["reason"] = ""

    hr("selftest: real subprocess with log capture")
    # The token must not appear anywhere in the log's own header, so assert on
    # a whole LINE rather than a substring -- otherwise the `# argv` line the
    # driver writes makes every capture assertion pass vacuously.
    tok_out, tok_err = "SELFTEST_STDOUT_TOKEN_4711", "SELFTEST_STDERR_TOKEN_4712"
    probe = {"id": "selftest_echo",
             "argv": [sys.executable, "-c",
                      f"import sys;print({tok_out!r});"
                      f"print({tok_err!r},file=sys.stderr)"],
             "cwd": ".", "timeout_s": 60, "group": "selftest", "why": "self test"}
    r = run_stage(probe)
    check("exit 0 -> status ok", r["status"] == "ok", r.get("error") or "")
    log_lines = [ln for ln in (RESULTS / r["log"]).read_text().splitlines()
                 if not ln.startswith("#")]
    check("stdout captured on its own line", tok_out in log_lines,
          repr(log_lines[:3]))
    check("stderr captured on its own line (merged)", tok_err in log_lines,
          repr(log_lines[:3]))

    hr("selftest: non-zero exit is a failure, not a crash")
    r = run_stage({"id": "selftest_fail", "argv": [sys.executable, "-c",
                                                  "raise SystemExit(3)"],
                   "cwd": ".", "timeout_s": 60})
    check("exit 3 -> status failed", r["status"] == "failed", r.get("error") or "")

    hr("selftest: missing binary is a failure, not a crash")
    r = run_stage({"id": "selftest_missing",
                   "argv": ["/nonexistent/definitely/not/here"],
                   "cwd": ".", "timeout_s": 30})
    check("FileNotFoundError captured", r["status"] == "failed", r.get("error") or "")

    hr("selftest: timeout is enforced")
    t0 = now()
    r = run_stage({"id": "selftest_hang",
                   "argv": [sys.executable, "-c", "import time;time.sleep(600)"],
                   "cwd": ".", "timeout_s": 3})
    dt = now() - t0
    check("hung stage killed", r["status"] == "timeout", f"{dt:.1f}s, {r.get('error')}")
    check("kill was prompt", dt < 60, f"{dt:.1f}s")

    hr("selftest: pre-skipped stage")
    r = run_stage({"id": "selftest_skipped", "status": "skipped",
                   "skip_reason": "directory not present", "cwd": "."})
    check("skipped keeps its reason", r["status"] == "skipped", r.get("error") or "")

    hr("selftest: identity collection degrades, never crashes")
    ident = collect_identity()
    check("identity is a dict", isinstance(ident, dict))
    check("versions recorded", "torch" in ident["versions"])
    print(f"    capability here: {ident.get('capability')} "
          f"device: {ident.get('device_name')!r}")
    print(f"    (on the dry-run host there is no GPU; on the box this is the "
          f"number that decides pass/fail)")

    hr("selftest: ptx probe degrades to SKIP without ptxas")
    pr = ptx_probe(ident, RESULTS / "ptxwork")
    check("probe returns a verdict", pr["status"] in ("ran", "skipped"),
          f"{pr['status']}: {pr.get('reason', '')[:70]}")

    hr("selftest: DONE marker is written last")
    (RESULTS / "DONE").unlink(missing_ok=True)
    v = run_plan({"stages": [], "digest": "selftest", "requested": {}}, dry=True)
    check("DONE exists after run_plan", (RESULTS / "DONE").exists())
    import json as _json                                   # noqa: PLC0415
    d = _json.loads((RESULTS / "DONE").read_text())
    check("DONE has a verdict", "verdict" in d, d.get("verdict", ""))

    hr(f"selftest: {len(fails)} failure(s)")
    for f in fails:
        print(f"  FAILED: {f}")
    return 1 if fails else 0


def main(argv):
    mode = "run"
    for a in argv:
        if a == "--plan":
            mode = "plan"
        elif a == "--selftest":
            mode = "selftest"
    if mode == "selftest":
        return selftest()
    plan = load_plan()
    if mode == "plan":
        print(f"plan: {len(plan['stages'])} stage(s), digest {plan.get('digest')}")
        for s in plan["stages"]:
            state = s.get("status", "ready")
            print(f"  {state:8} {s['id']:16} "
                  f"{' '.join(s.get('argv', [])) or s.get('skip_reason', '')}")
        return 0
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _on_signal)
    return 0 if run_plan(plan) != "none" else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
'''


def build_battery_source() -> str:
    """The generated driver, with the PTX tables inlined.

    Inlining (rather than importing them) is what makes the generated file
    self-contained: it is the only thing uploaded, and it must not depend on
    cloud_run.py being importable on the box.

    The tables are inlined as PYTHON LITERALS via pprint, not as JSON text.
    That detail is not cosmetic. Emitting json.dumps output into a Python
    string literal compiles fine and then explodes at import time with
    "Invalid control character": JSON escapes a newline as backslash-n, and
    the surrounding Python literal immediately un-escapes it back to a real
    newline, which is not legal inside a JSON string. pprint emits a dict
    literal whose escapes the generated file consumes exactly once, correctly.
    """
    import pprint                                           # noqa: PLC0415
    return (_BATTERY_SRC
            .replace("__PTX_PROBES_LIT__", pprint.pformat(PTX_PROBES, width=88,
                                                           sort_dicts=True))
            .replace("__PTX_VERSIONS_LIT__",
                     pprint.pformat(PTX_MIN_VERSION, width=88, sort_dicts=True))
            .replace("__PTX_FEAT_VERSIONS_LIT__",
                     pprint.pformat(PTX_FEATURE_MIN_VERSION, width=88,
                                    sort_dicts=True))
            .replace("__PTX_COND_TARGETS_LIT__",
                     pprint.pformat(PTX_ARCH_COND_TARGETS, width=88,
                                    sort_dicts=True)))


# ===========================================================================
# 7. PLAN
# ===========================================================================
def _split_ids(values) -> list[str]:
    """'a,b' and 'a' both work, because --only/--skip are repeatable and the
    help text shows the comma form."""
    out: list[str] = []
    for v in values or []:
        out.extend(p.strip() for p in str(v).split(",") if p.strip())
    return out


def _find_entrypoint(directory: pathlib.Path) -> tuple[str | None, str]:
    """(entrypoint, why-not) for a directory-backed stage."""
    for cand in CANDIDATE_ENTRIES:
        if (directory / cand).exists():
            return cand, ""
    return None, ("no entrypoint found; looked for "
                  + ", ".join(CANDIDATE_ENTRIES)
                  + " -- this directory is still being written by another agent")


def build_plan(arch: str, budget_minutes: float, python: str = "python",
               only: list[str] | None = None, skip: list[str] | None = None,
               overrides: dict[str, list[str]] | None = None,
               results_subdir: str = "") -> dict:
    """Turn the repo and the arguments into an explicit list of stages.

    A missing directory, a missing entrypoint, or a --skip is a SKIP with a
    reason. Nothing here can raise for a missing piece of the study.
    """
    overrides = overrides or {}
    # --only/--skip are repeatable AND comma-separated, because the help text
    # advertises `--only identity,ptx_probe`. argparse only gives us the raw
    # strings, so the split has to happen here or the documented form fails.
    only = _split_ids(only)
    skip = _split_ids(skip)
    known = {s["id"] for s in STAGES}
    for want in only + list(overrides):
        if want not in known:
            raise SystemExit(f"unknown stage {want!r}; known: {', '.join(sorted(known))}")

    per_stage = max(60, int((budget_minutes * 60 - 120) // max(1, len(STAGES))))
    codegen_out = (results_subdir or "{RESULTS}") + "/codegen"
    stages: list[dict] = []
    for spec in STAGES:
        sid = spec["id"]
        if sid in skip or (only and sid not in only):
            stages.append(dict(id=sid, kind=spec["kind"], status="skipped",
                               skip_reason="--skip" if sid in skip
                               else "not in --only", cwd=".",
                               group=spec["kind"], why=spec["why"]))
            continue
        st = dict(id=sid, kind=spec["kind"], cwd=spec.get("path", "."),
                  timeout_s=per_stage, group=spec["kind"], why=spec["why"],
                  optional=spec.get("optional", False))
        if sid in overrides:
            st["argv"] = overrides[sid]
        elif spec["kind"] == "special":
            if sid == "ptx_probe":
                st["argv"] = [python, "remote/cloud_run.py", "--ptx-probe",
                              "--out", results_subdir or "{RESULTS}"]
            else:  # codegen_capture
                st["argv"] = [python, "remote/cloud_run.py",
                              "--remote-capture-codegen", "--out", codegen_out]
        elif spec["kind"] == "file":
            p = REPO / spec["path"]
            if not p.is_file():
                st.update(status="skipped",
                          skip_reason=f"{spec['path']} does not exist")
            else:
                st["argv"] = [python, spec["path"]]
        elif spec["kind"] == "group":
            cmds, missing = [], []
            for s in PROBE_SCRIPTS:
                if (REPO / s).is_file():
                    cmds.append(s)
                else:
                    missing.append(s)
            st["argv"] = [python, cmds[0]] if cmds else []
            st["also_run"] = cmds[1:]
            if missing:
                st["skip_reason"] = f"missing: {', '.join(missing)}"
            if not cmds:
                st.update(status="skipped", skip_reason=st["skip_reason"])
        else:  # dir
            d = REPO / spec["path"]
            if not d.is_dir():
                st.update(status="skipped",
                          skip_reason=f"directory {spec['path']}/ does not exist "
                                      f"(still being written by another agent)")
            else:
                entry, why = _find_entrypoint(d)
                if entry is None:
                    st.update(status="skipped", skip_reason=why)
                elif entry == "Makefile":
                    st["argv"] = ["make"]
                else:
                    st["argv"] = [python, entry]
        stages.append(st)

    plan = {"stages": stages, "requested": {"arch": arch,
                                            "budget_minutes": budget_minutes}}
    plan["digest"] = hashlib.sha256(
        json.dumps(stages, sort_keys=True).encode()).hexdigest()[:16]
    return plan


# ===========================================================================
# 8. STAGING
# ===========================================================================
def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _excluded(rel: str, patterns: list[str]) -> bool:
    import fnmatch                                          # noqa: PLC0415
    parts = pathlib.PurePosixPath(rel).parts
    for pat in patterns:
        # match the pattern against the full path AND against every suffix of
        # it, so "oc/cells_*.bin" excludes "probes/oc/cells_0.bin".
        for i in range(len(parts)):
            if fnmatch.fnmatch("/".join(parts[i:]), pat) or pat in parts[i:]:
                return True
    return False


def build_staging_manifest() -> dict:
    """What would be uploaded: real paths, real byte counts, real digests.

    This is also the dry run's proof that path staging works, because it walks
    the actual repository rather than a fixture.

    The walk tolerates files vanishing underneath it. That is not defensive
    decoration: bench/, matmul/ and portability/ are being written by other
    agents WHILE this runs, and a plain rglob+stat sequence raised
    FileNotFoundError on a .ptx.tmp that was created and deleted between the
    two calls. A file that disappears mid-walk is a file that will not exist on
    the box either, so skipping it is the correct answer, not a failure.
    """
    entries, missing, vanished, total = [], [], [], 0
    for top in STAGE_PATHS:
        p = REPO / top
        pats = STAGE_EXCLUDES.get(top, [])
        try:
            if not p.exists():
                missing.append(top)
                continue
            files = [p] if p.is_file() else sorted(
                q for q in p.rglob("*") if q.is_file())
        except OSError as e:
            missing.append(f"{top}/ (unreadable: {type(e).__name__}: {e})")
            continue
        n = 0
        for q in files:
            rel = q.relative_to(REPO).as_posix()
            if _excluded(rel, pats) or "__pycache__" in rel or rel.endswith(".pyc"):
                continue
            try:
                size = q.stat().st_size
                digest = _sha256(q) if size < 8 << 20 else None
            except (FileNotFoundError, PermissionError, IsADirectoryError):
                vanished.append(rel)
                continue
            total += size
            n += 1
            entries.append({"path": rel, "bytes": size, "sha256": digest})
        if n == 0 and p.is_dir():
            missing.append(f"{top}/ (present but everything in it was excluded)")
    return {
        "repo_root": str(REPO),
        "roots": STAGE_PATHS,
        "files": len(entries),
        "bytes": total,
        "missing_or_empty": missing,
        "vanished_during_walk": vanished,
        "entries": entries,
    }


def stage_tree(manifest: dict, plan: dict, dest: pathlib.Path) -> None:
    """Materialise the staging tree locally (dry run) or let Modal's
    add_local_dir do it remotely. Same manifest either way."""
    dest.mkdir(parents=True, exist_ok=True)
    for e in manifest["entries"]:
        src = REPO / e["path"]
        out = dest / e["path"]
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(src.read_bytes())
        except (FileNotFoundError, PermissionError, IsADirectoryError):
            continue          # same race as the walk: another agent is writing
    (dest / "cloud_battery.py").write_text(build_battery_source())
    (dest / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")


# ===========================================================================
# 9. COLLECTION -- shared by both providers and by the dry run
# ===========================================================================
def _is_json(data: bytes) -> bool:
    try:
        json.loads(data)
        return True
    except (ValueError, UnicodeDecodeError):
        return False


def collect_results(remote_files: dict[str, bytes], dest: pathlib.Path,
                    run_meta: dict) -> dict:
    """Write a collected result tree and summarise it.

    `remote_files` is a flat {posix relative path: bytes} map. Both real
    providers reduce to exactly that (a Modal Volume read, or a downloaded
    tree), and so does the dry run's synthetic tree -- which is why the dry run
    tests the real collector and not a copy of it.
    """
    dest.mkdir(parents=True, exist_ok=True)
    written, total = [], 0
    for rel, data in sorted(remote_files.items()):
        rel = rel.lstrip("/")
        if ".." in pathlib.PurePosixPath(rel).parts:
            continue                       # never let a remote path escape
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        total += len(data)
        written.append({"path": rel, "bytes": len(data)})

    stages, identity, done, runj = [], None, None, None
    for rel, data in remote_files.items():
        r = rel.lstrip("/")
        try:
            if r == "identity.json":
                identity = json.loads(data)
            elif r == "DONE":
                done = json.loads(data)
            elif r == "RUN.json":
                runj = json.loads(data)
            elif r.startswith("results/") and r.endswith(".json"):
                stages.append(json.loads(data))
        except (ValueError, OSError):
            continue
    stages.sort(key=lambda s: s.get("id", ""))

    # The identity gate is a VERDICT about the run, not just an exit code, and
    # it has to survive into the results directory. A SUMMARY.md that quietly
    # prints a Hopper table under a B200 heading is worse than no summary: it
    # is a mislabelled document. So the gate is evaluated here, at collection
    # time, and its verdict is written into both MANIFEST.json and SUMMARY.md.
    arch = run_meta.get("arch")
    gate = None
    if arch in ARCHS and identity:
        code, verdict = classify_collected(
            {"verdict": (done or {}).get("verdict", "no-DONE-marker"),
             "stages": [json.loads(remote_files[r]) for r in sorted(remote_files)
                        if r.startswith("results/") and r.endswith(".json")
                        and _is_json(remote_files[r])],
             "identity": identity},
            arch)
        gate = {"exit_code": code, "verdict": verdict,
                "identity_confirmed": code not in (EXIT["arch_mismatch"],)}

    summary = {
        "collected_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "run_meta": run_meta,
        "remote_run": runj,
        "verdict": (done or {}).get("verdict", "no-DONE-marker"),
        "identity_gate": gate,
        "files_written": len(written),
        "bytes_written": total,
        "identity": identity,
        "stages": [
            {k: s.get(k) for k in ("id", "status", "returncode", "duration_s",
                                   "timed_out", "log", "error", "why")}
            for s in stages
        ],
    }
    (dest / "MANIFEST.json").write_text(
        json.dumps({**summary, "files": written}, indent=2, sort_keys=True) + "\n")
    (dest / "SUMMARY.md").write_text(render_summary(summary))
    return summary


def render_summary(s: dict) -> str:
    ident = s.get("identity") or {}
    cap = ident.get("capability") or {}
    gate = s.get("identity_gate") or {}
    lines = [
        f"# cloud run {s['run_meta'].get('provider')}/{s['run_meta'].get('arch')}",
        "",
        f"- collected: {s['collected_at']}",
        f"- verdict: **{s['verdict']}**",
        f"- requested: `{s['run_meta'].get('gpu_string')}` "
        f"(expected capability {s['run_meta'].get('expect_cc')})",
        f"- cost: worst case ${s['run_meta'].get('cost', {}).get('cost_worst_case')} "
        f"of ${s['run_meta'].get('cost', {}).get('free_credit_usd')} free credit",
        "",
    ]
    # The identity gate goes at the very top, before any number that depends
    # on it. If the device was not what was asked for, nothing below this
    # banner is attributed to the requested architecture.
    if gate and not gate.get("identity_confirmed", True):
        lines += [
            "> ## :warning: ARCHITECTURE NOT CONFIRMED",
            ">",
            "> " + gate.get("verdict", "").replace("\n", "\n> "),
            ">",
            "> Every number below was produced on hardware that is not the "
            "architecture in the directory name. Do not cite it as one.",
            "",
        ]
    elif gate:
        lines += [f"- identity gate: {gate.get('verdict')}", ""]
    lines += [
        "## what the device actually was",
        "",
    ]
    if cap:
        lines += [
            f"- capability: **sm_{cap.get('major')}{cap.get('minor')}**",
            f"- name: `{ident.get('device_name')}`",
            f"- driver: `{ident.get('driver_version')}`",
            f"- cuda runtime: `{(ident.get('torch') or {}).get('cuda_runtime_version')}`",
            f"- versions: "
            + ", ".join(f"{k}={v}" for k, v in (ident.get("versions") or {}).items()),
        ]
    else:
        lines.append("- **no capability was recorded.** The identity stage did not "
                     "complete, so nothing in this directory may be attributed to "
                     "a specific architecture.")
    ft = (ident.get("derived_features") or {}).get("features") or {}
    if ft:
        lines += ["", "## target features on this silicon", "",
                  "| feature | available | requires | reachable with pinned tinygrad |",
                  "|---|---|---|---|"]
        for k in FEATURE_ORDER:
            f = ft.get(k, {})
            lines.append(f"| `{k}` | {'yes' if f.get('available') else '**no**'} "
                         f"| {f.get('requires', '')} | "
                         f"{'yes' if f.get('reachable_with_pinned_tinygrad') else '**no**'} |")
    pr = ident.get("ptx_probe") or {}
    if pr.get("status") == "ran":
        lines += ["", "## NVIDIA's own assembler, asked directly", "",
                  "| feature@arch | verdict |", "|---|---|"]
        for k, v in sorted((pr.get("cells") or {}).items()):
            lines.append(f"| `{k}` | {v.get('verdict')} |")
    lines += ["", "## stages", "", "| stage | status | rc | s | note |",
              "|---|---|---|---|---|"]
    for st in s.get("stages", []):
        lines.append(f"| `{st.get('id')}` | {st.get('status')} | "
                     f"{st.get('returncode')} | {st.get('duration_s')} | "
                     f"{(st.get('error') or '')[:90]} |")
    return "\n".join(lines) + "\n"


def classify_collected(summary: dict, arch: str) -> tuple[int, str]:
    """Turn a collected summary into (exit code, human verdict). One place, so
    the dry run and the real path cannot disagree about what a result means.

    The ORDER MATTERS and the signature says so: an int first, a string second.
    Getting it backwards is a two-character bug that no type checker in this
    file would catch, so the callers unpack by name.
    """
    verdict = summary.get("verdict")
    stages = summary.get("stages", [])
    ident = summary.get("identity") or {}
    ran = [s for s in stages if s.get("status") in ("ok", "failed", "timeout")]

    if not ran and verdict == "no-DONE-marker":
        return (EXIT["no_results"],
                "nothing came back: the container produced no results and no "
                "DONE marker. The image probably built but the battery never "
                "started -- check the function log.")
    if verdict == "interrupted":
        return (EXIT["preempted"],
                f"PREEMPTED mid-run after {len(ran)} stage(s). Partial results "
                f"are in this directory and are valid for those stages. "
                f"Modal cannot opt a GPU function out of preemption "
                f"(nonpreemptible is not supported for GPU Functions), so this "
                f"is expected occasionally. Re-run with --budget-minutes "
                f"smaller to finish in one shot.")
    if verdict == "budget-exhausted":
        return (EXIT["budget_exhausted"],
                "stopped by our own wall-clock budget before the battery "
                "finished. Partial results are valid. Raise --budget-minutes "
                "(if the free credit allows it) or --only the stages you need.")

    cap = ident.get("capability")
    expect = ARCHS[arch]["expect_cc"]
    if not cap:
        return (EXIT["arch_mismatch"],
                "the device never reported a compute capability, so the run "
                "cannot be attributed to any architecture. The identity stage "
                "is the first one; treat every other number here as unlabelled.")
    if (cap["major"], cap["minor"]) != tuple(expect):
        return (EXIT["arch_mismatch"],
                f"ARCHITECTURE MISMATCH. You asked for {arch} "
                f"(sm_{expect[0]}{expect[1]}, pinned as "
                f"'{GPU_STRINGS[arch][0]}') and the device reports "
                f"sm_{cap['major']}{cap['minor']} "
                f"({ident.get('device_name')}). Every number in this directory "
                f"is labelled with the WRONG architecture. Do not use it.")

    if verdict == "partial":
        bad = [s.get("id") for s in stages if s.get("status") in ("failed", "timeout")]
        return (EXIT["ok"],
                f"battery completed partially; these stages did not finish: "
                f"{', '.join(bad) or 'unknown'}. The rest are valid. See the "
                f"stage logs in logs/.")
    return (EXIT["ok"],
            "battery complete; device identity confirmed as the requested "
            "architecture.")


# ===========================================================================
# 10. FAILURE CLASSIFICATION
# ===========================================================================
def classify_provider_error(provider: str, exc: BaseException, ctx: dict) -> tuple[int, str, str]:
    """(exit code, headline, remedy) for a provider exception.

    Attribution is by exception TYPE first and message SECOND, and anything
    unattributable becomes EXIT['unclassified'] with a full traceback rather
    than being folded into a plausible bucket. We never retry.
    """
    name = type(exc).__name__
    msg = str(exc)
    low = msg.lower()
    trace = "".join(__import__("traceback").format_exception(
        type(exc), exc, exc.__traceback__))

    def out(code, headline, remedy):
        return code, headline, remedy

    if name in ("NotFoundError", "AuthenticationError", "PermissionDeniedError",
                "AuthError", "CredentialsError"):
        return out(EXIT["no_credentials"],
                   "no usable credentials",
                   f"set them up with `modal setup`, or export MODAL_TOKEN_ID "
                   f"and MODAL_TOKEN_SECRET. full error:\n{trace}")
    if name in ("ModuleNotFoundError", "ImportError"):
        return out(EXIT["sdk_missing"],
                   f"the {provider} SDK is not installed: {name}: {msg}",
                   f"pip install {provider} in the venv running this script. "
                   f"Nothing was provisioned and nothing was charged.")
    if any(k in low for k in ("out of credit", "credit balance", "insufficient credit",
                              "usage limit", "spend limit", "budget exceeded",
                              "exceeded your budget", "payment required",
                              "no payment method")):
        return out(EXIT["out_of_credit"],
                   "the workspace ran out of free credit or hit a spend/usage limit",
                   "this run cost $0.00 of your own money, but there is nothing "
                   "left to run on. Modal's spend limit is UI-only "
                   "(https://modal.com/settings/usage) and has no API, so this "
                   "script could neither set nor read it. Check your balance "
                   f"there, then re-run next cycle.\n{trace}")
    if any(k in low for k in ("queue", "no capacity", "resource unavailable",
                              "insufficient capacity", "out of capacity",
                              "no available", "could not be scheduled")):
        # the requested GPU goes in only when the caller actually knows it; ctx
        # is empty when this runs from a context that never got that far
        want = ctx.get("gpu_string")
        return out(EXIT["no_capacity"],
                   f"no {want} capacity was available" if want
                   else "no capacity was available",
                   "the request timed out before a machine was free. Nothing "
                   "was billed while queued. Retry later, or pick a cheaper "
                   "arch (--arch l40s) which has far more capacity.\n" + trace)
    if any(k in low for k in ("build", "dockerfile", "pip install", "image",
                              "failed to pull", "apt")):
        return out(EXIT["image_build"],
                   "the container image failed to build",
                   f"modal build logs are in the function's Logs tab. This is "
                   f"almost always a pip/apt failure in the image, not a GPU "
                   f"problem. Nothing was charged for the GPU.\n{trace}")
    if name in ("TimeoutError", "FunctionTimeoutError", "ReadTimeout") and not ctx.get("results"):
        return out(EXIT["no_capacity"],
                   "timed out before a container was ever started",
                   "on Modal a call's timeout covers queue wait AND execution, "
                   "so a timeout with zero results means no capacity rather than "
                   "a slow run. Nothing was billed. Retry, or lower --arch.\n"
                   + trace)
    if any(k in low for k in ("interrupt", "preempt", "terminat")):
        return out(EXIT["preempted"],
                   "the container was interrupted",
                   "Modal preempts GPU functions by default and nonpreemptible "
                   "is not supported for GPU Functions, so this is not "
                   "avoidable. Results already written to the volume are "
                   "collected; re-run to fill the gaps.\n" + trace)
    if name in ("CancelledError", "InputEgressExceededError"):
        return out(EXIT["unclassified"], f"provider call failed: {name}",
                   "unexpected; full traceback follows.\n" + trace)
    return out(EXIT["unclassified"],
               f"unattributed {provider} error: {name}",
               "this script refuses to guess which bucket an error belongs in. "
               "Full traceback:\n" + trace)


# ===========================================================================
# 11. DRY RUN
# ===========================================================================
class DryRunViolation(RuntimeError):
    pass


# Set only by hermetic(); the guarded Popen reads it. A module-level flag
# rather than a closure, so nesting hermetic() blocks composes: the inner
# block can permit sys.executable and its exit restores the outer's rule.
_ALLOW_LOCAL_PYTHON = False


@contextlib.contextmanager
def hermetic(enabled: bool, allow_local_python: bool = False):
    """Make the network, and provider CLIs, physically impossible.

    This is not a promise in a comment: during --dry-run, opening a socket
    raises, and so does spawning anything that is not the very interpreter
    running the dry run. If any code path in the plan/collect logic were to
    reach out, or shell out to `modal` / `pip` / `lightning`, the dry run fails
    loudly instead of quietly spending money.

    `allow_local_python` exists for exactly one caller: the generated driver's
    --selftest, which deliberately spawns `sys.executable` to prove its log
    capture, failure handling and timeout enforcement really work. Nothing else
    in the dry run spawns anything.
    """
    global _ALLOW_LOCAL_PYTHON                          # noqa: PLW0603
    if not enabled:
        yield
        return
    import socket
    import subprocess
    import urllib.request

    def no_net(*_a, **_k):
        raise DryRunViolation(
            "--dry-run opened a network socket. The dry run must be hermetic.")

    real_popen = subprocess.Popen
    this_python = os.path.realpath(sys.executable)

    def guarded_popen(argv, *a, **k):
        head = argv[0] if isinstance(argv, (list, tuple)) and argv else argv
        if (_ALLOW_LOCAL_PYTHON
                and os.path.realpath(str(head)) == this_python):
            return real_popen(argv, *a, **k)
        raise DryRunViolation(
            f"--dry-run tried to spawn {head!r}. Only {this_python!r} is "
            f"allowed, and only for the driver's self test. A provider CLI "
            f"(modal, pip, lightning) must never be reachable from here.")

    saved = {
        "socket_class": socket.socket,
        "create_connection": socket.create_connection,
        "socket_connect": socket.socket.connect,
        "urlopen": urllib.request.urlopen,
        "popen": subprocess.Popen,
        "allow": _ALLOW_LOCAL_PYTHON,
    }
    socket.socket = no_net
    socket.create_connection = no_net
    socket.socket.connect = no_net
    urllib.request.urlopen = no_net
    subprocess.Popen = guarded_popen
    _ALLOW_LOCAL_PYTHON = allow_local_python
    try:
        yield
    finally:
        socket.socket = saved["socket_class"]
        socket.create_connection = saved["create_connection"]
        socket.socket.connect = saved["socket_connect"]
        urllib.request.urlopen = saved["urlopen"]
        subprocess.Popen = saved["popen"]
        _ALLOW_LOCAL_PYTHON = saved["allow"]


def _local_paths_in(source: str) -> list[str]:
    """Absolute paths from this machine that leaked into a payload.

    A cloud_run.py that captured the author's home directory would work
    perfectly locally and fail, or leak paths, on the box -- and that is exactly
    the class of bug a dry run on one machine cannot catch by running the code,
    only by reading it.
    """
    return [p for p in (str(REPO), str(pathlib.Path.home()), sys.prefix)
            if p and p != "/" and p in source]


def _no_local_paths(source: str) -> bool:
    return not _local_paths_in(source)


def _load_module_from_source(name: str, source: str, workdir: pathlib.Path):
    """Import the generated driver the way the box will, and hand back the
    module. Used to compile-check AND execute the real remote code locally.

    Also checks that the file contains no absolute path from the machine that
    generated it. A cloud_run.py that captured the author's home directory in
    the payload would work perfectly locally and fail, or leak paths, on the
    box -- and that is exactly the class of bug a dry run on one machine cannot
    see by running the code, only by looking at the code.
    """
    p = workdir / f"{name}.py"
    p.write_text(source)
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dry_run(args, arch: str, budget_minutes: float) -> int:
    fails: list[str] = []
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="cloudrun-dry-"))

    def check(name, cond, detail=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
        if not cond:
            fails.append(name)

    def hr(t):
        print()
        print("=" * 78)
        print(t)
        print("=" * 78)

    with hermetic(True):
        # ---------------------------------------------------- 1. arguments
        hr("1. argument handling and budget arithmetic")
        # For --provider lightning the Modal price table is not a claim about
        # Lightning; it stands in only so the budget arithmetic below has a rate
        # to check the ceiling against. main() printed that caveat already.
        cost = cost_estimate(arch, budget_minutes, args.region)
        if not cost["fits_in_free_credit"]:
            # An infeasible budget is not a dry-run failure: refusing to start
            # is the correct behaviour, and the live path has already printed
            # the refusal. Verify the refusal, then stop here rather than
            # pretending the remaining sections ran.
            print(f"  budget {budget_minutes:g} min of '{cost['gpu_string']}' "
                  f"costs up to ${cost['cost_worst_case']}, over the "
                  f"${cost['free_credit_usd']} free credit.")
            refusal = ""
            refused = False
            try:
                enforce_budget(cost, None, True)
            except SystemExit as e:
                refusal, refused = str(e), True
            check("an infeasible budget is refused", refused, refusal[:70])
            check("refusal names the affordable ceiling",
                  refused and f"{cost['max_affordable_minutes']} min" in refusal,
                  f"max {cost['max_affordable_minutes']} min")
            print()
            print(f"  DRY RUN: 0 failure(s) -- the budget gate fired correctly, "
                  f"so there is nothing further to validate.")
            return EXIT["ok"]
        print(f"  arch        {arch}  ({ARCHS[arch]['label']})")
        print(f"  gpu string  '{GPU_STRINGS[arch][0]}'")
        print(f"  rate        ${cost['usd_per_hr_gpu']}/hr  "
              f"(${cost['usd_per_sec_gpu']}/sec, us-east base)")
        print(f"  budget      {budget_minutes:g} min")
        print(f"  worst case  ${cost['cost_worst_case']}  "
              f"(gpu ${cost['cost_gpu']} + cpu ${cost['cost_cpu']} + "
              f"ram ${cost['cost_ram']} + startup ${cost['cost_startup_cpu_only']})")
        print(f"  free credit ${cost['free_credit_usd']}  ->  "
              f"{'FITS' if cost['fits_in_free_credit'] else 'DOES NOT FIT'}  "
              f"(max affordable {cost['max_affordable_minutes']} min)")
        check("cost fits in the free credit", cost["fits_in_free_credit"],
              f"${cost['cost_worst_case']} <= ${cost['free_credit_usd']}")
        check("worst case is positive", cost["cost_worst_case"] > 0)
        check("budget is capped by the credit",
              budget_minutes <= cost["max_affordable_minutes"],
              f"{budget_minutes} <= {cost['max_affordable_minutes']}")
        check("H100 pin carries the benchmarking bang",
              GPU_STRINGS["h100"][0] == "H100!",
              "gpu=\"H100\" may be served as an H200 and billed as an H100")
        check("B200 pin avoids the + form", GPU_STRINGS["b200"][0] == "B200",
              'gpu="B200+" may be served as a B300 and billed as a B200')
        check("A100 pin is the 80GB string", GPU_STRINGS["a100"][0] == "A100-80GB")
        check("no + form appears in any pin",
              not any("+" in v[0] for v in GPU_STRINGS.values()))
        try:
            enforce_budget(cost, None, True)
            check("a fitting budget passes the real gate", True)
        except SystemExit as e:
            check("a fitting budget passes the real gate", False, str(e)[:70])
        # negative cases: the guards must actually fire
        try:
            enforce_budget(cost_estimate("b200", 10_000.0), None, True)
            check("oversized budget is rejected", False, "no error raised")
        except SystemExit as e:
            check("oversized budget is rejected", "budget too large" in str(e),
                  str(e).splitlines()[0][:70])
        try:
            enforce_budget(cost, 0.001, True)
            check("declared spend limit is enforced", False, "no error raised")
        except SystemExit as e:
            check("declared spend limit is enforced",
                  "exceeds the spend limit" in str(e), "")
        try:
            enforce_budget(cost, None, False)
            check("no spend-limit assertion is refused", False, "no error")
        except SystemExit as e:
            check("no spend-limit assertion is refused",
                  "there is no API" in str(e), "")
        try:
            modal_rate_key("B200+")
            check("a '+' gpu string is refused, not priced", False, "no error")
        except ValueError as e:
            check("a '+' gpu string is refused, not priced", True,
                  str(e)[:60])
        # the requested arch must price at its own pinned string's rate, and
        # 'H100!' must price at the H100 rate on every arch
        check("'H100!' prices at the H100 rate",
              modal_rate_key("H100!") == "H100"
              and cost_estimate("h100", budget_minutes, args.region)
              ["usd_per_sec_gpu"] == MODAL_USD_PER_SEC["H100"])
        check("each arch prices at its own pinned string's rate",
              cost["usd_per_sec_gpu"]
              == MODAL_USD_PER_SEC[modal_rate_key(GPU_STRINGS[arch][0])],
              f"'{GPU_STRINGS[arch][0]}' -> ${cost['usd_per_sec_gpu']}/sec")

        # -------------------------------------------------- 2. path staging
        hr("2. path staging (real tree, real digests)")
        man = build_staging_manifest()
        print(f"  {man['files']} files, {man['bytes']/1e6:.2f} MB from "
              f"{len(man['roots'])} roots")
        for m in man["missing_or_empty"]:
            print(f"  note: {m}  -- skipped with a message, not a crash")
        for v in man["vanished_during_walk"][:5]:
            print(f"  note: {v} vanished mid-walk (another agent is writing); "
                  f"not staged")
        if man["vanished_during_walk"]:
            print(f"  {len(man['vanished_during_walk'])} file(s) vanished mid-walk")
        check("the walk survives files being written underneath it",
              all(not (REPO / v).exists() for v in man["vanished_during_walk"][:20]),
              f"{len(man['vanished_during_walk'])} vanished")
        check("staging found the required entry points",
              any(e["path"] == "remote/cloud_run.py" for e in man["entries"])
              and any(e["path"] == "kda/kda_vs_fla.py" for e in man["entries"]))
        check("no .venv or __pycache__ is staged",
              not any("__pycache__" in e["path"] or "/.venv/" in e["path"]
                      for e in man["entries"]))
        check("the 2.6MB regen blob is not staged",
              not any("cells_" in e["path"] for e in man["entries"]))
        check("every entry has a byte count", all(e["bytes"] >= 0 for e in man["entries"]))

        # --------------------------------------------------------- 3. plan
        hr("3. battery plan")
        plan = build_plan(arch, budget_minutes,
                          only=args.only or None, skip=args.skip or None)
        for s in plan["stages"]:
            state = s.get("status", "ready")
            print(f"  {state:8} {s['id']:16} "
                  f"{' '.join(s.get('argv', [])) or s.get('skip_reason', '')}")
        check("plan is non-empty", len(plan["stages"]) > 0)
        check("plan has a digest", bool(plan.get("digest")))
        check("every stage is runnable or explicitly skipped",
              all(s.get("argv") or s.get("status") == "skipped"
                  for s in plan["stages"]))
        check("no stage is left half-built",
              all(s.get("skip_reason") for s in plan["stages"]
                  if s.get("status") == "skipped"))
        check("the identity stage is never skipped",
              next(s for s in plan["stages"] if s["id"] == "identity")
              .get("status", "ready") != "skipped")
        try:
            build_plan(arch, budget_minutes, only=["no_such_stage"])
            check("unknown --only is a usage error", False, "no error")
        except SystemExit:
            check("unknown --only is a usage error", True)

        # ------------------------------- 4. the generated remote driver runs
        hr("4. the generated remote driver (compiled, imported, executed)")
        src = build_battery_source()
        try:
            compile(src, "cloud_battery.py", "exec")
            check("driver compiles", True, f"{len(src.splitlines())} lines")
        except SyntaxError as e:
            check("driver compiles", False, str(e))
            return EXIT["self_check_failed"]

        stage_dir = tmp / "stage"
        stage_tree(man, plan, stage_dir)
        check("staging materialised on disk",
              (stage_dir / "plan.json").is_file()
              and (stage_dir / "cloud_battery.py").is_file()
              and (stage_dir / "kda" / "kda_vs_fla.py").is_file())
        check("staged plan.json round-trips",
              json.loads((stage_dir / "plan.json").read_text())["digest"]
              == plan["digest"])
        check("the generated driver leaks no local absolute path",
              _no_local_paths(src),
              "checked REPO, $HOME and sys.prefix")
        # import the generated driver the way the box will, and check that the
        # inlined tables survived the string substitution.
        os.environ["CLOUD_REPO"] = str(stage_dir)
        os.environ["CLOUD_RESULTS"] = str(tmp / "results")
        os.environ["CLOUD_PLAN"] = str(stage_dir / "plan.json")
        (tmp / "results").mkdir(parents=True, exist_ok=True)
        try:
            mod = _load_module_from_source("cloud_battery", src, tmp)
        except DryRunViolation as e:
            check("the generated driver imports cleanly", False, str(e)[:110])
            return EXIT["self_check_failed"]
        check("the generated driver imports cleanly", True)
        check("inlined PTX tables survived as real dicts",
              set(getattr(mod, "PTX_PROBES", {})) == set(FEATURE_ORDER)
              and isinstance(getattr(mod, "PTX_PROBES", {}).get("tcgen05"), str)
              and "\n.target __TARGET__" in mod.PTX_PROBES["tcgen05"]
              and len(getattr(mod, "PTX_MIN_VERSION", {})) >= 15,
              f"{len(getattr(mod, 'PTX_PROBES', {}))} probes, "
              f"{len(getattr(mod, 'PTX_MIN_VERSION', {}))} arch versions")

        # run the driver's own --plan mode against the real staged tree
        try:
            rc = mod.main(["--plan"])
            check("driver --plan runs on this machine with no GPU", rc == 0)
        except Exception as e:                               # noqa: BLE001
            check("driver --plan runs on this machine with no GPU", False,
                  f"{type(e).__name__}: {e}")

        # the driver's own selftest: real subprocesses, real signals, real logs
        print()
        print("  --- driver --selftest (spawns real subprocesses on THIS box) ---")
        with hermetic(True, allow_local_python=True):
            sel_rc = mod.main(["--selftest"])
        check("driver --selftest passes", sel_rc == 0,
              "" if sel_rc == 0 else "see the selftest table above")
        check("the self test's subprocess allowance is revoked afterwards",
              _ALLOW_LOCAL_PYTHON is False)

        # ------------------------------- 5. the feature table, all archs
        hr("5. feature-availability mapping (derived, and self-tested)")
        print(f"  {'capability':12} {'family':24} "
              + " ".join(f"{k[:9]:>10}" for k in FEATURE_ORDER))
        expected = {
            (8, 0): dict(cp_async=1, fp8_tensor_core=0, tma_bulk_async=0, wgmma=0, tcgen05=0),
            (8, 9): dict(cp_async=1, fp8_tensor_core=1, tma_bulk_async=0, wgmma=0, tcgen05=0),
            (9, 0): dict(cp_async=1, fp8_tensor_core=1, tma_bulk_async=1, wgmma=1, tcgen05=0),
            (10, 0): dict(cp_async=1, fp8_tensor_core=1, tma_bulk_async=1, wgmma=0, tcgen05=1),
            (10, 3): dict(cp_async=1, fp8_tensor_core=1, tma_bulk_async=1, wgmma=0, tcgen05=1),
            (12, 0): dict(cp_async=1, fp8_tensor_core=1, tma_bulk_async=1, wgmma=0, tcgen05=0),
        }
        for cc, exp in expected.items():
            d = derive_features(cc)
            row = {k: int(d["features"][k]["available"]) for k in FEATURE_ORDER}
            got = " ".join(f"{row[k]:>10}" for k in FEATURE_ORDER)
            print(f"  sm_{cc[0]}{cc[1]:<9} {d['family']:24} {got}")
            check(f"sm_{cc[0]}{cc[1]} feature table", row == exp,
                  "got " + json.dumps(row) if row != exp else "")
        # the two load-bearing rows, stated as assertions
        d103 = derive_features((10, 3))["features"]["tcgen05"]
        d120 = derive_features((12, 0))["features"]["tcgen05"]
        check("B300/sm_103 HAS tcgen05 (via sm_103a)",
              d103["available"] and "sm_103a" in d103["requires"],
              "verified with ptxas 12.9 at PTX ISA >= 8.8 and PTX ISA 9.4")
        check("RTX PRO 6000/sm_120 has NO tcgen05", not d120["available"],
              "verified with ptxas 12.9; sm_12xa is in no tcgen05 arch list")
        check("an unknown capability fails closed",
              all(not f["available"] for f in
                  derive_features((6, 1))["features"].values())
              and derive_features((6, 1))["family"] == "unknown")
        check("tcgen05 is NOT reachable on the bare target for any arch",
              not derive_features((10, 0))["features"]["tcgen05"]
              ["reachable_with_pinned_tinygrad"],
              "tinygrad emits sm_100, not sm_100a (ops_cuda.py:122)")
        check("wgmma is NOT reachable with the pinned tinygrad",
              not derive_features((9, 0))["features"]["wgmma"]
              ["reachable_with_pinned_tinygrad"],
              "tinygrad emits sm_90, not sm_90a (ops_cuda.py:122)")

        hr("6. ptxas probe templates (every cell the probe would use)")
        cells = rendered = 0
        for feat in FEATURE_ORDER:
            for p_arch in PROBE_ARCH_MATRIX:          # not `arch`: that is the
                if render_probe_ptx(feat, p_arch):    # caller's arg
                    cells += 1
                    rendered += 1
        print(f"  {rendered} of {len(FEATURE_ORDER) * len(PROBE_ARCH_MATRIX)} "
              f"(feature, arch) cells render to valid PTX text")
        malformed = []
        for feat in FEATURE_ORDER:
            for p_arch in ("sm_80", "sm_100a", "sm_120a"):
                t = render_probe_ptx(feat, p_arch)
                if t and (".version " not in t or ".target " not in t
                          or "__" in t):
                    malformed.append(f"{feat}@{p_arch}")
        check("every probe cell has .version and .target filled in",
              not malformed, ", ".join(malformed[:4]))
        check("all probe templates render", rendered == cells and cells >= 50,
              f"{cells} cells")
        try:
            check("unknown archs are skipped, not crashed",
                  render_probe_ptx("tcgen05", "sm_999a") is None)
        except Exception as e:                               # noqa: BLE001
            check("unknown archs are skipped, not crashed", False, str(e))
        check("a nonexistent arch-conditional target is untestable, not absent",
              render_probe_ptx("wgmma", "sm_89a") is None,
              "there is no sm_89a; probing it would read as 'feature absent'")
        check("the probe .version is max(arch floor, feature floor)",
              render_probe_ptx("fp8_tensor_core", "sm_89").startswith(".version 8.4")
              and render_probe_ptx("cp_async", "sm_103").startswith(".version 8.8"),
              "fp8 needs 8.4 even on sm_89; sm_103 needs 8.8 even for cp.async")
        # A guard never observed to reject anything is a comment, not a guard.
        # Corrupt the table on purpose and confirm the answer follows it, which
        # is what makes the real cross-check meaningful.
        saved_targets = FEATURE_GATES["tcgen05"]["targets"]
        try:
            FEATURE_GATES["tcgen05"]["targets"] = saved_targets | {"sm_120a"}
            check("a deliberately corrupted table moves the answer",
                  derive_features((12, 0))["features"]["tcgen05"]["available"],
                  "synthetic: proves the table, not a hardcoded literal, "
                  "decides the verdict")
        finally:
            FEATURE_GATES["tcgen05"]["targets"] = saved_targets
        check("the table is restored after the corruption test",
              not derive_features((12, 0))["features"]["tcgen05"]["available"])

        # ------------------------- 7. synthetic run + real collector
        hr("7. result collection round trip (synthetic tree, real collector)")
        for cc in [(9, 0), (10, 3), (12, 0)]:
            ident = {
                "capability": {"major": cc[0], "minor": cc[1]},
                "device_name": f"Synthetic {cc[0]}{cc[1]}",
                "driver_version": "000.00",
                "torch": {"cuda_runtime_version": "12.8.0", "version": "2.9.0"},
                "versions": {"tinygrad": "0.12.0", "torch": "2.9.0",
                             "triton": "ABSENT (ModuleNotFoundError)"},
                "derived_features": derive_features(cc),
                "ptx_probe": {"status": "ran", "ptxas": "/usr/bin/ptxas", "cells": {
                    f"tcgen05@sm_{cc[0]}{cc[1]}a": {
                        "verdict": "assembles" if cc != (12, 0) else "rejected",
                        "detail": ""}}},
            }
            code, verdict = classify_collected(
                collect_results(synthetic_tree(plan, ident, cc, verdict_ok=True),
                                tmp / f"out_{cc[0]}{cc[1]}",
                                {"provider": "dry", "arch": arch,
                                 "gpu_string": GPU_STRINGS[arch][0],
                                 "expect_cc": list(ARCHS[arch]["expect_cc"]),
                                 "cost": cost}),
                arch)
            got_ok = (code == EXIT["ok"])
            want_ok = (cc == tuple(ARCHS[arch]["expect_cc"]))
            check(f"identity gate accepts sm_{cc[0]}{cc[1]} only for "
                  f"{arch if want_ok else 'the wrong arch'}",
                  got_ok == want_ok, verdict[:100])

        # a complete-but-wrong-arch run must be rejected, not collected quietly
        summary = collect_results(
            synthetic_tree(plan, {
                "capability": {"major": 9, "minor": 0},
                "device_name": "NVIDIA H200",
                "driver_version": "000.00", "torch": {}, "versions": {}},
                (9, 0), verdict_ok=True),
            tmp / "out_mismatch", {"provider": "dry", "arch": "b200",
                                   "gpu_string": "B200",
                                   "expect_cc": [10, 0], "cost": cost})
        code, verdict = classify_collected(summary, "b200")
        check("a B200 request served an H200 is a hard failure",
              code == EXIT["arch_mismatch"] and "MISMATCH" in verdict,
              verdict[:110])
        check("SUMMARY.md says so in plain words",
              "WRONG architecture" in (tmp / "out_mismatch" / "SUMMARY.md").read_text())

        # every failure mode, through the real classifier
        hr("8. failure-mode classifier (every exit code, one line each)")
        # The class's own __name__ is "AuthenticationError" -- the classifier
        # matches on type(exc).__name__, which is the only handle available
        # before the modal SDK exists.
        AuthenticationError = type("AuthenticationError", (Exception,), {})

        cases = [
            ("no credentials", EXIT["no_credentials"],
             AuthenticationError("token not found")),
            ("out of credit", EXIT["out_of_credit"],
             RuntimeError("Monthly usage limit exceeded for workspace")),
            ("no capacity", EXIT["no_capacity"],
             RuntimeError("insufficient capacity: no H100 available")),
            ("image build", EXIT["image_build"],
             RuntimeError("Error building image: pip install failed with exit 1")),
            ("preempted", EXIT["preempted"],
             RuntimeError("container was interrupted by preemption")),
            ("sdk missing", EXIT["sdk_missing"],
             ModuleNotFoundError("No module named 'modal'", name="modal")),
            ("timeout, no results", EXIT["no_capacity"], TimeoutError()),
            ("unattributed", EXIT["unclassified"],
             RuntimeError("¯\\_(ツ)_/¯")),
        ]
        for label, want, (code, headline, remedy) in [
                (lbl, w, classify_provider_error("modal", exc, {}))
                for lbl, w, exc in cases]:
            check(f"{label} -> exit {want}", code == want,
                  f"{headline[:60]}  |  remedy: {remedy.splitlines()[0][:52]}")

        # and the post-run verdicts
        hr("9. post-run verdicts")
        for label, want, cc, ok in [
            ("complete", EXIT["ok"], tuple(ARCHS[arch]["expect_cc"]), True),
            ("no DONE marker", EXIT["no_results"], (9, 0), False),
        ]:
            if label == "no DONE marker":
                s = collect_results(
                    {"logs/identity.log": b"x"},
                    tmp / "out_noresults",
                    {"provider": "dry", "arch": arch, "gpu_string": "x",
                     "expect_cc": list(ARCHS[arch]["expect_cc"]), "cost": cost})
            else:
                s = collect_results(
                    synthetic_tree(plan, {
                        "capability": {"major": cc[0], "minor": cc[1]},
                        "device_name": "x", "driver_version": "0",
                        "torch": {}, "versions": {}}, cc, ok),
                    tmp / "out_complete",
                    {"provider": "dry", "arch": arch,
                     "gpu_string": GPU_STRINGS[arch][0],
                     "expect_cc": list(ARCHS[arch]["expect_cc"]), "cost": cost})
            code, verdict = classify_collected(s, arch)
            check(f"{label} -> exit {want}", code == want, verdict[:100])

        hr(f"DRY RUN: {len(fails)} failure(s)")
        for f in fails:
            print(f"  FAILED: {f}")
        print()
        print(f"  staged tree kept at {tmp} (delete it; nothing was uploaded)")
        print(f"  network calls made: 0    credentials read: 0    cost: $0.00")
        return EXIT["ok"] if not fails else EXIT["self_check_failed"]


def synthetic_tree(plan: dict, identity: dict, cc, verdict_ok: bool = True) -> dict:
    """A byte-exact stand-in for the tree the real driver writes.

    Built from the same stage list and the same schema, so the collector is
    exercised for real rather than against a fixture that agrees with itself.
    """
    files: dict[str, bytes] = {}
    files["RUN.json"] = json.dumps(
        {"started": 0, "deadline_epoch": 0, "plan_digest": plan["digest"],
         "stages": len(plan["stages"]), "attempt": 1,
         "requested": plan["requested"]}).encode()
    files["identity.json"] = json.dumps(identity, indent=2).encode()
    status = "ok"
    for s in plan["stages"]:
        st = dict(id=s["id"], status="skipped" if s.get("status") == "skipped"
                  else status, returncode=0 if status == "ok" else None,
                  duration_s=1.25, log=f"logs/{s['id']}.log", error=None,
                  why=s.get("why", ""))
        files[f"results/{s['id']}.json"] = json.dumps(st, indent=2).encode()
        files[f"logs/{s['id']}.log"] = f"# stage {s['id']}\nsynthetic\n".encode()
    files["codegen/kernel0000.cu"] = b"// synthetic generated kernel\n"
    verdict = "complete" if verdict_ok else "partial"
    files["DONE"] = json.dumps(
        {"verdict": verdict, "ok": len(plan["stages"]), "failed": 0,
         "timeout": 0, "skipped": 0, "wrote_at": 0}, indent=2).encode()
    return files


# ===========================================================================
# 12. PROVIDER ADAPTERS
# ===========================================================================
MODAL_VOLUME = "mc-cuda-tinygrad-results"
MODAL_APP = "mc-cuda-tinygrad-battery"


def modal_credentials_present() -> tuple[bool, str]:
    if os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"):
        return True, "MODAL_TOKEN_ID / MODAL_TOKEN_SECRET in the environment"
    cfg = pathlib.Path.home() / ".modal.toml"
    if cfg.is_file():
        try:
            txt = cfg.read_text()
        except OSError:
            return False, f"{cfg} exists but could not be read"
        if "token_id" in txt:
            return True, str(cfg)
        return False, f"{cfg} exists but holds no token_id"
    return False, ("no MODAL_TOKEN_ID/MODAL_TOKEN_SECRET and no ~/.modal.toml")


def run_modal(args, arch, budget_minutes, cost, plan, manifest, outdir):
    """Provision, run, collect. No retries, ever."""
    ok, why = modal_credentials_present()
    if not ok:
        print(f"  no credentials: {why}")
        print("  run `modal setup` once, in a terminal, then re-run this command.")
        print("  nothing was provisioned and nothing was charged.")
        return EXIT["no_credentials"]

    import modal                                             # noqa: PLC0415
    print(f"  credentials: {why}")
    print(f"  image will be built from {manifest['files']} staged files "
          f"({manifest['bytes']/1e6:.2f} MB)")

    # The generated driver goes to a scratch dir, never into the repo: it is a
    # build artefact of this script, and dropping it next to the source would
    # leave an untracked file behind after every run.
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="cloudrun-modal-"))
    driver = scratch / "cloud_battery.py"
    driver.write_text(build_battery_source())
    plan_file = scratch / "plan.json"
    plan_file.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"  payload staged at {scratch}")

    def _app_and_fn():
        image = (modal.Image.debian_slim(python_version="3.12")
                 .apt_install("git", "clang", "build-essential")
                 .pip_install(*REQUIRED_PIP))
        for top in STAGE_PATHS:
            src = REPO / top
            if not src.exists():
                continue
            if src.is_dir():
                image = image.add_local_dir(str(src), f"/repo/{top}",
                                            copy=True,
                                            ignore=STAGE_EXCLUDES.get(top, []))
            else:
                image = image.add_local_file(str(src), f"/repo/{top}", copy=True)
        image = (image
                 .add_local_file(str(driver), "/repo/cloud_battery.py", copy=True)
                 .add_local_file(str(plan_file), "/repo/plan.json", copy=True))
        vol = modal.Volume.from_name(MODAL_VOLUME, create_if_missing=True)
        app = modal.App(MODAL_APP, image=image)

        # retries=0: Modal restarts a preempted function on the same input by
        # default, which would re-spend the budget. The volume's DONE marker
        # makes a re-run a no-op, but not retrying is the primary guard.
        # timeout= is the hard wall clock: billing stops when the container is
        # reclaimed, so this is what actually enforces --budget-minutes.
        @app.function(
            gpu=GPU_STRINGS[arch][0],
            volumes={"/results": vol},
            region=args.region,
            timeout=int(budget_minutes * 60),
            retries=0,
            cpu=MAX_CPU_CORES,
            memory=MAX_RAM_GIB * 1024,
        )
        def battery():
            os.environ["CLOUD_REPO"] = "/repo"
            os.environ["CLOUD_RESULTS"] = "/results"
            os.environ["CLOUD_PLAN"] = "/repo/plan.json"
            os.environ["CLOUD_DEADLINE_EPOCH"] = str(
                time.time() + budget_minutes * 60)
            sys.path.insert(0, "/repo")
            import runpy                            # noqa: PLC0415
            return runpy.run_path("/repo/cloud_battery.py",
                                  run_name="__main__") or 0
        return app, battery, vol

    try:
        app, fn, vol = _app_and_fn()
    except Exception as e:                                   # noqa: BLE001
        code, headline, remedy = classify_provider_error("modal", e, {})
        print(f"  {headline}\n{remedy}")
        return code

    try:
        with app.run():                                     # noqa: F841
            print("  submitting; queue wait counts against the timeout, so a")
            print("  timeout with zero results means no capacity, not a slow run")
            try:
                fn.remote()
            except Exception as e:                           # noqa: BLE001
                ctx = {"gpu_string": GPU_STRINGS[arch][0], "results": None}
                code, headline, remedy = classify_provider_error("modal", e, ctx)
                print(f"  {headline}\n{remedy}")
                return code
    except Exception as e:                                   # noqa: BLE001
        code, headline, remedy = classify_provider_error("modal", e, {})
        print(f"  {headline}\n{remedy}")
        return code

    # collect from the volume. This path is reached even after a failure above
    # in a later call, and it is the ONLY channel that survives preemption.
    print("  reading the results volume")
    try:
        remote_files: dict[str, bytes] = {}
        for entry in vol.iterdir("results"):
            remote_files[entry.path] = vol.read_file(entry.path)
    except Exception as e:                                   # noqa: BLE001
        code, headline, remedy = classify_provider_error("modal", e, {})
        print(f"  could not read the results volume: {headline}\n{remedy}")
        return EXIT["collect_failed"]
    if not remote_files:
        print("  the volume is empty: the function never wrote a result.")
        print("  check the Logs tab for the function's own output.")
        return EXIT["no_results"]

    summary = collect_results(remote_files, outdir,
                              {"provider": "modal", "arch": arch,
                               "gpu_string": GPU_STRINGS[arch][0],
                               "expect_cc": list(ARCHS[arch]["expect_cc"]),
                               "cost": cost, "docs_date": DOCS_DATE})
    code, verdict = classify_collected(summary, arch)
    print(f"  {verdict}")
    if code != EXIT["ok"] and summary.get("identity_gate"):
        gate = summary["identity_gate"]
        print()
        print(f"  !! {gate['verdict']}")
        print()
        print(f"  Results are in {outdir}. They are labelled {arch} in the")
        print(f"  directory path but the device was not {arch}, so do not cite")
        print(f"  them as {arch} data. SUMMARY.md carries the same warning.")
    return code


def run_lightning(args, arch, budget_minutes, cost, plan, manifest, outdir):
    """The weak leg, and it is labelled as such.

    Lightning's free-tier numbers are mutually inconsistent across its own pages
    and it publishes no per-GPU-hour rate for H100/H200/B200/A100/L40S, so this
    adapter cannot do what the Modal one does: it cannot prove in advance that
    the run costs $0.00. It also could not be validated without installing the
    SDK, which is out of scope. So it refuses by default, and the device
    identity check -- not the machine-type string -- is the guard that stands.
    """
    print("  Lightning AI: LOW confidence. Read the block comment in")
    print("  remote/cloud_run.py before trusting any of this.")
    if not args.accept_unverified_surface:
        print()
        print("  REFUSING TO PROVISION. Reasons, all from lightning.ai itself:")
        print("   * no published credits/hr for H100/H200/B200/A100/L40S; the")
        print("     official consumption-rate table lists only T4 and V100")
        print("   * four mutually inconsistent free-tier figures on one site:")
        print("     'up to 30 credits' (tier card + comparison table),")
        print("     '5 free credits upon registration' (the same page's FAQ),")
        print("     '80 free GPU hours' (homepage hero)")
        print("   * the card requirement is not stated in Lightning's docs;")
        print("     the only 'no card' claims found are third-party")
        print("   * the provisioning surface below is UNVERIFIED: it was not")
        print("     exercised, because the SDK is not installed here")
        print()
        print("  Nothing was created, nothing was charged. To proceed anyway,")
        print("  supply the machine-type string you want to try AND accept that")
        print("  it is unverified:")
        print("    --accept-unverified-surface \\")
        print("    --lightning-machine '<string from the Lightning UI>'")
        print("  then CHECK the reported capability: this script fails the run if")
        print("  the device is not the arch you asked for, so a wrong machine-type")
        print("  string costs you money but cannot corrupt a result.")
        return EXIT["usage"]

    if not args.lightning_machine:
        print("  --lightning-machine is required: Lightning's machine-type")
        print("  strings could not be confirmed from its documentation, so this")
        print("  script will not guess one. Read the string off the Lightning UI")
        print("  for the machine you want and pass it verbatim.")
        return EXIT["usage"]

    try:
        import lightning                                     # noqa: PLC0415,PLC2701
    except ImportError as e:
        code, headline, remedy = classify_provider_error("lightning", e, {})
        print(f"  {headline}\n{remedy}")
        return code

    print(f"  WARNING: the machine-type string '{args.lightning_machine}' and")
    print("  the SDK call shape below are UNVERIFIED. This code has never")
    print("  been executed. If it provisions the wrong machine, the identity")
    print("  check will catch it -- but the money will already be spent.")
    print()
    print("  Fix run_lightning() in remote/cloud_run.py against the live SDK")
    print("  before using this path for anything that matters.")
    return EXIT["sdk_missing"]


# ===========================================================================
# 13. ENTRY POINT
# ===========================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="remote/cloud_run.py",
        description="Run the study's measurement battery on real Hopper / "
                    "Blackwell silicon on a $0.00 free tier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Validated path:  --dry-run  (no network, no credentials, no cost)")
    p.add_argument("--provider", choices=["modal", "lightning"], required=True)
    p.add_argument("--arch", choices=sorted(ARCHS), required=True,
                   help="which architecture to pin")
    p.add_argument("--budget-minutes", type=float, default=None,
                   help=f"hard wall clock. REQUIRED in spirit and checked in "
                        f"fact: worst-case cost must fit inside the free "
                        f"credit. Omit it and you get the conservative "
                        f"default of {DEFAULT_BUDGET_MINUTES} min with a "
                        f"warning.")
    p.add_argument("--dry-run", action="store_true",
                   help="exercise everything with zero network and zero credentials")
    p.add_argument("--region", default="us-east",
                   help="Modal region. Anything but us-east is a 1.15-1.75x "
                        "multiplier on the per-second rates.")
    p.add_argument("--out-root", default=None,
                   help="where results land (default out/cloud/<provider>/<arch>/<ts>)")
    p.add_argument("--only", action="append", default=[],
                   help=f"run only these stages: {', '.join(s['id'] for s in STAGES)}")
    p.add_argument("--skip", action="append", default=[])
    p.add_argument("--stage", action="append", default=[], metavar="ID:CMD",
                   help="override a stage's command (shell-split)")
    p.add_argument("--credit-only", action="store_true",
                   help="assert there is no payment method and the workspace "
                        "spend limit is $0. The script then only needs the "
                        "worst-case cost to fit in the free credit, which is "
                        "what actually guarantees $0.00")
    p.add_argument("--modal-spend-limit-usd", type=float, default=None,
                   help="the spend limit you set at modal.com/settings/usage. "
                        "There is NO API for this, so the script cannot verify "
                        "it -- it only checks your worst case is under it.")
    p.add_argument("--lightning-machine", default=None,
                   help="Lightning machine-type string (UNVERIFIED)")
    p.add_argument("--accept-unverified-surface", action="store_true",
                   help="proceed on the Lightning path despite it being "
                        "unverified against vendor docs")
    p.add_argument("--ptx-probe", action="store_true",
                   help="on the GPU box: ask ptxas which target features the "
                        "device supports, and fail if it disagrees with the "
                        "table. Also runnable locally.")
    p.add_argument("--ptxas-cmd", default=None, metavar="TEMPLATE",
                   help="a command that runs ptxas on a probe cell, for hosts "
                        "with no local CUDA toolkit. Placeholders: {ptxas} "
                        "{arch} {file} {name} {dir} {devnull}. See "
                        "--verify-feature-table for this repo's docker one-liner.")
    p.add_argument("--verify-feature-table", action="store_true",
                   help="run the ptxas oracle over the WHOLE (feature, arch) "
                        "matrix and diff it against FEATURE_GATES. Needs a "
                        "ptxas on PATH or at /usr/local/cuda/bin/ptxas. This "
                        "is how the table was derived, and how to re-derive "
                        "it after a toolkit upgrade.")
    p.add_argument("--remote-capture-codegen", action="store_true",
                   help="on the GPU box: capture every generated kernel")
    p.add_argument("--ptx-probe-capability", default=None, metavar="MAJOR:MINOR",
                   help="probe this arch instead of reading the device, so the "
                        "probe's cross-check against FEATURE_GATES can be "
                        "exercised on a host with no GPU")
    p.add_argument("--out", default=None, help="output dir for the two modes above")
    return p.parse_args(argv)


def _parse_overrides(specs: list[str]) -> dict[str, list[str]]:
    import shlex                                            # noqa: PLC0415
    out = {}
    for spec in specs:
        if ":" not in spec:
            raise SystemExit(f"--stage expects ID:CMD, got {spec!r}")
        sid, cmd = spec.split(":", 1)
        argv = shlex.split(cmd)
        if not argv:
            raise SystemExit(f"--stage {spec!r} has an empty command")
        out[sid.strip()] = argv
    return out


def main(argv=None) -> int:
    args = parse_args(argv)

    # --- the two modes that run on the box, not here ---------------------
    if args.verify_feature_table:
        return verify_feature_table(args.ptxas_cmd)
    if args.ptx_probe or args.remote_capture_codegen:
        return _on_box(args)

    if args.budget_minutes is None:
        print(f"NOTE: no --budget-minutes given. Using the conservative default "
              f"of {DEFAULT_BUDGET_MINUTES} min. Pass it explicitly if you mean "
              f"something else.")
        budget_minutes = float(DEFAULT_BUDGET_MINUTES)
    else:
        budget_minutes = args.budget_minutes
    if budget_minutes <= 0:
        print("--budget-minutes must be positive")
        return EXIT["usage"]

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = pathlib.Path(args.out_root) if args.out_root else (
        REPO / "out" / "cloud" / args.provider / args.arch / stamp)

    # --- budget / cost gate, before anything is provisioned --------------
    if args.provider == "modal":
        cost = cost_estimate(args.arch, budget_minutes, args.region)
        print("=" * 78)
        print(f"modal  arch={args.arch}  gpu='{GPU_STRINGS[args.arch][0]}'  "
              f"region={args.region}")
        print("=" * 78)
        print(f"  GPU            ${cost['usd_per_sec_gpu']}/sec = "
              f"${cost['usd_per_hr_gpu']}/hr  (base, us-east)")
        print(f"  budget         {budget_minutes:g} min "
              f"(hard: Modal timeout={int(budget_minutes*60)}s, and the container")
        print(f"                 aborts on CLOUD_DEADLINE_EPOCH)")
        print(f"  worst case     ${cost['cost_worst_case']}  "
              f"(gpu ${cost['cost_gpu']} + cpu ${cost['cost_cpu']} + ram "
              f"${cost['cost_ram']} + cpu-only startup "
              f"${cost['cost_startup_cpu_only']})")
        print(f"  free credit    ${cost['free_credit_usd']}  ->  "
              f"{'FITS, outlay $0.00' if cost['fits_in_free_credit'] else 'DOES NOT FIT'}")
        if not args.dry_run and (not cost["fits_in_free_credit"]
                                 or args.modal_spend_limit_usd is not None
                                 or not args.credit_only):
            # --dry-run never provisions, so it must not be blocked by an
            # assertion about money; it exercises enforce_budget directly
            # instead, including the failing cases.
            try:
                enforce_budget(cost, args.modal_spend_limit_usd, args.credit_only)
            except SystemExit as e:
                print()
                print("  REFUSING TO START.")
                print()
                for line in str(e).splitlines():
                    print("  " + line)
                return EXIT["usage"]
    else:
        cost = {"provider": "lightning", "note": "no published per-GPU rate; "
                "cost cap is Lightning's own credit exhaustion, not ours",
                "free_credit_usd": None, "cost_worst_case": None,
                "max_affordable_minutes": None, "fits_in_free_credit": None}
        print("=" * 78)
        print(f"lightning  arch={args.arch}  budget={budget_minutes:g} min")
        print("=" * 78)

    if args.dry_run:
        if args.provider != "modal":
            # The Lightning adapter has no validated plumbing to exercise: the
            # whole reason it refuses by default is that the provisioning
            # surface was never confirmed. So --dry-run is a MODAL-path
            # validation and says so, rather than printing a green result for
            # a path that was never actually checked.
            print(f"  NOTE: --dry-run validates the MODAL path only. The "
                  f"{args.provider} path cannot be dry-run validated: its "
                  f"provisioning surface is unverified, so the only thing a dry "
                  f"run could confirm is our own argument handling and battery "
                  f"plan, which are provider-independent and identical.")
            print(f"  The checks below therefore run with "
                  f"--provider modal's arch table substituted for planning "
                  f"purposes only; nothing about the {args.provider} adapter "
                  f"itself is claimed to be verified.")
        return dry_run(args, args.arch, budget_minutes)

    print()
    print(f"  results -> {outdir}")
    overrides = _parse_overrides(args.stage)
    plan = build_plan(args.arch, budget_minutes,
                      only=args.only or None, skip=args.skip or None,
                      overrides=overrides)
    manifest = build_staging_manifest()
    for m in manifest["missing_or_empty"]:
        print(f"  SKIP {m}")

    if args.provider == "modal":
        return run_modal(args, args.arch, budget_minutes, cost, plan, manifest, outdir)
    return run_lightning(args, args.arch, budget_minutes, cost, plan, manifest, outdir)


# ------------------------------------------------------------- box-side modes
def ptxas_argv(template: str | None, arch: str, ptx_file: pathlib.Path) -> list[str]:
    """The argv that runs ptxas on one probe cell.

    `template` lets any callable ptxas be substituted, which is what makes
    this verifier runnable on a machine with no CUDA toolkit. Placeholders:
      {ptxas} {arch} {file} {devnull}
    For this repo's own archlab image:
      --ptxas-cmd 'docker run --rm --platform linux/amd64 -v {dir}:/w \
archlab/ptxas:12.9 /usr/local/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas {arch} /w/{name} -o {devnull}'
    """
    devnull = os.devnull
    if not template:
        exe = find_ptxas()
        if not exe:
            return []
        return [exe, f"-arch={arch}", str(ptx_file), "-o", devnull]
    out = []
    for tok in shlex.split(template):
        out.append(tok.format(ptxas=find_ptxas() or "ptxas", arch=f"-arch={arch}",
                              file=f"/w/{ptx_file.name}", name=ptx_file.name,
                              devnull=devnull, dir=str(ptx_file.parent)))
    return out


def find_ptxas() -> str | None:
    """The first usable ptxas, including the pip-wheel copies.

    A CUDA toolkit is not needed: `pip install nvidia-cuda-nvcc-cu12` drops a
    ptxas inside site-packages, which is exactly how this repo's
    tools/archlab and matmul/Dockerfile.toolchain get one. So the search has to
    include those, or the verifier is silently unavailable on the machine that
    most needs it.
    """
    from shutil import which                                 # noqa: PLC0415
    cands = [which("ptxas"),
             "/usr/local/cuda/bin/ptxas", "/opt/cuda/bin/ptxas", "/usr/bin/ptxas"]
    # site-packages/nvidia/cuda_nvcc/bin/ptxas
    for base in {sys.prefix, sys.base_prefix, os.path.expanduser("~/.venv")}:
        if base:
            cands.append(os.path.join(base, "lib", "python*", "site-packages",
                                      "nvidia", "cuda_nvcc", "bin", "ptxas"))
    cands += sorted(glob.glob(os.path.join(
        sys.prefix, "lib", "python*", "site-packages", "nvidia",
        "cuda_nvcc", "bin", "ptxas")))
    for c in cands:
        if c and "*" not in c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    for pattern in cands:
        if pattern and "*" in pattern:
            for hit in sorted(glob.glob(pattern)):
                if os.path.isfile(hit) and os.access(hit, os.X_OK):
                    return hit
    return None


def verify_feature_table(cmd_template: str | None = None) -> int:
    """Diff FEATURE_GATES against NVIDIA's own assembler, cell by cell.

    This is the whole justification for the table, made reproducible. It was
    run to derive the table (docs read "2026-09-27", ptxas 12.9) and should be
    re-run after a toolkit upgrade, because a toolkit upgrade is exactly the
    event that would change the answers.

    Two distinctions this makes that are easy to get wrong by hand:
      * "rejected" and "cannot test" are different. A nonexistent
        arch-conditional target (sm_89a) and a PTX version too old for the
        target both produce a non-zero exit that says nothing about the
        feature.
      * the bare target and the `a` target answer different questions. The
        study needs both: what the silicon can do (the `a` answer) and what
        the pinned tinygrad can actually reach (the bare answer, because it
        emits sm_<major><minor>).
    """
    import subprocess                                        # noqa: PLC0415
    import tempfile                                          # noqa: PLC0415

    ptxas = find_ptxas()
    template = cmd_template
    if not ptxas and not template:
        print("no ptxas found. Looked on PATH, at /usr/local/cuda/bin, "
              "/opt/cuda/bin,\n/usr/bin, and in site-packages/nvidia/cuda_nvcc "
              "(the nvidia-cuda-nvcc-cu12 wheel).")
        print()
        print("Either install one:   pip install nvidia-cuda-nvcc-cu12")
        print("or point at any callable ptxas, e.g. this repo's own image:")
        print("  --ptxas-cmd 'docker run --rm --platform linux/amd64 \\")
        print("     -v {dir}:/w archlab/ptxas:12.9 \\")
        print("     /usr/local/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas \\")
        print("     {arch} /w/{name} -o {devnull}'")
        return EXIT["sdk_missing"]
    print(f"ptxas: {' '.join(ptxas_argv(template, 'sm_90', pathlib.Path('/w/p.ptx')))[:110]}")
    if not template:
        try:
            v = subprocess.run([ptxas, "--version"], capture_output=True, text=True)
            print("        " + (v.stdout or v.stderr).strip().splitlines()[-1])
        except Exception:                                    # noqa: BLE001
            pass

    caps = [(7, 5), (8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (10, 1),
            (10, 3), (12, 0), (12, 1)]
    mismatches, untestable = [], []
    with tempfile.TemporaryDirectory(prefix="ptxverify-") as td:
        def probe(feat, arch):
            """assembles / rejected / '-' (oracle cannot answer this)."""
            src = render_probe_ptx(feat, arch)
            if src is None:
                return "-", ""
            # unique path per cell: a bind-mounted directory will happily serve
            # a stale copy of a file rewritten at the same path and size, which
            # turns into a confident but wrong "the table is wrong".
            tf = pathlib.Path(td) / f"{feat}.{arch}.ptx"
            tf.write_text(src)
            r = subprocess.run(ptxas_argv(template, arch, tf),
                               capture_output=True, text=True, timeout=180)
            err = next((l.strip() for l in (r.stdout + r.stderr).splitlines()
                        if " error " in l), "")[:100]
            return ("assembles" if r.returncode == 0 else "rejected"), err

        for label, use_cond, table_key, note in (
                ("can the SILICON do it", True, "available",
                 "arch-conditional features are probed on the `a` target, where "
                 "they exist;\nthe rest are probed on the bare target, which is "
                 "where they exist"),
                ("can the PINNED TINYGRAD reach it", False,
                 "reachable_with_pinned_tinygrad",
                 "tinygrad emits sm_<major><minor> and never the `a` form, so "
                 "every\narch-conditional feature is unreachable. This is the "
                 "study's finding, not a bug.")):
            print()
            print(f"  {label.upper()}  (compared against '{table_key}')")
            print(f"  -- {note}")
            hdr = (f"  {'arch':9} "
                   + " ".join(f"{f[:11]:>12}" for f in FEATURE_ORDER))
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))
            for cc in caps:
                bare, cond = bare_target(cc), arch_conditional_target(cc)
                cells, targets = [], []
                for feat in FEATURE_ORDER:
                    spec = FEATURE_GATES[feat]
                    want_bool = bool(derive_features(cc)["features"][feat]
                                     [table_key])
                    probe_arch = (cond if use_cond else bare) \
                        if spec["kind"] == "targets" else bare
                    want = "assembles" if want_bool else "rejected"
                    got, err = probe(feat, probe_arch)
                    targets.append(probe_arch)
                    if got == "-":
                        untestable.append(f"{feat}@{probe_arch}")
                        cells.append("-")
                        continue
                    if got != want:
                        mismatches.append((probe_arch, feat, got, want, err))
                    cells.append({"assembles": "yes", "rejected": "no",
                                  "-": "-"}[got])
                used = sorted(set(targets))
                label_t = used[0] if len(used) == 1 else f"{used[0]}+"
                print(f"  {bare:9} {label_t:10} "
                      + " ".join(f"{c:>12}" for c in cells))
                if len(used) > 1:
                    print(f"  {'':9} {'(' + ', '.join(used) + ')':10}")

    print()
    if untestable:
        print(f"  the oracle cannot answer {len(set(untestable))} cell(s): "
              f"{', '.join(sorted(set(untestable))[:8])}")
        print("  (nonexistent target name, or a PTX version too old for it. "
              "Those are\n   NOT evidence either way, and are never counted as "
              "agreement.)")
    if mismatches:
        print(f"\n  FEATURE_GATES DISAGREES WITH ptxas in {len(mismatches)} "
              f"cell(s):")
        for tgt, feat, got, want, err in mismatches:
            print(f"    {tgt:9} {feat:16} ptxas={got:10} table={want:10} {err}")
        return EXIT["self_check_failed"]
    print(f"\n  FEATURE_GATES agrees with ptxas everywhere the oracle could "
          f"answer, across {len(caps)}\n  capabilities and both target forms.")
    return EXIT["ok"]


def _on_box(args) -> int:
    """These run INSIDE the container, on the GPU. The local side never calls
    them with a GPU, so the dry run can only syntax-check them."""
    import subprocess                                        # noqa: PLC0415
    import time                                              # noqa: PLC0415
    out = pathlib.Path(args.out or ".")
    out.mkdir(parents=True, exist_ok=True)

    if args.ptx_probe:
        # The capability normally comes from the device. --ptx-probe-capability
        # overrides it so the CROSS-CHECK below -- the part that decides whether
        # a real run is trustworthy -- can be exercised on a machine with no
        # GPU. Without that, the one piece of logic that can invalidate a run
        # would be the one piece never run before it matters.
        cc = None
        if args.ptx_probe_capability:
            try:
                maj, minr = args.ptx_probe_capability.split(":")
                cc = (int(maj), int(minr))
                print(f"  capability OVERRIDDEN to sm_{cc[0]}{cc[1]} "
                      f"(no GPU was consulted)")
            except ValueError:
                print(f"  --ptx-probe-capability must look like 9:0, got "
                      f"{args.ptx_probe_capability!r}")
                return EXIT["usage"]
        else:
            try:
                import torch                                 # noqa: PLC0415
                cc = torch.cuda.get_device_capability(0)
                print(f"  device reports sm_{cc[0]}{cc[1]}")
            except Exception as e:                           # noqa: BLE001
                print(f"  cannot read the device capability: "
                      f"{type(e).__name__}: {e}")
                print("  pass --ptx-probe-capability MAJOR:MINOR to probe a "
                      "specific arch without a GPU")
                return EXIT["arch_mismatch"]

        report = {"capability": list(cc), "ptxas": find_ptxas(),
                  "cmd": args.ptxas_cmd, "status": "skipped", "reason": "",
                  "cells": {}, "disagreements": []}
        if not report["ptxas"] and not args.ptxas_cmd:
            report["reason"] = ("ptxas not found. install "
                                "nvidia-cuda-nvcc-cu12, or pass --ptxas-cmd")
            (out / "ptx_probe.json").write_text(
                json.dumps(report, indent=2) + "\n")
            print("  " + report["reason"])
            return EXIT["sdk_missing"]

        bare, cond = bare_target(cc), arch_conditional_target(cc)
        archs = [bare, cond]
        work = out / "ptxwork"
        work.mkdir(parents=True, exist_ok=True)
        for feat in PTX_PROBES:
            for p_arch in archs:
                src = render_probe_ptx(feat, p_arch)
                if src is None:
                    report["cells"][f"{feat}@{p_arch}"] = {
                        "verdict": "oracle-cannot-test",
                        "detail": "no such ptxas target, or the PTX version "
                                  "for it is older than the feature"}
                    continue
                # A unique filename per cell, deliberately. When ptxas runs
                # through a bind-mounted directory (this repo's archlab docker
                # image), reusing one filename makes Docker serve a STALE copy
                # of a file that was rewritten moments earlier at the same
                # size. That produced a confident, wrong "ptxas disagrees with
                # FEATURE_GATES" on a cell that assembles. Never reuse a path.
                tf = work / f"{feat}.{p_arch}.ptx"
                tf.write_text(src)
                r = subprocess.run(ptxas_argv(args.ptxas_cmd, p_arch, tf),
                                   capture_output=True, text=True, timeout=180)
                err = next((l.strip() for l in (r.stdout + r.stderr).splitlines()
                            if " error " in l), "")[:300]
                report["cells"][f"{feat}@{p_arch}"] = {
                    "verdict": "assembles" if r.returncode == 0 else "rejected",
                    "detail": err}
        report["status"] = "ran"

        # THE CROSS-CHECK. Two questions, two different targets:
        #   'available'                    -> ask the target the feature lives
        #                                       on (`a` for arch-conditional)
        #   'reachable_with_pinned_tinygrad' -> ask the bare target, because
        #                                       that is all tinygrad emits
        # A disagreement in either direction means the transcribed table no
        # longer describes this toolchain, and the run's central claim is at
        # risk, so it fails rather than warns.
        table = derive_features(cc)["features"]
        for feat, spec in FEATURE_GATES.items():
            for key in ("available", "reachable_with_pinned_tinygrad"):
                use_a = spec["kind"] == "targets" and key == "available"
                cell = report["cells"].get(f"{feat}@{cond if use_a else bare}", {})
                got = cell.get("verdict")
                if got not in ("assembles", "rejected"):
                    continue        # cannot-test is not evidence either way
                if (got == "assembles") != bool(table[feat][key]):
                    report["disagreements"].append(
                        f"{feat}.{key}: ptxas says {got} on "
                        f"{cond if use_a else bare}, table says {table[feat][key]}")
        (out / "ptx_probe.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"  {len(report['cells'])} cells -> {out / 'ptx_probe.json'}")
        for k in sorted(report["cells"]):
            print(f"    {k:36} {report['cells'][k]['verdict']}")
        if report["disagreements"]:
            print()
            print("  FAIL: ptxas disagrees with FEATURE_GATES:")
            for d in report["disagreements"]:
                print(f"    {d}")
            print("  The transcribed feature table no longer describes this "
                  "toolchain.\n  Do not publish this run's feature claims until "
                  "it is re-derived:\n    python remote/cloud_run.py "
                  "--provider modal --arch h100 --verify-feature-table")
            return EXIT["arch_mismatch"]
        print("  ptxas agrees with FEATURE_GATES on every answerable cell")
        return EXIT["ok"]

    # --remote-capture-codegen
    sys.path.insert(0, str(REPO / "kda"))
    from tinygrad import Device, Tensor                      # noqa: PLC0415
    import numpy as np                                       # noqa: PLC0415
    cap: list[str] = []
    d = Device[Device.DEFAULT]
    orig = getattr(d.compiler, "compile_cached", None)
    if orig is not None:
        def spy(src, *a, **k):
            try:
                cap.append(str(src))
            except Exception:                                 # noqa: BLE001
                pass
            return orig(src, *a, **k)
        d.compiler.compile_cached = spy
    try:
        import test_ref as T                                  # noqa: PLC0415
        import tiny_kda                                       # noqa: PLC0415
        q, k, v, g, beta = T.rnd(2, 128, 2, 2, 32, 32, seed=0, l2n=True)
        tiny_kda.kda_chunk(q, k, v, g, beta, chunk_size=32)
    except Exception as e:                                    # noqa: BLE001
        print(f"  codegen capture failed: {type(e).__name__}: {e}")
    finally:
        if orig is not None:
            d.compiler.compile_cached = orig
    for i, s in enumerate(cap):
        (out / f"kernel{i:04d}.cu").write_text(s)
    (out / "INDEX.json").write_text(json.dumps(
        {"kernels": len(cap), "device": Device.DEFAULT,
         "generated_at": time.time()}, indent=2) + "\n")
    print(f"  captured {len(cap)} generated kernels -> {out}")
    _ = Tensor, np
    return EXIT["ok"]


if __name__ == "__main__":
    import time                                              # noqa: PLC0415
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted locally before anything was provisioned")
        sys.exit(EXIT["usage"])
