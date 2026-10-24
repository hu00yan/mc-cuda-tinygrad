"""prof.py -- the measurement instrument for the TinyGrad-vs-hand-written study.

Why this exists
---------------
Wall-clock numbers say *how much slower*. Hardware counters say *why*. This module
is built so that the "why" survives on hardware where no profiler is allowed --
which is most of the hardware this study intends to use (Kaggle T4 = sm_75,
P100 = sm_60, both free and unlimited, both in containers where Nsight Compute
almost certainly fails: `ncu` needs NVreg_RestrictProfilingToAdminUsers=0 and
its serialised replay is 10-100x slower than the kernel it measures).

Three levels, plus a floor and a zero. Always report which one you got. Never let a
missing measurement look like a zero. NOTE the numbering is not a quality order: level 3 is
the *floor* (the thing that always works) and level 1 is the *best*. See LEVEL_RANK, which
converts a level into the monotone rank every capability question actually compares.

  level 3  Event timing + first-principles accounting. Needs no profiler, no
           privileges, no `ncu`. Times each launch with the backend's own device
           event (CUDA events on the CUDA path, GPU-side timestamps on the NV
           path), multiplies by the bytes and FLOPs the kernel provably must
           move and perform, and reports achieved GB/s, achieved TFLOP/s,
           arithmetic intensity and roofline position. This is the floor, and
           the floor is already enough to place a kernel on the roofline.
           Level 3 also has a *static* channel: the generated PTX is rendered
           for the target arch and inspected, which tests the vectorisation and
           tensor-pipe hypotheses with no GPU and no privileges at all.

  level 2  + Nsight Compute with a targeted `--metrics` list and a bounded
           kernel count. Detected first, skipped cleanly if unusable.

  level 1  + Nsight Compute `--set full`, i.e. section-based and source-level.
           Only attempted when level 2 is unavailable or the richer set is the
           point.

Level 0 means no CUDA roofline and no CUDA counters can be derived here (CPU backend, a
non-CUDA device, or CUDA absent). Timings are still taken where the backend provides them
and are labelled as what they are. That is a real, reportable result -- not a failure.

The mechanism each level actually uses, because "CUDA events" as a slogan and as a fact
are different things and it matters which one you got:
  CUDA device  cuEventRecord / cuEventElapsedTime around each launch.
  NV device    GPU-written timestamps on the submission queue, host-read after a sync.
               Device-side like an event, no profiler needed -- but it synchronises after
               every timed launch, so per-kernel times are an UPPER BOUND there. The harness
               brackets the phase with the host clock and flags the channel UNTRUSTED if the
               two disagree by more than 5x.

Two rules this module never breaks:
  1. An unavailable measurement is `None` with a stated reason. Never 0.
  2. Nothing device-specific is hardcoded in the measurement path. Every number
     is derived from the device at hand (driver attributes + the arch string the
     driver itself reports). The only constants are *structural* microarchitecture
     facts (cores per SM), they are labelled as such, and an unknown arch yields
     "unknown", never a guess.

Dependencies: tinygrad + the standard library. `numpy` only for the callers.
Every public entry point swallows its own exceptions: the instrument must never
raise out of the measurement path.

    from prof import Work, measure, describe_capability, report
    print(describe_capability().explain())          # which level, and why
    m = measure(Work("matmul 1024", lambda: a @ b, flops=2*1024**3, bytes=3*1024*1024*4))
    print(report(m))                                # the measurement, the evidence, the verdicts

`flops` and `bytes` on the Work are the CALLER's hand-computed minimum -- the denominator
the roofline uses, so they are the study's claim and not the instrument's. Leave them None
and you get rates but no roofline position, which is a correct and useful result.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

# --------------------------------------------------------------------------------------
# 0. tinygrad plumbing, imported once and used defensively everywhere
# --------------------------------------------------------------------------------------

def _tg():
  """Import the tinygrad bits this module needs. Raises; callers use _safe()."""
  from tinygrad import Tensor, Device                      # noqa: F401
  from tinygrad.helpers import Target, ansistrip, prod     # noqa: F401
  from tinygrad.uop.ops import Ops, UOp                    # noqa: F401
  from tinygrad.codegen import to_program                  # noqa: F401
  return Tensor, Device, Target, ansistrip, prod, Ops, UOp, to_program


def _safe(fn, *a, default=None, what: str = ""):
  """Run fn(*a); on ANY exception return `default` and record the reason.

  Nothing in the measurement path is allowed to raise. `default=None` is the
  right answer for "could not measure" in every case used here, which is exactly
  why a missing measurement can never be confused with a measured zero.
  """
  try:
    return fn(*a)
  except BaseException as e:  # noqa: BLE001 -- deliberate: instrument must not die
    name = what or getattr(fn, "__name__", "call")
    LAST_WARNINGS.append(f"{name}: {type(e).__name__}: {e}")
    return default


LAST_WARNINGS: list[str] = []


# --------------------------------------------------------------------------------------
# 1. formatting
# --------------------------------------------------------------------------------------

def fp32_mma_possible(arch: str | None) -> bool | None:
  """Can an fp32 matmul reach the matrix path at all on this arch?

  TinyGrad lowers fp32 matmul to FFMA, or to a TF32 MMA where TF32 exists. TF32 starts at
  sm_80. So on sm_60/sm_75 a zero-MMA result is architecture, not behaviour, and this
  function exists so the H2 verdict says so instead of implying a finding.
  """
  if not arch or not arch.startswith("sm_"):
    return None
  try:
    return int(arch[3:]) >= 80
  except ValueError:
    return None


def _si(x: float | None, unit: str = "", width: int = 0) -> str:
  """Human number, or the literal string for None. Never prints 0 for None."""
  if x is None:
    return f"{'n/a':>{width}s}"
  for scale, suf in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
    if abs(x) >= scale:
      return f"{x/scale:.3g}{suf}{unit}".rjust(width)
  return f"{x:.3g}{unit}".rjust(width)


def _b(x: int | None) -> str:
  if x is None:
    return "n/a"
  for scale, suf in ((1 << 40, "TiB"), (1 << 30, "GiB"), (1 << 20, "MiB"), (1 << 10, "KiB")):
    if abs(x) >= scale:
      return f"{x/scale:.3g} {suf}"
  return f"{x} B"


def _pct(x: float | None) -> str:
  return "n/a" if x is None else f"{x*100:.4g}%"


# --------------------------------------------------------------------------------------
# 2. The counter set, and the hypothesis each counter exists to test
# --------------------------------------------------------------------------------------
#
# A counter with no hypothesis attached is a number nobody knows what to do with.
# So every counter below names the claim it is evidence for, and every claim says
# which level can test it and which cannot. `min_level` is the lowest level at
# which the claim can be *decided*; `static_probe` names a zero-privilege check
# that can pre-empt or corroborate it.
#
# NAMING. These ids are M1..M8, not H1..H8, on purpose. `HYPOTHESES.md` in this directory
# is the study's pre-registration and already uses H1..H14 for whole-operator claims; these
# are the narrower MECHANISM questions -- "did this kernel vectorise its loads", "did it
# reach the matrix pipe" -- and each of the study's hypotheses is decided by a subset of
# these. The mapping is in README.md. Two files in one study using the same id for
# different claims is how a result gets attributed to the wrong hypothesis.

@dataclass(frozen=True)
class Counter:
  name: str                       # ncu metric name, exactly as passed to --metrics
  hyp: str                        # hypothesis id
  unit: str | None = None
  arch_note: str = ""             # availability caveat, e.g. "sm_80+ only"
  read: str = ""                  # what a value means, one clause


@dataclass(frozen=True)
class Hypothesis:
  id: str
  claim: str                       # the falsifiable statement
  min_level: int                   # lowest level that can decide it
  counters: tuple[str, ...]        # ncu metric names
  static_probe: str                # zero-privilege check, or "-" for none
  decides: str                     # what a value would prove
  cannot: str = ""                 # what stays untested even at the best level
  has_static: bool = True          # True iff static_probe is a real probe, not a "-" note


HYPOTHESES: dict[str, Hypothesis] = {
  "M1_COALESCE": Hypothesis(
    id="M1_COALESCE",
    claim="Generated code degrades strided/gathered access to scalar 32-bit loads instead of "
          "wide coalesced transactions (the 'Correct but Slow' RC1 signature: 12.55% load "
          "efficiency, 16.4 sectors per request).",
    min_level=2,
    counters=("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
              "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
              "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
              "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio",
              "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.avg.pct_of_peak_sustained_active",
              "l1tex__t_sectors_pipe_lsu_mem_global_op_st.avg.pct_of_peak_sustained_active",
              "smsp__inst_executed_op_global_ld.sum",
              "smsp__inst_executed_op_global_st.sum",
              "smsp__inst_executed_op_global_ld_replay.sum",
              "smsp__inst_executed_op_global_st_replay.sum"),
    static_probe="PTX: distribution of `ld.global.vN` widths. A kernel whose global loads are "
                 "all .v4/.v8/.v16 is vectorised; one that emits only ld.global.f32 cannot be, "
                 "whatever the profiler says.",
    decides="implied bytes/thread/request (sectors_per_request * 32 / 32) and load efficiency as "
            "a % of the L1 sector peak. Implied 4 B/thread on a matrix-shaped kernel = the RC1 "
            "failure, reproduced without a profiler.",
    cannot="The published 12.55%/%-of-peak figure is an L1 *counter*, not derivable from the "
           "source or from time, so level 3 can falsify 'the code is scalar' but cannot "
           "reproduce the published number. Sector efficiency also cannot distinguish 'one wide "
           "warp-coalesced request' from 'four narrow ones' when both move the same bytes."),

  "M2_MMA": Hypothesis(
    id="M2_MMA",
    claim="The generated reduction never reaches the hardware matrix path: no MMA is issued, so "
          "the work lands on the FP32 FMA pipe at a fraction of tensor throughput.",
    min_level=2,
    counters=("sm__inst_executed_pipe_tensor.sum",
              "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
              "sm__inst_executed_pipe_tensor_op_hmma.sum",
              "sm__inst_executed_pipe_tensor_op_imma.sum",
              "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
              "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
              "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active",
              "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
              "sm__inst_executed.sum"),
    static_probe="PTX: count of `mma.sync` / `wmma.mma` / `wgmma.mma` versus `fma.rn.f32`. Zero MMA "
                 "in code that should be a matmul is a positive identification, not an absence of "
                 "evidence. The FFMA count here is an instruction count in the emitted body, so it "
                 "identifies WHICH PIPE the code targets and not how much work reaches it -- the "
                 "magnitude needs sm__sass_thread_inst_executed_op_ffma_pred_on.sum, which is a "
                 "thread-instruction count.",
    decides="tensor pipe utilisation > 0 (matrix path used) or == 0 with a high FMA pipe and a "
            "thread-instruction FFMA count near 2*M*N*K (matrix path unused, work done in scalar "
            "FMA). This is the single measurement that decides it.",
    cannot="It cannot say *why* the MMA is missing -- that is the codegen lane's job, from the "
           "source and the schedule. And a kernel with no MMA on a device where MMA is not "
           "available (sm_60/75 fp32 has no TF32) is not a finding; check arch first."),

  "M3_OCCUPANCY": Hypothesis(
    id="M3_OCCUPANCY",
    claim="Register pressure or spilling collapsed achieved occupancy (RC3: 254 regs/thread, "
          "12.4% occupancy, 51.5 GB local-load + 34.4 GB local-store).",
    min_level=2,
    counters=("launch__registers_per_thread",
              "launch__occupancy_limit_registers",
              "launch__occupancy_limit_shared_mem",
              "launch__occupancy_limit_warps",
              "launch__occupancy_limit_blocks",
              "launch__shared_mem_per_block",
              "launch__shared_mem_per_block_driver",
              "launch__block_size",
              "launch__grid_size",
              "sm__warps_active.avg.pct_of_peak_sustained_active",
              "sm__maximum_warps_per_active_cycle_pct",
              "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
              "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
              "smsp__inst_executed_op_local_ld.sum",
              "smsp__inst_executed_op_local_st.sum"),
    static_probe="PTX: `.local` declarations and `.local` ld/st instructions. Presence of a `.local` "
                 "array is suggestive; absence of local traffic is proof of no spill in the PTX "
                 "that ptxas is handed (ptxas can still introduce spill code of its own, which is "
                 "why the counter is the arbiter).",
    decides="spill traffic > 0 (real bytes through the local path) and the achieved-occupancy "
            "percentage, plus which limit binds (registers / smem / warps / blocks).",
    cannot="Local-memory traffic is not proof of *spill*: an algorithm can address local memory "
           "on purpose. Registers-per-thread from launch__ is a ptxas output, so a `.local` array "
           "in PTX with zero allocated registers is a contradiction this instrument reports but "
           "does not resolve."),

  "M4_REUSE": Hypothesis(
    id="M4_REUSE",
    claim="Data was streamed rather than reused: L2 hit rate near zero on a working set that "
          "fits, DRAM bytes far above the minimum (RC2 / the 49.4%-vs-80.5% L2 comparison).",
    min_level=2,
    counters=("lts__t_sector_hit_rate.pct",
              "l1tex__t_sector_hit_rate.pct",
              "lts__t_sectors.sum",
              "lts__t_sectors_srcunit_tex_op_read.sum",
              "lts__t_sectors_srcunit_tex_op_write.sum",
              "lts__t_sectors_srcunit_tex_lookup_hit.sum",
              "lts__t_sectors_srcunit_tex_lookup_miss.sum",
              "dram__bytes_read.sum",
              "dram__bytes_write.sum",
              "dram__bytes.sum.per_second",
              "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
              "lts__throughput.avg.pct_of_peak_sustained_elapsed",
              "l1tex__throughput.avg.pct_of_peak_sustained_elapsed"),
    static_probe="PTX: count of `st.shared` / `ld.shared` / `cp.async` (cp.async is sm_80+ only, "
                 "so on sm_60/75 its absence is architectural, not a finding). zero shared-memory "
                 "traffic corroborates tinygrad's own documented 'we never copy the inputs to on "
                 "chip SRAM'.",
    decides="L2 hit rate, and DRAM bytes against the hand-computed lower bound. An L2 hit rate "
            "near zero on a working set that fits is the signature; nothing else is.",
    cannot="Level 3 cannot see cache behaviour at all: it only sees *distinct* bytes per kernel "
           "and aggregate time. A kernel that re-reads a hot buffer a million times and a kernel "
           "that streams it once look identical to level 3 if their distinct-byte sets match. "
           "Also, `lts__t_sector_hit_rate` is undefined for kernels with no L2 traffic, and this "
           "instrument will report None there rather than 0%."),

  "M5_STALL": Hypothesis(
    id="M5_STALL",
    claim="A confident guess about where the time goes is wrong; the truth is in the stall "
          "reasons and the warp state.",
    min_level=2,
    counters=("smsp__issue_active.avg.pct_of_peak_sustained_active",
              "smsp__warps_eligible.avg.per_cycle_active",
              "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
              "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
              "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
              "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
              "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
              "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
              "smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio",
              "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
              "smsp__average_warps_issue_stalled_membar_per_issue_active.ratio",
              "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
              "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio",
              "smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio",
              "smsp__average_warps_issue_stalled_drain_per_issue_active.ratio",
              "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio",
              "smsp__average_warps_issue_stalled_misc_per_issue_active.ratio",
              "smsp__average_warps_issue_stalled_selected_per_issue_active.ratio"),
    static_probe="- (stalls are a dynamic property; nothing in the source says where time went)",
    has_static=False,
    decides="which stall reason dominates issue slots, i.e. latency-bound vs issue-bound vs "
            "throttle-bound. This is the item that overturns confident guesses, which is why it "
            "is here and not dropped for being expensive.",
    cannot="A stall *ratio* is a ratio of stall-cycles to issue-cycles, not seconds. A kernel that "
           "is 90% stalled and one that is 30% stalled can have the same duration if the other "
           "part differs. This instrument will not convert stall ratios into time shares; doing "
           "so would be an invention."),

  "M6_MATERIALISE": Hypothesis(
    id="M6_MATERIALISE",
    claim="The KDA port materialises the [B,NT,HV,C,C,K] decay tensor where the hand-written "
          "kernel builds only [BC,K] per sub-chunk -- about 256x. If DRAM traffic matches the "
          "larger figure the intermediate is real traffic, not a suspicion.",
    min_level=3,
    counters=("dram__bytes_write.sum", "dram__bytes_read.sum"),
    static_probe="PTX per kernel: the global store footprint is the buffer the kernel writes, so "
                 "the presence of a store of a C*C*K-sized tensor is visible in the buffer "
                 "signature without any counter at all.",
    decides="level 3 already decides it: bytes_est (distinct bytes each kernel actually touches, "
            "read off the real buffers) against the caller's hand-computed bytes_min and "
            "materialised_bytes. Ratio ~256x is the confirmation; the dram__bytes counters only "
            "corroborate that the traffic reached memory rather than dying in L2.",
    cannot="Level 3 sees *requested* traffic, not *DRAM* traffic. A buffer that stays resident in "
           "L2 is counted as moved by level 3 and never reaches DRAM. So level 3 can prove an "
           "intermediate was written and can bound the traffic, but only the dram__bytes counters "
           "can prove it reached memory -- and only level 2/1 can do that."),

  "M7_PRECISION": Hypothesis(
    id="M7_PRECISION",
    claim="Precision was silently lowered: fp32 lowered to TF32 (sm_80+) or to fp16 accumulate, "
          "which is a correctness change disguised as a speedup.",
    min_level=3,
    counters=("sm__inst_executed_pipe_tensor_op_hmma.sum",),
    static_probe="PTX: the mma variant is in the text. `mma.sync...f32.tf32.tf32.f32` is TF32; "
                 "`...f16.f16.f32` is fp16 in, fp32 accumulate. Forcing the arch to the oldest "
                 "supported (sm_60) makes the absence of any TF32 MMA structural rather than "
                 "informative -- TF32 does not exist there.",
    decides="which matrix variant was emitted, from the source text. This is decidable at level 3 "
            "and needs no hardware at all.",
    cannot="It cannot see a precision change that happens in ptxas or in a library kernel, and it "
           "cannot tell you whether the result is *wrong* -- that is a numerics test "
           "(probes/fp32_correctness.py), not a counter."),

  "M8_SMEM": Hypothesis(
    id="M8_SMEM",
    claim="Inputs are never staged through shared memory / async copy, which tinygrad's own "
          "developer docs concede ('we never copy the inputs to on chip SRAM').",
    min_level=3,
    counters=("l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed",
              "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
              "smsp__inst_executed_op_shared_ld.sum",
              "smsp__inst_executed_op_shared_st.sum"),
    static_probe="PTX: `st.shared` / `ld.shared` / `cp.async` counts and `bar.sync` count.",
    decides="zero shared traffic with a non-trivial global-load count is a positive confirmation "
            "of the documented behaviour, at level 3, statically.",
    cannot="cp.async does not exist before sm_80, so on the T4 and P100 that are this study's "
           "primary instruments the *async* half of the claim is untestable. Absence of cp.async "
           "there is architecture, not behaviour."),
}

COUNTERS: list[Counter] = [
  # --- H1 coalescing / vectorisation -------------------------------------------------
  Counter("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", "M1_COALESCE", "sector",
          "", "global load sectors (transactions) -- the denominator of the efficiency argument"),
  Counter("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", "M1_COALESCE", "request",
          "", "global load requests -- requests x sectors/request = waste"),
  Counter("l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio", "M1_COALESCE",
          "sector/request", "", "4 => 32-bit/thread, 16 => 128-bit/thread. The RC1 signature."),
  Counter("l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio", "M1_COALESCE",
          "sector/request", "", "store-side equivalent"),
  Counter("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.avg.pct_of_peak_sustained_active", "M1_COALESCE",
          "%", "", "load efficiency as a % of L1 sector peak. The 12.55% in RC1 is this shape."),
  Counter("l1tex__t_sectors_pipe_lsu_mem_global_op_st.avg.pct_of_peak_sustained_active", "M1_COALESCE",
          "%", "", "store efficiency"),
  Counter("smsp__inst_executed_op_global_ld.sum", "M1_COALESCE", "inst", "", "global load instructions issued"),
  Counter("smsp__inst_executed_op_global_st.sum", "M1_COALESCE", "inst", "", "global store instructions issued"),
  Counter("smsp__inst_executed_op_global_ld_replay.sum", "M1_COALESCE", "inst", "", "replayed due to L1 miss"),
  Counter("smsp__inst_executed_op_global_st_replay.sum", "M1_COALESCE", "inst", "", "replayed due to L1 miss"),
  # --- H2 matrix path -----------------------------------------------------------------
  Counter("sm__inst_executed_pipe_tensor.sum", "M2_MMA", "inst", "sm_70+",
          "tensor-pipe instructions. ZERO on a matrix-shaped kernel is the finding."),
  Counter("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active", "M2_MMA", "%",
          "sm_75+/named differently on older archs", "tensor pipe utilisation"),
  Counter("sm__inst_executed_pipe_tensor_op_hmma.sum", "M2_MMA", "inst", "sm_80+ naming; Volta/Turing differ",
          "HMMA (floating-point matrix) instructions"),
  Counter("sm__inst_executed_pipe_tensor_op_imma.sum", "M2_MMA", "inst", "sm_80+ naming",
          "IMMA (integer matrix) instructions"),
  Counter("sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active", "M2_MMA", "%",
          "sm_80+ naming", "HMMA pipe utilisation"),
  Counter("sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active", "M2_MMA", "%", "",
          "FP32 FMA pipe utilisation. High FMA + zero tensor == the matrix ran on the scalar pipe."),
  Counter("sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active", "M2_MMA", "%", "",
          "integer/logic pipe utilisation"),
  Counter("sm__sass_thread_inst_executed_op_ffma_pred_on.sum", "M2_MMA", "thread-inst", "",
          "scalar FFMA thread-instructions; compare with 2*M*N*K to identify an FMA-based matmul"),
  Counter("sm__inst_executed.sum", "M2_MMA", "inst", "", "total instructions, for the instruction-mix ratio"),
  # --- H3 occupancy / spill -----------------------------------------------------------
  Counter("launch__registers_per_thread", "M3_OCCUPANCY", "reg/thread", "",
          "ptxas register allocation. >= 168 on a 2048-thread SM halves occupancy on its own."),
  Counter("launch__occupancy_limit_registers", "M3_OCCUPANCY", "block", "", "blocks resident, register-limited"),
  Counter("launch__occupancy_limit_shared_mem", "M3_OCCUPANCY", "block", "", "blocks resident, smem-limited"),
  Counter("launch__occupancy_limit_warps", "M3_OCCUPANCY", "block", "", "blocks resident, warp-limited"),
  Counter("launch__occupancy_limit_blocks", "M3_OCCUPANCY", "block", "", "blocks resident, hw-limit"),
  Counter("launch__shared_mem_per_block", "M3_OCCUPANCY", "byte", "", "dynamic shared memory per block"),
  Counter("launch__shared_mem_per_block_driver", "M3_OCCUPANCY", "byte", "", "driver-reserved smem per block"),
  Counter("launch__block_size", "M3_OCCUPANCY", "thread", "", "threads per block"),
  Counter("launch__grid_size", "M3_OCCUPANCY", "block", "", "blocks in the grid"),
  Counter("sm__warps_active.avg.pct_of_peak_sustained_active", "M3_OCCUPANCY", "%", "",
          "achieved occupancy. The 12.4% in RC3 is this shape."),
  Counter("sm__maximum_warps_per_active_cycle_pct", "M3_OCCUPANCY", "%", "",
          "theoretical occupancy the launch configuration allows"),
  Counter("l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum", "M3_OCCUPANCY", "sector", "",
          "local-load sectors. RC3's 51.5 GB of local traffic is this, x32 B."),
  Counter("l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum", "M3_OCCUPANCY", "sector", "",
          "local-store sectors. RC3's 34.4 GB."),
  Counter("smsp__inst_executed_op_local_ld.sum", "M3_OCCUPANCY", "inst", "", "local load instructions"),
  Counter("smsp__inst_executed_op_local_st.sum", "M3_OCCUPANCY", "inst", "", "local store instructions"),
  # --- H4 reuse / cache ---------------------------------------------------------------
  Counter("lts__t_sector_hit_rate.pct", "M4_REUSE", "%", "undefined for kernels with no L2 traffic",
          "L2 hit rate. 49.4% vs 80.5% in RC2b is this shape."),
  Counter("l1tex__t_sector_hit_rate.pct", "M4_REUSE", "%", "undefined for kernels with no L1 traffic",
          "L1 hit rate"),
  Counter("lts__t_sectors.sum", "M4_REUSE", "sector", "", "L2 sector traffic"),
  Counter("lts__t_sectors_srcunit_tex_op_read.sum", "M4_REUSE", "sector", "", "L2 read sectors"),
  Counter("lts__t_sectors_srcunit_tex_op_write.sum", "M4_REUSE", "sector", "", "L2 write sectors"),
  Counter("lts__t_sectors_srcunit_tex_lookup_hit.sum", "M4_REUSE", "sector", "", "L2 lookup hits"),
  Counter("lts__t_sectors_srcunit_tex_lookup_miss.sum", "M4_REUSE", "sector", "", "L2 lookup misses"),
  Counter("dram__bytes_read.sum", "M4_REUSE", "byte", "", "bytes actually read from DRAM"),
  Counter("dram__bytes_write.sum", "M4_REUSE", "byte", "", "bytes actually written to DRAM"),
  Counter("dram__bytes.sum.per_second", "M4_REUSE", "byte/s", "", "achieved DRAM bandwidth, hardware-measured"),
  Counter("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", "M4_REUSE", "%", "",
          "DRAM throughput as a % of peak -- the hardware's own roofline, no model needed"),
  Counter("lts__throughput.avg.pct_of_peak_sustained_elapsed", "M4_REUSE", "%", "", "L2 throughput vs peak"),
  Counter("l1tex__throughput.avg.pct_of_peak_sustained_elapsed", "M4_REUSE", "%", "", "L1 throughput vs peak"),
  # --- H5 stalls ----------------------------------------------------------------------
  Counter("smsp__issue_active.avg.pct_of_peak_sustained_active", "M5_STALL", "%", "",
          "issue-slot utilisation. Low + high stall ratio = latency-bound."),
  Counter("smsp__warps_eligible.avg.per_cycle_active", "M5_STALL", "warp/cycle", "",
          "warps eligible to issue per cycle"),
  *[Counter(f"smsp__average_warps_issue_stalled_{r}_per_issue_active.ratio", "M5_STALL", "ratio", "",
            f"{r} stall cycles per issued instruction")
    for r in ("long_scoreboard", "short_scoreboard", "wait", "mio_throttle", "math_pipe_throttle",
              "lg_throttle", "imc_miss", "barrier", "membar", "not_selected", "no_instruction",
              "branch_resolving", "drain", "dispatch_stall", "misc", "selected", "tex_throttle",
              "membar", "sleeping", "no_instruction")],
  # --- H6 materialisation (corroboration only; level 3 decides) ------------------------
  Counter("dram__bytes_write.sum", "M6_MATERIALISE", "byte", "",
          "did the materialised intermediate actually reach memory"),
  Counter("dram__bytes_read.sum", "M6_MATERIALISE", "byte", "", "how much of it was read back"),
  # --- H7 precision -------------------------------------------------------------------
  Counter("sm__inst_executed_pipe_tensor_op_hmma.sum", "M7_PRECISION", "inst", "sm_80+ naming",
          "which HMMA variant ran; level 3 reads the variant from PTX text instead"),
  # --- H8 shared memory ---------------------------------------------------------------
  Counter("l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed", "M8_SMEM",
          "%", "", "shared-memory pipe utilisation"),
  Counter("l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum", "M8_SMEM", "conflict", "",
          "smem bank conflicts"),
  Counter("smsp__inst_executed_op_shared_ld.sum", "M8_SMEM", "inst", "", "shared load instructions"),
  Counter("smsp__inst_executed_op_shared_st.sum", "M8_SMEM", "inst", "", "shared store instructions"),
]

# Dedupe by name (H6/H7 deliberately re-use counters from H2/H4) while keeping first
# occurrence order, and record every hypothesis a metric serves.
_by_name: dict[str, Counter] = {}
METRIC_HYP: dict[str, list[str]] = {}
for c in COUNTERS:
  if c.name not in _by_name:
    _by_name[c.name] = c
  METRIC_HYP.setdefault(c.name, [])
  if c.hyp not in METRIC_HYP[c.name]:
    METRIC_HYP[c.name].append(c.hyp)
COUNTERS = list(_by_name.values())
COUNTER_BY_NAME: dict[str, Counter] = {c.name: c for c in COUNTERS}

# The targeted level-2 list: one cheap pass, every hypothesis that can be decided with a raw
# counter, and nothing that needs a derived section. Kept deliberately short because replay
# cost -- not metric availability -- is the binding constraint.
#
# One entry serves no hypothesis and is not a fishing metric: `gpu__time_duration.sum` is the
# hardware's own kernel duration, and it exists to be compared against the level-3 event time.
# If the two disagree, one of them is wrong, and that is worth knowing before any other
# counter in the list is believed.
LEVEL2_METRICS: tuple[str, ...] = tuple(dict.fromkeys([
  # timing + memory transactions (M1)
  "gpu__time_duration.sum",
  "dram__bytes_read.sum", "dram__bytes_write.sum",
  "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
  "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.avg.pct_of_peak_sustained_active",
  "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
  "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
  # matrix path (M2)
  "sm__inst_executed_pipe_tensor.sum",
  "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
  "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
  "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
  # occupancy / spill (M3)
  "launch__registers_per_thread", "launch__occupancy_limit_registers",
  "launch__occupancy_limit_shared_mem", "launch__shared_mem_per_block", "launch__block_size",
  "sm__warps_active.avg.pct_of_peak_sustained_active",
  "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
  "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
  # cache (M4)
  "lts__t_sector_hit_rate.pct", "l1tex__t_sector_hit_rate.pct",
  "lts__t_sectors.sum", "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
  # stalls (M5) -- the three that actually discriminate
  "smsp__issue_active.avg.pct_of_peak_sustained_active",
  "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
  "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
  "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
  "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
]))

# Metrics used only for the capability probe: two, and both cheap, and both present on
# every arch from Fermi onwards. If these two fail, nothing else will.
PROBE_METRICS: tuple[str, ...] = ("gpu__time_duration.sum", "sm__cycles_elapsed.avg")

# --------------------------------------------------------------------------------------
# 3. Device facts and the roofline, derived from the device at hand
# --------------------------------------------------------------------------------------
#
# Every number below is read from the device. The driver is the only authority:
# cuDeviceGetAttribute gives SM count, both clocks, the memory bus width, L2 size and
# the occupancy limits, so a "peak bandwidth table" is not needed and a wrong table
# cannot silently produce a wrong roofline.
#
# The one thing the driver does not expose is cores per SM (fp32 lanes), which is a
# structural property of the microarchitecture rather than a measurement. That is the
# single per-arch constant in this file, it is taken from the CUDA C Programming
# Guide's compute-capability table, and an arch that is not in it yields
# peak_flops = None with a reason. Nothing is guessed.

# fp32 32-bit cores per SM. Source: CUDA C Programming Guide, "Compute Capabilities"
# (Maximum number of 32-bit floating-point cores per multiprocessor). Structural, not
# measured; used only to turn an SM count and a clock into a peak FLOP rate.
FP32_CORES_PER_SM: dict[str, int] = {
  "sm_53": 64, "sm_60": 64, "sm_61": 64, "sm_62": 64, "sm_70": 64, "sm_72": 64,
  "sm_73": 64, "sm_75": 64, "sm_80": 64, "sm_86": 64, "sm_87": 64, "sm_89": 128,
  "sm_90": 128, "sm_100": 128, "sm_101": 128, "sm_120": 128,
}

# Which of the three archs the study cares about, with the only arch-specific fact we
# need spelled out. Kept as a readable note, not as a table of results.
ARCH_NOTES: dict[str, str] = {
  "sm_60": "Pascal P100. No TF32. fp16 tensor cores only, so an fp32 matmul cannot use "
           "the matrix path for the right reason -- absence of MMA here is architectural, "
           "not a tinygrad finding.",
  "sm_75": "Turing T4. No TF32. Same caveat as sm_60, and cp.async does not exist.",
  "sm_89": "Ada L4. TF32 exists (sm_80+), so a TF32 MMA here would be a real finding, and "
           "an fp32-only MMA is a real finding in the other direction.",
}


@dataclass
class DeviceFacts:
  device: str = "unknown"            # the tinygrad device string
  name: str | None = None            # driver device name, e.g. "Tesla T4"
  arch: str | None = None            # e.g. "sm_75"
  attrs: dict[str, int] = field(default_factory=dict)   # raw driver attributes
  props: dict[str, Any] = field(default_factory=dict)   # backend device_props(), verbatim
  driver_version: int | None = None
  nvidia_smi: str | None = None
  restrict_profiling_to_admins: str | None = None       # from /proc/driver/nvidia/params
  notes: list[str] = field(default_factory=list)

  # convenience accessors, all None-safe
  @property
  def is_cuda(self) -> bool: return self.device.split(":")[0].upper() in ("CUDA", "NV")
  @property
  def sm_count(self) -> int | None: return self.attrs.get("MULTIPROCESSOR_COUNT")
  @property
  def sm_clock_khz(self) -> int | None: return self.attrs.get("CLOCK_RATE")
  @property
  def mem_clock_khz(self) -> int | None: return self.attrs.get("MEMORY_CLOCK_RATE")
  @property
  def bus_bits(self) -> int | None: return self.attrs.get("GLOBAL_MEMORY_BUS_WIDTH")
  @property
  def l2_bytes(self) -> int | None: return self.attrs.get("L2_CACHE_SIZE")
  @property
  def regs_per_sm(self) -> int | None: return self.attrs.get("MAX_REGISTERS_PER_MULTIPROCESSOR")
  @property
  def max_threads_per_sm(self) -> int | None: return self.attrs.get("MAX_THREADS_PER_MULTIPROCESSOR")
  @property
  def smem_per_sm(self) -> int | None: return self.attrs.get("MAX_SHARED_MEMORY_PER_MULTIPROCESSOR")
  @property
  def smem_per_block_optin(self) -> int | None: return self.attrs.get("MAX_SHARED_MEMORY_PER_BLOCK_OPTIN")
  @property
  def warp_size(self) -> int: return self.attrs.get("WARP_SIZE", 32)
  @property
  def max_blocks_per_sm(self) -> int | None: return self.attrs.get("MAX_BLOCKS_PER_MULTIPROCESSOR")

  @property
  def theoretical_max_threads_per_sm(self) -> int | None:
    """regs_per_sm / 32 bits-per-reg / 8 granule, the classic occupancy identity.

    Reported alongside the driver's own MAX_THREADS_PER_MULTIPROCESSOR so the two can be
    compared; a mismatch is a fact about the device worth having, not an error to hide.
    """
    r = self.regs_per_sm
    return (r // 256) * 32 if r else None

  def smem_occupancy(self, smem_per_block: int | None) -> tuple[int | None, str]:
    """Blocks resident given smem, from the device's own smem/SM budget. No arch table."""
    per_sm, blk = self.smem_per_sm, smem_per_block
    if not per_sm:
      return None, "device did not report MAX_SHARED_MEMORY_PER_MULTIPROCESSOR"
    if not blk:
      return self.max_blocks_per_sm, "no dynamic shared memory requested"
    return max(1, per_sm // max(1, blk)), f"{per_sm} B smem/SM / {blk} B per block"

  def reg_occupancy(self, regs_per_thread: int | None) -> tuple[int | None, str]:
    """Warps resident per SM given a register count, from the device's own register file."""
    file_regs, rpt = self.regs_per_sm, regs_per_thread
    if not file_regs:
      return None, "device did not report MAX_REGISTERS_PER_MULTIPROCESSOR"
    if not rpt:
      return None, "register count for this kernel not available"
    regs_per_warp = rpt * 32
    warps = (file_regs // max(1, regs_per_warp // 256 * 256)) if regs_per_warp else None
    return warps, f"{file_regs} regs/SM, {rpt} regs/thread (allocation granule 256 regs/warp)"


def _driver_attributes(ordinal: int = 0) -> tuple[dict[str, int], int | None, str | None, list[str]]:
  """Read device attributes through the CUDA driver API that tinygrad already binds.

  Uses tinygrad's own ctypes bindings, so there is no extra dependency, and it works for
  both CUDA backends (`CUDA`, the driver-API device, and `NV`, the host-queue device) --
  both are the CUDA driver underneath. Everything is optional: a missing attribute is an
  absent key, not a zero, and a total failure returns ({}, None, None, [reason]).
  """
  notes: list[str] = []
  out: dict[str, int] = {}
  try:
    from tinygrad.runtime.autogen import cuda  # noqa: PLC0415
  except Exception as e:  # noqa: BLE001
    return out, None, None, [f"CUDA driver bindings unavailable: {type(e).__name__}: {e}"]
  try:
    if cuda.cuInit(0) != 0:
      return out, None, None, ["cuInit returned non-zero (no CUDA driver in this process)"]
    dev = ctypes.c_int()
    if cuda.cuDeviceGet(ctypes.byref(dev), ordinal) != 0:
      return out, None, None, ["cuDeviceGet failed"]
    raw_ver = ctypes.c_int()
    _safe(cuda.cuDriverGetVersion, ctypes.byref(raw_ver))
    buf = ctypes.create_string_buffer(128)
    _safe(cuda.cuDeviceGetName, buf, ctypes.sizeof(buf), dev)
    name = buf.value.decode(errors="replace") or None
    wanted = {
      "MULTIPROCESSOR_COUNT": "MULTIPROCESSOR_COUNT",
      "CLOCK_RATE": "CLOCK_RATE",
      "MEMORY_CLOCK_RATE": "MEMORY_CLOCK_RATE",
      "GLOBAL_MEMORY_BUS_WIDTH": "GLOBAL_MEMORY_BUS_WIDTH",
      "L2_CACHE_SIZE": "L2_CACHE_SIZE",
      "MAX_REGISTERS_PER_MULTIPROCESSOR": "MAX_REGISTERS_PER_MULTIPROCESSOR",
      "MAX_SHARED_MEMORY_PER_MULTIPROCESSOR": "MAX_SHARED_MEMORY_PER_MULTIPROCESSOR",
      "MAX_SHARED_MEMORY_PER_BLOCK_OPTIN": "MAX_SHARED_MEMORY_PER_BLOCK_OPTIN",
      "MAX_THREADS_PER_MULTIPROCESSOR": "MAX_THREADS_PER_MULTIPROCESSOR",
      "MAX_BLOCKS_PER_MULTIPROCESSOR": "MAX_BLOCKS_PER_MULTIPROCESSOR",
      "MAX_THREADS_PER_BLOCK": "MAX_THREADS_PER_BLOCK",
      "MAX_REGISTERS_PER_BLOCK": "MAX_REGISTERS_PER_BLOCK",
      "WARP_SIZE": "WARP_SIZE",
      "PCI_BUS_ID": "PCI_BUS_ID",
      "COMPUTE_CAPABILITY_MAJOR": "COMPUTE_CAPABILITY_MAJOR",
      "COMPUTE_CAPABILITY_MINOR": "COMPUTE_CAPABILITY_MINOR",
    }
    for key, attr in wanted.items():
      v = ctypes.c_int()
      if _safe(cuda.cuDeviceGetAttribute, ctypes.byref(v), getattr(cuda, f"CU_DEVICE_ATTRIBUTE_{attr}"),
               dev, default=-1) == 0:
        out[key] = int(v.value)
      else:
        notes.append(f"driver attribute {attr} not readable")
    return out, raw_ver.value, name, notes
  except Exception as e:  # noqa: BLE001
    return out, None, None, [f"driver attribute probe failed: {type(e).__name__}: {e}"]


def _nvidia_smi() -> str | None:
  exe = shutil.which("nvidia-smi")
  if not exe:
    return None
  try:
    p = subprocess.run([exe, "--query-gpu=name,driver_version,compute_cap,memory.total",
                        "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
    return p.stdout.strip().splitlines()[0] if p.returncode == 0 and p.stdout.strip() else None
  except Exception:  # noqa: BLE001
    return None


def _restrict_profiling_param() -> str | None:
  """Read NVreg_RestrictProfilingToAdminUsers from the driver params, if visible.

  This is the single most useful pre-flight check for this study: on a Kaggle container
  the module parameter is readable and already tells you whether `ncu` has a chance,
  before spending a minute discovering it the slow way. Read-only; absent is not an error.
  """
  try:
    with open("/proc/driver/nvidia/params", "r") as f:
      for line in f:
        if line.startswith("NVreg_RestrictProfilingToAdminUsers:"):
          return line.split(":", 1)[1].strip()
  except Exception:  # noqa: BLE001
    pass
  return None


def _default_device() -> str | None:
  """tinygrad's chosen device, without raising if there is no usable device."""
  def go() -> str | None:
    from tinygrad import Device  # noqa: PLC0415
    d = Device.DEFAULT
    d = Device[Device.canonicalize(d)]
    return d.device
  return _safe(go, default=None, what="device selection")


def device_facts(device: str | None = None) -> DeviceFacts:
  """Everything known about the device at hand, from the device. Never raises."""
  facts = DeviceFacts()
  facts.device = device or _default_device() or "unknown"
  if device and device.split(":")[0].upper() in ("CUDA", "NV"):
    attrs, drv, name, notes = _safe(_driver_attributes, 0, default=({}, None, None, []),
                                    what="driver attributes")
    facts.attrs, facts.driver_version, facts.name = attrs, drv, name
    facts.notes += notes
    # The arch string comes from the device itself: the driver's compute capability, or the
    # backend's own arch field if the driver bindings were unreachable. Never inferred.
    cc = _safe(_compute_capability, default=(None, None), what="compute capability")
    facts.arch = cc[1] or _safe(_backend_arch, device, default=None, what="backend arch")
    if facts.arch is None:
      facts.notes.append("device reports no compute capability and no arch: the roofline cannot "
                         "be computed, and this is stated rather than guessed")
    facts.props = _safe(_device_props, device, default={}, what="device_props") or {}
  facts.nvidia_smi = _nvidia_smi()
  facts.restrict_profiling_to_admins = _restrict_profiling_param()
  if facts.arch in ARCH_NOTES:
    facts.notes.append(f"{facts.arch}: {ARCH_NOTES[facts.arch]}")
  return facts


def _compute_capability() -> tuple[int | None, str | None]:
  from tinygrad.runtime.autogen import cuda  # noqa: PLC0415
  dev = ctypes.c_int()
  if cuda.cuInit(0) != 0 or cuda.cuDeviceGet(ctypes.byref(dev), 0) != 0:
    return None, None
  maj, minr = ctypes.c_int(), ctypes.c_int()
  cuda.cuDeviceComputeCapability(ctypes.byref(maj), ctypes.byref(minr), dev)
  return maj.value, f"sm_{maj.value}{minr.value}"


def _backend_arch(device: str) -> str | None:
  """The arch the tinygrad backend reports for itself, when the driver bindings are absent."""
  from tinygrad import Device  # noqa: PLC0415
  a = getattr(Device[device], "arch", None)
  return str(a) if isinstance(a, str) and a.startswith("sm_") else None


def _device_props(device: str) -> dict[str, Any]:
  from tinygrad import Device  # noqa: PLC0415
  return dict(getattr(Device[device], "device_props", lambda: {})() or {})


# --------------------------------------------------------------------------------------
# 4. The roofline
# --------------------------------------------------------------------------------------

@dataclass
class Roofline:
  """Peak numbers for this device, with the provenance of every one of them.

  `flops_s` is fp32. `tensor_flops_s` is deliberately None: the tensor-core peak is not
  derivable from driver attributes, and embedding a vendor table to supply it is exactly
  the hardcoded-device-specific-number this module refuses to do. The study's matmuls
  are fp32 anyway; the tensor question is answered by counters and by the PTX, not by a
  peak.
  """
  arch: str | None = None
  sm_count: int | None = None
  sm_clock_hz: float | None = None
  cores_per_sm: int | None = None
  flops_s: float | None = None
  bytes_s: float | None = None
  tensor_flops_s: float | None = None
  l2_bytes: int | None = None
  ridge_point: float | None = None          # FLOP per byte where the two roofs meet
  flops_source: str = ""
  bytes_source: str = ""
  measured_flops_s: float | None = None     # from calibrate(), if it was run
  measured_bytes_s: float | None = None
  notes: list[str] = field(default_factory=list)

  @property
  def usable(self) -> bool: return self.flops_s is not None or self.bytes_s is not None

  def position(self, flops_s: float | None, bytes_s: float | None) -> dict[str, Any]:
    """Where a measured point sits relative to this device's roofs.

    The naive version of this -- min(flops/peak_flops, bytes/peak_bytes) -- is wrong, and
    wrong in a way that mislabels every kernel: the memory limit for a kernel is
    AI x peak_bytes, not peak_bytes, so dividing achieved bytes by the raw bandwidth compares
    two different quantities. The correct form, and the only one used here:

        AI           = flops / bytes
        memory limit = AI x peak_bytes            (what this kernel could sustain)
        compute limit= peak_flops
        achievable   = min(the two)               (the value at the ridge point)
        efficiency   = achieved / achievable
        bound        = whichever of the two is the min, i.e. AI against the ridge point

    An efficiency above 1.0 is not clamped. It means the peak is wrong or the work is
    miscounted, and both of those are findings, so the number goes out as it came in.

    Every field is None when its input is. In particular `bound` is None, not "memory", when
    there is nothing to place on the roofline.
    """
    out: dict[str, Any] = {"bound": None, "util": None, "flops_util": None, "bytes_util": None,
                           "ai": None, "achievable_s": None, "flops_roof": self.flops_s,
                           "bytes_roof": self.bytes_s, "why": None}
    if flops_s is not None and bytes_s:
      out["ai"] = flops_s / bytes_s
    if flops_s is not None and self.flops_s:
      out["flops_util"] = flops_s / self.flops_s
    mem_limit = out["ai"] * self.bytes_s if (out["ai"] is not None and self.bytes_s) else None
    if out["ai"] is not None and self.bytes_s and flops_s:
      out["bytes_util"] = bytes_s / mem_limit if mem_limit else None

    if flops_s is not None and mem_limit is not None:
      out["achievable_s"] = min(self.flops_s, mem_limit) if self.flops_s else mem_limit
      out["bound"] = "compute" if (self.flops_s is not None and self.flops_s <= mem_limit) else "memory"
      if out["achievable_s"]:
        out["util"] = flops_s / out["achievable_s"]
      if self.ridge_point is not None:
        which = "compute" if out["bound"] == "compute" else "memory"
        out["why"] = (f"AI {out['ai']:.4g} FLOP/B vs ridge {self.ridge_point:.4g} -> "
                      f"{out['bound']}-bound; {out['util']*100:.3g}% of the {which} limit"
                      + ("  *** above 100%: the peak or the work count is wrong ***"
                         if out["util"] and out["util"] > 1 else ""))
    elif flops_s is not None and self.flops_s:
      out["achievable_s"], out["bound"] = self.flops_s, "compute"
      out["util"] = flops_s / self.flops_s
      out["why"] = (f"{out['util']*100:.3g}% of the compute roof. No byte count, so intensity "
                    "is unknown and the classification is one-sided")
    elif bytes_s is not None and self.bytes_s:
      out["achievable_s"], out["bound"] = self.bytes_s, "memory"
      out["util"] = bytes_s / self.bytes_s
      out["why"] = (f"{out['util']*100:.3g}% of the memory roof, taken at face value. No FLOP "
                    "count, so this is a bandwidth number and not a roofline position")
    else:
      missing = []
      if flops_s is None:
        missing.append("no FLOP rate")
      if not bytes_s:
        missing.append("no byte count")
      if flops_s is not None and self.flops_s is None:
        missing.append("no compute roof (unknown arch, and none is guessed)")
      if bytes_s and self.bytes_s is None:
        missing.append("no memory roof")
      out["why"] = "cannot place on the roofline: " + ", ".join(missing or ["nothing to place"])
    return out


def roofline_from_attrs(attrs: dict[str, int], arch: str | None = None,
                        l2_bytes: int | None = None) -> Roofline:
  """Pure function of device attributes. Unit-testable with synthetic attributes.

  Kept separate from `device_facts` on purpose: the formula is the thing most likely to
  be wrong, and a pure function is the only way to test it on a machine with no GPU.
  """
  r = Roofline(arch=arch, l2_bytes=l2_bytes if l2_bytes is not None else attrs.get("L2_CACHE_SIZE"))
  sms, clk_khz, mem_khz, bus = (attrs.get("MULTIPROCESSOR_COUNT"), attrs.get("CLOCK_RATE"),
                                attrs.get("MEMORY_CLOCK_RATE"), attrs.get("GLOBAL_MEMORY_BUS_WIDTH"))
  r.sm_count, r.sm_clock_hz = sms, (clk_khz * 1e3) if clk_khz else None

  # Memory roof: DDR signalling is two transfers per clock, so peak = 2 * clock * bus/8.
  if mem_khz and bus:
    r.bytes_s = 2.0 * mem_khz * 1e3 * (bus / 8.0)
    r.bytes_source = (f"driver: MEMORY_CLOCK_RATE={mem_khz} kHz, GLOBAL_MEMORY_BUS_WIDTH={bus} bit; "
                      f"peak = 2 * clock * bus/8 (DDR transfers both edges)")
  else:
    r.notes.append("no memory roof: driver did not report MEMORY_CLOCK_RATE / GLOBAL_MEMORY_BUS_WIDTH")

  # Compute roof: SMs * fp32 lanes * 2 flops per FMA * clock. Lanes are structural.
  r.cores_per_sm = FP32_CORES_PER_SM.get(arch) if arch else None
  if arch and r.cores_per_sm is None:
    r.notes.append(f"unknown arch {arch!r}: FP32_CORES_PER_SM has no entry, so no compute roof is "
                   "claimed. Reporting a guess here would put every 'percent of peak' on a "
                   "fabricated denominator.")
  if r.cores_per_sm and sms and clk_khz:
    r.flops_s = float(sms) * r.cores_per_sm * 2.0 * clk_khz * 1e3
    r.flops_source = (f"driver: MULTIPROCESSOR_COUNT={sms}, CLOCK_RATE={clk_khz} kHz; "
                      f"structural: {r.cores_per_sm} fp32 cores/SM ({arch}, CUDA C Programming "
                      f"Guide compute-capability table); peak = SMs * cores * 2 flops/FMA * clock")
  elif r.cores_per_sm and not (sms and clk_khz):
    r.notes.append("no compute roof: driver did not report SM count and/or clock rate")
  r.tensor_flops_s = None
  r.notes.append("tensor-core peak is not derivable from driver attributes and is not embedded here "
                 "on purpose; the matrix-path question is answered by counters and PTX, not by a peak")
  if r.flops_s and r.bytes_s:
    r.ridge_point = r.flops_s / r.bytes_s
  return r


def roofline(facts: DeviceFacts | None = None) -> Roofline:
  facts = facts or device_facts()
  r = roofline_from_attrs(facts.attrs, facts.arch, facts.l2_bytes)
  # The NV backend knows its SM count and warp capacity without the driver bindings.
  if r.sm_count is None:
    nv_sms = _safe(_nv_sm_count, facts.device, default=None, what="NV SM count")
    if nv_sms:
      r.sm_count = nv_sms
      r.notes.append(f"SM count {nv_sms} taken from the NV backend's own GPU topology query, "
                     "not from cuDeviceGetAttribute")
  return r


def _nv_sm_count(device: str) -> int | None:
  from tinygrad import Device  # noqa: PLC0415
  d = Device[device]
  parts = [getattr(d, a, None) for a in ("num_gpcs", "num_tpc_per_gpc", "num_sm_per_tpc")]
  if all(isinstance(p, int) for p in parts):
    return int(parts[0] * parts[1] * parts[2])
  return None

# --------------------------------------------------------------------------------------
# 5. Capability: which level did we actually get, and why
# --------------------------------------------------------------------------------------

# THE LEVEL NUMBERS ARE NOT A QUALITY ORDER. Level 3 is the *floor* -- the thing that always
# works -- and level 1 is the *best*. So "min_level" cannot be compared as a number: at level
# 1 (full ncu) every counter is available even though 1 < 2. This table converts a level to a
# monotone rank, and every capability question goes through it. Getting this wrong inverts the
# ladder, so it is one table with one definition rather than an inline comparison.
LEVEL_RANK = {0: 0, 3: 1, 2: 2, 1: 3}

LEVEL_NAMES = {
  0: "no CUDA counters and no CUDA roofline (CPU backend, or a non-CUDA device)",
  1: "event timing + first-principles accounting + full Nsight Compute (source-level counters)",
  2: "event timing + first-principles accounting + targeted Nsight Compute (raw counters)",
  3: "event timing + first-principles accounting + static PTX inspection",
}

# Which timing mechanism each backend actually uses. This is the difference between
# "CUDA events" as a slogan and "CUDA events" as a fact, and it is worth being precise
# about because it is the thing that makes level 3 available with no privileges at all.
TIMING_SOURCE = {
  "CUDA": "cudaEventRecord/cuEventElapsedTime around each launch (driver-API events, no profiler, "
          "no privileges)",
  "NV":   "GPU-written timestamps on the submission queue around each launch (semaphore "
          "timestamps, host-read after sync). Equivalent to events: measured on the device, "
          "needs no profiler, but it serialises: each timed launch is followed by a sync.",
  "METAL": "GPU-written timestamps on the submission queue (semaphore timestamps), the same "
           "mechanism as NV. The measurement is real; only the counters and the CUDA roofline "
           "are out of reach, because they come from the CUDA driver.",
  "AMD":  "GPU-written timestamps on the submission queue, the same mechanism as NV. No CUDA "
          "roofline and no CUDA counters are derivable on this backend.",
  "CPU":  "host wall clock around a synchronised run. Not a device time and not comparable to "
          "any GPU number.",
}


@dataclass
class NCUStatus:
  available: bool = False
  path: str | None = None
  version: str | None = None
  reason: str = "not probed"
  classification: str = "NOT_PROBED"     # OK | NOT_FOUND | PERMISSION | DRIVER_MISMATCH |
                                         # UNSUPPORTED | TIMEOUT | NO_KERNELS | BROKEN | NOT_PROBED
  restrict_profiling_to_admins: str | None = None
  stderr_head: str = ""
  probe_seconds: float | None = None

  def one_line(self) -> str:
    return f"ncu {'OK ' if self.available else 'unavailable'} [{self.classification}] {self.reason}"


@dataclass
class Capability:
  level: int = 0
  level_name: str = LEVEL_NAMES[0]
  device: str = "unknown"
  timing_source: str | None = None
  per_kernel: bool = False             # can this backend time individual launches?
  aggregate: bool = False              # can it time a whole run with one sync? (always True)
  static: bool = False                 # PTX inspection available
  counters: bool = False
  counters_kind: str = "none"          # none | targeted | full
  static_arch: str | None = None       # the arch PTX is rendered for, if any
  ncu: NCUStatus = field(default_factory=NCUStatus)
  reasons: list[str] = field(default_factory=list)
  forced: bool = False                 # dry-run of the reporting logic, not a measurement
  facts: DeviceFacts = field(default_factory=DeviceFacts)
  roofline: Roofline = field(default_factory=Roofline)

  def decidable(self) -> list[Hypothesis]:
    """Hypotheses this capability can decide, and the ones it cannot."""
    yes = [h for h in HYPOTHESES.values() if self._can(h)]
    no = [h for h in HYPOTHESES.values() if not self._can(h)]
    return yes, no

  @property
  def rank(self) -> int:
    """How much measurement this capability obtained, as a monotone number. See LEVEL_RANK."""
    return LEVEL_RANK.get(self.level, 0)

  def _can(self, h: Hypothesis) -> bool:
    if self.rank >= LEVEL_RANK.get(h.min_level, 0):
      return True
    # a hypothesis with a real static probe is also decidable whenever the static channel is up
    return h.has_static and self.static

  def explain(self) -> str:
    lines = [f"level {self.level} (rank {self.rank}/3, higher is more measurement): "
             f"{self.level_name}",
             f"  device            : {self.device}",
             f"  timing            : {self.timing_source or 'none'}",
             f"  per-kernel times  : {self.per_kernel}",
             f"  aggregate times   : {self.aggregate}",
             f"  static PTX probe  : {self.static}"
             + (f" (rendering for {self.static_arch})" if self.static and self.static_arch else ""),
             f"  HW counters       : {self.counters} ({self.counters_kind})",
             f"  ncu               : {self.ncu.one_line()}",
             f"  roofline          : {self.facts.arch or 'unknown arch'}: "
             f"{_si(self.roofline.flops_s, 'FLOP/s')} fp32, {_si(self.roofline.bytes_s, 'B/s')} mem, "
             f"ridge {_si(self.roofline.ridge_point, ' FLOP/B') if self.roofline.ridge_point else 'n/a'}"]
    if self.forced:
      lines.append("  *** FORCED: this is a dry run of the reporting logic, not a measurement ***")
    for r in self.reasons:
      lines.append(f"  - {r}")
    return "\n".join(lines)


def find_ncu() -> str | None:
  """Locate ncu without importing anything. PATH first, then the usual CUDA locations."""
  found = shutil.which("ncu")
  if found:
    return found
  for cand in ("/usr/local/cuda/bin/ncu", "/opt/nvidia/nsight-compute/*/ncu",
               "/usr/local/cuda-*/bin/ncu", "/usr/bin/ncu"):
    hits = sorted(glob_simple(cand))
    if hits:
      return hits[-1]
  return None


def glob_simple(pattern: str) -> list[str]:
  import glob  # noqa: PLC0415
  return [p for p in glob.glob(pattern) if os.access(p, os.X_OK)]


def classify_ncu_failure(stderr: str, returncode: int, timed_out: bool = False) -> tuple[str, str]:
  """Turn ncu's failure into (classification, one-line reason). No guessing."""
  if timed_out:
    return "TIMEOUT", "ncu did not finish within the probe timeout; replay cost too high for this container"
  low = (stderr or "").lower()
  table = [
    ("nvgpuctrperm", "PERMISSION", "driver refused the profiling counters: NVreg_RestrictProfilingToAdminUsers "
                                     "must be 0 (and often CAP_SYS_ADMIN / the nvidia container runtime hook)"),
    ("permission denied", "PERMISSION", "profiling permission denied to this user"),
    ("operation not permitted", "PERMISSION", "profiling not permitted in this container"),
    ("access is denied", "PERMISSION", "access denied while opening the profiling interface"),
    ("not found", "NOT_FOUND", "ncu or its helper binaries are missing from this image"),
    ("no such file", "NOT_FOUND", "ncu or its helper binaries are missing from this image"),
    ("cuda_error_insufficient_driver", "DRIVER_MISMATCH", "ncu is newer than the installed driver"),
    ("insufficient driver", "DRIVER_MISMATCH", "ncu is newer than the installed driver"),
    ("cuda error", "DRIVER_MISMATCH", "ncu could not initialise against this driver/GPU"),
    ("not supported", "UNSUPPORTED", "this metric or this GPU is not supported by this ncu"),
    ("failed to load", "BROKEN", "ncu could not load; a partial CUDA install is the usual cause"),
    ("segmentation", "BROKEN", "ncu crashed"),
  ]
  for needle, cls, why in table:
    if needle in low:
      return cls, why
  if returncode != 0:
    return "UNKNOWN", f"ncu exited {returncode} with an unrecognised message"
  return "UNKNOWN", "ncu exited 0 but no kernel metrics were reported"


def probe_ncu(enable: bool = True, timeout: int = 180) -> NCUStatus:
  """Decide whether level 2 or 1 is reachable. Cheap first, then a real tiny run.

  Order matters. `--version` proves the binary exists. The kernel-count check reads one
  driver parameter. Only then do we spend a subprocess on an actual profiled launch,
  because that is the only test that can distinguish "ncu is installed" from "ncu is
  installed and permitted", which is the whole question.
  """
  st = NCUStatus(restrict_profiling_to_admins=_restrict_profiling_param())
  st.path = find_ncu()
  if not st.path:
    st.classification, st.reason = "NOT_FOUND", "no ncu binary on PATH or in the usual CUDA locations"
    return st
  st.version = _safe(_first_line, [st.path, "--version"], default=None, what="ncu --version")
  if st.version is None:
    st.classification, st.reason = "BROKEN", f"{st.path} --version produced no output"
    return st
  if not enable:
    st.classification, st.reason = "NOT_PROBED", "capability probe disabled by caller"
    return st

  # A driver parameter of 1 means admin-only profiling: ncu cannot work as this user.
  if st.restrict_profiling_to_admins is not None and st.restrict_profiling_to_admins.split()[0] == "1":
    st.classification = "PERMISSION"
    st.reason = ("NVreg_RestrictProfilingToAdminUsers=1 in /proc/driver/nvidia/params: hardware "
                 "counters are restricted to admin, so ncu cannot collect them as this user")
    return st

  t0 = time.perf_counter()
  cmd = [st.path, "--target-processes", "all", "--metrics", ",".join(PROBE_METRICS),
         "--launch-count", "1", "--", sys.executable, "-c",
         "from tinygrad import Tensor; (Tensor([1.0,2.0])*Tensor([3.0,4.0])).realize()"]
  try:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    st.probe_seconds = time.perf_counter() - t0
    st.stderr_head = _tail(p.stderr or p.stdout or "")
    cls, why = classify_ncu_failure(p.stderr, p.returncode, timed_out=False)
    got = bool(re.search(r"gpu__time_duration|sm__cycles_elapsed", p.stdout or ""))
    if p.returncode == 0 and got:
      st.available, st.classification = True, "OK"
      st.reason = (f"ncu profiled one kernel and reported the probe metrics in "
                   f"{st.probe_seconds:.1f}s: hardware counters are reachable at this level")
    elif p.returncode == 0:
      st.classification, st.reason = "NO_KERNELS", why
    else:
      st.classification, st.reason = cls, why
  except subprocess.TimeoutExpired:
    st.probe_seconds = time.perf_counter() - t0
    st.classification, st.reason = classify_ncu_failure("", -1, timed_out=True)
  except Exception as e:  # noqa: BLE001
    st.classification, st.reason = "BROKEN", f"{type(e).__name__}: {e}"
  return st


def _first_line(argv: list[str]) -> str | None:
  p = subprocess.run(argv, capture_output=True, text=True, timeout=30)
  out = (p.stdout or p.stderr or "").strip()
  return out.splitlines()[0][:200] if out else None


def _tail(text: str, n: int = 4, width: int = 160) -> str:
  lines = [l.strip() for l in text.splitlines() if l.strip()]
  return " | ".join(l[:width] for l in lines[-n:])


def describe_capability(device: str | None = None, probe: bool = True, force_level: int | None = None,
                        static_arch: str | None = None) -> Capability:
  """Report the level actually obtained, and the reason for every part of it.

  `static_arch` is the arch to render PTX for. It defaults to the device's own arch, and on
  a machine with no GPU it can be set to a target the study cares about (`PROF_ARCH=sm_75`),
  because the static channel needs a codegen target and nothing else. That is what lets the
  vectorisation and matrix-path hypotheses be tested on a laptop: the *code* is the same
  code, and the arch is a parameter of the code generator.

  `force_level` re-runs the *reporting* at a level this device did not achieve, for
  checking the degradation ladder itself. It changes no number and is flagged as forced in
  the output, because a capability report that claims a level it did not reach is exactly
  the failure mode this module exists to prevent.
  """
  cap = Capability()
  facts = device_facts(device)
  cap.facts, cap.device = facts, facts.device
  cap.roofline = roofline(facts)
  arch = static_arch or facts.arch or (os.environ.get("PROF_ARCH") or None)
  cap.static_arch = arch

  if facts.is_cuda:
    cap.level = 3
    cap.level_name = LEVEL_NAMES[3]
    cap.timing_source = TIMING_SOURCE.get("NV" if facts.device.startswith("NV") else "CUDA")
    cap.per_kernel = True
    cap.aggregate = True
    cap.static = _static_available(facts)
    if not cap.static:
      cap.reasons.append("static PTX probe unavailable: no PTX renderer, or no arch to render for "
                         "(set PROF_ARCH=sm_75 to render for a target other than this device)")
    if probe:
      cap.ncu = probe_ncu()
      if cap.ncu.available:
        cap.level, cap.level_name = 2, LEVEL_NAMES[2]
        cap.counters, cap.counters_kind = True, "targeted"
        cap.reasons.append(f"ncu reachable ({cap.ncu.version or 'version unknown'}), so the targeted "
                           "raw-counter set is available; --set full (level 1) costs 10-100x more "
                           "replay and is not attempted automatically")
      else:
        cap.reasons.append(f"level 2/1 unavailable: {cap.ncu.one_line()}")
  else:
    cap.level, cap.level_name = 0, LEVEL_NAMES[0]
    cap.timing_source = TIMING_SOURCE.get(facts.device.split(":")[0].upper(),
                                          "no device timing")
    cap.per_kernel = _host_per_kernel_possible(facts)
    cap.aggregate = True
    cap.counters, cap.counters_kind = False, "none"
    cap.reasons.append(
        f"device is {facts.device!r}, not CUDA: no CUDA roofline can be derived here (the peaks "
        "come from cuDeviceGetAttribute) and no CUDA counter path exists, so the level is 0. "
        "Device-side timings are still recorded where the backend provides them, and are "
        "labelled as such; on the CPU backend they are host time and worthless as a GPU "
        "result, which is why the level does not claim otherwise.")
    cap.reasons.append("level 2/1 cannot exist without an NVIDIA GPU and a profiler")
    # The static channel does not need a GPU, so it stays available at level 0. It is the
    # only evidence a CPU machine can produce about the two most important hypotheses.
    cap.static = _static_available(facts) and bool(arch)
    if cap.static:
      cap.reasons.append(f"static PTX probe IS available and will render for {arch}: it needs a "
                         "codegen target, not a GPU. M1_COALESCE (vectorisation), M2_MMA "
                         "(matrix path), M3_OCCUPANCY (spills), M6_MATERIALISE, M7_PRECISION and "
                         "M8_SMEM are therefore still decidable at level 0 -- statically, and "
                         "about the code rather than about this machine's execution")
    else:
      cap.reasons.append("static PTX probe unavailable: set PROF_ARCH=sm_75 (or pass static_arch=) "
                         "to render PTX for a target GPU without having that GPU")

  if force_level is not None:
    cap.level, cap.level_name, cap.forced = int(force_level), LEVEL_NAMES.get(int(force_level), "?"), True
    cap.counters, cap.counters_kind = (True, "full" if force_level == 1 else
                                       "targeted" if force_level == 2 else "none")
    cap.reasons.append(f"forced to level {force_level} for a dry run of the reporting logic")
  return cap


def _static_available(facts: DeviceFacts) -> bool:
  """Can the PTX renderer be constructed at all? Needs no GPU, only tinygrad."""
  try:
    from tinygrad.renderer.ptx import PTXRenderer  # noqa: F401,PLC0415
  except Exception:  # noqa: BLE001
    return False
  return True


def _host_per_kernel_possible(facts: DeviceFacts) -> bool:
  """True when the backend can return a time for an individual launch, on any device.

  On the CPU backend the answer is yes (the host-queue timestamps are host clock reads),
  which is why the local validation can exercise the whole per-kernel record path. The
  values are host time; `report()` labels them.
  """
  try:
    from tinygrad import Device  # noqa: PLC0415
    d = Device[facts.device]
    return isinstance(getattr(d, "runtime_t", None), type) or hasattr(d, "new_signal")
  except Exception:  # noqa: BLE001
    return False

# --------------------------------------------------------------------------------------
# 6. Records
# --------------------------------------------------------------------------------------

@dataclass
class Work:
  """One thing to measure.

  `build()` must return an *unrealized* Tensor, and it is called several times: once to plan,
  once for the cold plan timing and twice more for the steady-state one, because the
  difference between those two is a number this study cares about. Execution then replays
  the already-compiled linear, so `build()` is not called per timed repetition.

  Returning an unrealized Tensor matters: `Tensor.realize()` on an already-realized graph is
  a silent no-op, so timing a prebuilt expression measures nothing and reads as ~3ns.

  `flops` and `bytes` are the caller's hand-computed *minimum* -- the work and the traffic
  an ideal kernel is obliged to do. They are the denominator that decides the roofline, so
  they are the study's claim, not the instrument's. `materialised_bytes`, when given, is the
  size of the intermediate a naive graph builds instead; the ratio to `bytes` is the
  M6_MATERIALISE measurement.
  """
  name: str
  build: Callable[[], Any]
  flops: float | None = None
  bytes: float | None = None
  materialised_bytes: float | None = None
  note: str = ""
  # True when build() already realizes (most of the existing probes are written that way).
  # The aggregate device time is then measurable and the per-kernel breakdown is not, because
  # the schedule is internal to the call. That asymmetry is reported, not hidden.
  build_and_realize: bool = False


@dataclass
class StaticFacts:
  """What the generated code says, with no hardware involved.

  This is level 3's second channel and the only one that works on a machine with no GPU,
  which is why it exists. It is *static*: it describes the instructions that were emitted,
  not the execution. Where a counter and this disagree, the counter is the arbiter and the
  disagreement is worth reporting.
  """
  kernel: str
  arch: str | None = None
  mma_sync: int | None = None
  wmma: int | None = None
  wgmma: int | None = None
  mma_tf32: int | None = None
  mma_f16: int | None = None
  ffma: int | None = None
  ld_global: dict[str, int] = field(default_factory=dict)   # width -> count
  ld_global_scalar: int | None = None
  ld_global_vector: int | None = None
  widest_ld_bytes: int | None = None
  vectorised_load_frac: float | None = None
  ld_shared: int | None = None
  st_shared: int | None = None
  cp_async: int | None = None
  local_ld: int | None = None
  local_st: int | None = None
  local_decl: int | None = None
  bar_sync: int | None = None
  reg_decls: int | None = None
  shared_bytes_declared: int | None = None
  note: str = ""
  available: bool = True

  def summary(self) -> str:
    if not self.available:
      return "static: n/a (" + (self.note or "unavailable") + ")"
    mma = (self.mma_sync or 0) + (self.wmma or 0) + (self.wgmma or 0)
    n = lambda v: "n/a" if v is None else str(v)  # noqa: E731
    tf32 = n(self.mma_tf32) if mma else "n/a (no MMA to classify)"
    return (f"mma={mma} (tf32={tf32}, f16={n(self.mma_f16)}) ffma_insts={n(self.ffma)} "
            f"ld.global={self.ld_global or '{}'} local_ld/st={n(self.local_ld)}/{n(self.local_st)} "
            f"shared_ld/st={n(self.ld_shared)}/{n(self.st_shared)} cp.async={n(self.cp_async)} "
            f"bar.sync={n(self.bar_sync)} reg_decls={n(self.reg_decls)}")


@dataclass
class KernelRecord:
  index: int
  kernel: str
  kind: str = "program"              # program | copy | other
  device: str = ""
  ms: float | None = None            # device time for this launch, serialised (level 3)
  flops_est: int | None = None       # tinygrad's first-principles op count for this kernel
  bytes_est: int | None = None      # distinct bytes this kernel touches, capped at buffer size
  lds_est: int | None = None         # load+store bytes including re-reads
  in_bytes: int | None = None        # ground truth: sum of input buffer sizes
  out_bytes: int | None = None
  buffers: list[tuple[str, int]] = field(default_factory=list)
  achieved_gbs: float | None = None
  achieved_tflops: float | None = None
  arithmetic_intensity: float | None = None   # flops / distinct bytes
  roofline: dict[str, Any] = field(default_factory=dict)
  static: StaticFacts | None = None
  ms_wall: float | None = None       # host wall around this one launch, as a cross-check
  timing_untrusted: bool = False     # set when the per-kernel channel failed its wall check
  counters: dict[str, float | None] = field(default_factory=dict)
  counters_missing: list[str] = field(default_factory=list)
  note: str = ""


@dataclass
class OpRecord:
  """The aggregate. This is the number you quote; the per-kernel table is the evidence."""
  name: str
  level: int
  device: str
  gpu_ms: float | None = None            # aggregate device time, one sync around the run
  gpu_ms_per_kernel_sum: float | None = None   # sum of serialised per-kernel times (>= gpu_ms)
  plan_ms: float | None = None           # steady-state schedule cost, kernels already cached
  plan_ms_cold: float | None = None      # includes compiling the kernels; the first call
  wall_ms: float | None = None           # build + plan + execute, host
  bytes_min: float | None = None
  bytes_est: int | None = None
  bytes_mat: float | None = None         # caller's materialised_bytes
  copy_bytes: int | None = None          # bytes moved by H2D/D2H copies, reported apart
  flops_min: float | None = None
  flops_est: int | None = None
  achieved_gbs: float | None = None
  achieved_tflops: float | None = None
  arithmetic_intensity: float | None = None
  roofline: dict[str, Any] = field(default_factory=dict)
  n_kernels: int | None = None        # None when the workload realizes internally: unknown, not 0
  n_measured: int | None = None
  n_truncated: int = 0
  per_kernel_phase_wall_ms: float | None = None   # wall time of the individually-timed phase
  per_kernel_trust: str = "not measured"          # ok | suspect | not measured
  per_kernel_ratio: float | None = None            # sum(et) / phase wall
  reps: int = 0
  note: str = ""

  @property
  def bytes_ratio(self) -> float | None:
    if self.bytes_min and self.bytes_est is not None:
      return self.bytes_est / self.bytes_min
    return None

  @property
  def flops_ratio(self) -> float | None:
    if self.flops_min and self.flops_est is not None:
      return self.flops_est / self.flops_min
    return None

  @property
  def mat_ratio(self) -> float | None:
    if self.bytes_mat and self.bytes_est is not None:
      return self.bytes_est / self.bytes_mat
    return None


@dataclass
class Measurement:
  work: Work
  op: OpRecord
  kernels: list[KernelRecord] = field(default_factory=list)
  capability: Capability = field(default_factory=Capability)
  warnings: list[str] = field(default_factory=list)
  error: str | None = None


# --------------------------------------------------------------------------------------
# 7. The static channel: render PTX for the target arch and read the instructions
# --------------------------------------------------------------------------------------

# PTX memory operands are dotted token lists: `ld.global.nc.v4.f32` = non-coherent, 4-element
# vector, fp32. The width is a token in the middle, not the last component, so the operand is
# tokenised rather than split-and-take-the-tail.
_LD_GLOBAL = re.compile(r"\bld\.global((?:\.[A-Za-z0-9_:]+)+)")
_ST_GLOBAL = re.compile(r"\bst\.global((?:\.[A-Za-z0-9_:]+)+)")
_VEC_TOK = re.compile(r"^v(\d+)$")
_ITEMSIZE = {"f16": 2, "bf16": 2, "s16": 2, "u16": 2, "h16": 2, "f32": 4, "s32": 4, "u32": 4,
             "i32": 4, "b32": 4, "f64": 8, "s64": 8, "u64": 8, "i64": 8, "b64": 8,
             "u8": 1, "s8": 1, "i8": 1, "b8": 1, "b16": 2, "pred": 1}


def _ld_widths(text: str) -> tuple[dict[str, int], int, int, int | None]:
  """(per-token counts, scalar instructions, vector instructions, widest bytes per thread)."""
  widths: dict[str, int] = {}
  scalar = vector = 0
  widest: int | None = None
  for rx, is_store in ((_LD_GLOBAL, False), (_ST_GLOBAL, True)):
    del is_store
    for m in rx.finditer(text):
      toks = m.group(1).split(".")[1:]
      vtok = next((t for t in toks if _VEC_TOK.match(t)), None)
      ty = next((t for t in reversed(toks) if t in _ITEMSIZE), "b32")
      its = _ITEMSIZE.get(ty, 4)
      if vtok is not None:
        n = int(_VEC_TOK.match(vtok).group(1))          # elements per thread
        key = f"{vtok}.{ty}"
        widths[key] = widths.get(key, 0) + 1
        vector += 1
        widest = max(widest or 0, n * its)
      else:
        widths[f"scalar.{ty}"] = widths.get(f"scalar.{ty}", 0) + 1
        scalar += 1
        widest = max(widest or 0, its)
  return widths, scalar, vector, widest


def static_facts(linear: Any, arch: str | None, device: str = "CUDA") -> list[StaticFacts | None]:
  """Render every SINK in `linear` to PTX for `arch` and count the instructions.

  Returns one entry per call in `linear.src`, in order, None where the call is not a
  kernel -- so the result aligns with the compiled call list by *index*, not by name. That
  matters: the SINK's `arg.name` is still the placeholder "test" and the real kernel name is
  minted later by ProgramInfo.from_sink, and it differs between renderers (the local-size
  digits in it depend on the target). Joining on the name would silently attach nothing.

  Works with no GPU present (the PTX renderer is pure Python string generation), which is
  how the two most important hypotheses get tested on a laptop. A failure anywhere yields a
  record with available=False and a stated reason, never an invented count.
  """
  out: list[StaticFacts | None] = [None] * len(getattr(linear, "src", ()))
  if not arch:
    return [StaticFacts("<none>", arch=None, available=False,
                        note="no arch known, so no target to render PTX for")]
  try:
    from tinygrad.helpers import Target, ansistrip                      # noqa: PLC0415
    from tinygrad.codegen import to_program                             # noqa: PLC0415
    from tinygrad.renderer.ptx import PTXRenderer                        # noqa: PLC0415
    from tinygrad.runtime.support.compiler_cuda import PTXCompiler       # noqa: PLC0415
    from tinygrad.uop.ops import Ops                                    # noqa: PLC0415
  except Exception as e:  # noqa: BLE001
    return [StaticFacts("<none>", arch=arch, available=False,
                        note=f"PTX renderer unavailable: {type(e).__name__}: {e}")]
  try:
    # The target device is always "CUDA" even when the graph we are inspecting lives on
    # another backend: PTXRenderer picks its compiler by target device, and the alternative
    # (NVPTXCompiler) shells out to a docker image on macOS and to nvJitLink on Linux --
    # neither of which is a measurement, and one of which is a large surprise. PTXCompiler
    # is pure Python: it stamps .target/.version and returns the source. Then it is pinned
    # explicitly, so this function cannot spawn a process whatever the target says.
    renderer = PTXRenderer(Target(device="CUDA", renderer="PTX", arch=arch))
    if type(renderer.compiler) is not PTXCompiler:
      renderer.compiler = PTXCompiler(arch)
  except Exception as e:  # noqa: BLE001
    return [StaticFacts("<none>", arch=arch, available=False,
                        note=f"PTXRenderer construction failed: {type(e).__name__}: {e}")]
  for i, call in enumerate(getattr(linear, "src", ())):
    ast = call.src[0] if getattr(call, "src", None) else None
    if ast is None or ast.op is not Ops.SINK:
      continue
    rec = StaticFacts("<unrendered>", arch=arch)
    try:
      ptx = to_program(ast, renderer)
      raw = ptx.src[-1].arg
      text = raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)
      rec.kernel = ansistrip(getattr(ptx.arg, "name", "<unnamed>"))
      _fill_static(rec, text)
    except Exception as e:  # noqa: BLE001
      rec.available, rec.note = False, f"PTX render failed: {type(e).__name__}: {e}"
    out[i] = rec
  return out


def _fill_static(rec: StaticFacts, text: str) -> None:
  """Count the instructions that carry the hypotheses. Every field is a real count."""
  rec.mma_sync = len(re.findall(r"\bmma\.sync", text))
  rec.wmma = len(re.findall(r"\bwmma\.mma", text))
  rec.wgmma = len(re.findall(r"\bwgmma\.mma", text))
  if rec.mma_sync:
    rec.mma_tf32 = len(re.findall(r"mma\.sync[^\n;]*\.tf32\.", text))
    rec.mma_f16 = len(re.findall(r"mma\.sync[^\n;]*\.f16\.", text))
  rec.ffma = len(re.findall(r"\bfma\.rn\.f32", text))
  widths, scalar, vector, widest = _ld_widths(text)
  rec.ld_global, rec.ld_global_scalar, rec.ld_global_vector = widths, scalar, vector
  rec.widest_ld_bytes = widest
  tot = scalar + vector
  rec.vectorised_load_frac = (vector / tot) if tot else None
  # Zero is a finding here, not a missing value: the counts are taken by reading the whole
  # generated source, so a zero is a fact about the code. `None` is reserved for "the probe
  # did not run", which is set by the caller, never by an absent match.
  rec.ld_shared = len(re.findall(r"\bld\.shared", text))
  rec.st_shared = len(re.findall(r"\bst\.shared", text))
  rec.cp_async = len(re.findall(r"\bcp\.async", text))
  rec.local_ld = len(re.findall(r"\bld\.local", text))
  rec.local_st = len(re.findall(r"\bst\.local", text))
  rec.local_decl = len(re.findall(r"\.local\s+\.", text))
  rec.bar_sync = len(re.findall(r"\bbar\.sync", text))
  m = re.search(r"\.shared\.align\s+(\d+)", text)
  rec.shared_bytes_declared = int(m.group(1)) if m else None
  # `.reg .f32 %reg_f32_<16>;` declares 16 registers. Summing the counts on `.reg` lines is
  # the *declared* register total, which is an upper bound on what ptxas will allocate --
  # launch__registers_per_thread (level 2) is the real number. Not interchangeable.
  rec.reg_decls = sum(int(n) for line in text.splitlines() if ".reg" in line
                      for n in re.findall(r"<(\d+)>", line))


# --------------------------------------------------------------------------------------
# 8. Measurement (level 3)
# --------------------------------------------------------------------------------------

def measure(work: Work, *, reps: int = 3, warmup: int = 1, per_kernel: bool = True,
            max_kernels: int | None = None, static: bool = True, capability: Capability | None = None,
            progress: Callable[[str], None] | None = None) -> Measurement:
  """Measure one Work. Never raises: a failure becomes `error` plus a level-0 record.

  Three timings are taken, because they answer different questions and the study's own
  notes are emphatic that conflating them gives the wrong answer in both directions:

    plan_ms  schedule_linear + compile. Python and the scheduler. On the KDA port this is
             ~40x the device time, and it is a tinygrad property, not a kernel property.
    gpu_ms   the run itself, with one synchronisation around it. This is the number to quote.
    gpu_ms_per_kernel_sum
             the sum of individually timed launches. Each timed launch is followed by a
             synchronisation on the NV path, so this is an UPPER BOUND on gpu_ms and it
             inflates with kernel count. Use it for attribution, never for throughput.
  """
  cap = capability or describe_capability(probe=False)
  op = OpRecord(name=work.name, level=cap.level, device=cap.device)
  m = Measurement(work=work, op=op, capability=cap)
  say = progress or (lambda _s: None)
  n_before = len(LAST_WARNINGS)

  try:
    Tensor, Device, Target, ansistrip, prod, Ops, UOp, to_program = _tg()
  except Exception as e:  # noqa: BLE001
    m.error = f"tinygrad import failed: {type(e).__name__}: {e}"
    op.note = "no measurement possible: tinygrad could not be imported"
    return m

  # ---- mode: the workload realizes internally (an existing probe, unchanged) ------------
  if work.build_and_realize:
    return _measure_realized(m, work, cap, Device, say)

  # ---- phase A: plan + compile (timed separately; this is the scheduler cost) ----------
  try:
    lin = _safe(_plan, work, what="schedule+compile")
    if lin is None:
      raise RuntimeError("work.build() produced nothing to schedule")
  except Exception as e:  # noqa: BLE001
    m.error = f"planning failed: {type(e).__name__}: {e}"
    op.note = "no measurement possible: the graph could not be planned or compiled"
    return m

  # Cold includes compiling the kernels; steady-state is the per-call re-planning cost that
  # remains once the kernels are cached. notes/03-kda.md is entirely about the gap between
  # them, so both are reported and neither is called "the" plan time.
  op.plan_ms_cold = _safe(_time_call, _compile_only, work, default=None, what="plan timing (cold)")
  warm = [_safe(_time_call, _compile_only, work, default=None, what="plan timing (warm)")
          for _ in range(2)]
  warm = [w for w in warm if w is not None]
  op.plan_ms = min(warm) if warm else op.plan_ms_cold

  from tinygrad.engine.realize import (ExecContext, compile_linear, link_linear,   # noqa: PLC0415
                                      estimate_uop, pm_exec, resolve_params)

  cl = _safe(link_linear, _safe(compile_linear, lin, what="compile"), what="link")
  if cl is None:
    m.error = "compilation failed"
    op.note = "no measurement possible: kernels did not compile"
    return m

  # ---- static channel, from the pre-compile SINKs --------------------------------------
  m_static: list[StaticFacts | None] = []
  if static and cap.static:
    m_static = static_facts(lin, cap.static_arch, cap.device if cap.device != "unknown" else "CUDA")
    for rec in m_static:
      if rec is not None and not rec.available:
        m.warnings.append(f"static PTX probe: {rec.note}")
  elif static:
    m.warnings.append("static PTX probe skipped: no arch to render for. Set PROF_ARCH=sm_75 (or "
                      "pass static_arch=) to inspect the generated code for a target GPU.")

  # ---- per-kernel records: static estimates first (no device needed) ------------------
  calls = list(getattr(cl, "src", ()))
  records: list[KernelRecord] = []
  for i, call in enumerate(calls):
    kr = _record_from_call(i, call, Ops, estimate_uop, resolve_params, ansistrip, prod)
    records.append(kr)
  op.n_kernels = sum(1 for r in records if r.kind == "program")
  op.n_measured = 0
  for i, r in enumerate(records):
    r.static = m_static[i] if i < len(m_static) else None
    if r.static is not None and r.static.available and r.static.kernel != r.kernel:
      r.static.note = (r.static.note + "; " if r.static.note else "") + (
          f"rendered name {r.static.kernel!r} differs from the device name {r.kernel!r}: the PTX "
          "renderer targets CUDA and may pick different launch dims. Joined by position.")
  # Copies are transfers, not kernel traffic: they are reported separately so the bytes and
  # FLOPs totals mean "moved by kernels", which is what the roofline denominator refers to.
  progs = [r for r in records if r.kind == "program"]
  op.bytes_est = sum(r.bytes_est or 0 for r in progs) or None
  op.flops_est = sum(r.flops_est or 0 for r in progs) or None
  op.copy_bytes = sum(r.out_bytes or 0 for r in records if r.kind == "copy") or None
  op.bytes_min, op.flops_min, op.bytes_mat = work.bytes, work.flops, work.materialised_bytes
  if op.bytes_min is not None and op.bytes_mat is not None:
    m.warnings.append(f"caller's materialised_bytes is {op.bytes_mat/op.bytes_min:.4g}x the minimum; "
                      "a measured bytes_ratio near that value confirms the intermediate is real traffic")

  # ---- phase B: per-kernel device timing -----------------------------------------------
  if per_kernel and cap.per_kernel:
    ctx = ExecContext(wait=True, update_stats=False)
    keep = records if max_kernels is None else records[:max_kernels]
    op.n_truncated = max(0, len(records) - len(keep))
    tot = 0.0
    n = 0
    t_phase = time.perf_counter()
    for r in keep:
      say(f"  kernel {r.index}: {r.kernel}")
      ets = _safe(pm_exec.rewrite, calls[r.index], ctx, default=None, what="kernel launch")
      r.ms = _single_et(ets)
      if len([e for e in (ets or []) if e is not None]) > 1:
        r.note = (r.note + "; " if r.note else "") + (
                  f"call bundles {len(ets)} kernels; per-kernel attribution needs PROFILE=1")
      if r.ms is not None:
        tot += r.ms
        n += 1
    op.per_kernel_phase_wall_ms = (time.perf_counter() - t_phase) * 1e3
    # Spot-check the largest kernel: re-run it alone, wall-bracketed, and compare. The device
    # clock and the host clock are different instruments; when they disagree by orders of
    # magnitude the device-side number is not a duration, and reporting it as one would be
    # worse than reporting nothing. (This is not hypothetical: the CPU backend returns a
    # sub-microsecond "duration" for a millisecond-long kernel.)
    big = max((r for r in keep if r.kind == "program" and r.ms is not None),
              key=lambda r: r.bytes_est or 0, default=None)
    if big is not None:
      dev = _safe(_first_device, cl, Device, default=None, what="device for cross-check")
      big.ms_wall = _safe(_wall_ms, calls[big.index], pm_exec, ExecContext, dev,
                          what="per-kernel wall cross-check")
      if big.ms_wall is None and dev is None:
        big.ms_wall = None
        big.note = (big.note + "; " if big.note else "") + (
            "no device could be resolved for the wall cross-check, so it was not performed "
            "rather than performed badly")
      big.note = (big.note + "; " if big.note else "") + (
          f"wall-bracketed re-run of this kernel alone: {_fms(big.ms_wall)} (device said "
          f"{_fms(big.ms)}); the device-side value includes its own launch and timestamp "
          "overhead, so it should be the larger of the two, not the smaller by 100x")
    for r in keep:
      _finish_record(r, cap, op)
    op.gpu_ms_per_kernel_sum = tot if n else None
    op.n_measured = n
    op.per_kernel_ratio = (tot / op.per_kernel_phase_wall_ms
                           if tot and op.per_kernel_phase_wall_ms else None)
    if op.per_kernel_ratio is None:
      op.per_kernel_trust = "not measured (no device times returned)"
    elif op.per_kernel_ratio < 0.2:
      op.per_kernel_trust = "SUSPECT"
      m.warnings.append(
          f"per-kernel device times account for only {op.per_kernel_ratio*100:.3g}% of the wall "
          f"time of the phase that produced them ({_fms(tot)} of "
          f"{_fms(op.per_kernel_phase_wall_ms)}). On this backend the returned per-launch time "
          "is not a kernel duration, so the per-kernel ms column must not be used. The "
          "aggregate (one sync around the whole run) is the number to quote.")
    elif op.per_kernel_ratio > 3:
      op.per_kernel_trust = "SUSPECT (device times exceed wall time)"
      m.warnings.append(f"sum of per-kernel device times ({_fms(tot)}) exceeds the wall time of the "
                        f"phase ({_fms(op.per_kernel_phase_wall_ms)}) by "
                        f"{op.per_kernel_ratio:.3g}x, which should be impossible")
    else:
      op.per_kernel_trust = "ok (device times are consistent with wall time)"
    if op.per_kernel_trust.startswith("SUSPECT"):
      # Keep the raw value in the record for forensics, but mark it: a quotable number in a
      # table is more dangerous than a missing one, and the JSON keeps everything.
      for r in records:
        if r.ms is not None:
          r.timing_untrusted = True

  # ---- phase C: aggregate device time, the number to quote ------------------------------
  # Best of `reps` runs with one synchronisation each, after `warmup` discarded runs. Best-of
  # rather than mean, because the thing being rejected is scheduler and clock noise, and the
  # same choice probes/bench.py makes. The per-kernel phase above is deliberately single-shot:
  # timing every launch means a synchronisation after every launch, which changes the cache
  # and pipeline state, so a repeated per-kernel phase would be measuring a different program.
  t_wall0 = time.perf_counter()
  agg = None
  for i in range(max(1, warmup) + max(1, reps)):
    ms = _safe(_aggregate_ms, cl, pm_exec, ExecContext, Device, what="aggregate timing")
    if ms is not None and i >= warmup:
      agg = ms if agg is None else min(agg, ms)
  op.gpu_ms = agg
  # Wall includes graph build, so it is the number a training loop would pay if it re-planned
  # every step. On the KDA port this is ~40x the device time, and the difference is the
  # finding (see notes/03-kda.md), not a measurement error.
  op.wall_ms = (time.perf_counter() - t_wall0) * 1e3 + (op.plan_ms or 0.0)

  for r in records:
    if r.ms is None and r.static is None and r.kind != "program":
      r.note = (r.note + "; " if r.note else "") + "no device time and no static evidence for this call"
    _finish_record(r, cap, op)

  # ---- phase D: the derived rates -------------------------------------------------------
  t = (op.gpu_ms or op.gpu_ms_per_kernel_sum)
  t_s = t / 1e3 if t else None
  op.arithmetic_intensity = (op.flops_min / op.bytes_min) if (op.flops_min and op.bytes_min) else None
  if t_s:
    if op.bytes_min:
      op.achieved_gbs = op.bytes_min / t_s / 1e9
    if op.flops_min:
      op.achieved_tflops = op.flops_min / t_s / 1e12
  op.roofline = cap.roofline.position(op.achieved_tflops * 1e12 if op.achieved_tflops else None,
                                      op.achieved_gbs * 1e9 if op.achieved_gbs else None)
  op.reps = reps
  if cap.level == 0:
    op.note = (op.note + "; " if op.note else "") + (
        "level 0: no CUDA roofline and no counters, so the rates below have no denominator to be "
        "a fraction of")
  op.note = (op.note + "; " if op.note else "") + (work.note or "")

  m.kernels = records
  m.warnings += LAST_WARNINGS[n_before:]
  if cap.ncu.restrict_profiling_to_admins:
    m.warnings.append(f"NVreg_RestrictProfilingToAdminUsers="
                      f"{cap.ncu.restrict_profiling_to_admins}: hardware counters are admin-only on "
                      "this host, so level 2/1 is out of reach without a driver change")
  return m


def _measure_realized(m: Measurement, work: Work, cap: Capability, Device, say) -> Measurement:
  """Time a workload that realizes itself: aggregate only, and say what is missing.

  Most of the probes in probes/ and kda/ are written as `build(); .realize()` or return
  numpy, so this is how the instrument wraps them without editing them. What is lost is the
  per-kernel attribution and the plan/kernel split, and both losses are stated.
  """
  op = m.op
  op.note = ("the workload realizes internally, so there is no per-kernel attribution, no "
             "kernel count and no plan/kernel split here -- all three are unknown, not zero. "
             "Use PROFILE=1 and tinygrad's profile events if the per-kernel breakdown of such a "
             "workload is needed.")
  try:
    for _ in range(max(1, m.op.reps or 1)):
      work.build()                                   # warm up: compile, then time
  except Exception as e:  # noqa: BLE001
    m.error = f"workload raised: {type(e).__name__}: {e}"
    return m
  dev = _safe(lambda: Device[Device.canonicalize(cap.device)] if cap.device != "unknown" else None,
              default=None, what="device for sync")
  times = []
  for _ in range(max(1, op.reps or 1)):
    _safe(dev.synchronize) if dev is not None else None
    t0 = time.perf_counter()
    try:
      work.build()
    except Exception as e:  # noqa: BLE001
      m.error = f"workload raised on a timed repetition: {type(e).__name__}: {e}"
      return m
    _safe(dev.synchronize) if dev is not None else None
    times.append((time.perf_counter() - t0) * 1e3)
  op.gpu_ms = min(times) if times else None
  op.wall_ms = op.gpu_ms
  op.bytes_min, op.flops_min, op.bytes_mat = work.bytes, work.flops, work.materialised_bytes
  op.arithmetic_intensity = (op.flops_min / op.bytes_min) if (op.flops_min and op.bytes_min) else None
  t_s = op.gpu_ms / 1e3 if op.gpu_ms else None
  if t_s:
    if op.bytes_min:
      op.achieved_gbs = op.bytes_min / t_s / 1e9
    if op.flops_min:
      op.achieved_tflops = op.flops_min / t_s / 1e12
  op.roofline = cap.roofline.position(op.achieved_tflops * 1e12 if op.achieved_tflops else None,
                                      op.achieved_gbs * 1e9 if op.achieved_gbs else None)
  if cap.level == 0:
    op.note += (" level 0: no CUDA roofline and no counters, so the rate below has nothing to be "
                "compared against")
  op.note += " " + (work.note or "")
  if op.bytes_min and op.bytes_mat:
    m.warnings.append(f"caller declares {op.bytes_mat/op.bytes_min:.4g}x more bytes in the "
                      "materialised intermediate than the minimum. bytes_est is unavailable in "
                      "this mode (no per-kernel records), so the ratio cannot be formed here.")
  return m


def _plan(work: Work):
  Tensor, *_ = _tg()
  t = work.build()
  if t is None:
    raise ValueError("work.build() returned None")
  if hasattr(t, "uop"):
    return t.schedule_linear()
  return t


def _compile_only(work: Work):
  from tinygrad.engine.realize import compile_linear, link_linear  # noqa: PLC0415
  return link_linear(compile_linear(_plan(work)))


def _time_call(fn, *a, **kw) -> float | None:
  t0 = time.perf_counter()
  fn(*a, **kw)
  return (time.perf_counter() - t0) * 1e3


def _wall_ms(call: Any, pm_exec, ExecContext, dev: Any) -> float | None:
  """Host wall time around one launch, with a sync on both sides. An independent clock.

  The device comes from the caller. Resolving it here is not possible: the helper that finds
  it needs the linear, and on an asynchronous backend a missing sync would measure the
  submission rather than the kernel, which is precisely the error this cross-check exists
  to catch.
  """
  _safe(dev.synchronize) if dev is not None else None
  t0 = time.perf_counter()
  _safe(pm_exec.rewrite, call, ExecContext(wait=True, update_stats=False), what="cross-check launch")
  _safe(dev.synchronize) if dev is not None else None
  return (time.perf_counter() - t0) * 1e3


def _single_et(ets: Any) -> float | None:
  """One device time out of whatever the executor returned, or None."""
  vals = [e for e in (ets or []) if e is not None]
  if len(vals) == 1:
    return float(vals[0])
  if len(vals) > 1:
    return float(sum(vals))
  return None


def _nbytes(b: Any) -> int:
  """Buffer.nbytes, which is a property in some tinygrad versions and a method in others."""
  v = getattr(b, "nbytes", 0)
  return int(v() if callable(v) else v)


def _record_from_call(i: int, call: Any, Ops, estimate_uop, resolve_params, ansistrip, prod) -> KernelRecord:
  """Static accounting for one call: name, buffers, and first-principles ops/bytes.

  `in_bytes`/`out_bytes` come from the real Buffer objects, so they are ground truth about
  what was allocated, not a model. `bytes_est` is tinygrad's own distinct-bytes estimate
  (a buffer re-read many times is counted once, capped at its size), which is the number
  that answers "did this kernel really touch that much".
  """
  ast = call.src[0] if getattr(call, "src", None) else None
  opname = getattr(ast, "op", None)
  kr = KernelRecord(index=i, kernel=f"<{opname}>", kind="other")
  if opname is Ops.PROGRAM:
    kr.kind, kr.kernel = "program", ansistrip(ast.arg.name)
  elif opname is Ops.COPY:
    kr.kind, kr.kernel = "copy", "COPY"
  try:
    bufs = resolve_params(call, ())
    sizes = [(getattr(b, "device", "?") or "?", _nbytes(b)) for b in bufs]
    kr.buffers = sizes
    kr.device = str(sizes[0][0]) if sizes else ""
    kr.in_bytes = sum(s for _, s in sizes[:-1]) or None
    kr.out_bytes = sizes[-1][1] if sizes else None
  except Exception as e:  # noqa: BLE001
    kr.note = f"buffer accounting failed: {type(e).__name__}: {e}"
  try:
    est = estimate_uop(call)
    kr.flops_est = int(est.ops) if est.ops else 0
    kr.bytes_est = int(est.mem) if est.mem else 0
    kr.lds_est = int(est.lds) if est.lds else 0
  except Exception as e:  # noqa: BLE001
    kr.note = (kr.note + "; " if kr.note else "") + f"estimate failed: {type(e).__name__}: {e}"
  return kr


def _finish_record(r: KernelRecord, cap: Capability, op: OpRecord) -> None:
  """Derived per-kernel rates. None in, None out -- never a fabricated rate."""
  t_s = (r.ms / 1e3) if r.ms else None
  if t_s:
    if r.bytes_est:
      r.achieved_gbs = r.bytes_est / t_s / 1e9
    if r.flops_est:
      r.achieved_tflops = r.flops_est / t_s / 1e12
      r.arithmetic_intensity = r.flops_est / r.bytes_est if r.bytes_est else None
  r.roofline = cap.roofline.position(
    r.achieved_tflops * 1e12 if r.achieved_tflops else None,
    r.achieved_gbs * 1e9 if r.achieved_gbs else None)


def _aggregate_ms(linear: Any, pm_exec, ExecContext, Device) -> float | None:
  """One synchronisation around the whole run. The number to quote.

  Copies are included here and are invisible per-kernel (the runtime does not event-time
  them), which is the honest reason the aggregate can exceed the sum of kernel times.
  """
  dev = _safe(_first_device, linear, Device, default=None)
  ctx = ExecContext(wait=False, update_stats=False)
  _safe(dev.synchronize) if dev else None
  t0 = time.perf_counter()
  for c in getattr(linear, "src", ()):
    _safe(pm_exec.rewrite, c, ctx, what="launch")
  _safe(dev.synchronize) if dev else None
  return (time.perf_counter() - t0) * 1e3


def _first_device(linear: Any, Device):
  from tinygrad.uop.ops import Ops  # noqa: PLC0415
  for c in getattr(linear, "src", ()):
    ast = c.src[0] if getattr(c, "src", None) else None
    if ast is not None and ast.op in (Ops.PROGRAM, Ops.COPY):
      d = c.src[1].device if len(c.src) > 1 else None
      if d:
        name = d[0] if isinstance(d, tuple) else d
        return Device[str(name)]
  return None

# --------------------------------------------------------------------------------------
# 9. Levels 1 and 2: running under Nsight Compute
# --------------------------------------------------------------------------------------

# `ncu` is asked for a bounded number of launches, because replay -- not metric
# availability -- is the binding constraint. Each metric is a replay pass, so a full set on
# 12,090 kernels is not a slow run, it is an unfinished one.
# --csv already suppresses ncu's human-readable summary, so no --print-summary is needed.
# The launch bound is the important one: each metric is a replay pass over the kernel, so
# cost scales with (metrics x launches) and an unbounded run on a 12,090-kernel graph is an
# unfinished run rather than a slow one.
LEVEL2_OPTS = ["--csv", "--target-processes", "all"]
LEVEL1_OPTS = ["--csv", "--target-processes", "all", "--set", "full"]


@dataclass
class NCUResult:
  ran: bool = False
  level: int | None = None
  returncode: int | None = None
  rows: list[dict[str, str]] = field(default_factory=list)     # raw csv rows
  per_kernel: list[dict[str, Any]] = field(default_factory=list)  # name, occurrence, metrics
  requested: list[str] = field(default_factory=list)
  returned: list[str] = field(default_factory=list)
  missing: list[str] = field(default_factory=list)
  reason: str = ""
  classification: str = "NOT_RUN"
  seconds: float | None = None
  stderr_head: str = ""


def ncu_run(argv: list[str], *, level: int = 2, launch_count: int = 64, launch_skip: int = 0,
            metrics: Sequence[str] | None = None, timeout: int = 1800,
            ncu_path: str | None = None) -> NCUResult:
  """Run `argv` under ncu and parse the CSV into per-kernel metric dicts.

  `argv` is the *whole* command, so the target is whatever the study wants to profile
  (a probe script, a driver, a notebook cell dumped to a file). Level 2 collects a targeted
  raw-counter list; level 1 uses `--set full`. Nothing here is required for a valid
  measurement: if it fails, `describe_capability` already said so and the level-3 result
  stands on its own.
  """
  res = NCUResult(level=level)
  exe = ncu_path or find_ncu()
  if not exe:
    res.classification, res.reason = "NOT_FOUND", "no ncu binary"
    return res
  wanted = list(metrics) if metrics is not None else (
    list(LEVEL2_METRICS) if level == 2 else [])
  opts = list(LEVEL2_OPTS if level == 2 else LEVEL1_OPTS)
  if level == 2:
    opts = ["--metrics", ",".join(wanted)] + opts
  cmd = [exe, *opts, "--launch-count", str(launch_count), "--launch-skip", str(launch_skip), "--", *argv]
  res.requested = wanted
  t0 = time.perf_counter()
  try:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    res.seconds = time.perf_counter() - t0
    res.returncode = p.returncode
    res.stderr_head = _tail(p.stderr or "", 5)
    res.rows = _parse_ncu_csv(p.stdout or "")
    if p.returncode != 0:
      res.classification, res.reason = classify_ncu_failure(p.stderr or "", p.returncode)
      return res
    res.ran, res.classification, res.reason = True, "OK", f"collected {len(res.per_kernel)} kernels"
  except subprocess.TimeoutExpired:
    res.seconds = time.perf_counter() - t0
    res.classification, res.reason = classify_ncu_failure("", -1, timed_out=True)
    return res
  except Exception as e:  # noqa: BLE001
    res.classification, res.reason = "BROKEN", f"{type(e).__name__}: {e}"
    return res
  res.per_kernel = _pivot_ncu(res.rows)
  got = {m["metric"] for k in res.per_kernel for m in k["metrics"]}
  res.returned = [m for m in wanted if m in got]
  res.missing = [m for m in wanted if m not in got]
  return res


def _parse_ncu_csv(text: str) -> list[dict[str, str]]:
  """Parse `ncu --csv` output. Tolerant: header detection, ragged rows, blank noise."""
  import csv  # noqa: PLC0415
  rows: list[dict[str, str]] = []
  rd = csv.DictReader([l for l in text.splitlines() if l.strip() and not l.startswith("==")])
  for r in rd:
    if not r or r.get("Metric Name") is None:
      continue
    rows.append({(k or "").strip(): (v or "").strip() for k, v in r.items()})
  return rows


def _pivot_ncu(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
  """Long CSV (one row per metric) -> one dict per kernel, in launch order.

  ncu writes rows grouped by launch, with a `Metric Name` / `Metric Unit` / `Metric Value`
  triple. The kernel name is the only thing tying rows to launches, so kernels are keyed by
  (name, occurrence index) and the order is the order ncu reported, which is launch order.
  """
  out: list[dict[str, Any]] = []
  seen: dict[str, int] = {}
  for r in rows:
    name = r.get("Kernel Name") or r.get("Kernel Name (demangled)") or "?"
    key = (r.get("ID") or "", name)
    if not out or out[-1]["_key"] != key:
      k = seen.get(name, 0)
      seen[name] = k + 1
      out.append({"_key": key, "name": _short_kernel(name), "occurrence": k,
                  "id": r.get("ID"), "metrics": {}})
    mn, mv, mu = r.get("Metric Name"), r.get("Metric Value"), r.get("Metric Unit")
    if mn:
      out[-1]["metrics"][mn] = {"value": _ncu_number(mv), "unit": mu, "raw": mv}
  return out


def _short_kernel(name: str) -> str:
  """Strip the mangling ncu adds so names join against tinygrad ProgramInfo names."""
  n = re.sub(r"<.*?>", "", name)
  n = re.sub(r"\(.*?\)", "", n)
  return n.strip().split()[0] if n.strip() else name


def _ncu_number(v: str | None) -> float | None:
  if v is None:
    return None
  s = str(v).strip().replace(",", "")
  if s in ("", "n/a", "N/A", "-", "null"):
    return None
  m = re.match(r"^([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", s)
  return float(m.group(1)) if m else None


def attach_counters(m: Measurement, res: NCUResult) -> int:
  """Join ncu counters onto level-3 kernel records. Returns the number joined.

  Joins on (kernel name, occurrence). A join miss is reported, never silently dropped: a
  counter attached to the wrong kernel is worse than no counter.
  """
  if not res.per_kernel:
    return 0
  want: dict[str, list[KernelRecord]] = {}
  for r in m.kernels:
    want.setdefault(r.kernel, []).append(r)
  joined, misses = 0, []
  for k in res.per_kernel:
    cands = want.get(k["name"])
    if not cands or k["occurrence"] >= len(cands):
      misses.append(f"{k['name']}#{k['occurrence']}")
      continue
    rec = cands[k["occurrence"]]
    for mn, mv in k["metrics"].items():
      rec.counters[mn] = mv["value"]
    joined += 1
  if misses:
    m.warnings.append(f"ncu reported {len(misses)} launches with no matching level-3 record "
                      f"(first: {misses[:3]}); they are not attached")
  for r in m.kernels:
    if r.counters:
      r.counters_missing = [mn for mn in LEVEL2_METRICS if r.counters.get(mn) is None]
  if m.op.level < 2 and res.ran:
    m.op.level, m.capability.counters, m.capability.counters_kind = 2, True, "targeted"
  return joined


# --------------------------------------------------------------------------------------
# 10. Calibration: the device's own achievable peak, as a check on the derived peak
# --------------------------------------------------------------------------------------

def calibrate(cap: Capability | None = None, iters: int = 5, nbytes: int | None = None) -> dict[str, Any]:
  """Measure achievable bandwidth and fp32 throughput on this device.

  This exists because a roofline computed from a spec is a ceiling, and a ceiling is not
  evidence. The measured numbers are what the hardware actually delivered through
  tinygrad, so they bound every "percent of peak" in the study from the other side. On a
  CPU backend this measures host memory, which is why level 0 is the honest report there:
  the numbers are real and the interpretation does not transfer.
  """
  cap = cap or describe_capability(probe=False)
  # A bandwidth measurement from a buffer that fits in cache is a cache bandwidth, not a DRAM
  # bandwidth, and the difference is 10-100x. So the default buffer is 64 MiB, or 4x L2 if L2 is
  # bigger, and a cache-resident buffer is called out rather than reported as DRAM.
  l2 = cap.facts.l2_bytes
  if nbytes is None:
    nbytes = max(1 << 26, 4 * l2) if l2 else 1 << 26
  out: dict[str, Any] = {"device": cap.device, "level": cap.level, "bytes_s": None, "flops_s": None,
                         "note": "", "nbytes": nbytes, "l2_bytes": l2, "iters": iters,
                         "warnings": []}
  try:
    import numpy as np                                            # noqa: PLC0415
    from tinygrad import Tensor, Device                            # noqa: PLC0415
    dev = Device[cap.device] if cap.device != "unknown" else None
    if dev is None:
      out["note"] = "no device to calibrate on"
      return out
    # Bandwidth via an elementwise stream, not via assign/copy. Two reasons: `assign` is a
    # no-op on an already-realised buffer on some backends (it measured 16 TB/s here, which
    # is not a bandwidth any machine has), and a host-to-device copy measures PCIe, not the
    # memory system. A unary op over one big buffer reads n and writes n, so the traffic is
    # known exactly and the kernel is a real device kernel.
    nel = nbytes // 4
    src = Tensor(np.ones(nel, dtype=np.float32))
    t = _safe(lambda: _bench(lambda: (src * 1.0000001).realize(), iters, dev), default=None,
              what="bandwidth calibration")
    if t:
      out["bytes_s"] = 2 * _nbytes(src) / (t / 1e3)
      out["bandwidth_method"] = ("elementwise stream over one buffer, read n + write n, "
                                 "best of %d, one sync per iteration" % iters)
      if l2 and nbytes <= l2:
        out["warnings"].append(f"the {nbytes/2**20:.0f} MiB buffer fits inside the "
                               f"{l2/2**20:.0f} MiB L2, so this is a CACHE bandwidth and not a "
                               "DRAM bandwidth. Pass a larger nbytes to measure the roof.")
    else:
      out["warnings"].append("bandwidth calibration failed; see the module warnings for the reason")
    # fp32 throughput: a square matmul, 2*n^3 flops
    n = 1024
    a = Tensor(np.random.rand(n, n).astype(np.float32))
    b = Tensor(np.random.rand(n, n).astype(np.float32))
    t2 = _safe(lambda: _bench(lambda: (a @ b).realize(), iters, dev), default=None,
               what="flops calibration")
    if t2:
      out["flops_s"] = 2 * n ** 3 / (t2 / 1e3)
    out["note"] = ("measured through tinygrad, so it is the achievable rate for this stack on this "
                   "device, not the vendor peak")
    if out["bytes_s"] and out["flops_s"]:
      out["measured_ridge"] = out["flops_s"] / out["bytes_s"]
    return out
  except Exception as e:  # noqa: BLE001
    out["note"] = f"calibration failed: {type(e).__name__}: {e}"
    return out


def _bench(build, iters: int, dev) -> float | None:
  """Best-of-N device time in ms for a build-then-realize callable."""
  for _ in range(2):
    build()
  best = None
  for _ in range(iters):
    t0 = time.perf_counter()
    build()
    _safe(dev.synchronize)
    dt = (time.perf_counter() - t0) * 1e3
    best = dt if best is None else min(best, dt)
  return best


# --------------------------------------------------------------------------------------
# 11. Verdicts: what this measurement says about each hypothesis
# --------------------------------------------------------------------------------------

def verdict(m: Measurement | None, hyp_id: str) -> str:
  """One line per hypothesis, and the honest 'cannot tell' when that is the answer."""
  h = HYPOTHESES[hyp_id]
  if m is None:
    return f"{hyp_id}: no measurement"
  cap, op = m.capability, m.op
  if m.error:
    return f"{hyp_id}: untestable ({m.error})"
  if not cap._can(h):
    return (f"{hyp_id}: UNTESTABLE at level {cap.level}. Needs level {h.min_level}"
            + ("" if h.has_static else " and has no static probe, so nothing can decide it here"))
  rows = [r for r in m.kernels if r.kind == "program"]
  ctr = next((rr.counters for rr in rows if rr.counters), None)   # first kernel with counters
  if h.id == "M1_COALESCE":
    s = [r for r in rows if r.static and r.static.available]
    if not s:
      return (f"{hyp_id}: no PTX collected for this measurement, so no access pattern to "
              f"characterise at level {cap.level}")
    v = [r.static.vectorised_load_frac for r in s if r.static.vectorised_load_frac is not None]
    if not v:
      return (f"{hyp_id}: the generated code issues no global memory loads at all in these "
              "kernels, so there is no access pattern to characterise. That is itself a result: "
              "the data was constant-folded or already in registers")
    frac = sum(v) / len(v)
    return (f"{hyp_id}: {frac*100:.1f}% of static global loads are vectorised, widest "
            f"{max(r.static.widest_ld_bytes or 0 for r in s)} B/thread"
            + (f"; sectors/request={_c(ctr, 'l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio')}"
               f", load efficiency={_c(ctr, 'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.avg.pct_of_peak_sustained_active')}"
               if ctr else "; no counters (level 3)"))
  if h.id == "M2_MMA":
    s = [r for r in rows if r.static and r.static.available]
    if not s:
      return (f"{hyp_id}: no PTX collected for this measurement"
              + (" (the workload realizes internally, so there are no per-kernel records)"
                 if not rows else "")
              + f" -- level {cap.level} cannot decide it from execution time either")
    mma = sum((r.static.mma_sync or 0) + (r.static.wmma or 0) + (r.static.wgmma or 0) for r in s)
    ff = sum(r.static.ffma or 0 for r in s if r.static) if s else 0
    arch = cap.static_arch or cap.facts.arch
    arch_note = ARCH_NOTES.get(arch or "", "")
    tail = f" [{arch_note}]" if arch_note else ""
    if mma == 0 and fp32_mma_possible(arch) is False:
      tail += ("; STRUCTURAL: this arch has no TF32, so there is no fp32 MMA for the codegen to "
               "emit. Zero MMA here says nothing about tinygrad. Re-run with float16 inputs, or on "
               "sm_80+, to ask the question the hypothesis is actually asking.")
    if ctr:
      tail += (f"; tensor inst={_c(ctr, 'sm__inst_executed_pipe_tensor.sum')}"
               f", tensor pipe={_c(ctr, 'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active')}"
               f", fma pipe={_c(ctr, 'sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active')}")
    return (f"{hyp_id}: {mma} MMA instructions vs {ff} FFMA instructions in the generated PTX "
            f"(which pipe, not how much work){tail}")
  if h.id == "M3_OCCUPANCY":
    have = [r for r in rows if r.static and r.static.available]
    if not have and not ctr:
      return (f"{hyp_id}: no PTX and no counters for this measurement, so occupancy is not "
              f"determined at level {cap.level}")
    loc = sum(r.static.local_ld or 0 for r in have) + sum(r.static.local_st or 0 for r in have)
    c = None
    for rr in rows:
      if rr.counters.get("launch__registers_per_thread") is not None:
        c = rr.counters
        break
    if c:
      return (f"{hyp_id}: regs/thread={c.get('launch__registers_per_thread')}, achieved occupancy="
              f"{c.get('sm__warps_active.avg.pct_of_peak_sustained_active')}, local ld/st sectors="
              f"{c.get('l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum')}/"
              f"{c.get('l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum')}")
    return (f"{hyp_id}: {loc if loc else 'no'} local ld/st instructions in the PTX; "
            "registers/thread and achieved occupancy need level 2")
  if h.id == "M4_REUSE":
    c = ctr
    if c:
      return (f"{hyp_id}: L2 hit rate={_c(c, 'lts__t_sector_hit_rate.pct')}, L1 hit rate="
              f"{_c(c, 'l1tex__t_sector_hit_rate.pct')}, DRAM read/write="
              f"{_c(c, 'dram__bytes_read.sum')}/{_c(c, 'dram__bytes_write.sum')}")
    return (f"{hyp_id}: level 3 cannot see cache behaviour. bytes_est/bytes_min="
            f"{_r(op.bytes_ratio)} is a bound on traffic, not a measurement of reuse"
            + (f" (materialised/measured = {_r(op.mat_ratio)})" if op.mat_ratio else ""))
  if h.id == "M5_STALL":
    c = ctr
    if c:
      stalls = {k.split("stalled_")[1].split("_per")[0]: v for k, v in c.items()
                if "stalled_" in k and isinstance(v, float)}
      top = sorted(stalls.items(), key=lambda kv: -kv[1])[:3]
      return (f"{hyp_id}: issue active={_c(c, 'smsp__issue_active.avg.pct_of_peak_sustained_active')}, "
              f"dominant stalls " + ", ".join(f"{k}={v:.3g}" for k, v in top))
    return f"{hyp_id}: UNTESTABLE without counters -- stall reasons exist only in hardware. Not inferred from time."
  if h.id == "M6_MATERIALISE":
    if op.bytes_min is None:
      return (f"{hyp_id}: the caller declared no minimum byte count, so there is no denominator "
              "to compare the measured traffic against")
    if op.bytes_est is None:
      return (f"{hyp_id}: the caller declared {_b(op.bytes_min)} as the minimum and "
              f"{_b(op.bytes_mat)} as the materialised size, but this measurement mode produces "
              "no per-kernel traffic estimate, so the ratio cannot be formed. Use a Work whose "
              "build() returns an unrealized Tensor, not build_and_realize")
    msg = (f"{hyp_id}: measured {_b(op.bytes_est)} vs minimum {_b(op.bytes_min)} "
           f"= {_r(op.bytes_ratio)}")
    if op.mat_ratio:
      msg += f"; vs materialised {_b(op.bytes_mat)} = {_r(op.mat_ratio)}"
    return msg + ". Level 3 shows requested traffic; only dram__bytes proves it reached memory."
  if h.id == "M7_PRECISION":
    s = [r for r in rows if r.static and r.static.available]
    if not s:
      return f"{hyp_id}: no PTX collected"
    tf = sum(r.static.mma_tf32 or 0 for r in s)
    f16 = sum(r.static.mma_f16 or 0 for r in s)
    out = f"{hyp_id}: {tf} tf32 MMA, {f16} f16 MMA in the generated PTX for {cap.static_arch}"
    if tf == 0 and fp32_mma_possible(cap.static_arch) is False:
      out += ("; structural: this arch has no TF32, so a silent fp32->TF32 lowering is not "
              "available to be missed here. The hypothesis needs sm_80+ to be askable.")
    return out
  if h.id == "M8_SMEM":
    s = [r for r in rows if r.static and r.static.available]
    if not s:
      return f"{hyp_id}: no PTX collected"
    lds = sum(r.static.ld_shared or 0 for r in s)
    sts = sum(r.static.st_shared or 0 for r in s)
    ca = sum(r.static.cp_async or 0 for r in s)
    return (f"{hyp_id}: {lds} ld.shared / {sts} st.shared / {ca} cp.async; "
            + ("no shared-memory staging at all" if not (lds or sts or ca) else "shared memory is used"))
  return f"{hyp_id}: no verdict implemented"


def _c(c: dict[str, Any], k: str) -> Any:
  v = c.get(k)
  return "n/a" if v is None else v


def _r(x: float | None) -> str:
  return "n/a" if x is None else f"{x:.4g}x"

# --------------------------------------------------------------------------------------
# 12. Reporting
# --------------------------------------------------------------------------------------

_BAR = "-" * 100


def report(m: Measurement, *, kernel_rows: int = 20, verdicts: bool = True) -> str:
  """Human-readable report. Every absent measurement prints as n/a with a reason."""
  o: list[str] = []
  cap, op = m.capability, m.op
  o.append(f"=== {op.name} ===")
  o.append(f"level {op.level}: {cap.level_name}")
  o.append(f"device {cap.device}  arch {cap.facts.arch or 'unknown'}"
           + (f"  {cap.facts.name}" if cap.facts.name else ""))
  if m.error:
    o.append(f"ERROR: {m.error}")
    o.append(_BAR)
    return "\n".join(o)

  o.append(f"timing source   : {cap.timing_source}")
  o.append(f"roofline        : {_si(op_ := cap.roofline.flops_s, 'FLOP/s')} fp32  "
           f"{_si(cap.roofline.bytes_s, 'B/s')} DRAM  ridge {_si(cap.roofline.ridge_point, ' FLOP/B')}")
  for n in cap.roofline.notes:
    o.append(f"                 note: {n}")
  o.append(f"kernels         : {op.n_kernels if op.n_kernels is not None else 'n/a (unknown)'} program"
           + (f" (+{op.n_truncated} not individually timed)" if op.n_truncated else "")
           + (f"; individually timed {op.n_measured}" if op.n_measured else ""))
  o.append("")
  o.append(f"{'aggregate (quote this)':<34}{'value':>18}   {'note'}")
  o.append(_BAR)
  o.append(f"{'  device time (gpu_ms)':<34}{_fms(op.gpu_ms):>18}   one sync around the run")
  o.append(f"{'  sum of per-kernel times':<34}{_fms(op.gpu_ms_per_kernel_sum):>18}   "
           f"UPPER BOUND: each timed launch syncs (inflation grows with kernel count)")
  o.append(f"{'  ...of which the wall check says':<34}{op.per_kernel_trust[:18]:>18}   "
           f"sum(et)/wall = {op.per_kernel_ratio if op.per_kernel_ratio is None else f'{op.per_kernel_ratio:.4g}'}"
           + (f", phase wall {_fms(op.per_kernel_phase_wall_ms)}" if op.per_kernel_phase_wall_ms else ""))
  o.append(f"{'  plan steady-state (ms)':<34}{_fms(op.plan_ms):>18}   Python + scheduler, "
           "kernels already compiled")
  o.append(f"{'  plan cold (ms)':<34}{_fms(op.plan_ms_cold):>18}   first call: includes compiling "
           "the kernels")
  o.append(f"{'  wall (ms)':<34}{_fms(op.wall_ms):>18}")
  o.append("")
  o.append(f"{'first-principles accounting':<34}{'value':>18}   {'note'}")
  o.append(_BAR)
  o.append(f"{'  min bytes (caller)':<34}{_b(op.bytes_min):>18}   hand-computed, the roofline denominator")
  o.append(f"{'  bytes moved (est)':<34}{_b(op.bytes_est):>18}   distinct bytes actually touched")
  if op.bytes_mat:
    o.append(f"{'  materialised bytes':<34}{_b(op.bytes_mat):>18}   caller's intermediate size")
  o.append(f"{'  bytes ratio':<34}{_r(op.bytes_ratio):>18}   moved / minimum")
  if op.copy_bytes:
    o.append(f"{'  copy bytes (separate)':<34}{_b(op.copy_bytes):>18}   "
             "H2D/D2H transfers; excluded from the kernel totals above")
  if op.mat_ratio:
    o.append(f"{'  materialised ratio':<34}{_r(op.mat_ratio):>18}   moved / materialised")
  o.append(f"{'  min flops (caller)':<34}{_si(op.flops_min, ' FLOP'):>18}")
  o.append(f"{'  flops (est)':<34}{_si(op.flops_est, ' FLOP'):>18}")
  o.append(f"{'  flops ratio':<34}{_r(op.flops_ratio):>18}")
  o.append("")
  o.append(f"{'achieved':<34}{'value':>18}   {'note'}")
  o.append(_BAR)
  o.append(f"{'  GB/s (min bytes / time)':<34}{_si(op.achieved_gbs):>18}")
  o.append(f"{'  TFLOP/s (min flops / time)':<34}{_si(op.achieved_tflops):>18}")
  o.append(f"{'  arithmetic intensity':<34}{_si(op.arithmetic_intensity, ' FLOP/B'):>18}")
  rl = op.roofline
  o.append(f"{'  roofline util':<34}{_pct(rl.get('util')):>18}   bound: {rl.get('bound') or 'n/a'}")
  o.append(f"{'  {0}'.format('  why'):<34}{'':>18}   {rl.get('why') or ''}")
  if cap.level == 0:
    o.append("  *** level 0: no CUDA roofline and no counters. The times below are whatever this "
             "backend's timing is -- host time on CPU, device time on another GPU backend -- and "
             "none of them is a CUDA measurement. ***")

  if m.kernels:
    o.append("")
    o.append(f"per-kernel records (showing {min(kernel_rows, len(m.kernels))} of {len(m.kernels)}; "
             f"full set via Measurement.kernels)")
    o.append(f"{'idx':>4} {'ms':>9} {'GB/s':>9} {'TFLOP/s':>9} {'AI':>8} {'bytes':>11} {'flops':>11} "
             f"{'occ%':>6}  kernel")
    o.append(_BAR)
    for r in m.kernels[:kernel_rows]:
      occ = r.roofline.get("util")
      bad = r.timing_untrusted
      o.append(f"{r.index:>4} {('UNTRUSTED' if bad else _fms(r.ms)):>9} "
               f"{('n/a' if bad else _si(r.achieved_gbs)):>9} "
               f"{('n/a' if bad else _si(r.achieved_tflops)):>9} "
               f"{('n/a' if bad else _si(r.arithmetic_intensity)):>8} {_b(r.bytes_est):>11} "
               f"{_si(r.flops_est):>11} "
               f"{'n/a' if bad or occ is None else _pct(occ):>6}  {r.kernel[:40]}"
               + (f"  [{r.kind}]" if r.kind != "program" else ""))
      if r.note:
        o.append(f"     note: {r.note}")
    if any(r.timing_untrusted for r in m.kernels):
      o.append("     UNTRUSTED = the backend's per-launch time failed the wall-clock cross-check "
               "above; the raw value is in the record and in the JSON, and it is not a duration.")
    if op.n_truncated:
      o.append(f"     ... {op.n_truncated} more calls not individually timed "
               f"(max_kernels reached; raise it or trust the aggregate)")

  if cap.static and any(r.static and r.static.available for r in m.kernels):
    o.append("")
    o.append("static PTX inspection (level 3, no profiler, works with no GPU)")
    o.append(_BAR)
    seen = set()
    for r in m.kernels:
      if not r.static or r.kernel in seen:
        continue
      seen.add(r.kernel)
      o.append(f"  {r.kernel[:40]}: {r.static.summary()}")
      if r.static.note:
        o.append(f"     note: {r.static.note}")

  if verdicts:
    o.append("")
    o.append("hypothesis verdicts")
    o.append(_BAR)
    for hid in HYPOTHESES:
      o.append("  " + verdict(m, hid))
    yes, no = cap.decidable()
    o.append("")
    o.append(f"decidable at level {cap.level}: {', '.join(h.id for h in yes) or 'none'}")
    o.append(f"NOT decidable at level {cap.level}: {', '.join(h.id for h in no) or 'none'}")

  if m.warnings:
    o.append("")
    o.append("warnings (these are results, not noise)")
    o.append(_BAR)
    for w in m.warnings:
      o.append(f"  - {w}")
  if op.note:
    o.append(f"note: {op.note}")
  return "\n".join(o)


def _fms(x: float | None) -> str:
  return "n/a" if x is None else f"{x:.4g} ms"


def counter_table(limit: int | None = None) -> str:
  """The counter set, with the hypothesis each counter serves. Read this before editing."""
  o = [f"{'metric':<70} {'hyp':<16} {'unit':<13} read"]
  o.append(_BAR)
  for c in COUNTERS[:limit] if limit else COUNTERS:
    hy = ",".join(METRIC_HYP.get(c.name, [c.hyp]))
    o.append(f"{c.name:<70} {hy:<16} {(c.unit or ''):<13} {c.read[:40]}")
  o.append("")
  o.append(f"{len(COUNTERS)} counters, {len(HYPOTHESES)} hypotheses. "
           f"level-2 targeted list is {len(LEVEL2_METRICS)} metrics; level 1 uses --set full.")
  return "\n".join(o)


def hypothesis_table() -> str:
  o = []
  for h in HYPOTHESES.values():
    o.append(f"{h.id}  (min level {h.min_level})")
    o.append(f"  claim   : {h.claim}")
    o.append(f"  static  : {h.static_probe}")
    o.append(f"  decides : {h.decides}")
    o.append(f"  cannot  : {h.cannot or '-'}")
    o.append(f"  counters: {len(h.counters)} ncu metrics")
    o.append("")
  return "\n".join(o)


def to_dict(m: Measurement) -> dict[str, Any]:
  """JSON-serialisable dump. None survives as null, which is the point."""
  def rec(x: KernelRecord) -> dict[str, Any]:
    return {"index": x.index, "kernel": x.kernel, "kind": x.kind, "device": x.device, "ms": x.ms,
            "ms_wall": x.ms_wall, "timing_untrusted": x.timing_untrusted,
            "flops_est": x.flops_est, "bytes_est": x.bytes_est, "lds_est": x.lds_est,
            "in_bytes": x.in_bytes, "out_bytes": x.out_bytes, "copy_bytes": None,
            "buffers": x.buffers, "achieved_gbs": x.achieved_gbs,
            "achieved_tflops": x.achieved_tflops, "arithmetic_intensity": x.arithmetic_intensity,
            "roofline": x.roofline, "counters": x.counters, "counters_missing": x.counters_missing,
            "note": x.note, "static": None if x.static is None else x.static.summary()}
  return {"name": m.op.name, "level": m.op.level, "device": m.op.device, "error": m.error,
          "capability": {"level": m.capability.level, "timing_source": m.capability.timing_source,
                         "per_kernel": m.capability.per_kernel, "static": m.capability.static,
                         "counters": m.capability.counters, "counters_kind": m.capability.counters_kind,
                         "ncu": m.capability.ncu.one_line(), "reasons": m.capability.reasons,
                         "arch": m.capability.facts.arch, "device_name": m.capability.facts.name,
                         "attrs": m.capability.facts.attrs,
                         "restrict_profiling_to_admins": m.capability.facts.restrict_profiling_to_admins},
          "roofline": {"flops_s": m.capability.roofline.flops_s, "bytes_s": m.capability.roofline.bytes_s,
                       "ridge_point": m.capability.roofline.ridge_point,
                       "flops_source": m.capability.roofline.flops_source,
                       "bytes_source": m.capability.roofline.bytes_source,
                       "notes": m.capability.roofline.notes},
          "op": {k: getattr(m.op, k) for k in
                 ("name", "level", "device", "gpu_ms", "gpu_ms_per_kernel_sum", "plan_ms", "wall_ms",
                  "bytes_min", "bytes_est", "bytes_mat", "flops_min", "flops_est", "achieved_gbs",
                  "achieved_tflops", "arithmetic_intensity", "n_kernels", "n_measured", "n_truncated",
                  "plan_ms_cold",
                  "reps", "copy_bytes", "per_kernel_phase_wall_ms", "per_kernel_trust",
                  "per_kernel_ratio", "note")},
          "op_derived": {"bytes_ratio": m.op.bytes_ratio, "flops_ratio": m.op.flops_ratio,
                         "mat_ratio": m.op.mat_ratio, "roofline": m.op.roofline},
          "kernels": [rec(k) for k in m.kernels],
          "verdicts": {hid: verdict(m, hid) for hid in HYPOTHESES},
          "warnings": m.warnings}


def dumps(m: Measurement) -> str:
  return json.dumps(to_dict(m), indent=2, default=str)


if __name__ == "__main__":
  # `python bench/prof.py` prints the instrument's own state: what it is, what it can
  # measure here, and which hypotheses that leaves untested. The point of running it on a
  # machine with no GPU is that the answer is level 0, stated plainly.
  cap = describe_capability(probe="--no-ncu-probe" not in sys.argv)
  print(cap.explain())
  print()
  print("roofline derivation:")
  print(f"  flops source: {cap.roofline.flops_source or 'none'}")
  print(f"  bytes source: {cap.roofline.bytes_source or 'none'}")
  print(f"  SM count: {cap.roofline.sm_count}  cores/SM: {cap.roofline.cores_per_sm}  "
        f"clock: {_si(cap.roofline.sm_clock_hz, ' Hz')}  L2: {_b(cap.roofline.l2_bytes)}")
  print(f"  ridge point: {_si(cap.roofline.ridge_point, ' FLOP/B')}")
  print()
  print("hypotheses this instrument can test, and what it cannot:")
  print(hypothesis_table())
  print("counter -> hypothesis map:")
  print(counter_table())
  if cap.ncu.available:
    print()
    print("ncu is reachable: run `ncu_run([...], level=2, launch_count=N)` to attach counters.")
