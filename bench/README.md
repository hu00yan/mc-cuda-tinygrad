# bench/prof.py — the measurement instrument

**What this is.** A uniform way to measure a tinygrad workload and report, in one place,
what was actually obtained: wall time, per-kernel device time, bytes moved, FLOPs
performed, arithmetic intensity, roofline position, and — where the hardware allows it —
the hardware counters that say *why* a number is what it is.

**Why it exists.** This study compares compiler-generated GPU code against hand-written
kernels. A wall-clock number on a fast card says "how much slower". Hardware counters say
"why", and a mechanism transfers to another architecture even when no wall-clock number
exists for it. The study's primary instruments are old free cards (Kaggle T4 = sm_75,
P100 = sm_60) in containers where Nsight Compute will most likely fail: `ncu` needs
`NVreg_RestrictProfilingToAdminUsers=0`, and its serialised replay is 10–100× slower than
the kernel it measures. So the harness is built to degrade, and to **say which level it
degraded to**.

Two rules it never breaks, because breaking either is how a measurement becomes a lie:

1. **An unavailable measurement is `None` with a stated reason. Never `0`.** A count of
   zero is a finding (zero MMA instructions *were* issued); a missing measurement is not.
2. **Nothing device-specific is hardcoded in the measurement path.** Every peak, clock, SM
   count and bus width is read from the device at hand. The one constant is cores-per-SM,
   which is a structural fact of a microarchitecture, and an arch that is not in the table
   yields "unknown", never a guess.

---

## The three levels, and what each one gives you

| level | needs | what you get | what it cannot do |
|---|---|---|---|
| **3** | a device, no privileges, no profiler | per-kernel **device time** from the backend's own event mechanism, × **first-principles accounting** (bytes and FLOPs) → achieved GB/s, achieved TFLOP/s, arithmetic intensity, roofline position, per-kernel table. Plus a **static channel**: the generated PTX rendered for a target arch and inspected, which tests the vectorisation, matrix-path, spill, shared-memory and precision questions **with no GPU and no profiler at all**. | cannot see cache behaviour, occupancy, stall reasons, or DRAM traffic as opposed to requested traffic |
| **2** | `ncu` present *and permitted* | all of level 3, plus a **targeted** raw-counter set (`--metrics`, 28 metrics) on a **bounded** number of launches | no sections, no source-level attribution |
| **1** | same | all of level 2, plus `--set full` | 10–100× the replay cost of level 2; not attempted automatically |
| **0** | nothing | an honest "no device timing here", host timings labelled as host timings, and `n/a` for everything device-derived | everything |

**Level 3 is the floor and the floor is already enough to place a kernel on the roofline.**
That is the design: on hardware where nothing else is permitted, you still get an
achieved bandwidth, an arithmetic intensity, and a position relative to the device's own
ceilings.

### Why the "static channel" is part of level 3

The two most important questions in this study — *was the generated code vectorised* and
*did it reach the tensor pipe* — are questions about **the code**, and the code can be read
without a GPU. `prof.py` renders each kernel to PTX for a chosen arch and counts the
instructions: `ld.global.nc.v4.f32` versus `ld.global.f32` decides vectorisation;
`mma.sync` versus `fma.rn.f32` decides which pipe the work went to; `.local` traffic
suggests spilling; `st.shared`/`ld.shared`/`cp.async` measures shared-memory staging; the
MMA variant in the text (`…f32.tf32.tf32.f32`) decides whether precision was silently
lowered.

This is *static* evidence about *generated code*, not a measurement of execution, and the
report labels it that way. Set `PROF_ARCH=sm_75` to inspect the code for a card you do not
have — the arch is a parameter of the code generator, not a property of the machine doing
the reading. Where a counter and the PTX disagree, the counter is the arbiter and the
disagreement is worth reporting.

---

## How to tell which level you got

`describe_capability()` prints it, and every `Measurement` carries it:

```python
from prof import describe_capability
print(describe_capability().explain())
```

It reports the level, the device, **which timing mechanism was used**, whether per-kernel
timing and the static channel are available, the ncu classification, the roofline with the
provenance of each number, and a reason for every line. Look for:

* `level 0` → nothing device-derived is available here; every such field prints `n/a`.
* `level 3 … timing: cudaEventRecord/cuEventElapsedTime …` (the `CUDA` backend) or
  `… GPU-written timestamps on the submission queue …` (the `NV` backend). Both are
  device-side timing with no profiler. The NV path synchronises after every timed launch,
  which is why per-kernel times are an upper bound there (see below).
* `level 2 … HW counters: True (targeted)` → `ncu` was actually run and reported metrics.
* `ncu unavailable [PERMISSION] … NVreg_RestrictProfilingToAdminUsers=1` → the single most
  common Kaggle outcome, detected up front from `/proc/driver/nvidia/params` before spending
  a minute discovering it the slow way.

Two further guards, because a level claim is not the same as a good number:

* **The per-kernel channel is cross-checked against the host clock.** The phase that
  produces per-kernel times is bracketed by `perf_counter`, and the largest kernel is
  re-run alone and wall-bracketed. If the device-side times do not account for the wall time
  of the phase, the report prints `UNTRUSTED` instead of a number and adds a warning. This
  is not hypothetical: on this repo's CPU backend the per-launch "duration" comes back as
  1.2 µs for a 1.2 ms kernel, and the harness catches it every run.
* **`aggregate` and `per-kernel` are separate numbers and the report says which is which.**
  The aggregate is one synchronisation around the whole run — that is the number to quote.
  The sum of per-kernel times is an upper bound (a sync per launch; the inflation grows with
  kernel count) and is for attribution only.

---

## Running it

```bash
cd /Users/huyan00/mycode/mc-cuda-tinygrad

# what level does this machine reach, and why
.venv/bin/python bench/prof.py                 # also prints the counter->hypothesis table
.venv/bin/python bench/prof.py --no-ncu-probe  # skip the ncu capability probe

# validate the whole instrument on a machine with no GPU (this is what the lane shipped)
PROF_ARCH=sm_75 .venv/bin/python bench/selftest.py
```

Measuring your own workload — three lines:

```python
import sys; sys.path.insert(0, "bench")
import numpy as np
from tinygrad import Tensor
from prof import Work, measure, report

n = 1024
a = Tensor(np.random.rand(n, n).astype(np.float32))
b = Tensor(np.random.rand(n, n).astype(np.float32))

m = measure(Work("matmul 1024 fp32", lambda: a @ b,
                 flops=2 * n**3,            # the work an ideal kernel must do
                 bytes=3 * n * n * 4,       # the traffic an ideal kernel must move
                 materialised_bytes=None),  # what a naive graph builds instead, if different
                reps=3, warmup=1)
print(report(m))
```

`flops` and `bytes` are the **caller's** hand-computed minimum. They are the denominator the
roofline uses, so they are the study's claim, not the instrument's. Leave them `None` and
you get bytes/FLOPs rates but no roofline position, which is a correct and useful result.

`build()` must return an **unrealized** Tensor and is called afresh each time. That matters:
`Tensor.realize()` on an already-realized graph is a silent no-op, so timing a prebuilt
expression measures nothing and reads as ~3 ns.

Probing a workload that realizes itself (which is how most of `probes/` and `kda/` are
written) — use `build_and_realize=True`. You get the aggregate device time and lose the
per-kernel attribution and the plan/kernel split; both losses are stated in the output.

Attaching hardware counters (levels 1–2) on a machine where `ncu` works:

```python
from prof import ncu_run, attach_counters, LEVEL2_METRICS
res = ncu_run([sys.executable, "my_probe.py"], level=2, launch_count=64, launch_skip=0)
print(res.classification, res.reason)     # check this before believing res.rows
joined = attach_counters(m, res)           # joins on (kernel name, occurrence)
print(joined, "launches joined;", len(res.missing), "metrics not reported")
```

`launch_count` is not optional in practice: each metric is a replay pass, so cost scales
with (metrics × launches). On a 12,090-kernel graph an unbounded run is an *unfinished*
run, not a slow one. `attach_counters` reports joins it could not make rather than attaching
a counter to the wrong kernel.

Calibrating what the device actually achieves, as a second bound on every "percent of peak":

```python
from prof import calibrate
print(calibrate())    # achievable GB/s and TFLOP/s through tinygrad on this device
```

It warns when the probe buffer fits in L2, because then the number is cache bandwidth and
not DRAM bandwidth — a 10–100× error that looks like a great result.

---

## The roofline, and where it comes from

`roofline_from_attrs` is a pure function of device attributes, so the arithmetic is testable
on a machine with no GPU, and `selftest.py` tests it against the published bandwidth and
FLOP figures for the three cards this study uses.

* **Memory roof** = `2 × MEMORY_CLOCK_RATE × GLOBAL_MEMORY_BUS_WIDTH / 8`, all three read
  from `cuDeviceGetAttribute`. DDR moves data on both edges of the clock. This reproduces
  the published bandwidth exactly on the T4 (320 GB/s) and the P100 (224 GB/s).
* **Compute roof** = `MULTIPROCESSOR_COUNT × fp32 cores/SM × 2 × CLOCK_RATE`. The SM count
  and clock come from the driver; cores/SM is the single structural constant in the file
  (CUDA C Programming Guide compute-capability table) and an arch not in the table produces
  **no compute roof at all**, which is stated.
* **Ridge point** = compute roof / memory roof, in FLOP per byte. T4: 25.4. P100: 52.1.
  L4: 101. (Those are the derived values for the attributes the driver reports; the printed
  report always shows the inputs alongside them.)
* **Tensor peak is never reported**, because it is not derivable from driver attributes and
  embedding a vendor table to supply it is exactly the hardcoded device number this module
  refuses. The study's matmuls are fp32 anyway, and the matrix-path question is answered by
  counters and by the PTX.

One subtlety worth stating because the naive version is wrong: efficiency is **not**
`min(flops/peak_flops, bytes/peak_bytes)`. The memory limit for a kernel is
`AI × peak_bytes`, so the correct form is `achieved / min(peak_flops, AI × peak_bytes)`, and
`bound` is decided by comparing AI to the ridge point. `Roofline.position()` implements that
and has degenerate cases for it (one roof missing, one rate missing, no data at all). An
efficiency above 100 % is **not** clamped — it means the peak is wrong or the work is
miscounted, and both are findings.

---

## What this instrument cannot do

Stated plainly, because a measurement instrument that only advertises its reach is not
trustworthy about the rest.

* **It cannot produce a hardware number where the hardware is not.** Level 0 on a CPU gives
  host timings and `n/a` for everything device-derived. `selftest.py` on this Mac is a
  validation of the plumbing, the arithmetic, the static channel and the honesty; it
  validates no GPU number and does not try.
* **It cannot test stalls, cache residency, occupancy or spills at level 3 or 0.** Those are
  M4, M3, M5 (below) and they need `ncu`. There is no static substitute for a stall reason:
  it is a dynamic property, and nothing in the source says where time went.
* **It cannot distinguish a kernel that re-reads a hot buffer from one that streams it.**
  Level 3 sees *distinct* bytes per kernel; a hot buffer and a streamed one look identical
  if their distinct-byte sets match. Only `lts__t_sector_hit_rate` separates them.
* **It cannot turn stall *ratios* into time shares.** A stall ratio is stall cycles per
  issued instruction. Two kernels with very different durations can have the same ratio.
  Converting would be an invention, so it is not done.
* **It cannot see DRAM traffic at level 3.** Level 3 counts *requested* traffic read off
  the real buffers. A buffer that stays resident in L2 is counted as moved and never reaches
  DRAM. `dram__bytes_read/write` at level 2 is the only proof it reached memory.
* **It cannot tell a deliberate local-memory access from a spill.** `smsp__inst_executed_op_local_*`
  is zero or non-zero; it does not say why.
* **It cannot say *why* the MMA is missing** — only that it is. The codegen lane's job.
* **It cannot measure across architectures.** One run, one device. "Optimized" is
  architecture-indexed (a kernel tuned on one card can be 100× off on another with no
  correctness signal), so a finding here is a statement about one cell of the grid.
* **It does not autotune, sweep, or search anything.** `reps` repeats the same schedule. If
  a question is about the *search space* rather than the emitted code, the sweep is a
  separate experiment that calls `measure()` once per point.
* **It cannot run `cp.async` questions on sm_60/sm_75.** `cp.async` does not exist before
  sm_80, so its absence on the T4 and the P100 is architecture, not behaviour. The
  instrument says so rather than reporting a zero.
* **It cannot judge whether a result is numerically right.** That is
  `probes/fp32_correctness.py` and `kda/test_tiny.py`, not a counter.

---

## Files

| file | role |
|---|---|
| `prof.py` | the instrument. `describe_capability`, `measure`, `ncu_run`, `attach_counters`, `calibrate`, `report`, `verdict`, `to_dict`, `HYPOTHESES`, `COUNTERS`, `LEVEL2_METRICS` |
| `selftest.py` | validation on a machine with no GPU: capability, the degradation ladder, the roofline arithmetic against synthetic attributes, the `ncu` CSV path against a recorded transcript, the repo's own probes, a static-channel control, calibration, JSON |

---

## The counter set, and the hypothesis each counter is attached to

A counter with no hypothesis attached is a number nobody knows what to do with. So all 71
counters in `COUNTERS` name the claim they are evidence for, and every claim says which
level can decide it and which cannot. Print the live table with:

```bash
.venv/bin/python bench/prof.py | tail -80     # counter -> hypothesis map
.venv/bin/python bench/prof.py                # hypothesis -> claim / static probe / limits
```

The eight mechanism questions, `min level` = the lowest level at which the question is
*decided*:

| id | question | decided at | by what |
|---|---|---|---|
| **M1_COALESCE** | did the generated code move data in wide coalesced transactions, or degrade to scalar 32-bit loads? | **3** (statically) / **2** (with numbers) | `ld.global*.vN` width distribution in PTX; or `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio`, `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.avg.pct_of_peak_sustained_active`, request vs sector counts |
| **M2_MMA** | did the reduction reach the hardware matrix path at all, or did the work land on the FP32 FMA pipe? | **3** (statically) / **2** (with utilisation) | `mma.sync`/`wmma`/`wgmma` vs `fma.rn.f32` in PTX; or `sm__inst_executed_pipe_tensor.sum`, `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active`, `sm__pipe_fma_cycles_active...`, `sm__sass_thread_inst_executed_op_ffma_pred_on.sum` |
| **M3_OCCUPANCY** | did register pressure or spilling collapse occupancy? | **2** | `launch__registers_per_thread`, `launch__occupancy_limit_{registers,shared_mem,warps,blocks}`, `sm__warps_active.avg.pct_of_peak_sustained_active`, `l1tex__t_sectors_pipe_lsu_mem_local_op_{ld,st}.sum`; statically: `.local` ld/st and declarations |
| **M4_REUSE** | was the data reused, or streamed? is the working set L2-resident? | **2** | `lts__t_sector_hit_rate.pct`, `l1tex__t_sector_hit_rate.pct`, `lts__t_sectors*`, `dram__bytes_{read,write}.sum`, `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` |
| **M5_STALL** | where does the time actually go? | **2** (never statically) | `smsp__issue_active.avg.pct_of_peak_sustained_active`, `smsp__warps_eligible.avg.per_cycle_active`, and 20 `smsp__average_warps_issue_stalled_*_per_issue_active.ratio` |
| **M6_MATERIALISE** | was the intermediate really built, or is that a suspicion? | **3** | per-kernel distinct bytes (`bytes_est`, read off the real buffers) against the caller's `bytes_min` and `materialised_bytes`; `dram__bytes_*` at level 2 corroborates that it reached memory |
| **M7_PRECISION** | was precision silently lowered (fp32 → TF32 / fp16)? | **3** | the MMA variant in the PTX text: `…f32.tf32.tf32.f32` vs `…f16.f16.f32` |
| **M8_SMEM** | are inputs staged through shared memory / async copy, as tinygrad's docs concede they are not? | **3** | `st.shared`/`ld.shared`/`cp.async` counts in PTX; `smsp__inst_executed_op_shared_{ld,st}.sum`, `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` at level 2 |

`LEVEL2_METRICS` is the 28-metric targeted list — one cheap pass, every hypothesis that a
raw counter can decide, and nothing that needs a derived section. Level 1 adds `--set full`.

### Two measurement traps this mapping is built around

* **M2 has a confound that is bigger than the effect.** fp32 reaches a tensor pipe only
  where TF32 exists (sm_80+). On the T4 and the P100 a zero-MMA result is *architecture*,
  not behaviour, and `verdict()` says so in those words rather than letting "0 MMA" stand
  as a finding. `fp32_mma_possible(arch)` is the check.
* **The published RC1 signature is a counter, not a source property.** "12.55 % load
  efficiency, 16.4 sectors per request" is an L1 measurement. Level 3 can falsify "the code
  is scalar" from the PTX, which is arguably stronger for that claim, but it **cannot
  reproduce the published number**, and the report does not pretend otherwise.

---

## How this maps onto the study's pre-registration

`HYPOTHESES.md` in this directory is the study's frozen pre-registration and uses `H1..H14`
for whole-operator claims. The ids in `prof.py` are `M1..M8` for the narrower *mechanism*
questions, deliberately: two files in one study using `H1` for different claims is how a
result gets attributed to the wrong hypothesis. Each study hypothesis is decided by a subset
of the mechanism questions:

| study hypothesis | measured by | note for whoever runs it |
|---|---|---|
| **H1** KDA gram gets no tensor-core MMA | **M2_MMA** (level 3 static, or level 2 with `sm__inst_executed_pipe_tensor.sum`) | needs a dtype/arch on which an MMA is *possible* at all, or M2's confound check turns the result into "architectural" |
| **H2** the `[C,C,K]` decay really costs DRAM bytes | **M6_MATERIALISE** at level 3 for the traffic, `dram__bytes_write.sum` at level 2 for the DRAM | M6 is the one hypothesis level 3 decides on its own |
| **H3** kernel count is O(C²) from the unrolled substitution | per-kernel record count from `measure()` (`op.n_kernels`) | a count, not a timing; the harness reports it at every level |
| **H4** / **H13** fp32 KDA is memory bound, not compute bound | `Roofline.position()` on the measured rates, plus **M4_REUSE**'s `gpu__dram_throughput` | the position needs `flops` *and* `bytes` declared on the `Work` |
| **H5** 85 % of KDA wall is the scheduler | `plan_ms` vs `plan_ms_cold` vs `gpu_ms` in the same report | this is why the harness reports cold and steady-state planning separately |
| **H6** / **H12** the camera's per-step kernels and dead unroll work | per-kernel records + **M1_COALESCE** for the access pattern | "dead work" is a coverage question: a kernel whose `bytes_est` is far below its own occupancy cost |
| **H7** the gather plugin fails identically on CUDA | **M1_COALESCE** on both backends, same `Work` | the instrument runs unchanged on any device, which is the point |
| **H8** the gather's source-bound crossover is L2 capacity | **M4_REUSE**: L2 hit rate and DRAM bytes at two source sizes, plus the reported L2 size | needs level 2 for the hit rate; level 3 gives the traffic bound only |
| **H9** / **H10** CUDA divide rounding | none — this is arithmetic, not performance | `probes/fp32_correctness.py`; no counter decides it |
| **H11** no tensor-core path for the dtype on the free cards | `fp32_mma_possible(arch)` + the dtype's MMA entry in the static channel | a source/API fact; the instrument only supplies the arch check and the confirmation that no MMA was emitted |

Anything in `HYPOTHESES.md` §5 ("hypotheses that no available measurement can refute") is
outside this instrument by construction, and the README's "What this instrument cannot do"
section is the matching list of blind spots from the other direction.
