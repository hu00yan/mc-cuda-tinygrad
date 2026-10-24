#!/usr/bin/env python3
"""run_kda.py - a single-file GPU battery for the KDA path, for one Kaggle T4.

WHAT THIS IS
    One file, no arguments required, no notebook edits, no new dependencies.
    Drop it on the box, point it at the code under test, and it writes a
    directory of artefacts plus a one-screen summary. It is designed to be
    pushed once and read once: every stage is time-boxed, every stage fails
    independently, and the summary is written even when a stage dies.

WHY IT EXISTS
    The first real GPU run (`out/kaggle/20260928-014248-gpu/`, 2026-09-28) left
    three things unresolved or wrong. None of them can be fixed by re-reading
    that log, because the log is not a record of one batch.

    1. THE BATCH. The record states that the GPU batch and the payload batch
       were different runs: `logs/divide_precision.log` came from a Tesla T4
       with 15,360 MiB and 47.63 MB of realised traffic, while the payload
       delivered to that kernel was 26 files totalling about 0.5 MB. Every
       figure derived by comparing one of those logs against the other is
       therefore void, including the "1% GPU / 99% Python" split and the
       "3.78x" ratio (see the long note on 3.78x below). Those must be
       REGENERATED, not reinterpreted. Stage `manifest` is built so this class
       of error cannot recur silently.

    2. THE KERNEL COUNT. `kda/perf_kda.py` measured 12,082 launches at the
       bench shape on the T4. A device-free census of the same graph under
       `DEV="CUDA"` also gives 12,082, and finds the count to be
       `~12,042 + 5*NT` at C=64 - a constant of C, not of T. A reader who
       compares those two numbers against a *different* C (the codegen capture
       at `kernels/build_notebooks.py:590` uses C=32) sees two numbers that
       disagree and concludes one is wrong. What is actually true, and what
       this file measures and reports: **12,082 is a statement about C=64 and
       is meaningless without it.** At C=32, NT=8 the same graph schedules
       2,994 kernels; at C=16, 754. The count is a function of C - it is
       `C(C-1)` iterations of the unrolled forward substitution at
       `kda/tiny_kda.py:178-194`, about three kernels each - plus five per
       chunk for the inter-chunk scan. Over an eightfold span of T at fixed C
       it moves 1.3%. So the two numbers in the argument were never in
       conflict; they were two numbers from two different C, quoted without
       their shapes.

    3. THE INSTRUMENT. `DEBUG=2` wraps every launch in
       `cuEventCreate x2` + `cuEventRecord` + `cuEventSynchronize` +
       `cuEventElapsedTime` + `cuEventDestroy x2`
       (`tinygrad/runtime/ops_cuda.py:26-35`), and `engine/realize.py:318`
       forces `wait` on whenever `DEBUG>=2`. So `event1 - event0` contains the
       HOST LAUNCH LATENCY as well as the kernel: the 215.880 ms in
       `logs/perf_kda.log:32` is 12,082 event-bracketed, host-synchronised
       launches, i.e. an UPPER BOUND on GPU work, not a measurement of it. This
       file measures the same call with `DEBUG=0` and a single `synchronize()`
       around a warm `realize()` and reports both, plus the difference, so the
       instrument's own cost is a measured number instead of a hidden one.

THE 3.78x FIGURE, AND WHY IT MUST BE REGENERATED RATHER THAN REINTERPRETED
    A "3.78x" clock/host ratio has been quoted for this path. Its two inputs
    were constants that were never measured on the batch that produced the
    log:

    * **0.109 s of "GPU time"** - a hard-coded literal at `kda/replay_kda.py:97`
      and again at `:101-102`, printed as "measured separately in
      perf_kda.py". It was never read from perf_kda. The T4's own DEBUG=2
      per-kernel sum in that same run was **215,880 us = 215.9 ms**, and that
      number is itself an upper bound (point 3 above), so 0.109 s is wrong for
      the T4 by an unknown factor in an unknown direction.
    * **49.5 s of "COLD"** - a hard-coded literal at `kda/replay_kda.py:105`,
      printed in the same paragraph that the script's own line 9 contradicts:
      `logs/replay_kda.log:316` measured **COLD = 96,837 ms** on the T4. That
      is 1.96x the printed constant.

    So 3.78x was built from a Mac-era GPU constant and a Mac-era cold constant
    and then quoted against a T4 log, from a batch that is itself in question.
    **3.78x is not an input to this file and not a check on it.** Stage
    `ratios` recomputes every ratio the quantities measured here support, from
    one batch, naming the instrument behind each input. It deliberately does
    NOT try to reproduce 3.78x: the formula that produced 3.78x is recorded
    nowhere in the tree, so any attempt to reproduce it would be invention.
    If you want the ratio back, it has to come from a fresh single-batch run.

THE STAGES
    manifest      sha256 of every file the battery can see, of every module it
                  actually imported, of the payload tarball if one is present,
                  and of this file. Re-verified before every stage. One
                  `manifest_id` token so two runs are comparable by eye.
    environment   device, versions, nvidia-smi, the cost of an idle
                  `synchronize()` on this device (the floor that every other
                  sync-based number has to be read against).
    census        device-free launch count, as a function of C and of NT, one
                  shape per child process. No device, no execution, no timing
                  claims.
    launch_count  the same count measured by the DEVICE (tinygrad's own
                  `GlobalCounters.kernel_count` delta) at the shapes the budget
                  allows, so census and device can be compared at one shape.
    instrument    the same warm call timed with `DEBUG=0` + one `synchronize()`
                  and with `DEBUG=2`, in two separate processes, so the
                  per-launch host overhead is separable instead of baked in.
    decomposition graph construction / rewrite passes / lower+compile+issue /
                  device drain / copy-out, each measured separately, with the
                  unattributed residual reported rather than hidden.
    codegen       the emitted kernel sources, captured with `PARALLEL=0`
                  because the default worker pool compiles in subprocesses and
                  an in-process hook sees 1 kernel out of 12,082.
    divide        the float32 `/` probe, reimplemented here so this file stays
                  self-contained, and CROSS-CHECKED bit-for-bit against
              `probes/divide_precision.py` when that file is importable.
    ncu           opportunistic: if `ncu` exists, a bounded sample of per-kernel
                  device durations, which is the only instrument here that
                  excludes launch latency by construction.
    ratios        cold / warm / kernel, all from this batch, each labelled with
                  the instrument that produced it.

STATUS VOCABULARY. Every number this file prints carries one.
    MEASURED        an instrument on this machine, this run, this batch.
    RE-DERIVED      arithmetic on quantities this run measured, shown with the
                    arithmetic.
    MODELLED        a fit to measurements, printed with its residuals. A fit is
                    never printed as a measurement.
    DATASHEET       a published hardware figure. Never used in any derived
                    number here; printed only so a reader does not have to
                    look it up.
    SOURCE          read out of the installed tinygrad, quoted with file:line.
    UNAVAILABLE     not measured here, with the reason. A stage that did not
                    run NEVER reports a number in its place.

WHAT IS NOT VERIFIED WITHOUT A GPU
    This file was written and its logic exercised with no CUDA device present
    (`Device.DEFAULT` on the development host was CPU and METAL, in separate
    runs). On those hosts every field derived from a GPU is reported
    `UNAVAILABLE` with a reason, and NO GPU NUMBER IS INVENTED. Unverified until
    a T4 runs it:
      * every `device=CUDA` value, and the CUDA and CL divide rates;
      * `nvidia-smi` fields, SM count, clocks, persistence mode;
      * the DEBUG=2 host-overhead separation. The wrapper is a CUDA
        implementation detail (`cuEventCreate` / `cuEventSynchronize`);
        METAL and CPU take other branches, so the number this battery measures
        on METAL is NOT the T4's. The `debug2.wrapper_is_device_specific` field
        says so in the results;
      * `ncu`, which is absent on the development host by construction;
      * the H2D copy time. `_copyin` is a no-op observation on METAL (a
        shared-memory allocator writes through a mapping) and CPU, so the H2D
        line reads 0.000 ms on both. It is not a measurement of zero upload, and
        the instrument text in `h2d_host_ms` says so at the point of use.

    What IS verified without a GPU, and is the whole reason the kernel-count
    question is decidable: the device-free census, which needs no `Device` and
    launches no kernel. On the development host, at the bench shape
    (B=2, T=512, H=4, HV=4, K=64, V=64, C=64) under `DEV="CUDA"`, it returned
    **12,082 sink + 139 copy** - the T4's `logs/perf_kda.log:31` to the unit,
    and the copy count `notes/13` §5 states. Over the C=64, NT=1..16 sweep it
    fitted `sink ~= 2.99*C*(C-1) + 5.06*NT - 6.8` with a worst residual of 3.9
    kernels out of ~12,000 (0.041%), against a worst residual of 79.6% for a
    count proportional to T. The same host's DEVICE launch count at the bench
    shape, on its own METAL backend, was 12,229 = 12,090 sink + 139, which is
    `notes/13` §5's METAL figure to the unit and is 8 above the CUDA total of
    12,221 - the `DEVICE_MAX_BUFS` difference of
    `tinygrad/schedule/rangeify.py:354`, reproduced here and, in this file,
    reported as a labelled backend difference rather than a disagreement. At
    C=64/NT=1 the same comparison gave 12,192 on the device against 12,184 from
    the CUDA census, again a difference of 8.

HOW TO RUN IT
    Locally (validates the logic, no GPU required):
        .venv/bin/python remote/run_kda.py --quick --out /tmp/kda-quick
        .venv/bin/python remote/run_kda.py --only census,launch_count
    On the Kaggle box, once the code under test is unpacked somewhere:
        python3 run_kda.py --code-root /kaggle/input/<slug>/probe \
                           --out /kaggle/working/run_kda
    `kaggle_run.sh` does not carry this file today: `kernels/pack.py`'s INCLUDE
    list has no entry for it and `kda/replay_kda.py` prints a hard-coded 0.109
    rather than parsing `results.json`. So the integration step, owned
    elsewhere, is one line in `kernels/pack.py`:
        ("remote/run_kda.py", "probe/remote/run_kda.py"),
    and one notebook cell in `kernels/build_notebooks.py` that runs it. Until
    then this file is self-sufficient: `--code-root` accepts any directory that
    contains `kda/tiny_kda.py`, including the unpacked payload, and it writes
    its own manifest so the version it measured is on the record either way.

GOTCHAS THIS FILE RESPECTS (each one cost a real run somewhere)
    * `Tensor.realize()` on an ALREADY-realized graph is a SILENT NO-OP. Every
      timed unit here is therefore a fresh `kda_chunk(...)` call, which rebuilds
      every UOp from numpy. `kda/tiny_kda.py` takes numpy inputs, so a fresh
      call is a genuinely fresh graph, and the launch count of every timed
      iteration is recorded rather than trusted.
    * A PROBE CAN PERTURB WHAT IT MEASURES. Splitting `realize()` from
      `numpy()` so the synchronize can be placed between them costs exactly one
      extra launch per call at the development shape (777 -> 778, reproducible).
      So this file measures the same call with and without the probe and
      reports the difference as `instrument_perturbation_launches`, and the
      count it compares against the census and the DEBUG=2 parse is the
      UNPROBED one.
    * `DEBUG=2` per-kernel times must be DIVIDED BY THE NUMBER OF REALIZES.
      `kda/perf_kda.py` divides by 2, which is only correct if the cold and warm
      calls issue the same number of launches. They do here (verified: 777 at a
      small shape, cold == warm), but this file does not assume it - the child
      brackets each call with a marker line, the parse segments on those markers,
      divides by the number of warm segments it actually found, and reports that
      number as `divided_by`.
    * A census repeated in the SAME PROCESS is memoized by tinygrad's rewrite
      passes. The COUNT is stable (verified: 12,082 on three repeats in one
      process, including one after a different shape had been run in between)
      but the WALL TIME is not: on the development host the second identical
      census ran 5.4x faster than the first. So every census here is its own
      child process, and no census wall time is ever reported as a scheduler
      cost.
    * `GlobalCounters.kernel_count` counts COPIES as well as kernels and does
      not separate them. Comparing it against a DEBUG=2 parse that filters
      " copy " out, or against a census sink count, requires the copy count from
      the census. This file does that and says where the copy count came from.
    * `kda/perf_kda.py:143` prints the literal string "Metal" on a CUDA run. No
      string in this file names a device except `Device.DEFAULT` itself, and
      every count row carries the `census_dev` that produced it.
    * `kda/replay_kda.py` hard-codes 0.109 s and 49.5 s into a paragraph of
      arithmetic. No constant from any other file is used as an input here. The
      prior figures appear only in `PRIOR`, each with `PRIOR_SOURCE`, and only as
      check targets.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback

import numpy as np  # tinygrad's own dependency; not a new one

# ---------------------------------------------------------------- constants
HERE = os.path.dirname(os.path.abspath(__file__))
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# The shape `kda/perf_kda.py:90` benchmarks, and the shape the T4 measured
# 12,082 launches at (out/kaggle/20260928-014248-gpu/logs/perf_kda.log:6,31).
ANCHOR = dict(B=2, T=512, H=4, HV=4, K=64, V=64, C=64)
# The shape `kernels/build_notebooks.py:590` captured ONE kernel source at:
# C=32, not C=64. Named here because "12,082 launches" without a C is exactly
# the ambiguity that made the two numbers look like a contradiction.
CAPTURE_SHAPE = dict(B=2, T=128, H=2, HV=2, K=32, V=32, C=32)

# Prior figures, quoted as CHECK TARGETS. Never as inputs to anything.
PRIOR = {
    "launch_count_cuda_bench": 12082,   # perf_kda.log:31, DEBUG=2, T4
    "launch_count_cuda_copies": 139,     # perf_kda.py:52 filters " copy " out;
                                        # notes/13 §5 states 139
    "launch_count_metal_bench": 12090,   # notes/13 §5
    "debug2_kernel_sum_ms_bench": 215.880,   # perf_kda.log:32, UPPER BOUND
    "debug2_wall_ms_bench": 20342.8,     # perf_kda.log:38
    "numpy_ref_wall_ms_bench": 517.8,    # perf_kda.log:37
    "replay_cold_ms_t4": 96837.0,        # replay_kda.log:316
    "replay_warm_ms_t4": 19497.0,        # replay_kda.log:317
    "census_intercept_c64": 12042.0,     # notes/13 §6, SINK ~ 12,042 + 5*NT
    "census_slope_per_nt": 5.0,          # notes/13 §6
    "divide_cuda_pm2_wrong": 54270,      # divide_precision.log, N=200,000
    "divide_cuda_pm2_rate": 0.2713,
    "divide_cuda_aa_wrong": 30748,        # the a/a control, 15.37%
    "distinct_programs_bench_shape": 191,  # notes/13 §4, PARALLEL=0 required
    "launch_count_metal_bench_total": 12229,  # 12,090 sink + 139 copy
}
PRIOR_SOURCE = {
    "launch_count_cuda_bench": "out/kaggle/20260928-014248-gpu/logs/perf_kda.log:31",
    "launch_count_cuda_copies": "notes/13-schedule-vs-decay.md §5",
    "launch_count_metal_bench": "notes/13-schedule-vs-decay.md §5",
    "debug2_kernel_sum_ms_bench": "out/kaggle/20260928-014248-gpu/logs/perf_kda.log:32 (UPPER BOUND: per-launch host sync)",
    "debug2_wall_ms_bench": "out/kaggle/20260928-014248-gpu/logs/perf_kda.log:38",
    "numpy_ref_wall_ms_bench": "out/kaggle/20260928-014248-gpu/logs/perf_kda.log:37",
    "replay_cold_ms_t4": "out/kaggle/20260928-014248-gpu/logs/replay_kda.log:316",
    "replay_warm_ms_t4": "out/kaggle/20260928-014248-gpu/logs/replay_kda.log:317",
    "census_intercept_c64": "notes/13-schedule-vs-decay.md §6",
    "census_slope_per_nt": "notes/13-schedule-vs-decay.md §6",
    "divide_cuda_pm2_wrong": "out/kaggle/20260928-014248-gpu/logs/divide_precision.log",
    "divide_cuda_pm2_rate": "out/kaggle/20260928-014248-gpu/logs/divide_precision.log",
    "divide_cuda_aa_wrong": "out/kaggle/20260928-014248-gpu/logs/divide_precision.log §1",
    "distinct_programs_bench_shape": "notes/13-schedule-vs-decay.md §4 (to_program_cache entries at the bench shape; needs PARALLEL=0)",
    "launch_count_metal_bench_total": "notes/13-schedule-vs-decay.md §5 (12,090 sink + 139 copy = 12,229)",
}

STAGES = ["environment", "census", "launch_count", "instrument",
          "decomposition", "codegen", "divide", "ncu", "ratios"]


# ===========================================================================
# 0. plumbing: hashing, the result store, the budget, the stage runner
# ===========================================================================
def sha256_file(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{n:,.0f} B"
        n /= 1024.0
    return f"{n:,.1f} GB"


class Battery:
    """The result store. Rewritten to disk after every stage, so a session
    that is killed mid-run still leaves a readable results file."""

    def __init__(self, out_dir: str, budget_s: float):
        self.out = out_dir
        self.budget_s = budget_s
        self.t0 = time.perf_counter()
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "codegen"), exist_ok=True)
        self.results: dict = {
            "kind": "kda-single-file-battery",
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "battery_file": {"path": os.path.abspath(__file__),
                             "sha256": sha256_file(os.path.abspath(__file__))},
            "argv": sys.argv[1:],
            "host": {"platform": platform.platform(), "python": sys.version.split()[0],
                     "cwd": os.getcwd(), "executable": sys.executable},
            "prior_figures_used_as_check_targets": PRIOR,
            "prior_figure_sources": PRIOR_SOURCE,
            "stages": {},
            "order": [],
        }

    # -- budget -------------------------------------------------------------
    def spent(self) -> float:
        return time.perf_counter() - self.t0

    def left(self) -> float:
        return self.budget_s - self.spent()

    def log(self, msg: str) -> None:
        print(f"[{self.spent():7.1f}s] {msg}", flush=True)

    # -- persistence --------------------------------------------------------
    def flush(self) -> None:
        self.results["elapsed_s"] = round(self.spent(), 2)
        self.results["budget_s"] = self.budget_s
        tmp = os.path.join(self.out, ".results.json.tmp")
        with open(tmp, "w") as f:
            json.dump(self.results, f, indent=1, sort_keys=True, default=str)
        # rename inside the SAME directory: /tmp and /kaggle/working are
        # different filesystems and os.replace across a mount is EXDEV, which
        # is how the second real GPU run lost its summary.
        os.replace(tmp, os.path.join(self.out, "results.json"))

    def note(self, stage: str, **kw) -> None:
        st = self.results["stages"].setdefault(stage, {})
        st.update(kw)
        st["seconds"] = round(time.perf_counter() - self.results["stages"][stage].get("_t0", self.t0), 3)
        st.pop("_t0", None)
        self.flush()

    def write_text(self, rel: str, text: str) -> str:
        p = os.path.join(self.out, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(text)
        return p

    def run_stage(self, name: str, fn) -> dict:
        """One stage, one try/except. A stage that raises is recorded as
        FAILED with its traceback; every other stage still runs."""
        self.results["order"].append(name)
        st = self.results["stages"].setdefault(name, {"_t0": time.perf_counter()})
        if name in self.results["stages"] and self.results["stages"][name].get("status") == "SKIPPED":
            return self.results["stages"][name]
        if self.left() <= 5.0:
            st = {"status": "UNAVAILABLE", "reason": f"budget exhausted before this stage "
                   f"({self.spent():.0f}s of {self.budget_s:.0f}s used)"}
            self.results["stages"][name] = st
            self.flush()
            self.log(f"{name}: UNAVAILABLE (budget exhausted)")
            return st
        self.log(f"{name}: running")
        st["_t0"] = time.perf_counter()
        try:
            got = fn()
            st = dict(got or {})
            st.setdefault("status", "OK")
            st["seconds"] = round(time.perf_counter() - self.results["stages"][name]["_t0"], 3)
        except KeyboardInterrupt:
            st = {"status": "FAILED", "error": "KeyboardInterrupt"}
        except BaseException as e:  # noqa: BLE001 - a stage must never abort the battery
            st = {"status": "FAILED", "error": f"{type(e).__name__}: {e}",
                  "traceback": traceback.format_exc()[-2500:]}
        st.pop("_t0", None)
        st["manifest_id"] = self.results.get("manifest", {}).get("manifest_id")
        self.results["stages"][name] = st
        self.flush()
        self.log(f"{name}: {st.get('status')}"
                 + (f" ({st['error']})" if st.get("error") else ""))
        return self.results["stages"][name]

    # -- child processes ----------------------------------------------------
    def child(self, mode: str, shape: dict, env_extra: dict | None = None,
              timeout: float = 900.0) -> tuple[dict | None, str]:
        """Re-invoke this file as a child. A child gets a clean interpreter, so
        DEBUG is read at the right value and no rewrite memoization survives."""
        jout = os.path.join(self.out, "logs", f"{mode}-{shape.get('tag', 'shape')}.child.json")
        cmd = [sys.executable, os.path.abspath(__file__), "--_child", mode,
               "--child-out", jout, "--child-shape", json.dumps(shape),
               "--code-root", self.results.get("code_root", ".")]
        env = dict(os.environ)
        env.update(env_extra or {})
        env["PYTHONUNBUFFERED"] = "1"
        logp = os.path.join(self.out, "logs", f"{mode}-{shape.get('tag', 'shape')}.log")
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, env=env,
                               timeout=timeout, cwd=self.results.get("code_root") or None)
            out, err = p.stdout, p.stderr
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            err = "TIMEOUT after %.0fs" % timeout
            out = out + "\n" + err
        with open(logp, "w") as f:
            f.write("$ " + " ".join(cmd) + "\n" + out + "\n--- stderr ---\n" + err)
        data = None
        if os.path.exists(jout):
            try:
                with open(jout) as f:
                    data = json.load(f)
            except Exception as e:  # noqa: BLE001
                data = None
                err += f"\nchild json unreadable: {e}"
        if data is None:
            err = ANSI.sub("", err or "")[-1200:] + "\n" + ANSI.sub("", out or "")[-1200:]
        return data, err


# ===========================================================================
# 1. code discovery + the manifest
# ===========================================================================
SKIP_DIRS = {"__pycache__", ".git", ".venv", "payload", "out"}
SKIP_EXT = (".pyc", ".bin", ".so", ".png", ".json", ".lock")


def looks_like_root(p: str) -> bool:
    return os.path.isfile(os.path.join(p, "kda", "tiny_kda.py")) and \
           os.path.isfile(os.path.join(p, "kda", "test_ref.py"))


def discover_root(explicit: str | None) -> str | None:
    cands: list[str] = []
    if explicit:
        cands.append(explicit)
    cands.append(os.path.dirname(HERE))                 # <repo>/remote/..
    cands.append(os.path.dirname(os.path.dirname(HERE)))  # <payload>/probe/remote/..
    cands.append(os.getcwd())
    cands.append(os.path.join(os.getcwd(), "probe"))
    cands.append(os.path.dirname(os.getcwd()))
    cands.extend(sorted(glob.glob("/kaggle/input/*/probe")))
    cands.extend(sorted(glob.glob("/kaggle/input/*")))
    for c in cands:
        if c and looks_like_root(c):
            return os.path.abspath(c)
    return None


def walk_code(root: str) -> list[str]:
    """Every .py under kda/ and probes/ that a stage could import, plus
    remote/. Deliberately not every file: the ones that matter are the ones
    whose bytes end up in a number."""
    out = []
    for sub in ("kda", "probes", "remote"):
        base = os.path.join(root, sub)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in sorted(filenames):
                if fn.endswith(SKIP_EXT) or os.path.getsize(os.path.join(dirpath, fn)) > (4 << 20):
                    continue
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def pack_include_list(root: str) -> list[str] | None:
    """`kernels/pack.py` owns the list of what travels. Read it if it is here,
    so the manifest can say which of the files it sees are NOT in the payload."""
    p = os.path.join(root, "kernels", "pack.py")
    if not os.path.isfile(p):
        return None
    try:
        import ast
        tree = ast.parse(open(p).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "INCLUDE" for t in node.targets):
                return [ast.literal_eval(e) for e in node.value.elts]  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        return None
    return None


def build_manifest(bat: Battery, root: str) -> dict:
    files = walk_code(root)
    entries, total = [], 0
    for p in files:
        rel = os.path.relpath(p, root)
        sha, nbytes = sha256_file(p), os.path.getsize(p)
        entries.append({"path": rel, "sha256": sha, "bytes": nbytes})
        total += nbytes
    tok = "".join(f"{e['path']}:{e['sha256']}\n" for e in entries)
    manifest_id = hashlib.sha256(tok.encode()).hexdigest()

    tar = None
    for cand in (os.path.join(root, "kernels", "payload", "payload.tgz"),
                 os.path.join(root, "payload.tgz"),
                 "/kaggle/input" and None):
        if cand and os.path.isfile(cand):
            tar = {"path": cand, "sha256": sha256_file(cand),
                   "bytes": os.path.getsize(cand)}
            break
    sidecar = None
    for cand in (os.path.join(root, "kernels", "payload", "payload.sha256"),
                 os.path.join(root, "payload.sha256")):
        if os.path.isfile(cand):
            sidecar = open(cand).read().strip()
            break

    include = pack_include_list(root)
    rels = {e["path"] for e in entries}
    in_pack = None
    if include:
        packed = {src for src, _ in include}
        in_pack = {"present": sorted(packed & rels),
                   "declared_but_absent": sorted(packed - rels),
                   "here_but_not_in_pack": sorted(rels - packed)}

    return {
        "code_root": root,
        "manifest_id": manifest_id,
        "n_files": len(entries),
        "total_bytes": total,
        "files": entries,
        "payload_tarball": tar,
        "payload_sidecar": sidecar,
        "payload_tarball_matches_sidecar": (
            None if not (tar and sidecar) else (tar["sha256"] in sidecar)),
        "pack_include": in_pack,
        # A per-run session identity, so a log from a different Kaggle session
        # is distinguishable by eye from this one.
        "session": {k: os.environ.get(k) for k in
                    ("KAGGLE_CONTAINER_NAME", "KAGGLE_KERNEL_RUN_ID", "KAGGLE_KERNEL_ID")},
        "device": None,   # filled by stage_environment
    }


def rehash_manifest(man: dict) -> tuple[bool, list[str]]:
    """Re-hash everything and compare. A stage that reports a different set of
    bytes than the top of the run is measuring different code than the manifest
    describes, and says so."""
    changed = []
    for e in man["files"]:
        p = os.path.join(man["code_root"], e["path"])
        if sha256_file(p) != e["sha256"]:
            changed.append(e["path"])
    return (not changed), changed


def imported_under_root(root: str) -> dict:
    """Where each module actually came from. This is the check the previous run
    did not have: `perf_kda.log:2` records `cwd kda` while
    `divide_precision.log:2` records `cwd .`, and nothing recorded which bytes
    either of them loaded.

    Split in two on purpose. `kda` and `probes` ARE the code under test and must
    live under the code root that the manifest hashed. tinygrad and numpy are
    dependencies: they legitimately live in a venv that may be inside the repo
    (`.venv/`), so "under the code root" is true for them and means nothing. A
    site-packages path under the repo root must not be reported as if it were
    the code under test.
    """
    under_test, deps = {}, {}
    for name in ("tiny_kda", "test_ref", "tinygrad", "numpy"):
        m = sys.modules.get(name)
        if m is None:
            continue
        f = getattr(m, "__file__", None)
        p = os.path.abspath(f) if f else None
        in_root = bool(p and p.startswith(root + os.sep))
        rec = {"file": p, "sha256": sha256_file(p) if p else None,
               "in_code_root": in_root,
               "in_a_venv_or_site_packages": bool(p and ("site-packages" in p or "/.venv/" in p))}
        if name in ("tiny_kda", "test_ref"):
            rec["verdict"] = ("OK: the code under test, and its bytes are in the manifest"
                              if in_root else
                              "MISMATCH: the code under test was imported from OUTSIDE the "
                              "manifested tree. Every number this battery prints is "
                              "attributed to code it did not hash.")
            under_test[name] = rec
        else:
            rec["verdict"] = ("dependency: hashed for the record, not part of the payload"
                              if in_root else "dependency: outside the code root")
            deps[name] = rec
    return {"code_under_test": under_test, "dependencies": deps}


def stage_manifest(bat: Battery) -> dict:
    root = bat.results.get("code_root")
    if not root:
        return {"status": "FAILED", "error": f"no code root found; pass --code-root. "
                f"Tried: {HERE}/.., cwd, /kaggle/input/*/probe"}
    man = build_manifest(bat, root)
    bat.results["code_root"] = root
    bat.results["manifest"] = man
    bat.write_text("MANIFEST.txt", "\n".join([
        f"code_root      {root}",
        f"manifest_id    {man['manifest_id']}",
        f"n_files        {man['n_files']}   {human(man['total_bytes'])}",
        f"payload tgz    {man['payload_tarball'] or 'not present on this box'}",
        f"sidecar        {man['payload_sidecar'] or 'not present'}",
        f"session        {json.dumps(man['session'])}",
        "",
        f"{'bytes':>10}  sha256                                                            path",
    ] + [f"{e['bytes']:>10}  {e['sha256']}  {e['path']}" for e in man["files"]]))
    bat.write_text("manifest.json", json.dumps(man, indent=1))
    bat.log(f"manifest: {man['n_files']} files, {human(man['total_bytes'])}, "
            f"id {man['manifest_id'][:16]}")
    return {"status": "OK", "n_files": man["n_files"], "total_bytes": man["total_bytes"],
            "manifest_id": man["manifest_id"], "payload_tarball": man["payload_tarball"],
            "payload_tarball_matches_sidecar": man["payload_tarball_matches_sidecar"],
            "pack_include_summary": (None if not man["pack_include"] else {
                "n_declared": len(man["pack_include"]["present"]) + len(man["pack_include"]["declared_but_absent"]),
                "declared_but_absent": man["pack_include"]["declared_but_absent"],
                "here_but_not_in_pack": man["pack_include"]["here_but_not_in_pack"]})}


# ===========================================================================
# 2. environment
# ===========================================================================
DATASHEET_T4 = {"sm_count": 40, "lanes_per_sm": 2048, "boost_ghz": 1.410,
                "dram_gb_s": 320.0, "mib": 15360, "cc": "7.5",
                "status": "DATASHEET - published spec, NOT measured here, "
                          "never used in any derived number in this file"}


def smi(query: str) -> str:
    try:
        p = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
        return (p.stdout or p.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {type(e).__name__}: {e}"


def stage_environment(bat: Battery) -> dict:
    from tinygrad import Device  # noqa: PLC0415

    dev = Device.DEFAULT
    try:
        known = list(getattr(Device, "_devices", []))
    except Exception:  # noqa: BLE001
        known = []
    info: dict = {"device": dev, "device_names_tinygrad_knows": known,
                  "known_is_not_instantiated": "Device._devices is the list of device "
                  "NAMES tinygrad knows, not the ones opened; every stage below "
                  "records which ones actually answered"}
    try:
        info["tinygrad_version"] = __import__("tinygrad").__dict__.get("__version__", "unknown")
    except Exception:  # noqa: BLE001
        info["tinygrad_version"] = "unknown"
    info["numpy"] = np.__version__
    info["platform"] = platform.platform()

    # The floor every synchronize-based number has to be read against. On an
    # idle device this is what `synchronize()` costs by itself; without it, a
    # "device drain" of 2.6 ms on METAL is indistinguishable from real work.
    floor = None
    try:
        d = Device[dev]
        ts = []
        for _ in range(20):
            t0 = time.perf_counter()
            d.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        ts.sort()
        floor = {"n": len(ts), "median_ms": ts[len(ts) // 2], "min_ms": ts[0],
                 "max_ms": ts[-1],
                 "note": "MEASURED: cost of synchronize() on an IDLE device"}
    except Exception as e:  # noqa: BLE001
        floor = {"error": f"{type(e).__name__}: {e}"}
    info["sync_floor"] = floor

    if dev.startswith("CUDA") or dev.startswith("METAL") or dev.startswith("CL"):
        info["nvidia_smi_name"] = smi("name")
        info["nvidia_smi_mem"] = smi("memory.total")
        info["nvidia_smi_cc"] = smi("compute_cap")
        info["nvidia_smi_clocks_max_sm"] = smi("clocks.max.sm")
        info["nvidia_smi_clocks_sm_now"] = smi("clocks.sm")
        info["nvidia_smi_clocks_note"] = (
            "READ-FROM-SM-IQUERY. The boost clock nvidia-smi reports is the CURRENT "
            "clock, which is a load-dependent reading and not the T4's rated boost. "
            "The rated figure is in t4_datasheet and is labelled DATASHEET. Neither "
            "is used in any derived number in this file.")
        info["nvidia_smi_driver"] = smi("driver_version")
        info["nvidia_smi_persistence"] = smi("persistence_mode")
    else:
        for k in ("nvidia_smi_name", "nvidia_smi_mem", "nvidia_smi_cc",
                  "nvidia_smi_clocks_max_sm", "nvidia_smi_clocks_sm_now",
                  "nvidia_smi_driver", "nvidia_smi_persistence"):
            info[k] = f"UNAVAILABLE: device is {dev}, not a CUDA device"
    info["t4_datasheet"] = DATASHEET_T4
    info["imported_modules"] = imported_under_root(bat.results["code_root"])
    info["device_path_check"] = {
        "device_is_cuda": dev.startswith("CUDA"),
        "note": "nvidia-smi and every CUDA-specific field are UNAVAILABLE, not "
                "zero, on a non-CUDA device. No GPU number is invented here."}
    imp = imported_under_root(bat.results["code_root"])
    mism = [k for k, v in imp["code_under_test"].items() if "MISMATCH" in v["verdict"]]
    info["imported_modules"] = imp
    info["code_under_test_mismatch"] = mism or None
    if mism:
        info["code_under_test_mismatch_note"] = (
            f"{mism} imported from outside the manifested tree. Treat every number in "
            f"this run as attributed to code the manifest does not describe.")
    bat.results["manifest"]["device"] = dev
    bat.write_text("environment.json", json.dumps(info, indent=1, default=str))
    if floor and "median_ms" in floor:
        bat.log(f"device={dev}  sync_floor={floor['median_ms']:.4f} ms (idle)")
    else:
        bat.log(f"device={dev}  sync_floor=UNAVAILABLE ({floor})")
    return info


# ===========================================================================
# 3. shapes
# ===========================================================================
def shape_str(s: dict) -> str:
    C, T = s["C"], s["T"]
    return (f"B={s['B']} T={T} H={s['H']} HV={s['HV']} K={s['K']} V={s['V']} "
            f"C={C} NT={T // C if C else '?'}")


def ladder(quick: bool) -> tuple[list[dict], list[dict], list[dict]]:
    """(census shapes, device-launch shapes, decomposition shape).

    The T sweep at fixed C is where the two hypotheses are furthest apart: a
    count that is `a + b*NT` is flat over an 8x span of T, and a count that is
    proportional to T moves 8x over the same span. The C sweep is what shows
    the constant is a property of the unrolled forward substitution
    (`kda/tiny_kda.py:178-194`, `C(C-1)` iterations) and therefore of C.
    """
    if quick:
        # both C values must appear at the SAME NT values, or there is no C span
        # and the second half of the separation cannot be drawn at all
        base = dict(B=2, H=2, HV=2, K=16, V=16)
        cen = [dict(base, T=C * NT, C=C, tag=f"c{C}nt{NT}")
               for C in (8, 16) for NT in (1, 4)]
        dev = [dict(base, T=64, C=16, tag="dev-c16nt4")]
        dec = dict(base, T=64, C=16, tag="dec")
        return cen, dev, dec
    base = dict(B=2, H=4, HV=4, K=64, V=64)
    cen = [dict(base, T=64 * NT, C=64, tag=f"c64nt{NT}") for NT in (1, 2, 4, 8, 16)]
    cen += [dict(base, T=8 * C, C=C, tag=f"c{C}nt8") for C in (16, 32, 64)]
    dev = [dict(base, T=64, C=64, tag="dev-c64nt1"),
           dict(base, T=512, C=64, tag="dev-c64nt8"),
           dict(base, T=256, C=32, tag="dev-c32nt8")]
    dec = dict(base, T=512, C=64, tag="dec-c64nt8")
    return cen, dev, dec


def _fit(X, y):
    import numpy as np  # noqa: PLC0415
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return coef, float(np.sqrt((resid ** 2).mean())), float(np.abs(resid).max())


def fit_launch_models(points: list[dict]) -> dict:
    """Two competing readings of the same sweep, plus a null.

      const_of_C : count = a*C*(C-1) + b*NT + c     (C(C-1) = 4,032 at C=64 is the
                    unrolled forward substitution; b is the inter-chunk slope)
      prop_T     : count = p*T + q                 (the launch count is
                    proportional to the problem size)
      null       : count = k                       (it depends on nothing)

    A fit is a fit. It is reported with its worst residual and never as a
    measurement, and the verdict is which model's worst residual is smallest
    relative to the mean count - not which one looks tidier.
    """
    if len(points) < 3:
        return {"status": "UNAVAILABLE", "reason": f"only {len(points)} points; a fit needs >= 3"}
    C = [p["C"] for p in points]
    NT = [p["T"] // p["C"] for p in points]
    Tt = [p["T"] for p in points]
    y = [p["sink"] for p in points]
    mean = sum(y) / len(y)
    models = {
        # C(C-1) is the unrolled forward substitution at kda/tiny_kda.py:178-194
        "const_of_C": [[C[i] * (C[i] - 1), NT[i], 1.0] for i in range(len(C))],
        "prop_T": [[Tt[i], 1.0] for i in range(len(C))],
        "null": [[1.0] for _ in C],
    }
    labels = {"const_of_C": ["a_per_C(C-1)", "b_per_NT", "c_intercept"],
              "prop_T": ["p_per_T", "q"], "null": ["k"]}
    out: dict = {"n_points": len(points), "mean_sink": mean, "models": {},
                 "status": "MODELLED"}
    for name, X in models.items():
        if not X:
            continue
        coef, rms, mx = _fit(X, y)
        out["models"][name] = {
            "coef": dict(zip(labels[name], [float(c) for c in coef])),
            "rms_abs": rms, "max_abs": mx,
            "max_rel": (mx / mean) if mean else None,
            "fits_1pct": bool(mean and mx / mean < 0.01),
        }
    ranked = sorted(out["models"].items(), key=lambda kv: kv[1]["max_rel"])
    out["best_by_max_residual"] = ranked[0][0]
    out["verdict"] = (
        f"{ranked[0][0]} has the smallest worst-case residual "
        f"({ranked[0][1]['max_rel']*100:.3f}% of the mean count {mean:,.0f}); "
        f"const_of_C={out['models'].get('const_of_C', {}).get('max_rel', float('nan'))*100:.3f}%, "
        f"prop_T={out['models'].get('prop_T', {}).get('max_rel', float('nan'))*100:.3f}%")
    b = out["models"].get("const_of_C", {}).get("coef", {}).get("b_per_NT")
    if b is not None:
        out["slope_per_NT_measured"] = b
        out["slope_per_NT_prior"] = PRIOR["census_slope_per_nt"]
        out["intercept_at_C64_measured"] = \
            out["models"]["const_of_C"]["coef"]["c_intercept"] + \
            out["models"]["const_of_C"]["coef"]["a_per_C(C-1)"] * 64 * 63
        out["intercept_at_C64_prior"] = PRIOR["census_intercept_c64"]
    return out


def stage_census(bat: Battery) -> dict:
    """Device-free launch count, one shape per child process.

    No device, no execution, no allocation: `Tensor.numpy` is redirected to
    `Tensor.schedule_linear()`, which is the same rewrite chain `realize()` runs
    (engine/realize.py:317 -> compile_linear -> memory_plan_rewrite) and stops
    before codegen. So this is a count of what the scheduler WOULD launch.
    """
    from tinygrad import Device  # noqa: PLC0415
    dev = Device.DEFAULT
    census_dev = "CUDA" if os.environ.get("KDA_CENSUS_DEV") is None else os.environ["KDA_CENSUS_DEV"]
    if not os.environ.get("KDA_CENSUS_DEV"):
        census_dev = "CUDA" if dev.startswith("CUDA") else dev
    shapes, _, _ = ladder(bat.results.get("quick", False))
    rows, per_shape_s = [], []
    for s in shapes:
        if bat.left() < 20.0:
            per_shape_s.append({"shape": shape_str(s), "status": "UNAVAILABLE",
                                "reason": "budget exhausted"})
            continue
        t0 = time.perf_counter()
        data, err = bat.child("census", s, env_extra={"KDA_CENSUS_DEV": census_dev},
                              timeout=min(1200.0, max(120.0, bat.left() * 0.9)))
        wall = time.perf_counter() - t0
        if not data or "sink" not in data:
            per_shape_s.append({"shape": shape_str(s), "status": "UNAVAILABLE",
                                "reason": "census child did not return a count",
                                "child_stderr_tail": err[-800:]})
            continue
        row = dict(shape=data.get("shape", shape_str(s)), C=s["C"], T=s["T"],
                   NT=s["T"] // s["C"], census_dev=data.get("dev", census_dev),
                   sink=data["sink"], copy=data["copy"], total=data["sink"] + data["copy"],
                   child_wall_s=round(wall, 2),
                   wall_caveat="first pass in a fresh child process; tinygrad's "
                               "rewrite passes memoize, so a repeat in the SAME "
                               "process runs ~5x faster and its wall is NOT a "
                               "scheduler cost")
        rows.append(row)
        per_shape_s.append(row)
        bat.log(f"  census {shape_str(s):<48} sink={row['sink']:>7,} copy={row['copy']:>4,} "
                f"({wall:.1f}s)")

    for r in rows:
        if r["C"] == 64 and r["NT"] == 8 and r["census_dev"] == "CUDA":
            r["matches_prior_cuda_bench"] = (r["sink"] == PRIOR["launch_count_cuda_bench"])
            r["matches_prior_cuda_copies"] = (r["copy"] == PRIOR["launch_count_cuda_copies"])
            r["prior_source"] = PRIOR_SOURCE["launch_count_cuda_bench"]
        if r["C"] == 64 and r["NT"] == 8 and r["census_dev"] == "METAL":
            r["matches_prior_metal_bench"] = (r["sink"] == PRIOR["launch_count_metal_bench"])
    models = fit_launch_models(rows)
    bat.write_text("launch_counts.csv",
                   "census_dev,C,T,NT,sink,copy,total,child_wall_s\n" +
                   "\n".join(f"{r['census_dev']},{r['C']},{r['T']},{r['NT']},"
                             f"{r['sink']},{r['copy']},{r['total']},{r['child_wall_s']}"
                             for r in rows))
    st: dict = {"status": "OK" if rows else "UNAVAILABLE",
                "census_dev": census_dev,
                "note": "device-free. The count is what the scheduler would launch "
                        "for this shape; the DEVICE count is stage launch_count. "
                        "Every row carries C, T and NT, because a bare launch count "
                        "is not interpretable: the prior 12,082 is a statement about "
                        "C=64 and moves with C.",
                "n_shapes_run": len(rows), "n_shapes_total": len(shapes),
                "shapes": per_shape_s, "models": models}
    if not rows:
        st["status"] = "UNAVAILABLE"
        st["reason"] = "no census child returned a count"
    return st


def stage_launch_count(bat: Battery) -> dict:
    """The SAME count, measured by the device, so census and device can be
    compared at one shape instead of trusted to agree."""
    from tinygrad import Device  # noqa: PLC0415
    dev = Device.DEFAULT
    _, dev_shapes, _ = ladder(bat.results.get("quick", False))
    rows = []
    for s in dev_shapes:
        if bat.left() < 60.0:
            rows.append({"shape": shape_str(s), "status": "UNAVAILABLE",
                         "reason": "budget exhausted"})
            continue
        data, err = bat.child("launch", s, env_extra={"DEBUG": "0"},
                              timeout=min(2400.0, max(120.0, bat.left() * 0.9)))
        if not data:
            rows.append({"shape": shape_str(s), "status": "UNAVAILABLE",
                         "reason": "launch child did not return",
                         "child_stderr_tail": err[-800:]})
            continue
        row = dict(shape=shape_str(s), C=s["C"], T=s["T"], NT=s["T"] // s["C"],
                   device=data.get("device", dev))
        row.update(data)
        # Cross-check against the census at the SAME shape, and say whether the
        # census ran under the SAME backend. It usually does not: a census can be
        # run under any device name (it needs no device), and the count is
        # backend-dependent by exactly the DEVICE_MAX_BUFS entry at
        # tinygrad/schedule/rangeify.py:354 -- 8 more elementwise kernels on
        # METAL/CPU than on CUDA, which has no entry and so never runs the
        # buffer-limiting rewrite. Comparing a CUDA census against a METAL device
        # and calling the 8 a disagreement would be a category error.
        for c in bat.results["stages"].get("census", {}).get("shapes", []):
            if c.get("C") == s["C"] and c.get("NT") == s["T"] // s["C"] and "sink" in c:
                dev_launches = data.get("warm_launches")
                same_backend = c["census_dev"] == data.get("device", dev)
                row["census_same_shape"] = {"census_dev": c["census_dev"],
                                            "device": data.get("device", dev),
                                            "same_backend": same_backend,
                                            "sink": c["sink"], "copy": c["copy"]}
                if dev_launches is None:
                    row["device_sink"] = None
                    row["device_minus_census"] = None
                else:
                    row["device_sink"] = dev_launches - c["copy"]
                    row["device_minus_census"] = row["device_sink"] - c["sink"]
                    row["device_total_minus_census_total"] = \
                        dev_launches - (c["sink"] + c["copy"])
                row["comparison_verdict"] = (
                    "AGREE, same backend" if same_backend and row["device_minus_census"] == 0
                    else ("DISAGREE, same backend" if same_backend else
                          f"census ran under {c['census_dev']}, device is "
                          f"{data.get('device', dev)}: a difference here is the "
                          f"backends' DEVICE_MAX_BUFS entry, not an error. "
                          f"device_minus_census = {row['device_minus_census']}"))
                row["copy_count_source"] = ("census, by AST op (Ops.COPY). "
                                            "GlobalCounters.kernel_count counts copies "
                                            "and kernels together and does not split them.")
        rows.append(row)
        bat.log(f"  launch {shape_str(s):<48} warm={data.get('warm_launches')} "
                f"cold={data.get('cold_launches')}")
    ran = [r for r in rows if r.get("warm_launches") is not None]
    st = {"status": "OK" if ran else "UNAVAILABLE",
          "device": dev,
          "device_is_cuda": dev.startswith("CUDA"),
          "shapes": rows,
          "instrument": "tinygrad.helpers.GlobalCounters.kernel_count delta around one "
                        "warm kda_chunk. No DEBUG, no regex, no subprocess stdout. "
                        "Counts EVERY launch the runtime issues, including copies.",
          "device_caveat":
              f"This is the count for the {dev} backend. It is NOT the T4's count by "
              f"assumption: tinygrad/schedule/rangeify.py:354 has "
              f"DEVICE_MAX_BUFS = {{\"METAL\": 31, \"WEBGPU\": 8, \"CPU\": 31}} and NO "
              f"CUDA entry, so the buffer-limiting rewrite that splits kernels runs on "
              f"CPU and METAL and not on CUDA. A non-CUDA count is therefore evidence "
              f"about the schedule on that backend, and the difference between backends "
              f"is attributable to that one line.",
          "count_stable_across_reps":
              all(r.get("warm_launches_all") and len(set(r["warm_launches_all"])) == 1
                  for r in ran) if ran else None,
          "reason": None if ran else
                    f"no launch child returned a count on device {dev}"}
    if not ran:
        st["status"] = "UNAVAILABLE"
    return st


# ===========================================================================
# 4. the instrument comparison and the decomposition, in one child
# ===========================================================================
CHILD_INSTR = r'''
def child_instr(args, shape, out_path):
    import json, time
    import numpy as np
    from tinygrad import Device, Tensor
    from tinygrad.helpers import GlobalCounters
    root = args.code_root
    for p in (root, os.path.join(root, "kda")):
        if p not in sys.path: sys.path.insert(0, p)
    import tiny_kda, test_ref as T
    out = {"mode": "instr", "device": Device.DEFAULT, "shape": shape}
    C = shape["C"]
    args_ = (shape["B"], shape["T"], shape["H"], shape["HV"], shape["K"], shape["V"])
    q, k, v, g, beta = T.rnd(*args_, seed=0, l2n=True)
    dev = Device[Device.DEFAULT]
    real_numpy = Tensor.numpy

    def call(): return tiny_kda.kda_chunk(q, k, v, g, beta, chunk_size=C)

    # ---- COLD: the first call, which compiles. In THIS process, so no
    # ---- program cache carried in from anywhere.
    t0 = time.perf_counter(); GlobalCounters.reset()
    o, s = call()
    cold_wall = (time.perf_counter() - t0) * 1e3
    out["cold"] = {"wall_ms": cold_wall,
                   "launches": GlobalCounters.kernel_count,
                   "finite": bool(np.isfinite(o).all() and np.isfinite(s).all()),
                   "note": "MEASURED, includes compilation. This is the number "
                           "replay_kda.py:105 hard-coded as 49.5 s while the same "
                           "script measured 96,837 ms on the T4."}

    # ---- uncounted launches: optimize_local_size's try_exec calls real
    # ---- kernels during autotuning and bypasses the ExecContext, so they
    # ---- are in neither DEBUG=2 nor kernel_count.
    try:
        from tinygrad.engine import realize as _rz
        n_ls = len(getattr(_rz, "local_size_cache", {}))
        out["local_size_autotune"] = {"cached_local_sizes_after_cold": n_ls,
                                      "note": "SOURCE engine/realize.py:98-121: each "
                                              "uncached program is benchmarked with real "
                                              "cuLaunchKernel calls that no counter sees"}
    except Exception as e:
        out["local_size_autotune"] = {"error": f"{type(e).__name__}: {e}"}

    # ---- idle synchronize() floor, on this device, right here
    fl = []
    for _ in range(20):
        t0 = time.perf_counter(); dev.synchronize(); fl.append((time.perf_counter()-t0)*1e3)
    fl.sort(); out["sync_floor_ms"] = {"median": fl[len(fl)//2], "min": fl[0], "max": fl[-1],
                                       "note": "MEASURED: synchronize() on an idle device"}

    # ---- the instrumented warm pass -------------------------------------
    recs, h2d, d2h = [], [], []
    try:
        al = dev.allocator
        o_in, o_out = al._copyin, al._copyout
        def wi(dest, src):
            t0 = time.perf_counter(); r = o_in(dest, src)
            h2d.append([len(src), (time.perf_counter()-t0)*1e3]); return r
        def wo(dest, src):
            t0 = time.perf_counter(); r = o_out(dest, src)
            d2h.append([len(dest), (time.perf_counter()-t0)*1e3]); return r
        al._copyin, al._copyout = wi, wo
        out["copy_hooks_installed"] = True
    except Exception as e:
        out["copy_hooks_installed"] = False
        out["copy_hook_error"] = f"{type(e).__name__}: {e}"

    def split_numpy(self, *a, **kw):
        t0 = time.perf_counter()
        self.realize()                     # fresh graph: kda_chunk rebuilt it
        t1 = time.perf_counter()
        dev.synchronize()                  # exactly one sync per realize
        t2 = time.perf_counter()
        r = real_numpy(self, *a, **kw)     # D2H only; the graph is realized
        t3 = time.perf_counter()
        nb = getattr(self, "nbytes", 0)
        recs.append({"shape": list(self.shape), "nbytes": int(nb() if callable(nb) else nb),
                     "realize_ms": (t1-t0)*1e3, "sync_ms": (t2-t1)*1e3,
                     "copyout_ms": (t3-t2)*1e3})
        return r

    warm = []
    Tensor.numpy = split_numpy
    for i in range(3):
        # reset PER ITERATION, or the counter accumulates and the per-iteration
        # launch figure is n x the truth. The DEBUG=2 path has the same trap.
        GlobalCounters.reset()
        recs.clear(); n0 = len(h2d); m0 = len(d2h)
        t0 = time.perf_counter()
        try:
            o, s = call()
            wall = (time.perf_counter()-t0)*1e3
            warm.append({"iter": i, "wall_ms": wall, "realize_events": list(recs),
                         "launches": GlobalCounters.kernel_count,
                         "finite": bool(np.isfinite(o).all() and np.isfinite(s).all()),
                         "h2d": h2d[n0:], "d2h_hook": d2h[m0:]})
        except Exception as e:
            warm.append({"iter": i, "error": f"{type(e).__name__}: {e}"})
            break
    Tensor.numpy = real_numpy
    out["warm"] = warm
    out["debug_env"] = os.environ.get("DEBUG")

    # ---- how much the instrument itself perturbs the count -------------
    # Measured, not assumed. Splitting realize() from numpy() can add work of
    # its own (an already-realized tensor may need a contiguous copy that the
    # fused path folds into the same schedule), so the launch count under the
    # probe is not the launch count of the operator. Both are reported, and the
    # difference is the probe's own cost.
    clean = []
    for _ in range(3):
        GlobalCounters.reset()
        t0 = time.perf_counter()
        call()
        clean.append({"wall_ms": (time.perf_counter()-t0)*1e3,
                      "launches": GlobalCounters.kernel_count})
    out["clean_warm"] = clean

    # ---- build only, and build + rewrite passes, no codegen no launch ----
    def mode_wall(mode, n=3):
        if mode == "build": Tensor.numpy = lambda self, *a, **k: self
        elif mode == "sched": Tensor.numpy = lambda self, *a, **k: self.schedule_linear()
        else: Tensor.numpy = real_numpy
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            try: call()
            except Exception as e: ts = []; out.setdefault("mode_errors", {})[mode] = f"{type(e).__name__}: {e}"; break
            ts.append((time.perf_counter()-t0)*1e3)
        Tensor.numpy = real_numpy
        return ts
    out["build_only_ms"] = mode_wall("build")
    out["build_plus_rewrite_ms"] = mode_wall("sched")
    json.dump(out, open(out_path, "w"), indent=1, default=str)
    print("child instr done", file=sys.stderr)
'''


def _derive_decomposition(instr: dict) -> dict:
    """Turn the instrumented warm pass into a decomposition whose parts add up
    to what was measured, with the residual shown rather than distributed."""
    warm = [w for w in instr.get("warm", []) if "error" not in w]
    if not warm:
        return {"status": "UNAVAILABLE", "reason": "no clean warm iteration"}
    best = min(warm, key=lambda w: w["wall_ms"])
    ev = best["realize_events"]
    build = min(instr.get("build_only_ms") or [0.0])
    sched = min(instr.get("build_plus_rewrite_ms") or [0.0])
    n_ev = len(ev)
    parts = {
        "realize_and_issue_ms": sum(e["realize_ms"] for e in ev),
        "device_drain_ms": sum(e["sync_ms"] for e in ev),
        "copyout_ms": sum(e["copyout_ms"] for e in ev),
    }
    d: dict = {
        "status": "OK",
        "shape": instr.get("shape"),
        "device": instr.get("device"),
        "basis": f"best of {len(warm)} warm iterations, wall {best['wall_ms']:.1f} ms; "
                 f"{n_ev} realize event(s), {n_ev} synchronize() call(s)",
        "graph_build_ms": {"value": build, "instrument": "MEASURED: Tensor.numpy redirected "
                           "to a no-op, so the call builds UOps and never realizes"},
        "rewrite_passes_ms": {"value": max(0.0, sched - build),
                              "instrument": "RE-DERIVED: Tensor.numpy redirected to "
                              "schedule_linear() (build+rewrite) minus graph_build",
                              "build_plus_rewrite_ms": sched},
        "realize_and_issue_ms": {"value": parts["realize_and_issue_ms"],
                                 "instrument": "MEASURED: schedule + lower + compile(cached) "
                                 "+ issue, timed around realize() with the synchronize "
                                 "DEFERRED so the host-side part is visible"},
        "device_drain_ms": {"value": parts["device_drain_ms"],
                            "instrument": "MEASURED: the one synchronize() after issue. "
                                          "Upper bound on the device tail; subtract the "
                                          "idle synchronize() floor before reading it.",
                            "idle_sync_floor_ms": instr.get("sync_floor_ms", {}).get("median")},
        "copyout_ms": {"value": parts["copyout_ms"],
                       "instrument": "MEASURED: host->device is NOT here; this is the "
                                     "D2H of the fetched outputs, and on most backends "
                                     "the copy path runs its own device sync, so this "
                                     "includes an idle-drain cost."},
        "h2d_host_ms": {"value": sum(h[1] for e in ev for h in e.get("h2d", [])),
                        "n_calls": sum(len(e.get("h2d", [])) for e in ev),
                        "n_bytes": sum(h[0] for e in ev for h in e.get("h2d", [])),
                        "instrument": "MEASURED by wrapping the device allocator's _copyin. "
                                      "This is ENQUEUE time on an async copy. Zero means "
                                      "this backend does not route host->device through "
                                      "_copyin (a shared-memory allocator writes through a "
                                      "mapping instead), NOT that there was no upload."},
        "d2h_hook_ms": {"value": sum(h[1] for e in ev for h in e.get("d2h_hook", [])),
                        "n_calls": sum(len(e.get("d2h_hook", [])) for e in ev),
                        "n_bytes": sum(h[0] for e in ev for h in e.get("d2h_hook", [])),
                        "instrument": "MEASURED by wrapping _copyout; a cross-check on copyout_ms"},
    }
    residual = best["wall_ms"] - build - sum(
        d[k]["value"] for k in ("realize_and_issue_ms", "device_drain_ms", "copyout_ms"))
    d["unattributed_ms"] = {
        "value": residual,
        "instrument": "RE-DERIVED: wall(instrumented) - graph_build(best of 3 build-only "
                      "runs) - (realize+issue) - drain - copyout. The three realize-side "
                      "terms come from the single best instrumented iteration; graph_build "
                      "and the rewrite passes come from SEPARATE best-of-3 runs, because "
                      "they cannot be measured inside the instrumented pass without "
                      "changing it. A small NEGATIVE residual therefore means the parts "
                      "over-count the wall by that much, not that time was created. It is "
                      "reported with its sign and not clamped or distributed.",
        "magnitude_ms": abs(residual),
        "components_from_different_best_of_runs": True}
    d["kernel_ms"] = {"value": None, "instrument": "UNAVAILABLE in this stage: on a "
                     "compiled backend with no non-blocking per-kernel timer, per-kernel "
                     "device time is only obtainable with DEBUG=2, which is a DIFFERENT "
                     "run. stage instrument reports it from there, as an upper bound."}
    d["lower_and_compile_ms"] = {
        "value": None,
        "instrument": "UNAVAILABLE: cannot be separated from issue without a profiler. "
                      "realize_and_issue_ms contains it. Reporting a split here would be "
                      "a projection."}
    d["launches_this_iteration"] = best.get("launches")
    d["finite_output"] = best.get("finite")
    d["check_arithmetic"] = {
        "wall_ms": best["wall_ms"],
        "graph_build_plus_realize_plus_drain_plus_copyout":
            build + parts["realize_and_issue_ms"] + parts["device_drain_ms"] + parts["copyout_ms"],
        "residual_ms": residual,
        "reading": "graph_build + (realize_and_issue) + drain + copyout + residual == wall. "
                   "rewrite_passes_ms is a COMPONENT OF realize_and_issue_ms, not an "
                   "addition to it: schedule_linear() is what realize() calls. Do not "
                   "add the two lines."}
    return d


def stage_instrument(bat: Battery) -> dict:
    """DEBUG=0 + one synchronize(), against DEBUG=2, in two processes."""
    from tinygrad import Device  # noqa: PLC0415
    dev = Device.DEFAULT
    _, _, shape = ladder(bat.results.get("quick", False))
    shape = dict(shape)

    d0, err0 = bat.child("instr", shape, env_extra={"DEBUG": "0"},
                         timeout=min(2400.0, max(180.0, bat.left() * 0.55)))
    if not d0:
        return {"status": "UNAVAILABLE", "reason": "DEBUG=0 child did not return",
                "child_stderr_tail": err0[-1500:], "shape": shape_str(shape)}
    dec = _derive_decomposition(d0)
    warm = [w for w in d0.get("warm", []) if "error" not in w]
    best = min(warm, key=lambda w: w["wall_ms"]) if warm else None
    clean = [c for c in (d0.get("clean_warm") or []) if "error" not in c]
    best_clean = min(clean, key=lambda c: c["wall_ms"]) if clean else None
    inst_launches = best.get("launches") if best else None
    clean_launches = best_clean.get("launches") if best_clean else None
    perturbation = (inst_launches - clean_launches
                    if inst_launches is not None and clean_launches is not None else None)
    # the census copy count at THIS shape, for the like-for-like count comparison
    census_copy = None
    for c in bat.results["stages"].get("census", {}).get("shapes", []):
        if c.get("C") == shape["C"] and c.get("NT") == shape["T"] // shape["C"] and "copy" in c:
            census_copy = c["copy"]

    d2 = None
    err2 = None
    if bat.left() > 60.0:
        reps = 2
        d2, err2 = bat.child("debug2", dict(shape, reps=reps), env_extra={"DEBUG": "2"},
                             timeout=min(2400.0, max(180.0, bat.left() * 0.85)))
    else:
        err2 = "budget exhausted"

    st: dict = {"status": "OK", "shape": shape_str(shape), "device": dev,
                "decomposition": dec,
                "debug0": {
                    "instrument": "DEBUG=0, one Device[dev].synchronize() after realize()",
                    "cold_wall_ms": d0.get("cold", {}).get("wall_ms"),
                    "cold_launches": d0.get("cold", {}).get("launches"),
                    "warm_wall_ms": best["wall_ms"] if best else None,
                    "warm_launches_instrumented": inst_launches,
                    "warm_wall_ms_clean": best_clean["wall_ms"] if best_clean else None,
                    "warm_launches_clean": clean_launches,
                    "instrument_perturbation_launches": perturbation,
                    "instrument_perturbation_note":
                        "MEASURED, not assumed: the launch count of the SAME call with and "
                        "without the realize()/numpy() split. A non-zero value means the "
                        "probe changes the thing it measures, and the unperturbed count "
                        "is the one to compare against a census or a DEBUG=2 parse.",
                    "warm_iters": len(warm),
                    "clean_warm_iters": len(clean),
                    "sync_floor_ms": d0.get("sync_floor_ms", {}).get("median"),
                    "finite": d0.get("cold", {}).get("finite"),
                    "local_size_autotune": d0.get("local_size_autotune"),
                }}
    if d2 and d2.get("per_realize"):
        p = d2["per_realize"]
        st["debug2"] = {
            "instrument": "DEBUG=2 in a separate process. Per-kernel times are the "
                          "event-bracketed values from tinygrad/runtime/ops_cuda.py:26-35 "
                          "and include host launch latency, so this is an UPPER BOUND.",
            "realizes_parsed": d2.get("n_realizes"),
            "kernel_ms_per_realize": p.get("kernel_ms"),
            "copy_ms_per_realize": p.get("copy_ms"),
            "launch_lines_per_realize": p.get("launch_lines"),
            "launch_lines_minus_copy": p.get("launch_lines_minus_copy"),
            "kernel_count_delta_per_realize": p.get("kernel_count_delta"),
            "lines_without_a_time_per_realize": p.get("lines_without_a_time"),
            "wrapper_is_device_specific":
                "The host-blocking wrapper is a CUDA implementation detail "
                "(tinygrad/runtime/ops_cuda.py:26-35: cuEventCreate x2, cuEventRecord, "
                "cuEventSynchronize, cuEventElapsedTime, cuEventDestroy x2, with "
                "engine/realize.py:318 setting wait when DEBUG>=2). On METAL and CPU "
                "this stage's DEBUG=2 sum is NOT the T4's DEBUG=2 sum, and the ratio "
                "below is not the ratio the T4 would produce. Run it on CUDA.",
            "count_comparison": {
                "note": "compared like with like. GlobalCounters counts kernels AND "
                        "copies together; a DEBUG=2 parse that filters \" copy \" out "
                        "counts kernels only, which is what kda/perf_kda.py reports.",
                "debug2_lines_total_vs_debug0_clean_total":
                    (p.get("launch_lines", 0) + p.get("copy_lines", 0)) == clean_launches,
                "debug2_lines_total": (p.get("launch_lines", 0) + p.get("copy_lines", 0)),
                "debug0_clean_total": clean_launches,
                "debug2_kernel_lines_vs_debug0_minus_census_copies":
                    (p.get("launch_lines") == (clean_launches - census_copy)
                     if clean_launches is not None and census_copy is not None else None),
                "census_copies_used": census_copy,
            },
        }
        k = p.get("kernel_ms")
        w = best["wall_ms"] if best else None
        wc = best_clean["wall_ms"] if best_clean else None
        d2w = d2.get("warm_wall_ms")
        if k is not None and (w or wc):
            st["separation"] = {
                "debug2_kernel_sum_ms_upper_bound": k,
                "debug0_warm_wall_ms_one_sync_instrumented": w,
                "debug0_warm_wall_ms_one_sync_unprobed": wc,
                "ratio_debug2_sum_over_debug0_wall":
                    (k / wc if (k is not None and wc) else None),
                "debug2_run_wall_ms": d2w,
                "debug2_instrument_plus_host_ms":
                    (d2w - k) if (d2w is not None and k is not None) else None,
                "bounds": {
                    "true_device_time_is": "0 < true_device_time <= min(debug2_sum, "
                                           "debug0_wall). Both instruments over-report it "
                                           "or equal it; neither can under-report.",
                    "device_time_lower_bound_ms": 0.0,
                    "device_time_upper_bound_ms": min(x for x in (k, wc) if x is not None),
                    "note": "an interval, not a point. A point estimate here would be a "
                            "projection; the two instruments bracket it and ncu (if "
                            "present) is what would close it."},
                "reading": "debug2_sum >= true device time >= 0, and debug0_wall >= true "
                           "device time. If the two are close, the kernels are real work. "
                           "If debug2_sum is far above debug0_wall, the per-launch "
                           "cuEventSynchronize is the cost and the 215.9 ms of "
                           "perf_kda.log:32 is mostly instrumentation. If debug0_wall is "
                           "far above the kernels' share, the host is the bottleneck. "
                           "THIS IS A COMPARISON OF TWO MEASUREMENTS, not a correction "
                           "applied to either. And on a non-CUDA device the DEBUG=2 arm "
                           "has a different wrapper, so this is not the T4's answer.",
            }
    else:
        st["debug2"] = {"status": "UNAVAILABLE",
                        "reason": (d2 or {}).get("reason") or
                                  "DEBUG=2 child did not return a per-realize figure",
                        "segments_seen": (d2 or {}).get("segments_seen"),
                        "segments_closed": (d2 or {}).get("segments_closed"),
                        "raw_in": "logs/debug2-dec.raw",
                        "child_stderr_tail": (err2 or "")[-1500:]}
    st["cold_vs_prior"] = {
        "measured_cold_ms": d0.get("cold", {}).get("wall_ms"),
        "prior_t4_cold_ms": PRIOR["replay_cold_ms_t4"],
        "prior_metal_cold_literal_ms": 49500.0,
        "prior_literal_source": "kda/replay_kda.py:105 (hard-coded, never measured on the "
                                "batch whose log is being read)",
        "status": "MEASURED here, on this batch. The prior figures are NOT inputs."}
    return st


def stage_decomposition(bat: Battery) -> dict:
    """stage_instrument already produces the decomposition from its DEBUG=0
    child. This stage exists so the decomposition is addressable on its own and
    so it can be re-run without paying for DEBUG=2."""
    st = bat.results["stages"].get("instrument", {})
    dec = st.get("decomposition")
    if st.get("status") == "SKIPPED":
        return {"status": "SKIPPED", "reason": st.get("reason", "--only/--skip")}
    if not dec:
        return {"status": "UNAVAILABLE",
                "reason": f"stage instrument is {st.get('status')}"
                          f"{': ' + str(st.get('error'))[:120] if st.get('error') else ''} "
                          f"and produced no decomposition"}
    bat.write_text("decomposition.json", json.dumps(dec, indent=1, default=str))

    def cell(v):
        return f"{v:>12.3f}" if isinstance(v, (int, float)) else f"{'n/a':>12}"

    sh = dec.get("shape")
    sh = shape_str(sh) if isinstance(sh, dict) and "C" in sh else str(sh)
    lines = []
    for k, v in dec.items():
        if not (isinstance(v, dict) and "value" in v):
            continue
        extra = ""
        if "n_calls" in v:
            extra = f"   [{v.get('n_calls', 0)} call(s), {human(v.get('n_bytes', 0))}]"
        lines.append(f"{k:>24}  {cell(v.get('value'))} ms{extra}  "
                     f"{str(v.get('instrument', ''))[:70]}")
    chk = dec.get("check_arithmetic") or {}
    bat.write_text("decomposition.txt", "\n".join([
        f"KDA warm-call decomposition   device={dec.get('device')}   {sh}",
        f"basis: {dec.get('basis')}", ""] + lines + [
        "",
        f"  check: {cell(chk.get('graph_build_plus_realize_plus_drain_plus_copyout'))}"
        f" + {cell(chk.get('residual_ms'))} residual == {cell(chk.get('wall_ms'))} wall",
        f"  {chk.get('reading', '')}",
    ]))
    return {"status": "OK", **{k: (v.get("value") if isinstance(v, dict) else v)
                               for k, v in dec.items() if isinstance(v, dict) and "value" in v},
            "detail": "decomposition.json / decomposition.txt"}


# ===========================================================================
# 5. codegen capture
# ===========================================================================
def stage_codegen(bat: Battery) -> dict:
    """The emitted kernel sources.

    `PARALLEL=0` is mandatory, not an optimisation. `engine/realize.py:277`
    compiles in a multiprocessing worker pool whenever more than one kernel is
    outstanding, so an in-process `compile_cached` hook in the parent sees only
    the stragglers. That is why the notebook printed `captured 1 kernels from
    CUDA` against 12,082 launches. The count that comes back is DISTINCT
    PROGRAMS, not launches; the launch count is stage launch_count.
    """
    _, _, shape = ladder(bat.results.get("quick", False))
    data, err = bat.child("codegen", dict(shape), env_extra={"PARALLEL": "0", "DEBUG": "0"},
                          timeout=min(1800.0, max(120.0, bat.left() * 0.8)))
    if not data:
        return {"status": "UNAVAILABLE", "reason": "codegen child did not return",
                "child_stderr_tail": err[-1500:]}
    n = 0
    for i, src in enumerate(data.get("sources", [])):
        p = os.path.join(bat.out, "codegen", f"kernel{i:04d}.txt")
        with open(p, "w") as f:
            f.write(src)
        n += 1
    return {"status": "OK" if n else "UNAVAILABLE",
            "device": data.get("device"), "shape": shape_str(shape),
            "n_sources_written": n,
            "n_distinct_programs": data.get("n_distinct"),
            "n_launches": data.get("launches"),
            "parallel_env": data.get("parallel_env"),
            "matches_prior_distinct_programs": (
                (n == PRIOR["distinct_programs_bench_shape"])
                if (n and shape.get("C") == 64 and shape.get("T") == 512) else None),
            "prior_distinct_programs": PRIOR["distinct_programs_bench_shape"],
            "note": "DISTINCT PROGRAMS, deduplicated by tinygrad's program cache. "
                    "Captured with PARALLEL=0; the default worker pool compiles in "
                    "subprocesses and an in-process hook sees 1 out of thousands. "
                    "A source count far below the launch count is CORRECT (191 "
                    "programs against 12,229 launches at the bench shape) and is "
                    "not a partial capture.",
            "reasons_empty": None if n else "compiler.compile_cached absent on this backend"}


# ===========================================================================
# 6. divide_precision
# ===========================================================================
N_DIVIDE = 200_000


def _divide_inputs(regime: str, N: int, seed: int = 0):
    """A verbatim copy of `probes/divide_precision.py:53-63`. It has to be
    verbatim: the counts 54,270 and 30,748 are properties of these exact
    arrays, and a different generator gives different integers, which would make
    this stage look like a refutation when it is a re-roll. The cross-check
    below proves bit-equality against the real probe when it is importable."""
    rng = np.random.default_rng(seed)
    if regime == "near1":
        a = rng.random(N).astype(np.float32) * np.float32(0.02) + np.float32(0.99)
        return a, a.copy()
    if regime == "pm2":
        a = rng.random(N).astype(np.float32) * np.float32(4) - np.float32(2)
        b = rng.random(N).astype(np.float32) * np.float32(2) + np.float32(0.01)
        return a, b
    a = rng.random(N).astype(np.float32) * np.float32(2000) - np.float32(1000)
    return a, rng.random(N).astype(np.float32) + np.float32(0.5)


def stage_divide(bat: Battery) -> dict:
    from tinygrad import Device, Tensor  # noqa: PLC0415
    N = 20000 if bat.results.get("quick") else N_DIVIDE
    dev = Device.DEFAULT
    out: dict = {"status": "OK", "device": dev, "N": N,
                 "generator": "verbatim copy of probes/divide_precision.py:53-63",
                 "regimes": {}, "cross_check": {}}

    # ---- cross-check against the real probe, if it is here --------------
    # At the PROBE's own N, not this run's N. The probe's `inputs()` closes over
    # its module-level N, so calling it with a different N returns a different
    # length and the comparison would be meaningless. `--quick` shrinks the
    # device measurement; it must not shrink the generator check.
    probe = os.path.join(bat.results["code_root"], "probes", "divide_precision.py")
    if os.path.isfile(probe):
        out["cross_check"]["probe_path"] = probe
        out["cross_check"]["probe_sha256"] = sha256_file(probe)
        try:
            import importlib.util  # noqa: PLC0415
            spec = importlib.util.spec_from_file_location("_dp_probe", probe)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            probe_N = int(getattr(mod, "N", N_DIVIDE))
            out["cross_check"]["probe_N_default"] = probe_N
            ok = True
            for regime in ("pm2", "near1", "wide"):
                a_mine, b_mine = _divide_inputs(regime, probe_N)
                a_thr, b_thr = mod.inputs(regime)
                same = bool(a_thr.shape == a_mine.shape and b_thr.shape == b_mine.shape
                            and np.array_equal(a_mine, a_thr)
                            and np.array_equal(b_mine, b_thr))
                out["cross_check"][f"inputs_identical_{regime}"] = same
                ok = ok and same
            out["cross_check"]["inputs_identical_to_probe"] = ok
            # the two headline counts, recomputed here from the PROBE's arrays
            a, b = mod.inputs("pm2")
            ref = mod.correctly_rounded_div(a, b)
            rcp = (a * (np.float32(1.0) / b)).astype(np.float32)
            out["cross_check"]["recomputed_from_probe_arrays"] = {
                "N": int(a.size),
                "wrong_vs_correctly_rounded": int(((a * (np.float32(1.0) / b))
                                                   .astype(np.float32) != ref).sum()),
                "rcp_mul_wrong_vs_correctly_rounded": int((rcp != ref).sum()),
                "a_div_a_wrong": int(((a * (np.float32(1.0) / a))
                                      .astype(np.float32) != np.float32(1.0)).sum()),
                "note": "numpy arithmetic on the probe's OWN arrays. This is a "
                        "re-derivation of the counts in the T4 log, on this host, with "
                        "no device. It says what a correctly-rounded double rounding "
                        "gives; whether a backend matches it is what the per-backend "
                        "rows above measure."}
        except Exception as e:  # noqa: BLE001
            out["cross_check"]["error"] = f"{type(e).__name__}: {e}"
    else:
        out["cross_check"] = {"status": "UNAVAILABLE",
                              "reason": f"probes/divide_precision.py not under "
                                        f"{bat.results['code_root']}; the generator here "
                                        f"is a verbatim copy and is unverified against it"}

    def correctly_rounded_div(a, b):
        return (a.astype(np.float64) / b.astype(np.float64)).astype(np.float32)

    backends = [d for d in ("CPU", "PYTHON", "CL", dev) if d in getattr(Device, "_devices", [])]
    seen: set[str] = set()
    ordered: list[str] = []
    for b in backends:
        if b not in seen:
            seen.add(b)
            ordered.append(b)
    out["backends_attempted"] = ordered
    out["note_backends"] = ("Device._devices is a list of device NAMES tinygrad knows, "
                           "not the ones that answered. A backend in the list that is "
                           "not installed here raises and is recorded UNAVAILABLE per "
                           "regime -- it is never reported as 0% wrong.")

    for regime in ("pm2", "near1", "wide"):
        a, b = _divide_inputs(regime, N)
        ref = correctly_rounded_div(a, b)
        rcp_mul = (a * (np.float32(1.0) / b)).astype(np.float32)
        row: dict = {}
        # The a/a control is a DIFFERENT quantity from a/b and must never be
        # averaged with it (notes/12 §4.3: 15.37% is a/a, 27.13% is a/b). It is
        # a SEPARATE division, a over a, exactly as probes/divide_precision.py:81
        # does it -- not a/b compared against 1.0, which is a different array
        # with a different answer and would silently poison this field.
        aa_ref = correctly_rounded_div(a, a)
        aa_rcp = (a * (np.float32(1.0) / a)).astype(np.float32)
        for bd in ordered:
            try:
                got = (Tensor(a).to(bd) / Tensor(b).to(bd)).numpy()
                nb = int((got != ref).sum())
                row[bd] = {
                    "wrong_vs_correctly_rounded": nb,
                    "rate": nb / N,
                    "max_ulp_estimate": int(np.max(np.abs(
                        got.view(np.int32).astype(np.int64)
                        - ref.view(np.int32).astype(np.int64)))),
                    "equals_a_times_rcp_b_bitwise": bool((got == rcp_mul).all()),
                    "finite": bool(np.isfinite(got).all()),
                    "status": "MEASURED",
                }
                if regime == "pm2":
                    got_aa = (Tensor(a).to(bd) / Tensor(a).to(bd)).numpy()
                    row[bd]["a_div_a"] = {
                        "wrong": int((got_aa != np.float32(1.0)).sum()),
                        "rate": float((got_aa != np.float32(1.0)).sum()) / N,
                        "equals_a_times_rcp_a_bitwise": bool((got_aa == aa_rcp).all()),
                        "wrong_vs_correctly_rounded": int((got_aa != aa_ref).sum()),
                        "note": "a SEPARATE a/a division, not a/b. Compare against the "
                                "prior 30,748 (15.37%) only at equal N.",
                    }
            except Exception as e:  # noqa: BLE001
                row[bd] = {"status": "UNAVAILABLE", "error": f"{type(e).__name__}: {e}"}
        out["regimes"][regime] = {"N": N, "backends": row}

    # ---- the generated source, which is the mechanism, not a number ------
    try:
        d = Device[dev]
        cap: list[str] = []
        orig = getattr(d.compiler, "compile_cached", None)
        if orig is not None:
            def spy(src, *A, **K):
                cap.append(str(src))
                return orig(src, *A, **K)
            d.compiler.compile_cached = spy
            x = Tensor(np.arange(64, dtype=np.float32) + 1.0)
            y = Tensor(np.full(64, 7.0, dtype=np.float32))
            (x / y).realize()
            d.compiler.compile_cached = orig
            lines = [l.strip() for s in cap for l in s.split("\n")
                     if "data0" in l and "=" in l and "float" in l]
            out["generated_div_source"] = lines[:6]
            out["renderer_emits_a_times_recip"] = any("(1/" in l or "1.0f/" in l for l in lines)
            out["hook_note"] = ("an in-process compile_cached hook under the default "
                                "worker pool sees 1 program; this is a 1-kernel graph, "
                                "so it is not affected here")
        else:
            out["generated_div_source"] = "UNAVAILABLE: this backend has no compiler.compile_cached"
    except Exception as e:  # noqa: BLE001
        out["generated_div_source"] = f"UNAVAILABLE: {type(e).__name__}: {e}"

    pm2 = out["regimes"]["pm2"]["backends"]
    out["compare_to_prior_t4"] = {
        "prior": {k: PRIOR[k] for k in ("divide_cuda_pm2_wrong", "divide_cuda_pm2_rate",
                                        "divide_cuda_aa_wrong")},
        "prior_source": PRIOR_SOURCE["divide_cuda_pm2_wrong"],
        "this_run": {b: {"a_over_b_wrong": pm2[b].get("wrong_vs_correctly_rounded"),
                         "a_over_b_rate": pm2[b].get("rate"),
                         "a_over_a_wrong": (pm2[b].get("a_div_a") or {}).get("wrong")}
                     for b in ordered if isinstance(pm2.get(b), dict)},
        "note": "The prior CUDA row came from the batch whose provenance is in "
                "question (see the module docstring). If the rate reproduces, the "
                "mechanism reproduces; the prior row is not an input to anything here.",
        "N_caveat": f"the prior counts are at N={N_DIVIDE}; this run's rows are at "
                    f"N={N}. A rate is N-dependent, so a rate mismatch between two "
                    f"different N is not a disagreement about the mechanism. Compare "
                    f"the COUNTS at equal N, or compare against "
                    f"cross_check.recomputed_from_probe_arrays, which is at the "
                    f"probe's own N.",
    }
    bat.write_text("divide_precision.json", json.dumps(out, indent=1, default=str))
    return out


# ===========================================================================
# 7. ncu, opportunistically
# ===========================================================================
def stage_ncu(bat: Battery) -> dict:
    """`ncu --metrics gpu__time_duration.sum` is the only instrument here that
    excludes launch latency by construction. It is a SAMPLE, bounded by
    --launch-count, and is reported as a sample. Never as the full-call total."""
    try:
        p = subprocess.run(["ncu", "--version"], capture_output=True, text=True, timeout=30)
    except Exception as e:  # noqa: BLE001
        return {"status": "UNAVAILABLE", "reason": f"ncu not present: {type(e).__name__}: {e}"}
    if p.returncode != 0:
        return {"status": "UNAVAILABLE",
                "reason": f"ncu present but not runnable (rc={p.returncode}): "
                          f"{(p.stderr or p.stdout)[-300:]}"}
    _, dev_shapes, _ = ladder(bat.results.get("quick", False))
    shape = dict(dev_shapes[-1], tag="ncu")
    budget = min(1200.0, max(120.0, bat.left() * 0.8))
    cmd = ["ncu", "--metrics", "gpu__time_duration.sum", "--csv",
           "--launch-count", "200", sys.executable, os.path.abspath(__file__),
           "--_child", "ncu_one", "--child-out",
           os.path.join(bat.out, "logs", "ncu.child.json"),
           "--child-shape", json.dumps(shape), "--code-root", bat.results["code_root"]]
    bat.write_text("ncu.cmd", " ".join(cmd))
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=budget,
                           env=dict(os.environ, PARALLEL="0"))
    except subprocess.TimeoutExpired:
        return {"status": "UNAVAILABLE", "reason": f"ncu exceeded {budget:.0f}s; "
                f"the command is in ncu.cmd and can be re-run by hand"}
    out_csv = os.path.join(bat.out, "logs", "ncu.csv")
    bat.write_text("logs/ncu.stdout", (p.stdout or "") + "\n--- stderr ---\n" + (p.stderr or ""))
    with open(out_csv, "w") as f:
        f.write(p.stdout or "")
    if p.returncode != 0:
        return {"status": "UNAVAILABLE",
                "reason": f"ncu rc={p.returncode}: {(p.stderr or '')[-600:]}"}
    rows = []
    # ncu --csv emits a header, a units row, and one row per launch, with quoted
    # fields AND thousands separators inside them ("1,234"). A naive
    # line.split(",") cuts such a value in half and reports a number that is
    # wrong by three orders of magnitude while looking perfectly plausible, so
    # this uses the csv module. The unit either travels with the value
    # ("1,234ns") or sits in its own column ("nsecond"); when neither says, it
    # defaults to nanoseconds, ncu's unit for gpu__time_duration.sum, rather
    # than reporting a bare number in an unknown unit.
    UNITS = {"ns": 1e-3, "nsecond": 1e-3, "nsec": 1e-3,
             "us": 1.0, "usecond": 1.0, "usec": 1.0,
             "ms": 1e3, "msecond": 1e3, "msec": 1e3,
             "s": 1e6, "second": 1e6, "sec": 1e6}
    try:
        import csv as _csv  # noqa: PLC0415
        parsed = list(_csv.reader(io.StringIO(p.stdout or "")))
    except Exception as e:  # noqa: BLE001
        return {"status": "UNAVAILABLE", "reason": f"could not parse ncu CSV: {e}"}
    for fields in parsed:
        fields = [str(f).strip() for f in fields]
        if len(fields) < 2:
            continue
        kernel = next((f for f in fields if f.startswith(("E_", "r_", "c_", "k_"))), None)
        if kernel is None:
            continue
        line_unit = next((UNITS[f.lower()] for f in fields if f.lower() in UNITS), None)
        dur = None
        for f in reversed(fields):
            m = re.fullmatch(r"([-\d.eE+]+)\s*(ns|us|ms|s|nsec|usec|msec|sec)?", f.replace(",", ""))
            if not m:
                continue
            scale = UNITS[m.group(2)] if m.group(2) else (line_unit if line_unit else UNITS["ns"])
            dur = float(m.group(1)) * scale
            break
        if dur is not None:
            rows.append({"kernel": kernel, "dur_us": dur})
    if not rows:
        return {"status": "UNAVAILABLE",
                "reason": "ncu ran but produced no parsable durations (raw CSV in "
                          "logs/ncu.csv; the parse wants a field starting E_/r_/c_/k_ "
                          "and a numeric metric value, and defaults to nanoseconds)",
                "first_lines": (p.stdout or "").splitlines()[:6]}
    vals = sorted(r["dur_us"] for r in rows)
    return {"status": "OK", "n_sampled": len(rows),
            "launch_count_cap": 200,
            "sampled_total_ms": sum(vals) / 1000.0,
            "median_us": vals[len(vals) // 2], "min_us": vals[0], "max_us": vals[-1],
            "per_kernel_hist": {},
            "instrument": "ncu gpu__time_duration.sum, first 200 launches of ONE warm "
                          "kda_chunk. SAMPLE, not the full call. Excludes launch latency "
                          "by construction.",
            "caveat": "the first 200 launches of a 12,082-launch call are not a random "
                      "sample of it; treat as a per-kernel scale, not a total"}


# ===========================================================================
# 8. ratios, regenerated
# ===========================================================================
def stage_ratios(bat: Battery) -> dict:
    """Every ratio the measured quantities support, from ONE batch, with the
    instrument named for each input. The 3.78x figure is not reproduced here:
    the formula that produced it is recorded nowhere in the tree, so producing
    the number again would be invention. See the module docstring."""
    ins = bat.results["stages"].get("instrument", {})
    if ins.get("status") == "SKIPPED":
        return {"status": "SKIPPED", "reason": ins.get("reason", "--only/--skip")}
    if ins.get("status") != "OK":
        return {"status": "UNAVAILABLE",
                "reason": f"stage instrument is {ins.get('status')}"
                          f"{': ' + str(ins.get('error'))[:120] if ins.get('error') else ''}; "
                          f"there is nothing to form a ratio from and no ratio is reported "
                          f"in its place"}
    d0 = ins.get("debug0", {})
    d2 = ins.get("debug2", {})
    cold = d0.get("cold_wall_ms")
    warm = d0.get("warm_wall_ms")
    k = d2.get("kernel_ms_per_realize")
    rows: list[dict] = []

    def add(name, num, den, num_src, den_src, caveat=""):
        r: dict = {"ratio": name, "status": "RE-DERIVED"}
        if num is None or den in (None, 0):
            r.update({"status": "UNAVAILABLE", "reason": caveat or "an input was not measured"})
        else:
            r["value"] = num / den
        r.update({"numerator": num, "denominator": den,
                  "numerator_instrument": num_src, "denominator_instrument": den_src,
                  "caveat": caveat})
        rows.append(r)

    add("cold_over_warm", cold, warm,
        "MEASURED: first kda_chunk in a fresh process (includes compilation)",
        "MEASURED: best of 3 warm kda_chunk calls, one synchronize()",
        "cold includes NVRTC compilation; warm does not. This is the ratio the "
        "49.5 s / 96,837 ms constant was a component of.")
    add("warm_over_debug2_kernel_sum_upper_bound", warm, k,
        "MEASURED: DEBUG=0 warm wall, one synchronize()",
        "UPPER BOUND: DEBUG=2 event-bracketed per-kernel sum, which includes host "
        "launch latency (tinygrad/runtime/ops_cuda.py:26-35)",
        "A value near 1 means the kernels are real work. A value well above 1 "
        "means the DEBUG=2 instrument is inflating, and the ratio is then a "
        "statement about the instrument, not about the GPU.")
    if k is not None and warm:
        add("debug2_kernel_sum_over_debug0_wall", k, warm,
            "UPPER BOUND: DEBUG=2 event-bracketed sum", "MEASURED: DEBUG=0 warm wall",
            "The reciprocal of the row above; reported because the 215.880 ms of "
            "perf_kda.log:32 is the numerator of this one.")
    add("gpu_share_of_warm", k, warm,
        "UPPER BOUND: DEBUG=2 kernel sum", "MEASURED: DEBUG=0 warm wall",
        "An UPPER BOUND on the GPU share of the warm call, so the lower bound on "
        "the non-GPU share is 1 - this. The '1% GPU / 99% Python' split of "
        "perf_kda.log:40-41 is this ratio inverted, from a batch whose "
        "provenance is in question.")
    dec = ins.get("decomposition", {})
    gb = (dec.get("graph_build_ms") or {}).get("value")
    rw = (dec.get("rewrite_passes_ms") or {}).get("value")
    add("graph_build_share_of_warm", gb, warm,
        "MEASURED: build-only mode (Tensor.numpy -> no-op)", "MEASURED: warm wall",
        "This is the 'is it Python or is it the scheduler' split, and unlike the "
        "prior one it is from one process, one batch, one instrument set.")
    add("rewrite_passes_share_of_warm", rw, warm,
        "RE-DERIVED: (build+schedule_linear) - build", "MEASURED: warm wall",
        "rewrite passes = tinygrad re-running its rewrite chain on every realize(), "
        "which is what replay_kda.py calls 'the SCHEDULER'.")
    lau = d0.get("warm_launches")
    if k is not None and lau:
        add("debug2_ms_per_launch_upper_bound", k * 1000.0, float(lau),
            "UPPER BOUND: DEBUG=2 kernel sum, per call", "MEASURED: GlobalCounters launches",
            "17.868 us/launch is what notes/13 §7.1 recorded. It is an upper bound "
            "that contains the host launch and the event wrapper, not a per-kernel "
            "cost. The ncu stage, if it ran, is the instrument that would replace it.")

    # Per-shape launch ratios, where the hypotheses separate. Grouped BY C: a
    # T-span taken across different C values measures the C dependence, not the
    # T dependence, and would be reported as if it were the latter.
    shapes = [r for r in (bat.results["stages"].get("census", {}).get("shapes") or [])
              if "sink" in r]
    by_c: dict[int, list[dict]] = {}
    for r in shapes:
        by_c.setdefault(r["C"], []).append(r)
    for C in sorted(by_c):
        grp = sorted(by_c[C], key=lambda r: r["T"])
        if len(grp) < 2:
            continue
        lo, hi = grp[0], grp[-1]
        tspan = hi["T"] / lo["T"]
        cspan = hi["sink"] / lo["sink"] if lo["sink"] else None
        rows.append({
            "ratio": f"sink_count_ratio_over_T_span_at_C{C}",
            "status": "RE-DERIVED",
            "value": cspan,
            "C": C, "T_span": [lo["T"], hi["T"]], "NT_span": [lo["NT"], hi["NT"]],
            "sink_span": [lo["sink"], hi["sink"]],
            "T_moved_by": tspan,
            "reading": f"at C={C}, T moved {tspan:.0f}x (NT {lo['NT']}->{hi['NT']}) and "
                       f"the launch count moved {cspan:.4f}x ({lo['sink']:,} -> "
                       f"{hi['sink']:,}). A count proportional to T would have moved by "
                       f"{tspan:.0f}x. This is the measurement that separates a constant "
                       f"of C from a count proportional to the problem size, and it is "
                       f"why the sweep exists."})
    # The C span has to be taken at a FIXED NT, or it measures T as well. The NT
    # to use is the one present at the most distinct C values; the ladder says
    # which NTs it ran, and if none of them has 2+ C values this reports
    # UNAVAILABLE with the reason rather than quietly comparing across NT.
    nt_counts: dict[int, set] = {}
    for r in shapes:
        nt_counts.setdefault(r["NT"], set()).add(r["C"])
    fixed_nt = None
    for nt in sorted(nt_counts, reverse=True):
        if len(nt_counts[nt]) >= 2:
            fixed_nt = nt
            break
    if fixed_nt is not None:
        cgrp = sorted([(r["C"], r) for r in shapes if r["NT"] == fixed_nt])
        lo_c, lo_r = cgrp[0]
        hi_c, hi_r = cgrp[-1]
        rows.append({
            "ratio": f"sink_count_ratio_over_C_span_at_fixed_NT{fixed_nt}",
            "status": "RE-DERIVED",
            "value": (hi_r["sink"] / lo_r["sink"]) if lo_r["sink"] else None,
            "C_span": [lo_c, hi_c], "sink_span": [lo_r["sink"], hi_r["sink"]],
            "reading": f"at NT={fixed_nt}, moving C from {lo_c} to {hi_c} moved the "
                       f"count from {lo_r['sink']:,} to {hi_r['sink']:,}. C(C-1) is the "
                       f"unrolled forward substitution at kda/tiny_kda.py:178-194, so a "
                       f"count that is a function of C and not of T is what the "
                       f"schedule says. This is the other half of the separation, and it "
                       f"is the half that says 12,082 is a statement about C=64 and "
                       f"means nothing without it."})
    else:
        rows.append({
            "ratio": "sink_count_ratio_over_C_span_at_fixed_NT",
            "status": "UNAVAILABLE",
            "reason": f"no NT value was run at two or more C values, so a C span cannot "
                      f"be drawn. NT -> Cs seen: { {k: sorted(v) for k, v in nt_counts.items()} }"})
    out = {"status": "OK", "ratios": rows,
           "shares_do_not_sum_to_one":
               "graph_build_share_of_warm and rewrite_passes_share_of_warm are not two "
               "slices of the same pie. graph_build is Python UOp construction; "
               "rewrite_passes is INSIDE realize_and_issue, which also contains lower, "
               "compile and launch issue. Adding them double counts. The slices that do "
               "partition the warm call are in the decomposition's check_arithmetic.",
           "prior_not_reproduced": {
               "3.78x": "NOT REPRODUCED, BY DESIGN. It was derived from a 0.109 s GPU "
                        "baseline (hard-coded at kda/replay_kda.py:97,101-102) and a "
                        "49.5 s cold time (hard-coded at :105 against a measured "
                        "96,837 ms in the same log), and the formula that produced "
                        "3.78x from them is recorded nowhere in the tree. The inputs are "
                        "listed here so the mistake is not repeated; the ratio has to "
                        "come from a fresh run of this file, and it has to name its "
                        "instruments.",
               "1% GPU / 99% Python": "Regenerated above as gpu_share_of_warm, with the "
                                      "numerator labelled an UPPER BOUND."}}
    bat.write_text("ratios.json", json.dumps(out, indent=1, default=str))
    return out


# ===========================================================================
# 9. summary
# ===========================================================================
def fmt(v, nd=3, width=13):
    if v is None:
        return "n/a".rjust(width)
    if isinstance(v, float):
        return f"{v:,.{nd}f}".rjust(width)
    if isinstance(v, int):
        return f"{v:,}".rjust(width)
    return str(v)[:width].rjust(width)


def print_summary(bat: Battery) -> None:
    R = bat.results
    man = R.get("manifest", {})
    env = R.get("stages", {}).get("environment", {})
    # the RICH decomposition (value + instrument per component) lives on the
    # instrument stage; the `decomposition` stage is the flattened, addressable
    # copy of it, plus a pointer to decomposition.json
    ins_all = R.get("stages", {}).get("instrument", {})
    dec = ins_all.get("decomposition") or {}
    W = 78
    ins_skipped = ins_all.get("status") == "SKIPPED"
    print()
    print("=" * W)
    print(f"KDA battery   device={env.get('device', 'UNAVAILABLE')}   "
          f"out={bat.out}")
    print("=" * W)
    print(f"  manifest_id      {man.get('manifest_id', 'UNAVAILABLE')}")
    print(f"  code_root        {man.get('code_root', 'UNAVAILABLE')}")
    tar = man.get("payload_tarball")
    print(f"  payload          {('sha256 ' + tar['sha256'][:16] + '  ' + human(tar['bytes'])) if tar else 'not present on this box'}"
          f"   sidecar_match={man.get('payload_tarball_matches_sidecar')}")
    print(f"  files hashed     {man.get('n_files', 'UNAVAILABLE')}"
          f"   {human(man['total_bytes']) if man.get('total_bytes') else ''}")
    sf = (env.get("sync_floor") or {}).get("median_ms")
    print(f"  idle sync() cost {fmt(sf, 4)} ms   (the floor every drain number sits on)")
    print(f"  session          {json.dumps(man.get('session'))[:60]}")
    imp = R.get("imported_modules") or {}
    cut = imp.get("code_under_test") or {}
    for mod, rec in cut.items():
        tag = "OK" if rec.get("in_code_root") else "MISMATCH"
        print(f"  {mod:<16} {tag:<9} {str(rec.get('file'))[-46:]}  "
              f"sha256 {str(rec.get('sha256'))[:12]}")
        if not rec.get("in_code_root"):
            print(f"      !! the code under test was imported from OUTSIDE the manifested "
                  f"tree. Every number in this run is attributed to bytes the manifest "
                  f"does not describe.")

    cen = R["stages"].get("census", {})
    print()
    print(f"  census           dev={cen.get('census_dev', 'UNAVAILABLE')}  "
          f"{cen.get('n_shapes_run', 0)}/{cen.get('n_shapes_total', 0)} shapes  "
          f"[{cen.get('status')}]")
    for s in (cen.get("shapes") or [])[:8]:
        if "sink" in s:
            extra = ""
            if "matches_prior_cuda_bench" in s:
                extra = f"   reproduces perf_kda.log:31 = {s['matches_prior_cuda_bench']}"
            print(f"      {s['shape']:<48} sink={s['sink']:>7,} copy={s['copy']:>4,}{extra}")
    m = cen.get("models") or {}
    if m.get("models"):
        print(f"      fit: {m.get('verdict')}")
        for name, f in m["models"].items():
            print(f"           {name:<12} {json.dumps(f['coef'])}  max|resid| "
                  f"{f['max_abs']:,.1f} ({f['max_rel']*100:.3f}%)")

    lc = R["stages"].get("launch_count", {})
    print()
    print(f"  device launches  device={lc.get('device', 'UNAVAILABLE')}  [{lc.get('status')}]")
    for s in (lc.get("shapes") or []):
        if "warm_launches" in s:
            x = ""
            cs = s.get("census_same_shape")
            if cs:
                x = (f"   census[{cs['census_dev']}] sink={cs['sink']:,} "
                     f"delta={fmt(s.get('device_minus_census'), 0)}"
                     f"{'' if cs['same_backend'] else '  (other backend)'}")
            print(f"      {s['shape']:<48} warm={s['warm_launches']:>7,}{x}")
    for s in (lc.get("shapes") or []):
        if s.get("C") == 64 and s.get("T") == 512 and s.get("device") == "METAL":
            tot = s.get("warm_launches")
            if tot is not None:
                print(f"      bench shape on METAL total {tot:,}  "
                      f"reproduces 12,090+139 = "
                      f"{PRIOR['launch_count_metal_bench_total']:,}: "
                      f"{tot == PRIOR['launch_count_metal_bench_total']}")

    ins = R["stages"].get("instrument", {})
    d0, d2 = ins.get("debug0", {}), ins.get("debug2", {})
    print()
    if ins_skipped:
        print(f"  instrument       SKIPPED  {ins.get('reason', '')}")
    else:
        print(f"  instrument       {ins.get('shape', 'UNAVAILABLE')}  [{ins.get('status')}]")
    if not ins_skipped:
        print(f"      DEBUG=0 cold   wall {fmt(d0.get('cold_wall_ms'), 1)} ms   "
              f"launches {fmt(d0.get('cold_launches'), 0)}   (includes compile)")
        print(f"      DEBUG=0 warm   wall {fmt(d0.get('warm_wall_ms'), 1)} ms   "
              f"launches {fmt(d0.get('warm_launches_instrumented'), 0)}   (one synchronize)")
        print(f"      same, unprobed wall {fmt(d0.get('warm_wall_ms_clean'), 1)} ms   "
              f"launches {fmt(d0.get('warm_launches_clean'), 0)}   "
              f"probe costs {fmt(d0.get('instrument_perturbation_launches'), 0)} launches")
        print(f"      DEBUG=2 kernel {fmt(d2.get('kernel_ms_per_realize'), 3)} ms   "
              f"UPPER BOUND, /{d2.get('realizes_parsed', '?')} realize(s)")
        sep = ins.get("separation") or {}
        if sep:
            b = sep.get("bounds") or {}
            print(f"      ratio debug2/debug0 {fmt(sep.get('ratio_debug2_sum_over_debug0_wall'), 4)}   "
                  f"instrument+host {fmt(sep.get('debug2_instrument_plus_host_ms'), 1)} ms")
            print(f"      true device time is in [0, {fmt(b.get('device_time_upper_bound_ms'), 1)}] ms"
                  f"   (an interval, not a point)")
        if d2.get("count_comparison"):
            cc = d2["count_comparison"]
            print(f"      DEBUG=2 lines {fmt(cc.get('debug2_lines_total'), 0)} vs DEBUG=0 clean "
                  f"count {fmt(cc.get('debug0_clean_total'), 0)}  -> equal: "
                  f"{cc.get('debug2_lines_total_vs_debug0_clean_total')}")
            if cc.get("debug2_kernel_lines_vs_debug0_minus_census_copies") is not None:
                print(f"      DEBUG=2 kernel lines == DEBUG=0 count - census copies: "
                      f"{cc['debug2_kernel_lines_vs_debug0_minus_census_copies']}")

    print()
    if ins_skipped:
        print(f"  decomposition    SKIPPED  {ins_all.get('reason', '')}")
    else:
        print(f"  decomposition    [{dec.get('status')}]"
              + ("" if dec else f"  (instrument is {ins_all.get('status')})"))
    if dec:
        for key, label in (("graph_build_ms", "graph construction"),
                           ("rewrite_passes_ms", "  ^ inside realize+issue"),
                           ("realize_and_issue_ms", "lower+compile+issue"),
                           ("device_drain_ms", "device drain (post-sync)"),
                           ("copyout_ms", "copy-out D2H"),
                           ("h2d_host_ms", "copy-in H2D (enqueue)"),
                           ("unattributed_ms", "unattributed residual")):
            v = (dec.get(key) or {}).get("value")
            print(f"      {label:<26} {fmt(v, 3)} ms")
        chk = dec.get("check_arithmetic") or {}
        if chk.get("wall_ms") is not None:
            print(f"      {'check: sum + residual':<26} {fmt(chk.get('wall_ms'), 3)} ms  (wall)")
        for key in ("kernel_ms", "lower_and_compile_ms"):
            v = dec.get(key)
            if v and v.get("value") is None:
                print(f"      {key:<26} {'n/a':>13}      {v.get('instrument', '')[:44]}")

    cg = R["stages"].get("codegen", {})
    dv = R["stages"].get("divide", {})
    nc = R["stages"].get("ncu", {})
    rt = R["stages"].get("ratios", {})
    print()
    if cg.get("status") == "SKIPPED":
        print(f"  codegen          SKIPPED  {cg.get('reason', '')}")
    else:
        print(f"  codegen          {cg.get('n_sources_written', 0)} sources "
              f"(distinct programs) / {fmt(cg.get('n_launches'), 0)} launches  "
              f"[{cg.get('status')}]")
        if cg.get("matches_prior_distinct_programs") is not None:
            print(f"      reproduces the {cg.get('prior_distinct_programs')} distinct "
                  f"programs of notes/13 §4: {cg['matches_prior_distinct_programs']}  "
                  f"(PARALLEL={cg.get('parallel_env')})")
    if dv.get("status") == "SKIPPED":
        print(f"  divide           SKIPPED  {dv.get('reason', '')}")
    else:
        pm2 = ((dv.get("regimes") or {}).get("pm2") or {}).get("backends", {})
        cells = "  ".join(f"{b} {pm2[b].get('wrong_vs_correctly_rounded', 'n/a')}"
                          f"({(pm2[b].get('rate') or 0)*100:.2f}%)"
                          f" a/a {(pm2[b].get('a_div_a') or {}).get('wrong', 'n/a')}"
                          for b in (dv.get("backends_attempted") or [])
                          if isinstance(pm2.get(b), dict))
        print(f"  divide pm2 N={dv.get('N')}   {cells}  [{dv.get('status')}]")
        rc = ((dv.get("cross_check") or {}).get("recomputed_from_probe_arrays") or {})
        if rc:
            print(f"      re-derived from probes/divide_precision.py's OWN arrays: "
                  f"a/b wrong {rc.get('wrong_vs_correctly_rounded'):,}   "
                  f"a/a wrong {rc.get('a_div_a_wrong'):,}   (prior CUDA: 54,270 / 30,748)")
        print(f"      inputs identical to probes/divide_precision.py: "
              f"{(dv.get('cross_check') or {}).get('inputs_identical_to_probe', 'UNAVAILABLE')}")
    if nc.get("status") == "SKIPPED":
        print(f"  ncu              SKIPPED  {nc.get('reason', '')}")
    elif nc.get("status") == "OK":
        print(f"  ncu sample       {nc['n_sampled']} launches  median {nc['median_us']:.2f} us  "
              f"sum {nc['sampled_total_ms']:.2f} ms  [SAMPLE, not the full call]")
    else:
        print(f"  ncu              UNAVAILABLE  {str(nc.get('reason', ''))[:52]}")
    print()
    if rt.get("status") == "SKIPPED":
        print(f"  ratios           SKIPPED  {rt.get('reason', '')}")
    else:
        print(f"  ratios           [{rt.get('status')}]")
        for r in (rt.get("ratios") or []):
            if r.get("status") == "RE-DERIVED" and r.get("value") is not None:
                print(f"      {r['ratio']:<44} {r['value']:>14,.4f}")
            else:
                print(f"      {r['ratio']:<44} {'n/a':>14}   {str(r.get('reason', ''))[:40]}")
    print()
    bad = [(n, s.get("status"), s.get("error", ""))
           for n, s in R["stages"].items() if s.get("status") not in ("OK", "SKIPPED")]
    print("  stage status     " + "  ".join(
        f"{n}={s.get('status', '?')}" for n, s in R["stages"].items()))
    for n, s, e in bad:
        if e:
            print(f"      {n}: {str(e)[:70]}")
    print(f"  wall             {bat.spent():,.1f}s of {bat.budget_s:,.0f}s budget")
    print(f"  artefacts        {bat.out}")
    print("=" * W)


# ===========================================================================
# 10. children
# ===========================================================================
def child_census(args, shape, out_path) -> None:
    """Device-free launch count. `Context(DEV=...)` makes `Device.DEFAULT` read
    as that string, so the schedule is the one the named backend would get,
    without a device existing (engine/realize.py:317 -> compile_linear ->
    memory_plan_rewrite, which is a pure UOp rewrite)."""
    import json  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from tinygrad import Tensor  # noqa: PLC0415
    from tinygrad.helpers import Context  # noqa: PLC0415
    from tinygrad.uop.ops import Ops  # noqa: PLC0415
    import tinygrad.schedule as sch  # noqa: PLC0415
    for p in (args.code_root, os.path.join(args.code_root, "kda")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import tiny_kda, test_ref as T  # noqa: PLC0415
    dev = os.environ.get("KDA_CENSUS_DEV", "CUDA")
    n = {"sink": 0, "copy": 0}
    orig = sch.memory_plan_rewrite

    def spy(linear, held_bufs=None):
        for si in linear.src:
            if si.src[0].op is Ops.COPY:
                n["copy"] += 1
            else:
                n["sink"] += 1
        return orig(linear, held_bufs)

    sch.memory_plan_rewrite = spy
    real_numpy = Tensor.numpy
    Tensor.numpy = lambda self, *a, **k: (self.schedule_linear(),
                                          np.zeros((), dtype=np.float32))[1]
    try:
        with Context(DEV=dev):
            from tinygrad import Device  # noqa: PLC0415
            assert Device.DEFAULT == dev, f"{Device.DEFAULT!r} != {dev!r}"
            q, k, v, g, beta = T.rnd(shape["B"], shape["T"], shape["H"], shape["HV"],
                                      shape["K"], shape["V"], seed=0, l2n=True)
            tiny_kda.kda_chunk(q, k, v, g, beta, chunk_size=shape["C"])
    finally:
        sch.memory_plan_rewrite = orig
        Tensor.numpy = real_numpy
    json.dump({"dev": dev, "shape": shape_str(shape), "sink": n["sink"],
               "copy": n["copy"]}, open(out_path, "w"), indent=1)


def child_launch(args, shape, out_path) -> None:
    """The launch count, measured by the device. `GlobalCounters.kernel_count`
    is tinygrad's own counter, incremented in `track_stats` once per CALL node
    in the linear schedule (engine/realize.py:60-63), so it counts launches
    with no regex and no stdout parsing. Copies are included, so the census's
    sink count has the copy count subtracted before they are compared."""
    import json  # noqa: PLC0415
    from tinygrad import Device  # noqa: PLC0415
    from tinygrad.helpers import GlobalCounters  # noqa: PLC0415
    for p in (args.code_root, os.path.join(args.code_root, "kda")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import tiny_kda, test_ref as T  # noqa: PLC0415
    C = shape["C"]
    q, k, v, g, beta = T.rnd(shape["B"], shape["T"], shape["H"], shape["HV"],
                              shape["K"], shape["V"], seed=0, l2n=True)
    out = {"device": Device.DEFAULT, "shape": shape_str(shape)}
    GlobalCounters.reset()
    tiny_kda.kda_chunk(q, k, v, g, beta, chunk_size=C)
    out["cold_launches"] = GlobalCounters.kernel_count
    ws = []
    for _ in range(3):
        GlobalCounters.reset()
        tiny_kda.kda_chunk(q, k, v, g, beta, chunk_size=C)
        ws.append(GlobalCounters.kernel_count)
    out["warm_launches"] = min(ws) if ws else None
    out["warm_launches_all"] = ws
    out["cold_equals_warm"] = (out["cold_launches"] == out["warm_launches"])
    json.dump(out, open(out_path, "w"), indent=1)


def child_debug2(args, shape, out_path) -> None:
    """DEBUG=2, in its own process, because DEBUG is read at import time.
    Runs 1 cold + `reps` warm calls and divides the per-kernel sum by the
    number of warm calls ACTUALLY parsed - the gotcha `kda/perf_kda.py` gets
    right only by assuming the cold and warm calls launch the same number of
    times."""
    import json  # noqa: PLC0415
    from tinygrad import Device  # noqa: PLC0415
    from tinygrad.helpers import GlobalCounters  # noqa: PLC0415
    for p in (args.code_root, os.path.join(args.code_root, "kda")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import tiny_kda, test_ref as T  # noqa: PLC0415
    C = shape["C"]
    reps = int(shape.get("reps", 2))
    q, k, v, g, beta = T.rnd(shape["B"], shape["T"], shape["H"], shape["HV"],
                              shape["K"], shape["V"], seed=0, l2n=True)
    # The markers MUST bracket each call, in the same stream as the DEBUG=2 `***`
    # lines, or the parse sees all the kernel lines in one segment. The payload
    # (wall, launch count) is only known after the call, so each segment gets a
    # BEGIN line before and an END line with the numbers after.
    print("###KDA-COLD-BEGIN###", flush=True)
    GlobalCounters.reset()
    t0 = time.perf_counter()
    tiny_kda.kda_chunk(q, k, v, g, beta, chunk_size=C)
    print(f"###KDA-COLD-END### {(time.perf_counter()-t0)*1e3:.4f} "
          f"{GlobalCounters.kernel_count}", flush=True)
    for i in range(reps):
        print(f"###KDA-WARM-{i}-BEGIN###", flush=True)
        GlobalCounters.reset()
        t0 = time.perf_counter()
        tiny_kda.kda_chunk(q, k, v, g, beta, chunk_size=C)
        print(f"###KDA-WARM-{i}-END### {(time.perf_counter()-t0)*1e3:.4f} "
              f"{GlobalCounters.kernel_count}", flush=True)


def parse_debug2(text: str) -> dict:
    """Parse a DEBUG=2 log into per-realize figures. Divides by the number of
    warm realizes actually present, and reports that number."""
    txt = ANSI.sub("", text or "")
    lines = txt.splitlines()
    segs: list[dict] = []
    cur = None
    # The END marker carries its payload AFTER the marker, so `endswith` on the
    # marker alone is wrong and silently closes zero segments. Match the marker
    # with a regex that ignores whatever follows it.
    m_begin = re.compile(r"^###KDA-(COLD|WARM-\d+)-BEGIN###")
    m_end = re.compile(r"^###KDA-(COLD|WARM-\d+)-END###\s+([-\d.]+)\s+(\d+)")
    for i, l in enumerate(lines):
        mb = m_begin.match(l)
        me = m_end.match(l)
        if mb:
            cur = {"kind": "cold" if mb.group(1) == "COLD" else "warm",
                   "begin": i, "label": mb.group(1)}
            segs.append(cur)
        elif me and cur is not None:
            cur["end"] = i
            cur["wall_ms"] = float(me.group(2))
            cur["kernel_count_delta"] = int(me.group(3))
            cur = None
    out: dict = {"segments_seen": len(segs),
                 "segments_closed": sum(1 for s in segs if "end" in s)}
    warm: list[dict] = []
    cold = None
    for s in segs:
        if "end" not in s:
            continue
        seg = lines[s["begin"] + 1:s["end"]]
        ker_us = cp_us = 0.0
        nk = nc = nlines = 0
        for l in seg:
            if not l.startswith("***"):
                continue
            nlines += 1
            m = re.search(r"\b([\d.]+)(us|ms)/", l)
            if not m:
                continue
            v = float(m.group(1)) * (1000.0 if m.group(2) == "ms" else 1.0)
            if " copy " in l:
                cp_us += v
                nc += 1
            else:
                ker_us += v
                nk += 1
        rec = {"kernel_ms": ker_us / 1000.0, "copy_ms": cp_us / 1000.0,
               "launch_lines": nk, "copy_lines": nc, "total_lines": nlines,
               "lines_without_a_time": nlines - nk - nc,
               "wall_ms": s.get("wall_ms"),
               "kernel_count_delta": s.get("kernel_count_delta")}
        if s["kind"] == "cold":
            cold = rec
        else:
            warm.append(rec)
    out["cold"] = cold
    if not warm:
        out["per_realize"] = None
        out["n_realizes"] = 0
        out["reason"] = ("no closed warm segment in the DEBUG=2 log. The markers are "
                         "emitted by this file's --_child debug2_run; if the run died, "
                         "or if DEBUG=2 output went somewhere other than stdout, there "
                         "is nothing to segment.")
        return out
    n = len(warm)
    out["n_realizes"] = n
    out["warm_wall_ms"] = min(w["wall_ms"] for w in warm)
    out["per_realize"] = {
        "kernel_ms": sum(w["kernel_ms"] for w in warm) / n,
        "copy_ms": sum(w["copy_ms"] for w in warm) / n,
        "launch_lines": sum(w["launch_lines"] for w in warm) / n,
        "copy_lines": sum(w["copy_lines"] for w in warm) / n,
        "total_lines": sum(w["total_lines"] for w in warm) / n,
        "lines_without_a_time": sum(w["lines_without_a_time"] for w in warm) / n,
        "kernel_count_delta": [w["kernel_count_delta"] for w in warm],
        "divided_by": n,
        "per_realize_detail": warm,
    }
    out["per_realize"]["launch_lines_minus_copy"] = \
        out["per_realize"]["launch_lines"] - out["per_realize"]["copy_lines"]
    return out


def child_codegen(args, shape, out_path) -> None:
    import json  # noqa: PLC0415
    from tinygrad import Device  # noqa: PLC0415
    from tinygrad.helpers import GlobalCounters  # noqa: PLC0415
    for p in (args.code_root, os.path.join(args.code_root, "kda")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import tiny_kda, test_ref as T  # noqa: PLC0415
    d = Device[Device.DEFAULT]
    cap: list[str] = []
    orig = getattr(d.compiler, "compile_cached", None)
    if orig is not None:
        def spy(src, *a, **k):
            cap.append(str(src))
            return orig(src, *a, **k)
        d.compiler.compile_cached = spy
    q, k, v, g, beta = T.rnd(shape["B"], shape["T"], shape["H"], shape["HV"],
                              shape["K"], shape["V"], seed=0, l2n=True)
    GlobalCounters.reset()
    tiny_kda.kda_chunk(q, k, v, g, beta, chunk_size=shape["C"])
    if orig is not None:
        d.compiler.compile_cached = orig
    json.dump({"device": Device.DEFAULT, "n_distinct": len(cap), "sources": cap,
               "launches": GlobalCounters.kernel_count,
               "parallel_env": os.environ.get("PARALLEL")},
              open(out_path, "w"))


def child_ncu_one(args, shape, out_path) -> None:
    """One kda_chunk, for ncu to sample. Nothing else: any extra work in this
    process would be sampled too."""
    for p in (args.code_root, os.path.join(args.code_root, "kda")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import tiny_kda, test_ref as T  # noqa: PLC0415
    q, k, v, g, beta = T.rnd(shape["B"], shape["T"], shape["H"], shape["HV"],
                              shape["K"], shape["V"], seed=0, l2n=True)
    tiny_kda.kda_chunk(q, k, v, g, beta, chunk_size=shape["C"])


# ===========================================================================
# 11. main
# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(
        description="A single-file GPU battery for the KDA path. See the module "
                    "docstring for what each stage measures and why.")
    ap.add_argument("--code-root", default=None,
                    help="directory containing kda/tiny_kda.py. Default: search "
                         "this file's parent, the cwd, and /kaggle/input/*/probe.")
    ap.add_argument("--out", default=None,
                    help="output directory. Default: /kaggle/working/run_kda on "
                         "Kaggle, else out/kaggle/<stamp>-kda, else ./run_kda-out")
    ap.add_argument("--budget", type=float, default=2700.0,
                    help="total wall budget in seconds (default 2700 = 45 min). "
                         "Stages are skipped, and say so, once it runs out. Keep it "
                         "BELOW the Kaggle session cap (remote/kaggle_run.sh uses 75 "
                         "min): a session killed mid-battery never writes the summary. "
                         "results.json is rewritten after every stage regardless, so a "
                         "truncated run is still readable.")
    ap.add_argument("--quick", action="store_true",
                    help="small shapes and N=20,000 for the divide probe, so the "
                         "whole battery runs in a couple of minutes. Every stage "
                         "still runs; the shapes are recorded.")
    ap.add_argument("--only", default="", help="comma-separated stage names to run")
    ap.add_argument("--skip", default="", help="comma-separated stage names to skip")
    ap.add_argument("--list", action="store_true", help="list the stages and exit")
    ap.add_argument("--_child", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--child-out", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--child-shape", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.list:
        print("\n".join(STAGES))
        return 0

    # ---- child path: cheap, no manifest, no budget, no summary -----------
    if args._child:
        shape = json.loads(args.child_shape or "{}")
        # CHILD_INSTR is exec'd so the instrumented pass shares this file's
        # globals (np, Tensor, kda_chunk) instead of re-quoting them.
        if args._child == "instr":
            g = {"os": os, "sys": sys, "json": json, "time": time}
            import numpy as np  # noqa: PLC0415
            g["np"] = np
            exec(compile(CHILD_INSTR, "<child_instr>", "exec"), g)
            g["child_instr"](args, shape, args.child_out)
        elif args._child == "census":
            import numpy as np  # noqa: PLC0415
            child_census(args, shape, args.child_out)
        elif args._child == "launch":
            child_launch(args, shape, args.child_out)
        elif args._child == "codegen":
            child_codegen(args, shape, args.child_out)
        elif args._child == "ncu_one":
            child_ncu_one(args, shape, args.child_out)
        elif args._child == "debug2":
            # this one parses ITS OWN stdout, which is why the markers exist
            p = subprocess.run([sys.executable, os.path.abspath(__file__),
                                "--_child", "debug2_run", "--child-shape", args.child_shape,
                                "--code-root", args.code_root],
                               capture_output=True, text=True,
                               env=dict(os.environ, DEBUG="2"))
            with open(os.path.join(os.path.dirname(args.child_out), "debug2.raw"), "w") as f:
                f.write((p.stdout or "") + "\n--- stderr ---\n" + (p.stderr or ""))
            data = parse_debug2(p.stdout or "")
            data["device"] = None
            data["raw_segments_found"] = (p.stdout or "").count("###KDA-")
            data["raw_kernel_lines_found"] = sum(
                1 for l in (p.stdout or "").splitlines() if l.lstrip("\x1b[0;9;m").startswith("***"))
            json.dump(data, open(args.child_out, "w"), indent=1, default=str)
        elif args._child == "debug2_run":
            child_debug2(args, shape, args.child_out)
        else:
            raise SystemExit(f"unknown --_child {args._child}")
        return 0

    # ---- parent ----------------------------------------------------------
    root = discover_root(args.code_root)
    if root is None:
        print("FATAL: no code root found. Pass --code-root <dir> where "
              "<dir>/kda/tiny_kda.py exists.", file=sys.stderr)
        print("       tried: " + ", ".join([HERE, os.getcwd(),
                                           "/kaggle/input/*/probe"]), file=sys.stderr)
        return 2

    out = args.out
    if out is None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        # Kaggle only downloads what is under /kaggle/working, so that comes
        # first. Then the repo's own out/kaggle/<stamp>-kda, matching
        # remote/kaggle_run.sh. Both are outside the manifested tree (`out` is
        # in SKIP_DIRS), so writing results can never perturb the manifest.
        for cand in (f"/kaggle/working/run_kda-{stamp}",
                     os.path.join(root, "out", "kaggle", f"{stamp}-kda"),
                     os.path.join(os.getcwd(), f"run_kda-{stamp}")):
            try:
                os.makedirs(cand, exist_ok=True)
                probe = os.path.join(cand, ".w")
                open(probe, "w").close()
                os.unlink(probe)
                out = cand
                break
            except OSError:
                continue
    if out is None:
        out = os.path.join(os.getcwd(), "run_kda-out")
    bat = Battery(out, args.budget)
    bat.results["code_root"] = root
    bat.results["quick"] = bool(args.quick)
    bat.log(f"code_root {root}")
    bat.log(f"out       {out}   budget {args.budget:.0f}s")

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    bat.run_stage("manifest", lambda: stage_manifest(bat))
    man = bat.results.get("manifest", {})

    def guard(stage):
        """No stage runs on a code tree that changed under it, and a failed
        manifest is a loud stop rather than a silently mislabelled number."""
        ok, changed = rehash_manifest(man) if man else (False, ["no manifest"])
        if not ok:
            raise RuntimeError(
                f"the files changed under this battery since the manifest was taken: "
                f"{', '.join(changed)}. Every number this battery would print would be "
                f"attributed to the wrong version.")
        return {"manifest_stable": True, "manifest_id": man.get("manifest_id")}

    runners = {
        "environment": stage_environment,
        "census": stage_census,
        "launch_count": stage_launch_count,
        "instrument": stage_instrument,
        "decomposition": stage_decomposition,
        "codegen": stage_codegen,
        "divide": stage_divide,
        "ncu": stage_ncu,
        "ratios": stage_ratios,
    }
    try:
        for name in STAGES:
            if name in skip:
                bat.results["stages"][name] = {"status": "SKIPPED", "reason": "--skip"}
                bat.flush()
                continue
            if only and name not in only:
                bat.results["stages"][name] = {"status": "SKIPPED", "reason": "--only"}
                bat.flush()
                continue

            def wrapped(name=name):
                guard(name)
                return runners[name](bat)

            bat.run_stage(name, wrapped)
    finally:
        bat.results["imported_modules"] = imported_under_root(root)
        bat.flush()
        try:
            print_summary(bat)
            bat.write_text("SUMMARY.md", _summary_md(bat))
        except Exception as e:  # noqa: BLE001
            print(f"summary failed: {type(e).__name__}: {e}", file=sys.stderr)
        bat.flush()
    bad = [n for n, s in bat.results["stages"].items() if s.get("status") == "FAILED"]
    return 1 if bad else 0


def _summary_md(bat: Battery) -> str:
    """The one-screen summary, as a file. Same content as the printed one."""
    R = bat.results
    man, env = R.get("manifest", {}), R.get("stages", {}).get("environment", {})
    rows = ["| stage | status | seconds | note |", "|---|---|---|---|"]
    for n, s in R["stages"].items():
        note = s.get("error") or s.get("reason") or ""
        if n == "census":
            note = f"{s.get('n_shapes_run', 0)}/{s.get('n_shapes_total', 0)} shapes, dev={s.get('census_dev')}"
        if n == "launch_count":
            note = f"device={s.get('device')}, {len(s.get('shapes') or [])} shapes"
        if n == "instrument":
            note = f"DEBUG=0 vs DEBUG=2 at {s.get('shape')}"
        if n == "codegen":
            note = f"{s.get('n_sources_written', 0)} sources, {s.get('n_launches', 0)} launches"
        if n == "divide":
            note = f"N={s.get('N')}, backends={s.get('backends_attempted')}"
        rows.append(f"| {n} | {s.get('status')} | {s.get('seconds', '')} | {str(note)[:80]} |")
    return "\n".join([
        "# KDA battery", "",
        f"device: `{env.get('device', 'UNAVAILABLE')}`  ",
        f"code_root: `{man.get('code_root', 'UNAVAILABLE')}`  ",
        f"manifest_id: `{man.get('manifest_id', 'UNAVAILABLE')}`  ",
        f"payload sha256: `{(man.get('payload_tarball') or {}).get('sha256', 'not present')}`  ",
        f"session: `{json.dumps(man.get('session'))}`  ",
        f"wall: {bat.spent():,.1f}s of {bat.budget_s:,.0f}s", "",
        "## stages", "", *rows, "",
        "## what the prior run did not measure", "",
        "1. A manifest of the bytes under test, hashed, printed, and re-verified",
        "   before every stage, so a payload/GPU batch mismatch cannot be silent.",
        "2. The launch count with its shape stated, swept over C and over NT, from a",
        "   device-free census AND from the device, so constant-of-C and",
        "   proportional-to-T separate and the prior 12,082 can be reproduced to",
        "   the unit instead of assumed.",
        "3. The warm call timed with DEBUG=0 and ONE synchronize(), beside the",
        "   DEBUG=2 event-bracketed sum, with the instrument's own cost measured",
        "   rather than folded into the kernels.",
        "4. A decomposition into graph construction, rewrite passes,",
        "   lower+compile+issue, device drain, copy-out, with the unattributed",
        "   residual printed instead of distributed.",
        "5. The emitted kernel sources with PARALLEL=0, so the capture cannot",
        "   report 1 program out of 12,082 again.",
        "6. The float32 divide probe, with its generator cross-checked bit-for-bit",
        "   against probes/divide_precision.py.", "",
        "## not verified without a GPU", "",
        "CUDA-specific fields, nvidia-smi, the DEBUG=2 host-overhead separation, the",
        "CUDA/CL divide rates, and ncu. All report UNAVAILABLE with a reason on a",
        "non-CUDA device. No GPU number is invented.", "",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
