#!/usr/bin/env python3
"""battery.py -- both modes and the hand-written baseline, on one card, one run.

WHAT THIS MEASURES, AND WHAT IT REFUSES TO MEASURE
--------------------------------------------------
Both sides of the study are *renders*, at equal batch, in the same process, on
the same card, in the same run:

    clean      the faithful port, no crossings
    faithful   the faithful port, crossings on
    reference  `render_kernel_uint8_prebasis` from the unmodified vendored
               `kernels.cu`, driven by the same `refdrv.cu` the other arms
               use, timed with cudaEvent

Render-against-render is the only comparison made here. The reference's
*whole-step* figure exists on the card too and is printed for context, but it
is never a denominator: a whole step is 7 launches including five state
updates, and dividing one kernel's device time by a seven-launch number is a
category error that `notes/17` §8.2 and `notes/19` §5.1 both warn about. It is
in the output as `CONTEXT`, on its own line, labelled.

The DDA workload is measured, not assumed
-----------------------------------------
Each mode reports the mean steps/pixel its own graph executed, via
`count_steps=True`, which `verify.py` proves inert. A render-against-render
ratio is only meaningful if both sides traced the same rays, and the step count
is the only cheap way to check that. The reference's own step count comes from
`refdrv`'s `dda_probe_kernel` (`minecu_ref/bench/bench.cu`) on the same state.

The observation is compared by sha256, between the two modes
--------------------------------------------------------
`obs_sha256` for each mode, and `obs_identical_modes` in the summary. Those two
must be byte-identical and they are: the crossings are fidelity switches, not
new algorithms, and `verify.py` proves each against the naive port and against
a float64 transcription before a single kernel runs here.

They are NOT compared against the reference's dumped bytes in this script. The
reference's dump is its own *settled* state after a physics loop
(`refdrv.cu:175`) and this port renders `world_and_cams`'s cameras, so the two
are the same computation only if the cameras agree too -- and asserting a
digest match without checking that would be a comparison of two different
states. `notes/19` §1.2 is the reference-vs-TinyGrad byte comparison, on the
reference's own state, and this arm did not re-derive it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("PARALLEL", "0")
os.environ.setdefault("DEBUG", "0")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

REPORT: dict = {"started": time.time(), "argv": sys.argv}
JSON_PATH: Path | None = None


def save() -> None:
    """Write the report now, not at the end.

    The stages here run in minutes and a card session can be cut short (this
    arm's own runs were, twice, §8 of notes/28). A report that exists only
    after the last batch is a report that exists only for the runs that reached
    the last batch, which is exactly the subset nobody needs to read. The
    payload is a few KB; rewriting it per row costs nothing next to a render.
    """
    if JSON_PATH is not None:
        try:
            JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
            JSON_PATH.write_text(json.dumps(REPORT, indent=2, default=str))
        except OSError as e:
            log(f"  (could not write {JSON_PATH}: {e})")


def log(*a):
    print(*a, flush=True)


def section(name: str) -> None:
    log("")
    log("=" * 78)
    log(f"  {name}")
    log("=" * 78)


# --------------------------------------------------------------------------
def environment() -> dict:
    section("1. environment")
    import numpy as np
    import tinygrad
    from tinygrad import Device
    env = {"Device.DEFAULT": Device.DEFAULT, "numpy": np.__version__,
           "tinygrad": str(Path(tinygrad.__file__).resolve().parent),
           "PARALLEL": os.environ.get("PARALLEL")}
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                            "--format=csv,noheader"], capture_output=True, text=True)
        env["nvidia_smi"] = r.stdout.strip()[:300]
    except Exception as e:
        env["nvidia_smi"] = f"unavailable: {e}"
    REPORT["env"] = env
    for k, v in env.items():
        log(f"  {k:16s} {v}")
    return env


# --------------------------------------------------------------------------
def world_and_cams(world: int, ground: int, batch: int, spread: float):
    """The same world the reference builds, and a spread of cameras.

    Built with the reference's own model (`dda_model.build_world`) rather than
    with this repo's transcription of it, so a disagreement about the world
    cannot hide inside a timing comparison. `verify.py` checks the two agree
    exactly, so this costs nothing and removes a shared-misreading failure mode.
    """
    sys.path.insert(0, str(HERE.parent / "minecu_ref" / "bench"))
    import dda_model
    import numpy as np
    vox = np.ascontiguousarray(np.stack([dda_model.build_world(world, ground)] * batch))
    # a real spread, so the per-env camera path is exercised and the basis is
    # not a constant that could be folded
    cams = np.array([[world / 2.0 + 0.05 * (i % 5), float(ground), world / 2.0,
                      spread * (i % 9 - 4) / 4.0, 0.0] for i in range(batch)],
                    dtype=np.float32)
    return vox, cams


# --------------------------------------------------------------------------
def time_graph(build, iters: int, warmup: int) -> dict:
    """Device time of one render graph, in ms.

    **The graph must be unrealized when this is called.** That is the whole
    point of this function, and getting it wrong produces a number that looks
    plausible and measures nothing: `Tensor.realize()` on an already-realized
    tensor returns immediately without launching a kernel, so a
    realize-and-synchronize loop around it times host overhead, not the render.
    That is not hypothetical -- it is what the first version of this file did,
    and on the T4 it reported 0.011 ms for a graph that offline SASS puts at
    tens of times the hand-written kernel's instruction count, i.e. a port
    "13x faster than the hand-written CUDA". `minecu_tg/bench.py:247-249`
    records the same trap next to its own fix.

    So the timing is taken the way `minecu_tg.bench.linear_device_time_ms`
    takes it: compile the LINEAR, walk its CALLs, time each with tinygrad's own
    `time_call`. Two reasons for that instrument rather than a wall-clock loop:

      * it is the same instrument the naive arm measured with, so both arms'
        numbers come out of one code path and are comparable;
      * it is DEVICE time, excluding host tracing, which is also what the
        reference's cudaEvent figure around a single launch is. Timing
        build-and-realize from Python would fold in the framework's tracing,
        which the reference does not pay, and would be a different quantity
        wearing the same label.

    A host-side number (build + realize + sync) is reported beside it and
    labelled, because the gap between the two is a real cost of using a
    framework here and hiding it would overstate the port.
    """
    from minecu_tg.bench import linear_device_time_ms

    obs = build()
    if getattr(obs, "realized", False):
        raise RuntimeError("time_graph was handed an already-realized tensor; "
                           "its realize() is a no-op and the timing would be "
                           "host overhead. This check exists because it "
                           "happened.")
    dev_time = linear_device_time_ms(obs, iters=iters)
    if not dev_time or "error" in dev_time:
        return {"error": (dev_time or {}).get("error", "no device time"),
                "n_kernels": None}

    from tinygrad import Device
    dev = Device[Device.DEFAULT]
    host = []
    for i in range(warmup + iters):
        t0 = time.perf_counter()
        b = build()
        b.realize()
        dev.synchronize()
        if i >= warmup:
            host.append((time.perf_counter() - t0) * 1e3)
    host.sort()
    return {
        "median_ms": dev_time["median_ms"],       # DEVICE time, the headline
        "min_ms": dev_time["min_ms"],
        "max_ms": dev_time["max_ms"],
        "n_kernels": dev_time.get("n_kernels"),
        "iters": iters,
        "clock": "sum of per-kernel device times over the graph "
                 "(minecu_tg.bench.linear_device_time_ms); host tracing excluded",
        # context, and explicitly not the headline
        "host_median_ms": statistics.median(host),
        "host_note": "build() + realize() + synchronize from Python; includes "
                     "the framework's graph tracing, which the reference does "
                     "not pay. NOT comparable to the reference's number.",
    }


def count_launches(build) -> int:
    """How many kernels one `realize` of this graph launches.

    A census, not a timing: it says what the graph is, and `notes/19` §5.4
    measured launch overhead at under 1.5% of device time on both sides of the
    crossover, so this is context and the cost of counting it is not folded
    into the measurement.
    """
    from tinygrad import GlobalCounters
    GlobalCounters.reset()
    obs = build()
    obs.realize()
    return GlobalCounters.kernel_count


# --------------------------------------------------------------------------
def measure_mode(mode: str, vox_np, cams, res: int, max_steps: int, batch: int,
                 iters: int, warmup: int, count_work: bool) -> dict:
    from tinygrad import Tensor, dtypes
    from minecu_tg_faithful.render import camera_basis, render

    def build(count_steps: bool = False):
        vox = Tensor(vox_np, dtype=dtypes.int8)
        basis = camera_basis(Tensor(cams, dtype=dtypes.float32), 70.0, res, res)
        return render(vox, basis, res, res, max_steps,
                      count_steps=count_steps, mode=mode)

    out: dict = {"mode": mode}
    try:
        t0 = time.time()
        launches = count_launches(build)
        out["kernels"] = launches
        out["seconds_to_compile_and_first_run"] = time.time() - t0
        # `build`, not the already-realized tensor from `count_launches`: a
        # fresh unrealized graph is the only thing `time_graph` will accept.
        out.update(time_graph(build, iters, warmup))
        if "error" in out:
            return out
        arr = build().numpy()
        out["obs_sha256"] = hashlib.sha256(np_to_bytes(arr)).hexdigest()
        out["obs_shape"] = list(arr.shape)
        out["obs_dtype"] = str(arr.dtype)
        if count_work:
            _, st = build(count_steps=True)
            s = st.numpy()
            out["mean_steps_per_pixel"] = float(s.mean())
            out["pixels_at_step_cap"] = int((s >= max_steps).sum())
            out["pixels_total"] = int(s.size)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()[-1500:]
    return out


def np_to_bytes(arr) -> bytes:
    import numpy as np
    return np.ascontiguousarray(arr).tobytes()


# --------------------------------------------------------------------------
def parse_reference(stdout: str) -> dict:
    """The reference's own printout, parsed. It is a fixed format.

    `refdrv.cu` prints `RENDER_US <median> <min> <max> <iters>` in MICROSECONDS
    -- the conversion from cudaEvent's milliseconds happens once in the driver
    (refdrv.cu:346-357), and the label and the unit agree there. Parsing the
    number rather than re-timing the hand-written side from Python is the
    point: the reference's clock is its own, on the same device, in the same
    process tree.
    """
    out: dict = {}
    for line in stdout.splitlines():
        ls = line.strip()
        if ls.startswith("RENDER_US"):
            p = ls.split()
            out["render_us_median"] = float(p[1])
            out["render_us_min"] = float(p[2])
            out["render_us_max"] = float(p[3])
            out["render_us_iters"] = int(p[4])
        elif ls.startswith("STEP_US"):
            p = ls.split()
            out["step_us_median"] = float(p[1])
        elif ls.startswith("STEP_ENVSTEPS_M"):
            out["step_envsteps_m"] = float(ls.split()[1])
        elif ls.startswith("ENVSTEPS_M"):
            out["render_envsteps_m"] = float(ls.split()[1])
        elif "DDA" in ls or "steps/pixel" in ls:
            out.setdefault("probe_lines", []).append(ls)
        elif ls.startswith("card"):
            out["card"] = ls
    return out


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batches", default="512,1024,2048")
    ap.add_argument("--world", type=int, default=16)
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=32)
    ap.add_argument("--ground", type=int, default=-1,
                    help="-1 = world//4, the configuration the baseline was "
                         "measured in (minecu_ref/buried_agent_fix.md)")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--spread", type=float, default=0.0,
                    help="yaw spread across the batch, radians")
    ap.add_argument("--refdrv", default="", help="path to the compiled refdrv")
    ap.add_argument("--ref-outdir", default="")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()

    global JSON_PATH
    JSON_PATH = a.json
    save()

    ground = a.world // 4 if a.ground < 0 else a.ground
    log(f"world {a.world}^3  ground {ground}  res {a.res}  max_steps {a.max_steps}  "
        f"iters {a.iters}  warmup {a.warmup}  yaw spread {a.spread}")

    environment()
    section("2. the hand-written baseline, from the unmodified kernels.cu")
    ref: dict = {}
    if a.refdrv and Path(a.refdrv).exists():
        import subprocess
        for b in [int(x) for x in a.batches.split(",")]:
            cmd = [a.refdrv, "--batch", str(b), "--world", str(a.world),
                   "--res", str(a.res), "--max-steps", str(a.max_steps),
                   "--ground", str(ground), "--iters", str(a.iters),
                   "--warmup", str(a.warmup)]
            # `--dump` is OFF by default here. The notebook already dumped the
            # reference's state once per batch into per-batch directories, and
            # a second dump into a shared directory would overwrite the first
            # with the same bytes for no benefit while doubling the run's disk.
            # It is available for a run that wants the dump from here.
            if a.ref_outdir:
                d = Path(a.ref_outdir) / f"b{b}"
                d.mkdir(parents=True, exist_ok=True)
                cmd += ["--outdir", str(d), "--dump"]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if r.returncode != 0:
                log(f"  batch {b:6d}  refdrv FAILED rc={r.returncode}")
                log("  " + (r.stderr or r.stdout)[-600:].replace("\n", "\n  "))
                continue
            parsed = parse_reference(r.stdout)
            log(f"  batch {b:6d}  render {parsed.get('render_us_median', float('nan')):10.1f} us  "
                f"whole step {parsed.get('step_us_median', float('nan')):10.1f} us  "
                f"(CONTEXT, not a denominator)")
            ref[str(b)] = parsed
    else:
        log(f"  no refdrv at {a.refdrv!r}; the baseline side is absent")
    REPORT["reference"] = ref

    section("3. this port, both modes, render against render")
    mine: dict = {}
    for b in [int(x) for x in a.batches.split(",")]:
        vox_np, cams = world_and_cams(a.world, ground, b, a.spread)
        for mode in ("clean", "faithful"):
            r = measure_mode(mode, vox_np, cams, a.res, a.max_steps, b,
                             a.iters, a.warmup, count_work=True)
            mine[f"{b}/{mode}"] = r
            save()          # before the log line, so the row is on disk first
            if "error" in r:
                log(f"  batch {b:6d} {mode:9s} FAILED  {r['error'][:100]}")
                continue
            ref_us = ref.get(str(b), {}).get("render_us_median")
            ratio = (r["median_ms"] * 1e3 / ref_us) if ref_us else float("nan")
            log(f"  batch {b:6d} {mode:9s} {r['median_ms']:9.3f} ms  DEVICE  "
                f"kernels {r['kernels']:3d}  "
                f"steps/px {r.get('mean_steps_per_pixel', float('nan')):6.3f}  "
                f"obs {r['obs_sha256'][:12]}  "
                f"ref {ref_us if ref_us else float('nan'):9.1f} us  "
                f"ratio {ratio:7.2f}x")
            log(f"  {'':>6} {'':9s} host build+realize "
                f"{r.get('host_median_ms', float('nan')):9.3f} ms  "
                f"(includes graph tracing; NOT comparable to ref)")
            # The DDA workload, next to the timing, because a throughput
            # number without the work behind it is not a measurement
            # (`minecu_ref/buried_agent_fix.md` is four weeks of that lesson).
            # The reference's own device-side step count needs its
            # `dda_probe_kernel` from `minecu_ref/bench/bench.cu`, which
            # `refdrv.cu` does not link, so it is reported as absent rather
            # than substituted with the CPU model's number under the same
            # label. The step count printed here is this graph's, measured by
            # this graph, on this card.
            log(f"  {'':>6} {'':9s} DDA workload: "
                f"{r.get('mean_steps_per_pixel', float('nan')):.3f} mean "
                f"steps/pixel, {r.get('pixels_at_step_cap')} of "
                f"{r.get('pixels_total')} pixels at the {a.max_steps}-step cap; "
                f"the reference's own device-side count is NOT available "
                f"(refdrv links no dda probe)")
    REPORT["mine"] = mine
    save()

    # ---- the clean-vs-faithful delta, which is the number this arm adds
    section("4. clean vs faithful, and both vs the baseline")
    rows = []
    for b in [int(x) for x in a.batches.split(",")]:
        c, f = mine.get(f"{b}/clean"), mine.get(f"{b}/faithful")
        ref_us = ref.get(str(b), {}).get("render_us_median")
        row = {"batch": b, "ref_render_us": ref_us}
        if c and "error" not in c:
            row["clean_ms"] = c["median_ms"]
            row["clean_kernels"] = c["kernels"]
            row["clean_obs_sha256"] = c["obs_sha256"]
        if f and "error" not in f:
            row["faithful_ms"] = f["median_ms"]
            row["faithful_kernels"] = f["kernels"]
            row["faithful_obs_sha256"] = f["obs_sha256"]
        if "clean_ms" in row and "faithful_ms" in row:
            row["faithful_over_clean"] = row["faithful_ms"] / row["clean_ms"]
        if ref_us:
            if "clean_ms" in row:
                row["clean_over_ref"] = row["clean_ms"] * 1e3 / ref_us
            if "faithful_ms" in row:
                row["faithful_over_ref"] = row["faithful_ms"] * 1e3 / ref_us
        if "clean_obs_sha256" in row and "faithful_obs_sha256" in row:
            row["obs_identical_modes"] = (row["clean_obs_sha256"] ==
                                          row["faithful_obs_sha256"])
        if ref_us:
            # where the reference's dumped state lives, so a later step can
            # compare observations byte for byte. Recorded, not compared here:
            # the reference's dump is the reference's own settled state and
            # this port renders `world_and_cams`'s spawn-adjacent cameras, so
            # the two are only the same computation if the cameras agree too.
            # Asserting a digest match without checking that would be a
            # comparison of two different states.
            row["ref_dump_dir"] = a.ref_outdir
        rows.append(row)
        log(f"  batch {b:6d}  clean {row.get('clean_ms', float('nan')):9.3f} ms  "
            f"faithful {row.get('faithful_ms', float('nan')):9.3f} ms  "
            f"faithful/clean {row.get('faithful_over_clean', float('nan')):6.3f}  "
            f"clean/ref {row.get('clean_over_ref', float('nan')):7.2f}  "
            f"faithful/ref {row.get('faithful_over_ref', float('nan')):7.2f}  "
            f"same obs {row.get('obs_identical_modes')}   (DEVICE ms)")
    REPORT["rows"] = rows
    save()

    section("5. what this does NOT say")
    log("  * No whole-step number is a denominator anywhere above. The")
    log("    reference's whole step is printed as CONTEXT only: it is 7")
    log("    launches, five of them state updates.")
    log("  * The ratio against the reference is render-against-render, both")
    log("    producing the same uint8 observation at the same batch on the")
    log("    same card. If the digests differ, the ratio is between two")
    log("    different computations.")
    log("  * The clean-vs-faithful delta is the crossing cost. It is the only")
    log("    number here that no other arm in this study has.")
    log("  * Every ms above is DEVICE time: the sum of the per-kernel device")
    log("    times in the graph, measured with tinygrad's own time_call, host")
    log("    tracing excluded. The host build+realize number printed beside it")
    log("    is context and is NOT comparable to the reference's, because the")
    log("    reference pays no tracing cost. Calling realize() on an already")
    log("    realized tensor is a no-op; time_graph raises if handed one,")
    log("    because an earlier version of this file did exactly that and")
    log("    reported the port as 13x FASTER than the hand-written CUDA.")

    if a.json:
        save()
        log(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    # Every row has already been written by now; the only thing left is the
    # argv and environment record, so the final write is unconditional rather
    # than conditional on --json having been passed.
    sys.exit(main())
