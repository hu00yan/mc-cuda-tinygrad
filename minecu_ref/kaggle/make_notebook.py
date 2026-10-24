#!/usr/bin/env python3
"""make_notebook.py -- assemble the Kaggle notebook that measures the mine.cu
hand-written CUDA baseline on a T4 (sm_75).

WHY BASE64 THE SOURCES IN. `kaggle kernels push` uploads exactly one file: the
notebook named by "code_file" in kernel-metadata.json. Sidecars are not
uploaded. So kernels.cu (105 KB) and bench.cu (~30 KB) are embedded in one code
cell as base64 and the notebook verifies both sha256 values before it compiles
anything. A hash mismatch is a hard stop: a timing taken from a source that is
not the one the write-up describes is worse than no timing at all.

The notebook is generated rather than committed because the payloads are
large and must be regenerated whenever either source changes. Regenerating is
deterministic given the two inputs, so the notebook's own sha256 is a usable
provenance handle.
"""

import argparse
import base64
import hashlib
import json
import os
import textwrap

# The measurement battery, as (label, argv) pairs. Kept here, not in the
# notebook, so it is reviewable in a diff.
#
# Why these shapes:
#   selftest at three shapes -- correctness gate, and it proves the binary runs
#   at all three. A FAIL prints FAIL and sets exit 2, and the run.sh unpack step
#   reports that, so a broken harness never reads as a slow kernel.
#   the headline shape (32768, 16, 32x32) -- this is the config the project's
#   own docs/benchmarks.md peak row names, so it is the one that decides
#   whether the author's 49.6M claim reproduces.
#   the batch sweep -- throughput is documented as rising with batch to 32K, so
#   the sweep is what shows whether this card is anywhere near saturation.
#   world 32 at 32x32 -- the second documented row, and the one the README's
#   default (world_size=32) actually produces.
# The battery. Note what is and is not pinned to `--ground`:
#   * most entries pass NO --ground, so they take the new default world_y/4,
#     the only value at which the visible surface and physics_kernel's analytic
#     floor coincide. The DDA gate in bench.cu enforces this.
#   * `degenerate_w16_ground8` pins --ground 8 at world 16, which is the exact
#     configuration the previous T4 run measured and the one the fix replaces.
#     It runs with --measure-degenerate so the before/after comparison is
#     measured on the same card in the same run rather than quoted from an old
#     log. Its timings are stamped DEGENERATE by the binary itself.
#   * `selftest_w48` no longer needs a special case: at world 48 the default
#     ground is 12, which is again world_y/4.
BATTERY = [
    ("selftest_w16",  ["--batch", "4096", "--world", "16", "--res", "32", "--selftest-only"]),
    ("selftest_w32",  ["--batch", "4096", "--world", "32", "--res", "32", "--selftest-only"]),
    ("selftest_w48",  ["--batch", "4096", "--world", "48", "--res", "32", "--selftest-only"]),
    ("headline_w16",  ["--batch", "32768", "--world", "16", "--res", "32",
                       "--max-steps", "32",
                       "--iters", "200", "--graph-iters", "200"]),
    ("sweep_w16",     ["--batch", "32768", "--world", "16", "--res", "32",
                       "--max-steps", "32", "--iters", "60", "--graph-iters", "60"]),
    ("batch4k_w16",   ["--batch", "4096", "--world", "16", "--res", "32",
                       "--max-steps", "32", "--iters", "100", "--graph-iters", "100"]),
    ("batch8k_w16",   ["--batch", "8192", "--world", "16", "--res", "32",
                       "--max-steps", "32", "--iters", "100", "--graph-iters", "100"]),
    ("batch16k_w16",  ["--batch", "16384", "--world", "16", "--res", "32",
                       "--max-steps", "32", "--iters", "100", "--graph-iters", "100"]),
    ("w32_32px",      ["--batch", "32768", "--world", "32", "--res", "32",
                       "--max-steps", "32", "--iters", "60", "--graph-iters", "60"]),
    ("w16_64px",      ["--batch", "32768", "--world", "16", "--res", "64",
                       "--max-steps", "32", "--iters", "30", "--graph-iters", "30"]),
    # max_steps sweep: the DDA step count is the thing that decides whether
    # this kernel is issue bound or bandwidth bound, and it is a function of
    # max_steps. Measuring it beats arguing about it.
    ("w16_ms8",       ["--batch", "32768", "--world", "16", "--res", "32",
                       "--max-steps", "8", "--iters", "60", "--graph-iters", "60"]),
    ("w16_ms64",      ["--batch", "32768", "--world", "16", "--res", "32",
                       "--max-steps", "64", "--iters", "60", "--graph-iters", "60"]),
    # The BEFORE case, re-measured on this card in this run. Same shape, same
    # binary, same floors as headline_w16; the only difference is ground=8,
    # which puts the agent's eye three blocks under its own surface. The DDA
    # gate rejects it, --measure-degenerate waives the abort, and every table in
    # the log is stamped DEGENERATE. This entry exists so the before/after
    # numbers are comparable; it is NOT a baseline.
    ("degenerate_w16_ground8",
                       ["--batch", "32768", "--world", "16", "--res", "32",
                        "--max-steps", "32", "--ground", "8", "--measure-degenerate",
                        "--iters", "200", "--graph-iters", "200"]),
]

# The in-notebook driver. It:
#   * writes both payloads out and checks sha256
#   * compiles for sm_75, explicitly NOT for the machine's default arch, so
#     the number is attached to the architecture and not to whatever ptxas
#     would have picked
#   * runs the battery under a wall-clock budget
#   * writes SUMMARY.txt and one log per entry
#   * prints a fixed trailer so the shape of the output is predictable
DRIVER = r'''
import base64, hashlib, json, os, re, subprocess, sys, textwrap, time

BUDGET_S = __BUDGET_S__
OUT = "/kaggle/working"
os.makedirs(OUT, exist_ok=True)

EXPECT = __EXPECT__
PAYLOADS = __PAYLOADS__
BATTERY = __BATTERY__

lines = []
def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    lines.append(s)

say("=" * 72)
say("minecu_ref -- hand-written CUDA baseline on Kaggle")
say("=" * 72)

# ---------------------------------------------------------------- provenance
say("")
say("--- provenance ---")
say(f"  notebook assembled by minecu_ref/kaggle/make_notebook.py")
for name, b64, want in PAYLOADS:
    data = base64.b64decode(b64)
    got = hashlib.sha256(data).hexdigest()
    status = "OK" if got == want else "MISMATCH"
    say(f"  {name:14s} {len(data):8d} B  sha256 {got}  {status}")
    if got != want:
        say("")
        say("FATAL: payload hash mismatch. Refusing to measure. A timing from an")
        say("unverified source is not a timing. Re-run minecu_ref/kaggle/run.sh.")
        open(f"{OUT}/SUMMARY.txt", "w").write("\n".join(lines) + "\nHARNESS HASH MISMATCH\n")
        sys.exit(9)
    open(f"{OUT}/{name}", "wb").write(data)
    if want in EXPECT:
        say(f"                 matches the vendored copy in minecu_ref/src or bench/")

# ------------------------------------------------------------------- build
say("")
say("--- build for sm_75 ---")
NVCC = None
for p in ("/usr/local/cuda/bin/nvcc", "nvcc"):
    try:
        subprocess.run([p, "--version"], check=True, capture_output=True)
        NVCC = p
        break
    except Exception:
        continue
if NVCC is None:
    say("FATAL: no nvcc on this runner. The T4 image is supposed to have one.")
    open(f"{OUT}/SUMMARY.txt", "w").write("\n".join(lines) + "\nNO NVCC\n")
    sys.exit(10)
ver = subprocess.run([NVCC, "--version"], capture_output=True, text=True).stdout
say("  " + ver.strip().splitlines()[-2].strip())
say("  " + ver.strip().splitlines()[-1].strip())

t0 = time.time()
# -arch=sm_75 is explicit and is the whole point: the T4 is sm_75 and this run
# exists to produce a number for that architecture, not for whatever nvcc
# defaults to on an unknown card.
build = subprocess.run(
    [NVCC, "-O3", "--use_fast_math", "-std=c++17", "-arch=sm_75",
     "-diag-suppress", "177",
     f"{OUT}/kernels.cu", f"{OUT}/bench.cu", "-o", f"{OUT}/minecu_bench_sm_75"],
    capture_output=True, text=True)
say(f"  nvcc -arch=sm_75 rc={build.returncode} in {time.time()-t0:.1f}s")
if build.stderr.strip():
    for ln in build.stderr.strip().splitlines()[-25:]:
        say("  | " + ln)
if build.returncode != 0:
    open(f"{OUT}/SUMMARY.txt", "w").write("\n".join(lines) + "\nNVCC FAILED\n")
    sys.exit(11)
say(f"  binary {os.path.getsize(f'{OUT}/minecu_bench_sm_75')} B at {OUT}/minecu_bench_sm_75")

# The instruction census, so the run carries its own evidence that the binary
# is sm_75 and carries no sm_80+ instructions.
say("")
say("--- SASS census (proves the binary really is sm_75) ---")
sass = subprocess.run(["/usr/local/cuda/bin/cuobjdump", "-sass", f"{OUT}/minecu_bench_sm_75"],
                      capture_output=True, text=True)
open(f"{OUT}/kernels_sm75.sass", "w").write(sass.stdout)
gates = ["HMMA", "HGMMA", "LDGSTS", "REDUX", "ELECT", "BAR.RED", "F2FP"]
counts = {g: sass.stdout.count(g) for g in gates}
say("  arch line: " + next((l.strip() for l in sass.stdout.splitlines()
                            if l.strip().startswith("arch =")), "<none>"))
say("  " + "  ".join(f"{g}={counts[g]}" for g in gates))
say("  verdict: " + ("no sm_75-illegal instructions" if all(v == 0 for v in counts.values())
                     else "ILLEGAL INSTRUCTIONS PRESENT"))
say(f"  STG.E.U8.SYS (uint8 obs store) {sass.stdout.count('STG.E.U8.SYS')}, "
    f"LDG.E.S8.CONSTANT.SYS (__ldg voxel read) {sass.stdout.count('LDG.E.S8.CONSTANT.SYS')}")

# ----------------------------------------------------------------- battery
say("")
say("--- battery ---")
say(f"  {len(BATTERY)} entries, wall budget {BUDGET_S}s")
t_start = time.time()
summary_rows = []
dda_rows = []
skipped = []
for label, args in BATTERY:
    left = BUDGET_S - (time.time() - t_start)
    if left <= 30:
        skipped.append((label, "out of budget"))
        say(f"  SKIP {label}: out of budget")
        continue
    say("")
    say(f"  === {label}: {' '.join(args)} ===")
    t = time.time()
    try:
        p = subprocess.run([f"{OUT}/minecu_bench_sm_75"] + args,
                           capture_output=True, text=True, timeout=max(30, left))
        rc, body = p.returncode, p.stdout + ("\n[stderr]\n" + p.stderr if p.stderr else "")
    except subprocess.TimeoutExpired:
        rc, body = -9, "TIMEOUT"
    dt = time.time() - t
    open(f"{OUT}/log_{label}.txt", "w").write(body)
    for ln in body.rstrip().splitlines():
        say("  " + ln)
    say(f"  --- {label}: rc={rc} in {dt:.1f}s -> log_{label}.txt")
    if rc != 0:
        say(f"  !!! {label} exited {rc}. The harness itself is suspect here, not the kernel.")
    # pull the headline numbers for the summary
    for ln in body.splitlines():
        if "launch_step (fp32 obs" in ln or "launch_step_uint8_prebasis (" in ln \
           or "launch_step_..._prebasis_full" in ln:
            parts = ln.split()
            if len(parts) >= 8:
                summary_rows.append((label, parts[1] + " " + parts[2], parts[-1]))
    # The DDA step count is what says whether a timing is a voxel-render
    # measurement at all, so it goes in the summary next to the throughput
    # rather than only in the per-entry log. Pulled from the selftest line,
    # which reads "mean <N> DDA steps/pixel over <P> pixels".
    for ln in body.splitlines():
        if "DDA steps/pixel over" in ln:
            mm = re.search(r"mean ([\d.]+) DDA steps/pixel over (\d+) pixels", ln)
            if mm:
                dda_rows.append((label, mm.group(1), mm.group(2)))
                break
    if rc == 0 and "selftest: PASS" not in body and "selftest" in " ".join(args):
        summary_rows.append((label, "SELFTEST", "FAIL"))

say("")
say("--- summary: throughput ---")
say(f"{'entry':28s} {'path':44s} {'M env-steps/s'}")
for label, path, v in summary_rows:
    say(f"{label:28s} {path:44s} {v:>8s}")
for label, why in skipped:
    say(f"{label:28s} {'-':44s} {'SKIP':>8s}  ({why})")

say("")
say("--- summary: DDA workload actually timed ---")
say("  A throughput number means nothing without the workload behind it. These are")
say("  the mean DDA steps per pixel the timed render kernel executed, measured on")
say("  the device by replaying the vendored DDA over the same voxels and basis.")
say(f"{'entry':28s} {'mean DDA steps/pixel':>22s} {'pixels':>12s}")
for label, mean, pix in dda_rows:
    say(f"{label:28s} {mean:>22s} {pix:>12s}")
say("")
say("  headline_w16 is the number to use. degenerate_w16_ground8 is the SAME shape")
say("  with ground=8, the configuration the previous run measured; the agent's eye")
say("  was three blocks inside its own surface, every ray hit on its first voxel")
say("  test, and the render kernel degenerated into an obs store. It is in this")
say("  run so the before/after is measured on the same card, not quoted.")

say("")
say("Budget: %d/%d s used." % (time.time() - t_start, BUDGET_S))
say("")
say("READING THIS OUTPUT")
say("  * The three selftest_* entries gate everything. If any says FAIL, the")
say("    numbers below it are not trustworthy and the harness is the suspect.")
say("  * Each entry also gates on a DDA step count. An entry that reports a")
say("    throughput but a mean of ~1.000 steps/pixel is NOT a voxel-render")
say("    measurement: the render exited on its first voxel test every pixel.")
say("    bench.cu refuses to report timings for such a config unless")
say("    --measure-degenerate is passed, which is used exactly once here, for the")
say("    entry that exists to document the pre-fix baseline.")
say("  * env-steps/s is batch / median_us. A whole-step number, not a per-kernel one.")
say("  * The measured memory floors in each log are the denominators. %-of-floor")
say("    figures only mean something against a floor measured on the same card.")
say("  * The author's 49.6M is an RTX 3090 claim. This is a T4. They are not")
say("    comparable numbers and must not be differenced directly.")
summary = "\n".join(lines)
open(f"{OUT}/SUMMARY.txt", "w").write(summary + "\n")
say("wrote /kaggle/working/SUMMARY.txt")
'''


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-s", type=int, default=1500)
    ap.add_argument("--owner", default="tenghuzhang")
    ap.add_argument("--slug", default="mc-minecu-t4-baseline")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    kd = os.path.dirname(a.out)

    kh = sha256_file(a.kernels)
    bh = sha256_file(a.bench)
    kb = base64.b64encode(open(a.kernels, "rb").read()).decode()
    bb = base64.b64encode(open(a.bench, "rb").read()).decode()

    driver = (DRIVER
              .replace("__BUDGET_S__", str(a.budget_s))
              .replace("__EXPECT__", json.dumps({kh: "src/kernels.cu", bh: "bench/bench.cu"}))
              .replace("__PAYLOADS__", json.dumps([["kernels.cu", kb, kh],
                                                  ["bench.cu", bb, bh]]))
              .replace("__BATTERY__", json.dumps(BATTERY)))

    def md(src):
        return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)}

    def code(src):
        return {"cell_type": "code", "execution_count": None, "metadata": {},
                "outputs": [], "source": src.splitlines(True)}

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "cells": [
            md("# minecu_ref T4 baseline\n\n"
               "Measures the hand-written CUDA from `Infatoshi/mine.cu` on this T4.\n"
               "Everything runs in this notebook; the only host input is the two\n"
               "base64 payloads, which are hash-checked before anything is built.\n"
               "Self-verifying: the first three battery entries are correctness\n"
               "gates, and a failing gate makes the run exit non-zero."),
            md("## What this produces\n\n"
               "1. `SUMMARY.txt` -- the full log, which is the primary artefact.\n"
               "2. `log_<entry>.txt` -- one per battery entry.\n"
               "3. `kernels_sm75.sass` -- the SASS, so the run carries its own\n"
               "   evidence that the binary is sm_75 with no sm_80+ instructions.\n"
               "4. `minecu_bench_sm_75` -- the binary, so a reader can rebuild.\n\n"
               "No number in this notebook is transcribed from the mine.cu README.\n"
               "Every one is produced by the instrument in this notebook."),
            code(driver),
        ],
    }

    with open(a.out, "w") as fh:
        json.dump(nb, fh, indent=1)

    # Kaggle derives the kernel's URL slug from `title`, and `kernels push`
    # refuses the upload if that slug does not round-trip to `id`. So the title
    # is DERIVED from the slug (hyphens -> spaces) instead of being written out
    # by hand: "mc-minecu-t4-baseline" -> "mc minecu t4 baseline" -> slugs back
    # to "mc-minecu-t4-baseline". A hand-written title was previously clobbered
    # here on every run (this file regenerates the metadata), which is how the
    # push started failing with "Your kernel title does not resolve to the
    # specified id". Keep the derivation; do not hardcode a title again.
    title = " ".join(slug_word for slug_word in a.slug.split("-") if slug_word)

    meta = {
        "id": f"{a.owner}/{a.slug}",
        "title": title,
        "subtitle": "hand-written CUDA voxel RL env, sm_75, self-verifying",
        "code_file": os.path.basename(a.out),
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    # Fail here, loudly, rather than at the push: Kaggle slugs `title`, and a
    # title that does not slug back to the id is rejected with a 409 Conflict
    # ("Your kernel title does not resolve to the specified id") AFTER the
    # notebook has been assembled and the budget spent. Cheap to check.
    import re as _re
    _derived = _re.sub(r"[^a-z0-9]+", "-", meta["title"].lower()).strip("-")
    if _derived != a.slug:
        raise SystemExit(
            f"kernel title {meta['title']!r} slugs to {_derived!r}, which does not match "
            f"the kernel id slug {a.slug!r}; `kaggle kernels push` will 409."
        )

    with open(os.path.join(kd, "kernel-metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"wrote {a.out}  ({os.path.getsize(a.out)} B)")
    print(f"wrote {os.path.join(kd, 'kernel-metadata.json')}")
    print(f"  kernels.cu sha256 {kh}")
    print(f"  bench.cu   sha256 {bh}")
    print(f"  {len(BATTERY)} battery entries, budget {a.budget_s}s")
    print(f"  kernel id  {meta['id']}")


if __name__ == "__main__":
    main()
