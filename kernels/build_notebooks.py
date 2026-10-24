#!/usr/bin/env python3
"""build_notebooks.py -- generate the Kaggle notebooks from readable source.

The .ipynb files are the artefact that `kaggle kernels push` uploads, and they
are checked in, so the push never depends on this script. This script exists so
the notebook source is reviewable as Python with real indentation instead of as a
JSON array of escaped lines, and so the GPU and TPU variants share one copy of
the device-detection code.

    .venv/bin/python kernels/build_notebooks.py
    .venv/bin/python kernels/validate_notebooks.py     # then check the output

Writes:
    kernels/probe/probe.ipynb        the battery (GPU metadata attaches this)
    kernels/probe-tpu/tpu-facts.ipynb reachability + device facts only
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------
# shared cell: what am I running on?
# --------------------------------------------------------------------------
CELL_DEVICE = '''
import os, platform, subprocess, sys

OUT = "/kaggle/working"
INPUT = "/kaggle/input"
LOGD = os.path.join(OUT, "logs")
os.makedirs(LOGD, exist_ok=True)
FACTS = {}          # filled in below; later cells read it


def sh(cmd, timeout=120):
    """Run a shell command, return its combined output, never raise."""
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (p.stdout + p.stderr).strip()
    except subprocess.TimeoutExpired:
        return f"<timeout after {timeout}s>"
    except Exception as e:
        return f"<{type(e).__name__}: {e}>"


def sh_ok(cmd, timeout=120):
    """Like sh(), but empty unless the command SUCCEEDED.

    Needed because "nvidia-smi: command not found" on stderr is non-empty
    output from a command that failed, and counting it as a GPU is exactly the
    kind of lie this cell exists to prevent.
    """
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (p.stdout + p.stderr).strip() if p.returncode == 0 else ""
    except Exception:
        return ""


def say(*a):
    print(*a, flush=True)


def fact(k, v):
    FACTS[k] = v
    return v


# ---- what kind of machine is this? decided by evidence, not by the metadata
gpu_names = sh_ok("nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader")
n_gpu = len(gpu_names.splitlines())
tpu_lib = sh_ok("ls -1 /dev/accel* 2>/dev/null; ls -1 /usr/lib/libtpu* /usr/lib/x86_64-linux-gnu/libtpu* 2>/dev/null")

kind = "CPU"
if n_gpu > 0:
    kind = "GPU"
elif tpu_lib:
    # a TPU runner has no working nvidia-smi, but the accelerator host devices
    # and libtpu are both there. jax is asked next; this is the cheap check that
    # runs before paying for a jax import.
    kind = "TPU"
fact("kind", kind)

say("=" * 78)
say("device")
say("=" * 78)
say(f"  python      {sys.version.split()[0]}  {platform.platform()}")
say(f"  cwd         {os.getcwd()}")
say(f"  kaggle input{os.path.isdir(INPUT)}  "
    f"{sorted(os.listdir(INPUT)) if os.path.isdir(INPUT) else '(nothing attached)'}")
say(f"  nvidia-smi  {gpu_names or 'unavailable'}")
say(f"  tpu nodes   {tpu_lib or 'unavailable'}")
say(f"  ==> this is a {kind} runner")

env_keys = ("KAGGLE_", "CUDA_", "NVIDIA_", "TPU_", "PJRT_", "XLA_", "LD_LIBRARY_PATH")
with open(os.path.join(LOGD, "00-environment.txt"), "w") as f:
    f.write(f"# {platform.platform()} python {sys.version.split()[0]}\\n")
    f.write(f"# nvidia-smi: {gpu_names or 'unavailable'}\\n")
    f.write(f"# tpu: {tpu_lib or 'unavailable'}\\n")
    for k in sorted(os.environ):
        if k.startswith(env_keys):
            line = f"{k}={os.environ[k]}"
            say("  env " + line[:150])
            f.write(line + "\\n")
say(f"  disk        {sh('df -h ' + OUT + ' | tail -1')}")
say(f"  cpus        {os.cpu_count()}")
'''

# --------------------------------------------------------------------------
# payload + venv
# --------------------------------------------------------------------------
CELL_PAYLOAD = '''
import glob, hashlib, json, os, subprocess, sys, tarfile, tempfile

# The payload is unpacked to a temp dir, NOT into /kaggle/working. Everything
# under /kaggle/working comes back in the output zip, and the payload tree grows
# to ~25 MB the moment `make check` regenerates the oc reference cells. The
# artefacts (logs, SUMMARY.md, codegen) go to OUT; the source does not.
PARENT = os.path.join(tempfile.gettempdir(), "mc-cuda-tinygrad")
ROOT = os.path.join(PARENT, "probe")      # the tarball's own top-level directory
PY = None                                  # the interpreter the probes run under
AVAIL = {}                                 # module -> version, for skip logic

say("")
say("=" * 78)
say("payload")
say("=" * 78)

# ---- 1. the payload arrives as an attached Kaggle dataset, already mounted.
# The mount layout is not something to guess: on 2026-09-27 an attached dataset
# showed up as /kaggle/input/datasets/... , not /kaggle/input/<slug>/ as the docs
# suggest, and a one-level glob missed it and silently skipped the whole
# battery. So: walk, bounded depth, and print what was there.
def find_payload(root, max_depth=4):
    hits = []
    root = root.rstrip("/")
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath[len(root):].count("/")
        if depth >= max_depth:
            dirnames[:] = []
        for fn in filenames:
            if fn == "payload.tgz":
                hits.append(os.path.join(dirpath, fn))
    return sorted(hits)

cands = []
if os.path.isdir(INPUT):
    for dirpath, dirnames, filenames in os.walk(INPUT):
        cands.append(dirpath)
        if len(cands) > 40:
            break
tars = find_payload(INPUT) if os.path.isdir(INPUT) else []
if not tars:
    say("  no payload.tgz anywhere under " + INPUT)
    say(f"  saw: {cands[:20]}")
    say("  the kernel metadata must list dataset_sources, and the dataset")
    say("  version must be finished (kaggle datasets status -> ready) before")
    say("  the kernel is pushed")
else:
    tgz = tars[0]
    want = None
    side = os.path.join(os.path.dirname(tgz), "payload.sha256")
    if os.path.isfile(side):
        want = open(side).read().split()[0]
    got = hashlib.sha256(open(tgz, "rb").read()).hexdigest()
    say(f"  archive     {tgz}  {os.path.getsize(tgz)/1024:.1f} KB")
    say(f"  sha256      {got}")
    FACTS["payload_sha"] = got
    if want and want != got:
        say(f"  SHA MISMATCH, expected {want}")
        say("  refusing to run: the payload is not the one that was packed")
        tgz = None
    else:
        say("  sha256      matches payload.sha256")
        os.makedirs(PARENT, exist_ok=True)
        with tarfile.open(tgz, "r:gz") as tf:
            try:
                tf.extractall(PARENT, filter="data")
            except TypeError:       # python < 3.12 has no filter argument
                tf.extractall(PARENT)
        man = json.load(open(os.path.join(ROOT, "MANIFEST.json")))
        bad = 0
        for e in man["files"]:
            p = os.path.join(PARENT, e["path"])
            if not os.path.isfile(p) or hashlib.sha256(open(p, "rb").read()).hexdigest() != e["sha256"]:
                bad += 1
                say(f"  MANIFEST MISMATCH {e['path']}")
        say(f"  extracted   {len(man['files'])} files into {PARENT}, {bad} hash mismatches")

# ---- 2. an interpreter the probes can run under.
# probes/oc/Makefile hardcodes ../../.venv/bin/python and
# probes/oc/Makefile hardcodes ../../netherite/blaze/core, so the venv and the
# netherite clone must both land at the payload root. That is not a style
# choice, it is what the Makefile reads.
def make_venv(system_site):
    args = [sys.executable, "-m", "venv"]
    if system_site:
        args.append("--system-site-packages")
    args.append(os.path.join(ROOT, ".venv"))
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=300)
    except Exception as e:
        say(f"  venv        {' '.join(args[1:])} raised {type(e).__name__}: {e}")
        return None
    exe = os.path.join(ROOT, ".venv", "bin", "python")
    if p.returncode != 0 or not os.path.isfile(exe):
        # Kaggle's image has no python3-venv, so this is the expected path there;
        # the reason is printed because "it failed" is not a useful log line
        say(f"  venv        FAILED (rc={p.returncode}) "
            f"{' '.join(args[2:-1])}: {(p.stderr or p.stdout).strip()[-300:]}")
        return None
    return exe


def pip_install(exe, pkgs, timeout=900):
    cmd = [exe, "-m", "pip", "install", "--no-input", "-q"] + pkgs
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    return p.returncode == 0, (p.stdout + p.stderr)[-3000:]


for system_site in (True, False):
    exe = make_venv(system_site)
    if exe:
        say(f"  venv        {exe} (system_site_packages={system_site})")
        PY = exe
        break
if PY is None:
    # No venv module in the image (Kaggle's has no python3-venv). Two things
    # then have to be true: the packages must land in an interpreter that has
    # them, and .venv/bin/python must point at THAT interpreter, because
    # probes/oc/Makefile hardcodes ../../.venv/bin/python. A `python3 -m venv`
    # that died in ensurepip leaves a .venv directory with a python in it and no
    # packages, which is worse than nothing: make then runs that python and
    # every probe in the camera lane dies with ModuleNotFoundError: numpy.
    import shutil as _shutil
    PY = sys.executable
    _shutil.rmtree(os.path.join(ROOT, ".venv"), ignore_errors=True)
    say(f"  venv        unavailable; using the system interpreter {PY}")
    for extra in ([], ["--break-system-packages"], ["--user"]):
        ok, tail = pip_install(PY, extra + ["tinygrad", "numpy", "tabulate"])
        if ok:
            say(f"  pip install tinygrad numpy tabulate ok (with {extra or ['no flag']})")
            break
        say(f"  pip install {' '.join(extra)} failed: {tail[-200:]}")
    # the shim the Makefile reads
    vbin = os.path.join(ROOT, ".venv", "bin")
    try:
        os.makedirs(vbin, exist_ok=True)
        for nm in ("python", "python3"):
            dst = os.path.join(vbin, nm)
            if not os.path.exists(dst):
                os.symlink(PY, dst)
        say(f"  .venv       {vbin}/python -> {PY}  (shim, so the oc Makefile works)")
    except Exception as e:
        say(f"  .venv shim  FAILED ({type(e).__name__}: {e}) -- probes/oc will skip")
else:
    ok, tail = pip_install(PY, ["tinygrad", "numpy", "tabulate"])
    say(f"  base deps   {'ok' if ok else 'FAILED -- tinygrad/numpy missing means most probes skip'}")
    if not ok:
        say("  " + tail.replace("\\n", "\\n  ")[-2000:])

# ---- 3. the fla/triton comparison needs torch. On Kaggle's GPU image torch is
# already installed system-wide (visible through --system-site-packages), so
# this is usually just triton + fla. A failure here is NOT fatal: it only makes
# kda_vs_fla.py skippable.
if kind == "GPU":
    probe = subprocess.run([PY, "-c", "import torch;print(torch.__version__)"],
                           capture_output=True, text=True, timeout=300)
    if probe.returncode == 0:
        AVAIL["torch"] = probe.stdout.strip()
        say(f"  torch       {AVAIL['torch']} (preinstalled)")
    else:
        ok, tail = pip_install(PY, ["torch"])
        say(f"  torch       {'installed' if ok else 'FAILED'}")
        if ok:
            AVAIL["torch"] = sh(f"{PY} -c 'import torch;print(torch.__version__)'")
    for pkg in ("triton", "flash-linear-attention", "fla"):
        try:
            subprocess.run([PY, "-c", f"import {pkg}"], capture_output=True,
                           text=True, timeout=300)
        except Exception:
            pass
    ok, tail = pip_install(PY, ["triton", "flash-linear-attention"])
    say(f"  fla         {'ok' if ok else 'FAILED -- kda_vs_fla.py will be SKIPPED'}")
    if not ok:
        say("  " + tail.replace("\\n", "\\n  ")[-1500:])
    try:
        import importlib.metadata as _md
        for p in ("triton", "flash-linear-attention", "fla-core", "torch", "tinygrad", "numpy"):
            try:
                AVAIL[p] = _md.version(p)
            except Exception:
                pass
    except Exception:
        pass

with open(os.path.join(LOGD, "01-pip-freeze.txt"), "w") as f:
    f.write(sh(f"{PY} -m pip freeze 2>&1 | head -100"))
    f.write("\\n\\n# interpreter\\n")
    f.write(sh(f"{PY} -c 'import sys;print(sys.executable)'"))
say("  pip freeze written to logs/01-pip-freeze.txt")
FACTS["versions"] = ", ".join(f"{k} {v}" for k, v in sorted(AVAIL.items()))
FACTS["gpu"] = gpu_names or ""
say(f"  versions    {FACTS['versions']}")
say(f"  working tree {ROOT}  (outside {OUT}, so it is not in the output zip)")
'''

# --------------------------------------------------------------------------
# netherite + clang
# --------------------------------------------------------------------------
CELL_NETHERITE = '''
import os, subprocess

NETHERITE = os.path.join(ROOT, "netherite")
CORE = os.path.join(NETHERITE, "blaze", "core")

say("")
say("=" * 78)
say("camera C reference (netherite headers)")
say("=" * 78)

# The C reference includes netherite's own obs_camera.h, so blaze/ is not
# optional. netherite is public, so the notebook clones it rather than carrying
# it in the payload; a clone failure is a skip, not a failure.
if os.path.isdir(CORE):
    say(f"  headers     already present at {CORE}")
else:
    # absolute paths: the notebook's cwd is /kaggle/working, and the Makefile
    # looks for ../../netherite/blaze/core relative to probes/oc, i.e. at the
    # payload root, not at the cwd.
    r = sh(f"rm -rf {NETHERITE} && git clone --filter=blob:none --sparse --depth 1 "
           f"https://github.com/Infatoshi/netherite {NETHERITE} 2>&1 | tail -3", timeout=600)
    say(f"  clone       {r or 'ok'}")
    r2 = sh(f"cd {NETHERITE} && git sparse-checkout set blaze docs 2>&1 | tail -2", timeout=300)
    say(f"  sparse      {r2 or 'ok'}")
    if os.path.isdir(CORE):
        say(f"  headers     {CORE} ok")
    else:
        say("  headers     MISSING -- the camera C comparison will be SKIPPED")
        say("              (the TinyGrad camera port still runs)")

# clang is needed to build ref_oc. Kaggle's images are not guaranteed to have
# it; with enable_internet it can be installed, and if that fails the C
# comparison is skipped rather than failed.
# Note: `sh()` on a missing binary returns "sh: 1: clang: not found" -- non-empty
# output from a failed command. Testing for that text is how a run ends up
# claiming the camera C comparison is enabled when no compiler exists.
CLANG = sh_ok("clang --version")
CLANG = CLANG.splitlines()[0] if CLANG else ""
if not CLANG:
    say("  clang       not found, trying apt-get")
    ap = sh("apt-get update -qq 2>&1 | tail -3; "
            "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq clang 2>&1 | tail -5",
            timeout=900)
    say("  apt-get     " + (ap.replace("\\n", " / ")[:300] or "no output"))
    c = sh_ok("clang --version")
    CLANG = c.splitlines()[0] if c else ""
say(f"  clang       {CLANG or 'UNAVAILABLE'}")
HAVE_CAMERA_C = bool(CLANG) and os.path.isdir(CORE)
if not HAVE_CAMERA_C:
    say("  ==> camera C comparison SKIPPED (needs clang and blaze/core)")
else:
    say("  ==> camera C comparison ENABLED")

# ---- 4. a C compiler for TinyGrad's own CPU backend, decided HERE, after the
# clang install above. tinygrad's CPU device shells out to getenv("CC", "clang")
# (runtime/support/compiler_cpu.py) and passes CLANG-ONLY flags:
#   --target=x86_64-none-unknown-elf -march=native
# so gcc is NOT a substitute: with CC=gcc the CPU backend dies with
# CalledProcessError (seen on the third real GPU run, in divide_precision.py,
# which had been given CC=gcc precisely to avoid the missing clang).
# With no clang at all, probes that touch Device.CPU cannot run, and the honest
# thing is to say so rather than to point CC at something that fails differently.
_cc = sh_ok("command -v clang")
CC = _cc.splitlines()[0].strip() if _cc else ""
if CC:
    os.environ["CC"] = CC            # inherited by every probe subprocess
    say(f"  CC          {CC} (exported: tinygrad's CPU backend needs clang's flags)")
else:
    gcc = sh_ok("command -v gcc")
    say(f"  CC          UNSET. clang is absent; gcc is "
        f"{'present at ' + gcc if gcc else 'absent too'} but cannot stand in for it.")
    say("              probes that touch Device.CPU will fail; the CUDA ones are fine")
FACTS["cc"] = CC
say("")
'''

# --------------------------------------------------------------------------
# the battery
# --------------------------------------------------------------------------
CELL_BATTERY = '''
import json, os, subprocess, sys, time

LOGD = os.path.join(OUT, "logs")
RESULTS = []
# Cost discipline: the free tier is ~30 GPU-hours a week and these probes take
# minutes. Every item is bounded, and the whole battery stops launching new work
# past TOTAL_BUDGET_S so a pathological item cannot eat the session.
ITEM_TIMEOUT = __ITEM_TIMEOUT__
TOTAL_BUDGET_S = __BUDGET__
# Per-item overrides, baked in at BUILD time on purpose. `kaggle kernels push`
# does NOT forward the caller's environment into the notebook's runtime, so
# BATTERY_TIMEOUTS set on the shell that pushes has no effect on the runner --
# verified on the third real GPU run, where kda_vs_fla still timed out at 900s
# despite being given 1800s that way. Setting it when the notebooks are built is
# the only thing that works; the driver does that.
OVERRIDES = __OVERRIDES__
ONLY = __ONLY__
T0 = time.time()


def have(*rel):
    return all(os.path.exists(os.path.join(ROOT, p)) for p in rel)

def have_mod(m):
    if not PY:
        return False
    try:
        return subprocess.run([PY, "-c", f"import {m}"], capture_output=True,
                              timeout=180).returncode == 0
    except Exception:
        return False


BATTERY = [
    # key, cwd (relative to ROOT), argv, needs, note
    # Order is by value, not by convenience: if the run is cut short by the
    # session cap or the battery budget, what is lost is the least interesting.
    # kda_vs_fla is the comparison this whole project exists for, so it comes
    # before the two probes that merely characterise the port.
    ("preflight",   ".", [PY, "remote/preflight.py"], [], "is this box able to run the probes"),
    ("test_ref",    "kda", [PY, "test_ref.py"], [], "float64 reference, 17 tests"),
    ("test_tiny",   "kda", [PY, "test_tiny.py"], [], "TinyGrad port vs the reference"),
    ("codegen_diff", "kda", [PY, "codegen_diff.py"], [], "generated code vs hand-written Triton"),
    ("kda_vs_fla",  "kda", [PY, "kda_vs_fla.py"], ["torch", "fla"], "the N-vs-N comparison"),
    ("perf_kda",    "kda", [PY, "perf_kda.py"], [], "kernel count, GPU/wall split"),
    ("replay_kda",  "kda", [PY, "replay_kda.py"], [], "graph replay: the python tax"),
    ("cost",        "kda", [PY, "cost.py"], [], "analytic FLOP/byte model"),
    ("fp32_correctness", ".", [PY, "probes/fp32_correctness.py"], [], "is float32 correctly rounded"),
    ("divide_precision", ".", [PY, "probes/divide_precision.py"], [], "the a*(1/b) rewrite"),
    ("gather_floor", ".", [PY, "probes/gather_floor.py"], [], "O(source) floor on a gather"),
    ("gather_scale", ".", [PY, "probes/gather_scale.py"], [], "indexed load O(out) or O(src)"),
    ("real_gather", ".", [PY, "probes/real_gather.py"], [], "the custom_kernel attempt"),
]
# the camera needs a compiler and netherite's headers; without them it is skipped
# and the whole camera lane reports why. These are shell strings, not argv lists:
# subprocess.run(list, shell=True) runs the FIRST element and passes the rest to
# the shell as $0, $1 -- which silently reduces `["make", "check"]` to a bare
# `make`, i.e. the default goal, and reports success.
CAMERA_C = [
    ("oc_check",  "make check",  "camera vs the C reference, block-id agreement"),
    ("oc_report", "make report", "every number in notes/01"),
]


def record(key, verdict, secs, note, rc=None):
    RESULTS.append({"key": key, "verdict": verdict, "seconds": round(secs, 1),
                    "rc": rc, "note": note})
    say(f"  [{verdict:<7}] {key:<18} {secs:7.1f}s  {note}")


def run_one(key, cwd, argv, note, shell=False):
    if ONLY and key not in ONLY:
        record(key, "skip", 0.0, "not in BATTERY_ONLY")
        return
    log = os.path.join(LOGD, f"{key}.log")
    # per-item timeout override, so one slow probe does not make every probe
    # slow. Baked in by the builder; see OVERRIDES above.
    timeout_s = ITEM_TIMEOUT
    for kv in OVERRIDES.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            if k.strip() == key:
                timeout_s = int(v)
    left = TOTAL_BUDGET_S - (time.time() - T0)
    if left <= 0:
        record(key, "skip", 0.0, "out of the battery time budget")
        return
    cmd = argv if shell else [str(a) for a in argv]
    printable = cmd if isinstance(cmd, str) else " ".join(cmd)
    say("")
    say(f"--- {key}: {printable}  (cwd={cwd}, timeout={timeout_s}s) ---")
    t = time.time()
    try:
        p = subprocess.run(cmd, cwd=os.path.join(ROOT, cwd) if cwd != "." else ROOT,
                           capture_output=True, text=True, timeout=timeout_s,
                           shell=shell)
        out, rc = p.stdout + p.stderr, p.returncode
        verdict = "ok" if rc == 0 else f"rc={rc}"
    except subprocess.TimeoutExpired as e:
        # subprocess joins the RAW chunks it got from the selector, so even with
        # text=True the TimeoutExpired carries bytes. Concat first and decode
        # after and it raises TypeError -- which is what killed the first real
        # GPU run, 30 minutes in, and cost the whole output zip.
        def _s(x):
            if x is None:
                return ""
            return x.decode("utf-8", "replace") if isinstance(x, bytes) else str(x)
        out = _s(e.stdout) + _s(e.stderr)
        out += f"\\n<TIMED OUT after {timeout_s}s>"
        rc, verdict = None, "TIMEOUT"
    except FileNotFoundError as e:
        out, rc, verdict = f"<{e}>", None, "skip"
    except Exception as e:
        out, rc, verdict = f"<{type(e).__name__}: {e}>", None, "error"
    dt = time.time() - t
    with open(log, "w") as f:
        f.write(f"$ {printable}\\n# cwd {cwd}\\n\\n{out}\\n")
    tail = out.splitlines()[-45:]
    for ln in tail:
        say("    " + ln[:200])
    record(key, verdict, dt, note, rc)


say("")
say("=" * 78)
say("battery")
say("=" * 78)
if not PY or not os.path.isdir(ROOT):
    say("  no payload and no interpreter: nothing to run")
    say("  see logs/01-pip-freeze.txt and the cell output above")
else:
    # tinygrad is the one dependency everything needs. If it is missing, say so
    # once and skip the lot rather than producing 14 identical import errors.
    if not have_mod("tinygrad"):
        say("  tinygrad is not importable under {PY}: skipping the whole battery")
    else:
        for key, cwd, argv, needs, note in BATTERY:
            missing = [n for n in needs if not have_mod(n)]
            if missing:
                record(key, "skip", 0.0, f"missing {', '.join(missing)}")
                continue
            # argv paths are relative to the item's own cwd, which is relative
            # to ROOT -- both have to be applied to find the file
            gone = [a for a in argv[1:]
                    if str(a).endswith(".py") and not os.path.isfile(
                        os.path.join(ROOT, cwd, str(a)))]
            if gone:
                record(key, "skip", 0.0, f"payload has no {', '.join(gone)}")
                continue
            run_one(key, cwd, argv, note)
        for key, argv, note in CAMERA_C:
            if not HAVE_CAMERA_C:
                record(key, "skip", 0.0, "no clang and/or no netherite blaze/core")
                continue
            run_one(key, "probes/oc", argv, note, shell=True)
        # run_oc.py reads ref.bin and cells_*.bin unconditionally, and only the
        # C reference can produce those, so this is not a "no C reference" run
        # at all: it is a second pass over the same artefacts, with timings.
        if os.path.isfile(os.path.join(ROOT, "probes/oc/run_oc.py")):
            if os.path.isfile(os.path.join(ROOT, "probes/oc/ref.bin")):
                run_one("oc_port_only", "probes/oc",
                        [PY, "run_oc.py", "--bench", "8"],
                        "the TinyGrad camera port, timed, against the same ref")
            else:
                record("oc_port_only", "skip", 0.0,
                       "run_oc.py needs ref.bin, which only the C reference makes")
'''

# --------------------------------------------------------------------------
# codegen capture + summary
# --------------------------------------------------------------------------
CELL_SUMMARY = '''
import json, os, subprocess, sys, textwrap

say("")
say("=" * 78)
say("generated code")
say("=" * 78)

# Same capture the gcp script does: wrap the device compiler so every kernel
# TinyGrad emits for one KDA call is written out. This is the artefact the whole
# study is about, so it is worth a separate try/except.
CODEGEN = os.path.join(OUT, "codegen")   # where the captured kernels land
CAP = """import os, sys
sys.path.insert(0, "kda")
from tinygrad import Tensor, Device
import numpy as np
d = Device[Device.DEFAULT]
cap = []
orig = getattr(d.compiler, "compile_cached", None)
if orig is not None:
    def spy(src, *a, **k):
        cap.append(str(src)); return orig(src, *a, **k)
    d.compiler.compile_cached = spy
import test_ref as T, tiny_kda
q, k, v, g, beta = T.rnd(2, 128, 2, 2, 32, 32, seed=0, l2n=True)
tiny_kda.kda_chunk(q, k, v, g, beta, chunk_size=32)
if orig is not None:
    d.compiler.compile_cached = orig
os.makedirs("out/codegen", exist_ok=True)
for i, s in enumerate(cap):
    open(f"out/codegen/kernel{i:03d}.txt", "w").write(s)
print(f"captured {len(cap)} kernels from {Device.DEFAULT}")
"""
if PY and os.path.isdir(ROOT):
    p = subprocess.run([PY, "-c", CAP], cwd=ROOT, capture_output=True,
                       text=True, timeout=900)
    tail = (p.stdout + p.stderr).strip()
    say("  " + (tail[-1500:] if tail else f"<no output, rc={p.returncode}>"))
    # shutil.move, not os.replace: /tmp and /kaggle/working are different
    # filesystems, and os.replace across a mount point is EINVAL/EXDEV. That
    # killed the second real GPU run in its last cell, after the battery had
    # already run -- so the summary was never written either.
    src = os.path.join(ROOT, "out", "codegen")
    moved = 0
    try:
        if os.path.isdir(src):
            import shutil
            os.makedirs(CODEGEN, exist_ok=True)
            for f in sorted(os.listdir(src)):
                shutil.move(os.path.join(src, f), os.path.join(CODEGEN, f))
                moved += 1
            subprocess.run(["rm", "-rf", os.path.join(ROOT, "out")])
    except Exception as e:
        say(f"  moving the captured kernels failed: {type(e).__name__}: {e}")
    say(f"  {moved} kernel files moved into codegen/"
        + ("" if moved else "  (no compile_cached hook on this backend: "
                            "not every Device exposes device.compiler)"))

say("")
say("=" * 78)
say("nvidia-smi after")
say("=" * 78)
say("  " + sh("nvidia-smi --query-gpu=name,utilization.gpu,memory.used "
             "--format=csv,noheader").replace("\\n", "\\n  "))

# ---- the index that comes back in the zip
def _bucket(v):
    return v.split("=")[0] if "=" in v else v
counts = {}
for r in RESULTS:
    b = _bucket(r["verdict"])
    counts[b] = counts.get(b, 0) + 1
lines = [
    "# mc-cuda-tinygrad on Kaggle",
    "",
    f"- runner kind: **{kind}**",
    f"- accelerator: {FACTS.get('gpu') or 'see logs/00-environment.txt'}",
    f"- payload sha256: {FACTS.get('payload_sha', 'n/a')}",
    f"- versions: {FACTS.get('versions', '')}",
    f"- camera C reference: {'enabled' if HAVE_CAMERA_C else 'SKIPPED'}",
    f"- C compiler for tinygrad's CPU backend: {FACTS.get('cc') or 'NONE'}",
    f"- battery wall time: {time.time() - T0:.0f}s",
    "",
    "## results",
    "",
    "| item | verdict | seconds | note |",
    "|---|---|---|---|",
]
for r in RESULTS:
    lines.append(f"| {r['key']} | {r['verdict']} | {r['seconds']} | {r['note']} |")
lines += ["", "## counts", "", "```", json.dumps(counts, indent=2), "```", ""]
index = os.path.join(OUT, "SUMMARY.md")
open(index, "w").write("\\n".join(lines))
with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump({"kind": kind, "results": RESULTS, "counts": counts,
               "facts": FACTS}, f, indent=2, default=str)
say("")
say("=" * 78)
say("SUMMARY")
say("=" * 78)
for ln in lines:
    say(ln)
say(f"wrote {OUT}/SUMMARY.md and {OUT}/results.json")
'''

NOTEBOOK_MD = '''# mc-cuda-tinygrad on Kaggle

Runs this repo's probe battery on Kaggle's free accelerator and writes every
artefact into `/kaggle/working/`, which is what `kaggle kernels output` brings
back.

The notebook is the only file `kaggle kernels push` uploads, so the code it runs
arrives separately: as an attached Kaggle dataset
(`tenghuzhang/mc-cuda-tinygrad-payload`) mounted read-only at `/kaggle/input/`.
One tarball, one sha256, and per-file hashes in `MANIFEST.json` that this
notebook verifies before it runs anything.

Every step is independently guarded. A missing dependency, a missing file, a
failed pip, a probe that raises, a probe that hangs -- each is recorded as a
skip or a failure in `SUMMARY.md` and the battery continues. Nothing here can
take the run down.
'''

TPU_MD = '''# TPU lane: is it reachable, and what is on it

Reachability and device facts only. The TPU comparison design is separate; this
kernel exists so the lane is known to work before anything is built on it, and
it costs seconds of the ~20-30 TPU-hours a week.

No payload dataset is attached and nothing is installed: the answers come from
`jax`, `libtpu`, and the accelerator host devices that Kaggle exposes.
'''


def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _lines(src):
    """nbformat source arrays carry their own newlines. `kaggle kernels push`
    joins them with "".join(source), so a list whose elements lack the trailing
    "\\n" is silently concatenated into one unreadable line on the runner."""
    return src.strip("\n").splitlines(keepends=True)


def _subst(src):
    """Bake the build-time knobs into the notebook.

    They cannot be read from the environment at run time, because
    `kaggle kernels push` does not forward this shell's environment to the
    runner. The defaults here are the values remote/kaggle_run.sh uses, so a
    notebook built by hand still gets sane numbers.
    """
    return (src
            .replace("__ITEM_TIMEOUT__", os.environ.get("BATTERY_ITEM_TIMEOUT", "900"))
            .replace("__BUDGET__", os.environ.get("BATTERY_TOTAL_BUDGET_S", "3000"))
            .replace("__OVERRIDES__", json.dumps(
                os.environ.get("BATTERY_TIMEOUTS", "kda_vs_fla=1800")))
            .replace("__ONLY__", json.dumps(
                [s.strip() for s in os.environ.get("BATTERY_ONLY", "").split(",")
                 if s.strip()])))


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": _lines(_subst(src))}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(src)}


def write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
    print(f"wrote {os.path.relpath(path, os.path.dirname(HERE))} "
          f"({os.path.getsize(path)/1024:.1f} KB)")


# ---------------- probe battery notebook (GPU or TPU, it detects which) ------
write(os.path.join(HERE, "probe", "probe.ipynb"), nb([
    md(NOTEBOOK_MD),
    code(CELL_DEVICE),
    code(CELL_PAYLOAD),
    code(CELL_NETHERITE),
    code(CELL_BATTERY),
    code(CELL_SUMMARY),
]))

# ---------------- TPU reachability notebook ---------------------------------
# Two cells only: detect the machine, then report the TPU facts. It deliberately
# does not share the battery's payload cell -- it attaches no dataset, so there
# is nothing to unpack and nothing to install.
write(os.path.join(HERE, "probe-tpu", "tpu-facts.ipynb"), nb([
    md(TPU_MD),
    code(CELL_DEVICE),
    code('''
import os, shutil, subprocess, sys

OUT = "/kaggle/working"
LOGD = os.path.join(OUT, "logs")
FACTS = {}
FACTS["kind"] = "TPU"
FACTS["gpu"] = ""
FACTS["payload_sha"] = "n/a (none attached)"
FACTS["versions"] = ""

say("")
say("=" * 78)
say("TPU device facts")
say("=" * 78)

# ---- the host side
for label, cmd in (
    ("accel devices", "ls -l /dev/accel* 2>&1 | head"),
    ("libtpu", "ls -1 /usr/lib/libtpu* /usr/lib/x86_64-linux-gnu/libtpu* 2>&1 | head"),
    ("TPU env", "env | grep -iE '^(TPU|PJRT|XLA)' | sort"),
    ("nvidia (should be empty)", "nvidia-smi -L 2>&1 | head -2"),
):
    say(f"  {label:<26} " + sh(cmd, timeout=120).replace("\\n", "\\n  " + " " * 28))

# ---- jax: the question is not "is there a TPU" but "does jax see it"
try:
    import jax
    devs = jax.devices()
    FACTS["versions"] = f"jax {jax.__version__}"
    say(f"  {'jax':<26} {jax.__version__}")
    say(f"  {'jax.default_backend':<26} {jax.default_backend()}")
    say(f"  {'device count':<26} {jax.device_count()}")
    for d in devs:
        say(f"  {'device':<26} kind={d.device_kind} platform={d.platform} "
            f"id={getattr(d, 'id', '?')} process={getattr(d, 'process_index', '?')}")
        try:
            ms = d.memory_stats() or {}
            hbm = ms.get("bytes_limit") or ms.get("bytes_in_use")
            if hbm:
                say(f"  {'  memory_stats':<26} {hbm/1e9:.1f} GB  {ms}")
            else:
                say(f"  {'  memory_stats':<26} {ms or '(empty)'}")
        except Exception as e:
            say(f"  {'  memory_stats':<26} unsupported ({type(e).__name__}: {e})")
    FACTS["jax_devices"] = [f"{d.device_kind} x{len(devs)}" for d in devs][:1]
    FACTS["jax_sees_tpu"] = any(d.platform == "tpu" for d in devs)
    say(f"  ==> jax sees the TPU: {FACTS['jax_sees_tpu']}")
except Exception as e:
    FACTS["jax_sees_tpu"] = False
    say(f"  jax FAILED: {type(e).__name__}: {e}")

# ---- tensorflow is the other way in, and what the TPU plugin reports
try:
    import tensorflow as tf
    say(f"  {'tensorflow':<26} {tf.__version__}")
    tpus = tf.config.list_logical_devices("TPU")
    say(f"  {'tf logical TPUs':<26} {tpus}")
    res = tf.distribute.cluster_resolver.TPUClusterResolver()
    say(f"  {'tf cluster':<26} {res.cluster_spec().as_dict() if hasattr(res.cluster_spec(), 'as_dict') else res.cluster_spec()}")
    FACTS["tf_tpus"] = len(tpus)
except Exception as e:
    say(f"  tensorflow FAILED: {type(e).__name__}: {e}")

# ---- tinygrad has a TPU backend; the study cares whether it is reachable
try:
    from tinygrad import Device
    say(f"  {'tinygrad devices':<26} {[d for d in Device._devices if d]}")
    say(f"  {'tinygrad DEFAULT':<26} {Device.DEFAULT}")
    FACTS["tinygrad_default"] = Device.DEFAULT
except Exception as e:
    say(f"  tinygrad not importable ({type(e).__name__}: {e}) -- "
        f"pip install tinygrad to probe the TPU backend")

FACTS["versions"] = FACTS.get("versions", "")
lines = [
    "# TPU lane, Kaggle",
    "",
    f"- jax sees a TPU: **{FACTS.get('jax_sees_tpu')}**",
    f"- devices: {FACTS.get('jax_devices')}",
    f"- jax: {FACTS.get('versions')}",
    f"- tinygrad DEFAULT: {FACTS.get('tinygrad_default')}",
    "",
    "```",
    sh("ls -l /dev/accel* 2>&1"),
    "```",
]
open(os.path.join(OUT, "SUMMARY.md"), "w").write("\\n".join(lines))
import json
json.dump(FACTS, open(os.path.join(OUT, "facts.json"), "w"), indent=2, default=str)
for ln in lines:
    say(ln)
'''),
]))
print("done")
