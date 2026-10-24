#!/usr/bin/env python3
"""local_dryrun.py -- run the Kaggle notebook's own code, on this machine.

WHY
    The notebook is the thing that decides whether a run is worth anything, and
    it cannot be tested without spending accelerator quota. Almost all of it is
    ordinary Python: unpack the payload, check the hashes, discover what exists,
    run each probe in a subprocess, log it, time out, skip, summarise. That part
    is testable here, and the only parts that are not are the CUDA/TPU device
    facts and the pip installs.

WHAT IT DOES
    * reads kernels/probe/probe.ipynb and execs its code cells in order, in one
      namespace, exactly as the notebook runner would
    * rewrites the two path literals the cells hardcode (/kaggle/working and
      /kaggle/input) to a temp dir, because macOS will not let an unprivileged
      process create /kaggle, and because writing to a real /kaggle/working
      would be a side effect nobody asked for
    * chdir's into that temp dir, so relative shell commands (the netherite
      clone) land in the temp dir and not in the repo
    * puts the real payload where the notebook looks for it, so the sha256 and
      MANIFEST checks are the real ones
    * symlinks the repo's .venv and the repo's netherite/ into the extracted
      payload, which is what those two would be on the runner, so the camera
      Makefile's hardcoded ../../.venv/bin/python resolves

WHAT IT DOES NOT PROVE
    that CUDA works, that fla installs, that the free TPU appears, or that
    `machine_shape` names are accepted. Only a real run proves those. The
    battery it runs here is this Mac's backend, so the numbers are not the
    numbers the study wants -- what is under test is the plumbing.

USAGE
    .venv/bin/python kernels/local_dryrun.py                     # everything
    .venv/bin/python kernels/local_dryrun.py --only preflight,oc_check
    .venv/bin/python kernels/local_dryrun.py --keep
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
NB = os.path.join(HERE, "probe", "probe.ipynb")
TGZ = os.path.join(HERE, "payload", "payload.tgz")
INPUT_SLUG = "mc-cuda-tinygrad-payload"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="BATTERY_ONLY value")
    ap.add_argument("--item-timeout", type=int, default=180)
    ap.add_argument("--budget", type=int, default=900)
    ap.add_argument("--keep", action="store_true", help="keep the temp dir")
    a = ap.parse_args()

    if not os.path.isfile(TGZ):
        print("payload.tgz missing; run: .venv/bin/python kernels/pack.py")
        return 1

    # The battery's knobs (BATTERY_ONLY, the timeouts) are BAKED INTO the
    # notebook by build_notebooks.py, because a remote kernel cannot read this
    # shell's environment. So --only has to go through the builder too, and the
    # full-battery notebook is rebuilt afterwards so the checked-in file is
    # never left narrowed.
    def build_with(env):
        e = {k: v for k, v in os.environ.items() if not k.startswith("BATTERY_")}
        e.update(env)
        subprocess.run([sys.executable, os.path.join(HERE, "build_notebooks.py")],
                       env=e, capture_output=True, text=True, timeout=120, check=True)
    if a.only:
        build_with({"BATTERY_ONLY": a.only,
                    "BATTERY_ITEM_TIMEOUT": str(a.item_timeout),
                    "BATTERY_TOTAL_BUDGET_S": str(a.budget)})

    with open(NB) as f:
        nb = json.load(f)
    raw = [("".join(c["source"]) if isinstance(c["source"], list) else c["source"])
           for c in nb["cells"] if c["cell_type"] == "code"]
    tmp = tempfile.mkdtemp(prefix="kaggle-dryrun-")
    working = os.path.join(tmp, "working")
    indir = os.path.join(tmp, "input", INPUT_SLUG)
    os.makedirs(working)
    os.makedirs(indir)
    # the payload exactly as Kaggle mounts the dataset
    shutil.copy(TGZ, indir)
    shutil.copy(os.path.join(HERE, "payload", "payload.sha256"), indir)

    def fix(src):
        # bare-substring, so single- and double-quoted occurrences both move
        return (src.replace("/kaggle/working", working)
                   .replace("/kaggle/input", os.path.join(tmp, "input")))

    cells = [fix(c) for c in raw]
    # the venv/pip cell would create a venv and hit the network. Skip it and put
    # the repo's .venv where the Makefile expects one, after the payload is
    # extracted (the symlink is made below, once cell 1 has run).
    venv_idx = [i for i, c in enumerate(cells) if "def make_venv" in c]
    payload_idx = venv_idx[0] if venv_idx else None
    hook = '''
# ---- stands in for the notebook's payload cell: same extraction, same hash
# checks, but no venv is built and nothing is installed. The two symlinks below
# are what the runner would have produced for itself.
import glob, hashlib, json, os, tarfile, tempfile

PARENT = os.path.join(tempfile.gettempdir(), "mc-cuda-tinygrad")
ROOT = os.path.join(PARENT, "probe")
os.makedirs(PARENT, exist_ok=True)
tars = [d for d in sorted(os.listdir(INPUT))
        if os.path.isfile(os.path.join(INPUT, d, "payload.tgz"))]
tgz = os.path.join(INPUT, tars[0], "payload.tgz")
got = hashlib.sha256(open(tgz, "rb").read()).hexdigest()
want = open(os.path.join(INPUT, tars[0], "payload.sha256")).read().split()[0]
FACTS["payload_sha"] = got
say(f"  sha256      {got} {'matches' if want == got else 'MISMATCH'}")
with tarfile.open(tgz, "r:gz") as tf:
    try:
        tf.extractall(PARENT, filter="data")
    except TypeError:
        tf.extractall(PARENT)
man = json.load(open(os.path.join(ROOT, "MANIFEST.json")))
bad = [e["path"] for e in man["files"]
       if hashlib.sha256(open(os.path.join(os.path.dirname(ROOT), e["path"]), "rb").read()).hexdigest() != e["sha256"]]
say(f"  extracted   {len(man['files'])} files, {len(bad)} hash mismatches {bad}")
PY = %r
AVAIL = {}
for rel, target, what in ((".venv", %r, ".venv"),
                          ("netherite", %r, "netherite")):
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        os.symlink(target, p)
        say(f"  {what:<12} symlinked to {target} (local dry run)")
FACTS["versions"] = "local dry run"
FACTS["gpu"] = ""
''' % (os.path.join(REPO, ".venv", "bin", "python"),
       os.path.join(REPO, ".venv"), os.path.join(REPO, "netherite"))
    if payload_idx is not None:
        cells = cells[:payload_idx] + [hook] + cells[payload_idx + 1:]

    os.environ["BATTERY_ONLY"] = a.only
    os.environ["BATTERY_ITEM_TIMEOUT"] = str(a.item_timeout)
    os.environ["BATTERY_TOTAL_BUDGET_S"] = str(a.budget)
    os.chdir(working)      # on Kaggle the notebook's cwd is /kaggle/working
    print(f"temp dir      {tmp}")
    print(f"cells         {len(cells)} (venv/pip cell replaced by a symlink hook)")
    print(f"BATTERY_ONLY  {a.only or '(everything)'}")
    print(f"item timeout  {a.item_timeout}s   total budget {a.budget}s")
    print("=" * 78)

    ns = {"__name__": "__main__"}
    rc = 0
    for i, src in enumerate(cells):
        head = src.strip().split("\n")[0][:66]
        print(f"\n########## cell {i}: {head}")
        try:
            exec(compile(src, f"cell{i}", "exec"), ns)
        except Exception as e:
            traceback.print_exc()
            print(f"########## cell {i} RAISED {type(e).__name__}: {e}")
            print("    (on Kaggle a raising cell stops the run, so this matters)")
            rc = 1

    print()
    print("=" * 78)
    print("what came back, as the output zip would contain it:")
    total = 0
    for root, _dirs, files in os.walk(working):
        for f in sorted(files):
            p = os.path.join(root, f)
            sz = os.path.getsize(p)
            total += sz
            print(f"  {os.path.relpath(p, working):<46} {sz/1024:8.1f} KB")
    print(f"  total {total/1024:.0f} KB   (the payload tree lives in "
          f"{os.path.join(tempfile.gettempdir(), 'mc-cuda-tinygrad')}, not here)")
    if rc:
        print(f"\nrc={rc}: a cell raised -- do not push this notebook")
    shutil.rmtree(os.path.join(tempfile.gettempdir(), "mc-cuda-tinygrad"),
                  ignore_errors=True)
    if a.only:
        # put the full battery back into the checked-in notebook
        try:
            build_with({})
            print("rebuilt kernels/probe/probe.ipynb with the full battery")
        except Exception as e:
            print(f"WARNING: could not restore the full-battery notebook: {e}")
    if a.keep:
        print(f"kept: {tmp}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
