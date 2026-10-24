"""runner_nb.py -- the notebook gpush generates and pushes.

This is the code that runs on the Kaggle runner. It is generated rather than
hand-written so that it can carry the tree it was built for: the fingerprint is
baked into cell 1, which re-derives it from the files that actually arrived.

THE FIRST CELL IS THE POINT
    The lost run in DECISIONS.md #18 recorded `payload sha256: n/a`, because
    the notebook had nothing to compare against: an empty /kaggle/input and a
    correct one produced identical output, and 15 probes recorded `skip`.

    So cell 1 reads the attestation that shipped inside the archive, re-hashes
    every file in the payload, and writes an explicit verdict. If the dataset
    did not mount, or mounted a stale version, the run says so in the first
    lines of its output instead of 15 lines further down where it reads like a
    legitimate result.

    It also exits non-zero on a mismatch. A run that cannot prove its own
    payload should not be recorded as a completed measurement.

CELLS
    1  resolve the mount, verify the payload against the attestation, report
       the device, and write the resolved tree root for cell 2
    2  run the command in that tree root, streaming stdout
    3  copy the tree into /kaggle/working so `kernels output` returns it

TEMPLATING
    Substitution is by sentinel token and str.replace, never by `%` or `.format`
    or Template. The cells are real Python that itself uses `%` formatting and
    contains `{}` braces and a literal `"$"`; every one of those template
    languages collides with the cell text, and the failure is silent -- you get
    a notebook that parses and does the wrong thing. `_fill` refuses to leave a
    token unreplaced, so that failure cannot ship.
"""
from __future__ import annotations

import ast
import json

from .. import __version__
from ..probes import DEVICE_PROBE

VERIFY_CELL = r'''
import glob, hashlib, json, os, shutil, subprocess, sys, tarfile

PAYLOAD_FINGERPRINT = @@FINGERPRINT@@
WANT = @@FILES@@

def find_payload():
    """/kaggle/input has changed shape between runner versions: it has been both
    /kaggle/input/<slug>/ and /kaggle/input/datasets/<owner>/<slug>/. Search for
    the archive rather than assume a path."""
    cands = sorted(glob.glob("/kaggle/input/**/payload.tgz", recursive=True))
    for c in cands:
        d = os.path.dirname(c)
        if WANT and all(os.path.isfile(os.path.join(d, p)) for p in list(WANT)[:5]):
            return c, d
    return (cands[0], os.path.dirname(cands[0])) if cands else (None, None)

tgz, mount = find_payload()
print("=" * 70)
print("GPUSH runner @@VERSION@@")
print("=" * 70)
print("archive     ", tgz or "NOT FOUND")
print("mount       ", mount or "NOT FOUND")

# ---- device, always, before any work -------------------------------
print()
print("--- device ---")
print(subprocess.run([sys.executable, "-c", @@PROBE@@],
                     capture_output=True, text=True).stdout.strip())

os.makedirs("/kaggle/working", exist_ok=True)
with open("/kaggle/working/gpush-root.txt", "w") as f:
    f.write("")

if tgz is None:
    print()
    print("VERDICT: NO PAYLOAD. The dataset did not mount at all.")
    print("This is the DECISIONS.md #18 condition: the dataset version was not")
    print("ready when the kernel was pushed. Nothing was measured.")
    with open("/kaggle/working/GPUSH-VERIFY.json", "w") as f:
        json.dump({"ok": False, "reason": "no payload.tgz under /kaggle/input",
                   "fingerprint": PAYLOAD_FINGERPRINT}, f, indent=2)
    raise SystemExit(3)

# Kaggle mounts a dataset as its constituent FILES. payload.tgz is one of those
# files; nothing unpacks it for us. So the runner unpacks it itself, into
# /kaggle/working, and that directory becomes the tree root.
#
# This is also the better place to verify. Hashing the archive proves the
# uploaded bytes; hashing the EXTRACTED tree proves the bytes that are about to
# be executed, which is the claim that matters.
TREE = "/kaggle/working/tree"
if os.path.isdir(TREE):
    shutil.rmtree(TREE)
os.makedirs(TREE)
try:
    with tarfile.open(tgz, "r:gz") as tf:
        tf.extractall(TREE, filter="data")
except Exception as e:
    print("could not unpack the payload archive:", e)
    with open("/kaggle/working/GPUSH-VERIFY.json", "w") as f:
        json.dump({"ok": False, "reason": "unpackable archive: %s" % e,
                   "fingerprint": PAYLOAD_FINGERPRINT}, f, indent=2)
    raise SystemExit(4)

TREE_ROOT = os.path.join(TREE, "payload")
if not os.path.isdir(TREE_ROOT):
    print("the archive unpacked but contains no payload/ directory")
    with open("/kaggle/working/GPUSH-VERIFY.json", "w") as f:
        json.dump({"ok": False, "reason": "archive has no payload/ prefix",
                   "top": sorted(os.listdir(TREE))}, f, indent=2)
    raise SystemExit(4)
print("tree root   ", TREE_ROOT)
with open("/kaggle/working/gpush-root.txt", "w") as f:
    f.write(TREE_ROOT)

# ---- verify the tree we are about to execute ----------------------
got = {}
for dirpath, _dirs, files in os.walk(TREE_ROOT):
    for fn in files:
        p = os.path.join(dirpath, fn)
        rel = os.path.relpath(p, TREE_ROOT)
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        got[rel] = h.hexdigest()

missing = sorted(set(WANT) - set(got))
extra = sorted(set(got) - set(WANT))
differing = sorted(p for p in set(WANT) & set(got) if WANT[p] != got[p])
ok = not (missing or extra or differing)

print()
print("--- payload integrity ---")
print("expected %d files, the tree we will execute has %d" % (len(WANT), len(got)))
if missing:
    print("MISSING  %d: %s" % (len(missing), ", ".join(missing[:6])))
if extra:
    print("EXTRA    %d: %s" % (len(extra), ", ".join(extra[:6])))
if differing:
    print("DIFFER   %d: %s" % (len(differing), ", ".join(differing[:6])))
print("VERDICT:", "payload matches the fingerprint that was pushed"
      if ok else "PAYLOAD DOES NOT MATCH -- the numbers below would be a lie")

with open("/kaggle/working/GPUSH-VERIFY.json", "w") as f:
    json.dump({"ok": ok, "fingerprint": PAYLOAD_FINGERPRINT,
               "tree_root": TREE_ROOT,
               "missing": missing, "extra": extra, "differing": differing,
               "files_in_tree": len(got)}, f, indent=2)

if not ok:
    raise SystemExit(5)
print()
'''

RUN_CELL = r'''
import os, subprocess, sys, time

WORK = "/kaggle/working"
cmd = @@CMD@@

# The cwd is whatever cell 1 resolved, not a path assumed at push time. The two
# disagreeing is exactly how a command silently runs against the wrong tree, so
# they cannot be written independently.
try:
    with open(os.path.join(WORK, "gpush-root.txt")) as f:
        TREE_ROOT = f.read().strip()
except OSError:
    TREE_ROOT = ""
if not TREE_ROOT or not os.path.isdir(TREE_ROOT):
    print("TREE_ROOT is %r; refusing to run against an unknown tree" % TREE_ROOT)
    raise SystemExit(6)

print("=" * 70)
print("$", " ".join(cmd))
print("  cwd   %s" % TREE_ROOT)
print("  at    %s" % time.strftime("%Y-%m-%d %H:%M:%S %Z"))
print("=" * 70)

t0 = time.time()
proc = subprocess.run(cmd, cwd=TREE_ROOT, capture_output=True, text=True)
elapsed = time.time() - t0

sys.stdout.write(proc.stdout)
if proc.stderr:
    sys.stderr.write(proc.stderr)
sys.stdout.write("\n")
print("=" * 70)
print("exit %d after %.1f s" % (proc.returncode, elapsed))

with open(os.path.join(WORK, "gpush-run.txt"), "w") as f:
    f.write("$ " + " ".join(cmd) + "\n")
    f.write("cwd: %s\n" % TREE_ROOT)
    f.write("exit: %d\nwall: %.1f s\n" % (proc.returncode, elapsed))
    f.write("\n--- stdout ---\n" + proc.stdout)
    if proc.stderr:
        f.write("\n--- stderr ---\n" + proc.stderr)
'''

PULL_CELL = r'''
# Copy the tree into /kaggle/working, which is what `kaggle kernels output`
# returns. Anything the command wrote beside its own probe comes home with it,
# so a measurement that produces a file is shipped with that file rather than
# with a number pasted somewhere else.
import os, shutil

SRC = TREE_ROOT
DST = "/kaggle/working/artifacts"
if os.path.isdir(DST):
    shutil.rmtree(DST)
shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns(
    "__pycache__", "*.pyc", ".git", "*.so", "*.o", "node_modules"))
n = sum(len(f) for _, _, f in os.walk(DST))
print("collected %d files to %s" % (n, DST))
'''


def _fill(src: str, **vals) -> str:
    """Sentinel substitution, and a hard failure if any token survives."""
    out = src
    for k, v in vals.items():
        out = out.replace(f"@@{k}@@", v)
    left = [t for t in ("FINGERPRINT", "FILES", "VERSION", "PROBE", "CMD")
            if f"@@{t}@@" in out]
    if left:
        raise AssertionError(f"unsubstituted token(s) in the generated cell: {left}")
    return out


def _cell(src: str, cid: str) -> dict:
    # nbformat 4.5 requires a stable `id` on every cell. Without it the runner
    # prints a MissingIDFieldWarning on every execution and calls it a future
    # hard error -- so the cells carry deterministic ids derived from their
    # position, which is all nbformat actually wants.
    return {"cell_type": "code", "id": cid, "execution_count": None,
            "metadata": {}, "outputs": [], "source": src.strip("\n").splitlines(True)}


def _md(lines: list[str], cid: str) -> dict:
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": lines}


def build_runner_notebook(manifest, dataset: str, cmd: list[str], slug: str) -> dict:
    cells = [_md([
        "# gpush run\n\n",
        f"Backend `kaggle`, payload dataset `{dataset}`.\n\n",
        "The first cell resolves the mount as it actually is, unpacks the payload,\n",
        "verifies the unpacked tree against the attestation that was pushed with it,\n",
        "and reports the device. A run that got an empty or stale payload says so in\n",
        "its first lines instead of silently recording skips.\n",
    ], "intro")]
    cells.append(_cell(_fill(VERIFY_CELL,
                             FINGERPRINT=json.dumps(manifest.fingerprint),
                             FILES=json.dumps({e.path: e.sha256
                                               for e in manifest.sorted_entries()}),
                             # the template already carries the surrounding
                             # quotes for this one, so it is a bare literal;
                             # the others are substituted into expression
                             # position and so must be JSON.
                             VERSION=__version__,
                             PROBE=json.dumps(DEVICE_PROBE)), "verify"))
    cells.append(_cell(_fill(RUN_CELL, CMD=json.dumps(list(cmd))), "run"))
    cells.append(_cell(PULL_CELL, "pull"))
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "gpush": {"version": __version__, "fingerprint": manifest.fingerprint,
                      "dataset": dataset, "command": list(cmd)},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def validate(nb: dict) -> list[str]:
    """Check the notebook parses, every cell is valid Python, and ids exist.

    Cheap, offline, and it runs before the push. A notebook that is invalid
    Python wastes a session, and the payload's job is to make sure a session is
    never spent on a mistake that could have been caught on the Mac.
    """
    problems = []
    if nb.get("nbformat") != 4:
        problems.append(f"nbformat is {nb.get('nbformat')}, expected 4")
    minor = nb.get("nbformat_minor", 0)
    for i, c in enumerate(nb.get("cells", [])):
        if minor >= 5 and not c.get("id"):
            problems.append(f"cell {i} has no id (nbformat 4.5 requires one)")
        if c.get("cell_type") != "code":
            continue
        try:
            ast.parse("".join(c.get("source", [])))
        except SyntaxError as e:
            problems.append(f"cell {i} is not valid Python: line {e.lineno}: {e.msg}")
    return problems
