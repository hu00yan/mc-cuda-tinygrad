#!/usr/bin/env python3
"""sync.py -- the push loop, made incremental.

WHAT THIS IS
    `kernels/pack.py` builds a tarball, `remote/kaggle_run.sh` publishes it as a
    Kaggle dataset version and then runs a kernel. That path is proven: the
    mine.cu baseline went end to end through it. What it is not is *incremental*.
    Every push repacks and re-uploads the whole payload, so a one-line edit in
    `kda/tiny_kda.py` costs a full upload, and nothing anywhere says what changed.

    This is the incrementality layer, and it is three files' worth of work
    sitting on top of those two, not a reimplementation of them. `kernels/pack.py`
    is imported for its INCLUDE list and its builder; `remote/kaggle_run.sh` is
    invoked (with --skip-pack) for build/push/poll/download; `kernels/build_notebooks.py`
    is invoked by that script, not by this one.

TWO MODES, AND THE DISTINCTION IS THE POINT
    --check   manifest, diff, print. ZERO network calls, ZERO credentials. Run
              it before every push; it works with no token on the machine.
    --sync    the same diff, then publish a new dataset version *only if the
              diff is non-empty*. No kernel push, no GPU time, no cost.
    --run     sync, then hand off to remote/kaggle_run.sh --skip-pack: build the
              notebooks, validate them, push the kernel, poll to a bounded
              timeout, pull the output back. One GPU session.

WHY rclone IS NOT THE ANSWER, restated because it will be asked again
    rclone has no Kaggle backend, and Kaggle exposes no SFTP, no WebDAV, no S3
    endpoint. So `rclone sync` against Kaggle would be a wrapper around exactly
    the API this repo already uses, with a second dependency and no ability to
    see inside a Kaggle dataset version. There is also no partial-upload
    primitive: a Kaggle dataset version is a full snapshot, so the honest
    description of what follows is the delta *decision*, the delta *report*, and
    a real cost of zero for a no-op -- not a delta transfer. Anything claiming
    otherwise is describing a file-level diff layered on a snapshot upload, and
    the diff is what you want for review, not for bytes on the wire.

THE FAILURE MODES THIS EXISTS FOR
    * a stale file left in a payload, which silently measures the wrong code
    * a race between "dataset version published" and "dataset mounted", which
      cost DECISIONS.md #18 an entire GPU session (15/15 battery items skipped,
      `payload sha256: n/a`)
    * a silent omission: a file in the working tree that no include rule covers
      and nothing reports
    * a push that 409s because a hand-edited kernel title was clobbered back to
      a non-slugging one by make_notebook.py
    * a no-op push that becomes an empty dataset version

STORAGE
    remote/last-pushed.json is the last-pushed manifest. It is in the repo, not
    in a dotfile under $HOME, so it is diffable and reviewable: one line per
    file, sorted, so a one-line edit is a one-line diff. It is the baseline, and
    every report is a diff against it.

    `dataset` is cross-checked against the live dataset version's own metadata
    (`kaggle datasets files <id> --format json`, and the published
    payload.tgz's sha256 via `kaggle datasets download`). Disagreements are
    REPORTED, never silently resolved in favour of either side.

NO LOCAL CONTAINERS. This file runs on the Mac. Anything that needs nvcc, ptxas
or SASS disassembly happens server-side on Kaggle and its artefacts come back
through `kaggle kernels output`. Nothing here requires a GPU, a CUDA toolchain,
or a container to be installed locally.

USAGE
    remote/sync.sh --check              # offline, no credentials
    remote/sync.sh --sync               # publish only if something changed
    remote/sync.sh --run                # sync + kernel push + poll + results
    remote/sync.sh --check --json       # machine-readable plan
    remote/sync.sh --status             # what is pushed right now (one API call)
"""
from __future__ import annotations

import argparse
import ast
import datetime as _dt
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# The interpreter floor, and it is not arbitrary. `kda/kda_vs_fla.py:135` is
#     d = Device[d_ := Device.DEFAULT]
# and a walrus in an unparenthesised subscript is 3.10+ syntax: macOS's
# /usr/bin/python3 (3.9.6) rejects it, and would report a perfectly good file as
# unparseable. Kaggle's runner is 3.12, so the payload is fine where it runs --
# but a guard that cannot parse the payload cannot check the payload, so refuse
# early and say why, rather than blaming the file.
MIN_PY = (3, 10)

PAYLOAD_DIR = os.path.join(REPO, "kernels", "payload")
LAST_PUSHED = os.path.join(HERE, "last-pushed.json")
OUT_SYNC = os.path.join(REPO, "out", "sync")
OUT_KAGGLE = os.path.join(REPO, "out", "kaggle")
KAGGLE = os.path.join(REPO, ".venv", "bin", "kaggle")
KAGGLE_RUN = os.path.join(HERE, "kaggle_run.sh")
KERNEL_DIRS = {"gpu": os.path.join(REPO, "kernels", "probe"),
               "tpu": os.path.join(REPO, "kernels", "probe-tpu")}

SCHEMA = "mc-cuda-tinygrad/payload-manifest@1"
PYVER = ".".join(str(x) for x in sys.version_info[:3])

# Files in kernels/payload/ that Kaggle consumes as dataset *metadata* rather
# than storing as dataset content. `kaggle datasets files` will never list
# them, so a name-set comparison that expects them is comparing against a lie.
# Verified against the live dataset on 2026-09-29: the published version lists
# payload.tgz and payload.sha256, and does not list dataset-metadata.json.
METADATA_ONLY = {"dataset-metadata.json"}

# Scratch stays inside the repo, under out/sync/.scratch/, and is removed when
# the call is done. Not $TMPDIR: a cross-check that downloads the published
# tarball should be able to leave the bytes behind for inspection when it fails,
# and nothing here is big enough to need a real temp filesystem.
SCRATCH = os.path.join(OUT_SYNC, ".scratch")

# Files pack.py's docstring says are excluded on purpose. Anything in a payload
# source directory that is neither packed nor on this list is an omission with
# no explanation, and is reported as such. Keep in step with pack.py's
# "WHAT IT LEAVES OUT, ON PURPOSE" section.
KNOWN_EXCLUSIONS = (
    ("__pycache__/", "bytecode, regenerated"),
    (".pyc", "bytecode, regenerated"),
    ("probes/oc/cells_", "23 MB of regenerable reference data (pack.py)"),
    ("probes/oc/ref.bin", "regenerable: the C reference rebuilds it"),
    ("probes/oc/ref_oc", "regenerable: make builds it"),
)

# Exit codes. Distinct on purpose: a shell pipeline that only looks at `!= 0`
# should not have to guess which of three refusals it hit.
EXIT_OK = 0            # clean, or a correctly-detected no-op
EXIT_GUARD = 1         # a guard refused; nothing was published
EXIT_NOAUTH = 2        # no credentials (network modes only)
EXIT_DIRTY = 3         # --fail-on-dirty, and the plan has pending changes
EXIT_TOOL = 4          # the kaggle CLI or kaggle_run.sh is missing
EXIT_RUN = 5           # the delegated kernel run did not complete


class Refuse(Exception):
    """A guard refused to let the push happen. Carries the exit code."""

    def __init__(self, message, code=EXIT_GUARD):
        super().__init__(message)
        self.code = code


class Unstable(Exception):
    """A file's bytes moved while they were being hashed."""


class Scratch:
    """A directory under out/sync/.scratch, removed unless KEEP=1."""

    def __init__(self, tag):
        self.path = os.path.join(SCRATCH, f"{tag}-{stamp()}-{os.getpid()}")
        self.keep = os.environ.get("KEEP") == "1"

    def __enter__(self):
        os.makedirs(self.path, exist_ok=True)
        return self.path

    def __exit__(self, *exc):
        if self.keep:
            say(f"  scratch     kept at {os.path.relpath(self.path, REPO)}")
        else:
            import shutil
            shutil.rmtree(self.path, ignore_errors=True)
        return False


# --------------------------------------------------------------------- basics


def say(*a):
    print(*a, flush=True)


def stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def human(n: int) -> str:
    for unit, div in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= div:
            return f"{n / div:.1f} {unit}"
    return f"{n} B"


def check_interpreter():
    if sys.version_info < MIN_PY:
        need = ".".join(str(x) for x in MIN_PY)
        got = ".".join(str(x) for x in sys.version_info[:3])
        raise Refuse(
            f"running under python {got}, and the payload needs >= {need}.\n"
            f"  kda/kda_vs_fla.py:135 uses a walrus in an unparenthesised\n"
            f"  subscript, which is {need}+ syntax. A {got} ast.parse() cannot\n"
            f"  read the payload, so the import guard would have nothing to\n"
            f"  check, and it would blame the file for it. macOS's\n"
            f"  /usr/bin/python3 is 3.9.\n"
            f"  use {REPO}/.venv/bin/python, or: SYNC_PYTHON=<path> sync.sh",
            EXIT_TOOL)


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stat_key(st) -> tuple:
    """Everything about a file that must not move between hash and upload."""
    return (st.st_size, st.st_mtime_ns, st.st_ino, st.st_dev, st.st_mode)


def load_pack():
    """Import kernels/pack.py for its INCLUDE list and its builder.

    Delegating rather than transcribing: a second copy of the include list in
    this file would be a second thing to forget to update, and the whole point
    of the omission guard is that the include set has exactly one owner.
    """
    path = os.path.join(REPO, "kernels", "pack.py")
    spec = importlib.util.spec_from_file_location("mcct_pack", path)
    if spec is None or spec.loader is None:
        raise Refuse(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:                                   # noqa: BLE001
        raise Refuse(f"importing kernels/pack.py failed: {type(e).__name__}: {e}")
    for attr in ("INCLUDE", "DATASET_ID", "build", "verify"):
        if not hasattr(mod, attr):
            raise Refuse(f"kernels/pack.py has no {attr}; this file is written "
                         f"against a different pack.py")
    return mod


# ------------------------------------------------------------------ manifests


def hash_stable(abs_path: str):
    """Hash one file and prove its bytes did not move while we read them.

    Two stat calls around one read. A file edited by an editor that writes in
    place, or a formatter landing mid-hash, produces a digest of a state that
    never existed. Shipping that is how a run gets attributed to a tree that
    was only momentarily true.
    """
    s0 = os.stat(abs_path)
    digest = sha256_of(abs_path)
    s1 = os.stat(abs_path)
    if stat_key(s0) != stat_key(s1):
        raise Unstable(abs_path)
    return digest, s1


def build_manifest(pack):
    """The current content manifest: arcname -> {src, sha256, bytes, mode}.

    `mode` is recorded because a chmod is real information about the tree, but
    it deliberately does NOT decide whether a push is needed: pack.py writes
    every tar member as 0o644 with mtime 0, so a permission bit cannot change
    the payload. Treating it as a change would turn `chmod` into an upload and
    then into an API refusal ("a version with identical files").
    """
    files: dict[str, dict] = {}
    stamps: dict[str, tuple] = {}
    unstable: list[str] = []
    for src, arc in pack.INCLUDE:
        abs_path = os.path.join(REPO, src)
        if not os.path.isfile(abs_path):
            raise Refuse(f"{src} is in pack.py's INCLUDE but not on disk. The "
                         f"notebook would import a file the payload cannot "
                         f"carry; fix the tree or the include list, do not push.")
        try:
            digest, st = hash_stable(abs_path)
        except Unstable:
            unstable.append(src)
            continue
        files[arc] = {"src": src, "sha256": digest, "bytes": st.st_size,
                      "mode": "0o%o" % (st.st_mode & 0o777)}
        stamps[src] = stat_key(st)
    if unstable:
        raise Refuse("refusing to push: these files changed bytes while they "
                     "were being hashed, so the manifest describes a tree "
                     "state that did not exist:\n"
                     + "".join(f"   {s}\n" for s in unstable)
                     + "  re-run when the working tree is quiet")
    return files, stamps


def drift_since(stamps) -> list[str]:
    """Files that moved AFTER they were hashed.

    The hash pass proves each file was stable during its own read. This pass
    proves the whole set was still stable when the pass ended -- otherwise the
    tarball pack.py is about to build can differ from the manifest that
    justified publishing it.
    """
    out = []
    for src, st in stamps.items():
        try:
            now = stat_key(os.stat(os.path.join(REPO, src)))
        except OSError:
            out.append(src + " (vanished)")
            continue
        if now != st:
            out.append(src)
    return out


def tree_id(files) -> str:
    """One digest naming the exact tree state a result is attributable to."""
    blob = json.dumps({a: f["sha256"] for a, f in sorted(files.items())},
                      sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def load_last_pushed():
    if not os.path.isfile(LAST_PUSHED):
        return None
    try:
        with open(LAST_PUSHED) as f:
            doc = json.load(f)
    except (OSError, ValueError) as e:
        raise Refuse(f"{LAST_PUSHED} is unreadable ({e}). It is the baseline "
                     f"every diff is taken against; fix or delete it "
                     f"deliberately, do not let it be regenerated by accident.")
    if doc.get("schema") != SCHEMA:
        raise Refuse(f"{LAST_PUSHED} has schema {doc.get('schema')!r}, this "
                     f"file writes {SCHEMA!r}")
    return doc


def write_last_pushed(doc):
    tmp = LAST_PUSHED + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, LAST_PUSHED)


def diff(old_files, new_files) -> dict:
    old_keys, new_keys = set(old_files), set(new_files)
    both = old_keys & new_keys
    return {
        "added": sorted(new_keys - old_keys),
        "deleted": sorted(old_keys - new_keys),
        "changed": sorted(a for a in both
                          if old_files[a]["sha256"] != new_files[a]["sha256"]),
        "mode_only": sorted(a for a in both
                            if old_files[a]["sha256"] == new_files[a]["sha256"]
                            and old_files[a].get("mode") != new_files[a].get("mode")),
        "unchanged": sorted(a for a in both
                            if old_files[a]["sha256"] == new_files[a]["sha256"]),
    }


# ---------------------------------------------------------- the omission sweep


def payload_source_dirs(pack) -> list[str]:
    """Every directory the payload draws from, i.e. its blast radius."""
    dirs = set()
    for src, _ in pack.INCLUDE:
        parts = src.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            dirs.add("/".join(parts[:i]))
    return sorted(dirs)


def known_exclusion(rel: str):
    for pat, why in KNOWN_EXCLUSIONS:
        if rel.startswith(pat) or rel.endswith(pat):
            return why
    return None


def sweep(pack, manifest) -> dict:
    """Files in the working tree, in a payload source directory, not in the payload.

    Three classes, kept apart on purpose:
      imported   -- a packed file imports or names it. FATAL. This is the
                    silent-omission case that produces a run with a file
                    missing that only fails deep inside a probe.
      unpacked   -- in a payload source directory, not packed, no reason given.
                    Reported, not fatal: most of these are legitimately
                    local-only (kda/NOTES.md, remote/gcp_run.sh).
      explained  -- pack.py documents leaving it out (cells_*.bin, ref_oc, ...).
    """
    packed_srcs = {f["src"] for f in manifest.values()}
    found: dict[str, str] = {}
    for srcdir in payload_source_dirs(pack):
        abs_dir = os.path.join(REPO, srcdir)
        for dirpath, dirnames, filenames in os.walk(abs_dir):
            dirnames[:] = [x for x in dirnames if x != "__pycache__"]
            for fn in sorted(filenames):
                abs_p = os.path.join(dirpath, fn)
                rel = os.path.relpath(abs_p, REPO)
                if rel in packed_srcs or os.path.islink(abs_p) or not os.path.isfile(abs_p):
                    continue
                found[rel] = known_exclusion(rel) or ""
    return found


# --------------------------------------------------- the import-closure guard


def _dotted(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _fold(node, env):
    """Statically fold the small path-expression grammar the probes use.

    Just enough to answer "which directory did this file put on sys.path":
    string literals, names, os.path.join/abspath/dirname, and str(). Anything
    else returns None and is reported as unresolved, because a guard that
    silently loses sight of a sys.path edit is worse than no guard.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.Call):
        fn = _dotted(node.func)
        args = [_fold(a, env) for a in node.args]
        if fn in ("os.path.join", "ntpath.join", "posixpath.join") and args and all(a is not None for a in args):
            return os.path.normpath(os.path.join(*args))
        if fn in ("os.path.abspath", "os.path.realpath") and len(args) == 1 and args[0] is not None:
            return os.path.normpath(args[0])
        if fn in ("os.path.dirname", "posixpath.dirname", "ntpath.dirname") and len(args) == 1 and args[0] is not None:
            return os.path.dirname(args[0])
        if fn == "str" and len(args) == 1 and args[0] is not None:
            return args[0]
    return None


def _scope_env(tree, node, abs_self):
    """Constant-ish assignments visible to `node`: module level plus its function.

    `__file__` must be seeded BEFORE folding, not after. remote/preflight.py
    does the whole thing in two statements:

        here = os.path.dirname(os.path.abspath(__file__))
        kda  = os.path.join(os.path.dirname(here), "kda")
        sys.path.insert(0, kda)

    With __file__ unknown at fold time the first statement collapses to "" and
    the search dir becomes remote/kda instead of kda/, and every import inside
    that file then reads as external. A guard that is wrong in the permissive
    direction is worse than no guard, because it reports "fine".
    """
    env: dict[str, str] = {"__file__": abs_self}
    order = []
    for parent in _parents(tree, node):
        for child in ast.iter_child_nodes(parent):
            if isinstance(child, ast.Assign) and len(child.targets) == 1 and isinstance(child.targets[0], ast.Name):
                order.append(child)
    for child in order:
        val = _fold(child.value, env)
        if val is not None:
            env[child.targets[0].id] = val
    return env


def _parents(tree, target):
    """The chain of scopes enclosing `target`, innermost last."""
    chain = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.iter_child_nodes(node):
                if child is target or any(g is target for g in ast.walk(child)):
                    chain.append(node)
                    break
    return chain


def _search_dirs(tree, abs_self):
    """The directories this file puts on sys.path, as repo-relative paths.

    Seeded with the file's own directory, because `python kda/test_ref.py` puts
    that on sys.path[0] whether the file says so or not.
    """
    here = os.path.dirname(abs_self)
    dirs = [here]
    unresolved = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _dotted(node.func) == "sys.path.insert"):
            continue
        if not node.args:
            continue
        val = _fold(node.args[-1], _scope_env(tree, node, abs_self))
        if val is None:
            unresolved += 1
            continue
        d = val if os.path.isabs(val) else os.path.normpath(os.path.join(here, val))
        if d not in dirs:
            dirs.append(d)
    return dirs, unresolved


def _find_module(dirs, name):
    for d in dirs:
        for cand in (os.path.join(d, name + ".py"),
                     os.path.join(d, name, "__init__.py")):
            if os.path.isfile(cand):
                return cand
    return None


PATHY_PY = re.compile(r"^[A-Za-z0-9_./-]+\.py$")


def import_closure(pack, manifest):
    """Every local module a packed file needs, and whether the payload has it.

    Only repo files count. `numpy`, `tinygrad`, `torch` and `fla` are pip
    installed on the runner by the notebook's own cell, so a name that
    resolves to nothing in the working tree is external and fine. A name that
    DOES resolve in the tree and is not in the payload is the failure: the
    notebook will unpack a payload that cannot import, and the probe will die
    with a ModuleNotFoundError on the GPU, minutes into a session.
    """
    packed_srcs = {f["src"] for f in manifest.values()}

    external, resolved, missing, unresolved_syspath, parse_errors = set(), [], [], 0, []
    for src in sorted(packed_srcs):
        if not src.endswith(".py"):
            continue
        abs_self = os.path.join(REPO, src)
        try:
            tree = ast.parse(open(abs_self, "rb").read(), src)
        except (OSError, SyntaxError) as e:
            parse_errors.append(f"{src}: {e} [python {PYVER}]")
            continue
        dirs, unresolved = _search_dirs(tree, abs_self)
        unresolved_syspath += unresolved

        wanted = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    wanted.add(a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                wanted.add(node.module.split(".")[0])
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and PATHY_PY.match(node.value):
                # a script path by string literal: subprocess.run([PY, "test_ref.py"])
                wanted.add(node.value)
        for name in sorted(wanted):
            if name.endswith(".py"):
                # a script named by a string literal, e.g. subprocess.run([PY,
                # "test_ref.py"]). Tried as a path in every search dir, then
                # as a bare basename, because a payload that names
                # "probes/bench.py" and a payload that names "bench.py" are
                # both legal and mean different things.
                hit = next((os.path.join(d, name) for d in dirs
                            if os.path.isfile(os.path.join(d, name))), None)
                if hit is None:
                    base = os.path.basename(name)
                    hit = next((os.path.join(d, base) for d in dirs
                                if os.path.isfile(os.path.join(d, base))), None)
            else:
                hit = _find_module(dirs, name)
            if hit is None:
                external.add(os.path.basename(name) if name.endswith(".py") else name)
                continue
            rel = os.path.relpath(hit, REPO)
            if rel in packed_srcs:
                resolved.append((src, rel))
            else:
                missing.append((src, name, rel))
    return {"external": sorted(external), "resolved": resolved,
            "missing": missing, "unresolved_syspath": unresolved_syspath,
            "parse_errors": parse_errors}


def makefile_refs(pack, manifest):
    """Relative paths a packed Makefile names, and where they land.

    A Makefile is not code and there is no parser for one here, so this scans
    for `./` and `../` tokens in assignments, prerequisites and recipe lines.
    That is a scan, not a build-graph analysis, and it is scoped to those two
    shapes on purpose: a bare or absolute token in a Makefile is usually a
    compiler, not a file, and guessing at those produces noise that teaches you
    to ignore the guard.

    The three legitimate kinds of reference, all reported rather than assumed:
      packed     resolves inside the payload, fine
      produced   the Makefile builds it (./ref_oc, ./ref.bin), fine
      runtime    escapes the payload root, so the notebook must provide it
                 (../../netherite/blaze/core is sparse-cloned, ../../.venv is
                 built); a reference that escapes to somewhere else is a file
                 the payload does not carry and `make` will fail on
    """
    RUNTIME_ESCAPES = ("netherite", ".venv", "out")
    packed = {f["src"] for f in manifest.values()}
    resolved, produced, runtime, missing = [], [], [], []
    for arc, f in sorted(manifest.items()):
        if not (arc.endswith("Makefile") or arc.endswith(".mk")):
            continue
        # Comments name paths too (probes/oc/Makefile line 1 points at
        # ../../notes/01-obs-camera.md) and a comment references nothing make
        # will read, so they are dropped before the scan rather than reported.
        text = "\n".join(ln for ln in
                         open(os.path.join(REPO, f["src"]), errors="replace")
                         .read().splitlines()
                         if not ln.lstrip().startswith("#"))
        here = os.path.dirname(f["src"])            # repo-relative, the real one
        for tok in re.findall(r"(?<![\w/$.])\.{1,2}/[^\s$#:;)]+", text):
            if "$(" in tok:
                continue
            # probes/oc/Makefile's `../..` IS the payload root: two levels up
            # from probes/oc lands on the directory the notebook extracts into,
            # which is where it puts netherite/ and .venv/. So the question is
            # not "did it escape" but "is what it lands on something the
            # notebook provides", judged on the first component after the root.
            target = os.path.normpath(os.path.join(here, tok))
            if target.startswith(".."):
                missing.append((f["src"], tok, target))
            elif target in packed:
                resolved.append((f["src"], tok, target))
            elif target.split("/")[0] in RUNTIME_ESCAPES:
                # netherite/ and .venv/ are at the payload root on the runner
                # whatever this host happens to have. A .venv that exists HERE
                # is not the one make will use THERE, so local existence must
                # not promote it to "resolved".
                runtime.append((f["src"], tok, target))
            elif os.path.isfile(os.path.join(REPO, target)) or \
                    os.path.isdir(os.path.join(REPO, target)):
                resolved.append((f["src"], tok, target))
            else:
                # not on disk and not packed: the Makefile builds it
                produced.append((f["src"], tok, target))
    return {"resolved": resolved, "produced": produced,
            "runtime": runtime, "missing": missing}


def notebook_script_refs(kdir: str):
    """Payload-relative script paths named by the notebook that will run.

    `kernels/build_notebooks.py` runs each probe as a subprocess by path, so
    the notebook -- not the probe -- is what actually demands a file. Scanned
    out of the built .ipynb if it is there; if it is not (offline --check on a
    clean tree), the import closure above is the whole of the guard and this
    returns empty.
    """
    if not kdir or not os.path.isdir(kdir):
        return []
    for fn in sorted(os.listdir(kdir)):
        if not fn.endswith(".ipynb"):
            continue
        try:
            nb = json.load(open(os.path.join(kdir, fn)))
        except (OSError, ValueError):
            return []
        found = set()
        for cell in nb.get("cells", []):
            for line in "".join(cell.get("source", [])).splitlines():
                line = line.strip()
                if line.startswith(("import ", "from ")):
                    try:
                        for node in ast.walk(ast.parse(line)):
                            if isinstance(node, ast.Import):
                                found.update(a.name.split(".")[0] for a in node.names)
                            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                                found.add(node.module.split(".")[0])
                    except SyntaxError:
                        pass
        return sorted(found)
    return []


# ------------------------------------------------------------- the Kaggle side


def kaggle(*args, timeout=600):
    if not (os.path.isfile(KAGGLE) and os.access(KAGGLE, os.X_OK)):
        raise Refuse(f"the kaggle CLI is not at {KAGGLE}\n"
                     f"  uv pip install --python .venv/bin/python kaggle",
                     EXIT_TOOL)
    return subprocess.run([KAGGLE, *args], capture_output=True, text=True,
                          timeout=timeout)


def have_credentials() -> bool:
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    if os.environ.get("KAGGLE_CONFIG_DIR") and os.path.isfile(
            os.path.join(os.environ["KAGGLE_CONFIG_DIR"], "kaggle.json")):
        return True
    return os.path.isfile(os.path.expanduser("~/.kaggle/access_token"))


def require_credentials():
    if have_credentials():
        return
    raise Refuse(
        "no Kaggle credentials, and this mode makes network calls.\n"
        "  remote/sync.sh --check   needs none, and is the mode to run first\n"
        "  get a token at https://www.kaggle.com/settings/api\n"
        "  then:  kaggle auth login     (browser OAuth, writes "
        "~/.kaggle/access_token)", EXIT_NOAUTH)


def remote_files(dataset_id):
    """The published version's own file list, straight from the API.

    This is the only place the remote's idea of the payload exists, and it is
    where a stale file shows up: a name the API still lists that the local pack
    no longer produces is a file that would be measured as if it were current.
    """
    p = kaggle("datasets", "files", dataset_id, "--format", "json")
    if p.returncode != 0:
        raise Refuse(f"kaggle datasets files {dataset_id} failed:\n"
                     f"  {(p.stderr or p.stdout).strip()[-600:]}")
    try:
        rows = json.loads(p.stdout or "[]")
    except ValueError as e:
        raise Refuse(f"kaggle datasets files returned unparseable JSON ({e}); "
                     f"cannot cross-check the published version")
    return {r["name"]: r for r in rows}


def remote_tarball_sha256(dataset_id):
    """Hash the published payload.tgz by downloading it. ~72 KB, no GPU time.

    Worth it: `datasets files` proves the file NAMES and SIZES, not the bytes,
    and the bytes are what the notebook verifies at run time. A same-length
    stale tarball is unlikely, and this is the one check that would notice.

    `kaggle datasets download` hands back a ZIP of the whole version rather
    than the loose files, and `-f payload.tgz` still arrives wrapped. Both
    shapes are handled: read the member straight out of the archive, and fall
    back to looking for an unpacked payload.tgz on disk.
    """
    import zipfile
    with Scratch("xcheck") as tmp:
        p = kaggle("datasets", "download", dataset_id, "-p", tmp, "--force", "--quiet")
        if p.returncode != 0:
            return None, f"download failed: {(p.stderr or p.stdout).strip()[-300:]}"
        for dirpath, _, filenames in os.walk(tmp):
            for fn in sorted(filenames):
                path = os.path.join(dirpath, fn)
                if fn == "payload.tgz":
                    return sha256_of(path), None
                if fn.endswith(".zip"):
                    with zipfile.ZipFile(path) as z:
                        if "payload.tgz" in z.namelist():
                            h = hashlib.sha256()
                            with z.open("payload.tgz") as fh:
                                for chunk in iter(lambda: fh.read(1 << 20), b""):
                                    h.update(chunk)
                            return h.hexdigest(), None
        return None, "payload.tgz not present in the download"


def wait_ready(dataset_id, seconds=300, interval=10):
    """Block until the dataset version is mountable.

    This is the race that cost DECISIONS.md #18 a whole session. A version that
    is still being created does not mount: the kernel starts, /kaggle/input is
    empty, and every probe records `payload has no X` as a skip. The failure
    reads exactly like a broken notebook. Only `ready` means the bytes are
    there, so sync does not return before it says ready.
    """
    deadline = time.time() + seconds
    while True:
        p = kaggle("datasets", "status", dataset_id)
        st = (p.stdout or "").strip().splitlines()[-1].strip() if (p.stdout or "").strip() else ""
        if st == "ready":
            return True
        if time.time() >= deadline:
            raise Refuse(f"dataset {dataset_id} was not ready after "
                         f"{seconds // 60}m (status {st or '<none>'}). A kernel "
                         f"pushed now would run with an empty /kaggle/input and "
                         f"skip every probe. Not pushing.")
        time.sleep(interval)


def cross_check(dataset_id, local_names, expect_tar_sha, verbose=True):
    """Compare the published version against the local payload and the baseline.

    Every disagreement is printed and none of them is resolved by picking a
    side. Stale remote files are fatal in --sync: a name the API still lists
    that the local pack does not produce is precisely the stale-payload case.
    """
    out = {"remote": None, "problems": [], "stale_remote": [], "sha_match": None}
    files = remote_files(dataset_id)
    remote_names = set(files)
    # dataset-metadata.json describes the dataset and is never stored as one of
    # its files, so it is dropped from the local side before comparing sets.
    local_names = [n for n in local_names if n not in METADATA_ONLY]
    out["remote"] = {n: files[n].get("size") for n in sorted(files)}
    out["stale_remote"] = sorted(remote_names - set(local_names))
    out["missing_remote"] = sorted(set(local_names) - remote_names)
    if out["stale_remote"]:
        out["problems"].append(
            "the published version contains files this pack does not produce: "
            + ", ".join(out["stale_remote"])
            + " -- a stale file in a payload is a measurement of the wrong code")
    if out["missing_remote"]:
        out["problems"].append(
            "the published version is missing files this pack produces: "
            + ", ".join(out["missing_remote"]))
    for name in sorted(set(local_names) & remote_names):
        try:
            loc = os.path.getsize(os.path.join(PAYLOAD_DIR, name))
        except OSError:
            continue
        if files[name].get("size") not in (None, loc):
            out["problems"].append(
                f"{name}: published size {files[name]['size']} B != local "
                f"{loc} B")
    if expect_tar_sha:
        got, err = remote_tarball_sha256(dataset_id)
        if err:
            out["problems"].append(f"could not verify the published tarball: {err}")
        else:
            out["sha_match"] = got == expect_tar_sha
            out["remote_tar_sha256"] = got
            if not out["sha_match"]:
                out["problems"].append(
                    f"published payload.tgz sha256 {got[:16]}... != the "
                    f"{expect_tar_sha[:16]}... this repo just packed -- the "
                    f"remote and this repo disagree about what is deployed; "
                    f"not choosing a side")
    if verbose:
        for p in out["problems"]:
            say(f"  CROSS-CHECK  {p}")
    return out


# ------------------------------------------------------------------ the plan


def make_plan(args, pack):
    files, stamps = build_manifest(pack)
    drift = drift_since(stamps)
    if drift:
        raise Refuse("refusing to push: these files moved after they were "
                     "hashed, so the tarball about to be built would not match "
                     "the manifest that justified publishing it:\n"
                     + "".join(f"   {s}\n" for s in drift))

    baseline = load_last_pushed()
    old_files = baseline["files"] if baseline else {}
    d = diff(old_files, files)

    sweep_res = sweep(pack, files)
    closure = import_closure(pack, files)
    mk = makefile_refs(pack, files)
    kdir = KERNEL_DIRS[args.variant] if args.variant in KERNEL_DIRS else KERNEL_DIRS["gpu"]
    nb_imports = notebook_script_refs(kdir)

    packed_base = {os.path.basename(a)[:-3] for a in files if a.endswith(".py")}
    packed_base |= {os.path.basename(a) for a in files}
    tree_modules = set()
    for srcdir in payload_source_dirs(pack):
        for dirpath, dirnames, filenames in os.walk(os.path.join(REPO, srcdir)):
            dirnames[:] = [x for x in dirnames if x != "__pycache__"]
            for fn in filenames:
                if fn.endswith(".py"):
                    tree_modules.add(fn[:-3])
    # The notebook does not import the payload, it shells out to it. The one
    # place it names payload modules is the codegen capture cell, which runs
    # `python -c` against ROOT. So a name is a problem only when the PAYLOAD
    # lacks it and the TREE has it: os, json and numpy are the runner's, and
    # treating them as missing is what a first cut of this guard gets wrong.
    nb_missing = sorted(m for m in nb_imports
                        if m in tree_modules and m not in packed_base)
    # A deletion whose source is still in the tree is not a deletion. It is an
    # omission that has been renamed, and it is the exact failure the brief calls
    # a stale file in a payload: the tree says the file is current, the payload
    # says it is not, and a measurement of the payload is a measurement of the
    # wrong code. Fatal, separately listed from a real deletion.
    deleted_still_present = [a for a in d["deleted"]
                             if os.path.isfile(os.path.join(REPO, old_files[a]["src"]))]
    deleted_gone = [a for a in d["deleted"] if a not in deleted_still_present]

    guards = []
    if closure["parse_errors"]:
        guards.append(("fail", "unparseable payload file: "
                       + "; ".join(closure["parse_errors"])))
    for importer, name, rel in closure["missing"]:
        guards.append(("fail", f"{importer} needs {name}, which is in the tree at "
                              f"{rel} but is NOT in the payload"))
    for m in nb_missing:
        guards.append(("fail", f"the notebook imports {m}, which the payload "
                              f"does not carry"))
    for mkf, tok, target in mk["missing"]:
        guards.append(("fail", f"{mkf} reads {tok}, which resolves to {target}: "
                              f"outside the payload, not one of the paths the "
                              f"notebook provides at run time, and not on disk"))
    if closure["unresolved_syspath"]:
        guards.append(("warn", f"{closure['unresolved_syspath']} sys.path.insert() "
                               f"call(s) could not be resolved statically, so the "
                               f"import guard had less to see than it wanted"))
    # The brief's silent-omission case: a file that is in the working tree, is
    # inside a directory the payload draws from, and is not in the baseline.
    # Reported, never fatal, because most such files are legitimately
    # local-only and a guard that fires on all of them teaches you to skip it.
    # The fatal part is the intersection with "something in the payload needs
    # it", which is the import closure above.
    unexplained = sorted(r for r, why in sweep_res.items() if not why)
    explained = sorted(r for r, why in sweep_res.items() if why)
    baseline_paths = {f["src"] for f in old_files.values()}
    not_in_baseline = [r for r in unexplained if r not in baseline_paths]
    if unexplained:
        guards.append(("warn", f"{len(unexplained)} file(s) in the payload "
                               f"source directories are not packed and pack.py "
                               f"does not say why"
                               + (f"; {len(not_in_baseline)} of them are also "
                                  f"absent from the last-pushed baseline"
                                  if not_in_baseline else "")))
    for a in deleted_still_present:
        guards.append(("fail", f"{a} left the payload but {old_files[a]['src']} is "
                               f"still in the working tree: an omission, not a "
                               f"deletion"))

    changed = bool(d["added"] or d["changed"] or d["deleted"])
    return {
        "pack": pack, "files": files, "baseline": baseline, "diff": d,
        "sweep": sweep_res, "closure": closure, "guards": guards, "makefile": mk,
        "changed": changed, "tree": tree_id(files),
        "raw_bytes": sum(f["bytes"] for f in files.values()),
        "no_baseline": baseline is None,
        "deleted_gone": deleted_gone, "deleted_still_present": deleted_still_present,
        "unexplained": unexplained, "not_in_baseline": not_in_baseline,
        "explained": explained,
        "kdir": kdir, "nb_imports": nb_imports,
    }


def report(plan, args):
    pack, files, d = plan["pack"], plan["files"], plan["diff"]
    base = plan["baseline"] or {}
    n_all = len(files)
    fails = [g for g in plan["guards"] if g[0] == "fail"]
    warns = [g for g in plan["guards"] if g[0] == "warn"]
    verb = {EXIT_DIRTY: "dirty"}.get(args.fail_on_dirty and EXIT_DIRTY or 0, "")

    say(f"=== sync: {os.path.basename(REPO)} payload -> {pack.DATASET_ID} ===")
    say(f"  mode        {'check (offline: no network, no credentials)' if args.check else args.mode}")
    say(f"  tree        {plan['tree'][:16]}")
    if base:
        say(f"  baseline    {base.get('tree', '?')[:16]}  "
            f"pushed {base.get('pushed_at', '?')}  ({len(base.get('files', {}))} files)")
    else:
        say("  baseline    NONE -- remote/last-pushed.json does not exist, so "
            "every file counts as new")
    say(f"  payload     {n_all} files, {human(plan['raw_bytes'])} of source")

    if plan["changed"]:
        say(f"  delta       {len(d['changed'])} changed  {len(d['added'])} added  "
            f"{len(d['deleted'])} deleted  {len(d['unchanged'])} unchanged")
    else:
        say(f"  delta       NO-OP: 0 changed  0 added  0 deleted  "
            f"{len(d['unchanged'])} unchanged")
    if d["mode_only"]:
        say(f"  mode-only   {len(d['mode_only'])} file(s) had a permission bit "
            f"changed. pack.py writes 0o644, so the payload bytes are identical "
            f"and this is not an upload.")
    if plan["nb_imports"]:
        say(f"  notebook    imports {len(plan['nb_imports'])} module name(s); "
            f"all resolved in the payload")

    # the report proper
    def line(tag, arc, f, extra=""):
        say(f"  {tag:9s} {arc:<40s} {f['bytes']:>8d} B  {f['sha256'][:12]}{extra}")

    if d["changed"]:
        say("")
        for a in d["changed"]:
            line("CHANGED", a, files[a], f"   <- was {base['files'][a]['sha256'][:12]}")
    if d["added"]:
        for a in d["added"]:
            line("ADDED", a, files[a])
    if d["deleted"]:
        say("")
        for a in plan["deleted_still_present"]:
            say(f"  {'DELETED!':9s} {a:<40s} {base['files'][a]['bytes']:>8d} B  "
                f"{base['files'][a]['sha256'][:12]}   "
                f"source {base['files'][a]['src']} IS STILL IN THE TREE")
        for a in plan["deleted_gone"]:
            say(f"  {'DELETED':9s} {a:<40s} {base['files'][a]['bytes']:>8d} B  "
                f"{base['files'][a]['sha256'][:12]}   "
                f"source {base['files'][a]['src']} removed from the tree")
    if not plan["changed"] and base:
        say(f"  unchanged   {len(d['unchanged'])} files skipped, sha256 identical "
            f"to remote/last-pushed.json")

    # guards
    say("")
    if fails:
        for _, g in fails:
            say(f"  GUARD       FAIL  {g}")
    else:
        say(f"  GUARD       import closure ok -- {len(plan['closure']['resolved'])} "
            f"local import(s) resolved inside the payload, "
            f"{len(plan['closure']['external'])} external (pip on the runner)")
    mk = plan.get("makefile") or {}
    if mk.get("resolved") or mk.get("produced") or mk.get("runtime"):
        say(f"  MAKEFILE    {len(mk.get('resolved', []))} reference(s) resolve to "
            f"packed files, {len(mk.get('produced', []))} are built by the "
            f"Makefile, {len(mk.get('runtime', []))} are provided at run time by "
            f"the notebook (netherite clone, venv)")
    for _, g in warns:
        say(f"  GUARD       warn  {g}")
    if plan["unexplained"]:
        show = plan["unexplained"][:8]
        say(f"  NOT PACKED  {len(plan['unexplained'])} file(s) in the payload "
            f"source directories, not in pack.py's INCLUDE, no reason recorded.")
        say(f"               None is imported by a packed file, so nothing is "
            f"broken; this is the list to read when a result surprises you.")
        for r in show:
            say(f"                 {r}")
        if len(plan["unexplained"]) > len(show):
            say(f"                 ... and {len(plan['unexplained']) - len(show)} "
                f"more (full list in the run record)")
    if plan["explained"]:
        say(f"  EXCLUDED    {len(plan['explained'])} file(s) pack.py documents "
            f"leaving out (cells_*.bin, ref.bin, ref_oc, __pycache__)")
    say("")
    return fails


# ------------------------------------------------------------------- publish


def publish(pack, plan, args):
    """Create the dataset version, and only if the diff says there is one."""
    require_credentials()
    t0 = time.time()
    say("=== publish ===")
    p = subprocess.run([sys.executable, os.path.join(REPO, "kernels", "pack.py")],
                       capture_output=True, text=True)
    for ln in (p.stdout + p.stderr).strip().splitlines():
        say(f"  {ln}")
    if p.returncode != 0:
        raise Refuse("kernels/pack.py failed; nothing was published")

    tgz = os.path.join(PAYLOAD_DIR, "payload.tgz")
    tar_sha = sha256_of(tgz)
    tar_bytes = os.path.getsize(tgz)
    say(f"  tarball     {human(tar_bytes)}  sha256 {tar_sha[:16]}")

    # pack.py is deterministic, so the manifest this push ships must be
    # provably the manifest that was just diffed. Read the tarball's own
    # MANIFEST.json back and compare before spending a network call.
    import tarfile
    with tarfile.open(tgz, "r:gz") as tf:
        man = json.load(tf.extractfile("probe/MANIFEST.json"))
    packed = {e["path"]: e for e in man["files"]}
    if set(packed) != set(plan["files"]):
        only_tar = sorted(set(packed) - set(plan["files"]))
        only_plan = sorted(set(plan["files"]) - set(packed))
        raise Refuse(f"the tarball and the diff disagree on the file set.\n"
                     f"  only in the tarball: {only_tar}\n"
                     f"  only in the plan:     {only_plan}")
    bad = [a for a, e in packed.items() if e["sha256"] != plan["files"][a]["sha256"]]
    if bad:
        raise Refuse(f"the tarball and the diff disagree on the bytes of: {bad}. "
                     f"Refusing to publish a payload that does not match the "
                     f"manifest that was reviewed.")

    out = p = kaggle("datasets", "version", "-p", PAYLOAD_DIR,
                     "-m", f"payload {stamp()} tree {plan['tree'][:12]}")
    log = os.path.join(OUT_SYNC, f"dataset-version-{stamp()}.log")
    os.makedirs(OUT_SYNC, exist_ok=True)
    with open(log, "w") as f:
        f.write(out.stdout + out.stderr)
    if out.returncode != 0:
        if "identical" in (out.stdout + out.stderr).lower():
            say("  the API refused the version as identical to the published one")
            raise Refuse("pack.py produced bytes the API calls identical, yet the "
                         "manifest said something changed. That means the diff "
                         "and the payload disagree; nothing was published.")
        c = kaggle("datasets", "create", "-p", PAYLOAD_DIR)
        if c.returncode != 0:
            raise Refuse(f"could not publish the payload dataset:\n  "
                         f"{(out.stdout + out.stderr).strip()[-400:]}\n  "
                         f"{(c.stdout + c.stderr).strip()[-400:]}")
        say("  dataset     created")
    else:
        say("  dataset     version published")

    say("  waiting     for the version to be mountable (this is the race that "
        "cost DECISIONS.md #18 a session)")
    wait_ready(pack.DATASET_ID, seconds=args.ready_timeout)
    say("  dataset     ready")
    say(f"  upload      {time.time() - t0:.1f}s")

    # Cross-check against what THIS push just produced, not against the old
    # baseline: after `datasets version` the remote is supposed to hold these
    # bytes, and comparing to the baseline would report the success as a
    # disagreement. The baseline's own agreement is the no-op path's job.
    local_names = sorted(os.listdir(PAYLOAD_DIR))
    xc = cross_check(pack.DATASET_ID, local_names, tar_sha)
    if xc["stale_remote"]:
        raise Refuse("after publishing, the published version still lists files "
                     "this pack does not produce: " + ", ".join(xc["stale_remote"]))
    if xc["missing_remote"]:
        raise Refuse("after publishing, the published version is missing: "
                     + ", ".join(xc["missing_remote"]))
    if xc["problems"]:
        raise Refuse("published, but it does not match the payload:\n"
                     + "".join(f"   {x}\n" for x in xc["problems"]))
    say(f"  cross-check published version matches: {len(local_names)} files, "
        f"tarball sha256 verified against the live dataset")

    doc = {
        "schema": SCHEMA,
        "dataset": pack.DATASET_ID,
        "pushed_at": now_iso(),
        "tree": plan["tree"],
        "tarball": {"path": "payload.tgz", "bytes": tar_bytes, "sha256": tar_sha},
        "dataset_version_files": xc["remote"],
        "delta": {k: len(plan["diff"][k]) for k in ("added", "changed", "deleted", "unchanged")},
        "sync_seconds": round(time.time() - t0, 1),
        "note": "written by remote/sync.py after a successful dataset version; "
                "the baseline every diff is taken against",
        "files": plan["files"],
    }
    write_last_pushed(doc)
    say(f"  baseline    written to {os.path.relpath(LAST_PUSHED, REPO)} "
        f"({len(plan['files'])} files)")
    return doc


def adopt_remote(pack, plan, args):
    """Seed the baseline from what is actually published, without uploading.

    For the case where remote/last-pushed.json was lost, or after someone
    published by hand. The payload's own MANIFEST.json is inside the published
    tarball, so this is a re-derivation from the remote bytes rather than a
    transcription of anything.
    """
    require_credentials()
    with Scratch("adopt") as tmp:
        p = kaggle("datasets", "download", pack.DATASET_ID, "-p", tmp, "--force")
        if p.returncode != 0:
            raise Refuse(f"download failed:\n  {(p.stderr or p.stdout).strip()[-400:]}")
        # Same ZIP-wrapping shape as remote_tarball_sha256; see the comment there.
        import tarfile
        import zipfile
        man, tar_bytes, tar_sha = None, 0, ""
        for dirpath, _, filenames in os.walk(tmp):
            for fn in sorted(filenames):
                path = os.path.join(dirpath, fn)
                if fn == "payload.tgz":
                    tar_bytes, tar_sha = os.path.getsize(path), sha256_of(path)
                    with tarfile.open(path, "r:gz") as tf:
                        man = json.load(tf.extractfile("probe/MANIFEST.json"))
                elif fn.endswith(".zip"):
                    with zipfile.ZipFile(path) as z:
                        if "payload.tgz" not in z.namelist():
                            continue
                        blob = z.read("payload.tgz")
                        tar_bytes, tar_sha = len(blob), hashlib.sha256(blob).hexdigest()
                        import io
                        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
                            man = json.load(tf.extractfile("probe/MANIFEST.json"))
                if man:
                    break
            if man:
                break
        if man is None:
            raise Refuse("the published version has no readable payload.tgz")
    arc2src = {arc: src for src, arc in pack.INCLUDE}
    files = {}
    for e in man["files"]:
        src = arc2src.get(e["path"])
        if src is None:
            raise Refuse(f"the published payload has {e['path']}, which is not in "
                         f"pack.py's INCLUDE. Adopting it would create a "
                         f"permanent unexplained deletion.")
        files[e["path"]] = {"src": src, "sha256": e["sha256"],
                            "bytes": e["bytes"], "mode": "0o644"}
    doc = {
        "schema": SCHEMA, "dataset": pack.DATASET_ID, "pushed_at": now_iso(),
        "tree": tree_id(files),
        "tarball": {"path": "payload.tgz", "bytes": tar_bytes,
                    "sha256": tar_sha},
        "delta": {"adopted": True, "added": 0, "changed": 0, "deleted": 0,
                  "unchanged": len(files)},
        "note": "ADOPTED from the published dataset version by "
                "remote/sync.py --adopt-remote; not pushed by this repo",
        "files": files,
    }
    write_last_pushed(doc)
    say(f"  adopted     {len(files)} files from {pack.DATASET_ID}; "
        f"tree {doc['tree'][:16]}")
    return doc


# ----------------------------------------------------------------- the runner


def do_run(pack, plan, args, fails):
    """Delegate the kernel side to remote/kaggle_run.sh, unchanged.

    --skip-pack is the whole point of the handoff: the payload is already
    published and, crucially, already `ready`, which is the state that script
    had to guess at when it cost a session. Everything after that -- notebook
    build, validation, push, poll, download, index -- stays in the script that
    has already run it, rather than being reimplemented here.
    """
    if not os.path.isfile(KAGGLE_RUN):
        raise Refuse(f"{KAGGLE_RUN} is missing", EXIT_TOOL)
    if fails and not args.allow_omissions:
        raise Refuse("not running: the guards above failed. Fix them, or pass "
                     "--allow-omissions if you have read them and meant it.")
    if args.allow_omissions:
        for _, g in fails:
            say(f"  OVERRIDDEN  {g}")

    kdir = plan["kdir"]
    kid = ""
    meta = os.path.join(kdir, "kernel-metadata.json")
    if os.path.isfile(meta):
        kid = json.load(open(meta)).get("id", "")
    # Kaggle slugs the TITLE, and `kernels push` 409s when the title does not
    # slug back to the id. make_notebook.py regenerates metadata on every run,
    # so a hand-edited title is a 409 waiting to happen; check it here, where
    # it costs nothing, rather than after a notebook has been assembled.
    if kid:
        import re
        title = json.load(open(meta)).get("title", "")
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        want = kid.split("/", 1)[-1]
        if slug != want:
            raise Refuse(f"kernel title {title!r} slugs to {slug!r}, which is not "
                         f"the id slug {want!r}. `kaggle kernels push` will answer "
                         f"409 'Your kernel title does not resolve to the "
                         f"specified id'. Derive the title from the slug; do not "
                         f"hand-edit it, because the generator overwrites it.")
        say(f"  kernel      {kid}  (title slugs back to the id)")

    before = set(os.listdir(OUT_KAGGLE)) if os.path.isdir(OUT_KAGGLE) else set()
    cmd = [KAGGLE_RUN, "--skip-pack", "--variant", args.variant,
           "--timeout", str(args.timeout)]
    if args.only:
        cmd += ["--only", args.only]
    t0 = time.time()
    say("")
    say("=== delegate: remote/kaggle_run.sh --skip-pack ===")
    say(f"  cmd         {' '.join(cmd)}")
    say(f"  bound       queue wait {args.timeout}m, session cap "
        f"{args.session}m (Kaggle enforces the cap server-side)")
    p = subprocess.run(cmd, cwd=REPO)
    secs = time.time() - t0
    say(f"  delegated run finished in {secs / 60:.1f}m, rc={p.returncode}")

    after = set(os.listdir(OUT_KAGGLE)) if os.path.isdir(OUT_KAGGLE) else set()
    new = sorted(after - before)
    if not new:
        say("  no new run directory under out/kaggle/; nothing to index")
        return p.returncode, None, secs
    outdir = os.path.join(OUT_KAGGLE, new[-1])
    say(f"  results     {os.path.relpath(outdir, REPO)}")

    # the manifest travels with the results, so a number is always attributable
    base = plan["baseline"] if plan["baseline"] else load_last_pushed()
    if base:
        with open(os.path.join(outdir, "payload-manifest.json"), "w") as f:
            json.dump(base, f, indent=2, sort_keys=True)
            f.write("\n")
        say(f"  manifest    payload-manifest.json written beside the results "
            f"(tree {base.get('tree', '?')[:16]}, {len(base.get('files', {}))} files)")
    with open(os.path.join(outdir, "sync-plan.json"), "w") as f:
        json.dump({k: v for k, v in plan.items()
                   if k not in ("pack", "files", "baseline", "closure")}, f,
                  indent=2, sort_keys=True, default=str)
        f.write("\n")

    say("")
    say("=== index ===")
    rows = []
    for dirpath, _, filenames in os.walk(outdir):
        for fn in sorted(filenames):
            ap = os.path.join(dirpath, fn)
            rows.append((os.path.relpath(ap, outdir), os.path.getsize(ap)))
    for rel, sz in sorted(rows):
        say(f"  {rel:<58s} {human(sz):>10s}")
    say(f"  {len(rows)} files, {human(sum(s for _, s in rows))}, in "
        f"{os.path.relpath(outdir, REPO)}")
    if plan["baseline"]:
        say(f"  these results are attributable to tree "
            f"{plan['baseline'].get('tree', '?')[:16]} "
            f"(payload sha256 {(plan['baseline'].get('tarball') or {}).get('sha256', '?')[:16]}...)")
    return p.returncode, outdir, secs


# ----------------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="sync.py",
        description="incremental payload sync to the Kaggle dataset, and the loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("USAGE")[-1])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true",
                      help="offline: diff and report, zero network calls, zero credentials")
    mode.add_argument("--sync", action="store_true", help="publish only if something changed (default)")
    mode.add_argument("--run", action="store_true", help="sync, then push the kernel and collect results")
    mode.add_argument("--status", action="store_true", help="print what is published; one API call, no hashing")
    ap.add_argument("--adopt-remote", action="store_true",
                    help="seed remote/last-pushed.json from the published version instead of pushing")
    ap.add_argument("--variant", choices=("gpu", "tpu"), default="gpu")
    ap.add_argument("--only", default="", help="comma-separated probe keys, forwarded to the run")
    ap.add_argument("--timeout", type=int, default=90, help="queue-wait minutes (default 90)")
    ap.add_argument("--session", type=int, default=75, help="kernel session cap, minutes (default 75)")
    ap.add_argument("--ready-timeout", type=int, default=300, help="seconds to wait for the dataset to mount (default 300)")
    ap.add_argument("--force", action="store_true", help="publish even when the diff is empty")
    ap.add_argument("--allow-omissions", action="store_true", help="run even though a guard failed")
    ap.add_argument("--fail-on-dirty", action="store_true", help="exit 3 when there are pending changes")
    ap.add_argument("--json", action="store_true", help="also write the plan to out/sync/<stamp>/plan.json")
    args = ap.parse_args(argv)
    args.mode = "run" if args.run else ("check" if args.check else "sync")

    try:
        check_interpreter()
        if args.status:
            return do_status()
        pack = load_pack()
        t0 = time.time()
        plan = make_plan(args, pack)
        t_plan = time.time() - t0
        fails = report(plan, args)
        say(f"  plan        computed in {t_plan:.2f}s")
        if args.check:
            say("  check       offline. nothing was read from the network and "
                "no credential was looked at.")
        if args.json:
            dump = {k: v for k, v in plan.items() if k != "pack"}
            dump["guards"] = plan["guards"]
            dump["seconds"] = round(t_plan, 3)
            d = os.path.join(OUT_SYNC, stamp())
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "plan.json"), "w") as f:
                json.dump(dump, f, indent=2, sort_keys=True, default=str)
            say(f"  json        {os.path.relpath(os.path.join(d, 'plan.json'), REPO)}")

        if fails:
            raise Refuse("the guards above failed; nothing was published")
        if args.check:
            if args.fail_on_dirty and plan["changed"]:
                return EXIT_DIRTY
            return EXIT_OK
        if args.adopt_remote:
            adopt_remote(pack, plan, args)
            return EXIT_OK

        doc = None
        if plan["changed"] or args.force:
            if not plan["changed"]:
                say("  --force     publishing anyway despite an empty diff")
            doc = publish(pack, plan, args)
        else:
            say("  NO-OP       nothing changed since the last push. No dataset "
                "version was created.")
            say(f"              {len(plan['diff']['unchanged'])} files skipped: "
                f"every sha256 is identical to remote/last-pushed.json")
            say("              (--force to publish anyway)")
            if not args.check:
                # Credentials first, even though nothing is being uploaded: the
                # cross-check below makes network calls, and without this the
                # failure surfaces as "kaggle datasets files failed", which
                # reads like a Kaggle outage rather than a missing token.
                require_credentials()
                # On the no-op path the remote is supposed to match the
                # BASELINE, and the local payload.tgz on disk is a build
                # artefact of some earlier sync. Comparing the remote against
                # the baseline is the question that matters: does what is
                # deployed equal what we last pushed?
                want = (plan["baseline"] or {}).get("tarball", {}).get("sha256")
                xc = cross_check(pack.DATASET_ID, sorted(os.listdir(PAYLOAD_DIR)), want)
                if not xc["problems"]:
                    say(f"  cross-check the published version still equals the "
                        f"baseline: {len(xc['remote'] or {})} files, tarball "
                        f"sha256 verified by download")
        if args.run:
            rc, _, secs = do_run(pack, plan, args, fails)
            if rc != 0:
                return EXIT_RUN
            say("")
            say(f"=== done: sync+plan {t_plan:.2f}s, run {secs / 60:.1f}m ===")
        return EXIT_OK

    except Refuse as e:
        say("")
        say(f"REFUSED: {e}")
        return e.code
    except KeyboardInterrupt:
        say("\ninterrupted")
        return 130


def do_status() -> int:
    try:
        require_credentials()
        pack = load_pack()
        files = remote_files(pack.DATASET_ID)
        base = load_last_pushed()
        say(f"dataset   {pack.DATASET_ID}")
        say(f"status    {kaggle('datasets', 'status', pack.DATASET_ID).stdout.strip()}")
        for n in sorted(files):
            say(f"  {n:<32s} {human(files[n].get('size', 0)):>10s}  "
                f"{files[n].get('creationDate', '')}")
        sha, err = remote_tarball_sha256(pack.DATASET_ID)
        if err:
            say(f"  tarball sha256 unavailable: {err}")
        else:
            say(f"  payload.tgz sha256 {sha}")
        if base:
            same = base.get("tarball", {}).get("sha256") == sha
            say(f"baseline  {os.path.relpath(LAST_PUSHED, REPO)} tree "
                f"{base.get('tree', '?')[:16]} pushed {base.get('pushed_at')}")
            verdict = "YES" if same else (
                "NO -- the repo and the dataset disagree; run --check to see "
                "the diff, and do not trust a result until it is resolved")
            say(f"  agreement {verdict}")
        else:
            say(f"baseline  none. remote/last-pushed.json does not exist.")
        return EXIT_OK
    except Refuse as e:
        say(f"REFUSED: {e}")
        return e.code


if __name__ == "__main__":
    sys.exit(main())
