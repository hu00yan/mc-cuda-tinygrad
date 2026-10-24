#!/usr/bin/env python3
"""pack.py -- build the payload that the Kaggle notebook runs.

WHAT THIS IS FOR
    `kaggle kernels push` uploads exactly one thing: the notebook named by
    `code_file` in kernel-metadata.json. Sidecar files in the kernel directory
    are never uploaded (verified against kaggle 2.2.4's `KaggleApi.kernels_push`,
    which reads the metadata, reads the code file, and POSTs it as text). So a
    notebook cannot carry the repo's probe directories with it unless they are
    inside the notebook.

    Inlining ~250 KB of Python as escaped notebook source is possible and
    horrible: it cannot be reviewed, it makes every one-line change a new kernel
    version, and a syntax error in the payload is invisible to
    `compile()`-style checks.

    So the payload goes to a Kaggle **dataset** -- `tenghuzhang/mc-cuda-tinygrad-payload` --
    and the kernel attaches it with `"dataset_sources"`. On the runner it arrives
    read-only at `/kaggle/input/<slug>/`, with no download, no token, no third
    party, and no invented URL: it is the same Kaggle account, mounted by
    Kaggle itself, and versioned so a kernel run can be tied to an exact copy of
    the code. The tarball (rather than loose files) also sidesteps the dataset
    API's tabular-file conversion and gives the notebook one sha256 to verify.

    Other candidate sources, and why they are not used:
      * a public git host -- this repo has no remote and is not published, so
        there is no URL that exists. Inventing one would be a lie the notebook
        only discovers at run time.
      * base64 in the notebook -- works, but the payload becomes unreviewable
        and the notebook is rewritten on every payload change.
      * netherite -- that one *is* public, so the notebook clones it itself
        (sparse, blaze + docs), which is what pack.py excludes.

WHAT IT PACKS
    kda/*.py            the KDA reference, the TinyGrad port, the tests,
                        codegen_diff, and the fla/triton comparison
    probes/*.py         the backend probes
    probes/oc/*         the semantic camera: tiny port, C reference, Makefile
    remote/preflight.py the environment report that runs first on any box

WHAT IT LEAVES OUT, ON PURPOSE
    __pycache__/ *.pyc
    probes/oc/cells_*.bin  probes/oc/ref.bin  probes/oc/ref_oc
        23 MB of regenerable reference data; `make check` rebuilds all of it on
        the runner in well under a minute, and shipping it would make every
        payload version a 23 MB upload.
    netherite/         sparse-cloned by the notebook instead
    out/  .venv/  notes/  research/  kernels/   not needed to run the probes

USAGE
    .venv/bin/python kernels/pack.py            # -> kernels/payload/
    .venv/bin/python kernels/pack.py --verify   # re-verify an existing payload
"""
import argparse
import hashlib
import json
import os
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT_DIR = os.path.join(HERE, "payload")
TARBALL = os.path.join(OUT_DIR, "payload.tgz")
SIDECAR = os.path.join(OUT_DIR, "payload.sha256")

# (source path relative to repo root, path inside the payload tarball)
INCLUDE = [
    ("remote/preflight.py", "probe/remote/preflight.py"),
    ("kda/ref.py", "probe/kda/ref.py"),
    ("kda/test_ref.py", "probe/kda/test_ref.py"),
    ("kda/tiny_kda.py", "probe/kda/tiny_kda.py"),
    ("kda/test_tiny.py", "probe/kda/test_tiny.py"),
    ("kda/cost.py", "probe/kda/cost.py"),
    ("kda/perf_kda.py", "probe/kda/perf_kda.py"),
    ("kda/replay_kda.py", "probe/kda/replay_kda.py"),
    ("kda/codegen_diff.py", "probe/kda/codegen_diff.py"),
    ("kda/kda_vs_fla.py", "probe/kda/kda_vs_fla.py"),
    ("probes/bench.py", "probe/probes/bench.py"),
    ("probes/fp32_correctness.py", "probe/probes/fp32_correctness.py"),
    ("probes/divide_precision.py", "probe/probes/divide_precision.py"),
    ("probes/gather_floor.py", "probe/probes/gather_floor.py"),
    ("probes/gather_scale.py", "probe/probes/gather_scale.py"),
    ("probes/real_gather.py", "probe/probes/real_gather.py"),
    ("probes/oc/Makefile", "probe/probes/oc/Makefile"),
    ("probes/oc/ref_oc.c", "probe/probes/oc/ref_oc.c"),
    ("probes/oc/tiny_oc.py", "probe/probes/oc/tiny_oc.py"),
    ("probes/oc/np_oc.py", "probe/probes/oc/np_oc.py"),
    ("probes/oc/run_oc.py", "probe/probes/oc/run_oc.py"),
    ("probes/oc/oc_report.py", "probe/probes/oc/oc_report.py"),
    ("probes/oc/dbg_oc.py", "probe/probes/oc/dbg_oc.py"),
    ("probes/oc/trace_oc.py", "probe/probes/oc/trace_oc.py"),
    ("probes/oc/trace2_oc.py", "probe/probes/oc/trace2_oc.py"),
]

DATASET_ID = "tenghuzhang/mc-cuda-tinygrad-payload"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build():
    missing = [src for src, _ in INCLUDE if not os.path.isfile(os.path.join(REPO, src))]
    if missing:
        print("missing from the repo, cannot pack:", file=sys.stderr)
        for m in missing:
            print("   ", m, file=sys.stderr)
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    # A fixed mtime/uid/gid so an unchanged tree produces a byte-identical
    # tarball; the dataset API refuses a "new version" whose bytes are
    # identical, and we would rather that check be meaningful.
    entries = []
    with tarfile.open(TARBALL, "w:gz", compresslevel=9) as tf:
        def add(src, arc, info=None):
            data = open(os.path.join(REPO, src), "rb").read()
            ti = tarfile.TarInfo(arc)
            ti.size = len(data)
            ti.mtime = 0
            ti.mode = 0o644
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = "root"
            tf.addfile(ti, __import__("io").BytesIO(data))
            entries.append({"path": arc, "bytes": len(data),
                            "sha256": hashlib.sha256(data).hexdigest()})

        for src, arc in INCLUDE:
            add(src, arc)
        manifest = {
            "repo": os.path.basename(REPO),
            "files": entries,
            "note": "generated by kernels/pack.py -- do not edit",
        }
        blob = json.dumps(manifest, indent=2, sort_keys=True).encode()
        ti = tarfile.TarInfo("probe/MANIFEST.json")
        ti.size, ti.mtime, ti.mode = len(blob), 0, 0o644
        ti.uid = ti.gid = 0
        ti.uname = ti.gname = "root"
        tf.addfile(ti, __import__("io").BytesIO(blob))

    digest = sha256(TARBALL)
    with open(SIDECAR, "w") as f:
        f.write(f"{digest}  payload.tgz\n")

    with open(os.path.join(OUT_DIR, "dataset-metadata.json"), "w") as f:
        json.dump({
            "title": "mc-cuda-tinygrad probe payload",
            "id": DATASET_ID,
            "subtitle": "kda/ and probes/ as one archive, for the Kaggle kernels",
            "description": (
                "Payload for the mc-cuda-tinygrad Kaggle kernels: the KDA "
                "reference and TinyGrad port, the backend probes, and the "
                "semantic camera. Built by kernels/pack.py in the "
                "mc-cuda-tinygrad repo. One tarball, sha256 in payload.sha256, "
                "per-file hashes in probe/MANIFEST.json."
            ),
            "licenses": [{"name": "other"}],
            "isPrivate": True,
        }, f, indent=2)
        f.write("\n")

    print(f"payload.tgz  {os.path.getsize(TARBALL)/1024:.1f} KB  {len(entries)} files")
    print(f"sha256       {digest}")
    print(f"dataset      {DATASET_ID}")
    return 0


def verify():
    """Extract to a temp dir and check every hash in MANIFEST.json."""
    import io
    import tempfile
    ok = True
    with tarfile.open(TARBALL, "r:gz") as tf:
        man = json.load(tf.extractfile("probe/MANIFEST.json"))
        with tempfile.TemporaryDirectory() as tmp:
            for e in man["files"]:
                blob = tf.extractfile(e["path"]).read()
                got = hashlib.sha256(blob).hexdigest()
                if got != e["sha256"] or len(blob) != e["bytes"]:
                    ok = False
                    print(f"  MISMATCH {e['path']}")
            print(f"{'ok  ' if ok else 'FAIL'} {len(man['files'])} files match MANIFEST.json")
    with open(SIDECAR) as f:
        want = f.read().split()[0]
    got = sha256(TARBALL)
    print(f"{'ok  ' if want == got else 'FAIL'} tarball sha256 matches sidecar")
    ok = ok and want == got
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    sys.exit(verify() if a.verify else build())
