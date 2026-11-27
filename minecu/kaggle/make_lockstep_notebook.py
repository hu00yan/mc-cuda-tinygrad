#!/usr/bin/env python3
"""Kaggle: pin kernels.cu to Infatoshi + MineCUEnv↔CudaBackend lockstep."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

FILES = [
    "minecu/upstream.py",
    "minecu/pin_kernels.py",
        "minecu/__init__.py",
    "minecu/accel.py",
    "minecu/backend.py",
    "minecu/backend_cuda.py",
    "minecu/backend_tinygrad.py",
    "minecu/build_lib.sh",
    "minecu/config.py",
    "minecu/env.py",
    "minecu/lib.py",
    "minecu/lockstep_backends.py",
    "minecu/pin_kernels.py",
    "minecu/spaces.py",
    "minecu/viewer.py",
]


def cell(source: str, *, typ: str = "code") -> dict:
    lines = source.strip("\n").split("\n")
    src = [ln + "\n" for ln in lines[:-1]] + ([lines[-1] + "\n"] if lines else [])
    out: dict = {"cell_type": typ, "metadata": {}, "source": src}
    if typ == "code":
        out["outputs"] = []
        out["execution_count"] = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    repo = args.repo
    import sys
    sys.path.insert(0, str(repo))
    from minecu.upstream import ensure

    def _payload(raw: bytes) -> dict:
        return {
            "b64": base64.b64encode(raw).decode("ascii"),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "n": len(raw),
        }

    payloads = {"kernels.cu": _payload(ensure().read_bytes())}
    for rel in FILES:
        payloads[rel] = _payload((repo / rel).read_bytes())
    pj = json.dumps(payloads)
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": [
            cell("# pin kernels + MineCUEnv↔CudaBackend lockstep", typ="markdown"),
            cell(
                f"""
import base64, hashlib, json, os, subprocess, sys
from pathlib import Path
ROOT = Path('/kaggle/working'); os.chdir(ROOT)
for rel, meta in json.loads({pj!r}).items():
    raw = base64.b64decode(meta['b64'])
    assert hashlib.sha256(raw).hexdigest() == meta['sha256'], rel
    p = ROOT/rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(raw)
print('payloads', len(json.loads({pj!r})))
"""
            ),
            cell(
                """
import os, subprocess, sys
from pathlib import Path
ROOT = Path('/kaggle/working'); os.chdir(ROOT)
os.environ['PYTHONPATH'] = str(ROOT)
os.environ['MINECU_KERNELS'] = str(ROOT / 'kernels.cu')
print(subprocess.check_output(['nvidia-smi','-L'], text=True))
try:
    import cupy as cp; cp.zeros(1)
except Exception:
    subprocess.check_call([sys.executable,'-m','pip','uninstall','-y','cupy','cupy-cuda12x','cupy-cuda13x'])
    subprocess.check_call([sys.executable,'-m','pip','install','-q','cupy-cuda12x'])
print('=== pin_kernels (needs internet) ===')
subprocess.check_call([sys.executable,'-m','minecu.pin_kernels'])
print('=== build lib ===')
subprocess.check_call(['bash','minecu/build_lib.sh'], env={**os.environ,'SM':'sm_75'})
print('=== lockstep backends ===')
subprocess.check_call([sys.executable,'-m','minecu.lockstep_backends'])
Path('/kaggle/working/SUMMARY.txt').write_text('PIN+LOCKSTEP PASS\\n')
print('DONE')
"""
            ),
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(nb))
    print("wrote", args.out, args.out.stat().st_size)


if __name__ == "__main__":
    main()
