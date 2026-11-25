#!/usr/bin/env python3
"""Kaggle: pin kernels.cu to Infatoshi + MineCUEnv↔CudaBackend lockstep."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

FILES = [
    "minecu_ref/src/kernels.cu",
    "minecu_hybrid/__init__.py",
    "minecu_hybrid/accel.py",
    "minecu_hybrid/backend.py",
    "minecu_hybrid/backend_cuda.py",
    "minecu_hybrid/backend_tinygrad.py",
    "minecu_hybrid/build_lib.sh",
    "minecu_hybrid/config.py",
    "minecu_hybrid/env.py",
    "minecu_hybrid/lib.py",
    "minecu_hybrid/lockstep_backends.py",
    "minecu_hybrid/pin_kernels.py",
    "minecu_hybrid/spaces.py",
    "minecu_hybrid/viewer.py",
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
    payloads = {
        rel: {
            "b64": base64.b64encode((args.repo / rel).read_bytes()).decode(),
            "sha256": hashlib.sha256((args.repo / rel).read_bytes()).hexdigest(),
        }
        for rel in FILES
    }
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
print(subprocess.check_output(['nvidia-smi','-L'], text=True))
try:
    import cupy as cp; cp.zeros(1)
except Exception:
    subprocess.check_call([sys.executable,'-m','pip','uninstall','-y','cupy','cupy-cuda12x','cupy-cuda13x'])
    subprocess.check_call([sys.executable,'-m','pip','install','-q','cupy-cuda12x'])
print('=== pin_kernels (needs internet) ===')
subprocess.check_call([sys.executable,'-m','minecu_hybrid.pin_kernels'])
print('=== build lib ===')
subprocess.check_call(['bash','minecu_hybrid/build_lib.sh'], env={**os.environ,'SM':'sm_75'})
print('=== lockstep backends ===')
subprocess.check_call([sys.executable,'-m','minecu_hybrid.lockstep_backends'])
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
