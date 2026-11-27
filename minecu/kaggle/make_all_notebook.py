#!/usr/bin/env python3
"""Kaggle notebook: obs_parity + short wood train + kda.hybrid parity/bench."""
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
    "minecu/obs_parity.py",
    "minecu/parity_audit.py",
    "minecu/smoke.py",
    "minecu/spaces.py",
    "minecu/viewer.py",
    "minecu/examples/__init__.py",
    "minecu/examples/train_wood.py",
    "minecu/tinygrad/__init__.py",
    "minecu/tinygrad/blocks.py",
    "minecu/tinygrad/world.py",
    "minecu/tinygrad/render.py",
    "minecu/opt/__init__.py",
    "minecu/opt/render.py",
    "kda/hybrid/__init__.py",
    "kda/hybrid/accel.py",
    "kda/hybrid/kda.py",
    "kda/hybrid/parity.py",
    "kda/ref.py",
    "kda/test_ref.py",
    "kda/tiny_kda.py",
    "kda/tiny_kda_opt.py",
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
    payload_json = json.dumps(payloads)
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": [
            cell("# minecu + kda hybrid: obs parity, wood train, KDA accel", typ="markdown"),
            cell(
                f"""
import base64, hashlib, json, os, subprocess, sys, time
from pathlib import Path
ROOT = Path('/kaggle/working')
os.chdir(ROOT)
PAYLOADS = json.loads({payload_json!r})
for rel, meta in PAYLOADS.items():
    raw = base64.b64decode(meta['b64'])
    assert hashlib.sha256(raw).hexdigest() == meta['sha256'], rel
    dest = ROOT / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)
print('payloads', len(PAYLOADS))
"""
            ),
            cell(
                """
import os, subprocess, sys, time
from pathlib import Path
ROOT = Path('/kaggle/working')
os.chdir(ROOT)
os.environ['PYTHONPATH'] = str(ROOT)

print('=== device ===')
print(subprocess.check_output(['nvidia-smi', '-L'], text=True))

def ensure_cupy():
    try:
        import cupy as cp
        cp.zeros(1)
        print('cupy', cp.__version__)
        return
    except Exception as e:
        print('cupy bad', e)
    subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y',
                           'cupy', 'cupy-cuda11x', 'cupy-cuda12x', 'cupy-cuda13x'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'cupy-cuda12x'])
    import cupy as cp
    print('cupy installed', cp.__version__)

ensure_cupy()
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'tinygrad==0.14.0', 'torch', 'triton', 'flash-linear-attention'])
try:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'flash-kda'])
except Exception as e:
    print('flash-kda optional skip', e)

print('=== build libminecu.so ===')
subprocess.check_call(['bash', 'minecu/build_lib.sh'], env={**os.environ, 'SM': 'sm_75'})

print('=== obs_parity ===')
subprocess.check_call([sys.executable, '-m', 'minecu.obs_parity'])

print('=== train_wood (short) ===')
subprocess.check_call([
    sys.executable, '-m', 'minecu.examples.train_wood',
    '--steps', '100000', '--batch', '1024', '--log-interval', '5',
])

print('=== kda.hybrid parity ===')
subprocess.check_call([sys.executable, '-m', 'kda.hybrid.parity'])

Path('/kaggle/working/SUMMARY.txt').write_text(
    'obs_parity + train_wood + kda.hybrid OK\\n'
)
print('DONE')
"""
            ),
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(nb))
    print(f"wrote {args.out} ({args.out.stat().st_size} B)")


if __name__ == "__main__":
    main()
