#!/usr/bin/env python3
"""Kaggle notebook: obs_parity + short wood train + kda_hybrid parity/bench."""
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
    "minecu_hybrid/obs_parity.py",
    "minecu_hybrid/parity_audit.py",
    "minecu_hybrid/smoke.py",
    "minecu_hybrid/spaces.py",
    "minecu_hybrid/viewer.py",
    "minecu_hybrid/examples/__init__.py",
    "minecu_hybrid/examples/train_wood.py",
    "minecu_tg/__init__.py",
    "minecu_tg/blocks.py",
    "minecu_tg/world.py",
    "minecu_tg/render.py",
    "minecu_tg_opt/__init__.py",
    "minecu_tg_opt/render.py",
    "kda_hybrid/__init__.py",
    "kda_hybrid/accel.py",
    "kda_hybrid/kda.py",
    "kda_hybrid/parity.py",
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
    payloads = {}
    for rel in FILES:
        raw = (args.repo / rel).read_bytes()
        payloads[rel] = {
            "b64": base64.b64encode(raw).decode("ascii"),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "n": len(raw),
        }
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
subprocess.check_call(['bash', 'minecu_hybrid/build_lib.sh'], env={**os.environ, 'SM': 'sm_75'})

print('=== obs_parity ===')
subprocess.check_call([sys.executable, '-m', 'minecu_hybrid.obs_parity'])

print('=== train_wood (short) ===')
subprocess.check_call([
    sys.executable, '-m', 'minecu_hybrid.examples.train_wood',
    '--steps', '100000', '--batch', '1024', '--log-interval', '5',
])

print('=== kda_hybrid parity ===')
subprocess.check_call([sys.executable, '-m', 'kda_hybrid.parity'])

Path('/kaggle/working/SUMMARY.txt').write_text(
    'obs_parity + train_wood + kda_hybrid OK\\n'
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
