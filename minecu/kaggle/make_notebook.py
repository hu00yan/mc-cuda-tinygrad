#!/usr/bin/env python3
"""Assemble a self-contained Kaggle notebook for minecu CUDA smoke+parity.

Payloads travel as base64 inside the notebook (kernels push uploads one file).
"""
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
    "minecu/parity_audit.py",
    "minecu/smoke.py",
    "minecu/spaces.py",
    "minecu/viewer.py",
    # TinyGrad fallback used by smoke accel=off
    "minecu/tinygrad/__init__.py",
    "minecu/tinygrad/blocks.py",
    "minecu/tinygrad/world.py",
    "minecu/tinygrad/render.py",
    "minecu/opt/__init__.py",
    "minecu/opt/render.py",
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
    repo: Path = args.repo

    import sys
    sys.path.insert(0, str(repo))
    from minecu.upstream import ensure

    def _payload(raw: bytes) -> dict:
        return {
            "b64": base64.b64encode(raw).decode("ascii"),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "n": len(raw),
        }

    payloads = {}
    kernels = ensure()
    payloads["kernels.cu"] = _payload(kernels.read_bytes())
    for rel in FILES:
        p = repo / rel
        payloads[rel] = _payload(p.read_bytes())

    payload_json = json.dumps(payloads)

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": [
            cell("# minecu T4 — build libminecu.so, smoke, parity audit", typ="markdown"),
            cell(
                f"""
import base64, hashlib, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path('/kaggle/working')
os.chdir(ROOT)
PAYLOADS = json.loads({payload_json!r})

print('=== materialise payloads ===')
for rel, meta in PAYLOADS.items():
    raw = base64.b64decode(meta['b64'])
    dig = hashlib.sha256(raw).hexdigest()
    assert dig == meta['sha256'], (rel, dig, meta['sha256'])
    dest = ROOT / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)
    print(f'  ok {{rel}} {{meta[\"n\"]}} B')
print('payloads ok', len(PAYLOADS))
os.environ['MINECU_KERNELS'] = str(ROOT / 'kernels.cu')
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

print('=== cupy (prefer existing; else cuda12x) ===')
def _ensure_cupy():
    try:
        import cupy as cp
        print('cupy already', getattr(cp, '__version__', '?'))
        # smoke-test a tiny alloc
        cp.zeros(1)
        return
    except Exception as e:
        print('cupy unusable:', e)
    subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y',
                           'cupy', 'cupy-cuda11x', 'cupy-cuda12x', 'cupy-cuda13x'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'cupy-cuda12x'])
    import cupy as cp
    print('cupy installed', cp.__version__)
    cp.zeros(1)

_ensure_cupy()

print('=== tinygrad (for smoke accel=off) ===')
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'tinygrad==0.14.0'])
import tinygrad
print('tinygrad ok', getattr(tinygrad, '__version__', '0.14.0'))

print('=== build libminecu.so (sm_75) ===')
t0 = time.time()
subprocess.check_call(['bash', 'minecu/build_lib.sh'], env={**os.environ, 'SM': 'sm_75'})
print(f'build {time.time()-t0:.1f}s')

print('=== smoke ===')
subprocess.check_call([sys.executable, '-m', 'minecu.smoke'])

print('=== parity audit ===')
subprocess.check_call([sys.executable, '-m', 'minecu.parity_audit'])

print('=== throughput probe ===')
code = r'''
import time
from minecu import MineCUEnv
import cupy as cp
env = MineCUEnv(accel="on", batch=4096, world=16, width=32, height=32, max_steps=32,
                obs_dtype="uint8_prebasis_full", device_buffers=True, sync_every_step=False,
                episode_length=0)
obs = env.reset()
b, l = env.zero_actions()
b[:, 0] = 1
for _ in range(5):
    env.step(b, l)
cp.cuda.runtime.deviceSynchronize()
n = 50
t0 = time.perf_counter()
for _ in range(n):
    env.step(b, l)
cp.cuda.runtime.deviceSynchronize()
dt = time.perf_counter() - t0
print(f"B=4096 n={n} {dt*1e3:.1f} ms  {4096*n/dt:,.0f} env-steps/s")
env.close()
'''
subprocess.check_call([sys.executable, '-c', code], env={**os.environ, 'PYTHONPATH': str(ROOT)})
Path('/kaggle/working/SUMMARY.txt').write_text('minecu T4 smoke+parity OK\\n')
print('DONE')
"""
            ),
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(nb))
    print(f"wrote {args.out} ({args.out.stat().st_size} B, {len(FILES)} payloads)")


if __name__ == "__main__":
    main()
