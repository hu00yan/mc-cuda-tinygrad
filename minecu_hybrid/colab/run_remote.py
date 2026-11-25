"""One-shot Colab cell body: build libminecu.so and run smoke.

Uploaded next to the repo tree as /content/mc-cuda-tinygrad/...
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.environ.get("MINECU_ROOT", "/content/mc-cuda-tinygrad")
os.chdir(ROOT)
sys.path.insert(0, ROOT)

print("=== nvidia-smi ===")
print(subprocess.check_output(["nvidia-smi", "-L"], text=True))

print("=== install cupy if needed ===")
try:
    import cupy  # noqa: F401
    print("cupy ok", cupy.__version__)
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "cupy-cuda12x"])
    import cupy
    print("cupy installed", cupy.__version__)

print("=== build libminecu.so ===")
subprocess.check_call(["bash", "minecu_hybrid/build_lib.sh"], env={**os.environ, "SM": "sm_75"})

print("=== smoke ===")
subprocess.check_call([sys.executable, "-m", "minecu_hybrid.smoke"])
