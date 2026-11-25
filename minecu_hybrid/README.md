# minecu_hybrid — TinyGrad program + CUDA inline accelerators

This is **one** RL environment, not two backends fighting each other.

```
MineCUEnv  (Python control flow, spaces, reset/step API)
    │
    ├─ slow / irregular hot path ──► accel.py  (Infatoshi CUDA = "inline asm")
    │                                   launch_step_* / render_prebasis / worldgen
    │
    └─ otherwise ──────────────────► TinyGrad (minecu_tg_opt render + NumPy physics)
```

When TinyGrad’s codegen improves, turn `accel="off"` (or leave `auto` on a
machine without `libminecu.so`) and the **same** `MineCUEnv` API rides those
gains. When you need Infatoshi parity today, keep `accel="auto"`/`"on"` and the
hot block is the original kernels — still *called from* this program, not a
fork that abandons TinyGrad.

## Quick use

```python
from minecu_hybrid import MineCUEnv, viewer

env = MineCUEnv(batch=256)                 # accel=auto
print(env.using_cuda_accel)                # True on Colab after build_lib.sh
obs = env.reset()
obs, rew, done = env.step(*env.zero_actions())
viewer.save_mosaic(obs, "mosaic.ppm")      # watchable rollout frames
```

```bash
# NVIDIA host
./minecu_hybrid/build_lib.sh
pip install cupy-cuda12x
PYTHONPATH=. python -m minecu_hybrid.smoke

# force pure TinyGrad (no asm)
PYTHONPATH=. python -c "from minecu_hybrid import MineCUEnv; e=MineCUEnv(accel='off', batch=2); print(e.reset().shape)"
```

## Display

Obs are low-res RGB (agent eyes), not the Java client. `viewer` writes PPM
mosaics/sequences so you can **see** what the policy sees.

## TPU

Not supported. CUDA accel needs NVIDIA; TinyGrad has no TPU device here.
