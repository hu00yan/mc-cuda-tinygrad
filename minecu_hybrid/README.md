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
from minecu_hybrid import MineCUEnv, viewer, EnvConfig

# Infatoshi-compatible aliases work: batch_size, world_size, render_width, …
env = MineCUEnv(batch=256, obs_dtype="uint8_prebasis_full")  # accel=auto
print(env.using_cuda_accel)                # True on Colab after build_lib.sh
obs = env.reset()
obs, rew, done = env.step(*env.zero_actions())
viewer.save_mosaic(obs, "mosaic.ppm")      # watchable rollout frames

# Device-resident I/O (MineEnv-style, no H2D/D2H per step)
env = MineCUEnv(EnvConfig.upstream(batch_size=1024))
```

```bash
# No NVIDIA: pin + TG↔dda_model byte gate + TinyGrad env smoke
PYTHONPATH=. DEV=CPU python -m minecu_hybrid.cpu_gate

# NVIDIA host (T4: SM=sm_75)
./minecu_hybrid/build_lib.sh
pip install cupy-cuda12x
PYTHONPATH=. python -m minecu_hybrid.smoke
PYTHONPATH=. python -m minecu_hybrid.parity_audit
PYTHONPATH=. python -m minecu_hybrid.lockstep_backends   # MineCUEnv↔CudaBackend
PYTHONPATH=. python -m minecu_hybrid.pin_kernels

# force pure TinyGrad (no asm)
PYTHONPATH=. python -c "from minecu_hybrid import MineCUEnv; e=MineCUEnv(accel='off', batch=2); print(e.reset().shape)"
```

`obs_dtype` matches MineEnv: `float32`, `uint8`, `uint8_minimal`, `uint8_fp16`,
`uint8_prebasis`, `uint8_prebasis_full` (default hot path), `uint8_smem`.

## Extras

```bash
# same-state obs parity (CUDA vs TinyGrad render)
PYTHONPATH=. python -m minecu_hybrid.obs_parity

# wood-break REINFORCE (needs torch + CUDA env)
PYTHONPATH=. python -m minecu_hybrid.examples.train_wood --steps 200000

# Kaggle: obs + train + kda_hybrid
./minecu_hybrid/kaggle/run_all.sh
```

Related: `kda_hybrid/` wraps TinyGrad KDA with optional `fla` CUDA accel.

## Display

Obs are low-res RGB (agent eyes), not the Java client. `viewer` writes PPM
mosaics/sequences so you can **see** what the policy sees.

## TPU

Not supported. CUDA accel needs NVIDIA; TinyGrad has no TPU device here.
