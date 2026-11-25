# minecu_hybrid — stable RL API, swappable backends

Upper layer (`MineCUEnv`) stays fixed. Lower layer is either:

| backend | role |
|---|---|
| **`cuda`** (default on NVIDIA) | Infatoshi `kernels.cu` — today’s performance ceiling |
| **`tinygrad`** | `minecu_tg_opt` render + NumPy physics — rides TinyGrad codegen progress |

```python
from minecu_hybrid import MineCUEnv, EnvConfig, viewer

env = MineCUEnv(backend="cuda", batch=256)          # or backend="tinygrad"
obs = env.reset()
obs, rew, done = env.step(*env.zero_actions())
viewer.save_mosaic(obs, "out/mosaic.ppm")           # you CAN see the world
```

## Is this “Minecraft you can play on screen”?

**Same toy world / same raycast eyes as the agent — not the Java client.**

- Obs are small RGB images (32×32 or 64×64), batched on GPU.
- `viewer` writes PPM mosaics / frame sequences so a rollout is watchable.
- There is no desktop window, no mouse look UI, no infinite terrain — unless you
  build that on top. Original MineCU is the same: headless RL observations.

## Feature surface (vs Infatoshi MineEnv)

| feature | status |
|---|---|
| `reset` / `step` / `get_obs` / `close` | yes |
| masked `reset(mask=…)` | yes |
| buttons `[B,8]` + look `[B,2]` | yes |
| rewards (break target) / episode dones | yes (cuda exact; tg approximate interact) |
| worldgen flat+tree+house | yes |
| `EnvConfig.upstream()` (32³ / 64×64 / ep=60) | yes |
| non-cubic worlds | not yet (cubic only) |
| TPU | **no** — use NVIDIA |

## Build CUDA backend (Linux + NVIDIA)

```bash
./minecu_hybrid/build_lib.sh          # sm_75 default (Colab T4)
pip install cupy-cuda12x
PYTHONPATH=. python -m minecu_hybrid.smoke
```

TinyGrad-only smoke (Mac Metal OK):

```bash
PYTHONPATH=. python -c "from minecu_hybrid.smoke import _run; _run('tinygrad', 2)"
```
