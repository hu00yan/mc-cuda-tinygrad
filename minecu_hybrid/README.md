# minecu_hybrid — Infatoshi CUDA kernels + Python RL loop

Hot path is the **original** `kernels.cu` (byte-identical to Infatoshi/mine.cu).
Python owns buffers, world regen on reset, and the gym-style API.

| backend | status |
|---|---|
| NVIDIA CUDA (sm_75+; Colab T4 OK) | **supported** |
| TinyGrad-only render | still in `minecu_tg*` (study path) |
| TPU / Metal custom ops | **not supported** — original has no TPU; TinyGrad has no TPU device |

## Build the shared library (on a Linux NVIDIA box)

```bash
./minecu_hybrid/build_lib.sh          # default -arch=sm_75
SM=sm_80 ./minecu_hybrid/build_lib.sh
```

Produces `minecu_hybrid/libminecu.so`.

## Smoke test

```bash
python -m minecu_hybrid.smoke   # needs cupy + libminecu.so + GPU
```

## Env API

```python
from minecu_hybrid import MineCUEnv
env = MineCUEnv(batch=256, world=16, width=32, height=32)
obs = env.reset()
obs, rewards, dones = env.step(buttons, look)
```

`step` calls `launch_step_uint8_prebasis_full` (decode → physics → interact →
reset → episode → cameras → basis → render).
