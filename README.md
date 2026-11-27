# mc-cuda-tinygrad

Two studies on how far automatic codegen gets on irregular GPU ops, and how to
close the rest with inline CUDA:

1. **KDA** — TinyGrad port + optional fla CUDA accel
2. **MineCU** — Infatoshi’s voxel raymarcher: compare TinyGrad to the **external**
   hand-written CUDA, then wrap those kernels as inline accelerators in one RL env

## Layout

```
minecu/                 Minecraft / MineCU (this study)
  tinygrad/             naive TinyGrad DDA port
  opt/                  algorithmic ladder (naive → lean) + T4 timings
  dda_model.py          our CPU transcription (for parity gates)
  upstream.py           resolves Infatoshi/mine.cu (env / .deps fetch + pin)
  *.py, rl/             ★ RL env: TinyGrad control flow + CUDA inline accel

kda/                    Kimi Delta Attention
  hybrid/               TinyGrad API + optional fla CUDA inline accel

bench_cuda/             our timing harness against the external kernels
notes/  derivation/  out/
```

**Original CUDA is not vendored.** It comes from
[Infatoshi/mine.cu](https://github.com/Infatoshi/mine.cu):

```bash
# option A: point at a local checkout
export MINECU_MINE_CU=~/src/mine.cu

# option B: fetch the pinned commit into .deps/ (gitignored)
PYTHONPATH=. python -m minecu.upstream
```

```python
from minecu import MineCUEnv
from minecu.opt.render import render
from kda.hybrid import kda_chunk
```

## MineCU

```bash
PYTHONPATH=. python -m minecu.upstream                 # ensure kernels.cu
./minecu/build_lib.sh && pip install cupy-cuda12x      # NVIDIA
PYTHONPATH=. python -m minecu.smoke

PYTHONPATH=. DEV=CPU python -m minecu.cpu_gate
.venv/bin/python -m minecu.opt.verify_opt
PYTHONPATH=. python -m minecu.pin_kernels
```

| claim | result (Tesla T4) |
|---|---|
| Correctness | byte-identical to CUDA reference obs |
| Best TinyGrad render | ~9–11× slower than hand-written (~8–11% of CUDA) |
| Naive first port | ~27–37× slower (algorithm, not “compiler is slow”) |

Details: [`notes/35-ladder-report.md`](notes/35-ladder-report.md),
[`notes/36-rung-timings.md`](notes/36-rung-timings.md).

## KDA

```bash
PYTHONPATH=. python -m kda.test_tiny
PYTHONPATH=. python -m kda.hybrid.parity
```

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python tinygrad numpy tabulate
```
