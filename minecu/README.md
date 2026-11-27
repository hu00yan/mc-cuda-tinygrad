# minecu — TinyGrad RL env + CUDA inline accelerators

One RL environment. Hot render/step can call Infatoshi’s CUDA kernels
(external dep via `minecu.upstream`) as inline accelerators; otherwise it rides
`minecu.opt` TinyGrad.

```
minecu/
  tinygrad/    naive TinyGrad port
  opt/         algorithmic ladder used when accel=off
  dda_model.py our CPU parity model (not upstream)
  upstream.py  Infatoshi/mine.cu resolver + pin
  rl/          PPO / GRPO / async trainers
```

```bash
PYTHONPATH=. python -m minecu.upstream    # or: export MINECU_MINE_CU=/path/to/mine.cu
./minecu/build_lib.sh
PYTHONPATH=. python -m minecu.smoke
PYTHONPATH=. DEV=CPU python -m minecu.cpu_gate
```
