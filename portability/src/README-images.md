# Images and why they are what they are

Everything here runs in a container with `--platform linux/amd64` under OrbStack's
x86 emulation. No GPU, no driver, no host install outside the repo venv. This file
records the provenance so a reader can rebuild the environment.

## `archlab/nvcc:12.9` — the CUDA toolchain (borrowed)

Built by the `tools/archlab` lane, **read-only, not mine**. Contains
`nvcc`/`ptxas`/`nvdisasm` at `/usr/local/cuda-12.9/bin`, and g++-13.

I used it rather than building one because the multi-GB `nvidia/cuda` image pull
previously timed out at 10 minutes, and this image is 811 MB, already local, and
already has the host compiler CUTLASS 4.8 needs. Verified in this lane:

```
nvcc --version  -> release 12.9, V12.9.86
ptxas --help    -> Allowed values: compute_50 ... compute_121,
                  including compute_90a / compute_100a / compute_100f / compute_120a
```

**A single 12.9 toolkit can therefore target every rung in the matrix, including
`sm_100a`.** That matters: it means the Blackwell rows are measurements rather than
extrapolations, and it is why the brief's warning about never reporting a number for
an untargeted arch is satisfiable here. `sm_103` and `sm_121` were targeted too.

## `portab/triton:3.4.0` — built by this lane

The repo venv is macOS arm64 and has no Triton, and Triton has no macOS wheel. So
the linux/amd64 wheel is installed into a container and the result is committed:

```bash
docker run --name portab_triton_build -v "$PWD/wheels:/w" \
  --platform linux/amd64 python:3.12-slim-bookworm sh -lc \
  'pip download -q -d /w triton==3.4.0 && pip install -q /w/triton-3.4.0-*.whl'
docker commit portab_triton_build portab/triton:3.4.0
docker rm portab_triton_build
```

**Do not use `docker build` here.** OrbStack's BuildKit cannot resolve base-image
manifests from `registry-1.docker.io` in this environment (`TLS handshake timeout`),
and the legacy builder hangs. `docker commit` over a `docker run` works, so that is
the route. The wheel is cached in `portability/wheels/` to make the rebuild offline.

**Triton bundles its own assembler**, at
`site-packages/triton/backends/nvidia/bin/ptxas`, which is **12.8.93** — one minor
behind the system nvcc. Its target list is `compute_50 … compute_120` including
`compute_90a`, `compute_100a`, `compute_120a`, but it has no `compute_103` or
`compute_121`. **Any Triton row is therefore a statement about Triton's bundled
ptxas, not about the hardware.** This is recorded in the matrix rather than glossed.

## `portab/cvc:12.9` — an abandoned attempt

`ubuntu:24.04` + `pip install nvidia-cuda-nvcc-cu12` (the route the brief suggested).
It builds and the nvcc install works, but the `nvidia-cuda-crt-cu12` /
`nvidia-cuda-cccl-cu12` extras time out against the NVIDIA index, and it turned out
to be unnecessary: `archlab/nvcc:12.9` already has g++-13, and the CUTLASS
`platform.h` error I first hit was my own `device_memory::allocation<void>`
misuse, not a host-compiler version problem. The image exists; nothing in the
matrix depends on it.

## Subjects fetched as source, not as images

| what | where from | why not a wheel |
|---|---|---|
| CUTLASS 4.8.0 | `cutlass_lab/cutlass` (read-only, another lane) | already in the repo |
| flash-linear-attention | `subj/fla/flash-linear-attention-main` (GitHub tarball) | the PyPI wheel `flash_linear_attention-0.5.2-py3-none-any.whl` **ships no `fla/ops/`** — only `fla/layers` and `fla/models` — so the KDA kernels are unobtainable from it |
| FlashAttention 2.8.3 | `subj/fa/flash-attention-2.8.3` (GitHub tarball) | needs libtorch to build; see below |
| FlashAttention 3 | `subj/fa/flash-attention-main` (GitHub tarball) | same |

`codeload.github.com` is reachable; `raw.githubusercontent.com` is reachable;
`api.github.com` returns 403.

## What could not be fetched from this network

- **`download.pytorch.org` and `download-r2.pytorch.org`: read timeout.** This is
  why neither FA2 nor FA3 was compiled for real, and why the fla KDA kernels are
  `ast`-lifted and compiled without torch. A torch *CPU* wheel is the only missing
  piece for both.
- **`pypi.nvidia.org`**: the nvcc package installs but the crt/cccl extras time out.
- **`nvidia-cublas-cu12` / `nvidia-cudnn`**: not retrievable, which is the sole
  reason the cuBLAS/cuDNN row in the matrix is empty. Note that disassembling the
  shipped `.so` would answer it **without a GPU**, so this is the highest-value
  experiment left unrun, and it is blocked on a network problem, not on hardware.
