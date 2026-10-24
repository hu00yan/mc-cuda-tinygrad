#!/usr/bin/env bash
# Build the containers inspect_ptx.py needs.
#
# Why containers at all: ptxas/nvcc/cuobjdump are x86-64 binaries. On an
# arm64 Mac (or any host without CUDA) the only way to run them is a
# linux/amd64 container. No host packages are touched.
#
# The lean route is deliberate. The official nvidia/cuda:12.6.0-devel-ubuntu24.04
# image is ~4 GB and pulls cublas/cudnn/nccl you will never use. Instead:
#
#   archlab/ptxas:12.9  (232 MB)  ptxas only, from the 40 MB pip wheel
#                                   nvidia-cuda-nvcc-cu12
#   archlab/nvcc:12.9   (804 MB)  nvcc + ptxas + cuobjdump + nvdisasm, from
#                                   three small apt components off NVIDIA's
#                                   official repo
#
# Set KEEP_BUILD_CONTAINER=1 to reuse the existing build containers instead of
# rebuilding from scratch.
set -euo pipefail

NVCC_TAG=12.9
NVCC_IMAGE="archlab/nvcc:${NVCC_TAG}"
PTXAS_IMAGE="archlab/ptxas:${NVCC_TAG}"
BUILD_NVCC=archlab_nvcc_build
BUILD_PTXAS=archlab_ptxas_build

if [ "${KEEP_BUILD_CONTAINER:-0}" = "1" ]; then
  echo "==> committing existing build containers"
  docker commit "$BUILD_NVCC" "$NVCC_IMAGE" >/dev/null
  docker commit "$BUILD_PTXAS" "$PTXAS_IMAGE" >/dev/null
  docker images --format '{{.Repository}}:{{.Tag}}  {{.Size}}' | grep archlab
  exit 0
fi

docker rm -f "$BUILD_NVCC" "$BUILD_PTXAS" >/dev/null 2>&1 || true

# ---------------------------------------------------------------------------
# 1. ptxas-only image: one 40 MB pip wheel, no apt, no CUDA runtime.
# ---------------------------------------------------------------------------
echo "==> building $PTXAS_IMAGE (ptxas via pip wheel)"
docker run --name "$BUILD_PTXAS" --platform linux/amd64 -d python:3.12-slim-bookworm sleep infinity >/dev/null
# --timeout/--retries: x86 emulation is slow enough that the default read
# timeout is not enough for a 40 MB wheel.
docker exec "$BUILD_PTXAS" pip install --no-cache-dir --timeout 120 --retries 10 \
  nvidia-cuda-nvcc-cu12 >/dev/null
docker commit "$BUILD_PTXAS" "$PTXAS_IMAGE" >/dev/null

# ---------------------------------------------------------------------------
# 2. full compiler image: nvcc + cuobjdump + nvdisasm from NVIDIA's apt repo.
#    Only the compiler components -- no cudnn, no cublas, no nccl, no samples.
# ---------------------------------------------------------------------------
echo "==> building $NVCC_IMAGE (nvcc + cuobjdump + nvdisasm via apt)"
docker run --name "$BUILD_NVCC" --platform linux/amd64 -d ubuntu:24.04 sleep infinity >/dev/null
docker exec "$BUILD_NVCC" sh -c '
  set -e
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends ca-certificates wget gnupg
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/k.deb
  dpkg -i /tmp/k.deb
  apt-get update -qq
  # cuobjdump and nvdisasm are separate packages from nvcc; cuobjdump -sass
  # shells out to nvdisasm, so all three are required.
  apt-get install -y -qq --no-install-recommends \
      cuda-nvcc-'"${NVCC_TAG}"' cuda-cuobjdump-'"${NVCC_TAG}"' cuda-nvdisasm-'"${NVCC_TAG}"'
  /usr/local/cuda-'"${NVCC_TAG}"'/bin/nvcc --version | tail -2
'
docker commit "$BUILD_NVCC" "$NVCC_IMAGE" >/dev/null

docker images --format '{{.Repository}}:{{.Tag}}  {{.Size}}' | grep archlab
echo "==> done.  Sanity check:  bash tools/archlab/demo.sh"
