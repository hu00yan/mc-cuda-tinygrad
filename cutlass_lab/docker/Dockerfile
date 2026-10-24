# Lean CUTLASS build image.
#
# WHY NOT THE pip ROUTE (documented because it was tried first and failed):
#   The suggested shortcut was `pip install nvidia-cuda-nvcc-cu12`. In version
#   12.9.86 that wheel contains ONLY:
#       nvidia/cuda_nvcc/bin/ptxas
#       nvidia/cuda_nvcc/nvvm/{include,lib64,libdevice}
#       nvidia/cuda_nvcc/include/{crt,fatbinary_section.h,nvPTXCompiler.h}
#   There is no `nvcc` host driver in it, so there is no way to invoke the
#   compiler. Verified by `docker/probe_layout.sh`. A compiler that is not
#   invoked cannot compile, so the pip route is unusable for this task as
#   written. (ptxas alone is enough only to assemble hand-written PTX.)
#
# WHY NOT nvidia/cuda DEVEL IMAGE:
#   multi-GB; its pull previously timed out at 10 minutes in this environment.
#
# WHAT THIS DOES INSTEAD:
#   Ubuntu 24.04 + NVIDIA's own apt repository, with --no-install-recommends and
#   only the four packages CUTLASS actually needs. This is the same lean idea
#   (a few hundred MB, not a few GB) but it goes through apt instead of pip, and
#   apt's `cuda-nvcc-12-9` does ship the nvcc driver.
#
# ARCH: must be linux/amd64 to match the x86_64 wheels and binaries in the repo.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
      ca-certificates curl gnupg \
      build-essential ninja-build python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

# NVIDIA apt repository. The keyring is distributed as a .deb in current releases
# and as a bare .gpg in older ones; accept either.
RUN set -eux; \
    curl -fsSL -o /tmp/cuda-keyring.deb \
      https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb; \
    if dpkg-deb --info /tmp/cuda-keyring.deb >/dev/null 2>&1; then \
      dpkg -i /tmp/cuda-keyring.deb >/dev/null; \
    else \
      mkdir -p /usr/share/keyrings; \
      cp /tmp/cuda-keyring.deb /usr/share/keyrings/cuda-archive-keyring.gpg; \
    fi; \
    echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" \
      > /etc/apt/sources.list.d/cuda.list

# The minimum set for CUTLASS: the compiler driver + cicc, the CUDA runtime
# headers/library, and CCCL (libcu++/thrust/cub, which CUTLASS includes).
# cuda-nvcc pulls cuda-nvvm and cuda-crt as dependencies.
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
      cuda-nvcc-12-9 \
      cuda-cudart-dev-12-9 \
      cuda-cccl-12-9 \
      cuda-cuobjdump-12-9 \
      cuda-nvdisasm-12-9 \
    && rm -rf /var/lib/apt/lists/*

# Debian alternatives put nvcc on PATH via /usr/local/cuda-12.9/bin; make the
# conventional CUDA_HOME work so CUTLASS's own scripts and our build.sh agree.
RUN set -eux; \
    ln -sfn /usr/local/cuda-12.9 /usr/local/cuda; \
    for b in nvcc ptxas nvdisasm cuobjdump nvlink fatbinary cudafe++ cicc; do \
      if [ -x "/usr/local/cuda/bin/$b" ]; then \
        ln -sf "/usr/local/cuda/bin/$b" "/usr/local/bin/$b"; \
      else \
        echo "note: $b not in cuda-nvcc; expected to arrive via a sibling package"; \
      fi; \
    done

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH

# Fail the build, not the experiment, if the toolchain is incomplete.
RUN nvcc --version \
 && ptxas --version | tail -2 \
 && cuobjdump --version | tail -2 \
 && test -f /usr/local/cuda/include/cuda_runtime.h \
 && test -d /usr/local/cuda/include/cuda \
 && test -d /usr/local/cuda/include/cub \
 && test -f /usr/local/cuda/lib64/libcudart.so \
 && echo "TOOLCHAIN OK" && du -sh /usr/local/cuda
