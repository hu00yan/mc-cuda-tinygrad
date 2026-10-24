#!/usr/bin/env bash
# One-shot: discover the exact layout of the CUDA pip wheels inside the image.
# Kept as a script so it is reproducible rather than a one-off interactive shell.
set -eux
SP=/usr/local/lib/python3.12/dist-packages
echo "=== nvidia top level ==="
ls "$SP/nvidia"
echo "=== cuda_nvcc tree (depth 2) ==="
find "$SP/nvidia/cuda_nvcc" -maxdepth 2 | head -40
echo "=== anything that looks like a compiler driver ==="
find "$SP/nvidia" -maxdepth 4 \( -name 'nvcc*' -o -name 'ptxas*' -o -name 'cicc*' -o -name 'cudafe*' -o -name 'nvlink*' -o -name 'fatbinary*' \) | head -20
