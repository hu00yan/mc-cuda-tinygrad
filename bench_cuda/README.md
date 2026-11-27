# bench_cuda — measure Infatoshi/mine.cu (external)

This directory is **our** harness. The CUDA kernels themselves live in
[Infatoshi/mine.cu](https://github.com/Infatoshi/mine.cu) and are resolved by
`minecu.upstream` (env checkout or `.deps/` fetch + sha256 pin).

```bash
PYTHONPATH=. python -m minecu.upstream          # fetch pin into .deps/
./bench_cuda/build.sh sm_75                     # needs docker + nvcc image
```

`bench.cu` is a standalone timing driver; it is not part of upstream.
