# mc-cuda-tinygrad on Kaggle

- runner kind: **GPU**
- accelerator: Tesla T4, 15360 MiB, 7.5
Tesla T4, 15360 MiB, 7.5
- payload sha256: 53ab4a39448a8bcbcc29e6fe16214c681f48b7010b1bb50ee3734e3d9a8f8c58
- versions: fla-core 0.5.2, flash-linear-attention 0.5.2, numpy 2.0.2, tinygrad 0.14.0, torch 2.10.0+cu128, triton 3.6.0
- camera C reference: enabled
- C compiler for tinygrad's CPU backend: /usr/bin/clang
- battery wall time: 3158s

## results

| item | verdict | seconds | note |
|---|---|---|---|
| preflight | ok | 53.9 | is this box able to run the probes |
| test_ref | ok | 0.4 | float64 reference, 17 tests |
| test_tiny | ok | 131.8 | TinyGrad port vs the reference |
| codegen_diff | ok | 2.1 | generated code vs hand-written Triton |
| kda_vs_fla | TIMEOUT | 1800.7 | the N-vs-N comparison |
| perf_kda | ok | 317.6 | kernel count, GPU/wall split |
| replay_kda | ok | 349.5 | graph replay: the python tax |
| cost | ok | 0.1 | analytic FLOP/byte model |
| fp32_correctness | ok | 2.3 | is float32 correctly rounded |
| divide_precision | ok | 48.9 | the a*(1/b) rewrite |
| gather_floor | ok | 18.5 | O(source) floor on a gather |
| gather_scale | ok | 10.2 | indexed load O(out) or O(src) |
| real_gather | ok | 2.0 | the custom_kernel attempt |
| oc_check | ok | 184.3 | camera vs the C reference, block-id agreement |
| oc_report | rc=2 | 168.4 | every number in notes/01 |
| oc_port_only | skip | 0.0 | out of the battery time budget |

## counts

```
{
  "ok": 13,
  "TIMEOUT": 1,
  "rc": 1,
  "skip": 1
}
```
