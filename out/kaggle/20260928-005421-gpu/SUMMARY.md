# mc-cuda-tinygrad on Kaggle

- runner kind: **GPU**
- accelerator: Tesla T4, 15360 MiB, 7.5
Tesla T4, 15360 MiB, 7.5
- payload sha256: 41fb3e10a6eadb4bbec2c9c0cbd12c6b8b563d117ccc2d67324cc609550d805d
- versions: fla-core 0.5.2, flash-linear-attention 0.5.2, numpy 2.0.2, tinygrad 0.14.0, torch 2.10.0+cu128, triton 3.6.0
- camera C reference: enabled
- C compiler for tinygrad's CPU backend: /usr/bin/gcc
- battery wall time: 2176s

## results

| item | verdict | seconds | note |
|---|---|---|---|
| preflight | ok | 56.8 | is this box able to run the probes |
| test_ref | ok | 0.4 | float64 reference, 17 tests |
| test_tiny | ok | 138.8 | TinyGrad port vs the reference |
| codegen_diff | ok | 2.3 | generated code vs hand-written Triton |
| kda_vs_fla | TIMEOUT | 900.5 | the N-vs-N comparison |
| perf_kda | ok | 333.3 | kernel count, GPU/wall split |
| replay_kda | ok | 361.7 | graph replay: the python tax |
| cost | ok | 0.1 | analytic FLOP/byte model |
| fp32_correctness | ok | 2.4 | is float32 correctly rounded |
| divide_precision | rc=1 | 1.6 | the a*(1/b) rewrite |
| gather_floor | ok | 19.0 | O(source) floor on a gather |
| gather_scale | ok | 10.4 | indexed load O(out) or O(src) |
| real_gather | ok | 2.1 | the custom_kernel attempt |
| oc_check | rc=2 | 3.4 | camera vs the C reference, block-id agreement |
| oc_report | rc=2 | 1.1 | every number in notes/01 |
| oc_port_only | ok | 254.0 | the TinyGrad camera port, timed, against the same ref |

## counts

```
{
  "ok": 12,
  "TIMEOUT": 1,
  "rc": 3
}
```
