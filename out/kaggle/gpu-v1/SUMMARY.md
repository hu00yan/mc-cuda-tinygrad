# mc-cuda-tinygrad on Kaggle

- runner kind: **GPU**
- accelerator: Tesla T4, 15360 MiB, 7.5
Tesla T4, 15360 MiB, 7.5
- payload sha256: n/a
- versions: fla-core 0.5.2, flash-linear-attention 0.5.2, numpy 2.0.2, tinygrad 0.14.0, torch 2.10.0+cu128, triton 3.6.0
- camera C reference: enabled
- battery wall time: 40s

## results

| item | verdict | seconds | note |
|---|---|---|---|
| preflight | skip | 0.0 | payload has no remote/preflight.py |
| test_ref | skip | 0.0 | payload has no test_ref.py |
| test_tiny | skip | 0.0 | payload has no test_tiny.py |
| codegen_diff | skip | 0.0 | payload has no codegen_diff.py |
| perf_kda | skip | 0.0 | payload has no perf_kda.py |
| replay_kda | skip | 0.0 | payload has no replay_kda.py |
| cost | skip | 0.0 | payload has no cost.py |
| kda_vs_fla | skip | 0.0 | payload has no kda_vs_fla.py |
| fp32_correctness | skip | 0.0 | payload has no probes/fp32_correctness.py |
| divide_precision | skip | 0.0 | payload has no probes/divide_precision.py |
| gather_floor | skip | 0.0 | payload has no probes/gather_floor.py |
| gather_scale | skip | 0.0 | payload has no probes/gather_scale.py |
| real_gather | skip | 0.0 | payload has no probes/real_gather.py |
| oc_check | skip | 0.0 | camera vs the C reference, block-id agreement |
| oc_report | skip | 0.0 | every number in notes/01 |

## counts

```
{
  "skip": 15
}
```
