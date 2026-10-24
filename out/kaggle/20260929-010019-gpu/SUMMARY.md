# mc-cuda-tinygrad on Kaggle

- runner kind: **GPU**
- accelerator: Tesla T4, 15360 MiB, 7.5
Tesla T4, 15360 MiB, 7.5
- payload sha256: 7753e2c2303574e9f1906999836898f27ebc6101835931b8a2ed8339f7edc10a
- versions: fla-core 0.5.2, flash-linear-attention 0.5.2, numpy 2.0.2, tinygrad 0.14.0, torch 2.10.0+cu128, triton 3.6.0
- camera C reference: enabled
- C compiler for tinygrad's CPU backend: /usr/bin/clang
- battery wall time: 134s

## results

| item | verdict | seconds | note |
|---|---|---|---|
| preflight | ok | 58.2 | is this box able to run the probes |
| test_ref | skip | 0.0 | not in BATTERY_ONLY |
| test_tiny | skip | 0.0 | not in BATTERY_ONLY |
| codegen_diff | skip | 0.0 | not in BATTERY_ONLY |
| kda_vs_fla | skip | 0.0 | not in BATTERY_ONLY |
| perf_kda | skip | 0.0 | not in BATTERY_ONLY |
| replay_kda | skip | 0.0 | not in BATTERY_ONLY |
| cost | skip | 0.0 | not in BATTERY_ONLY |
| fp32_correctness | skip | 0.0 | not in BATTERY_ONLY |
| divide_precision | skip | 0.0 | not in BATTERY_ONLY |
| gather_floor | skip | 0.0 | not in BATTERY_ONLY |
| gather_scale | skip | 0.0 | not in BATTERY_ONLY |
| real_gather | skip | 0.0 | not in BATTERY_ONLY |
| oc_check | skip | 0.0 | not in BATTERY_ONLY |
| oc_report | skip | 0.0 | not in BATTERY_ONLY |
| oc_port_only | skip | 0.0 | run_oc.py needs ref.bin, which only the C reference makes |

## counts

```
{
  "ok": 1,
  "skip": 15
}
```
