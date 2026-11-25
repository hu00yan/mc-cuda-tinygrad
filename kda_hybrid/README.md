# kda_hybrid — TinyGrad KDA + fla CUDA inline

Same pattern as `minecu_hybrid`: one Python API, CUDA/Triton (via
`flash-linear-attention` / `flash_kda`) as the hot path when present.

```python
from kda_hybrid import kda_chunk, available

o, S = kda_chunk(q, k, v, g, beta, chunk_size=64, accel_mode="auto")
```

```bash
PYTHONPATH=. python -m kda_hybrid.parity
```
