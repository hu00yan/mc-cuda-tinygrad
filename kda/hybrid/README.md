# kda/hybrid — TinyGrad KDA + fla CUDA inline

Same pattern as `minecu`: one Python API, CUDA/Triton (via
`flash-linear-attention` / `flash_kda`) as the hot path when present.

```python
from kda.hybrid import kda_chunk, available

o, S = kda_chunk(q, k, v, g, beta, chunk_size=64, accel_mode="auto")
```

```bash
PYTHONPATH=. python -m kda.hybrid.parity
```
