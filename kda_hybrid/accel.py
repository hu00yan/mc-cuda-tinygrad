"""CUDA / Triton accelerators for KDA — fla (and flash_kda via fla) as inline hot path."""
from __future__ import annotations

_ERR: str | None = None
_READY: bool | None = None
_BACKEND: str | None = None


def available() -> bool:
    global _READY, _ERR, _BACKEND
    if _READY is not None:
        return _READY
    try:
        import torch

        if not torch.cuda.is_available():
            _ERR = "torch.cuda unavailable"
            _READY = False
            return False
        from fla.ops.kda import chunk_kda  # noqa: F401

        _BACKEND = "fla.ops.kda.chunk_kda"
        try:
            import flash_kda  # noqa: F401

            _BACKEND = "fla→flash_kda"
        except Exception:
            pass
        _READY = True
        _ERR = None
        return True
    except Exception as e:  # pragma: no cover
        _ERR = f"{type(e).__name__}: {e}"
        _READY = False
        return False


def why_unavailable() -> str:
    available()
    return _ERR or "ok"


def backend_name() -> str:
    available()
    return _BACKEND or "none"


def forward(q, k, v, g, beta, *, scale=None, initial_state=None, chunk_size=64):
    """Run fla ``chunk_kda``; return ``(o, S)`` numpy float32.

    Passes raw inputs like ``kda/kda_vs_fla.run_fla`` — fla applies scale/GVA.
    """
    import numpy as np
    import torch
    from fla.ops.kda import chunk_kda

    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    g = np.asarray(g, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)
    B, T, H, Kdim = q.shape
    HV, V = v.shape[2], v.shape[3]
    if scale is None:
        scale = Kdim ** -0.5

    def t(x):
        return torch.tensor(np.ascontiguousarray(x), dtype=torch.float32, device="cuda")

    kw = dict(
        scale=float(scale),
        use_qk_l2norm_in_kernel=False,
        output_final_state=True,
    )
    if initial_state is not None:
        kw["initial_state"] = t(initial_state)

    with torch.no_grad():
        o, S = chunk_kda(t(q), t(k), t(v), t(g), t(beta), **kw)
    o_np = o.float().detach().cpu().numpy()
    if S is None:
        S_np = np.zeros((B, HV, Kdim, V), dtype=np.float32)
    else:
        S_np = S.float().detach().cpu().numpy()
    return o_np, S_np
