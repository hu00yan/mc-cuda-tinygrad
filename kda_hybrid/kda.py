"""Orchestrator: TinyGrad opt path + optional fla CUDA accel."""
from __future__ import annotations

from . import accel

_LAST_ACCEL = False


def available() -> bool:
    return accel.available()


def why_unavailable() -> str:
    return accel.why_unavailable()


def using_accel() -> bool:
    return _LAST_ACCEL


def kda_chunk(
    q,
    k,
    v,
    g,
    beta,
    scale=None,
    initial_state=None,
    chunk_size=64,
    accel_mode: str = "auto",
    block=None,
):
    """KDA forward. Same signature as ``kda.tiny_kda_opt.kda_chunk`` + accel.

    Returns ``(o, S)`` numpy float32.
    """
    global _LAST_ACCEL
    mode = (accel_mode or "auto").lower()
    use = False
    if mode in ("on", "1", "true", "yes", "cuda", "fla", "force"):
        if not accel.available():
            raise RuntimeError(f"accel=on but unavailable: {accel.why_unavailable()}")
        use = True
    elif mode in ("off", "0", "false", "no", "tinygrad", "tg"):
        use = False
    else:
        use = accel.available()

    _LAST_ACCEL = use
    if use:
        return accel.forward(
            q, k, v, g, beta,
            scale=scale,
            initial_state=initial_state,
            chunk_size=chunk_size,
        )

    from kda import tiny_kda_opt

    return tiny_kda_opt.kda_chunk(
        q, k, v, g, beta,
        scale=scale,
        initial_state=initial_state,
        chunk_size=chunk_size,
        block=block,
    )
