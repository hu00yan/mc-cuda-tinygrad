"""KDA: TinyGrad-shaped API + optional fla/CUDA inline accelerators."""

from .kda import available, kda_chunk, using_accel, why_unavailable

__all__ = ["kda_chunk", "available", "why_unavailable", "using_accel"]
