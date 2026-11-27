"""Turn observation tensors into viewable images.

This is *not* a Minecraft Java client. Obs are low-res raycast RGB frames
(default 32×32 or 64×64). You can save them and watch a rollout — same game
state the agent sees — but there is no chunk streaming, UI, or desktop window
unless you hook one up yourself.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _as_u8(obs: np.ndarray) -> np.ndarray:
    x = np.asarray(obs)
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)
    if x.ndim == 3:
        x = x[None, ...]
    if x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError(f"expected [B,H,W,3], got {x.shape}")
    return x


def to_ppm(path: str | Path, frame: np.ndarray) -> Path:
    """Write one HxWx3 uint8 frame as binary PPM (no deps)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = np.asarray(frame)
    if f.ndim == 4:
        f = f[0]
    h, w, _ = f.shape
    raw = np.ascontiguousarray(f, dtype=np.uint8).tobytes()
    path.write_bytes(f"P6\n{w} {h}\n255\n".encode("ascii") + raw)
    return path


def mosaic(obs: np.ndarray, cols: int | None = None, scale: int = 8) -> np.ndarray:
    """Tile batch frames into one image, nearest-neighbour upscaled."""
    x = _as_u8(obs)
    B, H, W, C = x.shape
    if cols is None:
        cols = int(np.ceil(np.sqrt(B)))
    rows = int(np.ceil(B / cols))
    canvas = np.zeros((rows * H, cols * W, C), dtype=np.uint8)
    for i in range(B):
        r, c = divmod(i, cols)
        canvas[r * H : (r + 1) * H, c * W : (c + 1) * W] = x[i]
    if scale != 1:
        canvas = np.repeat(np.repeat(canvas, scale, axis=0), scale, axis=1)
    return canvas


def save_rollout(
    frames: list[np.ndarray] | np.ndarray,
    out_dir: str | Path,
    prefix: str = "frame",
    env_index: int = 0,
    scale: int = 8,
) -> list[Path]:
    """Save a list of obs (each [B,H,W,3] or [H,W,3]) as PPM sequence."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    seq = frames if isinstance(frames, list) else list(frames)
    for t, fr in enumerate(seq):
        f = _as_u8(fr)[env_index]
        if scale != 1:
            f = np.repeat(np.repeat(f, scale, axis=0), scale, axis=1)
        paths.append(to_ppm(out_dir / f"{prefix}_{t:04d}.ppm", f))
    return paths


def save_mosaic(obs: np.ndarray, path: str | Path, cols: int | None = None, scale: int = 4) -> Path:
    img = mosaic(obs, cols=cols, scale=scale)
    return to_ppm(path, img)
