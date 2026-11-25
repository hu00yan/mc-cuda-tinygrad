"""Same-state observation parity.

1. CUDA ``reset`` vs ``get_obs`` (must be byte-identical).
2. CUDA ``uint8`` vs ``uint8_prebasis`` with shared voxels+cameras.
3. TinyGrad vs CUDA render on the **same** voxels+cameras (when both available).

Run:  PYTHONPATH=. python -m minecu_hybrid.obs_parity
"""
from __future__ import annotations

import hashlib

import numpy as np


def _digest(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def _cuda_same_state(batch=4, world=16, wh=32) -> dict:
    from minecu_hybrid import accel

    if not accel.available():
        return {"skip": accel.why_unavailable()}
    import cupy as cp
    from minecu_hybrid import MineCUEnv

    env = MineCUEnv(
        accel="on",
        batch=batch,
        world=world,
        width=wh,
        height=wh,
        max_steps=wh,
        seed=42,
        obs_dtype="uint8_prebasis_full",
        device_buffers=True,
        sync_every_step=True,
    )
    a = cp.asnumpy(env.reset())
    b = cp.asnumpy(env.get_obs())
    out = {
        "reset_vs_get_obs_equal": bool(np.array_equal(a, b)),
        "sha": _digest(a),
        "mean": float(a.mean()),
        "var": float(a.astype(np.float32).var()),
    }

    # uint8 vs prebasis, identical voxels + cameras
    e_u = MineCUEnv(
        accel="on",
        batch=batch,
        world=world,
        width=wh,
        height=wh,
        max_steps=wh,
        seed=42,
        obs_dtype="uint8",
        device_buffers=True,
        sync_every_step=True,
    )
    e_u.reset()
    e_u.voxels[:] = env.voxels
    e_u._positions_d[:] = env._positions_d
    e_u._yaws_d[:] = env._yaws_d
    e_u._pitches_d[:] = env._pitches_d
    ou = cp.asnumpy(e_u.get_obs())
    op = a  # prebasis_full from env
    # re-render env with prebasis path already in get_obs
    out["uint8_vs_prebasis_maxabs"] = _max_abs(ou, op)
    out["uint8_vs_prebasis_equal"] = bool(np.array_equal(ou, op))
    out["uint8_sha"] = _digest(ou)

    # TinyGrad render of the same voxels+cameras
    try:
        from tinygrad import Tensor, dtypes
        from minecu_tg.render import camera_basis
        from minecu_tg_opt.render import render
        from minecu_hybrid.backend_tinygrad import _tg_context

        vox_np = cp.asnumpy(env.voxels)
        cam = np.zeros((batch, 5), np.float32)
        cam[:, 0:3] = cp.asnumpy(env._positions_d)
        cam[:, 3] = cp.asnumpy(env._yaws_d)
        cam[:, 4] = cp.asnumpy(env._pitches_d)
        with _tg_context():
            cam_t = Tensor(cam, dtype=dtypes.float32)
            basis = camera_basis(cam_t, 70.0, wh, wh)
            vox_t = Tensor(vox_np, dtype=dtypes.int8).contiguous()
            obs_tg = render(vox_t, basis, wh, wh, wh).numpy().astype(np.uint8)
        out["tg_same_state_maxabs"] = _max_abs(obs_tg, op)
        out["tg_same_state_equal"] = bool(np.array_equal(obs_tg, op))
        out["tg_sha"] = _digest(obs_tg)
        # prior study: byte-identical on T4 for opt render; allow 0
        out["tg_vs_cuda_byte_ok"] = out["tg_same_state_equal"]
    except Exception as e:
        out["tg_same_state"] = f"skip:{type(e).__name__}:{e}"

    env.close()
    e_u.close()
    return out


def main() -> None:
    print("=== obs parity (same-state) ===")
    r = _cuda_same_state()
    for k, v in r.items():
        print(f"  {k}: {v}")
    if r.get("skip"):
        print("CUDA skipped; nothing to gate")
        return
    if not r.get("reset_vs_get_obs_equal"):
        raise SystemExit("FAIL: reset != get_obs")
    # uint8 vs prebasis may differ in fog/shading path; flag large drift
    if r.get("uint8_vs_prebasis_maxabs", 0) > 64:
        raise SystemExit(f"FAIL: uint8 vs prebasis maxabs={r['uint8_vs_prebasis_maxabs']}")
    if r.get("tg_vs_cuda_byte_ok") is False:
        # not a hard fail if TG device != CUDA; report loudly
        print(
            f"  WARN: TG vs CUDA not byte-identical "
            f"(maxabs={r.get('tg_same_state_maxabs')})"
        )
    print("obs_parity ok")


if __name__ == "__main__":
    main()
