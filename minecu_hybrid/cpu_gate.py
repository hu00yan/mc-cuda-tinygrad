"""No-GPU correctness gate for minecu_hybrid.

Trust chain (does not need NVIDIA):
  1. ``kernels.cu`` byte-pinned to Infatoshi (``pin_kernels``).
  2. ``dda_model`` is a float32 transcription of that render kernel.
  3. TinyGrad opt render must match ``dda_model`` byte-for-byte on a fixed
     world+camera (same contract previously shown vs CUDA on T4).
  4. ``MineCUEnv(accel="off")`` reset/step stays non-flat and shape-stable.

Run:  PYTHONPATH=. DEV=CPU python -m minecu_hybrid.cpu_gate
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np

os.environ.setdefault("DEV", "CPU")
os.environ.setdefault("PARALLEL", "0")
os.environ.setdefault("MINECU_TG_DEV", os.environ.get("MINECU_TG_DEV", "CPU"))


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def _check(ok: bool, name: str, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    extra = f" — {detail}" if detail else ""
    print(f"  [{mark}] {name}{extra}")
    if not ok:
        raise SystemExit(f"FAIL: {name}")


def gate_pin() -> None:
    print("=== 1. pin kernels.cu ===")
    from minecu_hybrid.pin_kernels import main as pin_main

    pin_main()


def gate_tg_vs_dda() -> None:
    """TinyGrad opt == dda_model on identical voxels+camera."""
    print("=== 2. TG opt vs dda_model (same state) ===")
    from minecu_ref.bench.dda_model import build_world, camera_basis, render as dda_render
    from minecu_tg.render import camera_basis as tg_basis
    from minecu_tg_opt.render import render as tg_render
    from minecu_hybrid.backend_tinygrad import _tg_context
    from tinygrad import Tensor, dtypes

    # Same on-surface config as minecu_tg_opt/verify_opt.py: camera y is ground;
    # TG camera_basis adds eye height 1.6 (matches kernels.cu precompute).
    world, wh, max_steps = 16, 24, 24
    ground = world // 4
    vox = build_world(world, ground)
    yaw, pitch, fov = 0.0, -0.2, 70.0
    cam = np.array(
        [[world / 2.0, float(ground), world / 2.0, yaw, pitch]], dtype=np.float32
    )
    eye_h = 1.6
    eye = (float(cam[0, 0]), float(cam[0, 1]) + eye_h, float(cam[0, 2]))
    basis = camera_basis(yaw, pitch, fov, wh, wh)
    img_dda, _ = dda_render(
        vox, eye, basis, wh, wh, max_steps, pitch=pitch, yaw=yaw, variant="prebasis"
    )

    with _tg_context():
        cam_t = Tensor(cam, dtype=dtypes.float32)
        basis_t = tg_basis(cam_t, fov, wh, wh)
        vox_t = Tensor(
            np.ascontiguousarray(vox.reshape(1, world, world, world)),
            dtype=dtypes.int8,
        ).contiguous()
        img_tg = tg_render(vox_t, basis_t, wh, wh, max_steps).numpy()[0].astype(np.uint8)

    equal = np.array_equal(img_tg, img_dda)
    maxabs = float(np.max(np.abs(img_tg.astype(np.int16) - img_dda.astype(np.int16))))
    _check(
        equal,
        "tg_opt == dda_model uint8",
        f"maxabs={maxabs} sha_tg={_sha(img_tg)} sha_dda={_sha(img_dda)}",
    )


def gate_hybrid_off() -> None:
    print("=== 3. MineCUEnv(accel=off) smoke ===")
    from minecu_hybrid import MineCUEnv

    env = MineCUEnv(
        accel="off",
        batch=2,
        world=16,
        width=16,
        height=16,
        max_steps=16,
        seed=0,
        episode_length=8,
    )
    obs0 = env.reset()
    _check(obs0.shape == (2, 16, 16, 3), "reset shape", str(obs0.shape))
    _check(float(obs0.astype(np.float32).var()) > 1.0, "obs not flat", f"var={obs0.var():.1f}")
    buttons, look = env.zero_actions()
    buttons = buttons.copy()
    buttons[:, 0] = 1
    for t in range(4):
        obs, rew, done = env.step(buttons, look)
    _check(obs.shape == (2, 16, 16, 3), "step shape")
    _check(rew.shape == (2,) and done.shape == (2,), "rew/done shape")
    print(f"  trajectory sha={_sha(obs)} rew_sum={float(rew.sum()):.1f}")
    env.close()


def main() -> None:
    print("minecu_hybrid cpu_gate — no NVIDIA required")
    gate_pin()
    gate_tg_vs_dda()
    gate_hybrid_off()
    print("CPU_GATE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
