"""Structural + optional CUDA parity audit vs Infatoshi MineEnv surface.

Run:  PYTHONPATH=. python -m minecu_hybrid.parity_audit
"""
from __future__ import annotations

import inspect
import time

from minecu_hybrid import (
    BUTTON_DIM,
    BUTTON_NAMES,
    DEFAULT_ENV,
    FAST_ENV,
    LOOK_DIM,
    QUALITY_ENV,
    BlockType,
    EnvConfig,
    MineCUEnv,
    TrainConfig,
    accel,
)


def _ok(name: str, cond: bool, detail: str = "") -> bool:
    mark = "PASS" if cond else "FAIL"
    extra = f" — {detail}" if detail else ""
    print(f"  [{mark}] {name}{extra}")
    return cond


def audit_api() -> list[str]:
    fails: list[str] = []
    print("=== API surface (MineEnv-compatible) ===")

    # Construction aliases
    e = MineCUEnv(
        batch_size=4,
        world_size=8,
        render_width=16,
        render_height=16,
        max_steps=8,
        episode_length=None,
        accel="off",
        obs_dtype="uint8",
    )
    if not _ok("aliases batch_size/world_size/render_*", e.batch_size == 4 and e.world_size == 8
               and e.render_width == 16 and e.render_height == 16):
        fails.append("aliases")
    if not _ok("episode_length None→0", e.cfg.episode_length == 0):
        fails.append("episode_length")
    if not _ok("button/look dims", e.button_dim == BUTTON_DIM == 8 and e.look_dim == LOOK_DIM == 2):
        fails.append("dims")
    if not _ok("button_names", len(BUTTON_NAMES) == 8):
        fails.append("button_names")
    if not _ok("BlockType.OAKLOG", int(BlockType.OAKLOG) == 3):
        fails.append("BlockType")
    if not _ok("TrainConfig fields", set(TrainConfig.__dataclass_fields__)
               >= {"lr", "gamma", "total_steps", "log_interval", "hidden_dim"}):
        fails.append("TrainConfig")
    if not _ok("preset configs", isinstance(DEFAULT_ENV, EnvConfig)
               and FAST_ENV.batch == 32768 and QUALITY_ENV.width == 128):
        fails.append("presets")
    if not _ok("upstream()", EnvConfig.upstream().batch == 4096
               and EnvConfig.upstream().device_buffers is True):
        fails.append("upstream")
    if not _ok("tree_xz = spawn_z+2", e.cfg.tree_xz() == (4, 4 + 2)):
        # spawn for world=8: (4, ground+2, 4), tree (4, 6)
        fails.append("tree_xz")
    if not _ok("obs_dtype set", all(
        EnvConfig(obs_dtype=d, batch=1, world=8, width=8, height=8).obs_dtype == d
        for d in (
            "float32", "uint8", "uint8_minimal", "uint8_fp16",
            "uint8_prebasis", "uint8_prebasis_full", "uint8_smem",
        )
    )):
        fails.append("obs_dtypes")

    # TinyGrad reset/step shape
    obs = e.reset()
    if not _ok("reset shape", obs.shape == (4, 16, 16, 3)):
        fails.append("reset_shape")
    buttons, look = e.zero_actions()
    buttons = buttons.copy()
    buttons[:, 0] = 1
    obs, rew, done = e.step(buttons, look)
    if not _ok("step returns", obs.shape == (4, 16, 16, 3) and rew.shape == (4,) and done.shape == (4,)):
        fails.append("step_returns")
    if not _ok("spaces", e.observation_space is not None and e.action_space is not None):
        fails.append("spaces")

    # Method inventory vs MineEnv
    need = {"reset", "step", "get_obs", "close", "from_config"}
    have = {n for n, _ in inspect.getmembers(MineCUEnv, predicate=inspect.isfunction)} | set(
        dir(MineCUEnv)
    )
    if not _ok("MineEnv methods present", need <= have, str(sorted(need))):
        fails.append("methods")
    e.close()
    return fails


def audit_cuda() -> list[str]:
    fails: list[str] = []
    print("=== CUDA accel ===")
    if not accel.available():
        print(f"  [SKIP] CUDA unavailable: {accel.why_unavailable()}")
        return ["cuda_unavailable"]

    for dtype in ("uint8_prebasis_full", "uint8", "uint8_prebasis"):
        env = MineCUEnv(
            accel="on",
            batch=32,
            world=16,
            width=32,
            height=32,
            max_steps=32,
            obs_dtype=dtype,
            device_buffers=True,
            sync_every_step=False,
            episode_length=8,
        )
        obs = env.reset()
        import cupy as cp

        if not _ok(f"{dtype} reset device", isinstance(obs, cp.ndarray) and obs.shape == (32, 32, 32, 3)):
            fails.append(f"{dtype}_reset")
            env.close()
            continue
        buttons, look = env.zero_actions()
        buttons[:, 0] = 1
        # drive past episode_length
        saw_done = False
        t0 = time.perf_counter()
        n = 16
        for _ in range(n):
            obs, rew, done = env.step(buttons, look)
            if bool(done.any()):
                saw_done = True
        cp.cuda.runtime.deviceSynchronize()
        dt = time.perf_counter() - t0
        sps = 32 * n / dt
        if not _ok(f"{dtype} episode dones", saw_done, f"{sps:.0f} env-steps/s"):
            fails.append(f"{dtype}_episode")
        if not _ok(f"{dtype} obs variance", float(cp.asnumpy(obs).astype("float32").var()) > 1.0):
            fails.append(f"{dtype}_var")
        env.close()

    # Host-buffer path still works
    env = MineCUEnv(accel="on", batch=16, world=16, width=32, height=32, device_buffers=False)
    obs = env.reset()
    import numpy as np

    if not _ok("host buffers", isinstance(obs, np.ndarray)):
        fails.append("host")
    env.close()
    return fails


def main() -> None:
    fails = audit_api()
    fails += audit_cuda()
    print("=== summary ===")
    real = [f for f in fails if f != "cuda_unavailable"]
    if real:
        raise SystemExit(f"FAILED: {real}")
    if "cuda_unavailable" in fails:
        print("API ok; CUDA skipped (no GPU on this host)")
    else:
        print("all checks passed")


if __name__ == "__main__":
    main()
