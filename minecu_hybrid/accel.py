"""CUDA accelerators — the 'inline assembly' for TinyGrad-shaped code.

These are Infatoshi kernels you call from the Python/TinyGrad loop when the
automatic path is too slow. Same role as ``asm`` inside a C function: the
surrounding program stays high-level; only the hot block is native.

Usage from the env (not a separate product)::

    if accel.available():
        accel.render_prebasis(...)   # instead of minecu_tg_opt.render
        accel.fused_step(...)        # instead of decode+physics+… in Python
"""
from __future__ import annotations

from ctypes import c_int8
from pathlib import Path

_DLL = None
_ERR: str | None = None


def available() -> bool:
    """True when libminecu.so loads (NVIDIA host after build_lib.sh)."""
    global _DLL, _ERR
    if _DLL is not None:
        return True
    if _ERR is not None:
        return False
    try:
        from . import lib as minecu_lib

        if not minecu_lib.lib_path().is_file():
            _ERR = f"missing {minecu_lib.lib_path()}"
            return False
        _DLL = minecu_lib.load()
        return True
    except Exception as e:  # pragma: no cover
        _ERR = str(e)
        return False


def why_unavailable() -> str:
    available()
    return _ERR or "ok"


def _dll():
    if not available():
        raise RuntimeError(f"CUDA accel unavailable: {why_unavailable()}")
    return _DLL


def _p(arr) -> int:
    from .lib import ptr

    return ptr(arr)


def generate_world(voxels, batch: int, wx: int, wy: int, wz: int, ground: int, seed: int) -> None:
    d = _dll()
    d.launch_generate_world(_p(voxels), batch, wx, wy, wz, ground, seed)


def place_tree(voxels, batch, wx, wy, wz, tx, tz, ground) -> None:
    _dll().launch_place_tree(_p(voxels), batch, wx, wy, wz, int(tx), int(tz), ground)


def place_house(voxels, batch, wx, wy, wz, hx, hz, ground) -> None:
    _dll().launch_place_house(_p(voxels), batch, wx, wy, wz, int(hx), int(hz), ground)


def reset_agents(positions, velocities, yaws, pitches, on_ground, do_reset, batch, spawn) -> None:
    sx, sy, sz = spawn
    _dll().launch_reset(
        _p(positions), _p(velocities), _p(yaws), _p(pitches), _p(on_ground), _p(do_reset),
        batch, float(sx), float(sy), float(sz),
    )


def render_prebasis(voxels, cameras, basis, obs, batch, world, width, height, max_steps, fov) -> None:
    """Inline render: cameras → basis → uint8 obs (Infatoshi prebasis path)."""
    d = _dll()
    d.launch_precompute_camera_basis(_p(cameras), _p(basis), batch, width, height, float(fov))
    d.launch_render_uint8_prebasis(
        _p(voxels), _p(basis), _p(obs), batch, world, width, height, max_steps
    )


def fused_step(
    voxels, positions, velocities, yaws, pitches, on_ground,
    buttons, look, do_reset, step_count,
    forward_in, strafe_in, delta_yaw_in, delta_pitch_in,
    jump_in, do_break, do_place, speed_mult,
    cameras, basis, obs, rewards,
    *,
    batch, wx, wy, wz, width, height, max_steps, view_distance, fov,
    dt, gravity, walk_speed, sprint_mult, jump_vel,
    target_block, reward_value, spawn, episode_length,
) -> None:
    """One Infatoshi step graph — decode→physics→interact→reset→render."""
    sx, sy, sz = spawn
    _dll().launch_step_uint8_prebasis_full(
        _p(voxels),
        _p(positions), _p(velocities), _p(yaws), _p(pitches), _p(on_ground),
        _p(buttons), _p(look), _p(do_reset), _p(step_count),
        _p(forward_in), _p(strafe_in), _p(delta_yaw_in), _p(delta_pitch_in),
        _p(jump_in), _p(do_break), _p(do_place), _p(speed_mult),
        _p(cameras), _p(basis), _p(obs), _p(rewards),
        batch, wx, wy, wz, width, height,
        max_steps, float(view_distance), float(fov),
        float(dt), float(gravity), float(walk_speed), float(sprint_mult), float(jump_vel),
        c_int8(target_block), float(reward_value),
        float(sx), float(sy), float(sz),
        int(episode_length),
    )


def sync() -> None:
    import cupy as cp

    cp.cuda.runtime.deviceSynchronize()
