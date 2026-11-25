"""CUDA accelerators — Infatoshi kernels as inline hot blocks."""
from __future__ import annotations

from ctypes import c_int8

_DLL = None
_ERR: str | None = None


def available() -> bool:
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


def sync() -> None:
    import cupy as cp

    cp.cuda.runtime.deviceSynchronize()


def generate_world(voxels, batch, wx, wy, wz, ground, seed) -> None:
    _dll().launch_generate_world(_p(voxels), batch, wx, wy, wz, ground, seed)


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


def render_cameras(voxels, cameras, obs, batch, world, width, height, max_steps, view, fov, *, float_obs: bool) -> None:
    d = _dll()
    if float_obs:
        d.launch_render(
            _p(voxels), _p(cameras), _p(obs), batch, world, width, height,
            max_steps, float(view), float(fov),
        )
    else:
        d.launch_render_uint8(
            _p(voxels), _p(cameras), _p(obs), batch, world, width, height,
            max_steps, float(view), float(fov),
        )


def render_prebasis(voxels, cameras, basis, obs, batch, world, width, height, max_steps, fov) -> None:
    d = _dll()
    d.launch_precompute_camera_basis(_p(cameras), _p(basis), batch, width, height, float(fov))
    d.launch_render_uint8_prebasis(
        _p(voxels), _p(basis), _p(obs), batch, world, width, height, max_steps
    )


def raycast_interact(voxels, positions, yaws, pitches, do_break, do_place, rewards,
                     batch, world, target_block, reward_value) -> None:
    _dll().launch_raycast_interact(
        _p(voxels), _p(positions), _p(yaws), _p(pitches),
        _p(do_break), _p(do_place), _p(rewards),
        batch, world, c_int8(target_block), float(reward_value),
    )


def fused_step(
    obs_dtype: str,
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
    """Dispatch the Infatoshi step launcher matching MineEnv obs_dtype."""
    d = _dll()
    sx, sy, sz = spawn
    cubic = wx == wy == wz
    if not cubic and obs_dtype in (
        "float32", "uint8", "uint8_minimal", "uint8_fp16",
    ):
        raise ValueError(f"{obs_dtype} step launcher needs cubic world; got {wx,wy,wz}")

    if obs_dtype == "uint8_prebasis_full":
        d.launch_step_uint8_prebasis_full(
            _p(voxels),
            _p(positions), _p(velocities), _p(yaws), _p(pitches), _p(on_ground),
            _p(buttons), _p(look),
            _p(do_reset), _p(step_count),
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
        return

    if obs_dtype in ("uint8_prebasis", "uint8_smem"):
        fn = (
            d.launch_step_uint8_smem
            if obs_dtype == "uint8_smem"
            else d.launch_step_uint8_prebasis
        )
        fn(
            _p(voxels),
            _p(positions), _p(velocities), _p(yaws), _p(pitches), _p(on_ground),
            _p(buttons), _p(look),
            _p(do_reset),
            _p(forward_in), _p(strafe_in), _p(delta_yaw_in), _p(delta_pitch_in),
            _p(jump_in), _p(do_break), _p(do_place), _p(speed_mult),
            _p(cameras), _p(basis), _p(obs), _p(rewards),
            batch, wx, wy, wz, width, height,
            max_steps, float(view_distance), float(fov),
            float(dt), float(gravity), float(walk_speed), float(sprint_mult), float(jump_vel),
            c_int8(target_block), float(reward_value),
            float(sx), float(sy), float(sz),
        )
        return

    name = {
        "float32": "launch_step",
        "uint8": "launch_step_uint8",
        "uint8_minimal": "launch_step_uint8_minimal",
        "uint8_fp16": "launch_step_uint8_fp16",
    }.get(obs_dtype)
    if not name:
        raise ValueError(f"unsupported obs_dtype {obs_dtype!r}")
    fn = getattr(d, name)
    fn(
        _p(voxels),
        _p(positions), _p(velocities), _p(yaws), _p(pitches), _p(on_ground),
        _p(buttons), _p(look),
        _p(do_reset),
        _p(forward_in), _p(strafe_in), _p(delta_yaw_in), _p(delta_pitch_in),
        _p(jump_in), _p(do_break), _p(do_place), _p(speed_mult),
        _p(cameras),
        _p(obs), _p(rewards),
        batch, wx, width, height,
        max_steps, float(view_distance), float(fov),
        float(dt), float(gravity), float(walk_speed), float(sprint_mult), float(jump_vel),
        c_int8(target_block), float(reward_value),
        float(sx), float(sy), float(sz),
    )
