"""ctypes bindings to libminecu.so (compiled from minecu_ref/src/kernels.cu)."""
from __future__ import annotations

import ctypes
import os
from ctypes import (
    POINTER,
    c_bool,
    c_float,
    c_int,
    c_int8,
    c_uint,
    c_uint8,
    c_void_p,
)
from pathlib import Path

_LIB: ctypes.CDLL | None = None

_HERE = Path(__file__).resolve().parent
_DEFAULT_SO = _HERE / "libminecu.so"


def lib_path() -> Path:
    env = os.environ.get("MINECU_SO")
    return Path(env) if env else _DEFAULT_SO


def load(path: Path | None = None) -> ctypes.CDLL:
    global _LIB
    if _LIB is not None and path is None:
        return _LIB
    so = Path(path) if path else lib_path()
    if not so.is_file():
        raise FileNotFoundError(
            f"missing {so}; build with minecu_hybrid/build_lib.sh on a CUDA host"
        )
    dll = ctypes.CDLL(str(so))
    _bind(dll)
    _LIB = dll
    return dll


def _bind(dll: ctypes.CDLL) -> None:
    def f(name: str, restype, *argtypes):
        fn = getattr(dll, name)
        fn.restype = restype
        fn.argtypes = list(argtypes)
        return fn

    vp = c_void_p

    # voxels, B, wx, wy, wz, ground, seed
    f("launch_generate_world", None, vp, c_int, c_int, c_int, c_int, c_int, c_uint)
    # voxels, B, wx, wy, wz, tree_x, tree_z, ground
    f("launch_place_tree", None, vp, c_int, c_int, c_int, c_int, c_int, c_int, c_int)
    # voxels, B, wx, wy, wz, house_x, house_z, ground
    f("launch_place_house", None, vp, c_int, c_int, c_int, c_int, c_int, c_int, c_int)
    f("launch_reset", None, vp, vp, vp, vp, vp, vp, c_int, c_float, c_float, c_float)
    f(
        "launch_decode_actions",
        None,
        vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, c_int, c_float,
    )
    f(
        "launch_physics_v2",
        None,
        vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp,
        c_int, c_int, c_float, c_float, c_float, c_float,
    )
    f(
        "launch_raycast_interact",
        None,
        vp, vp, vp, vp, vp, vp, vp, c_int, c_int, c_int8, c_float,
    )
    f(
        "launch_precompute_camera_basis",
        None,
        vp, vp, c_int, c_int, c_int, c_float,
    )
    f(
        "launch_render_uint8_prebasis",
        None,
        vp, vp, vp, c_int, c_int, c_int, c_int, c_int,
    )
    f(
        "launch_step_uint8_prebasis",
        None,
        vp,  # voxels
        vp, vp, vp, vp, vp,  # pos vel yaw pitch on_ground
        vp, vp,  # buttons look
        vp,  # do_reset
        vp, vp, vp, vp,  # forward strafe dyaw dpitch
        vp, vp, vp, vp,  # jump break place speed
        vp,  # cameras
        vp,  # basis
        vp, vp,  # obs rewards
        c_int, c_int, c_int, c_int, c_int, c_int,  # B wx wy wz W H
        c_int, c_float, c_float,  # max_steps view fov
        c_float, c_float, c_float, c_float, c_float,  # dt g walk sprint jump
        c_int8, c_float,  # target reward
        c_float, c_float, c_float,  # spawn
    )
    f(
        "launch_step_uint8_prebasis_full",
        None,
        vp,  # voxels
        vp, vp, vp, vp, vp,  # pos vel yaw pitch on_ground
        vp, vp,  # buttons look
        vp,  # do_reset (writable)
        vp,  # step_count
        vp, vp, vp, vp,  # forward strafe dyaw dpitch
        vp, vp, vp, vp,  # jump break place speed
        vp,  # cameras
        vp,  # basis
        vp, vp,  # obs rewards
        c_int, c_int, c_int, c_int, c_int, c_int,
        c_int, c_float, c_float,
        c_float, c_float, c_float, c_float, c_float,
        c_int8, c_float,
        c_float, c_float, c_float,
        c_int,  # episode_length
    )


def ptr(arr) -> int:
    """Device pointer from a cupy ndarray (or object with .data.ptr)."""
    if hasattr(arr, "data") and hasattr(arr.data, "ptr"):
        return int(arr.data.ptr)
    raise TypeError(f"need cupy ndarray, got {type(arr)}")
