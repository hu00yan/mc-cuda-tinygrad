"""ctypes bindings to libminecu.so (compiled from Infatoshi/mine.cu kernels.cu via build_lib.sh)."""
from __future__ import annotations

import ctypes
import os
from ctypes import c_float, c_int, c_int8, c_uint, c_void_p
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
            f"missing {so}; build with minecu/build_lib.sh on a CUDA host"
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

    f("launch_generate_world", None, vp, c_int, c_int, c_int, c_int, c_int, c_uint)
    f("launch_place_tree", None, vp, c_int, c_int, c_int, c_int, c_int, c_int, c_int)
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
    f("launch_precompute_camera_basis", None, vp, vp, c_int, c_int, c_int, c_float)
    f("launch_render", None, vp, vp, vp, c_int, c_int, c_int, c_int, c_int, c_float, c_float)
    f("launch_render_uint8", None, vp, vp, vp, c_int, c_int, c_int, c_int, c_int, c_float, c_float)
    f(
        "launch_render_uint8_prebasis",
        None,
        vp, vp, vp, c_int, c_int, c_int, c_int, c_int,
    )

    # Shared tail for simple uint8/float32 step launchers (no basis, no step_count)
    step_tail = (
        c_int, c_int, c_int, c_int,  # B world W H  (cubic world_size)
        c_int, c_float, c_float,
        c_float, c_float, c_float, c_float, c_float,
        c_int8, c_float,
        c_float, c_float, c_float,
    )
    step_head = (
        vp,  # voxels
        vp, vp, vp, vp, vp,  # pos vel yaw pitch on_ground
        vp, vp,  # buttons look
        vp,  # do_reset
        vp, vp, vp, vp,  # forward strafe dyaw dpitch
        vp, vp, vp, vp,  # jump break place speed
        vp,  # cameras
        vp, vp,  # obs rewards
    )
    for name in (
        "launch_step",
        "launch_step_uint8",
        "launch_step_uint8_minimal",
        "launch_step_uint8_fp16",
    ):
        f(name, None, *step_head, *step_tail)

    # prebasis: + basis, world_x/y/z instead of cubic
    pre_head = (
        vp,
        vp, vp, vp, vp, vp,
        vp, vp,
        vp,
        vp, vp, vp, vp,
        vp, vp, vp, vp,
        vp,  # cameras
        vp,  # basis
        vp, vp,
    )
    pre_tail = (
        c_int, c_int, c_int, c_int, c_int, c_int,  # B wx wy wz W H
        c_int, c_float, c_float,
        c_float, c_float, c_float, c_float, c_float,
        c_int8, c_float,
        c_float, c_float, c_float,
    )
    f("launch_step_uint8_prebasis", None, *pre_head, *pre_tail)
    f(
        "launch_step_uint8_prebasis_full",
        None,
        vp,
        vp, vp, vp, vp, vp,
        vp, vp,
        vp,  # do_reset
        vp,  # step_count
        vp, vp, vp, vp,
        vp, vp, vp, vp,
        vp,
        vp,
        vp, vp,
        *pre_tail,
        c_int,  # episode_length
    )
    # smem: same as prebasis (basis + wx wy wz)
    f("launch_step_uint8_smem", None, *pre_head, *pre_tail)


def ptr(arr) -> int:
    if hasattr(arr, "data") and hasattr(arr.data, "ptr"):
        return int(arr.data.ptr)
    raise TypeError(f"need cupy ndarray, got {type(arr)}")
