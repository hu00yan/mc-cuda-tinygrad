"""Infatoshi CUDA kernels — performance floor/ceiling for NVIDIA."""
from __future__ import annotations

from ctypes import c_int8

import numpy as np

from . import lib as minecu_lib
from .config import EnvConfig

try:
    import cupy as cp
except ImportError as e:  # pragma: no cover
    raise ImportError("cuda backend needs cupy") from e


class CudaBackend:
    def __init__(self, cfg: EnvConfig):
        if cfg.world_x != cfg.world_y or cfg.world_x != cfg.world_z:
            # several launchers still take a single world_size for physics/interact;
            # full step_prebasis_full accepts wx,wy,wz for render — keep cubic for now.
            if not (cfg.world_x == cfg.world_y == cfg.world_z):
                raise ValueError("cuda backend currently requires a cubic world")
        self.cfg = cfg
        self.dll = minecu_lib.load()
        self._alloc()

    def _alloc(self) -> None:
        c = self.cfg
        B, W = c.batch, c.world_x
        H, Wd = c.height, c.width
        self.voxels = cp.zeros((B, c.world_y, c.world_x, c.world_z), dtype=cp.int8)
        self.positions = cp.zeros((B, 3), dtype=cp.float32)
        self.velocities = cp.zeros((B, 3), dtype=cp.float32)
        self.yaws = cp.zeros((B,), dtype=cp.float32)
        self.pitches = cp.zeros((B,), dtype=cp.float32)
        self.on_ground = cp.zeros((B,), dtype=cp.bool_)
        self.do_reset = cp.zeros((B,), dtype=cp.bool_)
        self.step_count = cp.zeros((B,), dtype=cp.int32)
        self.buttons = cp.zeros((B, 8), dtype=cp.int8)
        self.look = cp.zeros((B, 2), dtype=cp.float32)
        self.forward_in = cp.zeros((B,), dtype=cp.float32)
        self.strafe_in = cp.zeros((B,), dtype=cp.float32)
        self.delta_yaw_in = cp.zeros((B,), dtype=cp.float32)
        self.delta_pitch_in = cp.zeros((B,), dtype=cp.float32)
        self.jump_in = cp.zeros((B,), dtype=cp.bool_)
        self.do_break = cp.zeros((B,), dtype=cp.bool_)
        self.do_place = cp.zeros((B,), dtype=cp.bool_)
        self.speed_mult = cp.ones((B,), dtype=cp.float32)
        self.cameras = cp.zeros((B, 5), dtype=cp.float32)
        self.basis = cp.zeros((B, 14), dtype=cp.float32)
        self.obs = cp.zeros((B, H, Wd, 3), dtype=cp.uint8)
        self.rewards = cp.zeros((B,), dtype=cp.float32)

    def _p(self, a) -> int:
        return minecu_lib.ptr(a)

    def generate_world(self) -> None:
        c = self.cfg
        B = c.batch
        wx, wy, wz = c.world_x, c.world_y, c.world_z
        g = int(c.ground_height)
        self.dll.launch_generate_world(
            self._p(self.voxels), B, wx, wy, wz, g, c.seed
        )
        tx, tz = c.tree_xz()
        hx, hz = c.house_xz()
        self.dll.launch_place_tree(self._p(self.voxels), B, wx, wy, wz, tx, tz, g)
        self.dll.launch_place_house(self._p(self.voxels), B, wx, wy, wz, hx, hz, g)
        cp.cuda.runtime.deviceSynchronize()

    def reset(self, mask: np.ndarray | None = None) -> np.ndarray:
        c = self.cfg
        if mask is None:
            self.generate_world()
            self.do_reset.fill(True)
            self.step_count.fill(0)
        else:
            m = cp.asarray(mask, dtype=cp.bool_)
            if m.shape != (c.batch,):
                raise ValueError(f"mask shape {m.shape} != ({c.batch},)")
            self.do_reset[:] = m
            # world identical across envs; regen if any reset
            if bool(m.any()):
                self.generate_world()
            self.step_count = cp.where(m, 0, self.step_count)

        sx, sy, sz = c.spawn()
        self.dll.launch_reset(
            self._p(self.positions),
            self._p(self.velocities),
            self._p(self.yaws),
            self._p(self.pitches),
            self._p(self.on_ground),
            self._p(self.do_reset),
            c.batch,
            float(sx),
            float(sy),
            float(sz),
        )
        self.do_reset.fill(False)
        self.buttons.fill(0)
        self.look.fill(0)
        cp.cuda.runtime.deviceSynchronize()
        return self.get_obs()

    def _fill_cameras(self) -> None:
        self.cameras[:, 0] = self.positions[:, 0]
        self.cameras[:, 1] = self.positions[:, 1]
        self.cameras[:, 2] = self.positions[:, 2]
        self.cameras[:, 3] = self.yaws
        self.cameras[:, 4] = self.pitches

    def get_obs(self) -> np.ndarray:
        c = self.cfg
        self._fill_cameras()
        self.dll.launch_precompute_camera_basis(
            self._p(self.cameras),
            self._p(self.basis),
            c.batch,
            c.width,
            c.height,
            float(c.fov_degrees),
        )
        # launcher takes cubic world_size for prebasis
        self.dll.launch_render_uint8_prebasis(
            self._p(self.voxels),
            self._p(self.basis),
            self._p(self.obs),
            c.batch,
            c.world_x,
            c.width,
            c.height,
            c.max_steps,
        )
        cp.cuda.runtime.deviceSynchronize()
        return cp.asnumpy(self.obs)

    def step(
        self,
        buttons: np.ndarray | None = None,
        look: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = self.cfg
        if buttons is not None:
            self.buttons[:] = cp.asarray(buttons, dtype=cp.int8)
        if look is not None:
            self.look[:] = cp.asarray(look, dtype=cp.float32)

        sx, sy, sz = c.spawn()
        self.dll.launch_step_uint8_prebasis_full(
            self._p(self.voxels),
            self._p(self.positions),
            self._p(self.velocities),
            self._p(self.yaws),
            self._p(self.pitches),
            self._p(self.on_ground),
            self._p(self.buttons),
            self._p(self.look),
            self._p(self.do_reset),
            self._p(self.step_count),
            self._p(self.forward_in),
            self._p(self.strafe_in),
            self._p(self.delta_yaw_in),
            self._p(self.delta_pitch_in),
            self._p(self.jump_in),
            self._p(self.do_break),
            self._p(self.do_place),
            self._p(self.speed_mult),
            self._p(self.cameras),
            self._p(self.basis),
            self._p(self.obs),
            self._p(self.rewards),
            c.batch,
            c.world_x,
            c.world_y,
            c.world_z,
            c.width,
            c.height,
            c.max_steps,
            float(c.view_distance),
            float(c.fov_degrees),
            float(c.dt),
            float(c.gravity),
            float(c.walk_speed),
            float(c.sprint_mult),
            float(c.jump_vel),
            c_int8(c.target_block),
            float(c.reward_value),
            float(sx),
            float(sy),
            float(sz),
            int(c.episode_length),
        )
        cp.cuda.runtime.deviceSynchronize()
        dones = cp.asnumpy(self.do_reset)
        if dones.any():
            self.generate_world()
            self.do_reset.fill(False)
        return cp.asnumpy(self.obs), cp.asnumpy(self.rewards), dones

    def close(self) -> None:
        return
