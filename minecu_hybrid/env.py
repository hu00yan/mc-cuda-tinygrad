"""Gym-style batched MineCU env backed by Infatoshi CUDA step kernels."""
from __future__ import annotations

from ctypes import c_int8
from dataclasses import dataclass

import numpy as np

from . import lib as minecu_lib

try:
    import cupy as cp
except ImportError as e:  # pragma: no cover
    raise ImportError("minecu_hybrid requires cupy on the GPU host") from e


@dataclass
class EnvConfig:
    batch: int = 256
    world: int = 16
    width: int = 32
    height: int = 32
    max_steps: int = 32
    view_distance: float = 32.0
    fov_degrees: float = 70.0
    dt: float = 0.05
    gravity: float = -20.0
    walk_speed: float = 4.0
    sprint_mult: float = 1.5
    jump_vel: float = 8.0
    target_block: int = 3  # OAKLOG
    reward_value: float = 1.0
    episode_length: int = 0  # 0 = unlimited
    seed: int = 42


class MineCUEnv:
    """Batched Minecraft-toy env: CUDA for the whole step, Python for worldgen."""

    def __init__(self, **kwargs):
        self.cfg = EnvConfig(**{k: v for k, v in kwargs.items() if k in EnvConfig.__dataclass_fields__})
        self.dll = minecu_lib.load()
        self._alloc()
        g = self.cfg.world // 4
        self.ground = g
        self.spawn = (self.cfg.world / 2.0, float(g + 2), self.cfg.world / 2.0)
        self.tree_xz = (self.cfg.world // 2, self.cfg.world // 2 + 3)
        self.house_xz = (self.cfg.world // 2 + 6, self.cfg.world // 2 - 2)

    def _alloc(self) -> None:
        c = self.cfg
        B, W = c.batch, c.world
        H, Wd = c.height, c.width
        self.voxels = cp.zeros((B, W, W, W), dtype=cp.int8)
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
        B, W = c.batch, c.world
        self.dll.launch_generate_world(
            self._p(self.voxels), B, W, W, W, self.ground, c.seed
        )
        tx, tz = self.tree_xz
        hx, hz = self.house_xz
        self.dll.launch_place_tree(
            self._p(self.voxels), B, W, W, W, int(tx), int(tz), self.ground
        )
        self.dll.launch_place_house(
            self._p(self.voxels), B, W, W, W, int(hx), int(hz), self.ground
        )
        cp.cuda.runtime.deviceSynchronize()

    def reset(self) -> np.ndarray:
        """Reset all envs, rebuild world, return obs as numpy uint8 [B,H,W,3]."""
        c = self.cfg
        self.generate_world()
        self.do_reset.fill(True)
        sx, sy, sz = self.spawn
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
        self.step_count.fill(0)
        self.buttons.fill(0)
        self.look.fill(0)
        cp.cuda.runtime.deviceSynchronize()
        obs, _, _ = self.step()
        return obs

    def step(
        self,
        buttons: cp.ndarray | np.ndarray | None = None,
        look: cp.ndarray | np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One env step. Returns (obs_uint8, rewards, dones) on host."""
        c = self.cfg
        if buttons is not None:
            self.buttons[:] = cp.asarray(buttons, dtype=cp.int8)
        if look is not None:
            self.look[:] = cp.asarray(look, dtype=cp.float32)

        sx, sy, sz = self.spawn
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
            c.world,
            c.world,
            c.world,
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
            # worlds are identical across envs; cheap full regen
            self.generate_world()
            self.do_reset.fill(False)

        return cp.asnumpy(self.obs), cp.asnumpy(self.rewards), dones
