"""TinyGrad backend — same Env API; render rides TG improvements over time.

Physics / decode / episode are NumPy ports of the reference kernels so the
upper layer can train without CUDA. Hot observation path is
``minecu_tg_opt.render`` (algorithmic ladder), which gets faster when TinyGrad
codegen improves — without changing this env API.
"""
from __future__ import annotations

import math
import os

import numpy as np

from .config import BlockType, EnvConfig


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _tg_context():
    """Pin a TinyGrad device. Works on 0.14 (Device.DEFAULT) and newer (Context DEV)."""
    from tinygrad import Device

    dev = os.environ.get("MINECU_TG_DEV")
    if not dev:
        try:
            avail = list(Device.get_available_devices())
        except Exception:
            avail = [Device.DEFAULT]
        for cand in ("CUDA", "NV", "METAL", "PYTHON", "CPU"):
            if cand in avail or cand == Device.DEFAULT:
                dev = cand
                break
        else:
            dev = Device.DEFAULT
    try:
        from tinygrad import Context

        return Context(DEV=dev)
    except TypeError:
        # 0.14 era: mutate DEFAULT
        try:
            Device.DEFAULT = dev
        except Exception:
            pass
        return _NullCtx()


class TinygradBackend:
    def __init__(self, cfg: EnvConfig):
        if cfg.world_x != cfg.world_y or cfg.world_x != cfg.world_z:
            raise ValueError("tinygrad backend currently requires a cubic world")
        self.cfg = cfg
        B = cfg.batch
        self.positions = np.zeros((B, 3), dtype=np.float32)
        self.velocities = np.zeros((B, 3), dtype=np.float32)
        self.yaws = np.zeros((B,), dtype=np.float32)
        self.pitches = np.zeros((B,), dtype=np.float32)
        self.on_ground = np.zeros((B,), dtype=np.bool_)
        self.step_count = np.zeros((B,), dtype=np.int32)
        self.buttons = np.zeros((B, 8), dtype=np.int8)
        self.look = np.zeros((B, 2), dtype=np.float32)
        self.rewards = np.zeros((B,), dtype=np.float32)
        self._voxels_np: np.ndarray | None = None
        self._obs = np.zeros((B, cfg.height, cfg.width, 3), dtype=np.uint8)
        self._build_world()

    def _build_world(self) -> None:
        from minecu_tg.world import expand_world, generate_world

        with _tg_context():
            vox = generate_world(self.cfg.world_x, self.cfg.ground_height)
            vox_b = expand_world(vox, self.cfg.batch)
            self._voxels_tg = vox_b
            self._voxels_np = vox_b.numpy().astype(np.int8)

    def reset(self, mask: np.ndarray | None = None) -> np.ndarray:
        c = self.cfg
        if mask is None:
            mask = np.ones((c.batch,), dtype=np.bool_)
        else:
            mask = np.asarray(mask, dtype=np.bool_)
        if mask.any():
            self._build_world()
        sx, sy, sz = c.spawn()
        self.positions[mask] = (sx, sy, sz)
        self.velocities[mask] = 0
        self.yaws[mask] = 0
        self.pitches[mask] = 0
        self.on_ground[mask] = True
        self.step_count[mask] = 0
        self.buttons[mask] = 0
        self.look[mask] = 0
        return self.get_obs()

    def _decode(self):
        b = self.buttons
        fwd = b[:, 0].astype(np.float32) - b[:, 1].astype(np.float32)
        strafe = b[:, 3].astype(np.float32) - b[:, 2].astype(np.float32)
        jump = b[:, 4] != 0
        do_break = b[:, 5] != 0
        do_place = b[:, 6] != 0
        sprint = (b[:, 0] != 0) & (b[:, 7] != 0)
        speed = np.where(sprint, self.cfg.sprint_mult, 1.0).astype(np.float32)
        return fwd, strafe, self.look[:, 0], self.look[:, 1], jump, do_break, do_place, speed

    def _physics(self, fwd, strafe, dyaw, dpitch, jump, speed) -> None:
        c = self.cfg
        self.yaws += dyaw
        self.pitches = np.clip(self.pitches + dpitch, -1.56, 1.56)
        cos_y = np.cos(self.yaws)
        sin_y = np.sin(self.yaws)
        move_x = sin_y * fwd * c.walk_speed * speed * c.dt + cos_y * strafe * c.walk_speed * c.dt
        move_z = cos_y * fwd * c.walk_speed * speed * c.dt - sin_y * strafe * c.walk_speed * c.dt
        vel_y = self.velocities[:, 1].copy()
        vel_y = np.where(jump & self.on_ground, c.jump_vel, vel_y)
        vel_y = vel_y + c.gravity * c.dt
        new_x = self.positions[:, 0] + move_x
        new_y = self.positions[:, 1] + vel_y * c.dt
        new_z = self.positions[:, 2] + move_z
        ground_y = float(c.world_y // 4)
        on = new_y <= ground_y
        new_y = np.where(on, ground_y, new_y)
        vel_y = np.where(on, 0.0, vel_y)
        margin = 1.0
        new_x = np.clip(new_x, margin, c.world_x - margin)
        new_z = np.clip(new_z, margin, c.world_z - margin)
        self.positions[:, 0] = new_x
        self.positions[:, 1] = new_y
        self.positions[:, 2] = new_z
        self.velocities[:, 1] = vel_y
        self.on_ground = on

    def _interact(self, do_break: np.ndarray, do_place: np.ndarray) -> None:
        """Coarse break/place along look ray — enough for RL signal, not bit-exact."""
        assert self._voxels_np is not None
        c = self.cfg
        self.rewards[:] = 0
        vox = self._voxels_np
        for i in range(c.batch):
            if not (do_break[i] or do_place[i]):
                continue
            yaw, pitch = float(self.yaws[i]), float(self.pitches[i])
            eye = self.positions[i] + np.array([0, 1.6, 0], dtype=np.float32)
            fx = math.sin(yaw) * math.cos(pitch)
            fy = math.sin(pitch)
            fz = math.cos(yaw) * math.cos(pitch)
            hit = None
            for t in np.linspace(0.5, 6.0, 24):
                p = eye + t * np.array([fx, fy, fz], dtype=np.float32)
                ix, iy, iz = int(p[0]), int(p[1]), int(p[2])
                if not (0 <= ix < c.world_x and 0 <= iy < c.world_y and 0 <= iz < c.world_z):
                    break
                blk = int(vox[i, iy, ix, iz])
                if blk >= 0:
                    hit = (iy, ix, iz, blk)
                    break
            if hit is None:
                continue
            iy, ix, iz, blk = hit
            if do_break[i] and blk == c.target_block:
                vox[i, iy, ix, iz] = BlockType.AIR
                self.rewards[i] = c.reward_value
            elif do_place[i] and blk >= 0:
                # place in front of hit: pull back one cell
                pass
        # sync TG buffer from numpy (rare path)
        from tinygrad import Tensor, dtypes

        with _tg_context():
            self._voxels_tg = Tensor(vox, dtype=dtypes.int8).contiguous()

    def get_obs(self) -> np.ndarray:
        from tinygrad import Tensor, dtypes
        from minecu_tg.render import camera_basis
        from minecu_tg_opt.render import render

        c = self.cfg
        cameras = np.zeros((c.batch, 5), dtype=np.float32)
        cameras[:, 0:3] = self.positions
        cameras[:, 3] = self.yaws
        cameras[:, 4] = self.pitches
        with _tg_context():
            cam_t = Tensor(cameras, dtype=dtypes.float32)
            basis = camera_basis(cam_t, c.fov_degrees, c.width, c.height)
            obs = render(self._voxels_tg, basis, c.width, c.height, c.max_steps)
            self._obs = obs.numpy().astype(np.uint8)
        return self._obs.copy()

    def step(
        self,
        buttons: np.ndarray | None = None,
        look: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = self.cfg
        if buttons is not None:
            self.buttons[:] = np.asarray(buttons, dtype=np.int8)
        if look is not None:
            self.look[:] = np.asarray(look, dtype=np.float32)

        dones = np.zeros((c.batch,), dtype=np.bool_)
        if c.episode_length > 0:
            dones = self.step_count >= c.episode_length
            if dones.any():
                self.reset(dones)

        fwd, strafe, dyaw, dpitch, jump, do_break, do_place, speed = self._decode()
        self._physics(fwd, strafe, dyaw, dpitch, jump, speed)
        if do_break.any() or do_place.any():
            self._interact(do_break, do_place)
        else:
            self.rewards[:] = 0
        self.step_count += 1
        obs = self.get_obs()
        return obs, self.rewards.copy(), dones

    def close(self) -> None:
        return
