"""MineCUEnv — TinyGrad-shaped RL loop + optional CUDA inline accelerators."""
from __future__ import annotations

import math

import numpy as np

from . import accel
from .config import (
    BUTTON_DIM,
    BUTTON_NAMES,
    LOOK_DIM,
    BlockType,
    EnvConfig,
    TrainConfig,
    _normalize_kwargs,
)
from .spaces import Box, Dict, try_gymnasium


def _tg_ctx():
    from .backend_tinygrad import _tg_context

    return _tg_context()


def _is_cupy(x) -> bool:
    return type(x).__module__.startswith("cupy")


class MineCUEnv:
    def __init__(self, cfg: EnvConfig | None = None, **kwargs):
        kwargs = _normalize_kwargs(kwargs)
        if cfg is None:
            cfg = EnvConfig.from_dict(kwargs)
        elif kwargs:
            cfg = EnvConfig.from_dict({**cfg.__dict__, **kwargs})

        accel_mode = kwargs.get("accel", cfg.accel)
        if "backend" in kwargs or cfg.backend not in ("auto", "", None):
            b = kwargs.get("backend", cfg.backend)
            if b in ("tinygrad", "tg"):
                accel_mode = "off"
            elif b in ("cuda", "hybrid", "ref"):
                accel_mode = "on"
        self.cfg = cfg
        self.accel_mode = accel_mode
        self._use_cuda = self._resolve_accel(accel_mode)

        B = cfg.batch
        self.positions = np.zeros((B, 3), np.float32)
        self.velocities = np.zeros((B, 3), np.float32)
        self.yaws = np.zeros((B,), np.float32)
        self.pitches = np.zeros((B,), np.float32)
        self.on_ground = np.zeros((B,), np.bool_)
        self.step_count = np.zeros((B,), np.int32)
        self.buttons = np.zeros((B, 8), np.int8)
        self.look = np.zeros((B, 2), np.float32)
        self.rewards = np.zeros((B,), np.float32)
        self._obs_host = np.zeros(
            (B, cfg.height, cfg.width, 3),
            np.float32 if cfg.obs_is_float else np.uint8,
        )

        if self._use_cuda:
            self._init_cuda_buffers()
        else:
            self._voxels_np = None
            self._voxels_tg = None
            self._build_world_tg()

    def _resolve_accel(self, mode: str) -> bool:
        mode = (mode or "auto").lower()
        if mode in ("off", "0", "false", "no", "tinygrad", "tg"):
            return False
        if mode in ("on", "1", "true", "yes", "cuda", "force"):
            if not accel.available():
                raise RuntimeError(
                    f"accel=on but CUDA unavailable: {accel.why_unavailable()}"
                )
            return True
        return accel.available()

    # ---- spaces ----
    @property
    def observation_shape(self):
        return self.cfg.observation_shape

    @property
    def observation_space(self):
        _, gs = try_gymnasium()
        shape = self.cfg.observation_shape
        if self.cfg.obs_is_float:
            if gs:
                return gs.Box(0.0, 1.0, shape=shape, dtype=np.float32)
            return Box(0.0, 1.0, shape=shape, dtype=np.dtype(np.float32))
        if gs:
            return gs.Box(0, 255, shape=shape, dtype=np.uint8)
        return Box(0, 255, shape=shape, dtype=np.dtype(np.uint8))

    @property
    def action_space(self):
        _, gs = try_gymnasium()
        buttons = (
            gs.Box(0, 1, shape=(BUTTON_DIM,), dtype=np.int8)
            if gs
            else Box(0, 1, shape=(BUTTON_DIM,), dtype=np.dtype(np.int8))
        )
        look = (
            gs.Box(-np.pi, np.pi, shape=(LOOK_DIM,), dtype=np.float32)
            if gs
            else Box(-math.pi, math.pi, shape=(LOOK_DIM,), dtype=np.dtype(np.float32))
        )
        if gs:
            return gs.Dict({"buttons": buttons, "look": look})
        return Dict({"buttons": buttons, "look": look})

    @property
    def button_dim(self):
        return BUTTON_DIM

    @property
    def look_dim(self):
        return LOOK_DIM

    @property
    def button_names(self):
        return BUTTON_NAMES

    @property
    def batch_size(self):
        return self.cfg.batch

    @property
    def world_size(self):
        return self.cfg.world

    @property
    def render_width(self):
        return self.cfg.width

    @property
    def render_height(self):
        return self.cfg.height

    @property
    def action_dim(self):
        """Discrete-logit width used by simple CNN examples (buttons only)."""
        return BUTTON_DIM

    @property
    def using_cuda_accel(self) -> bool:
        return self._use_cuda

    def _maybe_sync(self) -> None:
        if self.cfg.sync_every_step:
            accel.sync()

    def _to_host(self, *arrs):
        cp = self._cp
        accel.sync()
        return tuple(cp.asnumpy(a) for a in arrs)

    # ---- CUDA buffers ----
    def _init_cuda_buffers(self) -> None:
        import cupy as cp

        c = self.cfg
        B = c.batch
        self._cp = cp
        self.voxels = cp.zeros((B, c.world_y, c.world_x, c.world_z), dtype=cp.int8)
        self._positions_d = cp.zeros((B, 3), dtype=cp.float32)
        self._velocities_d = cp.zeros((B, 3), dtype=cp.float32)
        self._yaws_d = cp.zeros((B,), dtype=cp.float32)
        self._pitches_d = cp.zeros((B,), dtype=cp.float32)
        self._on_ground_d = cp.zeros((B,), dtype=cp.bool_)
        self._do_reset_d = cp.zeros((B,), dtype=cp.bool_)
        self._step_count_d = cp.zeros((B,), dtype=cp.int32)
        self._buttons_d = cp.zeros((B, 8), dtype=cp.int8)
        self._look_d = cp.zeros((B, 2), dtype=cp.float32)
        self._forward_in = cp.zeros((B,), dtype=cp.float32)
        self._strafe_in = cp.zeros((B,), dtype=cp.float32)
        self._delta_yaw_in = cp.zeros((B,), dtype=cp.float32)
        self._delta_pitch_in = cp.zeros((B,), dtype=cp.float32)
        self._jump_in = cp.zeros((B,), dtype=cp.bool_)
        self._do_break = cp.zeros((B,), dtype=cp.bool_)
        self._do_place = cp.zeros((B,), dtype=cp.bool_)
        self._speed_mult = cp.ones((B,), dtype=cp.float32)
        self._cameras = cp.zeros((B, 5), dtype=cp.float32)
        self._basis = cp.zeros((B, 14), dtype=cp.float32)
        obs_dtype = cp.float32 if c.obs_is_float else cp.uint8
        self._obs_d = cp.zeros((B, c.height, c.width, 3), dtype=obs_dtype)
        self._rewards_d = cp.zeros((B,), dtype=cp.float32)

    def _cuda_worldgen(self) -> None:
        c = self.cfg
        g = int(c.ground_height)
        accel.generate_world(
            self.voxels, c.batch, c.world_x, c.world_y, c.world_z, g, c.seed
        )
        tx, tz = c.tree_xz()
        hx, hz = c.house_xz()
        accel.place_tree(
            self.voxels, c.batch, c.world_x, c.world_y, c.world_z, tx, tz, g
        )
        accel.place_house(
            self.voxels, c.batch, c.world_x, c.world_y, c.world_z, hx, hz, g
        )

    def _build_world_tg(self) -> None:
        from minecu_tg.world import expand_world, generate_world

        w = min(self.cfg.world_x, self.cfg.world_y, self.cfg.world_z)
        with _tg_ctx():
            vox = generate_world(w, self.cfg.ground_height)
            vox_b = expand_world(vox, self.cfg.batch)
            self._voxels_tg = vox_b
            self._voxels_np = vox_b.numpy().astype(np.int8)

    def _ingest_actions(self, buttons, look) -> None:
        """Accept host numpy or device cupy without redundant copies."""
        cp = self._cp if self._use_cuda else None
        if buttons is not None:
            if self._use_cuda and _is_cupy(buttons):
                self._buttons_d[:] = buttons
            else:
                self.buttons[:] = np.asarray(buttons, dtype=np.int8)
                if self._use_cuda:
                    self._buttons_d[:] = cp.asarray(self.buttons)
        if look is not None:
            if self._use_cuda and _is_cupy(look):
                self._look_d[:] = look
            else:
                self.look[:] = np.asarray(look, dtype=np.float32)
                if self._use_cuda:
                    self._look_d[:] = cp.asarray(self.look)

    # ---- API ----
    def reset(self, mask: np.ndarray | None = None):
        c = self.cfg
        if mask is None:
            mask_np = np.ones((c.batch,), dtype=np.bool_)
        else:
            mask_np = np.asarray(mask, dtype=np.bool_)

        if self._use_cuda:
            cp = self._cp
            if mask is None or mask_np.all():
                self._cuda_worldgen()
                self._do_reset_d.fill(True)
                self._step_count_d.fill(0)
            else:
                self._do_reset_d[:] = cp.asarray(mask_np)
                if mask_np.any():
                    self._cuda_worldgen()
                self._step_count_d = cp.where(self._do_reset_d, 0, self._step_count_d)
            accel.reset_agents(
                self._positions_d,
                self._velocities_d,
                self._yaws_d,
                self._pitches_d,
                self._on_ground_d,
                self._do_reset_d,
                c.batch,
                c.spawn(),
            )
            self._do_reset_d.fill(False)
            self._buttons_d.fill(0)
            self._look_d.fill(0)
            self._maybe_sync()
            if c.device_buffers:
                return self.get_obs()
            self.positions[:] = cp.asnumpy(self._positions_d)
            self.yaws[:] = cp.asnumpy(self._yaws_d)
            self.pitches[:] = cp.asnumpy(self._pitches_d)
            return self.get_obs()

        if mask_np.any():
            self._build_world_tg()
        sx, sy, sz = c.spawn()
        self.positions[mask_np] = (sx, sy, sz)
        self.velocities[mask_np] = 0
        self.yaws[mask_np] = 0
        self.pitches[mask_np] = 0
        self.on_ground[mask_np] = True
        self.step_count[mask_np] = 0
        self.buttons[mask_np] = 0
        self.look[mask_np] = 0
        return self.get_obs()

    def get_obs(self):
        if self._use_cuda:
            cp = self._cp
            c = self.cfg
            self._cameras[:, 0] = self._positions_d[:, 0]
            self._cameras[:, 1] = self._positions_d[:, 1]
            self._cameras[:, 2] = self._positions_d[:, 2]
            self._cameras[:, 3] = self._yaws_d
            self._cameras[:, 4] = self._pitches_d
            if c.obs_dtype in ("uint8_prebasis", "uint8_prebasis_full", "uint8_smem"):
                accel.render_prebasis(
                    self.voxels,
                    self._cameras,
                    self._basis,
                    self._obs_d,
                    c.batch,
                    c.world_x,
                    c.width,
                    c.height,
                    c.max_steps,
                    c.fov_degrees,
                )
            else:
                accel.render_cameras(
                    self.voxels,
                    self._cameras,
                    self._obs_d,
                    c.batch,
                    c.world_x,
                    c.width,
                    c.height,
                    c.max_steps,
                    c.view_distance,
                    c.fov_degrees,
                    float_obs=c.obs_is_float,
                )
            if c.device_buffers:
                self._maybe_sync()
                return self._obs_d
            obs = self._to_host(self._obs_d)[0]
            self._obs_host = obs
            return obs.copy()

        from tinygrad import Tensor, dtypes
        from minecu_tg.render import camera_basis
        from minecu_tg_opt.render import render

        c = self.cfg
        cameras = np.zeros((c.batch, 5), np.float32)
        cameras[:, 0:3] = self.positions
        cameras[:, 3] = self.yaws
        cameras[:, 4] = self.pitches
        with _tg_ctx():
            cam_t = Tensor(cameras, dtype=dtypes.float32)
            basis = camera_basis(cam_t, c.fov_degrees, c.width, c.height)
            obs = render(self._voxels_tg, basis, c.width, c.height, c.max_steps)
            out = obs.numpy()
            self._obs_host = out.astype(np.float32 if c.obs_is_float else np.uint8)
        return self._obs_host.copy()

    def step(self, buttons=None, look=None):
        if self._use_cuda:
            self._ingest_actions(buttons, look)
            return self._step_cuda()
        if buttons is not None:
            self.buttons[:] = np.asarray(buttons, dtype=np.int8)
        if look is not None:
            self.look[:] = np.asarray(look, dtype=np.float32)
        return self._step_tg()

    def _step_cuda(self):
        c = self.cfg
        cp = self._cp
        full = c.obs_dtype == "uint8_prebasis_full"

        # Non-full MineEnv paths: episode check lives in Python (full path does it in CUDA).
        if not full:
            if c.episode_length and c.episode_length > 0:
                self._do_reset_d[:] = self._step_count_d >= int(c.episode_length)
            else:
                self._do_reset_d.fill(False)

        accel.fused_step(
            c.obs_dtype,
            self.voxels,
            self._positions_d,
            self._velocities_d,
            self._yaws_d,
            self._pitches_d,
            self._on_ground_d,
            self._buttons_d,
            self._look_d,
            self._do_reset_d,
            self._step_count_d,
            self._forward_in,
            self._strafe_in,
            self._delta_yaw_in,
            self._delta_pitch_in,
            self._jump_in,
            self._do_break,
            self._do_place,
            self._speed_mult,
            self._cameras,
            self._basis,
            self._obs_d,
            self._rewards_d,
            batch=c.batch,
            wx=c.world_x,
            wy=c.world_y,
            wz=c.world_z,
            width=c.width,
            height=c.height,
            max_steps=c.max_steps,
            view_distance=c.view_distance,
            fov=c.fov_degrees,
            dt=c.dt,
            gravity=c.gravity,
            walk_speed=c.walk_speed,
            sprint_mult=c.sprint_mult,
            jump_vel=c.jump_vel,
            target_block=c.target_block,
            reward_value=c.reward_value,
            spawn=c.spawn(),
            episode_length=c.episode_length,
        )

        if not full:
            # MineEnv: step_count += 1; masked_fill_(do_reset, 1)
            self._step_count_d += 1
            self._step_count_d = cp.where(self._do_reset_d, 1, self._step_count_d)

        # World regen when any env reset — MineEnv leaves do_reset set (dones = clone).
        if c.device_buffers and not c.sync_every_step:
            if c.episode_length and c.episode_length > 0:
                accel.sync()
                if bool(self._do_reset_d.any()):
                    self._cuda_worldgen()
            dones = self._do_reset_d.copy()
            return self._obs_d, self._rewards_d, dones

        self._maybe_sync()
        dones = cp.asnumpy(self._do_reset_d)
        if dones.any():
            self._cuda_worldgen()
            self._maybe_sync()
        obs, rew = self._to_host(self._obs_d, self._rewards_d)
        self.positions[:] = cp.asnumpy(self._positions_d)
        self.yaws[:] = cp.asnumpy(self._yaws_d)
        self.pitches[:] = cp.asnumpy(self._pitches_d)
        self.rewards[:] = rew
        self._obs_host = obs
        return obs.copy(), rew.copy(), dones

    def _step_tg(self):
        c = self.cfg
        dones = np.zeros((c.batch,), np.bool_)
        if c.episode_length > 0:
            dones = self.step_count >= c.episode_length
            if dones.any():
                self.reset(dones)

        b = self.buttons
        fwd = b[:, 0].astype(np.float32) - b[:, 1].astype(np.float32)
        strafe = b[:, 3].astype(np.float32) - b[:, 2].astype(np.float32)
        jump = b[:, 4] != 0
        do_break = b[:, 5] != 0
        do_place = b[:, 6] != 0
        sprint = (b[:, 0] != 0) & (b[:, 7] != 0)
        speed = np.where(sprint, c.sprint_mult, 1.0).astype(np.float32)
        dyaw, dpitch = self.look[:, 0], self.look[:, 1]

        self.yaws += dyaw
        self.pitches = np.clip(self.pitches + dpitch, -1.56, 1.56)
        cos_y, sin_y = np.cos(self.yaws), np.sin(self.yaws)
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
        self.positions[:, 0] = np.clip(new_x, margin, c.world_x - margin)
        self.positions[:, 1] = new_y
        self.positions[:, 2] = np.clip(new_z, margin, c.world_z - margin)
        self.velocities[:, 1] = vel_y
        self.on_ground = on
        self.rewards[:] = 0
        if (do_break.any() or do_place.any()) and self._voxels_np is not None:
            self._interact_np(do_break, do_place)
        self.step_count += 1
        return self.get_obs(), self.rewards.copy(), dones

    def _interact_np(self, do_break, do_place) -> None:
        assert self._voxels_np is not None
        c = self.cfg
        vox = self._voxels_np
        for i in range(c.batch):
            if not (do_break[i] or do_place[i]):
                continue
            yaw, pitch = float(self.yaws[i]), float(self.pitches[i])
            eye = self.positions[i] + np.array([0, 1.6, 0], np.float32)
            d = np.array(
                [
                    math.sin(yaw) * math.cos(pitch),
                    math.sin(pitch),
                    math.cos(yaw) * math.cos(pitch),
                ],
                np.float32,
            )
            hit = None
            prev = None
            for t in np.linspace(0.5, 6.0, 24):
                p = eye + t * d
                ix, iy, iz = int(p[0]), int(p[1]), int(p[2])
                if not (
                    0 <= ix < c.world_x and 0 <= iy < c.world_y and 0 <= iz < c.world_z
                ):
                    break
                blk = int(vox[i, iy, ix, iz])
                if blk >= 0:
                    hit = (iy, ix, iz, blk)
                    break
                prev = (iy, ix, iz)
            if hit is None:
                continue
            iy, ix, iz, blk = hit
            if do_break[i] and blk == c.target_block:
                vox[i, iy, ix, iz] = int(BlockType.AIR)
                self.rewards[i] = c.reward_value
            elif do_place[i] and prev is not None:
                py, px, pz = prev
                if int(vox[i, py, px, pz]) < 0:
                    vox[i, py, px, pz] = int(BlockType.DIRT)
        from tinygrad import Tensor, dtypes

        with _tg_ctx():
            self._voxels_tg = Tensor(vox, dtype=dtypes.int8).contiguous()

    def close(self) -> None:
        return

    def zero_actions(self):
        B = self.cfg.batch
        if self._use_cuda and self.cfg.device_buffers:
            return self._buttons_d, self._look_d
        return (
            np.zeros((B, BUTTON_DIM), np.int8),
            np.zeros((B, LOOK_DIM), np.float32),
        )

    @classmethod
    def from_config(cls, cfg: EnvConfig) -> "MineCUEnv":
        return cls(cfg=cfg)


__all__ = [
    "MineCUEnv",
    "EnvConfig",
    "TrainConfig",
    "BlockType",
    "BUTTON_DIM",
    "LOOK_DIM",
    "BUTTON_NAMES",
]
