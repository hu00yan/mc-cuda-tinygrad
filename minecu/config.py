"""Stable config surface — mirrors Infatoshi MineEnv knobs."""
from __future__ import annotations

from dataclasses import dataclass, fields
from enum import IntEnum


class BlockType(IntEnum):
    AIR = -1
    GRASS = 0
    DIRT = 1
    STONE = 2
    OAKLOG = 3
    LEAVES = 4
    SAND = 5
    WATER = 6
    GLASS = 7
    BRICK = 8
    COBBLESTONE = 9
    PLANKS = 10
    SNOW = 11
    BEDROCK = 12


BUTTON_NAMES = (
    "forward", "back", "left", "right", "jump", "break", "place", "sprint",
)
BUTTON_DIM = 8
LOOK_DIM = 2

# MineEnv obs_dtype strings (Infatoshi)
OBS_DTYPES = (
    "float32",
    "uint8",
    "uint8_minimal",
    "uint8_fp16",
    "uint8_prebasis",
    "uint8_prebasis_full",  # default hot path
    "uint8_smem",
)

# Infatoshi MineEnv kwargs → EnvConfig field names
_ALIAS = {
    "batch_size": "batch",
    "world_size": "world",
    "render_width": "width",
    "render_height": "height",
}


def _normalize_kwargs(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        out[_ALIAS.get(k, k)] = v
    # MineEnv uses None for unlimited episodes; kernels use 0
    if "episode_length" in out and out["episode_length"] is None:
        out["episode_length"] = 0
    return out


@dataclass
class EnvConfig:
    """Batched MineCU env config.

    Study-friendly defaults (world=16, 32×32). Use ``upstream()`` for
    Infatoshi-like 32³ / 64×64 / episode=60.
    """

    batch: int = 256
    world: int = 16
    world_x: int | None = None
    world_y: int | None = None
    world_z: int | None = None
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
    target_block: int = int(BlockType.OAKLOG)
    reward_value: float = 1.0
    episode_length: int = 0  # 0 = unlimited (MineEnv None)
    seed: int = 42
    ground_height: int | None = None
    obs_dtype: str = "uint8_prebasis_full"
    # If True, step/reset/get_obs return cupy arrays (no H2D/D2H). Caller syncs.
    device_buffers: bool = False
    # If False (default when device_buffers), skip cudaDeviceSynchronize in step.
    sync_every_step: bool | None = None
    accel: str = "auto"
    backend: str = "auto"

    def __post_init__(self) -> None:
        wx = self.world_x if self.world_x is not None else self.world
        wy = self.world_y if self.world_y is not None else self.world
        wz = self.world_z if self.world_z is not None else self.world
        object.__setattr__(self, "world_x", wx)
        object.__setattr__(self, "world_y", wy)
        object.__setattr__(self, "world_z", wz)
        if self.ground_height is None:
            object.__setattr__(self, "ground_height", wy // 4)
        if self.obs_dtype not in OBS_DTYPES:
            raise ValueError(f"obs_dtype {self.obs_dtype!r} not in {OBS_DTYPES}")
        if self.sync_every_step is None:
            object.__setattr__(self, "sync_every_step", not self.device_buffers)
        if self.episode_length is None:
            object.__setattr__(self, "episode_length", 0)

    @classmethod
    def upstream(cls, **kwargs) -> "EnvConfig":
        base = dict(
            batch=4096,
            world=32,
            width=64,
            height=64,
            max_steps=64,
            view_distance=64.0,
            episode_length=60,
            obs_dtype="uint8_prebasis_full",
            device_buffers=True,
            sync_every_step=False,
        )
        base.update(_normalize_kwargs(kwargs))
        return cls(**base)

    @classmethod
    def from_dict(cls, d: dict) -> "EnvConfig":
        names = {f.name for f in fields(cls)}
        norm = _normalize_kwargs(d)
        return cls(**{k: v for k, v in norm.items() if k in names})

    @property
    def batch_size(self) -> int:
        return self.batch

    @property
    def world_size(self) -> int:
        return self.world

    @property
    def render_width(self) -> int:
        return self.width

    @property
    def render_height(self) -> int:
        return self.height

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, 3)

    @property
    def obs_is_float(self) -> bool:
        return self.obs_dtype == "float32"

    @property
    def button_dim(self) -> int:
        return BUTTON_DIM

    @property
    def look_dim(self) -> int:
        return LOOK_DIM

    def spawn(self) -> tuple[float, float, float]:
        g = int(self.ground_height)
        return (self.world_x / 2.0, float(g + 2), self.world_z / 2.0)

    def tree_xz(self) -> tuple[int, int]:
        # MineEnv: tree at (spawn_x, spawn_z + 2)
        sx, _, sz = self.spawn()
        return (int(sx), int(sz) + 2)

    def house_xz(self) -> tuple[int, int]:
        sx, _, sz = self.spawn()
        return (int(sx) + 6, int(sz) - 2)

    def memory_bytes(self) -> int:
        """Rough device footprint for voxels+obs+agent state (bytes)."""
        B = self.batch
        vox = B * self.world_x * self.world_y * self.world_z
        obs_b = B * self.height * self.width * 3 * (4 if self.obs_is_float else 1)
        agent = B * (3 + 3 + 1 + 1 + 1) * 4
        return vox + obs_b + agent

    def memory_mb(self) -> float:
        return self.memory_bytes() / (1024 * 1024)


@dataclass
class TrainConfig:
    """Training knobs — mirrors Infatoshi minecu.config.TrainConfig."""

    lr: float = 3e-4
    gamma: float = 0.99
    total_steps: int = 10_000_000
    log_interval: int = 100_000
    hidden_dim: int = 256


DEFAULT_ENV = EnvConfig()

FAST_ENV = EnvConfig(
    batch=32768,
    world=16,
    width=32,
    height=32,
    max_steps=32,
    obs_dtype="uint8_prebasis_full",
    device_buffers=True,
    sync_every_step=False,
)

QUALITY_ENV = EnvConfig(
    batch=4096,
    world=32,
    width=128,
    height=128,
    max_steps=128,
    view_distance=64.0,
    obs_dtype="uint8_prebasis_full",
    device_buffers=True,
    sync_every_step=False,
)
