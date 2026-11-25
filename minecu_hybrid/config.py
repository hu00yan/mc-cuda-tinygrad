"""Stable config surface — mirrors Infatoshi MineEnv knobs, study-friendly defaults."""
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


@dataclass
class EnvConfig:
    """Batched MineCU env config.

    Defaults match the project's measured study shape (world=16, 32×32).
    Call ``EnvConfig.upstream()`` for Infatoshi-like 32³ / 64×64 / episode=60.
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
    episode_length: int = 0  # 0 = unlimited
    seed: int = 42
    ground_height: int | None = None
    # accel: auto (CUDA asm if libminecu.so present) | on | off
    # legacy: backend="cuda"|"tinygrad" still accepted by MineCUEnv
    accel: str = "auto"
    backend: str = "auto"  # legacy alias; prefer accel=

    def __post_init__(self) -> None:
        wx = self.world_x if self.world_x is not None else self.world
        wy = self.world_y if self.world_y is not None else self.world
        wz = self.world_z if self.world_z is not None else self.world
        object.__setattr__(self, "world_x", wx)
        object.__setattr__(self, "world_y", wy)
        object.__setattr__(self, "world_z", wz)
        if self.ground_height is None:
            object.__setattr__(self, "ground_height", wy // 4)

    @classmethod
    def upstream(cls, **kwargs) -> "EnvConfig":
        """Closer to Infatoshi MineEnv defaults (still cubic for simplicity)."""
        base = dict(
            batch=4096,
            world=32,
            width=64,
            height=64,
            max_steps=64,
            view_distance=64.0,
            episode_length=60,
        )
        base.update(kwargs)
        return cls(**base)

    @classmethod
    def from_dict(cls, d: dict) -> "EnvConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, 3)

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
        return (self.world_x // 2, self.world_z // 2 + 3)

    def house_xz(self) -> tuple[int, int]:
        return (self.world_x // 2 + 6, self.world_z // 2 - 2)
