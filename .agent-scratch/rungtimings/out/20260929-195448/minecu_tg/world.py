"""The voxel world, as tensor ops.

Transcribed from three hand-written kernels, unmodified:

    generate_world_kernel   kernels.cu:1793
    place_tree_kernel       kernels.cu:1831
    place_house_kernel      kernels.cu:1868

Written the way a person writes it with array operations: broadcast the three
coordinate axes against each other, build the block id with nested `where`,
then stamp the tree and the house by scattering a handful of constant voxels.

Two properties of the reference are worth stating because they are load-bearing
for the comparison and neither is a codegen choice:

  * **The world is identical for every env.** `generate_world_kernel` accepts a
    `seed` argument and never reads it; the tree and the house are placed at
    fixed offsets. So `generate_world` returns one [Y, X, Z] volume and
    `expand` gives every env the same one. The reference materialises B
    identical copies and the render kernel reads B separate addresses; this
    matters for the memory-traffic half of the comparison and is called out
    there rather than being papered over.

  * **At world=16 the house does not fit.** Its corner is
    `(spawn_x + 6, ground, spawn_z - 2)` = `(14, ground, 6)` and it is 5 wide,
    so `dx` in {2, 3, 4} land at x in {16, 17, 18} and are dropped by the
    bounds test. `dda_model.py:71-103` says the same thing about the same
    world. The half-house is what the reference renders, so the tensor version
    builds a half-house too; the bounds test is written out rather than assumed.
"""

from tinygrad import Tensor, dtypes

from .blocks import AIR, BEDROCK, DIRT, GRASS, OAKLOG, LEAVES, COBBLESTONE, PLANKS, BRICK


def ground_of(world: int, ground: int | None = None) -> int:
    """The ground height, derived from the world size when not given.

    `physics_kernel` floors the agent on a plane at `world_y / 4` *integer*
    division (kernels.cu:1566) and `generate_world_kernel` puts the grass at
    `ground_height` (kernels.cu:1823). The two agree only when
    `ground_height == world_y / 4`, and disagreeing buries the agent inside
    its own surface, which collapses the DDA to ~1.3 steps/pixel. Deriving it
    is the configuration the baseline was measured in
    (`minecu_ref/buried_agent_fix.md`).
    """
    return world // 4 if ground is None else ground


def _axis(n: int, dim: int) -> Tensor:
    return Tensor(list(range(n)), dtype=dtypes.int32).reshape(
        n if dim == 0 else 1, n if dim == 1 else 1, n if dim == 2 else 1)


def generate_world(world: int, ground: int | None = None) -> Tensor:
    """One [Y, X, Z] int8 volume, exactly as the three worldgen kernels leave it.

    generate_world_kernel is a three-way branch on the y coordinate:

        y == 0                 -> BEDROCK
        y <  ground_height - 1 -> DIRT
        y == ground_height - 1 -> GRASS
        otherwise              -> AIR

    which as tensor ops is two nested `where`s over the y axis. Each axis is a
    [N, 1, 1] / [1, N, 1] / [1, 1, N] row so the three broadcast against each
    other into [Y, X, Z].
    """
    g = ground_of(world, ground)
    y, x, z = _axis(world, 0), _axis(world, 1), _axis(world, 2)

    blk = (y == 0).where(BEDROCK, (y < g - 1).where(DIRT, (y == g - 1).where(GRASS, AIR)))
    vox = (blk + x * 0 + z * 0).cast(dtypes.int8)   # broadcast to [Y, X, Z]

    vox = _place_tree(vox, world, g)
    vox = _place_house(vox, world, g)
    return vox.contiguous()


def _at(vox: Tensor, y: int, x: int, z: int, block: int) -> Tensor:
    """One voxel write, in bounds or not.

    The reference's SET_VOXEL macro bounds-checks before writing
    (kernels.cu:1896), so a house that does not fit writes nothing outside the
    world. Scattering a whole [Y, X, Z] mask of `block` into `vox` and
    `where`-ing it in reproduces that: an out-of-range coordinate simply never
    matches an element of the volume.
    """
    wy, wx, wz = vox.shape
    yy, xx, zz = _axis(wy, 0), _axis(wx, 1), _axis(wz, 2)
    m = ((yy == y) & (xx == x) & (zz == z))
    return m.where(block, vox)


def _place_tree(vox: Tensor, world: int, ground: int) -> Tensor:
    """place_tree_kernel (kernels.cu:1831): 3 OAKLOG at y = ground+dy .. +2, one
    LEAVES at ground+3, at (spawn_x, spawn_z + 2). The reference's `if (y < wy)`
    is the bounds test, so at a small world the top of the tree can be dropped.
    """
    tx, tz = world // 2, world // 2 + 2
    for dy in range(3):
        vox = _at(vox, ground + dy, tx, tz, OAKLOG)
    vox = _at(vox, ground + 3, tx, tz, LEAVES)
    return vox


def _place_house(vox: Tensor, world: int, ground: int) -> Tensor:
    """place_house_kernel (kernels.cu:1868): 5 wide x 4 deep x 4 tall at
    (spawn_x + 6, spawn_z - 2), base y = ground. Floor cobblestone, plank walls
    with a door opening, oak corner pillars, brick roof."""
    hx, hz, W, D, H = world // 2 + 6, world // 2 - 2, 5, 4, 4
    base = ground
    for dx in range(W):
        for dz in range(D):
            vox = _at(vox, base + 0, hx + dx, hz + dz, COBBLESTONE)
    for dy in range(1, 4):
        for dx in range(W):
            if dx == 2 and dy <= 2:
                continue                        # the door opening
            vox = _at(vox, base + dy, hx + dx, hz + 0, PLANKS)
            vox = _at(vox, base + dy, hx + dx, hz + D - 1, PLANKS)
        for dz in range(1, D - 1):
            vox = _at(vox, base + dy, hx + 0, hz + dz, PLANKS)
            vox = _at(vox, base + dy, hx + W - 1, hz + dz, PLANKS)
    for dy in range(1, 4):
        for dx, dz in ((0, 0), (W - 1, 0), (0, D - 1), (W - 1, D - 1)):
            vox = _at(vox, base + dy, hx + dx, hz + dz, OAKLOG)
    for dx in range(W):
        for dz in range(D):
            vox = _at(vox, base + H, hx + dx, hz + dz, BRICK)
    return vox


def expand_world(vox: Tensor, batch: int) -> Tensor:
    """[Y, X, Z] -> [B, Y, X, Z], the layout `render` gathers from.

    `repeat` rather than `expand` + `contiguous`, because the render kernel
    indexes `voxels[b*wxyz + ...]` and a materialised copy is what the
    reference has in memory; a broadcast view would hand the gather a stride-0
    axis and quietly turn a memory question into a layout trick. The cost of
    that choice is B*world^3 bytes written once, and it is the same bytes the
    reference writes.
    """
    return vox.reshape(1, *vox.shape).repeat(batch, 1, 1, 1).contiguous()
