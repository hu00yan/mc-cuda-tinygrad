"""Block ids and the colour tables, transcribed from the reference.

Everything here is copied from `Infatoshi/mine.cu src/kernels.cu`, which is the
hand-written side and is never modified. Line numbers in the comments are the
reference's.

    kernels.cu:11-24    the 13 block ids, AIR = -1
    kernels.cu:479-493  BLOCK_COLORS_U8
    kernels.cu:495      SKY_COLOR_U8
"""

from tinygrad import Tensor, dtypes

AIR = -1
GRASS, DIRT, STONE, OAKLOG, LEAVES, SAND, WATER, GLASS, BRICK, COBBLESTONE, \
    PLANKS, SNOW, BEDROCK = range(13)

BLOCK_NAMES = ("GRASS", "DIRT", "STONE", "OAKLOG", "LEAVES", "SAND", "WATER",
               "GLASS", "BRICK", "COBBLESTONE", "PLANKS", "SNOW", "BEDROCK")

# kernels.cu:479
BLOCK_COLORS_U8 = [
    [76, 166, 51],      # GRASS
    [140, 89, 51],      # DIRT
    [127, 127, 127],    # STONE
    [140, 102, 64],     # OAKLOG
    [51, 140, 38],      # LEAVES
    [229, 217, 153],    # SAND
    [51, 102, 204],     # WATER
    [217, 229, 242],    # GLASS
    [178, 89, 76],      # BRICK
    [102, 102, 102],    # COBBLESTONE
    [191, 153, 102],    # PLANKS
    [242, 242, 250],    # SNOW
    [38, 38, 38],       # BEDROCK
]

# kernels.cu:495
SKY_COLOR_U8 = [135, 206, 235]


def colour_lut():
    """The colour table as one [14, 3] uint8 table.

    Row 0 is GRASS .. row 12 is BEDROCK, exactly as `BLOCK_COLORS_U8` is
    indexed in the reference (`BLOCK_COLORS_U8[hit_block]`, with hit_block in
    [0, 12]). Row 13 is the sky, which the reference reaches by a *branch*
    rather than an index: `hit_block` is initialised to AIR == -1 and
    `output = SKY_COLOR_U8` when the DDA finds nothing. Appending the sky as
    row 13 turns that branch into the same table lookup the hit path already
    does, so "did not hit" and "hit" become one gather. That is a change of
    *shape*, not of schedule: no codegen knob, no fusion hint, no tile size.
    It is what a person writing the tensor version would do to avoid an
    `if` on a tensor.
    """
    rows = [list(map(int, c)) for c in BLOCK_COLORS_U8] + [list(map(int, SKY_COLOR_U8))]
    return Tensor(rows, dtype=dtypes.uint8)
