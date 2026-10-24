"""experiment_b -- the device-free half of the KDA codegen comparison.

Comparison (b): can a compiler reach the construction a human wrote? The
quantities are bytes materialised and instruction classes emitted, and both are
readable from a schedule and a renderer without a card.

Read `README.md` first. Every figure in `notes/15-experiment-b.md` is printed by
one of the scripts here, and every artifact any of them reads is pinned by
sha256 in `common.PINS`.
"""
