"""gpush -- sync a local working tree to a remote notebook-GPU box and run work there.

A general-purpose utility. This repository is its first user, not its purpose.

The backends (kaggle, colab) are genuinely different transports and are
implemented against one documented interface; see `gpush.backend`. The core
-- manifest, fingerprint, diff, guards, provenance -- knows nothing about any
of them.

Standard library only. No GPU, no CUDA toolchain, no container is required to
run any part of this: everything needing nvcc/ptxas/SASS is planned to happen
on the remote side.
"""

__version__ = "1.0.0"

# Bump when the meaning of a fingerprint changes. It is mixed into every
# fingerprint, so an old baseline can never be mistaken for a current one
# after a semantics change.
MANIFEST_SCHEMA = "gpush/manifest@1"
BASELINE_SCHEMA = "gpush/baseline@1"
PROVENANCE_SCHEMA = "gpush/provenance@1"
