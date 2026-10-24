"""backends/ -- one module per transport.

Importing this package registers every backend. That is the only thing the
core knows about transports: the name table. Adding a backend is adding a file
here and changing nothing else.

    kaggle.py   REST only. A tree is a dataset version; a notebook runs it.
    colab.py    ssh to a real Linux VM. tar in, argv, tar out.

A third transport -- plain ssh to a box with a stable hostname, Modal, crabbox
-- is a module that subclasses Backend, calls @register, and implements the six
methods in gpush/backend.py.
"""
from . import kaggle, colab  # noqa: F401  (registration is the import)

__all__ = ["kaggle", "colab"]
