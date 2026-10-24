"""backend.py -- the interface every transport implements.

THE INTERFACE, IN FULL
======================
A backend is six methods in three phases. There is no seventh operation, no
inherited helper that a backend must override, and nothing in the core that
branches on which backend it is talking to.

    SYNC   plan(tree)               -> Plan        offline; zero network, zero credentials
           push(tree, plan)         -> State       make the remote match the tree
           status()                  -> Status      what the remote is doing right now

    RUN    doctor()                 -> Report      is this transport reachable, with what, at what cost
           run(tree, argv)          -> Run         execute one command on the remote device

    OUT    fetch(state, dest)       -> [Path]      artefacts home, beside their provenance record

THE TWO DISTINCTIONS THAT MATTER

* `push` is not `run`. Push spends bytes and an API call. Run spends
  *accelerator quota*, which on Kaggle is a weekly budget and on a Colab free
  tier is a machine that may be reclaimed mid-session. Every subcommand in
  gpush is one of these two, and the CLI prints which one it is before doing
  it. A user should never have to read the source to know whether a command
  costs them GPU hours.

* `push` returns, `run` blocks. There is no "push and wait and download" verb
  in the backend; `run` does not push, and `gpush kaggle run` composes them.
  That composition is the only place the two meet, and it lives in the core, so
  a new backend inherits the ordering, the gates and the provenance for free.

WHAT A BACKEND MUST NOT DO
    * invent its own fingerprint, or accept an externally supplied one.
      `tree.fingerprint` is already salted with the backend name and the target
      id; a backend that hashes again, differently, has broken the property
      that makes the no-op check meaningful.
    * promote a baseline to `confirmed`. Only `fetch` returning real artefacts
      lets the core do that, and the core is the same code for every backend.
    * assume a device. Colab does not publish which GPU a free session got, so
      the backend reports what the machine says and refuses to label a
      CPU-only box as a GPU run. It surfaces compute capability and does not
      interpret it.
    * fall back to a local execution "for convenience". That is how a CPU
      number ends up in a table of GPU numbers.

ADDING A THIRD BACKEND
    Write one module that subclasses `Backend`, decorate it with `@register`,
    and put it in `gpush/backends/`. The core does not change. A plain-SSH
    backend needs nothing beyond `tar` and `ssh`; a Modal or crabbox backend
    needs whatever its own transport needs.
"""
from __future__ import annotations

import abc
import re
from dataclasses import dataclass, field

from .manifest import Manifest, Plan

# --------------------------------------------------------------------------
# Value types. Plain dataclasses, no behaviour, so a backend never has to import
# anything from the core to build one.
# --------------------------------------------------------------------------


@dataclass
class SyncState:
    """What `push` reports back: where the tree now lives, remotely."""
    ok: bool
    fingerprint: str = ""
    bytes_sent: int = 0
    files: int = 0
    remote: dict = field(default_factory=dict)     # backend-specific addressing
    transferred: bool = False                     # False => a real no-op
    skipped_reason: str = ""
    note: str = ""
    gates: list[tuple[str, bool, str]] = field(default_factory=list)


@dataclass
class RemoteStatus:
    """What `status` reports back. Reading only; never a side effect."""
    phase: str              # a backend-specific word: QUEUED / RUNNING / IDLE
    detail: str = ""
    device: dict = field(default_factory=dict)
    raw: str = ""


@dataclass
class Report:
    """What `doctor` reports back."""
    ok: bool
    checks: list[tuple[str, bool, str]] = field(default_factory=list)
    credentials: str = ""
    costs_gpu: bool = False
    note: str = ""


@dataclass
class Run:
    """What `run` reports back. This is the operation that spends quota."""
    ok: bool
    run_id: str = ""
    status: str = ""
    device: dict = field(default_factory=dict)
    log: str = ""
    exit_code: int | None = None
    started: str = ""
    finished: str = ""
    note: str = ""


# --------------------------------------------------------------------------
# The interface
# --------------------------------------------------------------------------


class Backend(abc.ABC):
    """Six methods. See the module docstring for the contract."""

    name: str = "abstract"
    #: True if `run` spends accelerator quota on the remote side. Printed by
    #: the CLI so the cost of a command is visible before it is run.
    spends_gpu: bool = False
    #: One-line statement of what this transport physically is, and therefore
    #: what it can and cannot do.
    transport: str = ""

    def __init__(self, **opts):
        self.opts = opts

    # -- SYNC
    @abc.abstractmethod
    def plan(self, tree: Manifest) -> Plan:
        """Offline diff of `tree` against the stored baseline.

        Must make no network call and read no credential. `gpush <b> plan` is
        the review surface, and a review surface that touches the network is
        not one.
        """

    @abc.abstractmethod
    def push(self, tree: Manifest, plan: Plan, *, full_resync: bool = False) -> SyncState:
        """Make the remote working tree equal `tree`.

        Honoured by the core before this is called: the mass-deletion guard and
        the no-op check. A backend must still refuse to claim success it did
        not observe -- in particular, an upload that was accepted is not a
        remote that has the bytes yet.
        """

    @abc.abstractmethod
    def status(self) -> RemoteStatus:
        """Read the remote's current state. No side effects, no quota."""

    # -- RUN
    @abc.abstractmethod
    def doctor(self) -> Report:
        """Is this transport usable right now, with what, and at what cost."""

    @abc.abstractmethod
    def run(self, tree: Manifest, argv: list[str], **kw) -> Run:
        """Execute one command on the remote device. THIS SPENDS QUOTA.

        argv is the exact command; it is recorded verbatim in the provenance
        record, so it must be the real argv and not a paraphrase of it.
        """

    # -- OUT
    @abc.abstractmethod
    def fetch(self, state: dict, dest: str) -> list[str]:
        """Bring one run's artefacts to `dest`. Returns the paths written."""

    def describe_target(self) -> dict:
        """How this backend addresses the remote, from configuration alone.

        Used as the last fallback when recording a run's provenance, so a run
        whose push was a no-op skip still names what it executed against. The
        default is empty, which is correct for a backend with no persistent
        remote address; backends with one should override.
        """
        return {}


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

_REGISTRY: dict[str, type[Backend]] = {}


def register(cls: type[Backend]) -> type[Backend]:
    if not cls.name or cls.name == "abstract":
        raise ValueError("a Backend must set a name")
    if cls.name in _REGISTRY:
        raise ValueError(f"backend {cls.name!r} is already registered")
    _REGISTRY[cls.name] = cls
    return cls


def load_all() -> dict[str, type[Backend]]:
    """Import every backend module, so `gpush --list-backends` is complete."""
    from . import backends  # noqa: F401  (import for side effect: registration)
    return dict(_REGISTRY)


def get(name: str, **opts) -> Backend:
    backends = load_all()
    if name not in backends:
        near = [n for n in backends if name and name in n]
        raise KeyError(f"no backend {name!r}"
                       + (f"; did you mean {' or '.join(near)}?" if near else "")
                       + f"; known: {', '.join(sorted(backends))}")
    return backends[name](**opts)


def known() -> list[str]:
    return sorted(load_all())


# --------------------------------------------------------------------------
# Shared helper: the slug/title rule Kaggle needs
# --------------------------------------------------------------------------


def slugify(text: str) -> str:
    """Kaggle's title -> slug rule, as observed in minecu_ref/kaggle/make_notebook.py.

    Lowercase, every run of non-alphanumerics to a single hyphen, trimmed. The
    push API answers 409 "Your kernel title does not resolve to the specified
    id" when this does not round-trip, and that rejection happens *after* the
    upload has been accepted -- so the check is done here instead.
    """
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def title_from_slug(slug: str) -> str:
    """The only correct way to write a Kaggle title.

    Derived from the slug, never written by hand, because `make_notebook.py`
    and friends regenerate metadata on every run and a hand-edited title is a
    409 waiting to happen. `mc-minecu-t4-baseline` -> `mc minecu t4 baseline`,
    which slugs straight back.
    """
    return " ".join(w for w in slug.split("-") if w)
