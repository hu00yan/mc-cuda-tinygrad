"""Shared machinery for `derivation/`.

WHAT THIS FILE IS FOR
    Every module in `derivation/` is an executable claim. This file gives all
    of them the same three properties, so that a document never has to think
    about how to ask for a number:

      1. STANDALONE.  `python derivation/<mod>.py` prints a full report.
      2. ADDRESSABLE. `python derivation/cite.py <mod>:<claim>` prints exactly
         one quotable value. That string is the only thing a document may
         contain, and it is the thing the citation names.
      3. PINNED.       A claim reads inputs it does not own -- the installed
         tinygrad's source, an emitted CUDA kernel in `out/`, a shipped log.
         `Pin` records a digest of each of those at the moment the claim was
         written down. `verify.py` recomputes them. If an input moves, the
         module exits non-zero and says which input moved, rather than printing
         a stale number that looks current.

THE RULE THIS ENFORCES
    No number in any output path is a literal in this file or in any module.
    Literals appear only as *inputs* (a seed, a shape, a file path), and a
    result is always computed from them. Where a document needs a constant that
    looks like a result -- `2^-25`, `1 ULP`, `256x` -- the module derives it
    from the object it is a property of (a `nextafter` pair, a measured maximum
    in ULPs, a shape ratio) and prints the derivation alongside the value.

FAIL LOUDLY
    `Pin.drift()` returns the list of pins whose digest changed. A module
    treats a non-empty drift list as fatal unless it is explicitly told to
    continue (`--accept-drift`), which exists so that a *deliberate* upstream
    bump is a one-command, visible act rather than a silent staleness.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# ------------------------------------------------------------------ digests

def sha(data) -> str:
    """Short, stable digest of anything. `data` is coerced to bytes; arrays
    hash by C-contiguous raw bytes so the digest tracks the numbers, not the
    dtype's spelling."""
    if isinstance(data, (bytes, bytearray, memoryview)):
        b = bytes(data)
    elif isinstance(data, str):
        b = data.encode()
    elif hasattr(data, "tobytes") and hasattr(data, "dtype"):
        b = data.tobytes()
    else:
        b = repr(data).encode()
    return hashlib.sha256(b).hexdigest()[:16]


# ------------------------------------------------------------------ pins

class PinDrift(RuntimeError):
    """An input a claim depends on is not the input the claim was written for."""


class Pin:
    """A named input digest, plus what to print when it moves.

    `kind` decides the reader:
      * `tinygrad:<relpath>:<line>` -- read that 1-based line of the installed
        package. Line numbers are the failure mode this study keeps hitting,
        so they are pinned, not looked up.
      * `file:<relpath>`           -- read a repo file whole.
      * `literal:<key>=<value>`    -- a declared parameter of the claim itself.
    """
    def __init__(self) -> None:
        self._items: list[tuple[str, str, str]] = []   # (name, kind, digest)

    def add(self, name: str, kind: str, data) -> str:
        d = sha(data)
        self._items.append((name, kind, d))
        return d

    # -- readers ----------------------------------------------------------
    def tinygrad_line(self, relpath: str, lineno: int, *, name: str = "",
                      must_contain: str = "") -> str:
        """Read and pin one line of the installed tinygrad. `must_contain` is
        a substring the line must carry or the pin is a wrong *line*, not a
        changed one -- a line that moved is a different claim."""
        path = tg_path(relpath)
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        if lineno < 1 or lineno > len(lines):
            raise PinDrift(f"tinygrad:{relpath}:{lineno} does not exist "
                           f"(file has {len(lines)} lines)")
        text = lines[lineno - 1]
        if must_contain and must_contain not in text:
            raise PinDrift(f"tinygrad:{relpath}:{lineno} does not contain "
                           f"{must_contain!r}.\n  it reads: {text.strip()!r}\n"
                           f"  the citation has drifted; find the new line and "
                           f"update the pin in the module, not the prose.")
        self.add(name or f"tinygrad:{relpath}:{lineno}",
                 f"tinygrad:{relpath}:{lineno}", text)
        return text

    def tinygrad_grep(self, pattern: str, *, name: str = "") -> tuple[int, str]:
        """Count package-wide matches and pin the hit list. Returns
        (count, joined `path:line: text` of every hit)."""
        root = os.path.dirname(tg_path("__init__.py"))
        out = subprocess.run(["grep", "-rn", "--include=*.py", "-e", pattern, root],
                             capture_output=True, text=True).stdout
        hits = [h for h in out.splitlines() if h.strip()]
        self.add(name or f"grep:{pattern}", f"grep:{pattern}", hits)
        return len(hits), "\n".join(hits)

    def file(self, relpath: str, *, name: str = "") -> str:
        path = os.path.join(ROOT, relpath)
        with open(path, "rb") as f:
            return self.add(name or f"file:{relpath}", f"file:{relpath}", f.read())

    def file_line(self, relpath: str, lineno: int, *, name: str = "",
                  must_contain: str = "") -> str:
        """Read and pin ONE line of a repo file. Same contract as
        `tinygrad_line`: a line that moved is a different claim, and a line
        that changed is pin drift."""
        path = os.path.join(ROOT, relpath)
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
        if lineno < 1 or lineno > len(lines):
            raise PinDrift(f"{relpath}:{lineno} does not exist "
                           f"(file has {len(lines)} lines)")
        text = lines[lineno - 1]
        if must_contain and must_contain not in text:
            raise PinDrift(f"{relpath}:{lineno} does not contain {must_contain!r}.\n"
                           f"  it reads: {text.strip()!r}\n"
                           f"  the citation has drifted; find the new line and "
                           f"update the pin in the module, not the prose.")
        self.add(name or f"file:{relpath}:{lineno}", f"file:{relpath}:{lineno}", text)
        return text

    def file_exists(self, relpath: str, *, name: str = "") -> bool:
        return os.path.exists(os.path.join(ROOT, relpath))

    def param(self, key: str, value) -> str:
        return self.add(f"param:{key}", f"literal:{key}", repr(value))

    # -- reporting --------------------------------------------------------
    def rows(self) -> list[tuple[str, str, str]]:
        return list(self._items)

    def digest(self) -> str:
        return sha([f"{n}|{k}|{d}" for n, k, d in self._items])

    def report(self) -> None:
        width = max((len(n) for n, _, _ in self._items), default=0)
        for n, k, d in self._items:
            print(f"    {n:<{width}}  {d}  {k}")

    def as_dict(self) -> dict[str, str]:
        return {n: d for n, _, d in self._items}


# The pin a claim is compared against. Filled in by
# `Pins.expect()` on first run and stored in the module's `EXPECTED` dict.
def check_pins(pins: Pin, expected: dict[str, str], *, accept_drift: bool = False) -> None:
    """Compare this run's digests against the ones recorded when the claim was
    written. Raises on drift."""
    now = pins.as_dict()
    missing = sorted(set(expected) - set(now))
    added = sorted(set(now) - set(expected))
    changed = sorted(k for k in set(now) & set(expected) if now[k] != expected[k])
    if not (missing or added or changed):
        return
    lines = ["INPUT DRIFT -- a claim below is not derived from the inputs it was written for."]
    for k in changed:
        lines.append(f"  CHANGED  {k}\n            recorded {expected[k]}  now {now[k]}")
    for k in missing:
        lines.append(f"  GONE     {k}  (recorded {expected[k]})")
    for k in added:
        lines.append(f"  NEW      {k}  = {now[k]}  (not in the recorded set)")
    lines += ["",
              "  Do not publish these numbers. Either the upstream input moved and the",
              "  claim must be re-derived and re-reviewed, or the pin is wrong. Look at",
              "  what changed, then either re-record with --accept-drift (a visible act,",
              "  and a diff on the EXPECTED dict) or fix the pin."]
    if accept_drift:
        print("\n".join(lines))
        print("\n  --accept-drift: continuing, and the digests above are now the recorded set.")
        print("  Re-run `python derivation/verify.py --write` to persist them.\n")
    else:
        raise PinDrift("\n".join(lines))


def tinygrad_version() -> str:
    import tinygrad
    from importlib.metadata import version
    try:
        return version("tinygrad")
    except Exception:
        return "unknown"


def tg_path(relpath: str) -> str:
    import tinygrad
    return os.path.join(os.path.dirname(os.path.abspath(tinygrad.__file__)), relpath)


# ------------------------------------------------------------------ values

class Value:
    """One derivable number, with the sentence that makes it quotable.

    `text`   the value as a document may quote it.
    `detail` the working, for the full report.
    """
    __slots__ = ("text", "detail", "unit")

    def __init__(self, text: str, detail: str = "", unit: str = "") -> None:
        self.text = text
        self.detail = detail
        self.unit = unit

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"Value({self.text!r})"


class ClaimError(RuntimeError):
    pass


class Module:
    """Collects claims for one module and runs them.

    `expected` is the recorded pin digests, `None` on a first run (in which
    case the module prints the digests it saw and says to write them).
    """

    def __init__(self, modname: str, doc: str, expected: dict[str, str] | None) -> None:
        self.modname = modname
        self.doc = doc
        self.expected = expected
        self.pins = Pin()
        self.claims: dict[str, callable] = {}
        self._printed_digests = False

    def claim(self, name: str):
        def deco(fn):
            if name in self.claims:
                raise ClaimError(f"{self.modname}: claim {name!r} declared twice")
            self.claims[name] = fn
            return fn
        return deco

    # -- pin bookkeeping --------------------------------------------------
    def check(self, *, accept_drift: bool = False) -> None:
        if self.expected is None:
            return
        check_pins(self.pins, self.expected, accept_drift=accept_drift)

    def seen_digests(self) -> dict[str, str]:
        return self.pins.as_dict()

    def print_digests(self) -> None:
        print(f"  input digests, {self.modname}:")
        self.pins.report()
        print(f"  combined: {self.pins.digest()}")

    # -- running ----------------------------------------------------------
    def run_one(self, name: str) -> Value:
        if name not in self.claims:
            raise ClaimError(f"{self.modname}: no claim named {name!r}. "
                             f"known: {', '.join(sorted(self.claims))}")
        v = self.claims[name]()
        if not isinstance(v, Value):
            raise ClaimError(f"{self.modname}:{name} returned {type(v).__name__}, "
                             f"not a derivation.Value")
        return v

    def run_all(self, names=None, *, out=sys.stdout) -> None:
        for name in names if names is not None else list(self.claims):
            v = self.run_one(name)
            print(f"{name}\n    {v.text}", file=out)


# ------------------------------------------------------------------ driver

def parse_flags(argv: list[str]) -> tuple[list[str], dict]:
    """Positional args are claim names. Flags: --accept-drift, --n N, --help."""
    names, flags = [], {"accept_drift": False, "n": None}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--accept-drift",):
            flags["accept_drift"] = True
        elif a == "--n":
            i += 1
            flags["n"] = int(argv[i])
        elif a.startswith("--n="):
            flags["n"] = int(a.split("=", 1)[1])
        elif a in ("-h", "--help"):
            flags["help"] = True
        else:
            names.append(a)
        i += 1
    return names, flags


def run(mod: Module, argv: list[str], *, report, banner: str = "") -> int:
    """Shared entry point. `report(mod, names, flags)` prints the full report;
    with claim names on the command line only those are resolved."""
    names, flags = parse_flags(argv)
    if flags.get("help"):
        print(mod.doc.strip())
        print("\nclaims:")
        for n in mod.claims:
            print(f"  {n}")
        return 0
    rc = 0
    try:
        if names:
            for n in names:
                try:
                    print(f"{mod.modname}:{n}\n    {mod.run_one(n).text}")
                except Exception as e:
                    print(f"{mod.modname}:{n}\n    FAILED: {type(e).__name__}: {e}")
                    rc = 1
        else:
            if banner:
                print(banner)
            report(mod, names, flags)
    except PinDrift as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        return 2
    except Exception as e:                                # noqa: BLE001
        print(f"[FATAL] {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc(limit=3)
        return 2
    try:
        mod.check(accept_drift=flags["accept_drift"])
    except PinDrift as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        return 2
    if mod.expected is None:
        print("\n  this module has no recorded digests yet (first run).")
        print("  To make its numbers pinnable, record them with:")
        print(f"    python derivation/verify.py --write {mod.modname}\n")
    return rc


def rule(title: str = "", width: int = 78) -> None:
    if title:
        print(f"\n{'-' * width}\n{title}\n{'-' * width}")
    else:
        print("-" * width)
