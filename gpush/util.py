"""util.py -- small shared helpers. No third-party imports anywhere in gpush."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap

IS_TTY = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if IS_TTY else text


def bold(t: str) -> str: return _c("1", t)
def dim(t: str) -> str: return _c("2", t)
def green(t: str) -> str: return _c("32", t)
def red(t: str) -> str: return _c("31", t)
def yellow(t: str) -> str: return _c("33", t)
def cyan(t: str) -> str: return _c("36", t)


def hr(t: str = "", ch: str = "=") -> str:
    w = min(shutil.get_terminal_size((78, 24)).columns, 78)
    if not t:
        return ch * w
    return f"{ch * 2} {t} " + ch * max(0, w - len(t) - 4)


class Proc:
    """The result of one subprocess, kept whole so failures can be explained."""

    def __init__(self, argv, rc, out, err, cwd=None):
        self.argv, self.rc, self.out, self.err, self.cwd = argv, rc, out, err, cwd

    def __bool__(self) -> bool:
        return self.rc == 0

    def short(self, n: int = 400) -> str:
        blob = (self.err or self.out or "").strip()
        return blob if len(blob) <= n else blob[:n] + " ..."

    def say_cmd(self) -> str:
        return " ".join(self.argv)


def run(argv, *, cwd=None, timeout=None, stdin=None, env=None, check=False) -> Proc:
    """Run a command and capture it. Never raises for a non-zero exit."""
    try:
        p = subprocess.run(argv, cwd=cwd, timeout=timeout, input=stdin,
                           capture_output=True, text=True, env=env)
        r = Proc(argv, p.returncode, p.stdout, p.stderr, cwd)
    except subprocess.TimeoutExpired as e:
        r = Proc(argv, 124, e.stdout or "", f"timed out after {timeout}s", cwd)
    except (OSError, ValueError) as e:
        r = Proc(argv, 127, "", f"could not run {argv[0]}: {e}", cwd)
    if check and r.rc != 0:
        raise RuntimeError(f"{r.say_cmd()}\n{r.short(2000)}")
    return r


def jdump(obj, indent=2) -> str:
    return json.dumps(obj, indent=indent, sort_keys=True, default=str)


def table(rows, headers, widths=None) -> str:
    """A left-aligned fixed-width table. No box drawing, no colour."""
    cols = len(headers)
    cells = [[str(r[i]) for i in range(cols)] for r in rows]
    w = [len(str(h)) for h in headers]
    for r in cells:
        for i, c in enumerate(r):
            w[i] = max(w[i], len(c))
    if widths:
        for i, x in enumerate(widths):
            w[i] = min(w[i], x) if x else w[i]
    out = ["  ".join(str(h).ljust(w[i]) for i, h in enumerate(headers)).rstrip()]
    out.append("  ".join("-" * w[i] for i in range(cols)))
    for r in cells:
        out.append("  ".join(c.ljust(w[i])[:w[i]] if w[i] else c
                             for i, c in enumerate(r)).rstrip())
    return "\n".join(out)


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    os.replace(tmp, path)


def wrap(t: str, indent: str = "  ", width: int = 76) -> str:
    return "\n".join(textwrap.wrap(t, width=width, initial_indent=indent,
                                   subsequent_indent=indent))


def stamp() -> str:
    import datetime
    return datetime.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")


def which(name: str) -> str | None:
    return shutil.which(name)
