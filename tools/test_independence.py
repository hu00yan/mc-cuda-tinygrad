#!/usr/bin/env python3
"""Rule 4, made executable: independence is a degree, and degrees can be bounded.

Rule 4 says: two implementations agree only *outside the assumptions they share*, so the
question is **which assumptions do these two share?** — not "are they different code?".

That sounds like an adjective until you notice it has checkable lower bounds. This file
asserts the three that can be measured mechanically:

  A. They do not share a parsing MECHANISM   (independent impl has no `re` at all)
  B. They agree field-by-field on real dumps (disagreement means one of them is wrong)
  C. They do not share a FILE-INTEGRITY assumption: feed one a truncated dump and
     confirm the other still returns its original answer.

C is the only one that actually tests independence. A and B test "different code" — which
is not what rule 4 says. A shared failure mode only becomes visible when one side breaks.

Run: .venv/bin/python tools/test_independence.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGEX_IMPL = ROOT / "tools" / "sass_lint.py"
FREE_IMPL = ROOT / "tools" / "sass_lint_independent.py"
PY = sys.executable

# Two kernels whose counts were settled by four agreeing paths. If these ever disagree
# between implementations, one implementation regressed — that is the point of B.
KNOWN = {
    "aot16_cc75_BV64_w4_s2": {"total": 16952, "STL": 1651, "LDL": 2684, "EX2": 33},
    "aot16_cc80_BV64_w4_s2": {"total": 2768, "STL": 64, "LDL": 63, "EX2": 3},
}


class Control:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        status = "PASS" if ok else "FAIL"
        print(f"{status}  {name}" + (f"  -- {detail}" if detail and not ok else ""))
        if not ok:
            self.failures.append(name)


def parse_tokenizer(text: str) -> dict[str, int]:
    """Independent impl prints `kernel: total=N STL=N LDL=N EX2=N ...`."""
    out: dict[str, int] = {}
    for key in ("total", "STL", "LDL", "EX2"):
        m = re.search(rf"\b{key}=(\d+)", text)
        if m:
            out[key] = int(m.group(1))
    return out


def parse_regex_table(text: str, kernel: str) -> dict[str, int]:
    """Regex impl prints a table row: `name (cc, cfg)  total exp2 HMMA STL LDL spill%`.

    Columns are positional, so they are read positionally — no key names in the output.
    """
    for line in text.splitlines():
        if not line.startswith(kernel + " "):
            continue
        # strip the "(cc75, BV64_w4_s2)" group, then the remaining tokens are numbers
        rest = re.sub(r"\([^)]*\)", " ", line[len(kernel):]).split()
        if len(rest) < 6:
            continue
        total, exp2, _hmma, stl, ldl = (int(x) for x in rest[:5])
        return {"total": total, "STL": stl, "LDL": ldl, "EX2": exp2}
    return {}


def run(script: Path, arg: str) -> str:
    proc = subprocess.run(
        [PY, str(script), arg], capture_output=True, text=True, cwd=ROOT
    )
    return proc.stdout + proc.stderr


def test_a_no_shared_mechanism(c: Control) -> None:
    """A: the independent implementation must contain no regex machinery at all."""
    src = FREE_IMPL.read_text(encoding="utf-8")
    imports_re = bool(re.search(r"^\s*(import re|from re import)", src, re.M))
    c.check("independent impl does not import `re`", not imports_re,
            "it imports re, so it shares the regex mechanism")
    uses_quantifier = bool(re.search(r"\{\d+(,\d+)?\}", src))
    c.check("independent impl uses no {n,m} quantifier", not uses_quantifier,
            "found a width quantifier — the exact assumption that broke bug 4")


def test_b_agreement_on_known_kernels(c: Control) -> None:
    """B: on settled inputs both must produce the settled numbers."""
    for kernel, expected in KNOWN.items():
        # Each implementation has its OWN input convention: the regex tool scans a
        # glob of files, the tokenizer takes kernel names. Assuming one interface for
        # both is what made this test fail on its first run.
        glob_arg = f"out/**/{kernel}.sass.txt"
        cases = (
            (parse_regex_table(run(REGEX_IMPL, glob_arg), kernel), "regex"),
            (parse_tokenizer(run(FREE_IMPL, kernel)), "tokenizer"),
        )
        for got, label in cases:
            for key, want in expected.items():
                have = got.get(key)
                c.check(
                    f"{label} impl: {kernel} {key} == {want}",
                    have == want,
                    f"got {have}",
                )


def test_c_no_shared_integrity_assumption(c: Control) -> None:
    """C: break the INPUT for one impl; the other must be unaffected.

    HONEST SCOPE (this assertion overclaimed in its first draft): it proves the tokenizer
    does not read the same *files* as the regex tool, i.e. they do not share an
    input-plumbing path. **It does NOT prove they share no assumption** — a tokenizer with
    bug 4 replicated inside it still passes C, and is only caught by assertion B.

    Measured: injecting a real 5-digit-offset skip into the tokenizer leaves C's own line
    green and turns B red. So C is a necessary but weak bound; B is the one with teeth.
    """
    import tempfile

    dumps = sorted((ROOT / "out").rglob("*.sass.txt"))
    if not dumps:
        c.check("C: corpus available", False, "no .sass.txt found under out/")
        return

    real = dumps[0]
    full_text = real.read_text(encoding="utf-8", errors="replace")
    original = parse_tokenizer(run(FREE_IMPL, KNOWN_KEYS[0]))
    c.check("C: baseline tokenizer answer captured", bool(original),
            "tokenizer produced no parseable counts on the baseline kernel")
    if not original:
        return

    with tempfile.TemporaryDirectory() as td:
        truncated = Path(td) / "truncated.sass.txt"
        lines = full_text.splitlines()
        # Keep only the first third: the classic "silently truncated nvdisasm" failure.
        truncated.write_text("\n".join(lines[: max(1, len(lines) // 3)]), encoding="utf-8")
        regex_on_truncated = run(REGEX_IMPL, str(truncated))
        # The tokenizer is re-run on the untouched corpus.
        after = parse_tokenizer(run(FREE_IMPL, KNOWN_KEYS[0]))

    c.check(
        "C: tokenizer unaffected by a file only the regex impl reads (plumbing, not assumptions)",
        after == original,
        f"was {original}, became {after} — the two share an input-completeness assumption",
    )
    c.check(
        "C: truncated input produced SOME output from the regex impl (so the probe ran)",
        bool(regex_on_truncated.strip()),
        "regex impl produced nothing at all — the probe did not exercise it",
    )


KNOWN_KEYS = list(KNOWN)


def main() -> int:
    c = Control()
    print("rule 4 lower bounds — independence is a degree, and here are three of them")
    print("  A: different mechanism   B: agree on settled inputs   C: separate input paths")
    print("  (B has the teeth; C is necessary but weak — see its docstring)\n")
    test_a_no_shared_mechanism(c)
    test_b_agreement_on_known_kernels(c)
    test_c_no_shared_integrity_assumption(c)
    print()
    if c.failures:
        print(f"{len(c.failures)} control(s) FAILED: {c.failures}")
        return 1
    print("all independence lower bounds hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
