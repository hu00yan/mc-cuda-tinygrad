#!/usr/bin/env python3
"""Rules 20 / 20b / 21 as a check: every table figure carries its instant and its subject.

The three rules were written tonight as prose, and Cindy's challenge was that a P2 rule
"tomorrow still works only because we did not make it P1 tonight". This is the attempt.

WHAT IT CHECKS
  A table row (line starting with '|') that quotes a measured figure must carry, on the
  same line, either
    - a timestamp (2026-10-24, 18:0x, ...), or
    - an explicit "stale on purpose" marker.

  A row is only in scope when it describes a CURRENT state of the world ("288 files are
  missing"). A row describing an event that happened once — the audit table's incident
  records, "3,842 vs 16,952" — is out of scope: an event does not expire, so a timestamp
  would be noise. That boundary is what a first run got wrong (27 rows in RULE-AUDIT.md,
  none of them stale), and it is why the check is applied to REPORTING-RULES.md only.

WHAT IT DOES NOT CHECK (printed on every run, rule 22)
  Whether the figure is TRUE. Whether the instant is the right one. Whether a row that
  looks like a measurement really is one. It greps shapes, so a figure written as prose
  rather than in a table passes silently.
"""
import re, sys, pathlib

# An instant is a DATE or a CLOCK TIME. Nothing else qualifies.
#
# Tim's radius test found the first version wrong: it also exempted the phrases
# "as measured", "same instant", "re-ran", "re-run" — which were meant to DESCRIBE
# a timestamp and instead SUBSTITUTED for one. A row reading "re-run this whenever
# you like" passed, carrying no instant and no subject at all. Four holes, one per
# phrase, each verified by feeding the phrase a legal-looking unlabelled figure.
TS = re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}")
NUM = re.compile(r"(?<![0-9a-zA-Z_.])[0-9]{2,}(?![0-9a-zA-Z_.])")
DELIBERATE = re.compile(r"deliberately stale|stays? (?:wrong|stale)|must stay|stale-and-labelled|no longer reproduces", re.I)
RULE = re.compile(r"^## (Rule )?[0-9]")


def check(path):
    """Return violations: rows quoting a 2+ digit figure with neither instant nor marker."""
    out = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if not line.startswith("|") or RULE.match(line):
            continue
        if not NUM.search(line):
            continue
        if TS.search(line) or DELIBERATE.search(line):
            continue
        out.append((i, line.strip()))
    return out


def main(argv):
    if len(argv) > 1 and argv[1] == "--self-test":
        return self_test()
    path = pathlib.Path(argv[1] if len(argv) > 1 else "tools/REPORTING-RULES.md")
    bad = check(path)
    if bad:
        print(f"FAIL {len(bad)} table row(s) quote a figure with no instant and no marker:")
        for i, line in bad[:10]:
            print(f"  line {i}: {line[:100]}")
        return 1
    print(f"table figures: OK ({path.name} — every measured figure carries an instant or a marker)")
    print("  NOT CHECKED: whether any figure is true, or whether the instant is the right one.")
    print("  NOT CHECKED: figures written as prose instead of in a table — they pass silently.")
    print("  SCOPE (Orion's three-way table, adopted after this check over-applied):")
    print("    describes a CURRENT state  -> needs instant + subject   [checked here]")
    print("    describes an EVENT record  -> needs nothing; it does not expire")
    print("    a FORMULA that varies      -> needs its validity range, not an instant")
    print("  Only the first row is checked. The other two are real and present in the")
    print("  rule file, and neither is enforced here.")
    return 0


def self_test():
    """Negative controls (rule 10): a check that cannot go red is not a check.

    Control 1  an unlabelled figure must FAIL
    Control 2  a timestamped figure must PASS
    Control 3  a deliberately-stale figure must PASS (it must be allowed to stay wrong)
    Control 4  a non-table line quoting a figure must PASS (prose is out of scope)
    Control 5  TIM'S RADIUS TEST: a phrase that merely DESCRIBES a timestamp must not
               exempt a row that has no instant. The first version exempted "as
               measured", "same instant", "re-ran", "re-run" — a row reading "re-run
               this whenever you like" passed with no instant and no subject.

    A check's exemption logic is the part that looks most intelligent and is least
    verified: every legal sample inside its radius vouches for the illegal ones
    outside. So the control is "a legal-looking row outside the radius", not a
    plausible-looking row inside it.
    """
    tmp = pathlib.Path("tools/.check_tables_selftest.md")
    cases = [
        ("unlabelled figure fails", "| a fresh claim | 1234 rows |", 1),
        ("timestamped passes", "| a claim | 1234 rows | *(2026-10-24 18:0xZ)* |", 0),
        ("deliberately-stale passes", "| a claim | 9 hits | *(deliberately stale, this is the finding)* |", 0),
        ("prose line is out of scope", "some text with 1234 in it but not a table", 0),
        ("radius: 'as measured' does not exempt", "| a claim | 1234 rows | as measured |", 1),
        ("radius: 're-run' does not exempt", "| a claim | 1234 rows | re-run whenever you like |", 1),
        ("radius: a real date does exempt", "| a claim | 1234 rows | 2026-10-24 |", 0),
    ]
    fails = 0
    try:
        for name, line, want in cases:
            tmp.write_text(line + "\n")
            got = check(tmp)
            ok = (bool(got) is bool(want))
            print(("PASS  " if ok else "FAIL  ") + name)
            fails += 0 if ok else 1
    finally:
        if tmp.exists():
            tmp.unlink()
    print("all negative controls passed" if not fails else f"{fails} negative control(s) FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
