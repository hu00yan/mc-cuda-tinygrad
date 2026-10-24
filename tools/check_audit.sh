#!/usr/bin/env bash
# Orion's requirement: classify, then verify the sum equals the original row count,
# counted by a DIFFERENT command.
set -u
AUDIT=tools/RULE-AUDIT.md
RULES=tools/REPORTING-RULES.md

p=$(grep -cE '\*\*P\*\* \|$' "$AUDIT")
r=$(grep -cE '\*\*R\*\* \|$' "$AUDIT")
n=$(grep -cE '\*\*N\*\* \|$' "$AUDIT")
sum=$((p + r + n))

# row count by a different route than the label greps: count leading "| <digit>" lines
rows=$(grep -cE '^\| [0-9]' "$AUDIT")
# original rule count, from the rules file, third route
rules=$(grep -cE '^## (Rule )?[0-9]' "$RULES")

echo "P=$p R=$r N=$n  sum=$sum  audit_rows=$rows  rules_in_file=$rules"
if [ "$sum" -eq "$rows" ] && [ "$rows" -eq "$rules" ]; then
  echo "OK: sum == rows == rules"
  echo "  (this proves only that nothing is UNCOUNTED; whether each classification is"
  echo "   CORRECT is not checked — see RULE-AUDIT.md, class N.)"

  echo "  VERIFIED FAILURE MODE (Orion, 2026-10-24): a row that LIES passes this"
  echo "  check. Relabelling rule 4 from P to R — a false claim about its behaviour —"
  echo "  left this output OK. It counts labels and compares totals; it never asks"
  echo "  whether a label describes a real event. So a P/R/N value is UNVERIFIED BY"
  echo "  CONSTRUCTION: no mechanical check can tell that an event happened."
  echo "  What IS checked: every rule has a row, every row has one of three labels,"
  echo "  and the three counts equal both the row count and the rule count."
else
  echo "MISMATCH — a row is unlabelled or a rule is unaudited"
  exit 1
fi
