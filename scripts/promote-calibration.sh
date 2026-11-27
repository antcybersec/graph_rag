#!/usr/bin/env bash
# Promote the calibration fix on a fresh quota day. One command.
#
#   ./scripts/promote-calibration.sh
#
# Why this exists. The fix on `calibration-evidential-state` was gated on
# 27 Nov over 4 failing questions and 3 controls and measured +1.60 with no
# regressions (docs/round2-risks.md 2p). It was not promoted that day for an
# arithmetic reason only: promotion means re-running BOTH halves from one code
# version, because a public-only re-run would leave the two halves built by
# different code and make the 15% repeatability claim false. That is 241
# answering calls plus 50 for the judge, against ~175 left in the 27 Nov pool.
#
# Budget on a fresh 500-call day: ~291 used here, ~209 spare to recover with.
#
# It does NOT tag and does NOT push. It stops at the point where a human reads
# the number and decides. Reverting is one command, printed at the end.
set -uo pipefail
cd "$(dirname "$0")/.."

BRANCH="${BRANCH:-calibration-evidential-state}"
BANKED_PCT="${BANKED_PCT:-92.3}"   # what the tagged submission measures
OUT="data/results/r2"

say() { printf '\n[%s] == %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }
die() { printf '\n!! %s\n' "$*" >&2; exit 1; }

# -- preconditions ---------------------------------------------------------
[ -z "$(git status --porcelain)" ] || die "working tree is dirty. Commit or stash first."
[ "$(git rev-parse --abbrev-ref HEAD)" = main ] || die "not on main."
git rev-parse --verify "$BRANCH" >/dev/null 2>&1 || die "no branch $BRANCH"

BEFORE="$(git rev-parse HEAD)"
say "main is at $BEFORE; the banked submission measures ${BANKED_PCT}%"

# The tag is the safety net, so make sure it really is one before touching main.
git rev-parse --verify 'r2-submission^{commit}' >/dev/null 2>&1 \
  || die "tag r2-submission is missing -- that tag is the only way back. Stop."

# -- back up the banked answers, which the run is about to overwrite --------
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
say "backing up the banked answer files (suffix .banked-$STAMP)"
for f in answers_public.jsonl answers_hidden.jsonl metrics.json \
         answers_public.jsonl.score.json; do
  [ -f "$OUT/$f" ] && cp "$OUT/$f" "$OUT/$f.banked-$STAMP"
done

# -- merge, then prove the code is sane before spending any quota ----------
say "merging $BRANCH"
git merge --no-ff "$BRANCH" -m "merge: bind the composer's status to one rule, not two

Gated on 27 Nov: +1.60 over 7 questions, no regressions. docs/round2-risks.md 2p.

Signed-off-by: anant <170962632+antcybersec@users.noreply.github.com>" \
  || die "merge failed -- resolve it by hand"

say "tests (before any quota is spent)"
.venv/bin/python -m tests.test_r2 | tail -2
.venv/bin/python -m tests.test_r2 >/dev/null 2>&1 \
  || { git reset --hard "$BEFORE"; die "tests fail on the merge -- main put back to $BEFORE"; }

# -- the run ---------------------------------------------------------------
say "both halves, one code version (this is the ~291-call step)"
./finish.sh
rc=$?

if [ "$rc" != 0 ]; then
  say "finish.sh stopped early (rc=$rc).
  The banked files are at $OUT/*.banked-$STAMP and the tag r2-submission is
  untouched, so nothing is lost. Either re-run this script (finish.sh resumes)
  or put main back:
      git reset --hard $BEFORE
      for f in $OUT/*.banked-$STAMP; do cp \"\$f\" \"\${f%.banked-$STAMP}\"; done"
  exit 1
fi

# -- the decision ----------------------------------------------------------
say "the number that decides it"
NEW_PCT="$(.venv/bin/python -c "
import json
print(json.load(open('$OUT/metrics.json'))['public']['accuracy_pct'])
")"
printf '  banked  %s%%\n  now     %s%%\n' "$BANKED_PCT" "$NEW_PCT"

.venv/bin/python - <<PY
banked, new = float("$BANKED_PCT"), float("$NEW_PCT")
d = new - banked
print(f"\n  delta   {d:+.1f} points")
if d >= 2.0:
    print("  -> the gate predicted about +3.2. This is consistent. Promote.")
elif d > 0:
    print("  -> positive but below the gate's prediction. Read the per-question")
    print("     score before promoting: something else moved as well.")
else:
    print("  -> NOT an improvement. Put main back; the gate did not generalise.")
PY

say "per-question changes against the banked score"
.venv/bin/python - <<PY
import json
old = {d["qid"]: d for d in json.load(open("$OUT/answers_public.jsonl.score.json.banked-$STAMP"))["per_question"]}
new = {d["qid"]: d for d in json.load(open("$OUT/answers_public.jsonl.score.json"))["per_question"]}
moved = [(q, old[q]["score"], new[q]["score"]) for q in sorted(old)
         if q in new and abs(old[q]["score"] - new[q]["score"]) > 1e-6]
for q, a, b in moved:
    print(f"  {q}  {a:.2f} -> {b:.2f}  {b - a:+.2f}")
down = [q for q, a, b in moved if b < a]
print(f"\n  {len(moved)} moved, {len(down)} down: {down or 'none'}")
PY

say "done. Nothing has been tagged or pushed.

If the number is good:
    git tag -f -a r2-submission -m 'Round 2 submission'
    git push origin main
    git push origin r2-submission --force

If it is not:
    git reset --hard $BEFORE
    for f in $OUT/*.banked-$STAMP; do cp \"\$f\" \"\${f%.banked-$STAMP}\"; done"
