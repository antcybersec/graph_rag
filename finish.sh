#!/usr/bin/env bash
# Finish both runs the moment the model quota comes back, then score, measure
# and validate. Start it and leave it:
#
#   nohup ./finish.sh > finish.log 2>&1 &
#   tail -f finish.log
#
# It probes with ONE cheap request every ten minutes rather than attacking the
# API, starts TigerGraph if it is stopped, and runs the public set with
# --resume so answers already on disk are kept rather than paid for twice.
# Every step is logged with a timestamp, and a quota wall mid-run leaves the
# answers that succeeded plus instructions, because run.py stops dead on one
# instead of writing a placeholder for every remaining question.
set -uo pipefail
cd "$(dirname "$0")"

Q_DIR="${Q_DIR:-$HOME/Downloads/r2}"
OUT="data/results/r2"
CONTAINER="${TG_CONTAINER:-tigergraph}"
PROBE_EVERY="${PROBE_EVERY:-600}"
MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-30}"

# shellcheck disable=SC1091
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false

say() { printf '\n[%s] == %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }

for f in "$Q_DIR/questions_public.jsonl" "$Q_DIR/questions_hidden.jsonl"; do
  [ -f "$f" ] || { echo "missing $f -- set Q_DIR to the folder holding them" >&2; exit 1; }
done

# Nothing should start while the agent is still being edited: the gate is a
# file, so clearing it is a one-liner from another terminal.
FREEZE="${FREEZE:-.freeze}"
hold() {
  [ -f "$FREEZE" ] || return 0
  say "held: $FREEZE exists (the code is still being changed). Waiting for it to go."
  until [ ! -f "$FREEZE" ]; do sleep 60; done
  say "freeze lifted"
}
hold

say "waiting for the model quota (one probe every ${PROBE_EVERY}s, giving up after ${MAX_WAIT_HOURS}h)"
deadline=$(( $(date +%s) + MAX_WAIT_HOURS * 3600 ))
until python -c '
import sys
from pydantic import BaseModel
from src.common.llm import generate
class T(BaseModel):
    ok: str
try:
    generate("say ok", response_schema=T, thinking_budget=0)
except Exception as e:
    t = f"{type(e).__name__}: {e}"
    sys.exit(2 if "RESOURCE_EXHAUSTED" in t or "429" in t else 3)
' 2>/dev/null; do
  rc=$?
  if [ "$rc" = 3 ]; then
    say "the provider failed for a reason that is not quota -- stopping so it gets looked at"
    exit 1
  fi
  [ "$(date +%s)" -ge "$deadline" ] && { say "gave up waiting"; exit 1; }
  sleep "$PROBE_EVERY"
done
say "quota is back"
# Checked again here, not only at startup. The wait is hours long and the
# first check happened before the night's editing: a quota reset arriving
# mid-edit would otherwise have started a submission run on half-changed code.
hold

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  say "starting TigerGraph"
  docker start "$CONTAINER" >/dev/null
  # Counts and queries read during warm-up are stale or fail outright, so wait
  # for the services rather than for the port.
  for _ in $(seq 1 90); do
    docker exec "$CONTAINER" /home/tigergraph/tigergraph/app/cmd/gadmin status 2>/dev/null \
      | grep -q "Online" && { sleep 5; break; }
    sleep 5
  done
fi
python -m src.r2.counts >/dev/null && say "graph is up"

# Both files must come from ONE code version, or a judge re-running the command
# gets answers that do not match the ones submitted -- and repeatability is 15%
# of the score. The evidence block has changed since the first 25 public answers
# were written, so they are discarded rather than resumed: a clean 50 costs
# ~130 requests and buys a file that reproduces.
say "public set (clean run -- single configuration)"
python -m src.r2.run "$Q_DIR/questions_public.jsonl" "$OUT/answers_public.jsonl"
public_rc=$?

say "hidden set"
python -m src.r2.run "$Q_DIR/questions_hidden.jsonl" "$OUT/answers_hidden.jsonl"
hidden_rc=$?

if [ "$public_rc" != 0 ] || [ "$hidden_rc" != 0 ]; then
  say "a run stopped early (public=$public_rc hidden=$hidden_rc). Re-run this script; \
--resume will keep what succeeded."
  exit 1
fi

say "scoring the public set against the published answers"
python -m src.r2.score "$Q_DIR/questions_public.jsonl" "$OUT/answers_public.jsonl"

say "metrics.json"
python -m src.r2.metrics

say "validating both runs"
python -m src.r2.check "$Q_DIR/questions_public.jsonl" "$OUT/answers_public.jsonl"
check_pub=$?
python -m src.r2.check "$Q_DIR/questions_hidden.jsonl" "$OUT/answers_hidden.jsonl"
check_hid=$?

# The hidden set has no gold answers, so this is the only check it gets. It
# costs nothing -- no model, no database -- and it is what caught Round 1's
# five wrong hidden answers.
say "auditing both files (no model, no quota)"
python -m src.r2.audit "$OUT/answers_public.jsonl"
python -m src.r2.audit "$OUT/answers_hidden.jsonl"

say "graph counts (deliverable 6)"
python -m src.r2.counts

if [ "$check_pub" = 0 ] && [ "$check_hid" = 0 ]; then
  say "everything passed. Review the accuracy above, then commit and tag:
    git add -A && git commit -m 'results: both runs'
    git tag -a r2-submission -m 'Round 2 submission'
    git push origin main --tags"
else
  say "validation found problems (public=$check_pub hidden=$check_hid) -- read them above"
  exit 1
fi
