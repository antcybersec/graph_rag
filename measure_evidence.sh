#!/usr/bin/env bash
# Stage 1 of promoting the source-selection guidance: measure it on the only
# questions it can change, before spending a day's quota on a full pair.
#
#   ./measure_evidence.sh            # waits for quota, then measures
#
# The guidance is passed to one `_select` call site -- the source catalogue,
# reached only by the evidence route -- so the 9 public evidence questions are
# the complete set of public questions whose answers can differ. A test
# asserts the other five catalogues' prompts are byte-identical.
#
# Baseline, from the tagged pair: 6.5 of 9.
#   1.0  r2-pub-022 039 040 043 044 047
#   0.5  r2-pub-041
#   0.0  r2-pub-049  (retrieved the right record; failed on status)
#   0.0  r2-pub-050  (selected one record, the wrong one)
#
# Writes to the scratch directory only. It never touches
# data/results/r2/answers_*.jsonl, so a worse result costs nothing but quota
# and the tagged submission stays exactly as it is.
set -uo pipefail
cd "$(dirname "$0")"

Q_DIR="${Q_DIR:-$HOME/Downloads/r2}"
OUT="${OUT:-${SCRATCH:?set SCRATCH to a scratch directory}}"
EV="r2-pub-022,r2-pub-039,r2-pub-040,r2-pub-041,r2-pub-043,r2-pub-044,r2-pub-047,r2-pub-049,r2-pub-050"
PROBE_EVERY="${PROBE_EVERY:-600}"
MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-30}"

# shellcheck disable=SC1091
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
say() { printf '\n[%s] == %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }

say "tests first -- a measurement of broken code measures nothing"
python -m tests.test_r2 | tail -1 || exit 1

say "waiting for the model quota (one probe every ${PROBE_EVERY}s)"
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
    if "RESOURCE_EXHAUSTED" in t or "429" in t:
        sys.exit(2)
    if any(s in t for s in ("503", "UNAVAILABLE", "high demand",
                            "504", "DEADLINE_EXCEEDED")):
        sys.exit(4)
    sys.exit(3)
' 2>/dev/null; do
  rc=$?
  [ "$rc" = 3 ] && { say "the provider failed for a reason that is neither quota nor an outage -- stopping"; exit 1; }
  [ "$rc" = 4 ] && say "the model is overloaded (503/504), not out of quota -- still waiting"
  [ "$(date +%s)" -ge "$deadline" ] && { say "gave up waiting"; exit 1; }
  sleep "$PROBE_EVERY"
done
say "quota is back"

if ! docker ps --format '{{.Names}}' | grep -qx "${TG_CONTAINER:-tigergraph}"; then
  say "starting TigerGraph"
  docker start "${TG_CONTAINER:-tigergraph}" >/dev/null
  for _ in $(seq 1 90); do
    docker exec "${TG_CONTAINER:-tigergraph}" \
      /home/tigergraph/tigergraph/app/cmd/gadmin status 2>/dev/null \
      | grep -q "Online" && { sleep 5; break; }
    sleep 5
  done
fi

say "re-answering the 9 evidence questions (~25 calls)"
python -m src.r2.run "$Q_DIR/questions_public.jsonl" "$OUT/ev_after.jsonl" \
  --only "$EV" --query-log "$OUT/ev_after_log.jsonl" || exit 1

say "scoring them against the published answers (~9 calls)"
python - "$Q_DIR/questions_public.jsonl" "$OUT/ev_questions.jsonl" "$EV" <<'PY'
import json, sys
src, dst, ev = sys.argv[1], sys.argv[2], set(sys.argv[3].split(","))
with open(dst, "w", encoding="utf-8") as out:
    for line in open(src, encoding="utf-8"):
        if json.loads(line)["qid"] in ev:
            out.write(line)
PY
python -m src.r2.score "$OUT/ev_questions.jsonl" "$OUT/ev_after.jsonl" | tail -20

say "verdict: compare the per-question scores above with the baseline 6.5/9 in \
this script's header. Promote with a full paired run ONLY if it improved."
