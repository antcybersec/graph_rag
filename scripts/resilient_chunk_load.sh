#!/bin/bash
# Keeps restarting load_documents_chunks.py after it gets killed (e.g. by
# system-wide low-memory pressure) until it reports nothing left to do.
# Safe because the underlying script checkpoints per-batch.
#
# The python script exits 2 (QUOTA_EXHAUSTED_EXIT_CODE) when its own circuit
# breaker trips (many consecutive embedding failures) instead of grinding
# through every remaining batch for nothing. On that signal we cool down
# before retrying, instead of hammering an exhausted quota every few seconds.
#
# Observed in production (2026-09-08/09): a flat 30-min cooldown is the wrong
# fix here. Gemini's free-tier *embedding* quota is a per-day (RPD) cap, not
# a per-minute one -- across a 5+ hour run the breaker tripped on the exact
# same 6 doc-batches, in the exact same order, every single retry. A 30-min
# sleep can't recover a quota that only resets once every 24h, so the loop
# just burns cycles indefinitely without making progress.
#
# Fix: the first trip still gets a short cooldown (covers a genuine transient
# RPM blip). If it trips again on the very next attempt, that's evidence of a
# full daily-quota outage, so we sleep until past Gemini's reset instead of
# retrying every 30 min. Gemini free-tier quotas reset at midnight Pacific
# Time; we assume PDT (UTC-7, correct Mar-Nov) since this project runs in
# September -- verify against ai.google.dev if this script is still in use
# after a DST change, or if Google alters the reset schedule.
cd "$(dirname "$0")/.."
source .venv/bin/activate

QUOTA_COOLDOWN_SEC=1800    # first-trip cooldown: 30 min, for a transient blip
RESET_BUFFER_SEC=300       # extra margin past the assumed reset instant
PACIFIC_UTC_OFFSET_HOURS=7 # PDT; would be 8 for PST (Nov-Mar)

seconds_until_pacific_midnight() {
  local now_epoch midnight_utc_epoch reset_epoch diff
  now_epoch=$(date -u +%s)
  # midnight UTC of the next day, then shift forward by the PT->UTC offset
  # so the result lands on the next midnight Pacific instant.
  midnight_utc_epoch=$(date -u -v+1d -v0H -v0M -v0S +%s 2>/dev/null)
  if [ -z "$midnight_utc_epoch" ]; then
    midnight_utc_epoch=$(date -u -d "tomorrow 00:00:00" +%s) # GNU date fallback
  fi
  reset_epoch=$((midnight_utc_epoch + PACIFIC_UTC_OFFSET_HOURS * 3600))
  diff=$((reset_epoch - now_epoch))
  if [ "$diff" -lt 0 ]; then
    diff=$((diff + 86400))
  fi
  echo "$diff"
}

consecutive_quota_trips=0

for i in $(seq 1 200); do
  echo "=== attempt $i ==="
  python -m src.ingestion.load_documents_chunks --workers 1 2>&1 | tee -a data/results/chunk_load_resilient.log
  exit_code=${PIPESTATUS[0]}

  if tail -5 data/results/chunk_load_resilient.log | grep -q "Nothing to do"; then
    echo "=== ALL DONE ==="
    break
  fi

  if [ "$exit_code" -eq 2 ]; then
    consecutive_quota_trips=$((consecutive_quota_trips + 1))
    if [ "$consecutive_quota_trips" -ge 2 ]; then
      cooldown=$(($(seconds_until_pacific_midnight) + RESET_BUFFER_SEC))
      echo "=== quota circuit breaker tripped again immediately (${consecutive_quota_trips} in a row) -- treating this as a full daily-quota outage. Sleeping ${cooldown}s until past Gemini's assumed midnight-Pacific reset. ==="
    else
      cooldown=$QUOTA_COOLDOWN_SEC
      echo "=== quota circuit breaker tripped, cooling down ${cooldown}s before retry (first trip -- may just be transient) ==="
    fi
    sleep "$cooldown"
  else
    consecutive_quota_trips=0
    sleep 3
  fi
done
