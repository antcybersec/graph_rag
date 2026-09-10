#!/bin/bash
# Keeps restarting run_benchmark.py after it gets killed (observed cause,
# 2026-09-10: system-wide low-memory pressure from OTHER apps on the
# machine -- our process is small, ~450MB, and just keeps getting picked as
# the OS's/sandbox's kill target when overall free RAM gets low). Safe
# because run_benchmark.py checkpoints per (qid, pipeline) pair, appending
# to data/results/benchmark_results.jsonl and skipping already-done pairs
# on the next run (see load_done_keys() in run_benchmark.py).
cd "$(dirname "$0")/.."
source .venv/bin/activate

RESULTS_PATH="data/results/benchmark_results.jsonl"
LOG_PATH="data/results/benchmark_full.log"
TOTAL_EXPECTED=300  # 100 public questions x 3 pipelines (rag, graphrag, agentic)

for i in $(seq 1 200); do
  echo "=== attempt $i ===" | tee -a "$LOG_PATH"
  python -m src.eval.run_benchmark --pipelines rag,graphrag,agentic >> "$LOG_PATH" 2>&1
  exit_code=$?

  done_count=$(wc -l < "$RESULTS_PATH" 2>/dev/null | tr -d ' ')
  echo "=== attempt $i finished: exit_code=$exit_code, done_count=${done_count:-0}/$TOTAL_EXPECTED ===" | tee -a "$LOG_PATH"

  if [ "${done_count:-0}" -ge "$TOTAL_EXPECTED" ]; then
    echo "=== ALL DONE ($done_count/$TOTAL_EXPECTED) ===" | tee -a "$LOG_PATH"
    break
  fi

  sleep 5
done
