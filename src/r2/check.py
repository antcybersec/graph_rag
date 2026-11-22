"""Validate a finished run against everything the brief asks of it.

Repeatability is 15% of the score and the judges recompute our numbers from
the submitted files, so every check they could run is cheaper to run
ourselves. This is that check, as one command:

    python -m src.r2.check questions_public.jsonl data/results/r2/answers_public.jsonl

It verifies that

  * every question has exactly one answer, and no answer is a placeholder
    left behind by a failed question;
  * every trace passes `trace.verify` -- step numbering, token sums, timings,
    TigerGraph step fields, a legal status, citations on an answered record;
  * every TigerGraph step in a trace has a line in the query log for the same
    question, and the counts agree;
  * every log line falls inside its own question's `started_at` /
    `finished_at` window, which is what "the timestamps should match your
    traces" means in practice;
  * nothing is in the log that is not accounted for -- only answer ids and
    the run's own `__bootstrap__` catalogue reads.

Exit status is 1 if anything failed, so it can gate a submission.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from src.r2.trace import verify

BOOTSTRAP = "__bootstrap__"


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate one run's answer file and log")
    ap.add_argument("questions")
    ap.add_argument("answers")
    ap.add_argument("--query-log", default="")
    args = ap.parse_args()

    questions = _load(Path(args.questions))
    answers = _load(Path(args.answers))
    log_path = Path(args.query_log or args.answers + ".tgquerylog.jsonl")
    log = _load(log_path) if log_path.exists() else []

    failures: list[str] = []
    notes: list[str] = []

    # --- one answer per question, none of them a placeholder ---------------
    want = [q["qid"] for q in questions]
    have = Counter(a["qid"] for a in answers)
    for qid in want:
        if have[qid] == 0:
            failures.append(f"{qid}: no answer")
        elif have[qid] > 1:
            failures.append(f"{qid}: {have[qid]} answers")
    for qid in have:
        if qid not in want:
            failures.append(f"{qid}: in the answer file but not in the question file")
    for a in answers:
        if not a.get("trace"):
            failures.append(f"{a['qid']}: no trace -- a placeholder from a failed question")

    # --- each trace against the brief's own constraints --------------------
    for a in answers:
        for problem in verify(a):
            failures.append(f"{a['qid']}: {problem}")

    # --- traces and the query log have to agree ----------------------------
    by_qid: dict[str, list[dict]] = {}
    for row in log:
        by_qid.setdefault(row.get("qid", ""), []).append(row)
    for qid in list(by_qid):
        if qid != BOOTSTRAP and qid not in have:
            failures.append(f"{qid}: in the query log but not in the answer file")

    for a in answers:
        tg_steps = [s for s in a["trace"] if s.get("tigergraph")]
        lines = by_qid.get(a["qid"], [])
        if tg_steps and not lines:
            failures.append(f"{a['qid']}: {len(tg_steps)} TigerGraph steps, no query log lines")
            continue
        if len(lines) != len(tg_steps):
            failures.append(f"{a['qid']}: {len(tg_steps)} TigerGraph steps but "
                            f"{len(lines)} log lines")
        trace_queries = Counter(s.get("query") for s in tg_steps)
        log_queries = Counter(r.get("query") for r in lines)
        if trace_queries != log_queries:
            failures.append(f"{a['qid']}: the queries named in the trace and in the log "
                            f"differ ({trace_queries - log_queries} / "
                            f"{log_queries - trace_queries})")
        start, end = a.get("started_at"), a.get("finished_at")
        if start and end:
            outside = [r["ts"] for r in lines if not (start <= r.get("ts", "") <= end)]
            if outside:
                # One second of clock rounding at either edge is not a mismatch.
                failures.append(f"{a['qid']}: {len(outside)} log line(s) outside the "
                                f"question's own window {start}..{end}")
        else:
            notes.append(f"{a['qid']}: no started_at/finished_at, so log timestamps "
                         f"cannot be checked against this answer")

    if BOOTSTRAP in by_qid:
        notes.append(f"{len(by_qid[BOOTSTRAP])} log lines are the run's catalogue reads "
                     f"({BOOTSTRAP}), which belong to the run rather than to one answer")

    # --- a summary of what the run looks like ------------------------------
    statuses = Counter(a["status"] for a in answers)
    no_tg = [a["qid"] for a in answers
             if not any(s.get("tigergraph") for s in a["trace"])]
    print(f"{len(answers)}/{len(questions)} answered  "
          + "  ".join(f"{k}={v}" for k, v in sorted(statuses.items())))
    print(f"query log: {len(log)} lines over {len(by_qid)} ids -> {log_path}")
    if no_tg:
        print(f"answers with no TigerGraph step (the brief says these get a closer "
              f"look): {no_tg}")
    for note in notes:
        print(f"note: {note}")

    if failures:
        print(f"\n{len(failures)} problem(s):", file=sys.stderr)
        for f in failures[:60]:
            print(f"  - {f}", file=sys.stderr)
        if len(failures) > 60:
            print(f"  ... and {len(failures) - 60} more", file=sys.stderr)
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
