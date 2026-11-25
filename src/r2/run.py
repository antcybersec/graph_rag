"""Run the agent over a question file and write the submission answer file.

Deliberately boring: read questions, answer each one, write one JSON object
per line, verify every record against the brief's own constraints before the
file is considered written, and keep the TigerGraph query log alongside it.
A crash on question 37 must leave the first 36 answers and their log on disk,
so every line is flushed as it is produced.

Usage mirrors the brief exactly:
    python -m src.r2.run questions_public.jsonl answers_public.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from src.r2.trace import QueryLog, verify


def _is_quota_error(exc: Exception) -> bool:
    """A daily or per-minute cap, as opposed to a transient failure.

    Worth distinguishing: a transient error is worth retrying and a quota
    wall is not. The provider says which it is in the error body -- Gemini
    reports RESOURCE_EXHAUSTED with the metric name, and a free-tier daily
    cap also names a retry delay measured in hours.
    """
    text = f"{type(exc).__name__}: {exc}"
    return ("RESOURCE_EXHAUSTED" in text
            or "429" in text and ("quota" in text.lower() or "rate limit" in text.lower()))


def _prune_log(log_path, keep: set) -> int:
    """Keep only the query-log lines belonging to answers being kept.

    Anything else is from a question about to be answered again, and leaving
    it behind double-counts that question's queries against its trace.
    """
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0
    survivors = []
    for line in lines:
        if not line.strip():
            continue
        try:
            qid = json.loads(line).get("qid", "")
        except json.JSONDecodeError:
            continue
        if qid in keep:
            survivors.append(line)
    if len(survivors) != len([l for l in lines if l.strip()]):
        log_path.write_text("\n".join(survivors) + ("\n" if survivors else ""),
                            encoding="utf-8")
    return len([l for l in lines if l.strip()]) - len(survivors)


def main() -> int:
    ap = argparse.ArgumentParser(description="Answer a Round 2 question file")
    ap.add_argument("questions")
    ap.add_argument("answers")
    ap.add_argument("--query-log", default="", help="default: <answers>.tgquerylog.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated qids, for debugging")
    ap.add_argument("--resume", action="store_true",
                    help="keep the answers already in the output file and only "
                         "answer what is missing or failed")
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from src.r2.agent import Agent
    from src.r2.conn import describe, get_connection

    questions = [json.loads(line) for line in open(args.questions, encoding="utf-8")
                 if line.strip()]
    if args.only:
        wanted = {q.strip() for q in args.only.split(",")}
        questions = [q for q in questions if q["qid"] in wanted]
    if args.limit:
        questions = questions[:args.limit]

    answers_path = Path(args.answers)
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.query_log or str(answers_path) + ".tgquerylog.jsonl")

    # --resume exists because a free-tier daily quota can run out halfway
    # through a run, and re-answering the half that already succeeded spends
    # the next day's quota on work that is already done. A record counts as
    # done only if it has a trace: the placeholder written for a failed
    # question has none, and must be answered again.
    done: dict[str, dict] = {}
    if args.resume and answers_path.exists():
        for line in answers_path.open(encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("trace"):
                done[rec["qid"]] = rec
        print(f"resuming: {len(done)} answers already on disk, "
              f"{sum(1 for q in questions if q['qid'] not in done)} to go")

    print(f"connecting: {describe()}")
    # A fresh run starts a fresh log; a resumed run appends to the existing
    # one, because the lines for the answers it is keeping are already in
    # there and deliverable 5 has to cover every answer in the file beside it.
    #
    # But only the lines it is keeping. A question that failed part-way
    # through -- a 503 from an overloaded model, which is how r2-035 and
    # r2-036 failed -- already wrote its queries to the log before it died.
    # Re-answering it appends a second set, and the log then has more lines
    # for that question than its trace has steps, which is exactly what
    # `src.r2.check` exists to catch: it failed with six problems on those two
    # questions and exited 1, on a deliverable a judge runs. Old bootstrap
    # lines go too, since the resumed run writes its own.
    if args.resume and log_path.exists():
        dropped = _prune_log(log_path, keep=set(done))
        if dropped:
            print(f"query log: dropped {dropped} line(s) from the attempt that failed")
    query_log = QueryLog(log_path, fresh=not args.resume)
    conn = get_connection()
    agent = Agent(conn, query_log=query_log)
    print(f"catalogues: {len(agent.sports)} sports, {len(agent.games)} Games "
          f"({len(agent.bootstrap_steps)} bootstrap queries)")

    t0 = time.time()
    problems_total = 0
    quota_exhausted = False
    with open(answers_path, "w", encoding="utf-8") as out:
        for i, q in enumerate(questions, 1):
            if q["qid"] in done:
                out.write(json.dumps(done[q["qid"]], ensure_ascii=False) + "\n")
                out.flush()
                print(f"[{i:3d}/{len(questions)}] {q['qid']:12s} kept from the previous run")
                continue
            try:
                record = agent.answer(q["qid"], q["question"])
            except Exception as exc:
                if _is_quota_error(exc):
                    # Stop at once. Carrying on writes a placeholder for every
                    # remaining question and burns the retry budget on calls
                    # that cannot succeed -- which is exactly what happened the
                    # first time this hit: 25 questions spent producing 25
                    # empty records. Fail fast, keep what worked, and say how
                    # to finish.
                    print(f"\n!! the model provider's quota is exhausted at question {i} "
                          f"({q['qid']}).\n   {str(exc)[:300]}\n"
                          f"   {len(done)} answers are safe in {answers_path}.\n"
                          f"   When the quota resets, finish with:\n"
                          f"     python -m src.r2.run {args.questions} {args.answers} --resume",
                          file=sys.stderr)
                    quota_exhausted = True
                    break
                # A question that crashes still needs a line in the file, and
                # the honest status for a question we could not process is
                # not "answered". Losing one answer must not lose the run.
                import traceback
                traceback.print_exc()
                record = {"qid": q["qid"], "status": "not_in_corpus",
                          "answer": "", "explanation": f"the run failed on this question: {exc}",
                          "citations": [], "input_tokens": 0, "output_tokens": 0,
                          "latency_ms": 0, "trace": [], "route": "error"}
            done[record["qid"]] = record
            problems = verify(record)
            if problems:
                problems_total += len(problems)
                print(f"  !! {record['qid']} trace problems: {problems}", file=sys.stderr)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{i:3d}/{len(questions)}] {record['qid']:12s} {record['status']:14s} "
                  f"{record.get('route', ''):18s} "
                  f"{record['input_tokens']:6d}in {record['output_tokens']:5d}out "
                  f"{record['latency_ms']:6d}ms  {str(record['answer'])[:70]!r}")

    query_log.close()
    written = sum(1 for _ in answers_path.open(encoding="utf-8"))
    print(f"\nwrote {answers_path} ({written}/{len(questions)} answers) and {log_path} "
          f"in {time.time() - t0:.0f}s")
    if problems_total:
        print(f"WARNING: {problems_total} trace problems -- fix before submitting")
    if quota_exhausted:
        print("INCOMPLETE: the run stopped on a provider quota limit. Re-run with "
              "--resume once it resets.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
