"""metrics.json, computed from the two answer files and nothing else.

The brief is explicit that the judges recompute every number in this file
from the answer files and that a mismatch counts against repeatability. So
no figure here is tracked separately or carried over from a log: each one is
derived from the records as written, which makes a mismatch impossible rather
than unlikely.

`accuracy_pct` for the public set is read from the scorer's output, and
`accuracy_method` states in one sentence how it was measured. The hidden set's
accuracy stays null, as instructed.
"""
from __future__ import annotations

import argparse
import calendar
import json
import time
from pathlib import Path

from src.r2.trace import INPUT_USD_PER_MTOK, OUTPUT_USD_PER_MTOK

ACCURACY_METHOD = (
    "Fact-by-fact partial credit against the 50 published answers, graded by a "
    "model (JUDGE_MODEL) different from the one that produced the answers. Each "
    "question scores the fraction of the key's facts correctly given, minus a "
    "deduction for any asserted fact the key does not support; a wrong status "
    "caps the score at 0.5 and answering a not_in_corpus question with a value "
    "scores 0. accuracy_pct is the mean of those per-question scores. The "
    "per-question grades are in answers_public.jsonl.score.json."
)


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return float(ordered[idx])


def _run_time_s(records: list[dict]):
    """Wall-clock span of the run, from the answer file alone.

    This was reported as null because it was a parameter nobody passed, which
    is the one thing metrics.json is not allowed to be: every other number in
    it is derived from the answer files, so a judge recomputing them gets the
    same values. The span is derivable too -- each record carries its own
    `started_at` and `finished_at` -- so it is derived here rather than
    measured by the thing being measured.

    It is the span from the first question starting to the last one finishing,
    so for a run that stopped and resumed it reports the elapsed wall clock
    and not the time spent working. Both submitted files are written by one
    process in one sitting, where the two are the same.
    """
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    starts, ends = [], []
    for r in records:
        for key, into in (("started_at", starts), ("finished_at", ends)):
            raw = r.get(key)
            if not raw:
                continue
            try:
                into.append(calendar.timegm(time.strptime(str(raw), fmt)))
            except (ValueError, TypeError):
                continue
    if not starts or not ends:
        return None
    return round(max(ends) - min(starts), 1)


def summarise(records: list[dict], accuracy_pct=None, run_time_s=None) -> dict:
    n = len(records) or 1
    latencies = [r.get("latency_ms", 0) for r in records]
    llm_steps = [sum(1 for s in r.get("trace", []) if not s.get("tigergraph")
                     and "input_tokens" in s) for r in records]
    tg_steps = [sum(1 for s in r.get("trace", []) if s.get("tigergraph")) for r in records]
    inp = sum(r.get("input_tokens", 0) for r in records)
    out = sum(r.get("output_tokens", 0) for r in records)
    return {
        "questions": len(records),
        "accuracy_pct": accuracy_pct,
        "accuracy_method": ACCURACY_METHOD if accuracy_pct is not None else
                           "not measured: the hidden answer key is held by the judges",
        "not_in_corpus_answers": sum(1 for r in records if r.get("status") == "not_in_corpus"),
        "partial_answers": sum(1 for r in records if r.get("status") == "partial"),
        "answered": sum(1 for r in records if r.get("status") == "answered"),
        "input_tokens_total": inp,
        "output_tokens_total": out,
        "tokens_per_question_avg": round((inp + out) / n, 1),
        "latency_ms_avg": round(sum(latencies) / n, 1),
        "latency_ms_p50": _pct(latencies, 0.50),
        "latency_ms_p95": _pct(latencies, 0.95),
        "llm_calls_per_question_avg": round(sum(llm_steps) / n, 2),
        "tigergraph_queries_per_question_avg": round(sum(tg_steps) / n, 2),
        "trace_steps_per_question_avg": round(
            sum(len(r.get("trace", [])) for r in records) / n, 2),
        "answers_with_tigergraph_step": sum(
            1 for r in records if any(s.get("tigergraph") for s in r.get("trace", []))),
        "total_run_time_s": run_time_s if run_time_s is not None else _run_time_s(records),
        "cost_usd_total": round(inp / 1e6 * INPUT_USD_PER_MTOK
                                + out / 1e6 * OUTPUT_USD_PER_MTOK, 4),
        "cost_usd_per_question": round((inp / 1e6 * INPUT_USD_PER_MTOK
                                        + out / 1e6 * OUTPUT_USD_PER_MTOK) / n, 5),
    }


def _load(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build metrics.json from the answer files")
    ap.add_argument("--public", default="data/results/r2/answers_public.jsonl")
    ap.add_argument("--hidden", default="data/results/r2/answers_hidden.jsonl")
    ap.add_argument("--public-score", default="")
    ap.add_argument("--public-run-time", type=float, default=None)
    ap.add_argument("--hidden-run-time", type=float, default=None)
    ap.add_argument("--out", default="data/results/r2/metrics.json")
    args = ap.parse_args()

    accuracy = None
    score_path = Path(args.public_score or args.public + ".score.json")
    if score_path.exists():
        accuracy = json.loads(score_path.read_text(encoding="utf-8")).get("accuracy_pct")

    metrics = {
        "public": summarise(_load(args.public), accuracy, args.public_run_time),
        "hidden": summarise(_load(args.hidden), None, args.hidden_run_time),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
