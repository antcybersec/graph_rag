"""Verify benchmark answers against corpus evidence, and score the LLM judge itself.

The LLM judge is unreliable on exactly the failure that matters: on the first
benchmark it gave accuracy 4-5 to 14 answers that are wrong, most of them
answers that said "the corpus does not contain this". This script measures
whether a TypeSafe System One judgment catches those, by asking a narrower
question than "how good is this answer": does this evidence support this claim?

Method. For every scored row it rebuilds the evidence from the question's own
gold documents (their infobox fields), pairs it with the pipeline's answer as
the claim, and asks `src.common.typesafe.check_claims`. Rows are then bucketed
against exact match, which is the ground truth here:

  judge_wrong   judge said 4-5, exact match says wrong -> should be FLAGGED
  agreed_right  judge said 4-5, exact match agrees     -> should be SUPPORTED

Measured 2026-09-19 over every scored row (194 claims, ~25 API calls at 8
claims per call):

  judge_wrong    14/14 flagged (12 contradicts, 2 says_nothing)
  agreed_right  176/180 supported -- 4 false alarms, all below the gate

So it catches every judge error, and disagrees with a correct answer 2.2% of
the time. Every one of those four (pub-028 rag+agentic, pub-044 rag+agentic)
scored 0.33-0.52 confidence, under the 0.8 gate, so none would be asserted --
they would be sent for review. The cost of that gate is over-referral: 20 of
180 controls fall below it, and 16 of those were right.

`--max-gold-docs` exists because evidence truncation corrupts the verdict. At a
cap of 6, four aggregation/superlative controls whose evidence spans 11-43
documents were flagged; raising the cap flipped all four to `supports`.
Truncating evidence does not make verification stricter, it makes it wrong.
Note that truncation does NOT explain the four residual false alarms above:
pub-028 has one gold document and pub-044 has ten, both well under the cap.

Usage:
  python -m src.eval.verify_evidence [--results PATH] [--max-gold-docs 20] [--limit N]
"""
import argparse
import json
from collections import Counter

from src.common.infobox import parse_event_doc
from src.common.typesafe import CONFIDENCE_THRESHOLD, check_claims, is_enabled, verdicts_to_status
from src.eval.exact_match import is_correct

RESULTS_PATH = "data/results/benchmark_results.jsonl"
QUESTIONS_PATH = "data/raw_dataset/questions/eval_public.jsonl"
CORPUS_PATH = "data/raw_dataset/corpus/corpus.jsonl"
OUTPUT_PATH = "data/results/evidence_verification.json"
MAX_CLAIM_CHARS = 400  # bounds tokens; the verdict rests on the answer's core claim


def load_corpus() -> dict:
    events = {}
    with open(CORPUS_PATH) as f:
        for line in f:
            event = parse_event_doc(json.loads(line))
            if event:
                events[event["event_id"]] = event
    return events


def evidence_for(row: dict, events: dict, max_docs: int):
    docs = [events[d] for d in row.get("gold_doc_ids", [])[:max_docs] if d in events]
    if not docs:
        return None
    return "\n".join(
        f"{e['title']}: competitors={e['competitors']}, nations={e['nations']}, "
        f"venue={e['venue']}, date={e['date_text']}, gold={e['gold']}" for e in docs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=RESULTS_PATH)
    parser.add_argument("--max-gold-docs", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None, help="cap rows per bucket (API calls cost money)")
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()

    if not is_enabled():
        raise SystemExit("TYPESAFE_API_KEY is not set -- see .env.example")

    events = load_corpus()
    gold = {json.loads(l)["qid"]: json.loads(l) for l in open(QUESTIONS_PATH)}

    rows = {}
    with open(args.results) as f:
        for line in f:
            row = json.loads(line)
            if "error" in row or (row.get("judge") or {}).get("accuracy") is None:
                continue
            rows[(row["qid"], row["pipeline"])] = row  # last write wins

    buckets = {"judge_wrong": [], "agreed_right": []}
    for row in rows.values():
        answers = gold[row["qid"]]["answer"]
        correct = is_correct(row["answer"], answers, row["question"], row.get("short_answer"))
        if row["judge"]["accuracy"] >= 4:
            buckets["judge_wrong" if not correct else "agreed_right"].append(row)
    for name in buckets:
        buckets[name].sort(key=lambda r: (r["qid"], r["pipeline"]))
        if args.limit:
            buckets[name] = buckets[name][:args.limit]

    report = {"confidence_threshold": CONFIDENCE_THRESHOLD, "max_gold_docs": args.max_gold_docs, "buckets": {}}
    for name, group in buckets.items():
        pairs, kept = [], []
        for row in group:
            evidence = evidence_for(row, events, args.max_gold_docs)
            if evidence:
                pairs.append((row["answer"][:MAX_CLAIM_CHARS], evidence))
                kept.append(row)
        if not pairs:
            continue
        checked = check_claims(pairs, pipeline="verify_evidence")
        flagged = sum(1 for c in checked if c["verdict"] in ("contradicts", "says_nothing"))
        detail = [{
            "qid": row["qid"], "pipeline": row["pipeline"],
            "judge_accuracy": row["judge"]["accuracy"],
            "verdict": c["verdict"], "confidence": c["confidence"],
            "grounded": c["grounded"], "needs_review": c["needs_review"],
            "reference_answer": row["reference_answer"],
        } for row, c in zip(kept, checked)]
        report["buckets"][name] = {
            "rows": len(pairs),
            "flagged": flagged,
            "supported": sum(1 for c in checked if c["verdict"] == "supports"),
            "needs_review": sum(1 for c in checked if c["needs_review"]),
            "verdicts": dict(Counter(c["verdict"] for c in checked)),
            "rollup": verdicts_to_status(checked),
            "detail": detail,
        }
        # judge_wrong: flagging is the win. agreed_right: supporting is the win.
        wanted = flagged if name == "judge_wrong" else report["buckets"][name]["supported"]
        print(f"{name}: {wanted}/{len(pairs)} as expected "
              f"({dict(Counter(c['verdict'] for c in checked))}, "
              f"{report['buckets'][name]['needs_review']} below the {CONFIDENCE_THRESHOLD} gate)")

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
