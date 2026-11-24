"""Score our public answers against the published key.

`accuracy_pct` in metrics.json is our own measurement and the judges re-check
it with their key, so the measurement has to be one we would be willing to
show them. Theirs is "fact by fact with partial credit", so this is too:
a model that sees only the question, the published answer and ours assigns
the fraction of the key's facts we got right, and subtracts for facts we
added that the key does not contain -- because the brief says adding wrong
items lowers the score.

Two things keep it from flattering us:

* The judge must be a different model from the one that answered (JUDGE_MODEL),
  so a system that likes its own phrasing cannot score itself up.
* Status is scored separately and bounds the fact score. Answering a
  `not_in_corpus` question with a confident name is not 70% right; it is the
  failure the honesty criterion exists to punish, so it scores zero.

Nothing in here is ever imported by the answering path. It reads the gold
file, which the agent must never see.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

# `generate` is imported lazily inside judge_one, so the only `load_dotenv()`
# in the project (src/common/llm.py) had not run by the time JUDGE_MODEL was
# read below. It came back unset, the fallback took GEN_MODEL, and the system
# graded its own answers while metrics.json claimed an independent judge.
load_dotenv()

from pydantic import BaseModel, Field

JUDGE_SYSTEM = """You grade one answer against an official answer key, fact by fact.

You are given the question, the key's expected status, the key's answer, and
the system's answer. Grade only agreement with the key. The key is correct by
definition, even where it disagrees with what you know.

Score from 0.0 to 1.0:
  1.0  every fact the key states is present and correct, and nothing is
       asserted that contradicts the key
  partial  the fraction of the key's facts correctly given, minus a
       deduction for each asserted fact that contradicts the key or that the
       key does not support. A complete list that adds one wrong item scores
       below one.
  0.0  nothing in the key is correctly given, or the answer asserts a value
       where the key says nothing is established

Different wording, spelling, ordering or formatting is not an error. A name
in a different form (initials, accents, native spelling) counts as correct.
Extra correct context is not an error. Saying something is uncertain when the
key states it plainly IS an error.

When the key's status is not_in_corpus, the answer must decline to give a
value: a correct decline scores 1.0 and any invented value scores 0.0.
When the key's status is partial, the answer must give the supported part and
say what is missing; giving the whole thing as if established is an error.

List what the answer missed and what it got wrong, briefly.
"""


class Judgement(BaseModel):
    score: float
    status_correct: bool
    missing: list[str] = Field(default_factory=list)
    wrong: list[str] = Field(default_factory=list)
    note: str = ""


def judge_one(gold: dict, got: dict, model: str | None) -> dict:
    from src.common.llm import generate
    prompt = (f"QUESTION: {gold['question']}\n\n"
              f"KEY STATUS: {gold['status']}\n"
              f"KEY ANSWER: {json.dumps(gold['answer'], ensure_ascii=False)}\n"
              + (f"KEY NOTE: {gold['note']}\n" if gold.get("note") else "")
              + f"\nSYSTEM STATUS: {got.get('status')}\n"
              f"SYSTEM ANSWER: {json.dumps(got.get('answer'), ensure_ascii=False)}\n")
    verdict, _ = generate(prompt, model=model, system_instruction=JUDGE_SYSTEM,
                          response_schema=Judgement, thinking_budget=0,
                          pipeline="r2_scoring", call_type="judge")
    score = max(0.0, min(1.0, verdict.score))
    # The status is not a presentational detail: it is what the honesty
    # criterion measures, and a right-sounding sentence under the wrong
    # status is the error the brief singles out.
    if got.get("status") != gold["status"]:
        score = min(score, 0.5)
    if gold["status"] == "not_in_corpus" and got.get("status") != "not_in_corpus":
        score = 0.0
    return {"qid": gold["qid"], "score": score,
            "status_expected": gold["status"], "status_got": got.get("status"),
            "status_correct": got.get("status") == gold["status"],
            "missing": verdict.missing, "wrong": verdict.wrong, "note": verdict.note}


def citation_overlap(gold: dict, got: dict) -> dict:
    want, have = set(gold.get("gold_doc_ids") or []), set(got.get("citations") or [])
    if not want:
        return {"gold_docs": 0, "cited": len(have), "hit": None, "precision": None}
    return {"gold_docs": len(want), "cited": len(have),
            "hit": len(want & have),
            "recall": round(len(want & have) / len(want), 3),
            "precision": round(len(want & have) / len(have), 3) if have else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser(description="Score answers against the published key")
    ap.add_argument("questions", help="questions_public.jsonl (with answers)")
    ap.add_argument("answers", help="our answers_public.jsonl")
    ap.add_argument("--out", default="", help="default: <answers>.score.json")
    args = ap.parse_args()

    gold = {q["qid"]: q for q in (json.loads(l) for l in open(args.questions, encoding="utf-8") if l.strip())}
    got = {a["qid"]: a for a in (json.loads(l) for l in open(args.answers, encoding="utf-8") if l.strip())}
    model = os.environ.get("JUDGE_MODEL") or None

    # Whatever model actually wrote these answers is recorded in them. If the
    # judge is that same model, the accuracy figure is self-assessment and
    # every claim of independence built on it is false -- so say so here,
    # loudly, rather than letting it reach a results table.
    answered_by = sorted({str(step.get("model")) for a in got.values()
                          for step in (a.get("trace") or [])
                          if isinstance(step, dict) and step.get("model")})
    judge = model or os.environ.get("GEN_MODEL")
    self_graded = bool(judge) and answered_by == [judge]
    if self_graded:
        print(f"!! the judge ({judge}) is the model that wrote these answers.\n"
              f"   This score is self-assessment, not an independent grade. Set\n"
              f"   JUDGE_MODEL to a different model id and score again.\n")
    elif answered_by:
        print(f"judge {judge}; answers were written by {', '.join(answered_by)}")

    results, cites = [], []
    for qid, g in gold.items():
        if qid not in got:
            results.append({"qid": qid, "score": 0.0, "status_expected": g["status"],
                            "status_got": None, "status_correct": False,
                            "missing": ["no answer produced"], "wrong": [], "note": ""})
            continue
        r = judge_one(g, got[qid], model)
        c = citation_overlap(g, got[qid])
        r["citations"] = c
        results.append(r)
        cites.append(c)
        flag = "ok  " if r["score"] >= 0.999 else ("PART" if r["score"] > 0 else "MISS")
        print(f"{flag} {qid}  {r['score']:.2f}  {r['status_expected']:>13s} -> "
              f"{str(r['status_got']):13s} {('missing=' + '; '.join(r['missing'])[:90]) if r['missing'] else ''}"
              f"{(' wrong=' + '; '.join(r['wrong'])[:90]) if r['wrong'] else ''}")

    n = len(results)
    full = sum(1 for r in results if r["score"] >= 0.999)
    accuracy = sum(r["score"] for r in results) / n * 100 if n else 0.0
    status_ok = sum(1 for r in results if r["status_correct"])
    recalls = [c["recall"] for c in cites if c.get("recall") is not None]
    # Precision was being computed per question and then dropped. Evidence is
    # 10% of the score and citing six documents to support one fact spends it
    # just as surely as citing none, so both halves get reported.
    precisions = [c["precision"] for c in cites if c.get("precision") is not None]
    # Only six of the fifty public questions are not plain `answered`, so a
    # 50-question status average hides the entire honesty criterion. Split it.
    by_status: dict = {}
    for r in results:
        b = by_status.setdefault(r["status_expected"], {"n": 0, "status_ok": 0, "score": 0.0})
        b["n"] += 1
        b["status_ok"] += 1 if r["status_correct"] else 0
        b["score"] += r["score"]
    for b in by_status.values():
        b["accuracy_pct"] = round(b["score"] / b["n"] * 100, 1) if b["n"] else 0.0
        del b["score"]
    summary = {
        "questions": n,
        "accuracy_pct": round(accuracy, 1),
        "fully_correct": full,
        "status_correct": status_ok,
        "citation_recall_avg": round(sum(recalls) / len(recalls), 3) if recalls else None,
        "citation_precision_avg": round(sum(precisions) / len(precisions), 3) if precisions else None,
        "by_expected_status": by_status,
        "judge_model": judge,
        "answered_by": answered_by,
        "judge_independent": not self_graded,
        "per_question": results,
    }
    out = Path(args.out or args.answers + ".score.json")
    out.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\naccuracy {accuracy:.1f}%  ({full}/{n} fully correct, "
          f"{status_ok}/{n} correct status, citations "
          f"recall {summary['citation_recall_avg']} / "
          f"precision {summary['citation_precision_avg']})")
    for st in sorted(by_status):
        b = by_status[st]
        print(f"  {st:>13s}  {b['n']:2d} questions  "
              f"status {b['status_ok']}/{b['n']}  accuracy {b['accuracy_pct']:.1f}%")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
