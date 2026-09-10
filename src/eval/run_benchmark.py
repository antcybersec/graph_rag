"""Runs all three pipelines (RAG, GraphRAG, Agentic GraphRAG) over the public
eval set, judges each answer, and writes one row per (question, pipeline) to
data/results/benchmark_results.jsonl -- the raw material for the metrics
dashboard.

Checkpointed per (qid, pipeline) pair so an interrupted run (rate limits,
crashes) never redoes already-scored questions.

Usage:
  python -m src.eval.run_benchmark [--limit N] [--pipelines rag,graphrag,agentic]
"""
import argparse
import json
import os
import time

from tqdm import tqdm

from src.common.token_tracker import TokenTracker
from src.common.tg_conn import get_connection
from src.eval.judge import score_answer
from src.pipelines.rag import pipeline as rag_pipeline
from src.pipelines.graphrag import pipeline as graphrag_pipeline
from src.pipelines.agentic import pipeline as agentic_pipeline

EVAL_PUBLIC_PATH = "data/raw_dataset/questions/eval_public.jsonl"
OUTPUT_PATH = "data/results/benchmark_results.jsonl"

PIPELINES = {
    "rag": rag_pipeline,
    "graphrag": graphrag_pipeline,
    "agentic": agentic_pipeline,
}


def load_questions(limit=None):
    qs = []
    with open(EVAL_PUBLIC_PATH) as f:
        for line in f:
            qs.append(json.loads(line))
    return qs[:limit] if limit else qs


def load_done_keys(path: str) -> set:
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    done.add((r["qid"], r["pipeline"]))
                except Exception:
                    pass
    return done


def doc_prf(retrieved: list, gold: list) -> dict:
    retrieved_set, gold_set = set(retrieved), set(gold)
    if not retrieved_set and not gold_set:
        return {"precision": 1.0, "recall": 1.0}
    precision = len(retrieved_set & gold_set) / len(retrieved_set) if retrieved_set else 0.0
    recall = len(retrieved_set & gold_set) / len(gold_set) if gold_set else 0.0
    return {"precision": precision, "recall": recall}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pipelines", type=str, default="rag,graphrag,agentic")
    args = parser.parse_args()
    pipeline_names = args.pipelines.split(",")

    os.makedirs("data/results", exist_ok=True)
    conn = get_connection()
    questions = load_questions(args.limit)
    done = load_done_keys(OUTPUT_PATH)

    jobs = [(q, p) for q in questions for p in pipeline_names if (q["qid"], p) not in done]
    print(f"Total (question,pipeline) pairs: {len(questions) * len(pipeline_names)}, already done: {len(done)}, remaining: {len(jobs)}")

    out_f = open(OUTPUT_PATH, "a")

    for q, pname in tqdm(jobs, desc="benchmarking"):
        tracker = TokenTracker()
        t0 = time.time()
        try:
            result = PIPELINES[pname].answer_question(conn, q["question"], tracker=tracker, question_id=q["qid"])
        except Exception as e:
            row = {"qid": q["qid"], "pipeline": pname, "qtype": q.get("qtype"), "error": str(e)[:500]}
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
            continue
        latency = time.time() - t0

        reference_answer = "; ".join(q.get("answer", [])) if q.get("answer") else ""
        try:
            judge = score_answer(
                q["question"], reference_answer, result.get("context_text", ""), result["answer"],
                tracker=tracker, question_id=q["qid"], pipeline=pname,
            )
            judge_dict = judge.model_dump()
        except Exception as e:
            judge_dict = {"accuracy": None, "completeness": None, "groundedness": None, "reasoning": f"judge_failed: {e}"}

        totals = tracker.totals_for(q["qid"])
        retrieved_doc_ids = result.get("retrieved_doc_ids", [])
        gold_doc_ids = q.get("gold_doc_ids", [])
        prf = doc_prf(retrieved_doc_ids, gold_doc_ids)

        row = {
            "qid": q["qid"],
            "pipeline": pname,
            "qtype": q.get("qtype"),
            "question": q["question"],
            "reference_answer": reference_answer,
            "answer": result["answer"],
            "judge": judge_dict,
            "retrieved_doc_ids": retrieved_doc_ids,
            "gold_doc_ids": gold_doc_ids,
            "doc_precision": prf["precision"],
            "doc_recall": prf["recall"],
            "latency_sec": latency,
            **totals,
        }
        if pname == "agentic":
            row["num_steps"] = result.get("num_steps")
            row["stop_reason"] = result.get("stop_reason")
            row["strategy_changed"] = result.get("strategy_changed")
            row["trace"] = result.get("trace")

        out_f.write(json.dumps(row) + "\n")
        out_f.flush()

    out_f.close()
    print("Done.")


if __name__ == "__main__":
    main()
