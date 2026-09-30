"""Export the 50 hidden-question results in the shape the judges asked for.

The benchmark rows carry scoring fields that are meaningless for the hidden set
(there is no gold answer to score against), so this writes a purpose-built
artifact instead: for each question, the answer, the tokens it cost, and the
full agentic trace.

    python scripts/export_hidden_results.py
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/results/hidden_answers_investigator.jsonl"
JSON_OUT = ROOT / "data/results/hidden_50_results.json"
CSV_OUT = ROOT / "data/results/hidden_50_results.csv"


def main() -> int:
    rows = [json.loads(line) for line in SOURCE.open() if "error" not in json.loads(line)]
    rows.sort(key=lambda r: r["qid"])

    export = []
    for r in rows:
        export.append({
            "qid": r["qid"],
            "question": r["question"],
            "qtype": r.get("qtype"),
            "answer": r["answer"],
            "short_answer": r.get("short_answer"),
            "citations": r.get("retrieved_doc_ids") or [],
            "tokens": {
                "input": r.get("total_input_tokens"),
                "output": r.get("total_output_tokens"),
                "context": r.get("total_context_tokens"),
                "total": r.get("pipeline_tokens"),
            },
            "llm_calls": r.get("num_llm_calls"),
            "latency_sec": round(r.get("latency_sec", 0), 2),
            "agentic_trace": {
                "steps": r.get("num_steps"),
                "tools_used": r.get("tools_used"),
                "stop_reason": r.get("stop_reason"),
                "changed_strategy": r.get("strategy_changed"),
                "actions": [
                    {
                        "step": i + 1,
                        "action": step.get("action"),
                        "arguments": step.get("event_query") or step.get("fact_query") or step.get("query") or step.get("entity_ids"),
                        "evidence_check": step.get("evidence_check"),
                        "new_evidence_items": step.get("new_evidence"),
                        "tool_result": step.get("tool_result"),
                        "tool_latency_sec": step.get("tool_latency_sec"),
                    }
                    for i, step in enumerate(r.get("trace") or [])
                ],
            },
            "evidence": [
                {
                    "citation_id": e.get("citation_id"),
                    "source_document": e.get("doc_id"),
                    "source_url": e.get("source_url"),
                    "produced_by": (e.get("provenance") or {}).get("tool"),
                    "query": (e.get("provenance") or {}).get("query") or (e.get("provenance") or {}).get("operation"),
                }
                for e in (r.get("evidence") or [])
            ],
        })

    meta = {
        "pipeline": "Agentic GraphRAG (full) — orchestrator over 5 retrieval tools",
        "generation_model": "gemini-3.1-flash-lite",
        "embedding_model": "BAAI/bge-base-en-v1.5 (local)",
        "graph_backend": "TigerGraph Savanna 4.2.5",
        "questions": len(export),
        "totals": {
            "avg_tokens_per_question": round(sum(r["tokens"]["total"] or 0 for r in export) / len(export)),
            "avg_steps_per_question": round(sum(r["agentic_trace"]["steps"] or 0 for r in export) / len(export), 2),
            "avg_latency_sec": round(sum(r["latency_sec"] for r in export) / len(export), 2),
        },
        "note": ("Hidden questions ship without gold answers, so no accuracy column is "
                 "included here. Correctness was checked separately against a deterministic "
                 "oracle over the corpus infoboxes (src/eval/oracle.py)."),
    }

    JSON_OUT.write_text(json.dumps({"metadata": meta, "results": export}, indent=2, ensure_ascii=False))

    with CSV_OUT.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["qid", "question", "answer", "short_answer", "tokens_total",
                         "llm_calls", "latency_sec", "steps", "tools_used", "stop_reason",
                         "changed_strategy", "citations"])
        for r in export:
            writer.writerow([
                r["qid"], r["question"], r["answer"], r["short_answer"], r["tokens"]["total"],
                r["llm_calls"], r["latency_sec"], r["agentic_trace"]["steps"],
                " | ".join(t for t in (r["agentic_trace"]["tools_used"] or []) if t != "answer"),
                r["agentic_trace"]["stop_reason"], r["agentic_trace"]["changed_strategy"],
                " ".join(r["citations"]),
            ])

    print(f"wrote {JSON_OUT.relative_to(ROOT)} and {CSV_OUT.relative_to(ROOT)}")
    print(f"  {meta['questions']} questions · avg {meta['totals']['avg_tokens_per_question']} tokens · "
          f"{meta['totals']['avg_steps_per_question']} steps · {meta['totals']['avg_latency_sec']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
