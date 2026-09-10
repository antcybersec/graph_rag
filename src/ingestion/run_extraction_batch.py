"""Runs entity/relationship extraction over the full corpus, batching several
documents into each LLM call (see extract_entities.make_token_budget_batches)
to stay well under free-tier Gemini request-count quotas, with per-document
checkpointing so an interrupted run never redoes an already-completed doc.

Output: data/results/extractions.jsonl, one line per document:
  {"doc_id": ..., "title": ..., "entities": [...], "relationships": [...]}

Usage:
  python -m src.ingestion.run_extraction_batch [--limit N] [--workers N] [--batch-tokens N]
"""
import argparse
import json
import os
import threading

from tqdm import tqdm

from src.common.token_tracker import TokenTracker
from src.ingestion.extract_entities import extract_from_documents_batch, make_token_budget_batches

CORPUS_PATH = "data/raw_dataset/corpus/corpus.jsonl"
OUTPUT_PATH = "data/results/extractions.jsonl"
FAILED_LOG_PATH = "data/results/extraction_failures.jsonl"

_write_lock = threading.Lock()


def load_done_ids(path: str) -> set:
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["doc_id"])
                except Exception:
                    pass
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="process only the first N docs (for testing)")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--batch-tokens", type=int, default=3500)
    parser.add_argument("--max-docs-per-batch", type=int, default=10)
    args = parser.parse_args()

    os.makedirs("data/results", exist_ok=True)

    docs = []
    with open(CORPUS_PATH) as f:
        for line in f:
            docs.append(json.loads(line))
    if args.limit:
        docs = docs[: args.limit]

    done_ids = load_done_ids(OUTPUT_PATH)
    todo = [d for d in docs if d["doc_id"] not in done_ids]
    print(f"Total docs: {len(docs)}, already done: {len(done_ids)}, remaining: {len(todo)}")

    if not todo:
        print("Nothing to do.")
        return

    batches = make_token_budget_batches(todo, target_tokens=args.batch_tokens, max_docs=args.max_docs_per_batch)
    print(f"Packed into {len(batches)} batches (avg {len(todo)/len(batches):.1f} docs/batch).")

    tracker = TokenTracker()
    out_f = open(OUTPUT_PATH, "a")
    fail_f = open(FAILED_LOG_PATH, "a")

    def worker(batch):
        batch_doc_ids = {d["doc_id"] for d in batch}
        try:
            results = extract_from_documents_batch(batch, tracker=tracker)
        except Exception as e:
            with _write_lock:
                for doc_id in batch_doc_ids:
                    fail_f.write(json.dumps({"doc_id": doc_id, "error": f"batch_call_failed: {e}"}) + "\n")
                fail_f.flush()
            return 0, len(batch_doc_ids)

        ok, missing = 0, []
        with _write_lock:
            for d in batch:
                doc_extraction = results.get(d["doc_id"])
                if doc_extraction is None:
                    missing.append(d["doc_id"])
                    continue
                rec = {
                    "doc_id": d["doc_id"],
                    "title": d["title"],
                    "entities": [e.model_dump() for e in doc_extraction.entities],
                    "relationships": [r.model_dump() for r in doc_extraction.relationships],
                }
                out_f.write(json.dumps(rec) + "\n")
                ok += 1
            out_f.flush()
            for doc_id in missing:
                fail_f.write(json.dumps({"doc_id": doc_id, "error": "missing_from_batch_response"}) + "\n")
            fail_f.flush()
        return ok, len(missing)

    n_ok, n_fail = 0, 0
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, b) for b in batches]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="extracting (batched)"):
            ok, fail = fut.result()
            n_ok += ok
            n_fail += fail

    out_f.close()
    fail_f.close()

    all_recs = tracker.all_records()
    total_tokens = sum(r["total_tokens"] for r in all_recs)
    print(f"\nDone. ok={n_ok} fail={n_fail}")
    print(f"Extraction LLM calls: {len(all_recs)}, total tokens used: {total_tokens}")
    tracker.dump("data/results/extraction_token_log.jsonl")


if __name__ == "__main__":
    main()
