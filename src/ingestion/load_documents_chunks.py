"""Chunks every corpus document, embeds the chunks, and loads
Document + Chunk vertices + PART_OF edges into TigerGraph.

Checkpointed at the doc-batch level (data/results/loaded_doc_ids.txt) so an
interrupted run never re-embeds chunks that already made it into TigerGraph --
important now that embedding calls are rate-limited (see src/common/llm.py's
_RateLimiter) and therefore slow/precious.

This is independent of entity extraction (which runs separately, see
run_extraction_batch.py) -- RAG only needs this step; GraphRAG/Agentic
additionally need entities loaded via load_entities_relationships.py.

Usage:
  python -m src.ingestion.load_documents_chunks [--limit N] [--workers N]
"""
import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from src.common.chunking import chunk_document, n_tokens
from src.common.llm import embed
from src.common.tg_conn import get_connection

CORPUS_PATH = "data/raw_dataset/corpus/corpus.jsonl"
CHECKPOINT_PATH = "data/results/loaded_doc_ids.txt"
DOCS_PER_BATCH = 12  # ~4.5 chunks/doc average -> ~50 chunks/embed call
EMBED_MAX_BATCH = 100  # Gemini embed_content hard cap per call; some doc
# batches run chunk-heavy and exceed this, which the API rejects wholesale
# (400 INVALID_ARGUMENT) rather than truncating -- so we sub-batch here.
CONSECUTIVE_FAIL_LIMIT = 6  # circuit breaker: this many failures in a row
# almost always means the embedding quota is exhausted for the window, not a
# fluke -- observed: without this, a quota outage burns hours retrying every
# remaining batch (each with its own 5x exponential-backoff retry) for
# nothing. Exit code 2 signals the wrapper script to cool down instead of
# immediately respawning.
QUOTA_EXHAUSTED_EXIT_CODE = 2

_write_lock = threading.Lock()


def load_corpus(limit=None):
    docs = []
    with open(CORPUS_PATH) as f:
        for line in f:
            docs.append(json.loads(line))
    return docs[:limit] if limit else docs


def load_checkpoint() -> set:
    if not os.path.exists(CHECKPOINT_PATH):
        return set()
    with open(CHECKPOINT_PATH) as f:
        return {line.strip() for line in f if line.strip()}


def append_checkpoint(doc_ids: list):
    with _write_lock:
        with open(CHECKPOINT_PATH, "a") as f:
            for d in doc_ids:
                f.write(d + "\n")


def batched(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def process_doc_batch(conn, docs: list):
    """Chunk + embed + upsert one batch of documents; returns doc_ids on success."""
    doc_vertices = [
        (
            d["doc_id"],
            {
                "title": d["title"],
                "url": d["url"],
                "wikidata_qid": d.get("wikidata_qid", ""),
                "wikipedia_pageid": str(d.get("wikipedia_pageid", "")),
                "approx_tokens": d.get("approx_tokens", 0),
            },
        )
        for d in docs
    ]

    all_chunks = []
    for d in docs:
        pieces = chunk_document(d["text"])
        for i, piece in enumerate(pieces):
            all_chunks.append(
                {
                    "chunk_id": f"{d['doc_id']}_c{i}",
                    "doc_id": d["doc_id"],
                    "chunk_index": i,
                    "text": piece,
                    "token_count": n_tokens(piece),
                }
            )

    texts = [c["text"] for c in all_chunks]
    vectors = []
    for i in range(0, len(texts), EMBED_MAX_BATCH):
        sub_vectors, _rec = embed(
            texts[i : i + EMBED_MAX_BATCH], task_type="RETRIEVAL_DOCUMENT", pipeline="ingestion"
        )
        vectors.extend(sub_vectors)

    chunk_vertices = []
    part_of_edges = []
    for c, vec in zip(all_chunks, vectors):
        chunk_vertices.append(
            (
                c["chunk_id"],
                {
                    "doc_id": c["doc_id"],
                    "chunk_index": c["chunk_index"],
                    "text": c["text"],
                    "token_count": c["token_count"],
                    "embedding": vec,
                },
            )
        )
        part_of_edges.append((c["chunk_id"], c["doc_id"], {}))

    with _write_lock:
        conn.upsertVertices("Document", doc_vertices)
        conn.upsertVertices("Chunk", chunk_vertices)
        conn.upsertEdges("Chunk", "PART_OF", "Document", part_of_edges)

    doc_ids = [d["doc_id"] for d in docs]
    append_checkpoint(doc_ids)
    return doc_ids, len(chunk_vertices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    os.makedirs("data/results", exist_ok=True)
    conn = get_connection()

    docs = load_corpus(args.limit)
    done_ids = load_checkpoint()
    todo = [d for d in docs if d["doc_id"] not in done_ids]
    print(f"Total docs: {len(docs)}, already loaded: {len(done_ids)}, remaining: {len(todo)}")

    if not todo:
        print("Nothing to do.")
        return

    batches = list(batched(todo, DOCS_PER_BATCH))
    print(f"Processing {len(batches)} batches of ~{DOCS_PER_BATCH} docs each.")

    n_docs_done, n_chunks_done, n_fail = 0, 0, 0
    consecutive_fails = 0
    tripped = False

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_doc_batch, conn, b): b for b in batches}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="loading Documents+Chunks"):
            batch = futures[fut]
            try:
                doc_ids, n_chunks = fut.result()
                n_docs_done += len(doc_ids)
                n_chunks_done += n_chunks
                consecutive_fails = 0
            except Exception as e:
                n_fail += len(batch)
                consecutive_fails += 1
                print(f"FAILED batch starting {batch[0]['doc_id']}: {str(e)[:200]}")
                if consecutive_fails >= CONSECUTIVE_FAIL_LIMIT:
                    tripped = True
                    print(
                        f"\nCIRCUIT BREAKER: {consecutive_fails} consecutive failures -- "
                        "stopping instead of grinding through every remaining batch "
                        "(almost certainly a quota outage, not a fluke)."
                    )
                    pool.shutdown(wait=False, cancel_futures=True)
                    break

    print(f"\nDone. Loaded {n_docs_done} docs, {n_chunks_done} chunks. Failed docs: {n_fail}.")
    if tripped:
        sys.exit(QUOTA_EXHAUSTED_EXIT_CODE)


if __name__ == "__main__":
    main()
