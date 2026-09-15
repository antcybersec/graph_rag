"""Local cross-encoder reranking (BGE reranker), shared by all three pipelines.

Vector search and graph traversal are recall tools: they return passages that
are *topically* near the question, not necessarily ones that answer it. A
cross-encoder reads each (query, passage) pair jointly and scores relevance
much more sharply. So every pipeline pulls a wide candidate pool for free (ANN
search / traversal cost no LLM tokens) and sends only the top few reranked
passages to the LLM -- fewer context tokens, and better ones.

Runs locally, like the BGE embedder in src/common/llm.py: no API, no quota.
"""
import os
import threading

from src.common.llm import _local_encode_lock

RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-base")
# Escape hatch for A/B comparisons: when disabled, rerank() just keeps the
# first top_n candidates in their original retrieval order.
RERANK_ENABLED = os.environ.get("RERANK_ENABLED", "true").lower() != "false"

_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    with _model_lock:
        if _model is None:
            from sentence_transformers import CrossEncoder
            import torch

            device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
            _model = CrossEncoder(RERANK_MODEL, device=device)
        return _model


def rerank(query: str, items: list, *, text_of, top_n: int) -> list:
    """Return the `top_n` items most relevant to `query`, best first.

    `text_of(item)` extracts the passage text to score. Items with identical
    text are scored once and only the first is kept, since graph traversal and
    vector search routinely return the same chunk.
    """
    seen, unique = set(), []
    for item in items:
        text = text_of(item)
        if text and text not in seen:
            seen.add(text)
            unique.append(item)

    if not RERANK_ENABLED or len(unique) <= 1:
        return unique[:top_n]

    model = _get_model()
    # Same device lock as local embedding: MPS isn't safe to drive from
    # concurrent threads (see _local_encode_lock in src/common/llm.py).
    with _local_encode_lock:
        scores = model.predict([(query, text_of(i)) for i in unique], batch_size=16, show_progress_bar=False)
    ranked = sorted(zip(scores, range(len(unique))), key=lambda p: -p[0])
    return [unique[idx] for _score, idx in ranked[:top_n]]
