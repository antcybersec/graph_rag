"""Single shared Gemini wrapper for generation + embeddings, with token tracking.

All three pipelines (RAG, GraphRAG, Agentic) MUST call through this module
(not the SDK directly) so every call is uniformly logged to a TokenTracker --
that log is the raw material for the metrics dashboard's token-efficiency
and per-question breakdown.

Embeddings can run against either Gemini's API or a local open-source model
(BGE, via sentence-transformers) -- see `embed()`. Generation always goes
through Gemini; only the embedding quota was the bottleneck in practice
(free-tier RPD cap hit mid-ingestion, 2026-09-09), so EMBED_PROVIDER=local
is the escape hatch for that, independent of GEN_MODEL/generate().
"""
import os

# Must be set before HF `tokenizers` is imported (transitively, via
# sentence-transformers below) -- its Rust-side thread pool forking under our
# own ThreadPoolExecutor is what caused a SIGSEGV crash in production
# (2026-09-10, EMBED_PROVIDER=local, --workers 4): tokenizers spins up its
# own threads per call, and that plus concurrent Python threads deadlocked/
# corrupted state instead of raising a catchable Python exception.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import threading
import time
from typing import Optional, Type, TypeVar

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_random_exponential

from src.common.token_tracker import TokenTracker, CallRecord, default_tracker

load_dotenv()

_client: Optional[genai.Client] = None
T = TypeVar("T", bound=BaseModel)

# BGE's own model card instruction: prepending this to the QUERY side only
# (never the document/passage side) measurably improves retrieval for the
# bge-*-en-v1.5 family -- this is a property of those checkpoints, not a
# general trick, so it's gated on "bge" being in the model name.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_local_embed_models: dict = {}  # model_name -> loaded SentenceTransformer, lazy + cached
_local_embed_lock = threading.Lock()
# Separate lock serializing actual encode() calls across worker threads --
# MPS/PyTorch's device context is not safely reentrant from concurrent
# threads (root cause of the crash above); chunking + TigerGraph upserts in
# the callers still overlap across threads, only this device step is serial.
_local_encode_lock = threading.Lock()


def _get_local_embed_model(model_name: str):
    """Lazily load (and cache) a local sentence-transformers model.

    Imported lazily -- torch/sentence-transformers are heavy and only needed
    when EMBED_PROVIDER=local, so importing at module load time would slow
    down every process (including ones that only ever call generate())."""
    with _local_embed_lock:
        model = _local_embed_models.get(model_name)
        if model is None:
            from sentence_transformers import SentenceTransformer
            import torch

            device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
            model = SentenceTransformer(model_name, device=device)
            _local_embed_models[model_name] = model
        return model


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    return _client


class _RateLimiter:
    """Paces calls per-model so we stay under free-tier Gemini RPM caps instead
    of bursting and eating 429s (observed empirically: generation and embedding
    each have their own strict per-minute cap on the free tier, low enough that
    a handful of concurrent worker threads blows through it in seconds)."""

    def __init__(self, min_interval_sec: float = 6.5):
        self.min_interval = min_interval_sec
        self._lock = threading.Lock()
        self._last_call: dict = {}

    def wait(self, key: str):
        with self._lock:
            now = time.time()
            last = self._last_call.get(key, 0.0)
            wait_time = self.min_interval - (now - last)
            # reserve this slot immediately (while holding the lock) so concurrent
            # threads queue up correctly instead of racing on the same timestamp
            self._last_call[key] = max(now, last + self.min_interval)
        if wait_time > 0:
            time.sleep(wait_time)


_rate_limiter = _RateLimiter(min_interval_sec=float(os.environ.get("LLM_MIN_INTERVAL_SEC", 6.5)))


@retry(wait=wait_random_exponential(min=2, max=60), stop=stop_after_attempt(5), reraise=True)
def generate(
    prompt: str,
    *,
    model: Optional[str] = None,
    system_instruction: Optional[str] = None,
    response_schema: Optional[Type[T]] = None,
    thinking_budget: Optional[int] = None,
    tracker: Optional[TokenTracker] = None,
    pipeline: str = "",
    call_type: str = "generation",
    question_id: Optional[str] = None,
    context_tokens: int = 0,
) -> tuple:
    """Generate text (or a parsed pydantic object if response_schema is given).

    Returns (result, usage_dict). `result` is a `str` normally, or an
    instance of `response_schema` when that argument is provided.

    thinking_budget: pass 0 to disable extended thinking (cheap/deterministic
    bulk calls like entity extraction); leave None to use the model default
    (better for orchestrator reasoning / judge calls where quality matters).
    """
    client = _get_client()
    model_name = model or os.environ.get("GEN_MODEL", "gemini-3.8-flash")

    config_kwargs = {}
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction
    if response_schema is not None:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = response_schema
    if thinking_budget is not None:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)

    config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

    _rate_limiter.wait(f"generate:{model_name}")
    t0 = time.time()
    resp = client.models.generate_content(model=model_name, contents=prompt, config=config)
    latency = time.time() - t0

    usage = resp.usage_metadata
    record = CallRecord(
        pipeline=pipeline,
        call_type=call_type,
        model=model_name,
        input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
        output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        thoughts_tokens=getattr(usage, "thoughts_token_count", 0) or 0,
        total_tokens=getattr(usage, "total_token_count", 0) or 0,
        context_tokens=context_tokens,
        latency_sec=latency,
        question_id=question_id,
    )
    (tracker or default_tracker).log(record)

    if response_schema is not None:
        if resp.parsed is None:
            # structured-output parsing occasionally fails validation silently;
            # raising here lets tenacity's @retry above retry the whole call.
            raise RuntimeError(
                f"response_schema parsing returned None for model={model_name} "
                f"(finish_reason={getattr(resp.candidates[0], 'finish_reason', '?') if resp.candidates else '?'})"
            )
        return resp.parsed, record
    return resp.text, record


def _embed_local(texts: list, model_name: str, task_type: str, tracker, pipeline, question_id) -> tuple:
    """Embed via a local sentence-transformers (BGE) model -- no network, no quota."""
    model = _get_local_embed_model(model_name)

    if "bge" in model_name.lower() and task_type == "RETRIEVAL_QUERY":
        texts = [_BGE_QUERY_INSTRUCTION + t for t in texts]

    t0 = time.time()
    with _local_encode_lock:
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
    latency = time.time() - t0

    record = CallRecord(
        pipeline=pipeline,
        call_type="embedding",
        model=model_name,
        input_tokens=sum(len(t) // 4 for t in texts),  # rough estimate, consistent with the Gemini path below
        latency_sec=latency,
        question_id=question_id,
        note=f"batch_size={len(texts)} provider=local",
    )
    (tracker or default_tracker).log(record)
    return vectors, record


@retry(wait=wait_random_exponential(min=2, max=60), stop=stop_after_attempt(5), reraise=True)
def embed(
    texts: list,
    *,
    model: Optional[str] = None,
    task_type: str = "RETRIEVAL_DOCUMENT",
    dim: Optional[int] = None,
    tracker: Optional[TokenTracker] = None,
    pipeline: str = "",
    question_id: Optional[str] = None,
) -> tuple:
    """Embed a batch of texts. Returns (list[list[float]], usage_dict).

    Provider is chosen by EMBED_PROVIDER ("gemini", the default, or "local"
    for an on-device BGE model via sentence-transformers) -- see module
    docstring. Do not mix providers within one TigerGraph instance: Gemini
    and BGE embeddings live in unrelated vector spaces, so cosine similarity
    across the two is meaningless.
    """
    model_name = model or os.environ.get("EMBED_MODEL", "gemini-embedding-001")
    provider = os.environ.get("EMBED_PROVIDER", "gemini").lower()

    if provider == "local":
        return _embed_local(texts, model_name, task_type, tracker, pipeline, question_id)

    client = _get_client()
    dim = dim or int(os.environ.get("EMBED_DIM", 768))

    _rate_limiter.wait(f"embed:{model_name}")
    t0 = time.time()
    resp = client.models.embed_content(
        model=model_name,
        contents=texts,
        config=types.EmbedContentConfig(output_dimensionality=dim, task_type=task_type),
    )
    latency = time.time() - t0

    vectors = [e.values for e in resp.embeddings]
    if len(vectors) != len(texts):
        # Caught in production: some Gemini embedding models (gemini-embedding-2) silently
        # collapse a list of texts into ONE combined embedding instead of one-per-text, which
        # zip() elsewhere would otherwise truncate to and mis-pair without ever raising --
        # verified gemini-embedding-001 batches correctly, hence the EMBED_MODEL default below.
        raise RuntimeError(
            f"embed() got {len(vectors)} vectors back for {len(texts)} input texts using "
            f"model={model_name} -- this model likely doesn't support multi-text batching in "
            f"one call (gemini-embedding-001 is known to work; gemini-embedding-2 does not)."
        )
    record = CallRecord(
        pipeline=pipeline,
        call_type="embedding",
        model=model_name,
        input_tokens=sum(len(t) // 4 for t in texts),  # rough estimate; API doesn't return usage for embeddings
        latency_sec=latency,
        question_id=question_id,
        note=f"batch_size={len(texts)}",
    )
    (tracker or default_tracker).log(record)
    return vectors, record


if __name__ == "__main__":
    text, rec = generate("Say the word OK and nothing else.", thinking_budget=0)
    print("gen:", text, rec)
    vecs, rec2 = embed(["hello world"])
    print("embed dim:", len(vecs[0]), rec2)
