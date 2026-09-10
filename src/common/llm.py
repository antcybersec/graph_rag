"""Single shared Gemini wrapper for generation + embeddings, with token tracking.

All three pipelines (RAG, GraphRAG, Agentic) MUST call through this module
(not the SDK directly) so every call is uniformly logged to a TokenTracker --
that log is the raw material for the metrics dashboard's token-efficiency
and per-question breakdown.

Embeddings can run against either Gemini's API or a local open-source model
(BGE, via sentence-transformers) -- see `embed()`. Generation can likewise run
against Gemini or a local Ollama model -- see `generate()` / GEN_PROVIDER.
Added 2026-09-10 after Gemini's free-tier *generation* quota (separate pool
from embeddings, and separate again from the embedding outage on 2026-09-09)
turned out to be a hard 500-requests/day cap with no billing account to lift
it -- EMBED_PROVIDER and GEN_PROVIDER are independent switches; embeddings
stay on local BGE either way since that's already resolved and free.
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

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_random_exponential

from src.common.token_tracker import TokenTracker, CallRecord, default_tracker

load_dotenv()

_client: Optional[genai.Client] = None
T = TypeVar("T", bound=BaseModel)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
# Local inference on a CPU/GPU-shared-memory Mac can genuinely take minutes for a
# large prompt (structured agentic-orchestrator calls especially) -- this is NOT
# the same "something is hanging" signal as GEMINI_REQUEST_TIMEOUT_MS, so it gets
# its own, much longer allowance.
OLLAMA_REQUEST_TIMEOUT_SEC = int(os.environ.get("OLLAMA_REQUEST_TIMEOUT_SEC", 300))

# OpenRouter: hosted inference (no local RAM cost, unlike Ollama), added
# 2026-09-11 as a third option alongside gemini/local -- free-tier (":free"
# model suffix) models are rate-limited but need no local compute at all.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_REQUEST_TIMEOUT_SEC = int(os.environ.get("OPENROUTER_REQUEST_TIMEOUT_SEC", 120))

# Groq: also hosted (no local RAM cost), no card required. Added 2026-09-11 after
# OpenRouter's free tier turned out to be a shared 50-requests/day cap across all
# free models. Groq's free tier is generous on request COUNT (up to 1000/day on
# some models) but several models cap total tokens-per-minute (TPM) at just 8000
# -- confirmed this flatly REJECTS (413, not throttles) a single ~15k-token
# GraphRAG-sized prompt outright. That's why LIMIT 8 (create_queries.py,
# graphrag/pipeline.py, agentic/pipeline.py) replaced LIMIT 20 the same day.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_REQUEST_TIMEOUT_SEC = int(os.environ.get("GROQ_REQUEST_TIMEOUT_SEC", 60))


def _to_strict_json_schema(schema: dict) -> dict:
    """OpenAI-style "strict" structured output (Groq included) requires, on EVERY
    object node: `additionalProperties: false`, AND every property key listed in
    `required` (an Optional[...] field is expressed by its nullable TYPE, e.g.
    anyOf: [{type}, {type: null}], not by omission from `required`) -- pydantic's
    model_json_schema() does neither by default. Confirmed via two separate 400s
    (\"must be set on every object\" / \"must be listed in required\"). Walks the
    schema recursively since a future schema could nest objects even though
    today's (JudgeScore, OrchestratorDecision) are flat."""
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            schema = {**schema, "additionalProperties": False}
            if "properties" in schema:
                schema["required"] = list(schema["properties"].keys())
        return {k: _to_strict_json_schema(v) for k, v in schema.items()}
    if isinstance(schema, list):
        return [_to_strict_json_schema(v) for v in schema]
    return schema

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


GEMINI_REQUEST_TIMEOUT_MS = int(os.environ.get("GEMINI_REQUEST_TIMEOUT_MS", 90_000))


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        # Explicit timeout matters here: confirmed in production (2026-09-10)
        # that some Gemini model aliases (gemini-3.8-flash, gemini-flash-latest
        # at the time) don't error on overload/unavailability -- they just
        # never respond. Without a client-side timeout, tenacity's @retry
        # below can't do its job: each "attempt" hangs indefinitely instead of
        # failing and moving to the next attempt/backoff.
        _client = genai.Client(
            api_key=os.environ["GOOGLE_API_KEY"],
            http_options=types.HttpOptions(timeout=GEMINI_REQUEST_TIMEOUT_MS),
        )
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


def _generate_local(
    prompt: str,
    model_name: str,
    system_instruction: Optional[str],
    response_schema: Optional[Type[T]],
    tracker: Optional[TokenTracker],
    pipeline: str,
    call_type: str,
    question_id: Optional[str],
    context_tokens: int,
) -> tuple:
    """Generate via a local Ollama model -- no network, no quota. No rate limiter
    needed (nothing external to protect); no thinking_budget support (most local
    models don't expose that control the way Gemini does, so it's silently
    ignored here rather than failing)."""
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    # Ollama defaults num_ctx to a small value (historically 2048-4096) regardless
    # of what the model itself supports -- silently truncating anything longer
    # instead of erroring. Our RAG/GraphRAG contexts run 10-20k+ tokens, so without
    # this override most of the actual context (and possibly the question/
    # instructions themselves, depending on template layout) gets silently dropped
    # before the model ever sees it -- observed in production as answers that
    # ignore the actual question entirely, not just "lower quality" ones.
    # 32768 was the first safe-looking value tried (see comment above) but its
    # KV cache alone pushed llama-server to ~10GB RSS -- plus additional
    # GPU/Metal buffer overhead on Apple Silicon's shared unified memory that
    # doesn't fully show up in per-process RSS -- enough to repeatedly OOM-kill
    # everything else on a 16GB machine (observed in production, 2026-09-11).
    # Real per-call contexts observed under this project's actual prompts
    # (GraphRAG's LIMIT-capped facts+chunks, agentic's accumulated evidence)
    # top out around 19k tokens, so 20480 keeps real headroom above that
    # ceiling while meaningfully cutting the fixed KV-cache memory cost vs
    # 32768. Raise this back up only alongside more free RAM.
    OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", 20480))
    body = {"model": model_name, "messages": messages, "stream": False, "options": {"num_ctx": OLLAMA_NUM_CTX}}
    if response_schema is not None:
        # Ollama's structured-output support: pass the target JSON schema directly
        # as `format` and it constrains decoding to match it.
        body["format"] = response_schema.model_json_schema()

    t0 = time.time()
    resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=body, timeout=OLLAMA_REQUEST_TIMEOUT_SEC)
    resp.raise_for_status()
    data = resp.json()
    latency = time.time() - t0

    content = data.get("message", {}).get("content", "")
    record = CallRecord(
        pipeline=pipeline,
        call_type=call_type,
        model=model_name,
        input_tokens=data.get("prompt_eval_count", 0) or 0,
        output_tokens=data.get("eval_count", 0) or 0,
        total_tokens=(data.get("prompt_eval_count", 0) or 0) + (data.get("eval_count", 0) or 0),
        context_tokens=context_tokens,
        latency_sec=latency,
        question_id=question_id,
        note="provider=local",
    )
    (tracker or default_tracker).log(record)

    if response_schema is not None:
        try:
            return response_schema.model_validate_json(content), record
        except Exception as e:
            raise RuntimeError(
                f"local model {model_name} returned invalid JSON for schema "
                f"{response_schema.__name__}: {content[:300]!r} ({e})"
            )
    return content, record


def _generate_openrouter(
    prompt: str,
    model_name: str,
    system_instruction: Optional[str],
    response_schema: Optional[Type[T]],
    tracker: Optional[TokenTracker],
    pipeline: str,
    call_type: str,
    question_id: Optional[str],
    context_tokens: int,
) -> tuple:
    """Generate via OpenRouter (OpenAI-compatible API) -- hosted, no local RAM cost.
    No rate limiter needed here either (OpenRouter enforces its own free-tier
    limits server-side and returns a normal 429 on breach, which tenacity's
    @retry on generate() already handles)."""
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    body = {"model": model_name, "messages": messages}
    if response_schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": response_schema.__name__,
                "schema": response_schema.model_json_schema(),
                "strict": True,
            },
        }

    t0 = time.time()
    resp = requests.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        json=body,
        timeout=OPENROUTER_REQUEST_TIMEOUT_SEC,
    )
    resp.raise_for_status()
    data = resp.json()
    latency = time.time() - t0

    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    record = CallRecord(
        pipeline=pipeline,
        call_type=call_type,
        model=model_name,
        input_tokens=usage.get("prompt_tokens", 0) or 0,
        output_tokens=usage.get("completion_tokens", 0) or 0,
        total_tokens=usage.get("total_tokens", 0) or 0,
        context_tokens=context_tokens,
        latency_sec=latency,
        question_id=question_id,
        note="provider=openrouter",
    )
    (tracker or default_tracker).log(record)

    if response_schema is not None:
        try:
            return response_schema.model_validate_json(content), record
        except Exception as e:
            raise RuntimeError(
                f"openrouter model {model_name} returned invalid JSON for schema "
                f"{response_schema.__name__}: {content[:300]!r} ({e})"
            )
    return content, record


def _generate_groq(
    prompt: str,
    model_name: str,
    system_instruction: Optional[str],
    response_schema: Optional[Type[T]],
    tracker: Optional[TokenTracker],
    pipeline: str,
    call_type: str,
    question_id: Optional[str],
    context_tokens: int,
) -> tuple:
    """Generate via Groq (OpenAI-compatible API) -- hosted, no local RAM cost,
    no card required. See GROQ_BASE_URL comment above for the TPM caveat."""
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    body = {"model": model_name, "messages": messages}
    if response_schema is not None:
        schema = _to_strict_json_schema(response_schema.model_json_schema())
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": response_schema.__name__, "schema": schema, "strict": True},
        }

    t0 = time.time()
    resp = requests.post(
        f"{GROQ_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
        json=body,
        timeout=GROQ_REQUEST_TIMEOUT_SEC,
    )
    resp.raise_for_status()
    data = resp.json()
    latency = time.time() - t0

    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    record = CallRecord(
        pipeline=pipeline,
        call_type=call_type,
        model=model_name,
        input_tokens=usage.get("prompt_tokens", 0) or 0,
        output_tokens=usage.get("completion_tokens", 0) or 0,
        total_tokens=usage.get("total_tokens", 0) or 0,
        context_tokens=context_tokens,
        latency_sec=latency,
        question_id=question_id,
        note="provider=groq",
    )
    (tracker or default_tracker).log(record)

    if response_schema is not None:
        try:
            return response_schema.model_validate_json(content), record
        except Exception as e:
            raise RuntimeError(
                f"groq model {model_name} returned invalid JSON for schema "
                f"{response_schema.__name__}: {content[:300]!r} ({e})"
            )
    return content, record


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
    Ignored entirely under GEN_PROVIDER=local (see module docstring).
    """
    model_name = model or os.environ.get("GEN_MODEL", "gemini-3.8-flash")
    provider = os.environ.get("GEN_PROVIDER", "gemini").lower()

    if provider == "local":
        return _generate_local(
            prompt, model_name, system_instruction, response_schema,
            tracker, pipeline, call_type, question_id, context_tokens,
        )
    if provider == "openrouter":
        return _generate_openrouter(
            prompt, model_name, system_instruction, response_schema,
            tracker, pipeline, call_type, question_id, context_tokens,
        )
    if provider == "groq":
        return _generate_groq(
            prompt, model_name, system_instruction, response_schema,
            tracker, pipeline, call_type, question_id, context_tokens,
        )

    client = _get_client()

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
