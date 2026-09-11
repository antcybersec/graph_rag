# Architecture & design rationale

This doc consolidates the "why" behind the schema, retrieval, and pipeline
design choices — most of it already lives as comments next to the code that
implements it; this is the one place that pulls it together. For results,
see the top-level `README.md`.

## Graph schema

```
Document -[PART_OF]- Chunk -[MENTIONS]-> Entity -[RELATED_TO]-> Entity -[IN_COMMUNITY]-> Community
```

Built in `src/ingestion/create_schema.py`. Two choices worth calling out:

- **`RELATED_TO` carries provenance fields from day one** (`source_url`,
  `source_type`, `observed_at`, `valid_from`, `valid_to`,
  `authority_score`) even though the current pipelines don't reason over
  temporal validity or conflicting facts. Retrofitting provenance onto an
  already-loaded graph is much more expensive than capturing it at load
  time, so it's captured now and left for future work to use.
- **Community vertices exist in the schema but aren't populated or used
  by any of the three pipelines** — see "What's deliberately out of scope"
  below.

## Corpus processing

- **Chunking** (`src/common/chunking.py`): paragraph-aware, token-based
  (~550 target tokens, 80 overlap), breaking on paragraph boundaries so
  entities/relations aren't split mid-sentence. Token counts use
  `tiktoken`'s `cl100k_base` as an approximate, consistent sizing metric
  (Gemini doesn't expose a local tokenizer); *exact* provider token counts
  are logged separately per-call at generation/embedding time
  (`src/common/token_tracker.py`).
- **Entity/relationship extraction** (`src/ingestion/extract_entities.py`):
  one LLM call per *document*, not per chunk. At this corpus size (~2,951
  docs, ~1,852 tokens average, comfortably inside a Gemini-class context
  window), per-chunk extraction would multiply LLM calls ~4x for no
  measurable accuracy benefit.
- **Entity resolution** (`src/ingestion/build_entities_graph.py`): entities
  are merged by *normalized name only*, deliberately not `(name,
  entity_type)`. The extractor is inconsistent about type across documents
  (e.g. tagging the same Olympic-games entity `Competition` in one doc and
  `Event` in another) — resolving on name alone avoids splitting what
  should be one entity into duplicates.

## Retrieval design

### Why GraphRAG's neighborhood traversal is capped tightly (LIMIT 8)

`entity_neighbors_1hop` (`src/ingestion/create_queries.py`) caps both the
related-entity and chunk selects at `LIMIT 8`, tightened down from an
initial `LIMIT 20`. Two independent reasons converged on the same fix:

1. **Context isn't free evidence — it's mostly noise past a point.**
   Observed in production (2026-09-10): one GraphRAG call against a hub
   entity ("2018 Winter Olympics", with hundreds/thousands of `RELATED_TO`
   edges and `MENTIONS` chunks) produced a 169k-token context and scored
   *worse* than plain RAG's 12k-token answer on the same question. A hub
   entity's raw neighborhood is mostly tangential facts, and dumping all of
   it into the prompt drowns the actually-relevant evidence.
2. **A hard rate-limit wall.** Separately, a free-tier provider's
   8,000-token-per-minute cap flatly rejected the ~15–19k-token contexts
   `LIMIT 20` produced.

`LIMIT 8` is kept as a deliberate design choice even where the quota
pressure isn't binding, because GraphRAG was already the *weaker*-
performing pipeline at `LIMIT 20` — there's no evidence more raw graph
context was helping accuracy, only hurting cost and latency.

Note the GSQL-side subtlety in `entity_neighbors_1hop`/the agentic
pipeline's traversal: `ACCUM` runs over *every* matched edge before `LIMIT`
trims the output vertex set, so the `LIMIT` bounds the result, not the
work done — both the GSQL query and the agentic pipeline's own defensive
cap on `related_edges` (`src/pipelines/agentic/pipeline.py`) account for
this.

### Why fixed-sequence GraphRAG underperforms RAG (see README for the numbers)

The retrieval-design implication of the headline result: entity linking is
a single point of failure in a fixed-sequence pipeline. When
`vector_search_entities` anchors on the wrong entity (or an entity whose
1-hop neighborhood doesn't actually contain the answer), GraphRAG has no
fallback — it synthesizes an answer from whatever the fixed traversal
returned, wrong anchor and all. Plain RAG's flat similarity search over
*all* chunks doesn't have this single point of failure, which is
consistent with GraphRAG's accuracy deficit being worst on `lookup` and
`multi_hop` questions (README: per-qtype breakdown) — exactly the
categories where getting the anchor entity right/wrong is most
consequential.

### Why the agentic loop closes most of that gap

`src/pipelines/agentic/pipeline.py`'s orchestrator can choose
`search_chunks` (bypassing the graph entirely) when `link_entities` or
`graph_traverse` isn't paying off, instead of being locked into one
strategy. Design choices in that loop, and why:

- **The orchestrator decides "sufficient evidence" and produces the final
  answer in the same call**, rather than a separate evaluator LLM call
  (the CRAG-style pattern). This halves the LLM-call count per question
  versus a separate-evaluator design — deliberately, given this project's
  observed free-tier rate limits (see README's provider section).
- **`MAX_ITERATIONS = 5` is non-negotiable.** Per the agentic-RAG survey /
  LangGraph literature, uncapped agentic retrieval loops are a documented
  failure mode ("retrieval thrash") — the model keeps gathering evidence
  without converging. Past the cap, the orchestrator is forced to answer
  with whatever it has, noting what's missing.
- **Every action is logged to a `trace`** (action, query/entity_ids taken,
  and eventually `stop_reason`) specifically so the metrics dashboard can
  show whether the agent changed strategy mid-investigation, how many
  steps it took, and why it stopped — this was a hackathon judging
  criterion ("agentic effectiveness"), not incidental instrumentation.
- **The action schema uses `Literal[...]`, not a plain `str`.** An earlier
  version let a local model (Ministral-3 via Ollama) emit syntactically
  valid-but-wrong actions like `"graph_traverse([entity:foo])"` — valid
  JSON, invalid action string, which fell through to an "unknown action"
  branch and aborted the whole investigation. `Literal` turns that into a
  schema-validation failure caught by `generate()`'s existing
  retry-on-parse-failure path, instead of a new failure mode this loop
  would have to detect and handle itself.
- **One failed action shouldn't torch the whole investigation.** Since
  `entity_ids` handed to `graph_traverse` can include orchestrator-
  hallucinated IDs (free-form LLM output, only best-effort filtered against
  known entities at the call site), `_run_graph_traverse` treats the entire
  TigerGraph lookup as one unit and converts any exception into "this
  action found nothing" — the orchestrator sees an empty result and tries
  a different action on the next iteration, rather than the whole
  `answer_question()` call crashing and losing every token already spent.

The cost of this flexibility: ~3x RAG's latency and ~1.9x RAG's LLM calls
per question (README numbers), since the orchestrator step itself is an
extra LLM call on top of whatever retrieval actions it chooses.

## Evaluation methodology

`src/eval/judge.py`: LLM-as-judge scoring on three 1–5 dimensions —
accuracy, completeness (both scored against the *reference* answer), and
groundedness (scored against the pipeline's own *retrieved context*, not
the reference — a pipeline can be faithful to bad context and still be
factually wrong, or unfaithful to good context, and the two failure modes
should be distinguishable). `JUDGE_MODEL` is deliberately a different,
stronger model than `GEN_MODEL` to reduce self-preference bias.

`src/eval/run_benchmark.py` is checkpointed per `(qid, pipeline)` pair.
Error rows are deliberately *not* counted as done, so a transient failure
(rate limit, quota, a crashed dependency call) is retried automatically on
the next run rather than permanently skipped.

## What's deliberately out of scope (Round 1 / time budget)

- **GraphRAG "global search"** (community-summary based retrieval, the
  other mode described in the GraphRAG literature) — the `Community`
  vertex and `IN_COMMUNITY` edge exist in the schema for this, but no
  pipeline currently populates or queries them. Only "local search"
  (entity-anchored) is implemented and measured here.
- **Reasoning over `RELATED_TO`'s temporal/provenance fields**
  (`valid_from`/`valid_to`/`authority_score`) — captured at load time (see
  schema section above) but no pipeline currently filters or ranks
  evidence by them. A natural next step: a query that discounts or
  excludes facts outside a question's implied time window, which would
  likely help the `temporal` question type specifically.
- **A separate evidence-sufficiency evaluator** for the agentic pipeline —
  folded into the orchestrator's own decision instead, for LLM-call-count
  reasons (see above). Worth a comparison in future work: does a dedicated
  evaluator call improve stopping decisions enough to justify the extra
  latency/tokens?
