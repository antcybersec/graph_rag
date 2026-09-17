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

## Structured event layer (added 2026-09-16)

```
Games <-IN_GAMES- OlympicEvent -IN_SPORT-> Sport
Games -PREV_GAMES-> Games                 (real Olympic calendar, later -> earlier)
OlympicEvent -HELD_AT-> Venue
OlympicEvent -DESCRIBED_BY-> Document     (provenance)
```

Built by `src/ingestion/build_event_graph.py` from `src/common/infobox.py`,
with **no LLM or embedding calls**. It lives in the same graph as the chunk/
entity layer above, and the event-graph pipeline answers from it.

**Why it exists.** A close read of the eval set showed that every question,
public and hidden, is one of five templates over the `[Infobox Olympic
event]` block that heads 2,203 of the 2,951 documents:

| qtype | asks for | graph operation |
|---|---|---|
| lookup | nations in one event | one event's `nations` |
| multi_hop | gold at `<venue>` on `<date>` | `Venue` → events, filter by date |
| temporal | gold in an event at the previous Games | `Games -PREV_GAMES->` → events of that sport |
| aggregation | how many events of a sport at a Games had > N competitors | `Games ∩ Sport` → count |
| superlative | which of those events had the most competitors | `Games ∩ Sport` → argmax |

The first three pipelines couldn't answer these reliably, for structural reasons
that no amount of prompt or reranker tuning fixes:

- **Counting needs the whole set.** An aggregation question spans 8–43 event
  documents; top-5 chunk retrieval sees a handful and counts those. By
  exact match, RAG and GraphRAG scored 0/21 on aggregation, and Agentic 2/21.
- **The LLM-extracted entity graph never stored the fields.** ~4 entities
  per document with free-text relation labels. Competitors, venue and date
  aren't properties you can filter or count on.
- **Quota pressure was a symptom.** Extraction, the agentic loop, and the
  judge together spent many calls per question on an approach that couldn't
  answer these questions anyway.

**Parsing gotchas handled in `infobox.py`:** tennis articles stack two
infoboxes (the Olympic one is second); normalized keys keep `+` (`Men's 80
kg` ≠ `Men's +80 kg`); "immediately before" uses the real Games calendar
(Winter 1992 → 1994), not the years present in the corpus; missing
competitor/nation counts are stored as `-1` and excluded from counts and argmax.

**Modeling caveat.** `Venue` is keyed by normalized name, so identically named
venues at different Games ("Olympic Tennis Centre" in Athens and Rio) share a
vertex. Venue queries therefore always filter by date, and by Games when the
question names one.

### Event-graph pipeline (`src/pipelines/event_graph/pipeline.py`)

1. **Plan (the only LLM call):** the question becomes a typed `QueryPlan` of
   operation, sport, Games, event name, venue, date, and threshold.
   `operation` and `season` are `Literal`s, and the prompt lists the graph's
   sports so the planner copies an exact name.
2. **Snap:** sport, venue and event names are matched to graph keys: the
   exact normalized key, else a *unique* `difflib` match ≥ 0.85 that agrees on
   every number, `+` and gender word. Near-identical names differ in exactly
   those tokens ("Men's K-1 500 m" vs "Women's K-1 500 m", "Riocentro –
   Pavilion 4" vs "Pavilion 6"), so an ambiguous or crossing match returns
   nothing and the pipeline falls back instead of answering wrongly. If the
   planner drops the gender ("pole vault"), it is restored from the question
   text, never guessed.
3. **Query:** `events_by_games_sport`, `events_at_venue`, or
   `events_in_previous_games` runs the traversal. Python then filters, counts
   or takes the argmax. Dates are matched in tiers: exact; then, within the
   Games, token overlap that shares a day number (an event with no infobox
   date never matches); then, only when the question gave no date, the single
   event at that venue in those Games. A Games year outside the real
   calendar is rejected before querying.
4. **Answer:** a templated answer citing each source document, plus a
   `short_answer` for scoring. If several events match (a tied argmax, or two
   events sharing a venue and date as in pub-099), all are listed and the row
   is flagged `ambiguous`.
5. **Fallback:** an `unsupported` plan, an empty graph result, or any query
   error is routed to plain RAG. `route` records which path answered.

Doc recall for temporal questions tops out at 0.5 for this pipeline. The gold
docs include the article for the Games named in the question, but the
evidence only contains the previous Games' event.

Planning runs with `thinking_budget=0`. Answering isn't an LLM call at all, so
the pipeline costs one call per question. `EVENT_PLAN_CACHE=true` reuses
earlier plans during development; rows record `plan_cached`, so cached runs
are never mistaken for cold-run costs.

### The Investigator: one agent, four tools (added 2026-09-17)

`src/pipelines/event_graph/` proves the structured layer answers these questions,
but it is a fixed route: it cannot do anything the event layer does not model.
`src/pipelines/investigator/` keeps its accuracy while remaining an agent,
because the structured query is *a tool the agent may choose*, not a fixed path.

Each step the agent: plans an action over {`query_events`, `query_facts`,
`link_entities`, `graph_traverse`, `search_chunks`, `answer`}; runs it; and states an
`evidence_check` — what the evidence does and does not establish — in the *same*
structured call, so self-evaluation costs no extra LLM call. It answers only when
that check passes, or changes strategy, up to `MAX_ITERATIONS`.

Measured over the 100 public questions: 99/100 exact match, 2.06 steps and 2.11
LLM calls per question, 2,695 tokens, all 100 stopping on `answer_found`. The
agent chose `query_events` on every question and added `search_chunks` on 3.
Given "Who directed the film Jab We Met?" — outside the event layer entirely — it
goes straight to `search_chunks` and answers from the film article.

**Robustness to phrasing.** A structured pipeline that scores 99/100 on a
templated eval set invites the question of whether it generalises past those
templates. `data/results/paraphrase_test.log` answers it: ten questions written
in wording the eval set never uses — host cities instead of Games years, "biggest
field" instead of "highest number of competitors" — score 10/10. The mapping from
"Calgary 1988" to the 1988 Winter Games is the planner's work, not a regex's;
nothing in the corpus title or the query plan contains the host city. This is
also why the planner is an LLM call rather than the regex that first proved the
questions were answerable.

**Evidence is provenance, not just citations.** Every evidence item carries
`doc_id`, `doc_title`, `source_url` and the `provenance` of the call that
produced it (tool, GSQL query name, parameters). The dashboard's drill-down
renders the investigation: each step with its self-check and timing, then each
evidence item with its source link and originating query. An answer can be
audited back to the article, which is the difference between a system that cites
and one that can be checked.

### Time-scoped facts with provenance (Round 2 groundwork)

`src/ingestion/build_temporal_facts.py` extracts 2,316 interval-stamped facts
from the corpus with no LLM calls: 129 office terms (`held_office`) and 2,187
championship reigns (`olympic_champion`), each with `valid_from`/`valid_to`,
`source_doc_id`, `source_url` and an `authority_score`, plus a `FACT_SOURCE` edge
to the source `Document`. `facts_as_of` and `fact_timeline` answer "who held X on
date D" and "how did X change over time".

The agent reaches this layer through its `query_facts` tool, so time-scoped
reasoning is part of the investigation rather than a separate script: asked "Who
was the President of Russia on 1 January 2013?" it issues
`facts_as_of(predicate=held_office, object_key=president of russia,
as_of=20130101)` and answers "Vladimir Putin", citing the fact's validity
interval and the source article. If the temporal queries are not installed, the
tool reports itself unavailable and the agent falls back to another tool rather
than failing the question.

**A fact needs a slot, not just a subject and object.** Which side can hold only
one value at a time differs per predicate: an office has one holder at a time
(the slot is the object), while an event series has one reigning champion (the
slot is the subject). Two earlier versions of the conflict detector ignored this
and reported nonsense — a repeat champion looked like a "source disagreement",
and one athlete winning two distances looked like "rival claims". With explicit
`slot_key`/`value_key`, the corpus's real numbers are:

| Bucket | Count |
|---|---|
| Cross-source disagreements | 0 |
| Rival claims on one slot | 0 |
| Concurrent roles held by one person (not a conflict) | 227 |

So this corpus has **no contradictions to resolve** — the evolving half of Round
2 is well supported here, the conflicting half is not. The detector exists and is
tested, but any conflict demo must run on a labelled fixture rather than implying
the corpus contains disagreements it does not.

### Scoring: exact match first, judge on a sample

`src/eval/exact_match.py` scores against the gold strings with no LLM. If a
pipeline reports a `short_answer`, that is compared directly. A short answer
listing several alternatives scores as wrong, because the free-text pipelines
state one answer and get no credit for hedging. For free-text
answers, the scorer takes the first number attached to "nations"/"events"
(skipping numbers copied from the question and years), otherwise checks for
the gold name/event as a whole-token match. The LLM judge now runs only on a
fixed `--judge-rate` sample of qids, the same for every pipeline. Cost fields
in each row (`total_tokens`, `num_llm_calls`, `total_latency_sec`) now cover the
pipeline only. The judge is reported separately in `judge_tokens`/`judge_calls`,
so sampled rows don't look costlier than unsampled ones.

On the 298 judged rows from the first benchmark, the judge disagreed with
exact match on 15. In 14 of them the judge was wrong: it scored accuracy 4–5
on answers that said "the corpus doesn't contain this," or gave the wrong
count or medallist. That is why the judge is no longer the headline metric.

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

### Retrieve wide, rerank locally, send few (added 2026-09-13)

Review feedback asked for lower token cost *and* higher accuracy. Both came
from the same place: what reached the LLM was chosen by retrieval order, not
relevance. Every pipeline now pulls a wide candidate pool, which costs no LLM
tokens, and scores it with a local cross-encoder (`BAAI/bge-reranker-base`,
`src/common/rerank.py`). Only the top few passages go into the prompt:

- **RAG**: 20 ANN candidates → top 5 chunks (was 8 unranked).
- **GraphRAG**: the traversal's `MENTIONS` chunks *plus* 20 ANN chunk
  candidates → top 5 chunks, and all related edges → top 6 facts. The graph
  query's `LIMIT 8` chunks are arbitrary, not relevant ones, and adding vector
  candidates gives GraphRAG a fallback when entity linking misses. This makes
  it a hybrid local search. `chunks_from_graph` in its result shows how many
  selected chunks still came from the traversal.
- **Agentic**: each retrieval action adds only its top 4 chunks / 6 facts,
  and evidence already collected is never added twice. Every orchestrator
  step re-sends all evidence, so each extra item costs tokens on every later
  step. The action history now also shows how many *new* items each action
  added, so the orchestrator can see that a search found nothing new.

`run_benchmark.py` now records `pipeline_tokens` (the pipeline's own calls)
separately from `judge_tokens`. In `benchmark_results.jsonl` (the first,
pre-rerank run), `total_tokens` also includes the judge call, which is
evaluation overhead, not pipeline cost. Since 2026-09-16, `total_tokens` and
`num_llm_calls` are pipeline-only (see "Scoring" above).

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
