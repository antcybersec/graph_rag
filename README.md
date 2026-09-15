# GraphRAG vs. RAG vs. Agentic GraphRAG

A head-to-head comparison of three retrieval strategies over the same corpus,
questions, and generation model, built for [hackathon eval set — see
`data/raw_dataset/README.md`]. The question this project answers: **does
adding graph structure, and then agentic control over retrieval, actually
improve answer quality — and at what cost in latency and tokens?**

All three pipelines share the same corpus (2,951 Wikipedia articles, loaded
into TigerGraph), the same 100-question public eval set, the same answer-
generation model, and the same LLM-as-judge scoring rubric. The only thing
that differs between them is *how they retrieve evidence*.

## The three pipelines

| Pipeline | Retrieval strategy | Code |
|---|---|---|
| **RAG** (baseline) | Vector similarity search over chunk embeddings only. No graph, no structure. The floor everything else is measured against. | `src/pipelines/rag/pipeline.py` |
| **GraphRAG** | Fixed sequence: entity linking (vector search over entity embeddings) → 1-hop graph traversal (`RELATED_TO` neighbors + their `MENTIONS`'d chunks) → answer synthesis grounded in both structured facts and chunk text. This is GraphRAG's "local search" mode; global search (community summaries) was skipped for the time budget — see `docs/architecture.md`. | `src/pipelines/graphrag/pipeline.py` |
| **Agentic GraphRAG** | An orchestrator LLM, run in a loop (max 5 iterations), picks the *next* action — `link_entities`, `graph_traverse`, `search_chunks`, or `answer` — based on the question and evidence collected so far, instead of following a fixed sequence. It also decides when evidence is "sufficient" and answers in that same call. Every action is logged to a trace for auditability. | `src/pipelines/agentic/pipeline.py` |
| **Event-graph GraphRAG** | One LLM call turns the question into a typed query plan. GSQL then traverses a structured event layer (`OlympicEvent`–`Games`–`Sport`–`Venue`, plus `PREV_GAMES`) that was built deterministically from the corpus infoboxes, and Python does the filtering, counting and argmax. It falls back to RAG when the graph can't answer. | `src/pipelines/event_graph/pipeline.py` |

The knowledge graph schema is:

```
Document -[PART_OF]- Chunk -[MENTIONS]-> Entity -[RELATED_TO]-> Entity -[IN_COMMUNITY]-> Community
```

See `docs/architecture.md` for the full design rationale (entity resolution
strategy, schema choices, why per-document extraction, why the agentic loop
is capped, etc.) — most of it is already documented as comments at the top
of the relevant source files; the doc pulls it together in one place.

## Results (100-question public eval set)

### Headline: exact match

Scored with `src/eval/exact_match.py`, which compares each answer
deterministically against the gold answer (no LLM). A pipeline that lists
several candidate answers is scored wrong.

| Pipeline | Exact match | Aggregation | Superlative | Multi-hop | Temporal | Lookup | Generation calls/q ³ | Tokens/q | Latency (s) | Doc P / R |
|---|---|---|---|---|---|---|---|---|---|---|
| RAG ¹ | 58/100 | 0/21 | 5/10 | 15/28 | 19/22 | 19/19 | 1.0 | ~9,980 ² | 9.8 | 0.30 / 0.69 |
| GraphRAG ¹ | 15/100 | 0/21 | 2/10 | 2/28 | 7/22 | 4/19 | 1.0 | ~14,330 ² | 16.9 | 0.12 / 0.20 |
| Agentic GraphRAG ¹ | 59/100 | 2/21 | 6/10 | 15/28 | 19/22 | 17/19 | 3.0 | ~15,600 ² | 29.4 | 0.15 / 0.30 |
| **Event-graph GraphRAG** | **99/100** | **21/21** | **10/10** | **27/28** | **22/22** | **19/19** | **1.01** | **~620** | **8.9** | **0.99 / 0.90** |

¹ From the first benchmark run (2026-09-11), before the local reranker was
added. The exact-match numbers are recomputed from those saved answers.
² Includes one judge call per question; later runs report pipeline-only tokens.
³ Generation-model calls only. The dashboard's "LLM calls" also counts local
embedding calls and, for the first run, the judge. Agentic's count comes from
its saved traces (orchestrator steps plus any forced final answer).
Event-graph's 1.01 is 100 planner calls plus one RAG fallback generation (pub-049).

Event-graph's single miss, pub-099, is a genuinely ambiguous question: two
events were held at Laura Biathlon & Ski Complex on 22 February 2014. The
pipeline names both, and the strict scorer counts that as wrong. 99 of 100
questions were answered from the graph and 1 via the RAG fallback. It also
answered all 50 hidden questions from the graph
(`data/results/hidden_answers_event_graph.jsonl`); one, eval-032, is ambiguous
in the same way.

**Why the jump.** Every question in the eval set asks about fields of the
`[Infobox Olympic event]` block: competitors, nations, venue, date, gold.
Aggregation questions span 8–43 documents, which no top-k retrieval can
count, and the LLM-extracted entity graph never stored those fields as
queryable properties. Modeling them as a graph turns counting, argmax,
venue+date lookup and "the previous Games" into traversals. See
`docs/architecture.md`, "Structured event layer".

### First run: LLM-judge scores (2026-09-11)

Judged by a separate LLM (`JUDGE_MODEL`, see `.env`) on a 1-5 scale for
accuracy and completeness (against the reference answer) and groundedness
(against the pipeline's own retrieved context). Full methodology in
`src/eval/judge.py`. In 14 of 298 judged rows the judge gave accuracy 4–5 to
an answer that exact match shows is wrong, often one that said "the corpus
has no information". The judge now runs only on a 20% sample (`--judge-rate`),
and exact match is the headline metric.

| Pipeline | Accuracy | Completeness | Groundedness | Doc precision | Doc recall | Latency (s) | Tokens/query | LLM calls |
|---|---|---|---|---|---|---|---|---|
| RAG | 3.54 | 3.56 | 4.27 | 0.30 | 0.69 | 9.8 | ~9,980 | 3.0 |
| GraphRAG | 1.89 | 1.85 | 4.19 | 0.12 | 0.20 | 16.9 | ~14,330 | 3.0 |
| **Agentic GraphRAG** | **3.61** | **3.56** | **4.35** | 0.15 | 0.30 | 29.4 | ~15,600 | 5.7 |

*(2 of 300 RAG rows have a null judge score — a judge-side parse failure,
excluded from the RAG averages above; see the dashboard for the full,
unfiltered breakdown.)*

### The headline finding: fixed-sequence GraphRAG underperforms plain RAG

This is the interesting/unexpected result, not a footnote. On this corpus
and question set, **GraphRAG's fixed entity-link → 1-hop-traverse → answer
sequence scores meaningfully *worse* than plain vector RAG** on every
accuracy-facing metric, and its retrieval (doc precision/recall) is worse
too — entity linking is evidently missing or mis-anchoring the right
starting entities often enough to drag the whole pipeline down, and the
fixed 1-hop traversal can't recover from a bad anchor the way a similarity
search over all chunks can.

**Agentic GraphRAG recovers the gap and edges out both**, because the
orchestrator can fall back to `search_chunks` when entity linking or graph
traversal doesn't pan out, rather than being locked into one path. It also
scores the highest groundedness of the three. The cost: ~3x the latency and
~1.9x the LLM calls of plain RAG per question.

### Breakdown by question type (mean accuracy)

| Question type | RAG | GraphRAG | Agentic GraphRAG |
|---|---|---|---|
| lookup | 5.00 | 2.26 | 4.79 |
| temporal | 4.64 | 2.59 | 4.64 |
| multi_hop | 3.14 | 1.29 | 3.29 |
| superlative | 3.00 | 2.80 | 3.80 |
| aggregation | 2.00 | 1.19 | 1.81 |

GraphRAG's deficit is worst on `lookup` and `multi_hop` — exactly the
categories where a wrong entity anchor is most costly. `aggregation` is hard
for all three pipelines (it requires reasoning over a count the corpus
context alone rarely spells out completely) and agentic control doesn't
meaningfully help there — a concrete limit of this architecture worth
naming rather than glossing over.

Full per-qtype charts, plus a per-question drill-down with side-by-side
answers and judge reasoning, are in the dashboard below.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in TG_HOST / TG_SECRET / GOOGLE_API_KEY, see comments in the file
```

Ingestion (one-time, builds the graph in TigerGraph):
```bash
python -m src.ingestion.create_schema
python -m src.ingestion.create_queries
python -m src.ingestion.load_documents_chunks
python -m src.ingestion.run_extraction_batch
python -m src.ingestion.build_entities_graph
```

Run the benchmark (checkpointed — safe to re-run after an interruption, see
`scripts/resilient_benchmark.sh` for a wrapper that retries through
transient provider rate limits):
```bash
python -m src.eval.run_benchmark
# or a quick subset:
python -m src.eval.run_benchmark --limit 10 --pipelines rag,graphrag,agentic,event_graph
# answer the hidden set (no gold answers, so no scoring):
python -m src.eval.run_benchmark --pipelines event_graph --questions data/raw_dataset/questions/eval_hidden.jsonl --output data/results/hidden_answers_event_graph.jsonl
```

The event layer is built once after the steps above, with no LLM calls:
```bash
python -m src.ingestion.build_event_graph
```

View the comparison dashboard:
```bash
streamlit run src/dashboard/app.py
```

### A note on generation providers

This project supports four generation providers (`GEN_PROVIDER` in `.env`):
`gemini`, `local` (Ollama), `openrouter`, `groq`. All four hit real free-tier
limits during this project's own 300-question run — daily request caps,
daily *token* caps (not just per-minute), and local RAM limits on a 16GB
machine. `.env.example`'s comments above `GEN_PROVIDER` document the exact
numbers observed and which knobs matter (TPM vs. TPD, `OLLAMA_NUM_CTX`,
etc.) — read those before switching providers on a large run. `PROGRESS.md`
has the blow-by-blow of the day this was fought through, if useful context
for why the code has retry/rate-limit plumbing in `src/common/llm.py`.

## Repo layout

```
src/common/       shared LLM/embedding client, TigerGraph connection, chunking, token tracking
src/ingestion/    schema creation, corpus loading, entity/relationship extraction, graph building
src/pipelines/    the three pipelines under comparison (rag, graphrag, agentic)
src/eval/         benchmark runner + LLM-as-judge scoring
src/dashboard/    Streamlit comparison dashboard
data/results/     benchmark_results.jsonl (raw eval output) + ingestion logs/checkpoints
docs/             architecture.md — full design rationale
```
