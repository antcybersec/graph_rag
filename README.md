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

The knowledge graph schema is:

```
Document -[PART_OF]- Chunk -[MENTIONS]-> Entity -[RELATED_TO]-> Entity -[IN_COMMUNITY]-> Community
```

See `docs/architecture.md` for the full design rationale (entity resolution
strategy, schema choices, why per-document extraction, why the agentic loop
is capped, etc.) — most of it is already documented as comments at the top
of the relevant source files; the doc pulls it together in one place.

## Results (100-question public eval set, 300 graded runs)

Judged by a separate/stronger LLM (`JUDGE_MODEL`, see `.env`) on a 1-5 scale
for accuracy and completeness (against the reference answer) and
groundedness (against the pipeline's own retrieved context). Full
methodology in `src/eval/judge.py`.

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
python -m src.eval.run_benchmark --limit 10 --pipelines rag,graphrag,agentic
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
