# Agentic GraphRAG on TigerGraph — Round 1 submission

An investigation agent that answers questions over a 2,951-document Wikipedia
corpus by choosing between structured graph queries, graph traversal, temporal
facts and vector search — and a second planner that answers the same questions
with **no generative model at all**.

Architecture diagrams: [`architecture-diagram.md`](architecture-diagram.md).

## Results

100 public questions, scored by exact match against gold (`src/eval/exact_match.py`,
no LLM in the loop). Every row is measured on the code in this repo.

| Pipeline | Correct | Calls/q | Tokens/q | Latency | Doc P / R |
|---|---|---|---|---|---|
| RAG (baseline) | 67/100 | 2.0 | 3,586 | 6.8s | 0.43 / 0.73 |
| GraphRAG (fixed local search) | 67/100 | 2.0 | 3,952 | 10.2s | 0.43 / 0.73 |
| Agentic GraphRAG | 70/100 | 4.08 | 6,065 | 20.2s | 0.46 / 0.70 |
| Event-graph (fixed route) | 99/100 | 1.02 | 619 | 8.9s | 0.99 / 0.90 |
| **Investigator (the agent)** | **99/100** | 2.08 | 3,412 | 13.2s | 0.96 / 0.85 |
| **Jev planner (zero LLM generation)** | **99/100** | 1.41 | 2,267 | **1.67s** | 0.99 / 0.89 |

Hidden set: 50/50 answered, **49/49 correct** on every question a deterministic
oracle can check (`data/results/hidden_answers_jev.jsonl`).

## What actually moved the number

**The questions are templates over infobox fields.** Every one of the 150
questions asks for `competitors`, `nations`, `venue`, `date` or `gold` from the
box at the top of an Olympic event article. Two consequences drove the design:

1. **Counting cannot be retrieved.** An aggregation question spans 8–43
   documents; top-5 chunk retrieval sees a handful and counts those. RAG and
   GraphRAG score **0–1 out of 21** on those questions, and no prompt or
   reranker fixes it. The LLM-extracted entity graph didn't help either: it
   stored ~4 entities per document with free-text relations, never the fields
   you would filter or count on.
2. **So model the fields as a graph.** `OlympicEvent`–`Games`–`Sport`–`Venue`,
   plus `PREV_GAMES` for "the Games before X", built deterministically from the
   infoboxes with **no LLM calls**. Counting, argmax, venue+date lookup and
   previous-Games become traversals. Aggregation goes 0/21 → 21/21.

**The agent is the front door, not a wrapper.** `src/pipelines/investigator/`
plans each step over five tools (`query_events`, `query_facts`, `link_entities`,
`graph_traverse`, `search_chunks`), states an `evidence_check` in the *same*
structured call so self-evaluation is free, and answers or changes strategy. It
chose the structured tool on all 100 questions and added vector search on 4
where the graph fell short; 96/100 investigations finished in two steps and all
100 stopped because they were satisfied, not because they ran out of budget.

**Planning does not need a generative model.** Every value the planner emits is
already a row in the graph — 47 sports, 20 Games, 316 venues, 475 event names —
so planning is *selection*. `src/pipelines/jev_planner/` uses two TypeSafe
System One requests (three independent Choices in the first, a narrowed event or
venue in the second), reads exact numbers and dates with regex, then runs the
same GSQL and the same templated answer. Same accuracy, a tenth of the latency,
**zero calls to a generation model** — so the demo cannot die on a rate limit
and anyone can reproduce the numbers without a paid key.

**Evidence is checkable, not asserted.** Every evidence item carries its source
document, URL and the exact query that produced it. And the evaluation checks
itself: the LLM judge gave 4–5 to 14 answers that are wrong, mostly fluent
refusals ("the corpus does not contain this"). A System One verification pass
(`src/eval/verify_evidence.py`) flags **14/14** of them, while supporting
176/180 correct controls; all four disagreements fall below the confidence gate
and route to review rather than being asserted.

**Round 2 groundwork.** `src/ingestion/build_temporal_facts.py` extracts 2,316
interval-stamped facts (office terms, championship reigns) with provenance, and
the agent queries them through `query_facts` — "who held this on that date" is a
traversal, not a search. Conflict detection is implemented and tested; the
corpus itself contains **0** genuine contradictions, and we say so rather than
manufacturing some.

## What went wrong, and how we know

A submission that only lists wins is hard to trust, so:

- **A stale artifact nearly shipped.** The hidden answers were generated two
  minutes before a bug fix and carried five factually wrong counts — three of
  them literally the evidence cap, `12`. Caught by re-answering with a second
  pipeline and scoring both against `src/eval/oracle.py`.
- **A "fix" cost 17 points.** Changing which short answer wins dropped exact
  match 99 → 82. The agent could not see what its tools had computed, so it
  recounted a capped evidence list. Found by re-measuring, not by reasoning.
- **A fix that made things worse.** The first repair of a confidence gate made
  one question fail 3/3 instead of 1/3, because a regex only matched year and
  season when adjacent.
- **The LLM judge is not trustworthy** on this task, which is why exact match is
  the headline metric and the judge is a 20% sample.

Each of these is pinned by a check in `tests/test_regressions.py`.

## Reproducing

```bash
pip install -r requirements.txt && cp .env.example .env   # fill TG_* and a key
python -m src.ingestion.create_schema && python -m src.ingestion.create_queries
python -m src.ingestion.load_documents_chunks
python -m src.ingestion.build_event_graph        # no LLM calls
python -m src.ingestion.build_temporal_facts     # no LLM calls

python -m tests.test_regressions                 # no LLM, no database
python -m src.eval.oracle data/results/hidden_answers_jev.jsonl
python -m src.eval.run_benchmark --pipelines jev_planner --judge-rate 0
streamlit run src/dashboard/app.py
```

## Limits

Global search over community summaries is not implemented; `Community` exists in
the schema, unused. The pipelines do not yet filter by the temporal layer's
validity intervals at answer time. pub-099 is unanswerable as posed — two events
share a venue and a date — and every pipeline misses it.
