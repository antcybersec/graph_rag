# When does an AI agent actually earn its cost? I measured it on 100 questions

*Built for the TigerGraph Agentic GraphRAG Hackathon. Repo and dashboard linked at the end.*

RAG retrieves text. GraphRAG adds structure. Agentic GraphRAG lets a model plan
its own investigation. Everyone assumes the third one is better — but by how
much, and on which questions? That's the question this hackathon asked, so I
built six pipelines over one corpus and measured them against the same 100
questions, with the same generation model throughout.

The short version: **exact match went from 67% to 99%, a 32-point jump.** But
the interesting part isn't that number. It's that an agent *by itself* only got
me 3 of those 32 points.

## The setup

2,951 Wikipedia articles in TigerGraph Savanna. 100 evaluation questions across
five types: lookup, temporal, multi-hop, superlative and aggregation. Every
pipeline uses `gemini-3.1-flash-lite` — a cheap model, deliberately — with local
BGE embeddings and TigerGraph's native vector index. No pipeline gets a better
model than any other.

Scoring is exact match against the gold answer, computed with no model in the
loop. I also ran an LLM judge, and I'll come back to why I stopped trusting it.

## The failure that taught me the most

Plain RAG scored 67%. GraphRAG — entity linking plus one-hop traversal — also
scored 67%. Identical. That was my first surprise.

Breaking it down by question type showed why. On aggregation questions —
*"how many cycling events had more than 30 competitors?"* — RAG got **1 out of
21**. GraphRAG got **0**.

The reason is structural, and no amount of prompt engineering touches it.
Answering that question requires *every* matching document; for one question,
43 of them. Retrieval fetches the top five and counts those. The model then
confidently reports a number that is simply the size of what it was shown.

My LLM-extracted entity graph didn't help either. It had 12,000 entities with
free-text relationship labels — but "competitors: 43" was never a property you
could filter or count on. I had modelled the prose and not the facts.

## What actually fixed it

Every one of those articles opens with an infobox: event, games, venue, date,
competitors, nations, gold medallist. So I built a second layer in the graph
directly from those boxes — `OlympicEvent` linked to its `Games`, `Sport` and
`Venue`, plus a `PREV_GAMES` edge so "the Olympics before 2016" is one hop.

That ingestion makes **zero LLM calls**. It's a parser. And it turns counting
from a retrieval problem into a graph traversal: aggregation went from 0/21 to
**21/21**.

## The agent, and the ablation that matters

My agent plans each step and chooses between five tools: a structured graph
query, time-scoped fact lookup, entity linking, one-hop traversal, and vector
search over chunks. It states what the evidence does and doesn't establish *in
the same structured call* as the routing decision, so self-evaluation costs no
extra call. It answers when satisfied, or changes strategy.

Here's the comparison that I think is the real result:

| Pipeline | Exact match | vs RAG |
|---|---|---|
| RAG | 67% | — |
| GraphRAG | 67% | +0 |
| Agent over text/entity tools only | 70% | **+3** |
| Same agent + structured graph tools | 99% | **+32** |

The agentic architecture alone bought 3 points. The agent *with the right
retrieval surface* bought 32. If I had only built the agent, I'd have concluded
agentic RAG barely helps. If I had only built the graph layer, I'd have missed
that the agent is what handles everything the layer doesn't model — ask it who
directed a film and it switches to vector search on its own.

## The result I didn't expect

If the agent picks the structured query on 100 out of 100 questions, does it
need to *reason* at all? I tested it: everything the planner decides already
exists as a row in the graph — 47 sports, 20 Games, 316 venues, 475 event names.
So planning isn't generation, it's **selection**.

I replaced the generative planner with two typed selection calls (TypeSafe's
System One), kept the same GSQL and the same templated answer, and got:

**Same 99%. 1.7 seconds per question instead of 13. Zero generation-model calls.**

That last number matters more than the speed. My free tier is 500 calls a day —
one benchmark run — and every measurement I took that week died on that wall,
three times in one day. A path with no generation calls can't hit a rate limit,
which means a live demo can't fail halfway through.

So the honest answer to the hackathon's question: **agents earn their cost on
open-ended questions, and they're overkill where a typed query settles it.** The
agent still ships, because you don't know in advance which kind of question is
coming.

## Things that went wrong

A submission that only lists wins isn't worth much, so:

**My LLM judge was unreliable.** It gave 4 or 5 out of 5 to **14 answers that
are wrong** — mostly fluent refusals like "the corpus does not contain this."
That's why exact match is my headline metric. I built a verification pass that
asks a narrower question — does this evidence support this claim? — and it
flagged all 14.

**A "fix" of mine cost 17 points.** I changed which answer field wins, and exact
match dropped from 99 to 82. The agent couldn't see what its tools had computed,
so it recounted a truncated evidence list and answered `12` — the size of the
cap — to seven counting questions.

**One fix made things worse before better.** My first repair made a question
fail three times out of three instead of one. A regex only matched year and
season when adjacent, so "the Summer Olympics held immediately before 2016"
produced no season at all.

**A stale artifact nearly shipped.** My hidden-set answers were generated two
minutes before a bug fix. Five counts were wrong. I only caught it by answering
the same questions with a second pipeline and scoring both against a
deterministic oracle.

Every one of those is now pinned by a regression test, because each was
invisible in normal use — the system kept answering, it just answered wrong.

## Numbers

| | RAG | GraphRAG | Agentic (ablation) | Agentic (full) | Selection planner |
|---|---|---|---|---|---|
| Exact match | 67% | 67% | 70% | **99%** | 99% |
| Aggregation | 1/21 | 0/21 | 3/21 | **21/21** | 21/21 |
| Tokens/question | 3,586 | 3,952 | 6,065 | 3,412 | 2,267 |
| Seconds | 6.8 | 10.2 | 20.2 | 13.2 | **1.7** |

The agent reaches 99% on **fewer tokens than plain RAG**. A cheap model with the
right architecture beats an expensive one with a lazy pipeline.

---

**Repo:** https://github.com/antcybersec/graph_rag
**Live dashboard:** https://graphrag-c.streamlit.app/
Built on TigerGraph Savanna with GSQL and its native vector index.
