# Demo video — script and shot list

**Want a script to read straight out? Use [`demo-narration.md`](demo-narration.md).**
This file is the longer shot list, including the terminal-based alternative.

Target 3:00. Everything below is a real command in this repo; nothing is mocked.

## Recording entirely from the dashboard

The dashboard's **Live investigation** section runs the real pipelines, so the
whole video can be one browser window: the comparison and per-question traces
above, then a live question at the bottom with the call counter on screen.

Run it locally for this — `streamlit run src/dashboard/app.py` — because the
public deploy has no TigerGraph or API credentials and shows the section
disabled. Use the pipeline radio to show both halves of the argument: the
**Investigator** picking tools, then the **Jev planner** answering with
**0 generation-model calls**.

The terminal shots below are an alternative, not a requirement.

## Before recording

- **Hide secrets.** Never show `.env` or `~/.claude`. If you open the repo in an
  editor, close those files first. `git status` is safe; `cat .env` is not.
- **Warm the services.** Run the demo once before recording: the first call
  builds the candidate catalog from the graph, and TigerGraph's first query of a
  session is slow.
- **Free some memory.** Long runs on this machine have been killed by macOS
  memory pressure; close Brave and any IDE before a live benchmark.
- **Terminal:** 16pt+, dark theme, window ~100 columns so text stays readable.

---

## 0:00–0:20 · The question

> "RAG retrieves text. GraphRAG adds structure. The question this hackathon
> asks is: when do you actually need an agent? We benchmarked six pipelines on
> the same corpus and the same questions, and got a clear answer."

**Shot:** the results table in the dashboard, or `docs/submission.md` open at
the table.

## 0:20–0:50 · Where simple retrieval breaks

> "One question type breaks every retrieval-based pipeline. 'How many cycling
> events had more than 30 competitors' needs *every* matching document — here,
> 18 of them. Top-5 chunk retrieval sees five and counts those. RAG gets 1 out
> of 21 of these. GraphRAG gets 0."

**Shot:** dashboard, "Quality by question type", pointing at the aggregation
column.

## 0:50–1:20 · Model the fields, not the prose

> "So we built a second graph layer straight from the article infoboxes — events
> linked to their Games, sport and venue, plus a link from each Games to the one
> before. No LLM in that ingestion at all. Counting becomes a traversal.
> Aggregation goes from 0 out of 21 to 21 out of 21."

**Shot:** `docs/architecture-diagram.md` — the system diagram.

## 1:20–2:10 · The agent, and its evidence

```bash
python -m src.pipelines.investigator.pipeline
```

> "The agent plans each step and picks between five tools: a structured graph
> query, time-scoped facts, entity linking, traversal, and vector search. It
> judges its own evidence in the same call, so self-evaluation is free. Watch
> it choose the structured tool for an Olympic question — and for a question
> about a film, which the event graph doesn't model, it switches to vector
> search on its own."

**Shot:** terminal showing both questions, the trace lines, and the `tools`
list differing between them. Then the dashboard drill-down for one question,
showing evidence with its **source URL and the exact query that produced it**.

## 2:10–2:40 · The finding: when the agent is overkill

```bash
python scripts/demo.py
```

> "Here's the part that answers the hackathon's question. For these templated
> factual questions, the agent is overkill. Everything the planner emits already
> exists as a row in the graph, so planning is *selection*, not generation. Two
> typed selections, then GSQL — same 99 out of 100, ten times faster, and zero
> calls to a generation model. Watch the counter."

**Shot:** the demo output, ending on the summary line:
`5 questions · 7 System One calls · 0 generation-model calls · ~2s average`.
(Measured cold: 3.4s for a single question, ~2s each across five once the
catalog is warm. The zero is the number that matters on camera.)

> "That matters beyond speed. The free tier here is 500 calls a day — one
> benchmark run — and this path can't hit a rate limit. The agent is still
> required for anything the graph doesn't model. That's the answer: agents earn
> their cost on open-ended questions, not on ones a typed query can settle."

## 2:40–3:00 · Verification and close

```bash
python -m tests.test_regressions
python -m src.eval.oracle data/results/hidden_answers_jev.jsonl
```

> "Both run with no API key and no database. We also check the checker: our LLM
> judge scored 4 or 5 on fourteen answers that are wrong, and a verification
> pass catches all fourteen. Everything in the repo — every number — is
> reproducible."

**Shot:** the two commands passing, then the repo URL on screen.

---

## If you only have 60 seconds

Cut to: the aggregation failure (0:20), the agent choosing different tools
(1:20), and the zero-generation-call demo (2:10). Those three beats carry the
whole argument.
