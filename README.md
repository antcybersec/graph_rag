# Agentic GraphRAG over TigerGraph — Round 2

A question-answering system over a fixed corpus of 3,530 documents that
reasons across many of them at once, weighs sources that contradict each
other, distinguishes what was true at one moment from what is true now, and
**says so when the corpus does not contain an answer**.

Round 1's head-to-head study of RAG vs GraphRAG vs agentic retrieval over the
same corpus is in [`docs/round1-results.md`](docs/round1-results.md). This
file describes the Round 2 system.

```bash
./build.sh corpus.jsonl                               # graph up, schema, queries, corpus
./run.sh questions_public.jsonl answers_public.jsonl  # answer a question file
./run.sh questions_hidden.jsonl answers_hidden.jsonl
./ui.sh                                               # the demo page
```

`build.sh` is safe to re-run: every GSQL object is `CREATE OR REPLACE` and
every row is upserted by primary id, so a second run converges on the same
graph rather than failing or duplicating.

---

## How documents become graph data

Everything structural is parsed **deterministically — no model, no extraction
errors.** The infobox at the head of a Wikipedia-style article is a table in
disguise, so it is read as one. 3,530 documents route to exactly one record
type each, so nothing is silently dropped: 2,750 Olympic events, 546 films,
152 people, 39 LA 2028 source records, 43 documents with no infobox at all.

Five layers, each earning its place from a question family that cannot be
reached without it:

| Layer | Vertices | Why it exists |
|---|---|---|
| **text** | `Document`, `Chunk` (+768-dim vector) | Some answers exist only in prose: why a podium changed, why a bout was suspended. |
| **events** | `OlympicEvent`, `Games`, `Sport`, `Venue`, `EventSeries` | `EventSeries` makes "this contest across the years" **one hop** instead of a scan of 2,750 vertices. |
| **medals** | `Athlete`, `Nation` | Two edges, not one — see below. |
| **reference** | `Film`, `Person`, `Office` | Films and political successions, in the corpus since Round 1. |
| **evidence** | `SourceRecord`, `Claim` | The layer Round 2 is actually about. |

**Two medal edges, not one.** A medal line can name a whole squad in a single
string. The nation is always known; the individual is not. So `MEDAL_FOR`
carries every line by nation and `WON` carries only the lines naming one
athlete — which is also what the published answers count when they ask for an
athlete's medals.

**`DISCRIMINATOR` is load-bearing.** TigerGraph keeps one edge per
(from, to, type) unless a discriminator distinguishes them, and this corpus has
357 events awarding two bronzes, frequently to the same nation. Without
`DISCRIMINATOR(medal, recipient)` on `MEDAL_FOR` those two collapse into one
edge and every medal table is quietly short. The same applies to repeated
non-consecutive terms in `HELD_OFFICE`.

**Source records become claims.** Each of the 39 LA 2028 documents is read
once at build time into claims: who asserted what, about which fact, from
when, whether the document is stating it or merely repeating someone else, and
how much weight its source carries. Extraction is cached by document and
prompt version, so a rebuild is reproducible and costs nothing per question.
**Losing claims are stored on purpose** — an answer has to be able to say *why*
the newest document was wrong, not merely omit it.

A claim is grounded to an event vertex where it can be. That matters more than
it sounds: a jury writes "men's individual sabre" and an aggregator writes
"men's sabre", and grouped by their own strings those are two different facts
with no disagreement between them. Resolving both to the same event is what
makes the contradiction visible — and it is why this is a graph problem.

## The agent's tools

**31 installed GSQL queries.** The agent has no other route to the data, which
is why every database call appears in the trace by name with its parameters and
row count. Grouped by what they are for:

- **Catalogues** — `list_sports`, `list_games`, `list_sources`,
  `series_by_sport`. These are what the model is allowed to choose from.
- **Coverage** — `coverage_check`. The honesty primitive.
- **Events** — `event_lookup`, `events_in_sport_games`, `series_trend`,
  `series_medals`, `neighbouring_edition`.
- **Medals** — `medals_by_nation`, `nation_medals_detail`,
  `medals_by_athlete_at_games`, `athlete_career`, `team_medals_for_name`,
  `repeat_champions`.
- **Entity resolution** — `athlete_search`, `person_search`, `film_search`.
- **Reference** — `films_by_director`, `office_chain`.
- **Evidence** — `claims_for_event`, `claims_by_subject`, `claims_search`,
  `claims_by_slot`, `claims_by_source`, `sources_for_event`.
- **Retrieval** — `vector_search_chunks`, `chunks_containing`,
  `chunks_of_document`, `doc_for_chunk`.

## How it decides what to do next

Five stages. **The model decides what to look up; the graph decides what
exists.**

1. **Plan** — one model call reads the question and returns a route and the
   entities it mentions. It is handed the sports and Games catalogues from the
   graph and may only answer with keys from them; anything else is discarded
   before it reaches a query.
2. **Resolve** — the names in the question become graph keys, but the
   candidates come from the database and the model picks among them. It may
   answer "none of these", which is how a question about something the corpus
   does not cover stops here instead of being forced onto the nearest match.
3. **Check coverage** — the graph is asked whether a document matching what was
   pinned down exists at all. Zero matches is evidence. When a year is missing,
   two checks separate the two reasons: the Games is absent from the corpus, or
   the Games is present and this contest was not on its programme.
4. **Gather** — the route runs its queries. For a disagreement, the claims
   found by name are expanded through `claims_by_slot` to **every** claim
   competing over the same fact before any of them is weighed, because the
   document that overturns a claim may name only the event.
5. **Answer** — one model call writes the answer from the gathered evidence
   alone and picks its citations from the documents actually retrieved.

**Adaptive, with an explicit stopping criterion.** If a route retrieves no
document at all, the agent widens to full-text search before composing. The
composer may also return `enough=false` and buy exactly **one** further round
of retrieval. One, not an open loop: an agent that can always look again never
has to report a gap.

**The honesty gate is deterministic and the model does not get to overrule
it.** Citations not present in the retrieved evidence are dropped. An
`answered` with nothing left behind it becomes `not_in_corpus` — because an
answer no retrieved document supports is the invented answer the brief scores
zero.

### Deciding what to trust

Documents in this corpus disagree, and none of them says so about itself. The
rule, in order:

1. **Authority** — a signed or certified decision outranks an official record,
   which outranks reporting, which outranks an aggregator, which outranks
   somebody's recollection. Inferred from how a document describes its own
   source, using generic signals (*signed*, *jury*, *blog*) rather than a list
   of this corpus's 39 sources, so it generalises to documents we have not
   seen.
2. **What the document says about itself** — a final result beats a draft from
   the same office. The *earliest* such phrase in the document wins, because a
   record states its own standing up front and describes what it replaces
   afterwards.
3. **Effective time** — when the fact took effect.

**Publication time never decides.** Two public questions exist to punish
systems that take the newest document, and in both the newest document is
wrong: a blog that republished a provisional scoreboard, and a directory entry
that recycled a superseded venue allocation.

Three further distinctions, each of which cost a wrong answer while they were
missing:

- **Repeating a claim is not making one.** An operations log that lists two
  reported stoppage durations *in order to say neither could be verified* is
  not asserting either of them.
- **An official non-confirmation outranks a confident witness.** When the only
  record states a value could not be established, a volunteer's notebook does
  not become the answer.
- **Different populations are not a contradiction.** 42 registered entries and
  40 starters are both true.

And one that is a question in its own right: **absence of a record is not a
negative fact.**

## No question-specific rules

Rule 4 forbids regexes on question wording, use of question ids, and answers
built into the code. Here that is structural rather than promised:

- **No code in the answering path reads the wording of a question.** The only
  component that interprets a question is the model.
- The only values the model may emit are keys the graph has just handed it.
- Deterministic string work happens exclusively on **entity names the model
  extracted** and on **graph keys** — never on the question.
- No question id reaches the agent; `run.py` passes only the text.

That is why the same code answers questions it has never seen: it has never
seen any question.

**Two places in this repository do pattern-match on question text, and neither
is in the Round 2 answering path.** Pointing them out is easier than hoping
nobody greps for `re.search`:

- `src/pipelines/` is the **Round 1** study (RAG, GraphRAG, and three agentic
  designs compared head to head). Round 1 permitted it;
  `src/pipelines/jev_planner` and `src/pipelines/event_graph` use it. Round 2
  runs none of this code — `run.sh` imports `src.r2.run`, which imports only
  `src/r2/`. Both files carry a header saying so.
- `scripts/r2_layer_check.py` hard-codes question parameters deliberately. It
  is a development probe that measured whether the parsed records could reach
  the published answers at all, before any graph or agent existed. It is never
  imported by `run.sh` and says so in its first line.

Both are kept because deleting them would erase a reproducible Round 1 result,
not because anything in Round 2 needs them.

## Reproducibility

- **The one model call in the build is cached and committed.** Claim
  extraction over the 39 source records is the only part of ingestion that
  calls a model; `data/results/r2_claims_cache.json` is in the repository, so
  a rebuild produces byte-identical claims rather than 39 fresh extractions
  that might differ. Delete it to re-extract.
- **Locked dependencies** — `requirements.lock.txt` is `pip freeze` from the
  environment that produced both answer files. `requirements.txt` lists the
  direct dependencies.
- **`.env.example`** lists every key. Defaults reach nothing paid: TigerGraph
  Community Edition in Docker, embeddings on-device.
- **Traces are written as the code runs.** `Trace.run_query` and
  `Trace.generate` are the only routes to the database and to a model, so the
  trace cannot be missing a step and the token totals cannot drift from the
  per-step numbers. `trace.verify()` re-checks every record against the
  brief's constraints before a run is considered finished.
- **The query log** is written beside the answers
  (`answers_*.jsonl.tgquerylog.jsonl`), one line per database call with a
  timestamp matching its trace step.
- **metrics.json** is computed from the two answer files and nothing else, so
  the judges' recomputation cannot disagree with it.

## A note on the latency figures

`latency_ms` is wall clock from receiving the question to returning the
answer, as the brief defines it. On a free-tier API key that includes
**rate-limit pacing**: calls to the model are spaced by
`LLM_MIN_INTERVAL_SEC` (4.5s) to stay under the per-minute cap, and that sleep
happens inside the call. So roughly `4.5s x (llm_calls_per_question - 1)` of
each question's latency is waiting, not work — the arithmetic is auditable
from `llm_calls_per_question_avg` in `metrics.json`. On a paid key the
variable goes to 0 and the figure drops accordingly.

The TigerGraph side is not the cost: the installed queries return in 100-300ms
each, including the substring scan over all 16,377 chunks.

## Known risks

A ranked audit of what could get this rejected or cost it the top place —
what each would cost, the evidence, and what was done — is in
[`docs/round2-risks.md`](docs/round2-risks.md). Four of the ten are closed,
three wait on the next measured run, three are process rather than code.

## Layout

```
src/r2/
  parse.py      corpus -> records, deterministically
  schema.py     14 vertex types, 16 edge types, 3 discriminators
  queries.py    the 31 installed queries: the agent's entire tool surface
  load.py       schema, structure, documents, chunks, evidence
  claims.py     source records -> claims; claims -> one resolved answer
  authority.py  how much weight a source carries, read off its own description
  agent.py      plan, resolve, check coverage, gather, answer
  trace.py      the per-question trace and the only route to the DB or a model
  run.py        a question file -> an answer file
  score.py      our answers vs the published key (never used while answering)
  metrics.py    metrics.json, from the answer files alone
  ui.py         the demo page
  conn.py       Community Edition or Savanna, from identical code
build.sh run.sh ui.sh
```
