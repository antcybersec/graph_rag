# Project history / status log

Running log of what's been done and why, so a new session (human or Claude)
can pick up without re-deriving everything from git log + logs. Newest entry
on top. Keep entries short — link to logs/commits for detail, don't duplicate
them here.

---

## 2026-09-17 — Found the actual rubric; built the Investigator agent + temporal layer

**The rubric changes the priorities.** Found the hackathon listing (TigerGraph
"Agentic GraphRAG Hackathon", Unstop): investigation accuracy **30%**, evidence
quality + explainability 15%, agentic effectiveness + efficiency 15%, agentic
design + code quality 15%, innovation 15%, final presentation 10%. Round 1 due
**2026-09-24**; top 15 go to Round 2 (due Oct 1, live demo), whose stated
challenge is "reasoning over evolving, conflicting, and uncertain facts".
So 30% of the score is explicitly about *agentic* design — and `event_graph`,
our 99/100 pipeline, is a fixed route, not an agent. That is the gap this entry closes.

**Built `src/pipelines/investigator/`** — one agent, four tools (`query_events`
= structured GSQL over the event layer, `link_entities`, `graph_traverse`,
`search_chunks`), with a required `evidence_check` self-evaluation folded into
the same structured call (no extra LLM call), and provenance on every evidence
item (doc id, title, source_url, originating query + params).
Result on the 100 public questions: **99/100 exact match, 2.06 steps, 2.11 LLM
calls, 2,695 tokens, 16.5s, doc P/R 0.97/0.85, judge 4.33**, all 100 stopping on
`answer_found`. Tool mix: `query_events` on 100, `search_chunks` on 3. Off-domain
check ("Who directed Jab We Met?") routes to `search_chunks` and answers correctly.

**Dashboard** now has 5 pipelines (validated 5-colour palette, light + dark) and
an investigation drill-down: every step with its self-check and timing, every
evidence item with source link and the query that produced it.

**Temporal layer** (`src/ingestion/build_temporal_facts.py`): 2,316
interval-stamped facts (129 office terms + 2,187 championship reigns) with
provenance, plus `facts_as_of` / `fact_timeline` GSQL.
**Honest finding:** with correct slot/value semantics the corpus contains
**0 cross-source disagreements and 0 rival claims** — 227 "concurrent roles" are
not conflicts. Two earlier versions of the detector reported fake conflicts
(repeat champions; one athlete winning two events); both were wrong and were
fixed. Conflict *handling* is implemented, but demoing it needs a labelled
fixture, not a claim that the corpus disagrees with itself.

**Robustness proven, not assumed:** 10 questions rewritten away from the eval
templates (host cities — "Sydney 2000", "Calgary 1988", "Seoul 1988", "Rio 2016"
— plus different vocabulary) score **10/10** (`data/results/paraphrase_test.log`).
One needed 3 steps: `query_events` fell short, the agent switched to
`search_chunks` and recovered. Host-city → Games-year mapping is done by the
planner; no regex or corpus field contains it. Use this against any
"it only works on the five templates" objection.

**Independent review + fixes (same day).** A code-reviewer pass reproduced every
headline number independently (2,316 facts = 129 + 2,187, zero fact_id
collisions, EM 99/100, 2.06 steps, 2.11 calls, per-qtype breakdown, conflicts
0/0/227) and found 2 HIGH bugs, both latent but aimed straight at the hidden set:
- `tool_short_answer` persisted across strategy changes, and `is_correct`
  short-circuits on `short_answer` — so a rejected structured result could score
  a correct prose answer **wrong**. Now cleared whenever a later tool adds evidence.
- 3 of 5 tool dispatches were unguarded: one malformed plan raised out of
  `answer_question` and lost the question as an error row. All dispatches now
  share one guard.
Also fixed: `entity:`-prefixed ids were filtered against bare ids (so traversal
silently fanned out to every known entity); silent `graph_traverse` failures;
`fact_query` missing from the action history; the forced-stop synthesis was
unrecorded and logged 0 context tokens; `query_facts` with no date matched only
still-open facts; dashboard printed `short answer: nan`; README claimed 2.2x
fewer tokens than RAG when the true figure is 1.3x (2.2x is vs the agentic baseline).

**Temporal interval bug (would have broken the Round 2 story).** Reigns started
31 December of the Games year, so for ~10 months after each Games the "reigning
champion" was the *previous* one. Reigns now start on the event's own date.
Two follow-on data bugs surfaced and were fixed: dates are validated per month
(`31 February` no longer becomes 20120231), and a parsed year more than one year
from the Games year is rejected as a typo — which keeps Tokyo 2020's legitimate
2021 dates (45 event articles) while dropping a badminton infobox that reads
"2012" for a 2008 event. That single typo had manufactured a phantom "rival
claim". Conflicts are now **0 cross-source, 0 rival claims**, 274 concurrent
holdings, 0 non-positive intervals.

**Two idempotency bugs in the ingestion scripts (found by re-running them).**
Both would bite anyone who clones the repo and re-runs, which is exactly what
"repository review" in the rubric means:
- `build_temporal_facts.load()` only upserted. `fact_id` embeds `valid_from`, so
  re-running after the interval correction minted *new* ids and left the previous
  generation in place — the graph held 2,603 facts and `facts_as_of` returned the
  same champion twice, once per generation. It now deletes the layer first.
- Both ingestion scripts used `CREATE QUERY`, which fails on re-run
  ("the query name is used by another object") and **leaves the old body
  installed** — an edited query would silently never take effect. Both now use
  `CREATE OR REPLACE QUERY`. `build_event_graph`'s docstring had claimed
  re-running was safe; it wasn't.
A scare along the way: `getVertexCount` read 2,305 right after the reload and I
took it for 11 lost facts. Diffing extracted ids against the graph showed
2,316/2,316 with nothing missing — the count was just stale. Verified the real
way (id set diff), not by trusting the counter.
Also confirmed, closing a reviewer's open question: GSQL `ORDER BY` **does**
survive `PRINT` of a vertex set — `fact_timeline` returns its 8 rows ascending.

**Hidden set answered by the Investigator** (`hidden_answers_investigator.jsonl`):
50/50, 0 errors, 2.04 steps, 2.06 calls, 3,089 tokens, all `answer_found`, every
row carrying a `short_answer` — and **0 disagreements** with the earlier
event_graph answers. Regenerate after the corrected temporal reload so the
submission artifact comes from final code.

**Short-answer precedence was still wrong after the first fix** (found by
regenerating the hidden set and diffing). Tool-first precedence corrupted two
rows while the *prose* answer was correct in both:
- eval-040: `gold_at_venue_date` matched 5 events on a partial date; its
  five-name list overrode the model's correctly disambiguated single name.
- eval-042: an exploratory `top_event_by_competitors` left an event title that
  overrode a correct medallist name.
Exact match reads `short_answer`, so both correct investigations would have
scored wrong. Now `decision.short_answer or tool_short_answer` (commit 2652518).

**STATE AS OF 2026-09-17 EOD — quota exhausted mid-re-measure. Resume tomorrow:**
Gemini hit its daily 429 wall after ~480 calls; a probe confirms it. Nothing
requiring the API can run until the daily reset (observed to reset earlier than
midnight Pacific — probe before assuming).

Result files and what each holds — **none of these is final**:
| File | Generation | Trust |
|---|---|---|
| `benchmark_v3_investigator.jsonl` | final code, public set | only 1 of 100 done (5 × 429) |
| `hidden_answers_investigator_toolfirst.jsonl` | tool-first precedence | eval-040/eval-042 short answers WRONG |
| `hidden_answers_investigator_prefix.jsonl` | pre-review-fix | superseded |
| `benchmark_v2.jsonl` investigator rows | pre-review-fix | the README's 99/100 comes from here |

The corrupted file was moved off the canonical path deliberately:
`run_benchmark` checkpoints per (qid, pipeline), so leaving it there would make
the regeneration skip all 50 questions and silently keep the bad rows.

Resume commands (in this order, one at a time — two Gemini processes double the
request rate and trip 429s):
```bash
python -m src.eval.run_benchmark --pipelines investigator --judge-rate 0.2 \
    --output data/results/benchmark_v3_investigator.jsonl
python -m src.eval.run_benchmark --pipelines investigator \
    --questions data/raw_dataset/questions/eval_hidden.jsonl \
    --output data/results/hidden_answers_investigator.jsonl
```
Then update README's results table from `benchmark_v3_investigator.jsonl` (its
99/100 currently predates every fix in this entry) and re-append investigator
rows to `benchmark_results.jsonl` for the dashboard.

**Still unknown (blocked on the user):** the official rules text, exactly what
Round 1 requires (repo only, or also a DEV article / demo video / hidden-set
answers file), and the required format for hidden-set answers.

---

## 2026-09-16 — Event-graph pipeline built: 99/100 exact match at ~1 LLM call/question

Acted on the root cause below (user chose: LLM planner + graph tools; exact
match on everything plus a 20% judge sample).

**Built:**
- `src/common/infobox.py`: deterministic infobox parser. 2,210 events, 47 sports, 316 venues.
- `src/ingestion/build_event_graph.py`: `OlympicEvent`/`Games`/`Sport`/`Venue`
  vertices; `IN_GAMES`/`IN_SPORT`/`HELD_AT`/`DESCRIBED_BY`/`PREV_GAMES` edges;
  3 GSQL queries. Already loaded and installed on Savanna graph `test`.
- `src/pipelines/event_graph/pipeline.py`: 1 planner call (typed `QueryPlan`),
  then GSQL + Python; falls back to RAG.
- `src/eval/exact_match.py`: strict scorer, with multi-candidate answers
  counted wrong. `run_benchmark.py` gained `event_graph`, `--questions`,
  `--judge-rate`, `exact_match`/`ambiguous`/`route`/`plan` fields; cost
  fields are now pipeline-only.
- Dashboard shows a 4th series and exact match as the headline.

**Results** (`data/results/benchmark_v2.jsonl`, also appended to
`benchmark_results.jsonl` for the dashboard):
- **Public:** 99/100 exact match. All of aggregation 21, superlative 10,
  temporal 22 and lookup 19 correct; multi_hop 27/28.
- **Cost:** 1.02 calls, ~620 tokens, 8.9s per question; doc P/R 0.99/0.90; 0 errors.
- **Only miss:** pub-099, which is ambiguous (two events share venue + date).
- **Hidden set:** 50/50 answered from the graph (`hidden_answers_event_graph.jsonl`);
  eval-032 is ambiguous.
- **Old runs by exact match:** RAG 58, GraphRAG 15, Agentic 59;
  aggregation 0/21 for RAG and GraphRAG, 2/21 for Agentic.

**Review:** an independent code-reviewer pass found 2 HIGH issues, both fixed
and regression-checked before the final run:
- date matching accepted events with no infobox date;
- " | " multi-answer scoring was lenient for event_graph only.

It also found MEDIUM issues, all fixed: gender-crossing fuzzy matches, the
planner dropping "men's"/"women's", invalid Games ids raising, and judge
cost mixed into row cost. An earlier partial run on pre-fix code was stopped
and kept as `benchmark_v2_precodefix_partial.jsonl`; don't use its numbers.

**Caveats / next up:**
- The rag/graphrag/agentic numbers predate the 2026-09-13 reranker. A rerun
  would cost roughly 600+ Gemini calls (about 1.5 days of quota).
- Temporal doc recall is capped at 0.5 by design (see architecture doc).
- The one RAG fallback, pub-049, happened because the planner left `season`
  null ("...Winter Olympics held immediately before 2022"). RAG still answered
  correctly. **Fixed 2026-09-16** (`execute_plan` fills a missing season from
  the question text; pub-049's saved plan now answers from the graph). The
  benchmark_v2 rows were produced before this fix and were not re-run.
- Committed as `238188a` (reranker) and `d67e24c` (event graph). Not pushed.
- **Baseline rerun COMPLETE** (2026-09-16): all 400 (qid, pipeline) pairs in
  `benchmark_v2.jsonl`, every pipeline on the same reranker-era code.
  Exact match: **rag 67, graphrag 67, agentic 70, event_graph 99**.
  Aggregation 1/21, 0/21, 3/21, 21/21. Superlative 4/10, 4/10, 4/10, 10/10.
  Tokens/q 3,586 / 3,952 / 6,065 / 619. Latency 6.8 / 10.2 / 20.2 / 8.9s.
  Judge (same 21-question sample) 3.86 / 3.67 / 4.05 / 5.00.
  The reranker is what lifted GraphRAG from 15 to 67 and cut tokens ~3x; it did
  nothing for aggregation, which is the structural limit the event layer fixes.
  Getting there took 3 attempts: the run was killed twice by macOS low-memory
  pressure (not quota) with the local BGE embedder + reranker resident. It
  checkpoints per pair, so re-running the same command each time resumed it; the
  last 15 pairs finished in a foreground run.
  Known latency inflation: graphrag/agentic pass the `seeds` VERTEX param as a
  plain value, so pyTigerGraph fails over POST and retries with GET on every
  traversal. Left unfixed so latency stays consistent within the run.
- A zero-quota regression test exists as a scratch script only
  (regex plans → `execute_plan` → exact match, 99/100). Worth moving into the repo.

---

## 2026-09-15 — Root cause of poor scores found: it's retrieval design, not quota

Deep-dive after a context loss. **Every eval question (100 public + 50 hidden)
is one of 5 templates over `[Infobox Olympic event]` fields** at the top of
each doc (2,203 of 2,951 docs have one), keyed by titles like
`"<Sport> at the <YYYY> <Summer|Winter> Olympics – <Event>"`:

| qtype | template | infobox field used |
|---|---|---|
| lookup | How many nations competed in `<title>`? | `nations` |
| multi_hop | Who won gold in the event held at `<venue>` on `<date>`? | `venue` + `date`/`dates` → `gold` (strings verbatim, glitches included) |
| temporal | Who won gold in `<event> <sport>` at the Games immediately before `<year>`? | previous Games → `gold` |
| aggregation | How many `<sport>` events at `<games>` had more than N competitors? | count over `competitors` |
| superlative | Which `<sport>` event at `<games>` had the most competitors? | argmax `competitors` |

**Proof:** a scratch parser with no LLM, embeddings, or TigerGraph answered
**99/100 public questions correctly** and gave an answer for all 50 hidden ones. The only miss
(pub-099) is genuinely ambiguous: two events share that venue and date.
Gotchas it hit: tennis docs carry two infoboxes (the Olympic one is second);
normalization must keep `+` (`80 kg` ≠ `+80 kg`); "immediately before"
means the real Olympic calendar (Winter 1992 → 1994).

**Why the current pipelines score badly:**
- aggregation/superlative gold answers span 8–43 docs; top-5 chunks can't
  count them (RAG doc recall 0.42 → accuracy 2.0). No prompt or reranker fixes that.
- The LLM-extracted entity graph (12k entities, ~4/doc, free-text
  relation labels) never stores competitors/venue/date as queryable
  properties, so GraphRAG has nothing to filter or count on.
- The quota walls were a symptom: LLM extraction, the LLM judge, and
  5-step agent loops spent ~10 calls/question on an approach that can't
  answer these questions.
- The LLM judge is weak signal: groundedness is 5 on almost every wrong
  answer. Gold answers are exact strings, so exact match is possible.
- Reranker smoke test (`smoke_rerank.jsonl`, 10 q): roughly halved tokens;
  GraphRAG 2.2→3.4, RAG/agentic flat. That's 10 questions, so treat it as noise.

**Next up:** build a deterministic Event layer in TigerGraph from the
infoboxes (no LLM ingestion) and expose it as graph query tools for the
GraphRAG/agentic pipelines; add exact-match scoring. Direction pending the
user's call.

---

## 2026-09-11 14:xx — Benchmark run interrupted by power outage, provider merry-go-round

**State:** `data/results/benchmark_results.jsonl` — 213/300 (question, pipeline)
pairs done successfully. 87 remaining.

**What happened today:**
1. Overnight/this morning: cycled generation provider Gemini → local (Ollama) →
   OpenRouter → Groq, each hitting its own hard quota wall under this
   project's real call volume (details + exact numbers in `.env` comments
   above `GEN_PROVIDER`/`GEN_MODEL` — read those before re-trying any
   provider, they're kept up to date):
   - **Gemini** (`gemini-3.1-flash-lite`): 500 requests/day cap, hit during
     extraction.
   - **Local Ollama**: no request cap, but real RAM problems on this 16GB
     machine — `ministral-3` (8.9B) OOM-killed other apps; `llama3.2:3b` was
     more stable but still needed Brave/Antigravity closed to leave headroom.
     User experience yesterday: unreliable, "didn't work."
   - **OpenRouter**: free tier is a hard **50 requests/day TOTAL across all
     `:free` models combined** (not per-model) — confirmed via its 429 body.
     Nowhere near enough for a 300-pair run.
   - **Groq** (`openai/gpt-oss-120b`): free tier allows 1000 req/day but only
     **200,000 tokens/day (TPD)** per model — confirmed via 429 body
     ("Used 195888, Requested 4945"). Commit `9d9d8f4` fixed the *per-minute*
     (TPM, 8000) pacing but that was the wrong knob — TPD is a separate,
     much harder wall that pacing can't fix. `gpt-oss` models also burn
     invisible "reasoning" tokens on every call (confirmed via response
     `usage.completion_tokens_details.reasoning_tokens`) and `_generate_groq`
     in `src/common/llm.py` never plumbs through `thinking_budget`/
     `reasoning_effort` the way the Gemini path does — worth capping later.
2. `scripts/resilient_benchmark.sh` was running on Groq, ~9.5h into its last
   attempt (79/88 of that batch's tqdm), when the user's power went out.
   Checked afterward: the **last ~90 rows written before the crash were ALL
   429 errors** — the run had already hit Groq's TPD wall well before the
   outage; the outage didn't cost real progress, the quota wall did.
3. Confirmed live (2026-09-11 ~14:xx) that Gemini's daily quota has reset
   (separate clock from Groq/OpenRouter) — switched `GEN_PROVIDER` back to
   `gemini` to finish the remaining 87 pairs. Gemini has its own 500/day cap
   too, so this may need another provider swap if it runs out again —
   check `.env`'s provider notes first if so, don't rediscover the same walls.

**Resuming:** `scripts/resilient_benchmark.sh` is checkpointed per
`(qid, pipeline)` via `load_done_keys()` in `src/eval/run_benchmark.py` —
safe to just rerun, it skips the 213 done pairs. Error rows are NOT counted
as done, so failed pairs retry automatically.

**Update 2026-09-11 ~14:35 — DONE:** benchmark finished for real. Confirmed
**300/300 unique successful (qid, pipeline) pairs** in
`data/results/benchmark_results.jsonl` (720 total lines in the file, but 420
are accumulated error rows across the whole day's provider-hopping — not
failures, just retried-and-succeeded noise; ignore raw `wc -l` on this file,
always recompute unique non-error `(qid, pipeline)` pairs). Gemini also threw
some `RESOURCE_EXHAUSTED` 429s partway through this last run but had enough
daily headroom to get through all 300 by the end.

**Next up:** the 3-way comparison data is complete. Nothing left to build on
the pipeline/eval side. Remaining work for the hackathon:
1. **`src/dashboard/`** — currently an empty stub. `requirements.txt` already
   has `streamlit` + `plotly` pulled in for it. Build a comparison dashboard
   off `benchmark_results.jsonl`: judge scores (accuracy/completeness/
   groundedness) + doc precision/recall + latency + token cost, per pipeline.
2. No README / write-up yet for the hackathon submission (`docs/`,
   `notebooks/` both empty).

---

## 2026-09-11 (later) — README + docs/architecture.md written; dashboard build in progress

Computed the actual aggregate numbers off the 300 valid rows (dedup'd,
error rows excluded) directly from `benchmark_results.jsonl` — see
`README.md`'s results table. Headline finding worth remembering: **fixed-
sequence GraphRAG (local search) scores WORSE than plain RAG on this eval
set** (accuracy 1.89 vs 3.54, doc_precision 0.12 vs 0.30) — entity-linking
mistakes have no fallback in a fixed pipeline. **Agentic GraphRAG recovers
the gap and edges out both** (accuracy 3.61, groundedness 4.35) because the
orchestrator can fall back to `search_chunks` — at ~3x RAG's latency and
~1.9x its LLM calls. Worst qtype for all three pipelines: `aggregation`
(needs a complete count the retrieved context rarely has in full). Note: 2
of 300 rows (`pub-054`/`pub-063`, both `rag`) have a null judge score —
judge-side parse failure, excluded from README's RAG averages.

Wrote `docs/architecture.md` consolidating the design rationale that was
already scattered as code comments (schema provenance-from-day-one choice,
why `entity_neighbors_1hop` is capped at LIMIT 8, why the agentic action
schema uses `Literal` not `str`, why the orchestrator folds evidence-
sufficiency into its own decision instead of a separate evaluator call,
what's out of scope — GraphRAG global-search/communities, temporal-fact
filtering). Multiple source files already referenced
`docs/architecture.md` in their docstrings before this doc existed — it
now does.

Dashboard build (`src/dashboard/app.py`) was delegated to an executor
subagent in parallel with the above, covering: per-pipeline summary
metrics, judge-score comparison overall + by qtype (the interesting
finding), latency/token cost tradeoff, doc precision/recall, and a
per-question drill-down table. Check its actual output/verification before
trusting it blindly — subagent-written, not yet reviewed by a human in
this session as of this entry.

**Next up:** review the dashboard subagent's diff once it reports back
(`git diff --stat` / run `streamlit run src/dashboard/app.py` yourself),
then this is essentially submission-ready — `notebooks/` is still an empty
stub, decide whether the hackathon rubric actually requires anything there
or whether README + dashboard + architecture doc covers it.

---

## 2026-09-11 (later still) — dashboard visual restyle: TigerGraph UI + Dribbble reference

Dashboard subagent's functional build (`src/dashboard/{app,data}.py`) actually
finished before its stop request landed — kept it rather than discard
verified work (0 exceptions across `AppTest`, real headless-server run, 300/
100-per-pipeline data checks). Did a visual restyle pass on top instead of a
rebuild:
- Browsed tigergraph.com/graphstudio + docs.tigergraph.com's GraphStudio
  screenshots for the actual brand: orange `#F26722` + blue accent, light
  surface, rounded white cards with soft shadow, icon-led left nav.
- Browsed Dribbble ("benchmark comparison dashboard" — Taras Migulko's
  "Benchmarks" shot) for the comparison-dashboard layout language: grouped
  bars with a background band per category cluster, flat tab-style section
  headers, generous whitespace over heavy borders.
- Applied as chrome only, never touching the already-validated 3-pipeline
  data palette: `.streamlit/config.toml` sets `primaryColor = "#F26722"`;
  `CSS_OVERRIDES` in `app.py` adds Inter font, a top orange→blue accent rule,
  rounded/shadowed cards (`st.container(border=True)` + colored top strip per
  pipeline on the KPI tiles), and accent-rule section headers
  (`section_header()`, replacing bare `st.subheader`); `band_categories()`
  adds the faint alternating-band grouping to the qtype×pipeline chart.
- Caught two real bugs in the first pass by actually screenshotting a live
  headless server rather than trusting the CSS by eye: the top accent bar
  targeted `stAppViewContainer`'s first child, which is the *sidebar*, not
  main content (fixed to target `stMainBlockContainer`); and a
  `:root[data-theme="dark"]` selector that Streamlit never sets on anything
  (confirmed via `document.documentElement.attributes` on a live app,
  1.63.0) — removed, `@media (prefers-color-scheme: dark)` is the only real
  dark-mode signal Streamlit gives here.
- Re-verified after fixes: `py_compile` clean, `AppTest` 0 exceptions,
  screenshots of every section (summary cards, judge-score charts with the
  qtype bands, cost scatters, retrieval charts, drill-down) look right in
  light mode. Server killed, no stray streamlit processes left running.

**Next up:** same as before — this is submission-ready pending a decision on
`notebooks/`.

---

## 2026-09-11 (final pass) — first restyle rejected as "plain"; full redesign, dark-only

User called the TigerGraph/Dribbble restyle above "just a plain dashboard" and
pointed at a real bad example from a competing hackathon submission
(sanctiontrace.netlify.app) — dark void, gradient headline text, five neon
unlabeled stat callouts ("+126%", "720x faster than human", no visible
baseline), glassmorphism cards. That's the generic "AI SaaS landing page"
template LLM coding assistants default to when asked for something that
looks impressive fast — optimized for a 10-second glance, not for someone
checking the claim. Wrong reference class for a data-analysis tool anyway
(should look like Linear/Stripe/Datadog, not a product launch page).

Also caught a real design flaw in the first pass while explaining this: it
used TigerGraph orange as BOTH the UI chrome accent (top rule, section
headers) AND GraphRAG's data series color — a genuine ambiguity (does orange
mean "brand" or "GraphRAG" in a given spot?), not just a taste issue.

Redid it properly:
- `.streamlit/config.toml`: dark-only theme (`base = "dark"`), hand-picked
  neutral palette (`#0b0d10` bg / `#14171c` secondary / `#e7e9ec` text), and
  `primaryColor` changed to a neutral slate (`#8b93a5`) instead of TigerGraph
  orange — chrome no longer competes with data color for meaning.
- `app.py` CSS_OVERRIDES rewritten: dropped the gradient top bar and colored
  section-header bars entirely (no gradients, no glow, anywhere). Added IBM
  Plex Mono for numerals only (KPI values, `st.metric`, table figures) —
  tabular numeral treatment borrowed from Linear/Vercel Analytics, used
  exactly once as a system, not decoratively. Cards use a hairline neutral
  border + a fill one step lighter than the page (elevation by contrast, not
  shadow — shadows don't read on dark surfaces). Section headers are now
  plain weight + a hairline rule underneath (a document section break, not a
  colored tab). KPI tiles replaced `st.metric` with custom HTML blocks:
  small muted label, large mono tabular number, muted detail line — real
  size/weight hierarchy instead of three identical boxed widgets.
- Fixed a real pre-existing label bug surfaced by actually looking at the
  rendered page: the "Quality by question type" y-axis read "Mean mean judge
  score (1-5)" (`f"Mean {METRIC_LABELS['judge_mean'].lower()}"` where
  `METRIC_LABELS['judge_mean']` is already `"Mean judge score"`) — only
  reproduces for the default "Mean judge score" dimension choice, so it's
  easy to miss without actually rendering it.
- `THEME["dark"]["surface"]` kept in sync with the new `backgroundColor`
  (`#0b0d10`) since `ring_markers()` strokes scatter points in that exact
  color.
- Re-verified same as before: `py_compile`, `AppTest` (base run + a filter
  change) both 0 exceptions, full visual walkthrough via a real headless
  server + browser screenshots at every section. Server killed after.

**Lesson for next time this comes up:** when asked for "good design" on a
data tool, the default LLM instinct (mine included) reaches for SaaS-
landing-page tropes (gradients, neon stat pills, glassmorphism) because
that's overrepresented in what coding assistants have seen labeled
"impressive." For an analysis/dashboard deliverable the right reference
class is a professional analytics tool (Linear/Stripe/Datadog), not a
marketing page — and the tell that something is the wrong reference class
is usually an unlabeled, baseline-free stat (e.g. "720x faster than human"
with no "than what, doing what" in sight).
