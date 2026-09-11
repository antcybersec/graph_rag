# Project history / status log

Running log of what's been done and why, so a new session (human or Claude)
can pick up without re-deriving everything from git log + logs. Newest entry
on top. Keep entries short — link to logs/commits for detail, don't duplicate
them here.

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
