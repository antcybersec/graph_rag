# Round 2 submission: the six deliverables

Everything below is produced by the two commands in the README. Nothing is
assembled by hand.

| # | Deliverable | Where it comes from |
|---|---|---|
| 1 | `answers_public.jsonl` | `./run.sh questions_public.jsonl data/results/r2/answers_public.jsonl` |
| 2 | `answers_hidden.jsonl` | `./run.sh questions_hidden.jsonl data/results/r2/answers_hidden.jsonl` |
| 3 | `metrics.json` | `python -m src.r2.metrics` — computed from (1) and (2) alone |
| 4 | A git tag | `git tag -a r2-submission -m "..."` on the commit that produced (1) and (2) |
| 5 | TigerGraph query log | written beside each answer file as `answers_*.jsonl.tgquerylog.jsonl` |
| 6 | Vertex and edge counts by type | `python -m src.r2.counts` → `data/results/r2/graph_counts.json` |

## Reproducing from nothing

```bash
cp .env.example .env            # fill in GOOGLE_API_KEY; the rest has working defaults
./build.sh corpus.jsonl         # Docker, schema, 32 queries, 3,530 documents, 16,377 vectors
                                # ends by running tests/test_r2.py and src.r2.smoke
./run.sh questions_public.jsonl data/results/r2/answers_public.jsonl
./run.sh questions_hidden.jsonl data/results/r2/answers_hidden.jsonl
python -m src.r2.score questions_public.jsonl data/results/r2/answers_public.jsonl
python -m src.r2.metrics
python -m src.r2.audit data/results/r2/answers_public.jsonl
python -m src.r2.audit data/results/r2/answers_hidden.jsonl
python -m src.r2.check questions_public.jsonl data/results/r2/answers_public.jsonl
python -m src.r2.check questions_hidden.jsonl data/results/r2/answers_hidden.jsonl
```

Or `./finish.sh`, which does all of it unattended and waits for the quota
first.

Three of those need neither a model nor a key, which is why they are worth
running often: `tests/test_r2.py` (81 pure-function checks), `src.r2.smoke`
(all ten route cases against the live graph) and `src.r2.audit` (each answer
held against the arithmetic its own record carries). The audit is the only
check the hidden set gets, because it ships with no answers.

`check` is the gate: it verifies every trace against the brief's constraints,
that each question has exactly one real answer, that every TigerGraph step has
a matching query-log line, and that each log line falls inside its own
question's `started_at`/`finished_at` window. It exits non-zero if anything
fails, so it can be run before submitting and believed.

## Free-tier quota, and how a run survives it

The free Gemini tier allows **500 generate_content requests per day per
model**. Per *model* matters: the answering model is `gemini-3.1-flash-lite`
and the judge is configured as `gemini-3.1-flash-lite-preview`. Two
corrections, both measured on 24 Nov. First, that judge setting was **not in
effect** for the committed score: `score.py` read `JUDGE_MODEL` before any
`load_dotenv()` had run, got nothing, and silently fell back to the answering
model, so the 91.6% is self-graded. Fixed, and the scorer now refuses to be
quiet about it. Second, a different model id is **not** known to be a
different quota pool: Google's rate-limit documentation says limits are
"applied per project, not per API key" and never says two ids get separate
daily allowances. Measured, ~371 answering calls in one Pacific day were
enough to exhaust it, so judge calls are budgeted against the same pool. A 50-question run makes roughly 120, so one day covers both runs
with room to spare — but it does not cover both runs *plus* several rehearsals,
and when it runs out it runs out mid-run. The 429 body says exactly when it
comes back (`retryDelay`, in seconds), which is better than guessing.

`run.py` handles this rather than thrashing against it:

- A quota error **stops the run immediately** instead of writing a placeholder
  for every remaining question. (Before this, hitting the wall at question 26
  spent the retry budget producing 25 empty records.)
- `--resume` keeps the answers already in the output file and answers only
  what is missing, so a reset day is not spent redoing work. A record counts
  as done only if it has a trace, so a failed question is always retried.
  **It is for recovering a run, not for producing a submission**: the two
  submitted files are each written by one process on one code version, because
  a judge re-running the command on a file assembled across two versions gets
  answers that do not match it.
- A resumed run **appends** to the same query log, because the lines for the
  answers it is keeping are already in it.

```bash
python -m src.r2.run questions_public.jsonl data/results/r2/answers_public.jsonl --resume
```

`build.sh` takes roughly 25 minutes from cold, almost all of it embedding
16,377 chunks on-device. It is safe to re-run: every GSQL object is
`CREATE OR REPLACE` and every row upserts by primary id.

## Before submitting, check

- [ ] `trace.verify()` reported no problems on either run (`run.py` prints them
      to stderr and exits noisily)
- [ ] every line of both answer files has a TigerGraph step
- [ ] the query log's timestamps bracket each run. Each answer record also
      carries `started_at` / `finished_at`, so every log line for a question
      can be checked to fall inside that question's own window
- [ ] `metrics.json` numbers match the answer files (they are derived from
      them, so this is a check that the right files were read)
- [ ] `.env` is not committed; `.env.example` lists every key
- [ ] the git tag points at the commit that produced both answer files

## Before submitting, also

- [ ] `python -m src.r2.audit` read on both files, and every line of its
      worklist either explained or fixed
- [ ] `python -m src.r2.smoke` green on all ten routes
- [ ] `python -m tests.test_r2` green
- [ ] `python -m src.r2.repro questions_public.jsonl \
      data/results/r2/answers_public.jsonl --sample 2` returns no `DIFFERENT`.
      This is the only check that tests the claim repeatability is graded on,
      and the only one that costs model calls (~3 requests a question), so it
      is run on a sample rather than the file

## For the judging call

- `python -m src.r2.ask "<their question>"` answers one question and prints the
  route, every step, the installed query behind each one and the token counts.
  It takes the question as an argument, so there is no id and nothing to look
  up — the same code path answers an unseen question and a submitted one.
- `./offline.sh "<their question>"` does the same with **no API key and no
  network**: planning, selection and composition on a local Ollama model, the
  local graph and the local embedder. 123s a question on an M4/16GB against 16s
  on Gemini. It is the answer to a quota wall on the day, and it is worth
  showing for its own sake.
- `python -m src.r2.repro <questions> <answers> --qid <id>` is the answer to
  "show me that this answer reproduces". It re-answers the question and prints
  `MATCH` / `CLOSE` / `DIFFERENT` against the submitted record, comparing the
  status, every figure in order, the citations and the query sequence — not the
  wording, which is model-written and moves without anything being wrong.
- `./ui.sh` opens the demo page. It reads the graph live, so it cannot show a
  stale number.
- The container keeps its data in the named volume `tg_r2_data`, so the graph
  survives a restart and the demo does not depend on a rebuild.
- **Counts read while TigerGraph is warming up are stale.** The first build of
  this graph reported `EventSeries 0` next to 296 of its own `INSTANCE_OF`
  edges. If a number looks impossible, wait and read it again.
- **Keep a day's quota in reserve for the call.** A handful of live questions
  costs ~40 requests, which is nothing against 500 — but a full batch run on
  the morning of the call is not, so do not run one.
- Judges will ask new questions. The honest answer to "how do you know it
  isn't tuned to these 50?" is in the code: no component in the answering path
  reads the wording of a question, and the two places in the repository that
  do pattern-match on question text are Round 1 code and a development probe,
  both named in the README and both carrying a header that says so.
