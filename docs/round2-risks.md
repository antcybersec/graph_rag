# What could get us rejected, or cost us first place

Ten risks, ordered by **expected score impact**, not by how alarming they
sound. Each says what it would cost, what the evidence is, and what was done.
Four are closed, three are measurement-dependent and open until the next run,
three are process rather than code.

The weights that make this ranking: hidden-set accuracy **40%**, repeatability
**15%**, live judging **15%**, honesty **10%**, citations **10%**, efficiency
**10%**.

---

## 1. Both answer files incomplete at the deadline — *fatal*

**Cost:** everything. Two of the six deliverables are the answer files.

**Evidence:** the free Gemini tier allows 500 requests/day per model. Three
restarted public runs plus smoke tests plus a scoring pass spent it, and the
last run died at question 26. 25 of 50 public answers exist; the hidden set
has not been run.

**Done:** `run.py` now stops dead on a quota error instead of writing a
placeholder for every remaining question, `--resume` finishes a half-run
without paying for it twice, and `finish.sh` waits for the quota with one
cheap probe every ten minutes, then runs both sets, scores, measures and
validates unattended.

**Open:** run it. ~260 requests for a clean 50 + 50, inside one day's 500.

**Measured 22 Nov:** the daily cap is **per model**, not per key. The answering
model is `gemini-3.1-flash-lite` and the judge is
`gemini-3.1-flash-lite-preview`, which is a different id and therefore a
different 500/day pool -- so scoring the public set does not spend any of the
budget the two runs need. The run plan for the day is ~240 requests on the
answering model (both sets, now that `event_series` costs one model call fewer)
and ~50 on the judge, against 500 each. The exhausted quota returns at **13:36
IST on 23 Nov**, which is not a guess: the 429 body carries
`retryDelay: 56168s`.

**Changed 22 Nov:** the public set is no longer resumed. Both files must come
from one code version or a judge re-running the command gets answers that do
not match the submitted ones, and the evidence block has changed since the
first 25 answers were written. A clean 50 costs ~130 requests and buys a file
that reproduces, which is the better trade. `finish.sh` also waits on a
`.freeze` file now, so a run cannot start while the agent is still being
edited.

## 2. Answers that cannot be reproduced in the judging call — *fatal*

**Cost:** the brief is explicit — "answers that your system cannot reproduce
… are disqualified."

**Evidence:** reproduction needs three things at once: the container up with
the graph loaded, quota available, and the submitted traces matching. Each is
a single point of failure and the first two both failed at least once today.

**Done:** the graph lives in the named volume `tg_r2_data`, so it survives a
container restart and the demo never depends on a rebuild. `finish.sh` starts
the container and waits for `gadmin status` to report Online rather than for
the port. The per-call provider split (risk 9) is a tested fallback if the
main provider is exhausted on the day.

**Closed on the quota leg, 22 Nov:** there is now a path with no provider at
all. `./offline.sh "<question>"` runs planning, selection and composition on a
local Ollama model; TigerGraph was already local and the embeddings were always
local BGE, so the whole system runs with no key and no network. Measured on an
M4/16GB with `qwen3:8b`: **123s and 2 model calls for one question, answered
correctly with citations**, against 16s on Gemini. RAM is the binding
constraint, not speed — TigerGraph holds ~5.6GB and the model at 12,288
context ~5.9GB, which fits; at 20,480 context (the Round 1 default) the run was
OOM-killed.

Two measured limits keep this as insurance rather than the default, and both
are worth stating to a judge rather than hiding: on five probe questions
`qwen3:8b` planned 4 correctly where Gemini planned 5 (it routed "athletes who
won medals at more than one Games" to the single-athlete route), and it cited
all 48 retrieved documents where the answer rested on far fewer, which costs
citation precision. `llama3.2:3b` was tried first and is not good enough — it
got 2 routes of 5 and dropped the slots the query needs.

**Open:** rehearse twice — once asking unseen questions through `./ui.sh` or
`python -m src.r2.ask "..."`, once reproducing two hidden-set answers from the
submitted file and showing the traces match. **Keep the day's quota in
reserve: live questions cost ~40 requests, a batch run costs 260** — or run the
live questions through `./offline.sh` and spend none.

## 3. The build fails on the judges' machine — *up to 15%*

**Cost:** repeatability is graded on "clean setup, locked dependencies" and a
judge may run `./build.sh` live.

**Evidence, found by testing rather than assuming:**
- The full repository lock carries Round 1's `igraph` and `leidenalg`, which
  build from source. Round 2 imports neither.
- A fresh venv on this machine's default `python3` (3.9) fails deep in
  dependency resolution complaining about `aiohappyeyeballs`, a transitive
  dependency of the Google client that requires 3.10+. Nothing in the repo
  stated a Python version.

**Done, and verified:** `requirements-r2.txt` is the transitive closure of
what the Round 2 path actually imports (84 packages), confirmed to resolve on
a clean 3.12 with no C extensions. `build.sh` checks for 3.10+, reports what
it found, and says how to point at a newer interpreter. `.python-version`
records 3.12.

## 4. Trace, log and metrics inconsistency — *up to 15%*

**Cost:** the judges recompute every number in `metrics.json` from the answer
files and say a mismatch counts against repeatability.

**Evidence:** `Trace.generate` read the usage object as a dict, but
`llm.generate` returns a dataclass. **Every token count in both submitted
files would have been 0**, failing the brief's own "tokens add up" check on
all 100 lines. Found only because a run was inspected rather than trusted.

**Done:** fixed; `metrics.json` is derived from the answer files alone, so a
mismatch is impossible rather than unlikely; each record carries
`started_at`/`finished_at` so log timestamps can be matched to it; and
`src/r2/check.py` gates the whole submission — every trace against the
brief's constraints, one real answer per question, every TigerGraph step
matched to a log line, every log line inside its own question's window. It
exits non-zero, so it can be believed.

## 5. Over-abstention: false `not_in_corpus` — *accuracy 40% + honesty 10%*

**Cost:** scores 0 on a question the corpus *does* answer, and reads as
evasion rather than honesty.

**Evidence:** the dominant failure in the first measured run — 10 of 15
imperfect answers were false `not_in_corpus`, and **every one was our defect,
not the model's caution**: an unlabelled medal tally the composer rightly
refused to read as a gold tally, a director column that was never returned, a
country name truncated to `JAP`, a grounding rule that read `10 m` and
`10 metre` as different events.

**Done:** all seven defects fixed; each affected question retested
individually and 12 of 15 now answer correctly.

**Open:** re-measure on a complete run. The number to watch is `status_correct`,
which was 40/50.

## 6. Citations that do not support the answer — *10%*

**Cost:** a tenth of the score, and the easiest to lose silently.

**Evidence, measured on the 25 completed answers:** recall **0.968**,
precision **0.953** (up from 0.779 before the fixes). Two outliers in opposite
directions: a count question cited all 18 event documents where the key cites
the 8 that were counted (precision 0.44), and a "which film" question cited
only the winner where the key cites all five candidates compared (recall 0.20).

**Done:** the composer is now told to cite what a reader would have to check —
for a count, the records counted; for a "which one of these", the candidates
compared; for a single fact, the document stating it — and that citing
everything in the evidence block is as wrong as citing too little.

**Open:** re-measure. The gate already drops any citation absent from the
retrieved evidence, so the failure mode left is over- or under-citing real
documents.

## 7. Hidden-set question families handled worse than the public ones — *40%*

**Partly closed 22 Nov:** the hidden set now gets two checks that need no gold
answers and no quota. `python -m src.r2.audit` holds each answer against the
arithmetic its own record carries and prints a worklist worst-first — a figure
the query never computed, a tie reported as one holder, a status contradicting
its own text, a route whose queries all returned nothing. `python -m
src.r2.smoke` proves all ten route cases still return evidence from the graph
before a run is paid for; it caught `event_series` returning 758 rows and
citing none of them. Neither replaces reading the answers, and the three
filters below still apply.


**Cost:** the hidden set is what is actually scored, and we cannot measure
against it.

**Evidence:** families visible in the hidden questions that the public 50
exercise only lightly — a team result where the medal line names a whole squad
(men's ice hockey), ratio arithmetic (gross ÷ budget), an as-of date, "the
Games immediately before/after", and a total-years-served calculation across
two office blocks in different articles.

**Open and the plan:** after the hidden run, review **every** `not_in_corpus`,
every answer with no citations, and every answer whose route looks wrong for
the question. Those three filters catch a family failing without needing a
key. The layer probe already showed all 30 public families reachable from the
parsed records, so a failure here is an agent problem, not a data one.

## 8. Rule 4 — question-specific rules in the repository — *judgement call, could be fatal*

**Cost:** the judges "will check the code." Being *suspected* of tuning is
nearly as bad as doing it.

**Evidence:** two places in this repository pattern-match on question text.
Both are Round 1: `src/pipelines/jev_planner` and `src/pipelines/event_graph`.
`scripts/r2_layer_check.py` hard-codes question parameters deliberately, as a
development probe.

**Done:** no code in the Round 2 answering path reads the wording of a
question — the model interprets it and may only emit keys the graph just
handed it. Both Round 1 files carry a header saying they are not in the Round
2 path, the probe says so in its first line, and **the README names all three
itself** rather than leaving them to be discovered by grep.

**Residual:** low, and accepted. Deleting them would erase a reproducible
Round 1 result.

## 9. Efficiency figures that look bad, or dishonest — *10%*

**Cost:** a tenth of the score, and a cost figure that cannot be audited is
worse than a high one.

**Evidence:** 2.6 model calls and ~5.2k tokens per question, median latency
16s. But the latency includes **rate-limit pacing**: calls are spaced to stay
under the per-minute cap and that sleep happens inside the call, so roughly
`4.5s × (calls − 1)` per question is waiting, not work.

**Done:** the README states the pacing and the arithmetic to subtract it, from
numbers already in `metrics.json`. Each trace step now records which provider
served it, and `cost_usd` prices only tokens actually billed — charging a free
call at a paid rate would have overstated cost roughly threefold.

**Also available, tested, not enabled:** `SMALL_CALL_PROVIDER` moves planning
and selection to a second provider. The planner prompt is ~1.3k tokens and a
selection 0.4–2.8k, while the answer prompt reaches 11k — and Groq's free tier
allows 1,000 requests a day but caps a request at 8,000 tokens per minute and
**rejects anything larger outright rather than throttling**. So every stage
except the answer fits there. Verified: the planner produces the correct plan
on Groq, and two questions ran end-to-end on it before a third exceeded the
window exactly as predicted. Left off by default because the submitted runs
should use one validated configuration, and accuracy at 40% outranks
efficiency at 10%.

## 10. Answer formatting that loses partial credit — *40%, at the margin*

**Cost:** scoring is fact by fact. A complete list buried in prose, or a
number without the thing it counts, loses marks the system has earned.

**Evidence:** the judge marked two answers down for incompleteness rather than
error — a missing reallocation after a disqualification, and one of several
reasons given in an article.

**Done:** the composer is told to give every item for a list and nothing else,
to state what a count counted, to give the consequence as well as the cause,
and to give every reason the text offers rather than the first.

**Residual:** real but small, and only measurable through the next score.

---

## What this says to do next, in order

1. **Run `finish.sh`** when the quota resets. Nothing else matters until both
   files exist. *(risk 1)*
2. **Read the score, then risks 5, 6 and 10** — `status_correct`, citation
   recall/precision, and the judge's "missing" notes are each a direct
   instruction about what to fix.
3. **Review the hidden run through the three filters** in risk 7. This is the
   only handle we have on 40% of the score.
4. **Rehearse the live reproduction twice**, and keep the day's quota free.
   *(risk 2)*
5. Tag only after `check.py` passes on both files.
