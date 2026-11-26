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

**Measured 22 Nov, and WRONG -- corrected 24 Nov.** The claim was that the
daily cap is per model, so the judge id `gemini-3.1-flash-lite-preview` had its
own 500/day pool and scoring was free. Two things are wrong with it. Google's
rate-limit page says limits are "applied per project, not per API key" and
says nothing about two model ids getting separate daily allowances; it
publishes no free-tier requests-per-day figure for any Gemini 3.x Flash Lite
id at all, so the 500 was never a documented number for this model. And
measured directly: 121 + 117 + ~72 + 61 = **~371 answering calls inside one
Pacific day** (timestamps checked, no reset between them) were enough to
return `429 RESOURCE_EXHAUSTED`. Budget judge calls against the same pool and
plan on ~350 usable calls a day, not 500. The judge setting was also not in
effect for the committed score -- see the self-grading entry below. The run plan for the day is ~240 requests on the
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

**Closed on the measurement leg, 23 Nov:** "showing the traces match" was a
thing to be done by eye, which is the wrong way to hold a fatal risk.
`python -m src.r2.repro <questions> <answers> --qid <id>` re-answers a
submitted question and returns one of three verdicts. It cannot be byte
equality — the composed sentence is model-written, and two runs of a perfectly
repeatable system word it differently — so what it compares is everything the
score is computed from: the status, **every number in the answer in order**,
the citation ids as a set, and the route with its sequence of TigerGraph
queries.

- `MATCH` — status, figures and citations all identical
- `CLOSE` — status and figures identical; citations or wording moved
- `DIFFERENT` — a status or a figure moved, and it exits non-zero

The figures check is the one that matters. Every count, argmax and tie is done
in code from graph rows, so a figure that moves between two runs means
something non-deterministic is feeding the arithmetic and the submitted answer
was luck — which is exactly how the `films_with_figures` `LIMIT 200` bug was
found, by hand, after it had already changed one answer twice.

**Open:** rehearse twice — once asking unseen questions through `./ui.sh` or
`python -m src.r2.ask "..."`, once running `src.r2.repro` on two hidden-set
answers. **Keep the day's quota in reserve: live questions cost ~40 requests,
a batch run costs 260** — or run the live questions through `./offline.sh` and
spend none. Note that a reproduction run has to use the same provider as the
run that wrote the file; `repro` prints the provider it used and says so when
it is not `google`.

## 2b. Citations on the evidence route — *measured 23 Nov, open*

**Cost:** citations are 10% of the score.

**Measured on the 50 public answers of the 23 Nov run**, against the key's
`gold_doc_ids`: **recall 0.883, precision 0.919** over the 48 questions that
have gold documents. 17 of the 48 are imperfect, and the misses are not spread
evenly — they are two distinct patterns:

- **The evidence route under-cites.** 9 of the 11 questions that lost a gold
  document are `evidence`, each missing one or two of the source records in a
  disagreement. r2-pub-048, 049 and 050 all cite the official record and miss
  the rival claims, even though the composer prompt says to cite every source
  in a disagreement. It is not the gate discarding them: **zero citations were
  dropped on the whole run**, and the rival records are registered as citable
  by `_cite`. The composer was shown them and chose one.
- **The medal routes over-cite.** 30 spurious citations, 26 of them from three
  `sport_games` questions: r2-pub-006 cites 18 documents where the key wants 8,
  r2-pub-033 cites 14 where it wants 4. It cites every event in the sport
  rather than the events the answer rests on.

**Candidate fix, not attempted yet, and the reason:** the graph knows which
claims concern the same slot and which `SUPERSEDES` which, so code could
complete a citation set the way code already does the arithmetic — cite one
member of a disagreement and the rest come with it. That is the right shape of
fix and it is deterministic. It is not in this run because it trades recall
against precision and both are scored, and measuring the trade costs a full
paid run. One day's quota buys one run, and the day it was found the budget was
already committed to the four status errors, which are worth more. **Measure
it on a fresh day's quota before shipping it.**

## 2c. What a half run bought — *measured 23 Nov, two fixes applied*

The day's remaining quota was spent re-measuring the public set after the
passage-per-document change, because that change touches every question that
reads article text and seven targeted checks are not a regression test. The
run stopped at **28 of 50** on the provider's daily limit. Scored against the
key the free way — status and `gold_doc_ids`, no judge calls — the 28 gave:

| | committed run | re-run |
|---|---|---|
| status correct | 27/28 | 26/28 |
| citation recall | 0.970 | 0.896 |
| citation precision | 0.872 | 0.902 |

One fix, two regressions, and both regressions were worth more than the noise
they looked like:

- **r2-pub-023 was fixed, but not by the change that was aimed at it.** The
  trace shows the route stayed `event_series` and the *widening fallback* read
  the prose after the first pass came up empty. The planner rule added the
  same day — "a *why* question always wants prose" — contributed nothing here.
- **r2-pub-022 was broken by that planner rule.** It asks why Romankov was
  absent from the 1984 men's foil. The committed run planned route `evidence`,
  found the claim in Q5443136 and answered correctly. The re-run planned route
  `text` with `wants_prose: true`, found nothing, and reported the corpus
  silent. The rule said a *why* question wants prose; the route list says
  `text` is "the prose of an article"; the planner joined them. The flag was
  never meant to pick a route, and the widening fallback means it does not
  need to. Fixed by saying so in the prompt — the flag never decides the
  route — and by demoting `text` in the route list to the last resort, for a
  question that names nothing the graph is organised by.
- **r2-pub-007 was right and scored `partial` anyway.** "Japan dominated judo
  at the 2020 Summer Olympics with 9 golds" — correct nation, correct figure,
  both computed in code — with **zero citations**, because the composer
  offered the literal string `sport_games` where a document id belongs. The
  gate dropped it correctly and demoted the answer, which is the right call in
  general (see r2-pub-029) but not here: when code did the arithmetic, the
  documents it counted over are known exactly. So the gate now substitutes the
  retrieved set when a tally is behind the answer and nothing else survived,
  bounded at 25 documents — the largest gold set in the public key is 16, and
  a comparison route can retrieve hundreds, where citing all of them would buy
  a citation mark with a precision one. This is the narrow, strictly
  non-regressive corner of the 2b fix: it only fires where the behaviour today
  is zero citations.

**The lesson that generalises:** a prompt rule written to fix one question was
measured, on a different question, to cost more than it gained — and the
question it was written for had already been fixed by a code change. Seven
targeted checks said 6/7. Twenty-eight measured questions said the net was
negative. Targeted validation cannot see a regression it did not aim at.

**Caveat on these two fixes:** both are verified by unit tests (63 checks, no
model, no database) and by reading the traces. Neither has been through a
paid run — quota was exhausted at the moment they were written. They are the
first thing the next full re-run has to confirm, specifically r2-pub-022 back
to route `evidence` and r2-pub-023 holding `answered`.

### A test-suite bug found while writing those tests

`CHECKS` in `tests/test_r2.py` was a module-level list built at import time,
so any check written below that line was collected before it existed and
silently skipped — four new ones, with the suite still reporting all green.
Collection now happens inside `main()`, so a check counts wherever it is
written in the file. Worth recording because a test suite that under-reports
its own size is the one kind of bug the suite cannot catch.

## 2d. What an audit of the unrehearsed 30% found — *24 Nov, seven defects*

Accuracy had been measured to death and the two criteria worth 30% together,
repeatability and live judging, had never been rehearsed once. Auditing them
found seven defects, every one of which would have been visible to a judge and
none of which any accuracy measurement could have surfaced.

**The arithmetic that justified looking there.** Of 4.19 accuracy points lost
across 7 public questions, 2.0 were already fixed, 1.5 is the pub-048/049
borderline where the model sits near even, and the three never-examined
questions are worth **0.69 points — 0.55% of the final score**. Thirty points
sat untested. Chasing accuracy was the expensive way to win a rounding error.

### 1. The reproduction tool disclaimed every reproduction it performed

`repro.py` compared the provider against the literal `"google"`. Nothing in
this project is ever `"google"` — the default is `gemini` in `llm.py` and
`trace.py`, `.env.example` says `gemini`, and **all 238 model steps across both
submitted answer files record `gemini`**. So the tool printed "this is not the
provider the submitted file was produced with, so a match here is evidence
that the arithmetic is deterministic, not that the submitted answers
reproduce" on every run, including the only configuration where that sentence
is false. On a judging call, on the 15% criterion, built to answer "show me
this reproduces". It now reads provider and model out of the submitted record,
which cannot drift. Five tests covered `compare()`'s verdict logic and none
touched the provider line; that is how it survived.

### 2. The reproduction gate compared two of the four things it promised

`compare()`'s docstring names status, figures, citations and retrieval as the
four that must not move. The verdict used two. Verified by calling it
directly: **"France won 13 silver medals" against "Italy won 13 silver
medals" returned `MATCH`** — `figures()` extracts only digits, so the identity
of the entity an answer names was never compared. For a system whose central
claim is that code does every count, argmax and tie, a reproduction check
blind to which holder won checks the easy half. A changed route and a halved
citation set also passed as `MATCH`, while "returns no `DIFFERENT`" is the gate
the submission doc shows a judge.

`names()` now compares stated proper names, treating a *substitution* as
`DIFFERENT` and an *addition* as a rewording, so "Japan" against "Japan (JPN)"
still matches. That asymmetry is deliberate: a check that cries wolf gets
ignored, which is exactly what happened to the provider disclaimer.

### 3. A fresh clone was not the system that was measured

`.env` is gitignored and `load_dotenv()` only finds it on the machine that
wrote it, so for anyone cloning the repository **the code defaults are the
configuration** — and they had drifted in five places: embeddings to
`gemini-embedding-001` instead of local BGE, the answering model to
`gemini-3.8-flash` which was never measured, plus the pacing and timeout the
reported efficiency figures came from.

The embedding default is the dangerous one because it does not fail. With a
key exported it embeds all 16,377 chunks at the same 768 dimensions into a
different vector space from the graph's, nothing errors, and `vector_search`
quietly returns noise — a judge sees a working build answering worse than the
submission, with no error explaining why. Defaults now equal `.env.example`,
`build.sh` writes a `.env` on first run, and a test fails if they ever
disagree, verified by reintroducing the drift and watching two checks fail.

### 4. The 91.6% was graded by the model that wrote the answers

`score.py` read `JUDGE_MODEL` at module scope, but the project's only
`load_dotenv()` lives in `src/common/llm.py`, which `score.py` imports lazily
*inside* the judge function. The read returned `None` and the fallback silently
took `GEN_MODEL`. `answers_public.jsonl.score.json` records `judge_model:
gemini-3.1-flash-lite` — the model that wrote all 121 steps it graded — while
`metrics.json` asserted a judge "different from the one that produced the
answers". Three committed statements were false.

A consequence worth keeping in view: **every per-question note in that score
file was written by a self-grading judge**, so they are weak evidence. One
such note described pub-049 as "an invented claim that LA28 archery operations
made a statement". The cited source `SYN-LA28-S028` *is* from LA28 archery
operations and *does* say "no official duration was recorded" and that the
reports "could not be verified" — the answer was a fair paraphrase and the
note was wrong. pub-049's failure is its status, nothing worse. Acting on that
note would have spent a run chasing a defect that does not exist.

### 5. A limit over an unordered select, still live

`chunks_containing` had `LIMIT k` with no `ORDER BY`, over the query that feeds
the whole lexical path, with `k=12` that a common fragment overflows easily.
This is the defect class the project already knew — `films_with_figures`
carries no limit at all because `LIMIT 200` over ~250 unordered rows changed
an answer between runs — sitting unfixed one query over.

Ordered by `ord ASC, chunk_id ASC`: position first rather than `doc_id` first,
because the caller keeps one passage per document, so ordering by position
takes the opening chunk of *every* matching document before the second chunk
of any. Measured after reinstalling all 32 queries: `"100 metres"` returns 12
chunks from **12 distinct documents** where before it was an arbitrary 12,
`"Cocoon"` 10 from 7 matching this query's own documented count, and three
successive calls return byte-identical rows. It improves retrieval as well as
repeatability. `_lexical_search` also sorts before capping, since the query's
order does not fix which chunk represents a document.

### 6. The live-demo path failed with a traceback instead of a reason

`offline.sh` — which genuinely plans, retrieves and composes an answer to an
invented question with no key and no network, and is the strongest asset for
the call — died on a fresh clone with "No such container: tigergraph". It ran
`docker start` with no `docker ps -a` check that `build.sh` has had all along.
It now says the graph must be built first and deliberately does **not** create
the container: `docker run` would bring up an empty graph, and an empty graph
does not look broken — it answers everything "not in the corpus", which reads
as a working system being confidently wrong.

A dead Docker daemon and an unbuilt graph both surface from `docker ps` as the
same failure, so the first was reported as the second, advising a rebuild when
the fix is starting Docker. A blank `GOOGLE_API_KEY` reached the Google client
as `ValueError: No API key was provided` — the empty string is falsy, so it
resolves to `None` rather than raising `KeyError` — and sat under a
five-attempt exponential retry: a config error cost ~30 seconds and five
identical failures before any message. Now one attempt, measured 0.00s,
printing the two ways to proceed.

### 7. The test suite under-reported its own size

`CHECKS` was a module-level list built at import time, so four checks written
below that line were collected before they existed and silently skipped while
the suite reported all green. Collection moved into `main()`. Worth recording
because a suite that cannot count itself is the one bug it cannot catch.

### Deliberately not fixed, and why

- **pub-013's over-answering** (0.12% of the final score). The athlete route
  calls `team_medals_for_name` unconditionally and the composer prompt invites
  the extras with "if you include them at all". A fix means a second composer
  edit in the same run as the Rule 4 removal, leaving the result unattributable
  — the mistake made on 23 Nov, when two simultaneous changes took three
  measurements to untangle.
- **pub-024's dropped prose fact.** The sentence was verbatim in the prompt at
  offset 1375 of the 4500-char clip, and the instruction to use it
  ("give every reason the text offers") is already present and was not
  followed. The only lever is more prompting, which already failed here. Three
  counterexamples on the identical path (pub-010, 011, 045, all 1.00) rule out
  the prompt-dilution explanation.
- **The risk 2b citation-completion fix.** It was predicated on a 500/day
  budget that does not exist.

## 2e. The live reproduction, rehearsed at last — *24 Nov, both halves pass*

Risk 2 has carried "rehearse the live reproduction twice" as open since it was
written, and `repro.py` had never once been run against a real answer file —
which is how the `"google"` provider bug survived five tests. Rehearsed now,
on the committed pair, on a cold-started container.

**Half one — "show me this answer reproduces."**

```
$ python -m src.r2.repro questions_public.jsonl answers_public.jsonl \
      --qid r2-pub-006,r2-pub-030
connecting: local @ http://localhost graph=r2
answering with: provider=gemini model=gemini-3.1-flash-lite
submitted file was produced with: gemini/gemini-3.1-flash-lite

ok  r2-pub-006  MATCH      figures ['2024','8'] -> same, citations 18 -> 18,
                           retrieval identical (4 steps), wording 92% similar
ok  r2-pub-030  MATCH      figures ['1985'] -> same, citations 1 -> 1,
                           retrieval identical (7 steps), wording 47% similar

2 re-answered: 2 MATCH, 0 CLOSE, 0 DIFFERENT
```

Two things worth pointing at in a call. The provider line now *names what the
file was made with* and prints no disclaimer, because the provider matches —
the old code said "this is not the provider the submitted file was produced
with" every single time. And **pub-030 matched at 47% wording similarity**:
the composed sentence is model-written and came back substantially reworded,
while status, every figure, the citation set and the whole query sequence were
identical. That is the distinction the checker exists to draw, and it is also
a live demonstration that the hardened `names()` check does not fire on
rewording — a 47%-different sentence naming the same entities still matches.

**Half two — "now ask it something we just made up."**

Asked, unseen and not in either question file: *"Which nation won the most
gold medals in judo at the 2024 Summer Olympics, and how many?"*

> `not_in_corpus` — "The corpus does not contain information regarding the
> gold medal results for judo at the 2024 Summer Olympics; the only document
> provided for that event is a qualification overview." Cited: Q114270340.

Verified against the corpus afterwards: it holds judo medal events for 1988
through 2020 and, for 2024, exactly one document — `Q114270340 "Judo at the
2024 Summer Olympics – Qualification"`. The refusal is literally correct and
cites the document that proves the gap.

This is the strongest demo available, because **the plausible wrong answer is
one edition away in the same graph**. r2-pub-007 asks almost the same thing of
2020 and is answered "Japan, 9 golds" from a tally of 15 documents. A system
that pattern-matched the sport would have produced that. The trace shows it
did the honest thing instead: `coverage_check`, `events_in_sport_games`,
`medals_by_nation`, compose, then — finding nothing citable — the widening
fallback (`embed_question`, `vector_search_chunks`, `doc_for_chunk`) and a
second compose, before reporting silence. 32.3s, 9,003 input and 312 output
tokens, three model calls.

**Rehearsal two, 25 Nov — and it found the thing rehearsal one could not.**
Run on three questions chosen at random (`--sample 3 --seed 7`) rather than
the two the first rehearsal happened to pick, it reported **2 of 3 correct
reproductions as DIFFERENT**:

- `r2-pub-021` — figures `['63.5','2028','40']` against `['40']`. The re-run
  simply stopped restating "the men's 63.5 kg at the 2028 Games". Its answer,
  suspended two hours on gusts over 40 km/h, was identical.
- `r2-pub-010` — the same seven per-edition nation counts, the same most and
  the same fewest, with the closing restatement phrased once instead of twice.
  Wording similarity 19%.

Both false alarms, from comparing figures as an *ordered list*. The composed
sentence is model-written, so how much of the question it restates varies run
to run while every fact holds. **A gate that fires on two of three good
reproductions is worse than no gate**, because a judging call reads it as
"this does not reproduce" — the identical failure mode as the hardcoded
`"google"` disclaimer, reached from the opposite direction. Rehearsal one
passed only because its two questions happened to phrase themselves stably.

Fixed by judging figures as a multiset under the same
substitution-not-addition rule as `names()`. Re-rehearsed on the same three:
**2 MATCH, 1 CLOSE, 0 DIFFERENT**, and pub-010's CLOSE now also surfaces that
its query sequence moved from 5 steps to 4 — which the old verdict discarded
without mentioning.

The sequence of three rehearsals is worth keeping in view, because each one
found a different class of defect in the same tool:

| | what it found |
|---|---|
| rehearsal 1 | the tool worked, and the provider disclaimer was gone |
| rehearsal 2 | the gate was too strict: 2 of 3 good runs called DIFFERENT |
| rehearsal 3 | the fix holds on the data that broke it |

**Still open:** rehearse once more before the call, and prefer `./offline.sh`
for practice since it needs no key and no quota. Note also that Docker Desktop
on this machine has twice needed a `pkill -f com.docker` and a relaunch after
being quit — `docker info` hangs rather than failing. Start it well before the
call, not during it.

## 2f. Two efficiency fixes, measured and declined — *24 Nov*

Both were proposed as one-line changes. Measuring them is what settled it, and
in one case the measurement contradicted the first reason for refusing.

**Gating `select_source_document` on the deterministic date match.**
`src/r2/agent.py:1092` serialises the whole source catalogue into a prompt for
every `evidence` question — 2,762 input tokens a call, 9.6% of all input
tokens — even when the `by_date` match two lines above already pinned the
document. The proposal is to skip the model call when `by_date` is non-empty.

Declined, on the measurement: **only 1 of the 9 public evidence questions
carries dates in its plan at all**, so the gate would skip one model call per
50-question set, not fourteen. And that single question is exactly the case the
two-channel design exists for — a question that gives a date *and* names a
second document, where `by_date` finds the first and the model finds the
other. One call saved, on the one question where losing a document is likeliest,
on the route that already under-cites (risk 2b). Not worth it.

(My first reason for declining was that r2-pub-041 is that shape. It is not —
its plan carries no dates, so the gate would not fire on it. The measurement
is the reason; the guess was wrong.)

**Memoising `claims_by_source`.** It runs in two loops with no shared memo
(`src/r2/agent.py:1097` and `:1109`, plus a third channel at `:1312`), so any
document reached by two channels is queried twice; it is 34 of 57 duplicate
queries. Declined for now, and **not because it is risky to dedupe a
read-only query** — because `seeds` accumulates across both loops under
`if len(seeds) > 60: break`. Removing duplicate rows changes when that break
fires, which changes the evidence, which changes answers. That makes it a
semantic change needing a paid run to verify, not a free cleanup. Worth doing
with the break condition reviewed at the same time.

**What the numbers actually say, re-measured on the committed re-run.** 418
public queries, mean 8.36 per question (down from 10.60 before today's route
changes), **median 4**, 5.0% byte-identical repeats within a question. The
entire query layer costs a few seconds of the run — around 0.6% of latency. So
the redundancy is worth a fraction of a second and the real cost of the
duplication is presentational: the mean invites a question the median answers.
Fixed in `README.md` by quoting both and naming the route split, since
`evidence` is 9 of 50 questions and 52% of all queries.

## 2g. A provider outage was indistinguishable from an empty corpus — *25 Nov, honesty 10% + accuracy 40%*

A paired verification run from `ba9cee1` was spoiled, and what spoiled it was
worth more than the run.

**What happened.** Gemini returned `503 UNAVAILABLE` — *"This model is
currently experiencing high demand"* — 39 times across the pair. Retries
exhausted, so `run.py` fell to its generic exception handler, which writes a
placeholder record. That placeholder's status is `not_in_corpus`. The hidden
half took 33 of them consecutively; the public half took 7, scattered.

**Why this is the worst possible failure shape.** `not_in_corpus` is a claim
about the corpus. It was written because the *model* could not be reached, and
those two things have nothing to do with each other. The resulting hidden file
has 34 `not_in_corpus` answers out of 47 and looks finished. One of them was
r2-041, whose answer in a good run traces Martin Fourcade's medals including
the 2018 Winter Mixed relay — a question the corpus plainly answers, recorded
as a question the corpus does not contain. A judge reading that file sees a
system that is confidently wrong about two thirds of a set.

**Why the existing guard missed it.** `run.py` already stops dead on a quota
wall rather than papering over the rest of the file, and `finish.sh` says so in
a comment. But `_is_quota_error` matches `RESOURCE_EXHAUSTED` and `429`, and a
503 is neither — correctly, since a 503 *is* transient and worth retrying. The
guard was built for the failure that had already happened. One transient 503
deserves a placeholder; thirty-three in a row is an outage.

**Fixed.** A streak counter: `OUTAGE_STREAK = 3` consecutive placeholder
writes stops the run with the same contract as the quota wall — keep what
worked, name the error, say to re-run with `--resume`. `--resume` already
re-answers any record with no trace, so the placeholders are not sticky. One
isolated failure still does not lose the run, which is what kept 43 real
answers in the public half of this very run. Four tests, two on the
classifier and two driving `main()` through a stubbed agent.

**What it cost.** The day's quota, and the run. 1–14 answered, 15–47 lost to
the outage, 48 hit the real 429 at ~20:00 IST. The committed artifacts were
never overwritten — the run wrote to a scratch path — so nothing regressed.

**What is still open.** The committed hidden pair still exits 1 on `check.py`,
with six problems, all of them the 12 orphan log lines left by the r2-035 and
r2-036 503s on 24 Nov: 6 extra for r2-035, 3 for r2-036, 3 duplicate
`__bootstrap__` lines, and `431 - 12 = 419`. `check.py` itself identifies each
one as *"outside the question's own window"*. The code fix (`_prune_log`) landed
in `e78af0a` but cannot retroactively repair an artifact written before it
existed. Hand-pruning those 12 lines would make the check pass; it would also
mean the shipped log is not the log the run wrote. The artifacts should come
from one clean run instead, and the committed public pair already passes with
exit 0.

**Also fixed, found by making the mistake.** `check.py` took its two
positional arguments the wrong way round here and raised `KeyError: 'trace'`
from the middle of a comparison. It now names the mistake and exits 2, and it
no longer assumes a `trace` key exists. A traceback on the deliverable a judge
runs by hand reads as a broken system; a sentence reads as a typo.

---

## 2h. Both halves pass at last — *26 Nov, the submitted pair*

One clean paired run from `b44af87`, the first commit carrying both outage
guards. Public 10:24-11:06Z, hidden 11:06-11:37Z, scored 11:37Z.

| | public | hidden |
|---|---|---|
| `check.py` exit code | **0** | **0** |
| answers | 50/50 | 50/50 |
| status split | 44 answered, 4 not_in_corpus, 2 partial | 38 answered, 4 not_in_corpus, 8 partial |
| answers with a TigerGraph step | 50 | 50 |
| query log | 422 lines / 51 ids | 434 lines / 51 ids |
| accuracy | **92.3%**, 44/50 fully correct, 46/50 status | key held by the judges |
| citations | recall 0.891 / precision 0.914 | — |

Graded by `gemini-3.1-flash-lite-preview` against answers written by
`gemini-3.1-flash-lite`, which `score.py` states in its own header and
`metrics.json` records as `judge_independent`.

**The outage guards earned their place on the first run after they landed.**
Three public questions failed mid-run -- r2-pub-025 and r2-pub-048 to 503
UNAVAILABLE, r2-pub-033 to **504 DEADLINE_EXCEEDED**, a fourth code this
project had not seen. They were scattered rather than consecutive, so
`OUTAGE_STREAK` correctly did *not* fire and 47 real answers survived. The
first score, with those three placeholders in the file, read 88.3% / 44 status
/ recall 0.829. `--resume` re-answered exactly those three, `_prune_log`
dropped their 14 orphan log lines, and the file went to 92.3% / 46 status /
recall 0.891 with `check.py` at 0. Without `_prune_log` that resume would have
produced the same orphaned-log failure the committed pair had been carrying
since 24 Nov.

504 was added to `finish.sh`'s keep-waiting set alongside 503. It is the same
congestion seen from the other end.

**What this costs against the previous pair.** The 24 Nov pair scored 92.5% /
47 status / recall 0.909 / precision 0.896 -- two tenths of a point and one
status better on accuracy, with higher recall and lower precision. It also
exited **1** on `check.py` for the hidden half. Trading 0.2 accuracy points for
a self-consistent deliverable 5 is the right trade: repeatability is 15% and a
judge runs that command by hand.

**The one real defect the score exposes, and it is a family.** Three questions
turn on *what the absence of a record means*, and the system gets it wrong in
both directions:

- **r2-pub-048** (0.50) — invented an official record stating the duration
  could not be established, where the key says the official status is null.
- **r2-pub-049** (0.00) — answered a question whose key is `not_in_corpus`.
- **r2-pub-050** (0.00) — said `not_in_corpus` where the key wants the answer
  that *a missing log entry does not prove the absence of the thing*.

That is one idea, missed three ways: an unrecorded event is neither a
confirmed absence nor licence to invent a record. 048 and 049 are
over-assertion, 050 is over-abstention, and all three would be addressed by
the same instruction. Worth up to ~2 points of the 40% weight and it is the
largest remaining lever on the public set. Not attempted on 26 Nov: the day's
quota went on the run and two scorings, and a prompt change to the evidence
block invalidates the pair that now passes, so it needs a full re-run behind
it rather than a patch on top of a submitted artifact.

---

## 2i. Fourth reproduction rehearsal, against the tagged pair — *26 Nov*

The first three rehearsals tested artifacts that are no longer the submission.
Three questions chosen at random from the newly tagged pair, re-answered
against the real provider:

```
3 re-answered: 0 MATCH, 3 CLOSE, 0 DIFFERENT
  r2-pub-006  answered -> answered   figures ['2024','8'] -> ['2024','8']
  r2-pub-028  not_in_corpus -> not_in_corpus   figures identical
  r2-pub-041  answered -> answered   figures [] -> []
```

Status identical on all three, figures identical on all three, no DIFFERENT.
The provider line reads `gemini/gemini-3.1-flash-lite` on both sides, which is
the line that silently disclaimed every reproduction before `repro.py` was
fixed to read it from the record.

CLOSE rather than MATCH on all three because retrieval moved (4->5, 9->7,
16->12 steps) and r2-pub-006's citation set went 18 -> 8. Retrieval moving is
the model choosing what to retrieve, which is the one decision it is supposed
to own; the gate is right to report it and right not to call it a failure.

**One thing checked rather than assumed.** r2-pub-006 answers "Japan, 8" to a
*which nation won the most, and how many* question, and losing 10 of 18
citations on a re-run looked like it might mean the model was doing the
counting. It is not. The record carries a code-computed block reading `gold
medals by nation, wrestling at 2024 Summer | unit: gold | highest: 8 ['JPN']`,
and the answer is that argmax read back. `computed` is present on 33 of 50
public answers and 30 of 50 hidden ones. The 18 citations are the documents the
count ran over, which is exactly the gate's computed-citation rule; the
re-run's 8 is the less complete set, not the better one.

Worth recording because the first two attempts to check this looked at the
wrong field (`tallies`, which does not exist — the field is `computed`) and
then at the wrong block of six (`nations per event`, a different axis from the
question). Both near-misses would have been reported as a central design rule
failing.

---

## 2j. The one question whose failure is retrieval, not judgement — *26 Nov, unmeasured*

Of the six public questions that lost points, `r2-pub-050` is the only one
whose answer is a sentence the corpus states outright.

> **Q.** Does the absence of a central operations-log entry prove that no
> equipment pause happened?
> **Key.** No. It means no verified entry exists; the event may or may not
> have had a pause. Both witness estimates remain unverified.
> **Ours.** `not_in_corpus` — "the provided documents do not address the
> evidential value of such absences".

`SYN-LA28-S028` ends: *"The check could neither confirm nor rule out a
pause."* We hold the answer and said we did not.

**The trace says why.** The plan came back `{"route": "evidence",
"wants_evidence": true}` and nothing else — no entities, no sport, no event
phrase, because the question names no athlete, sport, year or event. The
evidence route's lexical pass searches `plan.entities + event_phrase +
athlete + person`, so with an empty plan it searched nothing. Its own
docstring describes the capability that would have saved this question — *"a
search for 'equipment pause' finds the operations log that a search for the
athlete does not"* — and the plumbing to reach it was missing.

That leaves the source-catalogue selection as the only channel, and it
returned exactly one record: `SYN-LA28-S038`, the triathlon race director's
log, whose Source line reads *"signed operations log"*. Two of the 39 records
answer to "an operations log"; the question's other term, *equipment pause*,
appears in only one of them, and that is the one not chosen. The heading match
beat the body match.

**What was rejected, and why.** The obvious fix is to search on the question's
own words. The README and risk 8 both advertise that *no Round 2 code reads
the wording of a question*, and the judges "will check the code". Spending a
documented Rule 4 property — rated *could be fatal* — on one public point is a
bad trade, so no code was added that reads question text.

**What was done instead.** `_select` gained a `guidance` argument, passed at
exactly one of its six call sites: the source catalogue, reached only by the
evidence route. It tells the selector to judge a record by its whole text
rather than by how many of the question's words land in its title or Source
line, and to return every record that could bear on the question rather than
only the closest. The model still does all the interpreting; the code still
only uses keys the graph handed it. A test asserts the other five catalogues'
prompts are byte-identical, and another asserts the guidance names no
question, document id or gold value.

**Why the risk is lower than it looks**, which the existing traces settle for
free:

| question | score | source records selected |
|---|---|---|
| 039, 040, 043, 044, 047 | 1.0 | 11, 11, 20, 6, 18 |
| 022 | 1.0 | 0 |
| 041 | 0.5 | 5 |
| 049 | 0.0 | 15, including the right one |
| **050** | **0.0** | **1, and it is the wrong one** |

Every evidence question that scores full marks already selects a broad set,
and the one that fails is the only one that returned a single record. The
guidance asks for the behaviour the working cases already have; it is not a
new behaviour imposed on them.

**Scope, honestly.** This can only fix 050. `r2-pub-049` already retrieved
`SYN-LA28-S028` and still scored 0.0, because its failure is a status
judgement — the key declines a question that presupposes an officially
confirmed stoppage, and we answered it correctly on the facts with the wrong
status. `r2-pub-048`'s loss is the same family: our answer opens by asserting
the stoppage *occurred*, where the only official check "could neither confirm
nor rule out" it. Both want a rule about asserting an occurrence that no
record confirms, which is a composer change with a blast radius across all 100
questions. Not attempted without the quota to measure it.

**Status: unmeasured, and parked off `main`.** Today's quota went on the
submitted pair, two scorings, the resume and the rehearsal; the probe for this
change returned `429` after writing zero answers, which is the outage guard
behaving correctly.

It now lives on the branch `source-pick-guidance` and **not on `main`**,
because leaving it on `main` broke something worth more than the point it
might win. `r2-submission` is two commits behind `main`, so `main` carried a
38-line change to `agent.py` while still carrying the answer files produced
*without* it. A judge who clones `main` rather than the tag, re-runs, and gets
different answers than the committed ones is looking at a repeatability
failure — 15% of the score — caused entirely by an unmeasured change kept in
the wrong place. `main`'s `agent.py` and test suite are now byte-identical to
the tag's, verified by `git diff r2-submission -- src/r2/agent.py
tests/test_r2.py` returning nothing. This document and
`measure_evidence.sh` stay, because neither can change an answer.

The quota may not come back in time. 29 Nov is the submission day rather than
a working day, and 27 and 28 are not certain, so the tagged pair is the
submission unless a measurement clearly beats it. Merging the branch is
therefore gated on completing *both* stages — the 9-question gate and a full
paired re-run — in one day. A half-finished promotion is worse than no
promotion. A local-model probe of the selector returned nothing either way, so
qwen3:8b cannot stand in for this. `./measure_evidence.sh` runs stage one when
the quota resets: the 9 evidence questions and their scores against the
baseline 6.5/9, written to a scratch path that cannot touch the tagged
artifacts. A full paired run follows only if that improves. If it does not, the
change is reverted and `r2-submission` stands as tagged.

---

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

**Corrected 22 Nov:** the 40/50 figure was wrong. It counted 25 placeholder
records, written before `run.py` learned to stop dead on a quota error, as
wrong statuses. Measured against the public set's own `status` field on the 25
real answers, status was **25 of 25 correct**. Over-abstention is therefore not
the demonstrated problem it was recorded as -- but it is also untested, because
all six public questions that are not plain `answered` (four `not_in_corpus`,
two `partial`) fall in the half that has not been run.

Two of those four `not_in_corpus` questions carry **gold documents**
(r2-pub-046, r2-pub-049): the correct answer cites the documents that establish
the gap. The composer is now told so explicitly, and the audit no longer treats
`not_in_corpus` with citations as a defect -- a check that would have pushed the
system toward citing nothing on the questions where citations are worth most.

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
