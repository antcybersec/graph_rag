# The judging call: a runbook

Live judging is 15% of the Round 2 score and it is the one part that no amount
of code earns in advance. This is the order to do things in, what each command
proves, and how long it actually takes — every timing below was measured on
this machine on 26 Nov, not estimated.

The single most important line: **start Docker and the container before the
call, not during it.**

---

## Before the call

```bash
open -a "Docker Desktop"        # ~20s for the daemon
docker start tigergraph         # ~38s until gadmin reports Online
python -m src.r2.counts         # ~1s -- proves the graph is loaded, not empty
```

Measured: daemon 20s, TigerGraph **38s** to Online. Counts read during warm-up
are stale, which has sent people hunting a bug that was not there, so wait for
Online rather than for the port.

Docker Desktop on this machine has twice needed `pkill -f com.docker` and a
relaunch after being quit, and `docker info` **hangs** rather than failing when
it is wedged. If it hangs for more than a minute, kill and relaunch.

Then check the day's quota before you need it:

```bash
python -c 'from pydantic import BaseModel
from src.common.llm import generate
class T(BaseModel):
    ok: str
print(generate("say ok", response_schema=T, thinking_budget=0)[0])'
```

If that 429s, the live-question path is the offline one below and you should
say so early rather than discover it mid-answer. **One pool of 500 requests a
day covers answering and scoring both** (measured 26 Nov: a `-preview` call
429s blaming `gemini-3.1-flash-lite`, so the judge model shares the budget).
Resets midnight Pacific = 13:30 IST.

---

## The four things that cost nothing and prove the most

Run these first. They need no model, no key and no quota, each returns in about
a second, and between them they cover repeatability, honesty and citations.

| command | time | what it proves |
|---|---|---|
| `python -m tests.test_r2` | **1s** | 100 checks on the parts that decide accuracy, honesty and repeatability |
| `python -m src.r2.counts` | **1s** | 30,800 vertices and 48,900 edges by type — deliverable 6 |
| `python -m src.r2.audit data/results/r2/answers_hidden.jsonl` | **<1s** | every answer held against the arithmetic its own record carries |
| `python -m src.r2.check <questions> <answers>` | **<1s** | **exits 0 on both halves** — one answer per question, no placeholders, every trace step in the query log |

`check` is the one to run in front of a judge, because it is the gate and it is
the deliverable they can run themselves. Run it on **both** files and show both
exit codes. Do not pipe it through `tail` — the problem list prints *above* the
summary, and reading it through `tail -3` once produced a confident report of a
pass on a file that was exiting 1.

---

## "Ask it something new"

This is what rule 4 exists for: the question comes in as an argument, there is
no question id anywhere in the path, and the same code answers an unseen
question and a public one.

```bash
python -m src.r2.ask "<the judge's question>"
python -m src.r2.ask --json "<the judge's question>"   # the record, exactly as the answer files carry it
```

Measured on Gemini: median **16–30s** per question, and the p95 is **137–167s**
— a slow one is a slow one, so say "this takes up to two minutes" before
starting rather than watching silence.

**If the quota is gone**, the whole system runs on a local model with no key:

```bash
./offline.sh "<the judge's question>"
```

Measured **241s** — about four minutes. Warn the judge before you press return.
It is slow for a deliberate reason worth saying out loud: `OLLAMA_KEEP_ALIVE=0`
unloads the weights after each call, because ~6GB of idle weights beside
TigerGraph's 5.6GB OOM-killed the container on this 16GB machine, and a dead
graph mid-demo is worse than a slow answer.

Two honest caveats to volunteer rather than be caught on: the local model plans
correctly on 4 of 5 probe questions where Gemini gets 5, and its citation
behaviour differs from the submitted configuration. Use it to show the system
still answers, not to make claims about citation precision.

**The best thing in the demo is here.** Asked which nation won the most
wrestling gold at 2024, Gemini and the local model both answer *Japan, 8* — and
the local model's own explanation says it read the figure from
`ARITHMETIC ALREADY DONE FOR YOU`, the computed block. Swap the entire language
model and the number does not move, because the number was never the model's
job. `computed` is present on 33 of 50 public answers and 30 of 50 hidden ones.

---

## "Show me it reproduces"

```bash
python -m src.r2.repro <questions> data/results/r2/answers_public.jsonl --sample 3
```

Let the judge pick the questions with `--qid`, or let it sample at random —
random is the more convincing of the two and it is the default.

What it reports, and what to say about it. **Rehearsed on the submitted code
on 28 Nov, nine random questions returned 7 MATCH, 2 CLOSE, 0 DIFFERENT, with
retrieval identical on every one of them.** Two dates of history are worth
having ready, because the direction is the point: before 27 Nov no rehearsal
had ever returned a single MATCH, on 27 Nov it was 5 MATCH / 3 CLOSE / 1
DIFFERENT, and it is now 7 / 2 / 0.

Read the three verdicts out rather than the headline, because the detail is
better than the summary:

- The **2 CLOSE** are the same thing in both cases: the same figures restated
  differently — one re-run wrote the "100" of a weight category into the figure
  list and the submitted answer had not. Status, citation counts and retrieval
  identical.
- There is **no DIFFERENT** in this sample. The one the 27 Nov rehearsal found
  was prose variation with all four graded dimensions holding, so it is still
  worth describing if they ask what a DIFFERENT looks like here.
- The **7 MATCH** are the argument. Figures, statuses, citation counts and
  retrieval step counts all reproduce exactly; only the wording moves, and on
  one of them the wording is only 47% similar while every graded field is
  identical. That gap is the whole design in one line: the prose is generated
  and varies, and nothing graded depends on the prose.
- Reach for r2-018 and r2-019 if they want a mechanism. They are the questions
  the season fix repaired. Before it,
  the Winter Games were queried as Summer, nothing matched, and the composer
  counted medals out of the article prose — a number a language model produced,
  which is a number that moves between runs. Now the figure is read from
  `gold medals by nation ... highest 6 ['FRA']`, computed in code from graph
  rows. **The arithmetic rule is not only an accuracy argument, it is a
  repeatability argument**: every figure moved out of the model and into code
  is a figure that reproduces exactly. That is the sentence to say on this
  slide.

**"Why not just set the temperature to zero?"** — expect this one, and it is a
good question with a measured answer. Until 28 Nov no temperature was set
anywhere, so every call ran at the provider default of 1.0; say that plainly
first, because it is the sort of thing a reviewer finds with one grep and it
looks careless. Then give the measurement: it was set to 0 and the same five
questions were run twice with identical inputs, and **one of five came back
byte-identical while two statuses flipped between the two runs**. It was
reverted.

The reason is the useful part. Answering here is a cascade, not one call — the
planner picks a route, the route queries the graph, some routes call the model
again to choose a source document, and the composer only ever sees what that
produced. Greedy decoding at the last step does nothing about a different
choice at the first, and a different choice at the first means *different
evidence*. The variance is in what gets retrieved, which is the one decision
the model is meant to own.

So this lands exactly where the design already stood: the figures, statuses and
citations reproduce because they are computed in code rather than sampled,
which is why one of the 7 MATCH holds every graded field identical while its
prose is only 47% similar. There is also a consistency reason not to pin it
now — the submitted answers were generated at the default, so code fixing
temperature at 0 would no longer reproduce its own artifacts, and a judge
running the repository would get a third thing. `docs/round2-risks.md` 2r.

Still do not oversell it. Nine is a small sample and a judge who re-runs a
different nine may see a different split; the honest framing is that no graded
field moved in this one, not that nothing ever moves. We have the number for
that too — across the full 100-question re-run on 28 Nov, seven answers changed
in some way, and `docs/round2-risks.md` 2q lists every one. If their bar is
byte equality, say plainly that we do not meet it, that the composed sentence
is generated prose and cannot be, and that what holds still is the status, the
figures, the citations and the route.

---

## The questions worth having an answer ready for

**"How do I know this isn't tuned to the 50 public questions?"**
No component in the answering path reads the wording of a question. The model
interprets it and may only emit keys the graph just handed it. Two files in the
repository do pattern-match on question text; both are Round 1, both carry a
header saying they are not in the Round 2 path, and the README names all three
itself rather than leaving them to be found by grep. Offer the grep.

**"What is it worst at?"** — answer this one properly, it is the question that
builds the most credit, and the best version of the answer is the one where we
found the defect, measured it, fixed it and can say what the fix cost.

The 27 Nov submission scored 92.3%, which was 3.84 points lost over 6 of 50
questions. **Four of the six were a single defect worth 2.0 of those points —
52% of everything lost — and they failed in opposite directions**:

| qid | key says | the 27 Nov answer | now |
|---|---|---|---|
| r2-pub-050 | answered | `not_in_corpus` | **fixed** |
| r2-pub-042 | answered | "no evidence" for a venue the corpus documents | **fixed** |
| r2-pub-048 | `partial`, status null | answered, having *invented* an official record | status now `partial` |
| r2-pub-049 | `not_in_corpus` | answered | still wrong |

Say the opposite directions out loud, because it is what identified the cause:
no instruction to be bolder or more cautious could fix this, it would trade one
pair for the other. That is the signature of a rule that disagrees with itself,
not of a model that is miscalibrated.

The current number is **93.9%, with status correct on 49 of 50** — up from 46 —
and r2-pub-049 is the only question in the public set still scoring zero. It
asks for an officially confirmed duration that the corpus never confirms, and
we answer it instead of declining. That is the honest answer to "what is it
worst at" today: it is still, specifically, reluctant to say that a value does
not exist.

Volunteer that `audit.py` catches it unprompted — "status is answered but the
answer reads like a refusal" — and that a fix which acts on that flag instead
of printing it sits on `determinism-and-refusal-gate`, gated at 0.00 -> 1.00 on
that question with all three controls holding. It is not merged because the
same gate took r2-pub-048 from 0.50 to 0.00: 048's answer reports what a
document says, "the official record states that the duration could not be
established", and an attributed statement is content rather than a refusal,
which a leading-clause test cannot distinguish. Promoting it needs another
~291-call run across both halves on the submission date, with no second run to
recover. `docs/round2-risks.md` 2s. A fix measured, costed and declined is a
better answer here than a fix shipped on the last day.

**The cause is a contradiction in our own prompt, and it is worth showing them
the two passages.** Four lines apart in `COMPOSER_SYSTEM`, the `partial` clause
routed "the corpus records rival claims but confirms none of them" to
`partial`, and the fourth of the four shapes routed the same situation to
`not_in_corpus`. The key agrees with the first. A model handed both rules picks
one per question, which is precisely a failure in both directions, and no
amount of calibration advice fixes a rule that disagrees with itself.

Be straight about the first answer we gave to this, because the correction is
the more useful half. We first said the fix was to bind status to
`coverage_check`, which is called on every question and already knows whether a
matching document exists. Then we replayed that rule over all 100 committed
answers — the row count of every `coverage_check` is in the trace, so it cost
no quota — and it would have fired **zero** times. A plausible mechanism,
asserted without the free check that falsifies it. If they ask how we avoid
fooling ourselves, this is the example to give.

The real fix is one edit — remove the contradiction and let the `partial`
clause govern. It was gated on 27 Nov over 4 failing questions and 3 controls
for 24 calls and measured +1.60 with no regressions, then promoted on 28 Nov
across both halves from one code version. **Two more elaborate versions of it
measured worse**, and that is the part worth volunteering: splitting the rule
along value-versus-judgement scored +1.00, and adding a supersession rule
scored +0.50 while breaking a control from 1.00 to 0.00. Ranked on the four
failing questions alone, that worst version would have looked like the best
one. The controls are the only reason we know otherwise, and they were in the
gate from the start.

Be equally straight about what the promotion cost, because they may well find
it. Five public answers moved and two moved down — r2-pub-044 dropped a date
it had previously given, and r2-pub-047 attributed a count to a source the key
does not support. Two hidden answers changed status in a direction we think is
worse, and `docs/round2-risks.md` 2q names both. Three of those four do not go
through the passage that changed at all; they are the re-run's nondeterminism,
about one status flip in nine, which lands on any re-run whatever the code
says. The +1.6 points is net of all of it. `2p` has the gate tables, `2q` the
promotion.

Volunteer r2-pub-048 specifically: the system invented an official record to
justify a status, which is the worst failure available to something graded on
honesty, and we found it ourselves and wrote it down.
`docs/round2-risks.md` 2j has the diagnosis, down to which of 39 source records
the selector picked and why — the heading matched and the body did not. The fix
was written, measured on 27 Nov, and **rejected**: it changed nothing on the
question it was built for. Risk 2l is that measurement. Volunteer this; a
diagnosis that survived and a fix that did not is a better answer than a fix
that shipped unmeasured.

**"Is the accuracy number yours or a real one?"**
93.9% on the 50 public questions, graded fact by fact against the published key
by `gemini-3.1-flash-lite-preview`, a different model from the one that wrote
the answers — `metrics.json` records `judge_independent` and `score.py` prints
both model names in its header. It is still our own harness, so treat it as an
estimate. Say so first; it is the difference between a measurement and a claim.
The hidden half has no accuracy number because the key is theirs.

**"Show me a citation is real."**
426 citations across both files, 0 that are not genuine corpus documents, and 0
answered-or-partial answers with no citation at all. Public recall 0.900,
precision 0.945.

**"How do you know your changes help?"** — the best question to be asked, and
there is a straight answer. Three fixes were ready on 27 Nov, each with a
correct diagnosis and passing tests. Each was gated on the questions whose
traces touch its code path, found by grepping the committed answer files for
the query name, and each gate cost under 40 model calls against a ~300-call
paired run.

| change | gate | verdict |
|---|---|---|
| source selection | 9 public evidence questions | **rejected** — score rose 6.5 → 7.3 and the target question selected the same wrong record byte for byte, so the rise was noise |
| years in office | 4 questions calling `office_chain` | **rejected** — correct arithmetic, verified against all 73 officeholders, that turned a correct hidden answer into a wrong one |
| season from the graph | 0 public / 2 hidden, measured before writing it | **merged** |

The second one is the one to tell. It computes years in office correctly and
still made the system worse: handed totals for five unnamed infobox blocks, the
composer attached "his first presidency" to an 8.88-year block that was a
different office, having previously named the right 26-day term. More correct
arithmetic in front of a composer is not automatically an improvement, because
every figure added is a figure it can attach to the wrong subject.

Both rejected branches are pushed, with the measurement in the commit message.

**"What did you get wrong and fix?"**
`docs/round2-risks.md` is the honest list and it is better material than any
feature. The strongest items: a 503 outage that wrote `not_in_corpus` for 33
hidden questions — a claim about the corpus, made because the model was
unreachable; a judge that was grading its own answers because `score.py` read
`JUDGE_MODEL` before `load_dotenv()`; and a run that was reported as passing
`check.py` from output that had been truncated by `tail -3`. All three are
fixed, tested, and written down with the mechanism.

---

## What not to do

- Do not run `finish.sh` during the call. It waits for quota and then spends a
  day's worth.
- Do not re-run the submitted pair to "show it working". It costs ~250 requests
  and overwrites the artifacts that pass.
- Do not quit Docker Desktop and expect to restart it quickly. See above.
- Do not claim reproduction is clean. It returned 7 MATCH of 9 on 28 Nov with
  no graded field moving, which is a good result on a small sample and not a
  guarantee; quote the split and the sample size, not the best verdict in it.
  If anything in the repository still says a rehearsal has never returned
  MATCH, or still quotes 5 of 9, that line is older than the measurement.
