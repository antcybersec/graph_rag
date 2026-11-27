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

What it reports, and what to say about it: the latest rehearsal returned
**0 MATCH, 3 CLOSE, 0 DIFFERENT**, with status and figures identical on all
three. Do not oversell this. Every rehearsal so far has returned CLOSE rather
than MATCH, because retrieval moves between runs — 4→5, 9→7, 16→12 steps — and
one citation set went 18→8. That is the model choosing what to retrieve, which
is the one decision it is supposed to own; the status, the figures and the
arithmetic are what hold still. If a judge's bar is byte equality, say plainly
that we do not meet it and why the composed sentence cannot be byte-identical.

---

## The questions worth having an answer ready for

**"How do I know this isn't tuned to the 50 public questions?"**
No component in the answering path reads the wording of a question. The model
interprets it and may only emit keys the graph just handed it. Two files in the
repository do pattern-match on question text; both are Round 1, both carry a
header saying they are not in the Round 2 path, and the README names all three
itself rather than leaving them to be found by grep. Offer the grep.

**"What is it worst at?"** — answer this one properly, it is the question that
builds the most credit. Three public questions turn on what the absence of a
record means, and the system is wrong in both directions: it asserted a
stoppage occurred where the only official check could neither confirm nor rule
it out, and it declined a question the corpus answers in one sentence.
`docs/round2-risks.md` 2j has the diagnosis, down to which of 39 source records
the selector picked and why — the heading matched and the body did not. The fix
was written, measured on 27 Nov, and **rejected**: it changed nothing on the
question it was built for. Risk 2l is that measurement. Volunteer this; a
diagnosis that survived and a fix that did not is a better answer than a fix
that shipped unmeasured.

**"Is the accuracy number yours or a real one?"**
92.3% on the 50 public questions, graded fact by fact against the published key
by `gemini-3.1-flash-lite-preview`, a different model from the one that wrote
the answers — `metrics.json` records `judge_independent` and `score.py` prints
both model names in its header. It is still our own harness, so treat it as an
estimate. Say so first; it is the difference between a measurement and a claim.
The hidden half has no accuracy number because the key is theirs.

**"Show me a citation is real."**
400 citations across both files, 0 that are not genuine corpus documents, and 0
answered-or-partial answers with no citation at all. Public recall 0.891,
precision 0.914.

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
- Do not claim MATCH on reproduction. It has never returned MATCH.
