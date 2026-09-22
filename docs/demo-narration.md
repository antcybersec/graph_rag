# Demo narration — Anant's script

Read the quoted lines out loud. Italics are what you do, not what you say.
One browser window, the dashboard running locally:
`streamlit run src/dashboard/app.py`

**Before recording:** run it once and click **Investigate** once so nothing
stalls on camera. Keep `.env` closed. Full screen. About three minutes.

---

### 0:00 *(top of the page)*

> "Hi, I'm Anant. I built an agentic GraphRAG system on TigerGraph, and what
> you're looking at is its benchmark. Six pipelines, one corpus of about three
> thousand Wikipedia articles, and the same hundred questions for all of them.
> I built it this way because I wanted to answer the question this hackathon
> actually asks — when do you need an agent, and when is one just overkill?"

### 0:20 *(Summary tiles)*

> "Here's the scoreboard. Plain RAG gets sixty-seven percent. GraphRAG, using
> an entity graph, also gets sixty-seven. My agentic version gets seventy. The
> three on the right all hit ninety-nine."

> "I'm scoring on exact match against the gold answer, with no model in the
> loop, because I stopped trusting the LLM judge — and I'll show you why in a
> moment."

### 0:40 *(scroll to **Quality by question type**)*

> "The average hides the real story, so I broke it out by question type. Look
> at aggregation — questions like 'how many cycling events had more than thirty
> competitors'. RAG gets one out of twenty-one. GraphRAG gets zero. My agent
> manages three. The graph pipelines get all twenty-one."

> "That gap isn't a tuning problem, it's structural. To answer that question
> you need *every* matching document — for one of them, forty-three. Retrieval
> fetches the top five and counts those. No prompt fixes it."

### 1:05 *(stay here)*

> "So I built a second layer in the graph, straight from the infoboxes at the
> top of each article — every Olympic event linked to its Games, its sport, its
> venue, and to the Games before it. That ingestion doesn't call a language
> model at all. Counting stops being retrieval and becomes a graph traversal."

### 1:25 *(**Per-question drill-down**, select **pub-052**)*

> "Let me show you one question across all six. This asks how many nations
> competed in the 2022 biathlon women's relay."

*(point at the GraphRAG column)*

> "GraphRAG says the documents don't contain that information. It's wrong — and
> look at the judge score next to it. The judge gave that answer top marks.
> That's exactly why exact match is my headline number, and why I built a
> separate verification pass that catches all fourteen answers the judge
> wrongly approved."

*(open **Investigation** under Investigator)*

> "Now here's my agent on the same question. It ran a structured graph query
> first, the evidence came back short, so it switched to vector search on its
> own and answered from there. Three steps — and it stopped because it decided
> the evidence was sufficient, not because it ran out of budget."

### 1:55 *(open **Evidence & provenance**)*

> "Every piece of evidence carries where it came from — the source document,
> its URL, and the exact query that produced it. So my answers aren't just
> cited, you can check them."

### 2:10 *(scroll to **Cost**)*

> "That flexibility costs something, and I want to be straight about it. The
> agent runs about twice the calls of a fixed route, and takes thirteen seconds
> a question."

### 2:20 *(scroll to **Live investigation**, pipeline on **Jev planner**, click **Investigate**)*

> "Which brings me to what I think is the interesting finding. For these
> questions, my agent is overkill. Everything the planner has to decide already
> exists as a row in the graph — the sports, the Games, the venues, the event
> names. So instead of generating a plan, it selects one."

*(the answer appears)*

> "Two typed selections, then a graph query. Same answer, under two seconds.
> And look at the counter — **zero generation-model calls.**"

### 2:40 *(switch the radio to **Investigator**, ask "Who directed the film Jab We Met?")*

> "That's not just a speed trick. My free tier is five hundred calls a day, which
> is one benchmark run, and this path can't hit a rate limit — so a demo can't
> die halfway through. But the agent still earns its place on anything the graph
> doesn't model. Ask it who directed a film and it switches to vector search by
> itself."

### 2:55 *(close)*

> "So that's my answer. Agents earn their cost on open-ended questions, and
> they're overkill on questions a typed query can settle — and I can now show
> you which is which. Fifty out of fifty hidden questions answered, and
> everything here reproduces from the repo. Thanks for watching."

---

## The 60-second cut

Three beats: the **aggregation column** (0:40), the **judge scoring a wrong
answer 5** plus your agent changing strategy on pub-052 (1:25), and the
**zero generation-model calls** counter (2:20).

## If a judge asks

**"Why is Jev planner 'not judged'?"** — The judge is itself an LLM call. The
whole point of that pipeline is not needing one, so scoring it with the thing
it avoids would be incoherent. It's scored on exact match, like everything else.

**"What are the 579 error rows the caption mentions?"** — Retries. The free
tier is 500 calls a day and long runs hit it; the benchmark checkpoints per
question and resumes, so failed attempts stay in the file as history. 600
scored rows is six pipelines times a hundred questions.

**"Isn't 99% just overfitting to templated questions?"** — I tested that. Ten
questions rewritten to avoid the eval's wording, using host cities like
"Sydney 2000" that never appear in it: ten out of ten. The host-city-to-Games
mapping is the planner's inference, not a regex.

**"Which pipeline is the submission?"** — The Investigator is the agentic
system. The Jev planner is the finding about when you don't need it.

## Numbers

| | RAG | GraphRAG | Agentic | Event-graph | Investigator | Jev |
|---|---|---|---|---|---|---|
| Exact match | 67% | 67% | 70% | 99% | 99% | 99% |
| Aggregation | 1/21 | 0/21 | 3/21 | 21/21 | 21/21 | 21/21 |
| Tokens/q | 3,586 | 3,952 | 6,065 | 619 | 3,412 | 2,267 |
| Calls/q | 2.0 | 2.0 | 4.1 | 1.0 | 2.1 | 1.4 |
| Seconds | 6.8 | 10.2 | 20.2 | 8.9 | 13.2 | **1.7** |

Hidden set: 50/50 answered, 49/49 correct against a deterministic oracle.
