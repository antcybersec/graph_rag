# Demo narration — read this out loud

Everything here is on one screen: the dashboard, run locally.
`streamlit run src/dashboard/app.py`

**Before you hit record**
- Run it once and click **Investigate** once — that warms the connection so nothing stalls on camera.
- Don't open `.env`, and don't screen-share a terminal where it's visible.
- Full screen, and scroll slowly. Roughly 3 minutes at a normal speaking pace.

---

### 0:00 — Opening *(top of the page)*

> "This is an agentic GraphRAG system built on TigerGraph. It answers questions
> over about three thousand Wikipedia articles, and it benchmarks six pipelines
> against the same hundred questions. That lets us answer the question this
> hackathon actually asks: when do you need an agent, and when is one overkill?"

### 0:15 — The scoreboard *(Summary tiles)*

> "Here are the six pipelines. Plain RAG gets sixty-seven out of a hundred.
> GraphRAG — entity linking plus one-hop traversal — also gets sixty-seven.
> The agentic version gets seventy. The three on the right all reach
> ninety-nine."

### 0:35 — Where retrieval breaks *(scroll to **Quality by question type**)*

> "This chart shows where that gap comes from. Look at the aggregation column —
> questions like 'how many cycling events had more than thirty competitors'.
> RAG gets one out of twenty-one. GraphRAG gets zero. The agent manages three.
> Our graph pipelines get all twenty-one."

> "The reason is structural, not a tuning problem. Answering that needs *every*
> matching document — for one question, forty-three of them. Retrieval fetches
> the top five and counts those. No prompt fixes that."

### 1:00 — What we built instead *(stay on the chart, or open the architecture diagram)*

> "So we built a second layer in the graph, straight from the infoboxes in the
> articles: every Olympic event linked to its Games, its sport, its venue, and
> to the Games before it. That ingestion uses no language model at all.
> Counting stops being a retrieval problem and becomes a graph traversal."

### 1:20 — The agent working *(**Per-question drill-down** → pick `pub-023`, open **Investigation**)*

> "This is the agent itself. It plans each step and picks between five tools.
> On this question it ran a structured graph query first, the evidence came
> back short, so it switched to vector search on its own and answered from
> there. Three steps — and it stopped because it judged the evidence
> sufficient, not because it ran out of budget."

### 1:45 — Evidence you can check *(open **Evidence & provenance**)*

> "Every piece of evidence carries where it came from: the source document, its
> URL, and the exact query that produced it. So the answer isn't just cited,
> it's checkable — you can click through and verify it yourself."

### 2:00 — What agentic steps cost *(scroll to **Cost**)*

> "That flexibility isn't free. The agent uses about twice the calls of a fixed
> route, and around thirteen seconds a question."

### 2:10 — The finding *(scroll to **Live investigation**, pipeline set to **Jev planner**, click **Investigate**)*

> "Which brings me to the interesting result. For these questions, the agent is
> overkill. Everything the planner has to decide already exists as a row in the
> graph — the sports, the Games, the venues, the event names. So instead of
> generating a plan, it selects one."

*(the answer appears)*

> "Two typed selections, then a graph query. Same answer, under two seconds.
> And look at the counter: **zero generation-model calls.**"

### 2:35 — Why that matters *(switch the radio to **Investigator**, run the film question)*

> "That's more than a speed trick. The free tier here is five hundred calls a
> day — one benchmark run — and this path can't hit a rate limit, so a demo
> can't die on stage. But the agent is still needed for anything the graph
> doesn't model. Ask it who directed a film, and it switches to vector search
> by itself."

### 2:50 — Close

> "So that's the answer: agents earn their cost on open-ended questions, and
> they're overkill on questions a typed query can settle — and now we can show
> which is which. Fifty out of fifty hidden questions answered, and everything
> here is reproducible from the repo."

---

## If you only have 60 seconds

Three beats: the **aggregation column** (0:35), the agent **switching tools**
(1:20), and the **zero generation-model calls** counter (2:10). Those carry the
whole argument.

## Numbers you might be asked about

| | RAG | GraphRAG | Agentic | Event-graph | Investigator | Jev planner |
|---|---|---|---|---|---|---|
| Exact match | 67 | 67 | 70 | 99 | 99 | 99 |
| Aggregation | 1/21 | 0/21 | 3/21 | 21/21 | 21/21 | 21/21 |
| Tokens/question | 3,586 | 3,952 | 6,065 | 619 | 3,412 | 2,267 |
| LLM calls | 2.0 | 2.0 | 4.1 | 1.0 | 2.1 | 1.4 |
| Seconds | 6.8 | 10.2 | 20.2 | 8.9 | 13.2 | **1.7** |

Hidden set: 50/50 answered, 49/49 correct on every question a deterministic
oracle can check. Evidence verification catches 14 of 14 answers our own LLM
judge wrongly approved.
