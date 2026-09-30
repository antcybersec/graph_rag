# Social posts

## LinkedIn

I spent two weeks answering one question for the @TigerGraph Agentic GraphRAG
Hackathon: when does an AI agent actually earn its cost?

I built six retrieval pipelines over the same 2,951-document corpus, ran all of
them on the same 100 questions, with the same model throughout. Exact match went
from 67% to 99%.

But the number I keep thinking about is this one: the agentic architecture
*alone* only bought 3 of those 32 points. The other 29 came from giving the
agent the right thing to query.

On "how many events had more than 30 competitors", plain RAG scored 1 out of 21.
Not because the model is weak — because answering it needs all 43 matching
documents and retrieval hands you five. So I built a graph layer straight from
the articles' infoboxes, with no LLM in the ingestion at all. Counting became a
traversal: 21 out of 21.

Then the twist. If the agent picks the structured query every time, does it need
to reason? Everything it decides already exists as a row in the graph. So I
swapped the generative planner for typed selection: same 99%, 1.7s instead of
13s, and zero generation-model calls — which means a live demo can't die on a
rate limit.

Agents earn their cost on open-ended questions. They're overkill where a typed
query settles it. The agent still ships, because you don't know which kind of
question is coming next.

Also in the write-up: the 14 answers my own LLM judge scored 4-5 that were
wrong, the "fix" of mine that cost 17 points, and the stale artifact that nearly
shipped with five wrong answers.

Repo: https://github.com/antcybersec/graph_rag
Dashboard: https://graphrag-c.streamlit.app/

#TigerGraph #GraphRAG #AgenticAI #KnowledgeGraph

## X / Twitter

Built 6 retrieval pipelines for the @TigerGraph Agentic GraphRAG hackathon, same
corpus, same model, same 100 questions.

67% → 99% exact match.

The agent alone bought 3 of those 32 points. The other 29 came from giving it
the right thing to query.

Then: replacing the generative planner with typed selection held 99% at 1.7s
with ZERO generation-model calls. A demo that can't hit a rate limit.

Agents earn their cost on open-ended questions. They're overkill where a typed
query settles it.

Repo + dashboard 👇
https://github.com/antcybersec/graph_rag
