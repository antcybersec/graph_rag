"""A single explanatory page for the judging call.

The job of this screen is not to look like a product. It is to make four
things obvious to someone seeing the system for the first time, in the order
they will ask about them:

  1. ask it something and watch what it does
  2. what it just did, step by step, with the database calls named
  3. what the graph actually holds
  4. what it scored on the public set, including where it was wrong

Everything on the page is read live from TigerGraph or from the submitted
answer files. Nothing is typed in by hand, so nothing here can drift from
what the system really does -- if the graph is empty, this page says so
rather than showing a number from last week.

    streamlit run src/r2/ui.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import streamlit as st

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

RESULTS = Path("data/results/r2")

# One line per installed query, in plain words. The GSQL itself carries the
# reasoning; this is the version you can read out loud.
TOOLS = {
    "list_sports": "every sport in the corpus -- the list the planner may choose from",
    "list_games": "every Games in the corpus, so a year is never assumed",
    "list_sources": "the 39 LA 2028 source documents, with who issued each and when",
    "series_by_sport": "the contests held in a sport, across all editions",
    "coverage_check": "does a document for this exist at all? the honesty primitive",
    "event_lookup": "one event's fields and every medal awarded in it",
    "events_in_sport_games": "every event of one sport at one Games, with participation",
    "series_trend": "one contest across its editions -- a traversal, not a scan",
    "series_medals": "every edition of one contest with its medallists, in one call",
    "neighbouring_edition": "the editions either side of this one, as the corpus states them",
    "medals_by_nation": "medal counts by nation for a sport at one Games",
    "nation_medals_detail": "which events one nation won, and with whom",
    "medals_by_athlete_at_games": "medal counts per athlete at one Games",
    "athlete_career": "every individual medal one athlete won, across all Games",
    "team_medals_for_name": "medal lines naming a whole squad, searched by name",
    "repeat_champions": "athletes who won the same event at more than one Games",
    "athlete_search": "a name in a question becomes an athlete key in the graph",
    "person_search": "the same, for directors and politicians",
    "film_search": "a film by title, with its directors and money figures",
    "films_by_director": "every film one director made",
    "office_chain": "offices held, with predecessor and successor on each term",
    "claims_for_event": "every claim grounded to an event, winning and losing",
    "claims_by_subject": "claims whose subject could not be pinned to an event",
    "claims_search": "claims mentioning a name, anywhere in the assertion",
    "claims_by_slot": "every claim competing over the same fact -- slot completion",
    "claims_by_source": "everything one named document asserts",
    "sources_for_event": "the source documents behind an event's claims",
    "vector_search_chunks": "nearest passages by meaning, over 768-dim chunk vectors",
    "chunks_containing": "exact substring search inside article text",
    "chunks_of_document": "one document's prose, in order -- retrieval scoped by the graph",
    "doc_for_chunk": "a retrieved passage back to the document that can be cited",
}

STAGES = [
    ("Plan", "One model call reads the question and picks a route and the entities it "
             "mentions. It is given the catalogue of sports and Games from the graph and "
             "may only answer with keys from it."),
    ("Resolve", "The names in the question become graph keys -- but the candidates come "
                "from the database, and the model picks from them. A contest it cannot "
                "find among the candidates stays unfound, rather than being forced onto "
                "the nearest match."),
    ("Check coverage", "The graph is asked whether a document for what was pinned down "
                       "exists at all. Zero is an answer: it is the difference between "
                       "\"the corpus does not record this\" and \"I am not sure\"."),
    ("Gather", "The route runs its installed queries -- medals, editions, careers, "
               "claims, prose. Conflicting source records are expanded to every claim "
               "competing over the same fact before any of them is weighed."),
    ("Answer", "One model call writes the answer from the gathered evidence only, and "
               "picks its citations from the documents that were actually retrieved. A "
               "deterministic gate then drops any citation that was not in the evidence, "
               "and refuses to let an answer stand with nothing behind it."),
]


@st.cache_resource(show_spinner=False)
def get_agent():
    from src.r2.agent import Agent
    from src.r2.conn import describe, get_connection
    from src.r2.trace import QueryLog
    conn = get_connection()
    log = QueryLog(RESULTS / "ui_querylog.jsonl")
    return Agent(conn, query_log=log), conn, describe()


@st.cache_data(ttl=120, show_spinner=False)
def graph_counts():
    _, conn, _ = get_agent()
    return conn.getVertexCount("*"), conn.getEdgeCount("*")


def load_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


st.set_page_config(page_title="Agentic GraphRAG - Round 2", layout="wide")

st.title("Agentic GraphRAG over TigerGraph")
st.markdown(
    "**One question at a time, answered from a graph of 3,530 documents -- and an "
    "honest answer when the corpus does not contain one.**  \n"
    "The corpus is the only source of truth here. Where it is silent, this system is "
    "built to say so rather than to produce something plausible."
)

section = st.sidebar.radio(
    "  ",
    ["Ask it something", "How it works", "The graph", "The agent's tools", "Public results"],
    label_visibility="collapsed")
st.sidebar.markdown("---")
try:
    _, _, where = get_agent()
    st.sidebar.success(f"TigerGraph: {where}")
except Exception as exc:  # the page must say why, not show a stack trace
    st.sidebar.error(f"TigerGraph unreachable\n\n{exc}")

# ---------------------------------------------------------------- ask it ----
if section == "Ask it something":
    st.markdown("### Ask it something")
    st.caption("Type any question about the corpus -- Olympic events 1984-2028, films, "
               "US politicians, or the LA 2028 source records that disagree with each "
               "other. Nothing is cached: every answer below is produced now.")

    question = st.text_input("Question", placeholder="Which nation won the most rowing "
                                                     "golds at the 2024 Summer Olympics?",
                             label_visibility="collapsed")
    go = st.button("Answer it", type="primary")

    if go and question.strip():
        agent, _, _ = get_agent()
        with st.spinner("planning, querying the graph, composing..."):
            t0 = time.time()
            record = agent.answer(f"ui-{int(t0)}", question.strip())

        colour = {"answered": "green", "partial": "orange", "not_in_corpus": "red"}
        st.markdown(f":{colour.get(record['status'], 'grey')}[**{record['status'].upper()}**]"
                    f" &nbsp; *route chosen by the planner: `{record.get('route')}`*")
        st.markdown(f"## {record['answer']}")
        st.markdown(f"**How it got there.** {record['explanation']}")

        if record["citations"]:
            st.markdown("**Citations** (the documents that support it):")
            st.code("\n".join(record["citations"]), language=None)
        else:
            st.info("No citations -- which is the honest outcome when the corpus holds "
                    "nothing that supports an answer.")

        a, b, c, d = st.columns(4)
        a.metric("Latency", f"{record['latency_ms'] / 1000:.1f}s")
        b.metric("Tokens in / out", f"{record['input_tokens']:,} / {record['output_tokens']:,}")
        c.metric("TigerGraph queries",
                 sum(1 for s in record["trace"] if s["tigergraph"]))
        d.metric("Model calls",
                 sum(1 for s in record["trace"] if not s["tigergraph"] and "input_tokens" in s))

        st.markdown("#### Every step it took")
        st.caption("Written as it ran, not reconstructed afterwards. The rows marked "
                   "TigerGraph are real queries against the database -- they appear in "
                   "the query log with the same timestamps.")
        st.dataframe(
            [{"#": s["step"], "step": s["tool"],
              "TigerGraph": "yes" if s["tigergraph"] else "",
              "installed query": s.get("query", ""),
              "parameters": json.dumps(s.get("params", ""), ensure_ascii=False)[:120]
                            if s.get("params") else "",
              "rows": s.get("rows", ""),
              "tokens in": s.get("input_tokens", ""),
              "tokens out": s.get("output_tokens", ""),
              "ms": s["ms"]} for s in record["trace"]],
            width="stretch", hide_index=True)

        with st.expander("The submission record for this question (exact JSON)"):
            st.json(record)

# ------------------------------------------------------------- how it works --
elif section == "How it works":
    st.markdown("### How a question becomes an answer")
    st.caption("Five stages. The model decides *what to look up*; the graph decides "
               "*what exists*.")
    for i, (name, text) in enumerate(STAGES, 1):
        st.markdown(f"**{i}. {name}** — {text}")
    st.markdown("---")

    left, right = st.columns(2)
    with left:
        st.markdown("#### How documents became graph data")
        st.markdown(
            "- **3,530 documents** parsed deterministically -- no model, no extraction "
            "errors. The infobox at the head of each article is a table in disguise, so "
            "it is read as one: medallists, nations, competitors, venue, dates.\n"
            "- **Events** join to Games, Sport, Venue and an **EventSeries**, which "
            "exists so that \"this contest across the years\" is one hop instead of a "
            "scan of 2,750 vertices.\n"
            "- **Two medal edges, not one.** A medal line can name a whole squad: the "
            "nation is always known, the individual is not. `MEDAL_FOR` carries every "
            "line by nation, `WON` carries the ones naming a single athlete.\n"
            "- **Prose** is chunked and embedded locally (768-dim BGE) into "
            "TigerGraph's vector index, because some answers exist only in sentences.\n"
            "- **The 39 LA 2028 source records** become **Claims**: who asserted what, "
            "about which fact, from when, and how much weight the source carries. "
            "Losing claims are kept on purpose.")
    with right:
        st.markdown("#### Deciding what to trust")
        st.markdown(
            "The corpus contains documents that disagree, and none of them says so "
            "about itself. The rule this system applies, in order:\n\n"
            "1. **Authority** -- a signed or certified decision outranks an official "
            "record, which outranks reporting, which outranks an aggregator, which "
            "outranks somebody's recollection.\n"
            "2. **What the document says about itself** -- a final result beats a "
            "draft from the same office.\n"
            "3. **Effective time** -- when the fact took effect.\n\n"
            "**Publication time never decides.** Two of the public questions exist to "
            "punish systems that take the newest document, and in both the newest "
            "document is wrong: a blog that republished a provisional scoreboard, and "
            "a directory that recycled a superseded venue allocation.\n\n"
            "Two further distinctions carry real weight: *repeating* a claim is not "
            "*making* one, and an official record stating that something could not be "
            "verified outranks a witness who is certain of it. Absence of a record is "
            "never a negative fact.")

    st.markdown("---")
    st.markdown("#### No question-specific rules")
    st.markdown(
        "Rule 4 of the brief forbids regexes on question wording, use of question ids, "
        "and answers built into the code. Here that is structural rather than promised: "
        "**no code in the answering path reads the wording of a question.** The only "
        "component that interprets a question is the model, and the only values it may "
        "emit are keys the graph has just handed it. Deterministic string work happens "
        "exclusively on entity names the model extracted and on graph keys. That is why "
        "the same code answers questions it has never seen -- it has never seen any "
        "question.")

# -------------------------------------------------------------- the graph ---
elif section == "The graph":
    st.markdown("### What the graph holds")
    st.caption("Read live from TigerGraph. These are the vertex and edge counts "
               "submitted as deliverable 6.")
    try:
        vertices, edges = graph_counts()
    except Exception as exc:
        st.error(f"Could not read the graph: {exc}")
        st.stop()

    a, b = st.columns(2)
    a.metric("Vertices", f"{sum(vertices.values()):,}")
    b.metric("Edges", f"{sum(edges.values()):,}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Vertices by type**")
        st.dataframe([{"type": k, "count": v} for k, v in sorted(vertices.items())],
                     width="stretch", hide_index=True)
    with right:
        st.markdown("**Edges by type**")
        st.dataframe([{"type": k, "count": v} for k, v in sorted(edges.items())],
                     width="stretch", hide_index=True)
    st.caption("A note worth knowing: counts read while the database is still warming up "
               "come back stale -- a vertex type reporting zero beside thousands of its "
               "own edges. If a number here looks impossible, give it a minute and "
               "reload.")

# ------------------------------------------------------------------ tools ---
elif section == "The agent's tools":
    st.markdown(f"### The agent's {len(TOOLS)} tools")
    st.caption("Each one is an installed GSQL query. The agent has no other way to reach "
               "the data, which is why every database call it makes appears in the trace "
               "by name, with its parameters and the number of rows it returned.")
    st.dataframe([{"installed query": name, "what it answers": doc}
                  for name, doc in TOOLS.items()],
                 width="stretch", hide_index=True, height=600)

# ----------------------------------------------------------------- results ---
else:
    st.markdown("### Public set: 50 questions with published answers")
    answers = load_jsonl(RESULTS / "answers_public.jsonl")
    score_path = RESULTS / "answers_public.jsonl.score.json"
    if not answers:
        st.info("No public run on disk yet. `./run.sh questions_public.jsonl "
                "data/results/r2/answers_public.jsonl`")
        st.stop()

    score = json.loads(score_path.read_text(encoding="utf-8")) if score_path.exists() else None
    cols = st.columns(4)
    if score:
        cols[0].metric("Accuracy", f"{score['accuracy_pct']}%")
        cols[1].metric("Fully correct", f"{score['fully_correct']}/{score['questions']}")
        cols[2].metric("Correct status", f"{score['status_correct']}/{score['questions']}")
        cols[3].metric("Citation recall", score["citation_recall_avg"])
        st.caption("Graded fact by fact with partial credit by a different model from the "
                   "one that answered. The judges re-score these with their own key.")
    else:
        cols[0].metric("Answers", len(answers))

    by_qid = {s["qid"]: s for s in (score or {}).get("per_question", [])}
    st.dataframe(
        [{"qid": a["qid"],
          "score": by_qid.get(a["qid"], {}).get("score", ""),
          "status": a["status"],
          "expected": by_qid.get(a["qid"], {}).get("status_expected", ""),
          "route": a.get("route", ""),
          "answer": str(a["answer"])[:110],
          "tokens": a["input_tokens"] + a["output_tokens"],
          "ms": a["latency_ms"]} for a in answers],
        width="stretch", hide_index=True, height=520)

    wrong = [a for a in answers if by_qid.get(a["qid"], {}).get("score", 1) < 1]
    if wrong:
        st.markdown("#### Where it was not fully right")
        st.caption("Shown deliberately: these are the questions to ask about.")
        for a in wrong:
            s = by_qid.get(a["qid"], {})
            with st.expander(f"{a['qid']} — scored {s.get('score')} — "
                             f"expected {s.get('status_expected')}, said {a['status']}"):
                st.markdown(f"**Our answer.** {a['answer']}")
                if s.get("missing"):
                    st.markdown(f"**Missed.** {'; '.join(s['missing'])}")
                if s.get("wrong"):
                    st.markdown(f"**Wrong.** {'; '.join(s['wrong'])}")
