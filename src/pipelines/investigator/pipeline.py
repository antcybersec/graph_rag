"""Pipeline 5: the Investigator -- one agent, four retrieval tools, provable evidence.

This is the flagship: it keeps the accuracy of the structured event layer while
staying a real agent, because the structured query is a *tool the agent may
choose*, not a fixed route. Per investigation the agent:

  1. plans the next action from {query_events, link_entities, graph_traverse,
     search_chunks, answer} given the question and the evidence so far;
  2. runs it -- structured GSQL over the event layer, entity linking, 1-hop
     traversal, or vector search over chunks;
  3. judges whether the evidence actually answers the question (`evidence_check`
     is required in the same call, so self-evaluation costs no extra LLM call);
  4. answers with citations, or changes strategy and tries again, up to
     MAX_ITERATIONS.

Why a hybrid agent rather than either extreme (see docs/architecture.md):
  - Pure fixed routing (`src/pipelines/event_graph`) answers the templated
    questions perfectly but cannot handle anything outside them.
  - Pure text/graph search (`src/pipelines/agentic`) handles anything but
    cannot count across the 8-43 documents an aggregation question spans.
  The agent gets both: it reaches for the structured tool when the question is
  about Olympic events, and falls back to semantic search when it isn't.

Every step is recorded in `trace`, and every evidence item carries its own
provenance (`doc_id`, `source_url`, and the exact GSQL query + parameters that
produced it), so an answer can be audited back to the source article rather
than merely cited.
"""
import re
import time
from typing import Literal, Optional

from pydantic import BaseModel

from src.common.chunking import n_tokens
from src.common.infobox import norm_key
from src.common.llm import embed, generate
from src.common.rerank import rerank
from src.common.tg_conn import get_connection
from src.ingestion.build_temporal_facts import DATE_OPEN, parse_date
from src.pipelines.event_graph.pipeline import QueryPlan, execute_plan

MAX_ITERATIONS = 5
TOP_K_ENTITIES = 5
CANDIDATE_K_CHUNKS = 20
TOP_K_CHUNKS = 4
MAX_FACTS = 6
MAX_FACT_CANDIDATES = 50
MAX_EVENT_EVIDENCE = 12  # a counting question's own evidence can be 40+ events; cap what re-enters the prompt

ORCHESTRATOR_SYSTEM_INSTRUCTION = """You are the orchestrator of an investigation agent answering
questions from a knowledge graph, a vector index and a document corpus. The corpus is Wikipedia
articles: ~2,200 Olympic event articles (each with venue, date, competitor and nation counts and
medallists), plus films, politicians and other people.

Available actions:
  - query_events(event_query): run a STRUCTURED query over the Olympic event graph. Use this
    whenever the question is about Olympic events -- it is exact, cheap, and it can count or rank
    across every matching event, which text search cannot. `event_query.operation` is one of:
      * count_events_over: how many events of `sport` at the `year`/`season` Games had more than
        `min_competitors` competitors
      * top_event_by_competitors: which event of `sport` at those Games had the most competitors
      * event_field: one `field` (nations, competitors, gold, silver, bronze, venue, date_text) of a
        single event identified by `sport`, `year`, `season`, `event_name`
      * gold_at_venue_date: who won gold at `venue` on `date` (copy both exactly as written; if the
        question names the Games separately, put its year/season in `year`/`season`)
      * gold_previous_games: who won gold in `event_name` of `sport` at the Games immediately before
        `year` (`year` is the year named in the question, not the earlier Games)
    Keep any "Men's"/"Women's"/"Mixed" in `event_name`, and copy `sport` from the article title
    wording (Athletics, Cross-country skiing, Short-track speed skating, ...).
  - query_facts(fact_query): query time-scoped facts that have a validity interval and a source.
    Use it whenever the question is about a point in time or a change over time -- "who held OFFICE
    on DATE", "who was the reigning champion of EVENT in YEAR", "how did X change over time".
    `fact_query.mode` is "as_of" (state at one date; give `as_of_date`) or "timeline" (full
    history). `predicate` is "held_office" (subject = person, object = office) or
    "olympic_champion" (subject = "<Sport> – <Event>", object = the champion). Give whichever side
    you know and leave the other null; facts come back with their validity interval and source.
  - link_entities(query): semantic search for entities in the graph. Use it to identify a subject
    that is not an Olympic event, or before graph_traverse.
  - graph_traverse(entity_ids): expand 1 hop from linked entities, returning related entities, the
    relationship facts connecting them, and chunks mentioning them.
  - search_chunks(query): vector search over raw document text. Use it for anything the structured
    event graph does not model (films, people, free-form detail), or when other actions found
    nothing. For multi-part questions search for one missing piece at a time.
  - answer: stop and answer from the evidence collected.

Rules:
  - Prefer the cheapest action that can actually answer the question, and stop as soon as the
    evidence supports an answer. Do not gather more evidence than the question needs.
  - Never repeat an action with the same arguments. If an action returned nothing new, change
    strategy -- a different tool, or a narrower query.
  - `evidence_check` is REQUIRED on every decision: state in one line what the evidence so far does
    and does not establish. If it does not cover the question, do not choose answer yet.
  - When you choose answer: `final_answer` must cite the evidence tags you used exactly as given
    (e.g. [Q26233801], [Q123_c0], [edge:a--b--c]), and `short_answer` must hold ONLY the bare answer
    (a number, a name, an event title) with no sentence around it. For how-many/which-ones questions,
    enumerate the qualifying items with citations first, then give the count.
  - If the evidence genuinely does not answer the question, say so in `final_answer` rather than
    guessing, and leave `short_answer` null.
  - After {max_iterations} actions you must answer with your best effort, noting what is missing."""


class FactQuery(BaseModel):
    mode: Literal["as_of", "timeline"]
    predicate: Optional[Literal["held_office", "olympic_champion"]] = None
    subject: Optional[str] = None
    object: Optional[str] = None
    as_of_date: Optional[str] = None  # any readable form: "2013-01-01", "7 May 2012", "2013"


class InvestigatorDecision(BaseModel):
    reasoning: str
    evidence_check: str  # self-evaluation, folded into the routing call (no extra LLM call)
    action: Literal["query_events", "query_facts", "link_entities", "graph_traverse", "search_chunks", "answer"]
    event_query: Optional[QueryPlan] = None
    fact_query: Optional[FactQuery] = None
    query: Optional[str] = None
    entity_ids: Optional[list[str]] = None
    final_answer: Optional[str] = None
    short_answer: Optional[str] = None


def _evidence_line(e: dict) -> str:
    return f"[{e['citation_id']}] ({e['source']}) {e['text']}"


def _format_evidence(evidence: list) -> str:
    return "\n".join(_evidence_line(e) for e in evidence) if evidence else "(none yet)"


def _format_history(trace: list) -> str:
    if not trace:
        return "(none yet)"
    lines = []
    for i, t in enumerate(trace):
        args = t.get("query") or t.get("entity_ids") or t.get("event_query") or t.get("fact_query") or ""
        line = f"{i+1}. {t['action']}({args})"
        if "new_evidence" in t:
            line += f" -> {t['new_evidence']} new evidence items"
        lines.append(line)
    return "\n".join(lines)


def _build_prompt(question: str, evidence: list, trace: list) -> str:
    return f"""Question: {question}

Evidence collected so far:
{_format_evidence(evidence)}

Actions already taken:
{_format_history(trace)}

Decide the next action."""


def _add_evidence(evidence: list, new_evidence: list) -> int:
    seen = {e["citation_id"] for e in evidence}
    added = 0
    for e in new_evidence:
        if e["citation_id"] not in seen:
            seen.add(e["citation_id"])
            evidence.append(e)
            added += 1
    return added


def _chunk_text(c: dict) -> str:
    return c["attributes"].get("text", "")


def _doc_lookup(conn, doc_ids: list) -> dict:
    doc_ids = [d for d in doc_ids if d]
    if not doc_ids:
        return {}
    return {d["v_id"]: d["attributes"] for d in conn.getVerticesById("Document", doc_ids)}


def _chunk_evidence(conn, chunks: list, provenance: dict) -> list:
    docs = _doc_lookup(conn, [c["attributes"].get("doc_id", "") for c in chunks])
    out = []
    for c in chunks:
        attrs = c["attributes"]
        doc = docs.get(attrs.get("doc_id", ""), {})
        out.append({
            "citation_id": c["v_id"],
            "source": "chunk",
            "text": f"(from \"{doc.get('title', '?')}\") {attrs.get('text', '')}",
            "doc_id": attrs.get("doc_id", ""),
            "doc_title": doc.get("title", ""),
            "source_url": doc.get("url", ""),
            "provenance": provenance,
        })
    return out


def _run_query_events(conn, plan: QueryPlan, question: str) -> tuple:
    """Structured event-graph query. Returns (evidence, short_answer, summary)."""
    result = execute_plan(conn, plan, question)
    if result is None:
        return [], None, "no matching events in the event graph"
    events = result["evidence"][:MAX_EVENT_EVIDENCE]
    docs = _doc_lookup(conn, [e["event_id"] for e in events])
    provenance = {"tool": "query_events", "operation": plan.operation,
                  "params": {k: v for k, v in plan.model_dump().items() if v is not None}}
    evidence = [{
        "citation_id": e["event_id"],
        "source": "event",
        "text": (f"{e['title']}: competitors={e['competitors']}, nations={e['nations']}, "
                 f"venue={e['venue']}, date={e['date_text']}, gold={e['gold']}"),
        "doc_id": e["event_id"],
        "doc_title": e["title"],
        "source_url": docs.get(e["event_id"], {}).get("url", ""),
        "provenance": provenance,
    } for e in events]
    total = len(result["evidence"])
    summary = result["answer"] + (f" ({total} events examined)" if total > len(events) else "")
    return evidence, result["short_answer"], summary


def _run_query_facts(conn, fq: FactQuery) -> tuple:
    """Time-scoped facts from the temporal layer. Returns (evidence, summary)."""
    params = {
        "predicate": fq.predicate or "",
        "subject_key": norm_key(fq.subject) if fq.subject else "",
        "object_key": norm_key(fq.object) if fq.object else "",
    }
    if fq.mode == "as_of":
        as_of, _precision = parse_date(fq.as_of_date or "")
        # No date given means today. DATE_OPEN would be wrong: the query requires
        # valid_to >= as_of, so 9999-12-31 matches only still-open facts (536 of
        # 2,316, and 1 of 129 office terms) -- a near-empty result, silently.
        params["as_of"] = as_of or int(time.strftime("%Y%m%d"))
        rows = conn.runInstalledQuery("facts_as_of", params=params)[0]["Facts"]
    else:
        rows = conn.runInstalledQuery("fact_timeline", params=params)[0]["Facts"]

    provenance = {"tool": "query_facts", "query": "facts_as_of" if fq.mode == "as_of" else "fact_timeline",
                  "params": params}

    def window(a):
        start = str(a["valid_from"])
        end = "present" if a["valid_to"] >= DATE_OPEN else str(a["valid_to"])
        return f"{start} to {end}"

    evidence = []
    for r in rows[:MAX_EVENT_EVIDENCE]:
        a = r["attributes"]
        evidence.append({
            "citation_id": r["v_id"],
            "source": "temporal_fact",
            "text": f"{a['subject']} --[{a['predicate']}]--> {a['object']} (valid {window(a)}; {a['observed_at']})",
            "doc_id": a.get("source_doc_id", ""),
            "doc_title": "",
            "source_url": a.get("source_url", ""),
            "provenance": provenance,
        })
    summary = f"{len(rows)} time-scoped fact(s)" + ("" if len(rows) <= MAX_EVENT_EVIDENCE else f", showing {MAX_EVENT_EVIDENCE}")
    return evidence, summary


def _run_link_entities(conn, query: str, tracker, question_id) -> tuple:
    vec, _ = embed([query], task_type="RETRIEVAL_QUERY", pipeline="investigator", tracker=tracker, question_id=question_id)
    seeds = conn.runInstalledQuery("vector_search_entities", params={"query_vector": vec[0], "k": TOP_K_ENTITIES})[0]["Result"]
    provenance = {"tool": "link_entities", "query": "vector_search_entities", "params": {"k": TOP_K_ENTITIES, "text": query}}
    evidence = [{
        "citation_id": f"entity:{s['v_id']}",
        "source": "entity_link",
        "text": f"Linked entity '{s['attributes'].get('name')}' ({s['attributes'].get('entity_type')})",
        "doc_id": "", "doc_title": "", "source_url": "", "provenance": provenance,
    } for s in seeds]
    return [s["v_id"] for s in seeds], evidence


def _run_graph_traverse(conn, entity_ids: list, question: str) -> tuple:
    """Returns (new_entity_ids, evidence, error). `error` is surfaced in the trace:
    a failed traversal and an empty one both yield no evidence, and a reader of
    the audit trail must be able to tell them apart."""
    if not entity_ids:
        return [], [], None
    try:
        result = conn.runInstalledQuery("entity_neighbors_1hop", params={"seeds": entity_ids})
        neighbors = result[0]["Neighbors"]
        related_edges = result[1]["related_edges"][:MAX_FACT_CANDIDATES]
        chunks = result[2]["ChunksOfNeighbors"]
        all_ids = list({e["v_id"] for e in neighbors} | set(entity_ids))
        ent_lookup = {e["v_id"]: e["attributes"] for e in conn.getVerticesById("Entity", all_ids)} if all_ids else {}
    except Exception as exc:
        # One bad action (e.g. a hallucinated entity id) must not end the investigation.
        return [], [], f"{type(exc).__name__}: {exc}"[:200]

    def fact_line(e):
        a = ent_lookup.get(e["from_id"], {}).get("name", e["from_id"])
        b = ent_lookup.get(e["to_id"], {}).get("name", e["to_id"])
        attrs = e.get("attributes", {})
        return f"{a} --[{attrs.get('relation_label', 'related_to')}]--> {b}: {attrs.get('description', '')}"

    provenance = {"tool": "graph_traverse", "query": "entity_neighbors_1hop", "params": {"seeds": entity_ids}}
    evidence = []
    for e in rerank(question, related_edges, text_of=fact_line, top_n=MAX_FACTS):
        attrs = e.get("attributes", {})
        evidence.append({
            "citation_id": f"edge:{e['from_id']}--{attrs.get('relation_label', 'related_to')}--{e['to_id']}",
            "source": "fact", "text": fact_line(e),
            "doc_id": attrs.get("source_doc_id", ""), "doc_title": "",
            "source_url": attrs.get("source_url", ""), "provenance": provenance,
        })
    evidence.extend(_chunk_evidence(conn, rerank(question, chunks, text_of=_chunk_text, top_n=TOP_K_CHUNKS), provenance))
    return [n["v_id"] for n in neighbors], evidence, None


def _run_search_chunks(conn, query: str, tracker, question_id) -> list:
    vec, _ = embed([query], task_type="RETRIEVAL_QUERY", pipeline="investigator", tracker=tracker, question_id=question_id)
    result = conn.runInstalledQuery("vector_search_chunks", params={"query_vector": vec[0], "k": CANDIDATE_K_CHUNKS})
    chunks = rerank(query, result[0]["Result"], text_of=_chunk_text, top_n=TOP_K_CHUNKS)
    provenance = {"tool": "search_chunks", "query": "vector_search_chunks", "params": {"k": CANDIDATE_K_CHUNKS, "text": query}}
    return _chunk_evidence(conn, chunks, provenance)


def answer_question(conn, question: str, tracker=None, question_id=None, max_iterations: int = MAX_ITERATIONS) -> dict:
    evidence: list = []
    trace: list = []
    known_entity_ids: list = []
    tool_short_answer = None
    final_answer = None
    short_answer = None
    stop_reason = "max_iterations_hit"

    for i in range(max_iterations):
        decision, _rec = generate(
            _build_prompt(question, evidence, trace),
            system_instruction=ORCHESTRATOR_SYSTEM_INSTRUCTION.format(max_iterations=max_iterations),
            response_schema=InvestigatorDecision,
            pipeline="investigator",
            call_type="orchestrator",
            tracker=tracker,
            question_id=question_id,
            context_tokens=n_tokens(_format_evidence(evidence)),
        )
        step = {
            "iteration": i, "action": decision.action, "reasoning": decision.reasoning,
            "evidence_check": decision.evidence_check, "query": decision.query,
            "entity_ids": decision.entity_ids,
            "event_query": {k: v for k, v in decision.event_query.model_dump().items() if v is not None} if decision.event_query else None,
        }
        t0 = time.time()

        if decision.action == "answer":
            final_answer = decision.final_answer or decision.reasoning or "(no answer produced)"
            # The structured tool's own short answer is exact, so it wins over the
            # model's paraphrase -- but only while the answer still rests on it.
            # `tool_short_answer` is cleared whenever a later tool contributes
            # evidence, otherwise a rejected structured result could override a
            # correct answer the agent reached another way (and exact-match
            # scoring short-circuits on short_answer, so that would score wrong).
            short_answer = tool_short_answer or decision.short_answer
            stop_reason = "answer_found"
            trace.append(step)
            break

        try:
            if decision.action == "query_events":
                if decision.event_query is None:
                    step["error"] = "query_events chosen without an event_query"
                    step["new_evidence"] = 0
                else:
                    new_evidence, tool_answer, summary = _run_query_events(conn, decision.event_query, question)
                    tool_short_answer = tool_answer  # not sticky: a later miss must clear it
                    step["tool_result"] = summary
                    step["new_evidence"] = _add_evidence(evidence, new_evidence)

            elif decision.action == "query_facts":
                if decision.fact_query is None:
                    step["error"] = "query_facts chosen without a fact_query"
                    step["new_evidence"] = 0
                else:
                    step["fact_query"] = {k: v for k, v in decision.fact_query.model_dump().items() if v is not None}
                    new_evidence, summary = _run_query_facts(conn, decision.fact_query)
                    step["tool_result"] = summary
                    step["new_evidence"] = _add_evidence(evidence, new_evidence)
                    if step["new_evidence"]:
                        tool_short_answer = None

            elif decision.action == "link_entities":
                new_ids, new_evidence = _run_link_entities(conn, decision.query or question, tracker, question_id)
                known_entity_ids = list(set(known_entity_ids) | set(new_ids))
                step["new_evidence"] = _add_evidence(evidence, new_evidence)

            elif decision.action == "graph_traverse":
                # entity_ids is free-form model output: keep only ids we have actually
                # seen. Evidence shows entities as "entity:<id>" and the prompt says to
                # cite tags exactly, so the model passes the prefixed form -- strip it,
                # or every well-formed request would be filtered out as hallucinated.
                requested = [e.removeprefix("entity:") for e in (decision.entity_ids or [])]
                requested = [e for e in requested if e in known_entity_ids]
                new_ids, new_evidence, traverse_error = _run_graph_traverse(conn, requested or known_entity_ids, question)
                known_entity_ids = list(set(known_entity_ids) | set(new_ids))
                step["new_evidence"] = _add_evidence(evidence, new_evidence)
                if traverse_error:
                    step["error"] = traverse_error
                if step["new_evidence"]:
                    tool_short_answer = None

            elif decision.action == "search_chunks":
                new_evidence = _run_search_chunks(conn, decision.query or question, tracker, question_id)
                step["new_evidence"] = _add_evidence(evidence, new_evidence)
                if step["new_evidence"]:
                    tool_short_answer = None

        except Exception as exc:
            # Uniform guard: a malformed plan, an unknown vertex id or a transient
            # database error costs this action, not the whole investigation. The
            # agent sees the failure in its history and picks another tool.
            step["error"] = f"{type(exc).__name__}: {exc}"[:200]
            step["new_evidence"] = 0

        step["tool_latency_sec"] = round(time.time() - t0, 3)
        trace.append(step)

    if final_answer is None:
        decision, _rec = generate(
            _build_prompt(question, evidence, trace) + "\n\nYou are out of iterations. Answer now with your best effort.",
            system_instruction=ORCHESTRATOR_SYSTEM_INSTRUCTION.format(max_iterations=max_iterations),
            response_schema=InvestigatorDecision,
            pipeline="investigator",
            call_type="orchestrator_forced_stop",
            tracker=tracker,
            question_id=question_id,
            context_tokens=n_tokens(_format_evidence(evidence)),
        )
        final_answer = decision.final_answer or "(no answer produced)"
        short_answer = tool_short_answer or decision.short_answer
        # Record it: this call produces the answer, so leaving it out of the trace
        # would undercount steps and hide the reasoning behind the final answer.
        trace.append({
            "iteration": max_iterations, "action": "answer", "forced": True,
            "reasoning": decision.reasoning, "evidence_check": decision.evidence_check,
        })

    tools_used = {t["action"] for t in trace}
    return {
        "answer": final_answer,
        "short_answer": short_answer,
        "evidence": evidence,
        "trace": trace,
        "num_steps": len(trace),
        "stop_reason": stop_reason,
        "tools_used": sorted(tools_used),
        "strategy_changed": len(tools_used - {"answer"}) > 1,
        "chunks_used": [e["citation_id"] for e in evidence if e["source"] == "chunk"],
        "facts_used": [e["citation_id"] for e in evidence if e["source"] == "fact"],
        # Event evidence is already a doc id; chunk ids are "{doc_id}_c{n}".
        "retrieved_doc_ids": sorted({e["doc_id"] for e in evidence if e.get("doc_id")}
                                    | {re.sub(r"_c\d+$", "", e["citation_id"]) for e in evidence if e["source"] == "chunk"}),
        "context_text": _format_evidence(evidence),
    }


if __name__ == "__main__":
    conn = get_connection()
    for q in [
        "According to the provided corpus, how many biathlon events at the 2018 Winter Olympics had more than 73 competitors?",
        "Who directed the film Jab We Met?",
    ]:
        r = answer_question(conn, q)
        print(f"\nQ: {q}\nA: {r['answer'][:200]}\nshort: {r['short_answer']} | steps: {r['num_steps']} | tools: {r['tools_used']}")
        for t in r["trace"]:
            print("  -", t["action"], "|", (t.get("evidence_check") or "")[:90])
