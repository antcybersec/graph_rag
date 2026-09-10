"""Pipeline 3: Agentic GraphRAG. An orchestrator LLM call, run in a loop,
decides the NEXT action from {link_entities, graph_traverse, search_chunks,
answer} based on the original question, evidence collected so far, and
which actions have already been tried -- not a fixed sequence.

Design choices, and why (see docs/architecture.md for the fuller research
writeup this follows):
  - The orchestrator itself decides when evidence is "sufficient" and, when
    it is, produces the final answer in that SAME call (folding CRAG-style
    evidence evaluation into the routing decision instead of a separate LLM
    call) -- this halves the LLM-call count per question versus a
    separate-evaluator design, which matters given this project's observed
    free-tier rate limits.
  - A hard MAX_ITERATIONS cap is non-negotiable (Agentic RAG survey /
    LangGraph literature: uncapped agentic loops are a documented failure
    mode -- "retrieval thrash").
  - Every action taken is recorded in `trace` so the metrics dashboard can
    show whether the agent changed strategy, how many steps it took, and
    why it stopped (`stop_reason`) -- required by the hackathon's own
    "agentic effectiveness" judging criterion, not optional instrumentation.
"""
import re
from typing import Literal, Optional

from pydantic import BaseModel

from src.common.chunking import n_tokens
from src.common.llm import embed, generate
from src.common.tg_conn import get_connection

MAX_ITERATIONS = 5
TOP_K_ENTITIES = 5
TOP_K_CHUNKS = 6

ORCHESTRATOR_SYSTEM_INSTRUCTION = """You are the orchestrator of an investigation agent answering
questions over a knowledge graph + document corpus about the Olympics (and some unrelated topics).
Available actions:
  - link_entities(query): semantic search for entities in the graph matching `query`. Use this
    first, or when you need to identify a NEW entity not yet linked.
  - graph_traverse(entity_ids): expand 1 hop from the given entity ids, returning related entities,
    the relationship facts connecting them, and document chunks that mention them. Use this to find
    facts connected to entities you've already linked.
  - search_chunks(query): direct semantic search over document text chunks (bypasses the graph
    entirely). Use this when the question needs raw text/context the graph's structured facts won't
    capture, or when entity linking found nothing useful.
  - answer: STOP investigating and produce the final answer now, using only the evidence collected
    so far. Choose this as soon as you have enough evidence -- do not keep gathering more once you
    can already answer, and do not answer if the evidence so far clearly does not cover the question
    (in that case take one more action instead).

Rules:
  - Never repeat the exact same action+query/entity_ids combination twice in a row.
  - When you choose "answer", `final_answer` must cite the evidence you used, with citation tags
    exactly as given (e.g. "[Q123_c0]" for a chunk, "[edge:a--b--c]" for a fact). If the collected
    evidence is insufficient to answer, say so explicitly in `final_answer` rather than guessing.
  - If you have taken {max_iterations} actions already and still lack enough evidence, choose
    "answer" anyway and give your best-effort answer, noting what's missing."""


class OrchestratorDecision(BaseModel):
    reasoning: str
    # A plain `str` here let a local model (Ministral-3, via Ollama) emit
    # "graph_traverse([entity:foo])" -- valid JSON, invalid action, no schema-level
    # rejection -- which fell through to the "unknown action" branch and aborted the
    # whole investigation. Literal makes malformed values a hard schema-validation
    # failure (caught by generate()'s existing retry-on-parse-failure path) instead
    # of a value this loop has to detect and handle itself.
    action: Literal["link_entities", "graph_traverse", "search_chunks", "answer"]
    query: Optional[str] = None
    entity_ids: Optional[list[str]] = None
    final_answer: Optional[str] = None


def _format_evidence(evidence: list) -> str:
    if not evidence:
        return "(none yet)"
    return "\n".join(f"[{e['citation_id']}] ({e['source']}) {e['text']}" for e in evidence)


def _format_history(trace: list) -> str:
    if not trace:
        return "(none yet)"
    return "\n".join(f"{i+1}. {t['action']}({t.get('query') or t.get('entity_ids') or ''})" for i, t in enumerate(trace))


def _build_orchestrator_prompt(question: str, evidence: list, trace: list) -> str:
    return f"""Question: {question}

Evidence collected so far:
{_format_evidence(evidence)}

Actions already taken:
{_format_history(trace)}

Decide the next action."""


def _run_link_entities(conn, query: str, tracker, question_id) -> tuple:
    vec, _ = embed([query], task_type="RETRIEVAL_QUERY", pipeline="agentic", tracker=tracker, question_id=question_id)
    result = conn.runInstalledQuery("vector_search_entities", params={"query_vector": vec[0], "k": TOP_K_ENTITIES})
    seeds = result[0]["Result"]
    entity_ids = [s["v_id"] for s in seeds]
    evidence = [
        {"citation_id": f"entity:{s['v_id']}", "source": "entity_link", "text": f"Linked entity '{s['attributes'].get('name')}' ({s['attributes'].get('entity_type')})"}
        for s in seeds
    ]
    return entity_ids, evidence


def _run_graph_traverse(conn, entity_ids: list) -> tuple:
    if not entity_ids:
        return [], []
    try:
        # Everything below reuses `entity_ids` (getVerticesById as well as the
        # initial query) -- entity_ids can include orchestrator-hallucinated IDs
        # (decision.entity_ids is free-form LLM output, only best-effort filtered
        # against known entities at the call site) that any of these calls could
        # choke on, not just the first one. One bad action shouldn't torch an
        # entire multi-step investigation (and the tokens already spent on it),
        # so the whole lookup is one unit: treat any failure in it as "this
        # action found nothing" and let the orchestrator try something else.
        result = conn.runInstalledQuery("entity_neighbors_1hop", params={"seeds": entity_ids})
        neighbors = result[0]["Neighbors"]
        # Defensive cap independent of the GSQL-side LIMIT -- see graphrag/pipeline.py's
        # MAX_FACTS comment: ACCUM runs over every matched edge before LIMIT trims the
        # output vertex set, so this isn't guaranteed bounded by the query alone.
        related_edges = result[1]["related_edges"][:20]
        chunks = result[2]["ChunksOfNeighbors"]

        all_ids = list({e["v_id"] for e in neighbors} | set(entity_ids))
        ent_lookup = {e["v_id"]: e["attributes"] for e in conn.getVerticesById("Entity", all_ids)} if all_ids else {}
    except Exception:
        return [], []

    evidence = []
    for e in related_edges:
        a, b = e["from_id"], e["to_id"]
        a_name = ent_lookup.get(a, {}).get("name", a)
        b_name = ent_lookup.get(b, {}).get("name", b)
        attrs = e.get("attributes", {})
        label = attrs.get("relation_label", "related_to")
        cite = f"edge:{a}--{label}--{b}"
        evidence.append({"citation_id": cite, "source": "fact", "text": f"{a_name} --[{label}]--> {b_name}: {attrs.get('description', '')}"})

    doc_ids = list({c["attributes"].get("doc_id", "") for c in chunks if c["attributes"].get("doc_id")})
    doc_lookup = {d["v_id"]: d["attributes"] for d in conn.getVerticesById("Document", doc_ids)} if doc_ids else {}
    for c in chunks:
        attrs = c["attributes"]
        doc = doc_lookup.get(attrs.get("doc_id", ""), {})
        evidence.append({"citation_id": c["v_id"], "source": "chunk", "text": f"(from \"{doc.get('title','?')}\") {attrs.get('text', '')}"})

    new_entity_ids = [n["v_id"] for n in neighbors]
    return new_entity_ids, evidence


def _run_search_chunks(conn, query: str, tracker, question_id) -> list:
    vec, _ = embed([query], task_type="RETRIEVAL_QUERY", pipeline="agentic", tracker=tracker, question_id=question_id)
    result = conn.runInstalledQuery("vector_search_chunks", params={"query_vector": vec[0], "k": TOP_K_CHUNKS})
    chunks = result[0]["Result"]
    doc_ids = list({c["attributes"].get("doc_id", "") for c in chunks if c["attributes"].get("doc_id")})
    doc_lookup = {d["v_id"]: d["attributes"] for d in conn.getVerticesById("Document", doc_ids)} if doc_ids else {}
    evidence = []
    for c in chunks:
        attrs = c["attributes"]
        doc = doc_lookup.get(attrs.get("doc_id", ""), {})
        evidence.append({"citation_id": c["v_id"], "source": "chunk", "text": f"(from \"{doc.get('title','?')}\") {attrs.get('text', '')}"})
    return evidence


def answer_question(conn, question: str, tracker=None, question_id=None, max_iterations: int = MAX_ITERATIONS) -> dict:
    evidence: list = []
    trace: list = []
    known_entity_ids: list = []
    stop_reason = "max_iterations_hit"
    final_answer = None

    for i in range(max_iterations):
        prompt = _build_orchestrator_prompt(question, evidence, trace)
        sys_instr = ORCHESTRATOR_SYSTEM_INSTRUCTION.format(max_iterations=max_iterations)
        decision, _rec = generate(
            prompt,
            system_instruction=sys_instr,
            response_schema=OrchestratorDecision,
            pipeline="agentic",
            call_type="orchestrator",
            tracker=tracker,
            question_id=question_id,
            context_tokens=n_tokens(_format_evidence(evidence)),
        )

        step = {"iteration": i, "action": decision.action, "query": decision.query, "entity_ids": decision.entity_ids, "reasoning": decision.reasoning}

        if decision.action == "answer":
            # Some models (observed: Ministral-3 via Ollama) choose action="answer" and
            # write the actual answer into `reasoning` while leaving `final_answer`
            # empty, despite the schema and instructions -- reasoning is a much better
            # fallback than a placeholder when that happens.
            final_answer = decision.final_answer or decision.reasoning or "(no answer produced)"
            stop_reason = "answer_found"
            trace.append(step)
            break

        elif decision.action == "link_entities":
            new_ids, new_evidence = _run_link_entities(conn, decision.query or question, tracker, question_id)
            known_entity_ids = list(set(known_entity_ids) | set(new_ids))
            evidence.extend(new_evidence)

        elif decision.action == "graph_traverse":
            # decision.entity_ids is free-form LLM output -- prefer the subset that
            # matches entities we've actually linked/discovered so far over trusting
            # it outright (it can hallucinate IDs not in the graph at all -- observed
            # verbatim citation-tag copies like "[entity:foo]" from a local model).
            # Deliberately does NOT fall back to the raw/unfiltered decision.entity_ids
            # when nothing matches -- that was the whole hallucination this guards
            # against, so falling back to known_entity_ids (or nothing) is correct
            # even though it means occasionally ignoring the model's stated request.
            requested = [e for e in (decision.entity_ids or []) if e in known_entity_ids]
            target_ids = requested or known_entity_ids
            new_ids, new_evidence = _run_graph_traverse(conn, target_ids)
            known_entity_ids = list(set(known_entity_ids) | set(new_ids))
            evidence.extend(new_evidence)

        elif decision.action == "search_chunks":
            new_evidence = _run_search_chunks(conn, decision.query or question, tracker, question_id)
            evidence.extend(new_evidence)

        else:
            step["error"] = f"unknown action: {decision.action}"
            stop_reason = "error"
            trace.append(step)
            break

        trace.append(step)

    if final_answer is None:
        # ran out of iterations without the model choosing "answer" -- force a final synthesis
        prompt = _build_orchestrator_prompt(question, evidence, trace) + "\n\nYou are out of iterations. Answer now with your best effort."
        decision, _rec = generate(
            prompt,
            system_instruction=ORCHESTRATOR_SYSTEM_INSTRUCTION.format(max_iterations=max_iterations),
            response_schema=OrchestratorDecision,
            pipeline="agentic",
            call_type="orchestrator_forced_stop",
            tracker=tracker,
            question_id=question_id,
        )
        final_answer = decision.final_answer or "(no answer produced)"

    strategies_used = {t["action"] for t in trace}
    chunks_used = [e["citation_id"] for e in evidence if e["source"] == "chunk"]
    return {
        "answer": final_answer,
        "trace": trace,
        "num_steps": len(trace),
        "stop_reason": stop_reason,
        "strategy_changed": len(strategies_used) > 1,
        "chunks_used": chunks_used,
        "facts_used": [e["citation_id"] for e in evidence if e["source"] == "fact"],
        # chunk_id format is "{doc_id}_c{index}" (see load_documents_chunks.py) --
        # derived here rather than threaded through from each retrieval helper.
        # Was missing entirely before this fix, so run_benchmark.py's doc
        # precision/recall silently scored agentic against an empty list every
        # time (always 0.0), not because retrieval was bad but because this
        # field never existed.
        "retrieved_doc_ids": sorted({re.sub(r"_c\d+$", "", cid) for cid in chunks_used}),
        "context_text": _format_evidence(evidence),
    }


if __name__ == "__main__":
    conn = get_connection()
    q = "Who won the gold medal in the women's artistic individual all-around at the 2016 Summer Olympics?"
    result = answer_question(conn, q)
    print("ANSWER:", result["answer"])
    print("STEPS:", result["num_steps"], "STOP:", result["stop_reason"])
    for t in result["trace"]:
        print(" -", t["action"], t.get("query") or t.get("entity_ids"), "--", t["reasoning"][:80])
