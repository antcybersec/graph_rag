"""Pipeline 2: GraphRAG. Fixed sequence (no agentic decision-making):
  entity linking (vector search over Entity embeddings)
    -> 1-hop graph traversal (RELATED_TO neighbors + their MENTIONS chunks)
    -> answer synthesis, grounded in BOTH the structured relationship facts
       AND the retrieved chunk text.

This is the "local search" mode from the GraphRAG literature -- entity-
anchored retrieval, as opposed to the "global search" community-summary
mode (skipped for the hackathon's time budget; see docs/architecture.md).
"""
from src.common.chunking import n_tokens
from src.common.llm import embed, generate
from src.common.tg_conn import get_connection

TOP_K_ENTITIES = 5

SYSTEM_INSTRUCTION = """You answer questions using ONLY the provided evidence -- do not use outside
knowledge. Evidence comes in two forms: (1) structured relationship facts extracted from a
knowledge graph, each citing an edge id like [edge:entity_a--relation--entity_b], and (2) raw
document chunks, each citing a chunk_id like [Q123_c0]. Cite whichever evidence you actually used
right after each claim. If the evidence doesn't contain enough information to answer, say so
explicitly rather than guessing."""

PROMPT_TEMPLATE = """Question: {question}

Structured relationship facts from the knowledge graph:
{facts}

Supporting document chunks:
{chunks}

Answer the question, citing evidence as instructed."""


def link_entities(conn, question: str, k: int = TOP_K_ENTITIES, tracker=None, question_id=None):
    vec, _rec = embed([question], task_type="RETRIEVAL_QUERY", pipeline="graphrag", tracker=tracker, question_id=question_id)
    result = conn.runInstalledQuery("vector_search_entities", params={"query_vector": vec[0], "k": k})
    seeds = result[0]["Result"]
    return seeds  # list of {"v_id":..., "v_type":..., "attributes": {...}}


MAX_FACTS = 20  # defensive cap independent of the GSQL-side LIMIT -- GSQL's ACCUM
# still runs over every matched edge before LIMIT trims the output vertex set, so
# related_edges isn't guaranteed bounded by the query's own LIMIT alone (see
# create_queries.py). Chunks ARE bounded there (plain SELECT, no ACCUM), so no
# second cap needed on those.


def expand_neighborhood(conn, seed_ids: list):
    result = conn.runInstalledQuery("entity_neighbors_1hop", params={"seeds": seed_ids})
    neighbors = result[0]["Neighbors"]
    related_edges = result[1]["related_edges"][:MAX_FACTS]
    chunks = result[2]["ChunksOfNeighbors"]
    return neighbors, related_edges, chunks


def format_facts(related_edges: list, entity_lookup: dict) -> str:
    lines = []
    for e in related_edges:
        a_id, b_id = e["from_id"], e["to_id"]
        a_name = entity_lookup.get(a_id, {}).get("name", a_id)
        b_name = entity_lookup.get(b_id, {}).get("name", b_id)
        attrs = e.get("attributes", {})
        label = attrs.get("relation_label", "related_to")
        desc = attrs.get("description", "")
        edge_cite = f"edge:{a_id}--{label}--{b_id}"
        lines.append(f"[{edge_cite}] {a_name} --[{label}]--> {b_name}: {desc}")
    return "\n".join(lines) if lines else "(no structured facts found)"


def format_chunks(chunks: list, doc_lookup: dict) -> str:
    parts = []
    for c in chunks:
        attrs = c["attributes"]
        doc = doc_lookup.get(attrs.get("doc_id", ""), {})
        parts.append(
            f"[{c['v_id']}] (source: \"{doc.get('title', '?')}\", {doc.get('url', '')})\n{attrs.get('text', '')}"
        )
    return "\n\n".join(parts) if parts else "(no chunks found)"


def answer_question(conn, question: str, k_entities: int = TOP_K_ENTITIES, tracker=None, question_id=None) -> dict:
    seeds = link_entities(conn, question, k_entities, tracker=tracker, question_id=question_id)
    seed_ids = [s["v_id"] for s in seeds]

    if not seed_ids:
        # no entity in the graph matches this question at all -- fall back to "no evidence"
        neighbors, related_edges, chunks = [], [], []
    else:
        neighbors, related_edges, chunks = expand_neighborhood(conn, seed_ids)

    all_entity_ids = list({s["v_id"] for s in seeds} | {n["v_id"] for n in neighbors})
    entities = conn.getVerticesById("Entity", all_entity_ids) if all_entity_ids else []
    entity_lookup = {e["v_id"]: e["attributes"] for e in entities}

    doc_ids = list({c["attributes"].get("doc_id", "") for c in chunks if c["attributes"].get("doc_id")})
    docs = conn.getVerticesById("Document", doc_ids) if doc_ids else []
    doc_lookup = {d["v_id"]: d["attributes"] for d in docs}

    facts_text = format_facts(related_edges, entity_lookup)
    chunks_text = format_chunks(chunks, doc_lookup)
    context_tokens = n_tokens(facts_text) + n_tokens(chunks_text)

    prompt = PROMPT_TEMPLATE.format(question=question, facts=facts_text, chunks=chunks_text)
    answer, _rec = generate(
        prompt,
        system_instruction=SYSTEM_INSTRUCTION,
        pipeline="graphrag",
        call_type="generation",
        tracker=tracker,
        question_id=question_id,
        context_tokens=context_tokens,
    )

    return {
        "answer": answer,
        "seed_entities": seed_ids,
        "expanded_entities": [n["v_id"] for n in neighbors],
        "chunks_used": [c["v_id"] for c in chunks],
        "retrieved_doc_ids": sorted(doc_ids),
        "num_facts": len(related_edges),
        "context_text": facts_text + "\n\n" + chunks_text,
    }


if __name__ == "__main__":
    conn = get_connection()
    q = "Who won the gold medal in the women's artistic individual all-around at the 2016 Summer Olympics?"
    result = answer_question(conn, q)
    print("ANSWER:", result["answer"])
    print("SEEDS:", result["seed_entities"])
    print("EXPANDED:", result["expanded_entities"])
    print("CHUNKS:", result["chunks_used"])
