"""Pipeline 2: GraphRAG. Fixed sequence (no agentic decision-making):
  entity linking (vector search over Entity embeddings)
    -> 1-hop graph traversal (RELATED_TO neighbors + their MENTIONS chunks)
    -> rerank facts and chunks against the question (local cross-encoder)
    -> answer synthesis, grounded in BOTH the structured relationship facts
       AND the retrieved chunk text.

This is the "local search" mode from the GraphRAG literature -- entity-
anchored retrieval, as opposed to the "global search" community-summary
mode (skipped for the hackathon's time budget; see docs/architecture.md).

Chunk candidates are the traversal's MENTIONS chunks PLUS a vector-search
pool over chunk text (hybrid local search). The traversal query's chunk
LIMIT picks arbitrary chunks, not the relevant ones, and entity linking
misses often -- graph-only chunks scored 1.89/5 accuracy and answered
"insufficient evidence" on 58/100 public questions in the first benchmark.
"""
from src.common.chunking import n_tokens
from src.common.llm import embed, generate
from src.common.rerank import rerank
from src.common.tg_conn import get_connection

TOP_K_ENTITIES = 5
CANDIDATE_K_CHUNKS = 20
TOP_K_CHUNKS = 5
MAX_FACTS = 6
MAX_FACT_CANDIDATES = 50  # bound on reranker work -- GSQL's ACCUM runs over every
# matched edge before LIMIT trims the output vertex set, so related_edges isn't
# guaranteed bounded by the query's own LIMIT alone (see create_queries.py).

SYSTEM_INSTRUCTION = """You answer questions using ONLY the provided evidence -- do not use outside
knowledge. Evidence comes in two forms: (1) structured relationship facts extracted from a
knowledge graph, each citing an edge id like [edge:entity_a--relation--entity_b], and (2) raw
document chunks, each citing a chunk_id like [Q123_c0]. Cite whichever evidence you actually used
right after each claim. If the question asks how many, which ones, or for a list, first enumerate
every qualifying item found in the evidence (each with its citation), then give the count or list.
If the evidence doesn't contain enough information to answer, say so explicitly rather than
guessing."""

PROMPT_TEMPLATE = """Question: {question}

Structured relationship facts from the knowledge graph:
{facts}

Supporting document chunks:
{chunks}

Answer the question, citing evidence as instructed."""


def link_entities(conn, query_vector: list, k: int = TOP_K_ENTITIES):
    result = conn.runInstalledQuery("vector_search_entities", params={"query_vector": query_vector, "k": k})
    seeds = result[0]["Result"]
    return seeds  # list of {"v_id":..., "v_type":..., "attributes": {...}}


def expand_neighborhood(conn, seed_ids: list):
    result = conn.runInstalledQuery("entity_neighbors_1hop", params={"seeds": seed_ids})
    neighbors = result[0]["Neighbors"]
    related_edges = result[1]["related_edges"][:MAX_FACT_CANDIDATES]
    chunks = result[2]["ChunksOfNeighbors"]
    return neighbors, related_edges, chunks


def fact_text(e: dict, entity_lookup: dict) -> str:
    a_name = entity_lookup.get(e["from_id"], {}).get("name", e["from_id"])
    b_name = entity_lookup.get(e["to_id"], {}).get("name", e["to_id"])
    attrs = e.get("attributes", {})
    return f"{a_name} --[{attrs.get('relation_label', 'related_to')}]--> {b_name}: {attrs.get('description', '')}"


def format_facts(related_edges: list, entity_lookup: dict) -> str:
    lines = []
    for e in related_edges:
        label = e.get("attributes", {}).get("relation_label", "related_to")
        edge_cite = f"edge:{e['from_id']}--{label}--{e['to_id']}"
        lines.append(f"[{edge_cite}] {fact_text(e, entity_lookup)}")
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
    # One query embedding serves both entity linking and the chunk candidate pool.
    vec, _rec = embed([question], task_type="RETRIEVAL_QUERY", pipeline="graphrag", tracker=tracker, question_id=question_id)
    query_vector = vec[0]

    seeds = link_entities(conn, query_vector, k_entities)
    seed_ids = [s["v_id"] for s in seeds]

    if not seed_ids:
        # no entity in the graph matches this question at all -- vector chunks only
        neighbors, related_edges, graph_chunks = [], [], []
    else:
        neighbors, related_edges, graph_chunks = expand_neighborhood(conn, seed_ids)

    vector_chunks = conn.runInstalledQuery(
        "vector_search_chunks", params={"query_vector": query_vector, "k": CANDIDATE_K_CHUNKS}
    )[0]["Result"]

    all_entity_ids = list({s["v_id"] for s in seeds} | {n["v_id"] for n in neighbors})
    entities = conn.getVerticesById("Entity", all_entity_ids) if all_entity_ids else []
    entity_lookup = {e["v_id"]: e["attributes"] for e in entities}

    related_edges = rerank(question, related_edges, text_of=lambda e: fact_text(e, entity_lookup), top_n=MAX_FACTS)
    chunks = rerank(
        question, graph_chunks + vector_chunks,
        text_of=lambda c: c["attributes"].get("text", ""), top_n=TOP_K_CHUNKS,
    )

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

    graph_chunk_ids = {c["v_id"] for c in graph_chunks}
    return {
        "answer": answer,
        "seed_entities": seed_ids,
        "expanded_entities": [n["v_id"] for n in neighbors],
        "chunks_used": [c["v_id"] for c in chunks],
        "chunks_from_graph": sum(c["v_id"] in graph_chunk_ids for c in chunks),
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
    print("CHUNKS:", result["chunks_used"], "from graph:", result["chunks_from_graph"])
