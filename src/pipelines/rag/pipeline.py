"""Pipeline 1: baseline RAG. Similarity search over Chunk embeddings only --
no graph structure, no agentic loop. This is the floor every other pipeline
is measured against.

Uses the SAME GEN_MODEL (and no thinking-budget override) as GraphRAG and
Agentic GraphRAG's final answer-synthesis step, so the three-way comparison
isolates the effect of retrieval strategy rather than generation settings.
"""
from src.common.chunking import n_tokens
from src.common.llm import embed, generate
from src.common.tg_conn import get_connection

TOP_K = 8

SYSTEM_INSTRUCTION = """You answer questions using ONLY the provided context chunks -- do not use
outside knowledge. Cite the chunk_id(s) you relied on in square brackets right after each claim,
e.g. "Simone Biles won gold [Q26233801_c0]." If the context doesn't contain enough information to
answer, say so explicitly rather than guessing."""

PROMPT_TEMPLATE = """Question: {question}

Context:
{context}

Answer the question, citing chunk_ids as instructed."""


def retrieve(conn, question: str, k: int = TOP_K, tracker=None, question_id=None):
    vec, _rec = embed([question], task_type="RETRIEVAL_QUERY", pipeline="rag", tracker=tracker, question_id=question_id)
    result = conn.runInstalledQuery("vector_search_chunks", params={"query_vector": vec[0], "k": k})
    chunks = result[0]["Result"]
    distances = result[1].get("@@distances", {})
    return chunks, distances


def build_context(chunks: list, doc_lookup: dict) -> str:
    parts = []
    for c in chunks:
        attrs = c["attributes"]
        doc = doc_lookup.get(attrs.get("doc_id", ""), {})
        parts.append(
            f"[{c['v_id']}] (source: \"{doc.get('title', '?')}\", {doc.get('url', '')})\n{attrs.get('text', '')}"
        )
    return "\n\n".join(parts)


def answer_question(conn, question: str, k: int = TOP_K, tracker=None, question_id=None) -> dict:
    chunks, distances = retrieve(conn, question, k, tracker=tracker, question_id=question_id)

    doc_ids = list({c["attributes"].get("doc_id", "") for c in chunks if c["attributes"].get("doc_id")})
    docs = conn.getVerticesById("Document", doc_ids) if doc_ids else []
    doc_lookup = {d["v_id"]: d["attributes"] for d in docs}

    context = build_context(chunks, doc_lookup)
    context_tokens = n_tokens(context)

    prompt = PROMPT_TEMPLATE.format(question=question, context=context)
    answer, _rec = generate(
        prompt,
        system_instruction=SYSTEM_INSTRUCTION,
        pipeline="rag",
        call_type="generation",
        tracker=tracker,
        question_id=question_id,
        context_tokens=context_tokens,
    )

    retrieved_doc_ids = sorted(doc_ids)
    return {
        "answer": answer,
        "chunks_used": [c["v_id"] for c in chunks],
        "retrieved_doc_ids": retrieved_doc_ids,
        "distances": distances,
        "context_text": context,
    }


if __name__ == "__main__":
    conn = get_connection()
    q = "Who won the gold medal in the women's artistic individual all-around at the 2016 Summer Olympics?"
    result = answer_question(conn, q)
    print("ANSWER:", result["answer"])
    print("CHUNKS USED:", result["chunks_used"])
    print("DOC IDS:", result["retrieved_doc_ids"])
