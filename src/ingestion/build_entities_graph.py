"""Merges per-document entity/relationship extractions into a deduplicated
entity graph and loads Entity vertices + MENTIONS + RELATED_TO edges into
TigerGraph.

Entity resolution strategy (see docs/architecture.md, GraphRAG-foundations
research): merge purely by normalized name -- NOT (name, entity_type) --
because the extractor is occasionally inconsistent about type (e.g. calling
one document's Olympic-games entity "Competition" and another's "Event").
Majority-vote the type across all mentions instead. This is the cheap
exact-match tier of entity resolution; embedding-similarity-based merging of
near-duplicate names (e.g. "Bob Smith" vs "B. Smith") is a documented,
deliberately-skipped stretch goal given the hackathon's time budget.

MENTIONS edges are derived by re-chunking each document locally (same
deterministic chunker used at load time -> same chunk_ids) and substring-
matching each doc's entity names against each chunk's text -- no extra LLM
or embedding calls needed for this step.

Must run AFTER load_documents_chunks.py (MENTIONS edges reference Chunk
vertices; RELATED_TO/Entity vertices don't strictly need Chunks to exist
first, but running in this order keeps everything consistent).

Usage:
  python -m src.ingestion.build_entities_graph [--limit N]
"""
import argparse
import json
import re
from collections import Counter, defaultdict

from tqdm import tqdm

from src.common.chunking import chunk_document
from src.common.llm import embed
from src.common.tg_conn import get_connection

EXTRACTIONS_PATH = "data/results/extractions.jsonl"
CORPUS_PATH = "data/raw_dataset/corpus/corpus.jsonl"
UPSERT_BATCH_SIZE = 500
EMBED_BATCH_SIZE = 50


def normalize_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^\w\s-]", "", name)  # drop punctuation
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def slugify(name: str) -> str:
    s = re.sub(r"[^\w]+", "_", name.strip().lower()).strip("_")
    return s or "unknown"


def load_extractions(limit=None):
    records = []
    with open(EXTRACTIONS_PATH) as f:
        for line in f:
            records.append(json.loads(line))
    return records[:limit] if limit else records


def load_corpus_lookup():
    lookup = {}
    with open(CORPUS_PATH) as f:
        for line in f:
            d = json.loads(line)
            lookup[d["doc_id"]] = d
    return lookup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="limit to first N extraction records (testing)")
    parser.add_argument("--dry-run", action="store_true", help="build and print stats but don't write to TigerGraph")
    args = parser.parse_args()

    records = load_extractions(args.limit)
    print(f"Loaded {len(records)} document extractions.")
    corpus = load_corpus_lookup()

    # --- Pass 1: entity resolution (merge by normalized name) ---
    entity_names = Counter()      # normalized_name -> Counter of exact-case variants
    entity_types = defaultdict(Counter)
    entity_descriptions = defaultdict(list)
    entity_mention_count = Counter()

    for rec in records:
        for e in rec["entities"]:
            key = normalize_name(e["name"])
            if not key:
                continue
            entity_names[key] += 1
            entity_types[key][e["entity_type"]] += 1
            if len(entity_descriptions[key]) < 3 and e["description"] not in entity_descriptions[key]:
                entity_descriptions[key].append(e["description"])
            entity_mention_count[key] += 1

    # canonical display name = most common exact-case spelling ever seen per key
    exact_name_counts = defaultdict(Counter)
    for rec in records:
        for e in rec["entities"]:
            key = normalize_name(e["name"])
            if key:
                exact_name_counts[key][e["name"]] += 1

    entity_keys = list(entity_names.keys())
    entity_id_of = {key: slugify(key) for key in entity_keys}
    print(f"Resolved {len(entity_keys)} unique entities from {sum(entity_names.values())} mentions.")

    # --- Pass 2: relationships, deduped by (source_id, target_id_sorted, relation_label) ---
    rel_agg = {}  # (a_id, b_id, relation_label) -> dict(description, weight, source_doc_id, source_url)
    skipped_rels = 0
    for rec in records:
        doc = corpus.get(rec["doc_id"], {})
        for r in rec["relationships"]:
            a_key, b_key = normalize_name(r["source"]), normalize_name(r["target"])
            if a_key not in entity_id_of or b_key not in entity_id_of or a_key == b_key:
                skipped_rels += 1
                continue
            a_id, b_id = entity_id_of[a_key], entity_id_of[b_key]
            # undirected: canonicalize pair order so both directions dedupe to the same key
            pair = tuple(sorted([a_id, b_id]))
            rk = (pair[0], pair[1], r["relation_label"])
            if rk in rel_agg:
                rel_agg[rk]["weight"] += 1
            else:
                rel_agg[rk] = {
                    "description": r["description"],
                    "weight": 1,
                    "source_doc_id": rec["doc_id"],
                    "source_url": doc.get("url", ""),
                }
    print(f"Resolved {len(rel_agg)} unique (entity,entity,relation) edges ({skipped_rels} relationships skipped).")

    if args.dry_run:
        print("\n--- sample entities ---")
        for key in entity_keys[:10]:
            top_type = entity_types[key].most_common(1)[0][0]
            print(f"  {entity_id_of[key]}: '{exact_name_counts[key].most_common(1)[0][0]}' [{top_type}] "
                  f"(mentions={entity_mention_count[key]})")
        print("\n--- sample relationships ---")
        for rk, v in list(rel_agg.items())[:10]:
            print(f"  {rk[0]} --[{rk[2]}]--> {rk[1]} (weight={v['weight']}) -- {v['description']}")
        return

    conn = get_connection()

    # --- Embed entities (name + description) so vector_search_entities (entity linking for
    # GraphRAG/Agentic) has something to search over -- entity linking is dead in the water
    # without this. Batched via the same rate-limited embed() used for chunks. ---
    entity_embed_texts = []
    for key in entity_keys:
        canonical = exact_name_counts[key].most_common(1)[0][0]
        desc = " | ".join(entity_descriptions[key])
        entity_embed_texts.append(f"{canonical}: {desc}" if desc else canonical)

    entity_vectors = []
    for i in tqdm(range(0, len(entity_embed_texts), EMBED_BATCH_SIZE), desc="embedding Entities"):
        batch_texts = entity_embed_texts[i : i + EMBED_BATCH_SIZE]
        vecs, _rec = embed(batch_texts, task_type="RETRIEVAL_DOCUMENT", pipeline="ingestion")
        entity_vectors.extend(vecs)

    # --- Upsert Entity vertices ---
    entity_vertices = []
    for key, vec in zip(entity_keys, entity_vectors):
        top_type = entity_types[key].most_common(1)[0][0]
        canonical = exact_name_counts[key].most_common(1)[0][0]
        entity_vertices.append(
            (
                entity_id_of[key],
                {
                    "name": canonical,
                    "canonical_name": canonical,
                    "entity_type": top_type,
                    "description": " | ".join(entity_descriptions[key]),
                    "mention_count": entity_mention_count[key],
                    "embedding": vec,
                },
            )
        )
    for i in tqdm(range(0, len(entity_vertices), UPSERT_BATCH_SIZE), desc="upserting Entities"):
        conn.upsertVertices("Entity", entity_vertices[i : i + UPSERT_BATCH_SIZE])

    # --- Upsert RELATED_TO edges ---
    related_to_edges = []
    for (a_id, b_id, relation_label), v in rel_agg.items():
        related_to_edges.append(
            (
                a_id,
                b_id,
                {
                    "relation_label": relation_label,
                    "description": v["description"],
                    "source_chunk_id": "",
                    "source_doc_id": v["source_doc_id"],
                    "source_url": v["source_url"],
                    "source_type": "wikipedia",
                    "observed_at": "",
                    "valid_from": "",
                    "valid_to": "",
                    "authority_score": 1.0,
                    "weight": v["weight"],
                },
            )
        )
    for i in tqdm(range(0, len(related_to_edges), UPSERT_BATCH_SIZE), desc="upserting RELATED_TO"):
        conn.upsertEdges("Entity", "RELATED_TO", "Entity", related_to_edges[i : i + UPSERT_BATCH_SIZE])

    # --- MENTIONS edges: re-chunk each doc locally, substring-match entity names ---
    mentions_edges = []
    for rec in tqdm(records, desc="building MENTIONS (local, no API calls)"):
        doc = corpus.get(rec["doc_id"])
        if not doc:
            continue
        chunks = chunk_document(doc["text"])
        names_in_doc = [(e["name"], entity_id_of[normalize_name(e["name"])]) for e in rec["entities"] if normalize_name(e["name"]) in entity_id_of]
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{rec['doc_id']}_c{i}"
            lower_chunk = chunk_text.lower()
            for name, ent_id in names_in_doc:
                count = lower_chunk.count(name.lower())
                if count > 0:
                    mentions_edges.append((chunk_id, ent_id, {"weight": count}))

    print(f"Built {len(mentions_edges)} MENTIONS edges.")
    for i in tqdm(range(0, len(mentions_edges), UPSERT_BATCH_SIZE), desc="upserting MENTIONS"):
        conn.upsertEdges("Chunk", "MENTIONS", "Entity", mentions_edges[i : i + UPSERT_BATCH_SIZE], vertexMustExist=False)

    print("\nDone.")


if __name__ == "__main__":
    main()
