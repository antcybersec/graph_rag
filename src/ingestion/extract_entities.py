"""LLM-based entity + relationship extraction, one call per DOCUMENT (not
per chunk) -- see docs/architecture.md for why: with ~2951 docs averaging
~1852 tokens, per-chunk extraction (~10k+ chunks) would multiply LLM calls
~4x for no accuracy benefit at this corpus size, since Gemini's context
window easily holds a full document. Chunking is still done separately
(src/common/chunking.py) purely for vector-retrieval granularity.

Uses EXTRACTION_MODEL (a cheap/lite model, thinking disabled) since this
runs once over the whole corpus at ingestion time -- not part of any
pipeline's runtime cost, so it's excluded from per-question token metrics.
"""
import os
import json
from typing import Optional

from pydantic import BaseModel, Field

from src.common.llm import generate

EXTRACTION_MODEL = os.environ.get("EXTRACTION_MODEL", "gemini-3.5-flash-lite")

SYSTEM_INSTRUCTION = """You extract entities and relationships from a document for a knowledge graph.
Be precise and conservative: only extract entities and relationships that are explicitly stated or
directly and unambiguously implied by the text. Do not invent facts.

Entity types to use (pick the closest one, or "Other"): Person, Organization, Country, Team,
Venue, Event, Competition, Work (film/book/etc.), Concept, Other.

Two rules that matter more than anything else, because this graph will later be merged across
thousands of documents:
1. NEVER combine two people into one entity. If a source names a duo, pair, or relay team (e.g.
   "Rudolf Dombi and Roland Kökény", "Judith Arndt/Trixi Worrack"), extract each person as their
   own separate Person entity, and add a relationship between them (e.g. "partnered with" /
   "teammate of") if the text presents them as a unit.
2. Always use the full, canonical, unabbreviated name for countries and organizations (e.g.
   "Germany" not "GER", "United States" not "USA" or "USA team", "Russia" not "RUS"), so the same
   real-world entity mentioned in different documents merges into one node.

For each relationship, write a short natural-language description that could stand alone as a
fact (it will later be shown as evidence, with a citation back to this document)."""

EXTRACTION_PROMPT_TEMPLATE = """Document title: {title}
Document text:
---
{text}
---

Extract:
1. entities: every distinct named entity (people, organizations, countries, teams, venues, events,
   competitions, works, or important concepts). Use the FULL name as it appears in the text as
   `name`, plus a one-line `description` grounded in this document, and an `entity_type`.
2. relationships: pairs of entities from your own list above that are directly related according to
   this document, with a short `relation_label` (2-4 words, e.g. "won gold medal at", "is capital of",
   "competed against") and a one-sentence `description` of the specific fact."""


class ExtractedEntity(BaseModel):
    name: str
    entity_type: str
    description: str


class ExtractedRelationship(BaseModel):
    source: str = Field(description="must match an entity `name` from the entities list")
    target: str = Field(description="must match an entity `name` from the entities list")
    relation_label: str
    description: str


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity]
    relationships: list[ExtractedRelationship]


def extract_from_document(doc: dict, tracker=None) -> ExtractionResult:
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(title=doc["title"], text=doc["text"])
    result, _record = generate(
        prompt,
        model=EXTRACTION_MODEL,
        system_instruction=SYSTEM_INSTRUCTION,
        response_schema=ExtractionResult,
        pipeline="ingestion",
        call_type="extraction",
        tracker=tracker,
        question_id=doc.get("doc_id"),
    )
    return result


# --- Batched extraction: pack several documents into one LLM call -----------
# Free-tier Gemini quotas cap *request count* per day far more aggressively
# than token volume, so the single highest-leverage fix for ingesting ~2951
# docs is fewer, larger calls rather than many small ones. Batching by a
# token budget (not a fixed doc count) naturally gives an outlier-sized
# document its own solo batch instead of blowing up one call's output size.

class DocExtraction(BaseModel):
    doc_id: str
    entities: list[ExtractedEntity]
    relationships: list[ExtractedRelationship]


class BatchExtractionResult(BaseModel):
    documents: list[DocExtraction]


BATCH_SYSTEM_INSTRUCTION = SYSTEM_INSTRUCTION + """

You will be given MULTIPLE documents in one request, each wrapped in
=== DOCUMENT doc_id=... === ... === END DOCUMENT === markers. Extract entities and
relationships SEPARATELY for each document -- never let an entity or relationship from one
document leak into another's list, even if they mention the same real-world thing. Return one
`DocExtraction` per input document, in the same order, each carrying the exact `doc_id` given."""


def build_batch_prompt(docs: list) -> str:
    parts = []
    for d in docs:
        parts.append(
            f"=== DOCUMENT doc_id={d['doc_id']} ===\nTitle: {d['title']}\n{d['text']}\n=== END DOCUMENT {d['doc_id']} ==="
        )
    return "\n\n".join(parts)


def extract_from_documents_batch(docs: list, tracker=None) -> dict:
    """Returns {doc_id: DocExtraction} for whichever doc_ids the model returned
    (caller is responsible for detecting/retrying any doc_id missing from the result)."""
    prompt = build_batch_prompt(docs)
    result, _record = generate(
        prompt,
        model=EXTRACTION_MODEL,
        system_instruction=BATCH_SYSTEM_INSTRUCTION,
        response_schema=BatchExtractionResult,
        pipeline="ingestion",
        call_type="extraction",
        tracker=tracker,
        question_id=f"batch_{docs[0]['doc_id']}_{len(docs)}docs",
    )
    return {d.doc_id: d for d in result.documents}


def make_token_budget_batches(docs: list, target_tokens: int = 3500, max_docs: int = 10) -> list:
    batches, current, current_tokens = [], [], 0
    for d in docs:
        t = d.get("approx_tokens", 500)
        if current and (current_tokens + t > target_tokens or len(current) >= max_docs):
            batches.append(current)
            current, current_tokens = [], 0
        current.append(d)
        current_tokens += t
    if current:
        batches.append(current)
    return batches


if __name__ == "__main__":
    import sys

    docs = []
    with open("data/raw_dataset/corpus/corpus.jsonl") as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            docs.append(json.loads(line))

    for doc in docs:
        print(f"\n{'='*70}\nDOC: {doc['title']} ({doc['doc_id']}, {doc['approx_tokens']} tokens)")
        res = extract_from_document(doc)
        print(f"  {len(res.entities)} entities, {len(res.relationships)} relationships")
        for e in res.entities[:8]:
            print(f"    ENTITY: [{e.entity_type}] {e.name} -- {e.description}")
        if len(res.entities) > 8:
            print(f"    ... and {len(res.entities) - 8} more")
        for r in res.relationships[:8]:
            print(f"    REL: {r.source} --[{r.relation_label}]--> {r.target} -- {r.description}")
        if len(res.relationships) > 8:
            print(f"    ... and {len(res.relationships) - 8} more")
