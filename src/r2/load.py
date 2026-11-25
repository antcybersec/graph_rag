"""Build the Round 2 graph from corpus.jsonl. Idempotent, resumable, one command.

Everything structural is derived by `src.r2.parse` with no model involved, so a
rebuild is deterministic. The only model calls are (a) chunk embeddings, which
run locally via BGE and therefore need no API key or quota, and (b) the 39
cached claim extractions. A judge re-running `./build.sh corpus.jsonl` with the
cache present makes **zero** network calls to any LLM provider.

Upserts rather than loading jobs throughout: upsert is idempotent by primary
id, so a partial build can simply be re-run. Round 1 lost half a day to a
non-idempotent loader that produced 2,603 facts where 2,316 existed, with
duplicate champions in query results -- the fix there was to delete the layer
before writing, and the lesson here is to never need that.
"""
from __future__ import annotations

import argparse
import os
import time

from src.common.infobox import norm_key

BATCH = 1000


def _batched(rows, fn, label, size=BATCH):
    """Upsert in batches and report what the database actually accepted.

    The first version printed the number submitted and discarded the return
    value, which hid a load that silently dropped a third of the films and all
    96 claims. TigerGraph's upsert returns the count it accepted; anything less
    than submitted is a real failure and has to be loud, because every later
    number -- accuracy, citations, graph counts -- is computed off this.
    """
    accepted = 0
    for i in range(0, len(rows), size):
        batch = rows[i:i + size]
        try:
            got = fn(batch)
            accepted += int(got) if isinstance(got, (int, float)) else 0
        except Exception as exc:
            print(f"  !! {label}: batch at {i} rejected: {str(exc)[:220]}")
    flag = "" if accepted >= len(rows) else f"   <-- SHORT by {len(rows) - accepted}"
    print(f"  {label}: {accepted}/{len(rows)} accepted{flag}")
    return accepted


def load_structure(conn, records: dict) -> dict:
    """Events, medals, films, people and the vertices they hang off."""
    events = records["events"]
    counts = {}

    games = {(e["games_id"], e["games_year"], e["season"]) for e in events}
    _batched([(g[0], {"year": g[1], "season": g[2]}) for g in games],
             lambda b: conn.upsertVertices("Games", b), "Games")

    sports = {(e["sport_key"], e["sport"]) for e in events}
    _batched([(s[0], {"name": s[1]}) for s in sports],
             lambda b: conn.upsertVertices("Sport", b), "Sport")

    # Keyed by venue_key, not a set of pairs: several display spellings
    # normalize to one key (the corpus has "TechnologyUniversity" alongside
    # "Technology University"), and submitting both makes the accepted count
    # look short when nothing was lost.
    venues = {}
    for e in events:
        if e["venue_key"]:
            venues.setdefault(e["venue_key"], e["venue"])
    _batched([(k, {"name": v}) for k, v in venues.items()],
             lambda b: conn.upsertVertices("Venue", b), "Venue")

    series = {}
    for e in events:
        series.setdefault(e["series_key"], (e["sport_key"], e["event_name"], e["event_key"]))
    _batched([(k, {"sport_key": v[0], "event_name": v[1], "event_key": v[2]})
              for k, v in series.items()],
             lambda b: conn.upsertVertices("EventSeries", b), "EventSeries")

    _batched([(e["event_id"], {
                  "title": e["title"], "sport_key": e["sport_key"],
                  "event_name": e["event_name"], "event_key": e["event_key"],
                  "series_key": e["series_key"], "games_year": e["games_year"],
                  "season": e["season"], "venue": e["venue"],
                  "venue_key": e["venue_key"], "date_text": e["date_text"],
                  # -1 rather than 0 for "the infobox did not say": a real
                  # count of 0 and a missing count must stay distinguishable,
                  # or "fewest nations" silently selects the unknowns.
                  "competitors": e["competitors"] if e["competitors"] is not None else -1,
                  "nations": e["nations"] if e["nations"] is not None else -1,
                  "teams": e["teams"] if e["teams"] is not None else -1,
                  "win_value": e["win_value"],
                  "prev_year": e["prev_year"] or 0,
                  "next_year": e["next_year"] or 0,
                  "synthetic": e["synthetic"]})
              for e in events],
             lambda b: conn.upsertVertices("OlympicEvent", b), "OlympicEvent")

    _batched([(e["event_id"], e["games_id"], {}) for e in events],
             lambda b: conn.upsertEdges("OlympicEvent", "IN_GAMES", "Games", b), "IN_GAMES")
    _batched([(e["event_id"], e["sport_key"], {}) for e in events],
             lambda b: conn.upsertEdges("OlympicEvent", "IN_SPORT", "Sport", b), "IN_SPORT")
    _batched([(e["event_id"], e["venue_key"], {}) for e in events if e["venue_key"]],
             lambda b: conn.upsertEdges("OlympicEvent", "HELD_AT", "Venue", b), "HELD_AT")
    _batched([(e["event_id"], e["series_key"], {}) for e in events],
             lambda b: conn.upsertEdges("OlympicEvent", "INSTANCE_OF", "EventSeries", b), "INSTANCE_OF")

    # PREV_EDITION from the corpus's own prev field, matched within the series.
    by_series_year = {(e["series_key"], e["games_year"]): e["event_id"] for e in events}
    prev_edges = [(e["event_id"], by_series_year[(e["series_key"], e["prev_year"])], {})
                  for e in events
                  if e["prev_year"] and (e["series_key"], e["prev_year"]) in by_series_year]
    _batched(prev_edges,
             lambda b: conn.upsertEdges("OlympicEvent", "PREV_EDITION", "OlympicEvent", b),
             "PREV_EDITION")

    # Medals: every line by NOC, individual lines additionally by athlete.
    nations, medal_edges, athletes, won_edges = set(), [], {}, []
    for e in events:
        for m in e["medals"]:
            if m["noc"]:
                nations.add(m["noc"])
                medal_edges.append((m["noc"], e["event_id"], {
                    "medal": m["medal"], "recipient": m["recipient"],
                    "rank": m["rank"], "multi": m["multi"]}))
            if not m["multi"] and m["recipient_key"]:
                athletes[m["recipient_key"]] = m["recipient"]
                won_edges.append((m["recipient_key"], e["event_id"], {
                    "medal": m["medal"], "rank": m["rank"], "noc": m["noc"]}))
    _batched([(n, {}) for n in nations], lambda b: conn.upsertVertices("Nation", b), "Nation")
    _batched([(k, {"name": v}) for k, v in athletes.items()],
             lambda b: conn.upsertVertices("Athlete", b), "Athlete")
    _batched(medal_edges,
             lambda b: conn.upsertEdges("Nation", "MEDAL_FOR", "OlympicEvent", b), "MEDAL_FOR")
    _batched(won_edges,
             lambda b: conn.upsertEdges("Athlete", "WON", "OlympicEvent", b), "WON")
    counts["medal_lines"] = len(medal_edges)
    counts["individual_medals"] = len(won_edges)

    # Films, and a Person for each director (most have no article of their own).
    films = records["films"]
    _batched([(f["film_id"], {
                  "name": f["name"], "title": f["title"], "year": f["year"] or 0,
                  "released": f["released"],
                  "budget": f["budget"] or 0.0, "gross": f["gross"] or 0.0,
                  "budget_currency": f["budget_currency"],
                  "gross_currency": f["gross_currency"]})
              for f in films],
             lambda b: conn.upsertVertices("Film", b), "Film")

    people = {p["person_id"]: p for p in records["people"]}
    person_vertices = [(p["person_id"], {"name": p["name"], "name_key": norm_key(p["name"]),
                                         "kind": p["kind"], "party": p["party"]})
                       for p in people.values()]
    director_ids = {}
    for f in films:
        for d in f["directors"]:
            pid = f"name:{norm_key(d)}"
            if norm_key(d) and pid not in director_ids:
                director_ids[pid] = d
    person_vertices += [(pid, {"name": name, "name_key": norm_key(name),
                               "kind": "credit", "party": ""})
                        for pid, name in director_ids.items()]
    _batched(person_vertices, lambda b: conn.upsertVertices("Person", b), "Person")
    _batched([(f"name:{norm_key(d)}", f["film_id"], {})
              for f in films for d in f["directors"] if norm_key(d)],
             lambda b: conn.upsertEdges("Person", "DIRECTED_FILM", "Film", b), "DIRECTED_FILM")

    offices, office_edges = {}, []
    for p in people.values():
        for o in p["offices"]:
            # An unlabelled block still needs identity: key it to the person
            # and slot. Angus King's governorship has no office name at all,
            # and dropping it would lose pub-019 entirely.
            okey = norm_key(o["office"]) or f"{p['person_id']}#slot{o['slot']}"
            offices[okey] = o["office"] or f"(unnamed office, block {o['slot']})"
            office_edges.append((p["person_id"], okey, {
                "slot": o["slot"], "term_start": o["term_start"],
                "term_end": o["term_end"], "predecessor": o["predecessor"],
                "successor": o["successor"]}))
    _batched([(k, {"name": v}) for k, v in offices.items()],
             lambda b: conn.upsertVertices("Office", b), "Office")
    _batched(office_edges,
             lambda b: conn.upsertEdges("Person", "HELD_OFFICE", "Office", b), "HELD_OFFICE")
    return counts


def load_documents(conn, docs: list, records: dict) -> None:
    """Document vertices, plus the edges tying them to what they describe."""
    event_ids = {e["event_id"] for e in records["events"]}
    film_ids = {f["film_id"] for f in records["films"]}
    person_ids = {p["person_id"] for p in records["people"]}

    def doc_class(d):
        if d["doc_id"] in event_ids:
            return "event"
        if d["doc_id"] in film_ids:
            return "film"
        if d["doc_id"] in person_ids:
            return "person"
        return "source_record" if d.get("synthetic") else "text"

    _batched([(d["doc_id"], {"title": d["title"], "url": d.get("url") or "",
                             "doc_class": doc_class(d),
                             "synthetic": bool(d.get("synthetic")),
                             "approx_tokens": d.get("approx_tokens") or 0})
              for d in docs],
             lambda b: conn.upsertVertices("Document", b), "Document")
    for label, ids, etype, vtype in (("DESCRIBES_EVENT", event_ids, "DESCRIBES_EVENT", "OlympicEvent"),
                                     ("DESCRIBES_FILM", film_ids, "DESCRIBES_FILM", "Film"),
                                     ("DESCRIBES_PERSON", person_ids, "DESCRIBES_PERSON", "Person")):
        _batched([(i, i, {}) for i in ids],
                 lambda b, e=etype, v=vtype: conn.upsertEdges("Document", e, v, b), label)


def load_chunks(conn, docs: list, batch_size: int = 64) -> int:
    """Chunk every document, embed locally, and upsert with the vector.

    Embeddings are BGE on-device: no API key, no quota, no rate limit. This is
    the step that made Round 1 survivable after the Gemini quota wall, and it
    is why a judge can rebuild this graph with no credentials at all.
    """
    from src.common.chunking import chunk_document
    from src.common.llm import embed

    pending, n = [], 0
    for d in docs:
        for i, text in enumerate(chunk_document(d["text"])):
            pending.append((f"{d['doc_id']}_c{i}", d["doc_id"], i, text))
    print(f"  chunks to embed: {len(pending)}")

    for i in range(0, len(pending), batch_size):
        batch = pending[i:i + batch_size]
        vectors, _ = embed([b[3] for b in batch], task_type="RETRIEVAL_DOCUMENT",
                           pipeline="r2_load")
        conn.upsertVertices("Chunk", [
            (cid, {"doc_id": did, "ord": ord_, "text": text, "embedding": vec})
            for (cid, did, ord_, text), vec in zip(batch, vectors)])
        conn.upsertEdges("Document", "HAS_CHUNK", "Chunk",
                         [(did, cid, {}) for cid, did, _, _ in batch])
        n += len(batch)
        if (i // batch_size) % 20 == 0:
            print(f"    {n}/{len(pending)} chunks")
    return n


def load_evidence(conn, records: dict) -> int:
    """SourceRecord and Claim vertices, with ASSERTED_BY / ABOUT_EVENT."""
    from src.r2.claims import build_claims, ground_claims

    sources = records["sources"]
    # published_at is UTC, and 13 of the 39 LA 2028 source records were
    # published in the evening Pacific -- so their UTC date is the *next* day.
    # The questions use the date the document itself prints, and the answer
    # key does too ("a provisional July 23 scoreboard"), so the local string
    # the parser already keeps has to reach the model. Without it the jury's
    # certificate and the blog that recycled it both read as 24 July and the
    # chronology the question turns on disappears.
    _batched([(s["source_id"], {"title": s["title"], "source_text": s["source_text"],
                                "published_at": s["published_at"] or "",
                                "published_local": s.get("published_raw") or "",
                                "tier": 0, "tier_name": "", "tier_reason": ""})
              for s in sources],
             lambda b: conn.upsertVertices("SourceRecord", b), "SourceRecord")

    claims = ground_claims(build_claims(sources), records["events"])
    from src.r2.authority import classify_source
    _batched([(s["source_id"], {"tier": classify_source(s["source_text"])["tier"],
                                "tier_name": classify_source(s["source_text"])["tier_name"],
                                "tier_reason": classify_source(s["source_text"])["reason"]})
              for s in sources],
             lambda b: conn.upsertVertices("SourceRecord", b), "SourceRecord tiers")

    _batched([(c["claim_id"], {
                  "subject": c["subject"], "subject_key": c["subject_key"],
                  "slot_subject": c["slot_subject"], "predicate": c["predicate"],
                  "value": c["value"], "value_key": c["value_key"],
                  "population": c["population"], "stance": c["stance"],
                  "status": c["status"], "effective": c["effective"],
                  "effective_sort": c["effective_sort"], "tier": c["tier"],
                  "source_id": c["source_id"], "event_id": c["event_id"]})
              for c in claims],
             lambda b: conn.upsertVertices("Claim", b), "Claim")
    _batched([(c["claim_id"], c["source_id"], {}) for c in claims],
             lambda b: conn.upsertEdges("Claim", "ASSERTED_BY", "SourceRecord", b), "ASSERTED_BY")
    _batched([(c["claim_id"], c["event_id"], {}) for c in claims if c["event_id"]],
             lambda b: conn.upsertEdges("Claim", "ABOUT_EVENT", "OlympicEvent", b), "ABOUT_EVENT")

    # Materialise the supersession the resolver computes, so the graph itself
    # records which claim beat which. Resolution still happens at query time
    # (an as-of question needs a different winner), but these edges make the
    # reasoning inspectable -- and on screen in the judging call they show the
    # signed jury decision overriding the provisional board as a graph fact.
    from src.r2.claims import group_by_slot, resolve
    edges = []
    for slot_claims in group_by_slot(claims).values():
        r = resolve(slot_claims)
        for w in r["winners"]:
            for o in r["others"]:
                if o.get("why_not") != "agrees":
                    edges.append((w["claim_id"], o["claim_id"], {}))
    _batched(edges, lambda b: conn.upsertEdges("Claim", "SUPERSEDES", "Claim", b), "SUPERSEDES")
    return len(claims)


def graph_counts(conn) -> dict:
    """Vertex and edge counts by type -- submission deliverable 6."""
    return {"vertices": conn.getVertexCount("*"), "edges": conn.getEdgeCount("*")}


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the Round 2 TigerGraph graph")
    ap.add_argument("corpus")
    ap.add_argument("--skip-schema", action="store_true")
    ap.add_argument("--skip-chunks", action="store_true",
                    help="structure and evidence only; the text path stays empty")
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from src.r2.conn import describe, get_connection
    from src.r2 import queries, schema
    from src.r2.parse import classify, iter_docs

    t0 = time.time()
    docs = list(iter_docs(args.corpus))
    records = classify(docs)
    print(f"parsed {len(docs)} docs -> " +
          ", ".join(f"{k} {len(v)}" for k, v in records.items()))

    print(f"connecting: {describe()}")
    conn = get_connection()
    if not args.skip_schema:
        print("applying schema...")
        schema.apply(conn, schema.SCHEMA_JOB, "schema job")
        schema.apply(conn, schema.VECTOR_JOB, "vector job")
    else:
        # build.sh promises it is safe to re-run, which has to include a graph
        # built before an attribute existed. A fresh graph gets published_local
        # from the vertex definition; an existing one needs the additive job,
        # which touches SourceRecord alone and never Chunk.
        print(f"SourceRecord.published_local: {schema.ensure_published_local(conn)}")
        print("installing queries...")
        queries.install(conn)

    print("loading structure...")
    load_structure(conn, records)
    print("loading documents...")
    load_documents(conn, docs, records)
    print("loading evidence...")
    n = load_evidence(conn, records)
    print(f"  claims: {n}")
    if not args.skip_chunks:
        print("loading chunks (local BGE embeddings)...")
        load_chunks(conn, docs)

    counts = graph_counts(conn)
    print("\n=== graph counts by type ===")
    for kind in ("vertices", "edges"):
        for name, count in sorted(counts[kind].items()):
            print(f"  {kind[:-1]:6s} {name:20s} {count}")
    print(f"\nbuilt in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
