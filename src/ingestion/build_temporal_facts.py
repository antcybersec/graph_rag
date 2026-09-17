"""Round 2 groundwork: a time-scoped fact layer with provenance, built from the
corpus with no LLM calls.

The Round 2 challenge is "reasoning over evolving, conflicting, and uncertain
facts". A fact like "President of Russia" or "Olympic champion in the women's
500 m speed skating" is only true *for an interval*, and the corpus states such
facts in several articles that need not agree. This module extracts them as
first-class, interval-stamped, sourced facts:

    TemporalFact(subject, predicate, object, valid_from, valid_to,
                 source_doc_id, source_url, authority_score, observed_at)
    TemporalFact -FACT_SOURCE-> Document          (provenance, auditable)

Two real sources of evolving facts in this corpus (measured, not assumed):
  * 129 office terms from `officeholder`/`person` infoboxes, each with
    term_start/term_end -- so "who held office X on date D" is a genuine
    interval query, and consecutive holders of one office form a timeline.
  * 340 Olympic event series that span three or more Games -- the reigning
    champion of an event is valid from one Games until the next, so
    "who was the reigning champion of E as of D" changes over time.

Conflicts are detected rather than silently resolved: two facts about the same
(subject, predicate) whose intervals overlap with different objects are
returned together, ranked by authority_score then recency, so the agent can
report the disagreement instead of guessing. Dates are stored as sortable
YYYYMMDD integers (0 = unknown start, 99991231 = still open).

Usage (safe to re-run):
  python -m src.ingestion.build_temporal_facts [--dry-run] [--queries-only]
"""
import argparse
import json
import re
from collections import defaultdict

from tqdm import tqdm

from src.common.infobox import TITLE_RE, norm_key, parse_infobox
from src.common.tg_conn import get_connection

GRAPH_NAME = "test"
CORPUS_PATH = "data/raw_dataset/corpus/corpus.jsonl"
UPSERT_BATCH_SIZE = 500

DATE_OPEN = 99991231  # still valid ("present")
DATE_UNKNOWN = 0

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june",
     "july", "august", "september", "october", "november", "december"])}

SCHEMA_JOB = f"""
USE GRAPH {GRAPH_NAME}

CREATE SCHEMA_CHANGE JOB add_temporal_layer FOR GRAPH {GRAPH_NAME} {{
    ADD VERTEX TemporalFact (
        PRIMARY_ID fact_id STRING,
        subject STRING,
        subject_key STRING,
        predicate STRING,
        object STRING,
        object_key STRING,
        slot_key STRING,
        value_key STRING,
        valid_from INT,
        valid_to INT,
        source_doc_id STRING,
        source_url STRING,
        authority_score FLOAT,
        observed_at STRING
    ) WITH PRIMARY_ID_AS_ATTRIBUTE="true";

    ADD UNDIRECTED EDGE FACT_SOURCE (FROM TemporalFact, TO Document);
}}

RUN SCHEMA_CHANGE JOB add_temporal_layer
"""

QUERIES = {
    # "Who held <office> on <date>?" -- the object is known, the subject is asked for.
    "facts_as_of": f"""
USE GRAPH {GRAPH_NAME}

CREATE OR REPLACE QUERY facts_as_of(STRING predicate, STRING subject_key, STRING object_key, INT as_of) FOR GRAPH {GRAPH_NAME} SYNTAX v3 {{
  // Any of subject_key/object_key may be empty, meaning "don't filter on it":
  // the same query answers "what did X hold then" and "who held Y then".
  Facts = SELECT f FROM TemporalFact:f
          WHERE (predicate == "" OR f.predicate == predicate)
            AND (subject_key == "" OR f.subject_key == subject_key)
            AND (object_key == "" OR f.object_key == object_key)
            AND f.valid_from <= as_of AND f.valid_to >= as_of;
  PRINT Facts;
}}

INSTALL QUERY facts_as_of
""",
    # The full history of one subject/object, for "how did this change over time".
    "fact_timeline": f"""
USE GRAPH {GRAPH_NAME}

CREATE OR REPLACE QUERY fact_timeline(STRING predicate, STRING subject_key, STRING object_key) FOR GRAPH {GRAPH_NAME} SYNTAX v3 {{
  Facts = SELECT f FROM TemporalFact:f
          WHERE (predicate == "" OR f.predicate == predicate)
            AND (subject_key == "" OR f.subject_key == subject_key)
            AND (object_key == "" OR f.object_key == object_key)
          ORDER BY f.valid_from ASC;
  PRINT Facts;
}}

INSTALL QUERY fact_timeline
""",
}


def parse_date(text: str) -> tuple:
    """(YYYYMMDD, precision) for the messy date strings Wikipedia infoboxes use.

    Handles "7 May 2012", "March 2, 2017", "2001-06-15", "1999" and "present".
    Unknown -> DATE_UNKNOWN; open-ended -> DATE_OPEN. Precision ("day"/"month"/
    "year"/"none") is kept so an answer can say how exact its interval is."""
    t = (text or "").strip().lower()
    if not t:
        return DATE_UNKNOWN, "none"
    if "present" in t or "incumbent" in t:
        return DATE_OPEN, "open"
    # Day counts per month; February is given 29 because resolving leap years
    # would buy nothing here -- these values only have to order correctly.
    _DAYS_IN_MONTH = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

    def valid(month: int, day: int) -> bool:
        # Wikipedia infoboxes carry typos ("31 February 2012", "2012-13-45");
        # storing them as if real would put impossible dates in the fact layer.
        return 1 <= month <= 12 and 1 <= day <= _DAYS_IN_MONTH[month - 1]

    if m := re.search(r"(\d{4})-(\d{2})-(\d{2})", t):
        if valid(int(m[2]), int(m[3])):
            return int(f"{m[1]}{m[2]}{m[3]}"), "day"
    if m := re.search(r"(\d{1,2})\s+([a-z]+)\s+(\d{4})", t):
        if m[2] in _MONTHS and valid(_MONTHS[m[2]], int(m[1])):
            return int(f"{m[3]}{_MONTHS[m[2]]:02d}{int(m[1]):02d}"), "day"
    if m := re.search(r"([a-z]+)\s+(\d{1,2}),?\s*(\d{4})", t):
        if m[1] in _MONTHS and valid(_MONTHS[m[1]], int(m[2])):
            return int(f"{m[3]}{_MONTHS[m[1]]:02d}{int(m[2]):02d}"), "day"
    if m := re.search(r"([a-z]+)\s+(\d{4})", t):
        if m[1] in _MONTHS:
            return int(f"{m[2]}{_MONTHS[m[1]]:02d}01"), "month"
    if m := re.search(r"(\d{4})", t):
        return int(f"{m[1]}0101"), "year"
    return DATE_UNKNOWN, "none"


def _fact(subject, predicate, obj, valid_from, valid_to, doc, authority, note="", slot="subject"):
    """`slot` names which side is the single-occupancy role at a given moment.

    It differs by predicate and cannot be inferred from position: an office is
    held by one person at a time (slot = the object), while an event series has
    one reigning champion at a time (slot = the subject). Conflict detection is
    meaningless without this -- comparing the wrong side turns "won the same
    event twice" or "won two different events" into fake contradictions.
    """
    fid = norm_key(f"{subject}|{predicate}|{obj}|{valid_from}").replace(" ", "_")[:180]
    subject_key, object_key = norm_key(subject), norm_key(obj)
    return {
        "fact_id": fid,
        "subject": subject, "subject_key": subject_key,
        "predicate": predicate,
        "object": obj, "object_key": object_key,
        "slot_key": subject_key if slot == "subject" else object_key,
        "value_key": object_key if slot == "subject" else subject_key,
        "valid_from": valid_from, "valid_to": valid_to,
        "source_doc_id": doc["doc_id"], "source_url": doc.get("url", ""),
        # Every fact here is read straight out of an infobox, so they share one
        # authority tier; the field exists so a second, less reliable source
        # (LLM-extracted prose, say) can be added below it without a schema change.
        "authority_score": authority,
        "observed_at": note,
    }


def extract_facts(corpus_path: str = CORPUS_PATH) -> list:
    docs = [json.loads(l) for l in open(corpus_path)]
    facts = []

    # --- office terms: "<person> held <office> from <start> to <end>" ---
    for d in docs:
        f = parse_infobox(d["text"]) or {}
        if not f:
            # parse_infobox only returns the Olympic box; re-read the first box here
            m = re.match(r"\[Infobox ([^\]]+)\]\n((?:  [^\n]*\n)+)", d["text"][:4000])
            if not m or m[1] not in ("officeholder", "person", "writer", "scientist"):
                continue
            f = {l.strip().split(":", 1)[0]: l.strip().split(":", 1)[1].strip()
                 for l in m[2].splitlines() if ":" in l}
        name = f.get("name") or d["title"]
        for suffix in ("", "1", "2", "3", "4", "5", "6"):
            office, start = f.get("office" + suffix), f.get("term_start" + suffix)
            if not (office and start):
                continue
            vf, prec = parse_date(start)
            vt, _ = parse_date(f.get("term_end" + suffix, "")) if f.get("term_end" + suffix) else (DATE_OPEN, "open")
            # One office, one holder at a time -> the office is the slot.
            facts.append(_fact(name, "held_office", office, vf, vt or DATE_OPEN, d, 1.0,
                               f"date precision: {prec}", slot="object"))

    # --- Olympic champions: valid from one Games until the next in that series ---
    series = defaultdict(list)
    for d in docs:
        t = TITLE_RE.match(d["title"])
        if not t:
            continue
        sport, year, season, event = t.groups()
        box = parse_infobox(d["text"])
        if box.get("gold"):
            # A reign starts when the event was actually contested, not at year end:
            # the 2018 Winter Games finished in February, so a 31-December start
            # would report the 2014 champion for the first ten months of 2018.
            date_text = box.get("date") or box.get("dates") or ""
            if date_text and not re.search(r"\d{4}", date_text):
                date_text = f"{date_text} {year}"  # infobox dates often omit the year
            starts, _precision = parse_date(date_text)
            # Trust the date only if it lands on (or next to) the Games year. The
            # tolerance of one year is real: Tokyo 2020 was held in 2021, and 45
            # event articles correctly carry 2021 dates. Beyond that it is a typo
            # -- Badminton 2008 women's doubles reads "10 August to 15 August
            # 2012", which otherwise gives the 2008 champions a reign overlapping
            # the 2012 champions', i.e. a contradiction invented by a bad parse.
            if starts and starts != DATE_OPEN and abs(int(str(starts)[:4]) - int(year)) > 1:
                starts = 0
            if not starts or starts == DATE_OPEN:
                # No usable date: fall back to roughly when those Games are held.
                starts = int(f"{year}0801") if season == "Summer" else int(f"{year}0201")
            series[(sport, event, season)].append((int(year), starts, box["gold"], d))
    for (sport, event, season), editions in series.items():
        editions.sort()
        for i, (year, starts, gold, d) in enumerate(editions):
            next_starts = editions[i + 1][1] if i + 1 < len(editions) else None
            # next_starts - 1 can read like 20180200; that is deliberate, and sorts
            # exactly between 20180131 and 20180201 -- "the instant before the next
            # Games" -- which is all these integer comparisons need.
            valid_to = (next_starts - 1) if next_starts else DATE_OPEN
            # One event series has one reigning champion at a time -> the series is the slot.
            facts.append(_fact(
                f"{sport} – {event}", "olympic_champion", gold, starts, valid_to,
                d, 1.0, f"{season} {year} Games; reigning champion until the next Games in the corpus",
                slot="subject",
            ))
    return facts


# Offices many people hold at once (legislatures, boards): two simultaneous
# holders there are normal, not a contradiction.
_MULTI_HOLDER_RE = re.compile(r"^(member|members|delegate|representative|senator|judge|justice)\b", re.I)


def find_conflicts(facts: list) -> dict:
    """Genuine disagreements, kept apart from things that merely look like them.

    Three buckets, because conflating them is how a demo starts claiming
    contradictions that aren't there:

      cross_source   two DIFFERENT documents assert the same
                     (subject, predicate, object) over different intervals --
                     a real source disagreement about when a fact held.
      rival_claims   two subjects hold the same single-holder office at once --
                     a real contradiction unless the office is a legislature
                     seat (filtered by _MULTI_HOLDER_RE).
      concurrent     one subject, several objects, overlapping intervals --
                     NOT a conflict (Putin was President and party chairman at
                     the same time); surfaced only as context.
    """
    out = {"cross_source": [], "rival_claims": [], "concurrent": []}

    def overlaps(a, b):
        return a["valid_from"] <= b["valid_to"] and b["valid_from"] <= a["valid_to"]

    by_slot = defaultdict(list)
    by_value = defaultdict(list)
    for f in facts:
        by_slot[(f["predicate"], f["slot_key"])].append(f)
        by_value[(f["predicate"], f["value_key"])].append(f)

    for (predicate, slot_key), group in by_slot.items():
        if predicate == "held_office" and _MULTI_HOLDER_RE.match(slot_key):
            continue
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                if not overlaps(a, b):
                    continue  # consecutive holders/reigns are the normal case, not a conflict
                if a["value_key"] != b["value_key"]:
                    out["rival_claims"].append(((predicate, slot_key), a, b))
                elif a["source_doc_id"] != b["source_doc_id"] and (a["valid_from"], a["valid_to"]) != (b["valid_from"], b["valid_to"]):
                    # same slot, same value, two documents, different intervals
                    out["cross_source"].append(((predicate, slot_key), a, b))

    for (predicate, value_key), group in by_value.items():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                if a["slot_key"] != b["slot_key"] and overlaps(a, b):
                    out["concurrent"].append(((predicate, value_key), a, b))
    return out


def load(conn, facts: list):
    if "TemporalFact" not in conn.getVertexTypes():
        print(conn.gsql(SCHEMA_JOB))
        conn = get_connection()
    else:
        # Replace the layer rather than adding to it. `fact_id` embeds
        # `valid_from`, so re-running after an interval correction mints NEW ids
        # and a plain upsert leaves the previous generation in place: queries
        # then return one champion twice, once per generation (observed
        # 2026-09-17 -- 2,603 vertices where 2,316 were expected). Deleting
        # first also drops the incident FACT_SOURCE edges.
        deleted = conn.delVertices("TemporalFact")
        print(f"replaced existing temporal layer: deleted {deleted} stale facts")
    attrs = ("subject", "subject_key", "predicate", "object", "object_key", "slot_key", "value_key",
             "valid_from", "valid_to", "source_doc_id", "source_url", "authority_score", "observed_at")
    vertices = [(f["fact_id"], {k: f[k] for k in attrs}) for f in facts]
    for i in tqdm(range(0, len(vertices), UPSERT_BATCH_SIZE), desc="TemporalFact"):
        conn.upsertVertices("TemporalFact", vertices[i:i + UPSERT_BATCH_SIZE])
    edges = [(f["fact_id"], f["source_doc_id"], {}) for f in facts if f["source_doc_id"]]
    for i in tqdm(range(0, len(edges), UPSERT_BATCH_SIZE), desc="FACT_SOURCE"):
        conn.upsertEdges("TemporalFact", "FACT_SOURCE", "Document", edges[i:i + UPSERT_BATCH_SIZE])
    return conn


def install_queries(conn):
    # CREATE OR REPLACE, not CREATE: a plain CREATE fails on re-run with
    # "the query name is used by another object" and leaves the OLD body
    # installed, so an edited query would silently never take effect.
    for name, gsql in QUERIES.items():
        print(f"\n=== installing {name} ===")
        result = conn.gsql(gsql)
        print(result[-600:] if isinstance(result, str) else result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--queries-only", action="store_true")
    args = parser.parse_args()

    if args.queries_only:
        install_queries(get_connection())
        return

    facts = extract_facts()
    preds = defaultdict(int)
    for f in facts:
        preds[f["predicate"]] += 1
    print(f"{len(facts)} temporal facts: {dict(preds)}")
    conflicts = find_conflicts(facts)
    print(f"cross-source disagreements: {len(conflicts['cross_source'])} | "
          f"rival claims on one slot: {len(conflicts['rival_claims'])} | "
          f"concurrent roles held by one person (not conflicts): {len(conflicts['concurrent'])}")
    for bucket in ("cross_source", "rival_claims"):
        for key, a, b in conflicts[bucket][:3]:
            print(f"  [{bucket}] {key}: {a['subject']}->{a['object']} ({a['valid_from']}-{a['valid_to']}, {a['source_doc_id']}) "
                  f"vs {b['subject']}->{b['object']} ({b['valid_from']}-{b['valid_to']}, {b['source_doc_id']})")
    if args.dry_run:
        for f in facts[:5]:
            print(" ", f)
        return

    conn = load(get_connection(), facts)
    install_queries(conn)
    print("\nDone.")


if __name__ == "__main__":
    main()
