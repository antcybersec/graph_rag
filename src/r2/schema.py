"""TigerGraph schema for the Round 2 graph.

Five layers, each earning its place from a measured question family:

  text        Document, Chunk (+ 768-dim vector attribute) -- the only path to
              the "why" questions, whose answers exist solely in prose.
  events      OlympicEvent joined to Games, Sport, Venue and EventSeries.
              EventSeries exists so a trend across editions is one hop rather
              than a scan of 2,750 vertices.
  medals      Athlete -WON-> event for careers, Nation -MEDAL_FOR-> event for
              nation aggregation. Two edges rather than one because a medal
              line can name a whole squad: the NOC is always known, the
              individual is not.
  reference   Film, Person, Office for the film and succession questions.
  evidence    SourceRecord and Claim, with ASSERTED_BY / ABOUT_EVENT /
              SUPERSEDES. This is the layer Round 2 is actually about, and the
              losing claims are stored on purpose -- an answer has to be able
              to say *why* the newest document was wrong.

DISCRIMINATORs are not optional here. TigerGraph keeps one edge per
(from, to, type) unless a discriminator distinguishes them, and this corpus
has 357 events awarding two bronze medals -- frequently to the same nation.
Without DISCRIMINATOR(medal, recipient) on MEDAL_FOR those two collapse into
one edge and every medal table is quietly short. The same applies to
HELD_OFFICE: Putin's two non-consecutive terms as President of Russia are two
edges between the same pair of vertices.
"""
from __future__ import annotations

import os

GRAPH_NAME = os.environ.get("TG_GRAPHNAME", "r2")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "768"))

SCHEMA_JOB = f"""
USE GRAPH {GRAPH_NAME}

CREATE SCHEMA_CHANGE JOB r2_build_schema FOR GRAPH {GRAPH_NAME} {{

    ADD VERTEX Document (
        PRIMARY_ID doc_id STRING,
        title STRING,
        url STRING,
        doc_class STRING,
        synthetic BOOL,
        approx_tokens INT
    ) WITH PRIMARY_ID_AS_ATTRIBUTE="true";

    ADD VERTEX Chunk (
        PRIMARY_ID chunk_id STRING,
        doc_id STRING,
        ord INT,
        text STRING
    ) WITH PRIMARY_ID_AS_ATTRIBUTE="true";

    ADD VERTEX Games (
        PRIMARY_ID games_id STRING,
        year INT,
        season STRING
    ) WITH PRIMARY_ID_AS_ATTRIBUTE="true";

    ADD VERTEX Sport (
        PRIMARY_ID sport_key STRING,
        name STRING
    ) WITH PRIMARY_ID_AS_ATTRIBUTE="true";

    ADD VERTEX Venue (
        PRIMARY_ID venue_key STRING,
        name STRING
    ) WITH PRIMARY_ID_AS_ATTRIBUTE="true";

    ADD VERTEX EventSeries (
        PRIMARY_ID series_key STRING,
        sport_key STRING,
        event_name STRING,
        event_key STRING
    ) WITH PRIMARY_ID_AS_ATTRIBUTE="true";

    ADD VERTEX OlympicEvent (
        PRIMARY_ID event_id STRING,
        title STRING,
        sport_key STRING,
        event_name STRING,
        event_key STRING,
        series_key STRING,
        games_year INT,
        season STRING,
        venue STRING,
        venue_key STRING,
        date_text STRING,
        competitors INT,
        nations INT,
        teams INT,
        win_value STRING,
        prev_year INT,
        next_year INT,
        synthetic BOOL
    ) WITH PRIMARY_ID_AS_ATTRIBUTE="true";

    ADD VERTEX Athlete (
        PRIMARY_ID athlete_key STRING,
        name STRING
    ) WITH PRIMARY_ID_AS_ATTRIBUTE="true";

    ADD VERTEX Nation (
        PRIMARY_ID noc STRING
    ) WITH PRIMARY_ID_AS_ATTRIBUTE="true";

    ADD VERTEX Film (
        PRIMARY_ID film_id STRING,
        name STRING,
        title STRING,
        year INT,
        released STRING,
        budget DOUBLE,
        gross DOUBLE,
        budget_currency STRING,
        gross_currency STRING
    ) WITH PRIMARY_ID_AS_ATTRIBUTE="true";

    ADD VERTEX Person (
        PRIMARY_ID person_id STRING,
        name STRING,
        name_key STRING,
        kind STRING,
        party STRING
    ) WITH PRIMARY_ID_AS_ATTRIBUTE="true";

    ADD VERTEX Office (
        PRIMARY_ID office_key STRING,
        name STRING
    ) WITH PRIMARY_ID_AS_ATTRIBUTE="true";

    ADD VERTEX SourceRecord (
        PRIMARY_ID source_id STRING,
        title STRING,
        source_text STRING,
        published_at STRING,
        published_local STRING,
        tier INT,
        tier_name STRING,
        tier_reason STRING
    ) WITH PRIMARY_ID_AS_ATTRIBUTE="true";

    ADD VERTEX Claim (
        PRIMARY_ID claim_id STRING,
        subject STRING,
        subject_key STRING,
        slot_subject STRING,
        predicate STRING,
        value STRING,
        value_key STRING,
        population STRING,
        stance STRING,
        status STRING,
        effective STRING,
        effective_sort UINT,
        tier INT,
        source_id STRING,
        event_id STRING
    ) WITH PRIMARY_ID_AS_ATTRIBUTE="true";

    ADD DIRECTED EDGE HAS_CHUNK (FROM Document, TO Chunk);
    ADD DIRECTED EDGE DESCRIBES_EVENT (FROM Document, TO OlympicEvent);
    ADD DIRECTED EDGE DESCRIBES_FILM (FROM Document, TO Film);
    ADD DIRECTED EDGE DESCRIBES_PERSON (FROM Document, TO Person);

    ADD DIRECTED EDGE IN_GAMES (FROM OlympicEvent, TO Games);
    ADD DIRECTED EDGE IN_SPORT (FROM OlympicEvent, TO Sport);
    ADD DIRECTED EDGE HELD_AT (FROM OlympicEvent, TO Venue);
    ADD DIRECTED EDGE INSTANCE_OF (FROM OlympicEvent, TO EventSeries);
    ADD DIRECTED EDGE PREV_EDITION (FROM OlympicEvent, TO OlympicEvent);

    ADD DIRECTED EDGE WON (
        FROM Athlete, TO OlympicEvent,
        DISCRIMINATOR(medal STRING),
        rank INT,
        noc STRING
    );

    ADD DIRECTED EDGE MEDAL_FOR (
        FROM Nation, TO OlympicEvent,
        DISCRIMINATOR(medal STRING, recipient STRING),
        rank INT,
        multi BOOL
    );

    ADD DIRECTED EDGE DIRECTED_FILM (FROM Person, TO Film);

    ADD DIRECTED EDGE HELD_OFFICE (
        FROM Person, TO Office,
        DISCRIMINATOR(slot INT),
        term_start STRING,
        term_end STRING,
        predecessor STRING,
        successor STRING
    );

    ADD DIRECTED EDGE ASSERTED_BY (FROM Claim, TO SourceRecord);
    ADD DIRECTED EDGE ABOUT_EVENT (FROM Claim, TO OlympicEvent);
    ADD DIRECTED EDGE SUPERSEDES (FROM Claim, TO Claim);
}}

RUN SCHEMA_CHANGE JOB r2_build_schema
"""

# Vector attributes go in their own job: ADD VERTEX and ALTER VERTEX ... ADD
# VECTOR ATTRIBUTE cannot share a schema-change job (verified against Savanna
# 4.2.5 in Round 1 -- the combined job is rejected).
VECTOR_JOB = f"""
USE GRAPH {GRAPH_NAME}

CREATE SCHEMA_CHANGE JOB r2_add_vectors FOR GRAPH {GRAPH_NAME} {{
    ALTER VERTEX Chunk ADD VECTOR ATTRIBUTE embedding(dimension={EMBED_DIM}, metric="COSINE");
}}

RUN SCHEMA_CHANGE JOB r2_add_vectors
"""

# Adding an attribute to a vertex type that already exists cannot go in
# r2_build_schema: ADD VERTEX on an existing type is rejected, so re-running
# the build job would not pick it up. This job is additive and touches only
# SourceRecord -- in particular it does not go near Chunk, whose 16,377
# embeddings take about half an hour to rebuild.
#
# Safe to run on a loaded graph, and safe to skip: an existing attribute makes
# it fail, which `ensure_published_local` treats as already done.
PUBLISHED_LOCAL_JOB = f"""
USE GRAPH {GRAPH_NAME}

CREATE SCHEMA_CHANGE JOB r2_add_published_local FOR GRAPH {GRAPH_NAME} {{
    ALTER VERTEX SourceRecord ADD ATTRIBUTE (published_local STRING);
}}

RUN SCHEMA_CHANGE JOB r2_add_published_local
"""


def ensure_published_local(conn) -> str:
    """Add SourceRecord.published_local if it is not already there."""
    try:
        out = apply(conn, PUBLISHED_LOCAL_JOB, "r2_add_published_local")
        return "added"
    except RuntimeError as exc:
        text = str(exc).lower()
        # TigerGraph's wording for "that attribute is already there" is
        # "conflict with another attribute", not "already exists" -- checking
        # for the latter made a job that had in fact succeeded look like a
        # hard failure on every subsequent run.
        if ("conflict with another attribute" in text
                or "already exist" in text or "duplicate" in text):
            return "already present"
        raise


CREATE_GRAPH = f"CREATE GRAPH {GRAPH_NAME}()"


def apply(conn, gsql_text: str, label: str) -> str:
    """Run one GSQL block and fail loudly -- TigerGraph reports errors in the
    response body with a 200 status, so the text has to be inspected."""
    out = conn.gsql(gsql_text)
    low = (out or "").lower()
    if "error" in low or "failed" in low or "semantic check fails" in low:
        raise RuntimeError(f"{label} failed:\n{out}")
    return out
