"""Builds the structured event layer in TigerGraph straight from the corpus
infoboxes -- no LLM or embedding calls -- and installs the queries the
event_graph pipeline runs over it.

  Games <-IN_GAMES- OlympicEvent -IN_SPORT-> Sport
  Games -PREV_GAMES-> Games          (real Olympic calendar, later -> earlier)
  OlympicEvent -HELD_AT-> Venue
  OlympicEvent -DESCRIBED_BY-> Document   (provenance back to the source article)

Why (docs/architecture.md, "Structured event layer"): the eval questions are
filters, counts and argmaxes over infobox fields, and one hop back through
the Games calendar. Those are graph queries, not similarity searches.

Venue vertices are keyed by normalized venue name, so identically named
venues at different Games ("Olympic Tennis Centre" in Athens and in Rio)
share one vertex. Queries that start from a venue therefore also filter by
date and, when known, by Games.

Usage (safe to re-run; schema is only created if missing, upserts overwrite,
and queries use CREATE OR REPLACE -- a plain CREATE fails on re-run and leaves
the old body installed, so an edited query would silently never take effect):
  python -m src.ingestion.build_event_graph [--queries-only]
"""
import argparse

from tqdm import tqdm

from src.common.infobox import OLYMPIC_CALENDAR, games_id, load_events
from src.common.tg_conn import get_connection

GRAPH_NAME = "test"
UPSERT_BATCH_SIZE = 500

SCHEMA_JOB = f"""
USE GRAPH {GRAPH_NAME}

CREATE SCHEMA_CHANGE JOB add_event_layer FOR GRAPH {GRAPH_NAME} {{
    ADD VERTEX OlympicEvent (
        PRIMARY_ID event_id STRING,
        title STRING,
        sport STRING,
        event_name STRING,
        event_key STRING,
        games_year INT,
        season STRING,
        venue STRING,
        date_text STRING,
        date_key STRING,
        competitors INT,
        nations INT,
        gold STRING,
        silver STRING,
        bronze STRING
    ) WITH PRIMARY_ID_AS_ATTRIBUTE="true";

    ADD VERTEX Games (PRIMARY_ID games_id STRING, year INT, season STRING) WITH PRIMARY_ID_AS_ATTRIBUTE="true";
    ADD VERTEX Sport (PRIMARY_ID sport_key STRING, name STRING) WITH PRIMARY_ID_AS_ATTRIBUTE="true";
    ADD VERTEX Venue (PRIMARY_ID venue_key STRING, name STRING) WITH PRIMARY_ID_AS_ATTRIBUTE="true";

    ADD UNDIRECTED EDGE IN_GAMES (FROM OlympicEvent, TO Games);
    ADD UNDIRECTED EDGE IN_SPORT (FROM OlympicEvent, TO Sport);
    ADD UNDIRECTED EDGE HELD_AT (FROM OlympicEvent, TO Venue);
    ADD UNDIRECTED EDGE DESCRIBED_BY (FROM OlympicEvent, TO Document);
    ADD DIRECTED EDGE PREV_GAMES (FROM Games, TO Games);
}}

RUN SCHEMA_CHANGE JOB add_event_layer
"""

# competitors is -1 when the infobox has no count, so a threshold or argmax
# never picks an event whose size is unknown.
QUERIES = {
    "events_by_games_sport": f"""
USE GRAPH {GRAPH_NAME}

CREATE OR REPLACE QUERY events_by_games_sport(VERTEX<Games> games, VERTEX<Sport> sport, INT min_competitors = -1) FOR GRAPH {GRAPH_NAME} SYNTAX v3 {{
  G = {{games}};
  S = {{sport}};
  InGames = SELECT e FROM OlympicEvent:e -(IN_GAMES)- G:g;
  InSport = SELECT e FROM OlympicEvent:e -(IN_SPORT)- S:s;
  Pool = InGames INTERSECT InSport;
  Over = SELECT e FROM Pool:e WHERE e.competitors >= 0 AND e.competitors > min_competitors;
  PRINT Pool;
  PRINT Over;
}}

INSTALL QUERY events_by_games_sport
""",
    "events_at_venue": f"""
USE GRAPH {GRAPH_NAME}

CREATE OR REPLACE QUERY events_at_venue(VERTEX<Venue> venue) FOR GRAPH {GRAPH_NAME} SYNTAX v3 {{
  V = {{venue}};
  AtVenue = SELECT e FROM OlympicEvent:e -(HELD_AT)- V:v;
  PRINT AtVenue;
}}

INSTALL QUERY events_at_venue
""",
    "events_in_previous_games": f"""
USE GRAPH {GRAPH_NAME}

CREATE OR REPLACE QUERY events_in_previous_games(VERTEX<Games> games, VERTEX<Sport> sport) FOR GRAPH {GRAPH_NAME} SYNTAX v3 {{
  G = {{games}};
  S = {{sport}};
  Prev = SELECT p FROM G:g -(PREV_GAMES>)- Games:p;
  InGames = SELECT e FROM OlympicEvent:e -(IN_GAMES)- Prev:p;
  InSport = SELECT e FROM OlympicEvent:e -(IN_SPORT)- S:s;
  Pool = InGames INTERSECT InSport;
  PRINT Prev;
  PRINT Pool;
}}

INSTALL QUERY events_in_previous_games
""",
}


def _batched_upsert(label, items, fn):
    for i in tqdm(range(0, len(items), UPSERT_BATCH_SIZE), desc=label):
        fn(items[i : i + UPSERT_BATCH_SIZE])


def create_schema(conn):
    if "OlympicEvent" in conn.getVertexTypes():
        print("event layer schema already exists -- skipping schema job")
        return
    print(conn.gsql(SCHEMA_JOB))


def load_graph(conn, events):
    event_attrs = ("title", "sport", "event_name", "event_key", "games_year", "season", "venue",
                   "date_text", "date_key", "competitors", "nations", "gold", "silver", "bronze")
    _batched_upsert("OlympicEvent", [(e["event_id"], {k: e[k] for k in event_attrs}) for e in events],
                    lambda b: conn.upsertVertices("OlympicEvent", b))

    games = [(games_id(y, season), {"year": y, "season": season}) for season, years in OLYMPIC_CALENDAR.items() for y in years]
    conn.upsertVertices("Games", games)
    prev_edges = [(games_id(later, season), games_id(earlier, season), {})
                  for season, years in OLYMPIC_CALENDAR.items() for earlier, later in zip(years, years[1:])]
    conn.upsertEdges("Games", "PREV_GAMES", "Games", prev_edges)

    sports = {e["sport_key"]: e["sport"] for e in events}
    conn.upsertVertices("Sport", [(k, {"name": v}) for k, v in sports.items()])
    venues = {e["venue_key"]: e["venue"] for e in events if e["venue_key"]}
    _batched_upsert("Venue", [(k, {"name": v}) for k, v in venues.items()], lambda b: conn.upsertVertices("Venue", b))

    for edge, target_type, target_of in (
        ("IN_GAMES", "Games", lambda e: e["games_id"]),
        ("IN_SPORT", "Sport", lambda e: e["sport_key"]),
        ("HELD_AT", "Venue", lambda e: e["venue_key"]),
        ("DESCRIBED_BY", "Document", lambda e: e["event_id"]),
    ):
        edges = [(e["event_id"], target_of(e), {}) for e in events if target_of(e)]
        _batched_upsert(edge, edges, lambda b, edge=edge, target_type=target_type:
                        conn.upsertEdges("OlympicEvent", edge, target_type, b))

    print(f"loaded {len(events)} events, {len(games)} games, {len(sports)} sports, {len(venues)} venues")


def install_queries(conn):
    for name, gsql in QUERIES.items():
        print(f"\n=== installing {name} ===")
        result = conn.gsql(gsql)
        print(result[-800:] if isinstance(result, str) else result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries-only", action="store_true")
    args = parser.parse_args()

    conn = get_connection()
    if not args.queries_only:
        create_schema(conn)
        conn = get_connection()  # fresh token so the new schema version is visible
        load_graph(conn, load_events())
    install_queries(conn)


if __name__ == "__main__":
    main()
