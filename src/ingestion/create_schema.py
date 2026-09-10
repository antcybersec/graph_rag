"""Creates the graph schema in TigerGraph Savanna.

Design (see docs/architecture.md for rationale):
  Document -[PART_OF]- Chunk -[MENTIONS]-> Entity -[RELATED_TO]-> Entity -[IN_COMMUNITY]-> Community

RELATED_TO carries provenance/temporal fields from day one (source_url,
source_type, observed_at, valid_from, valid_to, authority_score) even though
Round 1 doesn't require reasoning over them -- this is the "do it now, it's
free" recommendation from the Round 2 (temporal/conflicting facts) research:
retrofitting provenance after ingestion is far more painful than capturing
it at load time.

Our Savanna API token is scoped to graph `test` (not global), so schema
changes must go through a graph-local `SCHEMA_CHANGE JOB ... FOR GRAPH test
{ ... }` block -- bare top-level ALTER/ADD statements and GLOBAL schema
change jobs are rejected with "Access Denied ... attempting operations on
global" under this auth scope.
"""
from src.common.tg_conn import get_connection

GRAPH_NAME = "test"
EMBED_DIM = 768

SCHEMA_JOB = f"""
USE GRAPH {GRAPH_NAME}

CREATE SCHEMA_CHANGE JOB build_graph_schema FOR GRAPH {GRAPH_NAME} {{
    ADD VERTEX Document (
        PRIMARY_ID doc_id STRING,
        title STRING,
        url STRING,
        wikidata_qid STRING,
        wikipedia_pageid STRING,
        approx_tokens INT
    ) WITH PRIMARY_ID_AS_ATTRIBUTE="true";

    ADD VERTEX Chunk (
        PRIMARY_ID chunk_id STRING,
        doc_id STRING,
        chunk_index INT,
        text STRING,
        token_count INT
    ) WITH PRIMARY_ID_AS_ATTRIBUTE="true";

    ADD VERTEX Entity (
        PRIMARY_ID entity_id STRING,
        name STRING,
        canonical_name STRING,
        entity_type STRING,
        description STRING,
        mention_count INT
    ) WITH PRIMARY_ID_AS_ATTRIBUTE="true";

    ADD VERTEX Community (
        PRIMARY_ID community_id STRING,
        level INT,
        title STRING,
        summary STRING
    ) WITH PRIMARY_ID_AS_ATTRIBUTE="true";

    ADD DIRECTED EDGE PART_OF (FROM Chunk, TO Document);

    ADD DIRECTED EDGE MENTIONS (FROM Chunk, TO Entity, weight INT);

    ADD UNDIRECTED EDGE RELATED_TO (
        FROM Entity, TO Entity,
        relation_label STRING,
        description STRING,
        source_chunk_id STRING,
        source_doc_id STRING,
        source_url STRING,
        source_type STRING,
        observed_at STRING,
        valid_from STRING,
        valid_to STRING,
        authority_score FLOAT,
        weight INT
    );

    ADD DIRECTED EDGE IN_COMMUNITY (FROM Entity, TO Community);
}}

RUN SCHEMA_CHANGE JOB build_graph_schema
"""

VECTOR_JOB = f"""
USE GRAPH {GRAPH_NAME}

CREATE SCHEMA_CHANGE JOB add_vector_attrs FOR GRAPH {GRAPH_NAME} {{
    ALTER VERTEX Chunk ADD VECTOR ATTRIBUTE embedding(dimension={EMBED_DIM}, metric="COSINE");
    ALTER VERTEX Entity ADD VECTOR ATTRIBUTE embedding(dimension={EMBED_DIM}, metric="COSINE");
}}

RUN SCHEMA_CHANGE JOB add_vector_attrs
"""


def run(conn, gsql_text: str, label: str):
    print(f"\n=== {label} ===")
    try:
        result = conn.gsql(gsql_text)
        print(result)
        return result
    except Exception as e:
        print(f"ERROR: {e}")
        raise


def main():
    conn = get_connection()
    run(conn, SCHEMA_JOB, "vertex/edge schema job")
    run(conn, VECTOR_JOB, "vector attribute job")

    conn2 = get_connection()  # fresh token to see committed schema
    print("\nFinal vertex types:", conn2.getVertexTypes())
    print("Final edge types:", conn2.getEdgeTypes())


if __name__ == "__main__":
    main()
