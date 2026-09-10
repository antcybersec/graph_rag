"""One-time schema fix: RELATED_TO was created without a DISCRIMINATOR, which
means TigerGraph treats (source, target) as the whole edge key -- a second
relationship between the same two entities (e.g. "partnered with" in one
event, "competed against" in another) would silently overwrite the first
instead of coexisting. Safe to run now since RELATED_TO has no data loaded
yet (Document/Chunk loading doesn't touch it).
"""
from src.common.tg_conn import get_connection

GRAPH_NAME = "test"

DROP_JOB = f"""
USE GRAPH {GRAPH_NAME}

CREATE SCHEMA_CHANGE JOB drop_related_to FOR GRAPH {GRAPH_NAME} {{
    DROP EDGE RELATED_TO;
}}

RUN SCHEMA_CHANGE JOB drop_related_to
"""

ADD_JOB = f"""
USE GRAPH {GRAPH_NAME}

CREATE SCHEMA_CHANGE JOB add_related_to_v2 FOR GRAPH {GRAPH_NAME} {{
    ADD UNDIRECTED EDGE RELATED_TO (
        FROM Entity, TO Entity,
        DISCRIMINATOR(relation_label STRING),
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
}}

RUN SCHEMA_CHANGE JOB add_related_to_v2
"""


def main():
    conn = get_connection()
    print("--- drop ---")
    print(conn.gsql(DROP_JOB))
    conn = get_connection()  # fresh token/schema-version view
    print("--- add ---")
    print(conn.gsql(ADD_JOB))
    conn2 = get_connection()
    print("\nedge types:", conn2.getEdgeTypes())


if __name__ == "__main__":
    main()
