"""Installs the GSQL queries all three pipelines rely on:
  - vector_search_chunks: ANN top-k over Chunk.embedding (used by RAG, and by
    Agentic GraphRAG's similarity-search tool)
  - vector_search_entities: ANN top-k over Entity.embedding (entity linking
    for GraphRAG / Agentic)
  - entity_neighborhood: 1-2 hop traversal from a set of seed entities,
    returning related entities + the RELATED_TO edges + their MENTIONS'd
    chunks (used by GraphRAG's "local search" and Agentic's graph-traversal
    tool)

Run once after the schema exists (create_schema.py) -- safe to re-run
(INSTALL QUERY overwrites).
"""
from src.common.tg_conn import get_connection

GRAPH_NAME = "test"

QUERIES = {
    "vector_search_chunks": f"""
USE GRAPH {GRAPH_NAME}

CREATE QUERY vector_search_chunks(LIST<FLOAT> query_vector, INT k=5) FOR GRAPH {GRAPH_NAME} SYNTAX v3 {{
  MapAccum<VERTEX, FLOAT> @@distances;
  Result = vectorSearch({{Chunk.embedding}}, query_vector, k, {{distance_map: @@distances}});
  PRINT Result;
  PRINT @@distances;
}}

INSTALL QUERY vector_search_chunks
""",
    "vector_search_entities": f"""
USE GRAPH {GRAPH_NAME}

CREATE QUERY vector_search_entities(LIST<FLOAT> query_vector, INT k=5) FOR GRAPH {GRAPH_NAME} SYNTAX v3 {{
  MapAccum<VERTEX, FLOAT> @@distances;
  Result = vectorSearch({{Entity.embedding}}, query_vector, k, {{distance_map: @@distances}});
  PRINT Result;
  PRINT @@distances;
}}

INSTALL QUERY vector_search_entities
""",
    "entity_neighbors_1hop": f"""
USE GRAPH {GRAPH_NAME}

CREATE QUERY entity_neighbors_1hop(SET<VERTEX<Entity>> seeds) FOR GRAPH {GRAPH_NAME} SYNTAX v3 {{
  // Both selects below are capped with LIMIT -- unbounded, a single hub
  // entity (e.g. "2018 Winter Olympics") can have hundreds/thousands of
  // RELATED_TO edges and MENTIONS chunks, blowing the answer-synthesis
  // context to 100k+ tokens for one question (observed in production,
  // 2026-09-10: one GraphRAG call hit 169k tokens and scored WORSE than
  // plain RAG's 12k-token answer on the same question -- more context
  // isn't better here, it's mostly noise that also risks quota exhaustion).
  // Tightened from 20 to 8 on 2026-09-11: a free-tier provider's 8k-token-
  // per-minute cap flatly rejected the ~15-19k-token contexts LIMIT 20
  // produced. 8 is a deliberate design choice, not just a quota workaround --
  // GraphRAG was already the weakest-performing pipeline at LIMIT 20, so
  // there's no evidence more evidence here was helping.
  ListAccum<EDGE> @@edges;
  Seeds = {{seeds}};
  Neighbors = SELECT t FROM Seeds:s -(RELATED_TO:e)- Entity:t
              WHERE t != s
              ACCUM @@edges += e
              LIMIT 8;
  PRINT Neighbors;
  PRINT @@edges AS related_edges;

  AllEntities = Seeds UNION Neighbors;
  ChunksOfNeighbors = SELECT c FROM Chunk:c -(MENTIONS>:m)- AllEntities:v
                      LIMIT 8;
  PRINT ChunksOfNeighbors;
}}

INSTALL QUERY entity_neighbors_1hop
""",
}


def main():
    conn = get_connection()
    for name, gsql in QUERIES.items():
        print(f"\n=== installing {name} ===")
        try:
            result = conn.gsql(gsql)
            print(result[-800:] if isinstance(result, str) else result)
        except Exception as e:
            print(f"ERROR installing {name}: {e}")
            raise


if __name__ == "__main__":
    main()
