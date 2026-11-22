"""Vertex and edge counts by type -- submission deliverable 6.

Read from the live graph, never typed in by hand.

One caveat worth knowing before trusting any number here: counts read while
TigerGraph's graph store is still warming up come back **stale**. The first
build of this graph reported `EventSeries 0` beside 296 of its own
`INSTANCE_OF` edges, which is impossible, and sent me looking for a loader bug
that did not exist. If a number looks wrong, let the services settle and read
it again.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Graph counts by type")
    ap.add_argument("--out", default="data/results/r2/graph_counts.json")
    args = ap.parse_args()

    from src.r2.conn import describe, get_connection
    conn = get_connection()
    vertices = conn.getVertexCount("*")
    edges = conn.getEdgeCount("*")
    payload = {
        "graph": describe(),
        "vertices_total": sum(vertices.values()),
        "edges_total": sum(edges.values()),
        "vertices": dict(sorted(vertices.items())),
        "edges": dict(sorted(edges.items())),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"{payload['graph']}\n")
    print(f"| vertex type | count |\n|---|---|")
    for name, count in payload["vertices"].items():
        print(f"| {name} | {count:,} |")
    print(f"| **total** | **{payload['vertices_total']:,}** |\n")
    print(f"| edge type | count |\n|---|---|")
    for name, count in payload["edges"].items():
        print(f"| {name} | {count:,} |")
    print(f"| **total** | **{payload['edges_total']:,}** |")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
