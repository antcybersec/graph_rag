#!/usr/bin/env bash
# One-command build: TigerGraph up, schema applied, queries installed, corpus loaded.
#
#   ./build.sh corpus.jsonl
#
# Safe to re-run. Every GSQL object is created with CREATE OR REPLACE and
# every row is upserted by primary id, so a second run converges on the same
# graph instead of failing or duplicating -- which is the failure a judge
# would hit re-running this in the call.
set -euo pipefail

CORPUS="${1:-data/corpus/corpus_r2.jsonl}"
CONTAINER="${TG_CONTAINER:-tigergraph}"
IMAGE="${TG_IMAGE:-tigergraph/community:4.2.6}"
VOLUME="${TG_VOLUME:-tg_r2_data}"
GRAPH="${TG_GRAPHNAME:-r2}"
cd "$(dirname "$0")"

say() { printf '\n== %s\n' "$*"; }

[ -f "$CORPUS" ] || { echo "corpus not found: $CORPUS" >&2; exit 1; }

say "python environment"
if [ ! -d .venv ]; then python3 -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.lock.txt
export TOKENIZERS_PARALLELISM=false

say "TigerGraph container ($IMAGE)"
if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    docker start "$CONTAINER"
  else
    # The data lives in a named volume, so the graph survives a container
    # being replaced -- a rebuild of the image must not mean a rebuild of
    # 16,377 embeddings.
    docker volume create "$VOLUME" >/dev/null
    docker run -d --name "$CONTAINER" --platform linux/amd64 \
      -p 14240:14240 -p 9000:9000 -p 14022:22 \
      -v "$VOLUME":/home/tigergraph/tigergraph/data \
      "$IMAGE" /bin/sh -c 'bash -c /home/tigergraph/entrypoint.sh'
  fi
fi

say "waiting for the database to accept queries"
# Not just for the port: counts read while the graph store is still warming
# come back stale (a vertex type reporting 0 beside thousands of its edges),
# and anything that trusts them is wrong in a way that looks like a data bug.
for i in $(seq 1 90); do
  if docker exec "$CONTAINER" /home/tigergraph/tigergraph/app/cmd/gadmin status \
       2>/dev/null | grep -q "Online"; then
    sleep 5; echo "ready after ~$((i * 5))s"; break
  fi
  sleep 5
  [ "$i" = 90 ] && { echo "database did not come up" >&2; exit 1; }
done

say "graph $GRAPH"
docker exec "$CONTAINER" /home/tigergraph/tigergraph/app/cmd/gsql \
  "CREATE GRAPH $GRAPH()" 2>/dev/null || echo "  (already exists)"

say "schema, queries and corpus"
python -m src.r2.load "$CORPUS"

say "graph counts (submission deliverable 6)"
python -m src.r2.counts

say "done"
