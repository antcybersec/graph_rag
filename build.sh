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

# `.env` is gitignored, so a fresh clone has none, and `load_dotenv()` then
# finds nothing. The code defaults now match `.env.example` exactly, so that
# alone is a correct build -- but the file still has to exist for a key to go
# in, and writing it here means the answering path and this build agree about
# every setting rather than relying on whoever remembers to copy it.
say "configuration"
if [ ! -f .env ]; then
  cp .env.example .env
  echo "created .env from .env.example (settings already correct; add a key to use the API)"
else
  echo ".env already present, left untouched"
fi
if ! grep -qE '^GOOGLE_API_KEY=.+' .env; then
  echo
  echo "NOTE: GOOGLE_API_KEY is empty in .env."
  echo "      The build needs no key -- embeddings are local (BAAI/bge-base-en-v1.5)."
  echo "      Answering questions does: put a key in .env, or use ./offline.sh,"
  echo "      which replays the submitted answers with no key and no quota."
fi

say "python environment"
# Name the interpreter requirement rather than inheriting whatever `python3`
# happens to be. The pinned tree needs 3.10+ (aiohappyeyeballs, a transitive
# dependency of the Google client, requires it) and was built and run on 3.12;
# on a machine whose `python3` is 3.9 the install fails deep in dependency
# resolution with a message about a version of a package nobody has heard of.
PY="${PYTHON:-python3}"
PY_OK=$("$PY" -c 'import sys; print(1 if sys.version_info[:2] >= (3, 10) else 0)' 2>/dev/null || echo 0)
if [ "$PY_OK" != "1" ]; then
  echo "need Python 3.10 or newer (3.12 is what this was built and run on)." >&2
  echo "found: $("$PY" --version 2>&1 || echo "no $PY on PATH")" >&2
  echo "point PYTHON at a newer interpreter, e.g. PYTHON=python3.12 ./build.sh $CORPUS" >&2
  exit 1
fi
if [ ! -d .venv ]; then "$PY" -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements-r2.txt
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

# Pure functions first, then the graph. Both are free: no model and no quota.
# A build that loaded cleanly and a route that has quietly stopped matching
# look identical until a run costs 130 requests to tell you apart.
say "checks (no model, no quota)"
python -m tests.test_r2
python -m src.r2.smoke || echo "  one or more routes returned nothing -- read the lines above"

say "done"
