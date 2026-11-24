#!/usr/bin/env bash
# Answer a question with no network and no API key.
#
#   ./offline.sh "Which nation won the most gold medals in athletics at the 2024 Summer Olympics?"
#
# Every model call -- planning, selection and composition -- runs on a local
# Ollama model. TigerGraph is already local, and the embeddings were always
# local (BGE), so with this switch the whole system runs on the machine in
# front of you: no key, no quota, no provider.
#
# Why it exists: the free Gemini tier is 500 requests a day, and the judging
# call has to reproduce answers live. A quota wall during the demo is the one
# failure the system cannot talk its way out of, so there is a path that cannot
# hit one.
#
# What it costs, measured on an M4 with 16GB (2026-11-22, qwen3:8b):
#
#   one question, start to finish   123s   (2 model calls, 6,908 in / 1,730 out)
#   the same question on Gemini      16s
#
# So it is the fallback and the reproduction path, not the way the submitted
# runs were produced -- those use one validated configuration, recorded in
# `metrics.json`. Two further caveats, both measured rather than assumed:
# qwen3:8b plans correctly on 4 of 5 probe questions where Gemini gets 5
# (`docs/round2-risks.md`, risk 2), and it cites more documents than the answer
# needs, which costs citation precision. Accuracy is 40% of the score and
# citations 10%, which is why this is insurance and not the default.
#
# RAM is the real constraint, not speed. TigerGraph holds ~5.6GB and qwen3:8b
# at 12,288 context holds ~5.9GB, which fits in 16GB with the embedder and
# leaves the machine usable. Raising the context to 20,480 -- the default for
# the Round 1 pipelines -- pushes the model to ~4.5GB for a 3B and OOM-killed
# the run, so the value here is deliberate and not a guess.
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${OFFLINE_MODEL:-qwen3:8b}"
CONTAINER="${TG_CONTAINER:-tigergraph}"

command -v ollama >/dev/null || { echo "ollama is not installed: https://ollama.com" >&2; exit 1; }
# `ollama list` only talks to a running server. Without this the first model
# call raises a bare ConnectionError, gets retried five times with backoff,
# and surfaces as a traceback -- for a server that needs starting.
if ! curl -fsS --max-time 5 "${OLLAMA_HOST:-http://localhost:11434}/api/tags" >/dev/null 2>&1; then
  echo "ollama is installed but not responding at ${OLLAMA_HOST:-http://localhost:11434}" >&2
  echo "  start it with:  ollama serve" >&2
  exit 1
fi
# Match the whole tag: "qwen3" matches a model called qwen3:4b that is not the
# one measured, and "ollama pull" on a model already present is a no-op anyway.
ollama list | awk 'NR>1{print $1}' | grep -qx "$MODEL" \
  || { echo "pulling $MODEL (about 5GB, once)"; ollama pull "$MODEL"; }

# Distinguish "Docker is not running" from "the graph was never built". They
# need opposite actions, and `docker ps` answers both with the same failure, so
# a missing daemon was being reported as a missing graph.
if ! docker info >/dev/null 2>&1; then
  echo "the Docker daemon is not responding." >&2
  echo "  Start Docker Desktop (or your docker daemon) and run this again." >&2
  exit 1
fi
if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  # Deliberately NOT creating the container here, unlike build.sh. `docker run`
  # would start an empty graph, and an empty graph does not look broken -- it
  # answers every question "not in the corpus", which in a demo reads as a
  # working system that is confidently wrong. Say what is missing instead.
  if ! docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "no TigerGraph container named '$CONTAINER' on this machine." >&2
    echo "  The graph has to be built once first:" >&2
    echo "    ./build.sh corpus.jsonl        (needs no API key -- embeddings are local)" >&2
    exit 1
  fi
  echo "starting TigerGraph"
  docker start "$CONTAINER" >/dev/null
  for _ in $(seq 1 90); do
    docker exec "$CONTAINER" /home/tigergraph/tigergraph/app/cmd/gadmin status 2>/dev/null \
      | grep -q "Online" && { sleep 5; break; }
    sleep 5
  done
fi

# shellcheck disable=SC1091
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
export GEN_PROVIDER=local SMALL_CALL_PROVIDER=local
export GEN_MODEL="$MODEL" SMALL_CALL_MODEL="$MODEL"
export OLLAMA_NUM_CTX="${OLLAMA_NUM_CTX:-12288}"
# Unload the model the instant each call returns. Ollama otherwise keeps it
# resident for five minutes, and ~6GB of idle weights beside TigerGraph on a
# 16GB machine got the container OOM-killed (`Exited (137)`) with the graph
# still loaded. The data survives in the volume either way, but a dead
# container in the middle of a demo does not look like a working system.
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-0}"

exec python -m src.r2.ask "$@"
