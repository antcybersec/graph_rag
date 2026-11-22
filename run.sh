#!/usr/bin/env bash
# Answer a question file.
#
#   ./run.sh questions_public.jsonl answers_public.jsonl
#
# Writes the answers, the TigerGraph query log beside them
# (<answers>.tgquerylog.jsonl, submission deliverable 5), and nothing else.
# Reads no answer key: the gold answers are only ever touched by the scorer.
set -euo pipefail
cd "$(dirname "$0")"
QUESTIONS="${1:?usage: ./run.sh <questions.jsonl> <answers.jsonl>}"
ANSWERS="${2:?usage: ./run.sh <questions.jsonl> <answers.jsonl>}"
# shellcheck disable=SC1091
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
exec python -m src.r2.run "$QUESTIONS" "$ANSWERS"
