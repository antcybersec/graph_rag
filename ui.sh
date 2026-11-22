#!/usr/bin/env bash
# The demo page. Needs the graph up (./build.sh) -- it reads it live.
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
exec streamlit run src/r2/ui.py
