#!/usr/bin/env bash
set -euo pipefail

# Run all golden-set evals and compare results.
# Usage:
#   evaluation/scripts/run_all_evals.sh
#   MLFLOW_EXPERIMENT=golden-set-eval evaluation/scripts/run_all_evals.sh
#   OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free evaluation/scripts/run_all_evals.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python3 evaluation/scripts/run_eval_encoder.py
python3 evaluation/scripts/run_eval_openrouter.py
python3 evaluation/scripts/run_eval_gemini.py

python3 evaluation/scripts/compare_eval_results.py
