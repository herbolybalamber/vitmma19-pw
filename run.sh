#!/usr/bin/env bash
# run.sh - run the full pipeline scripts in order
# This script executes the main pipeline stages in sequence.

set -euo pipefail

echo "[run.sh] Starting full pipeline run at $(date --iso-8601=seconds)"

python -B -m src.00_aggregate_jsons
python -B -m src.01_preprocess
python -B -m src.02_train
python -B -m src.03_evaluation
python -B -m src.04_inference

echo "[run.sh] Pipeline finished at $(date --iso-8601=seconds)"
