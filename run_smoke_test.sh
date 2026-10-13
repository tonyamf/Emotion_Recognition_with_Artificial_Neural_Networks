#!/usr/bin/env bash
# Executes cwANN_.ipynb end to end on a tiny SYNTHETIC dataset (EMOTION_SMOKE=1, see smoke_data.py).
# No FER2013 download, 1 epoch. Proves the code runs; the accuracy it prints is meaningless.
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:-$(mktemp -d)}"
EMOTION_SMOKE=1 TF_CPP_MIN_LOG_LEVEL=3 jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=600 cwANN_.ipynb --output-dir "$OUT" --output smoke_run.ipynb
echo "smoke run OK -> $OUT/smoke_run.ipynb"
