#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-inputs}"
MODELS_ROOT="${MODELS_ROOT:-models_all}"
DROP_DIR="${DROP_DIR:-feature_sets}"

echo "=== Training Linear + NN ==="
DATA_DIR="$DATA_DIR" OUT_ROOT="$MODELS_ROOT" DROP_DIR="$DROP_DIR" ./scripts/run_train_linear_nn_all.sh

echo
echo "=== Training SMOTEENN ==="
DATA_DIR="$DATA_DIR" MODELS_ROOT="$MODELS_ROOT" DROP_DIR="$DROP_DIR" ./scripts/run_train_smoteenn_all.sh

echo
echo "ALL TRAINING DONE. Models are under: $MODELS_ROOT/{linear,nn,smoteenn}"
