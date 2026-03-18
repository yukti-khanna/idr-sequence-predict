#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-inputs}"
HP="${HP:-models_all/hparams.json}"

python scripts/train_linear_from_hparams.py --data-dir "$DATA_DIR" --hparams "$HP"
python scripts/train_nn_from_hparams.py --data-dir "$DATA_DIR" --hparams "$HP"
python scripts/train_smoteenn_big_from_hparams.py --data-dir "$DATA_DIR" --hparams "$HP"

echo "DONE training from cached hyperparams (SMOTEENN = BIG ONLY)."
