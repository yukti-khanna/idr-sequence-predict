#!/usr/bin/env bash
set -euo pipefail

IDR_FEATS_REPO="${IDR_FEATS_REPO:-../idr.mol.feats}"
LIST_FILE="${1:-input_file_feats.txt}"

# run inside the idr.mol.feats repo (it expects DATA/ and writes RES/)
cd "$IDR_FEATS_REPO"
python run_feats.py "$LIST_FILE"

echo "Done. Outputs should be in: $IDR_FEATS_REPO/RES/"
