#!/usr/bin/env bash
set -euo pipefail

EXTERNAL_DIR="${EXTERNAL_DIR:-external}"
IDR_REPO_DIR="${IDR_REPO_DIR:-${EXTERNAL_DIR}/idr.mol.feats}"
IDR_REPO_URL="${IDR_REPO_URL:-https://github.com/IPritisanac/idr.mol.feats}"

mkdir -p "$EXTERNAL_DIR"

if [[ -d "$IDR_REPO_DIR/.git" ]]; then
  echo "[setup] idr.mol.feats already present at: $IDR_REPO_DIR"
  echo "[setup] Updating (git pull)..."
  git -C "$IDR_REPO_DIR" pull --ff-only
else
  echo "[setup] Cloning idr.mol.feats into: $IDR_REPO_DIR"
  git clone "$IDR_REPO_URL" "$IDR_REPO_DIR"
fi

mkdir -p "$IDR_REPO_DIR/DATA" "$IDR_REPO_DIR/RES"

echo
echo "[setup] Done."
echo "[setup] Set in config.yaml:"
echo "  idr_mol_feats_repo: \"$IDR_REPO_DIR\""
