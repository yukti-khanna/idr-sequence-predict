#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-inputs}"
OUT_ROOT="${OUT_ROOT:-trained_models}"
DROP_DIR="${DROP_DIR:-feature_sets}"

MULTS=(1 2 5 10)
FEATS=(69 126)

ptm_file () { local m="$1"; [[ "$m" == "1" ]] && echo "${DATA_DIR}/ptms_short.txt" || echo "${DATA_DIR}/ptms_short${m}.txt"; }
ran_file () { local m="$1"; [[ "$m" == "1" ]] && echo "${DATA_DIR}/random_short.txt" || echo "${DATA_DIR}/random_short${m}.txt"; }

mkdir -p "$OUT_ROOT"

for m in "${MULTS[@]}"; do
  for f in "${FEATS[@]}"; do

    # ---- PTM ----
    PTM_IN="$(ptm_file "$m")"
    if [[ -f "$PTM_IN" ]]; then
      echo "[Linear] PTM ${m}x feats${f}"
      python scripts/train_linear.py \
        --train "$PTM_IN" \
        --outdir "${OUT_ROOT}/linear/PTM/${m}x/feats${f}" \
        --tag PTM --feats "$f" --drop-cols-dir "$DROP_DIR"

      echo "[NN] PTM ${m}x feats${f}"
      python scripts/train_nn.py \
        --train "$PTM_IN" \
        --outdir "${OUT_ROOT}/nn/PTM/${m}x/feats${f}" \
        --tag PTM --feats "$f" --drop-cols-dir "$DROP_DIR"
    else
      echo "SKIP PTM ${m}x feats${f} (missing $PTM_IN)"
    fi

    # ---- Random ----
    RAN_IN="$(ran_file "$m")"
    if [[ -f "$RAN_IN" ]]; then
      echo "[Linear] Random ${m}x feats${f}"
      python scripts/train_linear.py \
        --train "$RAN_IN" \
        --outdir "${OUT_ROOT}/linear/Random/${m}x/feats${f}" \
        --tag Random --feats "$f" --drop-cols-dir "$DROP_DIR"

      echo "[NN] Random ${m}x feats${f}"
      python scripts/train_nn.py \
        --train "$RAN_IN" \
        --outdir "${OUT_ROOT}/nn/Random/${m}x/feats${f}" \
        --tag Random --feats "$f" --drop-cols-dir "$DROP_DIR"
    else
      echo "SKIP Random ${m}x feats${f} (missing $RAN_IN)"
    fi

    echo
  done
done

echo "Done. Linear+NN models saved under: $OUT_ROOT"
