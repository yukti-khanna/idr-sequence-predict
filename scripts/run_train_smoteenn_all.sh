#!/usr/bin/env bash
set -euo pipefail

# Train SMOTEENN models for PTM/Random, multipliers 1/2/5/10, feats 69/126.
# Saves into models_all/smoteenn/<TAG>/<mult>x/feats<feats>/

DATA_DIR="${DATA_DIR:-inputs}"
MODELS_ROOT="${MODELS_ROOT:-models_all}"
DROP_DIR="${DROP_DIR:-feature_sets}"

MULTS=(1 2 5 10)
FEATS=(69 126)
TAGS=(PTM Random)

ptm_file () { local m="$1"; [[ "$m" == "1" ]] && echo "${DATA_DIR}/ptms_short.txt" || echo "${DATA_DIR}/ptms_short${m}.txt"; }
ran_file () { local m="$1"; [[ "$m" == "1" ]] && echo "${DATA_DIR}/random_short.txt" || echo "${DATA_DIR}/random_short${m}.txt"; }

mkdir -p "${MODELS_ROOT}/smoteenn"

for tag in "${TAGS[@]}"; do
  for m in "${MULTS[@]}"; do
    for f in "${FEATS[@]}"; do

      if [[ "$tag" == "PTM" ]]; then
        INP="$(ptm_file "$m")"
      else
        INP="$(ran_file "$m")"
      fi

      if [[ ! -f "$INP" ]]; then
        echo "SKIP: missing $INP"
        continue
      fi

      OUT_DIR="${MODELS_ROOT}/smoteenn/${tag}/${m}x/feats${f}"
      mkdir -p "$OUT_DIR"

      echo "Training SMOTEENN: ${tag} ${m}x feats${f} -> ${OUT_DIR}"
      python scripts/train_smoteenn.py \
        --train "$INP" \
        --outdir "$OUT_DIR" \
        --tag "$tag" \
        --feats "$f" \
        --drop-cols-dir "$DROP_DIR"

    done
  done
done

echo "Done. SMOTEENN models saved under: ${MODELS_ROOT}/smoteenn"
