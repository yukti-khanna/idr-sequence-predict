#!/usr/bin/env bash
set -euo pipefail

FEATURES_FILE="${FEATURES_FILE:-inputs/final_all_noNaN.txt}"
MODELS_ROOT="${MODELS_ROOT:-models_all}"
OUT_ROOT="${OUT_ROOT:-results}"
DROP_DIR="${DROP_DIR:-feature_sets}"

MULTS=(1 2 5 10)
FEATS=(69 126)
TAGS=(PTM Random)
CRITS=(f1 max_recall fbeta)

PRED_DIR="${OUT_ROOT}/predictions"
UNIQ_DIR="${OUT_ROOT}/uniqs"

mkdir -p "$PRED_DIR" "$UNIQ_DIR"

uniq_one () {
  local infile="$1"
  local outuniq="$2"
  if [[ -f "$infile" ]]; then
    mkdir -p "$(dirname "$outuniq")"
    sort -u "$infile" > "$outuniq"
  fi
}

echo "Using features: $FEATURES_FILE"
echo "Models root:    $MODELS_ROOT"
echo "Outputs:        $OUT_ROOT"
echo

# ---- LINEAR (3 criteria) ----
echo "=== Predict: linear (f1/max_recall/fbeta) ==="
for crit in "${CRITS[@]}"; do
  for tag in "${TAGS[@]}"; do
    for m in "${MULTS[@]}"; do
      for f in "${FEATS[@]}"; do
        md="${MODELS_ROOT}/linear/${crit}/${tag}/${m}x/feats${f}"
        [[ -d "$md" ]] || { echo "SKIP missing $md"; continue; }

        outdir="${PRED_DIR}/linear/${crit}/${tag}/${m}x/feats${f}"
        mkdir -p "$outdir"

        python scripts/predict_linear.py \
          --features "$FEATURES_FILE" \
          --models-dir "$md" \
          --outdir "$outdir" \
          --tag "$tag" \
          --mult "$m" \
          --feats "$f" \
          --drop-cols-dir "$DROP_DIR" \
          --model-style tagged

        # uniqs (exactly once)
        uniqdir="${UNIQ_DIR}/linear/${crit}/${tag}/${m}x/feats${f}"
        uniq_one "${outdir}/Predicted_TADs_Scaled_SGDC_${tag}${m}_${f}_feats.txt" \
                 "${uniqdir}/Predicted_TADs_Scaled_SGDC_${tag}${m}_${f}_feats__uniq.txt"
        uniq_one "${outdir}/Predicted_TADs_Logistic_Regression_${tag}${m}_${f}_feats.txt" \
                 "${uniqdir}/Predicted_TADs_Logistic_Regression_${tag}${m}_${f}_feats__uniq.txt"
        uniq_one "${outdir}/Predicted_TADs_Partial_Fit_SGDC_${tag}${m}_${f}_feats.txt" \
                 "${uniqdir}/Predicted_TADs_Partial_Fit_SGDC_${tag}${m}_${f}_feats__uniq.txt"
      done
    done
  done
done
echo

# ---- NN (1 variation only) ----
echo "=== Predict: nn ==="
for tag in "${TAGS[@]}"; do
  for m in "${MULTS[@]}"; do
    for f in "${FEATS[@]}"; do
      md="${MODELS_ROOT}/nn/${tag}/${m}x/feats${f}"
      model="${md}/NN_${tag}_${f}_feats_model.sav"
      [[ -f "$model" ]] || { echo "SKIP missing $model"; continue; }

      outdir="${PRED_DIR}/nn/${tag}/${m}x/feats${f}"
      mkdir -p "$outdir"

      python scripts/predict_nn.py \
        --features "$FEATURES_FILE" \
        --model "$model" \
        --outdir "$outdir" \
        --tag "$tag" \
        --mult "$m" \
        --feats "$f" \
        --drop-cols-dir "$DROP_DIR"

      uniqdir="${UNIQ_DIR}/nn/${tag}/${m}x/feats${f}"
      uniq_one "${outdir}/Predicted_TADs_NN_${tag}${m}_${f}_feats.txt" \
               "${uniqdir}/Predicted_TADs_NN_${tag}${m}_${f}_feats__uniq.txt"
    done
  done
done
echo

# ---- SMOTEENN (BIG only) ----
echo "=== Predict: smoteenn (BIG only) ==="
for tag in "${TAGS[@]}"; do
  for f in "${FEATS[@]}"; do
    md="${MODELS_ROOT}/smoteenn_big/${tag}/feats${f}"
    [[ -d "$md" ]] || { echo "SKIP missing $md"; continue; }

    outdir="${PRED_DIR}/smoteenn_big/${tag}/feats${f}"
    mkdir -p "$outdir"

    python scripts/predict_smoteenn.py \
      --features "$FEATURES_FILE" \
      --models-dir "$md" \
      --outdir "$outdir" \
      --tag "$tag" \
      --feats "$f" \
      --drop-cols-dir "$DROP_DIR"

    uniqdir="${UNIQ_DIR}/smoteenn_big/${tag}/feats${f}"
    uniq_one "${outdir}/Predicted_TADs_Logistic_Regression_${tag}_BIG_${f}_feats.txt" \
             "${uniqdir}/Predicted_TADs_Logistic_Regression_${tag}_BIG_${f}_feats__uniq.txt"
    uniq_one "${outdir}/Predicted_TADs_Partial_Fit_SGDC_${tag}_BIG_${f}_feats.txt" \
             "${uniqdir}/Predicted_TADs_Partial_Fit_SGDC_${tag}_BIG_${f}_feats__uniq.txt"
    uniq_one "${outdir}/Predicted_TADs_Scaled_SGDC_${tag}_BIG_${f}_feats.txt" \
             "${uniqdir}/Predicted_TADs_Scaled_SGDC_${tag}_BIG_${f}_feats__uniq.txt"
  done
done

echo
echo "All predictions + uniq lists written under: $OUT_ROOT"
