#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

SAVE_DIR="${PROJECT_DIR}/experiments"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FOLDS="${FOLDS:-0 1 2 3 4}"

run_architecture() {
  local EXP_PREFIX="$1"
  local ARCHITECTURE="$2"

  for FOLD in ${FOLDS}; do
    local SEED=$((42 + FOLD))
    local EXP_ID="${EXP_PREFIX}_fold${FOLD}_seed${SEED}_${RUN_ID}"
    local EXP_DIR="${SAVE_DIR}/${EXP_ID}"
    local FOLD_DIR="csv/e17_folds/fold_${FOLD}"
    mkdir -p "${EXP_DIR}"
    echo "Starting ${EXP_ID}"

    "${PYTHON_BIN}" -u main.py \
      --csv_train "${FOLD_DIR}/train_labels.csv" \
      --csv_val "${FOLD_DIR}/val_labels.csv" \
      --csv_test "${FOLD_DIR}/val_labels.csv" \
      --root_dir data/raw/datasets \
      --save_dir "${SAVE_DIR}" \
      --experiment_id "${EXP_ID}" \
      --num_classes 3 \
      --backbone convnext_tiny \
      --pretrained \
      --architecture "${ARCHITECTURE}" \
      --concate_method 2fc \
      --decision_rule avg \
      --use_class_weights \
      --use_asymmetry_auxiliary \
      --asymmetry_loss_weight 0.3 \
      --img_height 1024 \
      --img_width 512 \
      --batch_size 4 \
      --gradient_accumulation_steps 8 \
      --num_workers 4 \
      --num_epochs 50 \
      --lr 1e-4 \
      --weight_decay 1e-4 \
      --mixed_precision \
      --early_stopping_patience 8 \
      --skip_test \
      --seed "${SEED}" \
      2>&1 | tee "${EXP_DIR}/run.log"

    echo "Completed ${EXP_ID}"
  done
}

run_architecture baseline_aux baseline
run_architecture cross_view_aux cross_view

echo "Matched auxiliary-asymmetry run ID: ${RUN_ID}"
echo "Evaluate baseline + auxiliary: python evaluate.py oof --experiment baseline_aux --run-id ${RUN_ID} --output-dir experiments/baseline_aux_oof_${RUN_ID}"
echo "Evaluate cross-view + auxiliary: python evaluate.py oof --experiment cross_view_aux --run-id ${RUN_ID} --output-dir experiments/cross_view_aux_oof_${RUN_ID}"
