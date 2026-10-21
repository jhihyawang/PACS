#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

SAVE_DIR="${PROJECT_DIR}/experiments"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FOLDS="${FOLDS:-0 1 2 3 4}"

for FOLD in ${FOLDS}; do
  SEED=$((42 + FOLD))
  EXP_ID="conditional_aux_cv_fold${FOLD}_seed${SEED}_${RUN_ID}"
  EXP_DIR="${SAVE_DIR}/${EXP_ID}"
  FOLD_DIR="csv/e17_folds/fold_${FOLD}"
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
    --architecture baseline \
    --concate_method concat \
    --decision_rule conditional \
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

  echo "Experiment completed: ${EXP_DIR}"
done

echo "Conditional + auxiliary run ID: ${RUN_ID}"
echo "Evaluate conditional + auxiliary: python evaluate.py oof --experiment conditional_aux --run-id ${RUN_ID} --output-dir experiments/conditional_aux_oof_${RUN_ID}"
