#!/bin/bash
# ===========================================
# run.sh  —  Train + Test MaMVT (stage-3 mid fusion)
# ===========================================
# Environment: assumed to match the paper’s hardware
# (Swin-Base @ 1536² input, batch size ≈4, mixed precision)

# ---- Basic configuration ----
ROOT_DIR="../datasets"                     # root where images are stored
TRAIN_CSV="../datasets/train_labels.csv"
VAL_CSV="../datasets/val_labels.csv"
TEST_CSV="../datasets/test_labels.csv"
OUT_DIR="runs/mamvt_stage3"
BACKBONE="swin_base_patch4_window12_384"

IMG_SIZE=384
BATCH_SIZE=4
EPOCHS=30
LR=1e-4
WD=1e-4
WORKERS=6

# Optional: class weights .npy file (leave empty if not used)
CLASS_WEIGHTS=""

# ---- Activate environment ----
# Example:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate mamvt
# or: source venv/bin/activate

# ---- Run training + testing ----
echo "==========================================="
echo " Training MaMVT (Stage-3 mid fusion)"
echo "==========================================="

python train_mvt.py \
  --train_csv "$TRAIN_CSV" \
  --val_csv   "$VAL_CSV" \
  --test_csv  "$TEST_CSV" \
  --root_dir  "$ROOT_DIR" \
  --out_dir   "$OUT_DIR" \
  --backbone  "$BACKBONE" \
  --img_size  $IMG_SIZE \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --lr $LR \
  --weight_decay $WD \
  --workers $WORKERS \
  ${CLASS_WEIGHTS:+--class_weights "$CLASS_WEIGHTS"}

# ---- Post-training summary ----
echo "==========================================="
echo " Training finished."
echo " Results and best checkpoint saved in:"
echo "   $OUT_DIR"
echo "==========================================="
