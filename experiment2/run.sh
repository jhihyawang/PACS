#!/bin/bash
# ===============================================================
# run.sh — Train + Test MaMVT (Stage-3 mid fusion, paper settings)
# ===============================================================
# Environment: Swin-Base pretrained on ImageNet-22k, 1-channel input
# Image size 1536×1536, batch size 8, 60 epochs, SGD + momentum 0.9

# ---------------- Basic paths ----------------
ROOT_DIR="../cropped_datasets"                # root where images are stored
TRAIN_CSV="../datasets/train_labels.csv"
VAL_CSV="../datasets/val_labels.csv"
TEST_CSV="../datasets/test_labels.csv"
OUT_DIR="runs/4channel"

# ---------------- Model backbone ----------------
# Options:
#   swin_base_patch4_window12_384_in22k   (Swin-V1, window = 12)
#   swinv2_base_window12to24_192to256_22k (Swin-V2, window = 24)
# BACKBONE="swin_base_patch4_window12_384_in22k"
# BATCH_SIZE=8
# STRUCTURE="swin"

# STRUCTURE="maxvit"
# BACKBONE="maxvit_small_tf_384"
# IMG_SIZE=384
# BATCH_SIZE=2

STRUCTURE="4channel"
BACKBONE="resnet50"
IMG_SIZE=512
BATCH_SIZE=8
EPOCHS=120
LR=1e-4
WD=1e-3
WORKERS=8

# Focal-loss and label smoothing settings
LOSS="focal"          # focal | ce
LABEL_SMOOTH=0.2
IN_CHANS=1            # 1 = grayscale (論文設定) ; 3 = RGB optional
OPTIM="sgd"           # SGD + momentum 0.9 (論文設定)
NO_AMP=""             # set to "--no_amp" if AMP causes issues

# Optional: class weights .npy file (leave empty if not used)
CLASS_WEIGHTS=""

# ---------------- Activate environment ----------------
# Example:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate mamvt
# or: source venv/bin/activate

# ---------------- Run training + testing ----------------
echo "=============================================================="
echo "  Training (Stage-3 mid fusion / paper setup)"
echo "=============================================================="

python train_mvt.py \
  --train_csv "$TRAIN_CSV" \
  --val_csv   "$VAL_CSV" \
  --test_csv  "$TEST_CSV" \
  --root_dir  "$ROOT_DIR" \
  --out_dir   "$OUT_DIR" \
  --backbone  "$BACKBONE" \
  --structure  "$STRUCTURE" \
  --img_size  $IMG_SIZE \
  --in_chans  $IN_CHANS \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --lr $LR \
  --weight_decay $WD \
  --workers $WORKERS \
  --optim $OPTIM \
  --loss $LOSS \
  --label_smoothing $LABEL_SMOOTH \
  $NO_AMP \
  ${CLASS_WEIGHTS:+--class_weights "$CLASS_WEIGHTS"}

# ---------------- Post-training summary ----------------
echo "=============================================================="
echo "  Training finished."
echo "  Results and best checkpoint saved in:"
echo "     $OUT_DIR"
echo "=============================================================="
