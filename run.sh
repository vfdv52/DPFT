#!/usr/bin/env bash
# run.sh — One-shot training and evaluation with configurable hyperparameters.
# Usage: bash run.sh [options]
#
# Example (override epochs and learning rate):
#   bash run.sh --epochs 100 --lr 0.0005

set -e

PYTHON=python

# Redirect all output (stdout + stderr) to log file, overwriting each run
LOG_FILE=results/run.log
mkdir -p results
exec > >(tee "$LOG_FILE") 2>&1
echo "Log: $LOG_FILE  ($(date))"

# Default hyperparameters (edit here or pass as CLI flags)
INPUT_LEN=96        # look-back window (steps)
PRED_LEN=24         # forecast horizon (steps)
BATCH_SIZE=32
EPOCHS=30
LR=0.001
PATIENCE=7          # ReduceLROnPlateau patience

HIDDEN_SIZE=64      # LSTM hidden size
D_MODEL=64          # Transformer model dimension
NHEAD=4             # Transformer attention heads
NUM_LAYERS=2        # Number of encoder / LSTM layers
DIM_FEEDFORWARD=256 # Transformer feedforward dimension
DROPOUT=0.2

MC_SAMPLES=50       # MC Dropout forward passes at test time
EARLY_STOP=15       # early stopping patience (epochs)

DATASETS="ETTh1 ETTh2"
TARGET_COL="OT"
RUN_NAME="default"
CLIP_GRAD=1.0
RUN_GNN=false
RUN_INFORMER=false

# Parse optional CLI overrides
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input_len)      INPUT_LEN="$2";       shift 2 ;;
        --pred_len)       PRED_LEN="$2";        shift 2 ;;
        --batch_size)     BATCH_SIZE="$2";      shift 2 ;;
        --epochs)         EPOCHS="$2";          shift 2 ;;
        --lr)             LR="$2";              shift 2 ;;
        --patience)       PATIENCE="$2";        shift 2 ;;
        --hidden_size)    HIDDEN_SIZE="$2";     shift 2 ;;
        --d_model)        D_MODEL="$2";         shift 2 ;;
        --nhead)          NHEAD="$2";           shift 2 ;;
        --num_layers)     NUM_LAYERS="$2";      shift 2 ;;
        --dim_feedforward) DIM_FEEDFORWARD="$2"; shift 2 ;;
        --dropout)        DROPOUT="$2";         shift 2 ;;
        --mc_samples)     MC_SAMPLES="$2";      shift 2 ;;
        --early_stop)     EARLY_STOP="$2";      shift 2 ;;
        --datasets)       DATASETS="$2";        shift 2 ;;
        --target_col)     TARGET_COL="$2";      shift 2 ;;
        --run_name)       RUN_NAME="$2";        shift 2 ;;
        --clip_grad)      CLIP_GRAD="$2";       shift 2 ;;
        --run_gnn)        RUN_GNN=true;         shift ;;
        --run_informer)   RUN_INFORMER=true;    shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Print config
echo "  COMP0197 CW2 — Probabilistic Time Series Forecasting"
echo "============================================================"
echo "  Datasets      : $DATASETS"
echo "  Target column : $TARGET_COL"
echo "  Input len     : $INPUT_LEN  |  Pred len: $PRED_LEN"
echo "  Batch size    : $BATCH_SIZE |  Epochs : $EPOCHS"
echo "  LR            : $LR        |  Patience: $PATIENCE"
echo "  LSTM hidden   : $HIDDEN_SIZE"
echo "  d_model       : $D_MODEL   |  nhead   : $NHEAD"
echo "  Layers        : $NUM_LAYERS |  FF dim  : $DIM_FEEDFORWARD"
echo "  Dropout       : $DROPOUT   |  MC samp.: $MC_SAMPLES"
echo "============================================================"

# Train
echo ""
echo "[1/2] Training..."
TRAIN_EXTRA=""
[[ "$RUN_GNN"      == "true" ]] && TRAIN_EXTRA="$TRAIN_EXTRA --run_gnn"
[[ "$RUN_INFORMER" == "true" ]] && TRAIN_EXTRA="$TRAIN_EXTRA --run_informer"

$PYTHON train.py \
    --input_len      "$INPUT_LEN"      \
    --pred_len       "$PRED_LEN"       \
    --batch_size     "$BATCH_SIZE"     \
    --epochs         "$EPOCHS"         \
    --lr             "$LR"             \
    --patience       "$PATIENCE"       \
    --hidden_size    "$HIDDEN_SIZE"    \
    --d_model        "$D_MODEL"        \
    --nhead          "$NHEAD"          \
    --num_layers     "$NUM_LAYERS"     \
    --dim_feedforward "$DIM_FEEDFORWARD" \
    --dropout        "$DROPOUT"        \
    --datasets       "$DATASETS"       \
    --target_col     "$TARGET_COL"     \
    --run_name       "$RUN_NAME"       \
    --clip_grad      "$CLIP_GRAD"      \
    --early_stop     "$EARLY_STOP"     \
    $TRAIN_EXTRA

# Test
echo ""
echo "[2/2] Evaluating..."
$PYTHON test.py \
    --input_len      "$INPUT_LEN"      \
    --pred_len       "$PRED_LEN"       \
    --batch_size     "$BATCH_SIZE"     \
    --hidden_size    "$HIDDEN_SIZE"    \
    --d_model        "$D_MODEL"        \
    --nhead          "$NHEAD"          \
    --num_layers     "$NUM_LAYERS"     \
    --dim_feedforward "$DIM_FEEDFORWARD" \
    --dropout        "$DROPOUT"        \
    --mc_samples     "$MC_SAMPLES"     \
    --datasets       "$DATASETS"       \
    --target_col     "$TARGET_COL"     \
    --run_name       "$RUN_NAME"       \
    $TRAIN_EXTRA

echo ""
echo "Done. Results saved to ./results/"
