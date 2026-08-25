#!/bin/bash
set -e
# FutureTST hourly forecasting pipeline
#
# Runs the full workflow: preprocessing (parquet -> npz), FutureTST training,
# and postprocessing (denormalization + evaluation metrics + plots).
#
# Usage:
#   bash run_forecast.sh                                    # default settings
#   bash run_forecast.sh --windows 168 --pred_len 18        # custom window
#   bash run_forecast.sh --parquet /path/to/data.parquet    # custom raw data
#   bash run_forecast.sh --basins "basin_id1 basin_id2"     # subset of basins
#
# Options:
#   --windows     input window length in hours (default 168)
#   --pred_len    forecast horizon in hours (default 18)
#   --epochs      training epochs (default 200)
#   --patience    early stopping patience (default 20)
#   --parquet     raw parquet data path (default ./data/camelsh_demo.parquet)
#   --npz_path    preprocessed npz path (default ../data_processing/data/prepped.npz)
#   --device      device (default cuda; use CUDA_VISIBLE_DEVICES=N to pick a GPU)
#   --seeds       random seed list (default [1])
#   --step_index  which forecast step to extract in postprocessing (default 0, i.e. 1-step-ahead)
#   --stride      test-set sliding window stride (default 24)
#   --basins      basin ids to use (default: all basins in the parquet)

WINDOWS=168
PRED_LEN=18
EPOCHS=200
PATIENCE=20
PARQUET="./data/camelsh_demo.parquet"
NPZ_PATH="../data_processing/data/prepped.npz"
DEVICE="cuda"
SEEDS="[1]"
STEP_INDEX=0
STRIDE=24
BASINS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --windows)    WINDOWS="$2"; shift 2 ;;
        --pred_len)   PRED_LEN="$2"; shift 2 ;;
        --epochs)     EPOCHS="$2"; shift 2 ;;
        --patience)   PATIENCE="$2"; shift 2 ;;
        --parquet)    PARQUET="$2"; shift 2 ;;
        --npz_path)   NPZ_PATH="$2"; shift 2 ;;
        --device)     DEVICE="$2"; shift 2 ;;
        --seeds)      SEEDS="$2"; shift 2 ;;
        --step_index) STEP_INDEX="$2"; shift 2 ;;
        --stride)     STRIDE="$2"; shift 2 ;;
        --basins)     BASINS="$2"; shift 2 ;;
        -h|--help)
            head -25 "$0" | tail -23
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODEL_DIR="$SCRIPT_DIR/futuretst"
DATA_DIR="$SCRIPT_DIR/data_processing"

# Resolve the parquet path relative to this script's directory
if [[ "$PARQUET" != /* ]]; then
    PARQUET="$SCRIPT_DIR/${PARQUET#./}"
fi

echo "=========================================="
echo "FutureTST Forecasting Pipeline"
echo "=========================================="
echo "Configuration:"
echo "  windows    = $WINDOWS"
echo "  pred_len   = $PRED_LEN"
echo "  epochs     = $EPOCHS"
echo "  patience   = $PATIENCE"
echo "  parquet    = $PARQUET"
echo "  npz_path   = $NPZ_PATH"
echo "  device     = $DEVICE"
echo "  seeds      = $SEEDS"
echo "  stride     = $STRIDE"
echo "  step_index = $STEP_INDEX"
echo "=========================================="
echo ""

echo "[Step 1/3] Preprocessing (parquet -> prepped.npz)..."
echo "=========================================="
cd "$DATA_DIR"
python3 preprocess_camelsh_forecast.py --parquet "$PARQUET" $BASINS
echo ""

cd "$MODEL_DIR"
export PYTHONPATH=./

# batch_size = number of basins
BATCH_SIZE=$(python3 -c "\
import numpy as np; \
print(int(np.load('$NPZ_PATH', allow_pickle=True)['n_segs']))")
echo "Batch size (n_segs): $BATCH_SIZE"
echo ""

echo "[Step 2/3] Training FutureTST forecasting model..."
echo "=========================================="
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 src/experiments/FutureTST_forecast.py \
    --dataset_type=CAMELS \
    --npz_path="$NPZ_PATH" \
    --device="$DEVICE" \
    --batch_size=$BATCH_SIZE \
    --horizon=1 \
    --windows=$WINDOWS \
    --pred_len=$PRED_LEN \
    --epochs=$EPOCHS \
    --patience=$PATIENCE \
    runs --seeds="$SEEDS"

echo ""
echo "[Step 3/3] Postprocessing predictions..."
echo "=========================================="
cd "$SCRIPT_DIR"
python3 data_processing/postprocess_forecast.py \
    --pred_dir "$MODEL_DIR/output/pred" \
    --partition tst \
    --step_index $STEP_INDEX \
    --window $WINDOWS \
    --pred_len $PRED_LEN \
    --stride $STRIDE

echo ""
echo "=========================================="
echo "Forecasting Pipeline Complete!"
echo "=========================================="
echo "Predictions (normalized): $MODEL_DIR/output/pred/"
echo "Predictions (denormalized): $MODEL_DIR/output/denorm/"
echo "Model checkpoints: $MODEL_DIR/results/"
echo "=========================================="
