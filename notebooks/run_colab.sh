#!/usr/bin/env bash
# Run preparation and training in Colab-friendly modes
set -euo pipefail

# Load defaults
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/colab_config.sh"

usage() {
  cat <<EOF
Usage: $0 --mode [smoke|full|prepare-only|train-base|train-refiner|train-vae|evaluate]

Modes:
  smoke         Prepare dataset and run 2-epoch smoke trains to validate pipeline (default)
  full          Run full pipeline: prepare -> train-base -> evaluate -> train-refiner -> train-vae
  prepare-only  Only run dataset preparation
  train-base    Only train base autoencoder
  train-refiner Only train refiner (requires base checkpoint in RESULTS_DIR)
  train-vae     Only train VAE
  evaluate      Run evaluation using available checkpoints

Examples:
  bash run_colab.sh --mode smoke
  bash run_colab.sh --mode full
EOF
}

MODE=${1:-""}
if [[ "$MODE" == "--mode" ]]; then
  MODE=${2:-"smoke"}
fi
if [[ -z "$MODE" ]]; then
  MODE="smoke"
fi

echo "Running mode=$MODE"

prepare_dataset() {
  echo "Preparing dataset..."
  python3 src/prepare_dataset.py --shapr_dir "${DATA_DIR}/raw" --output_dir "${PROCESSED_DIR}" --input_mode "${INPUT_MODE}"
}

train_base() {
  echo "Training base autoencoder (smoke)..."
  python3 src/train_reconstruction.py \
    --data_dir "${PROCESSED_DIR}" \
    --input_mode "${INPUT_MODE}" \
    --epochs ${BASE_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --output_dir "${RESULTS_DIR}"
}

train_refiner() {
  echo "Training refiner (smoke)..."
  python3 src/train_refiner.py \
    --data_dir "${PROCESSED_DIR}" \
    --base_model "${RESULTS_DIR}/best_autoencoder.pt" \
    --input_mode "${INPUT_MODE}" \
    --epochs ${REFINER_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --output_dir "${RESULTS_DIR}"
}

train_vae() {
  echo "Training VAE (smoke)..."
  python3 src/train_vae.py \
    --data_dir "${PROCESSED_DIR}" \
    --input_mode "${INPUT_MODE}" \
    --epochs ${VAE_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --eval_samples_k ${EVAL_SAMPLES_K} \
    --output_dir "${RESULTS_DIR}"
}

evaluate() {
  echo "Running evaluation..."
  python3 src/evaluate.py \
    --data_dir "${PROCESSED_DIR}" \
    --autoencoder "${RESULTS_DIR}/best_autoencoder.pt" \
    --refiner "${RESULTS_DIR}/best_refiner.pt" \
    --input_mode "${INPUT_MODE}" \
    --output_csv "${RESULTS_DIR}/evaluation_summary.csv"
}

case "$MODE" in
  smoke)
    prepare_dataset
    train_base
    train_refiner
    ;;
  full)
    prepare_dataset
    train_base
    evaluate
    train_refiner
    train_vae
    evaluate
    ;;
  prepare-only)
    prepare_dataset
    ;;
  train-base)
    train_base
    ;;
  train-refiner)
    train_refiner
    ;;
  train-vae)
    train_vae
    ;;
  evaluate)
    evaluate
    ;;
  *)
    usage
    exit 1
    ;;
esac

echo "Done mode=$MODE"
