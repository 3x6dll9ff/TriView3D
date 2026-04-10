#!/usr/bin/env bash
# Default configuration for running in Colab. Edit as needed.

# Base dirs (relative or absolute)
DATA_DIR="/content/data"
OUTPUT_DIR="/content/output"
REPO_DIR="/content/repo"

# Data preparation
INPUT_MODE="quad"    # tri or quad

# Training hyperparams (safe defaults for Colab T4)
BASE_EPOCHS=2
REFINER_EPOCHS=2
VAE_EPOCHS=2
BATCH_SIZE=4
LR=1e-3

# VAE evaluation samples
EVAL_SAMPLES_K=4

# Device: colab provides CUDA GPU; set to cpu if not available
DEVICE="cuda"

# Save paths
PROCESSED_DIR="${DATA_DIR}/processed"
RESULTS_DIR="${OUTPUT_DIR}/results"

mkdir -p "${PROCESSED_DIR}" "${RESULTS_DIR}"

echo "Config: DATA_DIR=${DATA_DIR}, OUTPUT_DIR=${OUTPUT_DIR}, INPUT_MODE=${INPUT_MODE}"
