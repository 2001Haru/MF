#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   DATA_DIR=/linxi/dataset/imagenet256_sd_ema_lmdb/train \
#   MAX_TRAIN_STEPS=20000 \
#   bash scripts/run_euler_phase1.sh bridge
#
# Available runs:
#   bridge              Existing JVP MeanFlow velocity baseline
#   euler-u             EulerMF velocity prediction
#   euler-u-weighted    EulerMF velocity prediction with endpoint-equivalent t^2 weight
#   euler-endpoint      EulerMF endpoint-like prediction

RUN_NAME="${1:?Choose one of: bridge, euler-u, euler-u-weighted, euler-endpoint}"
DATA_DIR="${DATA_DIR:-/linxi/dataset/imagenet256_sd_ema_lmdb/train}"
OUTPUT_DIR="${OUTPUT_DIR:-exp/euler-phase1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-5000}"
SAMPLING_STEPS="${SAMPLING_STEPS:-1000}"
EULER_DT="${EULER_DT:-0.01}"
MULTI_GPU="${MULTI_GPU:-1}"

COMMON_ARGS=(
  --exp-name "${RUN_NAME}"
  --output-dir "${OUTPUT_DIR}"
  --data-dir "${DATA_DIR}"
  --model "SiT-B/2"
  --resolution 256
  --batch-size "${BATCH_SIZE}"
  --max-train-steps "${MAX_TRAIN_STEPS}"
  --checkpointing-steps "${CHECKPOINTING_STEPS}"
  --sampling-steps "${SAMPLING_STEPS}"
  --num-workers "${NUM_WORKERS}"
  --allow-tf32
  --mixed-precision "fp16"
  --path-type "linear"
  --loss-type "l2"
  --time-sampler "uniform"
  --ratio-r-not-equal-t 0.75
  --cfg-prob 0.0
  --no-class-consist
  --no-debug
)

case "${RUN_NAME}" in
  bridge)
    EXTRA_ARGS=(
      --model-name "meanflow"
      --prediction-type "velocity"
      --loss-time-weight "none"
    )
    ;;
  euler-u)
    EXTRA_ARGS=(
      --model-name "euler_meanflow"
      --prediction-type "velocity"
      --loss-time-weight "none"
      --euler-dt "${EULER_DT}"
    )
    ;;
  euler-u-weighted)
    EXTRA_ARGS=(
      --model-name "euler_meanflow"
      --prediction-type "velocity"
      --loss-time-weight "endpoint"
      --euler-dt "${EULER_DT}"
    )
    ;;
  euler-endpoint)
    EXTRA_ARGS=(
      --model-name "euler_meanflow"
      --prediction-type "endpoint"
      --loss-time-weight "none"
      --euler-dt "${EULER_DT}"
    )
    ;;
  *)
    echo "Unknown run '${RUN_NAME}'. Choose: bridge, euler-u, euler-u-weighted, euler-endpoint" >&2
    exit 2
    ;;
esac

LAUNCH_ARGS=()
if [[ "${MULTI_GPU}" == "1" ]]; then
  LAUNCH_ARGS+=(--multi_gpu)
fi

accelerate launch "${LAUNCH_ARGS[@]}" train.py "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
