#!/usr/bin/env bash
set -euo pipefail

# Default shots to evaluate.
SHOTS="${SHOTS:-1 2 4 8 16}"

# Summary CSV (appended). Set empty to disable.
SHOT_METRICS_CSV="${SHOT_METRICS_CSV:-/nas_homes/dongjin/SOFS/shot_metrics_dusan_hole.csv}"

# If you want a train-count sweep (1,2,4,8,16) with metrics on both train/test splits:
#   SWEEP_COUNTS="1,2,4,8,16" SWEEP_TEST_COUNT=14 SWEEP_EPOCHS=50 bash test_dusan_hole.sh
if [[ -n "${SWEEP_COUNTS:-}" ]]; then
  SWEEP_TEST_COUNT="${SWEEP_TEST_COUNT:-14}"
  SWEEP_EPOCHS_OPT=()
  if [[ -n "${SWEEP_EPOCHS:-}" ]]; then
    SWEEP_EPOCHS_OPT=(--epochs "${SWEEP_EPOCHS}")
  fi
  CUDA_VISIBLE_DEVICES=0 python3 /nas_homes/dongjin/SOFS/tools/sweep_custom_counts.py \
    --cfg_train /nas_homes/dongjin/SOFS/method_config/CUSTOM/Train/SOFS_dusan_hole.yaml \
    --counts "${SWEEP_COUNTS}" \
    --test_count "${SWEEP_TEST_COUNT}" \
    --device "0" \
    "${SWEEP_EPOCHS_OPT[@]}"
  exit 0
fi

# Use the latest trained checkpoint from the training run.
TRAIN_OUTPUT_DIR="/nas_homes/dongjin/SOFS/log_CUSTOM_DUSAN_CUSTOM_MASK_ND_SOFS_54"
CKPT="$(ls -t "${TRAIN_OUTPUT_DIR}"/checkpoints/*.pth 2>/dev/null | head -n 1 || true)"

if [[ -z "${CKPT}" ]]; then
  echo "[ERROR] No checkpoint found under: ${TRAIN_OUTPUT_DIR}/checkpoints/"
  echo "        Please run training first (bash train_dusan_hole.sh) and ensure it finishes and saves a .pth."
  exit 1
fi

echo "[INFO] Using checkpoint: ${CKPT}"

if [[ -n "${SHOT_METRICS_CSV}" && ! -f "${SHOT_METRICS_CSV}" ]]; then
  echo "shot,s_in_shot,pixel_acc,dice,mIoU,output_dir,ckpt" > "${SHOT_METRICS_CSV}"
fi

for SHOT in ${SHOTS}; do
  echo ""
  echo "===================="
  echo "[INFO] Evaluating ${SHOT}-shot"
  echo "===================="

  # Separate output dir per shot so figure_save doesn't mix.
  BASE_OUT="./log_CUSTOM_DUSAN_TEST_SHOT_${SHOT}"
  LOG_FILE="/nas_homes/dongjin/SOFS/test_shot_${SHOT}.log"

  # Run test and capture logs (for parsing [SHOT_METRICS] line).
  CUDA_VISIBLE_DEVICES=0 python3 /nas_homes/dongjin/SOFS/main.py --device "0" \
    --cfg /nas_homes/dongjin/SOFS/method_config/CUSTOM/Test/SOFS_dusan_hole_test.yaml --prior_layer_pointer 5 6 7 8 9 10 \
    --opts NUM_GPUS 1 DEVICE 0 RNG_SEED 54 \
    OUTPUT_DIR "${BASE_OUT}" \
    DATASET.shot "${SHOT}" \
    DATASET.custom_support_from_train True \
    TEST.load_checkpoint True TEST.load_model_path "${CKPT}" \
    TEST.VISUALIZE.save_figure True TEST.VISUALIZE.sample_prob 1.0 \
    TRAIN.enable False TEST.enable True \
    | tee "${LOG_FILE}"

  # Parse the last [SHOT_METRICS] line and append to CSV.
  if [[ -n "${SHOT_METRICS_CSV}" ]]; then
    METRIC_LINE="$(grep -a "\\[SHOT_METRICS\\]" "${LOG_FILE}" | tail -n 1 || true)"
    if [[ -n "${METRIC_LINE}" ]]; then
      # Example:
      # [SHOT_METRICS] shot=1 s_in_shot=4 pixel_acc=0.123456 dice=0.123456 mIoU=0.123456
      SHOT_VAL="$(echo "${METRIC_LINE}" | sed -n 's/.*shot=\\([0-9]\\+\\).*/\\1/p')"
      SINSHOT_VAL="$(echo "${METRIC_LINE}" | sed -n 's/.*s_in_shot=\\([0-9]\\+\\).*/\\1/p')"
      PACC_VAL="$(echo "${METRIC_LINE}" | sed -n 's/.*pixel_acc=\\([0-9.]*\\).*/\\1/p')"
      DICE_VAL="$(echo "${METRIC_LINE}" | sed -n 's/.*dice=\\([0-9.]*\\).*/\\1/p')"
      MIOU_VAL="$(echo "${METRIC_LINE}" | sed -n 's/.*mIoU=\\([0-9.]*\\).*/\\1/p')"
      OUT_DIR_FULL="$(python3 -c "import os; print(os.path.abspath('${BASE_OUT}' + '_CUSTOM_MASK_ND_SOFS_54'))")"
      echo "${SHOT_VAL},${SINSHOT_VAL},${PACC_VAL},${DICE_VAL},${MIOU_VAL},${OUT_DIR_FULL},${CKPT}" >> "${SHOT_METRICS_CSV}"
      echo "[INFO] Appended metrics to ${SHOT_METRICS_CSV}"
    else
      echo "[WARN] No [SHOT_METRICS] line found in ${LOG_FILE}; CSV not updated for shot=${SHOT}"
    fi
  fi
done


