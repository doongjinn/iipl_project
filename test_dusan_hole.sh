#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root (directory containing this script) so the script works after `git clone` anywhere.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# Default shots to evaluate.
SHOTS="${SHOTS:-1 2 4 8 16}"

# Summary CSV (appended). Set empty to disable.
SHOT_METRICS_CSV="${SHOT_METRICS_CSV:-${REPO_ROOT}/shot_metrics_dusan_hole.csv}"

# Paths (override via env vars if needed)
CFG_TEST="${CFG_TEST:-${REPO_ROOT}/method_config/CUSTOM/Test/SOFS_dusan_hole_test.yaml}"
CFG_TRAIN="${CFG_TRAIN:-${REPO_ROOT}/method_config/CUSTOM/Train/SOFS_dusan_hole.yaml}"

# DINOv2 backbone checkpoint (expected to be present in this repo; override if you keep it elsewhere)
if [[ -n "${DINO_CKPT:-}" ]]; then
  : # use provided env var
elif [[ -f "${REPO_ROOT}/dinov2_vitb14_pretrain.pth" ]]; then
  DINO_CKPT="${REPO_ROOT}/dinov2_vitb14_pretrain.pth"
elif [[ -f "${REPO_ROOT}/checkpoints/dinov2_vitb14_pretrain.pth" ]]; then
  DINO_CKPT="${REPO_ROOT}/checkpoints/dinov2_vitb14_pretrain.pth"
else
  DINO_CKPT=""
fi

# If you want a train-count sweep (1,2,4,8,16) with metrics on both train/test splits:
#   SWEEP_COUNTS="1,2,4,8,16" SWEEP_TEST_COUNT=14 SWEEP_EPOCHS=50 bash test_dusan_hole.sh
if [[ -n "${SWEEP_COUNTS:-}" ]]; then
  SWEEP_TEST_COUNT="${SWEEP_TEST_COUNT:-14}"
  SWEEP_EPOCHS_OPT=()
  if [[ -n "${SWEEP_EPOCHS:-}" ]]; then
    SWEEP_EPOCHS_OPT=(--epochs "${SWEEP_EPOCHS}")
  fi
  CUDA_VISIBLE_DEVICES=0 python3 "${REPO_ROOT}/tools/sweep_custom_counts.py" \
    --cfg_train "${CFG_TRAIN}" \
    --counts "${SWEEP_COUNTS}" \
    --test_count "${SWEEP_TEST_COUNT}" \
    --device "0" \
    "${SWEEP_EPOCHS_OPT[@]}"
  exit 0
fi

#
# Model checkpoint selection
# - Default: use ./checkpoints/last.pth (ships with this repo in your setup)
# - Override: set CKPT_PATH=/abs/path/to/model.pth
#
CKPT_PATH="${CKPT_PATH:-}"
if [[ -z "${CKPT_PATH}" ]]; then
  if [[ -f "${REPO_ROOT}/checkpoints/last.pth" ]]; then
    CKPT_PATH="${REPO_ROOT}/checkpoints/last.pth"
  elif [[ -d "${REPO_ROOT}/checkpoints" ]]; then
    CKPT_PATH="$(ls -t "${REPO_ROOT}"/checkpoints/*.pth 2>/dev/null | grep -v "dinov2_vitb14_pretrain\\.pth" | head -n 1 || true)"
  fi
fi
if [[ -z "${CKPT_PATH}" ]]; then
  echo "[ERROR] No checkpoint found."
  echo "        Expected one of:"
  echo "          - ${REPO_ROOT}/checkpoints/last.pth"
  echo "          - any *.pth under ${REPO_ROOT}/checkpoints/"
  echo "        Or set CKPT_PATH=/path/to/your_model.pth"
  exit 1
fi

echo "[INFO] Using checkpoint: ${CKPT_PATH}"

if [[ -z "${DINO_CKPT}" ]]; then
  echo "[WARN] DINO_CKPT not found under repo root."
  echo "       Expected one of:"
  echo "         - ${REPO_ROOT}/dinov2_vitb14_pretrain.pth"
  echo "         - ${REPO_ROOT}/checkpoints/dinov2_vitb14_pretrain.pth"
  echo "       If your config needs it, set: DINO_CKPT=/path/to/dinov2_vitb14_pretrain.pth"
fi

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
  LOG_FILE="${REPO_ROOT}/test_shot_${SHOT}.log"

  # Run test and capture logs (for parsing [SHOT_METRICS] line).
  CUDA_VISIBLE_DEVICES=0 python3 "${REPO_ROOT}/main.py" --device "0" \
    --cfg "${CFG_TEST}" --prior_layer_pointer 5 6 7 8 9 10 \
    --opts NUM_GPUS 1 DEVICE 0 RNG_SEED 54 \
    OUTPUT_DIR "${BASE_OUT}" \
    DATASET.shot "${SHOT}" \
    DATASET.custom_support_from_train True \
    TEST.load_checkpoint True TEST.load_model_path "${CKPT_PATH}" \
    TEST.VISUALIZE.save_figure True TEST.VISUALIZE.sample_prob 1.0 \
    TRAIN.enable False TEST.enable True \
    ${DINO_CKPT:+TRAIN.backbone_checkpoint "${DINO_CKPT}"} \
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
      echo "${SHOT_VAL},${SINSHOT_VAL},${PACC_VAL},${DICE_VAL},${MIOU_VAL},${OUT_DIR_FULL},${CKPT_PATH}" >> "${SHOT_METRICS_CSV}"
      echo "[INFO] Appended metrics to ${SHOT_METRICS_CSV}"
    else
      echo "[WARN] No [SHOT_METRICS] line found in ${LOG_FILE}; CSV not updated for shot=${SHOT}"
    fi
  fi
done


