#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# Default: use repo checkpoint (override via CKPT_PATH=...)
DEFAULT_BEST_CKPT="${REPO_ROOT}/log_CUSTOM_DATASET_FINAL_CUSTOM_MASK_ND_SOFS_54/checkpoints/best_method_SOFS_CUSTOM_MASK_ND_split_0_.pth"
DEFAULT_LAST_CKPT="${REPO_ROOT}/checkpoints/last.pth"

# Prefer the best-method checkpoint from the dataset_final training log, if present.
if [[ -z "${CKPT_PATH:-}" ]]; then
  if [[ -f "${DEFAULT_BEST_CKPT}" ]]; then
    CKPT_PATH="${DEFAULT_BEST_CKPT}"
  else
    CKPT_PATH="${DEFAULT_LAST_CKPT}"
  fi
fi
if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "[ERROR] checkpoint not found: ${CKPT_PATH}"
  exit 1
fi

CFG_TEST="${CFG_TEST:-${REPO_ROOT}/method_config/CUSTOM/Test/SOFS_dataset_final_test.yaml}"

# DINO ckpt (optional override)
if [[ -z "${DINO_CKPT:-}" ]]; then
  if [[ -f "${REPO_ROOT}/dinov2_vitb14_pretrain.pth" ]]; then
    DINO_CKPT="${REPO_ROOT}/dinov2_vitb14_pretrain.pth"
  elif [[ -f "${REPO_ROOT}/checkpoints/dinov2_vitb14_pretrain.pth" ]]; then
    DINO_CKPT="${REPO_ROOT}/checkpoints/dinov2_vitb14_pretrain.pth"
  else
    DINO_CKPT=""
  fi
fi

# Shots to evaluate (space-separated). Example: SHOTS="1 2 4 8 16" bash test_dataset_final.sh
SHOTS="${SHOTS:-1 2 4 8 16}"
SHOTS_CSV="$(echo "${SHOTS}" | tr ' ' ',')"

# Output prefix (3 CSVs will be created: _overall.csv, _by_type.csv, _by_status.csv)
OUT_PREFIX="${OUT_PREFIX:-${REPO_ROOT}/dataset_final_metrics}"
# Dataloader workers (0 is safest if you suspect a hang)
NUM_WORKERS="${NUM_WORKERS:-0}"
# Save prediction visualizations
SAVE_FIGURES="${SAVE_FIGURES:-1}"
FIGURE_SAMPLE_PROB="${FIGURE_SAMPLE_PROB:-1.0}"

CUDA_VISIBLE_DEVICES=0 python "${REPO_ROOT}/tools/eval_grouped_custom_mask.py" \
  --cfg "${CFG_TEST}" \
  --ckpt "${CKPT_PATH}" \
  ${DINO_CKPT:+--dino_ckpt "${DINO_CKPT}"} \
  --device "0" \
  --seed 54 \
  --shots "${SHOTS_CSV}" \
  --out_prefix "${OUT_PREFIX}" \
  --num_workers "${NUM_WORKERS}" \
  $([[ "${SAVE_FIGURES}" == "1" ]] && echo --save_figures) \
  --figure_sample_prob "${FIGURE_SAMPLE_PROB}" \
  | tee "${REPO_ROOT}/test_dataset_final.log"


