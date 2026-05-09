#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <input_video> [src_fps]"
  echo "Example: $0 /root/autodl-tmp/GVHMR/myvideo/own_kickball_eg.mp4 50"
  exit 1
fi

INPUT_VIDEO="$1"
SRC_FPS="${2:-}"

GVHMR_ROOT="/root/autodl-tmp/GVHMR"
GMR_ROOT="/root/autodl-tmp/GMR"
ASAP_ROOT="/root/autodl-tmp/ASAP"

if [[ ! -f "${INPUT_VIDEO}" ]]; then
  echo "Input video not found: ${INPUT_VIDEO}"
  exit 1
fi

VIDEO_STEM="$(basename "${INPUT_VIDEO}")"
VIDEO_STEM="${VIDEO_STEM%.*}"

GVHMR_OUT_DIR="${GVHMR_ROOT}/outputs/demo/${VIDEO_STEM}"
GVHMR_PRED="${GVHMR_OUT_DIR}/hmr4d_results.pt"
GMR_PKL="${GMR_ROOT}/unitree_g1_gmr/${VIDEO_STEM}.pkl"
ASAP_PKL="${ASAP_ROOT}/humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-${VIDEO_STEM}_gvhmr.pkl"
ASAP_MOTION_NAME="0-${VIDEO_STEM}_gvhmr"

if [[ -z "${SRC_FPS}" ]]; then
  SRC_FPS="$(conda run -n GVHMR python -c "import imageio.v3 as iio; print(float(iio.immeta('${INPUT_VIDEO}', plugin='pyav').get('fps')))")"
  SRC_FPS="$(echo "${SRC_FPS}" | tail -n 1 | tr -d '\r')"
fi

echo "====================================================="
echo "Input video: ${INPUT_VIDEO}"
echo "Video stem : ${VIDEO_STEM}"
echo "Source FPS : ${SRC_FPS}"
echo "GVHMR pred : ${GVHMR_PRED}"
echo "GMR pkl    : ${GMR_PKL}"
echo "ASAP pkl   : ${ASAP_PKL}"
echo "====================================================="

echo "[1/3] Running GVHMR..."
cd "${GVHMR_ROOT}"
conda run -n GVHMR python tools/demo/demo.py --video "${INPUT_VIDEO}" -s

echo "[2/3] Running GMR retarget..."
cd "${GMR_ROOT}"
PYTHONPATH="${GMR_ROOT}:${GMR_ROOT}/third_party" \
xvfb-run -a conda run -n gmr python scripts/gvhmr_to_robot.py \
  --gvhmr_pred_file "${GVHMR_PRED}" \
  --src_fps "${SRC_FPS}" \
  --robot unitree_g1 \
  --save_as_pkl True

echo "[3/3] Converting to ASAP motion pkl..."
cd "${ASAP_ROOT}"
/root/miniconda3/envs/rl/bin/python tools/convert_gmr_pkl_to_asap_motion.py \
  --input "${GMR_PKL}" \
  --output "${ASAP_PKL}" \
  --motion-name "${ASAP_MOTION_NAME}"

echo "Done."
echo "ASAP motion file: ${ASAP_PKL}"
