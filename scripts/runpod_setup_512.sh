#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
INFERENCE_DIR="${INFERENCE_DIR:-${WORKSPACE}/Inference}"
PATHTESTS_DIR="${PATHTESTS_DIR:-${WORKSPACE}/PathTests}"
VIT_ADAPTER_DIR="${VIT_ADAPTER_DIR:-${WORKSPACE}/ViT-Adapter}"
VIT_ADAPTER_COMMIT="${VIT_ADAPTER_COMMIT:-fd64a5190f9f4530e8e5961cc87b2a3ad591190b}"

mkdir -p "${WORKSPACE}/models" "${WORKSPACE}/configs"

if [[ ! -d "${PATHTESTS_DIR}/.git" ]]; then
  git clone https://github.com/LukaVujosevic/PathTests.git "${PATHTESTS_DIR}"
fi
git -C "${PATHTESTS_DIR}" fetch --all --prune
git -C "${PATHTESTS_DIR}" checkout leonardo-cleanup
git -C "${PATHTESTS_DIR}" pull --ff-only || true

if [[ ! -d "${VIT_ADAPTER_DIR}/.git" ]]; then
  git clone https://github.com/czczup/ViT-Adapter.git "${VIT_ADAPTER_DIR}"
fi
git -C "${VIT_ADAPTER_DIR}" fetch --all --prune
git -C "${VIT_ADAPTER_DIR}" checkout "${VIT_ADAPTER_COMMIT}"

mkdir -p "${VIT_ADAPTER_DIR}/segmentation/configs/pathtests"
cp "${INFERENCE_DIR}/configs/mask2former_hibou_adapter_large_512_20k_pathtests_1mpp_v1.py" \
  "${VIT_ADAPTER_DIR}/segmentation/configs/pathtests/"

if [[ ! -e "${VIT_ADAPTER_DIR}/segmentation/ops" ]]; then
  ln -s ../detection/ops "${VIT_ADAPTER_DIR}/segmentation/ops"
fi

cd "${VIT_ADAPTER_DIR}/segmentation/ops"
bash make.sh

cat > "${INFERENCE_DIR}/.env.runpod.512" <<EOF
MODEL_BACKEND=mmseg
MODEL_CHECKPOINT=${WORKSPACE}/models/best_mean_micro_dice_1_4_iter_8000.state_dict.pth
MODEL_CONFIG=${VIT_ADAPTER_DIR}/segmentation/configs/pathtests/mask2former_hibou_adapter_large_512_20k_pathtests_1mpp_v1.py
MODEL_CODE_DIR=${VIT_ADAPTER_DIR}/segmentation
DEVICE=cuda:0
MODEL_VERSION=mask2former-hibou-512-iter8000
MIN_CONTOUR_AREA_PX=32
DISABLE_PRETRAINED=true
EOF

echo "RunPod setup complete."
echo
echo "Put your checkpoint here:"
echo "  ${WORKSPACE}/models/best_mean_micro_dice_1_4_iter_8000.pth"
echo
echo "Then strip it:"
echo "  python ${INFERENCE_DIR}/scripts/strip_checkpoint.py ${WORKSPACE}/models/best_mean_micro_dice_1_4_iter_8000.pth ${WORKSPACE}/models/best_mean_micro_dice_1_4_iter_8000.state_dict.pth"
echo
echo "Start service:"
echo "  bash ${INFERENCE_DIR}/scripts/runpod_start_512.sh"
