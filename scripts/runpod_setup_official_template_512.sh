#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
INFERENCE_DIR="${INFERENCE_DIR:-${WORKSPACE}/Inference}"
VIT_ADAPTER_DIR="${VIT_ADAPTER_DIR:-${WORKSPACE}/ViT-Adapter}"
VIT_ADAPTER_COMMIT="${VIT_ADAPTER_COMMIT:-fd64a5190f9f4530e8e5961cc87b2a3ad591190b}"
CONDA_DIR="${CONDA_DIR:-${WORKSPACE}/miniconda}"
CONDA_ENV="${CONDA_ENV:-pathai}"
GDRIVE_CHECKPOINT_URL="${GDRIVE_CHECKPOINT_URL:-}"

if [[ ! -d "${INFERENCE_DIR}" ]]; then
  echo "Missing ${INFERENCE_DIR}. Clone pathai-inference there first." >&2
  exit 1
fi

mkdir -p "${WORKSPACE}/models"

if [[ ! -x "${CONDA_DIR}/bin/conda" ]]; then
  echo "[setup] Installing Miniconda to ${CONDA_DIR}"
  wget -O "${WORKSPACE}/miniconda.sh" \
    https://repo.anaconda.com/miniconda/Miniconda3-py39_24.1.2-0-Linux-x86_64.sh
  bash "${WORKSPACE}/miniconda.sh" -b -p "${CONDA_DIR}"
fi

source "${CONDA_DIR}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
  echo "[setup] Creating conda env ${CONDA_ENV}"
  conda create -n "${CONDA_ENV}" python=3.9 -y
fi

conda activate "${CONDA_ENV}"

echo "[setup] Installing Python dependencies"
python -m pip install --upgrade "pip<25" wheel "setuptools<81"
python -m pip install \
  torch==1.9.0+cu111 \
  torchvision==0.10.0+cu111 \
  torchaudio==0.9.0 \
  -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install -r "${INFERENCE_DIR}/requirements.txt"
python -m pip install -r "${INFERENCE_DIR}/requirements-gpu.txt"
python -m pip install "numpy==1.23.5" "scipy==1.10.1"
python -m pip install gdown

echo "[setup] Preparing ViT-Adapter"
if [[ ! -d "${VIT_ADAPTER_DIR}/.git" ]]; then
  git clone https://github.com/czczup/ViT-Adapter.git "${VIT_ADAPTER_DIR}"
fi
git -C "${VIT_ADAPTER_DIR}" fetch --all --prune
git -C "${VIT_ADAPTER_DIR}" checkout "${VIT_ADAPTER_COMMIT}"

mkdir -p "${VIT_ADAPTER_DIR}/segmentation/configs/pathtests"
cp "${INFERENCE_DIR}/configs/mask2former_hibou_adapter_large_512_20k_pathtests_1mpp_v1.py" \
  "${VIT_ADAPTER_DIR}/segmentation/configs/pathtests/"

python "${INFERENCE_DIR}/scripts/patch_vit_adapter_hibou_swiglu.py"
python -m py_compile "${VIT_ADAPTER_DIR}/segmentation/mmseg_custom/models/backbones/base/vit.py"

if [[ ! -e "${VIT_ADAPTER_DIR}/segmentation/ops" ]]; then
  ln -s ../detection/ops "${VIT_ADAPTER_DIR}/segmentation/ops"
fi

echo "[setup] Building ViT-Adapter CUDA ops"
(
  cd "${VIT_ADAPTER_DIR}/segmentation/ops"
  bash make.sh
)

if [[ -n "${GDRIVE_CHECKPOINT_URL}" ]]; then
  echo "[setup] Downloading checkpoint from Google Drive"
  gdown --fuzzy "${GDRIVE_CHECKPOINT_URL}" \
    -O "${WORKSPACE}/models/best_mean_micro_dice_1_4_iter_8000.full.pth"

  python "${INFERENCE_DIR}/scripts/strip_checkpoint.py" \
    "${WORKSPACE}/models/best_mean_micro_dice_1_4_iter_8000.full.pth" \
    "${WORKSPACE}/models/best_mean_micro_dice_1_4_iter_8000.state_dict.pth"
else
  echo "[setup] Skipping checkpoint download. Set GDRIVE_CHECKPOINT_URL to download automatically."
fi

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

echo
echo "RunPod setup complete."
echo "Start the service with:"
echo "  bash ${INFERENCE_DIR}/scripts/runpod_start_512.sh"
