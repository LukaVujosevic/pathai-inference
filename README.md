# PathAI Inference Service

FastAPI model server for the PathAI app. It accepts an ROI image and returns model-native tissue polygons/features.

Endpoints:

- `POST /segment`
- `POST /segment/all`
- `POST /segment/bg`
- `POST /segment/tumor_gland`
- `POST /segment/tumor_stroma`
- `POST /segment/normal_gland`
- `POST /segment/normal_stroma`
- `GET /classes`
- `GET /health`
- `POST /warmup`

The service starts in `MODEL_BACKEND=mock` by default, so the API shape can be tested without installing PyTorch/MMCV/MMSeg locally.

## Model

This folder is prepared for the 512 Hibou checkpoint:

```text
best_mean_micro_dice_1_4_iter_8000.pth
```

The exact MMSeg config was recovered from that checkpoint and stored here:

```text
configs/mask2former_hibou_adapter_large_512_20k_pathtests_1mpp_v1.py
```

Checkpoint classes:

```text
0 bg
1 tumor_gland
2 tumor_stroma
3 normal_gland
4 normal_stroma
```

Use `POST /segment` or `POST /segment/all` to return all foreground classes. `bg` is available but usually should not be drawn in the viewer because it may produce one huge background polygon.

## About `pretrained=None`

The recovered config references:

```python
pretrained = "pretrained/hibou_l_vit_large_patch16_for_vitadapter.npz"
```

That file was used to initialize the HIBOU backbone before training. For inference with a full trained `.pth`, loading the pretrained `.npz` first is usually unnecessary because the trained checkpoint immediately overwrites those weights.

This service defaults to:

```bash
DISABLE_PRETRAINED=true
```

That means the service changes `pretrained` to `None` before loading the checkpoint. Implication:

- Good: you do not need the 1.2GB HIBOU `.npz` file just to serve the trained checkpoint.
- Good: startup is simpler and avoids missing-file errors.
- Risk: if the checkpoint is incomplete and does not contain all backbone weights, missing layers would be randomly initialized and predictions would be bad.

For this checkpoint, the state dict appears to include full backbone and head weights, so `DISABLE_PRETRAINED=true` is the practical default. If model loading reports missing backbone keys, set `DISABLE_PRETRAINED=false` and provide the `.npz`.

## Local Mock Test

No heavy install is required. If you want to test only the API shape:

```powershell
python -m pip install -r requirements.txt
$env:MODEL_BACKEND="mock"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then try:

```powershell
Invoke-RestMethod http://localhost:8000/classes
```

## RunPod Setup

Use the official RunPod PyTorch/Jupyter template. It gives you a working browser terminal and file browser; the setup script installs the old Python stack in a separate Miniconda env.

Recommended GPU:

```text
A40, A5000, A6000, A100
```

The tested pod used an A40. Avoid making RTX 4090/5090 your first attempt for this old stack. The training environment used:

```text
Python 3.9
PyTorch 1.9.0+cu111
TorchVision 0.10.0+cu111
MMCV-full 1.4.2
MMSegmentation 0.20.2
MMDetection 2.22.0
ViT-Adapter commit fd64a5190f9f4530e8e5961cc87b2a3ad591190b
```

### Start A Pod

Create a pod from RunPod's official PyTorch/Jupyter template and expose:

```text
HTTP ports: 8888,8000
Volume mount: /workspace
Volume disk: 40 GB or more
```

Open Jupyter, then open a terminal.

### One-Command Setup

```bash
cd /workspace
git clone https://github.com/LukaVujosevic/pathai-inference.git Inference
cd /workspace/Inference

export GDRIVE_CHECKPOINT_URL="https://drive.google.com/file/d/1pMOWqBytVE7TYmLkANg3tqT8tunPyZwF/view?usp=drive_link"
bash scripts/runpod_setup_512.sh
```

The script installs Miniconda to `/workspace/miniconda`, creates `pathai`, installs PyTorch/MMCV/MMSeg, clones and patches ViT-Adapter for HIBOU SwiGLU, downloads the checkpoint, strips optimizer state, builds CUDA ops, and writes `.env.runpod.512`.

### Start Service

```bash
bash /workspace/Inference/scripts/runpod_start_512.sh
```

The service listens on `0.0.0.0:8000`. Keep this terminal running.

### Smoke Test

Open another Jupyter terminal:

```bash
source /workspace/miniconda/etc/profile.d/conda.sh
conda activate pathai

curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/warmup
```

Then upload a 1 mpp PNG/JPG ROI to `/workspace/test_roi.png` and run:

```bash
curl -X POST http://127.0.0.1:8000/segment \
  -F "image=@/workspace/test_roi.png" \
  -F "mpp=1.0" \
  -F "image_id=smoke" \
  -o /workspace/result.json

python -m json.tool /workspace/result.json | head -80
```

### Notes

- If the downloaded checkpoint is 3.9 GB, it is the full checkpoint. The setup script strips it to a 1.3 GB `state_dict` file.
- If you recreate the pod often, keep `/workspace` as persistent volume so `/workspace/miniconda`, `/workspace/models`, and `/workspace/ViT-Adapter` survive while the pod exists.
- Stop the pod when done testing.

## Hooking Into PathAI

Once RunPod exposes the service URL, set the .NET backend:

```json
"Analysis": {
  "UseFakeData": false,
  "PythonApiBaseUrl": "https://YOUR-RUNPOD-URL"
}
```

The PathAI backend calls one model-native endpoint:

```text
POST /segment
```

The frontend displays returned classes:

```text
tumor_gland
tumor_stroma
normal_gland
normal_stroma
```

Ignore or hide `bg` in the viewer.
