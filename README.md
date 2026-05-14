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

Recommended first RunPod GPU:

```text
A100, A40, A5000, or A6000
```

Avoid making RTX 4090 your first attempt for this old stack. The training environment used:

```text
Python 3.9
PyTorch 1.9.0+cu111
TorchVision 0.10.0+cu111
MMCV-full 1.4.2
MMSegmentation 0.20.2
MMDetection 2.22.0
ViT-Adapter commit fd64a5190f9f4530e8e5961cc87b2a3ad591190b
```

### Option A: Build Docker Image

Build this image somewhere with Docker, preferably not your low-memory laptop:

```bash
cd /workspace/Inference
docker build --build-arg INSTALL_GPU_DEPS=true -t pathai-inference:512 .
```

If pushing to a registry:

```bash
docker tag pathai-inference:512 ghcr.io/YOUR_USER/pathai-inference:512
docker push ghcr.io/YOUR_USER/pathai-inference:512
```

The Dockerfile uses a CUDA devel image because ViT-Adapter CUDA ops need to compile.

### Option B: Install Inside A RunPod PyTorch Pod

Start a RunPod pod from a PyTorch CUDA 11.1/devel-style image, then run:

```bash
cd /workspace
git clone https://github.com/YOUR_USER/Inference.git Inference
cd /workspace/Inference
python -m pip install --upgrade pip wheel "setuptools<81"
python -m pip install -r requirements.txt
python -m pip install -r requirements-gpu.txt
```

If this folder is not in a GitHub repo yet, upload/copy it to `/workspace/Inference`.

### Prepare ViT-Adapter And Config

From inside RunPod:

```bash
cd /workspace/Inference
bash scripts/runpod_setup_512.sh
```

That script:

- clones `PathTests`
- checks out `leonardo-cleanup`
- clones `ViT-Adapter`
- checks out commit `fd64a5190f9f4530e8e5961cc87b2a3ad591190b`
- copies the recovered 512 config into `ViT-Adapter/segmentation/configs/pathtests`
- compiles `ViT-Adapter/segmentation/ops`
- writes `.env.runpod.512`

### Upload Checkpoint

Put the original checkpoint at:

```text
/workspace/models/best_mean_micro_dice_1_4_iter_8000.pth
```

Then strip optimizer state:

```bash
python /workspace/Inference/scripts/strip_checkpoint.py \
  /workspace/models/best_mean_micro_dice_1_4_iter_8000.pth \
  /workspace/models/best_mean_micro_dice_1_4_iter_8000.state_dict.pth
```

### Start Service

```bash
bash /workspace/Inference/scripts/runpod_start_512.sh
```

The service listens on:

```text
0.0.0.0:8000
```

### Smoke Test

Health only:

```bash
bash /workspace/Inference/scripts/runpod_smoke_512.sh
```

Health plus one PNG ROI:

```bash
bash /workspace/Inference/scripts/runpod_smoke_512.sh /workspace/test_roi.png
```

Or directly:

```bash
curl -X POST http://127.0.0.1:8000/segment \
  -F "image=@/workspace/test_roi.png" \
  -F "mpp=1.0" \
  -F "image_id=smoke" | python -m json.tool
```

## Hooking Into PathAI

Once RunPod exposes the service URL, set the .NET backend:

```json
"Analysis": {
  "UseFakeData": false,
  "PythonApiBaseUrl": "https://YOUR-RUNPOD-URL"
}
```

The current PathAI backend still needs to be updated from the old `/segment/tumor`, `/segment/glands`, `/segment/nuclei` calls to one model-native call:

```text
POST /segment
```

Then the frontend should display layers by returned `class`/`class_id`:

```text
tumor_gland
tumor_stroma
normal_gland
normal_stroma
```

Ignore or hide `bg` in the viewer.
