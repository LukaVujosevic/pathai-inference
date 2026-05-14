import hashlib
import os
import sys
import threading
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

from app.config import get_settings


class BaseSegmenter:
    loaded = False

    def load(self) -> None:
        self.loaded = True

    def predict(self, image: Image.Image, label: str, image_id: str) -> np.ndarray:
        self.load()
        raise NotImplementedError


class MockSegmenter(BaseSegmenter):
    def predict(self, image: Image.Image, label: str, image_id: str) -> np.ndarray:
        self.load()

        width, height = image.size
        mask = np.zeros((height, width), dtype=np.uint8)

        seed = int(hashlib.sha256(f"{image_id}:{label}:{width}x{height}".encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)

        normal_stroma_center = (int(width * 0.48), int(height * 0.55))
        normal_stroma_axes = (max(20, int(width * 0.34)), max(20, int(height * 0.28)))
        cv2.ellipse(mask, normal_stroma_center, normal_stroma_axes, -12, 0, 360, 4, -1)

        tumor_stroma_center = (int(width * 0.58), int(height * 0.46))
        tumor_stroma_axes = (max(16, int(width * 0.22)), max(16, int(height * 0.2)))
        cv2.ellipse(mask, tumor_stroma_center, tumor_stroma_axes, 18, 0, 360, 2, -1)

        for index in range(3):
            cx = int(width * (0.28 + 0.14 * index) + rng.integers(-18, 19))
            cy = int(height * (0.34 + 0.08 * (index % 2)) + rng.integers(-18, 19))
            axes = (max(8, int(width * 0.055)), max(8, int(height * 0.04)))
            cv2.ellipse(mask, (cx, cy), axes, int(rng.integers(0, 180)), 0, 360, 3, -1)

        for index in range(3):
            cx = int(width * (0.46 + 0.1 * index) + rng.integers(-14, 15))
            cy = int(height * (0.5 + 0.06 * (index % 2)) + rng.integers(-14, 15))
            axes = (max(8, int(width * 0.05)), max(8, int(height * 0.038)))
            cv2.ellipse(mask, (cx, cy), axes, int(rng.integers(0, 180)), 0, 360, 1, -1)

        return mask


class MmsegSegmenter(BaseSegmenter):
    def __init__(self) -> None:
        self._model: Optional[Any] = None
        self._lock = threading.Lock()

    def load(self) -> None:
        if self.loaded:
            return

        with self._lock:
            if self.loaded:
                return

            settings = get_settings()
            if not settings.model_checkpoint:
                raise RuntimeError("MODEL_CHECKPOINT is required when MODEL_BACKEND=mmseg.")
            if not settings.model_config_path:
                raise RuntimeError("MODEL_CONFIG is required when MODEL_BACKEND=mmseg.")
            if not os.path.exists(settings.model_checkpoint):
                raise RuntimeError(f"Checkpoint not found: {settings.model_checkpoint}")
            if not os.path.exists(settings.model_config_path):
                raise RuntimeError(f"MMSeg config not found: {settings.model_config_path}")

            if settings.model_code_dir:
                sys.path.insert(0, settings.model_code_dir)

            try:
                import mmcv_custom  # noqa: F401
                import mmseg_custom  # noqa: F401
            except Exception as exc:
                raise RuntimeError(
                    "Could not import ViT-Adapter custom modules. "
                    "Set MODEL_CODE_DIR to /workspace/ViT-Adapter/segmentation."
                ) from exc

            try:
                from mmcv import Config
            except Exception as exc:
                raise RuntimeError("Could not import mmcv.Config.") from exc

            try:
                from mmseg.apis import init_segmentor
            except Exception as exc:
                raise RuntimeError(
                    "Could not import MMSegmentation. Install requirements-gpu.txt in the Docker image."
                ) from exc

            cfg = Config.fromfile(settings.model_config_path)
            if settings.disable_pretrained:
                disable_pretrained_references(cfg)

            self._model = init_segmentor(
                cfg,
                settings.model_checkpoint,
                device=settings.device,
            )
            self.loaded = True

    def predict(self, image: Image.Image, label: str, image_id: str) -> np.ndarray:
        self.load()

        try:
            from mmseg.apis import inference_segmentor
        except Exception as exc:
            raise RuntimeError("Could not import mmseg.apis.inference_segmentor.") from exc

        rgb = np.asarray(image.convert("RGB"))
        bgr = rgb[:, :, ::-1].copy()
        result = inference_segmentor(self._model, bgr)
        mask = extract_semantic_mask(result)

        if mask.shape[:2] != (image.height, image.width):
            mask = cv2.resize(mask.astype(np.uint8), image.size, interpolation=cv2.INTER_NEAREST)

        return mask.astype(np.uint8)


def extract_semantic_mask(result: Any) -> np.ndarray:
    if isinstance(result, np.ndarray):
        return squeeze_mask(result)

    if isinstance(result, dict):
        for key in ("sem_pred", "semantic", "seg_pred", "pred_sem_seg", "mask"):
            value = result.get(key)
            if value is not None:
                return extract_semantic_mask(value)

    if isinstance(result, (list, tuple)):
        for value in result:
            if isinstance(value, np.ndarray):
                return squeeze_mask(value)
            if isinstance(value, (list, tuple, dict)):
                try:
                    return extract_semantic_mask(value)
                except RuntimeError:
                    continue

    raise RuntimeError(f"Unsupported MMSeg output type: {type(result)!r}")


def squeeze_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        elif mask.shape[-1] == 1:
            mask = mask[:, :, 0]
    if mask.ndim != 2:
        raise RuntimeError(f"Expected a 2D semantic mask, got shape {mask.shape}.")
    return mask


def disable_pretrained_references(cfg: Any) -> None:
    if hasattr(cfg, "pretrained"):
        cfg.pretrained = None
    if hasattr(cfg, "model"):
        model = cfg.model
        if isinstance(model, dict):
            model["pretrained"] = None
            backbone = model.get("backbone")
            if isinstance(backbone, dict):
                backbone["pretrained"] = None
        else:
            if hasattr(model, "pretrained"):
                model.pretrained = None
            if hasattr(model, "backbone") and hasattr(model.backbone, "pretrained"):
                model.backbone.pretrained = None


_segmenter: Optional[BaseSegmenter] = None
_segmenter_lock = threading.Lock()


def get_segmenter() -> BaseSegmenter:
    global _segmenter
    if _segmenter is not None:
        return _segmenter

    with _segmenter_lock:
        if _segmenter is not None:
            return _segmenter

        backend = get_settings().model_backend.lower()
        if backend == "mock":
            _segmenter = MockSegmenter()
        elif backend == "mmseg":
            _segmenter = MmsegSegmenter()
        else:
            raise RuntimeError(f"Unknown MODEL_BACKEND '{backend}'. Use 'mock' or 'mmseg'.")

        return _segmenter
