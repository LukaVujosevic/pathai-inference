import json
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.image_io import read_rgb_image
from app.model_service import get_segmenter
from app.polygonize import polygonize_semantic_mask
from app.schemas import (
    SegmentData,
    SegmentImageDimensions,
    SegmentMetadata,
    SegmentResponse,
)

settings = get_settings()

app = FastAPI(title="PathAI Inference Service", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict:
    return {
        "service": "PathAI Inference Service",
        "backend": settings.model_backend,
        "classes": settings.class_names,
        "endpoints": [
            "/segment",
            "/segment/all",
            "/segment/bg",
            "/segment/tumor_gland",
            "/segment/tumor_stroma",
            "/segment/normal_gland",
            "/segment/normal_stroma",
        ],
    }


@app.get("/health")
def health() -> dict:
    segmenter = get_segmenter()
    return {
        "ok": True,
        "backend": settings.model_backend,
        "loaded": segmenter.loaded,
        "model_version": settings.model_version,
        "checkpoint": settings.model_checkpoint,
        "config": settings.model_config_path,
    }


@app.get("/classes")
def classes() -> dict:
    return {
        "classes": [
            {"id": class_id, "name": class_name}
            for class_id, class_name in enumerate(settings.class_names)
        ],
        "segment_labels": sorted(settings.segment_class_ids.keys()),
    }


@app.post("/warmup")
def warmup() -> dict:
    segmenter = get_segmenter()
    segmenter.load()
    return {"ok": True, "loaded": segmenter.loaded, "backend": settings.model_backend}


@app.post("/segment", response_model=SegmentResponse)
async def segment_all(
    image: UploadFile = File(...),
    mpp: float = Form(...),
    image_id: str = Form("roi"),
    bbox_prompts: Optional[str] = Form(None),
) -> SegmentResponse:
    return await run_segmentation(
        label="all",
        image=image,
        mpp=mpp,
        image_id=image_id,
        bbox_prompts=bbox_prompts,
    )


@app.post("/segment/{label}", response_model=SegmentResponse)
async def segment(
    label: str,
    image: UploadFile = File(...),
    mpp: float = Form(...),
    image_id: str = Form("roi"),
    bbox_prompts: Optional[str] = Form(None),
) -> SegmentResponse:
    return await run_segmentation(
        label=label.lower(),
        image=image,
        mpp=mpp,
        image_id=image_id,
        bbox_prompts=bbox_prompts,
    )


async def run_segmentation(
    label: str,
    image: UploadFile,
    mpp: float,
    image_id: str,
    bbox_prompts: Optional[str],
) -> SegmentResponse:
    if label not in settings.segment_class_ids:
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Unknown segmentation label '{label}'.",
                "allowed": sorted(settings.segment_class_ids.keys()),
            },
        )

    prompts = parse_prompts(bbox_prompts)
    rgb_image = await read_rgb_image(image)

    try:
        mask = get_segmenter().predict(rgb_image, label=label, image_id=image_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    annotations, features = polygonize_semantic_mask(
        mask=mask,
        selected_class_ids=settings.segment_class_ids[label],
        class_names=settings.class_names,
        label=label,
        mpp=mpp,
        model_version=settings.model_version,
        min_area_px=settings.min_contour_area_px,
        target_size=rgb_image.size,
    )

    return SegmentResponse(
        success=True,
        data=SegmentData(
            image_id=image_id,
            annotations=annotations,
            features=features,
            prompts=prompts,
        ),
        metadata=SegmentMetadata(
            count=len(annotations),
            mpp=mpp,
            model_version=settings.model_version,
            image_dimensions=SegmentImageDimensions(width=rgb_image.width, height=rgb_image.height),
        ),
    )


def parse_prompts(raw: Optional[str]) -> Optional[list]:
    if not raw:
        return None

    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None

    return value if isinstance(value, list) else None
