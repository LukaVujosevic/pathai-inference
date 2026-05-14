from typing import List, Tuple

import cv2
import numpy as np

from app.schemas import SegmentAnnotation, SegmentFeature


def polygonize_semantic_mask(
    mask: np.ndarray,
    selected_class_ids: List[int],
    class_names: List[str],
    label: str,
    mpp: float,
    model_version: str,
    min_area_px: float,
    target_size: Tuple[int, int],
) -> Tuple[List[SegmentAnnotation], List[SegmentFeature]]:
    if not selected_class_ids:
        return [], []

    width, height = target_size
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)

    annotations: List[SegmentAnnotation] = []
    features: List[SegmentFeature] = []
    object_id = 1

    for class_id in selected_class_ids:
        class_mask = (mask == class_id).astype(np.uint8)
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area_px = float(cv2.contourArea(contour))
            if area_px < min_area_px:
                continue

            epsilon = max(1.0, 0.0025 * float(cv2.arcLength(contour, True)))
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape(-1, 2)
            if len(points) < 3:
                continue

            perimeter_px = float(cv2.arcLength(approx, True))
            area_um2 = area_px * mpp * mpp
            perimeter_um = perimeter_px * mpp
            circularity = 0.0 if perimeter_um <= 0 else (4.0 * np.pi * area_um2) / (perimeter_um * perimeter_um)
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            annotation_prefix = class_name if label == "all" else label
            annotation_id = f"{annotation_prefix}-{object_id}"

            annotations.append(
                SegmentAnnotation(
                    id=annotation_id,
                    object_id=object_id,
                    polygon=[[float(x), float(y)] for x, y in points],
                    class_=class_name,
                    class_id=class_id,
                    source="mask2former",
                    model_version=model_version,
                )
            )
            features.append(
                SegmentFeature(
                    id=annotation_id,
                    object_id=object_id,
                    area_um2=area_um2,
                    perimeter_um=perimeter_um,
                    circularity=float(circularity),
                    class_=class_name,
                    class_id=class_id,
                )
            )
            object_id += 1

    return annotations, features
