from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class SegmentAnnotation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    object_id: int
    polygon: List[List[float]]
    class_: str = Field(serialization_alias="class", alias="class")
    class_id: int
    source: str
    model_version: str


class SegmentFeature(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    object_id: int
    area_um2: float
    perimeter_um: float
    circularity: float
    class_: str = Field(serialization_alias="class", alias="class")
    class_id: int


class SegmentData(BaseModel):
    image_id: str
    annotations: List[SegmentAnnotation]
    features: List[SegmentFeature]
    prompts: Optional[list] = None


class SegmentImageDimensions(BaseModel):
    width: int
    height: int


class SegmentMetadata(BaseModel):
    count: int
    mpp: float
    deepzoom_level: Optional[int] = None
    model_version: str
    image_dimensions: SegmentImageDimensions


class SegmentResponse(BaseModel):
    success: bool
    data: SegmentData
    metadata: SegmentMetadata
