
from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    class_name: str = Field(..., alias="class")
    confidence: float
    box: BoundingBox


class DetectResponse(BaseModel):
    objects: List[DetectionOut]
    count: int
    inference_time_ms: float
    image_width: int
    image_height: int
    threshold_used: float
