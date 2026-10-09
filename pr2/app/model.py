"""
Модуль інференсу об'єктного детектора (Ultralytics YOLO).
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import List

from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

MODEL_WEIGHTS = "yolov8n.pt"
DEFAULT_CONFIDENCE_THRESHOLD = 0.25


@dataclass
class Detection:
    class_name: str
    confidence: float
    box: dict


@dataclass
class DetectionResult:
    detections: List[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0
    image_width: int = 0
    image_height: int = 0


class InvalidImageError(ValueError):
    """Файл, що надійшов на вхід, не вдалося прочитати як коректне зображення."""


class Detector:
    def __init__(self, weights: str = MODEL_WEIGHTS) -> None:
        self._model = YOLO(weights)
        self._model.predict(Image.new("RGB", (64, 64)), verbose=False)

    def detect(
        self,
        image_bytes: bytes,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> DetectionResult:
        if not image_bytes:
            raise InvalidImageError("Порожній файл: немає даних для обробки")

        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
            image = image.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError(
                "Файл не є коректним зображенням (перевірте формат і цілісність)"
            ) from exc

        started = time.perf_counter()
        results = self._model.predict(image, conf=confidence_threshold, verbose=False)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        result = results[0]
        names = result.names
        detections: List[Detection] = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            detections.append(
                Detection(
                    class_name=names[cls_id],
                    confidence=round(confidence, 4),
                    box={
                        "x1": round(x1, 1),
                        "y1": round(y1, 1),
                        "x2": round(x2, 1),
                        "y2": round(y2, 1),
                    },
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)

        return DetectionResult(
            detections=detections,
            inference_time_ms=round(elapsed_ms, 2),
            image_width=image.width,
            image_height=image.height,
        )