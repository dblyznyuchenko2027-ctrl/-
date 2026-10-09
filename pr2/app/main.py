
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.model import Detector, InvalidImageError
from app.schemas import BoundingBox, DetectionOut, DetectResponse

DEFAULT_THRESHOLD = 0.25
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  

detector: Optional[Detector] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    detector = Detector()  
    yield
    detector = None


app = FastAPI(title="Object Detection — local YOLO inference", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    with open("app/static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/detect", response_model=DetectResponse)
async def detect(
    file: UploadFile = File(..., description="Файл зображення (jpg/png/...)"),
    threshold: float = Query(
        DEFAULT_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Поріг confidence у діапазоні 0..1",
    ),
) -> DetectResponse:
    if detector is None:
        raise HTTPException(status_code=503, detail="Модель ще не готова, спробуйте пізніше")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Очікується файл зображення (image/*), отримано: {file.content_type}",
        )

    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Файл завеликий (ліміт 15 МБ)")

    try:
        result = detector.detect(raw_bytes, confidence_threshold=threshold)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return DetectResponse(
        objects=[
            DetectionOut(
                **{"class": d.class_name},
                confidence=d.confidence,
                box=BoundingBox(**d.box),
            )
            for d in result.detections
        ],
        count=len(result.detections),
        inference_time_ms=result.inference_time_ms,
        image_width=result.image_width,
        image_height=result.image_height,
        threshold_used=threshold,
    )
