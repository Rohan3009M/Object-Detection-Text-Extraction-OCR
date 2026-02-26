import time
from datetime import datetime, timezone
from typing import List
from zoneinfo import ZoneInfo

import httpx
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from .config import settings
from .detector import ProductDetector
from .schemas import AnalyzeResponse, Detection
from .utils import crop_bbox_bgr, encode_image_jpg, read_image_bgr


app = FastAPI(title="PoC Number Plate Detector", version="1.0")

detector = ProductDetector(
    model_path=settings.model_path_str,
    conf_thr=settings.conf_thr,
    iou_thr=settings.iou_thr,
    device=None,  # Prefer GPU automatically, fall back to CPU.
)

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/jpg"}
IST_ZONE = ZoneInfo("Asia/Kolkata")


def _now_timestamps() -> tuple[str, str]:
    captured_at_dt = datetime.now(timezone.utc)
    captured_at = captured_at_dt.isoformat()
    captured_at_readable = captured_at_dt.astimezone(IST_ZONE).strftime("%d-%b-%Y %I:%M:%S %p IST")
    return captured_at, captured_at_readable


def _error_response(error: str) -> AnalyzeResponse:
    captured_at, captured_at_readable = _now_timestamps()
    return AnalyzeResponse(
        present=False,
        confidence=0.0,
        detections=[],
        metrics={"error": error},
        latency_ms=0,
        captured_at=captured_at,
        captured_at_readable=captured_at_readable,
        plate_text=None,
        ocr_confidence=None,
    )


async def _read_plate_via_ocr_service(crop_bgr) -> tuple[str | None, float | None, dict]:
    try:
        crop_bytes = encode_image_jpg(crop_bgr, quality=90)
    except Exception as exc:
        return None, None, {"ocr_error": f"encode_failed:{type(exc).__name__}"}

    files = {"file": ("plate_crop.jpg", crop_bytes, "image/jpeg")}
    try:
        async with httpx.AsyncClient(timeout=settings.ocr_timeout_s) as client:
            resp = await client.post(settings.ocr_service_url, files=files)
        resp.raise_for_status()
        data = resp.json()
        return data.get("plate_text"), data.get("ocr_confidence"), {
            "ocr_service_latency_ms": data.get("latency_ms"),
            "ocr_service_url": settings.ocr_service_url,
        }
    except Exception as exc:
        return None, None, {"ocr_error": f"{type(exc).__name__}: {exc}"}


async def _analyze_upload(file: UploadFile, conf_thr: float | None, iou_thr: float | None) -> AnalyzeResponse:
    t0 = time.time()

    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported file type")

    try:
        img = await read_image_bgr(file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image") from exc

    # 1) YOLO number plate detection (GPU preferred via ProductDetector)
    dets = detector.predict(img, conf_thr=conf_thr, iou_thr=iou_thr)

    used_conf = detector.conf_thr if conf_thr is None else float(conf_thr)
    used_iou = detector.iou_thr if iou_thr is None else float(iou_thr)

    max_conf = max((d["conf"] for d in dets), default=0.0)
    present = max_conf >= used_conf

    # 2) OCR service on the best YOLO crop (PaddleOCR runs in a separate process/service)
    plate_text = None
    ocr_conf = None
    ocr_metrics: dict = {}
    if dets:
        best_det = max(dets, key=lambda d: d["conf"])
        crop = crop_bbox_bgr(img, best_det["bbox"])
        plate_text, ocr_conf, ocr_metrics = await _read_plate_via_ocr_service(crop)

    metrics = {
        "num_detections": len(dets),
        "thresholds_used": {"conf_thr": used_conf, "iou_thr": used_iou},
        **ocr_metrics,
    }
    latency_ms = int((time.time() - t0) * 1000)
    captured_at, captured_at_readable = _now_timestamps()

    return AnalyzeResponse(
        present=present,
        confidence=round(max_conf, 3),
        detections=[Detection(**d) for d in dets],
        metrics=metrics,
        latency_ms=latency_ms,
        captured_at=captured_at,
        captured_at_readable=captured_at_readable,
        plate_text=plate_text,
        ocr_confidence=None if ocr_conf is None else float(ocr_conf),
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    conf_thr: float = Query(default=None),
    iou_thr: float = Query(default=None),
):
    return await _analyze_upload(file, conf_thr=conf_thr, iou_thr=iou_thr)


@app.post("/v1/analyze/batch", response_model=List[AnalyzeResponse])
async def analyze_batch(
    files: List[UploadFile] = File(...),
    conf_thr: float = Query(default=None),
    iou_thr: float = Query(default=None),
):
    out: List[AnalyzeResponse] = []
    for file in files:
        if file.content_type not in ALLOWED_IMAGE_TYPES:
            out.append(_error_response("unsupported_file"))
            continue

        try:
            out.append(await _analyze_upload(file, conf_thr=conf_thr, iou_thr=iou_thr))
        except HTTPException as exc:
            if exc.status_code == 400:
                out.append(_error_response("invalid_image"))
            elif exc.status_code == 415:
                out.append(_error_response("unsupported_file"))
            else:
                raise

    return out
