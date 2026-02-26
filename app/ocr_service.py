import time
from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, UploadFile

from .schemas import OCRPlateResponse
from .utils import read_image_bgr


app = FastAPI(title="PoC Number Plate OCR Service", version="1.0")

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/jpg"}


@lru_cache(maxsize=1)
def _get_read_plate():
    from .ocr import read_plate

    return read_plate


@app.get("/health")
def health():
    try:
        _get_read_plate()
        return {"status": "ok"}
    except Exception as exc:
        return {"status": "degraded", "ocr_backend_error": f"{type(exc).__name__}: {exc}"}


@app.post("/v1/ocr/plate", response_model=OCRPlateResponse)
async def ocr_plate(file: UploadFile = File(...)):
    t0 = time.time()

    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported file type")

    try:
        img = await read_image_bgr(file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image") from exc

    try:
        read_plate = _get_read_plate()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"OCR backend unavailable: {type(exc).__name__}: {exc}") from exc

    plate_text, ocr_conf = read_plate(img)
    latency_ms = int((time.time() - t0) * 1000)

    return OCRPlateResponse(
        plate_text=plate_text,
        ocr_confidence=None if ocr_conf is None else float(ocr_conf),
        latency_ms=latency_ms,
    )
