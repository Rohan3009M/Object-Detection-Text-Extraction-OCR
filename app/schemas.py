from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class Detection(BaseModel):
    cls: str
    conf: float
    bbox: List[float]

class AnalyzeResponse(BaseModel):
    present: bool
    confidence: float                 # detection max_conf (YOLO)
    detections: List[Detection]
    metrics: Dict[str, Any]
    latency_ms: int
    captured_at: str                  # ISO timestamp
    captured_at_readable: str
    plate_text: Optional[str] = None
    ocr_confidence: Optional[float] = None


class OCRPlateResponse(BaseModel):
    plate_text: Optional[str] = None
    ocr_confidence: Optional[float] = None
    latency_ms: int
