import numpy as np
import cv2
from fastapi import UploadFile


async def read_image_bgr(file: UploadFile) -> np.ndarray:
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image bytes")
    return img


def crop_bbox_bgr(img_bgr: np.ndarray, bbox: list[float], pad: float = 0.12) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = map(float, bbox)

    pad_x = int((x2 - x1) * pad)
    pad_y = int((y2 - y1) * pad)

    x1 = max(0, int(x1) - pad_x)
    y1 = max(0, int(y1) - pad_y)
    x2 = min(w, int(x2) + pad_x)
    y2 = min(h, int(y2) + pad_y)

    if x2 <= x1 or y2 <= y1:
        return img_bgr
    return img_bgr[y1:y2, x1:x2]


def encode_image_jpg(img_bgr: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("Failed to encode image")
    return bytes(buf)
