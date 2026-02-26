import logging
import os
import re
import sys
from typing import Any, Iterable

import cv2
import numpy as np

# Paddle runtime flags: reduce Windows CPU backend instability when CPU fallback is used.
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_use_onednn", "0")
os.environ.setdefault("FLAGS_use_new_executor", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "1")

logger = logging.getLogger(__name__)

IGNORE_WORDS = {"IND", "INDIA", "IN"}
PLATE_PATTERNS = [
    re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{3,4}$"),  # MH12AB1234 / KA03H4025 / PY01CV8008
]


def _clear_paddle_modules() -> None:
    """Remove partially imported Paddle modules before retrying init in-process."""
    for name in list(sys.modules.keys()):
        if name == "paddle" or name.startswith("paddle."):
            sys.modules.pop(name, None)


def normalize(text: str) -> str:
    text = (text or "").upper()
    return re.sub(r"[^A-Z0-9]", "", text)


def looks_like_plate(text: str) -> bool:
    return bool(text) and any(p.match(text) for p in PLATE_PATTERNS)


def repair_missing_series_letter(text: str) -> str:
    # Common OCR miss: KA03H4025 -> KA03HL4025 (drops one series letter).
    if re.match(r"^[A-Z]{2}\d{2}[A-Z]\d{4}$", text):
        return text[:5] + "L" + text[5:]
    return text


def _build_ocr_engine():
    """
    Prefer GPU for speed. Fall back to CPU if the local Paddle install/runtime
    does not support GPU.
    """
    prefer_gpu = os.getenv("USE_OCR_GPU", "1") != "0"
    use_angle_cls = os.getenv("OCR_USE_ANGLE_CLS", "0") == "1"  # default off for speed

    # Import PaddleOCR first in the OCR-only process. PaddleOCR/PaddleX may import torch
    # transitively, and importing paddle first can trigger DLL conflicts on Windows.
    from paddleocr import PaddleOCR

    common_kwargs_new = {
        "lang": "en",
        # PaddleOCR 3.x/PaddleX defaults can enable document preprocessing models
        # (doc orientation/unwarping), which are unnecessary for plate crops and add latency.
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
    }

    # Newer PaddleOCR (device=...)
    if prefer_gpu:
        try:
            engine = PaddleOCR(device="gpu", **common_kwargs_new)
            logger.info("PaddleOCR initialized on GPU")
            return engine
        except Exception:
            logger.warning("PaddleOCR GPU init failed; falling back to CPU", exc_info=True)
            # Clear broken partial import state before trying a CPU fallback in the same process.
            _clear_paddle_modules()

    try:
        engine = PaddleOCR(
            device="cpu",
            enable_mkldnn=False,
            enable_hpi=False,
            enable_cinn=False,
            cpu_threads=1,
            **common_kwargs_new,
        )
        logger.info("PaddleOCR initialized on CPU (new API)")
        return engine
    except TypeError:
        logger.warning("PaddleOCR new API not supported by installed version", exc_info=True)
    except Exception:
        _clear_paddle_modules()
        logger.warning("PaddleOCR CPU init (new API) failed", exc_info=True)

    # Last-resort retry in a fresh process is safer than in-process legacy fallbacks on Windows.
    raise RuntimeError("Failed to initialize PaddleOCR with current process/runtime configuration")


ocr_engine = _build_ocr_engine()


def crop_plate(img_bgr: np.ndarray, bbox: list[float], pad: float = 0.12) -> np.ndarray:
    """
    Crop YOLO bbox with a small padding. Returns the original image if bbox is invalid.
    """
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


def _variants_for_ocr(crop_bgr: np.ndarray) -> list[np.ndarray]:
    """
    Keep preprocessing small to reduce latency:
    1) CLAHE gray (general case)
    2) Adaptive threshold (faded/low-contrast plates)
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )

    variants = []
    for img in (enhanced, adaptive):
        # 2x is usually enough for OCR while keeping latency lower than 3x.
        upscaled = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        variants.append(upscaled)
    return variants


def _iter_ocr_items(results: Any) -> Iterable[tuple[Any, Any, Any]]:
    """
    Yield (polygon, text, confidence) across common PaddleOCR output shapes.
    """
    if results is None:
        return

    # Classic PaddleOCR: [[ [poly, (text, conf)], ... ]]
    if isinstance(results, list):
        if len(results) == 1 and isinstance(results[0], list):
            lines = results[0]
            for item in lines:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    poly, rec = item[0], item[1]
                    if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                        yield poly, rec[0], rec[1]
            return

        # Direct list of (poly, (text, conf))
        for item in results:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                poly, rec = item[0], item[1]
                if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                    yield poly, rec[0], rec[1]
                    continue

            # PaddleX-style dict/object entries
            if isinstance(item, dict):
                rec_texts = item.get("rec_texts") or []
                rec_scores = item.get("rec_scores") or []
                rec_polys = item.get("rec_polys") or item.get("dt_polys") or []
                for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
                    yield poly, text, score
                continue

            rec_texts = getattr(item, "rec_texts", None)
            rec_scores = getattr(item, "rec_scores", None)
            rec_polys = getattr(item, "rec_polys", None) or getattr(item, "dt_polys", None)
            if rec_texts and rec_scores and rec_polys:
                for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
                    yield poly, text, score
        return

    if isinstance(results, dict):
        rec_texts = results.get("rec_texts") or []
        rec_scores = results.get("rec_scores") or []
        rec_polys = results.get("rec_polys") or results.get("dt_polys") or []
        for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
            yield poly, text, score
        return

    rec_texts = getattr(results, "rec_texts", None)
    rec_scores = getattr(results, "rec_scores", None)
    rec_polys = getattr(results, "rec_polys", None) or getattr(results, "dt_polys", None)
    if rec_texts and rec_scores and rec_polys:
        for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
            yield poly, text, score


def _tokens_from_results(results: Any, img_w: int, img_h: int) -> list[dict]:
    tokens: list[dict] = []
    for poly, text, conf in _iter_ocr_items(results):
        text_n = normalize(text)
        if not text_n or text_n in IGNORE_WORDS:
            continue

        try:
            score = float(conf)
        except Exception:
            score = 0.0

        try:
            xs = [float(p[0]) for p in poly]
            ys = [float(p[1]) for p in poly]
        except Exception:
            continue

        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        area = width * height

        # Fast junk filters to reduce false positives and token joining errors.
        if x2 <= 0.18 * img_w and len(text_n) >= 4:
            continue  # left-side emblem/strip text
        if len(text_n) >= 11:
            continue
        if area < 0.002 * (img_w * img_h):
            continue
        if len(text_n) == 1 and score < 0.70:
            continue

        tokens.append(
            {
                "text": text_n,
                "conf": score,
                "xc": sum(xs) / len(xs),
                "yc": sum(ys) / len(ys),
            }
        )
    return tokens


def _join_tokens_reading_order(tokens: list[dict]) -> tuple[str | None, float]:
    if not tokens:
        return None, 0.0

    tokens_sorted = sorted(tokens, key=lambda t: t["yc"])
    ys = [t["yc"] for t in tokens_sorted]
    if len(ys) >= 2:
        diffs = sorted(abs(ys[i] - ys[i - 1]) for i in range(1, len(ys)))
        line_tol = max(10.0, diffs[len(diffs) // 2])
    else:
        line_tol = 15.0

    lines: list[dict] = []
    for token in tokens_sorted:
        placed = False
        for line in lines:
            if abs(token["yc"] - line["y"]) <= line_tol:
                line["items"].append(token)
                line["y"] = sum(t["yc"] for t in line["items"]) / len(line["items"])
                placed = True
                break
        if not placed:
            lines.append({"y": token["yc"], "items": [token]})

    lines.sort(key=lambda line: line["y"])
    for line in lines:
        line["items"].sort(key=lambda t: t["xc"])

    line_texts = ["".join(t["text"] for t in line["items"]) for line in lines]

    # Plate bottom line often carries more digits. Swap two-line order if OCR line grouping flips them.
    if len(line_texts) == 2:
        def digit_ratio(s: str) -> float:
            return 0.0 if not s else sum(ch.isdigit() for ch in s) / len(s)

        if digit_ratio(line_texts[0]) > digit_ratio(line_texts[1]) + 0.20:
            line_texts = [line_texts[1], line_texts[0]]

    joined = "".join(line_texts)
    score = sum(t["conf"] for t in tokens) / max(1, len(tokens))
    return joined, score


def read_plate(crop_bgr: np.ndarray) -> tuple[str | None, float]:
    """
    OCR a cropped number plate image and return (text, confidence).
    Strategy:
    - run OCR on a small number of preprocessed variants for speed
    - join tokens in reading order
    - prefer plate-like strings, then fall back to best token
    """
    best_any: tuple[str | None, float] = (None, 0.0)
    best_plate: tuple[str | None, float] = (None, 0.0)

    for variant in _variants_for_ocr(crop_bgr):
        img_rgb = cv2.cvtColor(variant, cv2.COLOR_GRAY2RGB)

        try:
            results = ocr_engine.ocr(img_rgb)
        except Exception:
            logger.exception("PaddleOCR inference failed")
            return None, 0.0

        if not results:
            continue

        h, w = img_rgb.shape[:2]
        tokens = _tokens_from_results(results, img_w=w, img_h=h)
        if not tokens:
            continue

        for token in tokens:
            if token["conf"] > best_any[1]:
                best_any = (token["text"], token["conf"])

        joined_text, joined_conf = _join_tokens_reading_order(tokens)
        if not joined_text:
            continue

        candidates = [(joined_text, joined_conf)]
        repaired = repair_missing_series_letter(joined_text)
        if repaired != joined_text:
            candidates.append((repaired, joined_conf * 0.98))

        for text, score in candidates:
            if (looks_like_plate(text) or len(text) >= 7) and score > best_plate[1]:
                best_plate = (text, score)

        for token in tokens:
            text = token["text"]
            score = token["conf"]
            if (looks_like_plate(text) or len(text) >= 7) and score > best_plate[1]:
                best_plate = (text, score)

        # Early exit for fast path: a strong plate-like result is already found.
        if best_plate[0] is not None and best_plate[1] >= 0.90:
            return best_plate

    if best_plate[0] is not None:
        return best_plate
    if best_any[0] is not None and len(best_any[0]) >= 4:
        return best_any
    return None, 0.0
