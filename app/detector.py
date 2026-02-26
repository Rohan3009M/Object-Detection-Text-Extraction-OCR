import numpy as np
import torch
from ultralytics import YOLO


class ProductDetector:
    def __init__(
        self,
        model_path: str,
        conf_thr: float = 0.5,
        iou_thr: float = 0.5,
        device: str | None = None,
    ):
        self.model = YOLO(model_path)
        self.conf_thr = conf_thr
        self.iou_thr = iou_thr

        # Auto-pick GPU if available (or allow explicit override).
        self.device = device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")

    def predict(
        self,
        img_bgr: np.ndarray,
        conf_thr: float | None = None,
        iou_thr: float | None = None,
    ) -> list[dict]:
        """
        Returns list of dicts: {cls, conf, bbox=[x1,y1,x2,y2]} in pixel coords
        """
        conf = self.conf_thr if conf_thr is None else conf_thr
        iou = self.iou_thr if iou_thr is None else iou_thr

        res = self.model.predict(
            img_bgr,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False,
        )[0]

        dets: list[dict] = []
        if res.boxes is None or len(res.boxes) == 0:
            return dets

        for b in res.boxes:
            score = float(b.conf.item())
            if score < conf:
                continue

            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            cls_id = int(b.cls.item())
            cls_name = self.model.names.get(cls_id, str(cls_id))
            dets.append({"cls": cls_name, "conf": score, "bbox": [x1, y1, x2, y2]})

        return dets
