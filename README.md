# PoC-2: Product Detection on Shelf / Fridge Images

## Objective
PoC detects the numberplate of a car

---

## What this PoC does
- Detects product instances using a YOLOv8n object detection model
- Returns bounding boxes and confidence scores for each detected numberplate
- Produces explainable, auditable outputs suitable for analytics and reporting

---

## Model Training Command
- yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640