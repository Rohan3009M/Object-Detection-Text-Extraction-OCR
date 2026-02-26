# Number Plate Detection + OCR (YOLOv8 + PaddleOCR)

This project detects vehicle number plates in images using a YOLOv8 model and then extracts plate text using a separate OCR service (PaddleOCR). It includes:

- `FastAPI` detector API (`app.main`)
- `FastAPI` OCR microservice (`app.ocr_service`)
- `Streamlit` UI for single and batch analysis (`streamlit.py`)

## Project Structure

- `app/main.py` - detector API (YOLO + calls OCR service)
- `app/ocr_service.py` - OCR API (PaddleOCR)
- `app/ocr.py` - OCR preprocessing and plate-text extraction logic
- `app/detector.py` - YOLO inference wrapper
- `streamlit.py` - web UI for testing single/batch images
- `models/numberplate_yolov8n.pt` - trained number plate detection model
- `data.yaml` - YOLO dataset config for training
- `requirements.txt` - main app + UI dependencies
- `requirements-ocr.txt` - OCR service dependencies (for `.venv-ocr`)

## How It Works

1. Client uploads image(s) to detector API (`/v1/analyze` or `/v1/analyze/batch`)
2. Detector API runs YOLO on the image to find number plate bounding boxes
3. Best plate crop is sent to OCR service (`/v1/ocr/plate`)
4. OCR service runs PaddleOCR and returns plate text + OCR confidence
5. Detector API merges detection + OCR results and returns JSON

## Requirements

- Python `3.10` recommended (matches `Dockerfile`)
- Windows/Linux supported (Windows may need Visual C++ runtime for some ML packages)
- GPU optional
  - YOLO auto-uses CUDA if available (`torch.cuda.is_available()`)
  - OCR service prefers GPU and falls back to CPU

## Setup (Recommended: 2 Virtual Environments)

The project is designed to run detector and OCR services separately.

### 1. Main environment (`.venv`) for detector API + Streamlit

```powershell
cd "R:\Atrina Office\Numberplate\Product-analyzer"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. OCR environment (`.venv-ocr`) for PaddleOCR service

```powershell
cd "R:\Atrina Office\Numberplate\Product-analyzer"
python -m venv .venv-ocr
.\.venv-ocr\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-ocr.txt
```

Notes:

- `paddlepaddle` / `paddleocr` installation can be platform-dependent.
- If GPU OCR is not available, the code automatically falls back to CPU.
- For CUDA builds of Paddle, follow the official Paddle install matrix and then install the remaining packages from `requirements-ocr.txt`.

## Run the Services

Open separate terminals.

### Terminal 1: OCR Service (port `8001`)

```powershell
cd "R:\Atrina Office\Numberplate\Product-analyzer"
.\.venv-ocr\Scripts\Activate.ps1
uvicorn app.ocr_service:app --host 127.0.0.1 --port 8001
```

Health check:

```powershell
curl http://127.0.0.1:8001/health
```

### Terminal 2: Detector API (port `8000`)

```powershell
cd "R:\Atrina Office\Numberplate\Product-analyzer"
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Health check:

```powershell
curl http://127.0.0.1:8000/health
```

### Terminal 3: Streamlit UI (optional, port `8501`)

```powershell
cd "R:\Atrina Office\Numberplate\Product-analyzer"
.\.venv\Scripts\Activate.ps1
streamlit run streamlit.py
```

In the sidebar, keep the API base URL as:

- `http://127.0.0.1:8000`

## API Endpoints

### Detector API (`app.main`)

- `GET /health`
- `POST /v1/analyze`
- `POST /v1/analyze/batch`

#### `POST /v1/analyze`

Form-data:

- `file`: image file (`jpg`, `jpeg`, `png`)

Query params (optional):

- `conf_thr` (default `0.5`)
- `iou_thr` (default `0.5`)

Example:

```powershell
curl -X POST "http://127.0.0.1:8000/v1/analyze?conf_thr=0.5&iou_thr=0.5" `
  -F "file=@test1.jpg"
```

### OCR API (`app.ocr_service`)

- `GET /health`
- `POST /v1/ocr/plate`

Example:

```powershell
curl -X POST "http://127.0.0.1:8001/v1/ocr/plate" `
  -F "file=@test1.jpg"
```

## Training (YOLOv8)

Dataset config (`data.yaml`):

- `path: data/splits`
- `train: images/train`
- `val: images/val`
- class `0: number_plate`

Example training command:

```powershell
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640
```

Training outputs are generated in:

- `runs/`

## Docker (Current Repo Dockerfile)

`Dockerfile` currently builds a FastAPI service and runs:

- `uvicorn app.main:app`

If you use Docker for the detector API, ensure the OCR service is also running and reachable from the detector container (`ocr_service_url` in `app/config.py`).

## Configuration

Current defaults from `app/config.py`:

- model path: `models/numberplate_yolov8n.pt`
- detector `conf_thr`: `0.5`
- detector `iou_thr`: `0.5`
- OCR service URL: `http://127.0.0.1:8001/v1/ocr/plate`
- OCR timeout: `10s`

## Troubleshooting

### `403` on `git push`

Use GitHub PAT (not password) or SSH auth.

### `non-fast-forward` on push

Remote branch has commits already. Pull first or force-push if you intend to overwrite.

### OCR service returns degraded / unavailable

- Verify `.venv-ocr` dependencies are installed
- Start `app.ocr_service` first
- Check `http://127.0.0.1:8001/health`
- If Paddle GPU init fails, CPU fallback should be attempted automatically

### Detector API works but OCR text is empty

- Plate may be detected but OCR quality may be low for that crop
- Try clearer images / better plate crop quality
- Check OCR service logs for runtime errors

## Notes

- The project currently contains training artifacts and sample data/images locally.
- Ignore rules are configured to avoid committing generated caches and training outputs (`runs/`).
