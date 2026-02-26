from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    model_path: Path = Path("models/numberplate_yolov8n.pt")
    conf_thr: float = Field(default=0.5, ge=0.0, le=1.0)
    iou_thr: float = Field(default=0.5, ge=0.0, le=1.0)
    ocr_service_url: str = "http://127.0.0.1:8001/v1/ocr/plate"
    ocr_timeout_s: float = Field(default=10.0, gt=0.0, le=60.0)

    @property
    def model_path_str(self) -> str:
        """Convenience accessor for libraries that expect a plain string path."""
        return str(self.model_path)


settings = Settings()
