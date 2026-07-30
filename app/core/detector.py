"""
detector.py
===========
YOLO11-based PPE object detector for GuardianVision AI.

Wraps Ultralytics' YOLO inference API behind a clean, typed interface
so the rest of the pipeline (tracker, violation engine) never touches
the underlying model library directly. This keeps `pipeline.py` free
to swap detection backends later (e.g. ONNX Runtime, TensorRT) without
touching downstream code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from ultralytics import YOLO

from app.utils.config_loader import AppConfig, get_config
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Detection:
    """A single detected object in one frame."""

    class_id: int
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2) in pixel coordinates


class DetectorError(Exception):
    """Raised when the detection model fails to load or run inference."""


class PPEDetector:
    """
    Loads a YOLO11 model trained on PPE classes and runs frame-level inference.

    Parameters
    ----------
    config:
        Application configuration. Defaults to the global cached config
        if not supplied.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self._model: YOLO | None = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the YOLO model weights from disk."""
        model_path = Path(self.config.detection.model_path)

        if not model_path.exists():
            raise DetectorError(
                f"Model weights not found at '{model_path}'. "
                "Train or download weights before running inference "
                "(see docs/Training.md)."
            )

        try:
            self._model = YOLO(str(model_path))
            if self.config.detection.device != "cpu":
                self._model.to(self.config.detection.device)
            logger.info(
                f"Loaded PPE detection model from '{model_path}' "
                f"on device '{self.config.detection.device}'"
            )
        except Exception as exc:
            raise DetectorError(f"Failed to load YOLO model: {exc}") from exc

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on a single BGR frame (as produced by OpenCV).

        Parameters
        ----------
        frame:
            HxWx3 numpy array in BGR format.

        Returns
        -------
        List[Detection]
            All detections above the configured confidence threshold.

        Raises
        ------
        DetectorError
            If inference fails or the model was never loaded.
        """
        if self._model is None:
            raise DetectorError("Detector model is not loaded.")

        if frame is None or frame.size == 0:
            logger.warning("Received empty frame; skipping inference.")
            return []

        try:
            results = self._model.predict(
                source=frame,
                conf=self.config.detection.confidence_threshold,
                iou=self.config.detection.iou_threshold,
                imgsz=self.config.detection.image_size,
                half=self.config.detection.half_precision,
                verbose=False,
            )
        except Exception as exc:
            raise DetectorError(f"Inference failed: {exc}") from exc

        return self._parse_results(results)

    def _parse_results(self, results) -> List[Detection]:
        """Convert Ultralytics' Results objects into our Detection dataclass."""
        detections: List[Detection] = []

        if not results:
            return detections

        result = results[0]
        if result.boxes is None:
            return detections

        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            class_name = self.config.detection.classes.get(class_id, f"class_{class_id}")

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(round(x1), round(y1), round(x2), round(y2)),
                )
            )

        return detections

    def get_person_detections(self, detections: List[Detection]) -> List[Detection]:
        """Filter a detection list down to only 'person' class detections."""
        return [d for d in detections if d.class_name == "person"]

    def get_ppe_detections(self, detections: List[Detection]) -> List[Detection]:
        """Filter a detection list down to all non-person (PPE) detections."""
        return [d for d in detections if d.class_name != "person"]
