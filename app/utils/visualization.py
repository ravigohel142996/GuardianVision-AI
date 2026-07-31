"""
pipeline.py
===========
End-to-end per-camera processing pipeline for GuardianVision AI.

Wires together the detector, tracker, violation engine, and alert
manager into a single frame-processing loop, and exposes a
`run_stream()` generator that the dashboard and API both consume.

This is the module that turns four independent components into an
actual product.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Generator, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from app.core.alert_manager import Alert, AlertManager
from app.core.detector import Detection, PPEDetector
from app.core.tracker import WorkerTracker
from app.core.violation_engine import ComplianceResult, ViolationEngine
from app.utils.config_loader import AppConfig, CameraConfig, get_config
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FrameResult:
    """Everything produced from processing a single video frame."""

    frame: np.ndarray
    annotated_frame: np.ndarray
    compliance_results: List[ComplianceResult]
    new_alerts: List[Alert]
    fps: float
    worker_count: int
    violation_count: int


class PipelineError(Exception):
    """Raised when the pipeline cannot open or process a camera source."""


class CameraPipeline:
    """
    Processes a single camera's video stream end-to-end: detection,
    tracking, compliance evaluation, alerting, and annotation.
    """

    def __init__(self, camera: CameraConfig, config: Optional[AppConfig] = None) -> None:
        self.camera = camera
        self.config = config or get_config()
        self.zone = self.config.get_zone(camera.zone_id)

        self._detector = PPEDetector(self.config)
        # Tracker and detector share the same underlying YOLO model instance
        # to avoid loading weights twice.
        self._tracker = WorkerTracker(self._detector._model, self.config)
        self._violation_engine = ViolationEngine(self.config)
        self._alert_manager = AlertManager(self.config)

        self._capture: Optional[cv2.VideoCapture] = None

    def _open_capture(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self.camera.source)
        if not capture.isOpened():
            raise PipelineError(
                f"Failed to open camera source '{self.camera.source}' "
                f"for camera '{self.camera.id}'."
            )
        return capture

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        """
        Run the full detect -> track -> evaluate -> alert cycle on one frame.

        Returns
        -------
        FrameResult
            All outputs needed by the dashboard/API for this frame.
        """
        start_time = time.time()

        all_detections: List[Detection] = self._detector.detect(frame)
        ppe_detections = self._detector.get_ppe_detections(all_detections)

        tracked_workers = self._tracker.update(frame)

        compliance_results = self._violation_engine.evaluate(
            tracked_workers, ppe_detections, self.zone
        )

        new_alerts: List[Alert] = []
        for result in compliance_results:
            if self._violation_engine.should_alert(result):
                alert = self._alert_manager.raise_alert(
                    result, frame, self.camera.id, self.zone.id
                )
                new_alerts.append(alert)

        annotated_frame = self._annotate(frame, compliance_results)

        elapsed = max(time.time() - start_time, 1e-6)
        fps = 1.0 / elapsed
        violation_count = sum(1 for r in compliance_results if not r.is_compliant)

        return FrameResult(
            frame=frame,
            annotated_frame=annotated_frame,
            compliance_results=compliance_results,
            new_alerts=new_alerts,
            fps=round(fps, 1),
            worker_count=len(compliance_results),
            violation_count=violation_count,
        )

    def _annotate(self, frame: np.ndarray, results: List[ComplianceResult]) -> np.ndarray:
        """Draw bounding boxes color-coded by compliance status."""
        annotated = frame.copy()

        for result in results:
            x1, y1, x2, y2 = result.bbox
            color = (0, 200, 0) if result.is_compliant else (0, 0, 255)
            status = "COMPLIANT" if result.is_compliant else "VIOLATION"
            label = f"#{result.track_id} {status}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 8, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

        return annotated

    def run_stream(self) -> Generator[FrameResult, None, None]:
        """
        Continuously read frames from the camera source and yield
        FrameResult objects. Intended to be consumed by the dashboard's
        live view or the API's streaming endpoint.
        """
        self._capture = self._open_capture()
        logger.info(f"Started stream for camera '{self.camera.id}' ({self.camera.name})")

        try:
            while True:
                ok, frame = self._capture.read()
                if not ok:
                    logger.warning(
                        f"Camera '{self.camera.id}' stream ended or dropped a frame."
                    )
                    break

                yield self.process_frame(frame)
        finally:
            self.release()

    def release(self) -> None:
        """Release the underlying video capture handle."""
        if self._capture is not None:
            self._capture.release()
            logger.info(f"Released camera '{self.camera.id}'.")


class PipelineManager:
    """
    Manages multiple `CameraPipeline` instances — one per enabled camera
    in the configuration — for multi-camera deployments.
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or get_config()
        self.pipelines: dict[str, CameraPipeline] = {
            cam.id: CameraPipeline(cam, self.config)
            for cam in self.config.enabled_cameras()
        }
        logger.info(f"Initialized {len(self.pipelines)} camera pipeline(s).")

    def get_pipeline(self, camera_id: str) -> CameraPipeline:
        if camera_id not in self.pipelines:
            raise PipelineError(f"No pipeline configured for camera '{camera_id}'.")
        return self.pipelines[camera_id]

    def release_all(self) -> None:
        for pipeline in self.pipelines.values():
            pipeline.release()
