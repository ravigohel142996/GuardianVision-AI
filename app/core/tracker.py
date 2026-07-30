"""
tracker.py
==========
Multi-object tracking layer for GuardianVision AI, based on the
ByteTrack algorithm (via Ultralytics' built-in tracker integration).

Assigns a persistent `track_id` to each detected worker across frames,
which is essential for:
  - avoiding duplicate violation alerts for the same worker,
  - computing per-worker dwell time in a zone,
  - building an auditable incident trail per individual.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from ultralytics import YOLO

from app.utils.config_loader import AppConfig, get_config
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TrackedWorker:
    """A person detection enriched with a persistent tracking identity."""

    track_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float


class TrackerError(Exception):
    """Raised when the tracking backend fails."""


class WorkerTracker:
    """
    Wraps Ultralytics' ByteTrack integration to maintain persistent
    worker identities across video frames.

    Note: Ultralytics exposes tracking via `model.track(...)`, which
    internally instantiates ByteTrack per the `bytetrack.yaml` config
    it ships with. We configure thresholds through our own config file
    and pass them at call time for full control and reproducibility.
    """

    def __init__(self, model: YOLO, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self._model = model
        self._track_history: Dict[int, List[Tuple[int, int]]] = {}

    def update(self, frame: np.ndarray) -> List[TrackedWorker]:
        """
        Run detection + tracking on a single frame and return tracked workers.

        Parameters
        ----------
        frame:
            HxWx3 BGR numpy array.

        Returns
        -------
        List[TrackedWorker]
            Person detections with persistent track IDs assigned.
        """
        if frame is None or frame.size == 0:
            logger.warning("Received empty frame; skipping tracking update.")
            return []

        try:
            results = self._model.track(
                source=frame,
                persist=True,
                conf=self.config.detection.confidence_threshold,
                iou=self.config.detection.iou_threshold,
                tracker="bytetrack.yaml",
                classes=[0],  # track only the 'person' class
                verbose=False,
            )
        except Exception as exc:
            raise TrackerError(f"Tracking update failed: {exc}") from exc

        return self._parse_results(results)

    def _parse_results(self, results) -> List[TrackedWorker]:
        """Convert Ultralytics tracking results into TrackedWorker objects."""
        tracked_workers: List[TrackedWorker] = []

        if not results or results[0].boxes is None or results[0].boxes.id is None:
            return tracked_workers

        result = results[0]
        boxes = result.boxes

        for box, track_id in zip(boxes, boxes.id):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf.item())
            tid = int(track_id.item())

            worker = TrackedWorker(
                track_id=tid,
                bbox=(round(x1), round(y1), round(x2), round(y2)),
                confidence=confidence,
            )
            tracked_workers.append(worker)
            self._update_history(tid, worker.bbox)

        return tracked_workers

    def _update_history(self, track_id: int, bbox: Tuple[int, int, int, int]) -> None:
        """Track the centroid trail of a worker for future zone-dwell analytics."""
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2

        history = self._track_history.setdefault(track_id, [])
        history.append((cx, cy))

        max_len = self.config.tracking.track_buffer
        if len(history) > max_len:
            self._track_history[track_id] = history[-max_len:]

    def get_track_history(self, track_id: int) -> List[Tuple[int, int]]:
        """Return the recent centroid trail for a given worker's track ID."""
        return self._track_history.get(track_id, [])

    def active_track_count(self) -> int:
        """Return the number of workers currently being tracked."""
        return len(self._track_history)
