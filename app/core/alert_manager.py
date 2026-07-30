"""
alert_manager.py
=================
Turns a confirmed violation into a full alert: saves an annotated
screenshot, writes an Incident row to the database, and optionally
dispatches an outbound webhook notification (Slack/Teams).

Kept separate from `violation_engine.py` so the *decision* of "is this
a violation worth alerting on" stays independent of the *side effects*
of alerting (I/O, network calls) — easier to test and reason about.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests

from app.core.violation_engine import ComplianceResult
from app.utils.config_loader import AppConfig, get_config
from app.utils.database import save_incident
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Alert:
    """A fully processed alert, ready for the dashboard's live feed."""

    worker_id: int
    camera_id: str
    zone_id: str
    missing_ppe: list
    risk_level: str
    confidence: float
    screenshot_path: Optional[str]
    timestamp: str


class AlertManager:
    """
    Coordinates the side effects triggered by a confirmed PPE violation:
    screenshot capture, database logging, and webhook notification.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self._ensure_screenshot_dir()

    def _ensure_screenshot_dir(self) -> None:
        Path(self.config.alerts.screenshot_dir).mkdir(parents=True, exist_ok=True)

    def raise_alert(
        self,
        result: ComplianceResult,
        frame: np.ndarray,
        camera_id: str,
        zone_id: str,
    ) -> Alert:
        """
        Process a confirmed violation: capture evidence, persist it, notify.

        Parameters
        ----------
        result:
            The compliance result that triggered this alert (already
            confirmed by `ViolationEngine.should_alert`).
        frame:
            The current video frame, used to crop/save a screenshot.
        camera_id, zone_id:
            Identify where the violation occurred.

        Returns
        -------
        Alert
            The finalized alert object.
        """
        screenshot_path = None
        if self.config.alerts.save_screenshot:
            screenshot_path = self._save_screenshot(frame, result, camera_id)

        try:
            incident = save_incident(
                worker_id=result.track_id,
                camera_id=camera_id,
                zone_id=zone_id,
                missing_ppe=result.missing_ppe,
                risk_level=result.risk_level,
                confidence=self._estimate_confidence(result),
                screenshot_path=screenshot_path,
            )
            timestamp = incident.timestamp.isoformat()
        except Exception:
            logger.exception("Failed to persist incident to database.")
            timestamp = datetime.utcnow().isoformat()

        alert = Alert(
            worker_id=result.track_id,
            camera_id=camera_id,
            zone_id=zone_id,
            missing_ppe=result.missing_ppe,
            risk_level=result.risk_level,
            confidence=self._estimate_confidence(result),
            screenshot_path=screenshot_path,
            timestamp=timestamp,
        )

        self._dispatch_webhook(alert)
        return alert

    def _save_screenshot(
        self, frame: np.ndarray, result: ComplianceResult, camera_id: str
    ) -> Optional[str]:
        """Draw the violation bbox/label onto a frame copy and save it to disk."""
        try:
            annotated = frame.copy()
            x1, y1, x2, y2 = result.bbox
            label = f"Worker #{result.track_id} - Missing: {', '.join(result.missing_ppe)}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

            filename = (
                f"{camera_id}_worker{result.track_id}_"
                f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            )
            filepath = Path(self.config.alerts.screenshot_dir) / filename
            cv2.imwrite(str(filepath), annotated)

            return str(filepath)
        except Exception:
            logger.exception("Failed to save violation screenshot.")
            return None

    def _estimate_confidence(self, result: ComplianceResult) -> float:
        """
        Placeholder confidence heuristic for the overall violation call.
        A fuller implementation would aggregate per-detection confidences
        from the associated PPE/person boxes.
        """
        return 0.85 if result.risk_level == "high" else 0.75

    def _dispatch_webhook(self, alert: Alert) -> None:
        """Send an outbound notification if a webhook URL is configured."""
        webhook_url = self.config.alerts.notify_webhook
        if not webhook_url:
            return

        payload = {
            "text": (
                f"🚨 PPE Violation — Worker #{alert.worker_id} on {alert.camera_id}\n"
                f"Missing: {', '.join(alert.missing_ppe)}\n"
                f"Risk: {alert.risk_level.upper()}\n"
                f"Time: {alert.timestamp}"
            )
        }

        try:
            requests.post(webhook_url, json=payload, timeout=5)
        except requests.RequestException:
            logger.exception("Failed to dispatch webhook notification.")
