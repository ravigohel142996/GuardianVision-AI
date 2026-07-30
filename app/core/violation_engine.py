"""
violation_engine.py
====================
Core business logic for GuardianVision AI: decides whether a tracked
worker is PPE-compliant, based on spatial association between "person"
boxes (from the tracker) and PPE item boxes (from the detector).

This is the module that turns raw detections into an actionable safety
decision — the difference between a "detector" and a real compliance
system.

Algorithm
---------
1. For each tracked worker's bounding box, find PPE detections whose
   centroid falls inside (or sufficiently overlaps) the worker's box.
2. Compare the set of PPE items found against the zone's required list.
3. Missing items -> violation. Apply a grace period (to smooth out
   transient misses from occlusion/motion blur) and a cooldown (to
   avoid re-alerting on the same worker every frame).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from app.core.detector import Detection
from app.core.tracker import TrackedWorker
from app.utils.config_loader import AppConfig, ZoneConfig, get_config
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ComplianceResult:
    """Compliance evaluation outcome for a single tracked worker."""

    track_id: int
    is_compliant: bool
    missing_ppe: List[str]
    detected_ppe: List[str]
    risk_level: str
    bbox: Tuple[int, int, int, int]


@dataclass
class _WorkerViolationState:
    """Internal bookkeeping used to apply grace periods and cooldowns."""

    consecutive_violation_frames: int = 0
    last_alert_timestamp: float = 0.0


class ViolationEngine:
    """
    Evaluates PPE compliance for tracked workers and decides when a
    violation should trigger an alert (vs. being suppressed by the
    grace period / cooldown logic).
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self._worker_states: Dict[int, _WorkerViolationState] = {}

    def evaluate(
        self,
        tracked_workers: List[TrackedWorker],
        ppe_detections: List[Detection],
        zone: ZoneConfig,
    ) -> List[ComplianceResult]:
        """
        Evaluate compliance for every tracked worker in the current frame.

        Parameters
        ----------
        tracked_workers:
            Persons with assigned track IDs from the tracker.
        ppe_detections:
            All non-person detections from the detector (helmet, vest, etc.)
        zone:
            The zone configuration defining required PPE for this camera.

        Returns
        -------
        List[ComplianceResult]
            One result per tracked worker.
        """
        results: List[ComplianceResult] = []

        for worker in tracked_workers:
            associated_ppe = self._associate_ppe(worker, ppe_detections)
            detected_positive = self._positive_items(associated_ppe)
            missing = self._compute_missing(detected_positive, zone.required_ppe)

            is_compliant = len(missing) == 0
            risk_level = self._compute_risk_level(missing)

            results.append(
                ComplianceResult(
                    track_id=worker.track_id,
                    is_compliant=is_compliant,
                    missing_ppe=missing,
                    detected_ppe=detected_positive,
                    risk_level=risk_level,
                    bbox=worker.bbox,
                )
            )

        return results

    def should_alert(self, result: ComplianceResult) -> bool:
        """
        Decide whether a compliance result should generate a new alert,
        applying the grace period and cooldown rules from config.

        This maintains per-worker state across calls, so it must be
        invoked exactly once per (worker, frame) pair in frame order.
        """
        state = self._worker_states.setdefault(result.track_id, _WorkerViolationState())

        if result.is_compliant:
            state.consecutive_violation_frames = 0
            return False

        state.consecutive_violation_frames += 1

        grace_period = self.config.violation_engine.grace_period_frames
        if state.consecutive_violation_frames < grace_period:
            return False  # still within grace period, don't alert yet

        cooldown = self.config.violation_engine.violation_cooldown_seconds
        now = time.time()
        if now - state.last_alert_timestamp < cooldown:
            return False  # already alerted recently for this worker

        state.last_alert_timestamp = now
        logger.info(
            f"Violation confirmed for worker {result.track_id}: "
            f"missing={result.missing_ppe} risk={result.risk_level}"
        )
        return True

    def _associate_ppe(
        self, worker: TrackedWorker, ppe_detections: List[Detection]
    ) -> List[Detection]:
        """Return PPE detections whose centroid falls inside the worker's bbox."""
        wx1, wy1, wx2, wy2 = worker.bbox
        associated = []

        for det in ppe_detections:
            dx1, dy1, dx2, dy2 = det.bbox
            cx, cy = (dx1 + dx2) // 2, (dy1 + dy2) // 2

            if wx1 <= cx <= wx2 and wy1 <= cy <= wy2:
                associated.append(det)

        return associated

    def _positive_items(self, associated_ppe: List[Detection]) -> List[str]:
        """
        Extract the list of *positively worn* PPE item names, filtering out
        the model's explicit "no_X" negative classes (e.g. 'no_helmet').
        """
        return [
            det.class_name
            for det in associated_ppe
            if not det.class_name.startswith("no_")
        ]

    def _compute_missing(
        self, detected_positive: List[str], required: List[str]
    ) -> List[str]:
        """Return the required PPE items that were NOT positively detected."""
        return [item for item in required if item not in detected_positive]

    def _compute_risk_level(self, missing: List[str]) -> str:
        """
        Derive an overall risk level for a violation from the highest-risk
        missing item, per the risk_levels map in config.
        """
        if not missing:
            return "none"

        risk_order = {"high": 3, "medium": 2, "low": 1}
        levels = [
            self.config.violation_engine.risk_levels.get(item, "low") for item in missing
        ]
        return max(levels, key=lambda lvl: risk_order.get(lvl, 0))

    def reset_worker_state(self, track_id: int) -> None:
        """Clear tracked violation state for a worker (e.g. when they leave frame)."""
        self._worker_states.pop(track_id, None)
