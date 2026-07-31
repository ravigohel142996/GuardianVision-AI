"""
test_violation_engine.py
==========================
Unit tests for `app.core.violation_engine.ViolationEngine`.

These tests exercise the PPE-to-worker association logic and the
grace-period/cooldown alerting rules without requiring a real YOLO
model or camera — pure business-logic testing.
"""

from __future__ import annotations

import time

import pytest

from app.core.detector import Detection
from app.core.tracker import TrackedWorker
from app.core.violation_engine import ViolationEngine
from app.utils.config_loader import ZoneConfig, get_config


@pytest.fixture
def zone() -> ZoneConfig:
    return ZoneConfig(
        id="zone_test",
        name="Test Zone",
        camera_id="cam_test",
        required_ppe=["helmet", "vest", "gloves"],
    )


@pytest.fixture
def engine() -> ViolationEngine:
    return ViolationEngine(config=get_config())


def make_worker(track_id: int, bbox=(100, 100, 300, 400)) -> TrackedWorker:
    return TrackedWorker(track_id=track_id, bbox=bbox, confidence=0.9)


def make_ppe_detection(class_name: str, bbox) -> Detection:
    return Detection(class_id=0, class_name=class_name, confidence=0.9, bbox=bbox)


class TestPPEAssociation:
    def test_fully_compliant_worker(self, engine, zone):
        worker = make_worker(1)
        ppe = [
            make_ppe_detection("helmet", (150, 110, 200, 160)),
            make_ppe_detection("vest", (150, 200, 250, 350)),
            make_ppe_detection("gloves", (280, 300, 300, 340)),
        ]

        results = engine.evaluate([worker], ppe, zone)

        assert len(results) == 1
        assert results[0].is_compliant is True
        assert results[0].missing_ppe == []

    def test_missing_helmet_flagged(self, engine, zone):
        worker = make_worker(2)
        ppe = [
            make_ppe_detection("vest", (150, 200, 250, 350)),
            make_ppe_detection("gloves", (280, 300, 300, 340)),
        ]

        results = engine.evaluate([worker], ppe, zone)

        assert results[0].is_compliant is False
        assert "helmet" in results[0].missing_ppe
        assert results[0].risk_level == "high"

    def test_ppe_outside_worker_bbox_not_associated(self, engine, zone):
        worker = make_worker(3, bbox=(0, 0, 50, 50))
        ppe = [make_ppe_detection("helmet", (500, 500, 550, 550))]  # far away

        results = engine.evaluate([worker], ppe, zone)

        assert "helmet" in results[0].missing_ppe

    def test_negative_class_not_counted_as_worn(self, engine, zone):
        worker = make_worker(4)
        ppe = [
            make_ppe_detection("no_helmet", (150, 110, 200, 160)),
            make_ppe_detection("vest", (150, 200, 250, 350)),
            make_ppe_detection("gloves", (280, 300, 300, 340)),
        ]

        results = engine.evaluate([worker], ppe, zone)

        assert results[0].is_compliant is False
        assert "helmet" in results[0].missing_ppe


class TestAlertingRules:
    def test_no_alert_within_grace_period(self, engine, zone):
        worker = make_worker(5)
        ppe: list = []  # missing everything

        result = engine.evaluate([worker], ppe, zone)[0]

        # First violation frame should not immediately alert
        assert engine.should_alert(result) is False

    def test_alert_after_grace_period_elapses(self, engine, zone):
        worker = make_worker(6)
        ppe: list = []

        result = engine.evaluate([worker], ppe, zone)[0]
        grace_period = engine.config.violation_engine.grace_period_frames

        alerted = False
        for _ in range(grace_period + 1):
            alerted = engine.should_alert(result) or alerted

        assert alerted is True

    def test_no_duplicate_alert_within_cooldown(self, engine, zone):
        worker = make_worker(7)
        ppe: list = []
        result = engine.evaluate([worker], ppe, zone)[0]

        grace_period = engine.config.violation_engine.grace_period_frames
        for _ in range(grace_period + 1):
            engine.should_alert(result)

        # Immediately re-check — should be suppressed by cooldown
        assert engine.should_alert(result) is False

    def test_compliant_worker_resets_violation_streak(self, engine, zone):
        worker = make_worker(8)
        ppe_missing: list = []
        ppe_full = [
            make_ppe_detection("helmet", (150, 110, 200, 160)),
            make_ppe_detection("vest", (150, 200, 250, 350)),
            make_ppe_detection("gloves", (280, 300, 300, 340)),
        ]

        violating_result = engine.evaluate([worker], ppe_missing, zone)[0]
        engine.should_alert(violating_result)

        compliant_result = engine.evaluate([worker], ppe_full, zone)[0]
        assert engine.should_alert(compliant_result) is False

        state = engine._worker_states[worker.track_id]
        assert state.consecutive_violation_frames == 0
