"""
test_database.py
=================
Unit tests for `app.utils.database` — incident persistence, retrieval,
and aggregate stats, run against a temporary SQLite file per test.
"""

from __future__ import annotations

import app.utils.database as db_module
from app.utils.database import (
    get_incident_stats,
    get_recent_incidents,
    init_db,
    save_incident,
)


def _fresh_db(temp_db_path: str) -> None:
    """Force re-initialization against a temp DB, bypassing the module cache."""
    db_module._engine = None
    db_module._SessionLocal = None
    init_db(temp_db_path)


class TestIncidentPersistence:
    def test_save_and_retrieve_incident(self, temp_db_path):
        _fresh_db(temp_db_path)

        saved = save_incident(
            worker_id=1,
            camera_id="cam_01",
            zone_id="zone_a",
            missing_ppe=["helmet", "vest"],
            risk_level="high",
            confidence=0.91,
            screenshot_path="reports/screenshots/test.jpg",
        )

        assert saved.id is not None

        incidents = get_recent_incidents(limit=10)
        assert len(incidents) == 1
        assert incidents[0]["worker_id"] == 1
        assert incidents[0]["missing_ppe"] == ["helmet", "vest"]
        assert incidents[0]["risk_level"] == "high"

    def test_filter_by_camera(self, temp_db_path):
        _fresh_db(temp_db_path)

        save_incident(1, "cam_01", "zone_a", ["helmet"], "high", 0.9)
        save_incident(2, "cam_02", "zone_b", ["vest"], "medium", 0.8)

        cam_01_incidents = get_recent_incidents(limit=10, camera_id="cam_01")
        assert len(cam_01_incidents) == 1
        assert cam_01_incidents[0]["camera_id"] == "cam_01"

    def test_incident_stats_counts_high_risk(self, temp_db_path):
        _fresh_db(temp_db_path)

        save_incident(1, "cam_01", "zone_a", ["helmet"], "high", 0.9)
        save_incident(2, "cam_01", "zone_a", ["gloves"], "low", 0.7)
        save_incident(3, "cam_01", "zone_a", ["vest"], "high", 0.85)

        stats = get_incident_stats()
        assert stats["total_incidents"] == 3
        assert stats["high_risk_incidents"] == 2

    def test_recent_incidents_ordered_newest_first(self, temp_db_path):
        _fresh_db(temp_db_path)

        first = save_incident(1, "cam_01", "zone_a", ["helmet"], "high", 0.9)
        second = save_incident(2, "cam_01", "zone_a", ["vest"], "medium", 0.8)

        incidents = get_recent_incidents(limit=10)
        assert incidents[0]["id"] == second.id
        assert incidents[1]["id"] == first.id
