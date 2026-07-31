"""
routes.py
=========
Route handlers for the GuardianVision AI REST API.

Separated from `main.py` (which only wires up the FastAPI app and
middleware) so route logic can be unit-tested independently via
`tests/test_api.py` without spinning up the full ASGI app each time.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from api.schemas import (
    CameraStatus,
    DashboardSummaryResponse,
    HealthResponse,
    IncidentResponse,
    LiveStatusResponse,
    ReportRequest,
    ReportResponse,
    ViolationsListResponse,
)
from app.utils.config_loader import get_config
from app.utils.database import get_incident_stats, get_recent_incidents
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# In-memory registry updated by the running pipeline processes.
# In a multi-process deployment this would be backed by Redis; kept
# in-process here to match the project's single-service Docker setup.
_live_camera_state: dict = {}


def update_live_state(camera_id: str, camera_name: str, zone_id: str, fps: float,
                       worker_count: int, violation_count: int) -> None:
    """Called by the pipeline runner to publish the latest per-camera stats."""
    _live_camera_state[camera_id] = {
        "camera_id": camera_id,
        "camera_name": camera_name,
        "zone_id": zone_id,
        "is_streaming": True,
        "fps": fps,
        "worker_count": worker_count,
        "violation_count": violation_count,
    }


@router.get("/status", response_model=HealthResponse, tags=["Health"])
def get_status() -> HealthResponse:
    """Basic liveness check — used by Docker healthchecks and uptime monitors."""
    config = get_config()
    return HealthResponse(
        status="ok",
        version=config.version,
        environment=config.environment,
        active_cameras=len(config.enabled_cameras()),
    )


@router.get("/live", response_model=LiveStatusResponse, tags=["Monitoring"])
def get_live_status() -> LiveStatusResponse:
    """Real-time snapshot of every camera's current worker/violation counts."""
    cameras = [CameraStatus(**state) for state in _live_camera_state.values()]

    return LiveStatusResponse(
        timestamp=datetime.utcnow(),
        total_workers=sum(c.worker_count for c in cameras),
        total_violations=sum(c.violation_count for c in cameras),
        cameras=cameras,
    )


@router.get("/violations", response_model=ViolationsListResponse, tags=["Incidents"])
def list_violations(
    limit: int = Query(50, ge=1, le=500),
    camera_id: Optional[str] = Query(None),
) -> ViolationsListResponse:
    """Return recent logged violations, optionally filtered by camera."""
    try:
        incidents = get_recent_incidents(limit=limit, camera_id=camera_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ViolationsListResponse(
        count=len(incidents),
        incidents=[IncidentResponse(**incident) for incident in incidents],
    )


@router.get("/dashboard", response_model=DashboardSummaryResponse, tags=["Monitoring"])
def get_dashboard_summary() -> DashboardSummaryResponse:
    """Aggregate stats consumed by the dashboard's top summary cards."""
    try:
        stats = get_incident_stats()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    config = get_config()
    total_workers = sum(state["worker_count"] for state in _live_camera_state.values())

    return DashboardSummaryResponse(
        total_incidents=stats["total_incidents"],
        high_risk_incidents=stats["high_risk_incidents"],
        active_cameras=len(config.enabled_cameras()),
        total_workers_tracked=total_workers,
    )


@router.post("/report", response_model=ReportResponse, tags=["Reports"])
def generate_report(request: ReportRequest) -> ReportResponse:
    """
    Generate a filtered incident report and export it as CSV or JSON.

    Filters (camera, zone, risk level, date range) are applied in-memory
    against recently logged incidents; for very large histories this
    would move to a proper SQL WHERE clause.
    """
    config = get_config()

    try:
        incidents = get_recent_incidents(limit=5000, camera_id=request.camera_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    filtered = _apply_report_filters(incidents, request)

    file_path = None
    if request.export_format == "csv":
        file_path = _export_csv(filtered, config.reports.csv_export_path)
    elif request.export_format == "json":
        file_path = _export_json(filtered, config.reports.json_export_path)
    else:
        raise HTTPException(
            status_code=400, detail="export_format must be 'csv' or 'json'"
        )

    return ReportResponse(
        generated_at=datetime.utcnow(),
        record_count=len(filtered),
        export_format=request.export_format,
        file_path=file_path,
    )


def _apply_report_filters(incidents: List[dict], request: ReportRequest) -> List[dict]:
    """Apply zone/risk/date filters to an in-memory incident list."""
    result = incidents

    if request.zone_id:
        result = [i for i in result if i["zone_id"] == request.zone_id]

    if request.risk_level:
        result = [i for i in result if i["risk_level"] == request.risk_level]

    if request.start_date:
        result = [
            i for i in result
            if datetime.fromisoformat(i["timestamp"]) >= request.start_date
        ]

    if request.end_date:
        result = [
            i for i in result
            if datetime.fromisoformat(i["timestamp"]) <= request.end_date
        ]

    return result


def _export_csv(incidents: List[dict], path: str) -> str:
    """Write incidents to a CSV file and return the file path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id", "worker_id", "camera_id", "zone_id",
        "missing_ppe", "risk_level", "confidence",
        "screenshot_path", "timestamp",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for incident in incidents:
            row = incident.copy()
            row["missing_ppe"] = ";".join(row["missing_ppe"])
            writer.writerow(row)

    logger.info(f"Exported {len(incidents)} incidents to CSV at {output_path}")
    return str(output_path)


def _export_json(incidents: List[dict], path: str) -> str:
    """Write incidents to a JSON file and return the file path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(incidents, f, indent=2, default=str)

    logger.info(f"Exported {len(incidents)} incidents to JSON at {output_path}")
    return str(output_path)
