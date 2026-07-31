"""
schemas.py
==========
Pydantic models defining the request/response contracts for the
GuardianVision AI REST API. Keeping these separate from `main.py`
mirrors how production FastAPI services are structured, and gives
`docs/API.md` a single source to document against.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response for GET /status — basic liveness/health check."""

    status: str = Field(..., example="ok")
    version: str = Field(..., example="1.0.0")
    environment: str = Field(..., example="development")
    active_cameras: int = Field(..., example=2)


class CameraStatus(BaseModel):
    """Per-camera runtime status, part of the /live response."""

    camera_id: str
    camera_name: str
    zone_id: str
    is_streaming: bool
    fps: float
    worker_count: int
    violation_count: int


class LiveStatusResponse(BaseModel):
    """Response for GET /live — real-time snapshot across all cameras."""

    timestamp: datetime
    total_workers: int
    total_violations: int
    cameras: List[CameraStatus]


class IncidentResponse(BaseModel):
    """A single logged PPE violation incident."""

    id: int
    worker_id: int
    camera_id: str
    zone_id: str
    missing_ppe: List[str]
    risk_level: str
    confidence: float
    screenshot_path: Optional[str]
    timestamp: datetime


class ViolationsListResponse(BaseModel):
    """Response for GET /violations — paginated incident history."""

    count: int
    incidents: List[IncidentResponse]


class DashboardSummaryResponse(BaseModel):
    """Response for GET /dashboard — aggregate stats for the UI header cards."""

    total_incidents: int
    high_risk_incidents: int
    active_cameras: int
    total_workers_tracked: int


class ReportRequest(BaseModel):
    """Request body for POST /report — generate a filtered incident report."""

    camera_id: Optional[str] = None
    zone_id: Optional[str] = None
    risk_level: Optional[str] = Field(None, example="high")
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    export_format: str = Field("json", example="csv")


class ReportResponse(BaseModel):
    """Response for POST /report."""

    generated_at: datetime
    record_count: int
    export_format: str
    file_path: Optional[str] = None
