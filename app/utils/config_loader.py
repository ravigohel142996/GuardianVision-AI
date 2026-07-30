"""
config_loader.py
==================
Centralized configuration loader for GuardianVision AI.

Loads `configs/config.yaml`, applies environment variable overrides
(from `.env` or the process environment), validates required fields,
and exposes a single cached, typed `AppConfig` object to the rest of
the application via `get_config()`.

Design notes
------------
- Uses a module-level singleton (`_config_cache`) so YAML is parsed once
  per process, not on every import / call.
- Environment variables take precedence over YAML values, which take
  precedence over hardcoded defaults. This follows the standard
  12-factor app configuration pattern.
- Fails loudly and early (at startup) rather than deep inside a
  detection loop, via `ConfigError`.

Typical usage
-------------
    from app.utils.config_loader import get_config

    config = get_config()
    print(config.detection.confidence_threshold)
    print(config.get_zone("zone_a").required_ppe)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

# ------------------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------------------


class ConfigError(Exception):
    """Raised when configuration loading or validation fails."""


# ------------------------------------------------------------------------------
# Typed sub-config dataclasses
# ------------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectionConfig:
    model_path: str
    model_type: str
    confidence_threshold: float
    iou_threshold: float
    device: str
    image_size: int
    half_precision: bool
    classes: Dict[int, str]
    required_ppe: List[str]


@dataclass(frozen=True)
class TrackingConfig:
    algorithm: str
    track_thresh: float
    track_buffer: int
    match_thresh: float
    frame_rate: int
    min_box_area: int


@dataclass(frozen=True)
class ViolationEngineConfig:
    grace_period_frames: int
    violation_cooldown_seconds: int
    risk_levels: Dict[str, str]


@dataclass(frozen=True)
class ZoneConfig:
    id: str
    name: str
    camera_id: str
    required_ppe: List[str]


@dataclass(frozen=True)
class CameraConfig:
    id: str
    name: str
    source: Any  # int (webcam index) or str (file path / RTSP URL)
    zone_id: str
    enabled: bool


@dataclass(frozen=True)
class AlertsConfig:
    save_screenshot: bool
    screenshot_dir: str
    enable_sound: bool
    notify_webhook: Optional[str]


@dataclass(frozen=True)
class ReportsConfig:
    database_path: str
    csv_export_path: str
    json_export_path: str
    retention_days: int


@dataclass(frozen=True)
class APIConfig:
    host: str
    port: int
    reload: bool
    cors_origins: List[str]


@dataclass(frozen=True)
class DashboardConfig:
    host: str
    port: int
    theme: str
    refresh_interval_seconds: int


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    log_dir: str
    rotation: str
    retention: str
    format: str


@dataclass(frozen=True)
class AppConfig:
    """Top-level, immutable application configuration object."""

    project_name: str
    version: str
    environment: str
    detection: DetectionConfig
    tracking: TrackingConfig
    violation_engine: ViolationEngineConfig
    zones: List[ZoneConfig]
    cameras: List[CameraConfig]
    alerts: AlertsConfig
    reports: ReportsConfig
    api: APIConfig
    dashboard: DashboardConfig
    logging: LoggingConfig

    def get_zone(self, zone_id: str) -> ZoneConfig:
        """Look up a zone by its id. Raises ConfigError if not found."""
        for zone in self.zones:
            if zone.id == zone_id:
                return zone
        raise ConfigError(f"Zone '{zone_id}' not found in configuration.")

    def get_camera(self, camera_id: str) -> CameraConfig:
        """Look up a camera by its id. Raises ConfigError if not found."""
        for camera in self.cameras:
            if camera.id == camera_id:
                return camera
        raise ConfigError(f"Camera '{camera_id}' not found in configuration.")

    def enabled_cameras(self) -> List[CameraConfig]:
        """Return only cameras marked as enabled."""
        return [cam for cam in self.cameras if cam.enabled]


# ------------------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------------------

_config_cache: Optional[AppConfig] = None

_REQUIRED_TOP_LEVEL_KEYS = [
    "project",
    "detection",
    "tracking",
    "violation_engine",
    "zones",
    "cameras",
    "alerts",
    "reports",
    "api",
    "dashboard",
    "logging",
]


def _load_yaml(config_path: Path) -> Dict[str, Any]:
    """Read and parse the YAML config file into a raw dict."""
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found at: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML config: {exc}") from exc

    if raw is None:
        raise ConfigError(f"Configuration file is empty: {config_path}")

    return raw


def _validate_raw_config(raw: Dict[str, Any]) -> None:
    """Ensure all required top-level sections exist before building dataclasses."""
    missing = [key for key in _REQUIRED_TOP_LEVEL_KEYS if key not in raw]
    if missing:
        raise ConfigError(
            f"Missing required configuration section(s): {', '.join(missing)}"
        )


def _apply_env_overrides(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply environment variable overrides on top of the parsed YAML.

    Only a curated set of high-value overrides are supported (the ones
    most likely to differ between dev / staging / production), matching
    `.env.example`.
    """
    if os.getenv("MODEL_PATH"):
        raw["detection"]["model_path"] = os.getenv("MODEL_PATH")

    if os.getenv("MODEL_DEVICE"):
        raw["detection"]["device"] = os.getenv("MODEL_DEVICE")

    if os.getenv("API_HOST"):
        raw["api"]["host"] = os.getenv("API_HOST")

    if os.getenv("API_PORT"):
        raw["api"]["port"] = int(os.getenv("API_PORT"))

    if os.getenv("API_ALLOWED_ORIGINS"):
        raw["api"]["cors_origins"] = os.getenv("API_ALLOWED_ORIGINS").split(",")

    if os.getenv("DASHBOARD_HOST"):
        raw["dashboard"]["host"] = os.getenv("DASHBOARD_HOST")

    if os.getenv("DASHBOARD_PORT"):
        raw["dashboard"]["port"] = int(os.getenv("DASHBOARD_PORT"))

    if os.getenv("DATABASE_URL"):
        raw["reports"]["database_path"] = os.getenv("DATABASE_URL")

    if os.getenv("LOG_LEVEL"):
        raw["logging"]["level"] = os.getenv("LOG_LEVEL")

    if os.getenv("LOG_DIR"):
        raw["logging"]["log_dir"] = os.getenv("LOG_DIR")

    if os.getenv("SLACK_WEBHOOK_URL"):
        raw["alerts"]["notify_webhook"] = os.getenv("SLACK_WEBHOOK_URL")

    return raw


def _build_app_config(raw: Dict[str, Any]) -> AppConfig:
    """Convert the validated, override-applied raw dict into an AppConfig."""
    detection_raw = raw["detection"]
    detection = DetectionConfig(
        model_path=detection_raw["model_path"],
        model_type=detection_raw["model_type"],
        confidence_threshold=float(detection_raw["confidence_threshold"]),
        iou_threshold=float(detection_raw["iou_threshold"]),
        device=detection_raw["device"],
        image_size=int(detection_raw["image_size"]),
        half_precision=bool(detection_raw["half_precision"]),
        classes={int(k): v for k, v in detection_raw["classes"].items()},
        required_ppe=list(detection_raw["required_ppe"]),
    )

    tracking_raw = raw["tracking"]
    tracking = TrackingConfig(
        algorithm=tracking_raw["algorithm"],
        track_thresh=float(tracking_raw["track_thresh"]),
        track_buffer=int(tracking_raw["track_buffer"]),
        match_thresh=float(tracking_raw["match_thresh"]),
        frame_rate=int(tracking_raw["frame_rate"]),
        min_box_area=int(tracking_raw["min_box_area"]),
    )

    ve_raw = raw["violation_engine"]
    violation_engine = ViolationEngineConfig(
        grace_period_frames=int(ve_raw["grace_period_frames"]),
        violation_cooldown_seconds=int(ve_raw["violation_cooldown_seconds"]),
        risk_levels=dict(ve_raw["risk_levels"]),
    )

    zones = [
        ZoneConfig(
            id=z["id"],
            name=z["name"],
            camera_id=z["camera_id"],
            required_ppe=list(z["required_ppe"]),
        )
        for z in raw["zones"]
    ]

    cameras = [
        CameraConfig(
            id=c["id"],
            name=c["name"],
            source=c["source"],
            zone_id=c["zone_id"],
            enabled=bool(c["enabled"]),
        )
        for c in raw["cameras"]
    ]

    alerts_raw = raw["alerts"]
    alerts = AlertsConfig(
        save_screenshot=bool(alerts_raw["save_screenshot"]),
        screenshot_dir=alerts_raw["screenshot_dir"],
        enable_sound=bool(alerts_raw["enable_sound"]),
        notify_webhook=alerts_raw.get("notify_webhook"),
    )

    reports_raw = raw["reports"]
    reports = ReportsConfig(
        database_path=reports_raw["database_path"],
        csv_export_path=reports_raw["csv_export_path"],
        json_export_path=reports_raw["json_export_path"],
        retention_days=int(reports_raw["retention_days"]),
    )

    api_raw = raw["api"]
    api = APIConfig(
        host=api_raw["host"],
        port=int(api_raw["port"]),
        reload=bool(api_raw["reload"]),
        cors_origins=list(api_raw["cors_origins"]),
    )

    dashboard_raw = raw["dashboard"]
    dashboard = DashboardConfig(
        host=dashboard_raw["host"],
        port=int(dashboard_raw["port"]),
        theme=dashboard_raw["theme"],
        refresh_interval_seconds=int(dashboard_raw["refresh_interval_seconds"]),
    )

    logging_raw = raw["logging"]
    logging_config = LoggingConfig(
        level=logging_raw["level"],
        log_dir=logging_raw["log_dir"],
        rotation=logging_raw["rotation"],
        retention=logging_raw["retention"],
        format=logging_raw["format"],
    )

    return AppConfig(
        project_name=raw["project"]["name"],
        version=raw["project"]["version"],
        environment=raw["project"]["environment"],
        detection=detection,
        tracking=tracking,
        violation_engine=violation_engine,
        zones=zones,
        cameras=cameras,
        alerts=alerts,
        reports=reports,
        api=api,
        dashboard=dashboard,
        logging=logging_config,
    )


# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------


def get_config(
    config_path: Optional[str] = None, force_reload: bool = False
) -> AppConfig:
    """
    Load (or return the cached) application configuration.

    Parameters
    ----------
    config_path:
        Optional override path to the YAML config file. Defaults to
        the `CONFIG_PATH` env var, or `configs/config.yaml` relative
        to the project root.
    force_reload:
        If True, bypasses the in-memory cache and re-reads the file
        from disk. Useful in tests.

    Returns
    -------
    AppConfig
        Fully validated, typed configuration object.

    Raises
    ------
    ConfigError
        If the file is missing, malformed, or fails validation.
    """
    global _config_cache

    if _config_cache is not None and not force_reload:
        return _config_cache

    # Load .env file (if present) so os.getenv picks up overrides.
    load_dotenv(override=False)

    resolved_path = Path(
        config_path or os.getenv("CONFIG_PATH", "configs/config.yaml")
    )

    raw = _load_yaml(resolved_path)
    _validate_raw_config(raw)
    raw = _apply_env_overrides(raw)

    _config_cache = _build_app_config(raw)
    return _config_cache


def reset_config_cache() -> None:
    """Clear the cached config. Primarily used by the test suite."""
    global _config_cache
    _config_cache = None
