"""
logger.py
=========
Centralized logging configuration for GuardianVision AI, built on Loguru.

All modules should import `get_logger(__name__)` rather than configuring
their own handlers, so log format, rotation, and destination stay
consistent across the detector, tracker, API, and dashboard processes.

Usage
-----
    from app.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Detector initialized")
    logger.warning("Frame drop detected on cam_01")
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger as _loguru_logger

from app.utils.config_loader import get_config

_is_configured = False


def _configure_logging() -> None:
    """Set up Loguru sinks (console + rotating file) exactly once per process."""
    global _is_configured
    if _is_configured:
        return

    config = get_config()
    log_cfg = config.logging

    log_dir = Path(log_cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    _loguru_logger.remove()  # remove Loguru's default stderr handler

    # Console sink — human-readable, colorized
    _loguru_logger.add(
        sys.stdout,
        level=log_cfg.level,
        format=log_cfg.format,
        colorize=True,
        backtrace=False,
        diagnose=False,
    )

    # File sink — rotating, retained per config
    _loguru_logger.add(
        log_dir / "guardianvision.log",
        level=log_cfg.level,
        format=log_cfg.format,
        rotation=log_cfg.rotation,
        retention=log_cfg.retention,
        backtrace=True,
        diagnose=False,
        enqueue=True,  # process-safe writes
    )

    _is_configured = True


def get_logger(name: Optional[str] = None):
    """
    Return a Loguru logger bound with the calling module's name.

    Parameters
    ----------
    name:
        Typically `__name__` of the calling module, used to tag log lines.

    Returns
    -------
    Logger
        A Loguru logger instance bound with `module=name`.
    """
    _configure_logging()
    return _loguru_logger.bind(module=name or "guardianvision")
