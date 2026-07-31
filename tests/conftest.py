"""
conftest.py
===========
Shared pytest fixtures for the GuardianVision AI test suite.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

from app.utils.config_loader import reset_config_cache


@pytest.fixture(autouse=True)
def _reset_config_between_tests() -> Generator[None, None, None]:
    """Ensure each test gets a fresh config load, avoiding cross-test state leaks."""
    reset_config_cache()
    yield
    reset_config_cache()


@pytest.fixture
def sample_frame() -> np.ndarray:
    """A blank 640x480 BGR frame, useful for smoke-testing detector/tracker calls."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def temp_db_path() -> Generator[str, None, None]:
    """A temporary SQLite file path, cleaned up after the test."""
    tmp_dir = tempfile.mkdtemp()
    db_path = str(Path(tmp_dir) / "test_guardian_vision.db")
    yield db_path
    shutil.rmtree(tmp_dir, ignore_errors=True)
