"""Exercise application startup and cleanup around failed conversions."""

import asyncio
import os
from pathlib import Path
import subprocess
import sys

import pytest

from services.app.conversion import ConversionService
from services.app.schemas import ConversionOptions


def test_fresh_application_creates_storage_and_serves_artifacts(tmp_path):
    storage = tmp_path / "fresh" / "storage"
    script = """
import asyncio
from services.app.main import app, healthcheck
from services.app.config import STORAGE_DIR
assert STORAGE_DIR.is_dir()
assert healthcheck() == {"status": "ok"}
artifact = STORAGE_DIR / "test.txt"
artifact.write_text("artifact")
mount = next(route for route in app.routes if route.path == "/artifacts")
response = asyncio.run(mount.app.get_response("test.txt", {"method": "GET", "headers": []}))
assert response.status_code == 200
assert response.path == str(artifact)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "STORAGE_DIR": str(storage)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_failed_conversion_removes_partial_artifacts(tmp_path, monkeypatch):
    service = ConversionService(tmp_path / "storage")

    def fail_after_writing(*args, **kwargs):
        raise RuntimeError("Transcription failed")

    monkeypatch.setattr(service.pipeline, "run", fail_after_writing)
    with pytest.raises(RuntimeError, match="Transcription failed"):
        service.convert(tmp_path / "input.wav", ConversionOptions(), "http://testserver")
    assert list(service.storage_dir.iterdir()) == []
