"""Exercise multipart requests through FastAPI without a network server."""

import asyncio
import json
from types import SimpleNamespace

from fastapi import FastAPI
import pytest

from algo.audio import AudioInputError
from services.app import routes
from services.app.conversion import ConversionService


def upload(service, fields=None, content=b"audio", filename="recording.wav"):
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[routes.get_conversion_service] = lambda: service
    boundary = "test-upload-boundary"
    body = b""
    for name, value in (fields or {}).items():
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
    body += f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
    body += content + f"\r\n--{boundary}--\r\n".encode()
    messages = []

    async def request():
        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message):
            messages.append(message)

        await app({
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": "POST", "scheme": "http", "path": "/jobs/convert",
            "raw_path": b"/jobs/convert", "query_string": b"", "root_path": "",
            "server": ("testserver", 80), "client": ("127.0.0.1", 1234),
            "headers": [(b"host", b"testserver"), (b"content-type", f"multipart/form-data; boundary={boundary}".encode())],
        }, receive, send)

    asyncio.run(request())
    status = next(m["status"] for m in messages if m["type"] == "http.response.start")
    payload = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    return status, json.loads(payload)


@pytest.mark.parametrize("fields", [
    {"tempo_bpm": 0}, {"tempo_bpm": 241}, {"tempo_bpm": "fast"},
    {"instrument_name": "made-up-instrument"}, {"instrument_name": "   "},
])
def test_invalid_options_never_start_conversion(fields):
    status, payload = upload(SimpleNamespace(), fields)
    assert status == 422
    assert payload["detail"]


def test_empty_upload():
    status, payload = upload(SimpleNamespace(), content=b"")
    assert status == 400
    assert "empty" in payload["detail"]


def test_audio_mime_type_does_not_bypass_decoder_validation(tmp_path):
    service = ConversionService(tmp_path / "storage")
    status, payload = upload(service, content=b"not actually WAV audio")
    assert status == 422
    assert "decode" in payload["detail"]
    assert not list(service.storage_dir.iterdir())


def test_oversized_upload_is_removed(tmp_path, monkeypatch):
    monkeypatch.setattr(routes, "MAX_UPLOAD_BYTES", 4)
    original = routes.NamedTemporaryFile
    monkeypatch.setattr(routes, "NamedTemporaryFile", lambda **kwargs: original(dir=tmp_path, **kwargs))
    status, _ = upload(SimpleNamespace(), content=b"too large")
    assert status == 413
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("error, status, expected", [
    (AudioInputError("Recording is silent."), 422, "Recording is silent."),
    (RuntimeError("private server details"), 500, "Conversion failed."),
])
def test_conversion_errors_have_useful_responses_and_clean_uploads(error, status, expected):
    paths = []

    def convert(path, options, base_url):
        assert path.exists()
        paths.append(path)
        raise error

    actual_status, payload = upload(SimpleNamespace(convert=convert))
    assert actual_status == status
    assert expected in payload["detail"]
    assert "private server details" not in payload["detail"]
    assert paths and not paths[0].exists()


def test_valid_options_reach_conversion_and_success_cleans_upload():
    paths = []

    def convert(path, options, base_url):
        paths.append(path)
        assert path.read_bytes() == b"audio"
        assert options.tempo_bpm == 120
        assert options.instrument_name == "violin"
        return {
            "job_id": "test", "title": options.title, "composer": options.composer,
            "tempo_bpm": options.tempo_bpm, "instrument_name": options.instrument_name,
            "estimated_key": "C major", "estimated_key_candidates": [],
            "note_confidences": [], "note_count": 1, "artifacts": {},
        }

    status, payload = upload(SimpleNamespace(convert=convert), {"tempo_bpm": 120, "instrument_name": "violin"})
    assert status == 200
    assert payload["job_id"] == "test"
    assert not paths[0].exists()
