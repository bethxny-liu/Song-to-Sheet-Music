from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from algo.source_separation import is_available, isolate_piano


def test_is_available_false_when_demucs_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "demucs":
            raise ImportError("no demucs")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    monkeypatch.setattr("algo.source_separation.shutil.which", lambda _: None)
    assert is_available() is False


@patch("algo.source_separation.is_available", return_value=True)
@patch("algo.source_separation.subprocess.run")
def test_isolate_piano_returns_stem_path(mock_run: MagicMock, _mock_available: MagicMock, tmp_path: Path):
    input_wav = tmp_path / "song.wav"
    input_wav.write_bytes(b"wav")
    stem_dir = tmp_path / "_demucs" / "htdemucs" / "song"
    stem_dir.mkdir(parents=True)
    piano_wav = stem_dir / "piano.wav"
    piano_wav.write_bytes(b"stem")

    result = isolate_piano(input_wav, tmp_path / "_demucs")
    assert result == piano_wav
    mock_run.assert_called_once()
    assert "--two-stems" in mock_run.call_args.args[0]
    assert "piano" in mock_run.call_args.args[0]


@patch("algo.source_separation.is_available", return_value=False)
def test_isolate_piano_raises_when_unavailable(_mock_available: MagicMock, tmp_path: Path):
    with pytest.raises(RuntimeError, match="Demucs is not available"):
        isolate_piano(tmp_path / "x.wav", tmp_path)
