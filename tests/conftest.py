from __future__ import annotations

import json
from pathlib import Path

import pytest

from algo.metrics import ReferenceNote
from algo.synthetic_audio import (
    build_c_major_scale_reference,
    build_repeated_c_reference,
    synthesize_melody,
    write_wav,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
BASELINE_TARGETS_PATH = FIXTURES_DIR / "baseline_targets.json"


@pytest.fixture
def baseline_targets() -> dict:
    return json.loads(BASELINE_TARGETS_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def tmp_wav_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def c_major_scale_reference() -> list[ReferenceNote]:
    return build_c_major_scale_reference()


@pytest.fixture
def c_major_scale_wav(tmp_wav_dir: Path, c_major_scale_reference: list[ReferenceNote]) -> Path:
    signal, sr = synthesize_melody(c_major_scale_reference)
    return write_wav(tmp_wav_dir / "c_major_scale.wav", signal, sr)


@pytest.fixture
def repeated_c_reference() -> list[ReferenceNote]:
    return build_repeated_c_reference()


@pytest.fixture
def repeated_c_wav(tmp_wav_dir: Path, repeated_c_reference: list[ReferenceNote]) -> Path:
    signal, sr = synthesize_melody(repeated_c_reference, gap_sec=0.2)
    return write_wav(tmp_wav_dir / "repeated_c.wav", signal, sr)
