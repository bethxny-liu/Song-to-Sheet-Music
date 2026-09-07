from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from music21 import stream

NoteEvent = tuple[float | None, int, float, float | None, str, float]


@dataclass(frozen=True)
class DetectedNote:
    midi: int
    onset_sec: float
    duration_sec: float
    confidence: float = 1.0


@dataclass
class PipelineOptions:
    title: str
    composer: str
    tempo_bpm: int
    instrument_name: str
    layout: Literal["melody", "grand"] = "melody"
    isolate_piano: bool = False
    auto_detect_tempo: bool = False
    basic_pitch_onset_threshold: float | None = None
    basic_pitch_frame_threshold: float | None = None


@dataclass
class PipelineResult:
    estimated_key: str
    estimated_key_candidates: list[tuple[str, float]]
    note_count: int
    score: stream.Stream
    pitch_times_sec: list[float]
    pitch_midi: list[float | None]
    note_confidences: list[dict[str, float | str | int | None]]
    chord_events: list[dict[str, float | str]]
    transcription_engine: str = "pyin"
    preprocessing: str = "none"
    detected_notes: list[DetectedNote] = field(default_factory=list)
