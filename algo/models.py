from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

from music21 import stream

class NoteEvent(NamedTuple):
    """A processed pitch run. A missing pitch represents a rest.

    Timing is in analysis frames; score_builder converts it to musical beats.
    NamedTuple preserves the existing unpacking used by the processing passes.
    """

    pitch: float | None
    frames: int
    confidence: float
    reattack_confidence: float | None
    boundary_source: str
    boundary_confidence: float


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


@dataclass
class PipelineResult:
    estimated_key: str
    estimated_key_candidates: list[tuple[str, float]]
    note_count: int
    score: stream.Stream
    pitch_times_sec: list[float]
    pitch_midi: list[float | None]
    note_confidences: list[dict[str, float | str | int | None]]
    detected_notes: list[DetectedNote] = field(default_factory=list)
