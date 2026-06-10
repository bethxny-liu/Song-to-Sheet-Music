from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from music21 import stream

NoteEvent = tuple[float | None, int, float, float | None, str, float]


@dataclass
class PipelineOptions:
    title: str
    composer: str
    tempo_bpm: int
    instrument_name: str
    """melody: single treble staff, monophonic path only (simple melodies / tutorials).
    grand: piano grand staff; may use polyphonic top-voice when detection is weak."""
    layout: Literal["melody", "grand"] = "melody"


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
