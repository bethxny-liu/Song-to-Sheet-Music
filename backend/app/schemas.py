from typing import Literal

from pydantic import BaseModel


class ConversionOptions(BaseModel):
    title: str = "Untitled"
    composer: str = "Unknown"
    tempo_bpm: int = 90
    instrument_name: str = "piano"
    layout: Literal["melody", "grand"] = "melody"


class KeyEstimate(BaseModel):
    key: str
    score: float


class NoteConfidence(BaseModel):
    onset_quarter: float
    duration_quarter: float
    type: str
    midi: int | None
    confidence: float
    reattack_confidence: float | None
    boundary_source: str
    boundary_confidence: float
    hand: str | None = None


class ChordEvent(BaseModel):
    onset_sec: float
    duration_sec: float
    chord: str
    confidence: float


class ConversionResult(BaseModel):
    job_id: str
    title: str
    composer: str
    tempo_bpm: int
    instrument_name: str
    estimated_key: str
    estimated_key_candidates: list[KeyEstimate]
    note_confidences: list[NoteConfidence]
    chord_events: list[ChordEvent]
    note_count: int
    artifacts: dict[str, str]
