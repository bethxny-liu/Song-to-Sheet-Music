from music21 import instrument
from pydantic import BaseModel, Field, field_validator


class ConversionOptions(BaseModel):
    title: str = Field(default="Untitled", max_length=200)
    composer: str = Field(default="Unknown", max_length=200)
    tempo_bpm: int = Field(default=90, ge=40, le=240)
    instrument_name: str = Field(default="piano", min_length=1, max_length=80)

    @field_validator("instrument_name")
    @classmethod
    def validate_instrument(cls, value: str) -> str:
        value = value.strip()
        try:
            instrument.fromString(value)
        except instrument.InstrumentException as exc:
            raise ValueError("Unknown instrument. Try piano, violin, flute, or guitar.") from exc
        return value


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


class ConversionResult(BaseModel):
    job_id: str
    title: str
    composer: str
    tempo_bpm: int
    instrument_name: str
    estimated_key: str
    estimated_key_candidates: list[KeyEstimate]
    note_confidences: list[NoteConfidence]
    note_count: int
    artifacts: dict[str, str]
