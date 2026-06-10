from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

ANNOTATIONS_FPS = 22050 // 256  # basic_pitch.constants


@dataclass(frozen=True)
class TimedNote:
    onset_sec: float
    duration_sec: float
    midi: int
    confidence: float


def is_available() -> bool:
    try:
        return resolve_model_path() is not None
    except Exception:
        return False


def resolve_model_path():
    """Pick the best Basic Pitch runtime available on this machine."""
    from basic_pitch import (
        CT_PRESENT,
        FilenameSuffix,
        ONNX_PRESENT,
        TF_PRESENT,
        TFLITE_PRESENT,
        build_icassp_2022_model_path,
    )

    for suffix, present in (
        (FilenameSuffix.onnx, ONNX_PRESENT),
        (FilenameSuffix.coreml, CT_PRESENT),
        (FilenameSuffix.tflite, TFLITE_PRESENT),
        (FilenameSuffix.tf, TF_PRESENT),
    ):
        if present:
            return build_icassp_2022_model_path(suffix)
    return None


def transcribe(
    audio_path: Path,
    *,
    melody_only: bool = False,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length_ms: float = 80.0,
) -> list[TimedNote]:
    """Run Spotify Basic Pitch and return note events with absolute timing."""
    import scipy.signal
    from basic_pitch.inference import predict

    # basic-pitch 0.3.x expects scipy.signal.gaussian (removed in scipy 1.14+).
    if not hasattr(scipy.signal, "gaussian"):
        from scipy.signal.windows import gaussian as _gaussian

        scipy.signal.gaussian = _gaussian  # type: ignore[attr-defined]

    model_path = resolve_model_path()
    if model_path is None:
        raise RuntimeError(
            "Basic Pitch is installed but no inference backend is available. "
            "Install onnxruntime (recommended) or tensorflow."
        )

    _, _, note_events = predict(
        str(audio_path),
        model_path,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length_ms,
        melodia_trick=melody_only,
    )

    notes: list[TimedNote] = []
    for start, end, pitch, amplitude, _pitch_bends in note_events:
        duration = max(0.05, float(end) - float(start))
        notes.append(
            TimedNote(
                onset_sec=float(start),
                duration_sec=duration,
                midi=int(pitch),
                confidence=float(max(0.0, min(1.0, amplitude))),
            )
        )
    return sorted(notes, key=lambda n: (n.onset_sec, n.midi))


def timed_notes_to_pitch_track(
    notes: list[TimedNote],
    total_duration_sec: float,
    fps: float = float(ANNOTATIONS_FPS),
) -> tuple[list[float], list[float | None]]:
    """Rasterize timed notes into a pitch chart (highest simultaneous pitch per frame)."""
    if total_duration_sec <= 0:
        return [], []
    frame_count = max(1, int(total_duration_sec * fps) + 1)
    times = [i / fps for i in range(frame_count)]
    midi_track: list[float | None] = [None] * frame_count
    for note in notes:
        start_frame = max(0, int(note.onset_sec * fps))
        end_frame = min(frame_count, int((note.onset_sec + note.duration_sec) * fps) + 1)
        for frame in range(start_frame, end_frame):
            current = midi_track[frame]
            if current is None or note.midi > current:
                midi_track[frame] = float(note.midi)
    return times, midi_track


def timed_notes_to_runs(
    notes: list[TimedNote], frame_duration: float
) -> list[tuple[float | None, int, float, float | None, str, float]]:
    """Convert timed notes into frame-based runs for key estimation."""
    from algo.models import NoteEvent

    runs: list[NoteEvent] = []
    for note in notes:
        frames = max(1, int(round(note.duration_sec / max(frame_duration, 1e-6))))
        runs.append(
            (
                float(note.midi),
                frames,
                note.confidence,
                None,
                "basic_pitch",
                note.confidence,
            )
        )
    return runs
