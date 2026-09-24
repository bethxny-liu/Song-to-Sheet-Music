"""Optional Basic Pitch adapter for the polyphonic transcription path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TimedNote:
    onset_sec: float
    duration_sec: float
    midi: int
    confidence: float


def is_available() -> bool:
    """Return whether Basic Pitch and one of its inference runtimes are usable."""
    try:
        return resolve_model_path() is not None
    except Exception:
        return False


def resolve_model_path():
    """Select the first Basic Pitch runtime available on this machine."""
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
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length_ms: float = 80.0,
) -> list[TimedNote]:
    """Run Basic Pitch and return overlapping note events for experiments."""
    import scipy.signal
    from basic_pitch.inference import predict

    # Basic Pitch 0.3.x expects scipy.signal.gaussian, removed in newer SciPy.
    if not hasattr(scipy.signal, "gaussian"):
        from scipy.signal.windows import gaussian

        scipy.signal.gaussian = gaussian  # type: ignore[attr-defined]

    model_path = resolve_model_path()
    if model_path is None:
        raise RuntimeError("Basic Pitch has no available inference runtime.")

    _, _, note_events = predict(
        str(audio_path),
        model_path,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length_ms,
        # Keep simultaneous notes; this is the part pYIN cannot represent.
        melodia_trick=False,
    )
    return sorted(
        (
            TimedNote(
                onset_sec=float(start),
                duration_sec=max(0.05, float(end) - float(start)),
                midi=int(pitch),
                confidence=float(max(0.0, min(1.0, amplitude))),
            )
            for start, end, pitch, amplitude, _pitch_bends in note_events
        ),
        key=lambda note: (note.onset_sec, note.midi),
    )
