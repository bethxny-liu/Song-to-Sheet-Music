"""Decode recordings and reject inputs that cannot be transcribed."""

from pathlib import Path

import librosa
import numpy as np


class AudioInputError(ValueError):
    """A recording the user needs to replace or shorten."""


def load_audio(
    path: Path,
    max_duration_sec: float | None = None,
    *,
    trim_leading_silence: bool = False,
) -> tuple[np.ndarray, int]:
    try:
        # Decode at most one second beyond the limit to detect oversized recordings.
        signal, sample_rate = librosa.load(
            str(path), sr=None, mono=True,
            duration=None if max_duration_sec is None else max_duration_sec + 1,
        )
    except Exception as exc:
        raise AudioInputError(
            "Could not decode this recording. Try exporting it as WAV or MP3."
        ) from exc

    if max_duration_sec is not None and len(signal) > sample_rate * max_duration_sec:
        raise AudioInputError(f"Recording is too long. Upload at most {max_duration_sec:g} seconds.")
    if len(signal) < sample_rate * 0.1:
        raise AudioInputError("Recording is empty or too short. Upload at least 0.1 seconds of audio.")
    if not np.isfinite(signal).all():
        raise AudioInputError("Recording contains invalid audio samples. Try exporting it again.")
    if not np.any(signal):
        raise AudioInputError("Recording is silent. Upload a recording with audible notes.")
    if sample_rate < 8000:
        raise AudioInputError("Recording sample rate is too low. Use at least 8,000 Hz.")
    if trim_leading_silence:
        _, (start, _end) = librosa.effects.trim(
            signal, top_db=40, frame_length=2048, hop_length=512
        )
        # Preserve short musical rests; remove only an obvious recording lead-in.
        if start > sample_rate:
            signal = signal[start:]
    return signal, sample_rate
