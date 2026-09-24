"""Fallback pitch tracking used when pYIN finds too few voiced frames."""

from __future__ import annotations

import librosa
import numpy as np


def fallback_f0_from_piptrack(
    signal: np.ndarray, sample_rate: int, hop_length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monophonic melody from spectral peaks when pyin finds almost no voiced frames."""
    pitches, magnitudes = librosa.piptrack(
        y=signal,
        sr=sample_rate,
        hop_length=hop_length,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
    )
    frame_count = pitches.shape[1]
    f0_hz = np.full(frame_count, np.nan, dtype=float)
    voiced_flag = np.zeros(frame_count, dtype=bool)
    voiced_prob = np.zeros(frame_count, dtype=float)

    max_mag = float(np.max(magnitudes)) if magnitudes.size else 0.0
    mag_floor = max_mag * 0.08
    for i in range(frame_count):
        col_mag = magnitudes[:, i]
        idx = int(np.argmax(col_mag))
        peak_mag = float(col_mag[idx])
        peak_freq = float(pitches[idx, i])
        if peak_mag <= mag_floor or peak_freq <= 0.0:
            continue
        f0_hz[i] = peak_freq
        voiced_flag[i] = True
        voiced_prob[i] = min(1.0, peak_mag / (max_mag + 1e-8))
    return f0_hz, voiced_flag, voiced_prob
