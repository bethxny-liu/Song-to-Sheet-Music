"""Fallback pitch trackers used when pYIN is weak or polyphonic heuristics are needed."""

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


def polyphonic_top_voice_track(
    signal: np.ndarray, sample_rate: int, hop_length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate melody from polyphonic audio with continuity bias (legacy fallback)."""
    harmonic, _ = librosa.effects.hpss(signal)
    pitches, magnitudes = librosa.piptrack(
        y=harmonic,
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
    global_floor = max_mag * 0.04
    prev_midi: float | None = None
    for i in range(frame_count):
        col_pitch = pitches[:, i]
        col_mag = magnitudes[:, i]
        valid_idx = np.where((col_pitch > 0.0) & (col_mag > global_floor))[0]
        if valid_idx.size == 0:
            prev_midi = None
            continue

        candidate_freq = col_pitch[valid_idx]
        candidate_mag = col_mag[valid_idx]
        candidate_midi = librosa.hz_to_midi(candidate_freq)

        in_range = (candidate_midi >= 60.0) & (candidate_midi <= 88.0)
        if np.any(in_range):
            candidate_freq = candidate_freq[in_range]
            candidate_mag = candidate_mag[in_range]
            candidate_midi = candidate_midi[in_range]

        mag_norm = candidate_mag / (float(np.max(candidate_mag)) + 1e-8)
        midi_norm = (candidate_midi - 48.0) / 40.0
        if prev_midi is None:
            jump_penalty = np.zeros_like(candidate_midi)
        else:
            jump_penalty = np.minimum(np.abs(candidate_midi - prev_midi) / 12.0, 1.0)
        scores = (0.70 * mag_norm) + (0.25 * midi_norm) - (0.35 * jump_penalty)
        best_idx = int(np.argmax(scores))
        peak_freq = float(candidate_freq[best_idx])
        peak_mag = float(candidate_mag[best_idx])
        if peak_freq <= 0.0 or peak_mag <= 0.0:
            prev_midi = None
            continue
        f0_hz[i] = peak_freq
        voiced_flag[i] = True
        voiced_prob[i] = min(1.0, peak_mag / (max_mag + 1e-8))
        prev_midi = float(librosa.hz_to_midi(peak_freq))
    return f0_hz, voiced_flag, voiced_prob


def bridge_short_unvoiced_gaps(f0_hz: np.ndarray, max_gap_frames: int = 3) -> np.ndarray:
    """Fill tiny NaN gaps in f0 to avoid fragmented rests."""
    bridged = f0_hz.copy()
    i = 0
    n = len(bridged)
    while i < n:
        if not np.isnan(bridged[i]):
            i += 1
            continue
        start = i
        while i < n and np.isnan(bridged[i]):
            i += 1
        end = i
        gap = end - start
        left = start - 1
        right = end
        if (
            gap <= max_gap_frames
            and left >= 0
            and right < n
            and not np.isnan(bridged[left])
            and not np.isnan(bridged[right])
        ):
            if abs(librosa.hz_to_midi(bridged[left]) - librosa.hz_to_midi(bridged[right])) <= 2.0:
                bridged[start:end] = (bridged[left] + bridged[right]) / 2.0
    return bridged
