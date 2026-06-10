from __future__ import annotations

import librosa
import numpy as np


def normalize_tempo(detected_tempo: float | np.ndarray, fallback_bpm: int) -> int:
    if isinstance(detected_tempo, np.ndarray):
        if detected_tempo.size == 0:
            return fallback_bpm
        value = float(detected_tempo.reshape(-1)[0])
    else:
        value = float(detected_tempo)
    if np.isnan(value) or value < 30 or value > 240:
        return fallback_bpm
    return int(round(value))


def sparsify_frames(frames: np.ndarray, min_gap_frames: int) -> np.ndarray:
    if len(frames) == 0:
        return frames
    kept = [int(frames[0])]
    for frame in frames[1:]:
        frame_int = int(frame)
        if frame_int - kept[-1] >= min_gap_frames:
            kept.append(frame_int)
    return np.asarray(kept, dtype=int)


def hz_to_midi_track(f0_hz: np.ndarray, voiced_flag: np.ndarray) -> np.ndarray:
    midi_track = np.full(len(f0_hz), np.nan, dtype=float)
    for i, hz in enumerate(f0_hz):
        if not voiced_flag[i] or np.isnan(hz):
            continue
        midi_track[i] = librosa.hz_to_midi(hz)
    return midi_track


def median_smooth_midi(midi_track: np.ndarray, window: int = 5) -> np.ndarray:
    if len(midi_track) == 0 or window < 3:
        return midi_track
    half = window // 2
    smoothed = midi_track.copy()
    for i in range(len(midi_track)):
        start = max(0, i - half)
        end = min(len(midi_track), i + half + 1)
        vals = midi_track[start:end]
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            smoothed[i] = float(np.median(vals))
    return smoothed


def smooth_octave_errors(midi_track: np.ndarray) -> np.ndarray:
    if len(midi_track) == 0:
        return midi_track
    corrected = midi_track.copy()
    for i in range(1, len(corrected) - 1):
        if np.isnan(corrected[i]):
            continue
        prev_val = corrected[i - 1]
        next_val = corrected[i + 1]
        if np.isnan(prev_val) or np.isnan(next_val):
            continue
        if abs(prev_val - next_val) < 2 and abs(corrected[i] - prev_val) > 8:
            if abs((corrected[i] - 12) - prev_val) < 2:
                corrected[i] -= 12
            elif abs((corrected[i] + 12) - prev_val) < 2:
                corrected[i] += 12
    return corrected


def frames_with_pitch_discontinuity(
    midi_track: np.ndarray,
    half_window: int = 4,
    min_jump_semitones: float = 3.0,
    merge_within_frames: int = 8,
) -> list[int]:
    """Frames where local median MIDI jumps — catches melodic leaps before the first onset.

    Onset detection often fires late for the downbeat; the opening phrase can live entirely
    inside [0, first_onset) and would otherwise get one blended pitch from segment_pitches.
    """
    n = len(midi_track)
    if n < half_window * 2 + 1:
        return []
    candidates: list[int] = []
    for i in range(half_window, n - half_window):
        left = midi_track[i - half_window : i]
        right = midi_track[i : i + half_window]
        lv = left[~np.isnan(left)]
        rv = right[~np.isnan(right)]
        if len(lv) < 2 or len(rv) < 2:
            continue
        ml = float(np.median(lv))
        mr = float(np.median(rv))
        if abs(ml - mr) >= min_jump_semitones:
            candidates.append(i)
    if not candidates:
        return []
    merged = [candidates[0]]
    for j in candidates[1:]:
        if j - merged[-1] >= merge_within_frames:
            merged.append(j)
    return merged


def build_segment_boundaries(
    frame_count: int,
    onset_frames: np.ndarray,
    attack_frames: np.ndarray,
    midi_track: np.ndarray | None = None,
) -> list[int]:
    boundaries = {0, frame_count}
    for frame in np.asarray(onset_frames).tolist():
        if 0 < int(frame) < frame_count:
            boundaries.add(int(frame))
    for frame in np.asarray(attack_frames).tolist():
        if 0 < int(frame) < frame_count:
            boundaries.add(int(frame))
    if midi_track is not None and len(midi_track) == frame_count:
        first_voiced = None
        for i in range(frame_count):
            if not np.isnan(midi_track[i]):
                first_voiced = i
                break
        if first_voiced is not None and 0 < first_voiced < frame_count:
            boundaries.add(int(first_voiced))
        for frame in frames_with_pitch_discontinuity(midi_track):
            if 0 < int(frame) < frame_count:
                boundaries.add(int(frame))
    return sorted(boundaries)


def segment_confidence(
    voiced_ratio: float,
    voiced_probability: float,
    segment_rms: float,
    energy_threshold: float,
) -> float:
    energy_score = 1.0
    if energy_threshold > 0:
        energy_score = min(1.0, segment_rms / (energy_threshold * 2.0))
    confidence = (0.45 * voiced_probability) + (0.35 * voiced_ratio) + (0.20 * energy_score)
    return float(max(0.0, min(1.0, confidence)))


def boundary_info(
    boundary_frame: int,
    onset_frame_set: set[int],
    attack_frame_set: set[int],
    onset_env: np.ndarray,
    rms_delta: np.ndarray,
) -> tuple[str, float]:
    has_onset = boundary_frame in onset_frame_set
    has_attack = boundary_frame in attack_frame_set
    if has_onset and has_attack:
        source = "onset+attack"
    elif has_onset:
        source = "onset"
    elif has_attack:
        source = "attack"
    else:
        source = "other"
    env_value = float(onset_env[min(boundary_frame, len(onset_env) - 1)]) if len(onset_env) else 0.0
    delta_value = float(rms_delta[min(boundary_frame, len(rms_delta) - 1)]) if len(rms_delta) else 0.0
    env_norm = 0.0 if len(onset_env) == 0 else env_value / (float(np.max(onset_env)) + 1e-8)
    delta_norm = 0.0 if len(rms_delta) == 0 else max(0.0, delta_value) / (float(np.max(rms_delta)) + 1e-8)
    source_bonus = 0.15 if source == "onset+attack" else (0.08 if source in {"onset", "attack"} else 0.0)
    confidence = min(1.0, (0.45 * env_norm) + (0.40 * delta_norm) + source_bonus)
    return source, float(max(0.0, confidence))


def segment_pitches(
    midi_track: np.ndarray,
    boundaries: list[int],
    rms: np.ndarray,
    rms_delta: np.ndarray,
    energy_threshold: float,
    voiced_prob: np.ndarray,
    attack_frame_set: set[int],
    onset_frame_set: set[int],
    onset_env: np.ndarray,
    min_voiced_ratio: float = 0.20,
    min_voiced_prob: float = 0.45,
    require_energy_threshold: bool = True,
) -> list[tuple[float | None, int, bool, float, str, float]]:
    segments: list[tuple[float | None, int, bool, float, str, float]] = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if end <= start:
            continue
        window = midi_track[start:end]
        voiced = window[~np.isnan(window)]
        rms_end = min(end, len(rms))
        rms_start = min(start, rms_end)
        segment_rms = float(np.mean(rms[rms_start:rms_end])) if rms_end > rms_start else 0.0
        prob_end = min(end, len(voiced_prob))
        prob_start = min(start, prob_end)
        segment_voiced_prob = (
            float(np.mean(voiced_prob[prob_start:prob_end])) if prob_end > prob_start else 0.0
        )
        voiced_ratio = len(voiced) / max(1, len(window))
        confidence = segment_confidence(
            voiced_ratio=voiced_ratio,
            voiced_probability=segment_voiced_prob,
            segment_rms=segment_rms,
            energy_threshold=energy_threshold,
        )
        b_source, b_conf = boundary_info(
            start, onset_frame_set, attack_frame_set, onset_env, rms_delta
        )
        is_last_segment = i == (len(boundaries) - 2)
        terminal_voiced_tail = is_last_segment and len(voiced) > 0 and voiced_ratio >= 0.05
        if (
            len(voiced) == 0
            or voiced_ratio < min_voiced_ratio
            or segment_voiced_prob < min_voiced_prob
            or (require_energy_threshold and segment_rms < energy_threshold)
        ) and not terminal_voiced_tail:
            segments.append((None, end - start, start in attack_frame_set, confidence, b_source, b_conf))
            continue
        pitch = float(np.median(voiced))
        segments.append((pitch, end - start, start in attack_frame_set, confidence, b_source, b_conf))
    return segments
