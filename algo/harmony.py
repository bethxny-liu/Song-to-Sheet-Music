from __future__ import annotations

import librosa
import numpy as np

PC_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def detect_chord_events(
    signal: np.ndarray,
    sample_rate: int,
    hop_length: int,
) -> list[dict[str, float | str]]:
    chroma = librosa.feature.chroma_cqt(y=signal, sr=sample_rate, hop_length=hop_length)
    _, beat_frames = librosa.beat.beat_track(y=signal, sr=sample_rate, hop_length=hop_length)
    boundaries = _build_boundaries(chroma.shape[1], beat_frames)

    events: list[dict[str, float | str]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= start:
            continue
        segment = chroma[:, start:end]
        chord_name, confidence = _best_chord(segment)
        onset_sec = float(librosa.frames_to_time(start, sr=sample_rate, hop_length=hop_length))
        end_sec = float(librosa.frames_to_time(end, sr=sample_rate, hop_length=hop_length))
        events.append(
            {
                "onset_sec": onset_sec,
                "duration_sec": max(0.05, end_sec - onset_sec),
                "chord": chord_name,
                "confidence": confidence,
            }
        )
    return _merge_adjacent_same_chords(events)


def _build_boundaries(frame_count: int, beat_frames: np.ndarray) -> list[int]:
    boundaries = {0, frame_count}
    for frame in np.asarray(beat_frames).tolist():
        frame_int = int(frame)
        if 0 < frame_int < frame_count:
            boundaries.add(frame_int)
    return sorted(boundaries)


def _best_chord(segment_chroma: np.ndarray) -> tuple[str, float]:
    chroma_mean = np.mean(segment_chroma, axis=1)
    total = float(np.sum(chroma_mean)) + 1e-8
    chroma_norm = chroma_mean / total

    best_name = "N.C."
    best_score = -1.0
    second_best = -1.0

    for root in range(12):
        major_template = np.zeros(12)
        major_template[root] = 1.0
        major_template[(root + 4) % 12] = 0.9
        major_template[(root + 7) % 12] = 0.85
        major_score = float(np.dot(chroma_norm, major_template))
        if major_score > best_score:
            second_best = best_score
            best_score = major_score
            best_name = f"{PC_NAMES[root]}"
        elif major_score > second_best:
            second_best = major_score

        minor_template = np.zeros(12)
        minor_template[root] = 1.0
        minor_template[(root + 3) % 12] = 0.9
        minor_template[(root + 7) % 12] = 0.85
        minor_score = float(np.dot(chroma_norm, minor_template))
        if minor_score > best_score:
            second_best = best_score
            best_score = minor_score
            best_name = f"{PC_NAMES[root]}m"
        elif minor_score > second_best:
            second_best = minor_score

    confidence = max(0.0, min(1.0, best_score - max(0.0, second_best)))
    return best_name, confidence


def _merge_adjacent_same_chords(
    events: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    if not events:
        return events
    merged = [events[0].copy()]
    for event in events[1:]:
        prev = merged[-1]
        if event["chord"] == prev["chord"]:
            prev["duration_sec"] = float(prev["duration_sec"]) + float(event["duration_sec"])
            prev["confidence"] = max(float(prev["confidence"]), float(event["confidence"]))
        else:
            merged.append(event.copy())
    return merged
