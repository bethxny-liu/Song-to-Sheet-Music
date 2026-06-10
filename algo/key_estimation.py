from __future__ import annotations

import numpy as np

from algo.models import NoteEvent


def estimate_key_from_pitches(runs: list[NoteEvent]) -> tuple[str, list[tuple[str, float]]]:
    if not runs:
        return "C major", [("C major", 1.0)]

    pc_weights = np.zeros(12, dtype=float)
    for pitch, frames, _, _, _, _ in runs:
        if pitch is None:
            continue
        midi_value = int(round(pitch))
        pc_weights[midi_value % 12] += frames

    if np.sum(pc_weights) == 0:
        return "C major", [("C major", 1.0)]

    tonic_pitch_class = {
        "C": 0,
        "C#": 1,
        "D": 2,
        "Eb": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "G": 7,
        "Ab": 8,
        "A": 9,
        "Bb": 10,
        "B": 11,
    }
    key_signature_complexity = {
        "C": 0,
        "G": 1,
        "D": 2,
        "A": 3,
        "E": 4,
        "B": 5,
        "F#": 6,
        "C#": 7,
        "F": 1,
        "Bb": 2,
        "Eb": 3,
        "Ab": 4,
    }
    first_pitch = next((int(round(p)) % 12 for p, _, _, _, _, _ in runs if p is not None), None)
    last_pitch = next(
        (int(round(p)) % 12 for p, _, _, _, _, _ in reversed(runs) if p is not None), None
    )

    best_key = "C"
    best_score = -1.0
    scored_keys: list[tuple[str, float]] = []
    for key_name, tonic_pc in tonic_pitch_class.items():
        scale_pcs = {(tonic_pc + step) % 12 for step in (0, 2, 4, 5, 7, 9, 11)}
        in_scale = float(sum(pc_weights[pc] for pc in scale_pcs))
        out_of_scale = float(np.sum(pc_weights) - in_scale)
        tonic_bonus = pc_weights[tonic_pc] * 0.30
        boundary_bonus = 0.0
        if first_pitch is not None and first_pitch == tonic_pc:
            boundary_bonus += 0.04 * np.sum(pc_weights)
        if last_pitch is not None and last_pitch == tonic_pc:
            boundary_bonus += 0.08 * np.sum(pc_weights)
        complexity_penalty = key_signature_complexity.get(key_name, 0) * 0.02 * np.sum(pc_weights)
        out_of_scale_penalty = out_of_scale * 1.25
        score = in_scale + tonic_bonus + boundary_bonus - complexity_penalty - out_of_scale_penalty
        scored_keys.append((f"{key_name} major", float(score)))
        if score > best_score:
            best_key = key_name
            best_score = score

    ranked = sorted(scored_keys, key=lambda item: item[1], reverse=True)
    top_three = ranked[:3]
    score_sum = sum(max(item[1], 0.0) for item in top_three)
    if score_sum <= 0:
        normalized = [(name, 0.0) for name, _ in top_three]
    else:
        normalized = [(name, max(score, 0.0) / score_sum) for name, score in top_three]
    return f"{best_key} major", normalized
