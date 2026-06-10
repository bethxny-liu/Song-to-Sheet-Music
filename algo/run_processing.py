from __future__ import annotations

from music21 import note

from algo.models import NoteEvent


def compute_reattack_confidence(has_attack: bool, left_frames: int, right_frames: int) -> float:
    base = 0.0 if not has_attack else 0.65
    duration_support = min(0.35, min(left_frames, right_frames) / 80.0)
    return float(max(0.0, min(1.0, base + duration_support)))


def compress_pitch_track(
    segments: list[tuple[float | None, int, bool, float, str, float]]
) -> list[NoteEvent]:
    runs: list[tuple[float | None, int, bool, float, float | None, str, float]] = []
    for pitch, frames, has_attack, confidence, boundary_source, boundary_confidence in segments:
        if not runs:
            runs.append((pitch, frames, has_attack, confidence, None, boundary_source, boundary_confidence))
            continue
        (
            prev_pitch,
            prev_frames,
            prev_attack,
            prev_confidence,
            prev_reattack,
            prev_source,
            prev_boundary_conf,
        ) = runs[-1]
        if (pitch is None and prev_pitch is None) or (
            pitch is not None and prev_pitch is not None and abs(pitch - prev_pitch) < 0.5
        ):
            if has_attack and frames >= 24 and prev_frames >= 24:
                reattack_conf = compute_reattack_confidence(
                    has_attack=has_attack, left_frames=prev_frames, right_frames=frames
                )
                runs.append(
                    (
                        pitch,
                        frames,
                        has_attack,
                        confidence,
                        reattack_conf,
                        boundary_source,
                        boundary_confidence,
                    )
                )
            else:
                total_frames = prev_frames + frames
                weighted_conf = ((prev_confidence * prev_frames) + (confidence * frames)) / max(
                    1, total_frames
                )
                runs[-1] = (
                    prev_pitch,
                    total_frames,
                    prev_attack,
                    weighted_conf,
                    prev_reattack,
                    prev_source,
                    max(prev_boundary_conf, boundary_confidence),
                )
        else:
            runs.append((pitch, frames, has_attack, confidence, None, boundary_source, boundary_confidence))

    compressed = [
        (pitch, frames, conf, reattack, source, bconf)
        for pitch, frames, _, conf, reattack, source, bconf in runs
    ]
    return merge_short_runs(compressed)


def merge_short_runs(
    runs: list[NoteEvent], min_frames: int = 3, min_rest_frames: int = 8
) -> list[NoteEvent]:
    if not runs:
        return runs
    merged: list[NoteEvent] = []
    for pitch, frames, confidence, reattack, boundary_source, boundary_confidence in runs:
        if pitch is None and frames < min_rest_frames and merged and merged[-1][0] is not None:
            prev_pitch, prev_frames, prev_conf, prev_reattack, prev_source, prev_bconf = merged[-1]
            total = prev_frames + frames
            merged[-1] = (
                prev_pitch,
                total,
                ((prev_conf * prev_frames) + (confidence * frames)) / max(1, total),
                prev_reattack,
                prev_source,
                max(prev_bconf, boundary_confidence),
            )
            continue
        if merged and frames < min_frames and pitch is not None and merged[-1][0] is not None:
            prev_pitch, prev_frames, prev_conf, prev_reattack, prev_source, prev_bconf = merged[-1]
            if abs(float(pitch) - float(prev_pitch)) <= 1.0:
                total = prev_frames + frames
                merged[-1] = (
                    prev_pitch,
                    total,
                    ((prev_conf * prev_frames) + (confidence * frames)) / max(1, total),
                    prev_reattack,
                    prev_source,
                    max(prev_bconf, boundary_confidence),
                )
                continue
        merged.append((pitch, frames, confidence, reattack, boundary_source, boundary_confidence))
    return merged


def trim_leading_rests(runs: list[NoteEvent]) -> list[NoteEvent]:
    start_idx = 0
    while start_idx < len(runs) and runs[start_idx][0] is None:
        start_idx += 1
    return runs[start_idx:]


def merge_tiny_same_pitch_fragments(
    runs: list[NoteEvent], tiny_fragment_max_frames: int
) -> list[NoteEvent]:
    if not runs:
        return runs
    merged: list[NoteEvent] = [runs[0]]
    for pitch, frames, confidence, reattack, bsource, bconf in runs[1:]:
        prev_pitch, prev_frames, prev_confidence, prev_reattack, prev_bsource, prev_bconf = merged[-1]
        same_pitch = (
            pitch is not None
            and prev_pitch is not None
            and abs(float(pitch) - float(prev_pitch)) < 0.5
        )
        if same_pitch and frames <= tiny_fragment_max_frames:
            total = prev_frames + frames
            merged[-1] = (
                prev_pitch,
                total,
                ((prev_confidence * prev_frames) + (confidence * frames)) / max(1, total),
                prev_reattack if prev_reattack is not None else reattack,
                prev_bsource,
                max(prev_bconf, bconf),
            )
        else:
            merged.append((pitch, frames, confidence, reattack, bsource, bconf))
    return merged


def bridge_same_pitch_across_tiny_rests(
    runs: list[NoteEvent], tiny_rest_max_frames: int
) -> list[NoteEvent]:
    if len(runs) < 3:
        return runs
    bridged: list[NoteEvent] = []
    i = 0
    while i < len(runs):
        if i + 2 < len(runs):
            p1, f1, c1, r1, s1, b1 = runs[i]
            p2, f2, c2, r2, s2, b2 = runs[i + 1]
            p3, f3, c3, r3, s3, b3 = runs[i + 2]
            same_note = p1 is not None and p3 is not None and abs(float(p1) - float(p3)) < 0.5
            if same_note and p2 is None and f2 <= tiny_rest_max_frames:
                total = f1 + f2 + f3
                weighted_conf = ((c1 * f1) + (c2 * f2) + (c3 * f3)) / max(1, total)
                bridged.append((p1, total, weighted_conf, r1 if r1 is not None else r3, s1, max(b1, b2, b3)))
                i += 3
                continue
        bridged.append(runs[i])
        i += 1
    return bridged


def merge_low_confidence_boundary_splits(
    runs: list[NoteEvent],
    weak_boundary_threshold: float,
    short_note_max_frames: int,
    force_merge_boundary_threshold: float = 0.15,
) -> list[NoteEvent]:
    if not runs:
        return runs
    merged: list[NoteEvent] = [runs[0]]
    for pitch, frames, conf, reattack, source, bconf in runs[1:]:
        prev_pitch, prev_frames, prev_conf, prev_reattack, prev_source, prev_bconf = merged[-1]
        same_pitch = False
        if pitch is not None and prev_pitch is not None:
            same_pitch = (
                abs(float(pitch) - float(prev_pitch)) < 0.5
                or int(round(float(pitch))) == int(round(float(prev_pitch)))
            )
        weak_split = bconf < weak_boundary_threshold and frames <= short_note_max_frames
        very_weak_boundary = bconf < force_merge_boundary_threshold
        if same_pitch and (weak_split or very_weak_boundary):
            total = prev_frames + frames
            merged[-1] = (
                prev_pitch,
                total,
                ((prev_conf * prev_frames) + (conf * frames)) / max(1, total),
                prev_reattack,
                prev_source,
                max(prev_bconf, bconf),
            )
        else:
            merged.append((pitch, frames, conf, reattack, source, bconf))
    return merged


def final_merge_weak_same_pitch_events(
    runs: list[NoteEvent], weak_boundary_threshold: float
) -> list[NoteEvent]:
    if not runs:
        return runs
    merged: list[NoteEvent] = [runs[0]]
    for pitch, frames, conf, reattack, source, bconf in runs[1:]:
        prev_pitch, prev_frames, prev_conf, prev_reattack, prev_source, prev_bconf = merged[-1]
        if pitch is None or prev_pitch is None:
            merged.append((pitch, frames, conf, reattack, source, bconf))
            continue
        same_midi = int(round(float(pitch))) == int(round(float(prev_pitch)))
        weak_boundary = bconf < weak_boundary_threshold
        no_reattack = reattack is None
        if same_midi and weak_boundary and no_reattack:
            total = prev_frames + frames
            merged[-1] = (
                prev_pitch,
                total,
                ((prev_conf * prev_frames) + (conf * frames)) / max(1, total),
                prev_reattack,
                prev_source,
                max(prev_bconf, bconf),
            )
        else:
            merged.append((pitch, frames, conf, reattack, source, bconf))
    return merged


def major_scale_pitch_classes(tonic: str) -> set[int]:
    tonic_pc = note.Note(tonic).pitch.pitchClass
    return {(tonic_pc + step) % 12 for step in (0, 2, 4, 5, 7, 9, 11)}


def nearest_in_scale_midi(midi_int: int, scale_pcs: set[int]) -> int:
    if midi_int % 12 in scale_pcs:
        return midi_int
    for distance in range(1, 4):
        down = midi_int - distance
        up = midi_int + distance
        if down % 12 in scale_pcs:
            return down
        if up % 12 in scale_pcs:
            return up
    return midi_int


def stabilize_out_of_scale_notes(
    runs: list[NoteEvent], tonic: str, min_duration_frames: int
) -> list[NoteEvent]:
    scale_pcs = major_scale_pitch_classes(tonic)
    stabilized: list[NoteEvent] = []
    for pitch, frames, confidence, reattack, bsource, bconf in runs:
        if pitch is None:
            stabilized.append((pitch, frames, confidence, reattack, bsource, bconf))
            continue
        midi_int = int(round(pitch))
        if midi_int % 12 in scale_pcs or frames >= min_duration_frames:
            stabilized.append((pitch, frames, confidence, reattack, bsource, bconf))
            continue
        snapped = nearest_in_scale_midi(midi_int, scale_pcs)
        stabilized.append((float(snapped), frames, confidence, reattack, bsource, bconf))
    return stabilized
