"""Transcription evaluation metrics (mir_eval-based) for baseline regression tests."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import librosa
import mir_eval
import numpy as np

from algo.basic_pitch_transcriber import TimedNote
from algo.models import PipelineResult


@dataclass(frozen=True)
class ReferenceNote:
    midi: int
    onset_sec: float
    duration_sec: float

    @property
    def offset_sec(self) -> float:
        return self.onset_sec + self.duration_sec


@dataclass(frozen=True)
class TranscriptionMetrics:
    """Note-level overlap metrics (standard MIR transcription evaluation)."""

    precision: float
    recall: float
    f1: float
    onset_precision: float
    onset_recall: float
    onset_f1: float
    pitch_accuracy: float
    avg_overlap_ratio: float
    estimated_note_count: int
    reference_note_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def meets_thresholds(self, thresholds: dict[str, Any]) -> tuple[bool, list[str]]:
        failures: list[str] = []
        for key, minimum in thresholds.items():
            if key == "key_match" or key == "min_detected_notes":
                continue
            if key not in self.to_dict():
                continue
            value = float(self.to_dict()[key])
            if value < float(minimum):
                failures.append(f"{key}={value:.3f} < {minimum}")
        return len(failures) == 0, failures


def reference_to_arrays(
    notes: list[ReferenceNote],
) -> tuple[np.ndarray, np.ndarray]:
    if not notes:
        return np.empty((0, 2), dtype=float), np.empty(0, dtype=float)
    intervals = np.array([[n.onset_sec, n.offset_sec] for n in notes], dtype=float)
    pitches = np.array([n.midi for n in notes], dtype=float)
    return intervals, pitches


def note_confidences_to_arrays(
    events: list[dict[str, Any]], tempo_bpm: int
) -> tuple[np.ndarray, np.ndarray]:
    intervals: list[list[float]] = []
    pitches: list[float] = []
    for event in events:
        if event.get("type") != "note" or event.get("midi") is None:
            continue
        onset_q = float(event["onset_quarter"])
        dur_q = float(event["duration_quarter"])
        onset_sec = (onset_q * 60.0) / tempo_bpm
        offset_sec = ((onset_q + dur_q) * 60.0) / tempo_bpm
        intervals.append([onset_sec, offset_sec])
        pitches.append(float(event["midi"]))
    if not intervals:
        return np.empty((0, 2), dtype=float), np.empty(0, dtype=float)
    return np.array(intervals, dtype=float), np.array(pitches, dtype=float)


def timed_notes_to_arrays(notes: list[TimedNote]) -> tuple[np.ndarray, np.ndarray]:
    if not notes:
        return np.empty((0, 2), dtype=float), np.empty(0, dtype=float)
    intervals = np.array([[n.onset_sec, n.onset_sec + n.duration_sec] for n in notes], dtype=float)
    pitches = np.array([n.midi for n in notes], dtype=float)
    return intervals, pitches


def evaluate_transcription(
    estimated_intervals: np.ndarray,
    estimated_pitches: np.ndarray,
    reference_intervals: np.ndarray,
    reference_pitches: np.ndarray,
    *,
    onset_tolerance: float = 0.05,
    pitch_tolerance: float = 50.0,
) -> TranscriptionMetrics:
    """Compare estimated vs reference note sequences using mir_eval."""
    est_count = len(estimated_pitches)
    ref_count = len(reference_pitches)

    if ref_count == 0 and est_count == 0:
        return TranscriptionMetrics(
            precision=1.0,
            recall=1.0,
            f1=1.0,
            onset_precision=1.0,
            onset_recall=1.0,
            onset_f1=1.0,
            pitch_accuracy=1.0,
            avg_overlap_ratio=1.0,
            estimated_note_count=0,
            reference_note_count=0,
        )

    if ref_count == 0 or est_count == 0:
        return TranscriptionMetrics(
            precision=0.0,
            recall=0.0,
            f1=0.0,
            onset_precision=0.0,
            onset_recall=0.0,
            onset_f1=0.0,
            pitch_accuracy=0.0,
            avg_overlap_ratio=0.0,
            estimated_note_count=est_count,
            reference_note_count=ref_count,
        )

    ref_hz = librosa.midi_to_hz(reference_pitches.astype(float))
    est_hz = librosa.midi_to_hz(estimated_pitches.astype(float))
    precision, recall, f1, avg_overlap = mir_eval.transcription.precision_recall_f1_overlap(
        reference_intervals,
        ref_hz,
        estimated_intervals,
        est_hz,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=pitch_tolerance,
        offset_ratio=None,
    )
    onset_precision, onset_recall, onset_f1 = mir_eval.transcription.onset_precision_recall_f1(
        reference_intervals,
        estimated_intervals,
        onset_tolerance=onset_tolerance,
    )
    pitch_acc = _matched_pitch_accuracy(
        reference_intervals,
        reference_pitches,
        estimated_intervals,
        estimated_pitches,
        onset_tolerance=onset_tolerance,
        pitch_tolerance_cents=pitch_tolerance,
    )

    return TranscriptionMetrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        onset_precision=float(onset_precision),
        onset_recall=float(onset_recall),
        onset_f1=float(onset_f1),
        pitch_accuracy=float(pitch_acc),
        avg_overlap_ratio=float(avg_overlap),
        estimated_note_count=est_count,
        reference_note_count=ref_count,
    )


def evaluate_pipeline_result(
    result: PipelineResult,
    reference: list[ReferenceNote],
    tempo_bpm: int,
    *,
    onset_tolerance: float = 0.05,
    pitch_tolerance: float = 50.0,
) -> TranscriptionMetrics:
    ref_intervals, ref_pitches = reference_to_arrays(reference)
    est_intervals, est_pitches = note_confidences_to_arrays(result.note_confidences, tempo_bpm)
    return evaluate_transcription(
        est_intervals,
        est_pitches,
        ref_intervals,
        ref_pitches,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=pitch_tolerance,
    )


def _matched_pitch_accuracy(
    reference_intervals: np.ndarray,
    reference_pitches: np.ndarray,
    estimated_intervals: np.ndarray,
    estimated_pitches: np.ndarray,
    *,
    onset_tolerance: float,
    pitch_tolerance_cents: float,
) -> float:
    """Fraction of onset-matched notes within pitch_tolerance_cents of reference."""
    ref_onsets = reference_intervals[:, 0]
    est_onsets = estimated_intervals[:, 0]
    matching = mir_eval.util.match_events(ref_onsets, est_onsets, onset_tolerance)
    if not matching:
        return 0.0
    correct = 0
    for ref_idx, est_idx in matching:
        ref_hz = librosa.midi_to_hz(reference_pitches[ref_idx])
        est_hz = librosa.midi_to_hz(estimated_pitches[est_idx])
        if ref_hz <= 0 or est_hz <= 0:
            continue
        cents = abs(1200.0 * np.log2(est_hz / ref_hz))
        if cents <= pitch_tolerance_cents:
            correct += 1
    return float(correct) / float(len(matching))
