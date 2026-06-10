import numpy as np

from algo.metrics import (
    ReferenceNote,
    evaluate_transcription,
    reference_to_arrays,
)


def test_perfect_match_scores_one():
    notes = [
        ReferenceNote(midi=60, onset_sec=0.0, duration_sec=0.4),
        ReferenceNote(midi=62, onset_sec=0.6, duration_sec=0.4),
    ]
    intervals, pitches = reference_to_arrays(notes)
    metrics = evaluate_transcription(intervals, pitches, intervals, pitches)
    assert metrics.f1 == 1.0
    assert metrics.pitch_accuracy == 1.0
    assert metrics.estimated_note_count == 2


def test_no_estimates_scores_zero():
    ref = [ReferenceNote(midi=60, onset_sec=0.0, duration_sec=0.4)]
    ref_i, ref_p = reference_to_arrays(ref)
    empty_i = np.empty((0, 2))
    empty_p = np.empty(0)
    metrics = evaluate_transcription(empty_i, empty_p, ref_i, ref_p)
    assert metrics.f1 == 0.0
    assert metrics.recall == 0.0


def test_meets_thresholds_reports_failures():
    from algo.metrics import TranscriptionMetrics

    metrics = TranscriptionMetrics(
        precision=0.5,
        recall=0.4,
        f1=0.44,
        onset_precision=0.5,
        onset_recall=0.5,
        onset_f1=0.5,
        pitch_accuracy=0.6,
        avg_overlap_ratio=0.8,
        estimated_note_count=8,
        reference_note_count=8,
    )
    ok, failures = metrics.meets_thresholds({"f1": 0.5, "pitch_accuracy": 0.55})
    assert not ok
    assert any("f1" in f for f in failures)
