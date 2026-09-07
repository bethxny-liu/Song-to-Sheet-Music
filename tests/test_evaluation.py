from __future__ import annotations

from algo.evaluation import (
    compare_engines_from_benchmarks,
    evaluate_raw_pyin_baseline,
    format_markdown_report,
)
from algo.metrics import TranscriptionMetrics
from algo.synthetic_audio import build_tempo_reference, synthesize_benchmark_audio


def _benchmark(fixture: str, layout: str, f1: float, engine: str, passed: bool = True):
    metrics = TranscriptionMetrics(
        precision=f1,
        recall=f1,
        f1=f1,
        onset_precision=f1,
        onset_recall=f1,
        onset_f1=f1,
        pitch_accuracy=1.0,
        avg_overlap_ratio=f1,
        estimated_note_count=8,
        reference_note_count=8,
    )
    from algo.evaluation import FixtureBenchmark

    return FixtureBenchmark(
        fixture=fixture,
        layout=layout,
        estimated_key="C major",
        note_count=8,
        transcription_engine=engine,
        preprocessing="none",
        metrics=metrics,
        thresholds={},
        passed=passed,
        failures=[],
    )


def test_compare_engines_from_benchmarks():
    melody = _benchmark("c_major_scale", "melody", 0.625, "pyin")
    grand = _benchmark("c_major_scale", "grand", 0.125, "basic_pitch")
    row = compare_engines_from_benchmarks(melody, grand)
    assert row.melody_f1 == 0.625
    assert row.grand_f1 == 0.125
    assert row.delta_f1 == -0.5


def test_format_markdown_report_includes_tables():
    report = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "all_passed": True,
        "benchmarks": [
            {
                "fixture": "c_major_scale",
                "layout": "melody",
                "transcription_engine": "pyin",
                "note_count": 8,
                "passed": True,
                "metrics": {"f1": 0.625, "pitch_accuracy": 1.0},
            }
        ],
        "engine_comparison": [
            {
                "fixture": "c_major_scale",
                "melody_f1": 0.625,
                "grand_f1": 0.7,
                "delta_f1": 0.075,
                "grand_engine": "basic_pitch",
            }
        ],
        "controlled_benchmarks": [
            {
                "texture": "monophonic",
                "tempo_bpm": 60,
                "baseline_f1": 0.70,
                "engine": "pyin",
                "metrics": {
                    "precision": 0.90,
                    "recall": 0.80,
                    "f1": 0.85,
                    "onset_f1": 0.88,
                },
            },
            {
                "texture": "monophonic",
                "tempo_bpm": 180,
                "baseline_f1": 0.60,
                "engine": "pyin",
                "metrics": {
                    "precision": 0.75,
                    "recall": 0.70,
                    "f1": 0.72,
                    "onset_f1": 0.74,
                },
            },
            {
                "texture": "chords",
                "tempo_bpm": 90,
                "baseline_f1": None,
                "engine": "basic_pitch",
                "metrics": {
                    "precision": 0.50,
                    "recall": 0.40,
                    "f1": 0.44,
                    "onset_f1": 0.48,
                },
            },
        ],
    }
    md = format_markdown_report(report)
    assert "# Transcription Evaluation Report" in md
    assert "c_major_scale" in md
    assert "Engine comparison" in md
    assert "Controlled tempo benchmark" in md
    assert "Polyphony benchmark" in md
    assert "pYIN baseline F1" in md
    assert "60 BPM" in md
    assert "180 BPM" in md
    assert "chords F1" in md
    assert "PASS" in md


def test_raw_pyin_baseline_scores_synthetic_monophonic_phrase():
    reference = build_tempo_reference(120, "monophonic")
    signal, sample_rate = synthesize_benchmark_audio(reference)
    metrics = evaluate_raw_pyin_baseline(signal, sample_rate, reference)
    assert metrics.reference_note_count == 8
    assert metrics.estimated_note_count > 0
    assert 0.0 <= metrics.f1 <= 1.0


def test_controlled_references_separate_polyphony_levels():
    assert len(build_tempo_reference(120, "monophonic")) == 8
    assert len(build_tempo_reference(120, "two_note_harmony")) == 16
    assert len(build_tempo_reference(120, "chords")) == 32
