from __future__ import annotations

from algo.evaluation import compare_engines_from_benchmarks, format_markdown_report
from algo.metrics import TranscriptionMetrics


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
    }
    md = format_markdown_report(report)
    assert "# Transcription Evaluation Report" in md
    assert "c_major_scale" in md
    assert "Engine comparison" in md
    assert "PASS" in md
