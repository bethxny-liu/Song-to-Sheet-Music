"""Benchmark harness: regression baselines and engine comparison on synthetic fixtures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from algo.metrics import ReferenceNote, TranscriptionMetrics, evaluate_pipeline_result
from algo.models import PipelineOptions
from algo.pipeline import AudioToSheetPipeline
from algo.synthetic_audio import BENCHMARK_FIXTURES_DIR, FIXTURE_BUILDERS, synthesize_melody, write_wav

# Synthetic sine tones need lower Basic Pitch thresholds than real piano audio.
GRAND_BP_ONSET_THRESHOLD = 0.3
GRAND_BP_FRAME_THRESHOLD = 0.2


def load_baseline_targets(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class FixtureBenchmark:
    fixture: str
    layout: str
    estimated_key: str
    note_count: int
    transcription_engine: str
    preprocessing: str
    metrics: TranscriptionMetrics
    thresholds: dict[str, Any]
    passed: bool
    failures: list[str]


@dataclass(frozen=True)
class EngineComparisonRow:
    fixture: str
    melody_f1: float
    grand_f1: float
    melody_notes: int
    grand_notes: int
    grand_engine: str
    delta_f1: float


def run_fixture_benchmark(
    pipeline: AudioToSheetPipeline,
    fixture_name: str,
    *,
    layout: str,
    tempo_bpm: int,
    reference: list[ReferenceNote],
    targets: dict[str, Any],
    wav_path: Path,
    isolate_piano: bool = False,
    work_dir: Path | None = None,
    basic_pitch_onset_threshold: float | None = None,
    basic_pitch_frame_threshold: float | None = None,
) -> FixtureBenchmark:
    cfg = targets["fixtures"][fixture_name]
    thresholds = cfg["thresholds"]
    result = pipeline.run(
        wav_path,
        PipelineOptions(
            title=fixture_name,
            composer="Benchmark",
            tempo_bpm=tempo_bpm,
            instrument_name="piano",
            layout=layout,  # type: ignore[arg-type]
            isolate_piano=isolate_piano,
            basic_pitch_onset_threshold=basic_pitch_onset_threshold,
            basic_pitch_frame_threshold=basic_pitch_frame_threshold,
        ),
        work_dir=work_dir,
    )
    metrics = evaluate_pipeline_result(
        result,
        reference,
        tempo_bpm,
        onset_tolerance=targets.get("onset_tolerance_sec", 0.12),
        pitch_tolerance=targets.get("pitch_tolerance_cents", 50.0),
    )
    passed, failures = metrics.meets_thresholds(thresholds)
    if result.note_count < int(thresholds.get("min_detected_notes", 1)):
        passed = False
        failures = list(failures) + [
            f"note_count={result.note_count} < {thresholds.get('min_detected_notes')}"
        ]
    if result.estimated_key != cfg.get("expected_key"):
        passed = False
        failures = list(failures) + [
            f"key={result.estimated_key!r} != {cfg.get('expected_key')!r}"
        ]
    return FixtureBenchmark(
        fixture=fixture_name,
        layout=layout,
        estimated_key=result.estimated_key,
        note_count=result.note_count,
        transcription_engine=result.transcription_engine,
        preprocessing=result.preprocessing,
        metrics=metrics,
        thresholds=thresholds,
        passed=passed,
        failures=failures,
    )


def compare_engines_from_benchmarks(
    melody: FixtureBenchmark,
    grand: FixtureBenchmark,
) -> EngineComparisonRow:
    return EngineComparisonRow(
        fixture=melody.fixture,
        melody_f1=melody.metrics.f1,
        grand_f1=grand.metrics.f1,
        melody_notes=melody.note_count,
        grand_notes=grand.note_count,
        grand_engine=grand.transcription_engine,
        delta_f1=grand.metrics.f1 - melody.metrics.f1,
    )


def run_fixture_suite(
    pipeline: AudioToSheetPipeline,
    fixture_name: str,
    *,
    reference: list[ReferenceNote],
    targets: dict[str, Any],
    wav_path: Path,
    tempo_bpm: int,
    configured_layout: str,
) -> tuple[FixtureBenchmark, EngineComparisonRow]:
    """Run melody + grand once each; reuse for regression and comparison."""
    melody = run_fixture_benchmark(
        pipeline,
        fixture_name,
        layout="melody",
        tempo_bpm=tempo_bpm,
        reference=reference,
        targets=targets,
        wav_path=wav_path,
    )
    grand = run_fixture_benchmark(
        pipeline,
        fixture_name,
        layout="grand",
        tempo_bpm=tempo_bpm,
        reference=reference,
        targets=targets,
        wav_path=wav_path,
        basic_pitch_onset_threshold=GRAND_BP_ONSET_THRESHOLD,
        basic_pitch_frame_threshold=GRAND_BP_FRAME_THRESHOLD,
    )
    configured = melody if configured_layout == "melody" else grand
    return configured, compare_engines_from_benchmarks(melody, grand)


def run_full_evaluation(
    *,
    fixtures_dir: Path | None = None,
    output_dir: Path,
) -> dict[str, Any]:
    fixtures_dir = fixtures_dir or BENCHMARK_FIXTURES_DIR
    targets = load_baseline_targets(fixtures_dir / "baseline_targets.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = AudioToSheetPipeline()

    benchmarks: list[FixtureBenchmark] = []
    comparisons: list[EngineComparisonRow] = []

    for fixture_name, cfg in targets["fixtures"].items():
        reference = FIXTURE_BUILDERS[fixture_name]()
        signal, sr = synthesize_melody(reference)
        wav_path = write_wav(output_dir / f"{fixture_name}.wav", signal, sr)
        tempo_bpm = int(cfg["tempo_bpm"])
        configured_layout = str(cfg["layout"])

        benchmark, comparison = run_fixture_suite(
            pipeline,
            fixture_name,
            reference=reference,
            targets=targets,
            wav_path=wav_path,
            tempo_bpm=tempo_bpm,
            configured_layout=configured_layout,
        )
        benchmarks.append(benchmark)
        comparisons.append(comparison)

    all_passed = all(b.passed for b in benchmarks)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "all_passed": all_passed,
        "benchmarks": [_benchmark_to_dict(b) for b in benchmarks],
        "engine_comparison": [asdict(row) for row in comparisons],
    }
    return report


def _benchmark_to_dict(benchmark: FixtureBenchmark) -> dict[str, Any]:
    return {
        "fixture": benchmark.fixture,
        "layout": benchmark.layout,
        "estimated_key": benchmark.estimated_key,
        "note_count": benchmark.note_count,
        "transcription_engine": benchmark.transcription_engine,
        "preprocessing": benchmark.preprocessing,
        "metrics": benchmark.metrics.to_dict(),
        "thresholds": benchmark.thresholds,
        "passed": benchmark.passed,
        "failures": benchmark.failures,
    }


def format_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Transcription Evaluation Report",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        f"**Overall:** {'PASS' if report['all_passed'] else 'FAIL'}",
        "",
        "## Regression benchmarks",
        "",
        "| Fixture | Layout | Engine | F1 | Pitch acc. | Notes | Status |",
        "|---------|--------|--------|-----|------------|-------|--------|",
    ]
    for row in report["benchmarks"]:
        metrics = row["metrics"]
        status = "pass" if row["passed"] else "fail"
        lines.append(
            f"| {row['fixture']} | {row['layout']} | {row['transcription_engine']} "
            f"| {metrics['f1']:.3f} | {metrics['pitch_accuracy']:.3f} "
            f"| {row['note_count']} | {status} |"
        )

    lines.extend(
        [
            "",
            "## Engine comparison (melody vs grand)",
            "",
            "| Fixture | Melody F1 | Grand F1 | Δ F1 | Grand engine |",
            "|---------|-----------|----------|------|--------------|",
        ]
    )
    for row in report["engine_comparison"]:
        lines.append(
            f"| {row['fixture']} | {row['melody_f1']:.3f} | {row['grand_f1']:.3f} "
            f"| {row['delta_f1']:+.3f} | {row['grand_engine']} |"
        )

    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "- **F1**: note overlap precision/recall (mir_eval)",
            "- **Pitch accuracy**: cents tolerance on onset-matched notes",
            "- Thresholds: `tests/fixtures/baseline_targets.json`",
            "",
        ]
    )
    return "\n".join(lines)


def assert_benchmark_passes(benchmark: FixtureBenchmark) -> None:
    if benchmark.passed:
        return
    metrics = benchmark.metrics
    report = (
        f"\n{benchmark.fixture} baseline metrics:\n"
        f"  note F1={metrics.f1:.3f}  precision={metrics.precision:.3f}  "
        f"recall={metrics.recall:.3f}\n"
        f"  pitch_accuracy={metrics.pitch_accuracy:.3f}  onset_f1={metrics.onset_f1:.3f}\n"
        f"  detected={metrics.estimated_note_count}  reference={metrics.reference_note_count}\n"
        f"  key={benchmark.estimated_key}"
    )
    raise AssertionError(report + "\n  failures: " + ", ".join(benchmark.failures))


def write_evaluation_report(report: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "evaluation_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return json_path


def format_evaluation_summary(report: dict[str, Any]) -> str:
    lines = ["Evaluation summary", "=================="]
    for row in report["benchmarks"]:
        status = "PASS" if row["passed"] else "FAIL"
        metrics = row["metrics"]
        lines.append(
            f"  {row['fixture']} ({row['layout']}): F1={metrics['f1']:.3f} [{status}]"
        )
    lines.append("")
    lines.append("Engine comparison:")
    for row in report["engine_comparison"]:
        lines.append(
            f"  {row['fixture']}: melody={row['melody_f1']:.3f} "
            f"grand={row['grand_f1']:.3f} (Δ {row['delta_f1']:+.3f})"
        )
    lines.append("")
    lines.append(f"Overall: {'PASS' if report['all_passed'] else 'FAIL'}")
    return "\n".join(lines)


def run_evaluation_main(project_root: Path) -> int:
    out_dir = project_root / "reports" / "evaluation"
    report = run_full_evaluation(fixtures_dir=BENCHMARK_FIXTURES_DIR, output_dir=out_dir)
    json_path = write_evaluation_report(report, out_dir)
    docs_md = project_root / "docs" / "EVALUATION.md"
    docs_md.parent.mkdir(exist_ok=True)
    docs_md.write_text(format_markdown_report(report), encoding="utf-8")

    print(format_evaluation_summary(report))
    print(f"\nWrote {json_path}")
    print(f"Wrote {docs_md}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(run_evaluation_main(Path(__file__).resolve().parents[1]))
