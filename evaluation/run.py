"""Small regression runner for the supported monophonic melody path."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from algo.models import PipelineOptions
from algo.pipeline import AudioToSheetPipeline
from evaluation.metrics import ReferenceNote, TranscriptionMetrics, evaluate_pipeline_result
from evaluation.synthetic_audio import BENCHMARK_FIXTURES_DIR, FIXTURE_BUILDERS, synthesize_benchmark_audio, write_wav

CONTROLLED_TEMPOS = (60, 90, 120, 150, 180)


@dataclass(frozen=True)
class FixtureBenchmark:
    fixture: str
    estimated_key: str
    note_count: int
    metrics: TranscriptionMetrics
    thresholds: dict[str, Any]
    passed: bool
    failures: list[str]


def load_baseline_targets(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_raw_pyin_baseline(signal: np.ndarray, sample_rate: int, reference: list[ReferenceNote]) -> TranscriptionMetrics:
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as directory:
        wav = write_wav(Path(directory) / "benchmark.wav", signal, sample_rate)
        result = AudioToSheetPipeline().run(wav, PipelineOptions("Benchmark", "", 120, "piano"))
    return evaluate_pipeline_result(result, reference, 120)


def run_fixture_benchmark(pipeline: AudioToSheetPipeline, fixture: str, *, tempo_bpm: int, reference: list[ReferenceNote], targets: dict[str, Any], wav_path: Path) -> FixtureBenchmark:
    result = pipeline.run(wav_path, PipelineOptions(fixture, "", tempo_bpm, "piano"))
    metrics = evaluate_pipeline_result(result, reference, tempo_bpm, onset_tolerance=0.15)
    thresholds = dict(targets.get("fixtures", {}).get(fixture, {}).get("thresholds", {}))
    failures = [f"{name}={getattr(metrics, name):.3f} < {minimum}" for name, minimum in thresholds.items() if hasattr(metrics, name) and getattr(metrics, name) < minimum]
    return FixtureBenchmark(fixture, result.estimated_key, result.note_count, metrics, thresholds, not failures, failures)


def assert_benchmark_passes(benchmark: FixtureBenchmark) -> None:
    assert benchmark.passed, "; ".join(benchmark.failures)


def run_full_evaluation(*, fixtures_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = load_baseline_targets(fixtures_dir / "baseline_targets.json")
    benchmarks: list[dict[str, Any]] = []
    for fixture, config in targets.get("fixtures", {}).items():
        builder = FIXTURE_BUILDERS[fixture]
        reference = builder()
        signal, sample_rate = synthesize_benchmark_audio(reference)
        wav = write_wav(output_dir / f"{fixture}.wav", signal, sample_rate)
        benchmark = run_fixture_benchmark(AudioToSheetPipeline(), fixture, tempo_bpm=int(config["tempo_bpm"]), reference=reference, targets=targets, wav_path=wav)
        benchmarks.append({"fixture": fixture, "estimated_key": benchmark.estimated_key, "note_count": benchmark.note_count, "passed": benchmark.passed, "metrics": benchmark.metrics.to_dict()})
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "all_passed": all(row["passed"] for row in benchmarks), "benchmarks": benchmarks}


def write_evaluation_report(report: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "evaluation.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def format_markdown_report(report: dict[str, Any]) -> str:
    lines = ["# Transcription Evaluation Report", "", "| Fixture | Notes | F1 | Status |", "|---|---:|---:|---|"]
    for row in report.get("benchmarks", []):
        lines.append(f"| {row['fixture']} | {row['note_count']} | {row['metrics']['f1']:.3f} | {'PASS' if row['passed'] else 'FAIL'} |")
    return "\n".join(lines) + "\n"


def format_evaluation_summary(report: dict[str, Any]) -> str:
    return "\n".join([f"{row['fixture']}: {'PASS' if row['passed'] else 'FAIL'}" for row in report.get("benchmarks", [])])


def run_evaluation_main(project_root: Path) -> int:
    report = run_full_evaluation(fixtures_dir=project_root / "tests" / "fixtures", output_dir=project_root / "reports" / "evaluation")
    write_evaluation_report(report, project_root / "reports" / "evaluation")
    (project_root / "docs" / "EVALUATION.md").write_text(format_markdown_report(report), encoding="utf-8")
    print(format_evaluation_summary(report))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(run_evaluation_main(Path(__file__).resolve().parents[1]))
