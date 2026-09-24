"""End-to-end baseline metrics on synthetic audio fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.run import assert_benchmark_passes, run_fixture_benchmark
from evaluation.metrics import ReferenceNote
from algo.pipeline import AudioToSheetPipeline


@pytest.mark.parametrize("sample_rate", [22050, 44100])
def test_short_chromatic_note_is_not_rewritten_to_fit_key(tmp_path, sample_rate):
    from algo.models import PipelineOptions
    from evaluation.metrics import evaluate_pipeline_result
    from evaluation.synthetic_audio import synthesize_benchmark_audio, write_wav

    # F-sharp is a short passing tone in a phrase dominated by C-major notes.
    reference = []
    onset = 0.2
    for midi in [60, 64, 67, 66, 67, 64, 62, 60]:
        duration = 0.09 if midi == 66 else 0.4
        reference.append(ReferenceNote(midi, onset, duration))
        onset += duration + 0.12
    signal, sr = synthesize_benchmark_audio(reference, sample_rate=sample_rate)
    wav = write_wav(tmp_path / "chromatic.wav", signal, sr)
    result = AudioToSheetPipeline().run(wav, PipelineOptions("Chromatic", "Test", 120, "piano"))

    assert [n.midi for n in result.detected_notes] == [n.midi for n in reference]
    assert 66 in [n.pitch.midi for n in result.score.recurse().notes if n.isNote]
    assert evaluate_pipeline_result(result, reference, 120, onset_tolerance=0.12).f1 == 1.0


def test_melody_pipeline_returns_no_chord_data(tmp_path):
    from algo.models import PipelineOptions
    from evaluation.synthetic_audio import synthesize_melody, write_wav

    reference = [ReferenceNote(64, 0.2, 0.4), ReferenceNote(62, 0.8, 0.4)]
    signal, sr = synthesize_melody(reference)
    wav = write_wav(tmp_path / "melody.wav", signal, sr)
    result = AudioToSheetPipeline().run(wav, PipelineOptions("Melody", "Test", 90, "piano"))
    assert result.detected_notes


def _run_configured_fixture(
    wav_path: Path,
    reference: list[ReferenceNote],
    *,
    baseline_targets: dict,
    fixture_name: str,
) -> None:
    cfg = baseline_targets["fixtures"][fixture_name]
    benchmark = run_fixture_benchmark(
        AudioToSheetPipeline(),
        fixture_name,
        tempo_bpm=int(cfg["tempo_bpm"]),
        reference=reference,
        targets=baseline_targets,
        wav_path=wav_path,
    )
    assert_benchmark_passes(benchmark)


def test_c_major_scale_baseline(
    c_major_scale_wav: Path,
    c_major_scale_reference: list[ReferenceNote],
    baseline_targets: dict,
):
    _run_configured_fixture(
        c_major_scale_wav,
        c_major_scale_reference,
        baseline_targets=baseline_targets,
        fixture_name="c_major_scale",
    )


def test_repeated_c_baseline(
    repeated_c_wav: Path,
    repeated_c_reference: list[ReferenceNote],
    baseline_targets: dict,
):
    _run_configured_fixture(
        repeated_c_wav,
        repeated_c_reference,
        baseline_targets=baseline_targets,
        fixture_name="repeated_c",
    )
