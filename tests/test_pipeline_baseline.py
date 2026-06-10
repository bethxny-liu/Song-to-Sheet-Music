"""End-to-end baseline metrics on synthetic audio fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from algo.evaluation import assert_benchmark_passes, run_fixture_benchmark
from algo.metrics import ReferenceNote
from algo.pipeline import AudioToSheetPipeline


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
        layout=str(cfg["layout"]),
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


@pytest.mark.slow
def test_basic_pitch_on_synthetic_scale(
    tmp_path: Path,
    c_major_scale_reference: list[ReferenceNote],
):
    """Neural path baseline on clean synthetic audio (skipped if Basic Pitch unavailable)."""
    from algo.basic_pitch_transcriber import is_available, transcribe
    from algo.evaluation import GRAND_BP_FRAME_THRESHOLD, GRAND_BP_ONSET_THRESHOLD
    from algo.metrics import evaluate_transcription, reference_to_arrays, timed_notes_to_arrays
    from algo.synthetic_audio import synthesize_melody, write_wav

    if not is_available():
        pytest.skip("Basic Pitch not available")

    signal, sr = synthesize_melody(c_major_scale_reference)
    wav = write_wav(tmp_path / "bp_scale.wav", signal, sr)
    notes = transcribe(
        wav,
        melody_only=False,
        onset_threshold=GRAND_BP_ONSET_THRESHOLD,
        frame_threshold=GRAND_BP_FRAME_THRESHOLD,
        minimum_note_length_ms=50.0,
    )
    assert notes, "Basic Pitch returned no notes on synthetic scale"

    ref_i, ref_p = reference_to_arrays(c_major_scale_reference)
    est_i, est_p = timed_notes_to_arrays(notes)
    metrics = evaluate_transcription(est_i, est_p, ref_i, ref_p, onset_tolerance=0.15)
    assert metrics.f1 >= 0.5, f"Basic Pitch F1 too low: {metrics.to_dict()}"
