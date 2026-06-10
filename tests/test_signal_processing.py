import numpy as np

from algo.signal_processing import (
    frames_with_pitch_discontinuity,
    median_smooth_midi,
    normalize_tempo,
    smooth_octave_errors,
    sparsify_frames,
)


def test_normalize_tempo_uses_fallback_for_invalid_values():
    assert normalize_tempo(float("nan"), 90) == 90
    assert normalize_tempo(10.0, 90) == 90
    assert normalize_tempo(300.0, 90) == 90


def test_normalize_tempo_rounds_valid_detection():
    assert normalize_tempo(118.4, 90) == 118


def test_sparsify_frames_enforces_minimum_gap():
    frames = np.array([1, 2, 3, 10, 11, 20])
    kept = sparsify_frames(frames, min_gap_frames=5)
    assert kept.tolist() == [1, 10, 20]


def test_smooth_octave_errors_corrects_spike():
    track = np.array([60.0, 60.0, 72.0, 60.0, 60.0])
    corrected = smooth_octave_errors(track)
    assert abs(corrected[2] - 60.0) < 1.0


def test_median_smooth_midi_reduces_outlier():
    track = np.array([60.0, 60.0, 80.0, 60.0, 60.0])
    smoothed = median_smooth_midi(track, window=5)
    assert smoothed[2] < 75.0


def test_frames_with_pitch_discontinuity_finds_leap():
    track = np.full(40, np.nan)
    track[0:15] = 60.0
    track[15:40] = 67.0
    jumps = frames_with_pitch_discontinuity(track, half_window=3, min_jump_semitones=3.0)
    assert len(jumps) >= 1
