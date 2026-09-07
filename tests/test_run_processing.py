from algo.models import NoteEvent
from algo.run_processing import (
    bridge_same_pitch_across_tiny_rests,
    compress_pitch_track,
    merge_tiny_same_pitch_fragments,
    trim_leading_rests,
)


def test_trim_leading_rests():
    runs: list[NoteEvent] = [
        (None, 10, 0.5, None, "onset", 0.5),
        (60.0, 20, 0.9, None, "onset", 0.8),
    ]
    trimmed = trim_leading_rests(runs)
    assert len(trimmed) == 1
    assert trimmed[0][0] == 60.0


def test_merge_tiny_same_pitch_fragments():
    runs: list[NoteEvent] = [
        (60.0, 30, 0.9, None, "other", 0.8),
        (60.0, 8, 0.7, None, "other", 0.4),
    ]
    merged = merge_tiny_same_pitch_fragments(runs, tiny_fragment_max_frames=18)
    assert len(merged) == 1
    assert merged[0][1] == 38


def test_merge_tiny_same_pitch_keeps_onset_split():
    runs: list[NoteEvent] = [
        (60.0, 12, 0.9, None, "onset", 0.8),
        (60.0, 10, 0.85, None, "onset", 0.6),
    ]
    kept = merge_tiny_same_pitch_fragments(runs, tiny_fragment_max_frames=18)
    assert len(kept) == 2


def test_bridge_same_pitch_across_tiny_rests():
    runs: list[NoteEvent] = [
        (62.0, 20, 0.9, None, "onset", 0.8),
        (None, 6, 0.5, None, "other", 0.3),
        (62.0, 20, 0.85, None, "other", 0.05),
    ]
    bridged = bridge_same_pitch_across_tiny_rests(runs, tiny_rest_max_frames=8)
    assert len(bridged) == 1
    assert bridged[0][0] == 62.0


def test_bridge_same_pitch_preserves_onset_articulation():
    runs: list[NoteEvent] = [
        (64.0, 12, 0.9, None, "onset", 0.8),
        (None, 3, 0.4, None, "other", 0.2),
        (64.0, 12, 0.85, None, "onset", 0.7),
    ]
    kept = bridge_same_pitch_across_tiny_rests(runs, tiny_rest_max_frames=8)
    assert len(kept) == 3


def test_compress_pitch_track_splits_same_pitch_on_onset():
    segments = [
        (64.0, 10, False, 0.9, "onset", 0.8),
        (64.0, 10, False, 0.85, "onset", 0.7),
        (62.0, 10, False, 0.9, "onset", 0.7),
    ]
    runs = compress_pitch_track(segments, reattack_min_frames=4, min_note_frames=2, min_rest_frames=1)
    pitched = [run for run in runs if run[0] is not None]
    assert [int(run[0]) for run in pitched] == [64, 64, 62]


def test_compress_pitch_track_merges_identical_pitches():
    segments = [
        (60.0, 15, False, 0.9, "onset", 0.8),
        (60.0, 12, False, 0.85, "onset", 0.6),
        (62.0, 10, True, 0.9, "attack", 0.7),
    ]
    runs = compress_pitch_track(segments)
    assert runs[0][0] == 60.0
    assert runs[0][1] >= 15
