from algo.models import NoteEvent
from algo.run_processing import (
    bridge_same_pitch_across_tiny_rests,
    compress_pitch_track,
    merge_tiny_same_pitch_fragments,
)


def test_merge_tiny_same_pitch_fragments():
    runs: list[NoteEvent] = [
        NoteEvent(
            pitch=60.0,
            frames=30,
            confidence=0.9,
            reattack_confidence=None,
            boundary_source="other",
            boundary_confidence=0.8,
        ),
        NoteEvent(
            pitch=60.0,
            frames=8,
            confidence=0.7,
            reattack_confidence=None,
            boundary_source="other",
            boundary_confidence=0.4,
        ),
    ]
    merged = merge_tiny_same_pitch_fragments(runs, tiny_fragment_max_frames=18)
    assert len(merged) == 1
    assert merged[0].frames == 38


def test_merge_tiny_same_pitch_keeps_onset_split():
    runs: list[NoteEvent] = [
        NoteEvent(
            pitch=60.0,
            frames=12,
            confidence=0.9,
            reattack_confidence=None,
            boundary_source="onset",
            boundary_confidence=0.8,
        ),
        NoteEvent(
            pitch=60.0,
            frames=10,
            confidence=0.85,
            reattack_confidence=None,
            boundary_source="onset",
            boundary_confidence=0.6,
        ),
    ]
    kept = merge_tiny_same_pitch_fragments(runs, tiny_fragment_max_frames=18)
    assert len(kept) == 2


def test_bridge_same_pitch_across_tiny_rests():
    runs: list[NoteEvent] = [
        NoteEvent(
            pitch=62.0,
            frames=20,
            confidence=0.9,
            reattack_confidence=None,
            boundary_source="onset",
            boundary_confidence=0.8,
        ),
        NoteEvent(
            pitch=None,
            frames=6,
            confidence=0.5,
            reattack_confidence=None,
            boundary_source="other",
            boundary_confidence=0.3,
        ),
        NoteEvent(
            pitch=62.0,
            frames=20,
            confidence=0.85,
            reattack_confidence=None,
            boundary_source="other",
            boundary_confidence=0.05,
        ),
    ]
    bridged = bridge_same_pitch_across_tiny_rests(runs, tiny_rest_max_frames=8)
    assert len(bridged) == 1
    assert bridged[0].pitch == 62.0


def test_bridge_same_pitch_preserves_onset_articulation():
    runs: list[NoteEvent] = [
        NoteEvent(
            pitch=64.0,
            frames=12,
            confidence=0.9,
            reattack_confidence=None,
            boundary_source="onset",
            boundary_confidence=0.8,
        ),
        NoteEvent(
            pitch=None,
            frames=3,
            confidence=0.4,
            reattack_confidence=None,
            boundary_source="other",
            boundary_confidence=0.2,
        ),
        NoteEvent(
            pitch=64.0,
            frames=12,
            confidence=0.85,
            reattack_confidence=None,
            boundary_source="onset",
            boundary_confidence=0.7,
        ),
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
    pitched = [run for run in runs if run.pitch is not None]
    assert [int(run.pitch) for run in pitched] == [64, 64, 62]


def test_compress_pitch_track_merges_identical_pitches():
    segments = [
        (60.0, 15, False, 0.9, "onset", 0.8),
        (60.0, 12, False, 0.85, "onset", 0.6),
        (62.0, 10, True, 0.9, "attack", 0.7),
    ]
    runs = compress_pitch_track(segments)
    assert runs[0].pitch == 60.0
    assert runs[0].frames >= 15
