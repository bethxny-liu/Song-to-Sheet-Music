from algo.polyphonic import TimedNote
from evaluation.metrics import timed_notes_to_arrays


def test_polyphonic_notes_preserve_overlapping_events():
    notes = [
        TimedNote(onset_sec=0.0, duration_sec=1.0, midi=60, confidence=0.9),
        TimedNote(onset_sec=0.0, duration_sec=1.0, midi=64, confidence=0.8),
    ]
    intervals, pitches = timed_notes_to_arrays(notes)
    assert intervals.tolist() == [[0.0, 1.0], [0.0, 1.0]]
    assert pitches.tolist() == [60.0, 64.0]
