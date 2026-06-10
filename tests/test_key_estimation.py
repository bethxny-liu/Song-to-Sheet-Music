from algo.key_estimation import estimate_key_from_pitches
from algo.models import NoteEvent


def _runs_from_midis(midis: list[int], frames: int = 40) -> list[NoteEvent]:
    return [(float(m), frames, 0.9, None, "test", 0.9) for m in midis]


def test_c_major_scale_estimates_c_major():
    # C major scale pitch classes in octave 4–5
    midis = [60, 62, 64, 65, 67, 69, 71, 72]
    key, candidates = estimate_key_from_pitches(_runs_from_midis(midis))
    assert key == "C major"
    assert candidates[0][0] == "C major"


def test_empty_runs_default_to_c_major():
    key, candidates = estimate_key_from_pitches([])
    assert key == "C major"
    assert candidates[0][1] == 1.0


def test_f_major_pentachord():
    midis = [65, 67, 69, 70, 72]  # F G A Bb C
    key, _ = estimate_key_from_pitches(_runs_from_midis(midis))
    assert key == "F major"
