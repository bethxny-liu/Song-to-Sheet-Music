from algo.score_builder import quantize_to_beat_grid


def test_quantize_to_beat_grid_snaps_to_sixteenth_notes():
    onset_q, dur_q = quantize_to_beat_grid(0.5, 0.5, tempo_bpm=120)
    assert onset_q == 1.0
    assert dur_q == 1.0


def test_quantize_to_beat_grid_keeps_sixteenth_floor():
    onset_q, dur_q = quantize_to_beat_grid(0.20, 0.04, tempo_bpm=90)
    assert onset_q in {0.25, 0.5}
    assert dur_q >= 0.25
