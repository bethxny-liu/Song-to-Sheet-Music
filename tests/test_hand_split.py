from algo.hand_split import detect_hand


def test_middle_c_in_right_hand_by_default():
    assert detect_hand(60) == "right"


def test_low_bass_in_left_hand():
    assert detect_hand(48) == "left"


def test_split_point_boundary():
    assert detect_hand(52) == "right"
    assert detect_hand(51) == "left"
