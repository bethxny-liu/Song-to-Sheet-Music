from __future__ import annotations


def detect_hand(midi_value: int, split_point: int = 52) -> str:
    # Melody-first default: keep notes in right hand unless clearly low.
    return "left" if midi_value < split_point else "right"
