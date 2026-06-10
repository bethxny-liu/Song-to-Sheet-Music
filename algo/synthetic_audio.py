"""Synthetic audio fixtures for benchmarks and regression tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from algo.metrics import ReferenceNote

BENCHMARK_FIXTURES_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def synthesize_melody(
    notes: list[ReferenceNote],
    sample_rate: int = 44100,
    gap_sec: float = 0.12,
    amplitude: float = 0.45,
) -> tuple[np.ndarray, int]:
    """Build a clean monophonic WAV from reference note timings."""
    if not notes:
        return np.zeros(sample_rate, dtype=np.float32), sample_rate

    end_sec = max(n.offset_sec for n in notes) + 0.4
    length = int(end_sec * sample_rate)
    signal = np.zeros(length, dtype=np.float32)

    cursor = 0.0
    for note in sorted(notes, key=lambda n: n.onset_sec):
        start_sec = max(note.onset_sec, cursor + gap_sec * 0.25)
        duration = note.duration_sec
        freq = 440.0 * (2.0 ** ((note.midi - 69) / 12.0))
        t = np.arange(int(duration * sample_rate), dtype=np.float32) / sample_rate
        envelope = np.hanning(len(t)).astype(np.float32)
        tone = amplitude * envelope * np.sin(2.0 * np.pi * freq * t, dtype=np.float32)
        start_idx = int(start_sec * sample_rate)
        end_idx = min(length, start_idx + len(tone))
        signal[start_idx:end_idx] += tone[: end_idx - start_idx]
        cursor = start_sec + duration

    peak = float(np.max(np.abs(signal)))
    if peak > 0:
        signal = signal / peak * 0.9
    return signal, sample_rate


def write_wav(path: Path, signal: np.ndarray, sample_rate: int) -> Path:
    sf.write(path, signal, sample_rate)
    return path


def build_c_major_scale_reference() -> list[ReferenceNote]:
    """C4–C5 major scale, quarter-note-like spacing at 120 BPM."""
    midis = [60, 62, 64, 65, 67, 69, 71, 72]
    note_dur = 0.35
    gap = 0.12
    notes: list[ReferenceNote] = []
    t = 0.15
    for midi in midis:
        notes.append(ReferenceNote(midi=midi, onset_sec=t, duration_sec=note_dur))
        t += note_dur + gap
    return notes


def build_repeated_c_reference() -> list[ReferenceNote]:
    """Three sustained C4 notes — tests note segmentation on repeated pitch."""
    dur = 0.5
    gap = 0.2
    notes: list[ReferenceNote] = []
    t = 0.2
    for _ in range(3):
        notes.append(ReferenceNote(midi=60, onset_sec=t, duration_sec=dur))
        t += dur + gap
    return notes


FIXTURE_BUILDERS = {
    "c_major_scale": build_c_major_scale_reference,
    "repeated_c": build_repeated_c_reference,
}
