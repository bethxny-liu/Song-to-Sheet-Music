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


def synthesize_benchmark_audio(
    notes: list[ReferenceNote],
    sample_rate: int = 44100,
    amplitude: float = 0.32,
) -> tuple[np.ndarray, int]:
    """Render absolute note timings, including overlapping notes, with clear attacks."""
    if not notes:
        return np.zeros(sample_rate, dtype=np.float32), sample_rate

    end_sec = max(n.offset_sec for n in notes) + 0.25
    signal = np.zeros(int(end_sec * sample_rate), dtype=np.float32)
    for note in notes:
        sample_count = max(1, int(note.duration_sec * sample_rate))
        t = np.arange(sample_count, dtype=np.float32) / sample_rate
        frequency = 440.0 * (2.0 ** ((note.midi - 69) / 12.0))
        tone = (
            np.sin(2.0 * np.pi * frequency * t)
            + 0.20 * np.sin(4.0 * np.pi * frequency * t)
        ).astype(np.float32)

        attack = min(sample_count, max(1, int(0.008 * sample_rate)))
        release = min(sample_count - attack, max(1, int(0.025 * sample_rate)))
        envelope = np.ones(sample_count, dtype=np.float32)
        envelope[:attack] = np.linspace(0.0, 1.0, attack, dtype=np.float32)
        if release:
            envelope[-release:] = np.linspace(1.0, 0.0, release, dtype=np.float32)

        start = max(0, int(note.onset_sec * sample_rate))
        end = min(len(signal), start + sample_count)
        signal[start:end] += amplitude * tone[: end - start] * envelope[: end - start]

    peak = float(np.max(np.abs(signal)))
    if peak > 0:
        signal = signal / peak * 0.9
    return signal, sample_rate


def build_tempo_reference(
    tempo_bpm: int,
    texture: str = "monophonic",
) -> list[ReferenceNote]:
    """Eight sixteenths at `tempo_bpm`. texture: monophonic | two_note_harmony | chords."""
    melody = [64, 62, 60, 62, 64, 64, 64, 62]
    event_sec = 15.0 / float(tempo_bpm)
    duration = event_sec * 0.78
    start_sec = 0.20
    notes: list[ReferenceNote] = []
    for index, midi in enumerate(melody):
        onset = start_sec + index * event_sec
        pitches = [midi]
        if texture == "two_note_harmony":
            pitches.append(midi - 4)
        elif texture == "chords":
            root = midi - 12
            pitches.extend([root, root + 4, root + 7])
        elif texture != "monophonic":
            raise ValueError(f"Unknown benchmark texture: {texture}")
        notes.extend(
            ReferenceNote(midi=pitch, onset_sec=onset, duration_sec=duration)
            for pitch in sorted(set(pitches))
        )
    return notes


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
    """Three C4 notes with gaps; tests repeated-pitch splits."""
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
