"""Turn melody detector output into a treble-staff score."""

from __future__ import annotations

import math

from music21 import clef, duration, instrument, key, meter, metadata, note, stream, tempo

from algo.models import NoteEvent, PipelineOptions


def trim_trailing_rest_only_measures(parts: list[stream.Part], beats_per_measure: float = 4.0) -> None:
    """Remove full trailing measures that contain no notes."""
    last_note_end = 0.0
    for part in parts:
        for current_note in part.recurse().getElementsByClass(note.Note):
            last_note_end = max(last_note_end, float(current_note.offset + current_note.duration.quarterLength))
    if last_note_end <= 0.0:
        return
    keep_until = math.ceil(last_note_end / beats_per_measure) * beats_per_measure
    for part in parts:
        for element in list(part.recurse()):
            if isinstance(element, (note.Note, note.Rest)) and float(element.offset) >= keep_until and element.activeSite is not None:
                element.activeSite.remove(element)


def _init_score(options: PipelineOptions, tonic: str, mode: str, tempo_bpm: int) -> tuple[stream.Score, stream.Part]:
    score = stream.Score()
    score.insert(0, metadata.Metadata())
    score.metadata.title = options.title
    score.metadata.composer = options.composer
    score.metadata.movementName = ""
    part = stream.Part(id="Melody")
    part.append(instrument.fromString(options.instrument_name))
    part.append(key.Key(tonic, mode))
    part.append(tempo.MetronomeMark(number=tempo_bpm))
    part.append(meter.TimeSignature("4/4"))
    part.append(clef.TrebleClef())
    return score, part


def quantize_to_beat_grid(onset_sec: float, duration_sec: float, tempo_bpm: int, *, subdivision: int = 4, min_quarter_length: float = 0.25) -> tuple[float, float]:
    """Snap an absolute-time note onto a sixteenth-note beat grid."""
    beat_sec = 60.0 / max(int(tempo_bpm), 1)
    step_sec = beat_sec / max(subdivision, 1)
    onset_steps = max(0, int(round(max(0.0, onset_sec) / step_sec)))
    end_steps = max(onset_steps + 1, int(round(max(0.0, onset_sec + duration_sec) / step_sec)))
    return onset_steps / subdivision, max(min_quarter_length, (end_steps - onset_steps) / subdivision)


def _make_note(midi_value: int, quarter_length: float) -> note.Note:
    current_note = note.Note()
    current_note.pitch.midi = midi_value
    if current_note.pitch.accidental is not None and float(current_note.pitch.accidental.alter or 0.0) == 0.0:
        current_note.pitch.accidental = None
    current_note.duration = duration.Duration(quarterLength=quarter_length)
    return current_note


def build_score_from_runs(runs: list[NoteEvent], tempo_bpm: int, options: PipelineOptions, tonic: str, mode: str, frame_duration: float) -> tuple[stream.Stream, list[dict[str, float | str | int | None]]]:
    """Build one treble-staff score from sequential pYIN pitch runs."""
    score, part = _init_score(options, tonic, mode, tempo_bpm)
    note_confidences: list[dict[str, float | str | int | None]] = []
    elapsed_seconds = 0.0
    for index, (midi_pitch, segment_frames, confidence, reattack_confidence, boundary_source, boundary_confidence) in enumerate(runs):
        if segment_frames <= 0:
            continue
        seconds = max(segment_frames * frame_duration, frame_duration)
        onset_quarter, quarter_length = quantize_to_beat_grid(elapsed_seconds, seconds, tempo_bpm)
        elapsed_seconds += seconds
        if index == len(runs) - 1 and midi_pitch is not None and quarter_length > 1.0 and confidence < 0.80:
            quarter_length = 1.0
        event = {"onset_quarter": onset_quarter, "duration_quarter": quarter_length, "confidence": confidence, "reattack_confidence": reattack_confidence if midi_pitch is not None else None, "boundary_source": boundary_source, "boundary_confidence": boundary_confidence}
        if midi_pitch is None:
            note_confidences.append({**event, "type": "rest", "midi": None})
            part.insert(onset_quarter, note.Rest(quarterLength=quarter_length))
        else:
            midi_value = int(round(midi_pitch))
            part.insert(onset_quarter, _make_note(midi_value, quarter_length))
            note_confidences.append({**event, "type": "note", "midi": midi_value})
    trim_trailing_rest_only_measures([part])
    score.insert(0, part)
    return score, note_confidences
