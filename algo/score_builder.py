from __future__ import annotations

import math
import re

from music21 import chord as m21chord
from music21 import clef, duration, harmony, instrument, key, meter, metadata, note, stream, tempo

from algo.basic_pitch_transcriber import TimedNote
from algo.hand_split import detect_hand
from algo.models import NoteEvent, PipelineOptions


def safe_chord_symbol(chord_name: str) -> harmony.ChordSymbol | None:
    """Parse common lead-sheet chord names into music21 ChordSymbol objects."""
    raw = chord_name.strip()
    if not raw or raw == "N.C.":
        return None

    match = re.match(r"^([A-Ga-g])([#b]?)(.*)$", raw)
    if not match:
        return None

    root_letter, accidental, suffix = match.groups()
    root = root_letter.upper() + accidental
    suffix_norm = suffix.strip()
    suffix_lower = suffix_norm.lower()

    kind = "major"
    if suffix_lower.startswith("m") and not suffix_lower.startswith("maj"):
        kind = "minor"
        suffix_norm = suffix_norm[1:]

    figure = f"{root}{suffix_norm}"
    try:
        return harmony.ChordSymbol(figure)
    except Exception:
        pass

    try:
        return harmony.ChordSymbol(root=root, kind=kind)
    except Exception:
        return None


def trim_trailing_rest_only_measures(
    parts: list[stream.Part], beats_per_measure: float = 4.0
) -> None:
    """Remove full trailing measures that contain no notes."""
    last_note_end = 0.0
    for part in parts:
        for n in part.recurse().getElementsByClass(note.Note):
            last_note_end = max(last_note_end, float(n.offset + n.duration.quarterLength))

    if last_note_end <= 0.0:
        return

    keep_until = math.ceil(last_note_end / beats_per_measure) * beats_per_measure
    for part in parts:
        for el in list(part.recurse()):
            if not isinstance(el, (note.Note, note.Rest, harmony.ChordSymbol)):
                continue
            if float(el.offset) >= keep_until:
                parent = el.activeSite
                if parent is not None:
                    parent.remove(el)


def _init_parts(
    options: PipelineOptions, tonic: str, mode: str, tempo_bpm: int
) -> tuple[stream.Score, stream.Part, stream.Part | None, tempo.MetronomeMark]:
    mm = tempo.MetronomeMark(number=tempo_bpm)
    score = stream.Score()
    score.insert(0, metadata.Metadata())
    score.metadata.title = options.title
    score.metadata.composer = options.composer
    # music21 copies title into movement-title by default; OSMD renders both lines.
    score.metadata.movementName = ""

    right_hand = stream.Part(id="RightHand")
    right_hand.append(instrument.fromString(options.instrument_name))
    right_hand.append(key.Key(tonic, mode))
    right_hand.append(mm)
    right_hand.append(meter.TimeSignature("4/4"))
    right_hand.append(clef.TrebleClef())

    left_hand: stream.Part | None = None
    if options.layout == "grand":
        left_hand = stream.Part(id="LeftHand")
        left_hand.append(instrument.Piano())
        left_hand.append(key.Key(tonic, mode))
        left_hand.append(meter.TimeSignature("4/4"))
        left_hand.append(clef.BassClef())

    return score, right_hand, left_hand, mm


def _insert_chord_symbols(
    chord_events: list[dict[str, float | str]],
    tempo_bpm: int,
    right_hand: stream.Part,
    left_hand: stream.Part | None,
    *,
    add_chord_tones: bool,
) -> None:
    for chord_event in chord_events:
        chord_name = str(chord_event["chord"])
        if chord_name == "N.C.":
            continue
        onset_sec = float(chord_event["onset_sec"])
        dur_sec = float(chord_event["duration_sec"])
        raw_onset_q = (tempo_bpm * max(0.0, onset_sec)) / 60.0
        raw_dur_q = (tempo_bpm * max(0.0, dur_sec)) / 60.0
        onset_q = round(raw_onset_q)
        dur_q = max(1.0, round(raw_dur_q))
        cs = safe_chord_symbol(chord_name)
        if cs is None:
            continue
        cs.duration = duration.Duration(quarterLength=dur_q)
        right_hand.insert(onset_q, cs)
        if add_chord_tones and left_hand is not None:
            chord_pitches = [p.midi for p in cs.pitches[:4]]
            if chord_pitches:
                while min(chord_pitches) > 52:
                    chord_pitches = [p - 12 for p in chord_pitches]
                while max(chord_pitches) < 36:
                    chord_pitches = [p + 12 for p in chord_pitches]
                ch = m21chord.Chord(chord_pitches)
                ch.duration = duration.Duration(quarterLength=dur_q)
                left_hand.insert(onset_q, ch)


def _finalize_score(
    score: stream.Score, right_hand: stream.Part, left_hand: stream.Part | None
) -> stream.Score:
    trim_parts: list[stream.Part] = [right_hand]
    if left_hand is not None:
        trim_parts.append(left_hand)
    trim_trailing_rest_only_measures(trim_parts, beats_per_measure=4.0)
    score.insert(0, right_hand)
    if left_hand is not None:
        score.insert(0, left_hand)
    return score


def _make_note(midi_value: int, quarter_length: float) -> note.Note:
    n = note.Note()
    n.pitch.midi = midi_value
    if n.pitch.accidental is not None and float(n.pitch.accidental.alter or 0.0) == 0.0:
        n.pitch.accidental = None
    n.duration = duration.Duration(quarterLength=quarter_length)
    return n


def build_score_timed(
    timed_notes: list[TimedNote],
    tempo_bpm: int,
    options: PipelineOptions,
    tonic: str,
    mode: str,
    chord_events: list[dict[str, float | str]],
) -> tuple[stream.Stream, list[dict[str, float | str | int | None]]]:
    """Build a score from absolutely-timed notes (Basic Pitch output)."""
    score, right_hand, left_hand, _mm = _init_parts(options, tonic, mode, tempo_bpm)
    note_confidences: list[dict[str, float | str | int | None]] = []

    for timed_note in sorted(timed_notes, key=lambda n: (n.onset_sec, n.midi)):
        raw_onset_q = (tempo_bpm * max(0.0, timed_note.onset_sec)) / 60.0
        raw_dur_q = (tempo_bpm * max(0.0, timed_note.duration_sec)) / 60.0
        onset_q = max(0.0, round(raw_onset_q * 4) / 4)
        quarter_length = max(0.25, round(raw_dur_q * 4) / 4)

        n = _make_note(timed_note.midi, quarter_length)
        if options.layout == "melody":
            hand = "right"
            right_hand.insert(onset_q, n)
        else:
            hand = detect_hand(timed_note.midi)
            if hand == "left" and left_hand is not None:
                left_hand.insert(onset_q, n)
            else:
                right_hand.insert(onset_q, n)

        note_confidences.append(
            {
                "onset_quarter": onset_q,
                "duration_quarter": quarter_length,
                "type": "note",
                "midi": timed_note.midi,
                "confidence": timed_note.confidence,
                "reattack_confidence": None,
                "boundary_source": "basic_pitch",
                "boundary_confidence": timed_note.confidence,
                "hand": hand,
            }
        )

    _insert_chord_symbols(chord_events, tempo_bpm, right_hand, left_hand, add_chord_tones=False)
    return _finalize_score(score, right_hand, left_hand), note_confidences


def build_score_from_runs(
    runs: list[NoteEvent],
    tempo_bpm: int,
    options: PipelineOptions,
    tonic: str,
    mode: str,
    frame_duration: float,
    chord_events: list[dict[str, float | str]],
    add_chord_tones: bool,
) -> tuple[stream.Stream, list[dict[str, float | str | int | None]]]:
    """Build a score from sequential pitch runs (pYIN pipeline output)."""
    score, right_hand, left_hand, mm = _init_parts(options, tonic, mode, tempo_bpm)
    note_confidences: list[dict[str, float | str | int | None]] = []
    onset_quarter = 0.0

    for idx, (
        midi_pitch,
        segment_frames,
        confidence,
        reattack_confidence,
        boundary_source,
        boundary_confidence,
    ) in enumerate(runs):
        if segment_frames <= 0:
            continue
        seconds = max(segment_frames * frame_duration, 0.125)
        quarter_length = max(0.25, round(mm.secondsToDuration(seconds).quarterLength * 4) / 4)
        if (
            idx == len(runs) - 1
            and midi_pitch is not None
            and quarter_length > 1.0
            and confidence < 0.80
        ):
            quarter_length = 1.0

        event = {
            "onset_quarter": onset_quarter,
            "duration_quarter": quarter_length,
            "confidence": confidence,
            "reattack_confidence": reattack_confidence if midi_pitch is not None else None,
            "boundary_source": boundary_source,
            "boundary_confidence": boundary_confidence,
        }
        if midi_pitch is None:
            note_confidences.append({**event, "type": "rest", "midi": None})
        else:
            midi_value = int(round(midi_pitch))
            n = _make_note(midi_value, quarter_length)
            if options.layout == "melody":
                hand = "right"
                right_hand.insert(onset_quarter, n)
            else:
                hand = detect_hand(midi_value)
                if hand == "left" and left_hand is not None:
                    left_hand.insert(onset_quarter, n)
                else:
                    right_hand.insert(onset_quarter, n)
            note_confidences.append({**event, "type": "note", "midi": midi_value, "hand": hand})
        onset_quarter += quarter_length

    _insert_chord_symbols(
        chord_events, tempo_bpm, right_hand, left_hand, add_chord_tones=add_chord_tones
    )
    return _finalize_score(score, right_hand, left_hand), note_confidences
