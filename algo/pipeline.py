from __future__ import annotations

import math
from pathlib import Path
import re

import librosa
import mutagen
import numpy as np
from music21 import chord as m21chord
from music21 import clef, duration, harmony, instrument, key, meter, metadata, note, stream, tempo

from algo.hand_split import detect_hand
from algo.harmony import detect_chord_events
from algo.key_estimation import estimate_key_from_pitches
from algo.models import PipelineOptions, PipelineResult
from algo.run_processing import (
    bridge_same_pitch_across_tiny_rests,
    compress_pitch_track,
    final_merge_weak_same_pitch_events,
    merge_low_confidence_boundary_splits,
    merge_tiny_same_pitch_fragments,
    stabilize_out_of_scale_notes,
    trim_leading_rests,
)
from algo.signal_processing import (
    build_segment_boundaries,
    hz_to_midi_track,
    median_smooth_midi,
    normalize_tempo,
    segment_pitches,
    smooth_octave_errors,
    sparsify_frames,
)


class AudioToSheetPipeline:
    """Top-level orchestration for the audio -> sheet pipeline."""

    @staticmethod
    def _fallback_f0_from_piptrack(
        signal: np.ndarray, sample_rate: int, hop_length: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate a monophonic melody track from spectral peaks.

        This fallback is used when pyin returns almost no voiced frames
        (common on dense/polyphonic recordings).
        """
        pitches, magnitudes = librosa.piptrack(
            y=signal,
            sr=sample_rate,
            hop_length=hop_length,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
        )
        frame_count = pitches.shape[1]
        f0_hz = np.full(frame_count, np.nan, dtype=float)
        voiced_flag = np.zeros(frame_count, dtype=bool)
        voiced_prob = np.zeros(frame_count, dtype=float)

        max_mag = float(np.max(magnitudes)) if magnitudes.size else 0.0
        mag_floor = max_mag * 0.08
        for i in range(frame_count):
            col_mag = magnitudes[:, i]
            idx = int(np.argmax(col_mag))
            peak_mag = float(col_mag[idx])
            peak_freq = float(pitches[idx, i])
            if peak_mag <= mag_floor or peak_freq <= 0.0:
                continue
            f0_hz[i] = peak_freq
            voiced_flag[i] = True
            voiced_prob[i] = min(1.0, peak_mag / (max_mag + 1e-8))
        return f0_hz, voiced_flag, voiced_prob

    @staticmethod
    def _polyphonic_top_voice_track(
        signal: np.ndarray, sample_rate: int, hop_length: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate melody from polyphonic audio with continuity bias."""
        harmonic, _ = librosa.effects.hpss(signal)
        pitches, magnitudes = librosa.piptrack(
            y=harmonic,
            sr=sample_rate,
            hop_length=hop_length,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
        )
        frame_count = pitches.shape[1]
        f0_hz = np.full(frame_count, np.nan, dtype=float)
        voiced_flag = np.zeros(frame_count, dtype=bool)
        voiced_prob = np.zeros(frame_count, dtype=float)

        max_mag = float(np.max(magnitudes)) if magnitudes.size else 0.0
        global_floor = max_mag * 0.04
        prev_midi: float | None = None
        for i in range(frame_count):
            col_pitch = pitches[:, i]
            col_mag = magnitudes[:, i]
            valid_idx = np.where((col_pitch > 0.0) & (col_mag > global_floor))[0]
            if valid_idx.size == 0:
                prev_midi = None
                continue

            candidate_freq = col_pitch[valid_idx]
            candidate_mag = col_mag[valid_idx]
            candidate_midi = librosa.hz_to_midi(candidate_freq)

            # Prefer plausible RH melody range first; fallback to all candidates.
            in_range = (candidate_midi >= 60.0) & (candidate_midi <= 88.0)
            if np.any(in_range):
                candidate_freq = candidate_freq[in_range]
                candidate_mag = candidate_mag[in_range]
                candidate_midi = candidate_midi[in_range]

            mag_norm = candidate_mag / (float(np.max(candidate_mag)) + 1e-8)
            midi_norm = (candidate_midi - 48.0) / 40.0
            if prev_midi is None:
                jump_penalty = np.zeros_like(candidate_midi)
            else:
                jump_penalty = np.minimum(np.abs(candidate_midi - prev_midi) / 12.0, 1.0)
            scores = (0.70 * mag_norm) + (0.25 * midi_norm) - (0.35 * jump_penalty)
            best_idx = int(np.argmax(scores))
            peak_freq = float(candidate_freq[best_idx])
            peak_mag = float(candidate_mag[best_idx])
            if peak_freq <= 0.0 or peak_mag <= 0.0:
                prev_midi = None
                continue
            f0_hz[i] = peak_freq
            voiced_flag[i] = True
            voiced_prob[i] = min(1.0, peak_mag / (max_mag + 1e-8))
            prev_midi = float(librosa.hz_to_midi(peak_freq))
        return f0_hz, voiced_flag, voiced_prob

    @staticmethod
    def _bridge_short_unvoiced_gaps(f0_hz: np.ndarray, max_gap_frames: int = 3) -> np.ndarray:
        """Fill tiny NaN gaps in f0 to avoid fragmented rests."""
        bridged = f0_hz.copy()
        i = 0
        n = len(bridged)
        while i < n:
            if not np.isnan(bridged[i]):
                i += 1
                continue
            start = i
            while i < n and np.isnan(bridged[i]):
                i += 1
            end = i
            gap = end - start
            left = start - 1
            right = end
            if (
                gap <= max_gap_frames
                and left >= 0
                and right < n
                and not np.isnan(bridged[left])
                and not np.isnan(bridged[right])
            ):
                if abs(librosa.hz_to_midi(bridged[left]) - librosa.hz_to_midi(bridged[right])) <= 2.0:
                    bridged[start:end] = (bridged[left] + bridged[right]) / 2.0
        return bridged

    @staticmethod
    def _contains_any_note(
        events: list[tuple[float | None, int, float, float | None, str, float]]
    ) -> bool:
        return any((pitch is not None and frames > 0) for pitch, frames, *_ in events)

    @staticmethod
    def _rescue_runs_from_segments(
        segments: list[tuple[float | None, int, bool, float, str, float]]
    ) -> list[tuple[float | None, int, float, float | None, str, float]]:
        """Last-resort note extraction if merge pipeline wipes out melody."""
        rescued: list[tuple[float | None, int, float, float | None, str, float]] = []
        for pitch, frames, _is_attack, confidence, b_source, b_conf in segments:
            if frames <= 0:
                continue
            if pitch is None:
                continue
            rescued.append((pitch, frames, confidence, None, b_source, b_conf))
        return rescued

    @staticmethod
    def _safe_chord_symbol(chord_name: str) -> harmony.ChordSymbol | None:
        """Build a music21 chord symbol from common lead-sheet notation.

        Supports forms like C, Am, F#, Bb, F#m, Bb7. Falls back gracefully if parsing fails.
        """
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

    @staticmethod
    def _trim_trailing_rest_only_measures(
        parts: list[stream.Part], beats_per_measure: float = 4.0
    ) -> None:
        """Drop full trailing measures after the last actual note."""
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

    @staticmethod
    def _contains_any_note_in_segments(
        segments: list[tuple[float | None, int, bool, float, str, float]]
    ) -> bool:
        return any(pitch is not None and frames > 0 for pitch, frames, *_ in segments)

    @staticmethod
    def _segment_note_coverage(
        segments: list[tuple[float | None, int, bool, float, str, float]]
    ) -> float:
        total_frames = sum(max(0, int(frames)) for _pitch, frames, *_ in segments)
        if total_frames == 0:
            return 0.0
        note_frames = sum(
            max(0, int(frames))
            for pitch, frames, *_ in segments
            if pitch is not None and int(frames) > 0
        )
        return float(note_frames) / float(total_frames)

    def run(self, audio_path: Path, options: PipelineOptions) -> PipelineResult:
        hop_length = 512
        sample_rate = mutagen.File(str(audio_path)).info.sample_rate
        signal, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)

        onset_env = librosa.onset.onset_strength(y=signal, sr=sample_rate, hop_length=hop_length)
        detected_tempo, _ = librosa.beat.beat_track(
            y=signal,
            sr=sample_rate,
            onset_envelope=onset_env,
            hop_length=hop_length,
            start_bpm=max(options.tempo_bpm, 40),
            tightness=100,
        )
        tempo_bpm = normalize_tempo(detected_tempo, options.tempo_bpm)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sample_rate,
            hop_length=hop_length,
            units="frames",
            backtrack=True,
            pre_max=5,
            post_max=5,
            delta=0.12,
            wait=3,
        )

        f0_hz, voiced_flag, voiced_prob = librosa.pyin(
            signal,
            sr=sample_rate,
            hop_length=hop_length,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
        )
        voiced_ratio = float(np.mean(np.asarray(voiced_flag, dtype=float)))
        used_fallback = False
        if voiced_ratio < 0.02:
            f0_hz, voiced_flag, voiced_prob = self._fallback_f0_from_piptrack(
                signal=signal, sample_rate=sample_rate, hop_length=hop_length
            )
            used_fallback = True

        rms = librosa.feature.rms(y=signal, hop_length=hop_length)[0]
        energy_threshold = float(np.percentile(rms, 20))
        rms_delta = np.diff(rms, prepend=rms[0])
        attack_threshold = float(np.percentile(rms_delta, 97))
        attack_frames = sparsify_frames(np.where(rms_delta >= attack_threshold)[0], min_gap_frames=8)

        attack_frame_set = {int(frame) for frame in np.asarray(attack_frames).tolist()}
        onset_frame_set = {int(frame) for frame in np.asarray(onset_frames).tolist()}

        midi_track = hz_to_midi_track(f0_hz, voiced_flag)
        midi_track = median_smooth_midi(midi_track, window=5)
        midi_track = smooth_octave_errors(midi_track)

        boundaries = build_segment_boundaries(
            frame_count=len(midi_track),
            onset_frames=onset_frames,
            attack_frames=attack_frames,
            midi_track=midi_track,
        )
        segments = segment_pitches(
            midi_track=midi_track,
            boundaries=boundaries,
            rms=rms,
            rms_delta=rms_delta,
            energy_threshold=energy_threshold,
            voiced_prob=voiced_prob,
            attack_frame_set=attack_frame_set,
            onset_frame_set=onset_frame_set,
            onset_env=onset_env,
            min_voiced_ratio=0.10 if used_fallback else 0.20,
            min_voiced_prob=0.10 if used_fallback else 0.45,
            require_energy_threshold=not used_fallback,
        )
        note_coverage = self._segment_note_coverage(segments)
        # Polyphonic HPSS+piptrack prefers ~MIDI 60–88 and can erase low melody (e.g. E3).
        # Only use it in grand-staff mode when pyin fails entirely or coverage is very low.
        should_use_polyphonic_top_voice = not self._contains_any_note_in_segments(segments)
        if options.layout == "grand":
            should_use_polyphonic_top_voice = (
                should_use_polyphonic_top_voice or note_coverage < 0.38
            )
        if should_use_polyphonic_top_voice:
            f0_hz, voiced_flag, voiced_prob = self._polyphonic_top_voice_track(
                signal=signal, sample_rate=sample_rate, hop_length=hop_length
            )
            f0_hz = self._bridge_short_unvoiced_gaps(f0_hz, max_gap_frames=3)
            midi_track = hz_to_midi_track(f0_hz, voiced_flag)
            midi_track = median_smooth_midi(midi_track, window=5)
            midi_track = smooth_octave_errors(midi_track)
            boundaries = build_segment_boundaries(
                frame_count=len(midi_track),
                onset_frames=onset_frames,
                attack_frames=attack_frames,
                midi_track=midi_track,
            )
            segments = segment_pitches(
                midi_track=midi_track,
                boundaries=boundaries,
                rms=rms,
                rms_delta=rms_delta,
                energy_threshold=energy_threshold,
                voiced_prob=voiced_prob,
                attack_frame_set=attack_frame_set,
                onset_frame_set=onset_frame_set,
                onset_env=onset_env,
                min_voiced_ratio=0.06,
                min_voiced_prob=0.05,
                require_energy_threshold=False,
            )

        runs = compress_pitch_track(segments)
        runs = trim_leading_rests(runs)
        runs = merge_tiny_same_pitch_fragments(runs, tiny_fragment_max_frames=18)
        runs = bridge_same_pitch_across_tiny_rests(runs, tiny_rest_max_frames=8)
        runs = merge_low_confidence_boundary_splits(
            runs, weak_boundary_threshold=0.58, short_note_max_frames=16
        )
        estimated_key, key_candidates = estimate_key_from_pitches(runs)
        tonic, mode = estimated_key.split(" ", 1)
        runs = stabilize_out_of_scale_notes(runs, tonic=tonic, min_duration_frames=10)
        runs = final_merge_weak_same_pitch_events(runs, weak_boundary_threshold=0.20)
        if not self._contains_any_note(runs):
            runs = self._rescue_runs_from_segments(segments)
        note_event_count = sum(1 for pitch, frames, *_ in runs if pitch is not None and frames > 0)
        chord_events = detect_chord_events(signal=signal, sample_rate=sample_rate, hop_length=hop_length)

        score, note_confidences = self._build_score(
            runs=runs,
            tempo_bpm=tempo_bpm,
            options=options,
            tonic=tonic,
            mode=mode,
            frame_duration=hop_length / sample_rate,
            chord_events=chord_events,
            add_chord_tones=note_event_count < 12 and options.layout == "grand",
        )

        pitch_times = librosa.times_like(f0_hz, sr=sample_rate, hop_length=hop_length).tolist()
        pitch_midi = [
            None if m is None or np.isnan(m) else float(m) for m in np.asarray(midi_track).tolist()
        ]
        return PipelineResult(
            estimated_key=estimated_key,
            estimated_key_candidates=key_candidates,
            note_count=sum(1 for event in note_confidences if event.get("type") == "note"),
            score=score,
            pitch_times_sec=pitch_times,
            pitch_midi=pitch_midi,
            note_confidences=note_confidences,
            chord_events=chord_events,
        )

    def _build_score(
        self,
        runs: list[tuple[float | None, int, float, float | None, str, float]],
        tempo_bpm: int,
        options: PipelineOptions,
        tonic: str,
        mode: str,
        frame_duration: float,
        chord_events: list[dict[str, float | str]],
        add_chord_tones: bool,
    ) -> tuple[stream.Stream, list[dict[str, float | str | int | None]]]:
        mm = tempo.MetronomeMark(number=tempo_bpm)
        score = stream.Score()
        score.insert(0, metadata.Metadata())
        score.metadata.title = options.title
        score.metadata.composer = options.composer
        right_hand = stream.Part(id="RightHand")
        right_hand_key = key.Key(tonic, mode)

        right_hand.append(instrument.fromString(options.instrument_name))
        right_hand.append(right_hand_key)
        right_hand.append(mm)
        right_hand.append(meter.TimeSignature("4/4"))
        right_hand.append(clef.TrebleClef())

        left_hand: stream.Part | None = None
        if options.layout == "grand":
            left_hand = stream.Part(id="LeftHand")
            left_hand_key = key.Key(tonic, mode)
            left_hand.append(instrument.Piano())
            left_hand.append(left_hand_key)
            left_hand.append(meter.TimeSignature("4/4"))
            left_hand.append(clef.BassClef())

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
            # End-of-file fades can produce a weak, overlong terminal note event
            # that consumes closing bars and creates misleading rest notation.
            if (
                idx == len(runs) - 1
                and midi_pitch is not None
                and quarter_length > 1.0
                and confidence < 0.80
            ):
                quarter_length = 1.0
            note_duration = duration.Duration(quarterLength=quarter_length)

            event = {
                "onset_quarter": onset_quarter,
                "duration_quarter": quarter_length,
                "confidence": confidence,
                "reattack_confidence": reattack_confidence if midi_pitch is not None else None,
                "boundary_source": boundary_source,
                "boundary_confidence": boundary_confidence,
            }
            if midi_pitch is None:
                # Do not insert segmented rests directly: many tiny rest events
                # create cluttered notation (half+quarter+eighth, etc.).
                # Let notation/export derive rests from note gaps.
                note_confidences.append({**event, "type": "rest", "midi": None})
            else:
                midi_value = int(round(midi_pitch))
                n = note.Note()
                n.pitch.midi = midi_value
                # Avoid noisy courtesy naturals (e.g. C-natural in C major)
                # unless an actual sharp/flat accidental is present.
                if (
                    n.pitch.accidental is not None
                    and float(n.pitch.accidental.alter or 0.0) == 0.0
                ):
                    n.pitch.accidental = None
                n.duration = note_duration
                if options.layout == "melody":
                    hand = "right"
                    right_hand.insert(onset_quarter, n)
                else:
                    # Split between staves using pitch (grand staff).
                    hand = detect_hand(midi_value)
                    if hand == "left" and left_hand is not None:
                        left_hand.insert(onset_quarter, n)
                    else:
                        right_hand.insert(onset_quarter, n)
                note_confidences.append(
                    {**event, "type": "note", "midi": midi_value, "hand": hand}
                )
            onset_quarter += quarter_length

        for chord_event in chord_events:
            chord_name = str(chord_event["chord"])
            if chord_name == "N.C.":
                continue
            onset_sec = float(chord_event["onset_sec"])
            dur_sec = float(chord_event["duration_sec"])
            # secondsToDuration rejects zero; convert via tempo math instead.
            raw_onset_q = (tempo_bpm * max(0.0, onset_sec)) / 60.0
            raw_dur_q = (tempo_bpm * max(0.0, dur_sec)) / 60.0
            # Snap chord symbols to whole beats to avoid fragmented-looking
            # empty measures (half+quarter+eighth rest artifacts).
            onset_q = round(raw_onset_q)
            dur_q = max(1.0, round(raw_dur_q))
            cs = self._safe_chord_symbol(chord_name)
            if cs is None:
                continue
            cs.duration = duration.Duration(quarterLength=dur_q)
            right_hand.insert(onset_q, cs)
            if add_chord_tones and left_hand is not None:
                # Optionally render chord tones as playable notes (left hand),
                # only when melody extraction is very sparse.
                chord_pitches = [p.midi for p in cs.pitches[:4]]
                if chord_pitches:
                    while min(chord_pitches) > 52:
                        chord_pitches = [p - 12 for p in chord_pitches]
                    while max(chord_pitches) < 36:
                        chord_pitches = [p + 12 for p in chord_pitches]
                    ch = m21chord.Chord(chord_pitches)
                    ch.duration = duration.Duration(quarterLength=dur_q)
                    left_hand.insert(onset_q, ch)

        trim_parts: list[stream.Part] = [right_hand]
        if left_hand is not None:
            trim_parts.append(left_hand)
        self._trim_trailing_rest_only_measures(trim_parts, beats_per_measure=4.0)

        # Keep canonical piano order: treble (right hand) above bass (left hand).
        score.insert(0, right_hand)
        if left_hand is not None:
            score.insert(0, left_hand)
        return score, note_confidences

