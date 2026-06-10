"""Orchestrates audio loading, transcription, and sheet-music generation."""

from __future__ import annotations

from dataclasses import replace
import logging
from pathlib import Path

import librosa
import mutagen
import numpy as np

from algo.basic_pitch_transcriber import (
    is_available as basic_pitch_available,
    timed_notes_to_pitch_track,
    timed_notes_to_runs,
    transcribe as basic_pitch_transcribe,
)
from algo.harmony import detect_chord_events
from algo.key_estimation import estimate_key_from_pitches
from algo.models import NoteEvent, PipelineOptions, PipelineResult
from algo.pitch_tracking import (
    bridge_short_unvoiced_gaps,
    fallback_f0_from_piptrack,
    polyphonic_top_voice_track,
)
from algo.run_processing import (
    bridge_same_pitch_across_tiny_rests,
    compress_pitch_track,
    final_merge_weak_same_pitch_events,
    merge_low_confidence_boundary_splits,
    merge_tiny_same_pitch_fragments,
    stabilize_out_of_scale_notes,
    trim_leading_rests,
)
from algo.score_builder import build_score_from_runs, build_score_timed
from algo.source_separation import isolate_piano as demucs_isolate_piano
from algo.source_separation import is_available as demucs_available
from algo.signal_processing import (
    build_segment_boundaries,
    hz_to_midi_track,
    median_smooth_midi,
    normalize_tempo,
    segment_pitches,
    smooth_octave_errors,
    sparsify_frames,
)

logger = logging.getLogger(__name__)
HOP_LENGTH = 512


class AudioToSheetPipeline:
    """Top-level orchestration for the audio -> sheet pipeline."""

    def run(
        self,
        audio_path: Path,
        options: PipelineOptions,
        *,
        work_dir: Path | None = None,
    ) -> PipelineResult:
        work_path, preprocessing = self._prepare_audio(audio_path, options, work_dir)
        sample_rate = mutagen.File(str(work_path)).info.sample_rate
        signal, _ = librosa.load(str(work_path), sr=sample_rate, mono=True)

        if options.layout == "grand" and basic_pitch_available():
            try:
                result = self._run_basic_pitch(work_path, signal, sample_rate, options)
                return replace(
                    result,
                    transcription_engine="basic_pitch",
                    preprocessing=preprocessing,
                )
            except Exception:
                # Any Basic Pitch / ONNX failure should fall back to pYIN, not fail the job.
                logger.warning("Basic Pitch failed; falling back to pYIN.", exc_info=True)

        result = self._run_pyin(signal, sample_rate, options)
        return replace(
            result,
            transcription_engine="pyin",
            preprocessing=preprocessing,
        )

    @staticmethod
    def _prepare_audio(
        audio_path: Path,
        options: PipelineOptions,
        work_dir: Path | None,
    ) -> tuple[Path, str]:
        if not options.isolate_piano:
            return audio_path, "none"

        if not demucs_available():
            logger.warning("Piano isolation requested but Demucs/ffmpeg is not installed.")
            return audio_path, "none"

        stem_dir = work_dir if work_dir is not None else audio_path.parent
        stem_dir.mkdir(parents=True, exist_ok=True)
        piano_wav = demucs_isolate_piano(audio_path, stem_dir / "_demucs")
        return piano_wav, "demucs_piano"

    def _run_basic_pitch(
        self,
        audio_path: Path,
        signal: np.ndarray,
        sample_rate: int,
        options: PipelineOptions,
    ) -> PipelineResult:
        tempo_bpm = self._detect_tempo(signal, sample_rate, options.tempo_bpm)
        bp_kwargs: dict[str, float] = {}
        if options.basic_pitch_onset_threshold is not None:
            bp_kwargs["onset_threshold"] = options.basic_pitch_onset_threshold
        if options.basic_pitch_frame_threshold is not None:
            bp_kwargs["frame_threshold"] = options.basic_pitch_frame_threshold
        timed_notes = basic_pitch_transcribe(audio_path, melody_only=False, **bp_kwargs)
        if not timed_notes:
            raise ValueError("Basic Pitch detected zero notes.")

        frame_duration = HOP_LENGTH / sample_rate
        runs = timed_notes_to_runs(timed_notes, frame_duration=frame_duration)
        estimated_key, key_candidates = estimate_key_from_pitches(runs)
        tonic, mode = estimated_key.split(" ", 1)
        chord_events = detect_chord_events(signal, sample_rate, HOP_LENGTH)

        score, note_confidences = build_score_timed(
            timed_notes, tempo_bpm, options, tonic, mode, chord_events
        )
        total_duration = max(
            len(signal) / sample_rate,
            max((n.onset_sec + n.duration_sec for n in timed_notes), default=0.0),
        )
        pitch_times, pitch_midi = timed_notes_to_pitch_track(timed_notes, total_duration)

        return PipelineResult(
            estimated_key=estimated_key,
            estimated_key_candidates=key_candidates,
            note_count=sum(1 for e in note_confidences if e.get("type") == "note"),
            score=score,
            pitch_times_sec=pitch_times,
            pitch_midi=pitch_midi,
            note_confidences=note_confidences,
            chord_events=chord_events,
        )

    def _run_pyin(
        self, signal: np.ndarray, sample_rate: int, options: PipelineOptions
    ) -> PipelineResult:
        tempo_bpm = self._detect_tempo(signal, sample_rate, options.tempo_bpm)
        onset_env, onset_frames, attack_frames = self._detect_onsets(signal, sample_rate)
        f0_hz, voiced_flag, voiced_prob, midi_track, segments = self._extract_segments(
            signal, sample_rate, onset_env, onset_frames, attack_frames, options
        )

        runs = self._postprocess_segments(segments)
        estimated_key, key_candidates = estimate_key_from_pitches(runs)
        tonic, mode = estimated_key.split(" ", 1)
        runs = stabilize_out_of_scale_notes(runs, tonic=tonic, min_duration_frames=10)
        runs = final_merge_weak_same_pitch_events(runs, weak_boundary_threshold=0.20)
        if not _contains_any_note(runs):
            runs = _rescue_runs_from_segments(segments)

        note_event_count = sum(1 for pitch, frames, *_ in runs if pitch is not None and frames > 0)
        chord_events = detect_chord_events(signal, sample_rate, HOP_LENGTH)
        score, note_confidences = build_score_from_runs(
            runs,
            tempo_bpm,
            options,
            tonic,
            mode,
            HOP_LENGTH / sample_rate,
            chord_events,
            add_chord_tones=note_event_count < 12 and options.layout == "grand",
        )

        pitch_times = librosa.times_like(f0_hz, sr=sample_rate, hop_length=HOP_LENGTH).tolist()
        pitch_midi = [
            None if m is None or np.isnan(m) else float(m) for m in np.asarray(midi_track).tolist()
        ]
        return PipelineResult(
            estimated_key=estimated_key,
            estimated_key_candidates=key_candidates,
            note_count=sum(1 for e in note_confidences if e.get("type") == "note"),
            score=score,
            pitch_times_sec=pitch_times,
            pitch_midi=pitch_midi,
            note_confidences=note_confidences,
            chord_events=chord_events,
        )

    @staticmethod
    def _detect_tempo(signal: np.ndarray, sample_rate: int, fallback_bpm: int) -> int:
        onset_env = librosa.onset.onset_strength(y=signal, sr=sample_rate, hop_length=HOP_LENGTH)
        detected_tempo, _ = librosa.beat.beat_track(
            y=signal,
            sr=sample_rate,
            onset_envelope=onset_env,
            hop_length=HOP_LENGTH,
            start_bpm=max(fallback_bpm, 40),
            tightness=100,
        )
        return normalize_tempo(detected_tempo, fallback_bpm)

    @staticmethod
    def _detect_onsets(
        signal: np.ndarray, sample_rate: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        onset_env = librosa.onset.onset_strength(y=signal, sr=sample_rate, hop_length=HOP_LENGTH)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sample_rate,
            hop_length=HOP_LENGTH,
            units="frames",
            backtrack=True,
            pre_max=5,
            post_max=5,
            delta=0.12,
            wait=3,
        )
        rms = librosa.feature.rms(y=signal, hop_length=HOP_LENGTH)[0]
        rms_delta = np.diff(rms, prepend=rms[0])
        attack_threshold = float(np.percentile(rms_delta, 97))
        attack_frames = sparsify_frames(
            np.where(rms_delta >= attack_threshold)[0], min_gap_frames=8
        )
        return onset_env, onset_frames, attack_frames

    def _extract_segments(
        self,
        signal: np.ndarray,
        sample_rate: int,
        onset_env: np.ndarray,
        onset_frames: np.ndarray,
        attack_frames: np.ndarray,
        options: PipelineOptions,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        f0_hz, voiced_flag, voiced_prob = librosa.pyin(
            signal,
            sr=sample_rate,
            hop_length=HOP_LENGTH,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
        )
        used_fallback = float(np.mean(np.asarray(voiced_flag, dtype=float))) < 0.02
        if used_fallback:
            f0_hz, voiced_flag, voiced_prob = fallback_f0_from_piptrack(
                signal, sample_rate, HOP_LENGTH
            )

        rms = librosa.feature.rms(y=signal, hop_length=HOP_LENGTH)[0]
        energy_threshold = float(np.percentile(rms, 20))
        rms_delta = np.diff(rms, prepend=rms[0])
        attack_frame_set = {int(f) for f in np.asarray(attack_frames).tolist()}
        onset_frame_set = {int(f) for f in np.asarray(onset_frames).tolist()}

        midi_track = median_smooth_midi(
            smooth_octave_errors(hz_to_midi_track(f0_hz, voiced_flag)), window=5
        )
        boundaries = build_segment_boundaries(
            len(midi_track), onset_frames, attack_frames, midi_track=midi_track
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

        if options.layout == "grand" and (
            not _contains_any_note_in_segments(segments)
            or _segment_note_coverage(segments) < 0.38
        ):
            f0_hz, voiced_flag, voiced_prob = polyphonic_top_voice_track(
                signal, sample_rate, HOP_LENGTH
            )
            f0_hz = bridge_short_unvoiced_gaps(f0_hz)
            midi_track = median_smooth_midi(
                smooth_octave_errors(hz_to_midi_track(f0_hz, voiced_flag)), window=5
            )
            boundaries = build_segment_boundaries(
                len(midi_track), onset_frames, attack_frames, midi_track=midi_track
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

        return f0_hz, voiced_flag, voiced_prob, midi_track, segments

    @staticmethod
    def _postprocess_segments(segments: list) -> list[NoteEvent]:
        runs = compress_pitch_track(segments)
        runs = trim_leading_rests(runs)
        runs = merge_tiny_same_pitch_fragments(runs, tiny_fragment_max_frames=18)
        runs = bridge_same_pitch_across_tiny_rests(runs, tiny_rest_max_frames=8)
        return merge_low_confidence_boundary_splits(
            runs, weak_boundary_threshold=0.58, short_note_max_frames=16
        )


def _contains_any_note(runs: list[NoteEvent]) -> bool:
    return any(pitch is not None and frames > 0 for pitch, frames, *_ in runs)


def _contains_any_note_in_segments(
    segments: list[tuple[float | None, int, bool, float, str, float]],
) -> bool:
    return any(pitch is not None and frames > 0 for pitch, frames, *_ in segments)


def _segment_note_coverage(
    segments: list[tuple[float | None, int, bool, float, str, float]],
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


def _rescue_runs_from_segments(
    segments: list[tuple[float | None, int, bool, float, str, float]],
) -> list[NoteEvent]:
    rescued: list[NoteEvent] = []
    for pitch, frames, _is_attack, confidence, b_source, b_conf in segments:
        if frames > 0 and pitch is not None:
            rescued.append((pitch, frames, confidence, None, b_source, b_conf))
    return rescued
