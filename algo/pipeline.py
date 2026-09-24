"""Orchestrates audio loading, transcription, and sheet-music generation."""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np

from algo.audio import load_audio

from algo.key_estimation import estimate_key_from_pitches
from algo.models import DetectedNote, NoteEvent, PipelineOptions, PipelineResult
from algo.pitch_tracking import (
    fallback_f0_from_piptrack,
)
from algo.run_processing import (
    bridge_same_pitch_across_tiny_rests,
    compress_pitch_track,
    final_merge_weak_same_pitch_events,
    merge_low_confidence_boundary_splits,
    merge_tiny_same_pitch_fragments,
)
from algo.score_builder import build_score_from_runs
from algo.signal_processing import (
    build_segment_boundaries,
    hz_to_midi_track,
    median_smooth_midi,
    segment_pitches,
    smooth_octave_errors,
    sparsify_frames,
)

logger = logging.getLogger(__name__)
HOP_LENGTH = 512
MELODY_HOP_LENGTH = 256


class AudioToSheetPipeline:

    def run(
        self,
        audio_path: Path,
        options: PipelineOptions,
        *,
        max_duration_sec: float | None = None,
    ) -> PipelineResult:
        signal, sample_rate = load_audio(audio_path, max_duration_sec, trim_leading_silence=True)
        return self._run_pyin(signal, sample_rate, options)

    def _run_pyin(
        self, signal: np.ndarray, sample_rate: int, options: PipelineOptions
    ) -> PipelineResult:
        hop_length = MELODY_HOP_LENGTH
        tempo_bpm = options.tempo_bpm
        onset_env, onset_frames, attack_frames = self._detect_onsets(
            signal, sample_rate, tempo_bpm, hop_length=hop_length
        )
        f0_hz, voiced_flag, voiced_prob, midi_track, segments = self._extract_segments(
            signal,
            sample_rate,
            onset_env,
            onset_frames,
            attack_frames,
            hop_length=hop_length,
        )

        runs = self._postprocess_segments(
            segments, tempo_bpm=tempo_bpm, sample_rate=sample_rate, hop_length=hop_length
        )
        estimated_key, key_candidates = estimate_key_from_pitches(runs)
        tonic, mode = estimated_key.split(" ", 1)
        # Key estimation describes the recording; it must not rewrite detected pitches.
        runs = final_merge_weak_same_pitch_events(runs, weak_boundary_threshold=0.20)
        if not _contains_any_note(runs):
            runs = _rescue_runs_from_segments(segments)

        score, note_confidences = build_score_from_runs(
            runs,
            tempo_bpm,
            options,
            tonic,
            mode,
            hop_length / sample_rate,
        )

        pitch_times = librosa.times_like(
            f0_hz, sr=sample_rate, hop_length=hop_length
        ).tolist()
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
            detected_notes=_runs_to_detected_notes(runs, hop_length / sample_rate),
        )

    @staticmethod
    def _detect_onsets(
        signal: np.ndarray,
        sample_rate: int,
        tempo_bpm: int,
        *,
        hop_length: int = HOP_LENGTH,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        frame_duration = hop_length / sample_rate
        local_window = max(1, round(0.03 / frame_duration))
        minimum_event_frames = max(
            2, round((60.0 / max(tempo_bpm, 1) / 4.0) / frame_duration)
        )
        onset_env = librosa.onset.onset_strength(
            y=signal, sr=sample_rate, hop_length=hop_length
        )
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sample_rate,
            hop_length=hop_length,
            units="frames",
            backtrack=True,
            pre_max=local_window,
            post_max=local_window,
            delta=0.08,
            wait=max(1, minimum_event_frames // 2),
        )
        rms = librosa.feature.rms(y=signal, hop_length=hop_length)[0]
        rms_delta = np.diff(rms, prepend=rms[0])
        positive_delta = rms_delta[rms_delta > 0]
        attack_threshold = (
            float(np.percentile(positive_delta, 85))
            if len(positive_delta)
            else float("inf")
        )
        attack_frames = sparsify_frames(
            np.where(rms_delta >= attack_threshold)[0],
            min_gap_frames=minimum_event_frames,
        )
        return onset_env, onset_frames, attack_frames

    def _extract_segments(
        self,
        signal: np.ndarray,
        sample_rate: int,
        onset_env: np.ndarray,
        onset_frames: np.ndarray,
        attack_frames: np.ndarray,
        *,
        hop_length: int = HOP_LENGTH,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        f0_hz, voiced_flag, voiced_prob = librosa.pyin(
            signal,
            sr=sample_rate,
            hop_length=hop_length,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
        )
        used_fallback = float(np.mean(np.asarray(voiced_flag, dtype=float))) < 0.02
        if used_fallback:
            f0_hz, voiced_flag, voiced_prob = fallback_f0_from_piptrack(
                signal, sample_rate, hop_length
            )

        rms = librosa.feature.rms(y=signal, hop_length=hop_length)[0]
        energy_threshold = float(np.percentile(rms, 20))
        rms_delta = np.diff(rms, prepend=rms[0])
        attack_frame_set = {int(f) for f in np.asarray(attack_frames).tolist()}
        onset_frame_set = {int(f) for f in np.asarray(onset_frames).tolist()}

        midi_track = median_smooth_midi(
            smooth_octave_errors(hz_to_midi_track(f0_hz, voiced_flag)),
            window=3 if hop_length <= MELODY_HOP_LENGTH else 5,
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

        return f0_hz, voiced_flag, voiced_prob, midi_track, segments

    @staticmethod
    def _postprocess_segments(
        segments: list,
        *,
        tempo_bpm: int = 120,
        sample_rate: int = 44100,
        hop_length: int = HOP_LENGTH,
    ) -> list[NoteEvent]:
        frames_per_second = sample_rate / hop_length
        sixteenth_note_seconds = 60.0 / max(tempo_bpm, 1) / 4.0
        sixteenth_note_frames = max(3, round(sixteenth_note_seconds * frames_per_second))
        reattack_min_frames = max(2, round(sixteenth_note_frames * 0.40))
        tiny_rest_max_frames = max(1, round(0.018 * frames_per_second))
        tiny_fragment_max_frames = max(2, round(0.020 * frames_per_second))
        runs = compress_pitch_track(
            segments,
            reattack_min_frames=reattack_min_frames,
            min_note_frames=max(2, round(0.020 * frames_per_second)),
            min_rest_frames=max(1, tiny_rest_max_frames),
        )
        runs = merge_tiny_same_pitch_fragments(
            runs, tiny_fragment_max_frames=tiny_fragment_max_frames
        )
        runs = bridge_same_pitch_across_tiny_rests(
            runs, tiny_rest_max_frames=tiny_rest_max_frames
        )
        return merge_low_confidence_boundary_splits(
            runs,
            weak_boundary_threshold=0.58,
            short_note_max_frames=max(3, sixteenth_note_frames),
        )


def _contains_any_note(runs: list[NoteEvent]) -> bool:
    return any(event.pitch is not None and event.frames > 0 for event in runs)


def _rescue_runs_from_segments(
    segments: list[tuple[float | None, int, bool, float, str, float]],
) -> list[NoteEvent]:
    rescued: list[NoteEvent] = []
    for pitch, frames, _is_attack, confidence, b_source, b_conf in segments:
        if frames > 0 and pitch is not None:
            rescued.append(NoteEvent(
                pitch=pitch,
                frames=frames,
                confidence=confidence,
                reattack_confidence=None,
                boundary_source=b_source,
                boundary_confidence=b_conf,
            ))
    return rescued


def _runs_to_detected_notes(runs: list[NoteEvent], frame_duration: float) -> list[DetectedNote]:
    detected: list[DetectedNote] = []
    elapsed = 0.0
    for pitch, frames, confidence, *_rest in runs:
        duration = max(0.0, int(frames) * frame_duration)
        if pitch is not None and frames > 0:
            detected.append(
                DetectedNote(
                    midi=int(round(float(pitch))),
                    onset_sec=elapsed,
                    duration_sec=max(duration, frame_duration),
                    confidence=float(confidence),
                )
            )
        elapsed += duration
    return detected
