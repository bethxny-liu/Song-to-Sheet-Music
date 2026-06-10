from __future__ import annotations

from pathlib import Path
import re
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from music21 import harmony
from algo.pipeline import AudioToSheetPipeline, PipelineOptions

from backend.app.schemas import (
    ChordEvent,
    ConversionOptions,
    ConversionResult,
    KeyEstimate,
    NoteConfidence,
)


class ConversionService:
    def __init__(self, storage_dir: Path) -> None:
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline = AudioToSheetPipeline()

    def convert(
        self, input_file_path: Path, options: ConversionOptions, base_url: str
    ) -> ConversionResult:
        job_id = uuid4().hex
        output_dir = self.storage_dir / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        pipeline_result = self.pipeline.run(
            input_file_path,
            PipelineOptions(
                title=options.title,
                composer=options.composer,
                tempo_bpm=options.tempo_bpm,
                instrument_name=options.instrument_name,
                layout=options.layout,
                isolate_piano=options.isolate_piano,
            ),
            work_dir=output_dir,
        )

        xml_path = output_dir / "score.musicxml"
        txt_path = output_dir / "score.txt"
        chart_path = output_dir / "pitch_chart.png"
        json_path = output_dir / "result.json"

        chords_stripped = self._write_musicxml_with_chord_fallback(pipeline_result.score, xml_path)
        self._strip_courtesy_naturals(xml_path)
        pipeline_result.score.write("text", fp=str(txt_path))
        self._save_pitch_chart(
            pipeline_result.pitch_times_sec,
            pipeline_result.pitch_midi,
            chart_path,
            options.title,
        )

        artifacts = {
            "musicxml_url": f"{base_url}/artifacts/{job_id}/score.musicxml",
            "text_url": f"{base_url}/artifacts/{job_id}/score.txt",
            "pitch_chart_url": f"{base_url}/artifacts/{job_id}/pitch_chart.png",
            "result_json_url": f"{base_url}/artifacts/{job_id}/result.json",
        }
        result = ConversionResult(
            job_id=job_id,
            title=options.title,
            composer=options.composer,
            tempo_bpm=options.tempo_bpm,
            instrument_name=options.instrument_name,
            estimated_key=pipeline_result.estimated_key,
            estimated_key_candidates=[
                KeyEstimate(key=key_name, score=score)
                for key_name, score in pipeline_result.estimated_key_candidates
            ],
            note_confidences=[NoteConfidence(**event) for event in pipeline_result.note_confidences],
            chord_events=(
                [] if chords_stripped else [ChordEvent(**event) for event in pipeline_result.chord_events]
            ),
            note_count=pipeline_result.note_count,
            transcription_engine=pipeline_result.transcription_engine,
            preprocessing=pipeline_result.preprocessing,
            artifacts=artifacts,
        )
        json_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return result

    def _write_musicxml_with_chord_fallback(self, score, xml_path: Path) -> bool:
        """Write MusicXML; if chord symbols break export, retry without them."""
        try:
            score.write("musicxml", fp=str(xml_path))
            return False
        except (ValueError, OSError, TypeError):
            chord_symbols = list(score.recurse().getElementsByClass(harmony.ChordSymbol))
            for symbol in chord_symbols:
                parent = symbol.activeSite
                if parent is not None:
                    parent.remove(symbol)
            score.write("musicxml", fp=str(xml_path))
            return True

    def _strip_courtesy_naturals(self, xml_path: Path) -> None:
        """Remove explicit natural signs from MusicXML display output."""
        xml_text = xml_path.read_text(encoding="utf-8")
        cleaned = re.sub(r"\n\s*<accidental>natural</accidental>", "", xml_text)
        if cleaned != xml_text:
            xml_path.write_text(cleaned, encoding="utf-8")

    def _save_pitch_chart(
        self,
        pitch_times_sec: list[float],
        pitch_midi: list[float | None],
        chart_path: Path,
        title: str,
    ) -> None:
        y_values = [float("nan") if v is None else v for v in pitch_midi]
        plt.figure(figsize=(12, 4))
        plt.plot(pitch_times_sec, y_values, linewidth=1.0)
        plt.title(f"Pitch Track (MIDI) - {title}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("MIDI Pitch")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(chart_path)
        plt.close()
