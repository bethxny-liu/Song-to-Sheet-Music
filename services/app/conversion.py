from __future__ import annotations

from pathlib import Path
import shutil
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from algo.pipeline import AudioToSheetPipeline, PipelineOptions

from services.app.config import MAX_AUDIO_SECONDS
from services.app.schemas import (
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

        try:
            pipeline_result = self.pipeline.run(
                input_file_path,
                PipelineOptions(
                    title=options.title,
                    composer=options.composer,
                    tempo_bpm=options.tempo_bpm,
                    instrument_name=options.instrument_name,
                ),
                max_duration_sec=MAX_AUDIO_SECONDS,
            )

            xml_path = output_dir / "score.musicxml"
            txt_path = output_dir / "score.txt"
            chart_path = output_dir / "pitch_chart.png"
            json_path = output_dir / "result.json"

            pipeline_result.score.write("musicxml", fp=str(xml_path))
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
                note_count=pipeline_result.note_count,
                artifacts=artifacts,
            )
            json_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            return result
        except Exception:
            shutil.rmtree(output_dir, ignore_errors=True)
            raise

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
