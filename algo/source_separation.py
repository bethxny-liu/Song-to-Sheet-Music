"""Optional piano stem isolation via Demucs (for mixed vocal+piano recordings)."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def is_available() -> bool:
    try:
        import demucs  # noqa: F401

        return shutil.which("ffmpeg") is not None
    except ImportError:
        return False


def isolate_piano(input_path: Path, output_dir: Path) -> Path:
    """Extract a piano stem with Demucs two-stems mode.

    Returns the path to ``piano.wav``. Requires ``pip install demucs`` and ffmpeg.
    """
    if not is_available():
        raise RuntimeError(
            "Demucs is not available. Install with: pip install -r requirements-ml.txt "
            "(also requires ffmpeg on your PATH)."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "demucs",
        "--two-stems",
        "piano",
        "-n",
        "htdemucs",
        "-o",
        str(output_dir),
        str(input_path),
    ]
    logger.info("Running Demucs piano isolation: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    track_name = input_path.stem
    piano_wav = output_dir / "htdemucs" / track_name / "piano.wav"
    if not piano_wav.exists():
        raise FileNotFoundError(f"Demucs piano stem not found at {piano_wav}")
    return piano_wav
