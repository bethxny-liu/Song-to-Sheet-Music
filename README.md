<img width="991" height="192" alt="image" src="https://github.com/user-attachments/assets/1463e626-efa2-4112-8d38-3685a985ba61" />

Audio-to-sheet-music transcription for clear, single-line melodies.

## Overview

**Base models**

- [pYIN](https://librosa.org/doc/latest/generated/librosa.pyin.html) for melody pitch estimation

**Custom pipeline**

- onset and RMS-attack detection
- tempo-aware segmentation and minimum note duration
- median pitch smoothing and octave-error correction
- fallback spectral pitch tracking
- note-event post-processing (reattack splits, gap bridging)
- key estimation and music21 score generation
- mir_eval benchmark vs a raw pYIN baseline

pYIN is the public transcription engine. The rest is segmentation, post-processing, score building, and evaluation around it.

## Polyphonic notes (optional)

`algo/polyphonic.py` wraps Basic Pitch and returns overlapping timed notes. It
is separate from the web app and does not create chords or a grand staff.

Install its dependencies only if you want to run it:

```bash
.venv/bin/pip install -r requirements-polyphonic.txt
```

Use it directly when comparing note events, for example:

```python
from algo.polyphonic import transcribe

notes = transcribe("recording.mp3")
for note in notes:
    print(note.midi, note.onset_sec, note.duration_sec, note.confidence)
```

## Architecture

Single staff (melody):

```
Audio → pitch / onset extraction → segmentation / post-processing
     → key / rhythm inference → score generation → PDF
```

## Demo

*Mary Had a Little Lamb* — single staff, 90 BPM, detected key **C major**

**Input:** [mary-had-a-little-lamb.mp3](docs/demo/mary-had-a-little-lamb.mp3)

| Pitch detection | Sheet music |
|---|---|
| ![Pitch chart](docs/demo/pitch-chart.png) | ![Sheet music](docs/demo/sheet-music.png) |

**Output:** [mary-had-a-little-lamb.pdf](docs/demo/mary-had-a-little-lamb.pdf)

## Evaluation

Same eight-note phrase at 60–180 BPM, scored with mir_eval note-level F1. Baseline is raw pYIN (adjacent pitch frames collapsed, no onset post-processing). Full tables: [docs/EVALUATION.md](docs/EVALUATION.md).

| Tempo | Raw pYIN F1 | Pipeline F1 |
|------:|------------:|------------:|
| 60 | 0.800 | 1.000 |
| 90 | 0.778 | 1.000 |
| 120 | 0.857 | 1.000 |
| 150 | 0.857 | 0.875 |
| 180 | 0.857 | 0.933 |

Average F1: **0.830 → 0.962** (~16% relative). Perfect through 120 BPM; drops at 150–180.

The controlled set is **synthetic sine tones**, not real recordings.

## Known limitations

- Best on clean, simple monophonic melodies
- Accuracy falls on rapid passages
- The tempo benchmark does not measure timbre, noise, sustain, vibrato, or expressive timing

## Run locally

**Prerequisites:** Python 3.11+, Node 20+

```bash
make setup      # once
make services   # terminal 1 → http://localhost:8000
make client     # terminal 2 → http://localhost:3000
```

Uploads are limited to **25 MB and 240 seconds**. Tempo must be **40–240 BPM**.
Invalid, empty, and silent audio receive an explanation instead of starting a conversion.

## Tests

```bash
make test    # unit + baseline tests
make eval    # MIR evaluation report → docs/EVALUATION.md
cd client && npm test  # HTTP client error handling
```
