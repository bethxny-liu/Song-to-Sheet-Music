# Transcription Evaluation Report

Generated: `2026-09-07T13:35:13.087957+00:00`

**Overall:** PASS

## Regression benchmarks

| Fixture | Layout | Engine | F1 | Pitch acc. | Notes | Status |
|---------|--------|--------|-----|------------|-------|--------|
| c_major_scale | melody | pyin | 0.842 | 1.000 | 11 | pass |
| repeated_c | melody | pyin | 1.000 | 1.000 | 3 | pass |

## Engine comparison (melody vs grand)

| Fixture | Melody F1 | Grand F1 | Δ F1 | Grand engine |
|---------|-----------|----------|------|--------------|
| c_major_scale | 0.842 | 1.000 | +0.158 | basic_pitch |
| repeated_c | 1.000 | 1.000 | +0.000 | basic_pitch |

## Controlled tempo benchmark

Same eight-note phrase at each tempo. Baseline is raw pYIN frame collapse.

| Input | 60 BPM | 90 BPM | 120 BPM | 150 BPM | 180 BPM |
|-------|-------:|-------:|-------:|-------:|-------:|
| pYIN baseline F1 | 0.800 | 0.778 | 0.857 | 0.857 | 0.857 |
| Pipeline F1 | 1.000 | 1.000 | 1.000 | 0.875 | 0.933 |
| Δ F1 | +0.200 | +0.222 | +0.143 | +0.018 | +0.076 |
| Pipeline precision | 1.000 | 1.000 | 1.000 | 0.875 | 1.000 |
| Pipeline recall | 1.000 | 1.000 | 1.000 | 0.875 | 0.875 |
| Pipeline onset F1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.933 |

## Polyphony benchmark

Basic Pitch, grand staff.

| Texture | 60 BPM | 90 BPM | 120 BPM | 150 BPM | 180 BPM |
|---------|-------:|-------:|-------:|-------:|-------:|
| two note harmony F1 | 0.577 | 0.640 | 0.789 | 0.857 | 0.688 |
| chords F1 | 0.667 | 0.741 | 0.866 | 0.762 | 0.642 |

| Texture | Tempo | Engine | Precision | Recall | F1 | Onset F1 |
|---------|------:|--------|----------:|-------:|---:|---------:|
| two note harmony | 60 BPM | basic_pitch | 0.417 | 0.938 | 0.577 | 0.615 |
| two note harmony | 90 BPM | basic_pitch | 0.471 | 1.000 | 0.640 | 0.640 |
| two note harmony | 120 BPM | basic_pitch | 0.682 | 0.938 | 0.789 | 0.842 |
| two note harmony | 150 BPM | basic_pitch | 0.789 | 0.938 | 0.857 | 0.914 |
| two note harmony | 180 BPM | basic_pitch | 0.688 | 0.688 | 0.688 | 1.000 |
| chords | 60 BPM | basic_pitch | 0.500 | 1.000 | 0.667 | 0.667 |
| chords | 90 BPM | basic_pitch | 0.612 | 0.938 | 0.741 | 0.790 |
| chords | 120 BPM | basic_pitch | 0.829 | 0.906 | 0.866 | 0.955 |
| chords | 150 BPM | basic_pitch | 0.774 | 0.750 | 0.762 | 0.984 |
| chords | 180 BPM | basic_pitch | 0.810 | 0.531 | 0.642 | 0.792 |

## Metrics

- **F1**: note overlap precision/recall (mir_eval)
- **Onset F1**: onset matching only
- **Pitch accuracy**: cents tolerance on onset-matched notes
- Thresholds: `tests/fixtures/baseline_targets.json`
