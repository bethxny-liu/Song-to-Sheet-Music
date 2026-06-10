# Transcription Evaluation Report

Generated: `2026-06-10T05:08:20.043929+00:00`

**Overall:** PASS

## Regression benchmarks

| Fixture | Layout | Engine | F1 | Pitch acc. | Notes | Status |
|---------|--------|--------|-----|------------|-------|--------|
| c_major_scale | melody | pyin | 0.625 | 1.000 | 8 | pass |
| repeated_c | melody | pyin | 0.333 | 1.000 | 3 | pass |

## Engine comparison (melody vs grand)

| Fixture | Melody F1 | Grand F1 | Δ F1 | Grand engine |
|---------|-----------|----------|------|--------------|
| c_major_scale | 0.625 | 0.125 | -0.500 | basic_pitch |
| repeated_c | 0.333 | 0.333 | +0.000 | basic_pitch |

## Metrics

- **F1**: note overlap precision/recall (mir_eval)
- **Pitch accuracy**: cents tolerance on onset-matched notes
- Thresholds: `tests/fixtures/baseline_targets.json`
