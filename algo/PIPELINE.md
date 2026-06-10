# Audio-to-Sheet Pipeline Flow

```mermaid
flowchart TD
    A[Upload Audio File] --> B[Load waveform with librosa]
    B --> C[Compute onset envelope]
    C --> D[Detect tempo + beat frames]
    C --> E[Detect onset frames]
    B --> F[Run pYIN pitch detection]
    B --> G[Compute RMS energy]

    F --> H[Convert Hz to MIDI]
    H --> I[Median smooth MIDI track]
    I --> J[Fix octave jump artifacts]

    D --> K[Build segmentation boundaries]
    E --> K
    J --> L[Segment pitch by boundaries]
    G --> L

    L --> M[Classify each segment as note or rest]
    M --> N[Merge short noisy segments]
    N --> O[Estimate key from weighted pitch classes]
    N --> P[Convert segment durations to note lengths]

    O --> Q[Create music21 Score]
    P --> Q
    Q --> R[Insert tempo, meter, key, instrument]
    R --> S[Write MusicXML + text score]
    J --> T[Create pitch chart image]
    S --> U[Return artifact URLs + metadata JSON]
    T --> U
```

## Notes

- **Rests** come from low-energy / low-confidence pitch segments.
- **Tempo** comes from beat tracking with fallback to user tempo input.
- **Key** is estimated by weighted pitch-class matching over major keys.
- **Rhythm** is quantized to a sixteenth-note grid for cleaner notation.
