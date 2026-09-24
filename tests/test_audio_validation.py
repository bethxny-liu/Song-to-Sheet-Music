import numpy as np
import pytest
import soundfile as sf

from algo.audio import AudioInputError, load_audio


@pytest.mark.parametrize("kind", ["empty", "silence", "corrupt", "too_long", "nonfinite"])
def test_rejects_unusable_recordings(tmp_path, kind):
    path = tmp_path / "input.wav"
    sample_rate = 22050
    if kind == "corrupt":
        path.write_bytes(b"not an audio recording")
    else:
        samples = np.ones(sample_rate, dtype=np.float32) * 0.1
        if kind == "empty":
            samples = samples[:0]
        elif kind == "silence":
            samples[:] = 0
        elif kind == "nonfinite":
            samples[100] = np.nan
        sf.write(path, samples, sample_rate, subtype="FLOAT")
    with pytest.raises(AudioInputError):
        load_audio(path, max_duration_sec=0.5 if kind == "too_long" else 2)


def test_decodes_stereo_and_preserves_sample_rate(tmp_path):
    path = tmp_path / "input.wav"
    tone = np.sin(2 * np.pi * 440 * np.arange(22050) / 22050)
    sf.write(path, np.column_stack([tone, tone]), 22050)
    signal, sr = load_audio(path, max_duration_sec=1)
    assert sr == 22050
    assert signal.shape == (22050,)
    assert np.max(np.abs(signal)) > 0.9


def test_only_long_leading_silence_is_trimmed(tmp_path):
    path = tmp_path / "lead-in.wav"
    sample_rate = 22050
    lead_in = np.zeros(sample_rate * 2, dtype=np.float32)
    tone = np.sin(2 * np.pi * 440 * np.arange(sample_rate) / sample_rate).astype(np.float32)
    sf.write(path, np.concatenate([lead_in, tone]), sample_rate)
    signal, _ = load_audio(path, trim_leading_silence=True, max_duration_sec=4)
    assert len(signal) < sample_rate * 1.2
