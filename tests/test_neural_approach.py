from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from neural_approach.embedding_comparator import compare_embeddings
from neural_approach.pipeline import analyze
from neural_approach.scorer import compute_scoring_result
from neural_approach.wav2vec_extractor import extract_wav2vec_embeddings


TEST_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "test"
REFERENCE_AUDIO = TEST_DATA_DIR / "pronunciation_en_hello.wav"
HELLO_PERFECT_AUDIO = TEST_DATA_DIR / "hello_perfect.mp3"
HELLO_NORMAL_AUDIO = TEST_DATA_DIR / "hello_normal.wav"
HELLO_PROBLEM_AUDIO = TEST_DATA_DIR / "hello_problem.wav"
HELLO_WRONG_WORD_AUDIO = TEST_DATA_DIR / "hello_wrong_word.mp3"
HELLO_EMPTY_AUDIO = TEST_DATA_DIR / "hello_empty.wav"


def _proxy_extract_wav2vec_embeddings(
    samples,
    sample_rate,
    model_name="proxy/wav2vec2",
    device=None,
    hf_token=None,
):
    _ = (device, hf_token)

    frame_size = max(64, int(0.025 * sample_rate))
    hop_size = max(32, int(0.010 * sample_rate))
    window = np.hanning(frame_size).astype(np.float32)

    frames: list[list[float]] = []
    for start in range(0, max(1, len(samples) - frame_size + 1), hop_size):
        chunk = np.asarray(samples[start : start + frame_size], dtype=np.float32)
        if chunk.size < frame_size:
            chunk = np.pad(chunk, (0, frame_size - chunk.size))

        spectrum = np.abs(np.fft.rfft(chunk * window)).astype(np.float32)
        band_edges = np.linspace(1, len(spectrum) - 1, num=17, dtype=int)

        band_features: list[float] = []
        for left, right in zip(band_edges[:-1], band_edges[1:]):
            band = spectrum[left:right]
            value = float(np.log1p(np.mean(band))) if band.size else 0.0
            band_features.append(value)

        frames.append(band_features)

    frame_embeddings = np.asarray(frames, dtype=np.float32)
    pooled_embedding = np.mean(frame_embeddings, axis=0, dtype=np.float32)

    return SimpleNamespace(
        frame_embeddings=frame_embeddings,
        pooled_embedding=pooled_embedding,
        sample_rate=sample_rate,
        model_name=model_name,
        device="cpu",
    )


def _analyze_test_audio(monkeypatch, audio_path: Path):
    monkeypatch.setattr(
        "neural_approach.pipeline.extract_wav2vec_embeddings",
        _proxy_extract_wav2vec_embeddings,
    )
    return analyze(
        user_audio_path=str(audio_path),
        reference_audio_path=str(REFERENCE_AUDIO),
        transcript="hello",
        similarity="cosine",
        model_name="proxy/wav2vec2",
    )


def test_compare_embeddings_cosine_identical_sequences() -> None:
    frames = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    result = compare_embeddings(frames, frames, metric="cosine")

    assert result.metric == "cosine"
    assert np.isclose(result.similarity, 1.0, atol=1e-6)
    assert np.isclose(result.temporal_distance, 0.0, atol=1e-6)


def test_compare_embeddings_invalid_metric() -> None:
    frames = np.ones((3, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        compare_embeddings(frames, frames, metric="manhattan")


def test_compute_scoring_result_verdict_thresholds() -> None:
    good = compute_scoring_result(1.0, 0.0, metric="cosine", model_name="m")
    acceptable = compute_scoring_result(0.4, 0.1, metric="cosine", model_name="m")
    weak = compute_scoring_result(-0.9, 0.8, metric="cosine", model_name="m")

    assert good.verdict == "хорошо"
    assert acceptable.verdict == "удовлетворительно"
    assert weak.verdict == "неудовлетворительно"


def test_neural_analyze_uses_expected_test_data_files() -> None:
    assert TEST_DATA_DIR.exists()
    assert REFERENCE_AUDIO.exists()
    assert HELLO_PERFECT_AUDIO.exists()
    assert HELLO_NORMAL_AUDIO.exists()
    assert HELLO_PROBLEM_AUDIO.exists()
    assert HELLO_WRONG_WORD_AUDIO.exists()
    assert HELLO_EMPTY_AUDIO.exists()


def test_neural_analyze_hello_perfect_is_near_reference(monkeypatch) -> None:
    result = _analyze_test_audio(monkeypatch, HELLO_PERFECT_AUDIO)

    assert result.pronunciation_score > 90.0
    assert result.verdict == "хорошо"
    assert result.problematic_phonemes == []


def test_neural_analyze_hello_normal_is_mid_quality(monkeypatch) -> None:
    result = _analyze_test_audio(monkeypatch, HELLO_NORMAL_AUDIO)

    assert 40.0 <= result.pronunciation_score < 90.0
    assert result.verdict == "удовлетворительно"


def test_neural_analyze_hello_problem_is_low_quality(monkeypatch) -> None:
    result = _analyze_test_audio(monkeypatch, HELLO_PROBLEM_AUDIO)

    assert result.pronunciation_score < 45.0
    assert result.verdict == "неудовлетворительно"


def test_neural_analyze_hello_wrong_word_is_low(monkeypatch) -> None:
    result = _analyze_test_audio(monkeypatch, HELLO_WRONG_WORD_AUDIO)

    assert result.pronunciation_score < 25.0
    assert result.verdict == "неудовлетворительно"


def test_neural_analyze_real_audio_ordering(monkeypatch) -> None:
    perfect = _analyze_test_audio(monkeypatch, HELLO_PERFECT_AUDIO)
    normal = _analyze_test_audio(monkeypatch, HELLO_NORMAL_AUDIO)
    wrong_word = _analyze_test_audio(monkeypatch, HELLO_WRONG_WORD_AUDIO)
    problem = _analyze_test_audio(monkeypatch, HELLO_PROBLEM_AUDIO)

    assert perfect.pronunciation_score > normal.pronunciation_score > problem.pronunciation_score
    assert problem.pronunciation_score > wrong_word.pronunciation_score


def test_neural_analyze_hello_empty_file_is_marked_empty_audio(monkeypatch) -> None:
    monkeypatch.setattr(
        "neural_approach.pipeline.extract_wav2vec_embeddings",
        _proxy_extract_wav2vec_embeddings,
    )
    result = analyze(
        user_audio_path=str(HELLO_EMPTY_AUDIO),
        reference_audio_path=str(REFERENCE_AUDIO),
        transcript="hello",
        similarity="cosine",
        model_name="proxy/wav2vec2",
    )

    assert result.pronunciation_score == 0.0
    assert result.status == "empty_audio"
    assert result.reason == "insufficient_speech"


def test_neural_analyze_multiple_references_aggregates_results(monkeypatch) -> None:
    monkeypatch.setattr(
        "neural_approach.pipeline.extract_wav2vec_embeddings",
        _proxy_extract_wav2vec_embeddings,
    )

    single = analyze(
        user_audio_path=str(HELLO_NORMAL_AUDIO),
        reference_audio_path=str(REFERENCE_AUDIO),
        transcript="hello",
        similarity="cosine",
        model_name="proxy/wav2vec2",
    )
    multi = analyze(
        user_audio_path=str(HELLO_NORMAL_AUDIO),
        reference_audio_path=[str(REFERENCE_AUDIO), str(REFERENCE_AUDIO)],
        transcript="hello",
        similarity="cosine",
        model_name="proxy/wav2vec2",
    )

    assert np.isclose(multi.pronunciation_score, single.pronunciation_score, atol=1e-6)
    assert multi.verdict == single.verdict


def test_neural_analyze_empty_audio_returns_empty_audio_status() -> None:
    silent = np.zeros(16000, dtype=np.float32)

    with NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, silent, samplerate=16000)
        result = analyze(
            user_audio_path=tmp.name,
            reference_audio_path=str(REFERENCE_AUDIO),
            transcript="hello",
            similarity="cosine",
            model_name="proxy/wav2vec2",
        )

    assert result.pronunciation_score == 0.0
    assert result.status == "empty_audio"
    assert result.reason == "insufficient_speech"


def test_neural_analyze_nonword_tone_returns_empty_audio_status() -> None:
    sample_rate = 16000
    duration_sec = 1.0
    t = np.arange(int(sample_rate * duration_sec), dtype=np.float32) / float(sample_rate)
    tone = (0.2 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)

    with NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, tone, samplerate=sample_rate)
        result = analyze(
            user_audio_path=tmp.name,
            reference_audio_path=str(REFERENCE_AUDIO),
            transcript="hello",
            similarity="cosine",
            model_name="proxy/wav2vec2",
        )

    assert result.pronunciation_score == 0.0
    assert result.status == "empty_audio"
    assert result.reason == "insufficient_speech"


def test_extract_wav2vec_embeddings_input_validation() -> None:
    with pytest.raises(ValueError):
        extract_wav2vec_embeddings(np.ones(100, dtype=np.float32), sample_rate=8000)

    with pytest.raises(ValueError):
        extract_wav2vec_embeddings(np.array([], dtype=np.float32), sample_rate=16000)


def test_extract_wav2vec_embeddings_with_mocked_model(monkeypatch) -> None:
    class FakeFeatureExtractor:
        def __call__(self, speech, sampling_rate, return_tensors, return_attention_mask):
            _ = (return_tensors, return_attention_mask)
            assert sampling_rate == 16000
            input_values = torch.tensor(speech, dtype=torch.float32).unsqueeze(0)
            attention_mask = torch.ones_like(input_values, dtype=torch.long)
            return {"input_values": input_values, "attention_mask": attention_mask}

    class FakeModel:
        def __call__(self, input_values, attention_mask=None):
            _ = attention_mask
            hidden = input_values.unsqueeze(-1).repeat(1, 1, 4)
            return SimpleNamespace(last_hidden_state=hidden)

    monkeypatch.setattr(
        "neural_approach.wav2vec_extractor._resolve_device",
        lambda _device=None: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "neural_approach.wav2vec_extractor._load_model_bundle",
        lambda _model_name, _device_str, _hf_token: (FakeFeatureExtractor(), FakeModel()),
    )

    samples = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    result = extract_wav2vec_embeddings(samples, sample_rate=16000, model_name="fake/model")

    assert result.frame_embeddings.shape == (4, 4)
    assert result.pooled_embedding.shape == (4,)
    assert result.sample_rate == 16000
    assert result.model_name == "fake/model"
    assert result.device == "cpu"
