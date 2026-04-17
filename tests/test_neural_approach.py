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
from neural_approach.wav2vec_extractor import (
    DEFAULT_MODEL_NAME,
    HF_TOKEN_ENV_VAR,
    extract_wav2vec_embeddings,
    resolve_hf_token,
)


TEST_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "test"
REFERENCE_AUDIO = TEST_DATA_DIR / "pronunciation_en_hello.wav"
HELLO_PERFECT_AUDIO = TEST_DATA_DIR / "hello_perfect.mp3"
HELLO_NORMAL_AUDIO = TEST_DATA_DIR / "hello_normal.wav"
HELLO_PROBLEM_AUDIO = TEST_DATA_DIR / "hello_problem.wav"
HELLO_WRONG_WORD_AUDIO = TEST_DATA_DIR / "hello_wrong_word.mp3"
HELLO_EMPTY_AUDIO = TEST_DATA_DIR / "hello_empty.wav"


_ANALYZE_CACHE: dict[str, object] = {}


@pytest.fixture(autouse=True)
def _mock_vosk_word_gate(monkeypatch):
    monkeypatch.setattr(
        "neural_approach.pipeline.check_expected_text_for_preprocessed_audio",
        lambda samples, sample_rate, expected_text: SimpleNamespace(
            is_match=True,
            expected_text=(expected_text or "").strip().lower(),
            recognized_text=(expected_text or "").strip().lower(),
        ),
    )


def _require_hf_token() -> str:
    token = resolve_hf_token(None)
    if token:
        return token

    pytest.skip(
        "Neural integration tests require HF token in env. "
        f"Set {HF_TOKEN_ENV_VAR}"
    )


def _analyze_test_audio(audio_path: Path):
    cache_key = str(audio_path.resolve())
    cached = _ANALYZE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    result = analyze(
        user_audio_path=str(audio_path),
        reference_audio_path=str(REFERENCE_AUDIO),
        transcript="hello",
        similarity="cosine",
        model_name=DEFAULT_MODEL_NAME,
        hf_token=_require_hf_token(),
    )
    _ANALYZE_CACHE[cache_key] = result
    return result


def test_resolve_hf_token_explicit_overrides_env(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "env-token")
    assert resolve_hf_token("explicit-token") == "explicit-token"


def test_resolve_hf_token_reads_hf_token_env(monkeypatch) -> None:
    monkeypatch.delenv(HF_TOKEN_ENV_VAR, raising=False)
    monkeypatch.setenv(HF_TOKEN_ENV_VAR, "env-hf-token")
    assert resolve_hf_token(None) == "env-hf-token"


def test_resolve_hf_token_reads_dotenv_from_cwd(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv(HF_TOKEN_ENV_VAR, raising=False)
    (tmp_path / ".env").write_text("HF_TOKEN=dotenv-hf-token\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert resolve_hf_token(None) == "dotenv-hf-token"


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


def test_compare_embeddings_passes_sakoe_chiba_radius(monkeypatch) -> None:
    frames = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    captured: dict[str, int | None] = {}

    def fake_distance(seq_a, seq_b, window=None):
        _ = (seq_a, seq_b)
        captured["window"] = window
        return 0.0

    monkeypatch.setattr("neural_approach.embedding_comparator.dtw_ndim.distance", fake_distance)

    result = compare_embeddings(
        frames,
        frames,
        metric="cosine",
        sakoe_chiba_radius=9,
    )

    assert result.metric == "cosine"
    assert captured["window"] == 9


def test_compare_embeddings_disables_band_for_non_positive_radius(monkeypatch) -> None:
    frames = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    captured: dict[str, int | None] = {}

    def fake_distance(seq_a, seq_b, window=None):
        _ = (seq_a, seq_b)
        captured["window"] = window
        return 0.0

    monkeypatch.setattr("neural_approach.embedding_comparator.dtw_ndim.distance", fake_distance)

    compare_embeddings(
        frames,
        frames,
        metric="cosine",
        sakoe_chiba_radius=0,
    )

    assert captured["window"] is None


def test_compare_embeddings_invalid_metric() -> None:
    frames = np.ones((3, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        compare_embeddings(frames, frames, metric="manhattan")


def test_compare_embeddings_euclidean_not_supported() -> None:
    frames = np.ones((3, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        compare_embeddings(frames, frames, metric="euclidean")


def test_neural_analyze_euclidean_not_supported() -> None:
    with pytest.raises(ValueError):
        analyze(
            user_audio_path="unused.wav",
            reference_audio_path="unused.wav",
            transcript="hello",
            similarity="euclidean",
            model_name=DEFAULT_MODEL_NAME,
            hf_token=None,
        )


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


def test_neural_analyze_hello_perfect_is_near_reference() -> None:
    result = _analyze_test_audio(HELLO_PERFECT_AUDIO)

    assert result.pronunciation_score > 70.0
    assert result.verdict in {"хорошо", "удовлетворительно"}
    assert result.problematic_phonemes == []


def test_neural_analyze_hello_normal_is_mid_quality() -> None:
    result = _analyze_test_audio(HELLO_NORMAL_AUDIO)

    assert 40.0 <= result.pronunciation_score < 90.0
    assert result.verdict == "удовлетворительно"


def test_neural_analyze_hello_problem_is_low_quality() -> None:
    result = _analyze_test_audio(HELLO_PROBLEM_AUDIO)

    assert result.pronunciation_score < 45.0
    assert result.verdict == "неудовлетворительно"


def test_neural_analyze_hello_wrong_word_returns_zero_with_vosk(monkeypatch) -> None:
    _ANALYZE_CACHE.pop(str(HELLO_WRONG_WORD_AUDIO.resolve()), None)
    monkeypatch.setattr(
        "neural_approach.pipeline.check_expected_text_for_preprocessed_audio",
        lambda samples, sample_rate, expected_text: SimpleNamespace(
            is_match=False,
            expected_text="hello",
            recognized_text="happy",
        ),
    )

    result = analyze(
        user_audio_path=str(HELLO_WRONG_WORD_AUDIO),
        reference_audio_path=str(REFERENCE_AUDIO),
        transcript="hello",
        similarity="cosine",
        model_name=DEFAULT_MODEL_NAME,
        hf_token=None,
    )

    assert result.pronunciation_score == 0.0
    assert result.status == "wrong_word"
    assert "recognized:happy" in result.reason


def test_neural_analyze_real_audio_ordering() -> None:
    perfect = _analyze_test_audio(HELLO_PERFECT_AUDIO)
    normal = _analyze_test_audio(HELLO_NORMAL_AUDIO)
    problem = _analyze_test_audio(HELLO_PROBLEM_AUDIO)

    assert perfect.pronunciation_score > normal.pronunciation_score > problem.pronunciation_score


def test_neural_analyze_hello_empty_file_is_marked_empty_audio() -> None:
    result = analyze(
        user_audio_path=str(HELLO_EMPTY_AUDIO),
        reference_audio_path=str(REFERENCE_AUDIO),
        transcript="hello",
        similarity="cosine",
        model_name=DEFAULT_MODEL_NAME,
        hf_token=resolve_hf_token(None),
    )

    assert result.pronunciation_score == 0.0
    assert result.status == "empty_audio"
    assert result.reason == "insufficient_speech"


def test_neural_analyze_multiple_references_aggregates_results() -> None:
    hf_token = _require_hf_token()
    single = analyze(
        user_audio_path=str(HELLO_NORMAL_AUDIO),
        reference_audio_path=str(REFERENCE_AUDIO),
        transcript="hello",
        similarity="cosine",
        model_name=DEFAULT_MODEL_NAME,
        hf_token=hf_token,
    )
    multi = analyze(
        user_audio_path=str(HELLO_NORMAL_AUDIO),
        reference_audio_path=[str(REFERENCE_AUDIO), str(REFERENCE_AUDIO)],
        transcript="hello",
        similarity="cosine",
        model_name=DEFAULT_MODEL_NAME,
        hf_token=hf_token,
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
            model_name=DEFAULT_MODEL_NAME,
            hf_token=resolve_hf_token(None),
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
            model_name=DEFAULT_MODEL_NAME,
            hf_token=resolve_hf_token(None),
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
