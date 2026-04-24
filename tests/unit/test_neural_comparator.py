from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from neural_approach.embedding_comparator import compare_embeddings


pytestmark = pytest.mark.unit


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
        del seq_a
        del seq_b
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
        del seq_a
        del seq_b
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


def test_compare_embeddings_uses_precomputed_pooled_embeddings(monkeypatch) -> None:
    frames = np.ones((3, 4), dtype=np.float32)
    user = SimpleNamespace(
        frame_embeddings=frames,
        pooled_embedding=np.array([1.0, 0.0], dtype=np.float32),
    )
    reference = SimpleNamespace(
        frame_embeddings=frames,
        pooled_embedding=np.array([0.0, 1.0], dtype=np.float32),
    )

    monkeypatch.setattr(
        "neural_approach.embedding_comparator.dtw_ndim.distance",
        lambda seq_a, seq_b, window=None: 0.0,
    )

    result = compare_embeddings(user, reference, metric="cosine")

    assert result.similarity == 0.0
