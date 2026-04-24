from __future__ import annotations

import numpy as np
import pytest

from classic_approach.dtw import dtw_distance
from classic_approach.mfcc_extractor import apply_cmvn, extract_mfcc


pytestmark = pytest.mark.unit


def test_dtw_distance_identical_features_is_zero() -> None:
    features = np.array(
        [
            [0.0, 0.5, 1.0, 0.3],
            [1.0, 1.5, 0.8, 0.2],
        ],
        dtype=np.float32,
    )

    distance = dtw_distance(features, features)
    assert distance == 0.0


def test_dtw_distance_passes_sakoe_chiba_radius(monkeypatch) -> None:
    features = np.array(
        [
            [0.0, 0.2, 0.4],
            [1.0, 0.9, 0.8],
        ],
        dtype=np.float32,
    )
    captured: dict[str, int | None] = {}

    def fake_distance(seq_a, seq_b, window=None):
        del seq_a
        del seq_b
        captured["window"] = window
        return 0.0

    monkeypatch.setattr("classic_approach.dtw.dtw_ndim.distance", fake_distance)

    distance = dtw_distance(features, features, sakoe_chiba_radius=12)

    assert distance == 0.0
    assert captured["window"] == 12


def test_dtw_distance_disables_band_for_non_positive_radius(monkeypatch) -> None:
    features = np.array(
        [
            [0.0, 0.2, 0.4],
            [1.0, 0.9, 0.8],
        ],
        dtype=np.float32,
    )
    captured: dict[str, int | None] = {}

    def fake_distance(seq_a, seq_b, window=None):
        del seq_a
        del seq_b
        captured["window"] = window
        return 0.0

    monkeypatch.setattr("classic_approach.dtw.dtw_ndim.distance", fake_distance)

    distance = dtw_distance(features, features, sakoe_chiba_radius=0)

    assert distance == 0.0
    assert captured["window"] is None


def test_apply_cmvn_rowwise_mean_and_std() -> None:
    mfcc = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 11.0, 12.0, 13.0],
        ],
        dtype=np.float32,
    )

    normalized = apply_cmvn(mfcc)
    row_means = np.mean(normalized, axis=1)
    row_stds = np.std(normalized, axis=1)

    assert np.allclose(row_means, 0.0, atol=1e-6)
    assert np.allclose(row_stds, 1.0, atol=1e-6)


def test_extract_mfcc_appends_delta_and_delta_delta_features() -> None:
    sample_rate = 16_000
    time = np.linspace(0.0, 0.4, int(sample_rate * 0.4), endpoint=False)
    samples = np.sin(2.0 * np.pi * 440.0 * time).astype(np.float32)

    base_features = extract_mfcc(
        samples,
        sample_rate,
        n_mfcc=13,
        use_cmvn=False,
        use_deltas=False,
    )
    dynamic_features = extract_mfcc(
        samples,
        sample_rate,
        n_mfcc=13,
        use_cmvn=False,
        use_deltas=True,
        delta_width=9,
    )

    assert base_features.shape[0] == 13
    assert dynamic_features.shape[0] == 39
    assert dynamic_features.shape[1] == base_features.shape[1]
