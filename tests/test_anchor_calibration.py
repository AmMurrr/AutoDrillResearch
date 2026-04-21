from __future__ import annotations

from pathlib import Path

from scoring.anchor_calibration import (
    fit_sigmoid_from_anchor_distances,
    get_word_anchor_set,
    list_anchor_words,
    sigmoid_score,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF")


def test_get_word_anchor_set_collects_word_and_global_empty(tmp_path: Path) -> None:
    _touch(tmp_path / "happy_perfect" / "p1.wav")
    _touch(tmp_path / "happy_perfect" / "p2.wav")
    _touch(tmp_path / "happy_wrong" / "w1.wav")
    _touch(tmp_path / "happy_moderate" / "m1.wav")
    _touch(tmp_path / "_empty_word" / "e1.wav")
    _touch(tmp_path / "world_perfect" / "wp.wav")

    anchors = get_word_anchor_set("happy", anchor_root=tmp_path)

    assert len(anchors.perfect_paths) == 2
    assert len(anchors.wrong_paths) == 1
    assert len(anchors.moderate_paths) == 1
    assert len(anchors.empty_paths) == 1
    assert len(anchors.zero_paths) == 3
    assert anchors.has_required_anchors is True


def test_list_anchor_words_uses_perfect_quality(tmp_path: Path) -> None:
    _touch(tmp_path / "happy_perfect" / "p1.wav")
    _touch(tmp_path / "hello_wrong" / "w1.wav")

    words = list_anchor_words(anchor_root=tmp_path)

    assert words == ["happy"]


def test_sigmoid_calibration_maps_anchors_to_score_edges() -> None:
    params = fit_sigmoid_from_anchor_distances(
        distances_100=[0.10, 0.12, 0.11],
        distances_0=[0.50, 0.60, 0.55],
        epsilon=0.02,
    )

    score_for_d100 = sigmoid_score(params.d100, params)
    score_for_d0 = sigmoid_score(params.d0, params)

    assert score_for_d100 > 95.0
    assert score_for_d0 < 5.0


def test_sigmoid_calibration_handles_overlap() -> None:
    params = fit_sigmoid_from_anchor_distances(
        distances_100=[0.30, 0.31],
        distances_0=[0.30, 0.29],
    )

    assert params.d0 > params.d100
