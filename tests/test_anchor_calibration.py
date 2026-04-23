from __future__ import annotations

from pathlib import Path

from scoring.anchor_calibration import (
    AnchorDistanceProfile,
    build_anchor_distance_profiles,
    fit_sigmoid_from_anchor_profiles,
    score_from_anchor_profile,
    SigmoidCalibrationParams,
    fit_sigmoid_from_anchor_distances,
    get_word_anchor_set,
    list_anchor_words,
    should_force_zero_by_zero_anchors,
    sigmoid_score,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF")


def test_get_word_anchor_set_collects_word_and_global_empty(tmp_path: Path) -> None:
    _touch(tmp_path / "happy_perfect" / "p1.wav")
    _touch(tmp_path / "happy_perfect" / "p2.wav")
    _touch(tmp_path / "happy_fail" / "f1.wav")
    _touch(tmp_path / "happy_moderate" / "m1.wav")
    _touch(tmp_path / "_empty_word" / "e1.wav")
    _touch(tmp_path / "world_perfect" / "wp.wav")

    anchors = get_word_anchor_set("happy", anchor_root=tmp_path)

    assert len(anchors.perfect_paths) == 2
    assert len(anchors.fail_paths) == 1
    assert len(anchors.wrong_paths) == 1
    assert len(anchors.moderate_paths) == 1
    assert len(anchors.empty_paths) == 1
    assert len(anchors.zero_paths) == 2
    assert anchors.has_required_anchors is True


def test_list_anchor_words_uses_perfect_quality(tmp_path: Path) -> None:
    _touch(tmp_path / "happy_perfect" / "p1.wav")
    _touch(tmp_path / "hello_fail" / "f1.wav")

    words = list_anchor_words(anchor_root=tmp_path)

    assert words == ["happy"]


def test_multi_anchor_sigmoid_keeps_anchor_order() -> None:
    perfect_profiles = [
        AnchorDistanceProfile(0.10, 0.18, 0.26),
        AnchorDistanceProfile(0.11, 0.17, 0.28),
        AnchorDistanceProfile(0.09, 0.16, 0.24),
    ]
    moderate_profiles = [
        AnchorDistanceProfile(0.16, 0.12, 0.20),
        AnchorDistanceProfile(0.18, 0.13, 0.22),
    ]
    fail_profiles = [
        AnchorDistanceProfile(0.27, 0.20, 0.11),
        AnchorDistanceProfile(0.29, 0.21, 0.12),
        AnchorDistanceProfile(0.25, 0.19, 0.10),
    ]

    params = fit_sigmoid_from_anchor_profiles(
        perfect_profiles=perfect_profiles,
        moderate_profiles=moderate_profiles,
        fail_profiles=fail_profiles,
    )

    perfect_score = score_from_anchor_profile(perfect_profiles[0], params)
    moderate_score = score_from_anchor_profile(moderate_profiles[0], params)
    fail_score = score_from_anchor_profile(fail_profiles[0], params)

    assert perfect_score > moderate_score > fail_score
    assert 1.0 <= fail_score <= 99.0
    assert 1.0 <= perfect_score <= 99.0


def test_build_anchor_distance_profiles_uses_leave_one_out() -> None:
    perfect_items = ["p1", "p2"]
    moderate_items = ["m1"]
    fail_items = ["f1", "f2"]

    distances = {
        ("p1", "p2"): 0.10,
        ("p1", "m1"): 0.18,
        ("p2", "m1"): 0.16,
        ("p1", "f1"): 0.27,
        ("p1", "f2"): 0.29,
        ("p2", "f1"): 0.25,
        ("p2", "f2"): 0.24,
        ("m1", "f1"): 0.19,
        ("m1", "f2"): 0.21,
        ("f1", "f2"): 0.08,
    }

    def _distance(left: str, right: str) -> float:
        key = (left, right) if (left, right) in distances else (right, left)
        return distances[key]

    perfect_profiles, moderate_profiles, fail_profiles = build_anchor_distance_profiles(
        perfect_items=perfect_items,
        moderate_items=moderate_items,
        fail_items=fail_items,
        distance_fn=_distance,
    )

    assert len(perfect_profiles) == 2
    assert len(moderate_profiles) == 1
    assert len(fail_profiles) == 2
    assert perfect_profiles[0].perfect_distance == 0.10
    assert fail_profiles[0].fail_distance == 0.08


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


def test_sigmoid_calibration_limits_slope_for_small_anchor_gap() -> None:
    params = fit_sigmoid_from_anchor_distances(
        distances_100=[0.160, 0.165, 0.170],
        distances_0=[0.190, 0.195, 0.200],
    )

    assert params.a <= 80.0 + 1e-9
    assert (params.d0 - params.d100) >= 0.09


def test_should_force_zero_by_zero_anchors_uses_margin_and_midpoint() -> None:
    params = SigmoidCalibrationParams(
        d100=0.15,
        d0=0.25,
        a=77.0,
        b=0.20,
        epsilon=0.02,
    )

    assert should_force_zero_by_zero_anchors(
        user_distance=0.24,
        user_zero_distance=0.20,
        calibration_params=params,
    )
    assert not should_force_zero_by_zero_anchors(
        user_distance=0.18,
        user_zero_distance=0.14,
        calibration_params=params,
    )
    assert not should_force_zero_by_zero_anchors(
        user_distance=0.24,
        user_zero_distance=0.23,
        calibration_params=params,
    )
