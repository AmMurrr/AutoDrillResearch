from __future__ import annotations

from asr.vosk import compare_with_expected_text


def test_compare_with_expected_text_exact_match() -> None:
    result = compare_with_expected_text("world", "world")

    assert result.is_match is True
    assert result.expected_text == "world"
    assert result.recognized_text == "world"


def test_compare_with_expected_text_allows_leading_the() -> None:
    result = compare_with_expected_text("the world", "world")

    assert result.is_match is True


def test_compare_with_expected_text_allows_missing_the_in_expected() -> None:
    result = compare_with_expected_text("world", "the world")

    assert result.is_match is True


def test_compare_with_expected_text_rejects_different_word() -> None:
    result = compare_with_expected_text("the world", "word")

    assert result.is_match is False


def test_compare_with_expected_text_keeps_single_the_as_word() -> None:
    result = compare_with_expected_text("the", "world")

    assert result.is_match is False
