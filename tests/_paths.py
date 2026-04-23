from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA_DIR = PROJECT_ROOT / "data" / "test"
HAPPY_PERFECT_AUDIO = TEST_DATA_DIR / "happy_perfect.wav"
HAPPY_NORMAL_AUDIO = TEST_DATA_DIR / "happy_normal.wav"
HAPPY_PROBLEM_AUDIO = TEST_DATA_DIR / "happy_problem.wav"
HAPPY_WRONG_WORD_AUDIO = TEST_DATA_DIR / "happy_wrong_word.mp3"
HAPPY_EMPTY_AUDIO = TEST_DATA_DIR / "happy_empty.wav"
