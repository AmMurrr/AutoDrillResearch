from __future__ import annotations

import json
import os
import shutil
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp
import re

import numpy as np
from app.logging_config import get_logger
import requests
from vosk import KaldiRecognizer, Model, SetLogLevel


VOSK_MODEL_NAME = "vosk-model-small-en-us-0.15"
VOSK_MODEL_ARCHIVE_URL = f"https://alphacephei.com/vosk/models/{VOSK_MODEL_NAME}.zip"
VOSK_MODEL_DIR_ENV_VAR = "VOSK_MODEL_DIR"
DEFAULT_MODEL_BASE_DIR = Path.home() / ".cache" / "diploma-pronunciation" / "vosk"
DOWNLOAD_TIMEOUT_SECONDS = 240
RECOGNIZER_CHUNK_BYTES = 4000
TARGET_SAMPLE_RATE = 16000


_TOKEN_RE = re.compile(r"[a-z0-9']+")
_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: Model | None = None
_MODEL_PATH_CACHE: Path | None = None
logger = get_logger(__name__)


try:
    SetLogLevel(-1)
except Exception:
    # Log-level configuration is best-effort.
    pass


class VoskError(RuntimeError):
    """Raised when Vosk model loading/transcription fails."""


@dataclass(frozen=True)
class TranscriptCheckResult:
    expected_text: str
    recognized_text: str
    is_match: bool


def normalize_text(text: str) -> str:
    tokens = _TOKEN_RE.findall((text or "").lower())
    return " ".join(tokens)


def _extract_text(result_payload: str) -> str:
    try:
        parsed = json.loads(result_payload)
    except json.JSONDecodeError:
        return ""

    text = parsed.get("text", "")
    return text if isinstance(text, str) else ""


def _resolve_model_base_dir(model_base_dir: str | Path | None = None) -> Path:
    if model_base_dir is not None:
        logger.debug("Using explicit Vosk model base dir: %s", model_base_dir)
        return Path(model_base_dir).expanduser().resolve()

    env_model_dir = os.environ.get(VOSK_MODEL_DIR_ENV_VAR, "").strip()
    if env_model_dir:
        logger.debug("Using Vosk model dir from environment: %s", env_model_dir)
        return Path(env_model_dir).expanduser().resolve()

    return DEFAULT_MODEL_BASE_DIR.resolve()


def _resolve_model_path(model_base_dir: str | Path | None = None) -> Path:
    base_dir = _resolve_model_base_dir(model_base_dir)
    return base_dir / VOSK_MODEL_NAME


def _download_model_archive(archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading Vosk model archive to %s", archive_path)
    with requests.get(
        VOSK_MODEL_ARCHIVE_URL,
        stream=True,
        timeout=DOWNLOAD_TIMEOUT_SECONDS,
    ) as response:
        response.raise_for_status()
        with archive_path.open("wb") as archive_file:
            for chunk in response.iter_content(chunk_size=1024 * 512):
                if chunk:
                    archive_file.write(chunk)
    logger.info("Vosk model archive downloaded: %s", archive_path)


def _safe_extract_zip(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting Vosk archive %s to %s", archive_path, destination)
    with zipfile.ZipFile(archive_path, mode="r") as zip_file:
        for member in zip_file.infolist():
            extracted_path = (destination / member.filename).resolve()
            if not str(extracted_path).startswith(str(destination.resolve())):
                raise VoskError("Refusing to extract archive with invalid path")
        zip_file.extractall(destination)


def _locate_extracted_model_dir(extraction_dir: Path) -> Path:
    direct_candidate = extraction_dir / VOSK_MODEL_NAME
    if direct_candidate.exists() and direct_candidate.is_dir():
        return direct_candidate

    candidates = [
        path
        for path in extraction_dir.iterdir()
        if path.is_dir() and path.name.startswith("vosk-model")
    ]
    if len(candidates) == 1:
        return candidates[0]

    raise VoskError("Downloaded Vosk archive does not contain a recognizable model directory")


def _ensure_model_files(model_base_dir: str | Path | None = None) -> Path:
    model_path = _resolve_model_path(model_base_dir)
    if model_path.exists() and model_path.is_dir():
        logger.debug("Using existing Vosk model dir: %s", model_path)
        return model_path

    base_dir = model_path.parent
    base_dir.mkdir(parents=True, exist_ok=True)
    archive_path = base_dir / f"{VOSK_MODEL_NAME}.zip"

    if not archive_path.exists():
        try:
            _download_model_archive(archive_path)
        except Exception as exc:
            logger.error("Failed to download Vosk model archive: %s", exc)
            raise VoskError(
                f"Failed to download Vosk model archive from {VOSK_MODEL_ARCHIVE_URL}"
            ) from exc

    extraction_root = Path(mkdtemp(prefix="vosk_extract_", dir=base_dir))
    try:
        try:
            _safe_extract_zip(archive_path, extraction_root)
        except Exception as exc:
            logger.error("Failed to extract Vosk model archive %s: %s", archive_path, exc)
            raise VoskError(f"Failed to extract Vosk model archive: {archive_path}") from exc

        extracted_model_dir = _locate_extracted_model_dir(extraction_root)
        if model_path.exists():
            shutil.rmtree(model_path)
        shutil.move(str(extracted_model_dir), str(model_path))
        logger.info("Prepared Vosk model files at %s", model_path)
    finally:
        shutil.rmtree(extraction_root, ignore_errors=True)

    return model_path


def get_model(model_base_dir: str | Path | None = None) -> Model:
    model_path = _ensure_model_files(model_base_dir)

    global _MODEL_CACHE
    global _MODEL_PATH_CACHE

    if _MODEL_CACHE is not None and _MODEL_PATH_CACHE == model_path:
        logger.debug("Reusing cached Vosk model: %s", model_path)
        return _MODEL_CACHE

    with _MODEL_LOCK:
        if _MODEL_CACHE is not None and _MODEL_PATH_CACHE == model_path:
            logger.debug("Reusing cached Vosk model inside lock: %s", model_path)
            return _MODEL_CACHE

        try:
            loaded_model = Model(str(model_path))
        except Exception as exc:
            logger.error("Failed to initialize Vosk model from %s: %s", model_path, exc)
            raise VoskError(f"Failed to initialize Vosk model from {model_path}") from exc

        _MODEL_CACHE = loaded_model
        _MODEL_PATH_CACHE = model_path
        logger.info("Loaded Vosk model: %s", model_path)
        return loaded_model


def transcribe_preprocessed_audio(
    samples: np.ndarray,
    sample_rate: int,
    model_base_dir: str | Path | None = None,
) -> str:
    logger.info("Vosk transcription started (sample_rate=%s)", sample_rate)
    if int(sample_rate) != TARGET_SAMPLE_RATE:
        raise ValueError(f"Vosk expects sample rate {TARGET_SAMPLE_RATE} Hz, got {sample_rate} Hz")

    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    elif audio.ndim != 1:
        raise ValueError("Audio samples must be a 1D (mono) or 2D array")

    if audio.size == 0:
        logger.warning("Vosk transcription got empty audio")
        return ""

    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16).tobytes()

    model = get_model(model_base_dir)
    recognizer = KaldiRecognizer(model, float(sample_rate))
    recognizer.SetWords(False)
    recognizer.SetPartialWords(False)

    finalized_parts: list[str] = []
    for start in range(0, len(pcm16), RECOGNIZER_CHUNK_BYTES):
        chunk = pcm16[start : start + RECOGNIZER_CHUNK_BYTES]
        if recognizer.AcceptWaveform(chunk):
            chunk_text = _extract_text(recognizer.Result())
            if chunk_text:
                finalized_parts.append(chunk_text)

    final_text = _extract_text(recognizer.FinalResult())
    if final_text:
        finalized_parts.append(final_text)

    normalized = normalize_text(" ".join(finalized_parts))
    logger.info("Vosk transcription finished: '%s'", normalized)
    return normalized


def compare_with_expected_text(recognized_text: str, expected_text: str) -> TranscriptCheckResult:
    normalized_expected = normalize_text(expected_text)
    normalized_recognized = normalize_text(recognized_text)

    if not normalized_expected:
        return TranscriptCheckResult(
            expected_text=normalized_expected,
            recognized_text=normalized_recognized,
            is_match=True,
        )

    def _strip_optional_leading_the(tokens: list[str]) -> list[str]:
        stripped = list(tokens)
        while len(stripped) > 1 and stripped[0] == "the":
            stripped = stripped[1:]
        return stripped

    # Vosk can occasionally prepend "the" to a single spoken word.
    expected_tokens = _strip_optional_leading_the(normalized_expected.split())
    recognized_tokens = _strip_optional_leading_the(normalized_recognized.split())
    is_match = normalized_recognized == normalized_expected or recognized_tokens == expected_tokens

    return TranscriptCheckResult(
        expected_text=normalized_expected,
        recognized_text=normalized_recognized,
        is_match=is_match,
    )


def check_expected_text_for_preprocessed_audio(
    samples: np.ndarray,
    sample_rate: int,
    expected_text: str,
    model_base_dir: str | Path | None = None,
) -> TranscriptCheckResult:
    logger.info("Checking expected text with Vosk: expected='%s'", normalize_text(expected_text))
    recognized_text = transcribe_preprocessed_audio(
        samples=samples,
        sample_rate=sample_rate,
        model_base_dir=model_base_dir,
    )
    result = compare_with_expected_text(
        recognized_text=recognized_text,
        expected_text=expected_text,
    )
    logger.info(
        "Vosk expected-text check result: match=%s recognized='%s'",
        result.is_match,
        result.recognized_text,
    )
    return result
