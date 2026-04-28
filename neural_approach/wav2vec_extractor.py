from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

import numpy as np
from app.logging_config import get_logger
import torch
from transformers import AutoFeatureExtractor, AutoModel


DEFAULT_MODEL_NAME = "facebook/wav2vec2-base"
DEFAULT_EMBEDDING_LAYER = 6
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODELS_DIR = PROJECT_ROOT / "models"
HF_TOKEN_ENV_VAR = "HF_TOKEN"
logger = get_logger(__name__)


@dataclass
class Wav2VecEmbeddings:
    frame_embeddings: np.ndarray
    pooled_embedding: np.ndarray
    sample_rate: int
    model_name: str
    device: str
    embedding_layer: int | None = None


def statistical_pooling(frame_embeddings: np.ndarray) -> np.ndarray:
    frames = np.asarray(frame_embeddings, dtype=np.float32)
    if frames.ndim != 2:
        raise ValueError("Expected frame embeddings with shape (n_frames, emb_dim)")
    if frames.shape[0] == 0:
        raise ValueError("Cannot pool empty frame embeddings")

    mean = frames.mean(axis=0, dtype=np.float32)
    std = frames.std(axis=0).astype(np.float32, copy=False)
    return np.concatenate([mean, std]).astype(np.float32, copy=False)


def _resolve_device(device: str | None = None) -> torch.device:
    if device is not None and device.strip():
        requested = device.strip().lower()
        if requested == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable, falling back to CPU")
            return torch.device("cpu")
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _read_dotenv_value(dotenv_path: Path, key: str) -> str | None:
    try:
        content = dotenv_path.read_text(encoding="utf-8")
    except OSError:
        return None

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export ") :].strip()

        if "=" not in line:
            continue

        lhs, rhs = line.split("=", 1)
        if lhs.strip() != key:
            continue

        value = rhs.strip()
        if value and value[0] in {'"', "'"} and value[-1] == value[0]:
            value = value[1:-1]

        value = value.strip()
        return value or None

    return None


def _resolve_hf_token_from_dotenv() -> str | None:
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[1] / ".env",
    ]

    seen: set[str] = set()
    for candidate in candidates:
        candidate_key = str(candidate)
        if candidate_key in seen:
            continue
        seen.add(candidate_key)

        token = _read_dotenv_value(candidate, HF_TOKEN_ENV_VAR)
        if token:
            return token

    return None


def resolve_hf_token(hf_token: str | None = None) -> str | None:
    if hf_token is not None and hf_token.strip():
        return hf_token.strip()

    env_token = os.getenv(HF_TOKEN_ENV_VAR, "").strip()
    if env_token:
        return env_token

    dotenv_token = _resolve_hf_token_from_dotenv()
    if dotenv_token:
        return dotenv_token
    return None


def _use_local_files_only() -> bool:
    for env_var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        value = os.getenv(env_var, "").strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
    return False


def _local_model_dir_name(model_name: str) -> str:
    return model_name.strip().replace("/", "-")


def _has_transformers_model_files(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False

    has_config = (model_dir / "config.json").is_file()
    has_preprocessor = (model_dir / "preprocessor_config.json").is_file()
    has_weights = any(
        (model_dir / filename).is_file()
        for filename in ("pytorch_model.bin", "model.safetensors")
    )
    return has_config and has_preprocessor and has_weights


def resolve_model_name_or_path(model_name: str) -> str:
    requested = model_name.strip() or DEFAULT_MODEL_NAME
    requested_path = Path(requested).expanduser()
    if requested_path.exists():
        return str(requested_path)

    local_model_dir = LOCAL_MODELS_DIR / _local_model_dir_name(requested)
    if _has_transformers_model_files(local_model_dir):
        return str(local_model_dir)

    return requested


def _select_frame_embeddings(outputs, embedding_layer: int | None):
    if embedding_layer is None:
        return outputs.last_hidden_state

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise ValueError("Model output does not contain hidden states")

    layer_index = int(embedding_layer)
    if layer_index < 0:
        layer_index += len(hidden_states)
    if layer_index < 0 or layer_index >= len(hidden_states):
        raise ValueError(
            f"embedding_layer={embedding_layer} is outside available hidden states "
            f"0..{len(hidden_states) - 1}"
        )
    return hidden_states[layer_index]


@lru_cache(maxsize=8)
def _load_model_bundle(model_name: str, device_str: str, hf_token: str | None):
    resolved_model = resolve_model_name_or_path(model_name)
    local_files_only = _use_local_files_only() or Path(resolved_model).exists()
    logger.info(
        "Loading wav2vec bundle: model=%s resolved_model=%s device=%s",
        model_name,
        resolved_model,
        device_str,
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        resolved_model,
        token=hf_token,
        local_files_only=local_files_only,
    )
    model = AutoModel.from_pretrained(
        resolved_model,
        token=hf_token,
        local_files_only=local_files_only,
        use_safetensors=False,
    )
    model.eval()
    model.to(torch.device(device_str))
    logger.info("wav2vec bundle loaded: model=%s device=%s", model_name, device_str)
    return feature_extractor, model


def extract_wav2vec_embeddings(
    samples: np.ndarray,
    sample_rate: int,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
    hf_token: str | None = None,
    embedding_layer: int | None = DEFAULT_EMBEDDING_LAYER,
) -> Wav2VecEmbeddings:
    logger.info(
        "Extracting wav2vec embeddings: model=%s sample_rate=%s layer=%s",
        model_name,
        sample_rate,
        embedding_layer,
    )
    if sample_rate != 16000:
        raise ValueError("wav2vec2-base expects 16 kHz audio after preprocessing")

    speech = np.asarray(samples, dtype=np.float32)
    if speech.ndim != 1:
        raise ValueError("Expected mono waveform with shape (n_samples,)")
    if speech.size == 0:
        raise ValueError("Audio is empty after preprocessing")

    speech = np.clip(speech, -1.0, 1.0)
    resolved_device = _resolve_device(device)
    resolved_token = resolve_hf_token(hf_token)
    feature_extractor, model = _load_model_bundle(model_name, resolved_device.type, resolved_token)

    inputs = feature_extractor(
        speech,
        sampling_rate=sample_rate,
        return_tensors="pt",
        return_attention_mask=True,
    )

    input_values = inputs["input_values"].to(resolved_device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(resolved_device)

    with torch.inference_mode():
        outputs = model(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=embedding_layer is not None,
        )

    frame_embeddings = (
        _select_frame_embeddings(outputs, embedding_layer)
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    pooled_embedding = statistical_pooling(frame_embeddings)
    logger.info(
        "wav2vec embeddings extracted: frames=%s dim=%s pooled_dim=%s",
        frame_embeddings.shape[0],
        frame_embeddings.shape[1],
        pooled_embedding.shape[0],
    )

    return Wav2VecEmbeddings(
        frame_embeddings=frame_embeddings,
        pooled_embedding=pooled_embedding,
        sample_rate=sample_rate,
        model_name=model_name,
        device=resolved_device.type,
        embedding_layer=embedding_layer,
    )
