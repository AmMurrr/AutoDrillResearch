from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel


DEFAULT_MODEL_NAME = "facebook/wav2vec2-base"


@dataclass
class Wav2VecEmbeddings:
	frame_embeddings: np.ndarray
	pooled_embedding: np.ndarray
	sample_rate: int
	model_name: str
	device: str


def _resolve_device(device: str | None = None) -> torch.device:
	if device is not None and device.strip():
		requested = device.strip().lower()
		if requested == "cuda" and not torch.cuda.is_available():
			return torch.device("cpu")
		return torch.device(requested)

	if torch.cuda.is_available():
		return torch.device("cuda")
	return torch.device("cpu")


def _resolve_hf_token(hf_token: str | None = None) -> str | None:
	if hf_token is not None and hf_token.strip():
		return hf_token.strip()

	env_token = os.getenv("HF_TOKEN", "").strip()
	if env_token:
		return env_token
	return None


@lru_cache(maxsize=8)
def _load_model_bundle(model_name: str, device_str: str, hf_token: str | None):
	feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, token=hf_token)
	model = AutoModel.from_pretrained(model_name, token=hf_token)
	model.eval()
	model.to(torch.device(device_str))
	return feature_extractor, model


def extract_wav2vec_embeddings(
	samples: np.ndarray,
	sample_rate: int,
	model_name: str = DEFAULT_MODEL_NAME,
	device: str | None = None,
	hf_token: str | None = None,
) -> Wav2VecEmbeddings:
	if sample_rate != 16000:
		raise ValueError("wav2vec2-base expects 16 kHz audio after preprocessing")

	speech = np.asarray(samples, dtype=np.float32)
	if speech.ndim != 1:
		raise ValueError("Expected mono waveform with shape (n_samples,)")
	if speech.size == 0:
		raise ValueError("Audio is empty after preprocessing")

	speech = np.clip(speech, -1.0, 1.0)
	resolved_device = _resolve_device(device)
	resolved_token = _resolve_hf_token(hf_token)
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
		outputs = model(input_values=input_values, attention_mask=attention_mask)

	frame_embeddings = outputs.last_hidden_state.squeeze(0).detach().cpu().numpy().astype(np.float32)
	pooled_embedding = frame_embeddings.mean(axis=0, dtype=np.float32)

	return Wav2VecEmbeddings(
		frame_embeddings=frame_embeddings,
		pooled_embedding=pooled_embedding,
		sample_rate=sample_rate,
		model_name=model_name,
		device=resolved_device.type,
	)
