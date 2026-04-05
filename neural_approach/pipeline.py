from __future__ import annotations

from .embedding_comparator import compare_embeddings
from .preprocessing import preprocess_audio
from .scorer import ScoringResult, compute_scoring_result
from .wav2vec_extractor import DEFAULT_MODEL_NAME, extract_wav2vec_embeddings


def analyze(
	user_audio_path: str,
	reference_audio_path: str,
	transcript: str,
	similarity: str = "cosine",
	model_name: str = DEFAULT_MODEL_NAME,
	device: str | None = None,
	hf_token: str | None = None,
) -> ScoringResult:
	# transcript пока не используется в логике, оставлен для будущей
	# диагностики по словам/фонемам.
	_ = transcript

	user_audio = preprocess_audio(user_audio_path)
	reference_audio = preprocess_audio(reference_audio_path)

	user_embeddings = extract_wav2vec_embeddings(
		user_audio.samples,
		user_audio.sample_rate,
		model_name=model_name,
		device=device,
		hf_token=hf_token,
	)
	reference_embeddings = extract_wav2vec_embeddings(
		reference_audio.samples,
		reference_audio.sample_rate,
		model_name=model_name,
		device=device,
		hf_token=hf_token,
	)

	comparison = compare_embeddings(
		user_embeddings=user_embeddings.frame_embeddings,
		reference_embeddings=reference_embeddings.frame_embeddings,
		metric=similarity,
	)

	return compute_scoring_result(
		similarity=comparison.similarity,
		temporal_distance=comparison.temporal_distance,
		metric=comparison.metric,
		model_name=model_name,
		phoneme_issues=[],
	)
 