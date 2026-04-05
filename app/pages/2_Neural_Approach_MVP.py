from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st
from neural_approach.pipeline import analyze
from neural_approach.wav2vec_extractor import DEFAULT_MODEL_NAME
from app.reference_db import init_db, list_reference_paths

st.title("Нейросетевой MVP")
init_db()


st.markdown("### Входные данные")
references = list_reference_paths()
reference_options = ["data/reference/sample_reference.wav"] + [r["path"] for r in references]

selected_reference = st.selectbox(
    "Эталон из SQLite",
    options=reference_options,
)
manual_reference = st.text_input("Или укажите путь вручную (приоритет)", value="")
reference_path = manual_reference.strip() or selected_reference

attempt_path = st.text_input(
    "Путь к аудио пользователя",
    value="data/attempts/sample_attempt.wav",
)

st.markdown("### Аудио из браузера")
uploaded_attempt = st.file_uploader(
    "Загрузите аудио пользователя (опционально, приоритет над путем)",
    type=["wav", "mp3", "m4a", "ogg", "flac"],
    key="neural_uploaded_attempt",
)
recorded_attempt = st.audio_input("Или запишите попытку через микрофон", key="neural_recorded_attempt")

transcript = st.text_input("Ожидаемый текст фразы", value="Hello")

st.markdown("### Параметры")
col1, col2, col3 = st.columns(3)

with col1:
    similarity = st.selectbox("Similarity", options=["cosine", "euclidean"])
with col2:
    model_name = st.text_input("HF модель", value=DEFAULT_MODEL_NAME)
with col3:
    device_choice = st.selectbox("Device", options=["auto", "cpu", "cuda"])

hf_token = st.text_input(
    "HF_TOKEN (опционально)",
    value="",
    type="password",
    help="Если поле пустое, будет использована переменная окружения HF_TOKEN (если задана).",
)

if st.button("Запустить MVP", type="primary"):
    audio_source = recorded_attempt if recorded_attempt is not None else uploaded_attempt
    tmp_path = None
    resolved_attempt_path = attempt_path.strip()

    if audio_source is not None:
        suffix = Path(audio_source.name).suffix or ".wav"
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_source.getvalue())
            tmp_path = tmp.name
        resolved_attempt_path = tmp_path

    ref_exists = Path(reference_path).exists()
    att_exists = Path(resolved_attempt_path).exists()

    if not ref_exists or not att_exists:
        st.error("Проверьте пути: эталон и аудио пользователя должны существовать")
    else:
        with st.spinner("Загружаю wav2vec2 и считаю эмбеддинги..."):
            try:
                result = analyze(
                    user_audio_path=resolved_attempt_path,
                    reference_audio_path=reference_path,
                    transcript=transcript,
                    similarity=similarity,
                    model_name=model_name.strip() or DEFAULT_MODEL_NAME,
                    device=None if device_choice == "auto" else device_choice,
                    hf_token=hf_token.strip() or None,
                )
            except Exception as exc:
                st.error(f"Ошибка при анализе: {exc}")
            else:
                st.success("MVP выполнен")
                st.write(
                    {
                        "reference_exists": ref_exists,
                        "attempt_exists": att_exists,
                        "input_mode": "streamlit_audio" if audio_source is not None else "manual_path",
                        "attempt_path_used": resolved_attempt_path,
                        "transcript": transcript,
                        "metric": result.metric,
                        "model_name": result.model_name,
                        "pronunciation_score": result.pronunciation_score,
                        "verdict": result.verdict,
                        "embedding_similarity": result.similarity,
                        "temporal_distance": result.temporal_distance,
                        "problematic_phonemes": result.problematic_phonemes,
                    }
                )

