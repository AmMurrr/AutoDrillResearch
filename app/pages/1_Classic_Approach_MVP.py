from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st
from classic_approach.pipeline import analyze
from app.reference_db import init_db, list_reference_paths

st.title("Классический MVP")
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
    key="classic_uploaded_attempt",
)
recorded_attempt = st.audio_input("Или запишите попытку через микрофон", key="classic_recorded_attempt")

transcript = st.text_input("Ожидаемый текст фразы", value="Hello")

st.markdown("### Параметры")
col1, col2, col3 = st.columns(3)
with col1:
    n_mfcc = st.slider("MFCC коэффициенты", min_value=13, max_value=40, value=20)
with col2:
    frame_ms = st.slider("Длина окна (ms)", min_value=10, max_value=50, value=25)
with col3:
    hop_ms = st.slider("Шаг окна (ms)", min_value=5, max_value=20, value=10)

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
        result = analyze(
            user_audio_path=resolved_attempt_path,
            reference_audio_path=reference_path,
            transcript=transcript,
            n_mfcc=n_mfcc,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
        )
        st.success("MVP выполнен")
        st.write(
            {
                "reference_exists": ref_exists,
                "attempt_exists": att_exists,
                "input_mode": "streamlit_audio" if audio_source is not None else "manual_path",
                "attempt_path_used": resolved_attempt_path,
                "n_mfcc": n_mfcc,
                "frame_ms": frame_ms,
                "hop_ms": hop_ms,
                "pronunciation_score": result.dtw_score,
                "verdict": result.verdict,
                "problematic_phonemes": result.problematic_phonemes,
            }
        )


