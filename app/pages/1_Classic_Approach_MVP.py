from pathlib import Path

import streamlit as st

st.title("Классический MVP")




st.markdown("### Входные данные")
reference_path = st.text_input(
    "Путь к эталонному аудио",
    value="data/reference/sample_reference.wav",
)
attempt_path = st.text_input(
    "Путь к аудио пользователя",
    value="data/attempts/sample_attempt.wav",
)

st.markdown("### Параметры")
col1, col2, col3 = st.columns(3)
with col1:
    n_mfcc = st.slider("MFCC coefficients", min_value=13, max_value=40, value=20)
with col2:
    frame_ms = st.slider("Frame length (ms)", min_value=10, max_value=50, value=25)
with col3:
    hop_ms = st.slider("Hop length (ms)", min_value=5, max_value=20, value=10)

if st.button("Запустить MVP", type="primary"):
    ref_exists = Path(reference_path).exists()
    att_exists = Path(attempt_path).exists()

    st.success("выполнен")
    st.write(
        {
            "reference_exists": ref_exists,
            "attempt_exists": att_exists,
            "n_mfcc": n_mfcc,
            "frame_ms": frame_ms,
            "hop_ms": hop_ms,
            "dtw_distance": "TODO",
            "phoneme_alignment": "TODO",
            "pronunciation_score": "TODO",
        }
    )


