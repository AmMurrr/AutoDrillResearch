from pathlib import Path

import streamlit as st

st.title("ML MVP")


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
col1,col2 = st.columns(2)

with col1:
    similarity = st.selectbox("Similarity", options=["cosine", "euclidean"])

if st.button("Запустить MVP", type="primary"):
    ref_exists = Path(reference_path).exists()
    att_exists = Path(attempt_path).exists()

    st.success(" выполнен")
    st.write(
        {
            "reference_exists": ref_exists,
            "attempt_exists": att_exists,
            "similarity": similarity,
            "embedding_similarity": "TODO",
            "pronunciation_score": "TODO",
            "diagnostics": "TODO",
        }
    )

