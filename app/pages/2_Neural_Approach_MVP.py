from pathlib import Path

import streamlit as st
from reference_db import init_db, list_reference_paths

st.title("ML MVP")
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

