from datetime import datetime

import streamlit as st

st.title("Audio Demo")
st.caption("Демонстрация приема аудиосигнала через Streamlit")

st.markdown("### Запись через микрофон")
# check params
recorded_audio = st.audio_input("Нажмите и запишите фразу на английском") 

if recorded_audio is not None:
    st.success("Аудио с микрофона получено")
    st.audio(recorded_audio)
    st.write(
        {
            "source": "microphone",
            "filename": recorded_audio.name,
            "mime_type": recorded_audio.type,
            "size_bytes": recorded_audio.size,
            "captured_at": datetime.now().isoformat(timespec="seconds"),
        }
    )

st.divider()

st.markdown("### Загрузка файла")
uploaded = st.file_uploader(
    "Или загрузите готовый аудиофайл",
    type=["wav", "mp3", "m4a", "ogg", "flac"],
)

if uploaded is not None:
    st.success("Файл загружен")
    st.audio(uploaded)
    st.write(
        {
            "source": "file_upload",
            "filename": uploaded.name,
            "mime_type": uploaded.type,
            "size_bytes": uploaded.size,
        }
    )

