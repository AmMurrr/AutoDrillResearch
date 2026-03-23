from datetime import datetime
from pathlib import Path

import streamlit as st
from app.reference_db import REFERENCE_DIR, add_reference_path, init_db

init_db()


def save_to_reference(data: bytes, suffix: str, prefix: str) -> str:
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"
    output_path = REFERENCE_DIR / filename
    output_path.write_bytes(data)
    return output_path.relative_to(Path.cwd()).as_posix()

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
    if st.button("Сохранить запись в data/reference"):
        rel_path = save_to_reference(recorded_audio.getvalue(), ".wav", "mic")
        add_reference_path(rel_path, "from_mic")
        st.success(f"Сохранено и добавлено в БД: {rel_path}")

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
    if st.button("Сохранить файл в data/reference"):
        ext = Path(uploaded.name).suffix.lower() or ".wav"
        rel_path = save_to_reference(uploaded.getvalue(), ext, "upload")
        add_reference_path(rel_path, "from_upload")
        st.success(f"Сохранено и добавлено в БД: {rel_path}")

