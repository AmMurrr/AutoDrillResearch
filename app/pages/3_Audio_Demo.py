from datetime import datetime
from pathlib import Path

import streamlit as st
from app.reference_db import REFERENCE_DIR, add_reference_path, init_db
from app.logging_config import get_logger


logger = get_logger(__name__)
logger.info("Opened Streamlit page: Audio Demo")

init_db()


def save_to_reference(data: bytes, suffix: str, prefix: str) -> str:
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"
    output_path = REFERENCE_DIR / filename
    output_path.write_bytes(data)
    rel_path = output_path.relative_to(Path.cwd()).as_posix()
    logger.info("Saved audio to reference directory: %s", rel_path)
    return rel_path

st.title("Audio Demo")
st.caption("Демонстрация приема аудиосигнала через Streamlit")

reference_word = st.text_input("Слово для сохранения в БД", value="hello")

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
        logger.info("Audio demo: save microphone recording requested for word='%s'", reference_word)
        rel_path = save_to_reference(recorded_audio.getvalue(), ".wav", "mic")
        add_reference_path(reference_word, rel_path, "from_mic")
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
        logger.info("Audio demo: save uploaded file requested for word='%s'", reference_word)
        ext = Path(uploaded.name).suffix.lower() or ".wav"
        rel_path = save_to_reference(uploaded.getvalue(), ext, "upload")
        add_reference_path(reference_word, rel_path, "from_upload")
        st.success(f"Сохранено и добавлено в БД: {rel_path}")

