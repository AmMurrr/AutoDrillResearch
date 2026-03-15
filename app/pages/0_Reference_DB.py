import streamlit as st

from reference_db import (
    DB_PATH,
    REFERENCE_DIR,
    add_reference_path,
    delete_reference_path,
    init_db,
    list_reference_paths,
    scan_reference_dir,
)

st.title("Reference DB")
st.caption("SQLite-хранилище путей к эталонным записям")

init_db()

st.write(
    {
        "db_path": DB_PATH.as_posix(),
        "reference_dir": REFERENCE_DIR.as_posix(),
    }
)

col1, col2 = st.columns(2)

with col1:
    if st.button("Сканировать data/reference", type="primary"):
        added_count = scan_reference_dir()
        st.success(f"Сканирование завершено. Добавлено новых записей: {added_count}")

with col2:
    st.caption("Если записи уже есть в БД, дубликаты добавлены не будут")

st.divider()

st.markdown("### Добавить путь вручную")
manual_path = st.text_input("Путь к эталонному аудио", value="data/reference/example.wav")
manual_label = st.text_input("Метка (необязательно)", value="")

if st.button("Добавить путь"):
    if add_reference_path(manual_path, manual_label):
        st.success("Путь добавлен в БД")
    else:
        st.warning("Путь пустой или уже существует")

st.divider()

st.markdown("### Текущие записи")
records = list_reference_paths()

if not records:
    st.info("База пока пустая")
else:
    st.dataframe(records, use_container_width=True)
    delete_choice = st.selectbox(
        "Удалить запись",
        options=[f"{r['id']} | {r['path']}" for r in records],
    )
    if st.button("Удалить выбранную запись"):
        reference_id = int(delete_choice.split("|", maxsplit=1)[0].strip())
        if delete_reference_path(reference_id):
            st.success("Запись удалена")
        else:
            st.warning("Не удалось удалить запись")