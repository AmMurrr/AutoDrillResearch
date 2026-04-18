import streamlit as st

st.set_page_config(
    page_title=" Анализ произношения",
    page_icon="🎯",
    layout="wide",
)

st.title("Система анализа корректности произношения (MVP)")
st.caption("ВКР: исследование двух подходов для автоматизации языкового дриллинга")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1) Классический подход")
    st.markdown(
        """
        - Извлечение MFCC и доп. признаков
        - Сравнение эталона и попытки через DTW
        - Оценка итогового качества произношения
        """
    )

with col2:
    st.subheader("2) Нейросетевой подход")
    st.markdown(
        """
        - Эмбеддинги из wav2vec 2.0
        - Сравнение косинусного сходства
        - Агрегация в итоговый pronunciation score
        """
    )

st.divider()


st.info(
    "Заглушка главной страницы."
)
