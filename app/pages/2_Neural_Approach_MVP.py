from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st
from neural_approach.pipeline import analyze
from neural_approach.wav2vec_extractor import DEFAULT_MODEL_NAME, HF_TOKEN_ENV_VAR, resolve_hf_token
from app.reference_db import init_db, list_reference_paths


def _verdict_to_russian(verdict: str) -> str:
    verdict_map = {
        "good": "хорошо",
        "acceptable": "удовлетворительно",
        "needs_improvement": "неудовлетворительно",
        "хорошо": "хорошо",
        "удовлетворительно": "удовлетворительно",
        "неудовлетворительно": "неудовлетворительно",
    }
    return verdict_map.get(verdict, verdict)


def _render_verdict_block(verdict: str) -> None:
    verdict_ru = _verdict_to_russian(verdict)
    if verdict_ru == "хорошо":
        st.success("Итог: произношение близко к эталону по качеству и темпу.")
    elif verdict_ru == "удовлетворительно":
        st.warning("Итог: качество приемлемое, но есть зоны для улучшения.")
    else:
        st.error("Итог: требуется дополнительная практика и повторная попытка.")


st.title("Нейросетевой MVP")
init_db()


st.markdown("### Входные данные")
transcript = st.text_input("Ожидаемый текст фразы", value="Hello")

default_word = transcript.strip().lower()
reference_word = st.text_input("Слово для поиска эталонов в SQLite", value=default_word)

references = list_reference_paths(word=reference_word.strip())
if references:
    st.caption(f"Найдено эталонов по слову '{reference_word.strip()}': {len(references)}")
    st.dataframe(
        [{"word": row["word"], "path": row["path"], "created_at": row["created_at"]} for row in references],
        use_container_width=True,
        hide_index=True,
    )
    if len(references) == 1:
        references_to_use = 1
        st.caption("Доступен один эталон: он будет использован автоматически.")
    else:
        references_to_use = st.slider(
            "Сколько эталонов использовать в алгоритме",
            min_value=1,
            max_value=len(references),
            value=min(3, len(references)),
        )
else:
    st.info("По этому слову эталоны в БД не найдены")
    references_to_use = 0

manual_reference = st.text_input(
    "Или укажите путь вручную (используется только один эталон)",
    value="",
)

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

st.markdown("### Параметры")
col1, col2, col3 = st.columns(3)

with col1:
    model_name = st.text_input("HF модель", value=DEFAULT_MODEL_NAME)
with col2:
    device_choice = st.selectbox("Device", options=["auto", "cpu", "cuda"])
with col3:
    sakoe_chiba_radius = st.number_input(
        "Sakoe-Chiba band",
        min_value=0,
        max_value=100,
        value=12,
        step=1,
        help="0 отключает ограничение окна temporal DTW.",
    )

hf_token = st.text_input(
    "HF_TOKEN (опционально)",
    value="",
    help=(
        "Если поле пустое, токен будет взят из переменной окружения "
        f"{HF_TOKEN_ENV_VAR}"
    ),
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

    manual_reference_path = manual_reference.strip()
    if manual_reference_path:
        selected_reference_paths = [manual_reference_path]
        reference_mode = "manual_single"
    else:
        selected_reference_paths = [row["path"] for row in references[:references_to_use]]
        reference_mode = "sqlite_by_word"

    try:
        missing_references = [path for path in selected_reference_paths if not Path(path).exists()]
        att_exists = Path(resolved_attempt_path).exists()

        if not selected_reference_paths:
            st.error("Не удалось выбрать эталоны: укажите путь вручную или найдите эталоны по слову")
        elif missing_references or not att_exists:
            st.error("Проверьте пути: выбранные эталоны и аудио пользователя должны существовать")
            if missing_references:
                st.write({"отсутствующие_эталоны": missing_references})
        else:
            with st.spinner("Загружаю wav2vec2 и считаю эмбеддинги..."):
                try:
                    resolved_hf_token = resolve_hf_token(hf_token.strip() or None)
                    result = analyze(
                        user_audio_path=resolved_attempt_path,
                        reference_audio_path=selected_reference_paths,
                        transcript=transcript,
                        similarity="cosine",
                        model_name=model_name.strip() or DEFAULT_MODEL_NAME,
                        device=None if device_choice == "auto" else device_choice,
                        hf_token=resolved_hf_token,
                        sakoe_chiba_radius=int(sakoe_chiba_radius),
                    )
                except Exception as exc:
                    st.error(f"Ошибка при анализе: {exc}")
                else:
                    st.success("MVP выполнен")
                    result_payload = {
                        "reference_mode": reference_mode,
                        "reference_word": reference_word.strip(),
                        "reference_paths": selected_reference_paths,
                        "reference_count": len(selected_reference_paths),
                        "attempt_exists": att_exists,
                        "input_mode": "streamlit_audio" if audio_source is not None else "manual_path",
                        "attempt_path_used": resolved_attempt_path,
                        "transcript": transcript,
                        "sakoe_chiba_radius": int(sakoe_chiba_radius),
                        "metric": result.metric,
                        "model_name": result.model_name,
                        "pronunciation_score": result.pronunciation_score,
                        "verdict": result.verdict,
                        "status": result.status,
                        "reason": result.reason,
                        "embedding_similarity": result.similarity,
                        "temporal_distance": result.temporal_distance,
                        "problematic_phonemes": result.problematic_phonemes,
                    }

                    verdict_ru = _verdict_to_russian(result.verdict)

                    st.markdown("### Результат")
                    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
                    with metric_col_1:
                        st.metric("Оценка произношения", f"{result.pronunciation_score:.1f} / 100")
                    with metric_col_2:
                        st.metric("Вердикт", verdict_ru)
                    with metric_col_3:
                        st.metric("Сходство эмбеддингов", f"{result.similarity:.4f}")
                    with metric_col_4:
                        st.metric("Временная дистанция", f"{result.temporal_distance:.4f}")

                    st.progress(int(max(0, min(100, round(result.pronunciation_score)))))
                    if result.status == "empty_audio":
                        st.error(
                            "Нераспознанное слово: в записи недостаточно речевого сигнала "
                            "(пусто, слишком тихо или обрыв). Повторите попытку."
                        )
                    else:
                        _render_verdict_block(result.verdict)

                    details_col_1, details_col_2 = st.columns(2)
                    with details_col_1:
                        st.markdown("**Модель**")
                        st.info(result.model_name)
                    with details_col_2:
                        st.markdown("**Метрика**")
                        st.info(result.metric)

                    st.markdown("#### Проблемные зоны")
                    if result.problematic_phonemes:
                        st.write(", ".join(result.problematic_phonemes))
                    else:
                        st.info("Явных проблемных зон не обнаружено.")

                    with st.expander("DEBUG"):
                        st.write(result_payload)
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass

