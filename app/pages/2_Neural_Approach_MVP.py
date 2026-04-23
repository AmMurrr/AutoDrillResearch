from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st
from neural_approach.pipeline import analyze
from neural_approach.wav2vec_extractor import DEFAULT_MODEL_NAME, HF_TOKEN_ENV_VAR, resolve_hf_token
from app.logging_config import get_logger
from scoring.anchor_calibration import describe_anchor_set, get_word_anchor_set, list_anchor_words, normalize_word


logger = get_logger(__name__)
logger.info("Opened Streamlit page: Neural Approach MVP")


def _parse_word_mismatch_reason(reason: str) -> tuple[str, str]:
    expected = ""
    recognized = ""
    for part in (reason or "").split(";"):
        if ":" not in part:
            continue
        key, value = part.split(":", maxsplit=1)
        normalized_key = key.strip().lower()
        if normalized_key == "expected":
            expected = value.strip()
        elif normalized_key == "recognized":
            recognized = value.strip()
    return expected, recognized


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

st.markdown("### Входные данные")
transcript = st.text_input("Ожидаемое слово", value="happy")
anchor_word = normalize_word(transcript)

max_anchors_preview = st.slider(
    "Якорей на класс (лимит для калибровки)",
    min_value=1,
    max_value=30,
    value=12,
)

anchor_set = get_word_anchor_set(
    word=anchor_word,
    max_anchors_per_class=max_anchors_preview,
)
anchor_stats = describe_anchor_set(anchor_set)

available_words = list_anchor_words()
if available_words:
    st.caption("Слова с perfect-якорями в data/ref: " + ", ".join(available_words))
else:
    st.warning("В data/ref не найдены папки вида word_perfect с аудио")

if anchor_set.has_required_anchors:
    st.success(
        "Якоря для слова готовы: "
        f"perfect={anchor_stats['perfect']}, "
        f"fail={anchor_stats['fail']}, "
        f"moderate={anchor_stats['moderate']}, "
        f"empty_word={anchor_stats['empty_word']}"
    )
else:
    st.warning(
        "Для выбранного слова не хватает якорей. "
        "Нужны папки word_perfect и хотя бы один fail-класс "
        "(word_fail / word_moderate)."
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
col1, col2, col3, col4 = st.columns(4)

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
with col4:
    raw_distance_alpha = st.slider(
        "Вес temporal в raw distance",
        min_value=0.0,
        max_value=1.0,
        value=0.65,
        step=0.05,
    )

use_vosk = st.checkbox(
    "Проверять слово через Vosk перед эмбеддингами",
    value=True,
    help="Если Vosk распознает другое слово, результат сразу помечается как wrong_word с оценкой 0.",
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
    logger.info(
        "Neural page run requested: transcript='%s' model='%s' use_vosk=%s input_mode=%s",
        transcript,
        model_name,
        use_vosk,
        "streamlit_audio" if audio_source is not None else "manual_path",
    )

    if audio_source is not None:
        suffix = Path(audio_source.name).suffix or ".wav"
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_source.getvalue())
            tmp_path = tmp.name
        resolved_attempt_path = tmp_path

    try:
        att_exists = Path(resolved_attempt_path).exists()

        if not att_exists:
            logger.warning("Neural page run rejected: attempt path does not exist (%s)", resolved_attempt_path)
            st.error("Путь к аудио пользователя не существует")
        elif not anchor_set.has_required_anchors:
            logger.warning("Neural page run rejected: anchors are incomplete for word '%s'", anchor_word)
            st.error(
                "Нельзя запустить анализ: для слова отсутствуют обязательные якоря. "
                "Проверьте data/ref."
            )
        else:
            with st.spinner("Загружаю wav2vec2 и считаю якорную калибровку..."):
                try:
                    resolved_hf_token = resolve_hf_token(hf_token.strip() or None)
                    result = analyze(
                        user_audio_path=resolved_attempt_path,
                        transcript=transcript,
                        similarity="cosine",
                        model_name=model_name.strip() or DEFAULT_MODEL_NAME,
                        device=None if device_choice == "auto" else device_choice,
                        hf_token=resolved_hf_token,
                        sakoe_chiba_radius=int(sakoe_chiba_radius),
                        use_vosk=use_vosk,
                        raw_distance_alpha=float(raw_distance_alpha),
                        max_anchors_per_class=max_anchors_preview,
                    )
                except Exception as exc:
                    logger.error("Neural page analysis failed: %s", exc)
                    st.error(f"Ошибка при анализе: {exc}")
                else:
                    logger.info(
                        "Neural page analysis finished: status=%s score=%.2f verdict=%s",
                        result.status,
                        result.pronunciation_score,
                        result.verdict,
                    )
                    st.success("MVP выполнен")
                    result_payload = {
                        "word": anchor_word,
                        "anchors": anchor_stats,
                        "attempt_exists": att_exists,
                        "input_mode": "streamlit_audio" if audio_source is not None else "manual_path",
                        "attempt_path_used": resolved_attempt_path,
                        "transcript": transcript,
                        "sakoe_chiba_radius": int(sakoe_chiba_radius),
                        "raw_distance_alpha": float(raw_distance_alpha),
                        "use_vosk": use_vosk,
                        "metric": result.metric,
                        "model_name": result.model_name,
                        "pronunciation_score": result.pronunciation_score,
                        "verdict": result.verdict,
                        "status": result.status,
                        "reason": result.reason,
                        "embedding_similarity": result.similarity,
                        "temporal_distance": result.temporal_distance,
                        "raw_distance": result.raw_distance,
                        "d100": result.d100,
                        "d0": result.d0,
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
                        temporal_text = "∞" if result.temporal_distance == float("inf") else f"{result.temporal_distance:.4f}"
                        st.metric("Временная дистанция", temporal_text)

                    st.progress(int(max(0, min(100, round(result.pronunciation_score)))))
                    if result.status == "empty_audio":
                        st.error(
                            "Нераспознанное слово: в записи недостаточно речевого сигнала "
                            "(пусто, слишком тихо или обрыв). Повторите попытку."
                        )
                    elif result.status == "wrong_word":
                        expected, recognized = _parse_word_mismatch_reason(result.reason)
                        st.error(
                            "Распознано другое слово. "
                            f"Ожидалось: '{expected or transcript.strip().lower()}', "
                            f"распознано: '{recognized or 'не распознано'}'. "
                            "Оценка выставлена в 0."
                        )
                    elif result.status == "asr_error":
                        st.warning(
                            "Vosk не смог выполнить проверку слова. "
                            "Можно повторить попытку или временно отключить Vosk в параметрах."
                        )
                        if result.reason:
                            st.caption(result.reason)
                    elif result.status == "invalid_reference":
                        st.error("Не удалось выполнить анализ: для слова не собраны корректные якоря.")
                        if result.reason:
                            st.caption(result.reason)
                    else:
                        _render_verdict_block(result.verdict)

                    details_col_1, details_col_2 = st.columns(2)
                    with details_col_1:
                        st.markdown("**Модель**")
                        st.info(result.model_name)
                    with details_col_2:
                        st.markdown("**Метрика**")
                        st.info(result.metric)

                    with st.expander("Калибровка"):
                        st.write({
                            "d100": result.d100,
                            "d0": result.d0,
                            "raw_distance": result.raw_distance,
                            "alpha": raw_distance_alpha,
                        })

                    with st.expander("DEBUG"):
                        st.write(result_payload)
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass
