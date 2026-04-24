from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st
from classic_approach.pipeline import analyze
from app.logging_config import get_logger
from scoring.anchor_calibration import describe_anchor_set, get_word_anchor_set, list_anchor_words, normalize_word


logger = get_logger(__name__)
logger.info("Opened Streamlit page: Classic Approach MVP")


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


def _render_verdict_block(verdict: str) -> None:
    if verdict == "хорошо":
        st.success("Итог: произношение звучит уверенно и близко к эталону.")
    elif verdict == "удовлетворительно":
        st.warning("Итог: есть заметные отклонения, стоит доработать проблемные зоны.")
    else:
        st.error("Итог: выраженные отклонения от эталона, нужна дополнительная практика.")


st.title("Классический MVP")

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
    key="classic_uploaded_attempt",
)
recorded_attempt = st.audio_input("Или запишите попытку через микрофон", key="classic_recorded_attempt")

st.markdown("### Параметры")
col1, col2, col3, col4 = st.columns(4)
with col1:
    n_mfcc = st.slider("MFCC коэффициенты", min_value=13, max_value=40, value=20)
with col2:
    frame_ms = st.slider("Длина окна (ms)", min_value=10, max_value=50, value=25)
with col3:
    hop_ms = st.slider("Шаг окна (ms)", min_value=5, max_value=20, value=10)
with col4:
    sakoe_chiba_radius = st.number_input(
        "Sakoe-Chiba band",
        min_value=0,
        max_value=100,
        value=12,
        step=1,
        help="0 отключает ограничение окна DTW.",
    )

delta_col_1, delta_col_2 = st.columns(2)
with delta_col_1:
    use_deltas = st.checkbox(
        "Добавлять MFCC Δ и Δ²",
        value=True,
        help="Расширяет каждый кадр признаками динамики: MFCC + delta + delta-delta.",
    )
with delta_col_2:
    delta_width = st.number_input(
        "Окно Δ",
        min_value=3,
        max_value=25,
        value=9,
        step=2,
        disabled=not use_deltas,
    )

use_vosk = st.checkbox(
    "Проверять слово через Vosk перед DTW",
    value=True,
    help="Если Vosk распознает другое слово, результат сразу помечается как wrong_word с оценкой 0.",
)

if st.button("Запустить MVP", type="primary"):
    audio_source = recorded_attempt if recorded_attempt is not None else uploaded_attempt
    tmp_path = None
    resolved_attempt_path = attempt_path.strip()
    logger.info(
        "Classic page run requested: transcript='%s' use_vosk=%s input_mode=%s",
        transcript,
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
            logger.warning("Classic page run rejected: attempt path does not exist (%s)", resolved_attempt_path)
            st.error("Путь к аудио пользователя не существует")
        elif not anchor_set.has_required_anchors:
            logger.warning("Classic page run rejected: anchors are incomplete for word '%s'", anchor_word)
            st.error(
                "Нельзя запустить анализ: для слова отсутствуют обязательные якоря. "
                "Проверьте data/ref."
            )
        else:
            result = analyze(
                user_audio_path=resolved_attempt_path,
                transcript=transcript,
                n_mfcc=n_mfcc,
                frame_ms=frame_ms,
                hop_ms=hop_ms,
                use_deltas=use_deltas,
                delta_width=int(delta_width),
                sakoe_chiba_radius=int(sakoe_chiba_radius),
                use_vosk=use_vosk,
                max_anchors_per_class=max_anchors_preview,
            )

            logger.info(
                "Classic page analysis finished: status=%s score=%.2f verdict=%s",
                result.status,
                result.dtw_score,
                result.verdict,
            )

            st.success("MVP выполнен")
            result_payload = {
                "слово": anchor_word,
                "anchors": anchor_stats,
                "попытка_найдена": att_exists,
                "режим_ввода": "аудио_из_streamlit" if audio_source is not None else "путь_вручную",
                "использованный_путь_попытки": resolved_attempt_path,
                "n_mfcc": n_mfcc,
                "feature_dim": int(n_mfcc * (3 if use_deltas else 1)),
                "use_deltas": use_deltas,
                "delta_width": int(delta_width),
                "frame_ms": frame_ms,
                "hop_ms": hop_ms,
                "sakoe_chiba_radius": int(sakoe_chiba_radius),
                "use_vosk": use_vosk,
                "оценка_произношения": result.dtw_score,
                "вердикт": result.verdict,
                "status": result.status,
                "reason": result.reason,
                "dtw_дистанция": result.distance,
                "d100": result.d100,
                "d0": result.d0,
            }

            st.markdown("### Результат")
            metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
            with metric_col_1:
                st.metric("Оценка произношения", f"{result.dtw_score:.1f} / 100")
            with metric_col_2:
                st.metric("Вердикт", result.verdict)
            with metric_col_3:
                distance_text = "∞" if result.distance == float("inf") else f"{result.distance:.4f}"
                st.metric("DTW дистанция", distance_text)

            st.progress(int(max(0, min(100, round(result.dtw_score)))))
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

            with st.expander("Калибровка"):
                st.write({"d100": result.d100, "d0": result.d0})

            with st.expander("DEBUG"):
                st.write(result_payload)
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass
