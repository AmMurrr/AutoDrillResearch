from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st
from classic_approach.pipeline import analyze
from app.reference_db import init_db, list_reference_paths


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

use_vosk = st.checkbox(
    "Проверять слово через Vosk перед DTW",
    value=True,
    help="Если Vosk распознает другое слово, результат сразу помечается как wrong_word с оценкой 0.",
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
            result = analyze(
                user_audio_path=resolved_attempt_path,
                reference_audio_path=selected_reference_paths,
                transcript=transcript,
                n_mfcc=n_mfcc,
                frame_ms=frame_ms,
                hop_ms=hop_ms,
                sakoe_chiba_radius=int(sakoe_chiba_radius),
                use_vosk=use_vosk,
            )
            st.success("MVP выполнен")
            result_payload = {
                "режим_выбора_эталонов": reference_mode,
                "слово_поиска": reference_word.strip(),
                "использованные_эталоны": selected_reference_paths,
                "количество_эталонов": len(selected_reference_paths),
                "попытка_найдена": att_exists,
                "режим_ввода": "аудио_из_streamlit" if audio_source is not None else "путь_вручную",
                "использованный_путь_попытки": resolved_attempt_path,
                "n_mfcc": n_mfcc,
                "frame_ms": frame_ms,
                "hop_ms": hop_ms,
                "sakoe_chiba_radius": int(sakoe_chiba_radius),
                "use_vosk": use_vosk,
                "оценка_произношения": result.dtw_score,
                "вердикт": result.verdict,
                "status": result.status,
                "reason": result.reason,
                "проблемные_зоны": result.problematic_phonemes,
                "dtw_дистанция": result.distance,
                "локализация_ошибок": result.error_localization,
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
            else:
                _render_verdict_block(result.verdict)

            st.markdown("#### Локализация ошибок")
            if result.error_localization:
                rows = []
                for item in result.error_localization:
                    zone_errors = item.get("zone_errors", {})
                    rows.append(
                        {
                            "слово": item.get("word", ""),
                            "зона_проблемы": item.get("problem_zone", ""),
                            "статус": "проблема" if item.get("is_problematic") else "норма",
                            "относительная_ошибка": item.get("relative_error", 0.0),
                            "ошибка_начало": zone_errors.get("начало", zone_errors.get("beginning", 0.0)),
                            "ошибка_середина": zone_errors.get("середина", zone_errors.get("middle", 0.0)),
                            "ошибка_конец": zone_errors.get("конец", zone_errors.get("end", 0.0)),
                        }
                    )
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.info("Явных локальных проблем не обнаружено на уровне псевдо-align по DTW.")

            if result.problematic_phonemes:
                st.markdown("#### Краткий список проблемных зон")
                st.write(", ".join(result.problematic_phonemes))
            else:
                st.info("Критичных проблемных зон не выделено.")

            with st.expander("DEBUG"):
                st.write(result_payload)
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass


