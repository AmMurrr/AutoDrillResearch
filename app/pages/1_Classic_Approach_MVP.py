from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st
from classic_approach.pipeline import analyze
from app.reference_db import init_db, list_reference_paths


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

st.markdown("### Аудио из браузера")
uploaded_attempt = st.file_uploader(
    "Загрузите аудио пользователя (опционально, приоритет над путем)",
    type=["wav", "mp3", "m4a", "ogg", "flac"],
    key="classic_uploaded_attempt",
)
recorded_attempt = st.audio_input("Или запишите попытку через микрофон", key="classic_recorded_attempt")

transcript = st.text_input("Ожидаемый текст фразы", value="Hello")

st.markdown("### Параметры")
col1, col2, col3 = st.columns(3)
with col1:
    n_mfcc = st.slider("MFCC коэффициенты", min_value=13, max_value=40, value=20)
with col2:
    frame_ms = st.slider("Длина окна (ms)", min_value=10, max_value=50, value=25)
with col3:
    hop_ms = st.slider("Шаг окна (ms)", min_value=5, max_value=20, value=10)

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

    try:
        ref_exists = Path(reference_path).exists()
        att_exists = Path(resolved_attempt_path).exists()

        if not ref_exists or not att_exists:
            st.error("Проверьте пути: эталон и аудио пользователя должны существовать")
        else:
            result = analyze(
                user_audio_path=resolved_attempt_path,
                reference_audio_path=reference_path,
                transcript=transcript,
                n_mfcc=n_mfcc,
                frame_ms=frame_ms,
                hop_ms=hop_ms,
            )
            st.success("MVP выполнен")
            result_payload = {
                "эталон_найден": ref_exists,
                "попытка_найдена": att_exists,
                "режим_ввода": "аудио_из_streamlit" if audio_source is not None else "путь_вручную",
                "использованный_путь_попытки": resolved_attempt_path,
                "n_mfcc": n_mfcc,
                "frame_ms": frame_ms,
                "hop_ms": hop_ms,
                "оценка_произношения": result.dtw_score,
                "вердикт": result.verdict,
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


