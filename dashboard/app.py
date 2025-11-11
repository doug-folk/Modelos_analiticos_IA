from __future__ import annotations

import json
import sys
import io
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.preprocess import PreprocessConfig, TextPreprocessor  # noqa: E402
from src.utils import load_config  # noqa: E402

CONFIG_PATH = ROOT / "config" / "pipeline.yaml"
THEME = {
    "bg": "#F7F9FB",
    "card": "#0F4C75",
    "accent": "#FF7E67",
    "text_dark": "#0B3954",
    "muted": "#8795AF",
}

CUSTOM_CSS = f"""
<style>
body {{background-color: {THEME['bg']};}}
[data-testid="stHeader"] {{background-color: transparent;}}
.hero {{
    background: linear-gradient(120deg, #0F4C75 0%, #1B6CA8 50%, #3282B8 100%);
    padding: 28px 36px;
    border-radius: 20px;
    color: white;
    margin-bottom: 1rem;
    box-shadow: 0 10px 30px rgba(15,76,117,0.35);
}}
.hero h1 {{margin: 0;font-size: 32px;}}
.hero p {{margin: 6px 0 0;font-size: 16px;opacity:0.9;}}
.stTabs [role="tablist"] {{
    gap: 8px;
    background-color: transparent;
}}
.stTabs [role="tab"] {{
    padding: 10px 18px;
    border-radius: 12px;
    border: 1px solid #cfd9e1;
}}
.stTabs [aria-selected="true"] {{
    background-color: #ffffff;
    border-color: #0F4C75;
    color: #0F4C75 !important;
}}
</style>
"""


@st.cache_data(show_spinner=False)
def load_dataset(dataset_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(dataset_path)
    df["nota"] = df["nota"].astype(int)
    if "data" in df.columns:
        df["data"] = pd.to_datetime(df["data"], errors="coerce")
    return df


@st.cache_resource(show_spinner=False)
def load_model_objects(config: dict) -> Tuple[TextPreprocessor, joblib.load, joblib.load]:
    preprocess_cfg = PreprocessConfig(
        text_column=config["text_column"],
        cleaned_column="review_cleaned",
        min_tokens=config.get("min_review_length", 1),
        apply_stemming=config.get("preprocessing", {}).get("apply_stemming", True),
    )
    preprocessor = TextPreprocessor(preprocess_cfg)
    vectorizer = joblib.load(config["artifacts"]["vectorizer_path"])
    model = joblib.load(config["artifacts"]["model_path"])
    return preprocessor, vectorizer, model


@st.cache_data(show_spinner=False)
def load_metrics(metrics_path: Path) -> dict:
    with Path(metrics_path).open("r", encoding="utf-8") as fp:
        return json.load(fp)


@st.cache_data(show_spinner=False)
def load_classification_report(report_path: Path) -> pd.DataFrame:
    with Path(report_path).open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    rows = {label: metrics for label, metrics in data.items() if label.isdigit()}
    return pd.DataFrame(rows).T


@st.cache_data(show_spinner=False)
def load_confusion_matrix(cm_path: Path) -> pd.DataFrame:
    return pd.read_csv(cm_path)


def format_confusion_matrix(df: pd.DataFrame) -> pd.DataFrame:
    matrix = df.set_index(df.columns[0]).apply(pd.to_numeric, errors="coerce")
    row_sums = matrix.sum(axis=1).replace(0, pd.NA)
    pct = (matrix.div(row_sums, axis=0) * 100).fillna(0)
    return pct.round(1)


def read_uploaded_csv(uploaded_file) -> pd.DataFrame | None:
    content = uploaded_file.getvalue()
    for sep in (",", ";", "|", "\t"):
        try:
            buffer = io.StringIO(content.decode("utf-8", errors="ignore"))
            if sep == ",":
                df = pd.read_csv(buffer)
            else:
                df = pd.read_csv(buffer, sep=sep)
            return df
        except Exception:
            continue
    return None


def predict_external_reviews(
    df: pd.DataFrame,
    source_column: str,
    preprocessor: TextPreprocessor,
    vectorizer,
    model,
) -> pd.DataFrame:
    if source_column not in df.columns:
        raise KeyError(f"Coluna '{source_column}' não encontrada no CSV enviado.")
    work_df = df.copy()
    target_column = preprocessor.config.text_column
    cleaned_col = preprocessor.config.cleaned_column
    if source_column != target_column:
        work_df = work_df.rename(columns={source_column: target_column})
    processed = preprocessor.transform(work_df[[target_column]])
    valid_idx = processed[processed[cleaned_col].str.strip() != ""].index
    if valid_idx.empty:
        return processed.iloc[0:0]
    X = vectorizer.transform(processed.loc[valid_idx, cleaned_col])
    preds = model.predict(X)
    try:
        probs = model.predict_proba(X)
        confidence = probs.max(axis=1)
    except Exception:
        confidence = None
    result = work_df.loc[valid_idx].copy()
    result["predicted_rating"] = preds
    if confidence is not None:
        result["confidence"] = confidence
    return result


@st.cache_data(show_spinner=False)
def score_sample(
    df: pd.DataFrame,
    _preprocessor: TextPreprocessor,
    _vectorizer,
    _model,
    sample_size: int = 500,
) -> pd.DataFrame:
    text_col = _preprocessor.config.text_column
    cleaned_col = _preprocessor.config.cleaned_column
    sample = df.sample(min(sample_size, len(df)), random_state=42).copy()
    sample[cleaned_col] = sample[text_col].apply(_preprocessor.clean_text)
    sample["review_length"] = sample[cleaned_col].apply(lambda x: len(x.split()))
    valid = sample[sample["review_length"] >= _preprocessor.config.min_tokens].copy()
    if valid.empty:
        return valid
    X = _vectorizer.transform(valid[cleaned_col])
    preds = _model.predict(X)
    try:
        probs = _model.predict_proba(X)
        confidence = probs.max(axis=1)
    except Exception:
        confidence = None
    valid["predicted_rating"] = preds
    if confidence is not None:
        valid["confidence"] = confidence
    return valid


def kpi_card(label: str, value: str, delta: str | None = None):
    st.markdown(
        f"""
        <div style="background-color:{THEME['card']};padding:18px;border-radius:16px;color:#F7F9FB;box-shadow: 0 4px 12px rgba(15,76,117,0.3);">
            <div style="font-size:14px;opacity:0.8;">{label}</div>
            <div style="font-size:32px;font-weight:700;">{value}</div>
            {f'<div style="font-size:13px;color:{THEME['accent']};">{delta}</div>' if delta else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_plotly(fig, title: str) -> None:
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=THEME["text_dark"])),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=THEME["text_dark"]),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_coloraxes(showscale=False)


def render_note_filter(notas_disponiveis: list[int]) -> list[int]:
    st.markdown(
        "<span style='font-weight:600;color:#0F4C75'>Filtrar notas exibidas:</span>",
        unsafe_allow_html=True,
    )
    note_cols = st.columns(len(notas_disponiveis))
    selection: list[int] = []
    for idx, nota in enumerate(notas_disponiveis):
        with note_cols[idx]:
            checked = st.checkbox(
                f"{int(nota)}",
                value=True,
                key=f"note_checkbox_{int(nota)}",
            )
            if checked:
                selection.append(nota)
    return selection


def download_image_button(label: str, path: str | None) -> None:
    if not path:
        return
    file_path = Path(path)
    if not file_path.exists():
        return
    with file_path.open("rb") as fp:
        st.download_button(
            label,
            data=fp.read(),
            file_name=file_path.name,
            mime="image/png",
        )


def main() -> None:
    st.set_page_config(page_title="Inteligência de Reviews", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
            <h1>Inteligência de Avaliações</h1>
            <p>KPIs balanceados, análises visuais e monitoramento de novos lotes em um único lugar.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    config = load_config(CONFIG_PATH)
    dataset = load_dataset(Path(config["silver_dataset_path"]))
    metrics = load_metrics(Path(config["artifacts"]["metrics_path"]))
    class_report = load_classification_report(Path(config["artifacts"]["classification_report_path"]))
    confusion = load_confusion_matrix(Path(config["artifacts"]["confusion_matrix_path"]))
    confusion_pct = format_confusion_matrix(confusion)

    preprocessor, vectorizer, model = load_model_objects(config)
    sample_predictions = score_sample(dataset, preprocessor, vectorizer, model)
    notas_disponiveis = sorted(dataset["nota"].unique().tolist())

    overview_tab, reviews_tab, upload_tab = st.tabs(
        ["Visão Geral", "Reviews & Testes", "Monitoramento CSV"]
    )

    selected_notes: list[int] = notas_disponiveis

    with overview_tab:
        kpi_cols = st.columns(3)
        kpi_card_cols = [
            ("Balanced Accuracy", f"{metrics['balanced_accuracy']*100:.2f}%"),
            ("Reviews Processadas", f"{len(dataset):,}".replace(",", ".")),
            ("Atrações Monitoradas", str(dataset["attraction"].nunique())),
        ]
        for col, data in zip(kpi_cols, kpi_card_cols):
            with col:
                kpi_card(*data)

        selected_notes = render_note_filter(notas_disponiveis)
        filtered_dataset = (
            dataset if not selected_notes else dataset[dataset["nota"].isin(selected_notes)]
        )
        filtered_samples = (
            sample_predictions
            if not selected_notes
            else sample_predictions[sample_predictions["nota"].isin(selected_notes)]
        )

        chart_col, matrix_col = st.columns((2.2, 1))
        with chart_col:
            attraction_counts = (
                filtered_dataset.groupby(["attraction", "nota"]).size().reset_index(name="total")
            )
            attraction_counts["nota"] = attraction_counts["nota"].astype(str)
            fig = px.bar(
                attraction_counts,
                x="attraction",
                y="total",
                color="nota",
                color_discrete_sequence=px.colors.sequential.Tealgrn,
            )
            style_plotly(fig, "Distribuição real de notas por atração")
            st.plotly_chart(fig, use_container_width=True)
        with matrix_col:
            st.markdown("#### Matriz de Confusão")
            st.dataframe(confusion_pct.style.format("{:.1f}%"))
            download_image_button(
                "Baixar matriz em PNG",
                config["artifacts"].get("confusion_matrix_fig_path"),
            )
            download_image_button(
                "Baixar distribuição de notas",
                config["artifacts"].get("distribution_fig_path"),
            )

        st.markdown("#### Resumo por Classe")
        st.dataframe(class_report[["precision", "recall", "f1-score"]])

    with reviews_tab:
        st.subheader("Reviews com previsão do modelo")
        sample_filtered = (
            sample_predictions
            if not selected_notes
            else sample_predictions[sample_predictions["nota"].isin(selected_notes)]
        )
        review_columns = [
            "attraction",
            "nota",
            preprocessor.config.text_column,
            "predicted_rating",
        ]
        if "confidence" in sample_filtered.columns:
            review_columns.append("confidence")
        if sample_filtered.empty:
            st.info("Nenhuma amostra disponível para as notas filtradas.")
        else:
            st.dataframe(
                sample_filtered[review_columns].rename(
                    columns={
                        preprocessor.config.text_column: "comentário",
                        "nota": "nota_real",
                    }
                )
            )
            st.download_button(
                "Baixar amostras filtradas",
                data=sample_filtered.to_csv(index=False).encode("utf-8"),
                file_name="reviews_filtradas.csv",
                mime="text/csv",
            )

        st.subheader("Teste uma nova avaliação")
        user_input = st.text_area("Cole aqui o texto da avaliação", height=120)
        if st.button("Classificar", key="btn_predict") and user_input:
            df_input = pd.DataFrame({preprocessor.config.text_column: [user_input]})
            processed = preprocessor.transform(df_input)
            X = vectorizer.transform(processed[preprocessor.config.cleaned_column])
            pred = model.predict(X)[0]
            try:
                conf = model.predict_proba(X)[0].max()
            except Exception:
                conf = None
            st.success(
                f"Nota prevista: {int(pred)}"
                + (f" — confiança {conf*100:.1f}%" if conf is not None else "")
            )

    with upload_tab:
        st.subheader("Monitorar um CSV novo")
        uploaded_file = st.file_uploader("Envie um arquivo CSV com novas avaliações", type=["csv"])
        custom_text_column = st.text_input(
            "Nome da coluna com o texto das avaliações",
            value=preprocessor.config.text_column,
        )
        if uploaded_file is not None:
            df_uploaded = read_uploaded_csv(uploaded_file)
            if df_uploaded is None:
                st.error("Não foi possível ler o CSV enviado. Verifique o separador e tente novamente.")
            else:
                try:
                    scored_upload = predict_external_reviews(
                        df_uploaded,
                        custom_text_column,
                        preprocessor,
                        vectorizer,
                        model,
                    )
                    if scored_upload.empty:
                        st.warning("Nenhuma linha com texto válido foi encontrada para classificação.")
                    else:
                        counts = (
                            scored_upload.groupby("predicted_rating")
                            .size()
                            .reset_index(name="total")
                            .sort_values("predicted_rating")
                        )
                        fig_upload = px.bar(
                            counts,
                            x="predicted_rating",
                            y="total",
                            color="predicted_rating",
                            color_discrete_sequence=px.colors.sequential.Aggrnyl,
                        )
                        style_plotly(fig_upload, "Distribuição prevista para o CSV enviado")
                        st.plotly_chart(fig_upload, use_container_width=True)
                        st.dataframe(scored_upload.head(50))
                        st.download_button(
                            "Baixar CSV com previsões",
                            data=scored_upload.to_csv(index=False).encode("utf-8"),
                            file_name="previsoes_monitoramento.csv",
                            mime="text/csv",
                        )
                except KeyError as exc:
                    st.error(str(exc))


if __name__ == "__main__":
    main()
