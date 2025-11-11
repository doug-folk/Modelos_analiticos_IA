from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import joblib
import pandas as pd

from .preprocess import PreprocessConfig, TextPreprocessor
from .utils import load_config


def prepare_dataframe(texts: List[str], text_column: str) -> pd.DataFrame:
    return pd.DataFrame({text_column: texts})


def load_inputs(args: argparse.Namespace, text_column: str) -> pd.DataFrame:
    if args.text:
        return prepare_dataframe([args.text], text_column)
    if args.input_file:
        input_path = Path(args.input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        if input_path.suffix == ".csv":
            df = pd.read_csv(input_path)
        elif input_path.suffix in {".parquet", ".pq"}:
            df = pd.read_parquet(input_path)
        else:
            raise ValueError("Supported input formats are CSV and Parquet")
        if text_column not in df.columns:
            raise KeyError(f"Column '{text_column}' not found in input data")
        return df[[text_column]].copy()
    raise ValueError("Either --text or --input-file must be provided")


def run_inference(config_path: str, args: argparse.Namespace) -> pd.DataFrame:
    config = load_config(config_path)

    preprocess_cfg = PreprocessConfig(
        text_column=config["text_column"],
        cleaned_column="review_cleaned",
        min_tokens=config.get("min_review_length", 1),
        apply_stemming=config.get("preprocessing", {}).get("apply_stemming", True),
    )
    preprocessor = TextPreprocessor(preprocess_cfg)

    df = load_inputs(args, preprocess_cfg.text_column)
    df_processed = preprocessor.transform(df)

    artifacts_cfg = config["artifacts"]
    model = joblib.load(artifacts_cfg["model_path"])
    vectorizer = joblib.load(artifacts_cfg["vectorizer_path"])

    X = vectorizer.transform(df_processed[preprocess_cfg.cleaned_column])
    predictions = model.predict(X)
    try:
        proba = model.predict_proba(X)
    except Exception:
        proba = None

    df_output = df.copy()
    df_output["predicted_rating"] = predictions
    if proba is not None:
        df_output["confidence"] = proba.max(axis=1)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix == ".csv":
            df_output.to_csv(output_path, index=False)
        else:
            df_output.to_parquet(output_path, index=False)
    return df_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with the trained model")
    parser.add_argument("--config", type=str, default="config/pipeline.yaml")
    parser.add_argument("--text", type=str, help="Single review text to score")
    parser.add_argument("--input-file", type=str, help="CSV or Parquet with reviews to score")
    parser.add_argument("--output", type=str, help="Optional path to store predictions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df_output = run_inference(args.config, args)
    print(df_output)


if __name__ == "__main__":
    main()
