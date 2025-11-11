from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from .preprocess import PreprocessConfig, TextPreprocessor
from .utils import ensure_directories, load_config, save_json

LOGGER = logging.getLogger("trip_reviews.train")


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def build_pipeline(config: Dict) -> None:
    dataset_path = Path(config["silver_dataset_path"])
    df = load_dataset(dataset_path)

    preprocess_cfg = PreprocessConfig(
        text_column=config["text_column"],
        cleaned_column="review_cleaned",
        min_tokens=config.get("min_review_length", 1),
        apply_stemming=config.get("preprocessing", {}).get("apply_stemming", True),
    )
    preprocessor = TextPreprocessor(preprocess_cfg)
    df_processed = preprocessor.transform(df)
    df_processed = df_processed[df_processed[preprocess_cfg.cleaned_column].str.strip() != ""]

    vectorizer_cfg = config.get("vectorizer", {})
    vectorizer = TfidfVectorizer(
        max_features=vectorizer_cfg.get("max_features", 5000),
        ngram_range=tuple(vectorizer_cfg.get("ngram_range", [1, 2])),
        max_df=vectorizer_cfg.get("max_df", 1.0),
        min_df=vectorizer_cfg.get("min_df", 1),
        sublinear_tf=vectorizer_cfg.get("sublinear_tf", False),
    )
    X = vectorizer.fit_transform(df_processed[preprocess_cfg.cleaned_column])
    y = df_processed[config["label_column"]].astype(int)

    train_cfg = config.get("train", {})
    stratify = y if train_cfg.get("stratify", True) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=train_cfg.get("test_size", 0.2),
        random_state=train_cfg.get("random_state", 42),
        stratify=stratify,
    )

    model_params = train_cfg.get("model", {}).copy()
    model_params.setdefault("random_state", train_cfg.get("random_state", 42))
    model_grid = train_cfg.get("model_grid", {})
    grid_c_values = model_grid.get("C") if model_grid else None

    params_to_try = []
    if grid_c_values:
        for value in grid_c_values:
            candidate_params = model_params.copy()
            candidate_params["C"] = value
            params_to_try.append(candidate_params)
    else:
        params_to_try.append(model_params)

    grid_results = []
    best_model = None
    best_params = None
    best_bal_acc = -1.0
    best_predictions = None

    for params in params_to_try:
        candidate = LogisticRegression(**params)
        candidate.fit(X_train, y_train)
        candidate_pred = candidate.predict(X_test)
        candidate_bal_acc = balanced_accuracy_score(y_test, candidate_pred)
        grid_results.append(
            {
                "C": params.get("C"),
                "balanced_accuracy": round(candidate_bal_acc, 4),
            }
        )
        if candidate_bal_acc > best_bal_acc:
            best_bal_acc = candidate_bal_acc
            best_model = candidate
            best_params = params.copy()
            best_predictions = candidate_pred

    if best_model is None or best_predictions is None:
        raise RuntimeError("Grid search did not produce a trained model.")

    LOGGER.info("Balanced accuracy (best C=%s): %.4f", best_params.get("C"), best_bal_acc)

    report = classification_report(y_test, best_predictions, output_dict=True)
    labels_sorted = sorted(int(value) for value in y.unique())
    cm = confusion_matrix(y_test, best_predictions, labels=labels_sorted)
    cm_sum = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_pct = np.divide(cm.astype(float), cm_sum, where=cm_sum != 0) * 100
    cm_pct = np.nan_to_num(cm_pct)
    cm_pct = np.clip(cm_pct, 0, 100)

    artifacts_cfg = config["artifacts"]
    hyperparam_path = artifacts_cfg.get("hyperparam_results_path")
    ensure_directories(
        [
            artifacts_cfg["model_path"],
            artifacts_cfg["vectorizer_path"],
            artifacts_cfg["metrics_path"],
            artifacts_cfg["confusion_matrix_path"],
            artifacts_cfg["classification_report_path"],
            artifacts_cfg["label_mapping_path"],
            artifacts_cfg["vocabulary_preview_path"],
            hyperparam_path if hyperparam_path else artifacts_cfg["metrics_path"],
        ]
    )

    joblib.dump(best_model, artifacts_cfg["model_path"])
    joblib.dump(vectorizer, artifacts_cfg["vectorizer_path"])

    metrics = {
        "balanced_accuracy": round(best_bal_acc, 4),
        "best_C": best_params.get("C"),
        "samples_train": int(len(y_train)),
        "samples_test": int(len(y_test)),
    }
    save_json(metrics, artifacts_cfg["metrics_path"])
    save_json(report, artifacts_cfg["classification_report_path"])
    if hyperparam_path:
        save_json({"results": grid_results}, hyperparam_path)

    cm_df = pd.DataFrame(
        cm_pct.round(2),
        index=[f"true_{label}" for label in labels_sorted],
        columns=[f"pred_{label}" for label in labels_sorted],
    )
    cm_df.to_csv(artifacts_cfg["confusion_matrix_path"], index=True)

    label_mapping = {idx: label for idx, label in enumerate(labels_sorted)}
    save_json(label_mapping, artifacts_cfg["label_mapping_path"])

    feature_names = vectorizer.get_feature_names_out()
    tf_totals = np.asarray(X.sum(axis=0)).ravel()
    vocab_df = (
        pd.DataFrame({"token": feature_names, "importance": tf_totals})
        .sort_values(by="importance", ascending=False)
        .head(50)
    )
    vocab_df.to_csv(artifacts_cfg["vocabulary_preview_path"], index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train logistic regression model on TripAdvisor reviews")
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline.yaml",
        help="Path to YAML configuration",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    build_pipeline(config)


if __name__ == "__main__":
    main()
