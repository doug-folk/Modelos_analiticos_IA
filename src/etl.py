from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Tuple
import zipfile

import pandas as pd

from .utils import ensure_directories, load_config, timestamp

LOGGER = logging.getLogger("trip_reviews.etl")


SEPARATORS = [";", ",", "|", "\t"]
ENCODINGS = ["utf-8", "latin-1", "cp1252"]


def detect_separator(sample: str) -> str:
    counts = {sep: sample.count(sep) for sep in SEPARATORS}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def read_csv_safely(path: Path) -> Tuple[pd.DataFrame, str, str]:
    for encoding in ENCODINGS:
        try:
            with path.open("r", encoding=encoding) as fh:
                header = fh.readline()
            separator = detect_separator(header)
            df = pd.read_csv(
                path,
                encoding=encoding,
                sep=separator,
                on_bad_lines="skip",
            )
            return df, encoding, separator
        except Exception:
            continue
    raise ValueError(f"Unable to read {path} with the supported encodings/separators")


def extract_zip(zip_path: Path, bronze_dir: Path) -> Path:
    run_dir = bronze_dir / timestamp()
    run_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Extracting %s to %s", zip_path, run_dir)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(run_dir)
    return run_dir


def consolidate_csvs(csv_paths: Iterable[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in csv_paths:
        if path.suffix.lower() != ".csv":
            continue
        try:
            df, encoding, separator = read_csv_safely(path)
            LOGGER.info(
                "Loaded %s (encoding=%s, sep='%s', rows=%d)",
                path.name,
                encoding,
                separator,
                len(df),
            )
        except ValueError as err:
            LOGGER.warning("Skipping %s: %s", path.name, err)
            continue
        df.columns = [col.strip().lower() for col in df.columns]
        df["attraction"] = path.stem
        frames.append(df)
    if not frames:
        raise RuntimeError("No CSV files could be loaded.")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates()
    return combined


def clean_dataframe(df: pd.DataFrame, text_column: str, label_column: str) -> pd.DataFrame:
    if text_column not in df.columns or label_column not in df.columns:
        raise KeyError("Expected columns not found in dataframe")
    df = df.copy()
    df[label_column] = pd.to_numeric(df[label_column], errors="coerce")
    df[label_column] = df[label_column].round().astype("Int64")
    df = df.dropna(subset=[text_column, label_column])
    df[text_column] = df[text_column].astype(str)
    return df.reset_index(drop=True)


def run_etl(config_path: str | Path) -> Path:
    config = load_config(config_path)
    zip_path = Path(config["raw_zip_path"])
    bronze_dir = Path(config["bronze_dir"])
    ensure_directories([Path(config["silver_dataset_path"])])
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    extracted_dir = extract_zip(zip_path, bronze_dir)
    df_raw = consolidate_csvs(extracted_dir.glob("*.csv"))
    df_clean = clean_dataframe(df_raw, config["text_column"], config["label_column"])
    silver_path = Path(config["silver_dataset_path"])
    silver_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(silver_path, index=False)
    LOGGER.info("Saved consolidated dataset to %s (%d rows)", silver_path, len(df_clean))
    return silver_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ETL pipeline for TripAdvisor reviews")
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline.yaml",
        help="Path to YAML configuration file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_etl(args.config)


if __name__ == "__main__":
    main()
