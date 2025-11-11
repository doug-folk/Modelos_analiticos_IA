import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

import nltk
import yaml

LOGGER = logging.getLogger("trip_reviews")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def ensure_directories(paths: Iterable[str | Path]) -> None:
    for path in paths:
        Path(path).parent.mkdir(parents=True, exist_ok=True)


def timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def ensure_nltk_resources(resources: Iterable[str]) -> None:
    for resource in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            LOGGER.info("Downloading NLTK resource %s", resource)
            nltk.download(resource.split("/")[-1])


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)

