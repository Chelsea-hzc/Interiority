"""Input, cache, and JSONL helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .prompts import CANONICAL_LABELS


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "root": output_dir,
        "predictions": output_dir / "predictions",
        "diagnostics": output_dir / "behavioral_results",
        "final": output_dir / "final_test_results",
        "final_predictions": output_dir / "final_test_results" / "predictions",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def read_jsonl(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temp_path.replace(path)


def normalized_gold(value) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip().upper()
    return value if value in CANONICAL_LABELS else ""


def load_annotations(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"split", "paragraph", "gold_label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    return df


def sample_split(df: pd.DataFrame, split_name: str, pilot: bool, pilot_n: int, seed: int) -> pd.DataFrame:
    subset = df[df["split"].astype(str).str.lower() == split_name.lower()].copy()
    if pilot:
        subset = subset.sample(min(pilot_n, len(subset)), random_state=seed)
    return subset.reset_index(drop=True)


def dataframe_to_items(source_df: pd.DataFrame, split_name: str, prefix: str) -> List[dict]:
    items = []
    for pos, row in source_df.reset_index(drop=True).iterrows():
        source_id = row.get("index", pos)
        items.append(
            {
                "item_id": f"{prefix}_{source_id}",
                "title": row.get("title", ""),
                "paragraph": row.get("paragraph", ""),
                "gold_label": normalized_gold(row.get("gold_label", "")),
                "expected_label": "",
                "split": split_name,
                "notes": "",
            }
        )
    return items
