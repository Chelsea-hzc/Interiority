"""Parsing and metric helpers preserved from the notebook logic."""

from __future__ import annotations

import json
import re
from typing import Dict, List

import pandas as pd

from .prompts import CANONICAL_LABELS


ITEM_LEVEL_COLUMNS = [
    "run_id",
    "timestamp",
    "model_name",
    "model_version",
    "temperature",
    "test_name",
    "prompt_version",
    "split",
    "item_id",
    "title",
    "paragraph",
    "gold_label",
    "expected_label",
    "model_output_raw",
    "_thoughts",
    "parsed_label",
    "valid_output",
    "correct_against_expected",
    "correct_against_gold",
    "matches_baseline",
    "notes",
]

SUMMARY_METRIC_COLUMNS = [
    "run_id",
    "model_name",
    "model_version",
    "test_name",
    "prompt_version",
    "split",
    "n_items",
    "valid_output_rate",
    "accuracy",
    "micro_f1",
    "macro_f1",
    "weighted_f1",
    "high_f1",
    "low_f1",
    "none_f1",
    "agreement_with_full",
    "consistency_rate",
    "label_distribution",
    "invalid_output_count",
    "notes",
]

MODEL_COMPARISON_COLUMNS = [
    "model_name",
    "thinking_level",
    "legal_valid_rate",
    "example_recovery_accuracy",
    "order_consistency_rate",
    "generic_label_consistency_rate",
    "full_codebook_dev_accuracy",
    "full_codebook_dev_micro_f1",
    "full_codebook_dev_macro_f1",
    "full_codebook_dev_weighted_f1",
    "recommendation_notes",
]


def parse_answer(raw_answer: str, allowed_labels: List[str]) -> str | None:
    if not raw_answer or str(raw_answer).startswith("ERROR:"):
        return None
    text = str(raw_answer).upper().replace(r"\_", "_")
    sorted_labels = sorted({str(label).upper() for label in allowed_labels}, key=len, reverse=True)
    pattern = "|".join(re.escape(label) for label in sorted_labels)
    match = re.search(pattern, text)
    return match.group(0) if match else None


def parse_answers(raw_answers: List[str], allowed_labels: List[str]) -> List[str | None]:
    return [parse_answer(answer, allowed_labels) for answer in raw_answers]


def is_exact_valid_label(raw_answer: str, allowed_labels: List[str]) -> bool:
    if not raw_answer or str(raw_answer).startswith("ERROR:"):
        return False
    stripped = str(raw_answer).strip().upper()
    return stripped in {str(label).upper() for label in allowed_labels}


def per_label_f1(y_true: List[str], y_pred: List[str], labels: List[str] = CANONICAL_LABELS) -> Dict[str, float]:
    scores = {}
    for label in labels:
        tp = sum((true == label) and (pred == label) for true, pred in zip(y_true, y_pred))
        fp = sum((true != label) and (pred == label) for true, pred in zip(y_true, y_pred))
        fn = sum((true == label) and (pred != label) for true, pred in zip(y_true, y_pred))
        denom = (2 * tp) + fp + fn
        scores[label] = (2 * tp / denom) if denom else 0.0
    return scores


def classification_metrics(records: List[dict], target_field: str) -> dict:
    usable = [record for record in records if record.get(target_field)]
    if not usable:
        return {}

    y_true = [record[target_field] for record in usable]
    y_pred = [record.get("parsed_label", "") for record in usable]
    n = len(usable)
    accuracy = sum(true == pred for true, pred in zip(y_true, y_pred)) / n if n else ""
    f1s = per_label_f1(y_true, y_pred)

    total_tp = total_fp = total_fn = 0
    supports = {}
    for label in CANONICAL_LABELS:
        tp = sum((true == label) and (pred == label) for true, pred in zip(y_true, y_pred))
        fp = sum((true != label) and (pred == label) for true, pred in zip(y_true, y_pred))
        fn = sum((true == label) and (pred != label) for true, pred in zip(y_true, y_pred))
        total_tp += tp
        total_fp += fp
        total_fn += fn
        supports[label] = sum(true == label for true in y_true)

    # Invalid predictions are still false positives in single-label multiclass scoring.
    total_fp += sum(pred not in CANONICAL_LABELS for pred in y_pred)
    micro_denom = (2 * total_tp) + total_fp + total_fn
    micro_f1 = (2 * total_tp / micro_denom) if micro_denom else 0.0
    macro_f1 = sum(f1s.values()) / len(CANONICAL_LABELS)
    total_support = sum(supports.values())
    weighted_f1 = (
        sum(f1s[label] * supports[label] for label in CANONICAL_LABELS) / total_support
        if total_support
        else 0.0
    )
    return {
        "accuracy": accuracy,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "high_f1": f1s["HIGH"],
        "low_f1": f1s["LOW"],
        "none_f1": f1s["NONE"],
    }


def label_distribution(records: List[dict]) -> str:
    counts = pd.Series([record.get("parsed_label", "") or "INVALID" for record in records]).value_counts().to_dict()
    return json.dumps(counts, sort_keys=True)


def summary_row(
    run_id: str,
    model: str,
    temperature: float,
    test_name: str,
    prompt_version: str,
    records: List[dict],
    split: str,
    target_field: str | None = None,
    agreement_with_full="",
    consistency_rate="",
    notes="",
) -> dict:
    n_items = len(records)
    invalid_count = sum(not bool(record.get("valid_output")) for record in records)
    valid_rate = (n_items - invalid_count) / n_items if n_items else ""
    row = {
        "run_id": run_id,
        "model_name": model,
        "model_version": model,
        "test_name": test_name,
        "prompt_version": prompt_version,
        "split": split,
        "n_items": n_items,
        "valid_output_rate": valid_rate,
        "accuracy": "",
        "micro_f1": "",
        "macro_f1": "",
        "weighted_f1": "",
        "high_f1": "",
        "low_f1": "",
        "none_f1": "",
        "agreement_with_full": agreement_with_full,
        "consistency_rate": consistency_rate,
        "label_distribution": label_distribution(records),
        "invalid_output_count": invalid_count,
        "notes": notes,
    }
    if target_field:
        row.update(classification_metrics(records, target_field))
    return row


def confusion_matrix_from_records(records: List[dict], model: str) -> pd.DataFrame:
    rows = []
    for gold in CANONICAL_LABELS:
        subset = [record for record in records if record.get("gold_label") == gold]
        row = {"model_name": model, "gold_label": gold}
        for pred in CANONICAL_LABELS + ["INVALID"]:
            row[f"pred_{pred.lower()}"] = sum((record.get("parsed_label") or "INVALID") == pred for record in subset)
        rows.append(row)
    return pd.DataFrame(rows)
