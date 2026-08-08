"""Diagnostic and final zero-shot runners."""

from __future__ import annotations

import copy
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pandas as pd

from .gemini_client import GeminiRunnerClient
from .io_utils import dataframe_to_items, read_jsonl, safe_name, sample_split, write_jsonl
from .metrics import (
    ITEM_LEVEL_COLUMNS,
    MODEL_COMPARISON_COLUMNS,
    SUMMARY_METRIC_COLUMNS,
    confusion_matrix_from_records,
    is_exact_valid_label,
    parse_answer,
    parse_answers,
    summary_row,
)
from .prompts import (
    CANONICAL_LABELS,
    GENERIC_LABELS,
    GENERIC_TO_CANONICAL,
    build_full_prompt,
    build_generic_prompt,
    build_reordered_category_prompt,
    build_reordered_label_prompt,
    codebook_example_items,
)

PromptBuilder = Callable[[str], Tuple[str, str]]

DEFAULT_MODEL_PRICES = {
    "gemini-3-flash-preview": {
        "input_cost_per_1m_tokens": 0.50,
        "output_cost_per_1m_tokens": 3.00,
    },
    "gemini-3.5-flash": {
        "input_cost_per_1m_tokens": 1.50,
        "output_cost_per_1m_tokens": 9.00,
    },
    "gemini-3.1-pro-preview": {
        "input_cost_per_1m_tokens": 2.00,
        "output_cost_per_1m_tokens": 12.00,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_error(raw_output: str) -> bool:
    return str(raw_output or "").startswith("ERROR:")


def elapsed_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{seconds:.1f}s"


def empty_usage() -> dict:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "thinking_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "cost_available": True,
    }


def cost_for_usage(usage: dict, cost_rates: dict[str, float | None]) -> tuple[float, bool]:
    input_rate = cost_rates["input_cost_per_1m_tokens"]
    output_rate = cost_rates["output_cost_per_1m_tokens"]
    if input_rate is None or output_rate is None:
        return 0.0, False
    cost = (usage["input_tokens"] / 1_000_000 * input_rate) + (usage["output_tokens"] / 1_000_000 * output_rate)
    return cost, True


def usage_with_cost(usage: dict, cost_rates: dict[str, float | None]) -> dict:
    normalized = empty_usage()
    for key in ("input_tokens", "output_tokens", "thinking_tokens", "total_tokens"):
        normalized[key] = int(usage.get(key, 0) or 0)
    cost, cost_available = cost_for_usage(normalized, cost_rates)
    normalized["estimated_cost_usd"] = cost
    normalized["cost_available"] = cost_available
    return normalized


def add_usage(total: dict, usage: dict) -> None:
    for key in ("input_tokens", "output_tokens", "thinking_tokens", "total_tokens"):
        total[key] += int(usage.get(key, 0) or 0)
    total["estimated_cost_usd"] += float(usage.get("estimated_cost_usd", 0.0) or 0.0)
    total["cost_available"] = total.get("cost_available", True) and bool(usage.get("cost_available", True))


def subtract_usage(total: dict, previous: dict) -> dict:
    usage = empty_usage()
    for key in ("input_tokens", "output_tokens", "thinking_tokens", "total_tokens"):
        usage[key] = total.get(key, 0) - previous.get(key, 0)
    usage["estimated_cost_usd"] = total.get("estimated_cost_usd", 0.0) - previous.get("estimated_cost_usd", 0.0)
    usage["cost_available"] = total.get("cost_available", True)
    return usage


def copy_usage(usage: dict) -> dict:
    copied = empty_usage()
    copied.update(usage)
    return copied


def format_cost(usage: dict) -> str:
    if not usage.get("cost_available", True):
        return "cost unavailable"
    return f"${usage.get('estimated_cost_usd', 0.0):.6f}"


def counts_for(records: List[dict]) -> dict:
    valid_count = sum(bool(row.get("valid_output")) for row in records)
    error_count = sum(is_error(row.get("model_output_raw", "")) for row in records)
    invalid_count = len(records) - valid_count - error_count
    bad_count = len(records) - valid_count
    bad_rate = bad_count / len(records) if records else 0.0
    return {
        "rows": len(records),
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "error_count": error_count,
        "invalid_error_rate": bad_rate,
    }


def print_progress(processed: int, total: int, records: List[dict]) -> None:
    stats = counts_for(records)
    print(
        "Progress: "
        f"{processed}/{total} rows | "
        f"valid={stats['valid_count']} | "
        f"invalid={stats['invalid_count']} | "
        f"errors={stats['error_count']} | "
        f"invalid/error rate={stats['invalid_error_rate']:.3f}",
        flush=True,
    )


def print_usage_progress(
    stage: str,
    processed: int,
    total: int,
    interval_usage: dict,
    total_usage: dict,
    started_at: float,
) -> None:
    print(
        f"{stage} cost: "
        f"rows={processed}/{total} | "
        f"interval={format_cost(interval_usage)} | "
        f"total={format_cost(total_usage)} | "
        f"input_tokens={total_usage['input_tokens']} | "
        f"output_tokens={total_usage['output_tokens']} | "
        f"thinking_tokens={total_usage['thinking_tokens']} | "
        f"elapsed={elapsed_time(time.monotonic() - started_at)}",
        flush=True,
    )


def write_run_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True, default=str)
    temp_path.replace(path)


def read_json_file(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class EvaluationRunner:
    def __init__(
        self,
        df: pd.DataFrame,
        models: List[str],
        client: GeminiRunnerClient,
        output_paths: dict[str, Path],
        seed: int = 42,
        pilot: bool = False,
        pilot_n: int = 5,
        max_output_tokens: int = 4096,
        status_every: int = 30,
        input_cost_per_1m_tokens: float | None = None,
        output_cost_per_1m_tokens: float | None = None,
        resume: bool = False,
        resume_command: str = "",
        thinking_levels: List[str] | None = None,
    ):
        self.df = df
        self.client = client
        self.model_ids = list(models)
        self.thinking_levels = list(dict.fromkeys(thinking_levels or [client.config.classify_thinking_level]))
        show_level_in_name = len(self.thinking_levels) > 1
        variants = []
        for model_id in self.model_ids:
            for thinking_level in self.thinking_levels:
                name = f"{model_id} [thinking={thinking_level}]" if show_level_in_name else model_id
                variants.append(
                    {
                        "name": name,
                        "model_id": model_id,
                        "thinking_level": thinking_level,
                    }
                )
        self.models = [variant["name"] for variant in variants]
        self._variants_by_name = {variant["name"]: variant for variant in variants}
        self.paths = output_paths
        self.seed = seed
        self.pilot = pilot
        self.pilot_n = pilot_n
        self.temperature = "model_default"
        self.max_output_tokens = max_output_tokens
        self.status_every = status_every
        self.input_cost_per_1m_tokens = input_cost_per_1m_tokens
        self.output_cost_per_1m_tokens = output_cost_per_1m_tokens
        self.resume = resume
        self.resume_command = resume_command

    def _variant(self, model: str) -> dict[str, str]:
        variants = getattr(self, "_variants_by_name", {})
        return variants.get(
            model,
            {
                "name": model,
                "model_id": model,
                "thinking_level": self.client.config.classify_thinking_level,
            },
        )

    def _activate_variant(self, model: str) -> str:
        variant = self._variant(model)
        self.client.config.classify_thinking_level = variant["thinking_level"]
        return variant["model_id"]

    def _cost_rates(self, model: str) -> dict[str, float | None]:
        model_prices = DEFAULT_MODEL_PRICES.get(self._variant(model)["model_id"].lower(), {})
        return {
            "input_cost_per_1m_tokens": (
                self.input_cost_per_1m_tokens
                if self.input_cost_per_1m_tokens is not None
                else model_prices.get("input_cost_per_1m_tokens")
            ),
            "output_cost_per_1m_tokens": (
                self.output_cost_per_1m_tokens
                if self.output_cost_per_1m_tokens is not None
                else model_prices.get("output_cost_per_1m_tokens")
            ),
        }

    def _cache_path(self, model: str, condition: str, final: bool = False) -> Path:
        folder = self.paths["final_predictions"] if final else self.paths["predictions"]
        suffix = "_pilot" if self.pilot else ""
        return folder / f"{safe_name(model)}_{condition}{suffix}.jsonl"

    def _get_or_compute(self, path: Path, compute_fn) -> List[dict]:
        cached = read_jsonl(path)
        if cached is not None:
            print(f"  [cache hit] {path.name}: {len(cached)} records")
            return cached
        print(f"  [computing] {path.name}")
        records = compute_fn()
        write_jsonl(path, records)
        print(f"  Saved {len(records)} records to {path}")
        return records

    def _final_record(self, run_id: str, timestamp: str, model: str, item: dict, raw: str, thoughts: str = "") -> dict:
        valid_output = is_exact_valid_label(raw, CANONICAL_LABELS)
        parsed_label = str(raw).strip().upper() if valid_output else ""
        gold = item.get("gold_label", "")
        return {
            "run_id": run_id,
            "timestamp": timestamp,
            "model_name": model,
            "model_version": self._variant(model)["model_id"],
            "temperature": self.temperature,
            "test_name": "Final Zero-Shot Test",
            "prompt_version": "full_codebook",
            "split": "test",
            "item_id": item["item_id"],
            "title": item.get("title", ""),
            "paragraph": item["paragraph"],
            "gold_label": gold,
            "expected_label": "",
            "model_output_raw": raw,
            "_thoughts": thoughts,
            "parsed_label": parsed_label,
            "valid_output": valid_output,
            "correct_against_expected": "",
            "correct_against_gold": (parsed_label == gold) if gold else "",
            "matches_baseline": "",
            "notes": "",
        }

    def _run_final_item(self, run_id: str, timestamp: str, model: str, item: dict) -> tuple[dict, dict, dict]:
        prompt = build_full_prompt(item["paragraph"])
        model_id = self._activate_variant(model)
        raw = self.client.batch_generate(
            model_id,
            [prompt],
            max_tokens=self.max_output_tokens,
            show_progress=False,
            retry_on_error=True,
            classify=True,
            response_labels=CANONICAL_LABELS,
        )[0]
        usage = copy_usage(self.client.last_usage)
        request_metadata = (
            self.client.last_batch_request_metadata[0]
            if self.client.last_batch_request_metadata
            else {
                "attempts": 1,
                "failed": is_error(raw),
                "retry_exhausted": False,
                "error_message": raw if is_error(raw) else "",
            }
        )
        return (
            self._final_record(run_id, timestamp, model, item, raw, self.client.last_thoughts),
            usage,
            request_metadata,
        )

    def _record_api_failure(
        self,
        failures: list[dict],
        failures_path: Path,
        stage: str,
        model: str,
        item: dict,
        item_index: int,
        raw_output: str,
        request_metadata: dict,
    ) -> None:
        failures.append(
            {
                "timestamp": utc_now(),
                "stage": stage,
                "model": model,
                "item_index": item_index,
                "item_id": item.get("item_id", ""),
                "attempts": int(request_metadata.get("attempts", 1) or 1),
                "retry_exhausted": bool(request_metadata.get("retry_exhausted")),
                "error_message": request_metadata.get("error_message") or raw_output,
            }
        )
        write_run_config(failures_path, failures)

    def _write_resume_state(
        self,
        path: Path,
        run_id: str,
        timestamp: str,
        model_index: int,
        next_item_index: int,
        status: str,
        usage_total: dict,
        runtime_seconds: float,
    ) -> None:
        write_run_config(
            path,
            {
                "run_id": run_id,
                "timestamp": timestamp,
                "models": self.models,
                "model_variants": list(self._variants_by_name.values()),
                "model_index": model_index,
                "next_item_index": next_item_index,
                "status": status,
                "api_usage_total": copy_usage(usage_total),
                "runtime_seconds": round(runtime_seconds, 3),
                "resume_command": self.resume_command,
                "updated_at": utc_now(),
            },
        )

    def _print_resume_command(self) -> None:
        print("Resume command:", flush=True)
        print(self.resume_command or "Re-run the same command with --resume", flush=True)

    def _save_final_outputs(
        self,
        records: List[dict],
        run_config: dict,
        status: str,
        prediction_cache_paths: dict[str, Path] | None = None,
    ) -> dict[str, Path | str]:
        item_df = pd.DataFrame(records, columns=ITEM_LEVEL_COLUMNS).fillna("")

        summary_rows = []
        confusion_rows = []
        for model in self.models:
            model_records = [record for record in records if record.get("model_name") == model]
            if not model_records:
                continue
            summary_rows.append(
                summary_row(
                    run_config["run_id"],
                    model,
                    self.temperature,
                    "Final Zero-Shot Test",
                    "full_codebook",
                    model_records,
                    "test",
                    target_field="gold_label",
                )
            )
            confusion_rows.append(confusion_matrix_from_records(model_records, model))

        summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_METRIC_COLUMNS).fillna("")
        comparison_columns = [
            "model_name",
            "thinking_level",
            "model_version",
            "n_items",
            "valid_output_rate",
            "accuracy",
            "micro_f1",
            "macro_f1",
            "weighted_f1",
            "high_f1",
            "low_f1",
            "none_f1",
            "label_distribution",
            "invalid_output_count",
        ]
        if len(summary_df):
            comparison_df = summary_df.drop(columns=["model_version"]).copy()
            comparison_df.insert(1, "thinking_level", comparison_df["model_name"].map(lambda name: self._variant(name)["thinking_level"]))
            comparison_df.insert(2, "model_version", comparison_df["model_name"].map(lambda name: self._variant(name)["model_id"]))
            comparison_df = comparison_df[comparison_columns]
        else:
            comparison_df = pd.DataFrame(columns=comparison_columns)
        if len(comparison_df):
            comparison_df = comparison_df.sort_values(
                ["macro_f1", "weighted_f1", "accuracy"],
                ascending=False,
            ).reset_index(drop=True)
        confusion_df = pd.concat(confusion_rows, ignore_index=True).fillna("") if confusion_rows else pd.DataFrame()

        outputs = {
            "item_level_predictions": self.paths["final"] / "item_level_predictions.csv",
            "final_item_level": self.paths["final"] / "final_test_item_level_predictions.csv",
            "final_summary": self.paths["final"] / "final_test_summary_metrics.csv",
            "final_comparison": self.paths["final"] / "final_test_model_comparison.csv",
            "final_confusion": self.paths["final"] / "final_test_confusion_matrices.csv",
            "failed_items": "",
            "failures_json": self.paths["final"] / "failures.json",
            "resume_state": self.paths["final"] / "resume_state.json",
            "run_config": self.paths["final"] / "run_config.json",
        }
        item_df.to_csv(outputs["item_level_predictions"], index=False)
        item_df.to_csv(outputs["final_item_level"], index=False)
        summary_df.to_csv(outputs["final_summary"], index=False)
        comparison_df.to_csv(outputs["final_comparison"], index=False)
        confusion_df.to_csv(outputs["final_confusion"], index=False)

        failed = item_df[
            (~item_df["valid_output"].astype(bool))
            | (item_df["model_output_raw"].fillna("").astype(str).str.startswith("ERROR:"))
        ]
        if len(failed):
            failed_path = self.paths["final"] / "failed_items.csv"
            failed.to_csv(failed_path, index=False)
            outputs["failed_items"] = failed_path

        if prediction_cache_paths:
            for model, cache_path in prediction_cache_paths.items():
                model_records = [record for record in records if record.get("model_name") == model]
                if model_records:
                    write_jsonl(cache_path, model_records)

        run_config.update(
            {
                "status": status,
                "updated_at": utc_now(),
                "number_of_rows_processed": int(len(records)),
                "outputs": {key: str(value) for key, value in outputs.items()},
            }
        )
        write_run_config(outputs["run_config"], run_config)
        return {key: value for key, value in outputs.items() if value}

    def _save_pilot_outputs(self, pilot_records: List[dict], run_config: dict, status: str) -> Path:
        pilot_path = self.paths["final"] / "pilot_item_level_predictions.csv"
        pd.DataFrame(pilot_records, columns=ITEM_LEVEL_COLUMNS).fillna("").to_csv(pilot_path, index=False)
        run_config.update(
            {
                "status": status,
                "updated_at": utc_now(),
                "pilot_rows_processed": int(len(pilot_records)),
                "pilot_item_level_predictions": str(pilot_path),
            }
        )
        write_run_config(self.paths["final"] / "run_config.json", run_config)
        return pilot_path

    @staticmethod
    def _condition_name(test_name: str, prompt_version: str, split: str) -> str:
        return safe_name(f"prezero_{test_name}_{prompt_version}_{split}").lower()

    def _run_items_for_prompt(
        self,
        run_id: str,
        timestamp: str,
        model: str,
        test_name: str,
        prompt_version: str,
        items: List[dict],
        prompt_builder: PromptBuilder,
        raw_allowed_labels: List[str],
        parsed_label_map: Dict[str, str] | None = None,
        legal_label_strict: bool = False,
    ) -> List[dict]:
        split = "+".join(sorted({item["split"] for item in items}))
        condition = self._condition_name(test_name, prompt_version, split)
        path = self._cache_path(model, condition)

        def compute():
            prompts = [prompt_builder(item["paragraph"]) for item in items]
            model_id = self._activate_variant(model)
            raw_outputs = self.client.batch_generate(
                model_id,
                prompts,
                max_tokens=self.max_output_tokens,
                classify=True,
                response_labels=raw_allowed_labels,
            )
            thoughts = self.client.last_batch_thoughts
            raw_parsed = parse_answers(raw_outputs, raw_allowed_labels)
            rows = []
            for item, raw, parsed_raw, thought_text in zip(items, raw_outputs, raw_parsed, thoughts):
                parsed_label = parsed_label_map.get(parsed_raw, parsed_raw) if (parsed_label_map and parsed_raw) else parsed_raw
                valid_output = is_exact_valid_label(raw, raw_allowed_labels) if legal_label_strict else parsed_raw is not None
                expected = item.get("expected_label", "")
                gold = item.get("gold_label", "")
                notes = item.get("notes", "")
                if parsed_label_map and parsed_raw:
                    notes = f"raw_parsed_label={parsed_raw}" + (f"; {notes}" if notes else "")
                rows.append(
                    {
                        "run_id": run_id,
                        "timestamp": timestamp,
                        "model_name": model,
                        "model_version": model_id,
                        "temperature": self.temperature,
                        "test_name": test_name,
                        "prompt_version": prompt_version,
                        "split": item.get("split", split),
                        "item_id": item["item_id"],
                        "title": item.get("title", ""),
                        "paragraph": item["paragraph"],
                        "gold_label": gold,
                        "expected_label": expected,
                        "model_output_raw": raw,
                        "_thoughts": thought_text,
                        "parsed_label": parsed_label or "",
                        "valid_output": bool(valid_output),
                        "correct_against_expected": (parsed_label == expected) if expected else "",
                        "correct_against_gold": (parsed_label == gold) if gold else "",
                        "matches_baseline": "",
                        "notes": notes,
                    }
                )
            return rows

        records = self._get_or_compute(path, compute)
        for row in records:
            row["run_id"] = run_id
            row["timestamp"] = timestamp
            row["model_name"] = model
            row["model_version"] = self._variant(model)["model_id"]
            row["temperature"] = self.temperature
            row["test_name"] = test_name
            row["prompt_version"] = prompt_version
            row.setdefault("_thoughts", "")
        return records

    def _clone_records_for_context(
        self,
        records: List[dict],
        run_id: str,
        timestamp: str,
        test_name: str,
        prompt_version: str,
        legal_label_strict: bool = False,
        matches_baseline="",
        notes_prefix: str = "",
    ) -> List[dict]:
        cloned = []
        for row in records:
            new_row = copy.deepcopy(row)
            parsed_label = new_row.get("parsed_label", "")
            gold = new_row.get("gold_label", "")
            expected = new_row.get("expected_label", "")
            raw = new_row.get("model_output_raw", "")
            new_row["run_id"] = run_id
            new_row["timestamp"] = timestamp
            new_row["test_name"] = test_name
            new_row["prompt_version"] = prompt_version
            new_row["valid_output"] = (
                is_exact_valid_label(raw, CANONICAL_LABELS) if legal_label_strict else parsed_label in CANONICAL_LABELS
            )
            new_row["correct_against_expected"] = (parsed_label == expected) if expected else ""
            new_row["correct_against_gold"] = (parsed_label == gold) if gold else ""
            new_row["matches_baseline"] = matches_baseline
            if notes_prefix:
                original_notes = new_row.get("notes", "")
                new_row["notes"] = notes_prefix + (f"; {original_notes}" if original_notes else "")
            cloned.append(new_row)
        return cloned

    @staticmethod
    def _add_baseline_matches(records: List[dict], baseline_by_item: Dict[str, str]) -> None:
        for row in records:
            baseline = baseline_by_item.get(row["item_id"], "")
            row["matches_baseline"] = bool(baseline and row.get("parsed_label") == baseline)

    @staticmethod
    def _consistency_across_versions(version_records: Dict[str, List[dict]]) -> float:
        versions = list(version_records)
        if not versions:
            return 0.0
        item_ids = [record["item_id"] for record in version_records[versions[0]]]
        consistent = 0
        for item_id in item_ids:
            labels = []
            for version in versions:
                row = next(record for record in version_records[version] if record["item_id"] == item_id)
                labels.append(row.get("parsed_label", ""))
            if all(label in CANONICAL_LABELS for label in labels) and len(set(labels)) == 1:
                consistent += 1
        return consistent / len(item_ids) if item_ids else 0.0

    def run_diagnostics(self) -> dict[str, Path]:
        run_id = f"pre_zero_shot_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now(timezone.utc).isoformat()

        dev_items = dataframe_to_items(sample_split(self.df, "dev", self.pilot, self.pilot_n, self.seed), "dev", "dev")
        example_items = codebook_example_items()
        order_items = dev_items
        if not dev_items:
            raise ValueError("No dev split found. Diagnostics need rows where split == 'dev'.")

        all_item_rows = []
        all_summary_rows = []
        comparison_rows = []

        print(f"RUN_ID: {run_id}")
        print(f"Dev diagnostic items: {len(dev_items)}")
        print(f"Codebook example items: {len(example_items)}")

        for model in self.models:
            print(f"\n=== Diagnostics: {model} ===")
            full_dev_records = self._run_items_for_prompt(
                run_id,
                timestamp,
                model,
                "Full Codebook Dev",
                "full_codebook",
                dev_items,
                build_full_prompt,
                CANONICAL_LABELS,
            )

            full_dev_context_records = self._clone_records_for_context(
                full_dev_records,
                run_id,
                timestamp,
                "Full Codebook Dev",
                "full_codebook",
                matches_baseline=True,
            )
            all_item_rows.extend(full_dev_context_records)
            full_dev_summary = summary_row(
                run_id,
                model,
                self.temperature,
                "Full Codebook Dev",
                "full_codebook",
                full_dev_context_records,
                "dev",
                target_field="gold_label",
                agreement_with_full=1.0,
            )
            all_summary_rows.append(full_dev_summary)

            legal_records = self._clone_records_for_context(
                full_dev_records,
                run_id,
                timestamp,
                "Legal Label Test",
                "full_codebook",
                legal_label_strict=True,
                notes_prefix="reused_from=Full Codebook Dev/full_codebook",
            )
            all_item_rows.extend(legal_records)
            legal_summary = summary_row(run_id, model, self.temperature, "Legal Label Test", "full_codebook", legal_records, "dev")
            all_summary_rows.append(legal_summary)

            example_records = self._run_items_for_prompt(
                run_id,
                timestamp,
                model,
                "Example Recovery Test",
                "full_codebook",
                example_items,
                build_full_prompt,
                CANONICAL_LABELS,
            )
            all_item_rows.extend(example_records)
            example_summary = summary_row(
                run_id,
                model,
                self.temperature,
                "Example Recovery Test",
                "full_codebook",
                example_records,
                "codebook_examples",
                target_field="expected_label",
            )
            for col in ["micro_f1", "macro_f1", "weighted_f1", "high_f1", "low_f1", "none_f1"]:
                example_summary[col] = ""
            all_summary_rows.append(example_summary)

            order_records_by_version = {
                "original_order": self._clone_records_for_context(
                    full_dev_records,
                    run_id,
                    timestamp,
                    "Codebook Order Invariance Test",
                    "original_order",
                    matches_baseline=True,
                    notes_prefix="reused_from=Full Codebook Dev/full_codebook",
                )
            }
            order_builders: dict[str, PromptBuilder] = {
                "reordered_label_sections": build_reordered_label_prompt,
                "reordered_evidence_categories": lambda document: build_reordered_category_prompt(document, self.seed),
            }
            for version, builder in order_builders.items():
                order_records_by_version[version] = self._run_items_for_prompt(
                    run_id,
                    timestamp,
                    model,
                    "Codebook Order Invariance Test",
                    version,
                    order_items,
                    builder,
                    CANONICAL_LABELS,
                )

            order_baseline = {row["item_id"]: row["parsed_label"] for row in order_records_by_version["original_order"]}
            for version, records in order_records_by_version.items():
                if version != "original_order":
                    self._add_baseline_matches(records, order_baseline)
                all_item_rows.extend(records)
            order_consistency = self._consistency_across_versions(order_records_by_version)
            all_summary_rows.append(
                summary_row(
                    run_id,
                    model,
                    self.temperature,
                    "Codebook Order Invariance Test",
                    "all_order_versions",
                    order_records_by_version["original_order"],
                    "dev",
                    consistency_rate=order_consistency,
                    notes="Consistency across original_order, reordered_label_sections, and reordered_evidence_categories on dev.",
                )
            )

            generic_original_records = self._clone_records_for_context(
                full_dev_records,
                run_id,
                timestamp,
                "Generic Label Test",
                "original_labels_full",
                matches_baseline=True,
                notes_prefix="reused_from=Full Codebook Dev/full_codebook",
            )
            generic_records = self._run_items_for_prompt(
                run_id,
                timestamp,
                model,
                "Generic Label Test",
                "generic_abc",
                dev_items,
                build_generic_prompt,
                GENERIC_LABELS,
                parsed_label_map=GENERIC_TO_CANONICAL,
            )
            generic_baseline = {row["item_id"]: row["parsed_label"] for row in generic_original_records}
            self._add_baseline_matches(generic_records, generic_baseline)
            generic_consistency = (
                sum(bool(row["matches_baseline"]) for row in generic_records) / len(generic_records)
                if generic_records
                else 0.0
            )
            all_item_rows.extend(generic_original_records)
            all_item_rows.extend(generic_records)
            all_summary_rows.append(
                summary_row(
                    run_id,
                    model,
                    self.temperature,
                    "Generic Label Test",
                    "generic_abc",
                    generic_records,
                    "dev",
                    target_field="gold_label",
                    consistency_rate=generic_consistency,
                    notes="A/B/C outputs mapped back to HIGH/LOW/NONE before scoring.",
                )
            )

            comparison_rows.append(
                {
                    "model_name": model,
                    "thinking_level": self._variant(model)["thinking_level"],
                    "legal_valid_rate": legal_summary.get("valid_output_rate", ""),
                    "example_recovery_accuracy": example_summary.get("accuracy", ""),
                    "order_consistency_rate": order_consistency,
                    "generic_label_consistency_rate": generic_consistency,
                    "full_codebook_dev_accuracy": full_dev_summary.get("accuracy", ""),
                    "full_codebook_dev_micro_f1": full_dev_summary.get("micro_f1", ""),
                    "full_codebook_dev_macro_f1": full_dev_summary.get("macro_f1", ""),
                    "full_codebook_dev_weighted_f1": full_dev_summary.get("weighted_f1", ""),
                    "recommendation_notes": "Gemini-only local diagnostic pass.",
                }
            )

        item_df = pd.DataFrame(all_item_rows, columns=ITEM_LEVEL_COLUMNS).fillna("")
        summary_df = pd.DataFrame(all_summary_rows, columns=SUMMARY_METRIC_COLUMNS).fillna("")
        comparison_df = pd.DataFrame(comparison_rows, columns=MODEL_COMPARISON_COLUMNS).fillna("")

        outputs = {
            "diagnostic_item_level": self.paths["diagnostics"] / "all_item_level_predictions.csv",
            "diagnostic_summary": self.paths["diagnostics"] / "all_summary_metrics.csv",
            "diagnostic_comparison": self.paths["diagnostics"] / "model_comparison_master.csv",
        }
        item_df.to_csv(outputs["diagnostic_item_level"], index=False)
        summary_df.to_csv(outputs["diagnostic_summary"], index=False)
        comparison_df.to_csv(outputs["diagnostic_comparison"], index=False)
        return outputs

    def run_final_zero_shot(self) -> dict[str, Path]:
        started_at = time.monotonic()
        test_df = self.df[self.df["split"].astype(str).str.lower() == "test"].reset_index(drop=True)
        if self.pilot:
            test_df = test_df.head(self.pilot_n).reset_index(drop=True)
        test_items = dataframe_to_items(test_df, "test", "test")
        if not test_items:
            raise ValueError("No test split found. Final zero-shot needs rows where split == 'test'.")

        resume_state_path = self.paths["final"] / "resume_state.json"
        failures_path = self.paths["final"] / "failures.json"
        run_config_path = self.paths["final"] / "run_config.json"
        if self.resume:
            resume_state = read_json_file(resume_state_path, None)
            if not resume_state:
                raise ValueError(f"No resume state found at {resume_state_path}")
            if resume_state.get("models") != self.models:
                raise ValueError("Resume state model list does not match the current --models arguments.")
            run_id = resume_state["run_id"]
            timestamp = resume_state["timestamp"]
            usage_total = copy_usage(resume_state.get("api_usage_total", empty_usage()))
            resume_model_index = int(resume_state.get("model_index", 0))
            resume_next_item_index = int(resume_state.get("next_item_index", 0))
            prior_runtime_seconds = float(resume_state.get("runtime_seconds", 0.0) or 0.0)
            resume_requires_pilot = resume_state.get("status") in {"initialized", "pilot_pending", "pilot_running"}
            failures = read_json_file(failures_path, [])
            run_config = read_json_file(run_config_path, {})
            print(
                f"Resuming run {run_id} at model_index={resume_model_index}, "
                f"item_index={resume_next_item_index}.",
                flush=True,
            )
        else:
            run_id = f"final_zero_shot_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
            timestamp = datetime.now(timezone.utc).isoformat()
            usage_total = empty_usage()
            resume_model_index = 0
            resume_next_item_index = 0
            prior_runtime_seconds = 0.0
            resume_requires_pilot = False
            failures = []
            write_run_config(failures_path, failures)
            run_config = {}

        all_final_rows = []
        prediction_cache_paths: dict[str, Path] = {}
        run_config.update(
            {
                "run_id": run_id,
                "timestamp": timestamp,
                "data_rows": int(len(self.df)),
                "test_rows": int(len(test_items)),
                "models": self.models,
                "model_ids": self.model_ids,
                "thinking_levels": self.thinking_levels,
                "model_variants": list(self._variants_by_name.values()),
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
                "request_timeout_seconds": self.client.config.request_timeout_seconds,
                "thinking_level": self.thinking_levels[0] if len(self.thinking_levels) == 1 else None,
                "pilot_mode": bool(self.pilot),
                "pilot_n": self.pilot_n,
                "status_every": self.status_every,
                "api_usage_total": copy_usage(usage_total),
                "cost_by_model": run_config.get("cost_by_model", {}),
                "failures_json": str(failures_path),
                "resume_state": str(resume_state_path),
                "resume_command": self.resume_command,
                "status": "resuming" if self.resume else "initialized",
            }
        )
        write_run_config(run_config_path, run_config)
        if not self.resume:
            self._write_resume_state(
                resume_state_path,
                run_id,
                timestamp,
                0,
                0,
                "pilot_pending",
                usage_total,
                0.0,
            )

        print(f"FINAL_TEST_RUN_ID: {run_id}")
        print(f"Test items: {len(test_items)}")
        if self.pilot:
            print(f"PILOT MODE: final zero-shot uses the first {len(test_items)} test rows")

        total = len(test_items)
        for model_index, model in enumerate(self.models):
            print(f"\n=== Final zero-shot: {model} ===")
            condition = "final_zero_shot_full_codebook_test"
            path = self._cache_path(model, condition, final=True)
            prediction_cache_paths[model] = path
            cost_rates = self._cost_rates(model)
            run_config["cost_by_model"][model] = cost_rates

            cached = read_jsonl(path) or []
            cache_complete = len(cached) == total
            if self.resume and model_index < resume_model_index:
                if not cache_complete:
                    raise ValueError(f"Completed-model cache is incomplete: {path}")
                records = cached
                start_index = total
                cache_hit = True
            elif self.resume and model_index == resume_model_index and not resume_requires_pilot:
                if len(cached) < resume_next_item_index:
                    raise ValueError(
                        f"Resume cache has {len(cached)} rows but state expects "
                        f"{resume_next_item_index}: {path}"
                    )
                records = cached[:resume_next_item_index]
                start_index = resume_next_item_index
                cache_hit = False
                print(f"  [resume] {path.name}: continuing at row {start_index + 1}/{total}")
            elif cache_complete:
                records = cached
                start_index = total
                cache_hit = True
                print(f"  [cache hit] {path.name}: {len(cached)} records")
            else:
                cache_hit = False
                if not self.pilot:
                    pilot_items = test_items[: self.pilot_n]
                    print(f"Starting pilot check for {model} on {len(pilot_items)} rows...", flush=True)
                    pilot_records = []
                    checkpoint_usage = copy_usage(usage_total)
                    for idx, item in enumerate(pilot_items, start=1):
                        record, usage, request_metadata = self._run_final_item(run_id, timestamp, model, item)
                        usage = usage_with_cost(usage, cost_rates)
                        add_usage(usage_total, usage)
                        pilot_records.append(record)
                        if request_metadata.get("failed"):
                            self._record_api_failure(
                                failures,
                                failures_path,
                                "pilot",
                                model,
                                item,
                                idx - 1,
                                record["model_output_raw"],
                                request_metadata,
                            )

                        if self.status_every > 0 and (idx % self.status_every == 0 or idx == len(pilot_items)):
                            print_progress(idx, len(pilot_items), pilot_records)
                            interval_usage = subtract_usage(usage_total, checkpoint_usage)
                            print_usage_progress("Pilot", idx, len(pilot_items), interval_usage, usage_total, started_at)
                            checkpoint_usage = copy_usage(usage_total)

                    pilot_stats = counts_for(pilot_records)
                    run_config.update(
                        {
                            "api_usage_total": copy_usage(usage_total),
                            "pilot_summary": pilot_stats,
                            "runtime_seconds": round(time.monotonic() - started_at, 3),
                        }
                    )
                    pilot_path = self._save_pilot_outputs(pilot_records, run_config, status="pilot_completed")
                    print(f"Saved pilot item-level predictions to: {pilot_path}", flush=True)
                    passed = pilot_stats["invalid_count"] == 0 and pilot_stats["error_count"] == 0
                    print(
                        "Pilot summary: "
                        f"rows checked={pilot_stats['rows']} | "
                        f"valid={pilot_stats['valid_count']} | "
                        f"invalid={pilot_stats['invalid_count']} | "
                        f"errors={pilot_stats['error_count']} | "
                        f"invalid/error rate={pilot_stats['invalid_error_rate']:.3f} | "
                        f"status={'PASSED' if passed else 'FAILED'}",
                        flush=True,
                    )
                    if not passed:
                        run_config.update(
                            {
                                "api_usage_total": copy_usage(usage_total),
                                "runtime_seconds": round(time.monotonic() - started_at, 3),
                                "number_of_rows_processed": int(len(all_final_rows)),
                            }
                        )
                        self._save_pilot_outputs(pilot_records, run_config, status="pilot_failed")
                        raise RuntimeError("Pilot check failed; full final dataset was not started.")

                records = []
                start_index = 0

            if start_index < total:
                checkpoint_usage = copy_usage(usage_total)
                runtime_seconds = prior_runtime_seconds + (time.monotonic() - started_at)
                self._write_resume_state(
                    resume_state_path,
                    run_id,
                    timestamp,
                    model_index,
                    start_index,
                    "running",
                    usage_total,
                    runtime_seconds,
                )
                if start_index:
                    print(f"Resuming full run for {model} at row {start_index + 1}/{total}...", flush=True)
                else:
                    print(f"Starting full run for {model} on {total} rows...", flush=True)
                consecutive_api_failures = 0
                for item_index in range(start_index, total):
                    item = test_items[item_index]
                    record, usage, request_metadata = self._run_final_item(run_id, timestamp, model, item)
                    usage = usage_with_cost(usage, cost_rates)
                    add_usage(usage_total, usage)
                    records.append(record)

                    request_failed = bool(request_metadata.get("failed")) or is_error(record["model_output_raw"])
                    if request_failed:
                        consecutive_api_failures += 1
                        self._record_api_failure(
                            failures,
                            failures_path,
                            "full",
                            model,
                            item,
                            item_index,
                            record["model_output_raw"],
                            request_metadata,
                        )
                    else:
                        consecutive_api_failures = 0

                    write_jsonl(path, records)
                    runtime_seconds = prior_runtime_seconds + (time.monotonic() - started_at)
                    self._write_resume_state(
                        resume_state_path,
                        run_id,
                        timestamp,
                        model_index,
                        item_index + 1,
                        "running",
                        usage_total,
                        runtime_seconds,
                    )

                    if consecutive_api_failures >= 3:
                        retry_from_index = item_index - consecutive_api_failures + 1
                        records = records[:-consecutive_api_failures]
                        write_jsonl(path, records)
                        self._write_resume_state(
                            resume_state_path,
                            run_id,
                            timestamp,
                            model_index,
                            retry_from_index,
                            "stopped_after_consecutive_failures",
                            usage_total,
                            runtime_seconds,
                        )
                        run_config.update(
                            {
                                "api_usage_total": copy_usage(usage_total),
                                "runtime_seconds": round(runtime_seconds, 3),
                                "failure_count": len(failures),
                                "resume_from_model_index": model_index,
                                "resume_from_item_index": retry_from_index,
                            }
                        )
                        outputs = self._save_final_outputs(
                            all_final_rows + records,
                            run_config,
                            status="stopped_after_consecutive_failures",
                        )
                        print(
                            "Stopped cleanly after 3 consecutive API requests failed "
                            "after retry handling.",
                            flush=True,
                        )
                        self._print_resume_command()
                        return outputs

                    processed = item_index + 1
                    if self.status_every > 0 and (processed % self.status_every == 0 or processed == total):
                        print_progress(processed, total, records)
                        interval_usage = subtract_usage(usage_total, checkpoint_usage)
                        print_usage_progress("Full run", processed, total, interval_usage, usage_total, started_at)
                        checkpoint_usage = copy_usage(usage_total)
                        run_config.update(
                            {
                                "api_usage_total": copy_usage(usage_total),
                                "runtime_seconds": round(runtime_seconds, 3),
                                "failure_count": len(failures),
                            }
                        )
                        checkpoint_records = all_final_rows + records
                        self._save_final_outputs(
                            checkpoint_records,
                            run_config,
                            status="running",
                        )

            for row in records:
                row["run_id"] = run_id
                row["timestamp"] = timestamp
                row["model_name"] = model
                row["model_version"] = self._variant(model)["model_id"]
                row["temperature"] = self.temperature
                row["test_name"] = "Final Zero-Shot Test"
                row["prompt_version"] = "full_codebook"
                row["split"] = "test"
                row.setdefault("_thoughts", "")

            if not cache_hit:
                write_jsonl(path, records)
                print(f"  Saved {len(records)} records to {path}")

            all_final_rows.extend(records)
            runtime_seconds = prior_runtime_seconds + (time.monotonic() - started_at)
            self._write_resume_state(
                resume_state_path,
                run_id,
                timestamp,
                model_index + 1,
                0,
                "running",
                usage_total,
                runtime_seconds,
            )

        runtime_seconds = prior_runtime_seconds + (time.monotonic() - started_at)
        run_config.update(
            {
                "api_usage_total": copy_usage(usage_total),
                "runtime_seconds": round(runtime_seconds, 3),
                "failure_count": len(failures),
            }
        )
        outputs = self._save_final_outputs(
            all_final_rows,
            run_config,
            status="completed",
            prediction_cache_paths=prediction_cache_paths,
        )
        self._write_resume_state(
            resume_state_path,
            run_id,
            timestamp,
            len(self.models),
            0,
            "completed",
            usage_total,
            runtime_seconds,
        )
        print(
            "Run finished: "
            f"status=completed | "
            f"total_cost={format_cost(usage_total)} | "
            f"input_tokens={usage_total['input_tokens']} | "
            f"output_tokens={usage_total['output_tokens']} | "
            f"thinking_tokens={usage_total['thinking_tokens']} | "
            f"runtime={elapsed_time(time.monotonic() - started_at)}",
            flush=True,
        )
        return outputs
