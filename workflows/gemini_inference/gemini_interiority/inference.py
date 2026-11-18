"""Reliable inference-only runner for sampled novel passages.

This module deliberately depends on the existing prompt and Gemini client so the
classification task, structured label output, API retry policy, and model
settings do not drift from the annotated evaluation pipeline.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import pandas as pd

from .metrics import is_exact_valid_label
from .prompts import CANONICAL_LABELS, build_full_prompt

if TYPE_CHECKING:
    from .gemini_client import GeminiRunnerClient

PROMPT_VERSION = "full_codebook"
INPUT_COLUMNS = [
    "gutenberg_id",
    "title",
    "passage_id",
    "text",
    "n_words",
    "position_bin",
    "normalized_position",
    "is_base_sample",
    "is_core_extra",
]
PASSAGE_OUTPUT_COLUMNS = [
    "gutenberg_id",
    "title",
    "passage_id",
    "position_bin",
    "normalized_position",
    "is_base_sample",
    "is_core_extra",
    "model",
    "prompt_version",
    "predicted_label",
    "raw_response",
    "parse_status",
    "run_status",
    "attempt_count",
    "error_message",
]
BOOK_SUMMARY_COLUMNS = [
    "gutenberg_id",
    "title",
    "is_core",
    "scope",
    "n_requested",
    "n_success",
    "n_failed",
    "none_count",
    "low_count",
    "high_count",
    "none_pct",
    "low_pct",
    "high_pct",
    "avg_interiority_score",
]
SCORE_BY_LABEL = {"NONE": 0, "LOW": 1, "HIGH": 2}
MODEL_PRICES = {
    "gemini-3.5-flash": {
        "input_cost_per_1m_tokens": 1.50,
        "output_cost_per_1m_tokens": 9.00,
    }
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_runtime(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours, remainder = divmod(int(seconds), 3600)
    minutes, whole_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {whole_seconds}s"
    if minutes:
        return f"{minutes}m {whole_seconds}s"
    return f"{seconds:.1f}s"


def usage_totals(events: Iterable[dict[str, Any]]) -> dict[str, int]:
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "thinking_tokens": 0,
        "total_tokens": 0,
    }
    for event in events:
        for field in totals:
            totals[field] += int(event.get(f"_{field}", 0) or 0)
    return totals


def estimated_cost(model: str, usage: dict[str, int]) -> float | None:
    rates = MODEL_PRICES.get(str(model).lower())
    if rates is None:
        return None
    return (
        usage["input_tokens"] / 1_000_000 * rates["input_cost_per_1m_tokens"]
        + usage["output_tokens"] / 1_000_000 * rates["output_cost_per_1m_tokens"]
    )


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_boolean(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not pd.isna(value):
        if value in (0, 1):
            return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "t", "1", "yes"}:
        return True
    if normalized in {"false", "f", "0", "no"}:
        return False
    raise ValueError(f"{field} contains a non-boolean value: {value!r}")


def compact_input(source_path: Path, destination_path: Path) -> pd.DataFrame:
    """Read the full sampling CSV and write only the approved inference fields."""
    source = pd.read_csv(source_path)
    missing = sorted(set(INPUT_COLUMNS) - set(source.columns))
    if missing:
        raise ValueError(f"{source_path} is missing required columns: {', '.join(missing)}")

    compact = source[INPUT_COLUMNS].copy()
    if compact["text"].isna().any() or compact["text"].astype(str).str.strip().eq("").any():
        raise ValueError("Every sampled passage must contain non-empty text.")
    for field in ("is_base_sample", "is_core_extra"):
        compact[field] = compact[field].map(lambda value: normalize_boolean(value, field))
    if compact.duplicated(["gutenberg_id", "passage_id"]).any():
        count = int(compact.duplicated(["gutenberg_id", "passage_id"]).sum())
        raise ValueError(f"Input has {count} duplicate (gutenberg_id, passage_id) keys.")
    atomic_write_csv(destination_path, compact)
    return compact


def parse_label(raw_response: str) -> tuple[str, str, str]:
    """Return predicted label, parse status, and run status.

    The production Gemini call uses an enum schema, so accepting anything beyond
    an exact task label would hide malformed or corrupted generations.
    """
    raw = str(raw_response or "")
    if raw.startswith("ERROR:"):
        return "", "api_error", "failed"
    if not raw.strip():
        return "", "invalid_empty", "invalid"
    if is_exact_valid_label(raw, CANONICAL_LABELS):
        return raw.strip().upper(), "valid", "success"
    return "", "invalid_label", "invalid"


def prediction_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["gutenberg_id"]),
        str(row["passage_id"]),
        str(row["model"]),
        str(row["prompt_version"]),
    )


def append_checkpoint(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_checkpoint(path: Path, repair_truncated_tail: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Load checkpoint events and recover a final partially written JSON line."""
    if not path.exists():
        raise ValueError(f"Checkpoint does not exist: {path}")
    raw_lines = path.read_bytes().splitlines(keepends=True)
    decoded: list[dict[str, Any]] = []
    valid_raw_lines: list[bytes] = []
    recovered_tail = ""
    for index, raw_line in enumerate(raw_lines):
        if not raw_line.strip():
            continue
        try:
            decoded.append(json.loads(raw_line))
            valid_raw_lines.append(raw_line if raw_line.endswith(b"\n") else raw_line + b"\n")
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            is_last_nonempty = not any(line.strip() for line in raw_lines[index + 1 :])
            if not (repair_truncated_tail and is_last_nonempty):
                raise ValueError(f"Corrupted checkpoint line {index + 1} in {path}: {exc}") from exc
            recovered_tail = raw_line.decode("utf-8", errors="replace")
            recovery_path = path.with_suffix(path.suffix + f".corrupt_tail_{int(time.time())}")
            recovery_path.write_bytes(raw_line)
            temporary = path.with_suffix(path.suffix + ".repair.tmp")
            temporary.write_bytes(b"".join(valid_raw_lines))
            temporary.replace(path)
            break

    if not decoded or decoded[0].get("record_type") != "configuration":
        raise ValueError(f"Checkpoint has no configuration header: {path}")
    return decoded[0], [row for row in decoded[1:] if row.get("record_type") == "prediction"], recovered_tail


def latest_prediction_records(events: Iterable[dict[str, Any]]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    latest: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for event in events:
        latest[prediction_key(event)] = event
    return latest


def prompt_digest() -> str:
    system_message, user_message = build_full_prompt("__PASSAGE_TEXT__")
    payload = json.dumps(
        {"prompt_version": PROMPT_VERSION, "system": system_message, "user": user_message},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def configuration_fingerprint(configuration: dict[str, Any]) -> str:
    stable = {
        key: configuration[key]
        for key in (
            "model",
            "prompt_version",
            "prompt_sha256",
            "input_sha256",
            "backend",
            "thinking_level",
            "max_output_tokens",
            "valid_labels",
        )
    }
    serialized = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def book_summaries(input_frame: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize base and expanded scopes using successful parses only."""
    success = predictions[predictions["run_status"].eq("success")].copy() if len(predictions) else predictions.copy()
    success_lookup = {
        (str(row.gutenberg_id), str(row.passage_id)): row.predicted_label
        for row in success.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    books = input_frame[["gutenberg_id", "title"]].drop_duplicates().sort_values(["gutenberg_id", "title"])
    for book in books.itertuples(index=False):
        book_rows = input_frame[
            input_frame["gutenberg_id"].astype(str).eq(str(book.gutenberg_id))
        ]
        is_core = bool(book_rows["is_core_extra"].any())
        scopes = ("base", "expanded") if is_core else ("base",)
        for scope in scopes:
            requested = book_rows[book_rows["is_base_sample"]] if scope == "base" else book_rows
            labels = [
                success_lookup[(str(row.gutenberg_id), str(row.passage_id))]
                for row in requested.itertuples(index=False)
                if (str(row.gutenberg_id), str(row.passage_id)) in success_lookup
            ]
            n_success = len(labels)
            requested_keys = {
                (str(row.gutenberg_id), str(row.passage_id))
                for row in requested.itertuples(index=False)
            }
            attempted_failures = sum(
                (str(row.gutenberg_id), str(row.passage_id)) in requested_keys
                and row.run_status in {"failed", "invalid"}
                for row in predictions.itertuples(index=False)
            )
            counts = {label: labels.count(label) for label in CANONICAL_LABELS}
            denominator = n_success or None
            rows.append(
                {
                    "gutenberg_id": book.gutenberg_id,
                    "title": book.title,
                    "is_core": is_core,
                    "scope": scope,
                    "n_requested": len(requested),
                    "n_success": n_success,
                    "n_failed": attempted_failures,
                    "none_count": counts["NONE"],
                    "low_count": counts["LOW"],
                    "high_count": counts["HIGH"],
                    "none_pct": counts["NONE"] / denominator if denominator else math.nan,
                    "low_pct": counts["LOW"] / denominator if denominator else math.nan,
                    "high_pct": counts["HIGH"] / denominator if denominator else math.nan,
                    "avg_interiority_score": (
                        sum(SCORE_BY_LABEL[label] for label in labels) / denominator
                        if denominator
                        else math.nan
                    ),
                }
            )
    return pd.DataFrame(rows, columns=BOOK_SUMMARY_COLUMNS)


def validation_report(
    input_frame: pd.DataFrame,
    predictions: pd.DataFrame,
    summaries: pd.DataFrame,
    checkpoint_tail_recovered: bool = False,
) -> str:
    key_columns = ["gutenberg_id", "passage_id", "model", "prompt_version"]
    duplicate_predictions = int(predictions.duplicated(key_columns).sum()) if len(predictions) else 0
    successful = predictions[predictions["run_status"].eq("success")] if len(predictions) else predictions
    invalid_success_labels = int((~successful["predicted_label"].isin(CANONICAL_LABELS)).sum()) if len(successful) else 0
    label_counts = {
        label: int((successful["predicted_label"] == label).sum()) if len(successful) else 0
        for label in CANONICAL_LABELS
    }
    failures = int(predictions["run_status"].eq("failed").sum()) if len(predictions) else 0
    invalid = int(predictions["run_status"].eq("invalid").sum()) if len(predictions) else 0
    base_rows = int(input_frame["is_base_sample"].sum())
    core_book_ids = set(
        input_frame.loc[input_frame["is_core_extra"], "gutenberg_id"].astype(str)
    )
    expanded_rows = int(input_frame["gutenberg_id"].astype(str).isin(core_book_ids).sum())
    base_requested = int(summaries.loc[summaries["scope"].eq("base"), "n_requested"].sum()) if len(summaries) else 0
    expanded_requested = (
        int(summaries.loc[summaries["scope"].eq("expanded"), "n_requested"].sum()) if len(summaries) else 0
    )
    return "\n".join(
        [
            "# Inference validation report",
            "",
            f"- Input rows: {len(input_frame):,}",
            f"- Input books: {input_frame['gutenberg_id'].nunique():,}",
            f"- Duplicate input `(gutenberg_id, passage_id)` keys: {int(input_frame.duplicated(['gutenberg_id', 'passage_id']).sum()):,}",
            f"- Passage prediction rows: {len(predictions):,}",
            f"- Duplicate prediction keys: {duplicate_predictions:,}",
            f"- Successful predictions: {len(successful):,}",
            f"- Valid-label counts: NONE={label_counts['NONE']:,}, LOW={label_counts['LOW']:,}, HIGH={label_counts['HIGH']:,}",
            f"- Invalid labels among successful rows: {invalid_success_labels:,}",
            f"- API failures: {failures:,}",
            f"- Invalid generations: {invalid:,}",
            f"- Base sample size: {base_rows:,} input / {base_requested:,} summarized",
            f"- Expanded sample size: {expanded_rows:,} input / {expanded_requested:,} summarized",
            f"- Recovered truncated checkpoint tail: {'yes' if checkpoint_tail_recovered else 'no'}",
            "",
        ]
    )


class InferenceRunner:
    """Run one model/prompt configuration over the sampled passages."""

    def __init__(
        self,
        input_frame: pd.DataFrame,
        client: GeminiRunnerClient | None,
        output_dir: Path,
        model: str,
        backend: str,
        thinking_level: str,
        max_output_tokens: int = 4096,
        max_invalid_retries: int = 5,
        pilot_n: int = 5,
        status_every: int = 100,
        resume: bool = False,
        pilot_only: bool = False,
        prepare_only: bool = False,
        input_sha256: str = "",
        resume_command: str = "",
    ):
        self.input_frame = input_frame.reset_index(drop=True)
        self.client = client
        self.output_dir = output_dir
        self.model = model
        self.backend = backend
        self.thinking_level = thinking_level
        self.max_output_tokens = max_output_tokens
        self.max_invalid_retries = max_invalid_retries
        self.pilot_n = min(max(1, pilot_n), len(self.input_frame)) if len(self.input_frame) else 0
        self.status_every = status_every
        self.resume = resume
        self.pilot_only = pilot_only
        self.prepare_only = prepare_only
        self.input_sha256 = input_sha256
        self.resume_command = resume_command
        self._started_at = time.monotonic()
        self._prior_runtime_seconds = 0.0
        self.paths = {
            "checkpoint": output_dir / "checkpoint.jsonl",
            "predictions": output_dir / "passage_predictions.csv",
            "summary": output_dir / "book_summary.csv",
            "config": output_dir / "run_config.json",
            "validation": output_dir / "validation_report.md",
        }

    def _configuration(self) -> dict[str, Any]:
        config = {
            "run_id": str(uuid.uuid4()),
            "created_at": utc_now(),
            "model": self.model,
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": prompt_digest(),
            "input_sha256": self.input_sha256,
            "backend": self.backend,
            "thinking_level": self.thinking_level,
            "max_output_tokens": self.max_output_tokens,
            "max_api_retries": int(self.client.config.max_retries) if self.client else None,
            "max_invalid_retries": self.max_invalid_retries,
            "pilot_n": self.pilot_n,
            "valid_labels": list(CANONICAL_LABELS),
            "status_every": self.status_every,
            "cost_rates": MODEL_PRICES.get(self.model.lower()),
            "resume_command": self.resume_command,
            "status": "initialized",
            "outputs": {name: str(path) for name, path in self.paths.items()},
        }
        config["configuration_fingerprint"] = configuration_fingerprint(config)
        return config

    def _initialize_or_resume(self) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
        checkpoint_path = self.paths["checkpoint"]
        if self.resume:
            header, events, recovered_tail = load_checkpoint(checkpoint_path)
            current = self._configuration()
            expected = header.get("configuration_fingerprint")
            if expected != current["configuration_fingerprint"]:
                raise ValueError(
                    "Checkpoint configuration mismatch. Refusing to resume with a "
                    "different model, prompt, input, backend, thinking level, or output-token setting."
                )
            config = json.loads(self.paths["config"].read_text(encoding="utf-8"))
            config["status"] = "resuming"
            config["updated_at"] = utc_now()
            config["max_api_retries"] = (
                int(self.client.config.max_retries) if self.client else config.get("max_api_retries")
            )
            config["max_invalid_retries"] = self.max_invalid_retries
            config["status_every"] = self.status_every
            config["cost_rates"] = MODEL_PRICES.get(self.model.lower())
            config["checkpoint_truncated_tail_recovered"] = bool(recovered_tail)
            self._prior_runtime_seconds = float(config.get("runtime_seconds", 0.0) or 0.0)
            atomic_write_json(self.paths["config"], config)
            return config, events, bool(recovered_tail)

        if checkpoint_path.exists() or self.paths["config"].exists():
            raise ValueError(
                f"{self.output_dir} already contains run state. Use --resume with the "
                "same configuration or choose a new --output-dir."
            )
        config = self._configuration()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.paths["config"], config)
        append_checkpoint(
            checkpoint_path,
            {
                "record_type": "configuration",
                "configuration_fingerprint": config["configuration_fingerprint"],
                "run_id": config["run_id"],
                "created_at": config["created_at"],
            },
        )
        return config, [], False

    def _request(self, row: Any) -> dict[str, Any]:
        if self.client is None:
            raise RuntimeError("A Gemini client is required unless --prepare-only is used.")
        prompt = build_full_prompt(str(row.text))
        total_attempts = 0
        request_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "total_tokens": 0,
        }
        raw_response = ""
        error_message = ""
        thoughts = ""
        # API exceptions are retried inside GeminiRunnerClient. Only malformed
        # non-error generations are repeated here.
        for invalid_retry in range(self.max_invalid_retries + 1):
            raw_response = self.client.batch_generate(
                self.model,
                [prompt],
                max_tokens=self.max_output_tokens,
                show_progress=False,
                retry_on_error=True,
                classify=True,
                response_labels=CANONICAL_LABELS,
            )[0]
            metadata = (
                self.client.last_batch_request_metadata[0]
                if self.client.last_batch_request_metadata
                else {"attempts": 1, "error_message": ""}
            )
            last_usage = getattr(self.client, "last_usage", {}) or {}
            for field in request_usage:
                request_usage[field] += int(last_usage.get(field, 0) or 0)
            total_attempts += int(metadata.get("attempts", 1) or 1)
            error_message = str(metadata.get("error_message") or "")
            thoughts = str(getattr(self.client, "last_thoughts", "") or "")
            predicted_label, parse_status, run_status = parse_label(raw_response)
            if run_status != "invalid" or invalid_retry >= self.max_invalid_retries:
                break

        if run_status == "invalid" and not error_message:
            error_message = f"Generation did not contain exactly one valid label after {total_attempts} attempt(s)."
        if run_status == "failed" and not error_message:
            error_message = raw_response
        return {
            "record_type": "prediction",
            "run_id": "",
            "timestamp": utc_now(),
            "gutenberg_id": row.gutenberg_id,
            "title": row.title,
            "passage_id": row.passage_id,
            "position_bin": row.position_bin,
            "normalized_position": row.normalized_position,
            "is_base_sample": bool(row.is_base_sample),
            "is_core_extra": bool(row.is_core_extra),
            "model": self.model,
            "prompt_version": PROMPT_VERSION,
            "predicted_label": predicted_label,
            "raw_response": raw_response,
            "parse_status": parse_status,
            "run_status": run_status,
            "attempt_count": total_attempts,
            "error_message": error_message,
            "_thoughts": thoughts,
            "_input_tokens": request_usage["input_tokens"],
            "_output_tokens": request_usage["output_tokens"],
            "_thinking_tokens": request_usage["thinking_tokens"],
            "_total_tokens": request_usage["total_tokens"],
        }

    def _prediction_frame(self, latest: dict[tuple[str, str, str, str], dict[str, Any]]) -> pd.DataFrame:
        order = {
            (str(row.gutenberg_id), str(row.passage_id)): index
            for index, row in enumerate(self.input_frame.itertuples(index=False))
        }
        records = list(latest.values())
        records.sort(
            key=lambda record: order.get(
                (str(record["gutenberg_id"]), str(record["passage_id"])),
                len(order),
            )
        )
        return pd.DataFrame(records, columns=PASSAGE_OUTPUT_COLUMNS).fillna("")

    def _save_outputs(
        self,
        latest: dict[tuple[str, str, str, str], dict[str, Any]],
        recovered_tail: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        predictions = self._prediction_frame(latest)
        if len(predictions) and predictions.duplicated(
            ["gutenberg_id", "passage_id", "model", "prompt_version"]
        ).any():
            raise RuntimeError("Internal error: duplicate passage prediction keys.")
        summaries = book_summaries(self.input_frame, predictions)
        atomic_write_csv(self.paths["predictions"], predictions)
        atomic_write_csv(self.paths["summary"], summaries)
        self.paths["validation"].write_text(
            validation_report(self.input_frame, predictions, summaries, recovered_tail),
            encoding="utf-8",
        )
        return predictions, summaries

    def _runtime_seconds(self) -> float:
        return self._prior_runtime_seconds + (time.monotonic() - self._started_at)

    def _counts(
        self,
        latest: dict[tuple[str, str, str, str], dict[str, Any]],
    ) -> dict[str, int | float]:
        statuses = [row.get("run_status") for row in latest.values()]
        completed = len(latest)
        return {
            "completed": completed,
            "completed_pct": (completed / len(self.input_frame) * 100) if len(self.input_frame) else 100.0,
            "successful": statuses.count("success"),
            "failed": statuses.count("failed"),
            "invalid": statuses.count("invalid"),
            "remaining": max(0, len(self.input_frame) - completed),
        }

    def _progress(
        self,
        latest: dict[tuple[str, str, str, str], dict[str, Any]],
        events: list[dict[str, Any]],
    ) -> dict[str, int | float]:
        counts = self._counts(latest)
        usage = usage_totals(events)
        cost = estimated_cost(self.model, usage)
        cost_text = f"${cost:.6f}" if cost is not None else "unavailable"
        print(
            f"Progress: passages_run={counts['completed']:,}/{len(self.input_frame):,} "
            f"({counts['completed_pct']:.2f}%) | "
            f"successful={counts['successful']:,} | "
            f"failed={counts['failed']:,} | "
            f"invalid={counts['invalid']:,} | "
            f"remaining={counts['remaining']:,} | "
            f"runtime={format_runtime(self._runtime_seconds())} | "
            f"estimated_cost={cost_text}",
            flush=True,
        )
        return counts

    def _refresh_run_config(
        self,
        config: dict[str, Any],
        latest: dict[tuple[str, str, str, str], dict[str, Any]],
        events: list[dict[str, Any]],
        status: str,
    ) -> None:
        usage = usage_totals(events)
        cost = estimated_cost(self.model, usage)
        config.update(
            {
                "status": status,
                "counts": self._counts(latest),
                "api_usage_total": usage,
                "estimated_cost_usd": cost,
                "runtime_seconds": round(self._runtime_seconds(), 3),
                "updated_at": utc_now(),
            }
        )
        atomic_write_json(self.paths["config"], config)

    def _process_rows(
        self,
        rows: pd.DataFrame,
        config: dict[str, Any],
        events: list[dict[str, Any]],
        recovered_tail: bool,
        stage: str,
    ) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str, str], dict[str, Any]]]:
        latest = latest_prediction_records(events)
        successful_keys = {
            key for key, record in latest.items() if record.get("run_status") == "success"
        }
        processed_this_stage = 0
        for row in rows.itertuples(index=False):
            key = (str(row.gutenberg_id), str(row.passage_id), self.model, PROMPT_VERSION)
            if key in successful_keys:
                continue
            record = self._request(row)
            record["run_id"] = config["run_id"]
            append_checkpoint(self.paths["checkpoint"], record)
            events.append(record)
            latest[key] = record
            processed_this_stage += 1
            if self.status_every > 0 and processed_this_stage % self.status_every == 0:
                self._progress(latest, events)
                self._save_outputs(latest, recovered_tail)
                self._refresh_run_config(config, latest, events, f"{stage}_running")
        self._progress(latest, events)
        self._save_outputs(latest, recovered_tail)
        self._refresh_run_config(config, latest, events, f"{stage}_completed")
        return events, latest

    def run(self) -> dict[str, Path]:
        config, events, recovered_tail = self._initialize_or_resume()
        latest = latest_prediction_records(events)
        self._save_outputs(latest, recovered_tail)
        if self.prepare_only:
            self._refresh_run_config(config, latest, events, "prepared")
            return self.paths

        pilot_rows = self.input_frame.head(self.pilot_n)
        print(f"Starting inference pilot on {len(pilot_rows):,} passages...", flush=True)
        events, latest = self._process_rows(
            pilot_rows, config, events, recovered_tail, stage="pilot"
        )
        pilot_keys = {
            (str(row.gutenberg_id), str(row.passage_id), self.model, PROMPT_VERSION)
            for row in pilot_rows.itertuples(index=False)
        }
        pilot_records = [latest[key] for key in pilot_keys if key in latest]
        pilot_passed = len(pilot_records) == len(pilot_rows) and all(
            record.get("run_status") == "success" for record in pilot_records
        )
        # Reloading proves the checkpoint is readable and that successful pilot
        # keys are exactly the keys resume would skip.
        _, reloaded_events, reloaded_tail = load_checkpoint(self.paths["checkpoint"])
        reloaded_latest = latest_prediction_records(reloaded_events)
        resume_skip_verified = all(
            key in reloaded_latest and reloaded_latest[key].get("run_status") == "success"
            for key in pilot_keys
        )
        config["pilot_checks"] = {
            "n_requested": len(pilot_rows),
            "n_success": sum(record.get("run_status") == "success" for record in pilot_records),
            "api_calls_successful": all(record.get("run_status") != "failed" for record in pilot_records),
            "valid_output_formatting": all(record.get("parse_status") == "valid" for record in pilot_records),
            "label_parsing_successful": all(record.get("predicted_label") in CANONICAL_LABELS for record in pilot_records),
            "retry_metadata_recorded": all(int(record.get("attempt_count", 0)) >= 1 for record in pilot_records),
            "retry_observed": any(int(record.get("attempt_count", 0)) > 1 for record in pilot_records),
            "checkpoint_write_verified": len(reloaded_events) >= len(pilot_records),
            "resume_skip_verified": resume_skip_verified,
            "passed": pilot_passed and resume_skip_verified,
        }
        config["checkpoint_truncated_tail_recovered"] = recovered_tail or bool(reloaded_tail)
        config["updated_at"] = utc_now()
        if not config["pilot_checks"]["passed"]:
            config["status"] = "pilot_failed"
            self._save_outputs(latest, recovered_tail)
            self._refresh_run_config(config, latest, events, "pilot_failed")
            raise RuntimeError("Pilot check failed; full inference was not started.")
        self._refresh_run_config(config, latest, events, "pilot_completed")
        print("Pilot passed: API, formatting, parsing, checkpoint, and resume checks succeeded.", flush=True)

        if self.pilot_only:
            self._refresh_run_config(config, latest, events, "pilot_only_completed")
            return self.paths

        print(f"Starting full inference on {len(self.input_frame):,} passages...", flush=True)
        events, latest = self._process_rows(
            self.input_frame, config, events, recovered_tail, stage="full"
        )
        predictions, _ = self._save_outputs(latest, recovered_tail)
        counts = self._counts(latest)
        config["prediction_rows"] = len(predictions)
        self._refresh_run_config(
            config,
            latest,
            events,
            "completed" if counts["successful"] == len(self.input_frame) else "completed_with_failures",
        )
        return self.paths
