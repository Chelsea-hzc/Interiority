import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from gemini_interiority.inference import (
    BOOK_SUMMARY_COLUMNS,
    INPUT_COLUMNS,
    InferenceRunner,
    book_summaries,
    compact_input,
    estimated_cost,
    load_checkpoint,
    parse_label,
    sha256_file,
)
from gemini_interiority.inference_cli import parse_args


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.config = SimpleNamespace(max_retries=5)
        self.last_batch_request_metadata = []
        self.last_thoughts = ""

    def batch_generate(self, *_args, **_kwargs):
        response = next(self.responses)
        self.last_batch_request_metadata = [
            {"attempts": 1, "failed": response.startswith("ERROR:"), "error_message": ""}
        ]
        return [response]


def sample_frame():
    return pd.DataFrame(
        [
            {
                "gutenberg_id": 1,
                "title": "A",
                "passage_id": 10,
                "text": "Passage one",
                "n_words": 2,
                "position_bin": 1,
                "normalized_position": 0.1,
                "is_base_sample": True,
                "is_core_extra": False,
            },
            {
                "gutenberg_id": 1,
                "title": "A",
                "passage_id": 11,
                "text": "Passage two",
                "n_words": 2,
                "position_bin": 2,
                "normalized_position": 0.2,
                "is_base_sample": False,
                "is_core_extra": True,
            },
        ],
        columns=INPUT_COLUMNS,
    )


class InferenceRunnerTests(unittest.TestCase):
    def test_status_every_default_and_flash_cost(self):
        self.assertEqual(parse_args([]).status_every, 100)
        self.assertEqual(
            estimated_cost(
                "gemini-3.5-flash",
                {
                    "input_tokens": 1_000_000,
                    "output_tokens": 1_000_000,
                    "thinking_tokens": 0,
                    "total_tokens": 2_000_000,
                },
            ),
            10.5,
        )

    def test_progress_includes_count_percent_runtime_and_cost(self):
        frame = sample_frame()
        with tempfile.TemporaryDirectory() as directory:
            runner = InferenceRunner(
                frame,
                FakeClient([]),
                Path(directory),
                "gemini-3.5-flash",
                "developer",
                "high",
            )
            event = {
                "gutenberg_id": 1,
                "passage_id": 10,
                "model": "gemini-3.5-flash",
                "prompt_version": "full_codebook",
                "run_status": "success",
                "_input_tokens": 1_000_000,
                "_output_tokens": 1_000_000,
            }
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                runner._progress(
                    {("1", "10", "gemini-3.5-flash", "full_codebook"): event},
                    [event],
                )
            text = output.getvalue()
            self.assertIn("passages_run=1/2 (50.00%)", text)
            self.assertIn("runtime=", text)
            self.assertIn("estimated_cost=$10.500000", text)

    def test_parse_requires_exact_label(self):
        self.assertEqual(parse_label(" HIGH "), ("HIGH", "valid", "success"))
        self.assertEqual(parse_label("The answer is HIGH"), ("", "invalid_label", "invalid"))
        self.assertEqual(parse_label("ERROR: timeout"), ("", "api_error", "failed"))

    def test_compact_input_keeps_only_approved_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "full.csv"
            destination = Path(directory) / "llm_input.csv"
            frame = sample_frame()
            frame["gold_label"] = ["HIGH", "LOW"]
            frame.to_csv(source, index=False)
            compact = compact_input(source, destination)
            self.assertEqual(list(compact.columns), INPUT_COLUMNS)
            self.assertNotIn("gold_label", pd.read_csv(destination).columns)

    def test_book_summary_denominator_is_successful_predictions(self):
        frame = sample_frame()
        predictions = pd.DataFrame(
            [
                {
                    "gutenberg_id": 1,
                    "passage_id": 10,
                    "predicted_label": "HIGH",
                    "run_status": "success",
                },
                {
                    "gutenberg_id": 1,
                    "passage_id": 11,
                    "predicted_label": "",
                    "run_status": "failed",
                },
            ]
        )
        summary = book_summaries(frame, predictions)
        self.assertEqual(list(summary.columns), BOOK_SUMMARY_COLUMNS)
        base = summary[summary.scope.eq("base")].iloc[0]
        expanded = summary[summary.scope.eq("expanded")].iloc[0]
        self.assertTrue(base.is_core)
        self.assertTrue(expanded.is_core)
        self.assertEqual(base.n_success, 1)
        self.assertEqual(expanded.n_failed, 1)
        self.assertEqual(expanded.high_pct, 1.0)
        self.assertEqual(expanded.avg_interiority_score, 2.0)

    def test_non_core_book_has_only_base_scope(self):
        frame = sample_frame().copy()
        frame["is_base_sample"] = True
        frame["is_core_extra"] = False
        predictions = pd.DataFrame(
            columns=["gutenberg_id", "passage_id", "predicted_label", "run_status"]
        )
        summary = book_summaries(frame, predictions)
        self.assertEqual(summary["scope"].tolist(), ["base"])
        self.assertFalse(bool(summary.iloc[0].is_core))

    def test_pilot_retries_invalid_and_resume_skips_successes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            compact_path = output / "llm_input.csv"
            frame = sample_frame()
            frame.to_csv(compact_path, index=False)
            first = InferenceRunner(
                frame,
                FakeClient(["not a label", "HIGH"]),
                output,
                "fake-model",
                "developer",
                "high",
                max_invalid_retries=1,
                pilot_n=1,
                pilot_only=True,
                input_sha256=sha256_file(compact_path),
            )
            first.run()
            predictions = pd.read_csv(output / "passage_predictions.csv")
            self.assertEqual(predictions.iloc[0].predicted_label, "HIGH")
            self.assertEqual(predictions.iloc[0].attempt_count, 2)

            resumed = InferenceRunner(
                frame,
                FakeClient(["LOW"]),
                output,
                "fake-model",
                "developer",
                "high",
                max_invalid_retries=1,
                pilot_n=1,
                pilot_only=False,
                resume=True,
                input_sha256=sha256_file(compact_path),
            )
            resumed.run()
            predictions = pd.read_csv(output / "passage_predictions.csv")
            self.assertEqual(len(predictions), 2)
            self.assertEqual(predictions.iloc[0].predicted_label, "HIGH")
            self.assertEqual(predictions.iloc[1].predicted_label, "LOW")
            self.assertEqual(predictions.duplicated(
                ["gutenberg_id", "passage_id", "model", "prompt_version"]
            ).sum(), 0)

    def test_checkpoint_recovers_truncated_last_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl"
            path.write_text(
                '{"record_type":"configuration","configuration_fingerprint":"x"}\n{"broken":',
                encoding="utf-8",
            )
            header, events, tail = load_checkpoint(path)
            self.assertEqual(header["configuration_fingerprint"], "x")
            self.assertEqual(events, [])
            self.assertTrue(tail)
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
