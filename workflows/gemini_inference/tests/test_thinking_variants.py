import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from gemini_interiority.runners import DEFAULT_MODEL_PRICES, EvaluationRunner


class ThinkingVariantTests(unittest.TestCase):
    def test_gemini_31_pro_preview_default_prices(self):
        self.assertEqual(
            DEFAULT_MODEL_PRICES["gemini-3.1-pro-preview"],
            {
                "input_cost_per_1m_tokens": 2.00,
                "output_cost_per_1m_tokens": 12.00,
            },
        )

    def make_runner(self, root: Path, thinking_levels):
        client = SimpleNamespace(config=SimpleNamespace(classify_thinking_level="high"))
        paths = {
            "predictions": root / "predictions",
            "final_predictions": root / "final" / "predictions",
            "final": root / "final",
        }
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        return EvaluationRunner(
            df=pd.DataFrame(),
            models=["gemini-3.5-flash"],
            client=client,
            output_paths=paths,
            thinking_levels=thinking_levels,
        )

    def test_multiple_levels_create_distinct_variants_and_caches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = self.make_runner(Path(temp_dir), ["minimal", "low", "medium", "high"])

            self.assertEqual(
                runner.models,
                [
                    "gemini-3.5-flash [thinking=minimal]",
                    "gemini-3.5-flash [thinking=low]",
                    "gemini-3.5-flash [thinking=medium]",
                    "gemini-3.5-flash [thinking=high]",
                ],
            )
            cache_paths = {
                runner._cache_path(model, "final", final=True) for model in runner.models
            }
            self.assertEqual(len(cache_paths), 4)

            for model, expected_level in zip(runner.models, runner.thinking_levels):
                self.assertEqual(runner._activate_variant(model), "gemini-3.5-flash")
                self.assertEqual(runner.client.config.classify_thinking_level, expected_level)

    def test_single_level_preserves_original_model_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = self.make_runner(Path(temp_dir), ["medium"])

            self.assertEqual(runner.models, ["gemini-3.5-flash"])
            self.assertEqual(runner._variant(runner.models[0])["thinking_level"], "medium")

    def test_final_comparison_includes_each_thinking_level(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = self.make_runner(Path(temp_dir), ["minimal", "low", "medium", "high"])
            records = []
            for index, model in enumerate(runner.models):
                item = {
                    "item_id": f"test_{index}",
                    "paragraph": "Example",
                    "gold_label": "LOW",
                }
                records.append(runner._final_record("run", "timestamp", model, item, "LOW"))

            outputs = runner._save_final_outputs(records, {"run_id": "run"}, "completed")
            comparison = pd.read_csv(outputs["final_comparison"])

            self.assertEqual(set(comparison["thinking_level"]), {"minimal", "low", "medium", "high"})
            self.assertEqual(set(comparison["model_version"]), {"gemini-3.5-flash"})
            self.assertEqual(len(comparison), 4)


if __name__ == "__main__":
    unittest.main()
