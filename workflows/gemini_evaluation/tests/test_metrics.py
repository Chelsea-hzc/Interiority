import unittest
from types import SimpleNamespace

from gemini_interiority.metrics import classification_metrics
from gemini_interiority.runners import EvaluationRunner


class ClassificationMetricsTests(unittest.TestCase):
    def test_micro_f1_counts_invalid_predictions_as_errors(self):
        records = [
            {"gold_label": "HIGH", "parsed_label": "HIGH"},
            {"gold_label": "LOW", "parsed_label": "NONE"},
            {"gold_label": "NONE", "parsed_label": ""},
        ]

        metrics = classification_metrics(records, "gold_label")

        self.assertAlmostEqual(metrics["accuracy"], 1 / 3)
        self.assertAlmostEqual(metrics["micro_f1"], 1 / 3)

    def test_final_record_requires_an_exact_raw_label(self):
        runner = object.__new__(EvaluationRunner)
        runner.temperature = "model_default"
        runner.client = SimpleNamespace(config=SimpleNamespace(classify_thinking_level="high"))
        item = {
            "item_id": "test_0",
            "paragraph": "Example",
            "gold_label": "LOW",
        }

        invalid = runner._final_record("run", "timestamp", "model", item, "LOW because...", "reason")
        valid = runner._final_record("run", "timestamp", "model", item, " low ", "reason")

        self.assertFalse(invalid["valid_output"])
        self.assertEqual(invalid["parsed_label"], "")
        self.assertTrue(valid["valid_output"])
        self.assertEqual(valid["parsed_label"], "LOW")


if __name__ == "__main__":
    unittest.main()
