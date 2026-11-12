import unittest
from pathlib import Path

from scripts.create_three_novels_test_set import TARGET_TITLES, build_test_set


class ThreeNovelsTestSetTests(unittest.TestCase):
    def test_builds_balanced_book_test_set_without_changing_source_split_audit(self):
        root = Path(__file__).resolve().parents[1]
        result = build_test_set(root / "df_annotation_final.csv")

        self.assertEqual(len(result), 120)
        self.assertEqual(result["title"].value_counts().to_dict(), {title: 40 for title in TARGET_TITLES})
        self.assertEqual(set(result["split"]), {"test"})
        self.assertEqual(set(result["gold_label"]), {"HIGH", "LOW", "NONE"})
        self.assertEqual(
            result.groupby("title")["source_split"].first().to_dict(),
            {
                "Emma": "train",
                "Jane Eyre: An Autobiography": "dev",
                "Middlemarch": "test",
            },
        )


if __name__ == "__main__":
    unittest.main()
