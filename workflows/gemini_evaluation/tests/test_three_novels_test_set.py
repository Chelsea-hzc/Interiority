import unittest
from pathlib import Path

try:
    from scripts.create_three_novels_test_set import TARGET_TITLES, build_test_set
except ModuleNotFoundError as error:
    if error.name not in {"scripts", "scripts.create_three_novels_test_set"}:
        raise
    TARGET_TITLES = ()
    build_test_set = None


class ThreeNovelsTestSetTests(unittest.TestCase):
    @unittest.skipIf(
        build_test_set is None,
        "Missing historical dependency: scripts/create_three_novels_test_set.py",
    )
    def test_builds_balanced_book_test_set_without_changing_source_split_audit(self):
        repository_root = Path(__file__).resolve().parents[3]
        result = build_test_set(repository_root / "data" / "annotation" / "df_annotation_final.csv")

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
