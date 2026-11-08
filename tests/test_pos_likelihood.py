"""Small CPU fixtures for POS ratio counts from saved continuation posteriors."""
import csv
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from src.pos_likelihood import THRESHOLDS, UNIVERSAL_POS, save_pos_likelihood_report, summarize_pos_likelihood


CANDIDATES = ["en", "fr", "sw"]


class PosLikelihoodTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.model_dir = Path(temporary.name) / "demo"
        self.partition_dir = self.model_dir / "token_posteriors"
        self.partition_dir.mkdir(parents=True)

    def write_language(self, language, entries):
        rows = []
        for i, (layer, tag, status, scores) in enumerate(entries):
            rows.append({
                "model": "demo", "input_language": language, "sample_id": i + 1, "layer": layer,
                "upos": tag, "pos_status": status,
                **{f"P_{candidate}": score for candidate, score in zip(CANDIDATES, scores)},
            })
        pq.write_table(pa.Table.from_pylist(rows), self.partition_dir / f"{language}.parquet", row_group_size=2)

    def summarize(self, layers=None):
        return summarize_pos_likelihood(self.model_dir, model="demo", candidates=CANDIDATES, layers=layers)

    def test_counts_use_strict_cutoffs_and_only_defined_annotated_foreign_pairs(self):
        self.write_language("en", [
            (0, "NOUN", "ok", (0., 1., 0.)),
            (1, "NOUN", "ok", (0.1, 0.5, 0.4)),
            (1, "VERB", "ok", (0., 1., 0.)),  # 0/1 is dominated, 0/0 is undefined.
            (1, "ADJ", "ok", (0.8, 0.2, 0.)),  # Positive/0 is defined and never dominated.
            (1, "NOUN", "deferred", (0.02, 0.98, 0.)),
            (1, "NOT_UPOS", "ok", (0., 1., 0.)),
            (1, "X", "ok", (0.5, 0.5, 0.)),  # A ratio of 1 is outside the strict threshold.
            (2, "NOUN", "ok", (0., 1., 0.)),
        ])
        self.write_language("fr", [(1, "NOUN", "ok", (0.6, 0.4, 0.)), (2, "NOUN", "ok", (1., 0., 0.))])
        self.write_language("sw", [(1, "POS_UNAVAILABLE", "no_offset", (0., 0., 1.)),
                                   (2, "NOUN", "ok", (0., 1., 0.))])
        summary = self.summarize(frozenset({1}))
        self.assertEqual(summary.layers, (1,))
        self.assertEqual(summary.total_token_rows, 8)
        self.assertEqual(summary.tagged_token_rows, 5)
        self.assertEqual(int(summary.defined_pairs_by_pos.sum()), 9)
        self.assertEqual(int(summary.undefined_pairs_by_pos.sum()), 1)

        noun = UNIVERSAL_POS.index("NOUN")
        verb = UNIVERSAL_POS.index("VERB")
        misc = UNIVERSAL_POS.index("X")
        self.assertEqual(summary.defined_pairs_by_pos[noun], 4)
        self.assertEqual(summary.undefined_pairs_by_pos[verb], 1)
        self.assertEqual(summary.dominated_pairs[:, noun].tolist(), [0, 0, 2, 2, 2, 3])
        self.assertEqual(summary.dominated_pairs[:, verb].tolist(), [1] * len(THRESHOLDS))
        self.assertEqual(summary.dominated_pairs[:, misc].tolist(), [0] * len(THRESHOLDS))

        pdf_path, csv_path, coverage_path = save_pos_likelihood_report(self.model_dir / "figures", summary)
        self.assertTrue(pdf_path.read_bytes().startswith(b"%PDF-"))
        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        row = next(row for row in rows if row["upos"] == "NOUN" and row["threshold"] == "0.3")
        self.assertEqual(row["dominated_pairs"], "2")
        self.assertEqual(row["dominated_pairs_all_pos"], "3")
        self.assertEqual(row["defined_pairs_with_pos"], "4")
        self.assertEqual(row["excluded_pos_rows"], "3")
        self.assertAlmostEqual(float(row["composition_fraction"]), 2 / 3)
        self.assertAlmostEqual(float(row["within_pos_fraction"]), 0.5)
        self.assertAlmostEqual(float(row["annotation_coverage"]), 5 / 8)
        with coverage_path.open(newline="") as handle:
            coverage = {row["input_language"]: row for row in csv.DictReader(handle)}
        self.assertEqual(coverage["en"]["tagged_token_rows"], "4")
        self.assertEqual(coverage["en"]["total_token_rows"], "6")
        self.assertEqual(coverage["sw"]["excluded_pos_rows"], "1")
        self.assertEqual(float(coverage["sw"]["annotation_coverage"]), 0.0)

    def test_composition_is_undefined_when_no_pair_is_dominated(self):
        for i, language in enumerate(CANDIDATES):
            self.write_language(language, [(1, "NOUN", "ok", tuple(float(i == j) for j in range(3)))])
        summary = self.summarize()
        _, csv_path, _ = save_pos_likelihood_report(self.model_dir / "figures", summary)
        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        row = next(row for row in rows if row["upos"] == "NOUN")
        self.assertEqual(row["dominated_pairs_all_pos"], "0")
        self.assertEqual(row["composition_fraction"], "")
        self.assertEqual(float(row["within_pos_fraction"]), 0.0)

    def test_unannotated_archive_reports_the_cpu_annotation_command(self):
        for language in CANDIDATES:
            self.write_language(language, [(1, "POS_UNAVAILABLE", "deferred", (0., 1., 0.))])
        with self.assertRaisesRegex(ValueError, "scripts/annotate_outputs.py"):
            self.summarize()


if __name__ == "__main__":
    unittest.main()
