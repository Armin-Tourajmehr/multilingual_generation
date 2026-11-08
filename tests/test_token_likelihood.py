"""Numerical fixture for all-candidate token ratios and report exports."""
from __future__ import annotations

import csv
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.token_likelihood import RatioCounts, summarize_token_likelihood_ratios, write_token_likelihood_reports


class TokenLikelihoodTests(unittest.TestCase):
    candidates = ["en", "fr", "es"]

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.model_dir = Path(temporary.name) / "mini"
        fixture = {
            "en": [(1, [.6, .3, .1]), (1, [.3, .6, .1]), (2, [0, 0, 1])],
            "fr": [(1, [.4, .4, .2]), (2, [0, 1, 0])],
            "es": [(1, [.8, .1, .1]), (2, [0, 0, 1])],
        }
        for phase, directory_name in (("prompt", "prompt_token_posteriors"), ("generated", "token_posteriors")):
            directory = self.model_dir / directory_name
            directory.mkdir(parents=True)
            for source, records in fixture.items():
                records = [(0, [1.0, 0.0, 0.0]), *records]
                if phase == "generated" and source == "en":
                    records[1] = (1, [0.0, 1.0, 0.0])
                rows = [
                    {
                        "model": "mini", "input_language": source, "sample_id": index,
                        "layer": layer, **{f"P_{code}": float(score) for code, score in zip(self.candidates, scores)},
                    }
                    for index, (layer, scores) in enumerate(records)
                ]
                pq.write_table(pa.Table.from_pylist(rows), directory / f"{source}.parquet", row_group_size=1)

    def summarize(self, phase, layers=None):
        return summarize_token_likelihood_ratios(
            self.model_dir, model="mini", candidates=self.candidates, phase=phase, layers=layers,
        )

    def test_counts_include_all_foreign_candidates_and_distinguish_zeros(self):
        prompt = self.summarize("prompt")
        self.assertEqual(prompt.layers, {
            1: RatioCounts(below_one=2, equal_one=2, above_one=4),
            2: RatioCounts(below_one=1, above_one=4, undefined=1, infinite=4),
        })
        self.assertEqual(prompt.layers[2].total, 6)
        self.assertEqual(prompt.layers[2].defined, 5)
        expected_histogram = np.zeros(20, dtype=np.int64)
        expected_histogram[[0, 2, 10]] = 1  # Ratios 0, 0.125 and 0.5.
        np.testing.assert_array_equal(prompt.histogram, expected_histogram)

        generated = self.summarize("generated")
        self.assertEqual(generated.layers[1], RatioCounts(below_one=3, equal_one=2, above_one=2, undefined=1))
        self.assertEqual(int(generated.histogram.sum()), 4)

    def test_layer_selection_changes_the_pooled_histogram(self):
        prompt = self.summarize("prompt", layers=frozenset({2}))
        self.assertEqual(list(prompt.layers), [2])
        self.assertEqual(prompt.histogram[0], 1)
        self.assertEqual(int(prompt.histogram.sum()), 1)

    def test_pdf_and_numerical_csvs_cover_both_phases(self):
        pdf, layer_csv, histogram_csv = write_token_likelihood_reports(
            self.model_dir / "figures", model="mini", summaries=[self.summarize("prompt"), self.summarize("generated")],
            language_count=3,
        )
        self.assertTrue(pdf.read_bytes().startswith(b"%PDF"))
        with layer_csv.open(newline="", encoding="utf-8") as stream:
            layer_rows = list(csv.DictReader(stream))
        self.assertEqual([(row["phase"], row["layer"]) for row in layer_rows], [
            ("prompt", "1"), ("prompt", "2"), ("generated", "1"), ("generated", "2"),
        ])
        self.assertEqual(layer_rows[1]["undefined_pairs"], "1")
        self.assertEqual(layer_rows[1]["infinite_pairs"], "4")
        self.assertEqual(float(layer_rows[1]["below_one_fraction"]), .2)
        with histogram_csv.open(newline="", encoding="utf-8") as stream:
            histogram_rows = list(csv.DictReader(stream))
        self.assertEqual(len(histogram_rows), 40)
        for phase, expected in (("prompt", 3), ("generated", 4)):
            selected = [row for row in histogram_rows if row["phase"] == phase]
            self.assertEqual(sum(int(row["pair_count"]) for row in selected), expected)
            self.assertEqual(selected[0]["bin_lower"], "0.00")
            self.assertEqual(selected[-1]["bin_upper"], "1.00")

    def test_combined_report_rejects_different_layer_sets(self):
        with self.assertRaisesRegex(ValueError, "different selected transformer layers"):
            write_token_likelihood_reports(
                self.model_dir / "figures", model="mini",
                summaries=[self.summarize("prompt"), self.summarize("generated", layers=frozenset({2}))],
                language_count=3,
            )

    def test_histogram_writer_failure_preserves_prior_report_set(self):
        output = self.model_dir / "figures"
        summaries = [self.summarize("prompt")]
        paths = write_token_likelihood_reports(output, model="mini", summaries=summaries, language_count=3)
        previous = {path: path.read_bytes() for path in paths}
        original_writer = csv.writer

        def fail_histogram(stream):
            if Path(stream.name).name == "token_likelihood_histogram.csv":
                raise OSError("simulated histogram write failure")
            return original_writer(stream)

        with patch("src.token_likelihood.csv.writer", side_effect=fail_histogram):
            with self.assertRaisesRegex(OSError, "simulated histogram write failure"):
                write_token_likelihood_reports(output, model="mini", summaries=summaries, language_count=3)
        self.assertEqual({path: path.read_bytes() for path in paths}, previous)
        self.assertEqual(set(output.iterdir()), set(paths))


if __name__ == "__main__":
    unittest.main()
