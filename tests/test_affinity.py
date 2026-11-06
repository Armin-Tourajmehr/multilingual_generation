"""CPU-only checks of sample-balanced, matched affinity summaries."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.affinity import summarize_language


def posterior(sample_id, layer, en, *, token_count=None, model="demo", source="en"):
    row = {"model": model, "input_language": source, "sample_id": sample_id, "layer": layer,
           "P_en": en, "P_fa": 1 - en}
    if token_count is not None:
        row["num_output_tokens"] = token_count
        if token_count == 0:
            row["P_en"] = row["P_fa"] = 0.0
    return row


class AffinityTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.prompt_path = root / "prompt.parquet"
        self.continuation_path = root / "continuation.parquet"

    def write(self, prompts, continuations):
        prompt_table = pa.Table.from_pylist(prompts)
        for name in ("P_en", "P_fa"):
            index = prompt_table.schema.get_field_index(name)
            prompt_table = prompt_table.set_column(index, name, prompt_table[name].cast(pa.float32()))
        pq.write_table(prompt_table, self.prompt_path, row_group_size=2)
        pq.write_table(pa.Table.from_pylist(continuations), self.continuation_path, row_group_size=2)

    def summarize(self, batch_size=2):
        # Candidate order is deliberately opposite to the order in both files.
        return summarize_language(
            self.prompt_path, self.continuation_path, model="demo", source_language="en",
            candidates=["fa", "en"], batch_size=batch_size,
        )

    def test_pairs_by_layer_and_sample_and_weights_samples_equally_across_batches(self):
        self.write(
            [posterior("A", 4, 0.1), posterior("B", 1, 0.2), posterior("A", 1, 0.8),
             posterior("B", 1, 0.4), posterior("B", 4, 0.4), posterior("C", 1, 0.0),
             posterior("B", 1, 0.6), posterior("E", 1, 0.01), posterior("A", 9, 0.9)],
            [posterior("Z", 9, 0.5, token_count=3), posterior("A", 4, 0.6, token_count=3),
             posterior("D", 1, 0.2, token_count=4), posterior("A", 1, 0.3, token_count=10),
             posterior("E", 1, 0.0, token_count=0), posterior("B", 1, 0.7, token_count=1),
             posterior("E", 12, 0.0, token_count=0)],
        )

        # The three B prompt tokens span separate batches. B and A each have
        # one vote in layer 1, despite their different lengths in both phases.
        expected_prompt = [[0.4, 0.9, np.nan, np.nan], [0.6, 0.1, np.nan, np.nan]]
        expected_continuation = [[0.5, 0.4, np.nan, np.nan], [0.5, 0.6, np.nan, np.nan]]
        for batch_size in (1, 2, 65536):
            with self.subTest(batch_size=batch_size):
                summary = self.summarize(batch_size)
                self.assertIsNotNone(summary)
                self.assertEqual(summary.source_language, "en")
                self.assertEqual(summary.candidates, ["fa", "en"])
                np.testing.assert_array_equal(summary.layers, [1, 4, 9, 12])
                np.testing.assert_array_equal(summary.paired_counts, [2, 1, 0, 0])
                np.testing.assert_allclose(summary.prompt, expected_prompt)
                np.testing.assert_allclose(summary.continuation, expected_continuation)
                np.testing.assert_allclose(summary.difference, [[0.1, -0.5, np.nan, np.nan],
                                                               [-0.1, 0.5, np.nan, np.nan]], atol=1e-7)

    def test_returns_none_without_matched_nonempty_continuations(self):
        self.write(
            [posterior(1, 0, 0.9), posterior(2, 0, 0.5)],
            [posterior(1, 0, 0, token_count=0), posterior(2, 1, 0.8, token_count=2),
             posterior(3, 0, 0.3, token_count=2)],
        )
        self.assertIsNone(self.summarize())

    def test_rejects_duplicate_sentence_means_across_batches(self):
        self.write([posterior(1, 0, 0.5)],
                   [posterior(1, 0, 0, token_count=0), posterior(1, 0, 0.5, token_count=2)])
        with self.assertRaisesRegex(ValueError, "duplicate sentence means"):
            self.summarize(batch_size=1)

    def test_rejects_rows_from_another_partition(self):
        for row, field in ((posterior(1, 0, 0.5, model="other"), "model"),
                           (posterior(1, 0, 0.5, source="fa"), "input_language")):
            with self.subTest(field=field):
                self.write([row], [posterior(1, 0, 0.5, token_count=2)])
                with self.assertRaisesRegex(ValueError, f"outside the requested {field} partition"):
                    self.summarize()

    def test_rejects_mismatched_candidate_pool(self):
        self.write([posterior(1, 0, 0.5)], [posterior(1, 0, 0.5, token_count=2)])
        prompt = pq.read_table(self.prompt_path).append_column("P_extra", pa.array([0.0]))
        pq.write_table(prompt, self.prompt_path)
        with self.assertRaisesRegex(ValueError, "different candidate pool"):
            self.summarize()

    def test_rejects_bad_probabilities_in_either_phase(self):
        for updates, message in (({"P_en": float("nan")}, "posterior probabilities"),
                                 ({"P_en": 1.1}, "posterior probabilities"),
                                 ({"P_en": 0.4}, "do not sum to 1")):
            for phase in ("prompt", "continuation"):
                with self.subTest(updates=updates, phase=phase):
                    prompt, continuation = posterior(1, 0, 0.5), posterior(1, 0, 0.5, token_count=2)
                    (prompt if phase == "prompt" else continuation).update(updates)
                    self.write([prompt], [continuation])
                    with self.assertRaisesRegex(ValueError, message):
                        self.summarize()


if __name__ == "__main__":
    unittest.main()
