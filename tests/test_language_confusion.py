"""CPU checks of layer and pooled GMM confusion reports."""
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.language_confusion import save_language_confusion_report, summarize_language_confusion


CANDIDATES = ["fa", "en"]


class LanguageConfusionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.model_dir = Path(temporary.name) / "demo"
        prompt_scores = {
            "fa": {0: [0.5], 1: [0.5, 0.1], 2: [0.8], 3: [1.0, 0.9, 0.8]},
            "en": {0: [0.5], 1: [0.8, 0.1, 0.2], 2: [0.9], 3: [0.0]},
        }
        for phase, directory in (("prompt", "prompt_token_posteriors"), ("generated", "token_posteriors")):
            path = self.model_dir / directory
            path.mkdir(parents=True)
            for language, layers in prompt_scores.items():
                rows = []
                for layer, probabilities in layers.items():
                    for position, probability in enumerate(probabilities):
                        if phase == "generated":
                            probability = 0.05
                        rows.append({"model": "demo", "input_language": language, "sample_id": 101,
                                     "sequence_position": position, "token_id": 200 + position, "layer": layer,
                                     "P_en": 1 - probability, "P_fa": probability})
                table = pa.Table.from_pylist(rows)
                if phase == "prompt":
                    for field in ("P_en", "P_fa"):
                        table = table.set_column(table.schema.get_field_index(field), field, table[field].cast(pa.float32()))
                else:
                    table = table.append_column("pos_status", pa.array(["no_offset"] * len(rows)))
                pq.write_table(table, path / f"{language}.parquet", row_group_size=2)

    def summarize(self, phase, layers=None):
        return summarize_language_confusion(
            self.model_dir, model="demo", candidates=CANDIDATES, phase=phase, layers=layers,
        )

    def test_selected_layer_counts_and_pooled_csv_rates_for_both_phases(self):
        selected = frozenset({1, 3})
        prompt = self.summarize("prompt", selected)
        generated = self.summarize("generated", selected)
        np.testing.assert_array_equal(prompt.layers, [1, 3])
        np.testing.assert_array_equal(prompt.counts, [[[1, 1], [1, 2]], [[3, 0], [0, 1]]])
        np.testing.assert_array_equal(prompt.pooled_counts, [[4, 1], [1, 3]])
        np.testing.assert_array_equal(generated.pooled_counts, [[0, 5], [0, 4]])

        pdf_path = self.model_dir / "figures" / "language_confusion.pdf"
        csv_path = self.model_dir / "figures" / "language_confusion.csv"
        save_language_confusion_report(pdf_path, csv_path, model="demo", summaries=[prompt, generated])
        self.assertTrue(pdf_path.read_bytes().startswith(b"%PDF"))
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 24)
        self.assertEqual({row["phase"] for row in rows}, {"prompt", "generated"})
        self.assertEqual({row["layer"] for row in rows if row["scope"] == "layer"}, {"1", "3"})
        keyed = {(row["phase"], row["scope"], row["layer"], row["source_language"], row["candidate_language"]): row
                 for row in rows}

        # A tie goes to fa in candidate order, even though the file lists P_en first.
        tied_layer = keyed["prompt", "layer", "1", "fa", "fa"]
        self.assertEqual((int(tied_layer["count"]), float(tied_layer["row_rate"])), (1, 0.5))
        pooled = keyed["prompt", "pooled", "", "fa", "fa"]
        self.assertEqual((int(pooled["count"]), int(pooled["row_total"]), float(pooled["row_rate"])), (4, 5, 0.8))
        generated_cell = keyed["generated", "pooled", "", "fa", "en"]
        self.assertEqual((int(generated_cell["count"]), float(generated_cell["row_rate"])), (5, 1.0))

    def test_defaults_to_all_transformer_layers(self):
        summary = self.summarize("prompt")
        np.testing.assert_array_equal(summary.layers, [1, 2, 3])
        np.testing.assert_array_equal(summary.counts[1], [[1, 0], [1, 0]])


if __name__ == "__main__":
    unittest.main()
