"""CPU tests of pooled token NMI against a reference implementation."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import normalized_mutual_info_score

from src.affinity import _arithmetic_nmi, summarize_generated_nmi, summarize_prompt_nmi


CANDIDATES = ["fa", "en", "ja"]
PROMPT_SCHEMA = pa.schema([
    ("model", pa.string()), ("sample_id", pa.int64()), ("input_language", pa.string()),
    ("sequence_position", pa.int64()), ("token_id", pa.int64()), ("layer", pa.int64()),
    ("P_en", pa.float32()), ("P_ja", pa.float32()), ("P_fa", pa.float32()),
])
GENERATED_SCHEMA = pa.schema([
    *list(PROMPT_SCHEMA)[:-3],
    ("token", pa.string()), ("upos", pa.string()), ("pos_status", pa.string()),
    ("char_start", pa.int64()), ("char_end", pa.int64()),
    ("P_en", pa.float64()), ("P_ja", pa.float64()), ("P_fa", pa.float64()),
])


def one_hot(index):
    return [float(index == column) for column in range(len(CANDIDATES))]


class TokenNMITests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.model_dir = Path(temp.name)
        self.prompt_dir = self.model_dir / "prompt_token_posteriors"
        self.prompt_dir.mkdir()
        self.generated_dir = self.model_dir / "token_posteriors"
        self.generated_dir.mkdir()

    def write_language(self, language, tokens, *, model="demo", input_language=None, generated=False):
        rows = []
        for index, (layer, probabilities) in enumerate(tokens):
            row = {
                "model": model, "sample_id": 41, "input_language": input_language or language,
                "sequence_position": index, "token_id": 100 + index, "layer": layer,
                **{f"P_{label}": value for label, value in zip(CANDIDATES, probabilities)},
            }
            if generated:
                row.update(token="hello", upos="POS_UNAVAILABLE", pos_status="no_offset", char_start=0, char_end=0)
            rows.append(row)
        schema, directory = (GENERATED_SCHEMA, self.generated_dir) if generated else (PROMPT_SCHEMA, self.prompt_dir)
        table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(table, directory / f"{language}.parquet", row_group_size=2)

    def write_complete(self, layers=(1, 2), *, generated=False):
        for index, language in enumerate(CANDIDATES):
            self.write_language(language, [(layer, one_hot(index)) for layer in layers], generated=generated)

    def summarize(self, *, candidates=None, batch_size=2, generated=False):
        summarize_nmi = summarize_generated_nmi if generated else summarize_prompt_nmi
        return summarize_nmi(
            self.model_dir, model="demo", candidates=candidates or CANDIDATES, batch_size=batch_size,
        )

    def test_pools_token_labels_by_layer_and_breaks_ties_in_candidate_order(self):
        expected_labels = [[0, 0, 1], [1, 2], [2, 0, 2, 1]]
        for source_index, (language, predictions) in enumerate(zip(CANDIDATES, expected_labels)):
            tokens = []
            if language != "en":
                tokens.append((0, one_hot(1)))
            for position, prediction in enumerate(predictions):
                tokens.append((1, one_hot(prediction)))
                perfect = [0.5, 0.5, 0.0] if source_index == position == 0 else one_hot(source_index)
                tokens.append((2, perfect))
            self.write_language(language, tokens)

        source = [source_index for source_index, values in enumerate(expected_labels) for _ in values]
        predicted = [prediction for values in expected_labels for prediction in values]
        expected = normalized_mutual_info_score(source, predicted, average_method="arithmetic")
        for batch_size in (1, 2, 65536):
            with self.subTest(batch_size=batch_size):
                summary = self.summarize(batch_size=batch_size)
                np.testing.assert_array_equal(summary.layers, [1, 2])
                np.testing.assert_array_equal(summary.token_counts, [9, 9])
                np.testing.assert_allclose(summary.nmi, [expected, 1.0], atol=1e-12)
        reordered = self.summarize(candidates=["en", "fa", "ja"])
        self.assertLess(reordered.nmi[1], 1.0)

    def test_arithmetic_normalization_matches_sklearn_including_degenerate_labels(self):
        for raw, known in (
            ([[3, 0, 0], [0, 2, 0], [0, 0, 5]], 1.0),
            ([[0, 3], [5, 0]], 1.0),
            ([[2, 4], [3, 6]], 0.0),
            ([[4, 0], [3, 0]], 0.0),
            ([[0, 4], [0, 0]], 1.0),
            ([[0, 0], [0, 0]], 1.0),
            ([[11, 1], [3, 8]], None),
        ):
            with self.subTest(contingency=raw):
                table = np.array(raw)
                rows, columns = np.indices(table.shape)
                source = np.repeat(rows.ravel(), table.ravel())
                predicted = np.repeat(columns.ravel(), table.ravel())
                actual = _arithmetic_nmi(table)
                expected = normalized_mutual_info_score(source, predicted, average_method="arithmetic")
                self.assertAlmostEqual(actual, expected, places=12)
                if known is not None:
                    self.assertAlmostEqual(actual, known, places=12)

    def test_reports_missing_files_for_the_configured_pool(self):
        self.write_language("fa", [(1, one_hot(0))])
        with self.assertRaises(FileNotFoundError) as error:
            self.summarize()
        for name in ("en.parquet", "ja.parquet"):
            self.assertIn(name, str(error.exception))

    def test_rejects_empty_or_embedding_only_partitions(self):
        self.write_complete()
        for tokens in ([], [(0, one_hot(1))]):
            with self.subTest(tokens=tokens):
                self.write_language("en", tokens)
                with self.assertRaisesRegex(ValueError, r"en\.parquet has no scored transformer-layer prompt tokens"):
                    self.summarize()

    def test_rejects_mismatched_transformer_layers(self):
        self.write_complete()
        self.write_language("en", [(1, one_hot(1))])
        with self.assertRaisesRegex(ValueError, r"en\.parquet has a different transformer layer set \(missing \[2\]"):
            self.summarize()

    def test_rejects_common_gaps_in_transformer_layers(self):
        for layers in ((1, 3), (2, 3)):
            with self.subTest(layers=layers):
                self.write_complete(layers)
                with self.assertRaisesRegex(ValueError, "contiguous transformer layers starting at 1"):
                    self.summarize()

    def test_rejects_wrong_model_or_language_provenance(self):
        self.write_complete()
        for wrong, field in (({"model": "other"}, "model"), ({"input_language": "ja"}, "input_language")):
            with self.subTest(field=field):
                self.write_language("en", [(1, one_hot(1)), (2, one_hot(1))], **wrong)
                with self.assertRaisesRegex(ValueError, f"outside the requested {field} partition"):
                    self.summarize()

    def test_rejects_incomplete_candidate_columns(self):
        self.write_complete()
        path = self.prompt_dir / "en.parquet"
        pq.write_table(pq.read_table(path).drop(["P_fa"]), path)
        with self.assertRaisesRegex(ValueError, "different candidate pool"):
            self.summarize()

    def test_generated_curve_uses_input_labels_even_without_pos_alignment(self):
        self.write_complete()
        for source_index, (language, token_count) in enumerate(zip(CANDIDATES, (3, 2, 4))):
            tokens = [(1, one_hot(1)), (2, one_hot((source_index + 1) % len(CANDIDATES)))] * token_count
            self.write_language(language, tokens + [(0, one_hot(source_index))], generated=True)
        path = self.generated_dir / "ja.parquet"
        without_pos = pq.read_table(path).drop(["upos", "pos_status", "char_start", "char_end"])
        pq.write_table(without_pos, path)

        prompt = self.summarize()
        generated = self.summarize(generated=True, batch_size=1)
        np.testing.assert_array_equal(generated.layers, [1, 2])
        np.testing.assert_array_equal(generated.token_counts, [9, 9])
        np.testing.assert_allclose(prompt.nmi, [1.0, 1.0])
        # Every token has the same surface text, but its input-language label
        # varies. A constant predictor and a bijection therefore score 0 and 1.
        np.testing.assert_allclose(generated.nmi, [0.0, 1.0], atol=1e-12)

    def test_generated_nmi_rejects_missing_and_incomplete_outputs(self):
        self.write_complete()
        self.write_complete(generated=True)
        (self.generated_dir / "en.parquet").unlink()
        with self.assertRaisesRegex(FileNotFoundError, r"Missing generated posterior files.*en\.parquet"):
            self.summarize(generated=True)

        self.write_language("en", [(1, one_hot(1))], generated=True)
        with self.assertRaisesRegex(ValueError, r"en\.parquet has a different transformer layer set \(missing \[2\]"):
            self.summarize(generated=True)

        self.write_language("en", [], generated=True)
        with self.assertRaisesRegex(ValueError, r"en\.parquet has no scored transformer-layer generated tokens"):
            self.summarize(generated=True)


if __name__ == "__main__":
    unittest.main()
