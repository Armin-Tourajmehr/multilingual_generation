"""CPU-only equivalence checks for the batched analysis/output path."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from src import pipeline
from src.data import clear_aya_cache
from src.gmm import GPULanguageGMM
from src.lexical import align_pos_to_token
from src.pca_gpu import GPUPCA


class FakeTagger:
    def __init__(self, config):
        pass

    def tag_text(self, text, language):
        spans = [(0, 2, "NOUN"), (3, 4, "VERB")] if text == "Hi x" else [(0, 2, "NOUN"), (2, 3, "PUNCT")]
        return [dict(sentence_id=0, word_id=i, word_text=text[a:b], char_start=a, char_end=b,
                     upos=upos, xpos="", feats="") for i, (a, b, upos) in enumerate(spans)], "ok"

    def close(self):
        pass


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = {
            "project": {"seed": 42},
            "languages": ["en", "fa"],
            "data": {"root_dir": str(self.root / "data"), "input_column": "inputs",
                     "sample_id_column": "sample_id", "max_samples_per_language": 2000, "sampling_seed": 42},
            "runtime": {"generation_batch_size": 2, "posterior_batch_size": 1, "max_input_tokens": 512},
            "generation": {"max_new_tokens": 64},
            "representation": {"pca_dim": 2, "pca_solver": "gpu_covariance", "gmm_reg_covar": 1e-6},
            "wikipedia": {"dataset_name": "unused", "max_samples_per_language": 2, "max_tokens_per_document": 128},
            "lexical_analysis": {"enabled": False},
        }
        self.loaded = SimpleNamespace(
            name="fake", device=torch.device("cpu"),
            tokenizer=SimpleNamespace(convert_ids_to_tokens=lambda token_id: f"id{token_id}"),
            spec={"model_name": "fake/model", "torch_dtype": "float32"},
            model=SimpleNamespace(config=SimpleNamespace(_commit_hash="fixed")),
        )
        self.pcas = [GPUPCA(torch.zeros(2), torch.eye(2), torch.ones(2)) for _ in range(2)]
        self.gmms = [GPULanguageGMM(torch.tensor([[0., 0.], [1., 1.]]), torch.ones(2, 2),
                                    ["en", "fa"], torch.tensor([0.4, 0.6])) for _ in range(2)]
        self.fixtures = {
            "first": ("Hi x", [10, 11, 12], [(0, 2), (2, 3), (3, 4)], [[0., 0.], [9., 9.], [1., 1.]]),
            "empty": ("", [], [], []),
            "last": ("Go.", [13, 14], [(0, 2), (2, 3)], [[2., 1.], [1., 2.]]),
        }

    def batches(self, cfg, language, batch_size):
        if language != "en":
            return
        frame = pd.DataFrame({"sample_id": [101, 102, 103], "inputs": ["first", "empty", "last"]})
        for start in range(0, len(frame), batch_size):
            yield frame.iloc[start:start + batch_size].reset_index(drop=True)

    def generate(self, loaded, prompts, generation, runtime):
        return ([self.fixtures[p][1] for p in prompts], [self.fixtures[p][0] for p in prompts],
                1, [[{"first": 1, "empty": 2, "last": 3}[p]] for p in prompts])

    def replay(self, loaded, prompt_ids, generated_ids, texts, runtime):
        reverse = {value[0]: value for value in self.fixtures.values()}
        width = max((len(ids) for ids in generated_ids), default=0)
        hidden = torch.zeros(len(texts), width, 2)
        for i, text in enumerate(texts):
            values = reverse[text][3]
            if values:
                hidden[i, :len(values)] = torch.tensor(values)
        return ([hidden, hidden + 0.25], generated_ids, [reverse[t][2] for t in texts],
                [[1] * len(ids) for ids in generated_ids])

    def run_analysis(self, cfg=None, directory="results"):
        cfg = cfg or self.cfg
        path = self.root / directory
        with mock.patch.object(pipeline, "iter_aya_batches", side_effect=self.batches), \
                mock.patch.object(pipeline, "generate_batch", side_effect=self.generate), \
                mock.patch.object(pipeline, "forward_generated_texts_with_context", side_effect=self.replay), \
                mock.patch.object(pipeline, "StanzaTagger", FakeTagger), \
                mock.patch.object(pipeline.gc, "collect") as collect:
            pipeline._analyze_model(cfg, self.loaded, self.pcas, self.gmms, path)
        return path, collect.call_count

    def expected_rows(self, lexical_enabled=False, status="disabled"):
        tokens, sentences = [], []
        tagger = FakeTagger({})
        # Existing output order is generation batch, layer, sample, filtered token.
        for prompts, sids in [(["first", "empty"], [101, 102]), (["last"], [103])]:
            for layer in range(2):
                for prompt, sid in zip(prompts, sids):
                    text, ids, spans, vectors = self.fixtures[prompt]
                    words, tag_status = tagger.tag_text(text, "en") if lexical_enabled and text else ([], status)
                    rows = []
                    for seq, (token_id, (start, end)) in enumerate(zip(ids, spans)):
                        if end <= start or not text[start:end].strip():
                            continue
                        values = self.gmms[layer].posterior(
                            self.pcas[layer].transform(torch.tensor([vectors[seq]]) + layer * 0.25)
                        )[0].tolist()
                        row = {"model": "fake", "sample_id": sid, "input_language": "en", "token_position": len(rows),
                               "sequence_position": seq, "token": text[start:end], "model_token": f"id{token_id}",
                               "token_id": token_id, "char_start": start, "char_end": end,
                               **align_pos_to_token(text, start, end, words, tag_status)}
                        # Match the established parquet field order.
                        row = {key: row[key] for key in (
                            "model", "sample_id", "input_language", "token_position", "sequence_position", "token",
                            "model_token", "token_id", "char_start", "char_end", "word_text", "sentence_id", "word_id",
                            "upos", "xpos", "feats", "pos_status")}
                        row.update(layer=layer, P_en=values[0], P_fa=values[1])
                        rows.append(row)
                    tokens.extend(rows)
                    sentences.append({"model": "fake", "sample_id": sid, "input_language": "en", "layer": layer,
                                      "num_output_tokens": len(rows),
                                      **{f"P_{lang}": sum(row[f"P_{lang}"] for row in rows) / len(rows) if rows else 0.0
                                         for lang in ("en", "fa")}})
        return tokens, sentences

    def assert_parquet_matches(self, path, rows):
        expected = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
        actual = pq.read_table(path)
        self.assertTrue(actual.schema.equals(expected.schema, check_metadata=False))
        self.assertEqual(actual.column_names, expected.column_names)
        for actual_row, expected_row in zip(actual.to_pylist(), rows):
            for key, value in expected_row.items():
                if isinstance(value, float):
                    self.assertAlmostEqual(actual_row[key], value, places=7)
                else:
                    self.assertEqual(actual_row[key], value)
        self.assertEqual(actual.num_rows, len(rows))

    def test_columnar_outputs_preserve_schema_order_values_and_empty_samples(self):
        path, collections = self.run_analysis()
        tokens, sentences = self.expected_rows()
        self.assert_parquet_matches(path / "token_posteriors/en.parquet", tokens)
        self.assert_parquet_matches(path / "sentence_level_posteriors/en.parquet", sentences)
        outputs = [dict(model="fake", sample_id=sid, input_language="en", prompt=prompt,
                        generated_text=self.fixtures[prompt][0], generated_token_count=len(self.fixtures[prompt][1]))
                   for sid, prompt in zip([101, 102, 103], ["first", "empty", "last"])]
        self.assert_parquet_matches(path / "generated_outputs/en.parquet", outputs)
        self.assertEqual(collections, 0)

    def test_pos_summary_matches_token_aggregates_and_can_be_disabled(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["lexical_analysis"]["enabled"] = True
        path, _ = self.run_analysis(cfg)
        tokens, _ = self.expected_rows(lexical_enabled=True)
        self.assert_parquet_matches(path / "token_posteriors/en.parquet", tokens)
        summary = pd.read_csv(path / "lexical_analysis/pos_posterior_summary/en.csv")
        for row in summary.itertuples(index=False):
            selected = [token for token in tokens if token["layer"] == row.layer and token["upos"] == row.upos
                        and token["pos_status"] == row.pos_status]
            self.assertEqual(row.num_tokens, len(selected))
            self.assertAlmostEqual(row.mean_posterior, sum(t[f"P_{row.posterior_language}"] for t in selected) / len(selected))
        cfg["lexical_analysis"]["save_pos_posterior_summary"] = False
        other, _ = self.run_analysis(cfg)
        self.assertFalse((other / "lexical_analysis/pos_posterior_summary/en.csv").exists())

    def test_deferred_mode_does_not_construct_stanza_or_write_summary(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["lexical_analysis"].update(enabled=True, deferred=True)
        # run_analysis patches the tagger class internally; replace its constructor with a failure.
        with mock.patch.object(FakeTagger, "__init__", side_effect=AssertionError("Stanza must be deferred")):
            path, _ = self.run_analysis(cfg)
        rows = pq.read_table(path / "token_posteriors/en.parquet").to_pylist()
        self.assertTrue(all(row["pos_status"] == "deferred" and row["upos"] == "POS_UNAVAILABLE" for row in rows))
        self.assertFalse((path / "lexical_analysis/pos_posterior_summary").exists())

    def test_optional_gc_interval(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["runtime"]["gc_collect_every_batches"] = 2
        _, calls = self.run_analysis(cfg)
        self.assertEqual(calls, 1)

    def test_an_entire_empty_generation_batch_keeps_zero_sentence_rows(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["runtime"]["generation_batch_size"] = 1
        path, _ = self.run_analysis(cfg)
        rows = pq.read_table(path / "sentence_level_posteriors/en.parquet").to_pylist()
        empty = [row for row in rows if row["sample_id"] == 102]
        self.assertEqual(len(empty), 2)
        self.assertTrue(all(row["num_output_tokens"] == 0 and row["P_en"] == 0.0 and row["P_fa"] == 0.0
                            for row in empty))
        self.assertEqual(pq.read_table(path / "token_posteriors/en.parquet").num_rows, 8)

    def test_batched_gather_matches_per_token_reference_and_honors_chunks(self):
        hidden = [torch.arange(24, dtype=torch.float32).reshape(3, 4, 2) / 10,
                  torch.arange(24, dtype=torch.float32).reshape(3, 4, 2) / 20]
        sample_indices, positions = [0, 0, 1, 2, 2], [0, 3, 1, 0, 2]
        expected = []
        for layer in range(2):
            vectors = torch.stack([hidden[layer][sample, pos] for sample, pos in zip(sample_indices, positions)])
            expected.append(self.gmms[layer].posterior(self.pcas[layer].transform(vectors)).numpy())
            self.gmms[layer].posterior = mock.Mock(wraps=self.gmms[layer].posterior)
        with mock.patch.object(torch, "stack", wraps=torch.stack) as stack:
            actual = pipeline._batch_posteriors(hidden, self.pcas, self.gmms, sample_indices, positions, 2, 2)
        np.testing.assert_allclose(actual, np.stack(expected), rtol=1e-6, atol=1e-7)
        self.assertEqual(stack.call_count, 1)  # Stack layers once; no per-token stack/gather loop.
        for gmm in self.gmms:
            self.assertEqual([len(call.args[0]) for call in gmm.posterior.call_args_list], [2, 2, 1])
        empty = pipeline._batch_posteriors(hidden, self.pcas, self.gmms, [], [], 2, 2)
        self.assertEqual(empty.shape, (2, 0, 2))

    def test_fingerprint_changes_for_truncation_content_and_prior_policy(self):
        data_path = self.root / "data"
        data_path.mkdir()
        for language in self.cfg["languages"]:
            pd.DataFrame({"sample_id": [1, 2], "inputs": ["one", "two"]}).to_csv(data_path / f"{language}.csv", index=False)
        counts = {"en": 2, "fa": 2}
        fingerprint = pipeline._gmm_fingerprint(self.cfg, self.loaded, counts)
        changed = copy.deepcopy(self.cfg)
        changed["runtime"]["max_input_tokens"] = 32
        self.assertNotEqual(fingerprint, pipeline._gmm_fingerprint(changed, self.loaded, counts))
        changed = copy.deepcopy(self.cfg)
        changed["representation"]["equal_language_priors"] = True
        self.assertNotEqual(fingerprint, pipeline._gmm_fingerprint(changed, self.loaded, counts))
        pd.DataFrame({"sample_id": [1, 2], "inputs": ["replacement", "two"]}).to_csv(data_path / "en.csv", index=False)
        clear_aya_cache()
        self.assertNotEqual(fingerprint, pipeline._gmm_fingerprint(self.cfg, self.loaded, counts))

    def test_partition_writer_accepts_arrow_tables_and_legacy_rows(self):
        path = self.root / "partition.parquet"
        writer = pipeline.PartitionWriter(path)
        writer.write_table(pa.table({"value": [1, 2]}))
        writer.write([{"value": 3}])
        writer.close()
        self.assertEqual(pq.read_table(path)["value"].to_pylist(), [1, 2, 3])

    def test_pca_fingerprint_tracks_the_model_attention_backend(self):
        self.loaded.spec["attn_implementation"] = "eager"
        eager = pipeline._pca_fingerprint(self.cfg, self.loaded)
        self.loaded.spec["attn_implementation"] = "sdpa"
        self.assertNotEqual(eager, pipeline._pca_fingerprint(self.cfg, self.loaded))

    def test_uniform_prior_option_is_honored_and_empirical_default_is_preserved(self):
        counts = {"en": 1, "fa": 3}
        empirical = pipeline._configured_language_priors(self.cfg, counts, torch.device("cpu"))
        torch.testing.assert_close(empirical, torch.tensor([0.25, 0.75]))
        self.cfg["representation"]["equal_language_priors"] = True
        uniform = pipeline._configured_language_priors(self.cfg, counts, torch.device("cpu"))
        torch.testing.assert_close(uniform, torch.tensor([0.5, 0.5]))

    def test_optional_phase_measurement_is_inert_when_disabled(self):
        with mock.patch.object(torch.cuda, "synchronize") as synchronize:
            with pipeline._measure_phase(None, "analysis", torch.device("cuda")):
                pass
        synchronize.assert_not_called()
        performance = {"phase_seconds": {}}
        with pipeline._measure_phase(performance, "analysis", torch.device("cpu")):
            pass
        self.assertGreaterEqual(performance["phase_seconds"]["analysis"], 0)


if __name__ == "__main__":
    unittest.main()
