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
from src.pca_gpu import GPUPCA, load_pca, save_pca
from src.postprocess import annotate_token_file


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
            tokenizer=SimpleNamespace(
                convert_ids_to_tokens=lambda token_id: f"id{token_id}",
                all_special_ids=[98],
                added_tokens_decoder={99: SimpleNamespace(special=True)},
            ),
            spec={"model_name": "fake/model", "torch_dtype": "float32"},
            model=SimpleNamespace(config=SimpleNamespace(_commit_hash="fixed")),
        )
        self.pcas = [GPUPCA(torch.zeros(2), torch.eye(2), torch.ones(2)) for _ in range(2)]
        self.gmms = [GPULanguageGMM(torch.tensor([[0., 0.], [1., 1.]]), torch.ones(2, 2),
                                    ["en", "fa"], torch.tensor([0.4, 0.6])) for _ in range(2)]
        self.fixtures = {
            "first": (
                "Hi x", [10, 11, 99, 12, 15, 98],
                [(0, 2), (2, 3), (0, 0), (3, 4), (0, 0), (0, 0)],
                [[0., 0.], [9., 9.], [2., 2.], [1., 1.], [3., 3.], [4., 4.]],
            ),
            "empty": ("", [], [], []),
            "last": ("Go.", [13, 14], [(0, 2), (2, 3)], [[2., 1.], [1., 2.]]),
        }
        self.prompt_ids = {"first": [1, 98, 4], "empty": [2], "last": [3, 98]}

    def batches(self, cfg, language, batch_size):
        if language != "en":
            return
        frame = pd.DataFrame({"sample_id": [101, 102, 103], "inputs": ["first", "empty", "last"]})
        for start in range(0, len(frame), batch_size):
            yield frame.iloc[start:start + batch_size].reset_index(drop=True)

    def generate(self, loaded, prompts, generation, runtime):
        return ([self.fixtures[p][1] for p in prompts], [self.fixtures[p][0] for p in prompts],
                max(len(self.prompt_ids[p]) for p in prompts), [self.prompt_ids[p] for p in prompts])

    @staticmethod
    def prompt_vector(token_id):
        return [float(token_id % 10) / 2, -float(token_id % 7) / 2]

    def replay(self, loaded, prompt_ids, generated_ids, runtime):
        continuation_vectors = {tuple(fixture[1]): fixture[3] for fixture in self.fixtures.values()}
        sequences = [
            [self.prompt_vector(token_id) for token_id in prompt] + continuation_vectors[tuple(generated)]
            for prompt, generated in zip(prompt_ids, generated_ids)
        ]
        width = max(len(sequence) for sequence in sequences)
        left_padding = [width - len(sequence) for sequence in sequences]
        hidden = torch.full((len(sequences), width, 2), 7.0)
        for i, values in enumerate(sequences):
            if values:
                hidden[i, left_padding[i]:] = torch.tensor(values)
        return [hidden, hidden + 0.25], left_padding

    def offsets(self, tokenizer, generated_ids, texts):
        by_text = {fixture[0]: fixture[2] for fixture in self.fixtures.values()}
        return [by_text[text] for text in texts]

    def run_analysis(self, cfg=None, directory="results"):
        cfg = cfg or self.cfg
        path = self.root / directory
        with mock.patch.object(pipeline, "iter_aya_batches", side_effect=self.batches), \
                mock.patch.object(pipeline, "generate_batch", side_effect=self.generate), \
                mock.patch.object(pipeline, "replay_prompt_and_continuation", side_effect=self.replay), \
                mock.patch.object(pipeline, "align_generated_token_offsets", side_effect=self.offsets), \
                mock.patch.object(pipeline, "StanzaTagger", FakeTagger), \
                mock.patch.object(pipeline.gc, "collect") as collect:
            self.analysis_stats = pipeline._analyze_model(cfg, self.loaded, self.pcas, self.gmms, path)
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
                        if token_id in {98, 99}:
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
                        generated_text=self.fixtures[prompt][0], generated_token_count=len(self.fixtures[prompt][1]),
                        generated_token_ids=self.fixtures[prompt][1], prompt_token_ids=self.prompt_ids[prompt])
                   for sid, prompt in zip([101, 102, 103], ["first", "empty", "last"])]
        self.assert_parquet_matches(path / "generated_outputs/en.parquet", outputs)
        self.assertEqual(collections, 0)

    def test_original_token_positions_keep_invisible_ids_and_exclude_special_ids(self):
        path, _ = self.run_analysis()
        token_rows = pq.read_table(path / "token_posteriors/en.parquet").to_pylist()
        first = [row for row in token_rows if row["sample_id"] == 101 and row["layer"] == 0]
        self.assertEqual([row["token_id"] for row in first], [10, 11, 12, 15])
        self.assertEqual([row["sequence_position"] for row in first], [0, 1, 3, 4])
        self.assertEqual([row["token_position"] for row in first], [0, 1, 2, 3])
        self.assertEqual((first[1]["token"], first[1]["pos_status"]), (" ", "unaligned"))
        self.assertEqual((first[-1]["token"], first[-1]["model_token"], first[-1]["pos_status"]),
                         ("", "id15", "no_offset"))
        self.assertEqual((first[-1]["char_start"], first[-1]["char_end"]), (0, 0))
        self.assertEqual(self.analysis_stats["generated_tokens"], 8)
        self.assertEqual(self.analysis_stats["analyzed_output_tokens"], 6)

    def test_prompt_and_predecessor_views_score_each_physical_position_once(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["runtime"]["generation_batch_size"] = 3
        with mock.patch.object(pipeline, "_batch_posteriors", wraps=pipeline._batch_posteriors) as score:
            path, _ = self.run_analysis(cfg)

        # Joined widths are 9, 1 and 4. The pads are therefore 0, 8 and 5.
        score.assert_called_once()
        physical_positions = list(zip(score.call_args.args[3], score.call_args.args[4]))
        expected = {(0, pos) for pos in (0, 2, 3, 4, 5, 6, 7)} | {(1, 8)} | {(2, pos) for pos in (5, 6, 7, 8)}
        self.assertCountEqual(physical_positions, expected)

        prompt_table = pq.read_table(path / "prompt_token_posteriors/en.parquet")
        preceding_table = pq.read_table(path / "preceding_token_posteriors/en.parquet")
        consumed_table = pq.read_table(path / "token_posteriors/en.parquet")

        def layer_zero(table):
            return {(row["sample_id"], row["sequence_position"]): row
                    for row in table.to_pylist() if row["layer"] == 0}

        prompt = layer_zero(prompt_table)
        preceding = layer_zero(preceding_table)
        consumed = layer_zero(consumed_table)
        self.assertEqual({key: row["token_id"] for key, row in prompt.items()},
                         {(101, 0): 1, (101, 2): 4, (102, 0): 2, (103, 0): 3})
        self.assertEqual({key: row["token_id"] for key, row in preceding.items()},
                         {key: row["token_id"] for key, row in consumed.items()})
        self.assertEqual({key: (row["context_segment"], row["context_position"], row["context_token_id"])
                          for key, row in preceding.items()}, {
            (101, 0): ("prompt", 2, 4), (101, 1): ("continuation", 0, 10),
            (101, 3): ("continuation", 2, 99), (101, 4): ("continuation", 3, 12),
            (103, 0): ("prompt", 1, 98), (103, 1): ("continuation", 0, 13),
        })
        self.assertEqual(preceding[(101, 0)]["P_en"], prompt[(101, 2)]["P_en"])
        self.assertEqual(preceding[(101, 1)]["P_en"], consumed[(101, 0)]["P_en"])

        for row, vector in (
            (prompt[(102, 0)], self.prompt_vector(2)),
            (preceding[(101, 3)], self.fixtures["first"][3][2]),
            (preceding[(103, 0)], self.prompt_vector(98)),
            (consumed[(103, 0)], self.fixtures["last"][3][0]),
        ):
            expected_p = self.gmms[0].posterior(torch.tensor([vector]))[0].tolist()
            np.testing.assert_allclose([row["P_en"], row["P_fa"]], expected_p)

        for directory, full in (("prompt_token_posteriors", prompt_table),
                                ("preceding_token_posteriors", preceding_table)):
            empty = pq.read_table(path / directory / "fa.parquet")
            self.assertEqual(empty.num_rows, 0)
            self.assertTrue(empty.schema.equals(full.schema))
            self.assertEqual(full.schema.field("P_en").type, pa.float32())
        self.assertEqual(consumed_table.schema.field("P_en").type, pa.float64())

    def test_first_generated_id_has_no_predecessor_when_prompt_is_empty(self):
        self.prompt_ids["first"] = []
        path, _ = self.run_analysis()
        preceding = pq.read_table(path / "preceding_token_posteriors/en.parquet").to_pylist()
        first_sample = [row for row in preceding if row["sample_id"] == 101 and row["layer"] == 0]
        self.assertEqual([row["sequence_position"] for row in first_sample], [1, 3, 4])
        self.assertEqual(first_sample[0]["context_token_id"], 10)
        consumed = pq.read_table(path / "token_posteriors/en.parquet").to_pylist()
        self.assertTrue(any(row["sample_id"] == 101 and row["sequence_position"] == 0 for row in consumed))

    def test_empty_continuation_rerun_does_not_reuse_older_state_rows(self):
        path, _ = self.run_analysis()
        token_path = path / "token_posteriors/en.parquet"
        preceding_path = path / "preceding_token_posteriors/en.parquet"
        self.assertTrue(token_path.exists())
        preceding_schema = pq.read_schema(preceding_path)

        self.fixtures["first"] = self.fixtures["last"] = self.fixtures["empty"]
        self.run_analysis()

        self.assertFalse(token_path.exists())
        preceding = pq.read_table(preceding_path)
        self.assertEqual(preceding.num_rows, 0)
        self.assertTrue(preceding.schema.equals(preceding_schema))
        self.assertEqual(pq.read_table(path / "prompt_token_posteriors/en.parquet").num_rows, 8)

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
        token_path = path / "token_posteriors/en.parquet"
        rows = pq.read_table(token_path).to_pylist()
        status_by_token = {row["token_id"]: (row["pos_status"], row["upos"]) for row in rows}
        self.assertEqual(status_by_token[10], ("deferred", "POS_UNAVAILABLE"))
        self.assertEqual(status_by_token[11], ("unaligned", "NO_WORD_ALIGNMENT"))
        self.assertEqual(status_by_token[15], ("no_offset", "SPECIAL_OR_EMPTY_OFFSET"))
        summary_path = path / "lexical_analysis/pos_posterior_summary/en.csv"
        self.assertFalse(summary_path.exists())

        annotate_token_file(
            token_path, path / "generated_outputs/en.parquet",
            model="fake", language="en", languages=cfg["languages"],
            lexical_config=cfg["lexical_analysis"], summary_path=summary_path,
            batch_size=2, tagger_factory=FakeTagger,
        )
        annotated = pq.read_table(token_path).to_pylist()
        annotated_status = {row["token_id"]: (row["pos_status"], row["upos"]) for row in annotated}
        self.assertEqual(annotated_status[10], ("ok", "NOUN"))
        self.assertEqual(annotated_status[11], status_by_token[11])
        self.assertEqual(annotated_status[15], status_by_token[15])
        self.assertEqual([(row["sequence_position"], row["P_en"], row["P_fa"]) for row in annotated],
                         [(row["sequence_position"], row["P_en"], row["P_fa"]) for row in rows])
        self.assertTrue(summary_path.exists())

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
        self.assertEqual(pq.read_table(path / "token_posteriors/en.parquet").num_rows, 12)
        outputs = pq.read_table(path / "generated_outputs/en.parquet")
        self.assertEqual(outputs.schema.field("generated_token_ids").type, pa.list_(pa.int64()))
        self.assertEqual(outputs.schema.field("prompt_token_ids").type, pa.list_(pa.int64()))
        self.assertEqual(outputs["generated_token_ids"].to_pylist()[1], [])

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
        basis_fingerprint = pipeline._pca_basis_fingerprint(self.pcas)

        def fingerprint(cfg):
            return pipeline._gmm_fingerprint(cfg, self.loaded, counts, basis_fingerprint)

        original = fingerprint(self.cfg)
        changed = copy.deepcopy(self.cfg)
        changed["runtime"]["max_input_tokens"] = 32
        self.assertNotEqual(original, fingerprint(changed))
        changed = copy.deepcopy(self.cfg)
        changed["representation"]["equal_language_priors"] = True
        self.assertNotEqual(original, fingerprint(changed))
        pd.DataFrame({"sample_id": [1, 2], "inputs": ["replacement", "two"]}).to_csv(data_path / "en.csv", index=False)
        clear_aya_cache()
        self.assertNotEqual(original, fingerprint(self.cfg))

    def test_gmm_cache_follows_reloaded_pca_basis_and_rejects_mixed_layers(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["runtime"]["evaluation_batch_size"] = 2
        cfg["representation"].update(gmm_components=1, gmm_covariance_type="diag")
        data_path = self.root / "data"
        data_path.mkdir()
        features = {"en-1": [1., 2.], "en-2": [2., 1.], "fa-1": [4., -2.], "fa-2": [5., -1.]}
        for language in cfg["languages"]:
            pd.DataFrame({"sample_id": [1, 2], "inputs": [f"{language}-1", f"{language}-2"]}).to_csv(
                data_path / f"{language}.csv", index=False
            )

        gmm_root = self.root / "gmm"
        pca_root = self.root / "pca"
        counts = {"en": 2, "fa": 2}
        layer_path = gmm_root / self.loaded.name / "layer_000.pt"

        def reload_pcas(pcas):
            paths = [pca_root / self.loaded.name / f"layer_{i:03d}.pt" for i in range(len(pcas))]
            for path, pca in zip(paths, pcas):
                save_pca(path, pca)
            return [load_pca(path, self.loaded.device) for path in paths]

        def load_or_fit(pcas):
            return pipeline._load_gmms_or_fit(cfg, self.loaded, pcas, gmm_root, pca_root, counts)

        def cpu_stats(n_layers, n_languages, pca_dim, device):
            sums = torch.zeros(n_layers, n_languages, pca_dim, device=device)
            return sums, torch.zeros_like(sums), torch.zeros(n_layers, n_languages, dtype=torch.long, device=device)

        def forward_text(loaded, prompts, max_input_tokens, runtime):
            hidden = torch.tensor([features[prompt] for prompt in prompts]).unsqueeze(1)
            return [hidden, hidden + 0.5], torch.ones(hidden.shape[:2], dtype=torch.bool)

        # Use the real fit and serialization path with CPU statistics and fixed model vectors.
        with mock.patch.object(pipeline, "init_gmm_stats", side_effect=cpu_stats), \
                mock.patch.object(pipeline, "forward_text_batch", side_effect=forward_text) as forward:
            original = load_or_fit(reload_pcas(self.pcas))
            forward.assert_called()
            original_layer_bytes = layer_path.read_bytes()

            forward.reset_mock()
            reused = load_or_fit(reload_pcas(self.pcas))
            forward.assert_not_called()
            torch.testing.assert_close(reused[0].means, original[0].means)

            first, second = self.pcas
            sign = torch.tensor([-1., 1.])
            flipped = GPUPCA(first.mean, sign[:, None] * first.components, first.explained_variance)
            updated = load_or_fit(reload_pcas([flipped, second]))
            forward.assert_called()
            torch.testing.assert_close(updated[0].means, original[0].means * sign)
            torch.testing.assert_close(updated[1].means, original[1].means)

            forward.reset_mock()
            shift = torch.tensor([0.5, 1.25])
            centered = GPUPCA(second.mean + shift, second.components, second.explained_variance)
            current_basis = reload_pcas([flipped, centered])
            recentered = load_or_fit(current_basis)
            forward.assert_called()
            torch.testing.assert_close(recentered[1].means, original[1].means - shift)

            # Simulate a stale layer alongside current metadata and another current layer.
            layer_path.write_bytes(original_layer_bytes)
            forward.reset_mock()
            repaired = load_or_fit(current_basis)
            forward.assert_called()
            torch.testing.assert_close(repaired[0].means, recentered[0].means)
            torch.testing.assert_close(repaired[1].means, recentered[1].means)

            forward.reset_mock()
            load_or_fit(current_basis)
            forward.assert_not_called()

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
