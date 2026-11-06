"""Streaming POS annotation checks without Stanza downloads or model loading."""
from __future__ import annotations

import csv
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pyarrow as pa
import pyarrow.parquet as pq

from src.lexical import align_pos_to_token
from src.postprocess import ANNOTATION_COLUMNS, SUMMARY_COLUMNS, annotate_outputs, annotate_token_file


class FakeTagger:
    def __init__(self, cfg, *, fail_on=None):
        self.cfg = cfg
        self.fail_on = fail_on
        self.calls = []
        self.closed = 0

    def tag_text(self, text, language):
        self.calls.append((text, language))
        if text == self.fail_on:
            raise RuntimeError("tagging failed")
        return [{
            "word_text": char,
            "char_start": index,
            "char_end": index + 1,
            "sentence_id": 0,
            "word_id": index,
            "upos": "NOUN",
            "xpos": "NN",
            "feats": "Number=Sing",
        } for index, char in enumerate(text)], "ok"

    def close(self):
        self.closed += 1


class PostprocessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model_root = self.root / "outputs" / "analysis_results" / "tiny"
        self.token_path = self.model_root / "token_posteriors" / "en.parquet"
        self.generated_path = self.model_root / "generated_outputs" / "en.parquet"
        self.summary_path = self.model_root / "lexical_analysis" / "pos_posterior_summary" / "en.csv"
        self.cfg = {
            "_repo_root": str(self.root),
            "outputs": {"root_dir": "outputs", "result_dir": "analysis_results"},
            "models": {"tiny": {"enabled": True}},
            "languages": ["en", "fr"],
            "lexical_analysis": {"save_pos_posterior_summary": True, "stanza_model_dir": "resources"},
        }
        self.schema = pa.schema([
            ("model", pa.string()), ("sample_id", pa.int32()), ("input_language", pa.string()),
            ("token_position", pa.int16()), ("sequence_position", pa.int16()),
            ("token", pa.string()), ("model_token", pa.string()), ("token_id", pa.int16()),
            ("char_start", pa.int32()), ("char_end", pa.int32()), ("word_text", pa.string()),
            ("sentence_id", pa.int16()), ("word_id", pa.int16()), ("upos", pa.string()),
            ("xpos", pa.string()), ("feats", pa.string()), ("pos_status", pa.string()),
            ("layer", pa.int16()), ("P_en", pa.float32()), ("P_fr", pa.float64()),
            ("audit_id", pa.uint64()),
        ], metadata={b"experiment": b"preserve this schema metadata"})
        rows = []
        for layer in (0, 1):
            for sample, char, position, posterior in ((1, "a", 0, 0.125), (2, "y", 1, 0.375)):
                rows.append({
                    "model": "tiny", "sample_id": sample, "input_language": "en",
                    "token_position": 0, "sequence_position": position,
                    "token": char, "model_token": char, "token_id": ord(char),
                    "char_start": position, "char_end": position + 1,
                    "word_text": "", "sentence_id": -1, "word_id": -1,
                    "upos": "POS_UNAVAILABLE", "xpos": "", "feats": "", "pos_status": "deferred",
                    "layer": layer, "P_en": posterior, "P_fr": 1 - posterior,
                    "audit_id": 2 ** 63 + sample,
                })
        self.original = pa.Table.from_pylist(rows, schema=self.schema)
        self.token_path.parent.mkdir(parents=True)
        with pq.ParquetWriter(self.token_path, self.schema) as writer:
            writer.write_table(self.original.slice(0, 2))
            writer.write_table(self.original.slice(2, 2))
        self.write_generated([{"sample_id": 1, "generated_text": "abc"}, {"sample_id": 2, "generated_text": "xyz"}])

    def write_generated(self, rows):
        self.generated_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows, schema=pa.schema([
            ("sample_id", pa.int32()), ("generated_text", pa.string()),
        ])), self.generated_path)

    def assert_nonannotation_equal(self, before, after):
        self.assertTrue(before.schema.equals(after.schema, check_metadata=True))
        self.assertEqual(before.num_rows, after.num_rows)
        for name in before.schema.names:
            if name not in ANNOTATION_COLUMNS:
                self.assertTrue(before[name].combine_chunks().equals(after[name].combine_chunks()), name)

    def assert_no_temporary_files(self):
        self.assertEqual(list(self.model_root.rglob("*.tmp")) + list(self.model_root.rglob("*.backup")), [])

    def test_streamed_annotation_preserves_schema_values_order_and_caches_spans(self):
        taggers = []
        def factory(cfg):
            tagger = FakeTagger(cfg)
            taggers.append(tagger)
            return tagger
        with patch("src.postprocess.align_pos_to_token", wraps=align_pos_to_token) as align:
            results = annotate_outputs(self.cfg, batch_size=1, tagger_factory=factory)
        self.assertEqual(results[0].num_rows, 4)
        self.assertEqual(results[0].num_samples, 2)
        self.assertEqual(results[1].status, "no_token_file")
        self.assertEqual(len(taggers), 1)
        self.assertEqual(taggers[0].calls, [("abc", "en"), ("xyz", "en")])
        self.assertEqual(taggers[0].closed, 1)
        self.assertEqual(taggers[0].cfg["stanza_model_dir"], str(self.root / "resources"))
        self.assertEqual(align.call_count, 2)
        annotated = pq.read_table(self.token_path)
        self.assert_nonannotation_equal(self.original, annotated)
        self.assertEqual(annotated["word_text"].to_pylist(), ["a", "y", "a", "y"])
        self.assertEqual(annotated["word_id"].to_pylist(), [0, 1, 0, 1])
        self.assertEqual(annotated["pos_status"].to_pylist(), ["ok"] * 4)
        self.assertEqual(pq.ParquetFile(self.token_path).metadata.num_row_groups, 4)
        with self.summary_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            summaries = list(reader)
            self.assertEqual(tuple(reader.fieldnames), SUMMARY_COLUMNS)
        self.assertEqual(len(summaries), 4)
        self.assertEqual([float(row["mean_posterior"]) for row in summaries], [0.25, 0.75, 0.25, 0.75])
        self.assert_no_temporary_files()

    def test_rerun_is_idempotent_and_summary_flag_is_honored(self):
        self.cfg["lexical_analysis"]["save_pos_posterior_summary"] = False
        annotate_outputs(self.cfg, tagger_factory=FakeTagger)
        once = pq.read_table(self.token_path)
        annotate_outputs(self.cfg, tagger_factory=FakeTagger)
        twice = pq.read_table(self.token_path)
        self.assertTrue(once.equals(twice, check_metadata=True))
        self.assert_nonannotation_equal(self.original, twice)
        self.assertFalse(self.summary_path.exists())

    def test_tagging_failure_leaves_source_and_existing_summary_unchanged(self):
        before = self.token_path.read_bytes()
        self.summary_path.parent.mkdir(parents=True)
        self.summary_path.write_text("existing summary")
        tagger = FakeTagger({}, fail_on="xyz")
        with self.assertRaisesRegex(RuntimeError, "tagging failed"):
            annotate_outputs(self.cfg, batch_size=1, tagger_factory=lambda cfg: tagger)
        self.assertEqual(self.token_path.read_bytes(), before)
        self.assertEqual(self.summary_path.read_text(), "existing summary")
        self.assertEqual(tagger.closed, 1)
        self.assert_no_temporary_files()

    def test_writer_failure_leaves_source_unchanged(self):
        before = self.token_path.read_bytes()
        original_writer = pq.ParquetWriter
        class FailingWriter:
            def __init__(self, *args, **kwargs):
                self.inner = original_writer(*args, **kwargs)
            def write_batch(self, batch):
                self.inner.write_batch(batch)
                raise OSError("writer failed")
            def close(self):
                self.inner.close()
        tagger = FakeTagger({})
        with patch("src.postprocess.pq.ParquetWriter", side_effect=FailingWriter):
            with self.assertRaisesRegex(OSError, "writer failed"):
                annotate_outputs(self.cfg, batch_size=1, tagger_factory=lambda cfg: tagger)
        self.assertEqual(self.token_path.read_bytes(), before)
        self.assertEqual(tagger.closed, 1)
        self.assertFalse(self.summary_path.exists())
        self.assert_no_temporary_files()

    def test_summary_replace_failure_restores_original_tokens_and_summary(self):
        before = self.token_path.read_bytes()
        os.chmod(self.token_path, 0o640)
        self.summary_path.parent.mkdir(parents=True)
        self.summary_path.write_bytes(b"existing summary\n")
        original_replace = os.replace
        def fail_summary(source, destination):
            if Path(destination) == self.summary_path:
                raise PermissionError("summary destination is read-only")
            return original_replace(source, destination)
        with patch("src.postprocess.os.replace", side_effect=fail_summary):
            with self.assertRaisesRegex(PermissionError, "summary destination is read-only"):
                annotate_outputs(self.cfg, tagger_factory=FakeTagger)
        self.assertEqual(self.token_path.read_bytes(), before)
        self.assertEqual(stat.S_IMODE(self.token_path.stat().st_mode), 0o640)
        self.assertEqual(self.summary_path.read_bytes(), b"existing summary\n")
        self.assert_no_temporary_files()

    def test_copy_backup_fallback_also_restores_after_summary_failure(self):
        before = self.token_path.read_bytes()
        self.summary_path.parent.mkdir(parents=True)
        self.summary_path.write_bytes(b"existing summary\n")
        original_replace = os.replace
        original_copy = shutil.copy2
        def fail_summary(source, destination):
            if Path(destination) == self.summary_path:
                raise PermissionError("summary destination is read-only")
            return original_replace(source, destination)
        with patch("src.postprocess.os.link", side_effect=OSError("hard links unavailable")), \
             patch("src.postprocess.shutil.copy2", wraps=original_copy) as copy, \
             patch("src.postprocess.os.replace", side_effect=fail_summary):
            with self.assertRaisesRegex(PermissionError, "summary destination is read-only"):
                annotate_outputs(self.cfg, tagger_factory=FakeTagger)
        self.assertEqual(copy.call_count, 1)
        self.assertEqual(self.token_path.read_bytes(), before)
        self.assertEqual(self.summary_path.read_bytes(), b"existing summary\n")
        self.assert_no_temporary_files()

    def test_failed_rollback_preserves_original_backup_with_recovery_error(self):
        before = self.token_path.read_bytes()
        self.summary_path.parent.mkdir(parents=True)
        self.summary_path.write_bytes(b"existing summary\n")
        original_replace = os.replace
        def fail_summary_and_rollback(source, destination):
            if Path(destination) == self.summary_path:
                raise PermissionError("summary destination is read-only")
            if Path(source).suffix == ".backup":
                raise PermissionError("rollback denied")
            return original_replace(source, destination)
        with patch("src.postprocess.os.replace", side_effect=fail_summary_and_rollback):
            with self.assertRaisesRegex(RuntimeError, "Original token file preserved at") as failure:
                annotate_outputs(self.cfg, tagger_factory=FakeTagger)
        backups = list(self.token_path.parent.glob(f".{self.token_path.name}.*.backup"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), before)
        self.assertIn(str(backups[0]), str(failure.exception))
        self.assertEqual(self.summary_path.read_bytes(), b"existing summary\n")
        self.assertEqual(list(self.model_root.rglob("*.tmp")), [])

    def test_successful_replacement_preserves_original_token_permissions(self):
        os.chmod(self.token_path, 0o640)
        annotate_outputs(self.cfg, tagger_factory=FakeTagger)
        self.assertEqual(stat.S_IMODE(self.token_path.stat().st_mode), 0o640)
        self.assertEqual(pq.read_table(self.token_path)["pos_status"].to_pylist(), ["ok"] * 4)
        self.assert_no_temporary_files()

    def test_missing_generated_sample_is_explicit_and_leaves_source_unchanged(self):
        before = self.token_path.read_bytes()
        self.write_generated([{"sample_id": 1, "generated_text": "abc"}])
        tagger = FakeTagger({})
        with self.assertRaisesRegex(ValueError, "sample_id 2"):
            annotate_outputs(self.cfg, batch_size=1, tagger_factory=lambda cfg: tagger)
        self.assertEqual(self.token_path.read_bytes(), before)
        self.assertEqual(tagger.closed, 1)
        self.assert_no_temporary_files()

    def test_empty_token_file_needs_no_generated_texts_or_tagger(self):
        empty = pa.Table.from_pylist([], schema=self.schema)
        pq.write_table(empty, self.token_path)
        self.generated_path.unlink()
        factory = Mock(side_effect=AssertionError("no tagger should be needed"))
        results = annotate_outputs(self.cfg, tagger_factory=factory)
        self.assertEqual(results[0].num_rows, 0)
        self.assertEqual(results[0].num_samples, 0)
        self.assertTrue(pq.read_table(self.token_path).equals(empty, check_metadata=True))
        self.assertEqual(self.summary_path.read_text().strip(), ",".join(SUMMARY_COLUMNS))
        factory.assert_not_called()

    def test_empty_generated_text_is_marked_without_loading_tagger(self):
        self.write_generated([{"sample_id": 1, "generated_text": ""}, {"sample_id": 2, "generated_text": ""}])
        factory = Mock(side_effect=AssertionError("no tagger should be needed"))
        annotate_outputs(self.cfg, tagger_factory=factory)
        result = pq.read_table(self.token_path)
        self.assertEqual(result["pos_status"].to_pylist(), ["empty"] * 4)
        self.assert_nonannotation_equal(self.original, result)
        factory.assert_not_called()

    def test_no_token_file_and_invalid_batch_size_are_handled(self):
        missing = self.token_path.with_name("missing.parquet")
        result = annotate_token_file(
            missing, self.generated_path, model="tiny", language="en", languages=["en"], lexical_config={},
        )
        self.assertEqual(result.status, "no_token_file")
        with self.assertRaisesRegex(ValueError, "positive integer"):
            annotate_outputs(self.cfg, batch_size=0)
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            annotate_outputs(self.cfg, models=["unknown"])

    def test_import_and_cli_do_not_load_torch_or_transformers(self):
        code = "import sys; import src.postprocess; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True, text=True)
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "annotate_outputs.py"), "--help"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        self.assertIn("--models", result.stdout)
        self.assertIn("--batch-size", result.stdout)


if __name__ == "__main__":
    unittest.main()
