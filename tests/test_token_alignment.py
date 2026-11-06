"""Offline checks for aligning the original generated token stream."""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from src.token_alignment import NO_OFFSET, align_generated_token_offsets, tokenizer_special_ids


class TokenAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
        vocab = {token: number for number, token in enumerate(alphabet)}
        vocab["ab"] = len(vocab)
        backend = Tokenizer(models.BPE(vocab, [("a", "b")]))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        cls.tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<end>")

    def test_fast_offsets_are_used_when_original_ids_match(self):
        tokenizer = self.tokenizer
        ids = tokenizer.encode("abéc", add_special_tokens=False) + [tokenizer.eos_token_id]
        with patch.object(tokenizer, "decode", side_effect=AssertionError("matching IDs need no prefix decoding")):
            spans = align_generated_token_offsets(tokenizer, [ids], ["abéc"])
        # The fast tokenizer knows both byte tokens belong to the same character.
        self.assertEqual(spans, [[(0, 2), (2, 3), (2, 3), (3, 4), NO_OFFSET]])

    def test_mismatched_bpe_and_partial_unicode_keep_conservative_original_spans(self):
        tokenizer = self.tokenizer
        byte_ids = tokenizer.encode("é", add_special_tokens=False)
        self.assertEqual(len(byte_ids), 2)
        first, second = tokenizer.convert_tokens_to_ids(["a", "b"])
        tail = tokenizer.convert_tokens_to_ids(["c", "d"])
        ids = [first, tokenizer.eos_token_id, second, *byte_ids, *tail, tokenizer.eos_token_id]
        self.assertEqual(tokenizer.decode(ids[:4], skip_special_tokens=True), "ab\ufffd")
        self.assertEqual(tokenizer.decode(ids, skip_special_tokens=True), "abécd")
        self.assertNotEqual(tokenizer.encode("abécd", add_special_tokens=False), ids)

        spans = align_generated_token_offsets(tokenizer, [[], ids], ["", "abécd"])
        self.assertEqual(spans, [[], [(0, 1), NO_OFFSET, (1, 2), NO_OFFSET, NO_OFFSET, (3, 4), (4, 5), NO_OFFSET]])

    def test_zero_length_partial_decode_never_assigns_combined_character_to_last_byte(self):
        class SlowTokenizer:
            is_fast = False

            def decode(self, ids, **kwargs):
                pieces = {1: b"a", 2: b"\xc3", 3: b"\xa9", 4: b"b"}
                return b"".join(pieces[token] for token in ids).decode("utf-8", errors="ignore")

        spans = align_generated_token_offsets(SlowTokenizer(), [[1, 2, 3, 4]], ["aéb"])
        self.assertEqual(spans, [[(0, 1), NO_OFFSET, NO_OFFSET, (2, 3)]])

    def test_added_special_ids_are_excluded_and_ordinary_added_ids_are_kept(self):
        class FastTokenizer:
            all_special_ids = [90]
            added_tokens_decoder = {91: SimpleNamespace(special=True), 92: SimpleNamespace(special=False)}

            def __call__(self, texts, **kwargs):
                return {"input_ids": [[1, 2, 92]], "offset_mapping": [[(0, 1), (1, 2), (2, 3)]]}

        tokenizer = FastTokenizer()
        self.assertEqual(tokenizer_special_ids(tokenizer), {90, 91})
        self.assertEqual(
            align_generated_token_offsets(tokenizer, [[90, 1, 91, 2, 92]], ["abc"]),
            [[NO_OFFSET, (0, 1), NO_OFFSET, (1, 2), (2, 3)]],
        )

    def test_empty_and_special_only_continuations_need_no_tokenizer_work(self):
        class UnusedTokenizer:
            all_special_ids = [90]

            def __call__(self, texts, **kwargs):
                raise AssertionError("no visible text to align")

            def decode(self, ids, **kwargs):
                raise AssertionError("no visible text to decode")

        tokenizer = UnusedTokenizer()
        self.assertEqual(align_generated_token_offsets(tokenizer, [], []), [])
        self.assertEqual(align_generated_token_offsets(tokenizer, [[], [90], [90, 90]], ["", "", ""]), [[], [NO_OFFSET], [NO_OFFSET, NO_OFFSET]])
        with self.assertRaisesRegex(ValueError, "equal batch lengths"):
            align_generated_token_offsets(tokenizer, [[90]], [])

    def test_incompatible_decoded_text_leaves_ids_unaligned(self):
        ids = self.tokenizer.convert_tokens_to_ids(["a", "b"])
        self.assertEqual(align_generated_token_offsets(self.tokenizer, [ids], ["xy"]), [[NO_OFFSET, NO_OFFSET]])

    def test_long_mismatch_bounds_prefix_decoding_without_dropping_ids(self):
        class SlowTokenizer:
            is_fast = False

            def __init__(self):
                self.decode_calls = 0

            def decode(self, ids, **kwargs):
                self.decode_calls += 1
                return "a" * len(ids)

        tokenizer = SlowTokenizer()
        spans, = align_generated_token_offsets(tokenizer, [[1] * 300], ["a" * 300])
        self.assertEqual(len(spans), 300)
        self.assertEqual(spans[255], (255, 256))
        self.assertEqual(spans[256:], [NO_OFFSET] * 44)
        self.assertLessEqual(tokenizer.decode_calls, 257)


if __name__ == "__main__":
    unittest.main()
