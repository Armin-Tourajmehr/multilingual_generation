"""Offline representation alignment checks; no pretrained downloads or GPU needed."""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import (
    BloomConfig, BloomForCausalLM, GPT2Config, GPT2LMHeadModel,
    LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM,
)

from src.modeling import (
    LoadedModel,
    forward_generated_texts_with_context,
    forward_text_batch,
    forward_texts_with_offsets,
    load_model,
)


class CharacterTokenizer:
    """Small left-padding tokenizer with exact, transparent character offsets."""

    pad_token_id = 0
    all_special_ids = [0, 31]
    padding_side = "left"

    def __call__(self, texts, **kwargs):
        limit = kwargs.get("max_length") if kwargs.get("truncation") else None
        sequences = [[ord(char) - ord("a") + 1 for char in text[:limit]] for text in texts]
        width = max((len(ids) for ids in sequences), default=0)
        ids = torch.zeros((len(texts), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        offsets = torch.zeros((len(texts), width, 2), dtype=torch.long)
        for i, sequence in enumerate(sequences):
            if sequence:
                start = width - len(sequence)
                ids[i, start:] = torch.tensor(sequence)
                mask[i, start:] = 1
                offsets[i, start:] = torch.tensor([(j, j + 1) for j in range(len(sequence))])
        result = {"input_ids": ids, "attention_mask": mask}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = offsets
        if kwargs.get("return_special_tokens_mask"):
            result["special_tokens_mask"] = (mask == 0).long()
        return result


class ModelingTests(unittest.TestCase):
    runtime = {"mixed_precision": "none"}

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        cls.gpt = GPT2LMHeadModel(GPT2Config(
            n_layer=1, n_head=2, n_embd=16, vocab_size=32, n_positions=64,
            resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
        )).eval()
        cls.bloom = BloomForCausalLM(BloomConfig(
            n_layer=1, n_head=2, hidden_size=16, vocab_size=32,
            hidden_dropout=0, attention_dropout=0,
        )).eval()
        cls.llama = LlamaForCausalLM(LlamaConfig(
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            hidden_size=16, intermediate_size=32, vocab_size=32, max_position_embeddings=64,
        )).eval()
        cls.qwen = Qwen3ForCausalLM(Qwen3Config(
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            hidden_size=16, intermediate_size=32, vocab_size=32, max_position_embeddings=64,
        )).eval()
        cls.models = (cls.gpt, cls.bloom, cls.llama, cls.qwen)

    def loaded(self, model=None):
        return LoadedModel("offline", {}, CharacterTokenizer(), model if model is not None else self.gpt, torch.device("cpu"))

    def test_raw_forward_is_padding_invariant(self):
        for model in self.models:
            with self.subTest(model=model.config.model_type):
                loaded = self.loaded(model)
                alone, alone_mask = forward_text_batch(loaded, ["ab"], None, self.runtime)
                batched, batch_mask = forward_text_batch(loaded, ["ab", "cdefg"], None, self.runtime)
                for single_layer, batch_layer in zip(alone, batched):
                    torch.testing.assert_close(
                        single_layer[0, alone_mask[0]], batch_layer[0, batch_mask[0]], atol=2e-6, rtol=2e-6,
                    )

    def test_offset_forward_is_padding_invariant(self):
        loaded = self.loaded()
        alone, _, _, alone_mask = forward_texts_with_offsets(loaded, ["ab"], None, self.runtime)
        batched, _, offsets, batch_mask = forward_texts_with_offsets(loaded, ["ab", "cdefg"], None, self.runtime)
        self.assertEqual(offsets[0][-2:], [(0, 1), (1, 2)])
        for single_layer, batch_layer in zip(alone, batched):
            torch.testing.assert_close(
                single_layer[0, torch.tensor(alone_mask[0]).bool()],
                batch_layer[0, torch.tensor(batch_mask[0]).bool()], atol=2e-6, rtol=2e-6,
            )

    def test_replay_gathers_only_output_positions_and_zeros_padding(self):
        prompts = [[1, 2], [3], [4, 5, 6]]
        outputs = [[7, 8, 9], [], [10]]
        for model in self.models:
            with self.subTest(model=model.config.model_type):
                loaded = self.loaded(model)
                actual, ids, offsets, attention = forward_generated_texts_with_context(
                    loaded, prompts, outputs, ["ghi", "", "j"], self.runtime,
                )
                self.assertEqual(ids, outputs)
                self.assertEqual(offsets, [[(0, 1), (1, 2), (2, 3)], [], [(0, 1)]])
                self.assertEqual(attention, [[1, 1, 1], [], [1]])
                for sample, (prompt, output) in enumerate(zip(prompts, outputs)):
                    full_ids = torch.tensor([prompt + output])
                    kwargs = {"input_ids": full_ids, "attention_mask": torch.ones_like(full_ids)}
                    if model.config.model_type == "gpt2":
                        kwargs["position_ids"] = torch.arange(full_ids.shape[1])[None, :]
                    with torch.inference_mode():
                        expected = model.base_model(**kwargs, output_hidden_states=True, use_cache=False).hidden_states
                    for actual_layer, expected_layer in zip(actual, expected):
                        torch.testing.assert_close(
                            actual_layer[sample, :len(output)], expected_layer[0, len(prompt):], atol=2e-6, rtol=2e-6,
                        )
                        self.assertTrue(torch.equal(
                            actual_layer[sample, len(output):], torch.zeros_like(actual_layer[sample, len(output):]),
                        ))

    def test_replay_all_empty_continuations_has_zero_output_width(self):
        hidden, ids, offsets, attention = forward_generated_texts_with_context(
            self.loaded(), [[1], [2, 3]], [[], []], ["", ""], self.runtime,
        )
        self.assertEqual(ids, [[], []])
        self.assertEqual(offsets, [[], []])
        self.assertEqual(attention, [[], []])
        self.assertTrue(all(layer.shape == (2, 0, 16) for layer in hidden))

    def test_bloom_does_not_receive_unsupported_position_ids(self):
        recorded = []
        def capture(module, args, kwargs):
            recorded.append(set(kwargs))
        hook = self.bloom.base_model.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            loaded = self.loaded(self.bloom)
            forward_text_batch(loaded, ["ab", "cdefg"], None, self.runtime)
            forward_generated_texts_with_context(loaded, [[1]], [[2]], ["b"], self.runtime)
        finally:
            hook.remove()
        self.assertEqual(len(recorded), 2)
        self.assertTrue(all("position_ids" not in names for names in recorded))

    def test_wrapped_model_position_fallback_is_cached(self):
        class WrappedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(model_type="custom")
                self.rejected_positions = 0

            def forward(self, input_ids, attention_mask, **kwargs):
                if "position_ids" in kwargs:
                    self.rejected_positions += 1
                    raise TypeError("forward() got an unexpected keyword argument 'position_ids'")
                return SimpleNamespace(hidden_states=(input_ids.float().unsqueeze(-1),))

        model = WrappedModel()
        loaded = self.loaded(model)
        forward_text_batch(loaded, ["ab"], None, self.runtime)
        forward_text_batch(loaded, ["ab"], None, self.runtime)
        self.assertEqual(model.rejected_positions, 1)

    def test_attention_implementation_is_only_passed_when_requested(self):
        for implementation in (None, "sdpa"):
            with self.subTest(implementation=implementation):
                spec = {"model_name": "offline", "torch_dtype": "float16"}
                if implementation is not None:
                    spec["attn_implementation"] = implementation
                tokenizer = SimpleNamespace(pad_token="<pad>", padding_side="right")
                with patch("src.modeling._resolve_device", return_value=torch.device("cuda")), \
                     patch("src.modeling.AutoTokenizer.from_pretrained", return_value=tokenizer), \
                     patch("src.modeling.AutoModelForCausalLM.from_pretrained", side_effect=RuntimeError("stop before loading")) as load:
                    with self.assertRaisesRegex(RuntimeError, "stop before loading"):
                        load_model("offline", spec, self.runtime)
                if implementation is None:
                    self.assertNotIn("attn_implementation", load.call_args.kwargs)
                else:
                    self.assertEqual(load.call_args.kwargs["attn_implementation"], "sdpa")


if __name__ == "__main__":
    unittest.main()
