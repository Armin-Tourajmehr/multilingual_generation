from __future__ import annotations

import gc
import inspect
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .token_alignment import align_generated_token_offsets


@dataclass
class LoadedModel:
    name: str
    spec: dict[str, Any]
    tokenizer: Any
    model: Any
    device: torch.device


def _dtype(value: str | None):
    if value in (None, "auto"):
        return "auto"
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {value}")
    return mapping[value]


def _resolve_device(runtime: dict[str, Any]) -> torch.device:
    requested = runtime.get("device", "cuda")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA GPU is required for the main experiment. "
            "Enable a GPU accelerator and restart the session."
        )
    if requested in {"cuda", "gpu", "auto"}:
        return torch.device("cuda:0")
    if isinstance(requested, int) or (isinstance(requested, str) and requested.isdigit()):
        index = int(requested)
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"cuda:{index} is unavailable; found {torch.cuda.device_count()} device(s).")
        return torch.device(f"cuda:{index}")
    device = torch.device(str(requested))
    if device.type != "cuda":
        raise RuntimeError("This experiment is configured for CUDA-only execution.")
    return device


def _precision_context(device: torch.device, runtime: dict[str, Any]):
    mode = str(runtime.get("mixed_precision", "fp16")).lower()
    if mode in {"none", "off", "false"}:
        return nullcontext()
    if mode == "fp16":
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    if mode == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    raise ValueError(f"Unsupported mixed_precision mode: {mode}")


def load_model(model_key: str, spec: dict[str, Any], runtime: dict[str, Any]) -> LoadedModel:
    device = _resolve_device(runtime)
    trust_remote_code = bool(runtime.get("trust_remote_code", False))

    tokenizer = AutoTokenizer.from_pretrained(
        spec["model_name"],
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            raise RuntimeError(f"Tokenizer for {model_key} has no pad/eos/unk token to use for padding.")
    # Left padding is important for decoder-only batched generation.
    tokenizer.padding_side = "left"

    kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": bool(runtime.get("low_cpu_mem_usage", True)),
    }
    dtype = _dtype(spec.get("torch_dtype", "auto"))
    # torch_dtype remains broadly compatible across Transformers releases.
    kwargs["torch_dtype"] = dtype
    if spec.get("attn_implementation") is not None:
        kwargs["attn_implementation"] = spec["attn_implementation"]

    model = AutoModelForCausalLM.from_pretrained(spec["model_name"], **kwargs)
    model.to(device)
    model.eval()

    print(f"[{model_key}] GPU: {torch.cuda.get_device_name(device.index or 0)}")
    props = torch.cuda.get_device_properties(device.index or 0)
    print(f"[{model_key}] GPU memory: {props.total_memory / (1024**3):.1f} GiB")
    model_device = next(model.parameters()).device
    if model_device.type != "cuda":
        raise RuntimeError(f"Model parameters are on {model_device}, expected CUDA.")
    param_dtypes = sorted({str(p.dtype) for p in model.parameters()})
    print(f"[{model_key}] Parameter dtype(s): {param_dtypes}")
    if spec.get("torch_dtype") == "float16" and any(p.dtype != torch.float16 for p in model.parameters()):
        print(f"[{model_key}] WARNING: not all parameters are FP16; continuing with reported dtypes.")
    print(f"[{model_key}] Model parameters device: {model_device}")
    return LoadedModel(model_key, spec, tokenizer, model, device)


def _base_model(model):
    return getattr(model, "base_model", model)


def _supports_position_ids(body) -> bool:
    """Inspect once per model; BLOOM derives ALiBi positions from its mask."""
    cached = getattr(body, "_multilingual_supports_position_ids", None)
    if cached is not None:
        return bool(cached)
    try:
        parameters = inspect.signature(body.forward).parameters
    except (TypeError, ValueError):
        supported = True
    else:
        supported = "position_ids" in parameters
        if not supported and any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            # Wrapped/custom forwards may accept the argument through **kwargs.
            # BLOOM's **deprecated_arguments does not implement position_ids.
            supported = getattr(getattr(body, "config", None), "model_type", None) != "bloom"
    body._multilingual_supports_position_ids = supported
    return supported


def _forward_hidden_states(loaded: LoadedModel, model_inputs: dict[str, torch.Tensor], runtime: dict[str, Any]):
    """Use identical, mask-relative token positions in every representation pass."""
    body = _base_model(loaded.model)
    inputs = dict(model_inputs)
    if _supports_position_ids(body):
        mask = inputs["attention_mask"]
        positions = mask.long().cumsum(dim=-1) - 1
        positions.masked_fill_(mask == 0, 0)
        inputs["position_ids"] = positions
    with _precision_context(loaded.device, runtime):
        try:
            out = body(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        except TypeError as exc:
            # Some wrapped models hide their signature. Cache the discovery so
            # subsequent batches do not pay for an unsupported-argument retry.
            if "position_ids" not in inputs or "unexpected keyword" not in str(exc) or "position_ids" not in str(exc):
                raise
            body._multilingual_supports_position_ids = False
            inputs.pop("position_ids")
            out = body(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
    return list(out.hidden_states)


def _format_inputs(loaded: LoadedModel, prompts: list[str]) -> list[str]:
    if not loaded.spec.get("use_chat_template", False):
        return prompts
    kwargs = loaded.spec.get("chat_template_kwargs") or {}
    return [
        loaded.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )
        for prompt in prompts
    ]


def _tokenize(loaded: LoadedModel, texts: list[str], max_length: int | None = None):
    tok = loaded.tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {k: v.to(loaded.device, non_blocking=True) for k, v in tok.items()}


@torch.inference_mode()
def _forward_on_device(
    loaded: LoadedModel,
    texts: list[str],
    max_tokens: int | None,
    runtime: dict[str, Any],
):
    tok = _tokenize(loaded, texts, max_tokens)
    hidden_states = _forward_hidden_states(loaded, tok, runtime)
    mask = tok["attention_mask"].bool()
    del tok
    return hidden_states, mask


def forward_text_batch(
    loaded: LoadedModel,
    texts: list[str],
    max_tokens: int | None,
    runtime: dict[str, Any],
):
    """Forward a batch of reference/input texts and return all layer hidden states + mask."""
    return _forward_on_device(loaded, texts, max_tokens, runtime)


@torch.inference_mode()
def forward_texts_with_offsets(
    loaded: LoadedModel,
    texts: list[str],
    max_tokens: int | None,
    runtime: dict[str, Any],
) -> tuple[list[torch.Tensor], list[list[int]], list[list[tuple[int, int]]], list[list[int]]]:
    """Forward generated texts and return hidden states plus tokenizer character offsets.

    The text is tokenized again as an ordinary input sequence, so each hidden-state
    vector corresponds to the token represented at that position, rather than to
    the previous token's prediction state captured during generation.
    """
    try:
        enc = loaded.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_tokens,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
    except (NotImplementedError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Tokenizer for model {loaded.name} must support fast-tokenizer offset_mapping "
            "to align model tokens with Stanza POS tags."
        ) from exc

    offsets_raw = enc.pop("offset_mapping").cpu().tolist()
    enc.pop("special_tokens_mask", None)
    input_ids = enc["input_ids"].cpu().tolist()
    attention_mask = enc["attention_mask"].cpu().tolist()
    offsets = [[(int(a), int(b)) for a, b in row] for row in offsets_raw]
    model_inputs = {k: v.to(loaded.device, non_blocking=True) for k, v in enc.items()}
    hidden_states = _forward_hidden_states(loaded, model_inputs, runtime)
    del enc, model_inputs
    return hidden_states, input_ids, offsets, attention_mask


@torch.inference_mode()
def _generate_on_device(
    loaded: LoadedModel,
    prompts: list[str],
    generation: dict[str, Any],
    runtime: dict[str, Any],
) -> tuple[list[list[int]], list[str], int, list[list[int]]]:
    """Generate text without storing every layer's decode-time hidden states.

    The analysis stage replays the generated IDs to extract contextual states.
    """
    if not prompts:
        raise ValueError("Generation received an empty prompt batch.")
    rendered = _format_inputs(loaded, prompts)
    tok = _tokenize(loaded, rendered, runtime.get("max_input_tokens") or None)
    input_width = int(tok["input_ids"].shape[1])
    tokenized_prompt_ids = tok["input_ids"].detach().cpu().tolist()
    prompt_attention = tok["attention_mask"].detach().cpu().tolist()
    prompt_token_ids = [
        [int(token_id) for token_id, keep in zip(row, mask) if int(keep) == 1]
        for row, mask in zip(tokenized_prompt_ids, prompt_attention)
    ]

    do_sample = bool(generation.get("do_sample", True))
    kwargs: dict[str, Any] = {
        "max_new_tokens": int(generation["max_new_tokens"]),
        "do_sample": do_sample,
        "pad_token_id": loaded.tokenizer.pad_token_id,
        "use_cache": True,
    }
    min_new_tokens = generation.get("min_new_tokens")

    if min_new_tokens is not None:
        min_new_tokens = int(min_new_tokens)

        if not 0 <= min_new_tokens <= int(generation["max_new_tokens"]):
            raise ValueError(
                "min_new_tokens must be between 0 and max_new_tokens."
            )

    kwargs["min_new_tokens"] = min_new_tokens
    if do_sample:
        kwargs["temperature"] = float(generation.get("temperature", 0.7))
        kwargs["top_p"] = float(generation.get("top_p", 0.9))

    with _precision_context(loaded.device, runtime):
        generated = loaded.model.generate(**tok, **kwargs)

    generated_ids = generated[:, input_width:].detach().cpu().tolist()
    # Generation may stop on several different EOS IDs (e.g. Llama's end-of-turn
    # marker). Retain the sampled stop token and discard its batch-padding tail.
    eos = getattr(
        getattr(loaded.model, "generation_config", None),
        "eos_token_id",
        getattr(loaded.tokenizer, "eos_token_id", None),
    )
    eos_ids = {eos} if isinstance(eos, int) else set(eos or ())
    if eos_ids:
        for i, ids in enumerate(generated_ids):
            for pos, token_id in enumerate(ids):
                if token_id in eos_ids:
                    generated_ids[i] = ids[:pos + 1]
                    break
    texts = loaded.tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    del generated, tok, rendered, tokenized_prompt_ids, prompt_attention
    return generated_ids, texts, input_width, prompt_token_ids


@torch.inference_mode()
def replay_prompt_and_continuation(
    loaded: LoadedModel,
    prompt_token_ids: list[list[int]],
    generated_ids: list[list[int]],
    runtime: dict[str, Any],
) -> tuple[list[torch.Tensor], list[int]]:
    """Return all contextual layers and each joined sequence's left padding.

    For sample i, continuation token j is consumed at position
    left_padding[i] + len(prompt_token_ids[i]) + j. Its selection state is at
    the preceding position when that position belongs to the same sequence.
    """
    if len(prompt_token_ids) != len(generated_ids):
        raise ValueError("Prompt IDs and generated IDs must have equal batch lengths.")
    sequences = [list(prompt) + list(ids) for prompt, ids in zip(prompt_token_ids, generated_ids)]

    batch_size = len(sequences)
    max_sequence_len = max((len(seq) for seq in sequences), default=0)
    pad_token_id = loaded.tokenizer.pad_token_id
    if pad_token_id is None:
        raise RuntimeError(f"Tokenizer for {loaded.name} needs a pad_token_id for batched context forwarding.")

    sequence_lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    left_padding = max_sequence_len - sequence_lengths
    ids_tensor = torch.full((batch_size, max_sequence_len), int(pad_token_id), dtype=torch.long)
    # Scatter all variable-length sequences at once rather than assigning one
    # padded row at a time.
    flat_ids = torch.tensor([token for sequence in sequences for token in sequence], dtype=torch.long)
    sample_indices = torch.arange(batch_size).repeat_interleave(sequence_lengths)
    sequence_starts = (sequence_lengths.cumsum(0) - sequence_lengths).repeat_interleave(sequence_lengths)
    token_columns = torch.arange(flat_ids.numel()) - sequence_starts + left_padding[sample_indices]
    ids_tensor[sample_indices, token_columns] = flat_ids
    mask_tensor = (torch.arange(max_sequence_len)[None, :] >= left_padding[:, None]).long()

    model_inputs = {
        "input_ids": ids_tensor.to(loaded.device, non_blocking=True),
        "attention_mask": mask_tensor.to(loaded.device, non_blocking=True),
    }
    return _forward_hidden_states(loaded, model_inputs, runtime), left_padding.tolist()


@torch.inference_mode()
def forward_generated_texts_with_context(
    loaded: LoadedModel,
    prompt_token_ids: list[list[int]],
    generated_ids: list[list[int]],
    generated_texts: list[str],
    runtime: dict[str, Any],
) -> tuple[list[torch.Tensor], list[list[int]], list[list[tuple[int, int]]], list[list[int]]]:
    """Replay original prompt and continuation IDs, then align text offsets separately.

    The returned hidden states correspond to the positions after consuming each
    generated token, including special tokens. A token without a reliable text
    span still has its original position and hidden state.
    """
    if not (len(prompt_token_ids) == len(generated_ids) == len(generated_texts)):
        raise ValueError("Prompt IDs, generated IDs and generated texts must have equal batch lengths.")

    output_token_ids = [list(ids) for ids in generated_ids]
    output_attention = [[1] * len(ids) for ids in output_token_ids]
    full_hidden_states, _ = replay_prompt_and_continuation(loaded, prompt_token_ids, output_token_ids, runtime)
    max_sequence_len = max((len(prompt) + len(ids) for prompt, ids in zip(prompt_token_ids, output_token_ids)), default=0)

    max_output_len = max((len(ids) for ids in output_token_ids), default=0)
    output_lengths = torch.tensor([len(ids) for ids in output_token_ids], device=loaded.device)
    output_positions = torch.arange(max_output_len, device=loaded.device)[None, :]
    valid_output = output_positions < output_lengths[:, None]
    # Continuations end at the right edge of every left-padded full sequence.
    gather_positions = max_sequence_len - output_lengths[:, None] + output_positions
    gather_positions = gather_positions.clamp(max=max(max_sequence_len - 1, 0))
    output_hidden_states: list[torch.Tensor] = []
    for full_hidden in full_hidden_states:
        output_hidden = full_hidden.gather(
            1, gather_positions[:, :, None].expand(-1, -1, full_hidden.shape[-1])
        )
        output_hidden.masked_fill_(~valid_output[:, :, None], 0)
        output_hidden_states.append(output_hidden)

    del full_hidden_states
    output_offsets = align_generated_token_offsets(loaded.tokenizer, output_token_ids, generated_texts)
    return output_hidden_states, output_token_ids, output_offsets, output_attention


def generate_batch(
    loaded: LoadedModel,
    prompts: list[str],
    generation: dict[str, Any],
    runtime: dict[str, Any],
):
    return _generate_on_device(loaded, prompts, generation, runtime)


def release_model(loaded: LoadedModel) -> None:
    """Drop model/tokenizer references and clear PyTorch's unused CUDA cache."""
    model = getattr(loaded, "model", None)
    tokenizer = getattr(loaded, "tokenizer", None)
    loaded.model = None
    loaded.tokenizer = None
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
