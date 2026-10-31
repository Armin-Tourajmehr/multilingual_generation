from __future__ import annotations

import gc
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    body = _base_model(loaded.model)
    with _precision_context(loaded.device, runtime):
        out = body(**tok, output_hidden_states=True, return_dict=True, use_cache=False)
    hidden_states = list(out.hidden_states)
    mask = tok["attention_mask"].bool()
    del out, tok
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
    with _precision_context(loaded.device, runtime):
        out = _base_model(loaded.model)(
            **model_inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
    hidden_states = list(out.hidden_states)
    del out, enc, model_inputs
    return hidden_states, input_ids, offsets, attention_mask


@torch.inference_mode()
def _generate_on_device(
    loaded: LoadedModel,
    prompts: list[str],
    generation: dict[str, Any],
    runtime: dict[str, Any],
) -> tuple[list[list[int]], list[str], int, list[list[int]]]:
    """Generate text without storing every layer's decode-time hidden states.

    Generated texts are forwarded separately by the analysis stage. This avoids
    retaining all per-step generation hidden states and fixes token/state alignment.
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
    eos = loaded.tokenizer.eos_token_id
    if eos is not None:
        eos = int(eos)
        for i, ids in enumerate(generated_ids):
            for pos, token_id in enumerate(ids):
                if int(token_id) == eos:
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
def forward_generated_texts_with_context(
    loaded: LoadedModel,
    prompt_token_ids: list[list[int]],
    generated_ids: list[list[int]],
    generated_texts: list[str],
    runtime: dict[str, Any],
) -> tuple[list[torch.Tensor], list[list[int]], list[list[tuple[int, int]]], list[list[int]]]:
    """Forward decoded generated text appended to the actual tokenized prompt.

    This preserves prompt context for each output token. The output text is encoded
    without automatically-added BOS/EOS tokens to obtain character offsets; those
    token ids are appended to the actual prompt token ids, and the hidden states at
    those output positions are returned in a compact output-only tensor. If decode /
    re-encode changes a token sequence, the function warns but continues with the
    re-encoded surface text, which is what the POS offsets describe.
    """
    if not (len(prompt_token_ids) == len(generated_ids) == len(generated_texts)):
        raise ValueError("Prompt IDs, generated IDs and generated texts must have equal batch lengths.")

    try:
        enc = loaded.tokenizer(
            generated_texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (NotImplementedError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Tokenizer for model {loaded.name} must support fast-tokenizer offset_mapping "
            "to align model tokens with Stanza POS tags."
        ) from exc

    encoded_ids_all = enc["input_ids"].detach().cpu().tolist()
    encoded_mask_all = enc["attention_mask"].detach().cpu().tolist()
    offset_all = enc.pop("offset_mapping").detach().cpu().tolist()
    output_token_ids: list[list[int]] = []
    output_offsets: list[list[tuple[int, int]]] = []
    output_attention: list[list[int]] = []
    sequences: list[list[int]] = []

    all_special_ids = set(int(value) for value in (getattr(loaded.tokenizer, "all_special_ids", None) or []))
    mismatched_samples = 0
    for i, (prompt_ids, gen_ids, text) in enumerate(zip(prompt_token_ids, generated_ids, generated_texts)):
        keep_positions = [j for j, keep in enumerate(encoded_mask_all[i]) if int(keep) == 1]
        ids = [int(encoded_ids_all[i][j]) for j in keep_positions]
        offsets = [(int(offset_all[i][j][0]), int(offset_all[i][j][1])) for j in keep_positions]

        # Compare the surface-text round trip with the generated token stream, excluding
        # special markers that decode(skip_special_tokens=True) intentionally removed.
        generated_surface_ids = [int(t) for t in gen_ids if int(t) not in all_special_ids]
        if generated_surface_ids != ids:
            mismatched_samples += 1

        output_token_ids.append(ids)
        output_offsets.append(offsets)
        output_attention.append([1] * len(ids))
        sequences.append([int(t) for t in prompt_ids] + ids)

    if mismatched_samples:
        print(
            f"[{loaded.name}] WARNING: decoded/re-encoded output token IDs differed for "
            f"{mismatched_samples}/{len(generated_texts)} sample(s); using offsets/tokens "
            "from the decoded text and forwarding that surface form with the original prompt IDs."
        )

    batch_size = len(sequences)
    max_sequence_len = max((len(seq) for seq in sequences), default=0)
    pad_token_id = loaded.tokenizer.pad_token_id
    if pad_token_id is None:
        raise RuntimeError(f"Tokenizer for {loaded.name} needs a pad_token_id for batched context forwarding.")

    ids_tensor = torch.full((batch_size, max_sequence_len), int(pad_token_id), dtype=torch.long)
    mask_tensor = torch.zeros((batch_size, max_sequence_len), dtype=torch.long)
    position_tensor = torch.zeros((batch_size, max_sequence_len), dtype=torch.long)
    left_padding: list[int] = []
    for i, sequence in enumerate(sequences):
        pad_len = max_sequence_len - len(sequence)
        left_padding.append(pad_len)
        if sequence:
            ids_tensor[i, pad_len:] = torch.tensor(sequence, dtype=torch.long)
            mask_tensor[i, pad_len:] = 1
            position_tensor[i, pad_len:] = torch.arange(len(sequence), dtype=torch.long)

    model_inputs = {
        "input_ids": ids_tensor.to(loaded.device, non_blocking=True),
        "attention_mask": mask_tensor.to(loaded.device, non_blocking=True),
        "position_ids": position_tensor.to(loaded.device, non_blocking=True),
    }
    body = _base_model(loaded.model)
    with _precision_context(loaded.device, runtime):
        try:
            out = body(
                **model_inputs,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        except TypeError as exc:
            # Some architectures (notably ALiBi-based BLOOM) do not accept explicit
            # position_ids. Relative-position models can omit them safely.
            if "position_ids" not in str(exc):
                raise
            model_inputs.pop("position_ids", None)
            out = body(
                **model_inputs,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
    full_hidden_states = list(out.hidden_states)

    max_output_len = max((len(ids) for ids in output_token_ids), default=0)
    output_hidden_states: list[torch.Tensor] = []
    for full_hidden in full_hidden_states:
        output_hidden = torch.zeros(
            (batch_size, max_output_len, full_hidden.shape[-1]),
            device=full_hidden.device,
            dtype=full_hidden.dtype,
        )
        for i, (prompt_ids, out_ids) in enumerate(zip(prompt_token_ids, output_token_ids)):
            if not out_ids:
                continue
            start = left_padding[i] + len(prompt_ids)
            end = start + len(out_ids)
            output_hidden[i, :len(out_ids), :] = full_hidden[i, start:end, :]
        output_hidden_states.append(output_hidden)

    del out, full_hidden_states, model_inputs, ids_tensor, mask_tensor, position_tensor
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
