
from __future__ import annotations

import gc
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


def _dtype(value: str, device: torch.device):
    if value in (None, "auto"):
        return "auto"
    mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if value not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {value}")
    return mapping[value]


def load_model(model_key: str, spec: dict[str, Any], runtime: dict[str, Any]) -> LoadedModel:
    device = torch.device("cuda" if runtime.get("device", "auto") == "auto" and torch.cuda.is_available() else runtime.get("device", "cpu"))
    if device.type == "cuda":
        device_map = runtime.get("device_map", "auto")
    else:
        device_map = None
    tokenizer = AutoTokenizer.from_pretrained(spec["model_name"], trust_remote_code=runtime.get("trust_remote_code", False))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left padding is important for batched generation with decoder-only LMs.
    tokenizer.padding_side = "left"

    kwargs = {
        "trust_remote_code": runtime.get("trust_remote_code", False),
        "low_cpu_mem_usage": runtime.get("low_cpu_mem_usage", True),
    }
    dtype = _dtype(spec.get("torch_dtype", "auto"), device)
    if dtype != "auto":
        kwargs["dtype"] = dtype
    else:
        kwargs["dtype"] = "auto"
    if device_map is not None:
        kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(spec["model_name"], **kwargs)
    if device_map is None:
        model.to(device)
    model.eval()
    return LoadedModel(model_key, spec, tokenizer, model, device)


def _format_inputs(loaded: LoadedModel, prompts: list[str], runtime: dict[str, Any]) -> tuple[list[str], bool]:
    if not loaded.spec.get("use_chat_template", False):
        return prompts, False
    rendered = []
    for prompt in prompts:
        rendered.append(
            loaded.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                **(loaded.spec.get("chat_template_kwargs") or {}),
            )
        )
    return rendered, True


def generate_batch(loaded: LoadedModel, prompts: list[str], generation: dict[str, Any], runtime: dict[str, Any]):
    rendered, _ = _format_inputs(loaded, prompts, runtime)
    tok = loaded.tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=runtime.get("max_input_tokens") or None,
    )
    tok = {k: v.to(loaded.device) for k, v in tok.items()}
    input_width = tok["input_ids"].shape[1]
    with torch.no_grad():
        generated = loaded.model.generate(
            **tok,
            max_new_tokens=int(generation["max_new_tokens"]),
            do_sample=bool(generation.get("do_sample", True)),
            temperature=float(generation["temperature"]) if generation.get("do_sample", True) else None,
            top_p=float(generation["top_p"]) if generation.get("do_sample", True) else None,
            pad_token_id=loaded.tokenizer.pad_token_id,
        )
    # With left padding, every sequence in the batch has the same padded prompt width.
    # Generated tokens therefore start at input_width for every sample.
    generated_token_ids = []
    eos_ids = set()
    eos = loaded.tokenizer.eos_token_id
    if eos is not None:
        eos_ids.add(int(eos))
    pad = loaded.tokenizer.pad_token_id
    if pad is not None and pad != eos:
        # A distinct pad token can safely be excluded if generation returned one.
        pass
    for seq in generated:
        ids = seq[int(input_width):].tolist()
        if eos_ids:
            try:
                end = next(i for i, token_id in enumerate(ids) if int(token_id) in eos_ids)
                ids = ids[: end + 1]
            except StopIteration:
                pass
        generated_token_ids.append(ids)
    texts = [loaded.tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_token_ids]

    # Forward the complete generated sequences to obtain hidden states for the generated tokens.
    with torch.no_grad():
        full_out = loaded.model(generated, output_hidden_states=True, return_dict=True)
    hidden_states = full_out.hidden_states
    per_sample_hidden = []
    for i, ids in enumerate(generated_token_ids):
        per_sample_hidden.append([h[i, int(input_width): int(input_width) + len(ids), :].detach().float().cpu().numpy() for h in hidden_states])
    return generated_token_ids, texts, per_sample_hidden


def forward_text_batch(loaded: LoadedModel, texts: list[str], max_tokens: int):
    tok = loaded.tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_tokens,
    )
    tok = {k: v.to(loaded.device) for k, v in tok.items()}
    with torch.no_grad():
        out = loaded.model(**tok, output_hidden_states=True, return_dict=True)
    mask = tok["attention_mask"].detach().cpu().numpy().astype(bool)
    hidden_states = out.hidden_states
    result = []
    for layer_hidden in hidden_states:
        result.append(layer_hidden.detach().float().cpu().numpy())
    return result, mask


def release_model(loaded: LoadedModel) -> None:
    del loaded.model
    del loaded.tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
