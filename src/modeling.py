from __future__ import annotations

import gc
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
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
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {value}")
    return mapping[value]


def _resolve_device(runtime: dict[str, Any]) -> torch.device:
    requested = runtime.get("device", "auto")
    require_gpu = bool(runtime.get("require_gpu", False))
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU is required for this experiment, but CUDA is unavailable. "
            "Enable a CUDA GPU in Kaggle or run on a CUDA-enabled machine."
        )
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" or requested == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but CUDA is unavailable.")
        return torch.device("cuda:0")
    if isinstance(requested, int) or (isinstance(requested, str) and requested.isdigit()):
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA device was requested, but CUDA is unavailable.")
        index = int(requested)
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"cuda:{index} is unavailable. Found {torch.cuda.device_count()} device(s).")
        return torch.device(f"cuda:{index}")
    return torch.device(str(requested))


def _precision_context(device: torch.device, runtime: dict[str, Any]):
    mode = str(runtime.get("mixed_precision", "fp16")).lower()
    if device.type != "cuda" or mode in {"none", "off", "false"}:
        return nullcontext()
    if mode == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if mode == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise ValueError(f"Unsupported mixed_precision mode: {mode}")


def load_model(model_key: str, spec: dict[str, Any], runtime: dict[str, Any]) -> LoadedModel:
    device = _resolve_device(runtime)
    tokenizer = AutoTokenizer.from_pretrained(
        spec["model_name"],
        trust_remote_code=runtime.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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

    model = AutoModelForCausalLM.from_pretrained(spec["model_name"], **kwargs)
    model.to(device)
    model.eval()

    if device.type == "cuda":
        print(f"[{model_key}] CUDA available: True")
        print(f"[{model_key}] GPU: {torch.cuda.get_device_name(device.index or 0)}")
        print(f"[{model_key}] GPU memory: {torch.cuda.get_device_properties(device.index or 0).total_memory / (1024**3):.1f} GiB")
        model_device = next(model.parameters()).device
        if model_device.type != "cuda":
            raise RuntimeError(f"Model parameters are on {model_device}, expected CUDA.")
    else:
        if runtime.get("require_gpu", False):
            raise RuntimeError("GPU was required but the loaded model is on CPU.")
        print(f"[{model_key}] Using one inference device: {device}")

    print(f"[{model_key}] Model parameters device: {next(model.parameters()).device}")
    return LoadedModel(model_key, spec, tokenizer, model, device)


def _base_model(model):
    """Return the transformer body to avoid allocating CausalLM logits."""
    return getattr(model, "base_model", model)


def _format_inputs(loaded: LoadedModel, prompts: list[str], runtime: dict[str, Any]) -> list[str]:
    if not loaded.spec.get("use_chat_template", False):
        return prompts
    return [
        loaded.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **(loaded.spec.get("chat_template_kwargs") or {}),
        )
        for prompt in prompts
    ]


def _generate_on_device(
    loaded: LoadedModel,
    prompts: list[str],
    generation: dict[str, Any],
    runtime: dict[str, Any],
):
    device = loaded.device
    rendered = _format_inputs(loaded, prompts, runtime)
    tok = loaded.tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=runtime.get("max_input_tokens") or None,
    )
    tok = {k: v.to(device, non_blocking=True) for k, v in tok.items()}
    input_width = int(tok["input_ids"].shape[1])

    with torch.inference_mode(), _precision_context(device, runtime):
        generated = loaded.model.generate(
            **tok,
            max_new_tokens=int(generation["max_new_tokens"]),
            do_sample=bool(generation.get("do_sample", True)),
            temperature=float(generation["temperature"]) if generation.get("do_sample", True) else None,
            top_p=float(generation["top_p"]) if generation.get("do_sample", True) else None,
            pad_token_id=loaded.tokenizer.pad_token_id,
            use_cache=True,
        )

    eos_ids = set()
    eos = loaded.tokenizer.eos_token_id
    if eos is not None:
        eos_ids.add(int(eos))
    generated_token_ids = []
    for seq in generated:
        ids = seq[input_width:].tolist()
        if eos_ids:
            for pos, token_id in enumerate(ids):
                if int(token_id) in eos_ids:
                    ids = ids[:pos + 1]
                    break
        generated_token_ids.append(ids)
    texts = [loaded.tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_token_ids]

    # Important memory optimization: use the transformer body rather than the
    # CausalLM wrapper, so no [batch, sequence, vocabulary] logits tensor is built.
    body = _base_model(loaded.model)
    with torch.inference_mode(), _precision_context(device, runtime):
        full_out = body(
            input_ids=generated,
            attention_mask=tok.get("attention_mask"),
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    hidden_states = full_out.hidden_states
    per_sample_hidden = [
        [
            h[i, input_width: input_width + len(ids), :].detach().float().cpu().numpy()
            for h in hidden_states
        ]
        for i, ids in enumerate(generated_token_ids)
    ]
    del full_out, hidden_states, generated, tok
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return generated_token_ids, texts, per_sample_hidden


def generate_batch(loaded: LoadedModel, prompts: list[str], generation: dict[str, Any], runtime: dict[str, Any]):
    return _generate_on_device(loaded, prompts, generation, runtime)


def _forward_on_device(loaded: LoadedModel, texts: list[str], max_tokens: int):
    device = loaded.device
    tok = loaded.tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_tokens,
    )
    tok = {k: v.to(device, non_blocking=True) for k, v in tok.items()}
    body = _base_model(loaded.model)
    with torch.inference_mode(), _precision_context(device, {}):
        out = body(
            **tok,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
    mask = tok["attention_mask"].detach().cpu().numpy().astype(bool)
    hidden_states = [h.detach().cpu().numpy() for h in out.hidden_states]
    del out, tok
    return hidden_states, mask


def forward_text_batch(loaded: LoadedModel, texts: list[str], max_tokens: int):
    return _forward_on_device(loaded, texts, max_tokens)


def release_model(loaded: LoadedModel) -> None:
    del loaded.model
    del loaded.tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
