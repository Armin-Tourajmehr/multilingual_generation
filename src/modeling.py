from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import resolve_device, resolve_dtype


@dataclass
class LoadedModel:
    name: str
    model_name: str
    tokenizer: Any
    model: Any
    device: str
    use_chat_template: bool


def load_model(model_key: str, spec: dict[str, Any], runtime: dict[str, Any]) -> LoadedModel:
    device = resolve_device(runtime.get("device", "auto"))
    dtype = resolve_dtype(spec.get("torch_dtype", "auto"), device)

    tokenizer = AutoTokenizer.from_pretrained(
        spec["model_name"],
        trust_remote_code=runtime.get("trust_remote_code", False),
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": runtime.get("low_cpu_mem_usage", True),
        "trust_remote_code": runtime.get("trust_remote_code", False),
    }

    device_map = runtime.get("device_map", "auto")
    if device_map:
        model_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(spec["model_name"], **model_kwargs)
    model.eval()

    # device_map='auto' can leave the input on CPU while model.device points to
    # another location. The first parameter device is the safest input target.
    if not hasattr(model, "hf_device_map"):
        model.to(device)

    return LoadedModel(
        name=model_key,
        model_name=spec["model_name"],
        tokenizer=tokenizer,
        model=model,
        device=device,
        use_chat_template=spec.get("use_chat_template", False),
    )


def _input_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def prepare_prompt_inputs(loaded: LoadedModel, prompt: str, spec: dict[str, Any], runtime: dict[str, Any]):
    tokenizer = loaded.tokenizer
    max_input_tokens = runtime.get("max_input_tokens")

    if loaded.use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
        }
        chat_kwargs = spec.get("chat_template_kwargs", {}) or {}
        if chat_kwargs:
            kwargs["**chat_kwargs"] = chat_kwargs
            # Hugging Face does not accept a literal ** dict key; expand below.
            kwargs.pop("**chat_kwargs")
            kwargs.update(chat_kwargs)
        encoded = tokenizer.apply_chat_template(messages, **kwargs)
        if isinstance(encoded, dict):
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")
        else:
            input_ids = encoded
            attention_mask = torch.ones_like(input_ids)
    else:
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=max_input_tokens is not None,
            max_length=max_input_tokens,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")

    device = _input_device(loaded.model)
    batch = {"input_ids": input_ids.to(device)}
    if attention_mask is not None:
        batch["attention_mask"] = attention_mask.to(device)
    return batch


def generate_with_states(
    loaded: LoadedModel,
    prompt: str,
    model_spec: dict[str, Any],
    runtime_cfg: dict[str, Any],
    generation_cfg: dict[str, Any],
):
    inputs = prepare_prompt_inputs(loaded, prompt, model_spec, runtime_cfg)
    prompt_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        generated_ids = loaded.model.generate(
            **inputs,
            max_new_tokens=generation_cfg["max_new_tokens"],
            do_sample=generation_cfg.get("do_sample", True),
            temperature=generation_cfg.get("temperature", 0.7),
            top_p=generation_cfg.get("top_p", 0.9),
            pad_token_id=loaded.tokenizer.pad_token_id,
        )

        outputs = loaded.model(
            input_ids=generated_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    return generated_ids, prompt_length, outputs.hidden_states


def extract_generated_embeddings(hidden_states, prompt_length: int):
    embeddings = {}
    for layer_idx, layer_hidden in enumerate(hidden_states):
        # Keep the generated part only; layer 0 is the embedding output.
        gen = layer_hidden[0, prompt_length:, :]
        embeddings[layer_idx] = gen.float().cpu().numpy()
    return embeddings
