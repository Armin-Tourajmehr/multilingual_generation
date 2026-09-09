from __future__ import annotations

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


def _dtype(value: str):
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
            "A CUDA GPU is required for the Kaggle main experiment. "
            "Enable Settings -> Accelerator -> GPU and restart the session."
        )
    if requested in {"cuda", "gpu", "auto"}:
        return torch.device("cuda:0")
    if isinstance(requested, int) or (isinstance(requested, str) and requested.isdigit()):
        index = int(requested)
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"cuda:{index} is unavailable. Found {torch.cuda.device_count()} device(s).")
        return torch.device(f"cuda:{index}")
    device = torch.device(str(requested))
    if device.type != "cuda":
        raise RuntimeError("The Kaggle main experiment is GPU-only; a CUDA device is required.")
    return device


def _precision_context(device: torch.device, runtime: dict[str, Any]):
    mode = str(runtime.get("mixed_precision", "fp16")).lower()
    if mode in {"none", "off", "false"}:
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
    dtype = _dtype(spec.get("torch_dtype", "auto"))
    kwargs["dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(spec["model_name"], **kwargs)
    # Explicit single-GPU placement. No device_map/offload is used.
    model.to(device)
    model.eval()

    print(f"[{model_key}] CUDA available: {torch.cuda.is_available()}")
    print(f"[{model_key}] GPU: {torch.cuda.get_device_name(device.index or 0)}")
    props = torch.cuda.get_device_properties(device.index or 0)
    print(f"[{model_key}] GPU memory: {props.total_memory / (1024**3):.1f} GiB")
    model_device = next(model.parameters()).device
    if model_device.type != "cuda":
        raise RuntimeError(f"Model parameters are on {model_device}, expected CUDA.")

    param_dtypes = sorted({str(p.dtype) for p in model.parameters()})
    print(f"[{model_key}] Parameter dtype(s): {param_dtypes}")
    if spec.get("torch_dtype") == "float16" and any(p.dtype != torch.float16 for p in model.parameters()):
        raise RuntimeError(f"[{model_key}] Expected FP16 model parameters but found {param_dtypes}.")
    print(f"[{model_key}] Model parameters device: {model_device}")
    return LoadedModel(model_key, spec, tokenizer, model, device)


def _base_model(model):
    return getattr(model, "base_model", model)


def _format_inputs(loaded: LoadedModel, prompts: list[str]) -> list[str]:
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


def _tokenize(loaded: LoadedModel, texts: list[str], runtime: dict[str, Any], max_length: int | None = None):
    tok = loaded.tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {k: v.to(loaded.device, non_blocking=True) for k, v in tok.items()}


@torch.inference_mode()
def _forward_on_device(loaded: LoadedModel, texts: list[str], max_tokens: int | None, runtime: dict[str, Any]):
    tok = _tokenize(loaded, texts, runtime, max_tokens)
    body = _base_model(loaded.model)
    with _precision_context(loaded.device, runtime):
        out = body(
            **tok,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
    hidden_states = list(out.hidden_states)
    mask = tok["attention_mask"].bool()
    # Explicitly release lightweight wrapper objects before returning.
    del out, tok
    return hidden_states, mask


def forward_text_batch(loaded: LoadedModel, texts: list[str], max_tokens: int | None, runtime: dict[str, Any]):
    return _forward_on_device(loaded, texts, max_tokens, runtime)


@torch.inference_mode()
def _generate_on_device(
    loaded: LoadedModel,
    prompts: list[str],
    generation: dict[str, Any],
    runtime: dict[str, Any],
):
    if len(prompts) != 1:
        raise ValueError("Kaggle-safe generation requires generation_batch_size=1.")

    rendered = _format_inputs(loaded, prompts)
    max_input_tokens = runtime.get("max_input_tokens") or None
    tok = _tokenize(loaded, rendered, runtime, max_input_tokens)
    input_width = int(tok["input_ids"].shape[1])
    input_mask = tok["attention_mask"].bool()

    # Generation is the highest peak-memory operation in the pipeline.
    # Keeping the batch at 1 and model weights in FP16 makes mGPT viable on a T4.
    with _precision_context(loaded.device, runtime):
        generated = loaded.model.generate(
            **tok,
            max_new_tokens=int(generation["max_new_tokens"]),
            do_sample=bool(generation.get("do_sample", True)),
            temperature=float(generation["temperature"]) if generation.get("do_sample", True) else None,
            top_p=float(generation["top_p"]) if generation.get("do_sample", True) else None,
            pad_token_id=loaded.tokenizer.pad_token_id,
            use_cache=True,
        )

    generated_token_ids = generated[:, input_width:].detach().cpu().tolist()
    eos = loaded.tokenizer.eos_token_id
    if eos is not None:
        eos = int(eos)
        for i, ids in enumerate(generated_token_ids):
            for pos, token_id in enumerate(ids):
                if int(token_id) == eos:
                    generated_token_ids[i] = ids[: pos + 1]
                    break
    texts = loaded.tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)

    # We need the generated sequence for the representation forward pass, but
    # generation-time input tensors and tokenization wrappers can be released.
    full_attention = torch.ones_like(generated, dtype=torch.long, device=loaded.device)
    full_attention[:, :input_width] = input_mask.long()
    del tok, input_mask

    body = _base_model(loaded.model)
    with _precision_context(loaded.device, runtime):
        full_out = body(
            input_ids=generated,
            attention_mask=full_attention,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    hidden_states = list(full_out.hidden_states)
    # Do not keep additional references to the large wrapper/mask tensor.
    del full_out, full_attention, generated
    return generated_token_ids, texts, hidden_states, input_width


def generate_batch(loaded: LoadedModel, prompts: list[str], generation: dict[str, Any], runtime: dict[str, Any]):
    return _generate_on_device(loaded, prompts, generation, runtime)


def release_model(loaded: LoadedModel) -> None:
    del loaded.model
    del loaded.tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
