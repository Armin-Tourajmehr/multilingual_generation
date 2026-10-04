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


def _resolve_device(runtime):
    requested = runtime.get("device", "cuda")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but is not available.")

    if requested in {"cuda", "gpu", "auto"}:
        return torch.device("cuda:0")

    if isinstance(requested, int) or (
        isinstance(requested, str) and requested.isdigit()
    ):
        index = int(requested)

        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested cuda:{index}, "
                f"but only {torch.cuda.device_count()} GPU(s) are available."
            )

        return torch.device(f"cuda:{index}")

    device = torch.device(str(requested))

    if device.type != "cuda":
        raise RuntimeError("This pipeline requires a CUDA device.")

    return device


def _precision_context(device, runtime):
    mode = str(runtime.get("mixed_precision", "fp16")).lower()

    if mode in {"none", "off", "false"}:
        return nullcontext()

    if mode == "fp16":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )

    if mode == "bf16":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )

    raise ValueError(
        f"Unsupported mixed_precision mode: {mode}"
    )


def load_model(
    name: str,
    spec: dict[str, Any],
    runtime: dict[str, Any],
) -> LoadedModel:

    device = _resolve_device(runtime)

    tokenizer = AutoTokenizer.from_pretrained(
        spec["model_name"],
        trust_remote_code=bool(
            runtime.get("trust_remote_code", False)
        ),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Left padding is important for causal LM generation.
    tokenizer.padding_side = "left"

    kwargs = {
        "trust_remote_code": bool(
            runtime.get("trust_remote_code", False)
        ),
        "low_cpu_mem_usage": bool(
            runtime.get("low_cpu_mem_usage", True)
        ),
    }

    dtype = _dtype(spec.get("torch_dtype", "auto"))

    if dtype != "auto":
        kwargs["dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(
        spec["model_name"],
        **kwargs,
    )

    model.to(device)
    model.eval()

    print(f"Loaded model: {name}")
    print(f"Model name: {spec['model_name']}")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(device.index or 0)}"
        )

    return LoadedModel(
        name=name,
        spec=spec,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )


def _base_model(model):
    return getattr(model, "base_model", model)


def _format_inputs(
    loaded: LoadedModel,
    prompts: list[str],
) -> list[str]:

    if not loaded.spec.get("use_chat_template", False):
        return prompts

    kwargs = loaded.spec.get(
        "chat_template_kwargs",
        {},
    )

    return [
        loaded.tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )
        for prompt in prompts
    ]


def _tokenize(
    loaded: LoadedModel,
    texts: list[str],
    runtime: dict[str, Any],
    max_length: int | None = None,
):
    tok = loaded.tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    return {
        key: value.to(
            loaded.device,
            non_blocking=True,
        )
        for key, value in tok.items()
    }


@torch.inference_mode()
def _forward_on_device(
    loaded: LoadedModel,
    prompts: list[str],
    runtime: dict[str, Any],
):
    tok = _tokenize(
        loaded,
        prompts,
        runtime,
        runtime.get("max_input_tokens") or None,
    )

    body = _base_model(loaded.model)

    with _precision_context(
        loaded.device,
        runtime,
    ):
        out = body(
            **tok,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    hidden_states = list(out.hidden_states)
    mask = tok["attention_mask"].bool()

    del out
    del tok

    return hidden_states, mask


def forward_text_batch(
    loaded: LoadedModel,
    prompts: list[str],
    runtime: dict[str, Any],
):
    return _forward_on_device(
        loaded,
        prompts,
        runtime,
    )


@torch.inference_mode()
def _generate_on_device(
    loaded: LoadedModel,
    prompts: list[str],
    generation: dict[str, Any],
    runtime: dict[str, Any],
):
    """
    Generate a batch and capture the pre-emission hidden state
    associated with every generated token at every layer.

    Output:
        generated_token_ids:
            list[list[int]]
            One token sequence per input.

        generated_texts:
            list[str]

        hidden_states:
            list[Tensor]
            Each tensor has shape [B, T, D].

            hidden_states[layer][sample, token]
            corresponds to the same pre-emission representation
            used by the previous batch_size=1 implementation.

        input_width:
            Width of the padded input sequence.
    """

    batch_size = len(prompts)

    if batch_size == 0:
        raise ValueError("Cannot generate an empty batch.")

    rendered = _format_inputs(
        loaded,
        prompts,
    )

    max_input_tokens = (
        runtime.get("max_input_tokens") or None
    )

    tok = _tokenize(
        loaded,
        rendered,
        runtime,
        max_input_tokens,
    )

    input_width = int(
        tok["input_ids"].shape[1]
    )

    with _precision_context(
        loaded.device,
        runtime,
    ):
        generated = loaded.model.generate(
            **tok,
            max_new_tokens=int(
                generation["max_new_tokens"]
            ),
            do_sample=bool(
                generation.get("do_sample", True)
            ),
            temperature=(
                float(generation["temperature"])
                if generation.get("do_sample", True)
                else None
            ),
            top_p=(
                float(generation["top_p"])
                if generation.get("do_sample", True)
                else None
            ),
            pad_token_id=loaded.tokenizer.pad_token_id,
            use_cache=True,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )

    sequences = generated.sequences

    # ------------------------------------------------------------
    # Generated token IDs
    # ------------------------------------------------------------

    generated_token_tensor = sequences[
        :,
        input_width:,
    ]

    generated_token_ids = (
        generated_token_tensor
        .detach()
        .cpu()
        .tolist()
    )

    eos_id = loaded.tokenizer.eos_token_id
    pad_id = loaded.tokenizer.pad_token_id

    lengths = []

    for i, ids in enumerate(generated_token_ids):
        length = len(ids)

        if eos_id is not None:
            eos_id_int = int(eos_id)

            for pos, token_id in enumerate(ids):
                if int(token_id) == eos_id_int:
                    length = pos + 1
                    break

        elif pad_id is not None:
            pad_id_int = int(pad_id)

            for pos, token_id in enumerate(ids):
                if int(token_id) == pad_id_int:
                    length = pos
                    break

        lengths.append(length)

        generated_token_ids[i] = ids[:length]

    max_generated = max(lengths, default=0)

    if max_generated == 0:
        raise RuntimeError(
            "Generation produced zero tokens."
        )

    generated_texts = (
        loaded.tokenizer.batch_decode(
            generated_token_ids,
            skip_special_tokens=True,
        )
    )

    # ------------------------------------------------------------
    # Hidden states
    # ------------------------------------------------------------

    step_hidden_states = generated.hidden_states

    if not step_hidden_states:
        raise RuntimeError(
            "Generation returned no hidden states."
        )

    n_steps = len(step_hidden_states)

    n_layers = len(
        step_hidden_states[0]
    )

    # We collect the final position from every generation step.
    #
    # Each:
    #     step_hidden_states[t][layer]
    #
    # has shape approximately:
    #
    #     [B, sequence_length, D]
    #
    # We only need:
    #
    #     [:, -1, :]
    #
    # which is the pre-emission representation used by
    # the previous implementation.
    #
    # Result:
    #
    #     hidden_states[layer]
    #         -> [B, T, D]

    hidden_states = []

    for layer in range(n_layers):

        pieces = []

        for step_idx in range(n_steps):
            h = step_hidden_states[
                step_idx
            ][layer]

            pieces.append(
                h[:, -1, :]
            )

        layer_hidden = torch.stack(
            pieces,
            dim=1,
        )

        # Only retain the actual generated sequence length.
        #
        # Different samples may finish at different steps.
        # We keep the complete rectangular tensor and let the
        # caller use generated lengths to mask invalid positions.
        layer_hidden = layer_hidden[
            :,
            :max_generated,
            :,
        ]

        hidden_states.append(
            layer_hidden.contiguous()
        )

    del generated
    del step_hidden_states
    del tok
    del rendered
    del sequences
    del generated_token_tensor

    return (
        generated_token_ids,
        generated_texts,
        hidden_states,
        input_width,
        lengths,
    )


def generate_batch(
    loaded: LoadedModel,
    prompts: list[str],
    generation: dict[str, Any],
    runtime: dict[str, Any],
):
    return _generate_on_device(
        loaded,
        prompts,
        generation,
        runtime,
    )


def release_model(
    loaded: LoadedModel,
):
    del loaded.model
    del loaded.tokenizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()