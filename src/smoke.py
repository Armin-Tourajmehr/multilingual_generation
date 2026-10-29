from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


def build_smoke_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Create an isolated, lightweight config for an end-to-end smoke run.

    With 20 active languages and two rows per language, this processes at most
    40 Aya rows per model. All selected models are attempted sequentially.
    Outputs and cached PCA/GMM files are isolated under outputs/smoke_test.
    """
    smoke_cfg = deepcopy(cfg)
    opts = cfg.get("smoke_test", {})

    samples_per_language = int(opts.get("samples_per_language", 2))
    smoke_cfg["data"]["max_samples_per_language"] = samples_per_language
    smoke_cfg["data"]["sampling_seed"] = int(cfg["project"].get("seed", 42))

    smoke_cfg["wikipedia"]["max_samples_per_language"] = int(
        opts.get("wikipedia_samples_per_language", 2)
    )
    smoke_cfg["wikipedia"]["max_tokens_per_document"] = int(
        opts.get("max_tokens_per_document", 32)
    )

    smoke_cfg["representation"]["pca_dim"] = min(
        int(opts.get("pca_dim", 16)), int(cfg["representation"]["pca_dim"])
    )
    smoke_cfg["generation"]["max_new_tokens"] = int(
        opts.get("generation_max_new_tokens", 8)
    )

    runtime = smoke_cfg["runtime"]
    runtime["max_input_tokens"] = int(opts.get("max_input_tokens", 64))
    runtime["generation_batch_size"] = int(opts.get("generation_batch_size", 1))
    runtime["wikipedia_batch_size"] = int(opts.get("wikipedia_batch_size", 1))
    runtime["evaluation_batch_size"] = int(opts.get("evaluation_batch_size", 1))

    lexical = smoke_cfg.setdefault("lexical_analysis", {})
    lexical["download_missing_models"] = bool(opts.get("download_stanza_models", False))

    selected = list(opts.get("models", ["mgpt", "bloom", "qwen", "llama3_1_8b"]))
    configured_enabled = [name for name, spec in cfg["models"].items() if spec.get("enabled", False)]
    selected = [name for name in selected if name in cfg["models"]]
    if not selected:
        selected = configured_enabled
    if not selected:
        raise ValueError("Smoke test cannot run: no valid model was selected or enabled.")
    for name, spec in smoke_cfg["models"].items():
        spec["enabled"] = name in selected
        if name in selected:
            # Override large-model defaults so smoke settings remain effective.
            spec["runtime_overrides"] = {
                "wikipedia_batch_size": runtime["wikipedia_batch_size"],
                "evaluation_batch_size": runtime["evaluation_batch_size"],
                "generation_batch_size": runtime["generation_batch_size"],
                "max_input_tokens": runtime["max_input_tokens"],
            }

    original_output_root = Path(cfg["outputs"].get("root_dir", "outputs"))
    smoke_root = original_output_root / "smoke_test"
    smoke_cfg["outputs"]["root_dir"] = str(smoke_root)
    runtime["pca_data_path"] = str(smoke_root / "pca_data")
    return smoke_cfg
