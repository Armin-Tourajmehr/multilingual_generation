from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    validate_config(cfg)

    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    required = [
        "project",
        "languages",
        "language_names",
        "data",
        "wikipedia",
        "models",
        "generation",
        "representation",
        "runtime",
        "outputs",
    ]

    missing = [
        key
        for key in required
        if key not in cfg
    ]

    if missing:
        raise ConfigError(
            f"Missing config sections: {missing}"
        )

    # --------------------------------------------------------
    # Languages
    # --------------------------------------------------------

    langs = cfg["languages"]

    if len(langs) != len(set(langs)):
        raise ConfigError(
            "Duplicate languages in config."
        )

    if set(langs) != set(cfg["language_names"]):
        raise ConfigError(
            "languages and language_names must contain "
            "the same codes."
        )

    # --------------------------------------------------------
    # Aya data
    # --------------------------------------------------------

    data = cfg["data"]

    if data.get("input_column") != "inputs":
        raise ConfigError(
            "Aya input_column must be 'inputs'."
        )

    if not data.get("root_dir"):
        raise ConfigError(
            "data.root_dir is required."
        )

    # --------------------------------------------------------
    # Wikipedia
    # --------------------------------------------------------

    wiki = cfg["wikipedia"]

    if not wiki.get("dataset_name"):
        raise ConfigError(
            "wikipedia.dataset_name is required."
        )

    max_samples = wiki.get(
        "max_samples_per_language"
    )

    if (
        not isinstance(max_samples, int)
        or max_samples <= 0
    ):
        raise ConfigError(
            "wikipedia.max_samples_per_language "
            "must be a positive integer."
        )

    max_tokens = wiki.get(
        "max_tokens_per_document"
    )

    if (
        not isinstance(max_tokens, int)
        or max_tokens <= 0
    ):
        raise ConfigError(
            "wikipedia.max_tokens_per_document "
            "must be a positive integer."
        )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    enabled = [
        name
        for name, settings in cfg["models"].items()
        if settings.get("enabled", False)
    ]

    if not enabled:
        raise ConfigError(
            "At least one model must be enabled."
        )

    supported_models = {
        "mgpt",
        "bloom",
        "qwen",
    }

    unsupported = [
        name
        for name in enabled
        if name not in supported_models
    ]

    if unsupported:
        raise ConfigError(
            f"Unsupported enabled model(s): {unsupported}"
        )

    # Gemma has intentionally been removed.
    if cfg["models"].get("gemma3_4b") is not None:
        raise ConfigError(
            "Gemma is intentionally removed from this repository."
        )

    # --------------------------------------------------------
    # Representation
    # --------------------------------------------------------

    rep = cfg["representation"]

    if rep.get("pca_dim") != 700:
        raise ConfigError(
            "representation.pca_dim must be 700 "
            "for the main experiment."
        )

    if rep.get("pca_solver") != "gpu_covariance":
        raise ConfigError(
            "Main experiment uses GPU covariance PCA; "
            "CPU IncrementalPCA is not used."
        )

    if (
        rep.get("gmm_components") != 1
        or rep.get("gmm_covariance_type") != "diag"
    ):
        raise ConfigError(
            "Scalable main experiment requires one "
            "diagonal Gaussian component per "
            "language/layer."
        )

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    gen = cfg["generation"]

    if not gen.get("max_new_tokens", 0) > 0:
        raise ConfigError(
            "generation.max_new_tokens must be positive."
        )

    # --------------------------------------------------------
    # Runtime
    # --------------------------------------------------------

    runtime = cfg["runtime"]

    if (
        runtime.get("evaluation_batch_size", 0) <= 0
        or runtime.get("wikipedia_batch_size", 0) <= 0
    ):
        raise ConfigError(
            "runtime batch sizes must be positive."
        )

    #if int(runtime.get("generation_batch_size", 1)) != 1:
        #raise ConfigError(
            #"Kaggle-safe main experiment requires "
            #"generation_batch_size=1.")

    if int(
        runtime.get("max_input_tokens", 0)
    ) <= 0:
        raise ConfigError(
            "runtime.max_input_tokens must be "
            "a positive integer."
        )

    if (
        runtime.get("require_gpu", False)
        and runtime.get("device")
        not in {"cuda", "gpu", "auto"}
    ):
        raise ConfigError(
            "GPU-only runtime.device must be "
            "cuda, gpu, or auto."
        )

    mixed_precision = str(
        runtime.get(
            "mixed_precision",
            "fp16",
        )
    ).lower()

    if mixed_precision not in {
        "fp16",
        "bf16",
        "none",
        "off",
        "false",
    }:
        raise ConfigError(
            "runtime.mixed_precision must be "
            "fp16, bf16, or disabled."
        )


def enabled_models(
    cfg: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """
    Return enabled models in configuration order.
    """

    return [
        (name, settings)
        for name, settings in cfg["models"].items()
        if settings.get("enabled", False)
    ]
