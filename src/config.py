from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when experiment configuration is incomplete or inconsistent."""


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ConfigError("Configuration must be a YAML mapping.")

    validate_config(cfg)
    resolved_config = path.resolve()
    repo_root = resolved_config.parent.parent if resolved_config.parent.name == "configs" else resolved_config.parent
    cfg["_repo_root"] = str(repo_root)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    required = [
        "project", "languages", "language_names", "data", "wikipedia",
        "models", "generation", "representation", "runtime", "outputs",
    ]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ConfigError(f"Missing config sections: {missing}")

    langs = cfg["languages"]
    names = cfg["language_names"]
    if not isinstance(langs, list) or not langs or any(not isinstance(x, str) or not x for x in langs):
        raise ConfigError("languages must be a non-empty list of language codes.")
    if len(langs) != len(set(langs)):
        raise ConfigError("Duplicate languages in config.")
    if not isinstance(names, dict) or set(langs) != set(names):
        raise ConfigError("languages and language_names must contain the same codes.")
    if any(not isinstance(names[lang], str) or not names[lang].strip() for lang in langs):
        raise ConfigError("Every configured language needs a non-empty display name.")

    data = cfg["data"]
    for key in ("root_dir", "input_column", "sample_id_column"):
        if not data.get(key):
            raise ConfigError(f"data.{key} is required.")
    if data.get("input_column") != "inputs":
        raise ConfigError("Aya input_column must be 'inputs'.")
    _positive_int(data.get("max_samples_per_language"), "data.max_samples_per_language")
    if not isinstance(data.get("sampling_seed", cfg["project"].get("seed", 42)), int):
        raise ConfigError("data.sampling_seed must be an integer.")

    wiki = cfg["wikipedia"]
    if not wiki.get("dataset_name"):
        raise ConfigError("wikipedia.dataset_name is required.")
    _positive_int(wiki.get("max_samples_per_language"), "wikipedia.max_samples_per_language")
    _positive_int(wiki.get("max_tokens_per_document"), "wikipedia.max_tokens_per_document")

    models = cfg["models"]
    if not isinstance(models, dict) or not models:
        raise ConfigError("models must be a non-empty mapping.")
    enabled_models = []
    for name, spec in models.items():
        if not isinstance(spec, dict):
            raise ConfigError(f"models.{name} must be a mapping.")
        if not isinstance(spec.get("enabled", False), bool):
            raise ConfigError(f"models.{name}.enabled must be true or false.")
        if spec.get("enabled", False):
            enabled_models.append(name)
            if not spec.get("model_name"):
                raise ConfigError(f"models.{name}.model_name is required for enabled models.")
        dtype = spec.get("torch_dtype", "auto")
        if dtype not in {"auto", "float16", "bfloat16", "float32", None}:
            raise ConfigError(f"Unsupported torch_dtype for model {name}: {dtype}")
        overrides = spec.get("runtime_overrides", {})
        if not isinstance(overrides, dict):
            raise ConfigError(f"models.{name}.runtime_overrides must be a mapping.")
    if not enabled_models:
        raise ConfigError("At least one model must be enabled.")

    rep = cfg["representation"]
    _positive_int(rep.get("pca_dim"), "representation.pca_dim")
    if rep.get("pca_solver") != "gpu_covariance":
        raise ConfigError("Main experiment uses GPU covariance PCA.")
    if rep.get("gmm_components") != 1 or rep.get("gmm_covariance_type") != "diag":
        raise ConfigError("This implementation requires one diagonal Gaussian per language/layer.")
    if float(rep.get("gmm_reg_covar", 0.0)) <= 0:
        raise ConfigError("representation.gmm_reg_covar must be positive.")

    gen = cfg["generation"]
    _positive_int(gen.get("max_new_tokens"), "generation.max_new_tokens")
    if gen.get("do_sample", True):
        if float(gen.get("temperature", 0.7)) <= 0:
            raise ConfigError("generation.temperature must be positive.")
        if not 0 < float(gen.get("top_p", 0.9)) <= 1:
            raise ConfigError("generation.top_p must be in (0, 1].")

    runtime = cfg["runtime"]
    for key in ("evaluation_batch_size", "wikipedia_batch_size", "generation_batch_size", "max_input_tokens"):
        _positive_int(runtime.get(key), f"runtime.{key}")
    if runtime.get("require_gpu", False) and runtime.get("device", "cuda") not in {"cuda", "gpu", "auto"}:
        raise ConfigError("GPU-only runtime.device must be cuda, gpu, or auto.")
    mixed_precision = str(runtime.get("mixed_precision", "fp16")).lower()
    if mixed_precision not in {"fp16", "bf16", "none", "off", "false"}:
        raise ConfigError("runtime.mixed_precision must be fp16, bf16, or disabled.")

    lexical = cfg.get("lexical_analysis", {})
    if not isinstance(lexical.get("enabled", True), bool):
        raise ConfigError("lexical_analysis.enabled must be true or false.")
    spectral = cfg.get("spectral_embedding", {})
    if spectral.get("enabled", True):
        _positive_int(spectral.get("n_components", 2), "spectral_embedding.n_components")
        _positive_int(spectral.get("n_neighbors", 5), "spectral_embedding.n_neighbors")
    smoke = cfg.get("smoke_test", {})
    if smoke:
        _positive_int(smoke.get("samples_per_language", 5), "smoke_test.samples_per_language")
        _positive_int(smoke.get("wikipedia_samples_per_language", 10), "smoke_test.wikipedia_samples_per_language")


def _positive_int(value: Any, key: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{key} must be a positive integer.")


def enabled_models(cfg: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return enabled models in configuration insertion order."""
    return [(name, spec) for name, spec in cfg["models"].items() if spec.get("enabled", False)]
