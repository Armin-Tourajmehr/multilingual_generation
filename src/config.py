from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ConfigError("Configuration must be a YAML mapping.")
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    required = [
        "project", "languages", "language_names", "data", "models",
        "generation", "representation", "runtime", "outputs",
    ]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ConfigError(f"Missing configuration sections: {missing}")

    languages = cfg["languages"]
    if not languages:
        raise ConfigError("At least one language must be configured.")
    if len(languages) != len(set(languages)):
        raise ConfigError("Duplicate language codes found in config.")

    unknown_names = [lang for lang in languages if lang not in cfg["language_names"]]
    if unknown_names:
        raise ConfigError(f"Missing language names for: {unknown_names}")

    enabled = [name for name, spec in cfg["models"].items() if spec.get("enabled", False)]
    if not enabled:
        raise ConfigError("At least one model must be enabled.")

    train_fraction = cfg["data"].get("train_fraction")
    analysis_fraction = cfg["data"].get("analysis_fraction")
    if train_fraction is None or analysis_fraction is None:
        raise ConfigError("data.train_fraction and data.analysis_fraction are required.")
    if not (0 < train_fraction < 1) or not (0 < analysis_fraction < 1):
        raise ConfigError("Train and analysis fractions must both be in (0, 1).")
    if abs((train_fraction + analysis_fraction) - 1.0) > 1e-8:
        raise ConfigError("data.train_fraction + data.analysis_fraction must equal 1.0.")

    if cfg["data"].get("sampling", "random") not in {"random", "first"}:
        raise ConfigError("data.sampling must be 'random' or 'first'.")

    pca_dim = cfg["representation"].get("pca_dim")
    pca_variance = cfg["representation"].get("pca_variance")
    if pca_dim is None and pca_variance is None:
        raise ConfigError("Set either representation.pca_dim or representation.pca_variance.")
    if pca_dim is not None and pca_dim <= 0:
        raise ConfigError("representation.pca_dim must be positive.")
    if pca_variance is not None and not (0 < pca_variance <= 1):
        raise ConfigError("representation.pca_variance must be in (0, 1].")

    rep = cfg["representation"]
    if rep.get("pca_solver") == "incremental" and pca_variance is not None:
        raise ConfigError("Streaming IncrementalPCA requires representation.pca_variance=null.")
    if rep.get("pca_solver") == "incremental" and rep.get("gmm_components", 1) != 1:
        raise ConfigError("The scalable streaming path currently requires gmm_components=1.")
    if rep.get("pca_solver") == "incremental" and rep.get("gmm_covariance_type", "diag") != "diag":
        raise ConfigError("The scalable streaming path requires gmm_covariance_type='diag'.")
    if rep.get("pca_batch_size", 1) < max(2, int(pca_dim or 2)):
        raise ConfigError("representation.pca_batch_size must be at least pca_dim for incremental PCA.")


def enabled_models(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: spec
        for name, spec in cfg["models"].items()
        if spec.get("enabled", False)
    }
