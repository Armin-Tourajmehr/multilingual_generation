from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class DataError(RuntimeError):
    pass


def _read_language_file(cfg: dict[str, Any], language: str) -> pd.DataFrame:
    root = Path(cfg["data"]["root_dir"])
    path = root / f"{language}.csv"
    if not path.exists():
        raise DataError(f"Missing language file for '{language}': {path}")

    df = pd.read_csv(path)
    input_col = cfg["data"]["input_column"]
    sample_id_col = cfg["data"].get("sample_id_column", "sample_id")
    missing = [c for c in (input_col, sample_id_col) if c not in df.columns]
    if missing:
        raise DataError(f"{path} is missing required columns: {missing}")

    df = df[df[input_col].notna()].copy()
    df[input_col] = df[input_col].astype(str)
    df = df[df[input_col].str.strip().ne("")].copy()

    if df[sample_id_col].isna().any():
        raise DataError(f"{path} contains missing values in '{sample_id_col}'.")
    if df[sample_id_col].duplicated().any():
        raise DataError(f"{path} contains duplicate values in '{sample_id_col}'.")

    return df.reset_index(drop=True)


def validate_language_files(cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for language in cfg["languages"]:
        df = _read_language_file(cfg, language)
        rows.append({
            "language": language,
            "file": str(Path(cfg["data"]["root_dir"]) / f"{language}.csv"),
            "num_samples": len(df),
        })
    return pd.DataFrame(rows)


def _split_language_samples(
    df: pd.DataFrame,
    language: str,
    cfg: dict[str, Any],
    stage: str,
    seed: int,
) -> pd.DataFrame:
    data_cfg = cfg["data"]
    sampling = data_cfg.get("sampling", "random")
    train_fraction = float(data_cfg.get("train_fraction", 0.8))
    analysis_fraction = float(data_cfg.get("analysis_fraction", 0.2))

    if not 0 < train_fraction < 1:
        raise DataError("data.train_fraction must be in (0, 1).")
    if not 0 < analysis_fraction < 1:
        raise DataError("data.analysis_fraction must be in (0, 1).")
    if abs((train_fraction + analysis_fraction) - 1.0) > 1e-8:
        raise DataError("train_fraction + analysis_fraction must equal 1.0.")

    if sampling == "random":
        # Offset the seed by language so every language gets a deterministic,
        # independent shuffle while remaining reproducible across machines.
        language_offset = sum((i + 1) * ord(ch) for i, ch in enumerate(language))
        shuffled = df.sample(frac=1.0, random_state=seed + language_offset).reset_index(drop=True)
    elif sampling == "first":
        shuffled = df.reset_index(drop=True)
    else:
        raise DataError("data.sampling must be either 'random' or 'first'.")

    n_total = len(shuffled)
    n_train = int(n_total * train_fraction)

    # Preserve at least one training and one analysis sample whenever a
    # language has at least two usable samples.
    if n_total >= 2:
        n_train = max(1, min(n_total - 1, n_train))

    if stage == "train":
        selected = shuffled.iloc[:n_train].copy()
    elif stage == "analysis":
        selected = shuffled.iloc[n_train:].copy()
    else:
        raise ValueError("stage must be 'train' or 'analysis'")

    selected.insert(0, "source_language", language)
    selected["experiment_split"] = stage
    return selected.reset_index(drop=True)


def load_experiment_data(
    cfg: dict[str, Any],
    stage: str,
    seed: int,
) -> pd.DataFrame:
    if stage not in {"train", "analysis"}:
        raise ValueError("stage must be 'train' or 'analysis'")

    frames = []
    for language in cfg["languages"]:
        df = _read_language_file(cfg, language)
        frames.append(_split_language_samples(df, language, cfg, stage, seed))

    result = pd.concat(frames, ignore_index=True)
    result["experiment_row"] = range(len(result))
    return result


def validate_train_analysis_split(
    train: pd.DataFrame,
    analysis: pd.DataFrame,
    sample_id_col: str = "sample_id",
) -> None:
    for language in sorted(set(train["source_language"]) | set(analysis["source_language"])):
        train_ids = set(train.loc[train["source_language"] == language, sample_id_col].tolist())
        analysis_ids = set(analysis.loc[analysis["source_language"] == language, sample_id_col].tolist())
        overlap = train_ids & analysis_ids
        if overlap:
            raise DataError(
                f"Train/analysis leakage detected for '{language}': "
                f"{len(overlap)} overlapping sample IDs."
            )
