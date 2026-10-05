from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from datasets import load_dataset


class DataError(RuntimeError):
    pass


# ------------------------------------------------------------------
# In-memory Aya cache
# ------------------------------------------------------------------
# Each language is loaded and sampled only once per Python process.
# The same sampled dataframe is reused by fitting, generation,
# metadata loading, etc.
_AYA_CACHE: dict[str, pd.DataFrame] = {}


def _aya_file(cfg: dict[str, Any], language: str) -> Path:
    return Path(cfg["data"]["root_dir"]) / f"{language}.csv"


def _aya_max_samples(cfg: dict[str, Any]) -> int:
    return int(
        cfg["data"].get(
            "max_samples_per_language",
            2000,
        )
    )


def _aya_sampling_seed(cfg: dict[str, Any]) -> int:
    return int(
        cfg["data"].get(
            "sampling_seed",
            cfg["project"]["seed"],
        )
    )


def _load_aya_language(
    cfg: dict[str, Any],
    language: str,
) -> pd.DataFrame:
    """
    Load, clean and deterministically sample Aya for one language.

    This function is cached in memory, so each language CSV is read
    only once during the whole Python process.

    Sampling rule:
        N <= max_samples:
            use all valid samples

        N > max_samples:
            deterministic random sample of exactly max_samples
    """

    if language in _AYA_CACHE:
        return _AYA_CACHE[language]

    path = _aya_file(cfg, language)

    if not path.exists():
        raise DataError(
            f"Missing Aya file for {language}: {path}"
        )

    input_col = cfg["data"]["input_column"]
    sample_id_col = cfg["data"]["sample_id_column"]

    # Read only the columns actually needed.
    df = pd.read_csv(
        path,
        usecols=[
            input_col,
            sample_id_col,
        ],
    )

    # Validate sample IDs.
    if sample_id_col not in df.columns:
        raise DataError(
            f"{path} must contain '{sample_id_col}'."
        )

    if (
        df[sample_id_col].isna().any()
        or df[sample_id_col].duplicated().any()
    ):
        raise DataError(
            f"Invalid sample IDs in {path}"
        )

    # Remove missing / empty inputs.
    df = df[df[input_col].notna()].copy()

    df[input_col] = df[input_col].astype(str)

    df = df[
        df[input_col].str.strip().ne("")
    ].copy()

    # --------------------------------------------------------------
    # Deterministic sampling
    # --------------------------------------------------------------
    max_samples = _aya_max_samples(cfg)

    if len(df) > max_samples:
        df = df.sample(
            n=max_samples,
            random_state=_aya_sampling_seed(cfg),
        )

    # Reset index so batches are cheap and predictable.
    df = df.reset_index(drop=True)

    # Add language once here instead of doing it for every batch.
    df["source_language"] = language

    # Cache the final sampled dataframe.
    _AYA_CACHE[language] = df

    return df


def clear_aya_cache() -> None:
    """
    Clear the in-memory Aya cache.

    Normally this is not needed. It can be useful if the dataset
    or configuration changes during an interactive notebook session.
    """
    _AYA_CACHE.clear()


def validate_aya_files(
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """
    Validate Aya files and report the number of samples that will
    actually be used after the 2000-per-language cap.
    """

    rows = []

    input_col = cfg["data"]["input_column"]
    sample_id_col = cfg["data"]["sample_id_column"]

    max_samples = _aya_max_samples(cfg)

    for language in cfg["languages"]:

        path = _aya_file(cfg, language)

        if not path.exists():
            raise DataError(
                f"Missing Aya file for {language}: {path}"
            )

        # Load through the same cache used by the experiment.
        # This means validation also prepares the data for later use.
        df = _load_aya_language(
            cfg,
            language,
        )

        rows.append(
            {
                "language": language,
                "num_samples": len(df),
                "num_available_after_cleaning": len(df)
                if len(df) < max_samples
                else max_samples,
                "file": str(path),
            }
        )

    return pd.DataFrame(rows)


def iter_aya_batches(
    cfg: dict[str, Any],
    language: str,
    batch_size: int,
) -> Iterator[pd.DataFrame]:
    """
    Yield batches from the cached, deterministic Aya subset.

    No CSV is re-read here.
    """

    df = _load_aya_language(
        cfg,
        language,
    )

    total = len(df)

    for start in range(
        0,
        total,
        batch_size,
    ):
        end = min(
            start + batch_size,
            total,
        )

        # iloc gives a lightweight dataframe slice.
        batch = df.iloc[start:end].copy()

        yield batch.reset_index(drop=True)


def load_aya_metadata(
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """
    Return metadata for the exact same Aya subset used by
    representation fitting and generation.

    Uses the in-memory cache, so CSV files are not re-read.
    """

    frames = []

    for language in cfg["languages"]:

        df = _load_aya_language(
            cfg,
            language,
        )

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


def iter_wikipedia_documents(
    cfg: dict[str, Any],
    language: str,
    seed_offset: int = 0,
) -> Iterator[dict[str, Any]]:

    wiki = cfg["wikipedia"]

    dataset = load_dataset(
        wiki["dataset_name"],
        language,
        split=wiki.get(
            "split",
            "train",
        ),
        streaming=True,
        revision=wiki.get("revision") or None,
    )

    dataset = dataset.shuffle(
        seed=(
            int(cfg["project"]["seed"])
            + seed_offset
        ),
        buffer_size=int(
            wiki.get(
                "shuffle_buffer_size",
                10000,
            )
        ),
    )

    limit = int(
        wiki["max_samples_per_language"]
    )

    text_col = wiki.get(
        "text_column",
        "text",
    )

    for idx, row in enumerate(dataset):

        if idx >= limit:
            break

        text = row.get(text_col)

        if text is None:
            continue

        text = str(text).strip()

        if not text:
            continue

        yield {
            "language": language,
            "sample_index": idx,
            "dataset_id": row.get("_id"),
            "title": row.get("title"),
            "text": text,
        }