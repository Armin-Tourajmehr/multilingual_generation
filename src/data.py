
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from datasets import load_dataset


class DataError(RuntimeError):
    pass


def _aya_file(cfg: dict[str, Any], language: str) -> Path:
    return Path(cfg["data"]["root_dir"]) / f"{language}.csv"


def validate_aya_files(cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    input_col = cfg["data"]["input_column"]
    sample_id_col = cfg["data"]["sample_id_column"]
    for language in cfg["languages"]:
        path = _aya_file(cfg, language)
        if not path.exists():
            raise DataError(f"Missing Aya file for {language}: {path}")
        # Only read the columns needed for validation/counting.
        df = pd.read_csv(path, usecols=lambda c: c in {input_col, sample_id_col})
        if input_col not in df.columns or sample_id_col not in df.columns:
            raise DataError(f"{path} must contain '{input_col}' and '{sample_id_col}'.")
        if df[sample_id_col].isna().any() or df[sample_id_col].duplicated().any():
            raise DataError(f"Invalid sample IDs in {path}")
        nonempty = df[input_col].fillna("").astype(str).str.strip().ne("")
        rows.append({"language": language, "num_samples": int(nonempty.sum()), "file": str(path)})
    return pd.DataFrame(rows)


def iter_aya_batches(cfg: dict[str, Any], language: str, batch_size: int) -> Iterator[pd.DataFrame]:
    path = _aya_file(cfg, language)
    input_col = cfg["data"]["input_column"]
    sample_id_col = cfg["data"]["sample_id_column"]
    usecols = [input_col, sample_id_col]
    reader = pd.read_csv(path, usecols=usecols, chunksize=batch_size)
    for chunk in reader:
        chunk = chunk[chunk[input_col].notna()].copy()
        chunk[input_col] = chunk[input_col].astype(str)
        chunk = chunk[chunk[input_col].str.strip().ne("")].copy()
        chunk["source_language"] = language
        yield chunk.reset_index(drop=True)


def load_aya_metadata(cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for language in cfg["languages"]:
        path = _aya_file(cfg, language)
        df = pd.read_csv(path, usecols=[cfg["data"]["input_column"], cfg["data"]["sample_id_column"]])
        df["source_language"] = language
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def iter_wikipedia_documents(cfg: dict[str, Any], language: str, seed_offset: int = 0) -> Iterator[dict[str, Any]]:
    wiki = cfg["wikipedia"]
    dataset = load_dataset(
        wiki["dataset_name"],
        language,
        split=wiki.get("split", "train"),
        streaming=True,
        revision=wiki.get("revision") or None,
    )
    dataset = dataset.shuffle(
        seed=int(cfg["project"]["seed"]) + seed_offset,
        buffer_size=int(wiki.get("shuffle_buffer_size", 10000)),
    )
    limit = int(wiki["max_samples_per_language"])
    text_col = wiki.get("text_column", "text")
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
