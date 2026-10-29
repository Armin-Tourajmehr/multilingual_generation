from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from datasets import load_dataset


class DataError(RuntimeError):
    pass


# Cache the deterministic subset only. Include the sampling parameters in the key
# so smoke tests and full runs in one Python process cannot accidentally share rows.
_AYA_CACHE: dict[tuple[str, int, int, str, str, int, int], pd.DataFrame] = {}


def _aya_file(cfg: dict[str, Any], language: str) -> Path:
    root = Path(cfg["data"]["root_dir"])
    if not root.is_absolute():
        root = Path(cfg.get("_repo_root", Path.cwd())) / root
    return root / f"{language}.csv"


def _aya_max_samples(cfg: dict[str, Any]) -> int:
    return int(cfg["data"].get("max_samples_per_language", 2000))


def _aya_sampling_seed(cfg: dict[str, Any]) -> int:
    return int(cfg["data"].get("sampling_seed", cfg["project"].get("seed", 42)))


def _cache_key(cfg: dict[str, Any], language: str) -> tuple[str, int, int, str, str, int, int]:
    path = _aya_file(cfg, language)
    stat = path.stat() if path.exists() else None
    return (
        str(path.resolve()),
        _aya_max_samples(cfg),
        _aya_sampling_seed(cfg),
        str(cfg["data"]["input_column"]),
        str(cfg["data"]["sample_id_column"]),
        int(stat.st_mtime_ns) if stat is not None else 0,
        int(stat.st_size) if stat is not None else 0,
    )


def _load_aya_language(cfg: dict[str, Any], language: str) -> pd.DataFrame:
    """Read, clean and deterministically sample one Aya language CSV."""
    key = _cache_key(cfg, language)
    if key in _AYA_CACHE:
        return _AYA_CACHE[key]

    path = _aya_file(cfg, language)
    if not path.exists():
        raise DataError(
            f"Missing Aya file for {language}: {path}. "
            "Run scripts/prepare_aya.py first."
        )

    input_col = str(cfg["data"]["input_column"])
    sample_id_col = str(cfg["data"]["sample_id_column"])
    try:
        df = pd.read_csv(path, usecols=[input_col, sample_id_col])
    except (ValueError, pd.errors.ParserError) as exc:
        raise DataError(
            f"Could not read required columns {input_col!r}, {sample_id_col!r} from {path}: {exc}"
        ) from exc

    if df[sample_id_col].isna().any() or df[sample_id_col].duplicated().any():
        raise DataError(f"Invalid or duplicate sample IDs in {path}")

    df = df[df[input_col].notna()].copy()
    df[input_col] = df[input_col].astype(str)
    df = df[df[input_col].str.strip().ne("")].copy()

    max_samples = _aya_max_samples(cfg)
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=_aya_sampling_seed(cfg))

    df = df.reset_index(drop=True)
    df["source_language"] = language
    _AYA_CACHE[key] = df
    return df


def clear_aya_cache() -> None:
    """Clear in-memory Aya subsets, e.g. before running smoke tests in notebooks."""
    _AYA_CACHE.clear()


def validate_aya_files(cfg: dict[str, Any]) -> pd.DataFrame:
    """Validate that each configured language exists and report effective sample counts."""
    rows = []
    max_samples = _aya_max_samples(cfg)
    for language in cfg["languages"]:
        path = _aya_file(cfg, language)
        if not path.exists():
            raise DataError(
                f"Missing Aya file for {language}: {path}. "
                "Run scripts/prepare_aya.py --config configs/config.yaml first."
            )
        df = _load_aya_language(cfg, language)
        rows.append({
            "language": language,
            "num_samples": len(df),
            "sample_cap_per_language": max_samples,
            "file": str(path),
        })
    return pd.DataFrame(rows)


def iter_aya_batches(cfg: dict[str, Any], language: str, batch_size: int) -> Iterator[pd.DataFrame]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    df = _load_aya_language(cfg, language)
    for start in range(0, len(df), batch_size):
        yield df.iloc[start:start + batch_size].reset_index(drop=True).copy()


def load_aya_metadata(cfg: dict[str, Any]) -> pd.DataFrame:
    frames = [_load_aya_language(cfg, language) for language in cfg["languages"]]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def iter_wikipedia_documents(
    cfg: dict[str, Any], language: str, seed_offset: int = 0
) -> Iterator[dict[str, Any]]:
    """Stream a deterministic capped sample of Wikipedia documents for one language."""
    wiki = cfg["wikipedia"]
    dataset = load_dataset(
        wiki["dataset_name"],
        language,
        split=wiki.get("split", "train"),
        streaming=True,
        revision=wiki.get("revision") or None,
    )
    dataset = dataset.shuffle(
        seed=int(cfg["project"].get("seed", 42)) + seed_offset,
        buffer_size=int(wiki.get("shuffle_buffer_size", 10000)),
    )

    limit = int(wiki["max_samples_per_language"])
    text_col = wiki.get("text_column", "text")
    emitted = 0
    for idx, row in enumerate(dataset):
        if emitted >= limit:
            break
        text = row.get(text_col)
        if text is None:
            continue
        text = str(text).strip()
        if not text:
            continue
        yield {
            "language": language,
            "sample_index": emitted,
            "dataset_id": row.get("_id"),
            "title": row.get("title"),
            "text": text,
        }
        emitted += 1
