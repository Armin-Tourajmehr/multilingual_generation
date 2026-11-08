"""Shared CPU input and output handling for reports on saved token scores."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

import numpy as np
import pyarrow as pa

from .affinity import _partition_batches, _posterior_matrix, _score_columns
from .config import load_config


_TOKEN_DIRS = {"prompt": "prompt_token_posteriors", "generated": "token_posteriors"}


@dataclass(frozen=True)
class TokenReportOptions:
    models: list[str]
    candidates: list[str]
    phases: tuple[str, ...]
    layers: frozenset[int] | None
    results_dir: Path
    output_dir: Path


def parse_report_args(description: str, *, default_phase: str | None = "both") -> TokenReportOptions:
    parser = argparse.ArgumentParser(description=description)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--config", type=Path, default=root / "configs" / "config.yaml")
    parser.add_argument("--results-dir", type=Path, help="Directory containing the saved model results.")
    parser.add_argument("--output-dir", type=Path, help="Root directory for reports. Defaults to the results directory.")
    parser.add_argument("--models", nargs="+", help="Model keys to analyze. Defaults to the enabled models.")
    parser.add_argument("--layers", nargs="+", type=int, help="Transformer layer IDs to analyze. Defaults to all transformer layers.")
    if default_phase is not None:
        parser.add_argument("--phase", choices=["both", *_TOKEN_DIRS], default=default_phase)
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    models = args.models or [model for model, spec in cfg["models"].items() if spec.get("enabled", False)]
    unknown = set(models).difference(cfg["models"])
    if unknown:
        parser.error(f"Unknown model keys: {', '.join(sorted(unknown))}")
    if len(models) != len(set(models)):
        parser.error("Select each model only once.")
    layers = None
    if args.layers is not None:
        layers = frozenset(args.layers)
        if min(layers) < 1 or len(layers) != len(args.layers):
            parser.error("--layers requires unique positive transformer layer IDs.")
    phase = getattr(args, "phase", "generated")
    results_dir = args.results_dir or (
        Path(cfg["_repo_root"]) / cfg["outputs"]["root_dir"] / cfg["outputs"]["result_dir"]
    )
    return TokenReportOptions(
        models, cfg["languages"], tuple(_TOKEN_DIRS) if phase == "both" else (phase,),
        layers, results_dir, args.output_dir or results_dir,
    )


def iter_token_batches(
    model_dir: Path,
    *,
    model: str,
    candidates: list[str],
    phase: str,
    layers: frozenset[int] | None = None,
    extra_columns: tuple[str, ...] = (),
    batch_size: int = 65536,
) -> Iterator[tuple[int, np.ndarray, np.ndarray, pa.RecordBatch]]:
    """Stream projected scores and require complete source coverage at the selected layers."""
    score_columns = _score_columns(candidates, batch_size)
    directory = model_dir / _TOKEN_DIRS[phase]
    paths = [directory / f"{language}.parquet" for language in candidates]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {phase} posterior files for {model}: {', '.join(missing)}")

    expected = layers
    selected = sorted(layers) if layers is not None else None
    for source_index, (language, path) in enumerate(zip(candidates, paths)):
        observed = set()
        for batch in _partition_batches(
            path, model, language, score_columns, batch_size,
            continuation=False, extra_columns=extra_columns,
        ):
            layer_ids = batch.column("layer").to_numpy()
            keep = layer_ids > 0 if selected is None else np.isin(layer_ids, selected)
            if not keep.any():
                continue
            if not keep.all():
                batch = batch.filter(pa.array(keep))
                layer_ids = layer_ids[keep]
            observed.update(np.unique(layer_ids).tolist())
            yield source_index, layer_ids, _posterior_matrix(batch, score_columns, path), batch

        if not observed:
            raise ValueError(f"{path} has no scored tokens for the selected transformer layers")
        if expected is None:
            if sorted(observed) != list(range(1, len(observed) + 1)):
                raise ValueError(f"{path} must have contiguous transformer layers starting at 1 (found {sorted(observed)})")
            expected = frozenset(observed)
        elif observed != expected:
            raise ValueError(
                f"{path} has different transformer layer coverage "
                f"(missing {sorted(expected - observed)}, extra {sorted(observed - expected)})"
            )


def posterior_ratios(scores: np.ndarray, source_index: int) -> np.ndarray:
    """Return P(source)/P(candidate) in candidate order with the source column omitted."""
    foreign = np.concatenate((scores[:, :source_index], scores[:, source_index + 1:]), axis=1)
    np.abs(foreign, out=foreign)  # Treat either floating-point zero as a positive zero denominator.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        return np.divide(scores[:, source_index, None], foreign, dtype=np.float64)


@contextmanager
def atomic_report_path(path: Path) -> Iterator[Path]:
    """Publish a report file only after its writer completes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent, prefix=f".{path.stem}.") as directory:
        temporary_path = Path(directory) / path.name
        yield temporary_path
        temporary_path.replace(path)
