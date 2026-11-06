"""CPU aggregation of matched prompt and continuation language affinities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


@dataclass(frozen=True)
class AffinitySummary:
    source_language: str
    candidates: list[str]
    layers: np.ndarray
    prompt: np.ndarray
    continuation: np.ndarray
    difference: np.ndarray
    paired_counts: np.ndarray


def _partition_batches(
    path: Path,
    model: str,
    source_language: str,
    score_columns: list[str],
    batch_size: int,
    *,
    continuation: bool,
) -> Iterator[pa.RecordBatch]:
    fields = ["model", "input_language", "sample_id", "layer", *score_columns]
    if continuation:
        fields.append("num_output_tokens")
    with pq.ParquetFile(path) as source:
        schema = source.schema_arrow
        if len(schema.names) != len(set(schema.names)):
            raise ValueError(f"{path} has duplicate column names")
        available_scores = {name for name in schema.names if name.startswith("P_")}
        if available_scores != set(score_columns):
            missing = sorted(set(score_columns) - available_scores)
            extra = sorted(available_scores - set(score_columns))
            raise ValueError(f"{path} has a different candidate pool (missing {missing}, extra {extra})")
        missing = sorted(set(fields) - set(schema.names))
        if missing:
            raise ValueError(f"{path} is missing required columns {missing}")
        for name in score_columns:
            dtype = schema.field(name).type
            if not (pa.types.is_floating(dtype) or pa.types.is_integer(dtype)):
                raise ValueError(f"{path} column {name} must contain numeric probabilities")
        integer_fields = ("layer", "num_output_tokens") if continuation else ("layer",)
        for name in integer_fields:
            if not pa.types.is_integer(schema.field(name).type):
                raise ValueError(f"{path} column {name} must contain integers")

        for batch in source.iter_batches(batch_size=batch_size, columns=fields):
            if not batch.num_rows:
                continue
            for name, expected in (("model", model), ("input_language", source_language)):
                values = batch.column(name)
                if values.null_count or any(value != expected for value in pc.unique(values).to_pylist()):
                    raise ValueError(f"{path} has rows outside the requested {name} partition {expected!r}")
            sample_ids = batch.column("sample_id")
            if sample_ids.null_count or (
                pa.types.is_floating(sample_ids.type) and not pc.all(pc.is_finite(sample_ids)).as_py()
            ):
                raise ValueError(f"{path} contains missing or non-finite sample IDs")
            for name in integer_fields:
                values = batch.column(name)
                if values.null_count or pc.min(values).as_py() < 0:
                    raise ValueError(f"{path} column {name} must contain non-negative integers")
            yield batch


def _posterior_matrix(
    batch: pa.RecordBatch, score_columns: list[str], path: Path, usable: np.ndarray | None = None
) -> np.ndarray:
    values = np.column_stack([batch.column(name).to_numpy(zero_copy_only=False) for name in score_columns])
    checked = values if usable is None else values[usable]
    if not np.isfinite(checked).all() or np.any((checked < 0) | (checked > 1)):
        raise ValueError(f"{path} contains posterior probabilities outside [0, 1] or missing values")
    if not np.allclose(checked.sum(axis=1, dtype=np.float64), 1.0, rtol=0, atol=1e-5):
        raise ValueError(f"{path} contains posterior rows that do not sum to 1")
    return values


def summarize_language(
    prompt_path: Path,
    continuation_path: Path,
    *,
    model: str,
    source_language: str,
    candidates: list[str],
    batch_size: int = 65536,
) -> AffinitySummary | None:
    """Give each matched sample/layer pair equal weight in both phases."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if (not candidates or any(not isinstance(candidate, str) or not candidate for candidate in candidates)
            or len(set(candidates)) != len(candidates)):
        raise ValueError("candidates must contain unique, non-empty language codes")
    score_columns = [f"P_{candidate}" for candidate in candidates]
    layers: set[int] = set()
    continuations = {}
    seen = set()

    for batch in _partition_batches(
        continuation_path, model, source_language, score_columns, batch_size, continuation=True
    ):
        usable = batch.column("num_output_tokens").to_numpy() > 0
        scores = _posterior_matrix(batch, score_columns, continuation_path, usable)
        for sample_id, layer, keep, values in zip(
            batch.column("sample_id").to_pylist(), batch.column("layer").to_pylist(), usable, scores
        ):
            # Keep string and numeric IDs distinct, including bool/int and int/float.
            key = (type(sample_id), sample_id, layer)
            if key in seen:
                raise ValueError(f"{continuation_path} has duplicate sentence means for sample {sample_id!r}, layer {layer}")
            seen.add(key)
            layers.add(layer)
            if keep:
                continuations[key] = values.astype(np.float64, copy=True)
    del seen

    # Only sample/layer aggregates survive between prompt batches.
    prompt_totals = {}
    aggregations = [("sample_id", "count")] + [(name, "sum") for name in score_columns]
    for batch in _partition_batches(
        prompt_path, model, source_language, score_columns, batch_size, continuation=False
    ):
        _posterior_matrix(batch, score_columns, prompt_path)
        layers.update(pc.unique(batch.column("layer")).to_pylist())
        if not continuations:
            continue
        grouped = (
            pa.Table.from_batches([batch])
            .group_by(["sample_id", "layer"], use_threads=False)
            .aggregate(aggregations)
        )
        sums = np.column_stack([grouped[f"{name}_sum"].to_numpy() for name in score_columns])
        for sample_id, layer, count, values in zip(
            grouped["sample_id"].to_pylist(), grouped["layer"].to_pylist(), grouped["sample_id_count"].to_pylist(), sums
        ):
            key = (type(sample_id), sample_id, layer)
            if key in continuations:
                if key in prompt_totals:
                    previous, previous_count = prompt_totals[key]
                    previous += values
                    prompt_totals[key] = previous, previous_count + count
                else:
                    prompt_totals[key] = values.astype(np.float64, copy=True), count

    if not prompt_totals:
        return None

    sorted_layers = np.array(sorted(layers), dtype=np.int64)
    layer_index = {layer: index for index, layer in enumerate(sorted_layers)}
    shape = (len(candidates), len(sorted_layers))
    prompt_sums = np.zeros(shape, dtype=np.float64)
    continuation_sums = np.zeros(shape, dtype=np.float64)
    paired_counts = np.zeros(len(sorted_layers), dtype=np.int64)
    for key, (total, count) in prompt_totals.items():
        index = layer_index[key[2]]
        prompt_sums[:, index] += total / count
        continuation_sums[:, index] += continuations[key]
        paired_counts[index] += 1

    prompt = np.full(shape, np.nan)
    continuation = np.full(shape, np.nan)
    np.divide(prompt_sums, paired_counts, out=prompt, where=paired_counts > 0)
    np.divide(continuation_sums, paired_counts, out=continuation, where=paired_counts > 0)
    return AffinitySummary(
        source_language, list(candidates), sorted_layers, prompt, continuation, continuation - prompt, paired_counts
    )
