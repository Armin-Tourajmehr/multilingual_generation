"""CPU reports of GMM assignments by input language."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .token_reports import atomic_report_path, iter_token_batches


@dataclass(frozen=True)
class LanguageConfusionSummary:
    phase: str
    candidates: tuple[str, ...]
    layers: np.ndarray
    counts: np.ndarray

    @property
    def pooled_counts(self) -> np.ndarray:
        return self.counts.sum(axis=0, dtype=np.int64)


def summarize_language_confusion(
    model_dir: Path,
    *,
    model: str,
    candidates: list[str],
    phase: str,
    layers: frozenset[int] | None = None,
) -> LanguageConfusionSummary:
    """Count argmax assignments per transformer layer without keeping token labels."""
    n_languages = len(candidates)
    by_layer: dict[int, np.ndarray] = {}
    for source_index, layer_ids, scores, _ in iter_token_batches(
        model_dir, model=model, candidates=candidates, phase=phase, layers=layers,
    ):
        predictions = scores.argmax(axis=1)
        layer_numbers, layer_indices = np.unique(layer_ids, return_inverse=True)
        codes = layer_indices * n_languages + predictions
        totals = np.bincount(codes, minlength=len(layer_numbers) * n_languages).reshape(-1, n_languages)
        for layer, candidate_counts in zip(layer_numbers.tolist(), totals):
            if layer not in by_layer:
                by_layer[layer] = np.zeros((n_languages, n_languages), dtype=np.int64)
            by_layer[layer][source_index] += candidate_counts

    if not by_layer:
        raise ValueError(f"No scored {phase} tokens in the requested layers")
    sorted_layers = np.array(sorted(by_layer), dtype=np.int64)
    counts = np.stack([by_layer[layer] for layer in sorted_layers])
    return LanguageConfusionSummary(phase, tuple(candidates), sorted_layers, counts)


def _row_rates(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=-1, keepdims=True)
    rates = np.full(counts.shape, np.nan, dtype=np.float64)
    return np.divide(counts, totals, out=rates, where=totals > 0)


def _write_csv(path: Path, model: str, summaries: list[LanguageConfusionSummary]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", "phase", "scope", "layer", "source_language", "candidate_language",
                         "count", "row_total", "row_rate"))
        for summary in summaries:
            matrices = [("layer", int(layer), matrix) for layer, matrix in zip(summary.layers, summary.counts)]
            matrices.append(("pooled", None, summary.pooled_counts))
            for scope, layer, matrix in matrices:
                totals = matrix.sum(axis=1)
                rates = _row_rates(matrix)
                for source_index, source in enumerate(summary.candidates):
                    for candidate_index, candidate in enumerate(summary.candidates):
                        rate = rates[source_index, candidate_index]
                        writer.writerow((
                            model, summary.phase, scope, layer, source, candidate,
                            int(matrix[source_index, candidate_index]), int(totals[source_index]),
                            float(rate) if np.isfinite(rate) else None,
                        ))


def _layer_label(layers: np.ndarray) -> str:
    first, last = int(layers[0]), int(layers[-1])
    if first == last:
        return f"Layer {first}"
    if len(layers) == last - first + 1:
        return f"Layers {first}–{last}"
    if len(layers) <= 10:
        return "Layers " + ", ".join(map(str, layers))
    return f"{len(layers)} selected layers ({first}–{last})"


def _write_pdf(path: Path, model: str, summaries: list[LanguageConfusionSummary]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.ticker import PercentFormatter

    with plt.rc_context({"pdf.fonttype": 42, "font.family": "DejaVu Sans"}):
        with PdfPages(path, metadata={"Title": f"GMM language confusion matrices ({model})"}) as pdf:
            for summary in summaries:
                pooled = summary.pooled_counts
                rates = _row_rates(pooled)
                phase_label = "Prompt tokens" if summary.phase == "prompt" else "Generated tokens"
                phase_caption = (
                    "Prompt scores are in-sample for the Aya GMM fit. " if summary.phase == "prompt"
                    else "Generated tokens are grouped by their prompt language. "
                )
                n_languages = len(summary.candidates)
                size = max(7.6, 5.2 + 0.24 * n_languages)
                fig = plt.figure(figsize=(size + 1.0, size))
                try:
                    height = 0.62
                    width = height * size / (size + 1.0)
                    ax = fig.add_axes([0.175, 0.20, width, height])
                    color_axis = fig.add_axes([0.215 + width, 0.20, 0.024, height])
                    fig.text(0.135, 0.94, "GMM language confusion", fontsize=19, weight="bold")
                    fig.text(0.135, 0.895, f"{model}  |  {phase_label}  |  {_layer_label(summary.layers)}",
                             fontsize=11, color="#394351")
                    cmap = matplotlib.colormaps["Blues"].with_extremes(bad="#e4e7eb")
                    image = ax.imshow(np.ma.masked_invalid(rates), cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
                    ax.set_xticks(range(n_languages), summary.candidates, rotation=45, ha="right")
                    ax.set_yticks(range(n_languages), summary.candidates)
                    ax.set_xlabel("GMM argmax language")
                    ax.set_ylabel("Aya input language")
                    ax.tick_params(axis="both", length=0, labelsize=9)
                    for source, row in enumerate(rates):
                        for candidate, rate in enumerate(row):
                            if np.isfinite(rate) and (source == candidate or rate >= 0.1):
                                ax.text(candidate, source, f"{rate:.0%}", ha="center", va="center",
                                        fontsize=10 if n_languages <= 10 else 7,
                                        color="white" if rate >= 0.72 else "#0d1720")
                    bar = fig.colorbar(image, cax=color_axis, format=PercentFormatter(1))
                    bar.set_label("Share within the input language")
                    bar.outline.set_visible(False)
                    fig.text(0.135, 0.101,
                             f"{int(pooled.sum()):,} token assignments pooled across the selected layers. "
                             "Each row is normalized separately.", fontsize=9, color="#394351")
                    fig.text(0.135, 0.067,
                             phase_caption + "The CSV includes each layer and the pooled counts and rates.",
                             fontsize=9, color="#394351")
                    pdf.savefig(fig)
                finally:
                    plt.close(fig)


def save_language_confusion_report(
    pdf_path: Path,
    csv_path: Path,
    *,
    model: str,
    summaries: list[LanguageConfusionSummary],
) -> None:
    """Save pooled phase heatmaps plus per-layer and pooled counts and rates."""
    if not summaries:
        raise ValueError("Select at least one phase for the confusion report")
    if any(not np.array_equal(summary.layers, summaries[0].layers) for summary in summaries[1:]):
        raise ValueError("Prompt and generated token posteriors have different selected layer sets")
    with atomic_report_path(csv_path) as temp_csv, atomic_report_path(pdf_path) as temp_pdf:
        _write_csv(temp_csv, model, summaries)
        _write_pdf(temp_pdf, model, summaries)
