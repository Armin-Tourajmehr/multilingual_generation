"""Summarize cross-language posterior ratios by saved continuation POS tags."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .token_reports import atomic_report_path, iter_token_batches, posterior_ratios


# Universal Dependencies v2 tags, including the miscellaneous category X.
UNIVERSAL_POS = (
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
)
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 1.0)


@dataclass(frozen=True)
class PosLikelihoodSummary:
    model: str
    layers: tuple[int, ...]
    total_token_rows: int
    tagged_rows_by_pos: np.ndarray
    defined_pairs_by_pos: np.ndarray
    undefined_pairs_by_pos: np.ndarray
    dominated_pairs: np.ndarray  # threshold by POS tag
    coverage_by_source: tuple[tuple[str, int, int], ...]  # source, total rows, tagged rows

    @property
    def tagged_token_rows(self) -> int:
        return int(self.tagged_rows_by_pos.sum())


def summarize_pos_likelihood(
    model_dir: Path, *, model: str, candidates: list[str], layers: frozenset[int] | None = None,
) -> PosLikelihoodSummary:
    """Count tagged token-layer pairs with P(source)/P(candidate) below each threshold."""
    if len(candidates) < 2:
        raise ValueError("Cross-language posterior ratios require at least two candidate languages.")
    tags = {tag: index for index, tag in enumerate(UNIVERSAL_POS)}
    tagged_rows = np.zeros(len(tags), dtype=np.int64)
    defined_pairs = np.zeros_like(tagged_rows)
    undefined_pairs = np.zeros_like(tagged_rows)
    dominated = np.zeros((len(THRESHOLDS), len(tags)), dtype=np.int64)
    rows_by_source = np.zeros(len(candidates), dtype=np.int64)
    tagged_by_source = np.zeros_like(rows_by_source)
    observed_layers: set[int] = set()

    for source, layer_ids, scores, batch in iter_token_batches(
        model_dir, model=model, candidates=candidates, phase="generated", layers=layers,
        extra_columns=("upos", "pos_status"),
    ):
        rows_by_source[source] += batch.num_rows
        observed_layers.update(np.unique(layer_ids).tolist())
        codes = np.fromiter(
            (tags.get(tag, -1) if status == "ok" else -1 for tag, status in zip(
                batch.column("upos").to_pylist(), batch.column("pos_status").to_pylist(),
            )),
            dtype=np.intp, count=batch.num_rows,
        )
        usable = codes >= 0
        tagged_by_source[source] += np.count_nonzero(usable)
        if not usable.any():
            continue
        ratios = posterior_ratios(scores[usable], source)
        codes = codes[usable]
        for tag_index in np.unique(codes):
            selected = codes == tag_index
            tag_ratios = ratios[selected]
            tagged_rows[tag_index] += np.count_nonzero(selected)
            undefined = np.count_nonzero(np.isnan(tag_ratios))
            undefined_pairs[tag_index] += undefined
            defined_pairs[tag_index] += tag_ratios.size - undefined
            for threshold_index, threshold in enumerate(THRESHOLDS):
                dominated[threshold_index, tag_index] += np.count_nonzero(tag_ratios < threshold)

    if not tagged_rows.any():
        raise ValueError(
            "No usable continuation UPOS annotations were found in the selected layers. "
            "Run scripts/annotate_outputs.py on these results, then retry."
        )
    return PosLikelihoodSummary(
        model, tuple(sorted(observed_layers)), int(rows_by_source.sum()), tagged_rows, defined_pairs,
        undefined_pairs, dominated,
        tuple((source, int(total), int(tagged)) for source, total, tagged in zip(
            candidates, rows_by_source, tagged_by_source,
        )),
    )


def _write_csv(path: Path, summary: PosLikelihoodSummary) -> None:
    total_tagged = summary.tagged_token_rows
    coverage = {
        "total_token_rows": summary.total_token_rows,
        "tagged_token_rows": total_tagged,
        "excluded_pos_rows": summary.total_token_rows - total_tagged,
        "annotation_coverage": total_tagged / summary.total_token_rows,
        "defined_tagged_pairs": int(summary.defined_pairs_by_pos.sum()),
        "undefined_tagged_pairs": int(summary.undefined_pairs_by_pos.sum()),
    }
    layers = ",".join(map(str, summary.layers))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "model", "phase", "layers", "threshold", "upos", "tagged_token_rows_with_pos",
            "dominated_pairs", "dominated_pairs_all_pos", "defined_pairs_with_pos", "undefined_pairs_with_pos",
            "composition_fraction", "within_pos_fraction", *coverage,
        ])
        for threshold, counts in zip(THRESHOLDS, summary.dominated_pairs):
            all_dominated = int(counts.sum())
            for index, tag in enumerate(UNIVERSAL_POS):
                count = int(counts[index])
                defined = int(summary.defined_pairs_by_pos[index])
                writer.writerow([
                    summary.model, "generated", layers, threshold, tag, int(summary.tagged_rows_by_pos[index]),
                    count, all_dominated, defined, int(summary.undefined_pairs_by_pos[index]),
                    count / all_dominated if all_dominated else "", count / defined if defined else "",
                    *coverage.values(),
                ])


def _write_coverage_csv(path: Path, summary: PosLikelihoodSummary) -> None:
    layers = ",".join(map(str, summary.layers))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "model", "phase", "layers", "input_language", "total_token_rows", "tagged_token_rows",
            "excluded_pos_rows", "annotation_coverage",
        ])
        for source, total, tagged in summary.coverage_by_source:
            writer.writerow([summary.model, "generated", layers, source, total, tagged, total - tagged,
                             tagged / total if total else ""])


def _layer_caption(layers: tuple[int, ...]) -> str:
    if len(layers) == 1:
        return str(layers[0])
    if len(layers) == layers[-1] - layers[0] + 1:
        return f"{layers[0]}-{layers[-1]}"
    if len(layers) <= 10:
        return ", ".join(map(str, layers))
    return f"{len(layers)} selected layers ({layers[0]}-{layers[-1]}, IDs in CSV)"


def _write_pdf(path: Path, summary: PosLikelihoodSummary) -> None:
    from matplotlib import colormaps, rc_context
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter

    observed = np.flatnonzero(summary.tagged_rows_by_pos)
    tag_names = [UNIVERSAL_POS[index] for index in observed]
    counts = summary.dominated_pairs[:, observed].T
    denominators = (
        np.broadcast_to(summary.dominated_pairs.sum(axis=1), counts.shape),
        np.broadcast_to(summary.defined_pairs_by_pos[observed, None], counts.shape),
    )
    titles = ("A   Composition of dominated pairs", "B   Dominated fraction within each POS")
    page_height = max(4.8, 3.2 + 0.33 * len(observed))
    plot_top = 1 - 1.13 / page_height
    plot_bottom = 1.58 / page_height
    with rc_context({"pdf.fonttype": 42, "font.family": "DejaVu Sans"}):
        figure = Figure(figsize=(15, page_height))
        figure.subplots_adjust(left=0.083, right=0.905, top=plot_top, bottom=plot_bottom, wspace=0.12)
        axes = figure.subplots(ncols=2)
        cmap = colormaps["cividis"].with_extremes(bad="#e8ebee")
        for ax, title, denominator in zip(axes, titles, denominators):
            fractions = np.divide(
                counts, denominator, out=np.full(counts.shape, np.nan), where=denominator > 0,
            )
            image = ax.imshow(np.ma.masked_invalid(fractions), cmap=cmap, vmin=0, vmax=1,
                              aspect="auto", interpolation="nearest")
            ax.set_title(title, loc="left", fontsize=12, weight="bold", pad=8)
            ax.set_xticks(range(len(THRESHOLDS)), [f"< {threshold:g}" for threshold in THRESHOLDS])
            ax.set_yticks(range(len(observed)), tag_names)
            ax.set_xlabel("Strict posterior ratio cutoff", fontsize=9, labelpad=6)
            ax.tick_params(length=0, labelsize=9, pad=5)
            for row, column in np.ndindex(fractions.shape):
                fraction = fractions[row, column]
                base = denominator[row, column]
                label = f"{fraction:.1%}" if base else "n/a"
                ax.text(column, row, label, ha="center", va="center", fontsize=8,
                        color="white" if base and fraction < 0.55 else "#141923")
            for spine in ax.spines.values():
                spine.set_visible(False)

        bar = figure.colorbar(image, cax=figure.add_axes([0.929, plot_bottom, 0.012, plot_top - plot_bottom]),
                              ticks=[0, 0.25, 0.5, 0.75, 1], format=PercentFormatter(1, decimals=0))
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=8, length=3)
        figure.text(0.083, 1 - 0.38 / page_height, "POS and cross-language posterior ratios",
                    fontsize=19, weight="bold")
        figure.text(0.083, 1 - 0.69 / page_height,
                    f"{summary.model}  |  Generated token states  |  Transformer layers {_layer_caption(summary.layers)}",
                    fontsize=11, color="#394351")
        tagged = summary.tagged_token_rows
        coverage = tagged / summary.total_token_rows
        excluded = summary.total_token_rows - tagged
        figure.text(0.083, 0.95 / page_height,
                    f"Usable POS tags on {tagged:,} of {summary.total_token_rows:,} rows ({coverage:.1%}). "
                    f"{excluded:,} excluded rows. {summary.defined_pairs_by_pos.sum():,} defined tagged pairs "
                    f"and {summary.undefined_pairs_by_pos.sum():,} undefined 0/0 pairs.", fontsize=9, color="#394351")
        figure.text(0.083, 0.69 / page_height,
                    "A divides by all dominated tagged pairs. B divides by all defined pairs with that POS tag. "
                    "Exact counts are in the companion CSV.", fontsize=9, color="#394351")
        figure.text(0.083, 0.43 / page_height,
                    "Each event is a scored token, layer, and other candidate language. "
                    "The ratio is P(prompt language) / P(other candidate), pooled across sources and targets.",
                    fontsize=9, color="#394351")
        figure.text(0.083, 0.17 / page_height,
                    "POS tags use the prompt language's tagger. Language switches can affect the tags. "
                    "Coverage by prompt language is in the companion CSV.",
                    fontsize=9, color="#394351")
        figure.savefig(path, format="pdf", metadata={"Title": f"POS posterior ratio analysis ({summary.model})"})


def save_pos_likelihood_report(directory: Path, summary: PosLikelihoodSummary) -> tuple[Path, Path, Path]:
    """Save the PDF and companion counts and coverage CSVs after their writers complete."""
    pdf = directory / "pos_likelihood_ratios.pdf"
    csv_path = directory / "pos_likelihood_ratios.csv"
    coverage = directory / "pos_likelihood_coverage.csv"
    with (
        atomic_report_path(pdf) as temporary_pdf,
        atomic_report_path(csv_path) as temporary_csv,
        atomic_report_path(coverage) as temporary_coverage,
    ):
        _write_pdf(temporary_pdf, summary)
        _write_csv(temporary_csv, summary)
        _write_coverage_csv(temporary_coverage, summary)
    return pdf, csv_path, coverage
