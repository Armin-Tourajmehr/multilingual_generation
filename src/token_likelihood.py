"""Stream and plot per-token posterior ratios on CPU."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator, PercentFormatter, StrMethodFormatter
import numpy as np

from .token_reports import atomic_report_path, iter_token_batches, posterior_ratios


HISTOGRAM_EDGES = np.arange(21, dtype=np.float64) / 20
PHASE_LABELS = {"prompt": "Prompt tokens", "generated": "Generated tokens"}


@dataclass
class RatioCounts:
    below_one: int = 0
    equal_one: int = 0
    above_one: int = 0
    undefined: int = 0
    infinite: int = 0

    @property
    def defined(self) -> int:
        return self.below_one + self.equal_one + self.above_one

    @property
    def total(self) -> int:
        return self.defined + self.undefined

    def add(self, ratios: np.ndarray) -> None:
        self.below_one += int(np.count_nonzero(ratios < 1))
        self.equal_one += int(np.count_nonzero(ratios == 1))
        self.above_one += int(np.count_nonzero(ratios > 1))
        self.undefined += int(np.count_nonzero(np.isnan(ratios)))
        self.infinite += int(np.count_nonzero(np.isposinf(ratios)))


@dataclass(frozen=True)
class TokenLikelihoodSummary:
    phase: str
    layers: dict[int, RatioCounts]
    histogram: np.ndarray


def summarize_token_likelihood_ratios(
    model_dir: Path,
    *,
    model: str,
    candidates: list[str],
    phase: str,
    layers: frozenset[int] | None = None,
) -> TokenLikelihoodSummary:
    """Pool all foreign-candidate comparisons without retaining token rows."""
    if len(candidates) < 2:
        raise ValueError("Token ratios require at least two candidate languages.")
    counts: dict[int, RatioCounts] = {}
    histogram = np.zeros(len(HISTOGRAM_EDGES) - 1, dtype=np.int64)
    for source_index, layer_ids, scores, _ in iter_token_batches(
        model_dir, model=model, candidates=candidates, phase=phase, layers=layers,
    ):
        ratios = posterior_ratios(scores, source_index)
        for layer in np.unique(layer_ids):
            counts.setdefault(int(layer), RatioCounts()).add(ratios[layer_ids == layer])
        histogram += np.histogram(ratios[ratios < 1], bins=HISTOGRAM_EDGES)[0]
    if not counts:
        raise ValueError(f"[{model}/{phase}] No token scores matched the selected transformer layers.")
    return TokenLikelihoodSummary(phase, dict(sorted(counts.items())), histogram)


def _draw_ratio_page(fig, model: str, summary: TokenLikelihoodSummary, language_count: int) -> None:
    left = 0.085
    fig.text(left, 0.942, "Token posterior ratios by layer", fontsize=18, weight="bold")
    phase_label = PHASE_LABELS[summary.phase]
    fig.text(left, 0.894, f"{model}  |  {phase_label}  |  {language_count} configured languages", fontsize=11, color="#394351")
    defined = sum(counts.defined for counts in summary.layers.values())
    undefined = sum(counts.undefined for counts in summary.layers.values())
    below_one = int(summary.histogram.sum())
    fig.text(left, 0.852, f"{defined:,} defined pairs  |  {undefined:,} undefined pairs", fontsize=9, color="#394351")

    layer_ax, hist_ax = fig.subplots(1, 2, gridspec_kw={"width_ratios": (1.38, 1)})
    fig.subplots_adjust(left=left, right=0.97, bottom=0.26, top=0.77, wspace=0.29)
    layers = np.asarray(list(summary.layers))
    counts = list(summary.layers.values())
    denominators = np.asarray([item.defined for item in counts], dtype=np.float64)
    bottom = np.zeros(len(layers))
    for attribute, label, color in (
        ("below_one", "Ratio < 1", "#2b7ab3"),
        ("equal_one", "Ratio = 1", "#a4afbd"),
        ("above_one", "Ratio > 1", "#e59a45"),
    ):
        values = np.asarray([getattr(item, attribute) for item in counts])
        fractions = np.divide(values, denominators, out=np.zeros_like(denominators), where=denominators > 0)
        layer_ax.bar(layers, fractions, bottom=bottom, width=0.84, color=color, label=label)
        bottom += fractions
    tick_step = max(1, (len(layers) + 11) // 12)
    layer_ax.set_xticks(sorted({*layers[::tick_step], layers[-1]}))
    layer_ax.set_xlim(layers[0] - 0.66, layers[-1] + 0.66)
    layer_ax.set_ylim(0, 1)
    layer_ax.set_yticks(np.linspace(0, 1, 6))
    layer_ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    layer_ax.set_xlabel("Transformer layer (native index)")
    layer_ax.set_ylabel("Share of defined pairs")
    layer_ax.set_title("A   Ratio categories by layer", loc="left", fontsize=12, weight="bold", pad=8)
    layer_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncols=3, frameon=False, fontsize=9)

    hist_ax.bar(
        HISTOGRAM_EDGES[:-1], summary.histogram, width=np.diff(HISTOGRAM_EDGES), align="edge",
        color="#2b7ab3", edgecolor="white", linewidth=0.45,
    )
    hist_ax.set_xlim(0, 1)
    hist_ax.set_ylim(bottom=0, top=max(1, int(summary.histogram.max()) * 1.10))
    hist_ax.set_xticks(np.linspace(0, 1, 6))
    hist_ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    hist_ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    hist_ax.set_xlabel("Posterior ratio P(A) / P(B)")
    hist_ax.set_ylabel("Pairs (count)")
    hist_ax.set_title("B   Ratios below 1 across selected layers", loc="left", fontsize=12, weight="bold", pad=8)
    hist_ax.text(0.5, -0.20, f"{below_one:,} pairs below 1", transform=hist_ax.transAxes, ha="center", va="top", fontsize=9)

    for ax in (layer_ax, hist_ax):
        ax.tick_params(axis="both", labelsize=9, length=3, color="#59636e")
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e2e7ee", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#aeb8c4")

    fig.text(
        left, 0.086, "Each token-layer pair contributes one ratio for each foreign candidate B. A is the Aya prompt language.",
        fontsize=9, color="#394351",
    )
    fig.text(
        left, 0.060, "Positive / zero counts above 1. Zero / zero is undefined and excluded from layer percentages.",
        fontsize=9, color="#394351",
    )
    state_note = (
        "Prompt scores are in-sample for the Aya GMM fit." if summary.phase == "prompt"
        else "Generated scores use the state after consuming each token. A remains the prompt language."
    )
    fig.text(left, 0.034, state_note, fontsize=9, color="#394351")


def write_token_likelihood_reports(
    output_dir: Path,
    *,
    model: str,
    summaries: list[TokenLikelihoodSummary],
    language_count: int,
) -> tuple[Path, Path, Path]:
    """Write one PDF page per phase and the layer and histogram CSVs."""
    if not summaries:
        raise ValueError("At least one token ratio summary is required.")
    selected_layers = set(summaries[0].layers)
    if any(set(summary.layers) != selected_layers for summary in summaries[1:]):
        raise ValueError("Prompt and generated token scores have different selected transformer layers.")
    pdf_path = output_dir / "token_likelihood_ratios.pdf"
    layer_path = output_dir / "token_likelihood_ratios.csv"
    histogram_path = output_dir / "token_likelihood_histogram.csv"
    with (
        atomic_report_path(pdf_path) as temporary_pdf,
        atomic_report_path(layer_path) as temporary_layers,
        atomic_report_path(histogram_path) as temporary_histogram,
    ):
        with plt.rc_context({"pdf.fonttype": 42, "font.family": "DejaVu Sans"}):
            with PdfPages(temporary_pdf, metadata={"Title": f"Token posterior ratios by layer ({model})"}) as pdf:
                for summary in summaries:
                    fig = plt.figure(figsize=(13.6, 6.5))
                    try:
                        _draw_ratio_page(fig, model, summary, language_count)
                        pdf.savefig(fig)
                    finally:
                        plt.close(fig)

        with temporary_layers.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "model", "phase", "layer", "total_pairs", "defined_pairs", "undefined_pairs", "infinite_pairs",
                "below_one_pairs", "equal_one_pairs", "above_one_pairs",
                "below_one_fraction", "equal_one_fraction", "above_one_fraction",
            ])
            for summary in summaries:
                for layer, counts in summary.layers.items():
                    categories = [counts.below_one, counts.equal_one, counts.above_one]
                    fractions = [count / counts.defined if counts.defined else "" for count in categories]
                    writer.writerow([
                        model, summary.phase, layer, counts.total, counts.defined, counts.undefined, counts.infinite,
                        *categories, *fractions,
                    ])

        with temporary_histogram.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["model", "phase", "bin_lower", "bin_upper", "pair_count"])
            for summary in summaries:
                for lower, upper, count in zip(HISTOGRAM_EDGES[:-1], HISTOGRAM_EDGES[1:], summary.histogram):
                    writer.writerow([model, summary.phase, f"{lower:.2f}", f"{upper:.2f}", int(count)])
    return pdf_path, layer_path, histogram_path
