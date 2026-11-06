"""Headless rendering for the paired Aya language affinity report."""
from __future__ import annotations

from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter, StrMethodFormatter
import numpy as np

if TYPE_CHECKING:
    from .affinity import AffinitySummary


def _language_label(code: str, language_names: dict[str, str]) -> str:
    name = language_names.get(code, code)
    return f"{name} ({code})" if name != code else code


def _draw_page(fig, model: str, summary: AffinitySummary, language_names: dict[str, str], limit: float) -> None:
    left = 0.145
    fig.text(left, 0.952, "Figure 1. Language affinity across layers", fontsize=19, weight="bold")
    source = _language_label(summary.source_language, language_names)
    fig.text(left, 0.913, f"{model}  |  Prompts in {source}", fontsize=12, color="#394351")
    grid = fig.add_gridspec(
        3, 3, left=left, right=0.973, top=0.855, bottom=0.151,
        height_ratios=(5.8, 0.25, 1.25), wspace=0.14, hspace=0.48,
    )

    layer_count = len(summary.layers)
    tick_stride = max(1, (layer_count + 11) // 12)
    tick_positions = sorted({*range(0, layer_count, tick_stride), layer_count - 1})
    tick_labels = ["0*" if summary.layers[position] == 0 else str(summary.layers[position]) for position in tick_positions]
    embedding_positions = np.flatnonzero(summary.layers == 0)
    language_labels = [_language_label(code, language_names) for code in summary.candidates]

    probability_cmap = matplotlib.colormaps["viridis"].with_extremes(bad="#e2e5e9")
    difference_cmap = matplotlib.colormaps["RdBu_r"].with_extremes(bad="#e2e5e9")
    probability_norm = Normalize(vmin=0, vmax=100)
    difference_norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    panels = (
        ("A   Prompt", summary.prompt, probability_cmap, probability_norm),
        ("B   Continuation", summary.continuation, probability_cmap, probability_norm),
        ("C   Continuation − prompt", summary.difference, difference_cmap, difference_norm),
    )
    images = []
    for column, (title, values, cmap, norm) in enumerate(panels):
        ax = fig.add_subplot(grid[0, column])
        images.append(ax.imshow(np.ma.masked_invalid(values * 100), cmap=cmap, norm=norm, aspect="auto", interpolation="nearest"))
        ax.set_title(title, loc="left", fontsize=12, weight="bold", pad=9)
        ax.set_xticks(tick_positions, tick_labels)
        ax.set_xlabel("Native model layer", fontsize=9)
        ax.tick_params(axis="both", labelsize=9, length=0, pad=4)
        if column == 0:
            ax.set_ylabel("Candidate language", fontsize=10)
            ax.set_yticks(range(len(language_labels)), language_labels)
            for label, code in zip(ax.get_yticklabels(), summary.candidates):
                if code == summary.source_language:
                    label.set_weight("bold")
        else:
            ax.set_yticks([])
        if embedding_positions.size:
            ax.axvline(embedding_positions[0] + 0.5, color="white", linewidth=0.7)
        for spine in ax.spines.values():
            spine.set_color("#c6ccd4")
            spine.set_linewidth(0.6)

    probability_bar = fig.colorbar(
        images[0], cax=fig.add_subplot(grid[1, :2]), orientation="horizontal",
        ticks=[0, 25, 50, 75, 100], format=PercentFormatter(xmax=100, decimals=0),
    )
    probability_bar.set_label("Mean posterior probability (panels A and B)", fontsize=9)
    difference_bar = fig.colorbar(
        images[2], cax=fig.add_subplot(grid[1, 2]), orientation="horizontal",
        ticks=np.linspace(-limit, limit, 5),
        format=FuncFormatter(lambda value, _: "0" if value == 0 else f"{value:+.3g}"),
    )
    difference_bar.set_label("Difference (percentage points)", fontsize=9)
    for bar in (probability_bar, difference_bar):
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=9, length=3, color="#59636e")

    counts = summary.paired_counts
    minimum, maximum = int(counts.min()), int(counts.max())
    coverage_text = f"{minimum:,} each" if minimum == maximum else f"{minimum:,} to {maximum:,}"
    coverage = fig.add_subplot(grid[2, :])
    coverage.bar(np.arange(layer_count), counts, width=0.76, color="#70849b", zorder=2)
    coverage.set_title(f"Paired Aya samples per layer ({coverage_text})", loc="left", fontsize=10, weight="bold", pad=4)
    coverage.set_xlim(-0.5, layer_count - 0.5)
    coverage.set_ylim(0, max(1, maximum * 1.12))
    coverage.set_xticks(tick_positions, tick_labels)
    coverage.yaxis.set_major_locator(MaxNLocator(nbins=2, integer=True))
    coverage.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    coverage.tick_params(axis="both", labelsize=8, length=0, pad=4)
    coverage.grid(axis="y", color="#e5e9ef", linewidth=0.6)
    for side in ("top", "right", "left"):
        coverage.spines[side].set_visible(False)
    coverage.spines["bottom"].set_color("#c6ccd4")

    fig.text(
        left, 0.088,
        "Prompt scores (A) are in-sample for the Aya GMM fit. B shows the generated continuations to those prompts, grouped by prompt language.",
        fontsize=9, color="#394351",
    )
    fig.text(
        left, 0.064,
        "C shows B − A. Values average sample-level scores over paired samples at each layer. Grey cells indicate missing scores. * Layer 0 is the embedding output.",
        fontsize=9, color="#394351",
    )


def save_affinity_report(
    path: Path,
    model: str,
    summaries: list[AffinitySummary],
    language_names: dict[str, str],
    difference_limit: float,
) -> None:
    """Write a model PDF with one page per source and a shared difference scale."""
    if not summaries:
        raise ValueError("An affinity report needs at least one source summary.")
    limit = float(difference_limit) * 100
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("difference_limit must be finite and positive.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=f".{path.stem}.") as directory:
        temporary_path = Path(directory) / path.name
        with plt.rc_context({"pdf.fonttype": 42, "font.family": "DejaVu Sans"}):
            with PdfPages(temporary_path, metadata={"Title": f"Figure 1. Language affinity across layers ({model})"}) as pdf:
                for summary in summaries:
                    fig = plt.figure(figsize=(16.5, 10.5))
                    try:
                        _draw_page(fig, model, summary, language_names, limit)
                        pdf.savefig(fig)
                    finally:
                        plt.close(fig)
        temporary_path.replace(path)
