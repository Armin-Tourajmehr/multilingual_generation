from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.manifold import SpectralEmbedding

from .gmm import GPULanguageGMM


def save_language_manifold(
    gmms: list[GPULanguageGMM],
    languages: list[str],
    language_names: dict[str, str],
    output_dir: str | Path,
    cfg: dict[str, Any],
    model_name: str,
) -> pd.DataFrame:
    """Embed one GMM centroid per language with Spectral Embedding at each layer.

    This deliberately uses only L x PCA-dimension centroid matrices, rather than
    retaining every token vector, so the visualization has negligible memory cost.
    The plot describes language-level centroids, not within-language dispersion.
    """
    opts = cfg.get("spectral_embedding", {})
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Do not leave layer PNGs from a previous model/configuration behind.
    for stale in out.glob("layer_*.png"):
        stale.unlink()
    for stale_name in ("spectral_embedding_all_layers.pdf", "spectral_embedding_coordinates.csv", "metadata.json", "error.txt"):
        stale = out / stale_name
        if stale.exists():
            stale.unlink()
    if len(languages) < 3:
        raise ValueError("Spectral Embedding needs at least three configured languages.")
    n_neighbors = min(int(opts.get("n_neighbors", 5)), len(languages) - 1)
    n_components = int(opts.get("n_components", 2))
    if n_components != 2:
        raise ValueError("The current visualization/export path expects n_components=2.")

    all_rows: list[dict[str, Any]] = []
    failed_layers: list[dict[str, str]] = []
    pdf_path = out / "spectral_embedding_all_layers.pdf"
    with PdfPages(pdf_path) as pdf:
        for layer, gmm in enumerate(gmms):
            centers = gmm.means.detach().float().cpu().numpy()
            if centers.shape[0] != len(languages):
                raise ValueError(
                    f"Layer {layer}: {centers.shape[0]} GMM centroids for {len(languages)} languages."
                )
            try:
                embedding = SpectralEmbedding(
                    n_components=2,
                    affinity=opts.get("affinity", "nearest_neighbors"),
                    n_neighbors=n_neighbors,
                    random_state=int(opts.get("random_state", cfg.get("project", {}).get("seed", 42))),
                ).fit_transform(centers)
            except Exception as exc:
                failed_layers.append({"layer": str(layer), "error": str(exc)})
                warnings.warn(f"Spectral Embedding failed for {model_name}, layer {layer}: {exc}")
                continue

            layer_rows = []
            for i, lang in enumerate(languages):
                row = {
                    "model": model_name,
                    "layer": layer,
                    "language": lang,
                    "language_name": language_names.get(lang, lang),
                    "x": float(embedding[i, 0]),
                    "y": float(embedding[i, 1]),
                    "embedding_source": "PCA-space GMM centroid fitted on Aya token vectors",
                }
                layer_rows.append(row)
                all_rows.append(row)

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(embedding[:, 0], embedding[:, 1])
            for i, lang in enumerate(languages):
                ax.annotate(language_names.get(lang, lang), (embedding[i, 0], embedding[i, 1]), fontsize=8)
            ax.set_title(f"{model_name}: language-centroid Spectral Embedding — layer {layer}")
            ax.set_xlabel("Spectral component 1")
            ax.set_ylabel("Spectral component 2")
            fig.tight_layout()
            fig.savefig(out / f"layer_{layer:03d}.png", dpi=160)
            pdf.savefig(fig)
            plt.close(fig)

    coordinates = pd.DataFrame(all_rows)
    coordinates.to_csv(out / "spectral_embedding_coordinates.csv", index=False)
    metadata = {
        "model": model_name,
        "method": "sklearn.manifold.SpectralEmbedding",
        "n_components": 2,
        "n_neighbors": n_neighbors,
        "affinity": opts.get("affinity", "nearest_neighbors"),
        "random_state": int(opts.get("random_state", cfg.get("project", {}).get("seed", 42))),
        "languages": languages,
        "input": "per-language GMM mean vectors in the Wikipedia-PCA reference space",
        "interpretation": "One point per language and layer; this visualizes language centroids, not all individual samples or token dispersion.",
        "failed_layers": failed_layers,
        "successful_layer_count": int(coordinates["layer"].nunique()) if not coordinates.empty else 0,
    }
    with (out / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return coordinates
