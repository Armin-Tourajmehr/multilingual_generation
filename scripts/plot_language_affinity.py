#!/usr/bin/env python3
"""Plot paired prompt and continuation language posteriors on CPU."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.affinity import AffinitySummary, summarize_language
from src.affinity_plot import save_affinity_report
from src.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "config.yaml")
    parser.add_argument("--results-dir", type=Path, help="Override the directory containing the saved model results.")
    parser.add_argument("--output-dir", type=Path, help="Write reports under this directory instead of the results directory.")
    parser.add_argument("--models", nargs="+", help="Model keys to plot. Defaults to the models enabled in the config.")
    parser.add_argument("--languages", nargs="+", help="Source languages to plot. All configured candidates remain on the y-axis.")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    candidates = cfg["languages"]
    sources = args.languages or candidates
    models = args.models or [name for name, spec in cfg["models"].items() if spec.get("enabled", False)]
    for kind, selected, configured in (("model", models, cfg["models"]), ("language", sources, candidates)):
        unknown = set(selected).difference(configured)
        if unknown:
            parser.error(f"Unknown {kind} keys: {', '.join(sorted(unknown))}")
        if len(selected) != len(set(selected)):
            parser.error(f"Select each {kind} only once.")

    results_dir = args.results_dir
    if results_dir is None:
        results_dir = Path(cfg["_repo_root"]) / cfg["outputs"]["root_dir"] / cfg["outputs"]["result_dir"]
    output_dir = args.output_dir or results_dir
    reports: dict[str, list[AffinitySummary]] = {}
    for model in models:
        model_dir = results_dir / model
        summaries = []
        for language in sources:
            print(f"[{model}/{language}] Reading saved state scores.", flush=True)
            prompt_path = model_dir / "prompt_token_posteriors" / f"{language}.parquet"
            continuation_path = model_dir / "sentence_level_posteriors" / f"{language}.parquet"
            try:
                summary = summarize_language(
                    prompt_path, continuation_path, model=model, source_language=language, candidates=candidates,
                )
            except FileNotFoundError as exc:
                parser.error(f"{exc}. Use results from a completed state-analysis run.")
            except (OSError, ValueError) as exc:
                parser.error(f"[{model}/{language}] {exc}")
            if summary is None:
                print(f"[{model}/{language}] No paired items with usable scores. Skipping.", file=sys.stderr)
            else:
                summaries.append(summary)
        if not summaries:
            parser.error(f"[{model}] No paired prompt and continuation scores were found for the selected languages.")
        reports[model] = summaries

    # Keep the difference colors comparable across every page and model in this run.
    difference_limit = max(
        float(np.nanmax(np.abs(summary.difference)))
        for summaries in reports.values()
        for summary in summaries
    )
    difference_limit = max(difference_limit, 0.01)
    for model, summaries in reports.items():
        path = output_dir / model / "figures" / "figure1_language_affinity.pdf"
        save_affinity_report(path, model, summaries, cfg["language_names"], difference_limit)
        print(f"[{model}] Wrote {len(summaries)} page(s) to {path.resolve()}.")


if __name__ == "__main__":
    main()
