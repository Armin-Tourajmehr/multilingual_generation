#!/usr/bin/env python3
"""Plot layerwise prompt-language NMI from saved Parquet files on CPU."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.affinity import PromptNMISummary, summarize_prompt_nmi
from src.affinity_plot import save_prompt_nmi_plot
from src.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "config.yaml")
    parser.add_argument("--results-dir", type=Path, help="Override the directory containing the saved model results.")
    parser.add_argument("--output-dir", type=Path, help="Write the PDF to this directory instead of the results figures directory.")
    parser.add_argument("--models", nargs="+", help="Model keys to plot. Defaults to the models enabled in the config.")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    models = args.models or [name for name, spec in cfg["models"].items() if spec.get("enabled", False)]
    unknown = set(models).difference(cfg["models"])
    if unknown:
        parser.error(f"Unknown model keys: {', '.join(sorted(unknown))}")
    if len(models) != len(set(models)):
        parser.error("Select each model only once.")

    results_dir = args.results_dir
    if results_dir is None:
        results_dir = Path(cfg["_repo_root"]) / cfg["outputs"]["root_dir"] / cfg["outputs"]["result_dir"]
    candidates = cfg["languages"]
    summaries: dict[str, PromptNMISummary] = {}
    for model in models:
        print(f"[{model}] Pooling prompt token assignments across all {len(candidates)} languages.", flush=True)
        try:
            summary = summarize_prompt_nmi(results_dir / model, model=model, candidates=candidates)
        except FileNotFoundError as exc:
            parser.error(f"{exc}. Use a completed state-analysis run with prompt Parquet files for every language.")
        except (OSError, ValueError) as exc:
            parser.error(f"[{model}] {exc}")
        summaries[model] = summary

    output_dir = args.output_dir or results_dir / "figures"
    path = output_dir / "prompt_language_nmi.pdf"
    save_prompt_nmi_plot(path, summaries, len(candidates))
    print(f"Wrote {len(summaries)} model curve(s) to {path.resolve()}.")


if __name__ == "__main__":
    main()
