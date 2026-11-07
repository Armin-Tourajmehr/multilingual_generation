#!/usr/bin/env python3
"""Plot prompt-language NMI for prompt and generated tokens on CPU."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.affinity import PromptNMISummary, summarize_generated_nmi, summarize_prompt_nmi
from src.affinity_plot import save_prompt_nmi_plot
from src.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "config.yaml")
    parser.add_argument("--results-dir", type=Path, help="Override the directory containing the saved model results.")
    parser.add_argument("--output-dir", type=Path, help="Write the PDF to this directory instead of the results figures directory.")
    parser.add_argument("--models", nargs="+", help="Model keys to plot. Defaults to the models enabled in the config.")
    parser.add_argument("--prompt-only", action="store_true", help="Plot only the prompt panel using prompt-token Parquet files.")
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
    generated_summaries: dict[str, PromptNMISummary] | None = None if args.prompt_only else {}
    for model in models:
        model_dir = results_dir / model
        print(f"[{model}] Pooling prompt token assignments across all {len(candidates)} languages.", flush=True)
        try:
            summary = summarize_prompt_nmi(model_dir, model=model, candidates=candidates)
            if generated_summaries is not None:
                print(f"[{model}] Pooling generated token assignments across all {len(candidates)} prompt languages.", flush=True)
                generated = summarize_generated_nmi(model_dir, model=model, candidates=candidates)
                if not np.array_equal(summary.layers, generated.layers):
                    raise ValueError("Prompt and generated token files have different transformer layer sets.")
                generated_summaries[model] = generated
        except FileNotFoundError as exc:
            parser.error(f"{exc}. Use a completed run with Parquet files for every language in each requested panel.")
        except (OSError, ValueError) as exc:
            parser.error(f"[{model}] {exc}")
        summaries[model] = summary

    output_dir = args.output_dir or results_dir / "figures"
    path = output_dir / "prompt_language_nmi.pdf"
    save_prompt_nmi_plot(path, summaries, len(candidates), generated_summaries=generated_summaries)
    print(f"Wrote NMI for {len(summaries)} model(s) to {path.resolve()}.")


if __name__ == "__main__":
    main()
