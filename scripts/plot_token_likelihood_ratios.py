#!/usr/bin/env python3
"""Plot per-layer token posterior ratio categories and pooled histograms on CPU."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.token_likelihood import summarize_token_likelihood_ratios, write_token_likelihood_reports
from src.token_reports import parse_report_args


def main() -> None:
    options = parse_report_args(__doc__)
    for model in options.models:
        print(f"[{model}] Counting token posterior ratios for every foreign candidate.", flush=True)
        try:
            summaries = [
                summarize_token_likelihood_ratios(
                    options.results_dir / model, model=model, candidates=options.candidates,
                    phase=phase, layers=options.layers,
                )
                for phase in options.phases
            ]
            outputs = write_token_likelihood_reports(
                options.output_dir / model / "figures", model=model, summaries=summaries,
                language_count=len(options.candidates),
            )
        except (OSError, ValueError) as error:
            raise SystemExit(f"[{model}] {error}") from error
        print(f"[{model}] Wrote {', '.join(path.name for path in outputs)} to {outputs[0].parent.resolve()}.")


if __name__ == "__main__":
    main()
