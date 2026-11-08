#!/usr/bin/env python3
"""Analyze POS and cross-language posterior ratios in saved continuations on CPU."""
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pos_likelihood import save_pos_likelihood_report, summarize_pos_likelihood
from src.token_reports import parse_report_args


def main() -> None:
    options = parse_report_args(__doc__, default_phase=None)
    summaries = []
    for model in options.models:
        print(f"[{model}] Reading saved continuation scores and POS tags.", flush=True)
        try:
            summaries.append(summarize_pos_likelihood(
                options.results_dir / model, model=model, candidates=options.candidates, layers=options.layers,
            ))
        except (OSError, ValueError) as exc:
            raise SystemExit(f"[{model}] {exc}") from None

    for summary in summaries:
        directory = options.output_dir / summary.model / "figures"
        paths = save_pos_likelihood_report(directory, summary)
        print(f"[{summary.model}] Tagged {summary.tagged_token_rows:,} of {summary.total_token_rows:,} token rows.")
        for path in paths:
            print(f"[{summary.model}] Wrote {path.resolve()}.")


if __name__ == "__main__":
    main()
