#!/usr/bin/env python3
"""Report GMM language confusion by input language from saved token posteriors."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.language_confusion import save_language_confusion_report, summarize_language_confusion
from src.token_reports import parse_report_args


def main() -> None:
    options = parse_report_args(__doc__)
    for model in options.models:
        figures = options.output_dir / model / "figures"
        pdf_path = figures / "language_confusion.pdf"
        csv_path = figures / "language_confusion.csv"
        try:
            summaries = []
            for phase in options.phases:
                print(f"[{model}] Counting GMM assignments for {phase} tokens.", flush=True)
                summaries.append(summarize_language_confusion(
                    options.results_dir / model, model=model, candidates=options.candidates,
                    phase=phase, layers=options.layers,
                ))
            save_language_confusion_report(pdf_path, csv_path, model=model, summaries=summaries)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"[{model}] {exc}") from None
        print(f"[{model}] Wrote {pdf_path.resolve()} and {csv_path.resolve()}.")


if __name__ == "__main__":
    main()
