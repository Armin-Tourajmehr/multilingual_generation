#!/usr/bin/env python3
"""Add Stanza POS labels to a completed run using CPU resources only."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.postprocess import annotate_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate saved token posteriors with POS labels on CPU.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--models", nargs="+", help="Configured model keys to annotate; defaults to enabled models.")
    parser.add_argument("--batch-size", type=int, default=65536, help="Maximum token rows read per batch (default: 65536).")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    cfg = load_config(args.config)
    results = annotate_outputs(cfg, models=args.models, batch_size=args.batch_size)
    for result in results:
        if result.status == "no_token_file":
            print(f"[{result.model}/{result.language}] No token file; skipped.", flush=True)
        else:
            print(f"[{result.model}/{result.language}] Annotated {result.num_rows:,} token rows from {result.num_samples:,} samples.", flush=True)
    print("CPU POS annotation completed.")


if __name__ == "__main__":
    main()
