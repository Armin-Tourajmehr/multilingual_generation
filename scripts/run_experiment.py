#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random

# Reduce CUDA allocator fragmentation on long-running Kaggle jobs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch

from src.config import load_config
from src.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GPU-only multilingual representation experiment.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--pca-path",
        default=None,
        help="Path where PCA layer files are stored/reused. Defaults to outputs/pca_data.",
    )
    parser.add_argument(
        "--force-refit-pca",
        action="store_true",
        help="Recompute PCA even when a complete PCA directory already exists.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required. On Kaggle, enable Settings -> Accelerator -> GPU and restart the session."
        )

    print("CUDA preflight: PASS")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Visible GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name} | {props.total_memory / (1024**3):.1f} GiB")

    seed = int(cfg["project"]["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.set_grad_enabled(False)
    run(
        cfg,
        pca_path=args.pca_path or cfg["runtime"].get("pca_data_path"),
        force_refit_pca=args.force_refit_pca,
    )


if __name__ == "__main__":
    main()
