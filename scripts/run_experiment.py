#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from src.config import load_config
from src.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the main multilingual representation experiment.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    runtime = cfg["runtime"]
    if runtime.get("require_gpu", False):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU is required by this configuration, but torch.cuda.is_available() is False. "
                "On Kaggle, enable Settings -> Accelerator -> GPU and restart the session."
            )
        print("CUDA preflight: PASS")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Visible GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} | {props.total_memory / (1024**3):.1f} GiB")
    else:
        print(f"CUDA available: {torch.cuda.is_available()}")

    seed = int(cfg["project"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    run(cfg)


if __name__ == "__main__":
    main()
