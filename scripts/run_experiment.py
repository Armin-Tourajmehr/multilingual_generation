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

    seed = int(cfg["project"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    run(cfg)


if __name__ == "__main__":
    main()
