#!/usr/bin/env python3
from __future__ import annotations

import argparse

from src.config import load_config
from src.pipeline import run


def main():
    parser = argparse.ArgumentParser(description="Run multilingual representation analysis.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
