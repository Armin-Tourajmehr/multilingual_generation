from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config


def test_main_config_is_full_experiment():
    cfg = load_config(ROOT / "configs" / "config.yaml")
    assert len(cfg["languages"]) == 18
    assert cfg["representation"]["pca_dim"] == 700
    assert cfg["wikipedia"]["max_samples_per_language"] == 3000
    assert cfg["generation"]["max_new_tokens"] == 64
    assert cfg["models"]["mgpt"]["enabled"] is True
    assert cfg["models"]["bloom"]["enabled"] is True
    assert cfg["models"]["qwen"]["enabled"] is False
    assert "gemma3_4b" not in cfg["models"]
    assert "train_samples_per_language" not in cfg["data"]


def test_runtime_configuration():
    cfg = load_config(ROOT / "configs" / "config.yaml")
    runtime = cfg["runtime"]
    assert runtime["device_map"] is None
    assert runtime["wikipedia_batch_size"] == 32
    assert runtime["evaluation_batch_size"] == 8
    assert runtime["device"] == "cuda"
    assert runtime["require_gpu"] is True
    assert runtime["mixed_precision"] == "fp16"
