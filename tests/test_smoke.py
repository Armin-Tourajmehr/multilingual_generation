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
    assert cfg["representation"]["pca_solver"] == "gpu_covariance"
    assert cfg["runtime"]["pca_data_path"] == "outputs/pca_data"
    assert cfg["outputs"]["gmm_dir"] == "aya_gmms"


def test_runtime_configuration():
    cfg = load_config(ROOT / "configs" / "config.yaml")
    runtime = cfg["runtime"]
    assert runtime["device_map"] is None
    assert runtime["wikipedia_batch_size"] == 8
    assert runtime["evaluation_batch_size"] == 8
    assert runtime["device"] == "cuda"
    assert runtime["require_gpu"] is True
    assert runtime["mixed_precision"] == "fp16"


def test_gpu_math_interfaces():
    import torch
    from src.pca_gpu import GPUPCA
    from src.gmm import GPULanguageGMM

    pca = GPUPCA(
        mean=torch.zeros(4),
        components=torch.eye(4)[:2],
        explained_variance=torch.ones(2),
    )
    assert pca.transform(torch.ones(3, 4)).shape == (3, 2)

    gmm = GPULanguageGMM(
        means=torch.zeros(2, 2),
        variances=torch.ones(2, 2),
        languages=["en", "fa"],
    )
    post = gmm.posterior(torch.zeros(3, 2))
    assert post.shape == (3, 2)
    assert torch.allclose(post.sum(dim=1), torch.ones(3))
