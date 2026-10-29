from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import enabled_models, load_config, validate_config
from src.smoke import build_smoke_config


def test_main_config_has_dynamic_language_and_model_lists():
    cfg = load_config(ROOT / "configs" / "config.yaml")
    assert len(cfg["languages"]) == len(set(cfg["languages"]))
    assert {"zh", "ja"}.issubset(cfg["languages"])
    assert set(cfg["languages"]) == set(cfg["language_names"])
    assert cfg["models"]["mgpt"]["enabled"] is True
    assert cfg["models"]["bloom"]["enabled"] is True
    assert cfg["models"]["qwen"]["enabled"] is False
    assert cfg["models"]["llama3_1_8b"]["enabled"] is False
    assert cfg["representation"]["pca_solver"] == "gpu_covariance"
    assert cfg["runtime"]["pca_data_path"] == "outputs/pca_data"
    assert cfg["outputs"]["gmm_dir"] == "aya_gmms"


def test_runtime_configuration_and_model_overrides():
    cfg = load_config(ROOT / "configs" / "config.yaml")
    runtime = cfg["runtime"]
    assert runtime["device_map"] is None
    assert runtime["wikipedia_batch_size"] == 8
    assert runtime["evaluation_batch_size"] == 8
    assert runtime["generation_batch_size"] == 4
    assert runtime["device"] == "cuda"
    assert runtime["require_gpu"] is True
    assert runtime["mixed_precision"] == "fp16"
    assert cfg["models"]["llama3_1_8b"]["runtime_overrides"]["generation_batch_size"] == 1


def test_smoke_config_isolated_and_small():
    cfg = load_config(ROOT / "configs" / "config.yaml")
    smoke_cfg = build_smoke_config(cfg)
    assert smoke_cfg["data"]["max_samples_per_language"] == 5
    assert smoke_cfg["wikipedia"]["max_samples_per_language"] == 10
    assert smoke_cfg["representation"]["pca_dim"] == 32
    assert smoke_cfg["generation"]["max_new_tokens"] == 8
    assert smoke_cfg["outputs"]["root_dir"] == "outputs/smoke_test"
    assert smoke_cfg["runtime"]["pca_data_path"] == "outputs/smoke_test/pca_data"
    assert [k for k, v in smoke_cfg["models"].items() if v["enabled"]] == ["bloom"]
    # Building a smoke config must not mutate the main config.
    assert cfg["data"]["max_samples_per_language"] == 2000
    assert cfg["representation"]["pca_dim"] == 700


def test_pos_alignment_helper_without_loading_stanza():
    from src.lexical import align_pos_to_token

    words = [{
        "sentence_id": 0,
        "word_id": 0,
        "word_text": "runs",
        "char_start": 4,
        "char_end": 8,
        "upos": "VERB",
        "xpos": "VBZ",
        "feats": "Tense=Pres|Person=3|Number=Sing",
    }]
    result = align_pos_to_token("He runs.", 4, 8, words, "ok")
    assert result["upos"] == "VERB"
    assert result["word_text"] == "runs"
    assert result["pos_status"] == "ok"


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
        language_priors=torch.tensor([0.5, 0.5]),
    )
    post = gmm.posterior(torch.zeros(3, 2))
    assert post.shape == (3, 2)
    assert torch.allclose(post.sum(dim=1), torch.ones(3))


def test_future_model_can_be_added_without_changing_validation_code():
    from copy import deepcopy

    cfg = deepcopy(load_config(ROOT / "configs" / "config.yaml"))
    cfg["models"]["future_causal_lm"] = {
        "enabled": True,
        "model_name": "organization/future-causal-lm",
        "model_type": "causal_lm",
        "use_chat_template": False,
        "torch_dtype": "float16",
    }
    validate_config(cfg)
    assert [name for name, _ in enabled_models(cfg)][-1] == "future_causal_lm"


def test_spectral_embedding_writes_layer_coordinates():
    import tempfile
    import torch
    from src.gmm import GPULanguageGMM
    from src.manifold import save_language_manifold

    languages = ["en", "fa", "zh", "ja"]
    gmm = GPULanguageGMM(
        means=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        variances=torch.ones(4, 3),
        languages=languages,
        language_priors=torch.full((4,), 0.25),
    )
    with tempfile.TemporaryDirectory() as tmp:
        coordinates = save_language_manifold(
            gmms=[gmm],
            languages=languages,
            language_names={lang: lang for lang in languages},
            output_dir=tmp,
            cfg={"project": {"seed": 42}, "spectral_embedding": {"n_components": 2, "n_neighbors": 2}},
            model_name="test-model",
        )
        assert coordinates.shape[0] == len(languages)
        assert (Path(tmp) / "spectral_embedding_coordinates.csv").exists()
        assert (Path(tmp) / "layer_000.png").exists()
