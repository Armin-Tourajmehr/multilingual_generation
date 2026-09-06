from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from src.config import load_config
from src.data import load_experiment_data, validate_train_analysis_split
from src.gmm import fit_layer_models, fit_layer_models_streaming, posterior


def make_cfg(tmpdir: str):
    cfg = load_config("configs/config.yaml")
    cfg["languages"] = ["en", "fa", "ar"]
    cfg["language_names"] = {"en": "English", "fa": "Persian", "ar": "Arabic"}
    cfg["data"]["root_dir"] = tmpdir
    cfg["data"]["train_fraction"] = 0.75
    cfg["data"]["analysis_fraction"] = 0.25
    cfg["data"]["sampling"] = "random"
    cfg["representation"]["pca_dim"] = 3
    return cfg


def test_full_data_split_is_disjoint_and_covers_every_sample():
    with tempfile.TemporaryDirectory() as td:
        cfg = make_cfg(td)
        original = {}
        for lang in cfg["languages"]:
            ids = list(range(1, 13))
            original[lang] = set(ids)
            pd.DataFrame({
                "sample_id": ids,
                "inputs": [f"{lang} sample {i}" for i in ids],
                "targets": [""] * len(ids),
            }).to_csv(Path(td) / f"{lang}.csv", index=False)

        train = load_experiment_data(cfg, "train", 42)
        analysis = load_experiment_data(cfg, "analysis", 42)
        validate_train_analysis_split(train, analysis)

        for lang in cfg["languages"]:
            train_ids = set(train.loc[train.source_language == lang, "sample_id"])
            analysis_ids = set(analysis.loc[analysis.source_language == lang, "sample_id"])
            assert train_ids | analysis_ids == original[lang]
            assert train_ids & analysis_ids == set()


def test_posterior_is_normalized():
    rng = np.random.default_rng(42)
    languages = ["en", "fa", "ar"]
    hidden = {
        lang: {0: [x.astype(np.float32) for x in rng.normal(i * 3, 1, (6, 8))]}
        for i, lang in enumerate(languages)
    }
    models = fit_layer_models(
        hidden,
        languages,
        {"pca_dim": 3, "pca_variance": None, "pca_solver": "full", "gmm_components": 1, "gmm_covariance_type": "diag"},
        42,
    )
    probs = posterior(hidden["en"][0][0], models[0])
    assert set(probs) == set(languages)
    assert np.isclose(sum(probs.values()), 1.0)


def test_streaming_fit_and_posterior():
    rng = np.random.default_rng(7)
    languages = ["en", "fa", "ar"]
    samples = [(lang, {0: rng.normal(i * 2, 1, (7, 8)).astype(np.float32)})
               for i, lang in enumerate(languages) for _ in range(3)]

    def factory():
        for lang, emb in samples:
            yield lang, emb

    models = fit_layer_models_streaming(
        factory,
        languages,
        {
            "pca_dim": 3,
            "pca_variance": None,
            "pca_batch_size": 6,
            "gmm_components": 1,
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1e-6,
        },
        42,
    )
    probs = posterior(samples[0][1][0][0], models[0])
    assert set(probs) == set(languages)
    assert np.isclose(sum(probs.values()), 1.0)
