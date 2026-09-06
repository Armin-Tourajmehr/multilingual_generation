from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.special import logsumexp
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.mixture import GaussianMixture


class StreamingDiagonalGaussian:
    """One-component diagonal Gaussian fitted from streaming sufficient statistics.

    This is mathematically equivalent to the MLE mean/diagonal variance of a
    one-component diagonal Gaussian mixture, with sklearn-style `reg_covar`
    added to the variance for numerical stability.
    """

    def __init__(self, mean: np.ndarray, variance: np.ndarray):
        self.mean_ = np.asarray(mean, dtype=np.float64)
        self.covariance_ = np.asarray(variance, dtype=np.float64)
        self.n_features_in_ = len(self.mean_)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        var = np.maximum(self.covariance_, 1e-12)
        log_det = np.sum(np.log(var))
        quadratic = np.sum((X - self.mean_) ** 2 / var, axis=1)
        return -0.5 * (self.n_features_in_ * np.log(2.0 * np.pi) + log_det + quadratic)


def _pca_kwargs(rep: dict[str, Any], seed: int) -> dict[str, Any]:
    kwargs = {"random_state": seed}
    if rep.get("pca_variance") is not None:
        kwargs["n_components"] = rep["pca_variance"]
        kwargs["svd_solver"] = rep.get("pca_solver", "auto")
    else:
        kwargs["n_components"] = rep.get("pca_dim", 50)
        kwargs["svd_solver"] = rep.get("pca_solver", "full")
    return kwargs


def fit_layer_models(
    all_hidden_states: dict[str, dict[int, list[np.ndarray]]],
    languages: list[str],
    representation_cfg: dict[str, Any],
    seed: int,
) -> dict[int, dict[str, Any]]:
    """Reference in-memory fitter retained for small tests/controlled studies."""
    num_layers = len(next(iter(all_hidden_states.values())))
    models = {}

    for layer_idx in range(num_layers):
        arrays = [
            np.asarray(vecs, dtype=np.float32)
            for lang in languages
            for vecs in [np.vstack(all_hidden_states[lang][layer_idx])]
        ]
        combined = np.vstack(arrays)

        pca = PCA(**_pca_kwargs(representation_cfg, seed))
        combined_pca = pca.fit_transform(combined)

        gaussians = {}
        offset = 0
        for lang in languages:
            n = len(all_hidden_states[lang][layer_idx])
            X_lang = combined_pca[offset : offset + n]
            offset += n
            gmm = GaussianMixture(
                n_components=representation_cfg.get("gmm_components", 1),
                covariance_type=representation_cfg.get("gmm_covariance_type", "diag"),
                random_state=seed,
            )
            gmm.fit(X_lang)
            gaussians[lang] = gmm

        prior_value = 1.0 / len(languages)
        priors = {lang: prior_value for lang in languages}
        models[layer_idx] = {
            "layer": layer_idx,
            "languages": languages,
            "pca": pca,
            "gaussians": gaussians,
            "priors": priors,
        }

    return models


def _make_pca(rep: dict[str, Any], seed: int) -> IncrementalPCA:
    if rep.get("pca_variance") is not None:
        raise ValueError(
            "The streaming training path requires representation.pca_variance=null "
            "and a fixed pca_dim."
        )
    pca_dim = int(rep.get("pca_dim", 50))
    batch_size = int(rep.get("pca_batch_size", max(4 * pca_dim, 256)))
    return IncrementalPCA(n_components=pca_dim, batch_size=batch_size)


class _OnlineDiagonalStats:
    def __init__(self, dimension: int):
        self.n = 0
        self.mean = np.zeros(dimension, dtype=np.float64)
        self.m2 = np.zeros(dimension, dtype=np.float64)

    def update(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        if X.size == 0:
            return
        batch_n = X.shape[0]
        batch_mean = X.mean(axis=0)
        batch_m2 = ((X - batch_mean) ** 2).sum(axis=0)

        if self.n == 0:
            self.n = batch_n
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        total = self.n + batch_n
        delta = batch_mean - self.mean
        self.m2 = self.m2 + batch_m2 + delta**2 * (self.n * batch_n / total)
        self.mean = self.mean + delta * (batch_n / total)
        self.n = total

    def finalize(self, reg_covar: float) -> StreamingDiagonalGaussian:
        if self.n == 0:
            raise ValueError("Cannot finalize Gaussian statistics with zero observations.")
        variance = self.m2 / self.n
        variance = np.maximum(variance + reg_covar, 1e-12)
        return StreamingDiagonalGaussian(self.mean, variance)


def fit_layer_models_streaming(
    sample_iterator_factory,
    languages: list[str],
    representation_cfg: dict[str, Any],
    seed: int,
) -> dict[int, dict[str, Any]]:
    """Fit layer-wise PCA + one diagonal Gaussian per language without storing all embeddings.

    `sample_iterator_factory` must return a fresh iterator over training samples each time it is called.
    Two passes are used: pass 1 fits IncrementalPCA; pass 2 fits one diagonal Gaussian per language
    from streaming sufficient statistics in PCA space.
    """
    if representation_cfg.get("gmm_components", 1) != 1:
        raise ValueError("The scalable full-data training path currently supports gmm_components=1 only.")
    if representation_cfg.get("gmm_covariance_type", "diag") != "diag":
        raise ValueError("The scalable full-data training path requires gmm_covariance_type='diag'.")

    first_iterator = sample_iterator_factory()
    try:
        first_lang, first_embeddings = next(first_iterator)
    except StopIteration as exc:
        raise ValueError("No training samples were provided.") from exc

    num_layers = len(first_embeddings)

    def first_pass_iterator():
        yield first_lang, first_embeddings
        yield from first_iterator

    # Keep only the first sample alive until it is consumed by the first pass.

    pcas = {layer: _make_pca(representation_cfg, seed) for layer in range(num_layers)}
    pca_dim = int(representation_cfg.get("pca_dim", 50))
    batch_size = int(representation_cfg.get("pca_batch_size", max(4 * pca_dim, 256)))
    buffers = {layer: [] for layer in range(num_layers)}
    buffer_sizes = {layer: 0 for layer in range(num_layers)}

    def flush(layer: int, force: bool = False):
        if not buffers[layer]:
            return
        # Leave at least pca_dim observations in the buffer for the final pass.
        if not force and buffer_sizes[layer] < (batch_size + pca_dim):
            return
        batch = np.vstack(buffers[layer]).astype(np.float32, copy=False)
        pcas[layer].partial_fit(batch)
        buffers[layer].clear()
        buffer_sizes[layer] = 0

    for _, embeddings_by_layer in first_pass_iterator():
        for layer, array in embeddings_by_layer.items():
            buffers[layer].append(np.asarray(array, dtype=np.float32))
            buffer_sizes[layer] += len(array)
            flush(layer)

    for layer in range(num_layers):
        # IncrementalPCA requires at least n_components observations in a batch.
        if buffer_sizes[layer] < pca_dim:
            if not hasattr(pcas[layer], "components_"):
                raise ValueError(
                    f"Layer {layer}: fewer than {pca_dim} training vectors are available for PCA."
                )
            # No vectors remain only when the last non-final buffer was already
            # fitted; otherwise force-fit the remaining valid batch.
        flush(layer, force=True)

    reg_covar = float(representation_cfg.get("gmm_reg_covar", 1e-6))
    stats = {
        layer: {lang: _OnlineDiagonalStats(pca_dim) for lang in languages}
        for layer in range(num_layers)
    }

    for lang, embeddings_by_layer in sample_iterator_factory():
        for layer, array in embeddings_by_layer.items():
            transformed = pcas[layer].transform(np.asarray(array, dtype=np.float32))
            stats[layer][lang].update(transformed)

    models = {}
    prior_value = 1.0 / len(languages)
    priors = {lang: prior_value for lang in languages}
    for layer in range(num_layers):
        gaussians = {
            lang: stats[layer][lang].finalize(reg_covar)
            for lang in languages
        }
        models[layer] = {
            "layer": layer,
            "languages": languages,
            "pca": pcas[layer],
            "gaussians": gaussians,
            "priors": priors,
            "fitting": {
                "method": "streaming_incremental_pca_and_diagonal_gaussian",
                "pca_dim": pca_dim,
                "pca_batch_size": batch_size,
                "gmm_components": 1,
                "gmm_covariance_type": "diag",
                "gmm_reg_covar": reg_covar,
            },
        }
    return models


def posterior(vector: np.ndarray, layer_model: dict[str, Any]) -> dict[str, float]:
    z = layer_model["pca"].transform(np.asarray(vector, dtype=np.float32).reshape(1, -1))
    log_probs = {}
    for lang in layer_model["languages"]:
        score = layer_model["gaussians"][lang].score_samples(z)[0]
        prior = np.log(layer_model["priors"][lang])
        log_probs[lang] = score + prior

    keys = list(log_probs)
    values = np.asarray([log_probs[k] for k in keys], dtype=np.float64)
    values = np.exp(values - logsumexp(values))
    return {k: float(v) for k, v in zip(keys, values)}


def save_layer_models(models: dict[int, dict[str, Any]], out_dir: Path, model_key: str, overwrite: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    for layer_idx, obj in models.items():
        path = out_dir / model_key / f"layer_{layer_idx}.pkl"
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing model file: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        import pickle
        with path.open("wb") as f:
            pickle.dump(obj, f)


def load_layer_models(model_dir: Path, model_key: str) -> dict[int, dict[str, Any]]:
    import pickle
    model_dir = model_dir / model_key
    paths = sorted(model_dir.glob("layer_*.pkl"), key=lambda p: int(p.stem.split("_")[-1]))
    if not paths:
        raise FileNotFoundError(f"No trained layer models found in {model_dir}")
    result = {}
    for path in paths:
        with path.open("rb") as f:
            obj = pickle.load(f)
        result[obj["layer"]] = obj
    return result
