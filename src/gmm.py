
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from scipy.special import logsumexp
from sklearn.decomposition import IncrementalPCA


@dataclass
class DiagonalGaussian:
    mean: np.ndarray
    variance: np.ndarray

    def log_prob(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        var = np.maximum(self.variance, 1e-12)
        log_det = np.log(var).sum()
        quad = ((X - self.mean) ** 2 / var).sum(axis=1)
        d = X.shape[1]
        return -0.5 * (d * np.log(2.0 * np.pi) + log_det + quad)


@dataclass
class LayerModel:
    layer: int
    pca: IncrementalPCA
    gaussians: dict[str, DiagonalGaussian]
    languages: list[str]
    equal_priors: bool

    def posterior(self, X: np.ndarray) -> np.ndarray:
        logs = []
        for lang in self.languages:
            logs.append(self.gaussians[lang].log_prob(X))
        logs = np.vstack(logs).T
        if self.equal_priors:
            log_prior = -np.log(len(self.languages))
            logs = logs + log_prior
        return np.exp(logs - logsumexp(logs, axis=1, keepdims=True))


def make_incremental_pca(n_components: int, batch_size: int) -> IncrementalPCA:
    return IncrementalPCA(n_components=n_components, batch_size=batch_size)


def fit_pca(pca: IncrementalPCA, batches: Iterable[np.ndarray]) -> None:
    for X in batches:
        if len(X) >= pca.n_components:
            pca.partial_fit(X.astype(np.float32, copy=False))


def gaussian_from_pca_batches(pca: IncrementalPCA, batches: Iterable[np.ndarray], reg_covar: float) -> DiagonalGaussian:
    n = 0
    total = None
    total_sq = None
    for X in batches:
        if len(X) == 0:
            continue
        Z = pca.transform(X.astype(np.float32, copy=False)).astype(np.float64, copy=False)
        if total is None:
            total = np.zeros(Z.shape[1], dtype=np.float64)
            total_sq = np.zeros(Z.shape[1], dtype=np.float64)
        n += len(Z)
        total += Z.sum(axis=0)
        total_sq += np.square(Z).sum(axis=0)
    if n == 0:
        raise ValueError("No embeddings were available to fit a Gaussian.")
    mean = total / n
    var = np.maximum(total_sq / n - np.square(mean), 0.0) + float(reg_covar)
    return DiagonalGaussian(mean=mean, variance=var)


def save_layer_model(path: Path, model: LayerModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_layer_model(path: Path) -> LayerModel:
    return joblib.load(path)
