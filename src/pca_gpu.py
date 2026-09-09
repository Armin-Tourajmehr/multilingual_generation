from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


@dataclass
class GPUPCA:
    """PCA representation kept on CUDA.

    Components are learned from the covariance of all reference token vectors:
        C = E[x x^T] - mu mu^T
    The top eigenvectors of C are the PCA directions.
    """

    mean: torch.Tensor          # [hidden_dim]
    components: torch.Tensor   # [n_components, hidden_dim]
    explained_variance: torch.Tensor  # [n_components]

    @property
    def n_components(self) -> int:
        return int(self.components.shape[0])

    @property
    def hidden_dim(self) -> int:
        return int(self.components.shape[1])

    def to(self, device: torch.device | str) -> "GPUPCA":
        device = torch.device(device)
        self.mean = self.mean.to(device=device, dtype=torch.float32)
        self.components = self.components.to(device=device, dtype=torch.float32)
        self.explained_variance = self.explained_variance.to(device=device, dtype=torch.float32)
        return self

    @torch.inference_mode()
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim != 2:
            raise ValueError(f"Expected [N, D] tensor, got shape {tuple(X.shape)}")
        Xf = X.float()
        return (Xf - self.mean) @ self.components.T

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "mean": self.mean,
            "components": self.components,
            "explained_variance": self.explained_variance,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor], device: torch.device) -> "GPUPCA":
        obj = cls(
            mean=state["mean"],
            components=state["components"],
            explained_variance=state["explained_variance"],
        )
        return obj.to(device)


class GPUPCACovarianceAccumulator:
    """Accumulate exact global PCA sufficient statistics on CUDA."""

    def __init__(self, hidden_dim: int, device: torch.device):
        if device.type != "cuda":
            raise RuntimeError("GPU PCA requires a CUDA device.")
        self.hidden_dim = int(hidden_dim)
        self.device = device
        self.count = 0
        self.sum_x = torch.zeros(self.hidden_dim, device=device, dtype=torch.float32)
        self.sum_xx = torch.zeros((self.hidden_dim, self.hidden_dim), device=device, dtype=torch.float32)

    @torch.inference_mode()
    def update(self, X: torch.Tensor) -> None:
        if X.numel() == 0:
            return
        if X.ndim != 2 or X.shape[1] != self.hidden_dim:
            raise ValueError(f"Expected [N, {self.hidden_dim}], got {tuple(X.shape)}")
        Xf = X.float()
        self.count += int(Xf.shape[0])
        self.sum_x.add_(Xf.sum(dim=0))
        self.sum_xx.add_(Xf.T @ Xf)

    @torch.inference_mode()
    def finalize(self, n_components: int) -> GPUPCA:
        if self.count == 0:
            raise ValueError("No vectors were accumulated for PCA.")
        if n_components <= 0 or n_components > self.hidden_dim:
            raise ValueError(
                f"n_components must be in [1, {self.hidden_dim}], got {n_components}."
            )
        mean = self.sum_x / float(self.count)
        covariance = self.sum_xx / float(self.count) - torch.outer(mean, mean)
        # Numerical round-off may introduce tiny asymmetry/negative eigenvalues.
        covariance = (covariance + covariance.T) * 0.5
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        top_values = eigenvalues[-n_components:].flip(0).clamp_min_(0.0)
        top_vectors = eigenvectors[:, -n_components:].flip(dims=[1]).T.contiguous()
        return GPUPCA(
            mean=mean.contiguous(),
            components=top_vectors,
            explained_variance=top_values.contiguous(),
        )


def save_pca(path: Path, pca: GPUPCA) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pca.state_dict(), path)


def load_pca(path: Path, device: torch.device) -> GPUPCA:
    if not path.exists():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location=device, weights_only=True)
    return GPUPCA.from_state_dict(state, device)


def pca_paths(root: Path, model_name: str) -> list[Path]:
    return sorted((root / model_name).glob("layer_*.pt"))
