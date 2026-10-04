from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class GPULanguageGMM:
    """One-component diagonal GMM per language, stored entirely on CUDA.

    For one component and diagonal covariance, this is exactly a diagonal
    Gaussian. Means/variances are fitted from streaming sufficient statistics.
    """

    means: torch.Tensor       # [L, K]
    variances: torch.Tensor   # [L, K]
    languages: list[str]
    equal_priors: bool = True

    @property
    def log_prior(self) -> float:
        return -torch.log(torch.tensor(float(self.n_languages), device=self.means.device)).item() if self.equal_priors else 0.0

    @property
    def n_languages(self) -> int:
        return len(self.languages)

    @torch.inference_mode()
    def posterior(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim != 2:
            raise ValueError(f"Expected [N, K], got {tuple(X.shape)}")
        Xf = X.float()
        means = self.means
        variances = self.variances.clamp_min(1e-8)
        diff = Xf[:, None, :] - means[None, :, :]
        log_prob = -0.5 * (
            diff.square().div(variances[None, :, :]).sum(dim=-1)
            + torch.log(variances).sum(dim=-1)[None, :]
            + Xf.shape[1] * 1.8378770664093453  # log(2*pi)
        )
        if self.equal_priors:
            log_prob = log_prob + float(self.log_prior)
        return torch.softmax(log_prob, dim=1)


def init_gmm_stats(n_layers: int, n_languages: int, pca_dim: int, device: torch.device):
    if device.type != "cuda":
        raise RuntimeError("GMM training requires a CUDA device.")
    sums = torch.zeros((n_layers, n_languages, pca_dim), device=device, dtype=torch.float32)
    sums_sq = torch.zeros_like(sums)
    counts = torch.zeros((n_layers, n_languages), device=device, dtype=torch.int64)
    return sums, sums_sq, counts


@torch.inference_mode()
def update_gmm_stats(
    sums: torch.Tensor,
    sums_sq: torch.Tensor,
    counts: torch.Tensor,
    layer: int,
    language_index: int,
    Z: torch.Tensor,
) -> None:
    if Z.numel() == 0:
        return
    Zf = Z.float()
    sums[layer, language_index].add_(Zf.sum(dim=0))
    sums_sq[layer, language_index].add_(Zf.square().sum(dim=0))
    counts[layer, language_index] += int(Zf.shape[0])


@torch.inference_mode()
def finalize_gmm(
    sums: torch.Tensor,
    sums_sq: torch.Tensor,
    counts: torch.Tensor,
    languages: list[str],
    reg_covar: float,
    equal_priors: bool,
) -> list[GPULanguageGMM]:
    models: list[GPULanguageGMM] = []
    for layer in range(sums.shape[0]):
        n = counts[layer].float().unsqueeze(1).clamp_min(1.0)
        means = sums[layer] / n
        variances = (sums_sq[layer] / n - means.square()).clamp_min(0.0) + float(reg_covar)
        if (counts[layer] == 0).any():
            missing = [languages[i] for i, c in enumerate(counts[layer].tolist()) if c == 0]
            raise RuntimeError(f"No Aya vectors available for layer {layer}: {missing}")
        models.append(
            GPULanguageGMM(
                means=means.contiguous(),
                variances=variances.contiguous(),
                languages=list(languages),
                equal_priors=equal_priors,
            )
        )
    return models


def save_gmm(path: Path, model: GPULanguageGMM) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "means": model.means,
            "variances": model.variances,
            "languages": model.languages,
            "equal_priors": model.equal_priors,
        },
        path,
    )


def load_gmm(path: Path, device: torch.device) -> GPULanguageGMM:
    state = torch.load(path, map_location=device, weights_only=True)
    return GPULanguageGMM(
        means=state["means"].to(device=device, dtype=torch.float32),
        variances=state["variances"].to(device=device, dtype=torch.float32),
        languages=list(state["languages"]),
        equal_priors=bool(state["equal_priors"]),
    )
