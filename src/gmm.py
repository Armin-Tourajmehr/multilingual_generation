from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class GPULanguageGMM:
    """
    One-component diagonal Gaussian per language.

    Each language has:
        - mean vector
        - diagonal variance
        - empirical language prior

    The prior is based on the number of Aya samples per language.
    """

    means: torch.Tensor
    variances: torch.Tensor
    languages: list[str]
    language_priors: torch.Tensor

    @property
    def n_languages(self) -> int:
        return len(self.languages)

    @property
    def pca_dim(self) -> int:
        return int(self.means.shape[-1])

    @torch.inference_mode()
    def posterior(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute:

            P(language | representation)

        using:

            P(language | x)
            ∝
            P(x | language) P(language)
        """

        if X.ndim != 2:
            raise ValueError(
                f"Expected X with shape [N, K], got {tuple(X.shape)}"
            )

        if X.shape[1] != self.pca_dim:
            raise ValueError(
                f"Expected PCA dimension {self.pca_dim}, "
                f"got {X.shape[1]}"
            )

        Xf = X.float()

        means = self.means.float()
        variances = self.variances.float().clamp_min(1e-8)

        priors = self.language_priors.float().clamp_min(1e-12)

        if len(priors) != self.n_languages:
            raise ValueError(
                f"Expected {self.n_languages} priors, "
                f"got {len(priors)}"
            )

        # [N, L, K]
        diff = Xf[:, None, :] - means[None, :, :]

        # Log likelihood under diagonal Gaussian
        log_likelihood = -0.5 * (
            diff.square()
            .div(variances[None, :, :])
            .sum(dim=-1)
            + torch.log(variances).sum(dim=-1)[None, :]
            + self.pca_dim * 1.8378770664093453
        )

        # Add empirical language prior
        log_posterior = (
            log_likelihood
            + torch.log(priors)[None, :]
        )

        return torch.softmax(log_posterior, dim=1)


def init_gmm_stats(
    n_layers: int,
    n_languages: int,
    pca_dim: int,
    device: torch.device,
):
    """
    Initialize streaming statistics for GMM fitting.
    """

    if device.type != "cuda":
        raise RuntimeError(
            "GMM training requires a CUDA device."
        )

    sums = torch.zeros(
        (n_layers, n_languages, pca_dim),
        device=device,
        dtype=torch.float32,
    )

    sums_sq = torch.zeros_like(sums)

    counts = torch.zeros(
        (n_layers, n_languages),
        device=device,
        dtype=torch.int64,
    )

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
    """
    Update mean/variance statistics for one language and layer.
    """

    if Z.numel() == 0:
        return

    Zf = Z.float()

    sums[layer, language_index].add_(
        Zf.sum(dim=0)
    )

    sums_sq[layer, language_index].add_(
        Zf.square().sum(dim=0)
    )

    counts[layer, language_index] += int(
        Zf.shape[0]
    )


def build_language_priors(
    languages: list[str],
    sample_counts: dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    """
    Build empirical language priors from Aya SAMPLE counts.

    P(language=l) = N_l / sum(N)

    Example:

        English = 4000
        Persian = 1500

    then:

        P(English) = 4000 / total
        P(Persian) = 1500 / total

    IMPORTANT:
    These are sample counts, NOT token-vector counts.
    """

    counts = torch.tensor(
        [
            float(sample_counts[language])
            for language in languages
        ],
        device=device,
        dtype=torch.float32,
    )

    if (counts <= 0).any():
        missing = [
            language
            for language, count in zip(
                languages,
                counts.tolist(),
            )
            if count <= 0
        ]

        raise ValueError(
            f"All language sample counts must be positive. "
            f"Invalid languages: {missing}"
        )

    priors = counts / counts.sum()

    return priors


@torch.inference_mode()
def finalize_gmm(
    sums: torch.Tensor,
    sums_sq: torch.Tensor,
    counts: torch.Tensor,
    languages: list[str],
    reg_covar: float,
    language_priors: torch.Tensor,
) -> list[GPULanguageGMM]:
    """
    Finalize one diagonal Gaussian per language for each layer.

    language_priors:
        Empirical priors based on Aya sample counts.
    """

    if len(languages) != sums.shape[1]:
        raise ValueError(
            "Number of languages does not match GMM statistics."
        )

    language_priors = language_priors.to(
        device=sums.device,
        dtype=torch.float32,
    )

    if language_priors.ndim != 1:
        raise ValueError(
            "language_priors must have shape [n_languages]."
        )

    if len(language_priors) != len(languages):
        raise ValueError(
            f"Expected {len(languages)} priors, "
            f"got {len(language_priors)}."
        )

    if (language_priors <= 0).any():
        raise ValueError(
            "All language priors must be positive."
        )

    # Normalize just in case
    language_priors = (
        language_priors
        / language_priors.sum()
    )

    models: list[GPULanguageGMM] = []

    for layer in range(sums.shape[0]):

        layer_counts = counts[layer]

        if (layer_counts == 0).any():
            missing = [
                languages[i]
                for i, c in enumerate(
                    layer_counts.tolist()
                )
                if c == 0
            ]

            raise RuntimeError(
                f"No Aya vectors available for layer "
                f"{layer}: {missing}"
            )

        n = (
            layer_counts
            .float()
            .unsqueeze(1)
            .clamp_min(1.0)
        )

        means = sums[layer] / n

        variances = (
            sums_sq[layer] / n
            - means.square()
        )

        variances = (
            variances
            .clamp_min(0.0)
            + float(reg_covar)
        )

        models.append(
            GPULanguageGMM(
                means=means.contiguous(),
                variances=variances.contiguous(),
                languages=list(languages),
                language_priors=language_priors.clone(),
            )
        )

    return models


def save_gmm(
    path: Path,
    model: GPULanguageGMM,
    sample_counts: dict[str, int] | None = None,
) -> None:
    """
    Save GMM parameters and empirical language priors.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    state = {
        "means": model.means.detach().cpu(),
        "variances": model.variances.detach().cpu(),
        "languages": list(model.languages),
        "language_priors": model.language_priors.detach().cpu(),
    }

    if sample_counts is not None:
        state["sample_counts"] = {
            language: int(count)
            for language, count in sample_counts.items()
        }

    torch.save(state, path)


def load_gmm(
    path: Path,
    device: torch.device,
) -> GPULanguageGMM:
    """
    Load a GMM saved with the new format.

    Also supports old GMM files that used equal_priors.
    """

    state = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    means = state["means"].to(
        device=device,
        dtype=torch.float32,
    )

    variances = state["variances"].to(
        device=device,
        dtype=torch.float32,
    )

    languages = list(state["languages"])

    # New format
    if "language_priors" in state:

        language_priors = state[
            "language_priors"
        ].to(
            device=device,
            dtype=torch.float32,
        )

    # Backward compatibility with old files
    elif "equal_priors" in state:

        # Old implementation used uniform priors.
        language_priors = torch.full(
            (len(languages),),
            1.0 / len(languages),
            device=device,
            dtype=torch.float32,
        )

    else:
        raise KeyError(
            f"{path} does not contain language priors."
        )

    language_priors = (
        language_priors
        .clamp_min(1e-12)
    )

    language_priors = (
        language_priors
        / language_priors.sum()
    )

    return GPULanguageGMM(
        means=means,
        variances=variances,
        languages=languages,
        language_priors=language_priors,
    )