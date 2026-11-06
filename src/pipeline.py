from __future__ import annotations

import gc
import hashlib
import json
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from .data import (
    iter_aya_batches,
    iter_wikipedia_documents,
    validate_aya_files,
)

from .gmm import (
    GPULanguageGMM,
    build_language_priors,
    finalize_gmm,
    init_gmm_stats,
    save_gmm,
    load_gmm,
    update_gmm_stats,
)

from .modeling import (
    LoadedModel,
    forward_text_batch,
    generate_batch,
    load_model,
    replay_prompt_and_continuation,
    release_model,
)

from .lexical import StanzaTagger, align_pos_to_token
from .manifold import save_language_manifold
from .token_alignment import align_generated_token_offsets, tokenizer_special_ids

from .pca_gpu import (
    GPUPCA,
    GPUPCACovarianceAccumulator,
    load_pca,
    pca_paths,
    save_pca,
)


_PANDAS_STRING_TYPE = pa.array(pd.Series([""])).type


def _columnar_table(columns: dict[str, Any]) -> pa.Table:
    """Keep pandas' string schema convention without a high-volume DataFrame."""
    arrays = {}
    for name, values in columns.items():
        first = next((value for value in values if value is not None), None)
        arrays[name] = pa.array(values, type=_PANDAS_STRING_TYPE) if isinstance(first, str) else values
    return pa.table(arrays)


class PartitionWriter:
    """Append row batches to one compressed Parquet file."""

    def __init__(self, path: Path):
        self.path = path
        self.writer = None

    def write(self, rows: list[dict[str, Any]]):
        if not rows:
            return

        table = pa.Table.from_pandas(
            pd.DataFrame(rows),
            preserve_index=False,
        )
        self.write_table(table)

    def write_table(self, table: pa.Table, *, allow_empty: bool = False) -> None:
        """Append an Arrow batch without constructing per-row Python dictionaries."""
        if table.num_rows == 0 and not allow_empty:
            return

        if self.writer is None:
            self.path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            self.writer = pq.ParquetWriter(
                self.path,
                table.schema,
                compression="zstd",
            )

        self.writer.write_table(table)

    def close(self):
        if self.writer is not None:
            self.writer.close()


def _model_revision(loaded: LoadedModel) -> str | None:
    return getattr(getattr(loaded.model, "config", None), "_commit_hash", None)


def _pca_fingerprint(cfg: dict[str, Any], loaded: LoadedModel) -> str:
    payload = {
        "implementation_version": "b200-analysis-v1",
        "position_policy": "attention-mask-relative-v1",
        "model_key": loaded.name,
        "model_name": loaded.spec.get("model_name"),
        "model_revision": _model_revision(loaded),
        "torch_dtype": loaded.spec.get("torch_dtype", "auto"),
        "mixed_precision": cfg["runtime"].get("mixed_precision", "fp16"),
        "attn_implementation": loaded.spec.get("attn_implementation"),
        "trust_remote_code": bool(cfg["runtime"].get("trust_remote_code", False)),
        "languages": list(cfg["languages"]),
        "project_seed": int(cfg["project"].get("seed", 42)),
        "pca_dim": int(cfg["representation"]["pca_dim"]),
        "pca_solver": cfg["representation"].get("pca_solver"),
        "wiki_dataset": cfg["wikipedia"].get("dataset_name"),
        "wiki_split": cfg["wikipedia"].get("split", "train"),
        "wiki_revision": cfg["wikipedia"].get("revision"),
        "wiki_text_column": cfg["wikipedia"].get("text_column", "text"),
        "wiki_sample_cap": int(cfg["wikipedia"]["max_samples_per_language"]),
        "wiki_max_tokens": int(cfg["wikipedia"]["max_tokens_per_document"]),
        "wiki_shuffle_buffer_size": int(cfg["wikipedia"].get("shuffle_buffer_size", 10000)),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pca_basis_fingerprint(pcas: list[GPUPCA]) -> str:
    """Hash the fitted transforms in layer order, independent of device and storage layout."""
    digest = hashlib.sha256(b"pca-basis-v1\n")
    for pca in pcas:
        for tensor in (pca.mean, pca.components):
            values = tensor.detach().cpu().contiguous()
            digest.update(f"{values.dtype}:{tuple(values.shape)}\n".encode("ascii"))
            digest.update(memoryview(values.numpy()).cast("B"))
    return digest.hexdigest()


def _aya_content_fingerprints(cfg: dict[str, Any]) -> dict[str, str]:
    """Hash the selected, cleaned input rows rather than only their counts/paths."""
    hashes = {}
    columns = [cfg["data"]["sample_id_column"], cfg["data"]["input_column"]]
    for language in cfg["languages"]:
        digest = hashlib.sha256()
        for chunk in iter_aya_batches(cfg, language, 1024):
            for sample_id, text in chunk[columns].itertuples(index=False, name=None):
                encoded = json.dumps([str(sample_id), str(text)], ensure_ascii=False).encode("utf-8")
                digest.update(encoded)
                digest.update(b"\n")
        hashes[language] = digest.hexdigest()
    return hashes


def _prior_type(cfg: dict[str, Any]) -> str:
    return "uniform" if cfg["representation"].get("equal_language_priors", False) else "empirical_sample_count"


def _configured_language_priors(
    cfg: dict[str, Any], sample_counts: dict[str, int], device: torch.device
) -> torch.Tensor:
    languages = list(cfg["languages"])
    if cfg["representation"].get("equal_language_priors", False):
        return torch.full((len(languages),), 1.0 / len(languages), device=device, dtype=torch.float32)
    return build_language_priors(languages, sample_counts, device)


def _gmm_fingerprint(
    cfg: dict[str, Any],
    loaded: LoadedModel,
    expected_sample_counts: dict[str, int],
    pca_basis_fingerprint: str,
) -> str:
    payload = {
        "pca_fingerprint": _pca_fingerprint(cfg, loaded),
        "pca_basis_fingerprint": pca_basis_fingerprint,
        "languages": list(cfg["languages"]),
        "aya_root": str(cfg["data"].get("root_dir")),
        "input_column": cfg["data"].get("input_column"),
        "sample_id_column": cfg["data"].get("sample_id_column"),
        "aya_sample_cap": int(cfg["data"].get("max_samples_per_language", 2000)),
        "sampling_seed": int(cfg["data"].get("sampling_seed", cfg["project"].get("seed", 42))),
        "sample_counts": {k: int(v) for k, v in expected_sample_counts.items()},
        "aya_content": _aya_content_fingerprints(cfg),
        "max_input_tokens": cfg["runtime"].get("max_input_tokens"),
        "gmm_reg_covar": float(cfg["representation"].get("gmm_reg_covar", 1e-6)),
        "prior_type": _prior_type(cfg),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _language_batches(
    cfg: dict[str, Any],
    language: str,
    batch_size: int,
) -> Iterator[list[str]]:
    texts: list[str] = []

    for row in iter_wikipedia_documents(
        cfg,
        language,
        seed_offset=sum(
            (i + 1) * ord(c)
            for i, c in enumerate(language)
        ),
    ):
        texts.append(row["text"])

        if len(texts) == batch_size:
            yield texts
            texts = []

    if texts:
        yield texts


# ============================================================
# PCA
# ============================================================

def _fit_pca_on_wikipedia(
    cfg: dict[str, Any],
    loaded: LoadedModel,
    pca_root: Path,
) -> list[GPUPCA]:
    """
    Fit exact covariance PCA on the full configured
    Wikipedia reference stream.

    All vector math stays on CUDA.
    """

    if loaded.device.type != "cuda":
        raise RuntimeError(
            "PCA fitting requires CUDA."
        )

    rep = cfg["representation"]

    n_components = int(
        rep["pca_dim"]
    )

    batch_size = int(
        cfg["runtime"]["wikipedia_batch_size"]
    )

    max_tokens = int(
        cfg["wikipedia"][
            "max_tokens_per_document"
        ]
    )

    accumulators: list[
        GPUPCACovarianceAccumulator
    ] | None = None

    reference_document_counts = {
        lang: 0
        for lang in cfg["languages"]
    }

    print(
        f"[{loaded.name}] Wikipedia pass: "
        f"fitting GPU PCA "
        f"({n_components} components)"
    )

    for language in cfg["languages"]:

        for texts in tqdm(
            _language_batches(
                cfg,
                language,
                batch_size,
            ),
            desc=(
                f"{loaded.name} "
                f"Wikipedia/PCA/{language}"
            ),
        ):

            reference_document_counts[
                language
            ] += len(texts)

            hidden_states, mask = (
                forward_text_batch(
                    loaded,
                    texts,
                    max_tokens,
                    cfg["runtime"],
                )
            )

            if accumulators is None:

                hidden_dim = int(
                    hidden_states[0].shape[-1]
                )

                n_layers = len(
                    hidden_states
                )

                accumulators = [
                    GPUPCACovarianceAccumulator(
                        hidden_dim,
                        loaded.device,
                    )
                    for _ in range(n_layers)
                ]

                if n_components > hidden_dim:
                    raise ValueError(
                        f"PCA components "
                        f"({n_components}) exceed "
                        f"hidden dimension "
                        f"({hidden_dim}) for "
                        f"{loaded.name}."
                    )

                print(
                    f"[{loaded.name}] "
                    f"hidden_dim={hidden_dim}, "
                    f"hidden_states={n_layers}"
                )

            for layer, hidden in enumerate(
                hidden_states
            ):

                X = hidden[mask]

                accumulators[layer].update(X)
                del X

            del hidden_states, mask

    if accumulators is None:
        raise RuntimeError(
            "Wikipedia stream produced no batches."
        )

    pcas = [
        acc.finalize(n_components)
        for acc in accumulators
    ]

    model_pca_dir = (
        pca_root / loaded.name
    )

    model_pca_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    count_df = pd.DataFrame(
        [
            {
                "language": lang,
                "num_documents": (
                    reference_document_counts[
                        lang
                    ]
                ),
            }
            for lang in cfg["languages"]
        ]
    )

    count_df.to_csv(
        model_pca_dir
        / "wikipedia_document_counts.csv",
        index=False,
    )

    metadata = {
        "model": loaded.name,
        "pca_method": (
            "global covariance "
            "eigendecomposition on GPU"
        ),
        "pca_components": n_components,
        "fingerprint": _pca_fingerprint(cfg, loaded),
        "model_name": loaded.spec.get("model_name"),
        "model_revision": _model_revision(loaded),
        "languages": list(
            cfg["languages"]
        ),
        "wikipedia_max_samples_per_language": int(
            cfg["wikipedia"][
                "max_samples_per_language"
            ]
        ),
        "wikipedia_max_tokens_per_document": (
            max_tokens
        ),
    }

    with (
        model_pca_dir / "metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Remove stale layer files if a model/cache with a different layer count existed.
    for old_path in model_pca_dir.glob("layer_*.pt"):
        old_path.unlink()

    for layer, pca in enumerate(pcas):

        save_pca(
            model_pca_dir
            / f"layer_{layer:03d}.pt",
            pca,
        )

    del accumulators

    return pcas


def _load_pcas_or_fit(
    cfg: dict[str, Any],
    loaded: LoadedModel,
    pca_root: Path,
    force_refit: bool = False,
) -> list[GPUPCA]:
    model_pca_dir = pca_root / loaded.name
    paths = pca_paths(pca_root, loaded.name)
    expected_layers = int(getattr(loaded.model.config, "num_hidden_layers")) + 1
    metadata_path = model_pca_dir / "metadata.json"
    expected_fingerprint = _pca_fingerprint(cfg, loaded)

    if not force_refit and len(paths) == expected_layers and metadata_path.exists():
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            cache_is_current = (
                metadata.get("fingerprint") == expected_fingerprint
                and metadata.get("languages") == list(cfg["languages"])
                and int(metadata.get("pca_components", -1)) == int(cfg["representation"]["pca_dim"])
                and metadata.get("model_name") == loaded.spec.get("model_name")
            )
            if cache_is_current:
                pcas = [load_pca(path, loaded.device) for path in paths]
                pca_dim = int(cfg["representation"]["pca_dim"])
                if all(p.n_components == pca_dim for p in pcas):
                    print(f"[{loaded.name}] Reusing matching PCA data from: {model_pca_dir.resolve()}")
                    return pcas
                print(f"[{loaded.name}] Saved PCA dimensions do not match; rebuilding PCA.")
            else:
                print(f"[{loaded.name}] PCA cache is stale for the current model/languages/settings; rebuilding it.")
        except Exception as exc:
            print(f"[{loaded.name}] Could not validate PCA cache ({exc}); rebuilding it.")
    elif paths and not force_refit:
        print(
            f"[{loaded.name}] PCA cache is incomplete or has no compatible metadata "
            f"(found {len(paths)} layer files, expected {expected_layers}); rebuilding it."
        )

    return _fit_pca_on_wikipedia(cfg, loaded, pca_root)


# ============================================================
# GMM FITTING
# ============================================================

def _fit_aya_gmms(
    cfg: dict[str, Any],
    loaded: LoadedModel,
    pcas: list[GPUPCA],
    gmm_root: Path,
    pca_root: Path,
    expected_sample_counts: dict[str, int],
    *,
    fingerprint: str,
    pca_basis_fingerprint: str,
) -> list[GPULanguageGMM]:
    """
    Fit one diagonal, one-component GMM per language
    on 100% of Aya.

    Language priors are empirical:

        P(language) = N_language / total_N

    where N_language is the number of Aya samples
    for that language.
    """

    languages = list(
        cfg["languages"]
    )

    lang_to_idx = {
        lang: i
        for i, lang in enumerate(
            languages
        )
    }

    pca_dim = int(
        cfg["representation"][
            "pca_dim"
        ]
    )

    batch_size = int(
        cfg["runtime"][
            "evaluation_batch_size"
        ]
    )

    # Number of Aya samples actually processed.
    sample_counts = {
        lang: 0
        for lang in languages
    }

    # Streaming GMM statistics.
    sums, sums_sq, counts = (
        init_gmm_stats(
            n_layers=len(pcas),
            n_languages=len(languages),
            pca_dim=pca_dim,
            device=loaded.device,
        )
    )

    print(
        f"[{loaded.name}] "
        f"Aya pass 1/2: "
        f"fitting GMMs on the configured Aya subset"
    )

    # --------------------------------------------------------
    # Stream through all Aya samples
    # --------------------------------------------------------

    for language in languages:

        language_idx = (
            lang_to_idx[language]
        )

        for chunk in tqdm(
            iter_aya_batches(
                cfg,
                language,
                batch_size,
            ),
            desc=(
                f"{loaded.name} "
                f"Aya/GMM/{language}"
            ),
        ):

            sample_counts[
                language
            ] += len(chunk)

            prompts = chunk[
                cfg["data"][
                    "input_column"
                ]
            ].tolist()

            hidden_states, mask = (
                forward_text_batch(
                    loaded,
                    prompts,
                    cfg["runtime"].get(
                        "max_input_tokens"
                    )
                    or None,
                    cfg["runtime"],
                )
            )

            for layer, pca in enumerate(
                pcas
            ):

                X = hidden_states[
                    layer
                ][mask]

                if X.numel() == 0:
                    continue

                # PCA projection
                Z = pca.transform(X)

                # Update Gaussian statistics
                update_gmm_stats(
                    sums,
                    sums_sq,
                    counts,
                    layer,
                    language_idx,
                    Z,
                )
                del X, Z

            del hidden_states, mask

    # --------------------------------------------------------
    # Full-data audit
    # --------------------------------------------------------

    if (
        sample_counts
        != expected_sample_counts
    ):

        raise RuntimeError(
            f"Aya full-data audit failed. "
            f"Expected samples="
            f"{expected_sample_counts}, "
            f"processed="
            f"{sample_counts}"
        )

    # --------------------------------------------------------
    # Empirical language priors
    # --------------------------------------------------------

    language_priors = _configured_language_priors(cfg, expected_sample_counts, loaded.device)

    print(
        f"[{loaded.name}] "
        f"Language priors ({_prior_type(cfg)}):"
    )

    for language, prior in zip(
        languages,
        language_priors.tolist(),
    ):

        print(
            f"    {language}: "
            f"{prior:.6f}"
        )

    # --------------------------------------------------------
    # Finalize GMM
    # --------------------------------------------------------

    gmms = finalize_gmm(
        sums,
        sums_sq,
        counts,
        languages=languages,
        reg_covar=float(
            cfg["representation"][
                "gmm_reg_covar"
            ]
        ),
        language_priors=language_priors,
    )

    # --------------------------------------------------------
    # Save GMMs
    # --------------------------------------------------------

    model_dir = (
        gmm_root / loaded.name
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Mark the old cache incomplete before replacing any layer files.
    metadata_path = model_dir / "metadata.json"
    metadata_path.unlink(missing_ok=True)

    for layer, gmm in enumerate(
        gmms
    ):

        save_gmm(
            model_dir
            / f"layer_{layer:03d}.pt",
            gmm,
            sample_counts=(
                expected_sample_counts
            ),
            cache_fingerprint=f"{fingerprint}:{layer}",
        )

    # --------------------------------------------------------
    # Save token-vector counts
    # --------------------------------------------------------

    counts_df = pd.DataFrame(
        [
            {
                "layer": layer,
                "language": language,
                "num_token_vectors": int(
                    counts[
                        layer,
                        i,
                    ].item()
                ),
            }
            for layer in range(
                counts.shape[0]
            )
            for i, language in enumerate(
                languages
            )
        ]
    )

    counts_df.to_csv(
        model_dir
        / "aya_gmm_fit_counts.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save sample-count audit
    # --------------------------------------------------------

    pd.DataFrame(
        [
            {
                "language": language,
                "num_samples_processed": (
                    sample_counts[
                        language
                    ]
                ),
                "num_samples_expected": (
                    expected_sample_counts[
                        language
                    ]
                ),
            }
            for language in languages
        ]
    ).to_csv(
        model_dir
        / "aya_gmm_sample_counts.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {
        "model": loaded.name,
        "training_corpus": "Aya",
        "coverage": (
            "100% of available configured "
            "Aya samples for each language"
        ),
        "components": int(
            cfg["representation"][
                "gmm_components"
            ]
        ),
        "covariance_type": (
            cfg["representation"][
                "gmm_covariance_type"
            ]
        ),
        "languages": languages,

        "prior_type": _prior_type(cfg),

        "sample_counts": {
            language: int(
                expected_sample_counts[
                    language
                ]
            )
            for language in languages
        },

        "pca_path": str(
            (
                pca_root
                / loaded.name
            ).resolve()
        ),
        "fingerprint": fingerprint,
        "pca_fingerprint": _pca_fingerprint(cfg, loaded),
        "pca_basis_fingerprint": pca_basis_fingerprint,
    }

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    del sums, sums_sq, counts

    return gmms


def _load_gmms_or_fit(
    cfg: dict[str, Any],
    loaded: LoadedModel,
    pcas: list[GPUPCA],
    gmm_root: Path,
    pca_root: Path,
    expected_sample_counts: dict[str, int],
    force_refit: bool = False,
) -> list[GPULanguageGMM]:
    """Reuse saved Aya GMMs only for the same data, settings, and fitted PCA basis."""

    model_dir = (
        gmm_root / loaded.name
    )

    paths = [
        model_dir
        / f"layer_{layer:03d}.pt"
        for layer in range(
            len(pcas)
        )
    ]

    metadata_path = (
        model_dir / "metadata.json"
    )

    expected_counts = {
        language: int(
            expected_sample_counts[
                language
            ]
        )
        for language in cfg["languages"]
    }
    pca_basis_fingerprint = _pca_basis_fingerprint(pcas)
    fingerprint = _gmm_fingerprint(cfg, loaded, expected_counts, pca_basis_fingerprint)

    # --------------------------------------------------------
    # Try to reuse existing GMM
    # --------------------------------------------------------

    if (
        not force_refit
        and all(
            path.exists()
            for path in paths
        )
        and metadata_path.exists()
    ):

        try:

            with metadata_path.open(
                "r",
                encoding="utf-8",
            ) as f:

                meta = json.load(f)

            same_pca = (
                Path(
                    meta.get(
                        "pca_path",
                        "",
                    )
                ).resolve()
                == (
                    pca_root
                    / loaded.name
                ).resolve()
            )

            same_langs = (
                meta.get("languages")
                == list(
                    cfg["languages"]
                )
            )

            same_coverage = (
                meta.get("coverage")
                == (
                    "100% of available "
                    "configured Aya samples "
                    "for each language"
                )
            )

            same_prior_type = (
                meta.get("prior_type")
                == _prior_type(cfg)
            )

            same_sample_counts = (
                meta.get(
                    "sample_counts"
                )
                == expected_counts
            )
            same_fingerprint = meta.get("fingerprint") == fingerprint
            same_pca_basis = meta.get("pca_basis_fingerprint") == pca_basis_fingerprint

            if (
                same_pca
                and same_langs
                and same_coverage
                and same_prior_type
                and same_sample_counts
                and same_fingerprint
                and same_pca_basis
            ):
                gmms = [
                    load_gmm(path, loaded.device, expected_cache_fingerprint=f"{fingerprint}:{layer}")
                    for layer, path in enumerate(paths)
                ]
                print(
                    f"[{loaded.name}] "
                    f"Reusing saved Aya GMMs "
                    f"from: "
                    f"{model_dir.resolve()}"
                )
                return gmms

            print(
                f"[{loaded.name}] "
                f"Existing Aya GMMs do not "
                f"match the current settings or fitted PCA basis. "
                f"Refitting."
            )

        except Exception as exc:

            print(
                f"[{loaded.name}] "
                f"Saved GMM cache could "
                f"not be validated "
                f"({exc}); refitting."
            )

    # --------------------------------------------------------
    # Fit from Aya
    # --------------------------------------------------------

    return _fit_aya_gmms(
        cfg,
        loaded,
        pcas,
        gmm_root,
        pca_root,
        expected_sample_counts,
        fingerprint=fingerprint,
        pca_basis_fingerprint=pca_basis_fingerprint,
    )


# ============================================================
# ANALYSIS
# ============================================================

def _batch_posteriors(
    hidden_states: list[torch.Tensor],
    pcas: list[GPUPCA],
    gmms: list[GPULanguageGMM],
    sample_indices: list[int],
    sequence_positions: list[int],
    posterior_batch_size: int,
    n_languages: int,
) -> np.ndarray:
    """Gather token vectors in batches and copy all layer posteriors to CPU once."""
    n_layers = min(len(hidden_states), len(pcas), len(gmms))
    if not sample_indices or n_layers == 0:
        return np.empty((n_layers, len(sample_indices), n_languages), dtype=np.float32)
    device = hidden_states[0].device
    sample_index = torch.tensor(sample_indices, dtype=torch.long, device=device)
    sequence_index = torch.tensor(sequence_positions, dtype=torch.long, device=device)
    layer_posteriors = []
    for layer in range(n_layers):
        vectors = hidden_states[layer][sample_index, sequence_index]
        projected = pcas[layer].transform(vectors)
        parts = [
            gmms[layer].posterior(projected[start:start + posterior_batch_size])
            for start in range(0, len(sample_indices), posterior_batch_size)
        ]
        layer_posteriors.append(torch.cat(parts, dim=0))
    return torch.stack(layer_posteriors).detach().cpu().numpy()


def _state_posterior_table(
    model: str,
    language: str,
    sample_ids: pa.Array | pa.ChunkedArray,
    rows: list[tuple[int, int, int, int]],
    posteriors: np.ndarray,
    posterior_columns: list[str],
    contexts: list[tuple[str, int, int]] | None = None,
) -> pa.Table:
    """Build one batch of a prompt or predecessor view from shared scores."""
    sample_indices, positions, token_ids, score_rows = zip(*rows) if rows else ((), (), (), ())
    columns = {
        "model": pa.array([model] * len(rows), type=_PANDAS_STRING_TYPE),
        "sample_id": sample_ids.take(pa.array(sample_indices, type=pa.int64())),
        "input_language": pa.array([language] * len(rows), type=_PANDAS_STRING_TYPE),
        "sequence_position": pa.array(positions, type=pa.int64()),
        "token_id": pa.array(token_ids, type=pa.int64()),
    }
    if contexts is not None:
        segments, context_positions, context_ids = zip(*contexts) if contexts else ((), (), ())
        columns.update(
            context_segment=pa.array(segments, type=_PANDAS_STRING_TYPE),
            context_position=pa.array(context_positions, type=pa.int64()),
            context_token_id=pa.array(context_ids, type=pa.int64()),
        )
    metadata = pa.table(columns)

    def layer_table(layer: int, values: np.ndarray) -> pa.Table:
        table = metadata.append_column("layer", pa.array([layer] * len(rows), type=pa.int64()))
        for index, column in enumerate(posterior_columns):
            table = table.append_column(column, pa.array(values[:, index], type=pa.float32()))
        return table

    if not rows:
        return layer_table(0, np.empty((0, len(posterior_columns)), dtype=np.float32))
    indices = np.asarray(score_rows, dtype=np.intp)
    return pa.concat_tables([layer_table(layer, values[indices]) for layer, values in enumerate(posteriors)])


@contextmanager
def _measure_phase(performance: dict[str, Any] | None, phase: str, device: torch.device):
    """Synchronize only at optional measurement boundaries, never per batch."""
    if performance is None:
        yield
        return
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = perf_counter()
    try:
        yield
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        performance["phase_seconds"][phase] = perf_counter() - started


def _analyze_model(
    cfg: dict[str, Any],
    loaded: LoadedModel,
    pcas: list[GPUPCA],
    gmms: list[GPULanguageGMM],
    result_dir: Path,
):
    """Score prompt, predecessor, and post-emission states from one ID replay."""
    result_dir.mkdir(parents=True, exist_ok=True)
    gen_cfg = cfg["generation"]
    batch_size = int(cfg["runtime"].get("generation_batch_size", 1))
    posterior_batch_size = int(cfg["runtime"].get("posterior_batch_size", 512))
    gc_interval = int(cfg["runtime"].get("gc_collect_every_batches", 0))
    batches_processed = 0
    languages = list(cfg["languages"])
    posterior_columns = [f"P_{language}" for language in languages]
    special_ids = tokenizer_special_ids(loaded.tokenizer)

    token_writers: dict[str, PartitionWriter] = {}
    sentence_writers: dict[str, PartitionWriter] = {}
    output_writers: dict[str, PartitionWriter] = {}
    prompt_writers: dict[str, PartitionWriter] = {}
    preceding_writers: dict[str, PartitionWriter] = {}
    counts = {lang: 0 for lang in languages}
    generated_token_count = 0
    analyzed_token_count = 0

    lexical_cfg = dict(cfg.get("lexical_analysis", {}))
    lexical_enabled = bool(lexical_cfg.get("enabled", True))
    lexical_deferred = lexical_enabled and bool(lexical_cfg.get("deferred", False))
    save_pos_summary = bool(lexical_cfg.get("save_pos_posterior_summary", True)) and not lexical_deferred
    stanza_dir = Path(lexical_cfg.get("stanza_model_dir", "outputs/stanza_resources"))
    if not stanza_dir.is_absolute():
        stanza_dir = Path(cfg.get("_repo_root", Path.cwd())) / stanza_dir
    lexical_cfg["stanza_model_dir"] = str(stanza_dir)
    tagger = StanzaTagger(lexical_cfg) if lexical_enabled and not lexical_deferred else None

    try:
        print(f"[{loaded.name}] Aya pass 2/2: generation + token posterior/POS analysis")

        for language in languages:
            # Keep only small aggregates in memory; all token rows are streamed to Parquet.
            pos_buckets: dict[tuple[int, str, str], dict[str, Any]] = {}
            prompt_writer = prompt_writers[language] = PartitionWriter(
                result_dir / "prompt_token_posteriors" / f"{language}.parquet"
            )
            preceding_writer = preceding_writers[language] = PartitionWriter(
                result_dir / "preceding_token_posteriors" / f"{language}.parquet"
            )
            token_path = result_dir / "token_posteriors" / f"{language}.parquet"
            sample_id_type = pa.int64()
            summary_path = result_dir / "lexical_analysis" / "pos_posterior_summary" / f"{language}.csv"
            if not save_pos_summary and summary_path.exists():
                summary_path.unlink()  # A previous run's summary must not look current.

            for chunk in tqdm(
                iter_aya_batches(cfg, language, batch_size),
                desc=f"{loaded.name} Aya/analysis/{language}",
            ):
                prompts = chunk[cfg["data"]["input_column"]].tolist()
                sample_ids = chunk[cfg["data"]["sample_id_column"]].tolist()

                generated_ids, generated_texts, _, prompt_token_ids = generate_batch(
                    loaded, prompts, gen_cfg, cfg["runtime"]
                )
                counts[language] += len(sample_ids)
                generated_token_count += sum(len(ids) for ids in generated_ids)
                output_table = _columnar_table({
                    "model": [loaded.name] * len(sample_ids),
                    "sample_id": sample_ids,
                    "input_language": [language] * len(sample_ids),
                    "prompt": prompts,
                    "generated_text": generated_texts,
                    "generated_token_count": [len(ids) for ids in generated_ids],
                })
                output_table = output_table.append_column(
                    "generated_token_ids", pa.array(generated_ids, type=pa.list_(pa.int64()))
                ).append_column(
                    "prompt_token_ids", pa.array(prompt_token_ids, type=pa.list_(pa.int64()))
                )
                sample_id_array = output_table["sample_id"]
                sample_id_type = sample_id_array.type
                outputs_path = result_dir / "generated_outputs" / f"{language}.parquet"
                output_writers.setdefault(language, PartitionWriter(outputs_path)).write_table(output_table)

                hidden_states, left_padding = replay_prompt_and_continuation(
                    loaded, prompt_token_ids, generated_ids, cfg["runtime"]
                )
                offsets = align_generated_token_offsets(loaded.tokenizer, generated_ids, generated_texts)

                per_sample_refs: list[list[dict[str, Any]]] = []
                for sample_idx, text in enumerate(generated_texts):
                    if tagger is not None and text.strip():
                        words, tag_status = tagger.tag_text(text, language)
                    elif lexical_deferred:
                        words, tag_status = [], "deferred"
                    elif not lexical_enabled:
                        words, tag_status = [], "disabled"
                    else:
                        words, tag_status = [], "empty"

                    refs: list[dict[str, Any]] = []
                    for seq_pos, (token_id, span) in enumerate(
                        zip(generated_ids[sample_idx], offsets[sample_idx])
                    ):
                        token_id = int(token_id)
                        if token_id in special_ids:
                            continue
                        start_char, end_char = int(span[0]), int(span[1])
                        if not 0 <= start_char < end_char <= len(text):
                            start_char = end_char = 0
                        token_text = text[start_char:end_char]
                        pos_info = align_pos_to_token(text, start_char, end_char, words, tag_status)
                        try:
                            model_token = loaded.tokenizer.convert_ids_to_tokens(token_id)
                            if isinstance(model_token, list):
                                model_token = "".join(str(x) for x in model_token)
                        except Exception:
                            model_token = loaded.tokenizer.decode([token_id], skip_special_tokens=False)
                        refs.append({
                            "sample_index": sample_idx,
                            "sequence_position": seq_pos,
                            "token_id": token_id,
                            "token": token_text,
                            "model_token": str(model_token),
                            "char_start": start_char,
                            "char_end": end_char,
                            **pos_info,
                        })
                    per_sample_refs.append(refs)

                flat_refs = [ref for refs in per_sample_refs for ref in refs]
                n_tokens = len(flat_refs)
                analyzed_token_count += n_tokens
                token_counts = [len(refs) for refs in per_sample_refs]
                token_starts = np.cumsum([0, *token_counts])
                score_positions: dict[tuple[int, int], int] = {}
                # Consumed references are already unique. Insert them first so
                # their scores retain the existing order in one contiguous slice.
                for i, refs in enumerate(per_sample_refs):
                    continuation_start = left_padding[i] + len(prompt_token_ids[i])
                    for ref in refs:
                        score_positions[i, continuation_start + ref["sequence_position"]] = len(score_positions)

                def score_row(sample: int, physical_position: int) -> int:
                    return score_positions.setdefault((sample, physical_position), len(score_positions))

                prompt_rows = []
                preceding_rows = []
                preceding_contexts = []
                for i, (prompt_ids, continuation_ids, pad, refs) in enumerate(
                    zip(prompt_token_ids, generated_ids, left_padding, per_sample_refs)
                ):
                    for pos, token_id in enumerate(prompt_ids):
                        if token_id not in special_ids:
                            prompt_rows.append((i, pos, token_id, score_row(i, pad + pos)))
                    for ref in refs:
                        target = ref["sequence_position"]
                        if target > 0:
                            context = ("continuation", target - 1, continuation_ids[target - 1])
                        elif prompt_ids:
                            context = ("prompt", len(prompt_ids) - 1, prompt_ids[-1])
                        else:
                            continue
                        preceding_rows.append((
                            i, target, ref["token_id"], score_row(i, pad + len(prompt_ids) + target - 1)
                        ))
                        preceding_contexts.append(context)
                all_posteriors = _batch_posteriors(
                    hidden_states, pcas, gmms,
                    [sample for sample, _ in score_positions],
                    [pos for _, pos in score_positions],
                    posterior_batch_size, len(languages),
                )
                del hidden_states
                n_layers = len(all_posteriors)
                if prompt_rows and n_layers:
                    prompt_writer.write_table(_state_posterior_table(
                        loaded.name, language, sample_id_array, prompt_rows, all_posteriors, posterior_columns
                    ))
                if preceding_rows and n_layers:
                    preceding_writer.write_table(_state_posterior_table(
                        loaded.name, language, sample_id_array, preceding_rows, all_posteriors,
                        posterior_columns, preceding_contexts,
                    ))
                # These Arrow arrays are built once and reused by every layer table.
                metadata_table = None
                pos_groups: dict[tuple[str, str], list[int]] = {}
                if n_tokens:
                    metadata_columns = {
                        "model": [loaded.name] * n_tokens,
                        "sample_id": [sample_ids[ref["sample_index"]] for ref in flat_refs],
                        "input_language": [language] * n_tokens,
                        "token_position": [position for refs in per_sample_refs for position in range(len(refs))],
                    }
                    for key in (
                        "sequence_position", "token", "model_token", "token_id", "char_start", "char_end",
                        "word_text", "sentence_id", "word_id", "upos", "xpos", "feats", "pos_status",
                    ):
                        metadata_columns[key] = [ref[key] for ref in flat_refs]
                    metadata_table = _columnar_table(metadata_columns)
                    if save_pos_summary:
                        for index, ref in enumerate(flat_refs):
                            key = (str(ref["upos"]), str(ref["pos_status"]))
                            pos_groups.setdefault(key, []).append(index)

                token_tables = []
                sentence_tables = []
                for layer in range(n_layers):
                    values = all_posteriors[layer, :n_tokens]
                    if metadata_table is not None:
                        table = metadata_table.append_column("layer", pa.array([layer] * n_tokens))
                        for index, column in enumerate(posterior_columns):
                            table = table.append_column(column, pa.array(values[:, index], type=pa.float64()))
                        token_tables.append(table)

                    # Accumulate in float64, matching the previous Python float sums.
                    sentence_values = np.zeros((len(sample_ids), len(languages)), dtype=np.float64)
                    for sample_idx, count in enumerate(token_counts):
                        if count:
                            start, end = token_starts[sample_idx:sample_idx + 2]
                            sentence_values[sample_idx] = values[start:end].sum(axis=0, dtype=np.float64) / count
                    sentence_columns = {
                        "model": [loaded.name] * len(sample_ids),
                        "sample_id": sample_ids,
                        "input_language": [language] * len(sample_ids),
                        "layer": [layer] * len(sample_ids),
                        "num_output_tokens": token_counts,
                    }
                    for index, column in enumerate(posterior_columns):
                        sentence_columns[column] = sentence_values[:, index]
                    sentence_tables.append(_columnar_table(sentence_columns))

                    for (upos, status), indices in pos_groups.items():
                        bucket = pos_buckets.setdefault(
                            (layer, upos, status),
                            {"n": 0, "sums": {column: 0.0 for column in posterior_columns}},
                        )
                        bucket["n"] += len(indices)
                        sums = values[indices].sum(axis=0, dtype=np.float64)
                        for column, value in zip(posterior_columns, sums):
                            bucket["sums"][column] += float(value)

                if token_tables:
                    token_writers.setdefault(language, PartitionWriter(token_path)).write_table(pa.concat_tables(token_tables))
                if sentence_tables:
                    sentence_path = result_dir / "sentence_level_posteriors" / f"{language}.parquet"
                    sentence_writers.setdefault(language, PartitionWriter(sentence_path)).write_table(pa.concat_tables(sentence_tables))
                del per_sample_refs, flat_refs, all_posteriors
                batches_processed += 1
                if gc_interval and batches_processed % gc_interval == 0:
                    gc.collect()

            for writer, contexts in ((prompt_writer, None), (preceding_writer, [])):
                if writer.writer is None:
                    writer.write_table(_state_posterior_table(
                        loaded.name, language, pa.array([], type=sample_id_type), [],
                        np.empty((0, 0, len(posterior_columns)), dtype=np.float32), posterior_columns, contexts,
                    ), allow_empty=True)
            if language not in token_writers:
                # Keep an older continuation file from being paired with the new prompt scores.
                token_path.unlink(missing_ok=True)

            if save_pos_summary:
                # Small per-POS aggregation: average posterior probability for each candidate
                # language by layer and Stanza UPOS tag. The detailed token rows remain in Parquet.
                summary_rows: list[dict[str, Any]] = []
                for (layer, upos, status), bucket in sorted(pos_buckets.items()):
                    n_tokens = int(bucket["n"])
                    for candidate_language in languages:
                        col = f"P_{candidate_language}"
                        summary_rows.append({
                            "model": loaded.name,
                            "input_language": language,
                            "layer": layer,
                            "upos": upos,
                            "pos_status": status,
                            "posterior_language": candidate_language,
                            "num_tokens": n_tokens,
                            "mean_posterior": bucket["sums"][col] / max(n_tokens, 1),
                        })
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
            if tagger is not None:
                # Only one source-language Stanza pipeline is retained at a time.
                tagger.close()

    finally:
        for writers in (token_writers, sentence_writers, output_writers, prompt_writers, preceding_writers):
            for writer in writers.values():
                writer.close()
        if tagger is not None:
            tagger.close()

    pd.DataFrame([
        {"language": lang, "num_samples_analyzed": count}
        for lang, count in counts.items()
    ]).to_csv(result_dir / "analysis_counts.csv", index=False)
    return {
        "samples_analyzed": sum(counts.values()),
        "samples_by_language": counts,
        "generated_tokens": generated_token_count,
        "analyzed_output_tokens": analyzed_token_count,
        "generation_batches": batches_processed,
    }


# ============================================================
# MAIN RUN
# ============================================================

def run(
    cfg: dict[str, Any],
    pca_path: str | Path | None = None,
    force_refit_pca: bool = False,
    force_refit_gmm: bool = False,
) -> None:
    repo_root = Path(cfg.get("_repo_root", Path.cwd()))
    output_root = Path(cfg["outputs"]["root_dir"])
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifests").mkdir(parents=True, exist_ok=True)

    validation = validate_aya_files(cfg)
    validation.to_csv(output_root / "manifests" / "aya_input_summary.csv", index=False)
    print("Aya input summary:")
    print(validation.to_string(index=False))

    default_pca_root = Path(
        cfg["runtime"].get("pca_data_path") or (output_root / "pca_data")
    )
    pca_root = Path(pca_path) if pca_path is not None else default_pca_root
    if not pca_root.is_absolute():
        pca_root = repo_root / pca_root
    pca_root.mkdir(parents=True, exist_ok=True)
    print(f"PCA data root: {pca_root.resolve()}")

    gmm_root = output_root / cfg["outputs"]["gmm_dir"]
    expected_sample_counts = {
        str(row.language): int(row.num_samples)
        for row in validation.itertuples(index=False)
    }

    enabled = [(key, spec) for key, spec in cfg["models"].items() if spec.get("enabled", False)]
    if not enabled:
        raise RuntimeError("No enabled models remain after configuration validation.")

    # Sequential execution is intentional: one model is loaded, processed and released
    # before the next model enters GPU memory. Smoke mode can continue after an individual
    # model fails (e.g. a model does not fit in available VRAM); the full run remains strict.
    smoke_continue = bool(cfg.get("smoke_test", {}).get("continue_on_model_error", False))
    model_errors: list[dict[str, str]] = []

    for model_key, model_spec in enabled:
        print("\n" + "=" * 90)
        print(f"MODEL: {model_key} | {model_spec['model_name']}")
        print("=" * 90)

        model_cfg = deepcopy(cfg)
        model_runtime = dict(cfg["runtime"])
        model_runtime.update(model_spec.get("runtime_overrides", {}))
        model_cfg["runtime"] = model_runtime
        loaded = None
        performance = {"model": model_key, "phase_seconds": {}} if model_runtime.get("log_performance", False) else None
        model_started = perf_counter()

        try:
            loaded = load_model(model_key, model_spec, model_runtime)
            if performance is not None:
                torch.cuda.synchronize(loaded.device)
                performance["phase_seconds"]["model_load"] = perf_counter() - model_started
                # Current model weights remain included in the new peak baseline.
                torch.cuda.reset_peak_memory_stats(loaded.device)
            print(
                f"[{model_key}] GPU allocated after model load: "
                f"{torch.cuda.memory_allocated(loaded.device) / (1024**3):.2f} GiB"
            )

            with _measure_phase(performance, "pca", loaded.device):
                pcas = _load_pcas_or_fit(
                    model_cfg, loaded, pca_root, force_refit=force_refit_pca
                )
            with _measure_phase(performance, "gmm", loaded.device):
                gmms = _load_gmms_or_fit(
                    model_cfg,
                    loaded,
                    pcas,
                    gmm_root,
                    pca_root,
                    expected_sample_counts,
                    force_refit=force_refit_gmm,
                )

            result_dir = output_root / cfg["outputs"]["result_dir"] / model_key
            if model_cfg.get("spectral_embedding", {}).get("enabled", True):
                try:
                    save_language_manifold(
                        gmms=gmms,
                        languages=list(model_cfg["languages"]),
                        language_names=model_cfg.get("language_names", {}),
                        output_dir=result_dir / "manifold",
                        cfg=model_cfg,
                        model_name=loaded.name,
                    )
                except Exception as exc:
                    err_path = result_dir / "manifold" / "error.txt"
                    err_path.parent.mkdir(parents=True, exist_ok=True)
                    err_path.write_text(
                        f"Spectral Embedding failed: {type(exc).__name__}: {exc}\n",
                        encoding="utf-8",
                    )
                    print(f"[{loaded.name}] WARNING: Spectral Embedding failed: {exc}")
            with _measure_phase(performance, "analysis", loaded.device):
                analysis_counts = _analyze_model(model_cfg, loaded, pcas, gmms, result_dir)
            if performance is not None:
                performance.update(analysis_counts)
                performance["elapsed_seconds"] = perf_counter() - model_started
                performance["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(loaded.device)
                performance["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(loaded.device)
                performance["peak_scope"] = "After model load, including resident model weights"
                performance["samples_per_second"] = (
                    analysis_counts["samples_analyzed"] / max(performance["phase_seconds"]["analysis"], 1e-12)
                )
                performance["generation_tokens_per_second_including_analysis"] = (
                    analysis_counts["generated_tokens"] / max(performance["phase_seconds"]["analysis"], 1e-12)
                )
                with (result_dir / "performance.json").open("w", encoding="utf-8") as handle:
                    json.dump(performance, handle, indent=2)
            del pcas, gmms
            gc.collect()
        except Exception as exc:
            if not smoke_continue:
                raise
            message = f"{type(exc).__name__}: {exc}"
            print(f"[{model_key}] SMOKE TEST ERROR; continuing to next model: {message}")
            model_errors.append({"model": model_key, "error": message})
            # Release partial allocations after model-load/PCA/GMM failures.
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            gc.collect()
        finally:
            if loaded is not None:
                try:
                    release_model(loaded)
                except Exception as cleanup_exc:
                    print(f"[{model_key}] WARNING: cleanup failed: {cleanup_exc}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    if model_errors:
        errors_path = output_root / "manifests" / "smoke_model_errors.json"
        with errors_path.open("w", encoding="utf-8") as f:
            json.dump(model_errors, f, ensure_ascii=False, indent=2)
        print(f"Smoke-test model errors saved to: {errors_path}")

    effective_cfg = json.loads(json.dumps(cfg))
    effective_cfg.pop("_repo_root", None)
    effective_cfg.setdefault("runtime", {})["pca_data_path"] = str(pca_root.resolve())
    with (output_root / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(effective_cfg, f, ensure_ascii=False, indent=2)
    print("\nExperiment completed.")

