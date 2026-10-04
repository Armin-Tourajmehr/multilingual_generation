from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

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
    release_model,
)

from .pca_gpu import (
    GPUPCA,
    GPUPCACovarianceAccumulator,
    load_pca,
    pca_paths,
    save_pca,
)


# ============================================================
# PARQUET WRITER
# ============================================================

class PartitionWriter:
    """
    Buffered Parquet writer.

    Instead of creating a pandas DataFrame and writing a tiny
    Parquet row-group for every generation batch, rows are
    accumulated and periodically flushed.

    This preserves all rows while significantly reducing
    Python/Arrow/I/O overhead.
    """

    def __init__(
        self,
        path: Path,
        buffer_size: int = 8192,
    ):
        self.path = path
        self.buffer_size = int(
            buffer_size
        )

        self.writer = None
        self.buffer: list[
            dict[str, Any]
        ] = []

    def write(
        self,
        rows: list[dict[str, Any]],
    ):
        if not rows:
            return

        self.buffer.extend(rows)

        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return

        table = pa.Table.from_pylist(
            self.buffer
        )

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

        self.writer.write_table(
            table
        )

        self.buffer.clear()

    def close(self):
        self.flush()

        if self.writer is not None:
            self.writer.close()
            self.writer = None


# ============================================================
# DATA HELPERS
# ============================================================

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
            for i, c in enumerate(
                language
            )
        ),
    ):

        texts.append(
            row["text"]
        )

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

    if loaded.device.type != "cuda":
        raise RuntimeError(
            "PCA fitting requires CUDA."
        )

    rep = cfg[
        "representation"
    ]

    n_components = int(
        rep["pca_dim"]
    )

    batch_size = int(
        cfg["runtime"][
            "wikipedia_batch_size"
        ]
    )

    max_tokens = int(
        cfg["wikipedia"][
            "max_tokens_per_document"
        ]
    )

    accumulators = None

    reference_document_counts = {
        lang: 0
        for lang in cfg["languages"]
    }

    print(
        f"[{loaded.name}] "
        f"Wikipedia pass: fitting GPU PCA "
        f"({n_components} components)"
    )

    for language in cfg[
        "languages"
    ]:

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
                    hidden_states[
                        0
                    ].shape[-1]
                )

                n_layers = len(
                    hidden_states
                )

                accumulators = [
                    GPUPCACovarianceAccumulator(
                        hidden_dim,
                        loaded.device,
                    )
                    for _ in range(
                        n_layers
                    )
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

                accumulators[
                    layer
                ].update(X)

            del hidden_states
            del mask

    if accumulators is None:
        raise RuntimeError(
            "Wikipedia stream produced "
            "no batches."
        )

    pcas = [
        acc.finalize(
            n_components
        )
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
            for lang in cfg[
                "languages"
            ]
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
        model_pca_dir
        / "metadata.json"
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

    for layer, pca in enumerate(
        pcas
    ):

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

    model_pca_dir = (
        pca_root / loaded.name
    )

    paths = pca_paths(
        pca_root,
        loaded.name,
    )

    expected_layers = (
        int(
            getattr(
                loaded.model.config,
                "num_hidden_layers",
            )
        )
        + 1
    )

    if (
        not force_refit
        and len(paths)
        == expected_layers
    ):

        pcas = [
            load_pca(
                path,
                loaded.device,
            )
            for path in paths
        ]

        pca_dim = int(
            cfg["representation"][
                "pca_dim"
            ]
        )

        if any(
            p.n_components
            != pca_dim
            for p in pcas
        ):
            raise RuntimeError(
                f"PCA files under "
                f"{model_pca_dir} do not match "
                f"pca_dim={pca_dim}."
            )

        print(
            f"[{loaded.name}] "
            f"Reusing PCA data from: "
            f"{model_pca_dir.resolve()}"
        )

        return pcas

    if (
        paths
        and not force_refit
        and len(paths)
        != expected_layers
    ):

        raise RuntimeError(
            f"Incomplete PCA data under "
            f"{model_pca_dir}: "
            f"found {len(paths)} layer files, "
            f"expected {expected_layers}. "
            f"Use --force-refit-pca "
            f"to rebuild it."
        )

    return _fit_pca_on_wikipedia(
        cfg,
        loaded,
        pca_root,
    )


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
    force_refit: bool = False,
) -> list[GPULanguageGMM]:

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

    sample_counts = {
        lang: 0
        for lang in languages
    }

    sums, sums_sq, counts = (
        init_gmm_stats(
            n_layers=len(pcas),
            n_languages=len(
                languages
            ),
            pca_dim=pca_dim,
            device=loaded.device,
        )
    )

    print(
        f"[{loaded.name}] "
        f"Aya pass 1/2: "
        f"fitting GMMs on ALL Aya samples"
    )

    for language in languages:

        language_idx = (
            lang_to_idx[
                language
            ]
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
                    ) or None,
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

                Z = pca.transform(
                    X
                )

                update_gmm_stats(
                    sums,
                    sums_sq,
                    counts,
                    layer,
                    language_idx,
                    Z,
                )

            del hidden_states
            del mask

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

    language_priors = (
        build_language_priors(
            languages=languages,
            sample_counts=(
                expected_sample_counts
            ),
            device=loaded.device,
        )
    )

    print(
        f"[{loaded.name}] "
        f"Empirical language priors:"
    )

    for language, prior in zip(
        languages,
        language_priors.tolist(),
    ):

        print(
            f"    {language}: "
            f"{prior:.6f}"
        )

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

    model_dir = (
        gmm_root / loaded.name
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        )

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
        "prior_type": (
            "empirical_sample_count"
        ),
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
    }

    with (
        model_dir
        / "metadata.json"
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

    del sums
    del sums_sq
    del counts

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
        model_dir
        / "metadata.json"
    )

    expected_counts = {
        language: int(
            expected_sample_counts[
                language
            ]
        )
        for language in cfg[
            "languages"
        ]
    }

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
                == "empirical_sample_count"
            )

            same_sample_counts = (
                meta.get(
                    "sample_counts"
                )
                == expected_counts
            )

            if (
                same_pca
                and same_langs
                and same_coverage
                and same_prior_type
                and same_sample_counts
            ):

                print(
                    f"[{loaded.name}] "
                    f"Reusing saved Aya GMMs "
                    f"from: "
                    f"{model_dir.resolve()}"
                )

                return [
                    load_gmm(
                        path,
                        loaded.device,
                    )
                    for path in paths
                ]

            print(
                f"[{loaded.name}] "
                f"Existing Aya GMMs do not "
                f"match the current "
                f"empirical-prior setup. "
                f"Refitting."
            )

        except Exception as exc:

            print(
                f"[{loaded.name}] "
                f"Saved GMM metadata could "
                f"not be validated "
                f"({exc}); refitting."
            )

    return _fit_aya_gmms(
        cfg,
        loaded,
        pcas,
        gmm_root,
        pca_root,
        expected_sample_counts,
        force_refit=force_refit,
    )


# ============================================================
# ANALYSIS HELPERS
# ============================================================

def _analyze_model(
    cfg: dict[str, Any],
    loaded: LoadedModel,
    pcas: list[GPUPCA],
    gmms: list[GPULanguageGMM],
    result_dir: Path,
):

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    gen_cfg = cfg[
        "generation"
    ]

    batch_size = int(
        cfg["runtime"].get(
            "generation_batch_size",
            1,
        )
    )

    if batch_size < 1:
        raise ValueError(
            "generation_batch_size "
            "must be >= 1."
        )

    # --------------------------------------------------------
    # Writers
    # --------------------------------------------------------

    token_writers: dict[
        str,
        PartitionWriter,
    ] = {}

    sentence_writers: dict[
        str,
        PartitionWriter,
    ] = {}

    output_writers: dict[
        str,
        PartitionWriter,
    ] = {}

    counts = {
        lang: 0
        for lang in cfg[
            "languages"
        ]
    }

    pcols = [
        f"P_{x}"
        for x in cfg[
            "languages"
        ]
    ]

    token_decode_cache: dict[
        int,
        str,
    ] = {}

    def decode_token(
        token_id: int,
    ) -> str:

        token_id = int(
            token_id
        )

        cached = token_decode_cache.get(
            token_id
        )

        if cached is not None:
            return cached

        text = loaded.tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
        )

        token_decode_cache[
            token_id
        ] = text

        return text

    # --------------------------------------------------------
    # Buffered writer size
    # --------------------------------------------------------

    buffer_size = int(
        cfg["runtime"].get(
            "analysis_write_buffer",
            8192,
        )
    )

    try:

        print(
            f"[{loaded.name}] "
            f"Aya pass 2/2: "
            f"generation + posterior analysis"
        )

        print(
            f"[{loaded.name}] "
            f"Generation batch size: "
            f"{batch_size}"
        )

        for language in cfg[
            "languages"
        ]:

            # ------------------------------------------------
            # Create writers once per language
            # ------------------------------------------------

            output_writers[
                language
            ] = PartitionWriter(
                result_dir
                / "generated_outputs"
                / f"{language}.parquet",
                buffer_size=buffer_size,
            )

            token_writers[
                language
            ] = PartitionWriter(
                result_dir
                / "token_posteriors"
                / f"{language}.parquet",
                buffer_size=buffer_size,
            )

            sentence_writers[
                language
            ] = PartitionWriter(
                result_dir
                / "sentence_level_posteriors"
                / f"{language}.parquet",
                buffer_size=buffer_size,
            )

            # ------------------------------------------------
            # Aya
            # ------------------------------------------------

            for chunk in tqdm(
                iter_aya_batches(
                    cfg,
                    language,
                    batch_size,
                ),
                desc=(
                    f"{loaded.name} "
                    f"Aya/analysis/{language}"
                ),
            ):

                prompts = chunk[
                    cfg["data"][
                        "input_column"
                    ]
                ].tolist()

                sample_ids = chunk[
                    cfg["data"][
                        "sample_id_column"
                    ]
                ].tolist()

                (
                    token_ids,
                    generated_texts,
                    hidden_states,
                    input_width,
                    lengths,
                ) = generate_batch(
                    loaded,
                    prompts,
                    gen_cfg,
                    cfg["runtime"],
                )

                counts[
                    language
                ] += len(sample_ids)

                max_generated = max(
                    lengths,
                    default=0,
                )

                if max_generated == 0:

                    del hidden_states

                    continue

                # ------------------------------------------------
                # Generated outputs
                # ------------------------------------------------

                output_rows = [
                    {
                        "model": loaded.name,
                        "sample_id": sid,
                        "input_language": language,
                        "prompt": prompt,
                        "generated_text": gen_text,
                    }
                    for sid, prompt, gen_text
                    in zip(
                        sample_ids,
                        prompts,
                        generated_texts,
                    )
                ]

                output_writers[
                    language
                ].write(
                    output_rows
                )

                # ------------------------------------------------
                # Token posterior analysis
                # ------------------------------------------------
                #
                # Every layer is analyzed.
                #
                # Every valid generated token is analyzed.
                #
                # Every token gets:
                #
                # P(ar), P(bn), ..., P(vi)
                #
                # ------------------------------------------------

                for layer, (
                    pca,
                    gmm,
                ) in enumerate(
                    zip(
                        pcas,
                        gmms,
                    )
                ):

                    layer_hidden = (
                        hidden_states[
                            layer
                        ][
                            :,
                            :max_generated,
                            :,
                        ]
                    )

                    # Flatten valid tokens:
                    #
                    # [B, T, D]
                    #       ↓
                    # [N_valid, D]
                    valid_mask = (
                        torch.arange(
                            max_generated,
                            device=loaded.device,
                        ).unsqueeze(0)
                        < torch.tensor(
                            lengths,
                            device=loaded.device,
                            dtype=torch.long,
                        ).unsqueeze(1)
                    )

                    valid_hidden = (
                        layer_hidden[
                            valid_mask
                        ]
                    )

                    if valid_hidden.numel() == 0:
                        continue

                    # ------------------------------------------------
                    # PCA
                    # ------------------------------------------------

                    Z = pca.transform(
                        valid_hidden
                    )

                    # ------------------------------------------------
                    # GMM posterior
                    # ------------------------------------------------

                    post = gmm.posterior(
                        Z
                    )

                    # ------------------------------------------------
                    # One GPU → CPU transfer for ALL token
                    # posterior values in this layer.
                    #
                    # Previously there were additional CPU
                    # synchronizations for sentence means.
                    # ------------------------------------------------

                    post_cpu = (
                        post
                        .detach()
                        .cpu()
                        .tolist()
                    )

                    # Sentence means are calculated on GPU.
                    #
                    # We use the same token posterior values,
                    # so sentence-level results remain unchanged.
                    # ------------------------------------------------

                    sentence_means = []

                    offset = 0

                    for sample_len in lengths:
                        if sample_len == 0:
                            sentence_means.append(None)
                        else:
                            sentence_means.append(
                                post[
                                    offset:offset + sample_len
                                ].mean(dim=0)
                            )
                            offset += sample_len

                    non_empty_means = [
                        x for x in sentence_means
                        if x is not None
                    ]

                    if non_empty_means:
                        sentence_means_cpu = (
                            torch.stack(
                                non_empty_means,
                                dim=0,
                            )
                            .detach()
                            .cpu()
                            .tolist()
                        )
                    else:
                        sentence_means_cpu = []

                    # ------------------------------------------------
                    # Build rows
                    # ------------------------------------------------

                    offset = 0
                    mean_index = 0

                    token_rows = []
                    sentence_rows = []

                    for sample_index, (
                        sid,
                        ids,
                        sample_len,
                    ) in enumerate(
                        zip(
                            sample_ids,
                            token_ids,
                            lengths,
                        )
                    ):

                        if sample_len == 0:
                            continue

                        # --------------------------------------------
                        # Sentence posterior
                        # --------------------------------------------

                        means = (
                            sentence_means_cpu[
                                mean_index
                            ]
                        )

                        sentence_rows.append(
                            {
                                "model": loaded.name,
                                "sample_id": sid,
                                "input_language": language,
                                "layer": layer,
                                **{
                                    col: float(value)
                                    for col, value
                                    in zip(
                                        pcols,
                                        means,
                                    )
                                },
                            }
                        )

                        mean_index += 1

                        # --------------------------------------------
                        # Token posterior
                        # --------------------------------------------

                        sample_post_cpu = (
                            post_cpu[
                                offset:
                                offset
                                + sample_len
                            ]
                        )

                        for position, (
                            token_id,
                            posterior_values,
                        ) in enumerate(
                            zip(
                                ids,
                                sample_post_cpu,
                            )
                        ):

                            token_text = (
                                decode_token(
                                    token_id
                                )
                            )

                            row = {
                                "model": loaded.name,
                                "sample_id": sid,
                                "input_language": language,
                                "token_position": position,
                                "sequence_position": position,
                                "token": token_text,
                                "token_id": int(
                                    token_id
                                ),
                                "layer": layer,
                            }

                            row.update(
                                {
                                    col: float(value)
                                    for col, value
                                    in zip(
                                        pcols,
                                        posterior_values,
                                    )
                                }
                            )

                            token_rows.append(
                                row
                            )

                        offset += sample_len

                    # ------------------------------------------------
                    # Write this layer's results into buffered
                    # Parquet writers.
                    #
                    # They are NOT physically written for every
                    # generation batch because PartitionWriter
                    # buffers rows.
                    # ------------------------------------------------

                    token_writers[
                        language
                    ].write(
                        token_rows
                    )

                    sentence_writers[
                        language
                    ].write(
                        sentence_rows
                    )

                    del Z
                    del post
                    del post_cpu
                    del valid_hidden
                    del layer_hidden
                    del valid_mask
                    del sentence_means
                    del sentence_means_cpu

                del hidden_states

    finally:

        for writers in (
            token_writers,
            sentence_writers,
            output_writers,
        ):

            for writer in writers.values():
                writer.close()

    # --------------------------------------------------------
    # Analysis counts
    # --------------------------------------------------------

    pd.DataFrame(
        [
            {
                "language": lang,
                "num_samples_analyzed": count,
            }
            for lang, count in counts.items()
        ]
    ).to_csv(
        result_dir
        / "analysis_counts.csv",
        index=False,
    )


# ============================================================
# MAIN RUN
# ============================================================

def run(
    cfg: dict[str, Any],
    pca_path: str | Path | None = None,
    force_refit_pca: bool = False,
    force_refit_gmm: bool = False,
) -> None:

    output_root = Path(
        cfg["outputs"]["root_dir"]
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        output_root / "manifests"
    ).mkdir(
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Validate Aya
    # ---------------------------------------------------------

    validation = validate_aya_files(
        cfg
    )

    validation.to_csv(
        output_root
        / "manifests"
        / "aya_input_summary.csv",
        index=False,
    )

    print(
        "Aya input summary:"
    )

    print(
        validation.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # PCA root
    # ---------------------------------------------------------

    default_pca_root = Path(
        cfg["runtime"].get(
            "pca_data_path"
        )
        or (
            output_root
            / "pca_data"
        )
    )

    pca_root = (
        Path(pca_path)
        if pca_path is not None
        else default_pca_root
    )

    if not pca_root.is_absolute():

        pca_root = (
            Path.cwd()
            / pca_root
        )

    pca_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"PCA data root: "
        f"{pca_root.resolve()}"
    )

    # ---------------------------------------------------------
    # GMM root
    # ---------------------------------------------------------

    gmm_root = (
        output_root
        / cfg["outputs"]["gmm_dir"]
    )

    # ---------------------------------------------------------
    # Enabled models
    # ---------------------------------------------------------

    for model_key, model_spec in [
        (k, v)
        for k, v in cfg["models"].items()
        if v.get(
            "enabled",
            False,
        )
    ]:

        print(
            "\n"
            + "=" * 90
        )

        print(
            f"MODEL: {model_key} | "
            f"{model_spec['model_name']}"
        )

        print(
            "=" * 90
        )

        loaded = load_model(
            model_key,
            model_spec,
            cfg["runtime"],
        )

        try:

            print(
                f"[{model_key}] "
                f"GPU allocated after "
                f"model load: "
                f"{torch.cuda.memory_allocated(loaded.device) / (1024**3):.2f} GiB"
            )

            # -------------------------------------------------
            # PCA
            # -------------------------------------------------

            pcas = _load_pcas_or_fit(
                cfg,
                loaded,
                pca_root,
                force_refit=(
                    force_refit_pca
                ),
            )

            # -------------------------------------------------
            # Expected Aya sample counts
            # -------------------------------------------------

            expected_sample_counts = dict(
                zip(
                    validation[
                        "language"
                    ],
                    validation[
                        "num_samples"
                    ],
                )
            )

            # -------------------------------------------------
            # GMM
            # -------------------------------------------------

            gmms = _load_gmms_or_fit(
                cfg,
                loaded,
                pcas,
                gmm_root,
                pca_root,
                expected_sample_counts,
                force_refit=(
                    force_refit_gmm
                ),
            )

            # -------------------------------------------------
            # Pass 2
            # -------------------------------------------------

            result_dir = (
                output_root
                / cfg["outputs"][
                    "result_dir"
                ]
                / model_key
            )

            _analyze_model(
                cfg,
                loaded,
                pcas,
                gmms,
                result_dir,
            )

            del pcas
            del gmms

        finally:

            release_model(
                loaded
            )

    # ---------------------------------------------------------
    # Save effective configuration
    # ---------------------------------------------------------

    with (
        output_root
        / "run_config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        effective_cfg = json.loads(
            json.dumps(cfg)
        )

        effective_cfg.setdefault(
            "runtime",
        )[
            "pca_data_path"
        ] = str(
            pca_root.resolve()
        )

        json.dump(
            effective_cfg,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\nExperiment completed."
    )