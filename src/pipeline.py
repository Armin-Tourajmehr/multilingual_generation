from __future__ import annotations

import gc
import hashlib
import json
from copy import deepcopy
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
    forward_generated_texts_with_context,
    load_model,
    release_model,
)

from .lexical import StanzaTagger, align_pos_to_token
from .manifold import save_language_manifold

from .pca_gpu import (
    GPUPCA,
    GPUPCACovarianceAccumulator,
    load_pca,
    pca_paths,
    save_pca,
)


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
        "model_key": loaded.name,
        "model_name": loaded.spec.get("model_name"),
        "model_revision": _model_revision(loaded),
        "torch_dtype": loaded.spec.get("torch_dtype", "auto"),
        "mixed_precision": cfg["runtime"].get("mixed_precision", "fp16"),
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


def _gmm_fingerprint(
    cfg: dict[str, Any], loaded: LoadedModel, expected_sample_counts: dict[str, int]
) -> str:
    payload = {
        "pca_fingerprint": _pca_fingerprint(cfg, loaded),
        "languages": list(cfg["languages"]),
        "aya_root": str(cfg["data"].get("root_dir")),
        "input_column": cfg["data"].get("input_column"),
        "sample_id_column": cfg["data"].get("sample_id_column"),
        "aya_sample_cap": int(cfg["data"].get("max_samples_per_language", 2000)),
        "sampling_seed": int(cfg["data"].get("sampling_seed", cfg["project"].get("seed", 42))),
        "sample_counts": {k: int(v) for k, v in expected_sample_counts.items()},
        "gmm_reg_covar": float(cfg["representation"].get("gmm_reg_covar", 1e-6)),
        "prior_type": "empirical_sample_count",
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
    force_refit: bool = False,
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

        # Important:
        # priors are NOT uniform.
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
        "fingerprint": _gmm_fingerprint(cfg, loaded, expected_sample_counts),
        "pca_fingerprint": _pca_fingerprint(cfg, loaded),
    }

    with (
        model_dir / "metadata.json"
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
    """
    Load saved Aya GMMs if they match the current
    empirical-prior configuration.

    Otherwise fit them on all Aya samples.
    """

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
                == "empirical_sample_count"
            )

            same_sample_counts = (
                meta.get(
                    "sample_counts"
                )
                == expected_counts
            )
            same_fingerprint = (
                meta.get("fingerprint")
                == _gmm_fingerprint(cfg, loaded, expected_sample_counts)
            )

            if (
                same_pca
                and same_langs
                and same_coverage
                and same_prior_type
                and same_sample_counts
                and same_fingerprint
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
        force_refit=force_refit,
    )


# ============================================================
# ANALYSIS
# ============================================================

def _analyze_model(
    cfg: dict[str, Any],
    loaded: LoadedModel,
    pcas: list[GPUPCA],
    gmms: list[GPULanguageGMM],
    result_dir: Path,
):
    """Generate text, then forward the exact decoded output for token/POS analysis."""
    result_dir.mkdir(parents=True, exist_ok=True)
    gen_cfg = cfg["generation"]
    batch_size = int(cfg["runtime"].get("generation_batch_size", 1))
    languages = list(cfg["languages"])
    posterior_columns = [f"P_{language}" for language in languages]

    token_writers: dict[str, PartitionWriter] = {}
    sentence_writers: dict[str, PartitionWriter] = {}
    output_writers: dict[str, PartitionWriter] = {}
    counts = {lang: 0 for lang in languages}

    lexical_cfg = dict(cfg.get("lexical_analysis", {}))
    lexical_enabled = bool(lexical_cfg.get("enabled", True))
    stanza_dir = Path(lexical_cfg.get("stanza_model_dir", "outputs/stanza_resources"))
    if not stanza_dir.is_absolute():
        stanza_dir = Path(cfg.get("_repo_root", Path.cwd())) / stanza_dir
    lexical_cfg["stanza_model_dir"] = str(stanza_dir)
    tagger = StanzaTagger(lexical_cfg) if lexical_enabled else None

    try:
        print(f"[{loaded.name}] Aya pass 2/2: generation + token posterior/POS analysis")

        for language in languages:
            # Keep only small aggregates in memory; all token rows are streamed to Parquet.
            pos_buckets: dict[tuple[int, str, str], dict[str, Any]] = {}

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
                output_rows = [
                    {
                        "model": loaded.name,
                        "sample_id": sid,
                        "input_language": language,
                        "prompt": prompt,
                        "generated_text": text,
                        "generated_token_count": len(ids),
                    }
                    for sid, prompt, text, ids in zip(sample_ids, prompts, generated_texts, generated_ids)
                ]
                outputs_path = result_dir / "generated_outputs" / f"{language}.parquet"
                output_writers.setdefault(language, PartitionWriter(outputs_path)).write(output_rows)

                # A separate teacher-forced forward pass gives each output token its own
                # contextual hidden state; generation-time step states predict the next token.
                hidden_states, output_token_ids, offsets, attention = forward_generated_texts_with_context(
                    loaded,
                    prompt_token_ids,
                    generated_ids,
                    generated_texts,
                    cfg["runtime"],
                )

                per_sample_refs: list[list[dict[str, Any]]] = []
                for sample_idx, text in enumerate(generated_texts):
                    if tagger is not None and text.strip():
                        words, tag_status = tagger.tag_text(text, language)
                    elif not lexical_enabled:
                        words, tag_status = [], "disabled"
                    else:
                        words, tag_status = [], "empty"

                    refs: list[dict[str, Any]] = []
                    for seq_pos, (token_id, span, is_attended) in enumerate(
                        zip(output_token_ids[sample_idx], offsets[sample_idx], attention[sample_idx])
                    ):
                        start_char, end_char = int(span[0]), int(span[1])
                        if not is_attended or end_char <= start_char:
                            continue  # padding/special tokens have no text span
                        if start_char < 0 or end_char > len(text):
                            continue
                        token_text = text[start_char:end_char]
                        if not token_text or not token_text.strip():
                            continue
                        pos_info = align_pos_to_token(text, start_char, end_char, words, tag_status)
                        try:
                            model_token = loaded.tokenizer.convert_ids_to_tokens(int(token_id))
                            if isinstance(model_token, list):
                                model_token = "".join(str(x) for x in model_token)
                        except Exception:
                            model_token = loaded.tokenizer.decode([int(token_id)], skip_special_tokens=False)
                        refs.append({
                            "sample_index": sample_idx,
                            "sequence_position": seq_pos,
                            "token_id": int(token_id),
                            "token": token_text,
                            "model_token": str(model_token),
                            "char_start": start_char,
                            "char_end": end_char,
                            **pos_info,
                        })
                    per_sample_refs.append(refs)

                sentence_rows: list[dict[str, Any]] = []
                all_token_rows: list[dict[str, Any]] = []
                n_layers = min(len(hidden_states), len(pcas), len(gmms))

                for layer in range(n_layers):
                    flat_refs = [ref for refs in per_sample_refs for ref in refs]
                    if not flat_refs:
                        for sample_idx, (sid, _text) in enumerate(zip(sample_ids, generated_texts)):
                            sentence_rows.append({
                                "model": loaded.name,
                                "sample_id": sid,
                                "input_language": language,
                                "layer": layer,
                                "num_output_tokens": 0,
                                **{col: 0.0 for col in posterior_columns},
                            })
                        continue

                    vectors = torch.stack([
                        hidden_states[layer][ref["sample_index"], ref["sequence_position"], :]
                        for ref in flat_refs
                    ])
                    projected = pcas[layer].transform(vectors)
                    # Bound temporary [N, language, PCA-dimension] posterior tensors.
                    post_parts = [
                        gmms[layer].posterior(projected[i:i + 512])
                        for i in range(0, projected.shape[0], 512)
                    ]
                    posterior = torch.cat(post_parts, dim=0) if post_parts else projected.new_empty((0, len(languages)))
                    post_values = posterior.detach().cpu().tolist()
                    del vectors, projected, post_parts, posterior

                    sample_posteriors: list[list[list[float]]] = [[] for _ in sample_ids]
                    for ref, values in zip(flat_refs, post_values):
                        sample_idx = ref["sample_index"]
                        row = {
                            "model": loaded.name,
                            "sample_id": sample_ids[sample_idx],
                            "input_language": language,
                            "token_position": len(sample_posteriors[sample_idx]),
                            "sequence_position": ref["sequence_position"],
                            "token": ref["token"],
                            "model_token": ref["model_token"],
                            "token_id": ref["token_id"],
                            "char_start": ref["char_start"],
                            "char_end": ref["char_end"],
                            "word_text": ref["word_text"],
                            "sentence_id": ref["sentence_id"],
                            "word_id": ref["word_id"],
                            "upos": ref["upos"],
                            "xpos": ref["xpos"],
                            "feats": ref["feats"],
                            "pos_status": ref["pos_status"],
                            "layer": layer,
                        }
                        row.update({col: float(value) for col, value in zip(posterior_columns, values)})
                        sample_posteriors[sample_idx].append(values)
                        all_token_rows.append(row)

                        key = (layer, str(ref["upos"]), str(ref["pos_status"]))
                        bucket = pos_buckets.setdefault(key, {"n": 0, "sums": {col: 0.0 for col in posterior_columns}})
                        bucket["n"] += 1
                        for col, value in zip(posterior_columns, values):
                            bucket["sums"][col] += float(value)

                    for sample_idx, sid in enumerate(sample_ids):
                        sample_values = sample_posteriors[sample_idx]
                        if sample_values:
                            n_values = len(sample_values)
                            means = [sum(vals[j] for vals in sample_values) / n_values for j in range(len(languages))]
                        else:
                            n_values = 0
                            means = [0.0] * len(languages)
                        sentence_rows.append({
                            "model": loaded.name,
                            "sample_id": sid,
                            "input_language": language,
                            "layer": layer,
                            "num_output_tokens": n_values,
                            **{col: float(value) for col, value in zip(posterior_columns, means)},
                        })

                token_path = result_dir / "token_posteriors" / f"{language}.parquet"
                token_writers.setdefault(language, PartitionWriter(token_path)).write(all_token_rows)
                sentence_path = result_dir / "sentence_level_posteriors" / f"{language}.parquet"
                sentence_writers.setdefault(language, PartitionWriter(sentence_path)).write(sentence_rows)

                del hidden_states, per_sample_refs, all_token_rows, sentence_rows
                gc.collect()

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
            summary_path = result_dir / "lexical_analysis" / "pos_posterior_summary" / f"{language}.csv"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
            if tagger is not None:
                # Only one source-language Stanza pipeline is retained at a time.
                tagger.close()

    finally:
        for writers in (token_writers, sentence_writers, output_writers):
            for writer in writers.values():
                writer.close()
        if tagger is not None:
            tagger.close()

    pd.DataFrame([
        {"language": lang, "num_samples_analyzed": count}
        for lang, count in counts.items()
    ]).to_csv(result_dir / "analysis_counts.csv", index=False)


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

        try:
            loaded = load_model(model_key, model_spec, model_runtime)
            print(
                f"[{model_key}] GPU allocated after model load: "
                f"{torch.cuda.memory_allocated(loaded.device) / (1024**3):.2f} GiB"
            )

            pcas = _load_pcas_or_fit(
                model_cfg, loaded, pca_root, force_refit=force_refit_pca
            )
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
            _analyze_model(model_cfg, loaded, pcas, gmms, result_dir)
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

