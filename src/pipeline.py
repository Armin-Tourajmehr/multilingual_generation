from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from .data import iter_aya_batches, iter_wikipedia_documents, validate_aya_files
from .gmm import (
    GPULanguageGMM,
    finalize_gmm,
    init_gmm_stats,
    save_gmm,
    load_gmm,
    update_gmm_stats,
)
from .modeling import LoadedModel, forward_text_batch, generate_batch, load_model, release_model
from .pca_gpu import GPUPCA, GPUPCACovarianceAccumulator, load_pca, pca_paths, save_pca


class PartitionWriter:
    """Append row batches to one compressed Parquet file."""

    def __init__(self, path: Path):
        self.path = path
        self.writer = None

    def write(self, rows: list[dict[str, Any]]):
        if not rows:
            return
        table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="zstd")
        self.writer.write_table(table)

    def close(self):
        if self.writer is not None:
            self.writer.close()


def _language_batches(cfg: dict[str, Any], language: str, batch_size: int) -> Iterator[list[str]]:
    texts: list[str] = []
    for row in iter_wikipedia_documents(
        cfg,
        language,
        seed_offset=sum((i + 1) * ord(c) for i, c in enumerate(language)),
    ):
        texts.append(row["text"])
        if len(texts) == batch_size:
            yield texts
            texts = []
    if texts:
        yield texts


def _fit_pca_on_wikipedia(cfg: dict[str, Any], loaded: LoadedModel, pca_root: Path) -> list[GPUPCA]:
    """Fit exact covariance PCA on the full configured Wikipedia reference stream.

    All vector math stays on CUDA. The reference corpus is streamed once and
    covariance sufficient statistics are accumulated on GPU for every layer.
    """

    if loaded.device.type != "cuda":
        raise RuntimeError("PCA fitting requires CUDA.")

    rep = cfg["representation"]
    n_components = int(rep["pca_dim"])
    batch_size = int(cfg["runtime"]["wikipedia_batch_size"])
    max_tokens = int(cfg["wikipedia"]["max_tokens_per_document"])

    accumulators: list[GPUPCACovarianceAccumulator] | None = None
    reference_document_counts = {lang: 0 for lang in cfg["languages"]}

    print(f"[{loaded.name}] Wikipedia pass: fitting GPU PCA ({n_components} components)")
    for language in cfg["languages"]:
        for texts in tqdm(
            _language_batches(cfg, language, batch_size),
            desc=f"{loaded.name} Wikipedia/PCA/{language}",
        ):
            reference_document_counts[language] += len(texts)
            hidden_states, mask = forward_text_batch(loaded, texts, max_tokens, cfg["runtime"])

            if accumulators is None:
                hidden_dim = int(hidden_states[0].shape[-1])
                n_layers = len(hidden_states)
                accumulators = [
                    GPUPCACovarianceAccumulator(hidden_dim, loaded.device)
                    for _ in range(n_layers)
                ]
                if n_components > hidden_dim:
                    raise ValueError(
                        f"PCA components ({n_components}) exceed hidden dimension ({hidden_dim}) for {loaded.name}."
                    )
                print(f"[{loaded.name}] hidden_dim={hidden_dim}, hidden_states={n_layers}")

            for layer, hidden in enumerate(hidden_states):
                X = hidden[mask]
                accumulators[layer].update(X)

            del hidden_states, mask

    if accumulators is None:
        raise RuntimeError("Wikipedia stream produced no batches.")

    pcas = [acc.finalize(n_components) for acc in accumulators]
    model_pca_dir = pca_root / loaded.name
    model_pca_dir.mkdir(parents=True, exist_ok=True)

    count_df = pd.DataFrame(
        [{"language": lang, "num_documents": reference_document_counts[lang]} for lang in cfg["languages"]]
    )
    count_df.to_csv(model_pca_dir / "wikipedia_document_counts.csv", index=False)

    metadata = {
        "model": loaded.name,
        "pca_method": "global covariance eigendecomposition on GPU",
        "pca_components": n_components,
        "languages": list(cfg["languages"]),
        "wikipedia_max_samples_per_language": int(cfg["wikipedia"]["max_samples_per_language"]),
        "wikipedia_max_tokens_per_document": max_tokens,
    }
    with (model_pca_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    for layer, pca in enumerate(pcas):
        save_pca(model_pca_dir / f"layer_{layer:03d}.pt", pca)

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

    if not force_refit and len(paths) == expected_layers:
        pcas = [load_pca(path, loaded.device) for path in paths]
        pca_dim = int(cfg["representation"]["pca_dim"])
        if any(p.n_components != pca_dim for p in pcas):
            raise RuntimeError(f"PCA files under {model_pca_dir} do not match pca_dim={pca_dim}.")
        print(f"[{loaded.name}] Reusing PCA data from: {model_pca_dir.resolve()}")
        return pcas

    if paths and not force_refit and len(paths) != expected_layers:
        raise RuntimeError(
            f"Incomplete PCA data under {model_pca_dir}: found {len(paths)} layer files, expected {expected_layers}. "
            "Use --force-refit-pca to rebuild it."
        )
    return _fit_pca_on_wikipedia(cfg, loaded, pca_root)


def _fit_aya_gmms(
    cfg: dict[str, Any],
    loaded: LoadedModel,
    pcas: list[GPUPCA],
    gmm_root: Path,
    pca_root: Path,
    expected_sample_counts: dict[str, int],
    force_refit: bool = False,
) -> list[GPULanguageGMM]:
    """Fit one diagonal, one-component GMM per language on 100% of Aya."""

    languages = list(cfg["languages"])
    lang_to_idx = {lang: i for i, lang in enumerate(languages)}
    pca_dim = int(cfg["representation"]["pca_dim"])
    batch_size = int(cfg["runtime"]["evaluation_batch_size"])

    sample_counts = {lang: 0 for lang in languages}

    sums, sums_sq, counts = init_gmm_stats(
        n_layers=len(pcas),
        n_languages=len(languages),
        pca_dim=pca_dim,
        device=loaded.device,
    )

    print(f"[{loaded.name}] Aya pass 1/2: fitting GMMs on ALL Aya samples")
    for language in languages:
        language_idx = lang_to_idx[language]
        for chunk in tqdm(
            iter_aya_batches(cfg, language, batch_size),
            desc=f"{loaded.name} Aya/GMM/{language}",
        ):
            sample_counts[language] += len(chunk)
            prompts = chunk[cfg["data"]["input_column"]].tolist()
            hidden_states, mask = forward_text_batch(
                loaded,
                prompts,
                cfg["runtime"].get("max_input_tokens") or None,
                cfg["runtime"],
            )
            for layer, pca in enumerate(pcas):
                X = hidden_states[layer][mask]
                if X.numel() == 0:
                    continue
                Z = pca.transform(X)
                update_gmm_stats(sums, sums_sq, counts, layer, language_idx, Z)
            del hidden_states, mask

    if sample_counts != expected_sample_counts:
        raise RuntimeError(
            f"Aya full-data audit failed. Expected samples={expected_sample_counts}, "
            f"processed={sample_counts}"
        )

    gmms = finalize_gmm(
        sums,
        sums_sq,
        counts,
        languages=languages,
        reg_covar=float(cfg["representation"]["gmm_reg_covar"]),
        equal_priors=bool(cfg["representation"]["equal_language_priors"]),
    )

    model_dir = gmm_root / loaded.name
    model_dir.mkdir(parents=True, exist_ok=True)
    for layer, gmm in enumerate(gmms):
        save_gmm(model_dir / f"layer_{layer:03d}.pt", gmm)

    counts_df = pd.DataFrame(
        [
            {
                "layer": layer,
                "language": language,
                "num_token_vectors": int(counts[layer, i].item()),
            }
            for layer in range(counts.shape[0])
            for i, language in enumerate(languages)
        ]
    )
    counts_df.to_csv(model_dir / "aya_gmm_fit_counts.csv", index=False)
    pd.DataFrame(
        [{"language": lang, "num_samples_processed": sample_counts[lang], "num_samples_expected": expected_sample_counts[lang]}
         for lang in languages]
    ).to_csv(model_dir / "aya_gmm_sample_counts.csv", index=False)

    with (model_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": loaded.name,
                "training_corpus": "Aya",
                "coverage": "100% of available configured Aya samples for each language",
                "components": int(cfg["representation"]["gmm_components"]),
                "covariance_type": cfg["representation"]["gmm_covariance_type"],
                "languages": languages,
                "pca_path": str((pca_root / loaded.name).resolve()),
            },
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
    """Reuse a complete saved Aya-GMM fit; otherwise fit it once on all Aya."""
    model_dir = gmm_root / loaded.name
    paths = [model_dir / f"layer_{layer:03d}.pt" for layer in range(len(pcas))]
    metadata_path = model_dir / "metadata.json"
    if not force_refit and all(path.exists() for path in paths) and metadata_path.exists():
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            same_pca = Path(meta.get("pca_path", "")).resolve() == (pca_root / loaded.name).resolve()
            same_langs = meta.get("languages") == list(cfg["languages"])
            same_coverage = meta.get("coverage") == "100% of available configured Aya samples for each language"
            if same_pca and same_langs and same_coverage:
                print(f"[{loaded.name}] Reusing saved Aya GMMs from: {model_dir.resolve()}")
                return [load_gmm(path, loaded.device) for path in paths]
        except Exception as exc:
            print(f"[{loaded.name}] Saved GMM metadata could not be validated ({exc}); refitting.")
    return _fit_aya_gmms(cfg, loaded, pcas, gmm_root, pca_root, expected_sample_counts)


def _analyze_model(
    cfg: dict[str, Any],
    loaded: LoadedModel,
    pcas: list[GPUPCA],
    gmms: list[GPULanguageGMM],
    result_dir: Path,
):
    result_dir.mkdir(parents=True, exist_ok=True)
    gen_cfg = cfg["generation"]
    batch_size = int(cfg["runtime"].get("generation_batch_size", 1))
    token_writers: dict[str, PartitionWriter] = {}
    sentence_writers: dict[str, PartitionWriter] = {}
    output_writers: dict[str, PartitionWriter] = {}
    counts = {lang: 0 for lang in cfg["languages"]}
    pcols = [f"P_{x}" for x in cfg["languages"]]
    token_decode_cache: dict[int, str] = {}

    def decode_token(token_id: int) -> str:
        token_id = int(token_id)
        cached = token_decode_cache.get(token_id)
        if cached is not None:
            return cached
        text = loaded.tokenizer.decode([token_id], skip_special_tokens=False)
        token_decode_cache[token_id] = text
        return text

    try:
        print(f"[{loaded.name}] Aya pass 2/2: generation + posterior analysis")
        for language in cfg["languages"]:
            for chunk in tqdm(
                iter_aya_batches(cfg, language, batch_size),
                desc=f"{loaded.name} Aya/analysis/{language}",
            ):
                prompts = chunk[cfg["data"]["input_column"]].tolist()
                sample_ids = chunk[cfg["data"]["sample_id_column"]].tolist()
                token_ids, generated_texts, hidden_states, input_width = generate_batch(
                    loaded, prompts, gen_cfg, cfg["runtime"]
                )

                # Count every source Aya row regardless of whether generation
                # returned zero tokens for an individual sample.
                counts[language] += len(sample_ids)

                max_generated = max((len(ids) for ids in token_ids), default=0)
                if max_generated == 0:
                    del hidden_states
                    continue

                length_tensor = torch.tensor(
                    [len(ids) for ids in token_ids], device=loaded.device, dtype=torch.long
                )
                gen_positions = torch.arange(max_generated, device=loaded.device).unsqueeze(0)
                gen_mask = gen_positions < length_tensor.unsqueeze(1)

                output_rows = [
                    {
                        "model": loaded.name,
                        "sample_id": sid,
                        "input_language": language,
                        "prompt": prompt,
                        "generated_text": gen_text,
                    }
                    for sid, prompt, gen_text in zip(sample_ids, prompts, generated_texts)
                ]
                token_rows_by_layer: dict[int, list[dict[str, Any]]] = {i: [] for i in range(len(pcas))}
                sentence_rows: list[dict[str, Any]] = []

                # One PCA transform + one posterior call per layer/batch.
                for layer, (pca, gmm) in enumerate(zip(pcas, gmms)):
                    # hidden_states contains pre-emission states for generated tokens only.
                    layer_hidden = hidden_states[layer][:, :max_generated, :]
                    valid_hidden = layer_hidden[gen_mask]
                    Z = pca.transform(valid_hidden)
                    post = gmm.posterior(Z)
                    post_cpu = post.detach().cpu().tolist()

                    offset = 0
                    for sample_index, (sid, ids) in enumerate(zip(sample_ids, token_ids)):
                        sample_len = len(ids)
                        if sample_len == 0:
                            continue
                        sample_post = post[offset : offset + sample_len]
                        means = sample_post.mean(dim=0).detach().cpu().tolist()
                        sentence_rows.append(
                            {
                                "model": loaded.name,
                                "sample_id": sid,
                                "input_language": language,
                                "layer": layer,
                                **{c: float(v) for c, v in zip(pcols, means)},
                            }
                        )

                        rows = token_rows_by_layer[layer]
                        cpu_rows = post_cpu[offset : offset + sample_len]
                        for position, (token_id, posterior_values) in enumerate(zip(ids, cpu_rows)):
                            token_text = decode_token(int(token_id))
                            row = {
                                "model": loaded.name,
                                "sample_id": sid,
                                "input_language": language,
                                "token_position": position,
                                "sequence_position": position,
                                "token": token_text,
                                "token_id": int(token_id),
                                "layer": layer,
                            }
                            row.update({c: float(v) for c, v in zip(pcols, posterior_values)})
                            rows.append(row)
                        offset += sample_len

                outputs_path = result_dir / "generated_outputs" / f"{language}.parquet"
                output_writers.setdefault(language, PartitionWriter(outputs_path))
                output_writers[language].write(output_rows)

                token_path = result_dir / "token_posteriors" / f"{language}.parquet"
                token_writers.setdefault(language, PartitionWriter(token_path))
                for rows in token_rows_by_layer.values():
                    token_writers[language].write(rows)

                sentence_path = result_dir / "sentence_level_posteriors" / f"{language}.parquet"
                sentence_writers.setdefault(language, PartitionWriter(sentence_path))
                sentence_writers[language].write(sentence_rows)

                del hidden_states, length_tensor, gen_positions, gen_mask
    finally:
        for writers in (token_writers, sentence_writers, output_writers):
            for writer in writers.values():
                writer.close()

    pd.DataFrame(
        [{"language": lang, "num_samples_analyzed": count} for lang, count in counts.items()]
    ).to_csv(result_dir / "analysis_counts.csv", index=False)


def run(
    cfg: dict[str, Any],
    pca_path: str | Path | None = None,
    force_refit_pca: bool = False,
    force_refit_gmm: bool = False,
) -> None:
    output_root = Path(cfg["outputs"]["root_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifests").mkdir(exist_ok=True)

    validation = validate_aya_files(cfg)
    validation.to_csv(output_root / "manifests" / "aya_input_summary.csv", index=False)
    print("Aya input summary:")
    print(validation.to_string(index=False))

    default_pca_root = Path(cfg["runtime"].get("pca_data_path") or (output_root / "pca_data"))
    pca_root = Path(pca_path) if pca_path is not None else default_pca_root
    if not pca_root.is_absolute():
        pca_root = Path.cwd() / pca_root
    pca_root.mkdir(parents=True, exist_ok=True)
    print(f"PCA data root: {pca_root.resolve()}")

    gmm_root = output_root / cfg["outputs"]["gmm_dir"]

    for model_key, model_spec in [(k, v) for k, v in cfg["models"].items() if v.get("enabled", False)]:
        print("\n" + "=" * 90)
        print(f"MODEL: {model_key} | {model_spec['model_name']}")
        print("=" * 90)
        loaded = load_model(model_key, model_spec, cfg["runtime"])
        try:
            print(f"[{model_key}] GPU allocated after model load: {torch.cuda.memory_allocated(loaded.device) / (1024**3):.2f} GiB")
            pcas = _load_pcas_or_fit(cfg, loaded, pca_root, force_refit=force_refit_pca)
            expected_sample_counts = dict(zip(validation["language"], validation["num_samples"]))
            gmms = _load_gmms_or_fit(
                cfg, loaded, pcas, gmm_root, pca_root, expected_sample_counts, force_refit=force_refit_gmm
            )
            result_dir = output_root / cfg["outputs"]["result_dir"] / model_key
            _analyze_model(cfg, loaded, pcas, gmms, result_dir)
            del pcas, gmms
        finally:
            release_model(loaded)

    with (output_root / "run_config.json").open("w", encoding="utf-8") as f:
        effective_cfg = json.loads(json.dumps(cfg))
        effective_cfg.setdefault("runtime", {})["pca_data_path"] = str(pca_root.resolve())
        json.dump(effective_cfg, f, ensure_ascii=False, indent=2)

    print("\nExperiment completed.")
