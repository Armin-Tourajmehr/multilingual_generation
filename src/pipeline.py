
from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from .data import iter_aya_batches, iter_wikipedia_documents, validate_aya_files
from .gmm import DiagonalGaussian, LayerModel, make_incremental_pca, save_layer_model
from .modeling import LoadedModel, forward_text_batch, generate_batch, load_model, release_model


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


def _language_batches(cfg, loaded: LoadedModel, language: str, batch_size: int) -> Iterator[tuple[list[str], list[dict[str, Any]]]]:
    buffer = []
    meta = []
    for row in iter_wikipedia_documents(cfg, language, seed_offset=sum((i + 1) * ord(c) for i, c in enumerate(language))):
        buffer.append(row["text"])
        meta.append(row)
        if len(buffer) == batch_size:
            yield buffer, meta
            buffer, meta = [], []
    if buffer:
        yield buffer, meta


def _wiki_layer_batches(cfg, loaded: LoadedModel, language: str, layer: int, batch_size: int):
    for texts, _ in _language_batches(cfg, loaded, language, batch_size):
        hidden, mask = forward_text_batch(loaded, texts, int(cfg["wikipedia"]["max_tokens_per_document"]))
        X = hidden[layer]
        yield X[mask]



def _wiki_batches(cfg, loaded: LoadedModel, language: str):
    batch_size = int(cfg["runtime"]["wikipedia_batch_size"])
    texts, meta = [], []
    for row in iter_wikipedia_documents(
        cfg,
        language,
        seed_offset=sum((i + 1) * ord(c) for i, c in enumerate(language)),
    ):
        texts.append(row["text"])
        meta.append(row)
        if len(texts) == batch_size:
            yield texts, meta
            texts, meta = [], []
    if texts:
        yield texts, meta


def _collect_wikipedia_reference(cfg: dict[str, Any], loaded: LoadedModel, output_root: Path):
    """Fit all layer PCAs and diagonal language Gaussians using two passes over Wikipedia.

    Pass 1: model forward on Wikipedia -> IncrementalPCA.partial_fit for every layer.
    Pass 2: model forward on the same deterministic Wikipedia stream -> PCA transform
             -> streaming diagonal-Gaussian sufficient statistics for every language/layer.
    """
    languages = cfg["languages"]
    rep = cfg["representation"]
    pca_dim = int(rep["pca_dim"])
    pca_batch_size = int(rep["pca_batch_size"])
    wiki_max_tokens = int(cfg["wikipedia"]["max_tokens_per_document"])

    # Determine the number of hidden-state tensors (embedding output is index 0).
    probe_hidden, _ = forward_text_batch(loaded, ["representation probe"], wiki_max_tokens)
    n_layers = len(probe_hidden)
    del probe_hidden
    gc.collect()

    pcas = [make_incremental_pca(pca_dim, pca_batch_size) for _ in range(n_layers)]
    pending: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
    manifest_rows = []

    print(f"[{loaded.name}] Wikipedia pass 1/2: fitting {n_layers} PCAs")
    for language in languages:
        for texts, meta in tqdm(_wiki_batches(cfg, loaded, language), desc=f"{loaded.name} Wikipedia/PCA/{language}"):
            manifest_rows.extend({
                "language": language,
                "sample_index": row["sample_index"],
                "dataset_id": row["dataset_id"],
                "title": row["title"],
            } for row in meta)
            hidden, mask = forward_text_batch(loaded, texts, wiki_max_tokens)
            for layer in range(n_layers):
                X = hidden[layer][mask]
                if len(X) == 0:
                    continue
                if pending[layer]:
                    X = np.concatenate(pending[layer] + [X], axis=0)
                    pending[layer] = []
                # Feed a sufficiently large chunk. Keep a smaller remainder for the next batch.
                if len(X) >= pca_dim:
                    feed_n = (len(X) // pca_dim) * pca_dim
                    # Cap partial-fit chunks at a modest size while preserving >= pca_dim.
                    feed_n = min(feed_n, pca_batch_size)
                    pcas[layer].partial_fit(X[:feed_n].astype(np.float32, copy=False))
                    if feed_n < len(X):
                        pending[layer].append(X[feed_n:])
                else:
                    pending[layer].append(X)
            del hidden, mask
            gc.collect()

    # Flush any remaining data, provided it is large enough for PCA.
    for layer in range(n_layers):
        if pending[layer]:
            X = np.concatenate(pending[layer], axis=0)
            if len(X) >= pca_dim:
                pcas[layer].partial_fit(X[:pca_batch_size].astype(np.float32, copy=False))

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_root / "manifests" / "wikipedia_fit_samples.parquet"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(manifest_path, index=False)
    manifest.groupby("language").size().rename("num_documents").to_csv(
        output_root / "manifests" / "wikipedia_language_counts.csv"
    )

    # Streaming sufficient statistics for one diagonal Gaussian per language/layer.
    n = {(layer, lang): 0 for layer in range(n_layers) for lang in languages}
    total = {(layer, lang): np.zeros(pca_dim, dtype=np.float64) for layer in range(n_layers) for lang in languages}
    total_sq = {(layer, lang): np.zeros(pca_dim, dtype=np.float64) for layer in range(n_layers) for lang in languages}

    print(f"[{loaded.name}] Wikipedia pass 2/2: fitting diagonal Gaussians")
    for language in languages:
        for texts, _meta in tqdm(_wiki_batches(cfg, loaded, language), desc=f"{loaded.name} Wikipedia/GMM/{language}"):
            hidden, mask = forward_text_batch(loaded, texts, wiki_max_tokens)
            for layer in range(n_layers):
                X = hidden[layer][mask]
                if len(X) == 0:
                    continue
                Z = pcas[layer].transform(X.astype(np.float32, copy=False)).astype(np.float64, copy=False)
                key = (layer, language)
                n[key] += len(Z)
                total[key] += Z.sum(axis=0)
                total_sq[key] += np.square(Z).sum(axis=0)
            del hidden, mask
            gc.collect()

    model_dir = output_root / cfg["outputs"]["model_dir"] / loaded.name
    model_dir.mkdir(parents=True, exist_ok=True)
    for layer in range(n_layers):
        gaussians = {}
        for language in languages:
            key = (layer, language)
            if n[key] == 0:
                raise RuntimeError(f"No Wikipedia representations available for {language} at layer {layer}.")
            mean = total[key] / n[key]
            variance = np.maximum(total_sq[key] / n[key] - np.square(mean), 0.0) + float(rep["gmm_reg_covar"])
            gaussians[language] = DiagonalGaussian(mean=mean, variance=variance)
        save_layer_model(
            model_dir / f"layer_{layer:03d}.pkl",
            LayerModel(
                layer=layer,
                pca=pcas[layer],
                gaussians=gaussians,
                languages=list(languages),
                equal_priors=bool(rep["equal_language_priors"]),
            ),
        )

    gaussian_counts = pd.DataFrame([
        {"layer": layer, "language": lang, "num_token_vectors": n[(layer, lang)]}
        for layer in range(n_layers) for lang in languages
    ])
    gaussian_counts.to_csv(model_dir / "wikipedia_gaussian_fit_counts.csv", index=False)
    return manifest

def _load_layer_models(model_dir: Path) -> list[LayerModel]:
    import joblib
    paths = sorted(model_dir.glob("layer_*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No layer models found under {model_dir}")
    return [joblib.load(p) for p in paths]


def _analyze_model(cfg: dict[str, Any], loaded: LoadedModel, layer_models: list[LayerModel], result_dir: Path):
    result_dir.mkdir(parents=True, exist_ok=True)
    gen_cfg = cfg["generation"]
    batch_size = int(cfg["runtime"]["evaluation_batch_size"])
    token_writers: dict[str, PartitionWriter] = {}
    sentence_writers: dict[str, PartitionWriter] = {}
    output_writers: dict[str, PartitionWriter] = {}
    counts = {lang: 0 for lang in cfg["languages"]}

    try:
        for language in cfg["languages"]:
            for chunk in tqdm(iter_aya_batches(cfg, language, batch_size), desc=f"{loaded.name} Aya/{language}"):
                prompts = chunk[cfg["data"]["input_column"]].tolist()
                sample_ids = chunk[cfg["data"]["sample_id_column"]].tolist()
                token_ids, generated_texts, hidden_by_sample = generate_batch(loaded, prompts, gen_cfg, cfg["runtime"])

                output_rows = []
                token_rows = []
                sentence_rows = []
                pcols = [f"P_{x}" for x in cfg["languages"]]

                for sid, prompt, gen_text, ids, sample_hidden in zip(sample_ids, prompts, generated_texts, token_ids, hidden_by_sample):
                    output_rows.append({
                        "model": loaded.name,
                        "sample_id": sid,
                        "input_language": language,
                        "prompt": prompt,
                        "generated_text": gen_text,
                    })

                    for position, token_id in enumerate(ids):
                        token_text = loaded.tokenizer.decode([token_id], skip_special_tokens=False)
                        for lm in layer_models:
                            hs = sample_hidden[lm.layer]
                            if position >= len(hs):
                                continue
                            z = lm.pca.transform(hs[position:position + 1]).astype(np.float64)
                            post = lm.posterior(z)[0]
                            row = {
                                "model": loaded.name,
                                "sample_id": sid,
                                "input_language": language,
                                "token_position": position,
                                "sequence_position": position,
                                "token": token_text,
                                "token_id": int(token_id),
                                "layer": lm.layer,
                            }
                            row.update({c: float(v) for c, v in zip(pcols, post)})
                            token_rows.append(row)
                    # Sample-level/sentence-level posterior = mean over generated tokens for each layer.
                    for lm in layer_models:
                        hs = sample_hidden[lm.layer]
                        if len(hs) == 0:
                            continue
                        Z = lm.pca.transform(hs.astype(np.float32, copy=False))
                        post = lm.posterior(Z)
                        means = post.mean(axis=0)
                        row = {
                            "model": loaded.name,
                            "sample_id": sid,
                            "input_language": language,
                            "layer": lm.layer,
                        }
                        row.update({c: float(v) for c, v in zip(pcols, means)})
                        sentence_rows.append(row)
                    counts[language] += 1

                outputs_path = result_dir / "generated_outputs" / f"{language}.parquet"
                output_writers.setdefault(language, PartitionWriter(outputs_path))
                output_writers[language].write(output_rows)

                token_path = result_dir / "token_posteriors" / f"{language}.parquet"
                token_writers.setdefault(language, PartitionWriter(token_path))
                token_writers[language].write(token_rows)

                sentence_path = result_dir / "sentence_level_posteriors" / f"{language}.parquet"
                sentence_writers.setdefault(language, PartitionWriter(sentence_path))
                sentence_writers[language].write(sentence_rows)
    finally:
        for writers in (token_writers, sentence_writers, output_writers):
            for w in writers.values():
                w.close()

    pd.DataFrame([{"language": k, "num_samples_analyzed": v} for k, v in counts.items()]).to_csv(result_dir / "analysis_counts.csv", index=False)


def run(cfg: dict[str, Any]) -> None:
    output_root = Path(cfg["outputs"]["root_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifests").mkdir(exist_ok=True)

    validation = validate_aya_files(cfg)
    validation.to_csv(output_root / "manifests" / "aya_input_summary.csv", index=False)
    print("Aya input summary:")
    print(validation.to_string(index=False))

    for model_key, model_spec in [(k,v) for k,v in cfg["models"].items() if v.get("enabled", False)]:
        print("\n" + "=" * 90)
        print(f"MODEL: {model_key} | {model_spec['model_name']}")
        print("=" * 90)
        loaded = load_model(model_key, model_spec, cfg["runtime"])
        model_dir = output_root / cfg["outputs"]["model_dir"] / model_key
        result_dir = output_root / cfg["outputs"]["result_dir"] / model_key
        model_dir.mkdir(parents=True, exist_ok=True)
        wiki_manifest = _collect_wikipedia_reference(cfg, loaded, output_root)
        print(f"[{model_key}] Wikipedia fit documents: {len(wiki_manifest):,}")
        layers = _load_layer_models(model_dir)
        _analyze_model(cfg, loaded, layers, result_dir)
        del layers
        release_model(loaded)
        gc.collect()

    with (output_root / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print("\nExperiment completed.")
