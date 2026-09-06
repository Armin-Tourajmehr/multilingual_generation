from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from .data import load_experiment_data, validate_language_files, validate_train_analysis_split
from .gmm import fit_layer_models_streaming, load_layer_models, posterior, save_layer_models
from .modeling import extract_generated_embeddings, generate_with_states, load_model
from .utils import save_json, set_seed


def _model_paths(cfg: dict[str, Any], model_key: str):
    root = Path(cfg["outputs"]["root_dir"])
    return (
        root / cfg["outputs"]["model_dir"],
        root / cfg["outputs"]["result_dir"] / model_key,
    )


def _training_embedding_iterator_factory(cfg, loaded, model_key, samples):
    input_column = cfg["data"]["input_column"]
    model_spec = cfg["models"][model_key]
    runtime_cfg = cfg["runtime"]
    generation_cfg = cfg["generation"]

    def iterator():
        for row in samples.itertuples(index=False):
            prompt = getattr(row, input_column)
            lang = getattr(row, "source_language")
            _, prompt_length, hidden_states = generate_with_states(
                loaded, prompt, model_spec, runtime_cfg, generation_cfg
            )
            embeddings = extract_generated_embeddings(hidden_states, prompt_length)
            yield lang, embeddings
            del hidden_states
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return iterator


def train_model(cfg: dict[str, Any], model_key: str, samples: pd.DataFrame):
    model_spec = cfg["models"][model_key]
    loaded = load_model(model_key, model_spec, cfg["runtime"])
    iterator_factory = _training_embedding_iterator_factory(cfg, loaded, model_key, samples)
    layer_models = fit_layer_models_streaming(
        iterator_factory,
        cfg["languages"],
        cfg["representation"],
        cfg["project"]["seed"],
    )
    model_dir, _ = _model_paths(cfg, model_key)
    save_layer_models(
        layer_models,
        model_dir,
        model_key,
        overwrite=cfg["outputs"].get("overwrite", False),
    )
    return loaded, layer_models


def analyze_model(cfg: dict[str, Any], model_key: str, samples: pd.DataFrame, loaded=None):
    model_spec = cfg["models"][model_key]
    if loaded is None:
        loaded = load_model(model_key, model_spec, cfg["runtime"])
    layer_models = load_layer_models(_model_paths(cfg, model_key)[0], model_key)

    rows = []
    generated_rows = []
    generation_cfg = cfg["generation"]
    runtime_cfg = cfg["runtime"]

    for row in tqdm(samples.itertuples(index=False), total=len(samples), desc=f"Analyze {model_key}"):
        prompt = getattr(row, cfg["data"]["input_column"])
        lang = getattr(row, "source_language")
        sample_id = getattr(row, cfg["data"].get("sample_id_column", "sample_id"), None)

        generated_ids, prompt_length, hidden_states = generate_with_states(
            loaded, prompt, model_spec, runtime_cfg, generation_cfg
        )
        generated_token_ids = generated_ids[0, prompt_length:].detach().cpu().tolist()
        generated_tokens = loaded.tokenizer.convert_ids_to_tokens(generated_token_ids)
        generated_text = loaded.tokenizer.decode(generated_token_ids, skip_special_tokens=True)

        generated_rows.append({
            "model": model_key,
            "model_name": model_spec["model_name"],
            "sample_id": sample_id,
            "input_language": lang,
            "prompt": prompt,
            "generated_text": generated_text,
        })

        embeddings = extract_generated_embeddings(hidden_states, prompt_length)

        if cfg.get("runtime", {}).get("save_hidden_states", False):
            raw_dir = _model_paths(cfg, model_key)[1] / "raw_embeddings"
            raw_dir.mkdir(parents=True, exist_ok=True)
            safe_id = f"{lang}__{sample_id}"
            import numpy as np
            np.savez_compressed(
                raw_dir / f"{safe_id}.npz",
                **{f"layer_{layer_idx}": array for layer_idx, array in embeddings.items()},
            )

        for token_pos, (token_id, token) in enumerate(zip(generated_token_ids, generated_tokens)):
            sequence_position = prompt_length + token_pos
            for layer_idx, array in embeddings.items():
                if layer_idx not in layer_models:
                    continue
                probs = posterior(array[token_pos], layer_models[layer_idx])
                row_out = {
                    "model": model_key,
                    "model_name": model_spec["model_name"],
                    "sample_id": sample_id,
                    "input_language": lang,
                    "token_position": token_pos,
                    "sequence_position": sequence_position,
                    "token": token,
                    "token_id": token_id,
                    "layer": layer_idx,
                }
                for probability_language in cfg["languages"]:
                    row_out[f"P_{probability_language}"] = probs[probability_language]
                rows.append(row_out)

        del hidden_states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _, result_dir = _model_paths(cfg, model_key)
    result_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(result_dir / "token_layer_language_probabilities.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(generated_rows).to_csv(result_dir / "generated_outputs.csv", index=False, encoding="utf-8-sig")




def create_summary(cfg: dict[str, Any], model_key: str, rows: list[dict[str, Any]] | None = None):
    _, result_dir = _model_paths(cfg, model_key)
    token_path = result_dir / "token_layer_language_probabilities.csv"
    if not token_path.exists():
        return

    df = pd.read_csv(token_path)
    languages = cfg["languages"]
    pcols = [f"P_{lang}" for lang in languages]

    df["dominant_language"] = df[pcols].idxmax(axis=1).str.replace("P_", "", regex=False)
    df["max_probability"] = df[pcols].max(axis=1)

    mean_post = (
        df.groupby(["input_language", "layer"], as_index=False)[pcols]
        .mean()
    )
    mean_post.to_csv(result_dir / "mean_posterior_by_source_and_layer.csv", index=False)

    counts = (
        df.groupby(["input_language", "layer", "dominant_language"])
        .size()
        .rename("count")
        .reset_index()
    )
    counts["percentage"] = counts["count"] / counts.groupby(["input_language", "layer"])["count"].transform("sum") * 100
    counts.to_csv(result_dir / "dominant_language_percentages.csv", index=False)

    # Per-sample (sentence/input) posterior: average token posteriors
    # within each generated sample and layer.
    sentence_like = (
        df.groupby(["input_language", "sample_id", "layer"], as_index=False)[pcols]
        .mean()
    )
    sentence_like.to_csv(result_dir / "sentence_level_probabilities.csv", index=False)

    summary = {
        "model": model_key,
        "model_name": cfg["models"][model_key]["model_name"],
        "languages": languages,
        "num_languages": len(languages),
        "num_rows": int(len(df)),
        "layers": sorted(df["layer"].unique().tolist()),
    }
    save_json(summary, result_dir / "summary.json")


def run(cfg: dict[str, Any]) -> None:
    seed = cfg["project"]["seed"]
    set_seed(seed)

    check = validate_language_files(cfg)
    print("\nAya language files:")
    print(check.to_string(index=False))

    train_samples = load_experiment_data(cfg, "train", seed)
    analysis_samples = load_experiment_data(cfg, "analysis", seed)

    # Shuffle the complete partitions globally before model inference so the
    # streaming PCA sees mixed-language batches rather than language blocks.
    train_samples = train_samples.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    analysis_samples = analysis_samples.sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)

    sample_id_col = cfg["data"].get("sample_id_column", "sample_id")
    validate_train_analysis_split(train_samples, analysis_samples, sample_id_col)

    # Persist the exact sampled prompts so another machine can verify the
    # experimental split used by every enabled model.
    manifest_dir = Path(cfg["outputs"]["root_dir"]) / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    train_samples.to_csv(manifest_dir / "train_samples.csv", index=False, encoding="utf-8-sig")
    analysis_samples.to_csv(manifest_dir / "analysis_samples.csv", index=False, encoding="utf-8-sig")
    save_json({
        "seed": seed,
        "languages": cfg["languages"],
        "train_fraction": cfg["data"]["train_fraction"],
        "analysis_fraction": cfg["data"]["analysis_fraction"],
        "train_samples_total": int(len(train_samples)),
        "analysis_samples_total": int(len(analysis_samples)),
        "total_samples": int(len(train_samples) + len(analysis_samples)),
        "models": [name for name, spec in cfg["models"].items() if spec.get("enabled", False)],
    }, manifest_dir / "split_summary.json")

    for model_key, spec in cfg["models"].items():
        if not spec.get("enabled", False):
            continue
        print(f"\n{'=' * 80}\nMODEL: {model_key} | {spec['model_name']}\n{'=' * 80}")

        loaded = None
        if cfg.get("stages", {}).get("train_language_models", True):
            loaded, _ = train_model(cfg, model_key, train_samples)
        if cfg.get("stages", {}).get("analyze", True):
            analyze_model(cfg, model_key, analysis_samples, loaded=loaded)
        if cfg.get("stages", {}).get("create_summary", True):
            create_summary(cfg, model_key)

        del loaded
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
