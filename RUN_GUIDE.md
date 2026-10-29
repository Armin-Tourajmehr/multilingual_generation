# Run guide

## 1. Install dependencies

Use a Python environment with a CUDA-compatible PyTorch build. In Kaggle, select **GPU** before running the notebook. Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## 2. Prepare Aya CSVs

Run from the repository root (the scripts also support being called from elsewhere):

```bash
python scripts/prepare_aya.py --config configs/config.yaml
```

The script writes one CSV per configured language under `data/aya_language_datasets/`. It now includes Chinese (`zh`) and Japanese (`ja`). For future languages, standard ISO 639-3 codes are mapped to ISO 639-1 automatically when `pycountry` knows the mapping; explicit aliases remain for Aya dialect codes. If a configured language has no rows in the selected Aya dataset, preparation stops and names the missing language instead of creating an empty CSV.

## 3. Run the full experiment

```bash
python scripts/run_experiment.py --config configs/config.yaml
```

All four models (`mgpt`, `bloom`, `qwen`, and `llama3_1_8b`) are enabled in the current configuration and run sequentially in YAML order. Qwen3-8B and Llama 3.1 8B Instruct require substantially more GPU memory and may fail to load on a 16 GiB GPU. Llama access may require accepting Meta's license and authenticating with Hugging Face on the target server. For the full experiment, a model error stops the run; disable models that do not fit by setting `enabled: false`.

Large-model settings have conservative per-model batch-size overrides. They do not guarantee that an 8B model will fit on every GPU; hardware capacity still matters. The PCA method retains GPU covariance statistics for all layers, so large hidden sizes add a noticeable GPU-memory requirement.

## 4. Run the smoke test

```bash
python scripts/run_experiment.py --config configs/config.yaml --smoke-test
```

This attempts all four models sequentially. It uses up to two Aya rows per language (40 rows per model across the current 20 languages), two Wikipedia documents per language, 16 PCA components, short input contexts and four generated tokens. It writes into `outputs/smoke_test/` and does not overwrite the full-run PCA/GMM cache under `outputs/pca_data/` and `outputs/aya_gmms/`. If an individual model fails (including out-of-memory or gated model access), smoke mode logs the error to `outputs/smoke_test/manifests/smoke_model_errors.json` and continues to the next model. Successful models' outputs remain available.

The smoke test avoids automatic Stanza downloads so it does not unexpectedly download 20 language packages. If the Stanza resources are not already installed, token rows clearly report `POS_UNAVAILABLE`; the rest of the end-to-end run can still be checked. For the full run, `lexical_analysis.download_missing_models: true` downloads missing tokenizer/POS resources on first use. On an offline server, download the Stanza resources in advance or set that option to false.

## 5. Important outputs

Under `outputs/analysis_results/<model>/`:

- `generated_outputs/<language>.parquet`: generated text and prompt.
- `token_posteriors/<language>.parquet`: posterior per output token and layer, character offsets, Stanza `upos`/`xpos`/features, and POS alignment status.
- `sentence_level_posteriors/<language>.parquet`: average output-token posterior per sample and layer.
- `lexical_analysis/pos_posterior_summary/<language>.csv`: mean posterior for each candidate language grouped by layer and POS tag.
- `manifold/spectral_embedding_coordinates.csv`, per-layer PNGs and `spectral_embedding_all_layers.pdf`: Spectral Embedding of the per-language GMM centroids in PCA space.
- `analysis_counts.csv`: processed Aya row counts.

The manifold contains one point per language per layer (GMM centroids); it is not an individual-token scatter plot. The token-level file is the source for within-language and POS-level analyses.

## 6. Rerunning after config changes

PCA and GMM caches carry configuration fingerprints. If languages, reference-data settings, model identity/revision, sampling settings or PCA configuration change, stale PCA/GMM caches are rebuilt rather than reused silently. Use `--force-refit-pca` or `--force-refit-gmm` to force an otherwise valid cache to be recomputed.
