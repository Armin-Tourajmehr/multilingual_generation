# Multilingual Representation Analysis

A reproducible pipeline for analyzing how multilingual causal language models represent language information across layers. The experiment uses Aya inputs, generated continuations, hidden-state extraction, layer-wise PCA, language-conditional Gaussian mixture models, and posterior language probabilities.

## Models

The main configuration contains four models:

- `ai-forever/mGPT`
- `bigscience/bloom-560m`
- `Qwen/Qwen3-8B`
- `google/gemma-3-4b-it`

All four are enabled by default.

**Models are run sequentially, not simultaneously.** The pipeline trains and analyzes one enabled model, releases its model/memory, and then moves to the next model. This is intentional: running all four large models concurrently would unnecessarily increase GPU/CPU memory pressure.

## Experiment languages

The final experiment set contains 18 languages:

```text
ar  bn  en  es  eu  fa  fr  hi  id  ml  mr  ne  pt  sw  ta  te  ur  vi
```

Persian (`fa`) is included explicitly as the experimental language requested for the study.

## Repository structure

```text
multilingual-representation-analysis/
├── configs/
│   └── config.yaml
├── data/
│   └── aya_language_datasets/
│       └── README.md
├── outputs/
│   └── README.md
├── scripts/
│   ├── prepare_aya.py
│   └── run_experiment.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── gmm.py
│   ├── modeling.py
│   ├── pipeline.py
│   └── utils.py
├── tests/
│   └── test_smoke.py
├── .gitignore
├── requirements.txt
└── README.md
```

## 1. Environment

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For a CUDA machine, use a PyTorch build compatible with the installed CUDA stack when necessary.

## 2. Prepare the Aya data

The experiment expects one prepared CSV per language under:

```text
data/aya_language_datasets/
```

The helper script downloads `CohereLabs/aya_dataset`, normalizes Aya language codes, keeps the final 18-language experiment set, shuffles each language deterministically with seed `42`, assigns `sample_id`, and writes one CSV per language.

Run:

```bash
python scripts/prepare_aya.py
```

This creates files such as:

```text
data/aya_language_datasets/ar.csv
data/aya_language_datasets/bn.csv
...
data/aya_language_datasets/vi.csv
data/aya_language_datasets/aya_language_dataset_summary.csv
```

The CSV files are intentionally ignored by Git because the dataset is generated locally/on the experiment machine rather than stored in the repository.

## 3. Full-data experimental split

The main experiment uses **all available samples** from every configured language.

It does **not** use a fixed number such as 10 samples.

For each language, the complete CSV is deterministically shuffled and split into:

```yaml
data:
  train_fraction: 0.80
  analysis_fraction: 0.20
  sampling: random
```

Therefore every sample belongs to exactly one partition:

```text
all Aya samples for a language
          │
          ├── 80% train
          │
          └── 20% analysis
```

The exact sample IDs used in each partition are saved to:

```text
outputs/manifests/train_samples.csv
outputs/manifests/analysis_samples.csv
outputs/manifests/split_summary.json
```

This prevents train/analysis leakage while still using the complete dataset.

## 4. Configure the experiment

Normal experiment changes belong in `configs/config.yaml`.

### Languages

The `languages` list determines the common experimental set and the posterior columns written to the outputs.

### Models

Enable or disable models without editing Python source:

```yaml
models:
  mgpt:
    enabled: true
  bloom:
    enabled: true
  qwen3_8b:
    enabled: true
  gemma3_4b:
    enabled: true
```

### Generation

```yaml
generation:
  max_new_tokens: 30
  do_sample: true
  temperature: 0.7
  top_p: 0.9
```

### Representation model

```yaml
representation:
  pca_dim: 50
  pca_variance: null
  pca_solver: incremental
  pca_batch_size: 2048
  gmm_components: 1
  gmm_covariance_type: diag
  gmm_reg_covar: 1.0e-6
  equal_language_priors: true
```

## 5. Run the complete experiment

From the repository root:

```bash
python scripts/run_experiment.py --config configs/config.yaml
```

The pipeline will:

1. Validate all configured Aya CSV files.
2. Build one deterministic 80/20 train/analysis split per language using every available sample.
3. Save the exact split manifests.
4. Run each enabled model **sequentially**.
5. Generate continuations from Aya `inputs`.
6. Re-run the generated sequences with hidden-state output enabled.
7. Keep only generated-token hidden states for representation analysis.
8. Fit one shared PCA per layer using training embeddings from all languages. The main run uses a two-pass streaming implementation so all training samples can be used without retaining all hidden states in RAM.
9. Fit one one-component diagonal Gaussian per language and layer from streaming sufficient statistics in the PCA space. With one mixture component and diagonal covariance, this is the diagonal-Gaussian case of the configured GMM.
10. Compute posterior probabilities over the configured languages for analysis tokens.
11. Save token-level, sample-level, and summary outputs.
12. Release the current model before starting the next enabled model.

The run is fail-fast: if a model or a required experiment stage fails, the command stops rather than silently producing an incomplete paper result.

## 6. Output structure

For each enabled model:

```text
outputs/
├── manifests/
│   ├── train_samples.csv
│   ├── analysis_samples.csv
│   └── split_summary.json
├── trained_language_models/
│   ├── mgpt/
│   │   ├── layer_0.pkl
│   │   └── ...
│   ├── bloom/
│   ├── qwen3_8b/
│   └── gemma3_4b/
└── analysis_results/
    ├── mgpt/
    │   ├── generated_outputs.csv
    │   ├── token_layer_language_probabilities.csv
    │   ├── mean_posterior_by_source_and_layer.csv
    │   ├── dominant_language_percentages.csv
    │   ├── sentence_level_probabilities.csv
    │   └── summary.json
    ├── bloom/
    ├── qwen3_8b/
    └── gemma3_4b/
```

If `runtime.save_hidden_states: true`, raw generated-token hidden states are also stored under each model's `analysis_results/<model>/raw_embeddings/` directory. This is optional because it can require substantial storage.

## 7. Hugging Face access

Some configured model repositories may require accepting terms or authenticating with Hugging Face. Make sure the account/machine used for the experiment can access every enabled model before starting the full run.

For a server or Kaggle notebook, model weights are downloaded to the machine's cache when the first model is loaded. The pipeline then proceeds to the next model sequentially.

## 8. Running on Kaggle

The recommended Kaggle workflow is simple:

### Clone the repository

```python
!git clone https://github.com/<USERNAME>/<REPOSITORY>.git
%cd <REPOSITORY>
```

### Install dependencies

```python
!pip install -q -r requirements.txt
```

### Prepare Aya

```python
!python scripts/prepare_aya.py --output-dir data/aya_language_datasets
```

### Verify the prepared files

```python
!ls data/aya_language_datasets
```

You should see the 18 language CSV files plus the summary CSV.

### Enable GPU

In the Kaggle notebook settings, select a GPU accelerator. Then verify:

```python
import torch
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

### Run the main experiment

```python
!python scripts/run_experiment.py --config configs/config.yaml
```

No separate Kaggle-specific config is required. The same `configs/config.yaml` is the source of truth for the main experiment.

## 9. Tests

Run the lightweight unit tests before a full model run:

```bash
python -m pytest -q
```

These tests verify configuration validation, full-data train/analysis coverage without overlap, and normalized posterior probabilities. They do not download or execute the four large language models.

## Reproducibility

- Global seed: `42`
- Language order is fixed in `configs/config.yaml`.
- Aya language files receive deterministic `sample_id` values during preparation.
- The train/analysis split is deterministic per language.
- The exact split manifests are written to `outputs/manifests/`.
- PCA and GMM random states use the project seed.
- Language priors are equal by default.
- PCA is fitted separately at each hidden-state layer on the combined training embeddings across configured languages.
- GMMs are fitted separately for each language at each layer.
- Analysis uses generated-token representations only.
- Qwen3 is configured with `enable_thinking: false`.

## Important computational note

The main experiment intentionally uses the complete prepared dataset. This is substantially more expensive than a smoke test, especially for Qwen3-8B and Gemma 3. The training stage therefore uses two streaming passes over the full training split: one for layer-wise IncrementalPCA and one for the language-conditional diagonal Gaussian statistics. Hidden states for the whole training corpus are not accumulated in RAM.

The repository separates the concepts of **scientific sample coverage** (all Aya samples are used across train + analysis) and **model execution order** (models are processed one at a time).

Before the final paper run, verify that the target machine has enough GPU/CPU RAM, storage, and execution time for the complete experiment.

## Paper workflow

For a final result, keep `configs/config.yaml` unchanged during a run and preserve:

- the exact Git commit used,
- the prepared Aya language summary,
- `outputs/manifests/`, and
- the model-specific result directories.

These artifacts make the final experiment auditable and reproducible.
