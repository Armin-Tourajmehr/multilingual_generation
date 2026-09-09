# Multilingual Representation Analysis — Kaggle Main Experiment

GPU-first implementation of the multilingual representation experiment across 18 languages with mGPT and BLOOM.

## Experimental flow

```text
Wikipedia reference data
    ↓
model hidden states on GPU
    ↓
GPU covariance accumulation
    ↓
GPU eigendecomposition
    ↓
PCA (700 components)
    ↓
save/reuse PCA data
    ↓
complete Aya file for each language
    ↓
model hidden states on GPU
    ↓
PCA transform on GPU
    ↓
one diagonal, one-component GMM per language
    ↓
save/reuse Aya-fitted GMMs
    ↓
complete Aya analysis set
    ↓
generation (GPU, batch size 1)
    ↓
full-sequence hidden states on GPU
    ↓
one PCA transform + posterior calculation per layer/batch
    ↓
token + sentence-level outputs
```

Wikipedia is used **only** to fit the shared PCA representation. For every configured language, the GMM is trained on **100% of the available Aya sample rows** for that language. There is no Aya train/test split and no Aya sampling cap.

The configured one-component diagonal GMM is implemented with streaming sufficient statistics on CUDA. This is mathematically equivalent to a one-component diagonal Gaussian and avoids a CPU-side scikit-learn fitting step.

## GPU and memory design

The repository is designed for a 14–16 GiB single-GPU environment such as a Kaggle T4.

The main model weights are loaded explicitly in FP16. mGPT/BLOOM run sequentially on `cuda:0`; no CPU offload or multi-GPU device map is used.

The memory-sensitive settings are deliberate:

```yaml
runtime:
  wikipedia_batch_size: 8
  evaluation_batch_size: 8
  generation_batch_size: 1
  max_input_tokens: 512
```

`evaluation_batch_size=8` is used for the representation/GMM pass. Generation uses a separate batch size of 1 because autoregressive generation has a much higher peak memory requirement.

`max_input_tokens=512` does **not** remove Aya samples. Every Aya row is still processed. It only truncates unusually long input prompts before the model forward/generation so a single T4 remains safe.

The CUDA allocator is configured with expandable segments to reduce fragmentation during the long run.

### What stays on GPU

All large tensor operations remain on CUDA:

- transformer forward passes
- hidden-state tensors
- PCA covariance accumulation
- PCA eigendecomposition
- PCA transforms
- GMM sufficient-statistics accumulation
- GMM posterior calculations

Large hidden-state tensors are never converted to NumPy or copied to CPU.

### What necessarily uses CPU

The operating system and Python runtime still handle ordinary I/O and text work:

- CSV/Parquet reading and writing
- tokenization and text decoding
- output metadata
- small token-ID/posterior records required for serialization

These are not used for the numerical PCA/GMM/model computation.

## Avoiding redundant computation

The PCA stage performs one streaming Wikipedia pass per model. It accumulates the exact global covariance statistics directly on GPU and then computes the PCA once.

The Aya stage has two passes by design:

1. **Pass 1:** process the complete Aya input set and fit the final per-language GMMs.
2. **Pass 2:** process the complete Aya input set again for generation and posterior analysis.

During analysis, each layer/batch performs one PCA transform and one posterior computation. The resulting posterior matrix is reused to produce both token-level and sentence-level outputs.

## Models and languages

Enabled models:

- `ai-forever/mGPT`
- `bigscience/bloom-560m`

Configured languages:

```text
ar bn en es eu fa fr hi id ml mr ne pt sw ta te ur vi
```

Models run sequentially so GPU memory is released between models.

## Repository layout

```text
configs/config.yaml
scripts/
    prepare_aya.py
    run_experiment.py
src/
    config.py
    data.py
    gmm.py
    modeling.py
    pca_gpu.py
    pipeline.py
tests/
    test_smoke.py
```

Aya CSV files are kept outside the repository's source tree history and are expected under:

```text
data/aya_language_datasets/
```

## Installation

### Local CUDA environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA-capable PyTorch installation is required.

### Kaggle

Create a Kaggle notebook with **GPU** and **Internet** enabled. Then:

```python
!git clone https://github.com/<USERNAME>/<REPOSITORY>.git
%cd <REPOSITORY>
!pip install -q -r requirements.txt
```

Prepare the complete Aya language files:

```python
!python scripts/prepare_aya.py
```

Run the experiment:

```python
!python scripts/run_experiment.py --config configs/config.yaml
```

The program performs a CUDA preflight before loading any model and exits immediately when a GPU is unavailable.

## Reusing PCA data

The default PCA root is:

```text
outputs/pca_data/
```

A completed model directory contains:

```text
outputs/pca_data/<model>/layer_000.pt
outputs/pca_data/<model>/layer_001.pt
...
outputs/pca_data/<model>/metadata.json
```

Provide another location when PCA has already been generated:

```bash
python scripts/run_experiment.py \
  --config configs/config.yaml \
  --pca-path /kaggle/working/my_pca_data
```

Use `--force-refit-pca` only when an intentional PCA rebuild is required.

## Aya GMM outputs

GMMs are stored separately:

```text
outputs/aya_gmms/<model>/layer_000.pt
outputs/aya_gmms/<model>/layer_001.pt
...
outputs/aya_gmms/<model>/aya_gmm_fit_counts.csv
outputs/aya_gmms/<model>/aya_gmm_sample_counts.csv
outputs/aya_gmms/<model>/metadata.json
```

`aya_gmm_sample_counts.csv` is a coverage audit. The run fails when the number of processed Aya sample rows does not exactly match the validated input count for any language.

## Analysis outputs

```text
outputs/analysis_results/<model>/
    generated_outputs/<language>.parquet
    token_posteriors/<language>.parquet
    sentence_level_posteriors/<language>.parquet
    analysis_counts.csv
```

`token_posteriors` contains generated-token × layer posterior records.

`sentence_level_posteriors` contains one row per Aya sample and layer, with posterior probabilities averaged across generated tokens for that sample.

## Other run metadata

```text
outputs/manifests/aya_input_summary.csv
outputs/run_config.json
```

The effective configuration, including the resolved PCA path, is written to `run_config.json`.

## Validation

Run the lightweight tests before a full experiment:

```bash
pytest -q
```

These tests validate the main configuration, the 18-language setup, GPU-only numerical interfaces, and the Kaggle-safe generation settings.
