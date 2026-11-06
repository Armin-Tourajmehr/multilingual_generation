# Multilingual Representation Analysis — Kaggle Main Experiment

GPU-first implementation of the multilingual representation experiment across 20 configured languages with mGPT and BLOOM, plus optional Qwen and Llama runs.

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

Wikipedia is used **only** to fit the shared PCA representation. For every configured language, the GMM is trained on the configured Aya subset (up to `data.max_samples_per_language`, currently 2,000 valid rows). The same selected prompts are used for generation; there is no independent Aya train/test split.

The configured one-component diagonal GMM is implemented with streaming sufficient statistics on CUDA. This is mathematically equivalent to a one-component diagonal Gaussian and avoids a CPU-side scikit-learn fitting step. `representation.equal_language_priors: true` uses uniform priors; otherwise priors reflect selected Aya row counts.

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

### B200

Use a Blackwell-compatible PyTorch/CUDA environment and its matching NVIDIA driver. The generic `torch>=2.3` requirement alone does not establish Blackwell support; use the GPU provider's supported image or an appropriate current PyTorch CUDA build. See [PyTorch's Blackwell/CUDA guidance](https://pytorch.org/blog/pytorch-2-12-release-blog/).

From the repository root:

```bash
python scripts/prepare_aya.py --config configs/b200_config.yaml
python scripts/run_experiment.py --config configs/b200_config.yaml
```

The B200 preset uses BF16 model weights/autocast, larger batches and separate `outputs/b200/` caches/results. PCA/GMM accumulation remains FP32. It preserves the default model selections, model IDs, sample caps and token limits; enable Qwen/Llama explicitly if required. The initial batches are 128/64/32 for mGPT/BLOOM (Wikipedia/Aya/generation) and 64/32/16 for the optional 8B models. These are starting points to benchmark on the actual GPU. The loader now accepts per-model `attn_implementation`; this preset uses SDPA for supported models and eager attention for BLOOM, which does not support SDPA at the project's minimum Transformers version.

POS tagging is deferred in this preset to keep the CPU tagger outside the GPU job. Once the GPU job finishes, run the following on a CPU machine that can access the same outputs and Stanza resources:

```bash
python scripts/annotate_outputs.py --config configs/b200_config.yaml
```

This annotates existing token Parquet files atomically and creates the usual POS summaries without loading a language model. Posterior columns and row order are preserved. Until this command completes, `pos_status` is `deferred`; POS summaries are intentionally absent. Set `lexical_analysis.deferred: false` for inline tagging, or `lexical_analysis.enabled: false` to omit POS analysis. `save_pos_posterior_summary: false` suppresses summaries in either mode.

`runtime.posterior_batch_size` controls GPU posterior chunks (B200 preset: 4,096). Output metadata is shared between layers, posterior results cross to CPU once per generation batch, and Parquet is built directly from columns. `gc_collect_every_batches: 0` disables repeated full Python garbage collection inside the generation loop; model cleanup still runs. Positive values collect at the configured batch interval.

With `runtime.log_performance: true`, each model writes `performance.json` under its analysis results, containing phase elapsed times and peak GPU allocation. Benchmark a short regular run with representative lengths and batches, then increase one phase's batch at a time until throughput stops improving or memory becomes limiting. Smoke mode forces batches to 1 and checks correctness/access rather than throughput. GPU speedups have not been established by the offline tests.

The corrected mask-relative positions make mGPT representations independent of left padding. PCA/GMM fingerprints include the position policy, attention setting, input truncation and selected Aya content, so old fits rebuild automatically. Avoid mixing these corrected results with earlier cached fits.

Offline regression tests use small randomly initialized models and do not download checkpoints:

```bash
python -m pip install pytest
python -m pytest -q
```

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

`generated_outputs` records the prompt and continuation IDs used for replay, alongside the decoded text. The continuation includes the sampled stop token when present.

`token_posteriors` scores the contextual state after consuming each non-special generated token. `sequence_position` indexes the original continuation IDs. Character spans and POS tags are aligned separately. Tokens with ambiguous spans or whitespace still have scores and carry a `no_offset` or `unaligned` POS status.

`sentence_level_posteriors` averages the scores of all retained tokens per Aya sample and layer, including tokens without a POS alignment.

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


## Runtime optimizations

The main experiment is designed for a single NVIDIA T4-class GPU. PCA and Aya GMM statistics are computed on CUDA. Generation uses the model's cache, then a causal forward pass replays the original prompt and continuation IDs to collect contextual hidden states. Saved PCA and Aya-GMM fits are reused automatically on subsequent runs. Use `--force-refit-pca` or `--force-refit-gmm` for a deliberate refit.

The default Kaggle configuration uses FP16, generation batch size 1 for peak-memory safety, representation batch size 8, and a 512-token maximum input length. This processes every configured Aya sample row; only inputs longer than the configured token limit are truncated.

Example:

```bash
python scripts/run_experiment.py --config configs/config.yaml
```

To deliberately rebuild saved artifacts:

```bash
python scripts/run_experiment.py --config configs/config.yaml --force-refit-pca --force-refit-gmm
```
