# Multilingual Representation Analysis — Kaggle Main Experiment

This repository is the GPU-only main experiment for the multilingual representation analysis.

## Research design

```text
Wikipedia reference corpus (18 languages × up to 3,000 documents/language)
        ↓
model forward pass (GPU)
        ↓
all token hidden states (GPU)
        ↓
GPU covariance accumulation
        ↓
GPU eigendecomposition → PCA (700 components)
        ↓
save PCA data to the configured PCA path
        │
        ├────────────────────────────────────────────┐
        ↓                                            │
Aya, entire file for each language                   │
        ↓                                            │
model forward pass (GPU)                             │
        ↓                                            │
PCA.transform (GPU)                                  │
        ↓                                            │
fit one diagonal / one-component GMM per language   │
        ↓                                            │
save Aya-fitted GMMs                                 │
        │                                            │
        └────────────────────────────────────────────┘
                         ↓
                Aya generation + analysis
                         ↓
             generated hidden states (GPU)
                         ↓
                    PCA (GPU)
                         ↓
                Aya-fitted GMM posterior (GPU)
                         ↓
              token + sentence outputs
```

### Important methodological change

The Gaussian/GMM reference models are **not** fitted on Wikipedia anymore.

Wikipedia is used only to fit the shared layer-wise PCA representation.
For every configured language, the GMM is fitted on **100% of the available Aya samples for that language**, with no train/test split and no sampling cap.

Because the configured experiment uses `gmm_components: 1` and `gmm_covariance_type: diag`, the implementation uses streaming sufficient statistics for an exactly equivalent one-component diagonal Gaussian. This avoids a CPU-side scikit-learn GMM and keeps the fitted statistics on CUDA.

## PCA data path

The default PCA root is:

```text
outputs/pca_data/
```

For each model, layer files are saved as:

```text
outputs/pca_data/<model>/layer_000.pt
outputs/pca_data/<model>/layer_001.pt
...
outputs/pca_data/<model>/metadata.json
```

You can also provide an explicit PCA path at runtime:

```bash
python scripts/run_experiment.py \
  --config configs/config.yaml \
  --pca-path /kaggle/working/my_pca_data
```

When a complete PCA directory already exists, it is reused automatically. Refit with:

```bash
python scripts/run_experiment.py \
  --config configs/config.yaml \
  --pca-path /kaggle/working/my_pca_data \
  --force-refit-pca
```

## Aya GMM data

The GMM output is stored separately from PCA:

```text
outputs/aya_gmms/<model>/layer_000.pt
outputs/aya_gmms/<model>/layer_001.pt
...
outputs/aya_gmms/<model>/aya_gmm_fit_counts.csv
outputs/aya_gmms/<model>/aya_gmm_sample_counts.csv
outputs/aya_gmms/<model>/metadata.json
```

Each `layer_*.pt` contains the independent diagonal Gaussian parameters for all 18 languages at that layer. PCA is **not duplicated inside these GMM files**.

`aya_gmm_fit_counts.csv` records the number of Aya token vectors used for every language/layer. `aya_gmm_sample_counts.csv` records expected versus processed Aya sample rows; the run fails if those counts differ.

## GPU / CPU design

The main experiment is CUDA-only and fails immediately when CUDA is unavailable.

All representation/statistical tensor operations are performed on the GPU:

- transformer forward passes
- hidden-state extraction
- PCA mean/covariance accumulation
- PCA eigendecomposition
- PCA transforms
- Aya GMM sufficient-statistics accumulation
- GMM posterior calculations

The previous `hidden -> .cpu().numpy()` path was removed.

CPU is still necessarily used for ordinary non-tensor I/O and metadata handling, including tokenizer text processing, CSV/Parquet writing, and converting the small posterior/ID results needed by the output files. Raw hidden states are never copied to CPU.

`torch.cuda.empty_cache()` is not called inside the batch loops because repeated cache flushing can add synchronization overhead.

## Avoiding redundant computation

The PCA stage now uses a single streaming Wikipedia pass. Instead of running a second pass to fit a Gaussian reference model, the code directly accumulates PCA covariance statistics on GPU and saves the PCA once.

The Aya stage has two passes because the final GMM must be fitted on **all** Aya data before its posterior is used for output analysis:

1. Aya pass 1: fit the final GMMs on all Aya input tokens.
2. Aya pass 2: generate outputs, run one hidden-state forward pass, then perform one PCA transform + one posterior calculation per layer/batch.

Inside the analysis pass, token-level and sentence-level posteriors reuse the same layer/batch PCA and GMM result; the old code's per-token PCA calls followed by a second whole-sequence PCA call are gone.

## Models

Current main experiment:

- `ai-forever/mGPT`
- `bigscience/bloom-560m`

They run sequentially on `cuda:0`.

Qwen is retained as a disabled configuration slot for future experiments.
Gemma is intentionally not included.

## Languages

```text
ar bn en es eu fa fr hi id ml mr ne pt sw ta te ur vi
```

## Wikipedia reference data

The configured reference corpus is:

```text
CohereLabs/wikipedia-2023-11-embed-multilingual-v3
```

An equal deterministic cap of 3,000 documents per language is used. The dataset is streamed directly from Hugging Face.

Wikipedia is **not** used as Aya/GMM training data.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ is recommended. A CUDA-capable PyTorch build is required for the main experiment.

## Prepare Aya

Aya CSV files are intentionally not committed to GitHub. Prepare the complete configured dataset with:

```bash
python scripts/prepare_aya.py
```

This creates:

```text
data/aya_language_datasets/<language>.csv
```

## Run on Kaggle

Enable **Internet** and **GPU** in Kaggle.

```python
!git clone https://github.com/<USERNAME>/<REPOSITORY>.git
%cd <REPOSITORY>
!pip install -q -r requirements.txt
!python scripts/prepare_aya.py
!python scripts/run_experiment.py --config configs/config.yaml
```

Check the accelerator before the full run:

```python
import torch
print("CUDA:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
```

The program then fails fast unless CUDA is available and uses `cuda:0`.

## Outputs

```text
outputs/
├── manifests/
│   └── aya_input_summary.csv
├── pca_data/
│   └── <model>/
│       ├── layer_000.pt ...
│       ├── metadata.json
│       └── wikipedia_document_counts.csv
├── aya_gmms/
│   └── <model>/
│       ├── layer_000.pt ...
│       ├── metadata.json
│       ├── aya_gmm_fit_counts.csv
│       └── aya_gmm_sample_counts.csv
├── analysis_results/
│   └── <model>/
│       ├── generated_outputs/<language>.parquet
│       ├── token_posteriors/<language>.parquet
│       ├── sentence_level_posteriors/<language>.parquet
│       └── analysis_counts.csv
└── run_config.json
```

`token_posteriors/<language>.parquet` contains generated-token × layer posterior records.

`sentence_level_posteriors/<language>.parquet` contains one Aya sample × layer row, with posterior probabilities averaged over that sample's generated tokens.

## Reproducibility

The project seed is `42`. Wikipedia uses deterministic shuffled streaming with an equal per-language cap. Aya files are processed in their entirety for GMM fitting and analysis.

The effective PCA path and full configuration are written to `outputs/run_config.json`.
