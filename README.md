# Multilingual Representation Analysis — Kaggle Main Experiment

This repository contains the **full main experiment**. There is no smoke-test configuration.

## Research design

```text
CohereLabs/wikipedia-2023-11-embed-multilingual-v3
        │
        │  18 languages × up to 3,000 passages/language
        ↓
BLOOM / mGPT forward pass (no generation)
        ↓
Hidden states at every model layer
        ↓
Layer-wise Incremental PCA (700 components)
        ↓
One diagonal Gaussian per language and layer
        │
        │  frozen reference space
        ↓
All Aya samples (100% — no 80/20 split)
        ↓
Generation (up to 64 new tokens)
        ↓
Generated-token hidden states
        ↓
PCA.transform()
        ↓
Language posterior under Wikipedia-fitted Gaussians
        ↓
Token-level + sample/sentence-level outputs
```

Wikipedia is the **reference corpus** used only to fit the representation space. Aya is the **evaluation corpus**. Wikipedia is never used as an instruction-generation dataset.

The experiment uses the following 18 languages:

```text
ar bn en es eu fa fr hi id ml mr ne pt sw ta te ur vi
```

## Why the Wikipedia cap is 3,000 per language

The Wikipedia dataset has very large language subsets. The experiment therefore uses an equal deterministic cap of 3,000 passages for every configured language. This gives at most 54,000 Wikipedia passages per model and prevents high-resource languages from dominating the fitted reference space.

Wikipedia is streamed from Hugging Face, so the full dataset is not downloaded locally.

## Models

The current experiment contains:

- `ai-forever/mGPT`
- `bigscience/bloom-560m`

They run **sequentially**, one model at a time. Each model uses a single CUDA device (or CPU when CUDA is unavailable).

Qwen is retained as a disabled future configuration slot. Gemma is not included.

## Runtime configuration

The main experiment uses one GPU/device at a time. Wikipedia is processed in **8-passage batches per language**. A 3,000-passage language therefore runs as six pipeline chunks. This is intentionally separate from the Aya evaluation batch size, which remains smaller to control generation memory.

```yaml
runtime:
  device: auto
  device_map: null
  mixed_precision: fp16
  wikipedia_batch_size: 32
  evaluation_batch_size: 8
```

The implementation also avoids allocating the CausalLM vocabulary logits during Wikipedia representation extraction; it calls the underlying transformer body and requests hidden states only. This substantially reduces unnecessary GPU memory use.

## Installation

Python 3.10+ is recommended. A CUDA-capable GPU is strongly recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Prepare Aya

Aya CSV files are intentionally not committed to GitHub. Generate them with:

```bash
python scripts/prepare_aya.py
```

This creates the complete 18-language Aya dataset under:

```text
data/aya_language_datasets/
```

The generated CSV files contain the full available samples for the configured language set.

## Run the main experiment

```bash
python scripts/run_experiment.py --config configs/config.yaml
```

The script runs each enabled model sequentially. No model is generated from Wikipedia; Wikipedia is used for fitting PCA/Gaussian reference models. Every available Aya sample is then evaluated.

## Outputs

```text
outputs/
├── manifests/
│   ├── aya_input_summary.csv
│   ├── wikipedia_fit_samples.parquet
│   └── wikipedia_language_counts.csv
├── trained_language_models/
│   ├── mgpt/
│   │   ├── layer_000.pkl ...
│   │   └── wikipedia_gaussian_fit_counts.csv
│   └── bloom/
│       ├── layer_000.pkl ...
│       └── wikipedia_gaussian_fit_counts.csv
├── analysis_results/
│   ├── mgpt/
│   │   ├── generated_outputs/<language>.parquet
│   │   ├── token_posteriors/<language>.parquet
│   │   ├── sentence_level_posteriors/<language>.parquet
│   │   └── analysis_counts.csv
│   └── bloom/
│       └── ...
└── run_config.json
```

`token_posteriors/<language>.parquet` contains one row per **generated token × layer** and includes `token`, `token_id`, `token_position`, `sample_id`, `input_language`, `layer`, and `P_<language>` posterior columns.

`sentence_level_posteriors/<language>.parquet` contains one row per **Aya sample × layer**. The posterior columns are the mean posterior over that sample's generated tokens for the corresponding layer.

Parquet is used because the complete experiment can produce very large token-level output files.

## Reproducibility

The project seed is `42`. Wikipedia uses deterministic shuffled streaming with an equal per-language cap. Aya sample preparation is deterministic. The effective configuration is saved to `outputs/run_config.json`.

For a long-lived frozen dataset version, set `wikipedia.revision` in `configs/config.yaml` to a fixed Hugging Face revision rather than `null`.

## Kaggle

On Kaggle, enable **Internet** and **GPU**. The experiment uses one GPU (`cuda:0`) at a time; a second GPU is not required. GPU use is required by the main configuration.

```python
!git clone https://github.com/<USERNAME>/<REPOSITORY>.git
%cd <REPOSITORY>
!pip install -q -r requirements.txt
!python scripts/prepare_aya.py
!python scripts/run_experiment.py --config configs/config.yaml
```

Check GPU visibility before the full run:

```python
import torch
print("CUDA:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
```

The run performs a CUDA preflight and then places the model on `cuda:0`. It fails fast if CUDA is unavailable. The selected GPU and model parameter device are printed before inference.

The Wikipedia corpus is streamed directly from Hugging Face and does not need to be added as a Kaggle Dataset.

## Important computational note

This is the full research run. The reference phase uses up to 3,000 Wikipedia passages per language, PCA has 700 components, and the Aya stage processes every available sample. Token-level results are written incrementally to Parquet so they do not need to remain in RAM.
