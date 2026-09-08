# Multilingual Representation Analysis — Kaggle Main Experiment

This repository is a **full main-experiment Kaggle version**. It is not a smoke test and it does not use a reduced sample count.

## Experimental design

```text
Wikipedia multilingual text
        ↓
BLOOM / mGPT hidden states
        ↓
PCA fit (700 components, per layer)
        ↓
1-component diagonal Gaussian/GMM per language
        ↓
All Aya samples
        ↓
generation → generated-token hidden states
        ↓
PCA transform
        ↓
GMM posterior
```

The reference/fitting corpus is `CohereLabs/wikipedia-2023-11-embed-multilingual-v3`. It is streamed directly from Hugging Face. The dataset has very large language subsets, so the experiment uses an equal deterministic cap of **3,000 Wikipedia passages per language** by default. The dataset card reports close to 250M passages overall. citeturn790182search0

Aya is **not split into train/test or 80/20**. Every available sample in each of the 18 languages is passed through the evaluation stage.

## 18 languages

```text
ar bn en es eu fa fr hi id ml mr ne pt sw ta te ur vi
```

## Models

The current experiment runs only:

- mGPT — `ai-forever/mGPT`
- BLOOM — `bigscience/bloom-560m`

They run sequentially, one model at a time. Qwen is kept as a disabled placeholder in the config for future experiments with other Qwen sizes. Gemma is removed.

## Kaggle setup

### 1. Create a Kaggle Notebook

Use a GPU accelerator. Turn **Internet** on because the notebook streams Hugging Face datasets and downloads model weights.

### 2. Clone

```python
!git clone https://github.com/<USERNAME>/<REPOSITORY>.git
%cd <REPOSITORY>
```

### 3. Install

```python
!pip install -q -r requirements.txt
```

### 4. Prepare Aya

```python
!python scripts/prepare_aya.py
```

This creates the complete 18-language Aya CSV set under:

```text
data/aya_language_datasets/
```

The CSV files are intentionally not committed to GitHub. They are generated in the Kaggle working directory from the Aya dataset.

### 5. Run the main experiment

```python
!python scripts/run_experiment.py --config configs/config.yaml
```

Do not change the sample counts to 1, 10, or another smoke-test value. The supplied configuration is the main experiment.

## What happens during the run?

For each model:

1. Wikipedia is streamed, 3,000 passages per language.
2. Hidden states are extracted with a normal forward pass.
3. PCA is fitted layer-by-layer using 700 components.
4. A one-component diagonal Gaussian is fitted for every language at every layer in PCA space.
5. The entire Aya corpus is processed.
6. Each Aya instruction is used to generate up to 64 tokens.
7. Hidden states of the generated tokens are extracted.
8. Aya representations are transformed by the frozen Wikipedia PCA.
9. Language posterior probabilities are computed from the frozen Wikipedia-fitted Gaussian models.
10. Results are written incrementally to Parquet files grouped by language.

## Outputs

Look under:

```text
outputs/
```

In particular:

```text
outputs/analysis_results/mgpt/token_posteriors/
outputs/analysis_results/mgpt/sentence_level_posteriors/
outputs/analysis_results/bloom/token_posteriors/
outputs/analysis_results/bloom/sentence_level_posteriors/
```

Token-level data has posterior columns such as `P_en`, `P_fa`, etc. Sentence/sample-level data contains the mean posterior over the generated tokens for each layer.

## Important

This is the full experiment. It can create a large amount of output data because token-level results contain every generated token at every hidden-state layer for every Aya sample. Parquet is used to make this feasible to store and analyze.
