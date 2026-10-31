#!/usr/bin/env python3
"""Prepare per-language CSV files from CohereLabs/aya_dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from datasets import load_dataset

from src.config import load_config


# Dataset language codes (mostly ISO 639-3) -> project codes (ISO 639-1).
# Only Standard Arabic (arb) is allowed to map to project language "ar".
STRICT_ARABIC_CODE = "arb"

# Other Arabic varieties and the Arabic macrolanguage are excluded.
EXCLUDED_ARABIC_CODES = {
    "ar", "ara",
    "arz", "ary", "ars",
    "ajp", "acq", "apc",
    "aeb", "acm",
}

# Dataset language codes -> project language codes.
# Arabic is deliberately excluded from this general mapping.
AYA_CODE_TO_ISO1 = {
    "ben": "bn", "bn": "bn",
    "eng": "en", "en": "en",
    "spa": "es", "es": "es",
    "eus": "eu", "baq": "eu", "eu": "eu",
    "pes": "fa", "fas": "fa", "per": "fa", "fa": "fa",
    "fra": "fr", "fre": "fr", "fr": "fr",
    "hin": "hi", "hi": "hi",
    "ind": "id", "id": "id",
    "mal": "ml", "ml": "ml",
    "mar": "mr", "mr": "mr",
    "npi": "ne", "nep": "ne", "ne": "ne",
    "por": "pt", "pt": "pt",
    "swh": "sw", "swa": "sw", "sw": "sw",
    "tam": "ta", "ta": "ta",
    "tel": "te", "te": "te",
    "urd": "ur", "ur": "ur",
    "vie": "vi", "vi": "vi",
    "zho": "zh", "chi": "zh", "cmn": "zh", "zh": "zh",
    "jpn": "ja", "ja": "ja",
}


def _map_aya_language_code(value: object) -> str | None:
    if pd.isna(value):
        return None

    code = str(value).strip().lower()

    # Strict Arabic rule: only Standard Arabic is accepted.
    if code == STRICT_ARABIC_CODE:
        return "ar"

    # Explicitly reject Arabic dialects and the Arabic macrolanguage.
    if code in EXCLUDED_ARABIC_CODES:
        return None

    mapped = AYA_CODE_TO_ISO1.get(code)

    if mapped is not None:
        return mapped

    # Fallback for other languages only.
    try:
        import pycountry

        language = pycountry.languages.get(alpha_3=code)

        if language is not None and hasattr(language, "alpha_2"):
            alpha_2 = str(language.alpha_2)

            # Never let the fallback map another code to Arabic.
            if alpha_2 == "ar":
                return None

            return alpha_2

    except ImportError:
        pass

    # Two-letter fallback, explicitly excluding Arabic.
    if len(code) == 2 and code != "ar":
        return code

    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--dataset", default="CohereLabs/aya_dataset")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    languages = list(cfg["languages"])
    language_names = cfg["language_names"]
    output_dir = Path(args.output_dir or cfg["data"]["root_dir"])
    if not output_dir.is_absolute():
        output_dir = Path(cfg.get("_repo_root", ROOT)) / output_dir
    seed = int(args.seed if args.seed is not None else cfg["data"].get("sampling_seed", cfg["project"].get("seed", 42)))

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {args.dataset} ...")
    aya = load_dataset(args.dataset, split="train")
    needed = ["inputs", "targets", "language", "language_code"]
    missing = [column for column in needed if column not in aya.column_names]
    if missing:
        raise ValueError(f"Aya dataset is missing required columns: {missing}")
    df = aya.select_columns(needed).to_pandas()

    df["final_language_code"] = df["language_code"].apply(_map_aya_language_code)
    # Audit Arabic source codes before filtering.
    print("\nArabic source-code audit:")

    raw_codes = (
        df["language_code"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    arabic_audit = raw_codes[
        raw_codes.isin(EXCLUDED_ARABIC_CODES | {STRICT_ARABIC_CODE})
    ].value_counts()

    print(arabic_audit.to_string())

    # Only arb is permitted to produce project language "ar".
    selected_arabic = df[df["final_language_code"] == "ar"].copy()

    if not selected_arabic.empty:
        selected_codes = set(
            selected_arabic["language_code"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        if selected_codes != {"arb"}:
            raise RuntimeError(
                f"Arabic filtering failed. Unexpected source codes: {selected_codes}"
            )

    print(
        f"\nStandard Arabic (arb) selected: "
        f"{len(selected_arabic):,} samples"
    )

    df = df[df["final_language_code"].isin(languages)].copy()

    for lang in languages:
        language_df = df[df["final_language_code"] == lang].copy()
        if language_df.empty:
            raise RuntimeError(
                f"Aya contains no rows for configured language {lang} ({language_names[lang]}). "
                "Check the dataset language-code mapping and availability."
            )
        language_df = language_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        language_df.insert(0, "sample_id", range(1, len(language_df) + 1))
        language_df.insert(2, "main_language", language_names[lang])
        language_df = language_df[[
            "sample_id", "final_language_code", "main_language", "language",
            "language_code", "inputs", "targets",
        ]]
        language_df.to_csv(output_dir / f"{lang}.csv", index=False, encoding="utf-8-sig")
        print(f"{lang:>3} | {len(language_df):>8,}")

    summary = pd.DataFrame([
        {
            "language_code": lang,
            "language": language_names[lang],
            "num_samples": int((df["final_language_code"] == lang).sum()),
        }
        for lang in languages
    ])
    summary.to_csv(output_dir / "aya_language_dataset_summary.csv", index=False, encoding="utf-8-sig")
    print(f"Total samples: {summary['num_samples'].sum():,}")
    print(f"Saved per-language CSV files to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
