#!/usr/bin/env python3
"""Create language-separated Aya datasets for the main experiment.

Language variants/dialects are mapped to a main language, Persian is included
explicitly, and every sample belonging to a final language is preserved.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset


MGPT_LANGUAGES = {
    "af", "ar", "az", "ba", "be", "bg", "bn", "bxr", "cv", "da", "de", "el", "en", "es",
    "et", "eu", "fa", "fi", "fr", "he", "hi", "hu", "id", "it", "ja", "jv", "ka", "kk",
    "ko", "ky", "lt", "lv", "ml", "mn", "mr", "ms", "my", "ne", "os", "pl", "pt", "ro",
    "ru", "sax", "sv", "sw", "ta", "te", "tg", "th", "tk", "tr", "tt", "tyv", "uk", "ur",
    "uz", "vi", "xal", "yo"
}

BLOOM_LANGUAGES = {
    "ak", "ar", "as", "bm", "bn", "ca", "en", "es", "eu", "fon", "fr", "gu", "hi",
    "id", "ig", "ki", "kn", "lg", "ln", "ml", "mr", "ne", "nso", "ny", "or", "pa",
    "pt", "rn", "rw", "sn", "st", "sw", "ta", "te", "tn", "ts", "tum", "tw", "ur",
    "vi", "wo", "xh", "yo", "zu", "zh"
}

QWEN3_LANGUAGES = {
    "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "br", "bs", "ca", "cs", "cy",
    "da", "de", "el", "en", "es", "et", "eu", "fa", "fi", "fr", "ga", "gl", "gu", "he",
    "hi", "hr", "hu", "hy", "id", "is", "it", "ja", "jv", "ka", "kk", "km", "kn", "ko",
    "ky", "lo", "lt", "lv", "ml", "mn", "mr", "ms", "my", "ne", "nl", "no", "or", "pa",
    "pl", "pt", "ro", "ru", "sk", "sl", "sq", "sr", "sv", "sw", "ta", "te", "tg", "th",
    "tk", "tr", "tt", "uk", "ur", "uz", "vi", "zh"
}

GEMMA3_LANGUAGES = {
    "af", "ar", "as", "az", "ba", "be", "bg", "bn", "ca", "cs", "cy", "da", "de", "el",
    "en", "es", "et", "eu", "fa", "fi", "fr", "gu", "he", "hi", "hr", "hu", "id", "it",
    "ja", "jv", "ka", "kk", "kn", "ko", "ky", "lt", "lv", "ml", "mn", "mr", "ms", "my",
    "ne", "nl", "no", "or", "pa", "pl", "pt", "ro", "ru", "sk", "sl", "sr", "sv", "sw",
    "ta", "te", "tg", "th", "tk", "tr", "tt", "uk", "ur", "uz", "vi", "zh"
}

STRICT_INTERSECTION = MGPT_LANGUAGES & BLOOM_LANGUAGES & QWEN3_LANGUAGES & GEMMA3_LANGUAGES
MODEL_EXPERIMENT_LANGUAGES = STRICT_INTERSECTION | {"fa"}

LANGUAGE_NAMES = {
    "ar": "Arabic", "bn": "Bengali", "en": "English", "es": "Spanish", "eu": "Basque",
    "fa": "Persian", "fr": "French", "hi": "Hindi", "id": "Indonesian", "ml": "Malayalam",
    "mr": "Marathi", "ne": "Nepali", "pt": "Portuguese", "sw": "Swahili", "ta": "Tamil",
    "te": "Telugu", "ur": "Urdu", "vi": "Vietnamese",
}

AYA_CODE_TO_ISO1 = {
    "arb": "ar", "arz": "ar", "ary": "ar", "ars": "ar", "ajp": "ar", "acq": "ar", "ara": "ar", "ar": "ar",
    "ben": "bn", "bn": "bn", "eng": "en", "en": "en", "spa": "es", "es": "es",
    "eus": "eu", "baq": "eu", "eu": "eu",
    "pes": "fa", "fas": "fa", "per": "fa", "fa": "fa",
    "fra": "fr", "fre": "fr", "fr": "fr", "hin": "hi", "hi": "hi",
    "ind": "id", "id": "id", "mal": "ml", "ml": "ml", "mar": "mr", "mr": "mr",
    "npi": "ne", "nep": "ne", "ne": "ne", "por": "pt", "pt": "pt",
    "swh": "sw", "swa": "sw", "sw": "sw", "tam": "ta", "ta": "ta",
    "tel": "te", "te": "te", "urd": "ur", "ur": "ur", "vie": "vi", "vi": "vi",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="CohereLabs/aya_dataset")
    parser.add_argument("--output-dir", default="data/aya_language_datasets")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MODEL LANGUAGE INTERSECTION")
    print("=" * 80)
    print(f"\nStrict intersection: {len(STRICT_INTERSECTION)} languages")
    print(f"Final experiment languages (including Persian): {len(MODEL_EXPERIMENT_LANGUAGES)}")
    print(sorted(MODEL_EXPERIMENT_LANGUAGES))

    print("\n" + "=" * 80)
    print("LOADING AYA DATASET")
    print("=" * 80)
    aya = load_dataset(args.dataset, split="train")
    print(f"\nAya samples: {len(aya):,}")

    df = aya.select_columns(["inputs", "targets", "language", "language_code"]).to_pandas()

    def normalize_language(code):
        if pd.isna(code):
            return None
        return AYA_CODE_TO_ISO1.get(str(code).strip().lower())

    df["final_language_code"] = df["language_code"].apply(normalize_language)

    unmapped = (
        df[df["final_language_code"].isna()][["language", "language_code"]]
        .drop_duplicates()
        .sort_values(["language", "language_code"])
    )
    print("\n" + "=" * 80)
    print("AYA MAPPING CHECK")
    print("=" * 80)
    print(f"\nUnmapped Aya language/code combinations: {len(unmapped)}")
    if len(unmapped):
        print(unmapped.to_string(index=False))

    aya_available = set(df["final_language_code"].dropna().unique())
    final_languages = MODEL_EXPERIMENT_LANGUAGES & aya_available

    print("\n" + "=" * 80)
    print("FINAL EXPERIMENT LANGUAGES")
    print("=" * 80)
    print(f"\nNumber of final languages: {len(final_languages)}\n")
    for lang in sorted(final_languages):
        print(f"{lang:>3}  {LANGUAGE_NAMES[lang]}")

    summary = []
    for lang in sorted(final_languages):
        language_df = df[df["final_language_code"] == lang].copy()
        language_df = language_df.sample(frac=1, random_state=42).reset_index(drop=True)
        language_df.insert(0, "sample_id", range(1, len(language_df) + 1))
        language_df.insert(2, "main_language", LANGUAGE_NAMES[lang])
        language_df = language_df[[
            "sample_id", "final_language_code", "main_language",
            "language", "language_code", "inputs", "targets"
        ]]
        output_file = out / f"{lang}.csv"
        language_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        summary.append({
            "language_code": lang,
            "language": LANGUAGE_NAMES[lang],
            "num_samples": len(language_df),
            "file": str(output_file),
        })
        print(f"{lang:>3} | {len(language_df):>8,} samples | {output_file}")

    summary_df = pd.DataFrame(summary).sort_values("language_code").reset_index(drop=True)
    summary_df.to_csv(out / "aya_language_dataset_summary.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"\nLanguages created: {len(summary_df)}")
    print(f"Total samples across final languages: {summary_df['num_samples'].sum():,}")
    print(f"\nOutput directory: {out}")
    print(f"Summary file: {out / 'aya_language_dataset_summary.csv'}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
