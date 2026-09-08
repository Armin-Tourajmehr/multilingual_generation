
#!/usr/bin/env python3
"""Prepare the complete Aya inputs for the fixed 18-language experiment."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset

LANGUAGES = ["ar", "bn", "en", "es", "eu", "fa", "fr", "hi", "id", "ml", "mr", "ne", "pt", "sw", "ta", "te", "ur", "vi"]
LANGUAGE_NAMES = {
    "ar":"Arabic","bn":"Bengali","en":"English","es":"Spanish","eu":"Basque","fa":"Persian",
    "fr":"French","hi":"Hindi","id":"Indonesian","ml":"Malayalam","mr":"Marathi","ne":"Nepali",
    "pt":"Portuguese","sw":"Swahili","ta":"Tamil","te":"Telugu","ur":"Urdu","vi":"Vietnamese"
}
AYA_CODE_TO_ISO1 = {
    "arb":"ar","arz":"ar","ary":"ar","ars":"ar","ajp":"ar","acq":"ar","ara":"ar","ar":"ar",
    "ben":"bn","bn":"bn","eng":"en","en":"en","spa":"es","es":"es","eus":"eu","baq":"eu","eu":"eu",
    "pes":"fa","fas":"fa","per":"fa","fa":"fa","fra":"fr","fre":"fr","fr":"fr","hin":"hi","hi":"hi",
    "ind":"id","id":"id","mal":"ml","ml":"ml","mar":"mr","mr":"mr","npi":"ne","nep":"ne","ne":"ne",
    "por":"pt","pt":"pt","swh":"sw","swa":"sw","sw":"sw","tam":"ta","ta":"ta","tel":"te","te":"te",
    "urd":"ur","ur":"ur","vie":"vi","vi":"vi"
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="CohereLabs/aya_dataset")
    parser.add_argument("--output-dir", default="data/aya_language_datasets")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Loading {args.dataset} ...")
    aya = load_dataset(args.dataset, split="train")
    cols = ["inputs", "targets", "language", "language_code"]
    df = aya.select_columns(cols).to_pandas()

    df["final_language_code"] = df["language_code"].apply(
        lambda x: AYA_CODE_TO_ISO1.get(str(x).strip().lower()) if pd.notna(x) else None
    )
    df = df[df["final_language_code"].isin(LANGUAGES)].copy()

    for lang in LANGUAGES:
        language_df = df[df["final_language_code"] == lang].copy()
        language_df = language_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        language_df.insert(0, "sample_id", range(1, len(language_df) + 1))
        language_df.insert(2, "main_language", LANGUAGE_NAMES[lang])
        language_df = language_df[["sample_id","final_language_code","main_language","language","language_code","inputs","targets"]]
        language_df.to_csv(out / f"{lang}.csv", index=False, encoding="utf-8-sig")
        print(f"{lang:>3} | {len(language_df):>8,}")

    summary = pd.DataFrame([{"language_code": lang, "language": LANGUAGE_NAMES[lang], "num_samples": len(df[df["final_language_code"] == lang])} for lang in LANGUAGES])
    summary.to_csv(out / "aya_language_dataset_summary.csv", index=False, encoding="utf-8-sig")
    print(f"Total samples: {summary['num_samples'].sum():,}")


if __name__ == "__main__":
    main()
