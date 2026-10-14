"""Merge completed pseudo-label outputs with the original 1,500 annotated songs.

Place this script beside your `data` directory and run:
    python merge_labeled_songs.py

This script does NOT train or modify the model, review queue, or iteration history.
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent / "data"
ORIGINAL_TRAIN = DATA_DIR / "training_1000_provisional.csv"
ORIGINAL_MONITOR = DATA_DIR / "monitoring_500_provisional.csv"
PSEUDO_OUTPUT = DATA_DIR / "Training.csv"  # 38,500 candidates, including archived corrections
TRAIN_EXPORT = DATA_DIR / "Training_39500.csv"
COMPLETE_EXPORT = DATA_DIR / "Complete_Labeled_Dataset_40000.csv"
RANDOM_SEED = 42
VALID_LABELS = {"safe", "mild", "explicit"}


def read_songs(path, expected_count):
    if not path.is_file():
        raise FileNotFoundError(f"Missing input: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    required = {"song_title", "artist", "lyrics", "safety_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    if len(df) != expected_count:
        raise ValueError(f"{path.name}: expected {expected_count:,} rows; found {len(df):,}.")
    for col in ["song_title", "artist", "lyrics", "safety_label"]:
        if df[col].isna().any() or df[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"{path.name}: missing/blank {col} values")
        df[col] = df[col].astype(str).str.strip()
    df["safety_label"] = df["safety_label"].str.casefold()
    invalid = set(df["safety_label"]) - VALID_LABELS
    if invalid:
        raise ValueError(f"{path.name}: invalid labels {sorted(invalid)}")
    return df


def keys(df):
    # Match the training pipeline's case-insensitive title + artist identity rule.
    return (df["song_title"].str.casefold().str.strip() + "|||"
            + df["artist"].str.casefold().str.strip())


def main():
    original_train = read_songs(ORIGINAL_TRAIN, 1000)
    monitoring = read_songs(ORIGINAL_MONITOR, 500)
    candidates = read_songs(PSEUDO_OUTPUT, 38500)

    # Recreate the *same* deterministic monitoring split as the training script.
    validation, held_out_test = train_test_split(
        monitoring, test_size=0.5, random_state=RANDOM_SEED,
        stratify=monitoring["safety_label"]
    )

    groups = [
        (original_train, "train", "original_training"),
        (candidates, "train", "pseudo_labeling_output"),
        (validation, "validation", "original_monitoring"),
        (held_out_test, "test", "original_monitoring"),
    ]
    combined_parts = []
    for df, split, source in groups:
        part = df.copy()
        part["dataset_split"] = split
        part["dataset_source"] = source
        # Preserve annotation provenance: don't claim provisional labels are verified.
        if "label_source" not in part.columns:
            part["label_source"] = (
                "provisional_annotation" if source != "pseudo_labeling_output"
                else "pseudo_label"
            )
        combined_parts.append(part)

    combined = pd.concat(combined_parts, ignore_index=True, sort=False)
    combined["_identity_key"] = keys(combined)
    duplicate_mask = combined.duplicated("_identity_key", keep=False)
    if duplicate_mask.any():
        examples = combined.loc[duplicate_mask, ["song_title", "artist", "dataset_split"]].head(12)
        raise ValueError(
            f"Found {duplicate_mask.sum():,} rows sharing title+artist identity across inputs. "
            "No files were written. Investigate these records:\n"
            + examples.to_string(index=False)
        )

    if len(combined) != 40000 or combined["dataset_split"].value_counts().to_dict() != {
        "train": 39500, "validation": 250, "test": 250
    }:
        raise ValueError("Final dataset size or split sizes are incorrect; no files written.")

    combined = combined.drop(columns="_identity_key")
    training_only = combined[combined["dataset_split"] == "train"].copy()
    # Write only after all integrity checks pass.
    training_only.to_csv(TRAIN_EXPORT, index=False, encoding="utf-8-sig")
    combined.to_csv(COMPLETE_EXPORT, index=False, encoding="utf-8-sig")

    print(f"Training-only export: {TRAIN_EXPORT} ({len(training_only):,} rows)")
    print(f"Complete export:      {COMPLETE_EXPORT} ({len(combined):,} rows)")
    print("Split counts:", combined["dataset_split"].value_counts().to_dict())
    print("Unique song identities:", keys(combined).nunique())
    print("No model training or plateau history changes were performed.")
    print("Note: Provisional and pseudo-labels remain unverified unless separately reviewed.")


if __name__ == "__main__":
    main()
