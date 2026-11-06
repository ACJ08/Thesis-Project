"""Merge three song CSVs, then exclude songs already present in labeled.csv.

Put this script alongside your four CSV files and run:
    python merge_and_exclude_labeled.py

Outputs:
    merged_40000.csv                - all rows from the three source files
    merged_excluding_labeled.csv    - songs not found in labeled.csv

Song identity is determined by song_title + artist (case/whitespace insensitive).
Labels and all other original columns are preserved.
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
INPUT_FILES = [
    "remaining_38500_unlabeled.csv",
    "training_1000_provisional.csv",
    "monitoring_500_provisional.csv",
]
LABELED_FILE = "labeled.csv"
MERGED_OUTPUT = DATA_DIR / "merged_40000.csv"
FILTERED_OUTPUT = DATA_DIR / "merged_excluding_labeled.csv"
REQUIRED_COLUMNS = {"song_title", "artist", "lyrics"}


def find_input(filename: str) -> Path:
    """Accept both normal names and the uploaded '(1)' filename variants."""
    name = Path(filename)
    candidates = [DATA_DIR / filename, DATA_DIR / f"{name.stem}(1){name.suffix}"]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Cannot find {filename} (or its '(1)' variant) in {DATA_DIR}")


def load_csv(filename: str) -> pd.DataFrame:
    path = find_input(filename)
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    if df[["song_title", "artist"]].isna().any().any():
        raise ValueError(f"{path.name} contains missing song titles or artists")
    print(f"Read {path.name}: {len(df):,} songs")
    return df


def song_keys(df: pd.DataFrame) -> pd.Series:
    """Ignore capitalization, extra surrounding spaces, and repeated whitespace."""
    title = (df["song_title"].astype("string").str.strip()
             .str.replace(r"\s+", " ", regex=True).str.casefold())
    artist = (df["artist"].astype("string").str.strip()
              .str.replace(r"\s+", " ", regex=True).str.casefold())
    if title.eq("").any() or artist.eq("").any():
        raise ValueError("A source contains blank song_title or artist entries")
    return title + "|||" + artist


def main() -> None:
    sources = [load_csv(filename) for filename in INPUT_FILES]
    already_labeled = load_csv(LABELED_FILE)

    merged = pd.concat(sources, ignore_index=True, sort=False)
    merged.to_csv(MERGED_OUTPUT, index=False, encoding="utf-8-sig")
    print(f"\nSaved raw merged file: {MERGED_OUTPUT.name} ({len(merged):,} rows)")

    labeled_keys = set(song_keys(already_labeled))
    merged_keys = song_keys(merged)
    existing_mask = merged_keys.isin(labeled_keys)
    filtered = merged.loc[~existing_mask].copy()
    print(f"Excluded {int(existing_mask.sum()):,} songs also found in {LABELED_FILE}")

    # Defensive: if source files overlap, retain a row with an existing label.
    filtered["_key"] = song_keys(filtered)
    filtered["_original_order"] = range(len(filtered))
    if "safety_label" in filtered.columns:
        filtered["_has_label"] = (
            filtered["safety_label"].notna()
            & filtered["safety_label"].astype("string").str.strip().ne("")
        )
    else:
        filtered["_has_label"] = False

    filtered = (
        filtered.sort_values("_has_label", ascending=False, kind="stable")
        .drop_duplicates(subset="_key", keep="first")
        .sort_values("_original_order", kind="stable")
        .drop(columns=["_key", "_original_order", "_has_label"])
    )
    internal_duplicates_removed = int((~existing_mask).sum() - len(filtered))
    print(f"Removed {internal_duplicates_removed:,} additional duplicates within the merged sources")

    filtered.to_csv(FILTERED_OUTPUT, index=False, encoding="utf-8-sig")
    print(f"Saved filtered file: {FILTERED_OUTPUT.name} ({len(filtered):,} rows)")
    print("All original columns are preserved; no song in the filtered file matches labeled.csv.")


if __name__ == "__main__":
    main()
