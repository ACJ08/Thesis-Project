"""Relabel the cleaned splits with Spotify's Explicit flag."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from pathlib import Path
import re
import unicodedata


SPLITS = ("Training", "Validation", "Testing")
LABELS = {"yes": "Explicit", "no": "Safe"}
WORD = re.compile(r"[^\W_]+")


def identity(value):
    return "".join(WORD.findall(unicodedata.normalize("NFKC", value).casefold()))


def track_key(title, artist):
    return identity(title), identity(artist)


def lyric_key(value):
    words = WORD.findall(unicodedata.normalize("NFKC", value).casefold())
    return hashlib.sha256(" ".join(words).encode("utf-8")).digest()


def spotify_index(path, wanted):
    matches = defaultdict(lambda: defaultdict(set))
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source, strict=True):
            key = track_key(row["song"], row["Artist(s)"])
            if key in wanted:
                flag = row["Explicit"].strip().casefold()
                matches[key][lyric_key(row["text"])].add(flag)
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spotify", type=Path, default=Path.home() /
                        ".cache/kagglehub/datasets/devdope/900k-spotify/versions/3/spotify_dataset.csv")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    args = parser.parse_args()
    csv.field_size_limit(10_000_000)

    datasets = {}
    wanted = set()
    for split in SPLITS:
        path = args.data_dir / f"{split}_Cleaned.csv"
        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source, strict=True)
            fields = reader.fieldnames
            if not {"song_title", "artist", "lyrics", "safety_label"}.issubset(fields or []):
                raise ValueError(f"{path} is missing required columns")
            rows = list(reader)
        datasets[split] = fields, rows
        wanted.update(track_key(row["song_title"], row["artist"]) for row in rows)

    matches = spotify_index(args.spotify, wanted)
    failures = []
    for split, (_, rows) in datasets.items():
        counts = Counter()
        for index, row in enumerate(rows, start=1):
            key = track_key(row["song_title"], row["artist"])
            candidates = matches.get(key, {})
            exact = candidates.get(lyric_key(row["lyrics"])) if len(candidates) > 1 else None
            flags = exact or set().union(*candidates.values())
            if len(flags) == 1 and next(iter(flags)) in LABELS:
                label = LABELS[next(iter(flags))]
                row["safety_label"] = label
                counts[label] += 1
            else:
                reason = "no match" if not candidates else f"unresolved flags {sorted(flags)}"
                failures.append((split, index, row["song_title"], row["artist"], reason))
        print(f"{split}: {len(rows):,} rows, {dict(counts)}")

    if failures:
        for failure in failures[:20]:
            print(ascii(failure))
        parser.exit(1, f"{len(failures):,} unresolved rows. No output written.\n")

    for split, (fields, rows) in datasets.items():
        path = args.data_dir / f"{split}_Cleaned_binary.csv"
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)


if __name__ == "__main__":
    main()
