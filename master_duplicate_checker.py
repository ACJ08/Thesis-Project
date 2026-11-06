#!/usr/bin/env python3
"""Master duplicate checker and non-destructive cleaner for lyric-classifier datasets.

DEFAULT, NO ARGUMENTS:
  Place this script beside data/Training.csv, data/Validation.csv, data/Testing.csv
  (or beside the CSVs) and run: python master_duplicate_checker.py

OUTPUTS (in the same data folder by default):
  Training_Cleaned.csv, Validation_Cleaned.csv, Testing_Cleaned.csv
  duplicate_reports/duplicate_removals.csv
  duplicate_reports/possible_duplicates_review.csv
  duplicate_reports/dedup_summary.json

Automatic removal is conservative about lyrics but strict about song identities:
  * Matching normalized title + artist: keep only the highest-priority song.
  * Identical normalized lyrics of at least 20 words: keep only the highest-priority
    song, including when titles or performers differ (covers/remixes).
  * Potential near matches: --scan-near produces a REVIEW report without
    deleting them. Use --remove-near to additionally auto-remove exceptionally
    strong near matches (0.985 Jaccard AND text ratio).
  * Short repeated lyric snippets without matching title+artist are REVIEW ONLY.

Priority: Validation > Testing > Training. Within Training, user-provided
annotations outrank pseudo-labels. Rows surviving cleaning retain their exact
original column values/order. All input files remain unchanged.

The optional approximate near-match scanner uses bounded 5-word-shingle fingerprints.
It is NOT exhaustive. Human review is needed for possible re-recordings,
substantially reordered verses, or mislabeled covers.

Requires only the Python standard library; optional rapidfuzz accelerates
near-match comparisons (pip install rapidfuzz). Compatible with Python 3.10+.
"""

import argparse
import csv
import hashlib
import heapq
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher

try:
    from rapidfuzz.fuzz import ratio as _fast_text_ratio
except ImportError:
    _fast_text_ratio = None
from pathlib import Path
from typing import Dict, List, Tuple

SPLITS = ("Validation", "Testing", "Training")
ALIASES = {
    "title": ("song_title", "song", "title", "song title", "song_name", "track_name"),
    "artist": ("artist", "artist(s)", "artists", "artist_name"),
    "lyrics": ("lyrics", "text", "lyric", "song_lyrics"),
}
SECTION_TAG = re.compile(
    r"\[(?:verse|chorus|pre[- ]?chorus|post[- ]?chorus|bridge|intro|outro|"
    r"hook|refrain|interlude|instrumental|break|solo|repeat|spoken|sample|"
    r"drop|ending|part)(?:\s*[:#-]?\s*[^\]\n]*)?\]",
    re.IGNORECASE,
)
NONWORD = re.compile(r"[^\W_]+", re.UNICODE)
REMOVAL_COLUMNS = (
    "removed_split", "removed_csv_line", "removed_title", "removed_artist",
    "removed_label", "removed_label_source", "removed_dataset_split",
    "kept_split", "kept_csv_line",
    "kept_title", "kept_artist", "kept_label", "kept_label_source",
    "kept_dataset_split",
    "reason", "lyrics_same_after_normalization", "label_conflict",
    "lyric_similarity", "five_word_jaccard", "five_word_containment",
)
REVIEW_COLUMNS = REMOVAL_COLUMNS + ("review_note",)


def raise_csv_field_limit():
    maximum = sys.maxsize
    while maximum > 1024:
        try:
            csv.field_size_limit(maximum)
            break
        except OverflowError:
            maximum //= 10


def identity_part(text):
    norm = unicodedata.normalize("NFKC", text).casefold()
    cleaned = "".join(NONWORD.findall(norm))
    # Legitimate artist/song names such as "!!!", "???", ":", "-" or "..."
    # consist entirely of punctuation. Preserve them as distinct identities.
    return cleaned if cleaned else "symbol:" + "".join(norm.split())


def lyric_normalize(text):
    norm = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(NONWORD.findall(SECTION_TAG.sub(" ", norm)))


def column_names(columns, filename):
    """Resolve the same title/artist/lyrics aliases used by the three older scripts."""
    lowered = {col.casefold().strip(): col for col in columns}
    result = {}
    for concept, possibilities in ALIASES.items():
        found = [lowered[alias] for alias in possibilities if alias in lowered]
        if len(set(found)) != 1:
            raise ValueError(f"{filename}: expected exactly one {concept} column; got {found}.")
        result[concept] = found[0]
    return result


@dataclass
class Song:
    split: str
    line: int
    row: Dict[str, str]
    title: str
    artist: str
    label: str
    label_source: str
    identity: Tuple[str, str]
    lyrics: str
    word_count: int
    lyric_digest: str
    original_index: int


def load_csv(path: Path, split: str):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        if not columns:
            raise ValueError(f"{path}: empty CSV or no header")
        names = column_names(columns, path.name)
        songs = []
        for idx, row in enumerate(reader):
            if None in row:
                raise ValueError(f"{path}: row {idx + 2} has more values than headers")
            for key in names.values():
                if not (row.get(key) or "").strip():
                    raise ValueError(f"{path}: row {idx + 2} has an empty {key!r}")
            title = row[names["title"]]
            artist = row[names["artist"]]
            identity = (identity_part(title), identity_part(artist))
            text = lyric_normalize(row[names["lyrics"]])
            if not identity[0] or not identity[1] or not text:
                raise ValueError(f"{path}: row {idx + 2} has empty normalized metadata/lyrics")
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            songs.append(Song(
                split=split, line=idx + 2, row=row, title=title, artist=artist,
                label=(row.get("safety_label") or "").strip(),
                label_source=(row.get("label_source") or "").strip(),
                identity=identity, lyrics=text, word_count=len(text.split()),
                lyric_digest=digest, original_index=idx,
            ))
    return songs, columns


def priority(song):
    # Gold-standard evaluation sets are protected from training rows.
    source_rank = {"Validation": 0, "Testing": 1, "Training": 2}[song.split]
    human_rank = 0 if song.label_source.casefold() not in ("pseudo_label", "pseudo", "model") else 1
    return (source_rank, human_rank, song.original_index)


def compare_lyrics(a, b):
    if _fast_text_ratio is not None:
        return _fast_text_ratio(a.lyrics, b.lyrics) / 100.0
    # Faster standard-library fallback, though less exact on repetitive lyrics.
    return SequenceMatcher(None, a.lyrics, b.lyrics, autojunk=True).ratio()


def description(removed, kept, reason, similarity="", jaccard="", containment=""):
    return {
        "removed_split": removed.split, "removed_csv_line": removed.line,
        "removed_title": removed.title, "removed_artist": removed.artist,
        "removed_label": removed.label, "removed_label_source": removed.label_source,
        "removed_dataset_split": removed.row.get("dataset_split", ""),
        "kept_split": kept.split, "kept_csv_line": kept.line,
        "kept_title": kept.title, "kept_artist": kept.artist,
        "kept_label": kept.label, "kept_label_source": kept.label_source,
        "kept_dataset_split": kept.row.get("dataset_split", ""),
        "reason": reason,
        "lyrics_same_after_normalization": removed.lyric_digest == kept.lyric_digest,
        "label_conflict": bool(removed.label and kept.label and
                               removed.label.casefold() != kept.label.casefold()),
        "lyric_similarity": similarity, "five_word_jaccard": jaccard,
        "five_word_containment": containment,
    }


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def exact_clean(songs, min_lyrics_words):
    """Deduplicate based on strict song identity OR sufficiently long exact lyrics."""
    seen_id = {}
    seen_lyrics = {}
    kept = []
    removed = set()
    reports = []
    review = []
    for song in sorted(songs, key=priority):
        existing = seen_id.get(song.identity)
        if existing is not None:
            similarity = round(compare_lyrics(song, existing), 6)
            item = description(song, existing, "normalized_title_and_artist", similarity)
            reports.append(item)
            if similarity < 0.90 or item["label_conflict"]:
                note = dict(item)
                note["review_note"] = (
                    "Identity collision removed, but lyrics differ materially and/or "
                    "labels conflict. Inspect versions and taxonomy annotations."
                )
                review.append(note)
            removed.add((song.split, song.original_index))
            continue
        existing = seen_lyrics.get(song.lyric_digest)
        if existing is not None:
            if song.word_count >= min_lyrics_words and existing.word_count >= min_lyrics_words:
                item = description(song, existing, "identical_normalized_lyrics")
                reports.append(item)
                if item["label_conflict"]:
                    note = dict(item)
                    note["review_note"] = (
                        "Identical normalized lyrics have conflicting safety labels; "
                        "check annotation consistency before evaluating the model."
                    )
                    review.append(note)
                removed.add((song.split, song.original_index))
                continue
            item = description(song, existing, "short_identical_lyrics")
            item["review_note"] = "Short shared lyrics; retained unless title+artist also match."
            review.append(item)
        kept.append(song)
        seen_id[song.identity] = song
        # Even if short, retain one reference to flag future same snippets.
        seen_lyrics.setdefault(song.lyric_digest, song)
    return kept, removed, reports, review


def fingerprints(lyrics, k=5, limit=96):
    """Bottom-K hash signatures. Fast candidate discovery; not an exhaustive index."""
    words = lyrics.split()
    if len(words) < k:
        return (), set()
    grams = {hash(tuple(words[i:i+k])) for i in range(len(words) - k + 1)}
    return heapq.nsmallest(limit, grams), grams


def near_clean(kept, removed, removals, review, args):
    """Check full-lyrics near duplicates found by bounded 5-word-shingle fingerprints.

    For each new song, references are only higher-priority retained rows.
    Default is report-only. With --remove-near, remove exceptionally close text.
    """
    postings = defaultdict(list)
    reference = []
    for current, song in enumerate(sorted(kept, key=priority), 1):
        signatures, grams = fingerprints(song.lyrics, limit=args.fingerprint_count)
        counts = Counter()
        for sig in signatures:
            indices = postings.get(sig, ())
            if len(indices) <= args.max_postings:
                counts.update(indices)
        best = None
        # Score only plausible candidates; near matching may still be missed.
        for rid, overlap_count in counts.most_common(args.max_candidates):
            if overlap_count < args.min_shared_fingerprints:
                break
            prev = reference[rid][0]
            if min(song.word_count, prev.word_count) < args.min_lyrics_words:
                continue
            prev_grams = reference[rid][1]
            overlap = len(grams & prev_grams)
            if overlap == 0:
                continue
            jac = overlap / (len(grams) + len(prev_grams) - overlap)
            contain = overlap / min(len(grams), len(prev_grams))
            if max(jac, contain) < args.near_threshold:
                continue
            # Avoid expensive text comparison for marginal shingle matches.
            score = compare_lyrics(song, prev)
            item = description(song, prev, "possible_near_lyrics", round(score, 6),
                               round(jac, 6), round(contain, 6))
            item["review_note"] = (
                "Very similar lyric text; covers/remixes may have different titles/artists. "
                "Review before deciding whether to exclude."
            )
            review.append(item)
            if (args.remove_near and jac >= args.auto_threshold and
                    score >= args.auto_threshold):
                if best is None or (jac, score) > best[0]:
                    best = ((jac, score), prev, item)
        if best is not None:
            _, prev, item = best
            report = description(song, prev, "high_confidence_near_lyrics_auto_removed",
                                 item["lyric_similarity"], item["five_word_jaccard"],
                                 item["five_word_containment"])
            removals.append(report)
            removed.add((song.split, song.original_index))
        else:
            rid = len(reference)
            reference.append((song, grams))
            for sig in signatures:
                postings[sig].append(rid)
        if current % 5000 == 0:
            print(f"  Near scan: {current:,}/{len(kept):,} rows", flush=True)
    return removed, removals, review


def assert_no_exact_leakage(cleaned, min_lyrics_words):
    identities = set()
    long_lyrics = set()
    for song in cleaned:
        if song.identity in identities:
            raise RuntimeError("Output still has duplicate normalized song identities")
        identities.add(song.identity)
        if song.word_count >= min_lyrics_words:
            if song.lyric_digest in long_lyrics:
                raise RuntimeError("Output still has duplicate normalized long lyrics")
            long_lyrics.add(song.lyric_digest)


def parse_args():
    root = Path(__file__).resolve().parent
    auto_dir = root / "data" if (root / "data" / "Training.csv").exists() else root
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=auto_dir,
                        help="Folder containing the original three CSV files")
    for split in ("Training", "Validation", "Testing"):
        parser.add_argument("--" + split.casefold(), type=Path,
                            help=f"Optional direct path to {split}.csv")
    parser.add_argument("--output-dir", type=Path,
                        help="Where to save all _Cleaned.csv files and reports (default: data directory)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacing previous _Cleaned outputs/reports; NEVER overwrites originals")
    parser.add_argument("--min-lyrics-words", type=int, default=20)
    parser.add_argument("--scan-near", action="store_true",
                        help="Also run approximate cross-artist near-lyrics review scan (slower)")
    parser.add_argument("--skip-near", action="store_true",
                        help="Explicitly skip near-lyrics scan (default behavior)")
    parser.add_argument("--remove-near", action="store_true",
                        help="Also auto-remove near lyrics meeting exceptionally strict thresholds")
    parser.add_argument("--near-threshold", type=float, default=0.94,
                        help="Flag if 5-word shingle Jaccard OR containment >= this value")
    parser.add_argument("--auto-threshold", type=float, default=0.985,
                        help="With --remove-near, BOTH Jaccard and text ratio must exceed")
    parser.add_argument("--fingerprint-count", type=int, default=48)
    parser.add_argument("--min-shared-fingerprints", type=int, default=3)
    parser.add_argument("--max-postings", type=int, default=250)
    parser.add_argument("--max-candidates", type=int, default=60)
    args = parser.parse_args()
    if not 0 < args.near_threshold <= 1 or not 0 < args.auto_threshold <= 1:
        parser.error("Near thresholds must be within (0, 1]")
    for name in ("min_lyrics_words", "fingerprint_count", "min_shared_fingerprints",
                 "max_postings", "max_candidates"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    if (args.remove_near or args.scan_near) and args.skip_near:
        parser.error("Cannot combine --scan-near/--remove-near with --skip-near")
    return args


def main():
    args = parse_args()
    raise_csv_field_limit()
    inputs = {}
    for split in ("Training", "Validation", "Testing"):
        override = getattr(args, split.casefold())
        inputs[split] = override if override is not None else args.data_dir / f"{split}.csv"
        if not inputs[split].is_file():
            raise FileNotFoundError(
                f"Missing {inputs[split]}. Put this script beside your data/ directory, "
                f"or pass --{split.casefold()} with the exact file path."
            )
    output_dir = args.output_dir if args.output_dir else args.data_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "duplicate_reports"
    output_paths = {split: output_dir / f"{split}_Cleaned.csv" for split in inputs}
    removals_path = reports_dir / "duplicate_removals.csv"
    review_path = reports_dir / "possible_duplicates_review.csv"
    summary_path = reports_dir / "dedup_summary.json"
    outputs = [*output_paths.values(), removals_path, review_path, summary_path]
    input_resolved = {path.resolve() for path in inputs.values()}
    if any(p.resolve() in input_resolved for p in outputs):
        raise RuntimeError("An output would overwrite an original input. No files changed.")
    existing = [p for p in outputs if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Outputs already exist; use --overwrite to replace them: " +
                              ", ".join(str(p) for p in existing))

    all_songs = []
    columns_by_split = {}
    by_split = {}
    for split in SPLITS:
        songs, columns = load_csv(inputs[split], split)
        columns_by_split[split] = columns
        by_split[split] = songs
        all_songs.extend(songs)
        print(f"Read {split}: {len(songs):,} rows from {inputs[split]}", flush=True)
    print("Exact identity and lyric checks (Validation > Testing > Training)...", flush=True)
    kept, removed, removals, review = exact_clean(all_songs, args.min_lyrics_words)
    print(f"  Exact duplicate rows identified: {len(removed):,}", flush=True)
    scan_near = (args.scan_near or args.remove_near) and not args.skip_near
    if scan_near:
        print("Approximate near-lyrics audit across retained rows (may take time)...", flush=True)
        removed, removals, review = near_clean(kept, removed, removals, review, args)
    cleaned = [song for song in all_songs if (song.split, song.original_index) not in removed]
    assert_no_exact_leakage(cleaned, args.min_lyrics_words)
    summaries = {}
    for split in ("Training", "Validation", "Testing"):
        kept_in_split = [song.row for song in by_split[split]
                         if (song.split, song.original_index) not in removed]
        write_csv(output_paths[split], columns_by_split[split], kept_in_split)
        summaries[split] = {
            "original": len(by_split[split]),
            "removed": len(by_split[split]) - len(kept_in_split),
            "cleaned": len(kept_in_split),
            "input": str(inputs[split]), "output": str(output_paths[split]),
        }
        print(f"  {split}: {len(by_split[split]):,} -> {len(kept_in_split):,} "
              f"(removed {summaries[split]['removed']:,})")
    write_csv(removals_path, REMOVAL_COLUMNS, removals)
    write_csv(review_path, REVIEW_COLUMNS, review)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "priority": list(SPLITS),
            "same_title_and_artist": "Remove lower-priority records even if lyric versions differ",
            "normalized_lyrics_exact": f"Remove if at least {args.min_lyrics_words} words",
            "short_exact_lyrics": "Retain, flag for review",
            "near_matches": "Auto-remove strict near matches" if args.remove_near else
                            "Flag for review only" if scan_near else "Not scanned",
            "near_scan_is_exhaustive": False,
            "auto_threshold_jaccard_and_ratio": args.auto_threshold,
        },
        "counts": summaries,
        "total_rows_original": len(all_songs),
        "total_rows_cleaned": len(cleaned),
        "removal_reasons": dict(Counter(r["reason"] for r in removals)),
        "removed_training_subsplit_counts": dict(Counter(
            r["removed_dataset_split"] for r in removals
            if r["removed_split"] == "Training"
        )),
        "label_conflicts_among_removed": sum(bool(r["label_conflict"]) for r in removals),
        "review_pair_count": len(review),
        "reports": [str(removals_path), str(review_path)],
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Reports: {removals_path}; {review_path}; {summary_path}")
    print("Original CSVs were NOT modified. Labels are NOT changed. Review near matches.")
    exposed = report["removed_training_subsplit_counts"].get("train", 0)
    if exposed:
        print(f"METHODOLOGY ALERT: {exposed} removed records came from the teacher train split. "
              "If the teacher already trained on those records, cleaning the CSVs "
              "does not erase that earlier exposure to external evaluation songs.")
    print("IMPORTANT: Re-evaluate models on only the cleaned external Validation/Testing sets.")


if __name__ == "__main__":
    main()
