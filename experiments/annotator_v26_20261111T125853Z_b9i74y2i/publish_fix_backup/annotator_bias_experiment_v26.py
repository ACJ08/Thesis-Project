"""Controlled cross-annotator generalization using native v26 training.

See ANNOTATOR_EXPERIMENT_V26.md for commands, overlap policy and inference.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import os
from pathlib import Path
import platform
from tempfile import mkdtemp
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from learning_curve_v26 import add_metrics, class_counts, read_json, write_json
from annotator_bias_reporting_v26 import generate_reports

ROOT = Path(__file__).resolve().parent
PREFIX = "annotator_experiment_"


def load_sources(v, paths):
    frames, sources, invalid = [], {}, []
    for source, path in paths.items():
        raw = pd.read_csv(path, low_memory=False)
        if "label_source" in raw and raw["label_source"].fillna("").astype(str).str.contains(
                "pseudo", case=False, regex=False).any():
            raise ValueError(f"Pseudolabeled records found in {path}; this experiment requires human labels.")
        frame = v.normalize_dataset_columns(raw, "Training dataset")
        frame["source"] = source
        frame["source_row_index"] = np.arange(len(raw))
        frame["song_uid"] = source + ":" + frame["source_row_index"].astype(str)
        frame["annotator"] = "A" if source == "A" else "B"
        frame["text"] = frame["text"].map(v.clean_lyrics)
        frame["explicitness"] = frame["explicitness"].astype(str).str.strip().str.lower().map(v.LABEL_NORMALIZATION)
        valid = frame["text"].ne("") & frame["explicitness"].notna()
        if v.EXCLUDE_NEEDS_REVISION:
            valid &= ~v.training_review_mask(frame)
        for song in frame.loc[~valid].itertuples():
            invalid.append({"song_uid": song.song_uid, "reason": "v26_invalid_or_excluded_review"})
        frame = frame.loc[valid].copy()
        frame["label"] = frame["explicitness"].map(v.LABEL2ID).astype(int)
        for column in raw:
            if column.lower() in {"song_id", "track_id", "spotify_id", "spotify_track_id", "isrc"}:
                frame["id_" + column.lower()] = raw.loc[frame.index, column]
        sources[source] = {"path": str(path), "sha256": v.file_sha256(path),
                           "raw_rows": len(raw), "valid_rows": len(frame)}
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), sources, invalid


def audit_pool(v, frame, invalid):
    """Audit the whole A/B pool, including unused songs and internal matches."""
    parents = list(range(len(frame)))
    findings = []
    records = frame.to_dict("records")

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def record(i, j, reason, **scores):
        parents[find(j)] = find(i)
        findings.append({"first_uid": records[i]["song_uid"], "second_uid": records[j]["song_uid"],
                         "reason": reason, **scores})

    keys = defaultdict(list)
    texts = []
    id_columns = [column for column in frame if column.startswith("id_")]
    for i, row in enumerate(records):
        identity = (v.normalize_identity(row["song_name"]), v.normalize_identity(row["artist"]))
        if all(identity):
            keys["title_artist", identity].append(i)
        text = v.normalize_overlap_lyrics(row["text"])
        texts.append(text)
        if text:
            keys["normalized_lyrics", text].append(i)
        for column in id_columns:
            if pd.notna(row[column]) and str(row[column]).strip():
                namespace = "isrc" if column == "id_isrc" else "song_id"
                keys[namespace, str(row[column]).strip().casefold()].append(i)
    for (kind, _), indices in keys.items():
        for j in indices[1:]:
            record(indices[0], j, kind)
    if any(len(text.split()) >= 5 for text in texts):
        print("Auditing within-pool and cross-pool five-word shingle pairs...", flush=True)
        matrix = CountVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False,
                                 ngram_range=(5, 5), binary=True, dtype=np.int32).fit_transform(texts)
        counts = matrix.getnnz(axis=1)
        reference = matrix.T.tocsr()
        for start in range(0, len(frame), 64):
            pairs = (matrix[start:start + 64] @ reference).tocoo()
            i, j = start + pairs.row, pairs.col
            shorter = np.minimum(counts[i], counts[j])
            containment = pairs.data / np.maximum(shorter, 1)
            jaccard = pairs.data / np.maximum(counts[i] + counts[j] - pairs.data, 1)
            matched = (i < j) & (shorter >= 10) & ((containment >= .94) | (jaccard >= .85))
            for a, b, c, score in zip(i[matched], j[matched], containment[matched], jaccard[matched]):
                if texts[a] != texts[b]:
                    record(int(a), int(b), "near_duplicate", containment=float(c), jaccard=float(score))
            if start % 2048 == 0:
                print(f"Overlap audit {min(start + 64, len(frame))}/{len(frame)} records", flush=True)
    groups = defaultdict(list)
    for i in range(len(frame)):
        groups[find(i)].append(i)
    excluded, components = [], []
    for indices in groups.values():
        if len(indices) == 1:
            continue
        ordered = sorted(indices, key=lambda i: (records[i]["annotator"], records[i]["source"],
                                                records[i]["source_row_index"]))
        excluded.extend(ordered[1:])
        components.append({"retained_uid": records[ordered[0]]["song_uid"],
                           "excluded_uids": [records[i]["song_uid"] for i in ordered[1:]]})
    clean = frame.drop(index=excluded).copy()
    report = {
        "policy": "One representative per connected component; A first, then source and row order.",
        "near_duplicate_policy": {"ngram_words": 5, "minimum_unique_shingles": 10,
                                  "jaccard_at_least": .85, "or_containment_at_least": .94},
        "limitations": "Lexical detection can miss paraphrases and short near-matches.",
        "invalid_rows": invalid, "matches": findings, "components": components,
        "excluded_records": frame.iloc[excluded][
            ["song_uid", "source", "source_row_index", "annotator", "song_name", "artist"]].to_dict("records"),
        "before_counts": frame.annotator.value_counts().to_dict(),
        "after_counts": clean.annotator.value_counts().to_dict(),
    }
    print(f"Overlap audit: {len(findings)} findings, {len(excluded)} excluded records. "
          f"Available counts: {report['after_counts']}", flush=True)
    return clean, report


def sample_partitions(v, pool, seeds):
    partitions, manifest = {}, {}
    for seed in seeds:
        manifest[str(seed)] = {}
        for annotator in ("A", "B"):
            data = pool.loc[pool.annotator.eq(annotator)]
            if len(data) < 5986:
                raise ValueError(f"Annotator {annotator}: only {len(data)} unique valid songs after "
                                 "reported exclusions; 5986 required for 4000/1000/986.")
            indices = data.index.to_numpy()
            unused = np.array([], dtype=int)
            if len(data) > 5986:
                indices, unused = train_test_split(indices, train_size=5986, stratify=data.label,
                                                    random_state=seed)
            training, rest = train_test_split(indices, train_size=4000,
                                               stratify=pool.loc[indices, "label"], random_state=seed)
            validation, test = train_test_split(rest, train_size=1000,
                                                stratify=pool.loc[rest, "label"], random_state=seed)
            allocated = {"training": training, "validation": validation, "test": test, "unused": unused}
            manifest[str(seed)][annotator] = {"pool_class_counts": class_counts(v, data)}
            for split, selected in allocated.items():
                subset = pool.loc[sorted(selected)].copy()
                partitions[seed, annotator, split] = subset
                manifest[str(seed)][annotator][split] = {
                    "count": len(subset), "class_counts": class_counts(v, subset),
                    "rows": subset[["song_uid", "source", "source_row_index"]].to_dict("records")}
            print(f"Seed {seed}, {annotator}: 4000 training / 1000 validation / 986 test / "
                  f"{len(unused)} unused", flush=True)
    return partitions, manifest


@contextmanager
def experiment_lock(root):
    """OS lock releases on exit, including interrupted sessions."""
    with (root / "experiment.lock").open("a+b") as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def verified(v, directory):
    marker = directory / "completed.json"
    if not marker.is_file():
        return False
    saved = read_json(marker)
    for relative, digest in saved["artifacts"].items():
        path = directory / relative
        if not path.is_file() or v.file_sha256(path) != digest:
            raise ValueError(f"Completed artifact missing or changed: {path}")
    return True


def publish(v, work, destination):
    artifacts = {str(path.relative_to(work)): v.file_sha256(path)
                 for path in work.rglob("*") if path.is_file()}
    write_json(work / "completed.json", {"artifacts": artifacts})
    work.rename(destination)


def cleanup_gpu(v):
    gc.collect()
    if v.torch.cuda.is_available():
        v.torch.cuda.empty_cache()


def train_checkpoint(v, name, seed, annotator, parts, directory, smoke, verify_sources):
    if verified(v, directory):
        print(f"Verified checkpoint, skipping training: {directory}", flush=True)
        return directory
    work = Path(mkdtemp(prefix="training_attempt_", dir=directory.parent))
    training, validation = parts[seed, annotator, "training"], parts[seed, annotator, "validation"]
    test = parts[seed, annotator, "test"]
    source_dir = work / "inputs"
    source_dir.mkdir()
    paths = {}
    for filename, subset in ((v.TRAIN_CSV_FILE, training), (v.VAL_CSV_FILE, validation), (v.TEST_CSV_FILE, test)):
        paths[filename] = source_dir / filename
        subset.to_csv(paths[filename], index=False)
    native = work / "experiments" / f"{name}_v26_annotator"
    with patch.multiple(v, PROJECT_DIR=work, RUN_ID="annotator", RUN_DIR=native,
                        MODEL_SAVE_PATH=native / "best_model.pt", RANDOM_SEED=seed,
                        SMOKE_TEST=smoke, RUN_FINAL_TEST=False, RUN_DEMO_EXPLANATIONS=False,
                        CHECK_SPLIT_OVERLAP=True, resolve_dataset_path=paths.__getitem__, _baseline_manifest=None):
        verify_sources()
        started = time.perf_counter()
        if name == "hybrid":
            v.main()
        else:
            v.run_baseline(name)
        seconds = time.perf_counter() - started
        manifest = read_json(native / "manifest.json")
        effective_train, effective_val = training, validation
        if smoke:
            indices, _ = train_test_split(np.arange(len(training)), train_size=v.SMOKE_TRAIN_ROWS,
                                          stratify=training.label, random_state=seed)
            effective_train = training.iloc[indices]
            indices, _ = train_test_split(np.arange(len(validation)), train_size=v.SMOKE_VAL_ROWS,
                                          stratify=validation.label, random_state=seed)
            effective_val = validation.iloc[indices]
        if manifest["training_rows"] != len(effective_train) or manifest["validation_rows"] != len(effective_val):
            raise RuntimeError("Native v26 sample counts differ from the planned allocation.")
        if manifest["training_class_counts"] != class_counts(v, effective_train):
            raise RuntimeError("Native v26 training class counts differ.")
        if manifest["checkpoint_sha256"] != v.file_sha256(native / "best_model.pt"):
            raise RuntimeError("Native checkpoint hash mismatch.")
        predictions = pd.read_csv(native / "validation_predictions.csv")
        if predictions.actual_id.tolist() != effective_val.label.tolist():
            raise RuntimeError("Validation prediction order differs from the sampling manifest.")
        write_json(native / "training_metadata.json", {
            "model": name, "seed": seed, "training_annotator": annotator,
            "training_duration_seconds": seconds, "smoke_test": smoke,
            "validation": add_metrics(read_json(native / "validation_metrics.json")),
            "effective_training_uids": effective_train.song_uid.tolist(),
            "effective_validation_uids": effective_val.song_uid.tolist(),
        })
        verify_sources()
    publish(v, native, directory)
    cleanup_gpu(v)
    return directory


def evaluate_checkpoint(v, name, seed, annotator, parts, checkpoint_dir, destination, verify_sources):
    if verified(v, destination):
        print(f"Verified paired evaluation, skipping: {destination}", flush=True)
        return
    work = Path(mkdtemp(prefix="evaluation_attempt_", dir=destination.parent))
    manifest = read_json(checkpoint_dir / "manifest.json")
    tokenizer = v.tokenizer_for(name, parts[seed, annotator, "training"], saved_dir=checkpoint_dir)
    model = v.make_baseline(name, tokenizer if name == "lstm_cnn" else None).to(v.DEVICE)
    model.load_state_dict(v.torch.load(checkpoint_dir / "best_model.pt", map_location="cpu", weights_only=True))
    model.eval()
    try:
        for target in ("A", "B"):
            condition = f"{annotator}_to_{target}"
            test = parts[seed, target, "test"]
            print(f"Evaluating {name}, seed {seed}, {condition}, n={len(test)}", flush=True)
            loader = v.DataLoader(v.LyricsDataset(test, tokenizer), batch_size=v.BATCH_SIZE, shuffle=False)
            actual, predicted = v.evaluate_test_set(model, loader)
            if actual != test.label.tolist():
                raise RuntimeError("Test song ordering or labels changed.")
            with patch.object(v, "RUN_DIR", work):
                metrics = add_metrics(v.save_predictions_and_metrics(condition, test, actual, predicted))
            predictions = pd.read_csv(work / f"{condition}_predictions.csv")
            for column in ("song_uid", "source", "source_row_index"):
                predictions[column] = test[column].to_numpy()
            predictions.to_csv(work / f"{condition}_predictions.csv", index=False)
            write_json(work / f"{condition}_metrics.json", {"model": name, "seed": seed, "condition": condition,
                       "checkpoint_sha256": manifest["checkpoint_sha256"], "metrics": metrics})
        verify_sources()
        publish(v, work, destination)
    finally:
        del model
        cleanup_gpu(v)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--models", nargs="+", choices=["hybrid", "bert_cnn_lstm", "lstm_cnn", "roberta"])
    parser.add_argument("--a", type=Path)
    parser.add_argument("--b-validation", type=Path)
    parser.add_argument("--b-test", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Native reduced-data one-epoch diagnostic")
    args = parser.parse_args()
    import lyric_classifier_v26_all_models as v

    root = args.resume.resolve() if args.resume else None
    saved = read_json(root / "experiment.json") if root else None
    seeds = args.seeds or (saved["seeds"] if saved else [67, 42, 123])
    models = args.models or (saved["models"] if saved else list(v.ALL_NAMES))
    smoke = saved["smoke_test"] if saved else args.smoke_test
    if args.smoke_test and not smoke:
        parser.error("Cannot convert a full experiment into a smoke test on resume.")
    if len(seeds) != len(set(seeds)) or any(seed < 0 or seed >= 2**32 for seed in seeds):
        parser.error("Seeds must be unique unsigned 32-bit integers.")
    if len(models) != len(set(models)):
        parser.error("Models must be unique.")
    if smoke and len(seeds) != 1:
        parser.error("Smoke tests require one explicit seed, e.g. --seeds 67.")
    arguments = {"A": args.a, "B_validation": args.b_validation, "B_test": args.b_test}
    defaults = {"A": v.TRAIN_CSV_FILE, "B_validation": v.VAL_CSV_FILE, "B_test": v.TEST_CSV_FILE}
    paths = {name: (path or (Path(saved["sources"][name]["path"]) if saved else
                            ROOT / "data" / defaults[name])).resolve() for name, path in arguments.items()}
    if len(set(paths.values())) != 3:
        parser.error("A and the two B sources must be three distinct files.")
    pool, sources, invalid = load_sources(v, paths)
    config = {
        "seeds": seeds, "models": models, "smoke_test": smoke, "sources": sources,
        "runner_sha256": v.file_sha256(__file__), "v26_sha256": v.file_sha256(v.__file__),
        "reporting_sha256": v.file_sha256(ROOT / "annotator_bias_reporting_v26.py"),
        "learning_curve_sha256": v.file_sha256(ROOT / "learning_curve_v26.py"),
        "label_mapping": v.LABEL2ID, "annotator_provenance": "Human annotation assignments supplied by user.",
        "allocation": {"training": 4000, "validation": 1000, "test": 986},
        "resamples": 2000, "python": platform.python_version(), "device": str(v.DEVICE),
        "dependencies": {name: v.installed_version(name) for name in
                         ("torch", "transformers", "numpy", "pandas", "scikit-learn", "scipy", "matplotlib")},
        "model_configuration": {name: getattr(v, name) for name in (
            "EPOCHS", "BATCH_SIZE", "GRADIENT_ACCUMULATION_STEPS", "LEARNING_RATE",
            "BASELINE_LEARNING_RATE", "BASELINE_LSTM_LEARNING_RATE", "WARMUP_RATIO",
            "EARLY_STOPPING_PATIENCE", "EARLY_STOPPING_MIN_DELTA", "EXCLUDE_NEEDS_REVISION")},
    }
    if saved and config != saved:
        raise ValueError("Sources, selections, pipeline, runner or environment changed since this experiment.")
    if root is None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        root = Path(mkdtemp(prefix=f"annotator_v26_{stamp}_", dir=args.output_dir)).resolve()
        write_json(root / "experiment.json", config)
    print(f"Experiment directory: {root}", flush=True)

    def verify_sources():
        for name, path in paths.items():
            if v.file_sha256(path) != sources[name]["sha256"]:
                raise ValueError(f"Source changed during experiment: {path}")

    with experiment_lock(root):
        clean, overlap = audit_pool(v, pool, invalid)
        overlap_path = root / f"{PREFIX}overlap_report.json"
        if overlap_path.exists() and read_json(overlap_path) != overlap:
            raise ValueError("Regenerated overlap audit differs from saved audit.")
        if not overlap_path.exists():
            write_json(overlap_path, overlap)
        prevalence = clean.groupby(["annotator", "explicitness"]).size().unstack(fill_value=0)
        preflight_summary = (
            "# Annotator experiment preflight\n\n"
            f"Available valid unique songs after the overlap policy: {overlap['after_counts']}.\n\n"
            "Required per annotator: 4000 training, 1000 validation and 986 test songs.\n\n"
            "Class counts in the available pools\n\n```csv\n" + prevalence.to_csv() + "```\n\n"
            "The overlap report identifies every excluded record. Original CSV files are preserved.\n\n"
            "Human annotation assignments were supplied by the user. Different song collections, class prevalence, "
            "annotation conventions and optimization variability can contribute to the measured differences. "
            "Causal annotation-quality comparisons require both annotators to label the same songs.\n"
        )
        summary_path = root / f"{PREFIX}summary.md"
        if not summary_path.exists():
            summary_path.write_text(preflight_summary, encoding="utf-8")
        parts, sampling = sample_partitions(v, clean, seeds)
        manifest = {"sources": sources, "seeds": seeds, "label_mapping": v.LABEL2ID, "sampling": sampling}
        manifest_path = root / f"{PREFIX}sampling_manifest.json"
        if manifest_path.exists() and read_json(manifest_path) != manifest:
            raise ValueError("Regenerated partitions differ from saved manifests.")
        if not manifest_path.exists():
            write_json(manifest_path, manifest)
        verify_sources()
        if args.plan:
            print("Preflight passed. Sampling and overlap reports saved.", flush=True)
            return
        total = len(seeds) * len(models) * 2
        completed = 0
        for seed in seeds:
            for model in models:
                for annotator in ("A", "B"):
                    directory = root / "runs" / f"seed_{seed}" / model / f"train_{annotator}"
                    directory.mkdir(parents=True, exist_ok=True)
                    print(f"[{completed + 1}/{total}] {model}, seed {seed}, training annotator {annotator}", flush=True)
                    checkpoint = train_checkpoint(v, model, seed, annotator, parts,
                                                  directory / "checkpoint", smoke, verify_sources)
                    evaluate_checkpoint(v, model, seed, annotator, parts, checkpoint,
                                        directory / "evaluation", verify_sources)
                    completed += 1
        generate_reports(v, root, models, seeds, smoke)
        verify_sources()


if __name__ == "__main__":
    main()
