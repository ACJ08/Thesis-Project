"""Train all v26 models on nested samples from data/Training_Cleaned.csv.

Run with Python in the v26 environment, with matplotlib installed.
Use --plan to audit the fixed datasets and --resume DIR to continue a saved run.
Training duration is the wall time of the native v26 training entry point,
including setup and validation but excluding held-out test evaluation.
Power-law intervals describe seed variation conditional on the fixed holdouts.
"""

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import mkdtemp
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "Training_Cleaned.csv"
SIZES = (1000, 2000, 3000, 4000, 5000, "ALL")
DISPLAY = {
    "hybrid": "Hybrid", "bert_cnn_lstm": "BERT_CNN_LSTM",
    "lstm_cnn": "LSTM_CNN", "roberta": "RoBERTa",
}


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def class_counts(v, frame):
    return {label: int(frame["explicitness"].eq(label).sum()) for label in v.LABEL2ID}


def prepare(v, path, name):
    raw = v.pd.read_csv(path, low_memory=False)
    frame = v.normalize_dataset_columns(raw, f"{name.title()} dataset")
    if name == "training" and v.EXCLUDE_NEEDS_REVISION:
        frame = frame.loc[~v.training_review_mask(frame)].copy()
    frame["text"] = frame["text"].map(v.clean_lyrics)
    frame["explicitness"] = frame["explicitness"].astype(str).str.strip().str.lower().map(v.LABEL_NORMALIZATION)
    frame = frame.loc[frame["text"].ne("") & frame["explicitness"].notna()].copy()
    frame["label"] = frame["explicitness"].map(v.LABEL2ID)
    print(f"{name}: {len(frame):,} valid songs / {len(raw):,} source rows", flush=True)
    expected = {"validation": 5000, "test": 4998}.get(name)
    if expected and (len(raw) != expected or len(frame) != expected):
        raise ValueError(f"{path} must contain exactly {expected} usable rows in original order.")
    if not all(class_counts(v, frame).values()):
        raise ValueError(f"{path} does not contain all v26 classes.")
    return raw, frame


def audit_overlap(v, datasets, report):
    """Compare all split pairs using exact keys and exhaustive lexical shingles."""
    from collections import defaultdict
    from itertools import combinations, product
    from sklearn.feature_extraction.text import CountVectorizer

    def describe(name, row):
        song = datasets[name][1].iloc[row]
        return {"split": name, "source_row_index": int(song.name),
                "title": str(song.song_name), "artist": str(song.artist)}

    def keys(name):
        raw, frame = datasets[name]
        groups = defaultdict(list)
        lyrics = []
        id_columns = [c for c in raw if c.lower() in
                      {"song_id", "track_id", "spotify_id", "spotify_track_id", "isrc"}]
        for position, song in enumerate(frame.itertuples()):
            identity = (v.normalize_identity(song.song_name), v.normalize_identity(song.artist))
            if all(identity):
                groups["title_artist", identity].append(position)
            lyric = v.normalize_overlap_lyrics(song.text)
            lyrics.append(lyric)
            if lyric:
                groups["normalized_lyrics", lyric].append(position)
            for column in id_columns:
                value = raw.at[song.Index, column]
                if v.pd.notna(value) and str(value).strip():
                    groups["song_id", str(value).strip().casefold()].append(position)
        return groups, lyrics

    indexed = {name: keys(name) for name in datasets}
    hits = []

    def record(left, right, i, j, reason, **scores):
        hits.append({"first": describe(left, i), "second": describe(right, j),
                     "reason": reason, **scores})

    for left, right in combinations(datasets, 2):
        left_keys, left_texts = indexed[left]
        right_keys, right_texts = indexed[right]
        for key in sorted(left_keys.keys() & right_keys.keys()):
            for i, j in product(left_keys[key], right_keys[key]):
                record(left, right, i, j, key[0])
        # Sparse dot products count shared unique five-word sequences, in bounded batches.
        texts = left_texts + right_texts
        if not any(len(text.split()) >= 5 for text in texts):
            continue
        matrix = CountVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False,
                                 ngram_range=(5, 5), binary=True, dtype=v.np.int32).fit_transform(texts)
        count = matrix.getnnz(axis=1)
        offset = len(left_texts)
        reference = matrix[offset:].T.tocsr()
        for start in range(0, offset, 64):
            pairs = (matrix[start:min(start + 64, offset)] @ reference).tocoo()
            first = start + pairs.row
            minimum = v.np.minimum(count[first], count[offset + pairs.col])
            containment = pairs.data / minimum
            jaccard = pairs.data / (count[first] + count[offset + pairs.col] - pairs.data)
            match = (minimum >= 10) & ((containment >= 0.94) | (jaccard >= 0.85))
            for i, j, c, score in zip(first[match], pairs.col[match], containment[match], jaccard[match]):
                if left_texts[i] != right_texts[j]:
                    record(left, right, int(i), int(j), "near_duplicate",
                           containment=float(c), jaccard=float(score))
    if hits:
        write_json(report, hits)
        raise ValueError(f"Potential leakage ({len(hits)} findings). See {report}. Training stopped.")


def training_provenance(v, raw, train):
    fields = ("label_source", "review_status", "reviewed_by", "annotator", "human_label")
    metadata = {field: int(raw.loc[train.index, field].notna().sum())
                for field in fields if field in raw}
    sources = {}
    if "label_source" in raw:
        sources = raw.loc[train.index, "label_source"].dropna().astype(str).value_counts().to_dict()
    flags = int(v.training_review_mask(raw).loc[train.index].sum())
    return {"human_verification_confirmed": False, "nonmissing_provenance_fields": metadata,
            "observed_label_sources": sources, "review_flagged_valid_rows": flags,
            "uses_predicted_label_fallback": "predicted_label" in raw and
            not {"safety_label", "explicitness"}.intersection(raw.columns)}


def nested_order(v, train, seed):
    rng = v.np.random.default_rng(seed)
    ranked = []
    for label, group in train.groupby("label", sort=True):
        shuffled = rng.permutation(group.index.to_numpy())
        # Evenly interleave each class in proportion to its pool size.
        ranked.extend(((i + 0.5) / len(shuffled), label, int(row))
                      for i, row in enumerate(shuffled))
    return [row for _, _, row in sorted(ranked)]


def add_metrics(metrics):
    classes = list(metrics["per_class"].values())
    for key in ("precision", "recall"):
        metrics[f"macro_{key}"] = sum(row[key] for row in classes) / len(classes)
    for row in classes:
        row["error_rate"] = 1 - row["recall"] if row["support"] else None
    metrics["confusion_matrix_label_order"] = list(metrics["per_class"])
    return metrics


def train_configuration(v, name, seed, size, rows, paths, frames, destination, verify_inputs):
    prefix = f"{DISPLAY[name]}_{size}"
    # Each attempt has its own workspace. Publish only after all files are complete.
    attempt = Path(mkdtemp(prefix="try_", dir=destination.parent))
    run_id = "lc"
    native = attempt / "experiments" / f"{name}_v26_{run_id}"
    datasets = dict(zip((v.TRAIN_CSV_FILE, v.VAL_CSV_FILE, v.TEST_CSV_FILE), paths))
    train, validation, test = frames
    with patch.multiple(
        v, PROJECT_DIR=attempt, RUN_ID=run_id, RUN_DIR=native,
        MODEL_SAVE_PATH=native / "best_model.pt", RANDOM_SEED=seed, SMOKE_TEST=False,
        RUN_FINAL_TEST=False, RUN_DEMO_EXPLANATIONS=False, CHECK_SPLIT_OVERLAP=True,
        resolve_dataset_path=datasets.__getitem__, _baseline_manifest=None,
    ):
        started = time.perf_counter()
        if name == "hybrid":
            v.main()
        else:
            v.run_baseline(name)
        duration = time.perf_counter() - started
        manifest = read_json(native / "manifest.json")
        counts = class_counts(v, train)
        if manifest["training_rows"] != len(rows) or manifest["training_class_counts"] != counts:
            raise RuntimeError("Native v26 training data differs from the planned subset.")
        checkpoint = native / "best_model.pt"
        if v.file_sha256(checkpoint) != manifest["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint hash differs from the v26 manifest.")
        gc.collect()
        if v.torch.cuda.is_available():
            v.torch.cuda.empty_cache()
        tokenizer = v.tokenizer_for(name, train, saved_dir=native)
        model = v.make_baseline(name, tokenizer if name == "lstm_cnn" else None).to(v.DEVICE)
        model.load_state_dict(v.torch.load(checkpoint, map_location="cpu", weights_only=True))
        saved_validation = v.pd.read_csv(native / "validation_predictions.csv")
        if saved_validation["actual_id"].tolist() != validation["label"].tolist():
            raise RuntimeError("Validation prediction order differs from the fixed validation set.")
        model.eval()
        loader = v.DataLoader(v.LyricsDataset(test, tokenizer), batch_size=v.BATCH_SIZE)
        actual, predicted = v.evaluate_test_set(model, loader)
        if actual != test["label"].tolist():
            raise RuntimeError("Test prediction order differs from the fixed test split.")
        test_metrics = add_metrics(v.save_predictions_and_metrics("test", test, actual, predicted))
        validation_metrics = add_metrics(read_json(native / "validation_metrics.json"))
        write_json(native / "test_metrics.json", test_metrics)
        write_json(native / "validation_metrics.json", validation_metrics)
        best_epoch = manifest["best_epoch"]
        best = v.pd.read_csv(native / "epoch_history.csv").set_index("epoch").loc[best_epoch]
        record = {
            "model": DISPLAY[name], "size": str(size), "seed": seed,
            "training_songs": len(rows), "training_class_counts": counts,
            "class_weights": manifest["class_weights"],
            "validation_class_counts": class_counts(v, validation),
            "test_class_counts": class_counts(v, test),
            "training_duration_seconds": duration, "best_epoch": best_epoch,
            "best_validation_loss": float(best["validation_loss"]),
            "validation": validation_metrics, "test": test_metrics,
            "checkpoint": f"checkpoints/{prefix}_best_model.pt",
        }
        verify_inputs()
        for artifact in list(native.iterdir()):
            category = ("checkpoints" if artifact.suffix == ".pt" or artifact.name == "vocabulary.json"
                        else "predictions" if artifact.name.endswith("_predictions.csv") else "metrics")
            folder = native / category
            folder.mkdir(exist_ok=True)
            artifact.rename(folder / f"{prefix}_{artifact.name}")
        write_json(native / "metrics" / f"{prefix}_metrics.json", record)
        del model, loader
        gc.collect()
        if v.torch.cuda.is_available():
            v.torch.cuda.empty_cache()
    native.rename(destination)
    native.parent.rmdir()
    attempt.rmdir()
    return record


def paired_comparisons(v, completed, report):
    records = []
    np = v.np
    for (seed, size), group in completed.items():
        predictions = {}
        for name, directory in group.items():
            prefix = f"{DISPLAY[name]}_{size}"
            frame = v.pd.read_csv(directory / "predictions" / f"{prefix}_test_predictions.csv")
            predictions[name] = frame[["actual_id", "predicted_id"]].to_numpy()
        if "hybrid" not in predictions:
            continue
        actual, hybrid = predictions["hybrid"].T

        def macro_f1(truth, pred):
            cm = np.bincount(3 * truth + pred, minlength=9).reshape(3, 3)
            denominator = cm.sum(axis=0) + cm.sum(axis=1)
            return np.divide(2 * cm.diagonal(), denominator, out=np.zeros(3),
                             where=denominator != 0).mean()

        rng = np.random.default_rng(seed)
        for name in v.BASELINE_NAMES:
            if name not in predictions:
                continue
            truth, other = predictions[name].T
            if not np.array_equal(actual, truth):
                raise RuntimeError("Paired test predictions have different actual labels.")
            observed = macro_f1(actual, hybrid) - macro_f1(actual, other)
            differences, extreme = [], 0
            for _ in range(2000):
                idx = rng.integers(0, len(actual), len(actual))
                differences.append(macro_f1(actual[idx], hybrid[idx])
                                   - macro_f1(actual[idx], other[idx]))
                swap = rng.random(len(actual)) < 0.5
                permuted = (macro_f1(actual, np.where(swap, other, hybrid))
                            - macro_f1(actual, np.where(swap, hybrid, other)))
                extreme += abs(permuted) >= abs(observed) - 1e-12
            low, high = np.percentile(differences, [2.5, 97.5])
            records.append({
                "seed": seed, "size": size, "comparison": f"Hybrid_minus_{DISPLAY[name]}",
                "macro_f1_difference": observed, "paired_bootstrap_ci95_low": low,
                "paired_bootstrap_ci95_high": high, "paired_permutation_p_two_sided": (1 + extreme) / 2001,
                "bootstrap_resamples": 2000, "permutation_resamples": 2000,
            })
    if records:
        v.pd.DataFrame(records).to_csv(report / "paired_macro_f1_bootstrap.csv", index=False)


def power_law(v, points, targets, seed):
    from scipy.optimize import least_squares

    np = v.np
    scale = float(points.columns[0])
    x = points.columns.to_numpy(dtype=float) / scale
    y = points.to_numpy(dtype=float)

    def predict(params, size):
        ceiling, gap, exponent = params
        return ceiling - gap * size ** (-exponent)

    def fit(observed):
        result = least_squares(
            lambda params: predict(params, x) - observed,
            x0=(0.9, 0.3, 0.5), bounds=([0, 0, 0.001], [1, 1, 5]),
        )
        if not result.success:
            raise RuntimeError(result.message)
        return result.x

    params = fit(y.mean(axis=0))
    target_x = np.asarray(targets) / scale
    estimates = np.clip(predict(params, target_x), 0, 1)
    intervals = None
    # Resample whole seed trajectories to preserve correlation across nested sizes.
    if len(y) >= 3:
        rng = np.random.default_rng(seed)
        samples = []
        for _ in range(300):
            sample = y[rng.integers(len(y), size=len(y))].mean(axis=0)
            samples.append(np.clip(predict(fit(sample), target_x), 0, 1))
        intervals = np.percentile(samples, [2.5, 97.5], axis=0)
    return estimates, intervals, params


def reports(v, rows, completed, report, targets, fit_seed):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plots = report / "plots"
    plots.mkdir()
    flat = []
    for row in rows:
        item = {key: value for key, value in row.items()
                if key not in {"validation", "test", "checkpoint",
                               "training_class_counts", "class_weights",
                               "validation_class_counts", "test_class_counts"}}
        for field in ("training_class_counts", "class_weights", "validation_class_counts", "test_class_counts"):
            item.update({f"{field}_{label}": value for label, value in row[field].items()})
        for split in ("validation", "test"):
            metrics = row[split]
            for key, value in metrics.items():
                if isinstance(value, (float, int)):
                    item[f"{split}_{key}"] = value
            for label, scores in metrics["per_class"].items():
                item.update({f"{split}_{label}_{key}": value for key, value in scores.items()})
            labels = metrics["confusion_matrix_label_order"]
            for actual, counts in zip(labels, metrics["confusion_matrix"]):
                item.update({f"{split}_cm_{actual}_to_{predicted}": count
                             for predicted, count in zip(labels, counts)})
        flat.append(item)
    frame = v.pd.DataFrame(flat)
    frame.to_csv(report / "learning_curve_results.csv", index=False)
    metrics = {"test_accuracy": "Test accuracy", "test_macro_f1": "Test Macro-F1",
               "validation_accuracy": "Validation accuracy", "validation_macro_f1": "Validation Macro-F1"}
    combined, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    forecasts = []
    for ax, (metric, title) in zip(axes.flat, metrics.items()):
        single, single_ax = plt.subplots(figsize=(8, 5), layout="constrained")
        for model, group in frame.groupby("model", sort=False):
            points = group.pivot(index="seed", columns="training_songs", values=metric)
            means = points.mean()
            estimates = intervals = None
            if targets and metric.startswith("test_") and len(points.columns) >= 4:
                estimates, intervals, params = power_law(v, points, targets, fit_seed)
                for i, target in enumerate(targets):
                    forecasts.append({
                        "model": model, "metric": metric, "training_songs": target,
                        "extrapolated_score": estimates[i],
                        "seed_bootstrap_ci95_low": intervals[0, i] if intervals is not None else None,
                        "seed_bootstrap_ci95_high": intervals[1, i] if intervals is not None else None,
                        "fitted_ceiling": params[0], "fitted_gap": params[1], "fitted_exponent": params[2],
                        "size_scale": points.columns[0], "seeds": len(points),
                        "interpretation": "Exploratory; ceiling and long-range forecasts are weakly identified.",
                    })
            for panel in (ax, single_ax):
                line, = panel.plot(means.index, means, "o-", label=model)
                if len(points) > 1:
                    panel.errorbar(means.index, means, yerr=points.std(), fmt="none",
                                   color=line.get_color(), capsize=3)
                if estimates is not None:
                    panel.plot(targets, estimates, "x--", color=line.get_color(), alpha=0.75)
                    if intervals is not None:
                        panel.fill_between(targets, *intervals, color=line.get_color(), alpha=0.15)
        subtitle = "Dashed lines, where shown, are exploratory extrapolations"
        if frame["seed"].nunique() > 1:
            subtitle += " · bars = seed SD"
        for panel in (ax, single_ax):
            panel.set(title=f"{title}\n{subtitle}", xlabel="Training songs", ylabel=title,
                      ylim=(0, 1))
            panel.grid(alpha=0.2)
            panel.legend()
        single.savefig(plots / f"{metric}.png", dpi=160)
        plt.close(single)
    combined.savefig(plots / "all_models_comparison.png", dpi=160)
    plt.close(combined)
    if forecasts:
        v.pd.DataFrame(forecasts).to_csv(report / "power_law_extrapolations.csv", index=False)
    paired_comparisons(v, completed, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=DISPLAY)
    parser.add_argument("--sizes", nargs="+", metavar="N_OR_ALL")
    parser.add_argument("--seeds", nargs="+", type=int, help="Subset and model seeds (default: 67)")
    parser.add_argument("--validation", type=Path, help="Explicit path to the existing v26 validation CSV")
    parser.add_argument("--test", type=Path, help="Explicit path to the existing v26 test CSV")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments")
    parser.add_argument("--resume", type=Path, metavar="DIR")
    parser.add_argument("--plan", action="store_true", help="Validate and audit without training")
    parser.add_argument("--extrapolate", nargs="+", type=int, metavar="SONGS",
                        help="Fit a power law and predict at these sizes beyond the measured range")
    parser.add_argument("--require-verified-human", action="store_true",
                        help="Block runs whose human annotation provenance cannot be verified")
    args = parser.parse_args()
    import lyric_classifier_v26_all_models as v

    root = args.resume.resolve() if args.resume else None
    saved = read_json(root / "experiment.json") if root else None
    models = args.models or (saved["models"] if saved else list(v.ALL_NAMES))
    seeds = args.seeds or (saved["seeds"] if saved else [67])
    if len(set(models)) != len(models):
        parser.error("Models must be unique.")
    if len(set(seeds)) != len(seeds) or any(not 0 <= seed < 2**32 for seed in seeds):
        parser.error("Seeds must be unique and in the range 0 through 2**32-1.")
    paths = {"training": SOURCE}
    for name, filename in (("validation", v.VAL_CSV_FILE), ("test", v.TEST_CSV_FILE)):
        explicit = getattr(args, name)
        path = explicit or (Path(saved["datasets"][name]["path"]) if saved else ROOT / "data" / filename)
        if not path.is_file():
            parser.error(f"Cannot locate the fixed {name} CSV at {path}. Supply --{name} explicitly.")
        paths[name] = path.resolve()
    identities = {name: {"path": str(path), "sha256": v.file_sha256(path)}
                  for name, path in paths.items()}

    def verify_inputs():
        for name, path in paths.items():
            if identities[name]["sha256"] != v.file_sha256(path):
                raise ValueError(f"Dataset changed during the experiment: {path}")

    datasets = {name: prepare(v, path, name) for name, path in paths.items()}
    raw, train = datasets["training"]
    _, validation = datasets["validation"]
    _, test = datasets["test"]
    provenance = training_provenance(v, raw, train)
    print("Training annotation provenance:", json.dumps(provenance), flush=True)
    print("WARNING: Human annotation is unverified. Interpret this as a curve for the supplied labels.")
    model_source = any("pseudo" in source.lower() or "model" in source.lower()
                       for source in provenance["observed_label_sources"])
    if model_source or provenance["review_flagged_valid_rows"] or provenance["uses_predicted_label_fallback"]:
        print("WARNING: Training contains review-flagged rows or a model-label target column.")
    limitation = ("The fixed holdouts may have informed earlier thesis experiments. "
                  "Treat estimates as conditional on these datasets and prior model choices.")
    print(limitation)
    try:
        size_tokens = args.sizes or (saved["sizes"] if saved else SIZES)
        sizes = sorted({len(train) if str(size).upper() == "ALL" else int(size)
                        for size in size_tokens})
    except ValueError:
        parser.error("Sizes must be positive integers or ALL.")
    if any(size < 1 or size > len(train) for size in sizes):
        parser.error(f"Training sizes must be between 1 and {len(train)} valid songs.")
    targets = sorted(set(args.extrapolate if args.extrapolate is not None else
                         (saved["extrapolation_sizes"] if saved else [])))
    if targets and (len(sizes) < 4 or min(targets) <= max(sizes)):
        parser.error("Extrapolation needs at least four measured sizes and targets above the largest.")
    if targets:
        print("WARNING: Power-law forecasts are exploratory. Their asymptotes can be weakly "
              "identified, and a performance target above the fitted ceiling is unsupported.")
    orders = {seed: nested_order(v, train, seed) for seed in seeds}
    subsets = {(seed, size): sorted(orders[seed][:size]) for seed in seeds for size in sizes}
    spec = {
        "datasets": identities, "models": models, "sizes": sizes, "seeds": seeds,
        "v26_sha256": v.file_sha256(v.__file__), "runner_sha256": v.file_sha256(__file__),
        "extrapolation_sizes": saved["extrapolation_sizes"] if saved else targets,
        "training_provenance": provenance, "holdout_interpretation": limitation,
        "label_mapping": v.LABEL2ID, "device": str(v.DEVICE),
        "dependencies": {name: v.installed_version(name) for name in
                         ("torch", "transformers", "numpy", "pandas", "scikit-learn", "scipy", "matplotlib")},
    }
    if saved and saved != spec:
        raise ValueError("Current inputs, run selections, code, or environment differ from experiment.json.")
    for (seed, size), indices in subsets.items():
        counts = class_counts(v, train.loc[indices])
        if not all(counts.values()):
            raise ValueError(f"Missing training class for seed {seed}, size {size}: {counts}")
        print(f"Seed {seed} | {size:,} training songs | {counts}")
    if root is None:
        parent = args.output_dir.resolve()
        parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        root = Path(mkdtemp(prefix=f"learning_curve_v26_{stamp}_", dir=parent))
        write_json(root / "experiment.json", spec)
    print(f"Experiment directory: {root}", flush=True)
    audit_dir = Path(mkdtemp(prefix="preflight_", dir=root))
    audit_overlap(v, datasets, audit_dir / "leakage.json")
    verify_inputs()
    write_json(audit_dir / "passed.json", identities)
    if args.require_verified_human and not provenance["human_verification_confirmed"]:
        raise ValueError("Verified human annotation is required. No independent provenance was established.")

    manifests = root / "sampling_manifests"
    manifests.mkdir(exist_ok=True)
    for (seed, size), indices in subsets.items():
        manifest = manifests / f"seed_{seed}_{size}.json"
        sample = {"seed": seed, "training_songs": size, "source_row_indices": indices,
                  "source_sha256": identities["training"]["sha256"],
                  "class_counts": class_counts(v, train.loc[indices])}
        if manifest.exists():
            if read_json(manifest) != sample:
                raise ValueError(f"Sampling manifest differs: {manifest}")
        else:
            write_json(manifest, sample)
    if args.plan:
        print("Preflight passed. No training performed.")
        return
    if v.installed_version("matplotlib") is None:
        parser.error("Install matplotlib in the v26 Python environment.")
    session = Path(mkdtemp(prefix="session_", dir=root))
    results, completed = [], {}
    total = len(subsets) * len(models)
    for (seed, size), indices in subsets.items():
        seed_dir = root / "configurations" / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        subset_path = session / f"seed_{seed}_{size}_training.csv"
        group = completed[seed, size] = {}
        for name in models:
            prefix = f"{DISPLAY[name]}_{size}"
            directory = seed_dir / prefix
            group[name] = directory
            if directory.exists():
                print(f"Skipping completed seed {seed}, {prefix}", flush=True)
                record = read_json(directory / "metrics" / f"{prefix}_metrics.json")
            else:
                if not subset_path.exists():
                    raw.loc[indices].to_csv(subset_path, index=False)
                print(f"[{len(results) + 1}/{total}] Training seed {seed}, {prefix}", flush=True)
                verify_inputs()
                record = train_configuration(
                    v, name, seed, size, indices, (subset_path, paths["validation"], paths["test"]),
                    (train.loc[indices], validation, test), directory, verify_inputs,
                )
            results.append(record)
            print(f"Completed {len(results)}/{total}: seed {seed}, {prefix}", flush=True)
    report = Path(mkdtemp(prefix="report_", dir=root))
    reports(v, results, completed, report, targets, seeds[0])
    print(f"Reports: {report}")


if __name__ == "__main__":
    main()
