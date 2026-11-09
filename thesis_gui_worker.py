"""Process-isolated v26 operations for the desktop application."""
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import lyric_classifier_v26_all_models as clf


def report_progress(message):
    print(f"THESIS_PROGRESS {message}", flush=True)


def load_model(name, directory):
    checkpoint = directory / "best_model.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    manifest_path = directory / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("architecture") != name or manifest.get("pipeline_version") != "v26":
            raise ValueError(f"Select a v26 {name} run folder.")
        if manifest.get("smoke_test"):
            print("Using a smoke-test checkpoint.")
    tokenizer = clf.tokenizer_for(name, None, directory)
    model = clf.make_baseline(name, tokenizer if name == "lstm_cnn" else None)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.to(clf.DEVICE).eval(), tokenizer


def train_or_compare(job):
    clf.DATASET_OVERRIDES.update({
        name: path for name, path in zip(
            (clf.TRAIN_CSV_FILE, clf.VAL_CSV_FILE, clf.TEST_CSV_FILE), job["datasets"]
        ) if path
    })
    clf.RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    clf.EXPERIMENT_MODE = job["mode"]
    clf.SMOKE_TEST = job["smoke"]
    clf.FROZEN_TEST_EVALUATION = job["mode"] == "compare"
    clf.FROZEN_RUN_DIRS.update({name: path for name, path in job["runs"].items() if path})
    clf.main_v26()
    if clf.FROZEN_TEST_EVALUATION:
        folder = clf.PROJECT_DIR / "experiments" / f"comparison_v26_{clf.RUN_ID}"
        return {"message": str(folder), "tables": {
            "Held-out comparison": pd.read_csv(folder / "model_comparison.csv").to_dict("records")
        }}
    folder = clf.PROJECT_DIR / "experiments"
    rows = []
    for name in clf.ALL_NAMES:
        path = folder / f"{name}_v26_{clf.RUN_ID}" / "validation_metrics.json"
        if path.is_file():
            metrics = json.loads(path.read_text(encoding="utf-8"))
            rows.append({"Model": name, "Accuracy": metrics["accuracy"],
                         "Macro F1": metrics["macro_f1"], "Weighted F1": metrics["weighted_f1"]})
    return {"message": f"Run ID {clf.RUN_ID}\nResults in {folder}",
            "tables": {"Validation metrics": rows}}


def label(job, model, tokenizer):
    def predict(text):
        if not clf.clean_lyrics(text):
            return "", None, None, None
        target, confidence, index, total, _ = clf.predict_song_windows(text, model, tokenizer)
        return clf.ID2LABEL[target], confidence, index + 1, total

    columns = ["v26_label", "v26_evidence_probability", "v26_evidence_window", "v26_window_count"]
    if not job["source"]:
        return {"tables": {"Song label": [dict(zip(columns, predict(job["lyrics"])))]}}

    source = Path(job["source"])
    output = Path(job["output"])
    if source.resolve() == output.resolve():
        raise ValueError("Choose an output CSV different from the input CSV.")
    frame = pd.read_csv(source, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    lyric_column = job["column"] or next((c for c in ("lyrics", "text") if c in frame), "")
    if lyric_column not in frame:
        raise ValueError("Choose the lyrics column. Automatic detection accepts lyrics or text.")
    if any(column in frame for column in columns):
        raise ValueError("The input CSV already has v26 output columns.")
    predictions = [predict(text) for text in clf.tqdm(frame[lyric_column], desc="Labeling songs")]
    frame[columns] = pd.DataFrame(predictions, index=frame.index, columns=columns)
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    labels = frame["v26_label"].replace("", "Skipped empty lyrics")
    return {"message": str(output), "tables": {
        "Label counts": [{"Label": key, "Songs": int(value)} for key, value in labels.value_counts().items()]
    }}


def explain_song(text, model, tokenizer, method, rng, song_progress=""):
    text = clf.clean_lyrics(text)
    if not text:
        raise ValueError("Enter lyrics with at least one word.")
    song_class, _, index, total, _ = clf.predict_song_windows(text, model, tokenizer)
    ids = clf.full_token_ids(text, tokenizer)
    start, end = list(clf.window_spans(ids, tokenizer))[index]
    if clf.is_learned_tokenizer(tokenizer):
        words = tokenizer.tokens(text)[start:end]
    else:
        words = tokenizer.decode(ids[start:end], skip_special_tokens=True).split()
    capacity = clf.MAX_LENGTH - sum(map(len, clf.special_token_prefix_suffix(tokenizer)))
    while words and len(clf.full_token_ids(" ".join(words), tokenizer)) > capacity - 24:
        words.pop()
    if len(words) < 2:
        raise InsufficientFeatures("The selected evidence window needs at least two features for erasure.")

    @torch.inference_mode()
    def probabilities(texts):
        windows = []
        for value in texts:
            tokens = clf.full_token_ids(clf.clean_lyrics(value), tokenizer)[:capacity]
            windows.append(clf.make_window_ids(tokens, 0, len(tokens), tokenizer))
        outputs = []
        for offset in range(0, len(windows), clf.BATCH_SIZE):
            x, mask = clf.padded_window_batch(
                windows[offset:offset + clf.BATCH_SIZE], tokenizer, pad_to=clf.MAX_LENGTH
            )
            outputs.append(torch.softmax(model(x.to(clf.DEVICE), mask.to(clf.DEVICE)).float(), 1).cpu().numpy())
        return np.concatenate(outputs)

    evidence = " ".join(words)
    original_probs = probabilities([evidence])[0]
    target = int(original_probs.argmax())
    original = float(original_probs[target])
    stage = f"{method}{song_progress}"
    report_progress(f"{stage} · Computing contributions")
    if method == "LIME":
        explainer = clf.LimeTextExplainer(
            class_names=list(clf.ID2LABEL.values()), bow=False, split_expression=r"\s+",
            random_state=clf.RANDOM_SEED,
        )
        explanation = explainer.explain_instance(
            evidence, probabilities, labels=(target,), num_features=len(words), num_samples=300
        )
        indexed = explanation.domain_mapper.indexed_string
        if [indexed.word(i) for i in range(indexed.num_words())] != words:
            raise RuntimeError("LIME feature positions do not match the evidence window.")
        weights = dict(explanation.local_exp[target])
    else:
        # Each binary feature retains or deletes one position. A zero background
        # lets SHAP and the erasure test use the same feature boundaries.
        def score(masks):
            return probabilities([" ".join(word for word, keep in zip(words, mask) if keep)
                                  for mask in masks])[:, target]

        masker = clf.shap.maskers.Independent(np.zeros((1, len(words))))
        explainer = clf.shap.Explainer(score, masker, algorithm="permutation", seed=clf.RANDOM_SEED)
        explanation = explainer(np.ones((1, len(words))), max_evals=2 * (2 * len(words) + 1),
                                batch_size=16, silent=True)
        weights = dict(enumerate(explanation.values[0]))

    ranking = sorted(range(len(words)), key=lambda i: (-weights.get(i, 0.0), i))
    attribution = [{"Position": i + 1, "Feature": words[i], "Contribution": float(weights.get(i, 0.0))}
                   for i in sorted(ranking, key=lambda i: -abs(weights.get(i, 0.0)))]
    erasure = []
    for percent in (5, 10, 20):
        report_progress(f"{stage} · Testing {percent}% erasure")
        count = min(len(words) - 1, math.ceil(len(words) * percent / 100))
        selected = set(ranking[:count])
        deletions = [selected] + [set(rng.sample(range(len(words)), count)) for _ in range(3)]
        variants = [" ".join(word for i, word in enumerate(words) if i not in removed)
                    for removed in deletions]
        scores = probabilities(variants)[:, target]
        erasure.append({
            "Erasure %": percent, "Removed": count, "Original probability": original,
            "Post-erasure probability": float(scores[0]), "Comprehensiveness": original - float(scores[0]),
            "Random mean": original - float(scores[1:].mean()),
            "Erased features": " | ".join(words[i] for i in sorted(selected)),
        })
    message = (f"{method} | Song label {clf.ID2LABEL[song_class]} | Evidence window {index + 1}/{total}\n"
               f"Local target {clf.ID2LABEL[target]} ({original:.1%})\n\n{evidence}")
    return message, attribution, erasure


def explain_method(job, model, tokenizer, method):
    rng = random.Random(clf.RANDOM_SEED)
    if not job["validation"]:
        message, attribution, erasure = explain_song(job["lyrics"], model, tokenizer, method, rng)
        return {"message": message, "tables": {"Contributions": attribution, "Erasure": erasure},
                "highlights": {method: sorted(attribution, key=lambda row: row["Position"])}}

    from sklearn.model_selection import train_test_split

    path = job["validation"]
    if path == "default":
        path = clf.resolve_dataset_path(clf.VAL_CSV_FILE)
    frame = clf.normalize_dataset_columns(pd.read_csv(path), "Validation dataset")
    frame["text"] = frame["text"].map(clf.clean_lyrics)
    frame["explicitness"] = frame["explicitness"].astype(str).str.strip().str.lower().map(clf.LABEL_NORMALIZATION)
    frame = frame.loc[frame["text"].ne("") & frame["explicitness"].notna()]
    count = job["samples"]
    if 0 < count < len(frame):
        frame, _ = train_test_split(frame, train_size=count, stratify=frame["explicitness"],
                                    random_state=clf.RANDOM_SEED)
        frame = frame.sort_index()
    details = []
    skipped = 0
    for index, row in enumerate(frame.itertuples(), start=1):
        try:
            _, _, erasure = explain_song(
                row.text, model, tokenizer, method, rng, f" · Song {index}/{len(frame)}"
            )
        except InsufficientFeatures:
            skipped += 1
            continue
        details.extend({"CSV data row": row.Index + 1, "Song": row.song_name, "Artist": row.artist,
                        "Actual label": row.explicitness, **record} for record in erasure)
    if not details:
        raise ValueError("No eligible validation songs.")
    detail_frame = pd.DataFrame(details)
    summary = detail_frame.groupby("Erasure %").agg(
        Songs=("Comprehensiveness", "size"), Original=("Original probability", "mean"),
        Post_erasure=("Post-erasure probability", "mean"),
        Comprehensiveness=("Comprehensiveness", "mean"), Random_mean=("Random mean", "mean"),
    ).reset_index()
    folder = clf.PROJECT_DIR / "eraser_results" / f"{method.lower()}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
    folder.mkdir(parents=True)
    detail_frame.to_csv(folder / "eraser_per_song.csv", index=False)
    summary.to_csv(folder / "summary.csv", index=False)
    return {"message": f"{method} local evidence-window erasure | Skipped {skipped} short songs\n{folder}",
            "tables": {"Erasure summary": summary.to_dict("records")}}


def explain(job, model, tokenizer):
    if job["method"] != "Both":
        return explain_method(job, model, tokenizer, job["method"])
    combined = {"message": "", "tables": {}, "highlights": {}}
    messages = []
    for method in ("LIME", "SHAP"):
        result = explain_method(job, model, tokenizer, method)
        messages.append(result["message"])
        combined["tables"].update({f"{method} {name}": rows for name, rows in result["tables"].items()})
        combined["highlights"].update(result.get("highlights", {}))
    combined["message"] = "\n\n".join(messages)
    return combined


class InsufficientFeatures(ValueError):
    pass


def main():
    request, result = map(Path, sys.argv[1:])
    job = json.loads(request.read_text(encoding="utf-8"))
    print(f"v26 device: {clf.DEVICE}", flush=True)
    if job["action"] == "train":
        output = train_or_compare(job)
    else:
        model, tokenizer = load_model(job["model"], Path(job["run"]))
        operation = label if job["action"] == "label" else explain
        output = operation(job, model, tokenizer)
    result.write_text(json.dumps(output, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
