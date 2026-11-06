"""One-shot, full-lyrics DistilBERT teacher: 6k -> 34k -> Training.csv.

Inputs (either in a data/ subdirectory beside this script OR beside the script):
  labeled.csv                    6,000 annotated songs, including safety_label
  merged_excluding_labeled.csv  34,000 different songs to predict

Run: python train_and_label_once_chunking_fixed_v2.py

Lyrics longer than 512 tokens are split into overlapping <=512-token windows,
not truncated. The complete song is covered during teacher training, validation,
testing, and inference. Each song receives one label via MAX-SEVERITY pooling:
if ANY window is predicted explicit, the song is explicit; otherwise mild if
any window is mild; otherwise safe. This conservative approach improves recall
of end-of-song explicit material but may increase false positives.

All windows from the same training song inherit the song-level training label,
which introduces weak-label noise when only one verse is explicit. Windows are
weighted so long songs do not automatically count more than short songs.
Validation and test metrics, and early stopping, are measured per SONG.

One stratified split of labeled.csv: 5,000 train / 500 validation / 500 test.
Fine-tune the teacher only on the 5,000 training rows. Evaluate the model,
then assign a model label to EVERY row in merged_excluding_labeled.csv.

Final output: Training.csv, 40,000 unique songs in total, preserving all 6k
original labels plus all 34k predictions. Each row has dataset_split and
label_source to distinguish provided labels from model predictions. The
validation/test rows are included in the comprehensive Training.csv export,
but are NOT part of the teacher's training or its separate training-only export.
No iterative retraining, plateau detection, or human feedback loop.
"""

from pathlib import Path
import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

# =====================================================================
# CONFIGURATION -- change these only if your input filenames differ.
# =====================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
# Prefer data/ if it contains the labeled input. Otherwise use the script folder.
DATA_DIR = SCRIPT_DIR / "data" if (SCRIPT_DIR / "data" / "labeled.csv").is_file() else SCRIPT_DIR
LABELED_FILE = DATA_DIR / "labeled.csv"
UNLABELED_FILE = DATA_DIR / "merged_excluding_labeled.csv"
RESULTS_DIR = DATA_DIR / "one_pass_results"
FINAL_EXPORT = DATA_DIR / "Training.csv"

EXPECTED_LABELED = 6_000
EXPECTED_UNLABELED = 34_000
TRAIN_SIZE = 5_000
VALIDATION_SIZE = 500
TEST_SIZE = 500

MODEL_NAME = "distilbert-base-uncased"
LABELS = ["safe", "mild", "explicit"]
LABEL2ID = {name: index for index, name in enumerate(LABELS)}
ID2LABEL = {index: name for name, index in LABEL2ID.items()}
SEED = 42
MAX_LENGTH = 512                      # includes [CLS] and [SEP]
CHUNK_OVERLAP_TOKENS = 64           # adjacent windows overlap
INFERENCE_SONG_BATCH_SIZE = 128     # process entire candidate pool in bounded groups
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
MAX_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 2
MIN_EARLY_STOP_IMPROVEMENT = 0.001
LEARNING_RATE = 2e-5
USE_CLASS_WEIGHTS = True
MAX_CLASS_WEIGHT = 3.0

# For flagging uncertain *predictions*, not for suppressing labels.
THRESHOLD_CANDIDATES = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
TARGET_ACCEPTED_ACCURACY = 0.90
MIN_ACCEPTED_SAMPLES = 25
MIN_PREDICTIONS_PER_CLASS = 3
FALLBACK_ACCURACY_TOLERANCE = 0.02

BASIC_COLUMNS = ["song_title", "artist", "lyrics"]
LABELED_COLUMNS = BASIC_COLUMNS + ["safety_label"]


def load_csv(path: Path, labeled: bool, expected_rows: int) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing input: {path}")
    df = pd.read_csv(path, low_memory=False, keep_default_na=False, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    # Support previously used raw-song column names.
    for canonical, alternatives in {
        "song_title": ["song"],
        "artist": ["Artist(s)"],
        "lyrics": ["text"],
    }.items():
        if canonical not in df.columns:
            for alternative in alternatives:
                if alternative in df.columns:
                    df = df.rename(columns={alternative: canonical})
                    break

    required = LABELED_COLUMNS if labeled else BASIC_COLUMNS
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    if len(df) != expected_rows:
        raise ValueError(
            f"{path.name}: expected {expected_rows:,} rows, found {len(df):,}. "
            "Check your inputs before training."
        )

    # Preserve original supplementary columns for traceability. In the 34k
    # candidate pool, any previous provisional annotations are NOT used for
    # learning and will NOT override the newly predicted safety_label.
    for column in required:
        df[column] = df[column].astype(str).str.strip()
        if df[column].eq("").any():
            raise ValueError(f"{path.name}: blank {column} values")

    if labeled:
        df["safety_label"] = df["safety_label"].str.casefold()
        invalid = set(df["safety_label"]) - set(LABELS)
        if invalid:
            raise ValueError(f"{path.name}: invalid safety_label values: {sorted(invalid)}")
        if "label_source" not in df.columns:
            df["label_source"] = "provided_annotation"
        else:
            df["label_source"] = df["label_source"].astype(str).str.strip()
            df.loc[df["label_source"] == "", "label_source"] = "provided_annotation"
    else:
        # The previous merged file includes 1,272 provisional labels. Keep
        # these as prior_safety_label, but always get new teacher predictions.
        if "safety_label" in df.columns:
            df = df.rename(columns={"safety_label": "prior_safety_label"})
            df["prior_safety_label"] = df["prior_safety_label"].astype(str).str.strip().str.casefold()
        if "justification" in df.columns:
            # Old justifications may not apply to the fresh model prediction.
            df = df.rename(columns={"justification": "prior_justification"})
        if "annotation_status" in df.columns:
            df = df.rename(columns={"annotation_status": "prior_annotation_status"})
        if "annotation_method" in df.columns:
            df = df.rename(columns={"annotation_method": "prior_annotation_method"})
    return df.reset_index(drop=True)


def song_keys(df: pd.DataFrame) -> pd.Series:
    return df["song_title"].str.casefold() + "|||" + df["artist"].str.casefold()


def validate_unique_and_disjoint(labeled: pd.DataFrame, unlabeled: pd.DataFrame) -> None:
    labeled_keys = song_keys(labeled)
    unlabeled_keys = song_keys(unlabeled)
    for name, keys, df in [
        (LABELED_FILE.name, labeled_keys, labeled),
        (UNLABELED_FILE.name, unlabeled_keys, unlabeled),
    ]:
        duplicate = keys.duplicated(keep=False)
        if duplicate.any():
            examples = df.loc[duplicate, ["song_title", "artist"]].head(8)
            raise ValueError(
                f"{name} has {int(duplicate.sum()):,} rows with duplicate title+artist. "
                "Resolve them without silently reducing the expected song count:\n"
                + examples.to_string(index=False)
            )
    overlap = set(labeled_keys) & set(unlabeled_keys)
    if overlap:
        examples = labeled.loc[labeled_keys.isin(overlap), ["song_title", "artist"]].head(8)
        raise ValueError(
            f"The two input files overlap on {len(overlap):,} song identities. "
            "The 34,000-song pool must be separate from the 6,000 labeled songs:\n"
            + examples.to_string(index=False)
        )


def make_splits(labeled: pd.DataFrame):
    train, holdout = train_test_split(
        labeled,
        train_size=TRAIN_SIZE,
        test_size=VALIDATION_SIZE + TEST_SIZE,
        stratify=labeled["safety_label"],
        random_state=SEED,
    )
    validation, test = train_test_split(
        holdout,
        train_size=VALIDATION_SIZE,
        test_size=TEST_SIZE,
        stratify=holdout["safety_label"],
        random_state=SEED,
    )
    result = []
    for name, split_df in [("train", train), ("validation", validation), ("test", test)]:
        split_df = split_df.copy().reset_index(drop=True)
        if set(split_df["safety_label"]) != set(LABELS):
            raise ValueError(f"{name} set lacks one or more classes; inspect the class counts")
        split_df["dataset_split"] = name
        result.append(split_df)
    return tuple(result)


class SongChunkCollator:
    """Pad DistilBERT windows and preserve their optional per-song weights.

    This intentionally does not call tokenizer.pad(): the custom padding is
    compatible with both older and newer Transformers tokenizer backends.
    """

    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id
        self.padding_side = getattr(tokenizer, "padding_side", "right")
        if self.pad_token_id is None:
            raise ValueError("Tokenizer is missing the padding token required for batching")
        if self.padding_side not in ("left", "right"):
            raise ValueError(f"Unsupported padding side: {self.padding_side}")

    def __call__(self, features):
        if not features:
            raise ValueError("Cannot collate an empty batch")
        size = max(len(item["input_ids"]) for item in features)
        input_ids, attention_mask = [], []
        for item in features:
            ids = list(item["input_ids"])
            mask = list(item["attention_mask"])
            if len(ids) != len(mask) or len(ids) > MAX_LENGTH:
                raise ValueError("Invalid token or attention-mask length")
            pad = size - len(ids)
            if self.padding_side == "right":
                input_ids.append(ids + [self.pad_token_id] * pad)
                attention_mask.append(mask + [0] * pad)
            else:
                input_ids.append([self.pad_token_id] * pad + ids)
                attention_mask.append([0] * pad + mask)
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
        has_labels = ["labels" in item for item in features]
        has_weights = ["sample_weight" in item for item in features]
        if any(has_labels) and not all(has_labels):
            raise ValueError("Some batch elements contain labels and others do not")
        if any(has_weights) and not all(has_weights):
            raise ValueError("Some batch elements contain sample weights and others do not")
        if all(has_labels):
            batch["labels"] = torch.tensor([item["labels"] for item in features], dtype=torch.long)
        if all(has_weights):
            batch["sample_weight"] = torch.tensor(
                [item["sample_weight"] for item in features], dtype=torch.float32
            )
        return batch


class WeightedTrainer(Trainer):
    """Class-weighted loss with inverse-chunk-count per-song weighting."""

    # The custom loss is a mean over the current batch, not already scaled
    # over gradient-accumulation steps (Transformers 5.19+ honors this flag).
    loss_is_scaled_for_ga = False

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        # Required by older Trainer versions when overriding compute_loss with
        # a batch-mean loss; avoids incorrect accumulation normalization.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        per_chunk_weight = inputs.pop("sample_weight", None)
        targets = inputs.pop("labels")
        # DistilBERT does not need the labels for its forward pass because we
        # compute the (class- and song-weighted) cross entropy ourselves.
        outputs = model(**inputs)
        logits = outputs.logits
        per_chunk_loss = nn.CrossEntropyLoss(
            weight=(self.class_weights.to(logits.device)
                    if self.class_weights is not None else None),
            reduction="none",
        )(logits.reshape(-1, model.config.num_labels), targets.reshape(-1))
        if per_chunk_weight is not None:
            per_chunk_loss = per_chunk_loss * per_chunk_weight.to(
                device=logits.device, dtype=per_chunk_loss.dtype
            ).reshape(-1)
        loss = per_chunk_loss.mean()
        return (loss, outputs) if return_outputs else loss


def get_class_weights(train_df):
    counts = train_df["safety_label"].value_counts().reindex(LABELS, fill_value=0)
    raw = np.sqrt(len(train_df) / (len(LABELS) * counts.to_numpy(dtype=float)))
    raw = np.clip(raw, 1.0 / MAX_CLASS_WEIGHT, MAX_CLASS_WEIGHT)
    normalized = raw / raw.mean()
    final_weights = np.clip(normalized, 1.0 / MAX_CLASS_WEIGHT, MAX_CLASS_WEIGHT)
    print("Class counts:", counts.to_dict())
    print("Class weights:", dict(zip(LABELS, np.round(final_weights, 4).tolist())))
    return torch.tensor(final_weights, dtype=torch.float32) if USE_CLASS_WEIGHTS else None


def probabilities_from_logits(logits):
    if isinstance(logits, tuple):
        logits = logits[0]
    values = np.asarray(logits, dtype=np.float64)
    scaled = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(scaled)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def choose_review_threshold(validation_probs, validation_truth):
    pred = validation_probs.argmax(axis=1)
    confidence = validation_probs.max(axis=1)
    correct = pred == np.asarray(validation_truth)
    diagnostics = []
    for threshold in THRESHOLD_CANDIDATES:
        accepted = confidence >= threshold
        count = int(accepted.sum())
        accepted_accuracy = float(correct[accepted].mean()) if count else None
        class_counts = np.bincount(pred[accepted], minlength=len(LABELS)).tolist()
        eligible_count = count >= MIN_ACCEPTED_SAMPLES and min(class_counts) >= MIN_PREDICTIONS_PER_CLASS
        diagnostics.append({
            "threshold": threshold,
            "accepted_count": count,
            "coverage": float(accepted.mean()),
            "accepted_accuracy": accepted_accuracy,
            "accepted_by_predicted_class": dict(zip(LABELS, class_counts)),
            "eligible_count": eligible_count,
        })
    qualifying = [
        row for row in diagnostics
        if row["eligible_count"] and row["accepted_accuracy"] >= TARGET_ACCEPTED_ACCURACY
    ]
    if qualifying:
        selected = qualifying[0]["threshold"]
        reason = "lowest tested threshold meeting 90% observed accepted accuracy"
    else:
        viable = [row for row in diagnostics if row["eligible_count"]]
        if viable:
            best = max(row["accepted_accuracy"] for row in viable)
            selected = next(
                row["threshold"] for row in viable
                if row["accepted_accuracy"] >= best - FALLBACK_ACCURACY_TOLERANCE
            )
            reason = "fallback: near highest observed accepted accuracy (90% not demonstrated)"
        else:
            selected = 0.90
            reason = "conservative fallback: validation predictions insufficient to choose threshold"
    return float(selected), reason, diagnostics


def chunk_token_ids(token_ids, tokenizer):
    """Yield overlapping chunks covering EVERY token, with special tokens added."""
    # DistilBERT's single-sequence BERT tokenizer uses [CLS] + text + [SEP].
    # New Transformers tokenizers may not expose build_inputs_with_special_tokens.
    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise ValueError("Expected DistilBERT [CLS] and [SEP] token IDs")
    special_count = 2
    content_capacity = MAX_LENGTH - special_count
    if content_capacity <= 0:
        raise ValueError("MAX_LENGTH is too small to hold tokenizer special tokens")
    step = content_capacity - CHUNK_OVERLAP_TOKENS
    if step <= 0:
        raise ValueError("CHUNK_OVERLAP_TOKENS must be less than usable window length")

    # Even empty-token texts produce one [CLS]...[SEP] window.
    for start in range(0, max(len(token_ids), 1), step):
        piece = token_ids[start:start + content_capacity]
        ids = [tokenizer.cls_token_id, *piece, tokenizer.sep_token_id]
        if len(ids) > MAX_LENGTH:
            raise AssertionError("A constructed window exceeded MAX_LENGTH")
        yield {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if start + content_capacity >= len(token_ids):
            break


def preflight_chunking_and_collator(tokenizer):
    """Verify overlap, tail coverage, label handling, and sample weights."""
    fake_ids = list(range(1500))
    windows = list(chunk_token_ids(fake_ids, tokenizer))
    assert all(len(window["input_ids"]) <= MAX_LENGTH for window in windows)
    covered = set()
    for window in windows:
        ids = window["input_ids"]
        assert ids[0] == tokenizer.cls_token_id and ids[-1] == tokenizer.sep_token_id
        covered.update(ids[1:-1])
    if covered != set(fake_ids):
        raise RuntimeError("Preflight failed: token windows did not cover the whole song")
    collator = SongChunkCollator(tokenizer)
    a = {**windows[0], "labels": 2, "sample_weight": 0.5}
    b = {**windows[-1], "labels": 0, "sample_weight": 1.5}
    batch = collator([a, b])
    if batch["input_ids"].shape[0] != 2 or batch["labels"].tolist() != [2, 0]:
        raise RuntimeError("Preflight failed: label padding or batch shape")
    if batch["sample_weight"].tolist() != [0.5, 1.5]:
        raise RuntimeError("Preflight failed: song weights were lost")
    print("Preflight OK: every lyric token covered, batches padded, weights retained.")


def make_chunk_dataset(frame, tokenizer, include_labels=True, balance_training_songs=False):
    """Return (HuggingFace Dataset, chunk->song map, per-song window counts).

    Important: song IDs are kept OUTSIDE model features. Validation metrics
    reconstruct predictions with the map; the model only sees token windows.
    """
    examples = []
    chunk_song_ids = []
    counts = []
    for song_id, row in enumerate(frame.itertuples(index=False)):
        # Explicitly no truncation. Suppress the tokenizer's >512 warning,
        # since we deliberately split the long token sequence afterward.
        encoded = tokenizer(
            row.lyrics, add_special_tokens=False, truncation=False,
            return_attention_mask=False, return_token_type_ids=False,
            verbose=False,
        )["input_ids"]
        windows = list(chunk_token_ids(encoded, tokenizer))
        counts.append(len(windows))
        for window in windows:
            if include_labels:
                window["labels"] = LABEL2ID[row.safety_label]
            examples.append(window)
            chunk_song_ids.append(song_id)

    if balance_training_songs:
        # Each song's total contribution (summed across its windows) is equal;
        # scale weights to global mean 1 to maintain the loss magnitude.
        average_windows = len(examples) / len(frame)
        for example, song_id in zip(examples, chunk_song_ids):
            example["sample_weight"] = average_windows / counts[song_id]

    return (Dataset.from_list(examples),
            np.asarray(chunk_song_ids, dtype=np.int32),
            np.asarray(counts, dtype=np.int32))


def aggregate_song_predictions(chunk_logits, chunk_song_ids, song_count):
    """Pool a model's chunk predictions into exactly one prediction per song.

    Policy: use the highest severity among chunk ARGMAX classes; among those
    windows pick the most confident window as supporting evidence. prob_* and
    confidence belong to THAT window (not averages across unrelated windows).
    No tokens are discarded, and the final window has genuine song-tail text.
    """
    probs = probabilities_from_logits(chunk_logits)
    chunk_song_ids = np.asarray(chunk_song_ids, dtype=np.int32)
    if len(probs) != len(chunk_song_ids):
        raise ValueError("Chunk predictions/map length mismatch")
    if song_count <= 0:
        raise ValueError("Expected at least one song")
    selected_window = np.full(song_count, -1, dtype=np.int32)
    selected_severity = np.full(song_count, -1, dtype=np.int32)
    selected_conf = np.full(song_count, -1.0, dtype=np.float64)
    window_counts = np.zeros(song_count, dtype=np.int32)
    chunk_positions = np.zeros(song_count, dtype=np.int32)
    selected_positions = np.zeros(song_count, dtype=np.int32)
    class_masks = np.zeros(song_count, dtype=np.int32)
    chunk_pred = probs.argmax(axis=1)

    for window_idx, song_id in enumerate(chunk_song_ids):
        if not 0 <= song_id < song_count:
            raise ValueError("Chunk references an out-of-range song index")
        window_counts[song_id] += 1
        chunk_positions[song_id] += 1
        severity = int(chunk_pred[window_idx])
        class_masks[song_id] |= 1 << severity
        conf = float(probs[window_idx, severity])
        if severity > selected_severity[song_id] or (
            severity == selected_severity[song_id] and conf > selected_conf[song_id]
        ):
            selected_window[song_id] = window_idx
            selected_severity[song_id] = severity
            selected_conf[song_id] = conf
            selected_positions[song_id] = chunk_positions[song_id]

    if np.any(selected_window == -1):
        raise ValueError("At least one song did not receive any window prediction")
    return {
        "predicted_ids": selected_severity,
        "probs": probs[selected_window],
        "confidence": selected_conf,
        "num_windows": window_counts,
        "evidence_window": selected_positions,  # 1-based for easier inspection
        "window_disagreement": np.asarray([
            mask.bit_count() > 1 for mask in class_masks
        ], dtype=bool),
    }


def detailed_evaluation(song_predictions, song_truth):
    predicted = song_predictions["predicted_ids"]
    truth = np.asarray(song_truth)
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=[0, 1, 2], average="macro", zero_division=0)),
        "classification_report": classification_report(
            truth, predicted, labels=[0, 1, 2], target_names=LABELS,
            zero_division=0, output_dict=True
        ),
        "confusion_matrix_true_rows": confusion_matrix(truth, predicted, labels=[0, 1, 2]).tolist(),
    }


def write_csv(df, name):
    destination = RESULTS_DIR / name
    df.to_csv(destination, index=False, encoding="utf-8-sig")
    print(f"Saved {destination} ({len(df):,} songs)")


def main():
    set_seed(SEED)
    if EXPECTED_LABELED != TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE:
        raise ValueError("The training/validation/test sizes must sum to the labeled dataset size")
    if FINAL_EXPORT.exists():
        raise FileExistsError(
            f"Output already exists: {FINAL_EXPORT}. Rename/archive it before "
            "deliberately running another training experiment."
        )

    print("Loading and validating inputs BEFORE model training...")
    labeled = load_csv(LABELED_FILE, labeled=True, expected_rows=EXPECTED_LABELED)
    unlabeled = load_csv(UNLABELED_FILE, labeled=False, expected_rows=EXPECTED_UNLABELED)
    validate_unique_and_disjoint(labeled, unlabeled)
    train_df, validation_df, test_df = make_splits(labeled)
    print("Split sizes:", {"train": len(train_df), "validation": len(validation_df), "test": len(test_df)})
    for name, split_df in [
        ("Training", train_df), ("Validation", validation_df), ("Test", test_df)
    ]:
        print(f"{name} class distribution: {split_df['safety_label'].value_counts().to_dict()}")

    if not torch.cuda.is_available():
        print("WARNING: CUDA GPU unavailable; fine-tuning on the CPU will be much slower.")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in [
        ("train_5000.csv", train_df),
        ("validation_500.csv", validation_df),
        ("test_500.csv", test_df),
    ]:
        write_csv(df, name)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Tokenizer: {type(tokenizer).__name__}; "
          f"special token IDs: CLS={tokenizer.cls_token_id}, SEP={tokenizer.sep_token_id}")
    print(f"Full-lyrics windowing: {MAX_LENGTH} tokens max incl. special tokens; "
          f"{CHUNK_OVERLAP_TOKENS} content tokens overlap.")
    preflight_chunking_and_collator(tokenizer)
    train_dataset, train_chunk_song_ids, train_counts = make_chunk_dataset(
        train_df, tokenizer, include_labels=True, balance_training_songs=True
    )
    validation_dataset, val_chunk_song_ids, val_counts = make_chunk_dataset(
        validation_df, tokenizer, include_labels=True
    )
    test_dataset, test_chunk_song_ids, test_counts = make_chunk_dataset(
        test_df, tokenizer, include_labels=True
    )
    print(f"Windows: train={len(train_dataset):,}, validation={len(validation_dataset):,}, "
          f"test={len(test_dataset):,}")
    print(f"Songs >1 window: train={int((train_counts > 1).sum()):,}; "
          f"validation={int((val_counts > 1).sum()):,}; "
          f"test={int((test_counts > 1).sum()):,}")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    weights = get_class_weights(train_df)  # Use SONG counts, not window counts.
    train_steps_per_epoch = math.ceil(len(train_dataset) / TRAIN_BATCH_SIZE)
    validation_song_truth = validation_df["safety_label"].map(LABEL2ID).to_numpy()
    test_song_truth = test_df["safety_label"].map(LABEL2ID).to_numpy()

    def validation_song_metrics(eval_prediction):
        # This function runs during each epoch's validation. Early stopping
        # therefore selects the teacher by SONG-level macro-F1, not window-F1.
        pooled = aggregate_song_predictions(
            eval_prediction.predictions, val_chunk_song_ids, len(validation_df)
        )
        pred = pooled["predicted_ids"]
        return {
            "accuracy": float(accuracy_score(validation_song_truth, pred)),
            "macro_f1": float(f1_score(
                validation_song_truth, pred, labels=[0, 1, 2],
                average="macro", zero_division=0,
            )),
        }

    args = TrainingArguments(
        output_dir=RESULTS_DIR / "checkpoints",
        seed=SEED,
        data_seed=SEED,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=MAX_EPOCHS,
        weight_decay=0.01,
        warmup_steps=max(1, int(0.1 * train_steps_per_epoch * MAX_EPOCHS)),
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        bf16=use_bf16,
        fp16=use_fp16,
        save_total_limit=2,
        eval_accumulation_steps=16,  # move evaluation logits to CPU regularly
        logging_steps=50,
        report_to="none",
        # Keep custom sample_weight available to the collator.
        remove_unused_columns=False,
    )
    trainer = WeightedTrainer(
        model=model,
        args=args,
        class_weights=weights,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=SongChunkCollator(tokenizer),
        processing_class=tokenizer,
        compute_metrics=validation_song_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            early_stopping_threshold=MIN_EARLY_STOP_IMPROVEMENT,
        )],
    )
    print("\nTRAIN MODEL ONCE (epochs may stop early; there is no retraining loop).")
    trainer.train()
    final_model_dir = RESULTS_DIR / "distilbert_teacher"
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    # The training-time metric is tied to val_chunk_song_ids. Disable it for
    # explicit prediction calls and calculate song-level metrics ourselves.
    trainer.compute_metrics = None
    print("\nEvaluating validation songs (pooled predictions from ALL windows)...")
    validation_prediction = trainer.predict(validation_dataset)
    validation_song = aggregate_song_predictions(
        validation_prediction.predictions, val_chunk_song_ids, len(validation_df)
    )
    validation_metrics = detailed_evaluation(validation_song, validation_song_truth)
    threshold, threshold_reason, threshold_analysis = choose_review_threshold(
        validation_song["probs"], validation_song_truth
    )
    print("Validation song accuracy:", round(validation_metrics["accuracy"], 4))
    print("Validation song macro-F1:", round(validation_metrics["macro_f1"], 4))
    print(f"Review threshold: {threshold:.2f} ({threshold_reason})")

    # The held-out set is used for reporting ONLY, not model selection.
    print("\nEvaluating held-out test songs ONCE (song-level pooling)...")
    test_prediction = trainer.predict(test_dataset)
    test_song = aggregate_song_predictions(
        test_prediction.predictions, test_chunk_song_ids, len(test_df)
    )
    test_metrics = detailed_evaluation(test_song, test_song_truth)
    print("Test song accuracy:", round(test_metrics["accuracy"], 4))
    print("Test song macro-F1:", round(test_metrics["macro_f1"], 4))
    print("Test confusion matrix (true rows / predicted columns):")
    print(pd.DataFrame(test_metrics["confusion_matrix_true_rows"], index=LABELS, columns=LABELS))

    print("\nLabeling ALL 34,000 remaining songs with full-lyrics windowing...")
    # Build one small group of songs at a time: avoids retaining millions of
    # token IDs and every window's logits for the full 34k in RAM simultaneously.
    pooled_parts = []
    for start in range(0, len(unlabeled), INFERENCE_SONG_BATCH_SIZE):
        stop = min(len(unlabeled), start + INFERENCE_SONG_BATCH_SIZE)
        subset = unlabeled.iloc[start:stop].reset_index(drop=True)
        candidate_dataset, cand_map, cand_counts = make_chunk_dataset(
            subset, tokenizer, include_labels=False
        )
        output = trainer.predict(candidate_dataset)
        pooled_parts.append(aggregate_song_predictions(
            output.predictions, cand_map, len(subset)
        ))
        if stop % (INFERENCE_SONG_BATCH_SIZE * 10) == 0 or stop == len(unlabeled):
            print(f"Processed {stop:,}/{len(unlabeled):,} songs", flush=True)

    pooled = {
        name: np.concatenate([part[name] for part in pooled_parts], axis=0)
        for name in pooled_parts[0]
    }
    pseudo = unlabeled.copy()
    pseudo["safety_label"] = [ID2LABEL[int(i)] for i in pooled["predicted_ids"]]
    pseudo["predicted_label"] = pseudo["safety_label"]
    pseudo["confidence"] = pooled["confidence"]
    for class_index, class_name in enumerate(LABELS):
        pseudo[f"prob_{class_name}"] = pooled["probs"][:, class_index]
    pseudo["num_windows"] = pooled["num_windows"]
    pseudo["evidence_window"] = pooled["evidence_window"]
    pseudo["window_disagreement"] = pooled["window_disagreement"]
    # Both low-confidence and contradictory window-level predictions deserve
    # review, but ALL candidates retain a predicted safety_label either way.
    pseudo["needs_review"] = (
        (pseudo["confidence"] < threshold) | pseudo["window_disagreement"]
    )
    pseudo["label_source"] = "pseudo_label"
    pseudo["dataset_split"] = "pseudo_labeled"
    assert len(pseudo) == EXPECTED_UNLABELED

    # Only the initial 5,000 songs were used to fit the teacher. Keep the
    # 500 validation + 500 test records OUT of the training-only export.
    training_pool = pd.concat([train_df, pseudo], ignore_index=True, sort=False)

    # Full requested export: ALL 6,000 original labels plus 34,000 predictions.
    # Retain split/source columns to prevent mistaking the held-out records
    # for fitting examples or the pseudo-labels for verified human labels.
    complete = pd.concat([train_df, validation_df, test_df, pseudo], ignore_index=True, sort=False)
    expected_total = EXPECTED_LABELED + EXPECTED_UNLABELED
    if len(training_pool) != TRAIN_SIZE + EXPECTED_UNLABELED or len(complete) != expected_total:
        raise RuntimeError("Final output row count mismatch; refusing to export")
    if song_keys(complete).nunique() != len(complete):
        raise RuntimeError("Final merged dataset contains duplicate song identities")
    if complete["safety_label"].isna().any() or not complete["safety_label"].isin(LABELS).all():
        raise RuntimeError("Final export contains missing/invalid safety_label values")
    if complete["dataset_split"].value_counts().to_dict() != {
        "train": 5000, "validation": 500, "test": 500, "pseudo_labeled": 34000,
    }:
        raise RuntimeError("Unexpected split proportions in combined export")

    # Fill annotation/metadata blanks for CSV readability, while keeping
    # probability/confidence columns as empty for original annotated songs.
    metadata = [c for c in complete.columns if c not in (
        "confidence", "prob_safe", "prob_mild", "prob_explicit")]
    complete[metadata] = complete[metadata].fillna("")

    # Every one of the 34k receives a teacher prediction, even if flagged.
    write_csv(pseudo, "pseudo_labeled_34000.csv")
    write_csv(pseudo[pseudo["needs_review"]].copy(), "review_suggestions.csv")
    write_csv(training_pool, "training_with_pseudo_39000.csv")

    # Save Training.csv beside the two input files, NOT hidden in results/.
    complete.to_csv(FINAL_EXPORT, index=False, encoding="utf-8-sig")
    print(f"Saved {FINAL_EXPORT} ({len(complete):,} songs)")

    summary = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": MODEL_NAME,
        "random_seed": SEED,
        "input_files": {"labeled": LABELED_FILE.name, "unlabeled": UNLABELED_FILE.name},
        "counts": {
            "train": len(train_df),
            "validation": len(validation_df),
            "test": len(test_df),
            "pseudo_labeled": len(pseudo),
            "review_suggestions": int(pseudo["needs_review"].sum()),
            "training_with_pseudo": len(training_pool),
            "complete": len(complete),
        },
        "training_labels_provenance": "provided; not assumed independently verified",
        "long_lyrics_handling": {
            "max_window_tokens_including_specials": MAX_LENGTH,
            "overlap_content_tokens": CHUNK_OVERLAP_TOKENS,
            "chunk_pooling": "max severity of per-window argmax classes; evidence from highest-confidence selected-class window",
            "confidence_definition": "softmax probability of the selected evidence window, NOT calibrated song accuracy",
            "train_window_count": int(len(train_dataset)),
            "validation_window_count": int(len(validation_dataset)),
            "test_window_count": int(len(test_dataset)),
            "candidate_window_count": int(pseudo["num_windows"].sum()),
            "candidate_songs_with_multiple_windows": int((pseudo["num_windows"] > 1).sum()),
            "candidate_songs_with_window_disagreement": int(pseudo["window_disagreement"].sum()),
        },
        "validation": validation_metrics,
        "held_out_test": test_metrics,
        "review_threshold": threshold,
        "review_threshold_reason": threshold_reason,
        "review_flag_definition": "low selected-window confidence OR disagreement among window predicted classes",
        "threshold_analysis_validation_only": threshold_analysis,
        "pseudo_label_distribution": pseudo["safety_label"].value_counts().to_dict(),
        "previous_candidate_label_count": int(pseudo.get("prior_safety_label", pd.Series(dtype=str)).ne("").sum()),
        "complete_export": str(FINAL_EXPORT),
    }
    (RESULTS_DIR / "run_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nDONE. Teacher fine-tuned once; all candidates labeled once; exports merged.")
    print("Uncertain predictions flagged for possible manual review:", summary["counts"]["review_suggestions"])
    print("Requested combined CSV:", FINAL_EXPORT.resolve())
    print("Full supplementary results folder:", RESULTS_DIR.resolve())
    print("IMPORTANT: Training.csv includes the 500 validation + 500 test records; ")
    print("do not fit another model on the entire Training.csv and then reuse those ")
    print("same records as an independent held-out evaluation.")
    print("Note: Machine-generated labels are predictions, NOT verified ground truth.")
    print("Conservative max-severity window pooling may increase false positives.")


if __name__ == "__main__":
    main()
