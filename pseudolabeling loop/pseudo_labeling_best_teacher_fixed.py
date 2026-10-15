
from pathlib import Path
import math
import json
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ============================================================
# CONFIGURATION
# ============================================================

# Resolve all input/output paths relative to this script's folder.
DATA_DIR = Path(__file__).resolve().parent / "data"

MODEL_NAME = "distilbert-base-uncased"

# Separate training, monitoring, and candidate pools.
TRAIN_FILE = DATA_DIR / "training_1000_provisional.csv"
MONITOR_FILE = DATA_DIR / "monitoring_500_provisional.csv"
UNLABELED_FILE = DATA_DIR / "remaining_38500_unlabeled.csv"
# Provisional annotations are NOT independently verified ground truth.
ALLOW_PROVISIONAL_LABELS = True

OUTPUT_DIR = DATA_DIR / "distilbert_song_classifier"
ITERATION_MODELS_DIR = DATA_DIR / "teacher_iteration_models"
OUTPUT_FILE = DATA_DIR / "Training.csv"
REVIEW_FILE = DATA_DIR / "needs_revision_review.csv"
REVIEWED_LABELS_FILE = DATA_DIR / "reviewed_labels.csv"

# Cross-execution plateau detection (separate from epoch early stopping).
ITERATION_STATE_FILE = DATA_DIR / "iteration_history.json"
MIN_ITERATION_IMPROVEMENT = 0.005
PLATEAU_PATIENCE = 3
# Explicit manual reset: archive/remove iteration_history.json to begin a new experiment.


TEXT_COL = "lyrics"
LABEL_COL = "safety_label"

LABELS = ["safe", "mild", "explicit"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

MAX_LENGTH = 512
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
MAX_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 2
MIN_MACRO_F1_IMPROVEMENT = 0.001
LEARNING_RATE = 2e-5
# Threshold is selected using validation predictions, not guessed in advance.
THRESHOLD_CANDIDATES = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
TARGET_ACCEPTED_ACCURACY = 0.90
MIN_ACCEPTED_FOR_THRESHOLD = 25
FALLBACK_ACCURACY_TOLERANCE = 0.02  # within 2 percentage points of best observed accuracy
MIN_ACCEPTED_PER_PREDICTED_CLASS = 3

# Square-root inverse-frequency weights are less aggressive than raw inverse frequency.
USE_CLASS_WEIGHTS = True
MAX_CLASS_WEIGHT = 3.0
MIN_CLASS_WEIGHT = 1.0 / MAX_CLASS_WEIGHT
RANDOM_SEEDS = [42]
RANDOM_SEED = RANDOM_SEEDS[0]  # Used for the fixed stratified data split

set_seed(RANDOM_SEED)


# ============================================================
# LOAD, STANDARDIZE, AND SPLIT ANNOTATED DATA
# ============================================================

# Training: 1,000 provisional labels. Monitoring: 500 provisional labels.
# Split monitoring once into 250 validation (selection/threshold) and 250
# held-out test (diagnostic only). Never train on monitoring rows.

def load_annotated_file(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    required = {"song_title", "artist", "lyrics", "safety_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    if "annotation_status" in df.columns:
        provisional = df["annotation_status"].fillna("").astype(str).str.casefold().eq("needs_review")
        if provisional.any() and not ALLOW_PROVISIONAL_LABELS:
            raise ValueError(f"{path.name}: {provisional.sum()} provisional labels need human verification")
    df = df[["song_title", "artist", "lyrics", "safety_label"]].copy()
    df = df.dropna(subset=["song_title", "artist", "lyrics", "safety_label"])
    for col in ["song_title", "artist", "lyrics", "safety_label"]:
        df[col] = df[col].astype(str).str.strip()
    df[LABEL_COL] = df[LABEL_COL].str.casefold()
    df = df[(df["song_title"] != "") & (df["artist"] != "") & (df["lyrics"] != "")]
    invalid = set(df[LABEL_COL]) - set(LABELS)
    if invalid:
        raise ValueError(f"Invalid labels in {path.name}: {sorted(invalid)}")
    df = add_song_keys(df).drop_duplicates("_song_key", keep="first")
    return df.drop(columns=["_title_key", "_artist_key", "_song_key"]).reset_index(drop=True)


def add_song_keys(df):
    df = df.copy()
    df["_title_key"] = df["song_title"].astype(str).str.casefold().str.strip()
    df["_artist_key"] = df["artist"].astype(str).str.casefold().str.strip()
    df["_song_key"] = df["_title_key"] + "|||" + df["_artist_key"]
    return df

def fingerprint_rows(df, label_column):
    """Stable digest of the actual labeled training inputs, regardless of row order."""
    records = (
        (str(row.song_title).strip().casefold(), str(row.artist).strip().casefold(),
         str(row.lyrics).strip(), str(getattr(row, label_column)).strip().casefold())
        for row in df.itertuples(index=False)
    )
    digest = hashlib.sha256()
    for record in sorted(records):
        digest.update(json.dumps(record, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def save_iteration_state(state):
    ITERATION_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = ITERATION_STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(ITERATION_STATE_FILE)


def read_iteration_state():
    if not ITERATION_STATE_FILE.exists():
        return {"version": 1, "best_macro_f1": None, "best_iteration": None,
                "plateau_counter": 0, "stopped": False, "runs": []}
    state = json.loads(ITERATION_STATE_FILE.read_text(encoding="utf-8"))
    if state.get("version") != 1 or not isinstance(state.get("runs"), list):
        raise ValueError("Unsupported iteration history format; inspect the state file.")
    # Older histories lack persistent checkpoints. Never pretend the historical
    # best can be recovered from a model directory overwritten on later runs.
    if state.get("runs") and not state.get("stopped"):
        best_dir = state.get("best_model_dir")
        if not best_dir or not (Path(best_dir) / "config.json").is_file():
            raise RuntimeError(
                "LEGACY HISTORY: historical best teacher checkpoint is not available. "
                "Back up Training.csv, reviewed_labels.csv, and iteration_history.json; "
                "then start a NEW versioned experiment. Do not assume the existing "
                "distilbert_song_classifier directory contains the historical best."
            )
    return state


# Hard-stop before any expensive model work, even if the user adds more reviews.
iteration_state = read_iteration_state()
if iteration_state.get("stopped"):
    raise SystemExit(
        f"PLATEAU HARD STOP: {iteration_state['plateau_counter']} consecutive "
        f"non-improving iterations. Best validation macro-F1: "
        f"{iteration_state['best_macro_f1']:.4f} (iteration "
        f"{iteration_state['best_iteration']}). No training performed. "
        f"To deliberately start a NEW experiment, archive/rename {ITERATION_STATE_FILE.name}."
    )

train_df = load_annotated_file(TRAIN_FILE)
monitor_df = load_annotated_file(MONITOR_FILE)
if len(train_df) < 10 or len(monitor_df) < 10:
    raise ValueError("Not enough valid labeled rows in training/monitoring datasets")
val_df, final_test_df = train_test_split(
    monitor_df, test_size=0.5, random_state=RANDOM_SEED,
    stratify=monitor_df[LABEL_COL],
)
all_labeled_df = pd.concat([train_df, monitor_df], ignore_index=True)
if len(train_df) != 1000 or len(monitor_df) != 500:
    raise ValueError(f"Expected 1000/500 unique labeled songs, got {len(train_df)}/{len(monitor_df)}")

# Load the candidate pool produced by processing.py.
unlabeled_df = pd.read_csv(UNLABELED_FILE)
unlabeled_df.columns = unlabeled_df.columns.str.strip()

# Accept the processed schema, and also the original raw songs schema.
if "lyrics" not in unlabeled_df.columns and "text" in unlabeled_df.columns:
    unlabeled_df = unlabeled_df.rename(columns={"text": "lyrics"})
if "song_title" not in unlabeled_df.columns and "song" in unlabeled_df.columns:
    unlabeled_df = unlabeled_df.rename(columns={"song": "song_title"})
if "artist" not in unlabeled_df.columns and "Artist(s)" in unlabeled_df.columns:
    unlabeled_df = unlabeled_df.rename(columns={"Artist(s)": "artist"})

required_unlabeled = {"song_title", "artist", "lyrics"}
missing_unlabeled = required_unlabeled - set(unlabeled_df.columns)
if missing_unlabeled:
    raise ValueError(f"{UNLABELED_FILE.name} is missing columns: {sorted(missing_unlabeled)}")

unlabeled_df = unlabeled_df.dropna(subset=["song_title", "artist", "lyrics"]).copy()
for col in ["song_title", "artist", "lyrics"]:
    unlabeled_df[col] = unlabeled_df[col].astype(str).str.strip()
unlabeled_df = unlabeled_df[
    (unlabeled_df["song_title"] != "")
    & (unlabeled_df["artist"] != "")
    & (unlabeled_df["lyrics"] != "")
].reset_index(drop=True)

# Stable title/artist identity keys prevent data leakage.
all_labeled_df = add_song_keys(all_labeled_df)
train_df = add_song_keys(train_df)
val_df = add_song_keys(val_df)
final_test_df = add_song_keys(final_test_df)
unlabeled_df = add_song_keys(unlabeled_df)
before_candidate_dedup = len(unlabeled_df)
unlabeled_df = unlabeled_df.drop_duplicates(subset=["_song_key"], keep="first").copy()
print(f"Removed {before_candidate_dedup - len(unlabeled_df):,} duplicate candidate rows.")

# Do not pseudo-label songs already present in any part of the annotated pool.
annotated_keys = set(all_labeled_df["_song_key"])
before_overlap_filter = len(unlabeled_df)
unlabeled_df = unlabeled_df[~unlabeled_df["_song_key"].isin(annotated_keys)].copy()
print(f"Removed {before_overlap_filter - len(unlabeled_df):,} candidate rows overlapping the annotated pool.")

# Human-in-the-loop cycle:
# 1) reviewers fill human_label in REVIEW_FILE,
# 2) submitted labels are appended to persistent REVIEWED_LABELS_FILE,
# 3) every later run trains on the full correction archive and relabels remaining candidates.
if REVIEW_FILE.exists():
    prior_review_df = pd.read_csv(REVIEW_FILE).fillna("")
    prior_review_df.columns = prior_review_df.columns.str.strip()
    required_review = {"song_title", "artist", "lyrics", "human_label"}
    missing_review = required_review - set(prior_review_df.columns)
    if missing_review:
        raise ValueError(f"{REVIEW_FILE.name} is missing columns: {sorted(missing_review)}")
    for col in ["song_title", "artist", "lyrics", "human_label"]:
        prior_review_df[col] = prior_review_df[col].astype(str).str.strip()
    prior_review_df["human_label"] = prior_review_df["human_label"].str.casefold()
    nonblank_labels = prior_review_df["human_label"] != ""
    invalid_human = set(prior_review_df.loc[nonblank_labels, "human_label"]) - set(LABELS)
    if invalid_human:
        raise ValueError(
            f"Invalid human_label values in {REVIEW_FILE.name}: {sorted(invalid_human)}. "
            f"Use only: {LABELS}"
        )
    newly_reviewed_df = prior_review_df.loc[nonblank_labels, ["song_title", "artist", "lyrics", "human_label"]].copy()
    pending_review_df = prior_review_df.loc[~nonblank_labels].copy()
else:
    prior_review_df = pd.DataFrame(columns=["song_title", "artist", "lyrics", "human_label"])
    newly_reviewed_df = prior_review_df.copy()
    pending_review_df = prior_review_df.copy()

# Load the persistent correction archive so reviewed labels are never forgotten
# when the pending review queue is refreshed after a training round.
if REVIEWED_LABELS_FILE.exists():
    archived_reviewed_df = pd.read_csv(REVIEWED_LABELS_FILE).fillna("")
    archived_reviewed_df.columns = archived_reviewed_df.columns.str.strip()
    required_archive = {"song_title", "artist", "lyrics", "human_label"}
    missing_archive = required_archive - set(archived_reviewed_df.columns)
    if missing_archive:
        raise ValueError(f"{REVIEWED_LABELS_FILE.name} is missing columns: {sorted(missing_archive)}")
    archived_reviewed_df = archived_reviewed_df[["song_title", "artist", "lyrics", "human_label"]].copy()
    for col in ["song_title", "artist", "lyrics", "human_label"]:
        archived_reviewed_df[col] = archived_reviewed_df[col].astype(str).str.strip()
    archived_reviewed_df["human_label"] = archived_reviewed_df["human_label"].str.casefold()
else:
    archived_reviewed_df = pd.DataFrame(columns=["song_title", "artist", "lyrics", "human_label"])

all_reviewed_raw = pd.concat([archived_reviewed_df, newly_reviewed_df], ignore_index=True)
all_reviewed_raw = all_reviewed_raw[all_reviewed_raw["human_label"] != ""].copy()
invalid_archive_labels = set(all_reviewed_raw["human_label"]) - set(LABELS)
if invalid_archive_labels:
    raise ValueError(
        f"Invalid human_label values in {REVIEWED_LABELS_FILE.name}: {sorted(invalid_archive_labels)}. "
        f"Use only: {LABELS}"
    )

if len(all_reviewed_raw):
    all_reviewed_raw = add_song_keys(all_reviewed_raw)
    all_reviewed_raw = all_reviewed_raw.drop_duplicates(subset=["_song_key"], keep="last").reset_index(drop=True)
    reviewed_df = all_reviewed_raw.copy()
    reviewed_df[["song_title", "artist", "lyrics", "human_label"]].to_csv(REVIEWED_LABELS_FILE, index=False)
else:
    reviewed_df = all_reviewed_raw.copy()
    # Create the archive with headers on the first run for discoverability.
    reviewed_df[["song_title", "artist", "lyrics", "human_label"]].to_csv(REVIEWED_LABELS_FILE, index=False)

reviewed_df = add_song_keys(reviewed_df) if len(reviewed_df) and "_song_key" not in reviewed_df.columns else reviewed_df
if len(reviewed_df):
    # Human-reviewed records must not collide with any original annotated split.
    overlapping_review_keys = set(reviewed_df["_song_key"]) & annotated_keys
    if overlapping_review_keys:
        raise ValueError(
            "Some reviewed candidate songs overlap the training/monitoring records. Remove these "
            "duplicates from the reviewed-label archive before continuing; annotated splits must stay distinct."
        )
    reviewed_train = reviewed_df[["song_title", "artist", "lyrics", "_title_key", "_artist_key", "_song_key"]].copy()
    reviewed_train[LABEL_COL] = reviewed_df["human_label"].values
    reviewed_train = reviewed_train.drop_duplicates(subset=["_title_key", "_artist_key"])
    reviewed_train = reviewed_train.drop(columns=["_title_key", "_artist_key", "_song_key"], errors="ignore")
    train_df = pd.concat([train_df, reviewed_train], ignore_index=True)
    reviewed_keys = set(reviewed_df["_song_key"])
    unlabeled_df = unlabeled_df[~unlabeled_df["_song_key"].isin(reviewed_keys)].copy()
    print(f"New human corrections submitted this round: {len(newly_reviewed_df):,}")
    print(f"Total human-corrected rows included in training: {len(reviewed_train):,}")
else:
    reviewed_keys = set()

# Detect unchanged input before spending GPU time; use the corrected training
# records as well as the base provisional labels.
current_fingerprint = fingerprint_rows(train_df, LABEL_COL)
prior_runs = iteration_state["runs"]
if prior_runs and current_fingerprint == prior_runs[-1]["training_fingerprint"]:
    raise SystemExit(
        "NO NEW TRAINING DATA: the labeled training set is identical to the last "
        "completed iteration. Add verified human_label entries to the review CSV "
        "before rerunning. No training performed."
    )
next_iteration = len(prior_runs) + 1
print(f"Iteration {next_iteration}; plateau patience "
      f"{iteration_state['plateau_counter']}/{PLATEAU_PATIENCE}")

# Keep identity helper columns out of tokenization/output.
train_df = train_df.drop(columns=["_title_key", "_artist_key", "_song_key"], errors="ignore")
val_df = val_df.drop(columns=["_title_key", "_artist_key", "_song_key"], errors="ignore")
final_test_df = final_test_df.drop(columns=["_title_key", "_artist_key", "_song_key"], errors="ignore")
unlabeled_df = unlabeled_df.drop(columns=["_title_key", "_artist_key", "_song_key"], errors="ignore").reset_index(drop=True)

print(f"Training songs: {len(train_df):,}")
print(f"Validation songs: {len(val_df):,}")
print(f"Held-out test songs: {len(final_test_df):,}")
print(f"Pseudo-label candidates: {len(unlabeled_df):,}")
print("\nTraining distribution:")
print(train_df[LABEL_COL].value_counts())

for name, df in [("Training", train_df), ("Validation", val_df), ("Held-out test", final_test_df)]:
    missing_classes = set(LABELS) - set(df[LABEL_COL].unique())
    if missing_classes:
        raise ValueError(f"{name} split is missing classes: {sorted(missing_classes)}")

train_df = train_df.copy()
val_df = val_df.copy()
final_test_df = final_test_df.copy()
train_df["labels"] = train_df[LABEL_COL].map(LABEL2ID)
val_df["labels"] = val_df[LABEL_COL].map(LABEL2ID)
final_test_df["labels"] = final_test_df[LABEL_COL].map(LABEL2ID)


# ============================================================
# TOKENIZATION AND SHARED DATASETS
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_batch(batch):
    return tokenizer(
        batch[TEXT_COL],
        truncation=True,
        max_length=MAX_LENGTH,
    )


def make_dataset(df, include_labels=True):
    columns = [TEXT_COL]
    if include_labels:
        columns.append("labels")
    dataset = Dataset.from_pandas(
        df[columns].reset_index(drop=True),
        preserve_index=False,
    )
    return dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=[TEXT_COL],
    )


train_dataset = make_dataset(train_df)
val_dataset = make_dataset(val_df)
test_dataset = make_dataset(final_test_df)
unlabeled_dataset = make_dataset(unlabeled_df, include_labels=False)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# ============================================================
# METRICS, WEIGHTS, AND TRAINER
# ============================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels, predictions, labels=[0, 1, 2],
            average="macro", zero_division=0,
        ),
    }


class WeightedTrainer(Trainer):
    """Trainer using capped square-root inverse-frequency cross-entropy weights."""
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if labels is None:
            loss = outputs.get("loss")
            return (loss, outputs) if return_outputs else loss
        if self.class_weights is None:
            loss = outputs.get("loss")
        else:
            loss = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))(
                logits.reshape(-1, model.config.num_labels), labels.reshape(-1)
            )
        return (loss, outputs) if return_outputs else loss


class_counts = train_df[LABEL_COL].value_counts().reindex(LABELS, fill_value=0)
if (class_counts == 0).any():
    raise ValueError(
        "Cannot compute class weights; missing training classes: "
        f"{class_counts[class_counts == 0].index.tolist()}"
    )

raw_class_weights = np.sqrt(
    len(train_df) / (len(LABELS) * class_counts.to_numpy(dtype=float))
)
raw_class_weights = np.clip(raw_class_weights, MIN_CLASS_WEIGHT, MAX_CLASS_WEIGHT)
normalized_class_weights = raw_class_weights / raw_class_weights.mean()
# Re-clip after normalization, as requested. The final weights are then used as-is.
class_weight_values = np.clip(
    normalized_class_weights, MIN_CLASS_WEIGHT, MAX_CLASS_WEIGHT
)
class_weights = (
    torch.tensor(class_weight_values, dtype=torch.float32)
    if USE_CLASS_WEIGHTS else None
)
print("\nClass weights (sqrt inverse-frequency, normalized and re-clipped):")
print({label: round(float(weight), 4) for label, weight in zip(LABELS, class_weight_values)})

CUDA_AVAILABLE = torch.cuda.is_available()
BF16_SUPPORTED = CUDA_AVAILABLE and torch.cuda.is_bf16_supported()


def select_confidence_threshold(val_probabilities, val_true_ids, seed):
    val_pred = np.argmax(val_probabilities, axis=1)
    val_confidence = val_probabilities.max(axis=1)
    val_correct = val_pred == val_true_ids
    rows = []
    print(f"\nConfidence-threshold analysis (seed {seed}):")
    for threshold in THRESHOLD_CANDIDATES:
        accepted = val_confidence >= threshold
        count = int(accepted.sum())
        coverage = float(accepted.mean()) if len(accepted) else 0.0
        accepted_accuracy = float(val_correct[accepted].mean()) if count else 0.0
        class_counts_accepted = (
            pd.Series(val_pred[accepted]).value_counts() if count
            else pd.Series(dtype=int)
        )
        min_class_count = min(
            (int(class_counts_accepted.get(class_id, 0)) for class_id in range(len(LABELS))),
            default=0,
        )
        row = (threshold, count, coverage, accepted_accuracy, min_class_count)
        rows.append(row)
        print(
            f"Threshold={threshold:.2f} | accepted={count:,}/{len(val_true_ids):,} "
            f"({coverage:.1%}) | accepted accuracy={accepted_accuracy:.1%} "
            f"| minimum predicted-class count={min_class_count}"
        )

    eligible = [r for r in rows if r[3] >= TARGET_ACCEPTED_ACCURACY
                and r[1] >= MIN_ACCEPTED_FOR_THRESHOLD
                and r[4] >= MIN_ACCEPTED_PER_PREDICTED_CLASS]
    if eligible:
        chosen = min(eligible, key=lambda r: r[0])
        reason = f"lowest threshold meeting target accuracy {TARGET_ACCEPTED_ACCURACY:.0%}"
    else:
        viable = [r for r in rows if r[1] >= MIN_ACCEPTED_FOR_THRESHOLD
                  and r[4] >= MIN_ACCEPTED_PER_PREDICTED_CLASS]
        if viable:
            best_accuracy = max(r[3] for r in viable)
            near_best = [r for r in viable if r[3] >= best_accuracy - FALLBACK_ACCURACY_TOLERANCE]
            chosen = min(near_best, key=lambda r: r[0])
            reason = f"lowest threshold within {FALLBACK_ACCURACY_TOLERANCE:.0%} of best observed accuracy"
        else:
            chosen = (0.90, 0, 0.0, 0.0, 0)
            reason = "conservative fallback; no candidate met minimum counts"
    print(f"Selected threshold for seed {seed}: {chosen[0]:.2f} ({reason}).")
    return float(chosen[0]), val_confidence


# ============================================================
# SINGLE-SEED TRAINING AND EVALUATION
# ============================================================

run_results = []
best_run = None

for seed in RANDOM_SEEDS:
    print("\n" + "=" * 68)
    print(f"TRAINING RUN FOR SEED {seed}")
    print("=" * 68)
    set_seed(seed)
    print(
        f"Early stopping: monitor validation macro-F1; max epochs={MAX_EPOCHS}, "
        f"patience={EARLY_STOPPING_PATIENCE}, "
        f"minimum improvement={MIN_MACRO_F1_IMPROVEMENT:.3f}"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    run_output_dir = DATA_DIR / f"distilbert_song_classifier_seed_{seed}"
    training_args = TrainingArguments(
        output_dir=run_output_dir,
        seed=seed,
        data_seed=RANDOM_SEED,  # Keep the data split/order basis fixed across runs.
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=MAX_EPOCHS,
        weight_decay=0.01,
        # Approximate the former 10% warmup ratio using the maximum schedule.
        warmup_steps=max(
            1,
            int(0.1 * math.ceil(len(train_dataset) / TRAIN_BATCH_SIZE) * MAX_EPOCHS),
        ),
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=CUDA_AVAILABLE and not BF16_SUPPORTED,
        bf16=BF16_SUPPORTED,
        report_to="none",
        save_total_limit=2,
        logging_steps=50,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=MIN_MACRO_F1_IMPROVEMENT,
            )
        ],
    )
    trainer.train()

    # One prediction pass per split for this seed.
    val_output = trainer.predict(val_dataset)
    val_pred = np.argmax(val_output.predictions, axis=-1)
    val_probabilities = torch.softmax(torch.tensor(val_output.predictions), dim=-1).numpy()
    val_metrics = {
        "accuracy": float(accuracy_score(val_output.label_ids, val_pred)),
        "macro_f1": float(f1_score(val_output.label_ids, val_pred, labels=[0, 1, 2],
                                    average="macro", zero_division=0)),
    }
    print(f"\nValidation metrics for seed {seed}: {val_metrics}")
    print(classification_report(val_output.label_ids, val_pred, labels=[0, 1, 2],
                                target_names=LABELS, zero_division=0))
    print("Validation confusion matrix (true rows, predicted columns):")
    print(confusion_matrix(val_output.label_ids, val_pred, labels=[0, 1, 2]))

    threshold, val_confidence = select_confidence_threshold(
        val_probabilities, val_output.label_ids, seed
    )

    run = {
        "seed": seed,
        "trainer": trainer,
        "threshold": threshold,
        "val_metrics": val_metrics,
        "val_confidence": val_confidence,
        "model_dir": run_output_dir,
    }
    run_results.append(run)

    # Select the deployment/pseudo-labeling model by validation macro-F1 only;
    # never use test metrics to choose the model.
    if best_run is None or val_metrics["macro_f1"] > best_run["val_metrics"]["macro_f1"]:
        best_run = run


# Summarize variability using validation only. The held-out test split is evaluated
# exactly once, after selecting the model, and is never used for model/threshold selection.
print("\n" + "=" * 68)
print("SINGLE-SEED SUMMARY (validation metrics)")
print("=" * 68)
for metric in ["accuracy", "macro_f1"]:
    values = np.array([r["val_metrics"][metric] for r in run_results], dtype=float)
    print(f"Validation {metric}: mean={values.mean():.4f}, std={values.std(ddof=1) if len(values) > 1 else 0.0:.4f}")
print("Validation results:")
for run in run_results:
    print(f"  seed={run['seed']}: {run['val_metrics']}")
print(
    f"\nSelected pseudo-labeling model: seed {best_run['seed']} "
    f"(best validation macro-F1={best_run['val_metrics']['macro_f1']:.4f}); "
    f"threshold={best_run['threshold']:.2f}"
)

# Keep the held-out monitoring partition out of iteration decisions.
# Evaluate it only on the first run, not after repeatedly selecting models.
if next_iteration == 1:
    final_test_output = best_run["trainer"].predict(test_dataset)
    final_test_pred = np.argmax(final_test_output.predictions, axis=-1)
    final_test_metrics = {
        "accuracy": float(accuracy_score(final_test_output.label_ids, final_test_pred)),
        "macro_f1": float(f1_score(final_test_output.label_ids, final_test_pred,
                                    labels=[0, 1, 2], average="macro", zero_division=0)),
    }
    print(f"\nINITIAL HELD-OUT MONITORING DIAGNOSTIC: {final_test_metrics}")

# Keep an immutable checkpoint for EVERY completed iteration. A later run must
# never overwrite the historical best teacher, even if it fits more reviewed rows.
iteration_model_dir = ITERATION_MODELS_DIR / f"iteration_{next_iteration:03d}_seed_{best_run['seed']}"
if iteration_model_dir.exists():
    raise FileExistsError(f"Refusing to overwrite an existing iteration checkpoint: {iteration_model_dir}")
best_run["trainer"].save_model(iteration_model_dir)
tokenizer.save_pretrained(iteration_model_dir)
(iteration_model_dir / "selection_metadata.json").write_text(json.dumps({
    "iteration": next_iteration,
    "seed": best_run["seed"],
    "monitor_macro_f1": best_run["val_metrics"]["macro_f1"],
    "monitor_accuracy": best_run["val_metrics"]["accuracy"],
    "threshold": best_run["threshold"],
    "training_fingerprint": current_fingerprint,
    "monitor_labels_are_provisional": ALLOW_PROVISIONAL_LABELS,
}, indent=2), encoding="utf-8")

current_f1 = best_run["val_metrics"]["macro_f1"]
previous_best = iteration_state["best_macro_f1"]
# The historical best is the actual highest score. The 0.005 minimum is only
# the tolerance for resetting plateau patience, NOT for deciding best model.
is_new_best = previous_best is None or current_f1 > previous_best
significant = previous_best is None or current_f1 >= previous_best + MIN_ITERATION_IMPROVEMENT
if is_new_best:
    iteration_state["best_macro_f1"] = current_f1
    iteration_state["best_iteration"] = next_iteration
    iteration_state["best_model_dir"] = str(iteration_model_dir.resolve())
    iteration_state["best_threshold"] = float(best_run["threshold"])
    iteration_state["best_seed"] = int(best_run["seed"])
if significant:
    iteration_state["plateau_counter"] = 0
else:
    iteration_state["plateau_counter"] += 1
iteration_state["stopped"] = iteration_state["plateau_counter"] >= PLATEAU_PATIENCE
iteration_state["runs"].append({
    "iteration": next_iteration,
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "training_fingerprint": current_fingerprint,
    "training_count": len(train_df),
    "reviewed_count": len(reviewed_df),
    "validation_macro_f1": current_f1,
    "validation_accuracy": best_run["val_metrics"]["accuracy"],
    "significant_improvement": significant,
    "historical_best_updated": is_new_best,
    "model_dir": str(iteration_model_dir.resolve()),
    "confidence_threshold": float(best_run["threshold"]),
    "plateau_counter": iteration_state["plateau_counter"],
})
print(f"Iteration {next_iteration}: monitoring macro-F1={current_f1:.4f}; "
      f"historical best={iteration_state['best_macro_f1']:.4f} "
      f"(iteration {iteration_state['best_iteration']}); "
      f"plateau={iteration_state['plateau_counter']}/{PLATEAU_PATIENCE}")

# Always export candidate predictions using the HISTORICAL best teacher.
# Human corrections accumulated in later rounds are overlaid separately below.
selected_model_dir = Path(iteration_state["best_model_dir"])
if not (selected_model_dir / "config.json").is_file():
    raise FileNotFoundError(f"Historical best teacher checkpoint is missing: {selected_model_dir}")
CONFIDENCE_THRESHOLD = float(iteration_state["best_threshold"])
selected_iteration = int(iteration_state["best_iteration"])
selected_seed = int(iteration_state["best_seed"])
if is_new_best:
    best_trainer = best_run["trainer"]
else:
    # Trainer.predict uses trainer.model; replace the current teacher with
    # the historical checkpoint for inference without another training pass.
    historical_model = AutoModelForSequenceClassification.from_pretrained(selected_model_dir)
    best_trainer = best_run["trainer"]
    best_trainer.model = historical_model.to(best_trainer.args.device)

# Preserve a convenient deployable copy; iteration snapshots are authoritative.
best_trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
(OUTPUT_DIR / "selected_teacher.json").write_text(json.dumps({
    "selected_iteration": selected_iteration,
    "selected_seed": selected_seed,
    "historical_model_dir": str(selected_model_dir),
    "monitor_macro_f1": iteration_state["best_macro_f1"],
    "threshold": CONFIDENCE_THRESHOLD,
    "current_iteration": next_iteration,
    "reviewed_labels_overlaid": int(len(reviewed_df)),
}, indent=2), encoding="utf-8")
print(f"Pseudo-label export teacher: iteration {selected_iteration}, seed {selected_seed}; "
      f"threshold={CONFIDENCE_THRESHOLD:.2f}")

# ============================================================
# PSEUDO-LABEL CANDIDATE POOL + HUMAN REVIEW / RETRAINING LOOP
# ============================================================

if len(unlabeled_df):
    prediction_output = best_trainer.predict(unlabeled_dataset)
    probabilities = torch.softmax(torch.tensor(prediction_output.predictions), dim=-1).numpy()
    predicted_ids = np.argmax(probabilities, axis=1)
    confidence = probabilities.max(axis=1)

    results = unlabeled_df.copy()
    results["predicted_label"] = [ID2LABEL[int(i)] for i in predicted_ids]
    results["confidence"] = confidence
    for i, label in ID2LABEL.items():
        results[f"prob_{label}"] = probabilities[:, i]
    results["needs_revision"] = (confidence < CONFIDENCE_THRESHOLD).astype(bool)
    results[LABEL_COL] = results["predicted_label"]
    results["human_label"] = ""
    results["label_source"] = "pseudo_label"
else:
    results = pd.DataFrame(columns=[
        "song_title", "artist", "lyrics", "predicted_label", "confidence",
        "prob_safe", "prob_mild", "prob_explicit", "needs_revision",
        LABEL_COL, "human_label", "label_source"
    ])
    confidence = np.array([], dtype=float)

# The displayed validation confidence must belong to the selected teacher.
if is_new_best:
    selected_val_confidence = best_run["val_confidence"]
else:
    selected_val_output = best_trainer.predict(val_dataset)
    selected_val_probabilities = torch.softmax(torch.tensor(selected_val_output.predictions), dim=-1).numpy()
    selected_val_confidence = selected_val_probabilities.max(axis=1)
val_median_confidence = float(np.median(selected_val_confidence))
pool_median_confidence = float(np.median(confidence)) if len(confidence) else float("nan")

# Preserve manually reviewed labels as confirmed rows in Training.csv.
if len(reviewed_df):
    reviewed_output = reviewed_df[["song_title", "artist", "lyrics", "human_label"]].copy()
    reviewed_output["predicted_label"] = reviewed_output["human_label"]
    reviewed_output[LABEL_COL] = reviewed_output["human_label"]
    reviewed_output["confidence"] = np.nan
    for label in LABELS:
        reviewed_output[f"prob_{label}"] = np.nan
    reviewed_output["needs_revision"] = False
    reviewed_output["label_source"] = "human_reviewed"
    results = pd.concat([results, reviewed_output], ignore_index=True, sort=False)

# Existing pending rows remain marked needs_revision until a human confirms a label,
# even if a later model run becomes more confident about them.
if len(results) and len(pending_review_df):
    pending_keys = set(add_song_keys(pending_review_df)["_song_key"])
    result_keys = add_song_keys(results)
    results.loc[result_keys["_song_key"].isin(pending_keys), "needs_revision"] = True

# Create/refresh a review queue. Existing pending rows remain pending until a human
# fills human_label; newly flagged predictions are appended to that queue.
newly_flagged = results[
    results.get("needs_revision", pd.Series(dtype=bool)).astype(str).str.casefold().isin(["true", "1"])
].copy() if len(results) else pd.DataFrame()
if len(newly_flagged):
    newly_flagged["human_label"] = ""
    newly_flagged["review_status"] = "pending"
    queue_columns = list(dict.fromkeys(
        ["song_title", "artist", "lyrics", "predicted_label", "confidence",
         "prob_safe", "prob_mild", "prob_explicit", LABEL_COL,
         "needs_revision", "human_label", "review_status"]
    ))
    newly_flagged = newly_flagged.reindex(columns=queue_columns)

if len(pending_review_df):
    pending_review_df = pending_review_df.copy()
    if "review_status" not in pending_review_df.columns:
        pending_review_df["review_status"] = "pending"
    pending_review_df["review_status"] = "pending"

queue_parts = [df for df in [pending_review_df, newly_flagged] if len(df)]
if queue_parts:
    review_queue = pd.concat(queue_parts, ignore_index=True, sort=False)
    review_queue = review_queue.drop_duplicates(
        subset=["song_title", "artist"], keep="last"
    ).reset_index(drop=True)
    review_queue["human_label"] = review_queue["human_label"].fillna("").astype(str).str.strip().str.casefold()
    review_queue = review_queue[review_queue["human_label"] == ""].copy()
    review_queue["review_status"] = "pending"
    review_queue.to_csv(REVIEW_FILE, index=False)
else:
    review_queue = pd.DataFrame(columns=[
        "song_title", "artist", "lyrics", "predicted_label", "confidence",
        "prob_safe", "prob_mild", "prob_explicit", LABEL_COL,
        "needs_revision", "human_label", "review_status"
    ])
    review_queue.to_csv(REVIEW_FILE, index=False)

results.to_csv(OUTPUT_FILE, index=False)
# Commit history only after the output files are successfully written.
save_iteration_state(iteration_state)
if iteration_state["stopped"]:
    print("PLATEAU REACHED: future executions will stop before training. "
          "Archive iteration_history.json only if deliberately starting a new experiment.")
print("\nPseudo-labeling round completed!")
print(f"Training output: {OUTPUT_FILE}")
print(f"Human review queue: {REVIEW_FILE}")
print(f"Persistent reviewed-label archive: {REVIEWED_LABELS_FILE}")
print(f"Songs in output: {len(results):,}")
print(f"Selected historical teacher: iteration {selected_iteration}, seed {selected_seed}")
print(f"Confidence threshold used: {CONFIDENCE_THRESHOLD:.2f}")
if len(results):
    print("\nCurrent output label distribution:")
    print(results[LABEL_COL].value_counts(dropna=False))
print(f"\nNewly/previously pending rows in review queue: {len(review_queue):,}")
print(f"Human-corrected rows used for this retraining round: {len(reviewed_df):,}")
if len(confidence):
    print("Mean candidate confidence:", round(float(confidence.mean()), 4))
    print("Candidate confidence quantiles:", pd.Series(confidence).quantile([0, .25, .5, .75, .9, 1]).round(3).to_dict())
print(f"Validation median confidence: {val_median_confidence:.3f}")
print(f"Candidate-pool median confidence: {pool_median_confidence:.3f}")
print("\nHUMAN-IN-THE-LOOP INSTRUCTIONS:")
print(f"1. Open {REVIEW_FILE.name} and manually fill human_label for each reviewed row.")
print(f"2. Use only these labels: {LABELS} (leave uncertain rows blank).")
print("3. Save the CSV, then rerun this script. Corrected rows are added to training,")
print("   the model is retrained, and the remaining candidate pool is relabeled.")
print("4. Corrections are retained in reviewed_labels.csv and reused in every later training round.")
print("5. Unreviewed rows stay in the queue; do not treat them as confirmed labels.")
