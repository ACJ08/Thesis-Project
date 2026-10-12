
from pathlib import Path
import math

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
DATA_DIR = Path(__file__).resolve().parent

MODEL_NAME = "distilbert-base-uncased"

TEST_FILE = DATA_DIR / "test_5k_annotated.csv"
VALIDATION_FILE = DATA_DIR / "validation_5k_annotated.csv"
UNLABELED_FILE = DATA_DIR / "training_set.csv"

OUTPUT_DIR = DATA_DIR / "distilbert_song_classifier_v8"
OUTPUT_FILE = DATA_DIR / "songs_pseudo_labeled_v8.csv"

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
MIN_ACCEPTED_FOR_THRESHOLD = 100
FALLBACK_ACCURACY_TOLERANCE = 0.02  # within 2 percentage points of best observed accuracy
MIN_ACCEPTED_PER_PREDICTED_CLASS = 10

# Square-root inverse-frequency weights are less aggressive than raw inverse frequency.
USE_CLASS_WEIGHTS = True
MAX_CLASS_WEIGHT = 3.0
MIN_CLASS_WEIGHT = 1.0 / MAX_CLASS_WEIGHT
RANDOM_SEEDS = [42, 123, 2026]
RANDOM_SEED = RANDOM_SEEDS[0]  # Used for the fixed stratified data split

set_seed(RANDOM_SEED)


# ============================================================
# LOAD, STANDARDIZE, AND SPLIT ANNOTATED DATA
# ============================================================

# We are repurposing the two existing 5k annotated files as one
# 10k labeled pool. Their old Test/Validation roles no longer apply.
# A new stratified 80/10/10 split is created below.

def load_annotated_file(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    required = {"song_title", "artist", "lyrics", "safety_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    df = df[["song_title", "artist", "lyrics", "safety_label"]].copy()
    df = df.dropna(subset=["song_title", "artist", "lyrics", "safety_label"])
    df["song_title"] = df["song_title"].astype(str).str.strip()
    df["artist"] = df["artist"].astype(str).str.strip()
    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().str.casefold()
    df = df[(df["song_title"] != "") & (df["artist"] != "") & (df[TEXT_COL] != "")]

    invalid = set(df[LABEL_COL].unique()) - set(LABELS)
    if invalid:
        raise ValueError(f"Invalid labels in {path.name}: {sorted(invalid)}")
    return df.reset_index(drop=True)


test_annotated_df = load_annotated_file(TEST_FILE)
validation_annotated_df = load_annotated_file(VALIDATION_FILE)

all_labeled_df = pd.concat(
    [test_annotated_df, validation_annotated_df], ignore_index=True
)

# Remove exact duplicate title/artist pairs across the combined labeled pool.
all_labeled_df["_title_key"] = all_labeled_df["song_title"].str.casefold().str.strip()
all_labeled_df["_artist_key"] = all_labeled_df["artist"].str.casefold().str.strip()
before_dedup = len(all_labeled_df)
all_labeled_df = all_labeled_df.drop_duplicates(
    subset=["_title_key", "_artist_key"], keep="first"
).drop(columns=["_title_key", "_artist_key"]).reset_index(drop=True)
print(f"Combined annotated rows: {before_dedup:,}; after duplicate removal: {len(all_labeled_df):,}")

# First reserve 10% for test, then split the remaining 90% into 8:1 train/validation.
train_df, holdout_df = train_test_split(
    all_labeled_df,
    test_size=0.20,
    random_state=RANDOM_SEED,
    stratify=all_labeled_df[LABEL_COL],
)
val_df, final_test_df = train_test_split(
    holdout_df,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=holdout_df[LABEL_COL],
)

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
    print("Validation accepted accuracy is a selection estimate, not an unbiased estimate.")
    return float(chosen[0]), val_confidence


# ============================================================
# MULTI-SEED TRAINING AND EVALUATION
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

    run_output_dir = DATA_DIR / f"distilbert_song_classifier_v8_seed_{seed}"
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

    test_output = trainer.predict(test_dataset)
    test_pred = np.argmax(test_output.predictions, axis=-1)
    test_metrics = {
        "accuracy": float(accuracy_score(test_output.label_ids, test_pred)),
        "macro_f1": float(f1_score(test_output.label_ids, test_pred, labels=[0, 1, 2],
                                    average="macro", zero_division=0)),
    }
    print(f"\nHeld-out test metrics for seed {seed}: {test_metrics}")
    print(classification_report(test_output.label_ids, test_pred, labels=[0, 1, 2],
                                target_names=LABELS, zero_division=0))
    print("Held-out test confusion matrix (true rows, predicted columns):")
    print(confusion_matrix(test_output.label_ids, test_pred, labels=[0, 1, 2]))

    run = {
        "seed": seed,
        "trainer": trainer,
        "threshold": threshold,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_confidence": val_confidence,
        "model_dir": run_output_dir,
    }
    run_results.append(run)

    # Select the deployment/pseudo-labeling model by validation macro-F1 only;
    # never use test metrics to choose the model.
    if best_run is None or val_metrics["macro_f1"] > best_run["val_metrics"]["macro_f1"]:
        best_run = run


# Summarize variability across seeds. These runs share the same split, so this
# measures initialization/training variability, not split-to-split variability.
print("\n" + "=" * 68)
print("MULTI-SEED SUMMARY (same stratified split)")
print("=" * 68)
for metric in ["accuracy", "macro_f1"]:
    values = np.array([r["test_metrics"][metric] for r in run_results], dtype=float)
    print(f"Test {metric}: mean={values.mean():.4f}, std={values.std(ddof=1) if len(values) > 1 else 0.0:.4f}")
print("Per-seed test results:")
for run in run_results:
    print(f"  seed={run['seed']}: {run['test_metrics']}")
print(
    f"\nSelected pseudo-labeling model: seed {best_run['seed']} "
    f"(best validation macro-F1={best_run['val_metrics']['macro_f1']:.4f}); "
    f"threshold={best_run['threshold']:.2f}"
)

# Save the selected model/tokenizer to the common deployment directory.
best_run["trainer"].save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)


# ============================================================
# PSEUDO-LABEL TRAINING CANDIDATE POOL
# ============================================================

best_trainer = best_run["trainer"]
CONFIDENCE_THRESHOLD = best_run["threshold"]
prediction_output = best_trainer.predict(unlabeled_dataset)
probabilities = torch.softmax(torch.tensor(prediction_output.predictions), dim=-1).numpy()
predicted_ids = np.argmax(probabilities, axis=1)
confidence = probabilities.max(axis=1)
results = unlabeled_df.copy()
results["predicted_label"] = [ID2LABEL[int(i)] for i in predicted_ids]
results["confidence"] = confidence
for i, label in ID2LABEL.items():
    results[f"prob_{label}"] = probabilities[:, i]
# Keep every song labeled. Confidence controls review status, not label presence.
results["needs_revision"] = (confidence < CONFIDENCE_THRESHOLD).astype(bool)
results[LABEL_COL] = results["predicted_label"]

val_median_confidence = float(np.median(best_run["val_confidence"]))
pool_median_confidence = float(np.median(confidence)) if len(confidence) else float("nan")
if len(confidence) and pool_median_confidence < val_median_confidence - 0.10:
    print(
        "WARNING: Median confidence on the unlabeled pool is more than 0.10 "
        "below validation. The selected threshold may not transfer reliably."
    )

results.to_csv(OUTPUT_FILE, index=False)
print("\nPseudo-labeling completed!")
print(f"Output: {OUTPUT_FILE}")
print(f"Songs processed: {len(results)}")
print(f"Selected model seed: {best_run['seed']}")
print(f"Confidence threshold used: {CONFIDENCE_THRESHOLD:.2f}")
print("\nAll predicted class distribution:")
print(results["predicted_label"].value_counts())
print("\nFinal safety_label distribution (all songs retain predictions):")
print(results[LABEL_COL].value_counts(dropna=False))
print("\nSongs needing revision (needs_revision=True):", int(results["needs_revision"].sum()))
print("Mean confidence:", round(float(confidence.mean()), 4))
print("Unlabeled-pool confidence quantiles:", pd.Series(confidence).quantile([0, .25, .5, .75, .9, 1]).round(3).to_dict())
print(f"Validation median confidence: {val_median_confidence:.3f}")
print(f"Unlabeled-pool median confidence: {pool_median_confidence:.3f}")
print("All rows retain a predicted safety_label. Rows with needs_revision=True are provisional and should be reviewed before being treated as confirmed labels.")
