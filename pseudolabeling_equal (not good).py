
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
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"

# The two existing 5k annotated files are combined into one 10k pool,
# then re-split into stratified 80% train / 10% validation / 10% test.
TEST_SOURCE_FILE = DATA_DIR / "Testing.csv"
VALIDATION_SOURCE_FILE = DATA_DIR / "Validation].csv"
SPOTIFY_FILE = DATA_DIR / "spotify_dataset.csv"
# If the CSV is still in the KaggleHub cache, the script will use this known
# location as a fallback when the ./data copy does not exist.
SPOTIFY_CACHE_FILE = Path(
    r"C:\Users\Sephie\.cache\kagglehub\datasets\devdope\900k-spotify\versions\3\spotify_dataset.csv"
)

OUTPUT_DIR = DATA_DIR / "distilbert_song_classifier"
OUTPUT_ALL_PSEUDO_LABELS = DATA_DIR / "spotify_pseudo_labeled_all.csv"
OUTPUT_BALANCED_40K = DATA_DIR / "Training.csv"
PREDICTION_CHUNK_SIZE = 10000
TARGET_PER_CLASS = 13333  # One class gets one extra row for exactly 40,000.

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
# LOAD THE 10K ANNOTATED POOL, CREATE A STRATIFIED 80/10/10 SPLIT,
# AND LOAD THE ORIGINAL SPOTIFY DATASET
# ============================================================

# As agreed, the two 5k files are not kept in their former test/validation
# roles. They are combined into one labeled pool, deduplicated, then split
# 80/10/10 with stratification. The new test split remains held out from
# training and model selection.
def load_annotated_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required annotated file not found: {path}\n"
            "Place test_5k_annotated.csv and validation_5k_annotated.csv "
            "in the data folder. These two files will be merged and split "
            "into 80% training, 10% validation, and 10% test."
        )
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    required = {"song_title", "artist", "lyrics", "safety_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    keep = ["song_title", "artist", "lyrics", "safety_label"]
    if "justification" in df.columns:
        keep.append("justification")
    df = df[keep].copy()
    df = df.dropna(subset=["song_title", "artist", "lyrics", "safety_label"])
    for col in ["song_title", "artist", "lyrics"]:
        df[col] = df[col].astype(str).str.strip()
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().str.casefold()
    df = df[(df["song_title"] != "") & (df["artist"] != "") & (df[TEXT_COL] != "")]
    invalid = set(df[LABEL_COL].unique()) - set(LABELS)
    if invalid:
        raise ValueError(f"Invalid labels in {path.name}: {sorted(invalid)}")
    return df.reset_index(drop=True)


def add_match_keys(df):
    df = df.copy()
    df["_title_key"] = df["song_title"].astype(str).str.casefold().str.strip()
    df["_artist_key"] = df["artist"].astype(str).str.casefold().str.strip()
    return df


test_source_df = load_annotated_file(TEST_SOURCE_FILE)
validation_source_df = load_annotated_file(VALIDATION_SOURCE_FILE)
annotated_pool = pd.concat([test_source_df, validation_source_df], ignore_index=True)
annotated_pool = add_match_keys(annotated_pool)
rows_before_dedup = len(annotated_pool)
annotated_pool = annotated_pool.drop_duplicates(
    subset=["_title_key", "_artist_key"], keep="first"
).drop(columns=["_title_key", "_artist_key"]).reset_index(drop=True)
print(
    f"Combined annotated pool: {rows_before_dedup:,} rows; "
    f"{len(annotated_pool):,} after title/artist deduplication."
)
print("Combined annotated label distribution:")
print(annotated_pool[LABEL_COL].value_counts())

# First reserve 20% as a holdout, then split that equally into validation/test.
# This yields approximately 80/10/10 while preserving each class's proportions.
train_df, holdout_df = train_test_split(
    annotated_pool,
    test_size=0.20,
    random_state=RANDOM_SEED,
    stratify=annotated_pool[LABEL_COL],
)
val_df, final_test_df = train_test_split(
    holdout_df,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=holdout_df[LABEL_COL],
)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
final_test_df = final_test_df.reset_index(drop=True)

# Save the exact split used for reproducibility/auditing.
train_df.to_csv(DATA_DIR / "annotated_train_80.csv", index=False)
val_df.to_csv(DATA_DIR / "annotated_validation_10.csv", index=False)
final_test_df.to_csv(DATA_DIR / "annotated_test_10.csv", index=False)
print(f"\nTraining split (80%): {len(train_df):,}")
print(f"Validation split (10%): {len(val_df):,}")
print(f"Held-out test split (10%): {len(final_test_df):,}")
for name, split_df in [("Training", train_df), ("Validation", val_df), ("Held-out test", final_test_df)]:
    missing_classes = set(LABELS) - set(split_df[LABEL_COL].unique())
    if missing_classes:
        raise ValueError(f"{name} split is missing classes: {sorted(missing_classes)}")

# Load the actual source corpus; no 50k random sample is taken.
if not SPOTIFY_FILE.exists() and SPOTIFY_CACHE_FILE.exists():
    SPOTIFY_FILE = SPOTIFY_CACHE_FILE
if not SPOTIFY_FILE.exists():
    raise FileNotFoundError(
        f"Original Spotify dataset not found at {DATA_DIR / 'spotify_dataset.csv'} "
        f"or the configured KaggleHub cache path {SPOTIFY_CACHE_FILE}."
    )
spotify_df = pd.read_csv(SPOTIFY_FILE, low_memory=False)
spotify_df.columns = spotify_df.columns.str.strip()
rename_map = {}
if "song" in spotify_df.columns and "song_title" not in spotify_df.columns:
    rename_map["song"] = "song_title"
if "Artist(s)" in spotify_df.columns and "artist" not in spotify_df.columns:
    rename_map["Artist(s)"] = "artist"
if "text" in spotify_df.columns and "lyrics" not in spotify_df.columns:
    rename_map["text"] = "lyrics"
spotify_df = spotify_df.rename(columns=rename_map)
required_spotify = {"song_title", "artist", "lyrics"}
missing_spotify = required_spotify - set(spotify_df.columns)
if missing_spotify:
    raise ValueError(
        f"{SPOTIFY_FILE.name} is missing required columns after schema mapping: "
        f"{sorted(missing_spotify)}. Available columns: {spotify_df.columns.tolist()}"
    )
spotify_df = spotify_df.dropna(subset=["song_title", "artist", "lyrics"]).copy()
for col in ["song_title", "artist", "lyrics"]:
    spotify_df[col] = spotify_df[col].astype(str).str.strip()
spotify_df = spotify_df[
    (spotify_df["song_title"] != "") & (spotify_df["artist"] != "") & (spotify_df["lyrics"] != "")
].reset_index(drop=True)

# Exclude overlap with ALL labeled splits before inference, and deduplicate the
# source corpus itself by normalized title + artist.
spotify_df = add_match_keys(spotify_df)
labeled_keys = set()
for labeled_df in [train_df, val_df, final_test_df]:
    keyed = add_match_keys(labeled_df)
    labeled_keys.update(zip(keyed["_title_key"], keyed["_artist_key"]))
source_keys = list(zip(spotify_df["_title_key"], spotify_df["_artist_key"]))
not_labeled_mask = [key not in labeled_keys for key in source_keys]
removed_labeled_overlap = int(len(spotify_df) - sum(not_labeled_mask))
spotify_df = spotify_df.loc[not_labeled_mask].copy()
before_source_dedup = len(spotify_df)
spotify_df = spotify_df.drop_duplicates(subset=["_title_key", "_artist_key"], keep="first")
removed_source_duplicates = before_source_dedup - len(spotify_df)
spotify_df = spotify_df.drop(columns=["_title_key", "_artist_key"]).reset_index(drop=True)

print(f"Training songs: {len(train_df):,}")
print(f"Validation songs: {len(val_df):,}")
print(f"Held-out test songs: {len(final_test_df):,}")
print(f"Original Spotify rows remaining for pseudo-labeling: {len(spotify_df):,}")
print(f"Removed source rows matching labeled splits: {removed_labeled_overlap:,}")
print(f"Removed duplicate source title/artist pairs: {removed_source_duplicates:,}")
print("\nTraining distribution:")
print(train_df[LABEL_COL].value_counts())

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
# PSEUDO-LABEL THE FULL SPOTIFY CORPUS, THEN BUILD BALANCED 40K
# ============================================================

best_trainer = best_run["trainer"]
CONFIDENCE_THRESHOLD = best_run["threshold"]
all_result_chunks = []

print(f"\nPseudo-labeling {len(spotify_df):,} Spotify songs in chunks of {PREDICTION_CHUNK_SIZE:,}...")
for start_idx in range(0, len(spotify_df), PREDICTION_CHUNK_SIZE):
    end_idx = min(start_idx + PREDICTION_CHUNK_SIZE, len(spotify_df))
    chunk = spotify_df.iloc[start_idx:end_idx].reset_index(drop=True)
    chunk_dataset = make_dataset(chunk, include_labels=False)
    prediction_output = best_trainer.predict(chunk_dataset)
    probabilities = torch.softmax(torch.tensor(prediction_output.predictions), dim=-1).numpy()
    predicted_ids = np.argmax(probabilities, axis=1)
    confidence = probabilities.max(axis=1)

    chunk_results = chunk.copy()
    chunk_results["predicted_label"] = [ID2LABEL[int(i)] for i in predicted_ids]
    chunk_results["confidence"] = confidence
    for class_id, label in ID2LABEL.items():
        chunk_results[f"prob_{label}"] = probabilities[:, class_id]
    chunk_results[LABEL_COL] = chunk_results["predicted_label"]
    chunk_results["meets_confidence_threshold"] = confidence >= CONFIDENCE_THRESHOLD
    chunk_results["needs_revision"] = confidence < CONFIDENCE_THRESHOLD
    all_result_chunks.append(chunk_results)
    print(f"Predicted {end_idx:,}/{len(spotify_df):,}")

if all_result_chunks:
    results = pd.concat(all_result_chunks, ignore_index=True)
else:
    results = pd.DataFrame(columns=["song_title", "artist", "lyrics", "predicted_label", "confidence", LABEL_COL])

val_median_confidence = float(np.median(best_run["val_confidence"]))
pool_median_confidence = float(results["confidence"].median()) if len(results) else float("nan")
if len(results) and pool_median_confidence < val_median_confidence - 0.10:
    print(
        "WARNING: Median confidence on the Spotify corpus is more than 0.10 below "
        "validation. The selected threshold may not transfer reliably."
    )

# Save every prediction first, including low-confidence rows for auditing.
results.to_csv(OUTPUT_ALL_PSEUDO_LABELS, index=False)
print(f"\nAll pseudo-label predictions saved to: {OUTPUT_ALL_PSEUDO_LABELS}")
print(f"Songs processed: {len(results):,}")
print(f"Selected model seed: {best_run['seed']}")
print(f"Validation-selected confidence threshold: {CONFIDENCE_THRESHOLD:.2f}")
print("\nPredicted class distribution (all songs):")
print(results["predicted_label"].value_counts())
print("\nConfident class distribution:")
confident = results[results["meets_confidence_threshold"]].copy()
print(confident["predicted_label"].value_counts())
print("Rows below threshold:", int((~results["meets_confidence_threshold"]).sum()))
print("Mean confidence:", round(float(results["confidence"].mean()), 4) if len(results) else float("nan"))
print("Confidence quantiles:", results["confidence"].quantile([0, .25, .5, .75, .9, 1]).round(3).to_dict() if len(results) else {})
print(f"Validation median confidence: {val_median_confidence:.3f}")
print(f"Spotify median confidence: {pool_median_confidence:.3f}")

# Select the most confident qualifying songs in each predicted class. Never lower
# the threshold to fill quotas: fail clearly if a class has insufficient supply.
class_targets = {label: TARGET_PER_CLASS for label in LABELS}
class_targets["safe"] += 1  # 13,334 + 13,333 + 13,333 = exactly 40,000
selected_parts = []
shortages = {}
for label in LABELS:
    class_pool = confident[confident["predicted_label"] == label].sort_values(
        "confidence", ascending=False
    )
    target = class_targets[label]
    if len(class_pool) < target:
        shortages[label] = {"available": int(len(class_pool)), "required": int(target)}
    else:
        selected_parts.append(class_pool.head(target))

if shortages:
    raise RuntimeError(
        "Could not create the balanced 40k dataset without weakening the confidence "
        f"threshold. Per-class shortages: {shortages}. The full predictions are still "
        f"saved at {OUTPUT_ALL_PSEUDO_LABELS}. Increase the source pool or review the "
        "threshold-selection policy before retrying."
    )

balanced_40k = pd.concat(selected_parts, ignore_index=True)
balanced_40k = balanced_40k.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
balanced_40k.to_csv(OUTPUT_BALANCED_40K, index=False)

print("\nBALANCED DATASET CREATED")
print(f"Output: {OUTPUT_BALANCED_40K}")
print(f"Rows: {len(balanced_40k):,}")
print("Class distribution:")
print(balanced_40k[LABEL_COL].value_counts())
print("Minimum selected confidence by class:")
print(balanced_40k.groupby(LABEL_COL)["confidence"].min().round(4))
print("All selected rows meet the validation-selected confidence threshold.")
