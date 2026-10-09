
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

TRAIN_FILE = "train.csv"
VALIDATION_FILE = "validation.csv"
UNLABELED_FILE = "unlabeled_songs.csv"

OUTPUT_DIR = "./distilbert_song_classifier"
OUTPUT_FILE = "songs_pseudo_labeled.csv"

TEXT_COL = "text"
LABEL_COL = "Explicit"
METADATA_COLS = ["Artist(s)", "song"]

LABELS = ["safe", "mild", "explicit"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

MAX_LENGTH = 512
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
CONFIDENCE_THRESHOLD = 0.70

set_seed(42)


# ============================================================
# LOAD AND VALIDATE FILES
# ============================================================

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VALIDATION_FILE)
unlabeled_df = pd.read_csv(UNLABELED_FILE)

for df in [train_df, val_df, unlabeled_df]:
    df.columns = df.columns.str.strip()

required_cols = METADATA_COLS + [TEXT_COL]

for name, df in [
    ("Training", train_df),
    ("Validation", val_df),
    ("Unlabeled", unlabeled_df),
]:
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(
            f"{name} dataset is missing columns: {missing}"
        )

for name, df in [
    ("Training", train_df),
    ("Validation", val_df),
]:
    if LABEL_COL not in df.columns:
        raise ValueError(
            f"{name} dataset must contain '{LABEL_COL}'."
        )

    df.dropna(subset=[TEXT_COL, LABEL_COL], inplace=True)

    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df[LABEL_COL] = (
        df[LABEL_COL].astype(str).str.strip().str.lower()
    )

    invalid = set(df[LABEL_COL].unique()) - set(LABELS)

    if invalid:
        raise ValueError(
            f"{name} has invalid labels: {invalid}"
        )

unlabeled_df.dropna(subset=[TEXT_COL], inplace=True)
unlabeled_df[TEXT_COL] = unlabeled_df[TEXT_COL].astype(str)

# Ensure empty lyrics do not silently become valid samples.
for name, df in [
    ("Training", train_df),
    ("Validation", val_df),
    ("Unlabeled", unlabeled_df),
]:
    df[TEXT_COL] = df[TEXT_COL].str.strip()
    df.drop(
        index=df.index[df[TEXT_COL].eq("")],
        inplace=True,
    )

print(f"Training songs: {len(train_df)}")
print(f"Validation songs: {len(val_df)}")
print(f"Unlabeled songs: {len(unlabeled_df)}")

print("\nTraining distribution:")
print(train_df[LABEL_COL].value_counts())

for name, df in [
    ("Training", train_df),
    ("Validation", val_df),
]:
    missing_classes = set(LABELS) - set(df[LABEL_COL].unique())
    if missing_classes:
        raise ValueError(
            f"{name} is missing classes: {missing_classes}"
        )


# ============================================================
# MAP LABELS TO INTEGER IDS
# ============================================================

train_df = train_df.copy()
val_df = val_df.copy()

train_df["labels"] = train_df[LABEL_COL].map(LABEL2ID)
val_df["labels"] = val_df[LABEL_COL].map(LABEL2ID)


# ============================================================
# LOAD DISTILBERT
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)


# ============================================================
# TOKENIZATION
# ============================================================

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

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# ============================================================
# EVALUATION METRICS
# ============================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels,
            predictions,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        ),
    }


# ============================================================
# TRAINING
# ============================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,

    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    warmup_ratio=0.1,

    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,

    fp16=torch.cuda.is_available(),
    report_to="none",
    save_total_limit=2,
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()


# ============================================================
# VALIDATION RESULTS
# ============================================================

print("\nValidation metrics:")
print(trainer.evaluate())

val_output = trainer.predict(val_dataset)
val_pred = np.argmax(val_output.predictions, axis=-1)

print("\nClassification report:")
print(classification_report(
    val_output.label_ids,
    val_pred,
    labels=[0, 1, 2],
    target_names=LABELS,
    zero_division=0,
))

print("\nConfusion matrix (true rows, predicted columns):")
print(confusion_matrix(
    val_output.label_ids,
    val_pred,
    labels=[0, 1, 2],
))


# ============================================================
# SAVE FINE-TUNED MODEL
# ============================================================

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)


# ============================================================
# PSEUDO-LABEL 40,000 SONGS
# ============================================================

unlabeled_dataset = make_dataset(
    unlabeled_df,
    include_labels=False,
)

prediction_output = trainer.predict(unlabeled_dataset)

probabilities = torch.softmax(
    torch.tensor(prediction_output.predictions),
    dim=-1,
).numpy()

predicted_ids = np.argmax(probabilities, axis=1)
confidence = probabilities.max(axis=1)

results = unlabeled_df.copy()

results[LABEL_COL] = [
    ID2LABEL[int(i)] for i in predicted_ids
]
results["confidence"] = confidence

for i, label in ID2LABEL.items():
    results[f"prob_{label}"] = probabilities[:, i]

results["needs_review"] = confidence < CONFIDENCE_THRESHOLD

# Keep original columns first, then add prediction metadata.
results.to_csv(OUTPUT_FILE, index=False)

print("\nPseudo-labeling completed!")
print(f"Output: {OUTPUT_FILE}")
print(f"Songs labeled: {len(results)}")

print("\nPredicted class distribution:")
print(results[LABEL_COL].value_counts())

print(
    "\nSongs flagged for review:",
    int(results["needs_review"].sum()),
)

print("Mean confidence:", round(float(confidence.mean()), 4))
