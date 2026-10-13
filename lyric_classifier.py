import os
from pathlib import Path

import re

import random

import numpy as np

import pandas as pd

from tqdm import tqdm

import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, classification_report

from transformers import RobertaTokenizer, RobertaModel

from torch.optim import AdamW

from transformers import get_linear_schedule_with_warmup

from lime.lime_text import LimeTextExplainer

import shap

# Resolve input and output paths relative to this script, not the terminal's
# current working directory. Supports scripts at either the repository root or
# in a src/ subdirectory.
PROJECT_DIR = Path(__file__).resolve().parent

TRAIN_CSV_FILE = "Training.csv"
VAL_CSV_FILE = "Validation.csv"
TEST_CSV_FILE = "Testing.csv"


def resolve_dataset_path(filename):
    """Find a dataset file in the configured data directory or common locations."""
    candidates = [
        PROJECT_DIR / "data" / filename,
        PROJECT_DIR.parent / "data" / filename,
        PROJECT_DIR / filename,
        PROJECT_DIR.parent / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        f"Could not find {filename}. Checked these locations:\n{searched}\n"
        "Place the CSV in one of the listed locations or update resolve_dataset_path()."
    )

EXCLUDE_NEEDS_REVISION = True

MODEL_NAME = "roberta-base"

MAX_LENGTH = 512

BATCH_SIZE = 2

GRADIENT_ACCUMULATION_STEPS = 4

EPOCHS = 3

LEARNING_RATE = 2e-5

WARMUP_RATIO = 0.10

MAX_GRAD_NORM = 1.0

RANDOM_SEED = 67

MODEL_SAVE_PATH = PROJECT_DIR / "saved_model" / "lyric_classifier_v20.pt"
# Reuse an existing trained checkpoint by default. If v17 does not exist,
# the script will try the previous v16 and v15 checkpoints before training.
REUSE_EXISTING_MODEL = True
FORCE_RETRAIN = False
EXISTING_MODEL_PATHS = [
    MODEL_SAVE_PATH,
    PROJECT_DIR / "saved_model" / "lyric_classifier_v17.pt",
    PROJECT_DIR / "saved_model" / "lyric_classifier_v16.pt",
    PROJECT_DIR / "saved_model" / "lyric_classifier_v15.pt",
]

# Safe module-level configuration: selecting a device does not start training.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_NORMALIZATION = {
    "safe": "Safe",
    "mild": "Mild",
    "explicit": "Explicit",
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # More repeatable CUDA/cuDNN behavior; may reduce performance slightly.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

LABEL2ID = {
    "Safe": 0,
    "Mild": 1,
    "Explicit": 2
}

ID2LABEL = {
    0: "Safe",
    1: "Mild",
    2: "Explicit"
}

def clean_lyrics(text):
    """
    Basic lyric preprocessing.

    RoBERTa works well with natural language, so we avoid
    aggressive preprocessing such as removing stopwords
    or stemming words.
    """

    if not isinstance(text, str):
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse excessive blank lines
    text = re.sub(r"\n+", "\n", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text

def normalize_dataset_columns(df, dataset_name):
    """
    Normalize the incoming CSV schema to the internal columns used by this
    script. For the pseudo-labeled training pool, safety_label is the
    authoritative target; predicted_label is only a fallback if safety_label
    is absent. Unneeded probability/metadata columns are not passed to the model.
    """
    df = df.copy()

    # Prefer the V8 schema first, while still accepting older annotated schemas.
    aliases = {
        "song_name": ["song_title", "song_name", "track_name"],
        "artist": ["artist", "artists"],
        "text": ["lyrics", "text"],
        "explicitness": ["safety_label", "explicitness", "predicted_label", "Explicit", "explicit"],
    }

    rename_map = {}
    for target, candidates in aliases.items():
        found = next((col for col in candidates if col in df.columns), None)
        if found is None:
            raise ValueError(
                f"{dataset_name} is missing a column for '{target}'. "
                f"Accepted names: {candidates}. Found columns: {list(df.columns)}"
            )
        if found != target:
            rename_map[found] = target

    df = df.rename(columns=rename_map)

    # Keep only the fields needed by this classifier. Extra columns such as
    # confidence, probabilities, Explicit, and predicted_label are discarded
    # from this in-memory training/evaluation dataframe.
    needed = ["song_name", "artist", "text", "explicitness"]
    if "needs_revision" in df.columns:
        needed.append("needs_revision")
    return df[needed].copy()

class LyricsDataset(Dataset):

    def __init__(self, dataframe, tokenizer):
        self.dataframe = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        text = row["text"]
        label = row["label"]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)

        attention_mask = encoding[
            "attention_mask"
        ].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(
                label,
                dtype=torch.long
            )
        }

class RoBERTaCNNLSTM(nn.Module):

    def __init__(
        self,
        num_classes=3,
        lstm_hidden_size=128,
        lstm_layers=1,
        dropout=0.3
    ):

        super().__init__()

        # ----------------------------------------------------
        # RoBERTa
        # ----------------------------------------------------

        self.roberta = RobertaModel.from_pretrained(
            MODEL_NAME
        )

        roberta_hidden = (
            self.roberta.config.hidden_size
        )

        # ----------------------------------------------------
        # CNN
        # ----------------------------------------------------

        self.conv1 = nn.Conv1d(
            in_channels=roberta_hidden,
            out_channels=256,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv1d(
            in_channels=256,
            out_channels=256,
            kernel_size=5,
            padding=2
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(
            dropout
        )

        # ----------------------------------------------------
        # BiLSTM
        # ----------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
            dropout=(
                dropout
                if lstm_layers > 1
                else 0
            )
        )

        # ----------------------------------------------------
        # CLASSIFIER
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                lstm_hidden_size,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                128,
                num_classes
            )
        )


    def forward(
        self,
        input_ids,
        attention_mask
    ):

        # ----------------------------------------------------
        # RoBERTa
        # ----------------------------------------------------

        roberta_output = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        x = roberta_output.last_hidden_state

        # Shape:
        # [batch, sequence_length, 768]


        # ----------------------------------------------------
        # CNN
        # ----------------------------------------------------

        # Conv1D expects:
        # [batch, channels, sequence]

        x = x.transpose(1, 2)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Return to:
        # [batch, sequence, features]

        x = x.transpose(1, 2)


        # ----------------------------------------------------
        # BiLSTM
        # ----------------------------------------------------

        x, _ = self.lstm(x)


        # ----------------------------------------------------
        # MASKED GLOBAL MEAN POOLING
        # ----------------------------------------------------
        # Exclude padding positions from the pooled representation. Apply the
        # original token mask after the CNN/BiLSTM stack, where padding states
        # may no longer be zero.
        mask = attention_mask.unsqueeze(-1).to(dtype=x.dtype)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


        # ----------------------------------------------------
        # CLASSIFIER
        # ----------------------------------------------------

        logits = self.classifier(x)

        return logits

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scheduler
):

    model.train()

    total_loss = 0

    predictions = []
    actual = []

    progress = tqdm(
        loader,
        desc="Training"
    )
    optimizer.zero_grad(set_to_none=True)

    for batch_index, batch in enumerate(progress):

        input_ids = batch[
            "input_ids"
        ].to(DEVICE)

        attention_mask = batch[
            "attention_mask"
        ].to(DEVICE)

        labels = batch[
            "label"
        ].to(DEVICE)


        # Forward pass
        logits = model(
            input_ids,
            attention_mask
        )


        # Calculate loss
        loss = criterion(
            logits,
            labels
        )


        # Gradient accumulation: keep the effective batch size larger without
        # requiring the full batch to fit in VRAM at once.
        (loss / GRADIENT_ACCUMULATION_STEPS).backward()

        is_accumulation_boundary = (batch_index + 1) % GRADIENT_ACCUMULATION_STEPS == 0
        is_last_batch = (batch_index + 1) == len(loader)
        if is_accumulation_boundary or is_last_batch:
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()


        # Predictions
        preds = torch.argmax(
            logits,
            dim=1
        )


        predictions.extend(
            preds.detach()
            .cpu()
            .numpy()
        )

        actual.extend(
            labels.detach()
            .cpu()
            .numpy()
        )


        progress.set_postfix(
            loss=loss.item()
        )


    avg_loss = (
        total_loss / len(loader)
    )

    accuracy = accuracy_score(
        actual,
        predictions
    )

    return avg_loss, accuracy

def validate(
    model,
    loader,
    criterion
):
    # criterion is class-weighted CrossEntropyLoss, so this validation loss
    # is not directly comparable to unweighted-loss runs.
    model.eval()

    total_loss = 0

    predictions = []
    actual = []

    with torch.no_grad():

        for batch in tqdm(
            loader,
            desc="Validation"
        ):

            input_ids = batch[
                "input_ids"
            ].to(DEVICE)

            attention_mask = batch[
                "attention_mask"
            ].to(DEVICE)

            labels = batch[
                "label"
            ].to(DEVICE)


            logits = model(
                input_ids,
                attention_mask
            )


            loss = criterion(
                logits,
                labels
            )

            total_loss += loss.item()


            preds = torch.argmax(
                logits,
                dim=1
            )


            predictions.extend(
                preds.cpu().numpy()
            )

            actual.extend(
                labels.cpu().numpy()
            )


    avg_loss = (
        total_loss / len(loader)
    )

    accuracy = accuracy_score(
        actual,
        predictions
    )

    return (
        avg_loss,
        accuracy,
        actual,
        predictions
    )

def evaluate_test_set(
    model,
    loader
):

    model.eval()

    predictions = []
    actual = []

    with torch.no_grad():

        for batch in tqdm(
            loader,
            desc="Testing"
        ):

            input_ids = batch[
                "input_ids"
            ].to(DEVICE)

            attention_mask = batch[
                "attention_mask"
            ].to(DEVICE)

            labels = batch[
                "label"
            ].to(DEVICE)


            logits = model(
                input_ids,
                attention_mask
            )


            preds = torch.argmax(
                logits,
                dim=1
            )


            predictions.extend(
                preds.cpu().numpy()
            )

            actual.extend(
                labels.cpu().numpy()
            )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        actual,
        predictions
    )


    print("\n================================")
    print("FINAL TEST SET RESULTS")
    print("================================")

    print(
        f"\nTest Accuracy: {accuracy:.4f}"
    )


    print("\nClassification Report:")

    print(
        classification_report(
            actual,
            predictions,
            target_names=[
                "Safe",
                "Mild",
                "Explicit"
            ],
            zero_division=0
        )
    )


    return actual, predictions

def predict_lyrics(
    text,
    model,
    tokenizer
):

    model.eval()

    text = clean_lyrics(text)

    encoding = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    input_ids = encoding[
        "input_ids"
    ].to(DEVICE)

    attention_mask = encoding[
        "attention_mask"
    ].to(DEVICE)


    with torch.no_grad():

        logits = model(
            input_ids,
            attention_mask
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )


    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()

    confidence = probabilities[
        0,
        predicted_class
    ].item()


    return (
        ID2LABEL[predicted_class],
        confidence
    )

def model_predict_proba(texts, model, tokenizer):
    """Return class probabilities for a batch of texts using explicit dependencies."""
    if isinstance(texts, str):
        texts = [texts]
    cleaned_texts = [clean_lyrics(text) for text in texts]
    if not cleaned_texts:
        return np.empty((0, len(LABEL2ID)), dtype=np.float32)

    model.eval()
    # Chunk large explainer requests to avoid unnecessary VRAM spikes.
    inference_batch_size = 16 if DEVICE.type == "cuda" else 8
    all_probabilities = []

    with torch.no_grad():
        for start_idx in range(0, len(cleaned_texts), inference_batch_size):
            batch_texts = cleaned_texts[start_idx:start_idx + inference_batch_size]
            encoding = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encoding = {key: value.to(DEVICE) for key, value in encoding.items()}
            logits = model(**encoding)
            probabilities = torch.softmax(logits, dim=1)
            all_probabilities.append(probabilities.cpu().numpy())

    return np.concatenate(all_probabilities, axis=0)


def make_model_predict_proba(model, tokenizer):
    """Bind a model/tokenizer pair for explainers that expect a one-argument callable."""
    def predict_proba(texts):
        return model_predict_proba(texts, model, tokenizer)
    return predict_proba

def prepare_explanation_text(text, tokenizer):
    """Limit explainer input to the same token budget the classifier can process."""
    text = clean_lyrics(text)
    # Reserve room for RoBERTa's special tokens ([CLS]/[SEP]-equivalents).
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH - tokenizer.num_special_tokens_to_add(pair=False),
    )
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def is_punctuation_only(token):
    """True for explanation features containing no letters or digits."""
    return not any(character.isalnum() for character in str(token))


def explain_with_lime(text, model, tokenizer):

    print(
        "\n================================"
    )

    print(
        "LIME EXPLANATION"
    )

    print(
        "================================"
    )


    text = prepare_explanation_text(text, tokenizer)
    print(f"Explaining {len(tokenizer.encode(text, add_special_tokens=True))} RoBERTa tokens (maximum {MAX_LENGTH}).")
    predict_proba = make_model_predict_proba(model, tokenizer)
    explainer = LimeTextExplainer(
        class_names=[
            "Safe",
            "Mild",
            "Explicit"
        ]
    )


    probabilities = predict_proba([text])[0]
    predicted_class = int(np.argmax(probabilities))

    # Explicitly request the predicted class so LIME generates its explanation.
    explanation = explainer.explain_instance(
        text,
        predict_proba,
        labels=(predicted_class,),
        num_features=15,
        num_samples=1000
    )

    print(
        "\nPredicted class:",
        ID2LABEL[predicted_class]
    )

    print(
        "\nImportant words/phrases:"
    )

    # Defensive check: do not let a missing LIME class explanation crash the run.
    if predicted_class not in explanation.local_exp:
        print(
            f"LIME did not generate an explanation for "
            f"class {ID2LABEL[predicted_class]}."
        )
        print("Available explanation labels:", list(explanation.local_exp))
        return explanation

    for word, weight in explanation.as_list(label=predicted_class):
        direction = "supports" if weight > 0 else "opposes"
        print(f"{word:30s} {weight:+.4f} ({direction})")

    return explanation

def explain_with_shap(text, model, tokenizer):

    print(
        "\n================================"
    )

    print(
        "SHAP EXPLANATION"
    )

    print(
        "================================"
    )


    # --------------------------------------------------------
    # PREPARE A MODEL-LENGTH INPUT AND WORD-LEVEL MASKER
    # --------------------------------------------------------

    text = prepare_explanation_text(text, tokenizer)
    print(f"Explaining {len(tokenizer.encode(text, add_special_tokens=True))} RoBERTa tokens (maximum {MAX_LENGTH}).")
    predict_proba = make_model_predict_proba(model, tokenizer)

    # Explain whitespace-delimited text features rather than exposing every
    # RoBERTa subword fragment and punctuation mark as a separate feature.
    masker = shap.maskers.Text(r"\W+", collapse_mask_token=True)


    # --------------------------------------------------------
    # CREATE SHAP EXPLAINER
    # --------------------------------------------------------

    explainer = shap.Explainer(
        predict_proba,
        masker
    )


    # --------------------------------------------------------
    # GENERATE SHAP VALUES
    # --------------------------------------------------------

    shap_values = explainer(
        [text]
    )


    # --------------------------------------------------------
    # DETERMINE PREDICTED CLASS
    # --------------------------------------------------------

    probabilities = predict_proba([text])[0]


    predicted_class = int(np.argmax(
        probabilities
    ))


    print(
        "\nPredicted class:",
        ID2LABEL[predicted_class]
    )


    print(
        "\nSHAP feature contributions:"
    )


    # --------------------------------------------------------
    # EXTRACT VALUES
    # --------------------------------------------------------

    values = shap_values.values[0]
    tokens = shap_values.data[0]


    # SHAP can return:
    #
    # [tokens]
    #
    # or:
    #
    # [tokens, classes]

    if len(values.shape) > 1:

        class_values = values[
            :,
            predicted_class
        ]

    else:

        class_values = values


    # --------------------------------------------------------
    # RANK FEATURES
    # --------------------------------------------------------

    ranked_indices = np.argsort(
        np.abs(class_values)
    )[::-1]


    # --------------------------------------------------------
    # DISPLAY TOP FEATURES
    # --------------------------------------------------------

    displayed = 0
    for index in ranked_indices:
        token = str(tokens[index]).strip()
        value = float(class_values[index])

        # Punctuation-only features are not useful in this human-readable
        # summary. Their influence is not changed in the underlying SHAP values.
        if not token or is_punctuation_only(token):
            continue

        direction = "supports" if value > 0 else "opposes"
        print(f"{token:30s} {value:+.4f} ({direction})")
        displayed += 1
        if displayed >= 15:
            break

    if displayed == 0:
        print("No readable word-level features were returned for this explanation.")


    return shap_values


def main():
    """Run dataset preparation, training, evaluation, inference, and explanations."""
    set_seed(RANDOM_SEED)

    print("Using device:", DEVICE)

    print("\nLoading datasets...")

    train_path = resolve_dataset_path(TRAIN_CSV_FILE)
    val_path = resolve_dataset_path(VAL_CSV_FILE)
    test_path = resolve_dataset_path(TEST_CSV_FILE)

    print(f"Training data:   {train_path}")
    print(f"Validation data: {val_path}")
    print(f"Testing data:    {test_path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    train_df = normalize_dataset_columns(train_df, "Training dataset")

    val_df = normalize_dataset_columns(val_df, "Validation dataset")

    test_df = normalize_dataset_columns(test_df, "Test dataset")

    if EXCLUDE_NEEDS_REVISION:
        if "needs_revision" in train_df.columns:
            before = len(train_df)
            revision_values = train_df["needs_revision"].astype(str).str.strip().str.lower()
            train_df = train_df[~revision_values.isin(["true", "1", "yes", "y"])].copy()
            print(f"Excluded {before - len(train_df):,} low-confidence training rows flagged for revision.")
        else:
            print("WARNING: EXCLUDE_NEEDS_REVISION=True, but Training.csv has no 'needs_revision' column; no rows were filtered by this setting.")

    train_df["text"] = train_df["text"].apply(clean_lyrics)

    val_df["text"] = val_df["text"].apply(clean_lyrics)

    test_df["text"] = test_df["text"].apply(clean_lyrics)

    train_df = train_df[
        train_df["text"].str.len() > 0
    ].copy()

    val_df = val_df[
        val_df["text"].str.len() > 0
    ].copy()

    test_df = test_df[
        test_df["text"].str.len() > 0
    ].copy()


    for name, df in [("training", train_df), ("validation", val_df), ("test", test_df)]:
        df["explicitness"] = (
            df["explicitness"].astype(str).str.strip().str.lower()
            .map(LABEL_NORMALIZATION)
        )
        before = len(df)
        df.dropna(subset=["explicitness"], inplace=True)
        if len(df) != before:
            print(f"Removed {before - len(df):,} rows with missing/invalid labels from {name} data.")

    train_df["label"] = train_df["explicitness"].map(LABEL2ID)

    val_df["label"] = val_df["explicitness"].map(LABEL2ID)

    test_df["label"] = test_df["explicitness"].map(LABEL2ID)

    print("\nTraining songs:", len(train_df))

    print("Validation songs:", len(val_df))

    print("Test songs:", len(test_df))

    print("\nTraining class distribution:")

    print(train_df["explicitness"].value_counts())

    print("\nValidation class distribution:")

    print(val_df["explicitness"].value_counts())

    print("\nTest class distribution:")

    print(test_df["explicitness"].value_counts())

    print("\nLoading RoBERTa tokenizer...")

    tokenizer = RobertaTokenizer.from_pretrained(
        MODEL_NAME
    )

    train_dataset = LyricsDataset(
        train_df,
        tokenizer
    )

    val_dataset = LyricsDataset(
        val_df,
        tokenizer
    )

    test_dataset = LyricsDataset(
        test_df,
        tokenizer
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("\nLoading model...")

    model = RoBERTaCNNLSTM(
        num_classes=3
    )

    model = model.to(DEVICE)

    class_counts = train_df["label"].value_counts().reindex(range(len(LABEL2ID)), fill_value=0)

    if (class_counts == 0).any():
        missing = [ID2LABEL[i] for i, count in class_counts.items() if count == 0]
        raise ValueError(f"Training data has no examples for class(es): {missing}")

    class_weights = len(train_df) / (len(LABEL2ID) * class_counts.astype(float))

    class_weights_tensor = torch.tensor(class_weights.to_numpy(), dtype=torch.float32, device=DEVICE)

    print("\nClass weights:", {ID2LABEL[i]: round(float(class_weights_tensor[i].item()), 4) for i in range(len(LABEL2ID))})

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Check for a previously trained model BEFORE starting the training loop.
    # v15 and v16 use the same model architecture, so their state dictionaries
    # can be loaded into this version without retraining.
    checkpoint_to_load = None
    if REUSE_EXISTING_MODEL and not FORCE_RETRAIN:
        checkpoint_to_load = next(
            (path for path in EXISTING_MODEL_PATHS if path.is_file()),
            None,
        )

    if checkpoint_to_load is not None:
        print(f"\nReusing saved model: {checkpoint_to_load}")
        model.load_state_dict(
            torch.load(
                checkpoint_to_load,
                map_location=DEVICE,
                weights_only=True,
            )
        )
        model.eval()
        print("Saved model loaded. Skipping training.")
    else:
        if FORCE_RETRAIN:
            print("\nFORCE_RETRAIN=True: training a fresh model.")
        else:
            print("\nNo compatible saved checkpoint found. Training a new model.")

        optimizer = AdamW(
            model.parameters(),
            lr=LEARNING_RATE
        )

        optimizer_steps_per_epoch = (
            len(train_loader) + GRADIENT_ACCUMULATION_STEPS - 1
        ) // GRADIENT_ACCUMULATION_STEPS
        total_optimizer_steps = max(1, optimizer_steps_per_epoch * EPOCHS)
        warmup_steps = int(total_optimizer_steps * WARMUP_RATIO)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_optimizer_steps,
        )

        print(
            f"Scheduler: {warmup_steps} warmup steps / "
            f"{total_optimizer_steps} total optimizer steps; "
            f"gradient accumulation={GRADIENT_ACCUMULATION_STEPS}."
        )
        print("\nStarting training...\n")

        best_val_accuracy = float("-inf")

        for epoch in range(EPOCHS):
            print(f"\n========== Epoch {epoch + 1}/{EPOCHS} ==========")

            train_loss, train_accuracy = train_one_epoch(
                model, train_loader, optimizer, criterion, scheduler
            )
            val_loss, val_accuracy, val_actual, val_predictions = validate(
                model, val_loader, criterion
            )

            print(f"\nTrain Loss: {train_loss:.4f}")
            print(f"Train Accuracy: {train_accuracy:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.4f}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"Best model saved to {MODEL_SAVE_PATH}!")

        if not MODEL_SAVE_PATH.is_file():
            raise FileNotFoundError(
                f"Training finished but checkpoint was not saved: {MODEL_SAVE_PATH}"
            )

        model.load_state_dict(
            torch.load(
                MODEL_SAVE_PATH,
                map_location=DEVICE,
                weights_only=True,
            )
        )
        model.eval()
        print("\nBest model loaded.")

    (
        best_val_loss,
        best_val_accuracy,
        final_val_actual,
        final_val_predictions
    ) = validate(
        model,
        val_loader,
        criterion
    )

    print(
        "\n================================"
    )

    print(
        "FINAL VALIDATION REPORT"
    )

    print(
        "================================"
    )

    print(
        f"\nBest/Loaded Validation Accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    print(
        classification_report(
            final_val_actual,
            final_val_predictions,
            target_names=[
                "Safe",
                "Mild",
                "Explicit"
            ],
            zero_division=0
        )
    )

    test_actual, test_predictions = evaluate_test_set(
        model,
        test_loader
    )

    # Randomly select a labeled song from the validation set for the demo.
    # Use validation rather than test so the test set remains reserved for evaluation.
    if val_df.empty:
        print("Validation dataset has no usable labeled lyrics; skipping inference, LIME, and SHAP.")
        return

    selected_index = random.SystemRandom().choice(list(val_df.index))
    selected_song = val_df.loc[selected_index]
    example_lyrics = clean_lyrics(selected_song["text"])
    actual_label = str(selected_song["explicitness"])
    song_name = str(selected_song.get("song_name", "Unknown song"))
    artist_name = str(selected_song.get("artist", "Unknown artist"))

    if not example_lyrics:
        print("Randomly selected song has no usable lyrics; skipping inference, LIME, and SHAP.")
        return

    print("\nRandomly selected labeled song from Validation.csv")
    print(f"Song: {song_name}")
    print(f"Artist: {artist_name}")
    print(f"Actual label: {actual_label}")
    print(f"Lyrics preview: {example_lyrics[:300]}{'...' if len(example_lyrics) > 300 else ''}")

    prediction, confidence = predict_lyrics(
        example_lyrics,
        model,
        tokenizer
    )

    print(
        "\n================================"
    )

    print(
        "INFERENCE"
    )

    print(
        "================================"
    )

    print(
        "Prediction:",
        prediction
    )

    print(
        "Confidence:",
        f"{confidence * 100:.2f}%"
    )

    lime_explanation = explain_with_lime(
        example_lyrics,
        model,
        tokenizer
    )

    shap_explanation = explain_with_shap(
        example_lyrics,
        model,
        tokenizer
    )


if __name__ == "__main__":
    main()
