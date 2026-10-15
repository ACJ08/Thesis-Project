import os
import json
import hashlib
import platform
import unicodedata
import importlib.metadata
from datetime import datetime, timezone
from pathlib import Path

import re

import random

import numpy as np

import pandas as pd

from tqdm import tqdm

import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, precision_recall_fscore_support

from transformers import RobertaTokenizer, RobertaModel, BertTokenizer, BertModel, RobertaForSequenceClassification

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

EXCLUDE_NEEDS_REVISION = False

MODEL_NAME = "roberta-base"

MAX_LENGTH = 512

BATCH_SIZE = 2

GRADIENT_ACCUMULATION_STEPS = 4

# Maximum number of epochs; early stopping may finish sooner.
EPOCHS = 10
EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_MIN_DELTA = 0.001

LEARNING_RATE = 2e-5

WARMUP_RATIO = 0.10

MAX_GRAD_NORM = 1.0

RANDOM_SEED = 67

# Choose "all_models" (hybrid then all three baselines), "hybrid",
# "bert_cnn_lstm", "lstm_cnn", "roberta", "all_baselines", or "compare".
# "compare" only reads frozen test predictions; it never trains or touches raw test lyrics.
EXPERIMENT_MODE = "compare"
# Independent baseline smoke tests use SMOKE_TEST=True, just like the hybrid.
# Held-out testing is performed separately with FROZEN_TEST_EVALUATION=True.
FROZEN_TEST_EVALUATION = True
# Map each trained model to its checkpoint's run folder for frozen evaluation.
FROZEN_RUN_DIRS = {
    "hybrid": str(PROJECT_DIR / "saved_models" / "hybrid_v25_20261014T150657Z"),
    "bert_cnn_lstm": str(PROJECT_DIR / "saved_models" / "bert_cnn_lstm_v25_20261014T150657Z"),
    "lstm_cnn": str(PROJECT_DIR / "saved_models" / "lstm_cnn_v25_20261014T150657Z"),
    "roberta": str(PROJECT_DIR / "saved_models" / "roberta_v25_20261015T030347Z"),
}
# Baseline-specific hyperparameters; keep and report the chosen settings.
BASELINE_LSTM_EMBED_DIM = 256
BASELINE_LSTM_HIDDEN = 128
BASELINE_VOCAB_LIMIT = 30000
BASELINE_VOCAB_MIN_FREQ = 2
BASELINE_LEARNING_RATE = 2e-5  # transformer baselines
BASELINE_LSTM_LEARNING_RATE = 1e-3

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = PROJECT_DIR / "experiments" / f"hybrid_v25_{RUN_ID}"
MODEL_SAVE_PATH = RUN_DIR / "best_model.pt"
RUN_FINAL_TEST = False  # Only enable after all model/hyperparameter decisions are frozen.
RUN_DEMO_EXPLANATIONS = False  # SHAP can be extremely expensive.
RUN_PADDING_INVARIANCE_CHECK = True
PADDING_TOLERANCE = 1e-4  # diagnostic reference, not a guarantee of semantic equivalence
PADDING_HARD_FAIL_AFTER_TRAINING = False  # turn on after inspecting actual differences
SMOKE_TEST = False  # True: small stratified engineering run, NOT thesis results
SMOKE_TRAIN_ROWS = 500
SMOKE_VAL_ROWS = 100
# Setting EXCLUDE_NEEDS_REVISION=False intentionally includes UNREVIEWED flagged rows.
# This is not evidence that their pseudo-labels were human-verified.
CHECK_SPLIT_OVERLAP = True
# v21 is a fresh experiment with early stopping. Do not reuse v20 or older weights.

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

STRUCTURE_TAG_PATTERN = re.compile(
    r"\[(?:verse|chorus|pre[- ]?chorus|post[- ]?chorus|bridge|intro|outro|hook|refrain|interlude|instrumental|break|solo|repeat|spoken|sample|drop|ending|part)(?:\s*[:#-]?\s*[^\]\n]*)?\]",
    flags=re.IGNORECASE,
)


def normalize_overlap_lyrics(text):
    """Broad, conservative lyric identity check; does not establish semantic near-duplicates."""
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    text = STRUCTURE_TAG_PATTERN.sub(" ", text)
    return " ".join(re.findall(r"[^\W_]+", text, flags=re.UNICODE))


def normalize_identity(text):
    return "".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", str(text)).casefold()))


def clean_lyrics(text):
    """
    Basic lyric preprocessing.

    RoBERTa works well with natural language, so we avoid
    aggressive preprocessing such as removing stopwords
    or stemming words.
    """

    if not isinstance(text, str):
        return ""

    # Remove structural headers consistently in training, inference, and XAI.
    text = STRUCTURE_TAG_PATTERN.sub(" ", text)

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
    # Preserve provenance columns when supplied; do not infer verification from a label.
    if dataset_name == "Training dataset":
        needed.extend(col for col in ("human_label", "label_source", "review_status", "reviewed_by", "annotator") if col in df.columns)
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
        # LSTM
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

        # Zero padded encoder states before and between biased convolutions.
        # This prevents padding activations from feeding real token features.
        sequence_mask = attention_mask.unsqueeze(1).to(dtype=x.dtype)
        x = x * sequence_mask
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x) * sequence_mask

        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x) * sequence_mask
        x = x.transpose(1, 2)

        # Pack valid tokens so the LSTM does not process padding positions.
        lengths = attention_mask.sum(dim=1).to(dtype=torch.long).cpu()
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed)
        x, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=input_ids.size(1))

        # Masked mean pooling AFTER the LSTM (not max pooling before it).
        mask = attention_mask.unsqueeze(-1).to(dtype=x.dtype)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


        # ----------------------------------------------------
        # CLASSIFIER
        # ----------------------------------------------------

        logits = self.classifier(x)

        return logits

def train_one_epoch(model, loader, optimizer, criterion, scheduler):
    """Exact class-weighted effective-batch loss with gradient accumulation.

    CrossEntropyLoss(reduction='sum') accumulates weighted loss numerators;
    gradients are divided by the total target-weight sum of each accumulation
    group immediately before clipping and optimizer.step(). This handles the
    final partial group without underweighting it.
    """
    model.train()
    predictions, actual = [], []
    weighted_loss_sum, weight_sum = 0.0, 0.0
    group_weight_sum = 0.0
    optimizer.zero_grad(set_to_none=True)
    weights = criterion.weight
    for batch_index, batch in enumerate(tqdm(loader, desc="Training")):
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        logits = model(ids, mask)
        loss_sum = torch.nn.functional.cross_entropy(logits, labels, weight=weights, reduction="sum")
        loss_sum.backward()
        batch_weight = float(weights[labels].sum().item()) if weights is not None else float(labels.numel())
        group_weight_sum += batch_weight
        weighted_loss_sum += float(loss_sum.detach().item())
        weight_sum += batch_weight
        predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())
        actual.extend(labels.cpu().tolist())
        at_boundary = (batch_index + 1) % GRADIENT_ACCUMULATION_STEPS == 0
        if at_boundary or batch_index + 1 == len(loader):
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(group_weight_sum)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            group_weight_sum = 0.0
    return weighted_loss_sum / max(weight_sum, 1e-12), accuracy_score(actual, predictions)


def validate(model, loader, criterion):
    model.eval()
    actual, predictions = [], []
    weighted_loss_sum, weight_sum = 0.0, 0.0
    weights = criterion.weight
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            logits = model(ids, mask)
            weighted_loss_sum += float(torch.nn.functional.cross_entropy(
                logits, labels, weight=weights, reduction="sum").item())
            weight_sum += float(weights[labels].sum().item()) if weights is not None else float(labels.numel())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            actual.extend(labels.cpu().tolist())
    return weighted_loss_sum / max(weight_sum, 1e-12), accuracy_score(actual, predictions), actual, predictions


def save_predictions_and_metrics(name, dataframe, actual, predicted):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    output = dataframe[["song_name", "artist", "text", "explicitness"]].copy().reset_index(drop=True)
    output["actual_id"] = actual
    output["predicted_id"] = predicted
    output["predicted_label"] = [ID2LABEL[int(i)] for i in predicted]
    output["correct"] = output["actual_id"] == output["predicted_id"]
    output.to_csv(RUN_DIR / f"{name}_predictions.csv", index=False)
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=[0, 1, 2], zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=[0, 1, 2], average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(actual, predicted, labels=[0, 1, 2], average="weighted", zero_division=0)),
        "per_class": {ID2LABEL[i]: {"precision": float(precision[i]), "recall": float(recall[i]),
                                    "f1": float(f1[i]), "support": int(support[i])} for i in range(3)},
        "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1, 2]).tolist(),
        "directional_errors": {f"{ID2LABEL[i]}_to_{ID2LABEL[j]}": int(sum(
            a == i and b == j for a, b in zip(actual, predicted)))
            for i in range(3) for j in range(3) if i != j},
    }
    (RUN_DIR / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def installed_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_disjoint_splits(train, validation, test):
    """Detect exact normalized-lyrics and title/artist overlap across partitions."""
    def identities(frame):
        title_artist = set()
        lyrics = set()
        for row in frame.itertuples(index=False):
            title_artist.add((normalize_identity(row.song_name), normalize_identity(row.artist)))
            lyrics.add(normalize_overlap_lyrics(row.text))
        return title_artist, lyrics
    entries = [("training", train), ("validation", validation), ("test", test)]
    keys = {name: identities(frame) for name, frame in entries}
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            a, b = entries[i][0], entries[j][0]
            titles = keys[a][0] & keys[b][0]
            texts = (keys[a][1] & keys[b][1]) - {""}
            if titles or texts:
                raise ValueError(f"Partition overlap: {a}/{b}: {len(titles)} song identities, "
                                 f"{len(texts)} normalized lyric texts. Resolve overlaps before training.")


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
        padding="max_length",
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
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encoding = {key: value.to(DEVICE) for key, value in encoding.items()}
            logits = model(input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"])
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


def padding_invariance_check(model, tokenizer, val_df, *, phase, hard_fail=False):
    """Measure fixed/dynamic/mixed-batch prediction changes on 3 lyric lengths.

    Results are diagnostics; every measurement and argmax flip is retained.
    """
    if val_df.empty:
        raise ValueError("Cannot run padding check without validation songs")
    token_lengths = [(i, len(tokenizer.encode(t, truncation=True, max_length=MAX_LENGTH)))
                     for i, t in enumerate(val_df["text"].tolist()) if t]
    if not token_lengths:
        raise ValueError("No nonempty validation lyrics for padding check")
    ordered = sorted(token_lengths, key=lambda item: item[1])
    candidates = {"short": ordered[0], "median": ordered[len(ordered)//2],
                  "near_limit": min(ordered, key=lambda item: abs(item[1] - MAX_LENGTH))}
    results = []
    model.eval()
    with torch.inference_mode():
        for label, (idx, token_count) in candidates.items():
            text = val_df.iloc[idx]["text"]
            # A longer companion forces extra right-padding in a mixed batch.
            companion = " unrelated filler" * (MAX_LENGTH * 2)
            fixed = tokenizer(text, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            dynamic = tokenizer(text, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            mixed = tokenizer([text, companion], padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            def predict(enc):
                logits = model(input_ids=enc["input_ids"].to(DEVICE),
                               attention_mask=enc["attention_mask"].to(DEVICE))
                return torch.softmax(logits, dim=-1)[0].cpu().tolist()
            base, dyn, mix = predict(fixed), predict(dynamic), predict(mixed)
            base_class = int(np.argmax(base))
            max_diff = max(max(abs(x-y) for x,y in zip(base, dyn)),
                           max(abs(x-y) for x,y in zip(base, mix)))
            flipped = bool(base_class != int(np.argmax(dyn)) or base_class != int(np.argmax(mix)))
            results.append({"case": label, "validation_row_position": idx, "token_count": token_count,
                            "fixed_probabilities": base, "dynamic_probabilities": dyn,
                            "mixed_probabilities": mix, "max_absolute_difference": max_diff,
                            "argmax_flip": flipped, "exceeds_reference_tolerance": max_diff > PADDING_TOLERANCE})
    report = {"phase": phase, "reference_tolerance": PADDING_TOLERANCE,
              "hard_fail": hard_fail, "max_absolute_difference": max(r["max_absolute_difference"] for r in results),
              "any_argmax_flip": any(r["argmax_flip"] for r in results), "cases": results}
    report["reference_pass"] = not report["any_argmax_flip"] and report["max_absolute_difference"] <= PADDING_TOLERANCE
    path = RUN_DIR / f"padding_invariance_{phase}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Padding {phase}: max difference={report['max_absolute_difference']:.7g}; "
          f"argmax flip={report['any_argmax_flip']}; report={path}")
    if not report["reference_pass"]:
        print("WARNING: padding invariance diagnostic exceeded reference tolerance or changed predicted class.")
        if hard_fail:
            raise RuntimeError(f"Padding invariance failed in {phase}; see {path}")
    return report


def stratified_smoke_subset(df, count, name):
    """Deterministic small sample retaining every represented class."""
    if len(df) <= count:
        return df.copy().reset_index(drop=True)
    if count < df["label"].nunique():
        raise ValueError(f"{name} smoke sample too small to represent all classes")
    from sklearn.model_selection import train_test_split
    subset, _ = train_test_split(df, train_size=count, stratify=df["label"], random_state=RANDOM_SEED)
    return subset.reset_index(drop=True)


def provenance_report(train_df, *, raw_rows, flagged_raw, filename="training_provenance.json"):
    """Count observed provenance metadata without claiming pseudo-labels are verified."""
    report = {"raw_rows": int(raw_rows), "flagged_before_filter": int(flagged_raw),
              "retained_rows": int(len(train_df)), "exclude_needs_revision": EXCLUDE_NEEDS_REVISION}
    for col in ("needs_revision", "human_label", "label_source", "review_status", "reviewed_by", "annotator"):
        if col not in train_df:
            report[col] = {"available": False}
            continue
        series = train_df[col]
        values = series.dropna().astype(str).str.strip()
        nonempty = values[values.ne("")]
        entry = {"available": True, "nonempty_count": int(len(nonempty))}
        if col in ("needs_revision", "label_source", "review_status"):
            entry["values"] = {str(k): int(v) for k, v in nonempty.str.lower().value_counts().head(20).items()}
        report[col] = entry
    report["human_verification_confirmed"] = False  # requires schema-specific audit
    (RUN_DIR / filename).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Provenance report:", RUN_DIR / filename)
    print("IMPORTANT: Presence of human-label metadata alone does not prove labels were reviewed.")
    return report

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

    train_df = pd.read_csv(train_path, low_memory=False)
    val_df = pd.read_csv(val_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)

    train_df = normalize_dataset_columns(train_df, "Training dataset")

    val_df = normalize_dataset_columns(val_df, "Validation dataset")

    test_df = normalize_dataset_columns(test_df, "Test dataset")

    raw_training_rows = len(train_df)
    flagged_rows = 0
    if "needs_revision" in train_df.columns:
        revision_values = train_df["needs_revision"].astype(str).str.strip().str.lower()
        flagged_rows = int(revision_values.isin(["true", "1", "yes", "y"]).sum())
    if not EXCLUDE_NEEDS_REVISION and flagged_rows:
        print(f"WARNING: Including {flagged_rows:,} rows still flagged needs_revision. "
              "Their labels may be unverified pseudo-labels; this is a deliberate choice.")

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

    if CHECK_SPLIT_OVERLAP:
        assert_disjoint_splits(train_df, val_df, test_df)
        print("Partition identity and normalized-text overlap checks passed.")
    # Preserve full cleaned/validated training population provenance before smoke sampling.
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    full_provenance = provenance_report(
        train_df, raw_rows=raw_training_rows, flagged_raw=flagged_rows,
        filename="training_provenance.json")

    # Check overlap on the FULL partitions before reducing for a smoke test.
    if SMOKE_TEST:
        print("\n*** SMOKE TEST: engineering check only; results are NOT thesis metrics ***")
        train_df = stratified_smoke_subset(train_df, SMOKE_TRAIN_ROWS, "training")
        val_df = stratified_smoke_subset(val_df, SMOKE_VAL_ROWS, "validation")
        print(f"Smoke subset: {len(train_df)} training, {len(val_df)} validation rows")

    smoke_provenance = None
    if SMOKE_TEST:
        subset_flagged = 0
        if "needs_revision" in train_df.columns:
            subset_flagged = int(train_df["needs_revision"].astype(str).str.strip().str.lower().isin(
                ["true", "1", "yes", "y"]).sum())
        smoke_provenance = provenance_report(
            train_df, raw_rows=len(train_df), flagged_raw=subset_flagged,
            filename="smoke_training_provenance.json")
    manifest = {
        "run_id": RUN_ID, "smoke_test": SMOKE_TEST, "smoke_train_target": SMOKE_TRAIN_ROWS if SMOKE_TEST else None, "smoke_val_target": SMOKE_VAL_ROWS if SMOKE_TEST else None, "architecture": "RoBERTa-CNN-unidirectional LSTM-masked mean pooling",
        "max_length": MAX_LENGTH, "seed": RANDOM_SEED, "epochs_max": EPOCHS,
        "batch_size": BATCH_SIZE, "accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE, "class_weighting": "inverse_frequency", "exclude_needs_revision": EXCLUDE_NEEDS_REVISION,
        "preprocessing": "NFKC broad overlap audit; structural lyric section headers removed from model input",
        "input_policy": "first 512 RoBERTa tokens including special tokens; not full-song coverage",
        "input_max_length": MAX_LENGTH,
        "pooling": "masked mean after unidirectional LSTM",
        "run_padding_invariance_check": RUN_PADDING_INVARIANCE_CHECK, "padding_hard_fail_after_training": PADDING_HARD_FAIL_AFTER_TRAINING, "padding_reference_tolerance": PADDING_TOLERANCE, "provenance": full_provenance, "smoke_subset_provenance": smoke_provenance,
        "gradient_objective": "sum weighted cross-entropy over each effective batch / sum target class weights",
        "training_rows": len(train_df), "validation_rows": len(val_df), "test_rows": len(test_df),
        "training_rows_before_review_filter": raw_training_rows,
        "training_rows_flagged_needs_revision": flagged_rows,
        "input_sha256": {"training": file_sha256(train_path), "validation": file_sha256(val_path),
                         "test": file_sha256(test_path)},
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers_model": MODEL_NAME, "run_final_test": RUN_FINAL_TEST and not SMOKE_TEST,
        "dependencies": {name: installed_version(name) for name in ("torch", "transformers", "pandas", "numpy", "scikit-learn", "shap", "lime")},
    }
    (RUN_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

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

    manifest["class_weights"] = {ID2LABEL[i]: float(class_weights_tensor[i].item()) for i in range(len(LABEL2ID))}
    manifest["training_class_counts"] = {ID2LABEL[i]: int(class_counts[i]) for i in range(len(LABEL2ID))}
    print("\nClass weights:", {k: round(v, 4) for k, v in manifest["class_weights"].items()})

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    if RUN_PADDING_INVARIANCE_CHECK:
        manifest["padding_pretrain"] = padding_invariance_check(model, tokenizer, val_df, phase="pretrain", hard_fail=False)
        (RUN_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Always retrain: corrected preprocessing and masking change model behavior.
    print("\nFORCE_RETRAIN=True: training a fresh v25 model with early stopping.")

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    optimizer_steps_per_epoch = (
        len(train_loader) + GRADIENT_ACCUMULATION_STEPS - 1
    ) // GRADIENT_ACCUMULATION_STEPS
    total_optimizer_steps = max(1, optimizer_steps_per_epoch * (1 if SMOKE_TEST else EPOCHS))
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
    print(
        f"Early stopping: monitor=validation macro-F1, "
        f"patience={EARLY_STOPPING_PATIENCE}, "
        f"minimum improvement={EARLY_STOPPING_MIN_DELTA:.3f}; "
        f"maximum epochs={EPOCHS}."
    )
    print("\nStarting training...\n")

    best_val_macro_f1 = float("-inf")
    best_epoch = 0
    epoch_history = []
    epochs_without_improvement = 0

    for epoch in range(1 if SMOKE_TEST else EPOCHS):
        print(f"\n========== Epoch {epoch + 1}/{1 if SMOKE_TEST else EPOCHS} ==========")

        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, optimizer, criterion, scheduler
        )
        val_loss, val_accuracy, val_actual, val_predictions = validate(
            model, val_loader, criterion
        )
        val_macro_f1 = f1_score(
            val_actual,
            val_predictions,
            average="macro",
            zero_division=0,
        )

        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation Macro-F1: {val_macro_f1:.4f}")

        epoch_history.append({"epoch": epoch + 1, "train_loss": train_loss,
                              "train_accuracy": train_accuracy, "validation_loss": val_loss,
                              "validation_accuracy": val_accuracy, "validation_macro_f1": val_macro_f1})
        pd.DataFrame(epoch_history).to_csv(RUN_DIR / "epoch_history.csv", index=False)

        if val_macro_f1 > best_val_macro_f1 + EARLY_STOPPING_MIN_DELTA:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(
                f"Best v25 model saved to {MODEL_SAVE_PATH} "
                f"(epoch {best_epoch}, validation macro-F1={best_val_macro_f1:.4f})."
            )
        else:
            epochs_without_improvement += 1
            print(
                f"No qualifying macro-F1 improvement for "
                f"{epochs_without_improvement}/{EARLY_STOPPING_PATIENCE} epoch(s)."
            )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"\nEarly stopping triggered after epoch {epoch + 1}. "
                f"Best epoch: {best_epoch} "
                f"(validation macro-F1={best_val_macro_f1:.4f})."
            )
            break

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
    print(f"\nBest v25 checkpoint loaded from epoch {best_epoch}.")

    (
        final_val_loss,
        final_val_accuracy,
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
        f"\nLoaded Checkpoint Validation Accuracy: "
        f"{final_val_accuracy:.4f}"
    )
    print(f"Loaded Checkpoint Validation Loss: {final_val_loss:.4f}")
    print(
        "Loaded Checkpoint Validation Macro-F1: "
        f"{f1_score(final_val_actual, final_val_predictions, average='macro', zero_division=0):.4f}"
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

    validation_metrics = save_predictions_and_metrics("validation", val_df, final_val_actual, final_val_predictions)
    print("Saved validation metrics and directional errors:", RUN_DIR / "validation_metrics.json")
    if RUN_FINAL_TEST and not SMOKE_TEST:
        test_actual, test_predictions = evaluate_test_set(model, test_loader)
        save_predictions_and_metrics("test", test_df, test_actual, test_predictions)
    else:
        print("Final test evaluation skipped (smoke test or RUN_FINAL_TEST=False).")
    if RUN_PADDING_INVARIANCE_CHECK:
        manifest["padding_posttrain"] = padding_invariance_check(
            model, tokenizer, val_df, phase="posttrain", hard_fail=PADDING_HARD_FAIL_AFTER_TRAINING)
    manifest["optimizer_steps_planned"] = total_optimizer_steps
    manifest["warmup_steps_planned"] = warmup_steps
    manifest["optimizer_steps_completed"] = len(epoch_history) * optimizer_steps_per_epoch
    manifest["checkpoint_sha256"] = file_sha256(MODEL_SAVE_PATH)
    manifest["validation_macro_f1"] = validation_metrics["macro_f1"]
    manifest["best_epoch"] = best_epoch
    manifest["best_validation_macro_f1"] = best_val_macro_f1
    (RUN_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if SMOKE_TEST or not RUN_DEMO_EXPLANATIONS:
        print("Explanation demo skipped; enable RUN_DEMO_EXPLANATIONS when needed.")
        return

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





# ========================== v25 BASELINE COMPARISONS ==========================
# All baseline data preparation is shared with the hybrid, including exact
# clean_lyrics(), split overlap checks, class IDs and flagged-row policy.
from collections import Counter
from sklearn.model_selection import train_test_split

BASELINE_NAMES = ("bert_cnn_lstm", "lstm_cnn", "roberta")
ALL_NAMES = ("hybrid",) + BASELINE_NAMES


class BertCNNLSTM(nn.Module):
    """BERT backbone, with the hybrid's masked CNN -> packed LSTM head."""
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.conv1 = nn.Conv1d(self.bert.config.hidden_size, 256, 3, padding=1)
        self.conv2 = nn.Conv1d(256, 256, 5, padding=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(256, 128, batch_first=True, bidirectional=False)
        self.classifier = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 3))

    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(1).to(x.dtype)
        x = x.transpose(1, 2) * mask
        x = self.dropout(self.relu(self.conv1(x))) * mask
        x = self.dropout(self.relu(self.conv2(x))) * mask
        x = x.transpose(1, 2)
        lengths = attention_mask.sum(1).long().cpu()
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        x, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=input_ids.size(1))
        mask = attention_mask.unsqueeze(-1).to(x.dtype)
        x = (x * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.classifier(x)


class StandaloneRoberta(nn.Module):
    """Standard pretrained RoBERTa with its built-in classification head."""
    def __init__(self):
        super().__init__()
        self.encoder = RobertaForSequenceClassification.from_pretrained(
            "roberta-base", num_labels=3,
            id2label={int(k): v for k, v in ID2LABEL.items()},
            label2id=LABEL2ID,
        )

    def forward(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).logits


class LearnedVocabulary:
    """Train-only whitespace/punctuation tokenizer; 0=PAD, 1=UNK."""
    def __init__(self, token_to_id):
        self.token_to_id = token_to_id

    @staticmethod
    def tokens(text):
        return re.findall(r"\w+|[^\w\s]", str(text).casefold(), flags=re.UNICODE)

    @classmethod
    def fit(cls, texts):
        counts = Counter(token for text in texts for token in cls.tokens(text))
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for token, freq in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            if freq < BASELINE_VOCAB_MIN_FREQ or len(vocab) >= BASELINE_VOCAB_LIMIT:
                break
            vocab[token] = len(vocab)
        return cls(vocab)

    def encode(self, text, max_length=None):
        tokens = self.tokens(text)
        if max_length is not None:
            if max_length < 1:
                raise ValueError("max_length must be positive")
            tokens = tokens[:max_length]
        return [self.token_to_id.get(t, 1) for t in tokens] or [1]

    def num_special_tokens_to_add(self, pair=False):
        return 0

    def __call__(self, text, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt"):
        if return_tensors != "pt":
            raise ValueError("LearnedVocabulary supports return_tensors='pt' only")
        texts = text if isinstance(text, (list, tuple)) else [text]
        if not texts:
            raise ValueError("Cannot encode an empty batch")
        if max_length < 1:
            raise ValueError("max_length must be positive")
        sequences = [self.encode(item, max_length=max_length if truncation else None) for item in texts]
        if not truncation and any(len(seq) > max_length for seq in sequences):
            raise ValueError("Sequence exceeds max_length with truncation disabled")
        if padding == "max_length":
            target_length = max_length
        elif padding is True or padding == "longest":
            target_length = max(map(len, sequences))
        elif padding is False or padding is None:
            lengths = {len(seq) for seq in sequences}
            if len(lengths) != 1:
                raise ValueError("Unequal lengths require padding for tensor batches")
            target_length = len(sequences[0])
        else:
            raise ValueError(f"Unsupported padding option: {padding!r}")
        ids_batch, mask_batch = [], []
        for seq in sequences:
            pad_count = target_length - len(seq)
            ids_batch.append(seq + [0] * pad_count)
            mask_batch.append([1] * len(seq) + [0] * pad_count)
        return {"input_ids": torch.tensor(ids_batch, dtype=torch.long),
                "attention_mask": torch.tensor(mask_batch, dtype=torch.long)}


class LSTMCNN(nn.Module):
    """Learned embeddings -> packed unidirectional LSTM -> masked CNN -> pooling."""
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, BASELINE_LSTM_EMBED_DIM, padding_idx=0)
        self.lstm = nn.LSTM(BASELINE_LSTM_EMBED_DIM, BASELINE_LSTM_HIDDEN, batch_first=True)
        self.conv1 = nn.Conv1d(BASELINE_LSTM_HIDDEN, 256, 3, padding=1)
        self.conv2 = nn.Conv1d(256, 256, 5, padding=2)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 3))

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)
        lengths = attention_mask.sum(1).long().cpu()
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        x, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=input_ids.size(1))
        mask = attention_mask.unsqueeze(1).to(x.dtype)
        x = x.transpose(1, 2) * mask
        x = self.dropout(torch.relu(self.conv1(x))) * mask
        x = self.dropout(torch.relu(self.conv2(x))) * mask
        x = x.transpose(1, 2)
        mask = attention_mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.classifier(pooled)


def prepare_shared_data():
    paths = {name: resolve_dataset_path(filename) for name, filename in (
        ("training", TRAIN_CSV_FILE), ("validation", VAL_CSV_FILE), ("test", TEST_CSV_FILE))}
    frames = {name: normalize_dataset_columns(pd.read_csv(path, low_memory=False),
               "Training dataset" if name == "training" else name.title() + " dataset")
              for name, path in paths.items()}
    raw_rows = len(frames["training"])
    flags = frames["training"].get("needs_revision")
    flagged = int(flags.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"]).sum()) if flags is not None else 0
    if EXCLUDE_NEEDS_REVISION and flags is not None:
        frames["training"] = frames["training"].loc[~flags.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])].copy()
    elif flagged:
        print(f"WARNING: Including {flagged} needs_revision training rows, as configured.")
    for name, df in frames.items():
        df["text"] = df["text"].apply(clean_lyrics)
        df["explicitness"] = df["explicitness"].astype(str).str.strip().str.lower().map(LABEL_NORMALIZATION)
        before = len(df)
        df.dropna(subset=["explicitness"], inplace=True)
        df.drop(df[df["text"].str.len() == 0].index, inplace=True)
        if len(df) != before:
            print(f"{name}: removed {before-len(df)} invalid/empty rows")
        df["label"] = df["explicitness"].map(LABEL2ID).astype(int)
        frames[name] = df.reset_index(drop=True)
    if CHECK_SPLIT_OVERLAP:
        assert_disjoint_splits(frames["training"], frames["validation"], frames["test"])
        print("Full partition overlap audit passed.")
    return frames, paths, raw_rows, flagged


def make_baseline(name, vocab=None):
    if name == "bert_cnn_lstm":
        return BertCNNLSTM()
    if name == "roberta":
        return StandaloneRoberta()
    if name == "lstm_cnn":
        if vocab is None:
            raise ValueError("LSTM-CNN requires the training-only vocabulary")
        return LSTMCNN(len(vocab.token_to_id))
    if name == "hybrid":
        return RoBERTaCNNLSTM()
    raise ValueError(name)


def tokenizer_for(name, train_df, saved_dir=None):
    if name == "lstm_cnn":
        if saved_dir is not None:
            vocab_path = saved_dir / "vocabulary.json"
            if not vocab_path.is_file():
                raise FileNotFoundError(vocab_path)
            return LearnedVocabulary(json.loads(vocab_path.read_text(encoding="utf-8")))
        return LearnedVocabulary.fit(train_df["text"])
    if name == "bert_cnn_lstm":
        return BertTokenizer.from_pretrained("bert-base-uncased")
    return RobertaTokenizer.from_pretrained("roberta-base")


_baseline_manifest = None

def write_baseline_manifest():
    (RUN_DIR / "manifest.json").write_text(json.dumps(_baseline_manifest, indent=2), encoding="utf-8")


def run_baseline(name):
    global RUN_DIR, MODEL_SAVE_PATH, _baseline_manifest
    if name not in BASELINE_NAMES:
        raise ValueError(name)
    set_seed(RANDOM_SEED)
    frames, paths, raw_rows, flagged = prepare_shared_data()
    train_df, val_df = frames["training"], frames["validation"]
    RUN_DIR = PROJECT_DIR / "experiments" / f"{name}_v25_{RUN_ID}"
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    MODEL_SAVE_PATH = RUN_DIR / "best_model.pt"
    full_provenance = provenance_report(train_df, raw_rows=raw_rows, flagged_raw=flagged)
    if SMOKE_TEST:
        train_df = stratified_smoke_subset(train_df, SMOKE_TRAIN_ROWS, "training").reset_index(drop=True)
        val_df = stratified_smoke_subset(val_df, SMOKE_VAL_ROWS, "validation").reset_index(drop=True)
        print(f"SMOKE TEST {name}: {len(train_df)} train / {len(val_df)} val; NO TEST EVALUATION")
        subset_flags = train_df.get("needs_revision")
        subset_flagged = int(subset_flags.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"]).sum()) if subset_flags is not None else 0
        provenance_report(train_df, raw_rows=len(train_df), flagged_raw=subset_flagged,
                          filename="smoke_training_provenance.json")
    tokenizer = tokenizer_for(name, train_df)
    if name == "lstm_cnn":
        (RUN_DIR / "vocabulary.json").write_text(json.dumps(tokenizer.token_to_id), encoding="utf-8")
    train_loader = DataLoader(LyricsDataset(train_df, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(LyricsDataset(val_df, tokenizer), batch_size=BATCH_SIZE)
    model = make_baseline(name, tokenizer if name == "lstm_cnn" else None).to(DEVICE)
    counts = train_df["label"].value_counts().reindex(range(3), fill_value=0)
    if (counts == 0).any():
        raise ValueError(f"Missing training class: {counts.to_dict()}")
    weights = len(train_df) / (3 * counts.astype(float))
    weights_tensor = torch.tensor(weights.to_numpy(), dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    lr = BASELINE_LSTM_LEARNING_RATE if name == "lstm_cnn" else BASELINE_LEARNING_RATE
    optimizer = AdamW(model.parameters(), lr=lr)
    steps_per_epoch = (len(train_loader) + GRADIENT_ACCUMULATION_STEPS - 1) // GRADIENT_ACCUMULATION_STEPS
    epochs = 1 if SMOKE_TEST else EPOCHS
    total_steps = max(1, steps_per_epoch * epochs)
    warmup = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup, num_training_steps=total_steps)
    _baseline_manifest = {
        "architecture": name, "run_id": RUN_ID, "smoke_test": SMOKE_TEST,
        "checkpoint_selection": "validation_macro_f1", "test_evaluated_during_training": False,
        "preprocessing": "shared clean_lyrics from v25 (NFKC and section-tag removal)",
        "tokenizer": "train-only learned vocabulary; regex tokens" if name == "lstm_cnn" else ("bert-base-uncased" if name == "bert_cnn_lstm" else "roberta-base"),
        "token_policy": "first 512 tokens, not full-song coverage",
        "input_policy": "first 512 tokens, not full-song coverage",
        "input_max_length": MAX_LENGTH, "seed": RANDOM_SEED,
        "training_rows": len(train_df), "validation_rows": len(val_df), "heldout_test_rows": len(frames["test"]),
        "training_class_counts": {ID2LABEL[i]: int(counts[i]) for i in range(3)},
        "class_weights": {ID2LABEL[i]: float(weights[i]) for i in range(3)},
        "gradient_objective": "weighted CE numerator / sum target weights per effective batch",
        "max_epochs": epochs, "batch_size": BATCH_SIZE, "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": lr, "planned_optimizer_steps": total_steps, "planned_warmup_steps": warmup,
        "exclude_needs_revision": EXCLUDE_NEEDS_REVISION, "provenance": full_provenance,
        "input_sha256": {k: file_sha256(v) for k, v in paths.items()},
        "python": platform.python_version(), "torch": torch.__version__,
        "dependencies": {dep: installed_version(dep) for dep in ("transformers", "pandas", "scikit-learn")},
    }
    if name == "lstm_cnn":
        _baseline_manifest["vocab_size"] = len(tokenizer.token_to_id)
        _baseline_manifest["vocab_sha256"] = file_sha256(RUN_DIR / "vocabulary.json")
    write_baseline_manifest()
    if RUN_PADDING_INVARIANCE_CHECK and name != "lstm_cnn":
        _baseline_manifest["padding_pretrain"] = padding_invariance_check(model, tokenizer, val_df, phase="pretrain", hard_fail=False)
        write_baseline_manifest()
    best = float("-inf")
    best_epoch = 0
    no_improvement = 0
    history = []
    for epoch in range(1, epochs+1):
        print(f"\n{name} epoch {epoch}/{epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scheduler)
        val_loss, val_acc, actual, pred = validate(model, val_loader, criterion)
        score = float(f1_score(actual, pred, labels=[0, 1, 2], average="macro", zero_division=0))
        print(f"train loss {train_loss:.4f}, val loss {val_loss:.4f}, val Macro-F1 {score:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
                        "validation_loss": val_loss, "validation_accuracy": val_acc, "validation_macro_f1": score})
        pd.DataFrame(history).to_csv(RUN_DIR / "epoch_history.csv", index=False)
        if score > best + EARLY_STOPPING_MIN_DELTA:
            best = score
            best_epoch = epoch
            no_improvement = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("Best checkpoint saved:", MODEL_SAVE_PATH)
        else:
            no_improvement += 1
            if no_improvement >= EARLY_STOPPING_PATIENCE:
                print("Early stopping")
                break
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    _, _, actual, pred = validate(model, val_loader, criterion)
    val_metrics = save_predictions_and_metrics("validation", val_df, actual, pred)
    if RUN_PADDING_INVARIANCE_CHECK and name != "lstm_cnn":
        _baseline_manifest["padding_posttrain"] = padding_invariance_check(
            model, tokenizer, val_df, phase="posttrain", hard_fail=PADDING_HARD_FAIL_AFTER_TRAINING)
    _baseline_manifest.update({"best_epoch": best_epoch, "best_validation_macro_f1": best,
                               "validation_macro_f1": val_metrics["macro_f1"],
                               "optimizer_steps_completed": len(history)*steps_per_epoch,
                               "checkpoint_sha256": file_sha256(MODEL_SAVE_PATH)})
    write_baseline_manifest()
    print(f"Completed {name}. Frozen test remains untouched. Run directory: {RUN_DIR}")


def evaluate_frozen_checkpoints():
    """No training, no checkpoint selection, no parameter updates. One held-out pass."""
    global RUN_DIR
    if SMOKE_TEST:
        raise ValueError("Disable SMOKE_TEST before final frozen test evaluation")
    if not FROZEN_TEST_EVALUATION:
        raise ValueError("Set FROZEN_TEST_EVALUATION=True to authorize one final test pass")
    frames, paths, _, _ = prepare_shared_data()
    test_df = frames["test"]
    output_dir = PROJECT_DIR / "experiments" / f"comparison_v25_{RUN_ID}"
    output_dir.mkdir(parents=True, exist_ok=False)
    # Preflight every frozen run before evaluating any model on the held-out set.
    frozen_policies = {}
    frozen_input_lengths = {}
    expected_hashes = {k: file_sha256(v) for k, v in paths.items()}
    for candidate in ALL_NAMES:
        source = FROZEN_RUN_DIRS.get(candidate, "")
        if not source:
            raise ValueError(f"Specify FROZEN_RUN_DIRS[{candidate!r}] before testing")
        candidate_dir = Path(source).expanduser()
        if not candidate_dir.is_absolute():
            candidate_dir = PROJECT_DIR / candidate_dir
        manifest_file = candidate_dir / "manifest.json"
        checkpoint_file = candidate_dir / "best_model.pt"
        if not manifest_file.is_file() or not checkpoint_file.is_file():
            raise FileNotFoundError(f"Missing frozen run files: {candidate_dir}")
        meta = json.loads(manifest_file.read_text(encoding="utf-8"))
        if meta.get("smoke_test") or meta.get("input_sha256") != expected_hashes:
            raise ValueError(f"Smoke run or input-hash mismatch: {candidate_dir}")
        if candidate != "hybrid" and meta.get("architecture") != candidate:
            raise ValueError(f"Wrong architecture for {candidate}: {candidate_dir}")
        if meta.get("checkpoint_sha256") != file_sha256(checkpoint_file):
            raise ValueError(f"Checkpoint integrity mismatch: {candidate_dir}")
        if "exclude_needs_revision" not in meta:
            raise ValueError(f"Missing training policy in manifest: {candidate_dir}")
        frozen_policies[candidate] = meta["exclude_needs_revision"]
        length = meta.get("input_max_length", meta.get("max_length"))
        if isinstance(length, bool) or not isinstance(length, int):
            raise ValueError(f"Missing or invalid input maximum length in manifest: {candidate_dir}")
        frozen_input_lengths[candidate] = length
        if not (meta.get("input_policy") or meta.get("token_policy")):
            raise ValueError(f"Missing input/token policy description: {candidate_dir}")
    if len(set(frozen_input_lengths.values())) != 1 or next(iter(frozen_input_lengths.values())) != MAX_LENGTH:
        raise ValueError(f"Frozen checkpoints disagree on max input length: {frozen_input_lengths}; current={MAX_LENGTH}")
    if len(set(frozen_policies.values())) != 1 or next(iter(frozen_policies.values())) != EXCLUDE_NEEDS_REVISION:
        raise ValueError(f"Frozen checkpoints disagree on needs_revision inclusion: {frozen_policies}; current={EXCLUDE_NEEDS_REVISION}")
    records = []
    confusion_records = []
    all_preds = test_df[["song_name", "artist", "explicitness", "label"]].copy()
    all_preds.insert(0, "test_row_position", range(len(test_df)))
    for name in ALL_NAMES:
        source = FROZEN_RUN_DIRS.get(name, "")
        if not source:
            raise ValueError(f"Specify FROZEN_RUN_DIRS[{name!r}] first")
        run_dir = Path(source).expanduser()
        if not run_dir.is_absolute():
            run_dir = PROJECT_DIR / run_dir
        manifest_path = run_dir / "manifest.json"
        checkpoint = run_dir / "best_model.pt"
        if not manifest_path.is_file() or not checkpoint.is_file():
            raise FileNotFoundError(f"Missing frozen manifest/checkpoint in {run_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("smoke_test"):
            raise ValueError(f"Refusing smoke checkpoint: {run_dir}")
        if manifest.get("input_sha256") != {k: file_sha256(v) for k,v in paths.items()}:
            raise ValueError(f"Dataset hashes differ from frozen checkpoint: {run_dir}")
        if name != "hybrid" and manifest.get("architecture") != name:
            raise ValueError(f"Architecture mismatch: {name} at {run_dir}")
        if manifest.get("checkpoint_sha256") != file_sha256(checkpoint):
            raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")
        tokenizer = tokenizer_for(name, frames["training"], run_dir)
        model = make_baseline(name, tokenizer if name == "lstm_cnn" else None).to(DEVICE)
        model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
        model.eval()
        loader = DataLoader(LyricsDataset(test_df, tokenizer), batch_size=BATCH_SIZE)
        actual, pred = evaluate_test_set(model, loader)
        if actual != test_df["label"].tolist():
            raise RuntimeError(f"Test order mismatch for {name}")
        RUN_DIR = output_dir / name
        RUN_DIR.mkdir()
        metrics = save_predictions_and_metrics("test", test_df, actual, pred)
        all_preds[f"{name}_predicted_id"] = pred
        cm = confusion_matrix(actual, pred, labels=[0, 1, 2])
        for true_id in range(3):
            for predicted_id in range(3):
                confusion_records.append({"model": name, "true_label": ID2LABEL[true_id],
                                          "predicted_label": ID2LABEL[predicted_id],
                                          "count": int(cm[true_id, predicted_id])})
        records.append({"model": name, "test_accuracy": metrics["accuracy"],
                        "test_macro_f1": metrics["macro_f1"], "test_weighted_f1": metrics["weighted_f1"],
                        "checkpoint_dir": str(run_dir), "checkpoint_sha256": file_sha256(checkpoint)})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    RUN_DIR = output_dir
    pd.DataFrame(records).to_csv(output_dir / "model_comparison.csv", index=False)
    all_preds.to_csv(output_dir / "paired_predictions.csv", index=False)
    pd.DataFrame(confusion_records).to_csv(output_dir / "model_comparison_confusion.csv", index=False)
    # Paired bootstrap resamples the same songs for both models, not independent F1s.
    rng = np.random.default_rng(RANDOM_SEED)
    actual = all_preds["label"].to_numpy()
    hybrid = all_preds["hybrid_predicted_id"].to_numpy()
    paired = []
    for name in BASELINE_NAMES:
        other = all_preds[f"{name}_predicted_id"].to_numpy()
        observed = float(f1_score(actual, hybrid, labels=[0,1,2], average="macro", zero_division=0) -
                         f1_score(actual, other, labels=[0,1,2], average="macro", zero_division=0))
        diffs = []
        for _ in range(2000):
            idx = rng.integers(0, len(actual), len(actual))
            diffs.append(float(f1_score(actual[idx], hybrid[idx], labels=[0,1,2], average="macro", zero_division=0) -
                               f1_score(actual[idx], other[idx], labels=[0,1,2], average="macro", zero_division=0)))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        # Paired randomization test under the null of exchangeable model predictions.
        # Two-sided Monte Carlo p-value, with +1 correction.
        permutation_diffs = []
        for _ in range(2000):
            swap = rng.random(len(actual)) < 0.5
            perm_hybrid = np.where(swap, other, hybrid)
            perm_other = np.where(swap, hybrid, other)
            permutation_diffs.append(float(
                f1_score(actual, perm_hybrid, labels=[0,1,2], average="macro", zero_division=0) -
                f1_score(actual, perm_other, labels=[0,1,2], average="macro", zero_division=0)))
        p_value = (1 + sum(abs(value) >= abs(observed) - 1e-12 for value in permutation_diffs)) / 2001
        paired.append({"comparison": f"hybrid_minus_{name}",
                       "paired_permutation_p_two_sided": float(p_value),
                       "permutation_resamples": 2000, "macro_f1_difference": observed,
                       "paired_bootstrap_ci95_low": float(lo), "paired_bootstrap_ci95_high": float(hi),
                       "bootstrap_resamples": 2000})
    pd.DataFrame(paired).to_csv(output_dir / "paired_macro_f1_bootstrap.csv", index=False)
    print("Frozen four-model comparison complete:", output_dir)


def main_v25():
    global RUN_DIR, MODEL_SAVE_PATH
    if EXPERIMENT_MODE == "hybrid":
        if FROZEN_TEST_EVALUATION:
            raise ValueError("Use EXPERIMENT_MODE='compare' for frozen test evaluation")
        RUN_DIR = PROJECT_DIR / "experiments" / f"hybrid_v25_{RUN_ID}"
        MODEL_SAVE_PATH = RUN_DIR / "best_model.pt"
        main()  # v24 hybrid training flow, unchanged except v25 output naming
    elif EXPERIMENT_MODE == "all_models":
        if FROZEN_TEST_EVALUATION:
            raise ValueError("Training mode cannot evaluate frozen test; use 'compare' later")
        # Run sequentially, not concurrently: avoid simultaneous GPU model allocations.
        # Restore the hybrid's output paths explicitly because baselines mutate globals.
        order = ("hybrid",) + BASELINE_NAMES
        for position, name in enumerate(order, start=1):
            print(f"\n{'=' * 70}\nMODEL {position}/{len(order)}: {name}"
                  f" | {'SMOKE TEST' if SMOKE_TEST else 'FULL TRAINING'}\n{'=' * 70}", flush=True)
            if name == "hybrid":
                RUN_DIR = PROJECT_DIR / "experiments" / f"hybrid_v25_{RUN_ID}"
                MODEL_SAVE_PATH = RUN_DIR / "best_model.pt"
                main()
            else:
                run_baseline(name)
            print(f"Completed {name}. Output: {RUN_DIR}", flush=True)
        print("\nAll four model training runs completed."
              " Held-out testing was NOT performed."
              " Use EXPERIMENT_MODE='compare' after freezing checkpoints.", flush=True)
    elif EXPERIMENT_MODE in BASELINE_NAMES:
        if FROZEN_TEST_EVALUATION:
            raise ValueError("Training mode cannot evaluate frozen test")
        run_baseline(EXPERIMENT_MODE)
    elif EXPERIMENT_MODE == "all_baselines":
        if FROZEN_TEST_EVALUATION:
            raise ValueError("Training mode cannot evaluate frozen test")
        for name in BASELINE_NAMES:
            run_baseline(name)
    elif EXPERIMENT_MODE == "compare":
        evaluate_frozen_checkpoints()
    else:
        raise ValueError(f"Unknown EXPERIMENT_MODE={EXPERIMENT_MODE!r}")


if __name__ == "__main__":
    main_v25()
