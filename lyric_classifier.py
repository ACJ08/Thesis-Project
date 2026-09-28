import os
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

# Explainability
from lime.lime_text import LimeTextExplainer
import shap


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_CSV_FILE = "train.csv"
VAL_CSV_FILE = "validation.csv"
TEST_CSV_FILE = "test.csv"

MODEL_NAME = "roberta-base"

MAX_LENGTH = 512
BATCH_SIZE = 2
EPOCHS = 3
LEARNING_RATE = 2e-5

RANDOM_SEED = 67

MODEL_SAVE_PATH = "saved_model/lyric_classifier.pt"


# ============================================================
# SEED
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


# ============================================================
# LABELS
# ============================================================

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


# ============================================================
# TEXT PREPROCESSING
# ============================================================

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


# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_CSV_FILE)
val_df = pd.read_csv(VAL_CSV_FILE)
test_df = pd.read_csv(TEST_CSV_FILE)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "song_name",
    "artist",
    "text",
    "explicitness"
]

for column in required_columns:

    if column not in train_df.columns:
        raise ValueError(
            f"Training dataset is missing required column: {column}"
        )

    if column not in val_df.columns:
        raise ValueError(
            f"Validation dataset is missing required column: {column}"
        )

    if column not in test_df.columns:
        raise ValueError(
            f"Test dataset is missing required column: {column}"
        )


# ============================================================
# PREPROCESS DATASETS
# ============================================================

train_df["text"] = train_df["text"].apply(clean_lyrics)
val_df["text"] = val_df["text"].apply(clean_lyrics)
test_df["text"] = test_df["text"].apply(clean_lyrics)


# ============================================================
# REMOVE EMPTY LYRICS
# ============================================================

train_df = train_df[
    train_df["text"].str.len() > 0
].copy()

val_df = val_df[
    val_df["text"].str.len() > 0
].copy()

test_df = test_df[
    test_df["text"].str.len() > 0
].copy()


# ============================================================
# REMOVE INVALID LABELS
# ============================================================

train_df = train_df[
    train_df["explicitness"].isin(LABEL2ID.keys())
].copy()

val_df = val_df[
    val_df["explicitness"].isin(LABEL2ID.keys())
].copy()

test_df = test_df[
    test_df["explicitness"].isin(LABEL2ID.keys())
].copy()


# ============================================================
# CONVERT LABELS TO NUMBERS
# ============================================================

train_df["label"] = train_df["explicitness"].map(LABEL2ID)
val_df["label"] = val_df["explicitness"].map(LABEL2ID)
test_df["label"] = test_df["explicitness"].map(LABEL2ID)


# ============================================================
# DATASET INFORMATION
# ============================================================

print("\nTraining songs:", len(train_df))
print("Validation songs:", len(val_df))
print("Test songs:", len(test_df))


print("\nTraining class distribution:")
print(train_df["explicitness"].value_counts())


print("\nValidation class distribution:")
print(val_df["explicitness"].value_counts())


print("\nTest class distribution:")
print(test_df["explicitness"].value_counts())


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading RoBERTa tokenizer...")

tokenizer = RobertaTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# DATASET CLASS
# ============================================================

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


# ============================================================
# CREATE DATASETS
# ============================================================

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


# ============================================================
# CREATE DATALOADERS
# ============================================================

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


# ============================================================
# MODEL
# ============================================================
#
# RoBERTa → CNN → BiLSTM → Classifier
#
# ============================================================

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
            bidirectional=True,
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
                lstm_hidden_size * 2,
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
        # GLOBAL POOLING
        # ----------------------------------------------------

        x = torch.mean(
            x,
            dim=1
        )


        # ----------------------------------------------------
        # CLASSIFIER
        # ----------------------------------------------------

        logits = self.classifier(x)

        return logits


# ============================================================
# INITIALIZE MODEL
# ============================================================

print("\nLoading model...")

model = RoBERTaCNNLSTM(
    num_classes=3
)

model = model.to(DEVICE)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion
):

    model.train()

    total_loss = 0

    predictions = []
    actual = []

    progress = tqdm(
        loader,
        desc="Training"
    )

    for batch in progress:

        input_ids = batch[
            "input_ids"
        ].to(DEVICE)

        attention_mask = batch[
            "attention_mask"
        ].to(DEVICE)

        labels = batch[
            "label"
        ].to(DEVICE)


        # Reset gradients
        optimizer.zero_grad()


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


        # Backpropagation
        loss.backward()


        # Update parameters
        optimizer.step()


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


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def validate(
    model,
    loader,
    criterion
):

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


# ============================================================
# TEST SET EVALUATION
# ============================================================

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


# ============================================================
# TRAINING LOOP
# ============================================================

print(
    "\nStarting training...\n"
)

best_val_accuracy = 0.0


for epoch in range(EPOCHS):

    print(
        f"\n========== "
        f"Epoch {epoch + 1}/{EPOCHS}"
        f" =========="
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion
    )


    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    (
        val_loss,
        val_accuracy,
        val_actual,
        val_predictions
    ) = validate(
        model,
        val_loader,
        criterion
    )


    print(
        f"\nTrain Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: {train_accuracy:.4f}"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: {val_accuracy:.4f}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        os.makedirs(
            "saved_model",
            exist_ok=True
        )

        torch.save(
            model.state_dict(),
            MODEL_SAVE_PATH
        )

        print("Best model saved!")


# ============================================================
# LOAD BEST MODEL
# ============================================================

if os.path.exists(MODEL_SAVE_PATH):

    model.load_state_dict(
        torch.load(
            MODEL_SAVE_PATH,
            map_location=DEVICE
        )
    )

    model.eval()

    print(
        "\nBest model loaded."
    )

else:

    raise FileNotFoundError(
        "Best model was not saved."
    )


# ============================================================
# FINAL VALIDATION REPORT
# ============================================================

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
    f"\nBest Validation Accuracy: "
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


# ============================================================
# FINAL TEST SET
# ============================================================
#
# IMPORTANT:
# The test set is used only here.
#
# ============================================================

test_actual, test_predictions = evaluate_test_set(
    model,
    test_loader
)


# ============================================================
# INFERENCE
# ============================================================

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


# ============================================================
# MODEL PREDICTION WRAPPER
# ============================================================

def model_predict_proba(texts):
    """
    Wrapper used by LIME and SHAP.

    Returns:
        [Safe probability,
         Mild probability,
         Explicit probability]
    """

    model.eval()

    probabilities_list = []


    for text in texts:

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


        probabilities_list.append(
            probabilities
            .cpu()
            .numpy()[0]
        )


    return np.array(
        probabilities_list
    )


# ============================================================
# LIME EXPLANATION
# ============================================================

def explain_with_lime(text):

    print(
        "\n================================"
    )

    print(
        "LIME EXPLANATION"
    )

    print(
        "================================"
    )


    explainer = LimeTextExplainer(
        class_names=[
            "Safe",
            "Mild",
            "Explicit"
        ]
    )


    explanation = explainer.explain_instance(
        text,
        model_predict_proba,
        num_features=15,
        num_samples=500
    )


    probabilities = model_predict_proba(
        [text]
    )[0]


    predicted_class = np.argmax(
        probabilities
    )


    print(
        "\nPredicted class:",
        ID2LABEL[predicted_class]
    )


    print(
        "\nImportant words/phrases:"
    )


    for word, weight in explanation.as_list(
        label=predicted_class
    ):

        direction = (
            "supports"
            if weight > 0
            else "opposes"
        )

        print(
            f"{word:30s} "
            f"{weight:+.4f} "
            f"({direction})"
        )


    return explanation


# ============================================================
# SHAP EXPLANATION
# ============================================================

def explain_with_shap(text):

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
    # CREATE TEXT MASKER
    # --------------------------------------------------------

    masker = shap.maskers.Text(
        tokenizer=tokenizer
    )


    # --------------------------------------------------------
    # CREATE SHAP EXPLAINER
    # --------------------------------------------------------

    explainer = shap.Explainer(
        model_predict_proba,
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

    probabilities = model_predict_proba(
        [text]
    )[0]


    predicted_class = np.argmax(
        probabilities
    )


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

    for index in ranked_indices[:15]:

        token = tokens[index]
        value = class_values[index]

        direction = (
            "supports"
            if value > 0
            else "opposes"
        )

        print(
            f"{str(token):30s} "
            f"{value:+.4f} "
            f"({direction})"
        )


    return shap_values


# ============================================================
# EXAMPLE INFERENCE
# ============================================================

example_lyrics = """
Replace this text with the lyrics
you want the classifier to analyze.
"""


# ============================================================
# CLASSIFY LYRICS
# ============================================================

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


# ============================================================
# LIME EXPLANATION
# ============================================================

lime_explanation = explain_with_lime(
    example_lyrics
)


# ============================================================
# SHAP EXPLANATION
# ============================================================

shap_explanation = explain_with_shap(
    example_lyrics
)