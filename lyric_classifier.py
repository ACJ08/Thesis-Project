import os
import re
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from transformers import RobertaTokenizer, RobertaModel
from torch.optim import AdamW


#Notes: LIME and SHAP are still missing


# ============================================================
# CONFIGURATION
# ============================================================

CSV_FILE = "lyrics_dataset.csv"

MODEL_NAME = "roberta-base"

MAX_LENGTH = 512

BATCH_SIZE = 2
EPOCHS = 3

LEARNING_RATE = 2e-5

RANDOM_SEED = 67 #WEIIIIIII

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

    We intentionally avoid aggressive preprocessing because
    RoBERTa benefits from keeping natural language information.
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
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(CSV_FILE)


# Check required columns
# Subject to change should csv be different
required_columns = [
    "song_name",
    "artist",
    "text",
    "explicitness"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Dataset is missing required column: {column}"
        )


# Clean lyrics
df["text"] = df["text"].apply(clean_lyrics)


# Remove empty lyrics
df = df[df["text"].str.len() > 0].copy()


# Remove invalid labels
# Basic. Subject to update according to received CSV. Will not be removed for safety.
df = df[
    df["explicitness"].isin(LABEL2ID.keys())
].copy()


# Convert labels to integers
df["label"] = df["explicitness"].map(LABEL2ID)


print("Number of songs:", len(df))

print("\nClass distribution:")
print(df["explicitness"].value_counts())


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

# Basic.
# Will instead use direct dfs for the separate csv instead of actual separation.

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=df["label"]
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)


print("\nTraining samples:", len(train_df))
print("Validation samples:", len(val_df))


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


# ============================================================
# MODEL
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

        self.dropout = nn.Dropout(dropout)


        # ----------------------------------------------------
        # BiLSTM. Will change to LSTM later.
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

            nn.Dropout(dropout),

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

        # ====================================================
        # RoBERTa
        # ====================================================

        roberta_output = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Shape:
        #
        # [batch, sequence_length, 768]
        #

        x = roberta_output.last_hidden_state


        # ====================================================
        # CNN
        # ====================================================

        # Conv1D expects:
        #
        # [batch, channels, sequence]
        #

        x = x.transpose(1, 2)

        x = self.conv1(x)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.conv2(x)

        x = self.relu(x)

        x = self.dropout(x)


        # Return to:
        #
        # [batch, sequence, features]
        #

        x = x.transpose(1, 2)


        # ====================================================
        # BiLSTM
        # ====================================================

        x, _ = self.lstm(x)


        # ====================================================
        # GLOBAL POOLING
        # ====================================================

        x = torch.mean(
            x,
            dim=1
        )


        # ====================================================
        # CLASSIFIER
        # ====================================================

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
            preds.detach().cpu().numpy()
        )

        actual.extend(
            labels.detach().cpu().numpy()
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


            # Forward pass
            logits = model(
                input_ids,
                attention_mask
            )


            # Validation loss
            loss = criterion(
                logits,
                labels
            )


            total_loss += loss.item()


            # Predictions
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
# TRAINING LOOP
# ============================================================

print("\nStarting training...\n")


best_val_accuracy = 0


for epoch in range(EPOCHS):

    print(
        f"\n========== "
        f"Epoch {epoch + 1}/{EPOCHS}"
        f" =========="
    )


    # Training
    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion
    )


    # Validation
    (
        val_loss,
        val_accuracy,
        actual,
        predictions
    ) = validate(
        model,
        val_loader,
        criterion
    )


    print(
        f"\nTrain Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy:.4f}"
    )

    print(
        f"Validation Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{val_accuracy:.4f}"
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


        print(
            "Best model saved!"
        )


# ============================================================
# FINAL VALIDATION REPORT
# ============================================================

print("\n================================")
print("FINAL VALIDATION REPORT")
print("================================")


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


# ============================================================
# INFERENCE FUNCTION
# ============================================================

def predict_lyrics(
    lyrics,
    model,
    tokenizer
):

    model.eval()


    # Preprocess
    lyrics = clean_lyrics(
        lyrics
    )


    # Tokenize
    encoding = tokenizer(
        lyrics,
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


    # Prediction
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


    label = ID2LABEL[
        predicted_class
    ]


    confidence = probabilities[
        0,
        predicted_class
    ].item()


    return label, confidence


# ============================================================
# LOAD BEST MODEL
# ============================================================

if os.path.exists(
    MODEL_SAVE_PATH
):

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


# ============================================================
# EXAMPLE INFERENCE
# ============================================================

example_lyrics = """
Replace this text with the lyrics
you want the classifier to analyze.
"""


prediction, confidence = predict_lyrics(
    example_lyrics,
    model,
    tokenizer
)


print("\n================================")
print("INFERENCE")
print("================================")

print(
    "Prediction:",
    prediction
)

print(
    "Confidence:",
    f"{confidence * 100:.2f}%"
)