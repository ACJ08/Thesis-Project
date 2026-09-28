# Lyric Explicitness Classifier

A deep learning-based text classification system for classifying song lyrics into three levels of explicitness:

* **Safe**
* **Mild**
* **Explicit**

The system uses a hybrid **RoBERTa → CNN → BiLSTM** architecture for lyric classification and provides **LIME** and **SHAP** explanations for model predictions.

---

## Features

* RoBERTa-based contextual text representation
* CNN layers for extracting local textual patterns
* Bidirectional LSTM (BiLSTM) for sequential context
* Three-class lyric explicitness classification
* Separate training, validation, and test datasets
* Best-model selection based on validation accuracy
* Final evaluation on an independent test set
* LIME-based prediction explanations
* SHAP-based feature contribution analysis
* Single-lyric inference with prediction confidence

---

## Model Architecture

The classification pipeline is:

```text
Song Lyrics
     │
     ▼
  RoBERTa
     │
     ▼
    CNN
     │
     ▼
   BiLSTM
     │
     ▼
 Classifier
     │
     ▼
Safe / Mild / Explicit
```

### RoBERTa

RoBERTa is used to generate contextual representations of the lyrics.

### CNN

Two one-dimensional convolutional layers are used to capture local patterns in the token representations.

### BiLSTM

A bidirectional LSTM processes the sequence produced by the CNN layers, allowing the model to use contextual information from both directions.

### Classifier

The BiLSTM representation is passed through fully connected layers to produce predictions for the three explicitness classes.

---

## Dataset

The project uses three separate CSV files:

```text
train.csv
validation.csv
test.csv
```

The intended dataset distribution is:

| Dataset    | Number of Lyrics | Purpose                    |
| ---------- | ---------------: | -------------------------- |
| Training   |           40,000 | Model training             |
| Validation |            5,000 | Model selection and tuning |
| Test       |            5,000 | Final evaluation           |

The datasets are kept separate throughout the training process.

### Required Columns

Each CSV file must contain:

```text
song_name
artist
text
explicitness
```

Example:

| song_name    | artist         | text              | explicitness |
| ------------ | -------------- | ----------------- | ------------ |
| Example Song | Example Artist | Example lyrics... | Safe         |

### Labels

The classifier uses the following label mapping:

```text
Safe     → 0
Mild     → 1
Explicit → 2
```

---

## Preprocessing

Lyrics undergo basic preprocessing before being passed to RoBERTa.

The preprocessing includes:

* Normalizing line endings
* Removing excessive spaces and tabs
* Collapsing excessive blank lines
* Removing leading and trailing whitespace
* Removing empty lyric entries
* Removing entries with invalid explicitness labels

Aggressive preprocessing such as stopword removal and stemming is avoided because RoBERTa is designed to work with natural language context.

---

## Requirements

The project requires:

```text
torch
transformers
scikit-learn
pandas
numpy
tqdm
lime
shap
```

Install the dependencies with:

```bash
pip install -r requirements.txt
```

---

## Installation

### 1. Clone or download the project

Place the project files in the same directory.

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Place the datasets

Make sure the following files are available in the project directory:

```text
train.csv
validation.csv
test.csv
```

---

## Running the Model

Run:

```bash
python lyric_classifier.py
```

The program will:

1. Load the training, validation, and test datasets.
2. Preprocess the datasets.
3. Load the RoBERTa tokenizer.
4. Create the PyTorch datasets and dataloaders.
5. Initialize the RoBERTa → CNN → BiLSTM model.
6. Train the model using the training dataset.
7. Evaluate the model against the validation dataset after each epoch.
8. Save the model with the best validation accuracy.
9. Reload the best model.
10. Perform a final evaluation using the test dataset.
11. Run example lyric inference.
12. Generate LIME and SHAP explanations for the example.

---

## Training, Validation, and Testing

The three datasets have different roles.

### Training Set

The training set is used to update the model's parameters.

```text
train.csv
    ↓
Model training
    ↓
Updated model
```

### Validation Set

The validation set is evaluated after each training epoch.

The model with the highest validation accuracy is saved as the best model.

```text
validation.csv
    ↓
Validation accuracy
    ↓
Best model selection
```

### Test Set

The test set is not used during model training or model selection.

After training is complete, the best validation model is reloaded and evaluated against the test set.

```text
Best model
    ↓
test.csv
    ↓
Final performance
```

This provides an independent evaluation of the trained classifier.

---

## Model Configuration

The current configuration includes:

```text
Model:              roberta-base
Maximum sequence:   512 tokens
Batch size:         2
Epochs:             3
Learning rate:      2e-5
LSTM hidden size:   128
CNN channels:       256
Dropout:            0.3
Random seed:        67
```

The model uses a bidirectional LSTM, meaning the LSTM processes the sequence in both forward and backward directions.

---

## Evaluation

The model reports:

* Accuracy
* Precision
* Recall
* F1-score
* Per-class classification results

The final test evaluation is performed on the independent test dataset.

Example output:

```text
================================
FINAL TEST SET RESULTS
================================

Test Accuracy: 0.XXXX

Classification Report:

              precision    recall  f1-score   support

        Safe       ...       ...       ...      ...
        Mild       ...       ...       ...      ...
    Explicit       ...       ...       ...      ...
```

---

## Inference

The trained model can classify new lyrics.

The inference process returns:

* Predicted explicitness class
* Prediction confidence

Example:

```text
Prediction: Explicit
Confidence: 94.21%
```

Replace the example lyrics in `lyric_classifier.py` with the lyrics to be analyzed.

---

## Explainability

The project includes two post-hoc explainability methods:

### LIME

LIME identifies words or phrases that contribute toward or against the predicted class.

Example:

```text
Important words/phrases:

word_1    +0.1234 (supports)
word_2    -0.0872 (opposes)
```

### SHAP

SHAP provides feature contribution values for the model's prediction.

The implementation identifies influential tokens and indicates whether they support or oppose the predicted class.

These explainability methods are used after model prediction and are not part of the model's training process.

---

## Saved Model

The best-performing model is saved to:

```text
saved_model/lyric_classifier.pt
```

The saved model is selected using validation accuracy.

The model is then reloaded before final test-set evaluation and inference.

---

## Project Structure

```text
lyric-explicitness-classifier/
│
├── lyric_classifier.py
│
├── requirements.txt
├── README.md
│
├── train.csv
├── validation.csv
├── test.csv
│
└── saved_model/
    └── lyric_classifier.pt
```

---

## Notes and Limitations

### Maximum Sequence Length

The current implementation uses:

```text
MAX_LENGTH = 512
```

Lyrics longer than the maximum sequence length are truncated by the tokenizer.

Therefore, only the first portion of a lyric exceeding the maximum token length is processed by the model.

### Computational Requirements

RoBERTa is computationally intensive. Training time depends on the available hardware, dataset size, and batch size.

### Explainability Runtime

LIME and SHAP perform additional model evaluations and can therefore take considerably longer than a normal prediction.

They are intended for explaining selected predictions rather than necessarily being run over the entire test dataset.

---

## Technologies

* Python
* PyTorch
* Hugging Face Transformers
* RoBERTa
* NumPy
* pandas
* scikit-learn
* tqdm
* LIME
* SHAP

---

## Classification Labels

```text
0 = Safe
1 = Mild
2 = Explicit
```

The model's final output corresponds to one of these three classes.
