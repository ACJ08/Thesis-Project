# Binary Lyric Explicitness Classifier

A deep learning-based text classification system for classifying song lyrics into two classes.

* **Safe**
* **Explicit**

The system uses a hybrid **RoBERTa → CNN → LSTM** architecture for lyric classification and provides **LIME** and **SHAP** explanations for model predictions.

Run the commands in this guide from the `binary_v26` directory. The binary scripts read the shared CSVs in the project’s `../data` directory and write their runs under `binary_v26/experiments`. The data paths shown below are relative to the project root.

## v26 Binary desktop application

On Windows, double-click `dist/LyricClassificationStudio/LyricClassificationStudio.exe`. Keep its `_internal` folder beside it. The bundle includes Python and the app libraries, so launching it requires no Python installation or terminal.

The executable uses the project’s existing `data` and `experiments` folders. If you move it to another computer, copy the entire `LyricClassificationStudio` folder and place those asset folders inside it. Model weights and datasets are separate from the application bundle. Transformer models may need an internet connection to fetch pretrained assets on first use.

To rebuild the Windows app from source, run these commands from the project folder using the classifier’s Python environment.

```powershell
python -m pip install -r requirements.txt
python -m PyInstaller --noconfirm LyricClassificationStudio.spec
```

To run the GUI directly from source, use:

```powershell
python thesis_gui.py
```

Python needs Tkinter, which is included with the standard Windows Python installer when Tcl/Tk support is enabled. Transformer models may need to download their pretrained assets on first use. Jobs run in a background process and report progress in **Run log**.

- **Training** lets you select one model, all three baselines, or all four models. **Comparison** is a separate workflow for the frozen held-out comparison. Smoke test is enabled by default. Leave the CSV fields blank to use the project's cleaned splits, or browse to separate labeled training, validation, and test CSVs. Frozen comparison requires four full v26 binary run folders and the same dataset hashes recorded during training.
- **Classification** loads a trained checkpoint for inference. Choose **Single song** or **CSV batch** explicitly. Paste one song or select a CSV and save a labeled copy. The CSV needs a `lyrics` or `text` column, or you can enter its column name. Original columns are preserved, and empty lyrics are skipped. The displayed probability belongs to the selected evidence window.
- **LIME / SHAP** offers either method individually or **LIME + SHAP** in one run. Each method highlights features in the selected evidence window and reports signed contributions and comprehensiveness at 5%, 10%, and 20% erasure, with equal-count random deletion as a baseline. The status line and **Run log** show the current method and erasure stage. The validation action runs the selected methods on a stratified sample and writes per-song and summary CSVs under `eraser_results`.

For labeling and explanations, select a checkpoint from the dropdown or browse to an experiment folder containing `best_model.pt`. The dropdown discovers nested v26 binary experiment checkpoints and shows their model, file date, smoke/full status, and relative folder. The selected folder and manifest run ID are shown before execution. LSTM–CNN also needs the matching `vocabulary.json`.

The resizable divider gives output about 70% of the available workspace initially and about 77% when a run starts. **Expand output** enlarges logs/results; drag the divider downward to enlarge the scrollable configuration area. Results include a single-song prediction card, training curves, confusion matrices, model comparisons, and erasure charts when the corresponding operation produces those data. Elapsed time and live stages appear above the output. Failure is red; cancellation is yellow.

Selections are saved automatically in `.lyric_studio/settings.json`: model and checkpoint per workflow/model, CSV fields, explanation options, and browsing/output folders. Lyrics are not saved as preferences. Completed, failed, and cancelled runs are archived under `.lyric_studio/history` with configuration, log, summary, and available results. **Run history** reopens them; **Open output folder**, **Export results** (JSON, logs and CSV tables), and **View configuration** act on the selected run. Run configurations include the submitted lyrics for reproducibility. These local app files are ignored by Git.

`thesis_gui_schema.py` defines request schema version 1 for training, classification, and explanation. Both the GUI preflight and worker enforce it. It checks field types/enums, required input, CSV headers, separate dataset/output paths, checkpoint architecture/version, LSTM vocabulary, nonnegative sample sizes, and full-run dataset hashes for frozen comparison. Preflight runs in a background thread; the model pipeline additionally enforces its full scientific policy and checkpoint integrity checks.

Run regression checks without model downloads or training:

```powershell
python -m unittest test_thesis_gui -v
```

---

## Features

* RoBERTa-based contextual text representation
* CNN layers for extracting local textual patterns
* Unidirectional LSTM for sequential context
* Binary lyric explicitness classification
* Separate training, validation, and test datasets
* Best-model selection based on validation macro-F1
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
   LSTM
     │
     ▼
 Classifier
     │
     ▼
Safe / Explicit
```

### RoBERTa

RoBERTa is used to generate contextual representations of the lyrics.

### CNN

Two one-dimensional convolutional layers are used to capture local patterns in the token representations.

### LSTM

A unidirectional LSTM processes the sequence produced by the CNN layers, allowing the model to use contextual information from both directions.

### Classifier

The LSTM representation is passed through fully connected layers to produce predictions for the two explicitness classes.

---

## Dataset

The project uses three separate CSV files:

```text
data/Training_Cleaned_binary.csv
data/Validation_Cleaned_binary.csv
data/Testing_Cleaned_binary.csv
```

The supplied CSVs use Spotify's Explicit flag as the target. Their row counts are shown below.

| Dataset    | Number of Lyrics | Purpose                    |
| ---------- | ---------------: | -------------------------- |
| Training   |           5,988 | Model training             |
| Validation |            5,000 | Model selection and tuning |
| Test       |            4,998 | Final evaluation           |

The datasets are kept separate throughout the training process.

### Required Columns

Each CSV file must contain:

```text
song_title
artist
lyrics
safety_label
```

Example:

| song_title   | artist         | lyrics            | safety_label |
| ------------ | -------------- | ----------------- | ------------ |
| Example Song | Example Artist | Example lyrics... | Safe         |

### Labels

The classifier uses the following label mapping:

```text
Safe     → 0
Explicit → 1
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

### 4. Hugging Face authentication (optional)

The classifier downloads the pretrained `roberta-base` model from Hugging Face when it runs for the first time. Because this is a public model, authentication is generally optional. Logging in can help avoid some unauthenticated download limits.

Install or update the Hugging Face CLI through the same Python environment used to run the project:

```bash
python -m pip install -U huggingface_hub
```

Then log in:

```bash
hf auth login
```

Paste your own **Hugging Face read token** when prompted. You can create one in your Hugging Face account settings under **Access Tokens**. Do not put the token in this repository, in `requirements.txt`, or directly in the source code. The CLI stores it in your local user environment, and the Hugging Face libraries can use it automatically.

To verify the login:

```bash
hf auth whoami
```

If the `hf` command is not recognized, run it through the Scripts directory of the Python environment where `huggingface_hub` was installed, or activate that environment first. Authentication is optional for downloading public models.

### 5. Place the datasets

Make sure the following files are available in the project directory:

```text
data/Training_Cleaned_binary.csv
data/Validation_Cleaned_binary.csv
data/Testing_Cleaned_binary.csv
```

---

## Running the Model

Set `EXPERIMENT_MODE` in the classifier to choose a model or use `all_models` to train all four. The default is `lstm_cnn`. Set `SMOKE_TEST = True` for a one-epoch check. Full binary runs save checkpoints and validation outputs under `experiments/<model>_v26_binary_<run_id>`. Train fresh binary checkpoints before inference or frozen test evaluation.

Run:

```bash
python lyric_classifier_v26_all_models.py
```

The selected model uses its own tokenizer, trains on the binary splits, and reloads the checkpoint selected by validation macro-F1. To evaluate the held-out test split, set `EXPERIMENT_MODE = "compare"`, enable `FROZEN_TEST_EVALUATION`, and point `FROZEN_RUN_DIRS` to the four full binary runs.

---

## Training, Validation, and Testing

The three datasets have different roles.

### Training Set

The training set is used to update the model's parameters.

```text
data/Training_Cleaned_binary.csv
    ↓
Model training
    ↓
Updated model
```

### Validation Set

The validation set is evaluated after each training epoch.

The model with the highest validation macro-F1 is saved as the best model.

```text
data/Validation_Cleaned_binary.csv
    ↓
Validation macro-F1
    ↓
Best model selection
```

### Test Set

The test set is not used during model training or model selection.

The frozen comparison reloads each selected model and evaluates it on the test split.

```text
Best model
    ↓
data/Testing_Cleaned_binary.csv
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

The hybrid model uses a unidirectional LSTM.

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

Use the desktop application to classify a song or CSV with a trained binary checkpoint.

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
experiments/<model>_v26_binary_<run_id>/best_model.pt
```

The saved model is selected using validation macro-F1.

The model is then reloaded before final test-set evaluation and inference.

---

## Project Structure

```text
binary_v26/
│
├── lyric_classifier_v26_all_models.py
│
├── requirements.txt
├── README.md
│
├── ../data/Training_Cleaned_binary.csv
├── ../data/Validation_Cleaned_binary.csv
├── ../data/Testing_Cleaned_binary.csv
│
└── experiments/<model>_v26_binary_<run_id>/
    └── best_model.pt
```

---

## Notes and Limitations

### Maximum Sequence Length

The current implementation uses:

```text
MAX_LENGTH = 512
```

Long lyrics are split into overlapping windows. A song is classified as Explicit if any window is classified as Explicit.

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
1 = Explicit
```

The model's final output corresponds to one of these two classes.
