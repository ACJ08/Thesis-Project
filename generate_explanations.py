import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Load your existing v26 script without starting training
ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "lyric_classifier_v26_all_models.py"

spec = importlib.util.spec_from_file_location("classifier", SCRIPT)
clf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clf)

# Load your trained hybrid checkpoint
run_dir = Path(clf.FROZEN_RUN_DIRS["hybrid"])
checkpoint = run_dir / "best_model.pt"

tokenizer = clf.AutoTokenizer.from_pretrained(
    clf.MODEL_NAME, use_fast=True
)

model = clf.RoBERTaCNNLSTM(num_classes=3).to(clf.DEVICE)
model.load_state_dict(
    torch.load(checkpoint, map_location=clf.DEVICE,
               weights_only=True)
)
model.eval()

# Select a validation song
df = pd.read_csv(clf.resolve_dataset_path(
    "Validation_Cleaned.csv"
))
df = clf.normalize_dataset_columns(df, "Validation dataset")

song = df.iloc[0]
lyrics = clf.clean_lyrics(song["text"])

print("Song:", song["song_name"])
print("Artist:", song["artist"])
print("Prediction:", clf.predict_lyrics(
    lyrics, model, tokenizer
))

# Generate explanations
lime_result = clf.explain_with_lime(
    lyrics, model, tokenizer
)

shap_result = clf.explain_with_shap(
    lyrics, model, tokenizer
)

# Save outputs
out = ROOT / "explanations"
out.mkdir(exist_ok=True)

lime_result.save_to_file(str(out / "lime.html"))

# Export SHAP contributions as CSV
values = np.asarray(shap_result.values[0])
tokens = shap_result.data[0]

if values.ndim == 2:
    probabilities = clf.model_predict_proba(
        [clf.prepare_explanation_text(
            lyrics, tokenizer, model
        )], model, tokenizer
    )[0]
    predicted_class = int(np.argmax(probabilities))
    values = values[:, predicted_class]

pd.DataFrame({
    "token": tokens,
    "shap_value": values
}).to_csv(out / "shap_values.csv", index=False)

print("Saved explanations to:", out)
