# v25 — hybrid + three baseline models

Put `lyric_classifier_v25.py` alongside your `data/Training.csv`, `data/Validation.csv`, and `data/Testing.csv` (or in the same folder as the v24 script). v25 is a single Python script; it does not import v24.

## Modes

Edit the configuration near the top of the script:

- `EXPERIMENT_MODE = "hybrid"` — original RoBERTa → CNN → unidirectional LSTM pipeline.
- `EXPERIMENT_MODE = "bert_cnn_lstm"` — BERT-base-uncased → same masked CNN and LSTM head.
- `EXPERIMENT_MODE = "lstm_cnn"` — **train-only** regex vocabulary → learned embeddings → LSTM → masked CNN.
- `EXPERIMENT_MODE = "roberta"` — RoBERTa-base sequence classifier, standard classification head.
- `EXPERIMENT_MODE = "all_baselines"` — train all three baselines sequentially, with independent outputs.
- `EXPERIMENT_MODE = "compare"` — evaluate all **frozen** best checkpoints on held-out test data, produce aggregate and paired comparison reports.

## Safe sequence

1. Set `SMOKE_TEST = True`, `EXPERIMENT_MODE = "all_baselines"`, `FROZEN_TEST_EVALUATION = False`. Run `python lyric_classifier_v25.py`. This checks training and checkpoint save/load on 500 train / 100 validation samples, for **one epoch each**. No held-out testing occurs. To smoke-test the hybrid too, run separately with `EXPERIMENT_MODE = "hybrid"`.
2. Review the smoke logs and artifacts, including provenance, validation metrics, and (for transformer models) padding reports.
3. Set `SMOKE_TEST = False`, `EXPERIMENT_MODE = "all_baselines"`. Train the full baselines, each with early stopping by validation Macro-F1. The hybrid may already have a valid v24 checkpoint; it does not need to be retrained just because v25 exists, provided the dataset hashes, preprocessing and model architecture match.
4. Freeze all four checkpoints and record each run directory. Set `FROZEN_RUN_DIRS` to the exact experiment folder for each model. Set `EXPERIMENT_MODE = "compare"`, `FROZEN_TEST_EVALUATION = True`, `SMOKE_TEST = False`. **Run this only after model choices are final.** The comparison evaluates each frozen model on the same test rows, without training, and writes `model_comparison.csv`, `paired_predictions.csv`, and `paired_macro_f1_bootstrap.csv` (2000 paired bootstrap resamples).

## Important methodology notes

- All models use the same source CSVs, `clean_lyrics`, label mapping, train/validation/test partitioning, class-weighted CE objective, gradient accumulation and validation Macro-F1 checkpoint selection. BERT and RoBERTa use their own tokenizers; LSTM-CNN learns a vocabulary from the training partition only. Therefore *token IDs and segmentation differ by architecture*, even though lyric cleaning is shared.
- The maximum input is 512 **tokens**, not the full song. LSTM-CNN uses its own regex tokens; BERT/RoBERTa use subword tokens. Document this difference as a limitation.
- `BASELINE_LSTM_LEARNING_RATE=1e-3` and transformer baseline LR `2e-5` are declared choices; document any hyperparameter tuning and use validation only.
- The final comparison script checks all four checkpoint hashes and dataset hashes before testing. If you change any CSV after training, **do not bypass the hash check**. Reconcile the experiments first.
- A paired bootstrap confidence interval on Macro-F1 differences is included; it is not a replacement for a pre-registered hypothesis test if your thesis explicitly requires one.
- The LSTM-CNN has learned word embeddings, not pretrained GloVe.
- Smoke-run validation metrics are engineering diagnostics and must not be reported as thesis results.

## Runtime dependencies

The original script's requirements still apply: `torch`, `transformers`, `scikit-learn`, `pandas`, `numpy`, `tqdm`, `lime`, and `shap`. BERT and RoBERTa weights are downloaded the first time unless cached. Train the baselines sequentially to avoid holding multiple transformer models on your GPU at once.
