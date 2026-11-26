# November 7 v26 checkpoint error-rate statistics

The **RoBERTa–CNN–LSTM (Hybrid)** checkpoint misclassified **1,345 of 4,999 held-out songs (26.91%)**. Its saved validation error was **1,276 of 5,000 songs (25.52%)**. The checkpoint was saved at epoch 2 with seed 67. The training run started on November 7 in Manila. The held-out comparison folder is dated November 7 UTC, which was November 8 in Manila. [Hybrid checkpoint manifest](../hybrid_v26_20261107T000459Z/manifest.json)

## Where the results are recorded

| File | Contents |
|---|---|
| [model_comparison.csv](model_comparison.csv) | Test accuracy, macro and weighted F1, and the exact checkpoint hashes for all four models. The saved checkpoint paths refer to the original checkout. |
| [model_comparison_confusion.csv](model_comparison_confusion.csv) | Counts for every actual and predicted class combination. These counts are enough to recalculate all the test rates below. |
| [paired_predictions.csv](paired_predictions.csv) | One row per test song with its actual class and all four predictions. Label IDs are 0 = Safe, 1 = Mild, and 2 = Explicit. |
| Per-model test metrics | Accuracy, class precision, recall, F1 and support, the confusion matrix, and all six directional error counts. [Hybrid](hybrid/test_metrics.json) · [RoBERTa](roberta/test_metrics.json) · [LSTM–CNN](lstm_cnn/test_metrics.json) · [BERT–CNN–LSTM](bert_cnn_lstm/test_metrics.json) |
| Per-model test predictions | Song, artist, lyrics, actual label, predicted label, and a correctness flag. [Hybrid](hybrid/test_predictions.csv) · [RoBERTa](roberta/test_predictions.csv) · [LSTM–CNN](lstm_cnn/test_predictions.csv) · [BERT–CNN–LSTM](bert_cnn_lstm/test_predictions.csv) |
| Per-model validation metrics | Results for the saved checkpoints on the 5,000-song validation set. [Hybrid](../hybrid_v26_20261107T000459Z/validation_metrics.json) · [RoBERTa](../roberta_v26_20261107T000459Z/validation_metrics.json) · [LSTM–CNN](../lstm_cnn_v26_20261107T000459Z/validation_metrics.json) · [BERT–CNN–LSTM](../bert_cnn_lstm_v26_20261107T000459Z/validation_metrics.json) |
| Per-model validation predictions | Song-level actual and predicted classes from the saved checkpoints. [Hybrid](../hybrid_v26_20261107T000459Z/validation_predictions.csv) · [RoBERTa](../roberta_v26_20261107T000459Z/validation_predictions.csv) · [LSTM–CNN](../lstm_cnn_v26_20261107T000459Z/validation_predictions.csv) · [BERT–CNN–LSTM](../bert_cnn_lstm_v26_20261107T000459Z/validation_predictions.csv) |

The local checkpoint hashes match the comparison record. The paired test predictions match each model’s individual predictions and reconstruct all four saved confusion matrices exactly. To recreate the rates, tally actual and predicted label pairs, then divide the relevant error count by the stated denominator. Overall error is the sum of the off-diagonal counts divided by the number of songs, or `1 − accuracy`. The prediction exports are sufficient for this calculation.

## Overall error rates

| Model | Saved epoch | Test correct | Test errors / 4,999 | Test accuracy | Validation errors / 5,000 |
|---|---:|---:|---:|---:|---:|
| **RoBERTa–CNN–LSTM (Hybrid)** | 2 | 3,654 | **1,345 (26.91%)** | 73.09% | 1,276 (25.52%) |
| RoBERTa | 2 | 3,738 | 1,261 (25.23%) | 74.77% | 1,265 (25.30%) |
| LSTM–CNN | 3 | 3,578 | 1,421 (28.43%) | 71.57% | 1,364 (27.28%) |
| BERT–CNN–LSTM | 1 | 3,564 | 1,435 (28.71%) | 71.29% | 1,332 (26.64%) |

RoBERTa made 84 fewer test errors than Hybrid, a difference of 1.68 percentage points. Mild had the highest miss rate for every model and accounted for 793 of Hybrid’s 1,345 errors (58.96%).

## Held-out errors by actual class

Each percentage uses the number of songs with the actual class as its denominator. The total for each class is its false-negative rate, or `1 − recall`.

| Actual → predicted | Hybrid | RoBERTa | LSTM–CNN | BERT–CNN–LSTM |
|---|---:|---:|---:|---:|
| **Safe → any other label** (n = 2,412) | **373 (15.46%)** | 256 (10.61%) | 439 (18.20%) | 552 (22.89%) |
| Safe → Mild | 324 (13.43%) | 236 (9.78%) | 369 (15.30%) | 517 (21.43%) |
| Safe → Explicit | 49 (2.03%) | 20 (0.83%) | 70 (2.90%) | 35 (1.45%) |
| **Mild → any other label** (n = 1,640) | **793 (48.35%)** | 835 (50.91%) | 802 (48.90%) | 686 (41.83%) |
| Mild → Safe | 321 (19.57%) | 362 (22.07%) | 314 (19.15%) | 238 (14.51%) |
| Mild → Explicit | 472 (28.78%) | 473 (28.84%) | 488 (29.76%) | 448 (27.32%) |
| **Explicit → any other label** (n = 947) | **179 (18.90%)** | 170 (17.95%) | 180 (19.01%) | 197 (20.80%) |
| Explicit → Safe | 12 (1.27%) | 17 (1.80%) | 14 (1.48%) | 7 (0.74%) |
| Explicit → Mild | 167 (17.63%) | 153 (16.16%) | 166 (17.53%) | 190 (20.06%) |

## False positives and severity errors

A class’s false-positive rate divides incorrect predictions of that class by all songs whose actual label is in the other two classes. Severity rates use all 4,999 test songs, with the order Safe < Mild < Explicit.

| Held-out statistic | Hybrid | RoBERTa | LSTM–CNN | BERT–CNN–LSTM |
|---|---:|---:|---:|---:|
| False Safe / 2,587 | 333 (12.87%) | 379 (14.65%) | 328 (12.68%) | 245 (9.47%) |
| False Mild / 3,359 | 491 (14.62%) | 389 (11.58%) | 535 (15.93%) | 707 (21.05%) |
| False Explicit / 4,052 | 521 (12.86%) | 493 (12.17%) | 558 (13.77%) | 483 (11.92%) |
| Predicted severity too low | 500 (10.00%) | 532 (10.64%) | 494 (9.88%) | 435 (8.70%) |
| Predicted severity too high | 845 (16.90%) | 729 (14.58%) | 927 (18.54%) | 1,000 (20.00%) |
| Two-level error (Safe ↔ Explicit) | 61 (1.22%) | 37 (0.74%) | 84 (1.68%) | 42 (0.84%) |
| Balanced error rate (mean of class miss rates) | 27.57% | 26.49% | 28.70% | 28.51% |

Of the 1,289 songs Hybrid predicted as Explicit, 521 (40.42%) had a Safe or Mild reference label. That percentage uses the songs flagged Explicit as its denominator.
