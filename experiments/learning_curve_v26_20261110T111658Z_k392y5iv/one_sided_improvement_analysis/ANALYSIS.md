# One-sided improvement analysis

Exploratory reanalysis: H1 is Hybrid macro-F1 > baseline, alpha = 0.05. 10,000 paired song bootstrap resamples and 10,000 paired prediction-swap permutations per comparison. The 95% one-sided lower bound is the bootstrap 5th percentile. Original two-sided outputs are preserved.

All models were checked for identical ordered song names, artists, texts and true labels. No training or inference was repeated. Results describe seed 67 and sampling uncertainty over test songs, not variability from retraining. The direction was chosen after earlier results, so interpret as exploratory.

| Training size | Baseline | Delta macro-F1 (pp) | 95% lower bound (pp) | One-sided p | Holm p (3 per size) | Holm p (all 18) |
|---:|---|---:|---:|---:|---:|---:|
| 1000 | BERT_CNN_LSTM | 1.016 | 0.008 | 0.05379 | 0.05379 | 0.32277 |
| 1000 | LSTM_CNN | 11.856 | 10.635 | 0.00010 | 0.00030 | 0.00180 |
| 1000 | RoBERTa | 2.488 | 1.595 | 0.00010 | 0.00030 | 0.00180 |
| 2000 | BERT_CNN_LSTM | -0.615 | -1.633 | 0.83142 | 0.83142 | 1.00000 |
| 2000 | LSTM_CNN | 8.788 | 7.529 | 0.00010 | 0.00030 | 0.00180 |
| 2000 | RoBERTa | 1.515 | 0.591 | 0.00260 | 0.00520 | 0.02860 |
| 3000 | BERT_CNN_LSTM | 1.517 | 0.445 | 0.01220 | 0.01220 | 0.09899 |
| 3000 | LSTM_CNN | 8.431 | 7.209 | 0.00010 | 0.00030 | 0.00180 |
| 3000 | RoBERTa | 2.648 | 1.605 | 0.00020 | 0.00040 | 0.00240 |
| 4000 | BERT_CNN_LSTM | 0.074 | -0.890 | 0.45055 | 0.90111 | 1.00000 |
| 4000 | LSTM_CNN | 4.633 | 3.417 | 0.00010 | 0.00030 | 0.00180 |
| 4000 | RoBERTa | -0.760 | -1.599 | 0.93221 | 0.93221 | 1.00000 |
| 5000 | BERT_CNN_LSTM | 1.362 | 0.268 | 0.02050 | 0.02050 | 0.14349 |
| 5000 | LSTM_CNN | 5.885 | 4.749 | 0.00010 | 0.00030 | 0.00180 |
| 5000 | RoBERTa | 1.354 | 0.460 | 0.00510 | 0.01020 | 0.05099 |
| 5988 | BERT_CNN_LSTM | -0.494 | -1.363 | 0.81882 | 1.00000 | 1.00000 |
| 5988 | LSTM_CNN | 1.730 | 0.521 | 0.01100 | 0.03300 | 0.09899 |
| 5988 | RoBERTa | -0.131 | -1.004 | 0.59794 | 1.00000 | 1.00000 |

Holm corrections are shown both for the three comparisons within each training size and for the full family of 18. Bootstrap bounds are pointwise.
