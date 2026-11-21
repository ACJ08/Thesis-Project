# Stages B and C

The training and validation CSVs are checked against the November 17 run hashes. This campaign uses the current reference labels. Human adjudication remains a separate diagnostic. The training runner does not read the student test split.

## Full campaign

The published campaign completed all 18 RoBERTa fits across six arms and seeds 67, 101, and 202. It used a three-epoch budget and the same per-seed song grouping and order. An optimizer step uses eight songs and weights classes by inverse training frequency. The runner uses BF16 autocast with FP32 parameters, AdamW at 2e-5, linear warmup and decay, and gradient clipping. The validation checkpoint rule is highest Macro-F1 under maximum severity. Both maximum severity and mean probability are scored from that checkpoint. As validation selects the checkpoint, use these scores for diagnosis.

- **Stage B:** Native RoBERTa, masked mean, CNN-only, LSTM-only, and CNN followed by LSTM. All four custom arms apply the same 256-dimensional projection and the same classifier after masked mean pooling. The CNN and LSTM choices are shared across their respective arms. This is a controlled definition of the full head. Its parameterization differs from the earlier frozen hybrid.
- **Stage C supervision:** Native RoBERTa is the shared reference. Window supervision averages inherited-label cross-entropy per song. The song objective applies cross-entropy to the mean window probability, evaluated in log space. Both arms use the same optimizer-step grouping, class weights and evaluation rules.
- **Stage C context:** The Longformer 512- and 4,096-token arms ran in the smoke test only. Validation length bins in the full campaign use RoBERTa tokens including the two boundary tokens in every arm.

Compare metrics within seed before averaging differences. For the heads, compare CNN and LSTM with masked mean and the combined head with each component. For supervision, compare song versus window within each of the two inference rules.

`full/comparison.csv` contains the selected validation scores for all 18 fits. `full/paired_deltas.csv` contains the within-seed contrasts. The per-fit directories contain the manifests, epoch histories, validation metrics by class and length bin, and row-level predictions.

Regenerate the paired contrasts from the published metrics with `python summarize_stages_bc_v26.py experiments/stages_bc_v26/full`.

For fresh training, use a separate checkout and move the published `experiments/stages_bc_v26/full` folder out of the runner's output path. The runner checks the CSV bytes against `experiments/roberta_v26_20261117T162744Z/manifest.json`. Use the archived `data/40k/Training_Cleaned.csv` as `data/Training_Cleaned.csv` in that checkout, and check out both cleaned input CSVs with CRLF line endings to match the recorded hashes. With a BF16-capable CUDA device and `roberta-base` in the local model cache, run `python run_stages_bc_v26.py --arms native mean cnn lstm cnn_lstm song_loss --seeds 67 101 202 --epochs 3`.
