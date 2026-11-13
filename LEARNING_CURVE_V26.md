# v26 learning-curve experiment

Run from the repository root with the Python environment used by v26. Install matplotlib for plots and SciPy for optional fitting if they are missing.

```powershell
python -m pip install matplotlib scipy
python -u learning_curve_v26.py --plan
python -u learning_curve_v26.py
```

The default is four models, six training sizes, and seed 67 (24 configurations). The inputs are `data/Training_Cleaned.csv`, `data/Validation_Cleaned.csv`, and `data/Testing_Cleaned.csv`. The runner requires exactly 5,000 usable validation records and 4,998 usable test records in their original order. Use `--validation PATH --test PATH` to point to the existing fixed files if they are elsewhere. The full training size is derived from valid rows under v26 preprocessing and review settings. No new partitions are generated.

```powershell
python -u learning_curve_v26.py --models roberta hybrid --sizes 1000 2000 3000 4000 5000 ALL --seeds 67 68 69 --output-dir experiments --extrapolate 8000 12000
python -u learning_curve_v26.py --resume "experiments/<experiment-directory>"
```

A successful `--plan` creates a resumable experiment without training. Use the same selection on the plan command as for the intended run. Resume restores the selection from metadata, revalidates the inputs and skips published configurations. A failed or interrupted configuration restarts from its initial model state. Run one process per experiment directory. A new invocation without `--resume` creates a fresh directory. On resume, `--extrapolate` can change the forecasting sizes and generates a new report without overwriting earlier reports.

## Outputs

```text
experiment/
  experiment.json
  preflight_*/
    passed.json or leakage.json
  sampling_manifests/
    seed_67_1000.json
  configurations/seed_67/RoBERTa_1000/
    checkpoints/RoBERTa_1000_best_model.pt
    metrics/RoBERTa_1000_metrics.json
    metrics/                         # Also split metrics, v26 manifest and epoch history
    predictions/                     # Separate ordered validation and test predictions
  session_*/                        # Materialized training-only CSVs
  report_*/
    learning_curve_results.csv
    paired_macro_f1_bootstrap.csv    # When Hybrid and a baseline are both selected
    power_law_extrapolations.csv     # When requested
    plots/
```

Sampling manifests use zero-based CSV data-record indices (excluding the header). Each seed defines a single nested, proportionally interleaved class ordering. Every model receives the same selected records in source order. v26 shuffles only the training loader. Each configuration calls native v26 training with fresh initialization (a train-only vocabulary and random weights for LSTM-CNN), native class weights, padding checks, and validation Macro-F1 checkpoint selection. The optional explanation demo is disabled. The runner reloads the best checkpoint and runs the native test evaluator once per completed configuration. Recorded training duration includes v26 setup and validation and excludes held-out test evaluation. GPU cleanup runs between configurations.

## Interpretation

The preflight compares each pair of splits using normalized title and artist, common ID columns when present, and normalized lyrics. Its lexical near-duplicate check uses unique five-word shingles, requiring at least ten shingles in the shorter song and either Jaccard similarity of at least 0.85 or containment of at least 0.94. It can miss heavily rewritten songs and very short near-matches. Any finding stops training and records both source rows and the match reason. Investigate the findings and provenance with the dataset owner before proceeding.

The root training CSV has no annotation provenance fields. It does not establish whether each label came from a verified human annotation or whether pseudolabels are present. The runner reports any available source metadata, review flags, or predicted-label fallback and `--require-verified-human` blocks execution while provenance remains unverified. Interpret a run on the supplied file as a learning curve for its existing labels. Earlier thesis work may have used the fixed holdouts, so their scores are conditional on past model choices and may not represent performance on a fresh dataset.

Solid lines show measured seed means. Error bars show between-seed standard deviations when multiple seeds are run. Optional dashed power-law predictions are fitted separately to test accuracy and Macro-F1 as descriptive post hoc analyses. They do not select models or tune training. With three or more seeds the forecast includes an exploratory bootstrap interval over whole seed trajectories, conditional on the fixed datasets. Intervals with very few seeds are unstable, and no interval is reported for a single seed. A fitted ceiling is weakly identified across this range of training sizes. Forecasts far beyond it are unreliable, and a desired score at or above the ceiling has no finite sample-size solution under the fit. There is no assumed 90% target.
