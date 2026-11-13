# Controlled cross-annotator experiment

Run commands in the project folder using the existing v26 Python environment. The runner imports `lyric_classifier_v26_all_models.py`, the metric/JSON utilities in `learning_curve_v26.py`, and `annotator_bias_reporting_v26.py`. Matplotlib and SciPy must be installed in that environment.

Preflight audit and deterministic allocation

```powershell
python -u annotator_bias_experiment_v26.py --plan
```

One-seed native smoke diagnostic, selecting LSTM-CNN for a quick executable check

```powershell
python -u annotator_bias_experiment_v26.py --seeds 67 --models lstm_cnn --smoke-test
```

The smoke diagnostic uses the native v26 smoke settings, currently 500 training songs, 100 validation songs and one epoch. It evaluates both 986-song test subsets. Its directory and result rows record `smoke_test=true`. Full experiments use the requested 4000/1000/986 allocation and original epoch limit. Both modes require the complete standardized allocation to pass the initial audit.

Full experiment, defaulting to all four architectures and seeds 67, 42 and 123

```powershell
python -u annotator_bias_experiment_v26.py
```

Resume the directory printed by the original invocation

```powershell
python -u annotator_bias_experiment_v26.py --resume "experiments/annotator_v26_<timestamp>_<suffix>"
```

Use `--a PATH`, `--b-validation PATH` and `--b-test PATH` for explicit source files. Defaults point directly to `data/Training_Cleaned.csv`, `data/Validation_Cleaned.csv` and `data/Testing_Cleaned.csv`. `--output-dir PATH` changes the parent directory. `--seeds` and `--models` select a prespecified experiment scope. Resume restores the original selection and verifies source, code, dependency and saved artifact hashes. A valid completed checkpoint can resume directly into evaluation. An interrupted training attempt restarts that training run. Each fresh invocation creates a unique directory, and each report generation creates a unique report subdirectory. The experiment lock prevents concurrent writes to one experiment.

## Allocation and overlap policy

The user supplies the human-annotation assignments for A and B. The runner reports dataset hashes and valid counts under native v26 preprocessing, normalization and review exclusions. Explicit pseudolabel provenance triggers an error. A class label alone cannot independently verify human annotation.

The complete A/B pool is audited before selection, including within A, within B and across A/B. The audit checks ID columns, normalized title/artist, normalized lyrics and five-word shingles. Near-matches require at least ten unique shingles in the shorter text, with Jaccard similarity at least 0.85 or containment at least 0.94. These are conservative lexical screening rules inherited from the learning-curve audit. They can miss short near-matches and paraphrases.

Duplicate matches form connected components. The runner retains one deterministic representative per component, prioritizing A, then source identifier and source row index. This choice preserves the smaller A pool where cross-pool matches occur. All excluded records and component membership are saved in the overlap report. Exclusions apply to in-memory experiment allocation and materialized run inputs. The original CSVs are preserved.

If either available pool contains fewer than 5986 songs, execution stops after saving the overlap report and a preflight summary. It requires sufficient valid unique songs to proceed with 4000 training, 1000 validation and 986 test songs.

For every seed, each pool is stratified to 5986 selected songs when necessary, then stratified into training, validation and test. The remaining songs form the unused pool. Counts, labels and original zero-based CSV record indices are saved in the sampling manifest, together with source identities and hashes. Source record indices exclude the CSV header and support multiline records. Class prevalence is preserved approximately within each annotator pool. It can differ between A and B.

## Native pipeline reuse

Native `main()` handles Hybrid training and `run_baseline()` handles the three baselines. Module-level output globals and dataset resolution are temporarily redirected into isolated run directories. Their setup expects a test CSV, so the runner supplies the training annotator's test partition while `RUN_FINAL_TEST=False` disables final test evaluation during training. Native baseline training also leaves test evaluation to the caller. Both functions select checkpoints from validation Macro-F1.

Native optimizer, class weighting, tokenizer construction, windowing, padding checks, schedulers, early stopping and epoch limits are retained. Smoke mode invokes v26's explicit diagnostic settings. The optional random explanation demo is disabled. Training duration includes setup and validation, excluding final test evaluation.

The selected checkpoint is reloaded through `make_baseline()` and `tokenizer_for()`. One model instance then evaluates A and B with native `LyricsDataset`, `evaluate_test_set()` and `save_predictions_and_metrics()`. The same checkpoint hash appears in both condition records. Each test prediction includes source identity, row index, song UID, actual/predicted IDs and labels. Full execution trains 24 checkpoints and produces 48 condition records.

## Statistics

Macro-F1 uses label IDs 0, 1 and 2 in Safe, Mild and Explicit order, including all classes when a bootstrap sample omits one class. Delta A compares A-to-A with B-to-A on identical A test songs. Delta B compares B-to-B with A-to-B on identical B test songs. Positive differences indicate a measured within-annotator advantage.

Each architecture/seed/target comparison uses 2000 paired song bootstrap resamples for percentile intervals and 2000 paired prediction-swap permutations for two-sided p-values. Holm correction treats both target comparisons across all selected architectures and seeds as one prespecified family, giving 24 tests for the default experiment. Changing experiment scope changes the family and must be reported.

Delta overall is the mean of Delta A and Delta B. Its across-seed descriptive interval uses a crossed bootstrap with shared song weights across seeds and seed-block resampling. Song sampling is stratified by annotator. Repeated appearances of one song in different seed test partitions receive the same bootstrap weight. These exploratory intervals condition on trained checkpoints and do not reproduce full training uncertainty. One seed provides conditional test-song uncertainty. Three seeds provide a limited view of optimization and partition variability. Across-seed standard deviations are descriptive and do not assume independent test datasets.

## Output layout

```text
annotator_v26_<timestamp>_<suffix>/
  experiment.json
  annotator_experiment_overlap_report.json
  annotator_experiment_sampling_manifest.json
  annotator_experiment_summary.md                 # Preflight summary
  runs/seed_67/hybrid/train_A/
    checkpoint/
      best_model.pt
      manifest.json
      epoch_history.csv
      validation_predictions.csv
      validation_metrics.json
      training_metadata.json
      completed.json
    evaluation/
      A_to_A_predictions.csv
      A_to_A_metrics.json
      A_to_B_predictions.csv
      A_to_B_metrics.json
      completed.json
  report_<suffix>/
    annotator_experiment_results.csv
    annotator_experiment_per_class.csv
    annotator_experiment_confusion.csv
    annotator_experiment_paired_statistics.csv
    annotator_experiment_seed_summary.csv
    annotator_experiment_summary.md               # Observed results and interpretation
    plots/
```

Other architectures, seeds and B training use the same layout. Temporary attempt directories retain materialized inputs and partial work after interruptions. Published checkpoints and paired evaluations have completion markers with artifact hashes. Reporting errors preserve completed training and prediction artifacts for resume.

PNG figures include condition Macro-F1 and accuracy, within-minus-cross differences, per-class F1, all four confusion matrices for every architecture, seed variability and all six directional errors. Plot error bars describe seed standard deviations. The statistical CSVs provide paired confidence intervals and corrected p-values.

The summary reports evidence of annotator-associated generalization. Annotation conventions, prevalence, song difficulty, genre/domain, collection composition and optimization can all contribute. Both annotators independently labeling the same songs is necessary to isolate annotation differences from collection differences. Earlier thesis experiments used the B source songs, which also limits an interpretation as fresh external validation.
