Run from the project folder:

```powershell
python explain_eraser_lime_shap_v26.py
```

The default checkpoint is the frozen hybrid path configured in the v26 classifier.
To use a different frozen hybrid checkpoint:

```powershell
python explain_eraser_lime_shap_v26.py --checkpoint "C:\path\to\best_model.pt"
```

Defaults: 30 stratified validation songs, seed 67, 1,000 LIME samples,
five SHAP permutations per song, 20 random deletion repetitions per level,
and 10,000 bootstrap resamples. SHAP can take substantial time on CPU.
Inference batches up to eight texts (`--batch-size 1` disables batching),
and repeated perturbations are cached within each song.
No model training is performed. Dependencies are the project's requirements
plus scipy (normally installed with scikit-learn).

Each run creates a new timestamped folder under `eraser_lime_shap_results`.
Outputs include the selected validation row IDs, per-song evidence text,
LIME HTML, combined LIME/SHAP colored HTML, positional attribution CSVs,
per-song ERASER results, aggregate means, statistical results, and methodology.
Per-song diagnostics include LIME local fit R-squared and SHAP additivity residual.
Per-song results are saved after each completed song. A failed run remains
marked `running` and must not be treated as complete; automatic resume is not implemented.

Both methods explain the same original local predicted class on the same
evidence window selected by the hybrid model. Trailing words are trimmed to
reserve 24 BPE tokens for segmentation changes, matching the earlier ERASER
procedure. Perturbations exceeding the context are truncated and counted in
the methodology; inspect that count when interpreting results.
The truncation count concerns uniquely scored texts after caching.

SHAP is permutation SHAP on binary positional word-presence features with
an all-zero (empty-text) background. Missing words are deleted, preserving
the order of retained words. This gives both explainers and ERASER the same
word positions and deletion semantics. It differs from the classifier's
existing text-masker SHAP helper. Signed attributions are ranked descending,
so supporting words precede opposing words. Requested deletion percentages
can include nonpositive words when there are too few positive attributions.
LIME explicitly uses `mask_string=''`: its default positional masking would
insert `UNKWORDZ` placeholders instead of deleting words.

`--reuse-shap PATH` can reuse completed SHAP song outputs from an earlier run
while recomputing LIME and ERASER. The script checks checkpoint/validation paths,
selection, seed, SHAP budget, evidence text, target class, original probability,
word alignment, and SHAP additivity. Use only unchanged checkpoint and validation
files; path checks do not detect files overwritten in place. Reused attributions
retain the source run's perturbation provenance; the new truncation count covers
only newly scored texts.

ERASER removes 5%, 10%, and 20% of words (rounded up, retaining at least one).
Comprehensiveness is original target probability minus post-deletion target
probability; the target class and window never switch. Random deletion uses
the same count and the same random control for both methods on each song.
These are local-window results, not whole-song confidence results.

The statistics file tests mean confidence drop > 0 and mean confidence drop
above matched random deletion using one-sided paired t tests across songs.
Holm correction covers all 12 method/percentage/control tests. Bootstrap
95% percentile intervals resample songs separately within each comparison.
Intervals are unadjusted and may disagree with corrected decisions. These
tests assume independent songs and concern mean drops; with 30 songs,
inspect the per-song distribution and do not treat the result as proof of
universal explanation faithfulness. The three deletion levels reuse songs
and must not be pooled as independent observations. Neither a significant
drop alone nor a nonsignificant result establishes visual quality.

After a run completes, validate its outputs and generate the analysis report
and comparison figure:

```powershell
python analyze_lime_shap_eraser_v26.py "eraser_lime_shap_results\YOUR_RUN_FOLDER"
```

The analyzer also saves per-song success counts, local-fit diagnostics,
descriptive class breakdowns, and exploratory paired SHAP-versus-LIME tests.
It requires matplotlib for the figure and refuses incomplete runs.
