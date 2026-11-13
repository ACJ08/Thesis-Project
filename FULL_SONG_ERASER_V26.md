Run the complete-song experiment on the same 30 selected validation songs:

```powershell
python full_song_eraser_v26.py
```

The four models are the frozen v26 hybrid RoBERTa-CNN-LSTM, BERT-CNN-LSTM,
LSTM-CNN, and standalone RoBERTa configured in the classifier. Checkpoint and
vocabulary hashes and window policies are checked before inference. No training
or test-set evaluation is performed.

Every LIME/SHAP perturbation and ERASER deletion is applied to the entire cleaned
song. Original word positions and separators are preserved. The song is then
retokenized, all overlapping model windows are reconstructed, and the existing
maximum-severity pooler selects the song prediction and evidence window again.
No input is truncated to one evidence window. LIME uses an empty mask string
for actual deletion. SHAP uses positional binary word-presence features and an
empty-text background. Signed attributions are ranked in descending order for
the original predicted class.

The target class remains the original song prediction. Its score is the target
class probability in the evidence window selected by the full inference pipeline
after perturbation. This reproduces the repository's `model_predict_proba`
behavior. It is a full-pipeline score, not a calibrated whole-song probability.
The evidence window is allowed to change; the target class being scored is not.
The optimized batched scorer is checked against the original repository predictor
on every unperturbed model/song condition.

Defaults match the earlier corrected local experiment: seed 67, 1,000 LIME
perturbations, five SHAP forward/backward permutation passes, 20 random controls
per deletion level, and 10,000 song-bootstrap resamples. Every model uses the
same songs and positional random deletions. ERASER deletes ceil(5%, 10%, 20%) of
the full song's whitespace words, retaining at least one word. In addition to
confidence drops, the output records label flips and changes in window count
and evidence selection. The same random control is used for LIME and SHAP.

There are 120 model/song conditions, 720 method/deletion records, and 48 tests.
One-sided paired mean-drop t tests compare attributed deletion with zero drop
and matched random deletion. Holm correction covers all 48 comparisons as one
family. Bootstrap intervals are unadjusted. This assesses agreement between
attributions and sensitivity to deletion; it does not imply model intent or
semantic correctness. Interpret results as a validation-sample faithfulness
experiment, with limited explanation seeds and finite sampling budgets.

Results are written under `full_song_eraser_results/TIMESTAMP/`. Each model/song
folder includes full text, LIME HTML, combined LIME/SHAP highlighting, positional
attributions, and `completed.json`. The latter is the per-condition checkpoint.
Aggregate CSVs update after each completed song, with statistics finalized after
all requested conditions complete. Only a methodology status of `complete`
indicates a finished experiment.

If interrupted, resume the same run:

```powershell
python full_song_eraser_v26.py --resume "full_song_eraser_results\TIMESTAMP"
```

Use the same model selection, budgets, seed, input files, and classifier.
Configuration/input hashes are verified before completed conditions are reused.
A currently running incomplete song is recomputed. Full-song perturbations
require substantially more model inference than fixed-window explanations.
