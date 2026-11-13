"""Validate and report a completed full-song, multi-model ERASER experiment."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from full_song_eraser_v26 import WordPositions


def table(frame):
    lines = ['| ' + ' | '.join(map(str, frame.columns)) + ' |',
             '| ' + ' | '.join(['---']*len(frame.columns)) + ' |']
    lines.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in frame.itertuples(index=False, name=None))
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    root = args.results
    metadata = json.loads((root/'methodology.json').read_text(encoding='utf-8'))
    if metadata['status'] != 'complete':
        raise ValueError('Cannot report incomplete experiment')
    frame = pd.read_csv(root/'eraser_per_song.csv')
    original = pd.read_csv(root/'original_predictions.csv')
    stats = pd.read_csv(root/'faithfulness_statistics.csv')
    selection = pd.read_csv(root/'selected_songs.csv')
    models = metadata['config']['models']
    expected = len(models)*len(selection)
    assert len(original) == expected and len(frame) == expected*6 and len(stats) == len(models)*12
    assert not original.duplicated(['model', 'validation_row']).any()
    assert not frame.duplicated(['model', 'validation_row', 'method', 'erasure_percent']).any()
    assert frame.groupby(['model', 'validation_row']).size().eq(6).all()
    assert set(frame.model) == set(models)
    assert np.isfinite(frame.select_dtypes(include='number')).all().all()
    assert np.allclose(frame.original_score-frame.post_erasure_score, frame.comprehensiveness, atol=2e-7)
    assert np.allclose(original[['p_safe', 'p_mild', 'p_explicit']].sum(axis=1), 1, atol=2e-7)
    assert frame.groupby(['model', 'validation_row', 'erasure_percent']).random_comprehensiveness_mean.nunique().eq(1).all()
    assert frame.class_flipped.eq(frame.post_erasure_label.ne(frame.target_label)).all()
    assert frame.groupby(['model', 'validation_row']).original_score.nunique().eq(1).all()
    diagnostics = []
    for record in original.itertuples(index=False):
        folder = root/record.model/f'song_{record.validation_row:06d}'
        saved = json.loads((folder/'completed.json').read_text(encoding='utf-8'))
        positions = WordPositions((folder/'song.txt').read_text(encoding='utf-8'))
        attributions = pd.read_csv(folder/'attributions.csv', keep_default_na=False)
        assert attributions.word.tolist() == positions.words and len(positions.words) == record.word_count
        assert np.isfinite(attributions[['lime', 'shap']].to_numpy(dtype=float)).all()
        assert abs(saved['shap_additivity_residual']) < 1e-5
        subset = frame.loc[frame.model.eq(record.model) & frame.validation_row.eq(record.validation_row)]
        assert subset.target_label.eq(record.predicted_label).all()
        assert np.allclose(subset.original_score, record.score)
        for entry in subset.itertuples(index=False):
            ranking = json.loads(entry.erased_positions)
            assert len(ranking) == entry.erased_word_count == len(set(ranking))
            assert min(ranking) >= 0 and max(ranking) < record.word_count
        diagnostics.append(dict(model=record.model, validation_row=record.validation_row,
                                **{key: saved[key] for key in ['lime_local_fit_r2', 'shap_additivity_residual',
                                                               'unique_texts_scored', 'windows_scored']}))
    diagnostics = pd.DataFrame(diagnostics)
    diagnostics.to_csv(root/'explanation_diagnostics.csv', index=False)
    counts = frame.assign(positive=frame.comprehensiveness.gt(0),
        beats_random=frame.comprehensiveness.gt(frame.random_comprehensiveness_mean)).groupby(
        ['model', 'method', 'erasure_percent']).agg(songs=('validation_row', 'count'),
        positive_drop_songs=('positive', 'sum'), above_random_songs=('beats_random', 'sum'),
        class_flip_rate=('class_flipped', 'mean'), random_class_flip_rate=('random_class_flip_rate', 'mean')).reset_index()
    counts.to_csv(root/'success_and_flip_rates.csv', index=False)
    summaries = []
    decisions = []
    for model in models:
        for method in ['LIME', 'SHAP']:
            part = stats.loc[stats.model.eq(model) & stats.method.eq(method)]
            significant = {}
            for control in ['zero', 'random']:
                significant[control] = part.loc[part.control.eq(control) & part.reject_at_05, 'erasure_percent'].tolist()
            decisions.append(dict(Model=model, Method=method,
                **{'Significant drop levels': ', '.join(f'{x}%' for x in significant['zero']) or 'None',
                   'Significantly above random': ', '.join(f'{x}%' for x in significant['random']) or 'None'}))
            for percent in [5, 10, 20]:
                zero = part.loc[part.control.eq('zero') & part.erasure_percent.eq(percent)].iloc[0]
                random = part.loc[part.control.eq('random') & part.erasure_percent.eq(percent)].iloc[0]
                group = frame.loc[frame.model.eq(model) & frame.method.eq(method) & frame.erasure_percent.eq(percent)]
                summaries.append(dict(Model=model, Method=method, Deleted=f'{percent}%',
                    **{'Drop (pp)': f'{100*zero.mean_difference:.2f}',
                       '95% CI (pp)': f'[{100*zero.bootstrap_ci95_low:.2f}, {100*zero.bootstrap_ci95_high:.2f}]',
                       'Random drop (pp)': f'{100*group.random_comprehensiveness_mean.mean():.2f}',
                       'Holm p: drop > 0': f'{zero.p_holm:.6g}', 'Holm p: above random': f'{random.p_holm:.6g}'}))
    decision_frame = pd.DataFrame(decisions)
    decision_frame.to_csv(root/'hypothesis_decisions.csv', index=False)
    metrics = []
    labels = ['Safe', 'Mild', 'Explicit']
    for model, group in original.groupby('model'):
        metrics.append(dict(model=model, songs=len(group), correct=int(group.actual_label.eq(group.predicted_label).sum()),
            accuracy=accuracy_score(group.actual_label, group.predicted_label),
            macro_f1=f1_score(group.actual_label, group.predicted_label, labels=labels, average='macro', zero_division=0)))
    pd.DataFrame(metrics).to_csv(root/'descriptive_prediction_metrics.csv', index=False)
    per_model = diagnostics.groupby('model').agg(median_lime_fit=('lime_local_fit_r2', 'median'),
        minimum_lime_fit=('lime_local_fit_r2', 'min'), max_shap_residual=('shap_additivity_residual', lambda x: x.abs().max()),
        unique_perturbations=('unique_texts_scored', 'sum'), neural_windows_scored=('windows_scored', 'sum')).reset_index()
    per_model.to_csv(root/'model_diagnostics.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = dict(hybrid='RoBERTa-CNN-LSTM', bert_cnn_lstm='BERT-CNN-LSTM', lstm_cnn='LSTM-CNN', roberta='RoBERTa')
    for control, title, filename in [('zero', 'Confidence drop', 'full_song_confidence_drops.png'),
                                      ('random', 'Extra confidence drop above random', 'full_song_above_random.png')]:
        figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True, squeeze=False)
        for axis, model in zip(axes.flat, models):
            for method, color, shift in [('LIME', '#2274a5', -.2), ('SHAP', '#d9692b', .2)]:
                part = stats.loc[stats.model.eq(model) & stats.method.eq(method) & stats.control.eq(control)].sort_values('erasure_percent')
                x = part.erasure_percent.to_numpy()+shift
                y = 100*part.mean_difference.to_numpy()
                lo, hi = 100*part.bootstrap_ci95_low.to_numpy(), 100*part.bootstrap_ci95_high.to_numpy()
                axis.plot(x, y, 'o-', label=method, color=color)
                axis.vlines(x, lo, hi, color=color, linewidth=2)
                for xi, high, significant in zip(x, hi, part.reject_at_05):
                    if significant:
                        axis.annotate('*', (xi, high), xytext=(0, 3), textcoords='offset points', ha='center', color=color)
            axis.axhline(0, color='#777777', linestyle='--', linewidth=.8)
            axis.set_title(names[model])
            axis.set_xticks([5, 10, 20], ['5%', '10%', '20%'])
            axis.grid(axis='y', alpha=.15)
            axis.spines[['top', 'right']].set_visible(False)
        for axis in list(axes.flat)[len(models):]:
            axis.set_visible(False)
        axes[0, 0].legend(frameon=False)
        figure.supylabel('Pipeline score change (percentage points)')
        figure.supxlabel('Full-song words deleted')
        figure.suptitle(f'{title} · {len(selection)} validation songs', fontsize=14)
        figure.text(.5, .015, f'Bars: unadjusted 95% song-bootstrap intervals. * Holm-adjusted p < .05 across {len(stats)} one-sided tests.', ha='center', fontsize=8)
        figure.tight_layout(rect=(.025, .06, 1, .95))
        figure.savefig(root/filename, dpi=180)
        plt.close(figure)
    report = f'''# Full-song LIME/SHAP ERASER analysis

Run `{root.name}` evaluated all {len(selection)} previously selected validation songs
with {len(models)} frozen models. All {len(frame)} method/deletion records passed
the paired-data, attribution-alignment, and additivity consistency checks.

## Decisions

{table(decision_frame)}

"Significant drop" tests whether the mean original-class score reduction is greater
than zero. "Above random" is the stronger specificity test: highly attributed
words cause a larger mean reduction than equal-count random deletions. Use both
columns when describing faithfulness. Failure to reject means insufficient
evidence, not evidence that the model selects tokens randomly.

## Effect sizes and statistical results

{table(pd.DataFrame(summaries))}

Values are percentage points. Tests are one-sided paired t tests across songs.
Holm correction treats all {len(stats)} requested model/method/level/control
comparisons as one family at alpha .05. Intervals are unadjusted 95% percentile
song-bootstrap intervals with {metadata['config']['bootstrap_resamples']} resamples;
their procedure differs from the corrected tests. Repeated deletion levels are
not treated as independent songs. No between-model superiority test is performed.

![Full-song effects above random](full_song_above_random.png)

## What was explained

Entire cleaned songs were explained without truncating them to an evidence window.
Every masked/deleted input was retokenized and all overlapping windows recomputed.
The original predicted class was the fixed target. The existing maximum-severity
pooler and evidence-window selection were rerun after every perturbation. Thus,
the evidence window and predicted label could change. The optimized unperturbed
scores were checked against the repository's original prediction function for
every model/song condition.

The measured score is the original target class probability in the dynamically
selected evidence window. It is the repository's full-pipeline output score,
not a calibrated probability that the whole song belongs to a class. This
distinction matters because severity pooling is discrete and window selection
can switch. The output also records actual song-label flips, alongside confidence
changes. This differs from the earlier fixed-local-window experiment.

LIME used {metadata['config']['lime_samples']} actual word-deletion perturbations,
and SHAP used {metadata['config']['shap_permutations']} forward/backward permutation
passes over positional word-presence features with an empty-text background.
Original whitespace was preserved before the common preprocessing function.
Rankings used signed target-class attributions in descending order. At each
5%, 10%, and 20% level, deletion counts were rounded up, retaining at least one
word. {metadata['config']['random_repeats']} random deletions were averaged per
song and level. LIME and SHAP shared the same random controls, and all models
used the same song/position selections. Seed: {metadata['config']['seed']}.

## Explanation diagnostics

{table(per_model.round(6))}

Additivity checks verify bookkeeping, not SHAP convergence. Finite explanation
budgets, one explanation seed, surrogate fit variation, only 30 validation songs,
and the paired t-test assumptions limit generalization. Word deletion can create
unusual inputs. Faithfulness means attribution agreement with this model's
sensitivity; it does not establish semantic correctness or human-like intent.
Attributions can faithfully explain an incorrect prediction. Original-song
accuracy and Macro-F1 are saved as descriptive context, without testing the
model-performance hypothesis on this small validation sample.

## Files

- `original_predictions.csv`: all original full-song model predictions and pipeline scores.
- `eraser_per_song.csv`: attributed/random effects and label flips.
- `faithfulness_statistics.csv`: every test, confidence interval, and corrected decision.
- `success_and_flip_rates.csv`: individual-song success counts and song-label flip rates.
- `hypothesis_decisions.csv`: compact decision table above.
- `descriptive_prediction_metrics.csv`: accuracy and Macro-F1 on these 30 validation songs only.
- `model/song_*/explanations.html`: both methods' complete-song highlighting.
- `methodology.json`: exact budgets, input hashes, score definition, and checkpoint provenance.
'''
    (root/'ANALYSIS.md').write_text(report, encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
