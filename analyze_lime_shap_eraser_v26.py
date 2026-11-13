"""Validate a completed LIME/SHAP ERASER run and write a Markdown analysis."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp


def markdown_table(frame):
    columns = list(frame.columns)
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---']*len(columns)) + ' |']
    for values in frame.itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join(str(x) for x in values) + ' |')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    root = args.results
    metadata = json.loads((root/'methodology.json').read_text(encoding='utf-8'))
    if metadata['status'] != 'complete':
        raise ValueError('Only completed runs can be analyzed')
    frame = pd.read_csv(root/'eraser_per_song.csv')
    selected = pd.read_csv(root/'selected_songs.csv')
    stats = pd.read_csv(root/'faithfulness_statistics.csv')
    summary = pd.read_csv(root/'summary.csv')
    assert len(frame) == len(selected)*6
    assert not frame.duplicated(['validation_row', 'method', 'erasure_percent']).any()
    assert set(frame.validation_row) == set(selected.validation_row)
    assert frame.groupby('validation_row').size().eq(6).all()
    assert np.isfinite(frame.select_dtypes(include='number')).all().all()
    assert np.allclose(frame.comprehensiveness,
                       frame.original_probability-frame.post_erasure_probability, atol=1e-12)
    assert frame.groupby('validation_row').original_probability.nunique().eq(1).all()
    assert frame.groupby(['validation_row', 'erasure_percent']).random_comprehensiveness_mean.nunique().eq(1).all()
    assert set(stats.control) == {'zero', 'random'} and len(stats) == 12
    diagnostics = []
    for row in selected.validation_row:
        folder = root/f'song_{row:06d}'
        attributions = pd.read_csv(folder/'attributions.csv')
        assert len(attributions) == frame.loc[frame.validation_row.eq(row), 'word_count'].iloc[0]
        assert np.isfinite(attributions[['lime', 'shap']]).all().all()
        diagnostics.append(dict(validation_row=row, **json.loads((folder/'diagnostics.json').read_text())))
    diagnostics = pd.DataFrame(diagnostics)
    diagnostics.to_csv(root/'explanation_diagnostics.csv', index=False)
    # Permutation masks depend on seed, background, and feature count, not model
    # outputs. Reconstruct the same coalitions without rerunning neural inference.
    import shap
    from eraser_comprehensiveness_v26 import load_classifier
    clf = load_classifier(Path(metadata['options']['classifier']).resolve())
    tokenizer = clf.AutoTokenizer.from_pretrained(clf.MODEL_NAME, use_fast=True)
    capacity = clf.MAX_LENGTH-sum(map(len, clf.special_token_prefix_suffix(tokenizer)))
    audit = []
    for row in selected.validation_row:
        words = (root/f'song_{row:06d}'/'evidence.txt').read_text(encoding='utf-8').split()
        token_counts = {}

        def audit_masks(masks):
            for mask in masks:
                text = clf.clean_lyrics(' '.join(w for w, keep in zip(words, mask) if keep > .5))
                if text not in token_counts:
                    token_counts[text] = len(clf.full_token_ids(text, tokenizer))
            return np.asarray(masks).sum(axis=1)

        explainer = shap.Explainer(audit_masks, np.zeros((1, len(words))), algorithm='permutation',
                                  seed=metadata['seed']+int(row))
        explainer(np.ones((1, len(words))),
                  max_evals=metadata['options']['shap_permutations']*(2*len(words)+1), silent=True)
        audit.append(dict(validation_row=row, unique_shap_perturbations=len(token_counts),
                          overflow=sum(n > capacity for n in token_counts.values()),
                          max_bpe_tokens=max(token_counts.values()), capacity=capacity))
    audit = pd.DataFrame(audit)
    audit.to_csv(root/'shap_context_audit.csv', index=False)
    drop_rates = frame.assign(positive_drop=frame.comprehensiveness.gt(0),
                             beats_random=frame.comprehensiveness.gt(frame.random_comprehensiveness_mean))
    rates = drop_rates.groupby(['method', 'erasure_percent']).agg(
        songs_with_positive_drop=('positive_drop', 'sum'), songs_beating_random=('beats_random', 'sum')).reset_index()
    rates.to_csv(root/'per_song_success_counts.csv', index=False)
    merged = summary.merge(stats.loc[stats.control.eq('zero'),
        ['method', 'erasure_percent', 'bootstrap_ci95_low', 'bootstrap_ci95_high', 'p_holm']],
        on=['method', 'erasure_percent'])
    merged = merged.merge(stats.loc[stats.control.eq('random'), ['method', 'erasure_percent', 'p_holm']],
                          on=['method', 'erasure_percent'], suffixes=('_zero', '_random'))
    table = pd.DataFrame({
        'Method': merged.method, 'Deleted': merged.erasure_percent.astype(str)+'%',
        'Mean drop (pp)': merged.comprehensiveness_mean.map(lambda x: f'{100*x:.2f}'),
        '95% CI (pp)': [f'[{100*l:.2f}, {100*h:.2f}]' for l, h in zip(merged.bootstrap_ci95_low, merged.bootstrap_ci95_high)],
        'Random drop (pp)': merged.random_comprehensiveness_mean.map(lambda x: f'{100*x:.2f}'),
        'Holm p: drop > 0': merged.p_holm_zero.map(lambda x: f'{x:.6g}'),
        'Holm p: above random': merged.p_holm_random.map(lambda x: f'{x:.6g}')})
    # Exploratory direct method comparisons form a separate three-test family.
    rng = np.random.default_rng(metadata['seed'])
    comparisons = []
    for percent, group in frame.groupby('erasure_percent'):
        paired = group.pivot(index='validation_row', columns='method', values='comprehensiveness')
        difference = (paired.SHAP-paired.LIME).to_numpy()
        boot = np.array([rng.choice(difference, len(difference), replace=True).mean() for _ in range(10000)])
        low, high = np.quantile(boot, [.025, .975])
        comparisons.append(dict(erasure_percent=percent, shap_minus_lime_mean=difference.mean(),
                                bootstrap_ci95_low=low, bootstrap_ci95_high=high,
                                p_two_sided=float(ttest_1samp(difference, 0).pvalue)))
    previous = 0.
    for rank, i in enumerate(sorted(range(3), key=lambda i: comparisons[i]['p_two_sided'])):
        previous = max(previous, min(1., (3-rank)*comparisons[i]['p_two_sided']))
        comparisons[i]['p_holm'] = previous
    pd.DataFrame(comparisons).to_csv(root/'exploratory_shap_vs_lime.csv', index=False)
    comparison_table = pd.DataFrame({
        'Deleted': [f"{x['erasure_percent']}%" for x in comparisons],
        'SHAP minus LIME (pp)': [f"{100*x['shap_minus_lime_mean']:.2f}" for x in comparisons],
        '95% CI (pp)': [f"[{100*x['bootstrap_ci95_low']:.2f}, {100*x['bootstrap_ci95_high']:.2f}]" for x in comparisons],
        'Holm p (two-sided)': [f"{x['p_holm']:.6g}" for x in comparisons]})
    by_class = frame.groupby(['actual_label', 'method', 'erasure_percent']).agg(
        songs=('validation_row', 'count'), mean_drop=('comprehensiveness', 'mean'),
        random_mean_drop=('random_comprehensiveness_mean', 'mean')).reset_index()
    by_class.to_csv(root/'descriptive_by_actual_class.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for axis, control, title in zip(axes, ['zero', 'random'],
        ['Confidence drop after attributed-word deletion', 'Extra confidence drop above random deletion']):
        for method, color, shift in [('LIME', '#2274a5', -.25), ('SHAP', '#d9692b', .25)]:
            part = stats.loc[stats.method.eq(method) & stats.control.eq(control)].sort_values('erasure_percent')
            x = part.erasure_percent.to_numpy()+shift
            y = 100*part.mean_difference.to_numpy()
            low, high = 100*part.bootstrap_ci95_low.to_numpy(), 100*part.bootstrap_ci95_high.to_numpy()
            axis.plot(x, y, 'o-', label=method, color=color)
            axis.vlines(x, low, high, color=color, linewidth=2)
            for xi, hi, significant in zip(x, high, part.reject_at_05):
                if significant:
                    axis.annotate('*', (xi, hi), xytext=(0, 3), textcoords='offset points',
                                  ha='center', color=color, fontsize=14)
        axis.axhline(0, color='#777777', linewidth=.8, linestyle='--')
        axis.set_title(title, fontsize=10)
        axis.set_xticks([5, 10, 20], ['5%', '10%', '20%'])
        axis.set_xlabel('Words deleted')
        axis.grid(axis='y', alpha=.15)
        axis.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Probability change (percentage points)')
    axes[0].legend(frameon=False)
    fig.suptitle(f'LIME and SHAP ERASER · {len(selected)} validation songs', fontsize=14)
    fig.text(.5, .015, 'Bars: unadjusted 95% song-bootstrap intervals. * Holm-adjusted p < .05 across 12 one-sided tests.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, .95))
    fig.savefig(root/'faithfulness_comparison.png', dpi=180)
    plt.close(fig)
    decisions = []
    for method in ('LIME', 'SHAP'):
        method_stats = stats.loc[stats.method.eq(method)]
        zero_levels = method_stats.loc[method_stats.control.eq('zero') & method_stats.reject_at_05, 'erasure_percent'].tolist()
        random_levels = method_stats.loc[method_stats.control.eq('random') & method_stats.reject_at_05, 'erasure_percent'].tolist()
        decisions.append(f'- {method}: corrected significant confidence drops at {zero_levels or "no"} percent deletion; '
                         f'corrected significant advantages over random at {random_levels or "no"} percent deletion.')
    class_counts = selected.explicitness.value_counts().to_dict()
    song_rows = frame.drop_duplicates('validation_row')
    changed_targets = int(song_rows.predicted_label.ne(song_rows.full_song_predicted_label).sum())
    diagnostic_text = (f'LIME local-fit R-squared: median {diagnostics.lime_local_fit_r2.median():.4f}, '
        f'range {diagnostics.lime_local_fit_r2.min():.4f} to {diagnostics.lime_local_fit_r2.max():.4f}. '
        f'Maximum absolute SHAP additivity residual: {diagnostics.shap_additivity_residual.abs().max():.3g}. '
        'Additivity does not establish convergence or faithfulness.')
    report = f'''# LIME and SHAP ERASER analysis

Completed run: `{root.name}`. Evaluated {len(selected)} stratified validation songs
with seed {metadata['seed']}. Label allocation: {class_counts}.
All {len(frame)} song/method/deletion records passed consistency checks.

LIME used {metadata['options']['lime_samples']} perturbations with positional
word deletion (`mask_string=''`). SHAP used {metadata['options']['shap_permutations']}
forward/backward permutation passes on positional word-presence features with
an empty-text background. The random control averaged
{metadata['options']['random_repeats']} equal-count deletions per song and level.
The original local predicted class was held fixed for all scores.

## Confidence-drop results

![ERASER confidence drops and above-random effects](faithfulness_comparison.png)

{markdown_table(table)}

Drops are percentage points of the original local predicted-class probability.
Intervals are unadjusted 95% song-bootstrap intervals. P-values are one-sided
paired t tests with Holm correction across all 12 method/level/control tests.
The random control uses equal word counts and is shared between methods per song.

## Hypothesis interpretation

{chr(10).join(decisions)}

Use the zero-control results for the stated hypothesis of a confidence reduction.
Use the random-control results for the stronger claim that highly attributed
words matter more than equally sized random word sets. A nonsignificant
comparison means insufficient evidence, not evidence of no effect.
Report each method and deletion level; do not pool repeated levels as independent songs.

## Exploratory method comparison

{markdown_table(comparison_table)}

Positive differences mean SHAP-guided deletion caused a larger drop. These
paired two-sided t tests use a separate Holm family of three comparisons.
The percentile intervals are descriptive and unadjusted; they use a different
procedure from the t tests and can disagree, especially for skewed paired
differences. Use the corrected p-values for the stated method-comparison decisions.
They describe this experiment and do not establish superiority across models,
explanation budgets, backgrounds, seeds, or datasets.

## Individual-song variation

{markdown_table(rates)}

Counts use strict positive differences, before statistical testing.

## Diagnostics and limits

{diagnostic_text}

Unique scored perturbations truncated at the model context limit:
{metadata.get('overflow_unique_perturbations_truncated', 'unknown')}.
The evidence window reserves 24 BPE tokens, but deletion can change segmentation.
Any remaining truncations are part of the evaluated pipeline and can affect attributions.
The run reused {metadata.get('reused_shap_songs', 0)} verified SHAP explanations;
its runtime truncation count covers newly scored texts only. A separate tokenizer-only
reconstruction of all SHAP permutation coalitions found {int(audit.overflow.sum())}
over-context texts among {int(audit.unique_shap_perturbations.sum())} unique per-song
SHAP perturbations, including reused explanations. This audit reconstructs masks
with the recorded seed and budget; it does not recompute neural predictions.

These results concern the frozen hybrid model's fixed local evidence window,
after boundary trimming and retokenization. They do not establish full-song
faithfulness, visual quality, or fidelity
for every individual song. SHAP uses an empty-text background and five permutation
passes by default; convergence across explanation seeds/budgets was not tested.
LIME and SHAP use different approximation procedures. Both measure sensitivity
to word deletion, which may create unusual inputs. The small validation sample
and paired t-test assumptions limit generalization. Class breakdowns and the
separate three-test SHAP-versus-LIME analysis are exploratory.

The local predicted class differs from the full-song prediction for
{changed_targets} of {len(selected)} songs. The local target is held fixed
through all deletions; this is intentional, but it further limits whole-song claims.

## Supporting files

- `faithfulness_statistics.csv`: primary tests and bootstrap intervals.
- `per_song_success_counts.csv`: number of songs with positive drops and above-random drops.
- `exploratory_shap_vs_lime.csv`: paired method differences, two-sided tests, Holm across three levels.
- `descriptive_by_actual_class.csv`: descriptive class breakdowns.
- `explanation_diagnostics.csv`: LIME fit and SHAP additivity checks.
- `song_*/explanations.html`: both methods' highlighted words; panels scaled independently.
'''
    (root/'ANALYSIS.md').write_text(report, encoding='utf-8')
    print(report)
    print('\nPer-song success counts:\n', rates.to_string(index=False))
    print('\nExploratory method comparisons:\n', pd.DataFrame(comparisons).to_string(index=False))


if __name__ == '__main__':
    main()
