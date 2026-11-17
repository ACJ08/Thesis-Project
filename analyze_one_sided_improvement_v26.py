"""Separate exploratory upper-tail analysis of saved learning-curve predictions."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'experiments/learning_curve_v26_20261110T111658Z_k392y5iv'
OUTPUT = SOURCE / 'one_sided_improvement_analysis'
RESAMPLES = 10000


def macro_f1(y, pred):
    cm = np.bincount(3 * y + pred, minlength=9).reshape(3, 3)
    denom = cm.sum(0) + cm.sum(1)
    return np.divide(2 * cm.diagonal(), denom, out=np.zeros(3), where=denom != 0).mean()


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def main():
    OUTPUT.mkdir(exist_ok=True)
    records, sources = [], {}
    for size in [1000, 2000, 3000, 4000, 5000, 5988]:
        frames = {}
        for model in ['Hybrid', 'BERT_CNN_LSTM', 'LSTM_CNN', 'RoBERTa']:
            path = SOURCE / f'configurations/seed_67/{model}_{size}/predictions/{model}_{size}_test_predictions.csv'
            sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            frames[model] = pd.read_csv(path, keep_default_na=False)
        reference = frames['Hybrid']
        y = reference.actual_id.to_numpy(dtype=int)
        hybrid = reference.predicted_id.to_numpy(dtype=int)
        for number, model in enumerate(['BERT_CNN_LSTM', 'LSTM_CNN', 'RoBERTa']):
            other_frame = frames[model]
            # Verify song identity, text, labels and ordering, not only class totals.
            identity = ['song_name', 'artist', 'text', 'explicitness', 'actual_id']
            if not reference[identity].equals(other_frame[identity]):
                raise ValueError(f'Unpaired records: {size}, {model}')
            other = other_frame.predicted_id.to_numpy(dtype=int)
            if not all(np.isin(a, [0, 1, 2]).all() for a in [y, hybrid, other]):
                raise ValueError('Unexpected class ID')
            observed = macro_f1(y, hybrid) - macro_f1(y, other)
            rng = np.random.default_rng(np.random.SeedSequence([67, size, number]))
            differences = np.empty(RESAMPLES)
            extreme = 0
            for i in range(RESAMPLES):
                idx = rng.integers(0, len(y), len(y))
                differences[i] = macro_f1(y[idx], hybrid[idx]) - macro_f1(y[idx], other[idx])
                swap = rng.random(len(y)) < .5
                permuted = macro_f1(y, np.where(swap, other, hybrid)) - macro_f1(y, np.where(swap, hybrid, other))
                extreme += permuted >= observed - 1e-12
            records.append(dict(seed=67, size=size, test_songs=len(y), comparison=f'Hybrid_minus_{model}',
                                hybrid_macro_f1=macro_f1(y, hybrid), baseline_macro_f1=macro_f1(y, other),
                                macro_f1_difference=observed, bootstrap_lower95=float(np.percentile(differences, 5)),
                                permutation_p_greater=(1 + extreme) / (RESAMPLES + 1)))
        print(f'Completed training size {size}', flush=True)
    result = pd.DataFrame(records)
    result['holm_p_within_size'] = result.groupby('size').permutation_p_greater.transform(lambda x: holm(x.to_numpy()))
    result['holm_p_all_18'] = holm(result.permutation_p_greater)
    result['improvement_raw_alpha05'] = result.permutation_p_greater < .05
    result['improvement_holm_within_size'] = result.holm_p_within_size < .05
    result['improvement_holm_all_18'] = result.holm_p_all_18 < .05
    result.to_csv(OUTPUT / 'one_sided_results.csv', index=False)
    metadata = dict(hypothesis='Hybrid macro-F1 > baseline macro-F1', alpha=.05,
                    bootstrap_resamples=RESAMPLES, permutation_resamples=RESAMPLES,
                    bootstrap_interval='[5th percentile, +infinity); pointwise, not multiplicity-adjusted',
                    permutation='Song-wise prediction swaps; upper tail; +1 Monte Carlo correction',
                    exploratory=True, scope='seed 67, six training sizes, original saved test predictions; no retraining',
                    limitations='Conditional on fitted models; does not measure retraining variability. Randomization null assumes exchangeable predictions.',
                    source_sha256=sources)
    (OUTPUT / 'methodology.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    lines = ['# One-sided improvement analysis', '',
             'Exploratory reanalysis: H1 is Hybrid macro-F1 > baseline, alpha = 0.05. '
             '10,000 paired song bootstrap resamples and 10,000 paired prediction-swap permutations per comparison. '
             'The 95% one-sided lower bound is the bootstrap 5th percentile. Original two-sided outputs are preserved.', '',
             'All models were checked for identical ordered song names, artists, texts and true labels. '
             'No training or inference was repeated. Results describe seed 67 and sampling uncertainty over test songs, '
             'not variability from retraining. The direction was chosen after earlier results, so interpret as exploratory.', '',
             '| Training size | Baseline | Delta macro-F1 (pp) | 95% lower bound (pp) | One-sided p | Holm p (3 per size) | Holm p (all 18) |',
             '|---:|---|---:|---:|---:|---:|---:|']
    for r in result.itertuples():
        lines.append(f'| {r.size} | {r.comparison.removeprefix("Hybrid_minus_")} | {100*r.macro_f1_difference:.3f} | {100*r.bootstrap_lower95:.3f} | {r.permutation_p_greater:.5f} | {r.holm_p_within_size:.5f} | {r.holm_p_all_18:.5f} |')
    lines += ['', 'Holm corrections are shown both for the three comparisons within each training size and for the full family of 18. Bootstrap bounds are pointwise.']
    (OUTPUT / 'ANALYSIS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(result[result['size'] == 5988].to_string(index=False))


if __name__ == '__main__':
    main()
