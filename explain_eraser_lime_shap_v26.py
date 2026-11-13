"""Explain the same validation songs with LIME/SHAP and test local ERASER faithfulness.

No training or test-set access. SHAP uses positional word-presence features with
an empty-text background; both methods explain the fixed local predicted class.
"""
import argparse
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
from lime.lime_text import LimeTextExplainer
from scipy.stats import ttest_1samp
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from eraser_comprehensiveness_v26 import load_classifier


def statistics(frame, seed, resamples):
    """One-sided paired mean tests; Holm family includes both methods and controls."""
    rng = np.random.default_rng(seed)
    rows = []
    for (method, percent), group in frame.groupby(['method', 'erasure_percent']):
        for control in ('zero', 'random'):
            values = group.comprehensiveness.to_numpy().copy()
            if control == 'random':
                values -= group.random_comprehensiveness_mean.to_numpy()
            if len(values) < 2:
                raise ValueError('At least two songs required for inference')
            means = np.array([rng.choice(values, len(values), replace=True).mean()
                              for _ in range(resamples)])
            if np.all(values == values[0]):
                p = 0.0 if values[0] > 0 else 1.0
            else:
                p = float(ttest_1samp(values, 0, alternative='greater').pvalue)
            low, high = np.quantile(means, [.025, .975])
            rows.append(dict(method=method, erasure_percent=percent, control=control,
                             songs=len(values), mean_difference=values.mean(),
                             bootstrap_ci95_low=low, bootstrap_ci95_high=high,
                             p_one_sided=p))
    order = sorted(range(len(rows)), key=lambda i: rows[i]['p_one_sided'])
    previous = 0.0
    for rank, i in enumerate(order):
        previous = max(previous, min(1., rows[i]['p_one_sided'] * (len(rows)-rank)))
        rows[i]['p_holm'] = previous
        rows[i]['reject_at_05'] = previous < .05
    return pd.DataFrame(rows)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--classifier', type=Path, default=root/'lyric_classifier_v26_all_models.py')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--validation', type=Path, default=root/'data/Validation_Cleaned.csv')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--seed', type=int, default=67)
    parser.add_argument('--lime-samples', type=int, default=1000)
    parser.add_argument('--shap-permutations', type=int, default=5)
    parser.add_argument('--random-repeats', type=int, default=20)
    parser.add_argument('--bootstrap-resamples', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--reuse-shap', type=Path, help='Reuse verified completed SHAP song outputs from a prior run')
    args = parser.parse_args()
    if (args.samples < 2 or args.lime_samples < 50 or args.shap_permutations < 1
            or args.random_repeats < 1 or args.bootstrap_resamples < 100 or args.batch_size < 1):
        parser.error('Require samples>=2, lime-samples>=50, permutations/repeats>=1, bootstrap>=100')
    clf = load_classifier(args.classifier.resolve())
    clf.set_seed(args.seed)
    checkpoint = args.checkpoint or Path(clf.FROZEN_RUN_DIRS['hybrid'])/'best_model.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Checkpoint missing: {checkpoint}. Supply --checkpoint explicitly.')
    df = clf.normalize_dataset_columns(pd.read_csv(args.validation), 'Validation dataset')
    df['validation_row'] = np.arange(len(df))
    df['text'] = df.text.apply(clf.clean_lyrics)
    df['explicitness'] = df.explicitness.astype(str).str.strip().str.lower().map(clf.LABEL_NORMALIZATION)
    df = df.loc[df.text.str.len().gt(0) & df.explicitness.notna()]
    if len(df) < args.samples:
        raise ValueError('Not enough eligible validation songs')
    if len(df) > args.samples:
        df, _ = train_test_split(df, train_size=args.samples,
                                stratify=df.explicitness, random_state=args.seed)
    df = df.sort_values('validation_row')
    reuse_frame = None
    if args.reuse_shap:
        prior = json.loads((args.reuse_shap/'methodology.json').read_text(encoding='utf-8'))
        if (Path(prior['checkpoint']).resolve() != checkpoint.resolve()
                or Path(prior['validation']).resolve() != args.validation.resolve()
                or prior['seed'] != args.seed
                or prior['options']['shap_permutations'] != args.shap_permutations):
            raise ValueError('SHAP source checkpoint, validation, seed, or budget differs')
        old_selected = pd.read_csv(args.reuse_shap/'selected_songs.csv')
        if old_selected.validation_row.tolist() != df.validation_row.tolist():
            raise ValueError('SHAP source selected songs differ')
        reuse_frame = pd.read_csv(args.reuse_shap/'eraser_per_song.csv')
    output = args.output or root/'eraser_lime_shap_results'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    df.drop(columns='text').to_csv(output/'selected_songs.csv', index=False)
    tokenizer = clf.AutoTokenizer.from_pretrained(clf.MODEL_NAME, use_fast=True)
    model = clf.RoBERTaCNNLSTM(num_classes=3).to(clf.DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=clf.DEVICE, weights_only=True))
    model.eval()
    capacity = clf.MAX_LENGTH - sum(map(len, clf.special_token_prefix_suffix(tokenizer)))
    overflow = 0
    cache = {}

    def predict(texts):
        nonlocal overflow
        cleaned = [clf.clean_lyrics(str(text)) for text in texts]
        missing = list(dict.fromkeys(text for text in cleaned if text not in cache))
        with torch.inference_mode():
            for offset in range(0, len(missing), args.batch_size):
                batch = missing[offset:offset+args.batch_size]
                windows = []
                for text in batch:
                    ids = clf.full_token_ids(text, tokenizer)
                    if len(ids) > capacity:
                        overflow += 1
                        ids = ids[:capacity]
                    windows.append(clf.make_window_ids(ids, 0, len(ids), tokenizer))
                x, mask = clf.padded_window_batch(windows, tokenizer, pad_to=clf.MAX_LENGTH)
                probabilities = torch.softmax(model(x.to(clf.DEVICE), mask.to(clf.DEVICE)).float(), 1).cpu().numpy()
                cache.update(zip(batch, probabilities))
        return np.asarray([cache[text] for text in cleaned])

    records = []
    metadata = dict(status='running', checkpoint=str(checkpoint.resolve()),
                    validation=str(args.validation.resolve()), seed=args.seed,
                    scope='fixed local evidence window; not full-song confidence',
                    shap='permutation SHAP; positional word presence; all-zero empty-text background',
                    lime='positional LIME with empty mask string; actual word deletion',
                    ranking='descending signed attribution for original local predicted class',
                    statistics='one-sided paired t tests; 95% song bootstrap intervals; Holm across all 12 tests',
                    options={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    (output/'methodology.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    reused_shap_songs = 0
    for _, song in tqdm(df.iterrows(), total=len(df), desc='LIME + SHAP ERASER'):
        cache.clear()
        row = int(song.validation_row)
        song_class, _, evidence, window_count, _ = clf.predict_song_windows(song.text, model, tokenizer)
        ids = clf.full_token_ids(song.text, tokenizer)
        start, end = list(clf.window_spans(ids, tokenizer))[evidence]
        words = tokenizer.decode(ids[start:end], skip_special_tokens=True).strip().split()
        while words and len(clf.full_token_ids(' '.join(words), tokenizer)) > capacity-24:
            words.pop()
        if len(words) < 2:
            raise ValueError(f'Validation row {row} has fewer than two evidence words')
        text = ' '.join(words)
        original = predict([text])[0]
        target = int(original.argmax())
        song_output = output/f'song_{row:06d}'
        song_output.mkdir()
        (song_output/'evidence.txt').write_text(text, encoding='utf-8')
        lime = LimeTextExplainer(class_names=['Safe', 'Mild', 'Explicit'], bow=False,
                                 split_expression=r'\s+', mask_string='', random_state=args.seed+row)
        explanation = lime.explain_instance(text, predict, labels=(target,),
                                           num_features=len(words), num_samples=args.lime_samples)
        indexed = explanation.domain_mapper.indexed_string
        if indexed.num_words() != len(words) or any(indexed.word(i) != w for i, w in enumerate(words)):
            raise RuntimeError('LIME word positions differ from deletion positions')
        if clf.clean_lyrics(indexed.inverse_removing([0])) != ' '.join(words[1:]):
            raise RuntimeError('LIME perturbation is not word deletion')
        lime_weights = dict(explanation.local_exp[target])
        lime_values = np.array([lime_weights.get(i, 0.) for i in range(len(words))])
        explanation.save_to_file(str(song_output/'lime.html'))

        def presence_predict(masks):
            texts = [' '.join(w for w, keep in zip(words, mask) if keep > .5) for mask in masks]
            return predict(texts)[:, target]

        reused = reuse_frame is not None and row in set(reuse_frame.validation_row)
        if reused:
            source = args.reuse_shap/f'song_{row:06d}'
            old_rows = reuse_frame.loc[reuse_frame.validation_row.eq(row)]
            old_values = pd.read_csv(source/'attributions.csv', keep_default_na=False)
            if ((source/'evidence.txt').read_text(encoding='utf-8') != text
                    or old_values.word.tolist() != words
                    or not old_rows.predicted_label.eq(clf.ID2LABEL[target]).all()
                    or not np.allclose(old_rows.original_probability, original[target], atol=1e-7, rtol=0)):
                raise ValueError(f'SHAP source evidence or target differs for row {row}')
            shap_values = old_values.shap.to_numpy(dtype=float)
            base_value = float(presence_predict(np.zeros((1, len(words))))[0])
            reused_shap_songs += 1
        else:
            explainer = shap.Explainer(presence_predict, np.zeros((1, len(words))),
                                      algorithm='permutation', feature_names=[f'{i}: {w}' for i, w in enumerate(words)],
                                      seed=args.seed+row)
            sv = explainer(np.ones((1, len(words))), max_evals=args.shap_permutations*(2*len(words)+1))
            shap_values = np.asarray(sv.values[0], dtype=float)
            base_value = float(np.asarray(sv.base_values).reshape(-1)[0])
        residual = float(original[target]-base_value-shap_values.sum())
        if abs(residual) > 1e-5:
            raise ValueError(f'SHAP additivity failed for row {row}: {residual}')
        diagnostics = dict(lime_local_fit_r2=float(explanation.score),
                           shap_base_value=base_value, shap_additivity_residual=residual,
                           shap_reused=reused)
        (song_output/'diagnostics.json').write_text(json.dumps(diagnostics, indent=2), encoding='utf-8')
        pd.DataFrame(dict(position=range(len(words)), word=words, lime=lime_values,
                          shap=shap_values)).to_csv(song_output/'attributions.csv', index=False)
        panels = []
        for method, values in [('LIME', lime_values), ('SHAP', shap_values)]:
            scale = max(float(np.abs(values).max()), 1e-12)
            spans = ' '.join(f'<span title="position={i}; attribution={v:.8g}" style="background:rgba({"220,60,60" if v >= 0 else "60,100,220"},{abs(v)/scale:.3f})">{html.escape(w)}</span>'
                             for i, (w, v) in enumerate(zip(words, values)))
            panels.append(f'<h2>{method}</h2><p style="line-height:2">{spans}</p>')
        (song_output/'explanations.html').write_text('<meta charset="utf-8"><h1>Local evidence attributions</h1>'
            + f'<p>Target: {html.escape(str(clf.ID2LABEL[target]))}; confidence: {original[target]:.6f}. Red supports; blue opposes. Each panel scaled separately.</p>'
            + ''.join(panels), encoding='utf-8')
        rng = np.random.default_rng(args.seed+row)
        for percent in (5, 10, 20):
            count = min(len(words)-1, max(1, math.ceil(len(words)*percent/100)))
            def erase(indices):
                removed = set(indices)
                return ' '.join(w for i, w in enumerate(words) if i not in removed)
            random_texts = [erase(rng.choice(len(words), count, replace=False)) for _ in range(args.random_repeats)]
            random_drop = float(np.mean(original[target]-predict(random_texts)[:, target]))
            for method, values in [('LIME', lime_values), ('SHAP', shap_values)]:
                ranking = np.argsort(-values, kind='stable')[:count]
                post = float(predict([erase(ranking)])[0, target])
                records.append(dict(validation_row=row, song_name=song.song_name, artist=song.artist,
                    actual_label=song.explicitness, method=method, erasure_percent=percent,
                    predicted_label=clf.ID2LABEL[target], full_song_predicted_label=clf.ID2LABEL[song_class],
                    evidence_window=evidence+1, total_windows=window_count, word_count=len(words),
                    erased_word_count=count, original_probability=float(original[target]),
                    post_erasure_probability=post, comprehensiveness=float(original[target])-post,
                    random_comprehensiveness_mean=random_drop,
                    erased_positions=json.dumps(sorted(map(int, ranking))),
                    erased_words=' | '.join(words[i] for i in sorted(ranking))))
        pd.DataFrame(records).to_csv(output/'eraser_per_song.csv', index=False)
    frame = pd.DataFrame(records)
    frame.groupby(['method', 'erasure_percent']).agg(songs=('validation_row', 'count'),
        original_mean=('original_probability', 'mean'), post_erasure_mean=('post_erasure_probability', 'mean'),
        comprehensiveness_mean=('comprehensiveness', 'mean'), comprehensiveness_std=('comprehensiveness', 'std'),
        random_comprehensiveness_mean=('random_comprehensiveness_mean', 'mean')).to_csv(output/'summary.csv')
    statistics(frame, args.seed, args.bootstrap_resamples).to_csv(output/'faithfulness_statistics.csv', index=False)
    metadata.update(status='complete', songs_evaluated=int(frame.validation_row.nunique()),
                    overflow_unique_perturbations_truncated=overflow,
                    reused_shap_songs=reused_shap_songs,
                    overflow_scope='newly scored texts only; reused SHAP attribution provenance is the source run')
    (output/'methodology.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Saved results: {output.resolve()}')


if __name__ == '__main__':
    main()
