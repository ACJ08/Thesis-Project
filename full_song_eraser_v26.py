"""LIME/SHAP and matched-random ERASER for complete songs, all four frozen models.

Every perturbation is re-tokenized and re-windowed. The original target class
is fixed, but the deployed max-severity pooler may choose a different evidence
window. Its returned probability vector is a pipeline score, not a calibrated
whole-song probability. Word positions and original whitespace are preserved.
"""
import argparse
import gc
import html
import json
import math
import re
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
from lime.lime_text import LimeTextExplainer

from eraser_comprehensiveness_v26 import load_classifier
from explain_eraser_lime_shap_v26 import statistics


class WordPositions:
    def __init__(self, text):
        self.text = text
        self.words, self.prefixes = [], []
        end = 0
        for match in re.finditer(r'\S+', text):
            self.prefixes.append(text[end:match.start()])
            self.words.append(match.group())
            end = match.end()
        self.suffix = text[end:]

    def render(self, mask):
        return ''.join(prefix + (word if keep else '')
                       for prefix, word, keep in zip(self.prefixes, self.words, mask)) + self.suffix

    def erase(self, indices):
        mask = np.ones(len(self.words), dtype=bool)
        mask[list(indices)] = False
        return self.render(mask)


class FullSongPredictor:
    """Batch windows across texts and reuse identical windows; reproduce v26 pooling."""
    def __init__(self, clf, model, tokenizer, batch_size=16):
        self.clf, self.model, self.tokenizer = clf, model, tokenizer
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self.text_cache = {}
        self.window_cache = OrderedDict()
        self.windows_scored = 0
        self.unique_texts_scored = 0

    def details(self, texts):
        clf = self.clf
        cleaned = [clf.clean_lyrics(str(text)) for text in texts]
        missing = list(dict.fromkeys(text for text in cleaned if text not in self.text_cache))
        # Bound planning memory independently of the explainer's callback batch.
        for offset in range(0, len(missing), 64):
            jobs, planned = [], {}
            for text in missing[offset:offset+64]:
                ids = clf.full_token_ids(text, self.tokenizer)
                spans = list(clf.window_spans(ids, self.tokenizer))
                keys = []
                for start, end in spans:
                    window = clf.make_window_ids(ids, start, end, self.tokenizer)
                    key = np.asarray(window, dtype=np.int32).tobytes()
                    keys.append(key)
                    if key not in self.window_cache:
                        planned[key] = window
                jobs.append((text, keys, len(ids)))
            items = list(planned.items())
            with torch.inference_mode():
                for begin in range(0, len(items), self.batch_size):
                    batch = items[begin:begin+self.batch_size]
                    x, mask = clf.padded_window_batch([w for _, w in batch], self.tokenizer, pad_to=clf.MAX_LENGTH)
                    probs = torch.softmax(self.model(x.to(clf.DEVICE), mask.to(clf.DEVICE)).float(), 1).cpu().numpy()
                    for (key, _), probability in zip(batch, probs):
                        self.window_cache[key] = probability
                    self.windows_scored += len(batch)
            for text, keys, token_count in jobs:
                probabilities = np.asarray([self.window_cache[key] for key in keys])
                classes = probabilities.argmax(axis=1)
                severity = int(classes.max())
                candidates = np.flatnonzero(classes == severity)
                evidence = int(candidates[np.argmax(probabilities[candidates, severity])])
                self.text_cache[text] = (severity, probabilities[evidence].copy(), evidence, len(keys), token_count)
                self.unique_texts_scored += 1
            while len(self.window_cache) > 100000:
                self.window_cache.popitem(last=False)
        return [self.text_cache[text] for text in cleaned]

    def __call__(self, texts):
        return np.asarray([record[1] for record in self.details(texts)])


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def summarize(output, records, seed, resamples):
    frame = pd.DataFrame(records)
    frame.to_csv(output/'eraser_per_song.csv', index=False)
    frame.groupby(['model', 'method', 'erasure_percent']).agg(
        songs=('validation_row', 'count'), original_mean=('original_score', 'mean'),
        post_erasure_mean=('post_erasure_score', 'mean'),
        comprehensiveness_mean=('comprehensiveness', 'mean'),
        random_comprehensiveness_mean=('random_comprehensiveness_mean', 'mean'),
        class_flip_rate=('class_flipped', 'mean'),
        random_class_flip_rate=('random_class_flip_rate', 'mean')).to_csv(output/'summary.csv')
    parts = []
    for model, group in frame.groupby('model'):
        part = statistics(group, seed, resamples)
        part.insert(0, 'model', model)
        part = part.drop(columns=['p_holm', 'reject_at_05'])
        parts.append(part)
    stats = pd.concat(parts, ignore_index=True)
    # One family across every requested model/method/level/control (48 by default).
    previous = 0.
    for rank, i in enumerate(stats.p_one_sided.sort_values().index):
        previous = max(previous, min(1., (len(stats)-rank)*stats.loc[i, 'p_one_sided']))
        stats.loc[i, 'p_holm'] = previous
    stats['reject_at_05'] = stats.p_holm.lt(.05)
    stats.to_csv(output/'faithfulness_statistics.csv', index=False)
    return frame, stats


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, default=root/'eraser_lime_shap_results/20261113T082007192732Z/selected_songs.csv')
    parser.add_argument('--validation', type=Path, default=root/'data/Validation_Cleaned.csv')
    parser.add_argument('--classifier', type=Path, default=root/'lyric_classifier_v26_all_models.py')
    parser.add_argument('--models', nargs='+', choices=['hybrid', 'bert_cnn_lstm', 'lstm_cnn', 'roberta'],
                        default=['hybrid', 'bert_cnn_lstm', 'lstm_cnn', 'roberta'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--lime-samples', type=int, default=1000)
    parser.add_argument('--shap-permutations', type=int, default=5)
    parser.add_argument('--random-repeats', type=int, default=20)
    parser.add_argument('--bootstrap-resamples', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--shap-batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=67)
    args = parser.parse_args()
    if (args.lime_samples < 50 or args.shap_permutations < 1 or args.random_repeats < 1
            or args.bootstrap_resamples < 100 or args.batch_size < 1 or args.shap_batch_size < 1
            or len(set(args.models)) != len(args.models)):
        parser.error('Invalid sampling, batching, or duplicate models')
    clf = load_classifier(args.classifier.resolve())
    clf.set_seed(args.seed)
    selection = pd.read_csv(args.selection)
    if len(selection) != 30 or selection.validation_row.duplicated().any():
        raise ValueError('Expected exactly 30 distinct selected validation row IDs')
    validation = clf.normalize_dataset_columns(pd.read_csv(args.validation), 'Validation dataset')
    songs = validation.iloc[selection.validation_row.to_numpy()].copy()
    songs['validation_row'] = selection.validation_row.to_numpy()
    songs['text'] = songs.text.apply(clf.clean_lyrics)
    songs['explicitness'] = songs.explicitness.astype(str).str.strip().str.lower().map(clf.LABEL_NORMALIZATION)
    for field in ('song_name', 'artist', 'explicitness'):
        if songs[field].tolist() != selection[field].tolist():
            raise ValueError(f'Selected song identity mismatch: {field}')
    manifests = {}
    for name in args.models:
        directory = Path(clf.FROZEN_RUN_DIRS[name])
        manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        if (manifest.get('architecture') != name or manifest.get('pipeline_version') != clf.PIPELINE_VERSION
                or manifest.get('smoke_test') or any(manifest.get(k) != v for k, v in clf.window_policy().items())):
            raise ValueError(f'Frozen checkpoint architecture/window policy mismatch: {name}')
        if manifest.get('checkpoint_sha256') != clf.file_sha256(directory/'best_model.pt'):
            raise ValueError(f'Checkpoint integrity mismatch: {name}')
        if name == 'lstm_cnn' and manifest.get('vocab_sha256') != clf.file_sha256(directory/'vocabulary.json'):
            raise ValueError('LSTM vocabulary integrity mismatch')
        manifests[name] = manifest
    config = dict(models=args.models, seed=args.seed, lime_samples=args.lime_samples,
                  shap_permutations=args.shap_permutations, random_repeats=args.random_repeats,
                  bootstrap_resamples=args.bootstrap_resamples, validation_sha256=clf.file_sha256(args.validation),
                  selection_sha256=clf.file_sha256(args.selection), classifier_sha256=clf.file_sha256(args.classifier),
                  checkpoint_sha256={k: v['checkpoint_sha256'] for k, v in manifests.items()})
    output = args.resume or args.output or root/'full_song_eraser_results'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if args.resume:
        metadata = json.loads((output/'methodology.json').read_text(encoding='utf-8'))
        if metadata['config'] != config:
            raise ValueError('Resume configuration or input hashes changed')
    else:
        output.mkdir(parents=True, exist_ok=False)
        selection.to_csv(output/'selected_songs.csv', index=False)
        metadata = dict(status='running', config=config, scope='entire song; all overlapping windows recomputed',
                        class_policy='original target fixed; max-severity pooling and evidence selection rerun after every perturbation',
                        score='fixed target probability in dynamically selected evidence window; pipeline score, not calibrated whole-song probability',
                        perturbation='positional whitespace-word deletion; preserve original separators; no truncation',
                        shap='permutation SHAP; binary positional word presence; empty-text background',
                        inference='window batching/cache; original predictions verified against repository predictor',
                        frozen_runs={k: clf.FROZEN_RUN_DIRS[k] for k in args.models})
    metadata['status'] = 'running'
    save_json(output/'methodology.json', metadata)
    all_records, original_records = [], []
    for name in args.models:
        directory = Path(clf.FROZEN_RUN_DIRS[name])
        model_output = output/name
        model_output.mkdir(exist_ok=True)
        tokenizer = clf.tokenizer_for(name, None, directory)
        model = clf.make_baseline(name, tokenizer if name == 'lstm_cnn' else None).to(clf.DEVICE)
        model.load_state_dict(torch.load(directory/'best_model.pt', map_location=clf.DEVICE, weights_only=True))
        model.eval()
        predictor = FullSongPredictor(clf, model, tokenizer, args.batch_size)
        for _, song in songs.iterrows():
            row = int(song.validation_row)
            folder = model_output/f'song_{row:06d}'
            completed = folder/'completed.json'
            if completed.is_file():
                saved = json.loads(completed.read_text(encoding='utf-8'))
                all_records.extend(saved['records'])
                original_records.append(saved['original'])
                print(f'{name} row={row}: resumed completed song', flush=True)
                continue
            folder.mkdir(exist_ok=True)
            predictor.reset()
            positions = WordPositions(song.text)
            if len(positions.words) < 2 or positions.render(np.ones(len(positions.words))) != song.text:
                raise ValueError(f'Word alignment/length error: {row}')
            target, probabilities, evidence, windows, token_count = predictor.details([song.text])[0]
            reference = clf.predict_song_windows(song.text, model, tokenizer)
            if target != reference[0] or not np.allclose(probabilities, reference[4], atol=2e-5, rtol=0):
                raise ValueError(f'Batched pooling failed reference check: {name} row {row}')
            original = dict(model=name, validation_row=row, song_name=song.song_name, artist=song.artist,
                            actual_label=song.explicitness, predicted_label=clf.ID2LABEL[target],
                            score=float(probabilities[target]), evidence_window=evidence+1,
                            total_windows=windows, model_token_count=token_count, word_count=len(positions.words),
                            p_safe=float(probabilities[0]), p_mild=float(probabilities[1]), p_explicit=float(probabilities[2]))
            (folder/'song.txt').write_text(song.text, encoding='utf-8')
            print(f'{name} row={row}: {len(positions.words)} words, {windows} full-song windows; LIME', flush=True)
            lime = LimeTextExplainer(class_names=['Safe', 'Mild', 'Explicit'], bow=False,
                                     split_expression=r'\s+', mask_string='', random_state=args.seed+row)
            exp = lime.explain_instance(song.text, predictor, labels=(target,),
                                       num_features=len(positions.words), num_samples=args.lime_samples)
            indexed = exp.domain_mapper.indexed_string
            if (indexed.num_words() != len(positions.words)
                    or any(indexed.word(i) != word for i, word in enumerate(positions.words))
                    or clf.clean_lyrics(indexed.inverse_removing([0])) != clf.clean_lyrics(positions.erase([0]))):
                raise ValueError('LIME perturbation positions/deletion mismatch')
            lime_weights = dict(exp.local_exp[target])
            lime_values = np.array([lime_weights.get(i, 0.) for i in range(len(positions.words))])
            exp.save_to_file(str(folder/'lime.html'))

            def mask_scores(masks):
                return predictor([positions.render(mask > .5) for mask in masks])[:, target]

            print(f'{name} row={row}: SHAP', flush=True)
            explainer = shap.Explainer(mask_scores, np.zeros((1, len(positions.words))), algorithm='permutation', seed=args.seed+row)
            sv = explainer(np.ones((1, len(positions.words))), max_evals=args.shap_permutations*(2*len(positions.words)+1),
                           batch_size=args.shap_batch_size)
            shap_values = np.asarray(sv.values[0], dtype=float)
            base = float(np.asarray(sv.base_values).reshape(-1)[0])
            residual = float(probabilities[target]-base-shap_values.sum())
            if abs(residual) > 1e-5:
                raise ValueError('SHAP additivity check failed')
            pd.DataFrame(dict(position=range(len(positions.words)), word=positions.words,
                              lime=lime_values, shap=shap_values)).to_csv(folder/'attributions.csv', index=False)
            panels = []
            for method, values in [('LIME', lime_values), ('SHAP', shap_values)]:
                scale = max(float(np.abs(values).max()), 1e-12)
                content = ''.join(html.escape(prefix)+f'<span title="position={i}; attribution={value:.8g}" style="background:rgba({"220,60,60" if value >= 0 else "60,100,220"},{abs(value)/scale:.3f})">{html.escape(word)}</span>'
                                  for i, (prefix, word, value) in enumerate(zip(positions.prefixes, positions.words, values)))
                panels.append(f'<h2>{method}</h2><div style="white-space:pre-wrap;line-height:2">{content}</div>')
            (folder/'explanations.html').write_text('<meta charset="utf-8"><h1>Complete-song pipeline attributions</h1>'
                + f'<p>{html.escape(name)}; original target {html.escape(clf.ID2LABEL[target])}; score {probabilities[target]:.6f}. Red supports, blue opposes; scales are separate.</p>'
                + ''.join(panels), encoding='utf-8')
            rng = np.random.default_rng(args.seed+row)
            records = []
            for percent in (5, 10, 20):
                count = min(len(positions.words)-1, max(1, math.ceil(len(positions.words)*percent/100)))
                random_texts = [positions.erase(rng.choice(len(positions.words), count, replace=False))
                                for _ in range(args.random_repeats)]
                random_details = predictor.details(random_texts)
                random_drop = float(np.mean([probabilities[target]-detail[1][target] for detail in random_details]))
                random_flip = float(np.mean([detail[0] != target for detail in random_details]))
                for method, values in [('LIME', lime_values), ('SHAP', shap_values)]:
                    ranking = np.argsort(-values, kind='stable')[:count]
                    post_class, post_probabilities, post_evidence, post_windows, _ = predictor.details([positions.erase(ranking)])[0]
                    records.append(dict(model=name, method=method, validation_row=row, song_name=song.song_name,
                        artist=song.artist, actual_label=song.explicitness, target_label=clf.ID2LABEL[target],
                        erasure_percent=percent, word_count=len(positions.words), erased_word_count=count,
                        original_score=float(probabilities[target]), post_erasure_score=float(post_probabilities[target]),
                        comprehensiveness=float(probabilities[target]-post_probabilities[target]),
                        random_comprehensiveness_mean=random_drop, class_flipped=post_class != target,
                        post_erasure_label=clf.ID2LABEL[post_class], random_class_flip_rate=random_flip,
                        original_windows=windows, post_erasure_windows=post_windows,
                        original_evidence_window=evidence+1, post_erasure_evidence_window=post_evidence+1,
                        erased_positions=json.dumps(sorted(map(int, ranking))),
                        erased_words=' | '.join(positions.words[i] for i in sorted(ranking))))
            saved = dict(original=original, records=records, lime_local_fit_r2=float(exp.score),
                         shap_base_value=base, shap_additivity_residual=residual,
                         unique_texts_scored=predictor.unique_texts_scored, windows_scored=predictor.windows_scored)
            save_json(completed, saved)
            all_records.extend(records)
            original_records.append(original)
            pd.DataFrame(all_records).to_csv(output/'eraser_per_song.csv', index=False)
            pd.DataFrame(original_records).to_csv(output/'original_predictions.csv', index=False)
            print(f'{name} row={row}: complete; {len(original_records)}/{30*len(args.models)} model-song conditions', flush=True)
        del predictor, model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    frame, stats = summarize(output, all_records, args.seed, args.bootstrap_resamples)
    if len(frame) != len(args.models)*30*6 or frame.duplicated(['model', 'validation_row', 'method', 'erasure_percent']).any():
        raise ValueError('Incomplete or duplicated results')
    metadata.update(status='complete', model_song_conditions=len(original_records), records=len(frame),
                    tests=len(stats), multiple_testing='Holm across all requested models/methods/levels/controls')
    save_json(output/'methodology.json', metadata)
    print(f'COMPLETE: {output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
