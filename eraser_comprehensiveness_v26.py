"""LIME-guided local evidence-window comprehensiveness for v26.

Run beside lyric_classifier_v26_all_models.py (or pass --classifier).
No training, no test-set access. Uses validation songs and frozen hybrid checkpoint.
Scores the SAME original evidence window and original predicted class after erasure.
This is local-window comprehensiveness, NOT full-song probability faithfulness.
"""
import argparse
import importlib.util
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lime.lime_text import LimeTextExplainer
from tqdm import tqdm


def load_classifier(path):
    spec = importlib.util.spec_from_file_location('v26_eraser_import', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--classifier', type=Path, default=Path(__file__).with_name('lyric_classifier_v26_all_models.py'))
    p.add_argument('--checkpoint', type=Path, default=None)
    p.add_argument('--validation', type=Path, default=None)
    p.add_argument('--output', type=Path, default=Path(__file__).with_name('eraser_results'))
    p.add_argument('--samples', type=int, default=30, help='Stratified validation songs; 0 means all')
    p.add_argument('--lime-samples', type=int, default=300)
    p.add_argument('--random-repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=67)
    args = p.parse_args()
    if args.samples < 0 or args.lime_samples < 50 or args.random_repeats < 1:
        p.error('samples >= 0, lime-samples >= 50, random-repeats >= 1 required')
    clf = load_classifier(args.classifier.resolve())
    clf.set_seed(args.seed)
    checkpoint = args.checkpoint or Path(clf.FROZEN_RUN_DIRS['hybrid']) / 'best_model.pt'
    validation = args.validation or clf.resolve_dataset_path('Validation_Cleaned.csv')
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Missing checkpoint: {checkpoint}')
    df = clf.normalize_dataset_columns(pd.read_csv(validation), 'Validation dataset')
    df['text'] = df['text'].apply(clf.clean_lyrics)
    df['explicitness'] = df['explicitness'].astype(str).str.strip().str.lower().map(clf.LABEL_NORMALIZATION)
    df = df.loc[df['text'].str.len().gt(0) & df['explicitness'].notna()].reset_index(drop=True)
    df['original_row'] = df.index
    if args.samples and args.samples < len(df):
        from sklearn.model_selection import train_test_split
        try:
            df, _ = train_test_split(df, train_size=args.samples, stratify=df['explicitness'], random_state=args.seed)
        except ValueError as e:
            raise ValueError('Choose a larger --samples for stratified sampling, or use --samples 0') from e
        df = df.sort_values('original_row')
    tokenizer = clf.AutoTokenizer.from_pretrained(clf.MODEL_NAME, use_fast=True)
    model = clf.RoBERTaCNNLSTM(num_classes=3).to(clf.DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=clf.DEVICE, weights_only=True))
    model.eval()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # Fixed-window inference: deliberately DOES NOT call song-level severity pooling.
    overflow_perturbations = 0
    def fixed_window_prob(texts):
        nonlocal overflow_perturbations
        outputs = []
        with torch.inference_mode():
            for text in texts:
                ids = clf.full_token_ids(clf.clean_lyrics(text), tokenizer)
                capacity = clf.MAX_LENGTH - sum(map(len, clf.special_token_prefix_suffix(tokenizer)))
                if len(ids) > capacity:
                    # LIME deletion can alter BPE segmentation, increasing token count.
                    # Enforce the model's fixed context instead of crashing.
                    overflow_perturbations += 1
                    ids = ids[:capacity]
                window = clf.make_window_ids(ids, 0, len(ids), tokenizer)
                x, mask = clf.padded_window_batch([window], tokenizer, pad_to=clf.MAX_LENGTH)
                probs = torch.softmax(model(x.to(clf.DEVICE), mask.to(clf.DEVICE)).float(), dim=1)
                outputs.append(probs[0].cpu().numpy())
        return np.asarray(outputs)

    def erase(words, indices):
        return ' '.join(w for i, w in enumerate(words) if i not in indices)

    details = []
    for _, song in tqdm(df.iterrows(), total=len(df), desc='ERASER validation'):
        text = song['text']
        song_class, _, evidence_idx, window_count, _ = clf.predict_song_windows(text, model, tokenizer)
        ids = clf.full_token_ids(text, tokenizer)
        start, end = list(clf.window_spans(ids, tokenizer))[evidence_idx]
        evidence_text = tokenizer.decode(ids[start:end], skip_special_tokens=True).strip()
        # Decoding and re-tokenizing a 510-token window is not guaranteed to
        # preserve its original token count. Reserve room for BPE changes
        # during LIME perturbations, and document any remaining truncations.
        capacity = clf.MAX_LENGTH - sum(map(len, clf.special_token_prefix_suffix(tokenizer)))
        words = evidence_text.split()
        while words and len(clf.full_token_ids(' '.join(words), tokenizer)) > capacity - 24:
            words.pop()
        evidence_text = ' '.join(words)
        if len(words) < 2:
            continue
        # Use original window's predicted class; it can differ from song-level
        # selected class only if evidence selection changes, which is not done here.
        original_probs = fixed_window_prob([evidence_text])[0]
        target = int(original_probs.argmax())
        # Retokenization/word-boundary trimming can change the local class.
        # The local predicted class, not the full-song class, is the target.
        explainer = LimeTextExplainer(class_names=['Safe','Mild','Explicit'], bow=False,
                                     split_expression=r'\s+', random_state=args.seed)
        exp = explainer.explain_instance(evidence_text, fixed_window_prob,
                                         labels=(target,), num_features=len(words),
                                         num_samples=args.lime_samples)
        indexed = exp.domain_mapper.indexed_string
        if indexed.num_words() != len(words) or any(indexed.word(i) != words[i] for i in range(len(words))):
            raise RuntimeError('LIME whitespace token positions did not match erasure token positions')
        weights = dict(exp.local_exp[target])
        # Rank supporting attributions first. Negative weights should NOT be
        # ranked by absolute magnitude as if they supported the predicted class.
        ranking = sorted(range(len(words)), key=lambda i: (-weights.get(i, 0.0), i))
        for percent in (5, 10, 20):
            count = min(len(words)-1, max(1, math.ceil(len(words)*percent/100)))
            removed = set(ranking[:count])
            post = float(fixed_window_prob([erase(words, removed)])[0][target])
            random_scores = []
            for _ in range(args.random_repeats):
                indices = set(rng.sample(range(len(words)), count))
                random_post = float(fixed_window_prob([erase(words, indices)])[0][target])
                random_scores.append(float(original_probs[target]) - random_post)
            details.append(dict(validation_row=int(song['original_row']), song_name=song['song_name'],
                                artist=song['artist'], actual_label=song['explicitness'],
                                predicted_label=clf.ID2LABEL[target], full_song_predicted_label=clf.ID2LABEL[song_class], evidence_window=evidence_idx+1,
                                total_windows=window_count, word_count=len(words), erasure_percent=percent,
                                erased_word_count=count, original_probability=float(original_probs[target]),
                                post_erasure_probability=post,
                                comprehensiveness=float(original_probs[target])-post,
                                random_comprehensiveness_mean=float(np.mean(random_scores)),
                                erased_words=' | '.join(words[i] for i in sorted(removed))))
        pd.DataFrame(details).to_csv(args.output / 'eraser_per_song.csv', index=False)

    if not details:
        raise RuntimeError('No eligible validation songs were evaluated')
    result = pd.DataFrame(details)
    summary = result.groupby('erasure_percent', sort=True).agg(
        songs=('validation_row','count'), original_mean=('original_probability','mean'),
        post_erasure_mean=('post_erasure_probability','mean'),
        comprehensiveness_mean=('comprehensiveness','mean'),
        comprehensiveness_std=('comprehensiveness','std'),
        random_comprehensiveness_mean=('random_comprehensiveness_mean','mean')).reset_index()
    summary['above_random'] = summary.comprehensiveness_mean > summary.random_comprehensiveness_mean
    summary.to_csv(args.output / 'table_4_11.csv', index=False)
    metadata = dict(method='LIME-guided local evidence-window deletion',
                    scope='original v26 evidence window, NOT entire song',
                    predicted_class='fixed original predicted evidence-window class',
                    attribution='LIME positional whitespace-word weights, positive-first',
                    erasure='delete ceil(k% of whitespace-delimited words), preserve order',
                    original_prediction='v26 full-song max-severity selection, then local window scoring',
                    probability='softmax of re-tokenized local evidence text; fixed 510-token context',
                    local_window_trim='trim trailing whitespace words to <= capacity-24 BPE tokens',
                    overflow_perturbations_truncated=overflow_perturbations,
                    baseline='random deletion at equal count, averaged per song',
                    checkpoint=str(checkpoint), validation=str(validation), seed=args.seed,
                    samples_requested=args.samples, songs_evaluated=int(result.validation_row.nunique()),
                    lime_samples=args.lime_samples, random_repeats=args.random_repeats)
    (args.output / 'methodology.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('\nTable 4.11 (local evidence-window comprehensiveness):\n', summary.to_string(index=False))
    print('\nSaved:', args.output.resolve())

if __name__ == '__main__':
    main()
