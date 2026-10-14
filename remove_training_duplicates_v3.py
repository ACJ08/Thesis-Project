"""Lyrics-first duplicate audit, including covers by different artists.

pip install pandas rapidfuzz
python remove_training_duplicates_v2.py --training Training.csv --validation Validation.csv --testing Testing.csv

Exact normalized duplicates are removed automatically. Near matches are reviewed by
 default; pass --remove-near to remove high-confidence near matches too.
Gold-standard Validation and Testing files are never modified.
"""
import argparse
from collections import defaultdict, Counter
from pathlib import Path
import re
import unicodedata
import pandas as pd
from rapidfuzz.fuzz import ratio


def normalize(value):
    if pd.isna(value):
        return ''
    s = unicodedata.normalize('NFKC', str(value)).casefold()
    s = re.sub(r'[^\w\s]', ' ', s)
    return ' '.join(s.split())


def shingles(s, n=5):
    words = s.split()
    if len(words) < n:
        return set()
    return set(zip(*(words[i:] for i in range(n))))


def meta(source, idx, row):
    return dict(file=source, csv_row=idx + 2, title=str(row['song_title']), artist=str(row['artist']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--training', default='Training.csv')
    ap.add_argument('--validation', default='Validation.csv')
    ap.add_argument('--testing', default='Testing.csv')
    ap.add_argument('--out', default='Training_deduplicated.csv')
    ap.add_argument('--report', default='training_duplicate_removals.csv')
    ap.add_argument('--near-report', default='training_near_duplicate_review.csv')
    ap.add_argument('--near-threshold', type=float, default=0.94, help='Minimum token-shingle Jaccard or containment to flag')
    ap.add_argument('--remove-near', action='store_true', help='Remove high-confidence near matches (off by default)')
    ap.add_argument('--remove-all-near', action='store_true', help='Remove all flagged near matches from Training (aggressive)')
    ap.add_argument('--auto-threshold', type=float, default=0.985, help='Both token-shingle Jaccard and text ratio needed for auto removal')
    ap.add_argument('--min-words', type=int, default=20, help='Do not auto-remove short lyrics based on fuzzy matches')
    ap.add_argument('--max-postings', type=int, default=300, help='Ignore very common lyric shingles in candidate discovery')
    ap.add_argument('--max-candidates', type=int, default=250, help='Maximum fuzzy candidates scored per song')
    args = ap.parse_args()
    if not 0 < args.near_threshold <= 1 or not 0 < args.auto_threshold <= 1:
        ap.error('thresholds must be in (0, 1]')

    data = [(label, pd.read_csv(path, low_memory=False)) for label, path in
            [('Training', args.training), ('Validation', args.validation), ('Testing', args.testing)]]
    for label, df in data:
        required = {'lyrics', 'song_title', 'artist'}
        if not required.issubset(df.columns):
            raise ValueError(f'{label} missing {sorted(required - set(df.columns))}')

    entries = []
    for label, df in data:
        for idx, row in df.iterrows():
            lyric = normalize(row['lyrics'])
            if lyric:
                entries.append({'source': label, 'idx': idx, 'text': lyric,
                                'meta': meta(label, idx, row), 'words': len(lyric.split())})

    # Index gold-standard rows first. Within training, the first retained row wins.
    entries.sort(key=lambda e: (e['source'] == 'Training', e['idx']))
    exact = {}
    postings = defaultdict(list)
    reference = []
    removed = set()
    removals = []
    review = []

    def record(e, other, kind, jac=1.0, contain=1.0, text_score=1.0):
        return dict(training_csv_row=e['idx'] + 2,
                    training_title=e['meta']['title'], training_artist=e['meta']['artist'],
                    matching_file=other['meta']['file'], matching_csv_row=other['meta']['csv_row'],
                    matching_title=other['meta']['title'], matching_artist=other['meta']['artist'],
                    match_type=kind, shingle_jaccard=round(jac, 5),
                    shingle_containment=round(contain, 5), text_similarity=round(text_score, 5))

    for count, e in enumerate(entries, 1):
        review_start = len(review)
        text = e['text']
        is_training = e['source'] == 'Training'
        if is_training and text in exact:
            other = reference[exact[text]]
            removed.add(e['idx'])
            removals.append(record(e, other, 'exact_normalized_lyrics'))
            continue

        # Cross-artist/title discovery using 5-word sequences rather than metadata.
        grams = shingles(text)
        candidate_counts = Counter()
        for gram in grams:
            hits = postings.get(gram, ())
            if len(hits) <= args.max_postings:
                candidate_counts.update(hits)
        best = None
        if is_training and grams:
            for rid, overlap in candidate_counts.most_common(args.max_candidates):
                other = reference[rid]
                other_grams = other['grams']
                union = len(grams) + len(other_grams) - overlap
                jac = overlap / union if union else 0
                containment = overlap / min(len(grams), len(other_grams))
                if max(jac, containment) < args.near_threshold:
                    continue
                score = ratio(text, other['text']) / 100
                # High containment can indicate a cover with an omitted verse.
                # Such pairs are always for review, not auto-deletion.
                row = record(e, other, 'near_lyrics', jac, containment, score)
                review.append(row)
                if (args.remove_near and min(e['words'], other['words']) >= args.min_words
                        and jac >= args.auto_threshold and score >= args.auto_threshold):
                    if best is None or (jac, score) > best[0]:
                        best = ((jac, score), row)
        if is_training and args.remove_all_near and best is None:
            flagged = review[review_start:]
            if flagged:
                strongest = max(flagged, key=lambda r: (r['shingle_jaccard'], r['shingle_containment'], r['text_similarity']))
                best = ((strongest['shingle_jaccard'], strongest['text_similarity']), strongest)
        if best:
            removed.add(e['idx'])
            r = dict(best[1]); r['match_type'] = 'all_flagged_near_lyrics' if args.remove_all_near else 'high_confidence_near_lyrics'
            removals.append(r)
            continue

        # Gold rows and retained training rows become references.
        rid = len(reference)
        e['grams'] = grams
        reference.append(e)
        exact.setdefault(text, rid)
        for gram in grams:
            postings[gram].append(rid)
        if count % 10000 == 0:
            print(f'Processed {count:,}/{len(entries):,} lyric records...', flush=True)

    train = data[0][1]
    train.drop(index=list(removed)).to_csv(args.out, index=False, encoding='utf-8-sig')
    cols = ['training_csv_row', 'training_title', 'training_artist', 'matching_file',
            'matching_csv_row', 'matching_title', 'matching_artist', 'match_type',
            'shingle_jaccard', 'shingle_containment', 'text_similarity']
    pd.DataFrame(removals, columns=cols).to_csv(args.report, index=False, encoding='utf-8-sig')
    pd.DataFrame(review, columns=cols).to_csv(args.near_report, index=False, encoding='utf-8-sig')
    print(f'Training: {len(train):,} | Removed: {len(removed):,} | Retained: {len(train)-len(removed):,}')
    print(f'Near-match candidate pairs for review: {len(review):,}')
    print(f'Outputs: {args.out}, {args.report}, {args.near_report}')
    print('Row numbers are spreadsheet CSV row numbers, including header.')
    print('NOTE: Approximate candidate discovery is not exhaustive; manually review flagged pairs.')


if __name__ == '__main__':
    main()
