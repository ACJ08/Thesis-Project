"""Safely remove the confirmed Allman Brothers Band overlap from Testing.csv.

Run from anywhere: python remove_validation_test_overlap.py
Place script beside the data/ directory. Requires pandas.
"""
from pathlib import Path
from datetime import datetime
import shutil
import re
import unicodedata
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
VALIDATION = DATA / 'Validation.csv'
TESTING = DATA / 'Testing.csv'


def normalized(value):
    value = unicodedata.normalize('NFKC', str(value)).casefold()
    return ''.join(ch for ch in value if ch.isalnum())


def lyric_normalized(value):
    value = unicodedata.normalize('NFKC', str(value)).casefold()
    tag = re.compile(r'\[(?:verse|chorus|pre[- ]?chorus|post[- ]?chorus|bridge|intro|outro|hook|refrain|interlude|instrumental|break|solo|repeat|spoken|sample|drop|ending|part)(?:\s*[:#-]?\s*[^\]\n]*)?\]', re.I)
    value = tag.sub(' ', value)
    return ' '.join(re.findall(r'[^\W_]+', value, re.UNICODE))


def pick_column(df, options, name):
    matches = [c for c in df.columns if c.casefold().strip() in options]
    if len(matches) != 1:
        raise ValueError(f'{name}: expected one column from {options}; found {matches}')
    return matches[0]


def main():
    for path in (VALIDATION, TESTING):
        if not path.is_file():
            raise FileNotFoundError(path)
    val = pd.read_csv(VALIDATION, dtype=str, keep_default_na=False, low_memory=False)
    test = pd.read_csv(TESTING, dtype=str, keep_default_na=False, low_memory=False)
    def cols(df):
        return (pick_column(df, {'song', 'title', 'song title', 'song_title', 'song_name', 'track_name'}, 'song'),
                pick_column(df, {'artist(s)', 'artist', 'artists', 'artist_name'}, 'artist'),
                pick_column(df, {'text', 'lyrics', 'lyric', 'song_lyrics'}, 'lyrics'))
    vt, va, vl = cols(val)
    tt, ta, tl = cols(test)
    print(f'Validation columns: title={vt!r}, artist={va!r}, lyrics={vl!r}')
    print(f'Testing columns: title={tt!r}, artist={ta!r}, lyrics={tl!r}')
    target_title = normalized("You Don't Love Me")
    target_artist = normalized('Allman Brothers Band')
    val_hits = val.index[(val[vt].map(normalized) == target_title) &
                         (val[va].map(normalized) == target_artist)].tolist()
    test_hits = test.index[(test[tt].map(normalized) == target_title) &
                           (test[ta].map(normalized) == target_artist)].tolist()
    if len(val_hits) != 1 or len(test_hits) != 1:
        raise RuntimeError(f'Expected exactly one matching row per split; validation={len(val_hits)}, testing={len(test_hits)}. No files changed.')
    vi, ti = val_hits[0], test_hits[0]
    if lyric_normalized(val.at[vi, vl]) != lyric_normalized(test.at[ti, tl]):
        raise RuntimeError('Lyric text differs between splits; refusing to delete. No files changed.')
    if not lyric_normalized(test.at[ti, tl]):
        raise RuntimeError('Lyrics empty after normalization; refusing to delete. No files changed.')
    print(f'Confirmed identical song: validation line {vi + 2}; testing line {ti + 2}')
    backup = TESTING.with_name(f'{TESTING.stem}.backup_{datetime.now():%Y%m%d_%H%M%S}{TESTING.suffix}')
    if backup.exists():
        raise FileExistsError(backup)
    shutil.copy2(TESTING, backup)
    test.drop(index=ti).to_csv(TESTING, index=False, encoding='utf-8-sig')
    print(f'Backup: {backup}')
    print(f'Removed 1 row from {TESTING}; remaining test rows: {len(test)-1}')
    print('Rerun your full split-overlap audit before training.')

if __name__ == '__main__':
    main()
