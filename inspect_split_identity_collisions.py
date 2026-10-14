"""Report title+artist collisions across splits without changing any dataset."""
from pathlib import Path
import csv
import re
import unicodedata
from difflib import SequenceMatcher
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data' if (ROOT / 'data').is_dir() else ROOT
FILES = {'training':'Training.csv', 'validation':'Validation.csv', 'test':'Testing.csv'}
TAG = re.compile(r'\[(?:verse|chorus|pre[- ]?chorus|post[- ]?chorus|bridge|intro|outro|hook|refrain|interlude|instrumental|break|solo|repeat|spoken|sample|drop|ending|part)(?:\s*[:#-]?\s*[^\]\n]*)?\]', re.I)

def identity(x):
    return ''.join(re.findall(r'[^\W_]+', unicodedata.normalize('NFKC', str(x)).casefold(), re.UNICODE))

def lyric_key(x):
    return ' '.join(re.findall(r'[^\W_]+', TAG.sub(' ', unicodedata.normalize('NFKC', str(x)).casefold()), re.UNICODE))

def read_rows(path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        def find(candidates):
            return next((name for name in candidates if name in fields), None)
        title, artist, lyrics = find(['song_title','song_name','track_name']),find(['artist','artists']),find(['lyrics','text'])
        if not all((title,artist,lyrics)):
            raise ValueError(f'{path.name}: missing title/artist/lyrics columns. Found: {fields[:20]}')
        for line, row in enumerate(reader, start=2):
            yield {'line':line, 'title':row[title] or '', 'artist':row[artist] or '', 'lyrics':row[lyrics] or ''}

sets = {}
for split, filename in FILES.items():
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f'{path} not found. Put this script beside your data folder or CSV files.')
    lookup=defaultdict(list)
    for row in read_rows(path):
        lookup[(identity(row['title']),identity(row['artist']))].append(row)
    sets[split]=lookup

out = ROOT / 'split_identity_collisions.csv'
count=0
with out.open('w', encoding='utf-8-sig', newline='') as f:
    writer=csv.writer(f)
    writer.writerow(['split_a','split_b','csv_line_a','csv_line_b','title_a','title_b','artist_a','artist_b','lyrics_exact_after_normalization','lyrics_similarity_rough','lyrics_a_preview','lyrics_b_preview'])
    for a,b in [('training','validation'),('training','test'),('validation','test')]:
        for key in sorted(sets[a].keys() & sets[b].keys()):
            for left in sets[a][key]:
                for right in sets[b][key]:
                    ltext,rtext=lyric_key(left['lyrics']),lyric_key(right['lyrics'])
                    similarity=SequenceMatcher(None,ltext[:4000],rtext[:4000],autojunk=False).ratio()
                    writer.writerow([a,b,left['line'],right['line'],left['title'],right['title'],left['artist'],right['artist'],bool(ltext and ltext==rtext),round(similarity,4),left['lyrics'][:500],right['lyrics'][:500]])
                    count+=1
                    print(f'{a} line {left["line"]} <-> {b} line {right["line"]}: {left["title"]} / {left["artist"]} (similarity {similarity:.3f})')
print(f'\nSaved {count} matching row-pairs to: {out}')
print('Line numbers include CSV header; these are not pandas row indices. Review before removing anything.')
