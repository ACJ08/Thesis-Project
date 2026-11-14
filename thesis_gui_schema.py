"""Versioned request schema and lightweight preflight (no ML imports)."""
import csv
import hashlib
import json
from pathlib import Path

SCHEMA_VERSION = 1
MODEL_IDS = ('hybrid', 'bert_cnn_lstm', 'lstm_cnn', 'roberta')
DATASETS = ('Training_Cleaned.csv', 'Validation_Cleaned.csv', 'Testing_Cleaned.csv')
ALIASES = {'song_name': ('song_title', 'song_name', 'track_name'), 'artist': ('artist', 'artists'),
           'text': ('lyrics', 'text'), 'explicitness': ('safety_label', 'explicitness', 'predicted_label', 'Explicit', 'explicit')}
REQUEST_SCHEMA = {
    'version': SCHEMA_VERSION,
    'actions': {
        'train': {'mode': (*MODEL_IDS, 'all_baselines', 'all_models', 'compare'), 'smoke': bool, 'datasets': list, 'runs': dict},
        'label': {'model': MODEL_IDS, 'run': str, 'source': str, 'output': str, 'lyrics': str, 'column': str},
        'explain': {'model': MODEL_IDS, 'run': str, 'method': ('LIME', 'SHAP', 'Both'), 'lyrics': str, 'validation': str, 'samples': int},
    },
}


def read_json(path, default=None):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else ({} if default is None else default)
    except (OSError, ValueError):
        return {} if default is None else default


def default_dataset(root, filename):
    for folder in (root / 'data', root.parent / 'data', root, root.parent):
        path = folder / filename
        if path.is_file():
            return path
    raise ValueError(f'Dataset not found: {filename}. Select a CSV explicitly.')


def csv_schema(path, *, labeled=False, column=''):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'CSV not found: {path}')
    try:
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            headers = reader.fieldnames or []
            if not headers or len(set(headers)) != len(headers):
                raise ValueError(f'CSV needs unique column headers: {path.name}')
            if labeled:
                missing = [f'{key} ({" / ".join(names)})' for key, names in ALIASES.items() if not any(n in headers for n in names)]
                if missing:
                    raise ValueError(f'{path.name}: missing columns: {", ".join(missing)}')
            else:
                if (column and column not in headers) or (not column and not any(n in headers for n in ('lyrics', 'text'))):
                    raise ValueError(f'{path.name}: select an existing lyrics column. Found: {", ".join(headers)}')
                if any(n in headers for n in ('v26_label', 'v26_evidence_probability', 'v26_evidence_window', 'v26_window_count')):
                    raise ValueError('Input already contains v26 prediction columns.')
            if next(reader, None) is None:
                raise ValueError(f'CSV contains no data rows: {path.name}')
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError(f'Cannot read CSV {path}: {error}') from error
    return path.resolve()


def checkpoint(path, model, *, frozen=False):
    directory = Path(path)
    if not (directory / 'best_model.pt').is_file():
        raise ValueError(f'{model}: select a folder containing best_model.pt.')
    manifest_path = directory / 'manifest.json'
    meta = read_json(manifest_path)
    if manifest_path.exists() and not meta:
        raise ValueError(f'Invalid manifest: {manifest_path}')
    if meta and (meta.get('architecture') != model or meta.get('pipeline_version') != 'v26'):
        raise ValueError(f'{directory.name}: checkpoint must be v26 {model}.')
    if frozen and (not meta or meta.get('smoke_test') is not False):
        raise ValueError(f'{model}: frozen comparison needs a full run with a manifest.')
    if model == 'lstm_cnn' and not (directory / 'vocabulary.json').is_file():
        raise ValueError('LSTM–CNN checkpoint needs vocabulary.json.')
    for filename, field in (('best_model.pt', 'checkpoint_sha256'), ('vocabulary.json', 'vocab_sha256')):
        if meta.get(field) and sha256(directory / filename) != meta[field]:
            raise ValueError(f'Integrity check failed: {directory / filename}')
    return meta


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_job(job, root):
    """Validate types, cross-field rules and assets; return a normalized copy."""
    job = dict(job)
    if job.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Unsupported request schema version.')
    fields = REQUEST_SCHEMA['actions'].get(job.get('action'))
    if fields is None:
        raise ValueError('Unknown operation.')
    for name, rule in fields.items():
        value = job.get(name)
        valid = value in rule if isinstance(rule, tuple) else type(value) is rule
        if not valid:
            raise ValueError(f'Invalid or missing {name}.')
    root = Path(root)
    if job['action'] == 'train':
        if len(job['datasets']) != 3 or any(not isinstance(v, str) for v in job['datasets']):
            raise ValueError('Provide training, validation and test CSV paths.')
        paths = [csv_schema(value or default_dataset(root, filename), labeled=True)
                 for value, filename in zip(job['datasets'], DATASETS)]
        if len(set(paths)) != 3:
            raise ValueError('Training, validation and test must use separate CSV files.')
        job['datasets'] = list(map(str, paths))
        if job['mode'] == 'compare':
            if job['smoke']:
                raise ValueError('Frozen comparison cannot use smoke mode.')
            hashes = dict(zip(('training', 'validation', 'test'), map(sha256, paths)))
            for model in MODEL_IDS:
                value = job['runs'].get(model)
                if not isinstance(value, str) or not value:
                    raise ValueError(f'Select a frozen checkpoint for {model}.')
                meta = checkpoint(value, model, frozen=True)
                if meta.get('input_sha256') != hashes:
                    raise ValueError(f'{model}: selected datasets differ from training dataset hashes.')
    else:
        checkpoint(job['run'], job['model'])
        if job['action'] == 'label':
            if job['source']:
                source = csv_schema(job['source'], column=job['column'])
                output = Path(job['output'])
                if not job['output'] or not output.parent.is_dir() or output.is_dir():
                    raise ValueError('Choose an output CSV in an existing folder.')
                if output.suffix.lower() != '.csv' or source == output.resolve():
                    raise ValueError('Output must be a separate .csv file.')
            elif not job['lyrics'].strip():
                raise ValueError('Paste song lyrics.')
        elif job['samples'] < 0:
            raise ValueError('Sample size must be zero or a positive integer.')
        elif job['validation']:
            job['validation'] = str(csv_schema(default_dataset(root, DATASETS[1]) if job['validation'] == 'default' else job['validation'], labeled=True))
        elif not job['lyrics'].strip():
            raise ValueError('Paste lyrics for a single-song explanation.')
    return job
