"""Leakage-safe preprocessing for the instructor's reviews and joke CSV."""

import csv
import hashlib
import html
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

SPECIALS = ['<pad>', '<unk>', '<bos>', '<eos>']
TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*|[^\w\s]", re.ASCII)


def tokenize(text):
    text = re.sub(r'<[^>]+>', ' ', html.unescape(text))
    return TOKEN.findall(text.lower().replace('\u2019', "'"))


def detokenize(tokens):
    text = ' '.join(tokens)
    text = re.sub(r'\s+([.,!?;:%])', r'\1', text)
    text = re.sub(r'([!?.,])(?:\s*\1)+', r'\1', text)
    return text[:1].upper() + text[1:]


def vocabulary(documents, limit=20000, min_frequency=1):
    counts = Counter(t for doc in documents for t in doc)
    words = sorted(counts, key=lambda t: (-counts[t], t))
    return SPECIALS + [t for t in words if counts[t] >= min_frequency
                       and t not in SPECIALS][:limit - len(SPECIALS)]


def encode(tokens, word_to_id, max_length=None):
    # Preserve both introductory context and conclusions of long reviews.
    if max_length is not None and len(tokens) > max_length:
        head = max_length // 2
        tokens = tokens[:head] + tokens[-(max_length - head):]
    return [word_to_id.get(t, 1) for t in tokens] or [1]


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels=None):
        self.sequences, self.labels = sequences, labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i):
        return self.sequences[i], None if self.labels is None else self.labels[i]


def classification_data(root, *, seed=42, vocab_size=20000, max_length=300):
    root = Path(root) / 'text_classification'
    records = {}
    for split in ['train', 'test']:
        records[split] = []
        for label, name in enumerate(['neg', 'pos']):
            files = sorted((root / split / name).glob('*.txt'))
            if not files:
                raise FileNotFoundError(f'No reviews in {root / split / name}')
            for p in files:
                tokens = tokenize(p.read_text(encoding='utf-8'))
                records[split].append((str(p.relative_to(root)), label, tokens))
    labels = [r[1] for r in records['train']]
    train_i, val_i = train_test_split(np.arange(len(labels)), test_size=0.1,
                                    random_state=seed, stratify=labels)
    subsets = {'train': [records['train'][i] for i in train_i],
               'validation': [records['train'][i] for i in val_i],
               'test': records['test']}
    vocab = vocabulary([r[2] for r in subsets['train']], vocab_size, 2)
    mapping = {w: i for i, w in enumerate(vocab)}
    data = {s: SequenceDataset([encode(r[2], mapping, max_length) for r in rows],
                              [r[1] for r in rows]) for s, rows in subsets.items()}
    meta = {'task': 'classification', 'vocabulary_size': len(vocab),
            'seed': seed, 'max_length': max_length, 'truncation': 'head and tail',
            'splits': {s: len(rows) for s, rows in subsets.items()},
            'class_counts': {s: dict(Counter(r[1] for r in rows)) for s, rows in subsets.items()},
            'unknown_token_rates': {s: sum(t not in mapping for r in rows for t in r[2]) /
                                    sum(len(r[2]) for r in rows) for s, rows in subsets.items()},
            'median_training_length': float(np.median([len(r[2]) for r in subsets['train']])),
            'truncated_training_fraction': float(np.mean([len(r[2]) > max_length for r in subsets['train']])),
            'examples': {s: ' '.join(rows[0][2][:60]) for s, rows in subsets.items()}}
    return data, vocab, meta, {s: [r[0] for r in rows] for s, rows in subsets.items()}


def generation_data(root, *, seed=42, max_length=96):
    path = Path(root) / 'text_generation' / 'data'
    with path.open(encoding='utf-8', newline='') as f:
        raw = list(csv.DictReader(f))
    rows, seen = [], set()
    for row in raw:
        tokens = tokenize(row['Joke'])
        key = tuple(tokens)
        if key not in seen and len(tokens) >= 4:
            seen.add(key)
            rows.append({'id': row['ID'], 'text': row['Joke'], 'tokens': tokens})
    train_i, val_i = train_test_split(np.arange(len(rows)), test_size=0.1, random_state=seed)
    subsets = {'train': [rows[i] for i in train_i], 'validation': [rows[i] for i in val_i]}
    vocab = vocabulary([r['tokens'] for r in subsets['train']], limit=20000)
    mapping = {w: i for i, w in enumerate(vocab)}
    # Truncate at the end for causal language modeling, preserving the opening.
    data = {s: SequenceDataset([[2] + encode(r['tokens'][:max_length], mapping) + [3]
                               for r in part]) for s, part in subsets.items()}
    meta = {'task': 'generation', 'source_rows': len(raw), 'unique_usable_rows': len(rows),
            'removed_duplicate_or_short_rows': len(raw) - len(rows),
            'splits': {s: len(part) for s, part in subsets.items()}, 'seed': seed,
            'vocabulary_size': len(vocab), 'max_length': max_length,
            'training_tokens': sum(len(r['tokens']) for r in subsets['train']),
            'truncated_training_jokes': sum(len(r['tokens']) > max_length for r in subsets['train']),
            'validation_unknown_rate': sum(t not in mapping for r in subsets['validation'] for t in r['tokens']) /
                                       sum(len(r['tokens']) for r in subsets['validation'])}
    return data, vocab, meta, subsets


def glove_vectors(path, vocab, dimension=50, seed=42):
    """Stream plain or ZIP GloVe; retain only the training vocabulary."""
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0, 0.05, (len(vocab), dimension)).astype(np.float32)
    matrix[0] = 0
    needed = {w: i for i, w in enumerate(vocab) if i >= len(SPECIALS)}
    found = set()
    def consume(lines):
        for raw in lines:
            pieces = raw.decode('utf-8').rstrip().split(' ')
            if pieces[0] in needed and len(pieces) == dimension + 1:
                try:
                    vector = np.asarray(pieces[1:], dtype=np.float32)
                except ValueError:
                    continue
                if np.isfinite(vector).all():
                    matrix[needed[pieces[0]]] = vector
                    found.add(pieces[0])
    path = Path(path)
    if path.suffix == '.zip':
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.endswith('.txt') and not n.startswith('__MACOSX/')]
            if len(names) != 1:
                raise ValueError('GloVe ZIP must contain exactly one embedding .txt file')
            with archive.open(names[0]) as f:
                consume(f)
    else:
        with path.open('rb') as f:
            consume(f)
    if not found:
        raise ValueError(f'No {dimension}-dimensional GloVe vectors matched the vocabulary')
    return matrix, {'matched': len(found), 'eligible': len(needed),
                    'coverage': len(found) / len(needed), 'dimension': dimension,
                    'source_filename': path.name}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
