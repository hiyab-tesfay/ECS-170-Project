"""Prepare fixed splits, training-only vocabularies, and streamed GloVe caches."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from local_code.stage_4_code.Dataset_Loader import classification_data, generation_data, glove_vectors, file_sha256
from local_code.stage_4_code.experiment import json_write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'stage_4_data')
    parser.add_argument('--glove', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, default=ROOT / 'result/stage_4_result/cache')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-length', type=int, default=300)
    parser.add_argument('--vocab-size', type=int, default=20000)
    args = parser.parse_args()
    if args.max_length < 1 or args.vocab_size < 5:
        parser.error('max length must be positive and vocabulary size at least 5')
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for task in ['classification', 'generation']:
        print(f'Preparing {task}', flush=True)
        if task == 'classification':
            data, vocab, meta, records = classification_data(args.data_dir, seed=args.seed, vocab_size=args.vocab_size, max_length=args.max_length)
        else:
            data, vocab, meta, records = generation_data(args.data_dir, seed=args.seed)
        print(f'Loading GloVe for {len(vocab)} tokens', flush=True)
        vectors, coverage = glove_vectors(args.glove, vocab, seed=args.seed)
        meta['glove'] = coverage
        meta['glove']['sha256'] = file_sha256(args.glove)
        payload = {'data': {s: {'sequences': d.sequences, 'labels': d.labels} for s, d in data.items()},
                   'vocab': vocab, 'metadata': meta, 'records': records, 'vectors': torch.from_numpy(vectors)}
        torch.save(payload, args.cache_dir / f'{task}.pt')
        metadata[task] = meta
        json_write(args.cache_dir.parent / f'{task}_dataset.json', meta)
        if task == 'classification':
            json_write(args.cache_dir.parent / 'classification_split_manifest.json', records)
        else:
            json_write(args.cache_dir.parent / 'generation_split_manifest.json',
                       {s: [r['id'] for r in part] for s, part in records.items()})
        print(json.dumps(meta, indent=2), flush=True)


if __name__ == '__main__':
    main()
