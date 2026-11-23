"""Train RNN/LSTM/GRU on either Stage 4 task; preserve all experiment evidence."""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from local_code.stage_4_code.Dataset_Loader import SequenceDataset
from local_code.stage_4_code.experiment import choose_device, generation_samples, json_write, load_checkpoint, train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=['classification', 'generation'], required=True)
    p.add_argument('--cell', choices=['rnn', 'lstm', 'gru', 'all'], default='all')
    p.add_argument('--cache-dir', type=Path, default=ROOT / 'result/stage_4_result/cache')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'result/stage_4_result')
    p.add_argument('--run', default='comparison')
    p.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    p.add_argument('--epochs', type=int)
    p.add_argument('--hidden-size', type=int)
    p.add_argument('--layers', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--learning-rate', type=float)
    p.add_argument('--dropout', type=float)
    p.add_argument('--patience', type=int)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--freeze-embeddings', action='store_true')
    p.add_argument('--random-embeddings', action='store_true')
    p.add_argument('--unidirectional', action='store_true')
    p.add_argument('--skip-completed', action='store_true')
    args = p.parse_args()
    task = args.task
    if args.batch_size < 1 or args.layers < 1 or args.threads < 1:
        p.error('batch size, layers, and threads must be positive')
    epochs = args.epochs if args.epochs is not None else (12 if task == 'classification' else 100)
    patience = args.patience if args.patience is not None else (4 if task == 'classification' else 0)
    lr = args.learning_rate if args.learning_rate is not None else (0.001 if task == 'classification' else 0.003)
    if epochs < 1 or lr <= 0 or patience < 0:
        p.error('epochs and learning rate must be positive; patience must be nonnegative')
    if args.hidden_size is not None and args.hidden_size < 1:
        p.error('hidden size must be positive')
    dropout = args.dropout if args.dropout is not None else (0.4 if task == 'classification' else 0.2)
    if not 0 <= dropout < 1:
        p.error('dropout must be in [0, 1)')
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    payload = torch.load(args.cache_dir / f'{task}.pt', map_location='cpu', weights_only=True)
    if payload['metadata']['seed'] != args.seed:
        p.error('Seed must match the data preparation seed; prepare another cache for another split')
    data = {s: SequenceDataset(d['sequences'], d['labels']) for s, d in payload['data'].items()}
    cells = (['gru', 'lstm', 'rnn'] if task == 'classification' else ['rnn', 'lstm', 'gru']) if args.cell == 'all' else [args.cell]
    for cell in cells:
        config = dict(cell=cell, embedding_dim=payload['vectors'].shape[1],
                      hidden_size=args.hidden_size or (128 if task == 'classification' else 256),
                      layers=args.layers, dropout=dropout)
        if task == 'classification':
            config.update(bidirectional=not args.unidirectional, freeze_embeddings=args.freeze_embeddings)
        output = args.output_dir / task / args.run / cell
        if args.skip_completed and (output / 'metrics.json').exists():
            print(f'Skipping completed experiment: {output}', flush=True)
            continue
        print(f'Device={device}, output={output}', flush=True)
        model, metrics = train(task, data, payload['vocab'], config, output,
                               epochs=epochs, batch_size=args.batch_size, learning_rate=lr,
                               patience=patience, seed=args.seed, device=device,
                               pretrained=None if args.random_embeddings else payload['vectors'])
        settings = json.loads((output / 'config.json').read_text())
        settings['embedding_source'] = 'random' if args.random_embeddings else payload['metadata']['glove']
        json_write(output / 'config.json', settings)
        if task == 'generation':
            samples = generation_samples(model, payload['vocab'], payload['records'], device)
            json_write(output / 'samples_best.json', samples)
            last_model, _ = load_checkpoint(output / 'last.pt', device)
            json_write(output / 'samples_last.json', generation_samples(last_model, payload['vocab'], payload['records'], device))
        print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
