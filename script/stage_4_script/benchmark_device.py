"""Time matching training batches on CPU and GPU; choose devices using timings."""
import argparse
import copy
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from local_code.stage_4_code.Method_RNN import TextClassifier
from local_code.stage_4_code.experiment import json_write, seed_everything, sync


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'result/stage_4_result/device_benchmark.json')
    p.add_argument('--steps', type=int, default=5)
    args = p.parse_args()
    if args.steps < 1:
        p.error('steps must be positive')
    torch.set_num_threads(2)
    seed_everything(42)
    devices = ['cpu'] + (['mps'] if torch.backends.mps.is_available() else []) + (['cuda'] if torch.cuda.is_available() else [])
    x, y = torch.randint(4, 20000, (64, 240)), torch.randint(0, 2, (64,))
    for i in range(len(x)):
        x[i, 80 + (i * 7) % 161:] = 0
    results = []
    for cell in ['rnn', 'lstm', 'gru']:
        baseline = TextClassifier(20000, cell=cell)
        state = copy.deepcopy(baseline.state_dict())
        for name in devices:
            device = torch.device(name)
            model = TextClassifier(20000, cell=cell).to(device)
            model.load_state_dict(state)
            opt = torch.optim.Adam(model.parameters(), lr=0.001)
            bx, by = x.to(device), y.to(device)
            try:
                for _ in range(2):
                    opt.zero_grad(); F.cross_entropy(model(bx), by).backward(); opt.step()
                sync(device)
                start = time.perf_counter()
                for _ in range(args.steps):
                    opt.zero_grad(); F.cross_entropy(model(bx), by).backward(); opt.step()
                sync(device)
                seconds = (time.perf_counter() - start) / args.steps
                row = dict(cell=cell, device=name, seconds_per_training_batch=seconds,
                           batch_size=64, sequence_length=240, timed_batches=args.steps,
                           warmup_batches=2, threads=2, padded_sequences=True)
            except (RuntimeError, NotImplementedError) as exc:
                row = dict(cell=cell, device=name, unsupported_reason=str(exc)[:300])
            results.append(row)
            json_write(args.output, results)
            print(row, flush=True)


if __name__ == '__main__':
    main()
