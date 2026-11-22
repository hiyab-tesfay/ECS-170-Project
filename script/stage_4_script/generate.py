"""Generate a joke from exactly three words using a saved model, without data lookup."""
import argparse
import json
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from local_code.stage_4_code.experiment import choose_device, generate, load_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--prompt', required=True)
    p.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top-k', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-words', type=int, default=60)
    args = p.parse_args()
    torch.set_num_threads(2)
    device = choose_device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    if ckpt['config']['task'] != 'generation':
        p.error('checkpoint must be a generation model')
    print(json.dumps(generate(model, ckpt['vocabulary'], args.prompt, device,
                              temperature=args.temperature, top_k=args.top_k,
                              seed=args.seed, max_words=args.max_words), indent=2))


if __name__ == '__main__':
    main()
