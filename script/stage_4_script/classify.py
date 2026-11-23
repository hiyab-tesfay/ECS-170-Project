"""Classify a new review using a saved classifier and its training vocabulary."""
import argparse
import json
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from local_code.stage_4_code.Dataset_Loader import tokenize, encode
from local_code.stage_4_code.experiment import choose_device, load_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--text', required=True)
    p.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    p.add_argument('--max-length', type=int)
    args = p.parse_args()
    if args.max_length is not None and args.max_length < 1:
        p.error('max length must be positive')
    torch.set_num_threads(2)
    device = choose_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    if checkpoint['config']['task'] != 'classification':
        p.error('checkpoint must be a classification model')
    mapping = {w: i for i, w in enumerate(checkpoint['vocabulary'])}
    maximum = args.max_length or checkpoint['config'].get('preprocessing', {}).get('max_length', 300)
    tokens = torch.tensor([encode(tokenize(args.text), mapping, maximum)], device=device)
    with torch.no_grad():
        probs = model(tokens).softmax(-1)[0].cpu().tolist()
    print(json.dumps({'text': args.text, 'prediction': ['negative', 'positive'][int(probs[1] > probs[0])],
                      'negative_probability': probs[0], 'positive_probability': probs[1]}, indent=2))


if __name__ == '__main__':
    main()
