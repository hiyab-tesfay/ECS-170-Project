"""Attach complete provenance to finished checkpoints without changing weights."""
import json
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from local_code.stage_4_code.experiment import json_write


def main():
    root = ROOT / 'result/stage_4_result'
    for task in ['classification', 'generation']:
        dataset = json.loads((root / f'{task}_dataset.json').read_text())
        for path in (root / task).glob('*/*/config.json'):
            if not (path.parent / 'metrics.json').exists():
                raise RuntimeError(f'Experiment still incomplete: {path.parent}')
            config = json.loads(path.read_text())
            random = config.get('embedding_source') == 'random'
            config['embedding_source'] = 'random' if random else dataset['glove']
            config['initialization'] = ('Xavier input/output; gate-wise orthogonal recurrent; zero bias; '
                                        + ('random N(0, 1) embeddings' if random else 'pretrained GloVe embeddings'))
            config['learning_rate_schedule'] = ('validation plateau, patience 2, factor 0.5' if task == 'classification'
                                                else 'cosine to 10% of initial learning rate')
            config['preprocessing'] = {k: dataset[k] for k in ['seed', 'max_length', 'vocabulary_size', 'splits']}
            json_write(path, config)
            for name in ['best.pt', 'last.pt']:
                target = path.parent / name
                checkpoint = torch.load(target, map_location='cpu', weights_only=True)
                checkpoint['config'] = config
                torch.save(checkpoint, target)
            print(f'Finalized provenance: {path.parent.relative_to(root)}')


if __name__ == '__main__':
    main()
