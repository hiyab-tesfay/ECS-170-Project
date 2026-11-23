"""Recompute saved classification results and check required submission artifacts."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
ROOT = Path(__file__).resolve().parents[2]


def main():
    root = ROOT / 'result/stage_4_result'
    checks = []
    accuracies = []
    for cell in ['rnn', 'lstm', 'gru']:
        folder = root / 'classification/comparison' / cell
        metrics = json.loads((folder / 'metrics.json').read_text())
        with (folder / 'predictions.csv').open() as f:
            predictions = list(csv.DictReader(f))
        assert len(predictions) == 25000, f'{cell}: wrong test count'
        y = [int(r['true_label']) for r in predictions]
        pred = [int(r['predicted_label']) for r in predictions]
        acc = accuracy_score(y, pred)
        assert abs(acc - metrics['accuracy']) < 1e-12
        pr, re, f1, _ = precision_recall_fscore_support(y, pred, average='macro', zero_division=0)
        for name, value in zip(['precision_macro', 'recall_macro', 'f1_macro'], [pr, re, f1]):
            assert abs(value - metrics[name]) < 1e-12
        assert confusion_matrix(y, pred).tolist() == metrics['confusion_matrix']
        accuracies.append(acc)
        for name in ['history.csv', 'config.json', 'vocabulary.json', 'best.pt']:
            assert (folder / name).exists(), folder / name
        checks.append(f'{cell.upper()} classification metrics recomputed from all 25,000 predictions')
    assert max(accuracies) >= 0.85, 'No classifier meets the 85% requirement'
    ablation = root / 'classification/random_embeddings/gru'
    extra = json.loads((ablation / 'metrics.json').read_text())
    with (ablation / 'predictions.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 25000
    extra_accuracy = accuracy_score([int(r['true_label']) for r in rows],
                                    [int(r['predicted_label']) for r in rows])
    assert abs(extra_accuracy - extra['accuracy']) < 1e-12
    checks.append('Additional embedding ablation accuracy recomputed from predictions')
    manifests = json.loads((root / 'classification_split_manifest.json').read_text())
    assert not set(manifests['train']) & set(manifests['validation'])
    assert not set(manifests['train']) & set(manifests['test'])
    assert not set(manifests['validation']) & set(manifests['test'])
    gen_manifest = json.loads((root / 'generation_split_manifest.json').read_text())
    assert not set(gen_manifest['train']) & set(gen_manifest['validation'])
    checks.append('Train/validation/test manifests are disjoint')
    for cell in ['rnn', 'lstm', 'gru']:
        folder = root / 'generation/comparison' / cell
        for name in ['history.csv', 'config.json', 'metrics.json', 'vocabulary.json', 'best.pt', 'last.pt']:
            assert (folder / name).exists(), folder / name
        for name in ['samples_best.json', 'samples_last.json']:
            samples = json.loads((folder / name).read_text())
            assert len(samples) == 36
            assert all(len(s['prompt'].split()) == 3 for s in samples)
            assert all(s['generated_tokens'] > 0 for s in samples)
            assert all(s['text'].lower().startswith(s['prompt'].lower()) for s in samples)
            assert all('<pad>' not in s['text'] and '<unk>' not in s['text'] for s in samples)
        checks.append(f'{cell.upper()} generation trained; 72 decoded samples saved')
    for name in ['architectures.png', 'classification_learning_curves.png', 'generation_learning_curves.png', 'confusion_matrices.png']:
        assert (root / 'figures' / name).exists(), name
    import fitz
    report = ROOT / 'report/stage_4/Stage_4_RNN_Report.pdf'
    doc = fitz.open(report)
    assert 1 <= len(doc) <= 5, f'Report is {len(doc)} pages'
    text = '\n'.join(page.get_text() for page in doc)
    for section in ['Task Description', 'Model Description', 'Dataset Description',
                    'Detailed Experimental Setups', 'Evaluation Metrics', 'Source Code',
                    'Training Convergence Plot', 'Model Performance', 'Ablation Studies']:
        assert section in text, f'Report missing {section}'
    assert 'TODO' not in text and 'Enter Your' not in text
    checks.append(f'Report contains all required sections in {len(doc)} pages')
    manual = json.loads((root / 'manual_generation_review.json').read_text())
    assert all(manual[c]['readable_example_verified'] for c in ['rnn', 'lstm', 'gru'])
    checks.append('Representative decoded output manually reviewed for each generator')
    result = {'passed': True, 'best_test_accuracy': max(accuracies), 'checks': checks}
    (root / 'submission_checks.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
