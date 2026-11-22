"""Create report figures directly from saved, measured experiment histories."""
import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/ecs170-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
from matplotlib.lines import Line2D
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
COLORS = {'rnn': '#b65b35', 'lstm': '#2468a0', 'gru': '#218273'}


def history(path):
    with path.open() as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def curves(root, task, destination):
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.8), sharey=True)
    for ax, cell in zip(axes, ['rnn', 'lstm', 'gru']):
        folder = root / task / 'comparison' / cell
        rows = history(folder / 'history.csv')
        epochs = [r['epoch'] for r in rows]
        ax.plot(epochs, [r['training_loss'] for r in rows], color=COLORS[cell], label='Training')
        ax.plot(epochs, [r['validation_loss'] for r in rows], color=COLORS[cell], linestyle='--', label='Validation')
        if (folder / 'metrics.json').exists():
            best = json.loads((folder / 'metrics.json').read_text())['best_epoch']
            ax.axvline(best, color='#888888', linewidth=0.8, linestyle=':', label='Best validation epoch')
        ax.set_title(cell.upper(), fontweight='bold', fontsize=11)
        ax.set_xlabel('Epoch')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel('Cross-entropy loss')
    handles = [Line2D([], [], color='#444444', linestyle=s) for s in ['-', '--', ':']]
    fig.legend(handles, ['Training', 'Validation', 'Best validation epoch'], loc='lower center', ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    for ext in ['png', 'svg']:
        fig.savefig(destination / f'{task}_learning_curves.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    if task == 'classification':
        fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.8), sharey=True)
        for ax, cell in zip(axes, ['rnn', 'lstm', 'gru']):
            rows = history(root / task / 'comparison' / cell / 'history.csv')
            epochs = [r['epoch'] for r in rows]
            ax.plot(epochs, [r['training_accuracy'] for r in rows], color=COLORS[cell], label='Training')
            ax.plot(epochs, [r['validation_accuracy'] for r in rows], color=COLORS[cell], linestyle='--', label='Validation')
            ax.axhline(.85, color='#777777', linestyle=':', linewidth=0.9, label='85% reference')
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_title(cell.upper(), fontweight='bold'); ax.set_xlabel('Epoch'); ax.grid(alpha=.2)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        axes[0].set_ylabel('Accuracy')
        fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=3, frameon=False, fontsize=8)
        fig.tight_layout(rect=[0, .09, 1, 1])
        for ext in ['png', 'svg']:
            fig.savefig(destination / f'classification_accuracy_curves.{ext}', dpi=200, bbox_inches='tight')
        plt.close(fig)


def confusion(root, destination):
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.8))
    for ax, cell in zip(axes, ['rnn', 'lstm', 'gru']):
        metrics = json.loads((root / 'classification/comparison' / cell / 'metrics.json').read_text())
        matrix = np.array(metrics['confusion_matrix'])
        ax.imshow(matrix, cmap='Blues', vmin=0, vmax=12500)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{matrix[i,j]:,}', ha='center', va='center', color='white' if matrix[i,j] > 7000 else 'black', fontsize=10)
        ax.set_xticks([0, 1], ['Negative', 'Positive'])
        ax.set_yticks([0, 1], ['Negative', 'Positive'])
        ax.set_xlabel('Predicted')
        ax.set_title(f'{cell.upper()} | {metrics["accuracy"]:.2%}', fontsize=10)
    axes[0].set_ylabel('Actual')
    fig.tight_layout()
    fig.savefig(destination / 'confusion_matrices.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def architecture(destination):
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 2.6))
    rows = [
        ['Review\nword IDs', 'GloVe\n50 dimensions', 'Forward + reverse\nRNN / LSTM / GRU', 'Masked\nmean + max', '2 logits\nSentiment'],
        ['BOS + joke\nword IDs', 'GloVe\n50 dimensions', 'Causal\nRNN / LSTM / GRU', 'Vocabulary\nlogits per step', 'Next word\n+ EOS']]
    for ax, labels, title in zip(axes, rows, ['CLASSIFICATION', 'GENERATION']):
        ax.set_xlim(-0.3, 10.3); ax.set_ylim(-0.2, 1.4); ax.axis('off')
        ax.text(0, 1.12, title, fontsize=9, fontweight='bold', color='#27394b')
        for i, label in enumerate(labels):
            x = i * 2.1
            ax.add_patch(FancyBboxPatch((x, 0), 1.7, 0.8, boxstyle='round,pad=0.04', linewidth=0.8,
                                       edgecolor='#647b8b', facecolor='#eef3f6'))
            ax.text(x + 0.85, 0.4, label, ha='center', va='center', fontsize=11)
            if i < 4:
                ax.add_patch(FancyArrowPatch((x + 1.73, 0.4), (x + 2.04, 0.4), arrowstyle='->', mutation_scale=9, color='#647b8b'))
    fig.tight_layout(pad=0.1)
    fig.savefig(destination / 'architectures.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=ROOT / 'result/stage_4_result')
    args = p.parse_args()
    destination = args.results / 'figures'
    destination.mkdir(exist_ok=True, parents=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    architecture(destination)
    for task in ['classification', 'generation']:
        if all((args.results / task / 'comparison' / c / 'history.csv').exists() for c in COLORS):
            curves(args.results, task, destination)
    if all((args.results / 'classification/comparison' / c / 'metrics.json').exists() for c in COLORS):
        confusion(args.results, destination)
    for path in destination.glob('*.svg'):
        # Matplotlib emits trailing spaces inside multi-line path attributes.
        # Newlines remain token separators, so trimming is visually lossless.
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    print(f'Saved figures to {destination}')


if __name__ == '__main__':
    main()
