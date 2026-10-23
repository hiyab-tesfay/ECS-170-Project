"""Train two Stage 2 MLPs and save results for the project report.

Run from the repository root with:
    .venv/bin/python script/stage_2_script/script_mlp.py
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP


def save_loss_plot(path, histories):
    """Write a report-ready SVG plot without requiring another package."""
    width, height = 800, 480
    left, right, top, bottom = 80, 30, 45, 65
    plot_width = width - left - right
    plot_height = height - top - bottom
    epoch_count = max(len(losses) for losses in histories.values())
    y_max = max(loss for losses in histories.values() for loss in losses) * 1.1
    y_max = max(y_max, 0.1)

    def x_position(epoch):
        return left + (epoch - 1) * plot_width / max(epoch_count - 1, 1)

    def y_position(loss):
        return top + plot_height * (1 - loss / y_max)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Arial, sans-serif" fill="#222">',
        '<text x="400" y="28" text-anchor="middle" font-size="19">Training loss by epoch</text>',
    ]
    for tick in range(6):
        value = y_max * tick / 5
        y = y_position(value)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" '
            'stroke="#e5e7eb"/>'
        )
        parts.append(
            f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="12">'
            f'{value:.2f}</text>'
        )
    for epoch in range(1, epoch_count + 1):
        if epoch_count > 20 and epoch not in (1, epoch_count) and epoch % 5:
            continue
        x = x_position(epoch)
        parts.append(
            f'<text x="{x:.1f}" y="{height-bottom+20}" text-anchor="middle" '
            f'font-size="12">{epoch}</text>'
        )
    parts.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
        f'y2="{height-bottom}" stroke="#333"/>',
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-size="14">Epoch</text>',
        f'<text x="22" y="{height/2}" text-anchor="middle" font-size="14" '
        'transform="rotate(-90 22 240)">Mean training loss</text>',
    ])
    colors = ("#2563eb", "#dc2626")
    for index, (name, losses) in enumerate(histories.items()):
        color = colors[index % len(colors)]
        points = " ".join(
            f"{x_position(epoch):.1f},{y_position(loss):.1f}"
            for epoch, loss in enumerate(losses, start=1)
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="2.5" stroke-linejoin="round"/>'
        )
        legend_y = top + 18 + index * 23
        parts.append(
            f'<line x1="{width-220}" y1="{legend_y}" x2="{width-195}" '
            f'y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
        )
        parts.append(
            f'<text x="{width-188}" y="{legend_y+4}" font-size="12">'
            f'{escape(name.replace("_", " "))}</text>'
        )
    parts.extend(["</g>", "</svg>"])
    path.write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "result" / "stage_2_result",
    )
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("epochs, batch size, and learning rate must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available on this machine")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is not available on this machine")

    data = Dataset_Loader().load()
    print(
        f"Loaded {len(data['train']['y'])} training rows and "
        f"{len(data['test']['y'])} test rows.", flush=True
    )
    # Keep the supplied test split untouched during training.
    configurations = {
        "baseline": (128, 64),
        "one_hidden_layer": (128,),
    }
    histories = {}
    results = {}
    for name, hidden_layers in configurations.items():
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        model = Method_MLP(hidden_layers=hidden_layers)
        print(f"\n{name}: {' -> '.join(map(str, model.dimensions))}", flush=True)
        outcome = model.run(
            data["train"]["X"], data["train"]["y"], data["test"]["X"],
            epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, seed=args.seed, device=args.device,
        )
        precision, recall, f1, _ = precision_recall_fscore_support(
            data["test"]["y"], outcome["pred_y"], labels=list(range(10)),
            average="macro", zero_division=0,
        )
        histories[name] = outcome["training_loss"]
        results[name] = {
            "architecture": list(model.dimensions),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "accuracy": float(accuracy_score(data["test"]["y"], outcome["pred_y"])),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
            "final_training_loss": histories[name][-1],
        }
        print(f"  test accuracy = {results[name]['accuracy']:.4f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": "Stage 2 digits, supplied 60,000/10,000 train/test split",
        "feature_scaling": "pixel value / 255",
        "initialization": "Xavier uniform weights and zero biases",
        "activation": "ReLU after each hidden layer; raw output logits",
        "loss": "CrossEntropyLoss",
        "optimizer": "Adam",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "device": args.device,
        "metric_average": "macro across labels 0-9; zero_division=0",
        "models": results,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "loss_history.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("model", "epoch", "mean_training_loss"))
        for name, losses in histories.items():
            for epoch, loss in enumerate(losses, start=1):
                writer.writerow((name, epoch, loss))
    save_loss_plot(args.output_dir / "training_loss.svg", histories)
    print(f"\nSaved results to {args.output_dir}")


if __name__ == "__main__":
    main()
