"""Run reproducible Stage 3 CNN baselines and ablations from the repo root."""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN import make_model


VARIANTS = {"MNIST": ("baseline", "shallow", "kernel5"),
            "ORL": ("baseline", "shallow", "hidden64"),
            "CIFAR": ("baseline", "shallow", "avgpool")}
EPOCHS = {"MNIST": 10, "ORL": 35, "CIFAR": 10}
BATCH = {"MNIST": 128, "ORL": 32, "CIFAR": 128}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stratified_indices(labels, seed):
    rng = np.random.default_rng(seed)
    train, val = [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        count = max(1, round(len(indices) * .1))
        val.extend(indices[:count])
        train.extend(indices[count:])
    return np.asarray(train), np.asarray(val)


def class_metrics(actual, predicted, classes):
    precision, recall, f1 = [], [], []
    for label in range(classes):
        tp = np.sum((actual == label) & (predicted == label))
        p = tp / max(1, np.sum(predicted == label))
        r = tp / max(1, np.sum(actual == label))
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / (p + r) if p + r else 0)
    return float(np.mean(precision)), float(np.mean(recall)), float(np.mean(f1))


def choose_device(wanted):
    if wanted == "auto":
        return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if wanted == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if wanted == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    return wanted


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    actual, predicted = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item() * len(y)
            actual.extend(y.cpu().tolist())
            predicted.extend(logits.argmax(1).cpu().tolist())
    return total_loss / len(loader.dataset), np.asarray(actual), np.asarray(predicted)


def plot_curves(path, histories, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for variant, rows in histories.items():
        ax.plot([r["epoch"] for r in rows], [r["train_loss"] for r in rows], label=f"{variant} train")
        ax.plot([r["epoch"] for r in rows], [r["val_loss"] for r in rows], linestyle="--", label=f"{variant} val")
    ax.set(title=title, xlabel="Epoch", ylabel="Mean cross-entropy loss")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run_dataset(name, args, device):
    data = Dataset_Loader(name).load()
    train_set, test_set = data["train"], data["test"]
    labels = train_set.labels.numpy()
    train_idx, val_idx = stratified_indices(labels, args.seed)
    train = Subset(train_set, train_idx)
    val = Subset(train_set, val_idx)
    val_loader = DataLoader(val, batch_size=BATCH[name])
    test_loader = DataLoader(test_set, batch_size=BATCH[name])
    output = args.output_dir / name.lower()
    if args.variant != "all":
        output = output / f"single_{args.variant}"
    output.mkdir(parents=True, exist_ok=True)
    histories, results = {}, {}
    variants = VARIANTS[name] if args.variant == "all" else (args.variant,)
    for variant in variants:
        if variant not in VARIANTS[name]:
            raise ValueError(f"{variant} is not a valid {name} variant: {VARIANTS[name]}")
        seed_everything(args.seed)
        model, config = make_model(name, variant)
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        loader = DataLoader(train, batch_size=BATCH[name], shuffle=True,
                            generator=torch.Generator().manual_seed(args.seed))
        best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
        rows = []
        started = time.monotonic()
        for epoch in range(1, (args.epochs or EPOCHS[name]) + 1):
            model.train()
            total = 0.0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(y)
            val_loss, _, _ = evaluate(model, val_loader, device, criterion)
            rows.append({"epoch": epoch, "train_loss": total / len(train), "val_loss": val_loss})
            print(f"{name} {variant} {epoch}: train={rows[-1]['train_loss']:.4f} val={val_loss:.4f}", flush=True)
            if val_loss < best_loss - 1e-4:
                best_loss, best_epoch, stale = val_loss, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= args.patience:
                    break
        model.load_state_dict(best_state)
        test_loss, y_true, y_pred = evaluate(model, test_loader, device, criterion)
        precision, recall, f1 = class_metrics(y_true, y_pred, config["num_classes"])
        results[variant] = {
            "config": config, "parameters": sum(p.numel() for p in model.parameters()),
            "best_epoch": best_epoch, "epochs_run": len(rows), "best_val_loss": best_loss,
            "test_loss": test_loss, "correct": int((y_true == y_pred).sum()),
            "accuracy": float(np.mean(y_true == y_pred)), "precision_macro": float(precision),
            "recall_macro": float(recall), "f1_macro": float(f1),
            "duration_seconds": round(time.monotonic() - started, 2),
        }
        histories[variant] = rows
        print(f"{name} {variant}: test accuracy={results[variant]['accuracy']:.4f}", flush=True)
        with (output / f"{variant}_predictions.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(("true_label", "predicted_label"))
            offset = 1 if name == "ORL" else 0
            writer.writerows(zip(y_true + offset, y_pred + offset))
        torch.save(best_state, output / f"{variant}_model.pt")
    with (output / "loss_history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("variant", "epoch", "train_loss", "val_loss"))
        writer.writeheader()
        for variant, rows in histories.items():
            writer.writerows({"variant": variant, **row} for row in rows)
    plot_curves(output / "learning_curves.png", histories, f"{name} CNN learning curves")
    report = {"dataset": name, "train_size": len(train), "val_size": len(val),
              "test_size": len(test_set), "seed": args.seed, "device": device,
              "learning_rate": args.learning_rate, "batch_size": BATCH[name],
              "selection": "minimum validation loss; patience early stopping",
              "models": results}
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", "MNIST", "ORL", "CIFAR"), default="all")
    parser.add_argument("--variant", choices=("all", "baseline", "shallow", "kernel5", "hidden64", "avgpool"), default="all")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "result/stage_3_result")
    args = parser.parse_args()
    if args.epochs is not None and args.epochs <= 0 or args.patience <= 0 or args.learning_rate <= 0:
        parser.error("epochs, patience and learning rate must be positive")
    device = choose_device(args.device)
    for name in VARIANTS if args.dataset == "all" else (args.dataset,):
        run_dataset(name, args, device)


if __name__ == "__main__":
    main()
