"""Compare Stage 3 CNN training throughput on CPU and an available GPU."""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN import make_model


def available_accelerator():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def time_training(device, batches, epochs, seed):
    torch.manual_seed(seed)
    model, _ = make_model("MNIST")
    model.to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Warm up the backend without changing weights or optimizer state.
    x, y = batches[0]
    criterion(model(x.to(device)), y.to(device)).backward()
    model.zero_grad(set_to_none=True)
    synchronize(device)

    started = time.perf_counter()
    for _ in range(epochs):
        for x, y in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
    synchronize(device)
    seconds = time.perf_counter() - started
    images = epochs * sum(len(y) for _, y in batches)
    return {"seconds": round(seconds, 4), "images": images,
            "images_per_second": round(images / seconds, 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "result/stage_3_result/device_benchmark.json")
    args = parser.parse_args()
    if min(args.samples, args.epochs, args.batch_size, args.cpu_threads) <= 0:
        parser.error("Samples, epochs, batch size, and CPU threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    train = Dataset_Loader("MNIST").load()["train"]
    if args.samples > len(train):
        parser.error(f"Samples must be <= {len(train)}")
    batches = []
    for start in range(0, args.samples, args.batch_size):
        stop = min(start + args.batch_size, args.samples)
        images = torch.stack([train[i][0] for i in range(start, stop)])
        labels = train.labels[start:stop]
        batches.append((images, labels))

    accelerator = available_accelerator()
    result = {
        "dataset": "MNIST training partition only", "model": "baseline CNN",
        "platform": platform.platform(), "machine": platform.machine(),
        "pytorch": torch.__version__, "samples": args.samples,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "cpu_threads": args.cpu_threads, "seed": args.seed,
        "cpu": time_training("cpu", batches, args.epochs, args.seed),
        "gpu_device": accelerator,
        "gpu": None,
        "interpretation": "Throughput comparison only; this is not a test accuracy result.",
    }
    if accelerator:
        result["gpu"] = time_training(accelerator, batches, args.epochs, args.seed)
        result["speedup_vs_cpu"] = round(
            result["cpu"]["seconds"] / result["gpu"]["seconds"], 3)
    else:
        result["gpu_status"] = "Unavailable: CUDA and MPS are both unavailable on this machine."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
