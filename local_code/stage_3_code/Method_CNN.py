"""Configurable convolutional classifiers for the three Stage 3 datasets."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from local_code.base_class.method import method


class Method_CNN(method, nn.Module):
    def __init__(self, name, channels, num_classes, *, kernel_size=3, hidden_dim=128,
                 pool="max", convolutions_per_block=1):
        method.__init__(self, f"{name} CNN", "Convolutional image classifier")
        nn.Module.__init__(self)
        if kernel_size not in (3, 5) or pool not in ("max", "avg") or hidden_dim <= 0:
            raise ValueError("Invalid CNN configuration")
        if not channels or any(c <= 0 for c in channels) or convolutions_per_block <= 0:
            raise ValueError("Invalid convolution widths")
        input_channels = 3 if name == "CIFAR" else 1
        blocks = []
        for width in channels:
            for _ in range(convolutions_per_block):
                blocks += [nn.Conv2d(input_channels, width, kernel_size,
                                     padding=kernel_size // 2), nn.ReLU()]
                input_channels = width
            blocks.append(nn.MaxPool2d(2) if pool == "max" else nn.AvgPool2d(2))
        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d((4, 4)))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(input_channels * 16, hidden_dim),
                                        nn.ReLU(), nn.Linear(hidden_dim, num_classes))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.classifier(self.features(x))

    def run(self, trainData, trainLabel, testData, *, epochs=1, batch_size=128,
            learning_rate=.001, device="cpu", seed=42):
        """Fit from NCHW arrays/tensors and return predictions and epoch losses."""
        x = torch.as_tensor(trainData, dtype=torch.float32)
        y = torch.as_tensor(trainLabel, dtype=torch.long)
        test = torch.as_tensor(testData, dtype=torch.float32)
        if x.ndim != 4 or test.ndim != 4 or len(x) != len(y):
            raise ValueError("Expected NCHW images and matching labels")
        if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
            raise ValueError("Training parameters must be positive")
        loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(seed))
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        history = []
        for _ in range(epochs):
            self.train()
            total = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                loss = criterion(self(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(batch_y)
            history.append(total / len(y))
        self.eval()
        predictions = []
        with torch.no_grad():
            for batch in test.split(batch_size):
                predictions.append(self(batch.to(device)).argmax(1).cpu())
        return {"pred_y": torch.cat(predictions).numpy(), "training_loss": history}


BASELINES = {
    "MNIST": dict(channels=(32, 64), num_classes=10),
    "ORL": dict(channels=(16, 32, 64), num_classes=40),
    "CIFAR": dict(channels=(32, 64, 128), num_classes=10),
}


def make_model(name, variant="baseline"):
    config = BASELINES[name].copy()
    if variant == "shallow":
        config["channels"] = config["channels"][:-1]
    elif variant == "kernel5" and name == "MNIST":
        config["kernel_size"] = 5
    elif variant == "hidden64" and name == "ORL":
        config["hidden_dim"] = 64
    elif variant == "avgpool" and name == "CIFAR":
        config["pool"] = "avg"
    elif variant != "baseline":
        raise ValueError(f"Unsupported variant {name}/{variant}")
    return Method_CNN(name, **config), config
