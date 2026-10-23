"""Multi-layer perceptron for the Stage 2 digit dataset."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from local_code.base_class.method import method


class Method_MLP(method, nn.Module):
    """Train an MLP on 784 pixel values and predict one of 10 digits."""

    def __init__(self, mName="stage 2 MLP", mDescription="", hidden_layers=(128, 64)):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        if not hidden_layers or any(width <= 0 for width in hidden_layers):
            raise ValueError("hidden_layers must contain positive layer widths")

        dimensions = (784, *hidden_layers, 10)
        layers = []
        for index, (input_size, output_size) in enumerate(
            zip(dimensions[:-1], dimensions[1:])
        ):
            linear = nn.Linear(input_size, output_size)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            if index < len(dimensions) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
        self.dimensions = dimensions

    def forward(self, x):
        # CrossEntropyLoss expects raw scores (logits), not softmax probabilities.
        return self.network(x)

    def fit(self, X, y, *, epochs=10, batch_size=256, learning_rate=0.001, seed=42,
            device="cpu"):
        """Return the mean training loss for each epoch."""
        if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
            raise ValueError("epochs, batch_size, and learning_rate must be positive")

        train_data = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        generator = torch.Generator().manual_seed(seed)
        batches = DataLoader(
            train_data, batch_size=batch_size, shuffle=True, generator=generator
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_function = nn.CrossEntropyLoss()
        self.to(device)

        losses = []
        for epoch in range(1, epochs + 1):
            self.train()
            total_loss = 0.0
            for batch_X, batch_y in batches:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                loss = loss_function(self(batch_X), batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_y)

            mean_loss = total_loss / len(train_data)
            losses.append(mean_loss)
            print(f"  epoch {epoch:2d}/{epochs}: training loss = {mean_loss:.4f}", flush=True)
        return losses

    def predict(self, X, *, batch_size=256, device="cpu"):
        """Return predicted digit labels without tracking gradients."""
        self.to(device)
        self.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                batch = torch.from_numpy(X[start:start + batch_size]).to(device)
                predictions.append(self(batch).argmax(dim=1).cpu().numpy())
        return np.concatenate(predictions)

    def run(self, trainData, trainLabel, testData, *, epochs=10, batch_size=256,
            learning_rate=0.001, seed=42, device="cpu"):
        losses = self.fit(
            trainData, trainLabel, epochs=epochs, batch_size=batch_size,
            learning_rate=learning_rate, seed=seed, device=device
        )
        return {
            "pred_y": self.predict(testData, batch_size=batch_size, device=device),
            "training_loss": losses,
        }
