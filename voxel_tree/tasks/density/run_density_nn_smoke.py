"""Smoke-test training run for the density NN using the canonical NPZ dataset.

This script mirrors the notebook workflow but runs as a standalone script to ensure the
NPZ data can be loaded and used for training.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class SimpleDensityNet(nn.Module):
    def __init__(self, input_channels=11, output_channels=3):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv3d(64, output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        return x


def main():
    parser = argparse.ArgumentParser(description="Smoke-test density NN training")
    parser.add_argument("--data", default="noise_training_data/terrain_shaper_density_data.npz")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=1024)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    data = np.load(data_path)
    inputs = data["inputs"]
    outputs = data["outputs"]
    print(f"Loaded NPZ: inputs={inputs.shape}, outputs={outputs.shape}")

    # Use a subset to keep runtime small
    n = min(len(inputs), args.max_samples)
    inputs = inputs[:n]
    outputs = outputs[:n]

    # Split
    idx = int(n * 0.8)
    X_train = inputs[:idx]
    Y_train = outputs[:idx]
    X_val = inputs[idx:]
    Y_val = outputs[idx:]

    # Convert to torch tensors and permute to (N, C, D, H, W)
    def to_tensor(x):
        t = torch.from_numpy(x.astype(np.float32))
        # Notebook expects (N, 4, 48, 4, C) and permutes to (N, C, 4, 48, 4)
        return t.permute(0, 4, 1, 2, 3)

    X_train_t = to_tensor(X_train)
    Y_train_t = to_tensor(Y_train)
    X_val_t = to_tensor(X_val)
    Y_val_t = to_tensor(Y_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleDensityNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print(f"Running on device: {device}")

    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for i in range(0, X_train_t.shape[0], args.batch_size):
            xb = X_train_t[i : i + args.batch_size].to(device)
            yb = Y_train_t[i : i + args.batch_size].to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, (X_train_t.shape[0] // args.batch_size))
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(0, X_val_t.shape[0], args.batch_size):
                xb = X_val_t[i : i + args.batch_size].to(device)
                yb = Y_val_t[i : i + args.batch_size].to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item()
        val_loss /= max(1, (X_val_t.shape[0] // args.batch_size))
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{args.epochs} - train={train_loss:.5f} val={val_loss:.5f}")

    print("Done.")


if __name__ == "__main__":
    main()
