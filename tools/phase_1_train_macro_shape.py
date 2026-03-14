"""
Phase 1A: Macro-Shape Network (Continents, Erosion, Ridges)

Trains a shallow neural network to predict intermediate DensityFunction outputs
from the Minecraft terrain pipeline. Specifically:
- Continents value (3D Perlin noise)
- Erosion value (3D Perlin noise)
- Ridges value (folded 3D Perlin noise)
"""

import torch
import torch.nn as nn
import torch.optim as optim


class MacroShapeNet(nn.Module):
    """Shallow network: (X, Z) coords + chunk seed → (continents, erosion, ridges) floats"""

    def __init__(self, hidden_size=128):
        super().__init__()
        # Input: X, Z coordinates (normalized), chunk index
        self.fc1 = nn.Linear(3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 3)  # continents, erosion, ridges
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_phase_1a(
    data_loader,
    val_loader,
    epochs=500,
    lr=0.001,
    early_stopping_patience=50,
):
    """
    Train Phase 1A macro-shape network.

    Args:
        data_loader: Training data (X, Z, seed) → (continents, erosion, ridges)
        val_loader: Validation data
        epochs: Max epochs
        lr: Learning rate
        early_stopping_patience: Stop after N epochs without improvement
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training MacroShapeNet on {device}")

    model = MacroShapeNet(hidden_size=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    loss_history = []
    convergence_message_shown = False

    for epoch in range(epochs):
        # Training
        train_loss = 0.0
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(data_loader)

        # Validation
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred = model(batch_x)
                loss = loss_fn(pred, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        loss_history.append((train_loss, val_loss))
        scheduler.step(val_loss)

        # Grokking detection
        if len(loss_history) > 10:
            prev_val_loss = loss_history[max(0, epoch - 10)][1]
            pct_improvement = (prev_val_loss - val_loss) / prev_val_loss * 100

            if pct_improvement < 0.5 and not convergence_message_shown:
                convergence_message_shown = True
                print("\n🧠 PHASE 1A GROKKED: Model has converged!")
                print(f"   Epoch: {epoch+1}, Val Loss: {val_loss:.6f}")
                print(f"   Pct improvement (last 10 epochs): {pct_improvement:.4f}%")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            status = ""
            if len(loss_history) > 10:
                prev_val_loss = loss_history[max(0, epoch - 10)][1]
                pct_improvement = (prev_val_loss - val_loss) / prev_val_loss * 100
                status = f" | pct_improve={pct_improvement:.3f}%"
            print(f"Epoch {epoch+1}/{epochs}: train={train_loss:.6f}, val={val_loss:.6f}{status}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(
                    f"\nEarly stopping at epoch {epoch+1} (no improvement for {early_stopping_patience} epochs)"
                )
                break

    return model, loss_history


def main():
    """Placeholder: Load data and train Phase 1A."""
    print("Phase 1A: Macro-Shape Network")
    print("=" * 50)
    print()
    print("This script will:")
    print("1. Load macro-shape ground-truth data (X, Z, seed) → (continents, erosion, ridges)")
    print("2. Train a shallow network to predict intermediate noise values")
    print("3. Save the model for Phase 2 combiner training")
    print()
    print("Status: Framework ready, waiting for data extraction phase...")
    print()
    print("Next: Run phase_1_data_extraction.py to generate training data")


if __name__ == "__main__":
    main()
