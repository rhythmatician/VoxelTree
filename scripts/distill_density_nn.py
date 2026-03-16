# distill_density_nn.py
"""
Distill a fast student density NN from a slower, more accurate teacher.
Usage:
    python distill_density_nn.py --teacher unet --student sep --epochs 120 --alpha 0.5 --device cuda

Outputs:
    - Distilled student checkpoint (.pt)
    - Distillation summary (.json)
"""
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# --- Candidate architectures ---
from density_nn_shootout import CANDIDATES, count_parameters

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# --- Data loading ---
def find_repo_root() -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "noise_training_data" / "stage1_density_data.npz").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find repo root containing noise_training_data/stage1_density_data.npz"
    )


REPO_ROOT = find_repo_root()
DATA_PATH = REPO_ROOT / "noise_training_data" / "stage1_density_data.npz"
ARTIFACT_DIR = (
    REPO_ROOT / "VoxelTree" / "notebooks" / "experimental" / "artifacts" / "density_nn_shootout"
)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

data = np.load(DATA_PATH)
inputs = data["inputs"].astype(np.float32)
targets = data["outputs"].astype(np.float32)
INPUT_CH = inputs.shape[-1]
OUTPUT_CH = targets.shape[-1]
N = inputs.shape[0]

perm = np.random.default_rng(SEED).permutation(N)
n_train = max(1, int(0.70 * N))
n_val = max(1, int(0.15 * N))
train_idx = perm[:n_train]
val_idx = perm[n_train : n_train + n_val]
test_idx = perm[n_train + n_val :]
if len(test_idx) == 0:
    test_idx = val_idx

X_train = torch.from_numpy(inputs[train_idx]).permute(0, 4, 1, 2, 3).contiguous()
Y_train = torch.from_numpy(targets[train_idx]).permute(0, 4, 1, 2, 3).contiguous()
X_val = torch.from_numpy(inputs[val_idx]).permute(0, 4, 1, 2, 3).contiguous()
Y_val = torch.from_numpy(targets[val_idx]).permute(0, 4, 1, 2, 3).contiguous()
X_test = torch.from_numpy(inputs[test_idx]).permute(0, 4, 1, 2, 3).contiguous()
Y_test = torch.from_numpy(targets[test_idx]).permute(0, 4, 1, 2, 3).contiguous()

BATCH_SIZE = 8
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=BATCH_SIZE, shuffle=False)

BOUNDARY_EPS = 0.05
SIGN_LOSS_WEIGHT = 0.20
WEIGHT_DECAY = 1e-5


# --- Evaluation ---
@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    sum_mse = 0.0
    sum_mae = 0.0
    sum_sign = 0.0
    sum_boundary = 0.0
    sum_boundary_count = 0.0
    batches = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        sum_mse += F.mse_loss(pred, yb).item()
        sum_mae += F.l1_loss(pred, yb).item()
        sum_sign += ((pred > 0) == (yb > 0)).float().mean().item()
        boundary_mask = (yb.abs() <= BOUNDARY_EPS).float()
        if boundary_mask.sum().item() > 0:
            correct = (((pred > 0) == (yb > 0)).float() * boundary_mask).sum().item()
            sum_boundary += correct
            sum_boundary_count += boundary_mask.sum().item()
        batches += 1
    return {
        "mse": sum_mse / max(batches, 1),
        "mae": sum_mae / max(batches, 1),
        "sign_acc": sum_sign / max(batches, 1),
        "boundary_sign_acc": (
            (sum_boundary / sum_boundary_count) if sum_boundary_count else float("nan")
        ),
    }


@torch.no_grad()
def benchmark_latency_ms(model, sample, device, warmup=20, runs=200):
    model = copy.deepcopy(model).to(device).eval()
    x = sample.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        t0 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        t1 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        if device.type == "cuda":
            t0.record()
        else:
            start = time.perf_counter()
        _ = model(x)
        if device.type == "cuda":
            t1.record()
            torch.cuda.synchronize()
            times.append(t0.elapsed_time(t1))
        else:
            times.append((time.perf_counter() - start) * 1000.0)
    arr = np.array(times, dtype=np.float64)
    return {
        "median_ms": float(np.median(arr)),
        "p90_ms": float(np.percentile(arr, 90)),
        "chunks_per_s": float(1000.0 / max(np.median(arr), 1e-9)),
    }


# --- Distillation ---
def load_best_model(name, device):
    path = ARTIFACT_DIR / f"{name}_best.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {path}\nRun the main shootout first or manually train the teacher."
        )
    model = CANDIDATES[name]["build"]()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def distill_student(teacher_name, student_name, epochs, alpha, lr, device):
    teacher = load_best_model(teacher_name, device)
    student = CANDIDATES[student_name]["build"]().to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    history = []
    best_val = math.inf
    best_state = None
    for epoch in range(epochs):
        student.train()
        train_loss = 0.0
        train_mse = 0.0
        train_sign = 0.0
        batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            with torch.no_grad():
                teacher_pred = teacher(xb)
            pred = student(xb)
            mse_gt = F.mse_loss(pred, yb)
            mse_teacher = F.mse_loss(pred, teacher_pred)
            sign_target = (yb > 0).float()
            sign_loss = F.binary_cross_entropy_with_logits(pred * 8.0, sign_target)
            loss = alpha * mse_gt + (1 - alpha) * mse_teacher + SIGN_LOSS_WEIGHT * sign_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach())
            train_mse += float(mse_gt.detach())
            train_sign += ((pred > 0) == (yb > 0)).float().mean().item()
            batches += 1
        scheduler.step()
        val_metrics = evaluate_model(student, val_loader, device)
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss / max(batches, 1),
            "train_mse": train_mse / max(batches, 1),
            "train_sign_acc": train_sign / max(batches, 1),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        if row["val_mse"] < best_val:
            best_val = row["val_mse"]
            best_state = copy.deepcopy(student.state_dict())
        print(
            f"[distill {student_name}<-{teacher_name}] epoch {epoch+1:03d}/{epochs} | train_mse={row['train_mse']:.5f} | val_mse={row['val_mse']:.5f} | val_sign={row['val_sign_acc']:.4f}"
        )
    student.load_state_dict(best_state)
    val_best = evaluate_model(student, val_loader, device)
    test_metrics = evaluate_model(student, test_loader, device)
    sample = X_test[:1]
    cpu_bench = benchmark_latency_ms(student, sample, torch.device("cpu"))
    active_bench = benchmark_latency_ms(student, sample, device) if device.type == "cuda" else None
    ckpt_path = ARTIFACT_DIR / f"distill_{student_name}_from_{teacher_name}.pt"
    torch.save(best_state, ckpt_path)
    result = {
        "teacher": teacher_name,
        "student": student_name,
        "alpha": alpha,
        "epochs": epochs,
        "checkpoint_path": str(ckpt_path),
        "best_val_mse": val_best["mse"],
        "best_val_sign_acc": val_best["sign_acc"],
        "best_val_boundary_sign_acc": val_best["boundary_sign_acc"],
        "test_mse": test_metrics["mse"],
        "test_mae": test_metrics["mae"],
        "test_sign_acc": test_metrics["sign_acc"],
        "test_boundary_sign_acc": test_metrics["boundary_sign_acc"],
        "cpu_median_ms": cpu_bench["median_ms"],
        "cpu_p90_ms": cpu_bench["p90_ms"],
        "cpu_chunks_per_s": cpu_bench["chunks_per_s"],
        "active_device_median_ms": None if active_bench is None else active_bench["median_ms"],
        "history": history,
    }
    with open(
        ARTIFACT_DIR / f"distill_{student_name}_from_{teacher_name}.json", "w", encoding="utf-8"
    ) as fh:
        json.dump(result, fh, indent=2)
    return result


# --- CLI ---
def main():
    parser = argparse.ArgumentParser(description="Distill density NN student from teacher.")
    parser.add_argument("--teacher", type=str, required=True, help="Teacher model name (e.g. unet)")
    parser.add_argument("--student", type=str, required=True, help="Student model name (e.g. sep)")
    parser.add_argument("--epochs", type=int, default=120, help="Distillation epochs")
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Distillation alpha (ground truth vs teacher)"
    )
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    args = parser.parse_args()
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    print(f"Distilling {args.student} from {args.teacher} on {device}")
    result = distill_student(args.teacher, args.student, args.epochs, args.alpha, args.lr, device)
    print("Distillation complete. Results:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
