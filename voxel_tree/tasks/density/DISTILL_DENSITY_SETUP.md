# Distill Density NN - Pipeline Integration

## Overview
The density NN distillation workflow has been fully integrated into the terrain_shaper_density pipeline with GUI support, progress callbacks, and DAG registration.

## Components

### 1. Core Distillation Module
**File:** `VoxelTree/core/distill_density_nn.py`
- **Function:** `distill_student(teacher_name, student_name, epochs, alpha, lr, device, progress_callback=None)`
- **Purpose:** Core distillation logic that trains a fast student model from a slower teacher model
- **Progress Callback:** Invoked with `(epoch, total_epochs, metrics_dict)` at each epoch
- **Output:** Saves best checkpoint to `VoxelTree/notebooks/experimental/artifacts/density_nn_shootout/`

### 2. Pipeline Node
**File:** `VoxelTree/core/nodes/distill_density.py`
- **Class:** `DistillDensityNode`
- **Purpose:** Wraps distillation for pipeline DAG integration
- **Methods:**
  - `run(teacher_name, student_name, epochs, alpha, lr, device)` → Returns result dict
  - `set_progress_callback(callback)` → Connects to GUI progress bar
  - `get_parameters()` → Returns configurable parameters for GUI

### 3. Pipeline Configuration
**File:** `VoxelTree/profiles/terrain_shaper_density.yaml`
- **DAG Step:** `distill_density`
- **Prerequisites:** `train_terrain_shaper_density` (will run after teacher training completes)
- **Default Hyperparameters:** Can be overridden in YAML or GUI

## Usage

### Via Pipeline GUI
1. Open the VoxelTree pipeline UI
2. Navigate to the terrain_shaper_density pipeline
3. You should now see the **"Distill Density NN"** node in the DAG
4. Configure parameters:
   - **Teacher Model:** unet (default) | mlp | sep | axial
   - **Student Model:** sep (default) | mlp | axial | unet
   - **Epochs:** 120 (default)
   - **Alpha:** 0.5 (default) — weight on ground truth (0=teacher-only, 1=gt-only)
   - **Learning Rate:** 2e-3 (default)
   - **Device:** cuda (default) | cpu
5. Execute node — circular progress bar will show training progress
6. Checkpoint saved automatically upon completion

### Via CLI
```bash
python -m VoxelTree.scripts.terrain_shaper.distill_density \
  --teacher unet \
  --student sep \
  --epochs 120 \
  --alpha 0.5 \
  --lr 0.002 \
  --device cuda
```

### Programmatic (Python)
```python
from VoxelTree.scripts.terrain_shaper.distill_density import distill_student

result = distill_student(
    teacher_name='unet',
    student_name='sep',
    epochs=120,
    alpha=0.5,
    lr=2e-3,
    device='cuda',
    progress_callback=my_callback  # Optional
)
```

## Callback Interface

### GUI Progress Callback
The GUI passes a callback that accepts:
```python
def gui_callback(progress: float, message: str):
    # progress: 0.0 to 1.0
    # message: e.g., "Epoch 5/120 | val_mse=0.00123"
```

### Progress Callback from Core
The distill_student function invokes:
```python
progress_callback(epoch, total_epochs, {
    'epoch': epoch,
    'train_loss': float,
    'train_mse': float,
    'train_sign_acc': float,
    'val_mse': float,
    'val_mae': float,
    'val_sign_acc': float,
    'val_boundary_sign_acc': float,
})
```

## Output

### Checkpoints
- **Location:** `VoxelTree/notebooks/experimental/artifacts/density_nn_shootout/`
- **Filename:** `distill_{student_name}_from_{teacher_name}.pt`
- **Contents:** Best model state dict (lowest validation MSE)

### Results JSON
- **Filename:** `distill_{student_name}_from_{teacher_name}.json`
- **Contents:**
  - Training/validation metrics over epochs (history)
  - Best validation performance (MSE, sign accuracy, boundary accuracy)
  - Test set performance
  - Latency benchmarks (CPU, active device)
  - Hyperparameters used

## Architecture Support

All four candidate architectures supported as teacher and student:
1. **MLP** — Simple 3-layer dense network, fastest
2. **SEP** — Separable convolutions, good speed/accuracy tradeoff
3. **AXIAL** — Axial attention, slower but accurate
4. **UNET** — U-Net style encoder-decoder, highest accuracy (default teacher)

## Integration Points

### Prerequisites
- Depends on `train_terrain_shaper_density` (teacher checkpoint required)

### Downstream
- Checkpoint can be used for:
  - Export to ONNX for inference
  - Fine-tuning on custom data
  - Deployment in LODiffusion mod

## Troubleshooting

### Node Not Visible in Pipeline UI
- ✓ Already fixed: Added `distill_density` to DAG in `terrain_shaper_density.yaml`
- Refresh/restart the pipeline UI if needed

### Callback Not Updating Progress Bar
- Ensure GUI passes `set_progress_callback(callback)` before calling `run()`
- Callback signature: `callback(progress: float, message: str)`

### CUDA Out of Memory
- Reduce batch size (in `distill_density_nn.py`, `BATCH_SIZE = 8`)
- Use `--device cpu` for testing
- Try smaller student architecture (MLP instead of UNET)

### Checkpoint Not Found
- Ensure `train_terrain_shaper_density` completed successfully first
- Teacher checkpoint must be in `artifacts/density_nn_shootout/`

## Next Steps

1. **Verify Node Visibility** — Refresh pipeline UI, confirm distill_density appears
2. **Test Run** — Execute with default parameters (teacher=unet, student=sep)
3. **Monitor Progress** — Watch circular progress bar and val_mse metrics
4. **Save Results** — Checkpoint and metrics JSON generated automatically
5. **Deploy or Fine-tune** — Use checkpoint downstream as needed
