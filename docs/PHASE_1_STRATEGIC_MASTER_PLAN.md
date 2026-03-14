# Phase 1 Strategic Master Plan - Progressive Neural Terrain Generation

Complete execution strategy for training three parallel feature networks that will form the foundation of the staged Minecraft terrain generator.

---

## 🚨 PREREQUISITE: Parity Verification (CRITICAL)

**Before launching any training jobs, you MUST verify that your Python NoiseRouter emulator matches the Minecraft Java implementation.**

**Why?** If there's even a small drift in how you implement Perlin noise or seed handling, your neural networks will:
- Converge perfectly to a "Minecraft dialect" that doesn't exist in the actual game
- Perform well on your Python emulator but fail completely in-game

**Execution (30 minutes):**
1. Generate reference noise file: `noise_dump.csv` from Minecraft Java source (see PHASE_1_PARITY_VERIFICATION.md)
2. Run parity check: `python PHASE_1_PARITY_CHECK.py --reference noise_dump.csv`
3. Verify all samples match to **6+ decimal places** (`|Java - Python| < 1e-6`)

**Success Criteria:**
- ✅ All 1000+ test samples converge within tolerance
- ✅ Max error < 1e-5 for all DensityFunctions
- ✅ Reproducible across 3+ different world seeds
- ✅ Report saved to `docs/PHASE_1_PARITY_REPORT.md`

**IF PARITY CHECK FAILS:** Do NOT proceed to training. Debug the specific DensityFunction (continents, erosion, etc.) and retry.

**References:**
- Full guide: [PHASE_1_PARITY_VERIFICATION.md](PHASE_1_PARITY_VERIFICATION.md)
- Java reference: `reference-code/26.1-snapshot-11/net/minecraft/world/level/levelgen/`

---

## Executive Summary

**Goal:** Train three shallow, parallel neural networks to learn the three fundamental sub-systems of Minecraft's `NoiseRouter`:

| Network | Target | Training Time | Grok Metric | Dependency on Others |
|---------|--------|---|---|---|
| **Phase 1A** | Continents, Erosion, Ridges | 3-30 hours | MSE < 0.001 | **NONE** |
| **Phase 1B** | Temperature, Vegetation, Depth | 1-5 hours | Biome accuracy > 95% | **NONE** |
| **Phase 1C** | Cave/Aquifer probability | 4-40 hours | IoU > 0.75 | **NONE** |

**Key Insight:** These three networks have ZERO data dependencies. Train them simultaneously on separate hardware to save wall-clock time from 40+ hours to ~5-10 hours.

---

## Phase 1A: Macro-Shape Network (Terrain Shaping)

### What It Learns
Maps Perlin noise octaves → the three primary density functions that define terrain macro-features.

```
Input: Perlin octaves at scales 43, 52, 20 blocks
        + XYZ cell coordinates
Output: continents, erosion, ridges density values
```

### Network Architecture

```
Input Layer: (batch, 4, 48, 4, 11)  [cell grid + octaves]
    ↓
Conv3D(11 → 32, kernel=3x3x3, padding=1)
ReLU activation
    ↓
Conv3D(32 → 64, kernel=3x3x3, padding=1)
ReLU activation
    ↓
Conv3D(64 → 3, kernel=3x3x3, padding=1)   [3 outputs: continents, erosion, ridges]
Output: (batch, 4, 48, 4, 3)
```

**Why shallow:** The mapping from Perlin octaves to density functions is nearly deterministic (almost piecewise linear). A shallow network learns it in hours, not days.

### Training Configuration

```python
# Hyperparameters
learning_rate = 1e-3
batch_size = 128
epochs = 20-50 (until convergence, early stopping at MSE < 0.001)
loss_function = MSE
optimizer = Adam
scheduler = ReduceLROnPlateau(factor=0.5, patience=3)

# Data generation
num_chunks = 1000-2000
samples_per_chunk = 768 cells
total_samples = 768,000 - 1,536,000

# Validation
val_split = 0.2
grok_threshold = MSE < 0.001

# REFINEMENTS (from PHASE_1_REFINEMENTS.md):
# - Positional encoding: ✅ Enabled (n_freqs=8, 16 channels per coordinate)
# - Float16: ✅ Enabled (50% memory savings on GPU, required for CPU)
# - Early stopping: ✅ Enabled (patience=5, stop if MSE < 0.001 for 5 epochs)
```

### Hardware Requirements

```
GPU (recommended): 
  - 4 GB VRAM for batch_size=128
  - ~2-3 hours training
  
CPU (fallback):
  - ~30 hours training
  - Any processor works
  
Disk:
  - phase_1a_data.npz: ~150 MB (compressed)
  - Checkpoint files: ~30 MB
```

### Grok Validation Metrics

```python
def validate_phase_1a():
    MSE = compute_mse(predictions, ground_truth)
    MAE = compute_mae(predictions, ground_truth)
    R_squared = 1 - (MSE / variance(ground_truth))
    
    # Grok thresholds
    assert MSE < 0.001, f"MSE {MSE} exceeds threshold 0.001"
    assert MAE < 0.05,  f"MAE {MAE} exceeds threshold 0.05"
    assert R_squared > 0.99, f"R² {R_squared} too low (expect >0.99)"
    
    print("✓ Phase 1A GROKKED")
```

**Interpretation:**
- MSE < 0.001: Model outputs match ground truth to within ±0.03 on [-1, 1] scale
- R² > 0.99: Explains >99% of variance in density functions
- **Once grokked:** Lock weights, freeze layers → becomes hidden feature extractor for Phase 2

---

## Phase 1B: Climate & Biome Network

### What It Learns
Maps climate noise → temperature, vegetation, depth factors that determine biome assignment.

```
Input: Perlin octaves (temp, veg) + reused continents from 1A
        + XZ cell coordinates only (Y-reduced)
Output: temperature, vegetation, depth, biome_confidence
```

### Network Architecture

```
Input Layer: (batch, 4, 4, 4, 8)  [4 Y-levels instead of 48]
    ↓
Conv3D(8 → 32, kernel=3x3x3)
ReLU
    ↓
Conv3D(32 → 64, kernel=3x3x3)
ReLU
    ↓
Conv3D(64 → 4, kernel=3x3x3)   [4 outputs: temp, veg, depth, confidence]
Output: (batch, 4, 4, 4, 4)
```

**Key difference from 1A:** Only 4 Y-levels sampled (climate varies mostly horizontally). ~12x fewer parameters.

### Training Configuration

```python
learning_rate = 1e-3
batch_size = 256  # Can be larger due to smaller Y dimension
epochs = 10-20
loss_function = MSE + CrossEntropy (biome classification)
optimizer = Adam
```

### Grok Validation Metrics

```python
def validate_phase_1b():
    # Regression metrics (for temperature, vegetation)
    mse_climate = mse_loss(pred_temp_veg, true_temp_veg)
    
    # Classification metrics (biome assignment)
    # Infer biome from climate using known biome parameter centers
    pred_biome = classify_from_climate(pred_temp_veg)
    true_biome = classify_from_climate(true_temp_veg)
    biome_accuracy = (pred_biome == true_biome).mean()
    
    assert mse_climate < 0.005
    assert biome_accuracy > 0.95   # 95%+ biome placement accuracy
    
    print(f"✓ Phase 1B GROKKED: {biome_accuracy*100:.1f}% biome accuracy")
```

---

## Phase 1C: Subtractive Network (Aquifers & Carvers)

### What It Learns
Maps cave noise + depth → 3D probability of caves/aquifers (regions of "emptiness").

```
Input: Perlin octaves (cave_large, cave_detail, aquifer)
        + depth gradient (Y coordinate)
        + full 3D coordinates
Output: air_probability, water_probability, cave_uncertainty, carver_influence
```

### Network Architecture

```
Input Layer: (batch, 4, 48, 4, 13)  [full 3D, more channels]
    ↓
Conv3D(13 → 64, kernel=3x3x3)
ReLU
    ↓
Conv3D(64 → 128, kernel=3x3x3)
ReLU
    ↓
Conv3D(128 → 4, kernel=3x3x3)   [4 probability outputs]
Sigmoid activation (convert to [0, 1])
Output: (batch, 4, 48, 4, 4)
```

**Key difference from 1A/1B:** 
- Outputs are probabilities (sigmoid, not linear)
- Full 3D spatial resolution (not reduced Y)
- Slightly deeper network (more parameters for 3D pattern learning)

### Training Configuration

```python
learning_rate = 5e-4  # Slightly smaller (probability fine-tuning)
batch_size = 64  # Smaller due to 3D complexity
epochs = 30-50
# REFINEMENT: Weighted loss for sparse cave patterns
# See PHASE_1_REFINEMENTS.md for details
loss_function = CombinedLoss(alpha=0.5, beta=0.5)  # BCE + Dice
  # - Weighted BCE: pos_weight=9.0 (caves are 10% of volume)
  # - Dice Loss: directly optimizes IoU (your grok metric)
optimizer = Adam with scheduler (reduce_lr_on_plateau)
early_stopping = EarlyStoppingCallback(metric='iou', threshold=0.75, patience=5)
```

### Grok Validation Metrics

```python
def validate_phase_1c():
    predictions = model(inputs).sigmoid()  # Convert to probabilities
    
    # Primary metric: IoU for cave volume
    pred_caves = (predictions[:, :, :, :, 0] > 0.5).float()
    true_caves = (outputs[:, :, :, :, 0] > 0.5).float()
    
    intersection = (pred_caves * true_caves).sum()
    union = (pred_caves + true_caves - pred_caves * true_caves).sum()
    iou = intersection / (union + 1e-6)
    
    # Dice loss (F1-like metric for segmentation)
    dice = 2 * intersection / (pred_caves.sum() + true_caves.sum() + 1e-6)
    
    assert iou > 0.75, f"IoU {iou:.3f} below threshold 0.75"
    assert dice > 0.80, f"Dice {dice:.3f} below threshold 0.80"
    
    print(f"✓ Phase 1C GROKKED: IoU {iou:.3f}, Dice {dice:.3f}")
```

---

## Parallel Training Strategy

### Timeline: Wall-Clock Execution (With Early Stopping Optimization)

```
TIME
 │
 │  GPU#1          GPU#2          GPU#3          GPU#4
 │  Phase 1A       Phase 1B        Phase 1C       (spare)
 │  ┌─────────┐    ┌────────┐     ┌──────────┐
 │  │         │    │        │     │          │
 0h ├─ START  │    ├─ START │     ├─ START   │
 │  │         │    │        │     │          │
 2h │         │    │ DONE ✓ │     │          │  ← Phase 1B done (early stopping @ R²>0.99)
 │  │         │    └────────┘     │          │
 │  │         │                    │          │
 4h │         │                    │ DONE ✓   │  ← Phase 1C done (early stopping @ IoU>0.75)
 │  │         │                    └──────────┘
 │  │         │
 6h │ DONE ✓  │                                   ← Phase 1A done (early stopping @ MSE<0.001)
 │  └─────────┘
 │
```

**Expected Timing (with Early Stopping from PHASE_1_REFINEMENTS.md):**

| Phase | Without Early Stopping | With Early Stopping | Speedup |
|---|---|---|---|
| Phase 1A | 30 hours | **20 hours** | 1.5x |
| Phase 1B | 5 hours | **2 hours** | 2.5x |
| Phase 1C | 40 hours | **25 hours** | 1.6x |
| **Parallel** | 40 hours | **25 hours** | 1.6x |
| **Sequential (CPU)** | 75 hours | **47 hours** | 1.6x |

**With Float16 + Positional Encoding:** Additional 15-20% speedup from faster convergence.

---

## Execution Plan (Updated with Refinements)

1. **Hour 0:** Start data extraction for all three phases simultaneously
   ```bash
   python extract_phase_1.py --phase 1a --num-chunks 1000 &
   python extract_phase_1.py --phase 1b --num-chunks 1000 &
   python extract_phase_1.py --phase 1c --num-chunks 1000 &
   wait  # All complete in ~30 minutes on multi-core CPU
   ```

2. **Hour 0.5:** Launch three training jobs (one per GPU)
   ```bash
   # Terminal 1
   python train_phase_1a.py --data phase_1a_data.npz --gpu 0 &
   
   # Terminal 2
   python train_phase_1b.py --data phase_1b_data.npz --gpu 1 &
   
   # Terminal 3
   python train_phase_1c.py --data phase_1c_data.npz --gpu 2 &
   ```

3. **Hour 1-3:** Phase 1B converges first (smallest dataset, simplest mapping)
   ```
   Monitor: python monitor_training.py
   Output: "Phase 1B converged with 96.2% biome accuracy"
   Action: Freeze Phase 1B weights, save checkpoint
   ```

4. **Hour 5:** Phase 1C converges (medium dataset, 3D complexity)
   ```
   Output: "Phase 1C converged with IoU 0.782"
   Action: Freeze Phase 1C weights, save checkpoint
   ```

5. **Hour 10:** Phase 1A converges (largest dataset, deterministic mapping takes longer to fully optimize)
   ```
   Output: "Phase 1A converged with MSE 0.0007"
   Action: Freeze Phase 1A weights, save checkpoint
   ```

6. **Hour 10.5:** Combine into Phase 2
   ```python
   # Load all three frozen networks
   model_1a = load_checkpoint("phase_1a_best.pt")
   model_1b = load_checkpoint("phase_1b_best.pt")
   model_1c = load_checkpoint("phase_1c_best.pt")
   
   # Freeze their weights
   model_1a.eval()
   model_1b.eval()
   model_1c.eval()
   for param in model_1a.parameters(): param.requires_grad = False
   for param in model_1b.parameters(): param.requires_grad = False
   for param in model_1c.parameters(): param.requires_grad = False
   
   # Begin Phase 2: finalDensity combiner training
   ```

---

## Data Extraction Strategy

### Step 1: Generate Perlin Octaves (Pre-compute Once)

```python
# This runs ONCE for all three phases
# Takes ~30 minutes on modern CPU with 8 cores

extractor = NoiseRouterDataExtractor(
    world_seed=12345,
    num_chunks=1000,
    chunk_region_size=32  # 32x32 chunk grid = 1024 unique chunks
)

# Multi-threaded extraction
import concurrent.futures
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    # No inter-phase dependencies - all extract in parallel
    futures = [
        executor.submit(extractor.extract_phase_1a_data),
        executor.submit(extractor.extract_phase_1b_data),
        executor.submit(extractor.extract_phase_1c_data),
    ]
    concurrent.futures.wait(futures)

# Output files (ready for training):
# - phase_1a_data.npz (150 MB)
# - phase_1b_data.npz (50 MB)
# - phase_1c_data.npz (200 MB)
```

### Step 2: Verify Data Integrity

```python
def verify_data_shapes():
    """Ensure tensors match specifications"""
    import numpy as np
    
    data_1a = np.load("phase_1a_data.npz")
    assert data_1a['inputs'].shape == (1000, 4, 48, 4, 11)
    assert data_1a['outputs'].shape == (1000, 4, 48, 4, 3)
    
    data_1b = np.load("phase_1b_data.npz")
    assert data_1b['inputs'].shape == (1000, 4, 4, 4, 8)
    assert data_1b['outputs'].shape == (1000, 4, 4, 4, 4)
    
    data_1c = np.load("phase_1c_data.npz")
    assert data_1c['inputs'].shape == (1000, 4, 48, 4, 13)
    assert data_1c['outputs'].shape == (1000, 4, 48, 4, 4)
    
    print("✓ All data shapes validated")
```

---

## Failure Cases & Recovery

### If Phase 1A MSE > 0.001 After 50 Epochs

**Diagnosis:** Network can't fit the data. Possible causes:
- Bug in Perlin octave computation
- Incorrect scaling/normalization of inputs
- Network too shallow for this particular output function

**Recovery:**
1. Increase network depth (add 1-2 more Conv3D layers)
2. Verify ground truth by comparing with known Minecraft chunk samples
3. Check input octave ranges (should be roughly [-1, 1])
4. Try different learning rate schedule

### If Phase 1B Biome Accuracy < 90%

**Diagnosis:** Climate parameter mapping is harder than expected. Possible causes:
- Biome classification formula more complex than estimated
- Minecraft uses non-linear climate combination

**Recovery:**
1. Add more training data (extract 2000-5000 chunks instead of 1000)
2. Add classification head (separate network branch for biome clustering)
3. Use ensemble voting across multiple random seeds

### If Phase 1C IoU < 0.70

**Diagnosis:** 3D cave patterns don't match between our Perlin and actual Minecraft. Possible causes:
- Minecraft carver logic more complex (uses gradient/threshold on density)
- Aquifer interaction with surface cutoff

**Recovery:**
1. Increase network capacity (deeper, wider)
2. Add auxiliary losses (binary classification for air vs stone)
3. Use different loss weights for different regions (caves vs aquifers separately)
4. Pre-train on simpler 2D cave patterns before 3D

---

## Integration with Phase 2

**At the end of Phase 1:** You have three frozen feature extractors.

```python
class Phase1Backbone(nn.Module):
    """Frozen Phase 1 networks act as feature extractors"""
    
    def __init__(self):
        super().__init__()
        self.macro_shape = load_frozen_model("phase_1a_best.pt")
        self.climate = load_frozen_model("phase_1b_best.pt")
        self.subtractive = load_frozen_model("phase_1c_best.pt")
    
    def forward(self, 
                inputs_1a, 
                inputs_1b, 
                inputs_1c):
        """
        Extract frozen features from all three networks
        """
        features_macro = self.macro_shape(inputs_1a)      # (B, 4, 48, 4, 3)
        features_climate = self.climate(inputs_1b)       # (B, 4, 4, 4, 4)
        features_subtractive = self.subtractive(inputs_1c) # (B, 4, 48, 4, 4)
        
        # These become inputs to Phase 2
        return features_macro, features_climate, features_subtractive
```

**Phase 2:** Train the `finalDensity` combiner on top of these frozen features. Much faster because:
- Parameters reduced from 500K+ to ~50K
- Convergence in 2-5 hours instead of 10-40 hours
- Lower risk of overfitting

---

## Checkpoints & Reproducibility

Save checkpoints at regular intervals:

```python
# Every 5 epochs
torch.save({
    'epoch': epoch,
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'loss': loss_history,
    'world_seed': world_seed,
    'hyperparams': {
        'lr': learning_rate,
        'batch_size': batch_size,
        'architecture': model_architecture
    }
}, f"phase_1a_epoch_{epoch}.pt")

# Keep only last 5 checkpoints to save disk space
# When fully converged, save "phase_1a_best.pt"
```

---

## Success Criteria

| Milestone | Criterion | Expected Time |
|-----------|-----------|---|
| **Data Generated** | All three .npz files exist with correct shapes | 0.5h |
| **Phase 1A Trains** | Loss decreasing, no NaN/Inf | 1h |
| **Phase 1B Trains** | Biome accuracy improving | 0.5h |
| **Phase 1C Trains** | IoU improving | 1h |
| **Phase 1A Grokked** | MSE < 0.001 ✓ | 3-10h |
| **Phase 1B Grokked** | Biome accuracy > 95% ✓ | 1-3h |
| **Phase 1C Grokked** | IoU > 0.75 ✓ | 4-10h |
| **All Frozen** | Weights locked, ready for Phase 2 | 10.5h |

---

## Next Immediate Actions

```bash
# 1. Prepare environment
cd /path/to/MC/docs
pip install torch numpy scipy


---

## ✅ BEFORE LAUNCH CHECKLIST

**This is NOT optional.** Missing even one step can cause training failure or poor convergence.

### Phase 0: Parity Verification (CRITICAL - 30 minutes)

- [ ] Read [PHASE_1_PARITY_VERIFICATION.md](PHASE_1_PARITY_VERIFICATION.md)
- [ ] Generate reference noise file (Option A: Java export, Option B: pre-computed JSON)
  - [ ] File exists: `reference-code/noise_dump.csv` or `noise_reference_vectors.json`
- [ ] Create `PHASE_1_PARITY_CHECK.py` from verification guide
- [ ] Run parity check: `python PHASE_1_PARITY_CHECK.py --seed 12345`
- [ ] **VERIFY ALL SAMPLES PASS** (error < 1e-6)
- [ ] Save report: `docs/PHASE_1_PARITY_REPORT.md`
- [ ] Test with 3+ different world seeds (12345, -1, 999999)

**STOP HERE IF PARITY CHECK FAILS** ← Do not proceed to Phase 1A/1B/1C

### Phase 0B: Refinements Integration (1 hour)

These are from [PHASE_1_REFINEMENTS.md](PHASE_1_REFINEMENTS.md) and are **strongly recommended**:

**Refinement 1: Float16 + Positional Encoding**
- [ ] Update Phase 1A network class with `PositionalEncoder`
- [ ] Enable `use_float16=True` in network initialization
- [ ] Verify memory usage dropped by ~50%
- [ ] Test training 5 epochs to confirm no gradients are NaN

**Refinement 2: Phase 1C Weighted Loss**
- [ ] Implement `WeightedBCELoss` and `DiceLoss` classes
- [ ] Update Phase 1C loss function to `CombinedLoss(alpha=0.5, beta=0.5, pos_weight=9.0)`
- [ ] Verify IoU metric is being computed correctly

**Refinement 3: Early Stopping**
- [ ] Implement `EarlyStoppingCallback` class
- [ ] Configure for each phase:
    - [ ] Phase 1A: metric='mse', threshold=0.001, patience=5
    - [ ] Phase 1B: metric='r2', threshold=0.99, patience=5
    - [ ] Phase 1C: metric='iou', threshold=0.75, patience=5

### Phase 1A: Macro-Shape Network

- [ ] Data extracted: `phase_1a_data.npz` exists (~150 MB)
- [ ] Data shape verified: inputs (1000, 4, 48, 4, 11), outputs (1000, 4, 48, 4, 3)
- [ ] Network class created: `Phase1ANetworkWithEncoding`
- [ ] Hyperparameters set: lr=1e-3, batch_size=128, epochs=50 (early stop @ MSE<0.001)
- [ ] GPU assigned: GPU#0 (or CPU fallback)
- [ ] Training launched: `python train_phase_1a.py --gpu 0 > phase_1a.log &`
- [ ] Monitoring active: Watch `phase_1a.log` for convergence
- [ ] **Grok Threshold Met:** MSE < 0.001, R² > 0.99
- [ ] Checkpoint saved: `checkpoints/phase_1a_best.pt`
- [ ] Weights frozen: `model_1a.eval()`, `param.requires_grad = False`

### Phase 1B: Climate & Biome Network

- [ ] Data extracted: `phase_1b_data.npz` exists (~50 MB)
- [ ] Network class created: `Phase1BNetworkWithEncoding`
- [ ] Hyperparameters set: lr=1e-3, batch_size=256, epochs=20
- [ ] GPU assigned: GPU#1 (or CPU fallback)
- [ ] Training launched: `python train_phase_1b.py --gpu 1 > phase_1b.log &`
- [ ] **Grok Threshold Met:** Biome accuracy > 95%, R² > 0.99
- [ ] Checkpoint saved: `checkpoints/phase_1b_best.pt`
- [ ] Weights frozen

### Phase 1C: Subtractive Network (Caves & Aquifers)

- [ ] Data extracted: `phase_1c_data.npz` exists (~200 MB)
- [ ] Network class created: `Phase1CNetworkWithEncoding` with weighted loss
- [ ] Loss function: `CombinedLoss(alpha=0.5, beta=0.5)` (Dice + Weighted BCE)
- [ ] Hyperparameters set: lr=5e-4, batch_size=64, epochs=50
- [ ] GPU assigned: GPU#2 (or CPU fallback)
- [ ] Training launched: `python train_phase_1c.py --gpu 2 > phase_1c.log &`
- [ ] **Grok Threshold Met:** IoU > 0.75, Dice > 0.80
- [ ] Checkpoint saved: `checkpoints/phase_1c_best.pt`
- [ ] Weights frozen

### Post-Training Validation

- [ ] All three checkpoints exist and load without error
- [ ] Inference time < 10ms per 768-cell batch (sanity check)
- [ ] Generate validation plot: loss curves, grok metrics
- [ ] Document final metrics in `docs/PHASE_1_RESULTS.md`

---

## Timeline for This Checklist

| Step | Estimated Time | Critical Path |
|---|---|---|
| Parity Verification | 30 min | ✅ BLOCKING |
| Refinements Integration | 60 min | ✅ STRONG |
| Data Generation | 30 min | Sequential |
| Phase 1B Training | 2 hours | Parallel |
| Phase 1C Training | 5 hours | Parallel |
| Phase 1A Training | 20 hours | Parallel |
| **Total (Parallel)** | **~25 hours** | Assuming 3 GPUs |

**CPU-Only Timeline:** ~47 hours (sequential training)

---

## Next Immediate Actions

```bash
# 0. FIRST: Parity verification (CRITICAL - BLOCKING)
cd /path/to/MC/docs
python PHASE_1_PARITY_CHECK.py --reference reference-code/noise_dump.csv
# Wait for ✓ SUCCESS message before proceeding

# 1. Prepare environment
pip install torch numpy scipy pytorch-lightning

# 2. Integrate refinements from PHASE_1_REFINEMENTS.md
# Create/update these files with the classes from PHASE_1_REFINEMENTS.md:
#   - train_phase_1a.py (with PositionalEncoder, float16, early stopping)
#   - train_phase_1b.py
#   - train_phase_1c.py (with WeightedBCELoss + DiceLoss)

# 3. Extract ground truth (one time, ~30 minutes)
python PHASE_1_DATA_EXTRACTION.py --num-chunks 1000

# 4. Launch training (three separate terminals for GPU#0, #1, #2)
# Terminal 1:
python train_phase_1a.py --data phase_1a_data.npz --gpu 0 | tee phase_1a.log &

# Terminal 2:
python train_phase_1b.py --data phase_1b_data.npz --gpu 1 | tee phase_1b.log &

# Terminal 3:
python train_phase_1c.py --data phase_1c_data.npz --gpu 2 | tee phase_1c.log &

# 5. Monitor progress (in separate terminal)
watch -n 60 'tail -20 phase_1a.log phase_1b.log phase_1c.log'

# 6. When all three reach grok thresholds (~6 hours with GPU):
# Phase 2 is ready to begin
```

---

## References

- **Parity Verification:** [PHASE_1_PARITY_VERIFICATION.md](PHASE_1_PARITY_VERIFICATION.md) ← **START HERE**
- **Refinements & Optimizations:** [PHASE_1_REFINEMENTS.md](PHASE_1_REFINEMENTS.md) (Float16, Positional Encoding, Weighted Loss, Early Stopping)
- **Data Extraction Code:** [PHASE_1_DATA_EXTRACTION.md](PHASE_1_DATA_EXTRACTION.md)
- **Tensor Specifications:** [PHASE_1_TENSOR_ARCHITECTURE.md](PHASE_1_TENSOR_ARCHITECTURE.md)
- **Implementation Quick Start:** [PHASE_1_IMPLEMENTATION_INDEX.md](PHASE_1_IMPLEMENTATION_INDEX.md)
- **Original Analysis:** [MINECRAFT_TERRAIN_GENERATION_ANALYSIS.md](MINECRAFT_TERRAIN_GENERATION_ANALYSIS.md)


