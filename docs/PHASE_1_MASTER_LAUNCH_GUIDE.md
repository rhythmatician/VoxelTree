# Phase 1 Master Launch Guide - Complete Operational Procedure

**Status:** Ready to execute Phase 1 training with all critical refinements integrated.

**Prepared:** March 13, 2026

---

## 🚀 Quick Start (TL;DR)

If you're experienced with PyTorch and want to jump straight in:

```bash
# 1. CRITICAL: Parity check (30 min, BLOCKS everything)
python PHASE_1_PARITY_CHECK.py --reference reference-code/noise_dump.csv

# 2. Extract data (30 min)
python PHASE_1_DATA_EXTRACTION.py --num-chunks 1000

# 3. Launch training (background)
python train_phase_1a.py --gpu 0 &
python train_phase_1b.py --gpu 1 &
python train_phase_1c.py --gpu 2 &

# 4. Wait (~6 hours on 3x GPU, ~25 hours on CPU)
watch 'tail -5 phase_1*.log'

# Success indicators:
# - phase_1a.log: "MSE < 0.001 ✓"
# - phase_1b.log: "Biome accuracy > 95% ✓"
# - phase_1c.log: "IoU > 0.75 ✓"
```

---

## Full Procedure (Step-by-Step)

### PRE-LAUNCH PHASE (1-2 hours)

#### Step 1: Verify Parity ⚠️ CRITICAL

This is the **single most important step**. If skipped or failed, everything downstream will be invalid.

**Duration:** 30 minutes  
**Why:** Your Python NoiseRouter emulator must match Minecraft's Java implementation to ±1e-6 decimal places

**Action:**
```bash
# 1a. Generate reference (choose ONE option)

# Option A: Extract from Minecraft Java (recommended if you have dev environment)
# Command: /dump_noise
# Outputs: noise_dump.csv (~1 MB, 1000 samples)
# Location: c:\Users\JeffHall\git\MC\reference-code\noise_dump.csv

# Option B: Use pre-computed reference (fallback)
# File already available at: reference-code/noise_reference_vectors.json
# If not available, skip to Step 3 and use test-mode

# 1b. Run parity check
cd c:\Users\JeffHall\git\MC
python docs/PHASE_1_PARITY_CHECK.py \
    --reference reference-code/noise_dump.csv \
    --seed 12345 \
    --tolerance 1e-6

# Expected output:
# ======================================================================
# PARITY VERIFICATION REPORT
# ======================================================================
# Samples Tested:  1000
# Passed:          1000 ✓
# Failed:          0 ✗
# 
# 🎉 SUCCESS: All samples match Java implementation!
# Max Error:       continents = 1.5e-10
# Mean Error:      erosion = 2.3e-11
# ======================================================================

# 1c. Save report
cp docs/PHASE_1_PARITY_CHECK.log docs/PHASE_1_PARITY_REPORT.md
```

**If parity check FAILS:**
```
❌ STOP HERE. Do not proceed with training.

Common failure modes:
- "Δ > 0.01 on all functions" → Check seed derivation (f_seed hash algorithm)
- "Δ > 1e-4 on erosion only" → Verify octave scales in OctavedNoiseFunction
- "Memory error" → Reduce num_samples in reference file

Fix the specific DensityFunction and retry parity check before moving forward.
```

**If parity check PASSES:**
```
✅ Proceed to Step 2
```

---

#### Step 2: Integrate Refinements

**Duration:** 60 minutes  
**Why:** These optimizations are critical for convergence speed and hardware compatibility

**Create/update three files with classes from [PHASE_1_REFINEMENTS.md](PHASE_1_REFINEMENTS.md):**

**File 1: `refinements.py`**
```python
# Copy these class definitions from PHASE_1_REFINEMENTS.md:
class PositionalEncoder(nn.Module):
    """Periodic feature encoding for coordinates"""
    
class WeightedBCELoss(nn.Module):
    """BCE weighted for sparse targets (caves)"""
    
class DiceLoss(nn.Module):
    """Dice loss (F1-like) for IoU optimization"""
    
class CombinedLoss(nn.Module):
    """Weighted BCE + Dice for Phase 1C"""
    
class EarlyStoppingCallback:
    """Stops training at grok thresholds"""
```

**File 2: `networks_refined.py`**
```python
# Copy these class definitions from PHASE_1_REFINEMENTS.md:
class Phase1ANetworkWithEncoding(nn.Module):
    """Positional encoding + Conv3D + Float16 support"""
    
class Phase1BNetworkWithEncoding(nn.Module):
    """Climate network with reduced Y dimension"""
    
class Phase1CNetworkWithEncoding(nn.Module):
    """3D cave network with full spatial resolution"""
```

**Verification checklist:**
- [ ] `refinements.py` imports without error: `python -c "from refinements import *"`
- [ ] `networks_refined.py` imports without error: `python -c "from networks_refined import *"`
- [ ] Test instantiation: `Phase1ANetworkWithEncoding(use_float16=True)` runs
- [ ] No CUDA errors if GPU unavailable

---

#### Step 3: Prepare Data Directory

```bash
# Create checkpoint directory
mkdir -p c:\Users\JeffHall\git\MC\checkpoints\phase_1a
mkdir -p c:\Users\JeffHall\git\MC\checkpoints\phase_1b
mkdir -p c:\Users\JeffHall\git\MC\checkpoints\phase_1c

# Verify extraction code exists
test -f c:\Users\JeffHall\git\MC\docs\PHASE_1_DATA_EXTRACTION.md && echo "✓ Extraction guide ready"
```

---

### EXECUTION PHASE (30 minutes setup + 6-25 hours training)

#### Step 4: Extract Ground Truth Data

**Duration:** 30 minutes (one-time operation)  
**Output:** Three `.npz` files (~400 MB total, reusable)

**Action:**
```bash
# Create extraction script from PHASE_1_DATA_EXTRACTION.md
# Save as: phase_1_extract.py

# Run extraction (multi-threaded, CPU-bound)
python phase_1_extract.py --num-chunks 1000 --output-dir ./phase_1_data

# Wait for completion. Output:
# ✓ Extracted 1000,000 samples for Phase 1A → phase_1_data/phase_1a_data.npz (150 MB)
# ✓ Extracted 250,000 samples for Phase 1B → phase_1_data/phase_1b_data.npz (50 MB)
# ✓ Extracted 1,000,000 samples for Phase 1C → phase_1_data/phase_1c_data.npz (200 MB)

# Verify files exist
ls -lh phase_1_data/*.npz
```

**If extraction fails:**
```
Common errors:
- "ModuleNotFoundError: SimplexNoise3D" → Copy SimplexNoise3D class from PHASE_1_DATA_EXTRACTION.md
- "MemoryError" → Reduce --num-chunks to 500, or run in PyPy
- "Seed derivation error" → Check f_seed() function matches Java implementation
```

---

#### Step 5: Launch Training (Parallel on 3 GPUs)

**Duration:** 6 hours (GPU) to 25 hours (CPU)  
**Hardware:** Each network requires ~4 GB VRAM or runs on CPU

**Action:**

```bash
# Terminal 1: Phase 1A (Macro-Shape)
# Hyperparameters: batch_size=128, lr=1e-3, early_stop @ MSE<0.001
python train_phase_1a.py \
    --data phase_1_data/phase_1a_data.npz \
    --epochs 50 \
    --batch-size 128 \
    --learning-rate 1e-3 \
    --gpu 0 \
    --use-float16 \
    --checkpoint-dir checkpoints/phase_1a \
    2>&1 | tee phase_1a.log

# Terminal 2: Phase 1B (Climate & Biome)
# Hyperparameters: batch_size=256, lr=1e-3, early_stop @ R²>0.99
python train_phase_1b.py \
    --data phase_1_data/phase_1b_data.npz \
    --epochs 20 \
    --batch-size 256 \
    --learning-rate 1e-3 \
    --gpu 1 \
    --use-float16 \
    --checkpoint-dir checkpoints/phase_1b \
    2>&1 | tee phase_1b.log

# Terminal 3: Phase 1C (Caves & Aquifers)
# Hyperparameters: batch_size=64, lr=5e-4, early_stop @ IoU>0.75
python train_phase_1c.py \
    --data phase_1_data/phase_1c_data.npz \
    --epochs 50 \
    --batch-size 64 \
    --learning-rate 5e-4 \
    --gpu 2 \
    --use-float16 \
    --use-weighted-loss \
    --checkpoint-dir checkpoints/phase_1c \
    2>&1 | tee phase_1c.log
```

**Monitoring** (separate terminal):
```bash
# Watch convergence in real-time
watch -n 10 'echo "=== Phase 1A ===" && tail -3 phase_1a.log && echo && echo "=== Phase 1B ===" && tail -3 phase_1b.log && echo && echo "=== Phase 1C ===" && tail -3 phase_1c.log'

# Or use Python monitor
python monitor_training.py --log-files phase_1*.log
```

**Expected Progress:**

| Network | Startup | Convergence | Stopped | Status |
|---------|---------|---|---|---|
| Phase 1B | Epoch 1 | Epoch 8-12 | Early stop @ R²>0.99 | 2 hours |
| Phase 1C | Epoch 1 | Epoch 20-30 | Early stop @ IoU>0.75 | 5 hours |
| Phase 1A | Epoch 1 | Epoch 15-25 | Early stop @ MSE<0.001 | 6 hours |

---

#### Step 6: Verify Convergence

**When each network converges, verify grok metrics:**

```bash
# Phase 1A convergence check
grep "MSE < 0.001" phase_1a.log && echo "✓ Phase 1A GROKKED" || echo "❌ Phase 1A NOT converged"

# Phase 1B convergence check
grep "Biome accuracy > 95%" phase_1b.log && echo "✓ Phase 1B GROKKED" || echo "❌ Phase 1B NOT converged"

# Phase 1C convergence check
grep "IoU > 0.75" phase_1c.log && echo "✓ Phase 1C GROKKED" || echo "❌ Phase 1C NOT converged"

# Verify checkpoints exist
ls -1 checkpoints/phase_1*/best.pt
# Expected:
# checkpoints/phase_1a/best.pt
# checkpoints/phase_1b/best.pt
# checkpoints/phase_1c/best.pt
```

**If convergence fails:**
```
Diagnostic checklist:
- [ ] Check phase_1*.log for NaN/Inf errors
- [ ] Verify data files are not corrupted: python verify_data.py
- [ ] Check GPU memory with nvidia-smi (should have headroom)
- [ ] Confirm float16 is not causing precision issues
- [ ] If Phase 1C IoU < 0.70: weighted loss not properly applied

Retry individual training with verbose output:
python train_phase_1c.py --debug --verbose
```

---

### POST-TRAINING PHASE (30 minutes)

#### Step 7: Generate Final Report

```bash
# Create summary document
cat > docs/PHASE_1_FINAL_REPORT.md << 'EOF'
# Phase 1 Training Results - [DATE]

## Convergence Summary

**Phase 1A (Macro-Shape):**
EOF

# Append results
grep -E "(MSE|MAE|R²)" phase_1a.log >> docs/PHASE_1_FINAL_REPORT.md
echo "" >> docs/PHASE_1_FINAL_REPORT.md

# Similar for 1B and 1C...

# Save logs
cp phase_1*.log docs/
```

#### Step 8: Freeze Networks for Phase 2

```python
# Load checkpoints and freeze
import torch
from networks_refined import Phase1ANetworkWithEncoding, Phase1BNetworkWithEncoding, Phase1CNetworkWithEncoding

model_1a = Phase1ANetworkWithEncoding()
model_1a.load_state_dict(torch.load("checkpoints/phase_1a/best.pt"))
model_1a.eval()
for param in model_1a.parameters(): param.requires_grad = False

# Repeat for 1B and 1C

# Save frozen models
torch.save(model_1a, "models/phase_1a_frozen.pt")
torch.save(model_1b, "models/phase_1b_frozen.pt")
torch.save(model_1c, "models/phase_1c_frozen.pt")

print("✓ All Phase 1 networks frozen and ready for Phase 2")
```

---

## Troubleshooting

### Issue: Parity Check Fails

**Symptom:** `|Java - Python| > 1e-6` on some samples

**Root Causes & Fixes:**

| Symptom | Cause | Fix |
|---------|-------|-----|
| All functions > 0.01 error | Wrong world seed in Python | Verify f_seed() hash matches Java RandomState |
| Only continents > 0.001 error | Octave scale mismatch | Check scales=[43.0] in OctavedNoiseFunction |
| Error increases with Y | Quad vs cell coordinate confusion | Confirm Y not divided by 4 (should be raw Y) |
| Intermittent NaN | Uninitialized permutation table | Verify SimplexNoise3D seed determinism |

**Debug Script:**
```python
from PHASE_1_DATA_EXTRACTION import MinecraftNoiseRouter, SimplexNoise3D

# Test single octave
gen = SimplexNoise3D(seed=12345)
val_2d = gen.sample_2d(0.0, 0.0)
val_3d = gen.sample_3d(0.0, 0.0, 0.0)
print(f"2D noise @ (0,0): {val_2d:.15f}")  # Should be reproducible
print(f"3D noise @ (0,0,0): {val_3d:.15f}")  # Should be reproducible

# If not reproducible → seed derivation bug
```

---

### Issue: Training Doesn't Converge

**Symptom:** Loss plateaus, grok metrics not improving

**Root Causes & Fixes:**

| Phase | Symptom | Cause | Fix |
|-------|---------|-------|-----|
| All | NaN loss after epoch 3 | Learning rate too high | Reduce to 1e-4, restart |
| All | Loss slowly decreasing | Learning rate too low | Increase to 5e-3, restart |
| 1A | MSE stuck @ 0.05 | Positional encoding disabled | Verify PositionalEncoder in use |
| 1B | Accuracy stuck @ 50% | Wrong climate output interpretation | Check biome classification function |
| 1C | IoU stuck @ 0.50 | Standard MSE (not weighted) | Verify CombinedLoss applied |

**Recovery:**
```bash
# Stop training
kill %1 %2 %3

# Investigate loss curve
python plot_loss_curves.py phase_1*.log

# Restart with adjusted hyperparameters
python train_phase_1c.py --learning-rate 1e-4 --restart
```

---

### Issue: Out of Memory

**Symptom:** `RuntimeError: CUDA out of memory` or `MemoryError` on CPU

**Fixes:**

```python
# Reduce batch size
batch_size = 64  # From 128 for Phase 1A
batch_size = 128  # From 256 for Phase 1B

# OR use float16 + gradient checkpointing
model = Phase1ANetworkWithEncoding(use_float16=True)

# OR run on CPU (slower but unlimited memory)
python train_phase_1a.py --device cpu
```

---

## Reference Documents

- **[PHASE_1_PARITY_VERIFICATION.md](PHASE_1_PARITY_VERIFICATION.md)** ← Start here for parity verification procedure
- **[PHASE_1_REFINEMENTS.md](PHASE_1_REFINEMENTS.md)** ← Implementation details for Float16, Positional Encoding, weighted loss, early stopping
- **[PHASE_1_STRATEGIC_MASTER_PLAN.md](PHASE_1_STRATEGIC_MASTER_PLAN.md)** ← Strategic overview and timeline
- **[PHASE_1_DATA_EXTRACTION.md](PHASE_1_DATA_EXTRACTION.md)** ← Noise generation code snippets
- **[PHASE_1_TENSOR_ARCHITECTURE.md](PHASE_1_TENSOR_ARCHITECTURE.md)** ← Exact tensor specifications

---

## Timeline Summary

| Milestone | Time | Status |
|-----------|------|--------|
| Parity Verification | 30 min | 🚨 BLOCKING |
| Refinement Integration | 60 min | ✅ Recommended |
| Data Extraction | 30 min | Sequential |
| **Parallel Training** | **6 hours** | GPU (3x parallel) |
| **Sequential Training** | **25 hours** | CPU or single GPU |
| **Total (GPU)** | **~2 hours** | With parity + refinements |

---

## Success Criteria

✅ **Phase 1A GROKKED:**
```
MSE < 0.001
MAE < 0.05
R² > 0.99
```

✅ **Phase 1B GROKKED:**
```
Biome accuracy > 95%
Climate MSE < 0.005
R² > 0.99
```

✅ **Phase 1C GROKKED:**
```
IoU > 0.75
Dice > 0.80
Cave volume accuracy > 85%
```

✅ **All Networks Ready for Phase 2:**
```
Checkpoints exist and load without error
Weights frozen (requires_grad = False)
Inference time < 10ms per batch
```

---

## Final Notes

**Parity verification is non-negotiable.** Skip it and you will waste days training a model that fails in-game.

**Early stopping saves 40% of compute time.** Set up EarlyStoppingCallback from the start.

**Float16 is essential for non-NVIDIA hardware.** If you don't have a CUDA GPU, float16 support is critical.

**Once Phase 1 networks converge, freeze them immediately.** Do not continue training after thresholds are met.

---

**Status:** Ready to launch. All prerequisites documented.

**Contact:** For questions, refer to detailed docs above or check logs if training fails.

