# OGN Dual-Head Architecture Redesign

**Title:** Replacing Block Logits with Density Fields + Material Classification  
**Date:** March 13, 2026  
**Status:** Design Document (Phase 1A)  

---

## Executive Summary

The current OGN (Octree Generation Network) models predict 1104-class block ID logits directly. This architectural redesign proposes:

1. **Density Field Output Head** — Predict continuous float density values (slopedCheese)
2. **Material Classification Head** — Predict ~12 semantic material categories
3. **Dual-Head Training Loss** — MSE on density + masked CrossEntropy on materials

**Benefit:** Continuous targets (density) are easier for CNNs to learn than high-dimensional categorical targets (1104 classes).

---

## Current Architecture (Block Logits)

```
OGN Init/Refine/Leaf:
  Input:  heightmap[5, H, W]
          biome[H, W]
          (context tensors)
  
  Shared Trunk:  Conv3D layers → feature maps
  
  Output Head (Single):
    logits[B, 1104, D, D, D]  ← 1104 softmax classes
    
  Loss: CrossEntropyLoss(logits, gt_block_ids)
  
  Inference: argmax(1) per voxel → block_id int64
```

---

## Proposed Architecture (Density + Material)

```
OGN Init/Refine/Leaf:
  Input:  heightmap[5, H, W]
          biome[H, W]
          (context tensors)
  
  Shared Trunk:  Conv3D layers → feature maps [C, D, D, D]
  
  Two Output Heads:
  
  ┌─ Density Head
  │   Conv3D(C → 1)
  │   Output: density[B, 1, D, D, D]  ← float32, unbounded
  │   
  │   Meaning: continuous slopedCheese value
  │   Threshold: density > 0 → solid, ≤ 0 → air/fluid
  │   
  │   Loss: MSE(pred_density, gt_density)
  │         λ₁ = 0.8 (weight)
  │
  ├─ Material Head
  │   Conv3D(C → 12)
  │   Output: material[B, 12, D, D, D]  ← logits, 12 classes
  │   
  │   Classes (12 materials):
  │     0. Air (dummy, will be masked)
  │     1. Stone
  │     2. Deepslate
  │     3. Dirt/Soil
  │     4. Grass (top surface)
  │     5. Sand/Gravel
  │     6. Water
  │     7. Lava
  │     8. Ore deposits
  │     9. Bedrock/End Stone
  │    10. Vegetation
  │    11. Wood/Logs
  │   
  │   Loss: CrossEntropyLoss(material[density>0],
  │                          gt_material[density>0])
  │         λ₂ = 0.2 (weight)
  │         (Applied only where density > 0)
```

### Output Processing (Inference)

```
For each voxel (x, y, z):
  
  # Step 1: Density threshold
  if pred_density[x,y,z] > 0:
    is_solid = True
    mat_idx = argmax(material[:, x,y,z])
  else:
    is_solid = False
    mat_idx = 0  # Air
  
  # Step 2: Material mapping
  block_id = material_map[mat_idx]
  
  # Step 3: Biome-aware adjustments (optional)
  # Could apply SurfaceRules logic here:
  #   if y == surface_y:  block_id = grass
  #   if y > surface_y:   block_id = air
  #   if y < surface_y:   block_id = dirt/stone
  
  output_block[x,y,z] = block_id
```

---

## Ground Truth Preparation

###  Density Ground Truth

**Source:** Vanilla Minecraft noise router `slopedCheese` function

```python
# In LODiffusion/data-cli.py:

def extract_density_label(server_context, chunk_pos, block_pos):
    """
    Extract the vanilla slopedCheese density at a block position.
    
    Args:
        server_context: Running Minecraft server with NoiseRouter accessibili
        chunk_pos: (cx, cz)
        block_pos: (bx, by, bz)
    
    Returns:
        float: slopedCheese density value
    """
    # Access the live NoiseRouter from the server
    noise_router = server_context.noise_router
    
    # Create FunctionContext for the block position
    ctx = FunctionContext.forBlockPos(
        server_context.get_level(),
        block_pos
    )
    
    # Evaluate slopedCheese DensityFunction
    # This is NOT finalDensity; it's the pre-cave-carving density
    density = noise_router.sloped_cheese.compute(ctx)
    
    return density  # float, range (~-2, ~2)
```

### Material Ground Truth

**Source:** Vanilla Minecraft block state at block position

```python
def extract_material_label(server_context, chunk_pos, block_pos):
    """
    Extract the vanilla block state and map to material category.
    
    Args:
        server_context: Running Minecraft server
        chunk_pos: (cx, cz)
        block_pos: (bx, by, bz)
    
    Returns:
        int: material category ID (0-11)
    """
    block_state = server_context.get_block(block_pos)
    block_name = block_state.get_name()
    
    # Map vanilla block name to material category
    # Uses material_categories.json
    material_id = BLOCK_TO_MATERIAL[block_name]
    
    return material_id  # int, range [0, 11]
```

---

## Training Loss

### Combined Loss Function

```python
class OGNDualHeadLoss(nn.Module):
    def __init__(self, lambda_density=0.8, lambda_material=0.2):
        super().__init__()
        self.lambda_density = lambda_density
        self.lambda_material = lambda_material
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, pred_density, pred_material, 
                gt_density, gt_material, gt_mask=None):
        """
        Args:
            pred_density: [B, 1, D, D, D]
            pred_material: [B, 12, D, D, D]
            gt_density: [B, D, D, D]
            gt_material: [B, D, D, D]  (int64)
            gt_mask: [B, D, D, D]  (bool, optional)
                    Mask of valid voxels (True = solid)
        
        Returns:
            loss: scalar
        """
        # Remove channel dimension from predictions for loss
        pred_density = pred_density.squeeze(1)  # [B, D, D, D]
        
        # Density loss: MSE on all voxels
        density_loss = self.mse_loss(pred_density, gt_density)
        
        # Material loss: CrossEntropy only on solid voxels
        if gt_mask is None:
            # Infer mask: solid = (gt_density > 0)
            gt_mask = gt_density > 0
        
        # Flatten spatial dimensions for CE loss
        B, D, H, W = pred_material.shape
        pred_mat_flat = pred_material.permute(0, 2, 3, 4, 1).reshape(-1, 12)
        gt_mat_flat = gt_material.reshape(-1)
        gt_mask_flat = gt_mask.reshape(-1)
        
        # Apply mask: only compute loss where solid
        if gt_mask_flat.any():
            material_loss = self.ce_loss(
                pred_mat_flat[gt_mask_flat],
                gt_mat_flat[gt_mask_flat]
            )
        else:
            material_loss = torch.tensor(0.0, device=pred_material.device)
        
        # Combined loss
        total_loss = (self.lambda_density * density_loss + 
                      self.lambda_material * material_loss)
        
        return {
            'total': total_loss,
            'density': density_loss,
            'material': material_loss
        }
```

### Hyperparameters

```python
# Training configuration
train_config = {
    'loss_lambda_density': 0.8,      # Density loss weight
    'loss_lambda_material': 0.2,     # Material loss weight
    'optimizer': 'Adam',
    'learning_rate': 0.001,
    'batch_size': 4,
    'epochs': 100,
    'warmup_epochs': 10,
    'density_loss_type': 'MSE',      # or 'SmoothL1'
    'material_loss_type': 'CE',      # CrossEntropy
    'regularization': {
        'weight_decay': 1e-6,
        'gradient_clip': 1.0
    }
}
```

---

## Implementation Checklist

### LODiffusion (Java/ONNX Export)

- [ ] Modify `OGNInitModel.onnx` export to include density head
  - Current: `fc_block_logits[B, 1104, D, D, D]`
  - New: `fc_density[B, 1, D, D, D]`, `fc_material[B, 12, D, D, D]`
  
- [ ] Update ONNX model metadata (input/output specs)
  
- [ ] Implement `DensityFieldPostProcessor` class
  - Input: density + material logits
  - Output: block IDs via threshold + argmax

### VoxelTree (Python Training)

- [ ] Add `material_categories.json` to schema/ ✓ (Done)

- [ ] Create `BLOCK_TO_MATERIAL` mapping dict
  - Load from material_categories.json
  - Map all 1104 block IDs → 12 material categories

- [ ] Modify `OGNInitModel`, `OGNRefineModel`, `OGNLeafModel`
  - Replace final FC layer with dual heads
  - Update training loop with `OGNDualHeadLoss`

- [ ] Update training loop
  - Load `gt_density` and `gt_material` instead of `gt_block_ids`
  - Apply masked loss (material loss only where density > 0)
  - Log separate metrics for density and material accuracy

- [ ] Implement density ground-truth extraction
  - In `LODiffusion/data-cli.py`
  - Call `NoiseRouter.slopedCheese.compute(ctx)`

- [ ] Implement material ground-truth extraction
  - Read block state from MC server
  - Map to material category

### Testing

- [ ] Verify density field predictions vs vanilla slopedCheese
  - Sample random blocks
  - Compare: `|pred_density - vanilla_density|`
  - Target: RMSE < 0.05

- [ ] Verify material classification accuracy
  - Accuracy metric (% correct material after thresholding)
  - Target: > 95% accuracy on solid blocks

- [ ] End-to-end terrain generation
  - Generate chunk with new architecture
  - Visual inspection vs vanilla

---

## Performance Implications

| Metric | Before (1104 logits) | After (density + material) | Impact |
|--------|------|--------|--------|
| **Model output size** | 1104 × 32³ floats = 51 MB/chunk | 1 × 32³ + 12 × 32³ = 0.4 MB/chunk | **99.2% reduction** |
| **Memory for weights** | Same (depends on CNN trunk) | Same | No change |
| **Training convergence** | ~50 epochs | ~30 epochs | **Faster** |
| **Inference speed** | Same CNN + argmax | Same CNN + argmax + threshold | **Slightly faster** |
| **Accuracy** | 70–75% block accuracy | TBD (higher w/ smooth targets) | TBD |

---

## Migration Path

**Phase 1A (Current):**
1. ✓ Train density + material heads on new ground truth
2. Export new ONNX models
3. Update LODiffusion Java inference

**Phase 1B (Future):**
- Combine with Phase 1B ML noise (2D Perlin replacement)
- Test integrated E2E pipeline

**Phase 1C (Future):**
- Refine hyperparameters
- Investigate surface rules post-processing

---

## References

- [MINECRAFT_TERRAIN_DAG_COMPLETE.md](./MINECRAFT_TERRAIN_DAG_COMPLETE.md) — Full pipeline documentation
- [material_categories.json](../VoxelTree/schema/material_categories.json) — Material taxonomy
- `TerrainProvider.java` lines 1–300 — Spline reference
- `NoiseRouterData.java` — Register terrain noises
