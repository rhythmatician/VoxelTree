# Sparse-root distillation

Distills a smaller sparse-root student from a stronger teacher checkpoint while
still training on the real `sparse_octree_pairs.npz` supervision.

## Recommended command

```powershell
c:/Users/JeffHall/git/MC/.venv/Scripts/python.exe c:/Users/JeffHall/git/MC/VoxelTree/scripts/sparse_octree/distill.py \
  --data c:/Users/JeffHall/git/MC/noise_training_data/sparse_octree_pairs.npz \
  --teacher-checkpoint c:/Users/JeffHall/git/MC/tmp_real_sparse_octree_model.pt \
  --out c:/Users/JeffHall/git/MC/tmp_fast80_distilled_sparse_octree.pt \
  --student-variant fast \
  --student-hidden 80 \
  --epochs 20 \
  --batch-size 32 \
  --device cpu
```

## Main knobs

- `--hard-weight`: direct supervision weight
- `--split-distill-weight`: teacher guidance for split logits
- `--label-distill-weight`: teacher guidance for leaf-label logits
- `--temperature`: softens teacher distributions for label distillation

## Outputs

- distilled student checkpoint (`.pt`)
- JSON summary beside the checkpoint by default