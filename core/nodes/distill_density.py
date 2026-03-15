"""
Distill-density pipeline node for density NN teacher → student transfer.
Integrates with the GUI progress bar and pipeline DAG system.
"""
from pathlib import Path
from typing import Optional, Dict, Any

from VoxelTree.core.distill_density_nn import distill_student


class DistillDensityNode:
    """Pipeline node for distillation."""
    
    node_id = 'distill_density'
    node_name = 'Distill Density NN'
    node_description = 'Distill a fast student model from a slower teacher.'
    
    def __init__(self, context=None):
        self.context = context
        self.progress_callback = None
    
    def set_progress_callback(self, callback):
        """Set callback for progress updates (for GUI integration)."""
        self.progress_callback = callback
    
    def run(self,
            teacher_name: str = 'unet',
            student_name: str = 'sep',
            epochs: int = 120,
            alpha: float = 0.5,
            lr: float = 2e-3,
            device: str = 'cuda',
            output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Run distillation.
        
        Args:
            teacher_name: Name of teacher model (mlp, sep, axial, unet)
            student_name: Name of student model (mlp, sep, axial, unet)
            epochs: Training epochs
            alpha: Distillation alpha (ground truth vs teacher weight)
            lr: Learning rate
            device: Device to train on
            output_dir: Where to save results (optional)
        
        Returns:
            Dictionary with distillation results and checkpoint path
        """
        # Use GUI progress callback if available
        def progress_callback(epoch, total_epochs, metrics):
            if self.progress_callback:
                progress = epoch / total_epochs
                message = f"Epoch {epoch}/{total_epochs} | val_mse={metrics['val_mse']:.5f}"
                self.progress_callback(progress, message)
        
        result = distill_student(
            teacher_name=teacher_name,
            student_name=student_name,
            epochs=epochs,
            alpha=alpha,
            lr=lr,
            device=device,
            progress_callback=progress_callback
        )
        
        # Notify completion if callback is set
        if self.progress_callback:
            self.progress_callback(1.0, "Distillation complete.")
        
        return result
    
    @staticmethod
    def get_parameters() -> Dict[str, Any]:
        """Return node configuration parameters."""
        return {
            'teacher_name': {
                'type': 'choice',
                'label': 'Teacher Model',
                'options': ['mlp', 'sep', 'axial', 'unet'],
                'default': 'unet',
                'help': 'Slow, accurate teacher model'
            },
            'student_name': {
                'type': 'choice',
                'label': 'Student Model',
                'options': ['mlp', 'sep', 'axial', 'unet'],
                'default': 'sep',
                'help': 'Fast student model to train'
            },
            'epochs': {
                'type': 'int',
                'label': 'Epochs',
                'default': 120,
                'min': 1,
                'max': 1000,
                'help': 'Training epochs'
            },
            'alpha': {
                'type': 'float',
                'label': 'Distillation Alpha',
                'default': 0.5,
                'min': 0.0,
                'max': 1.0,
                'help': 'Weight on ground truth (vs teacher): 0=teacher-only, 1=gt-only'
            },
            'lr': {
                'type': 'float',
                'label': 'Learning Rate',
                'default': 2e-3,
                'min': 1e-5,
                'max': 1e-1,
                'help': 'Adam learning rate'
            },
            'device': {
                'type': 'choice',
                'label': 'Device',
                'options': ['cuda', 'cpu'],
                'default': 'cuda',
                'help': 'Training device'
            }
        }

