# distill_node.py
"""
GUI node for density NN distillation, with circular progress bar integration.
"""
from VoxelTree.core.distill_density_nn import distill_student

class DistillNode:
    def __init__(self, gui_context):
        self.gui_context = gui_context
        self.progress_bar = gui_context.get_circular_progress_bar()
        self.teacher = None
        self.student = None
        self.epochs = 120
        self.alpha = 0.5
        self.lr = 2e-3
        self.device = 'cuda'

    def configure(self, teacher, student, epochs=120, alpha=0.5, lr=2e-3, device='cuda'):
        self.teacher = teacher
        self.student = student
        self.epochs = epochs
        self.alpha = alpha
        self.lr = lr
        self.device = device

    def run(self):
        self.progress_bar.set_value(0.0)
        self.progress_bar.set_text('Starting distillation...')
        def progress_callback(epoch, total_epochs, metrics):
            self.progress_bar.set_value(epoch / total_epochs)
            self.progress_bar.set_text(f"Epoch {epoch}/{total_epochs} | val_mse={metrics['val_mse']:.5f}")
        result = distill_student(
            teacher_name=self.teacher,
            student_name=self.student,
            epochs=self.epochs,
            alpha=self.alpha,
            lr=self.lr,
            device=self.device,
            progress_callback=progress_callback
        )
        self.progress_bar.set_value(1.0)
        self.progress_bar.set_text('Distillation complete.')
        self.gui_context.notify_distillation_complete(result)
