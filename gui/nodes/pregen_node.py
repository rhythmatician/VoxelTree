"""GUI node for Chunky pregeneration with real-time progress tracking."""

from VoxelTree.preprocessing.cli import cmd_pregen_with_callback, PipelineConfig


class PregenNode:
    """GUI node that monitors Chunky pregeneration progress.

    Extracts percentage from chunky progress RCON responses and updates
    the GUI progress bar in real-time.

    Prerequisites:
        - Fabric server running with RCON enabled
        - Chunky mod installed
    """

    # For GUI node wiring / graph display
    prerequisites = []
    next_steps = ["voxy-import"]

    def __init__(self, gui_context):
        self.gui_context = gui_context
        self.progress_bar = gui_context.get_circular_progress_bar()

        # Configuration (can be set via configure())
        self.host = "localhost"
        self.port = 25575
        self.password = ""
        self.radius = 2048
        self.verbose = False
        self.dry_run = False

    def configure(
        self,
        host: str = "localhost",
        port: int = 25575,
        password: str = "",
        radius: int = 2048,
        verbose: bool = False,
        dry_run: bool = False,
    ):
        """Configure pregeneration parameters."""
        self.host = host
        self.port = port
        self.password = password
        self.radius = radius
        self.verbose = verbose
        self.dry_run = dry_run

    def run(self):
        """Execute pregeneration with progress tracking."""
        if not self.password:
            self.progress_bar.set_text("ERROR: Password required")
            self.gui_context.notify_error("Password required for RCON connection")
            return

        self.progress_bar.set_value(0.0)
        self.progress_bar.set_text("Starting pregeneration...")

        def progress_callback(percentage, status_text):
            """Called from cmd_pregen_with_callback with progress updates."""
            self.progress_bar.set_value(percentage / 100.0)
            self.progress_bar.set_text(status_text)

        # Create a PipelineConfig object
        cfg = PipelineConfig(
            host=self.host,
            port=self.port,
            password=self.password,
            radius=self.radius,
            verbose=self.verbose,
            dry_run=self.dry_run,
        )

        try:
            cmd_pregen_with_callback(cfg, progress_callback)
            self.progress_bar.set_value(1.0)
            self.progress_bar.set_text("Pregeneration complete!")
            self.gui_context.notify_pregen_complete()
        except Exception as e:
            self.progress_bar.set_text(f"Error: {str(e)}")
            self.gui_context.notify_error(f"Pregeneration failed: {str(e)}")
