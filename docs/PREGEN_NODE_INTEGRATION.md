"""
PregenNode Progress Tracking Integration Guide
==============================================

OVERVIEW
--------
The PregenNode GUI node now extracts the percentage from chunky logs and wires 
it directly to the progress bar. This eliminates the need for manual parsing or 
separate progress tracking mechanisms.

KEY COMPONENTS
--------------

1. PregenNode (gui/nodes/pregen_node.py)
   - GUI node class that manages the pregeneration workflow
   - Receives progress updates from cmd_pregen_with_callback
   - Updates the circular progress bar with percentage (0.0-1.0) and status text
   - Supports configuration of RCON parameters (host, port, password, radius)

2. cmd_pregen_with_callback (VoxelTree/preprocessing/cli.py)
   - Enhanced version of cmd_pregen that accepts a progress_callback parameter
   - Extracts percentage from chunky response using regex: r'\((\d+\.?\d*?)%\)'
   - Calls callback(percentage: 0-100, status_text: str) every 5 seconds
   - Handles freeze, configuration, and polling phases with appropriate progress values

USAGE EXAMPLE
-------------

from gui.nodes import PregenNode

# Create node with GUI context
node = PregenNode(gui_context)

# Configure parameters
node.configure(
    host="localhost",
    port=25575,
    password="your_rcon_password",
    radius=2048
)

# Run with automatic progress bar updates
node.run()

PROGRESS BAR UPDATES
--------------------

The progress bar receives updates at these stages:

1. Freezing world state:        0%   → 5%
2. Configuring Chunky:          5%   → 10%
3. Chunky progress polling:     10%  → 100% (extracted from chunky response)
4. Complete:                    100%

Each chunky response like:
  "Task running for minecraft:overworld. Processed: 2500 chunks (3.12%), ETA: 0:45:30"
  
Is parsed to extract "3.12%" and passed to the progress bar with the full status text.

REGEX PATTERN
-------------

The percentage is extracted using:
  r'\((\d+\.?\d*?)%\)'

This matches:
  (0%)      → 0.0
  (3.12%)   → 3.12
  (100%)    → 100.0

INTEGRATION WITH GUI CONTEXT
-----------------------------

The node expects the gui_context to provide:
  - get_circular_progress_bar()  → returns progress bar interface
  - notify_pregen_complete()     → called when pregen finishes
  - notify_error(message)        → called if an error occurs

Progress bar interface should support:
  - set_value(float 0.0-1.0)     → updates progress percentage
  - set_text(str)                → updates status text

TESTING
-------

To test without a running server (dry-run mode):

  node.configure(
      host="localhost",
      port=25575,
      password="test",
      radius=2048,
      dry_run=True
  )
  node.run()

This will show the command sequence without sending RCON commands.
"""
