"""
Smriti Memory Browser UI.

Launch a local web interface to browse and visualize your agent's memories.

Usage:
    from smriti_memcore.ui import launch
    launch(storage_path="~/.smriti/global")

    # Or from the terminal:
    python -m smriti_memcore.ui --storage ~/.smriti/global
    python -m smriti_memcore.ui --storage ~/.smriti/global --port 8765
"""

from smriti_memcore.ui.server import launch

__all__ = ["launch"]
