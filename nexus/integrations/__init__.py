"""
NEXUS Integration package.
Contains adapters and wrappers to plug NEXUS into standard agent frameworks.
"""

# Try to expose modules if dependencies are met, but don't crash on import if not.
__all__ = []

try:
    from nexus.integrations.mcp_server import mcp_server as nexus_mcp_server  # noqa: F401
    __all__.append("nexus_mcp_server")
except ImportError:
    pass  # mcp not installed — silently skip
