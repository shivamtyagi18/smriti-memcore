"""
SMRITI v2: Neuro-Inspired EXperience-Unified System
A novel memory architecture for AI agents inspired by human cognition.
"""

from smriti_memcore.core import SMRITI
from smriti_memcore.models import Memory, SalienceScore, MemorySource, Modality, SmritiConfig
from smriti_memcore.metrics import SmritiMetrics

try:
    from importlib.metadata import version
    __version__ = version("smriti-memcore")
except Exception:
    __version__ = "unknown"
__all__ = ["SMRITI", "SmritiConfig", "SmritiMetrics", "Memory", "SalienceScore", "MemorySource", "Modality"]

